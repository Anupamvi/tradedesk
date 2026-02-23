#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build exact Schwab-live trade ideas from X range logic (green/yellow tickers)."
    )
    p.add_argument("--date", required=True, help="As-of date (YYYY-MM-DD).")
    p.add_argument("--base-dir", default=r"C:\uw_root")
    p.add_argument("--handle", required=True)
    p.add_argument("--x-out-dir", default="")
    p.add_argument("--rulebook-config", default=r"C:\uw_root\uwos\rulebook_config.yaml")
    p.add_argument("--out-dir", default="")
    p.add_argument("--max-per-ticker", type=int, default=2)
    return p.parse_args()


def fnum(x: Any) -> float:
    try:
        if x is None or x == "":
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def parse_date(text: str) -> dt.date:
    return dt.datetime.strptime(str(text).strip()[:10], "%Y-%m-%d").date()


def status_from_logic_row(row: pd.Series) -> str:
    two_sided = bool(pd.notna(row.get("call_ceiling_min")) and pd.notna(row.get("put_floor_min")))
    call_only = bool(pd.notna(row.get("call_ceiling_min")) and pd.isna(row.get("put_floor_min")))
    events = int(fnum(row.get("events_count", 0)) if not math.isnan(fnum(row.get("events_count", 0))) else 0)
    style = str(row.get("inferred_style", ""))
    if two_sided and events >= 3:
        return "GREEN"
    if call_only and events >= 2:
        return "YELLOW"
    if "Directional/options commentary" in style or events <= 1:
        return "RED"
    return "ORANGE"


def detect_latest_screener(base_dir: Path, asof: dt.date) -> Optional[Path]:
    candidates = []
    pat = re.compile(r"stock-screener-(\d{4}-\d{2}-\d{2})\.csv$", re.I)
    for p in base_dir.rglob("stock-screener-*.csv"):
        m = pat.search(p.name)
        if not m:
            continue
        try:
            d = parse_date(m.group(1))
        except Exception:
            continue
        if d <= asof:
            candidates.append((d, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def width_from_tiers(spot: float, cfg: Dict[str, Any]) -> float:
    tiers = cfg.get("gates", {}).get("width_tiers", [])
    if not isinstance(tiers, list):
        return 5.0
    for t in tiers:
        lo = fnum(t.get("min_price"))
        hi = fnum(t.get("max_price"))
        default_w = fnum(t.get("default_width"))
        if math.isnan(lo) or math.isnan(hi) or math.isnan(default_w):
            continue
        if lo <= spot < hi:
            return float(default_w)
    return 5.0


def contract_rows(chain_payload: Dict[str, Any], right: str) -> pd.DataFrame:
    map_name = "callExpDateMap" if right == "C" else "putExpDateMap"
    exp_map = chain_payload.get(map_name, {}) or {}
    rows: List[Dict[str, Any]] = []
    for exp_key, strike_map in exp_map.items():
        expiry = str(exp_key).split(":")[0]
        try:
            expiry_date = parse_date(expiry)
        except Exception:
            continue
        for strike_key, contracts in (strike_map or {}).items():
            strike = fnum(strike_key)
            if math.isnan(strike):
                continue
            for c in contracts or []:
                rows.append(
                    {
                        "expiry": expiry_date,
                        "strike": float(strike),
                        "symbol": str(c.get("symbol", "")),
                        "bid": fnum(c.get("bid")),
                        "ask": fnum(c.get("ask")),
                        "delta": fnum(c.get("delta")),
                        "oi": fnum(c.get("openInterest")),
                        "volume": fnum(c.get("totalVolume")),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["expiry", "strike", "symbol", "bid", "ask", "delta", "oi", "volume"])
    return pd.DataFrame(rows)


def choose_expiry(df: pd.DataFrame, asof: dt.date, target_dte: float) -> Optional[dt.date]:
    if df.empty:
        return None
    exp = sorted(df["expiry"].dropna().unique().tolist())
    if not exp:
        return None
    tgt = max(7.0, float(target_dte))
    best = min(exp, key=lambda d: abs((d - asof).days - tgt))
    return best


def nearest_value(candidates: List[float], target: float) -> Optional[float]:
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - target))


def pick_credit_spread_legs(
    chain_df: pd.DataFrame,
    asof: dt.date,
    right: str,
    target_level: float,
    target_dte: float,
    width: float,
    entry_threshold: float,
    spot: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str], Optional[float], Optional[bool]]:
    if chain_df.empty:
        return None, None, "no_chain_rows", None, None

    exps = sorted(chain_df["expiry"].dropna().unique().tolist(), key=lambda d: abs((d - asof).days - max(7.0, target_dte)))
    if not exps:
        return None, None, "no_expiry", None, None

    best_pass: Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float]] = None
    best_any: Optional[Tuple[Dict[str, Any], Dict[str, Any], float, float]] = None

    for exp in exps[:8]:
        edf = chain_df[chain_df["expiry"] == exp].copy()
        if edf.empty:
            continue
        strikes = sorted(edf["strike"].dropna().astype(float).unique().tolist())
        if not strikes:
            continue

        # Keep strikes on the correct side of spot when possible.
        if right == "C":
            side_ok = [s for s in strikes if s > spot]
            strike_pool = side_ok if side_ok else strikes
            short_candidates = sorted(strike_pool, key=lambda s: abs(s - target_level))
        else:
            side_ok = [s for s in strikes if s < spot]
            strike_pool = side_ok if side_ok else strikes
            short_candidates = sorted(strike_pool, key=lambda s: abs(s - target_level))

        for short_strike in short_candidates:
            if right == "C":
                long_target = short_strike + width
                away = [s for s in strikes if s > short_strike]
                long_strike = nearest_value([s for s in away if s >= long_target], long_target) or nearest_value(away, long_target)
            else:
                long_target = short_strike - width
                away = [s for s in strikes if s < short_strike]
                long_strike = nearest_value([s for s in away if s <= long_target], long_target) or nearest_value(away, long_target)
            if long_strike is None:
                continue

            srow = edf[edf["strike"] == short_strike].head(1)
            lrow = edf[edf["strike"] == long_strike].head(1)
            if srow.empty or lrow.empty:
                continue
            srec = srow.iloc[0].to_dict()
            lrec = lrow.iloc[0].to_dict()
            sbid = fnum(srec.get("bid"))
            lask = fnum(lrec.get("ask"))
            if math.isnan(sbid) or math.isnan(lask):
                credit = math.nan
            else:
                credit = sbid - lask
            dist = abs(short_strike - target_level)

            if best_any is None or dist < best_any[3]:
                best_any = (srec, lrec, credit, dist)
            if not math.isnan(credit) and credit >= entry_threshold:
                if best_pass is None or dist < best_pass[3]:
                    best_pass = (srec, lrec, credit, dist)

    if best_pass is not None:
        srec, lrec, credit, _ = best_pass
        shifted = abs(fnum(srec.get("strike")) - target_level) > 0.01
        return srec, lrec, None, credit, shifted
    if best_any is not None:
        srec, lrec, credit, _ = best_any
        shifted = abs(fnum(srec.get("strike")) - target_level) > 0.01
        return srec, lrec, "no_credit_meet", credit, shifted
    return None, None, "missing_leg_rows", None, None


def format_money(x: Any) -> str:
    v = fnum(x)
    return "N/A" if math.isnan(v) else f"${v:,.2f}"


def format_px(x: Any) -> str:
    v = fnum(x)
    return "N/A" if math.isnan(v) else f"{v:.2f}"


def action_for_strategy(strategy: str, optimal: str) -> str:
    if str(optimal).lower().startswith("watch"):
        return "🟨 WATCH"
    if strategy == "Bull Put Credit":
        return "🛡️🟩 SELL"
    if strategy == "Bear Call Credit":
        return "🛡️🟥 SELL"
    return "WATCH"


def run() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    args = parse_args()
    asof = parse_date(args.date)
    base_dir = Path(args.base_dir).resolve()
    x_out_dir = (
        Path(args.x_out_dir).resolve()
        if args.x_out_dir
        else (base_dir / "out" / "x_strategy_analysis" / asof.isoformat() / args.handle).resolve()
    )
    out_dir = Path(args.out_dir).resolve() if args.out_dir else x_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    logic_csv = x_out_dir / "x_range_ticker_logic.csv"
    if not logic_csv.exists():
        raise FileNotFoundError(f"Missing range logic file: {logic_csv}")
    logic = pd.read_csv(logic_csv)
    if logic.empty:
        raise RuntimeError("x_range_ticker_logic.csv is empty.")

    rulebook = yaml.safe_load(Path(args.rulebook_config).read_text(encoding="utf-8")) or {}
    min_credit_pct_width = fnum(rulebook.get("gates", {}).get("min_credit_pct_width", 0.25))
    if math.isnan(min_credit_pct_width):
        min_credit_pct_width = 0.25

    logic["status"] = logic.apply(status_from_logic_row, axis=1)
    target_logic = logic[logic["status"].isin(["GREEN", "YELLOW"])].copy()
    if target_logic.empty:
        raise RuntimeError("No GREEN/YELLOW tickers found in x_range_ticker_logic.csv.")

    # Flow info from latest screener <= asof
    screener_file = detect_latest_screener(base_dir, asof)
    flow_map: Dict[str, Dict[str, Any]] = {}
    if screener_file and screener_file.exists():
        sdf = pd.read_csv(screener_file, low_memory=False)
        sdf["ticker"] = sdf["ticker"].astype(str).str.upper().str.strip()
        keep_cols = [
            "ticker",
            "bullish_premium",
            "bearish_premium",
            "put_call_ratio",
            "call_premium",
            "put_premium",
            "next_earnings_date",
            "close",
            "iv_rank",
        ]
        for c in keep_cols:
            if c not in sdf.columns:
                sdf[c] = math.nan
        flow_map = sdf[keep_cols].drop_duplicates("ticker").set_index("ticker").to_dict("index")

    # Live Schwab pull
    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    service = SchwabLiveDataService(config=config, interactive_login=False)
    service.connect()

    candidates: List[Dict[str, Any]] = []
    chain_cache: Dict[str, Dict[str, Any]] = {}
    for _, lr in target_logic.iterrows():
        ticker = str(lr.get("ticker", "")).strip().upper()
        if not ticker or ticker == "UNKNOWN":
            continue

        call_dte = fnum(lr.get("call_dte_min"))
        put_dte = fnum(lr.get("put_dte_min"))
        max_dte = max(
            60.0,
            call_dte if not math.isnan(call_dte) else 35.0,
            put_dte if not math.isnan(put_dte) else 45.0,
        )
        to_date = asof + dt.timedelta(days=int(min(720, max_dte + 45)))
        from_date = asof + dt.timedelta(days=7)
        chain = service.get_option_chain(
            symbol=ticker,
            strike_count=None,
            include_underlying_quote=True,
            from_date=from_date,
            to_date=to_date,
        )
        chain_cache[ticker] = chain
        if str(chain.get("status", "")).upper() != "SUCCESS":
            continue

        spot = fnum(chain.get("underlyingPrice"))
        if math.isnan(spot):
            und = chain.get("underlying", {}) or {}
            spot = fnum(und.get("last") if not math.isnan(fnum(und.get("last"))) else und.get("mark"))
        if math.isnan(spot):
            continue

        width = width_from_tiers(spot, rulebook)
        entry_threshold = max(0.05, round(width * float(min_credit_pct_width), 2))

        status = str(lr.get("status", "YELLOW"))
        base_conviction = 78.0 if status == "GREEN" else 65.0
        events_count = fnum(lr.get("events_count"))
        if not math.isnan(events_count):
            base_conviction += min(7.0, max(0.0, events_count - 1.0))
        confidence_tier = "CORE" if status == "GREEN" else "AGG"

        cdf = contract_rows(chain, "C")
        pdf = contract_rows(chain, "P")

        # Call-ceiling side -> Bear Call Credit
        call_level = fnum(lr.get("call_ceiling_min"))
        if not math.isnan(call_level):
            tdte = call_dte if not math.isnan(call_dte) else 35.0
            s_leg, l_leg, err, pre_credit, shifted = pick_credit_spread_legs(
                chain_df=cdf,
                asof=asof,
                right="C",
                target_level=call_level,
                target_dte=tdte,
                width=width,
                entry_threshold=entry_threshold,
                spot=spot,
            )
            if s_leg and l_leg:
                candidates.append(
                    {
                        "ticker": ticker,
                        "strategy": "Bear Call Credit",
                        "expiry": str(s_leg["expiry"]),
                        "short_leg": str(s_leg["symbol"]),
                        "long_leg": str(l_leg["symbol"]),
                        "net_type": "credit",
                        "entry_gate": f">= {entry_threshold:.2f} cr",
                        "width": abs(fnum(s_leg["strike"]) - fnum(l_leg["strike"])),
                        "conviction": round(base_conviction, 1),
                        "track": "SHIELD",
                        "confidence_tier": confidence_tier,
                        "optimal_stage1": "Yes-Good" if status == "GREEN" else "Watch Only",
                        "invalidation": f"Invalidate if close > {fnum(s_leg['strike']):.2f}.",
                        "x_status": status,
                        "x_style": str(lr.get("inferred_style", "")),
                        "x_call_level": call_level,
                        "x_put_level": fnum(lr.get("put_floor_min")),
                        "x_logic_notes": str(lr.get("logic_notes", "")),
                        "pre_live_credit": pre_credit,
                        "anchor_shifted": bool(shifted),
                        "build_reason": "ok" if err is None else err,
                    }
                )

        # Put-floor side -> Bull Put Credit (only if floor exists)
        put_level = fnum(lr.get("put_floor_min"))
        if not math.isnan(put_level):
            tdte = put_dte if not math.isnan(put_dte) else 45.0
            s_leg, l_leg, err, pre_credit, shifted = pick_credit_spread_legs(
                chain_df=pdf,
                asof=asof,
                right="P",
                target_level=put_level,
                target_dte=tdte,
                width=width,
                entry_threshold=entry_threshold,
                spot=spot,
            )
            if s_leg and l_leg:
                candidates.append(
                    {
                        "ticker": ticker,
                        "strategy": "Bull Put Credit",
                        "expiry": str(s_leg["expiry"]),
                        "short_leg": str(s_leg["symbol"]),
                        "long_leg": str(l_leg["symbol"]),
                        "net_type": "credit",
                        "entry_gate": f">= {entry_threshold:.2f} cr",
                        "width": abs(fnum(s_leg["strike"]) - fnum(l_leg["strike"])),
                        "conviction": round(base_conviction + 1.5, 1),
                        "track": "SHIELD",
                        "confidence_tier": confidence_tier,
                        "optimal_stage1": "Yes-Good" if status == "GREEN" else "Watch Only",
                        "invalidation": f"Invalidate if close < {fnum(s_leg['strike']):.2f}.",
                        "x_status": status,
                        "x_style": str(lr.get("inferred_style", "")),
                        "x_call_level": fnum(lr.get("call_ceiling_min")),
                        "x_put_level": put_level,
                        "x_logic_notes": str(lr.get("logic_notes", "")),
                        "pre_live_credit": pre_credit,
                        "anchor_shifted": bool(shifted),
                        "build_reason": "ok" if err is None else err,
                    }
                )

    if not candidates:
        raise RuntimeError("No live candidates could be built from GREEN/YELLOW tickers.")

    cdf = pd.DataFrame(candidates)
    cdf = cdf.sort_values(["ticker", "conviction"], ascending=[True, False]).groupby("ticker").head(int(args.max_per_ticker))
    cdf = cdf.sort_values("conviction", ascending=False).reset_index(drop=True)

    shortlist_csv = out_dir / f"x_range_shortlist_{asof.isoformat()}_{args.handle}.csv"
    cdf.to_csv(shortlist_csv, index=False)

    live_csv = out_dir / f"x_range_live_trade_table_{asof.isoformat()}_{args.handle}.csv"
    live_final_csv = out_dir / f"x_range_live_trade_table_{asof.isoformat()}_{args.handle}_final.csv"
    snapshot_json = out_dir / f"x_range_schwab_snapshot_{asof.isoformat()}_{args.handle}.json"

    cmd = [
        sys.executable,
        "-m",
        "uwos.pricer",
        "--shortlist-csv",
        str(shortlist_csv),
        "--out-dir",
        str(out_dir),
        "--out-csv",
        str(live_csv),
        "--out-final-csv",
        str(live_final_csv),
        "--snapshot-out-json",
        str(snapshot_json),
        "--min-conviction",
        "0",
    ]
    subprocess.run(cmd, check=True)

    live = pd.read_csv(live_csv, low_memory=False)
    if live.empty:
        raise RuntimeError("Live pricer output is empty.")

    # Build preferred table
    rows = []
    for _, r in live.sort_values(["is_final_live_valid", "conviction"], ascending=[False, False]).iterrows():
        ticker = str(r.get("ticker", "")).upper()
        strategy = str(r.get("strategy", ""))
        short_k = fnum(r.get("short_strike_live"))
        long_k = fnum(r.get("long_strike_live"))
        width = abs(short_k - long_k) if not (math.isnan(short_k) or math.isnan(long_k)) else fnum(r.get("width"))
        live_net = fnum(r.get("live_net_bid_ask"))
        entry_gate = str(r.get("entry_gate", ""))
        live_status = str(r.get("live_status", ""))
        is_valid = bool(r.get("is_final_live_valid", False))
        conviction = fnum(r.get("conviction"))
        if math.isnan(conviction):
            conviction = 0.0
        confidence_tier = str(r.get("confidence_tier", "AGG"))
        optimal = "Yes-Prime" if (is_valid and conviction >= 75) else ("Yes-Good" if is_valid else "Watch Only")

        expiry = str(r.get("expiry", ""))[:10]
        dte = ""
        try:
            dte = str((parse_date(expiry) - asof).days)
        except Exception:
            dte = "N/A"

        if strategy == "Bull Put Credit":
            strike_setup = f"Sell {format_px(short_k)}P / Buy {format_px(long_k)}P ({format_px(width)}w)"
            be = format_px(short_k - live_net) if not (math.isnan(short_k) or math.isnan(live_net)) else "N/A"
        elif strategy == "Bear Call Credit":
            strike_setup = f"Sell {format_px(short_k)}C / Buy {format_px(long_k)}C ({format_px(width)}w)"
            be = format_px(short_k + live_net) if not (math.isnan(short_k) or math.isnan(live_net)) else "N/A"
        else:
            strike_setup = "N/A"
            be = "N/A"

        action = action_for_strategy(strategy, optimal)
        flow = flow_map.get(ticker, {})
        bull_p = fnum(flow.get("bullish_premium"))
        bear_p = fnum(flow.get("bearish_premium"))
        pcr = fnum(flow.get("put_call_ratio"))
        flow_text = (
            f"Flow bull {bull_p/1e6:.1f}M vs bear {bear_p/1e6:.1f}M; PCR {pcr:.2f}"
            if not (math.isnan(bull_p) or math.isnan(bear_p) or math.isnan(pcr))
            else "Flow N/A"
        )

        if not math.isnan(live_net):
            net_text = f"Credit {live_net:.2f} (Target {entry_gate})"
        else:
            net_text = f"N/A (Target {entry_gate})"

        pre_credit = fnum(r.get("pre_live_credit"))
        anchor_shifted = bool(r.get("anchor_shifted", False))
        build_reason = str(r.get("build_reason", ""))
        gate_thr = fnum(r.get("entry_gate_threshold"))
        gate_gap = None
        if not math.isnan(gate_thr) and not math.isnan(live_net):
            gate_gap = live_net - gate_thr

        notes = (
            f"X-{str(r.get('x_status',''))} {str(r.get('x_style',''))}; "
            f"live_status={live_status}; {flow_text}; "
            f"X levels floor={format_px(r.get('x_put_level'))}, ceiling={format_px(r.get('x_call_level'))}; "
            f"build={build_reason}; pre_live_credit={format_px(pre_credit)}; "
            f"anchor_shifted={'yes' if anchor_shifted else 'no'}; "
            f"gate_gap={format_px(gate_gap) if gate_gap is not None else 'N/A'}"
        )
        source_parts = ["XRange", "Schwab"]
        if screener_file is not None:
            source_parts.append(screener_file.name)
        source = " + ".join(source_parts)

        rows.append(
            {
                "#": len(rows) + 1,
                "Ticker": ticker,
                "Action": action,
                "Strategy Type": strategy,
                "Strike Setup": strike_setup,
                "Expiry": expiry,
                "DTE": dte,
                "Net Credit/Debit": net_text,
                "Max Profit": format_money(r.get("live_max_profit")),
                "Max Loss": format_money(r.get("live_max_loss")),
                "Breakeven": be,
                "Conviction %": f"{int(round(conviction))}%",
                "Confidence Tier": confidence_tier,
                "Optimal": optimal,
                "Notes": notes,
                "Source": source,
            }
        )

    out_df = pd.DataFrame(rows)
    # Approved first then watch
    out_df["_rank"] = out_df["Optimal"].map({"Yes-Prime": 0, "Yes-Good": 1, "Watch Only": 2}).fillna(3)
    out_df = out_df.sort_values(["_rank", "Conviction %"], ascending=[True, False]).drop(columns=["_rank"]).reset_index(drop=True)
    out_df["#"] = range(1, len(out_df) + 1)

    output_md = out_dir / f"x_range_anu_expert_trade_table_{asof.isoformat()}_{args.handle}.md"
    lines = [
        f"As-of date used: {asof.isoformat()}",
        "Files used: "
        + ", ".join(
            [
                logic_csv.name,
                shortlist_csv.name,
                live_csv.name,
                live_final_csv.name,
                snapshot_json.name,
                (screener_file.name if screener_file is not None else "stock-screener:missing"),
            ]
        ),
        "",
        "## Anu Expert Trade Table",
        "",
        out_df.to_markdown(index=False),
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8-sig")

    print(f"Wrote: {output_md}")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
