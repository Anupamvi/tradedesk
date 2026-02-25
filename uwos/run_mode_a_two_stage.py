import argparse
import datetime as dt
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from uwos.eod_trade_scan_mode_a import (
    build_best_candidates,
    build_quotes,
    ensure_cols,
    fnum,
    md_tables,
)
from uwos.report import load_open_positions


REQ_CSV_PREFIXES = [
    "chain-oi-changes-",
    "dp-eod-report-",
    "hot-chains-",
    "stock-screener-",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_git_commit() -> str:
    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        out = (cp.stdout or "").strip()
        return out or "unknown"
    except Exception:
        return "unknown"


def unzip_inputs_if_needed(base_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("*.csv"))
    if existing:
        return
    zips = sorted(base_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No input CSV/ZIP files found in {base_dir}")
    for zp in zips:
        with zipfile.ZipFile(zp, "r") as zf:
            names = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])
            if not names:
                continue
            name = names[0]
            target = out_dir / Path(name).name
            with zf.open(name, "r") as src:
                target.write_bytes(src.read())


def detect_asof_from_names(paths):
    pat = re.compile(r"(\d{4}-\d{2}-\d{2})")
    vals = []
    for p in paths:
        m = pat.search(p.name)
        if m:
            vals.append(m.group(1))
    if not vals:
        raise ValueError("Could not detect as-of date from filenames.")
    return sorted(vals)[-1]


def pick_csvs(base_dir: Path):
    unz = base_dir / "_unzipped_mode_a"
    unzip_inputs_if_needed(base_dir, unz)
    csvs = sorted(unz.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {unz}")

    out = {}
    for pref in REQ_CSV_PREFIXES:
        matches = [p for p in csvs if p.name.startswith(pref)]
        if not matches:
            raise FileNotFoundError(f"Missing required CSV prefix: {pref}")
        out[pref] = sorted(matches)[-1]
    return out


def round_strike(x):
    try:
        return round(float(x), 3)
    except Exception:
        return math.nan


def build_leg_map(quotes: pd.DataFrame):
    q = quotes.copy()
    q["k"] = list(
        zip(
            q["ticker"].astype(str).str.upper(),
            q["right"].astype(str).str.upper(),
            q["expiry"],
            q["strike"].map(round_strike),
        )
    )
    return q.drop_duplicates("k").set_index("k")["option_symbol"].to_dict()


def strategy_right(strategy: str):
    s = str(strategy).strip()
    if s in {"Bull Call Debit", "Bear Call Credit"}:
        return "C"
    if s in {"Bear Put Debit", "Bull Put Credit"}:
        return "P"
    if s in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        return "IC"
    return ""


def action_cell(strategy: str, track: str, optimal: str):
    if optimal == "Watch Only":
        return "\U0001F525\U0001F7E8 WATCH ONLY" if str(track).upper() == "FIRE" else "\U0001F6E1\ufe0f\U0001F7E8 WATCH ONLY"
    s = str(strategy)
    if s == "Bull Call Debit":
        return "\U0001F525\U0001F7E6 BULL CALL DEBIT"
    if s == "Bear Put Debit":
        return "\U0001F525\U0001F7E7 BEAR PUT DEBIT"
    if s == "Bull Put Credit":
        return "\U0001F6E1\ufe0f\U0001F7E9 BULL PUT CREDIT"
    if s == "Bear Call Credit":
        return "\U0001F6E1\ufe0f\U0001F7E5 BEAR CALL CREDIT"
    if s == "Iron Condor":
        return "\U0001F6E1\ufe0f\U0001F7EA IRON CONDOR"
    if s == "Iron Butterfly":
        return "\U0001F6E1\ufe0f\U0001F7EA IRON BUTTERFLY"
    if s == "Long Iron Condor":
        return "\U0001F525\U0001F7EA LONG IRON CONDOR"
    return s.upper()

def strike_setup(
    strategy,
    long_strike,
    short_strike,
    width,
    long_put_strike=None,
    short_put_strike=None,
    short_call_strike=None,
    long_call_strike=None,
):
    ls = float(long_strike) if np.isfinite(fnum(long_strike)) else math.nan
    ss = float(short_strike) if np.isfinite(fnum(short_strike)) else math.nan
    w = float(width) if np.isfinite(fnum(width)) else math.nan
    if strategy == "Bull Call Debit":
        return f"Buy {ls:.2f}C / Sell {ss:.2f}C ({w:.2f}w)"
    if strategy == "Bear Put Debit":
        return f"Buy {ls:.2f}P / Sell {ss:.2f}P ({w:.2f}w)"
    if strategy == "Bull Put Credit":
        return f"Sell {ss:.2f}P / Buy {ls:.2f}P ({w:.2f}w)"
    if strategy == "Bear Call Credit":
        return f"Sell {ss:.2f}C / Buy {ls:.2f}C ({w:.2f}w)"
    if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        lp = fnum(long_put_strike)
        sp = fnum(short_put_strike)
        sc = fnum(short_call_strike)
        lc = fnum(long_call_strike)
        if np.isfinite(lp) and np.isfinite(sp) and np.isfinite(sc) and np.isfinite(lc):
            if strategy == "Long Iron Condor":
                return f"Buy {lp:.2f}P / Sell {sp:.2f}P + Buy {lc:.2f}C / Sell {sc:.2f}C"
            return f"Sell {sp:.2f}P / Buy {lp:.2f}P + Sell {sc:.2f}C / Buy {lc:.2f}C"
    return "N/A"


def parse_gate_value(entry_gate: str):
    m = re.match(r"^\s*(>=|<=)\s*([0-9]*\.?[0-9]+)\s*(cr|db)\s*$", str(entry_gate).strip(), re.I)
    if not m:
        return None, None, None
    op, val, unit = m.groups()
    return op, float(val), unit.lower()


def calc_target_max(net_type: str, width: float, net: float):
    if net_type == "credit":
        return net * 100.0, (width - net) * 100.0
    return (width - net) * 100.0, net * 100.0


def calc_be(strategy, long_strike, short_strike, net):
    ls = float(long_strike)
    ss = float(short_strike)
    if strategy == "Bull Call Debit":
        return ls + net
    if strategy == "Bear Put Debit":
        return ls - net
    if strategy == "Bull Put Credit":
        return ss - net
    if strategy == "Bear Call Credit":
        return ss + net
    return math.nan


def calc_be_text(row, net):
    strategy = str(row.get("strategy", "")).strip()
    if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        sp = fnum(row.get("short_put_strike", row.get("short_strike")))
        sc = fnum(row.get("short_call_strike"))
        lp = fnum(row.get("long_put_strike"))
        lc = fnum(row.get("long_call_strike"))
        if strategy == "Long Iron Condor" and np.isfinite(lp) and np.isfinite(lc) and np.isfinite(net):
            return f"{(lp - net):.2f} / {(lc + net):.2f}"
        if np.isfinite(sp) and np.isfinite(sc) and np.isfinite(net):
            return f"{(sp - net):.2f} / {(sc + net):.2f}"
        return "N/A"
    return px(calc_be(strategy, row.get("long_strike"), row.get("short_strike"), net))


def calc_reward_risk(net_type: str, width: float, net: float) -> float:
    w = fnum(width)
    n = fnum(net)
    if not np.isfinite(w) or not np.isfinite(n) or w <= 0 or n <= 0 or n >= w:
        return math.nan
    if str(net_type).strip().lower() == "credit":
        return n / max(1e-9, (w - n))
    return (w - n) / max(1e-9, n)


def money(x):
    return "N/A" if not np.isfinite(fnum(x)) else f"${float(x):,.2f}"


def px(x):
    return "N/A" if not np.isfinite(fnum(x)) else f"{float(x):.2f}"


def likelihood_strength(verdict: str, edge_pct: float, signals: float):
    v = str(verdict).strip().upper()
    e = fnum(edge_pct)
    n = fnum(signals)
    if v == "LOW_SAMPLE":
        return "Low Sample"
    if not np.isfinite(e):
        return "N/A"
    if e < 0:
        return "Negative-Strong" if e <= -15 else "Negative"
    if np.isfinite(n) and n < 100:
        return "Low Sample"
    if e >= 25:
        return "Strong"
    if e >= 10:
        return "Moderate"
    return "Weak"


def strategy_is_credit_family(strategy: str) -> bool:
    s = str(strategy or "").strip()
    return s in {"Bull Put Credit", "Bear Call Credit", "Iron Condor", "Iron Butterfly"}


def normalize_track(track: str, strategy: str) -> str:
    t = str(track or "").strip().upper()
    if t in {"FIRE", "SHIELD"}:
        return t
    s = str(strategy or "").strip()
    if s in {"Bull Put Credit", "Bear Call Credit", "Iron Condor", "Iron Butterfly"}:
        return "SHIELD"
    if s in {"Bull Call Debit", "Bear Put Debit", "Long Iron Condor"}:
        return "FIRE"
    return "UNKNOWN"


def fetch_open_positions_from_schwab(cache_csv: Path):
    try:
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    except Exception as exc:
        return False, f"import_error:{exc}"
    try:
        cfg_live = SchwabAuthConfig.from_env(load_dotenv_file=True)
        svc = SchwabLiveDataService(cfg_live)
        svc.connect()
        cli = svc._client
        resp = cli.get_accounts(fields=[cli.Account.Fields.POSITIONS])
        resp.raise_for_status()
        raw = resp.json()
        accounts = raw if isinstance(raw, list) else [raw]
        rows = []
        for a in accounts:
            sec = a.get("securitiesAccount", {}) if isinstance(a, dict) else {}
            acct_num = sec.get("accountNumber", "")
            for p in sec.get("positions", []) or []:
                inst = p.get("instrument") or {}
                rows.append(
                    {
                        "account_number": acct_num,
                        "symbol": inst.get("symbol", ""),
                        "description": inst.get("description", ""),
                        "asset_type": inst.get("assetType", ""),
                        "position_type": p.get("positionType", ""),
                        "long_quantity": p.get("longQuantity"),
                        "short_quantity": p.get("shortQuantity"),
                        "average_price": p.get("averagePrice"),
                        "market_value": p.get("marketValue"),
                        "maintenance_requirement": p.get("maintenanceRequirement"),
                        "current_day_profit_loss": p.get("currentDayProfitLoss"),
                        "current_day_profit_loss_pct": p.get("currentDayProfitLossPercentage"),
                    }
                )
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cache_csv, index=False)
        return True, ""
    except Exception as exc:
        return False, f"fetch_error:{exc}"


def build_portfolio_risk_book(open_positions_csv: Path):
    if not open_positions_csv.exists():
        return {"ok": False, "error": f"missing_open_positions_csv:{open_positions_csv}"}
    try:
        pos = load_open_positions(open_positions_csv)
    except Exception as exc:
        return {"ok": False, "error": f"load_open_positions_failed:{exc}"}
    if pos.empty:
        return {
            "ok": True,
            "total_risk": 0.0,
            "short_put_risk": 0.0,
            "symbol_risk": {},
            "short_put_expiry_risk": {},
        }
    is_option = (
        pos["asset_type"].astype(str).str.upper().eq("OPTION")
        | pos["strategy"].astype(str).str.contains("Option", case=False, na=False)
        | pos["symbol"].astype(str).str.contains(r"\d{6}[CP]\d{8}", na=False)
    )
    pos = pos[is_option].copy()
    pos["risk"] = pos["risk"].map(fnum).fillna(np.nan)
    pos = pos[pos["risk"].notna()].copy()
    pos["risk"] = pos["risk"].abs()
    pos = pos[pos["risk"] > 0].copy()
    if pos.empty:
        return {
            "ok": True,
            "total_risk": 0.0,
            "short_put_risk": 0.0,
            "symbol_risk": {},
            "short_put_expiry_risk": {},
        }
    short_put_mask = pos["strategy"].astype(str).isin(["Short Put Option", "Bull Put Credit"])
    symbol_risk = (
        pos.groupby(pos["underlying"].astype(str).str.upper().str.strip())["risk"].sum().to_dict()
    )
    expiry_risk = {}
    if short_put_mask.any():
        sp = pos[short_put_mask & pos["expiry"].notna()].copy()
        if not sp.empty:
            expiry_risk = sp.groupby(sp["expiry"].dt.date.astype(str))["risk"].sum().to_dict()
    total_risk = float(pos["risk"].sum())
    short_put_risk = float(pos.loc[short_put_mask, "risk"].sum())
    return {
        "ok": True,
        "total_risk": total_risk,
        "short_put_risk": short_put_risk,
        "symbol_risk": {str(k): float(v) for k, v in symbol_risk.items()},
        "short_put_expiry_risk": {str(k): float(v) for k, v in expiry_risk.items()},
    }


def candidate_uses_short_put_risk(strategy: str) -> bool:
    s = str(strategy or "").strip()
    return s in {"Bull Put Credit", "Iron Condor", "Iron Butterfly"}


def run():
    ap = argparse.ArgumentParser(description="MODE A two-stage runner (discovery + live execution)")
    ap.add_argument("--base-dir", default=r"c:\uw_root\2026-02-05")
    ap.add_argument("--config", default=str((Path(__file__).resolve().parent / "rulebook_config.yaml")))
    ap.add_argument("--out-dir", default=r"c:\uw_root\out")
    ap.add_argument("--top-trades", type=int, default=20)
    ap.add_argument("--output", default="")
    ap.add_argument(
        "--strict-stage2",
        action="store_true",
        help="Fail if Stage-2 live pricing fails. Default behavior reuses existing same-date live files if present.",
    )
    args = ap.parse_args()
    run_started_utc = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    base = Path(args.base_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(args.config).resolve()

    csvs = pick_csvs(base)
    asof_str = detect_asof_from_names(list(csvs.values()))
    asof = dt.datetime.strptime(asof_str, "%Y-%m-%d").date()

    if not args.output:
        output_path = base / f"anu-expert-trade-table-{asof_str}.md"
    else:
        output_path = Path(args.output).resolve()

    whale_md = base / f"whale-{asof_str}.md"
    if not whale_md.exists():
        alt = sorted(base.glob("whale-*.md"))
        whale_md = alt[-1] if alt else whale_md
    if not whale_md.exists():
        raise FileNotFoundError(f"Missing whale markdown in {base}")

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    approval_cfg = cfg.get("approval", {}) if isinstance(cfg, dict) else {}
    engine_cfg = cfg.get("engine", {}) if isinstance(cfg, dict) else {}
    discovery_multiplier = fnum(engine_cfg.get("discovery_multiplier", 5))
    if not np.isfinite(discovery_multiplier) or discovery_multiplier < 1:
        discovery_multiplier = 5
    discovery_top = max(int(args.top_trades), int(round(int(args.top_trades) * float(discovery_multiplier))))
    final_max_per_ticker = int(engine_cfg.get("final_max_trades_per_ticker", 1))
    final_max_per_ticker = max(1, final_max_per_ticker)
    backtest_min_signals = fnum(approval_cfg.get("min_signals", 100))
    if not np.isfinite(backtest_min_signals) or backtest_min_signals <= 0:
        backtest_min_signals = 100
    hot_df = pd.read_csv(csvs["hot-chains-"], low_memory=False)
    oi_df = pd.read_csv(csvs["chain-oi-changes-"], low_memory=False)
    dp_df = pd.read_csv(csvs["dp-eod-report-"], low_memory=False)
    sc_df = pd.read_csv(csvs["stock-screener-"], low_memory=False)

    ensure_cols(hot_df, csvs["hot-chains-"].name, ["option_symbol", "date", "bid", "ask", "volume", "open_interest"])
    ensure_cols(
        oi_df,
        csvs["chain-oi-changes-"].name,
        ["option_symbol", "curr_date", "last_bid", "last_ask", "curr_oi", "volume"],
    )
    ensure_cols(
        sc_df,
        csvs["stock-screener-"].name,
        [
            "ticker",
            "close",
            "issue_type",
            "is_index",
            "next_earnings_date",
            "bullish_premium",
            "bearish_premium",
            "call_premium",
            "put_premium",
            "put_call_ratio",
        ],
    )
    spot_map = (
        sc_df.assign(ticker=sc_df["ticker"].astype(str).str.upper().str.strip())
        .drop_duplicates("ticker")
        .set_index("ticker")["close"]
        .map(fnum)
        .to_dict()
    )
    max_strike_distance_pct = fnum(cfg.get("gates", {}).get("max_strike_distance_pct", 0.80))
    if not np.isfinite(max_strike_distance_pct) or max_strike_distance_pct <= 0:
        max_strike_distance_pct = math.nan
    _ = dp_df  # loaded intentionally; stage-1 model already relies on screener + quotes + whale tables.

    whale_tables = md_tables(whale_md.read_text(encoding="utf-8", errors="replace"))
    quotes = build_quotes(hot_df, oi_df, asof, csvs["hot-chains-"].name, csvs["chain-oi-changes-"].name)
    best = build_best_candidates(asof, cfg, sc_df, quotes, whale_tables, top_trades=discovery_top)
    if not best:
        raise RuntimeError("No stage-1 candidates produced.")

    leg_map = build_leg_map(quotes)
    shortlist_rows = []
    dropped_stage1 = []

    def strike_sanity_ok(
        ticker: str,
        strategy: str,
        long_strike_v: float,
        short_strike_v: float,
        long_put_v: float,
        short_put_v: float,
        short_call_v: float,
        long_call_v: float,
    ) -> bool:
        if not np.isfinite(max_strike_distance_pct):
            return True
        spot = fnum(spot_map.get(ticker))
        if not np.isfinite(spot) or spot <= 0:
            return True
        s = str(strategy).strip()
        to_check = []
        if s in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
            to_check.extend([long_put_v, short_put_v, short_call_v, long_call_v])
        else:
            to_check.extend([long_strike_v, short_strike_v])
        for strike_val in to_check:
            x = fnum(strike_val)
            if not np.isfinite(x) or x <= 0:
                continue
            dist = abs((x / spot) - 1.0)
            if dist > max_strike_distance_pct:
                return False
        return True

    for r in best:
        ticker = str(r["ticker"]).upper()
        strategy = str(r["strategy"])
        expiry = r["expiry"]
        long_strike = round_strike(r.get("long_strike"))
        short_strike = round_strike(r.get("short_strike"))

        short_leg = ""
        long_leg = ""
        short_put_leg = ""
        long_put_leg = ""
        short_call_leg = ""
        long_call_leg = ""
        long_put_strike = fnum(r.get("long_put_strike", long_strike))
        short_put_strike = fnum(r.get("short_put_strike", short_strike))
        short_call_strike = fnum(r.get("short_call_strike"))
        long_call_strike = fnum(r.get("long_call_strike"))
        if not strike_sanity_ok(
            ticker,
            strategy,
            long_strike,
            short_strike,
            long_put_strike,
            short_put_strike,
            short_call_strike,
            long_call_strike,
        ):
            dropped_stage1.append(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                    "stage": "stage1",
                    "drop_reason": "strike_sanity_fail",
                }
            )
            continue

        if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
            short_put_leg = str(r.get("short_put_symbol", "")).strip()
            long_put_leg = str(r.get("long_put_symbol", "")).strip()
            short_call_leg = str(r.get("short_call_symbol", "")).strip()
            long_call_leg = str(r.get("long_call_symbol", "")).strip()

            if not short_put_leg:
                short_put_key = (ticker, "P", expiry, round_strike(short_put_strike))
                short_put_leg = leg_map.get(short_put_key, "")
            if not long_put_leg:
                long_put_key = (ticker, "P", expiry, round_strike(long_put_strike))
                long_put_leg = leg_map.get(long_put_key, "")
            if not short_call_leg:
                short_call_key = (ticker, "C", expiry, round_strike(short_call_strike))
                short_call_leg = leg_map.get(short_call_key, "")
            if not long_call_leg:
                long_call_key = (ticker, "C", expiry, round_strike(long_call_strike))
                long_call_leg = leg_map.get(long_call_key, "")

            if not short_put_leg or not long_put_leg or not short_call_leg or not long_call_leg:
                dropped_stage1.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy,
                        "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                        "stage": "stage1",
                        "drop_reason": "missing_leg_symbol_mapping",
                    }
                )
                continue
            short_leg = short_put_leg
            long_leg = long_put_leg
        else:
            right = strategy_right(strategy)
            if strategy in {"Bull Call Debit", "Bear Put Debit"}:
                long_key = (ticker, right, expiry, long_strike)
                short_key = (ticker, right, expiry, short_strike)
            else:
                short_key = (ticker, right, expiry, short_strike)
                long_key = (ticker, right, expiry, long_strike)

            long_leg = leg_map.get(long_key)
            short_leg = leg_map.get(short_key)
            if not long_leg or not short_leg:
                dropped_stage1.append(
                    {
                        "ticker": ticker,
                        "strategy": strategy,
                        "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                        "stage": "stage1",
                        "drop_reason": "missing_leg_symbol_mapping",
                    }
                )
                continue

        net = fnum(r.get("net"))
        net_type = str(r.get("net_type", "")).strip().lower()
        if not np.isfinite(net):
            dropped_stage1.append(
                {
                    "ticker": ticker,
                    "strategy": strategy,
                    "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                    "stage": "stage1",
                    "drop_reason": "invalid_net",
                }
            )
            continue
        entry_gate = f">= {net:.2f} cr" if net_type == "credit" else f"<= {net:.2f} db"

        shortlist_rows.append(
            {
                "ticker": ticker,
                "strategy": strategy,
                "expiry": expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry),
                "short_leg": short_leg,
                "long_leg": long_leg,
                "short_put_leg": short_put_leg or short_leg,
                "long_put_leg": long_put_leg or long_leg,
                "short_call_leg": short_call_leg,
                "long_call_leg": long_call_leg,
                "net_type": net_type,
                "entry_gate": entry_gate,
                "width": float(r["width"]),
                "conviction": int(r["conviction"]),
                "track": str(r.get("track", "")),
                "confidence_tier": str(r.get("tier", "")),
                "optimal_stage1": str(r.get("optimal", "")),
                "notes_stage1": str(r.get("notes", "")),
                "thesis": str(r.get("thesis", "")),
                "invalidation": str(r.get("invalidation", "")),
                "sigma_pass_stage1": r.get("sigma_pass", np.nan),
                "core_ok_stage1": r.get("core_ok", np.nan),
                "high_beta_pass_stage1": r.get("high_beta_pass", np.nan),
                "earnings_label_stage1": str(r.get("earnings_label", "")),
                "range_neutrality_stage1": r.get("range_neutrality", np.nan),
                "long_strike": float(long_strike) if np.isfinite(fnum(long_strike)) else np.nan,
                "short_strike": float(short_strike) if np.isfinite(fnum(short_strike)) else np.nan,
                "long_put_strike": float(long_put_strike) if np.isfinite(long_put_strike) else np.nan,
                "short_put_strike": float(short_put_strike) if np.isfinite(short_put_strike) else np.nan,
                "short_call_strike": float(short_call_strike) if np.isfinite(short_call_strike) else np.nan,
                "long_call_strike": float(long_call_strike) if np.isfinite(long_call_strike) else np.nan,
                "put_width": float(r.get("put_width")) if np.isfinite(fnum(r.get("put_width"))) else np.nan,
                "call_width": float(r.get("call_width")) if np.isfinite(fnum(r.get("call_width"))) else np.nan,
            }
        )

    shortlist = pd.DataFrame(shortlist_rows)
    if shortlist.empty:
        raise RuntimeError("No shortlist rows with valid leg symbols.")
    stage1_rank = {"Yes-Prime": 0, "Yes-Good": 1, "Watch Only": 2}
    shortlist["_stage1_rank"] = shortlist["optimal_stage1"].map(stage1_rank).fillna(3).astype(int)
    shortlist = (
        shortlist.sort_values(["_stage1_rank", "conviction"], ascending=[True, False])
        .head(max(1, int(discovery_top)))
        .drop(columns=["_stage1_rank"])
        .reset_index(drop=True)
    )
    shortlist_csv = out_dir / f"shortlist_trades_{asof_str}_mode_a.csv"
    shortlist.to_csv(shortlist_csv, index=False)

    likelihood_csv = out_dir / f"setup_likelihood_{asof_str}.csv"
    likelihood_cmd = [
        sys.executable,
        "-m",
        "uwos.setup_likelihood_backtest",
        "--setups-csv",
        str(shortlist_csv),
        "--asof-date",
        asof_str,
        "--root-dir",
        str(Path.cwd().resolve()),
        "--out-dir",
        str(out_dir),
        "--cache-dir",
        str((out_dir / "cache" / "yf").resolve()),
        "--lookback-years",
        "2",
        "--min-signals",
        str(int(backtest_min_signals)),
    ]
    subprocess.run(likelihood_cmd, check=True)

    live_csv = out_dir / f"live_trade_table_{asof_str}.csv"
    live_final_csv = out_dir / f"live_trade_table_{asof_str}_final.csv"
    cmd = [
        sys.executable,
        "-m",
        "uwos.pricer",
        "--shortlist-csv",
        str(shortlist_csv),
        "--out-dir",
        str(out_dir),
        "--top",
        str(int(discovery_top)),
        "--min-conviction",
        "0",
        "--save-chain-dir",
        str((out_dir / f"schwab_snapshot_{asof_str}" / "chains").resolve()),
        "--snapshot-out-json",
        str((out_dir / f"schwab_snapshot_{asof_str}.json").resolve()),
    ]
    stage2_reused_existing = False
    stage2_error = ""
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        stage2_error = str(exc)
        if (not args.strict_stage2) and live_csv.exists() and live_final_csv.exists():
            stage2_reused_existing = True
            print(
                "WARN: Stage-2 live pricing failed; reusing existing same-date live outputs: "
                f"{live_csv.name}, {live_final_csv.name}"
            )
        else:
            raise

    if not live_csv.exists():
        raise FileNotFoundError(f"Missing live output: {live_csv}")
    if not live_final_csv.exists():
        raise FileNotFoundError(f"Missing live final output: {live_final_csv}")

    live = pd.read_csv(live_csv, low_memory=False)
    key = ["ticker", "strategy", "expiry", "short_leg", "long_leg", "short_call_leg", "long_call_leg"]
    for col in ["short_call_leg", "long_call_leg"]:
        if col not in shortlist.columns:
            shortlist[col] = ""
        if col not in live.columns:
            live[col] = ""
    for col in key:
        if col not in shortlist.columns:
            shortlist[col] = ""
        if col not in live.columns:
            live[col] = ""
        shortlist[col] = shortlist[col].fillna("").astype(str)
        live[col] = live[col].fillna("").astype(str)
    base_live_cols = [
        "live_status",
        "is_final_live_valid",
        "invalidation_breached_live",
        "invalidation_rule_op",
        "invalidation_rule_level",
        "invalidation_eval_price_live",
        "live_net_bid_ask",
        "live_max_profit",
        "live_max_loss",
        "gate_pass_live",
        "short_bid_live",
        "short_ask_live",
        "short_delta_live",
        "long_bid_live",
        "long_ask_live",
        "short_put_bid_live",
        "short_put_ask_live",
        "short_put_delta_live",
        "long_put_bid_live",
        "long_put_ask_live",
        "short_call_bid_live",
        "short_call_ask_live",
        "short_call_delta_live",
        "long_call_bid_live",
        "long_call_ask_live",
        "spot_live_last",
        "spot_live_bid",
        "spot_live_ask",
        "entry_structure_ok_live",
        "entry_structure_reason_live",
    ]
    live_cols = [c for c in (key + base_live_cols) if c in live.columns]
    mdf = shortlist.merge(live[live_cols], on=key, how="left", suffixes=("", "_live"))

    if likelihood_csv.exists():
        like_df = pd.read_csv(likelihood_csv, low_memory=False)
        like_df["ticker"] = like_df["ticker"].astype(str).str.upper().str.strip()
        like_df["strategy"] = like_df["strategy"].astype(str).str.strip()
        like_df["expiry"] = like_df["expiry"].astype(str).str[:10]
        like_df["entry_gate"] = like_df["entry_gate"].astype(str).str.strip()
        like_keep = [
            "ticker",
            "strategy",
            "expiry",
            "entry_gate",
            "hist_success_pct",
            "edge_pct",
            "signals",
            "verdict",
            "confidence",
            "credit_no_touch_pct",
        ]
        like_df = like_df[like_keep].drop_duplicates(subset=["ticker", "strategy", "expiry", "entry_gate"])
        mdf["entry_gate"] = mdf["entry_gate"].astype(str).str.strip()
        mdf["expiry"] = mdf["expiry"].astype(str).str[:10]
        mdf = mdf.merge(
            like_df,
            on=["ticker", "strategy", "expiry", "entry_gate"],
            how="left",
            suffixes=("", "_lk"),
        )
        mdf["verdict"] = mdf["verdict"].fillna("UNKNOWN").astype(str).str.upper().str.strip()
        mdf["confidence"] = mdf["confidence"].fillna("Unknown").astype(str)
    else:
        mdf["hist_success_pct"] = np.nan
        mdf["edge_pct"] = np.nan
        mdf["signals"] = np.nan
        mdf["verdict"] = "UNKNOWN"
        mdf["confidence"] = "Unknown"
        mdf["credit_no_touch_pct"] = np.nan

    require_likelihood_pass = bool(approval_cfg.get("require_likelihood_pass", True))
    shield_live_valid_overrides_quality = bool(
        approval_cfg.get("shield_live_valid_overrides_quality", False)
    )
    min_edge_pct = fnum(approval_cfg.get("min_edge_pct", 0.0))
    min_signals = fnum(approval_cfg.get("min_signals", 100))
    require_invalidation_clear = bool(approval_cfg.get("require_invalidation_clear", False))
    block_invalidation_warning = bool(approval_cfg.get("block_invalidation_warning", False))
    allow_stage1_watch_promotion = bool(approval_cfg.get("allow_stage1_watch_promotion", True))
    stage1_promote_min_conv = fnum(approval_cfg.get("stage1_watch_promotion_min_conviction", 58))
    stage1_promote_min_edge = fnum(approval_cfg.get("stage1_watch_promotion_min_edge_pct", 5.0))
    stage1_promote_min_signals = fnum(approval_cfg.get("stage1_watch_promotion_min_signals", min_signals))
    min_likelihood_strength = str(approval_cfg.get("min_likelihood_strength", "")).strip()
    disallow_likelihood_strengths = {
        str(x).strip().upper()
        for x in approval_cfg.get("disallow_likelihood_strengths", [])
        if str(x).strip()
    }
    disallow_confidence_tiers = {
        str(x).strip().upper()
        for x in approval_cfg.get("disallow_confidence_tiers", [])
        if str(x).strip()
    }
    require_shield_sigma_pass = bool(approval_cfg.get("require_shield_sigma_pass", False))
    shield_sigma_require_data = bool(approval_cfg.get("shield_sigma_require_data", True))
    require_shield_core = bool(approval_cfg.get("require_shield_core", False))
    require_live_shield_short_delta = bool(approval_cfg.get("require_live_shield_short_delta", False))
    max_abs_short_delta_shield = fnum(approval_cfg.get("max_abs_short_delta_shield", 0.20))
    entry_tol_debit_abs = fnum(approval_cfg.get("entry_tolerance_debit_abs", 0.20))
    entry_tol_credit_abs = fnum(approval_cfg.get("entry_tolerance_credit_abs", 0.20))
    entry_tol_pct = fnum(approval_cfg.get("entry_tolerance_pct", 0.05))
    require_spot_alignment = bool(approval_cfg.get("require_spot_alignment", True))
    spot_alignment_require_live = bool(approval_cfg.get("spot_alignment_require_live", True))
    max_spot_asof_drift_pct = fnum(approval_cfg.get("max_spot_asof_drift_pct", 0.35))
    max_bull_call_long_otm_pct = fnum(approval_cfg.get("max_bull_call_long_otm_pct"))
    max_bear_put_long_otm_pct = fnum(approval_cfg.get("max_bear_put_long_otm_pct"))
    exclude_debit_moneyness_fail_from_output = bool(
        approval_cfg.get("exclude_debit_moneyness_fail_from_output", False)
    )
    min_credit_no_touch_pct = fnum(approval_cfg.get("min_credit_no_touch_pct"))
    credit_no_touch_require_data = bool(approval_cfg.get("credit_no_touch_require_data", False))
    enable_restrike_optimizer = bool(approval_cfg.get("enable_restrike_optimizer", True))
    invalidation_eval_mode = str(approval_cfg.get("invalidation_eval_mode", "auto")).strip().lower()
    if invalidation_eval_mode not in {"auto", "live", "asof_close"}:
        invalidation_eval_mode = "auto"
    if invalidation_eval_mode == "asof_close":
        use_asof_close_for_invalidation = True
    elif invalidation_eval_mode == "live":
        use_asof_close_for_invalidation = False
    else:
        use_asof_close_for_invalidation = bool(asof < dt.date.today())
    if not np.isfinite(entry_tol_debit_abs) or entry_tol_debit_abs < 0:
        entry_tol_debit_abs = 0.0
    if not np.isfinite(entry_tol_credit_abs) or entry_tol_credit_abs < 0:
        entry_tol_credit_abs = 0.0
    if not np.isfinite(entry_tol_pct) or entry_tol_pct < 0:
        entry_tol_pct = 0.0
    gates_cfg_local = cfg.get("gates", {}) if isinstance(cfg, dict) else {}
    min_credit_pct_width_cfg = fnum(gates_cfg_local.get("min_credit_pct_width", 0.30))
    max_credit_pct_width_cfg = fnum(gates_cfg_local.get("max_credit_pct_width", 0.55))
    if not np.isfinite(min_credit_pct_width_cfg) or min_credit_pct_width_cfg <= 0:
        min_credit_pct_width_cfg = 0.30
    if not np.isfinite(max_credit_pct_width_cfg) or max_credit_pct_width_cfg <= 0:
        max_credit_pct_width_cfg = 0.55
    ideal_credit_low_pct = max(0.30, min_credit_pct_width_cfg)
    ideal_credit_high_pct = min(0.40, max_credit_pct_width_cfg)
    if ideal_credit_high_pct < ideal_credit_low_pct:
        ideal_credit_high_pct = ideal_credit_low_pct
    if not np.isfinite(max_spot_asof_drift_pct) or max_spot_asof_drift_pct < 0:
        max_spot_asof_drift_pct = 0.35
    if not np.isfinite(max_bull_call_long_otm_pct) or max_bull_call_long_otm_pct < 0:
        max_bull_call_long_otm_pct = math.nan
    if not np.isfinite(max_bear_put_long_otm_pct) or max_bear_put_long_otm_pct < 0:
        max_bear_put_long_otm_pct = math.nan
    if not np.isfinite(max_abs_short_delta_shield) or max_abs_short_delta_shield <= 0:
        max_abs_short_delta_shield = 0.20
    enforce_pretrade_caps = bool(approval_cfg.get("enforce_pretrade_portfolio_caps", False))
    pretrade_caps_require_data = bool(approval_cfg.get("pretrade_caps_require_data", False))
    pretrade_open_positions_csv = str(approval_cfg.get("pretrade_open_positions_csv", "")).strip()
    risk_cfg = cfg.get("playbook", {}).get("risk_limits", {}) if isinstance(cfg, dict) else {}
    short_put_limit = fnum(risk_cfg.get("short_put_max_share", 0.35))
    symbol_limit = fnum(risk_cfg.get("single_symbol_max_share", 0.10))
    expiry_limit = fnum(risk_cfg.get("single_expiry_max_share_short_put", 0.25))
    if not np.isfinite(short_put_limit) or short_put_limit <= 0:
        short_put_limit = 0.35
    if not np.isfinite(symbol_limit) or symbol_limit <= 0:
        symbol_limit = 0.10
    if not np.isfinite(expiry_limit) or expiry_limit <= 0:
        expiry_limit = 0.25

    def gate_context(row):
        net_type = str(row.get("net_type", "")).strip().lower()
        live_status = str(row.get("live_status", "")).strip()
        live_net = fnum(row.get("live_net_bid_ask"))
        gate_pass_raw = bool(row.get("gate_pass_live")) if pd.notna(row.get("gate_pass_live")) else False
        _, gate_target, _ = parse_gate_value(row.get("entry_gate", ""))
        tol_abs = entry_tol_credit_abs if net_type == "credit" else entry_tol_debit_abs
        tol_total = tol_abs
        if np.isfinite(gate_target) and np.isfinite(entry_tol_pct):
            tol_total = tol_abs + abs(gate_target) * entry_tol_pct

        near_miss = False
        pass_effective = gate_pass_raw
        miss_abs = math.nan
        if np.isfinite(gate_target) and np.isfinite(live_net):
            if net_type == "debit":
                miss_abs = max(0.0, live_net - gate_target)
            else:
                miss_abs = max(0.0, gate_target - live_net)
            if (not gate_pass_raw) and miss_abs <= tol_total and live_status == "fails_live_entry_gate":
                near_miss = True
                pass_effective = True

        return {
            "gate_target": gate_target,
            "gate_live_net": live_net,
            "gate_tol_total": tol_total,
            "gate_miss_abs": miss_abs,
            "gate_pass_effective": bool(pass_effective),
            "gate_near_miss": bool(near_miss),
        }

    gate_ctx_df = pd.DataFrame([gate_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), gate_ctx_df], axis=1)

    def stage1_context(row):
        opt = str(row.get("optimal_stage1", "")).strip()
        is_yes = opt in {"Yes-Prime", "Yes-Good"}
        verdict = str(row.get("verdict", "")).strip().upper()
        edge = fnum(row.get("edge_pct"))
        sig = fnum(row.get("signals"))
        conv = fnum(row.get("conviction"))
        promoted = False
        reason = ""
        if is_yes:
            reason = "stage1_yes"
        elif allow_stage1_watch_promotion:
            cond = (
                verdict == "PASS"
                and (not np.isfinite(stage1_promote_min_conv) or (np.isfinite(conv) and conv >= stage1_promote_min_conv))
                and (not np.isfinite(stage1_promote_min_edge) or (np.isfinite(edge) and edge >= stage1_promote_min_edge))
                and (
                    not np.isfinite(stage1_promote_min_signals)
                    or stage1_promote_min_signals <= 0
                    or (np.isfinite(sig) and sig >= stage1_promote_min_signals)
                )
            )
            promoted = bool(cond)
            reason = "stage1_promoted" if promoted else "stage1_watch_blocked"
        else:
            reason = "stage1_watch_blocked"
        return {
            "stage1_is_yes": bool(is_yes),
            "stage1_promoted": bool(promoted),
            "stage1_effective": bool(is_yes or promoted),
            "stage1_blocked": bool((not is_yes) and (not promoted)),
            "stage1_eval_reason": reason,
        }

    stage1_ctx_df = pd.DataFrame([stage1_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), stage1_ctx_df], axis=1)

    def invalidation_context(row):
        op = str(row.get("invalidation_rule_op", "")).strip()
        lvl = fnum(row.get("invalidation_rule_level"))
        live_px = fnum(row.get("invalidation_eval_price_live"))
        eval_source = "live"
        eval_px = live_px
        if use_asof_close_for_invalidation:
            ticker = str(row.get("ticker", "")).strip().upper()
            asof_px = fnum(spot_map.get(ticker))
            if np.isfinite(asof_px):
                eval_px = asof_px
                eval_source = "asof_close"
        breached = False
        if op in {"<", "<="} and np.isfinite(lvl) and np.isfinite(eval_px):
            breached = eval_px < lvl if op == "<" else eval_px <= lvl
        elif op in {">", ">="} and np.isfinite(lvl) and np.isfinite(eval_px):
            breached = eval_px > lvl if op == ">" else eval_px >= lvl
        elif pd.notna(row.get("invalidation_breached_live")):
            breached = bool(row.get("invalidation_breached_live"))
        return {
            "invalidation_eval_source": eval_source,
            "invalidation_eval_price_effective": eval_px,
            "invalidation_breached_effective": bool(breached),
        }

    invalid_ctx_df = pd.DataFrame([invalidation_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), invalid_ctx_df], axis=1)

    def spot_context(row):
        ticker = str(row.get("ticker", "")).strip().upper()
        asof_spot = fnum(spot_map.get(ticker))
        live_last = fnum(row.get("spot_live_last"))
        live_bid = fnum(row.get("spot_live_bid"))
        live_ask = fnum(row.get("spot_live_ask"))
        if np.isfinite(live_last):
            live_spot = float(live_last)
        elif np.isfinite(live_bid) and np.isfinite(live_ask):
            live_spot = (float(live_bid) + float(live_ask)) / 2.0
        elif np.isfinite(live_bid):
            live_spot = float(live_bid)
        elif np.isfinite(live_ask):
            live_spot = float(live_ask)
        else:
            live_spot = math.nan
        drift = math.nan
        if np.isfinite(asof_spot) and asof_spot > 0 and np.isfinite(live_spot):
            drift = abs(live_spot - asof_spot) / asof_spot
        return {
            "spot_asof_close": asof_spot,
            "spot_live_effective": live_spot,
            "spot_asof_live_drift_pct": drift,
        }

    spot_ctx_df = pd.DataFrame([spot_context(r) for _, r in mdf.iterrows()])
    mdf = pd.concat([mdf.reset_index(drop=True), spot_ctx_df], axis=1)

    def rank_likelihood_strength(value: str) -> int:
        s = str(value or "").strip().upper()
        order = {
            "NEGATIVE-STRONG": 0,
            "NEGATIVE": 1,
            "WEAK": 2,
            "MODERATE": 3,
            "STRONG": 4,
        }
        return order.get(s, -1)

    mdf["likelihood_strength"] = mdf.apply(
        lambda row: likelihood_strength(
            str(row.get("verdict", "")),
            fnum(row.get("edge_pct")),
            fnum(row.get("signals")),
        ),
        axis=1,
    )

    def ev_score(row):
        hs = fnum(row.get("hist_success_pct"))
        p = hs / 100.0 if np.isfinite(hs) else math.nan
        live_max_profit = fnum(row.get("live_max_profit"))
        live_max_loss = fnum(row.get("live_max_loss"))
        if not (np.isfinite(live_max_profit) and np.isfinite(live_max_loss)):
            gate_target = fnum(row.get("gate_target"))
            width = fnum(row.get("width"))
            net_type = str(row.get("net_type", "")).strip().lower()
            if np.isfinite(gate_target) and np.isfinite(width):
                tgt_max_profit, tgt_max_loss = calc_target_max(net_type, width, gate_target)
                live_max_profit = tgt_max_profit
                live_max_loss = tgt_max_loss
        ev_cash = math.nan
        if np.isfinite(p) and np.isfinite(live_max_profit) and np.isfinite(live_max_loss):
            ev_cash = (p * live_max_profit) - ((1.0 - p) * live_max_loss)
        ev_risk = ev_cash / live_max_loss if (np.isfinite(ev_cash) and np.isfinite(live_max_loss) and live_max_loss > 0) else math.nan
        conv = fnum(row.get("conviction"))
        edge = fnum(row.get("edge_pct"))
        gate_near_miss = bool(row.get("gate_near_miss")) if pd.notna(row.get("gate_near_miss")) else False

        score = 0.0
        if np.isfinite(ev_risk):
            score += 100.0 * ev_risk
        elif np.isfinite(ev_cash):
            score += ev_cash / 10.0
        if np.isfinite(edge):
            score += 0.5 * edge
        if np.isfinite(conv):
            score += 0.1 * (conv - 50.0)
        if gate_near_miss:
            score -= 0.5
        return score

    mdf["_ev_sort"] = mdf.apply(ev_score, axis=1)

    def strike_distance(base_row, cand_row):
        cols = [
            "long_strike",
            "short_strike",
            "long_put_strike",
            "short_put_strike",
            "short_call_strike",
            "long_call_strike",
        ]
        dist = 0.0
        used = False
        for c in cols:
            a = fnum(base_row.get(c))
            b = fnum(cand_row.get(c))
            if np.isfinite(a) and np.isfinite(b):
                dist += abs(a - b)
                used = True
        return dist if used else 1e9

    if enable_restrike_optimizer and not mdf.empty:
        stage1_rank_map = {"Yes-Prime": 0, "Yes-Good": 1, "Watch Only": 2}
        restrike_from = pd.Series([pd.NA] * len(mdf), index=mdf.index, dtype="Int64")
        restrike_reason = pd.Series([""] * len(mdf), index=mdf.index, dtype="string")
        selected_idx = []
        family_cols = ["ticker", "strategy", "expiry", "track"]

        for _, fam in mdf.groupby(family_cols, dropna=False):
            fam_local = fam.copy()
            fam_local["_stage1_rank"] = (
                fam_local["optimal_stage1"].map(stage1_rank_map).fillna(3).astype(int)
            )
            fam_local = fam_local.sort_values(
                ["_stage1_rank", "conviction", "_ev_sort"],
                ascending=[True, False, False],
            )
            base_idx = fam_local.index[0]
            base_row = mdf.loc[base_idx]
            pick_idx = base_idx

            base_live_status = str(base_row.get("live_status", "")).strip()
            base_gate_effective = (
                bool(base_row.get("gate_pass_effective"))
                if pd.notna(base_row.get("gate_pass_effective"))
                else False
            )
            base_struct_ok = (
                bool(base_row.get("entry_structure_ok_live"))
                if pd.notna(base_row.get("entry_structure_ok_live"))
                else True
            )

            if (
                (base_live_status == "fails_live_entry_gate" or (not base_gate_effective) or (not base_struct_ok))
                and len(fam_local) > 1
            ):
                exec_pool = fam_local[
                    fam_local["live_status"].astype(str).eq("ok_live")
                    & fam_local["gate_pass_effective"].fillna(False).astype(bool)
                    & fam_local["entry_structure_ok_live"].fillna(True).astype(bool)
                ].copy()
                if not exec_pool.empty:
                    exec_pool["_dist"] = exec_pool.apply(
                        lambda rr: strike_distance(base_row, rr), axis=1
                    )
                    exec_pool = exec_pool.sort_values(
                        ["_dist", "_ev_sort", "conviction"],
                        ascending=[True, False, False],
                    )
                    pick_idx = exec_pool.index[0]
                    restrike_from.loc[pick_idx] = int(base_idx)
                    restrike_reason.loc[pick_idx] = "family_restrike_from_gate_fail"

            selected_idx.append(int(pick_idx))

        selected_unique = sorted(set(selected_idx))
        mdf = mdf.loc[selected_unique].copy()
        mdf["restrike_replaced_from"] = restrike_from.loc[selected_unique].values
        mdf["restrike_reason"] = restrike_reason.loc[selected_unique].fillna("").astype(str).values
        mdf = mdf.reset_index(drop=True)

    def bool_or_none(value):
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in {"true", "t", "1", "yes", "y"}:
            return True
        if s in {"false", "f", "0", "no", "n"}:
            return False
        return None

    def approval_blockers(row):
        blockers = []
        live_status = str(row.get("live_status", "")).strip()
        strategy_local = str(row.get("strategy", "")).strip()
        track = normalize_track(str(row.get("track", "")), strategy_local)
        live_bad_status = live_status in {
            "chain_error",
            "chain_not_success",
            "bad_occ_symbol",
            "missing_leg_in_live_chain",
            "missing_underlying_quote",
            "invalid_entry_structure",
            "missing_live_quote",
        }
        if live_bad_status:
            blockers.append(f"live_status:{live_status or 'unknown'}")
            return blockers

        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        ok_live = ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
        if not ok_live:
            blockers.append("live_entry_gate_fail")
        shield_live_quality_override = bool(
            shield_live_valid_overrides_quality and track == "SHIELD" and ok_live
        )

        if require_likelihood_pass and not shield_live_quality_override:
            verdict = str(row.get("verdict", "")).strip().upper()
            edge = fnum(row.get("edge_pct"))
            sig = fnum(row.get("signals"))
            if verdict != "PASS":
                blockers.append(f"likelihood_verdict:{verdict or 'UNKNOWN'}")
            if np.isfinite(min_edge_pct) and (not np.isfinite(edge) or edge < min_edge_pct):
                blockers.append(f"edge_below:{edge if np.isfinite(edge) else 'nan'}<{min_edge_pct}")
            if np.isfinite(min_signals) and min_signals > 0 and (not np.isfinite(sig) or sig < min_signals):
                blockers.append(f"signals_below:{sig if np.isfinite(sig) else 'nan'}<{min_signals}")

        strength = str(row.get("likelihood_strength", "")).strip()
        strength_rank = rank_likelihood_strength(strength)
        min_strength_rank = rank_likelihood_strength(min_likelihood_strength)
        if (
            (not shield_live_quality_override)
            and min_likelihood_strength
            and min_strength_rank >= 0
            and strength_rank >= 0
            and strength_rank < min_strength_rank
        ):
            blockers.append(f"likelihood_strength_below:{strength}<{min_likelihood_strength}")
        if (
            (not shield_live_quality_override)
            and min_likelihood_strength
            and min_strength_rank >= 0
            and strength_rank < 0
        ):
            blockers.append(f"likelihood_strength_unranked:{strength or 'N/A'}")
        if (not shield_live_quality_override) and str(strength).strip().upper() in disallow_likelihood_strengths:
            blockers.append(f"likelihood_strength_blocked:{strength}")

        invalidated_effective = (
            bool(row.get("invalidation_breached_effective"))
            if pd.notna(row.get("invalidation_breached_effective"))
            else False
        )
        if (require_invalidation_clear or block_invalidation_warning) and invalidated_effective:
            blockers.append("invalidation_warning")

        stage1_effective = bool(row.get("stage1_effective")) if pd.notna(row.get("stage1_effective")) else False
        if not stage1_effective and not shield_live_quality_override:
            blockers.append("stage1_not_actionable")

        if require_spot_alignment:
            spot_asof = fnum(row.get("spot_asof_close"))
            spot_live = fnum(row.get("spot_live_effective"))
            if not np.isfinite(spot_live):
                if spot_alignment_require_live:
                    blockers.append("spot_live_missing")
            elif np.isfinite(spot_asof) and spot_asof > 0:
                drift = fnum(row.get("spot_asof_live_drift_pct"))
                if np.isfinite(drift) and drift > max_spot_asof_drift_pct:
                    blockers.append(f"spot_drift:{drift:.2%}>{max_spot_asof_drift_pct:.2%}")

        long_strike = fnum(row.get("long_strike"))
        spot_ref = fnum(row.get("spot_live_effective"))
        if not np.isfinite(spot_ref) or spot_ref <= 0:
            spot_ref = fnum(row.get("spot_asof_close"))
        if np.isfinite(spot_ref) and spot_ref > 0 and np.isfinite(long_strike):
            if strategy_local == "Bull Call Debit" and np.isfinite(max_bull_call_long_otm_pct):
                long_otm = (long_strike / spot_ref) - 1.0
                if long_otm > max_bull_call_long_otm_pct:
                    blockers.append(
                        f"bull_call_otm_too_far:{long_otm:.1%}>{max_bull_call_long_otm_pct:.1%}"
                    )
            elif strategy_local == "Bear Put Debit" and np.isfinite(max_bear_put_long_otm_pct):
                long_otm = 1.0 - (long_strike / spot_ref)
                if long_otm > max_bear_put_long_otm_pct:
                    blockers.append(
                        f"bear_put_otm_too_far:{long_otm:.1%}>{max_bear_put_long_otm_pct:.1%}"
                    )

        confidence_tier = str(row.get("confidence_tier", "")).strip().upper()
        if (
            confidence_tier
            and confidence_tier in disallow_confidence_tiers
            and not shield_live_quality_override
        ):
            blockers.append(f"confidence_tier_blocked:{confidence_tier}")

        if track == "SHIELD" and not shield_live_quality_override:
            if require_shield_sigma_pass:
                sigma_stage1 = bool_or_none(row.get("sigma_pass_stage1"))
                if sigma_stage1 is None:
                    if shield_sigma_require_data:
                        blockers.append("shield_sigma_unknown")
                elif sigma_stage1 is False:
                    blockers.append("shield_sigma_fail")
            if require_shield_core:
                core_stage1 = bool_or_none(row.get("core_ok_stage1"))
                if core_stage1 is not True:
                    blockers.append("shield_core_fail")
            if require_live_shield_short_delta:
                if strategy in {"Bull Put Credit", "Bear Call Credit"}:
                    short_delta = fnum(row.get("short_delta_live"))
                    if not np.isfinite(short_delta):
                        blockers.append("shield_delta_missing")
                    elif abs(short_delta) > max_abs_short_delta_shield:
                        blockers.append(f"shield_delta_fail:{short_delta:+.2f}")
                elif strategy in {"Iron Condor", "Iron Butterfly"}:
                    put_delta = fnum(row.get("short_put_delta_live"))
                    call_delta = fnum(row.get("short_call_delta_live"))
                    if not np.isfinite(put_delta) or not np.isfinite(call_delta):
                        blockers.append("shield_delta_missing")
                    elif abs(put_delta) > max_abs_short_delta_shield or abs(call_delta) > max_abs_short_delta_shield:
                        blockers.append(f"shield_delta_fail:put={put_delta:+.2f},call={call_delta:+.2f}")

            if np.isfinite(min_credit_no_touch_pct) and min_credit_no_touch_pct > 0:
                no_touch = fnum(row.get("credit_no_touch_pct"))
                if np.isfinite(no_touch):
                    if no_touch < min_credit_no_touch_pct:
                        blockers.append(
                            f"credit_no_touch_below:{no_touch:.2f}<{min_credit_no_touch_pct:.2f}"
                        )
                elif credit_no_touch_require_data:
                    blockers.append("credit_no_touch_unknown")

        return blockers

    mdf["approval_blockers"] = mdf.apply(
        lambda row: ";".join(approval_blockers(row)),
        axis=1,
    )
    mdf["approved"] = mdf["approval_blockers"].astype(str).str.len().eq(0)
    mdf["_edge_sort"] = pd.to_numeric(mdf.get("edge_pct"), errors="coerce").fillna(-1e9)
    mdf = (
        mdf.sort_values(["approved", "_ev_sort", "_edge_sort", "conviction"], ascending=[False, False, False, False])
        .drop(columns=["_ev_sort", "_edge_sort"])
        .reset_index(drop=True)
    )
    mdf["portfolio_cap_pass"] = pd.Series([pd.NA] * len(mdf), dtype="boolean")
    mdf["portfolio_cap_reason"] = ""
    portfolio_guard_status = "disabled"
    portfolio_guard_error = ""
    portfolio_guard_snapshot_csv = ""
    portfolio_guard_base = {}
    if enforce_pretrade_caps:
        portfolio_guard_status = "enabled"
        open_pos_csv = None
        if pretrade_open_positions_csv:
            candidate = Path(pretrade_open_positions_csv).expanduser().resolve()
            if candidate.exists():
                open_pos_csv = candidate
            else:
                portfolio_guard_error = f"configured_open_positions_csv_missing:{candidate}"
        if open_pos_csv is None:
            cache_csv = out_dir / "open_positions_from_schwab.csv"
            ok_fetch, fetch_err = fetch_open_positions_from_schwab(cache_csv)
            if ok_fetch and cache_csv.exists():
                open_pos_csv = cache_csv
            else:
                if portfolio_guard_error:
                    portfolio_guard_error += f" | {fetch_err}"
                else:
                    portfolio_guard_error = fetch_err
        if open_pos_csv is not None and open_pos_csv.exists():
            portfolio_guard_snapshot_csv = str(open_pos_csv)
            book = build_portfolio_risk_book(open_pos_csv)
            if not bool(book.get("ok")):
                portfolio_guard_error = str(book.get("error", "portfolio_risk_book_failed"))
            else:
                portfolio_guard_base = dict(book)
                total_risk = float(book.get("total_risk", 0.0))
                short_put_risk = float(book.get("short_put_risk", 0.0))
                symbol_risk = {str(k): float(v) for k, v in (book.get("symbol_risk", {}) or {}).items()}
                short_put_expiry_risk = {
                    str(k): float(v) for k, v in (book.get("short_put_expiry_risk", {}) or {}).items()
                }
                for idx, row in mdf.iterrows():
                    if not bool(row.get("approved")):
                        continue
                    strategy = str(row.get("strategy", "")).strip()
                    ticker = str(row.get("ticker", "")).strip().upper()
                    expiry = str(row.get("expiry", ""))[:10]
                    add_risk = fnum(row.get("live_max_loss"))
                    if not np.isfinite(add_risk) or add_risk <= 0:
                        gate_val = fnum(row.get("gate_target"))
                        width_val = fnum(row.get("width"))
                        net_type_val = str(row.get("net_type", "")).strip().lower()
                        if np.isfinite(gate_val) and np.isfinite(width_val):
                            _, tgt_max_loss = calc_target_max(net_type_val, width_val, gate_val)
                            add_risk = fnum(tgt_max_loss)
                    if not np.isfinite(add_risk) or add_risk <= 0:
                        mdf.at[idx, "approved"] = False
                        mdf.at[idx, "portfolio_cap_pass"] = False
                        mdf.at[idx, "portfolio_cap_reason"] = "missing_trade_risk"
                        continue

                    projected_total = total_risk + add_risk
                    projected_symbol = symbol_risk.get(ticker, 0.0) + add_risk
                    projected_symbol_share = (
                        projected_symbol / projected_total if projected_total > 0 else 0.0
                    )
                    reasons = []
                    if projected_symbol_share > symbol_limit:
                        reasons.append(
                            f"symbol_share {projected_symbol_share:.1%} > {symbol_limit:.1%} ({ticker})"
                        )

                    if candidate_uses_short_put_risk(strategy):
                        projected_short_put = short_put_risk + add_risk
                        projected_short_put_share = (
                            projected_short_put / projected_total if projected_total > 0 else 0.0
                        )
                        if projected_short_put_share > short_put_limit:
                            reasons.append(
                                f"short_put_share {projected_short_put_share:.1%} > {short_put_limit:.1%}"
                            )
                        projected_expiry = short_put_expiry_risk.get(expiry, 0.0) + add_risk
                        projected_expiry_share = (
                            projected_expiry / projected_short_put if projected_short_put > 0 else 0.0
                        )
                        if projected_expiry_share > expiry_limit:
                            reasons.append(
                                f"short_put_expiry_share {projected_expiry_share:.1%} > {expiry_limit:.1%} ({expiry})"
                            )

                    if reasons:
                        mdf.at[idx, "approved"] = False
                        mdf.at[idx, "portfolio_cap_pass"] = False
                        mdf.at[idx, "portfolio_cap_reason"] = "; ".join(reasons)
                    else:
                        mdf.at[idx, "portfolio_cap_pass"] = True
                        total_risk = projected_total
                        symbol_risk[ticker] = projected_symbol
                        if candidate_uses_short_put_risk(strategy):
                            short_put_risk = short_put_risk + add_risk
                            short_put_expiry_risk[expiry] = short_put_expiry_risk.get(expiry, 0.0) + add_risk
        if not portfolio_guard_snapshot_csv and pretrade_caps_require_data:
            mdf.loc[mdf["approved"] == True, "approved"] = False
            mdf.loc[mdf["portfolio_cap_reason"].astype(str).eq(""), "portfolio_cap_reason"] = (
                "pretrade_caps_data_unavailable"
            )
            mdf.loc[mdf["portfolio_cap_pass"].isna(), "portfolio_cap_pass"] = False
    if exclude_debit_moneyness_fail_from_output and not mdf.empty:
        moneyness_fail_mask = mdf["approval_blockers"].astype(str).str.contains(
            r"bull_call_otm_too_far|bear_put_otm_too_far",
            case=False,
            regex=True,
        )
        if moneyness_fail_mask.any():
            mdf = mdf.loc[~moneyness_fail_mask].copy()
    merged_rows_pre_filter = int(len(mdf))
    dropped_final = []
    kept_indices = []
    per_ticker_final = defaultdict(int)
    for idx, row in mdf.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if per_ticker_final[ticker] >= final_max_per_ticker:
            dropped_final.append(
                {
                    "ticker": ticker,
                    "strategy": str(row.get("strategy", "")),
                    "expiry": str(row.get("expiry", ""))[:10],
                    "stage": "final",
                    "drop_reason": "final_max_per_ticker_cap",
                    "details": f"cap={final_max_per_ticker}",
                }
            )
            continue
        per_ticker_final[ticker] += 1
        kept_indices.append(idx)
    mdf = mdf.loc[kept_indices].reset_index(drop=True)
    if len(mdf) > int(args.top_trades):
        for _, row in mdf.iloc[int(args.top_trades) :].iterrows():
            dropped_final.append(
                {
                    "ticker": str(row.get("ticker", "")).strip().upper(),
                    "strategy": str(row.get("strategy", "")),
                    "expiry": str(row.get("expiry", ""))[:10],
                    "stage": "final",
                    "drop_reason": "final_top_limit",
                    "details": f"top={int(args.top_trades)}",
                }
            )
        mdf = mdf.head(int(args.top_trades)).reset_index(drop=True)
    inv_close_confirms = fnum(approval_cfg.get("invalidation_close_confirmations", 2))
    inv_close_confirms = int(inv_close_confirms) if np.isfinite(inv_close_confirms) and inv_close_confirms >= 1 else 2

    out_rows = []
    for i, r in mdf.iterrows():
        approved = bool(r["approved"])
        strategy = str(r["strategy"])
        net_type = str(r["net_type"]).lower()
        watch_reason_flags = []
        op, gate_val, _ = parse_gate_value(r.get("entry_gate", ""))

        live_net = fnum(r.get("live_net_bid_ask"))
        if np.isfinite(live_net):
            net_txt = f"{'Credit' if net_type == 'credit' else 'Debit'} {live_net:.2f} (Target {r['entry_gate']})"
            max_profit = money(r.get("live_max_profit"))
            max_loss = money(r.get("live_max_loss"))
            be_txt = calc_be_text(r, live_net)
        elif gate_val is not None:
            tgt_max_p, tgt_max_l = calc_target_max(net_type, float(r["width"]), gate_val)
            net_txt = f"Target {r['entry_gate']}"
            max_profit = money(tgt_max_p)
            max_loss = money(tgt_max_l)
            be_txt = calc_be_text(r, gate_val)
        else:
            net_txt = "N/A"
            max_profit = "N/A"
            max_loss = "N/A"
            be_txt = "N/A"

        if approved:
            confidence_tier = str(r.get("confidence_tier", ""))
            optimal = str(r.get("optimal_stage1", ""))
            stage1_promoted = bool(r.get("stage1_promoted")) if pd.notna(r.get("stage1_promoted")) else False
            if stage1_promoted and optimal == "Watch Only":
                optimal = "Yes-Good (Promoted)"
                if confidence_tier:
                    confidence_tier = f"{confidence_tier} (Promoted)"
                else:
                    confidence_tier = "Promoted"

            gate_target_now = fnum(r.get("gate_target"))
            gate_live_now = fnum(r.get("gate_live_net"))
            gate_near_miss = bool(r.get("gate_near_miss")) if pd.notna(r.get("gate_near_miss")) else False
            gate_tol_now = fnum(r.get("gate_tol_total"))
            gate_miss_now = fnum(r.get("gate_miss_abs"))
            if np.isfinite(gate_target_now) and np.isfinite(gate_live_now):
                if net_type == "debit":
                    gate_dir = "<="
                else:
                    gate_dir = ">="
                if gate_near_miss:
                    gate_text = (
                        f"near-miss accepted (target {gate_dir} {gate_target_now:.2f}, live {gate_live_now:.2f}, "
                        f"miss {gate_miss_now:.2f}, tol {gate_tol_now:.2f})"
                    )
                else:
                    gate_text = f"PASS (target {gate_dir} {gate_target_now:.2f}, live {gate_live_now:.2f})"
            else:
                gate_text = f"PASS ({r.get('entry_gate')})"

            if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
                notes = (
                    f"Live executable; gate {gate_text}; "
                    f"put short BID/ASK {r.get('short_put_bid_live')}/{r.get('short_put_ask_live')}, "
                    f"put long BID/ASK {r.get('long_put_bid_live')}/{r.get('long_put_ask_live')}, "
                    f"call short BID/ASK {r.get('short_call_bid_live')}/{r.get('short_call_ask_live')}, "
                    f"call long BID/ASK {r.get('long_call_bid_live')}/{r.get('long_call_ask_live')}."
                )
            else:
                notes = (
                    f"Live executable; gate {gate_text}; short BID/ASK "
                    f"{r.get('short_bid_live')}/{r.get('short_ask_live')}, long BID/ASK "
                    f"{r.get('long_bid_live')}/{r.get('long_ask_live')}."
                )
            if net_type == "credit":
                width_eff = fnum(r.get("width"))
                put_w = fnum(r.get("put_width"))
                call_w = fnum(r.get("call_width"))
                if strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
                    candidate_w = [w for w in (put_w, call_w, width_eff) if np.isfinite(w) and w > 0]
                    width_eff = max(candidate_w) if candidate_w else math.nan
                if np.isfinite(width_eff) and width_eff > 0:
                    ideal_low = width_eff * ideal_credit_low_pct
                    ideal_high = width_eff * ideal_credit_high_pct
                    notes += (
                        f" Ideal credit guide: {ideal_low:.2f}-{ideal_high:.2f} "
                        f"({ideal_credit_low_pct:.0%}-{ideal_credit_high_pct:.0%} of {width_eff:.2f}w)."
                    )
            spot_asof = fnum(r.get("spot_asof_close"))
            spot_live = fnum(r.get("spot_live_effective"))
            spot_drift = fnum(r.get("spot_asof_live_drift_pct"))
            if np.isfinite(spot_asof) and np.isfinite(spot_live):
                if np.isfinite(spot_drift):
                    notes += (
                        f" Spot check asof/live: {spot_asof:.2f}/{spot_live:.2f} "
                        f"(drift {spot_drift:.1%})."
                    )
                else:
                    notes += f" Spot check asof/live: {spot_asof:.2f}/{spot_live:.2f}."
            if stage1_promoted:
                notes += " Stage-1 Watch was promoted by PASS likelihood + edge/conviction thresholds."
            restrike_reason = str(r.get("restrike_reason", "")).strip()
            if restrike_reason:
                notes += " Stage-2 restrike optimizer selected this executable strike from the same family."
            invalidated_effective = (
                bool(r.get("invalidation_breached_effective"))
                if pd.notna(r.get("invalidation_breached_effective"))
                else False
            )
            if invalidated_effective:
                inv_text = str(r.get("invalidation", "")).strip() or "invalidation rule not available"
                lvl = fnum(r.get("invalidation_rule_level"))
                px_eval = fnum(r.get("invalidation_eval_price_effective"))
                px_source = str(r.get("invalidation_eval_source", "live")).strip() or "live"
                lvl_txt = f"{lvl:.2f}" if np.isfinite(lvl) else "n/a"
                px_txt = f"{px_eval:.2f}" if np.isfinite(px_eval) else "n/a"
                notes += (
                    f" Invalidation warning only (spot check): breached ({inv_text}; level={lvl_txt}; {px_source}={px_txt}). "
                    f"Action trigger is close-confirmed: require {inv_close_confirms} daily close(s) beyond level."
                )
        else:
            confidence_tier = "Watch Only"
            optimal = "Watch Only"
            stage1_blocked = bool(r.get("stage1_blocked")) if pd.notna(r.get("stage1_blocked")) else False
            if stage1_blocked:
                watch_reason_flags.append("stage1_conviction_watch")
            blocker_items = [x for x in str(r.get("approval_blockers", "")).split(";") if str(x).strip()]
            for blk in blocker_items:
                b = str(blk).strip()
                if b.startswith("likelihood_"):
                    watch_reason_flags.append("likelihood_fail")
                elif b.startswith("edge_below"):
                    watch_reason_flags.append("edge_below_threshold")
                elif b.startswith("signals_below"):
                    watch_reason_flags.append("sample_below_threshold")
                elif b.startswith("invalidation_warning"):
                    watch_reason_flags.append("invalidation_warning")
                elif b.startswith("shield_sigma"):
                    watch_reason_flags.append("shield_sigma_fail")
                elif b.startswith("credit_no_touch"):
                    watch_reason_flags.append("credit_path_risk_fail")
                elif b.startswith("shield_core"):
                    watch_reason_flags.append("shield_core_fail")
                elif b.startswith("shield_delta"):
                    watch_reason_flags.append("shield_delta_fail")
                elif b.startswith("confidence_tier_blocked"):
                    watch_reason_flags.append("confidence_tier_blocked")
                elif b.startswith("live_entry_gate_fail") or b.startswith("live_status:"):
                    watch_reason_flags.append("live_entry_gate_miss")
                elif b.startswith("spot_drift") or b.startswith("spot_live_missing"):
                    watch_reason_flags.append("spot_data_mismatch")
                elif b.startswith("bull_call_otm_too_far") or b.startswith("bear_put_otm_too_far"):
                    watch_reason_flags.append("debit_moneyness_fail")
                elif b.startswith("stage1_not_actionable"):
                    watch_reason_flags.append("stage1_conviction_watch")
                else:
                    watch_reason_flags.append("other_watch")
            cur_txt = f"{live_net:.2f}" if np.isfinite(live_net) else "N/A"
            live_status = str(r.get("live_status", "missing"))
            live_valid_raw = bool(r.get("is_final_live_valid")) if pd.notna(r.get("is_final_live_valid")) else False
            gate_pass_live = bool(r.get("gate_pass_live")) if pd.notna(r.get("gate_pass_live")) else False
            gate_pass_effective = bool(r.get("gate_pass_effective")) if pd.notna(r.get("gate_pass_effective")) else False
            gate_near_miss = bool(r.get("gate_near_miss")) if pd.notna(r.get("gate_near_miss")) else False
            gate_tol_now = fnum(r.get("gate_tol_total"))
            gate_miss_now = fnum(r.get("gate_miss_abs"))
            live_valid_effective = live_valid_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
            verdict_now = str(r.get("verdict", "")).strip().upper()
            edge_now = fnum(r.get("edge_pct"))
            sig_now = fnum(r.get("signals"))
            min_edge_req = fnum(approval_cfg.get("min_edge_pct", 0.0))
            min_sig_req = fnum(approval_cfg.get("min_signals", 100))
            require_lk = bool(approval_cfg.get("require_likelihood_pass", True))
            invalidated_effective = (
                bool(r.get("invalidation_breached_effective"))
                if pd.notna(r.get("invalidation_breached_effective"))
                else False
            )
            if (not live_valid_effective) and live_status == "fails_live_entry_gate":
                watch_reason_flags.append("live_entry_gate_miss")
            if invalidated_effective:
                watch_reason_flags.append("invalidation_warning")
                inv_text = str(r.get("invalidation", "")).strip() or "invalidation rule not available"
                lvl = fnum(r.get("invalidation_rule_level"))
                px_eval = fnum(r.get("invalidation_eval_price_effective"))
                px_source = str(r.get("invalidation_eval_source", "live")).strip() or "live"
                lvl_txt = f"{lvl:.2f}" if np.isfinite(lvl) else "n/a"
                px_txt = f"{px_eval:.2f}" if np.isfinite(px_eval) else "n/a"
                notes = (
                    f"Watch Only: live_status={live_status}; invalidation warning only "
                    f"(rule: {inv_text}; level={lvl_txt}; {px_source}={px_txt}); "
                    f"close-confirm policy requires {inv_close_confirms} daily close(s) beyond level."
                )
                extra_blockers = [b for b in blocker_items if not str(b).startswith("invalidation_warning")]
                if extra_blockers:
                    notes += " Additional blockers: " + ", ".join(extra_blockers) + "."
            else:
                reasons = []
                if blocker_items:
                    reasons.append("approval blockers: " + ", ".join(blocker_items))
                if not live_valid_effective:
                    if live_status == "invalid_entry_structure":
                        structure_reason = str(r.get("entry_structure_reason_live", "")).strip()
                        if structure_reason:
                            reasons.append(f"invalid structure ({structure_reason})")
                        else:
                            reasons.append("invalid structure for current spot")
                        watch_reason_flags.append("invalid_entry_structure")
                    elif live_status == "missing_underlying_quote":
                        reasons.append("missing underlying quote for live structure check")
                        watch_reason_flags.append("missing_underlying_quote")
                    else:
                        reasons.append(f"live_status={live_status}")
                portfolio_cap_reason = str(r.get("portfolio_cap_reason", "")).strip()
                if portfolio_cap_reason:
                    reasons.append(f"portfolio cap breach ({portfolio_cap_reason})")
                    watch_reason_flags.append("portfolio_cap_breach")
                if np.isfinite(live_net) and gate_val is not None and not gate_pass_effective:
                    if net_type == "debit":
                        if np.isfinite(gate_miss_now) and np.isfinite(gate_tol_now):
                            reasons.append(
                                f"entry gate miss (need debit <= {gate_val:.2f}, live {cur_txt}, miss {gate_miss_now:.2f} > tol {gate_tol_now:.2f})"
                            )
                        else:
                            reasons.append(f"entry gate miss (need debit <= {gate_val:.2f}, live {cur_txt})")
                    else:
                        if np.isfinite(gate_miss_now) and np.isfinite(gate_tol_now):
                            reasons.append(
                                f"entry gate miss (need credit >= {gate_val:.2f}, live {cur_txt}, miss {gate_miss_now:.2f} > tol {gate_tol_now:.2f})"
                            )
                        else:
                            reasons.append(f"entry gate miss (need credit >= {gate_val:.2f}, live {cur_txt})")
                    watch_reason_flags.append("live_entry_gate_miss")
                elif np.isfinite(live_net) and gate_val is not None and gate_near_miss:
                    reasons.append(
                        f"entry near-miss tolerated (target {r.get('entry_gate')}, live {cur_txt}, tol {gate_tol_now:.2f})"
                    )
                if stage1_blocked:
                    stage1_eval = str(r.get("stage1_eval_reason", "")).strip() or "stage1_watch_blocked"
                    reasons.append(f"stage1 blocked ({stage1_eval})")
                restrike_reason = str(r.get("restrike_reason", "")).strip()
                if restrike_reason:
                    reasons.append("stage-2 restrike optimizer selected nearest executable family strike")
                if require_lk and verdict_now != "PASS":
                    if np.isfinite(edge_now):
                        reasons.append(f"likelihood {verdict_now or 'N/A'} (edge {edge_now:+.1f}%)")
                    else:
                        reasons.append(f"likelihood {verdict_now or 'N/A'}")
                    watch_reason_flags.append("likelihood_fail")
                if verdict_now == "PASS":
                    if np.isfinite(min_edge_req) and np.isfinite(edge_now) and edge_now < min_edge_req:
                        reasons.append(f"edge below threshold ({edge_now:+.1f}% < {min_edge_req:.1f}%)")
                        watch_reason_flags.append("edge_below_threshold")
                    if np.isfinite(min_sig_req) and np.isfinite(sig_now) and sig_now < min_sig_req:
                        reasons.append(f"sample below threshold ({int(sig_now)} < {int(min_sig_req)})")
                        watch_reason_flags.append("sample_below_threshold")
                if not reasons:
                    reasons.append(f"target {r.get('entry_gate', 'N/A')}; current live net={cur_txt}")
                    watch_reason_flags.append("other_watch")
                notes = "Watch Only: " + "; ".join(reasons) + "."
        if not approved and not watch_reason_flags:
            watch_reason_flags.append("other_watch")
        watch_reason_flags = sorted(set(watch_reason_flags))

        hist_success = fnum(r.get("hist_success_pct"))
        edge_pct = fnum(r.get("edge_pct"))
        signals = fnum(r.get("signals"))
        verdict = str(r.get("verdict", "")).strip().upper()
        strength = likelihood_strength(verdict, edge_pct, signals)
        if np.isfinite(hist_success):
            n_txt = f"{int(signals)}" if np.isfinite(signals) else "n/a"
            verdict_txt = verdict if verdict else "N/A"
            setup_likelihood = f"{hist_success:.1f}% {verdict_txt} ({strength}, edge {edge_pct:+.1f}%, n={n_txt})"
        else:
            if verdict == "UNKNOWN":
                setup_likelihood = "Unknown"
            else:
                setup_likelihood = "N/A"

        out_rows.append(
            {
                "#": i + 1,
                "Category": f"{'Approved' if approved else 'Watch Only'} - {normalize_track(r.get('track', ''), strategy)}",
                "Ticker": r["ticker"],
                "Action": action_cell(strategy, str(r.get("track", "")), optimal),
                "Strategy Type": strategy,
                "Strike Setup": strike_setup(
                    strategy,
                    r["long_strike"],
                    r["short_strike"],
                    r["width"],
                    long_put_strike=r.get("long_put_strike"),
                    short_put_strike=r.get("short_put_strike"),
                    short_call_strike=r.get("short_call_strike"),
                    long_call_strike=r.get("long_call_strike"),
                ),
                "Expiry": str(r["expiry"])[:10],
                "DTE": (dt.datetime.strptime(str(r["expiry"])[:10], "%Y-%m-%d").date() - asof).days,
                "Net Credit/Debit": net_txt,
                "Max Profit": max_profit,
                "Max Loss": max_loss,
                "Breakeven": be_txt,
                "Conviction %": f"{int(r['conviction'])}%",
                "Setup Likelihood": setup_likelihood,
                "Confidence Tier": confidence_tier,
                "Optimal": optimal,
                "Watch Reason Flags": ", ".join(watch_reason_flags) if not approved else "",
                "Notes": notes,
                "Source": "Stage1(ChainOI+DP+HotChains+Screener+Whale) + Stage2(uwos.pricer)",
            }
        )

    out_df = pd.DataFrame(
        out_rows,
        columns=[
            "#",
            "Category",
            "Ticker",
            "Action",
            "Strategy Type",
            "Strike Setup",
            "Expiry",
            "DTE",
            "Net Credit/Debit",
            "Max Profit",
            "Max Loss",
            "Breakeven",
            "Conviction %",
            "Setup Likelihood",
            "Confidence Tier",
            "Optimal",
            "Watch Reason Flags",
            "Notes",
            "Source",
        ],
    )
    cat_rank = {
        "Approved - FIRE": 0,
        "Approved - SHIELD": 1,
        "Watch Only - FIRE": 2,
        "Watch Only - SHIELD": 3,
        "Watch Only - UNKNOWN": 4,
        "Approved - UNKNOWN": 5,
    }
    out_df["_cat_rank"] = out_df["Category"].map(cat_rank).fillna(99).astype(int)
    out_df = out_df.sort_values(["_cat_rank", "#"], ascending=[True, True]).drop(columns=["_cat_rank"]).reset_index(drop=True)
    out_df["#"] = range(1, len(out_df) + 1)
    approved_count = int(mdf["approved"].sum()) if "approved" in mdf.columns else 0
    dropped_csv = out_dir / f"dropped_trades_{asof_str}.csv"
    dropped_rows = []
    for rec in dropped_stage1:
        dropped_rows.append(
            {
                "ticker": str(rec.get("ticker", "")),
                "strategy": str(rec.get("strategy", "")),
                "expiry": str(rec.get("expiry", ""))[:10],
                "stage": str(rec.get("stage", "stage1")),
                "drop_reason": str(rec.get("drop_reason", "unknown")),
                "details": str(rec.get("details", "")),
            }
        )
    for rec in dropped_final:
        dropped_rows.append(
            {
                "ticker": str(rec.get("ticker", "")),
                "strategy": str(rec.get("strategy", "")),
                "expiry": str(rec.get("expiry", ""))[:10],
                "stage": str(rec.get("stage", "final")),
                "drop_reason": str(rec.get("drop_reason", "unknown")),
                "details": str(rec.get("details", "")),
            }
        )
    dropped_df = pd.DataFrame(
        dropped_rows,
        columns=["ticker", "strategy", "expiry", "stage", "drop_reason", "details"],
    )
    dropped_df.to_csv(dropped_csv, index=False)

    manifest_path = out_dir / f"run_manifest_{asof_str}.json"
    category_order = [
        "Approved - FIRE",
        "Approved - SHIELD",
        "Watch Only - FIRE",
        "Watch Only - SHIELD",
        "Watch Only - UNKNOWN",
        "Approved - UNKNOWN",
    ]
    table_cols = [
        c
        for c in [
            "#",
            "Ticker",
            "Action",
            "Strategy Type",
            "Strike Setup",
            "Expiry",
            "DTE",
            "Net Credit/Debit",
            "Max Profit",
            "Max Loss",
            "Breakeven",
            "Conviction %",
            "Setup Likelihood",
            "Confidence Tier",
            "Optimal",
            "Notes",
            "Source",
        ]
        if c in out_df.columns
    ]
    mini_tables = []
    for cat in category_order:
        sub = out_df[out_df["Category"] == cat].copy()
        if sub.empty:
            continue
        mini_tables.extend(
            [
                f"### {cat}",
                sub[table_cols].to_markdown(index=False),
                "",
            ]
        )
    if not mini_tables:
        mini_tables = ["_No rows_", ""]
    watch_reason_order = [
        ("stage1_conviction_watch", "Stage-1 Conviction Watch"),
        ("portfolio_cap_breach", "Portfolio Cap Breach"),
        ("invalid_entry_structure", "Invalid Entry Structure"),
        ("missing_underlying_quote", "Missing Underlying Quote"),
        ("live_entry_gate_miss", "Live Entry Gate Miss"),
        ("invalidation_warning", "Invalidation Warning (Close-Confirm)"),
        ("likelihood_fail", "Likelihood Fail"),
        ("shield_sigma_fail", "Shield Sigma Gate Fail"),
        ("credit_path_risk_fail", "Credit Path-Risk Fail"),
        ("shield_core_fail", "Shield Core Gate Fail"),
        ("shield_delta_fail", "Shield Delta Gate Fail"),
        ("confidence_tier_blocked", "Confidence Tier Blocked"),
        ("edge_below_threshold", "Edge Below Threshold"),
        ("sample_below_threshold", "Sample Below Threshold"),
        ("other_watch", "Other Watch Reason"),
    ]
    watch_reason_tables = []
    watch_df = out_df[out_df["Category"].astype(str).str.startswith("Watch Only")].copy()
    reason_cols = [
        c
        for c in [
            "#",
            "Ticker",
            "Strategy Type",
            "Expiry",
            "Conviction %",
            "Setup Likelihood",
            "Watch Reason Flags",
            "Notes",
        ]
        if c in out_df.columns
    ]
    for code, title in watch_reason_order:
        if watch_df.empty:
            break
        sub = watch_df[watch_df["Watch Reason Flags"].astype(str).str.contains(rf"\b{re.escape(code)}\b", regex=True)]
        if sub.empty:
            continue
        watch_reason_tables.extend(
            [
                f"### {title}",
                sub[reason_cols].to_markdown(index=False),
                "",
            ]
        )
    if not watch_reason_tables:
        watch_reason_tables = ["_No watch-only reason rows_", ""]

    lines = [
        f"As-of date used: {asof_str}",
        "Files used: "
        + ", ".join(
            [
                csvs["chain-oi-changes-"].name,
                csvs["dp-eod-report-"].name,
                csvs["hot-chains-"].name,
                csvs["stock-screener-"].name,
                whale_md.name,
                shortlist_csv.name,
                likelihood_csv.name,
                live_csv.name,
                live_final_csv.name,
                dropped_csv.name,
                manifest_path.name,
            ]
        ),
        (
            "Stage-2 note: reused existing same-date live outputs because live pricing refresh failed."
            if stage2_reused_existing
            else ""
        ),
        f"Approved trades: {approved_count} / {len(out_df)}",
        "Category split: "
        + ", ".join(
            [
                f"Approved-FIRE={int((out_df['Category'] == 'Approved - FIRE').sum())}",
                f"Approved-SHIELD={int((out_df['Category'] == 'Approved - SHIELD').sum())}",
                f"Watch-FIRE={int((out_df['Category'] == 'Watch Only - FIRE').sum())}",
                f"Watch-SHIELD={int((out_df['Category'] == 'Watch Only - SHIELD').sum())}",
            ]
        ),
        ("Important: NO ACTIONABLE TRADES passed live + likelihood gates."
         if approved_count == 0
         else ""),
        "",
        "## Anu Expert Trade Table",
        "Mini tables by category:",
        "",
        *mini_tables,
        "## Watch Only Reason Tables",
        "",
        *watch_reason_tables,
        "Ticker thesis + invalidation (Yes-Prime / Yes-Good):",
    ]

    seen = set()
    for _, r in mdf.iterrows():
        if not bool(r["approved"]):
            continue
        t = str(r["ticker"])
        if t in seen:
            continue
        seen.add(t)
        lines.append(f"- {t}: {str(r.get('thesis', '')).strip()} Invalidation: {str(r.get('invalidation', '')).strip()}")
    if not seen:
        lines.append("- none")

    output_path.write_text("\n".join(lines), encoding="utf-8-sig")
    run_completed_utc = dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "asof_date": asof_str,
        "run_started_utc": run_started_utc,
        "run_completed_utc": run_completed_utc,
        "git_commit": safe_git_commit(),
        "config_path": str(cfg_path),
        "config_sha256": sha256_file(cfg_path) if cfg_path.exists() else "",
        "base_dir": str(base),
        "out_dir": str(out_dir),
        "output_md": str(output_path),
        "input_files": {
            "chain_oi_changes_csv": str(csvs["chain-oi-changes-"]),
            "dp_eod_report_csv": str(csvs["dp-eod-report-"]),
            "hot_chains_csv": str(csvs["hot-chains-"]),
            "stock_screener_csv": str(csvs["stock-screener-"]),
            "whale_md": str(whale_md),
        },
        "artifacts": {
            "shortlist_csv": str(shortlist_csv),
            "likelihood_csv": str(likelihood_csv),
            "live_csv": str(live_csv),
            "live_final_csv": str(live_final_csv),
            "dropped_csv": str(dropped_csv),
            "manifest_json": str(manifest_path),
            "snapshot_json": str((out_dir / f"schwab_snapshot_{asof_str}.json").resolve()),
            "snapshot_chain_dir": str((out_dir / f"schwab_snapshot_{asof_str}" / "chains").resolve()),
        },
        "settings": {
            "top_trades_requested": int(args.top_trades),
            "discovery_multiplier": float(discovery_multiplier),
            "discovery_top": int(discovery_top),
            "final_max_per_ticker": int(final_max_per_ticker),
            "strict_stage2": bool(args.strict_stage2),
            "stage2_reused_existing_live": bool(stage2_reused_existing),
            "stage2_error": stage2_error,
            "enforce_pretrade_portfolio_caps": bool(enforce_pretrade_caps),
            "pretrade_caps_require_data": bool(pretrade_caps_require_data),
            "pretrade_caps_status": portfolio_guard_status,
            "pretrade_caps_error": portfolio_guard_error,
            "pretrade_caps_snapshot_csv": portfolio_guard_snapshot_csv,
        },
        "counts": {
            "stage1_candidates_raw": int(len(best)),
            "stage1_shortlist_rows": int(len(shortlist)),
            "stage1_dropped": int(len(dropped_stage1)),
            "stage2_live_rows": int(len(live)),
            "merged_rows": int(merged_rows_pre_filter),
            "rows_after_final_caps": int(len(mdf)),
            "final_output_rows": int(len(out_df)),
            "approved_rows": int(approved_count),
            "final_dropped": int(len(dropped_final)),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {output_path}")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {dropped_csv}")
    try:
        print("\n".join(lines))
    except UnicodeEncodeError:
        print("\n".join(lines).encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    run()




