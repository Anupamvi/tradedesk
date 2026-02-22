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
    if s == "Iron Condor":
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
    if strategy == "Iron Condor":
        lp = fnum(long_put_strike)
        sp = fnum(short_put_strike)
        sc = fnum(short_call_strike)
        lc = fnum(long_call_strike)
        if np.isfinite(lp) and np.isfinite(sp) and np.isfinite(sc) and np.isfinite(lc):
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
    if strategy == "Iron Condor":
        sp = fnum(row.get("short_put_strike", row.get("short_strike")))
        sc = fnum(row.get("short_call_strike"))
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


def normalize_track(track: str, strategy: str) -> str:
    t = str(track or "").strip().upper()
    if t in {"FIRE", "SHIELD"}:
        return t
    s = str(strategy or "").strip()
    if s in {"Bull Put Credit", "Bear Call Credit", "Iron Condor"}:
        return "SHIELD"
    if s in {"Bull Call Debit", "Bear Put Debit"}:
        return "FIRE"
    return "UNKNOWN"


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
        if s == "Iron Condor":
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

        if strategy == "Iron Condor":
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
        "100",
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
        "long_bid_live",
        "long_ask_live",
        "short_put_bid_live",
        "short_put_ask_live",
        "long_put_bid_live",
        "long_put_ask_live",
        "short_call_bid_live",
        "short_call_ask_live",
        "long_call_bid_live",
        "long_call_ask_live",
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

    require_likelihood_pass = bool(approval_cfg.get("require_likelihood_pass", True))
    min_edge_pct = fnum(approval_cfg.get("min_edge_pct", 0.0))
    min_signals = fnum(approval_cfg.get("min_signals", 100))
    require_invalidation_clear = bool(approval_cfg.get("require_invalidation_clear", False))
    allow_stage1_watch_promotion = bool(approval_cfg.get("allow_stage1_watch_promotion", True))
    stage1_promote_min_conv = fnum(approval_cfg.get("stage1_watch_promotion_min_conviction", 58))
    stage1_promote_min_edge = fnum(approval_cfg.get("stage1_watch_promotion_min_edge_pct", 5.0))
    stage1_promote_min_signals = fnum(approval_cfg.get("stage1_watch_promotion_min_signals", min_signals))
    entry_tol_debit_abs = fnum(approval_cfg.get("entry_tolerance_debit_abs", 0.20))
    entry_tol_credit_abs = fnum(approval_cfg.get("entry_tolerance_credit_abs", 0.20))
    entry_tol_pct = fnum(approval_cfg.get("entry_tolerance_pct", 0.05))
    if not np.isfinite(entry_tol_debit_abs) or entry_tol_debit_abs < 0:
        entry_tol_debit_abs = 0.0
    if not np.isfinite(entry_tol_credit_abs) or entry_tol_credit_abs < 0:
        entry_tol_credit_abs = 0.0
    if not np.isfinite(entry_tol_pct) or entry_tol_pct < 0:
        entry_tol_pct = 0.0

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

    def is_approved(row):
        live_status = str(row.get("live_status", "")).strip()
        live_bad_status = live_status in {
            "chain_error",
            "chain_not_success",
            "bad_occ_symbol",
            "missing_leg_in_live_chain",
            "missing_live_quote",
        }
        if live_bad_status:
            return False

        ok_live_raw = bool(row.get("is_final_live_valid")) if pd.notna(row.get("is_final_live_valid")) else False
        gate_pass_effective = bool(row.get("gate_pass_effective")) if pd.notna(row.get("gate_pass_effective")) else False
        ok_live = ok_live_raw or (live_status == "fails_live_entry_gate" and gate_pass_effective)
        if not ok_live:
            return False

        if require_likelihood_pass:
            verdict = str(row.get("verdict", "")).strip().upper()
            edge = fnum(row.get("edge_pct"))
            sig = fnum(row.get("signals"))
            if verdict != "PASS":
                return False
            if np.isfinite(min_edge_pct) and (not np.isfinite(edge) or edge < min_edge_pct):
                return False
            if np.isfinite(min_signals) and min_signals > 0 and (not np.isfinite(sig) or sig < min_signals):
                return False

        if require_invalidation_clear:
            invalidated_live = (
                bool(row.get("invalidation_breached_live"))
                if pd.notna(row.get("invalidation_breached_live"))
                else False
            )
            if invalidated_live:
                return False

        stage1_effective = bool(row.get("stage1_effective")) if pd.notna(row.get("stage1_effective")) else False
        if not stage1_effective:
            return False
        return True

    def profit_score(row):
        edge = fnum(row.get("edge_pct"))
        hs = fnum(row.get("hist_success_pct"))
        conv = fnum(row.get("conviction"))
        width = fnum(row.get("width"))
        net_type = str(row.get("net_type", "")).strip().lower()
        net = fnum(row.get("gate_live_net"))
        if not np.isfinite(net):
            net = fnum(row.get("gate_target"))
        rr = calc_reward_risk(net_type, width, net)
        score = 0.0
        if np.isfinite(edge):
            score += 1.0 * edge
        if np.isfinite(hs):
            score += 0.35 * (hs - 50.0)
        if np.isfinite(conv):
            score += 0.20 * (conv - 50.0)
        if np.isfinite(rr):
            score += 4.0 * min(3.0, rr)
        if bool(row.get("gate_near_miss")):
            score -= 0.25
        return score

    mdf["approved"] = mdf.apply(is_approved, axis=1)
    mdf["_profit_sort"] = mdf.apply(profit_score, axis=1)
    mdf["_edge_sort"] = pd.to_numeric(mdf.get("edge_pct"), errors="coerce").fillna(-1e9)
    mdf = (
        mdf.sort_values(["approved", "_profit_sort", "_edge_sort", "conviction"], ascending=[False, False, False, False])
        .drop(columns=["_profit_sort", "_edge_sort"])
        .reset_index(drop=True)
    )
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

            if strategy == "Iron Condor":
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
            if stage1_promoted:
                notes += " Stage-1 Watch was promoted by PASS likelihood + edge/conviction thresholds."
            invalidated_live = (
                bool(r.get("invalidation_breached_live"))
                if pd.notna(r.get("invalidation_breached_live"))
                else False
            )
            if invalidated_live:
                inv_text = str(r.get("invalidation", "")).strip() or "invalidation rule not available"
                lvl = fnum(r.get("invalidation_rule_level"))
                px_live = fnum(r.get("invalidation_eval_price_live"))
                lvl_txt = f"{lvl:.2f}" if np.isfinite(lvl) else "n/a"
                px_txt = f"{px_live:.2f}" if np.isfinite(px_live) else "n/a"
                notes += (
                    f" Invalidation warning only (spot check): breached ({inv_text}; level={lvl_txt}; live spot={px_txt}). "
                    f"Action trigger is close-confirmed: require {inv_close_confirms} daily close(s) beyond level."
                )
        else:
            confidence_tier = "Watch Only"
            optimal = "Watch Only"
            stage1_blocked = bool(r.get("stage1_blocked")) if pd.notna(r.get("stage1_blocked")) else False
            if stage1_blocked:
                watch_reason_flags.append("stage1_conviction_watch")
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
            invalidated_live = (
                bool(r.get("invalidation_breached_live"))
                if pd.notna(r.get("invalidation_breached_live"))
                else False
            )
            if (not live_valid_effective) and live_status == "fails_live_entry_gate":
                watch_reason_flags.append("live_entry_gate_miss")
            if invalidated_live:
                watch_reason_flags.append("invalidation_warning")
                inv_text = str(r.get("invalidation", "")).strip() or "invalidation rule not available"
                lvl = fnum(r.get("invalidation_rule_level"))
                px_live = fnum(r.get("invalidation_eval_price_live"))
                lvl_txt = f"{lvl:.2f}" if np.isfinite(lvl) else "n/a"
                px_txt = f"{px_live:.2f}" if np.isfinite(px_live) else "n/a"
                notes = (
                    f"Watch Only: live_status={live_status}; invalidation warning only "
                    f"(rule: {inv_text}; level={lvl_txt}; live spot={px_txt}); "
                    f"close-confirm policy requires {inv_close_confirms} daily close(s) beyond level."
                )
            else:
                reasons = []
                if not live_valid_effective:
                    reasons.append(f"live_status={live_status}")
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
        ("live_entry_gate_miss", "Live Entry Gate Miss"),
        ("invalidation_warning", "Invalidation Warning (Close-Confirm)"),
        ("likelihood_fail", "Likelihood Fail"),
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




