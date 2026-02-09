#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


ENTRY_GATE_RE = re.compile(r"^\s*(>=|<=)\s*([0-9]*\.?[0-9]+)\s*(cr|db)\s*$", re.IGNORECASE)
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def parse_entry_gate(value: object) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    m = ENTRY_GATE_RE.match(str(value or "").strip())
    if not m:
        return None, None, None
    op, threshold, unit = m.groups()
    try:
        return op, float(threshold), unit.lower()
    except Exception:
        return None, None, None


def infer_asof_from_path(path: Path) -> Optional[dt.date]:
    m = DATE_RE.search(str(path))
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%Y-%m-%d").date()
    except Exception:
        return None


def safe_float(x: object) -> float:
    try:
        if x is None or pd.isna(x):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def required_win_rate_pct(strategy: str, width: float, net: float) -> float:
    w = safe_float(width)
    n = safe_float(net)
    if not np.isfinite(w) or not np.isfinite(n) or w <= 0:
        return math.nan
    s = str(strategy).strip()
    if s in {"Bull Call Debit", "Bear Put Debit"}:
        max_profit = max(0.0, w - n)
        max_loss = max(0.0, n)
    else:
        max_profit = max(0.0, n)
        max_loss = max(0.0, w - n)
    denom = max_profit + max_loss
    if denom <= 0:
        return math.nan
    return 100.0 * (max_loss / denom)


def breakeven(strategy: str, long_strike: float, short_strike: float, net: float) -> float:
    s = str(strategy).strip()
    ls = safe_float(long_strike)
    ss = safe_float(short_strike)
    n = safe_float(net)
    if not np.isfinite(ls) or not np.isfinite(ss) or not np.isfinite(n):
        return math.nan
    if s == "Bull Call Debit":
        return ls + n
    if s == "Bear Put Debit":
        return ls - n
    if s == "Bull Put Credit":
        return ss - n
    if s == "Bear Call Credit":
        return ss + n
    return math.nan


def detect_default_screener(asof: dt.date, root_dir: Path) -> Optional[Path]:
    p = root_dir / asof.isoformat() / "_unzipped_mode_a" / f"stock-screener-{asof.isoformat()}.csv"
    return p if p.exists() else None


def load_spot_map(screener_csv: Optional[Path]) -> Dict[str, float]:
    if not screener_csv or not screener_csv.exists():
        return {}
    df = pd.read_csv(screener_csv, low_memory=False)
    needed = {"ticker", "close"}
    if not needed.issubset(df.columns):
        return {}
    out = (
        df.assign(ticker=df["ticker"].astype(str).str.upper().str.strip())
        .drop_duplicates("ticker")
        .set_index("ticker")["close"]
        .map(safe_float)
        .to_dict()
    )
    return {k: float(v) for k, v in out.items() if np.isfinite(v) and v > 0}


def yfinance_spot_on_or_before(ticker: str, asof: dt.date) -> float:
    start = (asof - dt.timedelta(days=12)).isoformat()
    end = (asof + dt.timedelta(days=1)).isoformat()
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    except Exception:
        return math.nan
    if df is None or df.empty:
        return math.nan
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    c = pd.to_numeric(df.get("Close"), errors="coerce").dropna()
    if c.empty:
        return math.nan
    return float(c.iloc[-1])


def get_price_history(ticker: str, asof: dt.date, lookback_years: float) -> pd.DataFrame:
    start = (asof - dt.timedelta(days=int(round(365.25 * float(lookback_years))))).isoformat()
    end = (asof + dt.timedelta(days=1)).isoformat()
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    except Exception:
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for c in ["Open", "High", "Low", "Close"]:
        if c not in df.columns:
            return pd.DataFrame()
    out = df[["Open", "High", "Low", "Close"]].copy()
    out = out.dropna(subset=["High", "Low", "Close"])
    return out


def simulate_setup(
    strategy: str,
    hist: pd.DataFrame,
    dte: int,
    spot_asof: float,
    short_strike: float,
    breakeven_level: float,
) -> Tuple[int, int, float]:
    s = str(strategy).strip()
    if dte <= 0 or hist.empty:
        return 0, 0, math.nan
    if not np.isfinite(spot_asof) or spot_asof <= 0:
        return 0, 0, math.nan
    if not np.isfinite(short_strike) or not np.isfinite(breakeven_level):
        return 0, 0, math.nan

    be_ratio = float(breakeven_level) / float(spot_asof)
    short_ratio = float(short_strike) / float(spot_asof)

    wins = 0
    signals = 0
    no_touch = 0
    no_touch_n = 0

    n = len(hist)
    for i in range(0, n - dte):
        entry = safe_float(hist["Close"].iloc[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        window = hist.iloc[i + 1 : i + 1 + dte]
        if len(window) < dte:
            continue
        end_close = safe_float(window["Close"].iloc[-1])
        hi = safe_float(window["High"].max())
        lo = safe_float(window["Low"].min())
        if not (np.isfinite(end_close) and np.isfinite(hi) and np.isfinite(lo)):
            continue

        sim_be = float(entry) * be_ratio
        sim_short = float(entry) * short_ratio
        signals += 1

        if s == "Bull Call Debit":
            win = end_close >= sim_be
        elif s == "Bear Put Debit":
            win = end_close <= sim_be
        elif s == "Bull Put Credit":
            win = end_close >= sim_be
            no_touch_n += 1
            if lo > sim_short:
                no_touch += 1
        elif s == "Bear Call Credit":
            win = end_close <= sim_be
            no_touch_n += 1
            if hi < sim_short:
                no_touch += 1
        else:
            win = False

        wins += int(bool(win))

    no_touch_pct = (100.0 * no_touch / no_touch_n) if no_touch_n > 0 else math.nan
    return signals, wins, no_touch_pct


def confidence_bucket(signals: int) -> str:
    n = int(signals)
    if n >= 220:
        return "High"
    if n >= 120:
        return "Medium"
    if n >= 60:
        return "Low"
    return "Very Low"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Estimate setup success/fail likelihood from historical analog windows.")
    ap.add_argument("--setups-csv", required=True, help="Input setups CSV (ticker/strategy/expiry/strikes/entry_gate).")
    ap.add_argument("--asof-date", default="", help="Signal as-of date YYYY-MM-DD. If empty, infer from setups.")
    ap.add_argument("--root-dir", default=r"c:\uw_root", help="Root dir to auto-detect screener close file.")
    ap.add_argument("--screener-csv", default="", help="Optional stock-screener CSV with ticker + close.")
    ap.add_argument("--lookback-years", type=float, default=2.0, help="Historical lookback length for simulation.")
    ap.add_argument("--min-signals", type=int, default=100, help="Minimum historical windows for stable inference.")
    ap.add_argument("--out-dir", default=r"c:\uw_root\out", help="Output directory.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    setups_csv = Path(args.setups_csv).expanduser().resolve()
    if not setups_csv.exists():
        raise FileNotFoundError(f"Missing setups CSV: {setups_csv}")

    if args.asof_date.strip():
        asof = dt.datetime.strptime(args.asof_date.strip(), "%Y-%m-%d").date()
    else:
        inferred = infer_asof_from_path(setups_csv)
        if inferred is None:
            raise ValueError("Could not infer as-of date. Pass --asof-date YYYY-MM-DD.")
        asof = inferred

    root_dir = Path(args.root_dir).expanduser().resolve()
    screener_csv = Path(args.screener_csv).expanduser().resolve() if args.screener_csv.strip() else None
    if screener_csv is None:
        screener_csv = detect_default_screener(asof, root_dir)

    spot_map = load_spot_map(screener_csv)
    setups = pd.read_csv(setups_csv, low_memory=False)

    needed = {"ticker", "strategy", "expiry", "entry_gate", "long_strike", "short_strike", "width"}
    missing = [c for c in needed if c not in setups.columns]
    if missing:
        raise ValueError(f"Missing setup columns: {missing}")

    out_rows = []
    hist_cache: Dict[str, pd.DataFrame] = {}

    for r in setups.itertuples(index=False):
        ticker = str(r.ticker).upper().strip()
        strategy = str(r.strategy).strip()
        expiry = pd.to_datetime(r.expiry, errors="coerce")
        if pd.isna(expiry):
            continue
        dte = int((expiry.date() - asof).days)
        if dte <= 0:
            continue

        _, gate_net, _ = parse_entry_gate(r.entry_gate)
        if gate_net is None or not np.isfinite(gate_net):
            continue

        spot = spot_map.get(ticker, math.nan)
        if not np.isfinite(spot) or spot <= 0:
            spot = yfinance_spot_on_or_before(ticker, asof)
        if not np.isfinite(spot) or spot <= 0:
            continue

        be = breakeven(strategy, r.long_strike, r.short_strike, gate_net)
        req = required_win_rate_pct(strategy, r.width, gate_net)
        if not np.isfinite(be) or not np.isfinite(req):
            continue

        if ticker not in hist_cache:
            hist_cache[ticker] = get_price_history(ticker, asof, float(args.lookback_years))
        hist = hist_cache[ticker]
        if hist.empty:
            continue

        signals, wins, no_touch_pct = simulate_setup(
            strategy=strategy,
            hist=hist,
            dte=dte,
            spot_asof=float(spot),
            short_strike=safe_float(r.short_strike),
            breakeven_level=float(be),
        )
        if signals <= 0:
            continue

        hist_success_pct = 100.0 * wins / signals
        edge = hist_success_pct - req
        confidence = confidence_bucket(signals)
        if signals < int(args.min_signals):
            verdict = "LOW_SAMPLE"
        else:
            verdict = "PASS" if edge > 0 else "FAIL"

        out_rows.append(
            {
                "ticker": ticker,
                "strategy": strategy,
                "expiry": expiry.date().isoformat(),
                "dte": dte,
                "entry_gate": str(r.entry_gate),
                "spot_at_signal": round(float(spot), 4),
                "required_win_pct": round(float(req), 2),
                "hist_success_pct": round(float(hist_success_pct), 2),
                "edge_pct": round(float(edge), 2),
                "credit_no_touch_pct": round(float(no_touch_pct), 2) if np.isfinite(no_touch_pct) else np.nan,
                "signals": int(signals),
                "wins": int(wins),
                "confidence": confidence,
                "verdict": verdict,
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise RuntimeError("No rows scored. Check setups content and market data availability.")

    out = out.sort_values(["verdict", "edge_pct", "hist_success_pct", "signals"], ascending=[True, False, False, False])
    out = out.reset_index(drop=True)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"setup_likelihood_{asof.isoformat()}.csv"
    out_md = out_dir / f"setup_likelihood_{asof.isoformat()}.md"
    out.to_csv(out_csv, index=False)

    md_lines = [
        f"As-of date: {asof.isoformat()}",
        f"Setups: {setups_csv}",
        f"Screener spot source: {screener_csv if screener_csv else 'yfinance fallback'}",
        "",
        "## Setup Likelihood (Historical Analog)",
        out.to_markdown(index=False),
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Setups scored: {len(out)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
