#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
import time
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


def required_win_rate_pct(
    strategy: str,
    width: float,
    net: float,
    put_width: float = math.nan,
    call_width: float = math.nan,
) -> float:
    w = safe_float(width)
    n = safe_float(net)
    if not np.isfinite(n):
        return math.nan
    s = str(strategy).strip()
    if s == "Iron Condor":
        pw = safe_float(put_width)
        cw = safe_float(call_width)
        w_eff = max(pw, cw) if np.isfinite(pw) and np.isfinite(cw) else w
        if not np.isfinite(w_eff) or w_eff <= 0:
            return math.nan
        max_profit = max(0.0, n)
        max_loss = max(0.0, w_eff - n)
    elif not np.isfinite(w) or w <= 0:
        return math.nan
    elif s in {"Bull Call Debit", "Bear Put Debit"}:
        max_profit = max(0.0, w - n)
        max_loss = max(0.0, n)
    else:
        max_profit = max(0.0, n)
        max_loss = max(0.0, w - n)
    denom = max_profit + max_loss
    if denom <= 0:
        return math.nan
    return 100.0 * (max_loss / denom)


def breakeven_levels(
    strategy: str,
    long_strike: float,
    short_strike: float,
    net: float,
    short_call_strike: float = math.nan,
) -> Tuple[float, float]:
    s = str(strategy).strip()
    ls = safe_float(long_strike)
    ss = safe_float(short_strike)
    n = safe_float(net)
    if not np.isfinite(ss) or not np.isfinite(n):
        return math.nan, math.nan
    if s == "Bull Call Debit":
        return ls + n, ls + n
    if s == "Bear Put Debit":
        return ls - n, ls - n
    if s == "Bull Put Credit":
        return ss - n, ss - n
    if s == "Bear Call Credit":
        return ss + n, ss + n
    if s == "Iron Condor":
        sc = safe_float(short_call_strike)
        if not np.isfinite(sc):
            return math.nan, math.nan
        return ss - n, sc + n
    return math.nan, math.nan


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


def normalize_download(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close"]
    for col in needed:
        if col not in out.columns:
            return pd.DataFrame()
    out = out[needed].copy()
    out = out.dropna(subset=["High", "Low", "Close"])
    return out


def safe_symbol_for_file(ticker: str) -> str:
    t = str(ticker or "").strip().upper()
    t = re.sub(r"[^A-Z0-9._-]+", "_", t)
    return t or "UNKNOWN"


def yfinance_download_retry(
    ticker: str,
    start: str,
    end: str,
    retries: int = 3,
    pause_sec: float = 0.8,
) -> pd.DataFrame:
    last = pd.DataFrame()
    for attempt in range(max(1, int(retries))):
        try:
            raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
        except Exception:
            raw = pd.DataFrame()
        norm = normalize_download(raw)
        if not norm.empty:
            return norm
        last = norm
        if attempt < retries - 1:
            time.sleep(max(0.1, float(pause_sec)))
    return last


def load_history_cached(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Optional[Path],
    retries: int = 3,
) -> pd.DataFrame:
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_name = f"{safe_symbol_for_file(ticker)}_{start}_{end}.csv"
        cache_path = cache_dir / cache_name
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
                return normalize_download(cached)
            except Exception:
                pass

    hist = yfinance_download_retry(ticker, start=start, end=end, retries=retries)
    if cache_dir is not None and not hist.empty:
        try:
            cache_name = f"{safe_symbol_for_file(ticker)}_{start}_{end}.csv"
            cache_path = cache_dir / cache_name
            to_save = hist.reset_index().rename(columns={"index": "Date"})
            if "Date" not in to_save.columns:
                to_save.insert(0, "Date", hist.index)
            to_save.to_csv(cache_path, index=False)
        except Exception:
            pass
    return hist


def yfinance_spot_on_or_before(
    ticker: str,
    asof: dt.date,
    cache_dir: Optional[Path] = None,
) -> float:
    start = (asof - dt.timedelta(days=12)).isoformat()
    end = (asof + dt.timedelta(days=1)).isoformat()
    hist = load_history_cached(ticker=ticker, start=start, end=end, cache_dir=cache_dir)
    if hist.empty:
        return math.nan
    close_series = pd.to_numeric(hist.get("Close"), errors="coerce").dropna()
    if close_series.empty:
        return math.nan
    return float(close_series.iloc[-1])


def get_price_history(
    ticker: str,
    asof: dt.date,
    lookback_years: float,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    start = (asof - dt.timedelta(days=int(round(365.25 * float(lookback_years))))).isoformat()
    end = (asof + dt.timedelta(days=1)).isoformat()
    return load_history_cached(ticker=ticker, start=start, end=end, cache_dir=cache_dir)


def simulate_setup(
    strategy: str,
    hist: pd.DataFrame,
    dte: int,
    spot_asof: float,
    short_strike: float,
    breakeven_level: float,
    short_call_strike: float = math.nan,
    upper_breakeven_level: float = math.nan,
) -> Tuple[int, int, float]:
    s = str(strategy).strip()
    if dte <= 0 or hist.empty:
        return 0, 0, math.nan
    if not np.isfinite(spot_asof) or spot_asof <= 0:
        return 0, 0, math.nan
    if not np.isfinite(short_strike) or not np.isfinite(breakeven_level):
        return 0, 0, math.nan

    be_ratio = float(breakeven_level) / float(spot_asof)
    upper_be_ratio = (
        float(upper_breakeven_level) / float(spot_asof)
        if np.isfinite(upper_breakeven_level)
        else math.nan
    )
    short_ratio = float(short_strike) / float(spot_asof)
    short_call_ratio = (
        float(short_call_strike) / float(spot_asof)
        if np.isfinite(short_call_strike)
        else math.nan
    )

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
        elif s == "Iron Condor":
            if not (np.isfinite(upper_be_ratio) and np.isfinite(short_call_ratio)):
                continue
            sim_be_high = float(entry) * upper_be_ratio
            sim_short_call = float(entry) * short_call_ratio
            win = sim_be <= end_close <= sim_be_high
            no_touch_n += 1
            if lo > sim_short and hi < sim_short_call:
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
    ap.add_argument(
        "--cache-dir",
        default="",
        help="Optional cache directory for yfinance OHLC downloads.",
    )
    return ap.parse_args()


def unknown_row(
    ticker: str,
    strategy: str,
    expiry: str,
    dte: int,
    entry_gate: str,
    spot_at_signal: float,
    required_win_pct: float,
    reason: str,
) -> Dict[str, object]:
    return {
        "ticker": ticker,
        "strategy": strategy,
        "expiry": expiry,
        "dte": int(dte),
        "entry_gate": str(entry_gate),
        "spot_at_signal": round(float(spot_at_signal), 4) if np.isfinite(spot_at_signal) else np.nan,
        "required_win_pct": round(float(required_win_pct), 2) if np.isfinite(required_win_pct) else np.nan,
        "hist_success_pct": np.nan,
        "edge_pct": np.nan,
        "credit_no_touch_pct": np.nan,
        "signals": 0,
        "wins": 0,
        "confidence": "Unknown",
        "verdict": "UNKNOWN",
        "status_reason": str(reason),
    }


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
    cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir.strip() else None

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
            out_rows.append(
                unknown_row(
                    ticker=ticker,
                    strategy=strategy,
                    expiry="",
                    dte=0,
                    entry_gate=str(getattr(r, "entry_gate", "")),
                    spot_at_signal=math.nan,
                    required_win_pct=math.nan,
                    reason="invalid_expiry",
                )
            )
            continue
        expiry_iso = expiry.date().isoformat()
        dte = int((expiry.date() - asof).days)
        if dte <= 0:
            out_rows.append(
                unknown_row(
                    ticker=ticker,
                    strategy=strategy,
                    expiry=expiry_iso,
                    dte=dte,
                    entry_gate=str(getattr(r, "entry_gate", "")),
                    spot_at_signal=math.nan,
                    required_win_pct=math.nan,
                    reason="non_positive_dte",
                )
            )
            continue

        _, gate_net, _ = parse_entry_gate(r.entry_gate)
        if gate_net is None or not np.isfinite(gate_net):
            out_rows.append(
                unknown_row(
                    ticker=ticker,
                    strategy=strategy,
                    expiry=expiry_iso,
                    dte=dte,
                    entry_gate=str(getattr(r, "entry_gate", "")),
                    spot_at_signal=math.nan,
                    required_win_pct=math.nan,
                    reason="bad_entry_gate",
                )
            )
            continue

        spot = spot_map.get(ticker, math.nan)
        if not np.isfinite(spot) or spot <= 0:
            spot = yfinance_spot_on_or_before(ticker, asof, cache_dir=cache_dir)
        if not np.isfinite(spot) or spot <= 0:
            out_rows.append(
                unknown_row(
                    ticker=ticker,
                    strategy=strategy,
                    expiry=expiry_iso,
                    dte=dte,
                    entry_gate=str(getattr(r, "entry_gate", "")),
                    spot_at_signal=math.nan,
                    required_win_pct=math.nan,
                    reason="missing_spot",
                )
            )
            continue

        short_call_strike = safe_float(getattr(r, "short_call_strike", math.nan))
        put_width = safe_float(getattr(r, "put_width", math.nan))
        call_width = safe_float(getattr(r, "call_width", math.nan))

        be_low, be_high = breakeven_levels(
            strategy,
            r.long_strike,
            r.short_strike,
            gate_net,
            short_call_strike=short_call_strike,
        )
        req = required_win_rate_pct(
            strategy,
            r.width,
            gate_net,
            put_width=put_width,
            call_width=call_width,
        )
        if not np.isfinite(be_low) or not np.isfinite(req):
            out_rows.append(
                unknown_row(
                    ticker=ticker,
                    strategy=strategy,
                    expiry=expiry_iso,
                    dte=dte,
                    entry_gate=str(getattr(r, "entry_gate", "")),
                    spot_at_signal=float(spot),
                    required_win_pct=req,
                    reason="invalid_breakeven_or_required_win",
                )
            )
            continue

        if ticker not in hist_cache:
            hist_cache[ticker] = get_price_history(
                ticker,
                asof,
                float(args.lookback_years),
                cache_dir=cache_dir,
            )
        hist = hist_cache[ticker]
        if hist.empty:
            out_rows.append(
                unknown_row(
                    ticker=ticker,
                    strategy=strategy,
                    expiry=expiry_iso,
                    dte=dte,
                    entry_gate=str(getattr(r, "entry_gate", "")),
                    spot_at_signal=float(spot),
                    required_win_pct=req,
                    reason="missing_history",
                )
            )
            continue

        signals, wins, no_touch_pct = simulate_setup(
            strategy=strategy,
            hist=hist,
            dte=dte,
            spot_asof=float(spot),
            short_strike=safe_float(r.short_strike),
            breakeven_level=float(be_low),
            short_call_strike=short_call_strike,
            upper_breakeven_level=float(be_high) if np.isfinite(be_high) else math.nan,
        )
        if signals <= 0:
            out_rows.append(
                unknown_row(
                    ticker=ticker,
                    strategy=strategy,
                    expiry=expiry_iso,
                    dte=dte,
                    entry_gate=str(getattr(r, "entry_gate", "")),
                    spot_at_signal=float(spot),
                    required_win_pct=req,
                    reason="no_historical_windows",
                )
            )
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
                "expiry": expiry_iso,
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
                "status_reason": "scored",
            }
        )

    out = pd.DataFrame(out_rows)
    if out.empty:
        raise RuntimeError("No rows scored. Check setups content and market data availability.")

    verdict_rank = {"PASS": 0, "LOW_SAMPLE": 1, "UNKNOWN": 2, "FAIL": 3}
    out["_verdict_rank"] = out["verdict"].astype(str).str.upper().map(verdict_rank).fillna(9).astype(int)
    out = out.sort_values(
        ["_verdict_rank", "edge_pct", "hist_success_pct", "signals", "ticker", "strategy", "expiry"],
        ascending=[True, False, False, False, True, True, True],
    )
    out = out.drop(columns=["_verdict_rank"])
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
