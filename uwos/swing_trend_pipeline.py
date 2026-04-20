#!/usr/bin/env python3
"""
swing_trend_pipeline.py — Multi-day swing trade trend analysis.

Reads raw 4-CSV daily data (stock-screener, chain-oi-changes, hot-chains,
dp-eod-report) plus whale reports across N trading days, computes cross-day
trend signals, scores tickers, and produces swing trade recommendations.

Usage:
  python -m uwos.swing_trend_pipeline --lookback 10
  python -m uwos.swing_trend_pipeline --lookback 5 --as-of 2026-02-27
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OCC_RE = re.compile(r"^([A-Z]{1,6})\d{6}[CP]\d{8}$")
OCC_FULL_RE = re.compile(r"^([A-Z]{1,6})(\d{6})([CP])(\d{8})$")
WHALE_TICKER_RE = re.compile(r"\|\s*([A-Z]{1,6})\s*\|")

# ---------------------------------------------------------------------------
# Utility helpers (self-contained to avoid import issues)
# ---------------------------------------------------------------------------

def _fnum(x: Any) -> float:
    try:
        if x is None:
            return math.nan
        if isinstance(x, str):
            s = x.strip().replace(",", "").replace("$", "").replace("%", "")
            if not s or s.lower() == "nan":
                return math.nan
            return float(s)
        if isinstance(x, float) and math.isnan(x):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def _linear_slope(values: Sequence[float]) -> float:
    """Least-squares slope of values vs sequential index. Returns NaN if < 2 finite points."""
    arr = np.asarray(list(values), dtype="float64")
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return float("nan")
    y = arr[mask]
    x = np.arange(len(y), dtype="float64")
    if len(np.unique(x)) < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def _r_squared(values: Sequence[float]) -> float:
    """R-squared of linear fit of values vs sequential index."""
    arr = np.asarray(list(values), dtype="float64")
    mask = np.isfinite(arr)
    if mask.sum() < 3:
        return float("nan")
    y = arr[mask]
    x = np.arange(len(y), dtype="float64")
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot == 0:
        return 1.0
    return max(0.0, 1.0 - ss_res / ss_tot)


def _consistency(values: Sequence[float], positive: bool = True) -> float:
    """Proportion of finite values that are positive (or negative if positive=False)."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return 0.0
    if positive:
        return sum(1 for v in finite if v > 0) / len(finite)
    return sum(1 for v in finite if v < 0) / len(finite)


def _majority_direction(values: Sequence[float]) -> str:
    pos = sum(1 for v in values if math.isfinite(v) and v > 0)
    neg = sum(1 for v in values if math.isfinite(v) and v < 0)
    total = pos + neg
    if total == 0:
        return "neutral"
    if pos / total >= 0.6:
        return "bullish"
    if neg / total >= 0.6:
        return "bearish"
    return "mixed"


def _safe_div(a: float, b: float) -> float:
    if b == 0 or not math.isfinite(a) or not math.isfinite(b):
        return 0.0
    return a / b


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not math.isfinite(v):
        return lo
    return max(lo, min(hi, v))


def _parse_occ_underlying(sym: str) -> Optional[str]:
    """Extract underlying ticker from OCC symbol like NVDA260306P00170000."""
    m = OCC_RE.match(str(sym).strip().upper())
    return m.group(1) if m else None


def _parse_occ_right(sym: str) -> Optional[str]:
    """Extract C or P from OCC symbol."""
    m = OCC_FULL_RE.match(str(sym).strip().upper())
    return m.group(3) if m else None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ScreenerFeatures:
    close: float = math.nan
    iv30d: float = math.nan
    iv_rank: float = math.nan
    iv30d_1d_change: float = math.nan
    iv30d_1w_change: float = math.nan
    bullish_premium: float = 0.0
    bearish_premium: float = 0.0
    flow_bias: float = 0.0
    put_call_ratio: float = math.nan
    volume_ratio: float = math.nan
    implied_move_perc: float = math.nan
    sector: str = ""
    market_cap: float = 0.0
    next_earnings_date: Optional[dt.date] = None


@dataclass
class OIFeatures:
    net_call_oi_build: float = 0.0
    net_put_oi_build: float = 0.0
    net_call_oi_close: float = 0.0
    net_put_oi_close: float = 0.0
    top_strike_call: float = math.nan
    top_strike_put: float = math.nan
    total_call_premium: float = 0.0
    total_put_premium: float = 0.0
    oi_direction_score: float = 0.0


@dataclass
class HotChainFeatures:
    total_premium: float = 0.0
    bid_side_volume: float = 0.0
    ask_side_volume: float = 0.0
    flow_direction: float = 0.0
    sweep_volume: float = 0.0
    sweep_ratio: float = 0.0
    avg_iv: float = math.nan
    n_active_contracts: int = 0


@dataclass
class DPFeatures:
    total_premium: float = 0.0
    total_volume: float = 0.0
    n_prints: int = 0
    vwap: float = math.nan
    above_mid_pct: float = math.nan


@dataclass
class SwingSignals:
    ticker: str = ""
    sector: str = ""
    n_days_observed: int = 0
    latest_close: float = math.nan
    latest_date: Optional[dt.date] = None
    next_earnings_date: Optional[dt.date] = None

    # Price trend
    price_slope: float = math.nan
    price_direction: str = "neutral"
    price_r_squared: float = math.nan

    # IV regime
    iv30d_slope: float = math.nan
    iv_regime: str = "stable"
    latest_iv_rank: float = math.nan
    iv_level: str = "mid"

    # Flow persistence
    flow_consistency: float = 0.0
    flow_direction: str = "mixed"
    avg_flow_bias: float = 0.0

    # Put/call ratio trend
    pcr_slope: float = math.nan
    pcr_direction: str = "stable"

    # Volume
    avg_volume_ratio: float = math.nan
    volume_surge_days: int = 0

    # OI momentum
    oi_momentum_slope: float = math.nan
    oi_consistency: float = 0.0
    oi_direction: str = "mixed"
    top_call_strike: float = math.nan
    top_put_strike: float = math.nan

    # Hot chain signals
    hot_flow_direction: str = "mixed"
    hot_flow_consistency: float = 0.0
    avg_sweep_ratio: float = 0.0
    sweep_slope: float = math.nan

    # DP signals
    dp_direction: str = "neutral"
    dp_consistency: float = 0.0
    dp_vwap_slope: float = math.nan

    # Whale signals
    whale_appearances: int = 0


@dataclass
class SwingScore:
    ticker: str = ""
    composite_score: float = 0.0
    flow_persistence_score: float = 0.0
    oi_momentum_score: float = 0.0
    iv_regime_score: float = 0.0
    price_trend_score: float = 0.0
    whale_consensus_score: float = 0.0
    dp_confirmation_score: float = 0.0
    direction: str = "neutral"
    recommended_strategy: str = ""
    recommended_track: str = ""
    confidence_tier: str = "Low"
    thesis: str = ""
    # Trade structure
    target_expiry: str = ""          # YYYY-MM-DD
    target_dte: int = 0
    long_strike: float = math.nan    # Buy leg
    short_strike: float = math.nan   # Sell leg
    spread_width: float = math.nan
    strike_setup: str = ""           # e.g. "Buy 215C / Sell 225C (10w, ~$3.80 debit)"
    est_cost: float = math.nan       # Estimated debit or credit in $
    cost_type: str = ""              # "debit" or "credit"
    # Earnings safety
    earnings_safe: bool = True       # False if earnings falls within trade DTE window
    earnings_label: str = ""         # e.g. "EARNINGS 2026-04-10 (7d before expiry)"
    # Schwab live validation (populated when schwab_validation is enabled)
    live_validated: Optional[bool] = None    # None=not run, True=valid, False=failed
    live_spot: float = math.nan              # Live underlying price
    live_long_strike: float = math.nan       # Snapped long strike from real chain
    live_short_strike: float = math.nan      # Snapped short strike from real chain
    live_spread_cost: float = math.nan       # Mid-price spread cost from live quotes
    live_bid_ask_width: float = math.nan     # Spread bid/ask width (liquidity signal)
    live_strike_setup: str = ""              # Updated setup string with live prices
    live_validation_note: str = ""           # Error/warning message
    # Live greeks (populated during Schwab validation)
    short_delta_live: float = math.nan       # Short leg delta (SHIELD verticals)
    long_delta_live: float = math.nan        # Long leg delta (FIRE verticals)
    short_put_delta_live: float = math.nan   # IC put short delta
    short_call_delta_live: float = math.nan  # IC call short delta
    # GEX regime (populated during Schwab validation)
    net_gex: float = math.nan                # Net gamma exposure (positive=pinned, negative=volatile)
    gex_regime: str = ""                     # "pinned" or "volatile"
    gex_support: float = math.nan            # Highest put GEX wall below spot
    gex_resistance: float = math.nan         # Highest call GEX wall above spot
    # Backtest edge (populated when backtest is enabled)
    hist_success_pct: float = math.nan       # Historical win rate %
    required_win_pct: float = math.nan       # Breakeven win rate %
    edge_pct: float = math.nan               # hist_success_pct - required_win_pct
    backtest_signals: int = 0                # Number of historical analog windows
    backtest_verdict: str = ""               # PASS, FAIL, LOW_SAMPLE, UNKNOWN
    backtest_confidence: str = ""            # High, Medium, Low, Very Low
    # Optimizer / repair metadata
    variant_tag: str = "base"                # base, pre_earnings, liquid_debit, safer_credit, etc.
    repair_source: str = ""                  # Human-readable reason this variant was generated.


# ---------------------------------------------------------------------------
# Data discovery
# ---------------------------------------------------------------------------

def discover_trading_days(
    root: Path, lookback: int, as_of: Optional[dt.date] = None,
) -> List[Tuple[dt.date, Path]]:
    """Find the N most recent usable market-data days up to as_of date."""
    if as_of is None:
        as_of = dt.date.today()
    days: List[Tuple[dt.date, Path]] = []
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if not DATE_DIR_RE.match(name):
            continue
        try:
            d = dt.date.fromisoformat(name)
        except ValueError:
            continue
        if d <= as_of and d.weekday() < 5 and has_market_day_data(entry, d):
            days.append((d, entry))
    days.sort(key=lambda x: x[0], reverse=True)
    selected = days[:lookback]
    selected.reverse()  # chronological order
    return selected


def has_market_day_data(day_dir: Path, trade_date: dt.date) -> bool:
    """Return True when a dated folder has enough data to count as a market day."""
    return resolve_csv_for_day(day_dir, trade_date.isoformat(), "stock-screener") is not None


def resolve_csv_for_day(
    day_dir: Path, date_str: str, prefix: str,
) -> Optional[Path]:
    """
    Find a CSV file for a given prefix in a dated folder.
    Handles: _unzipped_mode_a/ subdir, bare CSVs, ZIP files.
    """
    # 1. Check _unzipped_mode_a/
    unzipped = day_dir / "_unzipped_mode_a"
    if unzipped.is_dir():
        for f in unzipped.iterdir():
            if f.name.lower().startswith(prefix.lower()) and f.suffix.lower() == ".csv":
                return f

    # 2. Check bare CSVs in day_dir
    for f in day_dir.iterdir():
        if f.name.lower().startswith(prefix.lower()) and f.suffix.lower() == ".csv":
            return f

    # 3. Check subdirectory with prefix name (older format like chain-oi-changes-2026-01-07/)
    sub = day_dir / f"{prefix}-{date_str}"
    if sub.is_dir():
        for f in sub.iterdir():
            if f.suffix.lower() == ".csv":
                return f
    # Also check without date
    for d in day_dir.iterdir():
        if d.is_dir() and d.name.lower().startswith(prefix.lower()):
            for f in d.iterdir():
                if f.suffix.lower() == ".csv":
                    return f

    # 4. Check ZIP files
    for f in day_dir.iterdir():
        if f.name.lower().startswith(prefix.lower()) and f.suffix.lower() == ".zip":
            return f  # will be read via read_csv_from_zip

    return None


def read_csv_from_path(
    path: Path,
    usecols: Optional[List[str]] = None,
    dtype: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Read CSV from a .csv file or the first CSV inside a .zip file."""
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path, "r") as zf:
                names = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
                if not names:
                    return pd.DataFrame()
                with zf.open(names[0]) as f:
                    return pd.read_csv(f, low_memory=False, usecols=usecols, dtype=dtype)
        return pd.read_csv(path, low_memory=False, usecols=usecols, dtype=dtype)
    except Exception as exc:
        print(f"  [WARN] Failed to read {path}: {exc}", file=sys.stderr)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Phase 1: Screener-first loading
# ---------------------------------------------------------------------------

def load_screeners(
    trading_days: List[Tuple[dt.date, Path]],
) -> Dict[dt.date, pd.DataFrame]:
    """Load stock-screener CSV for each day."""
    result: Dict[dt.date, pd.DataFrame] = {}
    for d, day_dir in trading_days:
        path = resolve_csv_for_day(day_dir, d.isoformat(), "stock-screener")
        if path is None:
            print(f"  [WARN] No stock-screener found for {d}", file=sys.stderr)
            continue
        df = read_csv_from_path(path)
        if df.empty:
            continue
        # Normalize ticker column
        ticker_col = None
        for c in df.columns:
            if c.strip().lower() in ("ticker", "symbol"):
                ticker_col = c
                break
        if ticker_col is None:
            continue
        df = df.rename(columns={ticker_col: "ticker"})
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        result[d] = df
    return result


def build_ticker_universe(
    screeners: Dict[dt.date, pd.DataFrame],
    cfg: Dict,
) -> Set[str]:
    """Build the set of tickers to analyze based on cumulative flow activity."""
    filters = cfg.get("filters", {})
    exclude_etfs = filters.get("exclude_etfs", True)
    etf_types = set(filters.get("etf_issue_types", ["ETF", "Index"]))
    min_mcap = _fnum(filters.get("min_market_cap", 0))
    min_vol = _fnum(filters.get("min_avg30_volume", 0))
    max_tickers = int(filters.get("max_tickers_to_score", 200))

    # Aggregate flow activity across all days using vectorized pandas
    frames = []
    for d, df in screeners.items():
        sub = df.copy()
        sub["ticker"] = sub["ticker"].astype(str).str.strip().str.upper()
        # Filter: valid ticker length
        sub = sub[sub["ticker"].str.len().between(1, 6)]

        # Filter ETFs
        if exclude_etfs and "issue_type" in sub.columns:
            sub = sub[~sub["issue_type"].astype(str).str.strip().isin(etf_types)]

        # Market cap filter
        mcap_col = "marketcap" if "marketcap" in sub.columns else "market_cap"
        if mcap_col in sub.columns and math.isfinite(min_mcap) and min_mcap > 0:
            sub[mcap_col] = pd.to_numeric(sub[mcap_col], errors="coerce").fillna(0)
            sub = sub[sub[mcap_col] >= min_mcap]

        # Volume filter
        if "avg30_volume" in sub.columns and math.isfinite(min_vol) and min_vol > 0:
            sub["avg30_volume"] = pd.to_numeric(sub["avg30_volume"], errors="coerce").fillna(0)
            sub = sub[sub["avg30_volume"] >= min_vol]

        if not sub.empty:
            frames.append(sub)

    if not frames:
        return set()

    combined = pd.concat(frames, ignore_index=True)

    # Compute flow activity per ticker
    for col in ("bullish_premium", "bearish_premium", "call_volume", "put_volume"):
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0)
        else:
            combined[col] = 0.0

    combined["_flow"] = (
        combined["bullish_premium"].abs()
        + combined["bearish_premium"].abs()
        + combined["call_volume"]
        + combined["put_volume"]
    )
    activity = combined.groupby("ticker")["_flow"].sum()
    ranked = activity.sort_values(ascending=False).head(max_tickers)
    return set(ranked.index)


# ---------------------------------------------------------------------------
# Phase 2: Full data loading (filtered to ticker universe)
# ---------------------------------------------------------------------------

def load_chain_oi(
    trading_days: List[Tuple[dt.date, Path]],
    ticker_set: Set[str],
) -> Dict[dt.date, pd.DataFrame]:
    """Load chain-oi-changes filtered to ticker_set."""
    result: Dict[dt.date, pd.DataFrame] = {}
    for d, day_dir in trading_days:
        path = resolve_csv_for_day(day_dir, d.isoformat(), "chain-oi-changes")
        if path is None:
            continue
        df = read_csv_from_path(path)
        if df.empty:
            continue
        # Find underlying column
        ul_col = None
        for c in df.columns:
            if c.strip().lower() in ("underlying_symbol", "underlying", "ticker"):
                ul_col = c
                break
        if ul_col is None:
            continue
        df = df.rename(columns={ul_col: "underlying_symbol"})
        df["underlying_symbol"] = df["underlying_symbol"].astype(str).str.strip().str.upper()
        df = df[df["underlying_symbol"].isin(ticker_set)].copy()
        if not df.empty:
            result[d] = df
    return result


def load_hot_chains(
    trading_days: List[Tuple[dt.date, Path]],
    ticker_set: Set[str],
) -> Dict[dt.date, pd.DataFrame]:
    """Load hot-chains filtered to ticker_set by parsing option_symbol."""
    result: Dict[dt.date, pd.DataFrame] = {}
    for d, day_dir in trading_days:
        path = resolve_csv_for_day(day_dir, d.isoformat(), "hot-chains")
        if path is None:
            continue
        df = read_csv_from_path(path)
        if df.empty:
            continue
        # Parse underlying from option_symbol
        sym_col = None
        for c in df.columns:
            if c.strip().lower() in ("option_symbol", "symbol"):
                sym_col = c
                break
        if sym_col is None:
            continue
        df["_underlying"] = df[sym_col].astype(str).str.strip().str.upper().map(
            lambda s: _parse_occ_underlying(s) or ""
        )
        df = df[df["_underlying"].isin(ticker_set)].copy()
        if not df.empty:
            result[d] = df
    return result


def load_dp_eod(
    trading_days: List[Tuple[dt.date, Path]],
    ticker_set: Set[str],
) -> Dict[dt.date, pd.DataFrame]:
    """Load dp-eod-report filtered to ticker_set. Uses chunked reading for efficiency."""
    result: Dict[dt.date, pd.DataFrame] = {}
    for d, day_dir in trading_days:
        path = resolve_csv_for_day(day_dir, d.isoformat(), "dp-eod-report")
        if path is None:
            continue
        df = read_csv_from_path(path)
        if df.empty:
            continue
        # Find ticker column
        tk_col = None
        for c in df.columns:
            if c.strip().lower() in ("ticker", "symbol"):
                tk_col = c
                break
        if tk_col is None:
            continue
        df = df.rename(columns={tk_col: "ticker"})
        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
        df = df[df["ticker"].isin(ticker_set)].copy()
        if not df.empty:
            result[d] = df
    return result


def load_whale_mentions(
    trading_days: List[Tuple[dt.date, Path]],
    ticker_set: Set[str],
) -> Dict[dt.date, Set[str]]:
    """Parse whale reports to find which tickers are mentioned each day."""
    result: Dict[dt.date, Set[str]] = {}
    for d, day_dir in trading_days:
        tickers_found: Set[str] = set()

        # Try markdown whale file
        whale_md = day_dir / f"whale-{d.isoformat()}.md"
        if whale_md.exists():
            try:
                text = whale_md.read_text(encoding="utf-8", errors="ignore")
                # Extract tickers from markdown table rows
                for m in WHALE_TICKER_RE.finditer(text):
                    t = m.group(1).strip().upper()
                    if t in ticker_set and len(t) <= 6 and t.isalpha():
                        tickers_found.add(t)
            except Exception:
                pass

        # Try CSV whale file
        if not tickers_found:
            for pattern in [
                f"whale_trades_filtered-{d.isoformat()}.csv",
                "whale_trades_filtered.csv",
            ]:
                whale_csv = day_dir / pattern
                if whale_csv.exists():
                    try:
                        wdf = pd.read_csv(whale_csv, low_memory=False)
                        for c in wdf.columns:
                            if c.strip().lower() in ("underlying_symbol", "ticker", "symbol"):
                                for t in wdf[c].astype(str).str.strip().str.upper().unique():
                                    if t in ticker_set:
                                        tickers_found.add(t)
                                break
                    except Exception:
                        pass
                    break

        if tickers_found:
            result[d] = tickers_found
    return result


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_screener_features(row: pd.Series) -> ScreenerFeatures:
    """Extract features from a single screener row."""
    bull = _fnum(row.get("bullish_premium", 0))
    bear = _fnum(row.get("bearish_premium", 0))
    total_prem = abs(bull) + abs(bear)
    flow_bias = _safe_div(bull - bear, total_prem) if total_prem > 0 else 0.0

    iv30d = _fnum(row.get("iv30d", math.nan))
    iv30d_1d = _fnum(row.get("iv30d_1d", math.nan))
    iv30d_1w = _fnum(row.get("iv30d_1w", math.nan))

    total_vol = _fnum(row.get("total_volume", 0))
    avg30 = _fnum(row.get("avg30_volume", 0))

    # Parse next earnings date
    earn_raw = row.get("next_earnings_date", None)
    next_earn: Optional[dt.date] = None
    if earn_raw is not None and str(earn_raw).strip() not in ("", "nan", "NaT"):
        try:
            next_earn = dt.datetime.strptime(str(earn_raw).strip()[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            pass

    return ScreenerFeatures(
        close=_fnum(row.get("close", math.nan)),
        iv30d=iv30d,
        iv_rank=_fnum(row.get("iv_rank", row.get("iv30d_rank", math.nan))),
        iv30d_1d_change=iv30d - iv30d_1d if math.isfinite(iv30d) and math.isfinite(iv30d_1d) else math.nan,
        iv30d_1w_change=iv30d - iv30d_1w if math.isfinite(iv30d) and math.isfinite(iv30d_1w) else math.nan,
        bullish_premium=bull if math.isfinite(bull) else 0.0,
        bearish_premium=bear if math.isfinite(bear) else 0.0,
        flow_bias=flow_bias,
        put_call_ratio=_fnum(row.get("put_call_ratio", math.nan)),
        volume_ratio=_safe_div(total_vol, avg30) if avg30 > 0 else math.nan,
        implied_move_perc=_fnum(row.get("implied_move_perc", row.get("implied_move", math.nan))),
        sector=str(row.get("sector", "")).strip(),
        market_cap=_fnum(row.get("marketcap", row.get("market_cap", 0))),
        next_earnings_date=next_earn,
    )


def extract_oi_features(oi_df: pd.DataFrame, ticker: str, spot: float = math.nan) -> OIFeatures:
    """Aggregate chain-oi-changes for one underlying on one day."""
    if oi_df.empty:
        return OIFeatures()

    sub = oi_df[oi_df["underlying_symbol"] == ticker].copy()
    if sub.empty:
        return OIFeatures()

    # Parse option right (C/P) from option_symbol
    sym_col = "option_symbol" if "option_symbol" in sub.columns else None
    if sym_col is None:
        for c in sub.columns:
            if "option_symbol" in c.lower() or "symbol" in c.lower():
                sym_col = c
                break
    if sym_col is None:
        return OIFeatures()

    sub["_right"] = sub[sym_col].astype(str).map(lambda s: _parse_occ_right(s) or "")
    sub["_oi_diff"] = pd.to_numeric(sub.get("oi_diff_plain", sub.get("oi_diff", 0)), errors="coerce").fillna(0)
    sub["_strike"] = pd.to_numeric(sub.get("strike", 0), errors="coerce").fillna(0)
    sub["_premium"] = pd.to_numeric(sub.get("premium", sub.get("prev_total_premium", 0)), errors="coerce").fillna(0)

    calls = sub[sub["_right"] == "C"]
    puts = sub[sub["_right"] == "P"]

    net_call_build = float(calls.loc[calls["_oi_diff"] > 0, "_oi_diff"].sum())
    net_put_build = float(puts.loc[puts["_oi_diff"] > 0, "_oi_diff"].sum())
    net_call_close = float(calls.loc[calls["_oi_diff"] < 0, "_oi_diff"].sum())
    net_put_close = float(puts.loc[puts["_oi_diff"] < 0, "_oi_diff"].sum())

    # Top strike by OI build — filter to strikes within 30% of spot for relevance
    top_call_strike = math.nan
    top_put_strike = math.nan
    strike_range_pct = 0.30
    if not calls.empty:
        call_builds = calls[calls["_oi_diff"] > 0]
        if math.isfinite(spot) and spot > 0:
            call_builds = call_builds[
                (call_builds["_strike"] >= spot * (1 - strike_range_pct))
                & (call_builds["_strike"] <= spot * (1 + strike_range_pct))
            ]
        if not call_builds.empty:
            top_call_strike = float(call_builds.loc[call_builds["_oi_diff"].idxmax(), "_strike"])

    if not puts.empty:
        put_builds = puts[puts["_oi_diff"] > 0]
        if math.isfinite(spot) and spot > 0:
            put_builds = put_builds[
                (put_builds["_strike"] >= spot * (1 - strike_range_pct))
                & (put_builds["_strike"] <= spot * (1 + strike_range_pct))
            ]
        if not put_builds.empty:
            top_put_strike = float(put_builds.loc[put_builds["_oi_diff"].idxmax(), "_strike"])

    call_prem = float(calls["_premium"].abs().sum())
    put_prem = float(puts["_premium"].abs().sum())
    total = net_call_build + net_put_build
    oi_dir = _safe_div(net_call_build - net_put_build, total) if total > 0 else 0.0

    return OIFeatures(
        net_call_oi_build=net_call_build,
        net_put_oi_build=net_put_build,
        net_call_oi_close=net_call_close,
        net_put_oi_close=net_put_close,
        top_strike_call=top_call_strike,
        top_strike_put=top_put_strike,
        total_call_premium=call_prem,
        total_put_premium=put_prem,
        oi_direction_score=oi_dir,
    )


def extract_hot_chain_features(hot_df: pd.DataFrame, ticker: str) -> HotChainFeatures:
    """Aggregate hot-chains for one underlying on one day."""
    if hot_df.empty:
        return HotChainFeatures()

    sub = hot_df[hot_df["_underlying"] == ticker].copy()
    if sub.empty:
        return HotChainFeatures()

    prem = pd.to_numeric(sub.get("premium", 0), errors="coerce").fillna(0)
    bid_vol = pd.to_numeric(sub.get("bid_side_volume", 0), errors="coerce").fillna(0)
    ask_vol = pd.to_numeric(sub.get("ask_side_volume", 0), errors="coerce").fillna(0)
    sweep_vol = pd.to_numeric(sub.get("sweep_volume", 0), errors="coerce").fillna(0)
    volume = pd.to_numeric(sub.get("volume", 0), errors="coerce").fillna(0)
    iv = pd.to_numeric(sub.get("iv", math.nan), errors="coerce")

    total_prem = float(prem.sum())
    total_bid = float(bid_vol.sum())
    total_ask = float(ask_vol.sum())
    total_sweep = float(sweep_vol.sum())
    total_vol = float(volume.sum())

    flow_dir = _safe_div(total_ask - total_bid, total_ask + total_bid)
    sweep_rat = _safe_div(total_sweep, total_vol)

    # Volume-weighted IV
    iv_finite = iv[iv.notna() & (iv > 0)]
    vol_finite = volume[iv.notna() & (iv > 0)]
    if len(iv_finite) > 0 and float(vol_finite.sum()) > 0:
        avg_iv = float((iv_finite * vol_finite).sum() / vol_finite.sum())
    else:
        avg_iv = math.nan

    return HotChainFeatures(
        total_premium=total_prem,
        bid_side_volume=total_bid,
        ask_side_volume=total_ask,
        flow_direction=flow_dir,
        sweep_volume=total_sweep,
        sweep_ratio=sweep_rat,
        avg_iv=avg_iv,
        n_active_contracts=len(sub),
    )


def extract_dp_features(dp_df: pd.DataFrame, ticker: str) -> DPFeatures:
    """Aggregate dp-eod-report for one ticker on one day."""
    if dp_df.empty:
        return DPFeatures()

    sub = dp_df[dp_df["ticker"] == ticker].copy()
    if sub.empty:
        return DPFeatures()

    price = pd.to_numeric(sub.get("price", 0), errors="coerce").fillna(0)
    size = pd.to_numeric(sub.get("size", 0), errors="coerce").fillna(0)
    premium = pd.to_numeric(sub.get("premium", 0), errors="coerce").fillna(0)
    nbbo_bid = pd.to_numeric(sub.get("nbbo_bid", 0), errors="coerce").fillna(0)
    nbbo_ask = pd.to_numeric(sub.get("nbbo_ask", 0), errors="coerce").fillna(0)

    total_prem = float(premium.abs().sum())
    total_vol = float(size.sum())
    n_prints = len(sub)

    # VWAP
    if total_vol > 0:
        vwap = float((price * size).sum() / total_vol)
    else:
        vwap = math.nan

    # Above-midpoint percentage weighted by premium (accumulation signal)
    # Premium-weighted gives more weight to large institutional prints
    mid = (nbbo_bid + nbbo_ask) / 2.0
    valid = (mid > 0) & (price > 0) & (premium.abs() > 0)
    if valid.sum() > 0:
        above_mask = price[valid] >= mid[valid]
        weights = premium[valid].abs()
        total_weight = float(weights.sum())
        if total_weight > 0:
            above_mid = float(weights[above_mask].sum()) / total_weight
        else:
            above_mid = float(above_mask.sum()) / float(valid.sum())
    else:
        above_mid = math.nan

    return DPFeatures(
        total_premium=total_prem,
        total_volume=total_vol,
        n_prints=n_prints,
        vwap=vwap,
        above_mid_pct=above_mid,
    )


# ---------------------------------------------------------------------------
# Cross-day trend computation
# ---------------------------------------------------------------------------

def compute_swing_signals(
    ticker: str,
    screener_series: List[Tuple[dt.date, ScreenerFeatures]],
    oi_series: List[Tuple[dt.date, OIFeatures]],
    hot_series: List[Tuple[dt.date, HotChainFeatures]],
    dp_series: List[Tuple[dt.date, DPFeatures]],
    whale_days: int,
    cfg: Dict,
) -> SwingSignals:
    """Compute cross-day trend signals for a single ticker."""
    sig = SwingSignals(ticker=ticker)

    if not screener_series:
        return sig

    sig.n_days_observed = len(screener_series)
    sig.latest_date = screener_series[-1][0]
    sig.latest_close = screener_series[-1][1].close
    sig.sector = screener_series[-1][1].sector
    # Take most recent non-None earnings date
    for _, sf in reversed(screener_series):
        if sf.next_earnings_date is not None:
            sig.next_earnings_date = sf.next_earnings_date
            break

    scoring_cfg = cfg.get("scoring", {})
    iv_cfg = scoring_cfg.get("iv_regime", {})
    price_cfg = scoring_cfg.get("price_trend", {})

    # --- Price trend ---
    closes = [sf.close for _, sf in screener_series]
    if closes and math.isfinite(closes[0]) and closes[0] > 0:
        norm_closes = [c / closes[0] for c in closes]
        sig.price_slope = _linear_slope(norm_closes)
        sig.price_r_squared = _r_squared(norm_closes)
    else:
        sig.price_slope = _linear_slope(closes)
        sig.price_r_squared = _r_squared(closes)

    min_slope = float(price_cfg.get("min_slope_for_trend", 0.002))
    if math.isfinite(sig.price_slope):
        if sig.price_slope > min_slope:
            sig.price_direction = "bullish"
        elif sig.price_slope < -min_slope:
            sig.price_direction = "bearish"
        else:
            sig.price_direction = "range_bound"

    # --- IV regime ---
    iv_values = [sf.iv30d for _, sf in screener_series]
    sig.iv30d_slope = _linear_slope(iv_values)
    sig.latest_iv_rank = screener_series[-1][1].iv_rank

    expanding_thresh = float(iv_cfg.get("expanding_slope_threshold", 0.5))
    compressing_thresh = float(iv_cfg.get("compressing_slope_threshold", -0.5))
    if math.isfinite(sig.iv30d_slope):
        if sig.iv30d_slope > expanding_thresh:
            sig.iv_regime = "expanding"
        elif sig.iv30d_slope < compressing_thresh:
            sig.iv_regime = "compressing"
        else:
            sig.iv_regime = "stable"

    low_ceil = float(iv_cfg.get("low_iv_rank_ceil", 40))
    high_floor = float(iv_cfg.get("high_iv_rank_floor", 55))
    if math.isfinite(sig.latest_iv_rank):
        if sig.latest_iv_rank < low_ceil:
            sig.iv_level = "low"
        elif sig.latest_iv_rank > high_floor:
            sig.iv_level = "high"
        else:
            sig.iv_level = "mid"

    # --- Flow persistence ---
    flow_biases = [sf.flow_bias for _, sf in screener_series]
    sig.avg_flow_bias = float(np.nanmean(flow_biases)) if flow_biases else 0.0
    sig.flow_direction = _majority_direction(flow_biases)
    if sig.flow_direction == "bullish":
        sig.flow_consistency = _consistency(flow_biases, positive=True)
    elif sig.flow_direction == "bearish":
        sig.flow_consistency = _consistency(flow_biases, positive=False)
    else:
        sig.flow_consistency = 0.5

    # --- Put/call ratio trend ---
    pcr_values = [sf.put_call_ratio for _, sf in screener_series]
    sig.pcr_slope = _linear_slope(pcr_values)
    if math.isfinite(sig.pcr_slope):
        if sig.pcr_slope < -0.02:
            sig.pcr_direction = "declining"  # bullish
        elif sig.pcr_slope > 0.02:
            sig.pcr_direction = "rising"  # bearish
        else:
            sig.pcr_direction = "stable"

    # --- Volume anomaly ---
    vol_ratios = [sf.volume_ratio for _, sf in screener_series]
    finite_vr = [v for v in vol_ratios if math.isfinite(v)]
    sig.avg_volume_ratio = float(np.mean(finite_vr)) if finite_vr else math.nan
    surge_thresh = float(price_cfg.get("volume_surge_ratio", 1.5))
    sig.volume_surge_days = sum(1 for v in finite_vr if v > surge_thresh)

    # --- OI momentum ---
    oi_dirs = [of.oi_direction_score for _, of in oi_series]
    sig.oi_momentum_slope = _linear_slope(oi_dirs)
    sig.oi_direction = _majority_direction(oi_dirs)
    if sig.oi_direction == "bullish":
        sig.oi_consistency = _consistency(oi_dirs, positive=True)
    elif sig.oi_direction == "bearish":
        sig.oi_consistency = _consistency(oi_dirs, positive=False)
    else:
        sig.oi_consistency = 0.5

    # Top strikes (most frequent across days)
    call_strikes = [round(of.top_strike_call, 2) for _, of in oi_series if math.isfinite(of.top_strike_call)]
    put_strikes = [round(of.top_strike_put, 2) for _, of in oi_series if math.isfinite(of.top_strike_put)]
    if call_strikes:
        sig.top_call_strike = float(Counter(call_strikes).most_common(1)[0][0])
    if put_strikes:
        sig.top_put_strike = float(Counter(put_strikes).most_common(1)[0][0])

    # --- Hot chain signals ---
    hot_flows = [hf.flow_direction for _, hf in hot_series]
    hot_dir = _majority_direction(hot_flows)
    sig.hot_flow_direction = hot_dir
    if hot_dir in ("bullish", "bearish"):
        sig.hot_flow_consistency = _consistency(hot_flows, positive=(hot_dir == "bullish"))
    else:
        sig.hot_flow_consistency = 0.5

    sweep_ratios = [hf.sweep_ratio for _, hf in hot_series]
    sig.avg_sweep_ratio = float(np.nanmean(sweep_ratios)) if sweep_ratios else 0.0
    sig.sweep_slope = _linear_slope(sweep_ratios)

    # --- DP signals ---
    dp_above_mids = [dpf.above_mid_pct for _, dpf in dp_series]
    dp_vwaps = [dpf.vwap for _, dpf in dp_series]
    dp_dir = _majority_direction([
        (v - 0.5) for v in dp_above_mids if math.isfinite(v)
    ])
    if dp_dir == "bullish":
        sig.dp_direction = "accumulation"
    elif dp_dir == "bearish":
        sig.dp_direction = "distribution"
    else:
        sig.dp_direction = "neutral"
    finite_dp = [v for v in dp_above_mids if math.isfinite(v)]
    if finite_dp:
        accum_thresh = float(scoring_cfg.get("dp_confirmation", {}).get("accumulation_threshold", 0.55))
        if dp_dir == "bearish":
            # For bearish/distribution: measure fraction BELOW (1 - threshold)
            distrib_thresh = 1.0 - accum_thresh
            sig.dp_consistency = sum(1 for v in finite_dp if v < distrib_thresh) / len(finite_dp)
        else:
            sig.dp_consistency = sum(1 for v in finite_dp if v > accum_thresh) / len(finite_dp)
    sig.dp_vwap_slope = _linear_slope(dp_vwaps)

    # --- Whale signals ---
    sig.whale_appearances = whale_days

    return sig


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_ticker(signals: SwingSignals, cfg: Dict) -> SwingScore:
    """Score a ticker based on its cross-day swing signals."""
    scoring_cfg = cfg.get("scoring", {})
    weights = scoring_cfg.get("weights", {})

    score = SwingScore(ticker=signals.ticker)

    # Determine primary direction from multiple sources
    dir_votes = []
    if signals.flow_direction in ("bullish", "bearish"):
        dir_votes.append(signals.flow_direction)
    if signals.price_direction in ("bullish", "bearish"):
        dir_votes.append(signals.price_direction)
    if signals.oi_direction in ("bullish", "bearish"):
        dir_votes.append("bullish" if signals.oi_direction == "bullish" else "bearish")
    if signals.pcr_direction == "declining":
        dir_votes.append("bullish")
    elif signals.pcr_direction == "rising":
        dir_votes.append("bearish")

    bull_count = sum(1 for v in dir_votes if v == "bullish")
    bear_count = sum(1 for v in dir_votes if v == "bearish")
    if bull_count > bear_count and bull_count >= 2:
        score.direction = "bullish"
    elif bear_count > bull_count and bear_count >= 2:
        score.direction = "bearish"
    else:
        score.direction = "neutral"

    is_directional = score.direction in ("bullish", "bearish")

    # --- Flow persistence score (0-100) ---
    flow_base = signals.flow_consistency * 60.0 + signals.hot_flow_consistency * 40.0
    # Bonus if screener flow and hot-chain flow agree on direction
    if (signals.flow_direction in ("bullish", "bearish")
            and signals.flow_direction == signals.hot_flow_direction):
        flow_base = min(100.0, flow_base + 10.0)
    # Penalty if they disagree
    elif (signals.flow_direction in ("bullish", "bearish")
          and signals.hot_flow_direction in ("bullish", "bearish")
          and signals.flow_direction != signals.hot_flow_direction):
        flow_base = max(0.0, flow_base - 10.0)
    score.flow_persistence_score = _clip(flow_base)

    # --- OI momentum score (0-100) ---
    oi_base = signals.oi_consistency * 50.0
    oi_slope_norm = _clip(abs(signals.oi_momentum_slope) * 50.0 if math.isfinite(signals.oi_momentum_slope) else 0.0, 0, 30)
    oi_base += oi_slope_norm
    # Strike clustering bonus: if top strike repeats, OI is concentrated
    if math.isfinite(signals.top_call_strike) or math.isfinite(signals.top_put_strike):
        oi_base += 15.0
    # Direction alignment bonus / contradiction penalty
    if (score.direction == "bullish" and signals.oi_direction == "bullish") or \
       (score.direction == "bearish" and signals.oi_direction == "bearish"):
        oi_base += 5.0
    elif (score.direction == "bullish" and signals.oi_direction == "bearish") or \
         (score.direction == "bearish" and signals.oi_direction == "bullish"):
        oi_base = max(0.0, oi_base - 10.0)
    score.oi_momentum_score = _clip(oi_base)

    # --- IV regime score (0-100) ---
    # Use actual iv_rank on a gradient for proper differentiation,
    # not just low/mid/high buckets which cluster everyone at 90.
    iv_rank = signals.latest_iv_rank if math.isfinite(signals.latest_iv_rank) else 50.0
    if is_directional:
        # Directional (debit) trades prefer LOW IV: iv_rank 0→100, 100→20
        iv_base = max(20.0, 100.0 - iv_rank * 0.8)
        # IV expanding is good for breakout plays
        if signals.iv_regime == "expanding":
            iv_base = min(100.0, iv_base + 10.0)
        elif signals.iv_regime == "compressing":
            iv_base = min(100.0, iv_base + 5.0)  # compressing = cheaper entry
    else:
        # Neutral/range (credit) trades prefer HIGH IV: iv_rank 0→20, 100→100
        iv_base = max(20.0, 20.0 + iv_rank * 0.8)
        # IV compressing is bad for credit trades
        if signals.iv_regime == "compressing":
            iv_base = max(0.0, iv_base - 15.0)
    score.iv_regime_score = _clip(iv_base)

    # --- Price trend score (0-100) ---
    r2 = signals.price_r_squared if math.isfinite(signals.price_r_squared) else 0.0
    price_base = r2 * 50.0
    price_base += min(20.0, signals.volume_surge_days * 5.0)
    # Direction alignment bonus
    if is_directional:
        if (score.direction == "bullish" and signals.price_direction == "bullish") or \
           (score.direction == "bearish" and signals.price_direction == "bearish"):
            price_base += 25.0
        elif signals.price_direction == "range_bound":
            price_base += 5.0
        else:
            price_base = max(0.0, price_base - 10.0)
    else:
        if signals.price_direction == "range_bound":
            price_base += 25.0
    score.price_trend_score = _clip(price_base)

    # --- Whale consensus score (0-100) ---
    n_days = max(1, signals.n_days_observed)
    whale_rate = min(1.0, signals.whale_appearances / n_days)
    whale_base = whale_rate * 80.0
    if signals.whale_appearances >= 2:
        whale_base += 20.0
    score.whale_consensus_score = _clip(whale_base)

    # --- DP confirmation score (0-100) ---
    dp_base = signals.dp_consistency * 60.0
    if (score.direction == "bullish" and signals.dp_direction == "accumulation") or \
       (score.direction == "bearish" and signals.dp_direction == "distribution"):
        dp_base += 40.0
    elif signals.dp_direction == "neutral":
        dp_base += 10.0
    elif (score.direction == "bullish" and signals.dp_direction == "distribution") or \
         (score.direction == "bearish" and signals.dp_direction == "accumulation"):
        # DP contradicts the directional thesis — penalize
        dp_base = max(0.0, dp_base - 20.0)
    score.dp_confirmation_score = _clip(dp_base)

    # --- Composite ---
    _expected_weight_keys = {"flow_persistence", "oi_momentum", "iv_regime",
                             "price_trend", "whale_consensus", "dp_confirmation"}
    _missing_keys = _expected_weight_keys - set(weights.keys())
    if _missing_keys:
        import warnings
        warnings.warn(f"[swing_trend] Missing weight keys in config (using defaults): {_missing_keys}")
    w = {
        "flow_persistence": float(weights.get("flow_persistence", 0.30)),
        "oi_momentum": float(weights.get("oi_momentum", 0.20)),
        "iv_regime": float(weights.get("iv_regime", 0.15)),
        "price_trend": float(weights.get("price_trend", 0.15)),
        "whale_consensus": float(weights.get("whale_consensus", 0.10)),
        "dp_confirmation": float(weights.get("dp_confirmation", 0.10)),
    }
    _wsum = sum(w.values())
    if abs(_wsum - 1.0) > 0.01:
        import warnings
        warnings.warn(f"[swing_trend] Score weights sum to {_wsum:.3f}, expected 1.0 — results may be skewed")
    score.composite_score = _clip(
        w["flow_persistence"] * score.flow_persistence_score
        + w["oi_momentum"] * score.oi_momentum_score
        + w["iv_regime"] * score.iv_regime_score
        + w["price_trend"] * score.price_trend_score
        + w["whale_consensus"] * score.whale_consensus_score
        + w["dp_confirmation"] * score.dp_confirmation_score
    )

    # Strategy selection
    score.recommended_strategy, score.recommended_track = select_strategy(
        signals, score, cfg,
    )

    # Spread structure (strikes + expiry)
    _compute_spread_structure(signals, score, cfg)

    # Earnings safety check: flag if earnings falls within the trade window
    _check_earnings_safety(signals, score, cfg)

    # Confidence tier
    if score.composite_score >= 70 and signals.n_days_observed >= 5:
        score.confidence_tier = "High"
    elif score.composite_score >= 50 and signals.n_days_observed >= 3:
        score.confidence_tier = "Moderate"
    else:
        score.confidence_tier = "Low"

    # Thesis
    score.thesis = _build_thesis(signals, score)

    return score


def select_strategy(signals: SwingSignals, score: SwingScore, cfg: Dict) -> Tuple[str, str]:
    """Select strategy and track based on trend regime."""
    direction = score.direction
    iv_level = signals.iv_level
    iv_regime = signals.iv_regime

    # IV expansion with directional bias -> breakout debit
    if iv_regime == "expanding" and direction in ("bullish", "bearish"):
        if direction == "bullish":
            return "Bull Call Debit", "FIRE"
        return "Bear Put Debit", "FIRE"

    # Directional with low/mid IV -> debit spread
    if direction == "bullish" and iv_level in ("low", "mid"):
        return "Bull Call Debit", "FIRE"
    if direction == "bearish" and iv_level in ("low", "mid"):
        return "Bear Put Debit", "FIRE"

    # Directional with high IV -> credit spread (sell premium in direction)
    if direction == "bullish" and iv_level == "high":
        return "Bull Put Credit", "SHIELD"
    if direction == "bearish" and iv_level == "high":
        return "Bear Call Credit", "SHIELD"

    # Neutral / range-bound
    if iv_level == "high":
        return "Iron Condor", "SHIELD"
    if iv_level == "mid":
        return "Iron Condor", "SHIELD"

    # Neutral + low IV -> Iron Condor (range-bound, cheap premium)
    if iv_level == "low":
        return "Iron Condor", "SHIELD"

    # Fallback
    return "Bull Call Debit", "FIRE"


def _compute_spread_structure(
    signals: SwingSignals, score: SwingScore, cfg: Dict,
) -> None:
    """Populate target_expiry, strike_setup, long/short strikes on score."""
    strat_cfg = cfg.get("strategy_selection", {})
    latest_date = signals.latest_date or dt.date.today()
    spot = signals.latest_close

    # Determine DTE from strategy config
    dte_range = [21, 70]
    target_dte = 45
    strategy = score.recommended_strategy.lower().replace(" ", "_")
    for key, block in strat_cfg.items():
        if not isinstance(block, dict):
            continue
        block_strat = str(block.get("strategy", "")).lower().replace(" ", "_")
        if block_strat and block_strat in strategy:
            dte_range = block.get("dte_range", dte_range)
            target_dte = int(block.get("target_dte", target_dte))
            break

    score.target_dte = target_dte
    # Round target_expiry to nearest Friday
    raw_expiry = latest_date + dt.timedelta(days=target_dte)
    days_to_friday = (4 - raw_expiry.weekday()) % 7
    expiry = raw_expiry + dt.timedelta(days=days_to_friday)
    score.target_expiry = expiry.isoformat()

    if not math.isfinite(spot) or spot <= 0:
        return

    # Determine width tier based on spot price (matching daily pipeline)
    if spot < 25:
        width = 2.5
    elif spot < 75:
        width = 5.0
    elif spot < 150:
        width = 10.0
    elif spot < 500:
        width = 10.0
    else:
        width = 20.0

    # Use OI strike clusters as anchor, otherwise use spot-based strikes
    call_anchor = signals.top_call_strike if math.isfinite(signals.top_call_strike) else None
    put_anchor = signals.top_put_strike if math.isfinite(signals.top_put_strike) else None

    strat = score.recommended_strategy

    if strat == "Bull Call Debit":
        # Buy ATM/slightly OTM call, sell higher call
        if call_anchor and call_anchor > spot:
            long_k = _round_strike(spot * 1.01, width)
            short_k = _round_strike(long_k + width, width)
        else:
            long_k = _round_strike(spot * 1.02, width)
            short_k = _round_strike(long_k + width, width)
        score.long_strike = long_k
        score.short_strike = short_k
        score.spread_width = short_k - long_k
        score.cost_type = "debit"
        score.est_cost = _estimate_spread_cost(spot, long_k, score.spread_width, "debit")
        score.strike_setup = (
            f"Buy {long_k:g}C / Sell {short_k:g}C "
            f"({score.spread_width:g}w, ~${score.est_cost:.2f} debit)"
        )

    elif strat == "Bear Put Debit":
        # Buy ATM/slightly OTM put, sell lower put
        if put_anchor and put_anchor < spot:
            long_k = _round_strike(spot * 0.99, width)
            short_k = _round_strike(long_k - width, width)
        else:
            long_k = _round_strike(spot * 0.98, width)
            short_k = _round_strike(long_k - width, width)
        score.long_strike = long_k
        score.short_strike = short_k
        score.spread_width = long_k - short_k
        score.cost_type = "debit"
        score.est_cost = _estimate_spread_cost(spot, long_k, score.spread_width, "debit")
        score.strike_setup = (
            f"Buy {long_k:g}P / Sell {short_k:g}P "
            f"({score.spread_width:g}w, ~${score.est_cost:.2f} debit)"
        )

    elif strat == "Bull Put Credit":
        # Sell OTM put, buy further OTM put
        if put_anchor and put_anchor < spot:
            short_k = _round_strike(put_anchor, width)
        else:
            short_k = _round_strike(spot * 0.92, width)
        long_k = _round_strike(short_k - width, width)
        score.short_strike = short_k
        score.long_strike = long_k
        score.spread_width = short_k - long_k
        score.cost_type = "credit"
        score.est_cost = _estimate_spread_cost(spot, short_k, score.spread_width, "credit")
        score.strike_setup = (
            f"Sell {short_k:g}P / Buy {long_k:g}P "
            f"({score.spread_width:g}w, ~${score.est_cost:.2f} credit)"
        )

    elif strat == "Bear Call Credit":
        # Sell OTM call, buy further OTM call
        if call_anchor and call_anchor > spot:
            short_k = _round_strike(call_anchor, width)
        else:
            short_k = _round_strike(spot * 1.08, width)
        long_k = _round_strike(short_k + width, width)
        score.short_strike = short_k
        score.long_strike = long_k
        score.spread_width = long_k - short_k
        score.cost_type = "credit"
        score.est_cost = _estimate_spread_cost(spot, short_k, score.spread_width, "credit")
        score.strike_setup = (
            f"Sell {short_k:g}C / Buy {long_k:g}C "
            f"({score.spread_width:g}w, ~${score.est_cost:.2f} credit)"
        )

    elif strat == "Iron Condor":
        # Put side: sell OTM put, buy further OTM put
        # Call side: sell OTM call, buy further OTM call
        put_short = _round_strike(spot * 0.92, width)
        put_long = _round_strike(put_short - width, width)
        call_short = _round_strike(spot * 1.08, width)
        call_long = _round_strike(call_short + width, width)
        score.short_strike = put_short  # inner put
        score.long_strike = call_short   # inner call
        score.spread_width = width
        score.cost_type = "credit"
        put_credit = _estimate_spread_cost(spot, put_short, width, "credit")
        call_credit = _estimate_spread_cost(spot, call_short, width, "credit")
        score.est_cost = round(put_credit + call_credit, 2)
        score.strike_setup = (
            f"Sell {put_short:g}P / Buy {put_long:g}P + "
            f"Sell {call_short:g}C / Buy {call_long:g}C "
            f"({width:g}w, ~${score.est_cost:.2f} credit)"
        )


def _estimate_spread_cost(
    spot: float, anchor_strike: float, width: float, cost_type: str,
) -> float:
    """Estimate the debit or credit for a vertical spread.

    Uses moneyness of the sold/bought strike relative to spot.
    Debit spreads: cost ~40-50% of width near ATM, less further OTM.
    Credit spreads: credit ~25-35% of width for typical OTM short strikes.
    """
    if width <= 0 or not math.isfinite(spot) or spot <= 0:
        return round(width * 0.35, 2) if math.isfinite(width) else 0.0
    moneyness = abs(anchor_strike - spot) / spot
    if cost_type == "debit":
        pct = max(0.25, 0.50 - moneyness * 3.0)
    else:  # credit
        pct = max(0.15, 0.35 - moneyness * 2.0)
    return round(width * pct, 2)


def _round_strike(raw: float, width: float) -> float:
    """Round a strike price to the nearest multiple of width."""
    if width <= 0 or not math.isfinite(raw):
        return raw
    return round(raw / width) * width


def _width_for_spot(spot: float) -> float:
    if spot < 25:
        return 2.5
    if spot < 75:
        return 5.0
    if spot < 500:
        return 10.0
    return 20.0


def _friday_on_or_before(value: dt.date) -> dt.date:
    return value - dt.timedelta(days=(value.weekday() - 4) % 7)


def _friday_on_or_after(value: dt.date) -> dt.date:
    return value + dt.timedelta(days=(4 - value.weekday()) % 7)


def _reset_live_and_backtest_fields(score: SwingScore) -> None:
    score.live_validated = None
    score.live_spot = math.nan
    score.live_long_strike = math.nan
    score.live_short_strike = math.nan
    score.live_spread_cost = math.nan
    score.live_bid_ask_width = math.nan
    score.live_strike_setup = ""
    score.live_validation_note = ""
    score.short_delta_live = math.nan
    score.long_delta_live = math.nan
    score.short_put_delta_live = math.nan
    score.short_call_delta_live = math.nan
    score.net_gex = math.nan
    score.gex_regime = ""
    score.gex_support = math.nan
    score.gex_resistance = math.nan
    score.hist_success_pct = math.nan
    score.required_win_pct = math.nan
    score.edge_pct = math.nan
    score.backtest_signals = 0
    score.backtest_verdict = ""
    score.backtest_confidence = ""


def _score_variant(base: SwingScore, tag: str, source: str) -> SwingScore:
    variant = replace(base)
    variant.variant_tag = tag
    variant.repair_source = source
    _reset_live_and_backtest_fields(variant)
    return variant


def _set_expiry(score: SwingScore, signals: SwingSignals, expiry: dt.date) -> None:
    latest = signals.latest_date or dt.date.today()
    score.target_expiry = expiry.isoformat()
    score.target_dte = max(0, int((expiry - latest).days))
    # Repaired structures are live-priced by Schwab; avoid rejecting them against
    # the rough heuristic estimate that was built for the original expiry.
    score.est_cost = math.nan


def _set_vertical_structure(
    score: SwingScore,
    *,
    strategy: str,
    spot: float,
    width: float,
    long_mult: float,
    short_mult: Optional[float] = None,
) -> bool:
    if not math.isfinite(spot) or spot <= 0 or width <= 0:
        return False

    score.recommended_strategy = strategy
    score.cost_type = "credit" if "Credit" in strategy else "debit"
    score.est_cost = math.nan

    if strategy == "Bull Call Debit":
        long_k = _round_strike(spot * long_mult, width)
        short_k = _round_strike(long_k + width, width)
        if short_k <= long_k:
            short_k = long_k + width
        score.long_strike = long_k
        score.short_strike = short_k
        score.spread_width = short_k - long_k
        score.strike_setup = f"Buy {long_k:g}C / Sell {short_k:g}C ({score.spread_width:g}w, live-priced debit)"
        return True

    if strategy == "Bear Put Debit":
        long_k = _round_strike(spot * long_mult, width)
        short_k = _round_strike(long_k - width, width)
        if short_k >= long_k:
            short_k = long_k - width
        score.long_strike = long_k
        score.short_strike = short_k
        score.spread_width = long_k - short_k
        score.strike_setup = f"Buy {long_k:g}P / Sell {short_k:g}P ({score.spread_width:g}w, live-priced debit)"
        return True

    if strategy == "Bull Put Credit":
        short_k = _round_strike(spot * float(short_mult or 0.88), width)
        long_k = _round_strike(short_k - width, width)
        if long_k >= short_k:
            long_k = short_k - width
        score.short_strike = short_k
        score.long_strike = long_k
        score.spread_width = short_k - long_k
        score.strike_setup = f"Sell {short_k:g}P / Buy {long_k:g}P ({score.spread_width:g}w, live-priced credit)"
        return True

    if strategy == "Bear Call Credit":
        short_k = _round_strike(spot * float(short_mult or 1.12), width)
        long_k = _round_strike(short_k + width, width)
        if long_k <= short_k:
            long_k = short_k + width
        score.short_strike = short_k
        score.long_strike = long_k
        score.spread_width = long_k - short_k
        score.strike_setup = f"Sell {short_k:g}C / Buy {long_k:g}C ({score.spread_width:g}w, live-priced credit)"
        return True

    return False


def _set_iron_condor_structure(
    score: SwingScore,
    *,
    spot: float,
    width: float,
    put_mult: float = 0.84,
    call_mult: float = 1.16,
) -> bool:
    if not math.isfinite(spot) or spot <= 0 or width <= 0:
        return False
    put_short = _round_strike(spot * put_mult, width)
    call_short = _round_strike(spot * call_mult, width)
    put_long = _round_strike(put_short - width, width)
    call_long = _round_strike(call_short + width, width)
    if not (put_long < put_short < call_short < call_long):
        return False

    score.recommended_strategy = "Iron Condor"
    score.cost_type = "credit"
    score.short_strike = put_short
    score.long_strike = call_short
    score.spread_width = width
    score.est_cost = math.nan
    score.strike_setup = (
        f"Sell {put_short:g}P / Buy {put_long:g}P + "
        f"Sell {call_short:g}C / Buy {call_long:g}C "
        f"({width:g}w, live-priced credit)"
    )
    return True


def generate_trade_repair_variants(
    scores: List[SwingScore],
    signals_map: Dict[str, SwingSignals],
    cfg: Dict,
) -> List[SwingScore]:
    """Create alternate trade structures before Schwab/backtest gates run.

    The base scorer intentionally creates one simple structure per ticker. This
    optimizer adds a small set of alternatives that may avoid earnings, reduce
    short-leg delta, or use a more liquid debit spread. All variants still have
    to pass Schwab validation and the historical likelihood backtest.
    """
    opt_cfg = cfg.get("trade_repair", {})
    if opt_cfg.get("enabled", True) is False:
        return []

    max_variants_per_score = int(opt_cfg.get("max_variants_per_score", 3))
    min_dte = int(opt_cfg.get("min_repair_dte", 7))
    max_total_variants = int(opt_cfg.get("max_total_variants", 150))
    variants: List[SwingScore] = []
    seen: Set[Tuple[str, str, str, float, float, float, str]] = set()

    def add_variant(v: SwingScore, sig: SwingSignals) -> None:
        if len(variants) >= max_total_variants:
            return
        key = (
            v.ticker,
            v.recommended_strategy,
            v.target_expiry,
            round(v.long_strike, 4) if math.isfinite(v.long_strike) else math.nan,
            round(v.short_strike, 4) if math.isfinite(v.short_strike) else math.nan,
            round(v.spread_width, 4) if math.isfinite(v.spread_width) else math.nan,
            v.variant_tag,
        )
        if key in seen:
            return
        _check_earnings_safety(sig, v, cfg)
        seen.add(key)
        variants.append(v)

    for base in scores:
        if len(variants) >= max_total_variants:
            break
        sig = signals_map.get(base.ticker)
        if sig is None:
            continue
        latest = sig.latest_date or dt.date.today()
        spot = sig.latest_close
        if not math.isfinite(spot) or spot <= 0:
            continue
        width = base.spread_width if math.isfinite(base.spread_width) and base.spread_width > 0 else _width_for_spot(spot)
        per_score = 0

        # 1. If earnings blocks the original expiry, try the Friday before the
        # configured earnings buffer. This is the most direct repair.
        earn = sig.next_earnings_date
        if earn is not None:
            buffer_days = int(cfg.get("filters", {}).get("earnings_buffer_days", 3))
            pre_earn_expiry = _friday_on_or_before(earn - dt.timedelta(days=buffer_days))
            if (pre_earn_expiry - latest).days >= min_dte and pre_earn_expiry != dt.date.fromisoformat(base.target_expiry):
                v = _score_variant(base, "pre_earnings", "expiry moved before earnings window")
                _set_expiry(v, sig, pre_earn_expiry)
                add_variant(v, sig)
                per_score += 1

        if per_score >= max_variants_per_score:
            continue

        strat = base.recommended_strategy
        base_expiry = dt.date.fromisoformat(base.target_expiry)

        # 2. Debit spreads blocked by tiny debit / wide markets get a more
        # intrinsic live-priced structure, which often improves bid/ask-to-debit.
        if strat == "Bull Call Debit":
            v = _score_variant(base, "liquid_debit", "more intrinsic call debit spread for liquidity")
            _set_expiry(v, sig, base_expiry)
            if _set_vertical_structure(v, strategy=strat, spot=spot, width=width, long_mult=0.97):
                add_variant(v, sig)
                per_score += 1
        elif strat == "Bear Put Debit":
            v = _score_variant(base, "liquid_debit", "more intrinsic put debit spread for liquidity")
            _set_expiry(v, sig, base_expiry)
            if _set_vertical_structure(v, strategy=strat, spot=spot, width=width, long_mult=1.03):
                add_variant(v, sig)
                per_score += 1

        if per_score >= max_variants_per_score:
            continue

        # 3. Credit spreads / condors get a farther-OTM version to reduce short
        # delta. Volatile neutral condors also get directional alternatives when
        # the underlying signals lean one way.
        if strat == "Bull Put Credit":
            v = _score_variant(base, "safer_credit", "put credit spread moved farther OTM for delta")
            _set_expiry(v, sig, base_expiry)
            if _set_vertical_structure(v, strategy=strat, spot=spot, width=width, long_mult=1.0, short_mult=0.84):
                add_variant(v, sig)
                per_score += 1
        elif strat == "Bear Call Credit":
            v = _score_variant(base, "safer_credit", "call credit spread moved farther OTM for delta")
            _set_expiry(v, sig, base_expiry)
            if _set_vertical_structure(v, strategy=strat, spot=spot, width=width, long_mult=1.0, short_mult=1.16):
                add_variant(v, sig)
                per_score += 1
        elif strat == "Iron Condor":
            v = _score_variant(base, "wide_condor", "iron condor moved farther OTM for short-delta risk")
            _set_expiry(v, sig, base_expiry)
            if _set_iron_condor_structure(v, spot=spot, width=width):
                add_variant(v, sig)
                per_score += 1

            if per_score < max_variants_per_score:
                if sig.price_direction == "bullish" or sig.flow_direction == "bullish":
                    v = _score_variant(base, "directional_repair", "neutral condor converted to bullish credit spread")
                    _set_expiry(v, sig, base_expiry)
                    if _set_vertical_structure(v, strategy="Bull Put Credit", spot=spot, width=width, long_mult=1.0, short_mult=0.84):
                        add_variant(v, sig)
                        per_score += 1
                elif sig.price_direction == "bearish" or sig.flow_direction == "bearish":
                    v = _score_variant(base, "directional_repair", "neutral condor converted to bearish credit spread")
                    _set_expiry(v, sig, base_expiry)
                    if _set_vertical_structure(v, strategy="Bear Call Credit", spot=spot, width=width, long_mult=1.0, short_mult=1.16):
                        add_variant(v, sig)
                        per_score += 1

        # 4. A listed-expiry repair: try a shorter standard Friday if the base
        # expiry is too far out or not listed in Schwab. Keep this after the
        # main repairs to avoid over-expanding the batch.
        if per_score < max_variants_per_score:
            alt_expiry = _friday_on_or_after(latest + dt.timedelta(days=28))
            if (alt_expiry - latest).days >= min_dte and alt_expiry != base_expiry:
                v = _score_variant(base, "listed_expiry", "alternate standard expiry for live chain availability")
                _set_expiry(v, sig, alt_expiry)
                add_variant(v, sig)

    return variants


def _check_earnings_safety(
    signals: SwingSignals, score: SwingScore, cfg: Dict,
) -> None:
    """Flag trades where earnings falls within the DTE window."""
    earn = signals.next_earnings_date
    if earn is None:
        score.earnings_safe = True
        score.earnings_label = ""
        return

    # Parse target_expiry back to a date
    try:
        expiry_date = dt.date.fromisoformat(score.target_expiry)
    except (ValueError, TypeError):
        score.earnings_safe = True
        score.earnings_label = ""
        return

    latest = signals.latest_date or dt.date.today()
    # Earnings buffer: configurable days before/after earnings to avoid
    earn_cfg = cfg.get("filters", {})
    buffer_days = int(earn_cfg.get("earnings_buffer_days", 3))

    days_to_earn = (earn - latest).days
    days_before_expiry = (expiry_date - earn).days
    days_after_expiry = (earn - expiry_date).days

    # Unsafe if earnings occurs before/on expiry. Also unsafe when the trade
    # expires inside the configured pre-earnings buffer. Expiring exactly on
    # the buffer boundary is allowed so a Friday expiry three calendar days
    # before a Monday report does not get mislabeled as "earnings in window."
    earnings_through_expiry = latest <= earn <= expiry_date
    buffer_overlap = (
        expiry_date < earn
        and 0 < days_after_expiry < buffer_days
        and earn >= latest
    )

    if earnings_through_expiry or buffer_overlap:
        score.earnings_safe = False
        if earnings_through_expiry:
            timing = f"{days_before_expiry}d before expiry"
        else:
            timing = f"{days_after_expiry}d after expiry, inside {buffer_days}d buffer"
        score.earnings_label = f"EARNINGS {earn.isoformat()} ({days_to_earn}d away, {timing})"
    else:
        score.earnings_safe = True
        score.earnings_label = ""


def _build_thesis(signals: SwingSignals, score: SwingScore) -> str:
    """Generate a human-readable thesis string."""
    parts = []
    direction = score.direction.capitalize()
    days = signals.n_days_observed

    parts.append(
        f"{direction} bias across {days} days"
    )

    # Note if flow agrees or disagrees with overall direction
    flow_agrees = (signals.flow_direction == score.direction) or signals.flow_direction == "mixed"
    if signals.flow_consistency >= 0.7:
        if flow_agrees:
            parts.append(f"strong flow consistency ({signals.flow_consistency:.0%})")
        else:
            parts.append(f"flow {signals.flow_direction} ({signals.flow_consistency:.0%}) but overridden")
    elif signals.flow_consistency >= 0.5:
        if flow_agrees:
            parts.append(f"moderate flow consistency ({signals.flow_consistency:.0%})")
        else:
            parts.append(f"flow {signals.flow_direction} ({signals.flow_consistency:.0%}) but overridden")

    if signals.oi_direction in ("bullish", "bearish"):
        parts.append(f"OI {signals.oi_direction}")

    if signals.iv_regime != "stable":
        parts.append(f"IV {signals.iv_regime}")

    if signals.volume_surge_days >= 2:
        parts.append(f"{signals.volume_surge_days} volume surge days")

    if signals.whale_appearances >= 2:
        parts.append(f"whale presence on {signals.whale_appearances} days")

    if signals.dp_direction == "accumulation":
        parts.append("DP accumulation")
    elif signals.dp_direction == "distribution":
        parts.append("DP distribution")

    if not score.earnings_safe:
        parts.append(f"**{score.earnings_label}**")

    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Schwab live validation
# ---------------------------------------------------------------------------

def _find_nearest_strike(
    available_strikes: List[float], target: float,
) -> Optional[float]:
    """Find the nearest available strike to target."""
    if not available_strikes or not math.isfinite(target):
        return None
    return min(available_strikes, key=lambda s: abs(s - target))


def _contract_mid_price(contract: Dict[str, Any]) -> Optional[float]:
    """Get mid-price from a chain contract dict."""
    bid = contract.get("bid")
    ask = contract.get("ask")
    if bid is not None and ask is not None:
        try:
            b, a = float(bid), float(ask)
            if math.isfinite(b) and math.isfinite(a) and b >= 0 and a >= 0:
                return (b + a) / 2.0
        except (TypeError, ValueError):
            pass
    mark = contract.get("mark")
    if mark is not None:
        try:
            m = float(mark)
            if math.isfinite(m) and m >= 0:
                return m
        except (TypeError, ValueError):
            pass
    return None


def _contract_bid_ask_spread(contract: Dict[str, Any]) -> float:
    """Get bid/ask spread width from a chain contract dict."""
    bid = contract.get("bid")
    ask = contract.get("ask")
    if bid is not None and ask is not None:
        try:
            b, a = float(bid), float(ask)
            if math.isfinite(b) and math.isfinite(a):
                return a - b
        except (TypeError, ValueError):
            pass
    return math.nan


def _compute_ticker_gex(
    chain_map: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]],
    spot: float,
    target_expiry_str: str,
) -> Dict[str, Any]:
    """Compute net GEX and GEX walls from option chain data.

    GEX = gamma x openInterest x 100 x spot
    Calls contribute positive GEX (dealers long gamma).
    Puts contribute negative GEX (dealers short gamma on hedge side).
    Net GEX > 0 => mean-reverting ("pinned"), < 0 => trending ("volatile").
    """
    exp_data = chain_map.get(target_expiry_str)
    if not exp_data or spot <= 0:
        return {"net_gex": math.nan, "gex_regime": "", "gex_support": math.nan, "gex_resistance": math.nan}

    call_contracts = exp_data.get("C", {})
    put_contracts = exp_data.get("P", {})

    total_call_gex = 0.0
    total_put_gex = 0.0
    call_gex_by_strike: Dict[float, float] = {}
    put_gex_by_strike: Dict[float, float] = {}

    for strike, contract in call_contracts.items():
        gamma = contract.get("gamma")
        oi = contract.get("openInterest")
        if gamma is not None and oi is not None:
            try:
                g, o = float(gamma), float(oi)
                if math.isfinite(g) and math.isfinite(o) and g >= 0 and o >= 0:
                    gex = g * o * 100.0 * spot
                    total_call_gex += gex
                    call_gex_by_strike[strike] = gex
            except (TypeError, ValueError):
                pass

    for strike, contract in put_contracts.items():
        gamma = contract.get("gamma")
        oi = contract.get("openInterest")
        if gamma is not None and oi is not None:
            try:
                g, o = float(gamma), float(oi)
                if math.isfinite(g) and math.isfinite(o) and g >= 0 and o >= 0:
                    gex = g * o * 100.0 * spot
                    total_put_gex += gex
                    put_gex_by_strike[strike] = gex
            except (TypeError, ValueError):
                pass

    net_gex = total_call_gex - total_put_gex
    gex_regime = "pinned" if net_gex >= 0 else "volatile"

    # GEX walls: strikes with highest gamma concentration
    gex_support = math.nan
    gex_resistance = math.nan

    put_below = {k: v for k, v in put_gex_by_strike.items() if k < spot}
    if put_below:
        gex_support = max(put_below, key=put_below.get)

    call_above = {k: v for k, v in call_gex_by_strike.items() if k > spot}
    if call_above:
        gex_resistance = max(call_above, key=call_above.get)

    return {
        "net_gex": round(net_gex, 2),
        "gex_regime": gex_regime,
        "gex_support": gex_support,
        "gex_resistance": gex_resistance,
    }


def validate_with_schwab(
    scores: List[SwingScore],
    signals_map: Dict[str, SwingSignals],
    cfg: Dict,
) -> None:
    """Validate shortlisted trades against live Schwab option chains.

    For each scored trade:
    1. Fetch the option chain for the ticker around target_expiry
    2. Snap long/short strikes to nearest real strikes
    3. Compute live mid-price spread cost
    4. Report bid/ask width for liquidity assessment

    Modifies SwingScore objects in-place with live_* fields.
    """
    schwab_cfg = cfg.get("schwab_validation", {})
    if not schwab_cfg.get("enabled", False):
        return

    try:
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    except ImportError:
        print("  [schwab] schwab_auth not available, skipping validation", file=sys.stderr)
        return

    print("  [schwab] Connecting to Schwab API for live validation...", file=sys.stderr)
    try:
        config = SchwabAuthConfig.from_env(load_dotenv_file=True)
        service = SchwabLiveDataService(config=config, interactive_login=False)
        service.connect()
    except Exception as exc:
        print(f"  [schwab] Connection failed: {exc}", file=sys.stderr)
        for s in scores:
            s.live_validated = False
            s.live_validation_note = f"Schwab connection failed: {exc}"
        return

    max_snap_pct = float(schwab_cfg.get("max_strike_snap_pct", 3.0))
    # Width-based entry gate tolerance
    entry_tol_width_pct = float(schwab_cfg.get("entry_tolerance_width_pct", 0.025))
    entry_tol_floor = float(schwab_cfg.get("entry_tolerance_floor", 0.25))

    # Group scores by ticker to minimize API calls
    ticker_scores: Dict[str, List[SwingScore]] = defaultdict(list)
    for s in scores:
        ticker_scores[s.ticker].append(s)

    print(f"  [schwab] Validating {len(scores)} trades across {len(ticker_scores)} tickers...", file=sys.stderr)

    for ticker, ticker_score_list in ticker_scores.items():
        # Determine expiry range across all scores for this ticker
        expiry_dates = []
        for s in ticker_score_list:
            try:
                expiry_dates.append(dt.date.fromisoformat(s.target_expiry))
            except (ValueError, TypeError):
                pass

        if not expiry_dates:
            for s in ticker_score_list:
                s.live_validated = False
                s.live_validation_note = "No valid target_expiry"
            continue

        from_date = min(expiry_dates) - dt.timedelta(days=3)
        to_date = max(expiry_dates) + dt.timedelta(days=3)

        # Fetch chain
        try:
            chain = service.get_option_chain(
                symbol=ticker,
                strike_count=None,  # All strikes
                include_underlying_quote=True,
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as exc:
            for s in ticker_score_list:
                s.live_validated = False
                s.live_validation_note = f"Chain fetch failed: {exc}"
            continue

        status = chain.get("status", "UNKNOWN")
        if status != "SUCCESS":
            for s in ticker_score_list:
                s.live_validated = False
                s.live_validation_note = f"Chain status: {status}"
            continue

        # Extract underlying price
        underlying = chain.get("underlying", {})
        spot = None
        for field_name in ("mark", "last", "close"):
            v = underlying.get(field_name)
            if v is not None:
                try:
                    fv = float(v)
                    if math.isfinite(fv) and fv > 0:
                        spot = fv
                        break
                except (TypeError, ValueError):
                    pass

        # Build strike→contract maps by expiry and right (C/P)
        # Structure: {expiry_str: {"C": {strike: contract}, "P": {strike: contract}}}
        chain_map: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = {}
        for map_name, right in [("callExpDateMap", "C"), ("putExpDateMap", "P")]:
            exp_map = chain.get(map_name, {}) or {}
            for exp_key, strike_map in exp_map.items():
                exp_date_str = exp_key.split(":")[0]
                if exp_date_str not in chain_map:
                    chain_map[exp_date_str] = {"C": {}, "P": {}}
                for strike_key, contracts in strike_map.items():
                    try:
                        strike = float(strike_key)
                    except (TypeError, ValueError):
                        continue
                    if contracts:
                        chain_map[exp_date_str][right][strike] = contracts[0]

        # Compute GEX for this ticker (use the nearest expiry to first score's target)
        gex_expiry = best_exp_str_for_gex = None
        for s_gex in ticker_score_list:
            try:
                t_exp = dt.date.fromisoformat(s_gex.target_expiry)
                for exp_str in chain_map:
                    try:
                        exp_d = dt.date.fromisoformat(exp_str)
                        if abs((exp_d - t_exp).days) <= 7:
                            gex_expiry = exp_str
                            break
                    except (ValueError, TypeError):
                        continue
                if gex_expiry:
                    break
            except (ValueError, TypeError):
                continue

        ticker_gex = {"net_gex": math.nan, "gex_regime": "", "gex_support": math.nan, "gex_resistance": math.nan}
        if gex_expiry and spot is not None:
            ticker_gex = _compute_ticker_gex(chain_map, spot, gex_expiry)

        # Now validate each score for this ticker
        for s in ticker_score_list:
            if spot is not None:
                s.live_spot = spot

            s.net_gex = ticker_gex["net_gex"]
            s.gex_regime = ticker_gex["gex_regime"]
            s.gex_support = ticker_gex["gex_support"]
            s.gex_resistance = ticker_gex["gex_resistance"]

            strat = s.recommended_strategy
            is_call = strat in ("Bull Call Debit", "Bear Call Credit")
            is_put = strat in ("Bear Put Debit", "Bull Put Credit")
            is_ic = strat in ("Iron Condor", "Iron Butterfly", "Long Iron Condor")
            if not (is_call or is_put or is_ic):
                s.live_validated = False
                s.live_validation_note = f"Unknown strategy: {strat}"
                continue

            # Find the closest expiry in the chain to our target
            try:
                target_exp = dt.date.fromisoformat(s.target_expiry)
            except (ValueError, TypeError):
                s.live_validated = False
                s.live_validation_note = "Invalid target_expiry"
                continue

            best_exp_str = None
            best_exp_delta = 999
            for exp_str in chain_map:
                try:
                    exp_d = dt.date.fromisoformat(exp_str)
                    delta = abs((exp_d - target_exp).days)
                    if delta < best_exp_delta:
                        best_exp_delta = delta
                        best_exp_str = exp_str
                except (ValueError, TypeError):
                    continue

            if best_exp_str is None or best_exp_delta > 7:
                s.live_validated = False
                s.live_validation_note = f"No chain expiry within 7d of {s.target_expiry}"
                continue

            # --- IC / IB / Long IC: 4-leg validation ---
            if is_ic:
                put_contracts = chain_map.get(best_exp_str, {}).get("P", {})
                call_contracts = chain_map.get(best_exp_str, {}).get("C", {})
                if not put_contracts or not call_contracts:
                    s.live_validated = False
                    s.live_validation_note = f"Missing put or call chain for {best_exp_str}"
                    continue

                put_strikes_avail = sorted(put_contracts.keys())
                call_strikes_avail = sorted(call_contracts.keys())

                # For IC: short_strike = put_short, long_strike = call_short
                # Reconstruct wing strikes from spread_width
                sw = s.spread_width
                if not math.isfinite(sw) or sw <= 0:
                    s.live_validated = False
                    s.live_validation_note = "spread_width is NaN/invalid; cannot reconstruct IC wings"
                    continue
                put_short_target = s.short_strike           # sell put
                put_long_target = s.short_strike - sw       # buy put (wing)
                call_short_target = s.long_strike           # sell call
                call_long_target = s.long_strike + sw       # buy call (wing)

                snap_ps = _find_nearest_strike(put_strikes_avail, put_short_target)
                snap_pl = _find_nearest_strike(put_strikes_avail, put_long_target)
                snap_cs = _find_nearest_strike(call_strikes_avail, call_short_target)
                snap_cl = _find_nearest_strike(call_strikes_avail, call_long_target)

                if any(v is None for v in (snap_ps, snap_pl, snap_cs, snap_cl)):
                    s.live_validated = False
                    s.live_validation_note = "Could not snap all 4 IC strikes to chain"
                    continue

                # Check snap distances
                targets = [put_short_target, put_long_target, call_short_target, call_long_target]
                snapped = [snap_ps, snap_pl, snap_cs, snap_cl]
                labels = ["put_short", "put_long", "call_short", "call_long"]
                snap_ok = True
                for tgt, snp, lbl in zip(targets, snapped, labels):
                    pct = abs(snp - tgt) / max(abs(tgt), 1.0) * 100
                    if pct > max_snap_pct:
                        s.live_validated = False
                        s.live_validation_note = f"IC {lbl} snap too far: {tgt:g}->{snp:g} ({pct:.1f}%)"
                        snap_ok = False
                        break
                if not snap_ok:
                    continue

                # Ordering: put_long < put_short < call_short < call_long
                if not (snap_pl < snap_ps < snap_cs < snap_cl):
                    s.live_validated = False
                    s.live_validation_note = (
                        f"IC strikes out of order: PL={snap_pl:g} PS={snap_ps:g} "
                        f"CS={snap_cs:g} CL={snap_cl:g}"
                    )
                    continue

                # Store inner strikes (short_strike=put_short, long_strike=call_short)
                s.live_short_strike = snap_ps
                s.live_long_strike = snap_cs

                # Compute 4-leg cost: credit = (put_short_mid - put_long_mid) + (call_short_mid - call_long_mid)
                ps_mid = _contract_mid_price(put_contracts.get(snap_ps, {}))
                pl_mid = _contract_mid_price(put_contracts.get(snap_pl, {}))
                cs_mid = _contract_mid_price(call_contracts.get(snap_cs, {}))
                cl_mid = _contract_mid_price(call_contracts.get(snap_cl, {}))

                if any(v is None for v in (ps_mid, pl_mid, cs_mid, cl_mid)):
                    s.live_validated = False
                    s.live_validation_note = "Missing bid/ask quotes on one or more IC legs"
                    continue

                if strat == "Long Iron Condor":
                    # Debit: buy the inner, sell the outer
                    net_cost = round((pl_mid - ps_mid) + (cl_mid - cs_mid), 2)
                    s.live_spread_cost = net_cost
                else:
                    # Credit IC/IB: sell inner, buy outer
                    net_credit = round((ps_mid - pl_mid) + (cs_mid - cl_mid), 2)
                    s.live_spread_cost = net_credit

                if s.live_spread_cost < 0:
                    s.live_validated = False
                    s.live_validation_note = f"Negative IC spread cost: ${s.live_spread_cost:.2f}"
                    continue

                # Bid/ask width (avg of all 4 legs)
                ba_vals = [_contract_bid_ask_spread(put_contracts.get(snap_ps, {})),
                           _contract_bid_ask_spread(put_contracts.get(snap_pl, {})),
                           _contract_bid_ask_spread(call_contracts.get(snap_cs, {})),
                           _contract_bid_ask_spread(call_contracts.get(snap_cl, {}))]
                finite_ba = [v for v in ba_vals if math.isfinite(v)]
                if finite_ba:
                    s.live_bid_ask_width = round(sum(finite_ba) / len(finite_ba), 2)

                put_w = round(snap_ps - snap_pl, 2)
                call_w = round(snap_cl - snap_cs, 2)
                cost_label = "credit" if strat != "Long Iron Condor" else "debit"
                s.live_strike_setup = (
                    f"Sell {snap_ps:g}P / Buy {snap_pl:g}P + "
                    f"Sell {snap_cs:g}C / Buy {snap_cl:g}C "
                    f"({put_w:g}pw/{call_w:g}cw, ${s.live_spread_cost:.2f} {cost_label})"
                )
                # Width-based entry gate for IC
                if math.isfinite(s.est_cost) and s.est_cost > 0:
                    ic_width = max(put_w, call_w)
                    gate_tol = max(entry_tol_floor, ic_width * entry_tol_width_pct)
                    if s.cost_type == "debit":
                        cost_miss = s.live_spread_cost - s.est_cost
                    else:
                        cost_miss = s.est_cost - s.live_spread_cost
                    if cost_miss > gate_tol:
                        s.live_validation_note = (
                            f"IC entry gate miss: live ${s.live_spread_cost:.2f} vs est ${s.est_cost:.2f} "
                            f"(miss ${cost_miss:.2f} > tol ${gate_tol:.2f})"
                        )
                        s.live_validated = False
                        continue

                # Extract live deltas for IC short legs
                ps_contract = put_contracts.get(snap_ps, {})
                cs_contract = call_contracts.get(snap_cs, {})
                ps_delta = ps_contract.get("delta")
                cs_delta = cs_contract.get("delta")
                if ps_delta is not None:
                    try:
                        s.short_put_delta_live = float(ps_delta)
                    except (TypeError, ValueError):
                        pass
                if cs_delta is not None:
                    try:
                        s.short_call_delta_live = float(cs_delta)
                    except (TypeError, ValueError):
                        pass
                s.live_validated = True
                continue

            # --- 2-leg vertical validation ---
            right = "C" if is_call else "P"

            strike_contracts = chain_map.get(best_exp_str, {}).get(right, {})
            if not strike_contracts:
                s.live_validated = False
                s.live_validation_note = f"No {right} contracts for {best_exp_str}"
                continue

            available_strikes = sorted(strike_contracts.keys())

            # Snap strikes to nearest available
            snapped_long = _find_nearest_strike(available_strikes, s.long_strike)
            snapped_short = _find_nearest_strike(available_strikes, s.short_strike)

            if snapped_long is None or snapped_short is None:
                s.live_validated = False
                s.live_validation_note = "Could not snap strikes to chain"
                continue

            # Check snap distance isn't too far
            long_snap_pct = abs(snapped_long - s.long_strike) / max(s.long_strike, 1.0) * 100
            short_snap_pct = abs(snapped_short - s.short_strike) / max(s.short_strike, 1.0) * 100
            if long_snap_pct > max_snap_pct or short_snap_pct > max_snap_pct:
                s.live_validated = False
                s.live_validation_note = (
                    f"Strike snap too far: long {s.long_strike:g}->{snapped_long:g} "
                    f"({long_snap_pct:.1f}%), short {s.short_strike:g}->{snapped_short:g} "
                    f"({short_snap_pct:.1f}%); max={max_snap_pct}%"
                )
                continue

            # Make sure snapped strikes maintain correct ordering
            if strat in ("Bull Call Debit", "Bear Call Credit"):
                if snapped_long >= snapped_short and strat == "Bull Call Debit":
                    s.live_validated = False
                    s.live_validation_note = f"Snapped strikes inverted: long={snapped_long:g} >= short={snapped_short:g}"
                    continue
                if snapped_short >= snapped_long and strat == "Bear Call Credit":
                    s.live_validated = False
                    s.live_validation_note = f"Snapped strikes inverted: short={snapped_short:g} >= long={snapped_long:g}"
                    continue
            elif strat in ("Bear Put Debit", "Bull Put Credit"):
                if snapped_long <= snapped_short and strat == "Bear Put Debit":
                    s.live_validated = False
                    s.live_validation_note = f"Snapped strikes inverted: long={snapped_long:g} <= short={snapped_short:g}"
                    continue
                if snapped_short <= snapped_long and strat == "Bull Put Credit":
                    s.live_validated = False
                    s.live_validation_note = f"Snapped strikes inverted: short={snapped_short:g} <= long={snapped_long:g}"
                    continue

            s.live_long_strike = snapped_long
            s.live_short_strike = snapped_short

            # Fetch contract data for both legs
            long_contract = strike_contracts.get(snapped_long, {})
            short_contract = strike_contracts.get(snapped_short, {})

            long_mid = _contract_mid_price(long_contract)
            short_mid = _contract_mid_price(short_contract)

            # Compute spread cost
            if long_mid is not None and short_mid is not None:
                if s.cost_type == "debit":
                    s.live_spread_cost = round(long_mid - short_mid, 2)
                else:
                    s.live_spread_cost = round(short_mid - long_mid, 2)

                # Sanity: cost should be positive
                if s.live_spread_cost < 0:
                    s.live_validation_note = (
                        f"Negative spread cost: ${s.live_spread_cost:.2f} "
                        f"(long mid=${long_mid:.2f}, short mid=${short_mid:.2f})"
                    )
                    s.live_validated = False
                    continue

                # Check cost doesn't exceed spread width
                live_width = abs(snapped_long - snapped_short)
                if s.cost_type == "debit" and s.live_spread_cost > live_width:
                    s.live_validation_note = (
                        f"Debit ${s.live_spread_cost:.2f} exceeds width ${live_width:g}"
                    )
                    s.live_validated = False
                    continue
            else:
                s.live_validation_note = "Missing bid/ask quotes on one or both legs"
                s.live_validated = False
                continue

            # Bid/ask width (avg of both legs for liquidity measure)
            long_ba = _contract_bid_ask_spread(long_contract)
            short_ba = _contract_bid_ask_spread(short_contract)
            if math.isfinite(long_ba) and math.isfinite(short_ba):
                s.live_bid_ask_width = round((long_ba + short_ba) / 2, 2)
            elif math.isfinite(long_ba):
                s.live_bid_ask_width = round(long_ba, 2)
            elif math.isfinite(short_ba):
                s.live_bid_ask_width = round(short_ba, 2)

            # Build live strike setup string
            live_width = abs(snapped_long - snapped_short)
            right_char = "C" if is_call else "P"
            if s.cost_type == "debit":
                s.live_strike_setup = (
                    f"Buy {snapped_long:g}{right_char} / Sell {snapped_short:g}{right_char} "
                    f"({live_width:g}w, ${s.live_spread_cost:.2f} debit)"
                )
            else:
                s.live_strike_setup = (
                    f"Sell {snapped_short:g}{right_char} / Buy {snapped_long:g}{right_char} "
                    f"({live_width:g}w, ${s.live_spread_cost:.2f} credit)"
                )

            # Width-based entry gate: live cost must be within tolerance of estimated cost
            if math.isfinite(s.est_cost) and s.est_cost > 0:
                gate_tol = max(entry_tol_floor, live_width * entry_tol_width_pct)
                if s.cost_type == "debit":
                    cost_miss = s.live_spread_cost - s.est_cost
                else:
                    cost_miss = s.est_cost - s.live_spread_cost
                if cost_miss > gate_tol:
                    s.live_validation_note = (
                        f"Entry gate miss: live ${s.live_spread_cost:.2f} vs est ${s.est_cost:.2f} "
                        f"(miss ${cost_miss:.2f} > tol ${gate_tol:.2f} = max(${entry_tol_floor}, "
                        f"{live_width:g}×{entry_tol_width_pct}))"
                    )
                    s.live_validated = False
                    continue

            # Extract live deltas for vertical legs
            short_delta_raw = short_contract.get("delta")
            long_delta_raw = long_contract.get("delta")
            if short_delta_raw is not None:
                try:
                    s.short_delta_live = float(short_delta_raw)
                except (TypeError, ValueError):
                    pass
            if long_delta_raw is not None:
                try:
                    s.long_delta_live = float(long_delta_raw)
                except (TypeError, ValueError):
                    pass
            s.live_validated = True

    validated = sum(1 for s in scores if s.live_validated is True)
    failed = sum(1 for s in scores if s.live_validated is False)
    print(f"  [schwab] Validation complete: {validated} valid, {failed} failed", file=sys.stderr)


# ---------------------------------------------------------------------------
# Historical backtest (edge)
# ---------------------------------------------------------------------------

def run_backtest(
    scores: List[SwingScore],
    signals_map: Dict[str, SwingSignals],
    cfg: Dict,
    out_dir: Path,
    as_of: dt.date,
    root: Path,
) -> None:
    """Run setup_likelihood_backtest on shortlisted trades and merge results back.

    Generates a backtest-compatible CSV, calls the backtest as a subprocess,
    reads back edge/verdict, and populates SwingScore.edge_* fields in-place.
    """
    import subprocess

    bt_cfg = cfg.get("backtest", {})
    if not bt_cfg.get("enabled", False):
        return

    if not scores:
        return

    print("  [backtest] Building backtest setups...", file=sys.stderr)

    # Build a backtest-compatible CSV
    def _entry_gate_for_score(score: SwingScore, cost: float) -> str:
        if score.cost_type == "credit":
            return f">= {cost:.2f} cr"
        return f"<= {cost:.2f} db"

    def _setup_id_for_score(
        score: SwingScore,
        *,
        entry_gate: str,
        long_strike: float,
        short_strike: float,
        width: float,
    ) -> str:
        return "|".join(
            [
                str(score.ticker).upper().strip(),
                str(score.recommended_strategy).strip(),
                str(score.target_expiry).strip(),
                f"{long_strike:.4f}",
                f"{short_strike:.4f}",
                f"{width:.4f}",
                str(entry_gate).strip(),
            ]
        )

    score_keys: Dict[int, str] = {}
    bt_rows = []
    for s in scores:
        sig = signals_map.get(s.ticker, SwingSignals())
        # Use live cost if validated, else heuristic est_cost
        cost = s.live_spread_cost if (s.live_validated is True and math.isfinite(s.live_spread_cost)) else s.est_cost
        if not math.isfinite(cost) or cost <= 0:
            continue
        # Use live strikes if validated, else heuristic strikes
        long_k = s.live_long_strike if (s.live_validated is True and math.isfinite(s.live_long_strike)) else s.long_strike
        short_k = s.live_short_strike if (s.live_validated is True and math.isfinite(s.live_short_strike)) else s.short_strike
        if not (math.isfinite(long_k) and math.isfinite(short_k)):
            continue

        _is_ic = s.recommended_strategy in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}
        if _is_ic:
            # For IC: short_strike=put_short, long_strike=call_short (inner strikes)
            # spread_width is the per-side width, NOT the body width
            # Do NOT fall back to abs(long_k - short_k) which is the body width
            width = s.spread_width if math.isfinite(s.spread_width) else math.nan
            if not math.isfinite(width) or width <= 0:
                continue
        else:
            width = abs(long_k - short_k)
        if width <= 0:
            continue

        entry_gate = _entry_gate_for_score(s, cost)
        setup_id = _setup_id_for_score(
            s,
            entry_gate=entry_gate,
            long_strike=float(long_k),
            short_strike=float(short_k),
            width=float(width),
        )
        score_keys[id(s)] = setup_id

        row = {
            "setup_id": setup_id,
            "ticker": s.ticker,
            "strategy": s.recommended_strategy,
            "expiry": s.target_expiry,
            "long_strike": round(long_k, 2),
            "short_strike": round(short_k, 2),
            "width": round(width, 2),
            "entry_gate": entry_gate,
            "_sort_score": s.composite_score,
            "_sort_live": 1 if s.live_validated is True else 0,
        }
        if _is_ic:
            # IC needs extra columns for breakeven_levels and required_win_rate_pct
            # short_strike = put_short, long_strike = call_short (stored in score)
            row["short_call_strike"] = round(long_k, 2)       # call_short
            row["long_call_strike"] = round(long_k + width, 2) # call_long (wing)
            row["put_width"] = round(width, 2)
            row["call_width"] = round(width, 2)
        bt_rows.append(row)

    if not bt_rows:
        print("  [backtest] No valid setups to backtest", file=sys.stderr)
        return

    bt_df = pd.DataFrame(bt_rows)
    max_setups = int(bt_cfg.get("max_setups", 160) or 0)
    if max_setups > 0 and len(bt_df) > max_setups:
        bt_df = (
            bt_df.sort_values(["_sort_live", "_sort_score"], ascending=[False, False])
            .head(max_setups)
            .copy()
        )
        kept_ids = set(bt_df["setup_id"].astype(str))
        score_keys = {
            score_obj_id: setup_id
            for score_obj_id, setup_id in score_keys.items()
            if setup_id in kept_ids
        }
        print(
            f"  [backtest] Capped setup batch to top {len(bt_df)} by score/live validation",
            file=sys.stderr,
        )
    bt_df = bt_df.drop(columns=["_sort_score", "_sort_live"], errors="ignore")
    bt_input_csv = out_dir / "_backtest_setups_tmp.csv"
    bt_df.to_csv(bt_input_csv, index=False)

    lookback_years = float(bt_cfg.get("lookback_years", 2.0))
    min_signals = int(bt_cfg.get("min_signals", 100))
    cache_dir = bt_cfg.get("cache_dir", "")

    cmd = [
        sys.executable, "-m", "uwos.setup_likelihood_backtest",
        "--setups-csv", str(bt_input_csv),
        "--asof-date", as_of.isoformat(),
        "--root-dir", str(root),
        "--lookback-years", str(lookback_years),
        "--min-signals", str(min_signals),
        "--out-dir", str(out_dir),
    ]
    if cache_dir:
        cmd.extend(["--cache-dir", str(cache_dir)])

    print(f"  [backtest] Running backtest on {len(bt_df)} setups (~{lookback_years}yr lookback)...", file=sys.stderr)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"  [backtest] Failed (exit {result.returncode}): {result.stderr[:200]}", file=sys.stderr)
            return
    except subprocess.TimeoutExpired:
        print("  [backtest] Timed out after 300s", file=sys.stderr)
        return
    except Exception as exc:
        print(f"  [backtest] Error: {exc}", file=sys.stderr)
        return

    # Read back results
    likelihood_csv = out_dir / f"setup_likelihood_{as_of.isoformat()}.csv"
    if not likelihood_csv.exists():
        print(f"  [backtest] Output not found: {likelihood_csv}", file=sys.stderr)
        return

    like_df = pd.read_csv(likelihood_csv, low_memory=False)
    like_df["ticker"] = like_df["ticker"].astype(str).str.upper().str.strip()
    like_df["strategy"] = like_df["strategy"].astype(str).str.strip()

    # Build lookup per concrete setup. Ticker+strategy alone is unsafe once the
    # optimizer tests multiple expiries/strikes for the same symbol.
    like_map: Dict[str, Dict[str, Any]] = {}
    fallback_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for _, row in like_df.iterrows():
        setup_id = str(row.get("setup_id", "") or "").strip()
        if setup_id and setup_id not in like_map:
            like_map[setup_id] = row.to_dict()
        fallback_key = (
            str(row["ticker"]).strip(),
            str(row["strategy"]).strip(),
            str(row.get("expiry", "")).strip(),
            str(row.get("entry_gate", "")).strip(),
        )
        if fallback_key not in fallback_map:
            fallback_map[fallback_key] = row.to_dict()

    # Merge into SwingScore objects
    merged = 0
    for s in scores:
        setup_id = score_keys.get(id(s))
        if setup_id is None:
            continue
        row = like_map.get(setup_id)
        if row is None:
            # Compatibility with older likelihood CSVs that did not preserve
            # setup_id. This is less precise and should only be a fallback.
            cost = s.live_spread_cost if (s.live_validated is True and math.isfinite(s.live_spread_cost)) else s.est_cost
            if not math.isfinite(cost) or cost <= 0:
                continue
            entry_gate = _entry_gate_for_score(s, cost)
            fallback_key = (s.ticker, s.recommended_strategy, s.target_expiry, entry_gate)
            row = fallback_map.get(fallback_key)
        if row is None:
            continue
        s.hist_success_pct = _fnum(row.get("hist_success_pct"))
        s.required_win_pct = _fnum(row.get("required_win_pct"))
        s.edge_pct = _fnum(row.get("edge_pct"))
        s.backtest_signals = int(_fnum(row.get("signals", 0)))
        s.backtest_verdict = str(row.get("verdict", "")).strip()
        s.backtest_confidence = str(row.get("confidence", "")).strip()
        merged += 1

    # Clean up temp file
    try:
        bt_input_csv.unlink()
    except OSError:
        pass

    passed = sum(1 for s in scores if s.backtest_verdict == "PASS")
    failed = sum(1 for s in scores if s.backtest_verdict == "FAIL")
    print(f"  [backtest] Merged {merged} results: {passed} PASS, {failed} FAIL", file=sys.stderr)


# ---------------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------------

def _fmt_pct(v: float) -> str:
    return f"{v:.0%}" if math.isfinite(v) else "-"


def _fmt_f1(v: float) -> str:
    return f"{v:.1f}" if math.isfinite(v) else "-"


def _fmt_f2(v: float) -> str:
    return f"{v:.2f}" if math.isfinite(v) else "-"


def _fmt_money(v: float) -> str:
    if not math.isfinite(v):
        return "-"
    return f"${v:,.2f}"


def _render_md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a simple markdown table."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [str(c).replace("|", "\\|") for c in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_report_markdown(
    scores: List[SwingScore],
    signals_map: Dict[str, SwingSignals],
    date_range: Tuple[dt.date, dt.date],
    n_tickers_analyzed: int,
    cfg: Dict,
    report_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    sector_overflow: Optional[List[SwingScore]] = None,
) -> str:
    """Generate the markdown swing trend report."""
    output_cfg = cfg.get("output", {})
    evidence_top_n = int(output_cfg.get("evidence_top_n", 10))

    lines: List[str] = []
    start, end = date_range
    lines.append(f"# Swing Trend Analysis Report ({start} to {end})")
    lines.append("")

    # VS Code file links
    if report_path or csv_path:
        lines.append("**Open in VS Code:**")
        if report_path:
            lines.append(f"- Report: `code \"{report_path}\"`")
        if csv_path:
            lines.append(f"- Shortlist: `code \"{csv_path}\"`")
        lines.append("")

    lines.append(f"Lookback: {len(signals_map)} tickers scored over {(end - start).days + 1} calendar days")
    lines.append(f"Tickers in screener universe: {n_tickers_analyzed}")
    lines.append(f"Recommendations: {len(scores)}")
    lines.append("")

    # Market regime summary
    all_iv_levels = [signals_map[s.ticker].iv_level for s in scores if s.ticker in signals_map]
    all_directions = [s.direction for s in scores]
    bull_count = sum(1 for d in all_directions if d == "bullish")
    bear_count = sum(1 for d in all_directions if d == "bearish")
    neutral_count = sum(1 for d in all_directions if d == "neutral")
    high_iv_count = sum(1 for l in all_iv_levels if l == "high")

    lines.append("## Market Regime Summary")
    lines.append(f"- Directional split: Bullish={bull_count}, Bearish={bear_count}, Neutral={neutral_count}")
    lines.append(f"- High-IV tickers: {high_iv_count} of {len(scores)}")
    lines.append("")

    # Group scores by category
    fire_bull = [s for s in scores if s.direction == "bullish" and s.recommended_track == "FIRE"]
    fire_bear = [s for s in scores if s.direction == "bearish" and s.recommended_track == "FIRE"]
    shield = [s for s in scores if s.recommended_track == "SHIELD"]

    has_backtest = any(s.backtest_verdict for s in scores)

    def _trade_table(title: str, items: List[SwingScore], track_icon: str) -> None:
        lines.append(f"## {title}")
        if not items:
            lines.append("_none_")
            lines.append("")
            return
        headers = [
            "#", "Ticker", "Score", "Strategy", "Strike Setup", "Expiry",
            "Track", "Days", "Flow", "OI", "IV", "Price",
            "Whale", "DP", "Confidence",
            "IV Rank", "Latest Close", "Sector",
        ]
        if has_backtest:
            headers.extend(["Edge", "Verdict"])
        headers.append("Thesis")
        rows = []
        for i, s in enumerate(items, 1):
            sig = signals_map.get(s.ticker, SwingSignals())
            ticker_display = s.ticker
            if not s.earnings_safe:
                ticker_display += " \u26a0\ufe0f"
            if s.live_validated is False:
                ticker_display += " \u274c"
            # Prefer live strike setup when Schwab validation succeeded
            setup_display = s.strike_setup or "-"
            if s.live_validated is True and s.live_strike_setup:
                setup_display = s.live_strike_setup
            row = [
                str(i),
                ticker_display,
                _fmt_f1(s.composite_score),
                s.recommended_strategy,
                setup_display,
                s.target_expiry or "-",
                f"{track_icon} {s.recommended_track}",
                str(sig.n_days_observed),
                _fmt_f1(s.flow_persistence_score),
                _fmt_f1(s.oi_momentum_score),
                _fmt_f1(s.iv_regime_score),
                _fmt_f1(s.price_trend_score),
                _fmt_f1(s.whale_consensus_score),
                _fmt_f1(s.dp_confirmation_score),
                s.confidence_tier,
                _fmt_f1(sig.latest_iv_rank),
                _fmt_money(sig.latest_close),
                sig.sector,
            ]
            if has_backtest:
                edge_str = f"{s.edge_pct:+.1f}%" if math.isfinite(s.edge_pct) else "-"
                row.extend([edge_str, s.backtest_verdict or "-"])
            row.append(s.thesis)
            rows.append(row)
        lines.append(_render_md_table(headers, rows))
        lines.append("")

    _trade_table("Bullish Swing Candidates (FIRE)", fire_bull, "\U0001f525")
    _trade_table("Bearish Swing Candidates (FIRE)", fire_bear, "\U0001f525")
    _trade_table("Range / Theta Plays (SHIELD)", shield, "\U0001f6e1\ufe0f")

    # Sector overflow (good trades that exceeded sector cap)
    if sector_overflow:
        lines.append("## Sector Overflow (Excess Sector Concentration)")
        lines.append(f"_These {len(sector_overflow)} trades scored well but were bumped to maintain sector diversity._")
        lines.append("")
        _trade_table("Sector Overflow Candidates", sector_overflow, "\U0001f4cb")

    # Earnings warnings
    earn_unsafe = [s for s in scores if not s.earnings_safe]
    if earn_unsafe:
        lines.append("## Earnings Warnings")
        lines.append("_These recommended tickers have earnings within the trade window. Consider adjusting DTE or skipping._")
        lines.append("")
        for s in earn_unsafe:
            lines.append(f"- **{s.ticker}**: {s.earnings_label}")
        lines.append("")

    # Live validation summary (if Schwab validation was run)
    any_validated = any(s.live_validated is not None for s in scores)
    if any_validated:
        live_ok = [s for s in scores if s.live_validated is True]
        live_fail = [s for s in scores if s.live_validated is False]
        lines.append("## Schwab Live Validation")
        lines.append(f"- Validated: {len(live_ok)} | Failed: {len(live_fail)} | Total: {len(scores)}")
        lines.append("")
        if live_ok:
            val_headers = ["Ticker", "Strategy", "Live Setup", "Live Cost", "Est Cost", "Bid/Ask Width", "Live Spot"]
            val_rows = []
            for s in live_ok:
                val_rows.append([
                    s.ticker,
                    s.recommended_strategy,
                    s.live_strike_setup or "-",
                    f"${s.live_spread_cost:.2f}" if math.isfinite(s.live_spread_cost) else "-",
                    f"${s.est_cost:.2f}" if math.isfinite(s.est_cost) else "-",
                    f"${s.live_bid_ask_width:.2f}" if math.isfinite(s.live_bid_ask_width) else "-",
                    _fmt_money(s.live_spot),
                ])
            lines.append(_render_md_table(val_headers, val_rows))
            lines.append("")
        if live_fail:
            lines.append("**Failed validations:**")
            for s in live_fail:
                lines.append(f"- **{s.ticker}** ({s.recommended_strategy}): {s.live_validation_note}")
            lines.append("")

    # Backtest edge summary
    if has_backtest:
        bt_scored = [s for s in scores if s.backtest_verdict]
        bt_pass = [s for s in bt_scored if s.backtest_verdict == "PASS"]
        bt_fail = [s for s in bt_scored if s.backtest_verdict == "FAIL"]
        bt_low = [s for s in bt_scored if s.backtest_verdict == "LOW_SAMPLE"]
        lines.append("## Historical Backtest Edge")
        lines.append(f"- PASS: {len(bt_pass)} | FAIL: {len(bt_fail)} | Low Sample: {len(bt_low)} | Total: {len(bt_scored)}")
        lines.append("")
        if bt_scored:
            bt_headers = ["Ticker", "Strategy", "Edge %", "Win Rate", "Req Win %", "Signals", "Verdict", "Confidence"]
            bt_rows = []
            for s in sorted(bt_scored, key=lambda x: x.edge_pct if math.isfinite(x.edge_pct) else -999, reverse=True):
                bt_rows.append([
                    s.ticker,
                    s.recommended_strategy,
                    f"{s.edge_pct:+.1f}%" if math.isfinite(s.edge_pct) else "-",
                    f"{s.hist_success_pct:.1f}%" if math.isfinite(s.hist_success_pct) else "-",
                    f"{s.required_win_pct:.1f}%" if math.isfinite(s.required_win_pct) else "-",
                    str(s.backtest_signals),
                    s.backtest_verdict,
                    s.backtest_confidence,
                ])
            lines.append(_render_md_table(bt_headers, bt_rows))
            lines.append("")

    # Multi-day evidence for top tickers
    if output_cfg.get("include_evidence_table", True):
        lines.append("## Multi-Day Evidence (Top Tickers)")
        lines.append("")
        for s in scores[:evidence_top_n]:
            sig = signals_map.get(s.ticker)
            if sig is None:
                continue
            dir_label = s.direction.capitalize()
            lines.append(f"### {s.ticker} (Score: {_fmt_f1(s.composite_score)}, {dir_label})")
            lines.append(f"- Price trend: {sig.price_direction} (slope={_fmt_f2(sig.price_slope)}, R\u00b2={_fmt_f2(sig.price_r_squared)})")
            lines.append(f"- IV regime: {sig.iv_regime} (iv30d slope={_fmt_f2(sig.iv30d_slope)}, rank={_fmt_f1(sig.latest_iv_rank)}, level={sig.iv_level})")
            lines.append(f"- Flow: {sig.flow_direction} (consistency={_fmt_pct(sig.flow_consistency)}, avg bias={_fmt_f2(sig.avg_flow_bias)})")
            lines.append(f"- OI: {sig.oi_direction} (consistency={_fmt_pct(sig.oi_consistency)}, slope={_fmt_f2(sig.oi_momentum_slope)})")
            if math.isfinite(sig.top_call_strike):
                lines.append(f"  - Top call strike cluster: ${sig.top_call_strike:,.0f}")
            if math.isfinite(sig.top_put_strike):
                lines.append(f"  - Top put strike cluster: ${sig.top_put_strike:,.0f}")
            lines.append(f"- Hot chains: sweep ratio={_fmt_f2(sig.avg_sweep_ratio)}, flow consistency={_fmt_pct(sig.hot_flow_consistency)}")
            lines.append(f"- Dark pool: {sig.dp_direction} (consistency={_fmt_pct(sig.dp_consistency)})")
            lines.append(f"- Whale appearances: {sig.whale_appearances} of {sig.n_days_observed} days")
            lines.append(f"- Volume surge days: {sig.volume_surge_days} (avg ratio={_fmt_f2(sig.avg_volume_ratio)})")
            lines.append("")

    # OI strike clustering guidance
    lines.append("## OI Strike Clustering (Target Guidance)")
    strike_headers = ["Ticker", "Direction", "Top Call Strike", "Top Put Strike", "Latest Close"]
    strike_rows = []
    for s in scores[:evidence_top_n]:
        sig = signals_map.get(s.ticker)
        if sig is None:
            continue
        strike_rows.append([
            s.ticker,
            s.direction.capitalize(),
            f"${sig.top_call_strike:,.0f}" if math.isfinite(sig.top_call_strike) else "-",
            f"${sig.top_put_strike:,.0f}" if math.isfinite(sig.top_put_strike) else "-",
            _fmt_money(sig.latest_close),
        ])
    if strike_rows:
        lines.append(_render_md_table(strike_headers, strike_rows))
    else:
        lines.append("_none_")
    lines.append("")

    return "\n".join(lines)


def generate_shortlist_csv(
    scores: List[SwingScore],
    signals_map: Dict[str, SwingSignals],
) -> pd.DataFrame:
    """Generate shortlist CSV compatible with existing pipeline."""
    rows = []
    for s in scores:
        sig = signals_map.get(s.ticker, SwingSignals())
        rows.append({
            "ticker": s.ticker,
            "strategy": s.recommended_strategy,
            "variant_tag": s.variant_tag,
            "repair_source": s.repair_source,
            "strike_setup": s.strike_setup,
            "target_expiry": s.target_expiry,
            "target_dte": s.target_dte,
            "long_strike": round(s.long_strike, 2) if math.isfinite(s.long_strike) else "",
            "short_strike": round(s.short_strike, 2) if math.isfinite(s.short_strike) else "",
            "spread_width": round(s.spread_width, 2) if math.isfinite(s.spread_width) else "",
            "est_cost": round(s.est_cost, 2) if math.isfinite(s.est_cost) else "",
            "cost_type": s.cost_type,
            "track": s.recommended_track,
            "direction": s.direction,
            "swing_score": round(s.composite_score, 1),
            "flow_persistence": round(s.flow_persistence_score, 1),
            "oi_momentum": round(s.oi_momentum_score, 1),
            "iv_regime": round(s.iv_regime_score, 1),
            "price_trend": round(s.price_trend_score, 1),
            "whale_consensus": round(s.whale_consensus_score, 1),
            "dp_confirmation": round(s.dp_confirmation_score, 1),
            "confidence_tier": s.confidence_tier,
            "days_observed": sig.n_days_observed,
            "latest_close": round(sig.latest_close, 2) if math.isfinite(sig.latest_close) else "",
            "latest_iv_rank": round(sig.latest_iv_rank, 1) if math.isfinite(sig.latest_iv_rank) else "",
            "iv_level": sig.iv_level,
            "iv_regime_label": sig.iv_regime,
            "price_direction": sig.price_direction,
            "flow_direction": sig.flow_direction,
            "oi_direction": sig.oi_direction,
            "dp_direction": sig.dp_direction,
            "whale_appearances": sig.whale_appearances,
            "sector": sig.sector,
            "earnings_safe": s.earnings_safe,
            "earnings_label": s.earnings_label,
            "next_earnings_date": sig.next_earnings_date.isoformat() if sig.next_earnings_date else "",
            "thesis": s.thesis,
            # Schwab live validation columns
            "live_validated": "" if s.live_validated is None else s.live_validated,
            "live_spot": round(s.live_spot, 2) if math.isfinite(s.live_spot) else "",
            "live_long_strike": round(s.live_long_strike, 2) if math.isfinite(s.live_long_strike) else "",
            "live_short_strike": round(s.live_short_strike, 2) if math.isfinite(s.live_short_strike) else "",
            "live_spread_cost": round(s.live_spread_cost, 2) if math.isfinite(s.live_spread_cost) else "",
            "live_bid_ask_width": round(s.live_bid_ask_width, 2) if math.isfinite(s.live_bid_ask_width) else "",
            "live_strike_setup": s.live_strike_setup,
            "live_validation_note": s.live_validation_note,
            # Backtest edge columns
            "hist_success_pct": round(s.hist_success_pct, 1) if math.isfinite(s.hist_success_pct) else "",
            "required_win_pct": round(s.required_win_pct, 1) if math.isfinite(s.required_win_pct) else "",
            "edge_pct": round(s.edge_pct, 1) if math.isfinite(s.edge_pct) else "",
            "backtest_signals": s.backtest_signals if s.backtest_signals > 0 else "",
            "backtest_verdict": s.backtest_verdict,
            "backtest_confidence": s.backtest_confidence,
            # Live greeks
            "short_delta_live": round(s.short_delta_live, 4) if math.isfinite(s.short_delta_live) else "",
            "long_delta_live": round(s.long_delta_live, 4) if math.isfinite(s.long_delta_live) else "",
            "short_put_delta_live": round(s.short_put_delta_live, 4) if math.isfinite(s.short_put_delta_live) else "",
            "short_call_delta_live": round(s.short_call_delta_live, 4) if math.isfinite(s.short_call_delta_live) else "",
            # GEX regime
            "net_gex": round(s.net_gex, 2) if math.isfinite(s.net_gex) else "",
            "gex_regime": s.gex_regime,
            "gex_support": round(s.gex_support, 2) if math.isfinite(s.gex_support) else "",
            "gex_resistance": round(s.gex_resistance, 2) if math.isfinite(s.gex_resistance) else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(
    cfg: Dict,
    root: Path,
    lookback: int,
    as_of: Optional[dt.date],
    out_dir: Path,
    max_recommendations: Optional[int] = None,
) -> Tuple[List[SwingScore], Dict[str, SwingSignals]]:
    """Main pipeline orchestration."""

    pipeline_cfg = cfg.get("pipeline", {})
    min_days = int(pipeline_cfg.get("min_days_required", 3))
    if max_recommendations is None:
        max_recommendations = int(cfg.get("output", {}).get("max_recommendations", 15))
    max_per_ticker = int(cfg.get("output", {}).get("max_per_ticker", 2))
    min_score = float(cfg.get("output", {}).get("min_composite_score", 35))

    # Phase 1: Discover trading days
    print(f"[1/7] Discovering trading days (lookback={lookback}) ...", file=sys.stderr)
    trading_days = discover_trading_days(root, lookback, as_of)
    if len(trading_days) < min_days:
        print(
            f"  Only {len(trading_days)} trading days found, need at least {min_days}. "
            f"Lowering min_days_required to {len(trading_days)}.",
            file=sys.stderr,
        )
        min_days = max(1, len(trading_days))
    dates_str = ", ".join(d.isoformat() for d, _ in trading_days)
    print(f"  Found {len(trading_days)} days: {dates_str}", file=sys.stderr)

    # Phase 2: Load screeners and build ticker universe
    print("[2/7] Loading stock-screener data ...", file=sys.stderr)
    screeners = load_screeners(trading_days)
    print(f"  Loaded screener data for {len(screeners)} days", file=sys.stderr)

    ticker_universe = build_ticker_universe(screeners, cfg)
    print(f"  Ticker universe: {len(ticker_universe)} tickers", file=sys.stderr)

    if not ticker_universe:
        print("  [ERROR] No tickers passed filters. Exiting.", file=sys.stderr)
        return [], {}

    # Phase 3: Load remaining data sources (filtered)
    print("[3/7] Loading chain-oi-changes ...", file=sys.stderr)
    chain_oi = load_chain_oi(trading_days, ticker_universe)
    print(f"  Loaded OI data for {len(chain_oi)} days", file=sys.stderr)

    print("[4/7] Loading hot-chains ...", file=sys.stderr)
    hot_chains = load_hot_chains(trading_days, ticker_universe)
    print(f"  Loaded hot-chains for {len(hot_chains)} days", file=sys.stderr)

    dp_filter = cfg.get("data_loading", {}).get("dp_filter_to_screener_tickers", True)
    print("[5/7] Loading dp-eod-report ...", file=sys.stderr)
    if dp_filter:
        dp_eod = load_dp_eod(trading_days, ticker_universe)
    else:
        dp_eod = {}
    print(f"  Loaded DP data for {len(dp_eod)} days", file=sys.stderr)

    print("  Loading whale mentions ...", file=sys.stderr)
    whale_mentions = load_whale_mentions(trading_days, ticker_universe)
    print(f"  Loaded whale data for {len(whale_mentions)} days", file=sys.stderr)

    # Phase 4: Feature extraction
    print("[6/7] Extracting features and computing signals ...", file=sys.stderr)

    # Build per-ticker, per-day feature series
    ticker_screener: Dict[str, List[Tuple[dt.date, ScreenerFeatures]]] = defaultdict(list)
    ticker_oi: Dict[str, List[Tuple[dt.date, OIFeatures]]] = defaultdict(list)
    ticker_hot: Dict[str, List[Tuple[dt.date, HotChainFeatures]]] = defaultdict(list)
    ticker_dp: Dict[str, List[Tuple[dt.date, DPFeatures]]] = defaultdict(list)
    ticker_whale_count: Dict[str, int] = defaultdict(int)

    for d, df in screeners.items():
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if ticker not in ticker_universe:
                continue
            features = extract_screener_features(row)
            ticker_screener[ticker].append((d, features))

    for d, df in chain_oi.items():
        for ticker in ticker_universe:
            if ticker in df["underlying_symbol"].values:
                # Get spot price from screener for this day if available
                spot = math.nan
                if d in screeners:
                    scr = screeners[d]
                    scr_row = scr[scr["ticker"] == ticker]
                    if not scr_row.empty:
                        spot = _fnum(scr_row.iloc[0].get("close", math.nan))
                features = extract_oi_features(df, ticker, spot=spot)
                ticker_oi[ticker].append((d, features))

    for d, df in hot_chains.items():
        for ticker in ticker_universe:
            if ticker in df["_underlying"].values:
                features = extract_hot_chain_features(df, ticker)
                ticker_hot[ticker].append((d, features))

    for d, df in dp_eod.items():
        for ticker in ticker_universe:
            if "ticker" in df.columns and ticker in df["ticker"].values:
                features = extract_dp_features(df, ticker)
                ticker_dp[ticker].append((d, features))

    for d, tickers in whale_mentions.items():
        for ticker in tickers:
            ticker_whale_count[ticker] += 1

    # Sort each series by date
    for ticker in ticker_universe:
        ticker_screener[ticker].sort(key=lambda x: x[0])
        ticker_oi[ticker].sort(key=lambda x: x[0])
        ticker_hot[ticker].sort(key=lambda x: x[0])
        ticker_dp[ticker].sort(key=lambda x: x[0])

    # Phase 5: Compute swing signals
    signals_map: Dict[str, SwingSignals] = {}
    for ticker in ticker_universe:
        if len(ticker_screener[ticker]) < min_days:
            continue
        signals = compute_swing_signals(
            ticker=ticker,
            screener_series=ticker_screener[ticker],
            oi_series=ticker_oi[ticker],
            hot_series=ticker_hot[ticker],
            dp_series=ticker_dp[ticker],
            whale_days=ticker_whale_count.get(ticker, 0),
            cfg=cfg,
        )
        signals_map[ticker] = signals

    print(f"  Computed signals for {len(signals_map)} tickers", file=sys.stderr)

    # Phase 6: Score and rank
    print("[7/7] Scoring and ranking ...", file=sys.stderr)
    all_scores: List[SwingScore] = []
    for ticker, signals in signals_map.items():
        s = score_ticker(signals, cfg)
        if s.composite_score >= min_score:
            all_scores.append(s)

    # Sort by composite score
    all_scores.sort(key=lambda s: s.composite_score, reverse=True)

    # Apply max_per_ticker cap and soft sector cap
    max_per_sector = int(cfg.get("filters", {}).get("max_per_sector", 99))
    final_scores: List[SwingScore] = []
    sector_overflow: List[SwingScore] = []
    ticker_counts: Dict[str, int] = {}
    sector_counts: Dict[str, int] = defaultdict(int)

    for s in all_scores:
        if ticker_counts.get(s.ticker, 0) >= max_per_ticker:
            continue
        sig = signals_map.get(s.ticker, SwingSignals())
        sector = sig.sector or "Unknown"

        if sector_counts[sector] >= max_per_sector:
            # Sector full — overflow instead of drop
            sector_overflow.append(s)
            ticker_counts[s.ticker] = ticker_counts.get(s.ticker, 0) + 1
        else:
            final_scores.append(s)
            ticker_counts[s.ticker] = ticker_counts.get(s.ticker, 0) + 1
            sector_counts[sector] += 1

        if len(final_scores) >= max_recommendations:
            break

    # If main list is short due to sector caps, backfill from overflow
    while len(final_scores) < max_recommendations and sector_overflow:
        s = sector_overflow.pop(0)
        final_scores.append(s)

    n_overflow = len(sector_overflow)
    print(
        f"  Final recommendations: {len(final_scores)}"
        + (f" (+{n_overflow} sector overflow)" if n_overflow else ""),
        file=sys.stderr,
    )

    # Phase 6b: Generate repaired alternatives before live validation/backtest.
    base_to_validate = final_scores + sector_overflow
    repair_variants = generate_trade_repair_variants(base_to_validate, signals_map, cfg)
    if repair_variants:
        print(f"  Trade repair variants generated: {len(repair_variants)}", file=sys.stderr)

    # Phase 6c: Schwab live validation (optional)
    all_to_validate = base_to_validate + repair_variants
    validate_with_schwab(all_to_validate, signals_map, cfg)

    # Phase 6d: Historical backtest (optional)
    as_of_resolved = trading_days[-1][0]
    run_backtest(all_to_validate, signals_map, cfg, out_dir, as_of_resolved, root)

    # Phase 7: Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    date_range = (trading_days[0][0], trading_days[-1][0])
    as_of_str = trading_days[-1][0].isoformat()

    lb_tag = f"L{lookback}"
    report_name = cfg.get("output", {}).get("report_md_name", "swing-trend-report-{date}.md") \
        .replace("{date}", as_of_str).replace(".md", f"-{lb_tag}.md")
    csv_name = cfg.get("output", {}).get("shortlist_csv_name", "swing_trend_shortlist_{date}.csv") \
        .replace("{date}", as_of_str).replace(".csv", f"-{lb_tag}.csv")

    report_path = out_dir / report_name
    csv_path = out_dir / csv_name
    report_md = generate_report_markdown(
        final_scores, signals_map, date_range, len(ticker_universe), cfg,
        report_path=report_path, csv_path=csv_path,
        sector_overflow=sector_overflow + repair_variants,
    )
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  Report written: {report_path}", file=sys.stderr)

    # CSV includes both main and overflow — marked with sector_overflow column
    all_for_csv = final_scores + sector_overflow + repair_variants
    shortlist_df = generate_shortlist_csv(all_for_csv, signals_map)
    shortlist_df["sector_overflow"] = (
        [False] * len(final_scores)
        + [True] * len(sector_overflow)
        + [False] * len(repair_variants)
    )
    shortlist_df["repair_variant"] = (
        [False] * len(final_scores)
        + [False] * len(sector_overflow)
        + [True] * len(repair_variants)
    )
    shortlist_df.to_csv(csv_path, index=False)
    print(f"  Shortlist CSV written: {csv_path}", file=sys.stderr)

    return final_scores, signals_map


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Multi-day swing trade trend analysis pipeline",
    )
    ap.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parent / "rulebook_config_swing_trend.yaml"),
        help="Path to swing trend YAML config",
    )
    ap.add_argument(
        "--root-dir",
        default=None,
        help="Root directory containing dated folders (overrides config)",
    )
    ap.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Override lookback_days from config",
    )
    ap.add_argument(
        "--as-of",
        default=None,
        help="As-of date YYYY-MM-DD (default: latest available)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Override output directory",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=None,
        help="Override max_recommendations",
    )
    ap.add_argument(
        "--no-schwab",
        action="store_true",
        help="Skip Schwab live validation even if enabled in config",
    )
    ap.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip historical edge backtest even if enabled in config",
    )
    args = ap.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # CLI overrides
    if args.no_schwab:
        cfg.setdefault("schwab_validation", {})["enabled"] = False
    if args.no_backtest:
        cfg.setdefault("backtest", {})["enabled"] = False

    pipeline_cfg = cfg.get("pipeline", {})
    root = Path(args.root_dir or pipeline_cfg.get("root_dir", r"c:\uw_root"))
    lookback = args.lookback or int(pipeline_cfg.get("lookback_days", 10))
    out_dir = Path(args.out_dir or pipeline_cfg.get("output_dir", r"c:\uw_root\out\swing_trend"))

    as_of: Optional[dt.date] = None
    if args.as_of:
        as_of = dt.date.fromisoformat(args.as_of)
        # Roll weekends back to Friday — markets are closed Sat/Sun
        if as_of.weekday() == 5:  # Saturday
            as_of = as_of - dt.timedelta(days=1)
        elif as_of.weekday() == 6:  # Sunday
            as_of = as_of - dt.timedelta(days=2)

    print(f"Swing Trend Pipeline", file=sys.stderr)
    print(f"  Config: {config_path}", file=sys.stderr)
    print(f"  Root:   {root}", file=sys.stderr)
    print(f"  Lookback: {lookback} days", file=sys.stderr)
    print(f"  As-of: {as_of or 'latest'}", file=sys.stderr)
    print(f"  Output: {out_dir}", file=sys.stderr)
    print("", file=sys.stderr)

    scores, signals = run_pipeline(
        cfg=cfg,
        root=root,
        lookback=lookback,
        as_of=as_of,
        out_dir=out_dir,
        max_recommendations=args.top,
    )

    print(f"\nDone. {len(scores)} recommendations generated.", file=sys.stderr)


if __name__ == "__main__":
    main()
