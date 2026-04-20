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
    if s in {"Iron Condor", "Iron Butterfly", "Long Iron Condor"}:
        pw = safe_float(put_width)
        cw = safe_float(call_width)
        w_eff = max(pw, cw) if np.isfinite(pw) and np.isfinite(cw) else w
        if not np.isfinite(w_eff) or w_eff <= 0:
            return math.nan
        if s == "Long Iron Condor":
            max_profit = max(0.0, w_eff - n)
            max_loss = max(0.0, n)
        else:
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
    long_call_strike: float = math.nan,
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
    if s in {"Iron Condor", "Iron Butterfly"}:
        sc = safe_float(short_call_strike)
        if not np.isfinite(sc):
            return math.nan, math.nan
        return ss - n, sc + n
    if s == "Long Iron Condor":
        lc = safe_float(long_call_strike)
        if not np.isfinite(ls) or not np.isfinite(lc):
            return math.nan, math.nan
        return ls - n, lc + n
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


def yfinance_symbol(ticker: str) -> str:
    t = str(ticker or "").strip().upper().replace("/", "-").replace(".", "-")
    compact_class_shares = {
        "BRKA": "BRK-A",
        "BRKB": "BRK-B",
    }
    if t in compact_class_shares:
        return compact_class_shares[t]
    return t


def yfinance_download_retry(
    ticker: str,
    start: str,
    end: str,
    retries: int = 3,
    pause_sec: float = 0.8,
) -> pd.DataFrame:
    last = pd.DataFrame()
    yf_ticker = yfinance_symbol(ticker)
    for attempt in range(max(1, int(retries))):
        try:
            raw = yf.download(yf_ticker, start=start, end=end, auto_adjust=True, progress=False)  # [T6] was: auto_adjust=False — caused splits to break price continuity
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
    long_call_strike: float = math.nan,
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
    # [T6] Convert calendar DTE to approximate trading days
    trading_days_dte = max(1, int(round(dte * 252 / 365)))
    for i in range(0, n - trading_days_dte):
        entry = safe_float(hist["Close"].iloc[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        window = hist.iloc[i + 1 : i + 1 + trading_days_dte]
        if len(window) < trading_days_dte:
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
        elif s in {"Iron Condor", "Iron Butterfly"}:
            if not (np.isfinite(upper_be_ratio) and np.isfinite(short_call_ratio)):
                continue
            sim_be_high = float(entry) * upper_be_ratio
            sim_short_call = float(entry) * short_call_ratio
            win = sim_be <= end_close <= sim_be_high
            no_touch_n += 1
            if lo > sim_short and hi < sim_short_call:
                no_touch += 1
        elif s == "Long Iron Condor":
            if not np.isfinite(upper_be_ratio):
                continue
            sim_be_high = float(entry) * upper_be_ratio
            win = end_close <= sim_be or end_close >= sim_be_high
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


def boolish(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "pass", "passed"}:
        return True
    if s in {"0", "false", "f", "no", "n", "fail", "failed"}:
        return False
    return None


def row_get(row: object, name: str, default: object = math.nan) -> object:
    return getattr(row, name, default)


def trend_bucket(ret20: float) -> str:
    r = safe_float(ret20)
    if not np.isfinite(r):
        return "unknown"
    if r >= 0.035:
        return "up"
    if r <= -0.035:
        return "down"
    return "flat"


def metric_bucket(value: float, low_cut: float, high_cut: float) -> str:
    v = safe_float(value)
    lo = safe_float(low_cut)
    hi = safe_float(high_cut)
    if not (np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi)):
        return "unknown"
    if v <= lo:
        return "low"
    if v >= hi:
        return "high"
    return "mid"


def _slice_context_window(hist: pd.DataFrame, idx: int, lookback: int = 20) -> pd.DataFrame:
    start = max(0, int(idx) - int(lookback) + 1)
    return hist.iloc[start : int(idx) + 1]


def context_metrics(hist: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    cache_key = f"_context_metrics_{int(lookback)}"
    cached = hist.attrs.get(cache_key)
    if isinstance(cached, pd.DataFrame) and len(cached) == len(hist):
        return cached

    close = pd.to_numeric(hist["Close"], errors="coerce")
    high = pd.to_numeric(hist["High"], errors="coerce")
    low = pd.to_numeric(hist["Low"], errors="coerce")
    lb = max(1, int(lookback))

    ret20 = close / close.shift(lb) - 1.0
    ret5 = close / close.shift(5) - 1.0
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    rv20 = returns.rolling(lb, min_periods=5).std(ddof=0) * math.sqrt(252.0)
    roll_hi = high.rolling(lb, min_periods=1).max()
    roll_lo = low.rolling(lb, min_periods=1).min()
    range20 = roll_hi / roll_lo - 1.0

    metrics = pd.DataFrame(
        {
            "ret20": ret20.replace([np.inf, -np.inf], np.nan),
            "ret5": ret5.replace([np.inf, -np.inf], np.nan),
            "rv20": rv20.replace([np.inf, -np.inf], np.nan),
            "range20": range20.replace([np.inf, -np.inf], np.nan),
        },
        index=hist.index,
    )
    hist.attrs[cache_key] = metrics
    return metrics


def price_context_at(
    hist: pd.DataFrame,
    idx: int,
    thresholds: Optional[Dict[str, float]] = None,
    lookback: int = 20,
) -> Dict[str, object]:
    if hist.empty or idx < 1 or idx >= len(hist):
        return {
            "ret20": math.nan,
            "ret5": math.nan,
            "rv20": math.nan,
            "range20": math.nan,
            "trend_bucket": "unknown",
            "vol_bucket": "unknown",
            "range_bucket": "unknown",
            "range_neutral": False,
        }

    metrics = context_metrics(hist, lookback=lookback)
    ret20 = safe_float(metrics["ret20"].iloc[idx])
    ret5 = safe_float(metrics["ret5"].iloc[idx])
    rv20 = safe_float(metrics["rv20"].iloc[idx])
    range20 = safe_float(metrics["range20"].iloc[idx])

    thresholds = thresholds or {}
    t_bucket = trend_bucket(ret20)
    v_bucket = metric_bucket(rv20, thresholds.get("rv20_low", math.nan), thresholds.get("rv20_high", math.nan))
    r_bucket = metric_bucket(range20, thresholds.get("range20_low", math.nan), thresholds.get("range20_high", math.nan))
    range_neutral = bool(np.isfinite(ret20) and abs(ret20) <= 0.055 and t_bucket == "flat")
    return {
        "ret20": ret20,
        "ret5": ret5,
        "rv20": rv20,
        "range20": range20,
        "trend_bucket": t_bucket,
        "vol_bucket": v_bucket,
        "range_bucket": r_bucket,
        "range_neutral": range_neutral,
    }


def context_thresholds(hist: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    cache_key = f"_context_thresholds_{int(lookback)}"
    cached = hist.attrs.get(cache_key)
    if isinstance(cached, dict):
        return cached

    metrics = context_metrics(hist, lookback=lookback)
    start = max(int(lookback), 1)
    rv_values = pd.to_numeric(metrics["rv20"].iloc[start:], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    range_values = pd.to_numeric(metrics["range20"].iloc[start:], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    def q(values, pct):
        if len(values) == 0:
            return math.nan
        return float(np.nanpercentile(values, pct))

    out = {
        "rv20_low": q(rv_values, 33),
        "rv20_high": q(rv_values, 67),
        "range20_low": q(range_values, 33),
        "range20_high": q(range_values, 67),
    }
    hist.attrs[cache_key] = out
    return out


def iv_bucket(iv_rank: float) -> str:
    iv = safe_float(iv_rank)
    if not np.isfinite(iv):
        return "unknown"
    if iv >= 60:
        return "high"
    if iv <= 25:
        return "low"
    return "mid"


def build_condition_profile(strategy: str, hist: pd.DataFrame, row: object) -> Dict[str, object]:
    thresholds = context_thresholds(hist)
    current = price_context_at(hist, len(hist) - 1, thresholds=thresholds)
    track = str(row_get(row, "track", "") or "").strip().upper()
    strat = str(strategy).strip()
    range_neutrality = safe_float(row_get(row, "range_neutrality_stage1", math.nan))
    stage1_range_neutral = bool(np.isfinite(range_neutrality) and range_neutrality >= 0.55)
    require_range_neutral = bool(
        strat in {"Iron Condor", "Iron Butterfly"}
        or (track == "SHIELD" and strat in {"Bull Put Credit", "Bear Call Credit"})
        or (track == "SHIELD" and stage1_range_neutral)
    )
    sigma_pass = boolish(row_get(row, "sigma_pass_stage1", None))
    high_beta_pass = boolish(row_get(row, "high_beta_pass_stage1", None))
    earnings_label = str(row_get(row, "earnings_label_stage1", "") or "").strip()
    iv_label = iv_bucket(safe_float(row_get(row, "iv_rank", math.nan)))
    flow_direction = str(row_get(row, "flow_direction", "") or "").strip()
    flow_confidence = str(row_get(row, "flow_confidence", "") or "").strip()
    flow_confirmation = str(row_get(row, "flow_confirmation", "") or "").strip()

    profile_range_neutral = bool(current.get("range_neutral", False) or stage1_range_neutral)
    profile_trend_bucket = "flat" if require_range_neutral else current.get("trend_bucket", "unknown")

    return {
        "trend_bucket": profile_trend_bucket,
        "vol_bucket": current.get("vol_bucket", "unknown"),
        "range_bucket": current.get("range_bucket", "unknown"),
        "range_neutral": profile_range_neutral,
        "require_range_neutral": require_range_neutral,
        "ret20": current.get("ret20", math.nan),
        "rv20": current.get("rv20", math.nan),
        "range20": current.get("range20", math.nan),
        "iv_bucket": iv_label,
        "sigma_pass_stage1": sigma_pass,
        "high_beta_pass_stage1": high_beta_pass,
        "earnings_label_stage1": earnings_label,
        "flow_direction": flow_direction,
        "flow_confidence": flow_confidence,
        "flow_confirmation": flow_confirmation,
        "unsupported_context": "historical_uw_flow,gex_state,earnings_history,iv_history",
    }


def condition_profile_label(profile: Dict[str, object]) -> str:
    parts = [
        f"trend={profile.get('trend_bucket', 'unknown')}",
        f"vol={profile.get('vol_bucket', 'unknown')}",
        f"range={profile.get('range_bucket', 'unknown')}",
        f"range_neutral={str(bool(profile.get('range_neutral', False))).lower()}",
        f"iv={profile.get('iv_bucket', 'unknown')}",
    ]
    sigma = profile.get("sigma_pass_stage1")
    if sigma is not None:
        parts.append(f"sigma_stage1={str(bool(sigma)).lower()}")
    high_beta = profile.get("high_beta_pass_stage1")
    if high_beta is not None:
        parts.append(f"high_beta_stage1={str(bool(high_beta)).lower()}")
    earnings = str(profile.get("earnings_label_stage1", "") or "").strip()
    if earnings:
        parts.append(f"earnings={earnings}")
    flow_direction = str(profile.get("flow_direction", "") or "").strip()
    flow_confidence = str(profile.get("flow_confidence", "") or "").strip()
    flow_confirmation = str(profile.get("flow_confirmation", "") or "").strip()
    if flow_direction:
        parts.append(f"flow={flow_direction}")
    if flow_confidence:
        parts.append(f"flow_conf={flow_confidence}")
    if flow_confirmation:
        parts.append(f"flow_confirm={flow_confirmation}")
    return ";".join(parts)


def context_matches_profile(ctx: Dict[str, object], profile: Dict[str, object], level: str) -> bool:
    profile_trend = str(profile.get("trend_bucket", "unknown"))
    if profile_trend != "unknown" and str(ctx.get("trend_bucket", "unknown")) != profile_trend:
        return False

    if level in {"strict", "same_trend_vol"}:
        profile_vol = str(profile.get("vol_bucket", "unknown"))
        if profile_vol != "unknown" and str(ctx.get("vol_bucket", "unknown")) != profile_vol:
            return False

    if bool(profile.get("require_range_neutral", False)):
        if not bool(ctx.get("range_neutral", False)):
            return False
        if level in {"strict", "same_trend_range"}:
            profile_range = str(profile.get("range_bucket", "unknown"))
            if profile_range != "unknown" and str(ctx.get("range_bucket", "unknown")) != profile_range:
                return False
    return True


def _simulate_setup_with_condition(
    strategy: str,
    hist: pd.DataFrame,
    dte: int,
    spot_asof: float,
    short_strike: float,
    breakeven_level: float,
    short_call_strike: float,
    upper_breakeven_level: float,
    long_call_strike: float,
    profile: Dict[str, object],
    level: str,
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
    thresholds = context_thresholds(hist)
    metrics = context_metrics(hist)
    ret20_arr = pd.to_numeric(metrics["ret20"], errors="coerce").to_numpy(dtype=float)
    rv20_arr = pd.to_numeric(metrics["rv20"], errors="coerce").to_numpy(dtype=float)
    range20_arr = pd.to_numeric(metrics["range20"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(hist["Close"], errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(hist["High"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(hist["Low"], errors="coerce").to_numpy(dtype=float)

    profile_trend = str(profile.get("trend_bucket", "unknown"))
    profile_vol = str(profile.get("vol_bucket", "unknown"))
    profile_range = str(profile.get("range_bucket", "unknown"))
    require_range_neutral = bool(profile.get("require_range_neutral", False))

    def context_matches_idx(idx: int) -> bool:
        ret20 = safe_float(ret20_arr[idx])
        rv20 = safe_float(rv20_arr[idx])
        range20 = safe_float(range20_arr[idx])
        t_bucket = trend_bucket(ret20)
        if profile_trend != "unknown" and t_bucket != profile_trend:
            return False
        if level in {"strict", "same_trend_vol"}:
            v_bucket = metric_bucket(rv20, thresholds.get("rv20_low", math.nan), thresholds.get("rv20_high", math.nan))
            if profile_vol != "unknown" and v_bucket != profile_vol:
                return False
        if require_range_neutral:
            range_neutral = bool(np.isfinite(ret20) and abs(ret20) <= 0.055 and t_bucket == "flat")
            if not range_neutral:
                return False
            if level in {"strict", "same_trend_range"}:
                r_bucket = metric_bucket(
                    range20,
                    thresholds.get("range20_low", math.nan),
                    thresholds.get("range20_high", math.nan),
                )
                if profile_range != "unknown" and r_bucket != profile_range:
                    return False
        return True

    wins = 0
    signals = 0
    no_touch = 0
    no_touch_n = 0
    trading_days_dte = max(1, int(round(dte * 252 / 365)))
    n = len(hist)
    for i in range(20, n - trading_days_dte):
        if not context_matches_idx(i):
            continue

        entry = safe_float(close_arr[i])
        if not np.isfinite(entry) or entry <= 0:
            continue
        high_window = high_arr[i + 1 : i + 1 + trading_days_dte]
        low_window = low_arr[i + 1 : i + 1 + trading_days_dte]
        if len(high_window) < trading_days_dte or len(low_window) < trading_days_dte:
            continue
        end_close = safe_float(close_arr[i + trading_days_dte])
        hi = safe_float(np.nanmax(high_window))
        lo = safe_float(np.nanmin(low_window))
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
        elif s in {"Iron Condor", "Iron Butterfly"}:
            if not (np.isfinite(upper_be_ratio) and np.isfinite(short_call_ratio)):
                continue
            sim_be_high = float(entry) * upper_be_ratio
            sim_short_call = float(entry) * short_call_ratio
            win = sim_be <= end_close <= sim_be_high
            no_touch_n += 1
            if lo > sim_short and hi < sim_short_call:
                no_touch += 1
        elif s == "Long Iron Condor":
            if not np.isfinite(upper_be_ratio):
                continue
            sim_be_high = float(entry) * upper_be_ratio
            win = end_close <= sim_be or end_close >= sim_be_high
        else:
            win = False

        wins += int(bool(win))

    no_touch_pct = (100.0 * no_touch / no_touch_n) if no_touch_n > 0 else math.nan
    return signals, wins, no_touch_pct


def simulate_setup_conditioned(
    strategy: str,
    hist: pd.DataFrame,
    dte: int,
    spot_asof: float,
    short_strike: float,
    breakeven_level: float,
    short_call_strike: float,
    upper_breakeven_level: float,
    long_call_strike: float,
    profile: Dict[str, object],
    min_signals: int,
) -> Tuple[str, int, int, float]:
    levels = ["strict", "same_trend_range", "same_trend_vol", "same_trend"]
    min_signals_i = max(1, int(min_signals))
    usable_min = min(min_signals_i, max(30, min_signals_i // 2))
    seen = []
    for level in levels:
        signals, wins, no_touch_pct = _simulate_setup_with_condition(
            strategy=strategy,
            hist=hist,
            dte=dte,
            spot_asof=spot_asof,
            short_strike=short_strike,
            breakeven_level=breakeven_level,
            short_call_strike=short_call_strike,
            upper_breakeven_level=upper_breakeven_level,
            long_call_strike=long_call_strike,
            profile=profile,
            level=level,
        )
        if signals > 0:
            seen.append((level, signals, wins, no_touch_pct))
        if signals >= usable_min:
            return level, signals, wins, no_touch_pct
    if seen:
        return max(seen, key=lambda x: x[1])
    return "no_conditioned_matches", 0, 0, math.nan


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
    setup_id: str = "",
) -> Dict[str, object]:
    return {
        "setup_id": str(setup_id or ""),
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
        "base_hist_success_pct": np.nan,
        "base_edge_pct": np.nan,
        "base_credit_no_touch_pct": np.nan,
        "base_signals": 0,
        "base_wins": 0,
        "conditioning_level": "unscored",
        "conditioning_profile": "",
        "unsupported_context": "",
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
        setup_id = str(getattr(r, "setup_id", "") or "").strip()
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
                    setup_id=setup_id,
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
                    setup_id=setup_id,
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
                    setup_id=setup_id,
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
                    setup_id=setup_id,
                )
            )
            continue

        short_call_strike = safe_float(getattr(r, "short_call_strike", math.nan))
        long_call_strike = safe_float(getattr(r, "long_call_strike", math.nan))
        put_width = safe_float(getattr(r, "put_width", math.nan))
        call_width = safe_float(getattr(r, "call_width", math.nan))

        be_low, be_high = breakeven_levels(
            strategy,
            r.long_strike,
            r.short_strike,
            gate_net,
            short_call_strike=short_call_strike,
            long_call_strike=long_call_strike,
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
                    setup_id=setup_id,
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
                    setup_id=setup_id,
                )
            )
            continue

        base_signals, base_wins, base_no_touch_pct = simulate_setup(
            strategy=strategy,
            hist=hist,
            dte=dte,
            spot_asof=float(spot),
            short_strike=safe_float(r.short_strike),
            breakeven_level=float(be_low),
            short_call_strike=short_call_strike,
            upper_breakeven_level=float(be_high) if np.isfinite(be_high) else math.nan,
            long_call_strike=long_call_strike,
        )
        if base_signals <= 0:
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
                    setup_id=setup_id,
                )
            )
            continue

        profile = build_condition_profile(strategy=strategy, hist=hist, row=r)
        conditioning_level, signals, wins, no_touch_pct = simulate_setup_conditioned(
            strategy=strategy,
            hist=hist,
            dte=dte,
            spot_asof=float(spot),
            short_strike=safe_float(r.short_strike),
            breakeven_level=float(be_low),
            short_call_strike=short_call_strike,
            upper_breakeven_level=float(be_high) if np.isfinite(be_high) else math.nan,
            long_call_strike=long_call_strike,
            profile=profile,
            min_signals=int(args.min_signals),
        )

        base_hist_success_pct = 100.0 * base_wins / base_signals
        base_edge = base_hist_success_pct - req
        if signals <= 0:
            out_rows.append(
                {
                    "setup_id": setup_id,
                    "ticker": ticker,
                    "strategy": strategy,
                    "expiry": expiry_iso,
                    "dte": dte,
                    "entry_gate": str(r.entry_gate),
                    "spot_at_signal": round(float(spot), 4),
                    "required_win_pct": round(float(req), 2),
                    "hist_success_pct": np.nan,
                    "edge_pct": np.nan,
                    "credit_no_touch_pct": np.nan,
                    "signals": 0,
                    "wins": 0,
                    "base_hist_success_pct": round(float(base_hist_success_pct), 2),
                    "base_edge_pct": round(float(base_edge), 2),
                    "base_credit_no_touch_pct": round(float(base_no_touch_pct), 2) if np.isfinite(base_no_touch_pct) else np.nan,
                    "base_signals": int(base_signals),
                    "base_wins": int(base_wins),
                    "conditioning_level": conditioning_level,
                    "conditioning_profile": condition_profile_label(profile),
                    "unsupported_context": str(profile.get("unsupported_context", "")),
                    "confidence": "Unknown",
                    "verdict": "UNKNOWN",
                    "status_reason": "no_conditioned_windows",
                }
            )
            continue

        hist_success_pct = 100.0 * wins / signals
        edge = hist_success_pct - req
        confidence = confidence_bucket(signals)
        if signals < int(args.min_signals):
            verdict = "FAIL" if edge < -5.0 else "LOW_SAMPLE"  # [T6] deeply negative edge is conclusive
        else:
            verdict = "PASS" if edge > 0 else "FAIL"

        out_rows.append(
            {
                "setup_id": setup_id,
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
                "base_hist_success_pct": round(float(base_hist_success_pct), 2),
                "base_edge_pct": round(float(base_edge), 2),
                "base_credit_no_touch_pct": round(float(base_no_touch_pct), 2) if np.isfinite(base_no_touch_pct) else np.nan,
                "base_signals": int(base_signals),
                "base_wins": int(base_wins),
                "conditioning_level": conditioning_level,
                "conditioning_profile": condition_profile_label(profile),
                "unsupported_context": str(profile.get("unsupported_context", "")),
                "confidence": confidence,
                "verdict": verdict,
                "status_reason": "conditioned_scored",
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
        "## Setup Likelihood (Conditioned Historical Analog)",
        "",
        "The primary likelihood columns are conditioned on pre-entry price context available in this dataset: trend bucket, realized-vol bucket, and range-neutral/range bucket for SHIELD income structures. Raw unconditioned ticker base-rate columns are retained for transparency only.",
        "",
        "Unsupported historical context not claimed by this model: UW flow, whale flow, GEX state, earnings history, and true historical IV rank.",
        "",
        out.to_markdown(index=False),
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Setups scored: {len(out)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
