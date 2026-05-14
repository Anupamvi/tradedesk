#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from uwos import swing_trend_pipeline as swing
from uwos.whale_source import find_bot_eod_source, open_bot_eod


DEFAULT_ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
DEFAULT_LOOKBACK = 30
DEFAULT_VALIDATE_DAYS = 30
DEFAULT_HORIZONS = (1, 3, 5, 10, 20)
DEFAULT_MAX_DAILY_ROWS = 80
DEFAULT_MIN_UNDERLYING_PRICE = 20.0
DEFAULT_MIN_OPTION_VOLUME = 10
DEFAULT_MIN_OPTION_OI = 50
DEFAULT_MIN_STRUCTURE_DTE = 7
DEFAULT_MAX_STRUCTURE_DTE = 60
DEFAULT_MAX_DEBIT_TO_WIDTH = 0.55
DEFAULT_MAX_QUOTE_WIDTH_TO_ENTRY = 0.45
DEFAULT_MAX_QUOTE_WIDTH_TO_SPREAD = 0.16
DEFAULT_MIN_VALIDATION_SAMPLES = 8
DEFAULT_MIN_VALIDATION_AVG_R = 0.02
DEFAULT_MIN_VALIDATION_PROFIT_FACTOR = 1.05
DEFAULT_MIN_PLAYBOOK_AVG_R = 0.05
DEFAULT_MIN_PLAYBOOK_PROFIT_FACTOR = 1.20
TECH_BULLISH_DRIFT_PLAYBOOK = "playbook_tech_bullish_drift"
BULLISH_BREAKOUT_PLAYBOOK = "playbook_bullish_breakout"
BEARISH_BREAKDOWN_PLAYBOOK = "playbook_bearish_breakdown"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OCC_RE = re.compile(r"^([A-Z0-9\.\-]{1,8})(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True)
class OptionStructure:
    ticker: str
    direction: str
    strategy: str
    expiry: dt.date
    dte: int
    right: str
    long_symbol: str
    short_symbol: str
    long_strike: float
    short_strike: float
    width: float
    entry_net: float
    max_risk: float
    quote_width: float
    long_bid: float
    long_ask: float
    short_bid: float
    short_ask: float
    long_volume: float
    short_volume: float
    long_open_interest: float
    short_open_interest: float
    quote_sanity: str

    @property
    def spread_label(self) -> str:
        suffix = "C" if self.right == "C" else "P"
        return (
            f"Buy {self.long_strike:g}{suffix} / Sell {self.short_strike:g}{suffix} "
            f"exp {self.expiry.isoformat()} @ {self.entry_net:.2f} debit"
        )

    @property
    def max_profit(self) -> float:
        return max(0.0, (self.width - self.entry_net) * 100.0)

    @property
    def reward_to_risk(self) -> float:
        return _safe_div(self.max_profit, self.max_risk, math.nan)


def _fnum(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        if isinstance(value, str):
            text = value.strip().replace(",", "").replace("$", "").replace("%", "")
            if not text or text.lower() in {"nan", "none", "nat"}:
                return default
            return float(text)
        return float(value)
    except Exception:
        return default


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    if not math.isfinite(num) or not math.isfinite(den) or den == 0:
        return default
    return float(num / den)


def _clip(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if not math.isfinite(value):
        return lo
    return max(lo, min(hi, float(value)))


def _parse_date(value: Any) -> Optional[dt.date]:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_horizons(value: str) -> List[int]:
    out: List[int] = []
    for part in str(value or "").split(","):
        text = part.strip()
        if not text:
            continue
        parsed = int(text)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("horizons must be positive market-day counts")
        out.append(parsed)
    return sorted(set(out))


def _pct_return(close: Any, prev: Any) -> float:
    c = _fnum(close)
    p = _fnum(prev)
    if not math.isfinite(c) or not math.isfinite(p) or p == 0:
        return math.nan
    return (c - p) / p


def _profit_factor(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    gains = sum(v for v in vals if v > 0)
    losses = abs(sum(v for v in vals if v < 0))
    if losses == 0:
        return math.inf if gains > 0 else 0.0
    return gains / losses


def _max_losing_streak(values: Iterable[float]) -> int:
    streak = 0
    worst = 0
    for value in values:
        if math.isfinite(float(value)) and float(value) < 0:
            streak += 1
            worst = max(worst, streak)
        else:
            streak = 0
    return worst


def _max_drawdown(values: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for value in values:
        if not math.isfinite(float(value)):
            continue
        equity += float(value)
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def _deterministic_hash(*parts: Any) -> int:
    text = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def parse_occ(symbol: Any) -> Optional[Tuple[str, dt.date, str, float]]:
    match = OCC_RE.match(str(symbol or "").strip().upper())
    if not match:
        return None
    ticker, yymmdd, right, strike_raw = match.groups()
    try:
        expiry = dt.date(int("20" + yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
        strike = int(strike_raw) / 1000.0
    except Exception:
        return None
    return ticker, expiry, right, float(strike)


def _fmt_float(value: Any, digits: int = 2, suffix: str = "") -> str:
    v = _fnum(value)
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}{suffix}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    return value


class DataCache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self._screeners: Dict[dt.date, pd.DataFrame] = {}
        self._hot: Dict[dt.date, pd.DataFrame] = {}
        self._options: Dict[dt.date, pd.DataFrame] = {}
        self._chain_oi: Dict[dt.date, pd.DataFrame] = {}
        self._whales: Dict[dt.date, Dict[str, float]] = {}

    def day_dir(self, day: dt.date) -> Path:
        return self.root / day.isoformat()

    def screener(self, day: dt.date) -> pd.DataFrame:
        if day in self._screeners:
            return self._screeners[day]
        path = swing.resolve_csv_for_day(self.day_dir(day), day.isoformat(), "stock-screener")
        df = swing.read_csv_from_path(path) if path is not None else pd.DataFrame()
        if not df.empty:
            ticker_col = next((c for c in df.columns if c.strip().lower() in {"ticker", "symbol"}), None)
            if ticker_col and ticker_col != "ticker":
                df = df.rename(columns={ticker_col: "ticker"})
            if "ticker" in df.columns:
                df["ticker"] = df["ticker"].fillna("").astype(str).str.strip().str.upper()
        self._screeners[day] = df
        return df

    def hot_chains(self, day: dt.date) -> pd.DataFrame:
        if day in self._hot:
            return self._hot[day]
        path = swing.resolve_csv_for_day(self.day_dir(day), day.isoformat(), "hot-chains")
        df = swing.read_csv_from_path(path) if path is not None else pd.DataFrame()
        self._hot[day] = df
        return df

    def option_snapshot(self, day: dt.date) -> pd.DataFrame:
        if day in self._options:
            return self._options[day]
        hot = self.hot_chains(day)
        if hot.empty or "option_symbol" not in hot.columns:
            self._options[day] = pd.DataFrame()
            return self._options[day]
        rows: List[Dict[str, Any]] = []
        for row in hot.itertuples(index=False):
            data = row._asdict()
            parsed = parse_occ(data.get("option_symbol"))
            if parsed is None:
                continue
            ticker, expiry, right, strike = parsed
            bid = _fnum(data.get("bid"))
            ask = _fnum(data.get("ask"))
            if not math.isfinite(bid) or not math.isfinite(ask) or ask <= 0 or ask < bid:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "expiry": expiry,
                    "dte": max(0, (expiry - day).days),
                    "right": right,
                    "strike": strike,
                    "option_symbol": str(data.get("option_symbol", "")).strip().upper(),
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2.0,
                    "volume": _fnum(data.get("volume"), 0.0),
                    "open_interest": _fnum(data.get("open_interest"), 0.0),
                    "premium": _fnum(data.get("premium"), 0.0),
                    "ask_side_volume": _fnum(data.get("ask_side_volume"), 0.0),
                    "bid_side_volume": _fnum(data.get("bid_side_volume"), 0.0),
                    "sector": str(data.get("sector", "") or "").strip(),
                }
            )
        self._options[day] = pd.DataFrame(rows)
        return self._options[day]

    def chain_oi(self, day: dt.date) -> pd.DataFrame:
        if day in self._chain_oi:
            return self._chain_oi[day]
        path = swing.resolve_csv_for_day(self.day_dir(day), day.isoformat(), "chain-oi-changes")
        df = swing.read_csv_from_path(path) if path is not None else pd.DataFrame()
        if not df.empty and "underlying_symbol" in df.columns:
            df["underlying_symbol"] = df["underlying_symbol"].fillna("").astype(str).str.strip().str.upper()
            if "option_symbol" in df.columns:
                parsed = df["option_symbol"].map(parse_occ)
                df["_right"] = parsed.map(lambda p: p[2] if p else "")
                df["_strike"] = parsed.map(lambda p: p[3] if p else math.nan)
        self._chain_oi[day] = df
        return df

    def whale_symbols(self, day: dt.date, wanted_symbols: Optional[set[str]] = None) -> Dict[str, float]:
        wanted = {str(s).strip().upper() for s in wanted_symbols or set() if str(s).strip()}
        if day in self._whales:
            cached = self._whales[day]
            return {symbol: premium for symbol, premium in cached.items() if symbol in wanted} if wanted else cached
        out: Dict[str, float] = {}
        try:
            source = find_bot_eod_source(self.day_dir(day), day.isoformat())
        except FileNotFoundError:
            source = None
        if source is None:
            self._whales[day] = out
            return out
        try:
            with open_bot_eod(source) as (handle, _label):
                for chunk in pd.read_csv(
                    handle,
                    usecols=lambda c: str(c) in {"underlying_symbol", "premium", "equity_type"},
                    chunksize=250_000,
                    low_memory=False,
                ):
                    if "underlying_symbol" not in chunk.columns:
                        continue
                    symbols = chunk["underlying_symbol"].fillna("").astype(str).str.strip().str.upper()
                    premium = pd.to_numeric(chunk.get("premium", 0), errors="coerce").fillna(0.0).abs()
                    if "equity_type" in chunk.columns:
                        issue = chunk["equity_type"].fillna("").astype(str).str.upper()
                        mask = issue.ne("ETF")
                    else:
                        mask = pd.Series(True, index=chunk.index)
                    if wanted:
                        mask = mask & symbols.isin(wanted)
                    if not mask.any():
                        continue
                    grouped = pd.DataFrame({"symbol": symbols[mask], "premium": premium[mask]}).groupby("symbol")[
                        "premium"
                    ].sum()
                    for symbol, prem in grouped.items():
                        if symbol:
                            out[str(symbol)] = out.get(str(symbol), 0.0) + float(prem)
        except (zipfile.BadZipFile, OSError, ValueError, pd.errors.ParserError):
            out = {}
        if not wanted:
            self._whales[day] = out
        return out


def available_market_days(root: Path) -> List[dt.date]:
    days = [day for day, _ in swing.discover_trading_days(Path(root), 10000, None)]
    return sorted(days)


def resolve_as_of(root: Path, value: Optional[str]) -> dt.date:
    days = available_market_days(root)
    if not days:
        raise RuntimeError(f"No usable UW dated folders found under {root}")
    if not value:
        return days[-1]
    parsed = dt.date.fromisoformat(value)
    eligible = [d for d in days if d <= parsed]
    if not eligible:
        raise RuntimeError(f"No usable UW dated folders on or before {parsed}")
    return eligible[-1]


def _window_days(days: Sequence[dt.date], as_of: dt.date, lookback: int) -> List[dt.date]:
    eligible = [d for d in days if d <= as_of]
    return eligible[-max(1, int(lookback)) :]


def _future_day(days: Sequence[dt.date], signal_date: dt.date, horizon: int) -> Optional[dt.date]:
    ordered = sorted(days)
    try:
        idx = ordered.index(signal_date)
    except ValueError:
        return None
    target_idx = idx + int(horizon)
    if target_idx >= len(ordered):
        return None
    return ordered[target_idx]


def _market_day_on_or_before(days: Sequence[dt.date], target: dt.date) -> Optional[dt.date]:
    eligible = [day for day in days if day <= target]
    return eligible[-1] if eligible else None


def market_regime_summary(cache: DataCache, days: Sequence[dt.date], as_of: dt.date, lookback: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    window = _window_days(days, as_of, lookback)
    latest = cache.screener(as_of)
    if latest.empty:
        return {"regime": "unknown", "reason": "latest stock-screener missing"}, pd.DataFrame()

    tickers = latest.get("ticker", pd.Series("", index=latest.index)).fillna("").astype(str).str.upper()

    def row_for(symbol: str) -> Optional[pd.Series]:
        sub = latest[tickers.eq(symbol)]
        return sub.iloc[0] if not sub.empty else None

    def series_close(symbol: str) -> List[Tuple[dt.date, float]]:
        rows = []
        for day in window:
            df = cache.screener(day)
            if df.empty or "ticker" not in df.columns:
                continue
            sub = df[df["ticker"].astype(str).str.upper().eq(symbol)]
            if not sub.empty:
                rows.append((day, _fnum(sub.iloc[0].get("close"))))
        return [(d, c) for d, c in rows if math.isfinite(c)]

    common = latest.copy()
    if "issue_type" in common.columns:
        common = common[common["issue_type"].fillna("").astype(str).str.lower().eq("common stock")]
    returns = common.apply(lambda r: _pct_return(r.get("close"), r.get("prev_close")), axis=1)
    returns = returns[np.isfinite(returns)]
    breadth = float(returns.gt(0).mean()) if len(returns) else math.nan

    spy = row_for("SPY")
    qqq = row_for("QQQ")
    iwm = row_for("IWM")
    vix = row_for("VIX")
    spy_1d = _pct_return(spy.get("close"), spy.get("prev_close")) if spy is not None else math.nan
    qqq_1d = _pct_return(qqq.get("close"), qqq.get("prev_close")) if qqq is not None else math.nan
    iwm_1d = _pct_return(iwm.get("close"), iwm.get("prev_close")) if iwm is not None else math.nan
    vix_close = _fnum(vix.get("close")) if vix is not None else math.nan
    vix_1d = _pct_return(vix.get("close"), vix.get("prev_close")) if vix is not None else math.nan
    index_pcr_values = [_fnum(r.get("put_call_ratio")) for r in (spy, qqq, iwm) if r is not None]
    index_pcr_values = [v for v in index_pcr_values if math.isfinite(v)]
    index_pcr = float(np.mean(index_pcr_values)) if index_pcr_values else math.nan

    def n_day_return(symbol: str, n: int = 5) -> float:
        series = series_close(symbol)
        if len(series) <= n:
            return math.nan
        prev = series[-n - 1][1]
        last = series[-1][1]
        return _safe_div(last - prev, prev, math.nan)

    spy_5d = n_day_return("SPY", 5)
    qqq_5d = n_day_return("QQQ", 5)
    iwm_5d = n_day_return("IWM", 5)

    constructive_tape = (
        math.isfinite(vix_close)
        and vix_close < 20
        and math.isfinite(breadth)
        and breadth >= 0.52
        and math.isfinite(spy_5d)
        and spy_5d > 0
        and math.isfinite(qqq_5d)
        and qqq_5d > 0
    )
    stress_index_pcr = math.isfinite(index_pcr) and index_pcr >= 1.35 and not constructive_tape

    if (
        (math.isfinite(vix_close) and vix_close >= 25)
        or (math.isfinite(breadth) and breadth < 0.40)
        or stress_index_pcr
        or (
            math.isfinite(spy_5d)
            and math.isfinite(qqq_5d)
            and spy_5d < -0.015
            and qqq_5d < -0.015
        )
    ):
        regime = "risk_off"
    elif constructive_tape:
        regime = "risk_on"
    else:
        regime = "mixed"

    sector_rows: List[Dict[str, Any]] = []
    if "sector" in common.columns:
        work = common.copy()
        work["_ret_1d"] = returns.reindex(work.index)
        bullish_premium = pd.to_numeric(
            work.get("bullish_premium", pd.Series(0, index=work.index)),
            errors="coerce",
        ).fillna(0)
        bearish_premium = pd.to_numeric(
            work.get("bearish_premium", pd.Series(0, index=work.index)),
            errors="coerce",
        ).fillna(0)
        work["_flow_premium"] = (
            bullish_premium.abs()
            + bearish_premium.abs()
        )
        for sector, sub in work.groupby(work["sector"].fillna("Unknown").astype(str)):
            if not sector or sector.lower() == "nan":
                sector = "Unknown"
            sector_rows.append(
                {
                    "sector": sector,
                    "symbols": int(sub["ticker"].nunique()) if "ticker" in sub.columns else int(len(sub)),
                    "avg_1d_return_pct": float(pd.to_numeric(sub["_ret_1d"], errors="coerce").mean() * 100),
                    "positive_breadth_pct": float(pd.to_numeric(sub["_ret_1d"], errors="coerce").gt(0).mean() * 100),
                    "flow_premium": float(pd.to_numeric(sub["_flow_premium"], errors="coerce").sum()),
                }
            )
    sector_df = pd.DataFrame(sector_rows)
    if not sector_df.empty:
        sector_df = sector_df.sort_values(["avg_1d_return_pct", "flow_premium"], ascending=[False, False])

    summary = {
        "as_of": as_of.isoformat(),
        "regime": regime,
        "breadth_pct": breadth * 100 if math.isfinite(breadth) else math.nan,
        "vix_close": vix_close,
        "vix_1d_return_pct": vix_1d * 100 if math.isfinite(vix_1d) else math.nan,
        "spy_1d_return_pct": spy_1d * 100 if math.isfinite(spy_1d) else math.nan,
        "qqq_1d_return_pct": qqq_1d * 100 if math.isfinite(qqq_1d) else math.nan,
        "iwm_1d_return_pct": iwm_1d * 100 if math.isfinite(iwm_1d) else math.nan,
        "spy_5d_return_pct": spy_5d * 100 if math.isfinite(spy_5d) else math.nan,
        "qqq_5d_return_pct": qqq_5d * 100 if math.isfinite(qqq_5d) else math.nan,
        "iwm_5d_return_pct": iwm_5d * 100 if math.isfinite(iwm_5d) else math.nan,
        "index_put_call_ratio": index_pcr,
        "reason": (
            f"{regime}: breadth {_fmt_float(breadth * 100 if math.isfinite(breadth) else math.nan, 0, '%')}, "
            f"VIX {_fmt_float(vix_close, 2)}, SPY 5D {_fmt_float(spy_5d * 100 if math.isfinite(spy_5d) else math.nan, 2, '%')}, "
            f"QQQ 5D {_fmt_float(qqq_5d * 100 if math.isfinite(qqq_5d) else math.nan, 2, '%')}, "
            f"index PCR {_fmt_float(index_pcr, 2)}"
        ),
    }
    return summary, sector_df


def _common_stock_universe(df: pd.DataFrame, min_price: float) -> pd.DataFrame:
    if df.empty or "ticker" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["ticker"] = out["ticker"].fillna("").astype(str).str.strip().str.upper()
    out = out[out["ticker"].str.match(r"^[A-Z][A-Z0-9\.\-]{0,7}$", na=False)].copy()
    if "issue_type" in out.columns:
        issue = out["issue_type"].fillna("").astype(str).str.lower()
        out = out[issue.eq("common stock")].copy()
    out["_close"] = pd.to_numeric(out.get("close"), errors="coerce")
    out = out[out["_close"].ge(float(min_price))].copy()
    return out


def _flow_bias_from_row(row: pd.Series) -> float:
    bull = _fnum(row.get("bullish_premium"), 0.0)
    bear = _fnum(row.get("bearish_premium"), 0.0)
    return _safe_div(bull - bear, abs(bull) + abs(bear), 0.0)


def _hot_bias(options: pd.DataFrame, ticker: str) -> Tuple[float, float]:
    if options.empty:
        return 0.0, 0.0
    sub = options[options["ticker"].eq(ticker)]
    if sub.empty:
        return 0.0, 0.0
    ask = pd.to_numeric(sub.get("ask_side_volume", 0), errors="coerce").fillna(0).sum()
    bid = pd.to_numeric(sub.get("bid_side_volume", 0), errors="coerce").fillna(0).sum()
    premium = pd.to_numeric(sub.get("premium", 0), errors="coerce").fillna(0).abs().sum()
    return _safe_div(float(ask - bid), float(ask + bid), 0.0), float(premium)


def _oi_bias(chain_oi: pd.DataFrame, ticker: str) -> float:
    if chain_oi.empty or "underlying_symbol" not in chain_oi.columns:
        return 0.0
    sub = chain_oi[chain_oi["underlying_symbol"].eq(ticker)]
    if sub.empty:
        return 0.0
    change = pd.to_numeric(sub.get("oi_change", sub.get("oi_diff_plain", 0)), errors="coerce").fillna(0.0)
    rights = sub.get("_right", pd.Series("", index=sub.index)).fillna("").astype(str)
    call = float(change[rights.eq("C")].sum())
    put = float(change[rights.eq("P")].sum())
    return _safe_div(call - put, abs(call) + abs(put), 0.0)


def _history_for_ticker(cache: DataCache, days: Sequence[dt.date], ticker: str) -> List[pd.Series]:
    rows: List[pd.Series] = []
    for day in days:
        df = cache.screener(day)
        if df.empty or "ticker" not in df.columns:
            continue
        sub = df[df["ticker"].eq(ticker)]
        if not sub.empty:
            rows.append(sub.iloc[0])
    return rows


def _width_for_spot(spot: float) -> float:
    if spot < 25:
        return 2.5
    if spot < 75:
        return 5.0
    if spot < 180:
        return 10.0
    return 20.0


def build_debit_structure(
    options: pd.DataFrame,
    *,
    ticker: str,
    spot: float,
    direction: str,
    as_of: dt.date,
    min_volume: int = DEFAULT_MIN_OPTION_VOLUME,
    min_open_interest: int = DEFAULT_MIN_OPTION_OI,
    min_dte: int = DEFAULT_MIN_STRUCTURE_DTE,
    max_dte: int = DEFAULT_MAX_STRUCTURE_DTE,
) -> Tuple[Optional[OptionStructure], List[str]]:
    if options.empty:
        return None, ["no_hot_chain_quotes"]
    side = str(direction).lower()
    right = "C" if side == "bullish" else "P"
    strategy = "Bull Call Debit" if side == "bullish" else "Bear Put Debit"
    width_target = _width_for_spot(spot)
    sub = options[
        options["ticker"].eq(ticker)
        & options["right"].eq(right)
        & options["dte"].between(max(1, int(min_dte)), max(1, int(max_dte)))
        & pd.to_numeric(options["volume"], errors="coerce").fillna(0).ge(min_volume)
        & pd.to_numeric(options["open_interest"], errors="coerce").fillna(0).ge(min_open_interest)
    ].copy()
    if sub.empty:
        return None, ["no_liquid_directional_options"]

    structures: List[OptionStructure] = []
    for expiry, exp_df in sub.groupby("expiry"):
        exp_df = exp_df.sort_values("strike").copy()
        strikes = exp_df["strike"].astype(float).tolist()
        for _, long_row in exp_df.iterrows():
            long_strike = float(long_row["strike"])
            if side == "bullish":
                if not (spot * 0.98 <= long_strike <= spot * 1.04):
                    continue
                short_pool = exp_df[exp_df["strike"].astype(float).gt(long_strike)]
                short_pool = short_pool.assign(_dist=(short_pool["strike"].astype(float) - (long_strike + width_target)).abs())
            else:
                if not (spot * 0.96 <= long_strike <= spot * 1.02):
                    continue
                short_pool = exp_df[exp_df["strike"].astype(float).lt(long_strike)]
                short_pool = short_pool.assign(_dist=(short_pool["strike"].astype(float) - (long_strike - width_target)).abs())
            if short_pool.empty:
                continue
            short_row = short_pool.sort_values(["_dist", "strike"]).iloc[0]
            short_strike = float(short_row["strike"])
            width = abs(short_strike - long_strike)
            if width <= 0:
                continue
            long_bid = _fnum(long_row.get("bid"))
            long_ask = _fnum(long_row.get("ask"))
            short_bid = _fnum(short_row.get("bid"))
            short_ask = _fnum(short_row.get("ask"))
            debit = long_ask - short_bid
            if not math.isfinite(debit) or debit <= 0:
                continue
            quote_width = max(0.0, long_ask - long_bid) + max(0.0, short_ask - short_bid)
            if debit / width > DEFAULT_MAX_DEBIT_TO_WIDTH:
                sanity = "debit_too_expensive"
            elif quote_width / max(debit, 0.01) > DEFAULT_MAX_QUOTE_WIDTH_TO_ENTRY:
                sanity = "wide_quote_vs_entry"
            elif quote_width / width > DEFAULT_MAX_QUOTE_WIDTH_TO_SPREAD:
                sanity = "wide_quote_vs_width"
            else:
                sanity = "ok"
            structures.append(
                OptionStructure(
                    ticker=ticker,
                    direction=side,
                    strategy=strategy,
                    expiry=expiry,
                    dte=max(0, (expiry - as_of).days),
                    right=right,
                    long_symbol=str(long_row["option_symbol"]),
                    short_symbol=str(short_row["option_symbol"]),
                    long_strike=long_strike,
                    short_strike=short_strike,
                    width=width,
                    entry_net=float(debit),
                    max_risk=float(debit * 100.0),
                    quote_width=float(quote_width),
                    long_bid=long_bid,
                    long_ask=long_ask,
                    short_bid=short_bid,
                    short_ask=short_ask,
                    long_volume=_fnum(long_row.get("volume"), 0.0),
                    short_volume=_fnum(short_row.get("volume"), 0.0),
                    long_open_interest=_fnum(long_row.get("open_interest"), 0.0),
                    short_open_interest=_fnum(short_row.get("open_interest"), 0.0),
                    quote_sanity=sanity,
                )
            )
    if not structures:
        return None, ["no_vertical_structure_found"]
    structures.sort(
        key=lambda s: (
            0 if s.quote_sanity == "ok" else 1,
            s.entry_net / max(s.width, 0.01),
            -s.reward_to_risk if math.isfinite(s.reward_to_risk) else 0.0,
            abs(s.dte - 35),
            s.quote_width / max(s.entry_net, 0.01),
        )
    )
    return structures[0], []


def _structure_to_dict(structure: Optional[OptionStructure]) -> Dict[str, Any]:
    if structure is None:
        return {}
    return {
        "strategy": structure.strategy,
        "expiry": structure.expiry.isoformat(),
        "dte": structure.dte,
        "long_symbol": structure.long_symbol,
        "short_symbol": structure.short_symbol,
        "long_strike": structure.long_strike,
        "short_strike": structure.short_strike,
        "width": structure.width,
        "entry_net": structure.entry_net,
        "max_risk": structure.max_risk,
        "max_profit": structure.max_profit,
        "reward_to_risk": structure.reward_to_risk,
        "quote_width": structure.quote_width,
        "quote_sanity": structure.quote_sanity,
        "trade_setup": structure.spread_label,
        "long_volume": structure.long_volume,
        "short_volume": structure.short_volume,
        "long_open_interest": structure.long_open_interest,
        "short_open_interest": structure.short_open_interest,
    }


def _candidate_universe_from_latest(latest: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if latest.empty:
        return latest.copy()
    work = latest.copy()
    work["_activity"] = (
        pd.to_numeric(work.get("bullish_premium", 0), errors="coerce").fillna(0).abs()
        + pd.to_numeric(work.get("bearish_premium", 0), errors="coerce").fillna(0).abs()
        + pd.to_numeric(work.get("call_volume", 0), errors="coerce").fillna(0)
        + pd.to_numeric(work.get("put_volume", 0), errors="coerce").fillna(0)
    )
    work["_ret_1d_abs"] = work.apply(
        lambda r: abs(_pct_return(r.get("close"), r.get("prev_close"))),
        axis=1,
    )
    work["_volume_ratio"] = (
        pd.to_numeric(work.get("total_volume", 0), errors="coerce").fillna(0)
        / pd.to_numeric(work.get("avg30_volume", 0), errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    work["_call_volume_ratio"] = (
        pd.to_numeric(work.get("call_volume", 0), errors="coerce").fillna(0)
        / pd.to_numeric(work.get("avg_30_day_call_volume", 0), errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    work["_put_volume_ratio"] = (
        pd.to_numeric(work.get("put_volume", 0), errors="coerce").fillna(0)
        / pd.to_numeric(work.get("avg_30_day_put_volume", 0), errors="coerce").replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    per_slice = max(75, int(max_rows) * 3)
    target = max(300, int(max_rows) * 10)
    pieces = [
        work.sort_values("_activity", ascending=False).head(max(target, per_slice)),
        work.sort_values("_ret_1d_abs", ascending=False).head(per_slice),
        work.sort_values("_volume_ratio", ascending=False).head(per_slice),
        work.sort_values("_call_volume_ratio", ascending=False).head(per_slice),
        work.sort_values("_put_volume_ratio", ascending=False).head(per_slice),
        work.sort_values("bullish_premium", ascending=False).head(per_slice),
        work.sort_values("bearish_premium", ascending=False).head(per_slice),
    ]
    out = pd.concat(pieces, ignore_index=False, sort=False)
    out = out[~out.index.duplicated(keep="first")].copy()
    out = out.sort_values(["_activity", "_ret_1d_abs", "_volume_ratio"], ascending=[False, False, False])
    return out.head(max(target, per_slice)).copy()


def _window_return(closes: Sequence[float], periods: int) -> float:
    if len(closes) <= periods:
        return math.nan
    start = closes[-periods - 1]
    end = closes[-1]
    return _safe_div(end - start, start, math.nan)


def _close_percentile_in_window(closes: Sequence[float]) -> float:
    vals = [float(v) for v in closes if math.isfinite(float(v))]
    if len(vals) < 3:
        return math.nan
    lo = min(vals)
    hi = max(vals)
    return _safe_div(vals[-1] - lo, hi - lo, 0.5)


def _tech_bullish_drift_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    return (
        df.get("sector", pd.Series("", index=df.index)).fillna("").astype(str).eq("Technology")
        & df.get("direction", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().eq("bullish")
        & df.get("strategy", pd.Series("", index=df.index)).fillna("").astype(str).eq("Bull Call Debit")
        & pd.to_numeric(df.get("ret_1d_pct", pd.Series(np.nan, index=df.index)), errors="coerce").between(0.0, 2.0)
        & pd.to_numeric(df.get("ret_5d_pct", pd.Series(np.nan, index=df.index)), errors="coerce").ge(5.0)
        & pd.to_numeric(df.get("flow_bias", pd.Series(np.nan, index=df.index)), errors="coerce").ge(0.0)
    )


def _bullish_breakout_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    ret5 = pd.to_numeric(df.get("ret_5d_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    ret10 = pd.to_numeric(df.get("ret_10d_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    ret20 = pd.to_numeric(df.get("ret_20d_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    close_rank = pd.to_numeric(df.get("close_window_percentile", pd.Series(np.nan, index=df.index)), errors="coerce")
    score = pd.to_numeric(df.get("score", pd.Series(np.nan, index=df.index)), errors="coerce")
    rr = pd.to_numeric(df.get("reward_to_risk", pd.Series(np.nan, index=df.index)), errors="coerce")
    return (
        df.get("direction", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().eq("bullish")
        & df.get("strategy", pd.Series("", index=df.index)).fillna("").astype(str).eq("Bull Call Debit")
        & (ret5.ge(6.0) | ret10.ge(9.0) | ret20.ge(14.0))
        & close_rank.ge(0.72)
        & score.ge(58.0)
        & rr.ge(0.65)
    )


def _bearish_breakdown_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(False, index=df.index)
    ret5 = pd.to_numeric(df.get("ret_5d_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    ret10 = pd.to_numeric(df.get("ret_10d_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    ret20 = pd.to_numeric(df.get("ret_20d_pct", pd.Series(np.nan, index=df.index)), errors="coerce")
    close_rank = pd.to_numeric(df.get("close_window_percentile", pd.Series(np.nan, index=df.index)), errors="coerce")
    score = pd.to_numeric(df.get("score", pd.Series(np.nan, index=df.index)), errors="coerce")
    rr = pd.to_numeric(df.get("reward_to_risk", pd.Series(np.nan, index=df.index)), errors="coerce")
    return (
        df.get("direction", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().eq("bearish")
        & df.get("strategy", pd.Series("", index=df.index)).fillna("").astype(str).eq("Bear Put Debit")
        & (ret5.le(-6.0) | ret10.le(-9.0) | ret20.le(-14.0))
        & close_rank.le(0.28)
        & score.ge(58.0)
        & rr.ge(0.65)
    )


def _annotate_supported_playbooks(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["validation_playbook"] = ""
    out["validation_playbook_horizon"] = np.nan
    out["validation_playbook_reason"] = ""
    mask = _tech_bullish_drift_mask(out)
    out.loc[mask, "validation_playbook"] = TECH_BULLISH_DRIFT_PLAYBOOK
    out.loc[mask, "validation_playbook_horizon"] = 5
    out.loc[
        mask,
        "validation_playbook_reason",
    ] = "Technology bullish 5D momentum, mild 1D drift, and positive options-flow bias"
    mask = _bullish_breakout_mask(out) & out["validation_playbook"].astype(str).eq("")
    out.loc[mask, "validation_playbook"] = BULLISH_BREAKOUT_PLAYBOOK
    out.loc[mask, "validation_playbook_horizon"] = 20
    out.loc[
        mask,
        "validation_playbook_reason",
    ] = "Bullish breakout: strong 5/10/20D price trend near the window high with defined-risk call-spread payoff"
    mask = _bearish_breakdown_mask(out) & out["validation_playbook"].astype(str).eq("")
    out.loc[mask, "validation_playbook"] = BEARISH_BREAKDOWN_PLAYBOOK
    out.loc[mask, "validation_playbook_horizon"] = 20
    out.loc[
        mask,
        "validation_playbook_reason",
    ] = "Bearish breakdown: strong 5/10/20D price trend near the window low with defined-risk put-spread payoff"
    return out


def _playbook_rows(df: pd.DataFrame, *, max_per_day: int = 5) -> pd.DataFrame:
    annotated = _annotate_supported_playbooks(df)
    if annotated.empty or "validation_playbook" not in annotated.columns:
        return pd.DataFrame()
    rows = annotated[annotated["validation_playbook"].astype(str).ne("")].copy()
    if rows.empty:
        return rows
    rows = rows.sort_values(["score", "hot_chain_premium"], ascending=[False, False])
    return rows.drop_duplicates("ticker", keep="first").head(max(1, int(max_per_day))).copy()


def _select_final_candidates(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    annotated = _annotate_supported_playbooks(df)
    pieces: List[pd.DataFrame] = []

    pieces.append(
        annotated.sort_values(
            ["score", "hot_chain_premium", "whale_premium"],
            ascending=[False, False, False],
        ).head(max_rows)
    )

    tradeable = annotated[annotated.get("entry_net", pd.Series(dtype=float)).notna()].copy()
    if not tradeable.empty:
        pieces.append(
            tradeable.sort_values(
                ["validation_playbook", "score", "reward_to_risk", "ret_5d_pct"],
                ascending=[False, False, False, False],
            ).head(max(25, max_rows // 2))
        )
        pieces.append(
            tradeable.sort_values(
                ["ret_5d_pct", "ret_10d_pct", "close_window_percentile", "reward_to_risk"],
                ascending=[False, False, False, False],
            ).head(max(25, max_rows // 2))
        )
        pieces.append(
            tradeable.sort_values(
                ["ret_5d_pct", "ret_10d_pct", "close_window_percentile", "reward_to_risk"],
                ascending=[True, True, True, False],
            ).head(max(15, max_rows // 3))
        )
        pieces.append(
            tradeable.sort_values(
                ["hot_chain_premium", "whale_premium", "score"],
                ascending=[False, False, False],
            ).head(max(25, max_rows // 2))
        )

    out = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=False, sort=False)
    out = out[~out.index.duplicated(keep="first")].copy()
    out["_selection_bucket"] = np.where(
        out.get("validation_playbook", pd.Series("", index=out.index)).fillna("").astype(str).ne(""),
        0,
        np.where(pd.to_numeric(out.get("score", pd.Series(0, index=out.index)), errors="coerce").ge(62), 1, 2),
    )
    out = out.sort_values(
        ["_selection_bucket", "score", "ret_5d_pct", "hot_chain_premium", "reward_to_risk"],
        ascending=[True, False, False, False, False],
    ).drop(columns=["_selection_bucket"], errors="ignore")
    return out.head(max_rows).reset_index(drop=True)


def build_signal_candidates(
    cache: DataCache,
    days: Sequence[dt.date],
    *,
    as_of: dt.date,
    lookback: int,
    max_rows: int = DEFAULT_MAX_DAILY_ROWS,
    include_whales: bool = True,
    whale_lookback_days: int = 5,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    window = _window_days(days, as_of, lookback)
    latest = _common_stock_universe(cache.screener(as_of), DEFAULT_MIN_UNDERLYING_PRICE)
    regime, sector_rotation = market_regime_summary(cache, days, as_of, lookback)
    if latest.empty:
        return pd.DataFrame(), regime, sector_rotation

    latest = _candidate_universe_from_latest(latest, max_rows)
    options = cache.option_snapshot(as_of)
    chain_oi = cache.chain_oi(as_of)
    position_symbols = load_open_position_underlyings(cache.root, as_of)
    live_chain_symbols, live_chain_dirs = load_live_chain_symbols(cache.root, as_of)
    latest_local_day = max(days) if days else as_of
    historical_as_of = as_of < latest_local_day
    latest_tickers = set(latest["ticker"].astype(str).str.upper())
    whale_window = window[-max(1, int(whale_lookback_days)) :] if include_whales else []
    whale_by_day = {day: cache.whale_symbols(day, latest_tickers) for day in whale_window}

    history_by_ticker: Dict[str, List[pd.Series]] = defaultdict(list)
    for day in window:
        day_df = cache.screener(day)
        if day_df.empty or "ticker" not in day_df.columns:
            continue
        sub = day_df[day_df["ticker"].astype(str).str.upper().isin(latest_tickers)]
        for _, hist_row in sub.iterrows():
            history_by_ticker[str(hist_row.get("ticker", "")).strip().upper()].append(hist_row)

    option_groups: Dict[str, pd.DataFrame] = {}
    if not options.empty and "ticker" in options.columns:
        option_groups = {str(ticker).upper(): group.copy() for ticker, group in options.groupby("ticker")}
    chain_groups: Dict[str, pd.DataFrame] = {}
    if not chain_oi.empty and "underlying_symbol" in chain_oi.columns:
        chain_groups = {str(ticker).upper(): group.copy() for ticker, group in chain_oi.groupby("underlying_symbol")}

    rows: List[Dict[str, Any]] = []
    for _, latest_row in latest.iterrows():
        ticker = str(latest_row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        hist_rows = history_by_ticker.get(ticker, [])
        if len(hist_rows) < max(3, min(lookback, 5)):
            continue
        closes = [_fnum(r.get("close")) for r in hist_rows]
        closes = [c for c in closes if math.isfinite(c)]
        if not closes:
            continue
        spot = closes[-1]
        ret_1d = _pct_return(latest_row.get("close"), latest_row.get("prev_close"))
        ret_5d = _window_return(closes, 5)
        ret_10d = _window_return(closes, 10)
        ret_20d = _window_return(closes, 20)
        close_rank = _close_percentile_in_window(closes)
        flow_series = [_flow_bias_from_row(r) for r in hist_rows]
        latest_flow = flow_series[-1] if flow_series else 0.0
        flow_persist = _safe_div(sum(1 for v in flow_series if v * latest_flow > 0), len(flow_series), 0.0) if latest_flow else 0.0
        ticker_options = option_groups.get(ticker, pd.DataFrame(columns=options.columns))
        hot_bias, hot_premium = _hot_bias(ticker_options, ticker)
        oi_bias = _oi_bias(chain_groups.get(ticker, pd.DataFrame(columns=chain_oi.columns)), ticker)
        whale_days = sum(1 for day in whale_window if ticker in whale_by_day.get(day, {}))
        whale_premium = sum(float(whale_by_day.get(day, {}).get(ticker, 0.0)) for day in whale_window)

        price_bull = _clip(
            50
            + (_fnum(ret_5d, 0.0) * 360)
            + (_fnum(ret_10d, 0.0) * 230)
            + (_fnum(ret_20d, 0.0) * 130)
            + (_fnum(ret_1d, 0.0) * 160)
            + ((_fnum(close_rank, 0.5) - 0.5) * 20)
        )
        price_bear = _clip(
            50
            - (_fnum(ret_5d, 0.0) * 360)
            - (_fnum(ret_10d, 0.0) * 230)
            - (_fnum(ret_20d, 0.0) * 130)
            - (_fnum(ret_1d, 0.0) * 160)
            + ((0.5 - _fnum(close_rank, 0.5)) * 20)
        )
        flow_bull = _clip(50 + latest_flow * 42 + flow_persist * 14)
        flow_bear = _clip(50 - latest_flow * 42 + flow_persist * 14)
        hot_bull = _clip(50 + hot_bias * 40)
        hot_bear = _clip(50 - hot_bias * 40)
        oi_bull = _clip(50 + oi_bias * 35)
        oi_bear = _clip(50 - oi_bias * 35)
        if include_whales and whale_window:
            whale_score = _clip(
                100 * _safe_div(whale_days, max(1, len(whale_window)), 0.0)
                + min(20, math.log10(max(1.0, whale_premium)) * 2)
            )
        else:
            whale_score = 50.0

        bull_score = 0.22 * flow_bull + 0.36 * price_bull + 0.16 * hot_bull + 0.10 * oi_bull + 0.16 * whale_score
        bear_score = 0.22 * flow_bear + 0.36 * price_bear + 0.16 * hot_bear + 0.10 * oi_bear + 0.16 * whale_score
        direction = "bullish" if bull_score >= bear_score else "bearish"
        score = max(bull_score, bear_score)
        margin = abs(bull_score - bear_score)
        earn = _parse_date(latest_row.get("next_earnings_date"))
        max_structure_dte = DEFAULT_MAX_STRUCTURE_DTE
        if earn is not None and earn > as_of:
            days_to_earn = max(0, (earn - as_of).days)
            if days_to_earn > DEFAULT_MIN_STRUCTURE_DTE:
                max_structure_dte = max(DEFAULT_MIN_STRUCTURE_DTE, min(DEFAULT_MAX_STRUCTURE_DTE, days_to_earn - 1))
        structure, structure_blockers = build_debit_structure(
            ticker_options,
            ticker=ticker,
            spot=spot,
            direction=direction,
            as_of=as_of,
            max_dte=max_structure_dte,
        )
        blockers: List[str] = []
        risk_flags: List[str] = []
        if score < 58:
            blockers.append(f"score_below_58:{score:.1f}")
        if margin < 4:
            blockers.append(f"direction_margin_low:{margin:.1f}")
        if structure is None:
            blockers.extend(structure_blockers)
        else:
            if structure.quote_sanity != "ok":
                blockers.append(structure.quote_sanity)
            if earn is not None and as_of <= earn <= structure.expiry:
                blockers.append(f"earnings_before_expiry:{earn.isoformat()}")
        if ticker in position_symbols:
            risk_flags.append("open_position_same_underlying")
        if regime.get("regime") == "risk_off" and direction == "bullish":
            risk_flags.append("risk_off_conflict_for_bullish_debit")
        if regime.get("regime") == "risk_on" and direction == "bearish":
            risk_flags.append("risk_on_conflict_for_bearish_debit")

        row: Dict[str, Any] = {
            "as_of": as_of.isoformat(),
            "ticker": ticker,
            "direction": direction,
            "score": round(float(score), 3),
            "bull_score": round(float(bull_score), 3),
            "bear_score": round(float(bear_score), 3),
            "direction_margin": round(float(margin), 3),
            "latest_close": spot,
            "ret_1d_pct": ret_1d * 100 if math.isfinite(ret_1d) else math.nan,
            "ret_5d_pct": ret_5d * 100 if math.isfinite(ret_5d) else math.nan,
            "ret_10d_pct": ret_10d * 100 if math.isfinite(ret_10d) else math.nan,
            "ret_20d_pct": ret_20d * 100 if math.isfinite(ret_20d) else math.nan,
            "close_window_percentile": close_rank,
            "flow_bias": latest_flow,
            "flow_persistence": flow_persist,
            "hot_chain_bias": hot_bias,
            "hot_chain_premium": hot_premium,
            "oi_bias": oi_bias,
            "whale_days": whale_days,
            "whale_lookback_days": len(whale_window),
            "whale_premium": whale_premium,
            "sector": str(latest_row.get("sector", "") or ""),
            "iv_rank": _fnum(latest_row.get("iv_rank")),
            "next_earnings_date": earn.isoformat() if earn else "",
            "market_regime": str(regime.get("regime", "unknown")),
            "market_regime_reason": str(regime.get("reason", "")),
            "live_chain_quote_sanity": (
                "historical_not_required"
                if historical_as_of
                else ("available" if ticker in live_chain_symbols else "missing_current_schwab_chain_artifact")
            ),
            "live_chain_artifact_dirs": ";".join(live_chain_dirs),
            "block_reasons": ";".join(dict.fromkeys(blockers)),
            "risk_flags": ";".join(dict.fromkeys(risk_flags)),
            "candidate_source": "trend_v2_multifactor",
        }
        row.update(_structure_to_dict(structure))
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, regime, sector_rotation
    out = _select_final_candidates(out, max_rows=max_rows)
    return _annotate_supported_playbooks(out.reset_index(drop=True)), regime, sector_rotation


def load_open_position_underlyings(root: Path, as_of: dt.date) -> set[str]:
    candidates = sorted(Path(root).glob("schwab_positions_*.json"))
    candidates.extend(sorted((Path(root) / as_of.isoformat()).glob("schwab_positions_*.json")))
    selected: Optional[Path] = None
    for path in candidates:
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
        day = _parse_date(date_match.group(1)) if date_match else None
        if day is not None and day <= as_of:
            selected = path
    symbols: set[str] = set()
    if selected is None:
        return symbols
    try:
        payload = json.loads(selected.read_text(encoding="utf-8"))
    except Exception:
        return symbols

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                lk = str(key).lower()
                if lk in {"symbol", "underlying_symbol", "underlying", "ticker"}:
                    text = str(value or "").strip().upper()
                    parsed = parse_occ(text)
                    if parsed:
                        symbols.add(parsed[0])
                    elif re.match(r"^[A-Z][A-Z0-9\.\-]{0,7}$", text):
                        symbols.add(text)
                else:
                    visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    visit(payload)
    return symbols


def load_live_chain_symbols(root: Path, as_of: dt.date) -> Tuple[set[str], List[str]]:
    symbols: set[str] = set()
    dirs: List[Path] = []
    daily_dir = Path(root) / "out" / f"daily_pipeline_{as_of.isoformat()}"
    candidates = [
        daily_dir / f"schwab_snapshot_{as_of.isoformat()}" / "chains",
        daily_dir / f"weekly_credit_fallback_{as_of.isoformat()}" / "live_pricer" / "chains",
    ]
    for directory in candidates:
        if not directory.is_dir():
            continue
        dirs.append(directory.resolve())
        for path in directory.glob("chain_*.json"):
            symbol = path.stem.removeprefix("chain_").strip().upper()
            if symbol:
                symbols.add(symbol)
    return symbols, [str(path) for path in dirs]


def _intrinsic_spread_value(row: pd.Series, future_close: float) -> float:
    strategy = str(row.get("strategy", ""))
    long_strike = _fnum(row.get("long_strike"))
    short_strike = _fnum(row.get("short_strike"))
    width = _fnum(row.get("width"), abs(short_strike - long_strike))
    if not all(math.isfinite(v) for v in (future_close, long_strike, short_strike, width)) or width <= 0:
        return math.nan
    if "Bull Call" in strategy:
        return max(0.0, min(width, future_close - long_strike))
    if "Bear Put" in strategy:
        return max(0.0, min(width, long_strike - future_close))
    return math.nan


def _quoted_vertical_exit_net(cache: DataCache, row: pd.Series, exit_day: dt.date) -> Tuple[float, str]:
    long_symbol = str(row.get("long_symbol", "") or "").strip().upper()
    short_symbol = str(row.get("short_symbol", "") or "").strip().upper()
    if not long_symbol or not short_symbol:
        return math.nan, "missing_leg_symbols"
    options = cache.option_snapshot(exit_day)
    if options.empty or "option_symbol" not in options.columns:
        return math.nan, "missing_exit_snapshot"
    sub = options[options["option_symbol"].isin({long_symbol, short_symbol})]
    if sub["option_symbol"].nunique() < 2:
        return math.nan, "missing_exit_leg_quote"
    indexed = sub.sort_values("quote_width" if "quote_width" in sub.columns else "option_symbol")
    indexed = indexed.drop_duplicates("option_symbol", keep="first").set_index("option_symbol")
    long_row = indexed.loc[long_symbol]
    short_row = indexed.loc[short_symbol]
    exit_net = _fnum(long_row.get("bid")) - _fnum(short_row.get("ask"))
    if not math.isfinite(exit_net):
        return math.nan, "invalid_exit_quote"
    return max(0.0, float(exit_net)), "exit_quotes_conservative"


def _scored_exit_value(
    row: pd.Series,
    *,
    cache: DataCache,
    all_days: Sequence[dt.date],
    planned_exit_day: dt.date,
    future_close: float,
) -> Tuple[float, str, dt.date]:
    expiry = _parse_date(row.get("expiry")) or _parse_date(row.get("target_expiry"))
    exit_day = planned_exit_day
    final = False
    if expiry is not None and planned_exit_day >= expiry:
        expiry_day = _market_day_on_or_before(all_days, expiry)
        signal_day = _parse_date(row.get("as_of"))
        if expiry_day is not None and (signal_day is None or expiry_day >= signal_day):
            exit_day = expiry_day
        final = True

    if not final:
        quoted, source = _quoted_vertical_exit_net(cache, row, exit_day)
        if math.isfinite(quoted):
            return quoted, source, exit_day

    screener = cache.screener(exit_day)
    if not screener.empty and "ticker" in screener.columns:
        ticker = str(row.get("ticker", "") or "").strip().upper()
        sub = screener[screener["ticker"].astype(str).str.upper().eq(ticker)]
        if not sub.empty:
            future_close = _fnum(sub.iloc[0].get("close"), future_close)
    intrinsic = _intrinsic_spread_value(row, future_close)
    if math.isfinite(intrinsic):
        return intrinsic, "expiry_intrinsic" if final else "intrinsic_proxy_missing_exit_quote", exit_day
    return math.nan, "missing_exit_value", exit_day


def score_candidate_outcomes(
    rows: pd.DataFrame,
    *,
    cache: DataCache,
    all_days: Sequence[dt.date],
    signal_date: dt.date,
    horizons: Sequence[int],
    baseline: str,
    tier: str,
    max_exit_date: Optional[dt.date] = None,
    slippage_pct: float = 0.05,
    fee_per_spread: float = 1.30,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        entry = _fnum(row.get("entry_net"))
        width = _fnum(row.get("width"))
        max_risk = max(_fnum(row.get("max_risk"), entry * 100.0), 1.0)
        for horizon in horizons:
            exit_day = _future_day(all_days, signal_date, int(horizon))
            base_outcome = {
                "baseline": baseline,
                "tier": tier,
                "signal_date": signal_date.isoformat(),
                "horizon": int(horizon),
                "exit_date": exit_day.isoformat() if exit_day is not None else "",
                "ticker": ticker,
                "direction": row.get("direction", ""),
                "strategy": row.get("strategy", ""),
                "trade_setup": row.get("trade_setup", ""),
                "expiry": row.get("expiry", row.get("target_expiry", "")),
                "entry_net": entry,
                "entry_after_slippage": math.nan,
                "exit_intrinsic": math.nan,
                "exit_source": "",
                "exit_after_slippage": math.nan,
                "future_close": math.nan,
                "pnl": math.nan,
                "net_r": math.nan,
                "win": False,
                "score_status": "UNSCORABLE",
                "score_status_reason": "",
                "market_regime": row.get("market_regime", ""),
                "score": _fnum(row.get("score")),
                "direction_margin": _fnum(row.get("direction_margin")),
                "ret_1d_pct": _fnum(row.get("ret_1d_pct")),
                "ret_5d_pct": _fnum(row.get("ret_5d_pct")),
                "ret_10d_pct": _fnum(row.get("ret_10d_pct")),
                "ret_20d_pct": _fnum(row.get("ret_20d_pct")),
                "close_window_percentile": _fnum(row.get("close_window_percentile")),
                "flow_bias": _fnum(row.get("flow_bias")),
                "flow_persistence": _fnum(row.get("flow_persistence")),
                "hot_chain_bias": _fnum(row.get("hot_chain_bias")),
                "hot_chain_premium": _fnum(row.get("hot_chain_premium")),
                "oi_bias": _fnum(row.get("oi_bias")),
                "whale_days": _fnum(row.get("whale_days")),
                "whale_premium": _fnum(row.get("whale_premium")),
                "sector": row.get("sector", ""),
                "quote_sanity": row.get("quote_sanity", ""),
                "quote_width": _fnum(row.get("quote_width")),
                "quote_width_to_entry": _safe_div(_fnum(row.get("quote_width")), entry, math.nan),
                "block_reasons": row.get("block_reasons", ""),
                "risk_flags": row.get("risk_flags", ""),
            }
            if not math.isfinite(entry):
                base_outcome["score_status_reason"] = "missing_entry_quote"
                out_rows.append(base_outcome)
                continue
            if exit_day is None:
                base_outcome["score_status_reason"] = "no_future_day_for_horizon"
                out_rows.append(base_outcome)
                continue
            if max_exit_date is not None and exit_day > max_exit_date:
                base_outcome["score_status_reason"] = "exit_after_validation_cutoff"
                out_rows.append(base_outcome)
                continue
            future_df = cache.screener(exit_day)
            if future_df.empty or "ticker" not in future_df.columns:
                base_outcome["score_status_reason"] = "missing_exit_screener"
                out_rows.append(base_outcome)
                continue
            fut = future_df[future_df["ticker"].eq(ticker)]
            if fut.empty:
                base_outcome["score_status_reason"] = "missing_exit_underlying"
                out_rows.append(base_outcome)
                continue
            future_close = _fnum(fut.iloc[0].get("close"))
            exit_value, exit_source, scored_exit_day = _scored_exit_value(
                row,
                cache=cache,
                all_days=all_days,
                planned_exit_day=exit_day,
                future_close=future_close,
            )
            if not math.isfinite(exit_value):
                base_outcome["score_status_reason"] = exit_source or "missing_exit_value"
                base_outcome["future_close"] = future_close
                out_rows.append(base_outcome)
                continue
            entry_after_slip = entry if str(row.get("long_symbol", "") or "").strip() else entry * (1.0 + slippage_pct)
            exit_after_slip = (
                max(0.0, exit_value)
                if exit_source == "exit_quotes_conservative"
                else max(0.0, exit_value * (1.0 - slippage_pct))
            )
            pnl = (exit_after_slip - entry_after_slip) * 100.0 - fee_per_spread
            net_r = pnl / max_risk
            score_status = "PARTIAL" if exit_source == "intrinsic_proxy_missing_exit_quote" else "SCORED"
            scored = {
                **base_outcome,
                "exit_date": scored_exit_day.isoformat(),
                "entry_after_slippage": entry_after_slip,
                "exit_intrinsic": _intrinsic_spread_value(row, future_close),
                "exit_source": exit_source,
                "exit_after_slippage": exit_after_slip,
                "future_close": future_close,
                "pnl": pnl,
                "net_r": net_r,
                "win": bool(net_r > 0),
                "score_status": score_status,
                "score_status_reason": exit_source,
            }
            out_rows.append(scored)
    return pd.DataFrame(out_rows)


def summarize_scorecard(outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "baseline",
        "tier",
        "horizon",
        "signal_count",
        "scored_count",
        "partial_count",
        "unscorable_count",
        "win_rate",
        "avg_net_r",
        "median_net_r",
        "profit_factor",
        "worst_losing_streak",
        "drawdown_proxy_r",
        "tradeable_with_real_quotes_pct",
        "avg_bid_ask_spread",
        "avg_quote_width_to_entry",
        "blocked_pct",
        "top_block_reasons",
        "score_status_counts",
        "regime",
    ]
    if outcomes.empty:
        return pd.DataFrame(columns=columns)
    rows: List[Dict[str, Any]] = []
    for keys, sub in outcomes.groupby(["baseline", "tier", "horizon"], dropna=False):
        baseline, tier, horizon = keys
        statuses = sub.get("score_status", pd.Series("SCORED", index=sub.index)).fillna("").astype(str)
        scored_mask = statuses.isin({"SCORED", "PARTIAL"})
        net_r = pd.to_numeric(sub.loc[scored_mask, "net_r"], errors="coerce").dropna()
        blockers = Counter()
        for text in sub.get("block_reasons", pd.Series(dtype=str)).fillna("").astype(str):
            for token in text.split(";"):
                token = token.strip()
                if token:
                    blockers[token] += 1
        regimes = sub.get("market_regime", pd.Series(dtype=str)).fillna("").astype(str)
        regime = regimes.mode().iloc[0] if not regimes.empty else ""
        status_counts = statuses.value_counts(dropna=False).to_dict()
        real_quote_pct = float(sub.get("exit_source", pd.Series("", index=sub.index)).fillna("").astype(str).eq("exit_quotes_conservative").mean())
        avg_quote_width = pd.to_numeric(sub.get("quote_width", pd.Series(dtype=float)), errors="coerce").mean()
        avg_quote_width_to_entry = pd.to_numeric(
            sub.get("quote_width_to_entry", pd.Series(dtype=float)), errors="coerce"
        ).mean()
        rows.append(
            {
                "baseline": baseline,
                "tier": tier,
                "horizon": int(horizon),
                "signal_count": int(len(sub)),
                "scored_count": int(statuses.eq("SCORED").sum()),
                "partial_count": int(statuses.eq("PARTIAL").sum()),
                "unscorable_count": int(statuses.eq("UNSCORABLE").sum()),
                "win_rate": float(net_r.gt(0).mean()) if len(net_r) else math.nan,
                "avg_net_r": float(net_r.mean()) if len(net_r) else math.nan,
                "median_net_r": float(net_r.median()) if len(net_r) else math.nan,
                "profit_factor": _profit_factor(net_r),
                "worst_losing_streak": _max_losing_streak(net_r),
                "drawdown_proxy_r": _max_drawdown(net_r),
                "tradeable_with_real_quotes_pct": real_quote_pct,
                "avg_bid_ask_spread": float(avg_quote_width) if math.isfinite(_fnum(avg_quote_width)) else math.nan,
                "avg_quote_width_to_entry": (
                    float(avg_quote_width_to_entry)
                    if math.isfinite(_fnum(avg_quote_width_to_entry))
                    else math.nan
                ),
                "blocked_pct": float(sub.get("block_reasons", pd.Series("", index=sub.index)).fillna("").astype(str).ne("").mean()),
                "top_block_reasons": ", ".join(f"{k}:{v}" for k, v in blockers.most_common(5)),
                "score_status_counts": ", ".join(f"{k}:{v}" for k, v in sorted(status_counts.items())),
                "regime": regime,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _old_trend_rows(root: Path, signal_date: dt.date, lookback: int, max_rows: int) -> pd.DataFrame:
    out_dir = root / "out" / "trend_analysis"
    suffix = f"{signal_date.isoformat()}-L{lookback}"
    for name in [
        f"trend-analysis-actionable-{suffix}.csv",
        f"trend-analysis-candidates-{suffix}.csv",
        f"trend_analysis_raw_{suffix}.csv",
    ]:
        path = out_dir / name
        if path.exists():
            try:
                df = pd.read_csv(path, low_memory=False)
            except Exception:
                continue
            if df.empty:
                return pd.DataFrame()
            df = df.copy()
            if "entry_net" not in df.columns and "est_cost" in df.columns:
                df["entry_net"] = pd.to_numeric(df["est_cost"], errors="coerce")
            if "max_risk" not in df.columns:
                df["max_risk"] = pd.to_numeric(df.get("entry_net", 0), errors="coerce").fillna(0) * 100
            if "width" not in df.columns and "spread_width" in df.columns:
                df["width"] = df["spread_width"]
            if "expiry" not in df.columns and "target_expiry" in df.columns:
                df["expiry"] = df["target_expiry"]
            if "score" not in df.columns and "swing_score" in df.columns:
                df["score"] = df["swing_score"]
            if "trade_setup" not in df.columns and "strike_setup" in df.columns:
                df["trade_setup"] = df["strike_setup"]
            df["candidate_source"] = "legacy_trend_analysis"
            return df.head(max_rows)
    return pd.DataFrame()


def _top_liquidity_random_rows(candidates: pd.DataFrame, signal_date: dt.date, count: int) -> pd.DataFrame:
    if candidates.empty or count <= 0:
        return pd.DataFrame()
    pool = candidates[candidates["entry_net"].notna()].copy()
    if pool.empty:
        return pd.DataFrame()
    pool["_random_rank"] = pool["ticker"].map(lambda t: _deterministic_hash(signal_date.isoformat(), t))
    return pool.sort_values("_random_rank").head(count).drop(columns=["_random_rank"])


def _unusual_volume_rows(candidates: pd.DataFrame, count: int) -> pd.DataFrame:
    if candidates.empty or count <= 0:
        return pd.DataFrame()
    work = candidates.copy()
    work["_activity_rank"] = pd.to_numeric(work.get("hot_chain_premium", 0), errors="coerce").fillna(0) + pd.to_numeric(
        work.get("whale_premium", 0), errors="coerce"
    ).fillna(0)
    return work.sort_values("_activity_rank", ascending=False).head(count).drop(columns=["_activity_rank"])


def _spy_qqq_baseline_rows(cache: DataCache, signal_date: dt.date, regime: Dict[str, Any]) -> pd.DataFrame:
    latest = cache.screener(signal_date)
    options = cache.option_snapshot(signal_date)
    if latest.empty:
        return pd.DataFrame()
    direction = "bearish" if regime.get("regime") == "risk_off" else "bullish"
    rows: List[Dict[str, Any]] = []
    for ticker in ("SPY", "QQQ"):
        sub = latest[latest["ticker"].astype(str).str.upper().eq(ticker)] if "ticker" in latest.columns else pd.DataFrame()
        if sub.empty:
            continue
        spot = _fnum(sub.iloc[0].get("close"))
        structure, blockers = build_debit_structure(options, ticker=ticker, spot=spot, direction=direction, as_of=signal_date)
        if structure is None:
            continue
        row = {
            "as_of": signal_date.isoformat(),
            "ticker": ticker,
            "direction": direction,
            "score": 50.0,
            "market_regime": regime.get("regime", ""),
            "block_reasons": ";".join(blockers),
            "candidate_source": "spy_qqq_directional_baseline",
        }
        row.update(_structure_to_dict(structure))
        rows.append(row)
    return pd.DataFrame(rows)


def _hindsight_rows(
    candidates: pd.DataFrame,
    *,
    cache: DataCache,
    all_days: Sequence[dt.date],
    signal_date: dt.date,
    regime: Optional[Dict[str, Any]] = None,
    horizon: int = 5,
    count: int = 5,
) -> pd.DataFrame:
    exit_day = _future_day(all_days, signal_date, horizon)
    if exit_day is None:
        return pd.DataFrame()
    today = _common_stock_universe(cache.screener(signal_date), DEFAULT_MIN_UNDERLYING_PRICE)
    future = cache.screener(exit_day)
    if today.empty or future.empty or "ticker" not in today.columns or "ticker" not in future.columns:
        return pd.DataFrame()
    merged = today[["ticker", "close"]].merge(
        future[["ticker", "close"]],
        on="ticker",
        suffixes=("_entry", "_exit"),
    )
    merged["_ret"] = (
        pd.to_numeric(merged["close_exit"], errors="coerce")
        - pd.to_numeric(merged["close_entry"], errors="coerce")
    ) / pd.to_numeric(merged["close_entry"], errors="coerce")
    top = merged[np.isfinite(merged["_ret"])].copy()
    if top.empty:
        return pd.DataFrame()
    top["_abs_ret"] = top["_ret"].abs()
    options = cache.option_snapshot(signal_date)
    if options.empty:
        return pd.DataFrame()
    option_groups = {str(ticker).upper(): group.copy() for ticker, group in options.groupby("ticker")}
    rows: List[Dict[str, Any]] = []
    for _, row in top.sort_values("_abs_ret", ascending=False).head(max(count * 6, 20)).iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        direction = "bullish" if _fnum(row.get("_ret"), 0.0) >= 0 else "bearish"
        structure, blockers = build_debit_structure(
            option_groups.get(ticker, pd.DataFrame(columns=options.columns)),
            ticker=ticker,
            spot=_fnum(row.get("close_entry")),
            direction=direction,
            as_of=signal_date,
        )
        if structure is None:
            continue
        out_row: Dict[str, Any] = {
            "as_of": signal_date.isoformat(),
            "ticker": ticker,
            "direction": direction,
            "score": 100.0,
            "market_regime": str((regime or {}).get("regime", "")),
            "block_reasons": "hindsight_benchmark_not_tradable",
            "candidate_source": "hindsight_missed_mover_benchmark",
            "forward_return_pct": float(_fnum(row.get("_ret")) * 100.0),
        }
        out_row.update(_structure_to_dict(structure))
        rows.append(out_row)
        if len(rows) >= count:
            break
    return pd.DataFrame(rows)


def run_validation(
    *,
    cache: DataCache,
    root: Path,
    as_of: dt.date,
    lookback: int,
    validate_days: int,
    horizons: Sequence[int],
    max_daily_rows: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    days = available_market_days(root)
    validation_signals = [d for d in days if d < as_of]
    validation_signals = validation_signals[-max(1, int(validate_days)) :]
    outcomes: List[pd.DataFrame] = []
    for signal_date in validation_signals:
        candidates, regime, _sector = build_signal_candidates(
            cache,
            days,
            as_of=signal_date,
            lookback=lookback,
            max_rows=max_daily_rows,
            include_whales=False,
        )
        tradeable = candidates[candidates.get("entry_net", pd.Series(dtype=float)).notna()].copy()
        clean = tradeable[tradeable.get("block_reasons", pd.Series("", index=tradeable.index)).fillna("").astype(str).eq("")]
        v2_top = clean.head(5)
        if not v2_top.empty:
            outcomes.append(
                score_candidate_outcomes(
                    v2_top,
                    cache=cache,
                    all_days=days,
                    signal_date=signal_date,
                    horizons=horizons,
                    baseline="trend_v2",
                    tier="actionable_gate",
                    max_exit_date=as_of,
                )
            )
        watch = tradeable.head(15)
        if not watch.empty:
            outcomes.append(
                score_candidate_outcomes(
                    watch,
                    cache=cache,
                    all_days=days,
                    signal_date=signal_date,
                    horizons=horizons,
                    baseline="trend_v2",
                    tier="watchlist",
                    max_exit_date=as_of,
                )
            )
        playbook = _playbook_rows(tradeable, max_per_day=5)
        if not playbook.empty:
            for playbook_name, playbook_group in playbook.groupby("validation_playbook"):
                if not str(playbook_name).strip():
                    continue
                outcomes.append(
                    score_candidate_outcomes(
                        playbook_group,
                        cache=cache,
                        all_days=days,
                        signal_date=signal_date,
                        horizons=horizons,
                        baseline="trend_v2",
                        tier=str(playbook_name),
                        max_exit_date=as_of,
                    )
                )
        old_rows = _old_trend_rows(root, signal_date, lookback, max_rows=5)
        if not old_rows.empty:
            outcomes.append(
                score_candidate_outcomes(
                    old_rows,
                    cache=cache,
                    all_days=days,
                    signal_date=signal_date,
                    horizons=horizons,
                    baseline="legacy_trend_analysis",
                    tier="emitted",
                    max_exit_date=as_of,
                )
            )
        random_rows = _top_liquidity_random_rows(tradeable, signal_date, max(1, len(v2_top) or 3))
        if not random_rows.empty:
            outcomes.append(
                score_candidate_outcomes(
                    random_rows,
                    cache=cache,
                    all_days=days,
                    signal_date=signal_date,
                    horizons=horizons,
                    baseline="random_same_date_liquidity",
                    tier="baseline",
                    max_exit_date=as_of,
                )
            )
        unusual_rows = _unusual_volume_rows(tradeable, max(1, len(v2_top) or 3))
        if not unusual_rows.empty:
            outcomes.append(
                score_candidate_outcomes(
                    unusual_rows,
                    cache=cache,
                    all_days=days,
                    signal_date=signal_date,
                    horizons=horizons,
                    baseline="unusual_options_volume_only",
                    tier="baseline",
                    max_exit_date=as_of,
                )
            )
        spy_rows = _spy_qqq_baseline_rows(cache, signal_date, regime)
        if not spy_rows.empty:
            outcomes.append(
                score_candidate_outcomes(
                    spy_rows,
                    cache=cache,
                    all_days=days,
                    signal_date=signal_date,
                    horizons=horizons,
                    baseline="spy_qqq_directional",
                    tier="baseline",
                    max_exit_date=as_of,
                )
            )
        hindsight = _hindsight_rows(
            tradeable,
            cache=cache,
            all_days=days,
            signal_date=signal_date,
            regime=regime,
            horizon=5,
            count=max(3, len(v2_top)),
        )
        if not hindsight.empty:
            outcomes.append(
                score_candidate_outcomes(
                    hindsight,
                    cache=cache,
                    all_days=days,
                    signal_date=signal_date,
                    horizons=horizons,
                    baseline="missed_mover_hindsight",
                    tier="benchmark_not_tradable",
                    max_exit_date=as_of,
                )
            )
    nonempty_outcomes = [df for df in outcomes if not df.empty]
    outcome_df = pd.concat(nonempty_outcomes, ignore_index=True) if nonempty_outcomes else pd.DataFrame()
    return outcome_df, summarize_scorecard(outcome_df)


def _lookup_summary(scorecard: pd.DataFrame, baseline: str, tier: str, horizon: int) -> Optional[pd.Series]:
    if scorecard.empty:
        return None
    sub = scorecard[
        scorecard["baseline"].astype(str).eq(baseline)
        & scorecard["tier"].astype(str).eq(tier)
        & pd.to_numeric(scorecard["horizon"], errors="coerce").eq(int(horizon))
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def _playbook_summary(scorecard: pd.DataFrame, playbook: str, horizon: int = 5) -> Optional[pd.Series]:
    return _lookup_summary(scorecard, "trend_v2", playbook, horizon)


def _playbook_is_proven(row: Optional[pd.Series], min_samples: int) -> bool:
    return (
        row is not None
        and int(row.get("signal_count", 0)) >= int(min_samples)
        and _fnum(row.get("avg_net_r")) >= DEFAULT_MIN_PLAYBOOK_AVG_R
        and _fnum(row.get("profit_factor")) >= DEFAULT_MIN_PLAYBOOK_PROFIT_FACTOR
    )


def _hard_trade_blockers(row: pd.Series) -> List[str]:
    blockers = [
        token.strip()
        for token in str(row.get("block_reasons", "") or "").split(";")
        if token.strip()
    ]
    hard_prefixes = (
        "no_hot_chain_quotes",
        "no_liquid_directional_options",
        "no_vertical_structure_found",
        "debit_too_expensive",
        "wide_quote",
        "open_position_same_underlying",
        "earnings_before_expiry",
    )
    return [token for token in blockers if token.startswith(hard_prefixes)]


def classify_current_candidates(
    candidates: pd.DataFrame,
    scorecard: pd.DataFrame,
    *,
    primary_horizon: int = 5,
    min_samples: int = DEFAULT_MIN_VALIDATION_SAMPLES,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if candidates.empty:
        return candidates.copy(), {"verdict": "NO_CANDIDATES", "reason": "no candidates generated"}
    out = candidates.copy()
    v2 = _lookup_summary(scorecard, "trend_v2", "actionable_gate", primary_horizon)
    legacy = _lookup_summary(scorecard, "legacy_trend_analysis", "emitted", primary_horizon)
    random_base = _lookup_summary(scorecard, "random_same_date_liquidity", "baseline", primary_horizon)
    unusual_base = _lookup_summary(scorecard, "unusual_options_volume_only", "baseline", primary_horizon)
    tech_drift = _playbook_summary(scorecard, TECH_BULLISH_DRIFT_PLAYBOOK, horizon=5)
    bullish_breakout = _playbook_summary(scorecard, BULLISH_BREAKOUT_PLAYBOOK, horizon=20)
    bearish_breakdown = _playbook_summary(scorecard, BEARISH_BREAKDOWN_PLAYBOOK, horizon=20)
    baseline_avg = max(
        [
            _fnum(legacy.get("avg_net_r")) if legacy is not None else -math.inf,
            _fnum(random_base.get("avg_net_r")) if random_base is not None else -math.inf,
            _fnum(unusual_base.get("avg_net_r")) if unusual_base is not None else -math.inf,
        ]
    )
    proof = {
        "primary_horizon": primary_horizon,
        "v2_samples": int(v2.get("signal_count")) if v2 is not None else 0,
        "v2_avg_net_r": _fnum(v2.get("avg_net_r")) if v2 is not None else math.nan,
        "v2_profit_factor": _fnum(v2.get("profit_factor")) if v2 is not None else math.nan,
        "best_comparison_avg_net_r": baseline_avg if math.isfinite(baseline_avg) else math.nan,
        "tech_bullish_drift_samples": int(tech_drift.get("signal_count")) if tech_drift is not None else 0,
        "tech_bullish_drift_avg_net_r": _fnum(tech_drift.get("avg_net_r")) if tech_drift is not None else math.nan,
        "tech_bullish_drift_profit_factor": _fnum(tech_drift.get("profit_factor")) if tech_drift is not None else math.nan,
        "bullish_breakout_samples": int(bullish_breakout.get("signal_count")) if bullish_breakout is not None else 0,
        "bullish_breakout_avg_net_r": _fnum(bullish_breakout.get("avg_net_r")) if bullish_breakout is not None else math.nan,
        "bullish_breakout_profit_factor": _fnum(bullish_breakout.get("profit_factor")) if bullish_breakout is not None else math.nan,
        "bearish_breakdown_samples": int(bearish_breakdown.get("signal_count")) if bearish_breakdown is not None else 0,
        "bearish_breakdown_avg_net_r": _fnum(bearish_breakdown.get("avg_net_r")) if bearish_breakdown is not None else math.nan,
        "bearish_breakdown_profit_factor": _fnum(bearish_breakdown.get("profit_factor")) if bearish_breakdown is not None else math.nan,
    }
    proven = (
        v2 is not None
        and int(v2.get("signal_count", 0)) >= int(min_samples)
        and _fnum(v2.get("avg_net_r")) >= DEFAULT_MIN_VALIDATION_AVG_R
        and _fnum(v2.get("profit_factor")) >= DEFAULT_MIN_VALIDATION_PROFIT_FACTOR
        and (not math.isfinite(baseline_avg) or _fnum(v2.get("avg_net_r")) > baseline_avg)
    )
    proven_playbooks = {
        TECH_BULLISH_DRIFT_PLAYBOOK: _playbook_is_proven(tech_drift, min_samples=min_samples),
        BULLISH_BREAKOUT_PLAYBOOK: _playbook_is_proven(bullish_breakout, min_samples=min_samples),
        BEARISH_BREAKDOWN_PLAYBOOK: _playbook_is_proven(bearish_breakdown, min_samples=min_samples),
    }
    out = _annotate_supported_playbooks(out)
    classifications: List[str] = []
    reasons: List[str] = []
    for _, row in out.iterrows():
        blockers = str(row.get("block_reasons", "") or "").strip()
        risk_flags = str(row.get("risk_flags", "") or "").strip()
        live_chain_ok = str(row.get("live_chain_quote_sanity", "")).strip().lower() in {
            "available",
            "historical_not_required",
        }
        playbook = str(row.get("validation_playbook", "") or "").strip()
        playbook_proven = bool(playbook and proven_playbooks.get(playbook, False))
        hard_blockers = _hard_trade_blockers(row)
        if proven and not blockers and _fnum(row.get("score")) >= 62:
            if live_chain_ok:
                classifications.append("TRADE")
                reason = "passes current gates and v2 validation beat comparison baselines"
                if risk_flags:
                    reason += f"; risk flags: {risk_flags}"
                reasons.append(reason)
            else:
                classifications.append("WATCH")
                reasons.append("watch only: current Schwab live-chain quote artifact is missing")
        elif playbook_proven:
            if not hard_blockers and live_chain_ok:
                classifications.append("TRADE")
                reason = f"passes validation-supported playbook: {row.get('validation_playbook_reason', playbook)}"
                if risk_flags:
                    reason += f"; risk flags: {risk_flags}"
                reasons.append(reason)
            else:
                classifications.append("WATCH")
                detail = "; ".join(hard_blockers) if hard_blockers else "current Schwab live-chain quote artifact is missing"
                reasons.append(
                    f"validated playbook watch only: {row.get('validation_playbook_reason', playbook)}; blocked by {detail}"
                )
        elif blockers:
            classifications.append("AVOID")
            reasons.append(blockers)
        else:
            classifications.append("WATCH")
            if not proven:
                reasons.append("watch only: v2 validation has not cleared actionable acceptance bar")
            else:
                reasons.append("watch only: current score below actionable threshold")
    out["classification"] = classifications
    out["classification_reason"] = reasons
    if proven:
        proof["verdict"] = "PROVEN_FOR_ACTIONABLE"
    elif any(proven_playbooks.values()):
        proof["verdict"] = "PROVEN_PLAYBOOKS_FOR_ACTIONABLE"
    else:
        proof["verdict"] = "NO_PROVEN_EDGE_FOR_ACTIONABLE"
    return out, proof


def build_news_summary(root: Path, as_of: dt.date, tickers: Sequence[str]) -> pd.DataFrame:
    browser_dir = root / as_of.isoformat() / "browser_text"
    rows: List[Dict[str, Any]] = []
    wanted = {str(t).upper() for t in tickers if str(t).strip()}
    if not browser_dir.is_dir():
        return pd.DataFrame(
            [
                {
                    "source": "local_browser_text",
                    "topic": "sentiment/news",
                    "status": "unavailable",
                    "summary": f"No local browser_text captures found for {as_of.isoformat()}.",
                    "urls": "",
                    "path": "",
                }
            ]
        )
    for path in sorted(browser_dir.glob("browser-text-capture-news-*.txt")):
        name = path.name.upper()
        topic = path.stem.replace("browser-text-capture-news-", "")
        if wanted and not any(t in name for t in wanted) and "MACRO" not in name:
            continue
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()]
        except Exception:
            continue
        urls = [line for line in lines if line.startswith(("http://", "https://"))][:8]
        useful = [line for line in lines if line and not line.startswith("http")][:8]
        rows.append(
            {
                "source": "local_browser_text",
                "topic": topic,
                "status": "available",
                "summary": " ".join(useful)[:900],
                "urls": ";".join(urls),
                "path": str(path.resolve()),
            }
        )
    if not rows:
        rows.append(
            {
                "source": "local_browser_text",
                "topic": "sentiment/news",
                "status": "unavailable",
                "summary": "Local browser_text captures exist, but none matched current candidates or macro.",
                "urls": "",
                "path": str(browser_dir.resolve()),
            }
        )
    return pd.DataFrame(rows)


def build_missed_mover_audit(
    *,
    cache: DataCache,
    root: Path,
    as_of: dt.date,
    lookback: int,
    candidates: pd.DataFrame,
    horizons: Sequence[int],
    move_threshold: float = 0.08,
    max_daily_rows: int = DEFAULT_MAX_DAILY_ROWS,
) -> pd.DataFrame:
    days = available_market_days(root)
    start_idx = max(0, days.index(as_of) - max(lookback, 20)) if as_of in days else 0
    signal_days = [d for d in days[start_idx:] if d < as_of]
    fallback_candidate_tickers = set(candidates.get("ticker", pd.Series(dtype=str)).astype(str).str.upper())
    candidate_cache: Dict[dt.date, Tuple[set[str], set[str], set[str]]] = {}

    def coverage_sets(day: dt.date) -> Tuple[set[str], set[str], set[str]]:
        if day in candidate_cache:
            return candidate_cache[day]
        try:
            day_candidates, _regime, _sector = build_signal_candidates(
                cache,
                days,
                as_of=day,
                lookback=lookback,
                max_rows=max(20, int(max_daily_rows)),
                include_whales=False,
            )
        except Exception:
            day_candidates = pd.DataFrame()
        if day_candidates.empty:
            all_seen = set(fallback_candidate_tickers if day == as_of else set())
            tradeable = set()
            playbook = set()
        else:
            tickers = day_candidates.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
            all_seen = {ticker for ticker in tickers if ticker}
            tradeable_df = day_candidates[day_candidates.get("entry_net", pd.Series(dtype=float)).notna()].copy()
            tradeable = set(tradeable_df.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).str.upper())
            playbook_df = _playbook_rows(tradeable_df, max_per_day=max(20, int(max_daily_rows)))
            playbook = set(playbook_df.get("ticker", pd.Series(dtype=str)).fillna("").astype(str).str.upper())
        candidate_cache[day] = (all_seen, tradeable, playbook)
        return candidate_cache[day]

    rows: List[Dict[str, Any]] = []
    for day in signal_days:
        entry = _common_stock_universe(cache.screener(day), DEFAULT_MIN_UNDERLYING_PRICE)
        if entry.empty:
            continue
        seen_tickers, tradeable_tickers, playbook_tickers = coverage_sets(day)
        for horizon in horizons:
            exit_day = _future_day(days, day, int(horizon))
            if exit_day is None or exit_day > as_of:
                continue
            exit_df = cache.screener(exit_day)
            if exit_df.empty or "ticker" not in exit_df.columns:
                continue
            merged = entry[["ticker", "close", "sector"]].merge(
                exit_df[["ticker", "close"]],
                on="ticker",
                suffixes=("_entry", "_exit"),
            )
            merged["_ret"] = (
                pd.to_numeric(merged["close_exit"], errors="coerce")
                - pd.to_numeric(merged["close_entry"], errors="coerce")
            ) / pd.to_numeric(merged["close_entry"], errors="coerce")
            movers = merged[merged["_ret"].abs().ge(move_threshold)].copy()
            for _, row in movers.sort_values("_ret", key=lambda s: s.abs(), ascending=False).head(20).iterrows():
                ticker = str(row.get("ticker", "")).upper()
                if ticker in playbook_tickers:
                    coverage = "validation_playbook"
                elif ticker in tradeable_tickers:
                    coverage = "tradeable_candidate"
                elif ticker in seen_tickers:
                    coverage = "raw_candidate"
                else:
                    coverage = "missed_by_signal_day_v2"
                rows.append(
                    {
                        "signal_date": day.isoformat(),
                        "horizon": int(horizon),
                        "exit_date": exit_day.isoformat(),
                        "ticker": ticker,
                        "sector": row.get("sector", ""),
                        "forward_return_pct": float(row["_ret"] * 100),
                        "coverage": coverage,
                    }
                )
    return pd.DataFrame(rows)


def _write_markdown_table(df: pd.DataFrame, columns: Sequence[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_None._"
    view = df.loc[:, [c for c in columns if c in df.columns]].head(max_rows).copy()
    return view.to_markdown(index=False)


def build_report(
    *,
    as_of: dt.date,
    lookback: int,
    regime: Dict[str, Any],
    proof: Dict[str, Any],
    candidates: pd.DataFrame,
    actionable: pd.DataFrame,
    watchlist: pd.DataFrame,
    blocked: pd.DataFrame,
    scorecard: pd.DataFrame,
    sector_rotation: pd.DataFrame,
    news_summary: pd.DataFrame,
    missed_movers: pd.DataFrame,
    paths: Dict[str, Path],
) -> str:
    if news_summary.empty:
        macro_news = pd.DataFrame()
        micro_news = pd.DataFrame()
        geopolitical_news = pd.DataFrame()
    else:
        topic = news_summary.get("topic", pd.Series("", index=news_summary.index)).fillna("").astype(str).str.lower()
        summary_text = news_summary.get("summary", pd.Series("", index=news_summary.index)).fillna("").astype(str).str.lower()
        macro_mask = topic.str.contains("macro|fed|cpi|jobs|rate|yield|dollar|oil|credit|vix", regex=True) | summary_text.str.contains(
            "fed|cpi|jobs|rate|yield|dollar|oil|credit|vix", regex=True
        )
        geopolitical_mask = topic.str.contains("geo|tariff|sanction|war|election|regulat", regex=True) | summary_text.str.contains(
            "tariff|sanction|war|election|regulat|supply-chain|supply chain", regex=True
        )
        macro_news = news_summary[macro_mask].copy()
        geopolitical_news = news_summary[geopolitical_mask].copy()
        micro_news = news_summary[~macro_mask & ~geopolitical_mask].copy()
    if not actionable.empty:
        trade_verdict = "TRADE queue is enabled by validation."
    elif str(proof.get("verdict", "")).upper() == "PROVEN_PLAYBOOKS_FOR_ACTIONABLE":
        trade_verdict = "TRADE queue is enabled by validated playbooks."
    else:
        trade_verdict = "No proven actionable edge today. Use the watchlist and blockers only."
    lines = [
        f"# UW Options Trend Pipeline v2 - {as_of.isoformat()}",
        "",
        "## Verdict",
        f"- {trade_verdict}",
        f"- Acceptance verdict: `{proof.get('verdict', 'unknown')}`",
        f"- Primary horizon: {proof.get('primary_horizon', 5)} market days",
        f"- V2 validation: samples={proof.get('v2_samples', 0)}, avg R={_fmt_float(proof.get('v2_avg_net_r'), 3)}, PF={_fmt_float(proof.get('v2_profit_factor'), 2)}",
        f"- Bullish breakout: samples={proof.get('bullish_breakout_samples', 0)}, avg R={_fmt_float(proof.get('bullish_breakout_avg_net_r'), 3)}, PF={_fmt_float(proof.get('bullish_breakout_profit_factor'), 2)}",
        f"- Bearish breakdown: samples={proof.get('bearish_breakdown_samples', 0)}, avg R={_fmt_float(proof.get('bearish_breakdown_avg_net_r'), 3)}, PF={_fmt_float(proof.get('bearish_breakdown_profit_factor'), 2)}",
        "",
        "## Market Regime",
        f"- {regime.get('reason', 'unknown')}",
        f"- VIX: {_fmt_float(regime.get('vix_close'), 2)} ({_fmt_float(regime.get('vix_1d_return_pct'), 2, '%')} 1D)",
        f"- SPY/QQQ/IWM 5D: {_fmt_float(regime.get('spy_5d_return_pct'), 2, '%')} / {_fmt_float(regime.get('qqq_5d_return_pct'), 2, '%')} / {_fmt_float(regime.get('iwm_5d_return_pct'), 2, '%')}",
        "",
        "## Macro Context",
        _write_markdown_table(macro_news, ["source", "topic", "status", "summary", "urls"], max_rows=8),
        "",
        "## Micro / Company Catalyst Context",
        _write_markdown_table(micro_news, ["source", "topic", "status", "summary", "urls"], max_rows=12),
        "",
        "## Geopolitical / Policy Context",
        _write_markdown_table(geopolitical_news, ["source", "topic", "status", "summary", "urls"], max_rows=8),
        "",
        "## Strongest Options-Market Themes",
        _write_markdown_table(
            sector_rotation,
            ["sector", "symbols", "avg_1d_return_pct", "positive_breadth_pct", "flow_premium"],
            max_rows=8,
        ),
        "",
        "## Actionable Trades",
        _write_markdown_table(
            actionable,
            [
                "classification",
                "ticker",
                "direction",
                "strategy",
                "trade_setup",
                "entry_net",
                "max_risk",
                "max_profit",
                "reward_to_risk",
                "ret_5d_pct",
                "score",
                "validation_playbook",
                "live_chain_quote_sanity",
                "risk_flags",
            ],
            max_rows=5,
        ),
        "",
        "## Watchlist / Research Setups",
        _write_markdown_table(
            watchlist,
            [
                "classification",
                "ticker",
                "direction",
                "strategy",
                "trade_setup",
                "entry_net",
                "max_risk",
                "max_profit",
                "reward_to_risk",
                "ret_5d_pct",
                "score",
                "validation_playbook",
                "live_chain_quote_sanity",
                "block_reasons",
                "risk_flags",
            ],
            max_rows=15,
        ),
        "",
        "## Blocked High-Interest Candidates",
        _write_markdown_table(
            blocked,
            [
                "classification",
                "ticker",
                "direction",
                "strategy",
                "trade_setup",
                "entry_net",
                "max_risk",
                "max_profit",
                "reward_to_risk",
                "ret_5d_pct",
                "score",
                "validation_playbook",
                "live_chain_quote_sanity",
                "block_reasons",
                "risk_flags",
            ],
            max_rows=20,
        ),
        "",
        "## Validation-Supported Playbooks",
        _write_markdown_table(
            scorecard[
                scorecard.get("baseline", pd.Series(dtype=str)).astype(str).eq("trend_v2")
                & scorecard.get("tier", pd.Series(dtype=str)).astype(str).str.startswith("playbook_")
            ]
            if not scorecard.empty
            else pd.DataFrame(),
            [
                "tier",
                "horizon",
                "signal_count",
                "win_rate",
                "avg_net_r",
                "median_net_r",
                "profit_factor",
                "worst_losing_streak",
                "drawdown_proxy_r",
                "scored_count",
                "partial_count",
                "unscorable_count",
                "tradeable_with_real_quotes_pct",
                "avg_bid_ask_spread",
            ],
            max_rows=20,
        ),
        "",
        "## Validation Scorecard",
        _write_markdown_table(
            scorecard,
            [
                "baseline",
                "tier",
                "horizon",
                "signal_count",
                "win_rate",
                "avg_net_r",
                "median_net_r",
                "profit_factor",
                "worst_losing_streak",
                "drawdown_proxy_r",
                "scored_count",
                "partial_count",
                "unscorable_count",
                "tradeable_with_real_quotes_pct",
                "avg_bid_ask_spread",
                "blocked_pct",
                "regime",
            ],
            max_rows=40,
        ),
        "",
        "## Sentiment / News Summary",
        _write_markdown_table(news_summary, ["source", "topic", "status", "summary", "urls"], max_rows=10),
        "",
        "## Missed-Mover Lessons",
        _write_markdown_table(
            missed_movers,
            ["signal_date", "horizon", "ticker", "forward_return_pct", "coverage"],
            max_rows=20,
        ),
        "",
        "## Prior Run Changes",
        "- Prior-run diff is metadata-backed when a previous v2 metadata file exists. This first run uses baseline comparison only.",
        "",
        "## Files",
    ]
    for label, path in paths.items():
        lines.append(f"- {label}: {path.resolve()}")
    lines.append("")
    lines.append("This report is a research and planning artifact. It does not execute trades.")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production UW options trend pipeline v2 with leakage-free validation and baseline scorecards."
    )
    parser.add_argument("as_of", nargs="?", default="", help="YYYY-MM-DD. Default: latest usable UW market-data date.")
    parser.add_argument("--root-dir", default=str(DEFAULT_ROOT))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--validate-days", type=int, default=DEFAULT_VALIDATE_DAYS)
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--max-daily-rows", type=int, default=DEFAULT_MAX_DAILY_ROWS)
    parser.add_argument("--min-validation-samples", type=int, default=DEFAULT_MIN_VALIDATION_SAMPLES)
    parser.add_argument(
        "--whale-lookback-days",
        type=int,
        default=1,
        help="Usable UW days of bot-EOD whale/institutional flow to scan for the current report.",
    )
    parser.add_argument("--no-whales", action="store_true", help="Skip bot-EOD whale scan for the current report.")
    return parser.parse_args(list(argv) if argv is not None else None)


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    root = Path(args.root_dir).expanduser().resolve()
    as_of = resolve_as_of(root, args.as_of or None)
    lookback = int(args.lookback)
    horizons = _parse_horizons(args.horizons)
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else root / "out" / "trend_analysis_v2"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{as_of.isoformat()}-L{lookback}"
    cache = DataCache(root)
    days = available_market_days(root)
    live_chain_symbols, live_chain_dirs = load_live_chain_symbols(root, as_of)

    candidates, regime, sector_rotation = build_signal_candidates(
        cache,
        days,
        as_of=as_of,
        lookback=lookback,
        max_rows=int(args.max_daily_rows),
        include_whales=not bool(args.no_whales),
        whale_lookback_days=int(args.whale_lookback_days),
    )
    outcomes, scorecard = run_validation(
        cache=cache,
        root=root,
        as_of=as_of,
        lookback=lookback,
        validate_days=int(args.validate_days),
        horizons=horizons,
        max_daily_rows=min(int(args.max_daily_rows), 50),
    )
    classified, proof = classify_current_candidates(
        candidates,
        scorecard,
        primary_horizon=5 if 5 in horizons else horizons[0],
        min_samples=int(args.min_validation_samples),
    )
    actionable = classified[classified.get("classification", pd.Series(dtype=str)).eq("TRADE")].head(5).copy()
    watchlist = classified[classified.get("classification", pd.Series(dtype=str)).eq("WATCH")].head(15).copy()
    blocked = classified[classified.get("classification", pd.Series(dtype=str)).eq("AVOID")].head(40).copy()
    if watchlist.empty:
        watchlist = classified[classified.get("classification", pd.Series(dtype=str)).ne("TRADE")].head(15).copy()

    news_summary = build_news_summary(root, as_of, classified.head(30).get("ticker", pd.Series(dtype=str)).tolist())
    missed_movers = build_missed_mover_audit(
        cache=cache,
        root=root,
        as_of=as_of,
        lookback=lookback,
        candidates=classified,
        horizons=[h for h in horizons if h in {1, 3, 5}] or [1, 3, 5],
        max_daily_rows=int(args.max_daily_rows),
    )

    paths = {
        "report": out_dir / f"trend-analysis-v2-{suffix}.md",
        "actionable_csv": out_dir / f"trend-analysis-v2-actionable-{suffix}.csv",
        "watchlist_csv": out_dir / f"trend-analysis-v2-watchlist-{suffix}.csv",
        "blocked_csv": out_dir / f"trend-analysis-v2-blocked-{suffix}.csv",
        "candidates_csv": out_dir / f"trend-analysis-v2-candidates-{suffix}.csv",
        "regime_csv": out_dir / f"trend-analysis-v2-market-regime-{suffix}.csv",
        "sector_rotation_csv": out_dir / f"trend-analysis-v2-sector-rotation-{suffix}.csv",
        "sentiment_news_csv": out_dir / f"trend-analysis-v2-sentiment-news-{suffix}.csv",
        "validation_scorecard_csv": out_dir / f"trend-analysis-v2-validation-scorecard-{suffix}.csv",
        "validation_outcomes_csv": out_dir / f"trend-analysis-v2-validation-outcomes-{suffix}.csv",
        "missed_movers_csv": out_dir / f"trend-analysis-v2-missed-movers-{suffix}.csv",
        "metadata_json": out_dir / f"trend-analysis-v2-metadata-{suffix}.json",
    }

    classified.to_csv(paths["candidates_csv"], index=False)
    actionable.to_csv(paths["actionable_csv"], index=False)
    watchlist.to_csv(paths["watchlist_csv"], index=False)
    blocked.to_csv(paths["blocked_csv"], index=False)
    pd.DataFrame([regime]).to_csv(paths["regime_csv"], index=False)
    sector_rotation.to_csv(paths["sector_rotation_csv"], index=False)
    news_summary.to_csv(paths["sentiment_news_csv"], index=False)
    scorecard.to_csv(paths["validation_scorecard_csv"], index=False)
    outcomes.to_csv(paths["validation_outcomes_csv"], index=False)
    missed_movers.to_csv(paths["missed_movers_csv"], index=False)

    metadata = {
        "command": "python3 -m uwos.trend_analysis_v2",
        "root_dir": str(root),
        "out_dir": str(out_dir),
        "as_of": as_of.isoformat(),
        "lookback": lookback,
        "validate_days": int(args.validate_days),
        "horizons": horizons,
        "whale_lookback_days": int(args.whale_lookback_days) if not bool(args.no_whales) else 0,
        "latest_usable_data_date": days[-1].isoformat() if days else "",
        "uses_current_schwab_for_historical_validation": False,
        "current_live_chain_symbol_count": len(live_chain_symbols),
        "current_live_chain_artifact_dirs": live_chain_dirs,
        "entry_quote_source": "local UW hot-chains bid/ask snapshot at signal date",
        "exit_scoring_source": "later local UW option quotes when available; expiry intrinsic value at/after expiry; non-expiry intrinsic proxy is PARTIAL",
        "score_status_policy": {
            "SCORED": "complete option-quote exit or expiry intrinsic value",
            "PARTIAL": "stock/intrinsic proxy because later option quotes were incomplete",
            "UNSCORABLE": "missing quote/underlying data or exit beyond validation cutoff; never counted as a win",
        },
        "news_source": "local browser_text captures when available",
        "proof": proof,
        "counts": {
            "candidates": int(len(classified)),
            "actionable": int(len(actionable)),
            "watchlist": int(len(watchlist)),
            "blocked": int(len(blocked)),
            "validation_outcomes": int(len(outcomes)),
            "missed_movers": int(len(missed_movers)),
        },
        "artifacts": {key: str(path.resolve()) for key, path in paths.items()},
    }
    paths["metadata_json"].write_text(json.dumps(_json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8")
    report = build_report(
        as_of=as_of,
        lookback=lookback,
        regime=regime,
        proof=proof,
        candidates=classified,
        actionable=actionable,
        watchlist=watchlist,
        blocked=blocked,
        scorecard=scorecard,
        sector_rotation=sector_rotation,
        news_summary=news_summary,
        missed_movers=missed_movers,
        paths=paths,
    )
    paths["report"].write_text(report, encoding="utf-8")
    print(f"Wrote: {paths['report']}")
    print(f"Wrote: {paths['validation_scorecard_csv']}")
    print(f"Wrote: {paths['missed_movers_csv']}")
    print(json.dumps(metadata["counts"], indent=2))
    return {"paths": paths, "metadata": metadata, "proof": proof}


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
