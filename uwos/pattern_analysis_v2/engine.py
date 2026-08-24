"""Leakage-safe, price-first Pattern Analysis V2 engine.

The legacy pipeline starts with option-flow families and uses price as a small
modifier. This module starts with every stock-screener row, builds only
backward-looking features, detects both same-day events and forward setups,
and treats an option ticket as a separate implementation problem.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import statistics
import sqlite3
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from . import PIPELINE_VERSION
from ..options_pattern_pipeline_v1.core import (
    find_csv_sources as find_bot_csv_sources,
    find_whale_filtered_sources as find_fallback_whale_sources,
    load_or_build_bot_eod_cache,
    source_fingerprint,
)


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OCC_RE = re.compile(r"^(.+?)(\d{6})([CP])(\d{8})$")
PRIMARY_TARGET_TICKERS = {"IBM", "MRVL", "MRNA", "MU", "PYPL", "TSM"}
# "OIL" is not a ticker in the downloaded stock-screener universe.  Keep the
# liquid crude/energy proxies in the audit set so an oil regime is not silently
# absent from the mover and option-request checks.
ENERGY_PROXY_TICKERS = {"USO", "XLE", "XOP", "XOM", "CVX", "OXY", "VLO", "MPC", "CLF"}
TARGET_TICKERS = PRIMARY_TARGET_TICKERS | ENERGY_PROXY_TICKERS
AUDIT_TICKER_ALIASES = {"OIL": sorted(ENERGY_PROXY_TICKERS)}
MARKET_TICKERS = ("SPY", "QQQ", "IWM")
HORIZONS = (1, 3, 5, 10, 20)
OPTION_HORIZONS = (1, 3, 5, 10, 20)
SHORT_OPTION_MAX_HORIZON = 5
SHORT_OPTION_MIN_DTE = 7
SHORT_OPTION_MAX_DTE = 45
OPTION_MIN_DTE = 25
OPTION_MAX_DTE = 90
LONG_OPTION_MIN_DTE = 91
LONG_OPTION_MAX_DTE = 365
LONG_OPTION_MAX_HORIZON = 20
LONG_OPTION_TARGET_DTE = 120.0
OPTION_MAX_SPREAD_PCT = 0.40
OPTION_MIN_QUOTE_BID = 0.05
MIN_OPTION_UNDERLYING_PRICE = 5.0
MIN_OPTION_AVG30_VOLUME = 250_000.0
OPTION_FEE_PER_CONTRACT = 1.50
MIN_OPTION_OUTCOME_COVERAGE = 0.80
MAX_OPTION_PATTERN_STALENESS_DAYS = 45
# Zero means every liquid signal is eligible for option discovery.  A positive
# value remains available as an explicit performance trade-off, but is never
# the silent default that can hide a valid pattern.
DEFAULT_MAX_OPTION_SCAN_PER_DATE = 0

# Small, predeclared implementation lanes. These are deliberately separate
# from the broad price families: contract filters are part of the hypothesis,
# not an after-the-fact label attached to a winning subset.
VALIDATION_LANE_SPECS = (
    {
        "name": "POST_EVENT_MEAN_REVERSION_COST_GATED_5D",
        "family": "POST_EVENT_MEAN_REVERSION",
        "direction": "bullish",
        "horizon": 5,
        "strategy": "LONG_OPTION",
        "max_spread": 0.10,
        "min_dte": 25,
        "max_dte": 60,
        "max_spread_to_implied_move": 0.25,
    },
    {
        "name": "PULLBACK_CALL_25_45DTE_TIGHT_SPREAD",
        "family": "TREND_PULLBACK_CONTINUATION",
        "direction": "bullish",
        "horizon": 20,
        "strategy": "LONG_OPTION",
        "min_volume_ratio": 1.25,
        "max_spread": 0.12,
        "min_dte": 25,
        "max_dte": 45,
    },
    {
        "name": "BOT_FLOW_DIRECTIONAL_25_45DTE_TIGHT_SPREAD",
        "family": "BOT_EOD_FLOW_PRESSURE",
        "direction": "any",
        "horizon": 20,
        "strategy": "LONG_OPTION",
        "min_bot_bias_abs": 0.20,
        "min_bot_premium": 100_000.0,
        "max_spread": 0.12,
        "min_dte": 25,
        "max_dte": 45,
    },
    {
        "name": "POST_EVENT_CONTINUATION_25_45DTE_TIGHT_SPREAD",
        "family": "POST_EVENT_CONTINUATION",
        "direction": "any",
        "horizon": 5,
        "strategy": "LONG_OPTION",
        "max_spread": 0.12,
        "min_dte": 25,
        "max_dte": 45,
    },
    {
        "name": "POST_EVENT_MEAN_REVERSION_25_45DTE_TIGHT_SPREAD",
        "family": "POST_EVENT_MEAN_REVERSION",
        "direction": "any",
        "horizon": 5,
        "strategy": "LONG_OPTION",
        "max_spread": 0.12,
        "min_dte": 25,
        "max_dte": 45,
    },
)


def as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def clean_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace("/", "-")


def pct_change(current: Optional[float], prior: Optional[float]) -> Optional[float]:
    if current is None or prior is None or prior <= 0:
        return None
    return current / prior - 1.0


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_dirs(base_dir: Path, start: Optional[str] = None, end: Optional[str] = None) -> List[str]:
    rows = []
    for path in base_dir.iterdir() if base_dir.exists() else ():
        if not path.is_dir() or not DATE_RE.fullmatch(path.name):
            continue
        if start and path.name < start:
            continue
        if end and path.name > end:
            continue
        rows.append(path.name)
    return sorted(rows)


@dataclass(frozen=True)
class SourceRef:
    path: Path
    member: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.path}::{self.member}" if self.member else str(self.path)

    @property
    def name(self) -> str:
        return self.member or self.path.name


@dataclass
class PriceRow:
    date: str
    ticker: str
    close: float
    high: Optional[float]
    low: Optional[float]
    prev_close: Optional[float]
    volume: Optional[float]
    avg30_volume: Optional[float]
    sector: str
    call_premium: Optional[float]
    put_premium: Optional[float]
    bullish_premium: Optional[float]
    bearish_premium: Optional[float]
    call_volume: Optional[float]
    put_volume: Optional[float]
    avg30_call_volume: Optional[float]
    avg30_put_volume: Optional[float]
    iv_rank: Optional[float]
    implied_move_perc: Optional[float]
    next_earnings_date: str
    source: str
    total_open_interest: Optional[float] = None
    call_open_interest: Optional[float] = None
    put_open_interest: Optional[float] = None
    call_volume_ask_side: Optional[float] = None
    call_volume_bid_side: Optional[float] = None
    put_volume_ask_side: Optional[float] = None
    put_volume_bid_side: Optional[float] = None
    net_call_premium: Optional[float] = None
    net_put_premium: Optional[float] = None
    prev_call_oi: Optional[float] = None
    prev_put_oi: Optional[float] = None


@dataclass
class PriceSignal:
    date: str
    ticker: str
    direction: str
    family: str
    role: str
    score: float
    reasons: List[str]
    feature: Dict[str, Any]

    @property
    def signal_id(self) -> str:
        raw = f"{self.date}|{self.ticker}|{self.direction}|{self.family}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def find_source(date_dir: Path, prefix: str, signal_date: str) -> List[SourceRef]:
    refs: List[SourceRef] = []
    for path in sorted(date_dir.glob(f"{prefix}*.csv")):
        if signal_date in path.name:
            refs.append(SourceRef(path))
    if refs:
        return refs
    for path in sorted(date_dir.glob(f"{prefix}*.zip")):
        try:
            with zipfile.ZipFile(path) as archive:
                for member in sorted(archive.namelist()):
                    if member.lower().endswith(".csv") and signal_date in Path(member).name:
                        refs.append(SourceRef(path, member))
        except (OSError, zipfile.BadZipFile):
            continue
    return refs


def iter_csv_rows(ref: SourceRef) -> Iterator[Dict[str, str]]:
    try:
        if ref.member:
            with zipfile.ZipFile(ref.path) as archive:
                with archive.open(ref.member) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8-sig", errors="replace", newline="")
                    yield from csv.DictReader(text)
        else:
            with ref.path.open("r", encoding="utf-8-sig", errors="replace", newline="") as text:
                yield from csv.DictReader(text)
    except (OSError, zipfile.BadZipFile, KeyError):
        return


def iter_cache_csv_rows(path: Path) -> Iterator[Dict[str, str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8-sig", errors="replace", newline="") as handle:
            yield from csv.DictReader(handle)
    except (OSError, EOFError, gzip.BadGzipFile):
        return


def fallback_flow_premium(row: Mapping[str, Any]) -> float:
    premium = abs(as_float(row.get("premium")) or 0.0)
    if premium > 0:
        return premium
    price = abs(as_float(row.get("price")) or 0.0)
    size = abs(as_float(row.get("size")) or 0.0)
    return price * size * 100.0


def load_fallback_option_flow_rows(
    refs: Sequence[SourceRef],
    signal_date: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """Aggregate dated fallback option prints when bot-EOD is absent.

    The fallback files use the bot-EOD trade schema, but their filenames are
    not stable: some downloads are named ``whale_trades_filtered.csv`` and
    others carry the date.  This function is intentionally separate from the
    bot cache so the report can preserve which source supplied the flow.
    """

    aggregates: Dict[str, Dict[str, Any]] = {}
    raw_rows = 0
    source_names: Dict[str, Set[str]] = defaultdict(set)
    for ref in refs:
        source_name = "option_trades" if "option-trades" in ref.name.lower() else "whale_filtered"
        for row in iter_csv_rows(ref):
            raw_rows += 1
            if str(row.get("canceled") or "").strip().lower() in {"true", "t", "1", "yes"}:
                continue
            ticker = clean_ticker(row.get("underlying_symbol"))
            if not ticker:
                continue
            premium = fallback_flow_premium(row)
            if premium <= 0:
                continue
            aggregate = aggregates.setdefault(
                ticker,
                {
                    "date": signal_date,
                    "ticker": ticker,
                    "row_count": 0,
                    "flow_call_ask_premium": 0.0,
                    "flow_put_ask_premium": 0.0,
                    "flow_call_bid_premium": 0.0,
                    "flow_put_bid_premium": 0.0,
                    "flow_total_premium": 0.0,
                    "flow_gross_premium": 0.0,
                    "flow_multileg_premium": 0.0,
                    "flow_call_trade_count": 0,
                    "flow_put_trade_count": 0,
                    "flow_source": "",
                },
            )
            aggregate["row_count"] += 1
            aggregate["flow_gross_premium"] += premium
            condition = str(row.get("upstream_condition_detail") or "").strip().lower()
            if condition in {"mlet", "mlat"}:
                aggregate["flow_multileg_premium"] += premium
            else:
                aggregate["flow_total_premium"] += premium
                option_type = str(row.get("option_type") or "").strip().lower()
                side = str(row.get("side") or "").strip().lower()
                if option_type == "call":
                    aggregate["flow_call_trade_count"] += 1
                    if side == "ask":
                        aggregate["flow_call_ask_premium"] += premium
                    elif side == "bid":
                        aggregate["flow_call_bid_premium"] += premium
                elif option_type == "put":
                    aggregate["flow_put_trade_count"] += 1
                    if side == "ask":
                        aggregate["flow_put_ask_premium"] += premium
                    elif side == "bid":
                        aggregate["flow_put_bid_premium"] += premium
            source_names[ticker].add(source_name)
    rows: List[Dict[str, Any]] = []
    for ticker, aggregate in aggregates.items():
        aggregate["flow_source"] = "+".join(sorted(source_names[ticker]))
        rows.append(aggregate)
    rows.sort(key=lambda row: (-float(row.get("flow_total_premium") or 0.0), str(row.get("ticker") or "")))
    return rows, raw_rows


def load_bot_eod_flow_history(
    base_dir: Path,
    dates: Sequence[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """Load bot-EOD flow, with dated fallback prints only when bot-EOD is absent."""

    cache_dir = base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
    flow_by_date: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    source_dates = 0
    cache_hits = 0
    cache_built = 0
    cache_missing = 0
    flow_rows = 0
    quote_rows = 0
    fallback_source_dates = 0
    fallback_raw_rows = 0
    fallback_flow_rows = 0
    fallback_sources: Counter[str] = Counter()
    for signal_date in dates:
        refs = find_bot_csv_sources(base_dir / signal_date, "bot-eod-report", signal_date, exact=True)
        if not refs:
            fallback_refs = find_bot_csv_sources(
                base_dir / signal_date,
                "option-trades",
                signal_date,
                exact=True,
            ) + find_fallback_whale_sources(base_dir / signal_date, signal_date)
            if not fallback_refs:
                continue
            fallback_rows, fallback_raw = load_fallback_option_flow_rows(fallback_refs, signal_date)
            if not fallback_rows:
                continue
            fallback_source_dates += 1
            fallback_raw_rows += fallback_raw
            fallback_flow_rows += len(fallback_rows)
            for row in fallback_rows:
                source = str(row.get("flow_source") or "fallback")
                fallback_sources[source] += 1
                ticker = clean_ticker(row.get("ticker"))
                if not ticker:
                    continue
                flow_by_date[signal_date][ticker] = dict(row)
                flow_rows += 1
            continue
        source_dates += 1
        flow_path = cache_dir / f"bot_eod_flow_by_ticker_{signal_date}.csv"
        compressed_flow_path = Path(f"{flow_path}.gz")
        meta_path = cache_dir / f"bot_eod_cache_{signal_date}.json"
        valid_cache = False
        if meta_path.exists() and (flow_path.exists() or compressed_flow_path.exists()):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                valid_cache = (
                    meta.get("schema_version") == 2
                    and meta.get("signal_date") == signal_date
                    and meta.get("source_fingerprints") == [source_fingerprint(ref) for ref in refs]
                )
                quote_rows += int(meta.get("quote_row_count") or 0)
            except (OSError, ValueError, TypeError):
                valid_cache = False
        if valid_cache:
            cache_hits += 1
            selected_path = flow_path if flow_path.exists() else compressed_flow_path
            rows = iter_cache_csv_rows(selected_path)
        else:
            # Build through the canonical cache writer; this is deliberately
            # not a size-based skip for a present bot-EOD source.
            try:
                built = load_or_build_bot_eod_cache(
                    refs,
                    signal_date,
                    {"bot_eod_cache_dir": str(cache_dir)},
                )
            except (OSError, ValueError, TypeError, KeyError):
                cache_missing += 1
                continue
            cache_built += int(bool(built.get("cache_built")))
            rows = iter(built.get("flow_rows") or [])
            quote_rows += sum(1 for _ in (built.get("quote_rows") or []))
        for row in rows:
            ticker = clean_ticker(row.get("ticker"))
            if not ticker:
                continue
            flow_by_date[signal_date][ticker] = dict(row)
            flow_rows += 1
    return dict(flow_by_date), {
        "bot_eod_source_dates": source_dates,
        "bot_eod_flow_dates_loaded": len(flow_by_date),
        "bot_eod_flow_rows": flow_rows,
        "bot_eod_quote_rows_cached": quote_rows,
        "bot_eod_cache_hits": cache_hits,
        "bot_eod_cache_built": cache_built,
        "bot_eod_cache_missing": cache_missing,
        "bot_eod_cache_dir": str(cache_dir),
        "option_flow_fallback_source_dates": fallback_source_dates,
        "option_flow_fallback_raw_rows": fallback_raw_rows,
        "option_flow_fallback_rows": fallback_flow_rows,
        "option_flow_fallback_sources": dict(sorted(fallback_sources.items())),
    }


def load_price_history(
    base_dir: Path,
    dates: Sequence[str],
) -> Tuple[Dict[str, Dict[str, PriceRow]], Dict[str, Any]]:
    history: Dict[str, Dict[str, PriceRow]] = defaultdict(dict)
    source_labels: Dict[str, List[str]] = {}
    source_rows = 0
    duplicate_rows = 0
    for signal_date in dates:
        refs = find_source(base_dir / signal_date, "stock-screener", signal_date)
        source_labels[signal_date] = [ref.label for ref in refs]
        for ref in refs:
            for row in iter_csv_rows(ref):
                ticker = clean_ticker(row.get("ticker"))
                close = as_float(row.get("close"))
                if not ticker or close is None or close <= 0:
                    continue
                source_rows += 1
                if signal_date in history[ticker]:
                    duplicate_rows += 1
                history[ticker][signal_date] = PriceRow(
                    date=signal_date,
                    ticker=ticker,
                    close=close,
                    high=as_float(row.get("high")),
                    low=as_float(row.get("low")),
                    prev_close=as_float(row.get("prev_close")),
                    volume=as_float(row.get("total_volume")),
                    avg30_volume=as_float(row.get("avg30_volume")),
                    sector=str(row.get("sector") or "").strip(),
                    call_premium=as_float(row.get("call_premium")),
                    put_premium=as_float(row.get("put_premium")),
                    bullish_premium=as_float(row.get("bullish_premium")),
                    bearish_premium=as_float(row.get("bearish_premium")),
                    call_volume=as_float(row.get("call_volume")),
                    put_volume=as_float(row.get("put_volume")),
                    avg30_call_volume=as_float(row.get("avg_30_day_call_volume")),
                    avg30_put_volume=as_float(row.get("avg_30_day_put_volume")),
                    iv_rank=as_float(row.get("iv_rank")),
                    implied_move_perc=as_float(row.get("implied_move_perc")),
                    next_earnings_date=str(row.get("next_earnings_date") or ""),
                    source=ref.label,
                    total_open_interest=as_float(row.get("total_open_interest")),
                    call_open_interest=as_float(row.get("call_open_interest")),
                    put_open_interest=as_float(row.get("put_open_interest")),
                    call_volume_ask_side=as_float(row.get("call_volume_ask_side")),
                    call_volume_bid_side=as_float(row.get("call_volume_bid_side")),
                    put_volume_ask_side=as_float(row.get("put_volume_ask_side")),
                    put_volume_bid_side=as_float(row.get("put_volume_bid_side")),
                    net_call_premium=as_float(row.get("net_call_premium")),
                    net_put_premium=as_float(row.get("net_put_premium")),
                    prev_call_oi=as_float(row.get("prev_call_oi")),
                    prev_put_oi=as_float(row.get("prev_put_oi")),
                )
    return history, {
        "stock_screener_source_rows": source_rows,
        "stock_screener_duplicate_rows": duplicate_rows,
        "stock_screener_dates": len(source_labels),
        "stock_screener_dates_with_source": sum(bool(v) for v in source_labels.values()),
        "stock_screener_dates_with_rows": len({
            row.date
            for rows in history.values()
            for row in rows.values()
        }),
        "source_labels": source_labels,
    }


def previous_rows(series: Sequence[PriceRow], signal_date: str) -> List[PriceRow]:
    return [row for row in series if row.date <= signal_date]


def prior_value(series: Sequence[PriceRow], index: int, periods: int) -> Optional[float]:
    prior_index = index - periods
    if prior_index < 0:
        return None
    return series[prior_index].close


def median(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    return statistics.median(clean) if clean else None


def standard_deviation(values: Sequence[float]) -> Optional[float]:
    if len(values) < 5:
        return None
    return statistics.stdev(values)


def adjusted_close_series(series: Sequence[PriceRow]) -> Tuple[List[float], List[float]]:
    """Build a point-in-time split-adjusted close series from ``prev_close``.

    UW's screener can expose a post-action close while the previous row still
    carries the pre-action raw close.  The current row's ``prev_close`` is the
    only information available at that timestamp to reconcile the bases.  We
    adjust prior observations when that factor appears, without rewriting the
    raw current close used for quotes and execution.
    """

    adjusted: List[float] = []
    factors: List[float] = []
    for index, row in enumerate(series):
        factor = 1.0
        if index > 0:
            previous_raw = as_float(series[index - 1].close)
            previous_close = as_float(row.prev_close)
            if previous_raw and previous_raw > 0 and previous_close and previous_close > 0:
                candidate = previous_close / previous_raw
                if math.isfinite(candidate) and candidate > 0:
                    factor = candidate
                    if not math.isclose(candidate, 1.0, rel_tol=0.0, abs_tol=0.005):
                        adjusted = [value * candidate for value in adjusted]
        adjusted.append(float(row.close))
        factors.append(factor)
    return adjusted, factors


def derive_price_features(
    history: Mapping[str, Mapping[str, PriceRow]],
    dates: Sequence[str],
    bot_flow_by_date: Optional[Mapping[str, Mapping[str, Mapping[str, Any]]]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Create per-date features using only rows on or before that date.

    Bot-EOD flow is same-session EOD information. It is therefore valid for an
    EOD planning signal, but it is never used to manufacture a pre-event claim.
    """

    series_by_ticker = {
        ticker: sorted(rows.values(), key=lambda row: row.date)
        for ticker, rows in history.items()
    }
    by_date: Dict[str, Dict[str, Dict[str, Any]]] = {signal_date: {} for signal_date in dates}
    for ticker, series in series_by_ticker.items():
        adjusted_closes, adjustment_factors = adjusted_close_series(series)
        for index, row in enumerate(series):
            if row.date not in by_date:
                continue
            closes = adjusted_closes[: index + 1]
            daily_returns = [
                adjusted_closes[j] / adjusted_closes[j - 1] - 1.0
                for j in range(max(1, index - 20), index + 1)
                if adjusted_closes[j - 1] > 0
            ]
            adjusted_close = adjusted_closes[index]
            ret = {
                f"return_{n}d": pct_change(
                    adjusted_close,
                    adjusted_closes[index - n] if index >= n else None,
                )
                for n in HORIZONS
            }
            recent20 = closes[-20:]
            recent60 = closes[-60:]
            high20 = max(recent20) if recent20 else None
            low20 = min(recent20) if recent20 else None
            high60 = max(recent60) if recent60 else None
            low60 = min(recent60) if recent60 else None
            ma20 = statistics.fmean(recent20) if len(recent20) >= 5 else None
            ma10 = statistics.fmean(closes[-10:]) if len(closes) >= 5 else None
            vol20 = standard_deviation(daily_returns)
            range_pct = (
                (row.high - row.low) / row.close
                if row.high is not None and row.low is not None and row.close > 0
                else None
            )
            volume_ratio = (
                row.volume / row.avg30_volume
                if row.volume is not None and row.avg30_volume and row.avg30_volume > 0
                else None
            )
            position20 = (
                (adjusted_close - low20) / (high20 - low20)
                if high20 is not None and low20 is not None and high20 > low20
                else None
            )
            position60 = (
                (adjusted_close - low60) / (high60 - low60)
                if high60 is not None and low60 is not None and high60 > low60
                else None
            )
            total_premium = (row.call_premium or 0.0) + (row.put_premium or 0.0)
            call_share = row.call_premium / total_premium if total_premium > 0 and row.call_premium is not None else None
            put_share = row.put_premium / total_premium if total_premium > 0 and row.put_premium is not None else None
            premium_bias = (
                ((row.bullish_premium or 0.0) - (row.bearish_premium or 0.0))
                / ((row.bullish_premium or 0.0) + (row.bearish_premium or 0.0))
                if (row.bullish_premium or 0.0) + (row.bearish_premium or 0.0) > 0
                else (call_share - put_share if call_share is not None and put_share is not None else None)
            )
            side_values = (
                row.call_volume_ask_side,
                row.call_volume_bid_side,
                row.put_volume_ask_side,
                row.put_volume_bid_side,
            )
            side_volume_observed = any(value is not None for value in side_values)
            bullish_side_volume = (
                (row.call_volume_ask_side or 0.0) + (row.put_volume_bid_side or 0.0)
                if side_volume_observed else None
            )
            bearish_side_volume = (
                (row.call_volume_bid_side or 0.0) + (row.put_volume_ask_side or 0.0)
                if side_volume_observed else None
            )
            total_side_volume = (
                bullish_side_volume + bearish_side_volume
                if bullish_side_volume is not None and bearish_side_volume is not None else None
            )
            side_flow_bias = (
                (bullish_side_volume - bearish_side_volume) / total_side_volume
                if total_side_volume and total_side_volume > 0 else None
            )
            total_option_volume = (row.call_volume or 0.0) + (row.put_volume or 0.0)
            side_volume_ratio = (
                total_side_volume / total_option_volume
                if total_side_volume is not None and total_option_volume > 0 else None
            )
            oi_values = (row.call_open_interest, row.put_open_interest)
            oi_observed = any(value is not None for value in oi_values)
            total_oi = (
                (row.call_open_interest or 0.0) + (row.put_open_interest or 0.0)
                if oi_observed else None
            )
            oi_bias = (
                ((row.call_open_interest or 0.0) - (row.put_open_interest or 0.0)) / total_oi
                if total_oi and total_oi > 0 else None
            )
            call_oi_change = pct_change(row.call_open_interest, row.prev_call_oi)
            put_oi_change = pct_change(row.put_open_interest, row.prev_put_oi)
            bot_flow = dict((bot_flow_by_date or {}).get(row.date, {}).get(ticker, {}) or {})
            flow_source = str(bot_flow.get("flow_source") or "bot_eod") if bot_flow else ""
            bot_call_ask = as_float(bot_flow.get("flow_call_ask_premium")) or 0.0
            bot_put_ask = as_float(bot_flow.get("flow_put_ask_premium")) or 0.0
            bot_call_bid = as_float(bot_flow.get("flow_call_bid_premium")) or 0.0
            bot_put_bid = as_float(bot_flow.get("flow_put_bid_premium")) or 0.0
            bot_directional_total = bot_call_ask + bot_put_ask + bot_call_bid + bot_put_bid
            bot_bullish_premium = bot_call_ask + bot_put_bid
            bot_bearish_premium = bot_call_bid + bot_put_ask
            bot_flow_bias = (
                (bot_bullish_premium - bot_bearish_premium) / bot_directional_total
                if bot_directional_total > 0 else None
            )
            by_date[row.date][ticker] = {
                "date": row.date,
                "ticker": ticker,
                "close": row.close,
                "adjusted_close": adjusted_close,
                "price_adjustment_factor": adjustment_factors[index],
                "high": row.high,
                "low": row.low,
                "prev_close": row.prev_close,
                "sector": row.sector,
                "volume": row.volume,
                "avg30_volume": row.avg30_volume,
                "volume_ratio_30d": volume_ratio,
                "range_pct": range_pct,
                "volatility_20d": vol20,
                "ma10": ma10,
                "ma20": ma20,
                "high20": high20,
                "low20": low20,
                "high60": high60,
                "low60": low60,
                "position20": position20,
                "position60": position60,
                "call_premium": row.call_premium,
                "put_premium": row.put_premium,
                "call_share": call_share,
                "put_share": put_share,
                "premium_bias": premium_bias,
                "net_call_premium": row.net_call_premium,
                "net_put_premium": row.net_put_premium,
                "call_volume": row.call_volume,
                "put_volume": row.put_volume,
                "call_volume_ratio_30d": pct_change(row.call_volume, row.avg30_call_volume) + 1.0
                if row.call_volume is not None and row.avg30_call_volume and row.avg30_call_volume > 0
                else None,
                "call_volume_ask_side": row.call_volume_ask_side,
                "call_volume_bid_side": row.call_volume_bid_side,
                "put_volume_ask_side": row.put_volume_ask_side,
                "put_volume_bid_side": row.put_volume_bid_side,
                "uw_side_flow_bias": side_flow_bias,
                "uw_side_volume_ratio": side_volume_ratio,
                "call_open_interest": row.call_open_interest,
                "put_open_interest": row.put_open_interest,
                "total_open_interest": row.total_open_interest or total_oi,
                "call_oi_change": call_oi_change,
                "put_oi_change": put_oi_change,
                "uw_oi_bias": oi_bias,
                "bot_eod_flow_bias": bot_flow_bias,
                "bot_eod_flow_source": flow_source,
                "bot_eod_flow_total_premium": as_float(bot_flow.get("flow_total_premium")),
                "bot_eod_call_ask_premium": bot_call_ask,
                "bot_eod_call_bid_premium": bot_call_bid,
                "bot_eod_put_ask_premium": bot_put_ask,
                "bot_eod_put_bid_premium": bot_put_bid,
                "bot_eod_flow_row_count": as_float(bot_flow.get("row_count")),
                "put_volume_ratio_30d": pct_change(row.put_volume, row.avg30_put_volume) + 1.0
                if row.put_volume is not None and row.avg30_put_volume and row.avg30_put_volume > 0
                else None,
                "iv_rank": row.iv_rank,
                "implied_move_perc": row.implied_move_perc,
                "next_earnings_date": row.next_earnings_date,
                "return_1d": ret["return_1d"],
                "return_3d": ret["return_3d"],
                "return_5d": ret["return_5d"],
                "return_10d": ret["return_10d"],
                "return_20d": ret["return_20d"],
                "history_observations": len(closes),
                "source": row.source,
            }

    market_returns: Dict[str, Dict[str, Optional[float]]] = {}
    for ticker in MARKET_TICKERS:
        series = series_by_ticker.get(ticker, [])
        adjusted_closes, _ = adjusted_close_series(series)
        values: Dict[str, Optional[float]] = {}
        for index, row in enumerate(series):
            values[row.date] = pct_change(
                adjusted_closes[index],
                adjusted_closes[index - 20] if index >= 20 else None,
            )
        market_returns[ticker] = values

    for signal_date, rows in by_date.items():
        market20 = median(market_returns[ticker].get(signal_date) for ticker in MARKET_TICKERS)
        sector_values: Dict[str, List[float]] = defaultdict(list)
        for row in rows.values():
            if row.get("sector") and row.get("return_20d") is not None:
                sector_values[str(row["sector"])].append(float(row["return_20d"]))
        sector_medians = {sector: median(values) for sector, values in sector_values.items()}
        for row in rows.values():
            row["market_return_20d"] = market20
            row["relative_strength_20d"] = (
                row["return_20d"] - market20
                if row.get("return_20d") is not None and market20 is not None
                else None
            )
            sector_median = sector_medians.get(str(row.get("sector") or ""))
            row["sector_relative_strength_20d"] = (
                row["return_20d"] - sector_median
                if row.get("return_20d") is not None and sector_median is not None
                else None
            )
            row["trend_above_ma20"] = bool(
                row.get("ma20") is not None
                and (as_float(row.get("adjusted_close")) or 0.0) > row["ma20"]
            )
            row["earnings_in_10d"] = earnings_in_window(signal_date, row.get("next_earnings_date"), 0, 10)
    return by_date


def earnings_in_window(signal_date: str, earnings_date: Any, minimum: int, maximum: int) -> bool:
    if not DATE_RE.fullmatch(str(earnings_date or "")):
        return False
    try:
        distance = (parse_date(str(earnings_date)) - parse_date(signal_date)).days
    except ValueError:
        return False
    return minimum <= distance <= maximum


def score_feature(row: Mapping[str, Any], terms: Sequence[float]) -> float:
    clean = [abs(float(term)) for term in terms if term is not None and math.isfinite(float(term))]
    return sum(clean) if clean else 0.0


def directional_score(
    direction: str,
    magnitude_terms: Sequence[float],
    directional_terms: Sequence[Optional[float]],
) -> float:
    """Score magnitude plus only sign-aligned relative-strength evidence."""

    magnitude = sum(
        abs(float(term))
        for term in magnitude_terms
        if term is not None and math.isfinite(float(term))
    )
    aligned = 0.0
    for term in directional_terms:
        if term is None or not math.isfinite(float(term)):
            continue
        value = float(term)
        if direction == "bullish" and value > 0:
            aligned += value
        elif direction == "bearish" and value < 0:
            aligned += -value
    return magnitude + aligned


def add_signal(
    signals: List[PriceSignal],
    row: Dict[str, Any],
    direction: str,
    family: str,
    role: str,
    score: float,
    reasons: Sequence[str],
) -> None:
    signals.append(
        PriceSignal(
            date=str(row["date"]),
            ticker=str(row["ticker"]),
            direction=direction,
            family=family,
            role=role,
            score=round(float(score), 6),
            reasons=list(reasons),
            feature=dict(row),
        )
    )


def generate_price_signals(
    features_by_date: Mapping[str, Mapping[str, Dict[str, Any]]],
    dates: Sequence[str],
) -> List[PriceSignal]:
    signals: List[PriceSignal] = []
    for signal_date in dates:
        for row in features_by_date.get(signal_date, {}).values():
            ret1 = row.get("return_1d")
            ret3 = row.get("return_3d")
            ret5 = row.get("return_5d")
            ret20 = row.get("return_20d")
            vol20 = row.get("volatility_20d") or 0.0
            range_pct = row.get("range_pct") or 0.0
            volume_ratio = row.get("volume_ratio_30d") or 0.0
            pos20 = row.get("position20")
            pos60 = row.get("position60")
            rel20 = row.get("relative_strength_20d")
            sector_rel20 = row.get("sector_relative_strength_20d")
            if ret1 is not None and abs(ret1) >= 0.05:
                direction = "bullish" if ret1 > 0 else "bearish"
                add_signal(
                    signals,
                    row,
                    direction,
                    "EVENT_SHOCK",
                    "same_day_event",
                    score_feature(row, (abs(ret1) / max(vol20, 0.01), math.log1p(volume_ratio))),
                    [
                        f"one-day move {ret1:+.2%}",
                        f"volume ratio {volume_ratio:.2f}x" if volume_ratio else "volume ratio unavailable",
                        "same-day event detection; not a pre-event prediction",
                    ],
                )
                event_score = score_feature(
                    row,
                    (abs(ret1) / max(vol20, 0.01), math.log1p(volume_ratio)),
                )
                add_signal(
                    signals,
                    row,
                    direction,
                    "POST_EVENT_CONTINUATION",
                    "post_event_setup",
                    event_score,
                    [
                        f"after a {ret1:+.2%} one-day move",
                        "EOD continuation hypothesis; entered after the event is known",
                    ],
                )
                add_signal(
                    signals,
                    row,
                    "bearish" if direction == "bullish" else "bullish",
                    "POST_EVENT_MEAN_REVERSION",
                    "post_event_setup",
                    event_score,
                    [
                        f"after a {ret1:+.2%} one-day move",
                        "EOD mean-reversion hypothesis; entered after the event is known",
                    ],
                )
            if (
                ret5 is not None
                and ret5 >= max(0.06, 1.5 * vol20)
                and (pos20 is None or pos20 >= 0.75)
                and row.get("trend_above_ma20")
            ):
                add_signal(
                    signals,
                    row,
                    "bullish",
                    "MOMENTUM_BREAKOUT",
                    "forward_setup",
                    directional_score(
                        "bullish",
                        (ret5 / max(vol20, 0.01), math.log1p(volume_ratio)),
                        (rel20, sector_rel20),
                    ),
                    [
                        f"five-day return {ret5:+.2%}",
                        "above 20-day average",
                        "near 20-day high",
                        "relative strength versus market/sector" if rel20 is not None else "price-only momentum",
                    ],
                )
            if (
                ret5 is not None
                and ret5 <= -max(0.06, 1.5 * vol20)
                and (pos20 is None or pos20 <= 0.25)
                and not row.get("trend_above_ma20")
            ):
                add_signal(
                    signals,
                    row,
                    "bearish",
                    "MOMENTUM_BREAKDOWN",
                    "forward_setup",
                    directional_score(
                        "bearish",
                        (abs(ret5) / max(vol20, 0.01), math.log1p(volume_ratio)),
                        (rel20, sector_rel20),
                    ),
                    [
                        f"five-day return {ret5:+.2%}",
                        "below 20-day average",
                        "near 20-day low",
                        "relative weakness versus market/sector" if rel20 is not None else "price-only weakness",
                    ],
                )
            if (
                ret20 is not None
                and ret20 >= 0.10
                and row.get("trend_above_ma20")
                and ret1 is not None
                and ret1 <= 0.01
                and (rel20 is None or rel20 > 0)
            ):
                add_signal(
                    signals,
                    row,
                    "bullish",
                    "TREND_PULLBACK_CONTINUATION",
                    "forward_setup",
                    score_feature(row, (ret20, rel20, sector_rel20, abs(ret1))),
                    [
                        f"20-day trend {ret20:+.2%}",
                        "pullback or pause inside an established uptrend",
                        "market-relative strength remains positive",
                    ],
                )
            if (
                ret20 is not None
                and ret20 <= -0.10
                and not row.get("trend_above_ma20")
                and ret1 is not None
                and ret1 >= -0.01
                and (rel20 is None or rel20 < 0)
            ):
                add_signal(
                    signals,
                    row,
                    "bearish",
                    "DOWNTREND_BOUNCE_CONTINUATION",
                    "forward_setup",
                    score_feature(row, (abs(ret20), abs(rel20 or 0.0), abs(sector_rel20 or 0.0), abs(ret1))),
                    [
                        f"20-day trend {ret20:+.2%}",
                        "bounce or pause inside an established downtrend",
                        "market-relative weakness remains negative",
                    ],
                )
            if (
                ret3 is not None
                and ret3 <= -0.06
                and (pos20 is None or pos20 <= 0.35)
                and volume_ratio >= 0.80
            ):
                strict_oversold = ret3 <= -0.08 and (pos20 is None or pos20 <= 0.20) and volume_ratio >= 1.0
                add_signal(
                    signals,
                    row,
                    "bullish",
                    "OVERSOLD_REBOUND_STRICT" if strict_oversold else "OVERSOLD_REBOUND",
                    "forward_setup",
                    score_feature(row, (abs(ret3), math.log1p(volume_ratio), 1.0 - (pos20 or 0.0))),
                    [
                        f"three-day drawdown {ret3:+.2%}",
                        "near the lower end of the recent range",
                        "counter-trend rebound candidate; higher failure risk",
                    ],
                )
            if (
                ret3 is not None
                and ret3 >= 0.08
                and (pos20 is None or pos20 >= 0.80)
                and volume_ratio >= 1.0
            ):
                add_signal(
                    signals,
                    row,
                    "bearish",
                    "OVERBOUGHT_REVERSAL",
                    "forward_setup",
                    score_feature(row, (abs(ret3), math.log1p(volume_ratio), pos20 or 0.0)),
                    [
                        f"three-day advance {ret3:+.2%}",
                        "near the upper end of the recent range",
                        "counter-trend reversal candidate; timing risk is high",
                    ],
                )
            if (
                row.get("earnings_in_10d")
                and (as_float(row.get("iv_rank")) or 0.0) >= 40.0
                and volume_ratio >= 0.80
            ):
                add_signal(
                    signals,
                    row,
                    "neutral",
                    "EARNINGS_VOLATILITY_EVENT",
                    "forward_setup",
                    score_feature(
                        row,
                        (
                            (as_float(row.get("iv_rank")) or 0.0) / 100.0,
                            volume_ratio,
                            as_float(row.get("implied_move_perc")) or 0.0,
                        ),
                    ),
                    [
                        "earnings inside the next ten calendar days",
                        f"IV rank {as_float(row.get('iv_rank')):.1f}",
                        "direction intentionally neutral; evaluate a defined-risk volatility structure",
                    ],
                )
            option_bias = row.get("premium_bias")
            if option_bias is not None and abs(option_bias) >= 0.20 and volume_ratio >= 1.25:
                direction = "bullish" if option_bias > 0 else "bearish"
                add_signal(
                    signals,
                    row,
                    direction,
                    "UW_FLOW_CONFIRMED_PRICE_SETUP",
                    "confirmation",
                    score_feature(row, (abs(option_bias), math.log1p(volume_ratio), abs(ret5 or 0.0))),
                    [
                        f"UW premium bias {option_bias:+.2f}",
                        f"volume ratio {volume_ratio:.2f}x",
                        "flow is confirmation; price event remains independently visible",
                    ],
                )
            side_bias = row.get("uw_side_flow_bias")
            side_volume_ratio = row.get("uw_side_volume_ratio")
            if (
                side_bias is not None
                and abs(side_bias) >= 0.25
                and (side_volume_ratio is None or side_volume_ratio >= 0.80)
                and volume_ratio >= 0.80
            ):
                direction = "bullish" if side_bias > 0 else "bearish"
                add_signal(
                    signals,
                    row,
                    direction,
                    "UW_AGGRESSOR_FLOW_SETUP",
                    "confirmation",
                    score_feature(row, (abs(side_bias), math.log1p(volume_ratio), abs(ret5 or 0.0))),
                    [
                        f"UW aggressor-side flow bias {side_bias:+.2f}",
                        f"underlying volume ratio {volume_ratio:.2f}x",
                        "call-at-ask/put-at-bid versus call-at-bid/put-at-ask pressure",
                    ],
                )
            bot_bias = row.get("bot_eod_flow_bias")
            bot_total_premium = row.get("bot_eod_flow_total_premium") or 0.0
            if bot_bias is not None and abs(bot_bias) >= 0.20 and bot_total_premium >= 100_000.0:
                direction = "bullish" if bot_bias > 0 else "bearish"
                flow_source = str(row.get("bot_eod_flow_source") or "bot_eod")
                flow_family = (
                    "BOT_EOD_FLOW_PRESSURE"
                    if flow_source == "bot_eod"
                    else "UW_FALLBACK_FLOW_PRESSURE"
                )
                add_signal(
                    signals,
                    row,
                    direction,
                    flow_family,
                    "confirmation",
                    score_feature(row, (abs(bot_bias), math.log10(max(bot_total_premium, 1.0)), abs(ret5 or 0.0))),
                    [
                        f"bot-EOD directional premium bias {bot_bias:+.2f}",
                        f"bot-EOD directional premium ${bot_total_premium:,.0f}",
                        (
                            "primary UW bot-EOD flow source; EOD confirmation, not pre-event information"
                            if flow_source == "bot_eod"
                            else f"dated fallback option-flow source ({flow_source}); EOD confirmation, not pre-event information"
                        ),
                    ],
                )
    signals.sort(key=lambda signal: (signal.date, signal.score, signal.ticker), reverse=True)
    return signals


def session_target(dates: Sequence[str], signal_date: str, horizon: int) -> Optional[str]:
    try:
        index = list(dates).index(signal_date)
    except ValueError:
        return None
    target_index = index + horizon
    return dates[target_index] if target_index < len(dates) else None


def realized_stock_return(
    features_by_date: Mapping[str, Mapping[str, Dict[str, Any]]],
    dates: Sequence[str],
    ticker: str,
    signal_date: str,
    target_date: str,
) -> Optional[float]:
    """Measure a forward return on one price basis without leaking features.

    The entry row remains point-in-time.  Any later split factors are applied
    only while measuring the realized outcome, so a split cannot masquerade as
    a predictive return while the signal was being formed.
    """

    try:
        signal_index = list(dates).index(signal_date)
        target_index = list(dates).index(target_date)
    except ValueError:
        return None
    if target_index <= signal_index:
        return None
    entry_row = features_by_date.get(signal_date, {}).get(ticker, {})
    target_row = features_by_date.get(target_date, {}).get(ticker, {})
    entry = as_float(entry_row.get("close"))
    target = as_float(target_row.get("close"))
    if entry is None or target is None or entry <= 0 or target <= 0:
        return None
    basis_factor = 1.0
    for index in range(signal_index + 1, target_index + 1):
        factor = as_float(
            features_by_date.get(dates[index], {}).get(ticker, {}).get("price_adjustment_factor")
        )
        if factor is not None and factor > 0 and math.isfinite(factor):
            basis_factor *= factor
    adjusted_entry = entry * basis_factor
    if adjusted_entry <= 0:
        return None
    return target / adjusted_entry - 1.0


def stock_outcome(
    features_by_date: Mapping[str, Mapping[str, Dict[str, Any]]],
    dates: Sequence[str],
    signal: PriceSignal,
    horizon: int,
) -> Optional[float]:
    target_date = session_target(dates, signal.date, horizon)
    if target_date is None:
        return None
    raw = realized_stock_return(features_by_date, dates, signal.ticker, signal.date, target_date)
    if raw is None:
        return None
    if signal.direction == "neutral":
        return abs(raw)
    return raw if signal.direction == "bullish" else -raw


def signal_rows(
    signals: Sequence[PriceSignal],
    features_by_date: Mapping[str, Mapping[str, Dict[str, Any]]],
    dates: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for signal in signals:
        row = {
            "signal_id": signal.signal_id,
            "signal_date": signal.date,
            "ticker": signal.ticker,
            "direction": signal.direction,
            "pattern_family": signal.family,
            "signal_role": signal.role,
            "score": signal.score,
            "reasons": "; ".join(signal.reasons),
            "close": signal.feature.get("close"),
            "adjusted_close": signal.feature.get("adjusted_close"),
            "price_adjustment_factor": signal.feature.get("price_adjustment_factor"),
            "return_1d": signal.feature.get("return_1d"),
            "return_3d": signal.feature.get("return_3d"),
            "return_5d": signal.feature.get("return_5d"),
            "return_20d": signal.feature.get("return_20d"),
            "volatility_20d": signal.feature.get("volatility_20d"),
            "volume_ratio_30d": signal.feature.get("volume_ratio_30d"),
            "relative_strength_20d": signal.feature.get("relative_strength_20d"),
            "sector_relative_strength_20d": signal.feature.get("sector_relative_strength_20d"),
            "premium_bias": signal.feature.get("premium_bias"),
            "uw_side_flow_bias": signal.feature.get("uw_side_flow_bias"),
            "uw_side_volume_ratio": signal.feature.get("uw_side_volume_ratio"),
            "uw_oi_bias": signal.feature.get("uw_oi_bias"),
            "call_oi_change": signal.feature.get("call_oi_change"),
            "put_oi_change": signal.feature.get("put_oi_change"),
            "bot_eod_flow_bias": signal.feature.get("bot_eod_flow_bias"),
            "bot_eod_flow_source": signal.feature.get("bot_eod_flow_source"),
            "bot_eod_flow_total_premium": signal.feature.get("bot_eod_flow_total_premium"),
            "bot_eod_flow_row_count": signal.feature.get("bot_eod_flow_row_count"),
            "avg30_volume": signal.feature.get("avg30_volume"),
            "sector": signal.feature.get("sector"),
            "earnings_in_10d": signal.feature.get("earnings_in_10d"),
        }
        for horizon in HORIZONS:
            row[f"stock_return_{horizon}d"] = stock_outcome(features_by_date, dates, signal, horizon)
        rows.append(row)
    return rows


def price_outcome_rows(price_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize wide signal outcomes for grouped validation statistics."""

    rows: List[Dict[str, Any]] = []
    for source in price_rows:
        for horizon in HORIZONS:
            value = as_float(source.get(f"stock_return_{horizon}d"))
            if value is None:
                continue
            row = dict(source)
            row["horizon"] = horizon
            row["stock_return"] = value
            rows.append(row)
    return rows


def parse_occ(symbol: str) -> Optional[Dict[str, Any]]:
    normalized = str(symbol or "").strip().upper().replace(" ", "")
    match = OCC_RE.match(normalized)
    if not match:
        return None
    ticker, expiry, option_type, strike_raw = match.groups()
    try:
        expiry_date = datetime.strptime(expiry, "%y%m%d").date()
        strike = int(strike_raw) / 1000.0
    except ValueError:
        return None
    return {
        "option_symbol": normalized,
        "ticker": clean_ticker(ticker),
        "expiry": expiry_date.isoformat(),
        "option_type": option_type,
        "strike": strike,
    }


def trading_dte(signal_date: str, expiry: str) -> int:
    try:
        return (parse_date(expiry) - parse_date(signal_date)).days
    except ValueError:
        return -1


def option_direction(option_type: str) -> str:
    return "bullish" if option_type == "C" else "bearish"


def option_quote_from_row(row: Mapping[str, str], signal_date: str) -> Optional[Dict[str, Any]]:
    parsed = parse_occ(row.get("option_symbol") or row.get("contract_symbol") or "")
    if not parsed or parsed["ticker"].startswith("^"):
        return None
    bid = as_float(row.get("bid"))
    ask = as_float(row.get("ask"))
    if bid is None or ask is None or bid < 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    spread_pct = (ask - bid) / mid if mid > 0 else None
    return {
        **parsed,
        "date": signal_date,
        "dte": trading_dte(signal_date, parsed["expiry"]),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_pct": spread_pct,
        "volume": as_float(row.get("volume")) or 0.0,
        "open_interest": as_float(row.get("open_interest")) or 0.0,
        "premium": as_float(row.get("premium")) or 0.0,
        "iv": as_float(row.get("iv")) or as_float(row.get("implied_volatility")),
        "underlying_close": as_float(row.get("close")) or as_float(row.get("underlying_price")),
        "source": "hot_chains",
        "source_date": signal_date,
        "quote_provenance": "same_session_hot_chain",
    }


def option_quote_score(
    quote: Mapping[str, Any],
    spot: float,
    horizon: int = 5,
    target_dte: Optional[float] = None,
) -> float:
    moneyness_distance = abs(float(quote["strike"]) / spot - 1.0) if spot > 0 else 9.0
    spread = float(quote.get("spread_pct") or 9.0)
    liquidity = math.log1p(float(quote.get("volume") or 0.0) + float(quote.get("open_interest") or 0.0))
    target = target_dte if target_dte is not None else (14.0 if horizon <= 3 else 35.0)
    dte_penalty = abs(float(quote.get("dte") or 0.0) - target) / 100.0
    return moneyness_distance * 10.0 + spread * 2.0 + dte_penalty - liquidity * 0.01


def option_quote_is_horizon_safe(
    quote: Mapping[str, Any],
    signal_date: str,
    dates: Sequence[str],
    horizon: int,
) -> bool:
    """Keep a selected contract alive through the horizon it represents."""

    expiry = str(quote.get("expiry") or "")
    target = session_target(dates, signal_date, horizon)
    if target and DATE_RE.fullmatch(expiry):
        return target <= expiry
    # A current signal has no observable target yet. Keep enough calendar
    # buffer for the intended session horizon instead of selecting a contract
    # that will expire before the trade's declared holding period.
    minimum_dte = 35 if horizon >= 10 else 10
    return int(quote.get("dte") or -1) >= minimum_dte


def quote_cache_path(base_dir: Path, kind: str, payload: Any) -> Path:
    cache_dir = base_dir / "out" / "pattern_analysis_v2" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "pipeline_version": PIPELINE_VERSION,
        "payload": payload,
    }
    fingerprint = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:20]
    return cache_dir / f"{kind}_{fingerprint}.json"


def quote_history_cache_path(base_dir: Path, kind: str, payload: Any) -> Path:
    """Use a streaming cache format for histories that can exceed a GB."""

    path = quote_cache_path(base_dir, kind, payload)
    return path.with_name(f"{path.stem}.jsonl.gz")


def load_quote_history_jsonl(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    history: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if not isinstance(row, dict):
                    continue
                symbol = str(row.get("symbol") or "")
                quote_date = str(row.get("quote_date") or "")
                quote = row.get("quote")
                if symbol and quote_date and isinstance(quote, dict):
                    history[symbol][quote_date] = quote
    except (OSError, EOFError, gzip.BadGzipFile):
        return {}
    return dict(history)


def write_quote_history_jsonl(path: Path, history: Mapping[str, Mapping[str, Dict[str, Any]]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            for symbol, rows in sorted(history.items()):
                for quote_date, quote in sorted(rows.items()):
                    handle.write(
                        json.dumps(
                            {
                                "symbol": symbol,
                                "quote_date": quote_date,
                                "quote": quote,
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def source_index_path(base_dir: Path, kind: str, ref: SourceRef) -> Path:
    payload = {
        "kind": kind,
        "schema_version": 2,
        "source": source_fingerprint(ref),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()[:24]
    path = base_dir / "out" / "pattern_analysis_v2" / "cache" / "source_index" / kind / f"{fingerprint}.jsonl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def source_sqlite_path(
    base_dir: Path,
    kind: str,
    ref: SourceRef,
    symbols: Optional[Set[str]] = None,
) -> Path:
    index_path = source_index_path(base_dir, kind, ref)
    symbol_fingerprint = hashlib.sha256(
        "\n".join(sorted(symbols or set())).encode("utf-8")
    ).hexdigest()[:16]
    return index_path.with_name(
        index_path.name.replace(".jsonl.gz", f".{symbol_fingerprint}.sqlite3")
    )


def build_chain_sqlite_index(
    path: Path,
    index_path: Path,
    symbols: Optional[Set[str]] = None,
) -> int:
    """Build a selective symbol/date index from the provenance-aware JSONL cache."""

    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    rows_written = 0
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE quotes (symbol TEXT NOT NULL, quote_date TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.execute("CREATE INDEX quotes_symbol_date ON quotes(symbol, quote_date)")
        batch: List[Tuple[str, str, str]] = []
        with gzip.open(index_path, "rt", encoding="utf-8") as source:
            for line in source:
                marker = line.find('"symbol":"')
                if marker < 0:
                    continue
                symbol_start = marker + len('"symbol":"')
                symbol_end = line.find('"', symbol_start)
                symbol = line[symbol_start:symbol_end] if symbol_end > symbol_start else ""
                if symbols is not None and symbol not in symbols:
                    continue
                try:
                    indexed = json.loads(line)
                except (TypeError, ValueError):
                    continue
                quote_date = str(indexed.get("quote_date") or "")
                if not symbol or not DATE_RE.fullmatch(quote_date):
                    continue
                batch.append((symbol, quote_date, json.dumps(indexed, separators=(",", ":"))))
                if len(batch) >= 5000:
                    connection.executemany("INSERT INTO quotes VALUES (?, ?, ?)", batch)
                    connection.commit()
                    rows_written += len(batch)
                    batch.clear()
        if batch:
            connection.executemany("INSERT INTO quotes VALUES (?, ?, ?)", batch)
            connection.commit()
            rows_written += len(batch)
        connection.execute("ANALYZE")
        connection.commit()
    finally:
        connection.close()
    temporary.replace(path)
    return rows_written


def build_chain_source_index(path: Path, ref: SourceRef, source_date: str) -> Tuple[int, int]:
    """Persist valid chain quotes once so future runs avoid raw archive scans."""

    temporary = path.with_name(path.name + ".tmp")
    raw_rows = 0
    valid_rows = 0
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            for raw in iter_csv_rows(ref):
                raw_rows += 1
                symbol = str(raw.get("option_symbol") or "").strip().upper().replace(" ", "")
                quote_date = str(raw.get("last_date") or "")[:10]
                bid = as_float(raw.get("last_bid"))
                ask = as_float(raw.get("last_ask"))
                parsed = parse_occ(symbol)
                if (
                    not parsed
                    or not DATE_RE.fullmatch(quote_date)
                    or bid is None
                    or ask is None
                    or ask <= 0
                    or ask < bid
                ):
                    continue
                handle.write(
                    json.dumps(
                        {
                            "symbol": symbol,
                            "ticker": parsed["ticker"],
                            "expiry": parsed["expiry"],
                            "option_type": parsed["option_type"],
                            "strike": parsed["strike"],
                            "quote_date": quote_date,
                            "source_date": source_date,
                            "bid": bid,
                            "ask": ask,
                            "prev_vol": as_float(raw.get("prev_vol")) or 0.0,
                            "last_oi": as_float(raw.get("last_oi")) or 0.0,
                            "prev_total_premium": as_float(raw.get("prev_total_premium")) or 0.0,
                            "stock_price": as_float(raw.get("stock_price")),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                valid_rows += 1
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return raw_rows, valid_rows


def iter_chain_source_index(path: Path) -> Iterator[Dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, dict):
                    yield value
    except OSError:
        return


def load_entry_option_quotes(
    base_dir: Path,
    dates: Sequence[str],
    requests: Mapping[str, Set[Tuple[str, str]]],
    features_by_date: Mapping[str, Mapping[str, Dict[str, Any]]],
) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], Dict[str, Any]]:
    cache_path = quote_cache_path(
        base_dir,
        "entry_quotes",
        {
            "pipeline_version": PIPELINE_VERSION,
            "selection_schema": 14,
            "dates": list(dates),
            "requests": {key: sorted(list(value)) for key, value in sorted(requests.items())},
        },
    )
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            chosen = {
                tuple(item["key"]): dict(item["quote"])
                for item in cached.get("rows", [])
            }
            return chosen, {
                "hot_chain_raw_rows_scanned": 0,
                "hot_chain_eligible_quote_rows": 0,
                "entry_option_request_count": sum(len(values) for values in requests.values()),
                "entry_option_quote_count": len(chosen),
                "entry_option_cache_hit": True,
            }
        except (OSError, ValueError, TypeError, KeyError):
            pass
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    source_count = 0
    raw_rows = 0
    for signal_date in dates:
        requested = requests.get(signal_date, set())
        if not requested:
            continue
        tickers = {ticker for ticker, _ in requested}
        for ref in find_source(base_dir / signal_date, "hot-chains", signal_date):
            for raw in iter_csv_rows(ref):
                raw_rows += 1
                quote = option_quote_from_row(raw, signal_date)
                if not quote or quote["ticker"] not in tickers:
                    continue
                direction = option_direction(str(quote["option_type"]))
                request_directions: List[str] = []
                if (quote["ticker"], "neutral") in requested:
                    request_directions.append("neutral")
                for requested_direction in ("bullish", "bearish"):
                    if (quote["ticker"], requested_direction) in requested:
                        request_directions.append(requested_direction)
                if not request_directions:
                    continue
                if not SHORT_OPTION_MIN_DTE <= int(quote["dte"]) <= LONG_OPTION_MAX_DTE:
                    continue
                if quote.get("spread_pct") is None or quote["spread_pct"] > OPTION_MAX_SPREAD_PCT:
                    continue
                if float(quote.get("bid") or 0.0) < OPTION_MIN_QUOTE_BID:
                    continue
                spot = float(features_by_date.get(signal_date, {}).get(quote["ticker"], {}).get("close") or 0.0)
                if spot <= 0:
                    continue
                quote["underlying_close"] = spot
                for request_direction in request_directions:
                    grouped[(signal_date, quote["ticker"], request_direction)].append(quote)
                    source_count += 1
    chosen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, quotes in grouped.items():
        _, ticker, direction = key
        spot = float(features_by_date[key[0]][ticker]["close"])
        if direction == "neutral":
            by_expiry: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: {"C": [], "P": []})
            for quote in quotes:
                if (
                    not OPTION_MIN_DTE <= int(quote.get("dte") or -1) <= OPTION_MAX_DTE
                    or not option_quote_is_horizon_safe(quote, key[0], dates, max(OPTION_HORIZONS))
                ):
                    continue
                by_expiry[str(quote["expiry"])][str(quote["option_type"])].append(quote)
            pairs: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
            for expiry, sides in by_expiry.items():
                for call in sides["C"]:
                    for put in sides["P"]:
                        distance = abs(float(call["strike"]) / spot - 1.0) + abs(float(put["strike"]) / spot - 1.0)
                        pairs.append((distance, call, put))
            if pairs:
                _, call_quote, put_quote = min(pairs, key=lambda item: item[0])
                entry_debit = float(call_quote["ask"]) + float(put_quote["ask"])
                variants: List[Dict[str, Any]] = []
                if entry_debit > 0:
                    variants.append(
                        {
                            **call_quote,
                            "strategy": "LONG_STRADDLE",
                            "put_option_symbol": put_quote["option_symbol"],
                            "put_option_type": put_quote["option_type"],
                            "put_strike": put_quote["strike"],
                            "put_expiry": put_quote["expiry"],
                            "put_bid": put_quote["bid"],
                            "put_ask": put_quote["ask"],
                            "put_spread_pct": put_quote["spread_pct"],
                            "entry_debit": entry_debit,
                        }
                    )
                sides = by_expiry.get(str(call_quote["expiry"]), {"C": [], "P": []})
                short_calls = [quote for quote in sides["C"] if float(quote["strike"]) > spot]
                short_puts = [quote for quote in sides["P"] if float(quote["strike"]) < spot]
                if short_calls and short_puts:
                    short_call = min(
                        short_calls,
                        key=lambda quote: abs(float(quote["strike"]) / spot - 1.08),
                    )
                    short_put = min(
                        short_puts,
                        key=lambda quote: abs(float(quote["strike"]) / spot - 0.92),
                    )
                    long_calls = [
                        quote for quote in sides["C"]
                        if float(quote["strike"]) > float(short_call["strike"])
                    ]
                    long_puts = [
                        quote for quote in sides["P"]
                        if float(quote["strike"]) < float(short_put["strike"])
                    ]
                    if long_calls and long_puts:
                        long_call = min(
                            long_calls,
                            key=lambda quote: abs(float(quote["strike"]) / spot - 1.16),
                        )
                        long_put = min(
                            long_puts,
                            key=lambda quote: abs(float(quote["strike"]) / spot - 0.84),
                        )
                        entry_credit = (
                            float(short_call["bid"])
                            + float(short_put["bid"])
                            - float(long_call["ask"])
                            - float(long_put["ask"])
                        )
                        call_width = abs(float(long_call["strike"]) - float(short_call["strike"]))
                        put_width = abs(float(short_put["strike"]) - float(long_put["strike"]))
                        if entry_credit > 0 and max(call_width, put_width) > entry_credit:
                            variants.append(
                                {
                                    **short_call,
                                    "strategy": "IRON_CONDOR",
                                    "iron_short_call_option_symbol": short_call["option_symbol"],
                                    "iron_long_call_option_symbol": long_call["option_symbol"],
                                    "iron_short_put_option_symbol": short_put["option_symbol"],
                                    "iron_long_put_option_symbol": long_put["option_symbol"],
                                    "iron_short_call_strike": short_call["strike"],
                                    "iron_long_call_strike": long_call["strike"],
                                    "iron_short_put_strike": short_put["strike"],
                                    "iron_long_put_strike": long_put["strike"],
                                    "iron_short_call_bid": short_call["bid"],
                                    "iron_short_call_ask": short_call["ask"],
                                    "iron_long_call_bid": long_call["bid"],
                                    "iron_long_call_ask": long_call["ask"],
                                    "iron_short_put_bid": short_put["bid"],
                                    "iron_short_put_ask": short_put["ask"],
                                    "iron_long_put_bid": long_put["bid"],
                                    "iron_long_put_ask": long_put["ask"],
                                    "iron_expiry": short_call["expiry"],
                                    "iron_call_width": call_width,
                                    "iron_put_width": put_width,
                                    "entry_credit": entry_credit,
                                }
                            )
                if variants:
                    chosen_quote = dict(variants[0])
                    chosen_quote["variants"] = variants
                    chosen[key] = chosen_quote
            continue
        long_type = "C" if direction == "bullish" else "P"
        standard_long_candidates = [
            quote
            for quote in quotes
            if str(quote.get("option_type")) == long_type
            and OPTION_MIN_DTE <= int(quote.get("dte") or -1) <= OPTION_MAX_DTE
            and option_quote_is_horizon_safe(quote, key[0], dates, max(OPTION_HORIZONS))
        ]
        long_dte_candidates = [
            quote
            for quote in quotes
            if str(quote.get("option_type")) == long_type
            and LONG_OPTION_MIN_DTE <= int(quote.get("dte") or -1) <= LONG_OPTION_MAX_DTE
            and option_quote_is_horizon_safe(quote, key[0], dates, LONG_OPTION_MAX_HORIZON)
        ]
        short_long_candidates = [
            quote
            for quote in quotes
            if str(quote.get("option_type")) == long_type
            and SHORT_OPTION_MIN_DTE <= int(quote.get("dte") or -1) <= SHORT_OPTION_MAX_DTE
            and option_quote_is_horizon_safe(quote, key[0], dates, SHORT_OPTION_MAX_HORIZON)
        ]
        if not standard_long_candidates and not long_dte_candidates and not short_long_candidates:
            continue
        long_dte_quote = min(
            long_dte_candidates,
            key=lambda quote: option_quote_score(
                quote,
                spot,
                horizon=LONG_OPTION_MAX_HORIZON,
                target_dte=LONG_OPTION_TARGET_DTE,
            ),
        ) if long_dte_candidates else None
        long_quote = min(
            standard_long_candidates,
            key=lambda quote: option_quote_score(quote, spot),
        ) if standard_long_candidates else long_dte_quote or min(
            short_long_candidates,
            key=lambda quote: option_quote_score(quote, spot, horizon=3),
        )
        variants: List[Dict[str, Any]] = []
        if standard_long_candidates:
            variants.append({**long_quote, "strategy": "LONG_OPTION"})
        if long_dte_quote:
            variants.append(
                {
                    **long_dte_quote,
                    "strategy": "LONG_OPTION_LONG_DTE",
                    "max_horizon": LONG_OPTION_MAX_HORIZON,
                }
            )
        short_long_quote = min(
            short_long_candidates,
            key=lambda quote: option_quote_score(quote, spot, horizon=3),
        ) if short_long_candidates else None
        if short_long_quote and (
            not standard_long_candidates
            or short_long_quote.get("option_symbol") != long_quote.get("option_symbol")
        ):
            variants.append(
                {
                    **short_long_quote,
                    "strategy": "LONG_OPTION_SHORT_DTE",
                    "max_horizon": SHORT_OPTION_MAX_HORIZON,
                }
            )
        if not variants:
            variants.append(
                {
                    **short_long_quote,
                    "strategy": "LONG_OPTION_SHORT_DTE",
                    "max_horizon": SHORT_OPTION_MAX_HORIZON,
                }
            )
        same_expiry = [
            quote
            for quote in standard_long_candidates
            if quote.get("expiry") == long_quote.get("expiry")
            and float(quote.get("bid") or 0.0) >= OPTION_MIN_QUOTE_BID
        ]
        if direction == "bullish":
            short_candidates = [
                quote for quote in same_expiry if float(quote["strike"]) > float(long_quote["strike"])
            ]
        else:
            short_candidates = [
                quote for quote in same_expiry if float(quote["strike"]) < float(long_quote["strike"])
            ]
        if short_candidates:
            short_quote = min(
                short_candidates,
                key=lambda quote: abs(float(quote["strike"]) / spot - (1.08 if direction == "bullish" else 0.92)),
            )
            debit = float(long_quote["ask"]) - float(short_quote["bid"])
            width = abs(float(short_quote["strike"]) - float(long_quote["strike"]))
            if debit > 0 and width > 0:
                variants.append(
                    {
                        **long_quote,
                        "strategy": "DEBIT_VERTICAL",
                        "short_option_symbol": short_quote["option_symbol"],
                        "short_option_type": short_quote["option_type"],
                        "short_strike": short_quote["strike"],
                        "short_expiry": short_quote["expiry"],
                        "short_bid": short_quote["bid"],
                        "short_ask": short_quote["ask"],
                        "short_spread_pct": short_quote["spread_pct"],
                        "vertical_width": width,
                        "entry_debit": debit,
                    }
                )
        short_debit_quote = short_long_quote
        short_debit_same_expiry = (
            [
                quote
                for quote in short_long_candidates
                if quote.get("expiry") == short_debit_quote.get("expiry")
                and float(quote.get("bid") or 0.0) >= OPTION_MIN_QUOTE_BID
            ]
            if short_debit_quote
            else []
        )
        if short_debit_quote and direction == "bullish":
            short_debit_candidates = [
                quote
                for quote in short_debit_same_expiry
                if float(quote["strike"]) > float(short_debit_quote["strike"])
            ]
        elif short_debit_quote:
            short_debit_candidates = [
                quote
                for quote in short_debit_same_expiry
                if float(quote["strike"]) < float(short_debit_quote["strike"])
            ]
        else:
            short_debit_candidates = []
        if short_debit_quote and short_debit_candidates:
            short_debit_short = min(
                short_debit_candidates,
                key=lambda quote: abs(
                    float(quote["strike"]) / spot
                    - (1.08 if direction == "bullish" else 0.92)
                ),
            )
            short_debit = float(short_debit_quote["ask"]) - float(short_debit_short["bid"])
            short_debit_width = abs(
                float(short_debit_short["strike"])
                - float(short_debit_quote["strike"])
            )
            if short_debit > 0 and short_debit_width > 0:
                variants.append(
                    {
                        **short_debit_quote,
                        "strategy": "DEBIT_VERTICAL_SHORT_DTE",
                        "max_horizon": SHORT_OPTION_MAX_HORIZON,
                        "short_option_symbol": short_debit_short["option_symbol"],
                        "short_option_type": short_debit_short["option_type"],
                        "short_strike": short_debit_short["strike"],
                        "short_expiry": short_debit_short["expiry"],
                        "short_bid": short_debit_short["bid"],
                        "short_ask": short_debit_short["ask"],
                        "short_spread_pct": short_debit_short["spread_pct"],
                        "vertical_width": short_debit_width,
                        "entry_debit": short_debit,
                    }
                )
        long_debit_quote = long_dte_quote
        long_debit_same_expiry = (
            [
                quote
                for quote in long_dte_candidates
                if quote.get("expiry") == long_debit_quote.get("expiry")
                and float(quote.get("bid") or 0.0) >= OPTION_MIN_QUOTE_BID
            ]
            if long_debit_quote
            else []
        )
        if long_debit_quote and direction == "bullish":
            long_debit_candidates = [
                quote
                for quote in long_debit_same_expiry
                if float(quote["strike"]) > float(long_debit_quote["strike"])
            ]
        elif long_debit_quote:
            long_debit_candidates = [
                quote
                for quote in long_debit_same_expiry
                if float(quote["strike"]) < float(long_debit_quote["strike"])
            ]
        else:
            long_debit_candidates = []
        if long_debit_quote and long_debit_candidates:
            long_debit_short = min(
                long_debit_candidates,
                key=lambda quote: abs(
                    float(quote["strike"]) / spot
                    - (1.08 if direction == "bullish" else 0.92)
                ),
            )
            long_debit = float(long_debit_quote["ask"]) - float(long_debit_short["bid"])
            long_debit_width = abs(
                float(long_debit_short["strike"])
                - float(long_debit_quote["strike"])
            )
            if long_debit > 0 and long_debit_width > 0:
                variants.append(
                    {
                        **long_debit_quote,
                        "strategy": "DEBIT_VERTICAL_LONG_DTE",
                        "max_horizon": LONG_OPTION_MAX_HORIZON,
                        "short_option_symbol": long_debit_short["option_symbol"],
                        "short_option_type": long_debit_short["option_type"],
                        "short_strike": long_debit_short["strike"],
                        "short_expiry": long_debit_short["expiry"],
                        "short_bid": long_debit_short["bid"],
                        "short_ask": long_debit_short["ask"],
                        "short_spread_pct": long_debit_short["spread_pct"],
                        "vertical_width": long_debit_width,
                        "entry_debit": long_debit,
                    }
                )
        credit_type = "P" if direction == "bullish" else "C"
        credit_quotes = [
            quote
            for quote in quotes
            if standard_long_candidates
            and str(quote.get("option_type")) == credit_type
            and quote.get("expiry") == long_quote.get("expiry")
        ]
        if direction == "bullish":
            credit_shorts = [quote for quote in credit_quotes if float(quote["strike"]) < spot]
        else:
            credit_shorts = [quote for quote in credit_quotes if float(quote["strike"]) > spot]
        if credit_shorts:
            credit_short = min(
                credit_shorts,
                key=lambda quote: abs(float(quote["strike"]) / spot - (0.92 if direction == "bullish" else 1.08)),
            )
            if direction == "bullish":
                credit_longs = [
                    quote for quote in credit_quotes if float(quote["strike"]) < float(credit_short["strike"])
                ]
            else:
                credit_longs = [
                    quote for quote in credit_quotes if float(quote["strike"]) > float(credit_short["strike"])
                ]
            if credit_longs:
                credit_long = min(
                    credit_longs,
                    key=lambda quote: abs(float(quote["strike"]) / spot - (0.84 if direction == "bullish" else 1.16)),
                )
                credit_received = float(credit_short["bid"]) - float(credit_long["ask"])
                width = abs(float(credit_short["strike"]) - float(credit_long["strike"]))
                if credit_received > 0 and width > credit_received:
                    variants.append(
                        {
                            **credit_short,
                            "strategy": "CREDIT_VERTICAL",
                            "credit_short_option_symbol": credit_short["option_symbol"],
                            "credit_long_option_symbol": credit_long["option_symbol"],
                            "credit_short_strike": credit_short["strike"],
                            "credit_long_strike": credit_long["strike"],
                            "credit_short_bid": credit_short["bid"],
                            "credit_short_ask": credit_short["ask"],
                            "credit_long_bid": credit_long["bid"],
                            "credit_long_ask": credit_long["ask"],
                            "credit_expiry": credit_short["expiry"],
                            "credit_width": width,
                            "entry_credit": credit_received,
                        }
                    )
        short_credit_by_expiry: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for quote in quotes:
            if (
                str(quote.get("option_type")) == credit_type
                and SHORT_OPTION_MIN_DTE <= int(quote.get("dte") or -1) <= SHORT_OPTION_MAX_DTE
                and option_quote_is_horizon_safe(quote, key[0], dates, SHORT_OPTION_MAX_HORIZON)
            ):
                short_credit_by_expiry[str(quote.get("expiry") or "")].append(quote)
        short_credit_variants: List[Dict[str, Any]] = []
        for expiry_quotes in short_credit_by_expiry.values():
            if direction == "bullish":
                short_credit_shorts = [
                    quote for quote in expiry_quotes if float(quote["strike"]) < spot
                ]
            else:
                short_credit_shorts = [
                    quote for quote in expiry_quotes if float(quote["strike"]) > spot
                ]
            if not short_credit_shorts:
                continue
            short_credit_short = min(
                short_credit_shorts,
                key=lambda quote: abs(
                    float(quote["strike"]) / spot
                    - (0.92 if direction == "bullish" else 1.08)
                ),
            )
            if direction == "bullish":
                short_credit_longs = [
                    quote
                    for quote in expiry_quotes
                    if float(quote["strike"]) < float(short_credit_short["strike"])
                ]
            else:
                short_credit_longs = [
                    quote
                    for quote in expiry_quotes
                    if float(quote["strike"]) > float(short_credit_short["strike"])
                ]
            if not short_credit_longs:
                continue
            short_credit_long = min(
                short_credit_longs,
                key=lambda quote: abs(
                    float(quote["strike"]) / spot
                    - (0.84 if direction == "bullish" else 1.16)
                ),
            )
            short_credit_received = (
                float(short_credit_short["bid"])
                - float(short_credit_long["ask"])
            )
            short_credit_width = abs(
                float(short_credit_short["strike"])
                - float(short_credit_long["strike"])
            )
            if short_credit_received <= 0 or short_credit_width <= short_credit_received:
                continue
            short_credit_variants.append(
                {
                    **short_credit_short,
                    "strategy": "CREDIT_VERTICAL_SHORT_DTE",
                    "max_horizon": SHORT_OPTION_MAX_HORIZON,
                    "credit_short_option_symbol": short_credit_short["option_symbol"],
                    "credit_long_option_symbol": short_credit_long["option_symbol"],
                    "credit_short_strike": short_credit_short["strike"],
                    "credit_long_strike": short_credit_long["strike"],
                    "credit_short_bid": short_credit_short["bid"],
                    "credit_short_ask": short_credit_short["ask"],
                    "credit_long_bid": short_credit_long["bid"],
                    "credit_long_ask": short_credit_long["ask"],
                    "credit_expiry": short_credit_short["expiry"],
                    "credit_width": short_credit_width,
                    "entry_credit": short_credit_received,
                }
            )
        if short_credit_variants:
            variants.append(
                min(
                    short_credit_variants,
                    key=lambda quote: (
                        abs(int(quote.get("dte") or 0) - 14),
                        float(quote.get("spread_pct") or 1.0),
                        abs(float(quote["credit_short_strike"]) / spot - (0.92 if direction == "bullish" else 1.08)),
                    ),
                )
            )
        chosen_quote = dict(variants[0])
        chosen_quote["variants"] = variants
        chosen[key] = chosen_quote
    json_write(
        cache_path,
        {"rows": [{"key": list(key), "quote": quote} for key, quote in chosen.items()]},
    )
    return chosen, {
        "hot_chain_raw_rows_scanned": raw_rows,
        "hot_chain_eligible_quote_rows": source_count,
        "entry_option_request_count": sum(len(values) for values in requests.values()),
        "entry_option_quote_count": len(chosen),
        "entry_option_cache_hit": False,
    }


def load_option_quote_history(
    base_dir: Path,
    dates: Sequence[str],
    symbols: Set[str],
    needed_dates: Optional[Mapping[str, Set[str]]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    cache_path = quote_history_cache_path(
        base_dir,
        "quote_history",
        {
            "schema": 4,
            "dates": list(dates),
            "symbols": sorted(symbols),
            "needed_dates": {
                symbol: sorted(targets)
                for symbol, targets in sorted((needed_dates or {}).items())
            },
        },
    )
    if cache_path.exists():
        history = load_quote_history_jsonl(cache_path)
        if history:
            return history, {
                "hot_chain_history_raw_rows_scanned": 0,
                "hot_chain_history_quote_rows": sum(
                    len(rows) for rows in history.values()
                ),
                "selected_option_symbols": len(symbols),
                "hot_chain_history_cache_hit": True,
                "hot_chain_history_target_only": needed_dates is not None,
            }
    history: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    raw_rows = 0
    quote_rows = 0
    if not symbols:
        return {}, {"hot_chain_history_raw_rows_scanned": 0, "hot_chain_history_quote_rows": 0}
    for signal_date in dates:
        for ref in find_source(base_dir / signal_date, "hot-chains", signal_date):
            for raw in iter_csv_rows(ref):
                raw_rows += 1
                symbol = str(raw.get("option_symbol") or "").strip().upper().replace(" ", "")
                if symbol not in symbols:
                    continue
                if needed_dates is not None and signal_date not in needed_dates.get(symbol, set()):
                    continue
                quote = option_quote_from_row(raw, signal_date)
                if not quote:
                    continue
                history[symbol][signal_date] = quote
                quote_rows += 1
    result = dict(history)
    write_quote_history_jsonl(cache_path, result)
    return result, {
        "hot_chain_history_raw_rows_scanned": raw_rows,
        "hot_chain_history_quote_rows": quote_rows,
        "selected_option_symbols": len(symbols),
        "hot_chain_history_cache_hit": False,
        "hot_chain_history_target_only": needed_dates is not None,
    }


def bot_eod_quote_from_cache_row(
    row: Mapping[str, Any],
    signal_date: str,
) -> Optional[Dict[str, Any]]:
    parsed = parse_occ(str(row.get("option_symbol") or ""))
    if not parsed or parsed["ticker"].startswith("^"):
        return None
    bid = as_float(row.get("bid"))
    ask = as_float(row.get("ask"))
    if bid is None or ask is None or bid < 0 or ask <= 0 or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return {
        **parsed,
        "date": signal_date,
        "source_date": signal_date,
        "dte": trading_dte(signal_date, parsed["expiry"]),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "spread_pct": (ask - bid) / mid if mid > 0 else None,
        "volume": as_float(row.get("volume")) or 0.0,
        "open_interest": as_float(row.get("open_interest")) or 0.0,
        "premium": as_float(row.get("premium")) or 0.0,
        "iv": as_float(row.get("iv")),
        "underlying_close": as_float(row.get("stock_close")),
        "source": "bot_eod",
        "quote_provenance": "same_session_bot_eod",
    }


def load_bot_eod_option_quote_history(
    base_dir: Path,
    dates: Sequence[str],
    symbols: Set[str],
    needed_dates: Optional[Mapping[str, Set[str]]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """Use cached bot-EOD NBBO rows as a selective, explicitly labeled source."""

    cache_path = quote_history_cache_path(
        base_dir,
        "bot_eod_quote_history",
        {
            "schema": 1,
            "dates": list(dates),
            "symbols": sorted(symbols),
            "needed_dates": {
                symbol: sorted(targets)
                for symbol, targets in sorted((needed_dates or {}).items())
            },
        },
    )
    if cache_path.exists():
        history = load_quote_history_jsonl(cache_path)
        if history:
            return history, {
                "bot_eod_option_quote_dates": len({date for rows in history.values() for date in rows}),
                "bot_eod_option_quote_rows": sum(len(rows) for rows in history.values()),
                "bot_eod_option_quote_cache_hit": True,
                "bot_eod_option_quote_target_only": needed_dates is not None,
            }
    history: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    source_dates = 0
    cache_quote_rows = 0
    retained_rows = 0
    cache_dir = base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
    for signal_date in dates:
        refs = find_bot_csv_sources(base_dir / signal_date, "bot-eod-report", signal_date, exact=True)
        if not refs:
            continue
        source_dates += 1
        try:
            cached = load_or_build_bot_eod_cache(
                refs,
                signal_date,
                {"bot_eod_cache_dir": str(cache_dir)},
            )
        except (OSError, ValueError, TypeError, KeyError):
            continue
        for raw in cached.get("quote_rows") or []:
            cache_quote_rows += 1
            symbol = str(raw.get("option_symbol") or "").strip().upper()
            if (
                symbol not in symbols
                or (needed_dates is not None and signal_date not in needed_dates.get(symbol, set()))
            ):
                continue
            quote = bot_eod_quote_from_cache_row(raw, signal_date)
            if not quote:
                continue
            history[symbol][signal_date] = quote
            retained_rows += 1
    result = dict(history)
    if result:
        write_quote_history_jsonl(cache_path, result)
    return result, {
        "bot_eod_option_quote_dates": source_dates,
        "bot_eod_option_quote_cache_rows_seen": cache_quote_rows,
        "bot_eod_option_quote_rows": retained_rows,
        "bot_eod_option_quote_cache_hit": False,
        "bot_eod_option_quote_target_only": needed_dates is not None,
    }


def entry_variants(entry: Mapping[str, Any]) -> List[Dict[str, Any]]:
    variants = entry.get("variants") if isinstance(entry, Mapping) else None
    if isinstance(variants, list) and variants:
        return [dict(variant) for variant in variants if isinstance(variant, Mapping)]
    return [dict(entry)] if entry else []


def entry_option_symbols(entry: Mapping[str, Any]) -> Set[str]:
    symbols: Set[str] = set()
    for variant in entry_variants(entry):
        for key in (
            "option_symbol",
            "put_option_symbol",
            "short_option_symbol",
            "credit_short_option_symbol",
            "credit_long_option_symbol",
            "iron_short_call_option_symbol",
            "iron_long_call_option_symbol",
            "iron_short_put_option_symbol",
            "iron_long_put_option_symbol",
        ):
            symbol = str(variant.get(key) or "").strip().upper()
            if symbol:
                symbols.add(symbol)
    return symbols


def quote_target_dates(
    entry_quotes: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    dates: Sequence[str],
) -> Dict[str, Set[str]]:
    """Return only target sessions that an entry variant can actually score."""

    needed: Dict[str, Set[str]] = defaultdict(set)
    for (signal_date, _ticker, _direction), entry in entry_quotes.items():
        for variant in entry_variants(entry):
            symbols = entry_option_symbols({"variants": [variant]})
            max_horizon = int(variant.get("max_horizon") or max(OPTION_HORIZONS))
            for horizon in OPTION_HORIZONS:
                if horizon > max_horizon:
                    continue
                target = session_target(dates, signal_date, horizon)
                if target:
                    for symbol in symbols:
                        needed[symbol].add(target)
    return dict(needed)


def load_chain_oi_fallback_history(
    base_dir: Path,
    dates: Sequence[str],
    symbols: Set[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """Read prior quotes for selected contracts without treating OI as flow."""

    cache_path = quote_cache_path(
        base_dir,
        "chain_oi_history",
        {"schema": 2, "dates": list(dates), "symbols": sorted(symbols)},
    )
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            history = dict(cached.get("history") or {})
            return history, {
                "chain_oi_raw_rows_scanned": 0,
                "chain_oi_quote_rows": sum(len(rows) for rows in history.values()),
                "chain_oi_history_cache_hit": True,
            }
        except (OSError, ValueError, TypeError):
            pass
    history: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    raw_rows = 0
    quote_rows = 0
    if not symbols:
        return {}, {"chain_oi_raw_rows_scanned": 0, "chain_oi_quote_rows": 0}
    for signal_date in dates:
        for ref in find_source(base_dir / signal_date, "chain-oi-changes", signal_date):
            for raw in iter_csv_rows(ref):
                raw_rows += 1
                symbol = str(raw.get("option_symbol") or "").strip().upper().replace(" ", "")
                if symbol not in symbols:
                    continue
                quote_date = str(raw.get("last_date") or "")[:10]
                bid = as_float(raw.get("last_bid"))
                ask = as_float(raw.get("last_ask"))
                parsed = parse_occ(symbol)
                if (
                    not parsed
                    or not DATE_RE.fullmatch(quote_date)
                    or quote_date > signal_date
                    or bid is None
                    or ask is None
                    or ask <= 0
                    or ask < bid
                ):
                    continue
                mid = (bid + ask) / 2.0
                history[symbol][quote_date] = {
                    **parsed,
                    "date": quote_date,
                    "dte": trading_dte(quote_date, parsed["expiry"]),
                    "bid": bid,
                    "ask": ask,
                    "mid": mid,
                    "spread_pct": (ask - bid) / mid if mid > 0 else None,
                    "volume": as_float(raw.get("prev_vol")) or 0.0,
                    "open_interest": as_float(raw.get("last_oi")) or 0.0,
                    "premium": as_float(raw.get("prev_total_premium")) or 0.0,
                    "iv": None,
                    "underlying_close": as_float(raw.get("stock_price")),
                    "source": "chain_oi_prior_quote",
                }
                quote_rows += 1
    result = dict(history)
    json_write(cache_path, {"history": result})
    return result, {
        "chain_oi_raw_rows_scanned": raw_rows,
        "chain_oi_quote_rows": quote_rows,
        "chain_oi_history_cache_hit": False,
    }


def load_chain_oi_fallback_history_indexed(
    base_dir: Path,
    dates: Sequence[str],
    symbols: Set[str],
    needed_dates: Optional[Mapping[str, Set[str]]] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    """Load chain fallback quotes through source-indexed compressed caches."""

    cache_path = quote_history_cache_path(
        base_dir,
        "chain_oi_target_history",
        {
            "schema": 3,
            "dates": list(dates),
            "symbols": sorted(symbols),
            "needed_dates": {
                symbol: sorted(targets)
                for symbol, targets in sorted((needed_dates or {}).items())
            },
        },
    )
    if cache_path.exists():
        history = load_quote_history_jsonl(cache_path)
        if history:
            return history, {
                "chain_oi_raw_rows_scanned": 0,
                "chain_oi_index_rows_scanned": 0,
                "chain_oi_quote_rows": sum(len(rows) for rows in history.values()),
                "selected_option_symbols": len(symbols),
                "chain_oi_history_cache_hit": True,
                "chain_oi_filtered_history_cache_hit": True,
                "chain_oi_source_index_hits": 0,
                "chain_oi_source_index_built": 0,
                "chain_oi_history_target_only": needed_dates is not None,
            }

    history: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    raw_rows = 0
    indexed_rows = 0
    quote_rows = 0
    index_hits = 0
    index_built = 0
    sqlite_hits = 0
    sqlite_built = 0
    if not symbols:
        return {}, {
            "chain_oi_raw_rows_scanned": 0,
            "chain_oi_index_rows_scanned": 0,
            "chain_oi_quote_rows": 0,
            "chain_oi_source_index_hits": 0,
            "chain_oi_source_index_built": 0,
        }
    for signal_date in dates:
        for ref in find_source(base_dir / signal_date, "chain-oi-changes", signal_date):
            index_path = source_index_path(base_dir, "chain_oi", ref)
            if not index_path.exists():
                source_raw_rows, _ = build_chain_source_index(index_path, ref, signal_date)
                raw_rows += source_raw_rows
                index_built += 1
            else:
                index_hits += 1
            sqlite_path = source_sqlite_path(base_dir, "chain_oi", ref, symbols)
            if not sqlite_path.exists():
                build_chain_sqlite_index(sqlite_path, index_path, symbols)
                sqlite_built += 1
            else:
                sqlite_hits += 1
            connection = sqlite3.connect(sqlite_path)
            try:
                symbol_list = sorted(symbols)
                for start in range(0, len(symbol_list), 500):
                    chunk = symbol_list[start : start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = connection.execute(
                        f"SELECT payload FROM quotes WHERE symbol IN ({placeholders})",
                        chunk,
                    )
                    for (payload,) in rows:
                        try:
                            indexed = json.loads(payload)
                        except (TypeError, ValueError):
                            continue
                        indexed_rows += 1
                        symbol = str(indexed.get("symbol") or "").strip().upper()
                        quote_date = str(indexed.get("quote_date") or "")[:10]
                        source_date = str(indexed.get("source_date") or "")[:10]
                        if (
                            not DATE_RE.fullmatch(quote_date)
                            or not DATE_RE.fullmatch(source_date)
                            or quote_date > signal_date
                            or source_date != signal_date
                            or quote_date > source_date
                            or (needed_dates is not None and quote_date not in needed_dates.get(symbol, set()))
                        ):
                            continue
                        bid = as_float(indexed.get("bid"))
                        ask = as_float(indexed.get("ask"))
                        if bid is None or ask is None or ask <= 0 or ask < bid:
                            continue
                        parsed = {
                            "option_symbol": symbol,
                            "ticker": clean_ticker(indexed.get("ticker")),
                            "expiry": str(indexed.get("expiry") or ""),
                            "option_type": str(indexed.get("option_type") or ""),
                            "strike": as_float(indexed.get("strike")),
                        }
                        mid = (bid + ask) / 2.0
                        history[symbol][quote_date] = {
                            **parsed,
                            "date": quote_date,
                            "source_date": source_date,
                            "dte": trading_dte(quote_date, parsed["expiry"]),
                            "bid": bid,
                            "ask": ask,
                            "mid": mid,
                            "spread_pct": (ask - bid) / mid if mid > 0 else None,
                            "volume": as_float(indexed.get("prev_vol")) or 0.0,
                            "open_interest": as_float(indexed.get("last_oi")) or 0.0,
                            "premium": as_float(indexed.get("prev_total_premium")) or 0.0,
                            "iv": None,
                            "underlying_close": as_float(indexed.get("stock_price")),
                            "source": "chain_oi_prior_quote",
                            "quote_provenance": "next_session_chain_oi_prior_quote",
                        }
                        quote_rows += 1
            finally:
                connection.close()
    result = dict(history)
    write_quote_history_jsonl(cache_path, result)
    return result, {
        "chain_oi_raw_rows_scanned": raw_rows,
        "chain_oi_index_rows_scanned": indexed_rows,
        "chain_oi_quote_rows": quote_rows,
        "selected_option_symbols": len(symbols),
        "chain_oi_history_cache_hit": bool(index_hits and not index_built),
        "chain_oi_source_index_hits": index_hits,
        "chain_oi_source_index_built": index_built,
        "chain_oi_sqlite_index_hits": sqlite_hits,
        "chain_oi_sqlite_index_built": sqlite_built,
        "chain_oi_history_target_only": needed_dates is not None,
        "chain_oi_filtered_history_cache_hit": False,
    }


def merge_quote_history(
    hot: Mapping[str, Mapping[str, Dict[str, Any]]],
    fallback: Mapping[str, Mapping[str, Dict[str, Any]]],
    bot: Optional[Mapping[str, Mapping[str, Dict[str, Any]]]] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    merged: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for symbol, rows in fallback.items():
        merged[symbol].update(rows)
    for symbol, rows in (bot or {}).items():
        merged[symbol].update(rows)
    for symbol, rows in hot.items():
        merged[symbol].update(rows)
    return dict(merged)


def option_required_symbols(entry: Mapping[str, Any]) -> List[str]:
    """Return every contract leg required to mark an option structure."""

    symbol = str(entry.get("option_symbol") or "")
    strategy = str(entry.get("strategy") or "LONG_OPTION")
    required = [symbol]
    if strategy in {
        "DEBIT_VERTICAL",
        "DEBIT_VERTICAL_SHORT_DTE",
        "DEBIT_VERTICAL_LONG_DTE",
    }:
        required.append(str(entry.get("short_option_symbol") or ""))
    elif strategy == "LONG_STRADDLE":
        required.append(str(entry.get("put_option_symbol") or ""))
    elif strategy in {"CREDIT_VERTICAL", "CREDIT_VERTICAL_SHORT_DTE"}:
        required.append(str(entry.get("credit_long_option_symbol") or ""))
    elif strategy == "IRON_CONDOR":
        required.extend(
            str(entry.get(key) or "")
            for key in (
                "iron_long_call_option_symbol",
                "iron_short_put_option_symbol",
                "iron_long_put_option_symbol",
            )
        )
    return required


def option_outcome_missing_reason(
    entry: Mapping[str, Any],
    quote_history: Mapping[str, Mapping[str, Dict[str, Any]]],
    dates: Sequence[str],
    horizon: int,
) -> str:
    """Explain why a fixed-horizon option outcome was not scored."""

    target_date = session_target(dates, str(entry.get("date") or ""), horizon)
    if not target_date:
        return "PENDING_FUTURE"
    if int(entry.get("max_horizon") or max(OPTION_HORIZONS)) < horizon:
        return "INELIGIBLE_HORIZON"
    expiry = str(entry.get("expiry") or "")
    if DATE_RE.fullmatch(expiry) and target_date > expiry:
        return "TARGET_AFTER_EXPIRY"
    required_symbols = option_required_symbols(entry)
    if any(not symbol for symbol in required_symbols):
        return "INVALID_CONTRACT_LEGS"
    missing_symbols = [
        symbol
        for symbol in required_symbols
        if target_date not in quote_history.get(symbol, {})
    ]
    if missing_symbols:
        return "TARGET_QUOTE_MISSING"
    return "INVALID_EXIT_QUOTE"


def option_outcome(
    entry: Mapping[str, Any],
    quote_history: Mapping[str, Mapping[str, Dict[str, Any]]],
    dates: Sequence[str],
    horizon: int,
) -> Optional[Dict[str, Any]]:
    symbol = str(entry.get("option_symbol") or "")
    target_date = session_target(dates, str(entry.get("date") or ""), horizon)
    if not symbol or not target_date:
        return None
    if int(entry.get("max_horizon") or max(OPTION_HORIZONS)) < horizon:
        return None
    expiry = str(entry.get("expiry") or "")
    if DATE_RE.fullmatch(expiry) and target_date > expiry:
        return None
    strategy = str(entry.get("strategy") or "LONG_OPTION")
    required_symbols = option_required_symbols(entry)
    if any(not required for required in required_symbols):
        return None
    histories = {required: quote_history.get(required, {}) for required in required_symbols}
    # This is a fixed-horizon mark, not a path-dependent stop/roll strategy.
    # Requiring a quote on every intermediate session incorrectly converted
    # valid target-day exits into missing outcomes.
    if any(target_date not in history_rows for history_rows in histories.values()):
        return None
    exit_date = target_date
    history = histories[symbol]
    exit_quote = history[exit_date]
    exit_sources = [
        str(history_rows[exit_date].get("source") or "")
        for history_rows in histories.values()
    ]
    exit_source_dates = [
        str(history_rows[exit_date].get("source_date") or "")
        for history_rows in histories.values()
    ]
    exit_provenances = {
        str(history_rows[exit_date].get("quote_provenance") or "")
        for history_rows in histories.values()
        if history_rows[exit_date].get("quote_provenance")
    }
    if len(exit_provenances) == 1:
        exit_quote_provenance = next(iter(exit_provenances))
    elif exit_provenances:
        exit_quote_provenance = "mixed_quote_provenance"
    else:
        exit_quote_provenance = "unknown_quote_provenance"
    entry_ask = as_float(entry.get("ask"))
    exit_bid = as_float(exit_quote.get("bid"))
    if strategy not in {"CREDIT_VERTICAL", "CREDIT_VERTICAL_SHORT_DTE", "IRON_CONDOR"} and (
        entry_ask is None or exit_bid is None or entry_ask <= 0 or exit_bid < 0
    ):
        return None
    if strategy in {
        "DEBIT_VERTICAL",
        "DEBIT_VERTICAL_SHORT_DTE",
        "DEBIT_VERTICAL_LONG_DTE",
    }:
        short_entry_bid = as_float(entry.get("short_bid"))
        short_exit_ask = as_float(quote_history[str(entry["short_option_symbol"])][exit_date].get("ask"))
        if short_entry_bid is None or short_exit_ask is None:
            return None
        entry_debit = entry_ask - short_entry_bid
        exit_credit = exit_bid - short_exit_ask
        if entry_debit <= 0:
            return None
        exit_credit = max(0.0, exit_credit)
        gross_pnl = (exit_credit - entry_debit) * 100.0
        net_pnl = gross_pnl - 4.0 * OPTION_FEE_PER_CONTRACT
        risk = entry_debit * 100.0 + 4.0 * OPTION_FEE_PER_CONTRACT
        entry_price = entry_debit
        exit_price = exit_credit
    elif strategy == "LONG_STRADDLE":
        put_entry_ask = as_float(entry.get("put_ask"))
        put_exit_bid = as_float(quote_history[str(entry["put_option_symbol"])][exit_date].get("bid"))
        if put_entry_ask is None or put_exit_bid is None or put_exit_bid < 0:
            return None
        entry_debit = entry_ask + put_entry_ask
        exit_credit = exit_bid + put_exit_bid
        gross_pnl = (exit_credit - entry_debit) * 100.0
        net_pnl = gross_pnl - 4.0 * OPTION_FEE_PER_CONTRACT
        risk = entry_debit * 100.0 + 4.0 * OPTION_FEE_PER_CONTRACT
        entry_price = entry_debit
        exit_price = exit_credit
    elif strategy in {"CREDIT_VERTICAL", "CREDIT_VERTICAL_SHORT_DTE"}:
        short_exit_ask = as_float(quote_history[str(entry["credit_short_option_symbol"])][exit_date].get("ask"))
        long_exit_bid = as_float(quote_history[str(entry["credit_long_option_symbol"])][exit_date].get("bid"))
        entry_credit = as_float(entry.get("entry_credit"))
        width = as_float(entry.get("credit_width"))
        if (
            short_exit_ask is None
            or long_exit_bid is None
            or entry_credit is None
            or width is None
            or short_exit_ask <= 0
            or entry_credit <= 0
            or width <= entry_credit
        ):
            return None
        exit_debit = short_exit_ask - long_exit_bid
        exit_debit = max(0.0, exit_debit)
        gross_pnl = (entry_credit - exit_debit) * 100.0
        net_pnl = gross_pnl - 4.0 * OPTION_FEE_PER_CONTRACT
        risk = (width - entry_credit) * 100.0 + 4.0 * OPTION_FEE_PER_CONTRACT
        entry_price = entry_credit
        exit_price = exit_debit
    elif strategy == "IRON_CONDOR":
        short_call_exit_ask = as_float(
            quote_history[str(entry["iron_short_call_option_symbol"])][exit_date].get("ask")
        )
        long_call_exit_bid = as_float(
            quote_history[str(entry["iron_long_call_option_symbol"])][exit_date].get("bid")
        )
        short_put_exit_ask = as_float(
            quote_history[str(entry["iron_short_put_option_symbol"])][exit_date].get("ask")
        )
        long_put_exit_bid = as_float(
            quote_history[str(entry["iron_long_put_option_symbol"])][exit_date].get("bid")
        )
        entry_credit = as_float(entry.get("entry_credit"))
        call_width = as_float(entry.get("iron_call_width"))
        put_width = as_float(entry.get("iron_put_width"))
        if (
            short_call_exit_ask is None
            or long_call_exit_bid is None
            or short_put_exit_ask is None
            or long_put_exit_bid is None
            or entry_credit is None
            or call_width is None
            or put_width is None
            or short_call_exit_ask <= 0
            or short_put_exit_ask <= 0
            or entry_credit <= 0
            or max(call_width, put_width) <= entry_credit
        ):
            return None
        exit_debit = (
            short_call_exit_ask
            + short_put_exit_ask
            - long_call_exit_bid
            - long_put_exit_bid
        )
        exit_debit = max(0.0, exit_debit)
        gross_pnl = (entry_credit - exit_debit) * 100.0
        net_pnl = gross_pnl - 4.0 * OPTION_FEE_PER_CONTRACT
        risk = (max(call_width, put_width) - entry_credit) * 100.0 + 4.0 * OPTION_FEE_PER_CONTRACT
        entry_price = entry_credit
        exit_price = exit_debit
    else:
        gross_pnl = (exit_bid - entry_ask) * 100.0
        net_pnl = gross_pnl - 2.0 * OPTION_FEE_PER_CONTRACT
        risk = entry_ask * 100.0 + 2.0 * OPTION_FEE_PER_CONTRACT
        entry_price = entry_ask
        exit_price = exit_bid
    return {
        "target_date": target_date,
        "exit_date": exit_date,
        "entry_ask": entry_ask,
        "exit_bid": exit_bid,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "net_R": net_pnl / risk,
        "win": net_pnl > 0,
        "exit_quote_source": ";".join(sorted(set(filter(None, exit_sources)))),
        "exit_quote_source_dates": ";".join(sorted(set(filter(None, exit_source_dates)))),
        "exit_quote_provenance": exit_quote_provenance,
    }


def option_outcome_status(
    signal_date: str,
    horizon: int,
    dates: Sequence[str],
    outcome: Optional[Mapping[str, Any]],
    entry: Optional[Mapping[str, Any]] = None,
    quote_history: Optional[Mapping[str, Mapping[str, Dict[str, Any]]]] = None,
) -> str:
    """Distinguish a future, not-yet-observable exit from a missing quote."""

    if outcome:
        return "SCORED"
    if entry is not None and quote_history is not None:
        reason = option_outcome_missing_reason(entry, quote_history, dates, horizon)
        if reason == "PENDING_FUTURE":
            return reason
        if reason.startswith("TARGET_AFTER_EXPIRY") or reason == "INELIGIBLE_HORIZON":
            return "INELIGIBLE_CONTRACT"
    try:
        signal_index = list(dates).index(signal_date)
    except ValueError:
        return "MISSING_EXIT_QUOTE"
    if signal_index + int(horizon) >= len(dates):
        return "PENDING_FUTURE"
    return "MISSING_EXIT_QUOTE"


def wilson_lower(wins: int, total: int, z: float = 1.96) -> Optional[float]:
    if total <= 0:
        return None
    p = wins / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    spread = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (centre - spread) / denominator


def profit_factor(values: Sequence[float]) -> Optional[float]:
    wins = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if losses <= 0:
        return float("inf") if wins > 0 else None
    return wins / losses


def max_drawdown(values: Sequence[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        equity += value
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return drawdown


def is_predictive_option_row(row: Mapping[str, Any]) -> bool:
    return row.get("status") == "SCORED" and row.get("signal_role") != "same_day_event"


def validation_lane_matches(row: Mapping[str, Any]) -> List[str]:
    """Return every predeclared contract lane matching an entry row."""

    matches: List[str] = []
    for spec in VALIDATION_LANE_SPECS:
        if str(row.get("pattern_family") or "") != spec["family"]:
            continue
        if spec["direction"] != "any" and str(row.get("direction") or "") != spec["direction"]:
            continue
        if int(row.get("horizon") or 0) != int(spec["horizon"]):
            continue
        if str(row.get("strategy") or "") != spec["strategy"]:
            continue
        spread = as_float(row.get("entry_spread_pct"))
        dte = as_float(row.get("dte"))
        if spread is None or dte is None or spread > float(spec["max_spread"]):
            continue
        if not float(spec["min_dte"]) <= dte <= float(spec["max_dte"]):
            continue
        max_spread_to_implied = spec.get("max_spread_to_implied_move")
        if max_spread_to_implied is not None:
            implied_move = as_float(row.get("implied_move_perc"))
            if (
                implied_move is None
                or implied_move <= 0
                or spread > implied_move * float(max_spread_to_implied)
            ):
                continue
        volume_ratio = as_float(row.get("volume_ratio_30d")) or 0.0
        if volume_ratio < float(spec.get("min_volume_ratio") or 0.0):
            continue
        bot_bias = as_float(row.get("bot_eod_flow_bias"))
        if spec.get("min_bot_bias_abs") is not None and (
            bot_bias is None or abs(bot_bias) < float(spec["min_bot_bias_abs"])
        ):
            continue
        if spec.get("min_bot_premium") is not None and (
            (as_float(row.get("bot_eod_flow_total_premium")) or 0.0) < float(spec["min_bot_premium"])
        ):
            continue
        matches.append(str(spec["name"]))
    return matches


def validation_lane_for_row(row: Mapping[str, Any]) -> str:
    """Return the first fixed contract lane, without looking at outcomes."""

    matches = validation_lane_matches(row)
    return matches[0] if matches else ""


def validation_lane_horizon(lane: str) -> Optional[int]:
    for spec in VALIDATION_LANE_SPECS:
        if str(spec.get("name") or "") == str(lane or ""):
            return int(spec["horizon"])
    return None


def validation_lane_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Clone predictive rows, including missing outcomes, into fixed lanes."""

    result: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("signal_role") == "same_day_event":
            continue
        lanes = validation_lane_matches(row)
        if not lanes:
            continue
        for lane in lanes:
            clone = dict(row)
            clone["pattern_family"] = f"LANE::{lane}"
            clone["validation_lane"] = lane
            result.append(clone)
    return result


def option_outcome_coverage(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Measure scored versus missing target exits without dropping missing rows."""

    groups: Dict[Tuple[str, str, int, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("signal_role") == "same_day_event":
            continue
        status = str(row.get("status") or "")
        if status not in {"SCORED", "MISSING_EXIT_QUOTE"}:
            continue
        groups[pattern_key(row)][status] += 1
    result: List[Dict[str, Any]] = []
    for family, direction, horizon, strategy in sorted(groups):
        counts = groups[(family, direction, horizon, strategy)]
        scored = counts["SCORED"]
        missing = counts["MISSING_EXIT_QUOTE"]
        total = scored + missing
        result.append(
            {
                "pattern_family": family,
                "direction": direction,
                "horizon": horizon,
                "strategy": strategy,
                "scored_count": scored,
                "missing_exit_count": missing,
                "total_outcome_count": total,
                "coverage_ratio": scored / total if total else None,
                "coverage_gate": "PASS" if total and scored / total >= MIN_OPTION_OUTCOME_COVERAGE else "FAIL",
            }
        )
    return result


def rolling_holdout_stats(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    window_size: int = 15,
    window_count: int = 4,
) -> List[Dict[str, Any]]:
    """Evaluate the most recent non-overlapping date windows chronologically."""

    groups: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(value_key))
        horizon = int(row.get("horizon") or 0)
        if value is None or not horizon:
            continue
        groups[
            (
                str(row.get("pattern_family") or ""),
                str(row.get("direction") or ""),
                horizon,
                str(row.get("strategy") or ""),
            )
        ].append(row)
    result: List[Dict[str, Any]] = []
    for (family, direction, horizon, strategy), members in sorted(groups.items()):
        dates = sorted({str(row.get("signal_date")) for row in members})
        for fold in range(window_count):
            end = len(dates) - fold * window_size
            start = max(0, end - window_size)
            if end <= start:
                continue
            test_dates = set(dates[start:end])
            test_rows = sorted(
                (row for row in members if str(row.get("signal_date")) in test_dates),
                key=lambda row: (str(row.get("signal_date") or ""), str(row.get("ticker") or ""), str(row.get("option_symbol") or "")),
            )
            values = [float(row[value_key]) for row in test_rows if as_float(row.get(value_key)) is not None]
            if not values:
                continue
            wins = sum(value > 0 for value in values)
            result.append(
                {
                    "pattern_family": family,
                    "direction": direction,
                    "horizon": horizon,
                    "strategy": strategy,
                    "validation_type": "rolling_holdout",
                    "fold": window_count - fold,
                    "test_start_date": min(test_dates),
                    "test_end_date": max(test_dates),
                    "test_sample_count": len(values),
                    "test_unique_signal_dates": len(test_dates),
                    "test_win_rate": wins / len(values),
                    "test_win_rate_lower_95": wilson_lower(wins, len(values)),
                    "test_average_value": statistics.fmean(values),
                    "test_profit_factor": profit_factor(values),
                    "test_max_drawdown": max_drawdown(values),
                }
            )
    return result


def grouped_stats(rows: Sequence[Mapping[str, Any]], value_key: str) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(value_key))
        horizon = int(row.get("horizon") or 0)
        if value is None or not horizon:
            continue
        groups[
            (
                str(row.get("pattern_family") or ""),
                str(row.get("direction") or ""),
                horizon,
                str(row.get("strategy") or ""),
            )
        ].append(row)
    result: List[Dict[str, Any]] = []
    for (family, direction, horizon, strategy), members in sorted(groups.items()):
        ordered_members = sorted(
            (row for row in members if as_float(row.get(value_key)) is not None),
            key=lambda row: (str(row.get("signal_date") or ""), str(row.get("ticker") or ""), str(row.get("option_symbol") or "")),
        )
        values = [float(row[value_key]) for row in ordered_members]
        wins = sum(value > 0 for value in values)
        dates = sorted({str(row.get("signal_date")) for row in ordered_members})
        latest_dates = set(dates[-20:])
        latest_values = [
            value for row, value in zip(ordered_members, values) if str(row.get("signal_date")) in latest_dates
        ]
        by_date: Dict[str, List[float]] = defaultdict(list)
        for row, value in zip(ordered_members, values):
            by_date[str(row.get("signal_date"))].append(value)
        date_values = [statistics.fmean(by_date[signal_date]) for signal_date in sorted(by_date)]
        date_stdev = statistics.stdev(date_values) if len(date_values) >= 2 else None
        date_lower_mean = (
            statistics.fmean(date_values) - 1.96 * date_stdev / math.sqrt(len(date_values))
            if date_stdev is not None and len(date_values) > 1
            else (date_values[0] if date_values else None)
        )
        stdev = statistics.stdev(values) if len(values) >= 2 else None
        lower_mean = (
            statistics.fmean(values) - 1.96 * stdev / math.sqrt(len(values))
            if stdev is not None and len(values) > 1
            else (values[0] if values else None)
        )
        result.append(
            {
                "pattern_family": family,
                "direction": direction,
                "horizon": horizon,
                "strategy": strategy,
                "sample_count": len(values),
                "unique_signal_dates": len(dates),
                "win_rate": wins / len(values) if values else None,
                "win_rate_lower_95": wilson_lower(wins, len(values)),
                "average_value": statistics.fmean(values) if values else None,
                "lower_mean_95": lower_mean,
                "profit_factor": profit_factor(values),
                "max_drawdown": max_drawdown(values),
                "date_average_value": statistics.fmean(date_values) if date_values else None,
                "date_lower_mean_95": date_lower_mean,
                "date_max_drawdown": max_drawdown(date_values),
                "latest_holdout_count": len(latest_values),
                "latest_holdout_average": statistics.fmean(latest_values) if latest_values else None,
                "latest_holdout_profit_factor": profit_factor(latest_values),
                "first_signal_date": dates[0] if dates else "",
                "last_signal_date": dates[-1] if dates else "",
            }
        )
    return result


def walk_forward_stats(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    block_count: int = 5,
) -> List[Dict[str, Any]]:
    """Evaluate fixed signal families on sequential, non-overlapping test blocks."""

    groups: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(value_key))
        horizon = int(row.get("horizon") or 0)
        if value is None or not horizon:
            continue
        groups[
            (
                str(row.get("pattern_family") or ""),
                str(row.get("direction") or ""),
                horizon,
                str(row.get("strategy") or ""),
            )
        ].append(row)
    result: List[Dict[str, Any]] = []
    for (family, direction, horizon, strategy), members in sorted(groups.items()):
        dates = sorted({str(row.get("signal_date")) for row in members})
        if len(dates) < block_count * 2:
            continue
        block_size = max(1, len(dates) // block_count)
        for block in range(1, block_count):
            start_index = block * block_size
            end_index = len(dates) if block == block_count - 1 else min(len(dates), (block + 1) * block_size)
            test_dates = set(dates[start_index:end_index])
            if not test_dates:
                continue
            train_rows = [row for row in members if str(row.get("signal_date")) < min(test_dates)]
            test_rows = sorted(
                (row for row in members if str(row.get("signal_date")) in test_dates),
                key=lambda row: (str(row.get("signal_date") or ""), str(row.get("ticker") or ""), str(row.get("option_symbol") or "")),
            )
            test_values = [float(row[value_key]) for row in test_rows if as_float(row.get(value_key)) is not None]
            if not test_values:
                continue
            train_values = [float(row[value_key]) for row in train_rows if as_float(row.get(value_key)) is not None]
            wins = sum(value > 0 for value in test_values)
            result.append(
                {
                    "pattern_family": family,
                    "direction": direction,
                    "horizon": horizon,
                    "strategy": strategy,
                    "validation_type": "blocked_walk_forward",
                    "fold": block,
                    "train_end_date": dates[start_index - 1],
                    "test_start_date": min(test_dates),
                    "test_end_date": max(test_dates),
                    "train_sample_count": len(train_values),
                    "train_average_value": statistics.fmean(train_values) if train_values else None,
                    "test_sample_count": len(test_values),
                    "test_unique_signal_dates": len(test_dates),
                    "test_win_rate": wins / len(test_values),
                    "test_win_rate_lower_95": wilson_lower(wins, len(test_values)),
                    "test_average_value": statistics.fmean(test_values),
                    "test_profit_factor": profit_factor(test_values),
                    "test_max_drawdown": max_drawdown(test_values),
                }
            )
    return result


def calibration_stats(
    rows: Sequence[Mapping[str, Any]],
    value_key: str,
    bin_count: int = 5,
) -> List[Dict[str, Any]]:
    """Calibrate win probability from prior outcomes only.

    The former implementation ranked all rows by the same-history score before
    measuring outcomes. That is a diagnostic description of a score, not a
    leakage-safe confidence estimate. This version walks each group in time
    order and shrinks the prior win rate toward 50% before assigning a fixed
    confidence bucket.
    """

    groups: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = as_float(row.get(value_key))
        horizon = int(row.get("horizon") or 0)
        if value is None or not horizon:
            continue
        groups[
            (
                str(row.get("pattern_family") or ""),
                str(row.get("direction") or ""),
                horizon,
                str(row.get("strategy") or ""),
            )
        ].append(row)
    result: List[Dict[str, Any]] = []
    for (family, direction, horizon, strategy), members in sorted(groups.items()):
        ordered = sorted(
            members,
            key=lambda row: (str(row.get("signal_date") or ""), str(row.get("ticker") or ""), str(row.get("option_symbol") or "")),
        )
        if not ordered:
            continue
        buckets: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        prior_wins = 0
        prior_total = 0
        for row in ordered:
            value = as_float(row.get(value_key))
            if value is None:
                continue
            # A 20-observation, 50/50 prior prevents a single early winner
            # from being presented as high confidence.
            predicted_probability = (prior_wins + 10.0) / (prior_total + 20.0)
            if predicted_probability >= 0.70:
                bucket_number = 1
            elif predicted_probability >= 0.60:
                bucket_number = 2
            elif predicted_probability >= 0.50:
                bucket_number = 3
            elif predicted_probability >= 0.40:
                bucket_number = 4
            else:
                bucket_number = 5
            buckets[bucket_number].append((predicted_probability, value))
            prior_wins += int(value > 0)
            prior_total += 1
        for bucket_number, pairs in sorted(buckets.items()):
            probabilities = [pair[0] for pair in pairs]
            values = [pair[1] for pair in pairs]
            wins = sum(value > 0 for value in values)
            result.append(
                {
                    "pattern_family": family,
                    "direction": direction,
                    "horizon": horizon,
                    "strategy": strategy,
                    "score_bin": bucket_number,
                    "sample_count": len(values),
                    "score_min": min(probabilities),
                    "score_max": max(probabilities),
                    "score_average": statistics.fmean(probabilities),
                    "calibration_method": "prior_only_beta_mean_win_rate",
                    "win_rate": wins / len(values),
                    "win_rate_lower_95": wilson_lower(wins, len(values)),
                    "average_value": statistics.fmean(values),
                    "profit_factor": profit_factor(values),
                }
            )
    return result


def known_mover_audit(
    features_by_date: Mapping[str, Mapping[str, Dict[str, Any]]],
    dates: Sequence[str],
    signals: Sequence[PriceSignal],
    top_per_date: int = 25,
) -> List[Dict[str, Any]]:
    signal_index: Dict[Tuple[str, str], List[PriceSignal]] = defaultdict(list)
    for signal in signals:
        signal_index[(signal.date, signal.ticker)].append(signal)
    audit: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for index, signal_date in enumerate(dates[:-1]):
        next_date = dates[index + 1]
        moves: List[Tuple[float, str, float]] = []
        for ticker, row in features_by_date.get(signal_date, {}).items():
            next_row = features_by_date.get(next_date, {}).get(ticker)
            if (
                not next_row
                or not row.get("adjusted_close", row.get("close"))
                or not next_row.get("adjusted_close", next_row.get("close"))
            ):
                continue
            move = realized_stock_return(
                features_by_date,
                dates,
                ticker,
                signal_date,
                next_date,
            )
            if move is None:
                continue
            moves.append((abs(move), ticker, move))
        movers = sorted(moves, reverse=True)[:top_per_date]
        tickers = set(TARGET_TICKERS) | {ticker for _, ticker, _ in movers}
        move_map = {ticker: (absolute, move) for absolute, ticker, move in moves}
        for ticker in sorted(tickers):
            if ticker not in move_map:
                continue
            absolute, move = move_map[ticker]
            key = (signal_date, ticker)
            if ticker not in TARGET_TICKERS and ticker not in {item[1] for item in movers}:
                continue
            matched = signal_index.get(key, [])
            expected = "bullish" if move >= 0 else "bearish"
            same_direction = [signal for signal in matched if signal.direction == expected]
            next_session_signals = signal_index.get((next_date, ticker), [])
            next_session_event = any(
                signal.role == "same_day_event" and signal.direction == expected
                for signal in next_session_signals
            )
            families = ";".join(sorted({signal.family for signal in same_direction}))
            audit.append(
                {
                    "signal_date": signal_date,
                    "next_date": next_date,
                    "ticker": ticker,
                    "next_day_move": move,
                    "abs_next_day_move": absolute,
                    "significant_move_5pct": absolute >= 0.05,
                    "next_day_rank_abs": next(
                        (rank for rank, item in enumerate(sorted(moves, reverse=True), 1) if item[1] == ticker),
                        None,
                    ),
                    "expected_direction": expected,
                    "price_signal_any_direction": bool(matched),
                    "price_signal_same_direction": bool(same_direction),
                    "same_direction_families": families,
                    "same_day_event_detected": any(signal.role == "same_day_event" for signal in matched),
                    "next_session_event_detected": next_session_event,
                    "miss_reason": (
                        "flagged_same_direction"
                        if same_direction
                        else "flagged_opposite_direction"
                        if matched
                        else "no_price_signal"
                    ),
                }
            )
    audit.sort(key=lambda row: (row["signal_date"], row["abs_next_day_move"]), reverse=True)
    return audit


def feature_coverage(
    history: Mapping[str, Mapping[str, PriceRow]],
    dates: Sequence[str],
    signals: Sequence[PriceSignal],
) -> List[Dict[str, Any]]:
    signal_index: Dict[Tuple[str, str], int] = defaultdict(int)
    for signal in signals:
        signal_index[(signal.date, signal.ticker)] += 1
    rows = []
    for ticker in sorted(set(history) & TARGET_TICKERS):
        observed = sorted(history[ticker])
        rows.append(
            {
                "ticker": ticker,
                "observed_sessions": len(observed),
                "first_session": observed[0] if observed else "",
                "last_session": observed[-1] if observed else "",
                "signal_date_ticker_count": sum(signal_index.get((d, ticker), 0) for d in dates),
                "source_present_every_session": len(observed) == len(dates),
            }
        )
    return rows


def option_signal_requests(
    signals: Sequence[PriceSignal],
    as_of: str,
    max_per_date: int,
) -> Dict[str, Set[Tuple[str, str]]]:
    grouped: Dict[str, List[PriceSignal]] = defaultdict(list)
    for signal in signals:
        if signal.date > as_of:
            continue
        grouped[signal.date].append(signal)
    requests: Dict[str, Set[Tuple[str, str]]] = {}
    for signal_date, rows in grouped.items():
        ranked = sorted(rows, key=lambda signal: (signal.score, signal.ticker), reverse=True)
        liquid = [
            row
            for row in ranked
            if (as_float(row.feature.get("close")) or 0.0) >= MIN_OPTION_UNDERLYING_PRICE
            and (as_float(row.feature.get("avg30_volume")) or 0.0) >= MIN_OPTION_AVG30_VOLUME
        ]
        selected = liquid if max_per_date <= 0 else liquid[:max_per_date]
        selected.extend(
            row
            for row in rows
            if (
                row.ticker in TARGET_TICKERS
                or row.family in {"OVERSOLD_REBOUND_STRICT", "EARNINGS_VOLATILITY_EVENT"}
            )
            and (as_float(row.feature.get("close")) or 0.0) >= MIN_OPTION_UNDERLYING_PRICE
            and (as_float(row.feature.get("avg30_volume")) or 0.0) >= MIN_OPTION_AVG30_VOLUME
        )
        requests[signal_date] = {(row.ticker, row.direction) for row in selected}
    return requests


def option_rows_for_signals(
    signals: Sequence[PriceSignal],
    entry_quotes: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    quote_history: Mapping[str, Mapping[str, Dict[str, Any]]],
    dates: Sequence[str],
    as_of: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for signal in signals:
        if signal.date > as_of:
            continue
        entry = entry_quotes.get((signal.date, signal.ticker, signal.direction))
        if not entry:
            continue
        for variant in entry_variants(entry):
            for horizon in OPTION_HORIZONS:
                if int(variant.get("max_horizon") or max(OPTION_HORIZONS)) < horizon:
                    continue
                outcome = option_outcome(variant, quote_history, dates, horizon)
                status = option_outcome_status(
                    signal.date,
                    horizon,
                    dates,
                    outcome,
                    entry=variant,
                    quote_history=quote_history,
                )
                row = {
                    "signal_id": signal.signal_id,
                    "signal_date": signal.date,
                    "ticker": signal.ticker,
                    "direction": signal.direction,
                    "pattern_family": signal.family,
                    "signal_role": signal.role,
                    "score": signal.score,
                    "strategy": variant.get("strategy", "LONG_OPTION"),
                    "option_symbol": variant.get("option_symbol"),
                    "short_option_symbol": variant.get("short_option_symbol", ""),
                    "put_option_symbol": variant.get("put_option_symbol", ""),
                    "credit_short_option_symbol": variant.get("credit_short_option_symbol", ""),
                    "credit_long_option_symbol": variant.get("credit_long_option_symbol", ""),
                    "iron_short_call_option_symbol": variant.get("iron_short_call_option_symbol", ""),
                    "iron_long_call_option_symbol": variant.get("iron_long_call_option_symbol", ""),
                    "iron_short_put_option_symbol": variant.get("iron_short_put_option_symbol", ""),
                    "iron_long_put_option_symbol": variant.get("iron_long_put_option_symbol", ""),
                    "option_type": variant.get("option_type"),
                    "strike": variant.get("strike"),
                    "short_strike": variant.get("short_strike"),
                    "put_strike": variant.get("put_strike"),
                    "credit_short_strike": variant.get("credit_short_strike"),
                    "credit_long_strike": variant.get("credit_long_strike"),
                    "expiry": variant.get("expiry"),
                    "dte": variant.get("dte"),
                    "entry_bid": variant.get("bid"),
                    "entry_ask": variant.get("ask"),
                    "entry_debit": variant.get("entry_debit"),
                    "entry_credit": variant.get("entry_credit"),
                    "vertical_width": variant.get("vertical_width"),
                    "credit_width": variant.get("credit_width"),
                    "max_horizon": variant.get("max_horizon"),
                    "entry_spread_pct": variant.get("spread_pct"),
                    "entry_quote_source": variant.get("source"),
                    "entry_quote_source_date": variant.get("source_date"),
                    "entry_quote_provenance": variant.get("quote_provenance"),
                    "close": signal.feature.get("close"),
                    "avg30_volume": signal.feature.get("avg30_volume"),
                    "volume_ratio_30d": signal.feature.get("volume_ratio_30d"),
                    "return_1d": signal.feature.get("return_1d"),
                    "return_5d": signal.feature.get("return_5d"),
                    "return_20d": signal.feature.get("return_20d"),
                    "relative_strength_20d": signal.feature.get("relative_strength_20d"),
                    "sector_relative_strength_20d": signal.feature.get("sector_relative_strength_20d"),
                    "premium_bias": signal.feature.get("premium_bias"),
                    "uw_side_flow_bias": signal.feature.get("uw_side_flow_bias"),
                    "uw_side_volume_ratio": signal.feature.get("uw_side_volume_ratio"),
                    "uw_oi_bias": signal.feature.get("uw_oi_bias"),
                    "bot_eod_flow_bias": signal.feature.get("bot_eod_flow_bias"),
                    "bot_eod_flow_total_premium": signal.feature.get("bot_eod_flow_total_premium"),
                    "iv_rank": signal.feature.get("iv_rank"),
                    "implied_move_perc": signal.feature.get("implied_move_perc"),
                    "horizon": horizon,
                    "status": status,
                    "outcome_missing_reason": (
                        ""
                        if status == "SCORED"
                        else option_outcome_missing_reason(variant, quote_history, dates, horizon)
                    ),
                }
                if outcome:
                    row.update(outcome)
                rows.append(row)
    return rows


def stats_lookup(rows: Sequence[Mapping[str, Any]], value_key: str) -> Dict[Tuple[str, str, int, str], Dict[str, Any]]:
    return {
        (
            str(row["pattern_family"]),
            str(row["direction"]),
            int(row["horizon"]),
            str(row.get("strategy") or ""),
        ): dict(row)
        for row in grouped_stats(rows, value_key)
    }


def contract_label(entry: Mapping[str, Any]) -> str:
    strategy = str(entry.get("strategy") or "LONG_OPTION")
    if strategy in {
        "DEBIT_VERTICAL",
        "DEBIT_VERTICAL_SHORT_DTE",
        "DEBIT_VERTICAL_LONG_DTE",
    }:
        return f"{entry.get('option_symbol', '')} / {entry.get('short_option_symbol', '')}"
    if strategy in {"CREDIT_VERTICAL", "CREDIT_VERTICAL_SHORT_DTE"}:
        return f"{entry.get('credit_short_option_symbol', '')} / {entry.get('credit_long_option_symbol', '')}"
    if strategy == "LONG_STRADDLE":
        return f"{entry.get('option_symbol', '')} + {entry.get('put_option_symbol', '')}"
    if strategy == "IRON_CONDOR":
        return (
            f"{entry.get('iron_short_call_option_symbol', '')} / "
            f"{entry.get('iron_long_call_option_symbol', '')} | "
            f"{entry.get('iron_short_put_option_symbol', '')} / "
            f"{entry.get('iron_long_put_option_symbol', '')}"
        )
    return str(entry.get("option_symbol") or "")


def pattern_key(row: Mapping[str, Any]) -> Tuple[str, str, int, str]:
    return (
        str(row.get("pattern_family") or ""),
        str(row.get("direction") or ""),
        int(row.get("horizon") or 0),
        str(row.get("strategy") or ""),
    )


def option_gate_sets(
    option_stats: Sequence[Mapping[str, Any]],
    walk_forward_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    rolling_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    as_of: Optional[str] = None,
) -> Tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    """Apply the same evidence gates to reports and the current action board."""

    coverage_lookup = {pattern_key(row): row for row in coverage_rows}
    qualified: List[Mapping[str, Any]] = []
    for row in option_stats:
        n = int(row.get("sample_count") or 0)
        dates = int(row.get("unique_signal_dates") or 0)
        avg = as_float(row.get("average_value"))
        pf = as_float(row.get("profit_factor"))
        latest = as_float(row.get("latest_holdout_average"))
        lower = as_float(row.get("lower_mean_95"))
        date_lower = as_float(row.get("date_lower_mean_95"))
        date_dd = as_float(row.get("date_max_drawdown"))
        coverage = as_float(coverage_lookup.get(pattern_key(row), {}).get("coverage_ratio"))
        last_signal_date = str(row.get("last_signal_date") or "")
        stale = False
        if as_of:
            if not DATE_RE.fullmatch(last_signal_date):
                stale = True
            else:
                try:
                    stale = (parse_date(as_of) - parse_date(last_signal_date)).days > MAX_OPTION_PATTERN_STALENESS_DAYS
                except ValueError:
                    stale = True
        if (
            n >= 50
            and dates >= 20
            and avg is not None
            and avg > 0
            and pf is not None
            and pf >= 1.20
            and latest is not None
            and latest > 0
            and lower is not None
            and lower > 0
            and date_lower is not None
            and date_lower > 0
            and (date_dd is None or date_dd >= -8.0)
            and coverage is not None
            and coverage >= MIN_OPTION_OUTCOME_COVERAGE
            and not stale
        ):
            qualified.append(row)

    walk_groups: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in walk_forward_rows:
        walk_groups[pattern_key(row)].append(row)
    qualified_walk_forward: List[Mapping[str, Any]] = []
    for row in qualified:
        folds = sorted(walk_groups.get(pattern_key(row), []), key=lambda item: int(item.get("fold") or 0))
        positive_folds = sum((as_float(item.get("test_average_value")) or 0.0) > 0 for item in folds)
        latest_fold = folds[-1] if folds else {}
        if len(folds) >= 4 and positive_folds == len(folds) and (as_float(latest_fold.get("test_average_value")) or 0.0) > 0:
            qualified_walk_forward.append(row)

    rolling_groups: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rolling_rows:
        rolling_groups[pattern_key(row)].append(row)
    qualified_rolling: List[Mapping[str, Any]] = []
    for row in qualified_walk_forward:
        windows = sorted(rolling_groups.get(pattern_key(row), []), key=lambda item: int(item.get("fold") or 0))
        positive_windows = sum((as_float(item.get("test_average_value")) or 0.0) > 0 for item in windows)
        latest_window = windows[-1] if windows else {}
        if len(windows) >= 4 and positive_windows == len(windows) and (as_float(latest_window.get("test_average_value")) or 0.0) > 0:
            qualified_rolling.append(row)

    calibration_groups: Dict[Tuple[str, str, int, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in calibration_rows:
        calibration_groups[pattern_key(row)].append(row)
    calibrated: List[Mapping[str, Any]] = []
    for row in qualified:
        top_bin = min(calibration_groups.get(pattern_key(row), []), key=lambda item: int(item.get("score_bin") or 999), default={})
        if (
            int(top_bin.get("sample_count") or 0) >= 10
            and (as_float(top_bin.get("win_rate_lower_95")) or 0.0) >= 0.50
            and (as_float(top_bin.get("average_value")) or 0.0) > 0
        ):
            calibrated.append(row)
    return qualified, qualified_walk_forward, qualified_rolling, calibrated


def price_gate_sets(
    price_stats: Sequence[Mapping[str, Any]],
    walk_forward_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    rolling_rows: Sequence[Mapping[str, Any]],
) -> Tuple[
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
]:
    """Run the same statistical gates for underlying-price signals.

    Price rows have no quote-coverage problem, so their coverage is explicitly
    one rather than being silently omitted from the shared gate logic.

    Neutral rows are magnitude diagnostics (for example, an earnings move),
    not directional return strategies. They remain visible in validation
    output, but cannot qualify as a directional price pattern.
    """

    directional_stats = [
        row for row in price_stats
        if str(row.get("direction") or "") in {"bullish", "bearish"}
    ]
    directional_keys = {pattern_key(row) for row in directional_stats}
    directional_walk_forward = [
        row for row in walk_forward_rows if pattern_key(row) in directional_keys
    ]
    directional_calibration = [
        row for row in calibration_rows if pattern_key(row) in directional_keys
    ]
    directional_rolling = [
        row for row in rolling_rows if pattern_key(row) in directional_keys
    ]

    coverage_rows = [
        {
            **{key: row.get(key) for key in ("pattern_family", "direction", "horizon", "strategy")},
            "coverage_ratio": 1.0,
        }
        for row in directional_stats
    ]
    return option_gate_sets(
        directional_stats,
        directional_walk_forward,
        directional_calibration,
        directional_rolling,
        coverage_rows,
    )


def build_current_board(
    current_signals: Sequence[PriceSignal],
    price_rows: Sequence[Mapping[str, Any]],
    option_rows: Sequence[Mapping[str, Any]],
    as_of: str,
    current_entry_quotes: Mapping[Tuple[str, str, str], Mapping[str, Any]],
    approved_pattern_keys: Set[Tuple[str, str, int, str]],
    coverage_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    price_stats = stats_lookup(price_outcome_rows(price_rows), "stock_return")
    predictive_option_rows = [row for row in option_rows if is_predictive_option_row(row)]
    lane_option_rows = validation_lane_rows(
        [row for row in option_rows if row.get("signal_role") != "same_day_event"]
    )
    option_stats = stats_lookup(
        predictive_option_rows + lane_option_rows,
        "net_R",
    )
    coverage_lookup = {pattern_key(row): row for row in coverage_rows}
    board: List[Dict[str, Any]] = []
    ranked_current = sorted(
        current_signals,
        key=lambda item: (
            (as_float(item.feature.get("close")) or 0.0) >= MIN_OPTION_UNDERLYING_PRICE
            and (as_float(item.feature.get("avg30_volume")) or 0.0) >= MIN_OPTION_AVG30_VOLUME,
            item.score,
            item.ticker,
        ),
        reverse=True,
    )
    selected_current = ranked_current[:50]
    selected_ids = {signal.signal_id for signal in selected_current}
    for signal in current_signals:
        if signal.ticker in TARGET_TICKERS and signal.signal_id not in selected_ids:
            selected_current.append(signal)
            selected_ids.add(signal.signal_id)
    for signal in selected_current:
        entry_bundle = current_entry_quotes.get((as_of, signal.ticker, signal.direction), {})
        variants = entry_variants(entry_bundle)
        evidence: List[Tuple[Dict[str, Any], str, Tuple[str, str, int, str], Dict[str, Any]]] = []
        for variant in variants:
            strategy = str(variant.get("strategy") or "LONG_OPTION")
            candidate_base = {
                "pattern_family": signal.family,
                "direction": signal.direction,
                "strategy": strategy,
                "entry_spread_pct": variant.get("spread_pct"),
                "dte": variant.get("dte"),
                "volume_ratio_30d": signal.feature.get("volume_ratio_30d"),
                "bot_eod_flow_bias": signal.feature.get("bot_eod_flow_bias"),
                "bot_eod_flow_total_premium": signal.feature.get("bot_eod_flow_total_premium"),
                "implied_move_perc": signal.feature.get("implied_move_perc"),
            }
            matched_lane = False
            for lane_horizon in sorted({int(spec["horizon"]) for spec in VALIDATION_LANE_SPECS}):
                lane = validation_lane_for_row({**candidate_base, "horizon": lane_horizon})
                if not lane:
                    continue
                key = (f"LANE::{lane}", signal.direction, lane_horizon, strategy)
                stats = dict(option_stats.get(key, {}))
                evidence.append((dict(variant), lane, key, stats))
                matched_lane = True
            if not matched_lane:
                key = (signal.family, signal.direction, 5, strategy)
                stats = dict(option_stats.get(key, {}))
                evidence.append((dict(variant), "", key, stats))
        selected_evidence = max(
            evidence,
            key=lambda item: (
                item[2] in approved_pattern_keys,
                bool(item[1]),
                (as_float(item[3].get("average_value")) or -999.0) > 0,
                int(item[3].get("sample_count") or 0) >= 20,
                as_float(item[3].get("average_value")) or -999.0,
                int(item[3].get("sample_count") or 0),
            ),
            default=({}, "", (signal.family, signal.direction, 5, ""), {}),
        )
        entry, validation_lane, evidence_key, ostat = selected_evidence
        strategy = str(entry.get("strategy") or "LONG_OPTION")
        pkey = (signal.family, signal.direction, 5, "")
        pstat = price_stats.get(pkey, {})
        quote_available = bool(variants)
        blockers: List[str] = []
        if (as_float(signal.feature.get("close")) or 0.0) < MIN_OPTION_UNDERLYING_PRICE:
            blockers.append("UNDERLYING_PRICE_TOO_LOW_FOR_EXECUTION")
        if (as_float(signal.feature.get("avg30_volume")) or 0.0) < MIN_OPTION_AVG30_VOLUME:
            blockers.append("UNDERLYING_VOLUME_TOO_LOW_FOR_EXECUTION")
        if not quote_available:
            blockers.append("NO_CURRENT_HOT_CHAIN_CONTRACT")
        if signal.direction != "neutral":
            if not pstat or int(pstat.get("sample_count") or 0) < 20:
                blockers.append("PRICE_PATTERN_NOT_PROVEN")
            elif (as_float(pstat.get("average_value")) or 0.0) <= 0:
                blockers.append("PRICE_EXPECTANCY_NOT_POSITIVE")
        if not ostat or int(ostat.get("sample_count") or 0) < 20:
            blockers.append("OPTION_LANE_NOT_PROVEN" if validation_lane else "OPTION_PROFILE_NOT_PROVEN")
        elif (as_float(ostat.get("average_value")) or 0.0) <= 0:
            blockers.append("OPTION_NET_EV_NOT_POSITIVE")
        elif (as_float(ostat.get("profit_factor")) or 0.0) < 1.20:
            blockers.append("OPTION_PROFIT_FACTOR_BELOW_1_20")
        latest = as_float(ostat.get("latest_holdout_average"))
        if latest is not None and latest <= 0:
            blockers.append("OPTION_LATEST_HOLDOUT_NOT_POSITIVE")
        coverage = as_float(coverage_lookup.get(evidence_key, {}).get("coverage_ratio"))
        if ostat and (coverage is None or coverage < MIN_OPTION_OUTCOME_COVERAGE):
            blockers.append("OPTION_EXIT_COVERAGE_BELOW_80_PERCENT")
        if evidence_key not in approved_pattern_keys:
            blockers.append("OPTION_PATTERN_NOT_FULLY_VALIDATED")
        negative_option_evidence = bool(ostat) and (
            (as_float(ostat.get("average_value")) is not None and (as_float(ostat.get("average_value")) or 0.0) <= 0)
            or (as_float(ostat.get("profit_factor")) is not None and (as_float(ostat.get("profit_factor")) or 0.0) < 1.0)
        )
        status = (
            "APPROVED_TRADE"
            if not blockers
            else "REJECTED_CURRENT"
            if quote_available and negative_option_evidence
            else "TRADE_REVIEW"
            if quote_available
            else "RESEARCH_PATTERN"
        )
        if strategy in {"CREDIT_VERTICAL", "CREDIT_VERTICAL_SHORT_DTE", "IRON_CONDOR"}:
            entry_display = f"credit {entry.get('entry_credit')}"
        elif strategy in {
            "DEBIT_VERTICAL",
            "DEBIT_VERTICAL_SHORT_DTE",
            "DEBIT_VERTICAL_LONG_DTE",
            "LONG_STRADDLE",
        }:
            entry_display = f"debit {entry.get('entry_debit')}"
        else:
            entry_display = f"ask {entry.get('ask')}"
        board.append(
            {
                "status": status,
                "as_of": as_of,
                "ticker": signal.ticker,
                "direction": signal.direction,
                "pattern_family": signal.family,
                "validation_lane": validation_lane,
                "validation_key": "|".join(str(value) for value in evidence_key),
                "signal_role": signal.role,
                "score": signal.score,
                "pattern_reasons": "; ".join(signal.reasons),
                "contract": contract_label(entry),
                "option_symbol": entry.get("option_symbol", ""),
                "option_strategy": strategy,
                "short_option_symbol": entry.get("short_option_symbol", ""),
                "put_option_symbol": entry.get("put_option_symbol", ""),
                "credit_short_option_symbol": entry.get("credit_short_option_symbol", ""),
                "credit_long_option_symbol": entry.get("credit_long_option_symbol", ""),
                "iron_short_call_option_symbol": entry.get("iron_short_call_option_symbol", ""),
                "iron_long_call_option_symbol": entry.get("iron_long_call_option_symbol", ""),
                "iron_short_put_option_symbol": entry.get("iron_short_put_option_symbol", ""),
                "iron_long_put_option_symbol": entry.get("iron_long_put_option_symbol", ""),
                "entry_debit": entry.get("entry_debit"),
                "entry_credit": entry.get("entry_credit"),
                "entry_ask": entry.get("ask"),
                "entry_display": entry_display,
                "entry_spread_pct": entry.get("spread_pct"),
                "price_sample_count_5d": pstat.get("sample_count"),
                "price_average_return_5d": pstat.get("average_value"),
                "price_profit_factor_5d": pstat.get("profit_factor"),
                "option_sample_count_5d": ostat.get("sample_count"),
                "option_average_net_R_5d": ostat.get("average_value"),
                "option_profit_factor_5d": ostat.get("profit_factor"),
                "option_latest_holdout_average_5d": ostat.get("latest_holdout_average"),
                "option_validation_horizon": evidence_key[2],
                "option_exit_coverage": coverage,
                "blockers": ";".join(blockers),
            }
        )
    return board


def csv_write(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen: Set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or ["status"], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if value is not None})


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def render_report(
    as_of: str,
    metadata: Mapping[str, Any],
    board: Sequence[Mapping[str, Any]],
    price_stats: Sequence[Mapping[str, Any]],
    option_stats: Sequence[Mapping[str, Any]],
    mover_rows: Sequence[Mapping[str, Any]],
) -> str:
    approved = [row for row in board if row.get("status") == "APPROVED_TRADE"]
    review = [row for row in board if row.get("status") == "TRADE_REVIEW"]
    research_setups = [row for row in board if row.get("status") == "RESEARCH_SETUP"]
    rejected = [row for row in board if row.get("status") == "REJECTED_CURRENT"]
    action_rows = sorted(
        approved + review,
        key=lambda row: (
            0 if row.get("status") == "APPROVED_TRADE" else 1,
            -(as_float(row.get("score")) or 0.0),
            str(row.get("ticker") or ""),
        ),
    )

    def compact(value: Any, limit: int = 120) -> str:
        text_value = " ".join(str(value or "").replace("|", "/").split())
        return text_value if len(text_value) <= limit else text_value[: limit - 3] + "..."

    lines = [
        f"# Pattern Rebuild | {as_of}",
        "",
        "## Action Board",
        "",
        f"- Approved trades: **{len(approved)}**",
        f"- Review-only candidates with a current contract: **{len(review)}**",
        f"- Rejected current contracts: **{len(rejected)}**",
        f"- Current signals on board: **{len(board)}**",
        "",
        "| Status | Ticker | Direction | Pattern | Strategy | Contract legs | Entry | Historical validation | Primary blocker |",
        "|---|---|---|---|---|---|---:|---|---|",
    ]
    if not action_rows:
        lines.append("| NO_EXECUTION_CANDIDATE | - | - | - | - | - | - | No current contract cleared the evidence gates | See blockers below |")
    for row in action_rows[:20]:
        lane = str(row.get("validation_lane") or "")
        pattern = str(row.get("pattern_family") or "")
        if lane:
            pattern = f"{pattern} [{lane}]"
        validation = (
            f"N {row.get('option_sample_count_5d', '')}; "
            f"net R {row.get('option_average_net_R_5d', '')}; "
            f"PF {row.get('option_profit_factor_5d', '')}; "
            f"h{row.get('option_validation_horizon', '')}"
        )
        blockers = str(row.get("blockers") or "none").split(";")
        primary_blocker = "; ".join(item for item in blockers[:2] if item) or "none"
        lines.append(
            f"| {row.get('status')} | {row.get('ticker')} | {row.get('direction')} | {compact(pattern, 72)} | {row.get('option_strategy')} | {compact(row.get('contract'), 120)} | {row.get('entry_display') or row.get('entry_ask')} | {compact(validation, 92)} | {compact(primary_blocker, 100)} |"
        )
    research = sorted(
        [row for row in board if row.get("status") == "RESEARCH_PATTERN"],
        key=lambda row: (-(as_float(row.get("score")) or 0.0), str(row.get("ticker") or "")),
    )
    lines.extend(
        [
            "",
            "## Research-Only Current Signals",
            "",
            "These signals have no current hot-chain contract and are not trade candidates.",
            "",
            "| Ticker | Direction | Pattern | Score | Primary blocker |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in research[:20]:
        blockers = str(row.get("blockers") or "none").split(";")
        primary_blocker = "; ".join(item for item in blockers[:2] if item) or "none"
        lines.append(
            f"| {row.get('ticker')} | {row.get('direction')} | {compact(row.get('pattern_family'), 72)} | {row.get('score')} | {compact(primary_blocker, 100)} |"
        )
    lines.extend(
        [
            "",
            "## Rejected Current Contracts",
            "",
            "These contracts had a quote but negative or sub-1.0 historical option evidence; they are not trade candidates.",
            "",
            "| Ticker | Direction | Pattern | Strategy | Contract legs | Entry | Net R | PF | Primary blocker |",
            "|---|---|---|---|---|---|---:|---:|---|",
        ]
    )
    rejected_rows = sorted(
        rejected,
        key=lambda row: (as_float(row.get("option_average_net_R_5d")) or -999.0, str(row.get("ticker") or "")),
    )
    for row in rejected_rows[:20]:
        blockers = str(row.get("blockers") or "none").split(";")
        primary_blocker = "; ".join(item for item in blockers[:2] if item) or "none"
        lines.append(
            f"| {row.get('ticker')} | {row.get('direction')} | {compact(row.get('pattern_family'), 48)} | {row.get('option_strategy')} | {compact(row.get('contract'), 100)} | {row.get('entry_display') or row.get('entry_ask')} | {row.get('option_average_net_R_5d')} | {row.get('option_profit_factor_5d')} | {compact(primary_blocker, 100)} |"
        )
    lines.extend(
        [
            "",
            "## What The Rebuild Can Claim",
            "",
            "- Same-day event detection and forward pattern prediction are separate outputs.",
            "- A stock pattern is not discarded because an option quote is missing.",
            "- An option quote is not called a trade unless its historical net-R profile clears the gates below.",
            "- Historical price moves cannot prove that the move was predictable before it happened.",
            "",
            "## Directional Pattern Status",
            "",
            f"- Price patterns clearing pooled, walk-forward, rolling, and calibration gates: **{metadata.get('qualified_price_pattern_count')}**.",
            f"- Price patterns clearing walk-forward gates: **{metadata.get('qualified_price_walk_forward_pattern_count')}**; rolling gates: **{metadata.get('qualified_price_rolling_holdout_pattern_count')}**.",
            f"- Option implementations clearing all corresponding gates: **{metadata.get('qualified_rolling_holdout_pattern_count')}**.",
            "",
            "| Pattern | Direction | Horizon | N | Dates | Avg return | PF | Latest | Date lower |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metadata.get("qualified_price_patterns", [])[:20]:
        lines.append(
            f"| {row.get('pattern_family')} | {row.get('direction')} | {row.get('horizon')} | {row.get('sample_count')} | {row.get('unique_signal_dates')} | {row.get('average_value')} | {row.get('profit_factor')} | {row.get('latest_holdout_average')} | {row.get('date_lower_mean_95')} |"
        )
    lines.extend(
        [
            "",
            "## Directional Price Pattern Validation",
            "",
            "| Pattern | Direction | Horizon | N | Dates | Avg | PF | Latest | Date lower |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    directional_price_stats = [
        row for row in price_stats
        if str(row.get("direction") or "") in {"bullish", "bearish"}
    ]
    for row in sorted(directional_price_stats, key=lambda item: (item.get("average_value") is not None, item.get("average_value") or -99), reverse=True)[:30]:
        lines.append(
            f"| {row.get('pattern_family')} | {row.get('direction')} | {row.get('horizon')} | {row.get('sample_count')} | {row.get('unique_signal_dates')} | {row.get('average_value')} | {row.get('profit_factor')} | {row.get('latest_holdout_average')} | {row.get('date_lower_mean_95')} |"
        )
    magnitude_stats = [
        row for row in price_stats if str(row.get("direction") or "") == "neutral"
    ]
    lines.extend(
        [
            "",
            "## Volatility And Magnitude Diagnostics",
            "",
            "Neutral rows measure absolute forward movement. They are not directional trade signals and are excluded from price-pattern qualification.",
            "",
            "| Diagnostic | Horizon | N | Dates | Avg abs move | Latest abs move |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(magnitude_stats, key=lambda item: (item.get("average_value") is not None, item.get("average_value") or -99), reverse=True)[:15]:
        lines.append(
            f"| {row.get('pattern_family')} | {row.get('horizon')} | {row.get('sample_count')} | {row.get('unique_signal_dates')} | {row.get('average_value')} | {row.get('latest_holdout_average')} |"
        )
    lines.extend(
        [
            "",
            "## Option Implementation Validation",
            "",
            "| Pattern | Direction | Horizon | N | Dates | Avg net R | PF | Latest | Date lower |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(option_stats, key=lambda item: (item.get("average_value") is not None, item.get("average_value") or -99), reverse=True)[:30]:
        lines.append(
            f"| {row.get('pattern_family')} | {row.get('direction')} | {row.get('horizon')} | {row.get('sample_count')} | {row.get('unique_signal_dates')} | {row.get('average_value')} | {row.get('profit_factor')} | {row.get('latest_holdout_average')} | {row.get('date_lower_mean_95')} |"
        )
    target_rows = [row for row in mover_rows if row.get("ticker") in TARGET_TICKERS]
    lines.extend(
        [
            "",
            "## Named And Oil-Proxy Audit",
            "",
            "| Ticker | Signal date | Next-day move | Same-direction families | Same direction flagged | Next-day event detected | Reason |",
            "|---|---|---:|---|---|---|---|",
        ]
    )
    for row in target_rows[:40]:
        lines.append(
            f"| {row.get('ticker')} | {row.get('signal_date')} | {row.get('next_day_move')} | {row.get('same_direction_families')} | {row.get('price_signal_same_direction')} | {row.get('next_session_event_detected')} | {row.get('miss_reason')} |"
        )
    lines.extend(
        [
            "",
            "## Gate Status",
            "",
            f"- Pipeline version: {metadata.get('pipeline_version')}",
            f"- Price signals: {metadata.get('price_signal_count')}",
            f"- Signal roles: {metadata.get('forward_setup_signal_count')} forward setups, {metadata.get('same_day_event_signal_count')} same-day events, {metadata.get('confirmation_signal_count')} confirmations.",
            f"- Price walk-forward rows: {metadata.get('price_walk_forward_row_count')}; rolling holdout rows: {metadata.get('price_rolling_holdout_row_count')}; calibrated price patterns: {metadata.get('calibrated_price_pattern_count')}.",
            f"- Point-in-time price-basis adjustments detected: {metadata.get('price_adjustment_event_count')} events across {metadata.get('price_adjustment_ticker_count')} tickers; raw closes remain available in the signal file.",
            f"- Option outcomes scored: {metadata.get('option_scored_outcome_count')}",
            f"- Predictive option outcomes scored: {metadata.get('option_predictive_scored_outcome_count')}",
            f"- Option signal candidates with an entry quote: {metadata.get('option_candidate_signal_count')}; contract variants: {metadata.get('option_contract_variant_count')}; missing eligible exits: {metadata.get('option_missing_exit_outcome_count')}; pending future exits: {metadata.get('option_pending_future_outcome_count')}.",
            f"- Scored exit quote provenance: {metadata.get('option_scored_exit_provenance_counts')}",
            f"- Bot-EOD option quote rows retained for target exits: {metadata.get('bot_eod_option_quote_rows')}; cache rows seen: {metadata.get('bot_eod_option_quote_cache_rows_seen')}.",
            f"- Option outcome coverage groups: {metadata.get('option_outcome_coverage_rows')}; below 80%: {metadata.get('option_outcome_coverage_fail_count')}.",
            f"- Bot-EOD flow dates: {metadata.get('bot_eod_source_dates')} source dates, {metadata.get('bot_eod_cache_hits')} cache hits, {metadata.get('bot_eod_cache_built')} built, {metadata.get('bot_eod_cache_missing')} missing.",
            f"- Known-mover audit rows: {metadata.get('known_mover_audit_rows')}",
            f"- Chronological walk-forward test rows: {metadata.get('walk_forward_row_count')}",
            f"- Recent rolling holdout rows: {metadata.get('rolling_holdout_row_count')}",
            f"- Walk-forward-qualified option families: {metadata.get('qualified_walk_forward_pattern_count')}",
            f"- Confidence calibration rows: {metadata.get('confidence_calibration_row_count')}",
            f"- Calibrated option families: {metadata.get('calibrated_option_pattern_count')}",
            f"- Named-ticker significant (>=5%) next-day recall: {metadata.get('named_ticker_significant_forward_recall')}",
            f"- Named-ticker significant (>=5%) next-day event detection: {metadata.get('named_ticker_significant_event_detection')}",
            f"- Named-ticker same-day event detection is reported separately from forward recall.",
            f"- OIL ticker alias used for audit: {', '.join(ENERGY_PROXY_TICKERS)}.",
            f"- Acceptance status: **{metadata.get('acceptance_status')}**",
            f"- Blockers: {', '.join(metadata.get('acceptance_blockers') or []) or 'none'}",
            f"- Managed research lane: {metadata.get('managed_research_status')}; research patterns: {metadata.get('managed_research_pattern_count')}; qualified managed patterns: {metadata.get('managed_research_qualified_count')}; cache hits/misses: {metadata.get('managed_research_cache_hits')}/{metadata.get('managed_research_cache_misses')}. See `{metadata.get('managed_research_out_dir')}`.",
            "",
            "This report is research and backtest evidence. It does not place orders or guarantee future profitability.",
            "",
        ]
    )
    return "\n".join(lines)


def _managed_contract_display(row: Mapping[str, Any]) -> str:
    """Render a managed position as a readable ticket, not an OCC-only id."""

    structure = str(row.get("structure") or "long_option")
    option_type = str(row.get("option_type") or "").upper()
    expiry = str(row.get("expiry") or "")
    strike = row.get("strike")
    option_symbol = str(row.get("option_symbol") or "")
    if structure == "long_option":
        return f"BUY {option_type} {strike} exp {expiry} @ ask {row.get('entry_ask')} ({option_symbol})"
    if structure == "long_straddle":
        second_symbol = str(row.get("second_option_symbol") or "")
        return (
            f"BUY C {strike} + BUY P {row.get('put_strike')} exp {expiry} "
            f"@ debit {row.get('entry_ask')} ({option_symbol} / {second_symbol})"
        )
    if structure == "iron_condor":
        short_put_symbol = str(row.get("short_option_symbol") or "")
        short_call_symbol = str(row.get("second_option_symbol") or "")
        long_call_symbol = str(row.get("long_call_option_symbol") or "")
        return (
            f"SELL P {row.get('short_strike')} / BUY P {strike} + "
            f"SELL C {row.get('short_call_strike')} / BUY C {row.get('long_call_strike')} "
            f"exp {expiry} @ credit {row.get('entry_credit') or row.get('entry_ask')} "
            f"({short_put_symbol} / {option_symbol} / {short_call_symbol} / {long_call_symbol})"
        )
    if structure == "cash_secured_put":
        return (
            f"SELL P {strike} exp {expiry} @ credit "
            f"{row.get('entry_credit') or row.get('entry_bid')} "
            f"(cash collateral {row.get('cash_collateral')}; "
            f"max loss to zero {row.get('max_loss_to_zero')}; {option_symbol})"
        )
    short_symbol = str(row.get("short_option_symbol") or "")
    short_strike = row.get("short_strike")
    if structure == "credit_vertical":
        return (
            f"SELL {option_type} {short_strike} / BUY {option_type} {strike} exp {expiry} "
            f"@ credit {row.get('entry_ask')} ({short_symbol} / {option_symbol})"
        )
    return (
        f"BUY {option_type} {strike} / SELL {option_type} {short_strike} exp {expiry} "
        f"@ debit {row.get('entry_ask')} ({option_symbol} / {short_symbol})"
    )


def _managed_directional_board(
    price_outcomes: Sequence[Mapping[str, Any]],
    price_pattern_validation: Sequence[Mapping[str, Any]],
    as_of: str,
    limit: int = 25,
) -> List[Dict[str, Any]]:
    """Render current proven stock setups without pretending they are options."""

    qualified = {
        (
            str(row.get("strategy_key") or ""),
            str(row.get("direction") or ""),
            int(row.get("horizon") or 0),
        ): row
        for row in price_pattern_validation
        if row.get("status") == "QUALIFIED_DIRECTIONAL"
    }
    if not qualified:
        return []
    candidates = [
        row
        for row in price_outcomes
        if str(row.get("signal_date") or "") <= as_of
        and (
            str(row.get("strategy_key") or ""),
            str(row.get("direction") or ""),
            int(row.get("horizon") or 0),
        ) in qualified
    ]
    if not candidates:
        return []
    latest_by_key = {
        key: max(
            str(row.get("signal_date") or "")
            for row in candidates
            if (
                str(row.get("strategy_key") or ""),
                str(row.get("direction") or ""),
                int(row.get("horizon") or 0),
            )
            == key
        )
        for key in qualified
        if any(
            (
                str(row.get("strategy_key") or ""),
                str(row.get("direction") or ""),
                int(row.get("horizon") or 0),
            )
            == key
            for row in candidates
        )
    }
    current = [
        row
        for row in candidates
        if str(row.get("signal_date") or "")
        == latest_by_key.get(
            (
                str(row.get("strategy_key") or ""),
                str(row.get("direction") or ""),
                int(row.get("horizon") or 0),
            )
        )
    ]
    current.sort(
        key=lambda row: (
            -(as_float(row.get("flow_bias")) or -999.0),
            -(as_float(row.get("position_52w")) or -999.0),
            str(row.get("ticker") or ""),
        )
    )
    board: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for row in current:
        pattern_key = (
            str(row.get("strategy_key") or ""),
            str(row.get("direction") or ""),
            int(row.get("horizon") or 0),
        )
        dedupe_key = (pattern_key[0], str(row.get("ticker") or ""))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        evidence = qualified[pattern_key]
        board.append(
            {
                "action": "STOCK_RESEARCH_ONLY",
                "status": "RESEARCH_STOCK_ONLY",
                "ticker": row.get("ticker"),
                "direction": row.get("direction"),
                "pattern_family": row.get("strategy_key"),
                "horizon": row.get("horizon"),
                "signal_date": row.get("signal_date"),
                "entry_rule": "next available stock session; no option ticket approved",
                "position_52w": row.get("position_52w"),
                "flow_bias": row.get("flow_bias"),
                "train_average_directional_return": evidence.get(
                    "train_average_directional_return"
                ),
                "validation_average_directional_return": evidence.get(
                    "validation_average_directional_return"
                ),
                "holdout_average_directional_return": evidence.get(
                    "holdout_average_directional_return"
                ),
                "holdout_profit_factor": evidence.get("holdout_profit_factor"),
                "blockers": "NO_PROVEN_OPTION_IMPLEMENTATION",
                "approval_status": "STOCK_RESEARCH_ONLY",
            }
        )
        if len(board) >= limit:
            break
    return board


def _managed_primary_report(
    as_of: str,
    metadata: Mapping[str, Any],
    board: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    scorecard: Sequence[Mapping[str, Any]],
    mover_rows: Sequence[Mapping[str, Any]],
    price_validation: Sequence[Mapping[str, Any]],
    price_pattern_validation: Sequence[Mapping[str, Any]],
    directional_board: Sequence[Mapping[str, Any]],
) -> str:
    """Render the rebuilt managed lane as the primary user-facing report."""

    def compact_blockers(value: Any) -> str:
        tokens = [token for token in str(value or "").split(";") if token]
        if not tokens:
            return "none"
        if len(tokens) <= 2:
            return "; ".join(tokens)
        groups = sorted({token.split("_", 1)[0] for token in tokens})
        return (
            f"NOT_APPROVED; {len(tokens)} gate checks ({'/'.join(groups)}); "
            "see managed_research_validation.csv"
        )

    approved = [row for row in board if row.get("status") == "APPROVED_TRADE"]
    historical = [
        row for row in board if row.get("status") == "HISTORICAL_APPROVED_TRADE"
    ]
    review = [row for row in board if row.get("status") == "TRADE_REVIEW"]
    research_setups = [row for row in board if row.get("status") == "RESEARCH_SETUP"]
    # The live board is execution-only.  Historical validated rows and
    # review candidates remain in CSV artifacts and are rendered separately,
    # so a strong backtest row cannot visually masquerade as a live ticket.
    report_rows = sorted(
        approved + review + research_setups,
        key=lambda row: (
            0
            if row.get("status") == "APPROVED_TRADE"
            else 1
            if row.get("status") == "TRADE_REVIEW"
            else 2,
            -(as_float(row.get("holdout_average_net_R")) or -999.0),
            str(row.get("ticker") or ""),
        ),
    )
    lines = [
        f"# Pattern Analysis V2 | {as_of}",
        "",
        "## Action Board",
        "",
        f"- Approved current trades: **{len(approved)}**",
        f"- Historical validated references: **{len(historical)}** (backtest-only; not executable)",
        f"- Conditional current trade setups: **{len(review)}** (next-session live quote recheck required)",
        f"- Research-only current option setups: **{len(research_setups)}**",
        "- Confirmed orders: **0** (this pipeline never places orders)",
        "",
        "| Action | Status | Ticker | Direction | Pattern | Contract | Signal date | Entry date | Holdout avg R | Holdout PF | Blockers |",
        "|---|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    if not report_rows:
        lines.append(
            "| DO_NOT_TRADE | NO_CURRENT_OPTION_SETUP | - | - | - | - | - | - | - | - | No exact current option setup was generated |"
        )
    for row in report_rows[:30]:
        lines.append(
            "| {action} | {status} | {ticker} | {direction} | {pattern} | {contract} | {signal_date} | {entry_date} | {holdout_average_net_R} | {holdout_profit_factor} | {blockers} |".format(
                action=row.get("action") or "DO_NOT_TRADE",
                status=row.get("status"),
                ticker=row.get("ticker"),
                direction=row.get("direction"),
                pattern=row.get("pattern_family"),
                contract=" ".join(str(row.get("contract") or "").split()).replace("|", "/"),
                signal_date=row.get("signal_date"),
                entry_date=row.get("entry_date"),
                holdout_average_net_R=row.get("holdout_average_net_R"),
                holdout_profit_factor=row.get("holdout_profit_factor"),
                blockers=compact_blockers(row.get("blockers")),
            )
        )
    lines.extend(
        [
            "",
            "## Historical Backtest References",
            "",
            "These rows passed historical validation but are not orders. Their entry dates are already past the report as-of date.",
            "",
            "| Status | Ticker | Pattern | Contract | Signal date | Entry date | Holdout avg R | Holdout PF | Action |",
            "|---|---|---|---|---|---|---:|---:|---|",
        ]
    )
    if not historical:
        lines.append("| NONE | - | - | - | - | - | - | - | NONE |")
    for row in historical:
        lines.append(
            "| HISTORICAL_ONLY | {ticker} | {pattern} | {contract} | {signal_date} | {entry_date} | {holdout_average_net_R} | {holdout_profit_factor} | DO_NOT_TRADE |".format(
                ticker=row.get("ticker"),
                pattern=row.get("pattern_family"),
                contract=" ".join(str(row.get("contract") or "").split()).replace("|", "/"),
                signal_date=row.get("signal_date"),
                entry_date=row.get("entry_date"),
                holdout_average_net_R=row.get("holdout_average_net_R"),
                holdout_profit_factor=row.get("holdout_profit_factor"),
            )
        )
    lines.extend(
        [
            "",
            "## Proven Directional Patterns",
            "",
            "These stock-direction hypotheses clear the fixed chronological gates. They are not option approvals.",
            "",
            "| Strategy | Direction | Horizon | Train avg | Validation avg | Holdout avg | Holdout PF | Holdout date lower | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    qualified_directional = [
        row
        for row in price_pattern_validation
        if row.get("status") == "QUALIFIED_DIRECTIONAL"
    ]
    if not qualified_directional:
        lines.append(
            "| - | - | - | - | - | - | - | - | No directional family cleared all stock gates |"
        )
    for row in qualified_directional:
        lines.append(
            "| {strategy_key} | {direction} | {horizon} | {train_average_directional_return} | {validation_average_directional_return} | {holdout_average_directional_return} | {holdout_profit_factor} | {holdout_date_lower_mean_95} | QUALIFIED_DIRECTIONAL |".format(
                **{
                    key: str(row.get(key, "")).replace("|", "/")
                    for key in (
                        "strategy_key",
                        "direction",
                        "horizon",
                        "train_average_directional_return",
                        "validation_average_directional_return",
                        "holdout_average_directional_return",
                        "holdout_profit_factor",
                        "holdout_date_lower_mean_95",
                    )
                }
            )
        )
    lines.extend(
        [
            "",
            "## Current Directional Research Setups",
            "",
            "These names are selected by a qualified stock pattern. They have no approved option structure or execution instruction.",
            "",
            "| Action | Status | Ticker | Direction | Pattern | Signal date | Horizon | 52-week position | Flow bias | Blocker |",
            "|---|---|---|---|---|---|---:|---:|---:|---|",
        ]
    )
    if not directional_board:
        lines.append(
            "| STOCK_RESEARCH_ONLY | RESEARCH_STOCK_ONLY | - | - | - | - | - | - | - | No current qualified directional setup |"
        )
    for row in directional_board:
        lines.append(
            "| {action} | {status} | {ticker} | {direction} | {pattern_family} | {signal_date} | {horizon} | {position_52w} | {flow_bias} | {blockers} |".format(
                **{
                    key: str(row.get(key, "")).replace("|", "/")
                    for key in (
                        "action",
                        "status",
                        "ticker",
                        "direction",
                        "pattern_family",
                        "signal_date",
                        "horizon",
                        "position_52w",
                        "flow_bias",
                        "blockers",
                    )
                }
            )
        )
    lines.extend(
        [
            "",
            "## What This Run Tests",
            "",
            "- Signals use point-in-time stock-screener fields and only backward-looking return/flow ranks.",
            "- Entry is the next available session at the quoted ask; exits use later dated bids.",
            "- Fees are charged on every leg; exact quotes or a bounded one-session last-observed quote are required to score exits, and longer gaps remain unscored.",
            "- TRAIN, VALIDATION, and HOLDOUT are chronological and are evaluated separately.",
            "- Production selection must add positive R versus a matched same-date/sector random control in train, validation, and untouched holdout.",
            "- Calibration is prior-only and date-grouped: each signal date receives a Beta-smoothed confidence from earlier dates, then the high-confidence bin must clear sample, 95% lower-bound, and positive-net-R gates in every split.",
            "- A recent positive holdout alone is research evidence, not permission to trade.",
            "",
            "## Underlying Directional Research",
            "",
            "This is stock-return evidence before option spread, bid/ask, fee, and exit-quote effects.",
            "",
            "| Strategy | Direction | Horizon | Sample | N | Avg directional return | PF | Lower 95 | Date lower 95 |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        price_validation,
        key=lambda item: (
            str(item.get("sample") or "") == "HOLDOUT",
            as_float(item.get("average_directional_return")) or -999.0,
        ),
        reverse=True,
    )[:30]:
        lines.append(
            "| {strategy_key} | {direction} | {horizon} | {sample} | {scored_count} | {average_directional_return} | {profit_factor} | {lower_mean_95} | {date_lower_mean_95} |".format(
                **{
                    key: str(row.get(key, "")).replace("|", "/")
                    for key in (
                        "strategy_key",
                        "direction",
                        "horizon",
                        "sample",
                        "scored_count",
                        "average_directional_return",
                        "profit_factor",
                        "lower_mean_95",
                        "date_lower_mean_95",
                    )
                }
            )
        )
    lines.extend(
        [
            "",
            "## Managed Validation",
            "",
            "| Strategy | Research status | Production status | Train avg R | Validation avg R | Holdout avg R | Holdout PF | Holdout cal. lower | Holdout dates | Holdout coverage | Blockers |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in validation:
        lines.append(
            "| {strategy_key} | {status} | {production_status} | {train_average_net_R} | {validation_average_net_R} | {holdout_average_net_R} | {holdout_profit_factor} | {holdout_calibration_score} | {holdout_unique_signal_dates} | {holdout_coverage} | {blockers} |".format(
                **{
                    key: str(row.get(key, "")).replace("|", "/")
                    for key in (
                        "strategy_key",
                        "status",
                        "production_status",
                        "train_average_net_R",
                        "validation_average_net_R",
                        "holdout_average_net_R",
                        "holdout_profit_factor",
                        "holdout_calibration_score",
                        "holdout_unique_signal_dates",
                        "holdout_coverage",
                        "blockers",
                    )
                }
            )
        )
    lines.extend(
        [
            "",
            "## Named Mover Audit",
            "",
            "Same-day event detection is reported separately from pre-event directional recall.",
            "",
            "| Ticker | Group | Event date | Move | Next-session move | Pre-event same-direction | Post-event same-direction | Result |",
            "|---|---|---|---:|---:|---|---|---|",
        ]
    )
    recent_movers = sorted(
        mover_rows,
        key=lambda row: (str(row.get("event_date") or ""), abs(as_float(row.get("event_return_1d")) or 0.0)),
        reverse=True,
    )
    for row in recent_movers[:40]:
        lines.append(
            "| {ticker} | {thematic_group} | {event_date} | {event_return_1d} | {next_session_return_1d} | {pre_event_same_direction_families} | {post_event_same_direction_families} | {miss_reason} / {post_event_followup_reason} |".format(
                **{
                    key: str(row.get(key, "")).replace("|", "/")
                    for key in (
                        "ticker",
                        "thematic_group",
                        "event_date",
                        "event_return_1d",
                        "next_session_return_1d",
                        "pre_event_same_direction_families",
                        "post_event_same_direction_families",
                        "miss_reason",
                        "post_event_followup_reason",
                    )
                }
            )
        )
    lines.extend(
        [
            "",
            "## Gate Status",
            "",
            f"- Pipeline version: {metadata.get('pipeline_version')}",
            f"- Source sessions: {metadata.get('session_count')} from {metadata.get('first_session')} through {metadata.get('last_session')}",
            f"- Stock-screener rows: {metadata.get('panel_metadata', {}).get('stock_screener_rows')}",
            f"- Quote source files: {metadata.get('quote_metadata', {}).get('quote_source_file_count')}; hot-chain files: {metadata.get('quote_metadata', {}).get('quote_source_kind_counts', {}).get('hot')}; chain-OI files: {metadata.get('quote_metadata', {}).get('quote_source_kind_counts', {}).get('chain_oi')}",
            f"- Materialized quote-cache hits/misses: {metadata.get('cache_hits')}/{metadata.get('cache_misses')}",
            f"- Supplemental bot-EOD flow: {(metadata.get('bot_flow_metadata') or {}).get('bot_flow_status')}; source files: {(metadata.get('bot_flow_metadata') or {}).get('bot_flow_source_files', 0)}; source dates: {len((metadata.get('bot_flow_metadata') or {}).get('bot_flow_source_dates') or [])}; ticker rows: {(metadata.get('bot_flow_metadata') or {}).get('bot_flow_rows', 0)}; excluded from approval until full split coverage exists.",
            f"- Managed exit reasons: {metadata.get('managed_exit_reason_counts')}; one-session stale exits scored: {metadata.get('managed_stale_exit_count', 0)}; longer quote gaps remain unscored.",
            f"- Managed structures: {metadata.get('managed_structure_counts')}",
            f"- Qualified managed strategies: {metadata.get('qualified_managed_count')}; research-only strategies: {metadata.get('research_pattern_count')}",
            f"- Production-qualified managed strategies after selection/control gates: {metadata.get('production_qualified_count', 0)}",
            f"- Predeclared selection audit: {metadata.get('selection_audit_status')}; blockers: {metadata.get('selection_audit_blockers') or []}; candidates: {metadata.get('selection_candidate_count')}; eligible before holdout: {metadata.get('selection_eligible_candidate_count')}; selected: {metadata.get('selected_candidate_key') or 'none'}; selection used holdout: {metadata.get('selection_holdout_used_for_selection')}",
            f"- Untouched selected holdout: {metadata.get('selected_final_holdout_pass')}; holdout start: {metadata.get('selection_final_holdout_start')}; regime rows: {metadata.get('selection_regime_rows')}",
            f"- Acceptance status: **{metadata.get('acceptance_status')}**",
            f"- Acceptance blockers: {', '.join(metadata.get('acceptance_blockers') or []) or 'none'}",
            "",
            "This report is research and backtest evidence. It does not place orders or guarantee future profitability.",
            "",
        ]
    )
    return "\n".join(lines)


def _managed_current_rows(
    trade_rows: Sequence[Mapping[str, Any]],
    strategy_status: Mapping[str, Mapping[str, Any]],
    as_of: str,
    production_strategy_keys: Optional[Set[str]] = None,
) -> Tuple[str, List[Dict[str, Any]], int]:
    """Keep qualified open positions visible even when another lane is newer."""

    def date_value(value: Any) -> str:
        text = str(value or "")[:10]
        return text if DATE_RE.fullmatch(text) else ""

    signal_rows = [
        dict(row)
        for row in trade_rows
        if str(row.get("control") or "") == "signal"
        and str(row.get("status") or "") == "PENDING_FUTURE"
        and date_value(row.get("signal_date")) <= as_of
        and date_value(row.get("entry_date")) <= as_of
    ]
    latest_signal_date = max(
        (date_value(row.get("signal_date")) for row in signal_rows),
        default="",
    )
    review_rows = [
        row
        for row in signal_rows
        if date_value(row.get("signal_date")) == latest_signal_date
    ]
    qualified_rows: List[Dict[str, Any]] = []
    for strategy_key, evidence in strategy_status.items():
        if str(evidence.get("status") or "") != "QUALIFIED_MANAGED":
            continue
        if (
            production_strategy_keys is not None
            and strategy_key not in production_strategy_keys
        ):
            continue
        strategy_rows = [
            row
            for row in signal_rows
            if str(row.get("pattern_family") or "") == strategy_key
        ]
        if not strategy_rows:
            continue
        latest_strategy_date = max(date_value(row.get("signal_date")) for row in strategy_rows)
        qualified_rows.extend(
            row
            for row in strategy_rows
            if date_value(row.get("signal_date")) == latest_strategy_date
        )

    current_rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()
    for row in qualified_rows + review_rows:
        key = (
            str(row.get("pattern_family") or ""),
            str(row.get("ticker") or ""),
            date_value(row.get("signal_date")),
            str(row.get("option_symbol") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        current_rows.append(row)
    return latest_signal_date, current_rows, len(review_rows)


def _managed_ticket_state(
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    as_of: str,
) -> Tuple[str, str, str, str]:
    """Separate fresh execution candidates from historical validation rows."""

    is_qualified = evidence.get("production_status") == "PRODUCTION_QUALIFIED"
    if row.get("candidate_timing") == "SAME_DAY_EOD_RESEARCH":
        if is_qualified:
            return (
                "TRADE_REVIEW",
                "RECHECK_NEXT_SESSION_QUOTE",
                "VALIDATED_STRATEGY_CONDITIONAL_ENTRY",
                "NEXT_SESSION_FILL_AND_LIVE_QUOTE_NOT_VALIDATED",
            )
        return (
            "RESEARCH_SETUP",
            "DO_NOT_TRADE",
            "RESEARCH_ONLY",
            evidence.get("production_blockers")
            or "PREDECLARED_SELECTION_AND_MATCHED_CONTROL_GATE_NOT_PASSED",
        )
    entry_date = str(row.get("entry_date") or "")[:10]
    fresh_entry = DATE_RE.fullmatch(entry_date) and entry_date > as_of
    if is_qualified and fresh_entry:
        return (
            "APPROVED_TRADE",
            "EXECUTE",
            evidence.get("approval_status") or "QUALIFIED_MANAGED",
            evidence.get("blockers") or "none",
        )
    if is_qualified:
        return (
            "HISTORICAL_APPROVED_TRADE",
            "DO_NOT_TRADE",
            "HISTORICAL_VALIDATED",
            "ENTRY_DATE_PAST_AS_OF",
        )
    return (
        "HISTORICAL_RESEARCH",
        "DO_NOT_TRADE",
        evidence.get("approval_status") or "NOT_APPROVED",
        evidence.get("blockers") or "none",
    )


def _cap_managed_action_board(
    rows: Sequence[Mapping[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Keep current setups ahead of historical references when capping output."""

    live = [
        dict(row)
        for row in rows
        if row.get("status") in {"APPROVED_TRADE", "TRADE_REVIEW", "RESEARCH_SETUP"}
    ][:limit]
    remaining = max(0, limit - len(live))
    historical_approved = [
        dict(row)
        for row in rows
        if row.get("status") == "HISTORICAL_APPROVED_TRADE"
    ][:remaining]
    remaining -= len(historical_approved)
    historical_research = [
        dict(row)
        for row in rows
        if row.get("status") == "HISTORICAL_RESEARCH"
    ][:remaining]
    return live + historical_approved + historical_research


def run_managed_primary_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the first-principles managed lane without the legacy bot replay."""

    base_dir = Path(args.base_dir).expanduser().resolve()
    requested_as_of = str(args.as_of)
    if requested_as_of == "latest":
        candidates = date_dirs(base_dir, args.start_date, None)
        if not candidates:
            raise SystemExit(f"No dated source folders found under {base_dir}")
        requested_as_of = candidates[-1]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else base_dir / "out" / "pattern_analysis_v2" / requested_as_of
    )
    managed_research_dir = out_dir / "managed_research"
    from .managed import run_managed_research

    cache_root = (
        Path(args.cache_dir).expanduser().resolve()
        if getattr(args, "cache_dir", None)
        else base_dir / "out" / "pattern_analysis_v2" / "cache" / "managed_quotes"
    )
    managed_result = run_managed_research(
        base_dir,
        args.start_date,
        requested_as_of,
        managed_research_dir,
        cache_root=cache_root,
    )
    managed_metadata = dict(managed_result.get("metadata") or {})
    actual_as_of = str(managed_metadata.get("last_session") or requested_as_of)
    managed_metadata["pipeline_version"] = PIPELINE_VERSION
    managed_metadata["requested_as_of"] = requested_as_of
    managed_metadata["actual_as_of"] = actual_as_of

    validation = [dict(row) for row in managed_result.get("validation") or []]
    scorecard = [dict(row) for row in managed_result.get("scorecard") or []]
    mover_rows = [dict(row) for row in managed_result.get("named_mover_audit") or []]
    price_outcomes = [dict(row) for row in managed_result.get("price_outcomes") or []]
    price_validation = [dict(row) for row in managed_result.get("price_validation") or []]
    price_pattern_validation = [
        dict(row) for row in managed_result.get("price_pattern_validation") or []
    ]
    calibration_rows = [
        dict(row) for row in managed_result.get("calibration_rows") or []
    ]
    current_setup_rows = [
        dict(row) for row in managed_result.get("current_setups") or []
    ]
    directional_board = _managed_directional_board(
        price_outcomes,
        price_pattern_validation,
        actual_as_of,
    )
    strategy_status = {
        str(row.get("strategy_key") or ""): row for row in validation
    }
    production_strategy_keys = {
        key
        for key, evidence in strategy_status.items()
        if evidence.get("production_status") == "PRODUCTION_QUALIFIED"
    }
    trades_path = managed_research_dir / "managed_research_trades.csv"
    try:
        import pandas as pd

        # The combined managed artifact intentionally contains optional leg
        # symbols, so pandas must infer the full file in one pass.
        trades_frame = (
            pd.read_csv(trades_path, low_memory=False)
            if trades_path.exists()
            else pd.DataFrame()
        )
        trade_rows = trades_frame.to_dict("records")
    except (OSError, ValueError, ImportError):
        trade_rows = []

    latest_signal_date, current_rows, current_review_count = _managed_current_rows(
        trade_rows,
        strategy_status,
        actual_as_of,
        production_strategy_keys,
    )
    current_rows = current_setup_rows + current_rows
    if current_setup_rows:
        latest_signal_date = actual_as_of
    board: List[Dict[str, Any]] = []
    for row in current_rows:
        strategy_key = str(row.get("pattern_family") or "")
        evidence = strategy_status.get(strategy_key, {})
        ticket_status, ticket_action, ticket_approval, ticket_blockers = _managed_ticket_state(
            row,
            evidence,
            actual_as_of,
        )
        board.append(
            {
                "status": ticket_status,
                "action": ticket_action,
                "ticker": row.get("ticker"),
                "direction": row.get("direction"),
                "pattern_family": strategy_key,
                "structure": row.get("structure"),
                "contract": _managed_contract_display(row),
                "option_symbol": row.get("option_symbol"),
                "short_option_symbol": row.get("short_option_symbol"),
                "second_option_symbol": row.get("second_option_symbol"),
                "long_call_option_symbol": row.get("long_call_option_symbol"),
                "strike": row.get("strike"),
                "short_strike": row.get("short_strike"),
                "put_strike": row.get("put_strike"),
                "short_call_strike": row.get("short_call_strike"),
                "long_call_strike": row.get("long_call_strike"),
                "signal_date": row.get("signal_date"),
                "entry_date": row.get("entry_date"),
                "entry_ask": row.get("entry_ask"),
                "entry_bid": row.get("entry_bid"),
                "entry_credit": row.get("entry_credit"),
                "collateral_per_share": row.get("collateral_per_share"),
                "cash_collateral": row.get("cash_collateral"),
                "max_loss_to_zero": row.get("max_loss_to_zero"),
                "entry_spread_pct": row.get("entry_spread_pct"),
                "entry_oi": row.get("entry_oi"),
                "holdout_average_net_R": evidence.get("holdout_average_net_R"),
                "holdout_profit_factor": evidence.get("holdout_profit_factor"),
                "blockers": ticket_blockers,
                "approval_status": ticket_approval,
                "candidate_timing": row.get("candidate_timing"),
            }
        )
    action_board_limit = 25
    board.sort(
        key=lambda row: (
            0 if row.get("status") == "APPROVED_TRADE" else 1
            if row.get("status") == "TRADE_REVIEW"
            else 2
            if row.get("status") == "HISTORICAL_APPROVED_TRADE"
            else 3,
            -(as_float(row.get("holdout_average_net_R")) or -999.0),
            -(as_float(row.get("holdout_profit_factor")) or -999.0),
            str(row.get("pattern_family") or ""),
            str(row.get("ticker") or ""),
        )
    )
    board = _cap_managed_action_board(board, action_board_limit)
    if not board:
        board = [
            {
                "status": "NO_EXECUTION_CANDIDATE",
                "action": "DO_NOT_TRADE",
                "ticker": "",
                "direction": "",
                "pattern_family": "",
                "contract": "",
                "signal_date": latest_signal_date,
                "entry_date": "",
                "blockers": "NO_CURRENT_MANAGED_POSITION",
                "approval_status": "NOT_APPROVED",
            }
        ]

    acceptance_blockers: List[str] = []
    if not production_strategy_keys:
        acceptance_blockers.append("NO_MANAGED_STRATEGY_CLEARS_TRAIN_VALIDATION_HOLDOUT_GATES")
    selection_status = str(managed_metadata.get("selection_audit_status") or "")
    if selection_status != "PASS":
        acceptance_blockers.append("PREDECLARED_SELECTION_AUDIT_NOT_PASSED")
    selected_candidate_key = str(managed_metadata.get("selected_candidate_key") or "")
    if selected_candidate_key and not any(
        str(row.get("strategy_key") or "") == selected_candidate_key
        and row.get("status") == "QUALIFIED_MANAGED"
        for row in validation
    ):
        acceptance_blockers.append("SELECTED_CANDIDATE_NOT_THE_QUALIFIED_PRODUCTION_LANE")
    if not any(
        row.get("status") in {"APPROVED_TRADE", "HISTORICAL_APPROVED_TRADE"}
        for row in board
    ):
        acceptance_blockers.append("NO_VALIDATED_MANAGED_ENTRY")
    acceptance_status = "PASS" if not acceptance_blockers else "FAIL_REQUIREMENTS_REMAIN"
    managed_metadata.update(
        {
            "pipeline_version": PIPELINE_VERSION,
            "primary_lane": "managed",
            "acceptance_status": acceptance_status,
            "acceptance_blockers": acceptance_blockers,
            "current_signal_date": latest_signal_date,
            "current_managed_position_count": len(current_rows),
            "current_review_position_count": current_review_count,
            "current_execution_candidate_count": sum(
                row.get("status") == "APPROVED_TRADE" for row in board
            ),
            "current_option_setup_count": len(current_setup_rows),
            "historical_validated_entry_count": sum(
                row.get("status") == "HISTORICAL_APPROVED_TRADE"
                for row in board
            ),
            "current_directional_signal_date": (
                directional_board[0].get("signal_date") if directional_board else ""
            ),
            "current_directional_candidate_count": len(directional_board),
            "action_board_display_count": len(board),
            "action_board_limit": action_board_limit,
            "managed_research_calibration_row_count": len(calibration_rows),
            "no_order_placement": True,
        }
    )
    json_write(managed_research_dir / "managed_research_metadata.json", managed_metadata)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_write(out_dir / "action_board.csv", board)
    csv_write(out_dir / "option_outcomes.csv", trade_rows)
    csv_write(out_dir / "option_pattern_validation.csv", validation)
    csv_write(out_dir / "known_mover_audit.csv", mover_rows)
    csv_write(out_dir / "price_pattern_signals.csv", price_outcomes)
    csv_write(out_dir / "price_pattern_validation.csv", price_validation)
    csv_write(
        out_dir / "managed_price_pattern_validation.csv",
        price_pattern_validation,
    )
    csv_write(out_dir / "directional_board.csv", directional_board)
    csv_write(out_dir / "managed_research_scorecard.csv", scorecard)
    csv_write(out_dir / "managed_calibration.csv", calibration_rows)
    csv_write(out_dir / "current_option_setups.csv", current_setup_rows)
    csv_write(out_dir / "managed_selection_audit.csv", managed_metadata.get("selection_audit_rows") or [])
    csv_write(out_dir / "managed_selection_regime.csv", managed_metadata.get("selection_regime_data") or [])
    json_write(out_dir / "metadata.json", managed_metadata)
    (out_dir / "daily_report.md").write_text(
        _managed_primary_report(
            actual_as_of,
            managed_metadata,
            board,
            validation,
            scorecard,
            mover_rows,
            price_validation,
            price_pattern_validation,
            directional_board,
        ),
        encoding="utf-8",
    )
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "as_of": actual_as_of,
        "requested_as_of": requested_as_of,
        "artifact_paths": {
            **{
                str(path.relative_to(out_dir)): str(path)
                for path in sorted(out_dir.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest.json"
            },
            "artifact_manifest.json": str(out_dir / "artifact_manifest.json"),
        },
        "schema_errors": [],
        "no_order_placement": True,
        "acceptance_status": acceptance_status,
        "acceptance_blockers": acceptance_blockers,
        "primary_lane": "managed",
        "legacy_core": "not_run",
    }
    json_write(out_dir / "artifact_manifest.json", manifest)
    return {
        "as_of": actual_as_of,
        "out_dir": str(out_dir),
        "acceptance_status": acceptance_status,
        "acceptance_blockers": acceptance_blockers,
        "managed_research": managed_metadata,
        "metadata": managed_metadata,
    }


def acceptance(
    option_stats: Sequence[Mapping[str, Any]],
    walk_forward_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    rolling_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    board: Sequence[Mapping[str, Any]],
    mover_rows: Sequence[Mapping[str, Any]],
    as_of: Optional[str] = None,
) -> Tuple[str, List[str], Dict[str, Any]]:
    blockers: List[str] = []
    qualified, qualified_walk_forward, qualified_rolling, calibrated = option_gate_sets(
        option_stats,
        walk_forward_rows,
        calibration_rows,
        rolling_rows,
        coverage_rows,
        as_of=as_of,
    )
    if not qualified:
        blockers.append("NO_OPTION_PATTERN_CLEARS_NET_EV_PF_SAMPLE_AND_LATEST_HOLDOUT_GATES")
    if not qualified_walk_forward:
        blockers.append("NO_OPTION_PATTERN_CLEARS_CHRONOLOGICAL_WALK_FORWARD_GATES")
    if not qualified_rolling:
        blockers.append("NO_OPTION_PATTERN_CLEARS_RECENT_ROLLING_HOLDOUT_GATES")
    if not calibrated:
        blockers.append("CALIBRATION_SCORE_MISSING_OR_WEAK")
    target_rows = [row for row in mover_rows if row.get("ticker") in TARGET_TICKERS]
    significant_rows = [row for row in target_rows if row.get("significant_move_5pct") in (True, "True")]
    target_same_direction = sum(bool(row.get("price_signal_same_direction")) for row in significant_rows)
    target_next = len(significant_rows)
    recall = target_same_direction / target_next if target_next else 0.0
    all_recall = (
        sum(bool(row.get("price_signal_same_direction")) for row in target_rows) / len(target_rows)
        if target_rows
        else 0.0
    )
    event_detection = (
        sum(bool(row.get("next_session_event_detected")) for row in significant_rows) / target_next
        if target_next
        else 0.0
    )
    if target_next < 20:
        blockers.append("NAMED_TICKER_AUDIT_TOO_SMALL")
    if recall < 0.30:
        blockers.append("NAMED_TICKER_SIGNIFICANT_FORWARD_RECALL_BELOW_30_PERCENT")
    if not any(row.get("status") == "APPROVED_TRADE" for row in board):
        blockers.append("NO_CURRENT_APPROVED_TRADE")
    return (
        "PASS" if not blockers else "FAIL_REQUIREMENTS_REMAIN",
        blockers,
        {
            "qualified_option_pattern_count": len(qualified),
            "qualified_walk_forward_pattern_count": len(qualified_walk_forward),
            "qualified_rolling_holdout_pattern_count": len(qualified_rolling),
            "calibrated_option_pattern_count": len(calibrated),
            "named_ticker_significant_forward_recall": recall,
            "named_ticker_all_move_forward_recall": all_recall,
            "named_ticker_significant_event_detection": event_detection,
            "named_ticker_significant_audit_rows": target_next,
            "named_ticker_audit_rows": len(target_rows),
        },
    )


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    if not getattr(args, "legacy_core", False):
        return run_managed_primary_pipeline(args)
    base_dir = Path(args.base_dir).expanduser().resolve()
    source_dates = date_dirs(base_dir, args.start_date, args.as_of)
    if not source_dates:
        raise SystemExit(f"No stock-screener date folders found under {base_dir}")
    history, source_meta = load_price_history(base_dir, source_dates)
    active_dates = sorted({
        row.date
        for rows in history.values()
        for row in rows.values()
    })
    if not active_dates:
        raise SystemExit(f"No stock-screener rows found under {base_dir}")
    as_of = args.as_of if args.as_of in active_dates else active_dates[-1]
    dates = [d for d in active_dates if d <= as_of]
    bot_flow_by_date, bot_meta = load_bot_eod_flow_history(base_dir, dates)
    features = derive_price_features(history, dates, bot_flow_by_date)
    signals = generate_price_signals(features, dates)
    price_rows = signal_rows(signals, features, dates)
    current_signals = [signal for signal in signals if signal.date == as_of]
    requests = option_signal_requests(signals, as_of, args.max_option_scan_per_date)
    entry_quotes, entry_meta = load_entry_option_quotes(base_dir, dates, requests, features)
    symbols: Set[str] = set()
    for entry in entry_quotes.values():
        symbols.update(entry_option_symbols(entry))
    target_dates = quote_target_dates(entry_quotes, dates)
    hot_history, hot_meta = load_option_quote_history(base_dir, dates, symbols, target_dates)
    bot_option_history, bot_option_meta = load_bot_eod_option_quote_history(
        base_dir,
        dates,
        symbols,
        target_dates,
    )
    chain_history: Dict[str, Dict[str, Dict[str, Any]]] = {}
    chain_meta: Dict[str, Any] = {}
    if not args.skip_chain_oi_fallback:
        chain_history, chain_meta = load_chain_oi_fallback_history_indexed(
            base_dir,
            dates,
            symbols,
            target_dates,
        )
    quote_history = merge_quote_history(hot_history, chain_history, bot_option_history)
    option_rows = option_rows_for_signals(signals, entry_quotes, quote_history, dates, as_of)
    scored_option_rows = [row for row in option_rows if row.get("status") == "SCORED"]
    exit_provenance_counts = Counter(
        str(row.get("exit_quote_provenance") or "unknown_quote_provenance")
        for row in scored_option_rows
    )
    entry_provenance_counts = Counter(
        str(row.get("entry_quote_provenance") or "unknown_quote_provenance")
        for row in scored_option_rows
    )
    option_missing_reason_counts = Counter(
        str(row.get("outcome_missing_reason") or "unknown")
        for row in option_rows
        if row.get("status") != "SCORED"
    )
    mover_rows = known_mover_audit(features, dates, signals, args.top_movers_per_date)
    target_coverage_rows = feature_coverage(history, dates, signals)
    price_outcome_data = price_outcome_rows(price_rows)
    price_stats = grouped_stats(price_outcome_data, "stock_return")
    price_walk_forward_rows = walk_forward_stats(price_outcome_data, "stock_return")
    price_rolling_rows = rolling_holdout_stats(price_outcome_data, "stock_return")
    price_calibration_rows = calibration_stats(price_outcome_data, "stock_return")
    qualified_price, qualified_price_walk, qualified_price_rolling, calibrated_price = price_gate_sets(
        price_stats,
        price_walk_forward_rows,
        price_calibration_rows,
        price_rolling_rows,
    )
    predictive_option_rows = [row for row in option_rows if is_predictive_option_row(row)]
    predictive_option_rows_all = [
        row for row in option_rows if row.get("signal_role") != "same_day_event"
    ]
    lane_option_rows = validation_lane_rows(predictive_option_rows_all)
    option_coverage_rows = option_outcome_coverage(predictive_option_rows_all + lane_option_rows)
    option_stats = grouped_stats(predictive_option_rows, "net_R") + grouped_stats(lane_option_rows, "net_R")
    walk_forward_rows = walk_forward_stats(
        predictive_option_rows + lane_option_rows,
        "net_R",
    )
    rolling_rows = rolling_holdout_stats(lane_option_rows, "net_R")
    option_calibration_rows = calibration_stats(
        predictive_option_rows + lane_option_rows,
        "net_R",
    )
    calibration_rows = [
        {"lane": "option", **row} for row in option_calibration_rows
    ] + [
        {"lane": "price", **row} for row in price_calibration_rows
    ]
    _, _, qualified_rolling, calibrated = option_gate_sets(
        option_stats,
        walk_forward_rows,
        option_calibration_rows,
        rolling_rows,
        option_coverage_rows,
        as_of=as_of,
    )
    approved_pattern_keys = {
        pattern_key(row) for row in qualified_rolling
    } & {
        pattern_key(row) for row in calibrated
    }
    board = build_current_board(
        current_signals,
        price_rows,
        option_rows,
        as_of,
        entry_quotes,
        approved_pattern_keys,
        option_coverage_rows,
    )
    acceptance_status, acceptance_blockers, acceptance_meta = acceptance(
        option_stats,
        walk_forward_rows,
        option_calibration_rows,
        rolling_rows,
        option_coverage_rows,
        board,
        mover_rows,
        as_of=as_of,
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else base_dir / "out" / "pattern_analysis_v2" / as_of
    out_dir.mkdir(parents=True, exist_ok=True)
    managed_research_dir = out_dir / "managed_research"
    managed_result: Dict[str, Any] = {}
    managed_status = "SKIPPED"
    managed_error = ""
    if not getattr(args, "skip_managed_research", False):
        from .managed import run_managed_research

        managed_result = run_managed_research(
            base_dir,
            args.start_date,
            as_of,
            managed_research_dir,
            cache_root=base_dir / "out" / "pattern_analysis_v2" / "cache" / "managed_quotes",
        )
        managed_status = "COMPLETED"
    metadata: Dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "as_of": as_of,
        "start_date": dates[0],
        "end_date": dates[-1],
        "session_count": len(dates),
        "ticker_count": len(history),
        "price_signal_count": len(signals),
        "current_signal_count": len(current_signals),
        "same_day_event_signal_count": sum(signal.role == "same_day_event" for signal in signals),
        "forward_setup_signal_count": sum(signal.role == "forward_setup" for signal in signals),
        "confirmation_signal_count": sum(signal.role == "confirmation" for signal in signals),
        "price_adjustment_event_count": sum(
            abs((as_float(row.get("price_adjustment_factor")) or 1.0) - 1.0) > 0.005
            for rows in features.values()
            for row in rows.values()
        ),
        "price_adjustment_ticker_count": len({
            str(row.get("ticker") or "")
            for rows in features.values()
            for row in rows.values()
            if abs((as_float(row.get("price_adjustment_factor")) or 1.0) - 1.0) > 0.005
        }),
        "price_walk_forward_row_count": len(price_walk_forward_rows),
        "price_rolling_holdout_row_count": len(price_rolling_rows),
        "qualified_price_pattern_count": len(qualified_price),
        "qualified_price_walk_forward_pattern_count": len(qualified_price_walk),
        "qualified_price_rolling_holdout_pattern_count": len(qualified_price_rolling),
        "calibrated_price_pattern_count": len(calibrated_price),
        "qualified_price_patterns": [
            {
                key: row.get(key)
                for key in (
                    "pattern_family",
                    "direction",
                    "horizon",
                    "strategy",
                    "sample_count",
                    "unique_signal_dates",
                    "average_value",
                    "profit_factor",
                    "latest_holdout_average",
                    "date_lower_mean_95",
                )
            }
            for row in qualified_price
        ],
        "option_candidate_signal_count": len({row["signal_id"] for row in option_rows}),
        "option_candidate_count": len({row["signal_id"] for row in option_rows}),
        "option_contract_variant_count": sum(
            len(entry_variants(entry)) for entry in entry_quotes.values()
        ),
        "target_quote_symbol_count": len(target_dates),
        "target_quote_date_pair_count": sum(len(values) for values in target_dates.values()),
        "option_missing_exit_outcome_count": sum(row.get("status") == "MISSING_EXIT_QUOTE" for row in option_rows),
        "option_pending_future_outcome_count": sum(row.get("status") == "PENDING_FUTURE" for row in option_rows),
        "option_ineligible_contract_count": sum(row.get("status") == "INELIGIBLE_CONTRACT" for row in option_rows),
        "option_missing_exit_reason_counts": dict(sorted(option_missing_reason_counts.items())),
        "option_scored_outcome_count": sum(row.get("status") == "SCORED" for row in option_rows),
        "option_scored_exit_provenance_counts": dict(sorted(exit_provenance_counts.items())),
        "option_scored_entry_provenance_counts": dict(sorted(entry_provenance_counts.items())),
        "option_predictive_scored_outcome_count": len(predictive_option_rows),
        "validation_lane_scored_outcome_count": sum(
            row.get("status") == "SCORED" for row in lane_option_rows
        ),
        "option_outcome_coverage_rows": len(option_coverage_rows),
        "option_outcome_coverage_fail_count": sum(
            row.get("coverage_gate") == "FAIL" for row in option_coverage_rows
        ),
        "walk_forward_row_count": len(walk_forward_rows),
        "rolling_holdout_row_count": len(rolling_rows),
        "confidence_calibration_row_count": len(calibration_rows),
        "known_mover_audit_rows": len(mover_rows),
        "target_ticker_coverage": target_coverage_rows,
        "acceptance_status": acceptance_status,
        "acceptance_blockers": acceptance_blockers,
        "managed_research_status": managed_status,
        "managed_research_error": managed_error,
        "managed_research_out_dir": str(managed_research_dir),
        "managed_research_qualified_count": managed_result.get("metadata", {}).get("qualified_managed_count", 0),
        "managed_research_pattern_count": managed_result.get("metadata", {}).get("research_pattern_count", 0),
        "managed_research_cache_hits": managed_result.get("metadata", {}).get("cache_hits", 0),
        "managed_research_cache_misses": managed_result.get("metadata", {}).get("cache_misses", 0),
        "managed_research_no_approved_trade": True,
        **acceptance_meta,
        **source_meta,
        **bot_meta,
        **entry_meta,
        **hot_meta,
        **bot_option_meta,
        **chain_meta,
        "no_order_placement": True,
    }
    csv_write(out_dir / "price_pattern_signals.csv", price_rows)
    csv_write(out_dir / "option_outcomes.csv", option_rows)
    csv_write(out_dir / "price_pattern_validation.csv", price_stats)
    csv_write(out_dir / "price_walk_forward_validation.csv", price_walk_forward_rows)
    csv_write(out_dir / "price_rolling_holdout_validation.csv", price_rolling_rows)
    csv_write(out_dir / "option_pattern_validation.csv", option_stats)
    csv_write(out_dir / "walk_forward_validation.csv", walk_forward_rows)
    csv_write(out_dir / "rolling_holdout_validation.csv", rolling_rows)
    csv_write(out_dir / "validation_lane_rules.csv", [dict(spec) for spec in VALIDATION_LANE_SPECS])
    csv_write(out_dir / "option_outcome_coverage.csv", option_coverage_rows)
    csv_write(out_dir / "confidence_calibration.csv", calibration_rows)
    csv_write(out_dir / "known_mover_audit.csv", mover_rows)
    csv_write(out_dir / "target_ticker_coverage.csv", target_coverage_rows)
    csv_write(out_dir / "action_board.csv", board)
    json_write(out_dir / "metadata.json", metadata)
    report = render_report(as_of, metadata, board, price_stats, option_stats, mover_rows)
    (out_dir / "daily_report.md").write_text(report, encoding="utf-8")
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "as_of": as_of,
        "artifact_paths": {
            **{
                str(path.relative_to(out_dir)): str(path)
                for path in sorted(out_dir.rglob("*"))
                if path.is_file() and path.name != "artifact_manifest.json"
            },
            "artifact_manifest.json": str(out_dir / "artifact_manifest.json"),
        },
        "schema_errors": [],
        "no_order_placement": True,
        "acceptance_status": acceptance_status,
        "acceptance_blockers": acceptance_blockers,
        "managed_research_status": managed_status,
        "managed_research_out_dir": str(managed_research_dir),
    }
    json_write(out_dir / "artifact_manifest.json", manifest)
    return {
        "as_of": as_of,
        "out_dir": str(out_dir),
        "acceptance_status": acceptance_status,
        "acceptance_blockers": acceptance_blockers,
        "managed_research": managed_result.get("metadata", {}),
        "metadata": metadata,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python3 -m uwos.pattern_analysis_v2")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--as-of", default="latest")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="validated materialized quote-cache directory (source signatures are rechecked)",
    )
    parser.add_argument("--max-option-scan-per-date", type=int, default=DEFAULT_MAX_OPTION_SCAN_PER_DATE)
    parser.add_argument("--top-movers-per-date", type=int, default=25)
    parser.add_argument("--skip-chain-oi-fallback", action="store_true")
    parser.add_argument("--skip-managed-research", action="store_true")
    parser.add_argument(
        "--legacy-core",
        action="store_true",
        help="run the pre-rebuild bot-heavy scorer as a secondary audit lane",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = run_pipeline(parse_args(argv))
    print(json.dumps({key: result[key] for key in ("as_of", "out_dir", "acceptance_status", "acceptance_blockers")}, indent=2))
    return 0
