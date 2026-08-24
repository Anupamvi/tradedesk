"""Leakage-safe managed-exit option research for Pattern Analysis V2.

This module is deliberately independent from the fixed-horizon scorer.  It
uses the stock screener at an EOD signal date, enters the next available
session at the ask on one OCC contract, and walks later dated chain quotes at
the bid until a predeclared profit target or time stop fires.

Two archive details are correctness-critical here:

* dated folders are not necessarily trading sessions; a folder can contain
  reports without a screener or chain archive;
* the chain archive's folder date is the download date, while ``last_date`` is
  the session represented by its quotes.  Holidays therefore require mapping
  a quote session to the first later source archive that actually carries it.

The output keeps unscored entries with an explicit status.  Missing quotes are
never converted into a mark at the last observed price.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .engine import (
    as_float,
    clean_ticker,
    find_fallback_whale_sources,
    load_fallback_option_flow_rows,
    parse_occ,
)


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# These issue types can have listed options.  Indexes, structured products,
# units, and blank classifications are intentionally excluded.
OPTIONABLE_ISSUE_TYPES = frozenset({"Common Stock", "ADR", "ETF"})


@dataclass(frozen=True)
class ManagedConfig:
    """A complete, predeclared trade hypothesis."""

    name: str = "TREND_CONTINUATION_MANAGED_LONG_OPTION"
    direction: str = "call"
    signal_direction: Optional[str] = None
    option_type: Optional[str] = None
    signal_rule: str = "trend_quantile"
    top_quantile: float = 0.90
    event_move: float = 0.05
    event_position_limit: float = 0.60
    min_marketcap: float = 2_000_000_000.0
    min_etf_marketcap: float = 500_000_000.0
    min_avg30_volume: float = 250_000.0
    max_iv_rank: Optional[float] = None
    min_flow_bias: float = 0.25
    min_flow_volume_ratio: float = 1.25
    min_bot_flow_premium: float = 100_000.0
    earnings_min_days: int = 0
    earnings_max_days: int = 10
    min_implied_move_perc: float = 0.0
    min_sector_names: int = 12
    min_dte: int = 60
    max_dte: int = 110
    target_dte: float = 80.0
    min_open_interest: float = 50.0
    min_entry_bid: float = 0.05
    max_spread_pct: float = 0.12
    moneyness: float = 1.05
    structure: str = "long_option"
    short_moneyness: float = 1.12
    short_put_moneyness: float = 0.95
    short_call_moneyness: float = 1.05
    long_put_moneyness: float = 0.90
    long_call_moneyness: float = 1.10
    market_filter: str = "ANY"
    max_positions_per_day: Optional[int] = None
    profit_target: float = 0.50
    stop_loss: Optional[float] = None
    max_hold_sessions: int = 40
    max_exit_quote_gap_sessions: int = 1
    fee_per_side: float = 1.50
    one_per_ticker: bool = True


@dataclass(frozen=True)
class ChainSource:
    path: str
    source_date: str
    quote_dates: Tuple[str, ...]
    kind: str = "chain_oi"


def _date_dirs(base_dir: Path, start_date: str, end_date: str) -> List[Path]:
    return sorted(
        path
        for path in base_dir.iterdir()
        if path.is_dir()
        and DATE_RE.fullmatch(path.name)
        and start_date <= path.name <= end_date
    )


def _archives(day: Path, stem: str) -> List[Path]:
    paths = [
        path
        for path in sorted(day.glob(f"{stem}-*.zip"))
        if path.is_file() and day.name in path.name
    ]
    parts = [path for path in paths if ".part-" in path.name]
    return parts if parts else paths[:1]


def _position_52w_features(frame: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Return raw and bounded 52-week position values.

    UW's published high/low fields can lag a corporate action or a new price
    extreme, so the normalized value is not guaranteed to lie in [0, 1].  A
    bounded rank feature is the only meaningful input to a cross-sectional
    trend selector; the raw value remains available for provenance auditing.
    """

    raw = (
        (frame["close"] - frame["week_52_low"])
        / (frame["week_52_high"] - frame["week_52_low"]).replace(0, np.nan)
    )
    return raw, raw.clip(lower=0.0, upper=1.0)


def _read_zip(path: Path, usecols: Sequence[str], nrows: Optional[int] = None) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not members:
            return pd.DataFrame()
        member = members[0]
        with archive.open(member) as handle:
            header = pd.read_csv(handle, nrows=0)
        selected = [column for column in usecols if column in header.columns]
        if not selected:
            return pd.DataFrame()
        with archive.open(member) as handle:
            return pd.read_csv(handle, usecols=selected, nrows=nrows, low_memory=False)


def _quote_dates(path: Path, kind: str = "chain_oi") -> Tuple[str, ...]:
    date_column = "date" if kind == "hot" else "last_date"
    frame = _read_zip(path, [date_column], nrows=2000)
    if frame.empty or date_column not in frame.columns:
        return ()
    dates = sorted(
        {
            str(value)[:10]
            for value in frame[date_column].dropna().tolist()
            if DATE_RE.fullmatch(str(value)[:10])
        }
    )
    return tuple(dates)


def build_chain_source_index(
    base_dir: Path,
    start_date: str,
    end_date: str,
) -> Tuple[Dict[str, List[ChainSource]], Dict[str, Any]]:
    """Map quote session dates to the later archive that contains them."""

    by_quote_date: Dict[str, List[ChainSource]] = {}
    source_rows: List[ChainSource] = []
    for day in _date_dirs(base_dir, start_date, end_date):
        for path in _archives(day, "chain-oi-changes"):
            quote_dates = _quote_dates(path)
            source = ChainSource(str(path), day.name, quote_dates, "chain_oi")
            source_rows.append(source)
            for quote_date in quote_dates:
                by_quote_date.setdefault(quote_date, []).append(source)
    for quote_date in by_quote_date:
        by_quote_date[quote_date] = sorted(
            by_quote_date[quote_date], key=lambda item: (item.source_date, item.path)
        )
    metadata = {
        "chain_source_file_count": len(source_rows),
        "chain_quote_date_count": len(by_quote_date),
        "chain_source_dates_with_multiple_quote_dates": sum(
            len(source.quote_dates) > 1 for source in source_rows
        ),
        "chain_quote_date_map": {
            quote_date: [source.source_date for source in sources]
            for quote_date, sources in sorted(by_quote_date.items())
        },
    }
    return by_quote_date, metadata


def build_quote_source_index(
    base_dir: Path,
    start_date: str,
    end_date: str,
) -> Tuple[Dict[str, List[ChainSource]], Dict[str, Any]]:
    """Map actual quote sessions to both same-day hot chains and OI snapshots."""

    by_quote_date: Dict[str, List[ChainSource]] = {}
    source_rows: List[ChainSource] = []
    for day in _date_dirs(base_dir, start_date, end_date):
        for stem, kind in (("hot-chains", "hot"), ("chain-oi-changes", "chain_oi")):
            for path in _archives(day, stem):
                quote_dates = _quote_dates(path, kind)
                source = ChainSource(str(path), day.name, quote_dates, kind)
                source_rows.append(source)
                for quote_date in quote_dates:
                    by_quote_date.setdefault(quote_date, []).append(source)
    for quote_date in by_quote_date:
        by_quote_date[quote_date] = sorted(
            by_quote_date[quote_date],
            key=lambda item: (
                0 if item.kind == "hot" else 1,
                item.source_date,
                item.path,
            ),
        )
    metadata = {
        "quote_source_file_count": len(source_rows),
        "quote_source_kind_counts": {
            kind: sum(source.kind == kind for source in source_rows)
            for kind in ("hot", "chain_oi")
        },
        "quote_date_count": len(by_quote_date),
        "quote_date_map": {
            quote_date: [
                {"source_date": source.source_date, "kind": source.kind}
                for source in sources
            ]
            for quote_date, sources in sorted(by_quote_date.items())
        },
    }
    return by_quote_date, metadata


def load_stock_panel(
    base_dir: Path,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load only point-in-time screener fields needed by the managed lane."""

    columns = [
        "ticker",
        "sector",
        "issue_type",
        "marketcap",
        "close",
        "prev_close",
        "avg30_volume",
        "iv_rank",
        "implied_move_perc",
        "next_earnings_date",
        "call_volume",
        "put_volume",
        "avg_30_day_call_volume",
        "avg_30_day_put_volume",
        "bullish_premium",
        "bearish_premium",
        "week_52_high",
        "week_52_low",
    ]
    frames: List[pd.DataFrame] = []
    source_dates: List[str] = []
    source_files = 0
    raw_rows = 0
    for day in _date_dirs(base_dir, start_date, end_date):
        paths = _archives(day, "stock-screener")
        if not paths:
            continue
        day_frames = []
        for path in paths:
            frame = _read_zip(path, columns)
            if frame.empty or "ticker" not in frame.columns:
                continue
            day_frames.append(frame)
            source_files += 1
            raw_rows += len(frame)
        if not day_frames:
            continue
        frame = pd.concat(day_frames, ignore_index=True)
        frame["date"] = day.name
        for column in [
            "marketcap",
            "close",
            "prev_close",
            "avg30_volume",
            "iv_rank",
            "implied_move_perc",
            "call_volume",
            "put_volume",
            "avg_30_day_call_volume",
            "avg_30_day_put_volume",
            "bullish_premium",
            "bearish_premium",
            "week_52_high",
            "week_52_low",
        ]:
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        for column in ["sector", "issue_type"]:
            if column not in frame.columns:
                frame[column] = ""
        if "next_earnings_date" not in frame.columns:
            frame["next_earnings_date"] = ""
        frame["ticker"] = frame["ticker"].map(clean_ticker)
        frame["sector"] = frame["sector"].fillna("").astype(str).str.strip()
        frame["issue_type"] = frame["issue_type"].fillna("").astype(str).str.strip()
        frame["position_52w_raw"], frame["position_52w"] = _position_52w_features(frame)
        frame["position_52w_out_of_range"] = (
            frame["position_52w_raw"].notna()
            & ~frame["position_52w_raw"].between(0.0, 1.0)
        )
        frame["return_1d"] = frame["close"] / frame["prev_close"].replace(0, np.nan) - 1.0
        total_premium = frame["bullish_premium"].abs() + frame["bearish_premium"].abs()
        frame["flow_bias"] = (
            (frame["bullish_premium"] - frame["bearish_premium"])
            / total_premium.replace(0, np.nan)
        )
        frame["flow_volume_ratio"] = (
            (frame["call_volume"] + frame["put_volume"])
            / (frame["avg_30_day_call_volume"] + frame["avg_30_day_put_volume"]).replace(0, np.nan)
        )
        earnings = pd.to_datetime(frame["next_earnings_date"], errors="coerce")
        signal_dates = pd.to_datetime(frame["date"], errors="coerce")
        frame["days_to_earnings"] = (earnings - signal_dates).dt.days.astype("float64")
        frame.loc[frame["days_to_earnings"] < 0, "days_to_earnings"] = np.nan
        frame = frame.drop_duplicates(["date", "ticker"], keep="last")
        frames.append(frame)
        source_dates.append(day.name)
    if not frames:
        raise ValueError("no dated stock-screener rows found")
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    adjusted_parts: List[pd.Series] = []
    for _ticker, group in panel.groupby("ticker", sort=False):
        raw = group["close"].tolist()
        adjusted: List[float] = []
        for index, value in enumerate(raw):
            if index > 0:
                prior_raw = as_float(raw[index - 1])
                previous_close = as_float(group.iloc[index]["prev_close"])
                if prior_raw and previous_close and prior_raw > 0 and previous_close > 0:
                    factor = previous_close / prior_raw
                    if math.isfinite(factor) and factor > 0:
                        adjusted = [item * factor for item in adjusted]
            adjusted.append(float(value) if value is not None and math.isfinite(float(value)) else np.nan)
        adjusted_parts.append(pd.Series(adjusted, index=group.index))
    panel["adjusted_close"] = pd.concat(adjusted_parts).sort_index()
    panel["return_5d"] = panel["adjusted_close"] / _session_shifted_price(panel, 5) - 1.0
    panel["return_20d"] = panel["adjusted_close"] / _session_shifted_price(panel, 20) - 1.0
    metadata = {
        "stock_screener_source_dates": len(source_dates),
        "stock_screener_source_files": source_files,
        "stock_screener_raw_rows": raw_rows,
        "stock_screener_rows": len(panel),
        "stock_screener_duplicate_rows_removed": raw_rows - len(panel),
        "stock_screener_dates": source_dates,
        "stock_screener_position_52w_out_of_range_rows": int(
            panel["position_52w_out_of_range"].sum()
        ),
        "stock_screener_position_52w_out_of_range_tickers": int(
            panel.loc[panel["position_52w_out_of_range"], "ticker"].nunique()
        ),
    }
    return panel, metadata


def load_cached_bot_flow(
    base_dir: Path,
    start_date: str,
    end_date: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load materialized bot flow plus small dated fallback exports.

    Some early sessions have ``whale_trades_filtered`` exports but no
    bot-EOD cache.  They are valid dated UW flow inputs and can be parsed
    without reopening the multi-GB bot-EOD archives.
    """

    cache_dir = base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
    columns = [
        "date",
        "ticker",
        "flow_call_ask_premium",
        "flow_put_ask_premium",
        "flow_total_premium",
        "flow_call_trade_count",
        "flow_put_trade_count",
    ]
    frames: List[pd.DataFrame] = []
    source_dates: List[str] = []
    source_files = 0
    fallback_source_dates = 0
    fallback_raw_rows = 0
    fallback_flow_rows = 0

    def normalize(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not {"date", "ticker"}.issubset(frame.columns):
            return pd.DataFrame(columns=columns)
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        frame = frame[columns].copy()
        frame["date"] = frame["date"].astype(str).str.slice(0, 10)
        frame["ticker"] = frame["ticker"].map(clean_ticker)
        for column in columns[2:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame[frame["date"].between(start_date, end_date)].copy()

    for path in sorted(cache_dir.glob("bot_eod_flow_by_ticker_*.csv")):
        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", path.stem)
        source_date = date_match.group(1) if date_match else ""
        if not DATE_RE.fullmatch(source_date) or not (start_date <= source_date <= end_date):
            continue
        try:
            frame = pd.read_csv(path, usecols=lambda column: column in columns)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        frame = normalize(frame)
        if frame.empty:
            continue
        frames.append(frame)
        source_dates.extend(frame["date"].dropna().astype(str).unique().tolist())
        source_files += 1

    # Fill only dates not already represented by the materialized cache.  This
    # keeps source coverage complete without rescanning the large bot archives.
    cached_dates = set(source_dates)
    for day in _date_dirs(base_dir, start_date, end_date):
        if day.name in cached_dates:
            continue
        refs = find_fallback_whale_sources(day, day.name)
        if not refs:
            continue
        fallback_rows, raw_rows = load_fallback_option_flow_rows(refs, day.name)
        if not fallback_rows:
            continue
        frame = normalize(pd.DataFrame(fallback_rows))
        if frame.empty:
            continue
        frames.append(frame)
        source_dates.extend(frame["date"].dropna().astype(str).unique().tolist())
        fallback_source_dates += 1
        fallback_raw_rows += int(raw_rows)
        fallback_flow_rows += len(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "bot_flow_bias", "bot_flow_total_premium"]), {
            "bot_flow_cache_dir": str(cache_dir),
            "bot_flow_source_files": 0,
            "bot_flow_source_dates": [],
            "bot_flow_rows": 0,
            "bot_flow_fallback_source_dates": 0,
            "bot_flow_fallback_raw_rows": 0,
            "bot_flow_fallback_rows": 0,
            "bot_flow_status": "CACHE_NOT_AVAILABLE",
        }
    flow = pd.concat(frames, ignore_index=True)
    flow = flow.drop_duplicates(["date", "ticker"], keep="last")
    denominator = flow["flow_call_ask_premium"] + flow["flow_put_ask_premium"]
    flow["bot_flow_bias"] = (
        (flow["flow_call_ask_premium"] - flow["flow_put_ask_premium"])
        / denominator.replace(0, np.nan)
    )
    metadata = {
        "bot_flow_cache_dir": str(cache_dir),
        "bot_flow_source_files": source_files,
        "bot_flow_source_dates": sorted(set(source_dates)),
        "bot_flow_rows": len(flow),
        "bot_flow_fallback_source_dates": fallback_source_dates,
        "bot_flow_fallback_raw_rows": fallback_raw_rows,
        "bot_flow_fallback_rows": fallback_flow_rows,
        "bot_flow_status": (
            "CACHE_AND_FALLBACK_LOADED"
            if source_files and fallback_source_dates
            else "FALLBACK_ONLY"
            if fallback_source_dates
            else "CACHE_LOADED"
        ),
    }
    return flow, metadata


def _session_shifted_price(panel: pd.DataFrame, horizon: int) -> pd.Series:
    """Look up a ticker's price exactly N global sessions earlier."""

    ordered_sessions = sorted(panel["date"].astype(str).unique())
    session_positions = {date: index for index, date in enumerate(ordered_sessions)}
    work = panel[["ticker", "date"]].copy()
    work["_row_id"] = np.arange(len(work))
    work["target_date"] = [
        ordered_sessions[session_positions[str(date)] - int(horizon)]
        if session_positions.get(str(date), int(horizon)) >= int(horizon)
        else ""
        for date in work["date"]
    ]
    lookup = panel[["ticker", "date", "adjusted_close"]].rename(
        columns={"date": "target_date", "adjusted_close": "prior_price"}
    )
    matched = work.merge(
        lookup,
        on=["ticker", "target_date"],
        how="left",
        sort=False,
    ).sort_values("_row_id")
    return pd.Series(
        pd.to_numeric(matched["prior_price"], errors="coerce").to_numpy(),
        index=panel.index,
    )


def eligible_sessions(panel: pd.DataFrame, source_index: Mapping[str, Sequence[ChainSource]]) -> List[str]:
    """Return real sessions, excluding report-only dated folders."""

    stock_dates = set(panel["date"].astype(str))
    return sorted(stock_dates & set(source_index))


def named_mover_audit(
    panel: pd.DataFrame,
    sessions: Sequence[str],
    strategies: Mapping[str, Tuple[ManagedConfig, str]],
    move_threshold: float = 0.05,
) -> List[Dict[str, Any]]:
    """Audit named movers separately from pre-event prediction.

    A same-day event flag is descriptive: it can confirm that a large move was
    present in the downloaded row, but it is not a claim that the move was
    known before the close.  Predictive families are evaluated only on the
    prior session.
    """

    from .engine import ENERGY_PROXY_TICKERS, PRIMARY_TARGET_TICKERS

    targets = set(PRIMARY_TARGET_TICKERS) | set(ENERGY_PROXY_TICKERS)
    by_date = {
        str(day): frame.set_index("ticker")
        for day, frame in panel.groupby("date", sort=False)
    }
    ordered_sessions = [str(day) for day in sessions if str(day) in by_date]
    rows: List[Dict[str, Any]] = []
    for index, event_date in enumerate(ordered_sessions):
        event_frame = by_date[event_date]
        prior_date = ordered_sessions[index - 1] if index > 0 else ""
        next_date = ordered_sessions[index + 1] if index + 1 < len(ordered_sessions) else ""
        prior_frame = by_date.get(prior_date)
        next_frame = by_date.get(next_date)
        for ticker in sorted(targets):
            if ticker not in event_frame.index:
                continue
            event_row = event_frame.loc[ticker]
            event_return = as_float(event_row.get("return_1d"))
            if event_return is None or abs(event_return) < move_threshold:
                continue
            pre_event_families: List[str] = []
            pre_event_same_direction: List[str] = []
            if prior_frame is not None:
                for name, (config, control) in strategies.items():
                    if control not in {"signal", "bot_flow"}:
                        continue
                    selected = _signals_for_day(prior_frame.reset_index(), config)
                    if ticker not in {item for names in selected.values() for item in names}:
                        continue
                    pre_event_families.append(name)
                    if (event_return > 0 and config.direction == "call") or (
                        event_return < 0 and config.direction == "put"
                    ):
                        pre_event_same_direction.append(name)
            next_return = None
            if next_frame is not None and ticker in next_frame.index:
                next_return = as_float(next_frame.loc[ticker].get("return_1d"))
            post_event_families: List[str] = []
            post_event_same_direction: List[str] = []
            if next_return is not None:
                for name, (config, control) in strategies.items():
                    if control not in {"signal", "bot_flow"}:
                        continue
                    selected = _signals_for_day(event_frame.reset_index(), config)
                    if ticker not in {item for names in selected.values() for item in names}:
                        continue
                    post_event_families.append(name)
                    if (next_return > 0 and config.direction == "call") or (
                        next_return < 0 and config.direction == "put"
                    ):
                        post_event_same_direction.append(name)
            rows.append(
                {
                    "ticker": ticker,
                    "thematic_group": "OIL_PROXY" if ticker in ENERGY_PROXY_TICKERS else "PRIMARY_TARGET",
                    "event_date": event_date,
                    "event_return_1d": event_return,
                    "move_direction": "up" if event_return > 0 else "down",
                    "prior_signal_date": prior_date,
                    "next_session_date": next_date,
                    "next_session_return_1d": next_return,
                    "significant_move_5pct": True,
                    "same_day_event_detected": True,
                    "pre_event_matching_families": ";".join(pre_event_families),
                    "pre_event_same_direction_families": ";".join(pre_event_same_direction),
                    "pre_event_any_family": bool(pre_event_families),
                    "pre_event_same_direction": bool(pre_event_same_direction),
                    "post_event_matching_families": ";".join(post_event_families),
                    "post_event_same_direction_families": ";".join(post_event_same_direction),
                    "post_event_any_family": bool(post_event_families),
                    "post_event_same_direction": bool(post_event_same_direction),
                    "post_event_followup_reason": (
                        "POST_EVENT_SIGNAL_MISSING"
                        if not post_event_families
                        else "POST_EVENT_DIRECTIONAL_SIGNAL_MISSING"
                        if not post_event_same_direction
                        else "POST_EVENT_SIGNAL_PRESENT"
                    ),
                    "miss_reason": (
                        "PRE_EVENT_SIGNAL_MISSING"
                        if not pre_event_families
                        else "PRE_EVENT_DIRECTIONAL_SIGNAL_MISSING"
                        if not pre_event_same_direction
                        else "PRE_EVENT_SIGNAL_PRESENT"
                    ),
                }
            )
    return rows


def managed_price_research(
    panel: pd.DataFrame,
    sessions: Sequence[str],
    strategies: Mapping[str, Tuple[ManagedConfig, str]],
    horizons: Sequence[int] = (1, 5, 20),
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Score the underlying directional signal before option costs.

    The signal is selected on ``signal_date`` and the label is a later close
    from the same ticker.  This lane is diagnostic: it cannot approve an
    option ticket by itself.
    """

    ordered_sessions = [str(day) for day in sessions]
    by_date = {
        str(day): frame.set_index("ticker")
        for day, frame in panel.groupby("date", sort=False)
    }
    session_positions = {str(date): index for index, date in enumerate(ordered_sessions)}
    future_labels: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    by_ticker = panel.sort_values(["ticker", "date"]).groupby("ticker", sort=False)
    for ticker, group in by_ticker:
        prices_by_date = {
            str(date): value
            for date, value in zip(
                group["date"].astype(str).tolist(),
                pd.to_numeric(group["adjusted_close"], errors="coerce").tolist(),
            )
        }
        for signal_date in prices_by_date:
            signal_position = session_positions.get(signal_date)
            for horizon in horizons:
                entry_position = (
                    signal_position + 1
                    if signal_position is not None
                    else None
                )
                target_position = (
                    entry_position + int(horizon)
                    if entry_position is not None
                    else None
                )
                entry_date = (
                    ordered_sessions[entry_position]
                    if entry_position is not None and entry_position < len(ordered_sessions)
                    else None
                )
                target_date = (
                    ordered_sessions[target_position]
                    if target_position is not None and target_position < len(ordered_sessions)
                    else None
                )
                entry_price = prices_by_date.get(entry_date) if entry_date else None
                future = prices_by_date.get(target_date) if target_date else None
                value = (
                    float(future / entry_price - 1.0)
                    if entry_price is not None
                    and future is not None
                    and math.isfinite(float(entry_price))
                    and math.isfinite(float(future))
                    and float(entry_price) > 0
                    else None
                )
                future_labels[(str(ticker), signal_date, int(horizon))] = {
                    "entry_date": entry_date or "",
                    "target_date": target_date or "",
                    "entry_price": entry_price,
                    "target_price": future,
                    "stock_return": value,
                }

    outcomes: List[Dict[str, Any]] = []
    for signal_date in ordered_sessions:
        frame = by_date.get(signal_date)
        if frame is None:
            continue
        for strategy_key, (config, control) in strategies.items():
            if control not in {"signal", "bot_flow"}:
                continue
            selected = _signals_for_day(frame.reset_index(), config)
            names = {name for values in selected.values() for name in values}
            for ticker in sorted(names):
                raw_row = frame.loc[ticker] if ticker in frame.index else None
                if raw_row is None:
                    continue
                for horizon in horizons:
                    label = future_labels.get(
                        (ticker, signal_date, int(horizon)),
                        {
                            "entry_date": "",
                            "target_date": "",
                            "entry_price": None,
                            "target_price": None,
                            "stock_return": None,
                        },
                    )
                    stock_return = label["stock_return"]
                    outcomes.append(
                        {
                            "ticker": ticker,
                            "signal_date": signal_date,
                            "entry_date": label["entry_date"],
                            "target_date": label["target_date"],
                            "entry_price": label["entry_price"],
                            "target_price": label["target_price"],
                            "strategy_key": strategy_key,
                            "direction": "bullish" if config.direction == "call" else "bearish",
                            "horizon": int(horizon),
                            "stock_return": stock_return,
                            "directional_return": (
                                stock_return if config.direction == "call" else -stock_return
                            )
                            if stock_return is not None
                            else None,
                            "position_52w": as_float(raw_row.get("position_52w")),
                            "flow_bias": as_float(raw_row.get("flow_bias")),
                            "status": "SCORED" if stock_return is not None else "PENDING_FUTURE",
                        }
                    )
    if not outcomes:
        return [], []
    frame = pd.DataFrame(outcomes)
    frame["sample"] = np.select(
        [frame["signal_date"] < "2026-04-14", frame["signal_date"] < "2026-06-15"],
        ["TRAIN", "VALIDATION"],
        default="HOLDOUT",
    )
    summary: List[Dict[str, Any]] = []
    for (strategy_key, direction, horizon, sample), group in frame.groupby(
        ["strategy_key", "direction", "horizon", "sample"], sort=True
    ):
        scored = group[group["status"].eq("SCORED")]
        values = pd.to_numeric(scored["directional_return"], errors="coerce").dropna()
        by_date_mean = scored.groupby("signal_date")["directional_return"].mean()
        lower = (
            float(values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values)))
            if len(values) > 1
            else float(values.mean()) if len(values) == 1 else None
        )
        date_lower = (
            float(
                by_date_mean.mean()
                - 1.96 * by_date_mean.std(ddof=1) / math.sqrt(len(by_date_mean))
            )
            if len(by_date_mean) > 1
            else float(by_date_mean.mean()) if len(by_date_mean) == 1 else None
        )
        positive = float(values[values > 0].sum())
        negative = float(-values[values < 0].sum())
        pending_future_count = int(group["status"].eq("PENDING_FUTURE").sum())
        eligible_count = max(0, len(group) - pending_future_count)
        summary.append(
            {
                "strategy_key": strategy_key,
                "direction": direction,
                "horizon": int(horizon),
                "sample": sample,
                "entry_count": int(len(group)),
                "eligible_count": eligible_count,
                "pending_future_count": pending_future_count,
                "scored_count": int(len(scored)),
                # A fixed-horizon label is right-censored at the end of the
                # as-of window.  Do not fail coverage merely because that
                # future label has not matured yet.
                "coverage": float(len(scored) / eligible_count) if eligible_count else None,
                "unique_signal_dates": int(scored["signal_date"].nunique()),
                "average_directional_return": float(values.mean()) if len(values) else None,
                "win_rate": float((values > 0).mean()) if len(values) else None,
                "profit_factor": positive / negative if negative > 0 else None,
                "lower_mean_95": lower,
                "date_average_directional_return": float(by_date_mean.mean()) if len(by_date_mean) else None,
                "date_lower_mean_95": date_lower,
                "date_max_drawdown": _date_max_drawdown(by_date_mean),
            }
        )
    return outcomes, summary


def _date_max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    curve = values.sort_index().cumsum()
    return float((curve - curve.cummax()).min())


def managed_price_validation_rows(
    price_validation: Sequence[Mapping[str, Any]],
    min_scored: int = 20,
    min_dates: int = 20,
    min_coverage: float = 0.70,
    min_profit_factor: float = 1.20,
    min_lower_mean_95: float = 0.0,
    max_drawdown: float = -10.0,
) -> List[Dict[str, Any]]:
    """Apply fixed chronology gates to the underlying directional lane.

    These rows intentionally do not approve an option ticket. They answer a
    separate question: whether the stock-direction hypothesis itself survives
    all three chronological samples after right-censored labels are removed
    from the coverage denominator.
    """

    def number(row: Mapping[str, Any], key: str) -> Optional[float]:
        value = row.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    grouped: Dict[Tuple[str, str, int], Dict[str, Mapping[str, Any]]] = {}
    for row in price_validation:
        key = (
            str(row.get("strategy_key") or ""),
            str(row.get("direction") or ""),
            int(row.get("horizon") or 0),
        )
        grouped.setdefault(key, {})[str(row.get("sample") or "")] = row

    output: List[Dict[str, Any]] = []
    for (strategy_key, direction, horizon), samples in sorted(grouped.items()):
        blockers: List[str] = []
        sample_pass: Dict[str, bool] = {}
        flattened: Dict[str, Any] = {
            "strategy_key": strategy_key,
            "direction": direction,
            "horizon": horizon,
        }
        for sample in ("TRAIN", "VALIDATION", "HOLDOUT"):
            row = samples.get(sample)
            prefix = sample.lower()
            if row is None:
                blockers.append(f"{sample}_SAMPLE_MISSING")
                sample_pass[sample] = False
                continue
            pending = number(row, "pending_future_count") or 0.0
            entry_count = number(row, "entry_count") or 0.0
            eligible_count = number(row, "eligible_count")
            if eligible_count is None:
                eligible_count = max(0.0, entry_count - pending)
            checks = {
                "scored_count": (number(row, "scored_count") or 0.0) >= min_scored,
                "unique_signal_dates": (number(row, "unique_signal_dates") or 0.0) >= min_dates,
                "coverage": (number(row, "coverage") or 0.0) >= min_coverage,
                "average_directional_return": (
                    number(row, "average_directional_return") or 0.0
                ) > 0.0,
                "profit_factor": (number(row, "profit_factor") or 0.0) >= min_profit_factor,
                "lower_mean_95": (number(row, "lower_mean_95") or 0.0) > min_lower_mean_95,
                "date_lower_mean_95": (
                    number(row, "date_lower_mean_95") or 0.0
                ) > min_lower_mean_95,
                "drawdown": (number(row, "date_max_drawdown") or float("-inf")) >= max_drawdown,
            }
            sample_pass[sample] = all(checks.values())
            for check, passed in checks.items():
                flattened[f"{prefix}_{check}"] = passed
                if not passed:
                    blockers.append(f"{sample}_{check.upper()}_GATE")
            for key in (
                "entry_count",
                "eligible_count",
                "pending_future_count",
                "scored_count",
                "coverage",
                "unique_signal_dates",
                "average_directional_return",
                "profit_factor",
                "lower_mean_95",
                "date_average_directional_return",
                "date_lower_mean_95",
                "date_max_drawdown",
            ):
                flattened[f"{prefix}_{key}"] = (
                    eligible_count if key == "eligible_count" else row.get(key)
                )
        qualified = all(
            sample_pass.get(sample, False)
            for sample in ("TRAIN", "VALIDATION", "HOLDOUT")
        )
        flattened.update(
            {
                "chronological_gate": "PASS" if qualified else "FAIL",
                "status": "QUALIFIED_DIRECTIONAL" if qualified else "RESEARCH_PATTERN",
                "blockers": ";".join(sorted(set(blockers))) or "",
                "approval_status": "STOCK_RESEARCH_ONLY" if qualified else "NOT_APPROVED",
            }
        )
        output.append(flattened)
    return output


def load_quotes_for_session(
    session: str,
    source_index: Mapping[str, Sequence[ChainSource]],
    source_cutoff: Optional[str] = None,
    underlying_prices: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Load valid bid/ask/OI rows whose actual quote date is ``session``."""

    sources = [
        source
        for source in source_index.get(session, ())
        if source_cutoff is None or source.source_date <= source_cutoff
    ]
    parts: List[pd.DataFrame] = []
    columns = [
        "option_symbol",
        "last_date",
        "last_bid",
        "last_ask",
        "curr_oi",
        "stock_price",
        "dte",
    ]
    for source in sources:
        if source.kind == "hot":
            frame = _read_zip(
                Path(source.path),
                ["option_symbol", "date", "bid", "ask", "open_interest", "volume"],
            )
            if frame.empty or "option_symbol" not in frame.columns:
                continue
            frame = frame.rename(
                columns={
                    "date": "last_date",
                    "bid": "last_bid",
                    "ask": "last_ask",
                    "open_interest": "curr_oi",
                }
            )
            frame["stock_price"] = np.nan
            frame["dte"] = np.nan
        else:
            frame = _read_zip(Path(source.path), columns)
        if frame.empty or "option_symbol" not in frame.columns:
            continue
        frame["last_date"] = frame["last_date"].astype(str).str.slice(0, 10)
        frame = frame[frame["last_date"].eq(session)].copy()
        if frame.empty:
            continue
        for column in ["last_bid", "last_ask", "curr_oi", "stock_price", "dte"]:
            if column not in frame.columns:
                frame[column] = np.nan
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["option_symbol"] = frame["option_symbol"].astype(str).str.upper().str.replace(" ", "", regex=False)
        parsed = frame["option_symbol"].map(parse_occ)
        frame["ticker"] = parsed.map(lambda row: row.get("ticker") if row else "")
        frame["expiry"] = parsed.map(lambda row: row.get("expiry") if row else "")
        frame["option_type"] = parsed.map(lambda row: row.get("option_type") if row else "")
        frame["strike"] = parsed.map(lambda row: row.get("strike") if row else np.nan)
        if source.kind == "hot":
            expiry_dates = pd.to_datetime(frame["expiry"], errors="coerce")
            session_date = pd.Timestamp(session)
            frame["dte"] = (expiry_dates - session_date).dt.days
        if underlying_prices is not None:
            fallback = frame["ticker"].map(underlying_prices)
            frame["stock_price"] = frame["stock_price"].fillna(fallback)
        frame = frame[
            frame["ticker"].ne("")
            & frame["last_ask"].gt(0)
            & frame["last_bid"].ge(0)
            & frame["last_ask"].ge(frame["last_bid"])
            & frame["dte"].gt(0)
        ].copy()
        frame["spread_pct"] = (
            (frame["last_ask"] - frame["last_bid"]) / frame["last_ask"]
        )
        frame["source_date"] = source.source_date
        frame["source_kind"] = source.kind
        frame["source_priority"] = 0 if source.kind == "hot" else 1
        parts.append(frame)
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    return result.sort_values(
        ["option_symbol", "source_priority", "source_date"]
    ).drop_duplicates(
        "option_symbol", keep="first"
    ).reset_index(drop=True)


class QuoteCache:
    """Small bounded cache so a long replay does not retain every chain."""

    def __init__(
        self,
        source_index: Mapping[str, Sequence[ChainSource]],
        source_cutoff: Optional[str] = None,
        underlying_prices_by_date: Optional[Mapping[str, Mapping[str, float]]] = None,
        max_entries: int = 8,
        materialized_dir: Optional[Path] = None,
        cache_key: str = "default",
    ) -> None:
        self.source_index = source_index
        self.source_cutoff = source_cutoff
        self.underlying_prices_by_date = underlying_prices_by_date or {}
        self.max_entries = max_entries
        self._cache: Dict[str, pd.DataFrame] = {}
        self.load_counts: Dict[str, int] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.materialized_dir: Optional[Path] = None
        if materialized_dir is not None:
            signature_rows = []
            for session, sources in sorted(source_index.items()):
                for source in sources:
                    path = Path(source.path)
                    try:
                        stat = path.stat()
                        signature = {
                            "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns,
                        }
                    except FileNotFoundError:
                        signature = {"size": None, "mtime_ns": None}
                    signature_rows.append(
                        {
                            "session": session,
                            "path": source.path,
                            "source_date": source.source_date,
                            "kind": source.kind,
                            **signature,
                        }
                    )
            signature_payload = json.dumps(
                {
                    "cache_version": "managed_quotes_v1",
                    "source_cutoff": source_cutoff,
                    "cache_key": cache_key,
                    "sources": signature_rows,
                },
                sort_keys=True,
            ).encode("utf-8")
            digest = hashlib.sha256(signature_payload).hexdigest()[:20]
            self.materialized_dir = Path(materialized_dir) / f"{cache_key}-{digest}"
            self.materialized_dir.mkdir(parents=True, exist_ok=True)
            manifest = {
                "cache_version": "managed_quotes_v1",
                "cache_key": cache_key,
                "source_signature": digest,
                "source_cutoff": source_cutoff,
                "source_count": len(signature_rows),
            }
            manifest_path = self.materialized_dir / "manifest.json"
            if not manifest_path.exists():
                manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, session: str) -> pd.DataFrame:
        if session not in self._cache:
            cached_path = (
                self.materialized_dir / f"quotes-{session}.pkl.gz"
                if self.materialized_dir is not None
                else None
            )
            if cached_path is not None and cached_path.exists():
                try:
                    self._cache[session] = pd.read_pickle(cached_path, compression="gzip")
                    self.cache_hits += 1
                except (OSError, EOFError, ValueError, ImportError):
                    self._cache[session] = load_quotes_for_session(
                        session,
                        self.source_index,
                        source_cutoff=self.source_cutoff,
                        underlying_prices=self.underlying_prices_by_date.get(session),
                    )
                    self._cache[session].to_pickle(cached_path, compression="gzip")
                    self.cache_misses += 1
            else:
                self._cache[session] = load_quotes_for_session(
                    session,
                    self.source_index,
                    source_cutoff=self.source_cutoff,
                    underlying_prices=self.underlying_prices_by_date.get(session),
                )
                if cached_path is not None:
                    self._cache[session].to_pickle(cached_path, compression="gzip")
                self.cache_misses += 1
            self.load_counts[session] = len(self._cache[session])
            while len(self._cache) > self.max_entries:
                self._cache.pop(next(iter(self._cache)))
        return self._cache[session]


def _signals_for_day(
    panel_day: pd.DataFrame,
    config: ManagedConfig,
) -> Dict[str, Set[str]]:
    """Return per-sector signal names from an EOD cross-sectional rank."""

    market_filter = str(config.market_filter or "ANY").upper()
    if market_filter != "ANY":
        market = panel_day[panel_day["ticker"].astype(str).eq("SPY")]
        if market.empty:
            return {}
        market_row = market.iloc[-1]
        return_5d = as_float(market_row.get("return_5d"))
        return_20d = as_float(market_row.get("return_20d"))
        filter_pass = {
            "SPY_5D_DOWN_OR_FLAT": return_5d is not None and return_5d <= 0.01,
            "SPY_5D_UP_OR_FLAT": return_5d is not None and return_5d >= -0.01,
            "SPY_20D_BULL": return_20d is not None and return_20d > 0.03,
            "SPY_20D_NOT_BEAR": return_20d is not None and return_20d >= -0.03,
            "SPY_20D_BEAR_OR_FLAT": return_20d is not None and return_20d <= 0.03,
        }.get(market_filter)
        if filter_pass is not True:
            return {}

    signal_direction = config.signal_direction or config.direction
    eligible = panel_day[
        panel_day["issue_type"].isin(OPTIONABLE_ISSUE_TYPES)
        & _marketcap_eligible(panel_day, config)
        & panel_day["avg30_volume"].ge(config.min_avg30_volume)
        & panel_day["position_52w"].notna()
        & panel_day["close"].gt(0)
    ].copy()
    eligible["signal_sector"] = _signal_sector(eligible)
    if config.max_iv_rank is not None:
        eligible = eligible[eligible["iv_rank"].le(config.max_iv_rank)]
    selected: Dict[str, Set[str]] = {}
    for sector, block in eligible.groupby("signal_sector", sort=True):
        if len(block) < config.min_sector_names:
            continue
        if config.signal_rule == "post_event_mean_reversion":
            if signal_direction == "call":
                # A bullish reversion follows an outsized down move.
                event = block["return_1d"].le(-abs(config.event_move))
                event &= block["position_52w"].le(config.event_position_limit)
            else:
                # A bearish reversion follows an outsized up move.  The old
                # implementation reused the down-event condition for puts,
                # making the bearish lane directionally wrong.
                event = block["return_1d"].ge(abs(config.event_move))
                event &= block["position_52w"].ge(1.0 - config.event_position_limit)
            names = block.loc[event, "ticker"]
        elif config.signal_rule == "flow_directional":
            if signal_direction == "call":
                flow = block["flow_bias"].ge(config.min_flow_bias)
            else:
                flow = block["flow_bias"].le(-config.min_flow_bias)
            flow &= block["flow_volume_ratio"].ge(config.min_flow_volume_ratio)
            names = block.loc[flow, "ticker"]
        elif config.signal_rule == "flow_quantile":
            ranked = block.dropna(subset=["flow_bias"]).copy()
            flow_rank = ranked["flow_bias"].rank(method="first", pct=True)
            if signal_direction == "call":
                names = ranked.loc[flow_rank.ge(config.top_quantile), "ticker"]
            else:
                names = ranked.loc[flow_rank.le(1.0 - config.top_quantile), "ticker"]
        elif config.signal_rule == "bot_flow_quantile":
            if "bot_flow_bias" not in block.columns:
                names = pd.Series(dtype=str)
            else:
                ranked = block.dropna(subset=["bot_flow_bias"]).copy()
                if "bot_flow_total_premium" in ranked.columns:
                    ranked = ranked[
                        ranked["bot_flow_total_premium"].ge(config.min_bot_flow_premium)
                    ]
                flow_rank = ranked["bot_flow_bias"].rank(method="first", pct=True)
                if signal_direction == "call":
                    names = ranked.loc[flow_rank.ge(config.top_quantile), "ticker"]
                else:
                    names = ranked.loc[flow_rank.le(1.0 - config.top_quantile), "ticker"]
        elif config.signal_rule == "earnings_flow":
            ranked = block.dropna(subset=["days_to_earnings", "implied_move_perc", "flow_bias"]).copy()
            ranked = ranked[
                ranked["days_to_earnings"].between(
                    config.earnings_min_days,
                    config.earnings_max_days,
                    inclusive="both",
                )
                & ranked["implied_move_perc"].ge(config.min_implied_move_perc)
            ]
            flow_rank = ranked["flow_bias"].rank(method="first", pct=True)
            if signal_direction == "call":
                names = ranked.loc[flow_rank.ge(config.top_quantile), "ticker"]
            else:
                names = ranked.loc[flow_rank.le(1.0 - config.top_quantile), "ticker"]
        elif config.signal_rule == "earnings_event":
            ranked = block.dropna(subset=["days_to_earnings", "implied_move_perc"]).copy()
            ranked = ranked[
                ranked["days_to_earnings"].between(
                    config.earnings_min_days,
                    config.earnings_max_days,
                    inclusive="both",
                )
                & ranked["implied_move_perc"].ge(config.min_implied_move_perc)
            ]
            event_rank = ranked["implied_move_perc"].rank(method="first", pct=True)
            names = ranked.loc[event_rank.ge(config.top_quantile), "ticker"]
        elif config.signal_rule in {"momentum_5", "momentum_20"}:
            return_column = "return_5d" if config.signal_rule == "momentum_5" else "return_20d"
            ranked = block.dropna(subset=[return_column]).copy()
            ranks = ranked[return_column].rank(method="first", pct=True)
            if signal_direction == "call":
                names = ranked.loc[ranks.ge(config.top_quantile), "ticker"]
            else:
                names = ranked.loc[ranks.le(1.0 - config.top_quantile), "ticker"]
        elif config.signal_rule == "momentum_flow":
            ranked = block.dropna(subset=["return_5d", "flow_bias"]).copy()
            momentum_rank = ranked["return_5d"].rank(method="first", pct=True)
            flow_rank = ranked["flow_bias"].rank(method="first", pct=True)
            if signal_direction == "call":
                matched = momentum_rank.ge(config.top_quantile) & flow_rank.ge(config.top_quantile)
            else:
                matched = momentum_rank.le(1.0 - config.top_quantile) & flow_rank.le(
                    1.0 - config.top_quantile
                )
            names = ranked.loc[matched, "ticker"]
        elif config.signal_rule in {"momentum_flow_composite", "trend_flow_composite"}:
            first_column = (
                "return_5d" if config.signal_rule == "momentum_flow_composite" else "position_52w"
            )
            ranked = block.dropna(subset=[first_column, "flow_bias"]).copy()
            first_rank = ranked[first_column].rank(method="first", pct=True)
            flow_rank = ranked["flow_bias"].rank(method="first", pct=True)
            composite_rank = (first_rank + flow_rank).rank(method="first", pct=True)
            if signal_direction == "call":
                names = ranked.loc[composite_rank.ge(config.top_quantile), "ticker"]
            else:
                names = ranked.loc[composite_rank.le(1.0 - config.top_quantile), "ticker"]
        elif config.signal_rule == "trend_flow":
            ranked = block.dropna(subset=["position_52w", "flow_bias"]).copy()
            trend_rank = ranked["position_52w"].rank(method="first", pct=True)
            flow_rank = ranked["flow_bias"].rank(method="first", pct=True)
            if signal_direction == "call":
                matched = trend_rank.ge(config.top_quantile) & flow_rank.ge(config.top_quantile)
            else:
                matched = trend_rank.le(1.0 - config.top_quantile) & flow_rank.le(
                    1.0 - config.top_quantile
                )
            names = ranked.loc[matched, "ticker"]
        else:
            ranks = block["position_52w"].rank(method="first", pct=True)
            if signal_direction == "call":
                names = block.loc[ranks.ge(config.top_quantile), "ticker"]
            else:
                names = block.loc[ranks.le(1.0 - config.top_quantile), "ticker"]
        selected[str(sector)] = set(names.astype(str))
    return selected


def _random_control_names(
    panel_day: pd.DataFrame,
    selected: Mapping[str, Set[str]],
    config: ManagedConfig,
    seed: int,
) -> Dict[str, Set[str]]:
    """Same sector/date counts as the signal, deterministic and auditable."""

    rng = np.random.default_rng(seed)
    result: Dict[str, Set[str]] = {}
    eligible = panel_day[
        panel_day["issue_type"].isin(OPTIONABLE_ISSUE_TYPES)
        & _marketcap_eligible(panel_day, config)
        & panel_day["avg30_volume"].ge(config.min_avg30_volume)
        & panel_day["position_52w"].notna()
        & panel_day["close"].gt(0)
    ]
    eligible = eligible.copy()
    eligible["signal_sector"] = _signal_sector(eligible)
    if config.max_iv_rank is not None:
        eligible = eligible[eligible["iv_rank"].le(config.max_iv_rank)]
    for sector, names in selected.items():
        pool = (
            eligible.loc[
                eligible["signal_sector"].eq(sector)
                & ~eligible["ticker"].astype(str).isin(names),
                "ticker",
            ]
            .astype(str)
            .unique()
        )
        count = min(len(names), len(pool))
        result[sector] = set(rng.choice(pool, size=count, replace=False).tolist()) if count else set()
    return result


def _signal_sector(frame: pd.DataFrame) -> pd.Series:
    """Provide a deterministic bucket for optionable rows without a sector."""

    sector = frame["sector"].fillna("").astype(str).str.strip()
    return sector.mask(
        sector.eq(""),
        np.where(frame["issue_type"].eq("ETF"), "ETF", "UNCLASSIFIED"),
    )


def _marketcap_eligible(frame: pd.DataFrame, config: ManagedConfig) -> pd.Series:
    """Apply market-cap gates without trusting inconsistent ETF semantics.

    The UW screener's ETF ``marketcap`` field changes scale/meaning across
    otherwise valid 2026 rows (for example, early SPY/XLE/XOP values are near
    share price).  ETF optionability is better controlled by the explicit
    price, volume, OI, and spread gates than by that unstable field.
    """

    etf_liquidity = (
        frame["issue_type"].eq("ETF")
        & frame["close"].gt(0)
        & frame["avg30_volume"].ge(config.min_avg30_volume)
    )
    return frame["marketcap"].ge(config.min_marketcap) | etf_liquidity


def _limit_selected_names(
    panel_day: pd.DataFrame,
    selected: Mapping[str, Set[str]],
    config: ManagedConfig,
) -> Dict[str, Set[str]]:
    """Apply an optional deterministic portfolio-cap ranking."""

    limit = config.max_positions_per_day
    if limit is None or limit <= 0:
        return {str(sector): set(names) for sector, names in selected.items()}
    names = {str(name) for values in selected.values() for name in values}
    if not names:
        return {}
    block = panel_day[panel_day["ticker"].astype(str).isin(names)].copy()
    signal_direction = config.signal_direction or config.direction
    if config.signal_rule in {"flow_quantile", "flow_directional"}:
        block["selection_score"] = block["flow_bias"] * (1.0 if signal_direction == "call" else -1.0)
    elif config.signal_rule == "bot_flow_quantile" and "bot_flow_bias" in block.columns:
        block["selection_score"] = block["bot_flow_bias"] * (1.0 if signal_direction == "call" else -1.0)
    elif config.signal_rule == "earnings_flow" and "flow_bias" in block.columns:
        block["selection_score"] = block["flow_bias"] * (1.0 if signal_direction == "call" else -1.0)
    elif config.signal_rule == "earnings_event" and "implied_move_perc" in block.columns:
        block["selection_score"] = block["implied_move_perc"]
    elif config.signal_rule == "momentum_flow_composite":
        block["selection_score"] = (
            block["return_5d"].rank(method="first", pct=True)
            + block["flow_bias"].rank(method="first", pct=True)
        ) * (1.0 if signal_direction == "call" else -1.0)
    elif config.signal_rule == "trend_flow_composite":
        block["selection_score"] = (
            block["position_52w"].rank(method="first", pct=True)
            + block["flow_bias"].rank(method="first", pct=True)
        ) * (1.0 if signal_direction == "call" else -1.0)
    elif config.signal_rule in {"momentum_5", "momentum_flow"}:
        block["selection_score"] = block["return_5d"] * (1.0 if signal_direction == "call" else -1.0)
    elif config.signal_rule == "momentum_20":
        block["selection_score"] = block["return_20d"] * (1.0 if signal_direction == "call" else -1.0)
    elif config.signal_rule == "post_event_mean_reversion":
        block["selection_score"] = block["return_1d"] * (-1.0 if signal_direction == "call" else 1.0)
    else:
        block["selection_score"] = block["position_52w"] * (1.0 if signal_direction == "call" else -1.0)
    block = block.sort_values(["selection_score", "ticker"], ascending=[False, True], na_position="last")
    keep = set(block.head(int(limit))["ticker"].astype(str))
    return {
        str(sector): {name for name in values if name in keep}
        for sector, values in selected.items()
        if any(name in keep for name in values)
    }


def _select_contracts(
    quotes: pd.DataFrame,
    names_by_sector: Mapping[str, Set[str]],
    config: ManagedConfig,
    held: Set[str],
) -> List[Dict[str, Any]]:
    if quotes.empty:
        return []
    required_columns = {
        "ticker",
        "option_type",
        "dte",
        "curr_oi",
        "spread_pct",
        "last_ask",
        "last_bid",
        "strike",
        "stock_price",
    }
    if not required_columns.issubset(quotes.columns):
        return []
    names = set().union(*names_by_sector.values()) if names_by_sector else set()
    option_type = config.option_type or ("C" if config.direction == "call" else "P")
    bid_floor = (
        0.0
        if config.structure in {"cash_secured_put", "credit_vertical", "iron_condor"}
        else config.min_entry_bid
    )
    option_types = (
        {"C", "P"}
        if config.structure in {"long_straddle", "iron_condor"}
        else {option_type}
    )
    candidates = quotes[
        quotes["ticker"].isin(names)
        & quotes["option_type"].isin(option_types)
        & quotes["dte"].between(config.min_dte, config.max_dte)
        & quotes["curr_oi"].ge(config.min_open_interest)
        & quotes["spread_pct"].le(config.max_spread_pct)
        & quotes["last_ask"].gt(0)
        & quotes["last_bid"].ge(bid_floor)
    ].copy()
    if config.one_per_ticker:
        candidates = candidates[~candidates["ticker"].isin(held)]
    if candidates.empty:
        return []
    sector_by_ticker = {
        ticker: sector
        for sector, names in names_by_sector.items()
        for ticker in names
    }
    rows: List[Dict[str, Any]] = []
    if config.structure == "long_straddle":
        for ticker, ticker_block in candidates.groupby("ticker", sort=True):
            pairs: List[Tuple[float, float, float, str, pd.Series, pd.Series]] = []
            for expiry, expiry_block in ticker_block.groupby("expiry", sort=True):
                call_block = expiry_block[expiry_block["option_type"].eq("C")].copy()
                put_block = expiry_block[expiry_block["option_type"].eq("P")].copy()
                if call_block.empty or put_block.empty:
                    continue
                stock_price = float(expiry_block["stock_price"].iloc[0])
                call_block["strike_gap"] = (
                    call_block["strike"] - stock_price * config.moneyness
                ).abs()
                put_block["strike_gap"] = (
                    put_block["strike"] - stock_price * config.moneyness
                ).abs()
                call_row = call_block.sort_values(
                    ["strike_gap", "spread_pct", "option_symbol"]
                ).iloc[0]
                put_row = put_block.sort_values(
                    ["strike_gap", "spread_pct", "option_symbol"]
                ).iloc[0]
                pairs.append(
                    (
                        abs(float(call_row["dte"]) - config.target_dte),
                        float(call_row["strike_gap"]),
                        float(put_row["strike_gap"]),
                        str(expiry),
                        call_row,
                        put_row,
                    )
                )
            if not pairs:
                continue
            _, _, _, _, call_row, put_row = min(pairs, key=lambda item: item[:4])
            entry_ask = float(call_row["last_ask"]) + float(put_row["last_ask"])
            entry_bid = float(call_row["last_bid"]) + float(put_row["last_bid"])
            if entry_ask <= 0 or entry_bid < 2.0 * bid_floor:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "sector": sector_by_ticker.get(ticker, ""),
                    "option_symbol": call_row["option_symbol"],
                    "second_option_symbol": put_row["option_symbol"],
                    "option_type": "C",
                    "strike": float(call_row["strike"]),
                    "put_strike": float(put_row["strike"]),
                    "expiry": call_row["expiry"],
                    "dte": float(call_row["dte"]),
                    "entry_bid": entry_bid,
                    "entry_ask": entry_ask,
                    "entry_spread_pct": max(
                        float(call_row["spread_pct"]),
                        float(put_row["spread_pct"]),
                    ),
                    "entry_oi": min(
                        float(call_row["curr_oi"]),
                        float(put_row["curr_oi"]),
                    ),
                    "underlying_price": float(call_row["stock_price"]),
                    "source_date": call_row.get("source_date", ""),
                    "structure": "long_straddle",
                }
            )
        return rows
    if config.structure == "iron_condor":
        for ticker, ticker_block in candidates.groupby("ticker", sort=True):
            condors: List[Tuple[float, float, str, Dict[str, Any]]] = []
            for expiry, expiry_block in ticker_block.groupby("expiry", sort=True):
                calls = expiry_block[expiry_block["option_type"].eq("C")].copy()
                puts = expiry_block[expiry_block["option_type"].eq("P")].copy()
                if calls.empty or puts.empty:
                    continue
                stock_price = float(expiry_block["stock_price"].iloc[0])
                short_puts = puts[puts["strike"].lt(stock_price)].copy()
                short_calls = calls[calls["strike"].gt(stock_price)].copy()
                if short_puts.empty or short_calls.empty:
                    continue
                short_put = short_puts.iloc[
                    (short_puts["strike"] - stock_price * config.short_put_moneyness)
                    .abs()
                    .argmin()
                ]
                short_call = short_calls.iloc[
                    (short_calls["strike"] - stock_price * config.short_call_moneyness)
                    .abs()
                    .argmin()
                ]
                long_puts = puts[puts["strike"].lt(float(short_put["strike"]))].copy()
                long_calls = calls[calls["strike"].gt(float(short_call["strike"]))].copy()
                if long_puts.empty or long_calls.empty:
                    continue
                long_put = long_puts.iloc[
                    (long_puts["strike"] - stock_price * config.long_put_moneyness)
                    .abs()
                    .argmin()
                ]
                long_call = long_calls.iloc[
                    (long_calls["strike"] - stock_price * config.long_call_moneyness)
                    .abs()
                    .argmin()
                ]
                entry_credit = (
                    float(short_put["last_bid"])
                    + float(short_call["last_bid"])
                    - float(long_put["last_ask"])
                    - float(long_call["last_ask"])
                )
                if entry_credit <= 0:
                    continue
                put_width = float(short_put["strike"]) - float(long_put["strike"])
                call_width = float(long_call["strike"]) - float(short_call["strike"])
                if put_width <= 0 or call_width <= 0:
                    continue
                leg_data = {
                    "ticker": ticker,
                    "sector": sector_by_ticker.get(ticker, ""),
                    "option_symbol": long_put["option_symbol"],
                    "short_option_symbol": short_put["option_symbol"],
                    "second_option_symbol": short_call["option_symbol"],
                    "long_call_option_symbol": long_call["option_symbol"],
                    "option_type": "P",
                    "strike": float(long_put["strike"]),
                    "short_strike": float(short_put["strike"]),
                    "short_call_strike": float(short_call["strike"]),
                    "long_call_strike": float(long_call["strike"]),
                    "expiry": short_put["expiry"],
                    "dte": float(short_put["dte"]),
                    "entry_bid": entry_credit,
                    "entry_ask": entry_credit,
                    "entry_credit": entry_credit,
                    "spread_width": max(put_width, call_width),
                    "entry_spread_pct": max(
                        float(long_put["spread_pct"]),
                        float(short_put["spread_pct"]),
                        float(short_call["spread_pct"]),
                        float(long_call["spread_pct"]),
                    ),
                    "entry_oi": min(
                        float(long_put["curr_oi"]),
                        float(short_put["curr_oi"]),
                        float(short_call["curr_oi"]),
                        float(long_call["curr_oi"]),
                    ),
                    "underlying_price": stock_price,
                    "source_date": short_put.get("source_date", ""),
                    "structure": "iron_condor",
                }
                condors.append(
                    (
                        abs(float(short_put["dte"]) - config.target_dte),
                        -entry_credit,
                        str(expiry),
                        leg_data,
                    )
                )
            if condors:
                rows.append(min(condors, key=lambda item: item[:3])[3])
        return rows
    for ticker, block in candidates.groupby("ticker", sort=True):
        block = block.copy()
        if config.structure == "cash_secured_put":
            short_block = block[
                block["option_type"].eq("P")
                & block["strike"].lt(block["stock_price"])
            ].copy()
            short_block["strike_gap"] = (
                short_block["strike"] - short_block["stock_price"] * config.moneyness
            ).abs()
            short_block["dte_gap"] = (short_block["dte"] - config.target_dte).abs()
            short_block = short_block.sort_values(
                ["dte_gap", "strike_gap", "option_symbol"]
            )
            if short_block.empty:
                continue
            short_row = short_block.iloc[0]
            entry_credit = float(short_row["last_bid"])
            if entry_credit <= 0:
                continue
            collateral_per_share = float(short_row["strike"])
            close_fee = 2.0 * config.fee_per_side
            rows.append(
                {
                    "ticker": ticker,
                    "sector": sector_by_ticker.get(ticker, ""),
                    "option_symbol": short_row["option_symbol"],
                    "option_type": "P",
                    "position_side": "short",
                    "strike": float(short_row["strike"]),
                    "expiry": short_row["expiry"],
                    "dte": float(short_row["dte"]),
                    "entry_bid": entry_credit,
                    "entry_ask": entry_credit,
                    "entry_credit": entry_credit,
                    "collateral_per_share": collateral_per_share,
                    "cash_collateral": collateral_per_share * 100.0,
                    "max_loss_to_zero": max(
                        0.0,
                        (collateral_per_share - entry_credit) * 100.0 + close_fee,
                    ),
                    "entry_spread_pct": float(short_row["spread_pct"]),
                    "entry_oi": float(short_row["curr_oi"]),
                    "underlying_price": float(short_row["stock_price"]),
                    "source_date": short_row.get("source_date", ""),
                    "structure": "cash_secured_put",
                }
            )
            continue
        if config.structure in {"debit_vertical", "credit_vertical"}:
            if config.structure == "credit_vertical":
                # A credit spread's short option must be OTM at entry.  Without
                # this invariant, proximity to ``moneyness`` can silently turn
                # a bull put into an ATM/ITM short, or a bear call into an
                # ATM/ITM short, changing the intended risk profile.
                if option_type == "P":
                    short_block = block[block["strike"].lt(block["stock_price"])].copy()
                else:
                    short_block = block[block["strike"].gt(block["stock_price"])].copy()
                short_block["strike_gap"] = (
                    short_block["strike"] - short_block["stock_price"] * config.moneyness
                ).abs()
                short_block["dte_gap"] = (short_block["dte"] - config.target_dte).abs()
                short_block = short_block.sort_values(
                    ["dte_gap", "strike_gap", "option_symbol"]
                )
                if short_block.empty:
                    continue
                short_row = short_block.iloc[0]
                long_block = block[block["expiry"].eq(short_row["expiry"])].copy()
                if option_type == "C":
                    long_block = long_block[long_block["strike"].gt(short_row["strike"])]
                else:
                    long_block = long_block[long_block["strike"].lt(short_row["strike"])]
                if long_block.empty:
                    continue
                long_block["strike_gap"] = (
                    long_block["strike"] - long_block["stock_price"] * config.short_moneyness
                ).abs()
                long_block["dte_gap"] = (long_block["dte"] - config.target_dte).abs()
                long_row = long_block.sort_values(
                    ["dte_gap", "strike_gap", "option_symbol"]
                ).iloc[0]
                entry_credit = float(short_row["last_bid"]) - float(long_row["last_ask"])
                if entry_credit <= 0:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "sector": sector_by_ticker.get(ticker, ""),
                        "option_symbol": long_row["option_symbol"],
                        "short_option_symbol": short_row["option_symbol"],
                        "option_type": option_type,
                        "strike": float(long_row["strike"]),
                        "short_strike": float(short_row["strike"]),
                        "expiry": short_row["expiry"],
                        "dte": float(short_row["dte"]),
                        "entry_bid": entry_credit,
                        "entry_ask": entry_credit,
                        "entry_credit": entry_credit,
                        "spread_width": abs(float(short_row["strike"]) - float(long_row["strike"])),
                        "entry_spread_pct": max(float(long_row["spread_pct"]), float(short_row["spread_pct"])),
                        "entry_oi": min(float(long_row["curr_oi"]), float(short_row["curr_oi"])),
                        "underlying_price": float(short_row["stock_price"]),
                        "source_date": short_row.get("source_date", ""),
                        "structure": "credit_vertical",
                    }
                )
                continue
            long_block = block.copy()
            long_block["strike_gap"] = (
                long_block["strike"] - long_block["stock_price"] * config.moneyness
            ).abs()
            long_block["dte_gap"] = (long_block["dte"] - config.target_dte).abs()
            long_block = long_block.sort_values(
                ["dte_gap", "strike_gap", "option_symbol"]
            )
            if long_block.empty:
                continue
            long_row = long_block.iloc[0]
            short_block = block[block["expiry"].eq(long_row["expiry"])].copy()
            if config.direction == "call":
                short_block = short_block[short_block["strike"].gt(long_row["strike"])]
            else:
                short_block = short_block[short_block["strike"].lt(long_row["strike"])]
            if short_block.empty:
                continue
            short_block["strike_gap"] = (
                short_block["strike"] - short_block["stock_price"] * config.short_moneyness
            ).abs()
            short_block["dte_gap"] = (short_block["dte"] - config.target_dte).abs()
            short_row = short_block.sort_values(
                ["dte_gap", "strike_gap", "option_symbol"]
            ).iloc[0]
            entry_debit = float(long_row["last_ask"]) - float(short_row["last_bid"])
            if entry_debit <= 0:
                continue
            rows.append(
                {
                    "ticker": ticker,
                    "sector": sector_by_ticker.get(ticker, ""),
                    "option_symbol": long_row["option_symbol"],
                    "short_option_symbol": short_row["option_symbol"],
                    "option_type": long_row["option_type"],
                    "strike": float(long_row["strike"]),
                    "short_strike": float(short_row["strike"]),
                    "expiry": long_row["expiry"],
                    "dte": float(long_row["dte"]),
                    "entry_bid": float(long_row["last_bid"]) - float(short_row["last_ask"]),
                    "entry_ask": entry_debit,
                    "entry_debit": entry_debit,
                    "entry_spread_pct": max(float(long_row["spread_pct"]), float(short_row["spread_pct"])),
                    "entry_oi": min(float(long_row["curr_oi"]), float(short_row["curr_oi"])),
                    "underlying_price": float(long_row["stock_price"]),
                    "source_date": long_row.get("source_date", ""),
                    "structure": "debit_vertical",
                }
            )
            continue
        block["strike_gap"] = (block["strike"] - block["stock_price"] * config.moneyness).abs()
        block["dte_gap"] = (block["dte"] - config.target_dte).abs()
        row = block.sort_values(
            ["dte_gap", "strike_gap", "option_symbol"]
        ).iloc[0]
        rows.append(
            {
                "ticker": ticker,
                "sector": sector_by_ticker.get(ticker, ""),
                "option_symbol": row["option_symbol"],
                "option_type": row["option_type"],
                "strike": float(row["strike"]),
                "expiry": row["expiry"],
                "dte": float(row["dte"]),
                "entry_bid": float(row["last_bid"]),
                "entry_ask": float(row["last_ask"]),
                "entry_spread_pct": float(row["spread_pct"]),
                "entry_oi": float(row["curr_oi"]),
                "underlying_price": float(row["stock_price"]),
                "source_date": row.get("source_date", ""),
                "structure": "long_option",
            }
        )
    return rows


def _position_result(
    position: Mapping[str, Any],
    status: str,
    exit_date: str = "",
    exit_bid: Optional[float] = None,
    exit_reason: str = "",
    config: Optional[ManagedConfig] = None,
    exit_underlying_price: Optional[float] = None,
) -> Dict[str, Any]:
    config = config or ManagedConfig()
    structure = str(position.get("structure") or "long_option")
    entry_ask = float(position.get("entry_ask") or 0.0)
    entry_credit = float(position.get("entry_credit") or 0.0)
    fee = (
        8.0
        if structure == "iron_condor"
        else 4.0
        if structure in {"debit_vertical", "credit_vertical", "long_straddle"}
        else 2.0
    ) * config.fee_per_side
    row = {
        key: value
        for key, value in position.items()
        if not str(key).startswith("_")
    }
    row.update(
        {
            "status": status,
            "exit_date": exit_date,
            "exit_bid": exit_bid,
            "exit_value": exit_bid,
            "exit_reason": exit_reason,
            "exit_underlying_price": exit_underlying_price,
            "gross_pnl": None,
            "net_pnl": None,
            "net_R": None,
            "win": None,
        }
    )
    if status != "SCORED" or exit_bid is None or not math.isfinite(float(exit_bid)):
        return row
    if structure in {"credit_vertical", "cash_secured_put", "iron_condor"}:
        gross_pnl = (entry_credit - float(exit_bid)) * 100.0
        if structure == "cash_secured_put":
            risk = (
                float(position.get("collateral_per_share") or position.get("strike") or 0.0)
                * 100.0
                + fee
            )
        else:
            risk = max(0.0, float(position.get("spread_width") or 0.0) - entry_credit) * 100.0 + fee
    else:
        gross_pnl = (float(exit_bid) - entry_ask) * 100.0
        risk = entry_ask * 100.0 + fee
    net_pnl = gross_pnl - fee
    row.update(
        {
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "net_R": net_pnl / risk if risk > 0 else None,
            "win": bool(net_pnl > 0),
        }
    )
    return row


def _intrinsic_value(
    option_type: str,
    strike: Any,
    underlying_price: Any,
) -> Optional[float]:
    """Return expiration settlement value per share for one option leg."""

    option = str(option_type or "").upper()
    strike_value = as_float(strike)
    underlying = as_float(underlying_price)
    if option not in {"C", "P"} or strike_value is None or underlying is None:
        return None
    if option == "C":
        return max(0.0, underlying - strike_value)
    return max(0.0, strike_value - underlying)


def _expiration_exit_value(
    position: Mapping[str, Any],
    underlying_price: Any,
) -> Optional[float]:
    """Value a missing quote at expiration without inventing a market mark."""

    long_value = _intrinsic_value(
        str(position.get("option_type") or ""),
        position.get("strike"),
        underlying_price,
    )
    if long_value is None:
        return None
    structure = str(position.get("structure") or "long_option")
    if structure == "long_option":
        return long_value
    if structure == "long_straddle":
        put_value = _intrinsic_value("P", position.get("put_strike"), underlying_price)
        return long_value + put_value if put_value is not None else None
    if structure == "iron_condor":
        short_put_value = _intrinsic_value(
            "P", position.get("short_strike"), underlying_price
        )
        long_put_value = _intrinsic_value(
            "P", position.get("strike"), underlying_price
        )
        short_call_value = _intrinsic_value(
            "C", position.get("short_call_strike"), underlying_price
        )
        long_call_value = _intrinsic_value(
            "C", position.get("long_call_strike"), underlying_price
        )
        if None in {
            short_put_value,
            long_put_value,
            short_call_value,
            long_call_value,
        }:
            return None
        return max(0.0, short_put_value - long_put_value) + max(
            0.0, short_call_value - long_call_value
        )
    short_strike = (
        position.get("strike")
        if structure == "cash_secured_put"
        else position.get("short_strike")
    )
    short_value = _intrinsic_value(
        str(position.get("option_type") or ""),
        short_strike,
        underlying_price,
    )
    if short_value is None:
        return None
    if structure == "cash_secured_put":
        return short_value
    if structure == "debit_vertical":
        return max(0.0, long_value - short_value)
    if structure == "credit_vertical":
        return max(0.0, short_value - long_value)
    return None


def _expiration_underlying_price(
    position: Mapping[str, Any],
    session: str,
    underlying_prices_by_date: Mapping[str, Mapping[str, Any]],
) -> Optional[float]:
    """Use the last known close on or before expiration, never a future close."""

    expiry = str(position.get("expiry") or "")[:10]
    if not DATE_RE.fullmatch(expiry) or session < expiry:
        return None
    eligible_dates = [
        str(date)
        for date in underlying_prices_by_date
        if str(date) <= expiry and str(date) <= session
    ]
    if not eligible_dates:
        return None
    settlement_date = max(eligible_dates)
    return as_float(
        underlying_prices_by_date[settlement_date].get(str(position.get("ticker") or ""))
    )


def _advance_state(
    state: Dict[str, Any],
    current_bid: pd.Series,
    current_ask: pd.Series,
    session: str,
    index: int,
    config: ManagedConfig,
    underlying_prices_by_date: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> None:
    still_open: List[Dict[str, Any]] = []
    for position in state["open_positions"]:
        age = index - int(position["entry_index"])
        structure = str(position.get("structure") or "long_option")
        if structure == "long_straddle":
            long_bid = current_bid.get(position["option_symbol"])
            put_bid = current_bid.get(position["second_option_symbol"])
            if (
                long_bid is not None
                and put_bid is not None
                and pd.notna(long_bid)
                and pd.notna(put_bid)
            ):
                current = float(long_bid) + float(put_bid)
            else:
                current = None
        elif structure == "cash_secured_put":
            current = current_ask.get(position["option_symbol"])
        elif structure == "iron_condor":
            long_put_bid = current_bid.get(position["option_symbol"])
            short_put_ask = current_ask.get(position["short_option_symbol"])
            short_call_ask = current_ask.get(position["second_option_symbol"])
            long_call_bid = current_bid.get(position["long_call_option_symbol"])
            if all(
                value is not None and pd.notna(value)
                for value in (
                    long_put_bid,
                    short_put_ask,
                    short_call_ask,
                    long_call_bid,
                )
            ):
                current = max(
                    0.0,
                    float(short_put_ask)
                    + float(short_call_ask)
                    - float(long_put_bid)
                    - float(long_call_bid),
                )
            else:
                current = None
        elif structure in {"debit_vertical", "credit_vertical"}:
            long_bid = current_bid.get(position["option_symbol"])
            short_ask = current_ask.get(position["short_option_symbol"])
            if (
                long_bid is not None
                and short_ask is not None
                and pd.notna(long_bid)
                and pd.notna(short_ask)
            ):
                if structure == "debit_vertical":
                    current = max(0.0, float(long_bid) - float(short_ask))
                else:
                    current = max(0.0, float(short_ask) - float(long_bid))
            else:
                current = None
        else:
            current = current_bid.get(position["option_symbol"])
        current_value = float(current) if current is not None and pd.notna(current) else None
        if current_value is not None and index > int(position["entry_index"]):
            position["_last_observed_value"] = current_value
            position["_last_observed_session"] = session
            position["_last_observed_index"] = index
        if current_value is None and underlying_prices_by_date:
            expiration_price = _expiration_underlying_price(
                position,
                session,
                underlying_prices_by_date,
            )
            expiration_value = _expiration_exit_value(position, expiration_price)
            if expiration_value is not None:
                state["results"].append(
                    _position_result(
                        position,
                        "SCORED",
                        session,
                        expiration_value,
                        "expiration_intrinsic",
                        config,
                        expiration_price,
                    )
                )
                state["held"].discard(position["ticker"])
                continue
        if current_value is None and age >= config.max_hold_sessions:
            last_index = position.get("_last_observed_index")
            last_value = as_float(position.get("_last_observed_value"))
            last_session = str(position.get("_last_observed_session") or "")
            try:
                quote_gap = index - int(last_index)
            except (TypeError, ValueError):
                quote_gap = math.inf
            if (
                last_value is not None
                and DATE_RE.fullmatch(last_session)
                and 0 < quote_gap <= max(0, int(config.max_exit_quote_gap_sessions))
            ):
                state["results"].append(
                    _position_result(
                        position,
                        "SCORED",
                        last_session,
                        last_value,
                        "time_stop_last_observed_quote",
                        config,
                    )
                )
                state["held"].discard(position["ticker"])
                continue
        if current_value is not None:
            if structure in {"cash_secured_put", "credit_vertical", "iron_condor"}:
                entry_value = float(position["entry_credit"])
                gain = 1.0 - current_value / entry_value if entry_value > 0 else -math.inf
                if gain >= config.profit_target:
                    target_value = entry_value * (1.0 - config.profit_target)
                    state["results"].append(
                        _position_result(position, "SCORED", session, target_value, "profit_target", config)
                    )
                    state["held"].discard(position["ticker"])
                    continue
                if config.stop_loss is not None and gain <= -config.stop_loss:
                    state["results"].append(
                        _position_result(position, "SCORED", session, current_value, "stop_loss", config)
                    )
                    state["held"].discard(position["ticker"])
                    continue
            else:
                gain = current_value / float(position["entry_ask"]) - 1.0
                if gain >= config.profit_target:
                    target_price = float(position["entry_ask"]) * (1.0 + config.profit_target)
                    state["results"].append(
                        _position_result(position, "SCORED", session, target_price, "profit_target", config)
                    )
                    state["held"].discard(position["ticker"])
                    continue
                if config.stop_loss is not None and gain <= -config.stop_loss:
                    state["results"].append(
                        _position_result(position, "SCORED", session, current_value, "stop_loss", config)
                    )
                    state["held"].discard(position["ticker"])
                    continue
            if age >= config.max_hold_sessions:
                state["results"].append(
                    _position_result(position, "SCORED", session, current_value, "time_stop", config)
                )
                state["held"].discard(position["ticker"])
                continue
        if age >= config.max_hold_sessions:
            state["results"].append(
                _position_result(
                    position,
                    "MISSING_EXIT_QUOTE",
                    session,
                    exit_reason="missing_time_stop_quote",
                    config=config,
                )
            )
            state["held"].discard(position["ticker"])
        else:
            still_open.append(position)
    state["open_positions"] = still_open


def _open_state_positions(
    state: Dict[str, Any],
    signal_day: pd.DataFrame,
    entry_quotes: pd.DataFrame,
    session: str,
    entry_date: str,
    entry_index: int,
    config: ManagedConfig,
    control: str,
    seed: int,
) -> None:
    selected = _signals_for_day(signal_day, config)
    if control == "random":
        selected = _random_control_names(signal_day, selected, config, seed)
    else:
        selected = _limit_selected_names(signal_day, selected, config)
    if not selected:
        return
    contracts = _select_contracts(entry_quotes, selected, config, state["held"])
    for contract in contracts:
        state["position_id"] += 1
        direction = (
            "bullish"
            if config.direction == "call"
            else "bearish"
            if config.direction == "put"
            else "neutral"
        )
        state["open_positions"].append(
            {
                **contract,
                "position_id": state["position_id"],
                "signal_date": session,
                "entry_date": entry_date,
                "entry_index": entry_index,
                "direction": direction,
                "pattern_family": config.name,
                "control": control,
            }
        )
        state["held"].add(contract["ticker"])


def build_current_option_setups(
    panel: pd.DataFrame,
    session: str,
    quotes: pd.DataFrame,
    config: ManagedConfig,
) -> List[Dict[str, Any]]:
    """Build same-day research tickets without treating them as backtest fills."""

    signal_day = panel[panel["date"].astype(str).eq(session)]
    if signal_day.empty:
        return []
    selected = _limit_selected_names(signal_day, _signals_for_day(signal_day, config), config)
    contracts = _select_contracts(quotes, selected, config, set())
    direction = (
        "bullish"
        if config.direction == "call"
        else "bearish"
        if config.direction == "put"
        else "neutral"
    )
    return [
        {
            **contract,
            "signal_date": session,
            "quote_date": session,
            "entry_date": "",
            "direction": direction,
            "pattern_family": config.name,
            "control": "signal",
            "candidate_timing": "SAME_DAY_EOD_RESEARCH",
        }
        for contract in contracts
    ]


def run_managed_strategies(
    panel: pd.DataFrame,
    sessions: Sequence[str],
    quotes: Callable[[str], pd.DataFrame],
    strategies: Mapping[str, Tuple[ManagedConfig, str]],
    seed: int = 20260821,
) -> Dict[str, pd.DataFrame]:
    """Run several fixed hypotheses while loading each dated quote once."""

    by_date = {str(date): frame for date, frame in panel.groupby("date", sort=False)}
    underlying_prices_by_date = {
        str(date): frame.set_index("ticker")["close"].to_dict()
        for date, frame in panel.groupby("date", sort=False)
    }
    states = {
        key: {
            "config": config,
            "control": control,
            "open_positions": [],
            "held": set(),
            "results": [],
            "position_id": 0,
        }
        for key, (config, control) in strategies.items()
    }
    for index, session in enumerate(sessions):
        current_quotes = quotes(session)
        current_bid = (
            current_quotes.set_index("option_symbol")["last_bid"]
            if {"option_symbol", "last_bid"}.issubset(current_quotes.columns)
            else pd.Series(dtype=float)
        )
        current_ask = (
            current_quotes.set_index("option_symbol")["last_ask"]
            if {"option_symbol", "last_ask"}.issubset(current_quotes.columns)
            else pd.Series(dtype=float)
        )
        for state in states.values():
            _advance_state(
                state,
                current_bid,
                current_ask,
                session,
                index,
                state["config"],
                underlying_prices_by_date,
            )
        if index + 1 >= len(sessions):
            continue
        signal_day = by_date.get(session)
        if signal_day is None:
            continue
        entry_date = sessions[index + 1]
        entry_quotes = quotes(entry_date)
        for key, state in states.items():
            _open_state_positions(
                state,
                signal_day,
                entry_quotes,
                session,
                entry_date,
                index + 1,
                state["config"],
                state["control"],
                seed + index + (sum(ord(char) for char in key) % 1000),
            )
    last_session = sessions[-1] if sessions else ""
    output: Dict[str, pd.DataFrame] = {}
    for key, state in states.items():
        config = state["config"]
        for position in state["open_positions"]:
            age = len(sessions) - 1 - int(position["entry_index"])
            status = "PENDING_FUTURE" if age < config.max_hold_sessions else "MISSING_EXIT_QUOTE"
            state["results"].append(
                _position_result(
                    position,
                    status,
                    last_session,
                    exit_reason="future_horizon" if status == "PENDING_FUTURE" else "missing_time_stop_quote",
                    config=config,
                )
            )
        output[key] = pd.DataFrame(state["results"])
    return output


def run_managed_strategy(
    panel: pd.DataFrame,
    sessions: Sequence[str],
    quotes: Callable[[str], pd.DataFrame],
    config: ManagedConfig,
    control: str = "signal",
    seed: int = 20260821,
) -> pd.DataFrame:
    """Walk one fixed strategy across sessions and preserve every entry state."""

    return run_managed_strategies(
        panel,
        sessions,
        quotes,
        {"single": (config, control)},
        seed=seed,
    )["single"]


def _normal_lower(values: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean() - 1.96 * clean.std(ddof=1) / math.sqrt(len(clean)))


def _profit_factor(values: pd.Series) -> Optional[float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    positive = float(clean[clean > 0].sum())
    negative = float(-clean[clean < 0].sum())
    return positive / negative if negative > 0 else (float("inf") if positive > 0 else None)


def _max_drawdown(date_values: pd.Series) -> float:
    curve = date_values.sort_index().cumsum()
    drawdown = curve - curve.cummax()
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _wilson_lower(wins: int, total: int, z: float = 1.96) -> Optional[float]:
    if total <= 0:
        return None
    proportion = wins / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + (z * z / (2.0 * total))) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + (z * z / (4.0 * total * total))
        )
        / denominator
    )
    return float(center - margin)


def managed_calibration_rows(
    trades: pd.DataFrame,
    split_date: str = "2026-04-14",
    holdout_date: str = "2026-06-15",
) -> List[Dict[str, Any]]:
    """Assign date-grouped, prior-only confidence to completed positions."""

    if trades.empty or "status" not in trades.columns:
        return []
    work = trades[trades["status"].eq("SCORED")].copy()
    if work.empty:
        return []
    work["_net_R"] = pd.to_numeric(work.get("net_R"), errors="coerce")
    work = work[work["_net_R"].notna()].copy()
    if work.empty:
        return []
    for column in ("ticker", "option_symbol"):
        if column not in work.columns:
            work[column] = ""
    work["sample"] = np.select(
        [work["signal_date"] < split_date, work["signal_date"] < holdout_date],
        ["TRAIN", "VALIDATION"],
        default="HOLDOUT",
    )
    rows: List[Dict[str, Any]] = []
    for control, control_group in work.groupby("control", sort=True):
        ordered = control_group.sort_values(
            ["signal_date", "ticker", "option_symbol"],
            na_position="last",
        )
        prior_wins = 0
        prior_total = 0
        for signal_date, day_group in ordered.groupby("signal_date", sort=True):
            predicted = (prior_wins + 10.0) / (prior_total + 20.0)
            score_bin = (
                1
                if predicted >= 0.70
                else 2
                if predicted >= 0.60
                else 3
                if predicted >= 0.50
                else 4
                if predicted >= 0.40
                else 5
            )
            for _, row in day_group.iterrows():
                net_r = float(row["_net_R"])
                rows.append(
                    {
                        "control": str(control),
                        "strategy_key": str(row.get("pattern_family") or ""),
                        "sample": str(row.get("sample") or ""),
                        "ticker": row.get("ticker"),
                        "signal_date": signal_date,
                        "entry_date": row.get("entry_date"),
                        "predicted_win_probability": predicted,
                        "score_bin": score_bin,
                        "win": bool(net_r > 0.0),
                        "net_R": net_r,
                        "calibration_method": (
                            "prior_only_date_grouped_beta_mean_win_rate"
                        ),
                    }
                )
            prior_wins += int((day_group["_net_R"] > 0.0).sum())
            prior_total += len(day_group)
    return rows


def _calibration_summary(
    calibration_rows: Sequence[Mapping[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not calibration_rows:
        return {}
    frame = pd.DataFrame(calibration_rows)
    output: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (control, sample), group in frame.groupby(["control", "sample"], sort=True):
        high = group[group["score_bin"].eq(1)].copy()
        net_r = pd.to_numeric(high["net_R"], errors="coerce").dropna()
        wins = int(high["win"].astype(bool).sum())
        count = len(high)
        output[(str(control), str(sample))] = {
            "calibration_sample_count": int(count),
            "calibration_predicted_probability": (
                float(pd.to_numeric(high["predicted_win_probability"], errors="coerce").mean())
                if count
                else None
            ),
            "calibration_win_rate": float(wins / count) if count else None,
            "calibration_win_rate_lower_95": _wilson_lower(wins, count),
            "calibration_average_net_R": float(net_r.mean()) if not net_r.empty else None,
            "calibration_score": _wilson_lower(wins, count),
            "calibration_method": (
                "prior_only_date_grouped_beta_mean_win_rate"
                if count
                else ""
            ),
        }
    return output


def frozen_holdout_calibration(
    trades: pd.DataFrame,
    holdout_start: str,
    *,
    control: str = "signal",
) -> Dict[str, Any]:
    """Evaluate holdout outcomes with confidence frozen before the holdout."""

    if trades.empty or "status" not in trades.columns:
        return {}
    work = trades[
        trades["status"].eq("SCORED")
        & trades.get("control", pd.Series(dtype=str)).astype(str).eq(control)
    ].copy()
    work["signal_date"] = work.get("signal_date", pd.Series(dtype=str)).astype(str)
    work["net_R"] = pd.to_numeric(work.get("net_R"), errors="coerce")
    work = work.dropna(subset=["net_R"])
    prior = work[work["signal_date"].lt(holdout_start)]
    holdout = work[work["signal_date"].ge(holdout_start)]
    prior_wins = int(prior["net_R"].gt(0).sum())
    prior_total = int(len(prior))
    predicted = (prior_wins + 10.0) / (prior_total + 20.0)
    high_confidence = predicted >= 0.70
    evaluated = holdout if high_confidence else holdout.iloc[0:0]
    count = int(len(evaluated))
    wins = int(evaluated["net_R"].gt(0).sum())
    return {
        "calibration_sample_count": count,
        "calibration_predicted_probability": predicted,
        "calibration_win_rate": float(wins / count) if count else None,
        "calibration_win_rate_lower_95": _wilson_lower(wins, count),
        "calibration_average_net_R": (
            float(evaluated["net_R"].mean()) if count else None
        ),
        "calibration_score": _wilson_lower(wins, count),
        "calibration_method": "frozen_pre_holdout_beta_mean_win_rate",
        "calibration_train_rows": prior_total,
        "calibration_train_through": (
            str(prior["signal_date"].max()) if prior_total else ""
        ),
    }


def summarize_managed(
    trades: pd.DataFrame,
    split_date: str = "2026-04-14",
    holdout_date: str = "2026-06-15",
) -> List[Dict[str, Any]]:
    """Produce one readable scorecard per control and chronological sample."""

    if trades.empty:
        return []
    work = trades.copy()
    work["sample"] = np.select(
        [work["signal_date"] < split_date, work["signal_date"] < holdout_date],
        ["TRAIN", "VALIDATION"],
        default="HOLDOUT",
    )
    calibration_summary = _calibration_summary(
        managed_calibration_rows(trades, split_date, holdout_date)
    )
    rows: List[Dict[str, Any]] = []
    for (control, sample), group in work.groupby(["control", "sample"], sort=True):
        scored = group[group["status"].eq("SCORED")].copy()
        pending_future_count = int(group["status"].eq("PENDING_FUTURE").sum())
        missing_exit_count = int(group["status"].eq("MISSING_EXIT_QUOTE").sum())
        eligible_count = max(0, len(group) - pending_future_count)
        net_r = pd.to_numeric(scored["net_R"], errors="coerce").dropna()
        # Daily risk is an equal-risk portfolio of the day's selected names.
        # Summing R by date makes a high-count day look worse (or better) only
        # because it has more positions and inflates drawdown mechanically.
        by_date = scored.groupby("signal_date")["net_R"].mean()
        summary = {
                "control": control,
                "sample": sample,
                "entry_count": len(group),
                "eligible_count": eligible_count,
                "pending_future_count": pending_future_count,
                "missing_exit_count": missing_exit_count,
                "scored_count": len(scored),
                "coverage": len(scored) / eligible_count if eligible_count else None,
                "unique_signal_dates": int(scored["signal_date"].nunique()),
                "average_net_R": float(net_r.mean()) if not net_r.empty else None,
                "median_net_R": float(net_r.median()) if not net_r.empty else None,
                "win_rate": float(scored["win"].mean()) if not scored.empty else None,
                "profit_factor": _profit_factor(net_r),
                "lower_mean_95": _normal_lower(net_r),
                "date_average_net_R": float(by_date.mean()) if not by_date.empty else None,
                "date_lower_mean_95": _normal_lower(by_date),
                "date_max_drawdown": _max_drawdown(by_date),
                "last_signal_date": str(scored["signal_date"].max()) if not scored.empty else "",
                "latest_holdout_average": (
                    float(net_r.mean()) if sample == "HOLDOUT" and not net_r.empty else None
                ),
            }
        summary.update(calibration_summary.get((str(control), str(sample)), {
            "calibration_sample_count": 0,
            "calibration_predicted_probability": None,
            "calibration_win_rate": None,
            "calibration_win_rate_lower_95": None,
            "calibration_average_net_R": None,
            "calibration_score": None,
            "calibration_method": "",
        }))
        rows.append(summary)
    return rows


def managed_validation_rows(
    scorecard: Sequence[Mapping[str, Any]],
    min_scored: int = 20,
    min_dates: int = 20,
    min_coverage: float = 0.70,
    min_profit_factor: float = 1.20,
    min_lower_mean_95: float = 0.0,
    max_drawdown: float = -10.0,
) -> List[Dict[str, Any]]:
    """Apply fixed chronology and confidence gates to managed signal lanes."""

    signal_rows = [row for row in scorecard if str(row.get("control")) == "signal"]
    by_strategy: Dict[str, Dict[str, Mapping[str, Any]]] = {}
    for row in signal_rows:
        by_strategy.setdefault(str(row.get("strategy_key") or row.get("pattern_family") or ""), {})[
            str(row.get("sample") or "")
        ] = row

    def number(row: Mapping[str, Any], key: str) -> Optional[float]:
        value = row.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if math.isfinite(parsed) else None

    output: List[Dict[str, Any]] = []
    required_samples = ("TRAIN", "VALIDATION", "HOLDOUT")
    for strategy_key, samples in sorted(by_strategy.items()):
        blockers: List[str] = []
        sample_pass: Dict[str, bool] = {}
        flattened: Dict[str, Any] = {"strategy_key": strategy_key}
        for sample in required_samples:
            row = samples.get(sample)
            prefix = sample.lower()
            if row is None:
                blockers.append(f"{sample}_SAMPLE_MISSING")
                sample_pass[sample] = False
                continue
            checks = {
                "scored_count": (number(row, "scored_count") or 0.0) >= min_scored,
                "unique_signal_dates": (number(row, "unique_signal_dates") or 0.0) >= min_dates,
                "coverage": (number(row, "coverage") or 0.0) >= min_coverage,
                "average_net_R": (number(row, "average_net_R") or 0.0) > 0.0,
                "profit_factor": (number(row, "profit_factor") or 0.0) >= min_profit_factor,
                "lower_mean_95": (number(row, "lower_mean_95") or 0.0) > min_lower_mean_95,
                "date_lower_mean_95": (number(row, "date_lower_mean_95") or 0.0) > min_lower_mean_95,
                "drawdown": (number(row, "date_max_drawdown") or float("-inf")) >= max_drawdown,
                "calibration_sample_count": (
                    number(row, "calibration_sample_count") or 0.0
                ) >= 10,
                "calibration_score": (
                    number(row, "calibration_score") or 0.0
                ) >= 0.50,
                "calibration_average_net_R": (
                    number(row, "calibration_average_net_R") or 0.0
                ) > 0.0,
            }
            sample_pass[sample] = all(checks.values())
            for check, passed in checks.items():
                flattened[f"{prefix}_{check}"] = passed
                if not passed:
                    blockers.append(f"{sample}_{check.upper()}_GATE")
            for key in (
                "entry_count",
                "scored_count",
                "coverage",
                "unique_signal_dates",
                "average_net_R",
                "profit_factor",
                "lower_mean_95",
                "date_lower_mean_95",
                "date_max_drawdown",
                "last_signal_date",
                "calibration_sample_count",
                "calibration_predicted_probability",
                "calibration_win_rate",
                "calibration_win_rate_lower_95",
                "calibration_average_net_R",
                "calibration_score",
                "calibration_method",
            ):
                flattened[f"{prefix}_{key}"] = row.get(key)
        all_samples_pass = all(sample_pass.get(sample, False) for sample in required_samples)
        flattened.update(
            {
                "chronological_gate": "PASS" if all_samples_pass else "FAIL",
                "status": "QUALIFIED_MANAGED" if all_samples_pass else "RESEARCH_PATTERN",
                "blockers": ";".join(sorted(set(blockers))) or "",
                "approval_status": "QUALIFIED_MANAGED" if all_samples_pass else "NOT_APPROVED",
            }
        )
        output.append(flattened)
    return output


def _managed_number(row: Optional[Mapping[str, Any]], key: str) -> Optional[float]:
    """Read a finite scorecard value without letting NaN pass a gate."""

    if row is None:
        return None
    value = row.get(key)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _managed_sample_row(
    rows: Sequence[Mapping[str, Any]],
    sample: str,
) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row.get("control") or "") == "signal" and str(row.get("sample") or "") == sample:
            return row
    return None


def predeclared_managed_selection_candidates() -> Dict[str, Tuple[ManagedConfig, str]]:
    """Return the contract variants allowed in the model-selection audit.

    The grid is deliberately small and registered in code before the final
    holdout is scored: two DTE bands and three fixed put moneyness choices.
    It is a research family, not a license to search arbitrary expiries,
    strikes, exits, or thresholds after seeing the holdout.
    """

    candidates: Dict[str, Tuple[ManagedConfig, str]] = {}
    for min_dte, max_dte, target_dte, max_hold in (
        (20, 60, 40, 20),
        (60, 110, 80, 40),
    ):
        for moneyness in (0.90, 0.95, 1.00):
            if (min_dte, max_dte, moneyness) == (60, 110, 1.00):
                name = "FLOW_QUANTILE_BULL_CSP_D60_110_H40_T50_IV60"
            else:
                name = (
                    f"FLOW_QUANTILE_BULL_CSP_D{min_dte}_{max_dte}"
                    f"_H{max_hold}_T50_IV60_M{int(moneyness * 100)}"
                )
            candidates[name] = (
                ManagedConfig(
                    name=name,
                    signal_rule="flow_quantile",
                    direction="call",
                    option_type="P",
                    structure="cash_secured_put",
                    max_iv_rank=60.0,
                    moneyness=moneyness,
                    top_quantile=0.90,
                    min_marketcap=2_000_000_000.0,
                    min_avg30_volume=250_000.0,
                    min_sector_names=12,
                    min_dte=min_dte,
                    max_dte=max_dte,
                    target_dte=target_dte,
                    min_open_interest=50.0,
                    min_entry_bid=0.05,
                    max_spread_pct=0.12,
                    profit_target=0.50,
                    stop_loss=None,
                    max_hold_sessions=max_hold,
                    fee_per_side=1.50,
                    one_per_ticker=True,
                ),
                "signal",
            )
    return candidates


def managed_selection_audit(
    trades_by_candidate: Mapping[str, pd.DataFrame],
    selection_end: str,
    holdout_start: str,
    split_date: str = "2026-04-14",
    min_scored: int = 20,
    min_dates: int = 20,
    min_coverage: float = 0.70,
    min_profit_factor: float = 1.20,
    min_lower_mean_95: float = 0.0,
    max_drawdown: float = -0.50,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select a contract family before the final holdout, then test it once.

    ``selection_end`` is the last date visible to the selector.  The selected
    candidate is ranked only by validation date-level lower confidence bound,
    then validation average and profit factor.  Holdout fields are calculated
    afterward for audit output and never influence ``selected_candidate_key``.
    """

    required_samples = ("TRAIN", "VALIDATION")
    final_sample = "HOLDOUT"
    metric_keys = (
        "entry_count",
        "scored_count",
        "coverage",
        "unique_signal_dates",
        "average_net_R",
        "profit_factor",
        "lower_mean_95",
        "date_average_net_R",
        "date_lower_mean_95",
        "date_max_drawdown",
        "last_signal_date",
        "calibration_sample_count",
        "calibration_predicted_probability",
        "calibration_win_rate",
        "calibration_win_rate_lower_95",
        "calibration_average_net_R",
        "calibration_score",
        "calibration_method",
    )

    def gate_row(row: Optional[Mapping[str, Any]]) -> Tuple[bool, List[str]]:
        if row is None:
            return False, ["SAMPLE_MISSING"]
        checks = {
            "scored_count": (_managed_number(row, "scored_count") or 0.0) >= min_scored,
            "unique_signal_dates": (_managed_number(row, "unique_signal_dates") or 0.0) >= min_dates,
            "coverage": (_managed_number(row, "coverage") or 0.0) >= min_coverage,
            "average_net_R": (_managed_number(row, "average_net_R") or 0.0) > 0.0,
            "profit_factor": (_managed_number(row, "profit_factor") or 0.0) >= min_profit_factor,
            "lower_mean_95": (_managed_number(row, "lower_mean_95") or 0.0) > min_lower_mean_95,
            "date_lower_mean_95": (_managed_number(row, "date_lower_mean_95") or 0.0) > min_lower_mean_95,
            "drawdown": (_managed_number(row, "date_max_drawdown") or float("-inf")) >= max_drawdown,
            "calibration_sample_count": (_managed_number(row, "calibration_sample_count") or 0.0) >= 10,
            "calibration_score": (_managed_number(row, "calibration_score") or 0.0) >= 0.50,
            "calibration_average_net_R": (_managed_number(row, "calibration_average_net_R") or 0.0) > 0.0,
        }
        return all(checks.values()), [key.upper() + "_GATE" for key, passed in checks.items() if not passed]

    def control_edge(
        signal_rows: pd.DataFrame,
        random_rows: pd.DataFrame,
        start: str,
        end: str,
        *,
        end_inclusive: bool = True,
    ) -> Dict[str, Any]:
        def daily(frame: pd.DataFrame) -> pd.Series:
            dates = frame.get("signal_date", pd.Series(dtype=str)).astype(str)
            in_window = dates.ge(start)
            in_window &= dates.le(end) if end_inclusive else dates.lt(end)
            scored = frame[
                frame.get("status", pd.Series(dtype=str)).eq("SCORED")
                & in_window
            ].copy()
            scored["net_R"] = pd.to_numeric(scored.get("net_R"), errors="coerce")
            return scored.dropna(subset=["net_R"]).groupby("signal_date")["net_R"].mean()

        paired = pd.concat(
            [daily(signal_rows).rename("signal"), daily(random_rows).rename("random")],
            axis=1,
            join="inner",
        ).dropna()
        edge = paired["signal"] - paired["random"] if not paired.empty else pd.Series(dtype=float)
        return {
            "paired_control_dates": int(len(edge)),
            "paired_control_average_edge_R": float(edge.mean()) if not edge.empty else None,
            "paired_control_lower_95_R": _normal_lower(edge),
            "paired_control_win_rate": float(edge.gt(0).mean()) if not edge.empty else None,
        }

    audit_rows: List[Dict[str, Any]] = []
    signal_candidates = {
        key: trades
        for key, trades in trades_by_candidate.items()
        if not str(key).endswith("_RANDOM")
    }
    for candidate_key, trades in sorted(signal_candidates.items()):
        work = trades.copy()
        random_work = trades_by_candidate.get(f"{candidate_key}_RANDOM", pd.DataFrame()).copy()
        if work.empty or "signal_date" not in work.columns:
            work = pd.DataFrame(columns=trades.columns if isinstance(trades, pd.DataFrame) else [])
        if random_work.empty or "signal_date" not in random_work.columns:
            random_work = pd.DataFrame(columns=work.columns)
        work["signal_date"] = work.get("signal_date", pd.Series(dtype=str)).astype(str)
        random_work["signal_date"] = random_work.get(
            "signal_date", pd.Series(dtype=str)
        ).astype(str)
        selection_trades = work[work["signal_date"] <= selection_end].copy()
        holdout_trades = work[work["signal_date"] >= holdout_start].copy()
        selection_scorecard = summarize_managed(
            selection_trades,
            split_date=split_date,
            holdout_date=holdout_start,
        )
        holdout_scorecard = summarize_managed(
            holdout_trades,
            split_date=split_date,
            holdout_date=holdout_start,
        )
        train_row = _managed_sample_row(selection_scorecard, "TRAIN")
        validation_row = _managed_sample_row(selection_scorecard, "VALIDATION")
        holdout_row = _managed_sample_row(holdout_scorecard, final_sample)
        train_pass, train_blockers = gate_row(train_row)
        validation_pass, validation_blockers = gate_row(validation_row)
        train_control = control_edge(
            work,
            random_work,
            "",
            split_date,
            end_inclusive=False,
        )
        validation_control = control_edge(
            work,
            random_work,
            split_date,
            selection_end,
        )
        holdout_control = control_edge(
            work,
            random_work,
            holdout_start,
            "9999-12-31",
        )
        train_control_pass = (
            train_control["paired_control_dates"] >= min_dates
            and (_managed_number(train_control, "paired_control_average_edge_R") or 0.0) > 0.0
        )
        validation_control_pass = (
            validation_control["paired_control_dates"] >= min_dates
            and (_managed_number(validation_control, "paired_control_average_edge_R") or 0.0) > 0.0
        )
        train_pass = train_pass and train_control_pass
        validation_pass = validation_pass and validation_control_pass
        if not train_control_pass:
            train_blockers.append("MATCHED_RANDOM_CONTROL_GATE")
        if not validation_control_pass:
            validation_blockers.append("MATCHED_RANDOM_CONTROL_GATE")
        selection_pass = train_pass and validation_pass

        calibration = managed_calibration_rows(work, split_date, holdout_start)
        calibration_summary = _calibration_summary(calibration)
        final_calibration = frozen_holdout_calibration(
            work,
            holdout_start,
        )
        final_calibration_row = {
            "calibration_sample_count": final_calibration.get("calibration_sample_count", 0),
            "calibration_predicted_probability": final_calibration.get("calibration_predicted_probability"),
            "calibration_win_rate": final_calibration.get("calibration_win_rate"),
            "calibration_win_rate_lower_95": final_calibration.get("calibration_win_rate_lower_95"),
            "calibration_average_net_R": final_calibration.get("calibration_average_net_R"),
            "calibration_score": final_calibration.get("calibration_score"),
            "calibration_method": final_calibration.get("calibration_method", ""),
        }
        final_for_gate = dict(holdout_row or {})
        final_for_gate.update(final_calibration_row)
        final_pass, final_blockers = gate_row(final_for_gate)
        final_control_pass = (
            holdout_control["paired_control_dates"] >= min_dates
            and (_managed_number(holdout_control, "paired_control_average_edge_R") or 0.0) > 0.0
        )
        final_pass = final_pass and final_control_pass
        if not final_control_pass:
            final_blockers.append("MATCHED_RANDOM_CONTROL_GATE")
        config = None
        if "pattern_family" in work.columns and not work.empty:
            # The config is recorded in the result's derived metadata below;
            # candidate dimensions are injected by the caller when available.
            config_name = str(work["pattern_family"].dropna().iloc[0]) if not work["pattern_family"].dropna().empty else candidate_key
        else:
            config_name = candidate_key
        row: Dict[str, Any] = {
            "candidate_key": candidate_key,
            "candidate_name": config_name,
            "selection_end": selection_end,
            "final_holdout_start": holdout_start,
            "selection_status": "ELIGIBLE" if selection_pass else "REJECTED_PRE_HOLDOUT",
            "selection_train_pass": train_pass,
            "selection_validation_pass": validation_pass,
            "selection_eligible": selection_pass,
            "selection_blockers": ";".join(sorted(set([f"TRAIN_{x}" for x in train_blockers] + [f"VALIDATION_{x}" for x in validation_blockers]))) or "",
            "final_holdout_status": "PASS" if final_pass else "FAIL",
            "final_holdout_pass": final_pass,
            "final_holdout_blockers": ";".join(sorted(set(final_blockers))) or "",
            "final_calibration_sample_count": final_calibration_row["calibration_sample_count"],
            "final_calibration_score": final_calibration_row["calibration_score"],
            "final_calibration_average_net_R": final_calibration_row["calibration_average_net_R"],
            "selection_train_control_pass": train_control_pass,
            "selection_validation_control_pass": validation_control_pass,
            "final_holdout_control_pass": final_control_pass,
        }
        for prefix, values in (
            ("selection_train", train_control),
            ("selection_validation", validation_control),
            ("final_holdout", holdout_control),
        ):
            for key, value in values.items():
                row[f"{prefix}_{key}"] = value
        for prefix, sample_row in (
            ("selection_train", train_row),
            ("selection_validation", validation_row),
            ("final_holdout", holdout_row),
        ):
            for key in metric_keys:
                if prefix == "final_holdout" and key.startswith("calibration_"):
                    # The holdout confidence must remain the prior-only
                    # confidence learned from all earlier signal dates.  A
                    # holdout-only recalibration is descriptive, not valid
                    # evidence for the final audit gate.
                    continue
                row[f"{prefix}_{key}"] = sample_row.get(key) if sample_row is not None else None
        for key, value in final_calibration_row.items():
            row[f"final_holdout_{key}"] = value
        audit_rows.append(row)

    eligible = [row for row in audit_rows if row.get("selection_eligible")]
    selected: Optional[Dict[str, Any]] = None
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                _managed_number(row, "selection_validation_date_lower_mean_95") or float("-inf"),
                _managed_number(row, "selection_validation_average_net_R") or float("-inf"),
                _managed_number(row, "selection_validation_profit_factor") or float("-inf"),
                str(row.get("candidate_key") or ""),
            ),
        )
        for row in audit_rows:
            row["selected_candidate"] = row.get("candidate_key") == selected.get("candidate_key")
    else:
        for row in audit_rows:
            row["selected_candidate"] = False

    selected_final_pass = bool(selected and selected.get("final_holdout_pass"))
    blockers: List[str] = []
    if selected is None:
        blockers.append("NO_CANDIDATE_CLEARS_PREDECLARED_SELECTION_GATES")
    elif not selected_final_pass:
        blockers.append("SELECTED_CANDIDATE_FAILS_FINAL_HOLDOUT_GATES")
    metadata = {
        "status": "PASS" if selected_final_pass else "FAIL_REQUIREMENTS_REMAIN",
        "blockers": blockers,
        "candidate_count": len(audit_rows),
        "eligible_candidate_count": len(eligible),
        "selected_candidate_key": selected.get("candidate_key") if selected else "",
        "selected_final_holdout_pass": selected_final_pass,
        "selection_end": selection_end,
        "final_holdout_start": holdout_start,
        "selection_rule": (
            "train_and_validation_fixed_gates_then_max_validation_date_lower_mean_95;"
            "tie_break_validation_average_net_R_then_profit_factor"
        ),
        "holdout_used_for_selection": False,
    }
    return audit_rows, metadata


def managed_regime_rows(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    holdout_start: str,
    min_dates: int = 20,
) -> List[Dict[str, Any]]:
    """Report selected-strategy results by a point-in-time SPY regime."""

    if trades.empty or panel.empty:
        return []
    scored = trades[
        trades.get("status", pd.Series(dtype=str)).eq("SCORED")
        & trades.get("signal_date", pd.Series(dtype=str)).astype(str).ge(holdout_start)
    ].copy()
    if scored.empty or "net_R" not in scored.columns:
        return []
    benchmark = panel[panel["ticker"].eq("SPY")][["date", "return_20d"]].copy()
    benchmark["date"] = benchmark["date"].astype(str)
    benchmark["return_20d"] = pd.to_numeric(benchmark["return_20d"], errors="coerce")
    scored["signal_date"] = scored["signal_date"].astype(str)
    scored = scored.merge(benchmark, left_on="signal_date", right_on="date", how="left")
    scored["regime"] = np.select(
        [scored["return_20d"].le(-0.02), scored["return_20d"].ge(0.02)],
        ["BEAR", "BULL"],
        default="SIDEWAYS",
    )
    rows: List[Dict[str, Any]] = []
    for regime, group in scored.groupby("regime", sort=True):
        values = pd.to_numeric(group["net_R"], errors="coerce").dropna()
        by_date = group.groupby("signal_date")["net_R"].mean()
        rows.append(
            {
                "regime": str(regime),
                "holdout_start": holdout_start,
                "scored_count": int(len(values)),
                "unique_signal_dates": int(group["signal_date"].nunique()),
                "average_net_R": float(values.mean()) if not values.empty else None,
                "profit_factor": _profit_factor(values),
                "date_average_net_R": float(by_date.mean()) if not by_date.empty else None,
                "date_lower_mean_95": _normal_lower(by_date),
                "status": (
                    "PASS"
                    if len(by_date) >= min_dates and not values.empty and float(values.mean()) > 0
                    else "POSITIVE_SMALL_SAMPLE"
                    if not values.empty and float(values.mean()) > 0
                    else "WEAK_SAMPLE"
                ),
            }
        )
    return rows


def default_managed_strategies() -> Dict[str, Tuple[ManagedConfig, str]]:
    """Return fixed, auditable research hypotheses for the managed lane."""

    common = dict(
        top_quantile=0.90,
        min_marketcap=2_000_000_000.0,
        min_avg30_volume=250_000.0,
        min_sector_names=12,
        min_dte=20,
        max_dte=60,
        target_dte=40,
        min_open_interest=50.0,
        min_entry_bid=0.05,
        max_spread_pct=0.12,
        profit_target=0.50,
        stop_loss=None,
        max_hold_sessions=20,
        fee_per_side=1.50,
        one_per_ticker=True,
    )
    long_term_common = dict(
        top_quantile=0.90,
        min_marketcap=2_000_000_000.0,
        min_avg30_volume=250_000.0,
        min_sector_names=12,
        min_dte=60,
        max_dte=110,
        target_dte=80,
        min_open_interest=50.0,
        min_entry_bid=0.05,
        max_spread_pct=0.12,
        moneyness=1.05,
        profit_target=0.50,
        stop_loss=None,
        max_hold_sessions=40,
        fee_per_side=1.50,
        one_per_ticker=True,
    )
    hypotheses = [
        ManagedConfig(
            name="FLOW_QUANTILE_CALL_D20_60_H20_T50",
            signal_rule="flow_quantile",
            direction="call",
            structure="long_option",
            moneyness=1.02,
            **common,
        ),
        ManagedConfig(
            name="FLOW_QUANTILE_CALL_D20_60_H20_T50_IV60",
            signal_rule="flow_quantile",
            direction="call",
            structure="long_option",
            max_iv_rank=60.0,
            moneyness=1.02,
            **common,
        ),
        ManagedConfig(
            name="MOMENTUM_FLOW_CALL_D20_60_H20_T50",
            signal_rule="momentum_flow",
            direction="call",
            structure="long_option",
            moneyness=1.02,
            **common,
        ),
        ManagedConfig(
            name="MOMENTUM_5_CALL_D20_60_H20_T50",
            signal_rule="momentum_5",
            direction="call",
            signal_direction="call",
            structure="long_option",
            moneyness=0.95,
            **common,
        ),
        ManagedConfig(
            name="MOMENTUM_5_PUT_D20_60_H20_T50",
            signal_rule="momentum_5",
            direction="put",
            signal_direction="put",
            structure="long_option",
            moneyness=1.05,
            **common,
        ),
        ManagedConfig(
            name="FLOW_QUANTILE_BULL_PUT_CREDIT_D20_60_H20_T50",
            signal_rule="flow_quantile",
            direction="call",
            option_type="P",
            structure="credit_vertical",
            moneyness=0.98,
            short_moneyness=0.90,
            **common,
        ),
        ManagedConfig(
            name="FLOW_QUANTILE_BULL_CSP_D60_110_H40_T50_IV60",
            signal_rule="flow_quantile",
            direction="call",
            option_type="P",
            structure="cash_secured_put",
            max_iv_rank=60.0,
            moneyness=1.00,
            **{
                key: value
                for key, value in long_term_common.items()
                if key != "moneyness"
            },
        ),
        ManagedConfig(
            name="POSITION_52W_CALL_D60_110_H40_T50",
            signal_rule="trend_quantile",
            direction="call",
            structure="long_option",
            **long_term_common,
        ),
        ManagedConfig(
            name="POSITION_52W_CALL_D60_110_H40_T100",
            signal_rule="trend_quantile",
            direction="call",
            structure="long_option",
            profit_target=1.00,
            **{
                key: value
                for key, value in long_term_common.items()
                if key != "profit_target"
            },
        ),
        ManagedConfig(
            name="TRENDFLOW_CALL_D60_110_H40_T50",
            signal_rule="trend_flow",
            direction="call",
            structure="long_option",
            **long_term_common,
        ),
        ManagedConfig(
            name="POST_EVENT_DOWN_CALL_D20_60_H20_T50",
            signal_rule="post_event_mean_reversion",
            direction="call",
            signal_direction="call",
            structure="long_option",
            moneyness=1.02,
            **common,
        ),
        ManagedConfig(
            name="POST_EVENT_UP_PUT_D20_60_H20_T50",
            signal_rule="post_event_mean_reversion",
            direction="put",
            signal_direction="put",
            structure="long_option",
            moneyness=0.98,
            **common,
        ),
        ManagedConfig(
            name="EARNINGS_FLOW_CALL_D20_60_H10_T50",
            signal_rule="earnings_flow",
            direction="call",
            signal_direction="call",
            structure="long_option",
            moneyness=1.00,
            earnings_min_days=0,
            earnings_max_days=10,
            min_implied_move_perc=0.0,
            max_hold_sessions=10,
            **{
                key: value
                for key, value in common.items()
                if key != "max_hold_sessions"
            },
        ),
        ManagedConfig(
            name="EARNINGS_FLOW_PUT_D20_60_H10_T50",
            signal_rule="earnings_flow",
            direction="put",
            signal_direction="put",
            structure="long_option",
            moneyness=1.00,
            earnings_min_days=0,
            earnings_max_days=10,
            min_implied_move_perc=0.0,
            max_hold_sessions=10,
            **{
                key: value
                for key, value in common.items()
                if key != "max_hold_sessions"
            },
        ),
        ManagedConfig(
            name="EARNINGS_STRADDLE_D20_60_H10_T50",
            signal_rule="earnings_event",
            direction="call",
            signal_direction="call",
            structure="long_straddle",
            moneyness=1.00,
            earnings_min_days=1,
            earnings_max_days=10,
            min_implied_move_perc=0.0,
            max_hold_sessions=10,
            **{
                key: value
                for key, value in common.items()
                if key != "max_hold_sessions"
            },
        ),
    ]
    strategies: Dict[str, Tuple[ManagedConfig, str]] = {}
    for config in hypotheses:
        strategies[config.name] = (config, "signal")
    for name in (
        "FLOW_QUANTILE_CALL_D20_60_H20_T50",
        "FLOW_QUANTILE_BULL_PUT_CREDIT_D20_60_H20_T50",
        "FLOW_QUANTILE_BULL_CSP_D60_110_H40_T50_IV60",
    ):
        config = next(config for config in hypotheses if config.name == name)
        strategies[f"{name}_RANDOM"] = (config, "random")
    bot_hypotheses = [
        ManagedConfig(
            name="BOT_FLOW_QUANTILE_CALL_D20_60_H20_T50",
            signal_rule="bot_flow_quantile",
            direction="call",
            signal_direction="call",
            structure="long_option",
            moneyness=0.95,
            **common,
        ),
        ManagedConfig(
            name="BOT_FLOW_QUANTILE_PUT_D20_60_H20_T50",
            signal_rule="bot_flow_quantile",
            direction="put",
            signal_direction="put",
            structure="long_option",
            moneyness=1.05,
            **common,
        ),
    ]
    for config in bot_hypotheses:
        strategies[config.name] = (config, "bot_flow")
    return strategies


def run_managed_research(
    base_dir: Path,
    start_date: str,
    end_date: str,
    out_dir: Path,
    cache_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the fixed managed research lane and write its complete evidence set."""

    panel, panel_metadata = load_stock_panel(base_dir, start_date, end_date)
    bot_flow, bot_flow_metadata = load_cached_bot_flow(base_dir, start_date, end_date)
    if not bot_flow.empty:
        panel = panel.merge(bot_flow, on=["date", "ticker"], how="left", sort=False)
    panel_metadata = dict(panel_metadata)
    panel_metadata["bot_flow_metadata"] = bot_flow_metadata
    source_index, quote_metadata = build_quote_source_index(base_dir, start_date, end_date)
    sessions = eligible_sessions(panel, source_index)
    underlying_prices = {
        str(day): frame.set_index("ticker")["close"].astype(float)
        for day, frame in panel.groupby("date", sort=False)
    }
    cache = QuoteCache(
        source_index,
        underlying_prices_by_date=underlying_prices,
        max_entries=8,
        materialized_dir=cache_root,
        cache_key=f"{start_date}_{end_date}_hot_oi",
    )
    strategies = default_managed_strategies()
    mover_rows = named_mover_audit(panel, sessions, strategies)
    price_outcomes, price_validation = managed_price_research(panel, sessions, strategies)
    price_pattern_validation = managed_price_validation_rows(price_validation)
    results = run_managed_strategies(panel, sessions, lambda day: cache.get(day), strategies)
    selection_candidates = predeclared_managed_selection_candidates()
    selection_strategies = dict(selection_candidates)
    for candidate_key, (config, _control) in selection_candidates.items():
        selection_strategies[f"{candidate_key}_RANDOM"] = (config, "random")
    selection_results = run_managed_strategies(
        panel,
        sessions,
        lambda day: cache.get(day),
        selection_strategies,
    )
    # The final holdout boundary is fixed before any candidate score is read.
    # For shorter as-of runs, keep the audit explicit rather than silently
    # relabeling an incomplete window as a validated selection.
    holdout_start = next((day for day in sessions if day >= "2026-06-15"), "")
    selection_end = ""
    if holdout_start:
        prior_sessions = [day for day in sessions if day < holdout_start]
        selection_end = prior_sessions[-1] if prior_sessions else ""
    if selection_end and holdout_start:
        selection_audit, selection_metadata = managed_selection_audit(
            selection_results,
            selection_end,
            holdout_start,
        )
        selected_candidate = selection_metadata.get("selected_candidate_key") or ""
        selected_trades = selection_results.get(selected_candidate, pd.DataFrame())
        selection_regime = managed_regime_rows(selected_trades, panel, holdout_start)
    else:
        selection_audit = []
        selection_regime = []
        selection_metadata = {
            "status": "FAIL_REQUIREMENTS_REMAIN",
            "blockers": ["SELECTION_HOLDOUT_WINDOW_NOT_AVAILABLE"],
            "candidate_count": len(selection_candidates),
            "eligible_candidate_count": 0,
            "selected_candidate_key": "",
            "selected_final_holdout_pass": False,
            "selection_end": selection_end,
            "final_holdout_start": holdout_start,
            "selection_rule": "predeclared_candidate_selection_unavailable",
            "holdout_used_for_selection": False,
        }
    selection_passed = selection_metadata.get("status") == "PASS"
    selected_candidate = str(selection_metadata.get("selected_candidate_key") or "")
    current_setups: List[Dict[str, Any]] = []
    if selected_candidate and sessions:
        selected_config = selection_candidates[selected_candidate][0]
        current_setups = build_current_option_setups(
            panel,
            sessions[-1],
            cache.get(sessions[-1]),
            selected_config,
        )
        production_status = (
            "PRODUCTION_QUALIFIED" if selection_passed else "RESEARCH_ONLY"
        )
        for row in current_setups:
            row["production_status"] = production_status
    out_dir.mkdir(parents=True, exist_ok=True)
    all_trades: List[pd.DataFrame] = []
    scorecard: List[Dict[str, Any]] = []
    calibration_rows: List[Dict[str, Any]] = []
    for key, trades in results.items():
        config, control = strategies[key]
        rows = summarize_managed(trades)
        for row in rows:
            row["strategy_key"] = key
        for row in managed_calibration_rows(trades):
            row["strategy_key"] = key
            calibration_rows.append(row)
        write_managed_artifacts(
            out_dir / key,
            config,
            trades,
            rows,
            {
                "pipeline": "pattern_analysis_v2_managed",
                "strategy_key": key,
                "control": control,
                "panel_metadata": panel_metadata,
                "quote_metadata": quote_metadata,
                "session_count": len(sessions),
                "first_session": sessions[0] if sessions else "",
                "last_session": sessions[-1] if sessions else "",
                "cache_hits": cache.cache_hits,
                "cache_misses": cache.cache_misses,
                "cache_dir": str(cache.materialized_dir) if cache.materialized_dir else "",
            },
        )
        if not trades.empty:
            all_trades.append(trades.assign(strategy_key=key))
        scorecard.extend(rows)
    validation = managed_validation_rows(scorecard)
    for row in validation:
        raw_qualified = row.get("status") == "QUALIFIED_MANAGED"
        production_qualified = (
            selection_passed
            and raw_qualified
            and str(row.get("strategy_key") or "") == selected_candidate
        )
        row["production_status"] = (
            "PRODUCTION_QUALIFIED" if production_qualified else "RESEARCH_ONLY"
        )
        row["production_blockers"] = (
            ""
            if production_qualified
            else "PREDECLARED_SELECTION_AND_MATCHED_CONTROL_GATE_NOT_PASSED"
        )
    if all_trades:
        combined_columns = sorted({column for frame in all_trades for column in frame.columns})
        combined = pd.DataFrame(
            [
                record
                for frame in all_trades
                for record in frame.to_dict(orient="records")
            ],
            columns=combined_columns,
        )
    else:
        combined = pd.DataFrame()
    combined.to_csv(out_dir / "managed_research_trades.csv", index=False)
    pd.DataFrame(scorecard).to_csv(out_dir / "managed_research_scorecard.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        out_dir / "managed_research_calibration.csv", index=False
    )
    pd.DataFrame(validation).to_csv(out_dir / "managed_research_validation.csv", index=False)
    pd.DataFrame(mover_rows).to_csv(out_dir / "managed_named_mover_audit.csv", index=False)
    pd.DataFrame(price_outcomes).to_csv(out_dir / "managed_price_outcomes.csv", index=False)
    pd.DataFrame(price_validation).to_csv(out_dir / "managed_price_validation.csv", index=False)
    pd.DataFrame(price_pattern_validation).to_csv(
        out_dir / "managed_price_pattern_validation.csv", index=False
    )
    pd.DataFrame(selection_audit).to_csv(
        out_dir / "managed_selection_audit.csv", index=False
    )
    pd.DataFrame(selection_regime).to_csv(
        out_dir / "managed_selection_regime.csv", index=False
    )
    pd.DataFrame(current_setups).to_csv(
        out_dir / "current_option_setups.csv", index=False
    )
    selection_candidates_dir = out_dir / "managed_selection_candidates"
    for candidate_key, candidate_trades in selection_results.items():
        candidate_dir = selection_candidates_dir / candidate_key
        candidate_dir.mkdir(parents=True, exist_ok=True)
        candidate_trades.to_csv(candidate_dir / "managed_exit_trades.csv", index=False)
        pd.DataFrame(summarize_managed(candidate_trades)).to_csv(
            candidate_dir / "managed_exit_scorecard.csv", index=False
        )
        (candidate_dir / "managed_exit_metadata.json").write_text(
            json.dumps(
                {
                    "pipeline": "pattern_analysis_v2_managed_selection",
                    "candidate_key": candidate_key,
                    "selection_end": selection_end,
                    "final_holdout_start": holdout_start,
                    "holdout_used_for_selection": False,
                    "no_order_placement": True,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    exit_reason_counts = (
        combined["exit_reason"].fillna("").astype(str).value_counts().to_dict()
        if "exit_reason" in combined.columns
        else {}
    )
    structure_counts = (
        combined["structure"].fillna("").astype(str).value_counts().to_dict()
        if "structure" in combined.columns
        else {}
    )
    metadata = {
        "pipeline": "pattern_analysis_v2_managed",
        "run_id": f"managed_{start_date}_{end_date}",
        "start_date": start_date,
        "end_date": end_date,
        "panel_metadata": panel_metadata,
        "stock_screener_rows": panel_metadata.get("stock_screener_rows"),
        "stock_screener_source_dates": panel_metadata.get("stock_screener_source_dates"),
        "bot_flow_metadata": bot_flow_metadata,
        "quote_metadata": quote_metadata,
        "quote_source_file_count": quote_metadata.get("quote_source_file_count"),
        "session_count": len(sessions),
        "first_session": sessions[0] if sessions else "",
        "last_session": sessions[-1] if sessions else "",
        "strategy_count": len(strategies),
        "bot_flow_strategy_count": sum(control == "bot_flow" for _, control in strategies.values()),
        "cache_hits": cache.cache_hits,
        "cache_misses": cache.cache_misses,
        "cache_dir": str(cache.materialized_dir) if cache.materialized_dir else "",
        "qualified_managed_count": sum(row.get("status") == "QUALIFIED_MANAGED" for row in validation),
        "production_qualified_count": sum(
            row.get("production_status") == "PRODUCTION_QUALIFIED"
            for row in validation
        ),
        "research_pattern_count": sum(row.get("status") == "RESEARCH_PATTERN" for row in validation),
        "named_mover_audit_rows": len(mover_rows),
        "known_mover_audit_rows": len(mover_rows),
        "named_mover_significant_count": sum(bool(row.get("significant_move_5pct")) for row in mover_rows),
        "price_outcome_rows": len(price_outcomes),
        "price_validation_rows": len(price_validation),
        "price_pattern_validation_rows": len(price_pattern_validation),
        "calibration_rows": len(calibration_rows),
        "selection_audit_status": selection_metadata.get("status"),
        "selection_audit_blockers": selection_metadata.get("blockers") or [],
        "selection_candidate_count": selection_metadata.get("candidate_count", 0),
        "selection_eligible_candidate_count": selection_metadata.get("eligible_candidate_count", 0),
        "selected_candidate_key": selection_metadata.get("selected_candidate_key", ""),
        "selected_final_holdout_pass": selection_metadata.get("selected_final_holdout_pass", False),
        "current_option_setup_count": len(current_setups),
        "selection_end": selection_metadata.get("selection_end", ""),
        "selection_final_holdout_start": selection_metadata.get("final_holdout_start", ""),
        "selection_rule": selection_metadata.get("selection_rule", ""),
        "selection_holdout_used_for_selection": selection_metadata.get("holdout_used_for_selection", True),
        "selection_regime_rows": len(selection_regime),
        "selection_audit_rows": selection_audit,
        "selection_regime_data": selection_regime,
        "qualified_price_pattern_count": sum(
            row.get("status") == "QUALIFIED_DIRECTIONAL"
            for row in price_pattern_validation
        ),
        "managed_exit_reason_counts": exit_reason_counts,
        "managed_structure_counts": structure_counts,
        "managed_stale_exit_count": int(exit_reason_counts.get("time_stop_last_observed_quote", 0)),
        "no_order_placement": True,
    }
    (out_dir / "managed_research_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "metadata": metadata,
        "scorecard": scorecard,
        "validation": validation,
        "named_mover_audit": mover_rows,
        "price_outcomes": price_outcomes,
        "price_validation": price_validation,
        "price_pattern_validation": price_pattern_validation,
        "calibration_rows": calibration_rows,
        "selection_audit": selection_audit,
        "selection_regime": selection_regime,
        "current_setups": current_setups,
        "selection_metadata": selection_metadata,
    }


def write_managed_artifacts(
    out_dir: Path,
    config: ManagedConfig,
    trades: pd.DataFrame,
    scorecard: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "managed_exit_trades.csv", index=False)
    pd.DataFrame(scorecard).to_csv(out_dir / "managed_exit_scorecard.csv", index=False)
    payload = {"config": asdict(config), **dict(metadata)}
    (out_dir / "managed_exit_metadata.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "ChainSource",
    "ManagedConfig",
    "QuoteCache",
    "build_chain_source_index",
    "build_current_option_setups",
    "build_quote_source_index",
    "eligible_sessions",
    "frozen_holdout_calibration",
    "load_quotes_for_session",
    "load_cached_bot_flow",
    "load_stock_panel",
    "managed_calibration_rows",
    "managed_regime_rows",
    "managed_price_research",
    "managed_price_validation_rows",
    "managed_selection_audit",
    "named_mover_audit",
    "managed_validation_rows",
    "default_managed_strategies",
    "predeclared_managed_selection_candidates",
    "run_managed_research",
    "run_managed_strategy",
    "run_managed_strategies",
    "summarize_managed",
    "write_managed_artifacts",
]
