"""Broad, provider-traceable equity universe and local Schwab screen.

The local screen is deliberately separate from option selection.  It preserves
every source constituent, records every missing quote, and marks securities
not admitted to the bounded ORATS funnel as budget-unresolved rather than
silently dropping them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from xml.etree import ElementTree
from zipfile import ZipFile

from .schwab import Quote, SchwabHTTPProvider, SchwabMarketDataBoundary
from .cohorts import eligible_members, load_point_in_time_universe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDINGS = PROJECT_ROOT / "var" / "universe" / "spy_holdings_2026-08-27.xlsx"
_CELL_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


class UniverseError(RuntimeError):
    """The broad universe or quote snapshot is incomplete or malformed."""


def _private_json(path: Path, value: Any) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise UniverseError("universe artifacts must remain inside Cultra") from exc
    resolved.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(resolved.parent, 0o700)
    temporary = resolved.with_name(".%s.tmp-%d" % (resolved.name, os.getpid()))
    data = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    try:
        with open(temporary, "xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, resolved)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return resolved


def _xlsx_rows(path: Path) -> Tuple[Tuple[str, ...], ...]:
    try:
        with ZipFile(path) as archive:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in root.iter("{%s}si" % _CELL_NS):
                    shared.append(
                        "".join(node.text or "" for node in item.iter("{%s}t" % _CELL_NS))
                    )
            sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    except (OSError, KeyError, ValueError) as exc:
        raise UniverseError("holdings workbook is unreadable") from exc
    result = []
    for row in sheet.iter("{%s}row" % _CELL_NS):
        values: Dict[int, str] = {}
        for cell in row.findall("{%s}c" % _CELL_NS):
            reference = str(cell.get("r", ""))
            letters = "".join(char for char in reference if char.isalpha())
            column = 0
            for char in letters:
                column = column * 26 + ord(char.upper()) - 64
            value_node = cell.find("{%s}v" % _CELL_NS)
            value = "" if value_node is None else str(value_node.text or "")
            if cell.get("t") == "s" and value:
                value = shared[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(
                    node.text or "" for node in cell.iter("{%s}t" % _CELL_NS)
                )
            if column:
                values[column] = value.strip()
        if values:
            result.append(tuple(values.get(index, "") for index in range(1, max(values) + 1)))
    return tuple(result)


def load_spy_holdings(path: Path = DEFAULT_HOLDINGS) -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve()
    rows = _xlsx_rows(resolved)
    if len(rows) < 500 or len(rows[0]) < 2:
        raise UniverseError("holdings workbook does not contain a broad universe")
    as_of = ""
    header_index: Optional[int] = None
    for index, row in enumerate(rows):
        if row and row[0] == "Holdings:" and len(row) > 1:
            as_of = row[1].replace("As of ", "").strip()
        if len(row) > 1 and row[0] == "Name" and row[1] == "Ticker":
            header_index = index
            break
    if header_index is None or not as_of:
        raise UniverseError("holdings workbook metadata is incomplete")
    holdings = []
    seen = set()
    for row in rows[header_index + 1 :]:
        if len(row) < 2:
            continue
        ticker = row[1].upper().strip()
        if not _TICKER_RE.fullmatch(ticker) or ticker in seen:
            continue
        seen.add(ticker)
        try:
            weight = float(row[4]) if len(row) > 4 and row[4] else None
        except ValueError:
            weight = None
        holdings.append(
            {
                "ticker": ticker,
                "name": row[0].strip(),
                "index_weight_percent": weight,
                "schwab_symbol": ticker.replace(".", "/"),
            }
        )
    if len(holdings) < 500:
        raise UniverseError("fewer than 500 unique equity holdings were parsed")
    return {
        "schema": "cultra.broad-universe.v1",
        "provider": "State Street Investment Management",
        "provider_url": "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-spy.xlsx",
        "provider_as_of": as_of,
        "source_sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
        "source_file": str(resolved),
        "constituent_count": len(holdings),
        "holdings": holdings,
    }


def _quote_row(holding: Mapping[str, Any], quote: Quote) -> Mapping[str, Any]:
    midpoint = (quote.bid + quote.ask) / 2.0
    price = quote.last if quote.last is not None and quote.last > 0 else midpoint
    spread_fraction = None if midpoint <= 0 else (quote.ask - quote.bid) / midpoint
    dollar_volume = (
        None
        if quote.total_volume is None or price is None
        else float(price) * int(quote.total_volume)
    )
    range_position = None
    if (
        quote.week52_low is not None
        and quote.week52_high is not None
        and quote.week52_high > quote.week52_low
        and price is not None
    ):
        range_position = (price - quote.week52_low) / (quote.week52_high - quote.week52_low)
    return {
        "ticker": holding["ticker"],
        "name": holding["name"],
        "schwab_symbol": holding["schwab_symbol"],
        "index_weight_percent": holding["index_weight_percent"],
        "asset_type": holding.get("asset_type"),
        "sampling_stratum": holding.get("sampling_stratum"),
        "bid": quote.bid,
        "ask": quote.ask,
        "last": quote.last,
        "quote_timestamp": quote.timestamp.isoformat(),
        "total_volume": quote.total_volume,
        "close": quote.close,
        "net_percent_change": quote.net_percent_change,
        "week52_high": quote.week52_high,
        "week52_low": quote.week52_low,
        "spread_fraction": spread_fraction,
        "dollar_volume": dollar_volume,
        "week52_position": range_position,
    }


def local_screen(
    rows: Sequence[Mapping[str, Any]], *, orats_capacity: Optional[int] = None
) -> Mapping[str, Any]:
    """Apply only deterministic liquidity rules unless a caller names a cap.

    A capacity is a diagnostic/request-budget constraint, never the definition
    of Cultra's eligible universe.  The production default therefore preserves
    every locally eligible symbol for downstream reconciliation.
    """

    if orats_capacity is not None and (
        isinstance(orats_capacity, bool) or int(orats_capacity) <= 0
    ):
        raise UniverseError("explicit ORATS Core capacity must be positive")
    eligible = []
    locally_rejected = []
    for row in rows:
        price = float(row["last"] or 0.0)
        dollar_volume = row.get("dollar_volume")
        spread = row.get("spread_fraction")
        reasons = []
        if price < 10.0:
            reasons.append("underlying below $10")
        if dollar_volume is None or float(dollar_volume) < 20_000_000.0:
            reasons.append("underlying dollar volume below $20M")
        if spread is None or float(spread) > 0.01:
            reasons.append("underlying quote spread above 1%")
        if reasons:
            locally_rejected.append(dict(row, disposition="LOCAL_SCREEN_REJECT", reasons=reasons))
            continue
        day_move = abs(float(row.get("net_percent_change") or 0.0))
        position = row.get("week52_position")
        range_extreme = 0.0 if position is None else abs(float(position) - 0.5) * 2.0
        liquidity = math.log10(max(float(dollar_volume), 1.0))
        score = liquidity + 0.18 * day_move + 0.60 * range_extreme
        eligible.append(dict(row, local_screen_score=score))
    ranked = sorted(
        eligible,
        key=lambda item: (-float(item["local_screen_score"]), str(item["ticker"])),
    )
    admitted = tuple(
        ranked if orats_capacity is None else ranked[: int(orats_capacity)]
    )
    budget_unresolved = tuple(
        dict(
            item,
            disposition="NOT_FULLY_EVALUATED_BUDGET",
            reason=(
                "passed local liquidity screen but fell outside the caller's "
                "explicit ORATS Core diagnostic capacity"
            ),
        )
        for item in (
            () if orats_capacity is None else ranked[int(orats_capacity) :]
        )
    )
    return {
        "admitted": admitted,
        "budget_unresolved": budget_unresolved,
        "locally_rejected": tuple(locally_rejected),
    }


def fetch_broad_quote_snapshot(
    *,
    universe_path: Path,
    as_of: date,
    output_path: Path,
    provider: Optional[SchwabMarketDataBoundary] = None,
    orats_capacity: Optional[int] = None,
) -> Mapping[str, Any]:
    point_in_time = load_point_in_time_universe(universe_path)
    current_members = eligible_members(
        point_in_time, selection_date=as_of, required_through=as_of
    )
    if len(current_members) < 100:
        raise UniverseError(
            "point-in-time broad universe has fewer than 100 optionable stocks and ETFs"
        )
    if len({item.ticker for item in current_members}) != len(current_members):
        raise UniverseError("point-in-time universe has overlapping memberships")
    holdings = [
        {
            "ticker": item.ticker,
            "name": item.ticker,
            "index_weight_percent": None,
            "schwab_symbol": item.ticker.replace(".", "/"),
            "asset_type": item.asset_type,
            "sampling_stratum": item.sampling_stratum,
        }
        for item in current_members
    ]
    universe = {
        "schema": "cultra.broad-universe.v2",
        "universe_id": point_in_time.universe_id,
        "provider": point_in_time.provider,
        "source_uri": point_in_time.source_uri,
        "source_sha256": point_in_time.source_sha256,
        "provider_as_of": as_of.isoformat(),
        "coverage": point_in_time.coverage,
        "universe_fingerprint": point_in_time.fingerprint,
        "constituent_count": len(holdings),
        "holdings": holdings,
    }
    boundary = provider or SchwabMarketDataBoundary(SchwabHTTPProvider.production())
    by_schwab = {item["schwab_symbol"]: item for item in universe["holdings"]}
    symbols = tuple(sorted(by_schwab))
    quotes: Dict[str, Quote] = {}
    for offset in range(0, len(symbols), 100):
        quotes.update(boundary.quotes(symbols[offset : offset + 100]))
    rows = []
    missing = []
    for symbol in symbols:
        quote = quotes.get(symbol)
        if quote is None:
            missing.append(
                {
                    "ticker": by_schwab[symbol]["ticker"],
                    "schwab_symbol": symbol,
                    "disposition": "DATA_UNAVAILABLE",
                    "reason": "Schwab returned no quote",
                }
            )
        else:
            rows.append(_quote_row(by_schwab[symbol], quote))
    screened = local_screen(rows, orats_capacity=orats_capacity)
    result = {
        "schema": "cultra.broad-schwab-screen.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": {key: value for key, value in universe.items() if key != "holdings"},
        "counts": {
            "source_constituents": universe["constituent_count"],
            "schwab_quotes": len(rows),
            "data_unavailable": len(missing),
            "locally_rejected": len(screened["locally_rejected"]),
            "orats_admitted": len(screened["admitted"]),
            "budget_unresolved": len(screened["budget_unresolved"]),
        },
        "orats_admitted_symbols": [item["ticker"] for item in screened["admitted"]],
        "quotes": rows,
        "data_unavailable": missing,
        "locally_rejected": list(screened["locally_rejected"]),
        "budget_unresolved": list(screened["budget_unresolved"]),
        "admitted": list(screened["admitted"]),
    }
    _private_json(output_path, result)
    return result


def rebuild_broad_screen_offline(
    *, source_path: Path, output_path: Path
) -> Mapping[str, Any]:
    """Remove a legacy capacity cut from an already saved quote snapshot.

    This is a pure provenance-preserving transformation: it re-runs only the
    deterministic local liquidity rules against the exact saved quote rows.
    It does not refresh quotes or contact Schwab or ORATS.
    """

    source = Path(source_path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise UniverseError("saved broad-screen artifact is unreadable") from exc
    if payload.get("schema") not in {
        "cultra.broad-schwab-screen.v1",
        "cultra.broad-schwab-screen.v2",
    }:
        raise UniverseError("saved broad-screen schema is unsupported")
    quotes = payload.get("quotes")
    unavailable = payload.get("data_unavailable")
    universe = payload.get("universe")
    if not isinstance(quotes, list) or not isinstance(unavailable, list):
        raise UniverseError("saved broad-screen rows are incomplete")
    if not isinstance(universe, dict):
        raise UniverseError("saved broad-screen universe provenance is missing")
    expected = int(universe.get("constituent_count", -1))
    if expected <= 0 or len(quotes) + len(unavailable) != expected:
        raise UniverseError("saved broad-screen universe does not reconcile")
    tickers = [str(item.get("ticker", "")) for item in quotes]
    if not all(tickers) or len(tickers) != len(set(tickers)):
        raise UniverseError("saved broad-screen quote tickers are invalid or duplicated")

    screened = local_screen(quotes)
    result = {
        "schema": "cultra.broad-schwab-screen.v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "universe": universe,
        "counts": {
            "source_constituents": expected,
            "schwab_quotes": len(quotes),
            "data_unavailable": len(unavailable),
            "locally_rejected": len(screened["locally_rejected"]),
            "orats_admitted": len(screened["admitted"]),
            "budget_unresolved": 0,
        },
        "orats_admitted_symbols": [
            item["ticker"] for item in screened["admitted"]
        ],
        "quotes": quotes,
        "data_unavailable": unavailable,
        "locally_rejected": list(screened["locally_rejected"]),
        "budget_unresolved": [],
        "admitted": list(screened["admitted"]),
        "offline_rebuild": {
            "network_attempted": False,
            "source_path": str(source),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "source_generated_at": payload.get("generated_at"),
            "transformation": "reapply deterministic local screen without capacity",
        },
    }
    _private_json(output_path, result)
    return result


def _history_metrics(symbol: str, bars: Sequence[Any]) -> Mapping[str, Any]:
    if len(bars) < 61:
        raise UniverseError("%s history contains fewer than 61 sessions" % symbol)
    closes = [float(item.close) for item in bars]
    highs = [float(item.high) for item in bars]
    lows = [float(item.low) for item in bars]
    returns = [math.log(right / left) for left, right in zip(closes, closes[1:])]
    gains = [max(right - left, 0.0) for left, right in zip(closes[-15:-1], closes[-14:])]
    losses = [max(left - right, 0.0) for left, right in zip(closes[-15:-1], closes[-14:])]
    average_gain = math.fsum(gains) / 14.0
    average_loss = math.fsum(losses) / 14.0
    rsi = 100.0 if average_loss == 0.0 else 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    true_ranges = []
    for index in range(len(bars) - 20, len(bars)):
        previous = float(bars[index - 1].close)
        item = bars[index]
        true_ranges.append(
            max(
                float(item.high) - float(item.low),
                abs(float(item.high) - previous),
                abs(float(item.low) - previous),
            )
        )
    sma20 = math.fsum(closes[-20:]) / 20.0
    sma50 = math.fsum(closes[-50:]) / 50.0
    momentum20 = closes[-1] / closes[-21] - 1.0
    momentum60 = closes[-1] / closes[-61] - 1.0
    sixty_range = max(highs[-60:]) - min(lows[-60:])
    return {
        "schwab_symbol": symbol,
        "ticker": symbol.replace("/", "."),
        "first_session": bars[0].timestamp.date().isoformat(),
        "last_session": bars[-1].timestamp.date().isoformat(),
        "sessions": len(bars),
        "last_close": closes[-1],
        "momentum_20": momentum20,
        "momentum_60": momentum60,
        "sma20": sma20,
        "sma50": sma50,
        "realized_volatility_20": statistics.stdev(returns[-20:]) * math.sqrt(252.0),
        "average_volume_20": math.fsum(float(item.volume) for item in bars[-20:]) / 20.0,
        "average_dollar_volume_20": math.fsum(
            float(item.volume) * float(item.close) for item in bars[-20:]
        )
        / 20.0,
        "atr20_fraction": math.fsum(true_ranges) / 20.0 / closes[-1],
        "rsi14": rsi,
        "breakout_position_60": (
            (closes[-1] - min(lows[-60:])) / sixty_range if sixty_range > 0 else 0.5
        ),
        "trend_score": momentum20
        + 0.5 * momentum60
        + 0.5 * (closes[-1] / sma20 - 1.0)
        + 0.25 * (sma20 / sma50 - 1.0),
    }


def fetch_history_snapshot(
    *,
    screen_path: Path,
    output_path: Path,
    as_of: date,
    workers: int = 4,
    provider_factory: Optional[Callable[[], SchwabMarketDataBoundary]] = None,
) -> Mapping[str, Any]:
    """Fetch 180 calendar days for every symbol admitted to the Core funnel."""

    if not 1 <= int(workers) <= 4:
        raise UniverseError("history workers must be between 1 and 4")
    screen = json.loads(Path(screen_path).read_text(encoding="utf-8"))
    symbols = tuple(str(item["schwab_symbol"]) for item in screen["admitted"])
    factory = provider_factory or (
        lambda: SchwabMarketDataBoundary(SchwabHTTPProvider.production(timeout_seconds=30.0))
    )

    def fetch(symbol: str) -> Mapping[str, Any]:
        bars = factory().price_history(
            symbol, start=as_of - timedelta(days=180), end=as_of
        )
        return _history_metrics(symbol, bars)

    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows.append(future.result())
            except BaseException as exc:
                errors.append(
                    {"symbol": symbol, "reason": "%s: %s" % (type(exc).__name__, str(exc))}
                )
    result = {
        "schema": "cultra.broad-history-screen.v1",
        "source": "Schwab price history read-only",
        "requested": len(symbols),
        "resolved": len(rows),
        "errors": sorted(errors, key=lambda item: item["symbol"]),
        "rows": sorted(rows, key=lambda item: item["ticker"]),
    }
    _private_json(output_path, result)
    return result


def fetch_finalist_chains(
    *,
    symbols: Sequence[str],
    output_path: Path,
    from_date: date,
    to_date: date,
    workers: int = 4,
    provider_factory: Optional[Callable[[], SchwabMarketDataBoundary]] = None,
    decision_refresh: bool = False,
) -> Mapping[str, Any]:
    """Fetch exact Schwab chains for an already frozen finalist set."""

    normalized = tuple(sorted(set(str(item).strip().upper() for item in symbols)))
    if not normalized:
        raise UniverseError("finalist chain set cannot be empty")
    if to_date < from_date or not 1 <= int(workers) <= 4:
        raise UniverseError("invalid finalist chain request")
    factory = provider_factory or (
        lambda: SchwabMarketDataBoundary(SchwabHTTPProvider.production(timeout_seconds=45.0))
    )

    def fetch(symbol: str) -> Mapping[str, Any]:
        chain = factory().option_chain(symbol, from_date=from_date, to_date=to_date)
        return {
            "ticker": symbol,
            "underlying_quote": {
                "bid": chain.underlying_quote.bid,
                "ask": chain.underlying_quote.ask,
                "last": chain.underlying_quote.last,
                "timestamp": chain.underlying_quote.timestamp.isoformat(),
            },
            "chain_timestamp": chain.timestamp.isoformat(),
            "contracts": [
                {
                    "occ_symbol": item.occ_symbol,
                    "expiration": item.expiration.isoformat(),
                    "strike": item.strike,
                    "option_type": item.option_type,
                    "bid": item.bid,
                    "ask": item.ask,
                    "timestamp": item.timestamp.isoformat(),
                    "volume": item.volume,
                    "open_interest": item.open_interest,
                    "delta": item.delta,
                }
                for item in chain.contracts
            ],
        }

    rows = []
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch, symbol): symbol for symbol in normalized}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows.append(future.result())
            except BaseException as exc:
                errors.append(
                    {"ticker": symbol, "reason": "%s: %s" % (type(exc).__name__, str(exc))}
                )
    quote_timestamps = [
        str(row["underlying_quote"]["timestamp"]) for row in rows
    ] + [
        str(contract["timestamp"])
        for row in rows
        for contract in row["contracts"]
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    result = {
        "schema": "cultra.finalist-schwab-chains.v1",
        "generated_at": generated_at,
        "requested": list(normalized),
        "resolved_count": len(rows),
        "error_count": len(errors),
        "errors": sorted(errors, key=lambda item: item["ticker"]),
        "chains": sorted(rows, key=lambda item: item["ticker"]),
    }
    if decision_refresh:
        result["decision_quote_refresh"] = {
            "source": "SCHWAB",
            "purpose": "MARKET_OPEN_DECISION",
            "complete": not errors and len(rows) == len(normalized),
            "requested_symbols": list(normalized),
            "resolved_symbols": sorted(str(item["ticker"]) for item in rows),
            "refreshed_at": generated_at,
            "oldest_quote_timestamp": min(quote_timestamps) if quote_timestamps else None,
            "newest_quote_timestamp": max(quote_timestamps) if quote_timestamps else None,
            "broker_order_surface": False,
        }
    _private_json(output_path, result)
    return result


__all__ = [
    "DEFAULT_HOLDINGS",
    "UniverseError",
    "fetch_broad_quote_snapshot",
    "fetch_finalist_chains",
    "fetch_history_snapshot",
    "load_spy_holdings",
    "local_screen",
    "rebuild_broad_screen_offline",
]
