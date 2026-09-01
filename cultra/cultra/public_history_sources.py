"""Offline audit of Cultra-owned public historical prerequisite sources.

The acquisition step is deliberately outside this module.  This module never
opens a socket: it validates the exact preserved Cboe, NYSE, OCC, SEC, and
Nasdaq bytes, derives only what those bytes can support, and records the gaps
that still prevent a historical prerequisite freeze.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple
from zoneinfo import ZoneInfo

from .artifacts import (
    ArtifactError,
    ArtifactWriter,
    assert_secret_free_bytes,
    canonical_json_bytes,
    verify_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = (PROJECT_ROOT / "out").resolve()
CAMPAIGN_START = date(2024, 11, 11)
CAMPAIGN_END = date(2026, 8, 28)
SELECTION_DATES = (
    date(2024, 11, 11),
    date(2025, 5, 7),
    date(2025, 10, 28),
    date(2026, 4, 22),
)
_CBOE_FIELDS = (
    "Trade Date",
    "Options Class",
    "Underlying",
    "Product Type",
    "Exchange",
    "Volume",
)
_CBOE_EXCHANGES = frozenset({"BATS", "C2", "CBOE", "EDGX"})
_CBOE_PRODUCT_TYPES = frozenset({"I", "S"})
_MINIMUM_DAILY_CBOE_VOLUME = 1000
_MINIMUM_CBOE_VENUES = 2
_OCC_FIELDS = ("number", "post date", "ex/eff date", " title")
_DLP_UNAVAILABLE = "File requested does not exist."

# These closures and early closes are transcribed from the preserved official
# NYSE yearly calendars.  The January 9, 2025 exception is bound separately to
# the preserved NYSE National Day of Mourning memo.
_NYSE_CLOSURES = frozenset(
    {
        date(2024, 11, 28),
        date(2024, 12, 25),
        date(2025, 1, 1),
        date(2025, 1, 9),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
    }
)
_NYSE_EARLY_CLOSES = frozenset(
    {
        date(2024, 11, 29),
        date(2024, 12, 24),
        date(2025, 7, 3),
        date(2025, 11, 28),
        date(2025, 12, 24),
    }
)


class PublicHistorySourceError(ValueError):
    """A preserved public source bundle is missing, malformed, or misleading."""


@dataclass(frozen=True)
class RawSourceSpec:
    relative_path: str
    role: str
    source_uri: str
    media_type: str


@dataclass(frozen=True)
class PublicSourceAnalysis:
    receipt: Mapping[str, Any]
    discovery: Mapping[str, Any]
    classification_queue: Mapping[str, Any]
    calendar: Mapping[str, Any]
    adjustment_index: Mapping[str, Any]
    audit: Mapping[str, Any]
    markdown: str


@dataclass(frozen=True)
class SavedPublicSourceAudit:
    run_dir: Path
    manifest_path: Path
    audit_path: Path
    board_path: Path
    status: str


def _cboe_specs() -> Tuple[RawSourceSpec, ...]:
    values = []
    for selection in SELECTION_DATES:
        month = selection.month
        name = "cboe_all_symbols_daily_%d_%d_%s.csv" % (
            selection.year,
            month,
            selection.isoformat(),
        )
        uri = (
            "https://www.cboe.com/us/options/market_statistics/historical_data/"
            "download/all_symbols/?reportType=volume&month=%d&year=%d&"
            "volumeType=sum&volumeAggType=daily&exchanges=CBOE&exchanges=BATS&"
            "exchanges=C2&exchanges=EDGX"
        ) % (month, selection.year)
        values.append(
            RawSourceSpec(
                "cboe_volume/%s" % name,
                "CBOE_ALL_SYMBOLS_DAILY_VOLUME_%s" % selection.isoformat(),
                uri,
                "text/csv",
            )
        )
    return tuple(values)


_OCC_SLICES = (
    (date(2024, 11, 11), date(2025, 1, 31)),
    (date(2025, 2, 1), date(2025, 4, 30)),
    (date(2025, 5, 1), date(2025, 7, 31)),
    (date(2025, 8, 1), date(2025, 10, 31)),
    (date(2025, 11, 1), date(2026, 1, 31)),
    (date(2026, 2, 1), date(2026, 4, 30)),
    (date(2026, 5, 1), date(2026, 6, 30)),
    (date(2026, 7, 1), date(2026, 8, 28)),
)


def _raw_specs() -> Tuple[RawSourceSpec, ...]:
    values: List[RawSourceSpec] = list(_cboe_specs())
    values.extend(
        [
            RawSourceSpec(
                "nyse_calendars/nyse_2024_trading_calendar.pdf",
                "NYSE_2024_YEARLY_TRADING_CALENDAR",
                "https://www.nyse.com/publicdocs/ICE_NYSE_2024_Yearly_Trading_Calendar.pdf",
                "application/pdf",
            ),
            RawSourceSpec(
                "nyse_calendars/nyse_2025_trading_calendar.pdf",
                "NYSE_2025_YEARLY_TRADING_CALENDAR",
                "https://www.nyse.com/publicdocs/ICE_NYSE_2025_Yearly_Trading_Calendar.pdf",
                "application/pdf",
            ),
            RawSourceSpec(
                "nyse_calendars/nyse_2026_trading_calendar.pdf",
                "NYSE_2026_YEARLY_TRADING_CALENDAR",
                "https://www.nyse.com/publicdocs/nyse/ICE_NYSE_2026_Yearly_Trading_Calendar.pdf",
                "application/pdf",
            ),
            RawSourceSpec(
                "nyse_calendars/nyse_2025_national_day_of_mourning.pdf",
                "NYSE_2025_NATIONAL_DAY_OF_MOURNING_MEMO",
                "https://www.nyse.com/publicdocs/nyse/markets/american-options/"
                "rule-interpretations/2025/National_Day_of_Mourning_20250102.pdf",
                "application/pdf",
            ),
        ]
    )
    for selection in SELECTION_DATES:
        compact = selection.strftime("%Y%m%d")
        values.append(
            RawSourceSpec(
                "occ_dlp_tombstones/occ_dlp_%s.txt" % selection.isoformat(),
                "OCC_DLP_UNAVAILABLE_%s" % selection.isoformat(),
                "https://marketdata.theocc.com/daily-delo-download?"
                "reportDate=%s&format=xml" % compact,
                "text/plain",
            )
        )
    for start, end in _OCC_SLICES:
        values.append(
            RawSourceSpec(
                "occ_info_memos/occ_contract_adjustments_%s_to_%s.csv"
                % (start.isoformat(), end.isoformat()),
                "OCC_CONTRACT_ADJUSTMENT_OPTIONS_INDEX_%s_TO_%s"
                % (start.isoformat(), end.isoformat()),
                "https://infomemo.theocc.com/infomemo/search",
                "text/csv",
            )
        )
    values.extend(
        [
            RawSourceSpec(
                "reference_current/sec_company_tickers_exchange.json",
                "SEC_CURRENT_COMPANY_TICKER_EXCHANGE_REFERENCE",
                "https://www.sec.gov/files/company_tickers_exchange.json",
                "application/json",
            ),
            RawSourceSpec(
                "reference_current/nasdaqlisted.txt",
                "NASDAQ_CURRENT_LISTED_SYMBOL_DIRECTORY",
                "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                "text/plain",
            ),
            RawSourceSpec(
                "reference_current/otherlisted.txt",
                "NASDAQ_CURRENT_OTHER_LISTED_SYMBOL_DIRECTORY",
                "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
                "text/plain",
            ),
        ]
    )
    return tuple(values)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _owned_root(path: Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise PublicHistorySourceError("%s cannot be a symlink" % label)
    supplied = candidate.resolve()
    try:
        supplied.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PublicHistorySourceError("%s must be Cultra-owned" % label) from exc
    if not supplied.is_dir() or supplied.is_symlink():
        raise PublicHistorySourceError("%s is unavailable or is a symlink" % label)
    return supplied


def _owned_file(root: Path, relative_path: str) -> Path:
    candidate = root / relative_path
    if candidate.is_symlink():
        raise PublicHistorySourceError("raw source cannot be a symlink: %s" % relative_path)
    try:
        supplied = candidate.resolve(strict=True)
        supplied.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PublicHistorySourceError("raw source is unavailable: %s" % relative_path) from exc
    if not supplied.is_file():
        raise PublicHistorySourceError("raw source is not a file: %s" % relative_path)
    return supplied


def _source_receipt(root: Path) -> Mapping[str, Any]:
    expected = _raw_specs()
    expected_paths = {item.relative_path for item in expected}
    actual_paths = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    missing = sorted(expected_paths.difference(actual_paths))
    unexpected = sorted(actual_paths.difference(expected_paths))
    if missing or unexpected:
        raise PublicHistorySourceError(
            "public source inventory mismatch; missing=%s unexpected=%s"
            % (",".join(missing) or "NONE", ",".join(unexpected) or "NONE")
        )
    artifacts = []
    for spec in sorted(expected, key=lambda item: item.relative_path):
        path = _owned_file(root, spec.relative_path)
        data = path.read_bytes()
        try:
            assert_secret_free_bytes(data, path=spec.relative_path)
        except ArtifactError as exc:
            raise PublicHistorySourceError(
                "raw source contains credential-shaped material: %s" % spec.relative_path
            ) from exc
        artifacts.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "role": spec.role,
                "source_uri": spec.source_uri,
                "media_type": spec.media_type,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    payload = {
        "schema": "cultra.public-history-source-receipt.v1",
        "source_root": root.relative_to(PROJECT_ROOT).as_posix(),
        "raw_artifact_count": len(artifacts),
        "artifacts": artifacts,
        "transport_headers_preserved": False,
        "audit_network_attempted": False,
        "orats_attempts": 0,
        "schwab_attempts": 0,
        "paid_data_attempts": 0,
    }
    return dict(payload, receipt_hash=hashlib.sha256(_canonical(payload)).hexdigest())


def _market_dates(start: date, end: date) -> Tuple[date, ...]:
    result = []
    current = start
    while current <= end:
        if current.weekday() < 5 and current not in _NYSE_CLOSURES:
            result.append(current)
        current += timedelta(days=1)
    return tuple(result)


def _month_market_dates(year: int, month: int) -> Set[date]:
    current = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = date(year, month + 1, 1) - timedelta(days=1)
    return set(_market_dates(current, end))


def _parse_cboe(
    root: Path,
    *,
    minimum_security_underlyings: int,
    minimum_liquid_candidates: int,
    require_complete_month: bool,
) -> Mapping[str, Any]:
    snapshots = []
    for selection, spec in zip(SELECTION_DATES, _cboe_specs()):
        path = _owned_file(root, spec.relative_path)
        by_underlying: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total_volume": 0, "venues": set(), "option_classes": set()}
        )
        seen_selected_rows = set()
        present_dates: Set[date] = set()
        row_count = 0
        try:
            handle = path.open("r", encoding="utf-8-sig", newline="")
        except (OSError, UnicodeError) as exc:
            raise PublicHistorySourceError("Cboe source is unreadable") from exc
        with handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _CBOE_FIELDS:
                raise PublicHistorySourceError("Cboe CSV fields changed: %s" % spec.relative_path)
            for row in reader:
                row_count += 1
                if set(row) != set(_CBOE_FIELDS):
                    raise PublicHistorySourceError("Cboe row fields changed")
                try:
                    trade_date = datetime.strptime(row["Trade Date"], "%Y/%m/%d").date()
                    volume = int(row["Volume"])
                except (TypeError, ValueError) as exc:
                    raise PublicHistorySourceError("Cboe row is malformed") from exc
                if trade_date.year != selection.year or trade_date.month != selection.month:
                    raise PublicHistorySourceError("Cboe monthly export contains an outside date")
                present_dates.add(trade_date)
                product_type = str(row["Product Type"]).strip()
                exchange = str(row["Exchange"]).strip()
                option_class = str(row["Options Class"]).strip()
                underlying = str(row["Underlying"]).strip().upper()
                if (
                    product_type not in _CBOE_PRODUCT_TYPES
                    or exchange not in _CBOE_EXCHANGES
                    or not option_class
                    or not underlying
                    or volume < 0
                ):
                    raise PublicHistorySourceError("Cboe row contains an unsupported value")
                if trade_date != selection or product_type != "S" or volume <= 0:
                    continue
                identity = (
                    row["Trade Date"],
                    option_class,
                    underlying,
                    product_type,
                    exchange,
                )
                if identity in seen_selected_rows:
                    raise PublicHistorySourceError(
                        "Cboe selected-date row is duplicated: %s" % (identity,)
                    )
                seen_selected_rows.add(identity)
                aggregate = by_underlying[underlying]
                aggregate["total_volume"] += volume
                aggregate["venues"].add(exchange)
                aggregate["option_classes"].add(option_class)
        if row_count == 0 or selection not in present_dates:
            raise PublicHistorySourceError("Cboe selection date is absent")
        if require_complete_month and present_dates != _month_market_dates(
            selection.year, selection.month
        ):
            missing_dates = sorted(
                _month_market_dates(selection.year, selection.month).difference(present_dates)
            )
            extra_dates = sorted(
                present_dates.difference(_month_market_dates(selection.year, selection.month))
            )
            raise PublicHistorySourceError(
                "Cboe month is incomplete for %s; missing=%s extra=%s"
                % (
                    selection.isoformat(),
                    ",".join(item.isoformat() for item in missing_dates) or "NONE",
                    ",".join(item.isoformat() for item in extra_dates) or "NONE",
                )
            )
        ranked = sorted(
            by_underlying.items(),
            key=lambda item: (-int(item[1]["total_volume"]), item[0]),
        )
        if len(ranked) < minimum_security_underlyings:
            raise PublicHistorySourceError(
                "Cboe selection-date population is not broad: %s has %d"
                % (selection.isoformat(), len(ranked))
            )
        count = len(ranked)
        eligible_tickers = [
            ticker
            for ticker, item in ranked
            if int(item["total_volume"]) >= _MINIMUM_DAILY_CBOE_VOLUME
            and len(item["venues"]) >= _MINIMUM_CBOE_VENUES
        ]
        if len(eligible_tickers) < minimum_liquid_candidates:
            raise PublicHistorySourceError(
                "Cboe liquid candidate frame is not broad: %s has %d"
                % (selection.isoformat(), len(eligible_tickers))
            )
        eligible_rank = {
            ticker: index for index, ticker in enumerate(eligible_tickers, start=1)
        }
        eligible_count = len(eligible_tickers)
        members = []
        for index, (ticker, item) in enumerate(ranked, start=1):
            candidate_rank = eligible_rank.get(ticker)
            decile = (
                min(10, ((candidate_rank - 1) * 10) // eligible_count + 1)
                if candidate_rank is not None
                else None
            )
            members.append(
                {
                    "ticker": ticker,
                    "liquidity_rank": index,
                    "candidate_liquidity_rank": candidate_rank,
                    "liquidity_stratum": (
                        "ELIGIBLE_VOLUME_DECILE_%02d" % decile
                        if decile is not None
                        else "BELOW_EXECUTION_RESEARCH_FLOOR"
                    ),
                    "total_cboe_options_volume": int(item["total_volume"]),
                    "venue_count": len(item["venues"]),
                    "option_class_count": len(item["option_classes"]),
                    "option_activity_observed": True,
                    "liquidity_eligible": candidate_rank is not None,
                    "asset_type": "UNRESOLVED_STOCK_OR_ETP",
                }
            )
        snapshots.append(
            {
                "selection_date": selection.isoformat(),
                "source_file": spec.relative_path,
                "source_sha256": _sha256(path),
                "monthly_raw_row_count": row_count,
                "monthly_trade_date_count": len(present_dates),
                "security_underlying_count": count,
                "liquid_candidate_count": eligible_count,
                "members": members,
            }
        )
    return {
        "schema": "cultra.cboe-point-in-time-option-activity-discovery.v2",
        "provider": "CBOE",
        "coverage": "POSITIVE_OPTIONS_VOLUME_ON_FOUR_CBOE_VENUES",
        "exchanges": sorted(_CBOE_EXCHANGES),
        "selection_dates": [item.isoformat() for item in SELECTION_DATES],
        "asset_classification_status": "UNRESOLVED_STOCK_OR_ETP",
        "exhaustive_all_us_options_venues": False,
        "liquidity_policy": {
            "minimum_total_daily_cboe_options_volume": _MINIMUM_DAILY_CBOE_VOLUME,
            "minimum_cboe_venues": _MINIMUM_CBOE_VENUES,
            "fixed_candidate_count": None,
            "purpose": "EXECUTION_RELEVANT_RESEARCH_FRAME_NOT_TICKET_SUPPRESSION",
        },
        "current_reference_files_used_for_historical_classification": False,
        "fixed_name_list_used": False,
        "top_n_suppression_used": False,
        "snapshots": snapshots,
    }


def _build_calendar(root: Path) -> Mapping[str, Any]:
    source_files = [
        "nyse_calendars/nyse_2024_trading_calendar.pdf",
        "nyse_calendars/nyse_2025_trading_calendar.pdf",
        "nyse_calendars/nyse_2026_trading_calendar.pdf",
        "nyse_calendars/nyse_2025_national_day_of_mourning.pdf",
    ]
    source_artifacts = []
    for relative in source_files:
        path = _owned_file(root, relative)
        payload = path.read_bytes()
        if (
            len(payload) < 1000
            or not payload.startswith(b"%PDF-")
            or b"%%EOF" not in payload[-1024:]
        ):
            raise PublicHistorySourceError("NYSE source is not a valid preserved PDF: %s" % relative)
        source_artifacts.append(
            {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
        )
    ny = ZoneInfo("America/New_York")
    session_dates = _market_dates(CAMPAIGN_START, CAMPAIGN_END)
    if len(session_dates) != 450:
        raise PublicHistorySourceError("derived NYSE campaign calendar is not 450 sessions")
    derived_selection_dates = tuple(session_dates[index] for index in (0, 120, 240, 360))
    if derived_selection_dates != SELECTION_DATES:
        raise PublicHistorySourceError("derived cohort selection dates drifted")
    sessions = []
    for session_date in session_dates:
        close_time = time(13, 0) if session_date in _NYSE_EARLY_CLOSES else time(16, 0)
        sessions.append(
            {
                "session_date": session_date.isoformat(),
                "close_at": datetime.combine(session_date, close_time, ny).isoformat(),
                "early_close": session_date in _NYSE_EARLY_CLOSES,
            }
        )
    blocks = []
    for index, selection in enumerate(SELECTION_DATES):
        block_start_index = index * 120
        block_end_index = min(block_start_index + 119, len(session_dates) - 1)
        blocks.append(
            {
                "block": index + 1,
                "selection_date": selection.isoformat(),
                "block_start": session_dates[block_start_index].isoformat(),
                "block_end": session_dates[block_end_index].isoformat(),
                "session_count": block_end_index - block_start_index + 1,
            }
        )
    payload = {
        "schema": "cultra.public-xnys-session-calendar-candidate.v1",
        "exchange": "XNYS",
        "timezone": "America/New_York",
        "campaign_start": CAMPAIGN_START.isoformat(),
        "campaign_end": CAMPAIGN_END.isoformat(),
        "session_count": len(sessions),
        "selection_dates": [item.isoformat() for item in SELECTION_DATES],
        "source_artifacts": source_artifacts,
        "closures": sorted(item.isoformat() for item in _NYSE_CLOSURES),
        "early_closes": sorted(
            item.isoformat()
            for item in _NYSE_EARLY_CLOSES
            if CAMPAIGN_START <= item <= CAMPAIGN_END
        ),
        "cohort_blocks": blocks,
        "sessions": sessions,
    }
    return dict(payload, calendar_hash=hashlib.sha256(_canonical(payload)).hexdigest())


def _build_classification_queue(discovery: Mapping[str, Any]) -> Mapping[str, Any]:
    """Freeze a full no-outcome traversal order for point-in-time classification."""

    snapshots = []
    for snapshot in discovery["snapshots"]:
        eligible = [item for item in snapshot["members"] if item["liquidity_eligible"]]
        by_stratum: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for item in sorted(
            eligible,
            key=lambda value: (value["candidate_liquidity_rank"], value["ticker"]),
        ):
            by_stratum[str(item["liquidity_stratum"])].append(item)
        ordered = []
        strata = sorted(by_stratum)
        while any(by_stratum.values()):
            for stratum in strata:
                if by_stratum[stratum]:
                    ordered.append(by_stratum[stratum].pop(0))
        queue = [
            {
                "queue_position": index,
                "ticker": item["ticker"],
                "candidate_liquidity_rank": item["candidate_liquidity_rank"],
                "liquidity_stratum": item["liquidity_stratum"],
                "total_cboe_options_volume": item["total_cboe_options_volume"],
                "venue_count": item["venue_count"],
                "classification_status": "PENDING_POINT_IN_TIME_PRIMARY_EVIDENCE",
            }
            for index, item in enumerate(ordered, start=1)
        ]
        snapshots.append(
            {
                "selection_date": snapshot["selection_date"],
                "queue_count": len(queue),
                "queue": queue,
            }
        )
    payload = {
        "schema": "cultra.point-in-time-classification-queue.v1",
        "selection_policy": "COMPLETE_LIQUID_FRAME_ROUND_ROBIN_BY_VOLUME_DECILE",
        "queue_truncated": False,
        "fixed_name_list_used": False,
        "outcome_data_used": False,
        "classification_stop_rule": {
            "cohort_size": 10,
            "minimum_verified_stocks": 8,
            "minimum_resolved_asset_types": 10,
            "unresolved_names_preserved": True,
            "continue_after_unresolved_or_duplicate": True,
        },
        "snapshots": snapshots,
    }
    return dict(payload, queue_hash=hashlib.sha256(_canonical(payload)).hexdigest())


def _parse_occ(root: Path) -> Mapping[str, Any]:
    records = []
    seen_numbers = set()
    slices = []
    previous_end = None
    for start, end in _OCC_SLICES:
        if previous_end is not None and start != previous_end + timedelta(days=1):
            raise PublicHistorySourceError("OCC search slices are not contiguous")
        previous_end = end
        relative = "occ_info_memos/occ_contract_adjustments_%s_to_%s.csv" % (
            start.isoformat(),
            end.isoformat(),
        )
        path = _owned_file(root, relative)
        count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _OCC_FIELDS:
                raise PublicHistorySourceError("OCC memo export fields changed: %s" % relative)
            for row in reader:
                count += 1
                if set(row) != set(_OCC_FIELDS):
                    raise PublicHistorySourceError("OCC memo row fields changed")
                number = str(row["number"]).strip()
                title = str(row[" title"]).strip()
                try:
                    post_date = datetime.strptime(
                        str(row["post date"]).strip(), "%b-%d-%Y"
                    ).date()
                except ValueError as exc:
                    raise PublicHistorySourceError("OCC memo post date is malformed") from exc
                effective_text = str(row["ex/eff date"]).strip()
                try:
                    effective_date = (
                        datetime.strptime(effective_text, "%b-%d-%Y").date()
                        if effective_text
                        else None
                    )
                except ValueError as exc:
                    raise PublicHistorySourceError("OCC memo effective date is malformed") from exc
                if not number.isdigit() or not title or not (start <= post_date <= end):
                    raise PublicHistorySourceError("OCC memo row is outside its frozen slice")
                if number in seen_numbers:
                    raise PublicHistorySourceError("OCC memo number is duplicated: %s" % number)
                seen_numbers.add(number)
                records.append(
                    {
                        "memo_number": number,
                        "post_date": post_date.isoformat(),
                        "effective_date": effective_date.isoformat() if effective_date else None,
                        "title": title,
                        "detail_retrieved": False,
                        "source_file": relative,
                    }
                )
        if count <= 0 or count >= 2500:
            raise PublicHistorySourceError(
                "OCC memo slice is empty or reached the 2500-row export cap"
            )
        slices.append(
            {
                "start": start.isoformat(),
                "end": end.isoformat(),
                "row_count": count,
                "source_file": relative,
                "source_sha256": _sha256(path),
            }
        )
    if _OCC_SLICES[0][0] != CAMPAIGN_START or _OCC_SLICES[-1][1] != CAMPAIGN_END:
        raise PublicHistorySourceError("OCC memo slices do not span the campaign")
    records.sort(key=lambda item: (item["post_date"], item["memo_number"]))
    return {
        "schema": "cultra.occ-contract-adjustment-index.v1",
        "provider": "OCC",
        "search_page": "https://infomemo.theocc.com/infomemo/search",
        "search_category": "Contract Adjustment",
        "search_topic": "Options",
        "campaign_start": CAMPAIGN_START.isoformat(),
        "campaign_end": CAMPAIGN_END.isoformat(),
        "slice_count": len(slices),
        "memo_count": len(records),
        "export_cap_reached": False,
        "complete_non_overlapping_post_date_slices": True,
        "detail_retrieval_status": "PENDING_FOR_SELECTED_SYMBOLS",
        "slices": slices,
        "records": records,
    }


def _parse_dlp_tombstones(root: Path) -> List[Mapping[str, Any]]:
    result = []
    for selection in SELECTION_DATES:
        relative = "occ_dlp_tombstones/occ_dlp_%s.txt" % selection.isoformat()
        path = _owned_file(root, relative)
        text = path.read_text(encoding="utf-8").strip()
        if text != _DLP_UNAVAILABLE:
            raise PublicHistorySourceError("OCC DLP negative response changed")
        result.append(
            {
                "selection_date": selection.isoformat(),
                "source_file": relative,
                "source_sha256": _sha256(path),
                "status": "HISTORICAL_FILE_NOT_RETAINED_AT_ENDPOINT",
            }
        )
    return result


def _parse_pipe_directory(path: Path, expected_fields: Sequence[str]) -> Mapping[str, Any]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 3:
        raise PublicHistorySourceError("Nasdaq symbol directory is empty")
    if tuple(lines[0].split("|")) != tuple(expected_fields):
        raise PublicHistorySourceError("Nasdaq symbol directory fields changed")
    if not lines[-1].startswith("File Creation Time: "):
        raise PublicHistorySourceError("Nasdaq symbol directory trailer is missing")
    rows = []
    for line in lines[1:-1]:
        values = line.split("|")
        if len(values) != len(expected_fields):
            raise PublicHistorySourceError("Nasdaq symbol directory row is malformed")
        rows.append(dict(zip(expected_fields, values)))
    etf_count = sum(1 for item in rows if item.get("ETF") == "Y")
    invalid_flags = sorted(set(item.get("ETF", "") for item in rows).difference({"N", "Y"}))
    if invalid_flags:
        raise PublicHistorySourceError("Nasdaq ETF flag contains unsupported values")
    return {
        "row_count": len(rows),
        "etf_count": etf_count,
        "non_etf_count": len(rows) - etf_count,
        "file_creation_time": lines[-1].split("|", 1)[0].split(": ", 1)[1],
    }


def _parse_current_references(root: Path) -> Mapping[str, Any]:
    sec_path = _owned_file(
        root, "reference_current/sec_company_tickers_exchange.json"
    )
    try:
        sec = json.loads(sec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicHistorySourceError("SEC ticker reference is unreadable") from exc
    if (
        not isinstance(sec, Mapping)
        or sec.get("fields") != ["cik", "name", "ticker", "exchange"]
        or not isinstance(sec.get("data"), list)
    ):
        raise PublicHistorySourceError("SEC ticker reference fields changed")
    for row in sec["data"]:
        if not isinstance(row, list) or len(row) != 4:
            raise PublicHistorySourceError("SEC ticker reference row is malformed")
    nasdaq = _parse_pipe_directory(
        _owned_file(root, "reference_current/nasdaqlisted.txt"),
        (
            "Symbol",
            "Security Name",
            "Market Category",
            "Test Issue",
            "Financial Status",
            "Round Lot Size",
            "ETF",
            "NextShares",
        ),
    )
    other = _parse_pipe_directory(
        _owned_file(root, "reference_current/otherlisted.txt"),
        (
            "ACT Symbol",
            "Security Name",
            "Exchange",
            "CQS Symbol",
            "ETF",
            "Round Lot Size",
            "Test Issue",
            "NASDAQ Symbol",
        ),
    )
    return {
        "status": "CURRENT_ONLY_DIAGNOSTIC_NOT_HISTORICAL_TRUTH",
        "used_for_historical_universe": False,
        "sec_company_ticker_rows": len(sec["data"]),
        "nasdaq_listed": nasdaq,
        "other_listed": other,
    }


def _build_audit(
    receipt: Mapping[str, Any],
    discovery: Mapping[str, Any],
    classification_queue: Mapping[str, Any],
    calendar: Mapping[str, Any],
    adjustment_index: Mapping[str, Any],
    dlp: Sequence[Mapping[str, Any]],
    references: Mapping[str, Any],
) -> Mapping[str, Any]:
    population_counts = {
        item["selection_date"]: item["security_underlying_count"]
        for item in discovery["snapshots"]
    }
    liquid_candidate_counts = {
        item["selection_date"]: item["liquid_candidate_count"]
        for item in discovery["snapshots"]
    }
    components = [
        {
            "component": "XNYS_450_SESSION_CALENDAR",
            "status": "READY",
            "evidence": "%d sessions; four expected selection dates"
            % calendar["session_count"],
        },
        {
            "component": "BROAD_POINT_IN_TIME_OPTION_ACTIVITY",
            "status": "READY_WITH_SCOPE_LIMIT",
            "evidence": "positive volume populations %s; variable liquid frames %s"
            % (
                ", ".join(
                    "%s=%s" % item for item in sorted(population_counts.items())
                ),
                ", ".join(
                    "%s=%s" % item
                    for item in sorted(liquid_candidate_counts.items())
                ),
            ),
        },
        {
            "component": "DETERMINISTIC_ASSET_CLASSIFICATION_QUEUE",
            "status": "READY",
            "evidence": "all variable-frame names are queued without outcomes or a top-N cutoff",
        },
        {
            "component": "POINT_IN_TIME_STOCK_ETP_CLASSIFICATION",
            "status": "BLOCKED",
            "evidence": "Cboe Product Type S does not distinguish stock from ETF/ETP; current references are not projected backward",
        },
        {
            "component": "OCC_CONTRACT_ADJUSTMENT_INDEX",
            "status": "READY_INDEX_ONLY",
            "evidence": "%d unique memos in eight non-overlapping slices"
            % adjustment_index["memo_count"],
        },
        {
            "component": "OCC_EXACT_ADJUSTMENT_DETAILS",
            "status": "PENDING_SELECTED_SYMBOLS",
            "evidence": "memo detail pages and deliverables have not been retrieved",
        },
        {
            "component": "EARNINGS_DIVIDENDS_SPLITS_DELISTINGS",
            "status": "BLOCKED_INCOMPLETE_COVERAGE",
            "evidence": "no complete point-in-time all-event source is preserved for the eventual cohort",
        },
        {
            "component": "HISTORICAL_PREREQUISITE_FREEZE",
            "status": "BLOCKED",
            "evidence": "universe classification and complete event coverage gates are unresolved",
        },
    ]
    blockers = [
        {
            "id": "POINT_IN_TIME_ASSET_TYPE",
            "impact": "Cannot enforce the stock-relevant cohort or label stocks versus ETFs without leakage.",
            "required_resolution": "Freeze a deterministic verification rule and prove each admitted cohort symbol's asset type from contemporaneous primary records.",
        },
        {
            "id": "ALL_VENUE_UNIVERSE_SCOPE",
            "impact": "The Cboe frame is broad but can omit a symbol whose volume occurred only on non-Cboe venues.",
            "required_resolution": "Accept and label the four-venue sampling frame before cohort selection, or acquire an all-US-options point-in-time directory.",
        },
        {
            "id": "COMPLETE_POINT_IN_TIME_EVENTS",
            "impact": "Historical exits, assignment, dividends, earnings, splits, mergers, symbol changes, and delistings cannot yet be adjusted or censored completely.",
            "required_resolution": "Acquire cohort-scoped contemporaneous event evidence with availability timestamps and exact OCC adjustment details, then attest coverage without empty-event defaults.",
        },
    ]
    payload = {
        "schema": "cultra.public-history-source-audit.v2",
        "as_of": "2026-08-31",
        "status": "PARTIAL_NOT_FREEZEABLE",
        "profit_confidence": "UNPROVEN",
        "manual_ticket_count": 0,
        "historical_campaign_authorized": False,
        "historical_campaign_expected_orats_attempts": 474,
        "recommended_orats_attempts_now": 0,
        "audit_network_attempted": False,
        "orats_attempts": 0,
        "schwab_attempts": 0,
        "paid_data_attempts": 0,
        "raw_artifact_count": receipt["raw_artifact_count"],
        "source_receipt_hash": receipt["receipt_hash"],
        "selection_dates": [item.isoformat() for item in SELECTION_DATES],
        "cboe_security_underlying_counts": population_counts,
        "cboe_liquid_candidate_counts": liquid_candidate_counts,
        "cboe_liquidity_policy": discovery["liquidity_policy"],
        "classification_queue_hash": classification_queue["queue_hash"],
        "classification_queue_counts": {
            item["selection_date"]: item["queue_count"]
            for item in classification_queue["snapshots"]
        },
        "occ_memo_count": adjustment_index["memo_count"],
        "occ_dlp_history": list(dlp),
        "current_reference_diagnostics": references,
        "fixed_name_list_used": False,
        "etf_only_universe_used": False,
        "top_n_suppression_used": False,
        "components": components,
        "blockers": blockers,
        "next_offline_gate": "Execute the frozen point-in-time classification queue against primary public evidence until each block has ten resolved names including eight verified stocks; then fetch cohort-scoped event evidence.",
    }
    return dict(payload, audit_hash=hashlib.sha256(_canonical(payload)).hexdigest())


def _markdown(audit: Mapping[str, Any]) -> str:
    component_icon = {
        "READY": "🟢",
        "READY_WITH_SCOPE_LIMIT": "🟡",
        "READY_INDEX_ONLY": "🟡",
        "PENDING_SELECTED_SYMBOLS": "🟡",
        "BLOCKED": "🔴",
        "BLOCKED_INCOMPLETE_COVERAGE": "🔴",
    }
    lines = [
        "# Cultra Public Historical Source Audit",
        "",
        "**Outcome: 🔴 `PARTIAL_NOT_FREEZEABLE`**",
        "",
        "The public acquisition produced a broad historical stock-or-ETP option-activity frame, a reproducible 450-session calendar, and a complete sliced OCC memo index. It did **not** prove point-in-time stock-versus-ETF classification or complete corporate-event coverage. Therefore the 474-attempt ORATS campaign remains unauthorized and the honest profit-confidence state remains **`UNPROVEN`**.",
        "",
        "## Exact evidence now preserved",
        "",
        "| Item | Result |",
        "|---|---:|",
        "| Raw public artifacts | %s |" % audit["raw_artifact_count"],
        "| ORATS attempts | 0 |",
        "| Schwab attempts | 0 |",
        "| Paid-data attempts | 0 |",
        "| XNYS sessions | 450 |",
        "| OCC adjustment-index memos | %s |" % audit["occ_memo_count"],
    ]
    for selection in audit["selection_dates"]:
        lines.append(
            "| Cboe security underlyings on %s | %s |"
            % (selection, audit["cboe_security_underlying_counts"][selection])
        )
        lines.append(
            "| Variable liquid research frame on %s | %s |"
            % (selection, audit["cboe_liquid_candidate_counts"][selection])
        )
    lines.extend(
        [
            "",
            "## Gate status",
            "",
            "| Gate | Status | Evidence |",
            "|---|---|---|",
        ]
    )
    for item in audit["components"]:
        lines.append(
            "| %s | %s `%s` | %s |"
            % (
                item["component"],
                component_icon[item["status"]],
                item["status"],
                item["evidence"],
            )
        )
    lines.extend(["", "## Remaining blockers", ""])
    for index, blocker in enumerate(audit["blockers"], start=1):
        lines.extend(
            [
                "%d. **%s** — %s" % (index, blocker["id"], blocker["impact"]),
                "   Resolution: %s" % blocker["required_resolution"],
            ]
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "**ORATS requests now: `0`.** The next work is still offline/public-source policy work. No cohort, historical campaign freeze, POP model, EV, or trade ticket is valid from these artifacts alone.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_public_history_sources(
    source_root: Path,
    *,
    minimum_security_underlyings: int = 1000,
    minimum_liquid_candidates: int = 100,
    require_complete_month: bool = True,
) -> PublicSourceAnalysis:
    """Validate preserved public bytes and derive a deterministic assessment."""

    root = _owned_root(source_root, "public source root")
    receipt = _source_receipt(root)
    discovery = _parse_cboe(
        root,
        minimum_security_underlyings=minimum_security_underlyings,
        minimum_liquid_candidates=minimum_liquid_candidates,
        require_complete_month=require_complete_month,
    )
    classification_queue = _build_classification_queue(discovery)
    calendar = _build_calendar(root)
    adjustment_index = _parse_occ(root)
    dlp = _parse_dlp_tombstones(root)
    references = _parse_current_references(root)
    audit = _build_audit(
        receipt,
        discovery,
        classification_queue,
        calendar,
        adjustment_index,
        dlp,
        references,
    )
    return PublicSourceAnalysis(
        receipt=receipt,
        discovery=discovery,
        classification_queue=classification_queue,
        calendar=calendar,
        adjustment_index=adjustment_index,
        audit=audit,
        markdown=_markdown(audit),
    )


def save_public_history_source_audit(
    *,
    source_root: Path,
    output_root: Path,
    run_id: str,
) -> SavedPublicSourceAudit:
    """Save one immutable, checksummed public-source audit run."""

    output = Path(output_root).expanduser().resolve()
    try:
        output.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise PublicHistorySourceError("public source audit output must remain in Cultra/out") from exc
    analysis = analyze_public_history_sources(source_root)
    writer = ArtifactWriter(output, run_id)
    writer.write_json("raw_evidence_receipt.json", analysis.receipt)
    writer.write_json("cboe_universe_discovery.json", analysis.discovery)
    writer.write_json("point_in_time_classification_queue.json", analysis.classification_queue)
    writer.write_json("market_session_candidate.json", analysis.calendar)
    writer.write_json("occ_contract_adjustment_index.json", analysis.adjustment_index)
    writer.write_json("public_source_audit.json", analysis.audit)
    writer.write_text("PUBLIC_SOURCE_AUDIT.md", analysis.markdown, "text/markdown")
    writer.finalize(
        as_of=date.fromisoformat(str(analysis.audit["as_of"])),
        overall_status=str(analysis.audit["status"]),
        metadata={
            "profit_confidence": "UNPROVEN",
            "audit_network_attempted": False,
            "orats_attempts": 0,
            "schwab_attempts": 0,
            "paid_data_attempts": 0,
            "source_receipt_hash": analysis.receipt["receipt_hash"],
        },
    )
    return SavedPublicSourceAudit(
        run_dir=writer.run_dir,
        manifest_path=writer.run_dir / "manifest.json",
        audit_path=writer.run_dir / "public_source_audit.json",
        board_path=writer.run_dir / "PUBLIC_SOURCE_AUDIT.md",
        status=str(analysis.audit["status"]),
    )


def verify_public_history_source_audit(run_dir: Path) -> Tuple[str, ...]:
    """Reconcile saved artifacts and reproduce them from the preserved raw bytes."""

    root = Path(run_dir).expanduser().resolve()
    errors = list(verify_manifest(root))
    required = {
        "raw_evidence_receipt.json",
        "cboe_universe_discovery.json",
        "point_in_time_classification_queue.json",
        "market_session_candidate.json",
        "occ_contract_adjustment_index.json",
        "public_source_audit.json",
        "PUBLIC_SOURCE_AUDIT.md",
    }
    for name in sorted(required):
        if not (root / name).is_file():
            errors.append("required public-source artifact is missing: %s" % name)
    if errors:
        return tuple(errors)
    try:
        receipt = json.loads((root / "raw_evidence_receipt.json").read_text(encoding="utf-8"))
        source_root = PROJECT_ROOT / str(receipt["source_root"])
        analysis = analyze_public_history_sources(source_root)
        expected_json = {
            "raw_evidence_receipt.json": analysis.receipt,
            "cboe_universe_discovery.json": analysis.discovery,
            "point_in_time_classification_queue.json": analysis.classification_queue,
            "market_session_candidate.json": analysis.calendar,
            "occ_contract_adjustment_index.json": analysis.adjustment_index,
            "public_source_audit.json": analysis.audit,
        }
        for name, expected in expected_json.items():
            actual = (root / name).read_bytes()
            if actual != canonical_json_bytes(expected):
                errors.append("public-source artifact is not reproducible: %s" % name)
        if (root / "PUBLIC_SOURCE_AUDIT.md").read_text(encoding="utf-8") != analysis.markdown:
            errors.append("public-source board is not reproducible")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append("public-source audit cannot be reproduced: %s" % exc)
    return tuple(errors)


__all__ = [
    "PublicHistorySourceError",
    "PublicSourceAnalysis",
    "SavedPublicSourceAudit",
    "analyze_public_history_sources",
    "save_public_history_source_audit",
    "verify_public_history_source_audit",
]
