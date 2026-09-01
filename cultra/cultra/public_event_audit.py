"""Offline audit of public historical event evidence for Cultra cohorts.

The network acquisition step is intentionally outside this module.  This file
only consumes immutable Cultra-owned bytes and answers a narrower question:
which event cells are supported well enough to freeze, and which still block
historical outcome construction?

The audit is deliberately conservative.  A provider response saying dividend
history is unavailable is never reinterpreted as evidence that no dividend
occurred.  Likewise, an OCC memo-index hit is not treated as an exact contract
deliverable until the detail bytes are preserved locally.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .artifacts import (
    ArtifactError,
    ArtifactWriter,
    assert_secret_free_bytes,
    canonical_json_bytes,
    verify_manifest,
)
from .public_classification import verify_public_classification_audit
from .public_history_sources import PROJECT_ROOT, verify_public_history_source_audit


OUT_ROOT = (PROJECT_ROOT / "out").resolve()
EVENT_TYPES = (
    "CONTRACT_ADJUSTMENT",
    "DELISTING",
    "DIVIDEND",
    "EARNINGS",
    "SPLIT",
)
_DIVIDEND_FILE = re.compile(r"^(?P<ticker>[A-Z][A-Z0-9.]{0,11})\.json$")
_DATE_FILE = re.compile(r"^(?P<day>[0-9]{4}-[0-9]{2}-[0-9]{2})\.json$")
_SURPRISE_FILE = _DIVIDEND_FILE
_WHOLE_SESSION_POLICY = "CONSERVATIVE_WHOLE_SESSION_BLACKOUT"
_COLLECTION_SCOPE = "TARGETED_CANDIDATE_DISCOVERY_NOT_COMPLETE"
_FOREIGN_FINANCIAL_DESCRIPTION = re.compile(
    r"\b(?:EARNINGS|RESULTS|TRADING STATEMENT|QUARTERLY REPORT|INTERIM REPORT)\b",
    re.IGNORECASE,
)
_SUCCESSOR_RE = re.compile(
    r"(?<![A-Z0-9])(?P<old>[A-Z][A-Z0-9]{0,7})\s+becomes\s+"
    r"(?P<new>[A-Z][A-Z0-9]{0,7})(?![A-Z0-9])",
    re.IGNORECASE,
)
_OPTION_TAIL_RE = re.compile(
    r"(?:ADJUSTED\s+)?OPTION\s+SYMBOLS?\s*:\s*(?P<tail>.*)$",
    re.IGNORECASE,
)


class PublicEventAuditError(ValueError):
    """Public event evidence is malformed, incomplete, or non-reproducible."""


@dataclass(frozen=True)
class PublicEventAnalysis:
    raw_receipt: Mapping[str, Any]
    event_candidates: Mapping[str, Any]
    coverage_matrix: Mapping[str, Any]
    audit: Mapping[str, Any]
    markdown: str


@dataclass(frozen=True)
class SavedPublicEventAudit:
    run_dir: Path
    board_path: Path
    audit_path: Path
    status: str


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


def _owned_directory(path: Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise PublicEventAuditError("%s cannot be a symlink" % label)
    supplied = candidate.resolve()
    try:
        supplied.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PublicEventAuditError("%s must be Cultra-owned" % label) from exc
    if not supplied.is_dir() or supplied.is_symlink():
        raise PublicEventAuditError("%s is unavailable" % label)
    return supplied


def _owned_json(path: Path, label: str) -> Mapping[str, Any]:
    supplied = Path(path).resolve()
    try:
        supplied.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise PublicEventAuditError("%s must be Cultra-owned" % label) from exc
    if supplied.is_symlink() or not supplied.is_file():
        raise PublicEventAuditError("%s is unavailable" % label)
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PublicEventAuditError("%s is unreadable" % label) from exc
    if not isinstance(value, Mapping):
        raise PublicEventAuditError("%s must be a JSON object" % label)
    return value


def _day(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise PublicEventAuditError("%s must use YYYY-MM-DD" % label) from exc


def _us_day(value: Any, label: str) -> date:
    try:
        month, day_value, year = (int(item) for item in str(value).split("/"))
        return date(year, month, day_value)
    except (TypeError, ValueError) as exc:
        raise PublicEventAuditError("%s must use M/D/YYYY" % label) from exc


def _money(value: Any, label: str) -> float:
    try:
        amount = float(str(value).strip().replace("$", "").replace(",", ""))
    except ValueError as exc:
        raise PublicEventAuditError("%s is not numeric" % label) from exc
    if amount <= 0.0:
        raise PublicEventAuditError("%s must be positive" % label)
    return amount


def _artifact(path: Path, *, role: str, source_uri: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PublicEventAuditError("public event artifact is unavailable")
    payload = path.read_bytes()
    try:
        assert_secret_free_bytes(payload, path=path.name)
    except ArtifactError as exc:
        raise PublicEventAuditError(
            "public event artifact contains credential-shaped material"
        ) from exc
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "role": role,
        "source_uri": source_uri,
        "media_type": "application/json",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _parallel_columns(recent: Mapping[str, Any]) -> int:
    required = (
        "form",
        "filingDate",
        "accessionNumber",
        "primaryDocument",
    )
    lengths = []
    for field in required:
        value = recent.get(field)
        if not isinstance(value, list):
            raise PublicEventAuditError("SEC recent filing column is missing: %s" % field)
        lengths.append(len(value))
    if len(set(lengths)) != 1:
        raise PublicEventAuditError("SEC recent filing columns are misaligned")
    for optional in ("items", "primaryDocDescription", "acceptanceDateTime"):
        value = recent.get(optional)
        if value is not None and (not isinstance(value, list) or len(value) != lengths[0]):
            raise PublicEventAuditError(
                "SEC recent filing column is misaligned: %s" % optional
            )
    return lengths[0]


def _column(recent: Mapping[str, Any], field: str, index: int) -> str:
    raw = recent.get(field)
    if not isinstance(raw, list):
        return ""
    return str(raw[index]).strip()


def _selected_rows(
    classifications: Mapping[str, Any], cohorts: Mapping[str, Any]
) -> Tuple[Mapping[str, Any], ...]:
    evaluated = classifications.get("evaluated")
    blocks = cohorts.get("blocks")
    if not isinstance(evaluated, list) or not isinstance(blocks, list):
        raise PublicEventAuditError("classification/cohort artifacts are malformed")
    by_key: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for item in evaluated:
        if not isinstance(item, Mapping):
            raise PublicEventAuditError("classification row is malformed")
        key = (str(item.get("selection_date")), str(item.get("ticker")))
        if key in by_key:
            raise PublicEventAuditError("classification row is duplicated")
        by_key[key] = item
    result = []
    seen: Set[str] = set()
    for block in blocks:
        if not isinstance(block, Mapping):
            raise PublicEventAuditError("cohort block is malformed")
        selection = str(block.get("selection_date"))
        start = str(block.get("block_start"))
        end = str(block.get("block_end"))
        for ticker in block.get("tickers", ()):
            normalized = str(ticker).strip().upper()
            if normalized in seen:
                raise PublicEventAuditError("cohort ticker is not disjoint")
            seen.add(normalized)
            try:
                classification = by_key[(selection, normalized)]
            except KeyError as exc:
                raise PublicEventAuditError(
                    "selected ticker lacks a classification: %s" % normalized
                ) from exc
            result.append(
                {
                    "ticker": normalized,
                    "selection_date": selection,
                    "block_start": start,
                    "block_end": end,
                    "asset_type": str(classification.get("asset_type")),
                    "cik": classification.get("cik"),
                    "classification": classification,
                }
            )
    if len(result) != 40:
        raise PublicEventAuditError("public event audit requires the frozen 40-name sample")
    return tuple(result)


def _title_mentions_option_symbol(title: str, ticker: str) -> bool:
    """Match ticker tokens only after an OCC option-symbol label.

    This avoids the original one-letter bug where cohort symbols ``S`` or
    ``T`` matched arbitrary company-name prose.
    """

    match = _OPTION_TAIL_RE.search(str(title))
    if match is None:
        return False
    pattern = re.compile(
        r"(?<![A-Z0-9])%s(?:[0-9]+)?(?![A-Z0-9])" % re.escape(ticker),
        re.IGNORECASE,
    )
    return pattern.search(match.group("tail")) is not None


def _successors(
    records: Sequence[Mapping[str, Any]], selected: Iterable[str]
) -> Mapping[str, Tuple[str, ...]]:
    selected_set = set(selected)
    values: Dict[str, Set[str]] = {ticker: set() for ticker in selected_set}
    for record in records:
        title = str(record.get("title", ""))
        for match in _SUCCESSOR_RE.finditer(title):
            old = match.group("old").upper()
            new = match.group("new").upper()
            if old in selected_set and new != old:
                values[old].add(new)
    return {
        ticker: tuple(sorted(items))
        for ticker, items in values.items()
        if items
    }


def _submission(root: Path, cik: Any) -> Mapping[str, Any]:
    try:
        numeric = int(cik)
    except (TypeError, ValueError) as exc:
        raise PublicEventAuditError("selected stock CIK is missing") from exc
    path = root / ("CIK%010d.json" % numeric)
    value = _owned_json(path, "SEC submission")
    if int(str(value.get("cik", "0"))) != numeric:
        raise PublicEventAuditError("SEC submission CIK does not match")
    return value


def _sec_financial_events(
    *, ticker: str, submission: Mapping[str, Any], start: date, end: date
) -> Tuple[Mapping[str, Any], ...]:
    filings = submission.get("filings")
    recent = filings.get("recent") if isinstance(filings, Mapping) else None
    if not isinstance(recent, Mapping):
        raise PublicEventAuditError("SEC recent filings are missing")
    count = _parallel_columns(recent)
    events = []
    for index in range(count):
        filing_date = _day(_column(recent, "filingDate", index), "SEC filing date")
        if not start <= filing_date <= end:
            continue
        form = _column(recent, "form", index).upper()
        items = {
            item.strip()
            for item in _column(recent, "items", index).split(",")
            if item.strip()
        }
        description = _column(recent, "primaryDocDescription", index).upper()
        reason: Optional[str] = None
        if form in {"8-K", "8-K/A"} and "2.02" in items:
            reason = "SEC_8K_ITEM_2_02"
        elif form in {"6-K", "6-K/A"} and _FOREIGN_FINANCIAL_DESCRIPTION.search(
            description
        ):
            reason = "SEC_6K_FINANCIAL_DESCRIPTION"
        if reason is None:
            continue
        accepted = _column(recent, "acceptanceDateTime", index)
        if accepted:
            try:
                parsed = datetime.fromisoformat(accepted.replace("Z", "+00:00"))
            except ValueError as exc:
                raise PublicEventAuditError("SEC acceptance timestamp is malformed") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise PublicEventAuditError("SEC acceptance timestamp lacks timezone")
        events.append(
            {
                "ticker": ticker,
                "event_type": "EARNINGS",
                "event_date": filing_date.isoformat(),
                "available_at": accepted or (filing_date.isoformat() + "T23:59:59Z"),
                "source": reason,
                "source_event_id": _column(recent, "accessionNumber", index),
                "primary_document": _column(recent, "primaryDocument", index),
                "timing_policy": _WHOLE_SESSION_POLICY,
            }
        )
    return tuple(
        sorted(
            events,
            key=lambda item: (item["event_date"], item["source_event_id"]),
        )
    )


def _calendar_earnings(
    files: Mapping[date, Mapping[str, Any]],
    *,
    ticker: str,
    start: date,
    end: date,
) -> Tuple[Mapping[str, Any], ...]:
    events = []
    for file_day, payload in sorted(files.items()):
        if not start <= file_day <= end:
            continue
        data = payload.get("data")
        rows = data.get("rows") if isinstance(data, Mapping) else None
        if not isinstance(rows, list):
            raise PublicEventAuditError("Nasdaq earnings calendar rows are missing")
        for row in rows:
            if not isinstance(row, Mapping):
                raise PublicEventAuditError("Nasdaq earnings calendar row is malformed")
            if str(row.get("symbol", "")).strip().upper() != ticker:
                continue
            events.append(
                {
                    "ticker": ticker,
                    "event_type": "EARNINGS",
                    "event_date": file_day.isoformat(),
                    "available_at": file_day.isoformat() + "T23:59:59Z",
                    "source": "NASDAQ_DAILY_EARNINGS_CALENDAR",
                    "source_event_id": "%s:%s" % (file_day.isoformat(), ticker),
                    "fiscal_quarter_ending": str(row.get("fiscalQuarterEnding", "")),
                    "provider_timing": str(row.get("time", "")),
                    "timing_policy": _WHOLE_SESSION_POLICY,
                }
            )
    return tuple(events)


def _surprise_earnings(
    payload: Mapping[str, Any], *, ticker: str, start: date, end: date
) -> Tuple[Mapping[str, Any], ...]:
    data = payload.get("data")
    if not isinstance(data, Mapping) or str(data.get("symbol", "")).upper() != ticker:
        raise PublicEventAuditError("Nasdaq earnings-surprise identity changed")
    table = data.get("earningsSurpriseTable")
    rows = table.get("rows") if isinstance(table, Mapping) else None
    if not isinstance(rows, list):
        raise PublicEventAuditError("Nasdaq earnings-surprise rows are missing")
    events = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PublicEventAuditError("Nasdaq earnings-surprise row is malformed")
        event_day = _us_day(row.get("dateReported"), "earnings date")
        if start <= event_day <= end:
            events.append(
                {
                    "ticker": ticker,
                    "event_type": "EARNINGS",
                    "event_date": event_day.isoformat(),
                    "available_at": event_day.isoformat() + "T23:59:59Z",
                    "source": "NASDAQ_EARNINGS_SURPRISE_HISTORY",
                    "source_event_id": "%s:%s" % (ticker, event_day.isoformat()),
                    "fiscal_quarter_ending": str(row.get("fiscalQtrEnd", "")),
                    "timing_policy": _WHOLE_SESSION_POLICY,
                }
            )
    return tuple(sorted(events, key=lambda item: item["event_date"]))


def _dividend_rows(
    payload: Mapping[str, Any], *, ticker: str
) -> Tuple[Mapping[str, Any], ...]:
    status = payload.get("status")
    if not isinstance(status, Mapping) or status.get("rCode") not in {200, 400}:
        raise PublicEventAuditError("Nasdaq dividend response status changed")
    data = payload.get("data")
    if data is None:
        return ()
    if not isinstance(data, Mapping):
        raise PublicEventAuditError("Nasdaq dividend response data is malformed")
    dividends = data.get("dividends")
    rows = dividends.get("rows") if isinstance(dividends, Mapping) else None
    if rows is None:
        return ()
    if not isinstance(rows, list):
        raise PublicEventAuditError("Nasdaq dividend history rows are malformed")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise PublicEventAuditError("Nasdaq dividend history row is malformed")
        event_day = _us_day(row.get("exOrEffDate"), "dividend ex date")
        event_type = str(row.get("type", "")).strip().upper()
        currency = str(row.get("currency", "")).strip().upper()
        if event_type != "CASH" or currency not in {"", "USD"}:
            raise PublicEventAuditError(
                "unsupported dividend history row for %s" % ticker
            )
        result.append(
            {
                "ticker": ticker,
                "event_type": "DIVIDEND",
                "event_date": event_day.isoformat(),
                "cash_amount": _money(row.get("amount"), "dividend amount"),
                "currency": currency or "USD",
                "declaration_date": str(row.get("declarationDate", "")),
                "record_date": str(row.get("recordDate", "")),
                "payment_date": str(row.get("paymentDate", "")),
                "source": "NASDAQ_SYMBOL_DIVIDEND_HISTORY",
                "source_event_id": "%s:%s" % (ticker, event_day.isoformat()),
            }
        )
    identities = [(item["ticker"], item["event_date"]) for item in result]
    if len(identities) != len(set(identities)):
        raise PublicEventAuditError("Nasdaq dividend history contains duplicates")
    return tuple(sorted(result, key=lambda item: item["event_date"]))


def _candidate_only_status(event_type: str, candidate_count: int) -> str:
    """Return a blocking status for targeted, non-exhaustive event evidence."""

    normalized = str(event_type).strip().upper()
    if normalized not in {"EARNINGS", "DIVIDEND"}:
        raise PublicEventAuditError("candidate-only status event type is unsupported")
    if candidate_count < 0:
        raise PublicEventAuditError("candidate-only status count cannot be negative")
    if candidate_count:
        return "BLOCKED_%s_CANDIDATES_PRESENT_COMPLETENESS_UNATTESTED" % normalized
    if normalized == "EARNINGS":
        return "BLOCKED_NO_FINANCIAL_EVENT_EVIDENCE"
    return "BLOCKED_NO_COMPLETE_DIVIDEND_HISTORY"


def _calendar_dividends(
    files: Mapping[date, Mapping[str, Any]],
) -> Mapping[Tuple[str, str], Mapping[str, Any]]:
    result: Dict[Tuple[str, str], Mapping[str, Any]] = {}
    for file_day, payload in sorted(files.items()):
        data = payload.get("data")
        calendar = data.get("calendar") if isinstance(data, Mapping) else None
        rows = calendar.get("rows") if isinstance(calendar, Mapping) else None
        if not isinstance(rows, list):
            raise PublicEventAuditError("Nasdaq dividend calendar rows are missing")
        for row in rows:
            if not isinstance(row, Mapping):
                raise PublicEventAuditError("Nasdaq dividend calendar row is malformed")
            ticker = str(row.get("symbol", "")).strip().upper()
            event_day = _us_day(row.get("dividend_Ex_Date"), "calendar ex date")
            if event_day != file_day:
                raise PublicEventAuditError("Nasdaq dividend calendar date changed")
            key = (ticker, event_day.isoformat())
            if key in result:
                raise PublicEventAuditError("Nasdaq dividend calendar row is duplicated")
            result[key] = {
                "ticker": ticker,
                "event_date": event_day.isoformat(),
                "cash_amount": _money(row.get("dividend_Rate"), "calendar dividend"),
                "company_name": str(row.get("companyName", "")),
                "source": "NASDAQ_DAILY_DIVIDEND_CALENDAR",
            }
    return result


def _load_event_root(
    root: Path,
    *,
    expected_dividend_symbols: Mapping[str, str],
    allowed_surprise_symbols: Set[str],
) -> Tuple[
    Mapping[str, Any],
    Mapping[str, Mapping[str, Any]],
    Mapping[date, Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
    Mapping[date, Mapping[str, Any]],
    Tuple[Mapping[str, Any], ...],
]:
    allowed_directories = {
        "nasdaq_dividend_history",
        "nasdaq_earnings_calendar",
        "nasdaq_earnings_surprise",
        "nasdaq_dividend_calendar",
    }
    allowed_entries = allowed_directories | {"collection_manifest.json"}
    direct = {item.name for item in root.iterdir()}
    if direct != allowed_entries or any(
        not (root / name).is_dir() or (root / name).is_symlink()
        for name in allowed_directories
    ):
        raise PublicEventAuditError("public event source inventory changed")

    collection_manifest = _owned_json(
        root / "collection_manifest.json", "public event collection manifest"
    )
    if (
        collection_manifest.get("schema") != "cultra.public-event-collection.v1"
        or collection_manifest.get("scope") != _COLLECTION_SCOPE
        or collection_manifest.get("complete_event_types") != []
        or collection_manifest.get("automatic_retries") != 0
        or collection_manifest.get("orats_attempts") != 0
        or collection_manifest.get("schwab_attempts") != 0
        or collection_manifest.get("paid_data_attempts") != 0
    ):
        raise PublicEventAuditError(
            "public event collection scope or request boundary changed"
        )
    queries = collection_manifest.get("saved_query_inventory")
    if not isinstance(queries, Mapping):
        raise PublicEventAuditError("public event saved-query inventory is missing")

    artifacts: List[Mapping[str, Any]] = [
        _artifact(
            root / "collection_manifest.json",
            role="CULTRA_PUBLIC_EVENT_COLLECTION_MANIFEST",
            source_uri="cultra://public-event-collection/2026-08-31",
        )
    ]
    dividend_history: Dict[str, Mapping[str, Any]] = {}
    dividend_dir = root / "nasdaq_dividend_history"
    for path in sorted(dividend_dir.iterdir(), key=lambda item: item.name):
        match = _DIVIDEND_FILE.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise PublicEventAuditError("unexpected dividend-history artifact")
        ticker = match.group("ticker")
        if ticker in dividend_history:
            raise PublicEventAuditError("dividend-history ticker is duplicated")
        dividend_history[ticker] = _owned_json(path, "Nasdaq dividend history")
        asset_type = expected_dividend_symbols.get(ticker)
        if asset_type not in {"STOCK", "ETF"}:
            raise PublicEventAuditError(
                "dividend-history symbol lacks a classified asset type"
            )
        asset = "etf" if asset_type == "ETF" else "stocks"
        artifacts.append(
            _artifact(
                path,
                role="NASDAQ_SYMBOL_DIVIDEND_HISTORY_%s" % ticker,
                source_uri=(
                    "https://api.nasdaq.com/api/quote/%s/dividends?assetclass=%s"
                    % (ticker, asset)
                ),
            )
        )
    expected_dividend_set = set(expected_dividend_symbols)
    if set(dividend_history) != expected_dividend_set:
        missing = sorted(expected_dividend_set - set(dividend_history))
        extra = sorted(set(dividend_history) - expected_dividend_set)
        raise PublicEventAuditError(
            "dividend-history symbol inventory mismatch: missing=%s extra=%s"
            % (",".join(missing), ",".join(extra))
        )

    def dated_files(directory: str, role: str, uri: str) -> Dict[date, Mapping[str, Any]]:
        result: Dict[date, Mapping[str, Any]] = {}
        for path in sorted((root / directory).iterdir(), key=lambda item: item.name):
            match = _DATE_FILE.fullmatch(path.name)
            if match is None or not path.is_file() or path.is_symlink():
                raise PublicEventAuditError("unexpected dated public-event artifact")
            file_day = _day(match.group("day"), "event filename")
            if file_day in result:
                raise PublicEventAuditError("dated public-event artifact is duplicated")
            result[file_day] = _owned_json(path, role)
            artifacts.append(
                _artifact(
                    path,
                    role="%s_%s" % (role, file_day.isoformat()),
                    source_uri=uri % file_day.isoformat(),
                )
            )
        if not result:
            raise PublicEventAuditError("dated public-event directory is empty")
        return result

    earnings_calendar = dated_files(
        "nasdaq_earnings_calendar",
        "NASDAQ_DAILY_EARNINGS_CALENDAR",
        "https://api.nasdaq.com/api/calendar/earnings?date=%s",
    )
    dividend_calendar = dated_files(
        "nasdaq_dividend_calendar",
        "NASDAQ_DAILY_DIVIDEND_CALENDAR",
        "https://api.nasdaq.com/api/calendar/dividends?date=%s",
    )

    surprises: Dict[str, Mapping[str, Any]] = {}
    surprise_dir = root / "nasdaq_earnings_surprise"
    for path in sorted(surprise_dir.iterdir(), key=lambda item: item.name):
        match = _SURPRISE_FILE.fullmatch(path.name)
        if match is None or not path.is_file() or path.is_symlink():
            raise PublicEventAuditError("unexpected earnings-surprise artifact")
        ticker = match.group("ticker")
        if ticker not in allowed_surprise_symbols or ticker in surprises:
            raise PublicEventAuditError("earnings-surprise symbol inventory changed")
        surprises[ticker] = _owned_json(path, "Nasdaq earnings surprise")
        artifacts.append(
            _artifact(
                path,
                role="NASDAQ_EARNINGS_SURPRISE_%s" % ticker,
                source_uri=(
                    "https://api.nasdaq.com/api/company/%s/earnings-surprise"
                    % ticker
                ),
            )
        )
    if not surprises:
        raise PublicEventAuditError("earnings-surprise evidence is empty")
    inventories = {
        "nasdaq_dividend_history_symbols": sorted(dividend_history),
        "nasdaq_earnings_calendar_dates": [
            item.isoformat() for item in sorted(earnings_calendar)
        ],
        "nasdaq_earnings_surprise_symbols": sorted(surprises),
        "nasdaq_dividend_calendar_dates": [
            item.isoformat() for item in sorted(dividend_calendar)
        ],
    }
    if queries != inventories:
        raise PublicEventAuditError(
            "public event saved-query inventory does not match preserved files"
        )
    if collection_manifest.get("saved_provider_artifact_count") != len(artifacts) - 1:
        raise PublicEventAuditError(
            "public event saved-provider-artifact count does not reconcile"
        )
    return (
        collection_manifest,
        dividend_history,
        earnings_calendar,
        surprises,
        dividend_calendar,
        tuple(sorted(artifacts, key=lambda item: item["path"])),
    )


def analyze_public_events(
    *,
    classification_run_dir: Path,
    event_source_root: Path,
) -> PublicEventAnalysis:
    classification_root = _owned_directory(
        classification_run_dir, "classification audit"
    )
    errors = verify_public_classification_audit(classification_root)
    if errors:
        raise PublicEventAuditError(
            "classification audit does not verify: %s" % "; ".join(errors)
        )
    classification_manifest = _owned_json(
        classification_root / "manifest.json", "classification manifest"
    )
    metadata = classification_manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise PublicEventAuditError("classification metadata is missing")
    public_source_root = _owned_directory(
        Path(str(metadata.get("source_audit_dir", ""))), "public source audit"
    )
    source_errors = verify_public_history_source_audit(public_source_root)
    if source_errors:
        raise PublicEventAuditError(
            "public source audit does not verify: %s" % "; ".join(source_errors)
        )
    sec_root = _owned_directory(
        Path(str(metadata.get("sec_submission_root", ""))), "SEC submission source"
    )
    classifications = _owned_json(
        classification_root / "point_in_time_classifications.json",
        "point-in-time classifications",
    )
    cohorts = _owned_json(
        classification_root / "rotating_cohorts.json", "rotating cohorts"
    )
    selected_rows = _selected_rows(classifications, cohorts)
    selected_symbols = {str(item["ticker"]) for item in selected_rows}
    adjustment_index = _owned_json(
        public_source_root / "occ_contract_adjustment_index.json",
        "OCC adjustment index",
    )
    records = adjustment_index.get("records")
    if (
        not isinstance(records, list)
        or adjustment_index.get("complete_non_overlapping_post_date_slices") is not True
        or adjustment_index.get("export_cap_reached") is not False
    ):
        raise PublicEventAuditError("OCC adjustment index is not complete")
    adjustment_records = tuple(item for item in records if isinstance(item, Mapping))
    if len(adjustment_records) != len(records):
        raise PublicEventAuditError("OCC adjustment record is malformed")
    successors = _successors(adjustment_records, selected_symbols)
    event_root = _owned_directory(event_source_root, "public event source")
    selected_asset_types = {
        str(item["ticker"]): str(item["asset_type"]) for item in selected_rows
    }
    dividend_symbols = dict(selected_asset_types)
    for predecessor, successor_symbols in successors.items():
        for successor in successor_symbols:
            dividend_symbols[successor] = selected_asset_types[predecessor]
    (
        collection_manifest,
        dividend_history,
        earnings_calendar,
        earnings_surprises,
        dividend_calendar_files,
        event_artifacts,
    ) = _load_event_root(
        event_root,
        expected_dividend_symbols=dividend_symbols,
        allowed_surprise_symbols=selected_symbols,
    )
    calendar_dividends = _calendar_dividends(dividend_calendar_files)

    relevant_memos: Dict[str, Tuple[Mapping[str, Any], ...]] = {}
    for row in selected_rows:
        ticker = str(row["ticker"])
        start = _day(row["block_start"], "block start")
        end = _day(row["block_end"], "block end")
        matches = []
        for record in adjustment_records:
            posted = _day(record.get("post_date"), "OCC post date")
            effective_raw = record.get("effective_date")
            effective = _day(effective_raw, "OCC effective date") if effective_raw else None
            in_window = start <= posted <= end or (
                effective is not None and start <= effective <= end
            )
            if in_window and _title_mentions_option_symbol(
                str(record.get("title", "")), ticker
            ):
                matches.append(record)
        relevant_memos[ticker] = tuple(
            sorted(matches, key=lambda item: str(item.get("memo_number")))
        )
    memo_numbers = {
        str(item.get("memo_number"))
        for matches in relevant_memos.values()
        for item in matches
    }

    earnings_events: List[Mapping[str, Any]] = []
    dividend_events: List[Mapping[str, Any]] = []
    adjustment_events: List[Mapping[str, Any]] = []
    coverage_rows = []
    blocking_cells = []
    positive_dividend_symbols = set()
    for row in selected_rows:
        ticker = str(row["ticker"])
        asset_type = str(row["asset_type"])
        start = _day(row["block_start"], "block start")
        end = _day(row["block_end"], "block end")
        ticker_earnings: Tuple[Mapping[str, Any], ...]
        if asset_type == "ETF":
            ticker_earnings = ()
            earnings_status = "NOT_APPLICABLE_ETF"
        else:
            submission = _submission(sec_root, row["cik"])
            sec_events = _sec_financial_events(
                ticker=ticker,
                submission=submission,
                start=start,
                end=end,
            )
            external_events = list(
                _calendar_earnings(
                    earnings_calendar, ticker=ticker, start=start, end=end
                )
            )
            if ticker in earnings_surprises:
                external_events.extend(
                    _surprise_earnings(
                        earnings_surprises[ticker],
                        ticker=ticker,
                        start=start,
                        end=end,
                    )
                )
            combined = {
                (item["event_date"], item["source_event_id"]): item
                for item in sec_events + tuple(external_events)
            }
            ticker_earnings = tuple(
                sorted(
                    combined.values(),
                    key=lambda item: (item["event_date"], item["source_event_id"]),
                )
            )
            earnings_status = _candidate_only_status(
                "EARNINGS", len(ticker_earnings)
            )
            blocking_cells.append(
                {"ticker": ticker, "event_type": "EARNINGS", "reason": earnings_status}
            )
        earnings_events.extend(ticker_earnings)

        chain_symbols = (ticker,) + successors.get(ticker, ())
        ticker_dividends = []
        history_statuses = []
        all_histories_complete = True
        for chain_symbol in chain_symbols:
            payload = dividend_history[chain_symbol]
            history = _dividend_rows(payload, ticker=chain_symbol)
            in_window = tuple(
                item
                for item in history
                if start <= _day(item["event_date"], "dividend event date") <= end
            )
            message = str(payload.get("message") or "")
            if message or not history:
                all_histories_complete = False
                history_statuses.append(
                    {"symbol": chain_symbol, "status": "UNAVAILABLE_OR_AMBIGUOUS", "message": message}
                )
            else:
                history_statuses.append(
                    {"symbol": chain_symbol, "status": "HISTORY_PRESENT", "message": ""}
                )
            for event in in_window:
                calendar = calendar_dividends.get(
                    (chain_symbol, str(event["event_date"]))
                )
                enriched = dict(event)
                if calendar is not None:
                    if abs(float(calendar["cash_amount"]) - float(event["cash_amount"])) > 1e-9:
                        raise PublicEventAuditError(
                            "Nasdaq dividend history/calendar amount mismatch"
                        )
                    enriched["calendar_corroborated"] = True
                    enriched["calendar_company_name"] = calendar["company_name"]
                else:
                    enriched["calendar_corroborated"] = False
                if chain_symbol != ticker:
                    enriched["cohort_ticker"] = ticker
                    enriched["successor_symbol"] = chain_symbol
                ticker_dividends.append(enriched)
        if ticker_dividends:
            positive_dividend_symbols.add(ticker)
        dividend_events.extend(ticker_dividends)
        strong_dividend_candidates = ticker_dividends and all_histories_complete and all(
            item.get("calendar_corroborated") is True for item in ticker_dividends
        )
        dividend_status = _candidate_only_status(
            "DIVIDEND", len(ticker_dividends) if strong_dividend_candidates else 0
        )
        blocking_cells.append(
            {"ticker": ticker, "event_type": "DIVIDEND", "reason": dividend_status}
        )

        memos = relevant_memos[ticker]
        for memo in memos:
            adjustment_events.append(
                {
                    "ticker": ticker,
                    "event_type": "CONTRACT_ADJUSTMENT",
                    "effective_date": memo.get("effective_date"),
                    "post_date": memo.get("post_date"),
                    "source_event_id": str(memo.get("memo_number")),
                    "title": str(memo.get("title")),
                    "detail_retrieved": bool(memo.get("detail_retrieved")),
                    "source_uri": "https://infomemo.theocc.com/infomemos?number=%s"
                    % memo.get("memo_number"),
                }
            )
        missing_detail = any(not bool(item.get("detail_retrieved")) for item in memos)
        adjustment_status = (
            "BLOCKED_EXACT_MEMO_DETAIL_BYTES_MISSING"
            if missing_detail
            else (
                "COMPLETE_OCC_INDEX_AND_EXACT_DETAILS"
                if memos
                else "COMPLETE_OCC_INDEX_NO_AFFECTING_MEMO"
            )
        )
        if missing_detail:
            blocking_cells.append(
                {
                    "ticker": ticker,
                    "event_type": "CONTRACT_ADJUSTMENT",
                    "reason": adjustment_status,
                }
            )
        has_split = any("SPLIT" in str(item.get("title", "")).upper() for item in memos)
        split_status = (
            "BLOCKED_EXACT_SPLIT_DETAIL_BYTES_MISSING"
            if has_split and missing_detail
            else "COMPLETE_OCC_OPTIONS_IMPACT_INDEX"
        )
        if split_status.startswith("BLOCKED"):
            blocking_cells.append(
                {"ticker": ticker, "event_type": "SPLIT", "reason": split_status}
            )
        has_transition = bool(successors.get(ticker)) or any(
            any(
                word in str(item.get("title", "")).upper()
                for word in ("MERGER", "LIQUIDATION", "CASH SETTLEMENT")
            )
            for item in memos
        )
        delisting_status = (
            "BLOCKED_EXACT_TRANSITION_DETAIL_BYTES_MISSING"
            if has_transition and missing_detail
            else "COMPLETE_OCC_OPTIONS_IMPACT_INDEX"
        )
        if delisting_status.startswith("BLOCKED"):
            blocking_cells.append(
                {"ticker": ticker, "event_type": "DELISTING", "reason": delisting_status}
            )

        coverage_rows.append(
            {
                "ticker": ticker,
                "asset_type": asset_type,
                "selection_date": row["selection_date"],
                "coverage_start": row["block_start"],
                "coverage_end": row["block_end"],
                "chain_symbols": list(chain_symbols),
                "event_statuses": {
                    "CONTRACT_ADJUSTMENT": adjustment_status,
                    "DELISTING": delisting_status,
                    "DIVIDEND": dividend_status,
                    "EARNINGS": earnings_status,
                    "SPLIT": split_status,
                },
                "earnings_event_count": len(ticker_earnings),
                "dividend_event_count": len(ticker_dividends),
                "occ_memo_numbers": [str(item.get("memo_number")) for item in memos],
                "dividend_history_sources": history_statuses,
            }
        )

    earnings_events = sorted(
        earnings_events,
        key=lambda item: (item["event_date"], item["ticker"], item["source_event_id"]),
    )
    dividend_events = sorted(
        dividend_events,
        key=lambda item: (item["event_date"], item["ticker"]),
    )
    adjustment_events = sorted(
        adjustment_events,
        key=lambda item: (item["post_date"], item["source_event_id"]),
    )
    blocking_cells = sorted(
        blocking_cells, key=lambda item: (item["ticker"], item["event_type"])
    )
    status_counts = {event_type: {"complete": 0, "blocked": 0} for event_type in EVENT_TYPES}
    for row in coverage_rows:
        for event_type, status in row["event_statuses"].items():
            bucket = "blocked" if str(status).startswith("BLOCKED") else "complete"
            status_counts[event_type][bucket] += 1

    raw_payload = {
        "schema": "cultra.public-event-source-receipt.v1",
        "event_source_root": event_root.relative_to(PROJECT_ROOT).as_posix(),
        "preserved_public_event_artifact_count": len(event_artifacts),
        "public_event_artifacts": list(event_artifacts),
        "collection_scope": collection_manifest["scope"],
        "collection_manifest_sha256": next(
            item["sha256"]
            for item in event_artifacts
            if item["role"] == "CULTRA_PUBLIC_EVENT_COLLECTION_MANIFEST"
        ),
        "classification_run_dir": classification_root.relative_to(PROJECT_ROOT).as_posix(),
        "public_source_run_dir": public_source_root.relative_to(PROJECT_ROOT).as_posix(),
        "sec_submission_root": sec_root.relative_to(PROJECT_ROOT).as_posix(),
        "automatic_retries": 0,
        "automatic_redirects": 0,
        "orats_attempts": 0,
        "schwab_attempts": 0,
        "paid_data_attempts": 0,
    }
    raw_receipt = dict(
        raw_payload,
        receipt_hash=hashlib.sha256(_canonical(raw_payload)).hexdigest(),
    )
    event_candidates = {
        "schema": "cultra.public-event-candidates.v1",
        "normalization_status": "CANDIDATES_ONLY_NOT_A_HISTORICAL_EVENT_MANIFEST",
        "collection_scope": collection_manifest["scope"],
        "timing_policy": _WHOLE_SESSION_POLICY,
        "earnings": earnings_events,
        "dividends": dividend_events,
        "contract_adjustments": adjustment_events,
        "successor_chains": [
            {"cohort_ticker": ticker, "successors": list(items)}
            for ticker, items in sorted(successors.items())
        ],
    }
    coverage_matrix = {
        "schema": "cultra.public-event-coverage-matrix.v1",
        "required_event_types": list(EVENT_TYPES),
        "selected_symbol_count": len(coverage_rows),
        "rows": coverage_rows,
        "blocking_cells": blocking_cells,
        "status_counts": status_counts,
    }
    audit_payload = {
        "schema": "cultra.public-event-audit.v1",
        "status": "EVENT_CANDIDATES_FOUND_NOT_FREEZEABLE",
        "profit_confidence": "UNPROVEN",
        "historical_campaign_authorized": False,
        "manual_ticket_count": 0,
        "orats_attempts": 0,
        "schwab_attempts": 0,
        "paid_data_attempts": 0,
        "recommended_orats_attempts_now": 0,
        "selected_symbol_count": len(coverage_rows),
        "earnings_event_count": len(earnings_events),
        "dividend_event_count": len(dividend_events),
        "positive_dividend_symbol_count": len(positive_dividend_symbols),
        "contract_adjustment_memo_count": len(adjustment_events),
        "exact_occ_detail_memos_missing": sorted(memo_numbers),
        "occ_detail_acquisition_status": (
            "OFFICIAL_INDEX_CONFIRMED_PROVIDER_DETAIL_BYTES_NOT_PRESERVED"
        ),
        "successor_mapping": {key: list(value) for key, value in successors.items()},
        "historical_identity_recoveries": sorted(
            str(item["ticker"])
            for item in selected_rows
            if str(item["classification"].get("classification_method", "")).startswith(
                "HISTORICAL_"
            )
        ),
        "event_status_counts": status_counts,
        "blocking_cell_count": len(blocking_cells),
        "blocking_symbol_count": len({item["ticker"] for item in blocking_cells}),
        "remaining_blockers": [
            "COMPLETE_POINT_IN_TIME_EARNINGS_HISTORY",
            "COMPLETE_DIVIDEND_HISTORY_OR_AUDITABLE_NO_DIVIDEND_EVIDENCE",
            "EXACT_PROVIDER_BYTES_FOR_AFFECTING_CONTRACT_ADJUSTMENT_MEMOS",
        ],
        "next_action": (
            "Preserve complete issuer/exchange earnings and dividend histories for "
            "blocked symbols plus every exact affecting OCC memo detail file; rerun "
            "this offline audit before freezing any historical event manifest."
        ),
    }
    audit = dict(
        audit_payload,
        audit_hash=hashlib.sha256(_canonical(audit_payload)).hexdigest(),
    )
    lines = [
        "# Cultra Public Event Evidence Audit",
        "",
        "**Outcome: 🔴 `EVENT_CANDIDATES_FOUND_NOT_FREEZEABLE`**",
        "",
        "Cultra now has earnings candidates, dividend records where the public provider actually returned history, and the complete cohort-scoped contract-adjustment memo index. It still does **not** have a freezeable event manifest: targeted candidate discovery is not complete event history, an unavailable dividend response is not proof of no dividend, and affecting adjustment memos lack preserved detail bytes.",
        "",
        "| Evidence | Result |",
        "|---|---:|",
        "| Sampled symbols | %d |" % audit["selected_symbol_count"],
        "| Earnings candidates | %d |" % audit["earnings_event_count"],
        "| Dividend candidates | %d |" % audit["dividend_event_count"],
        "| Affecting adjustment memos | %d |" % audit["contract_adjustment_memo_count"],
        "| Blocking event cells | %d |" % audit["blocking_cell_count"],
        "| ORATS / Schwab / paid attempts | 0 / 0 / 0 |",
        "",
        "## Coverage by event type",
        "",
        "| Event type | Complete or N/A | Blocked |",
        "|---|---:|---:|",
    ]
    for event_type in EVENT_TYPES:
        counts = status_counts[event_type]
        lines.append(
            "| %s | %d | %d |"
            % (event_type, counts["complete"], counts["blocked"])
        )
    lines.extend(
        [
            "",
            "## Material findings",
            "",
            "- Derived successor chains: `%s`. Current reused ticker associations are not substituted for the preserved historical identity."
            % (
                ", ".join(
                    "%s → %s" % (ticker, "/".join(items))
                    for ticker, items in sorted(successors.items())
                )
                or "none"
            ),
            "- Earnings with unknown intraday timing use a whole-session blackout; no event is moved earlier using hindsight.",
            "- Finding one or more earnings dates is not counted as complete earnings history.",
            "- Missing dividend histories remain blocked rather than being rewritten as zero dividends.",
            "- OCC's official search confirms the memo identifiers, but direct PDF transfer produced no preservable provider bytes; the detail gate remains closed.",
            "- Exact contract-adjustment detail remains missing for memos `%s`." % "`, `".join(sorted(memo_numbers)),
            "",
            "## Result",
            "",
            "ORATS historical option-chain acquisition and trade tickets remain disabled. Additional ORATS requests would not repair these independent event-source gaps, so the recommended ORATS count is still `0`.",
            "",
        ]
    )
    return PublicEventAnalysis(
        raw_receipt=raw_receipt,
        event_candidates=event_candidates,
        coverage_matrix=coverage_matrix,
        audit=audit,
        markdown="\n".join(lines),
    )


def save_public_event_audit(
    *,
    classification_run_dir: Path,
    event_source_root: Path,
    output_root: Path,
    run_id: str,
) -> SavedPublicEventAudit:
    output = Path(output_root).expanduser().resolve()
    try:
        output.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise PublicEventAuditError("event audit output must remain in Cultra/out") from exc
    analysis = analyze_public_events(
        classification_run_dir=classification_run_dir,
        event_source_root=event_source_root,
    )
    writer = ArtifactWriter(output, run_id)
    writer.write_json("raw_event_receipt.json", analysis.raw_receipt)
    writer.write_json("event_candidates.json", analysis.event_candidates)
    writer.write_json("event_coverage_matrix.json", analysis.coverage_matrix)
    writer.write_json("public_event_audit.json", analysis.audit)
    writer.write_text("PUBLIC_EVENT_AUDIT.md", analysis.markdown, "text/markdown")
    writer.finalize(
        as_of=date(2026, 8, 31),
        overall_status=str(analysis.audit["status"]),
        metadata={
            "profit_confidence": "UNPROVEN",
            "orats_attempts": 0,
            "schwab_attempts": 0,
            "paid_data_attempts": 0,
            "classification_run_dir": str(Path(classification_run_dir).resolve()),
            "event_source_root": str(Path(event_source_root).resolve()),
        },
    )
    return SavedPublicEventAudit(
        run_dir=writer.run_dir,
        board_path=writer.run_dir / "PUBLIC_EVENT_AUDIT.md",
        audit_path=writer.run_dir / "public_event_audit.json",
        status=str(analysis.audit["status"]),
    )


def verify_public_event_audit(run_dir: Path) -> Tuple[str, ...]:
    root = Path(run_dir).expanduser().resolve()
    errors = list(verify_manifest(root))
    manifest_path = root / "manifest.json"
    if errors or not manifest_path.is_file():
        return tuple(errors)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        analysis = analyze_public_events(
            classification_run_dir=Path(metadata["classification_run_dir"]),
            event_source_root=Path(metadata["event_source_root"]),
        )
        expected_json = {
            "raw_event_receipt.json": analysis.raw_receipt,
            "event_candidates.json": analysis.event_candidates,
            "event_coverage_matrix.json": analysis.coverage_matrix,
            "public_event_audit.json": analysis.audit,
        }
        for name, expected in expected_json.items():
            if (root / name).read_bytes() != canonical_json_bytes(expected):
                errors.append("public event artifact is not reproducible: %s" % name)
        if (root / "PUBLIC_EVENT_AUDIT.md").read_text(encoding="utf-8") != analysis.markdown:
            errors.append("public event board is not reproducible")
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        errors.append("public event audit cannot be reproduced: %s" % exc)
    return tuple(errors)


__all__ = [
    "PublicEventAnalysis",
    "PublicEventAuditError",
    "SavedPublicEventAudit",
    "analyze_public_events",
    "save_public_event_audit",
    "verify_public_event_audit",
]
