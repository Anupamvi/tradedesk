"""Strict point-in-time historical event evidence for Cultra campaigns."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVENT_TYPES = (
    "CONTRACT_ADJUSTMENT",
    "DELISTING",
    "DIVIDEND",
    "EARNINGS",
    "SPLIT",
)


class HistoricalEventError(ValueError):
    """Historical event evidence is incomplete or not point-in-time safe."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise HistoricalEventError("%s must use YYYY-MM-DD" % label) from exc


def _timestamp(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalEventError("%s must be an ISO timestamp" % label) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HistoricalEventError("%s must be timezone-aware" % label)
    return parsed


def _digest(value: Any, label: str) -> str:
    normalized = str(value).lower().removeprefix("sha256:")
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise HistoricalEventError("%s is not a SHA-256 digest" % label)
    return normalized


@dataclass(frozen=True)
class HistoricalEventRecord:
    ticker: str
    event_type: str
    effective_date: date
    observed_at: datetime
    available_at: datetime
    source_event_id: str
    status: str
    cash_amount: Optional[float]
    split_ratio: Optional[float]
    adjustment_reference: Optional[str]

    def __post_init__(self) -> None:
        ticker = str(self.ticker).strip().upper()
        if not ticker or len(ticker) > 12:
            raise HistoricalEventError("historical event ticker is invalid")
        object.__setattr__(self, "ticker", ticker)
        event_type = str(self.event_type).strip().upper()
        if event_type not in EVENT_TYPES:
            raise HistoricalEventError("historical event type is unsupported")
        object.__setattr__(self, "event_type", event_type)
        if self.available_at < self.observed_at:
            raise HistoricalEventError("historical event became available before observation")
        if not str(self.source_event_id).strip():
            raise HistoricalEventError("historical source event id is required")
        if self.status not in {"CONFIRMED", "CANCELLED", "REVISED"}:
            raise HistoricalEventError("historical event status is unsupported")
        if self.cash_amount is not None:
            cash = float(self.cash_amount)
            if not math.isfinite(cash) or cash <= 0.0:
                raise HistoricalEventError("historical event cash amount is invalid")
            object.__setattr__(self, "cash_amount", cash)
        if self.split_ratio is not None:
            ratio = float(self.split_ratio)
            if not math.isfinite(ratio) or ratio <= 0.0 or math.isclose(ratio, 1.0):
                raise HistoricalEventError("historical split ratio is invalid")
            object.__setattr__(self, "split_ratio", ratio)
        reference = (
            None
            if self.adjustment_reference is None
            else str(self.adjustment_reference).strip()
        )
        object.__setattr__(self, "adjustment_reference", reference)
        if event_type == "DIVIDEND" and self.cash_amount is None:
            raise HistoricalEventError("dividend event requires a cash amount")
        if event_type != "DIVIDEND" and self.cash_amount is not None:
            raise HistoricalEventError("cash amount is only valid for dividends")
        if event_type == "SPLIT" and self.split_ratio is None:
            raise HistoricalEventError("split event requires a ratio")
        if event_type != "SPLIT" and self.split_ratio is not None:
            raise HistoricalEventError("split ratio is only valid for splits")
        if event_type == "CONTRACT_ADJUSTMENT" and not reference:
            raise HistoricalEventError(
                "contract adjustment requires an immutable reference"
            )
        if event_type != "CONTRACT_ADJUSTMENT" and reference is not None:
            raise HistoricalEventError(
                "adjustment reference is only valid for contract adjustments"
            )


@dataclass(frozen=True)
class HistoricalEventManifest:
    provider: str
    source_uri: str
    source_sha256: str
    coverage_start: date
    coverage_end: date
    covered_tickers: Tuple[str, ...]
    complete_event_types: Tuple[str, ...]
    records: Tuple[HistoricalEventRecord, ...]
    point_in_time_revisions: bool
    manifest_hash: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.source_uri.strip():
            raise HistoricalEventError("historical event provider and source URI are required")
        _digest(self.source_sha256, "historical event source hash")
        if self.coverage_end < self.coverage_start:
            raise HistoricalEventError("historical event coverage is reversed")
        tickers = tuple(sorted(set(self.covered_tickers)))
        if not tickers or tickers != self.covered_tickers:
            raise HistoricalEventError("covered tickers must be non-empty, sorted and unique")
        if tuple(EVENT_TYPES) != self.complete_event_types:
            raise HistoricalEventError("historical event types are not completely covered")
        if self.point_in_time_revisions is not True:
            raise HistoricalEventError("historical event source lacks point-in-time revisions")
        identities = tuple(
            (
                item.ticker,
                item.event_type,
                item.effective_date,
                item.available_at,
                item.source_event_id,
            )
            for item in self.records
        )
        if len(identities) != len(set(identities)):
            raise HistoricalEventError("historical event records are duplicated")
        if any(item.ticker not in tickers for item in self.records):
            raise HistoricalEventError("historical event record leaves covered tickers")
        if any(
            not self.coverage_start <= item.effective_date <= self.coverage_end
            for item in self.records
        ):
            raise HistoricalEventError("historical event leaves the coverage window")
        _digest(self.manifest_hash, "historical event manifest hash")

    def known_events(
        self,
        *,
        ticker: str,
        signal_timestamp: datetime,
        through_date: date,
    ) -> Tuple[HistoricalEventRecord, ...]:
        if signal_timestamp.tzinfo is None or signal_timestamp.utcoffset() is None:
            raise HistoricalEventError("signal timestamp must be timezone-aware")
        normalized = str(ticker).strip().upper()
        if normalized not in self.covered_tickers:
            raise HistoricalEventError("ticker is outside historical event coverage")
        if through_date < signal_timestamp.date():
            raise HistoricalEventError("event query window is reversed")
        available = tuple(
            item
            for item in self.records
            if item.ticker == normalized and item.available_at <= signal_timestamp
        )
        latest = {}
        for item in sorted(available, key=lambda event: event.available_at):
            latest[item.source_event_id] = item
        return tuple(
            sorted(
                (
                    item
                    for item in latest.values()
                    if item.status != "CANCELLED"
                    and signal_timestamp.date()
                    <= item.effective_date
                    <= through_date
                ),
                key=lambda item: (
                    item.effective_date,
                    item.event_type,
                    item.available_at,
                    item.source_event_id,
                ),
            )
        )

    def events_in_window(
        self, *, ticker: str, start_date: date, end_date: date
    ) -> Tuple[HistoricalEventRecord, ...]:
        """Return the final recorded event state for realized-path handling."""

        normalized = str(ticker).strip().upper()
        if normalized not in self.covered_tickers:
            raise HistoricalEventError("ticker is outside historical event coverage")
        if end_date < start_date:
            raise HistoricalEventError("event query window is reversed")
        latest = {}
        for item in sorted(
            (record for record in self.records if record.ticker == normalized),
            key=lambda event: event.available_at,
        ):
            latest[item.source_event_id] = item
        return tuple(
            sorted(
                (
                    item
                    for item in latest.values()
                    if item.status != "CANCELLED"
                    and start_date <= item.effective_date <= end_date
                ),
                key=lambda item: (
                    item.effective_date,
                    item.event_type,
                    item.source_event_id,
                ),
            )
        )


def load_historical_event_manifest(path: Path) -> HistoricalEventManifest:
    supplied = Path(path).expanduser().resolve()
    try:
        supplied.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HistoricalEventError("historical event manifest must be Cultra-owned") from exc
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoricalEventError("historical event manifest is unreadable") from exc
    if not isinstance(value, Mapping) or value.get("schema") != "cultra.historical-events.v1":
        raise HistoricalEventError("historical event manifest schema is unsupported")
    allowed_root = {
        "schema",
        "provider",
        "source_uri",
        "source_sha256",
        "coverage_start",
        "coverage_end",
        "covered_tickers",
        "complete_event_types",
        "point_in_time_revisions",
        "records",
        "manifest_hash",
    }
    if set(value) != allowed_root:
        raise HistoricalEventError("historical event manifest contains unfrozen fields")
    supplied_hash = _digest(value["manifest_hash"], "historical event manifest hash")
    payload = dict(value)
    payload.pop("manifest_hash")
    if hashlib.sha256(_canonical(payload)).hexdigest() != supplied_hash:
        raise HistoricalEventError("historical event manifest hash does not reconcile")
    raw_records = value.get("records")
    if not isinstance(raw_records, list):
        raise HistoricalEventError("historical event records are missing")
    allowed_record = {
        "ticker",
        "event_type",
        "effective_date",
        "observed_at",
        "available_at",
        "source_event_id",
        "status",
        "cash_amount",
        "split_ratio",
        "adjustment_reference",
    }
    records = []
    for raw in raw_records:
        if not isinstance(raw, Mapping) or set(raw) != allowed_record:
            raise HistoricalEventError("historical event record contains unfrozen fields")
        records.append(
            HistoricalEventRecord(
                ticker=str(raw["ticker"]),
                event_type=str(raw["event_type"]),
                effective_date=_date(raw["effective_date"], "effective_date"),
                observed_at=_timestamp(raw["observed_at"], "observed_at"),
                available_at=_timestamp(raw["available_at"], "available_at"),
                source_event_id=str(raw["source_event_id"]),
                status=str(raw["status"]),
                cash_amount=raw["cash_amount"],
                split_ratio=raw["split_ratio"],
                adjustment_reference=raw["adjustment_reference"],
            )
        )
    return HistoricalEventManifest(
        provider=str(value["provider"]),
        source_uri=str(value["source_uri"]),
        source_sha256=str(value["source_sha256"]),
        coverage_start=_date(value["coverage_start"], "coverage_start"),
        coverage_end=_date(value["coverage_end"], "coverage_end"),
        covered_tickers=tuple(str(item).strip().upper() for item in value["covered_tickers"]),
        complete_event_types=tuple(str(item).strip().upper() for item in value["complete_event_types"]),
        records=tuple(records),
        point_in_time_revisions=value["point_in_time_revisions"],
        manifest_hash=supplied_hash,
    )


def event_manifest_payload(
    *,
    provider: str,
    source_uri: str,
    source_sha256: str,
    coverage_start: date,
    coverage_end: date,
    covered_tickers: Sequence[str],
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Build a deterministic payload; it still requires independent source data."""

    payload = {
        "schema": "cultra.historical-events.v1",
        "provider": str(provider),
        "source_uri": str(source_uri),
        "source_sha256": _digest(source_sha256, "historical event source hash"),
        "coverage_start": coverage_start.isoformat(),
        "coverage_end": coverage_end.isoformat(),
        "covered_tickers": sorted(set(str(item).strip().upper() for item in covered_tickers)),
        "complete_event_types": list(EVENT_TYPES),
        "point_in_time_revisions": True,
        "records": [dict(item) for item in records],
    }
    return dict(payload, manifest_hash=hashlib.sha256(_canonical(payload)).hexdigest())


__all__ = [
    "EVENT_TYPES",
    "HistoricalEventError",
    "HistoricalEventManifest",
    "HistoricalEventRecord",
    "event_manifest_payload",
    "load_historical_event_manifest",
]
