"""Append-only prospective recommendation registry for Options Agent.

The registry deliberately does not infer recommendations from historical output.
Only recommendations explicitly registered as live on their recommendation date
can become active or match a later broker execution.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
DEFAULT_VALIDITY_MINUTES = 15
GREEN_STATUS = "GREEN"
_OCC_RE = re.compile(r"^[A-Z0-9.]{1,8}\d{6}[CP]\d{8}$")


class RegistryValidationError(ValueError):
    """Raised when a registry operation receives invalid data."""


class RegistryCorruptionError(RuntimeError):
    """Raised when an existing registry line cannot be decoded."""


class BrokerMatchReason(str, Enum):
    MATCHED = "MATCHED"
    NO_ACTIVE_RECOMMENDATION = "NO_ACTIVE_RECOMMENDATION"
    PRE_REGISTRATION_FILL = "PRE_REGISTRATION_FILL"
    REVERSE_OR_DIFFERENT_LEGS = "REVERSE_OR_DIFFERENT_LEGS"
    ACCOUNT_MISMATCH = "ACCOUNT_MISMATCH"
    AMBIGUOUS_ACTIVE_RECOMMENDATIONS = "AMBIGUOUS_ACTIVE_RECOMMENDATIONS"


@dataclass(frozen=True, order=True)
class DirectedLeg:
    """One directed option leg and its positive integer structure ratio."""

    side: str
    occ_symbol: str
    ratio: int = 1

    def __post_init__(self) -> None:
        side = str(self.side).strip().upper()
        symbol = _canonical_occ_symbol(self.occ_symbol)
        ratio = _positive_int(self.ratio, field_name="ratio")
        if side not in {"BUY", "SELL"}:
            raise RegistryValidationError("leg side must be BUY or SELL")
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "occ_symbol", symbol)
        object.__setattr__(self, "ratio", ratio)

    def to_dict(self) -> dict[str, Any]:
        return {"side": self.side, "ratio": self.ratio, "occ_symbol": self.occ_symbol}


@dataclass(frozen=True)
class RecommendationEvent:
    schema_version: int
    sequence: int
    event_id: str
    registered_at: dt.datetime
    valid_until: dt.datetime
    logical_recommendation_id: str
    account_id: str
    recommendation_date: dt.date
    status: str
    live_current_date: bool
    eligible: bool
    eligibility_reason: str
    legs: tuple[DirectedLeg, ...]
    code_provenance: Mapping[str, Any]
    run_provenance: Mapping[str, Any]

    @property
    def logical_key(self) -> tuple[str, str]:
        return (self.account_id, self.logical_recommendation_id)

    @property
    def is_active(self) -> bool:
        return self.eligible and self.status == GREEN_STATUS

    def is_active_at(self, timestamp: dt.datetime) -> bool:
        return self.is_active and self.registered_at <= timestamp <= self.valid_until

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_type": "recommendation_registered",
            "sequence": self.sequence,
            "event_id": self.event_id,
            "registered_at": _format_utc(self.registered_at),
            "valid_until": _format_utc(self.valid_until),
            "logical_recommendation_id": self.logical_recommendation_id,
            "account_id": self.account_id,
            "recommendation_date": self.recommendation_date.isoformat(),
            "status": self.status,
            "live_current_date": self.live_current_date,
            "eligible": self.eligible,
            "eligibility_reason": self.eligibility_reason,
            "legs": [leg.to_dict() for leg in self.legs],
            "code_provenance": dict(self.code_provenance),
            "run_provenance": dict(self.run_provenance),
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RecommendationEvent":
        if row.get("schema_version") != SCHEMA_VERSION:
            raise RegistryCorruptionError(
                f"unsupported schema_version: {row.get('schema_version')!r}"
            )
        if row.get("event_type") != "recommendation_registered":
            raise RegistryCorruptionError(
                f"unsupported event_type: {row.get('event_type')!r}"
            )
        try:
            return cls(
                schema_version=SCHEMA_VERSION,
                sequence=int(row["sequence"]),
                event_id=str(row["event_id"]),
                registered_at=_parse_utc_datetime(
                    row["registered_at"], field_name="registered_at"
                ),
                valid_until=_parse_utc_datetime(
                    row["valid_until"], field_name="valid_until"
                ),
                logical_recommendation_id=_required_text(
                    row["logical_recommendation_id"], "logical_recommendation_id"
                ),
                account_id=_required_text(row["account_id"], "account_id"),
                recommendation_date=_parse_date(row["recommendation_date"]),
                status=_required_text(row["status"], "status").upper(),
                live_current_date=_strict_bool(
                    row["live_current_date"], "live_current_date"
                ),
                eligible=_strict_bool(row["eligible"], "eligible"),
                eligibility_reason=str(row["eligibility_reason"]),
                legs=_canonical_recommendation_legs(row["legs"]),
                code_provenance=_provenance(row["code_provenance"], "code_provenance"),
                run_provenance=_provenance(row["run_provenance"], "run_provenance"),
            )
        except (KeyError, TypeError, ValueError, RegistryValidationError) as exc:
            raise RegistryCorruptionError(f"invalid recommendation event: {exc}") from exc


@dataclass(frozen=True)
class BrokerMatch:
    matched: bool
    reason: BrokerMatchReason
    recommendation: RecommendationEvent | None = None
    candidate_logical_ids: tuple[str, ...] = ()


class ForwardRecommendationRegistry:
    """File-backed append-only registry with prospective broker matching."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def register(
        self,
        *,
        logical_recommendation_id: str,
        account_id: str,
        recommendation_date: dt.date | str,
        status: str,
        legs: Sequence[DirectedLeg | Mapping[str, Any]],
        code_provenance: Mapping[str, Any],
        run_provenance: Mapping[str, Any],
        live_current_date: bool = False,
        valid_until: dt.datetime | str | None = None,
    ) -> RecommendationEvent:
        """Append a state event, or return the latest event for an identical retry."""

        logical_id = _required_text(logical_recommendation_id, "logical_recommendation_id")
        account = _required_text(account_id, "account_id")
        rec_date = _parse_date(recommendation_date)
        normalized_status = _required_text(status, "status").upper()
        normalized_legs = _canonical_recommendation_legs(legs)
        code = _provenance(code_provenance, "code_provenance")
        run = _provenance(run_provenance, "run_provenance")
        explicitly_live = _strict_bool(live_current_date, "live_current_date")

        registered_at = _utc_now()
        expires_at = (
            _parse_utc_datetime(valid_until, field_name="valid_until")
            if valid_until is not None
            else registered_at + dt.timedelta(minutes=DEFAULT_VALIDITY_MINUTES)
        )
        if expires_at < registered_at:
            raise RegistryValidationError("valid_until cannot precede registered_at")
        eligible = explicitly_live and rec_date == registered_at.date()
        if eligible:
            eligibility_reason = "live_current_date"
        elif rec_date != registered_at.date():
            eligibility_reason = "backdated_or_future_recommendation_date"
        else:
            eligibility_reason = "not_explicitly_registered_live"

        request_payload = {
            "logical_recommendation_id": logical_id,
            "account_id": account,
            "recommendation_date": rec_date.isoformat(),
            "status": normalized_status,
            "live_current_date": explicitly_live,
            "valid_until": _format_utc(expires_at),
            "legs": [leg.to_dict() for leg in normalized_legs],
            "code_provenance": code,
            "run_provenance": run,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                events = _decode_lines(handle)
                latest = _latest_by_key(events).get((account, logical_id))
                if (
                    latest is not None
                    and _event_request_payload(latest) == request_payload
                ):
                    return latest

                event = RecommendationEvent(
                    schema_version=SCHEMA_VERSION,
                    sequence=max((item.sequence for item in events), default=0) + 1,
                    event_id=str(uuid.uuid4()),
                    registered_at=registered_at,
                    valid_until=expires_at,
                    logical_recommendation_id=logical_id,
                    account_id=account,
                    recommendation_date=rec_date,
                    status=normalized_status,
                    live_current_date=explicitly_live,
                    eligible=eligible,
                    eligibility_reason=eligibility_reason,
                    legs=normalized_legs,
                    code_provenance=code,
                    run_provenance=run,
                )
                handle.seek(0, 2)
                serialized = json.dumps(
                    event.to_dict(), sort_keys=True, separators=(",", ":")
                )
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                return event
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def events(self) -> tuple[RecommendationEvent, ...]:
        if not self.path.exists():
            return ()
        with self.path.open("r", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            try:
                return tuple(_decode_lines(handle))
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def current_state(
        self, *, account_id: str | None = None
    ) -> tuple[RecommendationEvent, ...]:
        """Return the latest event for each logical key, including downgrades."""

        account = _optional_account(account_id)
        latest = _latest_by_key(self.events()).values()
        return tuple(
            sorted(
                (event for event in latest if account is None or event.account_id == account),
                key=lambda event: event.sequence,
            )
        )

    def current_active_state(
        self, *, account_id: str | None = None
    ) -> tuple[RecommendationEvent, ...]:
        """Return current eligible GREEN recommendations after supersession."""

        now = _utc_now()
        return tuple(
            event
            for event in self.current_state(account_id=account_id)
            if event.is_active_at(now)
        )

    def match_broker_fill(
        self,
        *,
        account_id: str,
        fill_timestamp: dt.datetime | str,
        legs: Sequence[DirectedLeg | Mapping[str, Any]],
    ) -> BrokerMatch:
        """Match one broker execution to exactly one prospective recommendation."""

        account = _required_text(account_id, "account_id")
        filled_at = _parse_utc_datetime(fill_timestamp, field_name="fill_timestamp")
        fill_legs = _canonical_ratio_legs(legs)
        events = self.events()
        active = tuple(
            event
            for event in _latest_by_key(
                item for item in events if item.registered_at <= filled_at
            ).values()
            if event.is_active_at(filled_at)
        )

        same_structure = [event for event in active if event.legs == fill_legs]
        same_account_structure = [
            event for event in same_structure if event.account_id == account
        ]
        if not same_account_structure:
            future_same_account_structure = [
                event
                for event in events
                if event.account_id == account
                and event.legs == fill_legs
                and event.registered_at > filled_at
            ]
            if future_same_account_structure:
                return BrokerMatch(
                    False,
                    BrokerMatchReason.PRE_REGISTRATION_FILL,
                    candidate_logical_ids=_candidate_ids(future_same_account_structure),
                )
            if same_structure:
                return BrokerMatch(False, BrokerMatchReason.ACCOUNT_MISMATCH)
            if any(event.account_id == account for event in active):
                return BrokerMatch(False, BrokerMatchReason.REVERSE_OR_DIFFERENT_LEGS)
            return BrokerMatch(False, BrokerMatchReason.NO_ACTIVE_RECOMMENDATION)

        prospective = [
            event
            for event in same_account_structure
            if filled_at >= event.registered_at
        ]
        if not prospective:
            future_same_structure = [
                event
                for event in events
                if event.account_id == account
                and event.legs == fill_legs
                and event.registered_at > filled_at
            ]
            if not future_same_structure:
                return BrokerMatch(False, BrokerMatchReason.NO_ACTIVE_RECOMMENDATION)
            return BrokerMatch(
                False,
                BrokerMatchReason.PRE_REGISTRATION_FILL,
                candidate_logical_ids=_candidate_ids(future_same_structure),
            )
        if len(prospective) != 1:
            return BrokerMatch(
                False,
                BrokerMatchReason.AMBIGUOUS_ACTIVE_RECOMMENDATIONS,
                candidate_logical_ids=_candidate_ids(prospective),
            )
        return BrokerMatch(
            True,
            BrokerMatchReason.MATCHED,
            recommendation=prospective[0],
            candidate_logical_ids=(prospective[0].logical_recommendation_id,),
        )


def _required_text(value: Any, field_name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise RegistryValidationError(f"{field_name} is required")
    return text


def _optional_account(value: str | None) -> str | None:
    return None if value is None else _required_text(value, "account_id")


def _canonical_occ_symbol(value: Any) -> str:
    symbol = re.sub(r"\s+", "", _required_text(value, "occ_symbol").upper())
    if not _OCC_RE.fullmatch(symbol):
        raise RegistryValidationError(f"invalid OCC symbol: {value!r}")
    return symbol


def _positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise RegistryValidationError(f"{field_name} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError(f"{field_name} must be a positive integer") from exc
    if number <= 0 or (isinstance(value, float) and number != value):
        raise RegistryValidationError(f"{field_name} must be a positive integer")
    return number


def _strict_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise RegistryValidationError(f"{field_name} must be a boolean")
    return value


def _leg_from_value(value: DirectedLeg | Mapping[str, Any]) -> DirectedLeg:
    if isinstance(value, DirectedLeg):
        return value
    if not isinstance(value, Mapping):
        raise RegistryValidationError("each leg must be a DirectedLeg or mapping")
    side = value.get("side", value.get("action"))
    symbol = value.get("occ_symbol", value.get("symbol"))
    ratio = value.get("ratio", value.get("quantity", value.get("qty", 1)))
    return DirectedLeg(side=side, occ_symbol=symbol, ratio=ratio)


def _canonical_recommendation_legs(
    legs: Sequence[DirectedLeg | Mapping[str, Any]],
) -> tuple[DirectedLeg, ...]:
    normalized = tuple(sorted(_leg_from_value(leg) for leg in legs))
    if not normalized:
        raise RegistryValidationError("at least one directed leg is required")
    identities = [(leg.side, leg.occ_symbol) for leg in normalized]
    if len(set(identities)) != len(identities):
        raise RegistryValidationError("duplicate directed OCC leg")
    divisor = math.gcd(*(leg.ratio for leg in normalized))
    return tuple(
        DirectedLeg(leg.side, leg.occ_symbol, leg.ratio // divisor)
        for leg in normalized
    )


def _canonical_ratio_legs(
    legs: Sequence[DirectedLeg | Mapping[str, Any]],
) -> tuple[DirectedLeg, ...]:
    return _canonical_recommendation_legs(legs)


def _parse_date(value: dt.date | str) -> dt.date:
    if isinstance(value, dt.datetime):
        raise RegistryValidationError("recommendation_date must be a date, not a datetime")
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError("recommendation_date must use YYYY-MM-DD") from exc


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_utc_datetime(value: dt.datetime | str, *, field_name: str) -> dt.datetime:
    if isinstance(value, str):
        candidate = value.strip()
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            value = dt.datetime.fromisoformat(candidate)
        except ValueError as exc:
            raise RegistryValidationError(f"{field_name} must be an ISO-8601 datetime") from exc
    if not isinstance(value, dt.datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RegistryValidationError(f"{field_name} must be timezone-aware")
    return value.astimezone(dt.timezone.utc)


def _format_utc(value: dt.datetime) -> str:
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _provenance(value: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise RegistryValidationError(f"{field_name} must be a non-empty mapping")
    copied = dict(value)
    try:
        return json.loads(json.dumps(copied, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise RegistryValidationError(f"{field_name} must be JSON serializable") from exc


def _decode_lines(handle: Iterable[str]) -> list[RecommendationEvent]:
    events: list[RecommendationEvent] = []
    expected_sequence = 1
    for line_number, line in enumerate(handle, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            event = RecommendationEvent.from_dict(row)
        except (json.JSONDecodeError, RegistryCorruptionError) as exc:
            raise RegistryCorruptionError(f"invalid registry line {line_number}: {exc}") from exc
        if event.sequence != expected_sequence:
            raise RegistryCorruptionError(
                f"invalid registry sequence at line {line_number}: "
                f"expected {expected_sequence}, got {event.sequence}"
            )
        events.append(event)
        expected_sequence += 1
    return events


def _latest_by_key(
    events: Iterable[RecommendationEvent],
) -> dict[tuple[str, str], RecommendationEvent]:
    latest: dict[tuple[str, str], RecommendationEvent] = {}
    for event in events:
        latest[event.logical_key] = event
    return latest


def _event_request_payload(event: RecommendationEvent) -> dict[str, Any]:
    return {
        "logical_recommendation_id": event.logical_recommendation_id,
        "account_id": event.account_id,
        "recommendation_date": event.recommendation_date.isoformat(),
        "status": event.status,
        "live_current_date": event.live_current_date,
        "valid_until": _format_utc(event.valid_until),
        "legs": [leg.to_dict() for leg in event.legs],
        "code_provenance": dict(event.code_provenance),
        "run_provenance": dict(event.run_provenance),
    }


def _candidate_ids(events: Iterable[RecommendationEvent]) -> tuple[str, ...]:
    return tuple(sorted(event.logical_recommendation_id for event in events))


__all__ = [
    "BrokerMatch",
    "BrokerMatchReason",
    "DirectedLeg",
    "ForwardRecommendationRegistry",
    "GREEN_STATUS",
    "RecommendationEvent",
    "RegistryCorruptionError",
    "RegistryValidationError",
]
