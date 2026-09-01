"""Strict point-in-time market-session calendar for Cultra V2 research."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_TIMEZONE = "America/New_York"


class SessionCalendarError(ValueError):
    """The historical session calendar is incomplete or not reproducible."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: Any, label: str) -> str:
    normalized = str(value).lower().removeprefix("sha256:")
    if len(normalized) != 64 or any(
        char not in "0123456789abcdef" for char in normalized
    ):
        raise SessionCalendarError("%s is not a SHA-256 digest" % label)
    return normalized


def _timestamp(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SessionCalendarError("session close_at must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SessionCalendarError("session close_at must be timezone-aware")
    return parsed


@dataclass(frozen=True)
class MarketSession:
    session_date: date
    close_at: datetime

    def __post_init__(self) -> None:
        if self.session_date.weekday() >= 5:
            raise SessionCalendarError("market sessions cannot fall on weekends")
        if self.close_at.tzinfo is None or self.close_at.utcoffset() is None:
            raise SessionCalendarError("session close_at must be timezone-aware")
        local = self.close_at.astimezone(ZoneInfo(MARKET_TIMEZONE))
        if local.date() != self.session_date:
            raise SessionCalendarError("session close timestamp has the wrong market date")
        if not time(12, 0) <= local.time().replace(tzinfo=None) <= time(17, 0):
            raise SessionCalendarError("session close timestamp is outside market-close bounds")


@dataclass(frozen=True)
class HistoricalSessionCalendar:
    provider: str
    source_uri: str
    source_sha256: str
    timezone: str
    sessions: Tuple[MarketSession, ...]
    calendar_hash: str

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.source_uri.strip():
            raise SessionCalendarError("session provider and source URI are required")
        _digest(self.source_sha256, "session source hash")
        if self.timezone != MARKET_TIMEZONE:
            raise SessionCalendarError("session calendar timezone is not frozen")
        dates = tuple(item.session_date for item in self.sessions)
        if not dates or dates != tuple(sorted(set(dates))):
            raise SessionCalendarError("market sessions must be non-empty, sorted and unique")
        _digest(self.calendar_hash, "session calendar hash")

    @property
    def dates(self) -> Tuple[str, ...]:
        return tuple(item.session_date.isoformat() for item in self.sessions)

    def close_for(self, session_date: date) -> datetime:
        for item in self.sessions:
            if item.session_date == session_date:
                return item.close_at
        raise SessionCalendarError("date is outside the frozen session calendar")


def session_calendar_payload(
    *,
    provider: str,
    source_uri: str,
    source_sha256: str,
    sessions: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    payload = {
        "schema": "cultra.market-session-calendar.v1",
        "provider": str(provider),
        "source_uri": str(source_uri),
        "source_sha256": _digest(source_sha256, "session source hash"),
        "timezone": MARKET_TIMEZONE,
        "sessions": [dict(item) for item in sessions],
    }
    return dict(payload, calendar_hash=hashlib.sha256(_canonical(payload)).hexdigest())


def load_historical_session_calendar(
    path: Path, *, required_count: int = 450
) -> HistoricalSessionCalendar:
    supplied = Path(path).expanduser().resolve()
    try:
        supplied.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise SessionCalendarError("session calendar must be Cultra-owned") from exc
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SessionCalendarError("session calendar is unreadable") from exc
    if not isinstance(value, Mapping) or value.get("schema") != (
        "cultra.market-session-calendar.v1"
    ):
        raise SessionCalendarError("session calendar schema is unsupported")
    allowed_root = {
        "schema",
        "provider",
        "source_uri",
        "source_sha256",
        "timezone",
        "sessions",
        "calendar_hash",
    }
    if set(value) != allowed_root:
        raise SessionCalendarError("session calendar contains unfrozen fields")
    supplied_hash = _digest(value["calendar_hash"], "session calendar hash")
    payload = dict(value)
    payload.pop("calendar_hash")
    if hashlib.sha256(_canonical(payload)).hexdigest() != supplied_hash:
        raise SessionCalendarError("session calendar hash does not reconcile")
    raw_sessions = value.get("sessions")
    if not isinstance(raw_sessions, list) or len(raw_sessions) != int(required_count):
        raise SessionCalendarError(
            "session calendar must contain exactly %d sessions" % int(required_count)
        )
    sessions = []
    for raw in raw_sessions:
        if not isinstance(raw, Mapping) or set(raw) != {"session_date", "close_at"}:
            raise SessionCalendarError("session record contains unfrozen fields")
        try:
            session_date = date.fromisoformat(str(raw["session_date"]))
        except ValueError as exc:
            raise SessionCalendarError("session_date must use YYYY-MM-DD") from exc
        sessions.append(MarketSession(session_date, _timestamp(raw["close_at"])))
    return HistoricalSessionCalendar(
        provider=str(value["provider"]),
        source_uri=str(value["source_uri"]),
        source_sha256=str(value["source_sha256"]),
        timezone=str(value["timezone"]),
        sessions=tuple(sessions),
        calendar_hash=supplied_hash,
    )


__all__ = [
    "HistoricalSessionCalendar",
    "MARKET_TIMEZONE",
    "MarketSession",
    "SessionCalendarError",
    "load_historical_session_calendar",
    "session_calendar_payload",
]
