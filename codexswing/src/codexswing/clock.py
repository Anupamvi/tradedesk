"""Small UTC/session-time helpers with no market-data dependencies."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Optional
from zoneinfo import ZoneInfo


UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def iso_utc(value: datetime) -> str:
    return ensure_utc(value).isoformat().replace("+00:00", "Z")


def parse_timestamp(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp is empty")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    return ensure_utc(parsed)


def parse_optional_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None or not value.strip():
        return None
    return parse_timestamp(value)


def session_close_utc(session_date: str) -> datetime:
    parsed_date = date.fromisoformat(session_date)
    local_close = datetime.combine(parsed_date, time(hour=16), tzinfo=NEW_YORK)
    return local_close.astimezone(UTC)

