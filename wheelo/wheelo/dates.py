from datetime import datetime, timedelta
from typing import Optional

from wheelo.config import PIT_TZ

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


def today_et() -> str:
    if ZoneInfo is not None:
        now = datetime.now(ZoneInfo(PIT_TZ))
    else:
        now = datetime.now()
    return now.date().isoformat()


def parse_ymd(value: str) -> Optional[str]:
    text = (value or "").strip()[:10]
    if len(text) != 10:
        return None
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return text


def usable_date(value) -> Optional[str]:
    text = str(value or "").strip()[:10]
    if len(text) != 10:
        return None
    if text.startswith("0000") or text.lower() in ("none", "null", "nan"):
        return None
    try:
        datetime.strptime(text, "%Y-%m-%d")
    except ValueError:
        return None
    return text


def parse_any_date(value) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in ("none", "null", "nan") or text.startswith("0000"):
        return None
    iso = usable_date(text[:10]) if len(text) >= 10 and text[4:5] == "-" else None
    if iso:
        return iso
    cleaned = " ".join(text.replace(",", ", ").split())
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def days_until(asof: str, other: Optional[str]) -> Optional[int]:
    start = parse_ymd(asof)
    end = usable_date(other) or parse_any_date(other)
    if not start or not end:
        return None
    a = datetime.strptime(start, "%Y-%m-%d").date()
    b = datetime.strptime(end, "%Y-%m-%d").date()
    return (b - a).days


def add_days(asof: str, days: int) -> str:
    stamp = datetime.strptime(asof[:10], "%Y-%m-%d").date()
    return (stamp + timedelta(days=int(days))).isoformat()
