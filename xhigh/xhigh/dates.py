from datetime import datetime, timedelta
from typing import Optional

from xhigh.config import PIT_TZ

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
    if text.lower() in ("none", "null", "nan") or text.startswith("0000"):
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


def add_days(iso: str, days: int) -> Optional[str]:
    day = parse_ymd(iso) or usable_date(iso)
    if not day:
        return None
    return (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=int(days))).date().isoformat()


def days_between(later: str, earlier: str) -> Optional[int]:
    a = parse_ymd(later) or usable_date(later)
    b = parse_ymd(earlier) or usable_date(earlier)
    if not a or not b:
        return None
    return (datetime.strptime(a, "%Y-%m-%d") - datetime.strptime(b, "%Y-%m-%d")).days


def fmt_expiry(iso: str) -> str:
    day = parse_ymd(iso) or usable_date(iso)
    if not day:
        return ""
    return datetime.strptime(day, "%Y-%m-%d").strftime("%d %b")
