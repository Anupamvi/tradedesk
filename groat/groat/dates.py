from datetime import datetime
from typing import Optional

from groat.config import PIT_TZ

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
    """ISO YYYY-MM-DD only. Use parse_any_date for ORATS M/D/YYYY fields."""
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
    """ORATS Delayed uses ISO on lastErn and M/D/YYYY on ernDate1-12."""
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
    for fmt in ("%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y", "%b. %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date().isoformat()
        except ValueError:
            continue
    return None
