from datetime import datetime
from typing import Optional

from groat.config import PIT_TZ

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

# Regular session. Open auction is not a close list. After 16:00 ET is the daily close.
RTH_OPEN_HOUR = 9
RTH_OPEN_MINUTE = 45
RTH_CLOSE_HOUR = 16
RTH_CLOSE_MINUTE = 0


def now_et(now: Optional[datetime] = None) -> datetime:
    if now is not None:
        return now
    if ZoneInfo is not None:
        return datetime.now(ZoneInfo(PIT_TZ))
    return datetime.now()


def today_et(now: Optional[datetime] = None) -> str:
    return now_et(now).date().isoformat()


def session_phase(asof: str, today: str = "", now: Optional[datetime] = None) -> str:
    """open = pre/open auction; rth = 9:45–16:00 ET today; close = official session.

    Historical asof < today is always close. After 16:00 ET on asof==today is close.
    """
    day = (asof or "")[:10]
    today_s = (today or today_et(now))[:10]
    if not day:
        return "close"
    if day < today_s:
        return "close"
    if day > today_s:
        return "open"
    stamp = now_et(now)
    hm = stamp.hour * 60 + stamp.minute
    if hm < RTH_OPEN_HOUR * 60 + RTH_OPEN_MINUTE:
        return "open"
    if hm >= RTH_CLOSE_HOUR * 60 + RTH_CLOSE_MINUTE:
        return "close"
    return "rth"


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
