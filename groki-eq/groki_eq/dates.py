from datetime import datetime
from typing import Optional

from groki_eq.config import PIT_TZ

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
