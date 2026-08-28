"""Market-session clock helpers.

CORAT keys current-vs-historical routing to the New York trading date rather
than the host's local date or UTC date.  The explicit as-of date remains the
authoritative research cutoff.
"""

from __future__ import annotations

from datetime import datetime

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python 3.9+ includes zoneinfo
    ZoneInfo = None  # type: ignore


def today_new_york() -> str:
    if ZoneInfo is None:
        return datetime.now().date().isoformat()
    return datetime.now(ZoneInfo("America/New_York")).date().isoformat()
