"""Cache-first forward earnings-calendar evidence.

ORATS exact next-earnings dates are subscription-dependent. CORAT therefore
checks the public Nasdaq earnings calendar across the intended holding window.
Nasdaq labels this calendar as estimated; CORAT preserves that provenance and
uses it only to prevent an ordinary option swing from unknowingly crossing an
earnings event.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

from corat.models import SourceTrace
from corat.store import read_json, sha256_file, utc_now, write_json


@dataclass
class EarningsCalendarBundle:
    dates_by_ticker: Dict[str, str] = field(default_factory=dict)
    traces: List[SourceTrace] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    checked_through: str = ""


def _cache_path(cache_root: Path, day: str) -> Path:
    return cache_root / "nasdaq_earnings" / (day + ".json")


def _payload_rows(payload: object) -> List[dict]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    rows = data.get("rows")
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def fetch_forward_earnings_calendar(
    as_of: str,
    holding_sessions: int,
    cache_root: Path,
    timeout_seconds: float = 15.0,
    offline: bool = False,
    refresh: bool = False,
) -> EarningsCalendarBundle:
    start = date.fromisoformat(as_of) + timedelta(days=1)
    end = date.fromisoformat(as_of) + timedelta(days=max(3, int(holding_sessions * 1.8)))
    result = EarningsCalendarBundle(checked_through=end.isoformat())
    day = start
    while day <= end:
        if day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        day_text = day.isoformat()
        path = _cache_path(cache_root, day_text)
        cached = read_json(path)
        payload = cached.get("payload") if isinstance(cached, dict) else None
        fetched_at = str(cached.get("fetched_at_utc") or "") if isinstance(cached, dict) else ""
        status = "CACHED"
        error = ""
        if payload is None or refresh:
            if offline:
                error = "Nasdaq earnings calendar cache miss for {}".format(day_text)
            else:
                endpoint = "https://api.nasdaq.com/api/calendar/earnings?date={}".format(day_text)
                request = urllib.request.Request(
                    endpoint,
                    headers={
                        "Accept": "application/json, text/plain, */*",
                        "Origin": "https://www.nasdaq.com",
                        "Referer": "https://www.nasdaq.com/market-activity/earnings",
                        "User-Agent": "Mozilla/5.0 CORAT research-only",
                    },
                )
                try:
                    with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                        payload = json.loads((response.read() or b"{}").decode("utf-8"))
                    fetched_at = utc_now()
                    status = "LIVE_FETCH"
                    write_json(
                        path,
                        {
                            "schema_version": "corat.nasdaq_earnings_cache.v1",
                            "date": day_text,
                            "fetched_at_utc": fetched_at,
                            "payload": payload,
                        },
                    )
                except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeDecodeError) as exc:
                    error = "Nasdaq earnings calendar {} for {}".format(type(exc).__name__, day_text)
                    if isinstance(cached, dict):
                        payload = cached.get("payload")
                        fetched_at = str(cached.get("fetched_at_utc") or "")
                        status = "STALE_CACHE"
        rows = _payload_rows(payload)
        for row in rows:
            ticker = str(row.get("symbol") or "").strip().upper()
            if ticker and (ticker not in result.dates_by_ticker or day_text < result.dates_by_ticker[ticker]):
                result.dates_by_ticker[ticker] = day_text
        result.traces.append(
            SourceTrace(
                source="NASDAQ EARNINGS CALENDAR (ESTIMATED)",
                endpoint="/api/calendar/earnings",
                status="DATA UNAVAILABLE" if error and not rows else status,
                fetched_at_utc=fetched_at,
                latest_data_at=day_text,
                rows=len(rows),
                cache_path=str(path),
                cache_sha256=sha256_file(path) if path.is_file() else "",
                params={"date": day_text},
                error=error,
            )
        )
        if error:
            result.errors.append(error)
        day += timedelta(days=1)
    return result
