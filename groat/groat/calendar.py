from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from groat.config import MACRO_PATH
from groat.dates import usable_date
from groat.earnings import resolve as resolve_earnings


def load_macro_events() -> List[Dict[str, Any]]:
    if not MACRO_PATH.is_file():
        return []
    try:
        payload = json.loads(MACRO_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    events = payload.get("events") if isinstance(payload, dict) else payload
    out = []
    for row in events or []:
        if not isinstance(row, dict):
            continue
        day = usable_date(row.get("date"))
        if not day:
            continue
        out.append(
            {
                "date": day,
                "event": str(row.get("event") or ""),
                "impact": str(row.get("impact") or ""),
            }
        )
    return out


def events_between(start: str, end: str) -> List[Dict[str, Any]]:
    rows = []
    for row in load_macro_events():
        day = row["date"]
        if start <= day <= end:
            rows.append(row)
    return rows


def earnings_info(
    ticker: str,
    core: Optional[dict],
    asof: str,
    hist_rows: Optional[list] = None,
    use_web: bool = False,
    web_payload: Optional[dict] = None,
) -> Dict[str, Any]:
    return resolve_earnings(
        ticker,
        asof,
        core=core,
        hist_rows=hist_rows,
        use_web=use_web,
        web_payload=web_payload,
    )
