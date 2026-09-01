"""Ranked earnings dates. Never invent. Missing → no options."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from xhigh.dates import add_days, days_between, parse_any_date, usable_date
from xhigh.num import to_float

NASDAQ_EARNINGS = "https://api.nasdaq.com/api/company/%s/earnings-surprise"
WEB_UA = {
    "User-Agent": "Mozilla/5.0 xhigh-research",
    "Accept": "application/json",
}


def nasdaq_next(ticker: str, asof: str, fetch=None) -> Optional[Dict[str, Any]]:
    name = str(ticker).upper()
    url = NASDAQ_EARNINGS % name
    getter = fetch or _http_json
    payload = getter(url)
    if not isinstance(payload, dict):
        return None
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    dates = []
    for key in ("earningsDate", "nextEarningsDate", "announceDate"):
        got = parse_any_date(data.get(key)) if isinstance(data, dict) else None
        if got:
            dates.append(got)
    surprise = data.get("earningsSurpriseTable") if isinstance(data, dict) else None
    rows = []
    if isinstance(surprise, dict) and isinstance(surprise.get("rows"), list):
        rows = surprise["rows"]
    elif isinstance(surprise, list):
        rows = surprise
    for row in rows:
        if not isinstance(row, dict):
            continue
        got = parse_any_date(row.get("dateReported") or row.get("fiscalEnd") or row.get("date"))
        if got:
            dates.append(got)
    future = sorted({d for d in dates if d and d >= asof})
    if not future:
        return None
    return {"date": future[0], "source": "nasdaq.earnings-surprise", "url": url}


def _http_json(url: str, timeout: float = 20.0) -> Optional[Any]:
    req = urllib.request.Request(url, headers=WEB_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read() or b""
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def _history_from_core(core: Optional[dict]) -> List[str]:
    core = core or {}
    dates = []
    seen = set()
    last = parse_any_date(core.get("lastErn") or core.get("last_ern"))
    if last:
        dates.append(last)
        seen.add(last)
    raw = core.get("raw") if isinstance(core.get("raw"), dict) else core
    for i in range(1, 13):
        day = parse_any_date(raw.get("ernDate%s" % i) if isinstance(raw, dict) else None)
        if day and day not in seen:
            dates.append(day)
            seen.add(day)
    dates.sort()
    return dates


def cadence_next(history: List[str], asof: str) -> Optional[str]:
    past = [d for d in history if d <= asof]
    if len(past) < 2:
        return None
    gaps = []
    for i in range(1, len(past)):
        gap = days_between(past[i], past[i - 1])
        if gap and 60 <= gap <= 150:
            gaps.append(gap)
    if not gaps:
        return None
    gaps.sort()
    median = gaps[len(gaps) // 2]
    nxt = add_days(past[-1], median)
    while nxt and nxt <= asof:
        nxt = add_days(nxt, median)
    if nxt and nxt > asof:
        return nxt
    return None


def resolve(ticker: str, asof: str, core: Optional[dict] = None, nasdaq: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    name = str(ticker).upper()
    core = core or {}
    if nasdaq and nasdaq.get("date"):
        nxt = str(nasdaq["date"])[:10]
        return {
            "ticker": name,
            "date": nxt,
            "source": nasdaq.get("source") or "nasdaq",
            "url": nasdaq.get("url") or "",
            "usable": True,
            "days": days_between(nxt, asof),
            "note": "",
        }
    nxt = usable_date(core.get("next_ern") or core.get("nextErn"))
    if nxt and nxt >= asof:
        return {
            "ticker": name,
            "date": nxt,
            "source": "orats.nextErn",
            "url": "",
            "usable": True,
            "days": days_between(nxt, asof),
            "note": "",
        }
    wks = to_float(core.get("wks_next_ern") if "wks_next_ern" in core else core.get("wksNextErn"))
    if wks is not None and 1 <= wks <= 26:
        guess = add_days(asof, int(round(wks * 7)))
        if guess and guess > asof:
            return {
                "ticker": name,
                "date": guess,
                "source": "orats.wksNextErn",
                "url": "",
                "usable": True,
                "days": days_between(guess, asof),
                "note": "",
            }
    history = _history_from_core(core)
    cad = cadence_next(history, asof)
    if cad:
        return {
            "ticker": name,
            "date": cad,
            "source": "orats.ernDate_cadence",
            "url": "",
            "usable": True,
            "days": days_between(cad, asof),
            "note": "",
        }
    return {
        "ticker": name,
        "date": None,
        "source": "DATA UNAVAILABLE",
        "url": "",
        "usable": False,
        "days": None,
        "note": "DATA UNAVAILABLE",
    }


def options_allowed(earn: Dict[str, Any], expiry: str, buffer_days: int = 3) -> bool:
    if not earn.get("usable") or not earn.get("date"):
        return False
    gap = days_between(str(earn["date"]), expiry)
    if gap is None:
        return False
    return gap > int(buffer_days)
