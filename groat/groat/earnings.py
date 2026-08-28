"""Next earnings date.

ORATS Delayed cores.nextErn is NOT broken: the field is documented as
'available through another subscription'. daysToNextErn is deprecated.
Use lastErn, ernDate1-12 (M/D/YYYY), wksNextErn, /hist/earnings, then
Schwab (no calendar), then public web (AlphaQuery / Nasdaq).
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from groat.config import CODE_DIR, EARNINGS_EXEMPT, EARNINGS_HOLD_DAYS
from groat.dates import parse_any_date
from groat.num import to_float

ALPHA_NEXT = re.compile(
    r"next expected annou[n]?cement date[^\d]{0,80}(20\d{2}-\d{2}-\d{2})",
    re.I,
)
WEB_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def earnings_cache_dir() -> Path:
    return CODE_DIR / "var" / "earnings"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _days_between(later: str, earlier: str) -> Optional[int]:
    try:
        return (datetime.strptime(later[:10], "%Y-%m-%d") - datetime.strptime(earlier[:10], "%Y-%m-%d")).days
    except ValueError:
        return None


def _add_days(asof: str, days: int) -> Optional[str]:
    try:
        return (datetime.strptime(asof[:10], "%Y-%m-%d") + timedelta(days=int(days))).date().isoformat()
    except ValueError:
        return None


def parse_ern_history(core: Optional[dict]) -> List[str]:
    core = core or {}
    dates = []
    seen = set()
    last = parse_any_date(core.get("lastErn"))
    if last:
        dates.append(last)
        seen.add(last)
    for i in range(1, 13):
        day = parse_any_date(core.get("ernDate%s" % i))
        if day and day not in seen:
            dates.append(day)
            seen.add(day)
    dates.sort()
    return dates


def cadence_next(history: Sequence[str], asof: str) -> Optional[str]:
    past = [d for d in history if d <= asof]
    if len(past) < 2:
        return None
    gaps = []
    for i in range(1, len(past)):
        gap = _days_between(past[i], past[i - 1])
        if gap and 60 <= gap <= 150:
            gaps.append(gap)
    if not gaps:
        return None
    gaps.sort()
    median = gaps[len(gaps) // 2]
    nxt = _add_days(past[-1], median)
    if nxt and nxt > asof:
        return nxt
    return None


def from_cores(ticker: str, core: Optional[dict], asof: str) -> Dict[str, Any]:
    name = str(ticker).upper()
    if name in EARNINGS_EXEMPT:
        return {
            "ticker": name,
            "date": None,
            "last": None,
            "source": "exempt",
            "usable": True,
            "days": None,
            "overlaps_hold": False,
            "note": "ETF/index exempt from single-name earnings gate",
            "raw_nextErn": str((core or {}).get("nextErn") or ""),
            "raw_daysToNextErn": (core or {}).get("daysToNextErn"),
            "raw_wksNextErn": (core or {}).get("wksNextErn"),
        }
    core = core or {}
    history = parse_ern_history(core)
    last = history[-1] if history else None
    # nextErn / daysToNextErn are Delayed placeholders, not live calendar.
    nxt = parse_any_date(core.get("nextErn"))
    source = None
    if nxt and nxt >= asof:
        source = "orats.nextErn"
    else:
        nxt = None
    wks = to_float(core.get("wksNextErn"))
    if nxt is None and wks is not None and 1 <= wks <= 26:
        guess = _add_days(asof, int(round(wks * 7)))
        if guess and guess > asof:
            nxt = guess
            source = "orats.wksNextErn"
    if nxt is None:
        cad = cadence_next(history, asof)
        if cad:
            nxt = cad
            source = "orats.ernDate_cadence"
    days = _days_between(nxt, asof) if nxt else None
    overlaps = bool(days is not None and 0 <= days <= EARNINGS_HOLD_DAYS)
    usable = nxt is not None
    return {
        "ticker": name,
        "date": nxt,
        "last": last,
        "source": source or "DATA UNAVAILABLE",
        "usable": usable,
        "days": days,
        "overlaps_hold": overlaps,
        "note": "" if usable else "DATA UNAVAILABLE",
        "history": history[-8:],
        "raw_nextErn": str(core.get("nextErn") or ""),
        "raw_daysToNextErn": core.get("daysToNextErn"),
        "raw_wksNextErn": core.get("wksNextErn"),
    }


def _http_json(url: str, timeout: float = 25.0) -> Optional[Any]:
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


def _http_text(url: str, timeout: float = 25.0) -> Optional[str]:
    req = urllib.request.Request(url, headers=WEB_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read() or b""
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return None
    try:
        return raw.decode("utf-8", "replace")
    except Exception:
        return None


_SCHWAB_CALENDAR = None  # None unknown, False confirmed absent, str field name if present


def schwab_earnings_date(ticker: str) -> Dict[str, Any]:
    """Schwab instruments?projection=fundamental has EPS, not a next-earnings calendar."""
    global _SCHWAB_CALENDAR
    if _SCHWAB_CALENDAR is False:
        return {"ok": True, "date": None, "note": "schwab_fundamental_no_earnings_calendar"}
    from groat.schwab import MARKET, _access_token

    token = _access_token()
    if not token:
        return {"ok": False, "date": None, "note": "missing_schwab"}
    query = urllib.parse.urlencode({"symbol": ticker, "projection": "fundamental"})
    req = urllib.request.Request(
        "%s/instruments?%s" % (MARKET, query),
        headers={"Authorization": "Bearer %s" % token, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError):
        return {"ok": False, "date": None, "note": "schwab_http"}
    fund = {}
    instruments = payload.get("instruments") if isinstance(payload, dict) else None
    if isinstance(instruments, list) and instruments and isinstance(instruments[0], dict):
        fund = instruments[0].get("fundamental") or {}
    for key in ("earningsDate", "nextEarningsDate", "earnings_date"):
        day = parse_any_date(fund.get(key) if isinstance(fund, dict) else None)
        if day:
            _SCHWAB_CALENDAR = key
            return {"ok": True, "date": day, "note": "schwab.fundamental.%s" % key}
    _SCHWAB_CALENDAR = False
    return {
        "ok": True,
        "date": None,
        "note": "schwab_fundamental_no_earnings_calendar",
        "eps": (fund or {}).get("eps") if isinstance(fund, dict) else None,
    }


def nasdaq_last(ticker: str) -> Optional[str]:
    url = "https://api.nasdaq.com/api/company/%s/earnings-surprise" % urllib.parse.quote(ticker)
    payload = _http_json(url)
    if not isinstance(payload, dict):
        return None
    rows = (((payload.get("data") or {}) or {}).get("earningsSurpriseTable") or {}).get("rows") or []
    dates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        day = parse_any_date(row.get("dateReported"))
        if day:
            dates.append(day)
    dates.sort()
    return dates[-1] if dates else None


def alphaquery_next(ticker: str) -> Optional[str]:
    url = "https://www.alphaquery.com/stock/%s/earnings-history" % urllib.parse.quote(ticker)
    html = _http_text(url)
    if not html:
        return None
    match = ALPHA_NEXT.search(html)
    if not match:
        return None
    return parse_any_date(match.group(1))


def web_resolve(ticker: str, asof: str, use_web: bool = True) -> Dict[str, Any]:
    name = str(ticker).upper()
    cache = earnings_cache_dir() / ("%s.json" % name)
    cached = _read_json(cache)
    if isinstance(cached, dict) and cached.get("asof") == asof:
        return cached
    out = {
        "ticker": name,
        "asof": asof,
        "web_next": None,
        "web_last": None,
        "web_source": None,
        "schwab_note": None,
    }
    if not use_web:
        return out
    nxt = alphaquery_next(name)
    if nxt and nxt >= asof:
        out["web_next"] = nxt
        out["web_source"] = "web.alphaquery"
    last = nasdaq_last(name)
    if last:
        out["web_last"] = last
    if not out["web_next"]:
        schwab = schwab_earnings_date(name)
        out["schwab_note"] = schwab.get("note")
        if schwab.get("date"):
            out["web_next"] = schwab["date"]
            out["web_source"] = schwab.get("note")
    else:
        out["schwab_note"] = "skipped_web_already_had_next"
    _write_json(cache, out)
    return out


def merge_hist(base: Dict[str, Any], hist_rows: Optional[Sequence[dict]], asof: str) -> Dict[str, Any]:
    dates = list(base.get("history") or [])
    for row in hist_rows or []:
        day = parse_any_date((row or {}).get("earnDate"))
        if day and day not in dates:
            dates.append(day)
    dates.sort()
    past = [d for d in dates if d <= asof]
    last = past[-1] if past else base.get("last")
    out = dict(base)
    out["history"] = dates[-8:]
    out["last"] = last
    if not out.get("date"):
        cad = cadence_next(dates, asof)
        if cad:
            out["date"] = cad
            out["source"] = "orats.hist/earnings cadence"
            out["usable"] = True
            out["note"] = ""
            out["days"] = _days_between(cad, asof)
            out["overlaps_hold"] = bool(out["days"] is not None and 0 <= out["days"] <= EARNINGS_HOLD_DAYS)
    return out


def apply_web(base: Dict[str, Any], web: Dict[str, Any], asof: str) -> Dict[str, Any]:
    out = dict(base)
    if web.get("web_last") and (not out.get("last") or web["web_last"] > str(out.get("last") or "")):
        out["last"] = web["web_last"]
    nxt = web.get("web_next")
    if nxt and nxt >= asof:
        out["date"] = nxt
        out["source"] = web.get("web_source") or "web"
        out["usable"] = True
        out["note"] = ""
        out["days"] = _days_between(nxt, asof)
        out["overlaps_hold"] = bool(out["days"] is not None and 0 <= out["days"] <= EARNINGS_HOLD_DAYS)
    out["schwab_note"] = web.get("schwab_note")
    return out


def resolve(
    ticker: str,
    asof: str,
    core: Optional[dict] = None,
    hist_rows: Optional[Sequence[dict]] = None,
    use_web: bool = False,
    web_payload: Optional[dict] = None,
) -> Dict[str, Any]:
    info = from_cores(ticker, core, asof)
    if hist_rows:
        info = merge_hist(info, hist_rows, asof)
    if web_payload is not None:
        info = apply_web(info, web_payload, asof)
    elif use_web and str(ticker).upper() not in EARNINGS_EXEMPT:
        info = apply_web(info, web_resolve(ticker, asof, use_web=True), asof)
    return info
