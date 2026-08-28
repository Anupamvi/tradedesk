"""ORATS Delayed Data API. Cores/strikes only after a shortlist. Never log the token."""

from __future__ import annotations

import gzip
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from wheelo.config import (
    CODE_DIR,
    CORE_FIELDS,
    ORATS_BASE,
    ORATS_MAX_PER_MIN,
    ORATS_MONTHLY_CAP,
    ORATS_STRIKE_DTE,
    ORATS_TICKER_BATCH,
)
from wheelo.num import iv_decimal, to_float

BASE = ORATS_BASE
_TOKEN_QUERY = re.compile(r"(token=)[^&]+", re.I)
_last_request_mono = 0.0
_process_http = 0

GetFn = Callable[..., Tuple[int, Optional[Any], str]]


def archive_dir() -> Path:
    return CODE_DIR / "var" / "orats_archive"


def usage_path() -> Path:
    return archive_dir() / "usage.json"


def redact(text: str, token: Optional[str] = None) -> str:
    if not text:
        return text
    out = _TOKEN_QUERY.sub(r"\1REDACTED", text)
    if token:
        out = out.replace(token, "REDACTED")
    return out


def _month_key(now: Optional[datetime] = None) -> str:
    stamp = now or datetime.now()
    return stamp.strftime("%Y-%m")


def load_usage() -> Dict[str, Any]:
    path = usage_path()
    month = _month_key()
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if payload.get("month") == month:
            used = int(payload.get("used") or 0)
            cap = int(payload.get("cap") or ORATS_MONTHLY_CAP)
            return {
                "month": month,
                "used": used,
                "left": max(0, cap - used),
                "cap": cap,
            }
    return {"month": month, "used": 0, "left": ORATS_MONTHLY_CAP, "cap": ORATS_MONTHLY_CAP}


def save_usage(usage: Dict[str, Any]) -> None:
    path = usage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    cap = int(usage.get("cap") or ORATS_MONTHLY_CAP)
    used = int(usage.get("used") or 0)
    payload = {
        "month": usage.get("month") or _month_key(),
        "used": used,
        "left": max(0, cap - used),
        "cap": cap,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reset_process_http() -> None:
    global _process_http
    _process_http = 0


def process_http() -> int:
    return _process_http


def _throttle(max_per_min: int = ORATS_MAX_PER_MIN) -> None:
    global _last_request_mono
    min_interval = 60.0 / float(max_per_min or ORATS_MAX_PER_MIN)
    now = time.monotonic()
    wait = min_interval - (now - _last_request_mono)
    if _last_request_mono > 0 and wait > 0:
        time.sleep(wait)
    _last_request_mono = time.monotonic()


def _count_http() -> Dict[str, Any]:
    usage = load_usage()
    usage["used"] = int(usage.get("used") or 0) + 1
    usage["left"] = max(0, int(usage.get("cap") or ORATS_MONTHLY_CAP) - usage["used"])
    save_usage(usage)
    return usage


def can_http(max_requests: Optional[int]) -> bool:
    if max_requests is not None and _process_http >= int(max_requests):
        return False
    usage = load_usage()
    if int(usage.get("left") or 0) <= 0:
        return False
    return True


def http_get(
    path: str,
    query: Dict[str, str],
    token: str,
    timeout: float = 120.0,
) -> Tuple[int, Optional[Any], str]:
    global _process_http
    params = dict(query)
    params["token"] = token
    url = BASE + path + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    _throttle()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(resp.getcode() or 0)
            raw = resp.read() or b""
    except urllib.error.HTTPError as exc:
        status = int(exc.code or 0)
        try:
            raw = exc.read() or b""
        except Exception:
            raw = b""
        _process_http += 1
        _count_http()
        return status, None, "http_%s" % status
    except (urllib.error.URLError, TimeoutError, OSError):
        return 0, None, "url_error"
    _process_http += 1
    _count_http()
    if raw[:2] == b"\x1f\x8b":
        try:
            raw = gzip.decompress(raw)
        except OSError:
            return status, None, "bad_gzip"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return status, None, "bad_json"
    return status, payload, ""


def rows_of(payload: Any) -> list:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and payload.get("ticker"):
        return [payload]
    return []


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


def day_dir(asof: str) -> Path:
    return archive_dir() / asof


def cores_cache(asof: str) -> Path:
    return day_dir(asof) / "cores.json"


def strikes_cache(asof: str, ticker: str, dte: str) -> Path:
    tag = str(dte).replace(",", "-").replace(" ", "")
    return day_dir(asof) / ("strikes_%s_%s.json" % (ticker.upper(), tag))


def _chunks(items, size: int):
    seq = list(items)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _get(
    path: str,
    query: Dict[str, str],
    token: str,
    getter: Optional[GetFn],
    max_requests: Optional[int],
) -> Tuple[int, Optional[Any], str]:
    if not can_http(max_requests):
        return 0, None, "orats_budget"
    get = getter or http_get
    return get(path, query, token)


def cap_tickers(tickers: Sequence[str], limit: int) -> List[str]:
    out = []
    seen = set()
    for raw in tickers:
        name = str(raw or "").upper()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= int(limit):
            break
    return out


def fetch_cores(
    asof: str,
    tickers: Sequence[str],
    token: str,
    today: str,
    getter=None,
    max_requests: Optional[int] = None,
    max_tickers: int = 40,
    fields: str = CORE_FIELDS,
) -> Dict[str, Any]:
    wanted = cap_tickers(tickers, max_tickers)
    if not wanted:
        usage = load_usage()
        return {
            "ok": True,
            "rows": {},
            "http": 0,
            "error": "empty_shortlist",
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
            "cache": True,
            "capped": [],
        }
    cache = cores_cache(asof)
    cached = _read_json(cache)
    by_ticker = {}
    if cached is not None:
        for row in rows_of(cached):
            name = str(row.get("ticker") or "").upper()
            if name:
                by_ticker[name] = row
    missing = [t for t in wanted if t not in by_ticker]
    if not missing:
        usage = load_usage()
        return {
            "ok": True,
            "rows": {t: by_ticker[t] for t in wanted if t in by_ticker},
            "http": 0,
            "error": "",
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
            "cache": True,
            "capped": wanted,
        }
    hist = asof != today
    path = "/hist/cores" if hist else "/cores"
    planned = (len(missing) + ORATS_TICKER_BATCH - 1) // ORATS_TICKER_BATCH
    usage = load_usage()
    if planned and int(usage.get("left") or 0) < planned:
        return {
            "ok": False,
            "rows": {t: by_ticker[t] for t in wanted if t in by_ticker},
            "http": 0,
            "error": "orats_budget",
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
            "cache": False,
            "capped": wanted,
        }
    http_n = 0
    err = ""
    ok = True
    for group in _chunks(missing, ORATS_TICKER_BATCH):
        query = {"ticker": ",".join(group)}
        if fields:
            query["fields"] = fields
        if hist:
            query["tradeDate"] = asof
        status, payload, err = _get(path, query, token, getter, max_requests)
        http_n += 0 if err == "orats_budget" else 1
        if err == "orats_budget":
            ok = False
            break
        if status != 200 or payload is None:
            ok = False
            err = err or "http_%s" % status
            break
        for row in rows_of(payload):
            name = str(row.get("ticker") or "").upper()
            if name:
                by_ticker[name] = row
    if by_ticker:
        _write_json(cache, {"data": list(by_ticker.values())})
    usage = load_usage()
    return {
        "ok": ok and all(t in by_ticker for t in wanted),
        "rows": {t: by_ticker[t] for t in wanted if t in by_ticker},
        "http": http_n,
        "error": err,
        "used": usage.get("used") or 0,
        "left": usage.get("left") or 0,
        "cache": False,
        "capped": wanted,
    }


def fetch_strikes(
    asof: str,
    tickers: Sequence[str],
    token: str,
    today: str,
    getter=None,
    max_requests: Optional[int] = None,
    max_tickers: int = 20,
    dte: str = ORATS_STRIKE_DTE,
) -> Dict[str, Any]:
    wanted = cap_tickers(tickers, max_tickers)
    by_ticker = {}
    missing = []
    for name in wanted:
        cached = _read_json(strikes_cache(asof, name, dte))
        if cached is not None:
            by_ticker[name] = rows_of(cached)
        else:
            missing.append(name)
    if not missing:
        usage = load_usage()
        return {
            "ok": True,
            "rows": by_ticker,
            "http": 0,
            "error": "",
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
            "cache": True,
            "capped": wanted,
        }
    hist = asof != today
    path = "/hist/strikes" if hist else "/strikes"
    planned = (len(missing) + ORATS_TICKER_BATCH - 1) // ORATS_TICKER_BATCH
    usage = load_usage()
    if planned and int(usage.get("left") or 0) < planned:
        return {
            "ok": False,
            "rows": by_ticker,
            "http": 0,
            "error": "orats_budget",
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
            "cache": False,
            "capped": wanted,
        }
    http_n = 0
    err = ""
    ok = True
    for group in _chunks(missing, ORATS_TICKER_BATCH):
        query = {"ticker": ",".join(group), "dte": dte}
        if hist:
            query["tradeDate"] = asof
        status, payload, err = _get(path, query, token, getter, max_requests)
        http_n += 0 if err == "orats_budget" else 1
        if err == "orats_budget":
            ok = False
            break
        if status != 200 or payload is None:
            ok = False
            err = err or "http_%s" % status
            for name in group:
                by_ticker.setdefault(name, [])
            continue
        grouped = {}
        for row in rows_of(payload):
            name = str(row.get("ticker") or "").upper()
            grouped.setdefault(name, []).append(row)
        for name in group:
            rows = grouped.get(name) or []
            by_ticker[name] = rows
            _write_json(strikes_cache(asof, name, dte), {"data": rows})
    usage = load_usage()
    return {
        "ok": ok,
        "rows": by_ticker,
        "http": http_n,
        "error": err,
        "used": usage.get("used") or 0,
        "left": usage.get("left") or 0,
        "cache": False,
        "capped": wanted,
    }


def parse_core(row: Optional[dict]) -> Dict[str, Any]:
    row = row or {}
    iv30 = to_float(row.get("iv30d"))
    hv20 = to_float(row.get("orHv20d"))
    px = to_float(row.get("pxAtmIv"))
    if px is None:
        px = to_float(row.get("pxCls")) or to_float(row.get("stockPrice"))
    next_ern = str(row.get("nextErn") or "").strip()
    days = to_float(row.get("daysToNextErn"))
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "trade_date": str(row.get("tradeDate") or "")[:10],
        "px": px,
        "mkt_cap": to_float(row.get("mktCap")),
        "avg_opt_vol_20d": to_float(row.get("avgOptVolu20d")),
        "borrow30": to_float(row.get("borrow30")),
        "iv30": iv30,
        "iv30_dec": iv_decimal(iv30),
        "hv20": hv20,
        "iv_pctile_1y": to_float(row.get("ivPctile1y")),
        "iv_hv": to_float(row.get("ivHvXernRatio")),
        "next_ern": next_ern,
        "days_to_ern": None if days is None else int(days),
        "wks_next_ern": to_float(row.get("wksNextErn")),
        "div_yield": to_float(row.get("divYield")),
        "beta1y": to_float(row.get("beta1y")),
        "correl_spy_1y": to_float(row.get("correlSpy1y")),
        "c_vol": to_float(row.get("cVolu")),
        "p_vol": to_float(row.get("pVolu")),
        "asset_type": row.get("assetType"),
        "confidence": to_float(row.get("confidence")),
        "chg_1w": to_float(row.get("stkPxChng1wk")),
        "chg_1m": to_float(row.get("stkPxChng1m")),
        "chg_1y": to_float(row.get("stkPxChng1y")),
        "forecast_20d": to_float(row.get("orFcst20d")),
        "iv_forecast_20d": to_float(row.get("orIvFcst20d")),
        "sector": str(row.get("sectorName") or row.get("sector") or ""),
        "raw": bool(row),
    }
