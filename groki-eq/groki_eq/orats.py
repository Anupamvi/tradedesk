"""ORATS Delayed hist/dailies probe. Never log or write the token."""

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

from groki_eq.config import (
    CODE_DIR,
    ORATS_BASE,
    ORATS_MAX_PER_MIN,
    ORATS_MONTHLY_CAP,
    PROBE_PATH,
    PROBE_TICKER,
)

BASE = ORATS_BASE
_TOKEN_QUERY = re.compile(r"(token=)[^&]+", re.I)
_last_request_mono = 0.0
_process_http = 0


def archive_dir() -> Path:
    return CODE_DIR / "var" / "orats_archive"


def field_map_path() -> Path:
    return archive_dir() / "field_map.json"


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


def _throttle() -> None:
    global _last_request_mono
    min_interval = 60.0 / float(ORATS_MAX_PER_MIN)
    now = time.monotonic()
    wait = min_interval - (now - _last_request_mono)
    if _last_request_mono > 0 and wait > 0:
        time.sleep(wait)
    _last_request_mono = time.monotonic()


def _urlopen(req: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(req, timeout=timeout)


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
        with _urlopen(req, timeout=timeout) as resp:
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


def _first_row(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data:
            row = data[0]
            if isinstance(row, dict):
                return row
        if "ticker" in payload:
            return payload
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        return payload[0]
    return None


def _row_count(payload: Any) -> int:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return len(payload.get("data") or [])
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and payload.get("ticker"):
        return 1
    return 0


def map_dailies_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = sorted(str(k) for k in row.keys())

    def pick(*names: str) -> str:
        for name in names:
            if name in row:
                return name
        return ""

    close_key = pick("clsPx", "close", "stockPrice")
    high_key = pick("hiPx", "high")
    low_key = pick("loPx", "low")
    open_key = pick("open", "openPx")
    date_key = pick("tradeDate", "date")
    return {
        "endpoint": "/datav2/hist/dailies",
        "ticker": str(row.get("ticker") or PROBE_TICKER),
        "keys": keys,
        "close": {"key": close_key},
        "high": {"key": high_key},
        "low": {"key": low_key},
        "open": {"key": open_key},
        "trade_date": {"key": date_key},
        "sample_trade_date": str(row.get(date_key) or "")[:10] if date_key else "",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_field_map(mapped: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Path:
    body = dict(mapped)
    if extra:
        body.update(extra)
    path = field_map_path()
    _write_json(path, body)
    return path


def probe_spy_dailies(
    token: str,
    max_requests: Optional[int] = None,
    getter: Optional[Callable[..., Tuple[int, Optional[Any], str]]] = None,
) -> Dict[str, Any]:
    """One GET /datav2/hist/dailies?ticker=SPY. Cache payload. Write field_map.json."""
    usage = load_usage()
    if not can_http(max_requests):
        return {
            "ok": False,
            "http": 0,
            "rows": 0,
            "keys": [],
            "map": {},
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
            "error": "orats_budget",
        }
    get = getter or http_get
    status, payload, err = get(PROBE_PATH, {"ticker": PROBE_TICKER}, token)
    usage = load_usage()
    row = _first_row(payload)
    rows = _row_count(payload)
    mapped = map_dailies_row(row) if row else {
        "endpoint": "/datav2/hist/dailies",
        "ticker": PROBE_TICKER,
        "keys": [],
        "close": {"key": ""},
        "high": {"key": ""},
        "low": {"key": ""},
        "open": {"key": ""},
        "trade_date": {"key": ""},
        "sample_trade_date": "",
    }
    ok = status == 200 and row is not None
    probe_dir = archive_dir() / "probe"
    if payload is not None:
        _write_json(probe_dir / "dailies_SPY.json", payload)
        dailies = archive_dir() / "dailies"
        _write_json(dailies / "SPY.json", payload if isinstance(payload, dict) else {"data": payload})
    write_field_map(
        mapped,
        extra={
            "http": status,
            "rows": rows,
            "orats_ok": 1 if ok else 0,
        },
    )
    return {
        "ok": ok,
        "http": status,
        "rows": rows,
        "keys": mapped.get("keys") or [],
        "map": mapped,
        "used": usage.get("used") or 0,
        "left": usage.get("left") or 0,
        "error": err,
    }


def map_line(mapped: Dict[str, Any]) -> str:
    close = mapped.get("close") or {}
    high = mapped.get("high") or {}
    low = mapped.get("low") or {}
    open_ = mapped.get("open") or {}
    return "close=%s high=%s low=%s open=%s" % (
        close.get("key") or "",
        high.get("key") or "",
        low.get("key") or "",
        open_.get("key") or "",
    )


def public_orats_info(probe: Dict[str, Any]) -> Dict[str, Any]:
    mapped = probe.get("map") or {}
    return {
        "orats_ok": 1 if probe.get("ok") else 0,
        "orats_http": probe.get("http") or 0,
        "orats_rows": probe.get("rows") or 0,
        "orats_requests_used": probe.get("used") or 0,
        "orats_requests_left": probe.get("left") or 0,
        "orats_map": {
            "close": (mapped.get("close") or {}).get("key") or "",
            "high": (mapped.get("high") or {}).get("key") or "",
            "low": (mapped.get("low") or {}).get("key") or "",
            "open": (mapped.get("open") or {}).get("key") or "",
            "trade_date": (mapped.get("trade_date") or {}).get("key") or "",
        },
        "orats_fields": list(mapped.get("keys") or []),
    }


def _rows(payload: Any) -> list:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [row for row in payload["data"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def _read_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def dailies_cache(ticker: str) -> Path:
    return archive_dir() / "dailies" / ("%s.json" % str(ticker).upper())


def load_dailies_payload(ticker: str) -> Optional[Any]:
    return _read_json(dailies_cache(ticker))


def write_dailies_payload(ticker: str, payload: Any) -> Path:
    path = dailies_cache(ticker)
    _write_json(path, payload if isinstance(payload, dict) else {"data": payload})
    return path


def fetch_dailies_series(
    ticker: str,
    token: str,
    getter=None,
    max_requests: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    name = str(ticker).upper()
    cached = load_dailies_payload(name)
    if cached is not None and _rows(cached) and not refresh:
        usage = load_usage()
        return {
            "ok": True,
            "payload": cached,
            "rows": _rows(cached),
            "http": 0,
            "error": "",
            "cache": True,
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
        }
    if not can_http(max_requests):
        usage = load_usage()
        return {
            "ok": False,
            "payload": None,
            "rows": [],
            "http": 0,
            "error": "orats_budget",
            "cache": False,
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
        }
    get = getter or http_get
    status, payload, err = get("/hist/dailies", {"ticker": name}, token)
    usage = load_usage()
    rows = _rows(payload) if status == 200 else []
    if status == 200 and payload is not None:
        write_dailies_payload(name, payload if isinstance(payload, dict) else {"data": rows})
    return {
        "ok": status == 200 and bool(rows),
        "payload": payload,
        "rows": rows,
        "http": 0 if err == "orats_budget" else 1,
        "error": err or ("" if status == 200 else "http_%s" % status),
        "cache": False,
        "used": usage.get("used") or 0,
        "left": usage.get("left") or 0,
    }
