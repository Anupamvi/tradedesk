"""ORATS delayed Data API v2. Never log the token."""

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

from xhigh.config import (
    CODE_DIR,
    CORE_FIELDS,
    ORATS_BASE,
    ORATS_MAX_PER_MIN,
    ORATS_MONTHLY_CAP,
    ORATS_TICKER_BATCH,
)
from xhigh.num import to_float

_TOKEN_QUERY = re.compile(r"(token=)[^&]+", re.I)
_last_request_mono = 0.0
_process_http = 0


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
            return {"month": month, "used": used, "left": max(0, cap - used), "cap": cap}
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


def http_get(path: str, query: Dict[str, str], token: str, timeout: float = 120.0) -> Tuple[int, Optional[Any], str]:
    global _process_http
    params = dict(query)
    params["token"] = token
    url = ORATS_BASE + path + "?" + urllib.parse.urlencode(params)
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


def _rows(payload: Any) -> List[dict]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def fetch_cores(
    tickers: Sequence[str],
    token: str,
    max_requests: Optional[int] = None,
    fields: str = CORE_FIELDS,
) -> Dict[str, dict]:
    names = [str(t).upper() for t in tickers if t]
    out: Dict[str, dict] = {}
    for i in range(0, len(names), ORATS_TICKER_BATCH):
        if not can_http(max_requests):
            break
        group = names[i : i + ORATS_TICKER_BATCH]
        status, payload, err = http_get(
            "/cores",
            {"ticker": ",".join(group), "fields": fields},
            token,
        )
        if err or status != 200:
            continue
        for row in _rows(payload):
            name = str(row.get("ticker") or "").upper()
            if name:
                out[name] = row
    return out


def parse_core(row: Optional[dict]) -> Dict[str, Any]:
    row = row or {}
    iv_hv = to_float(row.get("ivHvXernRatio"))
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "trade_date": str(row.get("tradeDate") or "")[:10],
        "mkt_cap": to_float(row.get("mktCap")),
        "avg_opt_vol_20d": to_float(row.get("avgOptVolu20d")),
        "iv30": to_float(row.get("iv30d")),
        "iv_pctile_1y": to_float(row.get("ivPctile1y")),
        "iv_rank_1y": to_float(row.get("ivRank1y")),
        "iv_hv": iv_hv,
        "next_ern": str(row.get("nextErn") or ""),
        "wks_next_ern": to_float(row.get("wksNextErn")),
        "last_ern": str(row.get("lastErn") or ""),
        "tk_over": row.get("tkOver"),
        "raw": row,
    }
