"""ORATS Delayed Data API. Never log or write the token."""

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

from groat.config import (
    CODE_DIR,
    ORATS_BASE,
    ORATS_MAX_PER_MIN,
    ORATS_MONTHLY_CAP,
    ORATS_TICKER_BATCH,
    PROBE_TICKER,
)
from groat.num import to_float

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


def _empty_unusable(value: object) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    lowered = text.lower()
    if lowered in ("0000-00-00", "none", "null", "nan"):
        return True
    if lowered.startswith("0000"):
        return True
    return False


def _unit_for(value: object) -> str:
    number = to_float(value)
    if number is None or number == 0:
        return "unknown"
    if abs(number) <= 1.5:
        return "decimal"
    return "percent"


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
    vol_key = pick("stkVolu", "volume", "vol")
    return {
        "endpoint": "/datav2/hist/dailies",
        "ticker": str(row.get("ticker") or PROBE_TICKER),
        "keys": keys,
        "close": {"key": close_key},
        "high": {"key": high_key},
        "low": {"key": low_key},
        "open": {"key": open_key},
        "trade_date": {"key": date_key},
        "volume": {"key": vol_key},
        "sample_trade_date": str(row.get(date_key) or "")[:10] if date_key else "",
    }


def map_cores_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = sorted(str(k) for k in row.keys())
    iv_key = "iv30d" if "iv30d" in row else ""
    hv_key = ""
    if "orHv20d" in row:
        hv_key = "orHv20d"
    elif "clsHv20d" in row:
        hv_key = "clsHv20d"
    rank_key = ""
    if "ivPctile1y" in row:
        rank_key = "ivPctile1y"
    elif "ivRank1y" in row:
        rank_key = "ivRank1y"
    earn_key = "nextErn" if "nextErn" in row else ""
    earn_val = row.get(earn_key) if earn_key else ""
    fcst_key = "orFcst20d" if "orFcst20d" in row else ""
    return {
        "endpoint": "/datav2/cores",
        "ticker": str(row.get("ticker") or PROBE_TICKER),
        "keys": keys,
        "iv30": {"key": iv_key, "unit": _unit_for(row.get(iv_key)) if iv_key else "unknown"},
        "hv20": {"key": hv_key, "unit": _unit_for(row.get(hv_key)) if hv_key else "unknown"},
        "iv_rank": {"key": rank_key, "unit": "raw"},
        "forecast": {"key": fcst_key, "unit": _unit_for(row.get(fcst_key)) if fcst_key else "unknown"},
        "next_earnings": {
            "key": earn_key,
            "usable": bool(earn_key) and not _empty_unusable(earn_val),
            "sample": str(earn_val).strip() if earn_key else "",
        },
    }


def write_field_map(mapped: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Path:
    body = dict(mapped)
    if extra:
        body.update(extra)
    path = field_map_path()
    _write_json(path, body)
    return path


def map_line(mapped: Dict[str, Any]) -> str:
    if mapped.get("iv30"):
        iv = mapped.get("iv30") or {}
        hv = mapped.get("hv20") or {}
        rank = mapped.get("iv_rank") or {}
        earn = mapped.get("next_earnings") or {}
        return "iv30=%s:%s hv20=%s:%s iv_rank=%s next_earnings=%s" % (
            iv.get("key") or "",
            iv.get("unit") or "",
            hv.get("key") or "",
            hv.get("unit") or "",
            rank.get("key") or "",
            earn.get("key") or "",
        )
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


def day_dir(asof: str) -> Path:
    return archive_dir() / asof


def cores_cache(asof: str) -> Path:
    return day_dir(asof) / "cores.json"


def strikes_cache(asof: str, ticker: str, dte: str) -> Path:
    tag = str(dte).replace(",", "-").replace(" ", "")
    return day_dir(asof) / ("strikes_%s_%s.json" % (ticker.upper(), tag))


def dailies_cache(ticker: str) -> Path:
    return archive_dir() / "dailies" / ("%s.json" % str(ticker).upper())


def load_dailies_payload(ticker: str) -> Optional[Any]:
    return _read_json(dailies_cache(ticker))


def write_dailies_payload(ticker: str, payload: Any) -> Path:
    path = dailies_cache(ticker)
    _write_json(path, payload if isinstance(payload, dict) else {"data": payload})
    return path


def _chunks(items, size: int):
    seq = list(items)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _get(
    path: str,
    query: Dict[str, str],
    token: str,
    getter: Optional[Callable[..., Tuple[int, Optional[Any], str]]],
    max_requests: Optional[int],
) -> Tuple[int, Optional[Any], str]:
    if not can_http(max_requests):
        return 0, None, "orats_budget"
    get = getter or http_get
    return get(path, query, token)


def fetch_cores(
    asof: str,
    tickers: Sequence[str],
    token: str,
    today: str,
    getter=None,
    max_requests: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    wanted = [str(t).upper() for t in tickers if t]
    cache = cores_cache(asof)
    by_ticker = {}
    if not refresh:
        cached = _read_json(cache)
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
            "rows": by_ticker,
            "http": 0,
            "error": "",
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
            "cache": True,
        }
    hist = asof != today
    path = "/hist/cores" if hist else "/cores"
    planned = (len(missing) + ORATS_TICKER_BATCH - 1) // ORATS_TICKER_BATCH if missing else 0
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
        }
    http_n = 0
    err = ""
    ok = True
    for group in _chunks(missing, ORATS_TICKER_BATCH):
        query = {"ticker": ",".join(group)}
        if hist:
            query["tradeDate"] = asof
        status, payload, err = _get(path, query, token, getter, max_requests)
        http_n += 0 if err == "orats_budget" else 1
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
        sample = None
        for name in wanted:
            if name in by_ticker:
                sample = by_ticker[name]
                break
        if sample:
            write_field_map(map_cores_row(sample), extra={"http": 200, "orats_ok": 1})
    usage = load_usage()
    return {
        "ok": ok and all(t in by_ticker for t in wanted),
        "rows": by_ticker,
        "http": http_n,
        "error": err,
        "used": usage.get("used") or 0,
        "left": usage.get("left") or 0,
        "cache": False,
    }


def fetch_strikes(
    asof: str,
    tickers: Sequence[str],
    token: str,
    today: str,
    getter=None,
    max_requests: Optional[int] = None,
    dte: str = "21,30,45,60",
    refresh: bool = False,
) -> Dict[str, Any]:
    wanted = [str(t).upper() for t in tickers if t]
    by_ticker = {}
    missing = []
    for name in wanted:
        cached = None if refresh else _read_json(strikes_cache(asof, name, dte))
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
    }


def fetch_dailies_series(
    ticker: str,
    token: str,
    getter=None,
    max_requests: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    name = str(ticker).upper()
    cached = load_dailies_payload(name)
    if cached is not None and rows_of(cached) and not refresh:
        usage = load_usage()
        return {
            "ok": True,
            "payload": cached,
            "rows": rows_of(cached),
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
    rows = rows_of(payload) if status == 200 else []
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


def earnings_cache(ticker: str) -> Path:
    return archive_dir() / "earnings" / ("%s.json" % str(ticker).upper())


def fetch_hist_earnings(
    ticker: str,
    token: str,
    getter=None,
    max_requests: Optional[int] = None,
    refresh: bool = False,
) -> Dict[str, Any]:
    """GET /datav2/hist/earnings. Schema: ticker, earnDate, anncTod, updatedAt."""
    name = str(ticker).upper()
    cache = earnings_cache(name)
    cached = _read_json(cache)
    if cached is not None and rows_of(cached) and not refresh:
        usage = load_usage()
        return {
            "ok": True,
            "rows": rows_of(cached),
            "http": 0,
            "error": "",
            "cache": True,
            "used": usage.get("used") or 0,
            "left": usage.get("left") or 0,
        }
    status, payload, err = _get("/hist/earnings", {"ticker": name}, token, getter, max_requests)
    usage = load_usage()
    rows = rows_of(payload) if status == 200 else []
    if status == 200 and payload is not None:
        _write_json(cache, payload if isinstance(payload, dict) else {"data": rows})
    return {
        "ok": status == 200 and bool(rows),
        "rows": rows,
        "http": 0 if err == "orats_budget" else 1,
        "error": err or ("" if status == 200 else "http_%s" % status),
        "cache": False,
        "used": usage.get("used") or 0,
        "left": usage.get("left") or 0,
    }


def parse_core(row: Optional[dict]) -> Dict[str, Any]:
    """Extract ORATS vol fields. Missing keys stay None → DATA UNAVAILABLE. Never invent."""
    row = row or {}
    iv30 = to_float(row.get("iv30d"))
    hv20 = to_float(row.get("orHv20d"))
    if hv20 is None:
        hv20 = to_float(row.get("clsHv20d"))
    pctile = to_float(row.get("ivPctile1y"))
    rank = to_float(row.get("ivRank1y"))
    fcst = to_float(row.get("orFcst20d"))
    iv_fcst = to_float(row.get("orIvFcst20d"))
    ex_iv = to_float(row.get("exErnIv30d"))
    ex_hv = to_float(row.get("orHvXern20d"))
    slope = to_float(row.get("slope"))
    conf = to_float(row.get("confidence"))
    imp_move = to_float(row.get("impErnMv"))
    if imp_move is None:
        imp_move = to_float(row.get("impliedEarningsMove"))
    contango = to_float(row.get("contango"))
    dlt25 = to_float(row.get("dlt25Iv30d"))
    dlt75 = to_float(row.get("dlt75Iv30d"))
    vrp = None
    if iv30 is not None and hv20 is not None:
        vrp = iv30 - hv20
    iv_vs_fcst = None
    if iv30 is not None and fcst is not None:
        iv_vs_fcst = iv30 - fcst
    return {
        "ticker": str(row.get("ticker") or "").upper(),
        "trade_date": str(row.get("tradeDate") or "")[:10],
        "iv30": iv30,
        "hv20": hv20,
        "vrp": vrp,
        "iv_pctile_1y": pctile,
        "iv_rank_1y": rank,
        "forecast_20d": fcst,
        "iv_forecast_20d": iv_fcst,
        "iv_vs_forecast": iv_vs_fcst,
        "ex_ern_iv30": ex_iv,
        "ex_ern_hv20": ex_hv,
        "slope": slope,
        "dlt25_iv30": dlt25,
        "dlt75_iv30": dlt75,
        "contango": contango,
        "confidence": conf,
        "imp_ern_mv": imp_move,
        "c_oi": to_float(row.get("cOi")),
        "p_oi": to_float(row.get("pOi")),
        "c_vol": to_float(row.get("cVolu")),
        "p_vol": to_float(row.get("pVolu")),
        "oi": to_float(row.get("oi")),
        "avg_opt_vol_20d": to_float(row.get("avgOptVolu20d")),
        "mkt_width_vol": to_float(row.get("mktWidthVol")),
        "px": to_float(row.get("pxCls")) or to_float(row.get("stockPrice")),
        "mkt_cap": to_float(row.get("mktCap")),
        "sector": str(row.get("sector") or row.get("sectorName") or ""),
        "best_etf": str(row.get("bestEtf") or ""),
        "stk_volu": to_float(row.get("stkVolu")),
        "div_date": str(row.get("divDate") or ""),
        "raw": bool(row),
    }
