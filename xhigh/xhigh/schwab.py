"""Schwab tape: quotes, history, chains, movers. Market Data GET only. Never place trades."""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from xhigh.config import CODE_DIR, MOVER_INDEXES
from xhigh.dates import today_et
from xhigh.envload import schwab_credentials
from xhigh.num import to_float

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

MARKET = "https://api.schwabapi.com/marketdata/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

_http = 0


def reset_http() -> None:
    global _http
    _http = 0


def http_count() -> int:
    return _http


def schwab_cache_dir() -> Path:
    return CODE_DIR / "var" / "schwab_bars"


def use_live_schwab(date: str, live_flag: bool = False, no_schwab: bool = False, today: str = "") -> bool:
    if no_schwab:
        return False
    if live_flag:
        return True
    return date == (today or today_et())


def _token_blob(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("token"), dict):
        return payload
    return {"token": payload if isinstance(payload, dict) else {}}


def _save_token(path: Path, blob: dict) -> None:
    path.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")


def _access_token() -> Optional[str]:
    creds = schwab_credentials()
    if not creds:
        return None
    path = Path(creds["token_path"])
    if not path.is_file():
        return None
    blob = _token_blob(path)
    inner = blob.get("token") or {}
    access = (inner.get("access_token") or "").strip()
    refresh = (inner.get("refresh_token") or "").strip()
    if not access and not refresh:
        return None
    auth = base64.b64encode(("%s:%s" % (creds["api_key"], creds["app_secret"])).encode("utf-8")).decode("ascii")
    if refresh:
        body = urllib.parse.urlencode({"grant_type": "refresh_token", "refresh_token": refresh}).encode("utf-8")
        req = urllib.request.Request(
            TOKEN_URL,
            data=body,
            headers={
                "Authorization": "Basic %s" % auth,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                fresh = json.loads(resp.read().decode("utf-8"))
            if isinstance(fresh, dict) and fresh.get("access_token"):
                inner.update(fresh)
                blob["token"] = inner
                _save_token(path, blob)
                return str(fresh.get("access_token") or "")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
            pass
    return access or None


def _get_json(url: str, token: str, timeout: float = 45.0) -> Optional[Any]:
    global _http
    req = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer %s" % token, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _http += 1
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
        return None


def _et_day(ms: float) -> str:
    stamp = datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc)
    if ZoneInfo is not None:
        stamp = stamp.astimezone(ZoneInfo("America/New_York"))
    return stamp.date().isoformat()


def _write_bars_cache(ticker: str, bars: List[dict]) -> None:
    path = schwab_cache_dir() / ("%s.json" % ticker.upper())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"data": bars}, indent=2) + "\n", encoding="utf-8")


def price_history_bars(ticker: str, asof: str, lookback_days: int = 420, use_cache: bool = True) -> List[dict]:
    name = str(ticker).upper()
    cached = []
    cache_path = schwab_cache_dir() / ("%s.json" % name)
    if use_cache and cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = [b for b in (payload.get("data") or []) if isinstance(b, dict) and b.get("date")]
        except (OSError, ValueError):
            cached = []
    token = _access_token()
    if not token:
        return [b for b in cached if str(b.get("date") or "") <= asof[:10]]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=int(lookback_days))
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    query = urllib.parse.urlencode(
        {
            "symbol": name,
            "periodType": "year",
            "frequencyType": "daily",
            "frequency": "1",
            "startDate": str(start_ms),
            "endDate": str(end_ms),
            "needExtendedHoursData": "false",
        }
    )
    payload = _get_json("%s/pricehistory?%s" % (MARKET, query), token)
    time.sleep(0.05)
    if not isinstance(payload, dict):
        return [b for b in cached if str(b.get("date") or "") <= asof[:10]]
    bars = []
    for candle in payload.get("candles") or []:
        if not isinstance(candle, dict):
            continue
        ms = candle.get("datetime")
        if ms is None:
            continue
        try:
            day = _et_day(float(ms))
            o = float(candle.get("open"))
            h = float(candle.get("high"))
            low = float(candle.get("low"))
            c = float(candle.get("close"))
        except (TypeError, ValueError):
            continue
        vol = to_float(candle.get("volume"))
        bars.append({"date": day, "open": o, "high": h, "low": low, "close": c, "volume": vol})
    bars.sort(key=lambda b: b["date"])
    if bars:
        _write_bars_cache(name, bars)
        return [b for b in bars if b["date"] <= asof[:10]]
    return [b for b in cached if str(b.get("date") or "") <= asof[:10]]


def _parse_quote(wrap: dict, asof: str) -> Optional[dict]:
    if not isinstance(wrap, dict):
        return None
    q = wrap.get("quote") if isinstance(wrap.get("quote"), dict) else wrap
    last = to_float(q.get("lastPrice"))
    close = to_float(q.get("closePrice"))
    if last is None or last <= 0:
        last = close
    if last is None or last <= 0:
        return None
    bid = to_float(q.get("bidPrice")) or to_float(q.get("bid"))
    ask = to_float(q.get("askPrice")) or to_float(q.get("ask"))
    return {
        "date": asof,
        "close": close if close and close > 0 else last,
        "last": last,
        "bid": bid,
        "ask": ask,
        "volume": to_float(q.get("totalVolume")),
        "asset": str((wrap.get("assetMainType") or wrap.get("assetType") or "")),
    }


def quotes_many(tickers, asof: str) -> Dict[str, dict]:
    token = _access_token()
    if not token:
        return {}
    names = [str(t).upper() for t in tickers if t]
    out = {}
    for i in range(0, len(names), 20):
        group = names[i : i + 20]
        joined = ",".join(urllib.parse.quote(n, safe="") for n in group)
        payload = _get_json("%s/quotes?symbols=%s" % (MARKET, joined), token)
        time.sleep(0.05)
        if not isinstance(payload, dict):
            continue
        for key, wrap in payload.items():
            parsed = _parse_quote(wrap, asof) if isinstance(wrap, dict) else None
            if parsed:
                out[str(key).upper()] = parsed
    return out


def option_chain(ticker: str, from_date: str, to_date: str) -> Optional[dict]:
    token = _access_token()
    if not token:
        return None
    query = urllib.parse.urlencode(
        {
            "symbol": ticker,
            "contractType": "ALL",
            "includeUnderlyingQuote": "true",
            "strategy": "SINGLE",
            "range": "NTM",
            "strikeCount": 30,
            "fromDate": from_date,
            "toDate": to_date,
        }
    )
    return _get_json("%s/chains?%s" % (MARKET, query), token, timeout=60.0)


def flatten_chain(payload: Optional[dict]) -> List[dict]:
    if not isinstance(payload, dict):
        return []
    rows = []
    for key, side in (("callExpDateMap", "call"), ("putExpDateMap", "put")):
        block = payload.get(key) or {}
        if not isinstance(block, dict):
            continue
        for exp_key, strikes in block.items():
            expiry = str(exp_key).split(":")[0][:10]
            if not isinstance(strikes, dict):
                continue
            for strike_s, contracts in strikes.items():
                if not isinstance(contracts, list):
                    continue
                for c in contracts:
                    if not isinstance(c, dict):
                        continue
                    rows.append(
                        {
                            "side": side,
                            "expiry": expiry,
                            "strike": to_float(c.get("strikePrice")) or to_float(strike_s),
                            "bid": to_float(c.get("bid")),
                            "ask": to_float(c.get("ask")),
                            "delta": to_float(c.get("delta")),
                            "dte": to_float(c.get("daysToExpiration")),
                            "oi": to_float(c.get("openInterest")),
                        }
                    )
    return rows


def movers_symbols(indexes=MOVER_INDEXES) -> List[str]:
    token = _access_token()
    if not token:
        return []
    names = []
    seen = set()
    for idx in indexes:
        encoded = urllib.parse.quote(idx, safe="")
        payload = _get_json("%s/movers/%s" % (MARKET, encoded), token)
        time.sleep(0.05)
        rows = []
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            for key in ("screeners", "movers", "records"):
                if isinstance(payload.get(key), list):
                    rows = payload.get(key) or []
                    break
            if not rows:
                inner = payload.get(idx) or payload.get(idx.replace("$", ""))
                if isinstance(inner, list):
                    rows = inner
        for row in rows:
            if not isinstance(row, dict):
                continue
            sym = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
            if not sym or sym.startswith("$") or "/" in sym or " " in sym or "." in sym:
                continue
            if len(sym) > 5:
                continue
            if not sym.isalnum():
                continue
            if sym in seen:
                continue
            seen.add(sym)
            names.append(sym)
    return names
