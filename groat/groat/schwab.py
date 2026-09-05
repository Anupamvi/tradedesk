"""Schwab tape: history, quotes, chains, positions. No /orders."""

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

from groat.config import CODE_DIR
from groat.dates import today_et, session_phase
from groat.envload import schwab_credentials
from groat.num import to_float

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

MARKET = "https://api.schwabapi.com/marketdata/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
TRADER = "https://api.schwabapi.com/trader/v1"


def schwab_cache_dir() -> Path:
    return CODE_DIR / "var" / "schwab_bars"


def use_live_schwab(date: str, live_flag: bool = False, no_schwab: bool = False, today: str = "") -> bool:
    if no_schwab:
        return False
    if live_flag:
        return True
    return date == (today or today_et())


def live_note(date: str, live: bool) -> str:
    if not live:
        return ""
    if not schwab_credentials():
        return "missing_schwab"
    return ""


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
    req = urllib.request.Request(url, headers={"Authorization": "Bearer %s" % token, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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


def _read_bars_cache(ticker: str) -> List[dict]:
    path = schwab_cache_dir() / ("%s.json" % ticker.upper())
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    rows = payload.get("data") if isinstance(payload, dict) else payload
    out = []
    for row in rows or []:
        if isinstance(row, dict) and row.get("date"):
            out.append(row)
    return out


def price_history_bars(ticker: str, asof: str, lookback_days: int = 420, use_cache: bool = True) -> List[dict]:
    name = str(ticker).upper()
    cached = _read_bars_cache(name) if use_cache else []
    if cached:
        last = cached[-1]["date"]
        if last >= asof[:10]:
            return [b for b in cached if b["date"] <= asof[:10]]
    token = _access_token()
    if not token:
        return [b for b in cached if b["date"] <= asof[:10]]
    try:
        asof_d = datetime.strptime(asof[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return cached
    end = asof_d + timedelta(days=1)
    start = end - timedelta(days=int(lookback_days))
    start_ms = int(start.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(end.replace(tzinfo=timezone.utc).timestamp() * 1000)
    query = urllib.parse.urlencode(
        {
            "symbol": ticker,
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
        return [b for b in cached if b["date"] <= asof[:10]]
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
    return [b for b in cached if b["date"] <= asof[:10]]


def quote_bar(ticker: str, asof: str) -> Optional[dict]:
    if asof != today_et():
        return None
    if session_phase(asof, asof) != "rth":
        return None
    token = _access_token()
    if not token:
        return None
    payload = _get_json("%s/quotes?symbols=%s" % (MARKET, urllib.parse.quote(ticker)), token)
    if not isinstance(payload, dict):
        return None
    wrap = payload.get(ticker) or payload.get(str(ticker).upper())
    if not isinstance(wrap, dict):
        for value in payload.values():
            if isinstance(value, dict) and (isinstance(value.get("quote"), dict) or "lastPrice" in value):
                wrap = value
                break
    if not isinstance(wrap, dict):
        return None
    q = wrap.get("quote") if isinstance(wrap.get("quote"), dict) else wrap

    def num(*names):
        for name in names:
            val = to_float(q.get(name))
            if val is not None:
                return val
        return None

    last = num("regularMarketLastPrice", "lastPrice", "mark")
    if last is None:
        last = num("closePrice")
    if last is None:
        return None
    high = num("highPrice", "regularMarketDayHigh") or last
    low = num("lowPrice", "regularMarketDayLow") or last
    open_ = num("openPrice", "regularMarketOpen") or last
    vol = num("totalVolume", "volume")
    return {"date": asof, "open": open_, "high": high, "low": low, "close": last, "volume": vol}


def quotes_many(tickers, asof: str) -> Dict[str, dict]:
    if asof != today_et():
        return {}
    if session_phase(asof, asof) != "rth":
        return {}
    token = _access_token()
    if not token:
        return {}
    names = [str(t) for t in tickers if t]
    out = {}
    for i in range(0, len(names), 20):
        group = names[i : i + 20]
        joined = ",".join(urllib.parse.quote(n, safe="") for n in group)
        payload = _get_json("%s/quotes?symbols=%s" % (MARKET, joined), token)
        time.sleep(0.05)
        if not isinstance(payload, dict):
            continue
        for key, wrap in payload.items():
            if not isinstance(wrap, dict):
                continue
            q = wrap.get("quote") if isinstance(wrap.get("quote"), dict) else wrap
            last = (
                to_float(q.get("regularMarketLastPrice"))
                or to_float(q.get("lastPrice"))
                or to_float(q.get("mark"))
                or to_float(q.get("closePrice"))
            )
            if last is None:
                continue
            out[str(key).upper()] = {
                "date": asof,
                "open": to_float(q.get("openPrice")) or last,
                "high": to_float(q.get("highPrice")) or last,
                "low": to_float(q.get("lowPrice")) or last,
                "close": last,
                "volume": to_float(q.get("totalVolume")),
            }
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
            "range": "ALL",
            "fromDate": from_date,
            "toDate": to_date,
        }
    )
    return _get_json("%s/chains?%s" % (MARKET, query), token, timeout=60.0)


def load_positions() -> tuple:
    """Return (rows, error). Empty rows with no error means a real empty book. Failed fetch is an error."""
    from groat.book import underlying_symbol

    if not schwab_credentials():
        return [], ""
    token = _access_token()
    if not token:
        return [], "schwab_positions: token unusable"
    payload = _get_json("%s/accounts?fields=positions" % TRADER, token, timeout=45.0)
    if payload is None:
        return [], "schwab_positions: DATA UNAVAILABLE"
    rows = []
    accounts = payload if isinstance(payload, list) else []
    if isinstance(payload, dict):
        accounts = payload.get("accounts") or [payload]
    for acct in accounts:
        if not isinstance(acct, dict):
            continue
        sec = acct.get("securitiesAccount") or acct
        for pos in sec.get("positions") or []:
            if not isinstance(pos, dict):
                continue
            inst = pos.get("instrument") or {}
            raw = str(inst.get("symbol") or "").upper()
            asset = str(inst.get("assetType") or "")
            under = str(inst.get("underlyingSymbol") or "").upper()
            if asset.upper() == "OPTION" and under:
                ticker = under
            else:
                ticker = underlying_symbol(raw) or under or raw
            rows.append(
                {
                    "ticker": ticker,
                    "symbol": raw,
                    "asset": asset,
                    "quantity": pos.get("longQuantity") or pos.get("shortQuantity"),
                    "average_price": pos.get("averagePrice"),
                    "market_value": pos.get("marketValue"),
                }
            )
    return rows, ""


def positions_all() -> List[dict]:
    rows, _err = load_positions()
    return rows
