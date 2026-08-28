"""Schwab tape: price history + quotes. Positions for manage. No /orders."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from groki_eq.dates import today_et
from groki_eq.envload import load_schwab_env, schwab_credentials

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None

MARKET = "https://api.schwabapi.com/marketdata/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"
TRADER = "https://api.schwabapi.com/trader/v1"


def use_live_schwab(date: str, live_flag: bool = False, no_schwab: bool = False, today: str = "") -> bool:
    if no_schwab:
        return False
    if live_flag:
        return True
    return date == (today or today_et())


def live_note(date: str, live: bool) -> str:
    if not live:
        return ""
    creds = schwab_credentials()
    if not creds:
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


def _get_json(url: str, token: str, timeout: float = 30.0) -> Optional[Any]:
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


def price_history_bars(ticker: str, asof: str) -> List[dict]:
    token = _access_token()
    if not token:
        return []
    try:
        asof_d = datetime.strptime(asof[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return []
    end = asof_d + timedelta(days=1)
    start = end - timedelta(days=21)
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
    if not isinstance(payload, dict):
        return []
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
        bars.append({"date": day, "open": o, "high": h, "low": low, "close": c})
    bars.sort(key=lambda b: b["date"])
    return bars


def quote_bar(ticker: str, asof: str) -> Optional[dict]:
    if asof != today_et():
        return None
    token = _access_token()
    if not token:
        return None
    payload = _get_json("%s/quotes?symbols=%s" % (MARKET, urllib.parse.quote(ticker)), token)
    if not isinstance(payload, dict):
        return None
    wrap = payload.get(ticker) or payload.get(ticker.upper())
    if not isinstance(wrap, dict):
        # sometimes keyed differently
        for value in payload.values():
            if isinstance(value, dict) and isinstance(value.get("quote"), dict):
                wrap = value
                break
    if not isinstance(wrap, dict):
        return None
    q = wrap.get("quote") if isinstance(wrap.get("quote"), dict) else wrap
    def num(*names):
        for name in names:
            try:
                val = float(q.get(name))
            except (TypeError, ValueError):
                continue
            if val == val:
                return val
        return None
    last = num("lastPrice", "regularMarketLastPrice", "mark", "closePrice")
    if last is None:
        return None
    high = num("highPrice", "regularMarketDayHigh") or last
    low = num("lowPrice", "regularMarketDayLow") or last
    open_ = num("openPrice", "regularMarketOpen") or last
    return {"date": asof, "open": open_, "high": high, "low": low, "close": last}


def positions_universe(tickers) -> List[dict]:
    token = _access_token()
    if not token:
        return []
    payload = _get_json("%s/accounts?fields=positions" % TRADER, token, timeout=45.0)
    wanted = {str(t).upper() for t in tickers}
    rows = []
    accounts = payload if isinstance(payload, list) else []
    if isinstance(payload, dict):
        accounts = payload.get("accounts") or [payload]
    for acct in accounts:
        if not isinstance(acct, dict):
            continue
        sec = acct.get("securitiesAccount") or acct
        for pos in sec.get("positions") or []:
            inst = pos.get("instrument") or {}
            asset = str(inst.get("assetType") or "")
            sym = str(inst.get("symbol") or "").upper()
            if asset and asset.upper() not in ("EQUITY", "COLLECTIVE_INVESTMENT"):
                if "OPTION" in asset.upper():
                    continue
            if sym in wanted:
                rows.append({"ticker": sym, "quantity": pos.get("longQuantity") or pos.get("shortQuantity")})
    return rows
