"""Optional Schwab last prices for the saved book. Never invent quotes."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from compoundcore.sleeve import TICKER_ORDER

MARKET = "https://api.schwabapi.com/marketdata/v1"
TRADER = "https://api.schwabapi.com/trader/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

_ENV_LOADED = False


def _load_dotenv() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    roots = (
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    )
    for path in roots:
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


def _token_path() -> Optional[Path]:
    raw = (os.environ.get("SCHWAB_TOKEN_PATH") or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_file():
        return path
    roots = (
        Path(__file__).resolve().parent.parent,
        Path(__file__).resolve().parent.parent.parent,
    )
    for root in roots:
        cand = (root / raw).resolve()
        if cand.is_file():
            return cand
    return None


def _access_token() -> Optional[str]:
    _load_dotenv()
    api_key = (os.environ.get("SCHWAB_API_KEY") or "").strip()
    secret = (os.environ.get("SCHWAB_APP_SECRET") or "").strip()
    path = _token_path()
    if not api_key or not secret or path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    inner = payload.get("token") if isinstance(payload, dict) and isinstance(payload.get("token"), dict) else payload
    if not isinstance(inner, dict):
        return None
    access = str(inner.get("access_token") or "").strip()
    refresh = str(inner.get("refresh_token") or "").strip()
    if refresh:
        auth = base64.b64encode(("%s:%s" % (api_key, secret)).encode("utf-8")).decode("ascii")
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
            with urllib.request.urlopen(req, timeout=20) as resp:
                fresh = json.loads(resp.read().decode("utf-8"))
            if isinstance(fresh, dict) and fresh.get("access_token"):
                inner.update(fresh)
                if isinstance(payload, dict) and "token" in payload:
                    payload["token"] = inner
                    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                return str(fresh.get("access_token") or "")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError):
            pass
    return access or None


def _get_json(url: str, token: str, timeout: float = 20.0) -> Any:
    req = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer %s" % token, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError, json.JSONDecodeError):
        return None


def _last_from_wrap(wrap: object) -> Optional[float]:
    if not isinstance(wrap, dict):
        return None
    q = wrap.get("quote") if isinstance(wrap.get("quote"), dict) else wrap
    if not isinstance(q, dict):
        return None
    for key in ("lastPrice", "mark", "closePrice"):
        try:
            px = float(q.get(key))
        except (TypeError, ValueError):
            continue
        if px > 0:
            return px
    return None


def last_prices(tickers: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """Schwab lastPrice map. Empty dict if credentials or the tape are unavailable."""
    names = [str(t).upper() for t in (tickers or TICKER_ORDER) if t]
    token = _access_token()
    if not token or not names:
        return {}
    joined = ",".join(urllib.parse.quote(n, safe="") for n in names)
    payload = _get_json("%s/quotes?symbols=%s" % (MARKET, joined), token)
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, float] = {}
    for key, wrap in payload.items():
        px = _last_from_wrap(wrap)
        if px is not None:
            out[str(key).upper()] = px
    return out


def _iter_accounts(payload: Any) -> List[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("accounts"), list):
            return [row for row in payload["accounts"] if isinstance(row, dict)]
        return [payload]
    return []


def sleeve_positions() -> Dict[str, Dict[str, float]]:
    """Schwab lots for Compound Core tickers only. Empty if the account tape is unavailable."""
    token = _access_token()
    if not token:
        return {}
    payload = _get_json("%s/accounts?fields=positions" % TRADER, token, timeout=30.0)
    out: Dict[str, Dict[str, float]] = {}
    for acct in _iter_accounts(payload):
        sec = acct.get("securitiesAccount") if isinstance(acct.get("securitiesAccount"), dict) else acct
        for pos in sec.get("positions") or []:
            if not isinstance(pos, dict):
                continue
            inst = pos.get("instrument") if isinstance(pos.get("instrument"), dict) else {}
            ticker = str(inst.get("symbol") or "").upper()
            if ticker not in TICKER_ORDER:
                continue
            try:
                long_qty = float(pos.get("longQuantity") or 0)
            except (TypeError, ValueError):
                long_qty = 0.0
            try:
                short_qty = float(pos.get("shortQuantity") or 0)
            except (TypeError, ValueError):
                short_qty = 0.0
            shares = long_qty - short_qty
            if shares <= 0:
                continue
            try:
                avg = float(pos.get("averagePrice") or 0)
            except (TypeError, ValueError):
                avg = 0.0
            try:
                market = float(pos.get("marketValue") or 0)
            except (TypeError, ValueError):
                market = 0.0
            prev = out.get(ticker) or {"shares": 0.0, "market": 0.0, "cost": 0.0}
            prev["shares"] = round(prev["shares"] + shares, 6)
            prev["market"] = round(prev["market"] + max(market, 0.0), 2)
            if avg > 0:
                prev["cost"] = round(prev["cost"] + avg * shares, 2)
            out[ticker] = prev
    return out
