"""Optional Schwab last prices for the saved book. Never invent quotes."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, Optional

from compoundcore.sleeve import TICKER_ORDER

MARKET = "https://api.schwabapi.com/marketdata/v1"
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
    return path if path.is_file() else None


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
    req = urllib.request.Request(
        "%s/quotes?symbols=%s" % (MARKET, joined),
        headers={"Authorization": "Bearer %s" % token, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, float] = {}
    for key, wrap in payload.items():
        px = _last_from_wrap(wrap)
        if px is not None:
            out[str(key).upper()] = px
    return out
