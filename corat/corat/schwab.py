"""Read-only Schwab quote adapter.

CORAT never refreshes or writes the shared token file and exposes no trading
endpoint. If the access token is expired, quotes fail closed until the user's
separate token-renewal workflow succeeds.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from corat.models import Bar, SourceTrace
from corat.secrets import read_env, redact
from corat.store import canonical_json, sha256_bytes, sha256_file, utc_now, write_json


class SchwabError(RuntimeError):
    pass


@dataclass
class SchwabBundle:
    quotes: Dict[str, Mapping[str, Any]] = field(default_factory=dict)
    traces: List[SourceTrace] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _chunks(values: Sequence[str], size: int):
    for index in range(0, len(values), max(1, size)):
        yield values[index : index + max(1, size)]


def load_credentials(env_path: Path) -> Dict[str, Any]:
    resolved_env = env_path.expanduser().resolve()
    env = read_env(resolved_env)
    token_path = Path(str(env.get("SCHWAB_TOKEN_PATH") or ""))
    if token_path and not token_path.is_absolute():
        token_path = (resolved_env.parent / token_path).resolve()
    if not token_path.is_file():
        raise SchwabError("Schwab token file unavailable")
    try:
        payload = json.loads(token_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise SchwabError("Schwab token file is unreadable") from None
    token = payload.get("token") if isinstance(payload, dict) else None
    token = token if isinstance(token, dict) else payload
    if not isinstance(token, dict):
        raise SchwabError("Schwab token payload is invalid")
    access_token = str(token.get("access_token") or "")
    expires_at = _number(token.get("expires_at")) or 0.0
    if not access_token:
        raise SchwabError("Schwab access token missing")
    if expires_at and expires_at <= time.time() + 30:
        raise SchwabError("Schwab access token expired; CORAT does not mutate shared OAuth state")
    return {
        "access_token": access_token,
        "expires_at": expires_at,
        "token_path": str(token_path),
    }


class SchwabClient:
    def __init__(
        self,
        env_path: Path,
        base_url: str,
        cache_root: Path,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.credentials = load_credentials(env_path)
        self.base_url = base_url.rstrip("/")
        self.cache_root = cache_root
        self.timeout_seconds = float(timeout_seconds)

    def _request(self, endpoint: str, params: Mapping[str, str]) -> SchwabBundle:
        clean = {str(key): str(value) for key, value in params.items()}
        digest = sha256_bytes(canonical_json({"endpoint": endpoint, "params": clean}).encode("utf-8"))[:24]
        cache_path = self.cache_root / "schwab" / endpoint.strip("/").replace("/", "_") / (digest + ".json")
        url = self.base_url + endpoint + "?" + urllib.parse.urlencode(clean)
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "Authorization": "Bearer {}".format(self.credentials["access_token"]),
                "User-Agent": "CORAT/0.1 read-only research",
            },
        )
        fetched_at = utc_now()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads((response.read() or b"{}").decode("utf-8"))
                status = int(response.getcode() or 0)
        except urllib.error.HTTPError as exc:
            message = "Schwab HTTP {} for {}".format(int(exc.code or 0), endpoint)
            return SchwabBundle(errors=[redact(message, self.credentials["access_token"])])
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            message = "Schwab {} for {}".format(type(exc).__name__, endpoint)
            return SchwabBundle(errors=[redact(message, self.credentials["access_token"])])
        if not isinstance(payload, dict):
            return SchwabBundle(errors=["Schwab returned an invalid quote payload"])
        body = {
            "schema_version": "corat.schwab_cache.v1",
            "endpoint": endpoint,
            "params": clean,
            "fetched_at_utc": fetched_at,
            "http_status": status,
            "payload": payload,
        }
        write_json(cache_path, body)
        latest = ""
        for quote in payload.values():
            if not isinstance(quote, dict):
                continue
            values = quote.get("quote") if isinstance(quote.get("quote"), dict) else {}
            stamp = values.get("quoteTime") or values.get("tradeTime") or values.get("regularMarketTradeTime")
            latest = max(latest, str(stamp or ""))
        trace = SourceTrace(
            source="SCHWAB",
            endpoint=endpoint,
            status="LIVE_READ_ONLY_FETCH",
            fetched_at_utc=fetched_at,
            latest_data_at=latest,
            rows=len(payload),
            cache_path=str(cache_path),
            cache_sha256=sha256_file(cache_path),
            params=clean,
        )
        return SchwabBundle(
            quotes={str(key).upper(): value for key, value in payload.items() if isinstance(value, dict)},
            traces=[trace],
        )

    def fetch_quotes(self, tickers: Iterable[str], batch_size: int = 50) -> SchwabBundle:
        names = sorted({str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()})
        result = SchwabBundle()
        for chunk in _chunks(names, batch_size):
            current = self._request("/quotes", {"symbols": ",".join(chunk), "fields": "quote,reference,regular"})
            result.quotes.update(current.quotes)
            result.traces.extend(current.traces)
            result.errors.extend(current.errors)
        return result


def _timestamp_ms(value: Any) -> str:
    parsed = _number(value)
    if parsed is None or parsed <= 0:
        return ""
    if parsed > 10_000_000_000:
        parsed /= 1000.0
    return datetime.fromtimestamp(parsed, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def quote_timestamp(payload: Mapping[str, Any]) -> str:
    quote = payload.get("quote") if isinstance(payload.get("quote"), dict) else {}
    regular = payload.get("regular") if isinstance(payload.get("regular"), dict) else {}
    for key in ("quoteTime", "tradeTime", "regularMarketTradeTime", "quoteTimeInLong", "tradeTimeInLong"):
        raw = quote.get(key)
        if raw in (None, ""):
            raw = regular.get(key)
        if raw not in (None, ""):
            if isinstance(raw, (int, float)):
                stamp = _timestamp_ms(raw)
                if stamp:
                    return stamp
            text = str(raw)
            if "T" in text:
                return text
    return ""


def quote_to_bar(ticker: str, payload: Mapping[str, Any], as_of: str) -> Optional[Bar]:
    quote = payload.get("quote") if isinstance(payload.get("quote"), dict) else {}
    regular = payload.get("regular") if isinstance(payload.get("regular"), dict) else {}
    price = (
        _number(quote.get("lastPrice"))
        or _number(regular.get("regularMarketLastPrice"))
        or _number(quote.get("mark"))
        or _number(quote.get("closePrice"))
    )
    if price is None or price <= 0:
        return None
    open_ = _number(quote.get("openPrice")) or _number(regular.get("regularMarketOpenPrice")) or price
    high = _number(quote.get("highPrice")) or _number(regular.get("regularMarketDayHigh")) or max(open_, price)
    low = _number(quote.get("lowPrice")) or _number(regular.get("regularMarketDayLow")) or min(open_, price)
    volume = _number(quote.get("totalVolume")) or _number(regular.get("regularMarketVolume")) or 0.0
    return Bar(
        date=as_of,
        open=open_,
        high=max(high, low),
        low=min(high, low),
        close=price,
        volume=max(0.0, volume),
        complete=False,
        updated_at=quote_timestamp(payload),
        source="Schwab read-only quote",
    )


def merge_quote_bar(bars: Sequence[Bar], quote_bar: Bar) -> List[Bar]:
    by_date = {bar.date: bar for bar in bars}
    by_date[quote_bar.date] = quote_bar
    return [by_date[key] for key in sorted(by_date)]


def quote_is_fresh(payload: Mapping[str, Any], maximum_age_minutes: float) -> bool:
    stamp = quote_timestamp(payload)
    if not stamp:
        return False
    try:
        parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    age = (datetime.now(timezone.utc) - parsed).total_seconds() / 60.0
    return -5.0 <= age <= float(maximum_age_minutes)

