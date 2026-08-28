"""Minimal ORATS delayed/historical client with credential-safe errors."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from codexswing.clock import iso_utc, parse_timestamp, session_close_utc, utc_now
from codexswing.schemas.source import SourceRecord, canonical_json


ORATS_BASE_URL = "https://api.orats.io/datav2"
ALLOWED_ENDPOINTS = {
    "cores",
    "summaries",
    "ivrank",
    "strikes",
    "hist/cores",
    "hist/summaries",
    "hist/ivrank",
    "hist/strikes",
    "hist/earnings",
    "hist/dailies",
    "hist/hvs",
    "hist/splits",
}
TRADE_DATE_REQUIRED_ENDPOINTS = {"hist/strikes"}
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.-]{0,14}$")


class ORATSError(RuntimeError):
    pass


class ORATSHTTPError(ORATSError):
    """Structured HTTP failure so callers can distinguish absent slices."""

    def __init__(self, endpoint: str, status_code: int, excerpt: str) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.excerpt = excerpt
        super().__init__(
            "ORATS {} returned HTTP {}: {}".format(endpoint, status_code, excerpt)
        )


class ORATSCredentialUnavailable(ORATSError):
    pass


Transport = Callable[[str, Mapping[str, str]], Mapping[str, Any]]


def _normalize_tickers(tickers: Iterable[str]) -> Tuple[str, ...]:
    normalized: List[str] = []
    seen = set()
    for raw in tickers:
        ticker = raw.strip().upper()
        if not TICKER_RE.fullmatch(ticker):
            raise ValueError("invalid ticker: {}".format(raw))
        if ticker not in seen:
            normalized.append(ticker)
            seen.add(ticker)
    if not normalized:
        raise ValueError("at least one ticker is required")
    return tuple(normalized)


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _timestamp_or_none(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) <= 10:
        return None
    try:
        return parse_timestamp(value.strip())
    except ValueError:
        return None


def _session_date(row: Mapping[str, Any]) -> str:
    for key in ("tradeDate", "quoteDate", "date", "updatedAt"):
        value = row.get(key)
        if isinstance(value, str) and len(value) >= 10:
            candidate = value[:10]
            try:
                datetime.strptime(candidate, "%Y-%m-%d")
            except ValueError:
                continue
            return candidate
    raise ORATSError("ORATS row is missing a defensible session date")


class ORATSClient:
    def __init__(
        self,
        token: Optional[str],
        timeout_seconds: int = 30,
        transport: Optional[Transport] = None,
    ) -> None:
        if not token:
            raise ORATSCredentialUnavailable("ORATS_TOKEN is unavailable in the authorized .env")
        self._token = token
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._default_transport

    def _default_transport(self, endpoint: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        query_params = dict(params)
        query_params["token"] = self._token
        query = urllib.parse.urlencode(query_params)
        url = "{}/{}?{}".format(ORATS_BASE_URL, endpoint, query)
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "codexswing/0.1"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            try:
                excerpt = exc.read(512).decode("utf-8", errors="replace").replace(self._token, "***REDACTED***")
            except Exception:
                excerpt = "unavailable"
            raise ORATSHTTPError(endpoint, exc.code, excerpt) from None
        except urllib.error.URLError as exc:
            raise ORATSError("ORATS {} request failed: {}".format(endpoint, exc.reason)) from None
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ORATSError("ORATS {} returned invalid JSON".format(endpoint)) from None
        if not isinstance(payload, Mapping):
            raise ORATSError("ORATS {} returned an unexpected payload".format(endpoint))
        return payload

    def fetch_rows(self, endpoint: str, params: Mapping[str, str]) -> Tuple[Mapping[str, Any], ...]:
        if endpoint not in ALLOWED_ENDPOINTS:
            raise ValueError("unsupported ORATS endpoint: {}".format(endpoint))
        safe_params = {str(key): str(value) for key, value in params.items() if value is not None}
        if any("token" in key.lower() or "secret" in key.lower() for key in safe_params):
            raise ValueError("credentials must not be supplied as endpoint parameters")
        payload = self._transport(endpoint, safe_params)
        rows = payload.get("data")
        if not isinstance(rows, list):
            message = payload.get("message") or payload.get("error") or "missing data array"
            rendered = str(message).replace(self._token, "***REDACTED***")
            raise ORATSError("ORATS {} response error: {}".format(endpoint, rendered))
        if not all(isinstance(row, Mapping) for row in rows):
            raise ORATSError("ORATS {} returned a non-object row".format(endpoint))
        return tuple(rows)

    def fetch_tickers(
        self,
        endpoint: str,
        tickers: Iterable[str],
        extra_params: Optional[Mapping[str, str]] = None,
    ) -> Tuple[Mapping[str, Any], ...]:
        normalized = _normalize_tickers(tickers)
        rows: List[Mapping[str, Any]] = []
        for chunk in _chunks(normalized, 10):
            params: Dict[str, str] = dict(extra_params or {})
            params["ticker"] = ",".join(chunk)
            rows.extend(self.fetch_rows(endpoint, params))
        return tuple(rows)

    def rows_to_records(
        self,
        endpoint: str,
        rows: Iterable[Mapping[str, Any]],
        ingested_at: Optional[datetime] = None,
    ) -> Tuple[SourceRecord, ...]:
        if endpoint not in ALLOWED_ENDPOINTS:
            raise ValueError("unsupported ORATS endpoint: {}".format(endpoint))
        ingestion_time = ingested_at or utc_now()
        records: List[SourceRecord] = []
        for row in rows:
            date_text = _session_date(row)
            updated = None
            for key in ("updatedAt", "updated_at", "quoteTime", "timestamp"):
                updated = _timestamp_or_none(row.get(key))
                if updated is not None:
                    break
            published = updated or ingestion_time
            if published > ingestion_time + timedelta(minutes=5):
                published = ingestion_time
            event_time = None
            for key in ("quoteDate", "tradeDate", "date"):
                event_time = _timestamp_or_none(row.get(key))
                if event_time is not None:
                    break
            if event_time is None:
                event_time = session_close_utc(date_text)
            ticker = str(row.get("ticker") or row.get("symbol") or "UNKNOWN").upper()
            natural_parts = [
                endpoint,
                ticker,
                date_text,
                str(row.get("expirDate") or row.get("expiration") or ""),
                str(row.get("strike") or ""),
                str(row.get("putCall") or row.get("callPut") or ""),
            ]
            row_digest = hashlib.sha256(canonical_json(row).encode("utf-8")).hexdigest()[:20]
            source_id = ":".join(natural_parts + [row_digest])
            records.append(
                SourceRecord(
                    source="orats_{}".format(endpoint.replace("/", "_")),
                    source_id=source_id,
                    session_date=date_text,
                    event_time_utc=iso_utc(event_time),
                    published_at_utc=iso_utc(published),
                    first_seen_at_utc=iso_utc(ingestion_time),
                    available_at_utc=iso_utc(published),
                    ingested_at_utc=iso_utc(ingestion_time),
                    source_uri="{}/{}".format(ORATS_BASE_URL, endpoint),
                    revision=str(row.get("updatedAt") or row_digest),
                    payload=dict(row),
                )
            )
        return tuple(records)

    def probe(self, ticker: str = "SPY") -> Dict[str, Any]:
        rows = self.fetch_tickers("cores", [ticker])
        dates = sorted({_session_date(row) for row in rows})
        return {
            "status": "available",
            "endpoint": "cores",
            "ticker": ticker,
            "row_count": len(rows),
            "session_dates": dates,
        }
