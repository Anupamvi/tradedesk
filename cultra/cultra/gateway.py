"""The sole Cultra ORATS credential, URL, transport, cache, and ledger boundary."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import os
import random
import socket
import socketserver
import stat
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple

from .cache import (
    CacheError,
    CacheMiss,
    ContentAddressedCache,
    SingleFlight,
    SnapshotManifest,
    VintageExpectation,
    cache_key_for,
)
from .ledger import BudgetExhausted, RequestLedger, account_ledger_path
from .requesting import (
    IDEMPOTENCY_CONTRACTS,
    MAX_GET_URL_BYTES,
    PlannedRequest,
    RequestPlan,
    request_query_parameters,
)


ORATS_DELAYED_BASE_URL = "https://api.orats.io"
SCHWAB_API_BASE_URL = "https://api.schwabapi.com"
CULTRA_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CULTRA_ENV_PATH = (CULTRA_PROJECT_ROOT / ".env").resolve()


def redact_text(value: str, secrets: Iterable[str]) -> str:
    """Redact literal and URL-encoded secret values from diagnostic text."""

    result = str(value)
    for secret in secrets:
        if not secret:
            continue
        result = result.replace(secret, "[REDACTED]")
        result = result.replace(urllib.parse.quote_plus(secret), "[REDACTED]")
        result = result.replace(urllib.parse.quote(secret, safe=""), "[REDACTED]")
    return result


def _contains_secret(value: bytes, secret: str) -> bool:
    if not secret:
        return False
    candidates = {
        secret.encode("utf-8"),
        urllib.parse.quote_plus(secret).encode("ascii"),
        urllib.parse.quote(secret, safe="").encode("ascii"),
    }
    return any(candidate and candidate in value for candidate in candidates)


class GatewayError(RuntimeError):
    """A deliberately secret-free gateway failure."""


class GatewayRequestError(GatewayError):
    """A planned request could not produce a validated snapshot."""


class TokenSourceError(GatewayError):
    """Cultra could not safely load its own token source."""


class SafeTransportError(GatewayError):
    """Transport failed without exposing its URL, headers, or credential."""

    def __init__(self, category: str, *, retryable: bool, ambiguous: bool) -> None:
        super().__init__("ORATS transport failure: %s" % category)
        self.category = category
        self.retryable = bool(retryable)
        self.ambiguous = bool(ambiguous)


class SchwabSafeTransportError(GatewayError):
    """A secret-free Schwab transport failure with no implicit retry."""

    def __init__(self, category: str) -> None:
        super().__init__("Schwab market-data transport failure: %s" % category)
        self.category = category


@dataclass(frozen=True)
class TransportResponse:
    status_code: int
    body: bytes
    headers: Tuple[Tuple[str, str], ...]
    duration_ms: float

    def header(self, name: str) -> Optional[str]:
        target = name.lower()
        for key, value in self.headers:
            if key.lower() == target:
                return value
        return None


class TokenSource(Protocol):
    def load(self, *, force_reload: bool = False) -> str:
        ...


class Transport(Protocol):
    def send(self, request: PlannedRequest, token: str) -> TransportResponse:
        ...


class EnvFileTokenSource:
    """Strict reader for Cultra's own mode-0600, one-secret .env file."""

    def __init__(self, path: Path, *, variable: str = "ORATS_TOKEN") -> None:
        self._source_path = Path(os.path.abspath(os.path.expanduser(str(path))))
        if self._source_path.resolve() != CULTRA_ENV_PATH.resolve():
            raise TokenSourceError("token source must be Cultra's private .env")
        self._path = self._source_path.resolve()
        self._variable = variable
        self._cached: Optional[str] = None
        self._lock = threading.Lock()

    def load(self, *, force_reload: bool = False) -> str:
        with self._lock:
            if self._cached is not None and not force_reload:
                return self._cached
            try:
                if self._source_path.is_symlink():
                    raise TokenSourceError("Cultra token source may not be a symlink")
                metadata = self._path.stat()
                if not stat.S_ISREG(metadata.st_mode):
                    raise TokenSourceError("Cultra token source is not a regular file")
                if stat.S_IMODE(metadata.st_mode) & 0o077:
                    raise TokenSourceError("Cultra token source permissions must be 0600")
                lines = self._path.read_text(encoding="utf-8").splitlines()
            except TokenSourceError:
                raise
            except (OSError, UnicodeError) as exc:
                raise TokenSourceError("Cultra token source is unavailable") from exc
            values: Dict[str, str] = {}
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if "=" not in stripped:
                    raise TokenSourceError("Cultra token source has invalid syntax")
                key, value = stripped.split("=", 1)
                key = key.strip()
                if key != self._variable or key in values:
                    raise TokenSourceError("Cultra token source contains an unexpected key")
                clean = value.strip()
                if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in ("'", '"'):
                    clean = clean[1:-1]
                values[key] = clean
            token = values.get(self._variable, "")
            if not token or any(character.isspace() for character in token):
                raise TokenSourceError("Cultra token is missing or malformed")
            self._cached = token
            return token


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


class UrllibTransport:
    """One-attempt stdlib transport with redirects and implicit retries disabled."""

    def __init__(
        self,
        *,
        base_url: str = ORATS_DELAYED_BASE_URL,
        timeout_seconds: float = 30.0,
    ) -> None:
        if base_url.rstrip("/") != ORATS_DELAYED_BASE_URL:
            raise GatewayError("ORATS base URL is not the frozen delayed-data host")
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = float(timeout_seconds)
        # ProxyHandler({}) prevents ambient proxy inheritance; redirects are
        # surfaced as a single 3xx response and are never followed.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirectHandler()
        )

    @staticmethod
    def _request_parameters(request: PlannedRequest) -> Dict[str, Any]:
        return request_query_parameters(request)

    def send(self, request: PlannedRequest, token: str) -> TransportResponse:
        if not token:
            raise SafeTransportError("authentication", retryable=False, ambiguous=False)
        parameters = self._request_parameters(request)
        parameters["token"] = token
        encoded_query = urllib.parse.urlencode(parameters, doseq=True)
        url = self._base_url + request.endpoint.value
        body: Optional[bytes] = None
        headers = {"Accept": "application/json", "User-Agent": "Cultra/1"}
        if request.method == "GET":
            url = url + "?" + encoded_query
            if len(url.encode("utf-8")) > MAX_GET_URL_BYTES:
                raise SafeTransportError("planned_url_too_large", retryable=False, ambiguous=False)
        else:
            payload = dict(request.body)
            payload["entities"] = list(request.entities)
            payload["fields"] = list(request.fields)
            # Keep the token in the query only inside this non-reporting method.
            url = url + "?" + urllib.parse.urlencode({"token": token})
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        outbound = urllib.request.Request(
            url=url, data=body, headers=headers, method=request.method
        )
        started = time.monotonic()
        try:
            response = self._opener.open(outbound, timeout=self._timeout_seconds)
            try:
                response_body = response.read(request.max_response_bytes + 1)
                status = int(response.getcode())
                response_headers = response.headers
            finally:
                response.close()
        except urllib.error.HTTPError as exc:
            # HTTPError is the response for blocked redirects and known 4xx/5xx.
            try:
                response_body = exc.read(request.max_response_bytes + 1)
            except Exception:
                response_body = b""
            status = int(exc.code)
            response_headers = exc.headers
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            del exc
            raise SafeTransportError("network", retryable=True, ambiguous=True) from None
        except SafeTransportError:
            raise
        except Exception as exc:
            del exc
            raise SafeTransportError("unexpected", retryable=False, ambiguous=True) from None
        if len(response_body) > request.max_response_bytes:
            raise SafeTransportError("response_too_large", retryable=False, ambiguous=False)
        duration_ms = (time.monotonic() - started) * 1000.0
        safe_headers = []
        for name in ("Retry-After", "Content-Type", "Content-Length"):
            value = response_headers.get(name) if response_headers is not None else None
            if value is not None:
                safe_headers.append((name, str(value)))
        return TransportResponse(status, response_body, tuple(safe_headers), duration_ms)


class SchwabUrllibTransport:
    """Restricted one-attempt transport for Schwab OAuth and market data.

    The allowlist makes account and order endpoints structurally unreachable.
    Redirects, ambient proxies, and automatic retries are disabled.
    """

    _ALLOWED = frozenset(
        {
            ("POST", "/v1/oauth/token"),
            ("GET", "/marketdata/v1/quotes"),
            ("GET", "/marketdata/v1/chains"),
            ("GET", "/marketdata/v1/pricehistory"),
        }
    )

    def __init__(self, *, timeout_seconds: float = 30.0) -> None:
        self._timeout_seconds = float(timeout_seconds)
        if not math.isfinite(self._timeout_seconds) or self._timeout_seconds <= 0:
            raise GatewayError("Schwab timeout must be positive")
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirectHandler()
        )

    def send(
        self,
        *,
        method: str,
        path: str,
        query: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        body: Optional[bytes] = None,
        max_response_bytes: int = 25_000_000,
    ) -> TransportResponse:
        normalized_method = str(method).strip().upper()
        normalized_path = "/" + str(path).strip().lstrip("/")
        if (normalized_method, normalized_path) not in self._ALLOWED:
            raise SchwabSafeTransportError("endpoint_not_allowed")
        if normalized_method == "GET" and body is not None:
            raise SchwabSafeTransportError("invalid_get_body")
        if normalized_method == "POST" and query:
            raise SchwabSafeTransportError("oauth_query_not_allowed")
        if max_response_bytes <= 0 or max_response_bytes > 50_000_000:
            raise SchwabSafeTransportError("invalid_response_limit")
        url = SCHWAB_API_BASE_URL + normalized_path
        if query:
            url += "?" + urllib.parse.urlencode(dict(query), doseq=True)
        outbound = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers or {}),
            method=normalized_method,
        )
        started = time.monotonic()
        try:
            response = self._opener.open(outbound, timeout=self._timeout_seconds)
            try:
                response_body = response.read(max_response_bytes + 1)
                status = int(response.getcode())
                response_headers = response.headers
            finally:
                response.close()
        except urllib.error.HTTPError as exc:
            try:
                response_body = exc.read(max_response_bytes + 1)
            except Exception:
                response_body = b""
            status = int(exc.code)
            response_headers = exc.headers
        except (urllib.error.URLError, TimeoutError, OSError):
            raise SchwabSafeTransportError("network") from None
        except Exception:
            raise SchwabSafeTransportError("unexpected") from None
        if len(response_body) > max_response_bytes:
            raise SchwabSafeTransportError("response_too_large")
        safe_headers = []
        for name in ("Content-Type", "Content-Length"):
            value = response_headers.get(name) if response_headers is not None else None
            if value is not None:
                safe_headers.append((name, str(value)))
        return TransportResponse(
            status,
            response_body,
            tuple(safe_headers),
            (time.monotonic() - started) * 1000.0,
        )


@dataclass(frozen=True)
class ResponseMetadata:
    row_count: int
    returned_entities: Tuple[str, ...]
    provider_trade_dates: Tuple[str, ...]
    updated_at_min: Optional[str]
    updated_at_max: Optional[str]


@dataclass(frozen=True)
class GatewayResult:
    logical_request_id: str
    cache_hit: bool
    raw: bytes
    manifest: SnapshotManifest
    charged_attempts: int

    def to_public_dict(self, *, include_raw: bool = False) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "logical_request_id": self.logical_request_id,
            "cache_hit": self.cache_hit,
            "snapshot_id": self.manifest.snapshot_id,
            "row_count": self.manifest.row_count,
            "raw_bytes": self.manifest.raw_bytes,
            "charged_attempts": self.charged_attempts,
            "manifest": self.manifest.to_dict(),
        }
        if include_raw:
            result["raw_base64"] = base64.b64encode(self.raw).decode("ascii")
        return result

    @classmethod
    def from_public_dict(cls, value: Mapping[str, Any]) -> "GatewayResult":
        try:
            if set(value) != {
                "logical_request_id",
                "cache_hit",
                "snapshot_id",
                "row_count",
                "raw_bytes",
                "charged_attempts",
                "manifest",
                "raw_base64",
            }:
                raise ValueError
            if not isinstance(value["cache_hit"], bool):
                raise ValueError
            if (
                isinstance(value["charged_attempts"], bool)
                or not isinstance(value["charged_attempts"], int)
                or int(value["charged_attempts"]) < 0
            ):
                raise ValueError
            raw = base64.b64decode(str(value["raw_base64"]), validate=True)
            manifest = SnapshotManifest.from_dict(value["manifest"])
            result = cls(
                logical_request_id=str(value["logical_request_id"]),
                cache_hit=value["cache_hit"],
                raw=raw,
                manifest=manifest,
                charged_attempts=int(value["charged_attempts"]),
            )
            if (
                not result.logical_request_id
                or len(result.logical_request_id) > 128
                or value["snapshot_id"] != manifest.snapshot_id
                or value["row_count"] != manifest.row_count
                or value["raw_bytes"] != manifest.raw_bytes
                or len(raw) != manifest.raw_bytes
                or (result.cache_hit and result.charged_attempts != 0)
            ):
                raise ValueError
        except (CacheError, KeyError, TypeError, ValueError) as exc:
            raise GatewayRequestError("gateway daemon result is malformed") from exc
        if hashlib.sha256(raw).hexdigest() != manifest.raw_sha256:
            raise GatewayRequestError("gateway daemon raw response failed integrity check")
        return result


def _records_from_json(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                raise CacheError("provider response contains a non-object row")
            yield item
        return
    if isinstance(value, Mapping):
        containers = [key for key in ("data", "rows", "records", "results") if key in value]
        if len(containers) > 1:
            raise CacheError("provider response has ambiguous row containers")
        if containers:
            nested = value[containers[0]]
            if not isinstance(nested, list):
                raise CacheError("provider response row container is not a list")
            for item in nested:
                if not isinstance(item, Mapping):
                    raise CacheError("provider response contains a non-object row")
                yield item
            return
        yield value
        return
    raise CacheError("provider response JSON root is invalid")


def parse_response_metadata(raw: bytes, content_type: str = "") -> ResponseMetadata:
    """Extract bounded provenance without inventing absent provider fields."""

    records = list(_decode_response_records(raw, content_type))
    entities = set()
    dates = set()
    updated = []
    for record in records:
        for key in ("optionSymbol", "option_symbol", "ticker", "symbol"):
            value = record.get(key)
            if value:
                entities.add(str(value).strip().upper())
                break
        for key in ("tradeDate", "trade_date"):
            value = record.get(key)
            if value:
                dates.add(str(value))
                break
        for key in ("updatedAt", "updated_at"):
            value = record.get(key)
            if value:
                updated.append(str(value))
                break
    return ResponseMetadata(
        row_count=len(records),
        returned_entities=tuple(sorted(entities)),
        provider_trade_dates=tuple(sorted(dates)),
        updated_at_min=min(updated) if updated else None,
        updated_at_max=max(updated) if updated else None,
    )


def _decode_response_records(
    raw: bytes, content_type: str = ""
) -> Tuple[Mapping[str, Any], ...]:
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CacheError("provider response is not UTF-8") from exc
    try:
        parsed = json.loads(decoded)
        records = tuple(_records_from_json(parsed))
    except json.JSONDecodeError:
        if "csv" not in content_type.lower() and not decoded.lstrip().startswith(
            ("ticker,", "optionSymbol,", "symbol,")
        ):
            raise CacheError("provider response is neither validated JSON nor CSV")
        records = tuple(csv.DictReader(io.StringIO(decoded)))
    return tuple(records)


def validate_response_rows(
    request: PlannedRequest, raw: bytes, content_type: str = ""
) -> ResponseMetadata:
    """Validate the frozen row schema before any success body is cached."""

    records = _decode_response_records(raw, content_type)
    date_fields = {"tradeDate", "trade_date", "expirDate", "splitDate"}
    identities = set()
    for record in records:
        missing = set(request.fields).difference(record)
        if missing:
            raise CacheError("provider response is missing frozen profile fields")
        unknown = set(record).difference(request.fields)
        if unknown:
            raise CacheError("provider response contains fields outside the frozen profile")
        for field_name, value in record.items():
            if isinstance(value, (Mapping, list, tuple, set)):
                raise CacheError("provider response field has an invalid nested type")
            if isinstance(value, float) and not math.isfinite(value):
                raise CacheError("provider response contains a non-finite number")
            if field_name in date_fields and value not in (None, ""):
                try:
                    # Date-only validation is intentional; updatedAt is a
                    # provider timestamp string and is preserved verbatim.
                    from datetime import date as _date

                    _date.fromisoformat(str(value))
                except ValueError as exc:
                    raise CacheError("provider response contains an invalid date") from exc
        identity = tuple(
            str(record.get(name, ""))
            for name in (
                "optionSymbol",
                "ticker",
                "tradeDate",
                "expirDate",
                "strike",
                "splitDate",
            )
        )
        if identity in identities:
            raise CacheError("provider response contains duplicate row identity")
        identities.add(identity)
    return parse_response_metadata(raw, content_type)


class OratsGateway:
    """Execute only immutable planned IDs behind cache and durable permits."""

    def __init__(
        self,
        *,
        plan: RequestPlan,
        ledger: RequestLedger,
        cache: ContentAddressedCache,
        token_source: TokenSource,
        transport: Transport,
        single_flight: Optional[SingleFlight] = None,
        sleeper: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.plan = plan
        self.ledger = ledger
        self.cache = cache
        self._token_source = token_source
        self._transport = transport
        self._single_flight = single_flight or SingleFlight()
        self._sleeper = sleeper
        self._jitter = jitter
        self._token_cache: Optional[str] = None
        self._authorization_failed = threading.Event()
        if ledger.path != account_ledger_path():
            raise GatewayError("ORATS gateway requires the shared account ledger")
        # The token remains unloaded until a validated cache miss. Generic
        # credential-like keys were already rejected by the tokenless planner.
        # This is what makes a same-vintage warm run genuinely tokenless.
        self.ledger.assert_healthy()
        self.ledger.start_run(plan)

    @staticmethod
    def _expectation(request: PlannedRequest) -> VintageExpectation:
        return VintageExpectation.from_request(request)

    def execute(self, logical_request_id: str) -> GatewayResult:
        try:
            request = self.plan.get(logical_request_id)
        except KeyError as exc:
            raise GatewayRequestError("request ID is not present in the frozen plan") from exc
        expectation = self._expectation(request)
        key = cache_key_for(request, expectation)
        try:
            manifest, raw = self.cache.lookup(request, expectation)
            return GatewayResult(logical_request_id, True, raw, manifest, 0)
        except CacheMiss:
            pass
        except CacheError as exc:
            raise GatewayRequestError("validated cache lookup failed") from exc

        def leader() -> GatewayResult:
            try:
                manifest, raw = self.cache.lookup(request, expectation)
                return GatewayResult(logical_request_id, True, raw, manifest, 0)
            except CacheMiss:
                return self._execute_uncached(request, expectation)
            except CacheError as exc:
                raise GatewayRequestError("validated cache lookup failed") from exc

        return self._single_flight.run(key, leader)

    def _execute_uncached(
        self, request: PlannedRequest, expectation: VintageExpectation
    ) -> GatewayResult:
        if self._authorization_failed.is_set():
            raise GatewayRequestError("ORATS authorization circuit is open")
        token = self._load_token(force_reload=False)
        identity_bytes = json.dumps(
            request.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if _contains_secret(identity_bytes, token):
            raise GatewayRequestError("planned request contains credential material")
        retries_used = 0
        charged = 0
        while True:
            permit = self.ledger.reserve_attempt(
                self.plan.run_id,
                request.logical_request_id,
                retry_number=retries_used,
            )
            charged += 1
            self.ledger.mark_indeterminate(permit)
            try:
                response = self._transport.send(request, token)
            except SafeTransportError as exc:
                if exc.category != "planned_url_too_large":
                    self.ledger.record_provider_result(self.plan.run_id, success=False)
                if (
                    exc.retryable
                    and self._retry_is_idempotent(request)
                    and retries_used < request.retry_limit
                ):
                    retries_used += 1
                    self._bounded_backoff(retries_used, None)
                    continue
                raise GatewayRequestError("ORATS transport failed safely") from None
            except BaseException as exc:
                del exc
                raise GatewayRequestError("ORATS transport failed safely") from None

            try:
                metadata = parse_response_metadata(
                    response.body, response.header("Content-Type") or ""
                )
            except CacheError:
                metadata = ResponseMetadata(0, (), (), None, None)
            self.ledger.mark_confirmed(
                permit,
                status_code=response.status_code,
                rows_returned=metadata.row_count,
                bytes_returned=len(response.body),
                duration_ms=response.duration_ms,
                provider_trade_date=(
                    metadata.provider_trade_dates[0]
                    if len(metadata.provider_trade_dates) == 1
                    else None
                ),
                updated_at_min=metadata.updated_at_min,
                updated_at_max=metadata.updated_at_max,
            )
            if 200 <= response.status_code < 300:
                try:
                    if _contains_secret(response.body, token):
                        raise CacheError("provider response echoed credential material")
                    # Parse again here so malformed success bodies cannot be cached.
                    metadata = validate_response_rows(
                        request,
                        response.body,
                        response.header("Content-Type") or "",
                    )
                    manifest = self.cache.publish(
                        request=request,
                        expectation=expectation,
                        raw=response.body,
                        provider_trade_dates=metadata.provider_trade_dates,
                        returned_entities=metadata.returned_entities,
                        row_count=metadata.row_count,
                        updated_at_min=metadata.updated_at_min,
                        updated_at_max=metadata.updated_at_max,
                    )
                except CacheError as exc:
                    self.ledger.record_provider_result(self.plan.run_id, success=False)
                    self.ledger.mark_completed(permit, outcome_code="VALIDATION_FAILED")
                    raise GatewayRequestError("provider response validation failed") from exc
                self.ledger.record_provider_result(self.plan.run_id, success=True)
                self.ledger.mark_completed(permit, outcome_code="SUCCESS")
                return GatewayResult(
                    request.logical_request_id, False, response.body, manifest, charged
                )

            self.ledger.mark_completed(
                permit, outcome_code="HTTP_%d" % response.status_code
            )
            retryable_status = (
                response.status_code in (408, 429) or response.status_code >= 500
            )
            auth_failure = response.status_code in (401, 403)
            auth_retry = auth_failure and retries_used == 0
            if auth_failure:
                self._authorization_failed.set()
                self.ledger.record_provider_result(self.plan.run_id, success=False)
            if retryable_status:
                self.ledger.record_provider_result(self.plan.run_id, success=False)
            if (
                (retryable_status or auth_retry)
                and self._retry_is_idempotent(request)
                and retries_used < request.retry_limit
            ):
                retries_used += 1
                if auth_retry:
                    token = self._load_token(force_reload=True)
                self._bounded_backoff(retries_used, response.header("Retry-After"))
                continue
            raise GatewayRequestError("ORATS returned a non-success status")

    @staticmethod
    def _retry_is_idempotent(request: PlannedRequest) -> bool:
        if request.method == "GET":
            return True
        return bool(
            request.idempotency_contract
            and IDEMPOTENCY_CONTRACTS.get(request.endpoint)
            == request.idempotency_contract
        )

    def _load_token(self, *, force_reload: bool) -> str:
        if self._token_cache is not None and not force_reload:
            return self._token_cache
        try:
            token = self._token_source.load(force_reload=force_reload)
        except TokenSourceError as exc:
            del exc
            raise TokenSourceError("Cultra token source failed safely") from None
        except BaseException as exc:
            del exc
            raise TokenSourceError("Cultra token source failed safely") from None
        if not isinstance(token, str) or not token or any(char.isspace() for char in token):
            raise TokenSourceError("Cultra token source returned malformed material")
        self._token_cache = token
        return token

    def _bounded_backoff(self, retry_number: int, retry_after: Optional[str]) -> None:
        delay = min(8.0, float(2 ** max(0, retry_number - 1)))
        if retry_after:
            try:
                delay = min(30.0, max(delay, float(retry_after)))
            except ValueError:
                pass
        delay += min(0.5, max(0.0, float(self._jitter())) * 0.5)
        self._sleeper(delay)


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False


class _GatewaySocketHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        raw = self.rfile.readline(65_537)
        if len(raw) > 65_536:
            self._write_error("request_too_large")
            return
        try:
            message = json.loads(raw.decode("utf-8"))
            if (
                not isinstance(message, dict)
                or set(message) != {"operation", "planned_id"}
                or message["operation"] != "execute"
                or not isinstance(message["planned_id"], str)
            ):
                raise ValueError
            gateway = getattr(self.server, "gateway")
            result = gateway.execute(message["planned_id"])
            response = {"ok": True, "result": result.to_public_dict(include_raw=True)}
            self.wfile.write(json.dumps(response, sort_keys=True).encode("utf-8") + b"\n")
        except (ValueError, UnicodeError, json.JSONDecodeError):
            self._write_error("invalid_request")
        except BaseException:
            # Never serialize exception messages or chained transport details.
            self._write_error("gateway_failure")

    def _write_error(self, code: str) -> None:
        response = {"ok": False, "error": code}
        self.wfile.write(json.dumps(response, sort_keys=True).encode("utf-8") + b"\n")


class OratsGatewayServer:
    """Optional token-holding Unix-socket daemon for tokenless workers."""

    def __init__(self, socket_path: Path, gateway: OratsGateway) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.socket_path.parent, 0o700)
        if self.socket_path.exists():
            raise GatewayError("gateway socket path already exists")
        self._server = _ThreadingUnixServer(str(self.socket_path), _GatewaySocketHandler)
        setattr(self._server, "gateway", gateway)
        os.chmod(self.socket_path, 0o600)

    def serve_forever(self) -> None:
        self._server.serve_forever(poll_interval=0.1)

    def shutdown(self) -> None:
        self._server.shutdown()

    def close(self) -> None:
        self._server.server_close()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass


class OratsGatewayClient:
    """Tokenless client that can request only a frozen planned ID."""

    def __init__(self, socket_path: Path, *, timeout_seconds: float = 30.0) -> None:
        self.socket_path = Path(socket_path).expanduser().resolve()
        self.timeout_seconds = timeout_seconds

    def execute(self, planned_id: str) -> Dict[str, Any]:
        if not isinstance(planned_id, str) or not planned_id or len(planned_id) > 128:
            raise GatewayRequestError("invalid planned request ID")
        message = json.dumps(
            {"operation": "execute", "planned_id": planned_id}, sort_keys=True
        ).encode("utf-8") + b"\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout_seconds)
                connection.connect(str(self.socket_path))
                connection.sendall(message)
                stream = connection.makefile("rb")
                raw = stream.readline(100_000_001)
        except OSError as exc:
            raise GatewayRequestError("gateway daemon is unavailable") from exc
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise GatewayRequestError("gateway daemon returned invalid control data") from exc
        if not response.get("ok"):
            raise GatewayRequestError("gateway daemon rejected the request")
        return dict(response["result"])


def execute_plan_via_local_daemon(
    gateway: OratsGateway,
    planned_ids: Sequence[str],
    *,
    socket_path: Path,
    workers: int,
    client_timeout_seconds: float = 120.0,
) -> Tuple[Mapping[str, GatewayResult], Mapping[str, str]]:
    """Run tokenless workers through one local token-holding daemon.

    Cache hits are checked sequentially until the first physical success.  That
    first miss is an authorization canary: a 401/403 therefore stops the run
    before a worker pool can fan out invalid authenticated requests.
    """

    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4:
        raise GatewayError("workers must be between 1 and 4")
    if not math.isfinite(float(client_timeout_seconds)) or client_timeout_seconds <= 0:
        raise GatewayError("client timeout must be positive and finite")
    identifiers = tuple(planned_ids)
    if not identifiers or len(identifiers) != len(set(identifiers)):
        raise GatewayError("planned request IDs must be non-empty and unique")
    server = OratsGatewayServer(socket_path, gateway)
    thread = threading.Thread(target=server.serve_forever, name="cultra-orats-daemon")
    thread.start()
    completed: Dict[str, GatewayResult] = {}
    errors: Dict[str, str] = {}

    def execute_one(identifier: str) -> GatewayResult:
        public = OratsGatewayClient(
            socket_path, timeout_seconds=float(client_timeout_seconds)
        ).execute(identifier)
        return GatewayResult.from_public_dict(public)

    try:
        remaining = list(identifiers)
        while remaining:
            identifier = remaining.pop(0)
            try:
                result = execute_one(identifier)
                completed[identifier] = result
            except BaseException as exc:
                errors[identifier] = "%s: %s" % (type(exc).__name__, str(exc))
                return completed, errors
            if not result.cache_hit:
                break
        if remaining:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending = {
                    pool.submit(execute_one, identifier): identifier
                    for identifier in remaining
                }
                for future in as_completed(pending):
                    identifier = pending[future]
                    try:
                        completed[identifier] = future.result()
                    except BaseException as exc:
                        errors[identifier] = "%s: %s" % (
                            type(exc).__name__,
                            str(exc),
                        )
        return completed, errors
    finally:
        server.shutdown()
        thread.join()
        server.close()
