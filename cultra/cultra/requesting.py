"""Immutable, bounded ORATS request planning for Cultra.

This module is intentionally network- and credential-free.  It owns the
allowlist and turns already-selected entities into a complete plan before the
gateway is allowed to send anything.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


class PlanningError(ValueError):
    """The proposed request plan is unsafe or internally inconsistent."""


class EndpointPolicyError(PlanningError):
    """An endpoint, method, or run-type combination is not allowed."""


class SecretMaterialError(PlanningError):
    """Credential-like material was presented to a tokenless planner."""


class RunType(str, Enum):
    EOD = "eod"
    MORNING_DELTA = "morning_delta"
    AFTERNOON_DELTA = "afternoon_delta"
    ENTITLEMENT_DISCOVERY = "entitlement_discovery"
    HISTORICAL_BACKFILL = "historical_backfill"
    BACKTEST_VALIDATION = "backtest_validation"
    SCANNER_RESEARCH = "scanner_research"


class Endpoint(str, Enum):
    CORES = "/datav2/cores"
    SUMMARIES = "/datav2/summaries"
    MONIES_IMPLIED = "/datav2/monies/implied"
    MONIES_FORECAST = "/datav2/monies/forecast"
    EXACT_OPTIONS = "/datav2/strikes/options"
    STRIKES = "/datav2/strikes"
    IVRANK = "/datav2/ivrank"
    HIST_CORES = "/datav2/hist/cores"
    HIST_DAILIES = "/datav2/hist/dailies"
    HIST_SUMMARIES = "/datav2/hist/summaries"
    HIST_STRIKES = "/datav2/hist/strikes"
    HIST_STRIKES_OPTIONS = "/datav2/hist/strikes/options"
    HIST_SPLITS = "/datav2/hist/splits"
    BACKTEST = "/backtest"
    BACKTEST_STATUS = "/backtest/status"
    SCANNER = "/scanner"


class ContingencyKind(str, Enum):
    GROUPED_MISSING_RECOVERY = "GROUPED_MISSING_RECOVERY"
    SPLIT_CHILD = "SPLIT_CHILD"


@dataclass(frozen=True)
class EndpointRule:
    method: str
    run_types: Tuple[RunType, ...]
    entity_kind: str
    batch_size: int
    conditional: bool = False
    contract_status: str = "DOCUMENTED_NOT_PROBED"


_DAILY_RUNS = (
    RunType.EOD,
    RunType.MORNING_DELTA,
    RunType.AFTERNOON_DELTA,
    RunType.ENTITLEMENT_DISCOVERY,
)

# ORATS' current public delayed and historical documentation caps the ticker
# endpoints at ten comma-delimited underlyings.  The original 80-120 planning
# assumption is therefore not executable unless a separately authorized
# provider-contract probe proves a different account-specific contract.  Never
# inflate these limits merely to make a request forecast look smaller.
#
# Exact-option lookup is GET in the currently documented contract.  If a
# future entitlement probe establishes a different method, that is a
# schema-versioned policy change rather than an ad-hoc transport fallback.
ENDPOINT_RULES: Mapping[Endpoint, EndpointRule] = {
    Endpoint.CORES: EndpointRule("GET", _DAILY_RUNS, "ticker", 10),
    Endpoint.SUMMARIES: EndpointRule("GET", _DAILY_RUNS, "ticker", 10),
    Endpoint.MONIES_IMPLIED: EndpointRule("GET", _DAILY_RUNS, "ticker", 10),
    Endpoint.MONIES_FORECAST: EndpointRule("GET", _DAILY_RUNS, "ticker", 10),
    Endpoint.EXACT_OPTIONS: EndpointRule("GET", _DAILY_RUNS, "occ", 100),
    Endpoint.STRIKES: EndpointRule("GET", _DAILY_RUNS, "ticker", 10, True),
    Endpoint.IVRANK: EndpointRule("GET", _DAILY_RUNS, "ticker", 10, True),
    Endpoint.HIST_CORES: EndpointRule(
        "GET", (RunType.HISTORICAL_BACKFILL,), "ticker", 10
    ),
    Endpoint.HIST_DAILIES: EndpointRule(
        "GET", (RunType.HISTORICAL_BACKFILL,), "ticker", 10
    ),
    Endpoint.HIST_SUMMARIES: EndpointRule(
        "GET", (RunType.HISTORICAL_BACKFILL,), "ticker", 10
    ),
    Endpoint.HIST_STRIKES: EndpointRule(
        "GET", (RunType.HISTORICAL_BACKFILL,), "ticker", 10
    ),
    # The provider contract accepts one underlying/expiration/strike key and
    # returns that exact strike's complete EOD history.  This is intentionally
    # offline-only and must be deduplicated before plan freeze.
    Endpoint.HIST_STRIKES_OPTIONS: EndpointRule(
        "GET", (RunType.HISTORICAL_BACKFILL,), "ticker", 1
    ),
    Endpoint.HIST_SPLITS: EndpointRule(
        "GET", (RunType.HISTORICAL_BACKFILL,), "ticker", 10
    ),
    Endpoint.BACKTEST: EndpointRule(
        "POST", (RunType.BACKTEST_VALIDATION,), "job", 100
    ),
    Endpoint.BACKTEST_STATUS: EndpointRule(
        "GET", (RunType.BACKTEST_VALIDATION,), "job", 100
    ),
    Endpoint.SCANNER: EndpointRule(
        "POST", (RunType.SCANNER_RESEARCH,), "job", 100
    ),
}

# No provider idempotency contract has been authorized or entitlement-probed.
# Adding one is a frozen provider-contract change, never a caller assertion.
IDEMPOTENCY_CONTRACTS: Mapping[Endpoint, str] = {}


@dataclass(frozen=True)
class BudgetPolicy:
    target: int
    logical_cap: int
    hard_cap: int


BUDGET_POLICIES: Mapping[RunType, BudgetPolicy] = {
    # The approved daily contract is a 25-request target and a 60-request
    # frozen logical ceiling.  Request 99 may be charged; request 100 is never
    # admissible.  A larger historical campaign belongs to its own run type
    # and may not silently widen this daily envelope.
    RunType.EOD: BudgetPolicy(target=25, logical_cap=60, hard_cap=99),
    RunType.MORNING_DELTA: BudgetPolicy(target=0, logical_cap=0, hard_cap=1),
    RunType.AFTERNOON_DELTA: BudgetPolicy(target=10, logical_cap=15, hard_cap=30),
    RunType.ENTITLEMENT_DISCOVERY: BudgetPolicy(
        target=12, logical_cap=15, hard_cap=15
    ),
    RunType.HISTORICAL_BACKFILL: BudgetPolicy(
        target=60, logical_cap=90, hard_cap=90
    ),
    RunType.BACKTEST_VALIDATION: BudgetPolicy(
        target=25, logical_cap=40, hard_cap=40
    ),
    RunType.SCANNER_RESEARCH: BudgetPolicy(target=10, logical_cap=15, hard_cap=40),
}

REFERENCE_MAX_CORE_SYMBOLS = 600
REFERENCE_MAX_SUMMARY_SYMBOLS = 120
REFERENCE_MAX_MONIES_SYMBOLS = 40
REFERENCE_MAX_EXACT_CONTRACTS = 250
REFERENCE_MAX_BASE_REQUESTS = 60
REFERENCE_MAX_TOTAL_LOGICAL = 60
REFERENCE_MAX_RETRY_RESERVE = 0
REFERENCE_MAX_CHARGED_ATTEMPTS = 99
PROTOCOL_MAX_ATTEMPTS = 99
MAX_GET_URL_BYTES = 6000
GET_TOKEN_QUERY_RESERVE_BYTES = 512
MAX_SINGLE_RESPONSE_BYTES = 25_000_000
MAX_TOTAL_RESPONSE_BYTES_EOD = 250_000_000
MAX_GROUPED_MISSING_RECOVERIES = 1
MAX_SPLIT_CHILDREN = 6
MAX_SPLIT_DEPTH = 2
MAX_FILTERED_STRIKES_ROWS = 100_000

_OCC_RE = re.compile(r"^[A-Z0-9.]{1,6}\s*\d{6}[CP]\d{8}$")
_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")
_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_TRADE_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EXPIRATION_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SECRET_KEYS = frozenset(
    {
        "token",
        "apikey",
        "api_key",
        "authorization",
        "access_token",
        "orats_token",
    }
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _normalize_pairs(
    value: Optional[Mapping[str, Any]], *, label: str
) -> Tuple[Tuple[str, Any], ...]:
    if not value:
        return ()
    normalized = []
    for key, raw_value in value.items():
        key_text = str(key)
        if key_text.lower() in _SECRET_KEYS or "token" in key_text.lower():
            raise SecretMaterialError("%s contains a credential-like key" % label)
        if isinstance(raw_value, Mapping):
            nested = dict(_normalize_pairs(raw_value, label=label))
            clean_value: Any = nested
        elif isinstance(raw_value, (list, tuple, set, frozenset)):
            clean_value = tuple(sorted(str(item) for item in raw_value))
        elif raw_value is None or isinstance(raw_value, (bool, int, float, str)):
            clean_value = raw_value
        else:
            raise PlanningError("%s contains an unsupported value type" % label)
        normalized.append((key_text, clean_value))
    return tuple(sorted(normalized, key=lambda item: item[0]))


def normalize_entities(values: Iterable[str], kind: str) -> Tuple[str, ...]:
    """Return a deterministic, validated, duplicate-free entity tuple."""

    entities = tuple(sorted({str(value).strip().upper() for value in values}))
    if not entities:
        raise PlanningError("a planned request must contain at least one entity")
    validator = _OCC_RE if kind == "occ" else _TICKER_RE
    for entity in entities:
        if not validator.fullmatch(entity):
            raise PlanningError("invalid %s identifier: %s" % (kind, entity))
    return entities


def deterministic_batches(
    values: Iterable[str], batch_size: int, *, kind: str = "ticker"
) -> Tuple[Tuple[str, ...], ...]:
    if batch_size < 2:
        raise PlanningError("batch_size must be at least 2; per-entity calls are blocked")
    entities = normalize_entities(values, kind)
    return tuple(
        entities[index : index + batch_size]
        for index in range(0, len(entities), batch_size)
    )


def _strict_csv_values(raw: Any, *, label: str, uppercase: bool) -> Tuple[str, ...]:
    """Parse only an explicit CSV string or finite string sequence."""

    if isinstance(raw, str):
        values = raw.split(",")
    elif isinstance(raw, (list, tuple, set, frozenset)):
        values = raw
    else:
        raise EndpointPolicyError("/datav2/strikes %s must be bounded text" % label)
    normalized = []
    for item in values:
        if not isinstance(item, str):
            raise EndpointPolicyError("/datav2/strikes %s must contain text" % label)
        clean = item.strip()
        if not clean:
            raise EndpointPolicyError("/datav2/strikes %s cannot be empty" % label)
        normalized.append(clean.upper() if uppercase else clean)
    result = tuple(sorted(set(normalized)))
    if len(result) != len(normalized):
        raise EndpointPolicyError("/datav2/strikes %s cannot contain duplicates" % label)
    return result


def validate_endpoint(
    endpoint: Endpoint,
    method: str,
    run_type: RunType,
    *,
    conditional_authorized: bool = False,
    params: Optional[Mapping[str, Any]] = None,
    expected_rows: int = 0,
    entities: Sequence[str] = (),
    fields: Sequence[str] = (),
) -> None:
    try:
        rule = ENDPOINT_RULES[Endpoint(endpoint)]
    except (KeyError, ValueError) as exc:
        raise EndpointPolicyError("endpoint is not in the Cultra allowlist") from exc
    normalized_method = method.upper()
    if normalized_method != rule.method:
        raise EndpointPolicyError(
            "%s requires %s, not %s" % (endpoint.value, rule.method, normalized_method)
        )
    if run_type not in rule.run_types:
        raise EndpointPolicyError(
            "%s is not allowed for run type %s" % (endpoint.value, run_type.value)
        )
    if rule.conditional and not conditional_authorized:
        raise EndpointPolicyError(
            "%s requires explicit conditional authorization" % endpoint.value
        )
    if not entities or len(entities) > rule.batch_size:
        raise EndpointPolicyError(
            "%s exceeds its frozen bulk-request size" % endpoint.value
        )
    if endpoint == Endpoint.STRIKES:
        supplied = params or {}
        mandatory = {"ticker", "fields", "dte_min", "dte_max", "delta_min", "delta_max"}
        if set(supplied) != mandatory or expected_rows <= 0:
            raise EndpointPolicyError(
                "/datav2/strikes requires tickers, fields, bounded DTE/delta, "
                "and a positive expected-row budget"
            )
        supplied_tickers = _strict_csv_values(
            supplied["ticker"], label="ticker bounds", uppercase=True
        )
        supplied_fields = _strict_csv_values(
            supplied["fields"], label="field bounds", uppercase=False
        )
        if not supplied_tickers or supplied_tickers != tuple(entities):
            raise EndpointPolicyError("/datav2/strikes ticker bounds must match planned entities")
        if not supplied_fields or supplied_fields != tuple(fields):
            raise EndpointPolicyError("/datav2/strikes fields must match the frozen field profile")
        if len(supplied_fields) > 64 or any(
            not _FIELD_RE.fullmatch(value) for value in supplied_fields
        ):
            raise EndpointPolicyError("/datav2/strikes field bounds are invalid")
        dte_min = supplied["dte_min"]
        dte_max = supplied["dte_max"]
        delta_min = supplied["delta_min"]
        delta_max = supplied["delta_max"]
        if (
            isinstance(dte_min, bool)
            or isinstance(dte_max, bool)
            or not isinstance(dte_min, int)
            or not isinstance(dte_max, int)
            or not 0 <= dte_min <= dte_max <= 3_650
        ):
            raise EndpointPolicyError("/datav2/strikes DTE bounds are invalid")
        if (
            isinstance(delta_min, bool)
            or isinstance(delta_max, bool)
            or not isinstance(delta_min, (int, float))
            or not isinstance(delta_max, (int, float))
            or not math.isfinite(float(delta_min))
            or not math.isfinite(float(delta_max))
            or not -1.0 <= float(delta_min) <= float(delta_max) <= 1.0
        ):
            raise EndpointPolicyError("/datav2/strikes delta bounds are invalid")
        if expected_rows > MAX_FILTERED_STRIKES_ROWS:
            raise EndpointPolicyError("/datav2/strikes expected-row budget is too large")
    if endpoint == Endpoint.HIST_STRIKES:
        supplied = params or {}
        if set(supplied) != {"tradeDate", "dte", "delta"}:
            raise EndpointPolicyError(
                "/datav2/hist/strikes requires one tradeDate and bounded dte/delta filters"
            )
        trade_date = supplied["tradeDate"]
        if not isinstance(trade_date, str) or not _TRADE_DATE_RE.fullmatch(trade_date):
            raise EndpointPolicyError("historical strikes tradeDate must use YYYY-MM-DD")
        dte = _strict_csv_values(supplied["dte"], label="dte bounds", uppercase=False)
        delta = _strict_csv_values(
            supplied["delta"], label="delta bounds", uppercase=False
        )
        if len(dte) != 2 or len(delta) != 2:
            raise EndpointPolicyError("historical strikes filters require exactly two bounds")
        try:
            dte_min, dte_max = sorted(int(value) for value in dte)
            delta_min, delta_max = sorted(float(value) for value in delta)
        except (TypeError, ValueError) as exc:
            raise EndpointPolicyError("historical strikes filters are malformed") from exc
        if not 0 <= dte_min <= dte_max <= 3650:
            raise EndpointPolicyError("historical strikes DTE bounds are invalid")
        if not (
            math.isfinite(delta_min)
            and math.isfinite(delta_max)
            and 0.0 <= delta_min <= delta_max <= 1.0
        ):
            raise EndpointPolicyError("historical strikes delta bounds are invalid")
        if expected_rows > MAX_FILTERED_STRIKES_ROWS:
            raise EndpointPolicyError(
                "/datav2/hist/strikes expected-row budget is too large"
            )
    if endpoint == Endpoint.HIST_STRIKES_OPTIONS:
        supplied = params or {}
        if set(supplied) not in (
            {"expirDate", "strike"},
            {"expirDate", "strike", "tradeDate"},
        ):
            raise EndpointPolicyError(
                "/datav2/hist/strikes/options requires one expiry and strike"
            )
        if len(entities) != 1:
            raise EndpointPolicyError(
                "/datav2/hist/strikes/options accepts one underlying per request"
            )
        expiration = supplied.get("expirDate")
        if not isinstance(expiration, str) or not _EXPIRATION_DATE_RE.fullmatch(
            expiration
        ):
            raise EndpointPolicyError("historical exact strike expiry is invalid")
        try:
            strike = float(supplied.get("strike"))
        except (TypeError, ValueError) as exc:
            raise EndpointPolicyError("historical exact strike is invalid") from exc
        if not math.isfinite(strike) or strike <= 0.0:
            raise EndpointPolicyError("historical exact strike is invalid")
        trade_date = supplied.get("tradeDate")
        if trade_date is not None and (
            not isinstance(trade_date, str)
            or not _TRADE_DATE_RE.fullmatch(trade_date)
        ):
            raise EndpointPolicyError("historical exact strike tradeDate is invalid")


@dataclass(frozen=True)
class PlannedRequest:
    logical_request_id: str
    endpoint: Endpoint
    method: str
    entities: Tuple[str, ...]
    field_profile: str
    fields: Tuple[str, ...]
    purpose: str
    cache_policy: str
    expected_vintage: str
    expected_rows: int
    expected_bytes: int
    max_response_bytes: int = MAX_SINGLE_RESPONSE_BYTES
    required: bool = True
    contingency: bool = False
    retry_limit: int = 0
    idempotency_contract: Optional[str] = None
    conditional_authorized: bool = False
    contingency_kind: Optional[ContingencyKind] = None
    contingency_parent_id: Optional[str] = None
    split_depth: int = 0
    params: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    body: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.logical_request_id or len(self.logical_request_id) > 128:
            raise PlanningError("logical_request_id is missing or too long")
        if self.method != self.method.upper():
            raise PlanningError("HTTP method must be uppercase")
        if not self.field_profile or not self.fields:
            raise PlanningError("every planned bulk request needs a field profile and fields")
        if self.expected_rows <= 0 or self.expected_bytes <= 0:
            raise PlanningError("row and byte estimates must be positive and bounded")
        if not 0 < self.max_response_bytes <= MAX_SINGLE_RESPONSE_BYTES:
            raise PlanningError("max_response_bytes exceeds the client safeguard")
        if self.expected_bytes > self.max_response_bytes:
            raise PlanningError("expected response bytes exceed the per-request safeguard")
        if self.retry_limit != 0:
            raise PlanningError("automatic ORATS retries are disabled")
        rule = ENDPOINT_RULES.get(self.endpoint)
        if rule is None:
            raise EndpointPolicyError("endpoint is not allowlisted")
        frozen_idempotency = IDEMPOTENCY_CONTRACTS.get(self.endpoint)
        if self.idempotency_contract != frozen_idempotency:
            raise PlanningError("idempotency contract is not in the frozen provider registry")
        if self.method != "GET" and self.retry_limit and not frozen_idempotency:
            raise PlanningError("non-idempotent endpoint retries are prohibited")
        canonical_entities = normalize_entities(self.entities, rule.entity_kind)
        if canonical_entities != self.entities:
            raise PlanningError("entities must already be normalized and sorted")
        if tuple(sorted(set(self.fields))) != self.fields:
            raise PlanningError("fields must be unique and sorted")
        if not self.contingency:
            if (
                self.contingency_kind is not None
                or self.contingency_parent_id is not None
                or self.split_depth != 0
            ):
                raise PlanningError("base requests cannot carry contingency metadata")
        else:
            if not isinstance(self.contingency_kind, ContingencyKind):
                raise PlanningError("contingency requests require a frozen kind")
            if not 0 <= self.split_depth <= MAX_SPLIT_DEPTH:
                raise PlanningError("contingency split depth exceeds the frozen maximum")

    @property
    def fingerprint(self) -> str:
        return request_fingerprint(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logical_request_id": self.logical_request_id,
            "endpoint": self.endpoint.value,
            "contract_status": ENDPOINT_RULES[self.endpoint].contract_status,
            "method": self.method,
            "entities": list(self.entities),
            "field_profile": self.field_profile,
            "fields": list(self.fields),
            "purpose": self.purpose,
            "cache_policy": self.cache_policy,
            "expected_vintage": self.expected_vintage,
            "expected_rows": self.expected_rows,
            "expected_bytes": self.expected_bytes,
            "max_response_bytes": self.max_response_bytes,
            "required": self.required,
            "contingency": self.contingency,
            "retry_limit": self.retry_limit,
            "idempotency_contract": self.idempotency_contract,
            "conditional_authorized": self.conditional_authorized,
            "contingency_kind": (
                self.contingency_kind.value if self.contingency_kind is not None else None
            ),
            "contingency_parent_id": self.contingency_parent_id,
            "split_depth": self.split_depth,
            "params": dict(self.params),
            "body": dict(self.body),
        }


def request_fingerprint(request: PlannedRequest) -> str:
    """Hash only canonical request semantics; credentials are impossible here."""

    payload = request.to_dict()
    payload.pop("logical_request_id", None)
    payload.pop("purpose", None)
    payload.pop("expected_rows", None)
    payload.pop("expected_bytes", None)
    payload.pop("max_response_bytes", None)
    payload.pop("required", None)
    payload.pop("contingency", None)
    payload.pop("retry_limit", None)
    payload.pop("idempotency_contract", None)
    payload.pop("contingency_kind", None)
    payload.pop("contingency_parent_id", None)
    payload.pop("split_depth", None)
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class RequestPlan:
    run_id: str
    run_type: RunType
    requests: Tuple[PlannedRequest, ...]
    target: int
    hard_cap: int
    retry_reserve: int
    schema_version: str = "CULTRA_REQUEST_PLAN_V2"
    campaign_id: Optional[str] = None
    campaign_hard_cap: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.run_id or len(self.run_id) > 128:
            raise PlanningError("run_id is missing or too long")
        if (self.campaign_id is None) != (self.campaign_hard_cap is None):
            raise PlanningError("campaign id and cap must be supplied together")
        if self.campaign_id is not None:
            if not re.fullmatch(r"[A-Za-z0-9_.:@\-]{1,128}", self.campaign_id):
                raise PlanningError("campaign_id is invalid")
            if (
                isinstance(self.campaign_hard_cap, bool)
                or not isinstance(self.campaign_hard_cap, int)
                or not 1 <= self.campaign_hard_cap <= PROTOCOL_MAX_ATTEMPTS
            ):
                raise PlanningError("campaign hard cap must be between 1 and 99")
            if self.hard_cap > self.campaign_hard_cap:
                raise PlanningError("run hard cap cannot exceed its campaign cap")
        policy = BUDGET_POLICIES[self.run_type]
        if not 1 <= self.hard_cap <= min(policy.hard_cap, PROTOCOL_MAX_ATTEMPTS):
            raise PlanningError("hard cap exceeds the immutable run-type ceiling")
        if self.retry_reserve != 0:
            raise PlanningError("automatic ORATS retry reserve must be zero")
        ids = [item.logical_request_id for item in self.requests]
        if len(ids) != len(set(ids)):
            raise PlanningError("logical request IDs must be unique")
        base_count = sum(not item.contingency for item in self.requests)
        total_count = len(self.requests)
        if total_count > policy.logical_cap:
            raise PlanningError("plan exceeds the run-type logical request cap")
        if total_count + self.retry_reserve > self.hard_cap:
            raise PlanningError("worst admitted attempts exceed the run hard cap")
        if self.run_type == RunType.EOD:
            if base_count > REFERENCE_MAX_BASE_REQUESTS:
                raise PlanningError("EOD base plan exceeds 60 logical requests")
            if total_count > REFERENCE_MAX_TOTAL_LOGICAL:
                raise PlanningError("EOD plan including contingencies exceeds 60")
            if self.retry_reserve > REFERENCE_MAX_RETRY_RESERVE:
                raise PlanningError("EOD retry reserve exceeds 39")
            if total_count + self.retry_reserve > REFERENCE_MAX_CHARGED_ATTEMPTS:
                raise PlanningError("EOD worst admitted attempt envelope exceeds 99")
        retry_capacity = sum(item.retry_limit for item in self.requests)
        if retry_capacity and self.retry_reserve <= 0:
            raise PlanningError("retry-capable requests require a global retry reserve")
        for item in self.requests:
            validate_endpoint(
                item.endpoint,
                item.method,
                self.run_type,
                conditional_authorized=item.conditional_authorized,
                params=dict(item.params),
                expected_rows=item.expected_rows,
                entities=item.entities,
                fields=item.fields,
            )
            if item.method == "GET" and planned_get_url_bytes(item) > MAX_GET_URL_BYTES:
                raise PlanningError("planned GET exceeds the frozen encoded URL ceiling")
        if self.run_type == RunType.EOD:
            expected_total_bytes = sum(item.expected_bytes for item in self.requests)
            if expected_total_bytes > MAX_TOTAL_RESPONSE_BYTES_EOD:
                raise PlanningError("EOD expected response bytes exceed the total safeguard")
            reserved_total_bytes = sum(
                item.max_response_bytes for item in self.requests
            )
            if reserved_total_bytes > MAX_TOTAL_RESPONSE_BYTES_EOD:
                raise PlanningError(
                    "EOD response-byte reservations exceed the total safeguard"
                )
        if any(item.contingency for item in self.requests):
            raise PlanningError(
                "automatic recovery contingencies are disabled; freeze a new plan instead"
            )
        grouped_recoveries = tuple(
            item
            for item in self.requests
            if item.contingency_kind == ContingencyKind.GROUPED_MISSING_RECOVERY
        )
        split_children = tuple(
            item
            for item in self.requests
            if item.contingency_kind == ContingencyKind.SPLIT_CHILD
        )
        if len(grouped_recoveries) > MAX_GROUPED_MISSING_RECOVERIES:
            raise PlanningError("only one grouped missing-symbol recovery is allowed")
        if len(split_children) > MAX_SPLIT_CHILDREN:
            raise PlanningError("split-child contingency capacity exceeds six")
        by_id = {item.logical_request_id: item for item in self.requests}
        base_coverage = {}
        for item in self.requests:
            group = (
                item.endpoint.value,
                item.method,
                item.field_profile,
                item.expected_vintage,
                item.params,
                item.body,
            )
            if not item.contingency:
                prior = base_coverage.setdefault(group, set())
                overlap = prior.intersection(item.entities)
                if overlap:
                    raise PlanningError(
                        "overlapping entities must be deduplicated before plan freeze"
                    )
                prior.update(item.entities)
                continue
            if len(item.entities) < 2:
                raise PlanningError("contingencies cannot degenerate into per-entity calls")
            if item.contingency_kind == ContingencyKind.GROUPED_MISSING_RECOVERY:
                if item.contingency_parent_id is not None or item.split_depth != 0:
                    raise PlanningError("grouped recovery cannot be a split child")
                if not set(item.entities).issubset(base_coverage.get(group, set())):
                    raise PlanningError("grouped recovery must be a subset of frozen base coverage")
                continue
            parent = by_id.get(item.contingency_parent_id or "")
            if parent is None or parent.logical_request_id == item.logical_request_id:
                raise PlanningError("split child requires a frozen parent request")
            parent_group = (
                parent.endpoint.value,
                parent.method,
                parent.field_profile,
                parent.expected_vintage,
                parent.params,
                parent.body,
            )
            if parent_group != group or not set(item.entities).issubset(set(parent.entities)):
                raise PlanningError("split child must be a same-profile subset of its parent")
            if item.split_depth != parent.split_depth + 1:
                raise PlanningError("split-child depth must extend its parent by exactly one")
        children_by_parent: Dict[str, set] = {}
        for child in split_children:
            prior = children_by_parent.setdefault(child.contingency_parent_id or "", set())
            if prior.intersection(child.entities):
                raise PlanningError("sibling split children cannot overlap")
            prior.update(child.entities)

    @property
    def base_count(self) -> int:
        return sum(not item.contingency for item in self.requests)

    @property
    def contingency_count(self) -> int:
        return sum(item.contingency for item in self.requests)

    @property
    def logical_count(self) -> int:
        return len(self.requests)

    @property
    def worst_charged_attempts(self) -> int:
        return self.logical_count + self.retry_reserve

    @property
    def plan_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict(include_hash=False))).hexdigest()

    def get(self, logical_request_id: str) -> PlannedRequest:
        for item in self.requests:
            if item.logical_request_id == logical_request_id:
                return item
        raise KeyError("request ID is not present in the frozen plan")

    def to_dict(self, *, include_hash: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "run_type": self.run_type.value,
            "target": self.target,
            "hard_cap": self.hard_cap,
            "retry_reserve": self.retry_reserve,
            "campaign_id": self.campaign_id,
            "campaign_hard_cap": self.campaign_hard_cap,
            "base_count": self.base_count,
            "contingency_count": self.contingency_count,
            "logical_count": self.logical_count,
            "worst_charged_attempts": self.worst_charged_attempts,
            "requests": [item.to_dict() for item in self.requests],
        }
        if include_hash:
            result["plan_hash"] = hashlib.sha256(
                _canonical_json(result)
            ).hexdigest()
        return result


def request_query_parameters(request: PlannedRequest) -> Dict[str, Any]:
    """Build the public, credential-free query shape used by the gateway."""

    parameters: Dict[str, Any] = dict(request.params)
    parameters["fields"] = ",".join(request.fields)
    rule = ENDPOINT_RULES[request.endpoint]
    if request.endpoint == Endpoint.EXACT_OPTIONS:
        parameters["tickers"] = ",".join(request.entities)
    elif rule.entity_kind == "ticker":
        parameters["ticker"] = ",".join(request.entities)
    elif request.method == "GET":
        parameters["ids"] = ",".join(request.entities)
    return parameters


def planned_get_url_bytes(request: PlannedRequest) -> int:
    """Conservatively preflight a GET without loading or hashing the token."""

    if request.method != "GET":
        return 0
    query = urllib.parse.urlencode(request_query_parameters(request), doseq=True)
    prefix = "https://api.orats.io%s?%s&token=" % (request.endpoint.value, query)
    return len(prefix.encode("utf-8")) + GET_TOKEN_QUERY_RESERVE_BYTES


def _make_requests(
    *,
    run_id: str,
    endpoint: Endpoint,
    entities: Sequence[str],
    fields: Sequence[str],
    field_profile: str,
    purpose: str,
    vintage: str,
    batch_size: int,
    expected_rows_per_entity: int,
    expected_bytes_per_entity: int,
) -> Tuple[PlannedRequest, ...]:
    rule = ENDPOINT_RULES[endpoint]
    result = []
    batches = deterministic_batches(entities, batch_size, kind=rule.entity_kind)
    normalized_fields = tuple(sorted(set(str(item) for item in fields)))
    for index, batch in enumerate(batches, 1):
        identity = {
            "run_id": run_id,
            "endpoint": endpoint.value,
            "field_profile": field_profile,
            "batch": batch,
            "vintage": vintage,
        }
        suffix = hashlib.sha256(_canonical_json(identity)).hexdigest()[:12]
        request_id = "%s-%02d-%s" % (
            endpoint.name.lower().replace("_", "-"),
            index,
            suffix,
        )
        result.append(
            PlannedRequest(
                logical_request_id=request_id,
                endpoint=endpoint,
                method=rule.method,
                entities=batch,
                field_profile=field_profile,
                fields=normalized_fields,
                purpose=purpose,
                cache_policy="ONCE_PER_EXPECTED_VINTAGE",
                expected_vintage=vintage,
                expected_rows=max(1, len(batch) * expected_rows_per_entity),
                expected_bytes=max(1, len(batch) * expected_bytes_per_entity),
                max_response_bytes=min(
                    MAX_SINGLE_RESPONSE_BYTES,
                    max(1_000_000, len(batch) * expected_bytes_per_entity * 4),
                ),
                retry_limit=0,
            )
        )
    return tuple(result)


def build_reference_eod_plan(
    *,
    run_id: str,
    core_symbols: Sequence[str],
    summary_symbols: Sequence[str] = (),
    monies_symbols: Sequence[str] = (),
    option_symbols: Sequence[str] = (),
    expected_vintage: str,
    core_fields: Sequence[str] = ("ticker", "tradeDate", "updatedAt"),
    summary_fields: Sequence[str] = ("ticker", "tradeDate", "updatedAt"),
    monies_implied_fields: Sequence[str] = (
        "ticker",
        "tradeDate",
        "expirDate",
        "updatedAt",
    ),
    monies_forecast_fields: Sequence[str] = (
        "ticker",
        "tradeDate",
        "expirDate",
        "updatedAt",
    ),
    exact_option_fields: Sequence[str] = (
        "optionSymbol",
        "ticker",
        "tradeDate",
        "updatedAt",
    ),
    retry_reserve: Optional[int] = None,
    hard_cap: int = PROTOCOL_MAX_ATTEMPTS,
) -> RequestPlan:
    """Build the bounded, full-eligible-universe EOD request plan.

    Selection happens before this function.  Oversized inputs are rejected,
    never silently alphabetically truncated.
    """

    limits = (
        ("core", core_symbols, REFERENCE_MAX_CORE_SYMBOLS),
        ("summary", summary_symbols, REFERENCE_MAX_SUMMARY_SYMBOLS),
        ("monies", monies_symbols, REFERENCE_MAX_MONIES_SYMBOLS),
        ("exact contracts", option_symbols, REFERENCE_MAX_EXACT_CONTRACTS),
    )
    for label, values, maximum in limits:
        if len(set(values)) > maximum:
            raise PlanningError("%s selection exceeds the frozen funnel cap %d" % (label, maximum))
    requests = []
    if core_symbols:
        requests.extend(
            _make_requests(
                run_id=run_id,
                endpoint=Endpoint.CORES,
                entities=core_symbols,
                fields=core_fields,
                field_profile="CORE_SCREEN_V1",
                purpose="shared full-universe pre-screen enrichment",
                vintage=expected_vintage,
                batch_size=10,
                expected_rows_per_entity=1,
                expected_bytes_per_entity=3_000,
            )
        )
    if summary_symbols:
        requests.extend(
            _make_requests(
                run_id=run_id,
                endpoint=Endpoint.SUMMARIES,
                entities=summary_symbols,
                fields=summary_fields,
                field_profile="SUMMARY_ENRICH_V1",
                purpose="shared shortlist enrichment",
                vintage=expected_vintage,
                batch_size=10,
                expected_rows_per_entity=1,
                expected_bytes_per_entity=2_000,
            )
        )
    if monies_symbols:
        for endpoint, profile, fields in (
            (Endpoint.MONIES_IMPLIED, "MONEY_IMPLIED_V1", monies_implied_fields),
            (Endpoint.MONIES_FORECAST, "MONEY_FORECAST_V1", monies_forecast_fields),
        ):
            requests.extend(
                _make_requests(
                    run_id=run_id,
                    endpoint=endpoint,
                    entities=monies_symbols,
                    fields=fields,
                    field_profile=profile,
                    purpose="shared finalist volatility-surface enrichment",
                    vintage=expected_vintage,
                    batch_size=10,
                    expected_rows_per_entity=8,
                    expected_bytes_per_entity=12_000,
                )
            )
    if option_symbols:
        requests.extend(
            _make_requests(
                run_id=run_id,
                endpoint=Endpoint.EXACT_OPTIONS,
                entities=option_symbols,
                fields=exact_option_fields,
                field_profile="EXACT_OPTION_V1",
                purpose="deduplicated exact-contract analytical enrichment",
                vintage=expected_vintage,
                batch_size=100,
                expected_rows_per_entity=1,
                expected_bytes_per_entity=2_500,
            )
        )
    effective_retry_reserve = 0 if retry_reserve is None else retry_reserve
    return RequestPlan(
        run_id=run_id,
        run_type=RunType.EOD,
        requests=tuple(requests),
        target=25,
        hard_cap=hard_cap,
        retry_reserve=effective_retry_reserve,
    )


def make_planned_request(
    *,
    logical_request_id: str,
    endpoint: Endpoint,
    run_type: RunType,
    entities: Sequence[str],
    fields: Sequence[str],
    field_profile: str,
    purpose: str,
    expected_vintage: str,
    expected_rows: int,
    expected_bytes: int,
    max_response_bytes: int = MAX_SINGLE_RESPONSE_BYTES,
    required: bool = True,
    contingency: bool = False,
    retry_limit: int = 0,
    idempotency_contract: Optional[str] = None,
    conditional_authorized: bool = False,
    contingency_kind: Optional[ContingencyKind] = None,
    contingency_parent_id: Optional[str] = None,
    split_depth: int = 0,
    params: Optional[Mapping[str, Any]] = None,
    body: Optional[Mapping[str, Any]] = None,
) -> PlannedRequest:
    """Safe public constructor for non-reference, still-allowlisted requests."""

    endpoint = Endpoint(endpoint)
    rule = ENDPOINT_RULES[endpoint]
    normalized_params = _normalize_pairs(params, label="params")
    normalized_body = _normalize_pairs(body, label="body")
    item = PlannedRequest(
        logical_request_id=logical_request_id,
        endpoint=endpoint,
        method=rule.method,
        entities=normalize_entities(entities, rule.entity_kind),
        field_profile=field_profile,
        fields=tuple(sorted(set(fields))),
        purpose=purpose,
        cache_policy="ONCE_PER_EXPECTED_VINTAGE",
        expected_vintage=expected_vintage,
        expected_rows=expected_rows,
        expected_bytes=expected_bytes,
        max_response_bytes=max_response_bytes,
        required=required,
        contingency=contingency,
        retry_limit=retry_limit,
        idempotency_contract=idempotency_contract,
        conditional_authorized=conditional_authorized,
        contingency_kind=contingency_kind,
        contingency_parent_id=contingency_parent_id,
        split_depth=split_depth,
        params=normalized_params,
        body=normalized_body,
    )
    validate_endpoint(
        endpoint,
        item.method,
        run_type,
        conditional_authorized=conditional_authorized,
        params=dict(normalized_params),
        expected_rows=expected_rows,
        entities=item.entities,
        fields=item.fields,
    )
    return item
