"""Cache inventory and request-budget planning with no fetch capability."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set, Tuple

from codexswing.schemas.source import SourceRecord
from codexswing.store.immutable import read_batch
from codexswing.v5.spec import V5ResearchSpec


class NetworkUseForbidden(RuntimeError):
    """Raised when a caller attempts to authorize network use in v0.5."""


SOURCE_TO_ENDPOINT = {
    "orats_hist_cores": "hist/cores",
    "orats_hist_dailies": "hist/dailies",
    "orats_hist_strikes": "hist/strikes",
    "orats_hist_earnings": "hist/earnings",
    "orats_hist_summaries": "hist/summaries",
}
FULL_HISTORY_ENDPOINTS = {
    "hist/cores",
    "hist/dailies",
    "hist/earnings",
    "hist/summaries",
}
TICKER_BATCH_SIZE = {
    "hist/cores": 10,
    "hist/dailies": 10,
    "hist/earnings": 1,
    "hist/summaries": 10,
}


@dataclass(frozen=True, order=True)
class CacheKey:
    endpoint: str
    ticker: str
    session_date: str

    def __post_init__(self) -> None:
        if not self.endpoint or not self.ticker or not self.session_date:
            raise ValueError("cache key fields are required")
        if self.ticker != self.ticker.upper():
            raise ValueError("cache-key ticker must be uppercase")
        if self.session_date == "ALL_HISTORY" and self.endpoint not in FULL_HISTORY_ENDPOINTS:
            raise ValueError("ALL_HISTORY is not valid for this endpoint")
        if self.session_date != "ALL_HISTORY":
            try:
                date.fromisoformat(self.session_date)
            except ValueError:
                raise ValueError(
                    "cache-key session_date must be YYYY-MM-DD or ALL_HISTORY"
                ) from None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CacheKey":
        return cls(
            endpoint=str(value["endpoint"]),
            ticker=str(value["ticker"]).upper(),
            session_date=str(value["session_date"]),
        )

    def to_dict(self) -> Dict[str, str]:
        return {
            "endpoint": self.endpoint,
            "ticker": self.ticker,
            "session_date": self.session_date,
        }


@dataclass(frozen=True)
class CacheInventory:
    keys: frozenset
    known_unavailable_keys: frozenset = frozenset()

    @classmethod
    def from_keys(
        cls,
        keys: Iterable[CacheKey],
        known_unavailable_keys: Iterable[CacheKey] = (),
    ) -> "CacheInventory":
        available = frozenset(keys)
        unavailable = frozenset(known_unavailable_keys)
        if available & unavailable:
            raise ValueError("a cache slice cannot be available and unavailable")
        return cls(keys=available, known_unavailable_keys=unavailable)

    @classmethod
    def from_json_file(cls, path: Path) -> "CacheInventory":
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(payload.get("keys"), list):
            raise ValueError("cache inventory must contain a keys list")
        unavailable = payload.get("known_unavailable_keys") or []
        if not isinstance(unavailable, list):
            raise ValueError("known_unavailable_keys must be a list")
        return cls.from_keys(
            (CacheKey.from_dict(item) for item in payload["keys"]),
            (CacheKey.from_dict(item) for item in unavailable),
        )

    @classmethod
    def from_store(cls, root: Path) -> "CacheInventory":
        """Inspect immutable local records/batches; never fill a missing slice."""

        resolved = root.expanduser().resolve()
        available = set()
        unavailable = set()

        def consume(record: SourceRecord, full_history_response: bool = False) -> None:
            ticker = str(
                record.payload.get("ticker") or record.payload.get("symbol") or ""
            ).upper()
            if not ticker:
                return
            if record.source == "orats_hist_strikes_unavailable":
                requested_date = str(
                    record.payload.get("requestedTradeDate") or record.session_date
                )[:10]
                unavailable.add(CacheKey("hist/strikes", ticker, requested_date))
                return
            endpoint = SOURCE_TO_ENDPOINT.get(record.source)
            if endpoint:
                available.add(CacheKey(endpoint, ticker, record.session_date))
                if full_history_response and endpoint in FULL_HISTORY_ENDPOINTS:
                    available.add(CacheKey(endpoint, ticker, "ALL_HISTORY"))

        for path in sorted((resolved / "records").glob("orats_hist_*/*/*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("cache record is not an object: {}".format(path))
            consume(SourceRecord.from_dict(payload))
        for path in sorted((resolved / "batches").glob("orats_hist_*/*/*.jsonl.gz")):
            for record in read_batch(path):
                consume(record, full_history_response=path.parent.name == "_multi_session")
        # A later successful slice supersedes an older unavailability record.
        unavailable.difference_update(available)
        return cls.from_keys(available, unavailable)

    def contains(self, key: CacheKey) -> bool:
        if key in self.keys:
            return True
        return bool(
            key.endpoint in FULL_HISTORY_ENDPOINTS
            and CacheKey(key.endpoint, key.ticker, "ALL_HISTORY") in self.keys
        )

    def is_known_unavailable(self, key: CacheKey) -> bool:
        return key in self.known_unavailable_keys

    def to_dict(self) -> Dict[str, Any]:
        return {
            "keys": [item.to_dict() for item in sorted(self.keys)],
            "known_unavailable_keys": [
                item.to_dict() for item in sorted(self.known_unavailable_keys)
            ],
        }


@dataclass(frozen=True)
class RequestBudgetPlan:
    required_keys: Tuple[CacheKey, ...]
    missing_keys: Tuple[CacheKey, ...]
    known_unavailable_keys: Tuple[CacheKey, ...]
    authorized_requests: int
    reported_remaining_requests: int
    minimum_reserved_requests: int

    @property
    def status(self) -> str:
        if not self.missing_keys and not self.known_unavailable_keys:
            return "CACHE_ONLY_READY"
        if self.missing_keys and self.known_unavailable_keys:
            return "BLOCKED_MISSING_AND_KNOWN_UNAVAILABLE"
        if self.missing_keys:
            return "BLOCKED_MISSING_CACHE"
        return "BLOCKED_KNOWN_UNAVAILABLE"

    @property
    def conservative_request_upper_bound_if_later_authorized(self) -> int:
        """Coalesce full-history endpoints and documented ticker batch sizes."""

        strike_requests = sum(
            item.endpoint == "hist/strikes" for item in self.missing_keys
        )
        tickers_by_endpoint: Dict[str, Set[str]] = {}
        for item in self.missing_keys:
            if item.endpoint == "hist/strikes":
                continue
            tickers_by_endpoint.setdefault(item.endpoint, set()).add(item.ticker)
        history_requests = sum(
            int(math.ceil(len(tickers) / TICKER_BATCH_SIZE.get(endpoint, 1)))
            for endpoint, tickers in tickers_by_endpoint.items()
        )
        return strike_requests + history_requests

    @property
    def requests_executed(self) -> int:
        return 0

    def to_dict(self, include_missing: bool = False) -> Dict[str, Any]:
        output: Dict[str, Any] = {
            "status": self.status,
            "required_cache_keys": len(self.required_keys),
            "available_cache_keys": (
                len(self.required_keys)
                - len(self.missing_keys)
                - len(self.known_unavailable_keys)
            ),
            "missing_cache_slices": len(self.missing_keys),
            "known_unavailable_cache_slices": len(self.known_unavailable_keys),
            "conservative_request_upper_bound_if_later_authorized": (
                self.conservative_request_upper_bound_if_later_authorized
            ),
            "authorized_requests_this_execution": self.authorized_requests,
            "requests_executed": self.requests_executed,
            "reported_remaining_requests": self.reported_remaining_requests,
            "minimum_reserved_requests": self.minimum_reserved_requests,
        }
        if include_missing:
            output["missing"] = [item.to_dict() for item in self.missing_keys]
            output["known_unavailable"] = [
                item.to_dict() for item in self.known_unavailable_keys
            ]
        return output


def assert_network_denied(spec: V5ResearchSpec) -> None:
    if spec.network_policy != "DENY" or spec.authorized_orats_requests_this_execution != 0:
        raise NetworkUseForbidden("v0.5 execution is not allowed to use an API")


def plan_cache_only(
    spec: V5ResearchSpec,
    inventory: CacheInventory,
    required: Iterable[CacheKey],
) -> RequestBudgetPlan:
    assert_network_denied(spec)
    allowed = set(spec.cache_only_endpoints)
    unique: Set[CacheKey] = set(required)
    disallowed = sorted(key.endpoint for key in unique if key.endpoint not in allowed)
    if disallowed:
        raise ValueError("cache requirement uses undeclared endpoints: {}".format(disallowed))
    ordered = tuple(sorted(unique))
    known_unavailable = tuple(
        item for item in ordered if inventory.is_known_unavailable(item)
    )
    missing = tuple(
        item
        for item in ordered
        if not inventory.contains(item) and not inventory.is_known_unavailable(item)
    )
    return RequestBudgetPlan(
        required_keys=ordered,
        missing_keys=missing,
        known_unavailable_keys=known_unavailable,
        authorized_requests=spec.authorized_orats_requests_this_execution,
        reported_remaining_requests=spec.reported_remaining_orats_requests,
        minimum_reserved_requests=spec.minimum_reserved_orats_requests,
    )
