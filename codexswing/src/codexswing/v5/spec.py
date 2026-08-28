"""Frozen configuration contract for the v0.5 research lane."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from codexswing.schemas.source import canonical_json


SPEC_SCHEMA_VERSION = "codexswing.v5.research_spec.v1"
EXPECTED_HORIZONS = (3, 5, 10, 20)
EXPECTED_STRATEGIES = (
    "LONG_CALL",
    "LONG_PUT",
    "BULL_CALL_DEBIT",
    "BEAR_PUT_DEBIT",
    "BULL_PUT_CREDIT",
    "BEAR_CALL_CREDIT",
)


@dataclass(frozen=True)
class ExitPolicySpec:
    policy_id: str
    profit_target_r: float
    stop_loss_r: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExitPolicySpec":
        return cls(
            policy_id=str(value["policy_id"]),
            profit_target_r=float(value["profit_target_r"]),
            stop_loss_r=float(value["stop_loss_r"]),
        )

    def __post_init__(self) -> None:
        if not self.policy_id or self.policy_id != self.policy_id.upper():
            raise ValueError("exit policy_id must be a stable uppercase identifier")
        if self.profit_target_r < 0 or self.stop_loss_r < 0:
            raise ValueError("exit policy thresholds cannot be negative")
        if (self.profit_target_r == 0) != (self.stop_loss_r == 0):
            raise ValueError("fixed-hold policy must set both thresholds to zero")


@dataclass(frozen=True)
class V5ResearchSpec:
    model_version: str
    status: str
    network_policy: str
    reported_remaining_orats_requests: int
    authorized_orats_requests_this_execution: int
    minimum_reserved_orats_requests: int
    cache_only_endpoints: Tuple[str, ...]
    strategies: Tuple[str, ...]
    horizons_sessions: Tuple[int, ...]
    exit_policies: Tuple[ExitPolicySpec, ...]
    analog_count: int
    regime_features: Tuple[str, ...]
    earnings_exclusion: bool
    dividend_assignment_exclusion: bool
    multiple_testing_method: str
    family_alpha: float
    ledger_schema: str
    raw: Mapping[str, Any]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "V5ResearchSpec":
        if str(payload.get("schema_version")) != SPEC_SCHEMA_VERSION:
            raise ValueError("unsupported v0.5 research spec schema")
        budget = _mapping(payload, "api_budget")
        sources = _mapping(payload, "source_contract")
        variants = _mapping(payload, "replay_variants")
        regime = _mapping(payload, "regime_matching")
        events = _mapping(payload, "event_exclusions")
        testing = _mapping(payload, "multiple_testing")
        ledger = _mapping(payload, "prospective_ledger")
        return cls(
            model_version=str(payload["model_version"]),
            status=str(payload["status"]),
            network_policy=str(payload["network_policy"]),
            reported_remaining_orats_requests=int(budget["reported_remaining_orats_requests"]),
            authorized_orats_requests_this_execution=int(
                budget["authorized_orats_requests_this_execution"]
            ),
            minimum_reserved_orats_requests=int(budget["minimum_reserved_orats_requests"]),
            cache_only_endpoints=tuple(str(item) for item in sources["cache_only_endpoints"]),
            strategies=tuple(str(item) for item in variants["strategies"]),
            horizons_sessions=tuple(int(item) for item in variants["horizons_sessions"]),
            exit_policies=tuple(
                ExitPolicySpec.from_dict(item) for item in variants["exit_policies"]
            ),
            analog_count=int(regime["analog_count"]),
            regime_features=tuple(str(item) for item in regime["features"]),
            earnings_exclusion=bool(events["earnings"]),
            dividend_assignment_exclusion=bool(events["dividend_assignment"]),
            multiple_testing_method=str(testing["method"]),
            family_alpha=float(testing["family_alpha"]),
            ledger_schema=str(ledger["schema"]),
            raw=json.loads(canonical_json(payload)),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> "V5ResearchSpec":
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("v0.5 research spec must be a JSON object")
        return cls.from_dict(payload)

    def __post_init__(self) -> None:
        if self.model_version != "codexswing-v0.5":
            raise ValueError("unexpected v0.5 model version")
        if self.status != "IMPLEMENTED_NOT_EXECUTED":
            raise ValueError("v0.5 must remain explicitly unexecuted until a replay is run")
        if self.raw.get("validation_status") != "NO_REPLAY_RUN":
            raise ValueError("v0.5 validation status must remain NO_REPLAY_RUN")
        if self.network_policy != "DENY":
            raise ValueError("v0.5 network policy must be DENY")
        if self.authorized_orats_requests_this_execution != 0:
            raise ValueError("this frozen spec authorizes zero ORATS requests")
        if self.reported_remaining_orats_requests < 0:
            raise ValueError("reported ORATS request balance cannot be negative")
        if self.minimum_reserved_orats_requests != self.reported_remaining_orats_requests:
            raise ValueError("the full user-reported ORATS balance must remain reserved")
        if self.horizons_sessions != EXPECTED_HORIZONS:
            raise ValueError("v0.5 horizons must be exactly 3/5/10/20 sessions")
        if self.strategies != EXPECTED_STRATEGIES:
            raise ValueError("v0.5 must predeclare all six v0.4 strategies in stable order")
        if len(self.exit_policies) != 3 or len({item.policy_id for item in self.exit_policies}) != 3:
            raise ValueError("v0.5 must predeclare exactly three unique exit policies")
        variants = self.raw.get("replay_variants")
        if not isinstance(variants, Mapping) or variants.get("variant_count") != self.hypothesis_count:
            raise ValueError("declared variant_count does not match the frozen family")
        if self.analog_count != 250:
            raise ValueError("v0.5 analog count is frozen at 250")
        if len(self.regime_features) < 6 or len(set(self.regime_features)) != len(
            self.regime_features
        ):
            raise ValueError("regime feature contract must be unique and sufficiently broad")
        if not self.earnings_exclusion or not self.dividend_assignment_exclusion:
            raise ValueError("both event exclusions are mandatory")
        if self.multiple_testing_method != "CLUSTER_BOOTSTRAP_PLUS_HOLM_BONFERRONI":
            raise ValueError("unexpected multiple-testing method")
        if not 0 < self.family_alpha < 1:
            raise ValueError("family alpha must be between zero and one")
        if not self.cache_only_endpoints or len(set(self.cache_only_endpoints)) != len(
            self.cache_only_endpoints
        ):
            raise ValueError("cache-only endpoints must be non-empty and unique")

    @property
    def hypothesis_count(self) -> int:
        return len(self.strategies) * len(self.horizons_sessions) * len(self.exit_policies)

    @property
    def spec_sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.raw).encode("utf-8")).hexdigest()

    def public_summary(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "status": self.status,
            "network_policy": self.network_policy,
            "authorized_orats_requests_this_execution": self.authorized_orats_requests_this_execution,
            "reported_remaining_orats_requests": self.reported_remaining_orats_requests,
            "hypothesis_count": self.hypothesis_count,
            "strategies": list(self.strategies),
            "horizons_sessions": list(self.horizons_sessions),
            "exit_policies": [item.policy_id for item in self.exit_policies],
            "spec_sha256": self.spec_sha256,
            "validation_status": "NO_REPLAY_RUN",
        }


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError("{} must be an object".format(key))
    return value
