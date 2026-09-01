"""Load and validate Cultra's versioned, non-secret configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .catalog import CATALOG_VERSION, FROZEN_STRATEGY_CATALOG


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class EvidencePolicy:
    minimum_holding_sessions: int
    maximum_holding_sessions: int
    untouched_test_fraction: float
    embargo_sessions: int
    minimum_test_trades: int
    minimum_clusters: int
    maximum_ece: float
    quantity_policy: str
    portfolio_gates_enabled: bool
    output_top_n_cap: object
    manual_ticket_requires_finite_max_loss: bool
    manual_action_minimum_evidence_state: str
    prospective_shadow_mode: str
    shadow_minimum_calendar_days_before_action: int


@dataclass(frozen=True)
class StrategyCatalogConfig:
    schema_version: str
    catalog_version: str
    holding_sessions: List[int]
    families: List[str]
    ticket_rule: str


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ConfigurationError("configuration root must be an object")
    return value


def load_evidence_policy(path: Path) -> EvidencePolicy:
    raw = _read_json(path)
    holding = raw.get("holding_sessions", {})
    policy = EvidencePolicy(
        minimum_holding_sessions=int(holding["minimum"]),
        maximum_holding_sessions=int(holding["maximum"]),
        untouched_test_fraction=float(raw["untouched_test_fraction"]),
        embargo_sessions=int(raw["embargo_sessions"]),
        minimum_test_trades=int(raw["minimum_untouched_test_trades"]),
        minimum_clusters=int(raw["minimum_independent_clusters"]),
        maximum_ece=float(raw["maximum_expected_calibration_error"]),
        quantity_policy=str(raw["quantity_policy"]),
        portfolio_gates_enabled=bool(raw["portfolio_gates_enabled"]),
        output_top_n_cap=raw["output_top_n_cap"],
        manual_ticket_requires_finite_max_loss=bool(raw["manual_ticket_requires_finite_max_loss"]),
        manual_action_minimum_evidence_state=str(
            raw["manual_action_minimum_evidence_state"]
        ),
        prospective_shadow_mode=str(raw["prospective_shadow_mode"]),
        shadow_minimum_calendar_days_before_action=int(
            raw["shadow_minimum_calendar_days_before_action"]
        ),
    )
    if policy.minimum_holding_sessions != 20 or policy.maximum_holding_sessions != 60:
        raise ConfigurationError("Cultra v1 holding horizon must remain 20-60 sessions")
    if policy.portfolio_gates_enabled:
        raise ConfigurationError("portfolio gating is prohibited")
    if policy.output_top_n_cap is not None:
        raise ConfigurationError("arbitrary output caps are prohibited")
    if policy.quantity_policy != "USER DETERMINED":
        raise ConfigurationError("quantity must remain user determined")
    if not policy.manual_ticket_requires_finite_max_loss:
        raise ConfigurationError("manual tickets must have finite maximum loss")
    if policy.manual_action_minimum_evidence_state != "HOLDOUT_PASS":
        raise ConfigurationError("manual actions require untouched HOLDOUT_PASS evidence")
    if policy.prospective_shadow_mode != "CONTINUOUS_NONBLOCKING_REVOCATION_MONITOR":
        raise ConfigurationError("prospective shadow must remain nonblocking monitoring")
    if policy.shadow_minimum_calendar_days_before_action != 0:
        raise ConfigurationError("prospective shadow cannot impose a calendar wait")
    return policy


def load_strategy_catalog(path: Path) -> StrategyCatalogConfig:
    raw = _read_json(path)
    catalog = StrategyCatalogConfig(
        schema_version=str(raw["schema_version"]),
        catalog_version=str(raw["catalog_version"]),
        holding_sessions=[int(value) for value in raw["holding_sessions"]],
        families=[str(value) for value in raw["families"]],
        ticket_rule=str(raw["ticket_rule"]),
    )
    if not catalog.families or len(catalog.families) != len(set(catalog.families)):
        raise ConfigurationError("strategy families must be unique and non-empty")
    if any(value < 20 or value > 60 for value in catalog.holding_sessions):
        raise ConfigurationError("catalog holding sessions must stay inside 20-60")
    expected = [item.strategy_id for item in FROZEN_STRATEGY_CATALOG]
    if catalog.catalog_version != CATALOG_VERSION:
        raise ConfigurationError("strategy catalog version does not match executable catalog")
    if catalog.families != expected:
        raise ConfigurationError(
            "strategy catalog config must exactly match the frozen executable catalog"
        )
    return catalog
