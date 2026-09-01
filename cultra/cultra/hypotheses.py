"""Finite pre-holdout hypothesis registry for Cultra."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

from .catalog import FROZEN_STRATEGY_CATALOG
from .structures import (
    STRUCTURE_TEMPLATE_REGISTRY_HASH,
    STRUCTURE_TEMPLATE_VERSION,
    get_structure_template,
)


HYPOTHESIS_REGISTRY_VERSION = "cultra-hypothesis-registry-v1"
HOLDING_HORIZONS = (20, 40, 60)


@dataclass(frozen=True)
class HypothesisDefinition:
    hypothesis_id: str
    strategy_id: str
    signal_profile: str
    signal_bias: str
    holding_sessions: int
    structure_template_hash: str
    entry_policy: str
    exit_policy: str
    model_policy: str
    implementation_state: str


def _registry() -> Tuple[HypothesisDefinition, ...]:
    values = []
    for strategy in FROZEN_STRATEGY_CATALOG:
        template = get_structure_template(strategy.strategy_id)
        for horizon in HOLDING_HORIZONS:
            values.append(
                HypothesisDefinition(
                    hypothesis_id="%s__%s__H%02d"
                    % (
                        strategy.strategy_id,
                        template.signal_profile,
                        horizon,
                    ),
                    strategy_id=strategy.strategy_id,
                    signal_profile=template.signal_profile,
                    signal_bias=template.signal_bias,
                    holding_sessions=horizon,
                    structure_template_hash=template.template_hash,
                    entry_policy="DETERMINISTIC_CHAIN_GEOMETRY_V1",
                    exit_policy="STOP_FIRST_TARGET_STOP_TIME_H%02d_V1" % horizon,
                    model_policy="CHRONOLOGICAL_OOF_LINEAR_MODELS_V2",
                    implementation_state="OFFLINE_FROZEN_EXACT_PATH_ENGINE_V1",
                )
            )
    return tuple(values)


FROZEN_HYPOTHESIS_REGISTRY = _registry()
FROZEN_HYPOTHESIS_COUNT = len(FROZEN_HYPOTHESIS_REGISTRY)
if len({item.hypothesis_id for item in FROZEN_HYPOTHESIS_REGISTRY}) != FROZEN_HYPOTHESIS_COUNT:
    raise RuntimeError("duplicate hypothesis in frozen registry")


def registry_payload() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "schema": "cultra.hypothesis-registry.v1",
        "version": HYPOTHESIS_REGISTRY_VERSION,
        "structure_template_version": STRUCTURE_TEMPLATE_VERSION,
        "structure_template_registry_hash": STRUCTURE_TEMPLATE_REGISTRY_HASH,
        "holding_horizons": list(HOLDING_HORIZONS),
        "hypothesis_count": FROZEN_HYPOTHESIS_COUNT,
        "hypotheses": [asdict(item) for item in FROZEN_HYPOTHESIS_REGISTRY],
    }
    identity = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return dict(payload, registry_hash=hashlib.sha256(identity).hexdigest())


HYPOTHESIS_REGISTRY_HASH = str(registry_payload()["registry_hash"])


__all__ = [
    "FROZEN_HYPOTHESIS_COUNT",
    "FROZEN_HYPOTHESIS_REGISTRY",
    "HOLDING_HORIZONS",
    "HYPOTHESIS_REGISTRY_HASH",
    "HYPOTHESIS_REGISTRY_VERSION",
    "HypothesisDefinition",
    "registry_payload",
]
