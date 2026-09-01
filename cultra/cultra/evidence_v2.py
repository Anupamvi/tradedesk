"""Verified conversion from V2 model/holdout artifacts to ticket evidence.

The historical engine and the manual-ticket gate use deliberately different
types.  This module is the only bridge: it verifies the immutable model,
one-time holdout result, registry commit receipt, and durable registry state
before constructing ``FamilyEvidence``.  It performs no network access.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from .catalog import CATALOG_VERSION
from .domain import EvidenceState, FamilyEvidence, PeriodEvidence
from .evidence_registry import (
    DEFAULT_EVIDENCE_ROOT,
    EvidenceRegistry,
    RegistryState,
)
from .hypotheses import (
    FROZEN_HYPOTHESIS_COUNT,
    FROZEN_HYPOTHESIS_REGISTRY,
    HYPOTHESIS_REGISTRY_HASH,
    HYPOTHESIS_REGISTRY_VERSION,
)
from .modeling_v2 import load_frozen_models_v2
from .protocol import load_historical_campaign_protocol


class EvidenceV2Error(RuntimeError):
    """Saved V2 evidence is missing, inconsistent, or not holdout-pass."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceV2Error("%s is unavailable" % label) from exc
    if not isinstance(value, Mapping):
        raise EvidenceV2Error("%s is malformed" % label)
    return value


def _aware(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceV2Error("%s timestamp is invalid" % label) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceV2Error("%s timestamp is not timezone-aware" % label)
    return parsed.astimezone(timezone.utc)


def _period(name: str, payload: Mapping[str, Any]) -> PeriodEvidence:
    try:
        return PeriodEvidence(
            name=name,
            expectancy=float(payload["net_expectancy_dollars"]),
            lower_confidence_bound=float(
                payload["lower_net_expectancy_dollars_95"]
            ),
            resolved_trades=int(payload["selected_resolved_trades"]),
            independent_clusters=int(payload["ticker_date_clusters"]),
            start=date.fromisoformat(str(payload["start"])),
            end=date.fromisoformat(str(payload["end"])),
            confidence_level=0.95,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceV2Error("%s period evidence is incomplete" % name) from exc


def load_holdout_pass_family_evidence(
    *,
    hypothesis_id: str,
    model_artifact_path: Path,
    holdout_result_path: Path,
    evidence_registry_path: Path,
) -> FamilyEvidence:
    """Load one exact-hypothesis HOLDOUT_PASS record for ticket construction."""

    definition = next(
        (
            item
            for item in FROZEN_HYPOTHESIS_REGISTRY
            if item.hypothesis_id == hypothesis_id
        ),
        None,
    )
    if definition is None:
        raise EvidenceV2Error("hypothesis is outside the frozen registry")
    models = load_frozen_models_v2(model_artifact_path)
    model_matches = tuple(
        item
        for item in models["hypotheses"]
        if item.get("hypothesis_id") == hypothesis_id
    )
    if len(model_matches) != 1 or model_matches[0].get("state") != "VALIDATION_PASS":
        raise EvidenceV2Error("hypothesis did not pass frozen development validation")
    model = model_matches[0]

    result_path = Path(holdout_result_path).expanduser().resolve()
    registry_path = Path(evidence_registry_path).expanduser().resolve()
    for supplied, label in ((result_path, "holdout result"), (registry_path, "evidence registry")):
        try:
            supplied.relative_to(DEFAULT_EVIDENCE_ROOT.resolve())
        except ValueError as exc:
            raise EvidenceV2Error("%s is outside Cultra evidence storage" % label) from exc
    if not result_path.is_file() or not registry_path.is_file():
        raise EvidenceV2Error("holdout result or evidence registry is unavailable")
    manifest_path = result_path.with_suffix(result_path.suffix + ".manifest.json")
    receipt_path = result_path.with_suffix(result_path.suffix + ".registry.json")
    holdout = _load(result_path, "holdout result")
    manifest = _load(manifest_path, "holdout manifest")
    receipt = _load(receipt_path, "holdout registry receipt")
    model_path = Path(model_artifact_path).expanduser().resolve()
    if (
        holdout.get("schema") != "cultra.holdout-results-v2.v1"
        or manifest.get("schema") != "cultra.holdout-results-v2-manifest.v1"
        or Path(str(manifest.get("result", ""))).resolve() != result_path
        or int(manifest.get("result_bytes", -1)) != result_path.stat().st_size
        or manifest.get("result_sha256") != _sha256(result_path)
        or holdout.get("model_artifact_sha256") != _sha256(model_path)
        or manifest.get("model_artifact_sha256") != _sha256(model_path)
    ):
        raise EvidenceV2Error("holdout artifact does not reconcile")
    committed = receipt.get("committed_states")
    if (
        receipt.get("schema") != "cultra.holdout-registry-commit-v2.v1"
        or receipt.get("holdout_result_sha256") != _sha256(result_path)
        or Path(str(receipt.get("evidence_registry", ""))).resolve() != registry_path
        or not isinstance(committed, Mapping)
        or committed.get(hypothesis_id) != "HOLDOUT_PASS"
    ):
        raise EvidenceV2Error("holdout registry receipt is not a committed pass")
    with EvidenceRegistry(registry_path) as registry:
        record = registry.get(hypothesis_id)
    if record.state is not RegistryState.HOLDOUT_PASS or not record.holdout_consumed:
        raise EvidenceV2Error("durable evidence registry is not HOLDOUT_PASS")
    if record.pop_model_artifact_id != "sha256:" + hashlib.sha256(
        json.dumps(
            model,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest():
        raise EvidenceV2Error("registry model identity differs from the frozen model")

    results = holdout.get("results")
    if not isinstance(results, Mapping) or not isinstance(
        results.get(hypothesis_id), Mapping
    ):
        raise EvidenceV2Error("holdout result does not contain the hypothesis")
    tested = results[hypothesis_id]
    if tested.get("state") != "HOLDOUT_PASS" or tested.get("reasons") != []:
        raise EvidenceV2Error("hypothesis did not pass the untouched holdout")
    bootstrap = tested.get("bootstrap")
    period = tested.get("holdout_period")
    calibration = tested.get("calibration")
    development = model.get("selection_model_validation")
    if not all(
        isinstance(item, Mapping)
        for item in (bootstrap, period, calibration, development)
    ):
        raise EvidenceV2Error("holdout or development metrics are incomplete")
    pop = calibration.get("POP_NET")
    if not isinstance(pop, Mapping):
        raise EvidenceV2Error("holdout POP calibration is incomplete")
    try:
        training = _period("training", development["training_period"])
        validation = _period("validation", development["validation_period"])
        holdout_period = PeriodEvidence(
            name="holdout",
            expectancy=float(tested["net_expectancy_dollars"]),
            lower_confidence_bound=float(bootstrap["lower_net_pnl_dollars"]),
            resolved_trades=int(tested["selected_resolved_trades"]),
            independent_clusters=int(tested["ticker_date_clusters"]),
            start=date.fromisoformat(str(period["start"])),
            end=date.fromisoformat(str(period["end"])),
            confidence_level=float(bootstrap["confidence"]),
        )
        evaluated_at = _aware(holdout["prepared_at"], "holdout evaluation")
        model_frozen_at = _aware(models["model_frozen_at"], "model freeze")
        policy = load_historical_campaign_protocol()
        expires_at = evaluated_at + timedelta(
            days=int(
                policy["promotion_policy"][
                    "maximum_evidence_age_calendar_days"
                ]
            )
        )
        event_counts = tuple(
            (
                target,
                int(calibration[target]["positive_events"]),
                int(calibration[target]["negative_events"]),
            )
            for target in ("POP_NET", "P_TARGET", "P_STOP", "P_MAX_LOSS")
        )
        return FamilyEvidence(
            strategy_family=hypothesis_id,
            state=EvidenceState.HOLDOUT_PASS,
            training=training,
            validation=validation,
            holdout=holdout_period,
            shadow=None,
            holm_adjusted_p_value=float(tested["holm_adjusted_p_value"]),
            holm_family_size=int(tested["holm_family_size"]),
            holm_catalog_version=HYPOTHESIS_REGISTRY_VERSION,
            max_contribution_fraction=max(
                float(tested["ticker_profit_concentration"]),
                float(tested["calendar_profit_concentration"]),
            ),
            contribution_dimensions=("calendar_period", "ticker"),
            pop_ece=float(pop["expected_calibration_error"]),
            pop_brier_score=float(pop["brier"]),
            base_rate_brier_score=float(pop["development_base_rate_brier"]),
            cost_model_version=record.cost_model_version,
            model_version=str(models["model_version"]),
            pop_model_artifact_id=record.pop_model_artifact_id,
            frozen_catalog_version=CATALOG_VERSION,
            frozen_exit_policy=definition.exit_policy,
            holdout_consumed_once=True,
            hypothesis_registry_hash=HYPOTHESIS_REGISTRY_HASH,
            timing_policy_version="SIGNAL_CLOSE_T_ENTRY_T_PLUS_1_V1",
            universe_policy_version=(
                "POINT_IN_TIME_STRATIFIED_DETERMINISTIC_SAMPLE"
            ),
            model_frozen_at=model_frozen_at,
            holdout_evaluated_at=evaluated_at,
            evidence_expires_at=expires_at,
            holdout_resolved_candidates=int(tested["selected_resolved_trades"]),
            holdout_unresolved_candidates=int(
                tested["unresolved_selected_worst_case_count"]
            ),
            unresolved_worst_case_expectancy=float(
                tested["unresolved_worst_case_expectancy_dollars"]
            ),
            probability_event_counts=event_counts,
            two_way_clustered=True,
            point_in_time_membership=True,
            next_session_entry=True,
            holdout_registry_receipt=_sha256(receipt_path),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceV2Error("holdout-pass family evidence is incomplete") from exc


__all__ = ["EvidenceV2Error", "load_holdout_pass_family_evidence"]
