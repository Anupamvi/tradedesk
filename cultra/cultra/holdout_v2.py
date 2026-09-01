"""One-time untouched-holdout evaluation for frozen Cultra V2 models."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .calibration import (
    brier_score,
    expected_calibration_error,
    unconditional_brier_score,
)
from .evidence_registry import (
    DEFAULT_EVIDENCE_ROOT,
    EvidenceRegistry,
    HoldoutReuseError,
    RegistryState,
)
from .hypotheses import FROZEN_HYPOTHESIS_COUNT, FROZEN_HYPOTHESIS_REGISTRY
from .modeling_v2 import (
    FrozenLinearModel,
    TARGETS,
    _calendar,
    _open_outcomes,
    _payload_hash,
    calibrator_from_payload,
    coherent_exit_probabilities,
    frozen_calendar_split,
    load_frozen_models_v2,
)
from .protocol import load_historical_campaign_protocol
from .statistics import (
    contribution_concentration,
    holm_adjust_mapping,
    two_way_clustered_bootstrap_mean_ci,
    two_way_clustered_positive_mean_p_value,
)


HOLDOUT_SCHEMA = "cultra.holdout-results-v2.v1"


class HoldoutV2Error(RuntimeError):
    """The final holdout cannot be opened or promoted safely."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _private_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    with open(path, "xb") as handle:
        os.chmod(path, 0o600)
        handle.write(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
            + b"\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _probability(
    probability_model: Mapping[str, Any], features: Mapping[str, float]
) -> float:
    raw_model = FrozenLinearModel.from_dict(probability_model["raw_model"])
    raw = raw_model.predict_one(features)
    calibrator = calibrator_from_payload(probability_model["calibrator"])
    value = float(calibrator.predict_one(raw))  # type: ignore[attr-defined]
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise HoldoutV2Error("calibrated probability is invalid")
    return value


def _binary(outcome: Mapping[str, Any], target: str) -> int:
    if target == "POP_NET":
        return int(float(outcome["net_pnl"]) > 0.0)
    if target == "P_TARGET":
        return int(bool(outcome["target_hit"]))
    if target == "P_STOP":
        return int(bool(outcome["stop_hit"]))
    if target == "P_MAX_LOSS":
        return int(bool(outcome["max_loss_hit"]))
    raise HoldoutV2Error("probability target is invalid")


def _concentration(values: Sequence[float], clusters: Sequence[str]) -> float:
    try:
        return contribution_concentration(values, clusters).max_fraction
    except ValueError:
        return 1.0


def _holdout_rows(
    connection: sqlite3.Connection,
    hypothesis_id: str,
    start: date,
    end: date,
) -> Tuple[sqlite3.Row, ...]:
    return tuple(
        connection.execute(
            """
            SELECT * FROM candidate_ledger
             WHERE hypothesis_id = ? AND signal_date BETWEEN ? AND ?
             ORDER BY signal_date, ticker, record_id
            """,
            (hypothesis_id, start.isoformat(), end.isoformat()),
        )
    )


def evaluate_frozen_hypothesis_holdout(
    rows: Sequence[sqlite3.Row],
    model_artifact: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    seed: int,
    calendar_periods: Optional[Mapping[str, str]] = None,
) -> Mapping[str, Any]:
    """Evaluate one already-frozen hypothesis without suppressing candidates."""

    return_model = FrozenLinearModel.from_dict(model_artifact["return_model"])
    probability_models = model_artifact["probability_models"]
    geometry_total = 0
    resolved_geometry = 0
    selected = []
    unresolved_selected_worst_losses = []
    unknown_feature_rows = 0
    for row in rows:
        if row["selection_json"] is None:
            continue
        geometry_total += 1
        risk_payload = (
            None if row["risk_json"] is None else json.loads(str(row["risk_json"]))
        )
        risk_reference = (
            None if risk_payload is None else risk_payload.get("risk_reference")
        )
        if risk_reference is None or not math.isfinite(float(risk_reference)) or float(risk_reference) <= 0.0:
            raise HoldoutV2Error("geometrically selected row lacks finite risk reference")
        risk = float(risk_reference)
        if row["status"] == "RESOLVED" and row["outcome_json"] is not None:
            resolved_geometry += 1
        if row["features_json"] is None:
            # Selection cannot be reconstructed.  Assume it would have passed
            # and charge maximum modeled risk so missing features cannot help.
            unknown_feature_rows += 1
            unresolved_selected_worst_losses.append(risk)
            continue
        features = {
            str(key): float(value)
            for key, value in json.loads(str(row["features_json"])).items()
        }
        predicted_return = return_model.predict_one(features)
        if predicted_return <= 0.0:
            continue
        if row["status"] != "RESOLVED" or row["outcome_json"] is None:
            unresolved_selected_worst_losses.append(risk)
            continue
        outcome = json.loads(str(row["outcome_json"]))
        raw_probabilities = {
            target: _probability(probability_models[target], features)
            for target in TARGETS
        }
        coherent = coherent_exit_probabilities(raw_probabilities)
        probabilities = {
            target: float(coherent["metrics"][target]) for target in TARGETS
        }
        selected.append(
            {
                "record_id": str(row["record_id"]),
                "ticker": str(row["ticker"]),
                "signal_date": str(row["signal_date"]),
                "predicted_return_on_risk": predicted_return,
                "net_pnl": float(outcome["net_pnl"]),
                "risk_reference": float(outcome["risk_reference"]),
                "return_on_risk": float(outcome["net_pnl"])
                / float(outcome["risk_reference"]),
                "outcomes": {target: _binary(outcome, target) for target in TARGETS},
                "probabilities": probabilities,
                "joint_exit_probabilities": coherent["categories"],
                "probability_projection_l1_distance": coherent[
                    "projection_l1_distance"
                ],
            }
        )
    reasons = []
    promotion = protocol["promotion_policy"]
    calibration = protocol["calibration_policy"]
    resolution_rate = resolved_geometry / geometry_total if geometry_total else 0.0
    if resolution_rate < float(promotion["minimum_resolution_rate"]):
        reasons.append("holdout exact-path resolution rate is below 95 percent")
    selected_count = len(selected)
    if selected_count < int(promotion["minimum_holdout_resolved_trades"]):
        reasons.append("holdout has fewer than 100 resolved selected trades")
    cluster_count = len({(item["ticker"], item["signal_date"]) for item in selected})
    if cluster_count < int(promotion["minimum_holdout_ticker_date_clusters"]):
        reasons.append("holdout has fewer than 40 ticker/date clusters")

    interval_payload: Optional[Mapping[str, Any]] = None
    raw_p_value = 1.0
    expectancy: Optional[float] = None
    worst_case_expectancy: Optional[float] = None
    if selected:
        net_pnls = tuple(float(item["net_pnl"]) for item in selected)
        tickers = tuple(str(item["ticker"]) for item in selected)
        dates = tuple(str(item["signal_date"]) for item in selected)
        expectancy = math.fsum(net_pnls) / len(selected)
        try:
            interval = two_way_clustered_bootstrap_mean_ci(
                net_pnls,
                tickers,
                dates,
                confidence=float(promotion["holdout_confidence"]),
                iterations=5000,
                seed=seed,
            )
            raw_p_value = two_way_clustered_positive_mean_p_value(
                net_pnls, tickers, dates, iterations=5000, seed=seed + 1
            )
            interval_payload = {
                "point_net_pnl_dollars": interval.point,
                "lower_net_pnl_dollars": interval.lower,
                "upper_net_pnl_dollars": interval.upper,
                "confidence": interval.confidence,
                "iterations": interval.iterations,
                "ticker_clusters": interval.first_cluster_count,
                "date_clusters": interval.second_cluster_count,
                "ticker_date_clusters": interval.joint_cluster_count,
            }
            if interval.lower <= 0.0:
                reasons.append("holdout two-way clustered lower bound is not positive")
        except ValueError as exc:
            reasons.append("holdout two-way bootstrap unavailable: %s" % str(exc))
        if expectancy <= 0.0:
            reasons.append("holdout selected net expectancy is not positive")
        total_after_missing = math.fsum(float(item["net_pnl"]) for item in selected) - math.fsum(
            unresolved_selected_worst_losses
        )
        worst_case_count = len(selected) + len(unresolved_selected_worst_losses)
        worst_case_expectancy = total_after_missing / max(1, worst_case_count)
        if worst_case_expectancy <= 0.0:
            reasons.append("unresolved-path worst-case expectancy is not positive")
        ticker_concentration = _concentration(
            tuple(float(item["net_pnl"]) for item in selected), tickers
        )
        if calendar_periods is None:
            ordered_dates = tuple(
                sorted({str(row["signal_date"]) for row in rows})
            )
            block_size = int(promotion["calendar_concentration_period_sessions"])
            period_lookup = {
                value: "P%03d" % (index // block_size)
                for index, value in enumerate(ordered_dates)
            }
        else:
            period_lookup = dict(calendar_periods)
        period_concentration = _concentration(
            tuple(float(item["net_pnl"]) for item in selected),
            tuple(period_lookup[str(item["signal_date"])] for item in selected),
        )
        if max(ticker_concentration, period_concentration) > float(
            promotion["maximum_single_ticker_or_period_profit_fraction"]
        ):
            reasons.append("ticker/calendar contribution concentration exceeds 20 percent")
    else:
        ticker_concentration = 1.0
        period_concentration = 1.0
        reasons.append("holdout selected no resolved trades")

    calibration_results = {}
    for target in TARGETS:
        predictions = tuple(float(item["probabilities"][target]) for item in selected)
        outcomes = tuple(int(item["outcomes"][target]) for item in selected)
        if predictions:
            brier = brier_score(predictions, outcomes)
            base_rate = float(probability_models[target]["development_base_rate"])
            base_brier = unconditional_brier_score(outcomes, base_rate)
            ece = expected_calibration_error(predictions, outcomes)
            positives = sum(outcomes)
            negatives = len(outcomes) - positives
        else:
            brier = base_brier = ece = 1.0
            positives = negatives = 0
        calibration_results[target] = {
            "brier": brier,
            "development_base_rate_brier": base_brier,
            "expected_calibration_error": ece,
            "sample_size": len(outcomes),
            "positive_events": positives,
            "negative_events": negatives,
        }
        if brier >= base_brier:
            reasons.append("%s holdout Brier does not beat frozen base rate" % target)
        if ece > float(calibration["maximum_ece"]):
            reasons.append("%s holdout ECE exceeds tolerance" % target)
        if positives < int(calibration["minimum_positive_events_per_target"]):
            reasons.append("%s holdout lacks positive events" % target)
        if negatives < int(calibration["minimum_negative_events_per_target"]):
            reasons.append("%s holdout lacks negative events" % target)

    return {
        "hypothesis_id": model_artifact["hypothesis_id"],
        "strategy_id": model_artifact["strategy_id"],
        "state_before_holm": "HOLDOUT_PASS" if not reasons else "REJECTED",
        "reasons": sorted(set(reasons)),
        "geometrically_executable_rows": geometry_total,
        "resolved_geometrically_executable_rows": resolved_geometry,
        "resolution_rate": resolution_rate,
        "unknown_feature_rows_charged_as_selected_worst_case": unknown_feature_rows,
        "unresolved_selected_worst_case_count": len(unresolved_selected_worst_losses),
        "selected_resolved_trades": selected_count,
        "ticker_date_clusters": cluster_count,
        "net_expectancy_dollars": expectancy,
        "expected_return_on_risk": (
            math.fsum(float(item["return_on_risk"]) for item in selected)
            / len(selected)
            if selected
            else None
        ),
        "unresolved_worst_case_expectancy_dollars": worst_case_expectancy,
        "bootstrap": interval_payload,
        "raw_one_sided_p_value": raw_p_value,
        "ticker_profit_concentration": ticker_concentration,
        "calendar_profit_concentration": period_concentration,
        "calibration": calibration_results,
        "selected_observations": selected,
    }


def consume_historical_v2_holdout(
    *, model_artifact_path: Path, evidence_registry_path: Path, output_path: Path
) -> Mapping[str, Any]:
    """Open every development-pass holdout once and commit decisions atomically."""

    models = load_frozen_models_v2(model_artifact_path)
    outcome_path = Path(str(models["outcome_database"])).resolve()
    if _sha256(outcome_path) != models["outcome_database_sha256"]:
        raise HoldoutV2Error("frozen outcome database changed after model freeze")
    outcome_manifest_path = outcome_path.with_suffix(
        outcome_path.suffix + ".manifest.json"
    )
    if (
        not outcome_manifest_path.is_file()
        or _sha256(outcome_manifest_path) != models["outcome_manifest_sha256"]
    ):
        raise HoldoutV2Error("frozen outcome manifest changed after model freeze")
    output = Path(output_path).expanduser().resolve()
    registry_path = Path(evidence_registry_path).expanduser().resolve()
    for supplied, label in ((output, "holdout result"), (registry_path, "evidence registry")):
        try:
            supplied.relative_to(DEFAULT_EVIDENCE_ROOT.resolve())
        except ValueError as exc:
            raise HoldoutV2Error("%s must remain inside Cultra evidence storage" % label) from exc
    receipt = output.with_suffix(output.suffix + ".registry.json")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or receipt.exists() or manifest_path.exists():
        raise HoldoutV2Error("holdout result already exists")
    connection, outcome_manifest = _open_outcomes(outcome_path)
    sessions = _calendar(connection, Path(str(outcome_manifest["normalized_database"])).resolve())
    split = frozen_calendar_split(sessions)
    holdout_dates = split["holdout"]
    protocol = load_historical_campaign_protocol()
    period_size = int(
        protocol["promotion_policy"]["calendar_concentration_period_sessions"]
    )
    calendar_periods = {
        value.isoformat(): "HOLDOUT_P%03d" % (index // period_size)
        for index, value in enumerate(holdout_dates)
    }
    model_by_id = {str(item["hypothesis_id"]): item for item in models["hypotheses"]}
    eligible = []
    fingerprints: Dict[str, str] = {}
    with EvidenceRegistry(registry_path) as registry:
        for hypothesis in FROZEN_HYPOTHESIS_REGISTRY:
            result = model_by_id[hypothesis.hypothesis_id]
            record = registry.get(hypothesis.hypothesis_id)
            if record.pop_model_artifact_id != _payload_hash(result):
                connection.close()
                raise HoldoutV2Error("registry model identity does not match frozen artifact")
            if result["state"] == "VALIDATION_PASS":
                if record.holdout_consumed:
                    connection.close()
                    raise HoldoutReuseError("final holdout has already been consumed")
                if record.state is not RegistryState.VALIDATION_PASS:
                    connection.close()
                    raise HoldoutV2Error("development-pass registry state drifted")
                eligible.append(hypothesis.hypothesis_id)
                fingerprints[hypothesis.hypothesis_id] = record.holdout_fingerprint

    results: Dict[str, Mapping[str, Any]] = {}
    raw_p_values = {item.hypothesis_id: 1.0 for item in FROZEN_HYPOTHESIS_REGISTRY}
    try:
        for index, hypothesis_id in enumerate(eligible):
            evaluated = evaluate_frozen_hypothesis_holdout(
                _holdout_rows(
                    connection,
                    hypothesis_id,
                    holdout_dates[0],
                    holdout_dates[-1],
                ),
                model_by_id[hypothesis_id],
                protocol,
                seed=2048 + index * 17,
                calendar_periods=calendar_periods,
            )
            evaluated = dict(
                evaluated,
                holdout_period={
                    "start": holdout_dates[0].isoformat(),
                    "end": holdout_dates[-1].isoformat(),
                    "signal_start": split["holdout_signal"][0].isoformat(),
                    "signal_end": split["holdout_signal"][-1].isoformat(),
                },
            )
            results[hypothesis_id] = evaluated
            raw_p_values[hypothesis_id] = float(evaluated["raw_one_sided_p_value"])
    finally:
        connection.close()
    adjusted = holm_adjust_mapping(raw_p_values)
    decisions = []
    for hypothesis_id in eligible:
        result = dict(results[hypothesis_id])
        result["holm_adjusted_p_value"] = adjusted[hypothesis_id]
        result["holm_family_size"] = FROZEN_HYPOTHESIS_COUNT
        if adjusted[hypothesis_id] > float(protocol["promotion_policy"]["maximum_holm_adjusted_p_value"]):
            result["reasons"] = sorted(set(result["reasons"] + ["Holm-adjusted significance gate failed"]))
        passed = not result["reasons"]
        result["state"] = "HOLDOUT_PASS" if passed else "REJECTED"
        results[hypothesis_id] = result
        decisions.append((hypothesis_id, fingerprints[hypothesis_id], passed))
    for hypothesis in FROZEN_HYPOTHESIS_REGISTRY:
        if hypothesis.hypothesis_id not in results:
            results[hypothesis.hypothesis_id] = {
                "hypothesis_id": hypothesis.hypothesis_id,
                "strategy_id": hypothesis.strategy_id,
                "state": "REJECTED_DEVELOPMENT_HOLDOUT_NOT_OPENED",
                "reasons": list(model_by_id[hypothesis.hypothesis_id]["reasons"]),
                "holm_adjusted_p_value": 1.0,
                "holm_family_size": FROZEN_HYPOTHESIS_COUNT,
            }
    prepared_at = datetime.now(timezone.utc)
    payload = {
        "schema": HOLDOUT_SCHEMA,
        "prepared_at": prepared_at.isoformat(),
        "model_artifact": str(Path(model_artifact_path).expanduser().resolve()),
        "model_artifact_sha256": _sha256(Path(model_artifact_path).expanduser().resolve()),
        "outcome_database": str(outcome_path),
        "outcome_database_sha256": _sha256(outcome_path),
        "holdout_opened_hypotheses": sorted(eligible),
        "holdout_not_opened_hypotheses": sorted(set(model_by_id) - set(eligible)),
        "holm_family_size": FROZEN_HYPOTHESIS_COUNT,
        "selection_rule": "PREDICTED_NET_RETURN_ON_RISK_GREATER_THAN_ZERO",
        "no_top_n": True,
        "results": {key: results[key] for key in sorted(results)},
        "network_attempted": False,
    }
    _private_json(output, payload)
    manifest = {
        "schema": "cultra.holdout-results-v2-manifest.v1",
        "result": str(output),
        "result_bytes": output.stat().st_size,
        "result_sha256": _sha256(output),
        "model_artifact_sha256": payload["model_artifact_sha256"],
        "outcome_database_sha256": payload["outcome_database_sha256"],
        "network_attempted": False,
    }
    _private_json(manifest_path, manifest)
    committed = {}
    if decisions:
        with EvidenceRegistry(registry_path) as registry:
            records = registry.consume_holdout_batch(decisions, now=prepared_at)
            committed = {key: value.state.value for key, value in records.items()}
    commit_receipt = {
        "schema": "cultra.holdout-registry-commit-v2.v1",
        "holdout_result_sha256": manifest["result_sha256"],
        "evidence_registry": str(registry_path),
        "committed_states": committed,
        "committed_at": prepared_at.isoformat(),
    }
    _private_json(receipt, commit_receipt)
    return dict(
        manifest,
        registry_receipt=str(receipt),
        opened=len(eligible),
        passed=sum(value == "HOLDOUT_PASS" for value in committed.values()),
        rejected=sum(value == "REJECTED" for value in committed.values()),
    )


__all__ = [
    "HOLDOUT_SCHEMA",
    "HoldoutV2Error",
    "consume_historical_v2_holdout",
    "evaluate_frozen_hypothesis_holdout",
]
