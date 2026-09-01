"""Deterministic multivariate learning for Cultra pattern research.

The models in this module are deliberately small and inspectable.  They use
only Python's standard library, learn from entry-time features, and are judged
only on chronologically out-of-fold observations separated by the configured
embargo.  No probability is publishable as POP unless the complete OOF gate
passes.
"""

from __future__ import annotations

import copy
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .calibration import (
    IsotonicCalibrator,
    LogisticCalibrator,
    brier_score,
    expected_calibration_error as calibrated_ece,
)
from .statistics import clustered_bootstrap_mean_ci


TARGETS = ("POP_NET", "P_TARGET", "P_STOP", "P_MAX_LOSS")


class LearningError(RuntimeError):
    """A model or its leakage-safe evaluation could not be reproduced."""


def _finite(value: Any, name: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise LearningError("%s is not numeric" % name) from exc
    if not math.isfinite(converted):
        raise LearningError("%s is not finite" % name)
    return converted


def _sigmoid(value: float) -> float:
    bounded = max(-35.0, min(35.0, float(value)))
    return 1.0 / (1.0 + math.exp(-bounded))


def _solve(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Tuple[float, ...]:
    """Solve a small dense linear system with deterministic pivoting."""

    size = len(vector)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise LearningError("linear system dimensions are invalid")
    augmented = [
        [float(value) for value in row] + [float(vector[index])]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            raise LearningError("linear system is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0.0:
                continue
            augmented[row] = [
                left - factor * right
                for left, right in zip(augmented[row], augmented[column])
            ]
    return tuple(augmented[index][-1] for index in range(size))


def feature_vector(row: Mapping[str, Any]) -> Tuple[float, ...]:
    """Return only fields known when the exact structure is selected."""

    family = str(row["strategy_family"])
    bullish = family in {"LONG_CALL", "CALL_DEBIT_VERTICAL"}
    momentum = _finite(row["momentum_20"], "momentum_20")
    aligned = momentum if bullish else -momentum
    realized = max(1e-6, _finite(row["realized_volatility_20"], "realized_volatility_20"))
    implied = max(1e-6, _finite(row["smv_vol"], "smv_vol"))
    spread = _finite(row["relative_spread"], "relative_spread")
    maximum_loss = _finite(row["maximum_loss"], "maximum_loss")
    if maximum_loss <= 0.0:
        raise LearningError("maximum_loss must be positive")
    entry_debit = _finite(row["entry_debit"], "entry_debit")
    maximum_profit = row.get("maximum_profit")
    reward_to_risk = (
        5.0
        if maximum_profit is None
        else min(5.0, max(0.0, _finite(maximum_profit, "maximum_profit") / maximum_loss))
    )
    iv_to_realized = implied / realized
    return (
        aligned,
        abs(momentum),
        realized,
        implied,
        iv_to_realized,
        spread,
        entry_debit / maximum_loss,
        reward_to_risk,
        aligned * iv_to_realized,
    )


def _outcome(row: Mapping[str, Any], target: str) -> int:
    if target == "POP_NET":
        return int(_finite(row["net_pnl"], "net_pnl") > 0.0)
    if target == "P_TARGET":
        return int(bool(row["target_hit"]))
    if target == "P_STOP":
        return int(bool(row["stop_hit"]))
    if target == "P_MAX_LOSS":
        return int(bool(row["max_loss_hit"]))
    raise LearningError("unsupported probability target")


def _return_on_risk(row: Mapping[str, Any]) -> float:
    result = _finite(row["net_pnl"], "net_pnl") / _finite(
        row["maximum_loss"], "maximum_loss"
    )
    # Stops, gaps, and target overshoots remain visible.  This bound prevents a
    # single malformed quote from dominating a development-only regression.
    return max(-2.0, min(5.0, result))


def _standardizer(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[Tuple[float, ...], ...]]:
    raw = tuple(feature_vector(row) for row in rows)
    if not raw:
        raise LearningError("model training rows are empty")
    width = len(raw[0])
    means = tuple(statistics.mean(item[index] for item in raw) for index in range(width))
    scales = tuple(
        statistics.pstdev(item[index] for item in raw) or 1.0 for index in range(width)
    )
    transformed = tuple(
        (1.0,)
        + tuple(
            (value - means[index]) / scales[index]
            for index, value in enumerate(item)
        )
        for item in raw
    )
    return means, scales, transformed


@dataclass(frozen=True)
class LinearModel:
    kind: str
    means: Tuple[float, ...]
    scales: Tuple[float, ...]
    weights: Tuple[float, ...]
    sample_size: int
    converged: bool

    def predict(self, row: Mapping[str, Any]) -> float:
        raw = feature_vector(row)
        if len(raw) != len(self.means) or len(self.weights) != len(raw) + 1:
            raise LearningError("model feature dimensions do not match")
        values = (1.0,) + tuple(
            (value - self.means[index]) / self.scales[index]
            for index, value in enumerate(raw)
        )
        estimate = math.fsum(left * right for left, right in zip(self.weights, values))
        if self.kind == "LOGISTIC_L2":
            return _sigmoid(estimate)
        if self.kind == "RIDGE_RETURN_ON_RISK":
            return max(-2.0, min(5.0, estimate))
        raise LearningError("unsupported model kind")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "kind": self.kind,
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "sample_size": self.sample_size,
            "converged": self.converged,
        }


def fit_logistic(
    rows: Sequence[Mapping[str, Any]],
    target: str,
    *,
    l2: float,
    maximum_iterations: int,
    tolerance: float,
) -> LinearModel:
    means, scales, matrix = _standardizer(rows)
    outcomes = tuple(_outcome(row, target) for row in rows)
    width = len(matrix[0])
    if len(set(outcomes)) == 1:
        probability = (sum(outcomes) + 1.0) / (len(outcomes) + 2.0)
        intercept = math.log(probability / (1.0 - probability))
        return LinearModel(
            kind="LOGISTIC_L2",
            means=means,
            scales=scales,
            weights=(intercept,) + (0.0,) * (width - 1),
            sample_size=len(rows),
            converged=True,
        )
    base = (sum(outcomes) + 0.5) / (len(outcomes) + 1.0)
    weights = [math.log(base / (1.0 - base))] + [0.0] * (width - 1)
    converged = False
    for _iteration in range(int(maximum_iterations)):
        probabilities = tuple(
            _sigmoid(math.fsum(weight * value for weight, value in zip(weights, item)))
            for item in matrix
        )
        gradient = [0.0] * width
        information = [[0.0] * width for _ in range(width)]
        for item, outcome, probability in zip(matrix, outcomes, probabilities):
            residual = outcome - probability
            variance = max(1e-7, probability * (1.0 - probability))
            for left in range(width):
                gradient[left] += item[left] * residual
                for right in range(width):
                    information[left][right] += variance * item[left] * item[right]
        for index in range(1, width):
            gradient[index] -= float(l2) * weights[index]
            information[index][index] += float(l2)
        try:
            change = _solve(information, gradient)
        except LearningError:
            break
        weights = [value + delta for value, delta in zip(weights, change)]
        if max(abs(value) for value in change) <= float(tolerance):
            converged = True
            break
    return LinearModel(
        kind="LOGISTIC_L2",
        means=means,
        scales=scales,
        weights=tuple(weights),
        sample_size=len(rows),
        converged=converged,
    )


def fit_ridge(
    rows: Sequence[Mapping[str, Any]], *, l2: float
) -> LinearModel:
    means, scales, matrix = _standardizer(rows)
    outcomes = tuple(_return_on_risk(row) for row in rows)
    width = len(matrix[0])
    normal = [[0.0] * width for _ in range(width)]
    response = [0.0] * width
    for item, outcome in zip(matrix, outcomes):
        for left in range(width):
            response[left] += item[left] * outcome
            for right in range(width):
                normal[left][right] += item[left] * item[right]
    for index in range(1, width):
        normal[index][index] += float(l2)
    weights = _solve(normal, response)
    return LinearModel(
        kind="RIDGE_RETURN_ON_RISK",
        means=means,
        scales=scales,
        weights=weights,
        sample_size=len(rows),
        converged=True,
    )


def fit_bundle(
    rows: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> Mapping[str, Any]:
    probabilities = {
        target: fit_logistic(
            rows,
            target,
            l2=float(policy["logistic_l2"]),
            maximum_iterations=int(policy["maximum_newton_iterations"]),
            tolerance=float(policy["convergence_tolerance"]),
        )
        for target in TARGETS
    }
    return {
        "probabilities": probabilities,
        "return_on_risk": fit_ridge(rows, l2=float(policy["ridge_l2"])),
    }


def _brier(predictions: Sequence[float], outcomes: Sequence[int]) -> float:
    if not predictions or len(predictions) != len(outcomes):
        raise LearningError("Brier inputs are empty or mismatched")
    return statistics.mean(
        (float(prediction) - int(outcome)) ** 2
        for prediction, outcome in zip(predictions, outcomes)
    )


def _mse(predictions: Sequence[float], outcomes: Sequence[float]) -> float:
    if not predictions or len(predictions) != len(outcomes):
        raise LearningError("MSE inputs are empty or mismatched")
    return statistics.mean(
        (float(prediction) - float(outcome)) ** 2
        for prediction, outcome in zip(predictions, outcomes)
    )


def expected_calibration_error(
    predictions: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    if not predictions or len(predictions) != len(outcomes):
        raise LearningError("calibration inputs are empty or mismatched")
    result = 0.0
    for index in range(int(bins)):
        lower = index / float(bins)
        upper = (index + 1) / float(bins)
        selected = tuple(
            position
            for position, prediction in enumerate(predictions)
            if prediction >= lower
            and (prediction < upper or (index == bins - 1 and prediction <= 1.0))
        )
        if not selected:
            continue
        predicted = statistics.mean(predictions[position] for position in selected)
        observed = statistics.mean(outcomes[position] for position in selected)
        result += len(selected) / len(predictions) * abs(predicted - observed)
    return result


def _serialize_calibrator(calibrator: object) -> Mapping[str, Any]:
    if isinstance(calibrator, LogisticCalibrator):
        return {
            "method": "logistic",
            "sample_size": calibrator.sample_size,
            "intercept": calibrator.intercept,
            "slope": calibrator.slope,
            "converged": calibrator.converged,
        }
    if isinstance(calibrator, IsotonicCalibrator):
        return {
            "method": "isotonic",
            "sample_size": calibrator.sample_size,
            "x_thresholds": list(calibrator.x_thresholds),
            "y_values": list(calibrator.y_values),
        }
    raise LearningError("unsupported calibration object")


def _fit_calibrators(
    scores: Sequence[float], outcomes: Sequence[int]
) -> Tuple[LogisticCalibrator, IsotonicCalibrator]:
    try:
        return (
            LogisticCalibrator.fit(scores, outcomes),
            IsotonicCalibrator.fit(scores, outcomes),
        )
    except (ArithmeticError, TypeError, ValueError) as exc:
        raise LearningError("POP calibrator fit failed") from exc


def _nested_oof_calibration(
    oof: Sequence[Mapping[str, Any]],
    *,
    session_index: Mapping[str, int],
    policy: Mapping[str, Any],
    model_version: str,
) -> Mapping[str, Any]:
    """Calibrate raw outer-OOF scores on earlier OOF observations only.

    The raw classifier and its calibrator have separate chronological evidence
    paths.  For each calibration fold, the calibrator sees only prior outer-OOF
    predictions separated from that fold by the same market-session embargo.
    Logistic and isotonic calibration compete on these development validation
    rows.  The selected method is then refit on all outer-OOF rows solely for
    future, still-development-only inference.
    """

    minimum = int(policy.get("minimum_calibration_observations", 20))
    embargo = int(policy["embargo_sessions"])
    fold_indexes = sorted({int(item["fold_index"]) for item in oof})
    by_target: Dict[str, Any] = {}
    runtime: Dict[str, object] = {}
    for target in TARGETS:
        logistic_rows: List[Mapping[str, Any]] = []
        isotonic_rows: List[Mapping[str, Any]] = []
        fold_summaries: List[Mapping[str, Any]] = []
        all_logistic_converged = True
        for fold_index in fold_indexes:
            validation = tuple(
                item for item in oof if int(item["fold_index"]) == fold_index
            )
            if not validation:
                continue
            validation_start_index = min(
                session_index[str(item["entry_date"])] for item in validation
            )
            training_cutoff = validation_start_index - embargo
            training = tuple(
                item
                for item in oof
                if session_index[str(item["entry_date"])] < training_cutoff
            )
            if len(training) < minimum:
                continue
            training_scores = tuple(
                float(item["raw_probabilities"][target]) for item in training
            )
            training_outcomes = tuple(
                int(item["outcomes"][target]) for item in training
            )
            validation_scores = tuple(
                float(item["raw_probabilities"][target]) for item in validation
            )
            validation_outcomes = tuple(
                int(item["outcomes"][target]) for item in validation
            )
            logistic, isotonic = _fit_calibrators(
                training_scores, training_outcomes
            )
            logistic_predictions = logistic.predict(validation_scores)
            isotonic_predictions = isotonic.predict(validation_scores)
            base_rate = sum(training_outcomes) / len(training_outcomes)
            all_logistic_converged = all_logistic_converged and logistic.converged
            for source, logistic_value, isotonic_value in zip(
                validation, logistic_predictions, isotonic_predictions
            ):
                common = {
                    "record_id": str(source["record_id"]),
                    "entry_date": str(source["entry_date"]),
                    "fold_index": fold_index,
                    "raw_probability": float(source["raw_probabilities"][target]),
                    "training_base_rate": base_rate,
                    "outcome": int(source["outcomes"][target]),
                }
                logistic_rows.append(
                    dict(common, calibrated_probability=float(logistic_value))
                )
                isotonic_rows.append(
                    dict(common, calibrated_probability=float(isotonic_value))
                )
            training_indexes = tuple(
                session_index[str(item["entry_date"])] for item in training
            )
            validation_indexes = tuple(
                session_index[str(item["entry_date"])] for item in validation
            )
            fold_summaries.append(
                {
                    "fold_index": fold_index,
                    "training_count": len(training),
                    "training_start": min(str(item["entry_date"]) for item in training),
                    "training_end": max(str(item["entry_date"]) for item in training),
                    "training_end_session_index": max(training_indexes),
                    "embargo_sessions": embargo,
                    "validation_count": len(validation),
                    "validation_start": min(
                        str(item["entry_date"]) for item in validation
                    ),
                    "validation_end": max(
                        str(item["entry_date"]) for item in validation
                    ),
                    "validation_start_session_index": min(validation_indexes),
                    "logistic_brier": brier_score(
                        logistic_predictions, validation_outcomes
                    ),
                    "isotonic_brier": brier_score(
                        isotonic_predictions, validation_outcomes
                    ),
                    "base_rate_brier": brier_score(
                        (base_rate,) * len(validation_outcomes),
                        validation_outcomes,
                    ),
                    "logistic_converged": logistic.converged,
                }
            )
        if not logistic_rows:
            by_target[target] = {
                "status": "INSUFFICIENT_NESTED_CALIBRATION_FOLDS",
                "model_version": model_version,
                "minimum_calibration_observations": minimum,
                "oof_observations": 0,
                "folds": [],
            }
            continue
        outcomes = tuple(int(item["outcome"]) for item in logistic_rows)
        logistic_predictions = tuple(
            float(item["calibrated_probability"]) for item in logistic_rows
        )
        isotonic_predictions = tuple(
            float(item["calibrated_probability"]) for item in isotonic_rows
        )
        logistic_brier = brier_score(logistic_predictions, outcomes)
        isotonic_brier = brier_score(isotonic_predictions, outcomes)
        method = (
            "logistic"
            if all_logistic_converged and logistic_brier <= isotonic_brier
            else "isotonic"
        )
        selected_rows = logistic_rows if method == "logistic" else isotonic_rows
        all_scores = tuple(float(item["raw_probabilities"][target]) for item in oof)
        all_outcomes = tuple(int(item["outcomes"][target]) for item in oof)
        final_logistic, final_isotonic = _fit_calibrators(all_scores, all_outcomes)
        if method == "logistic" and not final_logistic.converged:
            method = "isotonic"
            selected_rows = isotonic_rows
        final_calibrator: object = (
            final_logistic if method == "logistic" else final_isotonic
        )
        selected_predictions = tuple(
            float(item["calibrated_probability"]) for item in selected_rows
        )
        baselines = tuple(float(item["training_base_rate"]) for item in selected_rows)
        by_target[target] = {
            "status": "DEVELOPMENT_CALIBRATED_NOT_HOLDOUT_VALIDATED",
            "model_version": model_version,
            "selected_method": method,
            "oof_brier": brier_score(selected_predictions, outcomes),
            "base_rate_brier": brier_score(baselines, outcomes),
            "expected_calibration_error": calibrated_ece(
                selected_predictions, outcomes
            ),
            "logistic_oof_brier": logistic_brier,
            "isotonic_oof_brier": isotonic_brier,
            "oof_observations": len(selected_rows),
            "positive_outcomes": sum(outcomes),
            "mean_prediction": statistics.mean(selected_predictions),
            "observed_rate": statistics.mean(outcomes),
            "minimum_calibration_observations": minimum,
            "folds": fold_summaries,
            "frozen_development_calibrator": _serialize_calibrator(final_calibrator),
            "selected_oof_predictions": selected_rows,
        }
        runtime[target] = final_calibrator
    return {"targets": by_target, "_runtime_calibrators": runtime}


def overlap_exposure_clusters(
    rows: Sequence[Mapping[str, Any]], session_index: Mapping[str, int]
) -> Mapping[str, str]:
    """Collapse connected overlapping positions into ticker exposure episodes."""

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ticker"]), []).append(row)
    result: Dict[str, str] = {}
    for ticker, values in sorted(grouped.items()):
        episode = -1
        current_end = -1
        ordered = sorted(
            values,
            key=lambda row: (
                session_index[str(row["entry_date"])],
                session_index[str(row["exit_date"])],
                str(row["record_id"]),
            ),
        )
        for row in ordered:
            start = session_index[str(row["entry_date"])]
            end = session_index[str(row["exit_date"])]
            if start > current_end:
                episode += 1
                current_end = end
            else:
                current_end = max(current_end, end)
            result[str(row["record_id"])] = "%s:EXPOSURE_%03d" % (ticker, episode)
    return result


def _model_metrics(
    oof: Sequence[Mapping[str, Any]],
    *,
    session_index: Mapping[str, int],
    policy: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> Mapping[str, Any]:
    cluster_map = overlap_exposure_clusters(oof, session_index)
    probability_metrics: Dict[str, Mapping[str, Any]] = {}
    raw_probability_metrics: Dict[str, Mapping[str, Any]] = {}
    for target in TARGETS:
        raw_predictions = tuple(
            float(item["raw_probabilities"][target]) for item in oof
        )
        raw_outcomes = tuple(int(item["outcomes"][target]) for item in oof)
        raw_baselines = tuple(
            float(item["raw_probability_bases"][target]) for item in oof
        )
        raw_probability_metrics[target] = {
            "oof_brier": _brier(raw_predictions, raw_outcomes),
            "base_rate_brier": _brier(raw_baselines, raw_outcomes),
            "expected_calibration_error": expected_calibration_error(
                raw_predictions, raw_outcomes
            ),
            "oof_observations": len(oof),
            "positive_outcomes": sum(raw_outcomes),
            "mean_prediction": statistics.mean(raw_predictions),
            "observed_rate": statistics.mean(raw_outcomes),
            "publishable_as_pop": False,
        }
        target_calibration = calibration["targets"].get(target, {})
        selected_rows = tuple(target_calibration.get("selected_oof_predictions", ()))
        probability_metrics[target] = {
            "status": target_calibration.get(
                "status", "INSUFFICIENT_NESTED_CALIBRATION_FOLDS"
            ),
            "selected_method": target_calibration.get("selected_method"),
            "oof_brier": target_calibration.get("oof_brier"),
            "base_rate_brier": target_calibration.get("base_rate_brier"),
            "expected_calibration_error": target_calibration.get(
                "expected_calibration_error"
            ),
            "oof_observations": int(
                target_calibration.get("oof_observations", 0)
            ),
            "positive_outcomes": int(
                target_calibration.get("positive_outcomes", 0)
            ),
            "mean_prediction": target_calibration.get("mean_prediction"),
            "observed_rate": target_calibration.get("observed_rate"),
            "independent_exposure_clusters": len(
                {
                    cluster_map[str(item["record_id"])]
                    for item in selected_rows
                    if str(item["record_id"]) in cluster_map
                }
            ),
            "publishable_as_pop": False,
        }
    predictions = tuple(float(item["predicted_return_on_risk"]) for item in oof)
    outcomes = tuple(float(item["actual_return_on_risk"]) for item in oof)
    baselines = tuple(float(item["return_base"]) for item in oof)
    selected = tuple(item for item in oof if float(item["predicted_return_on_risk"]) > 0.0)
    selected_interval = None
    if selected:
        selected_interval = clustered_bootstrap_mean_ci(
            tuple(float(item["actual_return_on_risk"]) for item in selected),
            tuple(cluster_map[str(item["record_id"])] for item in selected),
            confidence=0.95,
            iterations=int(policy["bootstrap_iterations"]),
            seed=int(policy["bootstrap_seed"]),
        )
    residual_interval = clustered_bootstrap_mean_ci(
        tuple(
            float(item["actual_return_on_risk"])
            - float(item["predicted_return_on_risk"])
            for item in oof
        ),
        tuple(cluster_map[str(item["record_id"])] for item in oof),
        confidence=0.95,
        iterations=int(policy["bootstrap_iterations"]),
        seed=int(policy["bootstrap_seed"]) + 1,
    )
    return_metrics = {
        "oof_mse": _mse(predictions, outcomes),
        "base_mean_mse": _mse(baselines, outcomes),
        "oof_mean_actual_return_on_risk": statistics.mean(outcomes),
        "oof_mean_predicted_return_on_risk": statistics.mean(predictions),
        "selected_oof_observations": len(selected),
        "selected_oof_mean_actual_return_on_risk": (
            statistics.mean(float(item["actual_return_on_risk"]) for item in selected)
            if selected
            else None
        ),
        "selected_oof_95_lower_return_on_risk": (
            selected_interval.lower if selected_interval is not None else None
        ),
        "selected_oof_95_upper_return_on_risk": (
            selected_interval.upper if selected_interval is not None else None
        ),
        "selected_independent_exposure_clusters": (
            selected_interval.cluster_count if selected_interval is not None else 0
        ),
        "residual_bias_95_lower": residual_interval.lower,
        "residual_bias_95_upper": residual_interval.upper,
        "independent_exposure_clusters": residual_interval.cluster_count,
    }
    pop = probability_metrics["POP_NET"]
    pop_reasons = []
    if pop["status"] != "DEVELOPMENT_CALIBRATED_NOT_HOLDOUT_VALIDATED":
        pop_reasons.append("POP_CALIBRATION_UNAVAILABLE")
    if int(pop["oof_observations"]) < int(policy["minimum_oof_observations"]):
        pop_reasons.append("INSUFFICIENT_OOF_OBSERVATIONS")
    if int(pop["independent_exposure_clusters"]) < int(
        policy["minimum_independent_exposure_clusters"]
    ):
        pop_reasons.append("INSUFFICIENT_INDEPENDENT_EXPOSURE_CLUSTERS")
    if bool(policy["pop_must_beat_base_brier"]) and (
        pop["oof_brier"] is None
        or pop["base_rate_brier"] is None
        or not float(pop["oof_brier"]) < float(pop["base_rate_brier"])
    ):
        pop_reasons.append("POP_DOES_NOT_BEAT_BASE_RATE_BRIER")
    if pop["expected_calibration_error"] is None or float(
        pop["expected_calibration_error"]
    ) > float(policy["maximum_pop_ece"]):
        pop_reasons.append("POP_ECE_EXCEEDS_LIMIT")
    ev_reasons = []
    if len(oof) < int(policy["minimum_oof_observations"]):
        ev_reasons.append("INSUFFICIENT_OOF_OBSERVATIONS")
    if bool(policy["ev_must_beat_base_mse"]) and not (
        float(return_metrics["oof_mse"]) < float(return_metrics["base_mean_mse"])
    ):
        ev_reasons.append("EV_MODEL_DOES_NOT_BEAT_BASE_MEAN_MSE")
    if selected_interval is None:
        ev_reasons.append("NO_POSITIVE_OOF_MODEL_SELECTIONS")
    else:
        if selected_interval.cluster_count < int(
            policy["minimum_independent_exposure_clusters"]
        ):
            ev_reasons.append("INSUFFICIENT_SELECTED_EXPOSURE_CLUSTERS")
        if bool(policy["selected_oof_return_lower_bound_must_exceed_zero"]) and not (
            selected_interval.lower > 0.0
        ):
            ev_reasons.append("SELECTED_OOF_RETURN_LOWER_BOUND_NOT_POSITIVE")
    return {
        "probabilities": probability_metrics,
        "raw_classifier_probabilities_not_pop": raw_probability_metrics,
        "return_model": return_metrics,
        "pop_gate_pass": not pop_reasons,
        "pop_gate_reasons": pop_reasons,
        "ev_gate_pass": not ev_reasons,
        "ev_gate_reasons": ev_reasons,
    }


def build_walk_forward_models(
    rows: Sequence[Mapping[str, Any]],
    sessions: Sequence[str],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Fit one family model and calculate embargoed OOF evidence."""

    if not rows or not sessions:
        raise LearningError("historical rows and session calendar are required")
    if len(sessions) != len(set(sessions)):
        raise LearningError("session calendar contains duplicates")
    session_index = {value: index for index, value in enumerate(sessions)}
    policy = config["learning_policy"]
    family_results: Dict[str, Any] = {}
    for family in config["families"]:
        family_rows = tuple(
            sorted(
                (row for row in rows if str(row["strategy_family"]) == family),
                key=lambda row: (str(row["entry_date"]), str(row["ticker"]), str(row["record_id"])),
            )
        )
        if not family_rows:
            family_results[family] = {
                "status": "NO_HISTORICAL_ROWS",
                "models": None,
                "oof_predictions": [],
                "metrics": None,
            }
            continue
        if any(str(row["entry_date"]) not in session_index for row in family_rows):
            raise LearningError("historical entry date leaves the frozen session calendar")
        if any(str(row["exit_date"]) not in session_index for row in family_rows):
            raise LearningError("historical exit date leaves the frozen session calendar")
        maximum_entry = max(session_index[str(row["entry_date"])] for row in family_rows)
        first_validation = int(policy["minimum_training_sessions"]) + int(
            policy["embargo_sessions"]
        )
        oof: List[Mapping[str, Any]] = []
        fold_summaries = []
        fold_index = 0
        for validation_start in range(
            first_validation,
            maximum_entry + 1,
            int(policy["step_sessions"]),
        ):
            validation_end = min(
                maximum_entry + 1,
                validation_start + int(policy["validation_sessions"]),
            )
            training_cutoff = validation_start - int(policy["embargo_sessions"])
            training = tuple(
                row
                for row in family_rows
                if session_index[str(row["entry_date"])] < training_cutoff
            )
            validation = tuple(
                row
                for row in family_rows
                if validation_start
                <= session_index[str(row["entry_date"])]
                < validation_end
            )
            if len(training) < 20 or not validation:
                continue
            bundle = fit_bundle(training, policy)
            probability_bases = {
                target: statistics.mean(_outcome(row, target) for row in training)
                for target in TARGETS
            }
            return_base = statistics.mean(_return_on_risk(row) for row in training)
            for row in validation:
                oof.append(
                    {
                        "record_id": str(row["record_id"]),
                        "ticker": str(row["ticker"]),
                        "entry_date": str(row["entry_date"]),
                        "exit_date": str(row["exit_date"]),
                        "fold_index": fold_index,
                        "raw_probabilities": {
                            target: bundle["probabilities"][target].predict(row)
                            for target in TARGETS
                        },
                        "raw_probability_bases": probability_bases,
                        "outcomes": {target: _outcome(row, target) for target in TARGETS},
                        "predicted_return_on_risk": bundle["return_on_risk"].predict(row),
                        "return_base": return_base,
                        "actual_return_on_risk": _return_on_risk(row),
                    }
                )
            fold_summaries.append(
                {
                    "fold_index": fold_index,
                    "training_count": len(training),
                    "training_end": sessions[training_cutoff - 1],
                    "embargo_sessions": int(policy["embargo_sessions"]),
                    "validation_count": len(validation),
                    "validation_start": sessions[validation_start],
                    "validation_end": sessions[validation_end - 1],
                }
            )
            fold_index += 1
        if not oof:
            family_results[family] = {
                "status": "INSUFFICIENT_WALK_FORWARD_FOLDS",
                "models": None,
                "oof_predictions": [],
                "metrics": None,
            }
            continue
        final_bundle = fit_bundle(family_rows, policy)
        calibration = _nested_oof_calibration(
            oof,
            session_index=session_index,
            policy=policy,
            model_version=str(config["version"]),
        )
        metrics = _model_metrics(
            oof,
            session_index=session_index,
            policy=policy,
            calibration=calibration,
        )
        family_results[family] = {
            "status": "DEVELOPMENT_ONLY",
            "historical_rows": len(family_rows),
            "historical_start": min(str(row["entry_date"]) for row in family_rows),
            "historical_end": max(str(row["entry_date"]) for row in family_rows),
            "folds": fold_summaries,
            "oof_predictions": oof,
            "metrics": metrics,
            "calibration": calibration,
            "models": {
                "probabilities": {
                    target: final_bundle["probabilities"][target].to_dict()
                    for target in TARGETS
                },
                "return_on_risk": final_bundle["return_on_risk"].to_dict(),
            },
            "_runtime_models": final_bundle,
            "_runtime_calibrators": calibration["_runtime_calibrators"],
        }
    return family_results


def public_model_evidence(results: Mapping[str, Any]) -> Mapping[str, Any]:
    """Remove runtime model objects before deterministic serialization."""

    public: Dict[str, Any] = {}
    for family, result in results.items():
        public[family] = {
            key: copy.deepcopy(value)
            for key, value in result.items()
            if key
            not in {
                "_runtime_models",
                "_runtime_calibrators",
                "oof_predictions",
            }
        }
        calibration = public[family].get("calibration")
        if isinstance(calibration, dict):
            calibration.pop("_runtime_calibrators", None)
            for target in calibration.get("targets", {}).values():
                if isinstance(target, dict):
                    target.pop("selected_oof_predictions", None)
        public[family]["oof_prediction_count"] = len(result.get("oof_predictions", ()))
    return public


__all__ = [
    "LearningError",
    "LinearModel",
    "TARGETS",
    "build_walk_forward_models",
    "expected_calibration_error",
    "feature_vector",
    "fit_bundle",
    "fit_logistic",
    "fit_ridge",
    "overlap_exposure_clusters",
    "public_model_evidence",
]
