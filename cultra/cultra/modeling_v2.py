"""Leakage-safe model freezing for the Cultra V2 historical outcome ledger.

This module never reads final-holdout outcomes.  It creates chronological
out-of-fold development predictions, freezes model/calibration identities, and
records terminal development decisions before the separate holdout command is
allowed to inspect the final 20 percent of sessions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .calibration import (
    IsotonicCalibrator,
    LogisticCalibrator,
    brier_score,
    choose_calibrator,
    expected_calibration_error,
    unconditional_brier_score,
    wilson_interval,
)
from .catalog import CATALOG_VERSION
from .evidence_registry import (
    DEFAULT_EVIDENCE_ROOT,
    EvidencePartitions,
    EvidenceRegistry,
    FrozenEvidenceIdentity,
    RegistryState,
)
from .historical_v2 import HISTORICAL_ROOT
from .hypotheses import (
    FROZEN_HYPOTHESIS_REGISTRY,
    HYPOTHESIS_REGISTRY_HASH,
    HypothesisDefinition,
)
from .protocol import historical_protocol_hash, load_historical_campaign_protocol
from .statistics import two_way_clustered_bootstrap_mean_ci


MODEL_VERSION = "CULTRA_CHRONOLOGICAL_OOF_MODELS_V2"
MODEL_ARTIFACT_SCHEMA = "cultra.frozen-models-v2.v1"
TARGETS = ("POP_NET", "P_TARGET", "P_STOP", "P_MAX_LOSS")
EXIT_CATEGORIES = (
    "TARGET",
    "TIME_PROFIT",
    "STOP",
    "MAX_LOSS",
    "TIME_LOSS",
)


class ModelingV2Error(RuntimeError):
    """The V2 model cannot be frozen without leakage or missing evidence."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _payload_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _solve(matrix: Sequence[Sequence[float]], values: Sequence[float]) -> Tuple[float, ...]:
    """Solve a small dense system with deterministic partial pivoting."""

    size = len(values)
    if size == 0 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise ModelingV2Error("linear system dimensions are invalid")
    augmented = [list(map(float, row)) + [float(value)] for row, value in zip(matrix, values)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: (abs(augmented[row][column]), -row))
        if abs(augmented[pivot][column]) <= 1e-12:
            raise ModelingV2Error("model matrix is singular")
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
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
    result = tuple(row[-1] for row in augmented)
    if not all(math.isfinite(value) for value in result):
        raise ModelingV2Error("model coefficients are non-finite")
    return result


@dataclass(frozen=True)
class FrozenLinearModel:
    kind: str
    feature_names: Tuple[str, ...]
    means: Tuple[float, ...]
    scales: Tuple[float, ...]
    coefficients: Tuple[float, ...]
    l2: float
    sample_size: int

    def __post_init__(self) -> None:
        width = len(self.feature_names)
        if self.kind not in {"LOGISTIC", "RIDGE"}:
            raise ValueError("linear model kind is invalid")
        if not self.feature_names or len(set(self.feature_names)) != width:
            raise ValueError("feature names must be non-empty and unique")
        if len(self.means) != width or len(self.scales) != width:
            raise ValueError("standardization vectors do not align")
        if len(self.coefficients) != width + 1:
            raise ValueError("coefficient vector does not align")
        if self.l2 <= 0.0 or self.sample_size <= 0:
            raise ValueError("model fit metadata is invalid")
        if not all(math.isfinite(value) for value in self.means + self.scales + self.coefficients):
            raise ValueError("model contains a non-finite value")
        if any(value <= 0.0 for value in self.scales):
            raise ValueError("model scales must be positive")

    def predict_one(self, features: Mapping[str, float]) -> float:
        standardized = []
        for name, mean, scale in zip(self.feature_names, self.means, self.scales):
            if name not in features:
                raise ModelingV2Error("prediction feature is missing: %s" % name)
            value = float(features[name])
            if not math.isfinite(value):
                raise ModelingV2Error("prediction feature is non-finite: %s" % name)
            standardized.append((value - mean) / scale)
        linear = self.coefficients[0] + math.fsum(
            coefficient * value
            for coefficient, value in zip(self.coefficients[1:], standardized)
        )
        return _sigmoid(linear) if self.kind == "LOGISTIC" else linear

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "l2": self.l2,
            "sample_size": self.sample_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrozenLinearModel":
        return cls(
            kind=str(value["kind"]),
            feature_names=tuple(str(item) for item in value["feature_names"]),
            means=tuple(float(item) for item in value["means"]),
            scales=tuple(float(item) for item in value["scales"]),
            coefficients=tuple(float(item) for item in value["coefficients"]),
            l2=float(value["l2"]),
            sample_size=int(value["sample_size"]),
        )


def _standardize(
    rows: Sequence[Mapping[str, float]], feature_names: Sequence[str]
) -> Tuple[Tuple[Tuple[float, ...], ...], Tuple[float, ...], Tuple[float, ...]]:
    if not rows:
        raise ModelingV2Error("model fit has no observations")
    columns = []
    for name in feature_names:
        values = tuple(float(row[name]) for row in rows)
        if not all(math.isfinite(value) for value in values):
            raise ModelingV2Error("training feature is non-finite: %s" % name)
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        columns.append((mean, math.sqrt(variance) if variance > 1e-20 else 1.0))
    means = tuple(item[0] for item in columns)
    scales = tuple(item[1] for item in columns)
    matrix = tuple(
        tuple((float(row[name]) - mean) / scale for name, mean, scale in zip(feature_names, means, scales))
        for row in rows
    )
    return matrix, means, scales


def fit_linear_model(
    rows: Sequence[Mapping[str, float]],
    targets: Sequence[float],
    feature_names: Sequence[str],
    *,
    kind: str,
    l2: float,
    maximum_iterations: int = 100,
    tolerance: float = 1e-8,
) -> FrozenLinearModel:
    """Fit one deterministic standardized ridge or L2-logistic model."""

    if len(rows) != len(targets) or not rows:
        raise ModelingV2Error("model rows and targets must align")
    l2 = float(l2)
    if l2 <= 0.0 or maximum_iterations <= 0 or tolerance <= 0.0:
        raise ModelingV2Error("model controls must be positive")
    checked_targets = tuple(float(value) for value in targets)
    if not all(math.isfinite(value) for value in checked_targets):
        raise ModelingV2Error("model target is non-finite")
    if kind == "LOGISTIC" and any(value not in (0.0, 1.0) for value in checked_targets):
        raise ModelingV2Error("logistic targets must be binary")
    matrix, means, scales = _standardize(rows, feature_names)
    design = tuple((1.0,) + row for row in matrix)
    width = len(feature_names) + 1
    if kind == "RIDGE":
        normal = [[0.0 for _ in range(width)] for _ in range(width)]
        right = [0.0 for _ in range(width)]
        for values, target in zip(design, checked_targets):
            for left in range(width):
                right[left] += values[left] * target
                for column in range(width):
                    normal[left][column] += values[left] * values[column]
        for index in range(1, width):
            normal[index][index] += l2
        coefficients = _solve(normal, right)
    elif kind == "LOGISTIC":
        base = (math.fsum(checked_targets) + 0.5) / (len(checked_targets) + 1.0)
        coefficients = [math.log(base / (1.0 - base))] + [0.0] * (width - 1)

        def objective(candidate: Sequence[float]) -> float:
            result = 0.5 * l2 * math.fsum(value * value for value in candidate[1:])
            for values, target in zip(design, checked_targets):
                linear = math.fsum(left * right for left, right in zip(candidate, values))
                result += max(linear, 0.0) - linear * target + math.log1p(math.exp(-abs(linear)))
            return result

        current = objective(coefficients)
        for _iteration in range(maximum_iterations):
            gradient = [0.0 for _ in range(width)]
            hessian = [[0.0 for _ in range(width)] for _ in range(width)]
            for values, target in zip(design, checked_targets):
                prediction = _sigmoid(math.fsum(left * right for left, right in zip(coefficients, values)))
                residual = prediction - target
                weight = max(1e-9, prediction * (1.0 - prediction))
                for left in range(width):
                    gradient[left] += residual * values[left]
                    for column in range(width):
                        hessian[left][column] += weight * values[left] * values[column]
            for index in range(1, width):
                gradient[index] += l2 * coefficients[index]
                hessian[index][index] += l2
            hessian[0][0] += 1e-9
            step = _solve(hessian, gradient)
            scale = 1.0
            accepted: Optional[List[float]] = None
            while scale >= 2.0 ** -20:
                candidate = [value - scale * move for value, move in zip(coefficients, step)]
                candidate_objective = objective(candidate)
                if candidate_objective <= current:
                    accepted = candidate
                    current = candidate_objective
                    break
                scale /= 2.0
            if accepted is None:
                break
            coefficients = accepted
            if max(abs(scale * move) for move in step) < tolerance:
                break
        coefficients = tuple(coefficients)
    else:
        raise ModelingV2Error("model kind is unsupported")
    return FrozenLinearModel(
        kind=kind,
        feature_names=tuple(feature_names),
        means=means,
        scales=scales,
        coefficients=tuple(coefficients),
        l2=l2,
        sample_size=len(rows),
    )


@dataclass(frozen=True)
class DevelopmentObservation:
    record_id: str
    ticker: str
    signal_date: date
    features: Mapping[str, float]
    net_pnl: float
    return_on_risk: float
    targets: Mapping[str, int]
    outcome_class: str = "TIME_PROFIT"


def _target(outcome: Mapping[str, Any], name: str) -> int:
    if name == "POP_NET":
        return int(float(outcome["net_pnl"]) > 0.0)
    if name == "P_TARGET":
        return int(bool(outcome["target_hit"]))
    if name == "P_STOP":
        return int(bool(outcome["stop_hit"]))
    if name == "P_MAX_LOSS":
        return int(bool(outcome["max_loss_hit"]))
    raise ModelingV2Error("unknown probability target")


def _outcome_class(outcome: Mapping[str, Any]) -> str:
    """Map one frozen exit path to a mutually exclusive ticket scenario."""

    if bool(outcome["max_loss_hit"]):
        return "MAX_LOSS"
    if bool(outcome["target_hit"]):
        return "TARGET"
    if bool(outcome["stop_hit"]):
        return "STOP"
    return "TIME_PROFIT" if float(outcome["net_pnl"]) > 0.0 else "TIME_LOSS"


def _simplex_projection(values: Sequence[float]) -> Tuple[float, ...]:
    """Euclidean projection onto the probability simplex."""

    checked = tuple(float(value) for value in values)
    if not checked or not all(math.isfinite(value) for value in checked):
        raise ModelingV2Error("probability projection input is invalid")
    ordered = sorted(checked, reverse=True)
    cumulative = 0.0
    rho = -1
    for index, value in enumerate(ordered):
        cumulative += value
        threshold = (cumulative - 1.0) / (index + 1)
        if value - threshold > 0.0:
            rho = index
    if rho < 0:
        raise ModelingV2Error("probability projection failed")
    threshold = (math.fsum(ordered[: rho + 1]) - 1.0) / (rho + 1)
    projected = tuple(max(0.0, value - threshold) for value in checked)
    total = math.fsum(projected)
    if total <= 0.0:
        raise ModelingV2Error("probability projection produced no mass")
    normalized = tuple(value / total for value in projected)
    return normalized


def coherent_exit_probabilities(
    probabilities: Mapping[str, float],
) -> Mapping[str, Any]:
    """Project four calibrated targets into one coherent exit distribution.

    Separately fitted binary models need not obey subset and exclusivity rules.
    Tickets cannot publish such contradictions.  The projection operates on
    the five mutually exclusive exit categories, after which all four reported
    probabilities are recomputed from the same distribution.
    """

    try:
        pop = float(probabilities["POP_NET"])
        target = float(probabilities["P_TARGET"])
        stop = float(probabilities["P_STOP"])
        maximum = float(probabilities["P_MAX_LOSS"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelingV2Error("calibrated probability bundle is incomplete") from exc
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (pop, target, stop, maximum)
    ):
        raise ModelingV2Error("calibrated probability bundle is invalid")
    initial = (
        target,
        max(0.0, pop - target),
        max(0.0, stop - maximum),
        maximum,
        max(0.0, 1.0 - pop - stop),
    )
    projected = _simplex_projection(initial)
    categories = dict(zip(EXIT_CATEGORIES, projected))
    metrics = {
        "POP_NET": categories["TARGET"] + categories["TIME_PROFIT"],
        "P_TARGET": categories["TARGET"],
        "P_STOP": categories["STOP"] + categories["MAX_LOSS"],
        "P_MAX_LOSS": categories["MAX_LOSS"],
    }
    return {
        "categories": categories,
        "metrics": metrics,
        "raw_metrics": {
            "POP_NET": pop,
            "P_TARGET": target,
            "P_STOP": stop,
            "P_MAX_LOSS": maximum,
        },
        "projection_l1_distance": math.fsum(
            abs(metrics[name] - probabilities[name]) for name in TARGETS
        ),
    }


def _open_outcomes(path: Path) -> Tuple[sqlite3.Connection, Mapping[str, Any]]:
    database = Path(path).expanduser().resolve()
    try:
        database.relative_to(HISTORICAL_ROOT)
    except ValueError as exc:
        raise ModelingV2Error("outcome database is outside Cultra historical V2") from exc
    manifest_path = database.with_suffix(database.suffix + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelingV2Error("outcome manifest is unavailable") from exc
    if (
        manifest.get("schema") != "cultra.historical-outcomes-v2-manifest.v1"
        or Path(str(manifest.get("database", ""))).resolve() != database
        or int(manifest.get("database_bytes", -1)) != database.stat().st_size
        or manifest.get("database_sha256") != _sha256(database)
    ):
        raise ModelingV2Error("outcome database does not reconcile to its manifest")
    connection = sqlite3.connect("file:%s?mode=ro" % database, uri=True)
    connection.row_factory = sqlite3.Row
    check = connection.execute("PRAGMA integrity_check").fetchone()
    if check is None or check[0] != "ok":
        connection.close()
        raise ModelingV2Error("outcome database failed integrity check")
    return connection, manifest


def _calendar(connection: sqlite3.Connection, normalized_database: Path) -> Tuple[date, ...]:
    source = sqlite3.connect("file:%s?mode=ro" % normalized_database.resolve(), uri=True)
    try:
        rows = tuple(source.execute("SELECT session_date FROM sessions ORDER BY session_index"))
    finally:
        source.close()
    result = tuple(date.fromisoformat(str(row[0])) for row in rows)
    if len(result) != 450 or result != tuple(sorted(set(result))):
        raise ModelingV2Error("frozen model requires exactly 450 ordered sessions")
    return result


def frozen_calendar_split(sessions: Sequence[date]) -> Mapping[str, Tuple[date, ...]]:
    """Return the only permitted cohort-aligned V2 evidence split.

    Each 120-session research block admits entries only during its first 59
    sessions; the remaining 61 sessions contain the T+1 entry and complete
    60-session path.  The prior 180:300 OOF window fell mostly inside these
    no-entry suffixes and could create empty folds.  The frozen split below
    uses the actual signal windows and keeps every later block separated by a
    complete maximum-horizon path.
    """

    values = tuple(sessions)
    if len(values) != 450 or values != tuple(sorted(set(values))):
        raise ModelingV2Error("V2 calendar must contain 450 unique sessions")
    research = values[0:59]
    tuning = values[120:179]
    validation = values[240:299]
    holdout = values[360:450]
    return {
        "research": research,
        "development_embargo_1": values[59:120],
        "tuning": tuning,
        "development_embargo_2": values[179:240],
        "validation": validation,
        "final_embargo": values[299:360],
        "holdout": holdout,
        "holdout_signal": values[360:389],
        "holdout_path": values[389:450],
        "oof": tuning + validation,
    }


def _folds(sessions: Sequence[date]) -> Tuple[Tuple[Tuple[date, ...], Tuple[date, ...]], ...]:
    split = frozen_calendar_split(sessions)
    return (
        (split["research"], split["tuning"]),
        (split["research"] + split["tuning"], split["validation"]),
    )


def _load_observations(
    connection: sqlite3.Connection,
    hypothesis: HypothesisDefinition,
    allowed_dates: Sequence[date],
) -> Tuple[DevelopmentObservation, ...]:
    allowed = {value.isoformat() for value in allowed_dates}
    if not allowed:
        raise ModelingV2Error("development date partition is empty")
    first_date, last_date = min(allowed), max(allowed)
    rows = tuple(
        connection.execute(
            """
            SELECT record_id, ticker, signal_date, features_json, outcome_json
              FROM candidate_ledger
             WHERE hypothesis_id = ? AND status = 'RESOLVED'
               AND features_json IS NOT NULL AND outcome_json IS NOT NULL
               AND signal_date BETWEEN ? AND ?
             ORDER BY signal_date, ticker, record_id
            """,
            (hypothesis.hypothesis_id, first_date, last_date),
        )
    )
    observations = []
    for row in rows:
        if str(row["signal_date"]) not in allowed:
            continue
        features = json.loads(str(row["features_json"]))
        outcome = json.loads(str(row["outcome_json"]))
        risk = float(outcome["risk_reference"])
        net = float(outcome["net_pnl"])
        if risk <= 0.0 or not math.isfinite(risk) or not math.isfinite(net):
            raise ModelingV2Error("resolved outcome economics are invalid")
        observations.append(
            DevelopmentObservation(
                record_id=str(row["record_id"]),
                ticker=str(row["ticker"]),
                signal_date=date.fromisoformat(str(row["signal_date"])),
                features={str(key): float(value) for key, value in features.items()},
                net_pnl=net,
                return_on_risk=net / risk,
                targets={name: _target(outcome, name) for name in TARGETS},
                outcome_class=_outcome_class(outcome),
            )
        )
    return tuple(observations)


def _oof_predictions(
    observations: Sequence[DevelopmentObservation],
    folds: Sequence[Tuple[Sequence[date], Sequence[date]]],
    feature_names: Sequence[str],
    *,
    target_name: Optional[str],
    kind: str,
    l2: float,
    policy: Mapping[str, Any],
) -> Mapping[str, float]:
    result: Dict[str, float] = {}
    for training_dates, validation_dates in folds:
        training_set = set(training_dates)
        validation_set = set(validation_dates)
        training = tuple(item for item in observations if item.signal_date in training_set)
        validation = tuple(item for item in observations if item.signal_date in validation_set)
        if not training or not validation:
            raise ModelingV2Error("chronological OOF fold lacks resolved observations")
        targets = (
            tuple(item.return_on_risk for item in training)
            if target_name is None
            else tuple(float(item.targets[target_name]) for item in training)
        )
        model = fit_linear_model(
            tuple(item.features for item in training),
            targets,
            feature_names,
            kind=kind,
            l2=l2,
            maximum_iterations=int(policy["maximum_newton_iterations"]),
            tolerance=float(policy["convergence_tolerance"]),
        )
        for item in validation:
            result[item.record_id] = model.predict_one(item.features)
    return result


def _choose_l2(
    observations: Sequence[DevelopmentObservation],
    folds: Sequence[Tuple[Sequence[date], Sequence[date]]],
    feature_names: Sequence[str],
    *,
    target_name: Optional[str],
    kind: str,
    policy: Mapping[str, Any],
    selection_dates: Sequence[date],
) -> Tuple[float, Mapping[str, float], Mapping[str, float]]:
    scores: Dict[str, float] = {}
    predictions: Dict[float, Mapping[str, float]] = {}
    by_id = {item.record_id: item for item in observations}
    tuning_dates = set(selection_dates)
    for raw_l2 in policy["l2_grid"]:
        l2 = float(raw_l2)
        values = _oof_predictions(
            observations,
            folds,
            feature_names,
            target_name=target_name,
            kind=kind,
            l2=l2,
            policy=policy,
        )
        predictions[l2] = values
        score_ids = tuple(
            record_id
            for record_id in values
            if by_id[record_id].signal_date in tuning_dates
        )
        if not score_ids:
            raise ModelingV2Error("L2 tuning window has no OOF observations")
        if target_name is None:
            score_id_set = set(score_ids)
            error = math.fsum(
                (prediction - by_id[record_id].return_on_risk) ** 2
                for record_id, prediction in values.items()
                if record_id in score_id_set
            ) / len(score_ids)
        else:
            error = brier_score(
                tuple(values[key] for key in sorted(score_ids)),
                tuple(by_id[key].targets[target_name] for key in sorted(score_ids)),
            )
        scores[str(l2)] = error
    selected = min(predictions, key=lambda value: (scores[str(value)], value))
    return selected, predictions[selected], scores


def _calibrator_payload(calibrator: object) -> Mapping[str, Any]:
    if isinstance(calibrator, LogisticCalibrator):
        return {
            "kind": "LOGISTIC",
            "intercept": calibrator.intercept,
            "slope": calibrator.slope,
            "sample_size": calibrator.sample_size,
            "converged": calibrator.converged,
        }
    if isinstance(calibrator, IsotonicCalibrator):
        return {
            "kind": "ISOTONIC",
            "x_thresholds": list(calibrator.x_thresholds),
            "y_values": list(calibrator.y_values),
            "sample_size": calibrator.sample_size,
        }
    raise ModelingV2Error("calibrator type is unsupported")


def calibrator_from_payload(value: Mapping[str, Any]) -> object:
    if value.get("kind") == "LOGISTIC":
        return LogisticCalibrator(
            intercept=float(value["intercept"]),
            slope=float(value["slope"]),
            sample_size=int(value["sample_size"]),
            converged=bool(value["converged"]),
        )
    if value.get("kind") == "ISOTONIC":
        return IsotonicCalibrator(
            x_thresholds=tuple(float(item) for item in value["x_thresholds"]),
            y_values=tuple(float(item) for item in value["y_values"]),
            sample_size=int(value["sample_size"]),
        )
    raise ModelingV2Error("calibrator payload is invalid")


def _refit_selected_calibrator(
    name: str, scores: Sequence[float], outcomes: Sequence[int]
) -> object:
    if name == "logistic":
        return LogisticCalibrator.fit(scores, outcomes)
    if name == "isotonic":
        return IsotonicCalibrator.fit(scores, outcomes)
    raise ModelingV2Error("selected calibrator name is invalid")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ModelingV2Error("expectancy gate has no selected observations")
    return math.fsum(float(value) for value in values) / len(values)


def _development_period_metrics(
    observations: Sequence[DevelopmentObservation], *, seed: int
) -> Mapping[str, Any]:
    if not observations:
        raise ModelingV2Error("development evidence period has no selected observations")
    pnls = tuple(item.net_pnl for item in observations)
    interval = two_way_clustered_bootstrap_mean_ci(
        pnls,
        tuple(item.ticker for item in observations),
        tuple(item.signal_date.isoformat() for item in observations),
        confidence=0.95,
        iterations=5000,
        seed=seed,
    )
    return {
        "start": min(item.signal_date for item in observations).isoformat(),
        "end": max(item.signal_date for item in observations).isoformat(),
        "selected_resolved_trades": len(observations),
        "ticker_date_clusters": len(
            {(item.ticker, item.signal_date) for item in observations}
        ),
        "net_expectancy_dollars": interval.point,
        "lower_net_expectancy_dollars_95": interval.lower,
        "upper_net_expectancy_dollars_95": interval.upper,
        "ticker_clusters": interval.first_cluster_count,
        "date_clusters": interval.second_cluster_count,
        "bootstrap_iterations": interval.iterations,
    }


def _hypothesis_artifact(
    connection: sqlite3.Connection,
    sessions: Sequence[date],
    hypothesis: HypothesisDefinition,
    protocol: Mapping[str, Any],
) -> Mapping[str, Any]:
    split = frozen_calendar_split(sessions)
    observations = _load_observations(connection, hypothesis, sessions[:300])
    feature_names = tuple(protocol["learning_policy"]["feature_profiles"][hypothesis.signal_profile])
    result: Dict[str, Any] = {
        "hypothesis_id": hypothesis.hypothesis_id,
        "strategy_id": hypothesis.strategy_id,
        "feature_names": list(feature_names),
        "state": "UNPROVEN",
        "reasons": [],
    }
    try:
        selected_l2, return_oof, return_scores = _choose_l2(
            observations,
            _folds(sessions),
            feature_names,
            target_name=None,
            kind="RIDGE",
            policy=protocol["learning_policy"],
            selection_dates=split["tuning"],
        )
        by_id = {item.record_id: item for item in observations}
        selected = tuple(by_id[key] for key, prediction in return_oof.items() if prediction > 0.0)
        research_dates = set(split["tuning"])
        validation_dates = set(split["validation"])
        research = tuple(item for item in selected if item.signal_date in research_dates)
        validation = tuple(item for item in selected if item.signal_date in validation_dates)
        research_expectancy = _mean(tuple(item.net_pnl for item in research))
        validation_expectancy = _mean(tuple(item.net_pnl for item in validation))
        result["selection_model_validation"] = {
            "selected_l2": selected_l2,
            "validation_mse_by_l2": return_scores,
            "rule": "PREDICTED_NET_RETURN_ON_RISK_GREATER_THAN_ZERO",
            "research_selected": len(research),
            "research_expectancy": research_expectancy,
            "validation_selected": len(validation),
            "validation_expectancy": validation_expectancy,
            "training_period": _development_period_metrics(
                research, seed=701
            ),
            "validation_period": _development_period_metrics(
                validation, seed=907
            ),
        }
        selected_residuals = tuple(
            item.return_on_risk - float(return_oof[item.record_id])
            for item in selected
        )
        if not selected_residuals:
            raise ModelingV2Error("return model selected no OOF observations")
        residual_interval = two_way_clustered_bootstrap_mean_ci(
            selected_residuals,
            tuple(item.ticker for item in selected),
            tuple(item.signal_date.isoformat() for item in selected),
            confidence=0.95,
            iterations=5000,
            seed=1103,
        )
        result["return_model_uncertainty"] = {
            "method": "TWO_WAY_CLUSTERED_OOF_RESIDUAL_MEAN_95CI",
            "sample_size": len(selected_residuals),
            "residual_mean": residual_interval.point,
            # A lower confidence bound can be positive when the model
            # systematically underpredicts.  That is useful evidence, but a
            # value labelled conservative must never improve the point EV.
            "conservative_return_on_risk_offset": min(
                0.0, residual_interval.lower
            ),
            "raw_residual_lower_95": residual_interval.lower,
            "upper_return_on_risk_offset": residual_interval.upper,
            "ticker_clusters": residual_interval.first_cluster_count,
            "date_clusters": residual_interval.second_cluster_count,
            "ticker_date_clusters": residual_interval.joint_cluster_count,
            "iterations": residual_interval.iterations,
        }
        if research_expectancy <= 0.0:
            result["reasons"].append("research selected net expectancy is not positive")
        if validation_expectancy <= 0.0:
            result["reasons"].append("validation selected net expectancy is not positive")

        final_development = tuple(item for item in observations if item.signal_date < sessions[300])
        return_model = fit_linear_model(
            tuple(item.features for item in final_development),
            tuple(item.return_on_risk for item in final_development),
            feature_names,
            kind="RIDGE",
            l2=selected_l2,
            maximum_iterations=int(protocol["learning_policy"]["maximum_newton_iterations"]),
            tolerance=float(protocol["learning_policy"]["convergence_tolerance"]),
        )
        result["return_model"] = return_model.to_dict()
        probability_models = {}
        calibration_policy = protocol["calibration_policy"]
        selected_ids = {item.record_id for item in selected}
        research_selected_ids = {item.record_id for item in research}
        validation_selected_ids = {item.record_id for item in validation}
        all_ids = tuple(sorted(selected_ids))
        validation_predictions: Dict[str, Dict[str, float]] = {}
        refit_predictions: Dict[str, Dict[str, float]] = {}
        validation_base_rates: Dict[str, float] = {}
        for target_name in TARGETS:
            probability_l2, probability_oof, probability_scores = _choose_l2(
                observations,
                _folds(sessions),
                feature_names,
                target_name=target_name,
                kind="LOGISTIC",
                policy=protocol["learning_policy"],
                selection_dates=split["tuning"],
            )
            calibration_train_ids = tuple(sorted(selected_ids & research_selected_ids))
            calibration_validation_ids = tuple(sorted(selected_ids & validation_selected_ids))
            if not calibration_train_ids or not calibration_validation_ids:
                raise ModelingV2Error("calibration split has no selected observations")
            choice = choose_calibrator(
                tuple(probability_oof[key] for key in calibration_train_ids),
                tuple(by_id[key].targets[target_name] for key in calibration_train_ids),
                tuple(probability_oof[key] for key in calibration_validation_ids),
                tuple(by_id[key].targets[target_name] for key in calibration_validation_ids),
                MODEL_VERSION,
            )
            calibrated_validation = {
                key: float(choice.calibrator.predict_one(probability_oof[key]))
                for key in calibration_validation_ids
            }
            validation_predictions[target_name] = calibrated_validation
            validation_outcomes = tuple(
                by_id[key].targets[target_name]
                for key in calibration_validation_ids
            )
            refit = _refit_selected_calibrator(
                choice.name,
                tuple(probability_oof[key] for key in all_ids),
                tuple(by_id[key].targets[target_name] for key in all_ids),
            )
            validation_brier = brier_score(
                tuple(calibrated_validation[key] for key in calibration_validation_ids),
                validation_outcomes,
            )
            base_rate = math.fsum(by_id[key].targets[target_name] for key in calibration_train_ids) / len(calibration_train_ids)
            validation_base_rates[target_name] = base_rate
            base_brier = unconditional_brier_score(validation_outcomes, base_rate)
            ece = expected_calibration_error(
                tuple(calibrated_validation[key] for key in calibration_validation_ids),
                validation_outcomes,
            )
            positives = sum(by_id[key].targets[target_name] for key in all_ids)
            negatives = len(all_ids) - positives
            if positives < int(calibration_policy["minimum_positive_events_per_target"]):
                result["reasons"].append("%s lacks positive calibration events" % target_name)
            if negatives < int(calibration_policy["minimum_negative_events_per_target"]):
                result["reasons"].append("%s lacks negative calibration events" % target_name)
            raw_model = fit_linear_model(
                tuple(item.features for item in final_development),
                tuple(float(item.targets[target_name]) for item in final_development),
                feature_names,
                kind="LOGISTIC",
                l2=probability_l2,
                maximum_iterations=int(protocol["learning_policy"]["maximum_newton_iterations"]),
                tolerance=float(protocol["learning_policy"]["convergence_tolerance"]),
            )
            calibration_bins = []
            all_predictions = tuple(
                float(refit.predict_one(probability_oof[key]))  # type: ignore[attr-defined]
                for key in all_ids
            )
            refit_predictions[target_name] = dict(zip(all_ids, all_predictions))
            all_outcomes = tuple(by_id[key].targets[target_name] for key in all_ids)
            for bin_index in range(10):
                members = tuple(
                    index
                    for index, prediction in enumerate(all_predictions)
                    if min(9, int(prediction * 10.0)) == bin_index
                )
                if not members:
                    continue
                successes = sum(all_outcomes[index] for index in members)
                lower, upper = wilson_interval(successes, len(members), 0.95)
                calibration_bins.append(
                    {
                        "bin_index": bin_index,
                        "lower_probability_inclusive": bin_index / 10.0,
                        "upper_probability_exclusive": (bin_index + 1) / 10.0,
                        "sample_size": len(members),
                        "successes": successes,
                        "mean_prediction": math.fsum(
                            all_predictions[index] for index in members
                        )
                        / len(members),
                        "observed_rate": successes / len(members),
                        "wilson_95_lower": lower,
                        "wilson_95_upper": upper,
                    }
                )
            probability_models[target_name] = {
                "raw_model": raw_model.to_dict(),
                "selected_l2": probability_l2,
                "validation_brier_by_l2": probability_scores,
                "selected_calibration_method": choice.name.upper(),
                "calibrator": _calibrator_payload(refit),
                "calibration_sample_size": len(all_ids),
                "calibration_positive_events": positives,
                "calibration_negative_events": negatives,
                "calibration_period": {
                    "start": min(by_id[key].signal_date for key in all_ids).isoformat(),
                    "end": max(by_id[key].signal_date for key in all_ids).isoformat(),
                },
                "validation_brier": validation_brier,
                "unconditional_brier": base_brier,
                "expected_calibration_error": ece,
                "development_base_rate": base_rate,
                "calibration_bins": calibration_bins,
            }
        projected_validation = {
            record_id: coherent_exit_probabilities(
                {
                    target: validation_predictions[target][record_id]
                    for target in TARGETS
                }
            )
            for record_id in sorted(validation_selected_ids)
        }
        for target_name in TARGETS:
            record_ids = tuple(sorted(projected_validation))
            predictions = tuple(
                float(projected_validation[key]["metrics"][target_name])
                for key in record_ids
            )
            outcomes = tuple(by_id[key].targets[target_name] for key in record_ids)
            projected_brier = brier_score(predictions, outcomes)
            projected_ece = expected_calibration_error(predictions, outcomes)
            projected_base_brier = unconditional_brier_score(
                outcomes, validation_base_rates[target_name]
            )
            probability_models[target_name][
                "projected_validation_brier"
            ] = projected_brier
            probability_models[target_name][
                "projected_validation_expected_calibration_error"
            ] = projected_ece
            probability_models[target_name][
                "projected_validation_unconditional_brier"
            ] = projected_base_brier
            if projected_brier >= projected_base_brier:
                result["reasons"].append(
                    "%s coherent calibration does not beat unconditional Brier"
                    % target_name
                )
            if projected_ece > float(calibration_policy["maximum_ece"]):
                result["reasons"].append(
                    "%s coherent calibration ECE exceeds tolerance" % target_name
                )

        projected_all = {
            record_id: coherent_exit_probabilities(
                {
                    target: refit_predictions[target][record_id]
                    for target in TARGETS
                }
            )
            for record_id in all_ids
        }
        joint_bins = []
        for bin_index in range(10):
            members = tuple(
                key
                for key in all_ids
                if min(
                    9,
                    int(
                        float(projected_all[key]["metrics"]["POP_NET"])
                        * 10.0
                    ),
                )
                == bin_index
            )
            if not members:
                continue
            intervals = {}
            for target_name in TARGETS:
                successes = sum(by_id[key].targets[target_name] for key in members)
                lower, upper = wilson_interval(successes, len(members), 0.95)
                intervals[target_name] = {
                    "successes": successes,
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                    "observed_rate": successes / len(members),
                    "mean_projected_probability": math.fsum(
                        float(projected_all[key]["metrics"][target_name])
                        for key in members
                    )
                    / len(members),
                }
            joint_bins.append(
                {
                    "bin_index": bin_index,
                    "bucket_axis": "PROJECTED_POP_NET",
                    "sample_size": len(members),
                    "targets": intervals,
                }
            )

        scenario_profile = {}
        for category in EXIT_CATEGORIES:
            members = tuple(
                item for item in selected if item.outcome_class == category
            )
            scenario_profile[category] = {
                "sample_size": len(members),
                "mean_net_return_on_risk": (
                    None
                    if not members
                    else math.fsum(item.return_on_risk for item in members)
                    / len(members)
                ),
                "minimum_net_return_on_risk": (
                    None
                    if not members
                    else min(item.return_on_risk for item in members)
                ),
                "maximum_net_return_on_risk": (
                    None
                    if not members
                    else max(item.return_on_risk for item in members)
                ),
            }
        result["probability_models"] = probability_models
        result["joint_calibration_bins"] = joint_bins
        result["scenario_return_profile"] = scenario_profile
        result["probability_coherence_policy"] = (
            "FIVE_EXIT_CATEGORY_SIMPLEX_PROJECTION_V1"
        )
    except (ArithmeticError, KeyError, ModelingV2Error, ValueError) as exc:
        result["reasons"].append("development model unavailable: %s" % str(exc))
    if not result["reasons"]:
        result["state"] = "VALIDATION_PASS"
    elif result.get("selection_model_validation", {}).get("research_expectancy", -math.inf) > 0.0:
        result["state"] = "RESEARCH_PASS_ONLY_REJECT_VALIDATION"
    else:
        result["state"] = "REJECT_RESEARCH"
    return result


def _partition_ids(
    connection: sqlite3.Connection,
    hypothesis_id: str,
    dates: Sequence[date],
) -> Tuple[str, ...]:
    date_values = tuple(item.isoformat() for item in dates)
    placeholders = ",".join("?" for _ in date_values)
    rows = connection.execute(
        "SELECT record_id FROM candidate_ledger WHERE hypothesis_id = ? AND signal_date IN (%s) ORDER BY record_id"
        % placeholders,
        (hypothesis_id,) + date_values,
    )
    return tuple(str(row[0]) for row in rows)


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


def freeze_historical_v2_models(
    *, outcome_database: Path, artifact_path: Path, evidence_registry_path: Path
) -> Mapping[str, Any]:
    """Freeze all 90 development models without reading holdout outcomes."""

    artifact = Path(artifact_path).expanduser().resolve()
    registry_path = Path(evidence_registry_path).expanduser().resolve()
    for supplied, label in ((artifact, "model artifact"), (registry_path, "evidence registry")):
        try:
            supplied.relative_to(DEFAULT_EVIDENCE_ROOT.resolve())
        except ValueError as exc:
            raise ModelingV2Error("%s must remain inside Cultra evidence storage" % label) from exc
    if artifact.exists() or artifact.with_suffix(artifact.suffix + ".manifest.json").exists():
        raise ModelingV2Error("model artifact already exists")
    connection, outcome_manifest = _open_outcomes(outcome_database)
    normalized_database = Path(str(outcome_manifest["normalized_database"])).resolve()
    sessions = _calendar(connection, normalized_database)
    split = frozen_calendar_split(sessions)
    protocol = load_historical_campaign_protocol()
    frozen_at = datetime.now(timezone.utc)
    hypotheses = []
    partitions: Dict[str, EvidencePartitions] = {}
    try:
        for hypothesis in FROZEN_HYPOTHESIS_REGISTRY:
            hypotheses.append(_hypothesis_artifact(connection, sessions, hypothesis, protocol))
            partitions[hypothesis.hypothesis_id] = EvidencePartitions(
                training_observation_ids=_partition_ids(connection, hypothesis.hypothesis_id, split["research"]),
                validation_observation_ids=_partition_ids(
                    connection,
                    hypothesis.hypothesis_id,
                    split["tuning"] + split["validation"],
                ),
                holdout_observation_ids=_partition_ids(connection, hypothesis.hypothesis_id, split["holdout"]),
            )
    finally:
        connection.close()
    payload = {
        "schema": MODEL_ARTIFACT_SCHEMA,
        "model_version": MODEL_VERSION,
        "model_frozen_at": frozen_at.isoformat(),
        "outcome_database": str(Path(outcome_database).expanduser().resolve()),
        "outcome_database_sha256": _sha256(Path(outcome_database).expanduser().resolve()),
        "outcome_manifest_sha256": _sha256(Path(outcome_database).expanduser().resolve().with_suffix(Path(outcome_database).suffix + ".manifest.json")),
        "protocol_hash": historical_protocol_hash(),
        "hypothesis_registry_hash": HYPOTHESIS_REGISTRY_HASH,
        "holdout_outcomes_read": False,
        "selection_rule": "PREDICTED_NET_RETURN_ON_RISK_GREATER_THAN_ZERO",
        "no_top_n": True,
        "calendar_split": {
            key: {"sessions": len(values), "start": values[0].isoformat(), "end": values[-1].isoformat()}
            for key, values in split.items()
        },
        "hypotheses": hypotheses,
        "network_attempted": False,
    }
    artifact.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _private_json(artifact, payload)
    manifest = {
        "schema": "cultra.frozen-models-v2-manifest.v1",
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": _sha256(artifact),
        "outcome_database_sha256": payload["outcome_database_sha256"],
        "hypothesis_registry_hash": HYPOTHESIS_REGISTRY_HASH,
        "network_attempted": False,
    }
    _private_json(artifact.with_suffix(artifact.suffix + ".manifest.json"), manifest)

    by_id = {item.hypothesis_id: item for item in FROZEN_HYPOTHESIS_REGISTRY}
    with EvidenceRegistry(registry_path) as registry:
        for result in hypotheses:
            hypothesis_id = str(result["hypothesis_id"])
            definition = by_id[hypothesis_id]
            identity = FrozenEvidenceIdentity(
                strategy_family=hypothesis_id,
                catalog_version=CATALOG_VERSION,
                hypothesis_fingerprint=_payload_hash(
                    {
                        "definition": definition.__dict__,
                        "protocol_hash": payload["protocol_hash"],
                        "model_artifact": _payload_hash(result),
                    }
                ),
                cost_model_version=str(protocol["cost_policy"]["version"]),
                exit_policy_version=definition.exit_policy,
                pop_model_version=MODEL_VERSION,
                pop_model_artifact_id=_payload_hash(result),
                model_frozen_at=frozen_at,
            )
            selected_partitions = partitions[hypothesis_id]
            registry.register(identity, selected_partitions, now=frozen_at)
            if result["state"] == "VALIDATION_PASS":
                registry.advance_development(
                    hypothesis_id,
                    RegistryState.RESEARCH_PASS,
                    selected_partitions.development_fingerprint,
                    now=frozen_at,
                )
                registry.advance_development(
                    hypothesis_id,
                    RegistryState.VALIDATION_PASS,
                    selected_partitions.development_fingerprint,
                    now=frozen_at,
                )
            elif result["state"] == "RESEARCH_PASS_ONLY_REJECT_VALIDATION":
                registry.advance_development(
                    hypothesis_id,
                    RegistryState.RESEARCH_PASS,
                    selected_partitions.development_fingerprint,
                    now=frozen_at,
                )
                registry.reject_development(
                    hypothesis_id,
                    RegistryState.VALIDATION_PASS,
                    selected_partitions.development_fingerprint,
                    now=frozen_at,
                )
            else:
                registry.reject_development(
                    hypothesis_id,
                    RegistryState.RESEARCH_PASS,
                    selected_partitions.development_fingerprint,
                    now=frozen_at,
                )
    return dict(
        manifest,
        validation_pass=sum(item["state"] == "VALIDATION_PASS" for item in hypotheses),
        rejected=sum(item["state"] != "VALIDATION_PASS" for item in hypotheses),
        evidence_registry=str(registry_path),
    )


def load_frozen_models_v2(path: Path) -> Mapping[str, Any]:
    artifact = Path(path).expanduser().resolve()
    manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelingV2Error("frozen V2 model artifact is unavailable") from exc
    if (
        payload.get("schema") != MODEL_ARTIFACT_SCHEMA
        or manifest.get("schema") != "cultra.frozen-models-v2-manifest.v1"
        or Path(str(manifest.get("artifact", ""))).resolve() != artifact
        or int(manifest.get("artifact_bytes", -1)) != artifact.stat().st_size
        or manifest.get("artifact_sha256") != _sha256(artifact)
        or payload.get("hypothesis_registry_hash") != HYPOTHESIS_REGISTRY_HASH
        or payload.get("protocol_hash") != historical_protocol_hash()
        or payload.get("holdout_outcomes_read") is not False
        or len(payload.get("hypotheses", ())) != len(FROZEN_HYPOTHESIS_REGISTRY)
    ):
        raise ModelingV2Error("frozen V2 model artifact does not reconcile")
    return payload


def score_current_candidate_v2(
    models: Mapping[str, Any],
    *,
    hypothesis_id: str,
    features: Mapping[str, float],
    finite_maximum_loss: float,
) -> Mapping[str, Any]:
    """Apply one frozen model without claiming quote/evidence ticket readiness."""

    risk = float(finite_maximum_loss)
    if not math.isfinite(risk) or risk <= 0.0:
        raise ModelingV2Error("current candidate requires finite positive maximum loss")
    matches = tuple(
        item
        for item in models.get("hypotheses", ())
        if item.get("hypothesis_id") == hypothesis_id
    )
    if len(matches) != 1:
        raise ModelingV2Error("current candidate hypothesis is not uniquely frozen")
    artifact = matches[0]
    if artifact.get("state") != "VALIDATION_PASS":
        raise ModelingV2Error("current candidate model did not pass development validation")
    return_model = FrozenLinearModel.from_dict(artifact["return_model"])
    point_return = return_model.predict_one(features)
    offset = float(
        artifact["return_model_uncertainty"][
            "conservative_return_on_risk_offset"
        ]
    )
    if not math.isfinite(offset) or offset > 0.0:
        raise ModelingV2Error(
            "current candidate conservative offset is invalid"
        )
    conservative_return = point_return + offset
    raw_probabilities = {}
    for target in TARGETS:
        probability_model = artifact["probability_models"][target]
        raw_probabilities[target] = _score_probability(probability_model, features)
    coherent = coherent_exit_probabilities(raw_probabilities)
    coherent_metrics = coherent["metrics"]
    bin_index = min(9, int(float(coherent_metrics["POP_NET"]) * 10.0))
    selected_bin = next(
        (
            item
            for item in artifact.get("joint_calibration_bins", ())
            if int(item["bin_index"]) == bin_index
        ),
        None,
    )
    probability_ready = selected_bin is not None and int(
        selected_bin["sample_size"]
    ) >= 20
    probabilities = {}
    for target in TARGETS:
        point = float(coherent_metrics[target])
        if not probability_ready:
            interval = None
            bin_sample_size = (
                0 if selected_bin is None else int(selected_bin["sample_size"])
            )
        else:
            target_interval = selected_bin["targets"][target]
            # The simplex projection can move a point outside its marginal
            # Wilson interval.  Enveloping both is conservative and keeps the
            # interval tied to the same joint bucket used by every target.
            interval = {
                "lower": min(
                    point, float(target_interval["wilson_95_lower"])
                ),
                "upper": max(
                    point, float(target_interval["wilson_95_upper"])
                ),
                "confidence": 0.95,
                "method": "JOINT_BUCKET_WILSON_PROJECTION_ENVELOPE",
            }
            bin_sample_size = int(selected_bin["sample_size"])
        probability_model = artifact["probability_models"][target]
        probabilities[target] = {
            "point": point,
            "interval": interval,
            "applicable_calibration_bin_sample_size": bin_sample_size,
            "calibration_period": probability_model["calibration_period"],
            "model_version": MODEL_VERSION,
        }
    scenario_profile = artifact.get("scenario_return_profile")
    if not isinstance(scenario_profile, Mapping):
        raise ModelingV2Error("current candidate scenario return profile is missing")
    scenario_returns = {}
    conservative_scenario_returns = {}
    scenario_profile_ready = True
    for category in EXIT_CATEGORIES:
        profile = scenario_profile.get(category)
        probability = float(coherent["categories"][category])
        if (
            not isinstance(profile, Mapping)
            or int(profile.get("sample_size", 0)) <= 0
            or profile.get("mean_net_return_on_risk") is None
        ):
            if probability > 1e-12:
                scenario_profile_ready = False
            point_category_return = 0.0
        else:
            point_category_return = float(profile["mean_net_return_on_risk"])
            if not math.isfinite(point_category_return):
                raise ModelingV2Error("current candidate scenario return is invalid")
        conservative_category_return = min(
            point_category_return, point_category_return + offset
        )
        # A defined-risk one-unit ticket cannot lose more than its maximum
        # loss, even under the conservative scenario distribution.
        point_category_return = max(-1.0, point_category_return)
        conservative_category_return = max(-1.0, conservative_category_return)
        scenario_returns[category] = point_category_return
        conservative_scenario_returns[category] = conservative_category_return
    scenario_point_return = math.fsum(
        float(coherent["categories"][category]) * scenario_returns[category]
        for category in EXIT_CATEGORIES
    )
    scenario_conservative_return = math.fsum(
        float(coherent["categories"][category])
        * conservative_scenario_returns[category]
        for category in EXIT_CATEGORIES
    )
    model_candidate_eligible = (
        point_return > 0.0
        and conservative_return > 0.0
        and scenario_point_return > 0.0
        and scenario_conservative_return > 0.0
        and probability_ready
        and scenario_profile_ready
    )
    return {
        "hypothesis_id": hypothesis_id,
        "model_version": MODEL_VERSION,
        "selection_model_point_return_on_maximum_loss": point_return,
        "selection_model_conservative_return_on_maximum_loss": conservative_return,
        "point_expected_return_on_maximum_loss": scenario_point_return,
        "conservative_expected_return_on_maximum_loss": scenario_conservative_return,
        "point_net_ev_dollars": scenario_point_return * risk,
        "conservative_net_ev_dollars": scenario_conservative_return * risk,
        "joint_exit_probabilities": coherent["categories"],
        "probability_projection_l1_distance": coherent[
            "projection_l1_distance"
        ],
        "scenario_net_returns_on_risk": scenario_returns,
        "conservative_scenario_net_returns_on_risk": (
            conservative_scenario_returns
        ),
        "probabilities": probabilities,
        "model_candidate_eligible": model_candidate_eligible,
        "manual_ticket_ready": False,
        "manual_ticket_blockers": [
            "SEPARATE_HOLDOUT_STATE_REQUIRED",
            "FRESH_SCHWAB_EXACT_LEG_QUOTES_REQUIRED",
            "COMPLETE_CURRENT_COSTS_AND_EVENT_CLEARANCE_REQUIRED",
            "CURRENT_SCENARIO_ECONOMICS_MUST_RECONCILE_TO_EXACT_LEG_PAYOFF",
        ],
    }


def _score_probability(
    probability_model: Mapping[str, Any], features: Mapping[str, float]
) -> float:
    raw = FrozenLinearModel.from_dict(probability_model["raw_model"]).predict_one(
        features
    )
    calibrator = calibrator_from_payload(probability_model["calibrator"])
    value = float(calibrator.predict_one(raw))  # type: ignore[attr-defined]
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ModelingV2Error("current calibrated probability is invalid")
    return value


__all__ = [
    "EXIT_CATEGORIES",
    "FrozenLinearModel",
    "MODEL_VERSION",
    "ModelingV2Error",
    "TARGETS",
    "calibrator_from_payload",
    "coherent_exit_probabilities",
    "fit_linear_model",
    "freeze_historical_v2_models",
    "frozen_calendar_split",
    "load_frozen_models_v2",
    "score_current_candidate_v2",
]
