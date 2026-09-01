"""Chronological out-of-fold POP calibration and frozen model artifacts."""

from dataclasses import dataclass
from datetime import date, datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .calibration import (
    IsotonicCalibrator,
    LogisticCalibrator,
    brier_score,
    expected_calibration_error,
    wilson_interval,
)
from .domain import HistoricalObservation, ProbabilityBundle, ProbabilityEstimate
from .validation import walk_forward_development_splits


POP_ARTIFACT_SCHEMA = "cultra.oof-pop-model.v1"


class ProbabilityTarget(str, Enum):
    POP_NET = "POP_NET"
    P_TARGET = "P_TARGET"
    P_STOP = "P_STOP"
    P_MAX_LOSS = "P_MAX_LOSS"


@dataclass(frozen=True)
class _CalendarOOFFold:
    """A POP fold whose embargo is measured on the market-session calendar.

    Strategy observations are intentionally sparse: a family does not emit a
    trade every session. Counting only trade dates would silently turn a
    60-session embargo into a much longer and strategy-dependent gap. These
    private folds retain only real observations while deriving every boundary
    from the complete Cultra session calendar.
    """

    fold_index: int
    training: Tuple[HistoricalObservation, ...]
    validation: Tuple[HistoricalObservation, ...]
    embargo_sessions: int
    embargo_start: date
    embargo_end: date

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index cannot be negative")
        if not self.training or not self.validation:
            raise ValueError("calendar POP fold observations cannot be empty")
        if self.embargo_sessions <= 0:
            raise ValueError("embargo_sessions must be positive")
        if not max(item.session_date for item in self.training) < self.embargo_start:
            raise ValueError("calendar POP training reaches the embargo")
        if not self.embargo_start <= self.embargo_end < min(
            item.session_date for item in self.validation
        ):
            raise ValueError("calendar POP embargo is not chronological")


def _calendar_oof_folds(
    observations: Sequence[HistoricalObservation],
    session_calendar: Sequence[date],
    *,
    min_training_sessions: int,
    validation_sessions: int,
    embargo_sessions: int,
    step_sessions: Optional[int],
) -> Tuple[_CalendarOOFFold, ...]:
    """Build expanding folds using complete market sessions, not trade dates."""

    for name, value in (
        ("min_training_sessions", min_training_sessions),
        ("validation_sessions", validation_sessions),
        ("embargo_sessions", embargo_sessions),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("%s must be a positive integer" % name)
    step = validation_sessions if step_sessions is None else step_sessions
    if isinstance(step, bool) or not isinstance(step, int) or step < validation_sessions:
        raise ValueError("step_sessions must be at least validation_sessions")

    calendar = tuple(session_calendar)
    if not calendar or any(not isinstance(item, date) for item in calendar):
        raise ValueError("session_calendar must contain dates")
    if len(calendar) != len(set(calendar)) or calendar != tuple(sorted(calendar)):
        raise ValueError("session_calendar must be unique and chronological")
    calendar_set = set(calendar)
    if any(item.session_date not in calendar_set for item in observations):
        raise ValueError("OOF observation falls outside the development calendar")

    by_session: Dict[date, list] = {}
    for observation in observations:
        by_session.setdefault(observation.session_date, []).append(observation)

    first_validation = min_training_sessions + embargo_sessions
    if first_validation + validation_sessions > len(calendar):
        raise ValueError("insufficient calendar sessions for one walk-forward fold")
    folds = []
    validation_start = first_validation
    scheduled_fold_index = 0
    while validation_start + validation_sessions <= len(calendar):
        training_end = validation_start - embargo_sessions
        training_dates = calendar[:training_end]
        embargo_dates = calendar[training_end:validation_start]
        validation_dates = calendar[
            validation_start : validation_start + validation_sessions
        ]
        training = tuple(
            sorted(
                (
                    observation
                    for session in training_dates
                    for observation in by_session.get(session, ())
                ),
                key=lambda item: (item.session_date, item.observation_id),
            )
        )
        validation = tuple(
            sorted(
                (
                    observation
                    for session in validation_dates
                    for observation in by_session.get(session, ())
                ),
                key=lambda item: (item.session_date, item.observation_id),
            )
        )
        # A no-signal validation window contributes no outcomes. It remains a
        # calendar gap but cannot produce an OOF score, so it is skipped rather
        # than populated with synthetic trades.
        if validation:
            if not training:
                raise ValueError("calendar POP fold has no training observations")
            folds.append(
                _CalendarOOFFold(
                    fold_index=scheduled_fold_index,
                    training=training,
                    validation=validation,
                    embargo_sessions=len(embargo_dates),
                    embargo_start=embargo_dates[0],
                    embargo_end=embargo_dates[-1],
                )
            )
        scheduled_fold_index += 1
        validation_start += step
    if not folds:
        raise ValueError("no qualifying observations in calendar validation folds")
    return tuple(folds)


def _text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("%s is required" % name)
    return normalized


def _probability(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError("%s must be a finite probability" % name)
    return converted


def _aware_iso(value: datetime, name: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("%s must be timezone-aware" % name)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class POPBucketIdentity:
    strategy_family: str
    regime_id: str
    target: ProbabilityTarget
    bucket_version: str

    def __post_init__(self) -> None:
        for name in ("strategy_family", "regime_id", "bucket_version"):
            object.__setattr__(self, name, _text(getattr(self, name), name))
        if not isinstance(self.target, ProbabilityTarget):
            raise TypeError("target must be ProbabilityTarget")

    @property
    def bucket_id(self) -> str:
        return _canonical_hash(
            {
                "strategy_family": self.strategy_family,
                "regime_id": self.regime_id,
                "target": self.target.value,
                "bucket_version": self.bucket_version,
            }
        )


@dataclass(frozen=True)
class OOFPOPObservation:
    observation_id: str
    session_date: date
    bucket_id: str
    raw_probability: float
    outcome: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "observation_id", _text(self.observation_id, "observation_id"))
        object.__setattr__(self, "bucket_id", _text(self.bucket_id, "bucket_id"))
        object.__setattr__(
            self,
            "raw_probability",
            _probability(self.raw_probability, "raw_probability"),
        )
        outcome = int(self.outcome) if isinstance(self.outcome, bool) else self.outcome
        if outcome not in (0, 1):
            raise ValueError("outcome must be binary")
        object.__setattr__(self, "outcome", int(outcome))


@dataclass(frozen=True)
class OOFPrediction:
    observation_id: str
    session_date: date
    fold_index: int
    raw_probability: float
    calibrated_probability: float
    training_base_rate: float
    outcome: int

    def __post_init__(self) -> None:
        _text(self.observation_id, "observation_id")
        if self.fold_index < 0:
            raise ValueError("fold_index cannot be negative")
        _probability(self.raw_probability, "raw_probability")
        _probability(self.calibrated_probability, "calibrated_probability")
        _probability(self.training_base_rate, "training_base_rate")
        if self.outcome not in (0, 1):
            raise ValueError("OOF outcome must be binary")


@dataclass(frozen=True)
class IntervalProvenance:
    method: str
    confidence: float
    successes: int
    sample_size: int
    lower: float
    upper: float

    def __post_init__(self) -> None:
        if self.method != "WILSON_SCORE":
            raise ValueError("interval method must be WILSON_SCORE")
        if not math.isclose(self.confidence, 0.95, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("POP interval confidence must be 95%")
        if self.sample_size <= 0 or not 0 <= self.successes <= self.sample_size:
            raise ValueError("interval counts are invalid")
        expected = wilson_interval(self.successes, self.sample_size, self.confidence)
        if not (
            math.isclose(self.lower, expected[0], rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(self.upper, expected[1], rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError("interval bounds do not reproduce from provenance")


@dataclass(frozen=True)
class OOFFoldMetrics:
    fold_index: int
    training_start: date
    training_end: date
    validation_start: date
    validation_end: date
    embargo_sessions: int
    training_count: int
    validation_count: int
    logistic_brier: float
    isotonic_brier: float
    base_rate_brier: float
    logistic_converged: bool

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index cannot be negative")
        if not self.training_start <= self.training_end < self.validation_start <= self.validation_end:
            raise ValueError("OOF fold periods are not chronological")
        if self.embargo_sessions != 60:
            raise ValueError("OOF POP folds require a 60-session embargo")
        if self.training_count <= 0 or self.validation_count <= 0:
            raise ValueError("OOF fold counts must be positive")
        for name in ("logistic_brier", "isotonic_brier", "base_rate_brier"):
            _probability(getattr(self, name), name)
        if not isinstance(self.logistic_converged, bool):
            raise TypeError("logistic_converged must be bool")


@dataclass(frozen=True)
class FrozenCalibrator:
    method: str
    sample_size: int
    intercept: Optional[float] = None
    slope: Optional[float] = None
    converged: Optional[bool] = None
    x_thresholds: Tuple[float, ...] = ()
    y_values: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.method not in ("logistic", "isotonic"):
            raise ValueError("unsupported frozen calibrator method")
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if self.method == "logistic":
            if self.intercept is None or self.slope is None or self.converged is not True:
                raise ValueError("frozen logistic calibrator must be converged and complete")
            if not math.isfinite(float(self.intercept)) or not math.isfinite(float(self.slope)):
                raise ValueError("frozen logistic coefficients must be finite")
            if self.x_thresholds or self.y_values:
                raise ValueError("logistic calibrator cannot contain isotonic thresholds")
        else:
            if not self.x_thresholds or len(self.x_thresholds) != len(self.y_values):
                raise ValueError("frozen isotonic calibrator thresholds are incomplete")
            if self.intercept is not None or self.slope is not None:
                raise ValueError("isotonic calibrator cannot contain logistic coefficients")
            # Reuse the calibrator's monotonicity and probability validation.
            IsotonicCalibrator(self.x_thresholds, self.y_values, self.sample_size)

    def predict_one(self, raw_probability: float) -> float:
        if self.method == "logistic":
            calibrator = LogisticCalibrator(
                float(self.intercept), float(self.slope), self.sample_size, True
            )
        else:
            calibrator = IsotonicCalibrator(
                self.x_thresholds, self.y_values, self.sample_size
            )
        return calibrator.predict_one(raw_probability)


@dataclass(frozen=True)
class OOFPOPModelArtifact:
    artifact_id: str
    bucket: POPBucketIdentity
    model_version: str
    selected_method: str
    frozen_calibrator: FrozenCalibrator
    model_frozen_at: datetime
    holdout_start: date
    development_start: date
    development_end: date
    oof_brier_score: float
    base_rate_brier_score: float
    expected_calibration_error: float
    interval: IntervalProvenance
    fold_metrics: Tuple[OOFFoldMetrics, ...]
    oof_predictions: Tuple[OOFPrediction, ...]
    development_data_fingerprint: str
    schema: str = POP_ARTIFACT_SCHEMA

    def __post_init__(self) -> None:
        _text(self.artifact_id, "artifact_id")
        _text(self.model_version, "model_version")
        _aware_iso(self.model_frozen_at, "model_frozen_at")
        if self.schema != POP_ARTIFACT_SCHEMA:
            raise ValueError("unsupported POP artifact schema")
        if self.selected_method != self.frozen_calibrator.method:
            raise ValueError("selected method and frozen calibrator disagree")
        if not self.fold_metrics or not self.oof_predictions:
            raise ValueError("OOF evidence is required")
        if self.development_start > self.development_end:
            raise ValueError("development period is invalid")
        if self.development_end >= self.holdout_start:
            raise ValueError("development data reaches the untouched holdout")
        if self.interval.sample_size != len(self.oof_predictions):
            raise ValueError("interval sample size does not match OOF predictions")
        identifiers = tuple(item.observation_id for item in self.oof_predictions)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("OOF prediction observations are duplicated")
        if any(item.session_date >= self.holdout_start for item in self.oof_predictions):
            raise ValueError("OOF predictions overlap the untouched holdout")
        if self.interval.successes != sum(item.outcome for item in self.oof_predictions):
            raise ValueError("interval successes do not match OOF outcomes")
        predictions = tuple(
            item.calibrated_probability for item in self.oof_predictions
        )
        baselines = tuple(item.training_base_rate for item in self.oof_predictions)
        outcomes = tuple(item.outcome for item in self.oof_predictions)
        if not math.isclose(
            self.oof_brier_score,
            brier_score(predictions, outcomes),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("OOF Brier score is not reproducible")
        if not math.isclose(
            self.base_rate_brier_score,
            brier_score(baselines, outcomes),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("OOF base-rate Brier score is not reproducible")
        if not math.isclose(
            self.expected_calibration_error,
            expected_calibration_error(predictions, outcomes),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("OOF expected calibration error is not reproducible")
        fold_indexes = {item.fold_index for item in self.fold_metrics}
        if fold_indexes != {item.fold_index for item in self.oof_predictions}:
            raise ValueError("OOF fold summaries and predictions disagree")
        for fold in self.fold_metrics:
            rows = tuple(
                item for item in self.oof_predictions if item.fold_index == fold.fold_index
            )
            if len(rows) != fold.validation_count:
                raise ValueError("OOF fold validation count is not reproducible")
            if min(item.session_date for item in rows) != fold.validation_start or max(
                item.session_date for item in rows
            ) != fold.validation_end:
                raise ValueError("OOF fold validation period is not reproducible")
        if self.artifact_id != _canonical_hash(self._payload(include_artifact_id=False)):
            raise ValueError("POP artifact ID does not match its immutable contents")

    def _payload(self, *, include_artifact_id: bool) -> Dict[str, Any]:
        payload = {
            "schema": self.schema,
            "bucket": {
                "strategy_family": self.bucket.strategy_family,
                "regime_id": self.bucket.regime_id,
                "target": self.bucket.target.value,
                "bucket_version": self.bucket.bucket_version,
                "bucket_id": self.bucket.bucket_id,
            },
            "model_version": self.model_version,
            "selected_method": self.selected_method,
            "frozen_calibrator": {
                "method": self.frozen_calibrator.method,
                "sample_size": self.frozen_calibrator.sample_size,
                "intercept": self.frozen_calibrator.intercept,
                "slope": self.frozen_calibrator.slope,
                "converged": self.frozen_calibrator.converged,
                "x_thresholds": list(self.frozen_calibrator.x_thresholds),
                "y_values": list(self.frozen_calibrator.y_values),
            },
            "model_frozen_at": _aware_iso(self.model_frozen_at, "model_frozen_at"),
            "holdout_start": self.holdout_start.isoformat(),
            "development_start": self.development_start.isoformat(),
            "development_end": self.development_end.isoformat(),
            "oof_brier_score": self.oof_brier_score,
            "base_rate_brier_score": self.base_rate_brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "interval": {
                "method": self.interval.method,
                "confidence": self.interval.confidence,
                "successes": self.interval.successes,
                "sample_size": self.interval.sample_size,
                "lower": self.interval.lower,
                "upper": self.interval.upper,
            },
            "fold_metrics": [
                {
                    "fold_index": item.fold_index,
                    "training_start": item.training_start.isoformat(),
                    "training_end": item.training_end.isoformat(),
                    "validation_start": item.validation_start.isoformat(),
                    "validation_end": item.validation_end.isoformat(),
                    "embargo_sessions": item.embargo_sessions,
                    "training_count": item.training_count,
                    "validation_count": item.validation_count,
                    "logistic_brier": item.logistic_brier,
                    "isotonic_brier": item.isotonic_brier,
                    "base_rate_brier": item.base_rate_brier,
                    "logistic_converged": item.logistic_converged,
                }
                for item in self.fold_metrics
            ],
            "oof_predictions": [
                {
                    "observation_id": item.observation_id,
                    "session_date": item.session_date.isoformat(),
                    "fold_index": item.fold_index,
                    "raw_probability": item.raw_probability,
                    "calibrated_probability": item.calibrated_probability,
                    "training_base_rate": item.training_base_rate,
                    "outcome": item.outcome,
                }
                for item in self.oof_predictions
            ],
            "development_data_fingerprint": self.development_data_fingerprint,
        }
        if include_artifact_id:
            payload["artifact_id"] = self.artifact_id
        return payload

    def to_dict(self) -> Dict[str, Any]:
        return self._payload(include_artifact_id=True)

    def verify(self) -> bool:
        return self.artifact_id == _canonical_hash(
            self._payload(include_artifact_id=False)
        )

    def probability_estimate(self, raw_probability: float) -> ProbabilityEstimate:
        """Create a ticket-ready estimate directly from this frozen artifact."""

        point = self.frozen_calibrator.predict_one(raw_probability)
        effective_successes = int(round(point * self.interval.sample_size))
        lower, upper = wilson_interval(
            effective_successes, self.interval.sample_size, 0.95
        )
        if not lower <= point <= upper:
            raise ValueError("predicted point falls outside its 95% interval")
        return ProbabilityEstimate(
            point=point,
            lower=lower,
            upper=upper,
            sample_size=self.interval.sample_size,
            model_version=self.model_version,
            calibration_start=self.development_start,
            calibration_end=self.development_end,
            confidence_level=0.95,
            interval_method="WILSON_SCORE_PREDICTED_COUNT",
            bucket_id=self.bucket.bucket_id,
            artifact_id=self.artifact_id,
            target_name=self.bucket.target.value,
        )


def build_probability_bundle(
    artifacts: Mapping[ProbabilityTarget, OOFPOPModelArtifact],
    raw_probabilities: Mapping[ProbabilityTarget, float],
) -> ProbabilityBundle:
    """Bind all four ticket probabilities to verified frozen OOF artifacts."""

    required = tuple(ProbabilityTarget)
    if set(artifacts) != set(required) or set(raw_probabilities) != set(required):
        raise ValueError("all four probability targets are required")
    selected = tuple(artifacts[target] for target in required)
    if not all(item.verify() for item in selected):
        raise ValueError("a POP artifact failed content verification")
    identity = {
        (
            item.bucket.strategy_family,
            item.bucket.regime_id,
            item.bucket.bucket_version,
            item.model_version,
            item.development_start,
            item.development_end,
        )
        for item in selected
    }
    if len(identity) != 1:
        raise ValueError("POP artifacts do not share one frozen strategy/regime model")
    estimates = {
        target: artifacts[target].probability_estimate(raw_probabilities[target])
        for target in required
    }
    return ProbabilityBundle(
        pop_net=estimates[ProbabilityTarget.POP_NET],
        p_target=estimates[ProbabilityTarget.P_TARGET],
        p_stop=estimates[ProbabilityTarget.P_STOP],
        p_max_loss=estimates[ProbabilityTarget.P_MAX_LOSS],
    )


def _freeze_calibrator(
    method: str, calibrator: object
) -> FrozenCalibrator:
    if method == "logistic":
        assert isinstance(calibrator, LogisticCalibrator)
        return FrozenCalibrator(
            method="logistic",
            sample_size=calibrator.sample_size,
            intercept=calibrator.intercept,
            slope=calibrator.slope,
            converged=calibrator.converged,
        )
    assert isinstance(calibrator, IsotonicCalibrator)
    return FrozenCalibrator(
        method="isotonic",
        sample_size=calibrator.sample_size,
        x_thresholds=calibrator.x_thresholds,
        y_values=calibrator.y_values,
    )


def build_oof_pop_model(
    observations: Sequence[OOFPOPObservation],
    bucket: POPBucketIdentity,
    *,
    model_version: str,
    holdout_start: date,
    model_frozen_at: datetime,
    min_training_sessions: int = 120,
    validation_sessions: int = 20,
    embargo_sessions: int = 60,
    step_sessions: Optional[int] = None,
    session_calendar: Optional[Sequence[date]] = None,
) -> OOFPOPModelArtifact:
    """Select and freeze a calibrator using chronological OOF predictions only."""

    model_version = _text(model_version, "model_version")
    _aware_iso(model_frozen_at, "model_frozen_at")
    if not observations:
        raise ValueError("OOF observations are required")
    identifiers = tuple(item.observation_id for item in observations)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("OOF observation ids must be unique")
    if any(item.bucket_id != bucket.bucket_id for item in observations):
        raise ValueError("OOF observations cross calibrated bucket identities")
    if any(item.session_date >= holdout_start for item in observations):
        raise ValueError("OOF development observations overlap the holdout")

    by_id = {item.observation_id: item for item in observations}
    splitting_observations = tuple(
        HistoricalObservation(
            observation_id=item.observation_id,
            session_date=item.session_date,
            cluster_id=bucket.bucket_id,
            net_pnl=float(item.outcome),
        )
        for item in observations
    )
    if session_calendar is None:
        folds = walk_forward_development_splits(
            splitting_observations,
            min_training_sessions=min_training_sessions,
            validation_sessions=validation_sessions,
            embargo_sessions=embargo_sessions,
            step_sessions=step_sessions,
        )
    else:
        folds = _calendar_oof_folds(
            splitting_observations,
            session_calendar,
            min_training_sessions=min_training_sessions,
            validation_sessions=validation_sessions,
            embargo_sessions=embargo_sessions,
            step_sessions=step_sessions,
        )

    logistic_predictions = []
    isotonic_predictions = []
    baseline_predictions = []
    outcomes = []
    fold_rows = []
    prediction_rows_by_method = {"logistic": [], "isotonic": []}
    all_logistic_converged = True
    for fold in folds:
        training = tuple(by_id[item.observation_id] for item in fold.training)
        validation = tuple(by_id[item.observation_id] for item in fold.validation)
        training_scores = tuple(item.raw_probability for item in training)
        training_outcomes = tuple(item.outcome for item in training)
        validation_scores = tuple(item.raw_probability for item in validation)
        validation_outcomes = tuple(item.outcome for item in validation)
        logistic = LogisticCalibrator.fit(training_scores, training_outcomes)
        isotonic = IsotonicCalibrator.fit(training_scores, training_outcomes)
        logistic_fold = logistic.predict(validation_scores)
        isotonic_fold = isotonic.predict(validation_scores)
        base_rate = sum(training_outcomes) / len(training_outcomes)
        baseline_fold = (base_rate,) * len(validation)
        logistic_predictions.extend(logistic_fold)
        isotonic_predictions.extend(isotonic_fold)
        baseline_predictions.extend(baseline_fold)
        outcomes.extend(validation_outcomes)
        all_logistic_converged = all_logistic_converged and logistic.converged
        fold_rows.append(
            OOFFoldMetrics(
                fold_index=fold.fold_index,
                training_start=min(item.session_date for item in training),
                training_end=max(item.session_date for item in training),
                validation_start=min(item.session_date for item in validation),
                validation_end=max(item.session_date for item in validation),
                embargo_sessions=fold.embargo_sessions,
                training_count=len(training),
                validation_count=len(validation),
                logistic_brier=brier_score(logistic_fold, validation_outcomes),
                isotonic_brier=brier_score(isotonic_fold, validation_outcomes),
                base_rate_brier=brier_score(baseline_fold, validation_outcomes),
                logistic_converged=logistic.converged,
            )
        )
        for method, calibrated in (
            ("logistic", logistic_fold),
            ("isotonic", isotonic_fold),
        ):
            prediction_rows_by_method[method].extend(
                OOFPrediction(
                    observation_id=item.observation_id,
                    session_date=item.session_date,
                    fold_index=fold.fold_index,
                    raw_probability=item.raw_probability,
                    calibrated_probability=prediction,
                    training_base_rate=base_rate,
                    outcome=item.outcome,
                )
                for item, prediction in zip(validation, calibrated)
            )

    logistic_brier = brier_score(logistic_predictions, outcomes)
    isotonic_brier = brier_score(isotonic_predictions, outcomes)
    selected_method = (
        "logistic"
        if all_logistic_converged and logistic_brier <= isotonic_brier
        else "isotonic"
    )
    selected_predictions = (
        tuple(logistic_predictions)
        if selected_method == "logistic"
        else tuple(isotonic_predictions)
    )
    final_scores = tuple(item.raw_probability for item in observations)
    final_outcomes = tuple(item.outcome for item in observations)
    if selected_method == "logistic":
        final_calibrator = LogisticCalibrator.fit(final_scores, final_outcomes)
        if not final_calibrator.converged:
            selected_method = "isotonic"
            selected_predictions = tuple(isotonic_predictions)
            final_calibrator = IsotonicCalibrator.fit(final_scores, final_outcomes)
    else:
        final_calibrator = IsotonicCalibrator.fit(final_scores, final_outcomes)
    frozen = _freeze_calibrator(selected_method, final_calibrator)
    selected_rows = tuple(prediction_rows_by_method[selected_method])
    interval_bounds = wilson_interval(sum(outcomes), len(outcomes), 0.95)
    interval = IntervalProvenance(
        method="WILSON_SCORE",
        confidence=0.95,
        successes=sum(outcomes),
        sample_size=len(outcomes),
        lower=interval_bounds[0],
        upper=interval_bounds[1],
    )
    development_rows = tuple(
        sorted(
            observations,
            key=lambda item: (item.session_date, item.observation_id),
        )
    )
    development_fingerprint = _canonical_hash(
        {
            "bucket_id": bucket.bucket_id,
            "observations": [
                {
                    "observation_id": item.observation_id,
                    "session_date": item.session_date.isoformat(),
                    "raw_probability": item.raw_probability,
                    "outcome": item.outcome,
                }
                for item in development_rows
            ],
        }
    )

    artifact_kwargs = dict(
        bucket=bucket,
        model_version=model_version,
        selected_method=selected_method,
        frozen_calibrator=frozen,
        model_frozen_at=model_frozen_at,
        holdout_start=holdout_start,
        development_start=min(item.session_date for item in observations),
        development_end=max(item.session_date for item in observations),
        oof_brier_score=brier_score(selected_predictions, outcomes),
        base_rate_brier_score=brier_score(baseline_predictions, outcomes),
        expected_calibration_error=expected_calibration_error(
            selected_predictions, outcomes
        ),
        interval=interval,
        fold_metrics=tuple(fold_rows),
        oof_predictions=selected_rows,
        development_data_fingerprint=development_fingerprint,
    )
    provisional = OOFPOPModelArtifact.__new__(OOFPOPModelArtifact)
    # Build the canonical hash using the same serializer without bypassing any
    # public validation on the returned object.
    for name, value in artifact_kwargs.items():
        object.__setattr__(provisional, name, value)
    object.__setattr__(provisional, "artifact_id", "pending")
    object.__setattr__(provisional, "schema", POP_ARTIFACT_SCHEMA)
    artifact_id = _canonical_hash(provisional._payload(include_artifact_id=False))
    return OOFPOPModelArtifact(artifact_id=artifact_id, **artifact_kwargs)


__all__ = [
    "FrozenCalibrator",
    "IntervalProvenance",
    "OOFFoldMetrics",
    "OOFPOPModelArtifact",
    "OOFPOPObservation",
    "OOFPrediction",
    "POPBucketIdentity",
    "POP_ARTIFACT_SCHEMA",
    "ProbabilityTarget",
    "build_probability_bundle",
    "build_oof_pop_model",
]
