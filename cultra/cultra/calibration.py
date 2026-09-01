"""Deterministic POP calibration and calibration-quality metrics."""

from bisect import bisect_right
from dataclasses import dataclass
import math
from statistics import NormalDist
from typing import Sequence, Tuple, Union


_CLIP = 1e-9


def _validate_scores_outcomes(
    scores: Sequence[float], outcomes: Sequence[int]
) -> Tuple[Tuple[float, ...], Tuple[int, ...]]:
    if len(scores) != len(outcomes) or not scores:
        raise ValueError("scores and outcomes must have equal non-zero length")
    checked_scores = []
    checked_outcomes = []
    for score, outcome in zip(scores, outcomes):
        score = float(score)
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError("scores must be finite probabilities")
        if isinstance(outcome, bool):
            outcome = int(outcome)
        if outcome not in (0, 1):
            raise ValueError("outcomes must be binary")
        checked_scores.append(score)
        checked_outcomes.append(int(outcome))
    return tuple(checked_scores), tuple(checked_outcomes)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _logit(probability: float) -> float:
    clipped = min(1.0 - _CLIP, max(_CLIP, probability))
    return math.log(clipped / (1.0 - clipped))


def brier_score(predictions: Sequence[float], outcomes: Sequence[int]) -> float:
    checked, binary = _validate_scores_outcomes(predictions, outcomes)
    return math.fsum((prediction - outcome) ** 2 for prediction, outcome in zip(checked, binary)) / len(checked)


def unconditional_brier_score(outcomes: Sequence[int], base_rate: float) -> float:
    return brier_score((float(base_rate),) * len(outcomes), outcomes)


def expected_calibration_error(
    predictions: Sequence[float], outcomes: Sequence[int], bins: int = 10
) -> float:
    """Equal-width expected calibration error with deterministic bin edges."""

    checked, binary = _validate_scores_outcomes(predictions, outcomes)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    grouped_predictions = [[] for _ in range(bins)]
    grouped_outcomes = [[] for _ in range(bins)]
    for prediction, outcome in zip(checked, binary):
        index = min(bins - 1, int(prediction * bins))
        grouped_predictions[index].append(prediction)
        grouped_outcomes[index].append(outcome)
    total = len(checked)
    error = 0.0
    for predicted, observed in zip(grouped_predictions, grouped_outcomes):
        if not predicted:
            continue
        mean_prediction = math.fsum(predicted) / len(predicted)
        mean_outcome = math.fsum(observed) / len(observed)
        error += (len(predicted) / total) * abs(mean_prediction - mean_outcome)
    return error


def wilson_interval(
    successes: int, sample_size: int, confidence: float = 0.95
) -> Tuple[float, float]:
    if isinstance(successes, bool) or not isinstance(successes, int):
        raise TypeError("successes must be an integer")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if sample_size <= 0 or successes < 0 or successes > sample_size:
        raise ValueError("require 0 <= successes <= positive sample_size")
    confidence = float(confidence)
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    z_value = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / sample_size
    denominator = 1.0 + z_value * z_value / sample_size
    center = (proportion + z_value * z_value / (2.0 * sample_size)) / denominator
    radius = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / sample_size
            + z_value * z_value / (4.0 * sample_size * sample_size)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def empirical_probability_interval(
    outcomes: Sequence[int], confidence: float = 0.95
) -> Tuple[float, float]:
    checked = tuple(outcomes)
    _scores, binary = _validate_scores_outcomes((0.5,) * len(checked), checked)
    return wilson_interval(sum(binary), len(binary), confidence)


@dataclass(frozen=True)
class LogisticCalibrator:
    intercept: float
    slope: float
    sample_size: int
    converged: bool

    @classmethod
    def fit(
        cls,
        raw_probabilities: Sequence[float],
        outcomes: Sequence[int],
        max_iterations: int = 200,
        tolerance: float = 1e-10,
        l2: float = 1e-6,
    ) -> "LogisticCalibrator":
        scores, binary = _validate_scores_outcomes(raw_probabilities, outcomes)
        if max_iterations <= 0 or tolerance <= 0.0 or l2 <= 0.0:
            raise ValueError("fit controls must be positive")
        features = tuple(_logit(score) for score in scores)
        smoothed_rate = (sum(binary) + 0.5) / (len(binary) + 1.0)
        intercept = _logit(smoothed_rate)
        slope = 0.0

        def objective(candidate_intercept: float, candidate_slope: float) -> float:
            loss = 0.5 * l2 * candidate_slope * candidate_slope
            for feature, outcome in zip(features, binary):
                linear = candidate_intercept + candidate_slope * feature
                # Stable logistic negative log-likelihood.
                loss += max(linear, 0.0) - linear * outcome + math.log1p(math.exp(-abs(linear)))
            return loss

        converged = False
        current_objective = objective(intercept, slope)
        for _ in range(max_iterations):
            gradient_intercept = 0.0
            gradient_slope = l2 * slope
            hessian_ii = l2
            hessian_is = 0.0
            hessian_ss = l2
            for feature, outcome in zip(features, binary):
                prediction = _sigmoid(intercept + slope * feature)
                residual = prediction - outcome
                weight = max(1e-12, prediction * (1.0 - prediction))
                gradient_intercept += residual
                gradient_slope += residual * feature
                hessian_ii += weight
                hessian_is += weight * feature
                hessian_ss += weight * feature * feature
            determinant = hessian_ii * hessian_ss - hessian_is * hessian_is
            if determinant <= 1e-20:
                break
            step_intercept = (
                hessian_ss * gradient_intercept - hessian_is * gradient_slope
            ) / determinant
            step_slope = (
                -hessian_is * gradient_intercept + hessian_ii * gradient_slope
            ) / determinant
            scale = 1.0
            accepted = False
            while scale >= 2.0 ** -20:
                candidate_intercept = intercept - scale * step_intercept
                candidate_slope = slope - scale * step_slope
                candidate_objective = objective(candidate_intercept, candidate_slope)
                if candidate_objective <= current_objective:
                    intercept = candidate_intercept
                    slope = candidate_slope
                    current_objective = candidate_objective
                    accepted = True
                    break
                scale /= 2.0
            if not accepted:
                break
            if max(abs(scale * step_intercept), abs(scale * step_slope)) < tolerance:
                converged = True
                break
        if not math.isfinite(intercept) or not math.isfinite(slope):
            raise ArithmeticError("logistic calibration did not produce finite coefficients")
        return cls(intercept, slope, len(scores), converged)

    def predict_one(self, raw_probability: float) -> float:
        raw_probability = float(raw_probability)
        if not math.isfinite(raw_probability) or not 0.0 <= raw_probability <= 1.0:
            raise ValueError("raw_probability must be a finite probability")
        return _sigmoid(self.intercept + self.slope * _logit(raw_probability))

    def predict(self, raw_probabilities: Sequence[float]) -> Tuple[float, ...]:
        return tuple(self.predict_one(value) for value in raw_probabilities)


@dataclass(frozen=True)
class IsotonicCalibrator:
    x_thresholds: Tuple[float, ...]
    y_values: Tuple[float, ...]
    sample_size: int

    @classmethod
    def fit(
        cls, raw_probabilities: Sequence[float], outcomes: Sequence[int]
    ) -> "IsotonicCalibrator":
        scores, binary = _validate_scores_outcomes(raw_probabilities, outcomes)
        grouped = []
        for score, outcome in sorted(zip(scores, binary), key=lambda pair: pair[0]):
            if grouped and score == grouped[-1][0]:
                grouped[-1][1] += 1
                grouped[-1][2] += outcome
            else:
                grouped.append([score, 1, outcome])

        # Each block stores inclusive indexes into `grouped`, total weight, sum.
        blocks = []
        for index, (_score, weight, outcome_sum) in enumerate(grouped):
            blocks.append([index, index, weight, float(outcome_sum)])
            while len(blocks) >= 2:
                previous = blocks[-2]
                current = blocks[-1]
                previous_mean = previous[3] / previous[2]
                current_mean = current[3] / current[2]
                if previous_mean <= current_mean:
                    break
                blocks[-2:] = [[previous[0], current[1], previous[2] + current[2], previous[3] + current[3]]]

        fitted = [0.0] * len(grouped)
        for start, end, weight, outcome_sum in blocks:
            value = outcome_sum / weight
            for index in range(start, end + 1):
                fitted[index] = value
        return cls(
            x_thresholds=tuple(row[0] for row in grouped),
            y_values=tuple(fitted),
            sample_size=len(scores),
        )

    def __post_init__(self) -> None:
        if not self.x_thresholds or len(self.x_thresholds) != len(self.y_values):
            raise ValueError("isotonic thresholds and values must align")
        if any(left >= right for left, right in zip(self.x_thresholds, self.x_thresholds[1:])):
            raise ValueError("x_thresholds must be strictly increasing")
        if any(left > right for left, right in zip(self.y_values, self.y_values[1:])):
            raise ValueError("y_values must be nondecreasing")

    def predict_one(self, raw_probability: float) -> float:
        value = float(raw_probability)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("raw_probability must be a finite probability")
        if value <= self.x_thresholds[0]:
            return self.y_values[0]
        if value >= self.x_thresholds[-1]:
            return self.y_values[-1]
        right = bisect_right(self.x_thresholds, value)
        left = right - 1
        x_left = self.x_thresholds[left]
        x_right = self.x_thresholds[right]
        if self.y_values[left] == self.y_values[right]:
            # Preserve a PAVA plateau bit-for-bit.  Interpolating identical
            # binary floats can otherwise create one-ulp downward artifacts.
            return self.y_values[left]
        weight = (value - x_left) / (x_right - x_left)
        interpolated = (
            self.y_values[left] * (1.0 - weight) + self.y_values[right] * weight
        )
        return min(self.y_values[right], max(self.y_values[left], interpolated))

    def predict(self, raw_probabilities: Sequence[float]) -> Tuple[float, ...]:
        return tuple(self.predict_one(value) for value in raw_probabilities)


Calibrator = Union[LogisticCalibrator, IsotonicCalibrator]


@dataclass(frozen=True)
class CalibrationSelection:
    name: str
    calibrator: Calibrator
    validation_brier: float
    logistic_validation_brier: float
    isotonic_validation_brier: float
    training_base_rate: float
    model_version: str


def choose_calibrator(
    training_scores: Sequence[float],
    training_outcomes: Sequence[int],
    validation_scores: Sequence[float],
    validation_outcomes: Sequence[int],
    model_version: str,
) -> CalibrationSelection:
    """Fit on training only, select on validation Brier score, never holdout."""

    if not model_version or not model_version.strip():
        raise ValueError("model_version is required")
    train_scores, train_binary = _validate_scores_outcomes(training_scores, training_outcomes)
    validation_scores_checked, validation_binary = _validate_scores_outcomes(
        validation_scores, validation_outcomes
    )
    logistic = LogisticCalibrator.fit(train_scores, train_binary)
    isotonic = IsotonicCalibrator.fit(train_scores, train_binary)
    logistic_score = brier_score(logistic.predict(validation_scores_checked), validation_binary)
    isotonic_score = brier_score(isotonic.predict(validation_scores_checked), validation_binary)
    # Stable tie-break toward the parametric model.
    if logistic_score <= isotonic_score:
        name = "logistic"
        selected = logistic  # type: Calibrator
        selected_score = logistic_score
    else:
        name = "isotonic"
        selected = isotonic
        selected_score = isotonic_score
    return CalibrationSelection(
        name=name,
        calibrator=selected,
        validation_brier=selected_score,
        logistic_validation_brier=logistic_score,
        isotonic_validation_brier=isotonic_score,
        training_base_rate=sum(train_binary) / len(train_binary),
        model_version=model_version,
    )
