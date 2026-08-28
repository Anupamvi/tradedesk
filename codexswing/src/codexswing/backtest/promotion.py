"""Fail-closed evidence gates; this module never authorizes a broker order."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

from codexswing.backtest.metrics import ReplayMetrics, ReturnLike, compute_replay_metrics


@dataclass(frozen=True)
class PromotionCriteria:
    minimum_oos_observations: int = 100
    minimum_effective_independent_observations: int = 100
    minimum_oos_dates: int = 40
    minimum_folds: int = 3
    minimum_positive_fold_fraction: float = 0.5
    minimum_shadow_sessions: int = 60
    maximum_ticker_share: float = 0.25
    maximum_date_share: float = 0.10


@dataclass(frozen=True)
class PromotionDecision:
    eligible_for_user_authorized_pilot: bool
    status: str
    failed_gates: Sequence[str]
    out_of_sample_metrics: ReplayMetrics
    holdout_metrics: ReplayMetrics
    positive_fold_fraction: float
    effective_independent_oos_observations: int
    criteria: PromotionCriteria
    broker_order_authorized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible_for_user_authorized_pilot": self.eligible_for_user_authorized_pilot,
            "status": self.status,
            "failed_gates": list(self.failed_gates),
            "out_of_sample_metrics": self.out_of_sample_metrics.to_dict(),
            "holdout_metrics": self.holdout_metrics.to_dict(),
            "positive_fold_fraction": self.positive_fold_fraction,
            "effective_independent_oos_observations": self.effective_independent_oos_observations,
            "criteria": asdict(self.criteria),
            "broker_order_authorized": self.broker_order_authorized,
        }


def evaluate_promotion(
    out_of_sample: Sequence[ReturnLike],
    fold_metrics: Sequence[ReplayMetrics],
    holdout: Sequence[ReturnLike],
    *,
    deterministic_replay: bool,
    provenance_complete: bool,
    leakage_checks_passed: bool,
    holdout_was_frozen_single_use: bool,
    independence_evidence_complete: bool,
    effective_independent_oos_observations: int,
    shadow_sessions: int,
    live_replay_feature_parity: bool,
    criteria: PromotionCriteria = PromotionCriteria(),
    bootstrap_repetitions: int = 2000,
    bootstrap_seed: int = 1729,
) -> PromotionDecision:
    oos_metrics = compute_replay_metrics(
        out_of_sample,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    holdout_metrics = compute_replay_metrics(
        holdout,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    positive_fold_fraction = (
        sum(1 for metrics in fold_metrics if metrics.mean_net_return > 0) / len(fold_metrics)
        if fold_metrics
        else 0.0
    )
    failures = []
    if oos_metrics.observation_count < criteria.minimum_oos_observations:
        failures.append("INSUFFICIENT_OOS_OBSERVATIONS")
    if (
        effective_independent_oos_observations
        < criteria.minimum_effective_independent_observations
    ):
        failures.append("INSUFFICIENT_EFFECTIVE_INDEPENDENT_OBSERVATIONS")
    if oos_metrics.unique_decision_dates < criteria.minimum_oos_dates:
        failures.append("INSUFFICIENT_OOS_DATES")
    if len(fold_metrics) < criteria.minimum_folds:
        failures.append("INSUFFICIENT_WALK_FORWARD_FOLDS")
    if positive_fold_fraction <= criteria.minimum_positive_fold_fraction:
        failures.append("MOST_FOLDS_NOT_POSITIVE")
    if holdout_metrics.mean_net_return <= 0:
        failures.append("FINAL_HOLDOUT_DIAGNOSTIC_NOT_POSITIVE")
    if oos_metrics.bootstrap_p05_mean_return <= 0:
        failures.append("BOOTSTRAP_LOWER_EXPECTANCY_NOT_POSITIVE")
    if oos_metrics.maximum_ticker_observation_share > criteria.maximum_ticker_share:
        failures.append("TICKER_CONCENTRATION_TOO_HIGH")
    if oos_metrics.maximum_date_observation_share > criteria.maximum_date_share:
        failures.append("DATE_CONCENTRATION_TOO_HIGH")
    if shadow_sessions < criteria.minimum_shadow_sessions:
        failures.append("SHADOW_HISTORY_TOO_SHORT")
    if not deterministic_replay:
        failures.append("REPLAY_NOT_DETERMINISTIC")
    if not provenance_complete:
        failures.append("PROVENANCE_INCOMPLETE")
    if not leakage_checks_passed:
        failures.append("LEAKAGE_CHECKS_FAILED")
    if not holdout_was_frozen_single_use:
        failures.append("HOLDOUT_NOT_FROZEN_SINGLE_USE")
    if not independence_evidence_complete:
        failures.append("OUTCOME_INDEPENDENCE_UNPROVEN")
    if not live_replay_feature_parity:
        failures.append("LIVE_REPLAY_PARITY_UNPROVEN")
    eligible = not failures
    return PromotionDecision(
        eligible_for_user_authorized_pilot=eligible,
        status=(
            "PILOT_ELIGIBLE_RESEARCH_GATE_MET_MANUAL_AUTH_REQUIRED"
            if eligible
            else "RESEARCH_ONLY_NO_PROMOTION"
        ),
        failed_gates=tuple(failures),
        out_of_sample_metrics=oos_metrics,
        holdout_metrics=holdout_metrics,
        positive_fold_fraction=positive_fold_fraction,
        effective_independent_oos_observations=effective_independent_oos_observations,
        criteria=criteria,
    )
