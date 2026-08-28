from dataclasses import dataclass

import pytest

from codexswing.backtest.ablation import compare_ablation_variants
from codexswing.backtest.metrics import compute_replay_metrics
from codexswing.backtest.promotion import (
    PromotionCriteria,
    evaluate_promotion,
)


@dataclass(frozen=True)
class Outcome:
    sample_id: str
    ticker: str
    decision_date: str
    net_return: float


def _outcomes(count: int, value: float = 0.01):
    return [
        Outcome(
            sample_id="sample-{}".format(index),
            ticker="T{:02d}".format(index % 10),
            decision_date="2026-{:02d}-{:02d}".format(1 + (index // 280), 1 + (index % 28)),
            net_return=value,
        )
        for index in range(count)
    ]


def test_metrics_are_deterministic_and_clustered_by_decision_date() -> None:
    outcomes = _outcomes(40)
    first = compute_replay_metrics(outcomes, bootstrap_repetitions=200, bootstrap_seed=7)
    second = compute_replay_metrics(outcomes, bootstrap_repetitions=200, bootstrap_seed=7)
    assert first == second
    assert first.mean_net_return == pytest.approx(0.01)
    assert first.bootstrap_p05_mean_return == pytest.approx(0.01)
    assert first.profit_factor_is_infinite is True


def test_paired_ablation_requires_the_same_samples() -> None:
    baseline = _outcomes(10, value=0.0)
    improved = [
        Outcome(item.sample_id, item.ticker, item.decision_date, 0.01) for item in baseline
    ]
    steps = compare_ablation_variants(
        {"PRICE": baseline, "PRICE_ORATS": improved},
        ["PRICE", "PRICE_ORATS"],
        bootstrap_repetitions=200,
    )
    assert steps[1].incremental_status == "INCREMENTAL_EDGE_CANDIDATE_NOT_PROMOTED"
    with pytest.raises(ValueError, match="same sample"):
        compare_ablation_variants(
            {"PRICE": baseline, "PRICE_ORATS": improved[:-1]},
            ["PRICE", "PRICE_ORATS"],
            bootstrap_repetitions=200,
        )


def test_promotion_fails_closed_when_operational_evidence_is_absent() -> None:
    oos = _outcomes(120)
    holdout = [
        Outcome("holdout-{}".format(index), item.ticker, item.decision_date, item.net_return)
        for index, item in enumerate(_outcomes(40))
    ]
    fold_metrics = [compute_replay_metrics(_outcomes(20)) for _ in range(3)]
    decision = evaluate_promotion(
        oos,
        fold_metrics,
        holdout,
        deterministic_replay=True,
        provenance_complete=False,
        leakage_checks_passed=True,
        holdout_was_frozen_single_use=False,
        independence_evidence_complete=False,
        effective_independent_oos_observations=24,
        shadow_sessions=0,
        live_replay_feature_parity=False,
        criteria=PromotionCriteria(maximum_ticker_share=0.20),
    )
    assert decision.eligible_for_user_authorized_pilot is False
    assert decision.broker_order_authorized is False
    assert "PROVENANCE_INCOMPLETE" in decision.failed_gates
    assert "SHADOW_HISTORY_TOO_SHORT" in decision.failed_gates
