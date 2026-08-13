import pandas as pd

from codexuw.daily_v4 import (
    _medium_debit_sleeve_eligible,
    _payoff_evidence_ready,
)


def _validated_route_row(**overrides):
    row = {
        "strategy": "Bull Call Debit Spread",
        "strategy_kind": "Debit",
        "payoff_route_key": "base::Debit|Bull Call|uptrend",
        "payoff_calibration_status": "PROBATIONARY",
        "payoff_sample_size": 26,
        "payoff_stress_10_average_pnl": 33.38,
        "payoff_stress_10_profit_factor": 1.647,
        "payoff_walk_forward_oos_sample": 9,
        "payoff_walk_forward_oos_average_pnl": 45.70,
        "payoff_walk_forward_oos_profit_factor": 2.612,
        "flow_quality": "directional",
        "oi_carryover_status": "supportive",
        "debit_policy_tier": "medium",
        "edge_match_level": "debit_policy_sleeve",
        "edge_sample_size": 26,
        "edge_avg_pnl": 33.38,
        "edge_profit_factor": 1.647,
    }
    row.update(overrides)
    return pd.Series(row)


def test_stale_uptrend_bull_call_route_no_longer_has_payoff_authority():
    row = _validated_route_row()

    assert not _medium_debit_sleeve_eligible(row)
    assert not _payoff_evidence_ready(row)


def test_validated_debit_route_keeps_exact_negative_edge_veto():
    row = _validated_route_row(
        edge_match_level="ticker_direction",
        edge_sample_size=20,
        edge_avg_pnl=-5.0,
        edge_profit_factor=0.9,
    )

    assert not _medium_debit_sleeve_eligible(row)


def test_validated_debit_route_does_not_authorize_range_market():
    row = _validated_route_row(
        payoff_route_key="base::Debit|Bull Call|range",
    )

    assert not _medium_debit_sleeve_eligible(row)


def test_validated_debit_route_requires_directional_flow_and_oi_support():
    assert not _medium_debit_sleeve_eligible(
        _validated_route_row(flow_quality="unclear")
    )
    assert not _medium_debit_sleeve_eligible(
        _validated_route_row(oi_carryover_status="contrary")
    )
