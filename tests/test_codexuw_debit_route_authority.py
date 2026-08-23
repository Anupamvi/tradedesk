import datetime as dt
import json

import pandas as pd

from codexuw import daily_shadow_books
from codexuw.daily_v4 import (
    _blocker_text,
    _effective_win_rate,
    _entry_limit_expectancy,
    _expectancy_safe_entry_price,
    _medium_debit_sleeve_eligible,
    _payoff_evidence_text,
    _payoff_evidence_ready,
    _post_pricing_expectancy,
    _ticket_entry_target,
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


def _production_debit_row(**overrides):
    row = {
        "strategy": "Bull Call Debit Spread",
        "strategy_kind": "Debit",
        "entry_type": "debit",
        "debit_wf_production_authorized": True,
        "debit_wf_live_guard_pass": True,
        "debit_wf_predicted_win_probability": 0.70,
        "debit_wf_expected_value": 25.0,
        "debit_wf_prior_sample_size": 722,
        "debit_wf_model_training_through": "2026-08-05",
        "v4_asof": "2026-08-14",
        "reward_risk": 1.75,
        "quote_width_pct": 0.20,
        "oi_carryover_status": "supportive",
        "natural_debit": 0.86,
        "spread_width": 2.0,
        "target_entry": 0.90,
    }
    row.update(overrides)
    return pd.Series(row)


def test_walk_forward_bull_call_has_one_contract_production_authority():
    row = _production_debit_row(
        debit_wf_policy_version="test-debit-production",
    )

    assert _medium_debit_sleeve_eligible(row)
    assert _effective_win_rate(row) == 0.70
    assert _expectancy_safe_entry_price(row) == 0.90
    expected_value, profit_factor, _, _ = _post_pricing_expectancy(row)
    assert expected_value > 0
    assert profit_factor > 1.25
    target_profit, max_loss, ticket_ev, ticket_pf = _entry_limit_expectancy(row)
    assert round(target_profit, 2) == 86.00
    assert round(max_loss, 2) == 86.00
    assert round(ticket_ev, 2) == round(expected_value, 2)
    assert round(ticket_pf, 2) == round(profit_factor, 2)
    evidence = _payoff_evidence_text(row)
    assert "policy=test-debit-production" in evidence
    assert "prior n=722" in evidence
    assert "predicted win=70%" in evidence
    assert "post-pricing EV=$25.00" in evidence
    assert "training through=2026-08-05" in evidence
    assert "live guard=PASS" in evidence
    assert "High confidence unavailable" in evidence
    assert "n=0" not in evidence


def test_medium_debit_work_limit_uses_the_same_current_debit_as_ticket_risk():
    row = _production_debit_row(natural_debit=0.57, target_entry=0.90)

    assert _ticket_entry_target(row, "Swing Target / Work Limit") == (
        "<= $0.57 debit; do not chase above $0.90"
    )
    target_profit, max_loss, _, _ = _entry_limit_expectancy(row)
    assert round(target_profit, 2) == 57.00
    assert round(max_loss, 2) == 57.00


def test_medium_debit_scout_keeps_current_debit_but_labels_review_only():
    row = _production_debit_row(natural_debit=0.57, target_entry=0.90)

    assert _ticket_entry_target(row, "Scout") == (
        "REVIEW ONLY - <= $0.57 debit; do not chase above $0.90; "
        "no execution authority"
    )


def test_authorized_work_limit_uses_final_selection_reason_not_legacy_scout_text():
    row = _production_debit_row(
        natural_debit=0.57,
        target_entry=0.90,
        trade_status="Watch",
        trade_status_reason="legacy Scout text incorrectly says direct Execute",
        v4_direct_disposition_reason=(
            "V4 Work Limit: another validated one-contract medium-debit setup "
            "ranks higher today."
        ),
    )

    assert _blocker_text(row) == (
        "V4 Work Limit: another validated one-contract medium-debit setup "
        "ranks higher today."
    )


def test_walk_forward_bull_call_keeps_live_quality_vetoes():
    assert not _medium_debit_sleeve_eligible(
        _production_debit_row(oi_carryover_status="contrary")
    )
    assert not _medium_debit_sleeve_eligible(
        _production_debit_row(quote_width_pct=0.251)
    )
    assert not _medium_debit_sleeve_eligible(
        _production_debit_row(reward_risk=1.249)
    )


def test_live_debit_summary_reports_production_authority(tmp_path, monkeypatch):
    history = tmp_path / "codexuw" / "history"
    history.mkdir(parents=True)
    validation = {
        "policy_version": "test-debit-production",
        "status": "PASS",
        "activation_date": "2020-01-01",
        "execution_authorized": True,
        "authority_scope": "one-contract-test",
        "live_quality_scope": {
            "minimum_reward_risk": 1.25,
            "maximum_quote_width_pct": 0.25,
            "allowed_oi_status": ["supportive", "matched_unconfirmed", "mixed"],
        },
    }
    (history / daily_shadow_books.DEBIT_PILOT_VALIDATION_FILE).write_text(
        json.dumps(validation)
    )
    evaluated = pd.DataFrame(
        [
            {
                "_v4_source_index": 0,
                "strategy": "Bull Call Debit Spread",
                "predicted_win_probability": 0.70,
                "predicted_ev_payoff_correct": 25.0,
                "prior_sample_size": 722,
                "model_training_through": "2026-08-05",
                "reward_risk": 1.75,
                "entry_quote_width_pct": 0.20,
                "oi_carryover_status": "supportive",
            }
        ]
    )
    monkeypatch.setattr(
        daily_shadow_books,
        "score_debit_shadow",
        lambda *args, **kwargs: (
            evaluated,
            pd.DataFrame(),
            {
                "status": "SHADOW_ACTIVE",
                "policy_version": "test-model",
                "training_through": "2026-08-05",
                "shadow_only": True,
                "execution_authorized": False,
            },
        ),
    )

    output, summary = daily_shadow_books.apply_live_debit_execution_model(
        pd.DataFrame([{"ticker": "TEST"}]),
        root=tmp_path,
        asof=dt.date(2026, 8, 14),
        allow_execution_authority=True,
    )

    assert bool(output.loc[0, "debit_wf_production_authorized"])
    assert summary["execution_authorized"] is True
    assert summary["production_authorized"] is True
    assert summary["shadow_only"] is False
    assert summary["production_candidate_rows"] == 1
    assert summary["policy_version"] == "test-debit-production"
