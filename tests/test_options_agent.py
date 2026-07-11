import datetime as dt
import hashlib
import importlib.abc
import inspect
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

from uwos.options_agent import audit, core, market_open_runner
from uwos.options_agent.core import RecommendationStatus, apply_portfolio_risk_annotations, output_paths, run_pipeline


STRICT_MODE_TESTS = {
    "test_strict_negative_strategy_expectancy_blocks_review_rows_from_target_surface",
    "test_green_ticket_requires_strategy_expectancy_annotation",
    "test_material_position_profit_blocks_green_and_marks_profit_floor_target",
    "test_send_now_requires_strong_credit_and_trade_quality",
    "test_send_now_green_requires_positive_structure_aligned_actual_forward_support",
    "test_closed_market_clean_live_row_can_still_be_green_when_other_gates_pass",
    "test_strategy_expectancy_blocks_opposite_or_unrelated_ticker_history",
    "test_recompute_live_capture_enforces_profitability_calibration",
    "test_price_candidates_includes_short_put_when_short_put_family_evidence_passes",
    "test_broad_vertical_route_evidence_does_not_create_green_without_ticker_strategy_proof",
    "test_short_put_cash_risk_blocks_green_but_keeps_yellow_target_surface",
    "test_negative_strategy_family_evidence_blocks_trade_ticket_surface",
    "test_profitability_calibration_blocks_ready_looking_green_row",
    "test_profitability_calibration_blocks_yellow_target_row_until_bucket_proven",
    "test_negative_route_family_evidence_keeps_row_off_yellow_target_surface",
    "test_short_put_route_stays_off_yellow_target_surface_with_negative_calibration",
    "test_short_put_route_stays_off_yellow_target_surface_with_uncalibrated_profit",
    "test_replay_blocked_calibration_stays_off_yellow_target_surface",
    "test_positive_actual_long_call_stays_on_yellow_target_surface_while_replay_bucket_pending",
    "test_weak_actual_long_call_stays_review_only_until_positive_support_exists",
    "test_uncalibrated_low_profit_row_stays_on_target_surface_with_missing_ticker_reviews",
    "test_report_target_order_table_uses_trade_ticket_surface_filters",
    "test_strategy_supported_debit_spread_stays_actionable_despite_cautions",
    "test_weak_flow_debit_spread_without_outcome_support_is_not_send_now",
}


@pytest.fixture(autouse=True)
def _strict_mode_for_goal_era_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    if request.node.name not in STRICT_MODE_TESTS:
        return
    monkeypatch.setattr(core, "OPTIONS_AGENT_V0_RECONSTRUCTION", False)
    monkeypatch.setattr(core, "ENABLE_CASH_SECURED_PUT_ROUTE", True)
    monkeypatch.setattr(core, "V0_LATE_EVIDENCE_GATES_DIAGNOSTIC_ONLY", False)
    monkeypatch.setattr(core, "V0_REQUIRE_PER_TICKER_AGENT_REVIEW", True)


def test_goal_runtime_defaults_are_locked() -> None:
    source = Path(core.__file__).read_text()

    assert core.PIPELINE_VERSION == "options-agent-v1.3-evidence-integrity-20260710-151249"
    assert core.PREVIOUS_PIPELINE_VERSIONS == (
        "options-agent-v1.2-blocker-carryforward-20260710-142154",
        "options-agent-v1.2-exact-reprice-20260710-093806",
        "options-agent-v1.1-contract-risk-20260709-193127",
        "options-agent-v1.1-contract-risk-20260709-184846",
        "options-agent-v1.0-exec-confidence-20260612-143405",
        "options-agent-v0",
    )
    assert core.PIPELINE_RELEASED_AT == "2026-07-10T15:12:49-07:00"
    assert core.MAX_LIVE_DISPATCH_SNAPSHOT_AGE_SECONDS == 0
    assert core.OPTIONS_AGENT_V0_RECONSTRUCTION is False
    assert core.ENABLE_CASH_SECURED_PUT_ROUTE is True
    assert core.V0_LATE_EVIDENCE_GATES_DIAGNOSTIC_ONLY is False
    assert core.V0_REQUIRE_PER_TICKER_AGENT_REVIEW is True
    assert core._v0_late_evidence_gates_diagnostic_only() is False
    assert core._v0_require_per_ticker_agent_review() is True
    assert core.MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING == 7.0
    assert core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING == 7.0
    assert core.assert_strict_goal_runtime_defaults() is None
    assert source.count("PIPELINE_VERSION =") == 1
    assert source.count("OPTIONS_AGENT_V0_RECONSTRUCTION =") == 1
    assert source.count("ENABLE_CASH_SECURED_PUT_ROUTE =") == 1
    assert source.count("V0_LATE_EVIDENCE_GATES_DIAGNOSTIC_ONLY =") == 1
    assert source.count("V0_REQUIRE_PER_TICKER_AGENT_REVIEW =") == 1
    assert source.count("MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING =") == 1
    assert source.count("MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING =") == 1


def test_goal_runtime_guard_blocks_v0_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(core, "PIPELINE_VERSION", "options-agent-v0")
    monkeypatch.setattr(core, "OPTIONS_AGENT_V0_RECONSTRUCTION", True)

    with pytest.raises(RuntimeError, match="strict goal runtime defaults drifted"):
        core.assert_strict_goal_runtime_defaults()


def test_green_ready_rows_are_not_counted_as_target_order_candidates() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "READY",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 READY 2026-07-17 100 Call / SELL 1 READY 2026-07-17 105 Call @ 1.00 DEBIT",
                "trade_plan": "BUY 1 READY 2026-07-17 100 Call / SELL 1 READY 2026-07-17 105 Call @ 1.00 DEBIT",
                "entry_limit": 1.0,
                "suggested_contracts": 2,
                "max_profit": 400.0,
                "max_loss": 100.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 5,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=5,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    green, target = core.split_trade_ticket_surfaces(tickets)

    assert decision["ready_to_enter"].tolist() == [True]
    assert decision["strategy_route"].tolist() == ["bull_call_debit"]
    assert decision["strategy_family"].tolist() == ["vertical_spread"]
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert core._target_order_candidate_count(decision) == 0
    assert tickets["strategy_route"].tolist() == ["bull_call_debit"]
    assert tickets["strategy_family"].tolist() == ["vertical_spread"]
    assert green["ready_to_enter"].tolist() == [True]
    assert target.empty


def test_deep_route_support_can_clear_confidence_without_ticker_history_at_one_lot() -> None:
    row = {
        "ticker": "SPCX",
        "strategy_route": "bull_call_debit",
        "trade_plan": "BUY 1 SPCX 2026-08-07 157.5 Call / SELL 1 SPCX 2026-08-07 160 Call @ 0.94 DEBIT",
        "entry_limit": 0.94,
        "max_profit": 156.0,
        "max_loss": 94.0,
        "trade_quality_status": "reviewable",
        "live_validation_status": "PASS",
        "agent_support_count": 8,
        "agent_caution_count": 5,
        "agent_objective_blocker_count": 0,
        "actual_forward_expectancy_status": "BLOCK",
        "actual_forward_expectancy_sample_size": 0,
        "actual_forward_strategy_expectancy_status": "PASS",
        "actual_forward_strategy_expectancy_sample_size": core.MIN_HIERARCHICAL_ROUTE_SAMPLE_SIZE,
        "actual_forward_strategy_expectancy_win_rate": 0.44,
        "actual_forward_strategy_expectancy_avg_pnl": 20.0,
        "actual_forward_strategy_expectancy_profit_factor": 1.25,
        "profitability_calibration_status": "PASS",
        "live_probability_proxy": 0.436,
        "live_quote_width_pct": 0.034,
        "live_theta_burn_pct": 0.003,
    }
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/agentic_reviews.json"),
        market_session_open=True,
    )

    score, rating, quality_score, quality_rating = core._execution_confidence(row, context, [])

    assert score >= core.MIN_EXECUTION_CONFIDENCE_SCORE
    assert rating in {"MEDIUM", "HIGH"}
    assert quality_score >= core.MIN_EXECUTION_CONFIDENCE_SCORE
    assert quality_rating in {"MEDIUM", "HIGH"}
    assert "route PF 1.25 / n=40 / win 44% / avg P/L $20.00" in core._ticket_contract_risk_summary(row)


def test_goal_confidence_gate_demotes_ready_rows_before_action_surfaces() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "READY",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 READY 2026-07-17 100 Call / SELL 1 READY 2026-07-17 105 Call @ 1.00 DEBIT",
                "trade_plan": "BUY 1 READY 2026-07-17 100 Call / SELL 1 READY 2026-07-17 105 Call @ 1.00 DEBIT",
                "entry_limit": 1.0,
                "suggested_contracts": 2,
                "max_profit": 400.0,
                "max_loss": 100.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 5,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=5,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    confidence_audit = pd.DataFrame(
        [
            {"metric": "profitability_confidence_rating", "rating": 3.0, "status": "BLOCK", "blockers": "not proven"},
            {"metric": "order_entry_confidence_rating", "rating": 5.0, "status": "BLOCK", "blockers": "not proven"},
            {"metric": "goal_confidence_gate", "rating": 3.0, "status": "BLOCK", "blockers": "not proven"},
        ],
        columns=core.CONFIDENCE_AUDIT_COLUMNS,
    )

    gated = core.apply_goal_confidence_gate_to_decision_board(decision, confidence_audit)
    tickets = core.build_trade_tickets(gated)
    green, target = core.split_trade_ticket_surfaces(tickets)

    assert decision["ready_to_enter"].tolist() == [True]
    assert gated["ready_to_enter"].tolist() == [False]
    assert gated["execution_status"].tolist() == ["needs_confidence"]
    assert gated["status_icon"].tolist() == ["🟡"]
    assert gated["target_order_status"].tolist() == ["target_order_candidate"]
    assert gated["status_label"].tolist() == ["YELLOW target"]
    assert core.GOAL_CONFIDENCE_GATE_BLOCKER in gated["execution_blockers"].iloc[0]
    assert decision["execution_confidence_score"].iloc[0] >= core.MIN_EXECUTION_CONFIDENCE_SCORE
    assert gated["execution_confidence_score"].tolist() == [0.0]
    assert gated["execution_confidence_rating"].tolist() == ["NOT_EXECUTION_READY"]
    assert gated["order_mechanics_confidence_score"].iloc[0] >= core.MIN_EXECUTION_CONFIDENCE_SCORE
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["execution_confidence_score"].tolist() == [0.0]
    assert tickets["execution_confidence_rating"].tolist() == ["NOT_EXECUTION_READY"]
    assert tickets["order_mechanics_confidence_score"].iloc[0] >= core.MIN_EXECUTION_CONFIDENCE_SCORE
    assert green.empty
    assert target["ticker"].tolist() == ["READY"]
    rendered = "\n".join(core._render_ticket_rows(target))
    assert "Trade Edge / Entry / Order" in rendered
    assert "Contract Risk" in rendered
    assert "NOT_EXECUTION_READY" in rendered
    assert "order" in rendered


def test_goal_confidence_blocked_rows_do_not_leak_onto_green_ticket_surface() -> None:
    decision = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "CONF",
                "status_icon": "🟢",
                "status_label": "GREEN ready",
                "trade_plan": "BUY 1 CONF 2026-07-17 100 Call / SELL 1 CONF 2026-07-17 105 Call @ 1.00 DEBIT",
                "entry_limit": 1.0,
                "suggested_contracts": 1,
                "max_profit": 400.0,
                "max_loss": 100.0,
                "ready_to_enter": True,
                "execution_status": "ready",
                "execution_gate_status": "pass",
                "target_order_status": "target_order_candidate",
                "execution_blockers": "",
                "live_validation_status": "PASS",
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "HIGH",
            },
            {
                "recommendation_rank": 2,
                "ticker": "PRICE",
                "status_icon": "🟡",
                "status_label": "YELLOW target",
                "trade_plan": "SELL 1 PRICE 2026-07-17 95 Put / BUY 1 PRICE 2026-07-17 90 Put @ 1.20 CREDIT",
                "entry_limit": 1.2,
                "suggested_contracts": 1,
                "max_profit": 120.0,
                "max_loss": 380.0,
                "ready_to_enter": False,
                "execution_status": "waiting_for_price",
                "execution_gate_status": "blocked",
                "target_order_status": "target_order_candidate",
                "execution_blockers": "send_now_credit_width_below_30pct",
                "live_validation_status": "PASS",
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
            },
        ]
    )
    confidence_audit = pd.DataFrame(
        [
            {"metric": "profitability_confidence_rating", "rating": 3.0, "status": "BLOCK", "blockers": "not proven"},
            {"metric": "order_entry_confidence_rating", "rating": 5.0, "status": "BLOCK", "blockers": "not proven"},
            {"metric": "goal_confidence_gate", "rating": 3.0, "status": "BLOCK", "blockers": "not proven"},
        ],
        columns=core.CONFIDENCE_AUDIT_COLUMNS,
    )

    gated = core.apply_goal_confidence_gate_to_decision_board(decision, confidence_audit)
    tickets = core.build_trade_tickets(gated)
    green, target = core.split_trade_ticket_surfaces(tickets)

    assert gated.loc[gated["ticker"].eq("CONF"), "target_order_status"].tolist() == [
        "target_order_candidate"
    ]
    assert gated.loc[gated["ticker"].eq("PRICE"), "target_order_status"].tolist() == [
        "target_order_candidate"
    ]
    assert tickets["ready_to_enter"].tolist() == [False, False]
    assert green.empty
    assert target["ticker"].tolist() == ["CONF", "PRICE"]


def test_goal_confidence_gate_replaces_stale_rating_annotation() -> None:
    decision = pd.DataFrame(
        [
            {
                "ticker": "CONF",
                "ready_to_enter": True,
                "execution_status": "ready",
                "execution_gate_status": "pass",
                "target_order_status": "target_order_candidate",
                "execution_blockers": "",
                "execution_confidence_score": 90.0,
                "execution_confidence_rating": "HIGH",
                "status_reason": "validated setup",
            }
        ]
    )

    first = core.apply_goal_confidence_gate_to_decision_board(
        decision,
        pd.DataFrame(
            [
                {"metric": "profitability_confidence_rating", "rating": 6.0, "status": "BLOCK"},
                {"metric": "order_entry_confidence_rating", "rating": 10.0, "status": "PASS"},
                {"metric": "order_mechanics_confidence_rating", "rating": 7.0, "status": "PASS"},
                {"metric": "goal_confidence_gate", "rating": 6.0, "status": "BLOCK"},
            ],
            columns=core.CONFIDENCE_AUDIT_COLUMNS,
        ),
    )
    second = core.apply_goal_confidence_gate_to_decision_board(
        first,
        pd.DataFrame(
            [
                {"metric": "profitability_confidence_rating", "rating": 6.0, "status": "BLOCK"},
                {"metric": "order_entry_confidence_rating", "rating": 0.0, "status": "BLOCK"},
                {"metric": "order_mechanics_confidence_rating", "rating": 7.0, "status": "PASS"},
                {"metric": "goal_confidence_gate", "rating": 0.0, "status": "BLOCK"},
            ],
            columns=core.CONFIDENCE_AUDIT_COLUMNS,
        ),
    )

    reason = second["status_reason"].iloc[0]
    assert reason.count("Overall confidence gate is blocked:") == 1
    assert "order_entry_surface=0.0/10" in reason
    assert "order_entry_surface=10.0/10" not in reason


def test_route_only_profit_hypothesis_row_stays_off_yellow_target_surface() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "EDGE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 EDGE 2026-07-17 235 Put / BUY 1 EDGE 2026-07-17 230 Put @ 1.08 CREDIT",
                "trade_plan": "SELL 1 EDGE 2026-07-17 235 Put / BUY 1 EDGE 2026-07-17 230 Put @ 1.08 CREDIT",
                "entry_limit": 1.08,
                "target_entry": 0.90,
                "suggested_contracts": 8,
                "max_profit": 108.0,
                "max_loss": 392.0,
                "credit_width_ratio": 0.216,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 5,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_actual_sample_size": 1,
                "profitability_calibration_actual_avg_pnl": 227.0,
                "profitability_calibration_actual_profit_factor": float("inf"),
                "profitability_calibration_replay_status": "WARN",
                "actual_forward_expectancy_status": "PASS",
                "actual_forward_expectancy_sample_size": 3,
                "actual_forward_expectancy_avg_pnl": 227.0,
                "actual_forward_expectancy_profit_factor": float("inf"),
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 3,
                "actual_forward_strategy_expectancy_avg_pnl": 227.0,
                "actual_forward_strategy_expectancy_profit_factor": float("inf"),
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=5,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    green, target = core.split_trade_ticket_surfaces(tickets)

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["target_entry"].tolist() == [0.90]
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["ticker"].tolist() == ["EDGE"]
    assert tickets["target_entry"].tolist() == [0.90]
    assert tickets["order_readiness"].tolist() == ["target_order_after_profitability_calibration"]
    assert green.empty
    assert target.empty


def test_final_recommendations_output_does_not_leak_enter_for_non_ready_rows() -> None:
    final = pd.DataFrame(
        [
            {"recommendation_rank": 1, "ticker": "READY", "recommendation_status": RecommendationStatus.ENTER.value},
            {"recommendation_rank": 2, "ticker": "EVIDENCE", "recommendation_status": RecommendationStatus.ENTER.value},
            {"recommendation_rank": 3, "ticker": "TARGET", "recommendation_status": RecommendationStatus.ENTER.value},
            {"recommendation_rank": 4, "ticker": "CSP", "recommendation_status": RecommendationStatus.ENTER.value},
        ]
    )
    decision = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "READY",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "execution_status": "ready",
                "execution_gate_status": "pass",
                "execution_blockers": "",
            },
            {
                "recommendation_rank": 2,
                "ticker": "EVIDENCE",
                "ready_to_enter": False,
                "target_order_status": "review_only_expectancy_evidence",
                "execution_status": "needs_confidence",
                "execution_gate_status": "blocked",
                "execution_blockers": core.PROFITABILITY_CALIBRATION_BLOCKER,
            },
            {
                "recommendation_rank": 3,
                "ticker": "TARGET",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "execution_status": "waiting_for_price",
                "execution_gate_status": "blocked",
                "execution_blockers": "send_now_credit_width_below_30pct",
            },
            {
                "recommendation_rank": 4,
                "ticker": "CSP",
                "ready_to_enter": False,
                "target_order_status": "not_actionable_risk_reward",
                "execution_status": "needs_confidence",
                "execution_gate_status": "blocked",
                "execution_blockers": "not_actionable_risk_reward",
            },
        ]
    )

    output = core.annotate_final_recommendations_with_execution_surface(final, decision)

    assert output["pre_execution_recommendation_status"].tolist() == [
        RecommendationStatus.ENTER.value,
        RecommendationStatus.ENTER.value,
        RecommendationStatus.ENTER.value,
        RecommendationStatus.ENTER.value,
    ]
    assert output["recommendation_status"].tolist() == [
        RecommendationStatus.ENTER.value,
        RecommendationStatus.REVIEW.value,
        RecommendationStatus.WAIT_FOR_PRICE.value,
        RecommendationStatus.AVOID.value,
    ]
    assert output["order_entry_status"].tolist() == [
        "ready_to_enter",
        "review_only",
        "target_order_candidate",
        "not_actionable",
    ]
    assert output.loc[output["ticker"].eq("EVIDENCE"), "execution_blockers"].tolist() == [
        core.PROFITABILITY_CALIBRATION_BLOCKER
    ]


def test_goal_confidence_gap_audit_names_concrete_evidence_gaps() -> None:
    confidence = pd.DataFrame(
        [
            {
                "metric": "profitability_confidence_rating",
                "rating": 2.5,
                "threshold": core.MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING,
                "status": "BLOCK",
                "sample_size": 30,
                "evidence": "broker outcomes negative",
                "blockers": "broker_matched_options_agent_outcomes_negative",
                "required_next_action": "Do not promote current strategy cohort.",
            },
            {
                "metric": "order_entry_confidence_rating",
                "rating": 0.0,
                "threshold": core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING,
                "status": "BLOCK",
                "sample_size": 0,
                "evidence": "ready_to_enter_rows=0",
                "blockers": "no_green_ready_orders",
                "required_next_action": "No order-entry confidence is possible until a green ready_to_enter row exists.",
            },
            {
                "metric": "goal_confidence_gate",
                "rating": 0.0,
                "threshold": min(
                    core.MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING,
                    core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING,
                ),
                "status": "BLOCK",
                "sample_size": 30,
                "evidence": "profitability=2.5/10; order_entry=0.0/10",
                "blockers": "broker_matched_options_agent_outcomes_negative; no_green_ready_orders",
                "required_next_action": "Do not loosen green gates.",
            },
        ],
        columns=core.CONFIDENCE_AUDIT_COLUMNS,
    )
    trade_tickets = pd.DataFrame(
        [{"ticker": "WMT", "ready_to_enter": False, "target_order_status": "target_order_candidate"}]
    )
    execution_readiness = pd.DataFrame(
        [{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]
    )
    outcome = pd.DataFrame(
        [
            {
                "source": "codexuw_execute_outcome_ledger",
                "status": "BLOCK",
                "realized_pnl_count": 0,
                "current_ticker_realized_count": 0,
                "contributes_to_expectancy": False,
            }
        ]
    )
    broker_match = pd.DataFrame(
        [
            {
                "closed_trade_key": "WMT|2026-06-18",
                "match_status": "BLOCK",
                "match_source": "codexuw_execute_outcome_ledger",
                "can_backfill_realized_pnl": False,
            }
        ]
    )
    broker_outcomes = pd.DataFrame(
        [{"ticker": "WMT", "realized_pnl": -100.0, "match_sources": "codexuw_execute_outcome_ledger"}]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "bull_call_debit",
                "status": "BLOCK",
                "actual_support_status": "BLOCK",
                "actual_support_scope": "actual_route",
                "replay_bucket_status": "BLOCK",
                "replay_bucket_sample_size": 0,
            }
        ]
    )
    gap_plan = pd.DataFrame(
        [
            {
                "gap_rank": 1,
                "strategy_route": "bull_call_debit",
                "current_tickers": "WMT",
                "primary_gap": "actual_closed_outcomes_negative_or_weak",
                "actual_support_sample_gap": 30,
                "replay_bucket_sample_gap": 30,
                "diagnostic_replay_relaxed_dimensions": "liquidity_bucket",
            }
        ]
    )
    bucket_atlas = pd.DataFrame(
        [
            {
                "status": "BLOCK",
                "actual_bucket_status": "BLOCK",
                "replay_bucket_status": "BLOCK",
                "current_ticket_count": 1,
                "primary_gap": "no_actual_and_replay_bucket_pass",
            }
        ]
    )
    monthly = pd.DataFrame([{"metric": "expectancy_evidence", "value": 0, "status": "BLOCK", "note": "missing"}])

    audit = core.build_goal_confidence_gap_audit(
        confidence,
        trade_tickets,
        execution_readiness,
        outcome,
        broker_match,
        broker_outcomes,
        calibration,
        gap_plan,
        bucket_atlas,
        monthly,
    )
    summary = core.summarize_goal_confidence_gap_audit(audit)
    by_area = audit.set_index("area")

    assert list(audit.columns) == core.GOAL_CONFIDENCE_GAP_AUDIT_COLUMNS
    assert summary["status"] == "block"
    assert "broker_attribution" in summary["blocking_areas"]
    assert "profitability_calibration" in summary["blocking_areas"]
    assert by_area.loc["broker_attribution", "current_value"].startswith("exact_closed_trades=0")
    assert "avg_pnl=-100.0" in by_area.loc["broker_attribution", "current_value"]
    assert "sample >= 30" in by_area.loc["broker_attribution", "threshold"]
    assert "ready_to_enter_rows=0" in by_area.loc["order_entry_surface", "current_value"]
    assert "profitability_gap_plan.csv" in by_area.loc["profitability_calibration", "source_artifacts"]


def test_report_keeps_focus_review_queue_without_internal_promotion_section() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "status_icon": "Y",
                "status_label": "YELLOW review",
                "trade_plan": "BUY 1 MSFT 2026-07-17 400 Put / SELL 1 MSFT 2026-07-17 395 Put @ 1.00 DEBIT",
                "entry_limit": 1.0,
                "suggested_contracts": 2,
                "max_loss": 100.0,
                "ready_to_enter": False,
                "execution_status": "needs_confidence",
                "execution_gate_status": "blocked",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "LOW",
                "target_order_status": "",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "quality_gate_reason": "live Schwab chain Bear Put validated at 1.00 debit",
            }
        ]
    )
    report = core.render_report(
        "2026-06-09",
        final,
        pd.DataFrame(),
        {
            "row_counts": {"green_trade_tickets": 0, "target_order_ticket_rows": 0},
            "confidence_audit_summary": {"status": "block"},
            "promotion_readiness_audit_summary": {
                "status": "blocked",
                "blocking_gate_count": 2,
                "blocking_gates": ["profitability_confidence_goal", "green_ready_orders_present"],
                "required_evidence": ["positive expectancy", "green rows"],
            },
            "artifacts": {"promotion_readiness_audit": "/tmp/promotion_readiness_audit.csv"},
        },
    )

    assert "## Promotion Readiness" not in report
    assert "Not promotable yet." not in report
    assert "/tmp/promotion_readiness_audit.csv" not in report
    assert "Overall profitability/send-now confidence is blocked, so these rows are diagnostics only." in report
    assert "| Ticker | Signal | Reason | Qty | Reviewed / Target Price | Max Loss | Trade Plan |" in report
    assert "BUY 1 MSFT 2026-07-17 400 Put" in report
    assert "| MSFT | 🟡 YELLOW review |" in report


def test_no_green_reason_leads_with_profitability_calibration_cause() -> None:
    reason = core._plain_no_green_reason(
        green_count=0,
        target_count=0,
        review_ticket_count=2,
        live_mode="live_schwab",
        row_counts={"live_chain_validation": 2},
        portfolio_context_status="live",
        blocking_gates=[],
        execution_context={"agentic_reviews_ready": True},
        calibration_summary={"status": "block"},
        confidence_summary={
            "order_entry_confidence_rating": 0.0,
            "order_mechanics_confidence_rating": 7.0,
        },
    )

    assert reason == (
        "point-in-time profitability calibration is not passing; "
        "no ticket qualifies for send-now entry"
    )


def test_yellow_exact_review_block_is_labeled_watch_only() -> None:
    row = {
        "ready_to_enter": False,
        "target_order_status": "target_order_candidate",
        "execution_blockers": "",
        "contract_review_status": "BLOCK",
        "contract_review_missing_agents": "structure_builder; skeptic",
    }

    assert core._ticket_recheck_summary(row) == (
        "watch-only; exact contract review missing (structure_builder; skeptic)"
    )
    assert core._ticket_order_readiness(row) == "target_order_after_exact_contract_review"
    assert core._ticket_action(row) == "watch_only_until_exact_contract_review"
    assert core._target_rows_require_watch_only_from_tickets(pd.DataFrame([row])) is True

    earnings_row = {
        "ready_to_enter": False,
        "contract_review_status": "PASS",
        "earnings_before_expiry": True,
        "event_exit_deadline": "2026-08-17",
    }
    assert core._ticket_recheck_summary(earnings_row) == (
        "conditional pre-earnings target; exit by 2026-08-17; do not carry through earnings"
    )


def test_promotion_readiness_audit_proves_blocked_promotion_without_counting_verdict_as_gate() -> None:
    confidence = pd.DataFrame(
        [
            {
                "metric": "profitability_confidence_rating",
                "rating": 3.0,
                "threshold": 7.0,
                "status": "BLOCK",
                "sample_size": 6,
                "evidence": "broker_backfilled_forward_outcomes=negative",
                "blockers": "broker_backfilled_forward_outcomes_not_positive",
                "required_next_action": "collect positive outcomes",
            },
            {
                "metric": "order_entry_confidence_rating",
                "rating": 0.0,
                "threshold": 7.0,
                "status": "BLOCK",
                "sample_size": 0,
                "evidence": "ready_to_enter_rows=0",
                "blockers": "no_green_ready_orders",
                "required_next_action": "produce validated green rows",
            },
            {
                "metric": "goal_confidence_gate",
                "rating": 0.0,
                "threshold": 7.0,
                "status": "BLOCK",
                "sample_size": 6,
                "evidence": "profitability=3.0/10; order_entry=0.0/10",
                "blockers": "broker_backfilled_forward_outcomes_not_positive; no_green_ready_orders",
                "required_next_action": "do not promote",
            },
        ],
        columns=core.CONFIDENCE_AUDIT_COLUMNS,
    )
    goal_gap = pd.DataFrame(
        [
            {
                "area": "broker_attribution",
                "status": "BLOCK",
                "current_value": "sample=6 avg_pnl=-83.17",
                "threshold": "positive broker attribution",
                "gap_detail": "not profitable",
                "required_evidence": "Backfill more positive exact broker matches.",
                "source_artifacts": "broker_backfilled_forward_outcomes.csv",
            }
        ],
        columns=core.GOAL_CONFIDENCE_GAP_AUDIT_COLUMNS,
    )
    outcome = pd.DataFrame(
        [
            {
                "source": "broker_backfilled_forward_outcomes",
                "status": "BLOCK",
                "realized_pnl_count": 6,
                "row_count": 6,
                "note": "negative",
            }
        ]
    )
    broker_backfilled = pd.DataFrame(
        [
            {"ticker": "META", "realized_pnl": -275.0, "source_ledger": "codexuw_execute_outcome_ledger"},
            {"ticker": "WMT", "realized_pnl": 20.0, "source_ledger": "codexuw_execute_outcome_ledger"},
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 0, "status": "BLOCK", "note": "none"},
            {"metric": "expectancy_evidence", "value": 2, "status": "BLOCK", "note": "negative"},
        ]
    )

    audit = core.build_promotion_readiness_audit(
        confidence,
        goal_gap,
        outcome,
        pd.DataFrame(columns=core.BROKER_OUTCOME_MATCH_AUDIT_COLUMNS),
        pd.DataFrame(columns=core.BROKER_MATCHED_OUTCOME_COLUMNS),
        broker_backfilled,
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        pd.DataFrame(columns=core.PROFITABILITY_BUCKET_ATLAS_COLUMNS),
        pd.DataFrame(),
        monthly,
    )
    summary = core.summarize_promotion_readiness_audit(audit)

    assert list(audit.columns) == core.PROMOTION_READINESS_AUDIT_COLUMNS
    assert audit.loc[audit["gate"].eq("promotion_verdict"), "status"].tolist() == ["BLOCK"]
    assert summary["status"] == "blocked"
    assert "promotion_verdict" not in summary["blocking_gates"]
    assert "broker_attribution_positive" in summary["blocking_gates"]
    assert "green_ready_orders_present" in summary["blocking_gates"]


def test_report_keeps_focus_review_diagnostics_when_goal_confidence_passes() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "status_icon": "Y",
                "status_label": "YELLOW review",
                "trade_plan": "BUY 1 MSFT 2026-07-17 400 Put / SELL 1 MSFT 2026-07-17 395 Put @ 1.00 DEBIT",
                "entry_limit": 1.0,
                "suggested_contracts": 2,
                "max_loss": 100.0,
                "ready_to_enter": False,
                "execution_status": "needs_confidence",
                "execution_gate_status": "blocked",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "LOW",
                "target_order_status": "",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "quality_gate_reason": "live Schwab chain Bear Put validated at 1.00 debit",
            }
        ]
    )
    report = core.render_report(
        "2026-06-09",
        final,
        pd.DataFrame(),
        {
            "row_counts": {"green_trade_tickets": 0, "target_order_ticket_rows": 0},
            "confidence_audit_summary": {"status": "pass"},
        },
    )

    assert "| Ticker | Signal | Reason | Qty | Reviewed / Target Price | Max Loss | Trade Plan |" in report
    assert "BUY 1 MSFT 2026-07-17 400 Put" in report


def _write_minimal_uw_fixture(root: Path) -> None:
    date_dir = root / "2026-05-22"
    date_dir.mkdir()
    pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "ticker": "WMT",
                "close": 100.0,
                "call_volume": 1000,
                "put_volume": 1200,
                "call_premium": 500_000,
                "put_premium": 1_500_000,
                "bullish_premium": 5_000_000,
                "bearish_premium": 100_000,
                "marketcap": 650_000_000_000,
                "issue_type": "Common Stock",
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "iv_rank": 60,
                "iv30d": 0.30,
            }
        ]
    ).to_csv(date_dir / "stock-screener-2026-05-22.csv", index=False)
    pd.DataFrame(
        [
            {
                "option_symbol": "WMT260619P00095000",
                "date": "2026-05-22",
                "volume": 5000,
                "open_interest": 20000,
                "premium": 1_000_000,
                "ask_side_volume": 3000,
                "bid_side_volume": 1500,
                "bid": 2.00,
                "ask": 2.20,
            },
            {
                "option_symbol": "WMT260619P00090000",
                "date": "2026-05-22",
                "volume": 2500,
                "open_interest": 15000,
                "premium": 300_000,
                "ask_side_volume": 500,
                "bid_side_volume": 1200,
                "bid": 0.80,
                "ask": 1.00,
            },
        ]
    ).to_csv(date_dir / "hot-chains-2026-05-22.csv", index=False)
    pd.DataFrame(
        [
            {
                "option_symbol": "WMT260619P00095000",
                "underlying_symbol": "WMT",
                "oi_diff_plain": 2500,
                "curr_oi": 20000,
                "volume": 5000,
            }
        ]
    ).to_csv(date_dir / "chain-oi-changes-2026-05-22.csv", index=False)


def _write_wmt_chain_snapshot(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir()
    payload = {
        "status": "SUCCESS",
        "symbol": "WMT",
        "underlyingPrice": 100.0,
        "putExpDateMap": {
            "2026-06-19:28": {
                "95.0": [
                    {
                        "symbol": "WMT  260619P00095000",
                        "strikePrice": 95.0,
                        "bid": 1.40,
                        "ask": 1.40,
                        "mark": 1.40,
                        "delta": -0.22,
                        "volatility": 0.32,
                        "openInterest": 5000,
                        "totalVolume": 1000,
                    }
                ],
                "90.0": [
                    {
                        "symbol": "WMT  260619P00090000",
                        "strikePrice": 90.0,
                        "bid": 0.40,
                        "ask": 0.40,
                        "mark": 0.40,
                        "delta": -0.10,
                        "volatility": 0.34,
                        "openInterest": 4000,
                        "totalVolume": 900,
                    }
                ],
            }
        },
        "callExpDateMap": {},
    }
    (snapshot_dir / "WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wmt_chain_snapshot_in_legacy_chains_layout(snapshot_dir: Path) -> None:
    _write_wmt_chain_snapshot(snapshot_dir)
    payload = json.loads((snapshot_dir / "WMT.json").read_text(encoding="utf-8"))
    (snapshot_dir / "WMT.json").unlink()
    chains_dir = snapshot_dir / "chains"
    chains_dir.mkdir()
    (chains_dir / "chain_WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _mark_strategy_expectancy_pass(frame: pd.DataFrame, tickers=None, sample: int = 5) -> pd.DataFrame:
    out = frame.copy()
    if "ticker" not in out.columns:
        return out
    mask = pd.Series(True, index=out.index)
    if tickers is not None:
        mask = out["ticker"].astype(str).isin(tickers)
    out.loc[mask, "actual_forward_expectancy_status"] = "PASS"
    out.loc[mask, "actual_forward_expectancy_sample_size"] = sample
    out.loc[mask, "actual_forward_strategy_expectancy_status"] = "PASS"
    out.loc[mask, "actual_forward_strategy_expectancy_sample_size"] = sample
    out.loc[mask, "actual_forward_strategy_expectancy_family"] = "vertical_spread"
    out.loc[mask, "actual_forward_strategy_expectancy_scope"] = "ticker_strategy"
    out.loc[mask, "profitability_calibration_status"] = "PASS"
    out.loc[mask, "profitability_calibration_scope"] = "actual_ticker_bucket"
    out.loc[mask, "profitability_calibration_replay_status"] = "PASS"
    out.loc[mask, "profitability_calibration_replay_sample_size"] = core.MIN_EXPECTANCY_SAMPLE_SIZE
    out.loc[mask, "contract_review_status"] = "PASS"
    out.loc[mask, "live_probability_proxy"] = 0.55
    out.loc[mask, "live_quote_width_pct"] = 0.10
    out.loc[mask, "live_theta_burn_pct"] = 0.01
    out.loc[mask, "live_breakeven_expected_move_ratio"] = 0.50
    out.loc[mask, "macro_calendar_status"] = "verified"
    out.loc[mask, "macro_event_count_before_expiry"] = 0
    out.loc[mask, "earnings_before_expiry"] = False
    if "dte" not in out.columns:
        out["dte"] = 30
    else:
        out.loc[mask & out["dte"].isna(), "dte"] = 30
    return out


def _write_wmt_wide_market_snapshot(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir()
    payload = {
        "status": "SUCCESS",
        "symbol": "WMT",
        "underlyingPrice": 100.0,
        "putExpDateMap": {
            "2026-06-19:28": {
                "95.0": [
                    {
                        "symbol": "WMT  260619P00095000",
                        "strikePrice": 95.0,
                        "bid": 2.00,
                        "ask": 4.00,
                        "mark": 3.00,
                        "delta": -0.22,
                        "volatility": 0.32,
                        "openInterest": 5000,
                        "totalVolume": 1000,
                    }
                ],
                "90.0": [
                    {
                        "symbol": "WMT  260619P00090000",
                        "strikePrice": 90.0,
                        "bid": 0.40,
                        "ask": 0.40,
                        "mark": 0.40,
                        "delta": -0.10,
                        "volatility": 0.34,
                        "openInterest": 4000,
                        "totalVolume": 900,
                    }
                ],
            }
        },
        "callExpDateMap": {},
    }
    (snapshot_dir / "WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wmt_call_debit_snapshot(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir()
    payload = {
        "status": "SUCCESS",
        "symbol": "WMT",
        "underlyingPrice": 100.0,
        "putExpDateMap": {},
        "callExpDateMap": {
            "2026-06-19:28": {
                "100.0": [
                    {
                        "symbol": "WMT  260619C00100000",
                        "strikePrice": 100.0,
                        "bid": 2.00,
                        "ask": 2.00,
                        "mark": 2.00,
                        "delta": 0.55,
                        "volatility": 0.32,
                        "openInterest": 5000,
                        "totalVolume": 1000,
                    }
                ],
                "105.0": [
                    {
                        "symbol": "WMT  260619C00105000",
                        "strikePrice": 105.0,
                        "bid": 0.60,
                        "ask": 0.60,
                        "mark": 0.60,
                        "delta": 0.30,
                        "volatility": 0.34,
                        "openInterest": 4000,
                        "totalVolume": 900,
                    }
                ],
            }
        },
    }
    (snapshot_dir / "WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wmt_call_debit_with_better_breakout_snapshot(snapshot_dir: Path) -> None:
    snapshot_dir.mkdir()
    payload = {
        "status": "SUCCESS",
        "symbol": "WMT",
        "underlyingPrice": 104.0,
        "putExpDateMap": {},
        "callExpDateMap": {
            "2026-06-19:28": {
                "100.0": [
                    {
                        "symbol": "WMT  260619C00100000",
                        "strikePrice": 100.0,
                        "bid": 3.50,
                        "ask": 3.50,
                        "mark": 3.50,
                        "delta": 0.55,
                        "volatility": 0.32,
                        "openInterest": 5000,
                        "totalVolume": 1000,
                    }
                ],
                "105.0": [
                    {
                        "symbol": "WMT  260619C00105000",
                        "strikePrice": 105.0,
                        "bid": 0.00,
                        "ask": 0.00,
                        "mark": 0.00,
                        "delta": 0.20,
                        "volatility": 0.34,
                        "openInterest": 4000,
                        "totalVolume": 900,
                    }
                ],
                "110.0": [
                    {
                        "symbol": "WMT  260619C00110000",
                        "strikePrice": 110.0,
                        "bid": 1.40,
                        "ask": 1.40,
                        "mark": 1.40,
                        "delta": 0.42,
                        "volatility": 0.33,
                        "openInterest": 6000,
                        "totalVolume": 1400,
                    }
                ],
                "115.0": [
                    {
                        "symbol": "WMT  260619C00115000",
                        "strikePrice": 115.0,
                        "bid": 0.00,
                        "ask": 0.00,
                        "mark": 0.00,
                        "delta": 0.18,
                        "volatility": 0.35,
                        "openInterest": 5500,
                        "totalVolume": 1200,
                    }
                ],
            }
        },
    }
    (snapshot_dir / "WMT.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_wmt_red_flag_news(root: Path) -> None:
    news_dir = root / "2026-05-22" / "browser_text"
    news_dir.mkdir(parents=True, exist_ok=True)
    (news_dir / "browser-text-capture-news-WMT-2026-05-22.txt").write_text(
        "WMT faces SEC probe after downgrade warning\nAnalysts cite investigation risk and guidance cut concerns.",
        encoding="utf-8",
    )


def test_options_agent_core_has_no_daily_v4_dependency() -> None:
    source = inspect.getsource(core)

    forbidden = ("codexuw.daily_v4", "daily_v4", "codexdaily_v4")
    for token in forbidden:
        assert token not in source


def test_options_agent_package_has_no_daily_v4_dependency() -> None:
    package_dir = Path(core.__file__).parent
    forbidden = ("codexuw.daily_v4", "daily_v4", "codexdaily_v4", "out/codexdaily")

    for path in package_dir.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{token} found in {path}"


def test_run_pipeline_does_not_import_daily_v4(tmp_path: Path) -> None:
    class DailyV4Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "codexuw.daily_v4":
                raise AssertionError("Options Agent imported Codex Daily V4")
            return None

    root = tmp_path
    _write_minimal_uw_fixture(root)
    blocker = DailyV4Blocker()
    prior = sys.modules.pop("codexuw.daily_v4", None)
    sys.meta_path.insert(0, blocker)
    try:
        run_pipeline("2026-05-22", root=root, top_trades=3)
    finally:
        sys.meta_path.remove(blocker)
        if prior is not None:
            sys.modules["codexuw.daily_v4"] = prior


def test_default_output_paths_use_options_agent_namespace(tmp_path: Path) -> None:
    paths = output_paths("2026-05-22", root=tmp_path)

    assert paths["out_dir"] == tmp_path / "out" / "options_agent" / "2026-05-22"
    assert paths["manifest"].name == "options_agent_manifest_2026-05-22.json"
    assert paths["report"].name == "options_agent_report_2026-05-22.md"
    assert paths["strategy_outcome_atlas"].name == "strategy_outcome_atlas.csv"
    assert paths["profitability_calibration"].name == "profitability_calibration.csv"
    assert paths["profitability_gap_plan"].name == "profitability_gap_plan.csv"
    assert (
        paths["profitability_calibration_intersection_gap"].name
        == "profitability_calibration_intersection_gap.csv"
    )
    assert paths["profitability_evidence_backfill_plan"].name == "profitability_evidence_backfill_plan.csv"
    assert paths["profitability_bucket_atlas"].name == "profitability_bucket_atlas.csv"
    assert paths["outcome_evidence_audit"].name == "outcome_evidence_audit.csv"
    assert paths["broker_outcome_match_audit"].name == "broker_outcome_match_audit.csv"
    assert paths["broker_matched_outcomes"].name == "broker_matched_outcomes.csv"
    assert paths["execution_fill_quality"].name == "execution_fill_quality.csv"
    assert paths["goal_confidence_gap_audit"].name == "goal_confidence_gap_audit.csv"


def test_pipeline_source_provenance_uses_code_root_for_overlay_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code_root = tmp_path / "code"
    core_path = code_root / "uwos" / "options_agent" / "core.py"
    core_path.parent.mkdir(parents=True)
    core_path.write_text("# pinned code\n", encoding="utf-8")
    monkeypatch.setattr(core, "project_root", lambda: code_root)

    provenance = core._pipeline_source_provenance(tmp_path / "overlay")

    assert provenance["code_root"] == str(code_root)
    assert provenance["file_sha256"]["uwos/options_agent/core.py"] == hashlib.sha256(
        core_path.read_bytes()
    ).hexdigest()


def test_write_json_replaces_non_finite_values_with_null(tmp_path: Path) -> None:
    path = tmp_path / "strict.json"

    core._write_json(path, {"nan": float("nan"), "infinity": float("inf"), "ok": 1.25})

    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    assert json.loads(text, parse_constant=lambda token: pytest.fail(token)) == {
        "infinity": None,
        "nan": None,
        "ok": 1.25,
    }


def test_portfolio_risk_annotations_do_not_suppress_qualified_trade() -> None:
    candidate = {
        "ticker": "WMT",
        "structure": "bull put spread",
        "quality_status": "qualified",
        "recommendation_status": "ENTER",
        "hard_rejects": "portfolio_concentration",
    }
    portfolio = {
        "status": "ok",
        "total_value": 100_000,
        "option_underlyings": ["WMT"],
        "large_equity_exposure": {"WMT": 7_500},
    }

    rows = apply_portfolio_risk_annotations([candidate], portfolio)

    assert len(rows) == 1
    row = rows[0]
    assert row["visible_in_final_board"] is True
    assert row["portfolio_risk_flag"] is True
    assert row["recommendation_status"] == RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value
    assert row["hard_rejects"] == ""
    assert "existing option exposure in WMT" in row["portfolio_risk_note"]
    assert "large equity exposure in WMT" in row["portfolio_risk_note"]
    assert "portfolio-only blocker converted to annotation" in row["portfolio_risk_note"]
    assert "execution gate unaffected" in row["portfolio_risk_note"]


def test_non_portfolio_hard_blocker_remains_hard_blocker() -> None:
    candidate = {
        "ticker": "HOOD",
        "quality_status": "qualified",
        "recommendation_status": "ENTER",
        "hard_rejects": "bad_liquidity; portfolio_concentration",
    }
    rows = apply_portfolio_risk_annotations([candidate], {"option_underlyings": ["HOOD"]})

    row = rows[0]
    assert row["visible_in_final_board"] is True
    assert row["hard_rejects"] == "bad_liquidity"
    assert row["recommendation_status"] == RecommendationStatus.AVOID.value
    assert "objective hard blocker: bad_liquidity" in row["status_reason"]
    assert "existing option exposure in HOOD" in row["portfolio_risk_note"]


def test_objective_concentration_reject_is_not_misread_as_portfolio_risk() -> None:
    candidate = {
        "ticker": "HOOD",
        "quality_status": "qualified",
        "recommendation_status": "ENTER",
        "hard_rejects": "strike_concentration_bad_fill",
    }
    rows = apply_portfolio_risk_annotations([candidate], {})

    row = rows[0]
    assert row["recommendation_status"] == RecommendationStatus.AVOID.value
    assert row["hard_rejects"] == "strike_concentration_bad_fill"
    assert row["portfolio_risk_flag"] is False


def test_portfolio_risk_does_not_upgrade_waiting_trade_to_enter() -> None:
    candidate = {
        "ticker": "WMT",
        "quality_status": "qualified",
        "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
        "status_reason": "fresh quote required",
    }

    row = apply_portfolio_risk_annotations([candidate], {"option_underlyings": ["WMT"]})[0]

    assert row["recommendation_status"] == RecommendationStatus.WAIT_FOR_PRICE.value
    assert row["portfolio_risk_flag"] is True
    assert row["status_reason"] == "fresh quote required"
    assert "portfolio annotation only" not in row["status_reason"]


def test_candidate_generation_can_include_all_directional_rows() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": f"T{i}",
                "bias": "bullish",
                "score": 70 - i,
                "signal_premium": 1_000_000 + i,
                "quality_status": "qualified",
                "flow_reason": f"reason {i}",
            }
            for i in range(150)
        ]
        + [
            {
                "ticker": "NEUT",
                "bias": "neutral",
                "score": 99,
                "signal_premium": 9_999_999,
                "quality_status": "watch",
                "flow_reason": "neutral",
            }
        ]
    )

    candidates = core.generate_candidates(raw, limit=None, focus_tickers=())

    assert len(candidates) == 150
    assert "NEUT" not in candidates["ticker"].tolist()
    assert candidates["candidate_rank"].iloc[-1] == 150


def test_candidate_generation_rescues_core_neutral_rows_when_price_tape_is_directional() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "bias": "neutral",
                "score": 55,
                "signal_premium": 1_000_000,
                "quality_status": "watch",
                "underlying_quality_tier": "core",
                "marketcap": 500_000_000_000,
                "close": 98.0,
                "prev_close": 100.0,
                "flow_reason": "neutral UW flow",
                "flow_bias_label": "neutral",
            }
            for ticker in ("SPY", "QQQ", "IWM", "DIA", "AAPL")
        ]
    )
    regime = core.build_market_price_regime(raw, "2026-06-09")
    annotated = core.annotate_macro_tape_candidates(raw, regime)

    candidates = core.generate_candidates(annotated, limit=None, focus_tickers=core.CORE_AUDIT_TICKERS)
    aapl = candidates[candidates["ticker"].eq("AAPL")].iloc[0]
    coverage = core.build_coverage_audit(annotated, candidates, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert regime["tape_direction"] == "bearish"
    assert aapl["candidate_source"] == "macro_tape_candidate"
    assert aapl["bias"] == "bearish"
    assert aapl["flow_bias_label"] == "neutral"
    assert coverage.loc[coverage["ticker"].eq("AAPL"), "coverage_status"].tolist() == ["MACRO_TAPE_CANDIDATE"]


def test_macro_tape_rescue_uses_ticker_down_move_when_index_tape_is_bullish() -> None:
    rows = []
    for ticker, close, prev_close in (
        ("SPY", 102.0, 100.0),
        ("QQQ", 102.0, 100.0),
        ("IWM", 102.0, 100.0),
        ("DIA", 102.0, 100.0),
        ("AAPL", 97.0, 100.0),
    ):
        rows.append(
            {
                "ticker": ticker,
                "bias": "neutral",
                "score": 65,
                "signal_premium": 1_000_000,
                "quality_status": "watch",
                "underlying_quality_tier": "core",
                "marketcap": 500_000_000_000,
                "close": close,
                "prev_close": prev_close,
                "flow_reason": "neutral UW flow",
                "flow_bias_label": "neutral",
            }
        )
    raw = pd.DataFrame(rows)

    regime = core.build_market_price_regime(raw, "2026-06-09")
    annotated = core.annotate_macro_tape_candidates(raw, regime)
    candidates = core.generate_candidates(annotated, limit=None, focus_tickers=core.CORE_AUDIT_TICKERS)
    aapl = candidates[candidates["ticker"].eq("AAPL")].iloc[0]

    assert regime["tape_direction"] == "bullish"
    assert aapl["candidate_source"] == "macro_tape_candidate"
    assert aapl["macro_tape_direction"] == "bearish"
    assert aapl["bias"] == "bearish"
    assert "despite bullish index tape" in aapl["macro_tape_reason"]


def test_market_price_regime_downgrades_bullish_index_when_mega_breadth_bearish() -> None:
    rows = []
    for ticker, close, prev_close in (
        ("SPY", 102.0, 100.0),
        ("QQQ", 102.0, 100.0),
        ("IWM", 102.0, 100.0),
        ("DIA", 102.0, 100.0),
        ("AAPL", 96.0, 100.0),
        ("MSFT", 95.0, 100.0),
        ("NVDA", 97.0, 100.0),
        ("AMZN", 96.0, 100.0),
        ("META", 97.0, 100.0),
        ("GOOG", 96.0, 100.0),
        ("TSLA", 95.0, 100.0),
        ("AMD", 105.0, 100.0),
    ):
        rows.append(
            {
                "ticker": ticker,
                "bias": "neutral",
                "score": 65,
                "signal_premium": 1_000_000,
                "quality_status": "watch",
                "underlying_quality_tier": "core",
                "marketcap": 500_000_000_000,
                "close": close,
                "prev_close": prev_close,
                "flow_reason": "neutral UW flow",
                "flow_bias_label": "neutral",
            }
        )
    raw = pd.DataFrame(rows)

    regime = core.build_market_price_regime(raw, "2026-06-09")
    market_regime = core.build_market_regime(raw, market_price_regime=regime)

    assert regime["tape_direction"] == "mixed"
    assert regime["regime"] == "mixed"
    assert regime["mega_cap_negative_breadth"] >= 0.60
    assert "mega-cap breadth is bearish" in regime["note"]
    assert market_regime["regime"] == "mixed"


def test_live_debit_fallback_keeps_route_reason_consistent() -> None:
    row = {
        "ticker": "AAPL",
        "strategy": "bear_call_credit",
        "strategy_route": "bear_call_credit",
        "route_reason": "bearish_core_credit_route",
        "signal_premium": 1_000_000,
        "combined_flow_bias": -0.5,
    }
    live = {
        "debit": 1.25,
        "spread_width": 5.0,
        "short_strike": 170.0,
        "long_strike": 175.0,
        "target_entry": 2.25,
        "short_leg": "AAPL  260717P00170000",
        "long_leg": "AAPL  260717P00175000",
        "long_delta": -0.40,
        "quote_width_pct": 0.05,
        "short_oi": 2_000,
        "short_volume": 200,
        "long_oi": 2_500,
        "long_volume": 300,
    }

    updated = core._apply_live_debit_spread(
        row,
        live,
        direction="Bear Put",
        expiry=dt.date(2026, 7, 17),
        spot=165.0,
        asof_date=dt.date(2026, 6, 9),
    )

    assert updated["strategy_route"] == "bear_put_debit"
    assert updated["structure"] == "bear put debit spread"
    assert updated["route_reason"] == "bearish_core_defined_risk_downside_route"


def test_live_long_call_validation_preserves_single_leg_route_and_gates() -> None:
    row = {
        "ticker": "AAPL",
        "strategy": "long_call",
        "strategy_route": "long_call",
        "signal_premium": 5_000_000,
        "combined_flow_bias": 0.5,
        "macro_tape_candidate": False,
    }
    live = {
        "debit": 2.00,
        "mid_debit": 1.95,
        "bid": 1.90,
        "ask": 2.00,
        "long_strike": 100.0,
        "target_entry": 2.00,
        "target_exit": 5.10,
        "long_leg": "AAPL  260717C00100000",
        "long_delta": 0.55,
        "long_theta": -0.05,
        "theta_burn_pct": 0.025,
        "expected_move_pct": 0.10,
        "breakeven_expected_move_ratio": 0.20,
        "dte": 38,
        "quote_width_pct": 0.05,
        "long_oi": 2_000,
        "long_volume": 200,
        "construction_source": "long_option_live",
        "construction_reason": "best liquid near-money long call from live Schwab chain",
    }

    updated = core._apply_live_long_option(
        row,
        live,
        direction="Long Call",
        expiry=dt.date(2026, 7, 17),
        spot=100.0,
        asof_date=dt.date(2026, 6, 9),
    )
    blockers = core._send_now_economics_blockers(
        updated,
        ticket=updated["trade_plan"],
        entry_limit=updated["entry_limit"],
    )

    assert updated["strategy_route"] == "long_call"
    assert updated["strategy_family"] == "long_call"
    assert updated["structure"] == "long call"
    assert updated["sell_leg"] == ""
    assert updated["buy_leg"].startswith("BUY 1 AAPL 2026-07-17 100 Call")
    assert updated["max_loss"] == 200.0
    assert updated["max_profit"] == 310.0
    assert updated["live_probability_proxy"] == pytest.approx(0.55)
    assert updated["live_net_theta_per_contract"] == pytest.approx(-5.0)
    assert updated["live_breakeven_expected_move_ratio"] == pytest.approx(0.20)
    assert "send_now_debit_reward_risk" not in "; ".join(blockers)


def test_live_credit_spread_preserves_both_leg_greeks_and_expected_move() -> None:
    row = {
        "ticker": "AAPL",
        "signal_premium": 5_000_000,
        "combined_flow_bias": 0.5,
        "macro_tape_candidate": False,
    }
    live = {
        "credit": 1.60,
        "mid_credit": 1.70,
        "natural_credit": 1.50,
        "spread_width": 5.0,
        "short_strike": 195.0,
        "long_strike": 190.0,
        "short_leg": "AAPL  260821P00195000",
        "long_leg": "AAPL  260821P00190000",
        "pop_delta_proxy": 0.73,
        "short_delta": -0.27,
        "long_delta": -0.16,
        "short_theta": -0.08,
        "long_theta": -0.04,
        "net_theta": 0.04,
        "expected_move_pct": 0.07,
        "breakeven_expected_move_ratio": 0.44,
        "quote_width_pct": 0.08,
        "short_oi": 2_000,
        "short_volume": 300,
        "long_oi": 1_500,
        "long_volume": 200,
    }

    updated = core._apply_live_credit_spread(
        row,
        live,
        direction="Bull Put",
        expiry=dt.date(2026, 8, 21),
        spot=205.0,
        asof_date=dt.date(2026, 7, 9),
    )
    tasks = core.build_contract_review_tasks(
        pd.DataFrame(
            [
                {
                    **updated,
                    "live_validation_status": "PASS",
                    "underlying_quality_tier": "core",
                }
            ]
        )
    )

    assert updated["live_probability_proxy"] == pytest.approx(0.73)
    assert updated["live_short_delta"] == pytest.approx(-0.27)
    assert updated["live_long_delta"] == pytest.approx(-0.16)
    assert updated["live_short_theta"] == pytest.approx(-0.08)
    assert updated["live_long_theta"] == pytest.approx(-0.04)
    assert updated["live_net_theta_per_share"] == pytest.approx(0.04)
    assert updated["live_net_theta_per_contract"] == pytest.approx(4.0)
    assert updated["live_expected_move_pct"] == pytest.approx(0.07)
    assert updated["live_breakeven_expected_move_ratio"] == pytest.approx(0.44)
    assert tasks["contract_count"] == 1
    assert tasks["contracts"][0]["live_long_delta"] == pytest.approx(-0.16)
    assert tasks["contracts"][0]["live_short_theta"] == pytest.approx(-0.08)
    assert tasks["contracts"][0]["live_long_theta"] == pytest.approx(-0.04)


def test_contract_review_tasks_do_not_apply_an_implicit_top_n_cap() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": f"T{idx}",
                "strategy_route": "bull_call_debit",
                "expiry": "2026-08-21",
                "dte": 35,
                "buy_leg": f"BUY 1 T{idx} 2026-08-21 100 Call",
                "sell_leg": f"SELL 1 T{idx} 2026-08-21 105 Call",
                "trade_plan": (
                    f"BUY 1 T{idx} 2026-08-21 100 Call / "
                    f"SELL 1 T{idx} 2026-08-21 105 Call @ 1.50 DEBIT"
                ),
                "recommendation_status": RecommendationStatus.ENTER.value,
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "score": 100.0 - idx,
            }
            for idx in range(61)
        ]
    )

    tasks = core.build_contract_review_tasks(priced)
    explicitly_limited = core.build_contract_review_tasks(priced, limit=10)

    assert tasks["contract_count"] == 61
    assert explicitly_limited["contract_count"] == 10
    assert all(contract["dte"] == 35 for contract in tasks["contracts"])


def test_contract_review_tasks_label_long_option_profit_as_target_not_theoretical_cap() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "strategy_route": "long_call",
                "structure": "long call",
                "expiry": "2026-08-21",
                "dte": 35,
                "buy_leg": "BUY 1 AAPL 2026-08-21 205 Call",
                "trade_plan": "BUY 1 AAPL 2026-08-21 205 Call @ 5.00 DEBIT",
                "entry_limit": 5.00,
                "target_exit": 9.00,
                "max_profit": 400.0,
                "max_loss": 500.0,
                "recommendation_status": RecommendationStatus.ENTER.value,
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
            }
        ]
    )

    task = core.build_contract_review_tasks(priced)["contracts"][0]

    assert task["max_profit"] == pytest.approx(400.0)
    assert task["planned_target_profit"] == pytest.approx(400.0)
    assert task["theoretical_max_profit"] == "uncapped"
    assert "planned profit at target exit" in task["profit_semantics"]


def test_completed_contract_caution_is_not_reported_as_missing_review() -> None:
    row = {
        "recommendation_status": RecommendationStatus.ENTER.value,
        "live_validation_status": "PASS",
        "contract_review_status": "WARN",
        "contract_review_missing_agents": "",
        "trade_plan": "SELL 1 AAPL 2026-08-21 195 Put / BUY 1 AAPL 2026-08-21 190 Put @ 1.60 CREDIT",
        "entry_limit": 1.60,
        "credit_width_ratio": 0.32,
        "suggested_contracts": 1,
    }

    blockers = core._execution_blockers_for_row(
        row,
        "ready",
        row["trade_plan"],
        row["entry_limit"],
        row["suggested_contracts"],
        core._execution_context_or_default({}),
    )

    assert "contract_specific_agent_review_caution" in blockers
    assert "contract_specific_agent_reviews_missing" not in blockers


def test_candidate_generation_keeps_core_neutral_rows_for_subagent_review_without_forcing_trades(tmp_path: Path) -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "bias": "neutral",
                "score": score,
                "signal_premium": 1_000_000,
                "quality_status": "watch",
                "underlying_quality_tier": "core",
                "marketcap": 500_000_000_000,
                "close": 100.0,
                "prev_close": 100.0,
                "flow_reason": "neutral UW flow",
                "flow_bias_label": "neutral",
            }
            for ticker, score in (("SPY", 72), ("QQQ", 71), ("AAPL", 70), ("NVDA", 69))
        ]
    )
    regime = core.build_market_price_regime(raw, "2026-06-09")
    annotated = core.annotate_macro_tape_candidates(raw, regime)

    candidates = core.generate_candidates(annotated, limit=None, focus_tickers=core.CORE_AUDIT_TICKERS)
    research = core.build_research_tasks(candidates, {"regime": "mixed"}, pd.DataFrame(), top_trades=20)
    dispatch = core.build_agent_dispatch_plan(research, "2026-06-09", output_paths("2026-06-09", root=tmp_path))
    coverage = core.build_coverage_audit(annotated, candidates, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert regime["tape_direction"] == "mixed"
    assert set(candidates["ticker"]) == {"SPY", "QQQ", "AAPL", "NVDA"}
    assert set(candidates["candidate_source"]) == {"core_audit_review"}
    assert set(task["ticker"] for task in research["tasks"]) == {"SPY", "QQQ", "AAPL", "NVDA"}
    assert {"SPY", "QQQ", "AAPL", "NVDA"}.issubset(dispatch["common_context"]["required_review_tickers"])
    by_ticker = coverage.set_index("ticker")
    assert by_ticker.loc["SPY", "coverage_status"] == "NO_DIRECTIONAL_EDGE"
    assert by_ticker.loc["AAPL", "status_color"] == "gray"


def test_candidate_generation_prioritizes_liquid_underlyings_before_speculative_flow() -> None:
    raw = pd.DataFrame(
        [
            {
                "ticker": "JUNK",
                "bias": "bullish",
                "score": 99,
                "signal_premium": 50_000_000,
                "quality_status": "qualified",
                "underlying_quality_tier": "speculative",
                "flow_reason": "huge flow but weak underlying",
            },
            {
                "ticker": "AAPL",
                "bias": "bullish",
                "score": 70,
                "signal_premium": 2_000_000,
                "quality_status": "qualified",
                "underlying_quality_tier": "core",
                "flow_reason": "liquid large cap",
            },
        ]
    )

    candidates = core.generate_candidates(raw, limit=None, focus_tickers=())

    assert candidates["ticker"].tolist() == ["AAPL", "JUNK"]


def test_agent_dispatch_prompt_lists_all_candidate_tickers(tmp_path: Path) -> None:
    tasks = {
        "tasks": [
            {
                "ticker": f"T{i}",
                "candidate_id": f"T{i}:bullish:70",
                "bias": "bullish",
                "score": 70 - i,
            }
            for i in range(40)
        ]
    }
    paths = output_paths("2026-05-22", root=tmp_path)

    dispatch = core.build_agent_dispatch_plan(tasks, "2026-05-22", paths)
    prompt = dispatch["subagent_tasks"][0]["prompt"]

    assert "T0" in prompt
    assert "T30" in prompt
    assert "T39" in prompt
    assert dispatch["common_context"]["required_review_ticker_count"] == 40
    assert dispatch["subagent_tasks"][0]["required_review_tickers"] == [f"T{i}" for i in range(40)]
    assert dispatch["subagent_tasks"][0]["input_task_count"] == 40


def test_agent_dispatch_prioritizes_bounded_required_tickers(tmp_path: Path) -> None:
    tasks = {
        "tasks": [
            {
                "ticker": f"T{i}",
                "candidate_id": f"T{i}:bullish:{i}",
                "bias": "bullish",
                "score": i,
            }
            for i in range(core.MAX_REQUIRED_SUBAGENT_REVIEW_TICKERS + 25)
        ]
        + [
            {
                "ticker": "AAPL",
                "candidate_id": "AAPL:bullish:1",
                "bias": "bullish",
                "score": 1,
            },
            {
                "ticker": "SPY",
                "candidate_id": "SPY:bearish:2",
                "bias": "bearish",
                "score": 2,
            },
        ]
    }
    paths = output_paths("2026-05-22", root=tmp_path)

    dispatch = core.build_agent_dispatch_plan(tasks, "2026-05-22", paths)
    required = dispatch["subagent_tasks"][0]["required_review_tickers"]
    prompt = dispatch["subagent_tasks"][0]["prompt"]

    assert len(required) == core.MAX_REQUIRED_SUBAGENT_REVIEW_TICKERS
    assert {"AAPL", "SPY", f"T{core.MAX_REQUIRED_SUBAGENT_REVIEW_TICKERS + 24}"}.issubset(required)
    assert "Return at least one review for every required review ticker" in prompt
    assert dispatch["subagent_tasks"][0]["tickers"] == required
    assert dispatch["common_context"]["candidate_universe_ticker_count"] == core.MAX_REQUIRED_SUBAGENT_REVIEW_TICKERS + 27


def test_agent_dispatch_prompt_requires_pass_one_exact_contract_reviews(tmp_path: Path) -> None:
    paths = output_paths("2026-05-22", root=tmp_path)
    dispatch = core.build_agent_dispatch_plan(
        {"tasks": [{"ticker": "AAPL", "candidate_id": "AAPL:bullish:75", "bias": "bullish", "score": 75}]},
        "2026-05-22",
        paths,
    )

    prompts = {task["agent"]: task["prompt"] for task in dispatch["subagent_tasks"]}

    for agent in ("structure_builder", "skeptic"):
        prompt = prompts[agent]
        assert "Dispatch-only/pass-1 writes live priced_candidates.csv" in prompt
        assert "If an exact contract task is present, review that contract" in prompt
        assert "A generic ticker review does not satisfy contract review coverage" in prompt
        assert "Return supportive when an exact contract passes the policy" in prompt
        assert "above 30%" in prompt
        assert "theoretical_max_profit is uncapped" in prompt


def test_external_pass_one_artifact_absence_review_is_caution_only(tmp_path: Path) -> None:
    reviews_json = tmp_path / "agentic_reviews.json"
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "GOOG",
                        "agent": "structure_builder",
                        "verdict": "caution",
                        "confidence": "high",
                        "note": (
                            "GOOG is a qualified REVIEW candidate, but "
                            "structure_attempts/priced_candidates/decision_board are empty, so legs and payoff math are absent."
                        ),
                        "objective_blocker": True,
                    },
                    {
                        "ticker": "GOOG",
                        "agent": "skeptic",
                        "verdict": "avoid",
                        "confidence": "medium",
                        "note": (
                            "No priced structure, target debit/credit, or decision-board row is present in the dispatch "
                            "artifacts, so this is not entry-ready until structure math is regenerated."
                        ),
                        "objective_blocker": True,
                    },
                    {
                        "ticker": "TSLA",
                        "agent": "skeptic",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": "objective thesis break from confirmed delisting event",
                        "objective_blocker": True,
                        "blocker_type": "thesis_break",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    reviews, warnings = core.load_external_agent_reviews(reviews_json)

    assert warnings == []
    by_ticker_agent = reviews.set_index(["ticker", "agent"])
    assert bool(by_ticker_agent.loc[("GOOG", "structure_builder"), "objective_blocker"]) is False
    assert bool(by_ticker_agent.loc[("GOOG", "skeptic"), "objective_blocker"]) is False
    assert by_ticker_agent.loc[("GOOG", "skeptic"), "verdict"] == "caution"
    assert "Pass-1 placeholder artifact absence is caution-only" in by_ticker_agent.loc[
        ("GOOG", "skeptic"), "note"
    ]
    assert bool(by_ticker_agent.loc[("TSLA", "skeptic"), "objective_blocker"]) is True


def test_external_agent_loader_rejects_unknown_lane_and_unverified_catalyst_contract(tmp_path: Path) -> None:
    reviews_json = tmp_path / "agentic_reviews.json"
    common = {
        "ticker": "AAPL",
        "verdict": "supportive",
        "confidence": "high",
        "note": "review completed",
        "objective_blocker": False,
    }
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {**common, "agent": "invented_lane"},
                    {
                        **common,
                        "agent": "catalyst_news",
                        "contract_specific": True,
                        "contract_key": "abc",
                        "strategy_route": "bull_call_debit",
                        "expiry": "2026-08-21",
                        "trade_plan": "BUY 1 AAPL 2026-08-21 200 Call @ 5.00 DEBIT",
                        "evidence": "stock screener only",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    reviews, warnings = core.load_external_agent_reviews(reviews_json)

    assert reviews.empty
    assert len(warnings) == 1
    assert "unknown_agent" in warnings[0]
    assert "issuer_source_url_required" in warnings[0]


def test_agentic_pass_reuses_pass_one_dispatch_contract_for_matching_reviews(tmp_path: Path) -> None:
    paths = output_paths("2026-05-22", root=tmp_path)
    paths["out_dir"].mkdir(parents=True)
    prior = core.build_agent_dispatch_plan(
        {"tasks": [{"ticker": "AAPL", "score": 80}, {"ticker": "SPY", "score": 79}]},
        "2026-05-22",
        paths,
    )
    current = core.build_agent_dispatch_plan(
        {"tasks": [{"ticker": "INTC", "score": 90}, {"ticker": "QCOM", "score": 85}]},
        "2026-05-22",
        paths,
    )
    paths["agent_dispatch_plan"].write_text(json.dumps(prior), encoding="utf-8")

    loaded = core._load_existing_agent_dispatch_plan(paths["agent_dispatch_plan"])
    resolved = core._resolve_agent_dispatch_plan_for_reviews(
        current,
        loaded,
        agent_reviews_json=paths["agentic_reviews"],
        expected_reviews_json=paths["agentic_reviews"],
    )
    drift = core._dispatch_plan_drift_summary(resolved, current)

    assert resolved["common_context"]["required_review_tickers"] == ["AAPL", "SPY"]
    assert drift["status"] == "drift"
    assert drift["added_required_ticker_examples"] == ["INTC", "QCOM"]


def test_contract_review_drift_compares_exact_keys_and_required_agents() -> None:
    prior = {
        "contracts": [
            {"contract_key": "AAPL|OLD", "required_review_agents": ["structure_builder", "skeptic"]},
            {"contract_key": "MSFT|KEEP", "required_review_agents": ["structure_builder", "skeptic"]},
            {"contract_key": "NVDA|AGENTS", "required_review_agents": ["structure_builder"]},
        ]
    }
    current = {
        "contracts": [
            {"contract_key": "MSFT|KEEP", "required_review_agents": ["structure_builder", "skeptic"]},
            {"contract_key": "NVDA|AGENTS", "required_review_agents": ["structure_builder", "skeptic"]},
            {"contract_key": "AMZN|NEW", "required_review_agents": ["structure_builder", "skeptic"]},
        ]
    }

    drift = core._contract_review_task_drift_summary(prior, current)

    assert drift["contract_status"] == "drift"
    assert drift["reviewed_contract_count"] == 3
    assert drift["current_contract_count"] == 3
    assert drift["added_contract_examples"] == ["AMZN|NEW"]
    assert drift["removed_contract_examples"] == ["AAPL|OLD"]
    assert drift["changed_required_agent_contract_examples"] == ["NVDA|AGENTS"]


def test_trade_quality_gates_reject_junk_setups() -> None:
    rejects = core._trade_quality_rejects(
        entry_credit=0.05,
        credit_width_ratio=0.05,
        max_loss=2_000,
        signal_premium=500_000,
        combined_flow_bias=0.01,
    )

    assert "entry_credit_below_0.25" in rejects
    assert "credit_width_ratio_below_18pct" in rejects
    assert "signal_premium_below_1000000" in rejects
    assert "directional_bias_below_0.10" in rejects
    assert "one_lot_max_loss_above_750" in rejects


def test_trade_tickets_require_executable_live_validated_entry() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "REVIEW",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "target_exit": 0.35,
                "invalidation": "breaks support",
                "live_validation_status": "PASS",
            },
            {
                "ticker": "WAIT",
                "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "AVOID",
                "recommendation_status": RecommendationStatus.AVOID.value,
                "quality_status": "qualified",
                "hard_rejects": "bad_liquidity",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "BLANK",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "ZERO",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 0.00 CREDIT",
                "entry_limit": 0.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "NOLIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
            },
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "target_exit": 0.35,
                "invalidation": "breaks support",
                "live_validation_status": "PASS",
            },
            {
                "ticker": "RISK",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "live_validation_status": "PASS",
                "portfolio_risk_flag": True,
            },
            {
                "ticker": "NOSIZE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "suggested_contracts": 0,
                "live_validation_status": "PASS",
            },
        ]
    )
    final["underlying_quality_tier"] = "core"
    final = _mark_strategy_expectancy_pass(final, {"BLANK", "ZERO", "NOLIVE", "LIVE", "RISK", "NOSIZE"})

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    tickets = core.build_trade_tickets(decision)

    assert decision.loc[decision["ticker"].eq("NOLIVE"), "execution_status"].tolist() == ["needs_live_validation"]
    assert decision.loc[decision["ticker"].eq("NOSIZE"), "execution_status"].tolist() == ["needs_sizing"]
    assert decision.loc[decision["ticker"].eq("NOSIZE"), "ready_to_enter"].tolist() == [False]
    assert tickets["ticker"].tolist() == ["LIVE", "RISK"]
    assert "REVIEW" not in tickets["ticker"].tolist()
    assert "WAIT" not in tickets["ticker"].tolist()
    assert "AVOID" not in tickets["ticker"].tolist()
    assert "NOLIVE" not in tickets["ticker"].tolist()
    assert "NOSIZE" not in tickets["ticker"].tolist()
    assert tickets.loc[tickets["ticker"].isin(["LIVE", "RISK"]), "live_validation_status"].tolist() == ["PASS", "PASS"]
    assert tickets.loc[tickets["ticker"].eq("LIVE"), "target_exit"].tolist() == [0.35]
    assert tickets.loc[tickets["ticker"].eq("LIVE"), "invalidation"].tolist() == ["breaks support"]


def test_trade_tickets_keep_live_review_only_mechanics_visible() -> None:
    decision = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "GOOG",
                "structure": "long call",
                "status_icon": "YELLOW",
                "status_label": "review",
                "ready_to_enter": False,
                "execution_status": "needs_confidence",
                "execution_gate_status": "blocked",
                "execution_confidence_score": 0.0,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_score": 72.0,
                "order_mechanics_confidence_rating": "MEDIUM",
                "trade_quality_confidence_rating": "MEDIUM",
                "target_order_status": "review_only_expectancy_evidence",
                "execution_blockers": core.PROFITABILITY_CALIBRATION_BLOCKER,
                "suggested_contracts": 3,
                "trade_plan": "BUY 1 GOOG 2026-07-17 365 Call @ 4.10 DEBIT",
                "expiry": "2026-07-17",
                "sell_leg": "",
                "buy_leg": "BUY 1 GOOG 2026-07-17 365 Call",
                "entry_limit": 4.10,
                "max_profit": 328.0,
                "max_loss": 410.0,
                "target_exit": 7.38,
                "live_validation_status": "PASS",
                "status_reason": "live validated; profitability calibration still required",
            }
        ]
    )

    tickets = core.build_trade_tickets(decision)
    green, target = core.split_trade_ticket_surfaces(tickets)

    assert tickets["ticker"].tolist() == ["GOOG"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["order_readiness"].tolist() == ["not_ready_confidence_required"]
    assert tickets["execution_confidence_rating"].tolist() == ["NOT_EXECUTION_READY"]
    assert core._order_mechanics_candidate_frame(tickets)["ticker"].tolist() == ["GOOG"]
    assert green.empty
    assert target.empty


def test_execution_ready_ticket_requires_run_level_gates() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "trade_plan": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 3,
                "external_agent_review_count": 4,
                "external_agent_distinct_review_count": 4,
                "external_agent_review_agents": "catalyst_news; macro_regime; structure_builder; skeptic",
            }
        ]
    )
    final["underlying_quality_tier"] = "core"
    final = _mark_strategy_expectancy_pass(final)
    blocked_context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=Path("/tmp/snapshots"),
        portfolio_context={"status": "unavailable", "total_value": 0},
        research_task_count=1,
        external_review_count=0,
        agent_reviews_json=None,
    )
    blocked_decision = core.synthesize_decision_board(
        final,
        market_regime={"regime": "mixed"},
        execution_context=blocked_context,
    )

    assert blocked_decision["ready_to_enter"].tolist() == [False]
    assert blocked_decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    blocked_tickets = core.build_trade_tickets(blocked_decision)
    assert blocked_tickets["ready_to_enter"].tolist() == [False]
    assert blocked_tickets["target_order_status"].tolist() == ["target_order_candidate"]

    ready_context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    ready_decision = core.synthesize_decision_board(
        final,
        market_regime={"regime": "mixed"},
        execution_context=ready_context,
    )
    tickets = core.build_trade_tickets(ready_decision)
    readiness = core.build_execution_readiness(ready_decision, ready_context)

    assert ready_decision["ready_to_enter"].tolist() == [True]
    assert ready_decision["execution_gate_status"].tolist() == ["pass"]
    assert ready_decision["execution_confidence_rating"].tolist() == ["MEDIUM"]
    assert ready_decision["execution_confidence_score"].iloc[0] >= core.MIN_EXECUTION_CONFIDENCE_SCORE
    assert core.summarize_execution_readiness(readiness)["status"] == "execution_ready"
    assert tickets["ticker"].tolist() == ["LIVE"]
    assert tickets["position_max_profit"].tolist() == [750.0]
    assert tickets["position_max_loss"].tolist() == [1750.0]


def test_strict_negative_strategy_expectancy_blocks_review_rows_from_target_surface() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "BADVERT",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "status_reason": "dated UW target with fresh Schwab chain math",
                "full_ticket": (
                    "SELL 1 BADVERT 2026-06-18 370 Call / "
                    "BUY 1 BADVERT 2026-06-18 372.5 Call @ 0.85 CREDIT"
                ),
                "trade_plan": (
                    "SELL 1 BADVERT 2026-06-18 370 Call / "
                    "BUY 1 BADVERT 2026-06-18 372.5 Call @ 0.85 CREDIT"
                ),
                "entry_limit": 0.85,
                "suggested_contracts": 1,
                "max_profit": 85.0,
                "max_loss": 165.0,
                "credit_width_ratio": 0.34,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 5,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE,
                "actual_forward_strategy_expectancy_avg_pnl": -25.0,
                "actual_forward_strategy_expectancy_win_rate": 0.25,
                "actual_forward_strategy_expectancy_profit_factor": 0.5,
                "actual_forward_strategy_expectancy_family": "vertical_spread",
                "profitability_calibration_status": "WARN",
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    _, target_tickets = core.split_trade_ticket_surfaces(tickets)

    assert decision["target_order_status"].tolist() == ["not_actionable_negative_strategy_expectancy"]
    assert core.NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]
    assert tickets.empty
    assert target_tickets.empty


def test_current_strategy_cohort_ignores_non_actionable_negative_family_rows() -> None:
    decision = pd.DataFrame(
        [
            {
                "ticker": "BADVERT",
                "final_action": RecommendationStatus.ENTER.value,
                "target_order_status": "not_actionable_negative_strategy_expectancy",
                "trade_plan": "BUY 1 BADVERT 2026-07-17 100 Call / SELL 1 BADVERT 2026-07-17 105 Call @ 2.00 DEBIT",
                "ready_to_enter": False,
            },
            {
                "ticker": "GOODPUT",
                "final_action": RecommendationStatus.ENTER.value,
                "target_order_status": "target_order_candidate",
                "trade_plan": "SELL 1 GOODPUT 2026-07-17 90 Put @ 1.10 CREDIT",
                "ready_to_enter": False,
            },
        ]
    )

    families = core._current_ticket_strategy_families(decision, pd.DataFrame())
    by_ticker = core._current_ticket_strategy_families_by_ticker(decision, pd.DataFrame())
    counts = core._current_ticket_count_by_strategy(decision, pd.DataFrame())

    assert families == {"short_put"}
    assert by_ticker == {"GOODPUT": {"short_put"}}
    assert counts == {"short_put": 1}
    assert core._current_ticket_count_for_ticker_strategy(decision, pd.DataFrame(), "BADVERT", "vertical_spread") == 0
    assert core._current_ticket_count_for_ticker_strategy(decision, pd.DataFrame(), "GOODPUT", "short_put") == 1


def test_current_strategy_cohort_uses_review_only_candidates_for_evidence() -> None:
    decision = pd.DataFrame(
        [
            {
                "ticker": "REVIEWCALL",
                "final_action": RecommendationStatus.ENTER.value,
                "target_order_status": "review_only_profitability_calibration",
                "trade_plan": "BUY 1 REVIEWCALL 2026-07-17 100 Call @ 2.00 DEBIT",
                "ready_to_enter": False,
            },
            {
                "ticker": "REVIEWPUT",
                "final_action": RecommendationStatus.ENTER.value,
                "target_order_status": "review_only_expectancy_evidence",
                "trade_plan": "SELL 1 REVIEWPUT 2026-07-17 90 Put @ 1.10 CREDIT",
                "ready_to_enter": False,
            },
            {
                "ticker": "BLOCKED",
                "final_action": RecommendationStatus.ENTER.value,
                "target_order_status": "blocked_objective_reject",
                "trade_plan": "BUY 1 BLOCKED 2026-07-17 100 Call / SELL 1 BLOCKED 2026-07-17 105 Call @ 2.00 DEBIT",
                "ready_to_enter": False,
            },
        ]
    )

    families = core._current_ticket_strategy_families(decision, pd.DataFrame())
    by_ticker = core._current_ticket_strategy_families_by_ticker(decision, pd.DataFrame())
    counts = core._current_ticket_count_by_strategy(decision, pd.DataFrame())

    assert families == {"long_call", "short_put"}
    assert by_ticker == {"REVIEWCALL": {"long_call"}, "REVIEWPUT": {"short_put"}}
    assert counts == {"long_call": 1, "short_put": 1}
    assert core._current_ticket_count_for_ticker_strategy(decision, pd.DataFrame(), "BLOCKED", "vertical_spread") == 0


def test_execution_fill_quality_audit_blocks_entries_worse_than_target() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "GOOD",
                "trade_plan": "SELL 1 GOOD 2026-07-17 100 Put @ 1.25 CREDIT",
                "entry_limit": 1.25,
                "target_entry": 1.10,
                "live_validation_status": "PASS",
                "live_quote_width_pct": 0.20,
                "live_short_oi": 500,
                "live_short_volume": 50,
            },
            {
                "recommendation_rank": 2,
                "ticker": "BAD",
                "trade_plan": "BUY 1 BAD 2026-07-17 100 Call / SELL 1 BAD 2026-07-17 105 Call @ 2.40 DEBIT",
                "entry_limit": 2.40,
                "target_entry": 1.80,
                "live_validation_status": "PASS",
                "live_quote_width_pct": 0.20,
                "live_short_oi": 500,
                "live_short_volume": 50,
                "live_long_oi": 500,
                "live_long_volume": 50,
            },
        ]
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOD",
                "trade_plan": "SELL 1 GOOD 2026-07-17 100 Put @ 1.25 CREDIT",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
            },
            {
                "ticker": "BAD",
                "trade_plan": "BUY 1 BAD 2026-07-17 100 Call / SELL 1 BAD 2026-07-17 105 Call @ 2.40 DEBIT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
            },
        ]
    )

    audit = core.build_execution_fill_quality_audit(final, tickets)
    summary = core.summarize_execution_fill_quality(audit)

    assert list(audit.columns) == core.EXECUTION_FILL_QUALITY_COLUMNS
    good = audit[audit["ticker"].eq("GOOD")].iloc[0]
    bad = audit[audit["ticker"].eq("BAD")].iloc[0]
    assert good["action_surface"] == "yellow_target"
    assert good["fill_quality_status"] == "PASS"
    assert good["price_improvement_vs_target"] == 0.15
    assert bad["action_surface"] == "green_send_now"
    assert bad["fill_quality_status"] == "BLOCK"
    assert bad["slippage_vs_target"] == 0.6
    assert "debit_above_target" in bad["reason"]
    assert summary["status"] == "blocked_green_fill_quality"
    assert summary["green_block_rows"] == 1


def test_green_ticket_requires_strategy_expectancy_annotation() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "NOEXP",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 NOEXP 2026-06-18 100 Put / BUY 1 NOEXP 2026-06-18 95 Put @ 1.60 CREDIT",
                "trade_plan": "SELL 1 NOEXP 2026-06-18 100 Put / BUY 1 NOEXP 2026-06-18 95 Put @ 1.60 CREDIT",
                "entry_limit": 1.6,
                "suggested_contracts": 5,
                "max_profit": 160.0,
                "max_loss": 340.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False]
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]
    assert decision["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    green, target = core.split_trade_ticket_surfaces(tickets)
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    assert green.empty
    assert target.empty


def test_material_position_profit_blocks_green_and_marks_profit_floor_target() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "TOY",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 TOY 2026-06-18 100 Put / BUY 1 TOY 2026-06-18 95 Put @ 1.00 CREDIT",
                "trade_plan": "SELL 1 TOY 2026-06-18 100 Put / BUY 1 TOY 2026-06-18 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "suggested_contracts": 1,
                "max_profit": 100.0,
                "max_loss": 400.0,
                "credit_width_ratio": 0.2,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
            {
                "ticker": "REAL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 REAL 2026-06-18 100 Put / BUY 1 REAL 2026-06-18 95 Put @ 1.60 CREDIT",
                "trade_plan": "SELL 1 REAL 2026-06-18 100 Put / BUY 1 REAL 2026-06-18 95 Put @ 1.60 CREDIT",
                "entry_limit": 1.6,
                "suggested_contracts": 5,
                "max_profit": 160.0,
                "max_loss": 340.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    ready, target = core.split_trade_ticket_surfaces(tickets)

    toy = decision[decision["ticker"].eq("TOY")].iloc[0]
    real = decision[decision["ticker"].eq("REAL")].iloc[0]
    assert bool(toy["ready_to_enter"]) is False
    assert core.POSITION_PROFIT_MATERIALITY_BLOCKER in toy["execution_blockers"]
    assert "send_now_credit_width_below_30pct" in toy["execution_blockers"]
    assert toy["target_order_status"] == "target_order_wait_for_price"
    assert bool(real["ready_to_enter"]) is True
    assert ready["ticker"].tolist() == ["REAL"]
    assert target["ticker"].tolist() == ["TOY"]
    assert target["order_readiness"].tolist() == ["target_order_profit_floor"]
    assert target["action"].tolist() == ["work_target_only_if_profit_floor_clears"]


def test_send_now_requires_strong_credit_and_trade_quality() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "THINCREDIT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 0.65 CREDIT",
                "trade_plan": "SELL 1 X / BUY 1 Y @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "suggested_contracts": 4,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "credit_width_ratio": 0.24,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
            {
                "ticker": "LOWQUALITY",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 X / SELL 1 Y @ 3.30 DEBIT",
                "trade_plan": "BUY 1 X / SELL 1 Y @ 3.30 DEBIT",
                "entry_limit": 3.3,
                "suggested_contracts": 2,
                "max_profit": 670.0,
                "max_loss": 330.0,
                "credit_width_ratio": 0.0,
                "trade_quality_status": "reviewable",
                "quality_gate_reason": "manual_quality_warning",
                "live_validation_status": "PASS",
                "agent_caution_count": 10,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
            {
                "ticker": "NARROWWIDTHGOOD",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "trade_plan": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            },
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False, False, True]
    assert decision["execution_gate_status"].tolist() == ["blocked", "blocked", "pass"]
    assert "send_now_credit_below_0.50" not in decision["execution_blockers"].iloc[0]
    assert "send_now_credit_width_below_30pct" in decision["execution_blockers"].iloc[0]
    assert decision.loc[decision["ticker"].eq("LOWQUALITY"), "target_order_status"].tolist() == [
        "review_only_low_trade_quality"
    ]
    assert decision.loc[decision["ticker"].eq("NARROWWIDTHGOOD"), "target_order_status"].tolist() == [
        "target_order_candidate"
    ]
    assert "NARROWWIDTHGOOD" in tickets[tickets["ready_to_enter"].map(bool)]["ticker"].tolist()
    assert "LOWQUALITY" not in tickets[tickets["ready_to_enter"].map(bool)]["ticker"].tolist()
    assert core._coverage_next_step("REVIEW_TICKET", decision.iloc[0]) == (
        "reprice in Schwab and resolve catalyst/quality review"
    )
    assert core._coverage_next_step("REVIEW_TICKET", decision.iloc[1]) == (
        "reprice in Schwab and resolve trade-quality review"
    )


def test_coverage_next_step_keeps_portfolio_notes_out_of_visible_action_text() -> None:
    plain_review = {"execution_blockers": "", "portfolio_risk_note": "", "requires_portfolio_ack": False}
    portfolio_review = {
        "execution_blockers": "",
        "portfolio_risk_note": "existing option exposure in AAPL",
        "requires_portfolio_ack": False,
    }
    portfolio_context_review = {
        "execution_blockers": "portfolio_context_required",
        "portfolio_risk_note": "",
        "requires_portfolio_ack": False,
    }

    assert "portfolio" not in core._coverage_next_step("REVIEW_TICKET", plain_review)
    assert core._coverage_next_step("REVIEW_TICKET", portfolio_review) == (
        "reprice in Schwab and resolve catalyst/quality review"
    )
    assert core._coverage_next_step("REVIEW_TICKET", portfolio_context_review) == (
        "refresh portfolio context before manual entry"
    )


def test_market_session_gate_respects_us_equity_holidays() -> None:
    holiday_noon = dt.datetime(2026, 5, 25, 10, 0, tzinfo=core.MARKET_TIME_ZONE)
    next_session = core.next_regular_market_session_start(
        dt.datetime(2026, 5, 24, 12, 0, tzinfo=core.MARKET_TIME_ZONE)
    )

    assert core.is_regular_market_day(dt.date(2026, 5, 25)) is False
    assert core.is_regular_market_session_open(holiday_noon) is False
    assert next_session == dt.datetime(2026, 5, 26, 6, 30, tzinfo=core.MARKET_TIME_ZONE)
    assert core.is_regular_market_session_open(dt.datetime(2026, 5, 26, 7, 0, tzinfo=core.MARKET_TIME_ZONE)) is True


def test_portfolio_risk_annotation_does_not_reduce_position_sizing() -> None:
    sized = core.apply_position_sizing(
        [
            {
                "ticker": "RISK",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "live_validation_status": "PASS",
                "max_loss": 200.0,
                "portfolio_risk_flag": True,
                "portfolio_risk_note": "concentration note only",
            }
        ],
        {"status": "ok", "total_value": 100_000.0, "cash": 100_000.0},
        {"sizing_stance": "normal"},
    )

    row = sized[0]
    assert row["suggested_contracts"] == 2
    assert row["risk_budget"] == 500.0
    assert row["max_position_loss"] == 400.0
    assert row["sizing_risk_flag"] is False
    assert "sizing uses the explicit risk budget" in row["sizing_note"]
    assert "portfolio annotation only" not in row["sizing_note"]
    assert row["portfolio_risk_note"] == "concentration note only"


def test_monthly_feasibility_uses_sized_position_max_profit() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GREEN",
                "ready_to_enter": True,
                "target_order_status": "",
                "max_profit": 100.0,
                "max_loss": 250.0,
                "suggested_contracts": 3,
            },
            {
                "ticker": "YELLOW",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "max_profit": 50.0,
                "max_loss": 150.0,
                "suggested_contracts": 4,
            },
        ]
    )

    monthly = core.build_monthly_feasibility(
        decision_board=pd.DataFrame(),
        trade_tickets=tickets,
        execution_context={
            "monthly_profit_target": 10_000.0,
            "fresh_live_quotes_ready": True,
            "portfolio_ready": True,
        },
        expectancy_evidence=pd.DataFrame(),
    )

    assert monthly.loc[monthly["metric"].eq("one_cycle_max_profit"), "value"].tolist() == [300.0]
    assert monthly.loc[monthly["metric"].eq("one_cycle_max_loss"), "value"].tolist() == [750.0]
    assert monthly.loc[monthly["metric"].eq("target_order_candidate_max_profit"), "value"].tolist() == [200.0]
    assert monthly.loc[monthly["metric"].eq("target_order_candidate_max_loss"), "value"].tolist() == [600.0]
    assert monthly.loc[monthly["metric"].eq("ready_ticket_expectancy_evidence"), "status"].tolist() == ["BLOCK"]


def test_monthly_feasibility_requires_expectancy_for_green_ticket_tickers() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GREEN",
                "ready_to_enter": True,
                "target_order_status": "",
                "max_profit": 3000.0,
                "max_loss": 1000.0,
                "suggested_contracts": 3,
            },
            {
                "ticker": "YELLOW",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "max_profit": 500.0,
                "max_loss": 500.0,
                "suggested_contracts": 5,
            },
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "YELLOW",
                "matched_current_count": 1,
                "open_or_unrealized_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "YELLOW",
                "matched_current_count": 1,
                "open_or_unrealized_count": 0,
                "note": "Broad current-ticket support is positive for a non-green ticker.",
            },
        ]
    )

    monthly = core.build_monthly_feasibility(
        decision_board=pd.DataFrame(),
        trade_tickets=tickets,
        execution_context={
            "monthly_profit_target": 10_000.0,
            "fresh_live_quotes_ready": True,
            "portfolio_ready": True,
        },
        expectancy_evidence=expectancy,
    )
    summary = core.summarize_monthly_feasibility(monthly)

    assert monthly.loc[monthly["metric"].eq("expectancy_evidence"), "status"].tolist() == ["PASS"]
    assert monthly.loc[monthly["metric"].eq("ready_ticket_expectancy_evidence"), "status"].tolist() == ["BLOCK"]
    assert "GREEN" in monthly.loc[monthly["metric"].eq("ready_ticket_expectancy_evidence"), "note"].iloc[0]
    assert summary["status"] == "not_proven"
    assert "ready_ticket_expectancy_evidence" in summary["blocking_metrics"]


def test_monthly_feasibility_passes_green_ticket_expectancy_when_ready_tickers_supported() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GREEN",
                "ready_to_enter": True,
                "target_order_status": "",
                "max_profit": 3000.0,
                "max_loss": 1000.0,
                "suggested_contracts": 5,
            },
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "GREEN",
                "matched_current_count": 1,
                "open_or_unrealized_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "GREEN",
                "matched_current_count": 1,
                "open_or_unrealized_count": 0,
                "note": "Actual closed/forward outcomes are positive for current green ticker.",
            },
        ]
    )

    monthly = core.build_monthly_feasibility(
        decision_board=pd.DataFrame(),
        trade_tickets=tickets,
        execution_context={
            "monthly_profit_target": 10_000.0,
            "fresh_live_quotes_ready": True,
            "portfolio_ready": True,
        },
        expectancy_evidence=expectancy,
    )

    assert monthly.loc[monthly["metric"].eq("ready_ticket_expectancy_evidence"), "status"].tolist() == ["PASS"]


def test_send_now_green_requires_positive_structure_aligned_actual_forward_support() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AMAT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 AMAT 2026-05-29 467.5 Call / BUY 1 AMAT 2026-05-29 472.5 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 AMAT 2026-05-29 467.5 Call / BUY 1 AMAT 2026-05-29 472.5 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_expectancy_status": "BLOCK",
                "actual_forward_expectancy_sample_size": 0,
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": 0,
            },
            {
                "ticker": "GOOGL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 400 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 400 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_expectancy_status": "PASS",
                "actual_forward_expectancy_sample_size": 14,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 14,
            },
        ]
    )
    final = _mark_strategy_expectancy_pass(final, tickers=["GOOGL"], sample=14)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision.loc[decision["ticker"].eq("AMAT"), "ready_to_enter"].tolist() == [False]
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in decision.loc[
        decision["ticker"].eq("AMAT"), "execution_blockers"
    ].iloc[0]
    assert decision.loc[decision["ticker"].eq("GOOGL"), "ready_to_enter"].tolist() == [True]
    assert decision.loc[decision["ticker"].eq("AMAT"), "target_order_status"].tolist() == [
        "review_only_expectancy_evidence"
    ]
    green, target = core.split_trade_ticket_surfaces(tickets)
    amat_ticket = tickets.loc[tickets["ticker"].eq("AMAT")]
    assert amat_ticket["ready_to_enter"].tolist() == [False]
    assert amat_ticket["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    assert green.loc[green["ticker"].eq("AMAT")].empty
    assert target.loc[target["ticker"].eq("AMAT")].empty
    assert tickets.loc[tickets["ticker"].eq("GOOGL"), "ready_to_enter"].tolist() == [True]


def test_closed_market_is_informational_and_does_not_block_target_ticket() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 LIVE 2026-06-18 100 Put / BUY 1 LIVE 2026-06-18 95 Put @ 1.00 CREDIT",
                "trade_plan": "SELL 1 LIVE 2026-06-18 100 Put / BUY 1 LIVE 2026-06-18 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.2,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "agent_support_count": 4,
                "external_agent_review_count": 4,
                "external_agent_distinct_review_count": 4,
                "external_agent_review_agents": "catalyst_news; macro_regime; structure_builder; skeptic",
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    closed_context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=4,
        external_review_count=4,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=False,
    )

    decision = core.synthesize_decision_board(
        final,
        market_regime={"regime": "mixed"},
        execution_context=closed_context,
    )
    tickets = core.build_trade_tickets(decision)
    readiness = core.build_execution_readiness(decision, closed_context)

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["execution_status"].tolist() == ["waiting_for_price"]
    assert "market_session_open_required" not in decision["execution_blockers"].iloc[0]
    assert "regular_session_quote_refresh_required" not in decision["execution_blockers"].iloc[0]
    assert "send_now_credit_width_below_30pct" in decision["execution_blockers"].iloc[0]
    assert tickets["target_order_status"].tolist() == ["target_order_wait_for_price"]
    assert tickets["order_readiness"].tolist() == ["target_order_wait_for_price"]
    assert tickets["action"].tolist() == ["work_target_limit_if_price_improves"]
    assert "leave at target limit" in core._ticket_next_step(tickets.iloc[0])
    coverage = core.build_coverage_audit(
        raw_universe=pd.DataFrame(),
        candidates=pd.DataFrame(),
        priced=pd.DataFrame(),
        decision_board=decision,
        no_trade=pd.DataFrame(),
        watchlist=["LIVE"],
    )
    assert coverage["next_step"].tolist() == ["reprice in Schwab and resolve catalyst/quality review"]
    quote_freshness = readiness.loc[readiness["gate"].eq("quote_freshness")]
    assert quote_freshness["status"].tolist() == ["INFO"]
    assert "execution_blocker=false" in quote_freshness["detail"].iloc[0]


def test_closed_market_clean_live_row_can_still_be_green_when_other_gates_pass() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 LIVE 2026-06-18 100 Put / BUY 1 LIVE 2026-06-18 95 Put @ 1.60 CREDIT",
                "trade_plan": "SELL 1 LIVE 2026-06-18 100 Put / BUY 1 LIVE 2026-06-18 95 Put @ 1.60 CREDIT",
                "entry_limit": 1.6,
                "suggested_contracts": 5,
                "max_profit": 160.0,
                "max_loss": 340.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=10,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=False,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [True]
    assert decision["execution_status"].tolist() == ["ready"]
    assert "market_session_open_required" not in decision["execution_blockers"].iloc[0]
    assert decision["execution_blockers"].tolist() == [""]
    assert tickets["ready_to_enter"].tolist() == [True]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["order_readiness"].tolist() == ["ready_to_enter"]


def test_market_open_recheck_queue_includes_only_market_session_only_targets() -> None:
    tickets = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "SESSION",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_market_open_and_live_recheck",
                "execution_blockers": "market_session_open_required",
                "entry_type": "CREDIT",
                "entry_limit": 0.65,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "trade_plan": "SELL 1 SESSION 2026-06-05 100 Call / BUY 1 SESSION 2026-06-05 105 Call @ 0.65 CREDIT",
            },
            {
                "recommendation_rank": 2,
                "ticker": "FRESH",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_live_recheck",
                "execution_blockers": "fresh_live_schwab_required",
                "entry_type": "DEBIT",
                "entry_limit": 0.65,
                "max_profit": 185.0,
                "max_loss": 65.0,
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 4,
                "trade_plan": "BUY 1 FRESH 2026-06-05 100 Call / SELL 1 FRESH 2026-06-05 105 Call @ 0.65 DEBIT",
            },
            {
                "recommendation_rank": 3,
                "ticker": "PORT",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_market_open_and_live_recheck",
                "execution_blockers": "market_session_open_required; portfolio_context_required",
                "entry_type": "CREDIT",
                "entry_limit": 0.65,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "trade_plan": "SELL 1 PORT 2026-06-05 100 Call / BUY 1 PORT 2026-06-05 105 Call @ 0.65 CREDIT",
            },
            {
                "recommendation_rank": 4,
                "ticker": "READY",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "order_readiness": "ready_to_enter",
                "execution_blockers": "",
                "entry_type": "CREDIT",
                "entry_limit": 0.65,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "trade_plan": "SELL 1 READY 2026-06-05 100 Call / BUY 1 READY 2026-06-05 105 Call @ 0.65 CREDIT",
            },
        ]
    )

    queue = core.build_market_open_recheck_queue(tickets)

    assert queue["ticker"].tolist() == ["FRESH", "SESSION"]
    assert queue["required_recheck"].str.contains("fresh Schwab quote").tolist() == [True, True]
    assert queue["recheck_action"].str.contains("ready_to_enter=true").tolist() == [True, True]


def test_market_session_only_targets_are_yellow_until_ready() -> None:
    row = {
        "ready_to_enter": False,
        "target_order_status": "target_order_candidate",
        "execution_blockers": "market_session_open_required",
        "trade_plan": "SELL 1 SESSION 2026-06-05 100 Call / BUY 1 SESSION 2026-06-05 105 Call @ 0.65 CREDIT",
    }

    assert core._decision_badge(row) == "🟡 YELLOW target"
    assert core._decision_icon(row) == "🟡"
    assert core._decision_status_label(row) == "YELLOW target"
    assert core._ticket_order_readiness(row) == "target_order_price_validation"
    assert core._ticket_action(row) == "work_target_limit"
    assert "shown target limit" in core._ticket_next_step(row)


def test_not_actionable_trade_plan_rows_are_red_no_action() -> None:
    for status in [
        "not_actionable_risk_reward",
        "not_actionable_negative_strategy_expectancy",
        "not_actionable_underlying_quality",
    ]:
        row = {
            "ready_to_enter": False,
            "target_order_status": status,
            "execution_status": "needs_confidence",
            "trade_plan": "SELL 1 RISK 2026-06-05 100 Put @ 1.50 CREDIT",
        }

        assert core._decision_badge(row) == "🔴 RED no-action"
        assert core._decision_icon(row) == "🔴"
        assert core._decision_status_label(row) == "RED no-action"


def test_actionability_proof_fails_green_labeled_non_ready_targets() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "SESSION",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "status_icon": "🟢",
                "status_label": "GREEN target",
                "entry_type": "CREDIT",
                "entry_limit": 0.65,
                "trade_plan": "SELL 1 SESSION 2026-06-05 100 Call / BUY 1 SESSION 2026-06-05 105 Call @ 0.65 CREDIT",
                "sell_leg": "SELL 1 SESSION 2026-06-05 100 Call",
                "buy_leg": "BUY 1 SESSION 2026-06-05 105 Call",
            }
        ]
    )
    green_proof = pd.DataFrame(
        [{"green_ticket_rows": 0, "valid_green_ticket_rows": 0, "invalid_green_ticket_rows": 0}]
    )

    packet = audit.build_actionability_surface_proof_packet(
        tickets=tickets,
        green_ticket_execution_proof=green_proof,
        market_open_recheck_queue=pd.DataFrame(),
    )

    assert packet["status"].tolist() == ["FAIL_ACTIONABILITY_SURFACE_INTEGRITY"]
    assert packet["target_green_label_rows"].tolist() == [1]
    assert packet["target_green_icon_rows"].tolist() == [1]


def test_trade_ticket_surfaces_sort_by_confidence() -> None:
    decision = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "LOWER",
                "trade_plan": "SELL 1 LOWER 2026-06-05 100 Call / BUY 1 LOWER 2026-06-05 105 Call @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "ready_to_enter": False,
                "execution_status": "needs_market_session",
                "execution_gate_status": "blocked",
                "execution_blockers": "market_session_open_required",
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "execution_confidence_score": 75,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 95,
                "suggested_contracts": 5,
                "max_profit": 65.0,
                "max_loss": 185.0,
            },
            {
                "recommendation_rank": 2,
                "ticker": "TOP",
                "trade_plan": "SELL 1 TOP 2026-06-05 100 Call / BUY 1 TOP 2026-06-05 105 Call @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "ready_to_enter": False,
                "execution_status": "needs_market_session",
                "execution_gate_status": "blocked",
                "execution_blockers": "market_session_open_required",
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "execution_confidence_score": 88,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 50,
                "suggested_contracts": 1,
                "max_profit": 65.0,
                "max_loss": 185.0,
            },
            {
                "recommendation_rank": 3,
                "ticker": "TIEHIGH",
                "trade_plan": "SELL 1 TIEHIGH 2026-06-05 100 Call / BUY 1 TIEHIGH 2026-06-05 105 Call @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "ready_to_enter": False,
                "execution_status": "needs_market_session",
                "execution_gate_status": "blocked",
                "execution_blockers": "market_session_open_required",
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "execution_confidence_score": 75,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 10,
                "suggested_contracts": 1,
                "max_profit": 65.0,
                "max_loss": 185.0,
            },
        ]
    )

    tickets = core.build_trade_tickets(decision)
    _, target = core.split_trade_ticket_surfaces(tickets)
    queue = core.build_market_open_recheck_queue(tickets)

    assert tickets["ticker"].tolist() == ["TOP", "TIEHIGH", "LOWER"]
    assert target["ticker"].tolist() == ["TOP", "TIEHIGH", "LOWER"]
    assert queue["ticker"].tolist() == ["TOP", "TIEHIGH", "LOWER"]

    mixed_readiness = pd.DataFrame(
        [
            {
                "ticker": "LOW_PRICE_REFRESH",
                "ready_to_enter": False,
                "order_readiness": "target_order_price_validation",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 35,
                "trade_quality_confidence_rating": "LOW",
                "execution_confidence_rating": "LOW",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 90,
                "recommendation_rank": 1,
            },
            {
                "ticker": "HIGH_CONFIDENCE_TARGET",
                "ready_to_enter": False,
                "order_readiness": "target_order_price_validation",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 76,
                "trade_quality_confidence_rating": "LOW",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 4,
                "synthesis_score": 10,
                "recommendation_rank": 2,
            },
        ]
    )
    sorted_mixed = core._sort_trades_by_confidence(mixed_readiness)
    assert sorted_mixed["ticker"].tolist() == ["HIGH_CONFIDENCE_TARGET", "LOW_PRICE_REFRESH"]


def test_target_sort_places_exact_review_pass_before_blocked_high_confidence() -> None:
    rows = pd.DataFrame(
        [
            {
                "ticker": "BLOCKED",
                "ready_to_enter": False,
                "order_readiness": "target_order_after_exact_contract_review",
                "contract_review_status": "BLOCK",
                "trade_quality_confidence_score": 99,
                "trade_quality_confidence_rating": "HIGH",
            },
            {
                "ticker": "REVIEWED",
                "ready_to_enter": False,
                "order_readiness": "target_order_after_exact_contract_review",
                "contract_review_status": "PASS",
                "trade_quality_confidence_score": 70,
                "trade_quality_confidence_rating": "MEDIUM",
            },
        ]
    )

    sorted_rows = core._sort_trades_by_confidence(rows)

    assert sorted_rows["ticker"].tolist() == ["REVIEWED", "BLOCKED"]


def test_ready_trade_tickets_sort_by_confidence_before_expectancy_status() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "PASSLOW",
                "ready_to_enter": True,
                "target_order_status": "ready_to_enter",
                "order_readiness": "ready_to_enter",
                "execution_confidence_score": 70,
                "trade_quality_confidence_rating": "MEDIUM",
                "execution_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 100,
                "recommendation_rank": 1,
                "actual_forward_strategy_expectancy_status": "PASS",
            },
            {
                "ticker": "BLOCKHIGH",
                "ready_to_enter": True,
                "target_order_status": "ready_to_enter",
                "order_readiness": "ready_to_enter",
                "execution_confidence_score": 95,
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 10,
                "recommendation_rank": 2,
                "actual_forward_strategy_expectancy_status": "BLOCK",
            },
        ]
    )

    ready, _ = core.split_trade_ticket_surfaces(tickets)

    assert ready["ticker"].tolist() == ["BLOCKHIGH", "PASSLOW"]


def test_ready_trade_tickets_sort_by_execution_confidence_not_mechanics_score() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "LOWER_EXECUTION",
                "ready_to_enter": True,
                "order_readiness": "ready_to_enter",
                "execution_confidence_score": 78,
                "execution_confidence_rating": "MEDIUM",
                "order_mechanics_confidence_score": 100,
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "synthesis_score": 100,
            },
            {
                "ticker": "HIGHER_EXECUTION",
                "ready_to_enter": True,
                "order_readiness": "ready_to_enter",
                "execution_confidence_score": 84,
                "execution_confidence_rating": "MEDIUM",
                "order_mechanics_confidence_score": 100,
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "synthesis_score": 10,
            },
        ]
    )

    sorted_tickets = core._sort_trades_by_confidence(tickets)

    assert sorted_tickets["ticker"].tolist() == ["HIGHER_EXECUTION", "LOWER_EXECUTION"]


def test_target_trade_tickets_sort_by_trade_edge_before_mechanics() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "LOWER_MECHANICS",
                "ready_to_enter": False,
                "order_readiness": "target_order_price_validation",
                "execution_confidence_score": 0,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_score": 80,
                "order_mechanics_confidence_rating": "MEDIUM",
                "trade_quality_confidence_score": 90,
                "trade_quality_confidence_rating": "HIGH",
                "synthesis_score": 100,
            },
            {
                "ticker": "HIGHER_MECHANICS",
                "ready_to_enter": False,
                "order_readiness": "target_order_price_validation",
                "execution_confidence_score": 0,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_score": 90,
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_score": 70,
                "trade_quality_confidence_rating": "MEDIUM",
                "synthesis_score": 10,
            },
        ]
    )

    sorted_tickets = core._sort_trades_by_confidence(tickets)

    assert sorted_tickets["ticker"].tolist() == ["LOWER_MECHANICS", "HIGHER_MECHANICS"]


def test_trade_tickets_keep_green_before_higher_confidence_yellow() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "YELLOWHIGH",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_price_validation",
                "live_validation_status": "PASS",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 99,
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 100,
                "recommendation_rank": 1,
            },
            {
                "ticker": "GREENLOW",
                "ready_to_enter": True,
                "target_order_status": "ready_to_enter",
                "order_readiness": "ready_to_enter",
                "execution_blockers": "",
                "execution_confidence_score": 70,
                "trade_quality_confidence_rating": "MEDIUM",
                "execution_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 50,
                "recommendation_rank": 2,
            },
        ]
    )

    sorted_tickets = core._sort_trades_by_confidence(tickets)

    assert sorted_tickets["ticker"].tolist() == ["GREENLOW", "YELLOWHIGH"]


def test_trade_tickets_keep_all_targets_before_review_only_rows() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "REVIEW_HIGH",
                "ready_to_enter": False,
                "order_readiness": "not_ready_wait_for_price",
                "trade_quality_confidence_score": 99,
                "trade_quality_confidence_rating": "HIGH",
                "synthesis_score": 100,
            },
            {
                "ticker": "PROFIT_FLOOR_TARGET",
                "ready_to_enter": False,
                "order_readiness": "target_order_profit_floor",
                "trade_quality_confidence_score": 40,
                "trade_quality_confidence_rating": "LOW",
                "synthesis_score": 10,
            },
            {
                "ticker": "CASH_RISK_TARGET",
                "ready_to_enter": False,
                "order_readiness": "target_order_after_cash_risk",
                "trade_quality_confidence_score": 30,
                "trade_quality_confidence_rating": "LOW",
                "synthesis_score": 5,
            },
        ]
    )

    sorted_tickets = core._sort_trades_by_confidence(tickets)

    assert sorted_tickets["ticker"].tolist() == [
        "PROFIT_FLOOR_TARGET",
        "CASH_RISK_TARGET",
        "REVIEW_HIGH",
    ]


def test_final_recommendations_sort_by_calibrated_confidence_before_synthesis_score() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "BLOCK_HIGH_SCORE",
                "recommendation_rank": 1,
                "profitability_calibration_status": "BLOCK",
                "profitability_calibration_actual_status": "BLOCK",
                "profitability_calibration_replay_status": "BLOCK",
                "profitability_calibration_actual_avg_pnl": "",
                "profitability_calibration_actual_profit_factor": "",
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "synthesis_score": 500,
                "score": 500,
                "signal_premium": 500,
            },
            {
                "ticker": "WARN_NEGATIVE_BUCKET",
                "recommendation_rank": 2,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route_bucket",
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_actual_sample_size": 12,
                "profitability_calibration_actual_avg_pnl": -25.0,
                "profitability_calibration_actual_profit_factor": 0.8,
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_replay_sample_size": 40,
                "actual_forward_strategy_expectancy_status": "PASS",
                "synthesis_score": 400,
                "score": 400,
                "signal_premium": 400,
            },
            {
                "ticker": "WARN_ZERO_SIZE_MATERIAL",
                "recommendation_rank": 3,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 36,
                "profitability_calibration_actual_avg_pnl": 45.0,
                "profitability_calibration_actual_profit_factor": 1.4,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_replay_sample_size": 12,
                "actual_forward_strategy_expectancy_status": "PASS",
                "max_profit": 1000,
                "suggested_contracts": 0,
                "synthesis_score": 600,
                "score": 600,
                "signal_premium": 600,
            },
            {
                "ticker": "WARN_AVOID_POSITIVE_BROAD",
                "recommendation_rank": 4,
                "recommendation_status": RecommendationStatus.AVOID.value,
                "trade_quality_status": "rejected",
                "hard_rejects": "objective_blocker",
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 36,
                "profitability_calibration_actual_avg_pnl": 45.0,
                "profitability_calibration_actual_profit_factor": 1.4,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_replay_sample_size": 12,
                "actual_forward_strategy_expectancy_status": "PASS",
                "max_profit": 1000,
                "suggested_contracts": 1,
                "synthesis_score": 900,
                "score": 900,
                "signal_premium": 900,
            },
            {
                "ticker": "WARN_TINY_POSITIVE_BROAD",
                "recommendation_rank": 5,
                "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 36,
                "profitability_calibration_actual_avg_pnl": 45.0,
                "profitability_calibration_actual_profit_factor": 1.4,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_replay_sample_size": 12,
                "actual_forward_strategy_expectancy_status": "PASS",
                "max_profit": 10,
                "suggested_contracts": 1,
                "synthesis_score": 500,
                "score": 500,
                "signal_premium": 500,
            },
            {
                "ticker": "WARN_WAIT_MATERIAL_BROAD",
                "recommendation_rank": 6,
                "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 36,
                "profitability_calibration_actual_avg_pnl": 45.0,
                "profitability_calibration_actual_profit_factor": 1.4,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_replay_sample_size": 12,
                "actual_forward_strategy_expectancy_status": "PASS",
                "max_profit": 1000,
                "suggested_contracts": 1,
                "synthesis_score": 700,
                "score": 700,
                "signal_premium": 700,
            },
            {
                "ticker": "WARN_MATERIAL_POSITIVE_BROAD",
                "recommendation_rank": 7,
                "recommendation_status": RecommendationStatus.ENTER.value,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 36,
                "profitability_calibration_actual_avg_pnl": 45.0,
                "profitability_calibration_actual_profit_factor": 1.4,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_replay_sample_size": 12,
                "actual_forward_strategy_expectancy_status": "PASS",
                "max_profit": 300,
                "suggested_contracts": 1,
                "synthesis_score": 50,
                "score": 50,
                "signal_premium": 50,
            },
            {
                "ticker": "PASS_LOW_SCORE",
                "recommendation_rank": 8,
                "profitability_calibration_status": "PASS",
                "profitability_calibration_scope": "actual_route_bucket",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 40,
                "profitability_calibration_actual_avg_pnl": 10.0,
                "profitability_calibration_actual_profit_factor": 1.2,
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_replay_sample_size": 35,
                "actual_forward_strategy_expectancy_status": "PASS",
                "synthesis_score": 1,
                "score": 1,
                "signal_premium": 1,
            },
        ]
    )

    ranked = core.apply_calibrated_final_ranking(final)

    assert ranked["ticker"].tolist() == [
        "PASS_LOW_SCORE",
        "WARN_MATERIAL_POSITIVE_BROAD",
        "WARN_WAIT_MATERIAL_BROAD",
        "WARN_TINY_POSITIVE_BROAD",
        "WARN_ZERO_SIZE_MATERIAL",
        "WARN_NEGATIVE_BUCKET",
        "WARN_AVOID_POSITIVE_BROAD",
        "BLOCK_HIGH_SCORE",
    ]
    assert ranked["recommendation_rank"].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]


def test_credit_direction_uses_option_legs_when_bias_is_missing() -> None:
    put_row = {
        "bias": "",
        "trade_plan": "SELL 1 SPY 2026-06-30 575 Put / BUY 1 SPY 2026-06-30 570 Put @ 1.20 CREDIT",
    }
    call_row = {
        "bias": "",
        "trade_plan": "SELL 1 SPY 2026-06-30 625 Call / BUY 1 SPY 2026-06-30 630 Call @ 1.20 CREDIT",
    }

    assert core._credit_direction(put_row) == "Bull Put"
    assert core._credit_direction(call_row) == "Bear Call"


def test_target_trade_tickets_sort_by_confidence() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "TINYHIGH",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_price_validation",
                "live_validation_status": "PASS",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 96,
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 100,
                "recommendation_rank": 1,
            },
            {
                "ticker": "CLEANLOW",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_price_validation",
                "live_validation_status": "PASS",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 78,
                "trade_quality_confidence_rating": "MEDIUM",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 50,
                "recommendation_rank": 2,
            },
        ]
    )

    _, target = core.split_trade_ticket_surfaces(tickets)

    assert target["ticker"].tolist() == ["TINYHIGH", "CLEANLOW"]


def test_trade_ticket_sort_prefers_confidence_before_exact_calibration() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "ROUTEHIGH",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_price_validation",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 99,
                "trade_quality_confidence_rating": "HIGH",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 100,
                "recommendation_rank": 1,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_scope": "actual_route",
            },
            {
                "ticker": "BUCKETMID",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_price_validation",
                "execution_blockers": "fresh_live_schwab_required",
                "execution_confidence_score": 75,
                "trade_quality_confidence_rating": "MEDIUM",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "external_agent_distinct_review_count": 5,
                "synthesis_score": 20,
                "recommendation_rank": 2,
                "profitability_calibration_status": "PASS",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_scope": "actual_route_bucket",
            },
        ]
    )

    sorted_tickets = core._sort_trades_by_confidence(tickets)

    assert sorted_tickets["ticker"].tolist() == ["ROUTEHIGH", "BUCKETMID"]


def test_market_closed_live_recheck_stays_off_trade_surface() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "trade_quality_status": "reviewable",
                "full_ticket": "SELL 1 AAPL 2026-06-18 200 Put / BUY 1 AAPL 2026-06-18 195 Put @ 1.50 CREDIT",
                "trade_plan": "SELL 1 AAPL 2026-06-18 200 Put / BUY 1 AAPL 2026-06-18 195 Put @ 1.50 CREDIT",
                "expiry": "2026-06-18",
                "sell_leg": "SELL 1 AAPL 2026-06-18 200 Put",
                "buy_leg": "BUY 1 AAPL 2026-06-18 195 Put",
                "entry_limit": 1.5,
                "target_exit": 0.52,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "suggested_contracts": 2,
                "external_agent_distinct_review_count": 4,
                "external_agent_review_count": 4,
                "external_agent_review_agents": "catalyst_news; macro_regime; structure_builder; skeptic",
                "agent_support_count": 5,
                "agent_caution_count": 0,
                "agent_objective_blocker_count": 0,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock",
                "live_validation_status": "MARKET_CLOSED_RECHECK",
                "status_reason": "dated UW EOD quote; refresh Schwab chain before entry",
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=1,
        external_review_count=4,
        external_review_agent_count=4,
        agent_dispatch_task_count=4,
        agent_reviews_json=Path("/tmp/agentic_reviews.json"),
        market_session_open=False,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    queue = core.build_market_open_recheck_queue(tickets)

    assert decision["target_order_status"].tolist() == ["review_only_live_validation"]
    assert tickets.empty
    assert queue.empty


def test_market_open_recheck_proof_blocks_incomplete_rows() -> None:
    queue = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "validation_lane": "live_readiness_probe",
                "source_kind": "live_probe",
                "source_dir": "/tmp/live",
                "ticker": "MISS",
                "entry_type": "CREDIT",
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_market_open_and_live_recheck",
                "entry_limit": 0.0,
                "target_exit": 0.1,
                "max_profit": 50,
                "max_loss": 200,
                "suggested_contracts": 0,
                "execution_confidence_score": 60,
                "trade_quality_confidence_rating": "LOW",
                "external_agent_distinct_review_count": 1,
                "trade_plan": "SELL 1 MISS 2026-06-05 100 Call",
                "required_recheck": "regular_market_session_open",
                "recheck_action": "rerun",
                "execution_blockers": "market_session_open_required; portfolio_context_required",
            }
        ],
        columns=audit.MARKET_QUEUE_AUDIT_COLUMNS,
    )

    details = audit.build_market_open_recheck_details(queue)
    packet = audit.build_market_open_recheck_proof_packet(details)

    assert packet["status"].tolist() == ["FAIL_MARKET_OPEN_RECHECK_ROWS_INCOMPLETE"]
    assert packet["row_fail_rows"].tolist() == [1]
    assert "entry_limit_not_positive" in details["fail_reasons"].iloc[0]
    assert "blockers_not_only_market_session_or_fresh_live_required" in details["fail_reasons"].iloc[0]


def test_session_only_green_shadow_proof_uses_position_scaled_fallbacks() -> None:
    details = pd.DataFrame(
        [
            {
                "source_kind": "live_probe",
                "ticker": "ALPHA",
                "entry_type": "CREDIT",
                "max_profit": 100.0,
                "max_loss": 300.0,
                "position_max_profit": pd.NA,
                "position_max_loss": pd.NA,
                "suggested_contracts": 2,
                "only_market_session_blocker": True,
                "row_pass": True,
                "fail_reasons": "",
            },
            {
                "source_kind": "live_probe",
                "ticker": "BETA",
                "entry_type": "DEBIT",
                "max_profit": 400.0,
                "max_loss": 700.0,
                "position_max_profit": 750.0,
                "position_max_loss": 500.0,
                "suggested_contracts": 1,
                "only_market_session_blocker": True,
                "row_pass": True,
                "fail_reasons": "",
            },
        ]
    )

    packet = audit.build_session_only_green_shadow_proof_packet(details)

    assert packet["status"].tolist() == ["PASS_SESSION_ONLY_GREEN_SHADOW_READY"]
    assert packet["shadow_candidate_rows"].tolist() == [2]
    assert packet["row_fail_rows"].tolist() == [0]
    assert packet["non_session_blocker_rows"].tolist() == [0]
    assert packet["credit_rows"].tolist() == [1]
    assert packet["debit_rows"].tolist() == [1]
    assert packet["position_max_profit"].tolist() == [950.0]
    assert packet["position_max_loss"].tolist() == [1100.0]
    assert packet["tickers"].tolist() == ["ALPHA, BETA"]
    assert "not execution permission" in packet["note"].iloc[0]


def test_live_rerun_preflight_blocks_mismatched_agent_reviews(tmp_path: Path) -> None:
    day = "2026-05-22"
    day_dir = tmp_path / day
    day_dir.mkdir()
    (day_dir / f"stock-screener-{day}.csv").write_text("ticker\nAAPL\n", encoding="utf-8")
    (day_dir / f"hot-chains-{day}.csv").write_text("option_symbol\nAAPL260605C00100000\n", encoding="utf-8")
    (day_dir / f"chain-oi-changes-{day}.csv").write_text("option_symbol\nAAPL260605C00100000\n", encoding="utf-8")

    reviews_json = tmp_path / "wrong_agentic_reviews.json"
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": ticker,
                        "agent": agent,
                        "verdict": "supportive",
                        "confidence": "high",
                        "note": "fixture review for wrong ticker",
                        "objective_blocker": False,
                    }
                    for ticker in ("X", "Y")
                    for agent in ("catalyst_news", "macro_regime", "structure_builder", "skeptic")
                ]
            }
        ),
        encoding="utf-8",
    )
    execution_packet = pd.DataFrame(
        [
            {
                "date": day,
                "out_dir": str(tmp_path / "fresh_live_rerun"),
                "agent_reviews_json": str(reviews_json),
                "command": (
                    "python3 -m uwos.options_agent "
                    f"--date {day} "
                    f"--base-dir {tmp_path} "
                    f"--out-dir {tmp_path / 'fresh_live_rerun'} "
                    "--live-schwab --live-portfolio "
                    f"--agent-reviews-json {reviews_json}"
                ),
            }
        ]
    )
    recheck_details = pd.DataFrame(
        [
            {"ticker": "A", "row_pass": True},
            {"ticker": "B", "row_pass": True},
        ]
    )

    preflight_details = audit.build_live_rerun_preflight_details(
        market_open_recheck_details=recheck_details,
        market_open_execution_packet=execution_packet,
    )
    preflight_packet = audit.build_live_rerun_preflight_proof_packet(
        base_dir=tmp_path,
        market_open_recheck_details=recheck_details,
        market_open_execution_packet=execution_packet,
        preflight_details=preflight_details,
    )

    assert preflight_packet["status"].tolist() == ["FAIL_LIVE_RERUN_PREFLIGHT"]
    assert preflight_packet["queue_tickers"].tolist() == ["A, B"]
    assert preflight_packet["covered_queue_ticker_count"].tolist() == [0]
    assert preflight_packet["missing_queue_tickers"].tolist() == ["A, B"]
    assert preflight_packet["agent_review_rows"].tolist() == [8]
    assert preflight_packet["distinct_agent_count"].tolist() == [4]
    assert preflight_packet["rerun_out_dir_clear"].tolist() == [True]
    assert preflight_packet["source_date_available"].tolist() == [True]
    assert "agent_reviews_json_missing_queue_tickers" in preflight_packet["failed_examples"].iloc[0]
    assert preflight_details["row_pass"].tolist() == [False, False]
    assert preflight_details["fail_reasons"].str.contains("ticker_missing_from_agent_reviews_json").all()


def test_target_preservation_counts_live_queue_debit_targets() -> None:
    target_audit = audit.build_target_preservation_audit(
        summary=pd.DataFrame([{"date": "2026-05-21"}, {"date": "2026-05-22"}]),
        tickets=pd.DataFrame(
            [
                {
                    "ticker": "CREDIT",
                    "entry_type": "CREDIT",
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                }
            ]
        ),
        market_open_recheck_queue=pd.DataFrame(
            [
                {
                    "ticker": "DEBIT",
                    "entry_type": "DEBIT",
                    "source_kind": "live_probe",
                }
            ]
        ),
    )

    assert target_audit.loc[target_audit["metric"].eq("credit_target_rows"), "status"].tolist() == ["PROVEN"]
    assert target_audit.loc[target_audit["metric"].eq("debit_target_rows"), "status"].tolist() == ["PROVEN"]
    assert target_audit.loc[target_audit["metric"].eq("debit_target_rows"), "value"].tolist() == [1]


def test_audit_markdown_tables_blank_nan_values(tmp_path: Path) -> None:
    path = tmp_path / "table.md"

    audit._write_markdown_table(
        path,
        "Fixture Table",
        pd.DataFrame([{"ticker": "NVDA", "rank": pd.NA, "score": float("nan")}]),
    )

    text = path.read_text(encoding="utf-8")
    assert "NVDA" in text
    assert "nan" not in text.lower()


def test_expanded_audit_writes_repeatable_goal_and_live_recheck_artifacts(tmp_path: Path) -> None:
    def write_run(run_dir: Path, day: str, entry_type: str, ticker: str, tier: str = "core", queue_rows=None) -> None:
        run_dir.mkdir(parents=True)
        review_tickers = [ticker]
        if queue_rows:
            review_tickers.extend(str(row.get("ticker", "")).upper() for row in queue_rows)
        review_tickers = sorted({symbol for symbol in review_tickers if symbol})
        review_agents = ["catalyst_news", "macro_regime", "structure_builder", "skeptic"]
        review_rows = [
            {
                "ticker": review_ticker,
                "agent": agent,
                "verdict": "supportive",
                "confidence": "high",
                "note": f"{agent} supports {review_ticker} in fixture",
                "objective_blocker": False,
            }
            for review_ticker in review_tickers
            for agent in review_agents
        ]
        manifest = {
            "as_of": day,
            "pipeline_version": "options_agent.test",
            "live_schwab_requested": run_dir.name.startswith("live_readiness_probe"),
            "chain_snapshot_dir": "" if run_dir.name.startswith("live_readiness_probe") else str(run_dir / "snapshots"),
            "row_counts": {
                "candidate_generation": 2,
                "research_tasks": 2,
                "priced_candidates": 1,
                "final_recommendations": 1,
                "decision_board": 1,
                "trade_tickets": 1,
                "target_order_candidates": 1,
                "no_trade_audit": 1,
                "ready_to_enter": 0,
                "market_open_recheck_queue": len(queue_rows or []),
                "agent_dispatch_tasks": 5,
                "external_agent_reviews": 50,
            },
            "execution_readiness_summary": {
                "status": "not_execution_ready",
                "blocking_gates": ["market_session_open", "ready_trade_tickets"],
            },
            "expectancy_evidence_summary": {
                "status": "not_proven",
                "summary_status": "BLOCK",
                "sample_size": 0,
                "note": "No sufficient positive expectancy evidence is available.",
            },
            "monthly_feasibility_summary": {
                "status": "not_proven",
                "blocking_metrics": ["ready_ticket_count", "expectancy_evidence"],
            },
            "agentic_orchestration": {
                "status": "reviews_ingested",
                "subagent_task_count": 5,
                "ingested_reviews_json": str(run_dir / "agentic_reviews.json"),
            },
            "execution_context": {
                "external_review_count": 50,
                "external_review_agent_count": 5,
                "agent_dispatch_task_count": 5,
                "agentic_review_coverage_basis": "subagent_lanes",
                "agentic_review_coverage_pct": 1.0,
                "agentic_review_lane_coverage_pct": 1.0,
                "broad_review_coverage_pct": 0.0129,
                "fresh_live_quotes_ready": run_dir.name.startswith("live_readiness_probe"),
                "portfolio_ready": run_dir.name.startswith("live_readiness_probe"),
                "market_session_open": False,
            },
        }
        (run_dir / f"options_agent_manifest_{day}.json").write_text(json.dumps(manifest), encoding="utf-8")
        (run_dir / "agentic_reviews.json").write_text(json.dumps({"reviews": review_rows}), encoding="utf-8")
        pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "bias": "bearish",
                    "quality_status": "qualified",
                    "score": 80,
                    "flow_reason": "qualified mock row",
                },
                {
                    "ticker": f"{ticker}W",
                    "bias": "bearish",
                    "quality_status": "watch",
                    "score": 50,
                    "flow_reason": "watch mock row",
                },
            ]
        ).to_csv(run_dir / "candidate_generation.csv", index=False)
        (run_dir / "research_tasks.json").write_text(
            json.dumps({"tasks": [{"ticker": ticker}, {"ticker": f"{ticker}W"}]}),
            encoding="utf-8",
        )
        pd.DataFrame([{"ticker": ticker, "quality_status": "qualified"}]).to_csv(
            run_dir / "priced_candidates.csv",
            index=False,
        )
        pd.DataFrame([{"ticker": ticker, "recommendation_status": "ENTER"}]).to_csv(
            run_dir / "final_recommendations.csv",
            index=False,
        )
        pd.DataFrame(
            [
                {
                    "ticker": f"{ticker}W",
                    "bias": "bearish",
                    "score": 50,
                    "reason": "watch mock row",
                    "hard_blocker": "insufficient_score_or_neutral_bias",
                }
            ]
        ).to_csv(run_dir / "no_trade_audit.csv", index=False)
        pd.DataFrame(
            [
                {
                    "recommendation_rank": 1,
                    "ticker": ticker,
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                    "order_readiness": "target_order_after_market_open_and_live_recheck",
                    "entry_type": entry_type,
                    "entry_limit": 0.5,
                    "max_profit": 50.0,
                    "max_loss": 200.0,
                    "underlying_quality_tier": tier,
                        "external_agent_review_count": 4,
                        "external_agent_distinct_review_count": 4,
                        "external_agent_review_agents": "catalyst; market_regime; skeptic; structure",
                        "trade_plan": f"SELL 1 {ticker} 2026-06-05 100 Call / BUY 1 {ticker} 2026-06-05 105 Call @ 0.50 {entry_type}",
                        "sell_leg": f"SELL 1 {ticker} 2026-06-05 100 Call",
                        "buy_leg": f"BUY 1 {ticker} 2026-06-05 105 Call",
                        "execution_blockers": "market_session_open_required",
                    }
                ]
        ).to_csv(run_dir / "trade_tickets.csv", index=False)
        coverage_rows = [
            {
                "ticker": focus_ticker,
                "coverage_status": "NO_DIRECTIONAL_EDGE",
                "status_color": "gray",
                "underlying_quality_tier": "core",
                "reason": f"{focus_ticker} has no directional edge in this mock run",
            }
            for focus_ticker in core.CORE_AUDIT_TICKERS
        ]
        overrides = {
            "AAPL": {
                "coverage_status": "REVIEW_TICKET",
                "status_color": "yellow",
                "underlying_quality_tier": "core",
                "reason": "needs live repricing",
            },
            "URA": {
                "coverage_status": "CANDIDATE_NOT_STRUCTURED",
                "status_color": "yellow",
                "underlying_quality_tier": "excluded",
                "reason": "excluded underlying",
            },
            "OKLO": {
                "coverage_status": "CANDIDATE_NOT_STRUCTURED",
                "status_color": "yellow",
                "underlying_quality_tier": "speculative",
                "reason": "speculative underlying",
            },
            "DVN": {
                "coverage_status": "NON_ACTIONABLE_UNDERLYING",
                "status_color": "red",
                "underlying_quality_tier": "liquid",
                "reason": "liquid non-core audit row",
            },
        }
        for row in coverage_rows:
            row.update(overrides.get(row["ticker"], {}))
        pd.DataFrame(coverage_rows).to_csv(run_dir / "ticker_coverage_audit.csv", index=False)
        pd.DataFrame(
            [
                {"metric": "one_cycle_max_profit", "value": 0, "status": "BLOCK", "note": ""},
                {"metric": "target_order_candidate_max_profit", "value": 50, "status": "INFO", "note": ""},
            ]
        ).to_csv(run_dir / "monthly_feasibility.csv", index=False)
        pd.DataFrame(
            [
                {
                    "source": "expectancy_summary",
                    "source_path": "",
                    "evidence_type": "summary",
                    "status": "BLOCK",
                    "sample_size": 0,
                    "win_rate": "",
                    "avg_pnl": "",
                    "total_pnl": "",
                    "profit_factor": "",
                    "max_drawdown": "",
                    "matched_current_tickers": ticker,
                    "matched_current_count": 1,
                    "open_or_unrealized_count": 0,
                    "note": "No sufficient positive expectancy evidence is available.",
                }
            ]
        ).to_csv(run_dir / "expectancy_evidence.csv", index=False)
        if queue_rows is None:
            (run_dir / "market_open_recheck_queue.csv").write_text("", encoding="utf-8")
        else:
            pd.DataFrame(queue_rows, columns=core.MARKET_OPEN_RECHECK_COLUMNS).to_csv(
                run_dir / "market_open_recheck_queue.csv",
                index=False,
            )
        pd.DataFrame(
            [
                {
                    "recommendation_rank": 1,
                    "ticker": ticker,
                    "live_market_quality_status": "PASS",
                    "actionability_impact": "eligible_for_yellow_or_green_surface",
                    "recommendation_status": "WAIT_FOR_PRICE",
                    "live_validation_status": "PASS",
                    "structure": "bull put spread",
                    "entry_type": entry_type,
                    "entry_limit": 0.5,
                    "target_entry": 0.4,
                    "spot_live": 100,
                    "short_strike": 100,
                    "long_strike": 105,
                    "spread_width": 5,
                    "live_quote_width_pct": 0.12,
                    "live_leg_min_liquidity": 450,
                    "live_leg_liquidity_status": "PASS",
                    "quality_gate_reason": "",
                    "trade_plan": f"SELL 1 {ticker} 2026-06-05 100 Call / BUY 1 {ticker} 2026-06-05 105 Call @ 0.50 {entry_type}",
                },
                {
                    "recommendation_rank": 2,
                    "ticker": f"{ticker}BAD",
                    "live_market_quality_status": "BLOCK",
                    "actionability_impact": "blocked_not_target_candidate",
                    "recommendation_status": "AVOID",
                    "live_validation_status": "WAIT_FOR_PRICE",
                    "structure": "bull put spread",
                    "entry_type": entry_type,
                    "entry_limit": 0.5,
                    "target_entry": 0.4,
                    "spot_live": 100,
                    "short_strike": 100,
                    "long_strike": 105,
                    "spread_width": 5,
                    "live_quote_width_pct": 0.55,
                    "live_leg_min_liquidity": 35,
                    "live_leg_liquidity_status": "BLOCK",
                    "quality_gate_reason": "live_quote_width_pct_above_30pct; live_leg_liquidity_below_100",
                    "trade_plan": f"SELL 1 {ticker} 2026-06-05 100 Call / BUY 1 {ticker} 2026-06-05 105 Call @ 0.50 {entry_type}",
                },
            ],
            columns=core.LIVE_SPREAD_QUALITY_AUDIT_COLUMNS,
        ).to_csv(run_dir / "live_spread_quality_audit.csv", index=False)

    for day in ("2026-05-20", "2026-05-21", "2026-05-22"):
        source_dir = tmp_path / day
        source_dir.mkdir()
        (source_dir / f"stock-screener-{day}.csv").write_text("ticker\nAAPL\n", encoding="utf-8")
        (source_dir / f"hot-chains-{day}.csv").write_text("option_symbol\nAAPL260605C00100000\n", encoding="utf-8")
        (source_dir / f"chain-oi-changes-{day}.csv").write_text("option_symbol\nAAPL260605C00100000\n", encoding="utf-8")

    run1 = tmp_path / "out" / "options_agent" / "multidate_quality_v017_2026-05-21"
    run2 = tmp_path / "out" / "options_agent" / "multidate_quality_v017_2026-05-22"
    live = tmp_path / "out" / "options_agent" / "live_readiness_probe_v017_2026-05-22"
    write_run(run1, "2026-05-21", "CREDIT", "AAPL", tier="core")
    write_run(run2, "2026-05-22", "DEBIT", "GOOGL", queue_rows=[])
    write_run(
        live,
        "2026-05-22",
        "CREDIT",
        "LIVEQ",
        queue_rows=[
            {
                "recommendation_rank": 1,
                "ticker": "LIVEQ",
                "entry_type": "CREDIT",
                "order_readiness": "target_order_after_market_open_and_live_recheck",
                "target_order_status": "target_order_candidate",
                "entry_limit": 0.65,
                "target_exit": 0.23,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "position_max_profit": 260.0,
                "position_max_loss": 740.0,
                "suggested_contracts": 4,
                "execution_confidence_score": 86,
                "trade_quality_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 4,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "fixture core queue row",
                "trade_plan": "SELL 1 LIVEQ 2026-06-05 100 Call / BUY 1 LIVEQ 2026-06-05 105 Call @ 0.65 CREDIT",
                "required_recheck": "regular_market_session_open + fresh Schwab chain",
                "recheck_action": "rerun Options Agent during regular market hours",
                "execution_blockers": "market_session_open_required",
            }
        ],
    )

    artifacts = audit.write_expanded_audit(
        base_dir=tmp_path,
        run_dirs=[run1, run2],
        live_probe_dirs=[live],
        rerun_agent_reviews_json=live / "agentic_reviews.json",
        output_prefix=tmp_path / "out" / "options_agent" / "expanded_test",
    )

    market_queue = pd.read_csv(artifacts.paths["market_open_recheck_queue"])
    market_open_recheck_details = pd.read_csv(artifacts.paths["market_open_recheck_details"])
    market_open_recheck_packet = pd.read_csv(artifacts.paths["market_open_recheck_proof_packet"])
    live_rerun_preflight_details = pd.read_csv(artifacts.paths["live_rerun_preflight_details"])
    live_rerun_preflight_packet = pd.read_csv(artifacts.paths["live_rerun_preflight_proof_packet"])
    execution_packet = pd.read_csv(artifacts.paths["market_open_execution_packet"])
    multi_date_packet = pd.read_csv(artifacts.paths["multi_date_readiness_proof_packet"])
    verification_plan = pd.read_csv(artifacts.paths["market_session_verification_plan"])
    post_rerun_packet = pd.read_csv(artifacts.paths["post_rerun_verification_packet"])
    green_proof = pd.read_csv(artifacts.paths["green_ticket_execution_proof_packet"])
    session_shadow = pd.read_csv(artifacts.paths["session_only_green_shadow_proof_packet"])
    actionability_packet = pd.read_csv(artifacts.paths["actionability_surface_proof_packet"])
    action_surface_quality_packet = pd.read_csv(artifacts.paths["action_surface_underlying_quality_proof_packet"])
    expectancy_packet = pd.read_csv(artifacts.paths["expectancy_proof_packet"])
    ticket_expectancy_packet = pd.read_csv(artifacts.paths["ticket_expectancy_proof_packet"])
    monthly_guardrail_packet = pd.read_csv(artifacts.paths["monthly_feasibility_guardrail_proof_packet"])
    agentic_packet = pd.read_csv(artifacts.paths["agentic_coverage_proof_packet"])
    validation_packet = pd.read_csv(artifacts.paths["validation_coverage_proof_packet"])
    cutoff_packet = pd.read_csv(artifacts.paths["cutoff_visibility_proof_packet"])
    live_spread_quality = pd.read_csv(artifacts.paths["live_spread_quality_audit"])
    live_spread_quality_packet = pd.read_csv(artifacts.paths["live_spread_quality_proof_packet"])
    quality_packet = pd.read_csv(artifacts.paths["underlying_quality_proof_packet"])
    major_packet = pd.read_csv(artifacts.paths["major_name_coverage_proof_packet"])
    completion_verdict = pd.read_csv(artifacts.paths["completion_verdict"])
    readiness_dashboard = pd.read_csv(artifacts.paths["readiness_dashboard"])
    target_audit = pd.read_csv(artifacts.paths["target_preservation_audit"])
    goal_audit = pd.read_csv(artifacts.paths["goal_completion_audit"])

    assert market_queue["ticker"].tolist() == ["LIVEQ"]
    assert list(market_queue.columns) == audit.MARKET_QUEUE_AUDIT_COLUMNS
    assert list(market_open_recheck_details.columns) == audit.MARKET_OPEN_RECHECK_DETAIL_COLUMNS
    assert market_open_recheck_packet["status"].tolist() == ["PASS_LIVE_MARKET_OPEN_RECHECK_QUEUE_READY"]
    assert market_open_recheck_packet["queue_rows"].tolist() == [1]
    assert market_open_recheck_packet["row_fail_rows"].tolist() == [0]
    assert market_open_recheck_packet["only_market_session_blocker_rows"].tolist() == [1]
    assert market_open_recheck_packet["positive_entry_rows"].tolist() == [1]
    assert market_open_recheck_packet["plain_language_leg_rows"].tolist() == [1]
    assert market_open_recheck_packet["tickers"].tolist() == ["LIVEQ"]
    assert list(live_rerun_preflight_details.columns) == audit.LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS
    assert list(live_rerun_preflight_packet.columns) == audit.LIVE_RERUN_PREFLIGHT_PROOF_COLUMNS
    assert live_rerun_preflight_packet["status"].tolist() == ["PASS_LIVE_RERUN_PREFLIGHT_READY"]
    assert live_rerun_preflight_packet["queue_ticker_count"].tolist() == [1]
    assert live_rerun_preflight_packet["covered_queue_ticker_count"].tolist() == [1]
    assert live_rerun_preflight_packet["missing_queue_ticker_count"].tolist() == [0]
    assert live_rerun_preflight_packet["agent_review_rows"].tolist() == [4]
    assert live_rerun_preflight_packet["distinct_agent_count"].tolist() == [4]
    assert live_rerun_preflight_packet["rerun_out_dir_clear"].tolist() == [True]
    assert live_rerun_preflight_packet["source_date_available"].tolist() == [True]
    preflight_md = artifacts.paths["live_rerun_preflight_proof_packet_md"].read_text(encoding="utf-8")
    assert "Live Rerun Preflight Proof Packet" in preflight_md
    assert "queue-ticker agent review prerequisites" in preflight_md
    recheck_md = artifacts.paths["market_open_recheck_proof_packet_md"].read_text(encoding="utf-8")
    assert "Market-Open Recheck Proof Packet" in recheck_md
    assert "regular-session/fresh-live recheck gate" in recheck_md
    assert execution_packet["status"].tolist() == ["ready_for_regular_session_rerun"]
    assert execution_packet["yellow_recheck_rows"].tolist() == [1]
    assert "next_regular_session_start" in execution_packet.columns
    assert "Full-day U.S. equity market holidays" in execution_packet["market_calendar_note"].iloc[0]
    assert "--live-schwab --live-portfolio" in execution_packet["command"].iloc[0]
    assert str(live / "agentic_reviews.json") in execution_packet["command"].iloc[0]
    planned_rerun = live.parent / "live_readiness_probe_v018_market_open_rerun_2026-05-22"
    assert execution_packet["out_dir"].tolist() == [str(planned_rerun)]
    assert str(planned_rerun) in execution_packet["command"].iloc[0]
    packet_md = artifacts.paths["market_open_execution_packet_md"].read_text(encoding="utf-8")
    assert "Do not enter rows from the yellow queue" in packet_md
    assert "LIVEQ" in packet_md
    assert "position_max_profit" in packet_md
    assert re.search(
        r"\|\s*2026-05-22\s*\|\s*LIVEQ\s*\|\s*CREDIT\s*\|\s*4\s*\|\s*0\.65\s*\|\s*0\.23\s*\|\s*260(?:\.0)?\s*\|\s*740(?:\.0)?\s*\|",
        packet_md,
    )
    assert multi_date_packet["status"].tolist() == ["PASS_MULTI_DATE_TARGETS_WAITING_FOR_REGULAR_SESSION_LIVE_GREEN"]
    assert multi_date_packet["validation_date_count"].tolist() == [2]
    assert multi_date_packet["latest_live_probe_date"].tolist() == ["2026-05-22"]
    assert multi_date_packet["live_probe_dates"].tolist() == ["2026-05-22"]
    assert multi_date_packet["dated_yellow_target_candidates"].tolist() == [2]
    assert multi_date_packet["live_yellow_recheck_rows"].tolist() == [1]
    multi_date_md = artifacts.paths["multi_date_readiness_proof_packet_md"].read_text(encoding="utf-8")
    assert "Multi-date validation is separate from the latest live-session probe" in multi_date_md
    assert "2026-05-21, 2026-05-22" in multi_date_md
    assert verification_plan["status"].tolist() == ["WAITING_FOR_REGULAR_SESSION"]
    assert verification_plan["rerun_out_dir"].tolist() == [str(planned_rerun)]
    assert verification_plan["green_ticket_file"].tolist() == [str(planned_rerun / "green_trade_tickets.csv")]
    assert "ready_to_enter=true" in verification_plan["pass_criteria"].iloc[0]
    assert "completion verdict" in verification_plan["completion_gate"].iloc[0]
    verification_md = artifacts.paths["market_session_verification_plan_md"].read_text(encoding="utf-8")
    assert "Market-Session Verification Plan" in verification_md
    assert str(planned_rerun / "expectancy_evidence.csv") in verification_md
    assert post_rerun_packet["status"].tolist() == ["WAITING_FOR_REGULAR_SESSION_LIVE_RERUN"]
    assert post_rerun_packet["can_mark_goal_complete"].tolist() == [False]
    assert post_rerun_packet["green_ticket_rows"].tolist() == [0]
    assert str(planned_rerun / "green_trade_tickets.csv") in post_rerun_packet["evidence_files"].iloc[0]
    assert "python3 -m uwos.options_agent.audit" in post_rerun_packet["audit_regeneration_command"].iloc[0]
    assert str(artifacts.paths["summary"]) in post_rerun_packet["audit_regeneration_command"].iloc[0]
    assert str(planned_rerun) in post_rerun_packet["audit_regeneration_command"].iloc[0]
    assert "--rerun-agent-reviews-json" in post_rerun_packet["audit_regeneration_command"].iloc[0]
    post_rerun_md = artifacts.paths["post_rerun_verification_packet_md"].read_text(encoding="utf-8")
    assert "Post-Rerun Verification Packet" in post_rerun_md
    assert "Regenerate This Verification" in post_rerun_md
    assert "green rows, structure-aligned ticket expectancy, and the completion verdict must all agree" in post_rerun_md
    assert green_proof["status"].tolist() == ["BLOCK_NO_GREEN_TICKETS"]
    assert green_proof["green_ticket_rows"].tolist() == [0]
    assert green_proof["valid_green_ticket_rows"].tolist() == [0]
    green_proof_md = artifacts.paths["green_ticket_execution_proof_packet_md"].read_text(encoding="utf-8")
    assert "Green-Ticket Execution Proof Packet" in green_proof_md
    assert "Every green row must have ready_to_enter=true" in green_proof_md
    assert session_shadow["status"].tolist() == ["PASS_SESSION_ONLY_GREEN_SHADOW_READY"]
    assert session_shadow["shadow_candidate_rows"].tolist() == [1]
    assert session_shadow["row_fail_rows"].tolist() == [0]
    assert session_shadow["non_session_blocker_rows"].tolist() == [0]
    assert session_shadow["position_max_profit"].tolist() == [260.0]
    assert session_shadow["position_max_loss"].tolist() == [740.0]
    assert session_shadow["tickers"].tolist() == ["LIVEQ"]
    session_shadow_md = artifacts.paths["session_only_green_shadow_proof_packet_md"].read_text(encoding="utf-8")
    assert "Session-Only Green Shadow Proof Packet" in session_shadow_md
    assert "not execution permission" in session_shadow_md
    assert actionability_packet["status"].tolist() == ["PASS_YELLOW_ONLY_SURFACE_SEPARATED"]
    assert actionability_packet["target_order_rows"].tolist() == [2]
    assert actionability_packet["target_ready_to_enter_rows"].tolist() == [0]
    assert actionability_packet["target_missing_entry_type_rows"].tolist() == [0]
    assert actionability_packet["target_missing_plain_language_leg_rows"].tolist() == [0]
    assert actionability_packet["entry_types"].tolist() == ["CREDIT, DEBIT"]
    actionability_md = artifacts.paths["actionability_surface_proof_packet_md"].read_text(encoding="utf-8")
    assert "Structural recommendation labels such as ENTER are not execution permission" in actionability_md
    assert expectancy_packet["status"].tolist() == ["blocked_no_green_orders_and_no_positive_monthly_expectancy"]
    assert expectancy_packet["monthly_claim_allowed"].tolist() == [False]
    assert expectancy_packet["minimum_sample_size"].tolist() == [core.MIN_EXPECTANCY_SAMPLE_SIZE]
    assert "green-ticket ticker support is proven separately" in expectancy_packet["required_evidence"].iloc[0]
    assert "expectancy_summary=3" in expectancy_packet["blocking_source_counts"].iloc[0]
    expectancy_md = artifacts.paths["expectancy_proof_packet_md"].read_text(encoding="utf-8")
    assert "Monthly claim allowed: False" in expectancy_md
    assert "This packet is a claim gate" in expectancy_md
    assert ticket_expectancy_packet["status"].tolist() == ["BLOCK_NO_GREEN_TICKERS_FOR_EXPECTANCY_CLAIM"]
    assert "LIVEQ" not in ticket_expectancy_packet["ticket_tickers"].iloc[0]
    ticket_expectancy_md = artifacts.paths["ticket_expectancy_proof_packet_md"].read_text(encoding="utf-8")
    assert "Structure-Aligned Ticket Expectancy Proof Packet" in ticket_expectancy_md
    assert "Replay-only or unrelated-strategy" in ticket_expectancy_md
    assert monthly_guardrail_packet["status"].tolist() == ["FAIL_STALE_MONTHLY_FEASIBILITY_GUARDRAIL"]
    assert monthly_guardrail_packet["missing_required_metric_count"].tolist() == [3]
    assert monthly_guardrail_packet["required_metric"].tolist() == ["ready_ticket_expectancy_evidence"]
    monthly_guardrail_md = artifacts.paths["monthly_feasibility_guardrail_proof_packet_md"].read_text(encoding="utf-8")
    assert "Monthly Feasibility Guardrail Proof Packet" in monthly_guardrail_md
    assert agentic_packet["status"].tolist() == ["PASS_FULL_AGENTIC_TICKET_COVERAGE"]
    assert agentic_packet["ticket_rows_with_agentic_ready"].tolist() == [2]
    assert agentic_packet["ticket_rows_without_agentic_ready"].tolist() == [0]
    assert agentic_packet["required_min_ticket_lanes"].tolist() == [core.MIN_AGENTIC_REVIEW_LANES_PER_TICKER]
    assert agentic_packet["min_ticket_distinct_review_count"].tolist() == [4]
    assert agentic_packet["ticket_rows_below_min_ticket_lanes"].tolist() == [0]
    agentic_md = artifacts.paths["agentic_coverage_proof_packet_md"].read_text(encoding="utf-8")
    assert "Every user-facing ticket row" in agentic_md
    assert "Ticket rows below lane minimum: 0" in agentic_md
    assert validation_packet["status"].tolist() == ["PROVEN_WINDOW_COVERED"]
    assert validation_packet["tested_date_count"].tolist() == [2]
    assert validation_packet["untested_available_date_count"].tolist() == [0]
    assert validation_packet["available_dates_outside_window_count"].tolist() == [1]
    validation_md = artifacts.paths["validation_coverage_proof_packet_md"].read_text(encoding="utf-8")
    assert "2026-05-21, 2026-05-22" in validation_md
    assert "2026-05-20" in validation_md
    assert cutoff_packet["status"].tolist() == ["PASS_NO_ARTIFICIAL_CUTOFFS"]
    assert cutoff_packet["candidate_rows"].tolist() == [4]
    assert cutoff_packet["research_task_rows"].tolist() == [4]
    assert cutoff_packet["qualified_candidate_rows"].tolist() == [2]
    assert cutoff_packet["priced_candidate_rows"].tolist() == [2]
    assert cutoff_packet["expected_no_trade_rows"].tolist() == [2]
    assert cutoff_packet["no_trade_audit_rows"].tolist() == [2]
    cutoff_md = artifacts.paths["cutoff_visibility_proof_packet_md"].read_text(encoding="utf-8")
    assert "Cutoff Visibility Proof Packet" in cutoff_md
    assert "not capped by top-trades" in cutoff_md
    assert list(live_spread_quality.columns) == audit.LIVE_SPREAD_QUALITY_ROLLUP_COLUMNS
    assert live_spread_quality_packet["status"].tolist() == ["PASS_LIVE_SPREAD_QUALITY_GATED"]
    assert live_spread_quality_packet["audited_rows"].tolist() == [6]
    assert live_spread_quality_packet["block_rows"].tolist() == [3]
    assert live_spread_quality_packet["blocked_still_actionable_rows"].tolist() == [0]
    assert live_spread_quality_packet["target_candidate_block_rows"].tolist() == [0]
    live_spread_quality_md = artifacts.paths["live_spread_quality_proof_packet_md"].read_text(encoding="utf-8")
    assert "Live Spread Quality Proof Packet" in live_spread_quality_md
    assert "Bad live/snapshot spread markets were blocked" in live_spread_quality_md
    assert quality_packet["status"].tolist() == ["PASS_CORE_ONLY_TICKETS"]
    assert quality_packet["not_core_or_liquid_ticket_rows"].tolist() == [0]
    assert quality_packet["liquid_non_core_ticket_rows"].tolist() == [0]
    assert "OKLO" in quality_packet["focus_speculative_examples"].iloc[0]
    assert "URA" in quality_packet["focus_excluded_examples"].iloc[0]
    quality_md = artifacts.paths["underlying_quality_proof_packet_md"].read_text(encoding="utf-8")
    assert "Only core large-cap/index/ETF underlyings" in quality_md
    assert "DVN" in quality_md
    assert action_surface_quality_packet["status"].tolist() == ["PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS"]
    assert action_surface_quality_packet["ticket_bad_underlying_rows"].tolist() == [0]
    assert action_surface_quality_packet["market_open_recheck_bad_underlying_rows"].tolist() == [0]
    assert action_surface_quality_packet["focus_bad_actionable_rows"].tolist() == [0]
    action_surface_quality_md = artifacts.paths["action_surface_underlying_quality_proof_packet_md"].read_text(encoding="utf-8")
    assert "Action-Surface Underlying Quality Proof Packet" in action_surface_quality_md
    assert "Red no-action audit tickers" in action_surface_quality_md
    assert major_packet["status"].tolist() == ["PASS_ALL_MAJOR_NAMES_EXPLAINED"]
    assert major_packet["required_ticker_count"].tolist() == [len(core.CORE_AUDIT_TICKERS)]
    assert major_packet["missing_required_ticker_count"].tolist() == [0]
    assert major_packet["required_rows_missing_reason"].tolist() == [0]
    major_md = artifacts.paths["major_name_coverage_proof_packet_md"].read_text(encoding="utf-8")
    assert "AAPL" in major_md
    assert "NVDA" in major_md
    assert "AVGO" in major_md
    assert "PLTR" in major_md
    assert target_audit.loc[target_audit["metric"].eq("credit_target_rows"), "status"].tolist() == ["PROVEN"]
    assert target_audit.loc[target_audit["metric"].eq("debit_target_rows"), "status"].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("use multi-agent orchestration evidence"),
        "artifact",
    ].tolist() == [str(artifacts.paths["agentic_coverage_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("validate across multiple available UW dates"),
        "artifact",
    ].tolist() == [str(artifacts.paths["validation_coverage_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove latest live probe is not the whole validation"),
        "artifact",
    ].tolist() == [str(artifacts.paths["multi_date_readiness_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove latest live probe is not the whole validation"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("avoid artificial trade-count cutoffs"),
        "artifact",
    ].tolist() == [str(artifacts.paths["cutoff_visibility_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("avoid artificial trade-count cutoffs"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prioritize liquid large-cap/index/high-volume names over junk"),
        "artifact",
    ].tolist() == [str(artifacts.paths["action_surface_underlying_quality_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("block bad live spread markets from actionable surfaces"),
        "artifact",
    ].tolist() == [str(artifacts.paths["live_spread_quality_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("block bad live spread markets from actionable surfaces"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove market-open recheck queue is complete and only session-blocked"),
        "artifact",
    ].tolist() == [str(artifacts.paths["market_open_recheck_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove market-open recheck queue is complete and only session-blocked"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove live rerun preflight has queue-ticker agent reviews"),
        "artifact",
    ].tolist() == [str(artifacts.paths["live_rerun_preflight_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("prove live rerun preflight has queue-ticker agent reviews"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("separate yellow target orders from green send-now orders"),
        "artifact",
    ].tolist() == [str(artifacts.paths["actionability_surface_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("explain major-name inclusion/exclusion"),
        "artifact",
    ].tolist() == [str(artifacts.paths["major_name_coverage_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("be execution-ready trade quality confidence pipeline"),
        "status",
    ].tolist() == ["NOT_ACHIEVED"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("be execution-ready trade quality confidence pipeline"),
        "artifact",
    ].tolist() == [str(artifacts.paths["green_ticket_execution_proof_packet"])]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("do not claim $10k/month readiness without evidence"),
        "artifact",
    ].tolist() == [str(artifacts.paths["ticket_expectancy_proof_packet"])]
    assert completion_verdict["can_mark_goal_complete"].tolist() == [False]
    assert completion_verdict["update_goal_action"].tolist() == ["do_not_call_update_goal_complete"]
    assert "be execution-ready trade quality confidence pipeline" in completion_verdict["blocking_requirements"].iloc[0]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("overall_completion"),
        "status",
    ].tolist() == ["ACTIVE_NOT_COMPLETE"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("overall_completion"),
        "required_next_action",
    ].iloc[0] == readiness_dashboard.loc[
        readiness_dashboard["area"].eq("execution_readiness"),
        "required_next_action",
    ].iloc[0]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("execution_readiness"),
        "status",
    ].tolist() == ["NOT_ACHIEVED"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("cutoff_visibility"),
        "status",
    ].tolist() == ["PROVEN"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("live_spread_quality"),
        "status",
    ].tolist() == ["PROVEN"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("market_open_recheck_quality"),
        "status",
    ].tolist() == ["PROVEN"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("live_rerun_preflight"),
        "status",
    ].tolist() == ["PROVEN"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("session_only_green_shadow"),
        "status",
    ].tolist() == ["PASS_SESSION_ONLY_GREEN_SHADOW_READY"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("action_surface_underlying_quality"),
        "status",
    ].tolist() == ["PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("monthly_feasibility_guardrail"),
        "status",
    ].tolist() == ["FAIL_STALE_MONTHLY_FEASIBILITY_GUARDRAIL"]
    assert readiness_dashboard.loc[
        readiness_dashboard["area"].eq("post_rerun_go_no_go"),
        "status",
    ].tolist() == ["WAITING_FOR_REGULAR_SESSION_LIVE_RERUN"]
    dashboard_md = artifacts.paths["readiness_dashboard_md"].read_text(encoding="utf-8")
    assert "Options Agent Readiness Dashboard" in dashboard_md
    assert "Use this dashboard as an index only" in dashboard_md
    completion_md = artifacts.paths["completion_verdict_md"].read_text(encoding="utf-8")
    assert "Can mark goal complete: False" in completion_md
    assert "do_not_call_update_goal_complete" in completion_md
    assert "Do not mark the goal complete yet." in artifacts.paths["goal_completion_audit_md"].read_text(encoding="utf-8")


def test_multi_date_scope_proof_passes_with_market_open_live_probe_even_without_green() -> None:
    summary = pd.DataFrame(
        [
            {"date": "2026-05-21", "trade_ticket_rows": 1, "green_ready_orders": 0, "yellow_target_candidates": 1},
            {"date": "2026-05-22", "trade_ticket_rows": 1, "green_ready_orders": 0, "yellow_target_candidates": 1},
        ]
    )
    live_probe_summary = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "market_session_open": True,
                "green_ready_orders": 0,
                "market_open_recheck_queue": 0,
            }
        ]
    )
    market_open_execution_packet = pd.DataFrame(
        [{"date": "2026-05-22", "status": "no_green_orders_present"}]
    )

    packet = audit.build_multi_date_readiness_proof_packet(
        summary=summary,
        live_probe_summary=live_probe_summary,
        market_open_execution_packet=market_open_execution_packet,
    )

    assert packet["status"].tolist() == ["PASS_MULTI_DATE_WITH_MARKET_OPEN_LIVE_PROBE_NO_GREEN_TICKETS"]
    assert packet["validation_date_count"].tolist() == [2]
    assert packet["live_market_session_open_count"].tolist() == [1]
    assert packet["live_green_ready_orders"].tolist() == [0]
    assert packet["live_yellow_recheck_rows"].tolist() == [0]


def test_market_open_probe_without_queue_supersedes_recheck_preflight(tmp_path: Path) -> None:
    day = "2026-05-22"
    day_dir = tmp_path / day
    day_dir.mkdir()
    (day_dir / f"stock-screener-{day}.csv").write_text("ticker\nAMAT\n", encoding="utf-8")
    (day_dir / f"hot-chains-{day}.csv").write_text("option_symbol\nAMAT260529C00467500\n", encoding="utf-8")
    (day_dir / f"chain-oi-changes-{day}.csv").write_text("option_symbol\nAMAT260529C00467500\n", encoding="utf-8")

    execution_packet = pd.DataFrame(
        [
            {
                "date": day,
                "status": "market_open_live_probe_no_green_orders",
                "fresh_live_quotes_ready": True,
                "portfolio_ready": True,
                "agentic_reviews_ready": True,
                "market_session_open": True,
                "green_ready_orders": 0,
                "yellow_recheck_rows": 0,
                "agent_reviews_json": str(tmp_path / "agentic_reviews.json"),
                "out_dir": str(tmp_path / "fresh_live_rerun"),
                "command": (
                    "python3 -m uwos.options_agent "
                    f"--date {day} "
                    f"--base-dir {tmp_path} "
                    f"--out-dir {tmp_path / 'fresh_live_rerun'} "
                    "--live-schwab --live-portfolio "
                    f"--agent-reviews-json {tmp_path / 'agentic_reviews.json'}"
                ),
            }
        ]
    )
    details = pd.DataFrame(columns=audit.MARKET_OPEN_RECHECK_DETAIL_COLUMNS)

    recheck_packet = audit.build_market_open_recheck_proof_packet(
        details,
        market_open_execution_packet=execution_packet,
    )
    preflight_packet = audit.build_live_rerun_preflight_proof_packet(
        base_dir=tmp_path,
        market_open_recheck_details=details,
        market_open_execution_packet=execution_packet,
        preflight_details=pd.DataFrame(columns=audit.LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS),
    )

    assert recheck_packet["status"].tolist() == [
        "PASS_NO_MARKET_OPEN_RECHECK_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]
    assert preflight_packet["status"].tolist() == [
        "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]

    green_packet = execution_packet.copy()
    green_packet.loc[0, "status"] = "green_orders_present_verify_ticket_scoped_expectancy"
    green_packet.loc[0, "green_ready_orders"] = 1
    green_recheck_packet = audit.build_market_open_recheck_proof_packet(
        details,
        market_open_execution_packet=green_packet,
    )
    green_preflight_packet = audit.build_live_rerun_preflight_proof_packet(
        base_dir=tmp_path,
        market_open_recheck_details=details,
        market_open_execution_packet=green_packet,
        preflight_details=pd.DataFrame(columns=audit.LIVE_RERUN_PREFLIGHT_DETAIL_COLUMNS),
    )

    assert green_recheck_packet["status"].tolist() == [
        "PASS_NO_MARKET_OPEN_RECHECK_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]
    assert "produced green orders" in green_recheck_packet["note"].iloc[0]
    assert green_preflight_packet["status"].tolist() == [
        "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]


def test_completed_market_open_probe_preflight_ignores_stale_dated_queue(tmp_path: Path) -> None:
    day = "2026-05-22"
    completed_out_dir = tmp_path / "live_readiness_probe_market_open_rerun_2026-05-22"
    completed_out_dir.mkdir()
    execution_packet = pd.DataFrame(
        [
            {
                "date": day,
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "fresh_live_quotes_ready": True,
                "portfolio_ready": True,
                "agentic_reviews_ready": True,
                "market_session_open": True,
                "green_ready_orders": 1,
                "yellow_recheck_rows": 0,
                "agent_reviews_json": "",
                "out_dir": str(completed_out_dir),
                "command": "",
            }
        ]
    )
    stale_dated_queue = pd.DataFrame([{"ticker": "AAPL", "source_kind": "dated_run", "row_pass": True}])

    preflight_details = audit.build_live_rerun_preflight_details(
        market_open_recheck_details=stale_dated_queue,
        market_open_execution_packet=execution_packet,
    )
    preflight_packet = audit.build_live_rerun_preflight_proof_packet(
        base_dir=tmp_path,
        market_open_recheck_details=stale_dated_queue,
        market_open_execution_packet=execution_packet,
        preflight_details=preflight_details,
    )

    assert preflight_details.empty
    assert preflight_packet["status"].tolist() == [
        "PASS_NO_LIVE_RERUN_QUEUE_AFTER_MARKET_OPEN_LIVE_PROBE"
    ]
    assert preflight_packet["queue_ticker_count"].tolist() == [0]
    assert preflight_packet["rerun_out_dir_clear"].tolist() == [False]
    assert "rerun_command_missing" not in str(preflight_packet["failed_examples"].iloc[0])


def test_completion_verdict_uses_best_market_open_packet_row() -> None:
    goal_audit = pd.DataFrame(
        [
            {"requirement": "multi-date", "status": "PROVEN"},
            {"requirement": "execution-ready", "status": "ACHIEVED"},
        ]
    )
    market_packet = pd.DataFrame(
        [
            {
                "status": "refresh_live_probe_inputs_before_rerun",
                "market_session_open": False,
                "next_regular_session_start": "2026-05-27T06:30:00-07:00",
            },
            {
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "market_session_open": True,
                "next_regular_session_start": "2026-05-26T06:30:00-07:00",
            },
        ]
    )

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=market_packet,
        expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "positive_expectancy_ready_for_monthly_claim_review",
                    "monthly_claim_allowed": True,
                }
            ]
        ),
        ticket_expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
    )

    assert verdict["can_mark_goal_complete"].tolist() == [True]
    assert verdict["market_open_packet_status"].tolist() == [
        "green_orders_present_verify_ticket_scoped_expectancy"
    ]


def test_completion_verdict_can_close_execution_goal_without_monthly_claim() -> None:
    goal_audit = pd.DataFrame(
        [
            {"requirement": "multi-date", "status": "PROVEN"},
            {"requirement": "monthly guardrail", "status": "PROVEN"},
            {"requirement": "execution-ready", "status": "ACHIEVED"},
        ]
    )
    market_packet = pd.DataFrame(
        [
            {
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "market_session_open": True,
                "next_regular_session_start": "2026-05-28T06:30:00-07:00",
            }
        ]
    )

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=market_packet,
        expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "blocked_no_positive_overall_strategy_expectancy",
                    "monthly_claim_allowed": False,
                }
            ]
        ),
        ticket_expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
    )

    assert verdict["can_mark_goal_complete"].tolist() == [True]
    assert verdict["monthly_claim_allowed"].tolist() == [False]
    assert verdict["update_goal_action"].tolist() == ["call_update_goal_complete"]
    assert "$10k/month" in verdict["note"].iloc[0]


def test_expectancy_proof_packet_keeps_overall_expectancy_separate_from_ticket_coverage() -> None:
    packet = audit.build_expectancy_proof_packet(
        summary=pd.DataFrame([{"date": "2026-05-22", "green_ready_orders": 0}]),
        tickets=pd.DataFrame([{"ticker": "GOOGL"}]),
        expectancy=pd.DataFrame(
            [
                {
                    "source": "expectancy_summary",
                    "status": "BLOCK",
                    "matched_current_tickers": "GOOGL",
                }
            ]
        ),
        live_probe_summary=pd.DataFrame([{"green_ready_orders": 1}]),
    )

    assert packet["status"].tolist() == ["blocked_no_positive_overall_strategy_expectancy"]
    assert packet["monthly_claim_allowed"].tolist() == [False]
    assert "ticket_scoped" not in packet["status"].iloc[0]
    assert "green-ticket ticker support is proven separately" in packet["required_evidence"].iloc[0]


def test_agentic_coverage_proof_blocks_partial_ticket_coverage() -> None:
    summary = pd.DataFrame(
        [
            {
                "date": "2026-05-21",
                "source_dir": "/tmp/non_agentic",
                "agentic_review_coverage_pct": 0.0,
                "agentic_review_lane_coverage_pct": 0.0,
            },
            {
                "date": "2026-05-22",
                "source_dir": "/tmp/agentic",
                "agentic_review_coverage_pct": 1.0,
                "agentic_review_lane_coverage_pct": 1.0,
            },
        ]
    )
    lanes = pd.DataFrame(
        [
            {
                "date": "2026-05-21",
                "ticker": "MSFT",
                "external_agent_distinct_review_count": 4,
                "run_agentic_reviews_ready": False,
            },
            {
                "date": "2026-05-22",
                "ticker": "GOOGL",
                "external_agent_distinct_review_count": 4,
                "run_agentic_reviews_ready": True,
            },
        ]
    )

    packet = audit.build_agentic_coverage_proof_packet(summary=summary, ticket_review_lanes=lanes)

    assert packet["status"].tolist() == ["PARTIAL_AGENTIC_TICKET_COVERAGE"]
    assert packet["ticket_rows"].tolist() == [2]
    assert packet["ticket_rows_with_agentic_ready"].tolist() == [1]
    assert packet["ticket_rows_without_agentic_ready"].tolist() == [1]
    assert packet["non_agentic_ticket_dates"].tolist() == ["2026-05-21"]


def test_agentic_coverage_proof_blocks_ticket_below_lane_minimum() -> None:
    summary = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "source_dir": "/tmp/agentic",
                "agentic_review_coverage_pct": 1.0,
                "agentic_review_lane_coverage_pct": 1.0,
            },
        ]
    )
    lanes = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "ticker": "GOOGL",
                "external_agent_distinct_review_count": core.MIN_AGENTIC_REVIEW_LANES_PER_TICKER - 1,
                "run_agentic_reviews_ready": True,
            },
        ]
    )

    packet = audit.build_agentic_coverage_proof_packet(summary=summary, ticket_review_lanes=lanes)

    assert packet["status"].tolist() == ["PARTIAL_AGENTIC_TICKET_COVERAGE"]
    assert packet["ticket_rows_with_agentic_ready"].tolist() == [1]
    assert packet["ticket_rows_without_agentic_ready"].tolist() == [0]
    assert packet["ticket_rows_below_min_ticket_lanes"].tolist() == [1]
    assert packet["below_min_ticket_lane_dates"].tolist() == ["2026-05-22"]


def test_cutoff_visibility_proof_blocks_stale_capped_no_trade_audit(tmp_path: Path) -> None:
    run_dir = tmp_path / "out" / "options_agent" / "stale_cap_2026-05-22"
    run_dir.mkdir(parents=True)
    (run_dir / "options_agent_manifest_2026-05-22.json").write_text(
        json.dumps({"as_of": "2026-05-22", "row_counts": {"research_tasks": 3}}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"ticker": "AAPL", "quality_status": "qualified"},
            {"ticker": "NVDA", "quality_status": "watch"},
            {"ticker": "MSFT", "quality_status": "watch"},
        ]
    ).to_csv(run_dir / "candidate_generation.csv", index=False)
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(run_dir / "priced_candidates.csv", index=False)
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(run_dir / "final_recommendations.csv", index=False)
    pd.DataFrame([{"ticker": "NVDA"}]).to_csv(run_dir / "no_trade_audit.csv", index=False)

    packet = audit.build_cutoff_visibility_proof_packet([run_dir])

    assert packet["status"].tolist() == ["FAIL_ARTIFICIAL_CUTOFF_OR_STALE_AUDIT_ROWS"]
    assert packet["expected_no_trade_rows"].tolist() == [2]
    assert packet["no_trade_audit_rows"].tolist() == [1]
    assert "stale_cap_2026-05-22" in packet["no_trade_missing_expected_runs"].iloc[0]


def test_green_ticket_execution_proof_requires_row_level_execution_gates() -> None:
    live_summary = pd.DataFrame([{"source_dir": "/tmp/live", "market_session_open": True}])
    details = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "validation_lane": "live_readiness_probe",
                "source_dir": "/tmp/live",
                "ticker": "AAPL",
                "ready_to_enter": True,
                "order_readiness": "ready_to_enter",
                "entry_type": "DEBIT",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "live_validation_status": "PASS",
                "execution_status": "ready",
                "execution_blockers": "",
                "execution_confidence_score": core.MIN_EXECUTION_CONFIDENCE_SCORE,
                "execution_confidence_rating": "MEDIUM",
                "trade_quality_confidence_rating": "MEDIUM",
                "confidence_score_pass": True,
                "execution_confidence_pass": True,
                "trade_quality_confidence_pass": True,
                "market_session_open": True,
                "trade_plan": "BUY 1 AAPL 2026-06-18 200 Call / SELL 1 AAPL 2026-06-18 205 Call @ 1.25 DEBIT",
                "sell_leg": "SELL 1 AAPL 2026-06-18 205 Call",
                "buy_leg": "BUY 1 AAPL 2026-06-18 200 Call",
                "row_pass": True,
                "fail_reasons": "",
            }
        ],
        columns=audit.GREEN_TICKET_EXECUTION_DETAIL_COLUMNS,
    )

    packet = audit.build_green_ticket_execution_proof_packet(
        details=details,
        live_probe_summary=live_summary,
    )

    assert packet["status"].tolist() == ["PASS_GREEN_TICKETS_EXECUTION_READY"]
    assert packet["green_ticket_rows"].tolist() == [1]
    assert packet["valid_green_ticket_rows"].tolist() == [1]
    assert packet["confidence_score_pass_rows"].tolist() == [1]
    assert packet["execution_confidence_pass_rows"].tolist() == [1]
    assert packet["trade_quality_confidence_pass_rows"].tolist() == [1]
    assert packet["plain_language_leg_rows"].tolist() == [1]


def test_green_ticket_execution_proof_rejects_low_confidence_rows() -> None:
    live_summary = pd.DataFrame([{"source_dir": "/tmp/live", "market_session_open": True}])
    details = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "validation_lane": "live_readiness_probe",
                "source_dir": "/tmp/live",
                "ticker": "AAPL",
                "ready_to_enter": True,
                "order_readiness": "ready_to_enter",
                "entry_type": "DEBIT",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "live_validation_status": "PASS",
                "execution_status": "ready",
                "execution_blockers": "",
                "execution_confidence_score": core.MIN_EXECUTION_CONFIDENCE_SCORE - 1,
                "execution_confidence_rating": "LOW",
                "trade_quality_confidence_rating": "LOW",
                "confidence_score_pass": False,
                "execution_confidence_pass": False,
                "trade_quality_confidence_pass": False,
                "market_session_open": True,
                "trade_plan": "BUY 1 AAPL 2026-06-18 200 Call / SELL 1 AAPL 2026-06-18 205 Call @ 1.25 DEBIT",
                "sell_leg": "SELL 1 AAPL 2026-06-18 205 Call",
                "buy_leg": "BUY 1 AAPL 2026-06-18 200 Call",
                "row_pass": False,
                "fail_reasons": "execution_confidence_score_below_threshold; execution_confidence_rating_not_MEDIUM_or_HIGH; trade_quality_confidence_rating_not_MEDIUM_or_HIGH",
            }
        ],
        columns=audit.GREEN_TICKET_EXECUTION_DETAIL_COLUMNS,
    )

    packet = audit.build_green_ticket_execution_proof_packet(
        details=details,
        live_probe_summary=live_summary,
    )

    assert packet["status"].tolist() == ["FAIL_INVALID_GREEN_TICKET_ROWS"]
    assert packet["confidence_score_pass_rows"].tolist() == [0]
    assert packet["execution_confidence_pass_rows"].tolist() == [0]
    assert packet["trade_quality_confidence_pass_rows"].tolist() == [0]
    assert "execution_confidence_score_below_threshold" in packet["invalid_examples"].iloc[0]


def test_green_ticket_execution_details_reject_occ_codes(tmp_path: Path) -> None:
    live_dir = tmp_path / "live_readiness_probe_v017_2026-05-22"
    live_dir.mkdir()
    (live_dir / "options_agent_manifest_2026-05-22.json").write_text(
        json.dumps({"as_of": "2026-05-22"}),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "ready_to_enter": True,
                "order_readiness": "ready_to_enter",
                "entry_type": "DEBIT",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "live_validation_status": "PASS",
                "execution_status": "ready",
                "execution_blockers": "",
                "execution_confidence_score": core.MIN_EXECUTION_CONFIDENCE_SCORE,
                "execution_confidence_rating": "MEDIUM",
                "trade_quality_confidence_rating": "MEDIUM",
                "trade_plan": "BUY 1 AAPL260618C00200000 / SELL 1 AAPL260618C00205000 @ 1.25 DEBIT",
                "sell_leg": "SELL 1 AAPL260618C00205000",
                "buy_leg": "BUY 1 AAPL260618C00200000",
            }
        ]
    ).to_csv(live_dir / "green_trade_tickets.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "source_dir": str(live_dir.resolve()),
                "market_session_open": True,
            }
        ]
    )

    details = audit.build_green_ticket_execution_details(
        live_probe_dirs=[live_dir],
        live_probe_summary=summary,
    )
    packet = audit.build_green_ticket_execution_proof_packet(
        details=details,
        live_probe_summary=summary,
    )

    assert details["row_pass"].tolist() == [False]
    assert "plain_language_buy_sell_legs_missing" in details["fail_reasons"].iloc[0]
    assert packet["status"].tolist() == ["FAIL_INVALID_GREEN_TICKET_ROWS"]


def test_ticket_expectancy_proof_blocks_green_ticker_without_actual_forward_support() -> None:
    tickets = pd.DataFrame([{"ticker": "WMT"}])
    green = pd.DataFrame([{"ticker": "WMT"}])
    expectancy = pd.DataFrame(
        [
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "WMT",
            }
        ]
    )

    coverage = audit.build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green,
        expectancy=expectancy,
    )
    packet = audit.build_ticket_expectancy_proof_packet(coverage=coverage)

    assert coverage["status"].tolist() == ["WARN_REPLAY_ONLY_EXPECTANCY"]
    assert packet["status"].tolist() == ["BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"]
    assert packet["green_tickers_without_positive_actual_forward"].tolist() == ["WMT"]


def test_ticket_expectancy_proof_passes_only_with_actual_forward_support() -> None:
    tickets = pd.DataFrame([{"ticker": "WMT"}])
    green = pd.DataFrame([{"ticker": "WMT"}])
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "WMT",
            }
        ]
    )

    coverage = audit.build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green,
        expectancy=expectancy,
    )
    packet = audit.build_ticket_expectancy_proof_packet(coverage=coverage)

    assert coverage["status"].tolist() == ["PASS_ACTUAL_FORWARD_EXPECTANCY"]
    assert packet["status"].tolist() == ["PASS_GREEN_TICKER_EXPECTANCY_COVERAGE"]
    assert packet["tickers_with_positive_actual_forward"].tolist() == ["WMT"]


def test_ticket_expectancy_proof_rejects_ticker_only_actual_support() -> None:
    tickets = pd.DataFrame([{"ticker": "WMT"}])
    green = pd.DataFrame([{"ticker": "WMT"}])
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades_by_ticker",
                "evidence_type": "actual_closed_trades_by_ticker",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "WMT",
            }
        ]
    )

    coverage = audit.build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green,
        expectancy=expectancy,
    )
    packet = audit.build_ticket_expectancy_proof_packet(coverage=coverage)

    assert coverage["status"].tolist() == ["BLOCK_NO_POSITIVE_TICKET_EXPECTANCY"]
    assert packet["status"].tolist() == ["BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"]


def test_ticket_expectancy_proof_rejects_broad_aggregate_actual_support() -> None:
    tickets = pd.DataFrame([{"ticker": "WMT"}])
    green = pd.DataFrame([{"ticker": "WMT"}])
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 40,
                "matched_current_tickers": "WMT",
            }
        ]
    )

    coverage = audit.build_ticket_expectancy_coverage(
        tickets=tickets,
        green_ticket_execution_details=green,
        expectancy=expectancy,
    )
    packet = audit.build_ticket_expectancy_proof_packet(coverage=coverage)

    assert coverage["status"].tolist() == ["BLOCK_NO_POSITIVE_TICKET_EXPECTANCY"]
    assert packet["status"].tolist() == ["BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"]


def test_expectancy_evidence_matches_goog_googl_share_class_alias(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": "GOOG", "realized_pnl": 100.0, "strategy": "vertical_spread"},
        {"ticker": "GOOG", "realized_pnl": 70.0, "strategy": "vertical_spread"},
        {"ticker": "GOOG", "realized_pnl": -20.0, "strategy": "vertical_spread"},
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOGL",
                "ready_to_enter": True,
                "trade_plan": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 397.5 Call @ 0.63 CREDIT",
            }
        ]
    )

    expectancy = core.build_expectancy_evidence(tmp_path, pd.DataFrame(), tickets)
    by_ticker = expectancy[expectancy["evidence_type"].eq("actual_closed_trades_by_ticker")]
    by_strategy = expectancy[expectancy["evidence_type"].eq("actual_closed_trades_by_ticker_strategy")]
    annotated = core.annotate_actual_forward_expectancy(
        pd.DataFrame(
            [
                {
                    "ticker": "GOOGL",
                    "trade_plan": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 397.5 Call @ 0.63 CREDIT",
                }
            ]
        ),
        tmp_path,
    )

    assert by_ticker["matched_current_tickers"].tolist() == ["GOOGL"]
    assert by_ticker["status"].tolist() == ["PASS"]
    assert by_strategy["matched_current_tickers"].tolist() == ["GOOGL"]
    assert by_strategy["status"].tolist() == ["PASS"]
    assert annotated["actual_forward_expectancy_status"].tolist() == ["PASS"]
    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["PASS"]
    assert annotated["actual_forward_expectancy_source_tickers"].tolist() == ["GOOG"]


def test_strategy_expectancy_blocks_opposite_or_unrelated_ticker_history(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": "GOOG", "realized_pnl": 300.0, "strategy": "long_call"},
        {"ticker": "GOOG", "realized_pnl": 200.0, "strategy": "long_call"},
        {"ticker": "GOOG", "realized_pnl": 100.0, "strategy": "long_call"},
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "GOOGL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                    "trade_plan": "SELL 1 GOOGL 2026-06-05 395 Call / BUY 1 GOOGL 2026-06-05 400 Call @ 1.50 CREDIT",
                    "entry_limit": 1.5,
                "suggested_contracts": 5,
                    "max_profit": 150.0,
                    "max_loss": 350.0,
                    "credit_width_ratio": 0.30,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            }
        ]
    )

    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=1,
        external_review_count=5,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    decision = core.synthesize_decision_board(annotated, market_regime={"regime": "mixed"}, execution_context=context)

    assert annotated["actual_forward_expectancy_status"].tolist() == ["PASS"]
    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["BLOCK"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]


def test_route_expectancy_overrides_negative_vertical_family_for_bull_call(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"BCDWIN{idx}", "realized_pnl": 100.0, "strategy": "Bull Call Debit Spread"}
        for idx in range(19)
    ]
    rows.extend(
        {"ticker": f"BCDLOSS{idx}", "realized_pnl": -62.5, "strategy": "Bull Call Debit Spread"}
        for idx in range(23)
    )
    rows.extend(
        {"ticker": f"BPCLOSS{idx}", "realized_pnl": -100.0, "strategy": "Bull Put Credit Spread"}
        for idx in range(80)
    )
    rows.extend(
        {"ticker": "NEW", "realized_pnl": -100.0, "strategy": "Long Call"}
        for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "NEW",
                "structure": "bull call debit spread",
                "trade_plan": "BUY 1 NEW 2026-07-17 100 Call / SELL 1 NEW 2026-07-17 105 Call @ 1.20 DEBIT",
            }
        ]
    )

    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)
    row = annotated.iloc[0]

    assert row["actual_forward_expectancy_status"] == "BLOCK"
    assert row["actual_forward_strategy_expectancy_status"] == "PASS"
    assert row["actual_forward_strategy_expectancy_scope"] == "strategy_route"
    assert row["actual_forward_strategy_expectancy_family"] == "vertical_spread"
    assert row["actual_forward_strategy_expectancy_sample_size"] == 42
    assert row["actual_forward_strategy_expectancy_profit_factor"] == 1.322
    assert "route-level evidence is preferred over broad strategy-family evidence" in row["actual_forward_strategy_expectancy_note"]
    assert core._negative_strategy_expectancy_blocks_green(row) is False
    assert core._positive_strategy_expectancy_ready_for_green(row) is True


def test_ticker_route_negative_expectancy_overrides_positive_route_support(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"GOOD{idx}", "realized_pnl": 100.0, "strategy": "Bull Call Debit Spread"}
        for idx in range(30)
    ]
    rows.extend(
        {"ticker": f"LOSS{idx}", "realized_pnl": -50.0, "strategy": "Bull Call Debit Spread"}
        for idx in range(20)
    )
    rows.extend(
        {"ticker": "BAD", "realized_pnl": -100.0, "strategy": "Bull Call Debit Spread"}
        for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "BAD",
                "structure": "bull call debit spread",
                "trade_plan": "BUY 1 BAD 2026-07-17 100 Call / SELL 1 BAD 2026-07-17 105 Call @ 1.20 DEBIT",
            }
        ]
    )

    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)
    row = annotated.iloc[0]

    assert row["actual_forward_strategy_expectancy_status"] == "BLOCK"
    assert row["actual_forward_strategy_expectancy_scope"] == "ticker_route"
    assert row["actual_forward_strategy_expectancy_sample_size"] == core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE
    assert row["actual_forward_strategy_expectancy_avg_pnl"] == -100.0
    assert "Route-aligned actual/forward realized support" in row["actual_forward_strategy_expectancy_note"]


def test_completion_verdict_only_allows_goal_close_when_all_proofs_pass() -> None:
    goal_audit = pd.DataFrame(
        [
            {"requirement": "multi-date", "status": "PROVEN"},
            {"requirement": "execution-ready", "status": "ACHIEVED"},
        ]
    )
    market_packet = pd.DataFrame(
        [
            {
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "next_regular_session_start": "2026-05-26T06:30:00-07:00",
            }
        ]
    )
    expectancy_packet = pd.DataFrame(
        [
            {
                "status": "positive_expectancy_ready_for_monthly_claim_review",
                "monthly_claim_allowed": True,
            }
        ]
    )

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=market_packet,
        expectancy_proof_packet=expectancy_packet,
        ticket_expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
    )

    assert verdict["can_mark_goal_complete"].tolist() == [True]
    assert verdict["update_goal_action"].tolist() == ["call_update_goal_complete"]


def test_completion_verdict_blocks_unrelated_positive_expectancy() -> None:
    goal_audit = pd.DataFrame(
        [
            {"requirement": "multi-date", "status": "PROVEN"},
            {"requirement": "execution-ready", "status": "ACHIEVED"},
        ]
    )
    market_packet = pd.DataFrame(
        [
            {
                "status": "green_orders_present_verify_ticket_scoped_expectancy",
                "next_regular_session_start": "2026-05-26T06:30:00-07:00",
            }
        ]
    )
    broad_expectancy_packet = pd.DataFrame(
        [
            {
                "status": "positive_expectancy_ready_for_monthly_claim_review",
                "monthly_claim_allowed": True,
            }
        ]
    )
    ticket_expectancy_packet = pd.DataFrame(
        [
            {
                "status": "BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY",
            }
        ]
    )

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=market_packet,
        expectancy_proof_packet=broad_expectancy_packet,
        ticket_expectancy_proof_packet=ticket_expectancy_packet,
    )

    assert verdict["can_mark_goal_complete"].tolist() == [False]
    assert verdict["update_goal_action"].tolist() == ["do_not_call_update_goal_complete"]
    assert verdict["ticket_expectancy_packet_status"].tolist() == [
        "BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY"
    ]


def test_goal_completion_audit_can_complete_with_live_green_and_ticket_expectancy() -> None:
    summary = pd.DataFrame(
        [
            {"date": "2026-05-21", "monthly_feasibility": "proven", "expectancy_summary_status": "PASS"},
            {"date": "2026-05-22", "monthly_feasibility": "proven", "expectancy_summary_status": "PASS"},
        ]
    )
    tickets = pd.DataFrame(
        [
            {"ticker": "AAPL", "entry_type": "CREDIT", "ready_to_enter": False, "target_order_status": "target_order_candidate"},
            {"ticker": "MSFT", "entry_type": "DEBIT", "ready_to_enter": False, "target_order_status": "target_order_candidate"},
            {"ticker": "NVDA", "entry_type": "CREDIT", "ready_to_enter": True, "target_order_status": ""},
        ]
    )
    paths = {
        "validation_coverage_proof_packet": Path("/tmp/validation.csv"),
        "multi_date_readiness_proof_packet": Path("/tmp/multi_date.csv"),
        "cutoff_visibility_proof_packet": Path("/tmp/cutoff.csv"),
        "agentic_coverage_proof_packet": Path("/tmp/agentic.csv"),
        "live_spread_quality_proof_packet": Path("/tmp/live_spread_quality.csv"),
        "market_open_recheck_proof_packet": Path("/tmp/market_open_recheck.csv"),
        "live_rerun_preflight_proof_packet": Path("/tmp/live_rerun_preflight.csv"),
        "underlying_quality_proof_packet": Path("/tmp/quality.csv"),
        "action_surface_underlying_quality_proof_packet": Path("/tmp/action_surface_quality.csv"),
        "target_preservation_audit": Path("/tmp/target.csv"),
        "actionability_surface_proof_packet": Path("/tmp/actionability.csv"),
        "major_name_coverage_proof_packet": Path("/tmp/major.csv"),
        "ticket_expectancy_proof_packet": Path("/tmp/ticket_expectancy.csv"),
        "green_ticket_execution_proof_packet": Path("/tmp/green.csv"),
    }

    goal_audit = audit.build_goal_completion_audit(
        summary=summary,
        tickets=tickets,
        focus_coverage=pd.DataFrame(),
        ticket_review_lanes=pd.DataFrame(),
        agentic_coverage_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_FULL_AGENTIC_TICKET_COVERAGE",
                    "agentic_ready_dates": "2026-05-21, 2026-05-22",
                    "ticket_rows": 3,
                    "ticket_rows_with_agentic_ready": 3,
                    "ticket_rows_without_agentic_ready": 0,
                    "non_agentic_ticket_dates": "",
                }
            ]
        ),
        validation_coverage_proof=pd.DataFrame(
            [
                {
                    "status": "PROVEN_WINDOW_COVERED",
                    "window_available_source_date_count": 2,
                    "untested_available_date_count": 0,
                    "base_available_source_date_count": 2,
                    "available_dates_outside_window_count": 0,
                }
            ]
        ),
        cutoff_visibility_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_NO_ARTIFICIAL_CUTOFFS",
                    "candidate_rows": 10,
                    "research_task_rows": 10,
                    "qualified_candidate_rows": 3,
                    "priced_candidate_rows": 3,
                    "expected_no_trade_rows": 7,
                    "no_trade_audit_rows": 7,
                    "problem_runs": "",
                }
            ]
        ),
        live_spread_quality_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_LIVE_SPREAD_QUALITY_GATED",
                    "audited_rows": 5,
                    "block_rows": 1,
                    "blocked_still_actionable_rows": 0,
                    "target_candidate_block_rows": 0,
                    "blocked_tickers": "WIDE",
                }
            ]
        ),
        underlying_quality_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_CORE_ONLY_TICKETS",
                    "not_core_or_liquid_ticket_rows": 0,
                    "liquid_non_core_ticket_rows": 0,
                }
            ]
        ),
        major_name_coverage_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_ALL_MAJOR_NAMES_EXPLAINED",
                    "required_ticker_count": 17,
                    "covered_required_ticker_count": 17,
                    "missing_required_tickers": "",
                    "required_rows_missing_reason": 0,
                }
            ]
        ),
        expectancy=pd.DataFrame([{"source": "schwab_closed_trades", "status": "PASS"}]),
        market_open_recheck_queue=pd.DataFrame([{"source_kind": "live_probe"}]),
        market_open_recheck_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_LIVE_MARKET_OPEN_RECHECK_QUEUE_READY",
                    "queue_rows": 1,
                    "row_fail_rows": 0,
                    "credit_rows": 1,
                    "debit_rows": 0,
                    "tickers": "AAPL",
                }
            ]
        ),
        live_rerun_preflight_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_LIVE_RERUN_PREFLIGHT_READY",
                    "queue_ticker_count": 1,
                    "covered_queue_ticker_count": 1,
                    "missing_queue_tickers": "",
                    "rerun_out_dir_clear": True,
                    "agent_reviews_json": "/tmp/reviews.json",
                }
            ]
        ),
        live_probe_summary=pd.DataFrame(
            [
                {
                    "green_ready_orders": 1,
                    "market_session_open": True,
                    "expectancy_summary_status": "PASS",
                }
            ]
        ),
        multi_date_readiness_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_MULTI_DATE_AND_LIVE_GREEN_EVIDENCE",
                    "validation_date_count": 2,
                    "latest_live_probe_date": "2026-05-22",
                    "live_probe_dates": "2026-05-22",
                    "dated_yellow_target_candidates": 2,
                    "live_yellow_recheck_rows": 0,
                    "live_green_ready_orders": 1,
                }
            ]
        ),
        actionability_surface_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_AND_YELLOW_SURFACES_SEPARATED",
                    "target_ready_to_enter_rows": 0,
                    "target_missing_entry_type_rows": 0,
                    "target_missing_plain_language_leg_rows": 0,
                }
            ]
        ),
        action_surface_underlying_quality_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS",
                    "ticket_bad_underlying_rows": 0,
                    "market_open_recheck_bad_underlying_rows": 0,
                    "focus_bad_actionable_rows": 0,
                }
            ]
        ),
        green_ticket_execution_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKETS_EXECUTION_READY",
                    "valid_green_ticket_rows": 1,
                    "invalid_green_ticket_rows": 0,
                }
            ]
        ),
        ticket_expectancy_proof=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
        paths=paths,
    )

    assert set(goal_audit["status"].tolist()) == {"PROVEN", "ACHIEVED"}
    assert goal_audit.loc[
        goal_audit["requirement"].eq("separate yellow target orders from green send-now orders"),
        "status",
    ].tolist() == ["PROVEN"]
    assert goal_audit.loc[
        goal_audit["requirement"].eq("do not claim $10k/month readiness without evidence"),
        "status",
    ].tolist() == ["PROVEN"]

    verdict = audit.build_completion_verdict(
        goal_audit=goal_audit,
        market_open_execution_packet=pd.DataFrame(
            [
                {
                    "status": "green_orders_present_verify_ticket_scoped_expectancy",
                    "next_regular_session_start": "2026-05-26T06:30:00-07:00",
                }
            ]
        ),
        expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "positive_expectancy_ready_for_monthly_claim_review",
                    "monthly_claim_allowed": True,
                }
            ]
        ),
        ticket_expectancy_proof_packet=pd.DataFrame(
            [
                {
                    "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                }
            ]
        ),
    )

    assert verdict["status"].tolist() == ["COMPLETE"]
    assert verdict["can_mark_goal_complete"].tolist() == [True]
    assert verdict["update_goal_action"].tolist() == ["call_update_goal_complete"]


def test_post_rerun_verification_passes_only_when_all_completion_evidence_agrees() -> None:
    plan = pd.DataFrame(
        [
            {
                "date": "2026-05-22",
                "status": "VERIFY_GREEN_ORDERS_AND_EXPECTANCY",
                "rerun_command": "python3 -m uwos.options_agent --date 2026-05-22",
                "green_ticket_file": "/tmp/live/green_trade_tickets.csv",
                "trade_ticket_file": "/tmp/live/trade_tickets.csv",
                "execution_readiness_file": "/tmp/live/execution_readiness.csv",
                "expectancy_file": "/tmp/live/expectancy_evidence.csv",
            }
        ]
    )
    live_summary = pd.DataFrame([{"market_session_open": True}])
    green_proof = pd.DataFrame(
        [
            {
                "status": "PASS_GREEN_TICKETS_EXECUTION_READY",
                "green_ticket_rows": 2,
                "valid_green_ticket_rows": 2,
                "invalid_green_ticket_rows": 0,
            }
        ]
    )
    ticket_expectancy = pd.DataFrame(
        [
            {
                "status": "PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
                "green_ticker_count": 2,
            }
        ]
    )
    completion = pd.DataFrame(
        [
            {
                "status": "COMPLETE",
                "can_mark_goal_complete": True,
                "update_goal_action": "call_update_goal_complete",
                "monthly_claim_allowed": True,
            }
        ]
    )

    packet = audit.build_post_rerun_verification_packet(
        market_session_verification_plan=plan,
        live_probe_summary=live_summary,
        green_ticket_execution_proof=green_proof,
        ticket_expectancy_proof=ticket_expectancy,
        completion_verdict=completion,
        audit_regeneration_command="python3 -m uwos.options_agent.audit --summary-csv /tmp/summary.csv",
    )

    assert packet["status"].tolist() == ["PASS_READY_TO_COMPLETE_GOAL"]
    assert packet["green_ticket_rows"].tolist() == [2]
    assert packet["valid_green_ticket_rows"].tolist() == [2]
    assert packet["green_ticker_count"].tolist() == [2]
    assert packet["update_goal_action"].tolist() == ["call_update_goal_complete"]
    assert "--summary-csv /tmp/summary.csv" in packet["audit_regeneration_command"].iloc[0]
    assert "/tmp/live/expectancy_evidence.csv" in packet["evidence_files"].iloc[0]


def test_audit_summary_csv_helper_loads_run_dirs(tmp_path: Path) -> None:
    run_dir = tmp_path / "out" / "options_agent" / "multidate_quality_v017_2026-05-22"
    run_dir.mkdir(parents=True)
    summary = tmp_path / "summary.csv"
    pd.DataFrame([{"source_dir": str(run_dir)}]).to_csv(summary, index=False)

    assert audit._run_dirs_from_summary_csv([str(summary)]) == [run_dir.resolve()]


def test_market_open_runner_blocks_closed_market_dry_run(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "plan.csv"
    out_dir = tmp_path / "market_open_rerun"
    pd.DataFrame(
        [
            {
                "status": "WAITING_FOR_REGULAR_SESSION",
                "rerun_out_dir": str(out_dir),
                "rerun_command": (
                    "python3 -m uwos.options_agent --date 2026-05-22 --base-dir /tmp/base "
                    f"--out-dir {out_dir} --live-schwab --live-portfolio "
                    "--agent-reviews-json /tmp/reviews.json"
                ),
            }
        ]
    ).to_csv(plan, index=False)
    monkeypatch.setattr(core, "is_regular_market_session_open", lambda: False)

    result = market_open_runner.run_from_plan(plan_csv=plan, dry_run=True)

    assert result.status == "BLOCKED"
    assert "regular_market_session_open=false" in result.errors


def test_market_open_runner_dry_run_ready_with_live_flags_and_fresh_out_dir(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "plan.csv"
    post = tmp_path / "post.csv"
    out_dir = tmp_path / "market_open_rerun"
    pd.DataFrame(
        [
            {
                "status": "WAITING_FOR_REGULAR_SESSION",
                "rerun_out_dir": str(out_dir),
                "rerun_command": (
                    "python3 -m uwos.options_agent --date 2026-05-22 --base-dir /tmp/base "
                    f"--out-dir {out_dir} --live-schwab --live-portfolio "
                    "--agent-reviews-json /tmp/reviews.json"
                ),
            }
        ]
    ).to_csv(plan, index=False)
    pd.DataFrame(
        [
            {
                "audit_regeneration_command": (
                    "python3 -m uwos.options_agent.audit --base-dir /tmp/base "
                    "--live-probe-dir /tmp/live --output-prefix /tmp/audit"
                )
            }
        ]
    ).to_csv(post, index=False)
    monkeypatch.setattr(core, "is_regular_market_session_open", lambda: True)

    result = market_open_runner.run_from_plan(plan_csv=plan, post_rerun_csv=post, dry_run=True)

    assert result.status == "DRY_RUN_READY"
    assert "--live-schwab" in result.rerun_command
    assert "--live-portfolio" in result.rerun_command
    assert "--agent-reviews-json" in result.rerun_command
    assert result.audit_command[:3] == ("python3", "-m", "uwos.options_agent.audit")


def test_market_open_runner_reports_no_go_when_post_rerun_packet_still_blocks(tmp_path: Path, monkeypatch) -> None:
    plan = tmp_path / "plan.csv"
    post = tmp_path / "post.csv"
    out_dir = tmp_path / "market_open_rerun"
    pd.DataFrame(
        [
            {
                "status": "WAITING_FOR_REGULAR_SESSION",
                "rerun_out_dir": str(out_dir),
                "rerun_command": (
                    "python3 -m uwos.options_agent --date 2026-05-22 --base-dir /tmp/base "
                    f"--out-dir {out_dir} --live-schwab --live-portfolio "
                    "--agent-reviews-json /tmp/reviews.json"
                ),
            }
        ]
    ).to_csv(plan, index=False)
    pd.DataFrame(
        [
            {
                "status": "FAIL_NO_GREEN_TICKETS_AFTER_RERUN",
                "can_mark_goal_complete": False,
                "update_goal_action": "do_not_call_update_goal_complete",
                "audit_regeneration_command": (
                    "python3 -m uwos.options_agent.audit --base-dir /tmp/base "
                    "--live-probe-dir /tmp/live --output-prefix /tmp/audit"
                ),
            }
        ]
    ).to_csv(post, index=False)
    calls: list[tuple[str, ...]] = []

    class Completed:
        returncode = 0

    def fake_run(command, cwd=None):
        calls.append(tuple(command))
        return Completed()

    monkeypatch.setattr(core, "is_regular_market_session_open", lambda: True)
    monkeypatch.setattr(market_open_runner.subprocess, "run", fake_run)

    result = market_open_runner.run_from_plan(plan_csv=plan, post_rerun_csv=post)

    assert result.status == "COMPLETED_NOT_READY"
    assert result.post_rerun_status == "FAIL_NO_GREEN_TICKETS_AFTER_RERUN"
    assert "can_mark_goal_complete=false" in result.errors
    assert result.update_goal_action == "do_not_call_update_goal_complete"
    assert len(calls) == 2


def test_market_open_runner_reports_ready_only_when_post_rerun_packet_allows_completion(
    tmp_path: Path, monkeypatch
) -> None:
    plan = tmp_path / "plan.csv"
    post = tmp_path / "post.csv"
    out_dir = tmp_path / "market_open_rerun"
    pd.DataFrame(
        [
            {
                "status": "WAITING_FOR_REGULAR_SESSION",
                "rerun_out_dir": str(out_dir),
                "rerun_command": (
                    "python3 -m uwos.options_agent --date 2026-05-22 --base-dir /tmp/base "
                    f"--out-dir {out_dir} --live-schwab --live-portfolio "
                    "--agent-reviews-json /tmp/reviews.json"
                ),
            }
        ]
    ).to_csv(plan, index=False)
    pd.DataFrame(
        [
            {
                "status": "PASS_READY_TO_COMPLETE_GOAL",
                "can_mark_goal_complete": True,
                "update_goal_action": "call_update_goal_complete",
                "audit_regeneration_command": (
                    "python3 -m uwos.options_agent.audit --base-dir /tmp/base "
                    "--live-probe-dir /tmp/live --output-prefix /tmp/audit"
                ),
            }
        ]
    ).to_csv(post, index=False)

    class Completed:
        returncode = 0

    monkeypatch.setattr(core, "is_regular_market_session_open", lambda: True)
    monkeypatch.setattr(market_open_runner.subprocess, "run", lambda command, cwd=None: Completed())

    result = market_open_runner.run_from_plan(plan_csv=plan, post_rerun_csv=post)

    assert result.status == "COMPLETED_READY_TO_COMPLETE_GOAL"
    assert result.post_rerun_status == "PASS_READY_TO_COMPLETE_GOAL"
    assert result.can_mark_goal_complete is True
    assert result.update_goal_action == "call_update_goal_complete"


def test_summarize_run_counts_visible_yellow_ticket_rows_not_internal_candidates(tmp_path: Path) -> None:
    run_dir = tmp_path / "out" / "options_agent" / "current_code_debit_scout_v047_agentic_2026-05-15"
    run_dir.mkdir(parents=True)
    (run_dir / "options_agent_manifest_2026-05-15.json").write_text(
        json.dumps(
            {
                "as_of": "2026-05-15",
                "row_counts": {
                    "trade_tickets": 2,
                    "ready_to_enter": 0,
                    "target_order_candidates": 4,
                    "target_order_ticket_rows": 2,
                    "market_open_recheck_queue": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {"ticker": "PEP", "target_order_status": "target_order_candidate", "ready_to_enter": False},
            {"ticker": "BX", "target_order_status": "target_order_candidate", "ready_to_enter": False},
        ]
    ).to_csv(run_dir / "trade_tickets.csv", index=False)

    summary = audit.summarize_run(run_dir)

    assert summary["trade_ticket_rows"] == 2
    assert summary["yellow_target_candidates"] == 2


def test_recompute_live_capture_enforces_profitability_calibration(tmp_path: Path) -> None:
    source = tmp_path / "source_live_capture"
    output = tmp_path / "current_code_recompute"
    source.mkdir()
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(
            json.dumps({"ticker": "AMAT", "realized_pnl": pnl, "strategy": "vertical_spread"})
            for pnl in (120.0, 90.0, -20.0)
        )
        + "\n",
        encoding="utf-8",
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    (source / "options_agent_manifest_2026-05-22.json").write_text(
        json.dumps(
            {
                "as_of": "2026-05-22",
                "pipeline_name": "Options Agent",
                "pipeline_version": "old",
                "mode": "agentic_synthesis_pass",
                "source_root": str(tmp_path),
                "source_dir": str(tmp_path / "2026-05-22"),
                "agents": [],
                "artifacts": {},
                "row_counts": {
                    "decision_board": 1,
                    "trade_tickets": 1,
                    "green_trade_tickets": 0,
                    "target_order_ticket_rows": 1,
                    "market_open_recheck_queue": 0,
                    "ready_to_enter": 0,
                },
                "execution_context": context,
                "agentic_orchestration": {"status": "reviews_ingested", "subagent_task_count": 5},
                "market_regime": {"regime": "risk_off"},
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "AMAT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "bias": "bearish",
                "structure": "bear call spread",
                "full_ticket": "SELL 1 AMAT 2026-05-29 467.5 Call / BUY 1 AMAT 2026-05-29 472.5 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 AMAT 2026-05-29 467.5 Call / BUY 1 AMAT 2026-05-29 472.5 Call @ 1.50 CREDIT",
                "expiry": "2026-05-29",
                "sell_leg": "SELL 1 AMAT 2026-05-29 467.5 Call",
                "buy_leg": "BUY 1 AMAT 2026-05-29 472.5 Call",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "external_agent_review_agents": "catalyst_news; macro_regime; portfolio_management; skeptic; structure_builder",
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "synthesis_score": 140.0,
                "score": 75.0,
                "target_exit": 0.30,
                "invalidation": "underlying violates breakeven",
                "sizing_note": "risk budget supports 5 contract(s)",
                "visible_in_final_board": True,
            }
        ]
    ).to_csv(source / "final_recommendations.csv", index=False)
    pd.DataFrame(columns=["ticker", "reason"]).to_csv(source / "no_trade_audit.csv", index=False)
    pd.DataFrame(columns=["ticker", "coverage_status"]).to_csv(source / "ticker_coverage_audit.csv", index=False)
    pd.DataFrame([{"ticker": "AMAT", "live_market_quality_status": "PASS", "quality_gate_reason": ""}]).to_csv(
        source / "live_spread_quality_audit.csv",
        index=False,
    )
    (source / "market_regime.json").write_text(json.dumps({"regime": "risk_off"}), encoding="utf-8")

    paths = audit.recompute_live_capture(source_dir=source, output_dir=output, base_dir=tmp_path)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    calibration = pd.read_csv(paths["profitability_calibration"])
    green = pd.read_csv(paths["green_trade_tickets"])
    tickets = pd.read_csv(paths["trade_tickets"])
    readiness = pd.read_csv(paths["execution_readiness"])
    coverage = pd.read_csv(paths["coverage_audit"])
    report = paths["report"].read_text(encoding="utf-8")

    assert manifest["mode"] == "captured_market_open_live_recompute_current_code"
    assert manifest["captured_live_recompute"]["fresh_quote_pull"] is False
    assert manifest["row_counts"]["profitability_calibration"] == len(calibration)
    assert manifest["row_counts"]["ready_to_enter"] == 0
    assert manifest["row_counts"]["green_trade_tickets"] == 0
    assert green.empty
    assert tickets["ticker"].tolist() == ["AMAT"]
    assert readiness.loc[readiness["gate"].eq("ready_trade_tickets"), "status"].tolist() == ["BLOCK"]
    assert "captured market-open live recompute" in manifest["warnings"][0]
    assert coverage.loc[coverage["ticker"].eq("AMAT"), "coverage_status"].tolist() == ["TARGET_ORDER_CANDIDATE"]
    assert "Captured-live recompute" in report
    assert "Monthly Readiness Gate" not in report
    assert "Green send-now rows are order-entry candidates only" not in report
    assert "## Target Orders" in report
    assert "| AMAT |" in report


def test_monthly_claim_requirement_status_passes_when_ticket_scoped_evidence_supports_claim() -> None:
    status, gap = audit._monthly_claim_requirement_status(
        ticket_expectancy_status="PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
        monthly_statuses=["proven"],
        expectancy_statuses=["PASS"],
        live_expectancy=["PASS"],
        live_green_ready_orders=2,
        green_ticket_status="PASS_GREEN_TICKETS_EXECUTION_READY",
    )

    assert status == "PROVEN"
    assert gap == ""


def test_monthly_claim_requirement_status_needs_review_for_ambiguous_positive_claim() -> None:
    status, gap = audit._monthly_claim_requirement_status(
        ticket_expectancy_status="PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
        monthly_statuses=["proven"],
        expectancy_statuses=["PASS"],
        live_expectancy=["PASS"],
        live_green_ready_orders=2,
        green_ticket_status="FAIL_INVALID_GREEN_TICKET_ROWS",
    )

    assert status == "NEEDS_REVIEW"
    assert "neither cleanly blocked nor fully supported" in gap


def test_execution_readiness_gap_does_not_request_duplicate_live_run_after_green_ticket() -> None:
    gap = audit._execution_readiness_remaining_gap(
        green_ticket_status="PASS_GREEN_TICKETS_EXECUTION_READY",
        live_green_ready_orders=1,
        ticket_expectancy_status="BLOCK_GREEN_TICKERS_WITHOUT_ACTUAL_FORWARD_EXPECTANCY",
        live_expectancy=["BLOCK"],
    )

    assert "Order-entry readiness is proven" in gap
    assert "positive structure-aligned actual/forward expectancy evidence" in gap
    assert "do not request another live run" in gap


def test_execution_readiness_gap_clears_after_ticket_expectancy_passes() -> None:
    gap = audit._execution_readiness_remaining_gap(
        green_ticket_status="PASS_GREEN_TICKETS_EXECUTION_READY",
        live_green_ready_orders=1,
        ticket_expectancy_status="PASS_GREEN_TICKER_EXPECTANCY_COVERAGE",
        live_expectancy=["BLOCK"],
    )

    assert gap == ""


def test_agentic_review_coverage_threshold_blocks_execution_ticket() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "trade_plan": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 3,
            }
        ]
    )
    final["underlying_quality_tier"] = "core"
    final = _mark_strategy_expectancy_pass(final)
    thin_review_context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=1,
        external_review_agent_count=1,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )

    decision = core.synthesize_decision_board(
        final,
        market_regime={"regime": "mixed"},
        execution_context=thin_review_context,
    )

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["execution_status"].tolist() == ["needs_agentic_review"]
    assert "agentic_review_coverage_below_threshold" in decision["execution_blockers"].iloc[0]
    tickets = core.build_trade_tickets(decision)
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]


def test_agentic_review_lane_coverage_can_pass_when_broad_universe_coverage_is_low() -> None:
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=3864,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )
    readiness = core.build_execution_readiness(pd.DataFrame(columns=["ready_to_enter"]), context)
    agentic_gate = readiness[readiness["gate"].eq("agentic_reviews")].iloc[0]

    assert context["agentic_reviews_ready"] is True
    assert context["agentic_review_coverage_basis"] == "subagent_lanes"
    assert context["agentic_review_coverage_pct"] == 1.0
    assert context["broad_review_coverage_pct"] == 0.0129
    assert agentic_gate["status"] == "PASS"
    assert "coverage_basis=subagent_lanes" in agentic_gate["detail"]
    assert "coverage=1.0" in agentic_gate["detail"]
    assert "broad_universe_coverage=0.0129" in agentic_gate["detail"]


def test_retry_normalized_subagent_reviews_count_as_ticket_lanes() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": "SHOP",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 SHOP 2026-07-17 105 Put @ 3.30 CREDIT",
                "trade_plan": "SELL 1 SHOP 2026-07-17 105 Put @ 3.30 CREDIT",
                "entry_limit": 3.3,
                "suggested_contracts": 5,
                "max_profit": 330.0,
                "max_loss": 10170.0,
                "credit_width_ratio": 0.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 36,
                "actual_forward_strategy_expectancy_avg_pnl": 91.57,
                "actual_forward_strategy_expectancy_profit_factor": 1.685,
                "profitability_calibration_status": "PASS",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_replay_status": "PASS",
            }
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "ticker": "SHOP",
                "agent": agent,
                "agent_type": "subagent_retry_normalized",
                "verdict": "supportive",
                "objective_blocker": False,
                "note": "retry lane supports",
            }
            for agent in ("catalyst_news", "macro_regime", "structure_builder", "skeptic", "portfolio_management")
        ]
    )
    reviewed = core.apply_agent_reviews(priced, reviews)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 200_000, "cash": 200_000},
        research_task_count=3916,
        external_review_count=len(reviews),
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/agentic_reviews.json"),
    )

    decision = core.synthesize_decision_board(reviewed, market_regime={"regime": "risk_on"}, execution_context=context)

    assert reviewed["external_agent_distinct_review_count"].tolist() == [5]
    assert "ticker_agentic_review_coverage_below_threshold" not in decision["execution_blockers"].iloc[0]


def test_agentic_review_contract_requires_required_ticker_lane_coverage(tmp_path: Path) -> None:
    paths = output_paths("2026-05-22", root=tmp_path)
    dispatch = core.build_agent_dispatch_plan(
        {"tasks": [{"ticker": "AAPL", "score": 80}, {"ticker": "INTC", "score": 79}]},
        "2026-05-22",
        paths,
    )
    reviews = pd.DataFrame(
        [
            {"ticker": "AAPL", "agent": agent, "agent_type": "subagent"}
            for agent in ("catalyst_news", "macro_regime", "structure_builder", "skeptic")
        ]
        + [
            {"ticker": "INTC", "agent": agent, "agent_type": "subagent"}
            for agent in ("catalyst_news", "macro_regime", "structure_builder")
        ]
    )

    coverage = core.build_agentic_review_contract_coverage(dispatch, reviews)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=len(reviews),
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=paths["agentic_reviews"],
        required_review_ticker_count=coverage["required_review_ticker_count"],
        required_review_ticker_reviewed_count=coverage["required_review_ticker_reviewed_count"],
        required_review_ticker_missing_count=coverage["required_review_ticker_missing_count"],
        required_review_ticker_coverage_pct=coverage["required_review_ticker_coverage_pct"],
    )

    assert coverage["status"] == "block"
    assert coverage["required_review_missing_ticker_examples"] == ["INTC"]
    assert context["agentic_reviews_ready"] is False
    assert context["agentic_review_coverage_basis"] == "subagent_lanes_and_required_tickers"
    assert "agentic_required_ticker_reviews_missing" in context["run_gate_blockers"]


def test_agentic_ticker_lane_coverage_does_not_count_exact_contract_reviews(tmp_path: Path) -> None:
    paths = output_paths("2026-05-22", root=tmp_path)
    dispatch = core.build_agent_dispatch_plan(
        {"tasks": [{"ticker": "AAPL", "score": 80}]},
        "2026-05-22",
        paths,
    )
    reviews = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "agent": agent,
                "agent_type": "subagent",
                "contract_specific": agent == "catalyst_news",
            }
            for agent in ("catalyst_news", "macro_regime", "structure_builder", "skeptic")
        ]
    )

    coverage = core.build_agentic_review_contract_coverage(dispatch, reviews)

    assert coverage["status"] == "block"
    assert coverage["required_review_missing_ticker_examples"] == ["AAPL"]


def test_per_ticket_lane_gate_does_not_count_exact_contract_agents_as_generic_lanes() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "expiry": "2026-08-21",
                "dte": 30,
                "buy_leg": "BUY 1 AAPL 2026-08-21 200 Call",
                "sell_leg": "SELL 1 AAPL 2026-08-21 205 Call",
                "trade_plan": "BUY 1 AAPL 2026-08-21 200 Call / SELL 1 AAPL 2026-08-21 205 Call @ 1.00 DEBIT",
                "entry_limit": 1.00,
                "target_entry": 1.20,
                "max_profit": 400.0,
                "max_loss": 100.0,
                "suggested_contracts": 1,
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "live_probability_proxy": 0.55,
                "live_quote_width_pct": 0.05,
                "live_theta_burn_pct": 0.01,
                "live_breakeven_expected_move_ratio": 0.40,
            }
        ]
    )
    key = core.contract_review_key(priced.iloc[0])
    reviews = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "agent": agent,
                "agent_type": "subagent",
                "verdict": "supportive",
                "contract_specific": False,
            }
            for agent in ("catalyst_news", "macro_regime")
        ]
        + [
            {
                "ticker": "AAPL",
                "agent": agent,
                "agent_type": "subagent",
                "verdict": "supportive",
                "contract_specific": True,
                "contract_key": key,
                "strategy_route": "bull_call_debit",
                "expiry": "2026-08-21",
                "trade_plan": priced.iloc[0]["trade_plan"],
            }
            for agent in core.CONTRACT_REVIEW_REQUIRED_AGENTS
        ]
    )

    reviewed = core.apply_agent_reviews(priced, reviews)
    reviewed = _mark_strategy_expectancy_pass(reviewed, sample=core.MIN_EXPECTANCY_SAMPLE_SIZE)
    row = reviewed.iloc[0].to_dict()
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=len(reviews),
        external_review_agent_count=4,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/agentic_reviews.json"),
        market_session_open=True,
    )
    blockers = core._execution_blockers_for_row(
        row,
        "ready",
        row["trade_plan"],
        row["entry_limit"],
        row["suggested_contracts"],
        context,
    )

    assert row["contract_review_status"] == "PASS"
    assert row["external_agent_distinct_review_count"] == 2
    assert "ticker_agentic_review_coverage_below_threshold" in blockers


def test_execution_readiness_distinguishes_no_send_now_orders_from_ready_pipeline() -> None:
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=1,
        external_review_count=4,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )
    readiness = core.build_execution_readiness(pd.DataFrame(columns=["ready_to_enter"]), context)

    summary = core.summarize_execution_readiness(readiness)

    assert summary["status"] == "gates_pass_no_send_now_orders"
    assert summary["blocking_gates"] == ["ready_trade_tickets"]


def test_target_order_candidates_exclude_unvalidated_and_low_quality_underlyings() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "NFLX",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 NFLX 2026-06-18 92 Call / BUY 1 NFLX 2026-06-18 97 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 NFLX 2026-06-18 92 Call / BUY 1 NFLX 2026-06-18 97 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "agent_support_count": 3,
            },
            {
                "ticker": "OKLO",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 OKLO 2026-06-18 75 Call / BUY 1 OKLO 2026-06-18 80 Call @ 0.94 CREDIT",
                "trade_plan": "SELL 1 OKLO 2026-06-18 75 Call / BUY 1 OKLO 2026-06-18 80 Call @ 0.94 CREDIT",
                "entry_limit": 0.94,
                "suggested_contracts": 1,
                "max_profit": 94.0,
                "max_loss": 406.0,
                "credit_width_ratio": 0.188,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "speculative",
                "underlying_quality_reason": "marketcap_below_20000000000",
                "agent_support_count": 3,
            },
            {
                "ticker": "DATED",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 DATED 2026-05-29 47 Put / BUY 1 DATED 2026-05-29 46 Put @ 0.32 CREDIT",
                "trade_plan": "SELL 1 DATED 2026-05-29 47 Put / BUY 1 DATED 2026-05-29 46 Put @ 0.32 CREDIT",
                "entry_limit": 0.32,
                "suggested_contracts": 1,
                "max_profit": 32.0,
                "max_loss": 68.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "",
                "status_reason": "dated UW EOD quote; refresh Schwab chain before entry",
                "underlying_quality_tier": "core",
                "agent_support_count": 3,
                "external_agent_review_count": 4,
                "external_agent_distinct_review_count": 4,
            },
            {
                "ticker": "CAUTION",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 CAUTION 2026-05-29 47 Put / BUY 1 CAUTION 2026-05-29 46 Put @ 0.32 CREDIT",
                "trade_plan": "SELL 1 CAUTION 2026-05-29 47 Put / BUY 1 CAUTION 2026-05-29 46 Put @ 0.32 CREDIT",
                "entry_limit": 0.32,
                "suggested_contracts": 3,
                "max_profit": 32.0,
                "max_loss": 68.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "",
                "status_reason": "dated UW EOD quote; refresh Schwab chain before entry; external agent caution",
                "underlying_quality_tier": "core",
                "agent_caution_count": 5,
                "external_agent_distinct_review_count": 4,
            },
            {
                "ticker": "CHAIN",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 CHAIN 2026-05-29 47 Put / BUY 1 CHAIN 2026-05-29 46 Put @ 0.32 CREDIT",
                "trade_plan": "SELL 1 CHAIN 2026-05-29 47 Put / BUY 1 CHAIN 2026-05-29 46 Put @ 0.32 CREDIT",
                "entry_limit": 0.32,
                "suggested_contracts": 1,
                "max_profit": 32.0,
                "max_loss": 68.0,
                "credit_width_ratio": 0.32,
                "trade_quality_status": "reviewable",
                "live_validation_status": "CHAIN_UNAVAILABLE",
                "underlying_quality_tier": "core",
                "agent_support_count": 3,
            },
            {
                "ticker": "GM",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 GM 2026-06-18 76 Call / BUY 1 GM 2026-06-18 81 Call @ 0.92 CREDIT",
                "trade_plan": "SELL 1 GM 2026-06-18 76 Call / BUY 1 GM 2026-06-18 81 Call @ 0.92 CREDIT",
                "entry_limit": 0.92,
                "suggested_contracts": 1,
                "max_profit": 92.0,
                "max_loss": 408.0,
                "credit_width_ratio": 0.184,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "liquid",
                "agent_caution_count": 5,
            },
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    blocked_context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=Path("/tmp/snapshots"),
        portfolio_context={"status": "unavailable", "total_value": 0},
        research_task_count=10,
        external_review_count=1,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=blocked_context)
    tickets = core.build_trade_tickets(decision)
    monthly = core.build_monthly_feasibility(
        decision,
        tickets,
        {"monthly_profit_target": 10_000, "fresh_live_quotes_ready": False, "portfolio_ready": False},
        pd.DataFrame(),
    )

    assert tickets["ticker"].tolist() == ["NFLX"]
    green, yellow = core.split_trade_ticket_surfaces(tickets)
    assert green.empty
    assert yellow["ticker"].tolist() == ["NFLX"]
    assert monthly.loc[monthly["metric"].eq("ready_ticket_count"), "value"].tolist() == [0]
    assert monthly.loc[monthly["metric"].eq("ready_ticket_count"), "status"].tolist() == ["BLOCK"]
    assert monthly.loc[monthly["metric"].eq("target_order_candidate_count"), "value"].tolist() == [1]
    assert decision.loc[decision["ticker"].eq("OKLO"), "target_order_status"].tolist() == [
        "not_actionable_underlying_quality"
    ]
    assert decision.loc[decision["ticker"].eq("CHAIN"), "target_order_status"].tolist() == [
        "review_only_live_validation"
    ]
    assert decision.loc[decision["ticker"].eq("DATED"), "target_order_status"].tolist() == [
        "review_only_live_validation"
    ]
    assert decision.loc[decision["ticker"].eq("CAUTION"), "target_order_status"].tolist() == [
        "review_only_live_validation"
    ]
    assert decision.loc[decision["ticker"].eq("GM"), "target_order_status"].tolist() == [
        "not_actionable_underlying_quality"
    ]


def test_target_order_candidate_preserves_debit_plan_without_credit_width_gate() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 AAPL 2026-06-18 200 Call / SELL 1 AAPL 2026-06-18 205 Call @ 1.50 DEBIT",
                "trade_plan": "BUY 1 AAPL 2026-06-18 200 Call / SELL 1 AAPL 2026-06-18 205 Call @ 1.50 DEBIT",
                "entry_limit": 1.5,
                "suggested_contracts": 3,
                "max_profit": 350.0,
                "max_loss": 150.0,
                "credit_width_ratio": 0.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "external_agent_review_count": 4,
                "external_agent_distinct_review_count": 4,
                "external_agent_review_agents": "catalyst_news; macro_regime; structure_builder; skeptic",
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=Path("/tmp/snapshots"),
        portfolio_context={"status": "unavailable", "total_value": 0},
        research_task_count=1,
        external_review_count=4,
        agent_reviews_json=Path("/tmp/reviews.json"),
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["entry_type"].tolist() == ["DEBIT"]


def test_expectancy_evidence_blocks_monthly_target_on_negative_actual_history(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {"ticker": "WMT", "strategy": "vertical_spread", "realized_pnl": -100.0 if i % 2 == 0 else 25.0}
        for i in range(40)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = out / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "pnl_1x": 100.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
            for _ in range(40)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    decision = pd.DataFrame([{"ticker": "WMT"}])
    tickets = pd.DataFrame([{"ticker": "WMT", "max_profit": 100.0, "max_loss": 400.0}])

    evidence = core.build_expectancy_evidence(tmp_path, decision, tickets)
    summary = core.summarize_expectancy_evidence(evidence)
    feasibility = core.build_monthly_feasibility(
        decision,
        tickets,
        {
            "monthly_profit_target": 10_000,
            "fresh_live_quotes_ready": True,
            "portfolio_ready": True,
        },
        evidence,
    )

    assert "schwab_closed_trades" in summary["blocking_sources"]
    assert summary["status"] == "not_proven"
    assert evidence.loc[evidence["source"].eq("codexuw_replay_decision_pass"), "status"].tolist() == ["PASS"]
    assert evidence.loc[evidence["source"].eq("expectancy_summary"), "status"].tolist() == ["BLOCK"]
    assert feasibility.loc[feasibility["metric"].eq("expectancy_evidence"), "status"].tolist() == ["BLOCK"]


def test_codexuw_replay_loader_prefers_new_goal_replay_dirs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    old_dir = out / "codexuw_v2_backtest_fixture"
    new_dir = out / "codexuw_replay_goal_fixture"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "OLD",
                "strategy": "Bull Put Credit Spread",
                "entry_side": "credit",
                "strategy_kind": "Credit",
                "direction": "Bull Put",
                "dte": 30,
                "pnl_1x": -100.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
        ]
    ).to_csv(old_dir / "codexuw_replay_detail.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "NEW",
                "strategy": "Bull Put Credit Spread",
                "entry_side": "credit",
                "strategy_kind": "Credit",
                "direction": "Bull Put",
                "dte": 30,
                "pnl_1x": 100.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
        ]
    ).to_csv(new_dir / "codexuw_replay_detail.csv", index=False)
    os.utime(old_dir / "codexuw_replay_detail.csv", (1_700_000_000, 1_700_000_000))
    os.utime(new_dir / "codexuw_replay_detail.csv", (1_700_000_100, 1_700_000_100))

    replay, path, error = core._codexuw_profitability_replay_frame(out)
    evidence_rows = core._expectancy_from_replay_history(out, {"NEW"})

    assert error == ""
    assert path.endswith("codexuw_replay_goal_fixture/codexuw_replay_detail.csv")
    assert replay["ticker"].tolist() == ["NEW"]
    assert evidence_rows[1]["source_path"].endswith("codexuw_replay_goal_fixture/codexuw_replay_detail.csv")
    assert evidence_rows[1]["status"] == "WARN"


def test_codexuw_replay_loader_uses_only_manifest_heldout_partition(tmp_path: Path) -> None:
    replay_dir = tmp_path / "out" / "codexuw_replay_goal_fixture"
    replay_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "TRAIN",
                "asof": "2026-05-15",
                "strategy": "Bull Call Debit Spread",
                "entry_side": "DEBIT",
                "pnl_1x": 500.0,
                "exact_evaluated": True,
                "decision_pass": True,
            },
            {
                "ticker": "TEST",
                "asof": "2026-05-16",
                "strategy": "Bull Call Debit Spread",
                "entry_side": "DEBIT",
                "pnl_1x": -25.0,
                "exact_evaluated": True,
                "decision_pass": True,
            },
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    (replay_dir / "codexuw_replay_manifest.json").write_text(
        json.dumps({"split_day": "2026-05-15"}),
        encoding="utf-8",
    )

    replay, _, error = core._codexuw_profitability_replay_frame(tmp_path / "out")
    evidence = core._expectancy_from_replay_history(tmp_path / "out", {"TRAIN", "TEST"})

    assert error == ""
    assert replay["ticker"].tolist() == ["TEST"]
    assert replay["replay_validation_scope"].tolist() == ["heldout_test"]
    model = next(row for row in evidence if row["source"] == "codexuw_replay_decision_pass_model")
    assert model["sample_size"] == 1
    assert model["avg_pnl"] == -25.0


def test_codexuw_replay_loader_returns_empty_when_heldout_has_no_decision_rows(tmp_path: Path) -> None:
    replay_dir = tmp_path / "out" / "codexuw_replay_goal_fixture"
    replay_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "TEST",
                "asof": "2026-05-16",
                "strategy": "Bull Call Debit Spread",
                "entry_side": "DEBIT",
                "pnl_1x": -25.0,
                "exact_evaluated": True,
                "decision_pass": False,
            }
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    (replay_dir / "codexuw_replay_manifest.json").write_text(
        json.dumps({"split_day": "2026-05-15"}),
        encoding="utf-8",
    )

    replay, _, error = core._codexuw_profitability_replay_frame(tmp_path / "out")

    assert error == ""
    assert replay.empty
    assert "strategy_route" in replay.columns


def test_options_agent_walkforward_replay_selection_is_outcome_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "codexuw_replay_detail.csv"
    rows = []
    for day in range(1, 5):
        for ticker_idx in range(5):
            rows.append(
                {
                    "asof": f"2026-06-0{day}",
                    "exit_day": f"2026-06-0{day}",
                    "ticker": f"T{ticker_idx}",
                    "strategy": "Bear Call Credit Spread",
                    "regime": "range",
                    "exact_evaluated": True,
                    "decision_score": ticker_idx,
                    "pnl_1x": 100.0,
                }
            )
    rows.extend(
        [
            {
                "asof": "2026-06-05",
                "exit_day": "2026-06-06",
                "ticker": "HIGH",
                "strategy": "Bear Call Credit Spread",
                "regime": "range",
                "exact_evaluated": True,
                "decision_score": 3.0,
                "pnl_1x": -500.0,
            },
            {
                "asof": "2026-06-05",
                "exit_day": "2026-06-06",
                "ticker": "MID",
                "strategy": "Bear Call Credit Spread",
                "regime": "range",
                "exact_evaluated": True,
                "decision_score": 2.0,
                "pnl_1x": 25.0,
            },
            {
                "asof": "2026-06-05",
                "exit_day": "2026-06-06",
                "ticker": "LOW",
                "strategy": "Bear Call Credit Spread",
                "regime": "range",
                "exact_evaluated": True,
                "decision_score": 1.0,
                "pnl_1x": 1_000.0,
            },
        ]
    )
    pd.DataFrame(rows).to_csv(replay_path, index=False)
    monkeypatch.setattr(
        core,
        "_codexuw_pinned_replay_path",
        lambda _out_root: (replay_path, "", True),
    )

    audit = core.build_options_agent_walkforward_replay_audit(tmp_path)

    assert audit["ticker"].tolist() == ["HIGH", "MID"]
    assert audit["realized_pnl"].tolist() == [-500.0, 25.0]
    assert audit["selection_rank_for_day"].tolist() == [1, 2]


def test_options_agent_walkforward_excludes_unavailable_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_path = tmp_path / "codexuw_replay_detail.csv"
    rows = []
    for day in range(1, 5):
        for ticker_idx in range(5):
            rows.append(
                {
                    "asof": f"2026-06-0{day}",
                    "exit_day": "2026-06-10",
                    "ticker": f"T{ticker_idx}",
                    "strategy": "Bear Call Credit Spread",
                    "regime": "range",
                    "exact_evaluated": True,
                    "decision_score": ticker_idx,
                    "pnl_1x": 100.0,
                }
            )
    rows.append(
        {
            "asof": "2026-06-05",
            "exit_day": "2026-06-06",
            "ticker": "TEST",
            "strategy": "Bear Call Credit Spread",
            "regime": "range",
            "exact_evaluated": True,
            "decision_score": 10.0,
            "pnl_1x": 100.0,
        }
    )
    pd.DataFrame(rows).to_csv(replay_path, index=False)
    monkeypatch.setattr(
        core,
        "_codexuw_pinned_replay_path",
        lambda _out_root: (replay_path, "", True),
    )

    audit = core.build_options_agent_walkforward_replay_audit(tmp_path)

    assert audit.empty


def test_options_agent_walkforward_summary_requires_sample_and_day_diversity() -> None:
    insufficient = pd.DataFrame(
        [
            {"signal_date": f"2026-06-{day:02d}", "realized_pnl": 100.0 if idx % 2 == 0 else -25.0}
            for idx, day in enumerate(range(1, 11))
        ]
    )
    sufficient = pd.concat([insufficient, insufficient, insufficient], ignore_index=True)

    partial_summary = core.summarize_options_agent_walkforward_replay(insufficient)
    pass_summary = core.summarize_options_agent_walkforward_replay(sufficient)

    assert partial_summary["status"] == "warn"
    assert partial_summary["sample_size"] == 10
    assert pass_summary["status"] == "pass"
    assert pass_summary["sample_size"] == 30
    assert pass_summary["day_count"] == 10


def test_wheel_csp_replay_loader_combines_goal_replays_without_duplicate_signals(tmp_path: Path) -> None:
    out = tmp_path / "out"
    old_dir = out / "fresh_wheel_replay_old"
    new_dir = out / "fresh_wheel_replay_goal"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    old_path = old_dir / "fresh-wheel-replay-outcomes-old.csv"
    new_path = new_dir / "fresh-wheel-replay-outcomes-new.csv"
    base_duplicate = {
        "signal_date": "2026-05-01",
        "ticker": "DUP",
        "action": "OPEN_CSP",
        "option_symbol": "DUP260619P00090000",
        "expiry": "2026-06-19",
        "strike": 90,
        "entry_credit": 1.00,
        "dte": 30,
        "entry_date": "2026-05-01",
        "exit_date": "2026-06-01",
        "outcome_status": "scored",
    }
    pd.DataFrame(
        [
            {**base_duplicate, "pnl_per_contract": 10.0},
            {
                "signal_date": "2026-05-02",
                "ticker": "OLD",
                "action": "OPEN_CSP",
                "option_symbol": "OLD260619P00090000",
                "expiry": "2026-06-19",
                "strike": 90,
                "entry_credit": 1.25,
                "dte": 29,
                "entry_date": "2026-05-02",
                "exit_date": "2026-06-02",
                "outcome_status": "scored",
                "pnl_per_contract": 30.0,
            },
        ]
    ).to_csv(old_path, index=False)
    pd.DataFrame(
        [
            {**base_duplicate, "pnl_per_contract": 20.0},
            {
                "signal_date": "2026-05-03",
                "ticker": "NEW",
                "action": "OPEN_CSP",
                "option_symbol": "NEW260619P00090000",
                "expiry": "2026-06-19",
                "strike": 90,
                "entry_credit": 1.50,
                "dte": 28,
                "entry_date": "2026-05-03",
                "exit_date": "2026-06-03",
                "outcome_status": "scored",
                "pnl_per_contract": 40.0,
            },
        ]
    ).to_csv(new_path, index=False)
    os.utime(old_path, (1_700_000_000, 1_700_000_000))
    os.utime(new_path, (1_700_000_100, 1_700_000_100))

    replay, path, error = core._wheel_csp_profitability_replay_frame(out, as_of=dt.date(2026, 6, 9))

    assert error == ""
    assert "fresh_wheel_replay_goal" in path
    assert "fresh_wheel_replay_old" in path
    assert replay["ticker"].tolist() == ["DUP", "NEW", "OLD"]
    assert replay.loc[replay["ticker"].eq("DUP"), "pnl_1x"].tolist() == [20.0]


def test_wheel_csp_replay_counts_one_managed_lifecycle_per_contract() -> None:
    base = {
        "signal_date": "2026-05-01",
        "ticker": "WMT",
        "action": "OPEN_CSP",
        "option_symbol": "WMT260619P00090000",
        "expiry": "2026-06-19",
        "strike": 90,
        "entry_date": "2026-05-01",
    }
    terminal = pd.DataFrame(
        [
            {**base, "exit_date": "2026-05-08", "exit_reason": "horizon_mark", "pnl_1x": -25.0},
            {**base, "exit_date": "2026-05-15", "exit_reason": "horizon_mark", "pnl_1x": 10.0},
            {**base, "exit_date": "2026-05-20", "exit_reason": "hit_50pct_target", "pnl_1x": 100.0},
        ]
    )
    horizon_only = pd.DataFrame(
        [
            {**base, "exit_date": "2026-05-08", "exit_reason": "horizon_mark", "pnl_1x": -25.0},
            {**base, "exit_date": "2026-05-15", "exit_reason": "horizon_mark", "pnl_1x": 10.0},
            {**base, "exit_date": "2026-05-30", "exit_reason": "horizon_mark", "pnl_1x": 50.0},
        ]
    )

    terminal_result = core._dedupe_wheel_csp_replay_frame(terminal)
    horizon_result = core._dedupe_wheel_csp_replay_frame(horizon_only)

    assert terminal_result["exit_reason"].tolist() == ["hit_50pct_target"]
    assert terminal_result["pnl_1x"].tolist() == [100.0]
    assert horizon_result["exit_date"].tolist() == ["2026-05-15"]
    assert horizon_result["pnl_1x"].tolist() == [10.0]


def test_expectancy_evidence_does_not_pass_on_unrelated_positive_actual_history(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {"ticker": "AAPL", "strategy": "vertical_spread", "realized_pnl": 100.0}
        for _ in range(40)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = out / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "pnl_1x": 100.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
            for _ in range(40)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    evidence = core.build_expectancy_evidence(tmp_path, pd.DataFrame([{"ticker": "WMT"}]), pd.DataFrame())
    summary = core.summarize_expectancy_evidence(evidence)

    assert evidence.loc[evidence["source"].eq("schwab_closed_trades"), "status"].tolist() == ["BLOCK"]
    assert evidence.loc[evidence["source"].eq("schwab_closed_trades"), "matched_current_count"].tolist() == [0]
    assert evidence.loc[evidence["source"].eq("expectancy_summary"), "status"].tolist() == ["WARN"]
    assert summary["status"] == "mixed"


def test_expectancy_evidence_blocks_on_negative_actual_strategy_cohort(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {"ticker": "AAPL", "strategy": "vertical_spread", "realized_pnl": -100.0 if i % 2 == 0 else 20.0}
        for i in range(40)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = out / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {"ticker": "WMT", "pnl_1x": 100.0, "exact_evaluated": True, "decision_pass": True}
            for _ in range(40)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    decision = pd.DataFrame([{"ticker": "WMT"}])
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "trade_plan": "SELL 1 WMT 2026-06-19 95 Put / BUY 1 WMT 2026-06-19 90 Put @ 1.00 CREDIT",
                "ready_to_enter": True,
            }
        ]
    )

    evidence = core.build_expectancy_evidence(tmp_path, decision, tickets)
    summary = core.summarize_expectancy_evidence(evidence)
    cohort = evidence[evidence["source"].eq("schwab_closed_trades_strategy_cohort")].iloc[0]

    assert cohort["evidence_type"] == "actual_closed_trades_strategy_cohort"
    assert cohort["status"] == "BLOCK"
    assert cohort["sample_size"] == 40
    assert cohort["matched_current_count"] == 0
    assert "vertical_spread" in cohort["note"]
    assert evidence.loc[evidence["source"].eq("expectancy_summary"), "status"].tolist() == ["BLOCK"]
    assert summary["status"] == "not_proven"


def test_strategy_cohort_prefers_current_route_over_broad_vertical_family(tmp_path: Path) -> None:
    closed_path = tmp_path / "closed_trades_acct_3326.jsonl"
    closed_path.write_text("", encoding="utf-8")
    actual = pd.DataFrame(
        [
            {
                "ticker": f"BCD{i}",
                "canonical_ticker": f"BCD{i}",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "realized_pnl": 100.0 if i < 24 else -20.0,
            }
            for i in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
        + [
            {
                "ticker": f"BPC{i}",
                "canonical_ticker": f"BPC{i}",
                "strategy_route": "bull_put_credit",
                "strategy_family": "vertical_spread",
                "realized_pnl": -100.0,
            }
            for i in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )

    row = core._expectancy_from_closed_trades_strategy_cohort(
        closed_path,
        {"vertical_spread"},
        current_strategy_routes={"bull_call_debit"},
        actual_frame=actual,
    )

    assert row["status"] == "PASS"
    assert row["sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["avg_pnl"] == 76.0
    assert "current ticket strategy routes: bull_call_debit" in row["note"]
    assert "unrelated spread variants" in row["note"]
    assert "exact contracts still require profitability calibration" in row["note"]


def test_strategy_outcome_atlas_surfaces_positive_and_negative_strategy_families(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = []
    rows.extend({"ticker": "AAPL", "strategy": "short_put", "realized_pnl": 100.0} for _ in range(34))
    rows.extend({"ticker": "WMT", "strategy": "vertical_spread", "realized_pnl": -100.0 if i % 2 == 0 else 20.0} for i in range(40))
    rows.extend({"ticker": "GOOG", "strategy": "long_call", "realized_pnl": 90.0} for _ in range(3))
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "trade_plan": "SELL 1 WMT 2026-06-19 95 Put / BUY 1 WMT 2026-06-19 90 Put @ 1.00 CREDIT",
                "ready_to_enter": False,
            }
        ]
    )

    atlas = core.build_strategy_outcome_atlas(tmp_path, pd.DataFrame(), tickets)
    summary = core.summarize_strategy_outcome_atlas(atlas)
    family = atlas[atlas["scope"].eq("strategy_family")]
    current = atlas[atlas["scope"].eq("current_ticker_strategy")].iloc[0]

    assert family.loc[family["strategy_family"].eq("short_put"), "status"].tolist() == ["PASS"]
    assert family.loc[family["strategy_family"].eq("vertical_spread"), "status"].tolist() == ["BLOCK"]
    assert current["ticker"] == "WMT"
    assert current["strategy_family"] == "vertical_spread"
    assert current["status"] == "BLOCK"
    assert current["suggested_action"] == "do_not_promote_current_strategy_family"
    assert summary["positive_strategy_families"] == ["short_put"]
    assert summary["negative_current_strategy_families"] == ["vertical_spread"]
    assert summary["blocking_current_ticker_strategy_rows"] == 1


def test_strategy_outcome_atlas_names_positive_current_route_without_broad_family_negative(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {
            "ticker": f"BCD{idx}",
            "canonical_ticker": f"BCD{idx}",
            "strategy": "Bull Call Debit Spread",
            "strategy_family": "vertical_spread",
            "strategy_route": "bull_call_debit",
            "realized_pnl": 100.0 if idx < 24 else -20.0,
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    rows.extend(
        {
            "ticker": f"BPC{idx}",
            "canonical_ticker": f"BPC{idx}",
            "strategy": "Bull Put Credit Spread",
            "strategy_family": "vertical_spread",
            "strategy_route": "bull_put_credit",
            "realized_pnl": -100.0,
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "trade_plan": "BUY 1 GOOG 2026-07-17 370 Call / SELL 1 GOOG 2026-07-17 372.5 Call @ 0.60 DEBIT",
                "ready_to_enter": False,
            }
        ]
    )

    atlas = core.build_strategy_outcome_atlas(tmp_path, pd.DataFrame(), tickets)
    summary = core.summarize_strategy_outcome_atlas(atlas)
    route = atlas[atlas["scope"].eq("strategy_route") & atlas["strategy_route"].eq("bull_call_debit")].iloc[0]

    assert route["status"] == "PASS"
    assert route["current_ticket_count"] == 1
    assert summary["positive_current_strategy_routes"] == ["bull_call_debit"]
    assert summary["negative_current_strategy_families"] == []


def test_strategy_outcome_atlas_requires_ticker_strategy_support_for_current_rows(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [{"ticker": "AAPL", "strategy": "short_put", "realized_pnl": 100.0} for _ in range(34)]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "trade_plan": "SELL 1 GOOG 2026-06-19 95 Put @ 1.00 CREDIT",
                "strategy": "short_put",
                "ready_to_enter": False,
            }
        ]
    )

    atlas = core.build_strategy_outcome_atlas(tmp_path, pd.DataFrame(), tickets)
    current = atlas[atlas["scope"].eq("current_ticker_strategy")].iloc[0]
    summary = core.summarize_strategy_outcome_atlas(atlas)

    assert summary["positive_strategy_families"] == ["short_put"]
    assert current["ticker"] == "GOOG"
    assert current["strategy_family"] == "short_put"
    assert current["status"] == "BLOCK"
    assert current["sample_size"] == 0
    assert current["suggested_action"] == "keep_watch_only_until_ticker_strategy_outcomes_exist"


def test_expectancy_evidence_uses_project_schwab_closed_trades_for_overlay_root(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    overlay_root = project / "overlays" / "options_agent_fixture"
    overlay_root.mkdir(parents=True)
    closed_dir = project / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_path = closed_dir / "closed_trades_acct_3326.jsonl"
    closed_rows = [{"ticker": "WMT", "strategy": "vertical_spread", "realized_pnl": 100.0} for _ in range(40)]
    closed_path.write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(core, "project_root", lambda: project)

    evidence = core.build_expectancy_evidence(
        overlay_root,
        pd.DataFrame([{"ticker": "WMT"}]),
        pd.DataFrame([{"ticker": "WMT", "ready_to_enter": True}]),
    )
    closed = evidence[evidence["source"].eq("schwab_closed_trades")].iloc[0]

    assert closed["source_path"] == str(closed_path)
    assert closed["matched_current_tickers"] == "WMT"
    assert closed["status"] == "PASS"


def test_expectancy_evidence_uses_project_forward_and_replay_for_overlay_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    overlay_root = project / "overlays" / "options_agent_fixture"
    overlay_root.mkdir(parents=True)
    out = project / "out"
    out.mkdir(parents=True)
    execute_path = out / "codexuw_execute_outcome_ledger.csv"
    pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "realized_pnl": 100.0,
                "trade_key": f"WMT|{idx}",
            }
            for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    ).to_csv(execute_path, index=False)
    replay_dir = out / "codexuw_replay_fixture"
    replay_dir.mkdir()
    replay_path = replay_dir / "codexuw_replay_detail.csv"
    pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "pnl_1x": 50.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    ).to_csv(replay_path, index=False)
    monkeypatch.setattr(core, "project_root", lambda: project)

    evidence = core.build_expectancy_evidence(
        overlay_root,
        pd.DataFrame([{"ticker": "WMT"}]),
        pd.DataFrame([{"ticker": "WMT", "ready_to_enter": False}]),
    )
    execute = evidence[evidence["source"].eq("codexuw_execute_outcome_ledger")].iloc[0]
    replay = evidence[evidence["source"].eq("codexuw_replay_decision_pass")].iloc[0]

    assert execute["source_path"] == str(execute_path)
    assert execute["sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert execute["status"] == "PASS"
    assert replay["source_path"] == str(replay_path)
    assert replay["sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert replay["status"] == "PASS"


def test_profitability_calibration_uses_project_replay_for_overlay_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    overlay_root = project / "overlays" / "options_agent_fixture"
    overlay_root.mkdir(parents=True)
    replay_dir = project / "out" / "codexuw_replay_fixture"
    replay_dir.mkdir(parents=True)
    replay_path = replay_dir / "codexuw_replay_detail.csv"
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "strategy": "Bull Call Debit Spread",
                "entry_side": "DEBIT",
                "regime": "mixed",
                "dte": 10,
                "max_profit": 200.0,
                "max_loss": 100.0,
                "pnl_1x": 75.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    ).to_csv(replay_path, index=False)
    monkeypatch.setattr(core, "project_root", lambda: project)

    current = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 AAPL 2026-07-17 210 Call / SELL 1 AAPL 2026-07-17 220 Call @ 1.00 DEBIT",
                "entry_limit": 1.0,
                "max_profit": 200.0,
                "max_loss": 100.0,
                "dte": 10,
                "regime": "mixed",
            }
        ]
    )
    actual = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "canonical_ticker": "AAPL",
                "realized_pnl": 50.0,
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "entry_type": "DEBIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "debit_reward_risk_mid",
                "source": "fixture_actual",
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )

    calibration = core.build_profitability_calibration(
        overlay_root,
        current,
        actual_frame=actual,
    )
    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]

    assert row["status"] == "PASS"
    assert row["replay_bucket_status"] == "PASS"
    assert row["replay_bucket_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert str(replay_path) in row["source_path"]
    assert str(overlay_root / "out") not in row["source_path"]


def test_profitability_bucket_atlas_uses_project_replay_for_overlay_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    overlay_root = project / "overlays" / "options_agent_fixture"
    overlay_root.mkdir(parents=True)
    replay_dir = project / "out" / "codexuw_replay_fixture"
    replay_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "strategy": "Bull Call Debit Spread",
                "entry_side": "DEBIT",
                "regime": "mixed",
                "dte": 10,
                "max_profit": 200.0,
                "max_loss": 100.0,
                "pnl_1x": 75.0,
                "exact_evaluated": True,
                "decision_pass": True,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    monkeypatch.setattr(core, "project_root", lambda: project)

    actual = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "canonical_ticker": "AAPL",
                "realized_pnl": 50.0,
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "entry_type": "DEBIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "debit_reward_risk_mid",
                "source": "fixture_actual",
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    tickets = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 AAPL 2026-07-17 210 Call / SELL 1 AAPL 2026-07-17 220 Call @ 1.00 DEBIT",
                "entry_limit": 1.0,
                "max_profit": 200.0,
                "max_loss": 100.0,
                "dte": 10,
                "regime": "mixed",
            }
        ]
    )

    atlas = core.build_profitability_bucket_atlas(
        overlay_root,
        tickets,
        tickets,
        actual_frame=actual,
    )
    summary = core.summarize_profitability_bucket_atlas(atlas)
    row = atlas[atlas["current_ticket_count"].gt(0)].iloc[0]

    assert row["status"] == "PASS"
    assert row["actual_bucket_status"] == "PASS"
    assert row["replay_bucket_status"] == "PASS"
    assert str(project / "out") in row["source_path"]
    assert str(overlay_root / "out") not in row["source_path"]
    assert summary["current_pass_bucket_rows"] == 1


def test_expectancy_evidence_prefers_visible_ticket_tickers_over_broad_decision_board(tmp_path: Path) -> None:
    out = tmp_path / "out"
    closed_dir = out / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [{"ticker": "AAPL", "strategy": "vertical_spread", "realized_pnl": 100.0} for _ in range(40)]
    closed_rows.extend(
        {"ticker": "WMT", "strategy": "vertical_spread", "realized_pnl": -100.0 if i % 2 == 0 else 25.0}
        for i in range(40)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = out / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {"ticker": "AAPL", "pnl_1x": 100.0, "exact_evaluated": True, "decision_pass": True}
            for _ in range(40)
        ]
        + [
            {"ticker": "WMT", "pnl_1x": 100.0, "exact_evaluated": True, "decision_pass": True}
            for _ in range(40)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    decision = pd.DataFrame([{"ticker": "AAPL"}, {"ticker": "WMT"}])
    tickets = pd.DataFrame([{"ticker": "WMT", "ready_to_enter": False}])

    evidence = core.build_expectancy_evidence(tmp_path, decision, tickets)
    closed = evidence[evidence["source"].eq("schwab_closed_trades")].iloc[0]
    replay = evidence[evidence["source"].eq("codexuw_replay_decision_pass")].iloc[0]
    replay_model = evidence[evidence["source"].eq("codexuw_replay_decision_pass_model")].iloc[0]

    assert closed["matched_current_tickers"] == "WMT"
    assert closed["status"] == "BLOCK"
    assert replay["matched_current_tickers"] == "WMT"
    assert replay_model["sample_size"] == 80
    assert replay_model["matched_current_tickers"] == "WMT"
    assert evidence.loc[evidence["source"].eq("expectancy_summary"), "status"].tolist() == ["BLOCK"]


def test_monthly_feasibility_positive_status_still_disclaims_guarantee() -> None:
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 4, "status": "PASS", "note": ""},
            {"metric": "one_cycle_max_profit", "value": 12_000, "status": "PASS", "note": ""},
            {"metric": "cycles_needed_at_max_profit", "value": 1, "status": "PASS", "note": ""},
            {"metric": "expectancy_evidence", "value": 80, "status": "PASS", "note": ""},
            {"metric": "monthly_profit_target", "value": 10_000, "status": "INFO", "note": "User target; not a guarantee."},
        ]
    )

    summary = core.summarize_monthly_feasibility(monthly)

    assert summary["status"] == "capacity_and_expectancy_positive_not_guaranteed"
    assert summary["blocking_metrics"] == []
    assert "not guaranteed" in summary["note"]


def test_management_plan_separates_review_from_entry_ready_rows() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "REVIEW",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "recommendation_rank": 1,
                "status_reason": "news needs review",
                "entry_limit": 1.0,
                "target_exit": 0.35,
                "invalidation": "thesis breaks",
                "suggested_contracts": 0,
            },
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "recommendation_rank": 2,
                "live_validation_status": "PASS",
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "target_exit": 0.35,
                "invalidation": "underlying breaks breakeven",
                    "suggested_contracts": 5,
            },
        ]
    )
    final["underlying_quality_tier"] = "core"
    final = _mark_strategy_expectancy_pass(final, {"LIVE"})
    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    plan = core.build_management_plan(final, decision)

    assert plan["management_action"].tolist() == ["REVIEW", "ENTRY_READY"]
    assert "Do not enter" in plan.loc[plan["ticker"].eq("REVIEW"), "entry_condition"].iloc[0]
    assert "live quote" in plan.loc[plan["ticker"].eq("LIVE"), "entry_condition"].iloc[0]
    assert plan.loc[plan["ticker"].eq("LIVE"), "target_exit"].tolist() == [0.35]


def test_synthesis_ranking_prefers_live_validated_entry_over_raw_flow_score() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "HIGH",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "score": 95.0,
                "signal_premium": 10_000_000,
                "full_ticket": "",
                "entry_limit": "",
            },
            {
                "ticker": "LIVE",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "score": 70.0,
                "signal_premium": 1_000_000,
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
            },
            {
                "ticker": "RISK",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "quality_status": "qualified",
                "score": 70.0,
                "signal_premium": 900_000,
                "full_ticket": "SELL 1 X / BUY 1 Y @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "live_validation_status": "PASS",
                "portfolio_risk_flag": True,
            },
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "ticker": "HIGH",
                "agent": "skeptic",
                "agent_type": "built_in",
                "verdict": "caution",
                "objective_blocker": False,
                "portfolio_risk_only": False,
            },
            {
                "ticker": "RISK",
                "agent": "portfolio_risk",
                "agent_type": "built_in",
                "verdict": "caution",
                "objective_blocker": False,
                "portfolio_risk_only": True,
            },
        ]
    )

    ranked = core.apply_synthesis_ranking(final, reviews, top_trades=3)

    assert ranked["ticker"].tolist() == ["LIVE", "RISK", "HIGH"]
    assert ranked["recommendation_rank"].tolist() == [1, 2, 3]
    assert ranked.loc[ranked["ticker"].eq("RISK"), "agent_portfolio_risk_only_count"].tolist() == [1]
    assert ranked.loc[ranked["ticker"].eq("LIVE"), "synthesis_score"].iloc[0] == ranked.loc[
        ranked["ticker"].eq("RISK"), "synthesis_score"
    ].iloc[0]
    assert ranked.loc[ranked["ticker"].eq("RISK"), "agent_caution_count"].tolist() == [0]
    assert "account-context review(s) kept audit-only +0" in ranked.loc[
        ranked["ticker"].eq("RISK"), "synthesis_reason"
    ].iloc[0]


def test_position_sizing_annotates_risk_without_suppressing_trade() -> None:
    rows = core.apply_position_sizing(
        [
            {
                "ticker": "WMT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "live_validation_status": "PASS",
                "max_loss": 200.0,
                "portfolio_risk_flag": False,
            },
            {
                "ticker": "HOOD",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "live_validation_status": "PASS",
                "max_loss": 600.0,
                "portfolio_risk_flag": True,
                "portfolio_risk_note": "existing exposure",
            },
        ],
        {"status": "ok", "total_value": 100_000},
        {"sizing_stance": "normal"},
    )

    assert rows[0]["suggested_contracts"] == 2
    assert rows[0]["max_position_loss"] == 400.0
    assert rows[0]["account_risk_pct"] == 0.004
    assert rows[0]["recommendation_status"] == RecommendationStatus.ENTER.value
    assert rows[0]["portfolio_risk_flag"] is False
    assert "one-lot exceeds normal risk budget" not in str(rows[0].get("portfolio_risk_note") or "")
    assert rows[1]["suggested_contracts"] == 1
    assert rows[1]["sizing_risk_flag"] is True
    assert "one-lot exceeds normal risk budget" in rows[1]["portfolio_risk_note"]
    assert rows[1]["recommendation_status"] == RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value


def test_no_trade_audit_is_not_capped_by_top_trades() -> None:
    candidates = pd.DataFrame(
        [
            {
                "ticker": f"MISS{idx}",
                "bias": "bearish",
                "score": 70 - idx,
                "flow_reason": f"candidate {idx}",
                "quality_status": "rejected",
            }
            for idx in range(6)
        ]
    )

    audit = core.build_no_trade_audit(candidates, pd.DataFrame(), top_trades=2)

    assert audit["ticker"].tolist() == ["MISS0", "MISS1", "MISS2", "MISS3", "MISS4", "MISS5"]
    assert audit["hard_blocker"].tolist() == ["insufficient_score_or_neutral_bias"] * 6


def test_price_candidates_default_does_not_cap_qualified_candidates(tmp_path: Path) -> None:
    _write_minimal_uw_fixture(tmp_path)
    candidates = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "close": 100.0,
                "quality_status": "qualified",
                "score": 80 - idx * 0.1,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.75,
                "issue_type": "Common Stock",
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "flow_reason": f"candidate {idx}",
            }
            for idx in range(25)
        ]
    )

    priced = core.price_candidates(tmp_path / "2026-05-22", "2026-05-22", candidates)

    assert len(priced) == 25
    assert priced["ticker"].tolist() == ["WMT"] * 25


def test_price_candidates_includes_short_put_when_short_put_family_evidence_passes(tmp_path: Path) -> None:
    _write_minimal_uw_fixture(tmp_path)
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": "AAPL", "realized_pnl": 100.0, "strategy": "short_put"}
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    rows.extend(
        {"ticker": "GOOG", "realized_pnl": -50.0, "strategy": "short_put"}
        for _ in range(4)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "close": 100.0,
                "quality_status": "qualified",
                "score": 80,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.75,
                "issue_type": "Common Stock",
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
            }
        ]
    )

    priced, routing = core.price_candidates_with_routing_audit(
        tmp_path / "2026-05-22",
        "2026-05-22",
        candidates,
        root=tmp_path,
    )

    assert "cash secured put" in priced["structure"].tolist()
    short_put = priced[priced["structure"].eq("cash secured put")].iloc[0]
    assert " / " not in short_put["trade_plan"]
    assert "SELL 1 WMT" in short_put["trade_plan"]
    assert routing["strategy"].tolist() == ["short_put", "bull_call_debit", "bull_put_credit"]
    assert routing.loc[routing["strategy"].eq("short_put"), "route_status"].tolist() == ["constructed"]
    assert routing.loc[routing["strategy"].eq("bull_call_debit"), "route_status"].tolist() == ["construction_failed"]


def test_short_put_replay_bridge_constructs_only_with_near_ready_positive_actual_evidence() -> None:
    candidate = {
        "ticker": "WMT",
        "bias": "bullish",
        "underlying_quality_tier": "core",
        "macro_tape_candidate": True,
        "combined_flow_bias": 0.50,
    }
    near_ready = {
        "short_put": {
            "status": "WARN",
            "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE - 5,
            "avg_pnl": 2.0,
            "profit_factor": 1.01,
        }
    }
    negative = {
        "short_put": {
            "status": "BLOCK",
            "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
            "avg_pnl": -20.0,
            "profit_factor": 0.50,
        }
    }

    bridged = core._candidate_strategy_routes(
        candidate,
        set(),
        near_ready,
        {},
        replay_supported_routes={"short_put"},
    )
    blocked = core._candidate_strategy_routes(
        candidate,
        set(),
        negative,
        {},
        replay_supported_routes={"short_put"},
    )

    assert bridged[0]["strategy"] == "short_put"
    assert bridged[0]["route_reason"] == "positive_short_put_replay_with_near_ready_actual_evidence"
    assert "short_put" not in {row["strategy"] for row in blocked}


def test_price_candidates_includes_near_ready_long_call_route_without_promoting(tmp_path: Path) -> None:
    _write_minimal_uw_fixture(tmp_path)
    hot_path = tmp_path / "2026-05-22" / "hot-chains-2026-05-22.csv"
    hot = pd.read_csv(hot_path)
    hot = pd.concat(
        [
            hot,
            pd.DataFrame(
                [
                    {
                        "option_symbol": "WMT260619C00100000",
                        "date": "2026-05-22",
                        "volume": 5000,
                        "open_interest": 20000,
                        "premium": 1_000_000,
                        "ask_side_volume": 3000,
                        "bid_side_volume": 1500,
                        "bid": 2.00,
                        "ask": 2.20,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    hot.to_csv(hot_path, index=False)
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"LC{idx}", "realized_pnl": 100.0, "strategy": "Long Call"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE - 2)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "close": 100.0,
                "quality_status": "qualified",
                "score": 80,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.75,
                "issue_type": "Common Stock",
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
            }
        ]
    )

    priced, routing = core.price_candidates_with_routing_audit(
        tmp_path / "2026-05-22",
        "2026-05-22",
        candidates,
        root=tmp_path,
    )

    long_call_route = routing[routing["strategy"].eq("long_call")].iloc[0]
    long_call = priced[priced["strategy_route"].eq("long_call")].iloc[0]
    assert long_call_route["evidence_status"] == "WARN"
    assert long_call_route["route_action"] == "construct_research_only_expectancy_missing"
    assert long_call["structure"] == "long call"
    assert long_call["trade_plan"].startswith("BUY 1 WMT 2026-06-19 100 Call")
    assert "SELL" not in long_call["trade_plan"]


def test_price_candidates_routes_bearish_core_to_put_debit_and_audits_credit_route(tmp_path: Path) -> None:
    _write_minimal_uw_fixture(tmp_path)
    candidates = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bearish",
                "close": 100.0,
                "quality_status": "qualified",
                "score": 80,
                "signal_premium": 5_000_000,
                "combined_flow_bias": -0.75,
                "issue_type": "Common Stock",
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "candidate_rank": 1,
            }
        ]
    )

    priced, routing = core.price_candidates_with_routing_audit(
        tmp_path / "2026-05-22",
        "2026-05-22",
        candidates,
        root=tmp_path,
    )

    assert priced["strategy"].tolist() == ["bear_put_debit"]
    assert priced["structure"].tolist() == ["bear put debit spread"]
    assert "BUY 1 WMT 2026-06-19 95 Put" in priced["trade_plan"].iloc[0]
    assert "SELL 1 WMT 2026-06-19 90 Put" in priced["trade_plan"].iloc[0]
    assert priced["entry_limit"].tolist() == [1.4]
    assert routing["strategy"].tolist() == ["bear_put_debit", "bear_call_credit"]
    assert routing.loc[routing["strategy"].eq("bear_put_debit"), "route_status"].tolist() == ["constructed"]
    assert routing.loc[routing["strategy"].eq("bear_call_credit"), "route_status"].tolist() == ["construction_failed"]


def test_actual_route_metrics_include_short_option_roll_lifecycle_outcomes() -> None:
    frame = pd.DataFrame(
        [
            {"ticker": "WMT", "strategy": "short_put", "strategy_route": "short_put", "realized_pnl": 100.0},
            {
                "ticker": "WMT",
                "strategy": "short_put",
                "strategy_route": "roll_adjustment",
                "realized_pnl": -25.0,
            },
            {
                "ticker": "AAPL",
                "strategy": "short_call",
                "strategy_route": "roll_adjustment",
                "realized_pnl": 30.0,
            },
            {
                "ticker": "SPY",
                "strategy": "vertical_spread",
                "strategy_route": "roll_adjustment",
                "realized_pnl": 10.0,
            },
        ]
    )

    metrics = core._actual_forward_metrics_by_strategy_route(frame)

    assert metrics["short_put"]["sample_size"] == 2
    assert metrics["short_put"]["avg_pnl"] == 37.5
    assert metrics["short_call"]["sample_size"] == 1
    assert metrics["roll_adjustment"]["sample_size"] == 1


def test_strategy_routing_prefers_route_evidence_over_broad_vertical_family(tmp_path: Path) -> None:
    _write_minimal_uw_fixture(tmp_path)
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"BPD{idx}", "realized_pnl": 100.0, "strategy": "bear_put_debit"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    rows.extend(
        {"ticker": f"BPC{idx}", "realized_pnl": -300.0, "strategy": "bull_put_credit"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    candidates = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bearish",
                "close": 100.0,
                "quality_status": "qualified",
                "score": 80,
                "signal_premium": 5_000_000,
                "combined_flow_bias": -0.75,
                "issue_type": "Common Stock",
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_volume": 15_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "candidate_rank": 1,
            }
        ]
    )

    priced, routing = core.price_candidates_with_routing_audit(
        tmp_path / "2026-05-22",
        "2026-05-22",
        candidates,
        root=tmp_path,
    )

    bear_put = routing[routing["strategy"].eq("bear_put_debit")].iloc[0]
    assert priced["strategy"].tolist() == ["bear_put_debit"]
    assert bear_put["evidence_scope"] == "strategy_route"
    assert bear_put["evidence_status"] == "PASS"
    assert bear_put["route_action"] == "construct_research_only_route_positive_ticker_proof_required"
    assert "negative_family" not in bear_put["route_action"]


def test_broad_vertical_route_evidence_does_not_create_green_without_ticker_strategy_proof(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"WIN{idx}", "realized_pnl": 100.0, "strategy": "vertical_spread"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "NEW",
                "bias": "bearish",
                "strategy": "bear_put_debit",
                "structure": "bear put debit spread",
                "trade_plan": "BUY 1 NEW 2026-06-19 95 Put / SELL 1 NEW 2026-06-19 90 Put @ 1.00 DEBIT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "live_validation_status": "PASS",
                "entry_limit": 1.0,
                "max_profit": 400.0,
                "max_loss": 100.0,
                "position_max_profit": 1200.0,
                "suggested_contracts": 3,
                "quality_status": "qualified",
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "external_agent_distinct_review_count": 4,
                "external_agent_review_count": 4,
                "agent_support_count": 4,
                "trade_quality_status": "reviewable",
            }
        ]
    )
    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)
    context = {
        "fresh_live_quotes_ready": True,
        "portfolio_ready": True,
        "agentic_reviews_ready": True,
        "min_agentic_review_lanes_per_ticker": 4,
        "run_gate_blockers": [],
        "portfolio_total_value": 100_000,
        "quote_mode": "live_schwab",
    }

    decision = core.synthesize_decision_board(annotated, market_regime={"regime": "risk_off"}, execution_context=context)

    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["BLOCK"]
    assert annotated["actual_forward_strategy_expectancy_scope"].tolist() == ["missing"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]


def test_short_put_family_fallback_is_explicit_and_does_not_mask_negative_ticker(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": "AAPL", "realized_pnl": 100.0, "strategy": "short_put"}
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    rows.extend(
        {"ticker": "BAD", "realized_pnl": -100.0, "strategy": "long_call"}
        for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {"ticker": "NEW", "structure": "cash secured put", "trade_plan": "SELL 1 NEW 2026-06-19 95 Put @ 1.00 CREDIT"},
            {"ticker": "BAD", "structure": "cash secured put", "trade_plan": "SELL 1 BAD 2026-06-19 95 Put @ 1.00 CREDIT"},
        ]
    )

    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)

    new_row = annotated[annotated["ticker"].eq("NEW")].iloc[0]
    bad_row = annotated[annotated["ticker"].eq("BAD")].iloc[0]
    assert new_row["actual_forward_strategy_expectancy_status"] == "PASS"
    assert new_row["actual_forward_strategy_expectancy_scope"] == "strategy_family"
    assert "Family-level actual/forward realized support" in new_row["actual_forward_strategy_expectancy_note"]
    assert bad_row["actual_forward_expectancy_status"] == "BLOCK"
    assert bad_row["actual_forward_strategy_expectancy_status"] == "BLOCK"


def test_short_put_cash_risk_is_portfolio_acknowledgement_not_execution_blocker() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "PUTRISK",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "structure": "cash secured put",
                "full_ticket": "SELL 1 PUTRISK 2026-07-17 150 Put @ 2.00 CREDIT",
                "trade_plan": "SELL 1 PUTRISK 2026-07-17 150 Put @ 2.00 CREDIT",
                "entry_limit": 2.0,
                "suggested_contracts": 1,
                "max_profit": 200.0,
                "max_loss": 14_800.0,
                "credit_width_ratio": "",
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "portfolio_cash": 10_000.0,
                "account_risk_pct": 0.148,
                "dte": 30,
                "macro_calendar_status": "verified",
                "macro_event_count_before_expiry": 0,
                "earnings_before_expiry": False,
                "contract_review_status": "PASS",
                "actual_forward_expectancy_status": "PASS",
                "actual_forward_expectancy_sample_size": 5,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "PASS",
                "profitability_calibration_scope": "actual_ticker_bucket",
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_replay_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000, "cash": 10_000},
        research_task_count=10,
        external_review_count=10,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [True]
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert "short_put_cash_required_above_75pct_cash" not in decision["execution_blockers"].iloc[0]
    assert "short_put_account_risk_above_2.00%" not in decision["execution_blockers"].iloc[0]
    assert "send_now_credit_width_below_30pct" not in decision["execution_blockers"].iloc[0]
    assert decision["requires_portfolio_ack"].tolist() == [True]
    assert "cash required above 75% of portfolio cash" in decision["portfolio_risk_note"].iloc[0]
    assert "account risk above 2.00%" in decision["portfolio_risk_note"].iloc[0]
    assert tickets["ticker"].tolist() == ["PUTRISK"]
    assert tickets["ready_to_enter"].tolist() == [True]
    assert tickets["order_readiness"].tolist() == ["ready_to_enter"]
    assert tickets["action"].tolist() == ["manual_entry_with_portfolio_ack"]


def test_goal_gate_does_not_hide_cash_risk_ticket_readiness() -> None:
    row = {
        "ready_to_enter": False,
        "target_order_status": "target_order_candidate",
        "execution_blockers": "short_put_account_risk_above_2.00%; goal_confidence_gate_blocked",
    }

    assert core._ticket_order_readiness(row) == "target_order_after_cash_risk"
    assert core._ticket_action(row) == "resize_or_skip_until_cash_risk_clears"


def test_negative_strategy_family_evidence_blocks_trade_ticket_surface(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"LOSS{idx}", "realized_pnl": -100.0, "strategy": "vertical_spread"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "NEW",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 NEW 2026-06-19 100 Call / BUY 1 NEW 2026-06-19 105 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 NEW 2026-06-19 100 Call / BUY 1 NEW 2026-06-19 105 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
            }
        ]
    )
    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=10,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(annotated, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["BLOCK"]
    assert annotated["actual_forward_strategy_expectancy_scope"].tolist() == ["strategy_family"]
    assert core.NEGATIVE_STRATEGY_EXPECTANCY_BLOCKER in decision["execution_blockers"].iloc[0]
    assert decision["target_order_status"].tolist() == ["not_actionable_negative_strategy_expectancy"]
    assert tickets.empty


def test_sparse_ticker_strategy_sample_does_not_mask_negative_family_evidence(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    rows = [
        {"ticker": f"LOSS{idx}", "realized_pnl": -100.0, "strategy": "vertical_spread"}
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    rows.append({"ticker": "MSFT", "realized_pnl": -331.0, "strategy": "vertical_spread"})
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "trade_plan": "SELL 1 MSFT 2026-07-17 410 Call / BUY 1 MSFT 2026-07-17 415 Call @ 1.24 CREDIT",
            }
        ]
    )

    annotated = core.annotate_actual_forward_expectancy(final, tmp_path)

    assert annotated["actual_forward_strategy_expectancy_status"].tolist() == ["BLOCK"]
    assert annotated["actual_forward_strategy_expectancy_sample_size"].tolist() == [core.MIN_EXPECTANCY_SAMPLE_SIZE + 1]
    assert annotated["actual_forward_strategy_expectancy_scope"].tolist() == ["strategy_family"]
    assert "Sparse ticker-specific MSFT vertical_spread support" in annotated["actual_forward_strategy_expectancy_note"].iloc[0]


def test_profitability_calibration_passes_only_with_actual_and_replay_bucket_support(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(9000 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Bull Call Debit Spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1200,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "PASS"
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]
    assert annotated["profitability_calibration_actual_sample_size"].tolist() == [core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE]
    assert annotated["profitability_calibration_actual_avg_pnl"].tolist() == [125.0]
    assert annotated["profitability_calibration_actual_profit_factor"].tolist() == ["inf"]
    assert annotated["profitability_calibration_replay_sample_size"].tolist() == [core.MIN_EXPECTANCY_SAMPLE_SIZE]
    assert annotated["profitability_calibration_replay_avg_pnl"].tolist() == [85.0]
    assert annotated["profitability_calibration_replay_profit_factor"].tolist() == ["inf"]


def test_profitability_calibration_uses_schwab_order_legs_for_vertical_route_support(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(1000 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "vertical_spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1200,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)
    summary = core.summarize_profitability_calibration(calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["replay_bucket_status"] == "PASS"
    assert summary["actual_support_status_counts"] == {"PASS": 1}
    assert summary["replay_bucket_status_counts"] == {"PASS": 1}
    assert summary["missing_replay_bucket_rows"] == 0
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]


def test_actual_calibration_keeps_roll_adjustments_out_of_fresh_entry_routes(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    order_id = "2222"
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        json.dumps(
            {
                "ticker": "PLTR",
                "realized_pnl": -387.0,
                "strategy": "short_put",
                "entry_order_ids": [order_id],
                "opened_at": "2026-03-27T14:39:37+00:00",
                "expiry": "2026-06-18",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        json.dumps(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "complexOrderStrategyType": "DIAGONAL",
                "price": 0.03,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "PLTR260618P00140000",
                            "putCall": "PUT",
                            "description": "PLTR 06/18/2026 $140 Put",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "CLOSING",
                        "instruction": "BUY_TO_CLOSE",
                        "instrument": {
                            "symbol": "PLTR260515P00145000",
                            "putCall": "PUT",
                            "description": "PLTR 05/15/2026 $145 Put",
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    actual = core._actual_calibration_frame(tmp_path, tmp_path / "out")

    assert actual["strategy_route"].tolist() == ["roll_adjustment"]
    assert actual["strategy_family"].tolist() == ["roll_adjustment"]
    assert actual["entry_type"].tolist() == ["DEBIT"]


def test_expectancy_evidence_uses_route_aware_actual_frame_for_ticker_strategy(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx, pnl in enumerate([100.0, 120.0, 80.0]):
        order_id = str(3000 + idx)
        closed_rows.append(
            {
                "ticker": "AMZN",
                "realized_pnl": pnl,
                "strategy": "vertical_spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-03-01T14:00:00+00:00",
                "expiry": "2026-04-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.50,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "AMZN260417C00250000",
                            "putCall": "CALL",
                            "description": "AMZN 04/17/2026 $250 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "AMZN260417C00255000",
                            "putCall": "CALL",
                            "description": "AMZN 04/17/2026 $255 Call",
                        },
                    },
                ],
            }
        )
    roll_order_id = "3999"
    closed_rows.append(
        {
            "ticker": "AMZN",
            "realized_pnl": -1000.0,
            "strategy": "vertical_spread",
            "entry_order_ids": [roll_order_id],
            "opened_at": "2026-03-27T14:39:37+00:00",
            "expiry": "2026-06-18",
        }
    )
    raw_order_rows.append(
        {
            "orderId": roll_order_id,
            "orderType": "NET_DEBIT",
            "complexOrderStrategyType": "DIAGONAL",
            "price": 0.03,
            "orderLegCollection": [
                {
                    "orderLegType": "OPTION",
                    "positionEffect": "OPENING",
                    "instruction": "SELL_TO_OPEN",
                    "instrument": {
                        "symbol": "AMZN260618P00220000",
                        "putCall": "PUT",
                        "description": "AMZN 06/18/2026 $220 Put",
                    },
                },
                {
                    "orderLegType": "OPTION",
                    "positionEffect": "CLOSING",
                    "instruction": "BUY_TO_CLOSE",
                    "instrument": {
                        "symbol": "AMZN260515P00225000",
                        "putCall": "PUT",
                        "description": "AMZN 05/15/2026 $225 Put",
                    },
                },
            ],
        }
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    decision = pd.DataFrame(
        [
            {
                "ticker": "AMZN",
                "target_order_status": "target_order_candidate",
                "trade_plan": "BUY 1 AMZN 2026-07-17 252.5 Call / SELL 1 AMZN 2026-07-17 255 Call @ 0.72 DEBIT",
            }
        ]
    )

    evidence = core.build_expectancy_evidence(tmp_path, decision, pd.DataFrame())
    ticker_strategy = evidence[evidence["source"].eq("schwab_closed_trades_by_ticker_strategy")].iloc[0]
    atlas = core.build_strategy_outcome_atlas(tmp_path, decision, pd.DataFrame())
    atlas_ticker_strategy = atlas[atlas["scope"].eq("current_ticker_strategy")].iloc[0]

    assert ticker_strategy["sample_size"] == 3
    assert ticker_strategy["avg_pnl"] == 100.0
    assert ticker_strategy["profit_factor"] == "inf"
    assert atlas_ticker_strategy["sample_size"] == 3
    assert atlas_ticker_strategy["status"] == "PASS"
    assert atlas_ticker_strategy["avg_pnl"] == 100.0


def test_profitability_calibration_backfills_actual_regime_from_opened_trade_date(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    regime_dir = tmp_path / "out" / "options_agent" / "2026-06-11"
    regime_dir.mkdir(parents=True)
    (regime_dir / "market_regime.json").write_text(json.dumps({"regime": "mixed"}), encoding="utf-8")
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(7100 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Bull Call Debit Spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "regime": "mixed",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1200,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "regime": "mixed",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "PASS"
    assert "|bullish|mixed|" in core._calibration_key_text(row)
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]


def test_profitability_calibration_uses_leakage_safe_wheel_csp_replay_for_short_put(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    for day in ["2026-04-01", "2026-06-09"]:
        regime_dir = tmp_path / "out" / "options_agent" / day
        regime_dir.mkdir(parents=True)
        (regime_dir / "market_regime.json").write_text(json.dumps({"regime": "mixed"}), encoding="utf-8")
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(7000 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Short Put",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-09T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "LIMIT",
                "price": 2.50,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717P00095000",
                            "putCall": "PUT",
                            "description": "WMT 07/17/2026 $95 Put",
                        },
                    }
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    wheel_dir = tmp_path / "out" / "fresh_wheel_replay_2026_full_ytd"
    wheel_dir.mkdir(parents=True)
    signal_dir = tmp_path / "2026-04-01"
    signal_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "option_symbol": "WMT260717P00095000",
                "date": "2026-04-01",
                "volume": 900,
                "open_interest": 1500,
                "premium": 100000.0,
                "bid": 2.40,
                "ask": 2.60,
            }
        ]
    ).to_csv(signal_dir / "hot-chains-2026-04-01.csv", index=False)
    past_rows = []
    for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE):
        signal_day = dt.date(2026, 3, 1) + dt.timedelta(days=idx)
        past_rows.append(
            {
                "signal_date": signal_day.isoformat(),
                "ticker": "WMT",
                "action": "OPEN_CSP",
                "option_symbol": "WMT260717P00095000",
                "entry_credit": 2.50,
                "dte": 38,
                "exit_date": (signal_day + dt.timedelta(days=30)).isoformat(),
                "pnl_per_contract": 150.0,
                "outcome_status": "scored",
                "regime": "mixed",
                "source_contract_oi": 1500,
            }
        )
    future_rows = [
        {
            "signal_date": "2026-06-01",
            "ticker": f"FUTURE{idx}",
            "action": "OPEN_CSP",
            "option_symbol": "WMT260717P00095000",
            "entry_credit": 2.50,
            "dte": 38,
            "exit_date": "2026-06-20",
            "pnl_per_contract": -500.0,
            "outcome_status": "scored",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(past_rows + future_rows).to_csv(
        wheel_dir / "fresh-wheel-replay-outcomes-2026-01-02_2026-06-20.csv",
        index=False,
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 2.50 CREDIT",
                "entry_limit": 2.50,
                "dte": 38,
                "regime": "mixed",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final, as_of_date="2026-06-09")
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["replay_bucket_status"] == "PASS"
    assert row["replay_bucket_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["replay_bucket_avg_pnl"] == 150.0
    assert "fresh_wheel_replay_2026_full_ytd" in row["source_path"]
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]


def test_wheel_replay_backfills_missing_regime_from_source_day_before_stale_history(tmp_path: Path) -> None:
    stale_regime_dir = tmp_path / "out" / "options_agent" / "current_code_full_v039_guardrail_2026-04-01"
    stale_regime_dir.mkdir(parents=True)
    (stale_regime_dir / "market_regime.json").write_text(json.dumps({"regime": "mixed"}), encoding="utf-8")

    source_dir = tmp_path / "2026-04-01"
    source_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ticker": ticker, "close": 101.0, "prev_close": 100.0, "bullish_premium": 1000.0, "bearish_premium": 200.0}
            for ticker in ["SPY", "QQQ", "IWM", "DIA"]
        ]
    ).to_csv(source_dir / "stock-screener-2026-04-01.csv", index=False)

    actual = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "canonical_ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "risk_on",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
            }
            for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    wheel_dir = tmp_path / "out" / "fresh_wheel_replay_source_regime"
    wheel_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "signal_date": "2026-04-01",
                "ticker": f"WMT{idx}",
                "action": "OPEN_CSP",
                "option_symbol": f"WMT{idx}260517P00095000",
                "entry_credit": 2.50,
                "dte": 38,
                "exit_date": "2026-05-01",
                "pnl_per_contract": 150.0,
                "outcome_status": "scored",
            }
            for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    ).to_csv(wheel_dir / "fresh-wheel-replay-outcomes-2026-04-01_2026-05-01.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-05-17 95 Put @ 2.50 CREDIT",
                "entry_limit": 2.50,
                "dte": 38,
                "regime": "risk_on",
            }
        ]
    )

    replay, _, _ = core._profitability_replay_frame(tmp_path / "out", as_of=dt.date(2026, 6, 9))
    assert replay["regime"].tolist() == ["risk_on"] * core.MIN_EXPECTANCY_SAMPLE_SIZE

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        as_of_date="2026-06-09",
        actual_frame=actual,
    )
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["replay_bucket_status"] == "PASS"
    assert "|bullish|risk_on|" in core._calibration_key_text(row)
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]


def test_actual_outcome_regime_uses_prior_source_day_not_same_day_eod(tmp_path: Path) -> None:
    same_day_history = tmp_path / "out" / "options_agent" / "2026-05-07"
    same_day_history.mkdir(parents=True)
    (same_day_history / "market_regime.json").write_text(json.dumps({"regime": "risk_off"}), encoding="utf-8")

    prior_dir = tmp_path / "2026-05-06"
    prior_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ticker": ticker, "close": 101.0, "prev_close": 100.0, "bullish_premium": 1000.0, "bearish_premium": 200.0}
            for ticker in ["SPY", "QQQ", "IWM", "DIA"]
        ]
    ).to_csv(prior_dir / "stock-screener-2026-05-06.csv", index=False)

    same_day_dir = tmp_path / "2026-05-07"
    same_day_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"ticker": ticker, "close": 99.0, "prev_close": 100.0, "bullish_premium": 200.0, "bearish_premium": 1000.0}
            for ticker in ["SPY", "QQQ", "IWM", "DIA"]
        ]
    ).to_csv(same_day_dir / "stock-screener-2026-05-07.csv", index=False)

    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "strategy": "short_put",
                "expiry": "2026-06-18",
                "opened_at": "2026-05-07T14:13:47+00:00",
                "closed_at": "2026-05-14T14:13:47+00:00",
                "realized_pnl": 125.0,
                "entry_order_ids": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    actual = core._actual_calibration_frame(tmp_path, tmp_path / "out")

    assert actual["regime"].tolist() == ["risk_on"]


def test_profitability_calibration_requires_matching_replay_liquidity_bucket(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(8000 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Bull Call Debit Spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 10,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert "liquidity_deep" in row["note"]
    assert annotated["profitability_calibration_status"].tolist() == ["WARN"]


def test_profitability_calibration_can_reuse_supplied_replay_bundle(tmp_path: Path, monkeypatch) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "canonical_ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
            }
            for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_unknown",
                "pnl_1x": 85.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.00,
                "dte": 38,
                "regime": "mixed",
            }
        ]
    )

    def _unexpected_replay_build(*args, **kwargs):
        raise AssertionError("replay frame should have been reused")

    monkeypatch.setattr(core, "_profitability_replay_frame", _unexpected_replay_build)

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=actual,
        replay_bundle=(replay, "shared_replay_bundle", ""),
    )

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["source_path"] == "shared_replay_bundle"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["actual_support_sample_gap"] == 0
    assert row["replay_bucket_status"] == "PASS"
    assert row["replay_bucket_sample_gap"] == 0
    assert row["diagnostic_replay_status"] == "PASS"
    assert row["diagnostic_replay_relaxed_dimensions"] == ""


def test_profitability_calibration_requires_matching_replay_dte_bucket(tmp_path: Path) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "canonical_ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
            }
            for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_unknown",
                "pnl_1x": 85.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.00,
                "dte": 38,
                "regime": "mixed",
            }
        ]
    )

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=actual,
        replay_bundle=(replay, "shared_replay_bundle", ""),
    )

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert row["replay_bucket_sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["diagnostic_replay_status"] == "PASS"
    assert row["diagnostic_replay_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["diagnostic_replay_relaxed_dimensions"] == "dte_bucket"


def test_profitability_calibration_requires_matching_replay_economics_bucket(tmp_path: Path) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "canonical_ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
            }
            for _ in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_unknown",
                "pnl_1x": 85.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "cash secured put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 1.00 CREDIT",
                "entry_limit": 1.00,
                "dte": 38,
                "regime": "mixed",
            }
        ]
    )

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=actual,
        replay_bundle=(replay, "shared_replay_bundle", ""),
    )

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert row["replay_bucket_sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["diagnostic_replay_status"] == "PASS"
    assert row["diagnostic_replay_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["diagnostic_replay_relaxed_dimensions"] == "economics_bucket"


def test_profitability_calibration_requires_matching_replay_regime_bucket(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(8100 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "Bull Call Debit Spread",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
                "regime": "mixed",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00100000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $100 Call",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "regime": "uptrend",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1500,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 WMT 2026-07-17 100 Call / SELL 1 WMT 2026-07-17 105 Call @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "regime": "mixed",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["actual_support_sample_gap"] == 0
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert row["replay_bucket_sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert "|mixed|" in core._calibration_key_text(row)
    assert "mixed" in row["note"]
    assert annotated["profitability_calibration_status"].tolist() == ["WARN"]


def test_profitability_calibration_requires_matching_replay_direction_bucket(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = []
    raw_order_rows = []
    for idx in range(core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE):
        order_id = str(8200 + idx)
        closed_rows.append(
            {
                "ticker": "WMT",
                "realized_pnl": 125.0,
                "strategy": "vertical_spread",
                "direction": "bearish",
                "entry_order_ids": [order_id],
                "opened_at": "2026-06-12T14:00:00+00:00",
                "expiry": "2026-07-17",
                "regime": "mixed",
            }
        )
        raw_order_rows.append(
            {
                "orderId": order_id,
                "orderType": "NET_DEBIT",
                "price": 1.60,
                "orderLegCollection": [
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717P00100000",
                            "putCall": "PUT",
                            "description": "WMT 07/17/2026 $100 Put",
                        },
                    },
                    {
                        "orderLegType": "OPTION",
                        "positionEffect": "OPENING",
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {
                            "symbol": "WMT260717C00105000",
                            "putCall": "CALL",
                            "description": "WMT 07/17/2026 $105 Call",
                        },
                    },
                ],
            }
        )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in raw_order_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "WMT",
            "strategy": "Vertical Spread",
            "direction": "bullish",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "regime": "mixed",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1500,
            "pnl_1x": 85.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "vertical_spread",
                "direction": "bearish",
                "trade_plan": "WMT vertical spread @ 1.60 DEBIT",
                "entry_limit": 1.6,
                "max_profit": 340.0,
                "max_loss": 160.0,
                "dte": 35,
                "iv_rank": 42,
                "regime": "mixed",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["direction_bucket"] == "bearish"
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_ticker_bucket"
    assert row["replay_bucket_status"] == "BLOCK"
    assert row["replay_bucket_sample_size"] == 0
    assert "|bearish|mixed|" in core._calibration_key_text(row)
    assert "bearish" in row["note"]
    assert annotated["profitability_calibration_status"].tolist() == ["WARN"]


def test_profitability_calibration_does_not_pass_from_broad_vertical_family_only_actual_support(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"WIN{idx}",
            "realized_pnl": 125.0,
            "strategy": "vertical_spread",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": "MSFT",
            "strategy": "Bear Call Credit Spread",
            "strategy_kind": "Credit",
            "entry_side": "credit",
            "dte": 10,
            "iv_rank": 35,
            "entry_credit_pct_width": 0.32,
            "source_contract_oi": 800,
            "pnl_1x": 75.0,
            "exact_evaluated": True,
            "decision_pass": True,
        }
        for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    final = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "structure": "Bear Call Credit Spread",
                "trade_plan": "SELL 1 MSFT 2026-06-19 430 Call / BUY 1 MSFT 2026-06-19 435 Call @ 1.60 CREDIT",
                "entry_limit": 1.6,
                "credit_width_ratio": 0.32,
                "dte": 10,
                "iv_rank": 35,
                "live_leg_min_liquidity": 800,
            }
        ]
    )

    calibration = core.build_profitability_calibration(tmp_path, final)
    annotated = core.annotate_profitability_calibration(final, calibration)
    summary = core.summarize_profitability_calibration(calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_strategy_family"
    assert row["replay_bucket_status"] == "PASS"
    assert "actual_bucket_precision=route_or_family_only" in row["note"]
    assert summary["actual_family_only_rows"] == 1
    assert summary["missing_replay_bucket_rows"] == 0
    assert annotated["profitability_calibration_status"].tolist() == ["WARN"]


def test_profitability_calibration_allows_actual_route_economics_bucket_with_exact_replay(
    tmp_path: Path,
) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": f"SRC{idx}",
                "canonical_ticker": f"SRC{idx}",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed" if idx % 2 else "regime_unknown",
                "dte_bucket": "dte_15_30" if idx % 3 else "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_unknown",
                "realized_pnl": 100.0 if idx < 24 else -40.0,
                "source": "schwab_closed_trades",
            }
            for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "risk_on",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "pnl_1x": 75.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    final = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Short Put",
                "trade_plan": "SELL 1 WMT 2026-07-17 100 Put @ 2.00 CREDIT",
                "entry_limit": 2.0,
                "max_profit": 200.0,
                "max_loss": 10000.0,
                "dte": 36,
                "regime": "risk_on",
                "live_leg_min_liquidity": 1500,
            }
        ]
    )

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=actual,
        replay_bundle=(replay, Path("replay_fixture.csv"), ""),
    )
    annotated = core.annotate_profitability_calibration(final, calibration)
    summary = core.summarize_profitability_calibration(calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_route_economics_bucket"
    assert row["actual_support_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["replay_bucket_status"] == "PASS"
    assert summary["actual_bucket_and_replay_pass_rows"] == 1
    assert annotated["profitability_calibration_status"].tolist() == ["PASS"]

    atlas = core.build_profitability_bucket_atlas(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        actual_frame=actual,
        replay_bundle=(replay, Path("replay_fixture.csv"), ""),
        profitability_calibration=calibration,
    )
    atlas_summary = core.summarize_profitability_bucket_atlas(atlas)
    atlas_row = atlas[atlas["current_ticket_count"].gt(0)].iloc[0]
    assert atlas_row["status"] == "PASS"
    assert atlas_row["actual_bucket_status"] == "PASS"
    assert atlas_row["replay_bucket_status"] == "PASS"
    assert atlas_summary["current_pass_bucket_rows"] == 1


def test_profitability_calibration_prefers_route_actual_support_over_broad_family_diagnostics(
    tmp_path: Path,
) -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 AAPL 2026-07-17 210 Call / SELL 1 AAPL 2026-07-17 220 Call @ 2.50 DEBIT",
                "entry_limit": 2.5,
                "max_profit": 750.0,
                "max_loss": 250.0,
                "dte": 38,
            }
        ]
    )
    actual_rows = [
        {
            "ticker": f"BCD{idx}",
            "canonical_ticker": f"BCD{idx}",
            "strategy_route": "bull_call_debit",
            "strategy_family": "vertical_spread",
            "entry_type": "DEBIT",
            "direction_bucket": "bullish",
            "regime": "regime_unknown",
            "dte_bucket": "dte_0_14",
            "economics_bucket": "debit_reward_risk_mid",
            "liquidity_bucket": "liquidity_unknown",
            "realized_pnl": 100.0,
        }
        for idx in range(5)
    ]
    actual_rows.extend(
        {
            "ticker": f"BPC{idx}",
            "canonical_ticker": f"BPC{idx}",
            "strategy_route": "bull_put_credit",
            "strategy_family": "vertical_spread",
            "entry_type": "CREDIT",
            "direction_bucket": "bullish",
            "regime": "regime_unknown",
            "dte_bucket": "dte_31_60",
            "economics_bucket": "credit_width_mid",
            "liquidity_bucket": "liquidity_unknown",
            "realized_pnl": -100.0,
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    )
    replay_bundle = (
        pd.DataFrame(
            columns=[
                "strategy_route",
                "entry_type",
                "direction_bucket",
                "regime",
                "dte_bucket",
                "economics_bucket",
                "liquidity_bucket",
                "pnl_1x",
            ]
        ),
        "test-replay.csv",
        "no replay rows",
    )

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=pd.DataFrame(actual_rows),
        replay_bundle=replay_bundle,
    )

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_scope"] == "actual_route"
    assert row["actual_support_sample_size"] == 5
    assert row["actual_support_avg_pnl"] == 100.0
    assert "actual_support=WARN sample=5 scope=actual_route" in row["note"]


def test_profitability_calibration_keeps_route_pass_when_bucket_is_under_sampled(
    tmp_path: Path,
) -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "CRM",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 CRM 2026-07-17 172.5 Call / SELL 1 CRM 2026-07-17 175 Call @ 0.57 DEBIT",
                "entry_limit": 0.57,
                "max_profit": 193.0,
                "max_loss": 57.0,
                "dte": 38,
                "regime": "risk_on",
                "live_leg_min_liquidity": 500,
            }
        ]
    )
    actual_rows = []
    for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE):
        actual_rows.append(
            {
                "ticker": f"BCD{idx}",
                "canonical_ticker": f"BCD{idx}",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "entry_type": "DEBIT",
                "direction_bucket": "bullish",
                "regime": "mixed" if idx < 27 else "risk_on",
                "dte_bucket": "dte_15_30" if idx < 27 else "dte_31_60",
                "economics_bucket": "debit_reward_risk_mid" if idx < 27 else "debit_reward_risk_high",
                "liquidity_bucket": "liquidity_unknown",
                "realized_pnl": 100.0 if idx < 24 else -20.0,
            }
        )
    replay = pd.DataFrame(
        [
            {
                "ticker": "CRM",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "entry_type": "DEBIT",
                "direction_bucket": "bullish",
                "regime": "risk_on",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "debit_reward_risk_high",
                "liquidity_bucket": "liquidity_adequate",
                "pnl_1x": 75.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=pd.DataFrame(actual_rows),
        replay_bundle=(replay, "replay_fixture.csv", ""),
    )

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_route"
    assert row["actual_support_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["replay_bucket_status"] == "PASS"
    assert row["suggested_action"] == "keep_yellow_with_hierarchical_route_support"
    assert "cannot promote an exact contract to green" in row["note"]


def test_profitability_calibration_records_route_replay_support_when_exact_bucket_is_thin(
    tmp_path: Path,
) -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "structure": "Bull Call Debit Spread",
                "trade_plan": "BUY 1 GOOG 2026-07-17 175 Call / SELL 1 GOOG 2026-07-17 180 Call @ 0.60 DEBIT",
                "entry_limit": 0.60,
                "max_profit": 440.0,
                "max_loss": 60.0,
                "dte": 38,
                "regime": "risk_on",
                "live_leg_min_liquidity": 500,
            }
        ]
    )
    actual = pd.DataFrame(
        [
            {
                "ticker": f"BCD{idx}",
                "canonical_ticker": f"BCD{idx}",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "entry_type": "DEBIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_15_30",
                "economics_bucket": "debit_reward_risk_mid",
                "liquidity_bucket": "liquidity_unknown",
                "realized_pnl": 100.0 if idx < 24 else -20.0,
            }
            for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay_rows = [
        {
            "ticker": "GOOG",
            "strategy_route": "bull_call_debit",
            "strategy_family": "vertical_spread",
            "entry_type": "DEBIT",
            "direction_bucket": "bullish",
            "regime": "risk_on",
            "dte_bucket": "dte_31_60",
            "economics_bucket": "debit_reward_risk_high",
            "liquidity_bucket": "liquidity_adequate",
            "pnl_1x": 40.0,
        }
    ]
    replay_rows.extend(
        {
            "ticker": f"BCD{idx}",
            "strategy_route": "bull_call_debit",
            "strategy_family": "vertical_spread",
            "entry_type": "DEBIT",
            "direction_bucket": "bullish",
            "regime": "mixed",
            "dte_bucket": "dte_15_30",
            "economics_bucket": "debit_reward_risk_mid",
            "liquidity_bucket": "liquidity_unknown",
            "pnl_1x": 100.0 if idx < 23 else -20.0,
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE - 1)
    )

    calibration = core.build_profitability_calibration(
        tmp_path,
        final,
        actual_frame=actual,
        replay_bundle=(pd.DataFrame(replay_rows), "replay_fixture.csv", ""),
    )
    summary = core.summarize_profitability_calibration(calibration)

    row = calibration[calibration["scope"].eq("current_trade_calibration")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_support_status"] == "PASS"
    assert row["actual_support_scope"] == "actual_route"
    assert row["replay_bucket_status"] == "WARN"
    assert row["route_replay_status"] == "PASS"
    assert row["route_replay_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["suggested_action"] == "keep_yellow_with_hierarchical_route_support"
    assert "cannot promote an exact contract to green" in row["note"]
    assert summary["actual_and_replay_pass_rows"] == 0
    assert summary["route_actual_and_replay_pass_rows"] == 1


def test_profitability_calibration_allows_one_lot_hierarchical_route_pass_at_pf_floor() -> None:
    actual = {
        "status": "PASS",
        "sample_size": core.MIN_HIERARCHICAL_ROUTE_SAMPLE_SIZE,
        "win_rate": 0.44,
        "avg_pnl": 20.0,
        "profit_factor": 1.25,
    }
    route_replay = {
        "status": "PASS",
        "sample_size": core.MIN_HIERARCHICAL_ROUTE_SAMPLE_SIZE,
        "win_rate": 0.50,
        "avg_pnl": 15.0,
        "profit_factor": 1.30,
    }
    sparse_exact_replay = {
        "status": "BLOCK",
        "sample_size": 0,
        "win_rate": "",
        "avg_pnl": "",
        "profit_factor": "",
    }

    status, action, note = core._current_calibration_verdict(
        ticker="AAPL",
        key={"strategy_route": "bull_call_debit"},
        actual_scope="actual_route",
        actual_metrics=actual,
        replay_metrics=sparse_exact_replay,
        route_replay_metrics=route_replay,
        replay_path=Path("replay.csv"),
        replay_error="",
    )

    assert status == "PASS"
    assert action == "eligible_for_one_lot_green_with_hierarchical_route_support"
    assert "capped at one contract" in note

    negative_exact_replay = {
        "status": "BLOCK",
        "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
        "win_rate": 0.30,
        "avg_pnl": -10.0,
        "profit_factor": 0.80,
    }
    blocked_status, _, _ = core._current_calibration_verdict(
        ticker="AAPL",
        key={"strategy_route": "bull_call_debit"},
        actual_scope="actual_route",
        actual_metrics=actual,
        replay_metrics=negative_exact_replay,
        route_replay_metrics=route_replay,
        replay_path=Path("replay.csv"),
        replay_error="",
    )

    assert blocked_status != "PASS"


def test_profitability_calibration_borrows_passing_heldout_model_for_positive_route() -> None:
    actual = {
        "status": "PASS",
        "sample_size": core.MIN_HIERARCHICAL_ROUTE_SAMPLE_SIZE,
        "win_rate": 0.44,
        "avg_pnl": 20.0,
        "profit_factor": 1.25,
    }
    route_replay = {
        "status": "WARN",
        "sample_size": core.MIN_HIERARCHICAL_ROUTE_REPLAY_SAMPLE_SIZE,
        "win_rate": 0.70,
        "avg_pnl": 45.0,
        "profit_factor": 1.80,
    }
    model_replay = {
        "status": "PASS",
        "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
        "win_rate": 0.70,
        "avg_pnl": 35.0,
        "profit_factor": 1.60,
    }
    sparse_exact_replay = {
        "status": "BLOCK",
        "sample_size": 0,
        "win_rate": "",
        "avg_pnl": "",
        "profit_factor": "",
    }

    status, action, note = core._current_calibration_verdict(
        ticker="AAPL",
        key={"strategy_route": "bull_call_debit"},
        actual_scope="actual_route",
        actual_metrics=actual,
        replay_metrics=sparse_exact_replay,
        route_replay_metrics=route_replay,
        model_replay_metrics=model_replay,
        replay_path=Path("replay.csv"),
        replay_error="",
    )

    assert status == "PASS"
    assert action == "eligible_for_one_lot_green_with_hierarchical_route_support"
    assert "held-out model support" in note

    route_replay["avg_pnl"] = -5.0
    route_replay["profit_factor"] = 0.80
    blocked_status, _, _ = core._current_calibration_verdict(
        ticker="AAPL",
        key={"strategy_route": "bull_call_debit"},
        actual_scope="actual_route",
        actual_metrics=actual,
        replay_metrics=sparse_exact_replay,
        route_replay_metrics=route_replay,
        model_replay_metrics=model_replay,
        replay_path=Path("replay.csv"),
        replay_error="",
    )

    assert blocked_status != "PASS"

    mature_weak_exact_replay = {
        "status": "BLOCK",
        "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
        "win_rate": 0.30,
        "avg_pnl": 5.0,
        "profit_factor": 1.10,
    }
    weak_status, _, _ = core._current_calibration_verdict(
        ticker="AAPL",
        key={"strategy_route": "bull_call_debit"},
        actual_scope="actual_route",
        actual_metrics=actual,
        replay_metrics=mature_weak_exact_replay,
        route_replay_metrics=route_replay,
        replay_path=Path("replay.csv"),
        replay_error="",
    )

    assert weak_status != "PASS"


def test_profitability_calibration_summary_names_bucket_shortfalls() -> None:
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "GOOD",
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "status": "PASS",
                "actual_support_scope": "actual_ticker_bucket",
                "actual_support_status": "PASS",
                "actual_support_sample_size": 3,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": 30,
                "diagnostic_replay_relaxed_dimensions": "",
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "SHORT",
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_deep",
                "status": "WARN",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_status": "WARN",
                "actual_support_sample_size": 10,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": 30,
                "diagnostic_replay_relaxed_dimensions": "dte_bucket",
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "FAM",
                "strategy_route": "bull_call_debit",
                "entry_type": "DEBIT",
                "dte_bucket": "dte_15_30",
                "economics_bucket": "debit_reward_risk_mid",
                "liquidity_bucket": "liquidity_adequate",
                "status": "WARN",
                "actual_support_scope": "actual_strategy_family",
                "actual_support_status": "BLOCK",
                "actual_support_sample_size": 42,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 4,
                "diagnostic_replay_relaxed_dimensions": "liquidity_bucket",
            },
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    summary = core.summarize_profitability_calibration(calibration)
    blocker_detail = core._profitability_calibration_blocker_detail(summary)
    examples_detail = core._calibration_bucket_examples_detail(summary)
    intersection_detail = core._calibration_intersection_examples_detail(summary)

    assert summary["bucket_precision_rows"] == 2
    assert summary["bucket_shortfall_rows"] == 2
    assert summary["bucket_shortfall_routes"] == ["bull_call_debit", "short_put"]
    assert summary["actual_and_replay_pass_rows"] == 1
    assert summary["actual_pass_replay_not_pass_rows"] == 0
    assert summary["actual_not_pass_replay_pass_rows"] == 1
    assert summary["actual_and_replay_not_pass_rows"] == 1
    assert summary["calibration_intersection_examples"][0]["gap_type"] == "replay_pass_actual_gap"
    assert summary["calibration_intersection_examples"][0]["ticker"] == "SHORT"
    assert len(summary["bucket_blocker_examples"]) == 2
    short_example = next(item for item in summary["bucket_blocker_examples"] if item["ticker"] == "SHORT")
    assert short_example["actual_sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE - 10
    assert short_example["replay_sample_gap"] == 0
    assert "actual_and_replay_pass_rows=1/3; actual_only=0; replay_only=1; neither=1" in blocker_detail
    assert "bucket_shortfall_rows=2 routes=bull_call_debit,short_put" in blocker_detail
    assert "SHORT short_put/direction_unknown/dte_31_60/credit_standard" in examples_detail
    assert "missing_dims=dte_bucket" in examples_detail
    assert "SHORT short_put/CREDIT/direction_unknown/regime_unknown/dte_31_60/credit_standard/liquidity_deep" in intersection_detail
    assert "needs_actual_gap=20 actual=WARN sample=10 replay=PASS sample=30" in intersection_detail


def test_calibration_intersection_detail_balances_actual_and_replay_gaps() -> None:
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "BAC",
                "strategy_route": "bear_put_debit",
                "entry_type": "DEBIT",
                "direction_bucket": "bearish",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "debit_reward_risk_high",
                "liquidity_bucket": "liquidity_deep",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_status": "WARN",
                "actual_support_sample_size": 1,
                "actual_support_sample_gap": 29,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 1,
                "replay_bucket_sample_gap": 29,
                "current_ticket_count": 1,
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "BAC",
                "strategy_route": "bear_put_debit",
                "entry_type": "DEBIT",
                "direction_bucket": "bearish",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "debit_reward_risk_high",
                "liquidity_bucket": "liquidity_deep",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_status": "WARN",
                "actual_support_sample_size": 1,
                "actual_support_sample_gap": 29,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 1,
                "replay_bucket_sample_gap": 29,
                "current_ticket_count": 1,
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "CMCSA",
                "strategy_route": "bear_put_debit",
                "entry_type": "DEBIT",
                "direction_bucket": "bearish",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "debit_reward_risk_high",
                "liquidity_bucket": "liquidity_deep",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_status": "WARN",
                "actual_support_sample_size": 1,
                "actual_support_sample_gap": 29,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 1,
                "replay_bucket_sample_gap": 29,
                "current_ticket_count": 1,
            },
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )
    summary = {
        "calibration_intersection_examples": [
            {
                "gap_type": "actual_pass_replay_gap",
                "ticker": "AAA",
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "credit_small",
                "liquidity_bucket": "liquidity_deep",
                "actual_status": "PASS",
                "actual_sample_size": 36,
                "actual_sample_gap": 0,
                "replay_status": "BLOCK",
                "replay_sample_size": 0,
                "replay_sample_gap": 30,
            },
            {
                "gap_type": "actual_pass_replay_gap",
                "ticker": "BBB",
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "dte_bucket": "dte_0_14",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "actual_status": "PASS",
                "actual_sample_size": 36,
                "actual_sample_gap": 0,
                "replay_status": "BLOCK",
                "replay_sample_size": 0,
                "replay_sample_gap": 30,
            },
            {
                "gap_type": "actual_pass_replay_gap",
                "ticker": "CCC",
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_small",
                "liquidity_bucket": "liquidity_deep",
                "actual_status": "PASS",
                "actual_sample_size": 36,
                "actual_sample_gap": 0,
                "replay_status": "WARN",
                "replay_sample_size": 2,
                "replay_sample_gap": 28,
            },
            {
                "gap_type": "replay_pass_actual_gap",
                "ticker": "DDD",
                "strategy_route": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "actual_status": "WARN",
                "actual_sample_size": 4,
                "actual_sample_gap": 26,
                "replay_status": "PASS",
                "replay_sample_size": 33,
                "replay_sample_gap": 0,
            },
            *core._calibration_intersection_examples(calibration),
        ]
    }

    detail = core._calibration_intersection_examples_detail(summary, per_type_limit=2)

    assert "needs_replay:" in detail
    assert "AAA short_put/CREDIT/bullish/regime_unknown/dte_0_14/credit_small/liquidity_deep" in detail
    assert "BBB short_put/CREDIT/bullish/regime_unknown/dte_0_14/credit_rich/liquidity_deep" in detail
    assert "CCC short_put" not in detail
    assert "needs_actual:" in detail
    assert "DDD short_put/CREDIT/bullish/regime_unknown/dte_31_60/credit_rich/liquidity_deep" in detail
    assert "needs_actual_gap=26 actual=WARN sample=4 replay=PASS sample=33" in detail
    assert detail.count("BAC,CMCSA bear_put_debit/DEBIT/bearish/regime_unknown/dte_0_14/debit_reward_risk_high/liquidity_deep") == 1
    assert "BAC bear_put_debit/DEBIT/bearish/regime_unknown/dte_0_14/debit_reward_risk_high/liquidity_deep" not in detail


def test_profitability_calibration_intersection_gap_groups_exact_buckets() -> None:
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "TLT",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "credit_small",
                "liquidity_bucket": "liquidity_deep",
                "actual_support_scope": "actual_route",
                "actual_support_status": "PASS",
                "actual_support_sample_size": 36,
                "actual_support_sample_gap": 0,
                "actual_support_avg_pnl": 43.92,
                "actual_support_profit_factor": 1.295,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 2,
                "replay_bucket_sample_gap": 28,
                "replay_bucket_avg_pnl": 23.75,
                "replay_bucket_profit_factor": "inf",
                "current_ticket_count": 1,
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "XLF",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "credit_small",
                "liquidity_bucket": "liquidity_deep",
                "actual_support_scope": "actual_route",
                "actual_support_status": "PASS",
                "actual_support_sample_size": 36,
                "actual_support_sample_gap": 0,
                "actual_support_avg_pnl": 43.92,
                "actual_support_profit_factor": 1.295,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 2,
                "replay_bucket_sample_gap": 28,
                "replay_bucket_avg_pnl": 23.75,
                "replay_bucket_profit_factor": "inf",
                "current_ticket_count": 1,
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "BX",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_status": "WARN",
                "actual_support_sample_size": 4,
                "actual_support_sample_gap": 26,
                "actual_support_avg_pnl": -232.0,
                "actual_support_profit_factor": 0.012,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": 33,
                "replay_bucket_sample_gap": 0,
                "replay_bucket_avg_pnl": 271.05,
                "replay_bucket_profit_factor": 3.324,
                "current_ticket_count": 1,
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "BAC",
                "strategy_route": "bear_put_debit",
                "strategy_family": "vertical_spread",
                "entry_type": "DEBIT",
                "direction_bucket": "bearish",
                "regime": "mixed",
                "dte_bucket": "dte_0_14",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "debit_reward_risk_high",
                "liquidity_bucket": "liquidity_deep",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_status": "WARN",
                "actual_support_sample_size": 1,
                "actual_support_sample_gap": 29,
                "actual_support_avg_pnl": -111.0,
                "actual_support_profit_factor": 0.0,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 1,
                "replay_bucket_sample_gap": 29,
                "replay_bucket_avg_pnl": 106.7,
                "replay_bucket_profit_factor": "inf",
                "current_ticket_count": 1,
            },
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    gaps = core.build_profitability_calibration_intersection_gap(calibration)
    summary = core.summarize_profitability_calibration_intersection_gap(gaps)

    assert len(gaps) == 3
    replay_gap = gaps[gaps["gap_type"].eq("actual_pass_replay_gap")].iloc[0]
    assert replay_gap["current_tickers"] == "TLT,XLF"
    assert replay_gap["current_ticket_count"] == 2
    assert replay_gap["replay_bucket_sample_gap"] == 28
    actual_gap = gaps[gaps["gap_type"].eq("replay_pass_actual_gap")].iloc[0]
    assert actual_gap["current_tickers"] == "BX"
    assert actual_gap["actual_support_sample_gap"] == 26
    both_gap = gaps[gaps["gap_type"].eq("actual_and_replay_gap")].iloc[0]
    assert both_gap["current_tickers"] == "BAC"
    assert both_gap["actual_support_sample_gap"] == 29
    assert both_gap["replay_bucket_sample_gap"] == 29
    assert summary["status"] == "block"
    assert summary["gap_rows"] == 3
    assert summary["total_current_ticket_count"] == 4
    assert summary["gap_type_counts"] == {
        "actual_pass_replay_gap": 1,
        "actual_and_replay_gap": 1,
        "replay_pass_actual_gap": 1,
    }


def test_profitability_gap_plan_names_exact_bucket_evidence_steps() -> None:
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "PG",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_deep",
                "status": "WARN",
                "actual_support_status": "WARN",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_sample_size": 10,
                "actual_support_sample_gap": core.MIN_EXPECTANCY_SAMPLE_SIZE - 10,
                "actual_support_avg_pnl": 14.0,
                "actual_support_profit_factor": 1.5,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "replay_bucket_sample_gap": 0,
                "replay_bucket_avg_pnl": 20.0,
                "replay_bucket_profit_factor": 2.0,
                "diagnostic_replay_status": "PASS",
                "diagnostic_replay_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "diagnostic_replay_relaxed_dimensions": "dte_bucket",
                "current_ticket_count": 1,
                "source_path": "replay.csv",
                "note": "short put needs actual samples",
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "XLF",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "credit_standard",
                "liquidity_bucket": "liquidity_deep",
                "status": "WARN",
                "actual_support_status": "WARN",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_sample_size": 10,
                "actual_support_sample_gap": core.MIN_EXPECTANCY_SAMPLE_SIZE - 10,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "replay_bucket_sample_gap": 0,
                "diagnostic_replay_status": "PASS",
                "diagnostic_replay_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "diagnostic_replay_relaxed_dimensions": "dte_bucket",
                "current_ticket_count": 1,
            },
            {
                "scope": "current_trade_calibration",
                "ticker": "KO",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "entry_type": "DEBIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_0_14",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "debit_reward_risk_weak",
                "liquidity_bucket": "liquidity_unknown",
                "status": "WARN",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "actual_support_sample_gap": 0,
                "replay_bucket_status": "WARN",
                "replay_bucket_sample_size": 1,
                "replay_bucket_sample_gap": core.MIN_EXPECTANCY_SAMPLE_SIZE - 1,
                "diagnostic_replay_status": "WARN",
                "diagnostic_replay_sample_size": 1,
                "diagnostic_replay_relaxed_dimensions": "liquidity_bucket",
                "current_ticket_count": 1,
            },
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    gap_plan = core.build_profitability_gap_plan(calibration)
    summary = core.summarize_profitability_gap_plan(gap_plan)
    detail = core._profitability_gap_plan_detail(summary)

    assert list(gap_plan.columns) == core.PROFITABILITY_GAP_PLAN_COLUMNS
    short_put = gap_plan[gap_plan["strategy_route"].eq("short_put")].iloc[0]
    assert short_put["current_tickers"] == "PG,XLF"
    assert short_put["current_ticket_count"] == 2
    assert short_put["primary_gap"] == "actual_closed_outcomes_sample_gap"
    assert "Need 20 more positive closed/forward outcomes" in short_put["next_evidence_needed"]
    assert "Nearest replay support only appears after relaxing dte_bucket" in short_put["next_evidence_needed"]
    debit = gap_plan[gap_plan["strategy_route"].eq("bull_call_debit")].iloc[0]
    assert debit["primary_gap"] == "replay_exact_bucket_sample_gap"
    assert "Need 29 more leakage-safe replay outcomes" in debit["next_evidence_needed"]
    assert summary["blocking_rows"] == 2
    assert summary["primary_gap_counts"]["actual_closed_outcomes_sample_gap"] == 1
    assert "PG,XLF short_put actual_closed_outcomes_sample_gap" in detail


def test_route_opportunity_gap_surfaces_near_ready_long_call_without_promoting(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"LC{idx}",
            "realized_pnl": 100.0,
            "strategy": "Long Call",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE - 1)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": f"LC{idx}",
            "strategy": "Long Call",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 35,
            "iv_rank": 42,
            "reward_risk": 2.0,
            "source_contract_oi": 1200,
            "pnl_1x": 90.0,
            "exact_evaluated": True,
            "decision_pass": True,
            "exit_day": "2026-05-15",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    gap = core.build_route_opportunity_gap(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("long_call")].iloc[0]
    assert row["route_status"] == "near_ready_more_actual_sample_needed"
    assert row["actual_status"] == "WARN"
    assert row["actual_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE - 1
    assert row["replay_status"] == "PASS"
    assert row["current_ticket_count"] == 0
    assert "actual_route_sample_below_30" in row["development_gap"]
    assert summary["near_ready_routes"] == ["long_call"]
    assert summary["candidate_expansion_routes"] == []


def test_profitability_bucket_atlas_requires_actual_and_replay_bucket_pass(tmp_path: Path) -> None:
    closed_rows = [
        {
            "ticker": f"CSP{idx}",
            "realized_pnl": 120.0,
            "strategy": "Short Put",
            "opened_at": "2026-05-01T14:00:00+00:00",
            "expiry": "2026-06-19",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    actual = pd.DataFrame(closed_rows)
    actual["canonical_ticker"] = actual["ticker"]
    actual["strategy_route"] = "short_put"
    actual["strategy_family"] = "short_put"
    actual["entry_type"] = "CREDIT"
    actual["direction_bucket"] = "bullish"
    actual["regime"] = "mixed"
    actual["dte_bucket"] = "dte_31_60"
    actual["economics_bucket"] = "credit_rich"
    actual["source"] = "fixture_closed_trades"
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "pnl_1x": 90.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "current_ticket_count": 1,
                "matched_current_tickers": "WMT",
            }
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    atlas = core.build_profitability_bucket_atlas(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        as_of_date="2026-06-09",
        actual_frame=actual,
        replay_bundle=(replay, "fixture_replay.csv", ""),
        profitability_calibration=calibration,
    )
    summary = core.summarize_profitability_bucket_atlas(atlas)

    row = atlas[atlas["bucket_key"].str.contains("short_put\\|short_put\\|CREDIT\\|bullish\\|mixed\\|dte_31_60\\|credit_rich\\|liquidity_deep")].iloc[0]
    assert row["status"] == "PASS"
    assert row["actual_bucket_status"] == "PASS"
    assert row["replay_bucket_status"] == "PASS"
    assert row["current_ticket_count"] == 1
    assert row["primary_gap"] == "execution_gates_remaining"
    assert summary["pass_rows"] == 1
    assert summary["current_pass_bucket_rows"] == 1


def test_profitability_bucket_atlas_counts_calibration_rows_without_summing_bucket_totals(tmp_path: Path) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": f"CSP{idx}",
                "canonical_ticker": f"CSP{idx}",
                "realized_pnl": 100.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "source": "fixture_closed_trades",
            }
            for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "pnl_1x": 100.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": ticker,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "current_ticket_count": 3,
                "matched_current_tickers": ticker,
            }
            for ticker in ["AAPL", "MSFT", "NVDA"]
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    atlas = core.build_profitability_bucket_atlas(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        actual_frame=actual,
        replay_bundle=(replay, "fixture_replay.csv", ""),
        profitability_calibration=calibration,
    )

    row = atlas[atlas["liquidity_bucket"].eq("liquidity_deep")].iloc[0]
    assert row["current_ticket_count"] == 3
    assert row["current_tickers"] == "AAPL,MSFT,NVDA"


def test_profitability_bucket_atlas_blocks_when_actual_bucket_losing_despite_replay_pass(tmp_path: Path) -> None:
    actual = pd.DataFrame(
        [
            {
                "ticker": f"CSP{idx}",
                "canonical_ticker": f"CSP{idx}",
                "realized_pnl": -100.0,
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "source": "fixture_closed_trades",
            }
            for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "pnl_1x": 125.0,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )

    atlas = core.build_profitability_bucket_atlas(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        actual_frame=actual,
        replay_bundle=(replay, "fixture_replay.csv", ""),
    )
    summary = core.summarize_profitability_bucket_atlas(atlas)

    row = atlas[atlas["liquidity_bucket"].eq("liquidity_deep")].iloc[0]
    assert row["status"] == "WARN"
    assert row["actual_bucket_status"] == "BLOCK"
    assert row["replay_bucket_status"] == "PASS"
    assert row["primary_gap"] == "actual_bucket_negative_or_weak"
    assert summary["status"] == "no_actual_and_replay_bucket_pass"
    assert summary["pass_rows"] == 0


def test_outcome_evidence_audit_surfaces_unrealized_forward_ledgers(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "strategy": "Bull Put Credit Spread",
                "report_date": "2026-06-05",
                "outcome_status": "OPEN_REVIEW_REQUIRED",
                "realized_pnl": "",
            }
        ]
    ).to_csv(out_dir / "codexuw_execute_outcome_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "strategy": "Bull Put Credit Spread",
                "report_date": "2026-06-05",
                "outcome_status": "CONDITIONAL_NOT_FILLED",
                "realized_pnl": "",
            }
        ]
    ).to_csv(out_dir / "codexuw_recommendation_outcome_ledger.csv", index=False)
    closed_dir = out_dir / "schwab_pull_state"
    closed_dir.mkdir()
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        json.dumps(
            {
                "ticker": "AAPL",
                "strategy": "short_put",
                "realized_pnl": -125.0,
                "closed_at": "2026-06-06T14:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = core.build_outcome_evidence_audit(
        tmp_path,
        pd.DataFrame([{"ticker": "AAPL"}]),
        pd.DataFrame(),
        as_of_date="2026-06-12",
    )
    summary = core.summarize_outcome_evidence_audit(audit)

    by_source = {row["source"]: row for _, row in audit.iterrows()}
    assert by_source["codexuw_execute_outcome_ledger"]["status"] == "BLOCK"
    assert by_source["codexuw_execute_outcome_ledger"]["blocker"] == "no_realized_pnl_rows"
    assert by_source["codexuw_recommendation_outcome_ledger"]["status"] == "BLOCK"
    assert by_source["schwab_closed_trades"]["status"] == "PASS"
    assert by_source["schwab_closed_trades"]["realized_pnl_count"] == 1
    assert summary["realized_pnl_count"] == 1
    assert summary["contributing_sources"] == ["schwab_closed_trades"]
    assert summary["forward_sources_without_realized_outcomes"] == [
        "codexuw_execute_outcome_ledger",
        "codexuw_recommendation_outcome_ledger",
    ]
    assert summary["blocking_sources"] == [
        "codexuw_execute_outcome_ledger",
        "codexuw_recommendation_outcome_ledger",
    ]


def test_outcome_evidence_audit_includes_broker_backfill_context(tmp_path: Path) -> None:
    broker_matched = pd.DataFrame(
        [
            {"ticker": "WMT", "realized_pnl": 120.0, "closed_at": "2026-05-12T14:00:00+00:00"},
            {"ticker": "META", "realized_pnl": -60.0, "closed_at": "2026-05-13T14:00:00+00:00"},
        ],
        columns=core.BROKER_MATCHED_OUTCOME_COLUMNS,
    )

    audit = core.build_outcome_evidence_audit(
        tmp_path,
        pd.DataFrame([{"ticker": "WMT"}]),
        pd.DataFrame(),
        broker_matched_outcomes=broker_matched,
    )
    summary = core.summarize_outcome_evidence_audit(audit)
    broker_row = audit[audit["source"].eq("broker_matched_outcomes")].iloc[0]

    assert broker_row["realized_pnl_count"] == 2
    assert broker_row["current_ticker_realized_count"] == 1
    assert broker_row["latest_evidence_date"] == "2026-05-13"
    assert broker_row["contributes_to_expectancy"] is False or broker_row["contributes_to_expectancy"] == False
    assert "Exact broker-to-recommendation backfill outcomes" in broker_row["note"]
    assert summary["broker_backfill_realized_count"] == 2
    assert summary["broker_backfill_status"] in {"block", "warn", "pass"}


def test_profitability_evidence_backfill_plan_names_exact_samples() -> None:
    outcome_audit = pd.DataFrame(
        [
            {
                "source": "codexuw_execute_outcome_ledger",
                "source_path": "/tmp/codexuw_execute_outcome_ledger.csv",
                "evidence_type": "forward_realized_outcomes",
                "status": "BLOCK",
                "row_count": 61,
                "realized_pnl_count": 0,
                "current_ticker_realized_count": 0,
                "open_or_unrealized_count": 61,
                "note": "rows are open/unrealized",
            }
        ]
    )
    broker_match_audit = pd.DataFrame(
        [
            {
                "match_source": "options_agent_history",
                "match_status": "BLOCK",
                "closed_trade_key": "META|2026-06-18|1",
            }
        ]
    )
    broker_matched = pd.DataFrame(
        [
            {
                "ticker": "META",
                "realized_pnl": -275.0,
                "match_sources": "options_agent_history",
                "match_scope": "options_agent_green_history",
            },
            {
                "ticker": "UPS",
                "realized_pnl": 200.0,
                "match_sources": "options_agent_history",
                "match_scope": "options_agent_green_history",
            },
        ]
    )
    gap_plan = pd.DataFrame(
        [
            {
                "exact_bucket_key": "short_put|CREDIT|bullish|mixed|dte_31_60|credit_rich|liquidity_deep",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "current_ticket_count": 2,
                "current_tickers": "BX,VRT",
                "actual_support_status": "WARN",
                "actual_support_scope": "actual_route_bucket",
                "actual_support_sample_size": 4,
                "actual_support_sample_gap": 26,
                "actual_support_avg_pnl": -232.0,
                "actual_support_profit_factor": 0.012,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": 33,
                "replay_bucket_sample_gap": 0,
                "primary_gap": "actual_closed_outcomes_sample_gap",
                "next_evidence_needed": "Need 26 more positive closed/forward outcomes in exact bucket.",
                "source_path": "/tmp/replay.csv",
            }
        ]
    )
    strategy_atlas = pd.DataFrame(
        [
            {
                "scope": "strategy_family",
                "strategy_family": "vertical_spread",
                "status": "BLOCK",
                "sample_size": 78,
                "avg_pnl": -36.03,
                "profit_factor": 0.677,
                "source_tickers": "META,MSFT,NOW,UPS",
                "current_ticket_count": 19,
                "source_path": "/tmp/closed.jsonl",
                "note": "Actual Schwab closed-trade cohort for vertical_spread.",
            },
            {
                "scope": "strategy_family",
                "strategy_family": "short_put",
                "status": "PASS",
                "sample_size": 36,
                "avg_pnl": 43.92,
                "profit_factor": 1.295,
                "source_tickers": "AAPL,AMD",
                "current_ticket_count": 11,
                "source_path": "/tmp/closed.jsonl",
                "note": "Actual Schwab closed-trade cohort for short_put.",
            },
        ]
    )

    plan = core.build_profitability_evidence_backfill_plan(
        outcome_audit,
        broker_match_audit,
        broker_matched,
        gap_plan,
        strategy_outcome_atlas=strategy_atlas,
    )
    summary = core.summarize_profitability_evidence_backfill_plan(plan)
    by_gap = plan.set_index("evidence_gap")

    assert plan["evidence_gap"].iloc[0] == "current_strategy_family_negative_or_weak"
    assert by_gap.loc["current_strategy_family_negative_or_weak", "strategy_family"] == "vertical_spread"
    assert by_gap.loc["current_strategy_family_negative_or_weak", "avg_pnl"] == -36.03
    assert by_gap.loc["options_agent_broker_attribution_negative_or_weak", "sample_gap"] == 28
    assert by_gap.loc["actual_closed_outcomes_sample_gap", "sample_gap"] == 26
    assert by_gap.loc["actual_closed_outcomes_sample_gap", "current_tickers"] == "BX,VRT"
    assert by_gap.loc["forward_ledger_realized_pnl_missing", "sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert summary["status"] == "block"
    assert summary["total_sample_gap"] == 84


def test_profitability_evidence_backfill_plan_counts_positive_options_agent_green_sample_gap() -> None:
    broker_match_audit = pd.DataFrame(
        [
            {
                "match_source": "options_agent_history",
                "match_status": "BLOCK",
                "closed_trade_key": "META|2026-06-18|1",
            }
        ]
    )
    broker_matched = pd.DataFrame(
        [
            {
                "ticker": "PG",
                "realized_pnl": 272.0,
                "match_sources": "options_agent_history",
                "match_scope": "options_agent_green_history",
            },
            {
                "ticker": "UPS",
                "realized_pnl": 200.0,
                "match_sources": "options_agent_history",
                "match_scope": "options_agent_green_history",
            },
            {
                "ticker": "META",
                "realized_pnl": -275.0,
                "match_sources": "options_agent_history",
                "match_scope": "options_agent_diagnostic_history",
            },
        ]
    )

    plan = core.build_profitability_evidence_backfill_plan(
        pd.DataFrame(),
        broker_match_audit,
        broker_matched,
        pd.DataFrame(),
    )
    by_gap = plan.set_index("evidence_gap")
    row = by_gap.loc["options_agent_broker_attribution_sample_gap"]

    assert row["ticker_scope"] == "options_agent_green_history"
    assert row["current_sample_size"] == 2
    assert row["sample_gap"] == core.MIN_EXPECTANCY_SAMPLE_SIZE - 2
    assert row["avg_pnl"] == 236.0
    assert row["profit_factor"] == "inf"
    assert "sample=2" in row["note"]
    assert "Diagnostic yellow/recheck Options-Agent matches are excluded" in row["note"]


def test_broker_outcome_match_audit_requires_unique_exact_contract_match(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    closed_dir = out_dir / "schwab_pull_state"
    closed_dir.mkdir()
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "orderId": 111,
                        "orderType": "NET_CREDIT",
                        "orderLegCollection": [
                            {
                                "instruction": "SELL_TO_OPEN",
                                "positionEffect": "OPENING",
                                "orderLegType": "OPTION",
                                "instrument": {
                                    "symbol": "WMT   260618P00125000",
                                    "putCall": "PUT",
                                },
                            },
                            {
                                "instruction": "BUY_TO_OPEN",
                                "positionEffect": "OPENING",
                                "orderLegType": "OPTION",
                                "instrument": {
                                    "symbol": "WMT   260618P00120000",
                                    "putCall": "PUT",
                                },
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "orderId": 222,
                        "orderType": "LIMIT",
                        "orderLegCollection": [
                            {
                                "instruction": "SELL_TO_OPEN",
                                "positionEffect": "OPENING",
                                "orderLegType": "OPTION",
                                "instrument": {
                                    "symbol": "AAPL  260626P00300000",
                                    "putCall": "PUT",
                                },
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "orderId": 333,
                        "orderType": "NET_CREDIT",
                        "orderLegCollection": [
                            {
                                "instruction": "SELL_TO_OPEN",
                                "positionEffect": "OPENING",
                                "orderLegType": "OPTION",
                                "instrument": {
                                    "symbol": "MSFT  260618P00325000",
                                    "putCall": "PUT",
                                },
                            },
                            {
                                "instruction": "BUY_TO_OPEN",
                                "positionEffect": "OPENING",
                                "orderLegType": "OPTION",
                                "instrument": {
                                    "symbol": "MSFT  260618P00315000",
                                    "putCall": "PUT",
                                },
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "orderId": 444,
                        "orderType": "LIMIT",
                        "orderLegCollection": [
                            {
                                "instruction": "SELL_TO_OPEN",
                                "positionEffect": "OPENING",
                                "orderLegType": "OPTION",
                                "instrument": {
                                    "symbol": "TSLA  260626P00400000",
                                    "putCall": "PUT",
                                },
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "orderId": 555,
                        "orderType": "NET_CREDIT",
                        "orderLegCollection": [
                            {
                                "instruction": "SELL_TO_OPEN",
                                "positionEffect": "OPENING",
                                "orderLegType": "OPTION",
                                "instrument": {
                                    "symbol": "META  260618P00600000",
                                    "putCall": "PUT",
                                },
                            },
                            {
                                "instruction": "BUY_TO_OPEN",
                                "positionEffect": "OPENING",
                                "orderLegType": "OPTION",
                                "instrument": {
                                    "symbol": "META  260618P00590000",
                                    "putCall": "PUT",
                                },
                            },
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ticker": "WMT",
                        "strategy": "vertical_spread",
                        "expiry": "2026-06-18",
                        "opened_at": "2026-05-08T14:21:05+00:00",
                        "closed_at": "2026-05-27T14:26:44+00:00",
                        "realized_pnl": -225.0,
                        "entry_order_ids": ["111"],
                    }
                ),
                json.dumps(
                    {
                        "ticker": "AAPL",
                        "strategy": "short_put",
                        "expiry": "2026-06-26",
                        "opened_at": "2026-05-28T13:54:20+00:00",
                        "closed_at": "2026-06-10T13:55:02+00:00",
                        "realized_pnl": -956.0,
                        "entry_order_ids": ["222"],
                    }
                ),
                json.dumps(
                    {
                        "ticker": "MSFT",
                        "strategy": "vertical_spread",
                        "expiry": "2026-06-18",
                        "opened_at": "2026-05-09T14:21:05+00:00",
                        "closed_at": "2026-05-28T14:26:44+00:00",
                        "realized_pnl": 80.0,
                        "entry_order_ids": ["333"],
                    }
                ),
                json.dumps(
                    {
                        "ticker": "TSLA",
                        "strategy": "short_put",
                        "expiry": "2026-06-26",
                        "opened_at": "2026-05-28T14:21:05+00:00",
                        "closed_at": "2026-06-04T14:26:44+00:00",
                        "realized_pnl": 125.0,
                        "entry_order_ids": ["444"],
                    }
                ),
                json.dumps(
                    {
                        "ticker": "META",
                        "strategy": "vertical_spread",
                        "expiry": "2026-06-18",
                        "opened_at": "2026-05-14T14:14:15+00:00",
                        "closed_at": "2026-06-08T13:42:20+00:00",
                        "realized_pnl": -275.0,
                        "entry_order_ids": ["555"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "report_date": "2026-05-08",
                "trade_key": "run_a|2026-05-08|WMT|Bull Put Credit Spread|2026-06-18|WMT   260618P00125000|WMT   260618P00120000",
                "ticker": "WMT",
                "strategy": "Bull Put Credit Spread",
                "lane": "Execute Now",
            }
        ]
    ).to_csv(out_dir / "codexuw_execute_outcome_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "run_b",
                "report_date": "2026-05-08",
                "trade_key": "run_b|2026-05-08|WMT|Bull Put Credit Spread|2026-06-18|WMT   260618P00125000|WMT   260618P00120000",
                "ticker": "WMT",
                "strategy": "Bull Put Credit Spread",
                "lane": "Execute Now",
            },
            {
                "run_id": "run_c",
                "report_date": "2026-05-08",
                "trade_key": "run_c|2026-05-08|WMT|Bull Put Credit Spread|2026-06-18|WMT   260618P00125000|WMT   260618P00120000",
                "ticker": "WMT",
                "strategy": "Bull Put Credit Spread",
                "lane": "Execute Now",
            },
            {
                "run_id": "run_d",
                "report_date": "2026-05-08",
                "trade_key": "run_d|2026-05-08|MSFT|Bull Put Credit Spread|2026-06-18|MSFT  260618P00325000|MSFT  260618P00315000",
                "ticker": "MSFT",
                "strategy": "Bull Put Credit Spread",
                "lane": "Execute Now",
            },
            {
                "run_id": "run_e",
                "report_date": "2026-05-09",
                "trade_key": "run_e|2026-05-09|MSFT|Bull Put Credit Spread|2026-06-18|MSFT  260618P00325000|MSFT  260618P00315000",
                "ticker": "MSFT",
                "strategy": "Bull Put Credit Spread",
                "lane": "Execute Now",
            },
            {
                "run_id": "run_future",
                "report_date": "2026-06-04",
                "trade_key": "run_future|2026-06-04|AAPL|Short Put|2026-06-26|AAPL  260626P00300000",
                "ticker": "AAPL",
                "strategy": "Short Put",
                "lane": "Execute Now",
            },
        ]
    ).to_csv(out_dir / "codexuw_recommendation_outcome_ledger.csv", index=False)
    options_history = out_dir / "options_agent"
    pre_entry = options_history / "2026-05-27"
    pre_entry.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "TSLA",
                "strategy": "short_put",
                "final_action": RecommendationStatus.ENTER.value,
                "execution_status": "ready",
                "setup_quality_status": "qualified",
                "visible_in_final_board": True,
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "order_readiness": "ready_to_enter",
                "live_validation_status": "PASS",
                "trade_plan": "SELL 1 TSLA 2026-06-26 400 Put @ 5.00 CREDIT",
                "sell_leg": "SELL 1 TSLA 2026-06-26 400 Put",
                "expiry": "2026-06-26",
            }
        ]
    ).to_csv(pre_entry / "decision_board.csv", index=False)
    (pre_entry / "options_agent_manifest_2026-05-27.json").write_text(
        json.dumps({"as_of": "2026-05-27", "pipeline_name": "Options Agent"}),
        encoding="utf-8",
    )
    meta_first = options_history / "current_code_full_v041_guardrail_2026-05-14"
    meta_first.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "META",
                "strategy": "complete_agentic_reviews_then_live_recheck",
                "final_action": RecommendationStatus.WAIT_FOR_PRICE.value,
                "execution_status": "waiting_for_price",
                "setup_quality_status": "qualified",
                "visible_in_final_board": True,
                "target_order_status": "target_order_candidate",
                "trade_plan": "SELL 1 META 2026-06-18 600 Put / BUY 1 META 2026-06-18 590 Put @ 2.40 CREDIT",
                "sell_leg": "SELL 1 META 2026-06-18 600 Put",
                "buy_leg": "BUY 1 META 2026-06-18 590 Put",
                "expiry": "2026-06-18",
            }
        ]
    ).to_csv(meta_first / "trade_tickets.csv", index=False)
    (meta_first / "options_agent_manifest_2026-05-14.json").write_text(
        json.dumps({"as_of": "2026-05-14", "pipeline_name": "Options Agent"}),
        encoding="utf-8",
    )
    meta_second = options_history / "current_code_full_v044_agentic_2026-05-14"
    meta_second.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "META",
                "strategy": "load_portfolio_then_live_recheck",
                "final_action": RecommendationStatus.WAIT_FOR_PRICE.value,
                "execution_status": "waiting_for_price",
                "setup_quality_status": "qualified",
                "visible_in_final_board": True,
                "target_order_status": "target_order_candidate",
                "trade_plan": "SELL 1 META 2026-06-18 600 Put / BUY 1 META 2026-06-18 590 Put @ 2.40 CREDIT",
                "sell_leg": "SELL 1 META 2026-06-18 600 Put",
                "buy_leg": "BUY 1 META 2026-06-18 590 Put",
                "expiry": "2026-06-18",
            }
        ]
    ).to_csv(meta_second / "trade_tickets.csv", index=False)
    (meta_second / "options_agent_manifest_2026-05-14.json").write_text(
        json.dumps({"as_of": "2026-05-14", "pipeline_name": "Options Agent"}),
        encoding="utf-8",
    )
    future_entry = options_history / "2026-06-04"
    future_entry.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "strategy": "short_put",
                "final_action": RecommendationStatus.ENTER.value,
                "execution_status": "waiting_for_price",
                "setup_quality_status": "qualified",
                "visible_in_final_board": True,
                "trade_plan": "SELL 1 AAPL 2026-06-26 300 Put @ 4.00 CREDIT",
                "sell_leg": "SELL 1 AAPL 2026-06-26 300 Put",
                "expiry": "2026-06-26",
            }
        ]
    ).to_csv(future_entry / "decision_board.csv", index=False)
    (future_entry / "options_agent_manifest_2026-06-04.json").write_text(
        json.dumps({"as_of": "2026-06-04", "pipeline_name": "Options Agent"}),
        encoding="utf-8",
    )

    audit = core.build_broker_outcome_match_audit(tmp_path)
    summary = core.summarize_broker_outcome_match_audit(audit)

    execute_wmt = audit[
        audit["match_source"].eq("codexuw_execute_outcome_ledger")
        & audit["ticker"].eq("WMT")
    ].iloc[0]
    recommendation_wmt = audit[
        audit["match_source"].eq("codexuw_recommendation_outcome_ledger")
        & audit["ticker"].eq("WMT")
    ].iloc[0]
    recommendation_msft = audit[
        audit["match_source"].eq("codexuw_recommendation_outcome_ledger")
        & audit["ticker"].eq("MSFT")
    ].iloc[0]
    recommendation_aapl = audit[
        audit["match_source"].eq("codexuw_recommendation_outcome_ledger")
        & audit["ticker"].eq("AAPL")
    ].iloc[0]
    options_tsla = audit[
        audit["match_source"].eq("options_agent_history")
        & audit["ticker"].eq("TSLA")
    ].iloc[0]
    options_aapl = audit[
        audit["match_source"].eq("options_agent_history")
        & audit["ticker"].eq("AAPL")
    ].iloc[0]
    options_meta = audit[
        audit["match_source"].eq("options_agent_history")
        & audit["ticker"].eq("META")
    ].iloc[0]
    aapl_rows = audit[audit["ticker"].eq("AAPL")]

    assert execute_wmt["match_status"] == "PASS"
    assert execute_wmt["can_backfill_realized_pnl"] is True or execute_wmt["can_backfill_realized_pnl"] == True
    assert recommendation_wmt["match_status"] == "PASS"
    assert recommendation_wmt["matched_recommendation_count"] == 1
    assert "run_b" in recommendation_wmt["matched_run_ids"]
    assert "run_c" in recommendation_wmt["matched_run_ids"]
    assert recommendation_msft["match_status"] == "WARN"
    assert recommendation_msft["blocker"] == "ambiguous_duplicate_contract_matches"
    assert recommendation_aapl["match_status"] == "BLOCK"
    assert recommendation_aapl["blocker"] == "no_pre_entry_exact_contract_match"
    assert "leakage-safe realized P/L backfill is blocked" in recommendation_aapl["note"]
    assert options_tsla["match_status"] == "BLOCK"
    assert options_tsla["blocker"] == "legacy_options_agent_history_not_prospective"
    assert options_tsla["matched_report_dates"] == "2026-05-27"
    assert "2026-05-27/decision_board.csv" in options_tsla["matched_run_ids"]
    assert options_tsla["matched_readiness_scope"] == "green_ready"
    assert options_tsla["matched_ready_to_enter_count"] == 1
    assert options_meta["match_status"] == "BLOCK"
    assert options_meta["blocker"] == "legacy_options_agent_history_not_prospective"
    assert options_meta["matched_recommendation_count"] == 1
    assert options_meta["matched_readiness_scope"] == "yellow_or_recheck"
    assert options_meta["matched_ready_to_enter_count"] == 0
    assert options_meta["matched_target_order_count"] >= 1
    assert "current_code_full_v041_guardrail_2026-05-14/trade_tickets.csv" in options_meta["matched_run_ids"]
    assert "current_code_full_v044_agentic_2026-05-14/trade_tickets.csv" in options_meta["matched_run_ids"]
    assert "diagnostic only" in options_meta["note"]
    assert options_aapl["match_status"] == "BLOCK"
    assert options_aapl["blocker"] == "no_pre_entry_exact_contract_match"
    assert set(aapl_rows["match_status"]) == {"BLOCK"}
    assert {"no_exact_contract_match", "no_pre_entry_exact_contract_match", "forward_registry_missing"}.issubset(
        set(aapl_rows["blocker"])
    )
    assert summary["backfillable_rows"] == 2
    assert summary["backfillable_closed_trades"] == 1
    assert summary["ambiguous_rows"] == 1
    assert summary["ambiguous_closed_trades"] == 1
    assert summary["unmatched_closed_trades"] == 3
    assert summary["backfillable_by_source"] == {
        "codexuw_execute_outcome_ledger": 1,
        "codexuw_recommendation_outcome_ledger": 1,
    }
    assert summary["backfillable_closed_trades_by_source"] == {
        "codexuw_execute_outcome_ledger": 1,
        "codexuw_recommendation_outcome_ledger": 1,
    }
    assert summary["blocked_by_source"] == {
        "codexuw_execute_outcome_ledger": 4,
        "codexuw_recommendation_outcome_ledger": 3,
        "options_agent_forward_registry": 5,
        "options_agent_history": 5,
    }
    assert summary["unmatched_closed_trades_by_source"] == {
        "codexuw_execute_outcome_ledger": 4,
        "codexuw_recommendation_outcome_ledger": 3,
        "options_agent_forward_registry": 5,
        "options_agent_history": 5,
    }
    assert summary["ambiguous_by_source"] == {"codexuw_recommendation_outcome_ledger": 1}


def test_options_agent_broker_match_uses_latest_pre_entry_recommendation(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    closed_dir = out_dir / "schwab_pull_state"
    closed_dir.mkdir()
    (closed_dir / "raw_orders_acct_3326.jsonl").write_text(
        json.dumps(
            {
                "orderId": 999,
                "orderType": "NET_DEBIT",
                "orderLegCollection": [
                    {
                        "instruction": "BUY_TO_OPEN",
                        "positionEffect": "OPENING",
                        "orderLegType": "OPTION",
                        "instrument": {
                            "symbol": "PG    260618C00145000",
                            "putCall": "CALL",
                        },
                    },
                    {
                        "instruction": "SELL_TO_OPEN",
                        "positionEffect": "OPENING",
                        "orderLegType": "OPTION",
                        "instrument": {
                            "symbol": "PG    260618C00150000",
                            "putCall": "CALL",
                        },
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        json.dumps(
            {
                "ticker": "PG",
                "strategy": "vertical_spread",
                "expiry": "2026-06-18",
                "opened_at": "2026-06-05T14:17:08+00:00",
                "closed_at": "2026-06-15T16:33:45+00:00",
                "realized_pnl": 272.0,
                "entry_order_ids": ["999"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    options_history = out_dir / "options_agent"
    older = options_history / "2026-05-28"
    older.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "ticker": "PG",
                "strategy": "bull_call_debit",
                "final_action": RecommendationStatus.WAIT_FOR_PRICE.value,
                "execution_status": "waiting_for_price",
                "setup_quality_status": "qualified",
                "visible_in_final_board": True,
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_live_recheck",
                "trade_plan": "BUY 1 PG 2026-06-18 145 Call / SELL 1 PG 2026-06-18 150 Call @ 1.40 DEBIT",
                "buy_leg": "BUY 1 PG 2026-06-18 145 Call",
                "sell_leg": "SELL 1 PG 2026-06-18 150 Call",
                "expiry": "2026-06-18",
            }
        ]
    ).to_csv(older / "trade_tickets.csv", index=False)
    (older / "options_agent_manifest_2026-05-28.json").write_text(
        json.dumps({"as_of": "2026-05-28", "pipeline_name": "Options Agent"}),
        encoding="utf-8",
    )
    latest = options_history / "2026-06-04_overlay_2026-06-05"
    latest.mkdir()
    latest_row = {
        "ticker": "PG",
        "strategy": "bull_call_debit",
        "final_action": RecommendationStatus.ENTER.value,
        "execution_status": "ready",
        "setup_quality_status": "qualified",
        "visible_in_final_board": True,
        "ready_to_enter": True,
        "target_order_status": "ready_to_enter",
        "order_readiness": "ready_to_enter",
        "live_validation_status": "PASS",
        "trade_plan": "BUY 1 PG 2026-06-18 145 Call / SELL 1 PG 2026-06-18 150 Call @ 1.55 DEBIT",
        "buy_leg": "BUY 1 PG 2026-06-18 145 Call",
        "sell_leg": "SELL 1 PG 2026-06-18 150 Call",
        "expiry": "2026-06-18",
    }
    pd.DataFrame([latest_row]).to_csv(latest / "trade_tickets.csv", index=False)
    pd.DataFrame([latest_row]).to_csv(latest / "decision_board.csv", index=False)
    (latest / "options_agent_manifest_2026-06-04.json").write_text(
        json.dumps({"as_of": "2026-06-04", "pipeline_name": "Options Agent"}),
        encoding="utf-8",
    )

    audit = core.build_broker_outcome_match_audit(tmp_path)
    row = audit[audit["match_source"].eq("options_agent_history")].iloc[0]

    assert row["match_status"] == "BLOCK"
    assert row["blocker"] == "legacy_options_agent_history_not_prospective"
    assert row["matched_report_dates"] == "2026-06-04"
    assert row["matched_readiness_scope"] == "green_ready"
    assert row["matched_ready_to_enter_count"] == 1
    assert row["matched_target_order_count"] == 0
    assert row["matched_live_validation_status"] == "PASS"
    assert "diagnostic only" in row["note"]
    assert "2026-05-28/trade_tickets.csv" not in row["matched_run_ids"]
    assert "2026-06-04_overlay_2026-06-05/trade_tickets.csv" in row["matched_run_ids"]


def test_broker_matched_outcomes_deduplicates_same_closed_trade_across_sources() -> None:
    match_audit = pd.DataFrame(
        [
            {
                "match_source": "codexuw_execute_outcome_ledger",
                "match_status": "PASS",
                "closed_trade_key": "META|2026-06-18|100|open|close",
                "ticker": "META",
                "strategy": "vertical_spread",
                "strategy_family": "vertical_spread",
                "expiry": "2026-06-18",
                "opened_at": "open",
                "closed_at": "close",
                "realized_pnl": -275.0,
                "entry_order_ids": "100",
                "entry_symbols": "META260618P00590000,META260618P00600000",
                "matched_trade_keys": "execute_key",
                "matched_report_dates": "",
                "matched_run_ids": "execute_run",
                "can_backfill_realized_pnl": True,
            },
            {
                "match_source": "codexuw_recommendation_outcome_ledger",
                "match_status": "PASS",
                "closed_trade_key": "META|2026-06-18|100|open|close",
                "ticker": "META",
                "strategy": "vertical_spread",
                "strategy_family": "vertical_spread",
                "expiry": "2026-06-18",
                "opened_at": "open",
                "closed_at": "close",
                "realized_pnl": -275.0,
                "entry_order_ids": "100",
                "entry_symbols": "META260618P00590000,META260618P00600000",
                "matched_trade_keys": "recommendation_key",
                "matched_report_dates": "2026-06-04",
                "matched_run_ids": "recommendation_run",
                "can_backfill_realized_pnl": True,
            },
        ],
        columns=core.BROKER_OUTCOME_MATCH_AUDIT_COLUMNS,
    )

    outcomes = core.build_broker_matched_outcomes(match_audit)
    summary = core.summarize_broker_matched_outcomes(outcomes)

    assert len(outcomes) == 1
    row = outcomes.iloc[0]
    assert row["realized_pnl"] == -275.0
    assert row["match_sources"] == "codexuw_execute_outcome_ledger,codexuw_recommendation_outcome_ledger"
    assert row["match_scope"] == "historical_recommendation_ledger"
    assert "execute_key" in row["matched_trade_keys"]
    assert "recommendation_key" in row["matched_trade_keys"]
    assert summary["sample_size"] == 1
    assert summary["status"] == "not_positive"
    assert summary["total_pnl"] == -275.0
    assert summary["match_scope_counts"] == {"historical_recommendation_ledger": 1}
    assert summary["options_agent_sample_size"] == 0


def test_broker_matched_outcomes_split_options_agent_green_and_diagnostic_history() -> None:
    match_audit = pd.DataFrame(
        [
            {
                "match_source": "options_agent_history",
                "match_status": "PASS",
                "closed_trade_key": "TSLA|2026-06-26|444|open|close",
                "ticker": "TSLA",
                "strategy": "short_put",
                "strategy_family": "short_put",
                "expiry": "2026-06-26",
                "opened_at": "open",
                "closed_at": "close",
                "realized_pnl": 125.0,
                "entry_order_ids": "444",
                "entry_symbols": "TSLA260626P00400000",
                "matched_trade_keys": "green_key",
                "matched_report_dates": "2026-05-27",
                "matched_run_ids": "2026-05-27/decision_board.csv",
                "matched_readiness_scope": "green_ready",
                "matched_ready_to_enter_count": 1,
                "matched_target_order_count": 1,
                "matched_order_readiness": "ready_to_enter",
                "matched_live_validation_status": "PASS",
                "can_backfill_realized_pnl": True,
            },
            {
                "match_source": "options_agent_history",
                "match_status": "PASS",
                "closed_trade_key": "META|2026-06-18|555|open|close",
                "ticker": "META",
                "strategy": "vertical_spread",
                "strategy_family": "vertical_spread",
                "expiry": "2026-06-18",
                "opened_at": "open",
                "closed_at": "close",
                "realized_pnl": -275.0,
                "entry_order_ids": "555",
                "entry_symbols": "META260618P00590000,META260618P00600000",
                "matched_trade_keys": "yellow_key",
                "matched_report_dates": "2026-05-14",
                "matched_run_ids": "current_code_full_v041_guardrail_2026-05-14/trade_tickets.csv",
                "matched_readiness_scope": "yellow_or_recheck",
                "matched_ready_to_enter_count": 0,
                "matched_target_order_count": 1,
                "matched_order_readiness": "target_order_after_live_recheck",
                "matched_live_validation_status": "MARKET_CLOSED_RECHECK",
                "can_backfill_realized_pnl": True,
            },
        ],
        columns=core.BROKER_OUTCOME_MATCH_AUDIT_COLUMNS,
    )

    outcomes = core.build_broker_matched_outcomes(match_audit)
    summary = core.summarize_broker_matched_outcomes(outcomes)

    by_ticker = {row["ticker"]: row for _, row in outcomes.iterrows()}
    assert by_ticker["TSLA"]["match_scope"] == "options_agent_green_history"
    assert by_ticker["META"]["match_scope"] == "options_agent_diagnostic_history"
    assert summary["options_agent_sample_size"] == 1
    assert summary["options_agent_avg_pnl"] == 125.0
    assert summary["options_agent_diagnostic_sample_size"] == 1
    assert summary["options_agent_diagnostic_avg_pnl"] == -275.0


def test_broker_backfilled_forward_outcomes_deduplicate_closed_trade_across_ledgers() -> None:
    match_audit = pd.DataFrame(
        [
            {
                "match_source": "codexuw_execute_outcome_ledger",
                "match_status": "PASS",
                "closed_trade_key": "META|2026-06-18|100|open|close",
                "ticker": "META",
                "strategy": "vertical_spread",
                "strategy_family": "vertical_spread",
                "expiry": "2026-06-18",
                "opened_at": "open",
                "closed_at": "close",
                "realized_pnl": -275.0,
                "entry_order_ids": "100",
                "entry_symbols": "META260618P00590000,META260618P00600000",
                "matched_trade_keys": "execute_key",
                "matched_report_dates": "2026-05-14",
                "matched_run_ids": "execute_run",
                "can_backfill_realized_pnl": True,
                "source_path": "closed=/tmp/closed.jsonl; raw_orders=/tmp/raw.jsonl",
            },
            {
                "match_source": "codexuw_recommendation_outcome_ledger",
                "match_status": "PASS",
                "closed_trade_key": "META|2026-06-18|100|open|close",
                "ticker": "META",
                "strategy": "vertical_spread",
                "strategy_family": "vertical_spread",
                "expiry": "2026-06-18",
                "opened_at": "open",
                "closed_at": "close",
                "realized_pnl": -275.0,
                "entry_order_ids": "100",
                "entry_symbols": "META260618P00590000,META260618P00600000",
                "matched_trade_keys": "recommendation_key",
                "matched_report_dates": "2026-05-14",
                "matched_run_ids": "recommendation_run",
                "can_backfill_realized_pnl": True,
                "source_path": "closed=/tmp/closed.jsonl; raw_orders=/tmp/raw.jsonl",
            },
        ],
        columns=core.BROKER_OUTCOME_MATCH_AUDIT_COLUMNS,
    )

    backfilled = core.build_broker_backfilled_forward_outcomes(match_audit)
    summary = core.summarize_broker_backfilled_forward_outcomes(backfilled)

    assert len(backfilled) == 1
    row = backfilled.iloc[0]
    assert row["realized_pnl"] == -275.0
    assert row["source_ledger"] == "codexuw_execute_outcome_ledger,codexuw_recommendation_outcome_ledger"
    assert "execute_key" in row["matched_trade_keys"]
    assert "recommendation_key" in row["matched_trade_keys"]
    assert summary["sample_size"] == 1
    assert summary["status"] == "not_positive"
    assert summary["profit_factor"] == 0.0


def test_broker_backfilled_forward_outcomes_feed_expectancy_and_audit(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame(
        [
            {
                "ticker": "META",
                "strategy": "Bull Put Credit Spread",
                "report_date": "2026-05-14",
                "outcome_status": "OPEN_REVIEW_REQUIRED",
                "realized_pnl": "",
            }
        ]
    ).to_csv(out_dir / "codexuw_execute_outcome_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                "ticker": "META",
                "strategy": "Bull Put Credit Spread",
                "report_date": "2026-05-14",
                "outcome_status": "OPEN_REVIEW_REQUIRED",
                "realized_pnl": "",
            }
        ]
    ).to_csv(out_dir / "codexuw_recommendation_outcome_ledger.csv", index=False)
    closed_dir = out_dir / "schwab_pull_state"
    closed_dir.mkdir()
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text("", encoding="utf-8")

    backfilled = pd.DataFrame(
        [
            {
                "source_ledger": "codexuw_execute_outcome_ledger",
                "closed_trade_key": "META|2026-06-18|100|open|close",
                "ticker": "META",
                "strategy": "vertical_spread",
                "strategy_family": "vertical_spread",
                "expiry": "2026-06-18",
                "opened_at": "2026-05-14T14:00:00+00:00",
                "closed_at": "2026-06-08T14:00:00+00:00",
                "realized_pnl": -275.0,
                "entry_order_ids": "100",
                "entry_symbols": "META260618P00590000,META260618P00600000",
            }
        ],
        columns=core.BROKER_BACKFILLED_FORWARD_OUTCOME_COLUMNS,
    )

    evidence = core.build_expectancy_evidence(
        tmp_path,
        pd.DataFrame([{"ticker": "META"}]),
        pd.DataFrame(),
        broker_backfilled_forward_outcomes=backfilled,
    )
    audit = core.build_outcome_evidence_audit(
        tmp_path,
        pd.DataFrame([{"ticker": "META"}]),
        pd.DataFrame(),
        broker_backfilled_forward_outcomes=backfilled,
    )
    audit_summary = core.summarize_outcome_evidence_audit(audit)

    backfilled_evidence = evidence[evidence["source"].eq("broker_backfilled_forward_outcomes")].iloc[0]
    assert backfilled_evidence["evidence_type"] == "forward_realized_outcomes"
    assert backfilled_evidence["status"] == "BLOCK"
    assert backfilled_evidence["sample_size"] == 1
    assert backfilled_evidence["avg_pnl"] == -275.0
    assert audit_summary["forward_sources_without_realized_outcomes"] == []
    assert audit_summary["forward_broker_backfill_realized_count"] == 1


def test_actual_forward_outcomes_do_not_double_count_broker_backfill_and_closed_trade(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        json.dumps(
            {
                "ticker": "RKLB",
                "strategy": "Bull Call Debit Spread",
                "realized_pnl": 155.0,
                "entry_order_ids": ["1006322921251"],
                "opened_at": "2026-05-12T14:03:15+00:00",
                "expiry": "2026-05-22",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    backfilled = pd.DataFrame(
        [
            {
                "ticker": "RKLB",
                "strategy": "Bull Call Debit Spread",
                "strategy_family": "vertical_spread",
                "realized_pnl": 155.0,
                "entry_order_ids": "1006322921251",
                "opened_at": "2026-05-12T14:03:15+00:00",
                "expiry": "2026-05-22",
            }
        ]
    )

    actual = core._actual_forward_outcome_frame(
        tmp_path,
        tmp_path / "out",
        broker_backfilled_forward_outcomes=backfilled,
    )

    assert len(actual) == 1
    assert actual.iloc[0]["source"] == "schwab_closed_trades"
    assert actual.iloc[0]["realized_pnl"] == 155.0


def test_expectancy_evidence_includes_broker_matched_outcomes_as_diagnostic_only(tmp_path: Path) -> None:
    broker_matched = pd.DataFrame(
        [
            {"ticker": "META", "realized_pnl": -275.0, "match_sources": "codexuw_execute_outcome_ledger"},
            {"ticker": "CRM", "realized_pnl": -140.0, "match_sources": "codexuw_execute_outcome_ledger"},
            {"ticker": "HOOD", "realized_pnl": -34.0, "match_sources": "codexuw_execute_outcome_ledger"},
        ],
        columns=core.BROKER_MATCHED_OUTCOME_COLUMNS,
    )

    evidence = core.build_expectancy_evidence(
        tmp_path,
        pd.DataFrame([{"ticker": "META"}]),
        pd.DataFrame(),
        broker_matched_outcomes=broker_matched,
    )

    matched = evidence[evidence["evidence_type"].eq("broker_matched_recommendation_outcomes")].iloc[0]
    options_agent = evidence[evidence["evidence_type"].eq("broker_matched_options_agent_outcomes")].iloc[0]
    summary = evidence[evidence["source"].eq("expectancy_summary")].iloc[0]

    assert matched["status"] == "BLOCK"
    assert matched["sample_size"] == 3
    assert matched["avg_pnl"] == -149.67
    assert matched["matched_current_tickers"] == "META"
    assert "Diagnostic only" in matched["note"]
    assert options_agent["status"] == "BLOCK"
    assert options_agent["sample_size"] == 0
    assert "Options-Agent green ready history" in options_agent["note"]
    assert summary["sample_size"] == 0


def test_route_opportunity_gap_requires_bucket_calibration_before_execution_gap_status(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"CSP{idx}",
            "realized_pnl": 100.0,
            "strategy": "Short Put",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    wheel_dir = tmp_path / "out" / "fresh_wheel_replay_2026_full_ytd"
    wheel_dir.mkdir(parents=True)
    wheel_rows = [
        {
            "signal_date": "2026-04-01",
            "ticker": f"CSP{idx}",
            "action": "OPEN_CSP",
            "entry_credit": 2.00,
            "dte": 38,
            "exit_date": "2026-05-01",
            "pnl_per_contract": 125.0,
            "outcome_status": "scored",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(wheel_rows).to_csv(
        wheel_dir / "fresh-wheel-replay-outcomes-2026-01-02_2026-05-01.csv",
        index=False,
    )
    decision = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "Short Put",
                "trade_plan": "SELL 1 WMT 2026-07-17 95 Put @ 2.00 CREDIT",
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "short_put",
                "strategy_family": "short_put",
                "entry_type": "CREDIT",
                "direction_bucket": "bullish",
                "regime": "mixed",
                "dte_bucket": "dte_31_60",
                "iv_rank_bucket": "iv_unknown",
                "economics_bucket": "credit_rich",
                "liquidity_bucket": "liquidity_deep",
                "status": "WARN",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_route",
                "actual_support_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "actual_support_sample_gap": 0,
                "replay_bucket_status": "PASS",
                "replay_bucket_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "replay_bucket_sample_gap": 0,
                "diagnostic_replay_status": "PASS",
                "diagnostic_replay_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "current_ticket_count": 1,
            }
        ],
        columns=core.PROFITABILITY_CALIBRATION_COLUMNS,
    )

    gap = core.build_route_opportunity_gap(
        tmp_path,
        decision,
        pd.DataFrame(),
        calibration,
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("short_put")].iloc[0]
    assert row["actual_status"] == "PASS"
    assert row["replay_status"] == "PASS"
    assert row["current_ticket_count"] == 1
    assert row["calibration_pass_rows"] == 0
    assert row["calibration_warn_rows"] == 1
    assert row["route_status"] == "current_rows_need_bucket_calibration"
    assert row["development_gap"] == "current_rows_need_route_bucket_calibration"
    assert row["best_current_bucket_key"] == "short_put|CREDIT|bullish|mixed|dte_31_60|credit_rich|liquidity_deep"
    assert row["best_current_bucket_gap"] == "actual_bucket_precision_gap"
    assert "Actual support is only actual_route" in row["next_bucket_evidence_needed"]
    assert summary["bucket_calibration_routes"] == ["short_put"]
    assert summary["current_route_execution_gap_routes"] == []
    assert "bucket_calibration_needed=short_put" in core._route_opportunity_gap_detail(summary)


def test_route_opportunity_gap_uses_leakage_safe_pattern_validation_replay_for_long_call(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"LC{idx}",
            "realized_pnl": 100.0,
            "strategy": "Long Call",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE - 1)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    validation_dir = tmp_path / "out" / "options_pattern_pipeline_v1" / "2026-06-09"
    validation_dir.mkdir(parents=True)
    validation_rows = [
        {
            "sample": "VALIDATION",
            "status": "SCORED",
            "blocked": False,
            "strategy_type": "Long Call Debit",
            "net_r": 0.40,
            "signal_date": "2026-05-01",
            "target_date": "2026-05-15",
            "managed_exit_date": "",
            "lead_option_symbol": "VAL260620C00100000",
            "ticker": f"VAL{idx}",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    validation_rows.extend(
        [
            {
                "sample": "TRAIN",
                "status": "SCORED",
                "blocked": False,
                "strategy_type": "Long Call Debit",
                "net_r": -5.0,
                "signal_date": "2026-05-01",
                "target_date": "2026-05-15",
                "managed_exit_date": "",
                "lead_option_symbol": "TRAIN260620C00100000",
                "ticker": "TRAIN",
            },
            {
                "sample": "VALIDATION",
                "status": "SCORED",
                "blocked": False,
                "strategy_type": "Long Call Debit",
                "net_r": -5.0,
                "signal_date": "2026-05-01",
                "target_date": "2026-06-20",
                "managed_exit_date": "",
                "lead_option_symbol": "FUTURE260620C00100000",
                "ticker": "FUTURE",
            },
            {
                "sample": "VALIDATION",
                "status": "SCORED",
                "blocked": True,
                "strategy_type": "Long Call Debit",
                "net_r": -5.0,
                "signal_date": "2026-05-01",
                "target_date": "2026-05-15",
                "managed_exit_date": "",
                "lead_option_symbol": "BLOCK260620C00100000",
                "ticker": "BLOCKED",
            },
        ]
    )
    pd.DataFrame(validation_rows).to_csv(validation_dir / "validation_details.csv", index=False)

    gap = core.build_route_opportunity_gap(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )

    row = gap[gap["strategy_route"].eq("long_call")].iloc[0]
    assert row["route_status"] == "near_ready_more_actual_sample_needed"
    assert row["actual_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE - 1
    assert row["replay_status"] == "PASS"
    assert row["replay_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["replay_avg_pnl"] == 0.4
    assert "validation_details.csv" in row["source_path"]
    replay, _, _ = core._pattern_validation_replay_frame(tmp_path / "out", as_of=dt.date(2026, 6, 9))
    long_call = replay[replay["strategy_route"].eq("long_call")]
    assert set(long_call["dte_bucket"]) == {"dte_31_60"}
    assert set(long_call["economics_bucket"]) == {"debit_unknown"}


def test_pattern_validation_replay_concatenates_all_leakage_safe_sources(tmp_path: Path) -> None:
    old_dir = tmp_path / "out" / "options_pattern_pipeline_v1" / "2026-05-20"
    new_dir = tmp_path / "out" / "options_pattern_pipeline_v1" / "2026-06-09"
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    old_rows = [
        {
            "sample": "VALIDATION",
            "status": "SCORED",
            "blocked": False,
            "strategy_type": "Long Call Debit",
            "net_r": 0.30,
            "signal_date": "2026-05-01",
            "target_date": "2026-05-20",
            "managed_exit_date": "",
            "lead_option_symbol": f"OLD{idx}260620C00100000",
            "ticker": f"OLD{idx}",
        }
        for idx in range(3)
    ]
    new_rows = [
        {
            "sample": "VALIDATION",
            "status": "SCORED",
            "blocked": False,
            "strategy_type": "Bull Put Credit Spread",
            "net_r": 0.10,
            "signal_date": "2026-06-01",
            "target_date": "2026-06-08",
            "managed_exit_date": "",
            "lead_option_symbol": f"NEW{idx}260620P00100000",
            "entry_credit": 1.0,
            "ticker": f"NEW{idx}",
        }
        for idx in range(4)
    ]
    pd.DataFrame(old_rows).to_csv(old_dir / "validation_details.csv", index=False)
    pd.DataFrame(new_rows).to_csv(new_dir / "validation_details.csv", index=False)

    replay, source_path, error = core._pattern_validation_replay_frame(
        tmp_path / "out",
        as_of=dt.date(2026, 6, 9),
    )

    assert error == ""
    assert "2026-05-20" in source_path
    assert "2026-06-09" in source_path
    assert len(replay) == 7
    assert set(replay["strategy_route"]) == {"long_call", "bull_put_credit"}


def test_pattern_validation_replay_counts_one_fixed_horizon_per_contract(tmp_path: Path) -> None:
    validation_dir = tmp_path / "out" / "options_pattern_pipeline_v1" / "2026-06-09"
    validation_dir.mkdir(parents=True)
    rows = [
        {
            "sample": "VALIDATION",
            "status": "SCORED",
            "blocked": False,
            "strategy_type": "Long Call Debit",
            "net_r": pnl,
            "signal_date": "2026-05-01",
            "target_date": target,
            "managed_exit_date": target,
            "horizon": horizon,
            "lead_option_symbol": "AAPL260620C00200000",
            "ticker": "AAPL",
        }
        for horizon, target, pnl in [
            ("1d", "2026-05-02", -0.50),
            ("3d", "2026-05-06", -0.25),
            ("5d", "2026-05-08", 0.40),
            ("10d", "2026-05-15", 0.90),
        ]
    ]
    pd.DataFrame(rows).to_csv(validation_dir / "validation_details.csv", index=False)

    replay, _, error = core._pattern_validation_replay_frame(
        tmp_path / "out",
        as_of=dt.date(2026, 6, 9),
    )

    assert error == ""
    assert len(replay) == 1
    assert replay["horizon"].tolist() == ["5d"]
    assert replay["pnl_1x"].tolist() == [0.40]


def test_pattern_validation_replay_source_selection_caps_large_history_without_undated_leakage(tmp_path: Path) -> None:
    root = tmp_path / "out" / "options_pattern_pipeline_v1"
    names = [
        "2026-06-09_goal_uncapped_current_v1",
        "2026-06-08_goal_uncapped_current_v1",
        "2026-06-09",
        "2026-06-08",
        "2026-05-28_goal_evidence_v1",
        "2026-05-27_goal_evidence_v1",
        "latest_goal_acceptance",
        "2026-06-12_goal_uncapped_current_v1",
    ]
    for name in names:
        run_dir = root / name
        run_dir.mkdir(parents=True)
        (run_dir / "validation_details.csv").write_text(
            "sample,status,blocked,strategy_type,net_r\n",
            encoding="utf-8",
        )

    selected = core._pattern_validation_replay_source_paths(
        tmp_path / "out",
        as_of=dt.date(2026, 6, 9),
    )
    selected_names = [path.parent.name for path in selected]

    assert len(selected_names) <= core.MAX_PATTERN_VALIDATION_REPLAY_SOURCE_FILES
    assert "2026-06-09_goal_uncapped_current_v1" in selected_names
    assert "2026-06-08_goal_uncapped_current_v1" not in selected_names
    assert "2026-06-09" in selected_names
    assert "2026-05-28_goal_evidence_v1" in selected_names
    assert "2026-05-27_goal_evidence_v1" not in selected_names
    assert "latest_goal_acceptance" not in selected_names
    assert "2026-06-12_goal_uncapped_current_v1" not in selected_names


def test_replay_calibration_matches_unknown_liquidity_when_exact_source_lacks_liquidity() -> None:
    replay = pd.DataFrame(
        [
            {
                "strategy_route": "bull_call_debit",
                "entry_type": "DEBIT",
                "direction_bucket": "bullish",
                "regime": "risk_on",
                "dte_bucket": "dte_15_30",
                "economics_bucket": "debit_reward_risk_high",
                "liquidity_bucket": "liquidity_unknown",
                "pnl_1x": 1.25,
            }
            for _ in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
        ]
    )
    key = {
        "strategy_route": "bull_call_debit",
        "entry_type": "DEBIT",
        "direction_bucket": "bullish",
        "regime": "risk_on",
        "dte_bucket": "dte_15_30",
        "economics_bucket": "debit_reward_risk_high",
        "liquidity_bucket": "liquidity_deep",
    }

    matched = core._replay_calibration_slice(replay, key)
    metrics = core._calibration_metrics_row(matched["pnl_1x"], status_func=core._expectancy_status)

    assert len(matched) == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert set(matched["liquidity_bucket"]) == {"liquidity_unknown"}
    assert metrics["status"] == "PASS"


def test_pattern_validation_replay_buckets_credit_spread_width_from_legs_json(tmp_path: Path) -> None:
    validation_dir = tmp_path / "out" / "options_pattern_pipeline_v1" / "2026-06-09"
    validation_dir.mkdir(parents=True)
    legs_json = json.dumps(
        [
            {"action": "SELL", "option_symbol": "SPY260620P00540000", "strike": 540.0},
            {"action": "BUY", "option_symbol": "SPY260620P00535000", "strike": 535.0},
        ]
    )
    rows = [
        {
            "sample": "VALIDATION",
            "status": "SCORED",
            "blocked": False,
            "strategy_type": "Bull Put Credit Spread",
            "net_r": 0.25,
            "signal_date": "2026-05-21",
            "target_date": "2026-05-28",
            "lead_option_symbol": "SELL SPY260620P00540000 / BUY SPY260620P00535000",
            "entry_credit": 2.00,
            "legs_json": legs_json,
            "ticker": f"SPY{idx}",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(rows).to_csv(validation_dir / "validation_details.csv", index=False)

    replay, _, _ = core._pattern_validation_replay_frame(tmp_path / "out", as_of=dt.date(2026, 6, 9))

    bull_put = replay[replay["strategy_route"].eq("bull_put_credit")]
    assert len(bull_put) == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert set(bull_put["dte_bucket"]) == {"dte_15_30"}
    assert set(bull_put["economics_bucket"]) == {"credit_width_high"}


def test_route_opportunity_gap_blocks_negative_actual_vertical_route_despite_replay(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"BPC{idx}",
            "realized_pnl": -80.0,
            "strategy": "Bull Put Credit Spread",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": f"BPC{idx}",
            "strategy": "Bull Put Credit Spread",
            "strategy_kind": "Credit",
            "entry_side": "credit",
            "dte": 28,
            "iv_rank": 45,
            "entry_credit_pct_width": 0.32,
            "source_contract_oi": 1200,
            "pnl_1x": 70.0,
            "exact_evaluated": True,
            "decision_pass": True,
            "exit_day": "2026-05-15",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)
    decision = pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "structure": "Bull Put Credit Spread",
                "trade_plan": "SELL 1 SPY 2026-06-30 540 Put / BUY 1 SPY 2026-06-30 535 Put @ 1.60 CREDIT",
            }
        ]
    )

    gap = core.build_route_opportunity_gap(
        tmp_path,
        decision,
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("bull_put_credit")].iloc[0]
    assert row["route_status"] == "actual_outcomes_negative_or_weak"
    assert row["actual_status"] == "BLOCK"
    assert row["actual_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE
    assert row["replay_status"] == "PASS"
    assert row["current_ticket_count"] == 1
    assert row["suggested_action"] == "Do not promote this route; require new positive closed-trade evidence before green eligibility."
    assert summary["negative_or_weak_routes"] == ["bull_put_credit"]


def test_expectancy_status_allows_positive_skew_with_lower_win_rate() -> None:
    assert core._expectancy_status(
        core.MIN_EXPECTANCY_SAMPLE_SIZE + 12,
        0.4524,
        23.81,
        1.322,
    ) == "PASS"
    assert core._expectancy_status(
        core.MIN_EXPECTANCY_SAMPLE_SIZE + 12,
        0.35,
        23.81,
        1.322,
    ) == "BLOCK"
    assert core._expectancy_status(
        core.MIN_EXPECTANCY_SAMPLE_SIZE + 12,
        0.4524,
        23.81,
        1.01,
    ) == "BLOCK"


def test_route_opportunity_gap_treats_positive_skew_debit_route_as_supported(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"BCDWIN{idx}",
            "realized_pnl": 100.0,
            "strategy": "Bull Call Debit Spread",
        }
        for idx in range(19)
    ]
    closed_rows.extend(
        {
            "ticker": f"BCDLOSS{idx}",
            "realized_pnl": -62.5,
            "strategy": "Bull Call Debit Spread",
        }
        for idx in range(23)
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": f"BCD{idx}",
            "strategy": "Bull Call Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 28,
            "iv_rank": 45,
            "reward_risk": 2.2,
            "source_contract_oi": 1200,
            "pnl_1x": 70.0,
            "exact_evaluated": True,
            "decision_pass": True,
            "exit_day": "2026-05-15",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    gap = core.build_route_opportunity_gap(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("bull_call_debit")].iloc[0]
    assert row["actual_status"] == "PASS"
    assert row["actual_sample_size"] == 42
    assert row["actual_win_rate"] == 0.4524
    assert row["actual_avg_pnl"] == 11.01
    assert row["actual_profit_factor"] == 1.322
    assert row["replay_status"] == "PASS"
    assert row["route_status"] == "evidence_ready_no_current_ticket"
    assert "bull_call_debit" not in summary["negative_or_weak_routes"]


def test_route_opportunity_gap_does_not_call_negative_warn_actual_near_ready(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        json.dumps(
            {
                "ticker": "UPS",
                "realized_pnl": -40.0,
                "strategy": "Bear Put Debit Spread",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": f"BPD{idx}",
            "strategy": "Bear Put Debit Spread",
            "strategy_kind": "Debit",
            "entry_side": "debit",
            "dte": 28,
            "iv_rank": 45,
            "reward_risk": 2.2,
            "source_contract_oi": 1200,
            "pnl_1x": 70.0,
            "exact_evaluated": True,
            "decision_pass": True,
            "exit_day": "2026-05-15",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    gap = core.build_route_opportunity_gap(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("bear_put_debit")].iloc[0]
    assert row["actual_status"] == "WARN"
    assert row["actual_sample_size"] == 1
    assert row["actual_avg_pnl"] == -40.0
    assert row["replay_status"] == "PASS"
    assert row["route_status"] == "actual_closed_trade_support_needed"
    assert row["strategy_route"] not in summary["near_ready_routes"]


def test_route_opportunity_gap_requires_near_ready_profit_factor_threshold(tmp_path: Path) -> None:
    closed_dir = tmp_path / "out" / "schwab_pull_state"
    closed_dir.mkdir(parents=True)
    closed_rows = [
        {
            "ticker": f"CSP{idx}",
            "realized_pnl": 10.0,
            "strategy": "Cash Secured Put",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE - 2)
    ]
    closed_rows.append(
        {
            "ticker": "CSPLOSS",
            "realized_pnl": -250.0,
            "strategy": "Cash Secured Put",
        }
    )
    (closed_dir / "closed_trades_acct_3326.jsonl").write_text(
        "\n".join(json.dumps(row) for row in closed_rows) + "\n",
        encoding="utf-8",
    )
    replay_dir = tmp_path / "out" / "codexuw_v2_backtest_fixture"
    replay_dir.mkdir(parents=True)
    replay_rows = [
        {
            "ticker": f"CSP{idx}",
            "strategy": "Cash Secured Put",
            "strategy_kind": "Credit",
            "entry_side": "credit",
            "dte": 28,
            "iv_rank": 45,
            "entry_credit_pct_width": 0.32,
            "source_contract_oi": 1200,
            "pnl_1x": 70.0,
            "exact_evaluated": True,
            "decision_pass": True,
            "exit_day": "2026-05-15",
        }
        for idx in range(core.MIN_EXPECTANCY_SAMPLE_SIZE)
    ]
    pd.DataFrame(replay_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    gap = core.build_route_opportunity_gap(
        tmp_path,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(columns=core.PROFITABILITY_CALIBRATION_COLUMNS),
        as_of_date="2026-06-09",
    )
    summary = core.summarize_route_opportunity_gap(gap)

    row = gap[gap["strategy_route"].eq("short_put")].iloc[0]
    assert row["actual_status"] == "WARN"
    assert row["actual_sample_size"] == core.MIN_EXPECTANCY_SAMPLE_SIZE - 1
    assert row["actual_profit_factor"] < core.MIN_EXPECTANCY_PROFIT_FACTOR
    assert row["replay_status"] == "PASS"
    assert row["route_status"] == "actual_closed_trade_support_needed"
    assert "short_put" not in summary["near_ready_routes"]


def test_profitability_calibration_blocks_ready_looking_green_row() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "NOCAL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 NOCAL 2026-06-19 100 Call / BUY 1 NOCAL 2026-06-19 105 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 NOCAL 2026-06-19 100 Call / BUY 1 NOCAL 2026-06-19 105 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 5,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_expectancy_status": "PASS",
                "actual_forward_expectancy_sample_size": 10,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 10,
                "profitability_calibration_status": "BLOCK",
                "profitability_calibration_sample_size": 0,
                "profitability_calibration_actual_status": "BLOCK",
                "profitability_calibration_replay_status": "BLOCK",
                "profitability_calibration_note": "bucket missing",
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=10,
        external_review_count=10,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert decision["ready_to_enter"].tolist() == [False]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in blockers
    assert decision["target_order_status"].tolist() == ["review_only_profitability_calibration"]
    green, target = core.split_trade_ticket_surfaces(tickets)
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["review_only_profitability_calibration"]
    assert green.empty
    assert target.empty


def test_profitability_calibration_blocks_yellow_target_row_until_bucket_proven() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "CALWAIT",
                "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 CALWAIT 2026-06-19 100 Call / BUY 1 CALWAIT 2026-06-19 105 Call @ 1.50 CREDIT",
                "trade_plan": "SELL 1 CALWAIT 2026-06-19 100 Call / BUY 1 CALWAIT 2026-06-19 105 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 1,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "credit_width_ratio": 0.3,
                "trade_quality_status": "reviewable",
                "live_validation_status": "TARGET_QUOTE_REFRESH",
                "status_reason": "dated UW target from EOD; fresh Schwab chain target quote required",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "profitability_calibration_status": "WARN",
                "profitability_calibration_sample_size": 12,
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_note": "needs more route-precise actual evidence",
            }
        ]
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    tickets = core.build_trade_tickets(decision)

    decision_blockers = str(decision["execution_blockers"].iloc[0])
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in decision_blockers
    assert decision["target_order_status"].tolist() == ["review_only_live_validation"]
    assert tickets.empty


def test_negative_route_family_evidence_keeps_row_off_yellow_target_surface() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "WEAKVERT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 WEAKVERT 2026-07-17 100 Put / BUY 1 WEAKVERT 2026-07-17 95 Put @ 2.50 CREDIT",
                "trade_plan": "SELL 1 WEAKVERT 2026-07-17 100 Put / BUY 1 WEAKVERT 2026-07-17 95 Put @ 2.50 CREDIT",
                "entry_limit": 2.5,
                "suggested_contracts": 4,
                "max_profit": 250.0,
                "max_loss": 250.0,
                "credit_width_ratio": 0.50,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "PASS",
                "route_action": "construct_research_only_negative_family_evidence",
                "route_evidence_status": "BLOCK",
                "route_evidence_sample_size": 76,
                "route_evidence_avg_pnl": -28.42,
                "route_evidence_profit_factor": 0.732,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert core.NEGATIVE_ROUTE_FAMILY_EVIDENCE_BLOCKER in blockers
    assert decision["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert tickets.empty


def test_weak_positive_route_family_evidence_is_not_labeled_negative() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "WEAKCALL",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 WEAKCALL 2026-07-17 100 Call / SELL 1 WEAKCALL 2026-07-17 105 Call @ 1.50 DEBIT",
                "trade_plan": "BUY 1 WEAKCALL 2026-07-17 100 Call / SELL 1 WEAKCALL 2026-07-17 105 Call @ 1.50 DEBIT",
                "entry_limit": 1.5,
                "suggested_contracts": 4,
                "max_profit": 350.0,
                "max_loss": 150.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "WARN",
                "actual_forward_strategy_expectancy_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_replay_status": "BLOCK",
                "route_action": "construct_research_only_weak_family_evidence",
                "route_evidence_status": "BLOCK",
                "route_evidence_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "route_evidence_avg_pnl": 20.61,
                "route_evidence_profit_factor": 1.272,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert core.NEGATIVE_ROUTE_FAMILY_EVIDENCE_BLOCKER not in blockers
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in blockers


def test_positive_actual_support_keeps_non_csp_yellow_despite_negative_route_family() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AMZN",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "quality_status": "qualified",
                "structure": "bull call debit spread",
                "full_ticket": "BUY 1 AMZN 2026-07-17 252.5 Call / SELL 1 AMZN 2026-07-17 255 Call @ 0.72 DEBIT",
                "trade_plan": "BUY 1 AMZN 2026-07-17 252.5 Call / SELL 1 AMZN 2026-07-17 255 Call @ 0.72 DEBIT",
                "entry_limit": 0.72,
                "suggested_contracts": 5,
                "max_profit": 178.0,
                "max_loss": 72.0,
                "trade_quality_status": "reviewable",
                "trade_quality_confidence_rating": "HIGH",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 3,
                "actual_forward_strategy_expectancy_avg_pnl": 24.56,
                "actual_forward_strategy_expectancy_profit_factor": 1.364,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route_economics_bucket",
                "profitability_calibration_sample_size": 16,
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_actual_sample_size": 16,
                "profitability_calibration_actual_avg_pnl": 24.56,
                "profitability_calibration_actual_profit_factor": 1.364,
                "profitability_calibration_replay_status": "BLOCK",
                "profitability_calibration_replay_sample_size": 0,
                "route_action": "construct_research_only_negative_family_evidence",
                "route_evidence_status": "BLOCK",
                "route_evidence_sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "route_evidence_avg_pnl": -28.42,
                "route_evidence_profit_factor": 0.732,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert core.NEGATIVE_ROUTE_FAMILY_EVIDENCE_BLOCKER in blockers
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in blockers
    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["ticker"].tolist() == ["AMZN"]
    assert tickets["structure"].tolist() == ["bull call debit spread"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["order_readiness"].tolist() == ["target_order_after_profitability_calibration"]


def test_broad_route_actual_support_without_strategy_support_stays_review_only() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "BAC",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "structure": "long call",
                "full_ticket": "BUY 1 BAC 2026-07-17 60 Call @ 0.64 DEBIT",
                "trade_plan": "BUY 1 BAC 2026-07-17 60 Call @ 0.64 DEBIT",
                "entry_limit": 0.64,
                "suggested_contracts": 5,
                "max_profit": 51.0,
                "max_loss": 64.0,
                "trade_quality_status": "reviewable",
                "trade_quality_confidence_rating": "LOW",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": 0,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route",
                "profitability_calibration_sample_size": 28,
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_actual_sample_size": 28,
                "profitability_calibration_actual_avg_pnl": 97.93,
                "profitability_calibration_actual_profit_factor": 2.531,
                "profitability_calibration_replay_status": "BLOCK",
                "profitability_calibration_replay_sample_size": 0,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in str(decision["execution_blockers"].iloc[0])
    assert decision["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    green, target = core.split_trade_ticket_surfaces(tickets)
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    assert green.empty
    assert target.empty


def test_short_put_route_stays_off_yellow_target_surface_with_negative_calibration() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "BADBUCKET",
                "recommendation_status": RecommendationStatus.WAIT_FOR_PRICE.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 BADBUCKET 2026-07-17 100 Put @ 1.50 CREDIT",
                "trade_plan": "SELL 1 BADBUCKET 2026-07-17 100 Put @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 1,
                "max_profit": 150.0,
                "max_loss": 9850.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "TARGET_QUOTE_REFRESH",
                "status_reason": "dated UW target from EOD; fresh Schwab chain target quote required",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_actual_sample_size": 3,
                "profitability_calibration_actual_avg_pnl": -111.33,
                "profitability_calibration_actual_profit_factor": 0.032,
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_note": "actual bucket is under-sampled and losing",
            }
        ]
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"})
    tickets = core.build_trade_tickets(decision)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert decision["target_order_status"].tolist() == ["review_only_profitability_calibration"]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in blockers
    assert core.PROFITABILITY_CALIBRATION_ACTUAL_NEGATIVE_BLOCKER in blockers
    assert tickets.empty


def test_short_put_route_stays_visible_with_positive_actual_support_despite_low_profit() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "SMALLWARN",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 SMALLWARN 2026-07-17 100 Put @ 1.00 CREDIT",
                "trade_plan": "SELL 1 SMALLWARN 2026-07-17 100 Put @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "suggested_contracts": 1,
                "max_profit": 100.0,
                "max_loss": 9900.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 34,
                "profitability_calibration_actual_avg_pnl": 92.09,
                "profitability_calibration_actual_profit_factor": 1.823,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_note": "route-level support only; exact bucket still needs proof",
            }
        ]
    )

    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )
    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in blockers
    assert core.POSITION_PROFIT_MATERIALITY_BLOCKER in blockers
    assert tickets["ticker"].tolist() == ["SMALLWARN"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["order_readiness"].tolist() == ["target_order_profit_floor"]


def test_replay_blocked_calibration_stays_off_yellow_target_surface() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "REPLAYMISS",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 REPLAYMISS 2026-07-17 100 Put @ 10.00 CREDIT",
                "trade_plan": "SELL 1 REPLAYMISS 2026-07-17 100 Put @ 10.00 CREDIT",
                "entry_limit": 10.0,
                "suggested_contracts": 1,
                "max_profit": 1000.0,
                "max_loss": 9000.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 34,
                "profitability_calibration_actual_avg_pnl": 92.09,
                "profitability_calibration_actual_profit_factor": 1.823,
                "profitability_calibration_replay_status": "BLOCK",
                "profitability_calibration_replay_sample_size": 0,
                "profitability_calibration_note": "exact replay bucket missing",
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 1_000_000, "cash": 1_000_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["target_order_status"].tolist() == ["review_only_profitability_calibration"]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in str(decision["execution_blockers"].iloc[0])
    green, target = core.split_trade_ticket_surfaces(tickets)
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["review_only_profitability_calibration"]
    assert green.empty
    assert target.empty


def test_positive_actual_long_call_stays_on_yellow_target_surface_while_replay_bucket_pending() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "structure": "long call",
                "full_ticket": "BUY 1 GOOG 2026-07-17 390 Call @ 5.90 DEBIT",
                "trade_plan": "BUY 1 GOOG 2026-07-17 390 Call @ 5.90 DEBIT",
                "entry_limit": 5.90,
                "suggested_contracts": 2,
                "max_profit": 472.0,
                "max_loss": 590.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 4,
                "actual_forward_strategy_expectancy_avg_pnl": 115.29,
                "actual_forward_strategy_expectancy_profit_factor": 1.637,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 4,
                "profitability_calibration_actual_avg_pnl": 115.29,
                "profitability_calibration_actual_profit_factor": 1.637,
                "profitability_calibration_replay_status": "BLOCK",
                "profitability_calibration_replay_sample_size": 0,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert decision["ready_to_enter"].tolist() == [False]
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in blockers
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["ticker"].tolist() == ["GOOG"]
    assert tickets["structure"].tolist() == ["long call"]
    assert core._ticket_structure(tickets.iloc[0]) == "Long call"
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["order_readiness"].tolist() == ["target_order_after_profitability_calibration"]


def test_weak_actual_long_call_stays_review_only_until_positive_support_exists() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "META",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "structure": "long call",
                "full_ticket": "BUY 1 META 2026-07-17 600 Call @ 2.80 DEBIT",
                "trade_plan": "BUY 1 META 2026-07-17 600 Call @ 2.80 DEBIT",
                "entry_limit": 2.80,
                "suggested_contracts": 4,
                "max_profit": 224.0,
                "max_loss": 280.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "WARN",
                "actual_forward_strategy_expectancy_sample_size": 1,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "WARN",
                "profitability_calibration_actual_sample_size": 1,
                "profitability_calibration_actual_avg_pnl": 20.0,
                "profitability_calibration_actual_profit_factor": 1.2,
                "profitability_calibration_replay_status": "BLOCK",
                "profitability_calibration_replay_sample_size": 0,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert core.POSITIVE_STRATEGY_EXPECTANCY_BLOCKER in blockers
    assert decision["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    green, target = core.split_trade_ticket_surfaces(tickets)
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["review_only_expectancy_evidence"]
    assert green.empty
    assert target.empty


def test_uncalibrated_low_profit_row_stays_on_target_surface_with_missing_ticker_reviews() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "SMALLCOVER",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "SELL 1 SMALLCOVER 2026-07-17 100 Call / BUY 1 SMALLCOVER 2026-07-17 105 Call @ 1.00 CREDIT",
                "trade_plan": "SELL 1 SMALLCOVER 2026-07-17 100 Call / BUY 1 SMALLCOVER 2026-07-17 105 Call @ 1.00 CREDIT",
                "entry_limit": 1.0,
                "suggested_contracts": 1,
                "max_profit": 100.0,
                "max_loss": 400.0,
                "credit_width_ratio": 0.2,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "external_agent_distinct_review_count": 0,
                "underlying_quality_tier": "core",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 34,
                "profitability_calibration_actual_avg_pnl": 92.09,
                "profitability_calibration_actual_profit_factor": 1.823,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_note": "route-level support only; exact bucket still needs proof",
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    blockers = str(decision["execution_blockers"].iloc[0])
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert "ticker_agentic_review_coverage_below_threshold" in blockers
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in blockers
    assert core.POSITION_PROFIT_MATERIALITY_BLOCKER in blockers
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["order_readiness"].tolist() == ["target_order_profit_floor"]


def test_report_labels_no_trade_section_as_preview_of_full_csv() -> None:
    no_trade = pd.DataFrame(
        [
            {"ticker": f"MISS{idx}", "bias": "bearish", "score": 70 - idx, "reason": f"candidate {idx}"}
            for idx in range(22)
        ]
    )

    report = core.render_report("2026-05-22", pd.DataFrame(), no_trade, {"row_counts": {}, "warnings": []})

    assert "Showing first 20 of 22 rows; full audit is in `no_trade_audit.csv`." in report
    assert "2 additional no-trade rows in no_trade_audit.csv" in report


def test_report_uses_position_scaled_profit_loss_for_target_order_tables() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "GOOGL",
                "ready_to_enter": False,
                "execution_status": "waiting_for_price",
                "execution_gate_status": "pass",
                "execution_blockers": "send_now_credit_width_below_30pct",
                "target_order_status": "target_order_candidate",
                "suggested_contracts": 4,
                "trade_plan": "SELL 1 GOOGL 2026-06-05 392.5 Call / BUY 1 GOOGL 2026-06-05 395 Call @ 0.65 CREDIT",
                "entry_limit": 0.65,
                "target_exit": 0.23,
                "max_profit": 65.0,
                "max_loss": 185.0,
                "max_position_loss": 740.0,
                "live_validation_status": "PASS",
                "trade_quality_confidence_rating": "HIGH",
                "external_agent_distinct_review_count": 4,
                "execution_confidence_score": 88,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "underlying_quality_tier": "core",
                "status_reason": "fixture target row",
            }
        ]
    )

    report = core.render_report(
        "2026-05-22",
        final,
        pd.DataFrame(),
        {"row_counts": {}, "warnings": []},
    )

    assert "Max Profit" in report
    assert "Max Loss" in report
    assert "Target Orders - Target Credits/Debits" in report
    assert (
        "| GOOGL | 🟡 YELLOW target | Call credit spread | 2026-06-05 | "
            "SELL 1 GOOGL 2026-06-05 392.5 Call | BUY 1 GOOGL 2026-06-05 395 Call | "
            "4 | 0.65 CREDIT | 0.23 | 260.0 | 740.0 | "
            "edge HIGH; entry NOT_EXECUTION_READY / 0; order HIGH / 88 | "
            "contract metrics unavailable | credit/width too weak for send-now |"
        ) in report


def test_report_keeps_uncalibrated_watch_plans_out_of_target_order_table() -> None:
    base = {
        "ready_to_enter": False,
        "execution_status": "needs_confidence",
        "execution_gate_status": "blocked",
        "target_order_status": "target_order_candidate",
        "suggested_contracts": 1,
        "entry_limit": 1.0,
        "target_exit": 1.8,
        "max_profit": 100.0,
        "max_loss": 100.0,
        "live_validation_status": "PASS",
        "trade_quality_confidence_rating": "HIGH",
        "execution_confidence_rating": "NOT_EXECUTION_READY",
        "underlying_quality_tier": "core",
    }
    final = pd.DataFrame(
        [
            {
                **base,
                "ticker": "CALPASS",
                "trade_plan": "BUY 1 CALPASS 2026-08-21 100 Call @ 1.00 DEBIT",
                "profitability_calibration_status": "PASS",
            },
            {
                **base,
                "ticker": "CALWARN",
                "trade_plan": "BUY 1 CALWARN 2026-08-21 100 Call @ 1.00 DEBIT",
                "profitability_calibration_status": "WARN",
            },
        ]
    )

    report = core.render_report("2026-07-10", final, pd.DataFrame(), {"row_counts": {}, "warnings": []})
    target_section = report.split("## Target Orders - Target Credits/Debits", 1)[1].split(
        "## Watch Plans - Not Orders", 1
    )[0]

    assert "CALPASS" in target_section
    assert "CALWARN" not in target_section
    assert "1 additional plans lack passing profitability calibration" in report


def test_report_snapshot_counts_review_only_visible_tickets() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "BA",
                "ready_to_enter": False,
                "execution_status": "needs_confidence",
                "execution_blockers": "profitability_calibration_required_for_green; positive_strategy_expectancy_required_for_green",
                "target_order_status": "review_only_expectancy_evidence",
                "suggested_contracts": 3,
                "trade_plan": "BUY 1 BA 2026-07-17 225 Call @ 3.45 DEBIT",
                "entry_limit": 3.45,
                "max_profit": 276.0,
                "max_loss": 345.0,
                "live_validation_status": "PASS",
                "trade_quality_confidence_rating": "HIGH",
                "order_mechanics_confidence_score": 100,
                "order_mechanics_confidence_rating": "HIGH",
                "execution_confidence_score": 0,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "underlying_quality_tier": "core",
                "status_reason": "fixture review ticket",
            }
        ]
    )

    coverage = pd.DataFrame(
        [
            {
                "ticker": "BA",
                "bias": "bullish",
                "score": 70,
                "coverage_status": "REVIEW_TICKET",
                "status_color": "yellow",
                "reason": "fixture universe audit row",
                "next_step": "review only",
            }
        ]
    )
    report = core.render_report(
        "2026-06-11",
        final,
        pd.DataFrame(),
        {
            "row_counts": {
                "trade_tickets": 1,
                "green_trade_tickets": 0,
                "target_order_ticket_rows": 0,
            },
            "execution_readiness_summary": {"blocking_gates": ["ready_trade_tickets"]},
            "warnings": [],
        },
        coverage,
    )

    assert "- Executable status: NOT TRADE READY" in report
    assert "- Green send-now orders: 0" in report
    assert "- Yellow target orders: 0" in report
    assert "- Review-only candidates: 1" in report
    assert "- Research/backtest profitability evidence: 0.0/10 (not order-entry readiness)" in report
    assert "No yellow target orders." in report
    assert (
        "Next action: do not enter review rows; require at least 30 nonduplicated matching outcomes "
        "with PF >= 1.20 before promotion"
    ) in report
    assert "- Trade rows: 0 green send-now, 0 target-order candidates, 1 review-only visible tickets" in report
    assert report.index("## Focus Review Queue - Not Trades") < report.index("## Review Board - Not Orders")


def test_zero_quantity_contract_risk_reports_unit_theta_not_position_theta() -> None:
    summary = core._ticket_contract_risk_summary(
        {
            "suggested_contracts": 0,
            "live_net_theta_per_contract": 4.25,
        }
    )

    assert "theta $4.25/contract/day" in summary
    assert "theta $4.25/day" not in summary


def test_report_sanitizes_market_session_blocker_in_execution_quality() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "SPY",
                "ready_to_enter": False,
                "execution_status": "needs_market_session",
                "execution_blockers": "market_session_open_required",
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "trade_quality_confidence_rating": "HIGH",
                "target_order_status": "target_order_candidate",
                "trade_plan": "SELL 1 SPY 2026-06-05 600 Call / BUY 1 SPY 2026-06-05 605 Call @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "suggested_contracts": 1,
                "max_profit": 150.0,
                "max_loss": 350.0,
                "underlying_quality_tier": "core",
            }
        ]
    )

    report = core.render_report("2026-06-09", final, pd.DataFrame(), {"row_counts": {}, "warnings": []})

    assert "market_session_open_required" not in report
    assert "fresh quote refresh" in report
    assert "work_target_limit_after_market_open_recheck" not in report
    assert "target_order_after_market_open_and_live_recheck" not in report
    assert "Market Open Recheck Queue" not in report
    assert "Market-open recheck" not in report


def test_report_target_order_table_uses_trade_ticket_surface_filters() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "PEP",
                "ready_to_enter": False,
                "execution_status": "needs_review",
                "execution_gate_status": "pass",
                "execution_blockers": "fresh_live_schwab_required",
                "target_order_status": "target_order_candidate",
                "suggested_contracts": 5,
                "trade_plan": "BUY 1 PEP 2026-06-18 155 Call / SELL 1 PEP 2026-06-18 160 Call @ 0.88 DEBIT",
                "entry_limit": 0.88,
                "target_exit": 1.58,
                "max_profit": 412.0,
                "max_loss": 88.0,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 5,
                "execution_confidence_score": 73,
                "underlying_quality_tier": "core",
                "live_validation_status": "PASS",
            },
            {
                "recommendation_rank": 2,
                "ticker": "UNH",
                "ready_to_enter": False,
                "execution_status": "needs_review",
                "execution_gate_status": "pass",
                "execution_blockers": "ticker_agentic_review_coverage_below_threshold",
                "target_order_status": "target_order_candidate",
                "suggested_contracts": 4,
                "trade_plan": "BUY 1 UNH 2026-06-18 380 Put / SELL 1 UNH 2026-06-18 370 Put @ 3.38 DEBIT",
                "entry_limit": 3.38,
                "target_exit": 6.08,
                "max_profit": 662.0,
                "max_loss": 338.0,
                "trade_quality_confidence_rating": "MEDIUM",
                "external_agent_distinct_review_count": 2,
                "execution_confidence_score": 72,
                "underlying_quality_tier": "core",
                "live_validation_status": "PASS",
            },
        ]
    )

    report = core.render_report("2026-05-15", final, pd.DataFrame(), {"row_counts": {}, "warnings": []})

    assert (
        "| PEP | 🟡 YELLOW target | Call debit spread | 2026-06-18 | "
        "SELL 1 PEP 2026-06-18 160 Call | BUY 1 PEP 2026-06-18 155 Call | "
        "5 | 0.88 DEBIT | 1.58 | 2060.0 | 440.0 | "
        "edge MEDIUM; entry NOT_EXECUTION_READY / 0; order MEDIUM / 73 | "
        "contract metrics unavailable | fresh Schwab chain |"
    ) in report
    assert (
        "| UNH | 🟡 YELLOW target | Put debit spread | 2026-06-18 | "
        "SELL 1 UNH 2026-06-18 370 Put | BUY 1 UNH 2026-06-18 380 Put | "
        "4 | 3.38 DEBIT | 6.08 | 2648.0 | 1352.0 | "
        "edge MEDIUM; entry NOT_EXECUTION_READY / 0; order MEDIUM / 72 | "
        "contract metrics unavailable | agent review coverage |"
    ) in report


def test_report_coverage_audit_blanks_nan_rank_values() -> None:
    coverage = pd.DataFrame(
        [
            {
                "ticker": "NVDA",
                "underlying_quality_tier": "core",
                "raw_rank": 23.0,
                "candidate_rank": pd.NA,
                "bias": "neutral",
                "score": 72.7100,
                "coverage_status": "NO_DIRECTIONAL_EDGE",
                "status_color": "gray",
                "reason": "neutral flow bias",
                "next_step": "wait for directional flow",
            },
            {
                "ticker": "URA",
                "underlying_quality_tier": "excluded",
                "raw_rank": 3847.0,
                "candidate_rank": 2744.0,
                "bias": "bearish",
                "score": 17.13,
                "coverage_status": "CANDIDATE_NOT_STRUCTURED",
                "status_color": "yellow",
                "reason": "excluded underlying",
                "next_step": "run structure expansion or live-chain construction",
            }
        ]
    )

    report = core.render_report(
        "2026-05-22",
        pd.DataFrame(),
        pd.DataFrame(),
        {"row_counts": {}, "warnings": []},
        coverage,
    )

    assert "| NVDA | ⚪ GRAY no-edge | neutral | 72.71 | GRAY no-edge | neutral flow bias | wait for directional flow |" in report
    assert "| URA | 🔴 RED no-action |" in report
    assert "YELLOW candidate | URA" not in report
    assert "nan" not in report.lower()


def test_coverage_audit_marks_speculative_and_excluded_candidates_non_actionable() -> None:
    candidates = pd.DataFrame(
        [
            {
                "ticker": "URA",
                "bias": "bearish",
                "score": 17.13,
                "underlying_quality_tier": "excluded",
                "underlying_quality_reason": "non-core ETF; not in actionable ETF allowlist",
                "flow_reason": "excluded underlying",
            },
            {
                "ticker": "OKLO",
                "bias": "neutral",
                "score": 31.53,
                "underlying_quality_tier": "speculative",
                "underlying_quality_reason": "liquidity below actionable thresholds",
                "flow_reason": "speculative underlying",
            },
            {
                "ticker": "DVN",
                "bias": "bullish",
                "score": 71.63,
                "underlying_quality_tier": "liquid",
                "underlying_quality_reason": "liquid non-core underlying",
                "flow_reason": "liquid candidate",
            },
        ]
    )

    coverage = core.build_coverage_audit(
        pd.DataFrame(),
        candidates,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        watchlist=["URA", "OKLO", "DVN"],
    )

    by_ticker = coverage.set_index("ticker")
    assert by_ticker.loc["URA", "coverage_status"] == "NON_ACTIONABLE_UNDERLYING"
    assert by_ticker.loc["URA", "status_color"] == "red"
    assert by_ticker.loc["OKLO", "coverage_status"] == "NON_ACTIONABLE_UNDERLYING"
    assert by_ticker.loc["OKLO", "status_color"] == "red"
    assert by_ticker.loc["DVN", "coverage_status"] == "NON_ACTIONABLE_UNDERLYING"
    assert by_ticker.loc["DVN", "status_color"] == "red"

    report = core.render_report(
        "2026-05-22",
        pd.DataFrame(),
        pd.DataFrame(),
        {"row_counts": {}, "warnings": []},
        coverage,
    )

    assert "| URA | 🔴 RED no-action |" in report
    assert "| OKLO | 🔴 RED no-action |" in report
    assert "YELLOW candidate | URA" not in report
    assert "YELLOW candidate | OKLO" not in report
    assert "| DVN | 🔴 RED no-action |" in report


def test_action_surface_underlying_quality_proof_blocks_audit_only_names_on_action_surfaces() -> None:
    packet = audit.build_action_surface_underlying_quality_proof_packet(
        tickets=pd.DataFrame(
            [
                {
                    "ticker": "URA",
                    "underlying_quality_tier": "excluded",
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                }
            ]
        ),
        market_open_recheck_queue=pd.DataFrame(
            [
                {
                    "ticker": "OKLO",
                    "underlying_quality_tier": "speculative",
                }
            ]
        ),
        focus_coverage=pd.DataFrame(
            [
                {
                    "ticker": "LOWQ",
                    "underlying_quality_tier": "unknown",
                    "coverage_status": "CANDIDATE_NOT_STRUCTURED",
                }
            ]
        ),
    )

    assert packet["status"].tolist() == ["FAIL_LOW_QUALITY_UNDERLYING_ACTION_SURFACE"]
    assert packet["ticket_bad_underlying_rows"].tolist() == [1]
    assert packet["market_open_recheck_bad_underlying_rows"].tolist() == [1]
    assert packet["focus_bad_actionable_rows"].tolist() == [1]
    assert packet["ticket_bad_tickers"].tolist() == ["URA"]
    assert packet["market_open_recheck_bad_tickers"].tolist() == ["OKLO"]
    assert packet["focus_bad_actionable_tickers"].tolist() == ["LOWQ"]


def test_action_surface_underlying_quality_proof_blocks_liquid_non_core_action_rows() -> None:
    packet = audit.build_action_surface_underlying_quality_proof_packet(
        tickets=pd.DataFrame(
            [
                {
                    "ticker": "DVN",
                    "underlying_quality_tier": "liquid",
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                }
            ]
        ),
        market_open_recheck_queue=pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "underlying_quality_tier": "core",
                }
            ]
        ),
        focus_coverage=pd.DataFrame(
            [
                {
                    "ticker": "URA",
                    "underlying_quality_tier": "excluded",
                    "coverage_status": "NON_ACTIONABLE_UNDERLYING",
                }
            ]
        ),
    )

    assert packet["status"].tolist() == ["FAIL_LOW_QUALITY_UNDERLYING_ACTION_SURFACE"]
    assert packet["ticket_bad_underlying_rows"].tolist() == [1]
    assert packet["market_open_recheck_bad_underlying_rows"].tolist() == [0]
    assert packet["focus_bad_actionable_rows"].tolist() == [0]
    assert packet["audit_only_focus_rows"].tolist() == [1]
    assert packet["audit_only_focus_tickers"].tolist() == ["URA"]
    assert packet["liquid_non_core_action_tickers"].tolist() == ["DVN"]


def test_action_surface_underlying_quality_proof_passes_core_only_action_rows() -> None:
    packet = audit.build_action_surface_underlying_quality_proof_packet(
        tickets=pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "underlying_quality_tier": "core",
                    "ready_to_enter": False,
                    "target_order_status": "target_order_candidate",
                }
            ]
        ),
        market_open_recheck_queue=pd.DataFrame(
            [
                {
                    "ticker": "MSFT",
                    "underlying_quality_tier": "core",
                }
            ]
        ),
        focus_coverage=pd.DataFrame(
            [
                {
                    "ticker": "URA",
                    "underlying_quality_tier": "excluded",
                    "coverage_status": "NON_ACTIONABLE_UNDERLYING",
                }
            ]
        ),
    )

    assert packet["status"].tolist() == ["PASS_ACTION_SURFACES_EXCLUDE_LOW_QUALITY_UNDERLYINGS"]
    assert packet["ticket_bad_underlying_rows"].tolist() == [0]
    assert packet["market_open_recheck_bad_underlying_rows"].tolist() == [0]
    assert packet["focus_bad_actionable_rows"].tolist() == [0]


def test_duplicate_catalyst_reviews_do_not_expand_or_crash_dispatch() -> None:
    candidates = pd.DataFrame(
        [
            {"ticker": "MSFT", "bias": "bullish", "score": 80, "flow_reason": "call flow"},
            {"ticker": "MSFT", "bias": "bullish", "score": 75, "flow_reason": "follow-on flow"},
        ]
    )
    catalyst_reviews = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "catalyst_status": "clear",
                "catalyst_note": "first review",
                "days_to_earnings": "",
                "news_sentiment": "neutral",
                "red_flag_terms": "",
                "support_terms": "",
                "objective_blocker": False,
            },
            {
                "ticker": "MSFT",
                "catalyst_status": "clear",
                "catalyst_note": "duplicate review",
                "days_to_earnings": "",
                "news_sentiment": "neutral",
                "red_flag_terms": "",
                "support_terms": "",
                "objective_blocker": False,
            },
        ]
    )
    priced = pd.DataFrame([{"ticker": "MSFT", "recommendation_status": RecommendationStatus.ENTER.value}])

    tasks = core.build_research_tasks(candidates, {"regime": "risk_on"}, catalyst_reviews, top_trades=1)
    merged = core.apply_catalyst_reviews(priced, catalyst_reviews)

    assert len(tasks["tasks"]) == 2
    assert {task["catalyst_note"] for task in tasks["tasks"]} == {"first review"}
    assert len(merged) == 1
    assert merged["catalyst_note"].tolist() == ["first review"]


def test_run_pipeline_writes_independent_recommendation_artifacts(tmp_path: Path) -> None:
    root = tmp_path
    _write_minimal_uw_fixture(root)

    paths = run_pipeline("2026-05-22", root=root, top_trades=3)
    manifest = json.loads(paths["manifest"].read_text())
    orchestration = json.loads(paths["agent_orchestration"].read_text())
    research_tasks = json.loads(paths["research_tasks"].read_text())
    dispatch_plan = json.loads(paths["agent_dispatch_plan"].read_text())
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    green_tickets = pd.read_csv(paths["green_trade_tickets"])
    target_tickets = pd.read_csv(paths["target_order_candidates"])
    market_open_queue = pd.read_csv(paths["market_open_recheck_queue"])
    catalyst = pd.read_csv(paths["catalyst_reviews"])
    review_board = pd.read_csv(paths["agent_review_board"])
    structure_attempts = pd.read_csv(paths["structure_attempts"])
    strategy_routing = pd.read_csv(paths["strategy_routing_audit"])
    live_quality = pd.read_csv(paths["live_spread_quality_audit"])
    execution_fill_quality = pd.read_csv(paths["execution_fill_quality"])
    sizing = pd.read_csv(paths["sizing_audit"])
    management = pd.read_csv(paths["management_plan"])
    expectancy = pd.read_csv(paths["expectancy_evidence"])
    outcome_audit = pd.read_csv(paths["outcome_evidence_audit"])
    broker_match = pd.read_csv(paths["broker_outcome_match_audit"])
    broker_matched = pd.read_csv(paths["broker_matched_outcomes"])
    strategy_atlas = pd.read_csv(paths["strategy_outcome_atlas"])
    profitability_calibration = pd.read_csv(paths["profitability_calibration"])
    profitability_gap_plan = pd.read_csv(paths["profitability_gap_plan"])
    route_gap = pd.read_csv(paths["route_opportunity_gap"])
    backfill_plan = pd.read_csv(paths["profitability_evidence_backfill_plan"])
    feasibility = pd.read_csv(paths["monthly_feasibility"])
    confidence_audit = pd.read_csv(paths["confidence_audit"])
    confidence_summary = json.loads(paths["confidence_audit_json"].read_text())
    goal_gap_audit = pd.read_csv(paths["goal_confidence_gap_audit"])
    report = paths["report"].read_text(encoding="utf-8")

    assert paths["out_dir"] == root / "out" / "options_agent" / "2026-05-22"
    assert manifest["pipeline_name"] == "Options Agent"
    assert "codexdaily_v4" not in json.dumps(manifest)
    assert manifest["row_counts"]["agent_review_board"] == len(review_board)
    assert manifest["row_counts"]["agent_dispatch_tasks"] == len(dispatch_plan["subagent_tasks"])
    assert manifest["row_counts"]["structure_attempts"] == len(structure_attempts)
    assert manifest["row_counts"]["strategy_routing_audit"] == len(strategy_routing)
    assert manifest["row_counts"]["live_spread_quality_audit"] == len(live_quality)
    assert manifest["row_counts"]["execution_fill_quality"] == len(execution_fill_quality)
    assert manifest["row_counts"]["sizing_audit"] == len(sizing)
    assert manifest["row_counts"]["management_plan"] == len(management)
    assert manifest["row_counts"]["expectancy_evidence"] == len(expectancy)
    assert manifest["row_counts"]["outcome_evidence_audit"] == len(outcome_audit)
    assert manifest["row_counts"]["broker_outcome_match_audit"] == len(broker_match)
    assert manifest["row_counts"]["broker_matched_outcomes"] == len(broker_matched)
    assert manifest["row_counts"]["strategy_outcome_atlas"] == len(strategy_atlas)
    assert manifest["row_counts"]["profitability_calibration"] == len(profitability_calibration)
    assert manifest["row_counts"]["profitability_gap_plan"] == len(profitability_gap_plan)
    assert manifest["row_counts"]["route_opportunity_gap"] == len(route_gap)
    assert manifest["row_counts"]["profitability_evidence_backfill_plan"] == len(backfill_plan)
    assert manifest["row_counts"]["confidence_audit"] == len(confidence_audit)
    assert manifest["row_counts"]["goal_confidence_gap_audit"] == len(goal_gap_audit)
    assert manifest["row_counts"]["market_open_recheck_queue"] == len(market_open_queue)
    assert manifest["row_counts"]["catalyst_evidence"] == 1
    assert manifest["agent_review_summary"]["by_agent_type"]["built_in"] == len(review_board)
    assert manifest["agent_review_summary"]["portfolio_risk_only"] == 0
    assert manifest["agentic_orchestration"]["status"] == "awaiting_subagents"
    assert "Profitability gap plan" in report
    assert manifest["artifacts"]["profitability_gap_plan"].endswith("profitability_gap_plan.csv")
    assert manifest["artifacts"]["profitability_evidence_backfill_plan"].endswith(
        "profitability_evidence_backfill_plan.csv"
    )
    assert manifest["artifacts"]["goal_confidence_gap_audit"].endswith("goal_confidence_gap_audit.csv")
    assert manifest["artifacts"]["execution_fill_quality"].endswith("execution_fill_quality.csv")
    assert research_tasks["schema_version"] == "options_agent.dispatch_tasks.v1"
    assert research_tasks["dispatch_model"] == "codex_subagents"
    assert research_tasks["tasks"][0]["candidate_id"].startswith("WMT:")
    assert dispatch_plan["dispatch_tool"] == "multi_agent_v1.spawn_agent"
    assert {task["agent"] for task in dispatch_plan["subagent_tasks"]} == {
        "catalyst_news",
        "macro_regime",
        "structure_builder",
        "skeptic",
        "portfolio_management",
    }
    assert {"candidate_id", "agent_type", "review_stage", "portfolio_risk_only", "source_artifact"}.issubset(review_board.columns)
    assert {"market_regime", "catalyst", "structure", "skeptic"}.issubset(set(review_board["agent"]))
    assert "portfolio_risk" not in set(review_board["agent"])
    assert orchestration["execution_model"].startswith("two-pass Codex multi-agent dispatch")
    assert {"from": "research_dispatch", "to": "external_subagents", "artifact": "research_tasks.json"} in orchestration[
        "handoffs"
    ]
    assert {"from": "research_dispatch", "to": "codex_subagents", "artifact": "agent_dispatch_plan.json"} in orchestration[
        "handoffs"
    ]
    assert {"from": "codex_subagents", "to": "research_dispatch", "artifact": "agentic_reviews.json"} in orchestration[
        "handoffs"
    ]
    assert {"from": "structure", "to": "synthesis", "artifact": "strategy_routing_audit.csv"} in orchestration["handoffs"]
    assert dispatch_plan["common_context"]["input_artifacts"]["strategy_routing_audit"].endswith("strategy_routing_audit.csv")
    assert strategy_routing["strategy"].tolist() == ["bull_call_debit", "bull_put_credit"]
    assert {"from": "external_subagents", "to": "research_dispatch", "artifact": "external_agent_reviews.csv"} in orchestration[
        "handoffs"
    ]
    assert {"from": "research_dispatch", "to": "synthesis", "artifact": "agent_review_board.csv"} in orchestration["handoffs"]
    assert {"from": "structure", "to": "synthesis", "artifact": "structure_attempts.csv"} in orchestration["handoffs"]
    assert {"from": "portfolio_risk", "to": "synthesis", "artifact": "final_recommendations.csv"} in orchestration["handoffs"]
    assert catalyst["ticker"].tolist() == ["WMT"]
    assert structure_attempts["attempt_stage"].tolist() == ["dated_hot_chain"]
    assert structure_attempts["attempt_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert "codexdaily_v4" not in structure_attempts.to_json()
    assert final["ticker"].tolist() == ["WMT"]
    assert final["visible_in_final_board"].tolist() == [True]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert final["max_profit"].tolist() == [100.0]
    assert decision["status_label"].tolist() == ["YELLOW review"]
    assert decision["status_icon"].tolist() == ["🟡"]
    assert decision["final_action"].tolist() == [RecommendationStatus.REVIEW.value]
    assert decision["execution_status"].tolist() == ["needs_review"]
    assert tickets.empty
    assert green_tickets.empty
    assert target_tickets.empty
    assert {"trade_plan", "sell_leg", "buy_leg", "expiry"}.issubset(tickets.columns)
    assert {"status_icon", "status_label"}.issubset(tickets.columns)
    assert "full_ticket" not in tickets.columns
    assert "live Schwab validation was not requested" in "; ".join(manifest["warnings"])
    assert decision["trade_plan"].tolist() == [
        "SELL 1 WMT 2026-06-19 95 Put / BUY 1 WMT 2026-06-19 90 Put @ 1.00 CREDIT"
    ]
    assert management["management_action"].tolist() == [RecommendationStatus.REVIEW.value]
    assert manifest["expectancy_evidence_summary"]["status"] == "not_proven"
    assert "outcome_evidence_audit_summary" in manifest
    assert "broker_outcome_match_audit_summary" in manifest
    assert "broker_matched_outcomes_summary" in manifest
    assert "strategy_outcome_atlas_summary" in manifest
    assert manifest["confidence_audit_summary"]["status"] == "block"
    assert manifest["confidence_audit_summary"]["order_entry_confidence_rating"] == 0.0
    assert manifest["goal_confidence_gap_audit_summary"]["status"] == "block"
    assert "order_entry_confidence" in manifest["goal_confidence_gap_audit_summary"]["blocking_areas"]
    assert confidence_summary["status"] == "block"
    assert confidence_summary["order_entry_confidence_rating"] == 0.0
    assert confidence_audit["metric"].tolist() == [
        "profitability_confidence_rating",
        "order_entry_confidence_rating",
        "order_mechanics_confidence_rating",
        "goal_confidence_gate",
    ]
    assert "expectancy_evidence" in feasibility["metric"].tolist()
    assert "Expectancy evidence" not in report
    assert "Monthly Readiness Gate" not in report
    assert "Monthly target is not proven" not in report
    assert "Structure attempt rows: 1" in report
    assert "Live spread quality audit" in report
    assert "Execution fill quality" in report
    assert "Send Now Orders" in report
    assert "Target Orders - Target Credits/Debits" in report
    assert "No yellow target orders." in report
    assert "Structural status counts, not order readiness" in report
    assert "## Top Line" in report
    assert "Research/backtest profitability evidence" in report
    assert "not order-entry readiness" in report
    assert "Profitability confidence:" not in report
    assert "Order-entry confidence: 0.0/10" in report
    assert "Order mechanics confidence: 0.0/10" in report
    assert "Goal confidence gap audit:" not in report
    assert "goal_confidence_gap_audit.csv" not in report
    assert "profitability_evidence_backfill_plan.csv" in report
    assert "Outcome evidence audit:" in report
    assert "Broker outcome match audit:" in report
    assert "Broker matched outcomes:" in report
    assert "Profitability calibration:" in report
    assert "Profitability evidence backfill:" in report
    assert "Route opportunity gaps:" in report
    assert "Strategy outcome atlas:" in report
    assert "SELL 1 WMT 2026-06-19 95 Put / BUY 1 WMT 2026-06-19 90 Put @ 1.00 CREDIT" not in report
    assert "No green send-now orders" in report
    assert "WMT260619P00095000" not in report


def test_dispatch_only_writes_subagent_dispatch_plan_without_synthesis(tmp_path: Path) -> None:
    root = tmp_path
    _write_minimal_uw_fixture(root)

    paths = run_pipeline("2026-05-22", root=root, top_trades=3, dispatch_only=True)
    manifest = json.loads(paths["manifest"].read_text())
    research_tasks = json.loads(paths["research_tasks"].read_text())
    dispatch_plan = json.loads(paths["agent_dispatch_plan"].read_text())
    orchestration = json.loads(paths["agent_orchestration"].read_text())
    final = pd.read_csv(paths["final_recommendations"])
    review_board = pd.read_csv(paths["agent_review_board"])
    agentic_reviews = json.loads(paths["agentic_reviews"].read_text())
    confidence_audit = pd.read_csv(paths["confidence_audit"])
    confidence_summary = json.loads(paths["confidence_audit_json"].read_text())
    goal_gap_audit = pd.read_csv(paths["goal_confidence_gap_audit"])
    outcome_audit = pd.read_csv(paths["outcome_evidence_audit"])
    broker_match = pd.read_csv(paths["broker_outcome_match_audit"])
    broker_matched = pd.read_csv(paths["broker_matched_outcomes"])
    strategy_atlas = pd.read_csv(paths["strategy_outcome_atlas"])
    profitability_gap_plan = pd.read_csv(paths["profitability_gap_plan"])
    route_gap = pd.read_csv(paths["route_opportunity_gap"])
    backfill_plan = pd.read_csv(paths["profitability_evidence_backfill_plan"])
    execution_fill_quality = pd.read_csv(paths["execution_fill_quality"])

    assert manifest["mode"] == "agentic_dispatch_pass"
    assert manifest["agentic_orchestration"]["status"] == "dispatch_ready"
    assert manifest["row_counts"]["research_tasks"] == 1
    assert manifest["row_counts"]["agent_dispatch_tasks"] == 5
    assert manifest["row_counts"]["final_recommendations"] == 0
    assert manifest["row_counts"]["confidence_audit"] == 4
    assert manifest["row_counts"]["outcome_evidence_audit"] == 0
    assert manifest["row_counts"]["broker_outcome_match_audit"] == 0
    assert manifest["row_counts"]["broker_matched_outcomes"] == 0
    assert manifest["row_counts"]["strategy_outcome_atlas"] == 0
    assert manifest["row_counts"]["profitability_gap_plan"] == 0
    assert manifest["row_counts"]["route_opportunity_gap"] == 0
    assert manifest["row_counts"]["profitability_evidence_backfill_plan"] == 0
    assert manifest["row_counts"]["execution_fill_quality"] == 0
    assert manifest["row_counts"]["goal_confidence_gap_audit"] == len(goal_gap_audit)
    assert outcome_audit.empty
    assert broker_match.empty
    assert broker_matched.empty
    assert strategy_atlas.empty
    assert profitability_gap_plan.empty
    assert route_gap.empty
    assert backfill_plan.empty
    assert execution_fill_quality.empty
    assert confidence_audit["metric"].tolist() == [
        "profitability_confidence_rating",
        "order_entry_confidence_rating",
        "order_mechanics_confidence_rating",
        "goal_confidence_gate",
    ]
    assert confidence_summary["order_entry_confidence_rating"] == 0.0
    assert manifest["goal_confidence_gap_audit_summary"]["status"] == "block"
    assert "goal_confidence_gate" in manifest["goal_confidence_gap_audit_summary"]["blocking_areas"]
    assert research_tasks["dispatch_model"] == "codex_subagents"
    assert dispatch_plan["dispatch_status"] == "ready_for_codex_subagents"
    assert len(dispatch_plan["subagent_tasks"]) == 5
    assert agentic_reviews == {"reviews": []}
    assert final.empty
    assert review_board.empty
    assert {"from": "research_dispatch", "to": "codex_subagents", "artifact": "agent_dispatch_plan.json"} in orchestration[
        "handoffs"
    ]


def test_local_news_red_flag_keeps_live_validated_trade_in_review(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    _write_wmt_red_flag_news(root)

    paths = run_pipeline("2026-05-22", root=root, top_trades=3, chain_snapshot_dir=snapshot_dir)
    evidence = pd.read_csv(paths["catalyst_evidence"])
    catalyst = pd.read_csv(paths["catalyst_reviews"])
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    review_board = pd.read_csv(paths["agent_review_board"])

    news_rows = evidence[evidence["evidence_type"].eq("local_news")]
    assert news_rows["evidence_status"].tolist() == ["news_red_flag"]
    assert "sec probe" in news_rows["red_flag_terms"].iloc[0]
    assert catalyst["catalyst_status"].tolist() == ["news_red_flag"]
    assert catalyst["news_sentiment"].tolist() == ["negative"]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert decision["execution_status"].tolist() == ["needs_review"]
    assert tickets.empty
    catalyst_reviews = review_board[review_board["agent"].eq("catalyst")]
    assert catalyst_reviews["verdict"].tolist() == ["caution"]
    assert catalyst_reviews["confidence"].tolist() == ["high"]


def test_legacy_chains_snapshot_layout_can_promote_ready_trade(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot_in_legacy_chains_layout(snapshot_dir)

    paths = run_pipeline("2026-05-22", root=root, top_trades=3, chain_snapshot_dir=snapshot_dir)
    manifest = json.loads(paths["manifest"].read_text())
    live = pd.read_csv(paths["live_chain_validation"])
    live_quality = pd.read_csv(paths["live_spread_quality_audit"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])

    assert live["live_validation_status"].tolist() == ["PASS"]
    assert manifest["row_counts"]["live_spread_quality_audit"] == 1
    assert manifest["live_spread_quality_summary"]["status"] == "pass"
    assert live_quality["live_market_quality_status"].tolist() == ["PASS"]
    assert live["chain_source"].str.contains("chains/chain_WMT.json", regex=False).tolist() == [True]
    assert decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert "fresh_live_schwab_required" in decision["execution_blockers"].iloc[0]
    assert "Schwab snapshot chain" in decision["status_reason"].iloc[0]
    assert "live Schwab chain" not in decision["status_reason"].iloc[0]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]


def test_snapshot_validation_can_fallback_to_debit_target_candidate(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_wmt_call_debit_snapshot(snapshot_dir)
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "structure": "bull put spread",
                "quality_status": "qualified",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
                "anchor_strike": 100.0,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.35,
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "trade_quality_status": "reviewable",
                "status_reason": "dated credit structure missing",
            }
        ]
    )

    updated, live, _ = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
    )
    context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=snapshot_dir,
        portfolio_context={"status": "missing", "total_value": 0},
        research_task_count=1,
        external_review_count=1,
        agent_reviews_json=tmp_path / "agentic_reviews.json",
    )
    updated["external_agent_distinct_review_count"] = 4
    updated["external_agent_review_count"] = 4
    updated["external_agent_review_agents"] = "catalyst_news; macro_regime; structure_builder; skeptic"
    updated["agent_support_count"] = 4
    updated = _mark_strategy_expectancy_pass(updated)
    decision = core.synthesize_decision_board(updated, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert updated["live_validation_status"].tolist() == ["PASS"]
    assert updated["recommendation_status"].tolist() == [RecommendationStatus.ENTER.value]
    assert updated["structure"].tolist() == ["bull call debit spread"]
    assert "DEBIT" in updated["trade_plan"].iloc[0]
    assert live["trade_plan"].str.contains("DEBIT", regex=False).tolist() == [True]
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert tickets["entry_type"].tolist() == ["DEBIT"]


def test_review_snapshot_prevents_untasked_contract_reconstruction(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_wmt_call_debit_snapshot(snapshot_dir)
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "structure": "bull put spread",
                "quality_status": "qualified",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
                "anchor_strike": 100.0,
                "status_reason": "diagnostic row excluded from pass-1 contract tasks",
            }
        ]
    )

    updated, live, _ = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
        reviewed_contract_keys={"reviewed-contract-key"},
    )

    assert updated["live_validation_status"].tolist() == ["REVIEW_SNAPSHOT_EXCLUDED"]
    assert updated["construction_source"].tolist() == ["unreviewed_snapshot_diagnostic"]
    assert updated.get("buy_leg", pd.Series([""])).fillna("").tolist() == [""]
    assert "do not construct replacement legs" in updated["live_validation_note"].iloc[0]
    assert live["live_validation_status"].tolist() == ["REVIEW_SNAPSHOT_EXCLUDED"]


def test_live_validation_prefers_clean_debit_alternative_over_flow_anchored_reject(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_wmt_call_debit_with_better_breakout_snapshot(snapshot_dir)
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "structure": "bull put spread",
                "quality_status": "qualified",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
                "anchor_strike": 100.0,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.35,
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "trade_quality_status": "reviewable",
                "status_reason": "dated credit structure missing",
                "iv30d": 0.30,
            }
        ]
    )

    updated, live, _ = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
    )

    assert updated["live_validation_status"].tolist() == ["PASS"]
    assert updated["recommendation_status"].tolist() == [RecommendationStatus.ENTER.value]
    assert updated["structure"].tolist() == ["bull call debit spread"]
    assert updated["quality_gate_reason"].fillna("").tolist() == [""]
    assert updated["construction_source"].tolist() == ["lower_debit_better_reward_risk"]
    assert "110 Call" in updated["trade_plan"].iloc[0]
    assert "115 Call" in updated["trade_plan"].iloc[0]
    assert "debit_width_ratio_above_65pct" not in updated["status_reason"].iloc[0]
    assert live["trade_plan"].str.contains("110 Call", regex=False).tolist() == [True]


def test_live_validation_rejects_wide_live_markets_as_non_actionable(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    _write_wmt_wide_market_snapshot(snapshot_dir)
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "bias": "bullish",
                "structure": "bull put spread",
                "quality_status": "qualified",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
                "anchor_strike": 95.0,
                "signal_premium": 5_000_000,
                "combined_flow_bias": 0.35,
                "marketcap": 650_000_000_000,
                "avg30_volume": 20_000_000,
                "total_open_interest": 500_000,
                "underlying_quality_tier": "core",
                "underlying_quality_reason": "large-cap liquid common stock with sufficient option open interest",
                "trade_quality_status": "reviewable",
                "status_reason": "dated credit structure requires live validation",
            }
        ]
    )

    updated, live, _ = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
    )
    updated["suggested_contracts"] = 1
    updated["external_agent_distinct_review_count"] = 4
    updated["external_agent_review_count"] = 4
    updated["external_agent_review_agents"] = "catalyst_news; macro_regime; structure_builder; skeptic"
    updated["agent_support_count"] = 4
    context = core.build_execution_context(
        live_schwab=False,
        chain_snapshot_dir=snapshot_dir,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=1,
        external_review_count=4,
        external_review_agent_count=4,
        agent_reviews_json=tmp_path / "agentic_reviews.json",
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(updated, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)
    live_quality = core.build_live_spread_quality_audit(updated)

    assert live["live_validation_status"].tolist() == ["PASS"]
    assert updated["recommendation_status"].tolist() == [RecommendationStatus.AVOID.value]
    assert "live_quote_width_pct_above_30pct" in updated["quality_gate_reason"].iloc[0]
    assert live_quality["live_market_quality_status"].tolist() == ["BLOCK"]
    assert live_quality["actionability_impact"].tolist() == ["blocked_not_target_candidate"]
    assert live_quality["live_leg_min_liquidity"].tolist() == [4900.0]
    assert "setup quality gate reject" in updated["status_reason"].iloc[0]
    assert decision["execution_status"].tolist() == ["blocked"]
    assert tickets.empty


def test_live_spread_quality_audit_keeps_no_realistic_spread_rows_visible() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "NFLX",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "live_validation_status": "no_realistic_spread",
                "trade_plan": "SELL 1 NFLX 2026-06-18 92 Call / BUY 1 NFLX 2026-06-18 93 Call @ 0.28 CREDIT",
                "entry_limit": 0.28,
                "quality_gate_reason": "",
            }
        ]
    )

    live_quality = core.build_live_spread_quality_audit(final)

    assert live_quality["ticker"].tolist() == ["NFLX"]
    assert live_quality["live_validation_status"].tolist() == ["NO_REALISTIC_SPREAD"]
    assert live_quality["live_market_quality_status"].tolist() == ["BLOCK"]
    assert live_quality["live_leg_liquidity_status"].tolist() == ["MISSING"]


def test_live_spread_quality_audit_defers_market_closed_recheck_rows() -> None:
    final = pd.DataFrame(
        [
            {
                "recommendation_rank": 1,
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "live_validation_status": "MARKET_CLOSED_RECHECK",
                "trade_plan": "SELL 1 AAPL 2026-06-18 200 Put / BUY 1 AAPL 2026-06-18 195 Put @ 1.50 CREDIT",
                "entry_limit": 1.5,
                "quality_gate_reason": "",
            }
        ]
    )

    live_quality = core.build_live_spread_quality_audit(final)

    assert live_quality["ticker"].tolist() == ["AAPL"]
    assert live_quality["live_market_quality_status"].tolist() == ["DEFERRED_QUOTE_REFRESH"]
    assert live_quality["actionability_impact"].tolist() == ["target_order_price_validation"]
    assert core.summarize_live_spread_quality(live_quality)["status"] == "pass"


def test_live_spread_quality_proof_blocks_bad_markets_that_stay_actionable() -> None:
    live_quality = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "live_market_quality_status": "BLOCK",
                "actionability_impact": "eligible_for_yellow_or_green_surface",
                "live_quote_width_pct": 0.62,
                "live_leg_min_liquidity": 25,
                "live_leg_liquidity_status": "BLOCK",
                "quality_gate_reason": "live_quote_width_pct_above_30pct; live_leg_liquidity_below_100",
            }
        ]
    )

    packet = audit.build_live_spread_quality_proof_packet(live_quality)

    assert packet["status"].tolist() == ["FAIL_BLOCKED_LIVE_MARKETS_STILL_ACTIONABLE"]
    assert packet["blocked_still_actionable_rows"].tolist() == [1]
    assert packet["target_candidate_block_rows"].tolist() == [1]
    assert packet["quote_width_block_rows"].tolist() == [1]
    assert packet["liquidity_block_rows"].tolist() == [1]


def test_live_spread_quality_proof_allows_blocked_audit_visible_rows_without_nan_examples() -> None:
    live_quality = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "live_market_quality_status": "BLOCK",
                "actionability_impact": "visible_for_review",
                "live_quote_width_pct": float("nan"),
                "live_leg_min_liquidity": float("nan"),
                "live_leg_liquidity_status": "MISSING",
                "quality_gate_reason": float("nan"),
            }
        ]
    )

    packet = audit.build_live_spread_quality_proof_packet(live_quality)

    assert packet["status"].tolist() == ["PASS_LIVE_SPREAD_QUALITY_GATED"]
    assert packet["blocked_still_actionable_rows"].tolist() == [0]
    assert packet["target_candidate_block_rows"].tolist() == [0]
    assert "nan" not in packet["blocked_examples"].iloc[0].lower()
    assert "AAPL" in packet["blocked_examples"].iloc[0]


def test_snapshot_only_validation_does_not_fall_back_to_live_for_missing_chain(tmp_path: Path, monkeypatch) -> None:
    from codexuw.schwab_live import SchwabChainValidator

    def fail_if_live_service_is_requested(self):
        raise AssertionError("snapshot-only validation attempted live Schwab fallback")

    monkeypatch.setattr(SchwabChainValidator, "_service", fail_if_live_service_is_requested)
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    priced = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "structure": "bull put spread",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
            }
        ]
    )

    updated, live, notes = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        chain_snapshot_dir=snapshot_dir,
        allow_live_fallback=False,
    )

    assert notes == []
    assert updated["live_validation_status"].tolist() == ["CHAIN_UNAVAILABLE"]
    assert "snapshot missing for WMT" in updated["live_validation_note"].iloc[0]
    assert live["live_validation_status"].tolist() == ["CHAIN_UNAVAILABLE"]


def test_live_validation_cap_defers_lower_priority_tickers_without_schwab_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codexuw.schwab_live import SchwabChainValidator

    class TimeoutService:
        def __init__(self):
            self.calls = []

        def get_option_chain(self, symbol, **kwargs):
            self.calls.append(symbol)
            raise RuntimeError(f"Schwab option-chain request timed out for {symbol} after 1.0s")

    service = TimeoutService()
    monkeypatch.setenv("UWOS_OPTIONS_AGENT_LIVE_CHAIN_TICKER_CAP", "1")
    monkeypatch.setattr(SchwabChainValidator, "_service", lambda self: service)
    priced = pd.DataFrame(
        [
            {
                "ticker": "HIGH",
                "score": 100.0,
                "structure": "bull put spread",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "trade_quality_status": "PASS",
                "underlying_quality_tier": "core",
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
            },
            {
                "ticker": "LOW",
                "score": 1.0,
                "structure": "bull put spread",
                "recommendation_status": RecommendationStatus.REVIEW.value,
                "quality_status": "qualified",
                "trade_quality_status": "reviewable",
                "underlying_quality_tier": "core",
                "expiry": "2026-06-19",
                "anchor_expiry": "2026-06-19",
            },
        ]
    )

    updated, live, notes = core.validate_priced_candidates_live(
        priced,
        "2026-05-22",
        tmp_path / "out",
        allow_live_fallback=True,
        market_session_open=True,
    )

    assert service.calls == ["HIGH"]
    assert "limited to top 1 tickers" in notes[0]
    assert updated["live_validation_status"].tolist() == ["CHAIN_UNAVAILABLE", "LIVE_CHAIN_DEFERRED"]
    assert "not eligible for send-now order entry" in updated["live_validation_note"].iloc[1]
    assert live["live_validation_status"].tolist() == ["CHAIN_UNAVAILABLE", "LIVE_CHAIN_DEFERRED"]


def test_live_chain_default_has_no_ticker_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UWOS_OPTIONS_AGENT_LIVE_CHAIN_TICKER_CAP", raising=False)

    assert core._live_chain_ticker_cap() is None


def test_schwab_chain_validator_maps_berkshire_alias_for_api(tmp_path: Path) -> None:
    from codexuw.schwab_live import SchwabChainValidator

    calls = []

    class FakeService:
        def get_option_chain(self, symbol, **kwargs):
            calls.append(symbol)
            return {"symbol": symbol, "callExpDateMap": {}, "putExpDateMap": {}}

    validator = SchwabChainValidator(tmp_path)
    validator.service = FakeService()

    chain = validator.get_chain("BRKB", dt.date(2026, 7, 17), dt.date(2026, 8, 21))

    assert calls == ["BRK/B"]
    assert chain["symbol"] == "BRK/B"
    assert "BRKB" in validator.chains


def test_class_share_aliases_match_evidence_and_schwab_api(tmp_path: Path) -> None:
    from codexuw.schwab_live import SchwabChainValidator

    calls = []

    class FakeService:
        def get_option_chain(self, symbol, **kwargs):
            calls.append(symbol)
            return {"symbol": symbol, "callExpDateMap": {}, "putExpDateMap": {}}

    validator = SchwabChainValidator(tmp_path)
    validator.service = FakeService()

    chain = validator.get_chain("BFB", dt.date(2026, 7, 17), dt.date(2026, 8, 21))

    assert core.canonical_ticker_key("BF.B") == "BFB"
    assert core.canonical_ticker_key("BF/B") == "BFB"
    assert core.tickers_match("BF-B", "BFB")
    assert core.tickers_match("BRK/B", "BRKB")
    assert calls == ["BF/B"]
    assert chain["symbol"] == "BF/B"
    assert "BFB" in validator.chains


def test_live_expiry_selection_stays_inside_daily_trade_window() -> None:
    asof = dt.date(2026, 5, 22)
    contracts = pd.DataFrame(
        [
            {"right": "P", "expiry": dt.date(2026, 6, 19)},
            {"right": "P", "expiry": dt.date(2028, 1, 21)},
        ]
    )

    selected = core._select_live_expiry(
        contracts,
        asof,
        preferred_expiry=dt.date(2028, 1, 21),
        direction="Bull Put",
    )

    assert selected == dt.date(2026, 6, 19)
    assert (
        core._select_live_expiry(
            pd.DataFrame([{"right": "P", "expiry": dt.date(2028, 1, 21)}]),
            asof,
            preferred_expiry=dt.date(2028, 1, 21),
            direction="Bull Put",
        )
        is None
    )


def test_live_direction_helpers_respect_explicit_bias_before_structure_text() -> None:
    bearish_put_debit = {"bias": "bearish", "structure": "bear put debit spread"}
    bullish_call_debit = {"bias": "bullish", "structure": "bull call debit spread"}

    assert core._credit_direction(bearish_put_debit) == "Bear Call"
    assert core._debit_direction(bearish_put_debit) == "Bear Put"
    assert core._credit_direction(bullish_call_debit) == "Bull Put"
    assert core._debit_direction(bullish_call_debit) == "Bull Call"


def test_live_debit_replacement_relabels_stale_credit_route() -> None:
    row = {
        "ticker": "WMT",
        "bias": "bullish",
        "strategy": "bull_put_credit",
        "strategy_family": "vertical_spread",
        "strategy_route": "bull_put_credit",
        "entry_type": "CREDIT",
        "signal_premium": 2_000_000,
        "combined_flow_bias": 0.30,
    }
    live = {
        "debit": 1.20,
        "spread_width": 5.0,
        "short_strike": 105.0,
        "long_strike": 100.0,
        "short_leg": "WMT260717C00105000",
        "long_leg": "WMT260717C00100000",
        "target_entry": 2.25,
        "live_status": "PASS",
    }

    out = core._apply_live_debit_spread(
        row,
        live,
        direction="Bull Call",
        expiry=dt.date(2026, 7, 17),
        spot=102.0,
        asof_date=dt.date(2026, 6, 9),
    )

    assert out["strategy"] == "bull_call_debit"
    assert out["strategy_route"] == "bull_call_debit"
    assert out["strategy_family"] == "vertical_spread"
    assert out["entry_type"] == "DEBIT"
    assert out["direction"] == "Bull Call"
    assert out["structure"] == "bull call debit spread"
    assert "DEBIT" in out["trade_plan"]


def test_dated_credit_spread_preserves_width_and_route_fields() -> None:
    candidate = {
        "ticker": "WMT",
        "bias": "bullish",
        "close": 102.0,
        "score": 75.0,
        "signal_premium": 3_000_000,
        "combined_flow_bias": 0.35,
        "quality_status": "qualified",
        "issue_type": "Common Stock",
        "marketcap": 700_000_000_000,
        "avg30_volume": 9_000_000,
        "total_open_interest": 500_000,
    }
    hot = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "right": "P",
                "expiry_dt": dt.date(2026, 7, 17),
                "dte": 38,
                "strike": 100.0,
                "bid": 1.25,
                "ask": 1.35,
                "premium": 2_000_000,
                "volume": 1_000,
                "option_symbol": "WMT260717P00100000",
            },
            {
                "ticker": "WMT",
                "right": "P",
                "expiry_dt": dt.date(2026, 7, 17),
                "dte": 38,
                "strike": 95.0,
                "bid": 0.25,
                "ask": 0.35,
                "premium": 1_000_000,
                "volume": 900,
                "option_symbol": "WMT260717P00095000",
            },
        ]
    )

    row = core.construct_credit_spread(candidate, hot)

    assert row["strategy"] == "bull_put_credit"
    assert row["strategy_route"] == "bull_put_credit"
    assert row["strategy_family"] == "vertical_spread"
    assert row["entry_type"] == "CREDIT"
    assert row["direction"] == "Bull Put"
    assert row["short_strike"] == 100.0
    assert row["long_strike"] == 95.0
    assert row["spread_width"] == 5.0
    assert row["credit_width_ratio"] == 0.18
    assert row["target_entry"] == 0.9
    assert "CREDIT" in row["trade_plan"]


def test_dated_hot_chain_construction_rejects_far_dated_only_expiry() -> None:
    candidate = {
        "ticker": "MESO",
        "bias": "bearish",
        "close": 10.0,
        "score": 61.0,
        "signal_premium": 2_000_000,
        "combined_flow_bias": -0.5,
        "quality_status": "qualified",
    }
    hot = pd.DataFrame(
        [
            {
                "ticker": "MESO",
                "right": "C",
                "expiry_dt": dt.date(2028, 1, 21),
                "dte": 609,
                "strike": 20.0,
                "bid": 0.4,
                "ask": 0.5,
                "premium": 1_000_000,
                "volume": 1_000,
                "option_symbol": "MESO280121C00020000",
            },
            {
                "ticker": "MESO",
                "right": "C",
                "expiry_dt": dt.date(2028, 1, 21),
                "dte": 609,
                "strike": 22.0,
                "bid": 0.1,
                "ask": 0.1,
                "premium": 500_000,
                "volume": 800,
                "option_symbol": "MESO280121C00022000",
            },
        ]
    )

    row = core.construct_credit_spread(candidate, hot)

    assert row["trade_plan"] == ""
    assert row["expiry"] == ""
    assert row["recommendation_status"] == RecommendationStatus.REVIEW.value
    assert "no dated UW hot-chain expiry in 7-60 DTE window" in row["status_reason"]


def test_live_snapshot_validation_promotes_visible_trade_and_then_portfolio_annotates(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        portfolio_context={
            "status": "ok",
            "total_value": 100_000,
            "option_underlyings": ["WMT"],
            "large_equity_exposure": {"WMT": 7_500},
        },
    )
    manifest = json.loads(paths["manifest"].read_text())
    live = pd.read_csv(paths["live_chain_validation"])
    structure_attempts = pd.read_csv(paths["structure_attempts"])
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    risk = pd.read_csv(paths["risk_audit"])
    sizing = pd.read_csv(paths["sizing_audit"])
    management = pd.read_csv(paths["management_plan"])

    assert manifest["row_counts"]["live_chain_validation"] == 1
    assert manifest["row_counts"]["structure_attempts"] == 2
    assert live["live_validation_status"].tolist() == ["PASS"]
    assert set(structure_attempts["attempt_stage"]) == {"dated_hot_chain", "live_schwab_chain"}
    assert structure_attempts.loc[structure_attempts["attempt_stage"].eq("live_schwab_chain"), "attempt_status"].tolist() == [
        "PASS"
    ]
    assert final["ticker"].tolist() == ["WMT"]
    assert final["pre_execution_recommendation_status"].tolist() == [
        RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value
    ]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert final["order_entry_status"].tolist() == ["review_only"]
    assert final["entry_limit"].tolist() == [1.0]
    assert final["max_profit"].tolist() == [100.0]
    assert final["max_loss"].tolist() == [400.0]
    assert final["portfolio_risk_flag"].tolist() == [True]
    assert "existing option exposure in WMT" in final["portfolio_risk_note"].iloc[0]
    assert risk["visibility_action"].tolist() == ["annotated_not_hidden"]
    assert decision["final_action"].tolist() == [RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value]
    assert decision["setup_quality_status"].tolist() == ["qualified"]
    assert decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    assert decision["execution_gate_status"].tolist() == ["blocked"]
    assert decision["portfolio_fit_status"].tolist() == ["risk_flagged"]
    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["suggested_contracts"].tolist() == [1]
    assert "fresh_live_schwab_required" in decision["execution_blockers"].iloc[0]
    assert "agentic_reviews_required" in decision["execution_blockers"].iloc[0]
    assert "sizing uses the explicit risk budget" in decision["sizing_note"].iloc[0]
    assert "portfolio annotation only" not in decision["sizing_note"].iloc[0]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert sizing["visibility_action"].tolist() == ["annotated_not_hidden"]
    assert management["management_action"].tolist() == ["REPRICE"]


def test_external_agent_caution_keeps_target_ticket_visible(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "skeptic",
                        "verdict": "caution",
                        "confidence": "high",
                        "note": "news check requires human confirmation",
                        "objective_blocker": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
        portfolio_context={
            "status": "ok",
            "total_value": 100_000,
            "option_underlyings": ["WMT"],
        },
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    reviews = pd.read_csv(paths["external_agent_reviews"])
    research_tasks = json.loads(paths["research_tasks"].read_text())
    review_board = pd.read_csv(paths["agent_review_board"])
    manifest = json.loads(paths["manifest"].read_text())

    assert research_tasks["tasks"][0]["ticker"] == "WMT"
    assert reviews["verdict"].tolist() == ["caution"]
    assert manifest["agent_review_summary"]["external_reviews_present"] is True
    external_rows = review_board[review_board["agent_type"].eq("external")]
    assert external_rows["note"].tolist() == ["news check requires human confirmation"]
    assert final["ticker"].tolist() == ["WMT"]
    assert final["pre_execution_recommendation_status"].tolist() == [
        RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value
    ]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert final["order_entry_status"].tolist() == ["review_only"]
    assert final["portfolio_risk_flag"].tolist() == [True]
    assert "external agent caution: news check requires human confirmation" in final["status_reason"].iloc[0]
    assert decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    assert decision["portfolio_fit_status"].tolist() == ["risk_flagged"]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]


def test_built_in_caution_annotates_without_downgrading_entry_status() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "status_reason": "Schwab chain validated",
            }
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "agent": "market_regime",
                "agent_type": "built_in",
                "verdict": "caution",
                "confidence": "medium",
                "objective_blocker": False,
                "portfolio_risk_only": False,
                "note": "risk_off tape; use smaller size",
            }
        ]
    )

    final = core.apply_agent_reviews(priced, reviews)

    assert final["recommendation_status"].tolist() == [RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value]
    assert "built-in agent caution: risk_off tape; use smaller size" in final["status_reason"].iloc[0]


def test_strategy_supported_debit_spread_stays_actionable_despite_cautions() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "recommendation_status": RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 AAPL 2026-05-29 315 Call / SELL 1 AAPL 2026-05-29 317.5 Call @ 0.62 DEBIT",
                "trade_plan": "BUY 1 AAPL 2026-05-29 315 Call / SELL 1 AAPL 2026-05-29 317.5 Call @ 0.62 DEBIT",
                "entry_limit": 0.62,
                "suggested_contracts": 5,
                "max_profit": 188.0,
                "max_loss": 62.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 4,
                "agent_caution_count": 4,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "actual_forward_expectancy_status": "PASS",
                "actual_forward_expectancy_sample_size": 10,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 6,
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "risk_off"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["trade_quality_confidence_rating"].tolist() == ["HIGH"]
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert decision["ready_to_enter"].tolist() == [True]
    assert tickets["ticker"].tolist() == ["AAPL"]


def test_short_dated_far_otm_debit_spread_is_not_send_now() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "AMD",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 AMD 2026-06-05 462.5 Put / SELL 1 AMD 2026-06-05 460 Put @ 0.69 DEBIT",
                "trade_plan": "BUY 1 AMD 2026-06-05 462.5 Put / SELL 1 AMD 2026-06-05 460 Put @ 0.69 DEBIT",
                "entry_limit": 0.69,
                "suggested_contracts": 5,
                "max_profit": 181.0,
                "max_loss": 69.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 5,
                "agent_caution_count": 0,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "spot_live": 499.0,
                "breakeven": 461.81,
                "dte": 8,
            }
        ]
    )
    final = _mark_strategy_expectancy_pass(final)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "risk_off"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["execution_status"].tolist() == ["waiting_for_price"]
    assert "send_now_debit_breakeven_move_above_4pct" in decision["execution_blockers"].iloc[0]
    assert tickets["status_label"].tolist() == ["YELLOW target"]
    assert "breakeven move too large for send-now" in tickets.apply(core._ticket_recheck_summary, axis=1).iloc[0]


def test_weak_flow_debit_spread_without_outcome_support_is_not_send_now() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "IWM",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "full_ticket": "BUY 1 IWM 2026-06-30 277 Put / SELL 1 IWM 2026-06-30 257 Put @ 2.40 DEBIT",
                "trade_plan": "BUY 1 IWM 2026-06-30 277 Put / SELL 1 IWM 2026-06-30 257 Put @ 2.40 DEBIT",
                "entry_limit": 2.40,
                "suggested_contracts": 3,
                "max_profit": 1760.0,
                "max_loss": 240.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 7,
                "agent_caution_count": 1,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "spot_live": 288.55,
                "breakeven": 274.60,
                "dte": 28,
                "combined_flow_bias": -0.104,
                "actual_forward_expectancy_status": "BLOCK",
                "actual_forward_expectancy_sample_size": 0,
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": 0,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=50,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "risk_off"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["execution_status"].tolist() == ["waiting_for_price"]
    assert decision["status_label"].tolist() == ["YELLOW review"]
    assert "send_now_debit_directional_edge_below_threshold" in decision["execution_blockers"].iloc[0]
    green, target = core.split_trade_ticket_surfaces(tickets)
    assert tickets["ready_to_enter"].tolist() == [False]
    assert green.empty
    assert target.empty


def test_portfolio_management_process_note_does_not_count_as_quality_caution() -> None:
    reviews = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "agent": "portfolio_management",
                "verdict": "caution",
                "objective_blocker": False,
                "portfolio_risk_only": False,
                "note": "Portfolio risk annotation: no sized spread, target exit, or invalidation is present; fresh portfolio context is required.",
            },
            {
                "ticker": "AAPL",
                "agent": "skeptic",
                "verdict": "caution",
                "objective_blocker": False,
                "portfolio_risk_only": False,
                "note": "flow is modest",
            },
        ]
    )

    summary = core._review_summary_by_ticker(reviews)

    assert summary["AAPL"]["caution"] == 1


def test_subagent_review_metadata_is_preserved_into_agent_review_board(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "agentic_reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "candidate_id": "WMT:bullish:88",
                        "ticker": "WMT",
                        "agent": "portfolio_management",
                        "agent_type": "subagent",
                        "review_stage": "portfolio_management",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": "portfolio concentration only; setup quality remains valid",
                        "objective_blocker": False,
                        "portfolio_risk_only": True,
                        "blocker_type": "portfolio",
                        "evidence": "existing WMT exposure noted by subagent",
                        "source_artifact": "agentic_reviews.json",
                        "as_of": "2026-05-22",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
    )
    manifest = json.loads(paths["manifest"].read_text())
    agentic_reviews = json.loads(paths["agentic_reviews"].read_text())
    review_board = pd.read_csv(paths["agent_review_board"])
    subagent_rows = review_board[review_board["agent_type"].eq("subagent")]

    assert manifest["agentic_orchestration"]["status"] == "reviews_ingested"
    assert agentic_reviews["reviews"][0]["candidate_id"] == "WMT:bullish:88"
    assert subagent_rows["candidate_id"].tolist() == ["WMT:bullish:88"]
    assert subagent_rows["review_stage"].tolist() == ["portfolio_management"]
    assert subagent_rows["portfolio_risk_only"].astype(bool).tolist() == [True]
    assert subagent_rows["blocker_type"].tolist() == ["portfolio"]
    assert subagent_rows["evidence"].tolist() == ["existing WMT exposure noted by subagent"]
    assert subagent_rows["source_artifact"].tolist() == ["agentic_reviews.json"]


def test_portfolio_caution_review_does_not_stamp_every_ticket_as_portfolio_risk(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "agentic_reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "portfolio_management",
                        "agent_type": "subagent",
                        "verdict": "caution",
                        "confidence": "high",
                        "note": "correlated watch only; do not stamp as actual portfolio risk",
                        "objective_blocker": False,
                        "portfolio_risk_only": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
        portfolio_context={"status": "ok", "total_value": 100_000},
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    report = paths["report"].read_text(encoding="utf-8")

    assert final["portfolio_risk_flag"].tolist() == [False]
    assert str(final["portfolio_risk_note"].iloc[0]) in {"", "nan"}
    assert "Portfolio risk annotation" not in str(final["status_reason"].iloc[0])
    assert "Portfolio risk annotation" not in str(final["external_agent_review_note"].iloc[0])
    assert decision["portfolio_fit_status"].tolist() == ["clear"]
    assert "portfolio risk noted" not in report
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert "portfolio annotation only" not in report


def test_portfolio_management_avoid_without_account_exposure_is_not_portfolio_risk(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "agentic_reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "portfolio_management",
                        "agent_type": "subagent",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": (
                            "Portfolio risk annotation: no sized spread, target exit, or invalidation "
                            "is present; reduced risk-off sizing and fresh portfolio context are required."
                        ),
                        "objective_blocker": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
        portfolio_context={"status": "ok", "total_value": 100_000},
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    report = paths["report"].read_text(encoding="utf-8")

    assert final["portfolio_risk_flag"].tolist() == [False]
    assert str(final["portfolio_risk_note"].iloc[0]) in {"", "nan"}
    assert decision["portfolio_fit_status"].tolist() == ["clear"]
    assert "portfolio risk noted" not in report
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]


def test_external_portfolio_avoid_annotates_without_blocking_ready_trade(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "portfolio_risk",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": "portfolio crowding only; setup quality remains good",
                        "objective_blocker": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])
    tickets = pd.read_csv(paths["trade_tickets"])
    review_board = pd.read_csv(paths["agent_review_board"])
    report = paths["report"].read_text(encoding="utf-8")
    external_rows = review_board[review_board["agent_type"].eq("external")]

    assert final["pre_execution_recommendation_status"].tolist() == [
        RecommendationStatus.ENTER_WITH_PORTFOLIO_RISK.value
    ]
    assert final["recommendation_status"].tolist() == [RecommendationStatus.REVIEW.value]
    assert final["order_entry_status"].tolist() == ["review_only"]
    assert final["portfolio_risk_flag"].tolist() == [True]
    assert "external_agent_objective_blocker" not in str(final["hard_rejects"].iloc[0])
    assert "external portfolio risk review" in final["portfolio_risk_note"].iloc[0]
    assert external_rows["portfolio_risk_only"].astype(bool).tolist() == [True]
    assert external_rows["blocker_type"].tolist() == ["portfolio"]
    assert decision["execution_status"].tolist() == ["needs_fresh_live_quote"]
    assert decision["requires_portfolio_ack"].tolist() == [False]
    assert "fresh_live_schwab_required" in decision["execution_blockers"].iloc[0]
    assert "portfolio_context_required" in decision["execution_blockers"].iloc[0]
    assert tickets["ready_to_enter"].tolist() == [False]
    assert tickets["target_order_status"].tolist() == ["target_order_candidate"]
    assert "portfolio annotation only" not in report
    assert "portfolio note is annotation only" not in report


def test_external_agent_objective_blocker_blocks_without_hiding_row(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    reviews_json = tmp_path / "reviews.json"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)
    reviews_json.write_text(
        json.dumps(
            {
                "reviews": [
                    {
                        "ticker": "WMT",
                        "agent": "skeptic",
                        "verdict": "avoid",
                        "confidence": "high",
                        "note": "objective thesis break",
                        "objective_blocker": True,
                        "blocker_type": "thesis_break",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=reviews_json,
    )
    final = pd.read_csv(paths["final_recommendations"])
    decision = pd.read_csv(paths["decision_board"])

    assert final["recommendation_status"].tolist() == [RecommendationStatus.AVOID.value]
    assert final["visible_in_final_board"].tolist() == [True]
    assert "external_agent_objective_blocker" in final["hard_rejects"].iloc[0]
    assert decision["setup_quality_status"].tolist() == ["blocked"]
    assert decision["execution_status"].tolist() == ["blocked"]


def test_confidence_audit_blocks_goal_when_current_strategy_cohort_is_negative_and_no_green_orders() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "MSFT",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "actual_forward_strategy_expectancy_status": "BLOCK",
                "actual_forward_strategy_expectancy_sample_size": 0,
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "WARN",
                "sample_size": 15,
                "win_rate": 0.60,
                "avg_pnl": 85.53,
                "total_pnl": 1283.0,
                "profit_factor": 1.448,
                "matched_current_tickers": "MSFT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "WARN",
                "sample_size": 4,
                "win_rate": 0.75,
                "avg_pnl": 86.85,
                "total_pnl": 347.4,
                "profit_factor": 4.229,
                "matched_current_tickers": "MSFT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "BLOCK",
                "sample_size": 76,
                "win_rate": 0.3947,
                "avg_pnl": -28.42,
                "total_pnl": -2160.0,
                "profit_factor": 0.732,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "BLOCK",
                "sample_size": 95,
                "matched_current_tickers": "MSFT",
                "matched_current_count": 1,
                "note": "Actual closed-trade evidence is not positive enough.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 0, "status": "BLOCK", "note": "none"},
            {"metric": "expectancy_evidence", "value": 95, "status": "BLOCK", "note": "not proven"},
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
    )
    summary = core.summarize_confidence_audit(audit)
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]

    assert profitability["rating"] == 3.0
    assert profitability["status"] == "BLOCK"
    assert "current_strategy_cohort_negative" in profitability["blockers"]
    assert order_entry["rating"] == 0.0
    assert "no_green_ready_orders" in order_entry["blockers"]
    assert summary["status"] == "block"
    assert summary["profitability_confidence_rating"] == 3.0
    assert summary["order_entry_confidence_rating"] == 0.0


def test_confidence_audit_counts_goal_gated_rows_for_order_entry_mechanics() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_goal_confidence",
                "execution_blockers": core.GOAL_CONFIDENCE_GATE_BLOCKER,
                "live_validation_status": "PASS",
                "entry_limit": 5.75,
                "suggested_contracts": 1,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "PASS",
                "trade_plan": "SELL 1 GOOG 2026-07-17 350 Put @ 5.75 CREDIT",
            },
            {
                "ticker": "CRM",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_goal_confidence",
                "execution_blockers": core.GOAL_CONFIDENCE_GATE_BLOCKER,
                "live_validation_status": "PASS",
                "entry_limit": 4.60,
                "suggested_contracts": 1,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "LOW",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "PASS",
                "trade_plan": "SELL 1 CRM 2026-07-17 155 Put @ 4.60 CREDIT",
            },
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "BLOCK",
                "sample_size": 0,
                "note": "Profitability is not proven yet.",
            }
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "action_surface": "yellow_target",
                "fill_quality_status": "PASS",
                "trade_plan": "SELL 1 GOOG 2026-07-17 350 Put @ 5.75 CREDIT",
            },
            {
                "ticker": "CRM",
                "action_surface": "yellow_target",
                "fill_quality_status": "PASS",
                "trade_plan": "SELL 1 CRM 2026-07-17 155 Put @ 4.60 CREDIT",
            },
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=pd.DataFrame(
            [
                {
                    "scope": "current_trade_calibration",
                    "ticker": "GOOG",
                    "strategy_route": "short_put",
                    "status": "PASS",
                }
            ]
        ),
        execution_fill_quality=fill_quality,
    )
    summary = core.summarize_confidence_audit(audit)
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]

    assert profitability["status"] == "BLOCK"
    assert "no_green_ready_orders" not in profitability["blockers"]
    assert "profitability_calibration_not_proven" not in profitability["blockers"]
    assert "profitability_calibration=PASS on 1 ticket/proof rows" in profitability["evidence"]
    assert order_entry["status"] == "PASS"
    assert order_entry["rating"] >= core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING
    assert order_entry["sample_size"] == 1
    assert "goal_gate_neutral_ready_rows=2" in order_entry["evidence"]
    assert "goal_gate_neutral_qualified_rows=1" in order_entry["evidence"]
    assert "visible_order_candidate_rows=2" in order_entry["evidence"]
    assert "no_green_ready_orders" not in order_entry["blockers"]
    assert summary["status"] == "block"
    assert summary["order_entry_confidence_rating"] >= core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING


def test_confidence_audit_separates_yellow_order_mechanics_from_send_now_gate() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_profitability_calibration",
                "execution_blockers": "profitability_calibration_required_for_green; goal_confidence_gate_blocked",
                "live_validation_status": "PASS",
                "entry_limit": 3.80,
                "suggested_contracts": 3,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "WARN",
                "trade_plan": "BUY 1 GOOG 2026-07-17 370 Call @ 3.80 DEBIT",
            },
            {
                "ticker": "AMZN",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_profitability_calibration",
                "execution_blockers": "profitability_calibration_required_for_green; goal_confidence_gate_blocked",
                "live_validation_status": "PASS",
                "entry_limit": 0.72,
                "suggested_contracts": 5,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "WARN",
                "trade_plan": "BUY 1 AMZN 2026-07-17 252.5 Call / SELL 1 AMZN 2026-07-17 255 Call @ 0.72 DEBIT",
            },
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "action_surface": "yellow_target",
                "fill_quality_status": "PASS",
                "trade_plan": "BUY 1 GOOG 2026-07-17 370 Call @ 3.80 DEBIT",
            },
            {
                "ticker": "AMZN",
                "action_surface": "yellow_target",
                "fill_quality_status": "PASS",
                "trade_plan": "BUY 1 AMZN 2026-07-17 252.5 Call / SELL 1 AMZN 2026-07-17 255 Call @ 0.72 DEBIT",
            },
        ]
    )
    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        pd.DataFrame([{"source": "expectancy_summary", "evidence_type": "summary", "status": "BLOCK"}]),
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        execution_fill_quality=fill_quality,
    )
    summary = core.summarize_confidence_audit(audit)
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]
    mechanics = audit[audit["metric"].eq("order_mechanics_confidence_rating")].iloc[0]

    assert "visible_non_send_now_rows=2" in profitability["evidence"]
    assert "no_green_ready_orders" not in profitability["blockers"]
    assert order_entry["status"] == "BLOCK"
    assert order_entry["rating"] == 0.0
    assert order_entry["sample_size"] == 0
    assert "ready_to_enter_rows=0" in order_entry["evidence"]
    assert "visible_order_candidate_rows=2" in order_entry["evidence"]
    assert "no_green_ready_orders" in order_entry["blockers"]
    assert mechanics["status"] == "PASS"
    assert mechanics["rating"] >= core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING
    assert mechanics["sample_size"] == 2
    assert "profitability_gate_excluded_from_mechanics_rating" in mechanics["evidence"]
    assert "order_mechanics_fill_quality=PASS" in mechanics["evidence"]
    assert summary["order_entry_confidence_rating"] == 0.0
    assert summary["order_mechanics_confidence_rating"] >= core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING


def test_confidence_audit_defers_portfolio_refresh_for_complete_yellow_targets() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_profitability_calibration",
                "execution_blockers": "portfolio_context_required; profitability_calibration_required_for_green; goal_confidence_gate_blocked",
                "live_validation_status": "PASS",
                "entry_limit": 0.66,
                "suggested_contracts": 1,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "WARN",
                "trade_plan": "BUY 1 GOOG 2026-07-17 370 Call / SELL 1 GOOG 2026-07-17 372.5 Call @ 0.66 DEBIT",
            }
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "GOOG",
                "action_surface": "yellow_target",
                "fill_quality_status": "PASS",
                "trade_plan": "BUY 1 GOOG 2026-07-17 370 Call / SELL 1 GOOG 2026-07-17 372.5 Call @ 0.66 DEBIT",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame(
            [
                {"gate": "portfolio_sizing", "status": "BLOCK", "detail": "portfolio_status=unavailable"},
                {"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"},
            ]
        ),
        pd.DataFrame([{"source": "expectancy_summary", "evidence_type": "summary", "status": "BLOCK"}]),
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": False, "agentic_reviews_ready": True},
        execution_fill_quality=fill_quality,
    )
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]

    assert order_entry["status"] == "BLOCK"
    assert order_entry["rating"] == 0.0
    assert "ready_to_enter_rows=0" in order_entry["evidence"]
    assert "visible_order_candidate_rows=1" in order_entry["evidence"]
    assert "no_green_ready_orders" in order_entry["blockers"]
    assert "portfolio_not_ready" not in order_entry["blockers"]
    assert "portfolio_sizing" not in order_entry["blockers"]


def test_confidence_audit_scores_fill_quality_qualified_visible_subset() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "GOOD",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_profitability_calibration",
                "execution_blockers": "profitability_calibration_required_for_green; goal_confidence_gate_blocked",
                "live_validation_status": "PASS",
                "entry_limit": 0.50,
                "suggested_contracts": 5,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "WARN",
                "trade_plan": "BUY 1 GOOD 2026-07-17 100 Call / SELL 1 GOOD 2026-07-17 101 Call @ 0.50 DEBIT",
            },
            {
                "ticker": "RICH",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_profitability_calibration",
                "execution_blockers": "wait_for_price; profitability_calibration_required_for_green; goal_confidence_gate_blocked",
                "live_validation_status": "PASS",
                "entry_limit": 1.20,
                "suggested_contracts": 5,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "WARN",
                "trade_plan": "BUY 1 RICH 2026-07-17 100 Call / SELL 1 RICH 2026-07-17 101 Call @ 1.20 DEBIT",
            },
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "GOOD",
                "action_surface": "yellow_target",
                "fill_quality_status": "PASS",
                "trade_plan": "BUY 1 GOOD 2026-07-17 100 Call / SELL 1 GOOD 2026-07-17 101 Call @ 0.50 DEBIT",
            },
            {
                "ticker": "RICH",
                "action_surface": "yellow_target",
                "fill_quality_status": "BLOCK",
                "trade_plan": "BUY 1 RICH 2026-07-17 100 Call / SELL 1 RICH 2026-07-17 101 Call @ 1.20 DEBIT",
            },
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        pd.DataFrame([{"source": "expectancy_summary", "evidence_type": "summary", "status": "BLOCK"}]),
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        execution_fill_quality=fill_quality,
    )
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]
    mechanics = audit[audit["metric"].eq("order_mechanics_confidence_rating")].iloc[0]

    assert order_entry["status"] == "BLOCK"
    assert order_entry["rating"] == 0.0
    assert order_entry["sample_size"] == 0
    assert "ready_to_enter_rows=0" in order_entry["evidence"]
    assert "visible_order_candidate_rows=2" in order_entry["evidence"]
    assert "no_green_ready_orders" in order_entry["blockers"]
    assert mechanics["status"] == "PASS"
    assert mechanics["sample_size"] == 1
    assert "order_mechanics_candidate_rows_before_fill_quality=2" in mechanics["evidence"]
    assert "order_mechanics_candidate_rows_excluded_by_fill_quality=1" in mechanics["evidence"]
    assert "order_mechanics_fill_quality_not_all_pass" not in mechanics["blockers"]


def test_confidence_audit_counts_review_ticket_fill_quality_for_mechanics_only() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "BA",
                "ready_to_enter": False,
                "target_order_status": "review_only_expectancy_evidence",
                "order_readiness": "review_only_after_profitability_calibration",
                "execution_blockers": "profitability_calibration_required_for_green; goal_confidence_gate_blocked",
                "live_validation_status": "PASS",
                "entry_limit": 3.50,
                "suggested_contracts": 1,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "WARN",
                "profitability_calibration_status": "WARN",
                "trade_plan": "BUY 1 BA 2026-07-17 225 Call @ 3.50 DEBIT",
            },
            {
                "ticker": "CRM",
                "ready_to_enter": False,
                "target_order_status": "review_only_expectancy_evidence",
                "order_readiness": "review_only_after_profitability_calibration",
                "execution_blockers": "profitability_calibration_required_for_green; goal_confidence_gate_blocked",
                "live_validation_status": "PASS",
                "entry_limit": 2.32,
                "suggested_contracts": 1,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "WARN",
                "profitability_calibration_status": "WARN",
                "trade_plan": "BUY 1 CRM 2026-07-17 165 Call @ 2.32 DEBIT",
            },
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "BA",
                "action_surface": "ticket_review",
                "fill_quality_status": "PASS",
                "trade_plan": "BUY 1 BA 2026-07-17 225 Call @ 3.50 DEBIT",
            },
            {
                "ticker": "CRM",
                "action_surface": "ticket_review",
                "fill_quality_status": "PASS",
                "trade_plan": "BUY 1 CRM 2026-07-17 165 Call @ 2.32 DEBIT",
            },
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        pd.DataFrame([{"source": "expectancy_summary", "evidence_type": "summary", "status": "BLOCK"}]),
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        execution_fill_quality=fill_quality,
    )
    summary = core.summarize_confidence_audit(audit)
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]
    mechanics = audit[audit["metric"].eq("order_mechanics_confidence_rating")].iloc[0]

    assert order_entry["status"] == "BLOCK"
    assert order_entry["rating"] == 0.0
    assert order_entry["sample_size"] == 0
    assert "ready_to_enter_rows=0" in order_entry["evidence"]
    assert "visible_order_candidate_rows=2" in order_entry["evidence"]
    assert "no_green_ready_orders" in order_entry["blockers"]
    assert mechanics["status"] == "PASS"
    assert mechanics["rating"] >= core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING
    assert mechanics["sample_size"] == 2
    assert "order_mechanics_fill_quality=PASS" in mechanics["evidence"]
    assert "order_mechanics_fill_quality_not_all_pass" not in mechanics["blockers"]
    assert summary["order_entry_confidence_rating"] == 0.0
    assert summary["order_mechanics_confidence_rating"] >= core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING


def test_small_calibrated_short_put_counts_for_order_entry_after_goal_gate() -> None:
    final = pd.DataFrame(
        [
            {
                "ticker": "UBER",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "structure": "cash secured put",
                "full_ticket": "SELL 1 UBER 2026-07-17 70 Put @ 1.65 CREDIT",
                "trade_plan": "SELL 1 UBER 2026-07-17 70 Put @ 1.65 CREDIT",
                "entry_limit": 1.65,
                "suggested_contracts": 1,
                "max_profit": 165.0,
                "max_loss": 6835.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "agent_support_count": 8,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "contract_review_status": "PASS",
                "portfolio_cash": 200_000.0,
                "account_risk_pct": 0.0092,
                "dte": 30,
                "macro_calendar_status": "verified",
                "macro_event_count_before_expiry": 0,
                "earnings_before_expiry": False,
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 34,
                "profitability_calibration_status": "PASS",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 30,
                "profitability_calibration_actual_avg_pnl": 87.5,
                "profitability_calibration_actual_profit_factor": 1.9,
                "profitability_calibration_replay_status": "PASS",
                "profitability_calibration_replay_sample_size": 32,
            }
        ]
    )
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 739_277.93, "cash": 200_000.0},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(final, market_regime={"regime": "mixed"}, execution_context=context)
    tickets = core.build_trade_tickets(decision)

    assert decision["ready_to_enter"].tolist() == [True]
    assert core.POSITION_PROFIT_MATERIALITY_BLOCKER not in str(decision["execution_blockers"].iloc[0])
    assert tickets["ready_to_enter"].tolist() == [True]

    gate_audit = pd.DataFrame(
        [
            {
                "metric": "profitability_confidence_rating",
                "rating": 4.0,
                "threshold": core.MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING,
                "status": "BLOCK",
                "sample_size": 0,
                "evidence": "profitability still unproven",
                "blockers": "profitability_calibration_not_proven",
                "required_next_action": "collect positive realized outcomes",
            },
            {
                "metric": "order_entry_confidence_rating",
                "rating": 9.0,
                "threshold": core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING,
                "status": "PASS",
                "sample_size": 1,
                "evidence": "row-level mechanics pass",
                "blockers": "",
                "required_next_action": "keep yellow until profitability gate passes",
            },
            {
                "metric": "goal_confidence_gate",
                "rating": 4.0,
                "threshold": core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING,
                "status": "BLOCK",
                "sample_size": 1,
                "evidence": "profitability=4.0/10; order_entry=9.0/10",
                "blockers": "profitability_calibration_not_proven",
                "required_next_action": "do not publish green rows",
            },
        ]
    )

    gated_decision = core.apply_goal_confidence_gate_to_decision_board(decision, gate_audit)
    gated_tickets = core.build_trade_tickets(gated_decision)
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "UBER",
                "action_surface": "yellow_target",
                "fill_quality_status": "PASS",
                "trade_plan": "SELL 1 UBER 2026-07-17 70 Put @ 1.65 CREDIT",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "BLOCK",
                "sample_size": 0,
                "note": "Profitability confidence remains a separate global gate.",
            }
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "UBER",
                "strategy_route": "short_put",
                "status": "PASS",
            }
        ]
    )
    audit = core.build_confidence_audit(
        gated_decision,
        gated_tickets,
        core.build_execution_readiness(gated_decision, context),
        expectancy,
        pd.DataFrame(),
        context,
        profitability_calibration=calibration,
        execution_fill_quality=fill_quality,
    )
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]

    assert gated_tickets["ready_to_enter"].tolist() == [False]
    assert gated_tickets["order_readiness"].tolist() == ["target_order_after_goal_confidence"]
    assert gated_tickets["execution_blockers"].tolist() == [core.GOAL_CONFIDENCE_GATE_BLOCKER]
    assert order_entry["status"] == "PASS"
    assert order_entry["rating"] >= core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING
    assert "proof_origin=goal_gate_neutral" in order_entry["evidence"]
    assert "goal_gate_neutral_qualified_rows=1" in order_entry["evidence"]
    assert "no_green_ready_orders" not in order_entry["blockers"]


def test_target_ticket_preserves_and_displays_intended_limit() -> None:
    decision = pd.DataFrame(
        [
            {
                "ticker": "AMZN",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "structure": "bull call debit spread",
                "status_icon": "yellow",
                "status_label": "TARGET",
                "ready_to_enter": False,
                "execution_status": "waiting_for_price",
                "execution_gate_status": "pass",
                "target_order_status": "target_order_wait_for_price",
                "live_validation_status": "PASS",
                "trade_plan": "BUY 1 AMZN 2026-07-24 242.5 Call / SELL 1 AMZN 2026-07-24 245 Call @ 1.59 DEBIT",
                "entry_limit": 1.59,
                "target_entry": 1.12,
                "suggested_contracts": 1,
                "max_profit": 91.0,
                "max_loss": 159.0,
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "profitability_calibration_status": "PASS",
            }
        ]
    )

    tickets = core.build_trade_tickets(decision)

    assert tickets["target_entry"].tolist() == [1.12]
    assert core._ticket_limit_display(tickets.iloc[0]) == "<=1.12 DEBIT"
    assert "current debit 1.59 is too high" in core._ticket_recheck_summary(tickets.iloc[0])


def test_confidence_audit_does_not_name_strategy_cohort_gap_without_action_candidates() -> None:
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "BLOCK",
                "sample_size": 0,
                "matched_current_count": 0,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "BLOCK",
                "sample_size": 0,
                "matched_current_count": 0,
                "note": "no current ticket strategy family to compare",
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "BLOCK",
                "sample_size": 0,
                "matched_current_count": 0,
            },
        ]
    )
    monthly = pd.DataFrame([{"metric": "ready_ticket_count", "value": 0, "status": "BLOCK"}])

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert "current_strategy_cohort_not_proven" not in profitability["blockers"]
    assert "current_strategy_cohort=no_action_candidates" in profitability["evidence"]


def test_confidence_audit_names_zero_pass_profitability_bucket_atlas() -> None:
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "WARN",
                "sample_size": 15,
                "win_rate": 0.60,
                "avg_pnl": 85.0,
                "profit_factor": 1.4,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "WARN",
                "sample_size": 4,
                "win_rate": 0.75,
                "avg_pnl": 86.0,
                "profit_factor": 4.0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "WARN",
                "sample_size": 19,
                "note": "Partial positive evidence only.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 0, "status": "BLOCK", "note": "none"},
            {"metric": "expectancy_evidence", "value": 19, "status": "BLOCK", "note": "not proven"},
        ]
    )
    atlas = pd.DataFrame(
        [
            {
                "bucket_key": "short_put|short_put|CREDIT|bullish|mixed|dte_31_60|credit_rich|liquidity_deep",
                "status": "WARN",
                "actual_bucket_status": "BLOCK",
                "replay_bucket_status": "PASS",
                "current_ticket_count": 1,
                "primary_gap": "actual_bucket_negative_or_weak",
            }
        ],
        columns=core.PROFITABILITY_BUCKET_ATLAS_COLUMNS,
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_bucket_atlas=atlas,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]
    summary = core.summarize_confidence_audit(audit)

    assert profitability["rating"] <= 3.0
    assert "no_actual_and_replay_bucket_pass" in profitability["blockers"]
    assert "profitability_bucket_atlas=0 pass/1 buckets" in profitability["evidence"]
    assert "no_actual_and_replay_bucket_pass" in summary["blockers"]


def test_confidence_audit_names_unrealized_forward_outcome_ledgers() -> None:
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "WARN",
                "sample_size": 4,
                "win_rate": 0.75,
                "avg_pnl": 75.0,
                "profit_factor": 2.0,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "win_rate": 0.70,
                "avg_pnl": 100.0,
                "profit_factor": 2.0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "WARN",
                "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE + 4,
                "note": "Partial positive evidence only.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 0, "status": "BLOCK", "note": "none"},
            {"metric": "expectancy_evidence", "value": 34, "status": "BLOCK", "note": "not proven"},
        ]
    )
    outcome_audit = pd.DataFrame(
        [
            {
                "source": "codexuw_execute_outcome_ledger",
                "status": "BLOCK",
                "row_count": 10,
                "realized_pnl_count": 0,
                "current_ticker_realized_count": 0,
                "contributes_to_expectancy": False,
            },
            {
                "source": "codexuw_recommendation_outcome_ledger",
                "status": "BLOCK",
                "row_count": 20,
                "realized_pnl_count": 0,
                "current_ticker_realized_count": 0,
                "contributes_to_expectancy": False,
            },
            {
                "source": "schwab_closed_trades",
                "status": "PASS",
                "row_count": 4,
                "realized_pnl_count": 4,
                "current_ticker_realized_count": 4,
                "contributes_to_expectancy": True,
            },
        ],
        columns=core.OUTCOME_EVIDENCE_AUDIT_COLUMNS,
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        outcome_evidence_audit=outcome_audit,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert profitability["rating"] <= 3.0
    assert "forward_outcome_ledgers_have_no_realized_pnl" in profitability["blockers"]
    assert "outcome_sources=1 contributing/3 sources; realized_pnl_rows=4" in profitability["evidence"]


def test_confidence_audit_allows_positive_broker_backfill_to_clear_forward_ledger_gap() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE,
                "profitability_calibration_status": "PASS",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "WARN",
                "sample_size": 4,
                "win_rate": 0.75,
                "avg_pnl": 75.0,
                "profit_factor": 2.0,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "win_rate": 0.70,
                "avg_pnl": 100.0,
                "profit_factor": 2.0,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "PASS",
                "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "win_rate": 0.70,
                "avg_pnl": 100.0,
                "profit_factor": 2.0,
            },
            {
                "source": "broker_matched_options_agent_outcomes",
                "evidence_type": "broker_matched_options_agent_outcomes",
                "status": "PASS",
                "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "win_rate": 0.70,
                "avg_pnl": 100.0,
                "profit_factor": 2.0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE * 3 + 4,
                "note": "Positive evidence.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "expectancy_evidence", "value": 94, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    outcome_audit = pd.DataFrame(
        [
            {"source": "codexuw_execute_outcome_ledger", "status": "BLOCK", "realized_pnl_count": 0, "contributes_to_expectancy": False},
            {"source": "codexuw_recommendation_outcome_ledger", "status": "BLOCK", "realized_pnl_count": 0, "contributes_to_expectancy": False},
            {"source": "broker_matched_outcomes", "status": "PASS", "realized_pnl_count": core.MIN_EXPECTANCY_SAMPLE_SIZE, "contributes_to_expectancy": False},
        ],
        columns=core.OUTCOME_EVIDENCE_AUDIT_COLUMNS,
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "short_put",
                "status": "PASS",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_ticker_bucket",
                "replay_bucket_status": "PASS",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
        outcome_evidence_audit=outcome_audit,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert "forward_outcome_ledgers_have_no_realized_pnl" not in profitability["blockers"]
    assert "forward_outcome_ledgers_empty_but_broker_backfill_attribution_passed" in profitability["evidence"]
    assert profitability["rating"] >= core.MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING


def test_confidence_audit_names_negative_broker_matched_outcomes() -> None:
    expectancy = pd.DataFrame(
        [
            {
                "source": "broker_matched_outcomes",
                "evidence_type": "broker_matched_recommendation_outcomes",
                "status": "BLOCK",
                "sample_size": 3,
                "win_rate": 0.0,
                "avg_pnl": -149.67,
                "profit_factor": 0.0,
                "matched_current_tickers": "META",
                "matched_current_count": 1,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "BLOCK",
                "sample_size": 0,
                "note": "No sufficient positive expectancy evidence is available.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 0, "status": "BLOCK", "note": "none"},
            {"metric": "expectancy_evidence", "value": 0, "status": "BLOCK", "note": "not proven"},
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert profitability["rating"] <= 3.0
    assert "broker_matched_options_agent_outcomes_negative" in profitability["blockers"]
    assert "broker_matched_options_agent=negative sample=3" in profitability["evidence"]


def test_confidence_audit_treats_small_unrelated_negative_broker_sample_as_insufficient() -> None:
    expectancy = pd.DataFrame(
        [
            {
                "source": "broker_matched_options_agent_outcomes",
                "evidence_type": "broker_matched_options_agent_outcomes",
                "status": "BLOCK",
                "sample_size": 2,
                "win_rate": 0.5,
                "avg_pnl": -37.5,
                "profit_factor": 0.727,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "BLOCK",
                "sample_size": 2,
                "note": "No sufficient positive expectancy evidence is available.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 0, "status": "BLOCK", "note": "none"},
            {"metric": "expectancy_evidence", "value": 0, "status": "BLOCK", "note": "not proven"},
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert "broker_matched_options_agent_outcomes_negative" not in profitability["blockers"]
    assert "broker_matched_options_agent_outcomes_sample_too_small" in profitability["blockers"]
    assert "broker_matched_options_agent=negative_diagnostic_sample sample=2" in profitability["evidence"]


def test_confidence_audit_caps_profitability_without_sufficient_broker_attribution() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": core.MIN_TICKER_EXPECTANCY_SAMPLE_SIZE,
                "profitability_calibration_status": "PASS",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 35,
                "win_rate": 0.60,
                "avg_pnl": 100.0,
                "total_pnl": 3500.0,
                "profit_factor": 1.5,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "win_rate": 0.70,
                "avg_pnl": 80.0,
                "total_pnl": 3200.0,
                "profit_factor": 2.0,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "PASS",
                "sample_size": 34,
                "win_rate": 0.6471,
                "avg_pnl": 92.09,
                "total_pnl": 3131.0,
                "profit_factor": 1.823,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "broker_matched_options_agent_outcomes",
                "evidence_type": "broker_matched_options_agent_outcomes",
                "status": "WARN",
                "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE - 1,
                "win_rate": 0.62,
                "avg_pnl": 77.0,
                "total_pnl": 2233.0,
                "profit_factor": 1.6,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 109,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "one_cycle_max_profit", "value": 1000, "status": "PASS", "note": "capacity"},
            {"metric": "cycles_needed_at_max_profit", "value": 4, "status": "PASS", "note": "capacity"},
            {"metric": "expectancy_evidence", "value": 109, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "short_put",
                "status": "PASS",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_ticker_bucket",
                "replay_bucket_status": "PASS",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert profitability["status"] == "BLOCK"
    assert profitability["rating"] == 6.0
    assert "broker_matched_options_agent_outcomes_sample_too_small" in profitability["blockers"]
    assert "broker_matched_options_agent=insufficient_pipeline_attribution" in profitability["evidence"]


def test_confidence_audit_passes_only_with_positive_expectancy_and_green_order_entry_proof() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 5,
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 35,
                "win_rate": 0.60,
                "avg_pnl": 100.0,
                "total_pnl": 3500.0,
                "profit_factor": 1.5,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "win_rate": 0.70,
                "avg_pnl": 80.0,
                "total_pnl": 3200.0,
                "profit_factor": 2.0,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "PASS",
                "sample_size": 34,
                "win_rate": 0.6471,
                "avg_pnl": 92.09,
                "total_pnl": 3131.0,
                "profit_factor": 1.823,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 5,
                "win_rate": 0.60,
                "avg_pnl": 75.0,
                "total_pnl": 375.0,
                "profit_factor": 1.4,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "broker_matched_outcomes",
                "evidence_type": "broker_matched_recommendation_outcomes",
                "status": "PASS",
                "sample_size": core.MIN_EXPECTANCY_SAMPLE_SIZE,
                "win_rate": 0.60,
                "avg_pnl": 88.0,
                "total_pnl": 2640.0,
                "profit_factor": 1.7,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 109,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "one_cycle_max_profit", "value": 1000, "status": "PASS", "note": "capacity"},
            {"metric": "cycles_needed_at_max_profit", "value": 4, "status": "PASS", "note": "capacity"},
            {"metric": "expectancy_evidence", "value": 109, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "action_surface": "green_send_now",
                "fill_quality_status": "PASS",
                "trade_plan": "",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        execution_fill_quality=fill_quality,
    )
    summary = core.summarize_confidence_audit(audit)

    assert audit.loc[audit["metric"].eq("profitability_confidence_rating"), "status"].tolist() == ["PASS"]
    assert audit.loc[audit["metric"].eq("order_entry_confidence_rating"), "status"].tolist() == ["PASS"]
    assert audit.loc[audit["metric"].eq("goal_confidence_gate"), "status"].tolist() == ["PASS"]
    assert summary["profitability_confidence_rating"] >= core.MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING
    assert summary["order_entry_confidence_rating"] >= core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING


def test_confidence_audit_caps_order_entry_when_green_fill_quality_fails() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "profitability_calibration_status": "PASS",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 40,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            }
        ]
    )
    fill_quality = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "action_surface": "green_send_now",
                "fill_quality_status": "BLOCK",
                "trade_plan": "SELL 1 WMT 2026-06-19 95 Put @ 1.25 CREDIT",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        execution_fill_quality=fill_quality,
    )
    order_entry = audit[audit["metric"].eq("order_entry_confidence_rating")].iloc[0]

    assert order_entry["rating"] == 6.0
    assert order_entry["status"] == "BLOCK"
    assert "green_execution_fill_quality_not_all_pass" in order_entry["blockers"]


def test_confidence_audit_caps_profitability_when_strategy_cohort_is_weak_not_losing() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 5,
                "profitability_calibration_status": "PASS",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 35,
                "win_rate": 0.60,
                "avg_pnl": 100.0,
                "total_pnl": 3500.0,
                "profit_factor": 1.5,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "win_rate": 0.70,
                "avg_pnl": 80.0,
                "total_pnl": 3200.0,
                "profit_factor": 2.0,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "BLOCK",
                "sample_size": 110,
                "win_rate": 0.4727,
                "avg_pnl": 8.83,
                "total_pnl": 971.0,
                "profit_factor": 1.082,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 185,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "one_cycle_max_profit", "value": 1000, "status": "PASS", "note": "capacity"},
            {"metric": "cycles_needed_at_max_profit", "value": 4, "status": "PASS", "note": "capacity"},
            {"metric": "expectancy_evidence", "value": 185, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "short_put",
                "status": "PASS",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_route",
                "replay_bucket_status": "PASS",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]
    summary = core.summarize_confidence_audit(audit)

    assert profitability["status"] == "BLOCK"
    assert profitability["rating"] == 6.0
    assert "current_strategy_cohort_weak_under_threshold" in profitability["blockers"]
    assert "current_strategy_cohort_negative" not in profitability["blockers"]
    assert summary["profitability_confidence_rating"] < core.MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING
    assert "current_strategy_cohort_weak_under_threshold" in summary["blockers"]


def test_confidence_audit_rates_weak_positive_strategy_support_at_seven() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "AMZN",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_profitability_calibration",
                "execution_blockers": core.GOAL_CONFIDENCE_GATE_BLOCKER,
                "live_validation_status": "PASS",
                "entry_limit": 0.72,
                "suggested_contracts": 5,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 3,
                "profitability_calibration_status": "PASS",
                "trade_plan": "BUY 1 AMZN 2026-07-17 252.5 Call / SELL 1 AMZN 2026-07-17 255 Call @ 0.72 DEBIT",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 171,
                "win_rate": 0.58,
                "avg_pnl": 44.0,
                "total_pnl": 7524.0,
                "profit_factor": 1.42,
            },
            {
                "source": "broker_matched_options_agent_outcomes",
                "evidence_type": "broker_matched_options_agent_outcomes",
                "status": "WARN",
                "sample_size": 2,
                "win_rate": 1.0,
                "avg_pnl": 236.0,
                "total_pnl": 472.0,
                "profit_factor": float("inf"),
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "BLOCK",
                "sample_size": 143,
                "win_rate": 0.4895,
                "avg_pnl": 12.48,
                "total_pnl": 1785.0,
                "profit_factor": 1.113,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "WARN",
                "sample_size": 4,
                "win_rate": 0.75,
                "avg_pnl": 86.85,
                "total_pnl": 347.4,
                "profit_factor": 4.229,
            },
            {
                "source": "codexuw_replay_decision_pass_model",
                "evidence_type": "replay_backtest_decision_pass_model",
                "status": "PASS",
                "sample_size": 142,
                "win_rate": 0.9366,
                "avg_pnl": 130.46,
                "total_pnl": 18525.3,
                "profit_factor": 13.343,
                "matched_current_tickers": "AMZN",
                "matched_current_count": 1,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 719,
                "note": "Actual evidence exists, but promotion evidence is incomplete.",
            },
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "AMZN",
                "strategy_route": "bull_call_debit",
                "status": "PASS",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_route_economics_bucket",
                "replay_bucket_status": "PASS",
            }
        ]
    )
    bucket_atlas = pd.DataFrame(
        [
            {
                "bucket_key": "vertical_spread|DEBIT|bullish|mixed|dte_31_60|debit_defined|liquidity_deep",
                "status": "PASS",
                "actual_bucket_status": "PASS",
                "replay_bucket_status": "PASS",
                "current_ticket_count": 1,
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
        profitability_bucket_atlas=bucket_atlas,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert profitability["rating"] == 7.0
    assert profitability["status"] == "PASS"
    assert "current_strategy_cohort_weak_under_threshold" in profitability["blockers"]
    assert "current_strategy_cohort=weak_positive_under_threshold" in profitability["evidence"]
    assert "leakage_safe_replay_decision_pass_model=PASS" in profitability["evidence"]
    assert "broker_matched_options_agent_outcomes_sample_too_small" in profitability["blockers"]
    assert "replay_decision_pass_sample_too_small" not in profitability["blockers"]


def test_confidence_audit_names_actual_bucket_precision_gap_when_route_support_exists() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 36,
                "profitability_calibration_status": "PASS",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 36,
                "win_rate": 0.61,
                "avg_pnl": 43.92,
                "total_pnl": 1581.0,
                "profit_factor": 1.295,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "broker_matched_options_agent_outcomes",
                "evidence_type": "broker_matched_options_agent_outcomes",
                "status": "PASS",
                "sample_size": 30,
                "win_rate": 0.70,
                "avg_pnl": 80.0,
                "total_pnl": 2400.0,
                "profit_factor": 2.0,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "PASS",
                "sample_size": 36,
                "win_rate": 0.61,
                "avg_pnl": 43.92,
                "total_pnl": 1581.0,
                "profit_factor": 1.295,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 32,
                "win_rate": 0.70,
                "avg_pnl": 71.31,
                "total_pnl": 2281.92,
                "profit_factor": 1.262,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 134,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "one_cycle_max_profit", "value": 1000, "status": "PASS", "note": "capacity"},
            {"metric": "cycles_needed_at_max_profit", "value": 4, "status": "PASS", "note": "capacity"},
            {"metric": "expectancy_evidence", "value": 134, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "short_put",
                "status": "WARN",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_route",
                "replay_bucket_status": "PASS",
                "route_replay_status": "PASS",
                "route_replay_sample_size": 30,
            }
        ]
    )
    bucket_atlas = pd.DataFrame(
        [
            {
                "bucket_key": "short_put|CREDIT|bullish|risk_on|dte_31_60|credit_rich|liquidity_deep",
                "status": "WARN",
                "actual_support_status": "WARN",
                "replay_bucket_status": "PASS",
                "current_ticket_count": 1,
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
        profitability_bucket_atlas=bucket_atlas,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert "actual_bucket_precision_gap" in profitability["blockers"]
    assert "no_actual_and_replay_bucket_pass" not in profitability["blockers"]
    assert "route_level_actual_and_replay_support=1 rows; exact_bucket_pass=0" in profitability["evidence"]
    assert profitability["rating"] == 6.0


def test_confidence_audit_counts_route_positive_support_without_global_actual_negative_label() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "BA",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "ready_to_enter": False,
                "target_order_status": "target_order_candidate",
                "order_readiness": "target_order_after_profitability_calibration",
                "execution_blockers": "profitability_calibration_required_for_green; goal_confidence_gate_blocked",
                "live_validation_status": "PASS",
                "entry_limit": 0.62,
                "suggested_contracts": 5,
                "execution_confidence_rating": "NOT_EXECUTION_READY",
                "order_mechanics_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 69,
                "profitability_calibration_status": "WARN",
                "trade_plan": "BUY 1 BA 2026-07-17 235 Call / SELL 1 BA 2026-07-17 237.5 Call @ 0.62 DEBIT",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "WARN",
                "sample_size": 11,
                "win_rate": 0.3636,
                "avg_pnl": -56.73,
                "profit_factor": 0.558,
                "matched_current_tickers": "BA",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "PASS",
                "sample_size": 69,
                "win_rate": 0.5072,
                "avg_pnl": 51.99,
                "profit_factor": 1.733,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "broker_matched_options_agent_outcomes",
                "evidence_type": "broker_matched_options_agent_outcomes",
                "status": "WARN",
                "sample_size": 2,
                "win_rate": 1.0,
                "avg_pnl": 236.0,
                "profit_factor": float("inf"),
            },
            {
                "source": "codexuw_replay_decision_pass_model",
                "evidence_type": "replay_backtest_decision_pass_model",
                "status": "PASS",
                "sample_size": 142,
                "win_rate": 0.9366,
                "avg_pnl": 130.46,
                "profit_factor": 13.343,
                "matched_current_tickers": "BA",
                "matched_current_count": 1,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "WARN",
                "sample_size": 7500,
                "matched_current_tickers": "BA",
                "matched_current_count": 1,
                "note": "Replay decision-pass evidence is positive for current tickers, but live/closed Options Agent outcomes are still missing.",
            },
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "BA",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "status": "WARN",
                "actual_support_status": "PASS",
                "actual_support_scope": "actual_route",
                "actual_support_sample_size": 42,
                "actual_support_avg_pnl": 23.81,
                "actual_support_profit_factor": 1.322,
                "replay_bucket_status": "WARN",
                "route_replay_status": "PASS",
                "route_replay_sample_size": 65,
            }
        ]
    )
    bucket_atlas = pd.DataFrame(
        [
            {
                "bucket_key": "bull_call_debit|DEBIT|bullish|risk_on|dte_31_60|debit_reward_risk_mid|liquidity_deep",
                "status": "WARN",
                "actual_support_status": "PASS",
                "replay_bucket_status": "WARN",
                "current_ticket_count": 1,
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "BLOCK", "detail": "ready_to_enter_rows=0"}]),
        expectancy,
        pd.DataFrame(),
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
        profitability_bucket_atlas=bucket_atlas,
    )
    profitability = audit[audit["metric"].eq("profitability_confidence_rating")].iloc[0]

    assert "actual_closed_or_forward_outcomes_not_positive" not in profitability["blockers"]
    assert "actual_closed_route_or_broad_outcomes=PASS route_or_broad_rows=1" in profitability["evidence"]
    assert "actual_bucket_precision_gap" in profitability["blockers"]
    assert profitability["rating"] == 7.0
    assert profitability["status"] == "PASS"
    assert "leakage_safe_replay_decision_pass_model_strong=PASS" in profitability["evidence"]


def test_confidence_audit_blocks_goal_when_green_row_lacks_profitability_calibration() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "WMT",
                "ready_to_enter": True,
                "target_order_status": "target_order_candidate",
                "live_validation_status": "PASS",
                "entry_limit": 1.25,
                "suggested_contracts": 2,
                "execution_confidence_rating": "HIGH",
                "trade_quality_confidence_rating": "HIGH",
                "actual_forward_strategy_expectancy_status": "PASS",
                "actual_forward_strategy_expectancy_sample_size": 5,
                "profitability_calibration_status": "BLOCK",
            }
        ]
    )
    expectancy = pd.DataFrame(
        [
            {
                "source": "schwab_closed_trades",
                "evidence_type": "actual_closed_trades",
                "status": "PASS",
                "sample_size": 35,
                "win_rate": 0.60,
                "avg_pnl": 100.0,
                "total_pnl": 3500.0,
                "profit_factor": 1.5,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "codexuw_replay_decision_pass",
                "evidence_type": "replay_backtest_decision_pass",
                "status": "PASS",
                "sample_size": 40,
                "win_rate": 0.70,
                "avg_pnl": 80.0,
                "total_pnl": 3200.0,
                "profit_factor": 2.0,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "schwab_closed_trades_strategy_cohort",
                "evidence_type": "actual_closed_trades_strategy_cohort",
                "status": "PASS",
                "sample_size": 34,
                "win_rate": 0.6471,
                "avg_pnl": 92.09,
                "total_pnl": 3131.0,
                "profit_factor": 1.823,
                "matched_current_tickers": "",
                "matched_current_count": 0,
            },
            {
                "source": "schwab_closed_trades_by_ticker_strategy",
                "evidence_type": "actual_closed_trades_by_ticker_strategy",
                "status": "PASS",
                "sample_size": 5,
                "win_rate": 0.60,
                "avg_pnl": 75.0,
                "total_pnl": 375.0,
                "profit_factor": 1.4,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
            },
            {
                "source": "expectancy_summary",
                "evidence_type": "summary",
                "status": "PASS",
                "sample_size": 109,
                "matched_current_tickers": "WMT",
                "matched_current_count": 1,
                "note": "Actual closed/forward outcomes and replay decision-pass evidence are positive.",
            },
        ]
    )
    monthly = pd.DataFrame(
        [
            {"metric": "ready_ticket_count", "value": 1, "status": "PASS", "note": "one green"},
            {"metric": "one_cycle_max_profit", "value": 1000, "status": "PASS", "note": "capacity"},
            {"metric": "cycles_needed_at_max_profit", "value": 4, "status": "PASS", "note": "capacity"},
            {"metric": "expectancy_evidence", "value": 109, "status": "PASS", "note": "positive"},
            {"metric": "ready_ticket_expectancy_evidence", "value": 1, "status": "PASS", "note": "supported"},
        ]
    )
    calibration = pd.DataFrame(
        [
            {
                "scope": "current_trade_calibration",
                "ticker": "WMT",
                "strategy_route": "bull_call_debit",
                "status": "BLOCK",
            }
        ]
    )

    audit = core.build_confidence_audit(
        pd.DataFrame(),
        tickets,
        pd.DataFrame([{"gate": "ready_trade_tickets", "status": "PASS", "detail": "ready_to_enter_rows=1"}]),
        expectancy,
        monthly,
        {"fresh_live_quotes_ready": True, "portfolio_ready": True, "agentic_reviews_ready": True},
        profitability_calibration=calibration,
    )
    summary = core.summarize_confidence_audit(audit)

    assert audit.loc[audit["metric"].eq("goal_confidence_gate"), "status"].tolist() == ["BLOCK"]
    assert "profitability_calibration_not_proven" in summary["blockers"]
    assert "green_profitability_calibration_not_all_pass" in summary["blockers"]
    assert summary["profitability_confidence_rating"] < core.MIN_GOAL_PROFITABILITY_CONFIDENCE_RATING
    assert summary["order_entry_confidence_rating"] < core.MIN_GOAL_ORDER_ENTRY_CONFIDENCE_RATING


def test_calibrated_order_entry_blocker_summary_names_remaining_blockers() -> None:
    decision = pd.DataFrame(
        [
            {
                "ticker": "BX",
                "strategy_route": "short_put",
                "trade_plan": "SELL 1 BX 2026-07-17 115 Put @ 3.60 CREDIT",
                "ready_to_enter": False,
                "profitability_calibration_status": "PASS",
                "execution_status": "waiting_for_price",
                "target_order_status": "target_order_candidate",
                "execution_blockers": "goal_confidence_gate_blocked",
                "entry_limit": 3.60,
                "suggested_contracts": 1,
            },
            {
                "ticker": "VRT",
                "strategy_route": "short_put",
                "trade_plan": "SELL 1 VRT 2026-07-17 280 Put @ 15.15 CREDIT",
                "ready_to_enter": False,
                "profitability_calibration_status": "PASS",
                "execution_status": "needs_confidence",
                "target_order_status": "target_order_candidate",
                "execution_blockers": "send_now_credit_width_below_30pct",
                "entry_limit": 15.15,
                "suggested_contracts": 1,
            },
            {
                "ticker": "OK",
                "strategy_route": "short_put",
                "ready_to_enter": True,
                "profitability_calibration_status": "PASS",
                "execution_blockers": "",
            },
            {
                "ticker": "WARN",
                "ready_to_enter": False,
                "profitability_calibration_status": "WARN",
                "execution_blockers": "profitability_calibration_required_for_green",
            },
        ]
    )

    summary = core.summarize_calibrated_order_entry_blockers(decision)

    assert summary["calibrated_rows"] == 3
    assert summary["ready_rows"] == 1
    assert summary["blocked_rows"] == 2
    assert summary["blocker_counts"] == {
        "goal_confidence_gate_blocked": 1,
        "send_now_credit_width_below_30pct": 1,
    }
    assert summary["examples"][0]["ticker"] == "BX"
    assert "goal_confidence_gate_blocked" in core._calibrated_order_entry_blocker_detail(summary)


def test_verified_event_overrides_fix_ups_earnings_and_hd_macro_crossing() -> None:
    calendar = core.load_options_event_calendar(core.project_root())
    contracts = pd.DataFrame(
        [
            {
                "ticker": "UPS",
                "expiry": "2026-08-21",
                "dte": 43,
                "days_to_earnings": 26,
                "status_reason": "fixture",
            },
            {
                "ticker": "HD",
                "expiry": "2026-07-17",
                "dte": 8,
                "days_to_earnings": 40,
                "status_reason": "fixture",
            },
        ]
    )

    annotated = core.annotate_contract_event_risk(
        contracts,
        as_of="2026-07-09",
        event_calendar=calendar,
    ).set_index("ticker")

    assert annotated.loc["UPS", "earnings_event_date"] == "2026-07-28"
    assert bool(annotated.loc["UPS", "earnings_before_expiry"])
    assert annotated.loc["UPS", "event_exit_deadline"] == "2026-07-27"
    assert "investors.ups.com" in annotated.loc["UPS", "earnings_event_source"]
    assert not bool(annotated.loc["HD", "earnings_before_expiry"])
    assert annotated.loc["HD", "macro_event_count_before_expiry"] == 2
    assert "CPI 2026-07-14" in annotated.loc["HD", "macro_events_before_expiry"]
    assert "PPI 2026-07-15" in annotated.loc["HD", "macro_events_before_expiry"]
    assert "short-DTE contract crosses" in annotated.loc["HD", "contract_event_risk_note"]
    assert calendar["status"] == "verified"
    assert any(event["event"] == "FOMC decision" and event["date"] == "2026-07-29" for event in calendar["macro_events"])
    corporate = {event["ticker"]: event for event in calendar["corporate_events"]}
    assert corporate["IBM"]["date"] == "2026-07-22"
    assert corporate["IBM"]["source"] == "https://www.ibm.com/investor/events"
    assert corporate["INTC"]["date"] == "2026-07-23"
    assert corporate["INTC"]["source"].startswith("https://www.intc.com/")
    assert corporate["MSFT"]["date"] == "2026-07-29"
    assert corporate["MSFT"]["source"].startswith("https://www.microsoft.com/")


def test_agent_review_cannot_bypass_unverified_equity_earnings() -> None:
    row = {
        "ticker": "AAPL",
        "recommendation_status": RecommendationStatus.ENTER.value,
        "strategy_route": "long_call",
        "dte": 30,
        "earnings_source_status": "unverified",
        "earnings_before_expiry": False,
        "macro_calendar_status": "verified",
        "macro_event_count_before_expiry": 0,
        "contract_review_status": "BLOCK",
        "contract_review_agents": "structure_builder; skeptic",
        "live_probability_proxy": 0.55,
        "live_quote_width_pct": 0.05,
        "live_theta_burn_pct": 0.01,
        "live_breakeven_expected_move_ratio": 0.50,
    }

    blocked = core._send_now_economics_blockers(
        row,
        ticket="BUY 1 AAPL 2026-08-21 200 Call @ 5.00 DEBIT",
        entry_limit=5.0,
    )
    agent_reviewed = core._send_now_economics_blockers(
        {
            **row,
            "contract_review_status": "PASS",
            "contract_review_agents": "catalyst_news; structure_builder; skeptic",
        },
        ticket="BUY 1 AAPL 2026-08-21 200 Call @ 5.00 DEBIT",
        entry_limit=5.0,
    )

    assert "send_now_earnings_calendar_unverified" in blocked
    assert "send_now_earnings_calendar_unverified" in agent_reviewed


def test_unverified_macro_calendar_blocks_long_dte_send_now() -> None:
    row = {
        "ticker": "AAPL",
        "recommendation_status": RecommendationStatus.ENTER.value,
        "strategy_route": "long_call",
        "dte": 45,
        "earnings_source_status": "verified",
        "earnings_before_expiry": False,
        "macro_calendar_status": "unverified",
        "macro_event_count_before_expiry": 0,
        "live_probability_proxy": 0.55,
        "live_quote_width_pct": 0.05,
        "live_theta_burn_pct": 0.01,
        "live_breakeven_expected_move_ratio": 0.50,
    }

    blockers = core._send_now_economics_blockers(
        row,
        ticket="BUY 1 AAPL 2026-08-21 200 Call @ 5.00 DEBIT",
        entry_limit=5.0,
    )

    assert "send_now_macro_calendar_unverified" in blockers


def test_hd_like_short_dte_debit_is_rejected_for_quote_width_and_theta() -> None:
    row = {
        "ticker": "HD",
        "signal_premium": 5_000_000.0,
        "combined_flow_bias": 0.50,
        "underlying_quality_tier": "core",
        "quality_status": "qualified",
    }
    live = {
        "debit": 2.40,
        "mid_debit": 2.18,
        "natural_debit": 2.58,
        "target_entry": 2.50,
        "spread_width": 10.0,
        "long_strike": 345.0,
        "short_strike": 355.0,
        "long_leg": "HD  260717C00345000",
        "short_leg": "HD  260717C00355000",
        "long_delta": 0.327,
        "short_delta": 0.126,
        "long_theta": -0.315,
        "short_theta": -0.173,
        "net_theta": -0.142,
        "theta_burn_pct": 0.0592,
        "quote_width_pct": 0.345,
        "expected_move_pct": 0.035,
        "breakeven_expected_move_ratio": 0.73,
        "long_oi": 5000,
        "long_volume": 1000,
        "short_oi": 4000,
        "short_volume": 900,
    }

    result = core._apply_live_debit_spread(
        row,
        live,
        direction="Bull Call",
        expiry=dt.date(2026, 7, 17),
        spot=338.73,
        asof_date=dt.date(2026, 7, 9),
    )

    assert result["recommendation_status"] == RecommendationStatus.AVOID.value
    assert "live_quote_width_pct_above_30pct" in result["quality_gate_reason"]
    assert "short_dte_live_quote_width_pct_above_20pct" in result["quality_gate_reason"]
    assert "short_dte_theta_burn_above_3pct_per_day" in result["quality_gate_reason"]
    assert result["live_probability_proxy"] == pytest.approx(0.327)
    assert result["live_net_theta_per_contract"] == pytest.approx(-14.2)
    assert result["live_breakeven_expected_move_ratio"] == pytest.approx(0.73)


def test_live_debit_expected_move_falls_back_to_chain_iv_with_correct_ratio_direction() -> None:
    from codexuw.schwab_live import find_debit_spread_alternatives

    expiry = dt.date(2026, 7, 17)
    contracts = pd.DataFrame(
        [
            {
                "expiry": expiry,
                "right": "C",
                "strike": 345.0,
                "symbol": "HD  260717C00345000",
                "bid": 2.80,
                "ask": 3.30,
                "mark": 3.05,
                "delta": 0.327,
                "theta": -0.315,
                "gamma": 0.02,
                "vega": 0.10,
                "iv": 0.35,
                "open_interest": 5000,
                "volume": 1000,
            },
            {
                "expiry": expiry,
                "right": "C",
                "strike": 355.0,
                "symbol": "HD  260717C00355000",
                "bid": 0.72,
                "ask": 1.02,
                "mark": 0.87,
                "delta": 0.126,
                "theta": -0.173,
                "gamma": 0.01,
                "vega": 0.07,
                "iv": 0.34,
                "open_interest": 4000,
                "volume": 900,
            },
        ]
    )

    alternatives = find_debit_spread_alternatives(
        contracts,
        direction="Bull Call",
        expiry=expiry,
        spot=338.73,
        preferred_width=10.0,
        as_of_date=dt.date(2026, 7, 9),
    )

    assert alternatives
    result = alternatives[0]
    assert result["expected_move_pct"] == pytest.approx(0.345 * (8 / 365) ** 0.5)
    assert result["breakeven_expected_move_ratio"] == pytest.approx(
        result["breakeven_distance_pct"] / result["expected_move_pct"]
    )
    assert result["expected_move_ratio"] == pytest.approx(
        result["expected_move_pct"] / result["breakeven_distance_pct"]
    )


def test_live_debit_selection_prefers_quality_across_expiries() -> None:
    asof = dt.date(2026, 7, 9)
    short_expiry = dt.date(2026, 7, 17)
    quality_expiry = dt.date(2026, 8, 14)
    contracts = pd.DataFrame(
        [
            {"right": "C", "expiry": short_expiry},
            {"right": "C", "expiry": quality_expiry},
        ]
    )
    expiries = core._live_expiry_candidates(contracts, asof, short_expiry, "Bull Call")
    alternatives = [
        {
            "live_status": "PASS",
            "selected_expiry": short_expiry.isoformat(),
            "dte": 8,
            "spread_width": 1.0,
            "debit": 0.30,
            "reward_risk": 2.33,
            "target_entry": 0.45,
            "long_delta": 0.27,
            "theta_burn_pct": 0.05,
            "breakeven_expected_move_ratio": 0.70,
            "quote_width_pct": 0.05,
            "short_oi": 1_000,
            "short_volume": 100,
            "long_oi": 1_000,
            "long_volume": 100,
            "liq_score": 1_100,
        },
        {
            "live_status": "PASS",
            "selected_expiry": quality_expiry.isoformat(),
            "dte": 36,
            "spread_width": 1.0,
            "debit": 0.40,
            "reward_risk": 1.50,
            "target_entry": 0.45,
            "long_delta": 0.52,
            "theta_burn_pct": 0.01,
            "breakeven_expected_move_ratio": 0.55,
            "quote_width_pct": 0.08,
            "short_oi": 800,
            "short_volume": 100,
            "long_oi": 800,
            "long_volume": 100,
            "liq_score": 900,
        },
    ]

    selected = core._select_live_alternative(
        {"signal_premium": 1_000_000.0, "combined_flow_bias": 0.30},
        alternatives,
        entry_type="DEBIT",
    )

    assert expiries == [quality_expiry, short_expiry]
    assert selected["selected_expiry"] == quality_expiry.isoformat()


def test_live_long_option_selection_prefers_probability_and_theta_over_lowest_debit() -> None:
    alternatives = [
        {
            "live_status": "PASS",
            "selected_expiry": "2026-07-17",
            "dte": 8,
            "debit": 2.00,
            "long_delta": 0.31,
            "theta_burn_pct": 0.029,
            "breakeven_distance_pct": 0.04,
            "breakeven_expected_move_ratio": 0.80,
            "quote_width_pct": 0.05,
            "long_oi": 1_000,
            "long_volume": 100,
            "liq_score": 1_100,
        },
        {
            "live_status": "PASS",
            "selected_expiry": "2026-08-14",
            "dte": 36,
            "debit": 3.00,
            "long_delta": 0.52,
            "theta_burn_pct": 0.01,
            "breakeven_distance_pct": 0.03,
            "breakeven_expected_move_ratio": 0.55,
            "quote_width_pct": 0.08,
            "long_oi": 800,
            "long_volume": 100,
            "liq_score": 900,
        },
    ]

    selected = core._select_live_long_option_alternative(
        {"signal_premium": 1_000_000.0, "combined_flow_bias": 0.30},
        alternatives,
    )

    assert selected["selected_expiry"] == "2026-08-14"


def test_live_credit_expected_move_uses_breakeven_not_short_strike() -> None:
    from codexuw.schwab_live import find_credit_spread_alternatives

    expiry = dt.date(2026, 8, 21)
    contracts = pd.DataFrame(
        [
            {
                "expiry": expiry,
                "right": "P",
                "strike": 195.0,
                "symbol": "AAPL  260821P00195000",
                "bid": 3.00,
                "ask": 3.20,
                "mark": 3.10,
                "delta": -0.27,
                "theta": -0.08,
                "iv": 0.30,
                "open_interest": 2_000,
                "volume": 300,
            },
            {
                "expiry": expiry,
                "right": "P",
                "strike": 190.0,
                "symbol": "AAPL  260821P00190000",
                "bid": 1.50,
                "ask": 1.70,
                "mark": 1.60,
                "delta": -0.16,
                "theta": -0.04,
                "iv": 0.32,
                "open_interest": 1_500,
                "volume": 200,
            },
        ]
    )

    alternatives = find_credit_spread_alternatives(
        contracts,
        direction="Bull Put",
        expiry=expiry,
        spot=205.0,
        preferred_width=5.0,
        as_of_date=dt.date(2026, 7, 9),
    )

    assert alternatives
    result = alternatives[0]
    expected_breakeven = result["short_strike"] - result["credit"]
    expected_move = 0.31 * ((expiry - dt.date(2026, 7, 9)).days / 365) ** 0.5
    expected_ratio = abs(expected_breakeven - 205.0) / 205.0 / expected_move
    assert result["breakeven"] == pytest.approx(expected_breakeven)
    assert result["expected_move_pct"] == pytest.approx(expected_move)
    assert result["breakeven_expected_move_ratio"] == pytest.approx(expected_ratio)
    assert result["expected_move_ratio"] == pytest.approx(expected_ratio)


def test_ups_like_route_support_stays_yellow_and_is_capped_at_one_contract() -> None:
    trade = pd.DataFrame(
        [
            {
                "ticker": "UPS",
                "strategy_route": "bull_call_debit",
                "strategy_family": "vertical_spread",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "quality_status": "qualified",
                "structure": "bull call debit spread",
                "trade_plan": "BUY 1 UPS 2026-08-21 115 Call / SELL 1 UPS 2026-08-21 120 Call @ 1.70 DEBIT",
                "full_ticket": "BUY 1 UPS 2026-08-21 115 Call / SELL 1 UPS 2026-08-21 120 Call @ 1.70 DEBIT",
                "expiry": "2026-08-21",
                "dte": 43,
                "buy_leg": "BUY 1 UPS 2026-08-21 115 Call",
                "sell_leg": "SELL 1 UPS 2026-08-21 120 Call",
                "entry_limit": 1.70,
                "suggested_contracts": 5,
                "max_profit": 330.0,
                "max_loss": 170.0,
                "portfolio_total_value": 100_000.0,
                "trade_quality_status": "reviewable",
                "live_validation_status": "PASS",
                "spot_live": 110.74,
                "breakeven": 116.70,
                "live_probability_proxy": 0.377,
                "live_quote_width_pct": 0.10,
                "live_theta_burn_pct": 0.008,
                "live_breakeven_expected_move_ratio": 0.50,
                "agent_support_count": 5,
                "external_agent_review_count": 5,
                "external_agent_distinct_review_count": 5,
                "underlying_quality_tier": "core",
                "contract_review_status": "PASS",
                "actual_forward_expectancy_status": "WARN",
                "actual_forward_expectancy_sample_size": 1,
                "actual_forward_strategy_expectancy_status": "WARN",
                "actual_forward_strategy_expectancy_sample_size": 42,
                "actual_forward_strategy_expectancy_avg_pnl": 25.0,
                "actual_forward_strategy_expectancy_profit_factor": 1.32,
                "profitability_calibration_status": "WARN",
                "profitability_calibration_scope": "actual_route",
                "profitability_calibration_actual_status": "PASS",
                "profitability_calibration_actual_sample_size": 42,
                "profitability_calibration_actual_avg_pnl": 25.0,
                "profitability_calibration_actual_profit_factor": 1.32,
                "profitability_calibration_replay_status": "WARN",
                "profitability_calibration_replay_sample_size": 3,
            }
        ]
    )
    trade = core.annotate_contract_event_risk(
        trade,
        as_of="2026-07-09",
        event_calendar=core.load_options_event_calendar(core.project_root()),
    )
    trade = core.apply_evidence_aware_size_caps(trade)
    context = core.build_execution_context(
        live_schwab=True,
        chain_snapshot_dir=None,
        portfolio_context={"status": "ok", "total_value": 100_000},
        research_task_count=100,
        external_review_count=100,
        external_review_agent_count=5,
        agent_dispatch_task_count=5,
        agent_reviews_json=Path("/tmp/reviews.json"),
        market_session_open=True,
    )

    decision = core.synthesize_decision_board(
        trade,
        market_regime={"regime": "mixed"},
        execution_context=context,
    )

    assert trade["suggested_contracts"].tolist() == [1]
    assert trade["evidence_size_cap"].tolist() == [1]
    assert decision["ready_to_enter"].tolist() == [False]
    assert decision["target_order_status"].tolist() == ["target_order_candidate"]
    assert decision["trade_quality_confidence_rating"].tolist() == ["LOW"]
    blockers = decision["execution_blockers"].iloc[0]
    assert "send_now_earnings_before_expiry" in blockers
    assert "send_now_probability_proxy_below_40pct" in blockers
    assert core.PROFITABILITY_CALIBRATION_BLOCKER in blockers


def test_exact_contract_review_allows_non_blocking_caution_but_rejects_missing_or_avoid() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": "UPS",
                "strategy_route": "bull_call_debit",
                "expiry": "2026-08-21",
                "buy_leg": "BUY 1 UPS 2026-08-21 115 Call",
                "sell_leg": "SELL 1 UPS 2026-08-21 120 Call",
                "trade_plan": "BUY 1 UPS 2026-08-21 115 Call / SELL 1 UPS 2026-08-21 120 Call @ 1.70 DEBIT",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
            }
        ]
    )
    key = core.contract_review_key(priced.iloc[0])
    generic_reviews = pd.DataFrame(
        [
            {
                "ticker": "UPS",
                "agent": agent,
                "agent_type": "subagent",
                "verdict": "supportive",
                "contract_specific": False,
            }
            for agent in ("structure_builder", "skeptic")
        ]
    )

    generic_result = core.apply_agent_reviews(priced, generic_reviews)
    assert generic_result["contract_review_status"].tolist() == ["BLOCK"]
    assert generic_result["contract_review_count"].tolist() == [0]

    caution_reviews = pd.concat(
        [
            generic_reviews,
            pd.DataFrame(
                [
                    {
                        "ticker": "UPS",
                        "agent": "structure_builder",
                        "agent_type": "subagent",
                        "verdict": "supportive",
                        "contract_specific": True,
                        "contract_key": key,
                        "strategy_route": priced.iloc[0]["strategy_route"],
                        "expiry": priced.iloc[0]["expiry"],
                        "trade_plan": priced.iloc[0]["trade_plan"],
                    },
                    {
                        "ticker": "UPS",
                        "agent": "skeptic",
                        "agent_type": "subagent",
                        "verdict": "caution",
                        "contract_specific": True,
                        "contract_key": key,
                        "strategy_route": priced.iloc[0]["strategy_route"],
                        "expiry": priced.iloc[0]["expiry"],
                        "trade_plan": priced.iloc[0]["trade_plan"],
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    caution_result = core.apply_agent_reviews(priced, caution_reviews)
    assert caution_result["contract_review_status"].tolist() == ["PASS"]
    assert "skeptic=caution" in caution_result["contract_review_verdicts"].iloc[0]

    supportive_reviews = caution_reviews.copy()
    exact_skeptic = supportive_reviews["contract_specific"].map(core._truthy) & supportive_reviews["agent"].eq("skeptic")
    supportive_reviews.loc[exact_skeptic, "verdict"] = "supportive"
    supportive_result = core.apply_agent_reviews(priced, supportive_reviews)
    assert supportive_result["contract_review_status"].tolist() == ["PASS"]
    assert supportive_result["contract_review_count"].tolist() == [2]

    improved_debit = priced.copy()
    improved_debit.loc[0, "entry_limit"] = 1.60
    improved_debit.loc[0, "trade_plan"] = (
        "BUY 1 UPS 2026-08-21 115 Call / SELL 1 UPS 2026-08-21 120 Call @ 1.60 DEBIT"
    )
    improved_result = core.apply_agent_reviews(improved_debit, supportive_reviews)
    assert improved_result["contract_review_status"].tolist() == ["PASS"]

    worse_debit = priced.copy()
    worse_debit.loc[0, "entry_limit"] = 1.80
    worse_debit.loc[0, "trade_plan"] = (
        "BUY 1 UPS 2026-08-21 115 Call / SELL 1 UPS 2026-08-21 120 Call @ 1.80 DEBIT"
    )
    worse_result = core.apply_agent_reviews(worse_debit, supportive_reviews)
    assert worse_result["contract_review_status"].tolist() == ["BLOCK"]
    assert worse_result["contract_review_count"].tolist() == [0]

    mechanically_revalidated = worse_debit.assign(
        recommendation_status=RecommendationStatus.ENTER.value,
        trade_quality_status="reviewable",
        quality_gate_reason="",
        hard_rejects="",
        target_entry=1.90,
        dte=43,
        max_profit=320.0,
        max_loss=180.0,
        spot_live=110.0,
        breakeven=116.8,
        live_probability_proxy=0.50,
            live_theta_burn_pct=0.01,
            live_breakeven_expected_move_ratio=0.50,
            earnings_source_status="verified",
            earnings_before_expiry=False,
            macro_calendar_status="verified",
            macro_event_count_before_expiry=0,
        )
    unproven_result = core.apply_agent_reviews(mechanically_revalidated, supportive_reviews)
    assert unproven_result["contract_review_status"].tolist() == ["BLOCK"]

    mechanically_revalidated["construction_source"] = "reviewed_contract_fresh_reprice"
    revalidated_result = core.apply_agent_reviews(mechanically_revalidated, supportive_reviews)
    assert revalidated_result["contract_review_status"].tolist() == ["PASS"]
    assert revalidated_result["contract_review_count"].tolist() == [2]


def test_exact_credit_review_remains_valid_only_at_same_or_better_credit() -> None:
    reviewed_row = {
        "ticker": "AAPL",
        "strategy_route": "bull_put_credit",
        "expiry": "2026-08-21",
        "buy_leg": "BUY 1 AAPL 2026-08-21 195 Put",
        "sell_leg": "SELL 1 AAPL 2026-08-21 200 Put",
        "trade_plan": "SELL 1 AAPL 2026-08-21 200 Put / BUY 1 AAPL 2026-08-21 195 Put @ 1.60 CREDIT",
    }
    review = {
        "ticker": "AAPL",
        "contract_specific": True,
        "contract_key": core.contract_review_key(reviewed_row),
        "strategy_route": "bull_put_credit",
        "expiry": "2026-08-21",
        "trade_plan": reviewed_row["trade_plan"],
    }
    better = {
        **reviewed_row,
        "entry_limit": 1.70,
        "trade_plan": "SELL 1 AAPL 2026-08-21 200 Put / BUY 1 AAPL 2026-08-21 195 Put @ 1.70 CREDIT",
    }
    worse = {
        **reviewed_row,
        "entry_limit": 1.50,
        "trade_plan": "SELL 1 AAPL 2026-08-21 200 Put / BUY 1 AAPL 2026-08-21 195 Put @ 1.50 CREDIT",
    }

    assert core._contract_review_applies_to_row(review, better) is True
    assert core._contract_review_applies_to_row(review, worse) is False


def test_price_independent_exact_blocker_survives_worse_fresh_reprice() -> None:
    reviewed = {
        "ticker": "PEP",
        "strategy_route": "bull_call_debit",
        "expiry": "2026-08-21",
        "buy_leg": "BUY 1 PEP 2026-08-21 135 Call",
        "sell_leg": "SELL 1 PEP 2026-08-21 140 Call",
        "trade_plan": "BUY 1 PEP 2026-08-21 135 Call / SELL 1 PEP 2026-08-21 140 Call @ 2.70 DEBIT",
        "entry_limit": 2.70,
        "recommendation_status": RecommendationStatus.ENTER.value,
        "live_validation_status": "PASS",
        "underlying_quality_tier": "core",
    }
    key = core.contract_review_key(reviewed)
    reviews = pd.DataFrame(
        [
            {
                "ticker": "PEP",
                "agent": agent,
                "agent_type": "subagent",
                "verdict": "avoid",
                "objective_blocker": True,
                "blocker_type": "stale_event",
                "contract_specific": True,
                "contract_key": key,
                "strategy_route": reviewed["strategy_route"],
                "expiry": reviewed["expiry"],
                "trade_plan": reviewed["trade_plan"],
                "note": "pre-earnings exit deadline is stale",
            }
            for agent in core.CONTRACT_REVIEW_REQUIRED_AGENTS
        ]
    )
    repriced = pd.DataFrame(
        [
            {
                **reviewed,
                "entry_limit": 2.90,
                "trade_plan": "BUY 1 PEP 2026-08-21 135 Call / SELL 1 PEP 2026-08-21 140 Call @ 2.90 DEBIT",
            }
        ]
    )

    blocked = core.apply_agent_reviews(repriced, reviews).iloc[0]

    assert blocked["contract_review_status"] == "BLOCK"
    assert blocked["contract_review_count"] == 2
    assert blocked["contract_review_missing_agents"] == ""
    assert blocked["recommendation_status"] == RecommendationStatus.AVOID.value
    assert "external_agent_objective_blocker" in blocked["hard_rejects"]

    price_sensitive_reviews = reviews.copy()
    price_sensitive_reviews["blocker_type"] = "quote_width"
    price_sensitive = core.apply_agent_reviews(repriced, price_sensitive_reviews).iloc[0]
    assert price_sensitive["contract_review_count"] == 0
    assert price_sensitive["recommendation_status"] == RecommendationStatus.ENTER.value


def test_fresh_reprice_keeps_exact_reviewed_spread_legs() -> None:
    expiry = dt.date(2026, 8, 21)
    reviewed = {
        "ticker": "AAPL",
        "strategy_route": "bull_call_debit",
        "strategy_family": "vertical_spread",
        "structure": "bull call debit spread",
        "expiry": expiry.isoformat(),
        "dte": 43,
        "buy_leg": "BUY 1 AAPL 2026-08-21 100 Call",
        "sell_leg": "SELL 1 AAPL 2026-08-21 105 Call",
        "long_leg": "BUY 1 AAPL 2026-08-21 100 Call",
        "short_leg": "SELL 1 AAPL 2026-08-21 105 Call",
        "long_strike": 100.0,
        "short_strike": 105.0,
        "trade_plan": "BUY 1 AAPL 2026-08-21 100 Call / SELL 1 AAPL 2026-08-21 105 Call @ 2.10 DEBIT",
        "entry_limit": 2.10,
        "signal_premium": 1_000_000.0,
        "combined_flow_bias": 0.80,
        "macro_tape_candidate": False,
        "live_expected_move_pct": 0.10,
    }
    contracts = pd.DataFrame(
        [
            {
                "expiry": expiry,
                "right": "C",
                "strike": 100.0,
                "symbol": "AAPL  260821C00100000",
                "bid": 2.80,
                "ask": 3.00,
                "mark": 2.90,
                "delta": 0.55,
                "theta": -0.08,
                "iv": 0.30,
                "open_interest": 800,
                "volume": 200,
            },
            {
                "expiry": expiry,
                "right": "C",
                "strike": 105.0,
                "symbol": "AAPL  260821C00105000",
                "bid": 1.00,
                "ask": 1.20,
                "mark": 1.10,
                "delta": 0.35,
                "theta": -0.04,
                "iv": 0.28,
                "open_interest": 700,
                "volume": 150,
            },
            {
                "expiry": expiry,
                "right": "C",
                "strike": 101.0,
                "symbol": "AAPL  260821C00101000",
                "bid": 2.00,
                "ask": 2.10,
                "mark": 2.05,
                "delta": 0.52,
                "theta": -0.06,
                "iv": 0.29,
                "open_interest": 2_000,
                "volume": 500,
            },
            {
                "expiry": expiry,
                "right": "C",
                "strike": 106.0,
                "symbol": "AAPL  260821C00106000",
                "bid": 1.20,
                "ask": 1.30,
                "mark": 1.25,
                "delta": 0.32,
                "theta": -0.03,
                "iv": 0.27,
                "open_interest": 2_000,
                "volume": 500,
            },
        ]
    )

    updated, error = core._fresh_reprice_reviewed_contract(
        reviewed,
        contracts,
        spot=101.0,
        asof_date=dt.date(2026, 7, 9),
        chain_source="live:test",
    )

    assert error == ""
    assert updated is not None
    assert updated["long_strike"] == pytest.approx(100.0)
    assert updated["short_strike"] == pytest.approx(105.0)
    assert updated["entry_limit"] == pytest.approx(1.98)
    assert core.contract_review_key(updated) == core.contract_review_key(reviewed)
    assert updated["construction_source"] == "reviewed_contract_fresh_reprice"


def test_exact_contract_blocker_does_not_poison_another_contract_for_same_ticker() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "strategy_route": "long_call",
                "expiry": "2026-08-21",
                "buy_leg": "BUY 1 AAPL 2026-08-21 220 Call",
                "sell_leg": "",
                "trade_plan": "BUY 1 AAPL 2026-08-21 220 Call @ 5.00 DEBIT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "status_reason": "long call priced",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
            },
            {
                "ticker": "AAPL",
                "strategy_route": "bull_put_credit",
                "expiry": "2026-08-21",
                "buy_leg": "BUY 1 AAPL 2026-08-21 195 Put",
                "sell_leg": "SELL 1 AAPL 2026-08-21 200 Put",
                "trade_plan": "SELL 1 AAPL 2026-08-21 200 Put / BUY 1 AAPL 2026-08-21 195 Put @ 1.60 CREDIT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "status_reason": "put spread priced",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
            },
        ]
    )
    long_call_key = core.contract_review_key(priced.iloc[0])
    put_spread_key = core.contract_review_key(priced.iloc[1])
    reviews = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "agent": agent,
                "agent_type": "subagent",
                "contract_specific": True,
                "contract_key": long_call_key,
                "strategy_route": "long_call",
                "expiry": "2026-08-21",
                "trade_plan": "BUY 1 AAPL 2026-08-21 220 Call @ 5.00 DEBIT",
                "verdict": "avoid",
                "objective_blocker": True,
                "note": "exact long call is objectively invalid",
            }
            for agent in core.CONTRACT_REVIEW_REQUIRED_AGENTS
        ]
        + [
            {
                "ticker": "AAPL",
                "agent": agent,
                "agent_type": "subagent",
                "contract_specific": True,
                "contract_key": put_spread_key,
                "strategy_route": "bull_put_credit",
                "expiry": "2026-08-21",
                "trade_plan": "SELL 1 AAPL 2026-08-21 200 Put / BUY 1 AAPL 2026-08-21 195 Put @ 1.60 CREDIT",
                "verdict": "supportive",
                "objective_blocker": False,
                "note": "exact put spread is valid",
            }
            for agent in core.CONTRACT_REVIEW_REQUIRED_AGENTS
        ]
    )

    result = core.apply_agent_reviews(priced, reviews)
    long_call = result.loc[result["strategy_route"].eq("long_call")].iloc[0]
    put_spread = result.loc[result["strategy_route"].eq("bull_put_credit")].iloc[0]

    assert long_call["recommendation_status"] == RecommendationStatus.AVOID.value
    assert long_call["contract_review_status"] == "BLOCK"
    assert "external_agent_objective_blocker" in long_call["hard_rejects"]
    assert put_spread["recommendation_status"] == RecommendationStatus.ENTER.value
    assert put_spread["contract_review_status"] == "PASS"
    assert "external_agent_objective_blocker" not in str(put_spread.get("hard_rejects", ""))
    assert "objectively invalid" not in put_spread["status_reason"]


def test_internal_structure_blocker_does_not_poison_another_contract_for_same_ticker() -> None:
    priced = pd.DataFrame(
        [
            {
                "ticker": "HOOD",
                "bias": "bullish",
                "strategy_route": "long_call",
                "expiry": "2026-07-17",
                "buy_leg": "BUY 1 HOOD 2026-07-17 120 Call",
                "sell_leg": "",
                "full_ticket": "BUY 1 HOOD 2026-07-17 120 Call @ 2.71 DEBIT",
                "trade_plan": "BUY 1 HOOD 2026-07-17 120 Call @ 2.71 DEBIT",
                "recommendation_status": RecommendationStatus.AVOID.value,
                "status_reason": "long call rejected",
                "hard_rejects": "short_dte_theta_burn_above_3pct_per_day",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "quality_status": "qualified",
                "score": 70.0,
                "signal_premium": 1_000_000.0,
            },
            {
                "ticker": "HOOD",
                "bias": "bullish",
                "strategy_route": "bull_call_debit",
                "expiry": "2026-07-17",
                "buy_leg": "BUY 1 HOOD 2026-07-17 120 Call",
                "sell_leg": "SELL 1 HOOD 2026-07-17 121 Call",
                "full_ticket": "BUY 1 HOOD 2026-07-17 120 Call / SELL 1 HOOD 2026-07-17 121 Call @ 0.32 DEBIT",
                "trade_plan": "BUY 1 HOOD 2026-07-17 120 Call / SELL 1 HOOD 2026-07-17 121 Call @ 0.32 DEBIT",
                "recommendation_status": RecommendationStatus.ENTER.value,
                "status_reason": "spread passes exact quality checks",
                "hard_rejects": "",
                "live_validation_status": "PASS",
                "underlying_quality_tier": "core",
                "quality_status": "qualified",
                "score": 70.0,
                "signal_premium": 1_000_000.0,
            },
        ]
    )
    reviews = core.build_internal_agent_reviews(
        pd.DataFrame([{"ticker": "HOOD", "bias": "bullish"}]),
        {"regime": "mixed", "status": "ok", "note": "mixed tape"},
        pd.DataFrame(),
        priced,
        as_of="2026-07-09",
    )
    structure_reviews = reviews[reviews["agent"].isin(["structure", "skeptic"])]

    assert structure_reviews["contract_specific"].map(core._truthy).all()
    assert structure_reviews["contract_key"].nunique() == 2

    reviewed = core.apply_agent_reviews(priced, reviews)
    spread = reviewed.loc[reviewed["strategy_route"].eq("bull_call_debit")].iloc[0]
    assert spread["recommendation_status"] == RecommendationStatus.ENTER.value
    assert "external_agent_objective_blocker" not in str(spread.get("hard_rejects", ""))
    assert "short_dte_theta_burn_above_3pct_per_day" not in spread["status_reason"]

    ranked = core.apply_synthesis_ranking(reviewed, reviews, top_trades=2)
    blocker_counts = ranked.set_index("strategy_route")["agent_objective_blocker_count"].to_dict()
    assert blocker_counts["long_call"] == 2
    assert blocker_counts["bull_call_debit"] == 0


def test_dispatch_only_with_snapshot_writes_exact_contract_review_tasks(tmp_path: Path) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        dispatch_only=True,
        chain_snapshot_dir=snapshot_dir,
    )
    manifest = json.loads(paths["manifest"].read_text())
    priced = pd.read_csv(paths["priced_candidates"])
    validation = pd.read_csv(paths["live_chain_validation"])
    structures = pd.read_csv(paths["structure_attempts"])
    tasks = json.loads(paths["contract_review_tasks"].read_text())

    assert not priced.empty
    assert validation["live_validation_status"].tolist() == ["PASS"]
    assert not structures.empty
    assert tasks["contract_count"] == 1
    assert tasks["contracts"][0]["ticker"] == "WMT"
    assert tasks["contracts"][0]["contract_key"]
    assert manifest["row_counts"]["contract_review_tasks"] == 1
    assert paths["dispatch_priced_candidates"].exists()
    assert paths["dispatch_live_chain_validation"].exists()
    assert paths["dispatch_structure_attempts"].exists()
    assert paths["dispatch_strategy_routing_audit"].exists()
    assert paths["dispatch_contract_review_tasks"].exists()


def test_pass_two_reuses_dispatch_contract_and_preserves_exact_reviews(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path
    snapshot_dir = tmp_path / "snapshots"
    _write_minimal_uw_fixture(root)
    _write_wmt_chain_snapshot(snapshot_dir)

    paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        dispatch_only=True,
        chain_snapshot_dir=snapshot_dir,
    )
    task = json.loads(paths["dispatch_contract_review_tasks"].read_text())["contracts"][0]
    reviews = []
    for agent in task["required_review_agents"]:
        reviews.append(
            {
                "ticker": task["ticker"],
                "agent": agent,
                "agent_type": "subagent",
                "verdict": "supportive",
                "confidence": "high",
                "note": f"{agent} reviewed the exact live contract",
                "objective_blocker": False,
                "contract_specific": True,
                "contract_key": task["contract_key"],
                "strategy_route": task["strategy_route"],
                "expiry": task["expiry"],
                "trade_plan": task["trade_plan"],
                "evidence": "https://investor.example.com/earnings" if agent == "catalyst_news" else "priced contract artifacts",
            }
        )
    paths["agentic_reviews"].write_text(json.dumps({"reviews": reviews}), encoding="utf-8")
    external_reviews, review_errors = core.load_external_agent_reviews(paths["agentic_reviews"])
    prior_plan = json.loads(paths["agent_dispatch_plan"].read_text())
    snapshot, snapshot_status = core._load_dispatch_pricing_snapshot(
        paths,
        prior_plan=prior_plan,
        agent_reviews_json=paths["agentic_reviews"],
        expected_reviews_json=paths["agentic_reviews"],
        external_agent_reviews=external_reviews,
        max_snapshot_age_seconds=3600,
    )

    assert review_errors == []
    assert snapshot is not None
    assert snapshot_status == "reused_fresh_exact_contract_snapshot"

    mismatched_reviews = external_reviews.copy()
    mismatched_reviews.loc[mismatched_reviews["contract_specific"].map(core._truthy), "trade_plan"] += " CHANGED"
    mismatched_snapshot, mismatch_status = core._load_dispatch_pricing_snapshot(
        paths,
        prior_plan=prior_plan,
        agent_reviews_json=paths["agentic_reviews"],
        expected_reviews_json=paths["agentic_reviews"],
        external_agent_reviews=mismatched_reviews,
        max_snapshot_age_seconds=3600,
    )

    assert mismatched_snapshot is None
    assert mismatch_status == "dispatch_contract_review_identity_mismatch"

    def unexpected_reprice(*args: object, **kwargs: object) -> object:
        raise AssertionError("pass two must reuse the exact pass-one pricing snapshot")

    monkeypatch.setattr(core, "price_candidates_with_routing_audit", unexpected_reprice)
    monkeypatch.setattr(core, "validate_priced_candidates_live", unexpected_reprice)

    rerun_paths = run_pipeline(
        "2026-05-22",
        root=root,
        top_trades=3,
        chain_snapshot_dir=snapshot_dir,
        agent_reviews_json=paths["agentic_reviews"],
    )
    manifest = json.loads(rerun_paths["manifest"].read_text())
    priced = pd.read_csv(rerun_paths["priced_candidates"], keep_default_na=False)
    reviewed = priced[priced["contract_key"].astype(str).eq(task["contract_key"])]

    assert manifest["agentic_orchestration"]["dispatch_pricing_snapshot_reused"] is True
    assert reviewed["contract_review_count"].tolist() == [len(task["required_review_agents"])]
    assert reviewed["contract_review_status"].tolist() == ["PASS"]
    assert reviewed["contract_review_missing_agents"].tolist() == [""]
    assert not priced["hard_rejects"].astype(str).str.lower().str.contains(r"(?:^|; )nan(?:;|$)").any()

    old_time = dt.datetime.now().timestamp() - 120
    for key in (
        "dispatch_priced_candidates",
        "dispatch_strategy_routing_audit",
        "dispatch_live_chain_validation",
        "dispatch_structure_attempts",
        "dispatch_contract_review_tasks",
    ):
        os.utime(paths[key], (old_time, old_time))
    expired_snapshot, expired_status = core._load_dispatch_pricing_snapshot(
        paths,
        prior_plan=prior_plan,
        agent_reviews_json=paths["agentic_reviews"],
        expected_reviews_json=paths["agentic_reviews"],
        external_agent_reviews=external_reviews,
        max_snapshot_age_seconds=60,
    )

    assert expired_snapshot is None
    assert expired_status == "dispatch_snapshot_expired"
