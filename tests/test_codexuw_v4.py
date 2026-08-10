from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from codexuw.engine import apply_data_quality_gate
from codexuw.daily_v4 import (
    PIPELINE_NAME_V4,
    _compact_decision_candidate_table,
    _default_out_dir,
    _disposition,
    _execution_quote_blocker,
    _expectancy_safe_entry_price,
    _hard_blocker_reason,
    _trade_legs,
    apply_v4_prospective_book_concentration,
    apply_v4_risk_cap,
    apply_v4_professional_dispositions,
    apply_v4_safety_calibration,
    build_v4_opportunity_board,
    build_strategy_generation_coverage,
    build_no_miss_audit,
    build_construction_attempts,
    build_candidate_disposition,
    build_secondary_liquidity_sweep,
    build_suppression_audit,
    build_v4_safety_calibration,
    build_v4_swing_target_tickets,
    build_v4_target_model,
    parse_args,
    run_v4_daily,
    write_v4_outputs,
)
from codexuw.strategy_registry import apply_strategy_registry_gate, build_strategy_registry


ASOF = dt.date(2026, 5, 20)
EXPIRY = "2026-06-19"


def _candidate(**overrides) -> dict:
    row = {
        "ticker": "AAA",
        "sector": "Technology",
        "direction": "Bull Put",
        "strategy": "Bull Put Credit Spread",
        "expiry": EXPIRY,
        "dte": 30,
        "iv30d": 0.42,
        "realized_volatility_30d": 0.30,
        "iv_rank": 45.0,
        "trade_status": "Research",
        "trade_tier": "",
        "trade_status_reason": "credit target miss but thesis remains reviewable",
        "hard_rejects": "",
        "penalties": "credit_below_min_16pct_width",
        "credit": 0.75,
        "mid_credit": 0.75,
        "natural_credit": 0.62,
        "required_entry": 0.90,
        "credit_pct_width": 0.15,
        "spread_width": 5.0,
        "max_profit": 75.0,
        "max_loss": 425.0,
        "short_strike": 100.0,
        "long_strike": 95.0,
        "short_leg": "AAA260529P00100000",
        "long_leg": "AAA260529P00095000",
        "live_status": "PASS",
        "regular_session_quote": True,
        "displayed_entry_size": 10,
        "quote_width_pct": 0.08,
        "flow_quality": "directional",
        "oi_carryover_status": "supportive",
        "oi_carryover_reason": "exact-leg OI supports direction",
        "edge_verdict": "acceptable",
        "replay_ev_verdict": "acceptable",
        "edge_sample_size": 12,
        "edge_profit_factor": 1.30,
        "edge_win_rate": 0.62,
        "edge_avg_pnl": 24.0,
        "edge_match_level": "exact",
        "regime": "downtrend",
        "regime_trend": "downtrend",
        "confirmation_score": 7.2,
        "score": 6.5,
        "catalyst_status": "mixed",
        "catalyst_earnings_date": "2026-08-01",
        "catalyst_earnings_days": 73,
        "target_entry": 0.90,
        "price_annotation": "current credit $0.75 is below target $0.90; show as work-limit",
        "payoff_calibration_status": "PASS",
        "payoff_route_level": "base",
        "payoff_route_key": "base::Credit|Bull Put|downtrend",
        "payoff_minimum_sample_required": 20,
        "payoff_sample_size": 27,
        "payoff_stress_10_win_rate": 0.78,
        "payoff_stress_10_win_rate_lower_bound": 0.60,
        "payoff_stress_10_average_pnl": 24.0,
        "payoff_stress_10_average_win_risk_fraction": 0.25,
        "payoff_stress_10_average_loss_risk_fraction": 0.50,
        "payoff_stress_10_profit_factor": 1.61,
        "payoff_walk_forward_oos_sample": 14,
        "payoff_walk_forward_oos_profit_factor": 1.72,
        "payoff_post_activation_oos_sample": 2,
        "payoff_post_activation_oos_average_pnl": 48.0,
        "payoff_post_activation_oos_profit_factor": 2.0,
        "payoff_entry_pct_width_p25": 0.18,
        "payoff_entry_pct_width_p75": 0.30,
    }
    row.update(overrides)
    return row


def _medium_debit_candidate(**overrides) -> dict:
    row = _candidate(
        ticker="DEBIT",
        direction="Bull Call",
        strategy="Bull Call Debit Spread",
        expiry="2026-06-17",
        dte=28,
        trade_status="Watch",
        trade_tier="Scout",
        trade_status_reason="",
        penalties="",
        hard_rejects="",
        credit=None,
        mid_credit=None,
        natural_credit=None,
        credit_pct_width=None,
        debit=1.00,
        mid_debit=1.00,
        natural_debit=0.95,
        debit_pct_width=0.20,
        required_entry=1.10,
        target_entry=1.10,
        spread_width=5.0,
        max_profit=400.0,
        max_loss=100.0,
        reward_risk=4.0,
        breakeven_distance_pct=0.02,
        long_strike=100.0,
        short_strike=105.0,
        long_leg="DEBIT260617C00100000",
        short_leg="DEBIT260617C00105000",
        expected_move_ratio=1.50,
        combined_flow_bias=0.25,
        flow_quality="directional",
        bot_flow_source_status="bot_eod_loaded",
        regime="uptrend",
        regime_trend="uptrend",
        quote_width_pct=0.10,
        entry_quote_width_pct=0.10,
        iv_rank=40,
        oi_carryover_status="supportive",
        edge_verdict="positive",
        replay_ev_verdict="positive",
        edge_match_level="debit_policy_sleeve",
        edge_sample_size=18,
        edge_profit_factor=1.40,
        edge_win_rate=0.65,
        edge_avg_pnl=20.0,
        catalyst_status="clear",
        catalyst_earnings_date="2026-08-01",
        catalyst_earnings_days=73,
        price_annotation="",
        confidence_calibration_status="PASS",
        confidence_model_tier="strategy_family_validated",
        confidence_calibration_sample_size=185,
        confidence_probability_lower_bound=0.61,
        confidence_calibration_brier=0.23,
        confidence_calibration_baseline_brier=0.25,
        payoff_route_level="base",
        payoff_route_key="base::Debit|Bull Call|uptrend",
    )
    row.update(overrides)
    return row


def test_v4_cli_and_default_output_folders_say_v4(tmp_path: Path) -> None:
    args = parse_args(["run", "--date", "2026-05-20"])

    assert args.command == "run"
    assert args.report_mode == "post-close"
    assert args.index_income_mode == "primary"
    assert args.risk_budget == 0
    assert args.validation_account_value == 0
    assert _default_out_dir(tmp_path, ASOF, "run") == tmp_path / "out" / "codexdaily_v4_2026-05-20"
    assert _default_out_dir(tmp_path, ASOF, "validation") == tmp_path / "out" / "codexdaily_v4_validation_2026-05-20"
    assert _default_out_dir(tmp_path, ASOF, "overlay", dt.date(2026, 5, 21)) == tmp_path / "out" / "codexdaily_v4_overlay_2026-05-20_overlay_2026-05-21"


def test_price_target_miss_stays_visible_as_v4_work_limit_ticket() -> None:
    scored = pd.DataFrame([_candidate()])
    top_flow = pd.DataFrame([{"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"}])

    tickets = build_v4_swing_target_tickets(
        scored=scored,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        top_flow=top_flow,
    )

    assert len(tickets) == 1
    ticket = tickets.iloc[0]
    assert ticket["final disposition"] == "Swing Target / Work Limit"
    assert not str(ticket["display status"]).startswith("🟢")
    assert "NOT AN ORDER" in ticket["manual review instruction"]
    assert ticket["next-session swing entry target"] == ">= $0.90 credit"
    assert "sell AAA 2026-05-29 100P / buy AAA 2026-05-29 95P" in ticket["trade legs"]
    assert "AAA260529" not in ticket["trade legs"]
    assert "credit is 15.0% of $5 width" in ticket["target price methodology"]


def test_negative_edge_credit_cannot_become_v4_swing_target() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                ticker="NOW",
                credit=1.67,
                mid_credit=1.67,
                natural_credit=1.60,
                required_entry=1.40,
                target_entry=1.40,
                confirmation_score=9.0,
                score=7.8,
                replay_ev_verdict="acceptable_secondary_income",
                edge_verdict="thin_sample",
                edge_sample_size=17,
                edge_win_rate=0.5294117647,
                edge_avg_pnl=-33.897,
            )
        ]
    )

    adjusted = apply_v4_professional_dispositions(scored)
    board = build_v4_opportunity_board(adjusted, top_flow=pd.DataFrame())
    tickets = build_v4_swing_target_tickets(
        scored=adjusted,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        top_flow=pd.DataFrame([{"rank": 1, "ticker": "NOW", "net_premium": 2_000_000, "flow_direction": "bullish"}]),
    )

    assert adjusted.iloc[0]["trade_status"] == "Avoid"
    assert "negative_edge_avg_pnl" in adjusted.iloc[0]["v4_direct_disposition_reason"]
    assert not board[board["Ticker"].eq("NOW")]["Status"].astype(str).str.contains("WORK LIMIT|ENTER|SCOUT", regex=True).any()
    assert tickets.empty


def test_v4_hard_event_risk_blocks_target_ticket() -> None:
    row = _candidate(catalyst_earnings_days=0, catalyst_status="caution", penalties="earnings_news_risk")
    scored = pd.DataFrame([row])

    tickets = build_v4_swing_target_tickets(
        scored=scored,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        top_flow=pd.DataFrame(),
    )

    assert tickets.empty
    assert _disposition(pd.Series(row), targetable=True) == "Avoid"
    assert _hard_blocker_reason(pd.Series(row)) == "earnings/event risk invalidates the structure"


def test_no_miss_audit_disposes_top_flow_tickers_without_candidates() -> None:
    top_flow = pd.DataFrame(
        [
            {"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"},
            {"rank": 2, "ticker": "BBB", "net_premium": 1_500_000, "flow_direction": "bearish"},
        ]
    )
    scored = pd.DataFrame([_candidate(ticker="AAA")])
    tickets = build_v4_swing_target_tickets(scored=scored, board=pd.DataFrame(), regime={"trend": "uptrend"}, top_flow=top_flow)
    dispositions = build_candidate_disposition(candidates=scored, scored=scored, top_flow=top_flow, tickets=tickets)
    attempts = build_construction_attempts(scored=scored, top_flow=top_flow, tickets=tickets, portfolio={"status": "ok"})

    audit = build_no_miss_audit(top_flow=top_flow, scored=scored, dispositions=dispositions, attempts=attempts, tickets=tickets)
    missing = audit[audit["ticker"].eq("BBB")].iloc[0]

    assert bool(missing["candidate_generated"]) is False
    assert missing["constructions_attempted"] == 0
    assert missing["final_disposition"] == "Research"
    assert "no candidate generated" in missing["if_not_targetable_exact_reason"]


def test_suppression_audit_flags_no_price_miss_silent_drop() -> None:
    scored = pd.DataFrame([_candidate(), _candidate(ticker="CCC", hard_rejects="no_usable_liquidity")])
    top_flow = pd.DataFrame([{"rank": 1, "ticker": "AAA"}, {"rank": 2, "ticker": "CCC"}])
    tickets = build_v4_swing_target_tickets(scored=scored.iloc[[0]], board=pd.DataFrame(), regime={"trend": "uptrend"}, top_flow=top_flow)
    dispositions = build_candidate_disposition(candidates=scored, scored=scored, top_flow=top_flow, tickets=tickets)

    suppression = build_suppression_audit(dispositions)

    assert not suppression.empty
    assert not suppression["targetable_trade_hidden_by_price_miss"].any()
    assert "CCC" in set(suppression["ticker"])


def test_v4_target_model_uses_swing_tickets_not_only_execute() -> None:
    tickets = pd.DataFrame(
        [
            {"profit target": "$500.00", "max loss": "$1,000.00", "final disposition": "Swing Target / Work Limit", "setup family": "Credit spreads", "expected win rate": "60%"},
            {"profit target": "$300.00", "max loss": "$500.00", "final disposition": "Scout", "setup family": "Debit spreads", "expected win rate": "50%"},
        ]
    )

    model = build_v4_target_model(
        asof=ASOF,
        tickets=tickets,
        portfolio={"status": "ok", "cash": 10_000},
        monthly_profit_target=10_000,
        month_to_date_realized_pnl=1_000,
        open_unrealized_pnl=0,
        risk_budget=5_000,
    )

    assert model["execute_profit_potential"] == 0
    assert model["swing_target_profit_potential_if_filled"] == 800
    assert model["theoretical_target_inventory_fill_adjusted_potential"] > 0
    assert model["realistic_fill_adjusted_target_potential"] == 0
    assert model["required_number_of_target_tickets"] is not None


def test_v4_strategy_slump_mutes_exact_family_to_research() -> None:
    scored = pd.DataFrame([_candidate(trade_status="Execute")])
    ledger = pd.DataFrame(
        [
            {"report_date": "2026-05-18", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": -100},
            {"report_date": "2026-05-17", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": -50},
            {"report_date": "2026-05-16", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": 20},
        ]
    )

    calibration = build_v4_safety_calibration(scored=scored, outcome_ledger=ledger, asof=ASOF)
    adjusted = apply_v4_safety_calibration(scored, calibration)

    assert bool(calibration.iloc[0]["strategy_slump_muted"]) is True
    assert adjusted.iloc[0]["trade_status"] == "Research"
    assert "v4_strategy_slump_muted" in adjusted.iloc[0]["penalties"]
    assert _disposition(adjusted.iloc[0], targetable=True) == "Research"


def test_v4_negative_shadow_ev_downgrades_execute_without_slump() -> None:
    scored = pd.DataFrame([_candidate(trade_status="Execute")])
    ledger = pd.DataFrame(
        [
            {"report_date": "2026-05-18", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": -200},
            {"report_date": "2026-05-17", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": -100},
            {"report_date": "2026-05-16", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": 20},
            {"report_date": "2026-05-15", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": 20},
            {"report_date": "2026-05-14", "ticker": "AAA", "strategy": "Bull Put Credit Spread", "expiry": EXPIRY, "realized_pnl": 20},
        ]
    )

    calibration = build_v4_safety_calibration(scored=scored, outcome_ledger=ledger, asof=ASOF)
    adjusted = apply_v4_safety_calibration(scored, calibration)

    assert calibration.iloc[0]["shadow_backtest_status"] == "negative_ev"
    assert bool(calibration.iloc[0]["strategy_slump_muted"]) is False
    assert adjusted.iloc[0]["trade_status"] == "Research"
    assert "v4_negative_shadow_ev" in adjusted.iloc[0]["penalties"]


def test_v4_target_ticket_includes_gap_risk_and_oco_bracket_for_execute() -> None:
    scored = pd.DataFrame([_candidate(trade_status="Execute", penalties="", trade_status_reason="all live checks pass")])

    tickets = build_v4_swing_target_tickets(
        scored=scored,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        top_flow=pd.DataFrame([{"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"}]),
    )

    assert tickets.iloc[0]["final disposition"] == "Execute"
    assert "Gap +1%" in tickets.iloc[0]["gap-risk plan +/-1% open"]
    assert "OCO" in tickets.iloc[0]["OCO bracket order logic"]
    assert "BUY TO CLOSE" in tickets.iloc[0]["OCO bracket order logic"]


def test_v4_proposed_book_keeps_one_execute_per_sector_and_preserves_visibility() -> None:
    scored = pd.DataFrame(
        [
            _candidate(ticker="BEST", sector="Technology", trade_status="Execute", score=8.5, edge_sample_size=30),
            _candidate(ticker="ALT", sector="Technology", trade_status="Execute", score=7.0, edge_sample_size=20),
            _candidate(ticker="HEALTH", sector="Healthcare", trade_status="Execute", score=7.5, edge_sample_size=25),
        ]
    )

    adjusted = apply_v4_prospective_book_concentration(scored)

    assert adjusted.loc[adjusted["ticker"].eq("BEST"), "trade_status"].iloc[0] == "Execute"
    assert adjusted.loc[adjusted["ticker"].eq("HEALTH"), "trade_status"].iloc[0] == "Execute"
    alt = adjusted.loc[adjusted["ticker"].eq("ALT")].iloc[0]
    assert alt["trade_status"] == "Watch"
    assert alt["trade_tier"] == "work-limit-sector-concentration"
    assert "Proposed-book sector concentration" in alt["trade_status_reason"]


def test_v4_debit_execute_requires_full_policy_evidence() -> None:
    weak = _candidate(
        ticker="DEBIT",
        direction="Bull Call",
        strategy="Bull Call Debit Spread",
        trade_status="Execute",
        penalties="",
        hard_rejects="",
        debit=1.25,
        mid_debit=1.25,
        natural_debit=1.20,
        debit_pct_width=0.25,
        credit=None,
        credit_pct_width=None,
        required_entry=1.30,
        target_entry=1.30,
        max_profit=375.0,
        max_loss=125.0,
        reward_risk=2.5,
        expected_move_ratio=1.5,
        combined_flow_bias=0.20,
        flow_quality="directional",
        bot_flow_source_status="bot_eod_loaded",
        regime_trend="uptrend",
        dte=21,
        quote_width_pct=0.10,
        iv_rank=40,
        oi_carryover_status="supportive",
        edge_sample_size=25,
        edge_profit_factor=0.90,
        edge_avg_pnl=5.0,
        confidence_calibration_status="PASS",
        confidence_model_tier="strategy_family_validated",
        confidence_calibration_sample_size=185,
        confidence_probability_lower_bound=0.61,
        confidence_calibration_brier=0.23,
        confidence_calibration_baseline_brier=0.25,
        payoff_route_level="base",
        payoff_route_key="base::Debit|Bull Call|uptrend",
    )

    adjusted = apply_v4_professional_dispositions(pd.DataFrame([weak]), asof=ASOF)

    assert adjusted["trade_status"].iloc[0] != "Execute"
    assert "debit_edge_pf_below_1.25" in adjusted["v4_direct_disposition_reason"].iloc[0]


def test_v4_validated_medium_debit_sleeve_executes_at_one_contract() -> None:
    adjusted = apply_v4_professional_dispositions(pd.DataFrame([_medium_debit_candidate()]), asof=ASOF)

    assert adjusted["trade_status"].iloc[0] == "Execute"
    assert adjusted["debit_policy_tier"].iloc[0] == "medium"
    assert "Medium Debit" in adjusted["trade_tier"].iloc[0]
    assert adjusted["contracts"].iloc[0] == 1
    assert adjusted["v4_execution_authority"].iloc[0] == "validated_medium_debit_one_lot"

    tickets = build_v4_swing_target_tickets(
        scored=adjusted,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "directional"},
        top_flow=pd.DataFrame([{"rank": 1, "ticker": "DEBIT", "net_premium": 2_000_000, "flow_direction": "bullish"}]),
    )

    assert tickets.iloc[0]["final disposition"] == "Execute"
    assert tickets.iloc[0]["suggested size"] == "1 contract only; Medium debit sleeve"

    capped = apply_v4_professional_dispositions(
        pd.DataFrame([_medium_debit_candidate(), _medium_debit_candidate(ticker="DEBIT2")]),
        asof=ASOF,
    )
    assert capped["trade_status"].eq("Execute").sum() == 1
    assert capped["trade_tier"].eq("work-limit-medium-debit-cap").sum() == 1


def test_v4_medium_debit_route_evidence_sets_safe_limit_without_family_pass() -> None:
    row = _medium_debit_candidate(
        natural_debit=0.50,
        debit=0.50,
        mid_debit=0.49,
        required_entry=2.25,
        target_entry=2.25,
        max_profit=450.0,
        max_loss=50.0,
        reward_risk=9.0,
        edge_win_rate=17 / 26,
        edge_effective_win_rate=(17 + 0.5) / 27,
        payoff_calibration_status="PROBATIONARY",
        confidence_calibration_status="INSUFFICIENT",
        confidence_model_tier="strategy_family_insufficient",
        confidence_probability=2 / 3,
        confidence_probability_lower_bound=0.5583,
    )

    safe_limit = _expectancy_safe_entry_price(row)
    adjusted = apply_v4_professional_dispositions(pd.DataFrame([row]), asof=ASOF)

    assert safe_limit == 1.99
    assert adjusted.iloc[0]["trade_status"] == "Execute"
    assert adjusted.iloc[0]["contracts"] == 1
    assert adjusted.iloc[0]["v4_execution_authority"] == "validated_medium_debit_one_lot"

    ticket = build_v4_swing_target_tickets(
        scored=adjusted,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "directional"},
        top_flow=pd.DataFrame(),
    ).iloc[0]
    assert ticket["next-session swing entry target"] == "<= $0.50 debit; do not chase above $1.99"
    assert ticket["max loss"] == "$50.00"
    assert ticket["profit target"] == "$202.50"
    assert ticket["expected value"] == "$113.66"
    assert ticket["implied profit factor"] == "7.46"
    assert "take-profit SELL TO CLOSE near $2.52 credit" in ticket["OCO bracket order logic"]
    assert "spread value falls near $0.25" in ticket["OCO bracket order logic"]
    assert ticket["payoff evidence"].startswith("VALIDATED MEDIUM-DEBIT ROUTE; n=18")


def test_v4_medium_debit_execute_survives_registry_and_data_quality_gates() -> None:
    row = _medium_debit_candidate(
        natural_debit=0.50,
        debit=0.50,
        mid_debit=0.49,
        required_entry=2.25,
        target_entry=2.25,
        max_profit=450.0,
        max_loss=50.0,
        reward_risk=9.0,
        edge_win_rate=17 / 26,
        edge_effective_win_rate=(17 + 0.5) / 27,
        payoff_calibration_status="PROBATIONARY",
        confidence_calibration_status="INSUFFICIENT",
        confidence_model_tier="strategy_family_insufficient",
        confidence_probability=2 / 3,
        confidence_probability_lower_bound=0.5583,
    )
    registry = build_strategy_registry(
        payoff_summary={"status": "NO_VALIDATED_LANES"},
        payoff_groups=pd.DataFrame(),
        confidence_summary={"family_validation": {"Debit": {"status": "INSUFFICIENT"}}},
    )

    disposed = apply_v4_professional_dispositions(pd.DataFrame([row]), asof=ASOF)
    registered = apply_strategy_registry_gate(disposed, registry)
    gated = apply_data_quality_gate(
        registered,
        {"status": "warning", "critical_blockers": [], "warnings": ["schwab_quotes_available"]},
    )

    assert gated.iloc[0]["trade_status"] == "Execute"
    assert gated.iloc[0]["contracts"] == 1
    assert gated.iloc[0]["strategy_execution_authority"] == "validated_medium_debit_one_lot"


def test_v4_probationary_credit_route_executes_one_contract_pilot() -> None:
    row = _candidate(
        direction="Bear Call",
        strategy="Bear Call Credit Spread",
        regime="uptrend",
        regime_trend="uptrend",
        short_strike=110.0,
        long_strike=115.0,
        short_leg="AAA260619C00110000",
        long_leg="AAA260619C00115000",
        combined_flow_bias=-0.20,
        credit=1.30,
        mid_credit=1.30,
        natural_credit=1.25,
        required_entry=1.25,
        target_entry=1.25,
        credit_pct_width=0.26,
        quote_width_pct=0.10,
        iv_hv_ratio=0.87,
        iv_rank=54.0,
        realized_volatility_30d=0.26,
        payoff_calibration_status="PROBATIONARY",
        payoff_route_level="base",
        payoff_route_key="base::Credit|Bear Call|uptrend",
        payoff_sample_size=29,
        payoff_minimum_sample_required=20,
        payoff_stress_10_win_rate=26 / 29,
        payoff_stress_10_average_pnl=34.0,
        payoff_stress_10_average_win_risk_fraction=0.2146,
        payoff_stress_10_average_loss_risk_fraction=1.0364,
        payoff_stress_10_profit_factor=1.87,
        payoff_walk_forward_oos_sample=10,
        payoff_walk_forward_oos_average_pnl=22.0,
        payoff_walk_forward_oos_profit_factor=1.57,
        payoff_post_activation_oos_sample=0,
        confidence_calibration_status="CONSERVATIVE",
        confidence_model_tier="strategy_family_conservative",
        confidence_calibration_sample_size=139,
        confidence_probability=0.82,
        confidence_probability_lower_bound=0.78,
        confidence_calibration_brier=0.13,
        confidence_calibration_baseline_brier=0.25,
        edge_sample_size=0,
        edge_profit_factor=None,
        edge_avg_pnl=None,
        edge_match_level="unavailable",
        primary_blocker="",
        penalties="",
        hard_rejects="",
        confirmation_score=9.0,
        catalyst_status="mixed",
        oi_carryover_status="supportive",
        live_status="PASS",
    )

    adjusted = apply_v4_professional_dispositions(pd.DataFrame([row]), asof=ASOF)

    assert adjusted.iloc[0]["trade_status"] == "Execute"
    assert adjusted.iloc[0]["trade_tier"] == "Execute V4 Pilot - 1 Contract"
    assert adjusted.iloc[0]["contracts"] == 1
    assert adjusted.iloc[0]["v4_execution_authority"] == "probationary_one_lot"
    assert adjusted.iloc[0]["v4_post_pricing_profit_factor"] >= 1.25

    tickets = build_v4_swing_target_tickets(
        scored=adjusted,
        board=pd.DataFrame(),
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        top_flow=pd.DataFrame(),
    )
    ticket = tickets.iloc[0]
    assert ticket["final disposition"] == "Execute"
    assert ticket["suggested size"] == "1 contract only; probationary evidence cap"
    assert ticket["next-session swing entry target"] == ">= $1.25 credit"
    assert ticket["profit target"] == "$62.50"
    assert "entry Bear Call Credit Spread" in ticket["OCO bracket order logic"]
    assert "at >= $1.25 credit" in ticket["OCO bracket order logic"]
    assert "probationary final-structure EV=" in ticket["why this is worth reviewing tomorrow"]
    assert ticket["expected win rate"] == "78%"
    assert ticket["win-rate basis"] == "confidence lower bound shown; EV uses route 10%-stress outcomes"
    assert "10% fill-stress win=90%" in ticket["payoff evidence"]
    assert ticket["expected value"] != "UNVALIDATED"
    assert ticket["implied profit factor"] != "UNVALIDATED"
    assert _disposition(adjusted.iloc[0], targetable=True) == "Execute"

    capped = apply_v4_professional_dispositions(
        pd.DataFrame([row, {**row, "ticker": "BBB"}]),
        asof=ASOF,
    )
    assert capped["trade_status"].eq("Execute").sum() == 1
    assert capped["trade_tier"].eq("work-limit-probationary-cap").sum() == 1


def test_v4_probationary_base_credit_cannot_authorize_ambiguous_flow() -> None:
    row = _candidate(
        direction="Bear Call",
        strategy="Bear Call Credit Spread",
        regime="uptrend",
        regime_trend="uptrend",
        short_strike=110.0,
        long_strike=115.0,
        short_leg="AAA260619C00110000",
        long_leg="AAA260619C00115000",
        combined_flow_bias=0.01,
        flow_quality="unclear",
        credit=1.30,
        mid_credit=1.30,
        natural_credit=1.25,
        required_entry=1.25,
        target_entry=1.25,
        credit_pct_width=0.26,
        quote_width_pct=0.10,
        payoff_calibration_status="PROBATIONARY",
        payoff_route_level="base",
        payoff_route_key="base::Credit|Bear Call|uptrend",
        payoff_sample_size=29,
        payoff_minimum_sample_required=20,
        payoff_stress_10_win_rate=26 / 29,
        payoff_stress_10_average_pnl=34.0,
        payoff_stress_10_average_win_risk_fraction=0.25,
        payoff_stress_10_average_loss_risk_fraction=0.50,
        payoff_stress_10_profit_factor=1.87,
        payoff_walk_forward_oos_sample=10,
        payoff_walk_forward_oos_average_pnl=22.0,
        payoff_walk_forward_oos_profit_factor=1.57,
        payoff_post_activation_oos_sample=0,
        confidence_calibration_status="CONSERVATIVE",
        confidence_model_tier="strategy_family_conservative",
        confidence_calibration_sample_size=139,
        confidence_probability=0.82,
        confidence_probability_lower_bound=0.78,
        confidence_calibration_brier=0.13,
        confidence_calibration_baseline_brier=0.25,
        edge_sample_size=0,
        edge_profit_factor=None,
        edge_avg_pnl=None,
        edge_match_level="unavailable",
        primary_blocker="no_flow_edge_alignment",
        penalties="no_flow_edge_alignment",
        trade_status_reason="flow_not_directional:unclear;no_flow_edge_alignment;thin_replay_sample:n=0",
        hard_rejects="",
        confirmation_score=8.5,
        catalyst_status="mixed",
        oi_carryover_status="matched_unconfirmed",
        live_status="PASS",
    )

    adjusted = apply_v4_professional_dispositions(pd.DataFrame([row]), asof=ASOF)

    assert adjusted.iloc[0]["trade_status"] == "Watch"
    assert adjusted.iloc[0]["trade_tier"] == "Scout"
    assert "observation-only" in adjusted.iloc[0]["v4_direct_disposition_reason"]


def test_v4_renders_arbitrary_complex_legs() -> None:
    row = {
        "strategy": "Iron Condor",
        "legs_json": __import__("json").dumps(
            [
                {"instrument": "option", "quantity": 1, "symbol": "AAA260619P00090000"},
                {"instrument": "option", "quantity": -1, "symbol": "AAA260619P00095000"},
                {"instrument": "option", "quantity": -1, "symbol": "AAA260619C00105000"},
                {"instrument": "option", "quantity": 1, "symbol": "AAA260619C00110000"},
            ]
        ),
    }

    rendered = _trade_legs(row)

    assert "buy AAA 2026-06-19 90P" in rendered
    assert "sell AAA 2026-06-19 95P" in rendered
    assert "sell AAA 2026-06-19 105C" in rendered
    assert "buy AAA 2026-06-19 110C" in rendered


def test_strategy_generation_coverage_counts_real_rows_not_registry_flags() -> None:
    registry = pd.DataFrame(
        [
            {"strategy_key": "long_call", "display_name": "Long Call", "category": "directional", "live_builder": True, "pipeline_status": "RESEARCH_ONLY", "execution_authorized": False, "status_reason": "research"},
            {"strategy_key": "iron_condor", "display_name": "Iron Condor", "category": "range", "live_builder": True, "pipeline_status": "RESEARCH_ONLY", "execution_authorized": False, "status_reason": "research"},
        ]
    )
    candidates = pd.DataFrame(
        [{"ticker": "AAA", "sector": "Technology", "strategy_registry_key": "long_call"}, {"ticker": "BBB", "sector": "Energy", "strategy_registry_key": "iron_condor"}]
    )
    scored = pd.DataFrame(
        [
            {"ticker": "AAA", "sector": "Technology", "strategy_registry_key": "long_call", "live_status": "PASS", "trade_status": "Research"},
            {"ticker": "BBB", "sector": "Energy", "strategy_registry_key": "iron_condor", "live_status": "no_realistic_structure", "trade_status": "Research"},
        ]
    )

    coverage = build_strategy_generation_coverage(
        candidates=candidates,
        scored=scored,
        registry=registry,
    ).set_index("strategy_key")

    assert coverage.loc["long_call", "generation_status"] == "LIVE_CONSTRUCTED"
    assert coverage.loc["long_call", "seed_sectors"] == 1
    assert coverage.loc["long_call", "constructed_sectors"] == 1
    assert coverage.loc["iron_condor", "generation_status"] == "SEEDED_NO_LIVE_STRUCTURE"


def test_v4_preopen_zero_size_option_quote_cannot_execute() -> None:
    row = _candidate(
        credit=1.30,
        mid_credit=1.30,
        natural_credit=1.25,
        required_entry=0.90,
        target_entry=0.90,
        credit_pct_width=0.26,
        penalties="",
        regular_session_quote=False,
        displayed_entry_size=0,
    )

    adjusted = apply_v4_professional_dispositions(pd.DataFrame([row]), asof=ASOF)

    assert adjusted.iloc[0]["trade_status"] == "Watch"
    assert adjusted.iloc[0]["trade_tier"] == "Scout"
    assert "outside the regular options session" in adjusted.iloc[0]["v4_direct_disposition_reason"]


def test_v4_closed_session_untraded_legs_cannot_execute() -> None:
    row = _candidate(
        credit=1.30,
        mid_credit=1.30,
        natural_credit=1.25,
        required_entry=0.90,
        target_entry=0.90,
        credit_pct_width=0.26,
        penalties="",
        regular_session_quote=True,
        displayed_entry_size=10,
        market_session_open_at_validation=False,
    )

    adjusted = apply_v4_professional_dispositions(pd.DataFrame([row]), asof=ASOF)

    assert adjusted.iloc[0]["trade_status"] == "Watch"
    assert "no leg traded in the quoted session" in adjusted.iloc[0]["v4_direct_disposition_reason"]


def test_v4_closed_session_traded_legs_clear_the_quote_gate() -> None:
    """An EOD planning run must not be blocked purely by the wall clock."""
    row = _candidate(
        regular_session_quote=True,
        displayed_entry_size=0,
        market_session_open_at_validation=False,
        short_volume=140,
        long_volume=95,
    )

    assert _execution_quote_blocker(row) == ""


def test_v4_open_session_requires_displayed_size() -> None:
    row = _candidate(
        regular_session_quote=True,
        displayed_entry_size=0,
        market_session_open_at_validation=True,
        short_volume=140,
        long_volume=95,
    )

    assert "no displayed entry size" in _execution_quote_blocker(row)


def test_v4_open_session_accepts_displayed_size_when_contract_timestamp_is_missing() -> None:
    row = _candidate(
        regular_session_quote=False,
        displayed_entry_size=8,
        market_session_open_at_validation=True,
        short_volume=0,
        long_volume=0,
    )

    assert _execution_quote_blocker(row) == ""


def test_v4_validated_structural_credit_route_replaces_sparse_exact_edge_only() -> None:
    row = _candidate(
        ticker="RANGE_CALL",
        direction="Bear Call",
        strategy="Bear Call Credit Spread",
        regime="range",
        regime_trend="range",
        expiry="2026-08-21",
        dte=32,
        short_strike=110.0,
        long_strike=115.0,
        short_leg="RANGE_CALL260821C00110000",
        long_leg="RANGE_CALL260821C00115000",
        credit=1.30,
        mid_credit=1.32,
        natural_credit=1.25,
        credit_pct_width=0.26,
        required_entry=1.25,
        target_entry=1.25,
        penalties="thin_replay_sample",
        edge_match_level="unavailable",
        edge_sample_size=0,
        edge_profit_factor=float("nan"),
        edge_avg_pnl=float("nan"),
        # bearish bias on a Bear Call is aligned flow (align = bias * -1 = +0.20)
        combined_flow_bias=-0.20,
        flow_quality="directional",
        oi_carryover_status="supportive",
        catalyst_status="clear",
        catalyst_earnings_date="2026-09-30",
        catalyst_earnings_days=100,
        distance_pct=0.08,
        expected_move_ratio=1.50,
        payoff_route_level="flow_cost",
        payoff_route_key="flow_cost::Credit|Bear Call|range|flow=directional|cost=18to30",
        payoff_minimum_sample_required=12,
        payoff_sample_size=17,
        payoff_stress_10_win_rate=0.882,
        payoff_stress_10_average_pnl=35.85,
        payoff_stress_10_average_win_risk_fraction=0.20,
        payoff_stress_10_average_loss_risk_fraction=0.40,
        payoff_stress_10_profit_factor=2.36,
        payoff_walk_forward_oos_sample=8,
        payoff_walk_forward_oos_profit_factor=float("inf"),
        payoff_post_activation_oos_sample=3,
        payoff_post_activation_oos_average_pnl=53.15,
        payoff_post_activation_oos_profit_factor=float("inf"),
        confidence_calibration_status="PASS",
        confidence_model_tier="strategy_family_validated",
        confidence_calibration_sample_size=185,
        confidence_probability_lower_bound=0.61,
        confidence_calibration_brier=0.23,
        confidence_calibration_baseline_brier=0.25,
    )

    adjusted = apply_v4_professional_dispositions(pd.DataFrame([row]), asof=ASOF)

    assert adjusted.iloc[0]["trade_status"] == "Execute"


def test_v4_validated_flow_cost_route_uses_empirical_entry_floor() -> None:
    row = _candidate(
        ticker="RANGE_CALL",
        direction="Bear Call",
        strategy="Bear Call Credit Spread",
        regime="range",
        regime_trend="range",
        expiry="2026-08-21",
        dte=32,
        short_strike=110.0,
        long_strike=115.0,
        credit=1.10,
        mid_credit=1.12,
        natural_credit=1.10,
        credit_pct_width=0.22,
        max_profit=110.0,
        max_loss=390.0,
        required_entry=1.25,
        target_entry=1.25,
        penalties="thin_replay_sample",
        edge_match_level="unavailable",
        edge_sample_size=0,
        edge_profit_factor=float("nan"),
        edge_avg_pnl=float("nan"),
        # bearish bias on a Bear Call is aligned flow (align = bias * -1 = +0.20)
        combined_flow_bias=-0.20,
        flow_quality="directional",
        oi_carryover_status="supportive",
        catalyst_status="clear",
        catalyst_earnings_date="2026-09-30",
        catalyst_earnings_days=133,
        distance_pct=0.08,
        expected_move_ratio=1.50,
        payoff_route_level="flow_cost",
        payoff_route_key="flow_cost::Credit|Bear Call|range|flow=directional|cost=18to30",
        payoff_minimum_sample_required=12,
        payoff_sample_size=17,
        payoff_stress_10_win_rate=0.882,
        payoff_stress_10_average_pnl=35.85,
        payoff_stress_10_average_win_risk_fraction=0.20,
        payoff_stress_10_average_loss_risk_fraction=0.40,
        payoff_stress_10_profit_factor=2.36,
        payoff_walk_forward_oos_sample=8,
        payoff_walk_forward_oos_profit_factor=float("inf"),
        payoff_post_activation_oos_sample=3,
        payoff_post_activation_oos_average_pnl=53.15,
        payoff_post_activation_oos_profit_factor=float("inf"),
        payoff_entry_pct_width_p25=0.18,
        payoff_entry_pct_width_p75=0.22,
        confidence_calibration_status="PASS",
        confidence_model_tier="strategy_family_validated",
        confidence_calibration_sample_size=185,
        confidence_probability_lower_bound=0.61,
        confidence_calibration_brier=0.23,
        confidence_calibration_baseline_brier=0.25,
    )

    adjusted = apply_v4_professional_dispositions(pd.DataFrame([row]), asof=ASOF)

    assert adjusted.iloc[0]["v4_expectancy_safe_entry_price"] == 0.90
    assert adjusted.iloc[0]["trade_status"] == "Execute"

    board = build_v4_opportunity_board(adjusted, top_flow=pd.DataFrame())
    assert board.iloc[0]["Modeled win rate"] == "88%"
    assert board.iloc[0]["Win-rate basis"] == "validated payoff route; 10% fill stress"
    assert "10% fill-stress PF=2.36" in board.iloc[0]["Payoff evidence"]
    assert "route evidence shown separately" in board.iloc[0]["Per-ticket replay edge"]
    assert "PF=" in board.iloc[0]["Post-pricing EV / PF"]
    assert board.iloc[0]["Edge sample size / win rate / avg P/L"] == "exact/ticker match unavailable; validated route evidence reported separately"


def test_v4_medium_debit_sleeve_does_not_generalize_to_range_or_bear_put() -> None:
    rows = pd.DataFrame(
        [
            _medium_debit_candidate(
                ticker="RANGE",
                regime="range",
                regime_trend="range",
                payoff_route_key="base::Debit|Bull Call|range",
            ),
            _medium_debit_candidate(
                ticker="BEAR",
                direction="Bear Put",
                strategy="Bear Put Debit Spread",
                regime="downtrend",
                regime_trend="downtrend",
                combined_flow_bias=-0.25,
            ),
        ]
    )

    adjusted = apply_v4_professional_dispositions(rows, asof=ASOF)

    assert not adjusted["trade_status"].eq("Execute").any()
    reasons = adjusted.set_index("ticker")["v4_direct_disposition_reason"]
    assert "debit quality policy" in reasons["RANGE"]
    assert "realized payoff lane" in reasons["BEAR"]


def test_v4_risk_cap_is_advisory_and_never_downgrades_a_ticket() -> None:
    tickets = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "trade legs": "spread",
                "final disposition": "Swing Target / Work Limit",
                "max loss": "$10,000.00",
                "profit target": "$1,000.00",
                "suggested size": "5 contracts; only if checks pass",
                "blocker before entry": "",
                "manual review instruction": "",
                "safety calibration flags": "",
            },
            {
                "ticker": "BBB",
                "trade legs": "spread",
                "final disposition": "Execute",
                "max loss": "$3,000.00",
                "profit target": "$300.00",
                "suggested size": "1 contract; only if checks pass",
                "blocker before entry": "",
                "manual review instruction": "",
                "safety calibration flags": "",
            },
        ]
    )

    capped, audit = apply_v4_risk_cap(tickets, {"status": "ok", "total_value": 100_000})
    indexed = capped.set_index("ticker")

    assert indexed.loc["AAA", "max loss"] == "$10,000.00"
    assert indexed.loc["AAA", "suggested size"] == "5 contracts; only if checks pass"
    assert indexed.loc["AAA", "final disposition"] == "Swing Target / Work Limit"
    assert indexed.loc["BBB", "final disposition"] == "Execute"
    assert "Sizing note" in indexed.loc["BBB", "safety calibration flags"]
    assert int(audit["risk_capped"].sum()) == 0
    assert set(audit["final disposition before"]) == set(audit["final disposition after"])


def test_v4_secondary_liquidity_sweep_triggers_below_three_candidates() -> None:
    top_flow = pd.DataFrame(
        [
            {"rank": 1, "ticker": "AAA", "net_premium": 1_000_000, "volume_oi_ratio": 0.7, "max_rolling_5m_premium": 800_000},
            {"rank": 2, "ticker": "BBB", "net_premium": 900_000, "volume_oi_ratio": 0.2, "max_rolling_5m_premium": 500_000},
        ]
    )
    velocity = pd.DataFrame([{"ticker": "AAA", "rolling_5m_premium": 800_000, "rolling_15m_premium": 1_500_000, "flow_velocity_signal": True}])
    correlation = pd.DataFrame([{"ticker": "AAA", "benchmark": "SPY", "rolling_correlation": 0.91, "reason": "beta noise"}])

    sweep = build_secondary_liquidity_sweep(
        candidates=pd.DataFrame([_candidate()]),
        scored=pd.DataFrame([_candidate()]),
        top_flow=top_flow,
        flow_velocity=velocity,
        correlation=correlation,
    )

    assert sweep["triggered"].all()
    first = sweep.set_index("ticker").loc["AAA"]
    assert bool(first["relaxed_uw_block_size_filters"]) is True
    assert first["flow_velocity_scan"] == "pass"
    assert bool(first["beta_noise_ignored"]) is True


def test_decision_lane_audit_surfaces_eligible_candidate_and_veto_reason() -> None:
    scored = pd.DataFrame(
        [
            _candidate(
                decision_eligible=True,
                decision_tier="secondary_income",
                payoff_calibration_status="VETO",
                payoff_calibration_reason="10% fill-stress PF 0.87 < 1.25",
                v4_confirmation_status="manual",
                edge_verdict="unavailable",
                edge_sample_size=0,
            )
        ]
    )

    audit = _compact_decision_candidate_table(scored)

    assert len(audit) == 1
    assert audit.iloc[0]["ticker"] == "AAA"
    assert audit.iloc[0]["payoff status"] == "VETO"
    assert audit.iloc[0]["confirmation"] == "manual"
    assert "PF 0.87 < 1.25" in audit.iloc[0]["why not promoted"]


def test_write_v4_outputs_writes_v4_report_order_and_required_artifacts_without_v3_core(tmp_path: Path) -> None:
    base_dir = tmp_path / "2026-05-20"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    scored = pd.DataFrame([_candidate()])
    top_flow = pd.DataFrame([{"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"}])
    args = parse_args(["run", "--date", "2026-05-20", "--out-dir", str(out_dir)])

    manifest = write_v4_outputs(
        out_dir=out_dir,
        base_dir=base_dir,
        asof=ASOF,
        args=args,
        candidates=scored,
        scored=scored,
        board=pd.DataFrame(),
        top_flow=top_flow,
        flow_velocity=pd.DataFrame(),
        correlation=pd.DataFrame(),
        macro=pd.DataFrame(),
        confirmation=pd.DataFrame(),
        data_quality={"status": "ok", "items": []},
        portfolio={"status": "ok", "cash": 25_000, "risk_actions": []},
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        regime_context={"base_regime": {"trend": "uptrend", "volatility": "low", "flow": "weak"}},
        recent_performance={"status": "unavailable"},
        live_outcomes={"status": "unavailable"},
        loss_review={"status": "unavailable"},
        liquidity_summary={"status": "ok"},
    )

    report = Path(manifest["report_path"]).read_text(encoding="utf-8")
    assert manifest["pipeline_name"] == PIPELINE_NAME_V4
    assert manifest["status"] == "ok"
    assert "| Pipeline | Codex Daily V4 |" in report
    ordered = [
        "## First Screen",
        "## Market Insight For Tomorrow",
        "## Swing Target Tickets For Tomorrow",
        "## Decision-Lane Audit",
        "## Portfolio Repair / Open Risk",
        "## $10k/month Target Math",
        "## No-Miss Audit",
        "## Opportunity Board",
        "## Detailed artifacts",
    ]
    positions = [report.index(item) for item in ordered]
    assert positions == sorted(positions)
    for name in [
        "codexdaily_v4_raw_universe_2026-05-20.csv",
        "codexdaily_v4_candidate_disposition_2026-05-20.csv",
        "codexdaily_v4_swing_target_tickets_2026-05-20.csv",
        "codexdaily_v4_suppression_audit_2026-05-20.csv",
        "codexdaily_v4_construction_attempts_2026-05-20.csv",
        "codexdaily_v4_no_miss_audit_2026-05-20.csv",
        "codexdaily_v4_safety_calibration_2026-05-20.csv",
        "codexdaily_v4_confidence_calibration_predictions_2026-05-20.csv",
        "codexdaily_v4_confidence_calibration_summary_2026-05-20.json",
        "codexdaily_v4_risk_cap_audit_2026-05-20.csv",
        "codexdaily_v4_secondary_liquidity_sweep_2026-05-20.csv",
        "codexdaily_v4_strategy_registry_2026-05-20.csv",
        "codexdaily_v4_sector_rotation_2026-05-20.csv",
    ]:
        assert (out_dir / name).exists()


def test_v4_max_final_trades_is_not_a_visibility_cap(tmp_path: Path, monkeypatch) -> None:
    base_dir = tmp_path / "2026-05-20"
    out_dir = tmp_path / "out"
    base_dir.mkdir()
    scored = pd.DataFrame(
        [
                _candidate(ticker="AAA", expiry="2026-06-19", dte=30, trade_status="Execute", penalties="", credit=1.0, mid_credit=1.0, natural_credit=1.0, required_entry=0.9, target_entry=0.9, credit_pct_width=0.25, expected_move_ratio=0.80, iv30d=0.30, realized_volatility_30d=0.20, combined_flow_bias=0.20, bot_flow_source_status="bot_eod_loaded", edge_match_level="exact", edge_sample_size=20, edge_profit_factor=1.50, edge_win_rate=0.95),
                _candidate(ticker="BBB", sector="Healthcare", expiry="2026-06-19", dte=30, trade_status="Execute", penalties="", credit=1.1, mid_credit=1.1, natural_credit=1.1, required_entry=0.9, target_entry=0.9, credit_pct_width=0.26, expected_move_ratio=0.80, iv30d=0.30, realized_volatility_30d=0.20, combined_flow_bias=0.20, bot_flow_source_status="bot_eod_loaded", edge_match_level="exact", edge_sample_size=20, edge_profit_factor=1.50, edge_win_rate=0.95),
        ]
    )
    top_flow = pd.DataFrame(
        [
            {"rank": 1, "ticker": "AAA", "net_premium": 2_000_000, "flow_direction": "bullish"},
            {"rank": 2, "ticker": "BBB", "net_premium": 1_500_000, "flow_direction": "bullish"},
        ]
    )
    args = parse_args(["run", "--date", "2026-05-20", "--out-dir", str(out_dir), "--max-final-trades", "1"])
    monkeypatch.setattr(
        "codexuw.daily_v4.build_default_payoff_calibration",
        lambda **_: ({"status": "TEST_BYPASS"}, pd.DataFrame(), pd.DataFrame()),
    )
    monkeypatch.setattr("codexuw.daily_v4.apply_payoff_calibration", lambda frame, groups: frame.copy())

    manifest = write_v4_outputs(
        out_dir=out_dir,
        base_dir=base_dir,
        asof=ASOF,
        args=args,
        candidates=scored,
        scored=scored,
        board=pd.DataFrame(),
        top_flow=top_flow,
        flow_velocity=pd.DataFrame(),
        correlation=pd.DataFrame(),
        macro=pd.DataFrame(),
        confirmation=pd.DataFrame(),
        data_quality={"status": "ok", "items": []},
        portfolio={"status": "ok", "cash": 25_000, "total_value": 100_000, "risk_actions": []},
        regime={"trend": "uptrend", "volatility": "low", "flow": "weak"},
        regime_context={"base_regime": {"trend": "uptrend", "volatility": "low", "flow": "weak"}},
        recent_performance={"status": "unavailable"},
        live_outcomes={"status": "unavailable"},
        loss_review={"status": "unavailable"},
        liquidity_summary={"status": "ok"},
    )

    assert manifest["opportunity_counts"]["execute"] == 1
    assert manifest["opportunity_counts"]["swing_target_work_limit"] == 1
    assert manifest["visible_signal_policy"]["active_execute_cap"] is None
    assert manifest["visible_signal_policy"]["max_final_trades_arg"] == 1
    assert manifest["visible_signal_policy"]["aggregate_risk_budget_applied"] is False
    assert manifest["target_model"]["aggregate_risk_budget_applied"] is False
    report = Path(manifest["report_path"]).read_text(encoding="utf-8")
    assert "Visible signal cap | none" in report
    assert "Aggregate risk budget | not configured" in report


def test_run_v4_daily_is_not_a_v3_wrapper() -> None:
    names = run_v4_daily.__code__.co_names

    assert "run_v3_daily" not in names
    assert "write_v4_outputs_from_core" not in names


def test_public_disposition_preserves_internal_scout_without_authorizing_execute() -> None:
    from codexuw.daily_v4 import _disposition

    row = {
        "trade_status": "Watch",
        "trade_tier": "Scout",
        "payoff_calibration_status": "INSUFFICIENT",
        "hard_rejects": "",
        "penalties": "",
        "strategy": "Bull Call Debit Spread",
    }

    assert _disposition(row) == "Scout"
