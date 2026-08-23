from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from codexuw.catalysts import load_catalyst_context
from codexuw.daily_v4 import apply_v4_professional_dispositions
from codexuw.engine import (
    apply_confidence_components,
    apply_data_quality_gate,
    apply_confirmation_framework,
    apply_replay_edge_model,
    apply_oi_carryover,
    apply_portfolio_context,
    assign_trade_statuses,
    build_entry_watchlist,
    build_data_quality_status,
    build_target_capital_model,
    classify_flow_quality,
    select_final_trades,
    _compact_action_rows,
    _expected_move_pct,
)
from codexuw.schwab_live import chain_to_contracts, find_credit_spread_alternatives


ASOF = dt.date(2026, 5, 5)
EXPIRY = dt.date(2026, 5, 15)
# Credit spreads have a 28-DTE floor, so they need their own longer expiry.
CREDIT_EXPIRY = dt.date(2026, 6, 4)


def _credit_row(**overrides) -> dict:
    row = {
        "ticker": "AAA",
        "sector": "Technology",
        "direction": "Bull Put",
        "strategy": "Bull Put Credit Spread",
        "regime": "downtrend",
        "regime_trend": "downtrend",
        "strategy_kind": "Credit",
        "expiry": CREDIT_EXPIRY,
        "dte": 30,
        "iv_rank": 45,
        "hard_rejects": "",
        "penalties": "",
        "credit": 1.30,
        "credit_pct_width": 0.26,
        "spread_width": 5.0,
        "max_profit": 130.0,
        "max_loss": 370.0,
        "breakeven": 98.7,
        "distance_pct": 0.05,
        "expected_move_ratio": 0.85,
        "iv30d": 0.24,
        "realized_volatility_30d": 0.16,
        "combined_flow_bias": 0.12,
        "bot_flow_source_status": "bot_eod_loaded",
        "score": 7.5,
        "confidence": "High",
        "live_status": "PASS",
        "regular_session_quote": True,
        "displayed_entry_size": 10,
        "quote_width_pct": 0.15,
        "short_oi": 1000,
        "short_volume": 500,
        "long_oi": 1000,
        "long_volume": 500,
        "short_leg": "AAA260604P00100000",
        "long_leg": "AAA260604P00095000",
        "flow_quality": "directional",
        "flow_quality_reason": "premium bias aligns",
        "oi_carryover_status": "supportive",
        "replay_ev_verdict": "acceptable",
        "edge_sample_size": 10,
        "edge_profit_factor": 1.30,
        "edge_win_rate": 0.65,
        "edge_avg_pnl": 45.0,
        "confirmation_score": 8.0,
        "catalyst_status": "supportive",
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


def _debit_row(**overrides) -> dict:
    row = {
        "ticker": "BBB",
        "sector": "Technology",
        "direction": "Bull Call",
        "strategy": "Bull Call Debit Spread",
        "strategy_kind": "Debit",
        "expiry": EXPIRY,
        "dte": 10,
        "hard_rejects": "",
        "penalties": "",
        "debit": 2.00,
        "debit_pct_width": 0.40,
        "spread_width": 5.0,
        "reward_risk": 1.5,
        "max_profit": 300.0,
        "max_loss": 200.0,
        "breakeven": 102.0,
        "breakeven_distance_pct": 0.02,
        "expected_move_ratio": 1.2,
        "iv_rank": 45,
        "iv30d": 0.28,
        "combined_flow_bias": 0.12,
        "score": 7.5,
        "confidence": "High",
        "live_status": "PASS",
        "quote_width_pct": 0.15,
        "short_oi": 1000,
        "short_volume": 500,
        "long_oi": 1000,
        "long_volume": 500,
        "short_leg": "BBB260515C00105000",
        "long_leg": "BBB260515C00100000",
        "flow_quality": "directional",
        "flow_quality_reason": "premium bias aligns",
        "oi_carryover_status": "supportive",
        "replay_ev_verdict": "acceptable_proxy",
        "edge_sample_size": 10,
        "edge_win_rate": 0.65,
        "edge_avg_pnl": 45.0,
        "confirmation_score": 8.0,
        "catalyst_status": "supportive",
    }
    row.update(overrides)
    return row


def test_catalyst_context_uses_uw_fallback_earnings_date(tmp_path) -> None:
    event_date = ASOF + dt.timedelta(days=20)
    out = load_catalyst_context(
        tmp_path,
        ["CORZ"],
        asof=ASOF,
        fallback_earnings={"CORZ": event_date},
    )

    row = out.iloc[0]
    assert row["catalyst_earnings_date"] == event_date
    assert row["catalyst_resolution"] == "stock_screener"
    assert row["catalyst_status"] == "mixed"


def test_v4_earnings_crossing_expiry_cannot_execute() -> None:
    event_date = ASOF + dt.timedelta(days=5)
    scored = pd.DataFrame(
        [
            _credit_row(
                expiry=ASOF + dt.timedelta(days=20),
                catalyst_status="mixed",
                catalyst_earnings_date=event_date,
                catalyst_earnings_days=5,
                required_entry=1.20,
                mid_credit=1.30,
                natural_credit=1.25,
                edge_verdict="acceptable",
            )
        ]
    )

    out = apply_v4_professional_dispositions(scored, asof=ASOF)

    assert out["trade_status"].iloc[0] == "Avoid"
    assert "on or before expiry" in out["v4_direct_disposition_reason"].iloc[0]


def test_v4_known_earnings_after_expiry_can_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                expiry=ASOF + dt.timedelta(days=30),
                dte=30,
                catalyst_status="mixed",
                catalyst_earnings_date=ASOF + dt.timedelta(days=40),
                catalyst_earnings_days=40,
                required_entry=1.25,
                mid_credit=1.30,
                natural_credit=1.25,
                edge_verdict="acceptable",
                edge_sample_size=12,
                edge_profit_factor=1.30,
                edge_win_rate=0.90,
                edge_match_level="exact",
            )
        ]
    )

    out = apply_v4_professional_dispositions(scored, asof=ASOF)

    assert out["trade_status"].iloc[0] == "Execute"


def test_v4_unknown_single_name_earnings_cannot_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                catalyst_status="unknown",
                required_entry=1.20,
                mid_credit=1.30,
                natural_credit=1.25,
                edge_verdict="acceptable",
            )
        ]
    )

    out = apply_v4_professional_dispositions(scored, asof=ASOF)

    assert out["trade_status"].iloc[0] == "Watch"
    assert "earnings date unresolved" in out["v4_direct_disposition_reason"].iloc[0]


def test_v4_invalid_natural_credit_cannot_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                required_entry=1.20,
                mid_credit=1.30,
                natural_credit=-0.10,
                edge_verdict="acceptable",
            )
        ]
    )

    out = apply_v4_professional_dispositions(scored, asof=ASOF)

    assert out["trade_status"].iloc[0] == "Avoid"
    assert "non-positive" in out["v4_direct_disposition_reason"].iloc[0]


def test_v4_mid_target_without_natural_fill_becomes_work_limit() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                required_entry=1.20,
                mid_credit=1.30,
                natural_credit=1.00,
                edge_verdict="acceptable",
                edge_match_level="exact",
                edge_sample_size=12,
                edge_profit_factor=1.30,
                edge_avg_pnl=45.0,
            )
        ]
    )

    out = apply_v4_professional_dispositions(scored, asof=ASOF)

    assert out["trade_status"].iloc[0] == "Watch"
    assert out["trade_tier"].iloc[0] == "approved-work-limit-price-target"


def test_v4_contrary_exact_leg_oi_cannot_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                required_entry=1.20,
                mid_credit=1.30,
                natural_credit=1.25,
                oi_carryover_status="contrary",
                edge_verdict="acceptable",
            )
        ]
    )

    out = apply_v4_professional_dispositions(scored, asof=ASOF)

    assert out["trade_status"].iloc[0] == "Watch"
    assert "exact-leg OI" in out["v4_direct_disposition_reason"].iloc[0]


def test_v4_only_one_same_ticker_structure_can_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                ticker="MRVL",
                expiry=ASOF + dt.timedelta(days=30),
                dte=30,
                short_strike=230.0,
                long_strike=225.0,
                required_entry=1.25,
                mid_credit=1.30,
                natural_credit=1.25,
                edge_verdict="acceptable",
                edge_sample_size=12,
                edge_profit_factor=1.30,
                edge_win_rate=0.90,
                edge_match_level="exact",
                oi_carryover_status="supportive",
            ),
            _credit_row(
                ticker="MRVL",
                expiry=ASOF + dt.timedelta(days=30),
                dte=30,
                short_strike=232.5,
                long_strike=227.5,
                required_entry=1.25,
                mid_credit=1.35,
                natural_credit=1.25,
                edge_verdict="acceptable",
                edge_sample_size=12,
                edge_profit_factor=1.30,
                edge_win_rate=0.90,
                edge_match_level="exact",
                oi_carryover_status="matched_unconfirmed",
            ),
        ]
    )

    out = apply_v4_professional_dispositions(scored, asof=ASOF)

    assert out["trade_status"].eq("Execute").sum() == 1
    assert out["trade_status"].eq("Watch").sum() == 1
    kept = out[out["trade_status"].eq("Execute")].iloc[0]
    alternative = out[out["trade_status"].eq("Watch")].iloc[0]
    assert kept["oi_carryover_status"] == "supportive"
    assert "do not stack" in alternative["v4_direct_disposition_reason"]


def test_portfolio_exposure_annotates_without_hard_rejecting_valid_trade() -> None:
    scored = pd.DataFrame([_credit_row(ticker="AAA")])
    portfolio = {
        "status": "ok",
        "total_value": 100_000,
        "option_underlyings": ["AAA"],
        "large_equity_exposure": {},
    }

    out = apply_portfolio_context(scored, portfolio)

    assert out["hard_rejects"].iloc[0] == ""
    assert "portfolio_size_cap" not in out.columns or pd.isna(out["portfolio_size_cap"].iloc[0])
    assert "existing option exposure" in out["portfolio_note"].iloc[0]


def test_explicit_portfolio_size_cap_still_caps_execute_size_to_one_lot() -> None:
    scored = pd.DataFrame([_credit_row(ticker="AAA", portfolio_size_cap=1)])
    final = select_final_trades(
        assign_trade_statuses(scored),
        regime={"sizing_stance": "normal"},
        risk_budget=5000,
        recent_performance={"status": "unavailable"},
    )

    assert final["contracts"].tolist() == [1]


def test_large_equity_exposure_warns_without_veto_or_size_cap() -> None:
    scored = pd.DataFrame([_credit_row(ticker="AAA", max_loss=100.0)])
    portfolio = {
        "status": "ok",
        "total_value": 100_000,
        "option_underlyings": [],
        "large_equity_exposure": {"AAA": 6_000},
    }

    annotated = apply_portfolio_context(scored, portfolio)
    final = select_final_trades(
        assign_trade_statuses(annotated),
        regime={"sizing_stance": "normal"},
        risk_budget=5000,
        recent_performance={"status": "unavailable"},
    )

    assert annotated["hard_rejects"].iloc[0] == ""
    assert "portfolio_size_cap" not in annotated.columns or pd.isna(annotated["portfolio_size_cap"].iloc[0])
    assert "execution gate unaffected" in annotated["portfolio_note"].iloc[0]
    assert final["contracts"].tolist() == [11]


def test_generic_portfolio_requirements_are_explicit() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(ticker="NO_SHARES", requires_equity_shares=100),
            _credit_row(ticker="NO_CASH", requires_cash=25_000),
            _credit_row(ticker="MARGIN", requires_margin_model=True),
            _credit_row(ticker="COVERED", requires_equity_shares=100),
        ]
    )
    portfolio = {
        "status": "ok",
        "total_value": 100_000,
        "cash": 10_000,
        "option_underlyings": [],
        "large_equity_exposure": {},
        "equity_shares": {"COVERED": 100},
        "portfolio_income_mode": "existing-core-review",
        "covered_income_allowed_tickers": [],
    }

    annotated = apply_portfolio_context(scored, portfolio).set_index("ticker")

    # Capital requirements are advisory, but unmet requirements must not be labeled PASS.
    for ticker in ("NO_SHARES", "NO_CASH", "MARGIN"):
        assert annotated.loc[ticker, "portfolio_requirements_status"] == "WARN"
    assert annotated.loc["COVERED", "portfolio_requirements_status"] == "PASS"
    assert "requires 100 shares" in annotated.loc["NO_SHARES", "portfolio_requirements_reason"]
    assert "requires $25,000 cash" in annotated.loc["NO_CASH", "portfolio_requirements_reason"]
    assert "portfolio_collateral_or_margin_blocked" not in str(annotated.loc["NO_CASH", "penalties"])


def test_elevated_equity_exposure_does_not_block_additive_execute() -> None:
    scored = pd.DataFrame([_credit_row(ticker="AAA", max_loss=100.0)])
    portfolio = {
        "status": "ok",
        "total_value": 100_000,
        "option_underlyings": [],
        "large_equity_exposure": {"AAA": 9_000},
    }

    annotated = apply_portfolio_context(scored, portfolio)
    out = assign_trade_statuses(annotated)

    assert out["trade_status"].iloc[0] == "Execute"
    assert "portfolio_concentration_additive" not in out["trade_status_reason"].iloc[0]
    assert "execution gate unaffected" in out["portfolio_note"].iloc[0]


def test_risk_budget_scales_size_but_does_not_drop_the_trade() -> None:
    scored = pd.DataFrame([_credit_row(max_loss=4000.0)])

    final = select_final_trades(
        assign_trade_statuses(scored),
        regime={"sizing_stance": "normal"},
        risk_budget=3000,
        recent_performance={"status": "unavailable"},
    )

    assert len(final) == 1
    assert int(final["contracts"].iloc[0]) == 1


def test_target_model_marks_tiny_execute_profit_stretched() -> None:
    final = pd.DataFrame(
        [
            {
                "target_profit_total": 120.0,
                "position_max_loss": 200.0,
            }
        ]
    )

    model = build_target_capital_model(
        asof=dt.date(2026, 5, 18),
        monthly_profit_target=10_000,
        month_to_date_realized_pnl=0,
        risk_budget=3_000,
        risk_config={"max_risk_per_day": 3_000},
        portfolio={"status": "ok", "total_value": 100_000, "cash": 10_000},
        final=final,
    )

    assert model["target_feasibility"] == "stretched"
    assert model["execute_target_profit"] == 120.0
    assert "below required daily pace" in model["binding_constraint"]
    assert model["risk_required_for_daily_target"] == 1800.0
    assert model["risk_gap_for_daily_target"] == -1200.0


def test_target_aware_selection_outputs_target_contribution_columns() -> None:
    scored = pd.DataFrame([_credit_row(max_loss=100.0, credit=1.0, max_profit=100.0)])

    final = select_final_trades(
        assign_trade_statuses(scored),
        regime={"sizing_stance": "normal"},
        risk_budget=5_000,
        recent_performance={"status": "unavailable"},
        risk_config={"monthly_profit_target": 10_000, "max_contracts_per_trade": 3},
    )

    assert final["contracts"].tolist() == [3]
    assert final["target_profit_total"].iloc[0] == 150.0
    assert final["position_max_loss"].iloc[0] == 300.0
    assert final["target_contribution_pct"].iloc[0] == 0.015


def test_selection_stays_one_lot_until_closed_live_outcome_gate_passes() -> None:
    scored = pd.DataFrame([_credit_row(max_loss=100.0, credit=1.25, max_profit=125.0)])

    final = select_final_trades(
        assign_trade_statuses(scored),
        regime={"sizing_stance": "normal"},
        risk_budget=5_000,
        recent_performance={"status": "unavailable"},
        risk_config={
            "monthly_profit_target": 10_000,
            "max_contracts_per_trade": 10,
            "allow_size_up": False,
            "size_up_evidence": "no closed V3 outcomes",
        },
    )

    assert final["contracts"].tolist() == [1]
    assert final["size_up_evidence"].tolist() == ["no closed V3 outcomes"]


def test_zero_max_final_trades_means_uncapped_visibility() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(ticker="AAA", max_loss=100.0, score=8.0),
            _credit_row(ticker="BBB", max_loss=100.0, score=7.8),
        ]
    )

    final = select_final_trades(
        assign_trade_statuses(scored),
        regime={"sizing_stance": "normal"},
        risk_budget=5_000,
        recent_performance={"status": "unavailable"},
        max_final_trades=0,
        risk_config={"risk_mandate": "target-growth", "max_contracts_per_trade": 1},
    )

    assert final["ticker"].tolist() == ["AAA", "BBB"]


def test_target_growth_mandate_can_size_more_than_capital_preservation() -> None:
    scored = pd.DataFrame([_credit_row(max_loss=100.0, credit=1.0, max_profit=100.0)])
    executable = assign_trade_statuses(scored)

    conservative = select_final_trades(
        executable,
        regime={"sizing_stance": "normal"},
        risk_budget=1_000,
        recent_performance={"status": "unavailable"},
        risk_config={"risk_mandate": "capital-preservation", "max_contracts_per_trade": 20},
    )
    target = select_final_trades(
        executable,
        regime={"sizing_stance": "normal"},
        risk_budget=1_000,
        recent_performance={"status": "unavailable"},
        risk_config={"risk_mandate": "target-growth", "max_contracts_per_trade": 20},
    )

    assert conservative["contracts"].tolist() == [2]
    assert target["contracts"].tolist() == [5]
    assert "target-growth mandate" in target["sizing_rationale"].iloc[0]


def test_flow_classifier_labels_directional_hedge_roll_spread_leg_unclear() -> None:
    directional, _ = classify_flow_quality(
        {
            "direction": "Bull Call",
            "combined_flow_bias": 0.10,
            "flow_total_premium": 20_000_000,
            "source_side_bias": "bullish",
        }
    )
    hedge, _ = classify_flow_quality(
        {
            "direction": "Bear Put",
            "combined_flow_bias": -0.12,
            "flow_total_premium": 20_000_000,
            "bot_put_ask_premium": 9_000_000,
        }
    )
    roll, _ = classify_flow_quality(
        {
            "direction": "Bull Put",
            "combined_flow_bias": 0.10,
            "flow_total_premium": 20_000_000,
            "bot_volume_oi_ratio": 1.1,
            "bot_unique_expiries": 1,
        }
    )
    spread_leg, _ = classify_flow_quality(
        {
            "direction": "Bull Put",
            "combined_flow_bias": 0.10,
            "flow_total_premium": 20_000_000,
            "source_multileg_ratio": 0.60,
        }
    )
    unclear, _ = classify_flow_quality(
        {
            "direction": "Bull Put",
            "combined_flow_bias": 0.01,
            "flow_total_premium": 20_000_000,
            "bot_bull_premium": 10_000_000,
            "bot_bear_premium": 9_000_000,
        }
    )

    assert [directional, hedge, roll, spread_leg, unclear] == [
        "directional",
        "hedge",
        "roll",
        "spread_leg",
        "unclear",
    ]


def test_unclear_noisy_flow_routes_to_research_unless_confirmations_are_strong() -> None:
    scored = pd.DataFrame([_credit_row(flow_quality="unclear", confirmation_score=6.0)])

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"


def test_high_confirmation_score_does_not_override_core_thesis_failures() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                flow_quality="unclear",
                confirmation_score=9.0,
                penalties="no_flow_edge_alignment",
                confirmations_failed="price_action_trend",
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"
    assert "flow_not_directional:unclear" in out["trade_status_reason"].iloc[0]
    assert "no_flow_edge_alignment" in out["trade_status_reason"].iloc[0]


def test_oi_carryover_exact_leg_matching_populates_support_fields() -> None:
    scored = pd.DataFrame([_credit_row()])
    oi = pd.DataFrame(
        [
            {
                "option_symbol": "AAA260604P00100000",
                "right": "P",
                "oi_diff_plain": 250,
                "prev_bid_volume": 1000,
                "prev_ask_volume": 100,
                "prev_total_premium": 1_000_000,
            },
            {
                "option_symbol": "AAA260604P00095000",
                "right": "P",
                "oi_diff_plain": 100,
                "prev_bid_volume": 600,
                "prev_ask_volume": 100,
                "prev_total_premium": 250_000,
            },
        ]
    )
    oi.attrs["source_path"] = "chain-oi-changes-latest-2026-05-06.csv"

    out = apply_oi_carryover(scored, oi)

    assert out["oi_carryover_status"].iloc[0] == "supportive"
    assert out["short_leg_oi_change"].iloc[0] == 250
    assert out["short_leg_side_bias"].iloc[0] == "bullish"
    assert "latest" in out["oi_source_file"].iloc[0]


def test_near_trigger_trade_appears_as_watch_with_exact_limit_order() -> None:
    scored = pd.DataFrame([_credit_row(credit=1.10, credit_pct_width=0.22, replay_ev_verdict="acceptable")])

    status = assign_trade_statuses(scored)
    watch = build_entry_watchlist(status)

    assert status["trade_status"].iloc[0] == "Watch"
    assert watch["ticker"].tolist() == ["AAA"]
    assert watch["required_credit"].iloc[0] == 1.25
    assert "No chase below $1.25" in watch["trigger"].iloc[0]


def test_compact_action_board_keeps_one_watch_row_per_ticker() -> None:
    watch = pd.DataFrame(
        [
            _debit_row(ticker="TSLA", score=5.1, expiry=dt.date(2026, 5, 22)),
            _debit_row(ticker="TSLA", score=5.4, expiry=dt.date(2026, 6, 18)),
            _debit_row(ticker="QQQ", score=5.9),
        ]
    )

    action = _compact_action_rows(pd.DataFrame(), watch, pd.DataFrame())

    assert action["Ticker"].tolist().count("TSLA") == 1
    assert "QQQ" in action["Ticker"].tolist()


def test_price_already_better_than_trigger_is_not_watch_when_thesis_fails() -> None:
    scored = pd.DataFrame([_debit_row(flow_quality="unclear", debit=1.00, replay_ev_verdict="acceptable_proxy")])

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"


def test_execute_watch_avoid_research_statuses_are_produced() -> None:
    rows = [
        _credit_row(ticker="EXEC"),
        _credit_row(ticker="WATCH", credit=1.10, credit_pct_width=0.22, replay_ev_verdict="acceptable"),
        _credit_row(ticker="AVOID", hard_rejects="earnings_within_7d:4"),
        _credit_row(ticker="RESEARCH", flow_quality="unclear", confirmation_score=6.0),
    ]

    out = assign_trade_statuses(pd.DataFrame(rows))

    assert set(out["trade_status"]) == {"Execute", "Watch", "Avoid", "Research"}


def test_earnings_within_seven_days_income_trade_cannot_execute() -> None:
    scored = pd.DataFrame([_credit_row(hard_rejects="earnings_within_7d:4")])

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Avoid"


def test_credit_tier_logic_has_no_sub_twenty_five_percent_execute_lane() -> None:
    rows = [
        _credit_row(ticker="APLUS", credit=1.25, credit_pct_width=0.25, oi_carryover_status="matched_unconfirmed"),
        _credit_row(ticker="BTIER", credit=1.00, credit_pct_width=0.20, oi_carryover_status="supportive"),
        _credit_row(ticker="WATCH", credit=1.10, credit_pct_width=0.22, replay_ev_verdict="acceptable"),
    ]

    out = assign_trade_statuses(pd.DataFrame(rows))

    assert out.set_index("ticker").loc["APLUS", "trade_tier"] == "Execute A"
    assert out.set_index("ticker").loc["BTIER", "trade_tier"] == "near-trigger"
    assert out.set_index("ticker").loc["BTIER", "trade_status"] == "Watch"
    assert out.set_index("ticker").loc["WATCH", "trade_status"] == "Watch"


def test_news_caution_and_final_guard_cannot_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                ticker="CAUTION",
                penalties="regime_transition;news_catalyst_caution;final_guard_near_term_news_caution",
                catalyst_status="caution",
                confirmations_failed="earnings_news_risk",
                decision_reason="decision_final_quality_guard",
                score=6.9,
                confidence="Medium",
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"
    reason = out["trade_status_reason"].iloc[0]
    assert "news_catalyst_caution" in reason
    assert "earnings_news_risk" in reason
    assert "final_quality_guard" in reason


def test_regime_transition_allows_only_high_confidence_execute() -> None:
    rows = [
        _credit_row(ticker="MEDIUM", penalties="regime_transition", score=7.4, confidence="Medium"),
        _credit_row(ticker="HIGH", penalties="regime_transition", score=7.8, confidence="High"),
    ]

    out = assign_trade_statuses(pd.DataFrame(rows)).set_index("ticker")

    assert out.loc["MEDIUM", "trade_status"] == "Research"
    assert "regime_transition_defensive" in out.loc["MEDIUM", "trade_status_reason"]
    assert out.loc["HIGH", "trade_status"] == "Execute"


def test_replay_validated_secondary_income_credit_can_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                ticker="SEC",
                flow_quality="spread_leg",
                credit=1.3,
                credit_pct_width=0.26,
                expected_move_ratio=0.35,
                replay_ev_verdict="acceptable_secondary_income",
                decision_eligible=True,
                decision_tier="secondary_income",
                confirmation_score=6.5,
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Execute"
    assert out["trade_tier"].iloc[0] == "Execute Secondary"


def test_secondary_income_proxy_cannot_promote_reject_confidence_trade() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                ticker="PROXYSEC",
                flow_quality="spread_leg",
                credit=1.1,
                credit_pct_width=0.22,
                expected_move_ratio=0.35,
                replay_ev_verdict="secondary_income_proxy",
                decision_eligible=False,
                decision_tier="",
                score=3.9,
                confidence="Reject",
                confirmation_score=8.5,
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"
    assert "credit_ev_not_supported:secondary_income_proxy" in out["trade_status_reason"].iloc[0]


def test_debit_spread_lane_requires_reward_risk_iv_breakeven_and_confirmation() -> None:
    valid = _debit_row(ticker="GOOD", replay_ev_verdict="acceptable")
    bad = _debit_row(
        ticker="BAD",
        debit=2.80,
        debit_pct_width=0.56,
        reward_risk=0.8,
        iv_rank=90,
        expected_move_ratio=0.7,
    )

    out = assign_trade_statuses(pd.DataFrame([valid, bad]))

    assert out.set_index("ticker").loc["GOOD", "trade_status"] == "Execute"
    assert out.set_index("ticker").loc["BAD", "trade_status"] != "Execute"


def test_tactical_debit_lane_can_execute_small_risk_transition_setup() -> None:
    scored = pd.DataFrame(
        [
            _debit_row(
                ticker="TACT",
                flow_quality="unclear",
                penalties="regime_transition",
                replay_ev_verdict="positive",
                edge_verdict="positive",
                edge_sample_size=10,
                debit=1.88,
                debit_pct_width=0.376,
                reward_risk=1.66,
                expected_move_ratio=1.27,
                quote_width_pct=0.03,
                max_loss=188.0,
                confirmation_score=10.0,
                score=7.02,
                confidence="High",
                catalyst_status="mixed",
                oi_carryover_status="matched_unconfirmed",
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Execute"
    assert out["trade_tier"].iloc[0] == "Execute Tactical"


def test_tactical_debit_lane_requires_news_confirmation() -> None:
    scored = pd.DataFrame(
        [
            _debit_row(
                ticker="TACT",
                flow_quality="unclear",
                penalties="regime_transition;news_unconfirmed",
                replay_ev_verdict="positive",
                edge_verdict="positive",
                edge_sample_size=10,
                debit=1.88,
                debit_pct_width=0.376,
                reward_risk=1.66,
                expected_move_ratio=1.27,
                quote_width_pct=0.03,
                max_loss=188.0,
                confirmation_score=10.0,
                score=7.02,
                confidence="High",
                catalyst_status="unknown",
                oi_carryover_status="matched_unconfirmed",
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"


def test_positive_thin_replay_sample_cannot_execute() -> None:
    scored = pd.DataFrame(
        [
            _debit_row(
                ticker="THINPOS",
                replay_ev_verdict="positive",
                edge_verdict="positive",
                edge_sample_size=4,
                confirmation_score=10.0,
                score=8.0,
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"
    assert "thin_replay_sample:n=4" in out["trade_status_reason"].iloc[0]


def test_bullish_debit_needs_market_regime_alignment_in_weak_flow() -> None:
    scored = pd.DataFrame(
        [
            _debit_row(
                ticker="BULL",
                replay_ev_verdict="positive",
                edge_verdict="positive",
                edge_sample_size=12,
                catalyst_status="mixed",
                confirmation_score=10.0,
                score=8.0,
            )
        ]
    )

    confirmed = apply_confirmation_framework(
        scored,
        asof=ASOF,
        regime={"trend": "range", "flow": "weak", "sizing_stance": "normal"},
    )
    out = assign_trade_statuses(confirmed)

    assert out["trade_status"].iloc[0] == "Research"
    assert "market_regime_alignment" in out["trade_status_reason"].iloc[0]


def test_debit_proxy_ev_routes_to_research_not_execute() -> None:
    scored = pd.DataFrame([_debit_row(ticker="PROXY", replay_ev_verdict="acceptable_proxy")])

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"
    assert "debit_proxy_ev_only" in out["trade_status_reason"].iloc[0]


def test_missing_ticker_news_can_surface_manual_scout_without_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                ticker="SCOUT",
                direction="Bear Call",
                strategy="Bear Call Credit Spread",
                short_leg="SCOUT260515C00105000",
                long_leg="SCOUT260515C00110000",
                combined_flow_bias=-0.03,
                flow_quality="unclear",
                penalties="news_unconfirmed",
                catalyst_status="unknown",
                replay_ev_verdict="acceptable",
                edge_verdict="acceptable",
                edge_sample_size=40,
                confirmation_score=10.0,
                score=8.2,
                credit=1.20,
                credit_pct_width=0.24,
                max_loss=380.0,
                quote_width_pct=0.03,
                oi_carryover_status="matched_unconfirmed",
            )
        ]
    )

    status = assign_trade_statuses(scored)
    watch = build_entry_watchlist(status)

    assert status["trade_status"].iloc[0] == "Watch"
    assert status["trade_tier"].iloc[0] == "manual-confirmation-scout"
    assert "news_unconfirmed" in status["trade_status_reason"].iloc[0]
    assert watch["watch_kind"].iloc[0] == "manual_confirmation_scout"
    assert "SCOUT ONLY" in watch["trigger"].iloc[0]
    assert watch["target_entry"].iloc[0] == ">= $1.25 credit"
    action = _compact_action_rows(pd.DataFrame(), watch, pd.DataFrame())
    assert "manual check must clear" in action["Why"].iloc[0]


def test_manual_scout_does_not_override_price_action_failure() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                ticker="BADSCOUT",
                flow_quality="unclear",
                confirmations_failed="price_action_trend",
                replay_ev_verdict="positive",
                edge_verdict="positive",
                edge_sample_size=40,
                confirmation_score=9.0,
                score=8.0,
                credit=1.20,
                credit_pct_width=0.24,
                quote_width_pct=0.03,
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Research"
    assert "price_action_trend" in out["trade_status_reason"].iloc[0]


def test_debit_above_target_stays_visible_with_price_annotation() -> None:
    scored = pd.DataFrame(
        [
            _debit_row(
                ticker="PRICEY",
                debit=2.90,
                debit_pct_width=0.58,
                reward_risk=0.72,
                replay_ev_verdict="acceptable",
                edge_verdict="acceptable",
                confirmation_score=7.0,
            )
        ]
    )

    status = assign_trade_statuses(scored)
    watch = build_entry_watchlist(status)

    assert status["trade_status"].iloc[0] == "Watch"
    assert "above target" in status["price_annotation"].iloc[0]
    assert watch["ticker"].tolist() == ["PRICEY"]
    assert watch["max_debit"].iloc[0] == 2.25
    assert "No chase above $2.25" in watch["trigger"].iloc[0]


def test_etf_fallback_execute_demotes_when_single_name_execute_quality_exists() -> None:
    single = _credit_row(ticker="AAA", index_fallback=False)
    etf = _credit_row(ticker="SPY", index_fallback=True)

    out = assign_trade_statuses(pd.DataFrame([single, etf]))

    assert out.set_index("ticker").loc["AAA", "trade_status"] == "Execute"
    assert out.set_index("ticker").loc["SPY", "trade_status"] == "Research"
    assert "fallback disabled" in out.set_index("ticker").loc["SPY", "trade_status_reason"]


def test_index_primary_mode_lets_index_income_compete_with_single_name_execute() -> None:
    single = _credit_row(ticker="AAA", index_fallback=False)
    etf = _credit_row(ticker="SPY", index_fallback=True)

    out = assign_trade_statuses(pd.DataFrame([single, etf]), index_income_mode="primary")

    assert out.set_index("ticker").loc["AAA", "trade_status"] == "Execute"
    assert out.set_index("ticker").loc["SPY", "trade_status"] == "Execute"
    assert "fallback disabled" not in out.set_index("ticker").loc["SPY", "trade_status_reason"]


def test_replay_edge_model_positive_match_promotes_live_credit_candidate(tmp_path) -> None:
    replay_dir = tmp_path / "codexuw_replay_edge"
    replay_dir.mkdir()
    history_rows = []
    for i in range(12):
        won = i < 9
        history_rows.append(
            {
                "asof": f"2026-04-{i + 1:02d}",
                "ticker": f"H{i}",
                "sector": "Technology",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "expiry": "2026-05-15",
                "dte": 30,
                "stock_price_eod": 100.0,
                "short_strike_eod": 95.0,
                "long_strike_eod": 90.0,
                "entry_credit_pct_width": 0.25,
                "entry_quote_width_pct": 0.12,
                "expected_move_ratio": 0.85,
                "iv_rank": 45,
                "iv30d": 0.25,
                "realized_volatility_30d": 0.17,
                "combined_flow_bias": 0.12,
                "flow_quality": "directional",
                "regime": "downtrend",
                "exact_evaluated": True,
                "decision_pass": True,
                "exact_win": won,
                "pnl_1x": 80.0 if won else -30.0,
            }
        )
    pd.DataFrame(history_rows).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    scored = pd.DataFrame([_credit_row(replay_ev_verdict="structure_proxy", regime_trend="downtrend")])
    edged = apply_replay_edge_model(scored, tmp_path)

    assert edged["edge_verdict"].iloc[0] == "positive"
    assert edged["replay_ev_verdict"].iloc[0] == "positive"
    assert edged["edge_sample_size"].iloc[0] == 12


def test_replay_edge_model_negative_match_hard_rejects_execute(tmp_path) -> None:
    replay_dir = tmp_path / "codexuw_replay_edge"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "asof": f"2026-04-{i + 1:02d}",
                "ticker": "AAA",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "expiry": "2026-05-15",
                "dte": 30,
                "stock_price_eod": 100.0,
                "short_strike_eod": 95.0,
                "long_strike_eod": 90.0,
                "entry_credit_pct_width": 0.25,
                "entry_quote_width_pct": 0.12,
                "expected_move_ratio": 0.85,
                "iv_rank": 45,
                "iv30d": 0.25,
                "realized_volatility_30d": 0.17,
                "combined_flow_bias": 0.12,
                "flow_quality": "directional",
                "regime": "downtrend",
                "exact_evaluated": True,
                "decision_pass": True,
                "exact_win": False,
                "pnl_1x": -80.0,
            }
            for i in range(12)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    edged = apply_replay_edge_model(pd.DataFrame([_credit_row(regime_trend="downtrend")]), tmp_path)
    out = assign_trade_statuses(edged)

    assert edged["edge_verdict"].iloc[0] == "negative"
    assert "negative_replay_edge" in edged["hard_rejects"].iloc[0]
    assert out["trade_status"].iloc[0] == "Avoid"


def test_v4_edge_history_is_namespaced_and_asof_safe(tmp_path) -> None:
    v4_dir = tmp_path / "codexdaily_v4_edge_history_v1"
    other_dir = tmp_path / "codexuw_replay_options_agent"
    v4_dir.mkdir()
    other_dir.mkdir()

    def history_row(asof: str, exit_day: str, pnl: float, *, expiry: str = "2026-04-15") -> dict:
        return {
            "asof": asof,
            "exit_day": exit_day,
            "ticker": "AAA",
            "direction": "Bull Put",
            "strategy": "Bull Put Credit Spread",
            "dte": 30,
            "expiry": expiry,
            "short_strike_eod": 100.0,
            "long_strike_eod": 95.0,
            "entry_credit_pct_width": 0.26,
            "expected_move_ratio": 0.85,
            "combined_flow_bias": 0.12,
            "flow_quality": "directional",
            "regime": "downtrend",
            "iv30d": 0.24,
            "realized_volatility_30d": 0.16,
            "iv_rank": 45,
            "entry_quote_width_pct": 0.15,
            "next_earnings_dt": "2026-08-01",
            "exact_evaluated": True,
            "decision_pass": True,
            "exact_win": pnl > 0,
            "pnl_1x": pnl,
        }

    pd.DataFrame(
        [
            history_row("2026-04-01", "2026-04-10", 50.0),
            history_row("2026-04-02", "2026-04-11", 40.0),
            history_row("2026-04-03", "2026-04-12", 30.0),
            history_row("2026-05-06", "2026-05-08", -500.0, expiry="2026-06-15"),
        ]
    ).to_csv(v4_dir / "codexuw_replay_detail.csv", index=False)
    pd.DataFrame([history_row("2026-04-04", "2026-04-13", -500.0) for _ in range(3)]).to_csv(
        other_dir / "codexuw_replay_detail.csv", index=False
    )

    edged = apply_replay_edge_model(
        pd.DataFrame([_credit_row(regime_trend="downtrend")]),
        tmp_path,
        asof=dt.date(2026, 5, 5),
        history_namespace="codexdaily_v4_edge_history",
    )

    assert edged["edge_sample_size"].iloc[0] == 3
    assert edged["edge_verdict"].iloc[0] == "thin_sample"
    assert edged["edge_history_namespace"].iloc[0] == "codexdaily_v4_edge_history"


def test_thin_replay_sample_can_watch_but_not_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                ticker="THIN",
                credit=1.10,
                credit_pct_width=0.22,
                replay_ev_verdict="thin_sample",
                edge_verdict="thin_sample",
                edge_sample_size=2,
                edge_avg_pnl=25.0,
                confirmation_score=7.0,
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Watch"


def test_unclear_spread_leg_flow_can_watch_with_promising_edge() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                flow_quality="spread_leg",
                credit=1.10,
                credit_pct_width=0.22,
                replay_ev_verdict="acceptable",
                edge_verdict="acceptable",
                confirmation_score=7.0,
            )
        ]
    )

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Watch"


def test_credit_spread_alternatives_emit_labeled_constructions() -> None:
    contracts = pd.DataFrame(
        [
            {"expiry": EXPIRY, "right": "P", "strike": 95.0, "symbol": "AAA260604P00095000", "bid": 1.40, "ask": 1.50, "bid_size": 10, "ask_size": 10, "regular_session_quote": True, "mark": 1.45, "delta": -0.22, "open_interest": 1000, "volume": 500},
            {"expiry": EXPIRY, "right": "P", "strike": 90.0, "symbol": "AAA260604P00090000", "bid": 0.35, "ask": 0.45, "bid_size": 10, "ask_size": 10, "regular_session_quote": True, "mark": 0.40, "delta": -0.12, "open_interest": 1000, "volume": 500},
            {"expiry": EXPIRY, "right": "P", "strike": 94.0, "symbol": "AAA260604P00094000", "bid": 1.15, "ask": 1.25, "bid_size": 10, "ask_size": 10, "regular_session_quote": True, "mark": 1.20, "delta": -0.18, "open_interest": 1000, "volume": 500},
            {"expiry": EXPIRY, "right": "P", "strike": 89.0, "symbol": "AAA260604P00089000", "bid": 0.20, "ask": 0.30, "bid_size": 10, "ask_size": 10, "regular_session_quote": True, "mark": 0.25, "delta": -0.10, "open_interest": 1000, "volume": 500},
        ]
    )

    alternatives = find_credit_spread_alternatives(
        contracts,
        direction="Bull Put",
        expiry=EXPIRY,
        spot=100.0,
        preferred_width=5.0,
        anchor_strike=95.0,
        expected_move_pct=0.06,
    )

    assert len(alternatives) >= 2
    assert {row["construction_source"] for row in alternatives}
    assert all("target_entry" in row for row in alternatives)


def test_expected_move_uses_candidate_dte_instead_of_daily_uw_field() -> None:
    expected = _expected_move_pct(
        pd.Series({"iv30d": 0.2265, "dte": 21, "implied_move_perc": 0.001151})
    )

    assert expected == pytest.approx(0.2265 * (21 / 365) ** 0.5)


def test_confirmation_framework_exempts_etf_from_company_earnings() -> None:
    row = _credit_row(
        ticker="QQQ",
        next_earnings_dt=ASOF + dt.timedelta(days=2),
        catalyst_status="mixed",
    )

    confirmed = apply_confirmation_framework(
        pd.DataFrame([row]),
        asof=ASOF,
        regime={"trend": "downtrend", "flow": "weak"},
    )

    assert "earnings_news_risk" not in confirmed.iloc[0]["confirmations_failed"]


def test_credit_confirmation_does_not_restore_deleted_distance_gate() -> None:
    row = _credit_row(
        expected_move_ratio=0.40,
        distance_pct=0.02,
        iv30d=0.30,
    )

    confirmed = apply_confirmation_framework(
        pd.DataFrame([row]),
        asof=ASOF,
        regime={"trend": "downtrend", "flow": "weak"},
    )

    failed = confirmed.iloc[0]["confirmations_failed"]
    assert "expected_move_buffer" not in failed
    assert "level_or_gex_protection" not in failed


def test_actionable_credit_construction_does_not_require_deleted_distance_gate() -> None:
    contracts = pd.DataFrame(
        [
            {"expiry": CREDIT_EXPIRY, "right": "P", "strike": 95.0, "symbol": "AAA260604P00095000", "bid": 1.75, "ask": 1.80, "bid_size": 10, "ask_size": 10, "regular_session_quote": True, "mark": 1.775, "delta": -0.30, "open_interest": 1000, "volume": 500},
            {"expiry": CREDIT_EXPIRY, "right": "P", "strike": 90.0, "symbol": "AAA260604P00090000", "bid": 0.40, "ask": 0.45, "bid_size": 10, "ask_size": 10, "regular_session_quote": True, "mark": 0.425, "delta": -0.12, "open_interest": 1000, "volume": 500},
        ]
    )

    alternatives = find_credit_spread_alternatives(
        contracts,
        direction="Bull Put",
        expiry=CREDIT_EXPIRY,
        spot=100.0,
        preferred_width=5.0,
        expected_move_pct=0.10,
        as_of_date=ASOF,
    )

    actionable = next(row for row in alternatives if row["construction_source"] == "actionable_quality")
    assert actionable["credit_pct_width"] >= 0.25
    assert actionable["breakeven_expected_move_ratio"] < 0.75


def test_open_interest_cannot_substitute_for_displayed_option_size() -> None:
    contracts = pd.DataFrame(
        [
            {"expiry": CREDIT_EXPIRY, "right": "P", "strike": 95.0, "symbol": "AAA260604P00095000", "bid": 1.75, "ask": 1.80, "bid_size": 0, "ask_size": 0, "regular_session_quote": False, "mark": 1.775, "delta": -0.30, "open_interest": 7000, "volume": 0},
            {"expiry": CREDIT_EXPIRY, "right": "P", "strike": 90.0, "symbol": "AAA260604P00090000", "bid": 0.40, "ask": 0.45, "bid_size": 0, "ask_size": 0, "regular_session_quote": False, "mark": 0.425, "delta": -0.12, "open_interest": 1600, "volume": 0},
        ]
    )

    alternatives = find_credit_spread_alternatives(
        contracts,
        direction="Bull Put",
        expiry=CREDIT_EXPIRY,
        spot=100.0,
        preferred_width=5.0,
        as_of_date=ASOF,
    )

    assert not any(row["construction_source"] == "actionable_quality" for row in alternatives)
    assert all(row["displayed_entry_size"] == 0 for row in alternatives)


def test_chain_parser_preserves_displayed_size_and_session_state() -> None:
    quoted_at = dt.datetime(2026, 5, 5, 14, 0, tzinfo=dt.timezone.utc)
    chain = {
        "callExpDateMap": {
            "2026-06-04:30": {
                "100.0": [{
                    "strikePrice": 100.0,
                    "symbol": "AAA260604C00100000",
                    "bid": 1.0,
                    "ask": 1.1,
                    "bidSize": 12,
                    "askSize": 8,
                    "quoteTimeInLong": int(quoted_at.timestamp() * 1000),
                }]
            }
        }
    }

    contract = chain_to_contracts(chain).iloc[0]

    assert contract["bid_size"] == 12
    assert contract["ask_size"] == 8
    assert bool(contract["regular_session_quote"])


def test_data_quality_gate_demotes_execute_when_portfolio_is_missing() -> None:
    scored = assign_trade_statuses(pd.DataFrame([_credit_row()]))
    assert scored["trade_status"].iloc[0] == "Execute"

    gated = apply_data_quality_gate(
        scored,
        {"status": "critical", "critical_blockers": ["schwab_portfolio_available"], "items": []},
    )

    assert gated["trade_status"].iloc[0] == "Research"
    assert "data_gate_missing_portfolio_state" in gated["data_quality_blockers"].iloc[0]


def test_data_quality_gate_does_not_apply_company_news_blocker_to_etf() -> None:
    scored = assign_trade_statuses(
        pd.DataFrame([_credit_row(ticker="QQQ", catalyst_status="unknown", penalties="news_unconfirmed")])
    )
    gated = apply_data_quality_gate(
        scored,
        {"status": "warning", "critical_blockers": [], "warnings": ["browser_news_notes_present"], "items": []},
    )

    assert "data_gate_news_unconfirmed" not in str(gated.iloc[0].get("data_quality_blockers") or "")


def test_data_quality_status_reports_required_live_inputs() -> None:
    status = build_data_quality_status(
        input_provenance={"exports": {"stock_screener": {}, "hot_chains": {}, "bot_eod_report": {}}, "browser_text_count": 0},
        scored=pd.DataFrame([_credit_row(live_status="PASS")]),
        portfolio={"status": "unavailable", "error": "token expired"},
        catalysts=pd.DataFrame([{"ticker": "AAA", "catalyst_status": "unknown"}]),
        recent_performance={"status": "unavailable", "reason": "no replay"},
        live_outcomes={"status": "unavailable", "reason": "no ledger"},
        run_mode="Intraday live execution",
    )

    assert status["status"] == "critical"
    assert "schwab_portfolio_available" in status["critical_blockers"]
    assert "browser_news_notes_present" in status["warnings"]


def test_data_quality_status_requires_all_five_uw_exports() -> None:
    status = build_data_quality_status(
        input_provenance={
            "exports": {
                "stock_screener": {},
                "hot_chains": {},
                "chain_oi_changes": {},
                "bot_eod_report": {},
                "dp_eod_report": {},
            },
            "browser_text_count": 0,
        },
        scored=pd.DataFrame([_credit_row(live_status="PASS")]),
        portfolio={"status": "ok", "position_count": 3},
        catalysts=pd.DataFrame([{"ticker": "AAA", "catalyst_status": "mixed"}]),
        recent_performance={"status": "ok", "latest_asof": "2026-07-21", "window": 30},
        live_outcomes={"status": "ok", "latest_report_date": "2026-07-21", "window": 30},
        run_mode="EOD swing target plan",
    )

    assert status["status"] == "warning"
    assert status["critical_blockers"] == []
    assert status["warnings"] == ["browser_news_notes_present"]


def test_data_quality_status_keeps_preopen_live_prices_as_row_level_warning() -> None:
    status = build_data_quality_status(
        input_provenance={
            "exports": {
                "stock_screener": {},
                "hot_chains": {},
                "chain_oi_changes": {},
                "bot_eod_report": {},
                "dp_eod_report": {},
            },
            "browser_text_count": 1,
        },
        scored=pd.DataFrame(
            [_credit_row(live_status="PASS", regular_session_quote=False, displayed_entry_size=0)]
        ),
        portfolio={"status": "ok", "position_count": 3},
        catalysts=pd.DataFrame([{"ticker": "AAA", "catalyst_status": "mixed"}]),
        recent_performance={"status": "ok", "latest_asof": "2026-07-21", "window": 30},
        live_outcomes={"status": "ok", "latest_report_date": "2026-07-21", "window": 30},
        run_mode="EOD swing target plan",
    )

    quote = next(item for item in status["items"] if item["check"] == "Schwab quotes available")
    assert status["status"] == "warning"
    assert status["critical_blockers"] == []
    assert quote["status"] == "warning"
    assert "0 row-level executable quotes" in quote["detail"]
    assert "1 live-priced PASS rows" in quote["detail"]


def test_negative_live_outcome_family_blocks_execute_confidence() -> None:
    live_outcomes = {
        "status": "ok",
        "family_summary": {
            "credit spreads": {"outcomes": 4, "avg_pnl": -75.0, "expectancy": "negative"}
        },
    }
    adjusted = apply_confidence_components(pd.DataFrame([_credit_row()]), live_outcomes=live_outcomes)
    status = assign_trade_statuses(adjusted)

    assert "negative_live_expectancy:credit spreads" in adjusted["penalties"].iloc[0]
    assert status["trade_status"].iloc[0] == "Research"
    assert "negative_live_expectancy" in status["trade_status_reason"].iloc[0]
