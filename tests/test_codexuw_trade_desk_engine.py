from __future__ import annotations

import datetime as dt

import pandas as pd

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
    classify_flow_quality,
    select_final_trades,
    _compact_action_rows,
)
from codexuw.schwab_live import find_credit_spread_alternatives


ASOF = dt.date(2026, 5, 5)
EXPIRY = dt.date(2026, 5, 15)


def _credit_row(**overrides) -> dict:
    row = {
        "ticker": "AAA",
        "sector": "Technology",
        "direction": "Bull Put",
        "strategy": "Bull Put Credit Spread",
        "strategy_kind": "Credit",
        "expiry": EXPIRY,
        "dte": 10,
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
        "combined_flow_bias": 0.12,
        "score": 7.5,
        "confidence": "High",
        "live_status": "PASS",
        "quote_width_pct": 0.15,
        "short_oi": 1000,
        "short_volume": 500,
        "long_oi": 1000,
        "long_volume": 500,
        "short_leg": "AAA260515P00100000",
        "long_leg": "AAA260515P00095000",
        "flow_quality": "directional",
        "flow_quality_reason": "premium bias aligns",
        "oi_carryover_status": "supportive",
        "replay_ev_verdict": "acceptable",
        "confirmation_score": 8.0,
        "catalyst_status": "supportive",
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
        "confirmation_score": 8.0,
        "catalyst_status": "supportive",
    }
    row.update(overrides)
    return row


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


def test_risk_cap_breach_still_blocks_selection() -> None:
    scored = pd.DataFrame([_credit_row(max_loss=4000.0)])

    final = select_final_trades(
        assign_trade_statuses(scored),
        regime={"sizing_stance": "normal"},
        risk_budget=3000,
        recent_performance={"status": "unavailable"},
    )

    assert final.empty


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
                "option_symbol": "AAA260515P00100000",
                "right": "P",
                "oi_diff_plain": 250,
                "prev_bid_volume": 1000,
                "prev_ask_volume": 100,
                "prev_total_premium": 1_000_000,
            },
            {
                "option_symbol": "AAA260515P00095000",
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
    scored = pd.DataFrame([_credit_row(credit=0.82, credit_pct_width=0.164, replay_ev_verdict="acceptable")])

    status = assign_trade_statuses(scored)
    watch = build_entry_watchlist(status)

    assert status["trade_status"].iloc[0] == "Watch"
    assert watch["ticker"].tolist() == ["AAA"]
    assert watch["required_credit"].iloc[0] == 0.9
    assert "No chase below $0.90" in watch["trigger"].iloc[0]


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
        _credit_row(ticker="WATCH", credit=0.82, credit_pct_width=0.164, replay_ev_verdict="acceptable"),
        _credit_row(ticker="AVOID", hard_rejects="earnings_within_7d:4"),
        _credit_row(ticker="RESEARCH", flow_quality="unclear", confirmation_score=6.0),
    ]

    out = assign_trade_statuses(pd.DataFrame(rows))

    assert set(out["trade_status"]) == {"Execute", "Watch", "Avoid", "Research"}


def test_earnings_within_seven_days_income_trade_cannot_execute() -> None:
    scored = pd.DataFrame([_credit_row(hard_rejects="earnings_within_7d:4")])

    out = assign_trade_statuses(scored)

    assert out["trade_status"].iloc[0] == "Avoid"


def test_credit_tier_logic_execute_a_execute_b_and_watch() -> None:
    rows = [
        _credit_row(ticker="APLUS", credit=1.25, credit_pct_width=0.25, oi_carryover_status="matched_unconfirmed"),
        _credit_row(ticker="BTIER", credit=1.00, credit_pct_width=0.20, oi_carryover_status="supportive"),
        _credit_row(ticker="WATCH", credit=0.82, credit_pct_width=0.164, replay_ev_verdict="acceptable"),
    ]

    out = assign_trade_statuses(pd.DataFrame(rows))

    assert out.set_index("ticker").loc["APLUS", "trade_tier"] == "Execute A"
    assert out.set_index("ticker").loc["BTIER", "trade_tier"] == "Execute B"
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
                credit=1.1,
                credit_pct_width=0.22,
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


def test_replay_edge_model_positive_match_promotes_live_credit_candidate(tmp_path) -> None:
    replay_dir = tmp_path / "codexuw_replay_edge"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "asof": "2026-04-20",
                "ticker": "AAA",
                "sector": "Technology",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "expiry": "2026-05-15",
                "dte": 18,
                "stock_price_eod": 100.0,
                "short_strike_eod": 95.0,
                "long_strike_eod": 90.0,
                "entry_credit_pct_width": 0.22,
                "entry_quote_width_pct": 0.12,
                "iv_rank": 45,
                "iv30d": 0.25,
                "combined_flow_bias": 0.12,
                "flow_quality": "directional",
                "regime": "range",
                "exact_evaluated": True,
                "exact_win": True,
                "pnl_1x": 90.0,
            },
            {
                "asof": "2026-04-21",
                "ticker": "AAA",
                "sector": "Technology",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "expiry": "2026-05-15",
                "dte": 18,
                "stock_price_eod": 101.0,
                "short_strike_eod": 96.0,
                "long_strike_eod": 91.0,
                "entry_credit_pct_width": 0.21,
                "entry_quote_width_pct": 0.10,
                "iv_rank": 46,
                "iv30d": 0.25,
                "combined_flow_bias": 0.11,
                "flow_quality": "directional",
                "regime": "range",
                "exact_evaluated": True,
                "exact_win": True,
                "pnl_1x": 80.0,
            },
            {
                "asof": "2026-04-22",
                "ticker": "AAA",
                "sector": "Technology",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "expiry": "2026-05-15",
                "dte": 18,
                "stock_price_eod": 102.0,
                "short_strike_eod": 97.0,
                "long_strike_eod": 92.0,
                "entry_credit_pct_width": 0.20,
                "entry_quote_width_pct": 0.10,
                "iv_rank": 46,
                "iv30d": 0.25,
                "combined_flow_bias": 0.11,
                "flow_quality": "directional",
                "regime": "range",
                "exact_evaluated": True,
                "exact_win": False,
                "pnl_1x": -30.0,
            },
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    scored = pd.DataFrame([_credit_row(replay_ev_verdict="structure_proxy", regime_trend="range")])
    edged = apply_replay_edge_model(scored, tmp_path)

    assert edged["edge_verdict"].iloc[0] == "positive"
    assert edged["replay_ev_verdict"].iloc[0] == "positive"
    assert edged["edge_sample_size"].iloc[0] == 3


def test_replay_edge_model_negative_match_hard_rejects_execute(tmp_path) -> None:
    replay_dir = tmp_path / "codexuw_replay_edge"
    replay_dir.mkdir()
    pd.DataFrame(
        [
            {
                "asof": f"2026-04-2{i}",
                "ticker": "AAA",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "expiry": "2026-05-15",
                "dte": 18,
                "stock_price_eod": 100.0,
                "short_strike_eod": 95.0,
                "long_strike_eod": 90.0,
                "entry_credit_pct_width": 0.22,
                "entry_quote_width_pct": 0.12,
                "iv30d": 0.25,
                "combined_flow_bias": 0.12,
                "regime": "range",
                "exact_evaluated": True,
                "exact_win": False,
                "pnl_1x": -80.0,
            }
            for i in range(3)
        ]
    ).to_csv(replay_dir / "codexuw_replay_detail.csv", index=False)

    edged = apply_replay_edge_model(pd.DataFrame([_credit_row(regime_trend="range")]), tmp_path)
    out = assign_trade_statuses(edged)

    assert edged["edge_verdict"].iloc[0] == "negative"
    assert "negative_replay_edge" in edged["hard_rejects"].iloc[0]
    assert out["trade_status"].iloc[0] == "Avoid"


def test_thin_replay_sample_can_watch_but_not_execute() -> None:
    scored = pd.DataFrame(
        [
            _credit_row(
                ticker="THIN",
                credit=0.82,
                credit_pct_width=0.164,
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
                credit=0.82,
                credit_pct_width=0.164,
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
            {"expiry": EXPIRY, "right": "P", "strike": 95.0, "symbol": "AAA260515P00095000", "bid": 1.40, "ask": 1.50, "mark": 1.45, "delta": -0.22, "open_interest": 1000, "volume": 500},
            {"expiry": EXPIRY, "right": "P", "strike": 90.0, "symbol": "AAA260515P00090000", "bid": 0.35, "ask": 0.45, "mark": 0.40, "delta": -0.12, "open_interest": 1000, "volume": 500},
            {"expiry": EXPIRY, "right": "P", "strike": 94.0, "symbol": "AAA260515P00094000", "bid": 1.15, "ask": 1.25, "mark": 1.20, "delta": -0.18, "open_interest": 1000, "volume": 500},
            {"expiry": EXPIRY, "right": "P", "strike": 89.0, "symbol": "AAA260515P00089000", "bid": 0.20, "ask": 0.30, "mark": 0.25, "delta": -0.10, "open_interest": 1000, "volume": 500},
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


def test_data_quality_gate_demotes_execute_when_portfolio_is_missing() -> None:
    scored = assign_trade_statuses(pd.DataFrame([_credit_row()]))
    assert scored["trade_status"].iloc[0] == "Execute"

    gated = apply_data_quality_gate(
        scored,
        {"status": "critical", "critical_blockers": ["schwab_portfolio_available"], "items": []},
    )

    assert gated["trade_status"].iloc[0] == "Research"
    assert "data_gate_missing_portfolio_state" in gated["data_quality_blockers"].iloc[0]


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
    assert "browser_news_notes_present" in status["critical_blockers"]


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
