import pandas as pd

from codexuw.credit_policy import assess_credit_spread, credit_spread_confidence, credit_spread_edge_lane
from codexuw.daily_v4 import apply_v4_credit_sleeve_cap


def _credit_row(**overrides):
    row = {
        "direction": "Bull Put",
        "dte": 21,
        "entry_credit_pct_width": 0.25,
        "combined_flow_bias": 0.20,
        "entry_quote_width_pct": 0.15,
        "iv30d": 0.30,
        "realized_volatility_30d": 0.32,
        "expected_move_ratio": 0.80,
        "flow_quality": "unclear",
        "bot_flow_source_status": "bot_eod_loaded",
        "regime": "uptrend",
        "oi_carryover_status": "matched_unconfirmed",
        "edge_match_level": "exact",
        "edge_sample_size": 12,
        "edge_profit_factor": 1.25,
        "edge_avg_pnl": 20.0,
    }
    row.update(overrides)
    return row


def test_credit_requires_distance_even_with_volatility_premium():
    volatility_only = _credit_row(expected_move_ratio=0.55)
    assert credit_spread_edge_lane(volatility_only) == "none"
    ok, reasons = assess_credit_spread(volatility_only, live=False)
    assert not ok
    assert "credit_short_strike_inside_distance_buffer" in reasons

    distance = _credit_row(iv30d=0.20, realized_volatility_30d=0.35, expected_move_ratio=0.80)
    assert credit_spread_edge_lane(distance) == "distance_buffer"
    assert assess_credit_spread(distance, live=False)[0]


def test_credit_without_volatility_or_distance_edge_is_rejected():
    ok, reasons = assess_credit_spread(
        _credit_row(iv30d=0.20, realized_volatility_30d=0.35, expected_move_ratio=0.60),
        live=False,
    )
    assert not ok
    assert "credit_short_strike_inside_distance_buffer" in reasons


def test_credit_execute_floor_is_exactly_twenty_five_percent_width():
    assert assess_credit_spread(_credit_row(entry_credit_pct_width=0.25), live=False)[0]

    ok, reasons = assess_credit_spread(_credit_row(entry_credit_pct_width=0.249), live=False)

    assert not ok
    assert "credit_pct_width_outside_0.25_0.30" in reasons


def test_hedge_flow_is_rejected_but_spread_leg_is_medium_eligible():
    assert not assess_credit_spread(_credit_row(flow_quality="hedge"), live=False)[0]
    assert assess_credit_spread(_credit_row(flow_quality="spread_leg"), live=False)[0]


def test_credit_live_confidence_distinguishes_medium_and_high():
    medium, medium_reasons = credit_spread_confidence(_credit_row(), live=True)
    assert medium == "medium"
    assert "high_requires_directional_contract_flow" in medium_reasons

    high, high_reasons = credit_spread_confidence(
        _credit_row(
            flow_quality="directional",
            oi_carryover_status="supportive",
            iv30d=0.35,
            realized_volatility_30d=0.30,
            expected_move_ratio=1.05,
            edge_sample_size=24,
            edge_profit_factor=1.50,
        ),
        live=True,
    )
    assert high == "high"
    assert high_reasons == []


def test_credit_direction_must_match_market_regime():
    ok, reasons = assess_credit_spread(_credit_row(regime="downtrend"), live=False)
    assert not ok
    assert "credit_regime_not_aligned:Bull Put:downtrend" in reasons

    bear = _credit_row(
        direction="Bear Call",
        regime="downtrend",
        combined_flow_bias=-0.20,
    )
    assert assess_credit_spread(bear, live=False)[0]


def test_live_credit_sleeve_keeps_unvalidated_additional_credit_as_watch():
    rows = pd.DataFrame(
        [
            {"ticker": "AAA", "direction": "Bull Put", "strategy": "Bull Put Credit Spread", "trade_status": "Execute", "trade_tier": "Execute", "score": 8.0, "confirmation_score": 2.0, "edge_sample_size": 20, "quote_width_pct": 0.10, "trade_status_reason": ""},
            {"ticker": "BBB", "direction": "Bear Call", "strategy": "Bear Call Credit Spread", "trade_status": "Execute", "trade_tier": "Execute", "score": 6.0, "confirmation_score": 1.0, "edge_sample_size": 12, "quote_width_pct": 0.15, "trade_status_reason": ""},
            {"ticker": "CCC", "direction": "Bull Call", "strategy": "Bull Call Debit Spread", "trade_status": "Execute", "trade_tier": "Execute", "score": 5.0, "confirmation_score": 1.0, "trade_status_reason": ""},
        ]
    )
    capped = apply_v4_credit_sleeve_cap(rows)
    assert capped.set_index("ticker").at["AAA", "trade_status"] == "Execute"
    assert capped.set_index("ticker").at["BBB", "trade_status"] == "Watch"
    assert capped.set_index("ticker").at["CCC", "trade_status"] == "Execute"


def test_credit_book_allows_two_validated_independent_buckets_without_stacking_index_or_sector():
    def validated_credit(*, ticker, sector, index_fallback, score):
        return {
            "ticker": ticker,
            "sector": sector,
            "index_fallback": index_fallback,
            "direction": "Bear Call",
            "strategy": "Bear Call Credit Spread",
            "regime": "range",
            "regime_trend": "range",
            "trade_status": "Execute",
            "trade_tier": "Execute",
            "score": score,
            "confirmation_score": 8.0,
            "edge_sample_size": 20,
            "quote_width_pct": 0.10,
            "trade_status_reason": "",
            "confidence_calibration_status": "PASS",
            "confidence_model_tier": "strategy_family_validated",
            "confidence_calibration_sample_size": 150,
            "confidence_probability_lower_bound": 0.65,
            "confidence_calibration_brier": 0.18,
            "confidence_calibration_baseline_brier": 0.25,
            "payoff_calibration_status": "PASS",
            "payoff_route_key": "flow_cost::Credit|Bear Call|range|flow=directional|cost=18to30",
            "payoff_minimum_sample_required": 12,
            "payoff_sample_size": 17,
            "payoff_stress_10_profit_factor": 2.0,
            "payoff_walk_forward_oos_sample": 8,
            "payoff_walk_forward_oos_profit_factor": 1.5,
            "payoff_post_activation_oos_sample": 3,
            "payoff_post_activation_oos_average_pnl": 35.0,
            "payoff_post_activation_oos_profit_factor": 1.5,
        }

    rows = pd.DataFrame(
        [
            validated_credit(ticker="QQQ", sector="ETF", index_fallback=True, score=9.0),
            validated_credit(ticker="SPY", sector="ETF", index_fallback=True, score=8.0),
            validated_credit(ticker="NVDA", sector="Technology", index_fallback=False, score=7.0),
            validated_credit(ticker="AMD", sector="Technology", index_fallback=False, score=6.0),
        ]
    )

    allocated = apply_v4_credit_sleeve_cap(rows)
    by_ticker = allocated.set_index("ticker")

    assert by_ticker.at["QQQ", "trade_status"] == "Execute"
    assert by_ticker.at["NVDA", "trade_status"] == "Execute"
    assert by_ticker.at["SPY", "trade_status"] == "Watch"
    assert by_ticker.at["AMD", "trade_status"] == "Watch"
    assert by_ticker.at["QQQ", "v4_credit_risk_bucket"] == "broad-index"
    assert "broad-index" in by_ticker.at["SPY", "v4_direct_disposition_reason"]
    assert "sector:technology" in by_ticker.at["AMD", "v4_direct_disposition_reason"]
