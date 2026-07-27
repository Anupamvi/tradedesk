import pandas as pd

from codexuw.credit_policy import assess_credit_spread, credit_spread_confidence, credit_spread_edge_lane
from codexuw.daily_v4 import apply_v4_credit_sleeve_cap


def _credit_row(**overrides):
    row = {
        "direction": "Bull Put",
        "dte": 30,
        "entry_credit_pct_width": 0.25,
        "combined_flow_bias": 0.20,
        "entry_quote_width_pct": 0.15,
        # IV/HV 1.40 -- comfortably above the 0.90 sanity bound
        "iv30d": 0.42,
        "realized_volatility_30d": 0.30,
        "iv_rank": 45.0,
        "expected_move_ratio": 0.80,
        "flow_quality": "unclear",
        "bot_flow_source_status": "bot_eod_loaded",
        "regime": "downtrend",
        "oi_carryover_status": "matched_unconfirmed",
        "edge_match_level": "exact",
        "edge_sample_size": 12,
        "edge_profit_factor": 1.25,
        "edge_avg_pnl": 20.0,
    }
    row.update(overrides)
    return row


def test_credit_gates_on_volatility_richness_not_iv_rank():
    """IV/HV is the binding volatility bound; iv_rank is only a preference.

    The bound sits at 0.90 rather than the 1.30 the capture proxy suggested:
    on replayed vertical P&L no threshold separates from no-threshold and the
    sweep is non-monotone, so 1.30 would have cut trade count 82% for noise.
    """
    # premium above the bound but low iv_rank -> still tradable
    rich_low_rank = _credit_row(iv_rank=20.0)
    assert credit_spread_edge_lane(rich_low_rank) == "volatility_premium"
    assert assess_credit_spread(rich_low_rank, live=False)[0]

    # high iv_rank but implied cheaper than realised -> rejected
    rank_only = _credit_row(iv30d=0.20, realized_volatility_30d=0.35, iv_rank=45.0)
    assert credit_spread_edge_lane(rank_only) == "none"
    ok, reasons = assess_credit_spread(rank_only, live=False)
    assert not ok
    assert "iv_hv_ratio_below_0.90" in reasons


def test_credit_near_cash_denominator_artifact_is_rejected():
    """A near-zero realised vol manufactures a huge ratio out of nothing.

    Without this floor the richest names screened were cash-like ETFs (ICSH,
    BOXX, JPST) at 6-12x IV/HV on 0.3-3.8% realised vol, where the credit is too
    small to survive costs.
    """
    artifact = _credit_row(iv30d=0.088, realized_volatility_30d=0.012)
    assert credit_spread_edge_lane(artifact) == "none"
    ok, reasons = assess_credit_spread(artifact, live=False)
    assert not ok
    assert "realized_vol_below_0.15" in reasons


def test_credit_earnings_inside_holding_window_is_not_a_gate():
    """Excluding earnings names was tested on replayed P&L and rejected.

    Inside the regime map they are the better half (n=154, win 81.2%, PF 1.27),
    a 21-day exclusion is neutral (delta -1.7) and a 7-day one is significantly
    harmful (delta -5.2, 90% CI [-9.8, -1.0]). The dangerous case, an event
    landing inside the position's life, is already a hard reject elsewhere.
    """
    assert assess_credit_spread(_credit_row(days_to_earnings=9), live=False)[0]
    assert assess_credit_spread(_credit_row(days_to_earnings=40), live=False)[0]


def test_credit_short_dated_expiries_are_rejected():
    """Sub-28-DTE credit spreads are where gamma destroys the edge."""
    ok, reasons = assess_credit_spread(_credit_row(dte=21), live=False)
    assert not ok
    assert any("dte" in reason.lower() for reason in reasons)


def test_credit_distance_buffer_is_no_longer_a_gate():
    """Distance is collinear with the credit band and was anti-predictive."""
    assert assess_credit_spread(_credit_row(expected_move_ratio=0.20), live=False)[0]


def test_credit_without_volatility_edge_is_rejected():
    ok, reasons = assess_credit_spread(
        _credit_row(iv30d=0.20, realized_volatility_30d=0.35, iv_rank=10.0),
        live=False,
    )
    assert not ok
    assert "iv_hv_ratio_below_0.90" in reasons


def test_credit_flow_alignment_remains_a_gate():
    """Flow carries no directional signal, but dropping the gate would loosen entry.

    Kept at 0.10 because on replayed P&L it is neutral (delta +1.4, 90% CI
    [-12.5, +16.1]) rather than harmful. It must not be used for ranking.
    """
    ok, reasons = assess_credit_spread(_credit_row(combined_flow_bias=0.02), live=False)
    assert not ok
    assert "flow_alignment_below_0.10" in reasons


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
            iv30d=0.42,
            realized_volatility_30d=0.30,
            iv_rank=50.0,
            expected_move_ratio=1.05,
            edge_sample_size=24,
            edge_profit_factor=1.50,
        ),
        live=True,
    )
    assert high == "high"
    assert high_reasons == []


def test_credit_direction_must_match_market_regime():
    """Credit verticals are contrarian: sell puts into a downtrend, calls into an uptrend."""
    ok, reasons = assess_credit_spread(_credit_row(regime="uptrend"), live=False)
    assert not ok
    assert "credit_regime_not_aligned:Bull Put:uptrend" in reasons

    bear = _credit_row(
        direction="Bear Call",
        regime="uptrend",
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
