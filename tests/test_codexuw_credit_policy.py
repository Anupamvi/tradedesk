import pandas as pd

from codexuw.credit_policy import assess_credit_spread, credit_spread_confidence, credit_spread_edge_lane
from codexuw.daily_v4 import apply_v4_credit_sleeve_cap


def _credit_row(**overrides):
    row = {
        "direction": "Bull Put",
        "dte": 21,
        "entry_credit_pct_width": 0.20,
        "combined_flow_bias": 0.20,
        "entry_quote_width_pct": 0.15,
        "iv30d": 0.30,
        "realized_volatility_30d": 0.32,
        "expected_move_ratio": 0.80,
        "flow_quality": "unclear",
        "bot_flow_source_status": "bot_eod_loaded",
        "oi_carryover_status": "matched_unconfirmed",
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


def test_live_credit_sleeve_keeps_only_highest_ranked_execute():
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
