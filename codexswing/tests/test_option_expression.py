import math

import pytest

from codexswing.options.expected_pnl import (
    CostAssumptions,
    ForecastDistribution,
    evaluate_long_option,
    evaluate_stock,
    evaluate_vertical,
)
from codexswing.options.pricing import black_scholes_price
from codexswing.options.structures import OptionQuote, StructureError, vertical_from_orats_rows
from codexswing.research.contracts import _quote_session_date, _round_up_to_tick


def _row(strike, call_bid, call_ask):
    return {
        "ticker": "TEST",
        "tradeDate": "2026-08-26",
        "quoteDate": "2026-08-26",
        "expirDate": "2026-09-25",
        "strike": strike,
        "stockPrice": 100.0,
        "spotPrice": 100.0,
        "callBidPrice": call_bid,
        "callAskPrice": call_ask,
        "callMidIv": 0.25,
        "callOpenInterest": 500,
        "callVolume": 100,
        "putBidPrice": max(strike - 99.0, 0.5),
        "putAskPrice": max(strike - 98.5, 0.7),
        "putMidIv": 0.27,
        "putOpenInterest": 450,
        "putVolume": 90,
        "smvVol": 0.26,
        "residualRate": 0.04,
        "updatedAt": "2026-08-26T19:46:00Z",
    }


def test_black_scholes_put_call_parity() -> None:
    call = black_scholes_price(100, 100, 0.5, 0.04, 0.2, "call")
    put = black_scholes_price(100, 100, 0.5, 0.04, 0.2, "put")
    expected = 100 - 100 * math.exp(-0.04 * 0.5)
    assert call - put == pytest.approx(expected, abs=1e-10)


def test_exact_call_debit_expression_is_costed() -> None:
    rows = [_row(100, 3.0, 3.2), _row(105, 1.0, 1.2)]
    spread = vertical_from_orats_rows(rows, "call", 100, 105, "2026-09-25")
    assert spread.strategy == "call_debit"
    assert spread.entry_debit_per_share == pytest.approx(2.2)
    forecast = ForecastDistribution(mean_simple_return=0.03, sigma_log_return=0.08, horizon_days=5)
    result = evaluate_vertical(
        spread,
        forecast,
        reference_spot=100.0,
        risk_free_rate=0.04,
        costs=CostAssumptions(contracts=1),
    )
    assert result.status == "CURRENT_CONTRACT_MODEL_ONLY"
    assert result.natural_entry_debit_dollars == pytest.approx(220.0)
    assert result.modeled_exit_cost_dollars > 0
    assert result.round_trip_commissions == pytest.approx(2.60)
    assert result.expiry_max_loss_dollars < 0
    assert 0 <= result.probability_positive_after_costs <= 1


def test_explicit_limit_drives_ev_and_risk_instead_of_natural_quote() -> None:
    rows = [_row(100, 3.0, 3.4), _row(105, 1.0, 1.4)]
    spread = vertical_from_orats_rows(rows, "call", 100, 105, "2026-09-25")
    forecast = ForecastDistribution(mean_simple_return=0.03, sigma_log_return=0.08, horizon_days=5)
    natural = evaluate_vertical(spread, forecast, 100.0, 0.04)
    limit = evaluate_vertical(
        spread,
        forecast,
        100.0,
        0.04,
        entry_debit_per_share=2.10,
        entry_price_source="TEST_LIMIT",
    )
    assert limit.modeled_entry_debit_dollars == pytest.approx(210.0)
    assert limit.entry_price_source == "TEST_LIMIT"
    assert limit.expected_pnl_after_costs > natural.expected_pnl_after_costs
    assert limit.expiry_max_loss_dollars > natural.expiry_max_loss_dollars


def test_spot_disagreement_fails_closed() -> None:
    rows = [_row(100, 3.0, 3.2), _row(105, 1.0, 1.2)]
    spread = vertical_from_orats_rows(rows, "call", 100, 105, "2026-09-25")
    forecast = ForecastDistribution(mean_simple_return=0.01, sigma_log_return=0.05, horizon_days=5)
    with pytest.raises(StructureError, match="disagree"):
        evaluate_vertical(spread, forecast, reference_spot=90.0, risk_free_rate=0.04)


def test_stock_expression_uses_same_forecast_distribution() -> None:
    forecast = ForecastDistribution(mean_simple_return=0.02, sigma_log_return=0.05, horizon_days=5)
    result = evaluate_stock("TEST", 100.0, forecast)
    assert result.expected_pnl_after_costs > 0
    assert result.p05_pnl_after_costs < 0


def test_long_call_expression_uses_explicit_limit_and_one_leg_costs() -> None:
    quote = OptionQuote.from_orats(_row(100, 3.0, 3.4), "call")
    forecast = ForecastDistribution(mean_simple_return=0.03, sigma_log_return=0.08, horizon_days=5)
    result = evaluate_long_option(
        quote,
        forecast,
        reference_spot=100.0,
        risk_free_rate=0.04,
        entry_debit_per_share=3.30,
        entry_price_source="75_PERCENT_SINGLE_LEG_SPREAD",
    )
    assert result.strategy == "long_call"
    assert result.modeled_entry_debit_dollars == pytest.approx(330.0)
    assert result.round_trip_commissions == pytest.approx(1.30)
    assert result.expiry_max_loss_dollars == pytest.approx(-331.30)
    assert result.expiry_max_profit_dollars is None


def test_zero_expected_stock_return_is_negative_after_costs() -> None:
    forecast = ForecastDistribution(mean_simple_return=0.0, sigma_log_return=0.05, horizon_days=5)
    result = evaluate_stock("TEST", 100.0, forecast)
    assert result.expected_pnl_after_costs < 0


def test_option_leg_quote_time_maps_to_market_session() -> None:
    assert _quote_session_date(1787860787912) == "2026-08-27"


def test_single_option_limit_rounds_conservatively_to_a_manual_tick() -> None:
    assert _round_up_to_tick(3.525) == 3.55
    assert _round_up_to_tick(2.514) == 2.52
