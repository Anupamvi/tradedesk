import pytest

from codexswing.backtest.orats_option_replay import (
    close_historical_single_option,
    close_historical_vertical,
    select_historical_single_option,
    select_historical_vertical,
)


def _row(strike, delta, call_bid, call_ask, put_bid, put_ask):
    return {
        "ticker": "XYZ",
        "tradeDate": "2026-01-05",
        "expirDate": "2026-02-13",
        "strike": strike,
        "stockPrice": 100,
        "delta": delta,
        "callBidPrice": call_bid,
        "callAskPrice": call_ask,
        "putBidPrice": put_bid,
        "putAskPrice": put_ask,
        "callVolume": 500,
        "putVolume": 500,
        "callOpenInterest": 1000,
        "putOpenInterest": 1000,
    }


def test_bull_call_uses_conservative_66_percent_entry_and_exact_exit() -> None:
    entry = [
        _row(98, 0.58, 5.8, 6.0, 2.7, 2.9),
        _row(104, 0.30, 2.7, 2.9, 7.6, 7.9),
    ]
    vertical, reason = select_historical_vertical(
        entry, "XYZ", "LONG", "2026-01-05", "BULL_CALL_DEBIT"
    )
    assert reason == "selected"
    assert vertical is not None
    assert vertical.modeled_entry_signed_debit == pytest.approx(3.164)
    assert vertical.modeled_entry_signed_debit < vertical.natural_open_signed_debit
    exit_rows = [dict(row, tradeDate="2026-01-12") for row in entry]
    exit_rows[0]["callBidPrice"] = 7.0
    exit_rows[0]["callAskPrice"] = 7.2
    exit_rows[1]["callAskPrice"] = 3.0
    value, close_reason = close_historical_vertical(vertical, exit_rows, "2026-01-12")
    assert close_reason == "closed"
    assert value == 4.0


def test_bull_put_credit_geometry_is_supported() -> None:
    entry = [
        _row(90, 0.90, 10.1, 10.3, 0.9, 1.0),
        _row(95, 0.75, 6.8, 7.0, 2.4, 2.5),
    ]
    vertical, reason = select_historical_vertical(
        entry, "XYZ", "LONG", "2026-01-05", "BULL_PUT_CREDIT"
    )
    assert reason == "selected"
    assert vertical is not None
    assert vertical.modeled_entry_signed_debit < 0
    assert vertical.maximum_risk_dollars > 0


def test_long_call_uses_75_percent_entry_and_exact_bid_exit() -> None:
    entry = [_row(100, 0.52, 3.0, 3.4, 2.8, 3.2)]
    option, reason = select_historical_single_option(
        entry, "XYZ", "LONG", "2026-01-05", "LONG_CALL"
    )
    assert reason == "selected"
    assert option is not None
    assert option.modeled_entry_signed_debit == pytest.approx(3.30)
    exit_rows = [dict(entry[0], tradeDate="2026-01-12", callBidPrice=4.2, callAskPrice=4.4)]
    value, close_reason = close_historical_single_option(option, exit_rows, "2026-01-12")
    assert close_reason == "closed"
    assert value == pytest.approx(4.2)
