from datetime import date, timedelta

import pytest

from codexswing.backtest.labels import (
    DailyBar,
    LabelDataError,
    exact_next_open_outcome,
    parse_orats_daily_rows,
)


def _bars(count: int = 8):
    start = date(2026, 1, 1)
    result = []
    for index in range(count):
        opening = 100.0 + index
        result.append(
            DailyBar(
                ticker="TEST",
                trade_date=(start + timedelta(days=index)).isoformat(),
                open=opening,
                high=opening + 2.0,
                low=opening - 1.0,
                close=opening + 1.0,
            )
        )
    return result


def test_exact_next_open_and_horizon_close_are_used() -> None:
    outcome = exact_next_open_outcome(
        sample_id="sample",
        ticker="TEST",
        side="LONG",
        decision_date="2026-01-01",
        bars=_bars(),
        horizon_sessions=3,
        round_trip_cost_bps=10.0,
    )
    assert outcome.entry_date == "2026-01-02"
    assert outcome.entry_price == 101.0
    assert outcome.label_end_date == "2026-01-04"
    assert outcome.exit_price == 104.0
    assert outcome.net_return == pytest.approx(104.0 / 101.0 - 1.0 - 0.001)
    assert outcome.maximum_favorable_excursion > 0
    assert outcome.maximum_adverse_excursion < 0


def test_incomplete_forward_horizon_fails_closed() -> None:
    with pytest.raises(LabelDataError, match="not yet complete"):
        exact_next_open_outcome(
            sample_id="sample",
            ticker="TEST",
            side="LONG",
            decision_date="2026-01-07",
            bars=_bars(),
            horizon_sessions=3,
        )


def test_orats_daily_parser_uses_adjusted_fields_and_rejects_duplicates() -> None:
    row = {
        "ticker": "TEST",
        "tradeDate": "2026-01-02",
        "open": 101,
        "hiPx": 103,
        "loPx": 100,
        "clsPx": 102,
    }
    parsed = parse_orats_daily_rows([row])
    assert parsed["TEST"][0].close == 102
    with pytest.raises(LabelDataError, match="duplicate"):
        parse_orats_daily_rows([row, row])
