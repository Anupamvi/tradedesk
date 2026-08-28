from codexswing.v5.events import evaluate_event_exclusions, parse_orats_events
from codexswing.v5.regimes import CoreRegimeObservation, select_nearest_analogs


def _regime(trade_date, side="LONG", iv=0.30, ticker="SPY"):
    return CoreRegimeObservation(
        ticker=ticker,
        trade_date=trade_date,
        side=side,
        iv_30d=iv,
        iv_percentile_1y=0.60,
        realized_forecast_20d=0.25,
        implied_vol_forecast_20d=0.31,
        term_slope=0.02,
        contango=0.03,
        stock_change_1w=0.01,
        stock_change_1m=0.04,
    )


def test_core_regime_parser_handles_orats_percentage_and_decimal_fields():
    observation = CoreRegimeObservation.from_orats(
        {
            "ticker": "SPY",
            "tradeDate": "2026-01-06",
            "iv30d": 30,
            "ivPctile1y": 75,
            "orFcst20d": 0.24,
            "orIvFcst20d": 31,
            "slope": 2,
            "contango": -0.019,
            "stkPxChng1wk": 1,
            "stkPxChng1m": 4,
        },
        side="LONG",
    )

    assert observation.iv_30d == 0.30
    assert observation.iv_percentile_1y == 0.75
    assert observation.implied_vol_forecast_20d == 0.31
    assert observation.term_slope == 0.02
    assert observation.contango == -0.019
    assert observation.stock_change_1w == 0.01
    assert len(observation.vector()) == 7


def test_analog_selection_is_prior_same_ticker_same_side_and_label_free():
    current = _regime("2026-01-10", iv=0.30)
    closest = _regime("2026-01-08", iv=0.301)
    farther = _regime("2026-01-07", iv=0.40)
    matches = select_nearest_analogs(
        current,
        [
            farther,
            closest,
            _regime("2026-01-09", side="SHORT", iv=0.30),
            _regime("2026-01-11", iv=0.30),
            _regime("2026-01-09", ticker="QQQ", iv=0.30),
        ],
    )

    assert [item.observation.trade_date for item in matches] == ["2026-01-08", "2026-01-07"]
    assert matches[0].distance < matches[1].distance


def test_earnings_blocks_every_strategy_and_dividend_blocks_short_calls():
    events = parse_orats_events(
        "SPY",
        [{"ticker": "SPY", "earnDate": "2026-01-08", "exDivDate": "2026-01-10"}],
    )
    earnings = evaluate_event_exclusions(
        "SPY", "LONG_CALL", "2026-01-06", "2026-01-09", events
    )
    assert not earnings.eligible
    assert earnings.reasons == ("EARNINGS_WINDOW:2026-01-08",)

    short_call = evaluate_event_exclusions(
        "SPY", "BEAR_CALL_CREDIT", "2026-01-09", "2026-01-10", events
    )
    assert not short_call.eligible
    assert short_call.reasons == ("SHORT_CALL_DIVIDEND_ASSIGNMENT_WINDOW:2026-01-10",)

    long_put = evaluate_event_exclusions(
        "SPY", "LONG_PUT", "2026-01-09", "2026-01-10", events
    )
    assert long_put.eligible


def test_orats_unknown_event_date_sentinel_is_not_treated_as_an_event():
    assert parse_orats_events("SPY", [{"ticker": "SPY", "nextErn": "0000-00-00"}]) == ()
