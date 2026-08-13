"""Strangle lane: quote construction and outcome scoring.

The strangle is the structure the volatility research validated, so its
economics have to be right or the whole lane is measuring fiction.
"""

import uwos.options_pattern_pipeline_v1.core as core


def _quote(symbol, ticker, option_type, strike, *, bid=2.0, ask=2.2, stock=100.0, dte=45):
    return {
        "option_symbol": symbol,
        "ticker": ticker,
        "option_type": option_type,
        "strike": strike,
        "expiry": "2026-09-18",
        "dte": dte,
        "bid": bid,
        "ask": ask,
        "volume": 500,
        "open_interest": 300,
        "spread_pct": (ask - bid) / ask,
        "stock_close": stock,
        "iv": 0.5,
        "direction": "bullish" if option_type == "call" else "bearish",
    }


def _chain():
    return {
        "C105": _quote("C105", "ABC", "call", 105),
        "C110": _quote("C110", "ABC", "call", 110),
        "P95": _quote("P95", "ABC", "put", 95),
        "P90": _quote("P90", "ABC", "put", 90),
    }


def test_strangle_buys_the_wings_closest_to_five_percent_out():
    built = core.select_best_long_strangles(_chain())
    strangle = built[("ABC", "neutral")]
    assert strangle["strike"] == 105
    assert strangle["long_strike"] == 95
    assert [(leg["action"], leg["option_type"]) for leg in strangle["legs"]] == [
        ("BUY", "call"),
        ("BUY", "put"),
    ]


def test_strangle_is_non_directional_and_risks_only_the_debit():
    strangle = core.select_best_long_strangles(_chain())[("ABC", "neutral")]
    assert strangle["direction"] == "neutral"
    assert strangle["strategy_kind"] == "long_strangle"
    # debit is 2.2 + 2.2, and a long option cannot lose more than it cost
    assert strangle["ask"] == 4.4
    assert abs(strangle["max_risk"] - 440.0) < 1e-6


def test_breakeven_is_the_nearer_wing_plus_the_premium_paid():
    strangle = core.select_best_long_strangles(_chain())[("ABC", "neutral")]
    # up: 105 + 4.40 = 109.40 -> 9.4% ; down: 95 - 4.40 = 90.60 -> 9.4%
    assert abs(strangle["breakeven_move_pct"] - 0.094) < 1e-6


def test_strangle_rejected_when_the_move_required_is_implausible():
    chain = _chain()
    for key in chain:
        chain[key]["bid"] = 14.0
        chain[key]["ask"] = 15.0
        chain[key]["spread_pct"] = 1.0 / 15.0
    # a 20 point debit on a 100 dollar stock needs a move the 5-day horizon
    # will not deliver, and that bucket lost money in the backtest
    assert core.select_best_long_strangles(chain) == {}


def test_strangle_rejected_when_a_wing_is_illiquid():
    chain = _chain()
    chain["P95"]["volume"] = 10
    chain["P90"]["volume"] = 10
    assert core.select_best_long_strangles(chain) == {}


def test_strangle_rejected_outside_the_thirty_to_sixty_day_band():
    for dte in (14, 90):
        chain = {k: dict(v, dte=dte) for k, v in _chain().items()}
        assert core.select_best_long_strangles(chain) == {}


def test_strangle_entry_pays_both_spreads():
    single = core.signal_entry_slippage_dollars(
        {"strategy_kind": "long_option", "entry_bid": 2.0, "entry_ask": 2.2}
    )
    both = core.signal_entry_slippage_dollars(
        {"strategy_kind": "long_strangle", "entry_bid": 4.0, "entry_ask": 4.4, "quote_spread": 0.4}
    )
    assert abs(both - 2 * single) < 1e-9


def test_strangle_cost_model_charges_two_legs_round_trip():
    strangle = core.scoring_cost_model("long_strangle")
    long_option = core.scoring_cost_model("long_option")
    assert strangle["round_trip_fees"] > long_option["round_trip_fees"]
    assert strangle["exit_fill_assumption"] == "future_call_bid_plus_future_put_bid"


def test_earnings_window_filter_covers_the_strangle_lane():
    config = {"long_vol_earnings_window_days": [10, 45]}
    rows = [
        {"ticker": "IN", "strategy_kind": "long_strangle", "earnings_dte": 20, "status": "AUTO_APPROVED"},
        {"ticker": "OUT", "strategy_kind": "long_strangle", "earnings_dte": 3, "status": "AUTO_APPROVED"},
    ]
    kept, rejected = core.apply_long_vol_earnings_window(rows, config)
    assert [r["ticker"] for r in kept] == ["IN"]
    assert [r["ticker"] for r in rejected] == ["OUT"]
    assert "LONG_VOL_EARNINGS_WINDOW" in rejected[0]["block_reasons"]


def _snapshot(date, quotes, close):
    return core.Snapshot(
        signal_date=date,
        source_files=[],
        skipped_sources=[],
        features={"ABC": {"ticker": "ABC", "close": close}},
        option_quotes=quotes,
        best_options={},
        market_regime={"regime": "MIXED"},
        counts={},
    )


def _strangle_signal():
    return {
        "date": "2026-07-01",
        "ticker": "ABC",
        "direction": "neutral",
        "pattern_family": "VOL_PREMIUM_EXPANSION",
        "strategy_kind": "long_strangle",
        "close": 100.0,
        "entry_ask": 4.4,
        "entry_bid": 4.0,
        "quote_spread": 0.4,
        "legs_json": core.stable_json(
            [
                {"action": "BUY", "option_symbol": "C105", "option_type": "call", "strike": 105},
                {"action": "BUY", "option_symbol": "P95", "option_type": "put", "strike": 95},
            ]
        ),
    }


def _score(exit_quotes, exit_close):
    entry = {"C105": {"bid": 2.0, "ask": 2.2}, "P95": {"bid": 2.0, "ask": 2.2}}
    snapshots = {
        "2026-07-01": _snapshot("2026-07-01", entry, 100.0),
        "2026-07-08": _snapshot("2026-07-08", exit_quotes, exit_close),
    }
    return core.score_signal_horizon(
        _strangle_signal(), snapshots, ["2026-07-01", "2026-07-08"], "split", "sample", 1
    )


def test_strangle_exit_sells_both_wings_at_the_bid():
    outcome = _score({"C105": {"bid": 9.0, "ask": 9.2}, "P95": {"bid": 0.10, "ask": 0.20}}, 112.0)
    assert outcome["status"] == "SCORED"
    assert abs(outcome["exit_bid"] - 9.10) < 1e-9
    assert outcome["win"] == 1
    # (9.10 - 4.40) * 100 net of fees and slippage, over the debit at risk
    assert abs(outcome["net_r"] - 0.9374) < 1e-3


def test_strangle_loses_nearly_everything_when_the_tape_does_not_move():
    flat = {"bid": 0.05, "ask": 0.10}
    outcome = _score({"C105": flat, "P95": flat}, 100.5)
    assert outcome["win"] == 0
    assert outcome["net_r"] < -0.95


def test_strangle_move_is_scored_without_direction():
    up = _score({"C105": {"bid": 9.0, "ask": 9.2}, "P95": {"bid": 0.10, "ask": 0.20}}, 112.0)
    down = _score({"C105": {"bid": 0.10, "ask": 0.20}, "P95": {"bid": 9.0, "ask": 9.2}}, 88.0)
    assert up["stock_proxy_move"] > 0
    assert down["stock_proxy_move"] > 0
    assert abs(up["net_r"] - down["net_r"]) < 1e-9
