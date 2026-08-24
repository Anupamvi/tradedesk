import sqlite3
import zipfile

import pandas as pd
import pytest

import uwos.pattern_analysis_v2.engine as pattern_core
import uwos.pattern_analysis_v2.managed as managed_module
from uwos.pattern_analysis_v2.engine import (
    PriceRow,
    PriceSignal,
    SourceRef,
    build_chain_source_index,
    build_chain_sqlite_index,
    adjusted_close_series,
    build_current_board,
    derive_price_features,
    directional_score,
    generate_price_signals,
    load_bot_eod_flow_history,
    iter_chain_source_index,
    load_entry_option_quotes,
    option_outcome_coverage,
    option_outcome,
    option_outcome_missing_reason,
    option_outcome_status,
    option_gate_sets,
    option_quote_is_horizon_safe,
    parse_occ,
    price_gate_sets,
    price_outcome_rows,
    render_report,
    realized_stock_return,
    stock_outcome,
)
from uwos.pattern_analysis_v2.managed import (
    ManagedConfig,
    QuoteCache,
    _signals_for_day,
    _random_control_names,
    _limit_selected_names,
    _select_contracts,
    _position_52w_features,
    _marketcap_eligible,
    _expiration_exit_value,
    _position_result,
    build_chain_source_index as managed_build_chain_source_index,
    build_quote_source_index,
    eligible_sessions,
    managed_calibration_rows,
    frozen_holdout_calibration,
    named_mover_audit,
    load_cached_bot_flow,
    load_quotes_for_session,
    managed_price_research,
    managed_price_validation_rows,
    managed_regime_rows,
    managed_selection_audit,
    managed_validation_rows,
    predeclared_managed_selection_candidates,
    run_managed_strategy,
    _session_shifted_price,
    summarize_managed,
)


def make_row(day, ticker, close, previous, volume=1000.0):
    return PriceRow(
        date=day,
        ticker=ticker,
        close=close,
        high=close * 1.01,
        low=close * 0.99,
        prev_close=previous,
        volume=volume,
        avg30_volume=1000.0,
        sector="Technology",
        call_premium=1000.0,
        put_premium=500.0,
        bullish_premium=1000.0,
        bearish_premium=500.0,
        call_volume=100.0,
        put_volume=50.0,
        avg30_call_volume=100.0,
        avg30_put_volume=50.0,
        iv_rank=50.0,
        implied_move_perc=0.05,
        next_earnings_date="",
        source="fixture",
    )


def test_features_do_not_use_future_rows():
    dates = [f"2026-01-{day:02d}" for day in range(1, 23)]
    rows = {}
    price = 100.0
    for index, day in enumerate(dates):
        previous = price
        price += 1.0
        rows[day] = make_row(day, "AAA", price, previous)
    history = {"AAA": {row.date: row for row in rows.values()}}
    before = derive_price_features(history, dates)
    history["AAA"][dates[-1]].close = 100000.0
    after = derive_price_features(history, dates)
    assert before[dates[-2]]["AAA"] == after[dates[-2]]["AAA"]


def test_price_returns_use_point_in_time_split_adjustment():
    dates = ["2026-01-02", "2026-01-05", "2026-01-06"]
    rows = {
        dates[0]: make_row(dates[0], "SPLIT", 100.0, 100.0),
        # A 10-for-1 split: UW exposes the current raw close at 10 and the
        # adjusted previous close at 10, while the prior raw row is 100.
        dates[1]: make_row(dates[1], "SPLIT", 10.0, 10.0),
        dates[2]: make_row(dates[2], "SPLIT", 11.0, 10.0),
    }
    adjusted, factors = adjusted_close_series(list(rows.values()))
    assert adjusted == [10.0, 10.0, 11.0]
    assert factors[1] == 0.1
    features = derive_price_features({"SPLIT": rows}, dates)
    assert features[dates[1]]["SPLIT"]["return_1d"] == 0.0
    assert abs(features[dates[2]]["SPLIT"]["return_1d"] - 0.1) < 1e-12
    assert abs(realized_stock_return(features, dates, "SPLIT", dates[0], dates[1])) < 1e-12
    signal = PriceSignal(
        date=dates[0],
        ticker="SPLIT",
        direction="bullish",
        family="TEST",
        role="forward_setup",
        score=1.0,
        reasons=[],
        feature=features[dates[0]]["SPLIT"],
    )
    assert abs(stock_outcome(features, dates, signal, 1)) < 1e-12


def test_bot_eod_flow_is_same_day_directional_confirmation():
    dates = [f"2026-01-{day:02d}" for day in range(1, 6)]
    rows = {}
    price = 100.0
    for day in dates:
        previous = price
        price += 1.0
        rows[day] = make_row(day, "AAA", price, previous)
    features = derive_price_features(
        {"AAA": rows},
        dates,
        {
            dates[-1]: {
                "AAA": {
                    "flow_call_ask_premium": 800.0,
                    "flow_put_bid_premium": 200.0,
                    "flow_call_bid_premium": 100.0,
                    "flow_put_ask_premium": 100.0,
                    "flow_total_premium": 1200.0,
                }
            }
        },
    )
    assert features[dates[-1]]["AAA"]["bot_eod_flow_bias"] == 0.6666666666666666


def test_fallback_whale_flow_is_loaded_when_bot_eod_is_absent(tmp_path):
    date_dir = tmp_path / "2026-01-05"
    date_dir.mkdir()
    (date_dir / "whale_trades_filtered.csv").write_text(
        "underlying_symbol,side,option_type,premium,price,size,canceled,upstream_condition_detail\n"
        "AAA,ask,call,125000,1.25,1000,f,auto\n"
        "AAA,bid,put,25000,0.25,1000,f,auto\n"
        "BBB,ask,call,50000,0.50,1000,t,auto\n",
        encoding="utf-8",
    )
    flow, metadata = load_bot_eod_flow_history(tmp_path, ["2026-01-05"])
    assert metadata["bot_eod_source_dates"] == 0
    assert metadata["option_flow_fallback_source_dates"] == 1
    assert flow["2026-01-05"]["AAA"]["flow_source"] == "whale_filtered"
    assert flow["2026-01-05"]["AAA"]["flow_call_ask_premium"] == 125000.0
    assert flow["2026-01-05"]["AAA"]["flow_put_bid_premium"] == 25000.0
    assert "BBB" not in flow["2026-01-05"]


def test_cached_bot_flow_is_loaded_with_bias_and_provenance(tmp_path):
    cache_dir = tmp_path / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
    cache_dir.mkdir(parents=True)
    (cache_dir / "bot_eod_flow_by_ticker_2026-01-05.csv").write_text(
        "date,ticker,flow_call_ask_premium,flow_put_ask_premium,flow_total_premium,flow_call_trade_count,flow_put_trade_count\n"
        "2026-01-05,AAA,800000,200000,1000000,8,2\n"
        "2026-01-05,BBB,1000,900,1900,1,1\n",
        encoding="utf-8",
    )
    flow, metadata = load_cached_bot_flow(tmp_path, "2026-01-01", "2026-01-06")
    assert metadata["bot_flow_status"] == "CACHE_LOADED"
    assert metadata["bot_flow_source_dates"] == ["2026-01-05"]
    assert metadata["bot_flow_rows"] == 2
    assert flow.loc[flow["ticker"].eq("AAA"), "bot_flow_bias"].iloc[0] == 0.6


def test_bot_flow_quantile_applies_premium_floor_and_direction():
    rows = []
    for index in range(12):
        rows.append(
            {
                "ticker": f"T{index:02d}",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 0.50,
                "close": 100.0,
                "sector": "Technology",
                "iv_rank": 50.0,
                "bot_flow_bias": (index - 6) / 10.0,
                "bot_flow_total_premium": 200_000.0 if index != 11 else 1_000.0,
            }
        )
    frame = pd.DataFrame(rows)
    config = ManagedConfig(
        signal_rule="bot_flow_quantile",
        direction="call",
        signal_direction="call",
        top_quantile=0.90,
        min_sector_names=12,
    )
    selected = _signals_for_day(frame, config)
    names = {name for values in selected.values() for name in values}
    assert "T10" in names
    assert "T11" not in names


def test_random_control_excludes_signal_names():
    panel = pd.DataFrame(
        [
            {
                "ticker": ticker,
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 0.5,
                "close": 100.0,
                "sector": "Technology",
                "iv_rank": 30.0,
            }
            for ticker in ("AAA", "BBB", "CCC")
        ]
    )
    selected = {"Technology": {"AAA"}}

    control = _random_control_names(
        panel,
        selected,
        ManagedConfig(min_sector_names=1),
        seed=1,
    )

    assert control["Technology"]
    assert control["Technology"].isdisjoint(selected["Technology"])


def test_earnings_flow_uses_only_known_event_window_and_directional_flow():
    rows = []
    for index in range(12):
        rows.append(
            {
                "ticker": f"E{index:02d}",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 0.50,
                "close": 100.0,
                "sector": "Technology",
                "iv_rank": 50.0,
                "days_to_earnings": 5.0 if index < 11 else 20.0,
                "implied_move_perc": 0.06 if index != 10 else 0.01,
                "flow_bias": (index - 6) / 10.0,
            }
        )
    config = ManagedConfig(
        signal_rule="earnings_flow",
        direction="call",
        signal_direction="call",
        top_quantile=0.90,
        min_sector_names=12,
        min_implied_move_perc=0.05,
    )
    selected = _signals_for_day(pd.DataFrame(rows), config)
    names = {name for values in selected.values() for name in values}
    assert "E09" in names
    assert "E10" not in names
    assert "E11" not in names


def test_managed_straddle_selects_two_same_expiry_legs_and_intrinsic_value():
    quotes = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "option_type": "C",
                "option_symbol": "AAA260220C00100000",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "dte": 40.0,
                "curr_oi": 500.0,
                "spread_pct": 0.05,
                "last_bid": 1.90,
                "last_ask": 2.00,
                "stock_price": 100.0,
            },
            {
                "ticker": "AAA",
                "option_type": "P",
                "option_symbol": "AAA260220P00100000",
                "expiry": "2026-02-20",
                "strike": 100.0,
                "dte": 40.0,
                "curr_oi": 500.0,
                "spread_pct": 0.05,
                "last_bid": 1.90,
                "last_ask": 2.00,
                "stock_price": 100.0,
            },
        ]
    )
    config = ManagedConfig(
        structure="long_straddle",
        min_sector_names=1,
        min_dte=20,
        max_dte=60,
        target_dte=40,
        moneyness=1.0,
    )
    selected = _select_contracts(quotes, {"Technology": {"AAA"}}, config, set())
    assert len(selected) == 1
    assert selected[0]["structure"] == "long_straddle"
    assert selected[0]["entry_ask"] == pytest.approx(4.0)
    assert selected[0]["second_option_symbol"] == "AAA260220P00100000"
    assert _expiration_exit_value(selected[0], 90.0) == pytest.approx(10.0)


def test_managed_straddle_walks_both_legs_and_charges_four_leg_fees():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    rows = []
    for day in sessions:
        for index in range(12):
            rows.append(
                {
                    "date": day,
                    "ticker": f"T{index:02d}",
                    "sector": "Technology",
                    "issue_type": "Common Stock",
                    "marketcap": 3_000_000_000.0,
                    "close": 100.0,
                    "avg30_volume": 1_000_000.0,
                    "position_52w": index / 11.0,
                    "days_to_earnings": 5.0 if day == "2026-01-02" else 20.0,
                    "implied_move_perc": 0.10 if index >= 10 else 0.01,
                }
            )
    symbols = {
        ticker: (f"{ticker}260220C00100000", f"{ticker}260220P00100000")
        for ticker in ("T10", "T11")
    }

    def quotes(day):
        if day == "2026-01-05":
            result = []
            for ticker, (call_symbol, put_symbol) in symbols.items():
                result.extend(
                    [
                        {
                            "option_symbol": call_symbol,
                            "ticker": ticker,
                            "option_type": "C",
                            "strike": 100.0,
                            "expiry": "2026-02-20",
                            "dte": 40.0,
                            "last_bid": 1.90,
                            "last_ask": 2.00,
                            "curr_oi": 500.0,
                            "stock_price": 100.0,
                            "spread_pct": 0.05,
                            "source_date": day,
                        },
                        {
                            "option_symbol": put_symbol,
                            "ticker": ticker,
                            "option_type": "P",
                            "strike": 100.0,
                            "expiry": "2026-02-20",
                            "dte": 40.0,
                            "last_bid": 1.90,
                            "last_ask": 2.00,
                            "curr_oi": 500.0,
                            "stock_price": 100.0,
                            "spread_pct": 0.05,
                            "source_date": day,
                        },
                    ]
                )
            return pd.DataFrame(result)
        if day == "2026-01-06":
            result = []
            for call_symbol, put_symbol in symbols.values():
                result.extend(
                    [
                        {"option_symbol": call_symbol, "last_bid": 3.00, "last_ask": 3.10},
                        {"option_symbol": put_symbol, "last_bid": 3.00, "last_ask": 3.10},
                    ]
                )
            return pd.DataFrame(result)
        return pd.DataFrame()

    config = ManagedConfig(
        name="EARNINGS_STRADDLE_FIXTURE",
        signal_rule="earnings_event",
        structure="long_straddle",
        direction="call",
        signal_direction="call",
        top_quantile=0.90,
        min_sector_names=12,
        min_dte=20,
        max_dte=60,
        target_dte=40,
        moneyness=1.0,
        earnings_min_days=1,
        earnings_max_days=10,
        profit_target=0.50,
        max_hold_sessions=2,
    )
    result = run_managed_strategy(pd.DataFrame(rows), sessions, quotes, config)
    assert len(result) == 2
    assert set(result["structure"]) == {"long_straddle"}
    assert set(result["exit_reason"]) == {"profit_target"}
    assert result["entry_ask"].tolist() == pytest.approx([4.0, 4.0])
    assert result["exit_bid"].tolist() == pytest.approx([6.0, 6.0])
    assert result["gross_pnl"].tolist() == pytest.approx([200.0, 200.0])
    assert result["net_pnl"].tolist() == pytest.approx([194.0, 194.0])


def test_managed_iron_condor_uses_conservative_close_debit_and_eight_leg_fees():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]
    rows = []
    for day in sessions:
        for index in range(12):
            rows.append(
                {
                    "date": day,
                    "ticker": f"T{index:02d}",
                    "sector": "Technology",
                    "issue_type": "Common Stock",
                    "marketcap": 3_000_000_000.0,
                    "close": 100.0,
                    "avg30_volume": 1_000_000.0,
                    "position_52w": index / 11.0,
                    "days_to_earnings": 5.0 if day == "2026-01-02" else 20.0,
                    "implied_move_perc": 0.10 if index == 11 else 0.01,
                }
            )
    leg_specs = [
        ("T11260618P00090000", "P", 90.0, 0.10, 0.20),
        ("T11260618P00095000", "P", 95.0, 1.00, 1.10),
        ("T11260618C00105000", "C", 105.0, 1.00, 1.10),
        ("T11260618C00110000", "C", 110.0, 0.10, 0.20),
    ]

    def quotes(day):
        if day == "2026-01-05":
            return pd.DataFrame(
                [
                    {
                        "option_symbol": symbol,
                        "ticker": "T11",
                        "option_type": option_type,
                        "strike": strike,
                        "expiry": "2026-06-18",
                        "dte": 164.0,
                        "last_bid": bid,
                        "last_ask": ask,
                        "curr_oi": 500.0,
                        "stock_price": 100.0,
                        "spread_pct": 0.10,
                        "source_date": day,
                    }
                    for symbol, option_type, strike, bid, ask in leg_specs
                ]
            )
        if day == "2026-01-06":
            return pd.DataFrame(
                [
                    {"option_symbol": symbol, "last_bid": bid, "last_ask": ask}
                    for symbol, _option_type, _strike, bid, ask in [
                        (leg_specs[0][0], "P", 90.0, 0.10, 0.20),
                        (leg_specs[1][0], "P", 95.0, 0.20, 0.30),
                        (leg_specs[2][0], "C", 105.0, 0.20, 0.30),
                        (leg_specs[3][0], "C", 110.0, 0.10, 0.20),
                    ]
                ]
            )
        return pd.DataFrame()

    config = ManagedConfig(
        name="IRON_CONDOR_FIXTURE",
        signal_rule="earnings_event",
        direction="neutral",
        signal_direction="call",
        structure="iron_condor",
        top_quantile=0.90,
        min_sector_names=12,
        min_dte=100,
        max_dte=200,
        target_dte=164,
        max_hold_sessions=2,
        profit_target=0.50,
    )
    result = run_managed_strategy(pd.DataFrame(rows), sessions, quotes, config)
    assert len(result) == 1
    assert result.iloc[0]["structure"] == "iron_condor"
    assert result.iloc[0]["direction"] == "neutral"
    assert result.iloc[0]["entry_credit"] == pytest.approx(1.60)
    assert result.iloc[0]["exit_reason"] == "profit_target"
    assert result.iloc[0]["exit_bid"] == pytest.approx(0.80)
    assert result.iloc[0]["gross_pnl"] == pytest.approx(80.0)
    assert result.iloc[0]["net_pnl"] == pytest.approx(68.0)
    assert _expiration_exit_value(result.iloc[0].to_dict(), 120.0) == pytest.approx(5.0)


def test_price_first_lane_surfaces_symmetric_breakout_and_breakdown():
    dates = [f"2026-01-{day:02d}" for day in range(1, 31)]
    up = {}
    down = {}
    up_price = 100.0
    down_price = 100.0
    for index, day in enumerate(dates):
        up_previous = up_price
        down_previous = down_price
        if index >= 24:
            up_price *= 1.025
            down_price *= 0.975
        else:
            up_price += 0.1
            down_price -= 0.1
        up[day] = make_row(day, "UP", up_price, up_previous, 2500.0 if index >= 24 else 1000.0)
        down[day] = make_row(day, "DOWN", down_price, down_previous, 2500.0 if index >= 24 else 1000.0)
    features = derive_price_features({"UP": up, "DOWN": down}, dates)
    signals = generate_price_signals(features, dates)
    families = {(signal.ticker, signal.direction, signal.family) for signal in signals}
    assert any(ticker == "UP" and direction == "bullish" for ticker, direction, _ in families)
    assert any(ticker == "DOWN" and direction == "bearish" for ticker, direction, _ in families)


def test_directional_score_does_not_reward_adverse_relative_strength():
    aligned = directional_score("bullish", (2.0,), (0.5, 0.25))
    adverse = directional_score("bullish", (2.0,), (-0.5, -0.25))
    assert aligned == 2.75
    assert adverse == 2.0
    assert directional_score("bearish", (2.0,), (-0.5, -0.25)) == 2.75


def test_same_day_event_detection_does_not_require_volume_confirmation():
    dates = [f"2026-01-{day:02d}" for day in range(1, 26)]
    rows = {}
    price = 100.0
    for index, day in enumerate(dates):
        previous = price
        price = 105.0 if index == len(dates) - 1 else 100.0
        rows[day] = make_row(day, "SHOCK", price, previous, 10.0)
    features = derive_price_features({"SHOCK": rows}, dates)
    signals = generate_price_signals(features, dates)
    assert any(
        signal.ticker == "SHOCK"
        and signal.date == dates[-1]
        and signal.family == "EVENT_SHOCK"
        and signal.role == "same_day_event"
        and signal.direction == "bullish"
        for signal in signals
    )
    assert any(
        signal.ticker == "SHOCK"
        and signal.date == dates[-1]
        and signal.family == "POST_EVENT_CONTINUATION"
        and signal.role == "post_event_setup"
        and signal.direction == "bullish"
        for signal in signals
    )
    assert any(
        signal.ticker == "SHOCK"
        and signal.date == dates[-1]
        and signal.family == "POST_EVENT_MEAN_REVERSION"
        and signal.role == "post_event_setup"
        and signal.direction == "bearish"
        for signal in signals
    )


def test_occ_parser_and_option_outcome_use_conservative_entry_exit_quotes():
    parsed = parse_occ("AAPL260918C00330000")
    assert parsed["ticker"] == "AAPL"
    assert parsed["strike"] == 330.0
    signal = PriceSignal(
        date="2026-01-02",
        ticker="AAPL",
        direction="bullish",
        family="TEST",
        role="forward_setup",
        score=1.0,
        reasons=[],
        feature={"close": 330.0},
    )
    entry = {
        "date": signal.date,
        "option_symbol": "AAPL260918C00330000",
        "ask": 5.0,
    }
    history = {
        "AAPL260918C00330000": {
            "2026-01-07": {"bid": 6.5},
        }
    }
    outcome = option_outcome(entry, history, ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"], 3)
    assert outcome["exit_bid"] == 6.5
    assert outcome["net_pnl"] == 147.0
    assert outcome["net_R"] > 0


def test_managed_quote_index_uses_actual_quote_date_across_holiday_gap(tmp_path):
    day = tmp_path / "2026-04-23"
    day.mkdir()
    path = day / "chain-oi-changes-2026-04-23.zip"
    csv = (
        "option_symbol,last_date,last_bid,last_ask,curr_oi,stock_price,dte\n"
        "AAA260618C00100000,2026-04-22,1.0,1.1,100,,57\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("chain-oi-changes-2026-04-23.csv", csv)
    source_index, metadata = managed_build_chain_source_index(tmp_path, "2026-04-22", "2026-04-23")
    assert "2026-04-22" in source_index
    assert source_index["2026-04-22"][0].source_date == "2026-04-23"
    assert metadata["chain_quote_date_count"] == 1
    quotes = load_quotes_for_session(
        "2026-04-22", source_index, underlying_prices=pd.Series({"AAA": 100.0})
    )
    assert len(quotes) == 1
    assert quotes.iloc[0]["option_symbol"] == "AAA260618C00100000"
    assert quotes.iloc[0]["stock_price"] == 100.0


def test_managed_quote_index_uses_same_day_hot_chain_quotes(tmp_path):
    day = tmp_path / "2026-04-22"
    day.mkdir()
    path = day / "hot-chains-2026-04-22.zip"
    csv = (
        "option_symbol,date,bid,ask,open_interest,volume\n"
        "AAA260618C00100000,2026-04-22,1.0,1.1,100,25\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("hot-chains-2026-04-22.csv", csv)
    source_index, metadata = build_quote_source_index(tmp_path, "2026-04-22", "2026-04-22")
    assert source_index["2026-04-22"][0].kind == "hot"
    assert metadata["quote_source_kind_counts"] == {"hot": 1, "chain_oi": 0}
    quotes = load_quotes_for_session(
        "2026-04-22", source_index, underlying_prices={"AAA": 100.0}
    )
    assert len(quotes) == 1
    assert quotes.iloc[0]["dte"] == 57
    assert quotes.iloc[0]["source_kind"] == "hot"
    assert quotes.iloc[0]["stock_price"] == 100.0


def test_managed_quote_cache_reuses_only_matching_source_namespace(tmp_path):
    day = tmp_path / "2026-04-22"
    day.mkdir()
    path = day / "hot-chains-2026-04-22.zip"
    csv = (
        "option_symbol,date,bid,ask,open_interest,volume\n"
        "AAA260618C00100000,2026-04-22,1.0,1.1,100,25\n"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("hot-chains-2026-04-22.csv", csv)
    source_index, _ = build_quote_source_index(tmp_path, "2026-04-22", "2026-04-22")
    cache_root = tmp_path / "quote-cache"
    first = QuoteCache(
        source_index,
        underlying_prices_by_date={"2026-04-22": {"AAA": 100.0}},
        materialized_dir=cache_root,
        cache_key="fixture",
    )
    first_frame = first.get("2026-04-22")
    assert first.cache_misses == 1
    second = QuoteCache(
        source_index,
        underlying_prices_by_date={"2026-04-22": {"AAA": 100.0}},
        materialized_dir=cache_root,
        cache_key="fixture",
    )
    second_frame = second.get("2026-04-22")
    assert second.cache_hits == 1
    assert second.cache_misses == 0
    pd.testing.assert_frame_equal(first_frame, second_frame)


def test_managed_strategy_excludes_report_only_dates_and_marks_target_at_bid():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]
    rows = []
    for index in range(12):
        for day in sessions:
            rows.append(
                {
                    "date": day,
                    "ticker": f"T{index:02d}",
                    "sector": "Technology",
                    "issue_type": "Common Stock",
                    "marketcap": 3_000_000_000.0,
                    "close": 100.0,
                    "avg30_volume": 1_000_000.0,
                    "position_52w": index / 11.0,
                }
            )
    panel = pd.DataFrame(rows)
    symbols = {
        "T10": "T10260618C00105000",
        "T11": "T11260618C00105000",
    }

    def quotes(day):
        if day == "2026-01-05":
            return pd.DataFrame(
                [
                    {
                        "option_symbol": symbol,
                        "ticker": ticker,
                        "option_type": "C",
                        "strike": 105.0,
                        "expiry": "2026-06-18",
                        "dte": 164.0,
                        "last_bid": 0.9,
                        "last_ask": 1.0,
                        "curr_oi": 100.0,
                        "stock_price": 100.0,
                        "spread_pct": 0.10,
                        "source_date": day,
                    }
                    for ticker, symbol in symbols.items()
                ]
            )
        if day == "2026-01-06":
            return pd.DataFrame(
                [
                    {"option_symbol": symbol, "last_bid": 1.6}
                    for symbol in symbols.values()
                ]
            )
        return pd.DataFrame()

    config = ManagedConfig(
        min_sector_names=12,
        min_dte=100,
        max_dte=200,
        target_dte=164,
        max_hold_sessions=2,
    )
    result = run_managed_strategy(panel, sessions, quotes, config)
    assert len(result) == 2
    assert set(result["status"]) == {"SCORED"}
    assert set(result["exit_reason"]) == {"profit_target"}
    assert result["exit_bid"].tolist() == [1.5, 1.5]
    assert (result["net_pnl"] > 0).all()


def test_managed_strategy_scores_expired_contract_from_underlying_intrinsic_value():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    rows = []
    for day in sessions:
        for index in range(12):
            rows.append(
                {
                    "date": day,
                    "ticker": f"T{index:02d}",
                    "sector": "Technology",
                    "issue_type": "Common Stock",
                    "marketcap": 3_000_000_000.0,
                    "close": 110.0 if day == "2026-01-06" else 100.0,
                    "avg30_volume": 1_000_000.0,
                    "position_52w": index / 11.0,
                }
            )
    symbol = "T11260106C00100000"

    def quotes(day):
        if day == "2026-01-05":
            return pd.DataFrame(
                [
                    {
                        "option_symbol": symbol,
                        "ticker": "T11",
                        "option_type": "C",
                        "strike": 100.0,
                        "expiry": "2026-01-06",
                        "dte": 1.0,
                        "last_bid": 4.0,
                        "last_ask": 5.0,
                        "curr_oi": 100.0,
                        "stock_price": 100.0,
                        "spread_pct": 0.10,
                        "source_date": day,
                    }
                ]
            )
        return pd.DataFrame()

    config = ManagedConfig(
        min_sector_names=12,
        min_dte=1,
        max_dte=5,
        target_dte=1,
        moneyness=1.0,
        max_hold_sessions=20,
    )
    result = run_managed_strategy(pd.DataFrame(rows), sessions, quotes, config)
    assert len(result) == 1
    assert result.iloc[0]["status"] == "SCORED"
    assert result.iloc[0]["exit_reason"] == "expiration_intrinsic"
    assert result.iloc[0]["exit_bid"] == pytest.approx(10.0)
    assert result.iloc[0]["exit_underlying_price"] == pytest.approx(110.0)


def test_managed_strategy_uses_only_one_session_last_observed_quote_at_time_stop():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    rows = [
        {
            "date": day,
            "ticker": f"T{index:02d}",
            "sector": "Technology",
            "issue_type": "Common Stock",
            "marketcap": 3_000_000_000.0,
            "close": 100.0,
            "avg30_volume": 1_000_000.0,
            "position_52w": index / 11.0,
        }
        for day in sessions
        for index in range(12)
    ]
    symbol = "T11260618C00105000"

    def quotes(day):
        if day == "2026-01-05":
            return pd.DataFrame(
                [
                    {
                        "option_symbol": symbol,
                        "ticker": "T11",
                        "option_type": "C",
                        "strike": 105.0,
                        "expiry": "2026-06-18",
                        "dte": 164.0,
                        "last_bid": 0.9,
                        "last_ask": 1.0,
                        "curr_oi": 100.0,
                        "stock_price": 100.0,
                        "spread_pct": 0.10,
                        "source_date": day,
                    }
                ]
            )
        if day == "2026-01-06":
            return pd.DataFrame(
                [{"option_symbol": symbol, "last_bid": 1.2, "last_ask": 1.3}]
            )
        return pd.DataFrame()

    config = ManagedConfig(
        min_sector_names=12,
        min_dte=100,
        max_dte=200,
        target_dte=164,
        max_hold_sessions=2,
    )
    result = run_managed_strategy(pd.DataFrame(rows), sessions, quotes, config)
    assert len(result) == 1
    assert result.iloc[0]["status"] == "SCORED"
    assert result.iloc[0]["exit_reason"] == "time_stop_last_observed_quote"
    assert result.iloc[0]["exit_date"] == "2026-01-06"
    assert result.iloc[0]["exit_bid"] == pytest.approx(1.2)


def test_managed_debit_vertical_uses_conservative_two_leg_quotes_and_fees():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]
    rows = [
        {
            "date": day,
            "ticker": f"T{index:02d}",
            "sector": "Technology",
            "issue_type": "Common Stock",
                    "marketcap": 3_000_000_000.0,
            "close": 100.0,
            "avg30_volume": 1_000_000.0,
            "position_52w": index / 11.0,
        }
        for day in sessions
        for index in range(12)
    ]
    entry_symbols = {
        "T10": ("T10260618C00105000", "T10260618C00112000"),
        "T11": ("T11260618C00105000", "T11260618C00112000"),
    }

    def quotes(day):
        if day == "2026-01-05":
            rows = []
            for ticker, (long_symbol, short_symbol) in entry_symbols.items():
                rows.extend(
                    [
                        {
                            "option_symbol": long_symbol,
                            "ticker": ticker,
                            "option_type": "C",
                            "strike": 105.0,
                            "expiry": "2026-06-18",
                            "dte": 164.0,
                            "last_bid": 0.9,
                            "last_ask": 1.0,
                            "curr_oi": 100.0,
                            "stock_price": 100.0,
                            "spread_pct": 0.10,
                            "source_date": day,
                        },
                        {
                            "option_symbol": short_symbol,
                            "ticker": ticker,
                            "option_type": "C",
                            "strike": 112.0,
                            "expiry": "2026-06-18",
                            "dte": 164.0,
                            "last_bid": 0.3,
                            "last_ask": 0.4,
                            "curr_oi": 100.0,
                            "stock_price": 100.0,
                            "spread_pct": 0.10,
                            "source_date": day,
                        },
                    ]
                )
            return pd.DataFrame(rows)
        if day == "2026-01-06":
            rows = []
            for long_symbol, short_symbol in entry_symbols.values():
                rows.extend(
                    [
                        {"option_symbol": long_symbol, "last_bid": 1.5, "last_ask": 1.6},
                        {"option_symbol": short_symbol, "last_bid": 0.1, "last_ask": 0.2},
                    ]
                )
            return pd.DataFrame(rows)
        return pd.DataFrame()

    config = ManagedConfig(
        structure="debit_vertical",
        min_sector_names=12,
        min_dte=100,
        max_dte=200,
        target_dte=164,
        moneyness=1.05,
        short_moneyness=1.12,
        max_hold_sessions=2,
    )
    result = run_managed_strategy(pd.DataFrame(rows), sessions, quotes, config)
    assert len(result) == 2
    assert set(result["structure"]) == {"debit_vertical"}
    assert result["entry_debit"].tolist() == pytest.approx([0.7, 0.7])
    assert result["exit_bid"].tolist() == pytest.approx([1.05, 1.05])
    assert result["gross_pnl"].tolist() == pytest.approx([35.0, 35.0])
    assert result["net_pnl"].tolist() == pytest.approx([29.0, 29.0])


def test_managed_credit_vertical_uses_close_debit_and_max_loss_risk():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]
    rows = [
        {
            "date": day,
            "ticker": f"T{index:02d}",
            "sector": "Technology",
            "issue_type": "Common Stock",
            "marketcap": 3_000_000_000.0,
            "close": 100.0,
            "avg30_volume": 1_000_000.0,
            "position_52w": index / 11.0,
        }
        for day in sessions
        for index in range(12)
    ]
    entry_symbols = {
        "T10": ("T10260618P00090000", "T10260618P00098000"),
        "T11": ("T11260618P00090000", "T11260618P00098000"),
    }

    def quotes(day):
        if day == "2026-01-05":
            rows = []
            for ticker, (long_symbol, short_symbol) in entry_symbols.items():
                rows.extend(
                    [
                        {
                            "option_symbol": long_symbol,
                            "ticker": ticker,
                            "option_type": "P",
                            "strike": 90.0,
                            "expiry": "2026-06-18",
                            "dte": 164.0,
                            "last_bid": 0.2,
                            "last_ask": 0.3,
                            "curr_oi": 100.0,
                            "stock_price": 100.0,
                            "spread_pct": 0.10,
                            "source_date": day,
                        },
                        {
                            "option_symbol": short_symbol,
                            "ticker": ticker,
                            "option_type": "P",
                            "strike": 98.0,
                            "expiry": "2026-06-18",
                            "dte": 164.0,
                            "last_bid": 1.0,
                            "last_ask": 1.1,
                            "curr_oi": 100.0,
                            "stock_price": 100.0,
                            "spread_pct": 0.10,
                            "source_date": day,
                        },
                    ]
                )
            return pd.DataFrame(rows)
        if day == "2026-01-06":
            rows = []
            for long_symbol, short_symbol in entry_symbols.values():
                rows.extend(
                    [
                        {"option_symbol": long_symbol, "last_bid": 0.1, "last_ask": 0.2},
                        {"option_symbol": short_symbol, "last_bid": 0.1, "last_ask": 0.2},
                    ]
                )
            return pd.DataFrame(rows)
        return pd.DataFrame()

    config = ManagedConfig(
        structure="credit_vertical",
        direction="call",
        option_type="P",
        min_sector_names=12,
        min_dte=100,
        max_dte=200,
        target_dte=164,
        moneyness=0.98,
        short_moneyness=0.90,
        max_hold_sessions=2,
    )
    result = run_managed_strategy(pd.DataFrame(rows), sessions, quotes, config)
    assert len(result) == 2
    assert set(result["structure"]) == {"credit_vertical"}
    assert set(result["option_type"]) == {"P"}
    assert result["entry_credit"].tolist() == pytest.approx([0.7, 0.7])
    assert result["exit_value"].tolist() == pytest.approx([0.35, 0.35])
    assert result["gross_pnl"].tolist() == pytest.approx([35.0, 35.0])
    assert result["net_pnl"].tolist() == pytest.approx([29.0, 29.0])
    assert (result["net_R"] > 0).all()


def test_52_week_position_is_bounded_but_raw_value_is_preserved():
    frame = pd.DataFrame(
        {
            "close": [150.0, 50.0, 75.0],
            "week_52_high": [100.0, 100.0, 100.0],
            "week_52_low": [0.0, 0.0, 0.0],
        }
    )
    raw, bounded = _position_52w_features(frame)
    assert raw.tolist() == pytest.approx([1.5, 0.5, 0.75])
    assert bounded.tolist() == pytest.approx([1.0, 0.5, 0.75])


def test_credit_vertical_selector_requires_otm_short_leg_for_puts_and_calls():
    base = {
        "ticker": "AAA",
        "expiry": "2026-06-18",
        "dte": 164.0,
        "curr_oi": 100.0,
        "stock_price": 100.0,
        "spread_pct": 0.10,
        "last_bid": 2.0,
        "last_ask": 2.1,
        "source_date": "2026-01-05",
    }

    put_quotes = pd.DataFrame(
        [
            {**base, "option_symbol": "AAA260618P00100000", "option_type": "P", "strike": 100.0},
            {**base, "option_symbol": "AAA260618P00098000", "option_type": "P", "strike": 98.0},
            {**base, "option_symbol": "AAA260618P00090000", "option_type": "P", "strike": 90.0, "last_ask": 0.5},
        ]
    )
    put_config = ManagedConfig(
        name="PUT_CREDIT_SELECTOR_FIXTURE",
        direction="call",
        option_type="P",
        structure="credit_vertical",
        moneyness=1.0,
        short_moneyness=0.90,
        min_dte=100,
        max_dte=200,
        target_dte=164,
    )
    put_selected = _select_contracts(put_quotes, {"Technology": {"AAA"}}, put_config, set())
    assert len(put_selected) == 1
    assert put_selected[0]["short_strike"] < put_selected[0]["underlying_price"]
    assert put_selected[0]["strike"] < put_selected[0]["short_strike"]
    assert put_selected[0]["short_strike"] == pytest.approx(98.0)

    call_quotes = pd.DataFrame(
        [
            {**base, "option_symbol": "AAA260618C00100000", "option_type": "C", "strike": 100.0},
            {**base, "option_symbol": "AAA260618C00102000", "option_type": "C", "strike": 102.0},
            {**base, "option_symbol": "AAA260618C00110000", "option_type": "C", "strike": 110.0, "last_ask": 0.5},
        ]
    )
    call_config = ManagedConfig(
        name="CALL_CREDIT_SELECTOR_FIXTURE",
        direction="put",
        option_type="C",
        structure="credit_vertical",
        moneyness=1.0,
        short_moneyness=1.10,
        min_dte=100,
        max_dte=200,
        target_dte=164,
    )
    call_selected = _select_contracts(call_quotes, {"Technology": {"AAA"}}, call_config, set())
    assert len(call_selected) == 1
    assert call_selected[0]["short_strike"] > call_selected[0]["underlying_price"]
    assert call_selected[0]["strike"] > call_selected[0]["short_strike"]
    assert call_selected[0]["short_strike"] == pytest.approx(102.0)


def test_cash_secured_put_selector_requires_otm_put_and_uses_collateral():
    base = {
        "ticker": "AAA",
        "expiry": "2026-06-18",
        "dte": 80.0,
        "curr_oi": 100.0,
        "stock_price": 100.0,
        "spread_pct": 0.10,
        "last_ask": 2.1,
        "source_date": "2026-01-05",
    }
    quotes = pd.DataFrame(
        [
            {
                **base,
                "option_symbol": "AAA260618P00100000",
                "option_type": "P",
                "strike": 100.0,
                "last_bid": 2.0,
            },
            {
                **base,
                "option_symbol": "AAA260618P00095000",
                "option_type": "P",
                "strike": 95.0,
                "last_bid": 1.0,
            },
        ]
    )
    config = ManagedConfig(
        name="CSP_SELECTOR_FIXTURE",
        direction="call",
        option_type="P",
        structure="cash_secured_put",
        moneyness=0.95,
        min_dte=60,
        max_dte=100,
        target_dte=80,
    )
    selected = _select_contracts(quotes, {"Technology": {"AAA"}}, config, set())
    assert len(selected) == 1
    assert selected[0]["option_type"] == "P"
    assert selected[0]["strike"] < selected[0]["underlying_price"]
    assert selected[0]["strike"] == pytest.approx(95.0)
    assert selected[0]["entry_credit"] == pytest.approx(1.0)
    assert selected[0]["collateral_per_share"] == pytest.approx(95.0)
    assert selected[0]["cash_collateral"] == pytest.approx(9500.0)
    assert selected[0]["max_loss_to_zero"] == pytest.approx(9403.0)


def test_cash_secured_put_uses_put_intrinsic_and_full_collateral_risk():
    position = {
        "ticker": "AAA",
        "structure": "cash_secured_put",
        "option_type": "P",
        "strike": 95.0,
        "entry_ask": 1.0,
        "entry_credit": 1.0,
        "collateral_per_share": 95.0,
    }
    assert _expiration_exit_value(position, 100.0) == pytest.approx(0.0)
    assert _expiration_exit_value(position, 90.0) == pytest.approx(5.0)
    result = _position_result(
        position,
        "SCORED",
        exit_date="2026-06-18",
        exit_bid=0.0,
        exit_reason="expiration_intrinsic",
        config=ManagedConfig(fee_per_side=1.50),
        exit_underlying_price=100.0,
    )
    assert result["gross_pnl"] == pytest.approx(100.0)
    assert result["net_pnl"] == pytest.approx(97.0)
    assert result["net_R"] == pytest.approx(97.0 / 9503.0)


def test_cash_secured_put_contract_display_is_a_readable_short_ticket():
    display = pattern_core._managed_contract_display(
        {
            "structure": "cash_secured_put",
            "option_type": "P",
            "strike": 95.0,
            "expiry": "2026-06-18",
            "entry_credit": 1.0,
            "collateral_per_share": 95.0,
            "cash_collateral": 9500.0,
            "max_loss_to_zero": 9403.0,
            "option_symbol": "AAA260618P00095000",
        }
    )
    assert display.startswith("SELL P 95.0 exp 2026-06-18 @ credit 1.0")
    assert "BUY" not in display
    assert "cash collateral 9500.0" in display
    assert "max loss to zero 9403.0" in display


def test_managed_selector_keeps_optionable_adrs_and_etfs_but_excludes_indexes():
    rows = []
    for index in range(10):
        rows.append(
            {
                "ticker": f"T{index:02d}",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": index / 11.0,
                "close": 100.0,
            }
        )
    rows.extend(
        [
            {
                "ticker": "T10",
                "sector": "Technology",
                "issue_type": "ADR",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 10 / 11.0,
                "close": 100.0,
            },
            {
                "ticker": "T11",
                "sector": "Technology",
                "issue_type": "ETF",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 1.0,
                "close": 100.0,
            },
            {
                "ticker": "INDEX",
                "sector": "Technology",
                "issue_type": "Index",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 1.0,
                "close": 100.0,
            },
        ]
    )
    selected = _signals_for_day(
        pd.DataFrame(rows),
        ManagedConfig(signal_rule="trend_quantile", top_quantile=0.90, min_sector_names=12),
    )
    assert "T10" in selected["Technology"]
    assert "T11" in selected["Technology"]
    assert "INDEX" not in selected["Technology"]

    etf_rows = [
        {
            "ticker": f"E{index:02d}",
            "sector": "",
            "issue_type": "ETF",
            "marketcap": 600_000_000.0,
            "avg30_volume": 1_000_000.0,
            "position_52w": index / 11.0,
            "close": 100.0,
        }
        for index in range(12)
    ]
    etf_selected = _signals_for_day(
        pd.DataFrame(etf_rows),
        ManagedConfig(signal_rule="trend_quantile", top_quantile=0.90, min_sector_names=12),
    )
    assert "E11" in etf_selected["ETF"]


def test_etf_eligibility_uses_price_and_volume_when_marketcap_field_is_unreliable():
    frame = pd.DataFrame(
        [
            {
                "issue_type": "ETF",
                "marketcap": 681.812269,
                "close": 683.17,
                "avg30_volume": 75_000_000.0,
            },
            {
                "issue_type": "ETF",
                "marketcap": 681.812269,
                "close": 683.17,
                "avg30_volume": 10_000.0,
            },
            {
                "issue_type": "Common Stock",
                "marketcap": 681.812269,
                "close": 10.0,
                "avg30_volume": 75_000_000.0,
            },
        ]
    )
    eligible = _marketcap_eligible(frame, ManagedConfig())
    assert eligible.tolist() == [True, False, False]


def test_managed_flow_loader_uses_dated_whale_fallback_when_cache_is_absent(tmp_path):
    day = tmp_path / "2026-01-02"
    day.mkdir()
    (day / "whale_trades_filtered.csv").write_text(
        "underlying_symbol,option_type,side,premium,size,price,canceled,upstream_condition_detail\n"
        "AAA,call,ask,100000,,,false,\n"
        "AAA,put,bid,25000,,,false,\n",
        encoding="utf-8",
    )
    flow, metadata = load_cached_bot_flow(tmp_path, "2026-01-01", "2026-01-02")
    assert flow["ticker"].tolist() == ["AAA"]
    assert flow.iloc[0]["flow_total_premium"] == pytest.approx(125000.0)
    assert flow.iloc[0]["bot_flow_bias"] == pytest.approx(1.0)
    assert metadata["bot_flow_fallback_source_dates"] == 1
    assert metadata["bot_flow_status"] == "FALLBACK_ONLY"


def test_managed_price_research_uses_global_session_horizon():
    panel = pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "ticker": "AAA",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 0.5,
                "close": 100.0,
                "adjusted_close": 100.0,
            },
            {
                "date": "2026-01-05",
                "ticker": "AAA",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 1.0,
                "close": 110.0,
                "adjusted_close": 110.0,
            },
            {
                "date": "2026-01-06",
                "ticker": "AAA",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 1.0,
                "close": 120.0,
                "adjusted_close": 120.0,
            },
            {
                "date": "2026-01-07",
                "ticker": "AAA",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 1.0,
                "close": 130.0,
                "adjusted_close": 130.0,
            },
            {
                "date": "2026-01-08",
                "ticker": "AAA",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 1.0,
                "close": 150.0,
                "adjusted_close": 150.0,
            },
        ]
    )
    outcomes, summary = managed_price_research(
        panel,
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
        {
            "test": (
                ManagedConfig(name="test", min_sector_names=1, top_quantile=0.90),
                "signal",
            )
        },
        horizons=(1,),
    )
    jan5 = [row for row in outcomes if row["signal_date"] == "2026-01-05"]
    assert len(jan5) == 1
    assert jan5[0]["status"] == "SCORED"
    assert jan5[0]["entry_date"] == "2026-01-06"
    assert jan5[0]["target_date"] == "2026-01-07"
    assert jan5[0]["stock_return"] == pytest.approx(130.0 / 120.0 - 1.0)
    train = [row for row in summary if row["sample"] == "TRAIN" and row["horizon"] == 1]
    assert len(train) == 1
    assert train[0]["entry_count"] == 5
    assert train[0]["pending_future_count"] == 2
    assert train[0]["eligible_count"] == 3
    assert train[0]["coverage"] == 1.0


def test_managed_price_validation_can_qualify_directional_pattern_after_maturity_fix():
    rows = []
    for sample in ("TRAIN", "VALIDATION", "HOLDOUT"):
        rows.append(
            {
                "strategy_key": "TEST_FLOW",
                "direction": "bullish",
                "horizon": 20,
                "sample": sample,
                "entry_count": 25,
                "eligible_count": 20,
                "pending_future_count": 5,
                "scored_count": 20,
                "coverage": 1.0,
                "unique_signal_dates": 20,
                "average_directional_return": 0.02,
                "profit_factor": 1.5,
                "lower_mean_95": 0.01,
                "date_average_directional_return": 0.02,
                "date_lower_mean_95": 0.005,
                "date_max_drawdown": -1.0,
            }
        )
    result = managed_price_validation_rows(rows)
    assert len(result) == 1
    assert result[0]["status"] == "QUALIFIED_DIRECTIONAL"
    assert result[0]["approval_status"] == "STOCK_RESEARCH_ONLY"


def test_directional_board_uses_latest_qualified_signal_without_option_approval():
    validation = [
        {
            "strategy_key": "TEST_FLOW",
            "direction": "bullish",
            "horizon": 20,
            "status": "QUALIFIED_DIRECTIONAL",
            "train_average_directional_return": 0.02,
            "validation_average_directional_return": 0.03,
            "holdout_average_directional_return": 0.025,
            "holdout_profit_factor": 1.5,
        }
    ]
    outcomes = [
        {
            "strategy_key": "TEST_FLOW",
            "direction": "bullish",
            "horizon": 20,
            "signal_date": "2026-08-19",
            "ticker": "OLD",
            "position_52w": 0.9,
            "flow_bias": 0.9,
        },
        {
            "strategy_key": "TEST_FLOW",
            "direction": "bullish",
            "horizon": 20,
            "signal_date": "2026-08-20",
            "ticker": "NEW",
            "position_52w": 0.8,
            "flow_bias": 0.8,
        },
    ]
    board = pattern_core._managed_directional_board(outcomes, validation, "2026-08-20")
    assert [row["ticker"] for row in board] == ["NEW"]
    assert board[0]["status"] == "RESEARCH_STOCK_ONLY"
    assert board[0]["action"] == "STOCK_RESEARCH_ONLY"
    assert board[0]["approval_status"] == "STOCK_RESEARCH_ONLY"


def test_session_shifted_price_does_not_skip_missing_ticker_sessions():
    panel = pd.DataFrame(
        [
            {"ticker": "AAA", "date": "2026-01-02", "adjusted_close": 100.0},
            {"ticker": "AAA", "date": "2026-01-05", "adjusted_close": 110.0},
            {"ticker": "BBB", "date": "2026-01-06", "adjusted_close": 200.0},
            {"ticker": "AAA", "date": "2026-01-07", "adjusted_close": 130.0},
        ]
    )
    prior = _session_shifted_price(panel, 1)
    assert pd.isna(prior.iloc[0])
    assert prior.iloc[1] == pytest.approx(100.0)
    assert pd.isna(prior.iloc[2])


def test_managed_momentum_flow_rule_requires_both_confirmations():
    panel_day = pd.DataFrame(
        [
            {
                "ticker": f"T{index:02d}",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 0.5,
                "close": 100.0,
                "return_5d": (index - 5) / 100.0,
                "flow_bias": (index - 5) / 10.0,
            }
            for index in range(100)
        ]
    )
    calls = _signals_for_day(
        panel_day,
        ManagedConfig(signal_rule="momentum_flow", top_quantile=0.99, min_sector_names=12),
    )
    puts = _signals_for_day(
        panel_day,
        ManagedConfig(
            signal_rule="momentum_flow",
            direction="put",
            top_quantile=0.99,
            min_sector_names=12,
        ),
    )
    flow_calls = _signals_for_day(
        panel_day,
        ManagedConfig(signal_rule="flow_quantile", top_quantile=0.99, min_sector_names=12),
    )
    assert calls["Technology"] == {"T98", "T99"}
    assert puts["Technology"] == {"T00"}
    assert flow_calls["Technology"] == {"T98", "T99"}


def test_managed_trend_flow_rule_requires_both_confirmations():
    panel_day = pd.DataFrame(
        [
            {
                "ticker": f"T{index:02d}",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": index / 100.0,
                "close": 100.0,
                "return_5d": 0.0,
                "flow_bias": index / 100.0,
            }
            for index in range(100)
        ]
    )
    calls = _signals_for_day(
        panel_day,
        ManagedConfig(signal_rule="trend_flow", top_quantile=0.99, min_sector_names=12),
    )
    assert calls["Technology"] == {"T98", "T99"}
    composite_calls = _signals_for_day(
        panel_day,
        ManagedConfig(
            signal_rule="trend_flow_composite",
            top_quantile=0.99,
            min_sector_names=12,
        ),
    )
    assert composite_calls["Technology"] == {"T98", "T99"}


def test_managed_post_event_reversion_uses_opposite_event_direction():
    panel_day = pd.DataFrame(
        [
            {
                "ticker": f"T{index:02d}",
                "sector": "Technology",
                "issue_type": "Common Stock",
                "marketcap": 3_000_000_000.0,
                "avg30_volume": 1_000_000.0,
                "position_52w": 0.5,
                "close": 100.0,
                "return_1d": 0.06 if index == 99 else (-0.06 if index == 0 else 0.0),
            }
            for index in range(100)
        ]
    )
    calls = _signals_for_day(
        panel_day,
        ManagedConfig(signal_rule="post_event_mean_reversion", direction="call", min_sector_names=12),
    )
    puts = _signals_for_day(
        panel_day,
        ManagedConfig(
            signal_rule="post_event_mean_reversion",
            direction="put",
            min_sector_names=12,
        ),
    )
    assert calls["Technology"] == {"T00"}
    assert puts["Technology"] == {"T99"}


def test_managed_market_filter_is_point_in_time():
    rows = [
        {
            "ticker": "SPY",
            "sector": "ETF",
            "issue_type": "ETF",
            "marketcap": 1_000_000_000_000.0,
            "avg30_volume": 1_000_000.0,
            "position_52w": 0.5,
            "close": 100.0,
            "return_5d": -0.02,
            "return_20d": 0.01,
        }
    ] + [
        {
            "ticker": f"T{index:02d}",
            "sector": "Technology",
            "issue_type": "Common Stock",
            "marketcap": 3_000_000_000.0,
            "avg30_volume": 1_000_000.0,
            "position_52w": index / 20.0,
            "close": 100.0,
            "return_5d": index / 100.0,
            "return_20d": index / 100.0,
        }
        for index in range(20)
    ]
    panel_day = pd.DataFrame(rows)
    filtered = _signals_for_day(
        panel_day,
        ManagedConfig(
            signal_rule="momentum_5",
            top_quantile=0.90,
            min_sector_names=12,
            market_filter="SPY_5D_DOWN_OR_FLAT",
        ),
    )
    assert filtered["Technology"] == {"T17", "T18", "T19"}
    blocked = _signals_for_day(
        panel_day.assign(return_5d=lambda frame: frame["return_5d"].where(frame["ticker"].ne("SPY"), 0.02)),
        ManagedConfig(
            signal_rule="momentum_5",
            top_quantile=0.90,
            min_sector_names=12,
            market_filter="SPY_5D_DOWN_OR_FLAT",
        ),
    )
    assert blocked == {}


def test_managed_position_cap_ranks_selected_names_deterministically():
    panel_day = pd.DataFrame(
        [
            {"ticker": "A", "position_52w": 0.90, "flow_bias": 0.1, "return_5d": 0.1, "return_20d": 0.1, "return_1d": 0.0},
            {"ticker": "B", "position_52w": 0.80, "flow_bias": 0.2, "return_5d": 0.2, "return_20d": 0.2, "return_1d": 0.0},
            {"ticker": "C", "position_52w": 0.70, "flow_bias": 0.3, "return_5d": 0.3, "return_20d": 0.3, "return_1d": 0.0},
        ]
    )
    selected = _limit_selected_names(
        panel_day,
        {"Technology": {"A", "B", "C"}},
        ManagedConfig(signal_rule="trend_quantile", max_positions_per_day=2),
    )
    assert selected == {"Technology": {"A", "B"}}


def test_managed_strategy_keeps_unobservable_future_exit_unscored():
    sessions = ["2026-01-02", "2026-01-05", "2026-01-06"]
    rows = [
        {
            "date": day,
            "ticker": f"T{index:02d}",
            "sector": "Technology",
            "issue_type": "Common Stock",
            "marketcap": 3_000_000_000.0,
            "close": 100.0,
            "avg30_volume": 1_000_000.0,
            "position_52w": index / 11.0,
        }
        for day in sessions
        for index in range(12)
    ]
    entry = pd.DataFrame(
        [
            {
                "option_symbol": "T11260618C00105000",
                "ticker": "T11",
                "option_type": "C",
                "strike": 105.0,
                "expiry": "2026-06-18",
                "dte": 164.0,
                "last_bid": 0.9,
                "last_ask": 1.0,
                "curr_oi": 100.0,
                "stock_price": 100.0,
                "spread_pct": 0.10,
                "source_date": "2026-01-05",
            }
        ]
    )

    def quotes(day):
        return entry if day == "2026-01-05" else pd.DataFrame()

    config = ManagedConfig(
        min_sector_names=12,
        min_dte=100,
        max_dte=200,
        target_dte=164,
        max_hold_sessions=2,
    )
    result = run_managed_strategy(pd.DataFrame(rows), sessions, quotes, config)
    assert len(result) == 1
    assert result.iloc[0]["status"] == "PENDING_FUTURE"
    assert pd.isna(result.iloc[0]["net_R"])


def test_managed_coverage_excludes_right_censored_positions():
    trades = pd.DataFrame(
        [
            {
                "control": "signal",
                "strategy_key": "TEST",
                "signal_date": "2026-01-02",
                "status": "SCORED",
                "net_R": 0.25,
                "win": True,
            },
            {
                "control": "signal",
                "strategy_key": "TEST",
                "signal_date": "2026-01-02",
                "status": "SCORED",
                "net_R": 0.15,
                "win": True,
            },
            {
                "control": "signal",
                "strategy_key": "TEST",
                "signal_date": "2026-01-03",
                "status": "PENDING_FUTURE",
                "net_R": None,
                "win": None,
            },
        ]
    )
    scorecard = summarize_managed(trades)
    assert scorecard[0]["entry_count"] == 3
    assert scorecard[0]["eligible_count"] == 2
    assert scorecard[0]["pending_future_count"] == 1
    assert scorecard[0]["coverage"] == 1.0
    assert scorecard[0]["date_average_net_R"] == 0.2
    validation = managed_validation_rows(scorecard, min_scored=1, min_dates=1)
    assert validation[0]["train_coverage"] == 1.0


def test_managed_calibration_is_prior_only_and_grouped_by_signal_date():
    trades = pd.DataFrame(
        [
            {
                "control": "signal",
                "pattern_family": "TEST",
                "ticker": "AAA",
                "option_symbol": "AAA1",
                "signal_date": "2026-01-02",
                "entry_date": "2026-01-05",
                "status": "SCORED",
                "net_R": 0.10,
            },
            {
                "control": "signal",
                "pattern_family": "TEST",
                "ticker": "BBB",
                "option_symbol": "BBB1",
                "signal_date": "2026-01-02",
                "entry_date": "2026-01-05",
                "status": "SCORED",
                "net_R": 0.20,
            },
            {
                "control": "signal",
                "pattern_family": "TEST",
                "ticker": "CCC",
                "option_symbol": "CCC1",
                "signal_date": "2026-01-05",
                "entry_date": "2026-01-06",
                "status": "SCORED",
                "net_R": -0.10,
            },
        ]
    )
    rows = managed_calibration_rows(trades)
    assert [row["predicted_win_probability"] for row in rows[:2]] == pytest.approx(
        [0.50, 0.50]
    )
    assert rows[2]["predicted_win_probability"] == pytest.approx(12.0 / 22.0)
    assert rows[0]["score_bin"] == rows[1]["score_bin"]


def test_holdout_calibration_is_frozen_before_holdout():
    trades = pd.DataFrame(
        [
            {
                "control": "signal",
                "signal_date": f"2026-01-{index + 1:02d}",
                "status": "SCORED",
                "net_R": 0.1,
            }
            for index in range(20)
        ]
        + [
            {
                "control": "signal",
                "signal_date": "2026-06-15",
                "status": "SCORED",
                "net_R": -0.1,
            },
            {
                "control": "signal",
                "signal_date": "2026-06-16",
                "status": "SCORED",
                "net_R": 0.1,
            },
        ]
    )

    summary = frozen_holdout_calibration(trades, "2026-06-15")

    assert summary["calibration_predicted_probability"] == pytest.approx(0.75)
    assert summary["calibration_sample_count"] == 2
    assert summary["calibration_method"] == "frozen_pre_holdout_beta_mean_win_rate"
    assert summary["calibration_train_through"] < "2026-06-15"


def test_predeclared_selection_uses_only_earlier_validation_and_marks_selected_candidate():
    dates = [date.strftime("%Y-%m-%d") for date in pd.bdate_range("2026-01-02", periods=95)]
    split_date = dates[35]
    holdout_start = dates[70]
    selection_end = dates[69]

    def candidate_rows(name, selection_value, holdout_value, control="signal"):
        rows = []
        for index, day in enumerate(dates):
            value = selection_value if index < 70 else holdout_value
            if index % 10 == 0:
                value = -0.01
            rows.append(
                {
                    "control": control,
                    "pattern_family": name,
                    "ticker": f"T{index:03d}",
                    "option_symbol": f"{name}{index}",
                    "signal_date": day,
                    "entry_date": day,
                    "status": "SCORED",
                    "net_R": value,
                    "win": True,
                }
            )
        return pd.DataFrame(rows)

    audit, metadata = managed_selection_audit(
        {
            "CANDIDATE_A": candidate_rows("CANDIDATE_A", 0.04, 0.01),
            "CANDIDATE_A_RANDOM": candidate_rows(
                "CANDIDATE_A", 0.005, 0.005, control="random"
            ),
            # This candidate has the stronger holdout on purpose.  It must not
            # win because holdout rows are unavailable to the selector.
            "CANDIDATE_B": candidate_rows("CANDIDATE_B", 0.02, 0.20),
            "CANDIDATE_B_RANDOM": candidate_rows(
                "CANDIDATE_B", 0.015, 0.015, control="random"
            ),
        },
        selection_end=selection_end,
        holdout_start=holdout_start,
        split_date=split_date,
    )
    selected = next(row for row in audit if row["selected_candidate"])
    assert selected["candidate_key"] == "CANDIDATE_A"
    assert metadata["selected_candidate_key"] == "CANDIDATE_A"
    assert metadata["holdout_used_for_selection"] is False
    assert metadata["status"] == "PASS"
    assert selected["selection_train_control_pass"] is True
    assert selected["selection_validation_control_pass"] is True
    assert selected["final_holdout_control_pass"] is True
    assert selected["final_holdout_calibration_score"] == selected["final_calibration_score"]
    assert all(row["final_holdout_start"] == holdout_start for row in audit)


def test_predeclared_selection_rejects_holdout_that_loses_to_random_control():
    dates = [date.strftime("%Y-%m-%d") for date in pd.bdate_range("2026-01-02", periods=95)]
    split_date = dates[35]
    holdout_start = dates[70]
    selection_end = dates[69]

    def rows(control, selection_value, holdout_value):
        values = [
            -0.01
            if index % 10 == 0
            else selection_value
            if index < 70
            else holdout_value
            for index in range(len(dates))
        ]
        return pd.DataFrame(
            {
                "control": [control] * len(dates),
                "pattern_family": ["CANDIDATE"] * len(dates),
                "ticker": [f"T{index:03d}" for index in range(len(dates))],
                "option_symbol": [f"O{index:03d}" for index in range(len(dates))],
                "signal_date": dates,
                "entry_date": dates,
                "status": ["SCORED"] * len(dates),
                "net_R": values,
                "win": [value > 0 for value in values],
            }
        )

    audit, metadata = managed_selection_audit(
        {
            "CANDIDATE": rows("signal", 0.04, 0.01),
            "CANDIDATE_RANDOM": rows("random", 0.01, 0.02),
        },
        selection_end=selection_end,
        holdout_start=holdout_start,
        split_date=split_date,
    )

    selected = audit[0]
    assert selected["selection_eligible"] is True
    assert selected["final_holdout_control_pass"] is False
    assert "MATCHED_RANDOM_CONTROL_GATE" in selected["final_holdout_blockers"]
    assert metadata["status"] == "FAIL_REQUIREMENTS_REMAIN"


def test_predeclared_candidate_grid_is_small_and_contains_production_csp_lane():
    candidates = predeclared_managed_selection_candidates()
    assert len(candidates) == 6
    assert "FLOW_QUANTILE_BULL_CSP_D60_110_H40_T50_IV60" in candidates
    assert {config.structure for config, _control in candidates.values()} == {"cash_secured_put"}
    assert {config.moneyness for config, _control in candidates.values()} == {0.90, 0.95, 1.00}


def test_managed_regime_rows_use_signal_date_benchmark_and_keep_weak_samples_explicit():
    trades = pd.DataFrame(
        [
            {
                "status": "SCORED",
                "signal_date": "2026-06-15",
                "net_R": 0.10,
                "ticker": "AAA",
            },
            {
                "status": "SCORED",
                "signal_date": "2026-06-16",
                "net_R": 0.20,
                "ticker": "BBB",
            },
            {
                "status": "SCORED",
                "signal_date": "2026-06-17",
                "net_R": 0.30,
                "ticker": "CCC",
            },
        ]
    )
    panel = pd.DataFrame(
        [
            {"ticker": "SPY", "date": "2026-06-15", "return_20d": -0.05},
            {"ticker": "SPY", "date": "2026-06-16", "return_20d": 0.00},
            {"ticker": "SPY", "date": "2026-06-17", "return_20d": 0.05},
        ]
    )
    rows = managed_regime_rows(trades, panel, "2026-06-15")
    assert {row["regime"] for row in rows} == {"BEAR", "BULL", "SIDEWAYS"}
    assert all(row["average_net_R"] > 0 for row in rows)
    assert all(row["status"] == "POSITIVE_SMALL_SAMPLE" for row in rows)


def test_price_outcomes_are_normalized_for_validation():
    rows = price_outcome_rows(
        [
            {
                "signal_id": "s1",
                "pattern_family": "TEST",
                "direction": "bullish",
                "signal_date": "2026-01-02",
                "stock_return_1d": 0.02,
                "stock_return_5d": 0.10,
            }
        ]
    )
    assert {(row["horizon"], row["stock_return"]) for row in rows} == {(1, 0.02), (5, 0.10)}


def test_neutral_magnitude_rows_do_not_qualify_as_directional_price_patterns():
    neutral = {
        "pattern_family": "EARNINGS_VOLATILITY_EVENT",
        "direction": "neutral",
        "horizon": 20,
        "strategy": "",
        "sample_count": 200,
        "unique_signal_dates": 40,
        "average_value": 0.10,
        "profit_factor": float("inf"),
        "latest_holdout_average": 0.08,
        "lower_mean_95": 0.08,
        "date_lower_mean_95": 0.07,
        "date_max_drawdown": 0.0,
    }
    qualified, qualified_walk, qualified_rolling, calibrated = price_gate_sets(
        [neutral],
        [],
        [],
        [],
    )
    assert qualified == []
    assert qualified_walk == []
    assert qualified_rolling == []
    assert calibrated == []


def test_stale_option_pattern_cannot_qualify_on_old_latest_holdout_window():
    row = {
        "pattern_family": "STALE",
        "direction": "bullish",
        "horizon": 20,
        "strategy": "LONG_OPTION",
        "sample_count": 100,
        "unique_signal_dates": 25,
        "average_value": 0.10,
        "profit_factor": 1.50,
        "latest_holdout_average": 0.08,
        "lower_mean_95": 0.02,
        "date_lower_mean_95": 0.01,
        "date_max_drawdown": -1.0,
        "last_signal_date": "2026-01-30",
    }
    coverage = {
        "pattern_family": "STALE",
        "direction": "bullish",
        "horizon": 20,
        "strategy": "LONG_OPTION",
        "coverage_ratio": 1.0,
    }
    qualified, qualified_walk, qualified_rolling, calibrated = option_gate_sets(
        [row],
        [],
        [],
        [],
        [coverage],
        as_of="2026-08-18",
    )
    assert qualified == []
    assert qualified_walk == []
    assert qualified_rolling == []
    assert calibrated == []


def test_current_board_uses_declared_horizon_for_five_day_validation_lane():
    signal = PriceSignal(
        date="2026-08-18",
        ticker="AAA",
        direction="bullish",
        family="POST_EVENT_MEAN_REVERSION",
        role="post_event_setup",
        score=8.0,
        reasons=[],
        feature={"close": 100.0, "avg30_volume": 1_000_000.0, "volume_ratio_30d": 1.0},
    )
    board = build_current_board(
        [signal],
        [],
        [
            {
                "pattern_family": "POST_EVENT_MEAN_REVERSION",
                "direction": "bullish",
                "signal_role": "post_event_setup",
                "strategy": "LONG_OPTION",
                "horizon": 5,
                "status": "SCORED",
                "net_R": 0.25,
                "entry_spread_pct": 0.05,
                "dte": 35,
            }
        ],
        "2026-08-18",
        {
            ("2026-08-18", "AAA", "bullish"): {
                "variants": [
                    {
                        "strategy": "LONG_OPTION",
                        "option_symbol": "AAA260918C00100000",
                        "ask": 1.0,
                        "spread_pct": 0.05,
                        "dte": 35,
                    }
                ]
            }
        },
        set(),
        [],
    )
    assert board[0]["validation_lane"] == "POST_EVENT_MEAN_REVERSION_25_45DTE_TIGHT_SPREAD"
    assert board[0]["option_sample_count_5d"] == 1


def test_managed_board_keeps_qualified_open_lane_when_newer_review_lane_exists():
    trade_rows = [
        {
            "control": "signal",
            "status": "PENDING_FUTURE",
            "pattern_family": "QUALIFIED_CSP",
            "ticker": "AAA",
            "option_symbol": "AAA260618P00095000",
            "signal_date": "2026-08-18",
            "entry_date": "2026-08-19",
        },
        {
            "control": "signal",
            "status": "PENDING_FUTURE",
            "pattern_family": "NEWER_REVIEW",
            "ticker": "BBB",
            "option_symbol": "BBB260918C00100000",
            "signal_date": "2026-08-19",
            "entry_date": "2026-08-20",
        },
    ]
    latest, current, review_count = pattern_core._managed_current_rows(
        trade_rows,
        {
            "QUALIFIED_CSP": {"status": "QUALIFIED_MANAGED"},
            "NEWER_REVIEW": {"status": "RESEARCH_PATTERN"},
        },
        "2026-08-20",
    )
    assert latest == "2026-08-19"
    assert {(row["pattern_family"], row["ticker"]) for row in current} == {
        ("QUALIFIED_CSP", "AAA"),
        ("NEWER_REVIEW", "BBB"),
    }
    assert review_count == 1


def test_managed_ticket_state_does_not_execute_stale_historical_quotes():
    stale = pattern_core._managed_ticket_state(
        {"entry_date": "2026-08-19"},
        {
            "status": "QUALIFIED_MANAGED",
            "production_status": "PRODUCTION_QUALIFIED",
            "approval_status": "QUALIFIED_MANAGED",
        },
        "2026-08-20",
    )
    assert stale == (
        "HISTORICAL_APPROVED_TRADE",
        "DO_NOT_TRADE",
        "HISTORICAL_VALIDATED",
        "ENTRY_DATE_PAST_AS_OF",
    )
    fresh = pattern_core._managed_ticket_state(
        {"entry_date": "2026-08-21"},
        {
            "status": "QUALIFIED_MANAGED",
            "production_status": "PRODUCTION_QUALIFIED",
            "approval_status": "QUALIFIED_MANAGED",
        },
        "2026-08-20",
    )
    assert fresh[:2] == ("APPROVED_TRADE", "EXECUTE")
    conditional = pattern_core._managed_ticket_state(
        {"candidate_timing": "SAME_DAY_EOD_RESEARCH"},
        {"production_status": "PRODUCTION_QUALIFIED"},
        "2026-08-20",
    )
    assert conditional[:2] == ("TRADE_REVIEW", "RECHECK_NEXT_SESSION_QUOTE")
    research = pattern_core._managed_ticket_state(
        {"candidate_timing": "SAME_DAY_EOD_RESEARCH"},
        {"production_status": "RESEARCH_ONLY"},
        "2026-08-20",
    )
    assert research[:2] == ("RESEARCH_SETUP", "DO_NOT_TRADE")


def test_managed_report_separates_historical_rows_from_live_action_board():
    report = pattern_core._managed_primary_report(
        "2026-08-20",
        {"current_review_position_count": 1},
        [
            {
                "status": "HISTORICAL_APPROVED_TRADE",
                "action": "DO_NOT_TRADE",
                "ticker": "BCO",
                "pattern_family": "CSP",
                "contract": "SELL P 100 exp 2026-11-20 @ credit 3.60",
                "signal_date": "2026-08-18",
                "entry_date": "2026-08-19",
                "holdout_average_net_R": 0.03,
                "holdout_profit_factor": 14.0,
                "blockers": "ENTRY_DATE_PAST_AS_OF",
            },
            {
                "status": "TRADE_REVIEW",
                "action": "DO_NOT_TRADE",
                "ticker": "MRK",
                "pattern_family": "CALL",
                "contract": "BUY C 155 exp 2026-11-20 @ ask 7.70",
                "signal_date": "2026-08-19",
                "entry_date": "2026-08-20",
                "blockers": "TRAIN_GATE;VALIDATION_GATE",
            },
        ],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    live_section = report.split("## Historical Backtest References", 1)[0]
    assert "| DO_NOT_TRADE | TRADE_REVIEW | MRK" in live_section
    assert "BCO" not in live_section
    assert "Historical Backtest References" in report
    assert "HISTORICAL_ONLY | BCO" in report


def test_live_setup_is_not_truncated_by_historical_research_rows():
    board = [
        {
            "status": "HISTORICAL_RESEARCH",
            "ticker": f"H{index:02d}",
            "holdout_average_net_R": 1.0,
        }
        for index in range(30)
    ]
    board.append(
        {
            "status": "RESEARCH_SETUP",
            "ticker": "LIVE",
            "holdout_average_net_R": 0.0,
        }
    )
    capped = pattern_core._cap_managed_action_board(board, 25)

    assert capped[0]["ticker"] == "LIVE"
    assert len(capped) == 25


def test_named_mover_audit_separates_pre_event_and_post_event_followup(
    monkeypatch,
):
    panel = pd.DataFrame(
        [
            {"date": "2026-01-05", "ticker": "MRNA", "return_1d": 0.01},
            {"date": "2026-01-06", "ticker": "MRNA", "return_1d": 0.06},
            {"date": "2026-01-07", "ticker": "MRNA", "return_1d": -0.07},
        ]
    )

    def fake_signals(frame, config):
        return {"Technology": {"MRNA"}}

    monkeypatch.setattr(managed_module, "_signals_for_day", fake_signals)
    rows = named_mover_audit(
        panel,
        ["2026-01-05", "2026-01-06", "2026-01-07"],
        {"TEST_PUT": (ManagedConfig(direction="put"), "signal")},
    )
    row = next(item for item in rows if item["event_date"] == "2026-01-06")
    assert row["pre_event_same_direction"] is False
    assert row["post_event_same_direction"] is True
    assert row["post_event_followup_reason"] == "POST_EVENT_SIGNAL_PRESENT"


def test_named_mover_audit_includes_mrvl_as_an_explicit_target():
    panel = pd.DataFrame(
        [
            {"date": "2026-01-05", "ticker": "MRVL", "return_1d": 0.01},
            {"date": "2026-01-06", "ticker": "MRVL", "return_1d": 0.08},
        ]
    )

    rows = named_mover_audit(panel, ["2026-01-05", "2026-01-06"], {})

    assert [row["ticker"] for row in rows] == ["MRVL"]
    assert rows[0]["pre_event_same_direction"] is False


def test_option_pattern_with_missing_last_signal_date_is_stale_when_as_of_is_given():
    row = {
        "pattern_family": "MISSING_DATE",
        "direction": "bullish",
        "horizon": 20,
        "strategy": "LONG_OPTION",
        "sample_count": 100,
        "unique_signal_dates": 25,
        "average_value": 0.10,
        "profit_factor": 1.50,
        "latest_holdout_average": 0.08,
        "lower_mean_95": 0.02,
        "date_lower_mean_95": 0.01,
        "date_max_drawdown": -1.0,
    }
    coverage = {**{key: row[key] for key in ("pattern_family", "direction", "horizon", "strategy")}, "coverage_ratio": 1.0}
    qualified, qualified_walk, qualified_rolling, calibrated = option_gate_sets(
        [row], [], [], [], [coverage], as_of="2026-08-18"
    )
    assert qualified == []
    assert qualified_walk == []
    assert qualified_rolling == []
    assert calibrated == []


def test_quote_cache_key_changes_with_pipeline_version(tmp_path, monkeypatch):
    first = pattern_core.quote_cache_path(tmp_path, "quotes", {"schema": 1})
    monkeypatch.setattr(pattern_core, "PIPELINE_VERSION", "test-next-version")
    second = pattern_core.quote_cache_path(tmp_path, "quotes", {"schema": 1})
    assert first != second


def test_report_separates_rejected_contracts_and_displays_net_entry_price():
    report = render_report(
        "2026-08-18",
        {
            "pipeline_version": "test",
            "qualified_price_pattern_count": 0,
            "qualified_price_walk_forward_pattern_count": 0,
            "qualified_price_rolling_holdout_pattern_count": 0,
            "qualified_rolling_holdout_pattern_count": 0,
        },
        [
            {
                "status": "REJECTED_CURRENT",
                "ticker": "AAA",
                "direction": "bearish",
                "pattern_family": "FLOW",
                "option_strategy": "CREDIT_VERTICAL",
                "contract": "AAA261016C00100000 / AAA261016C00110000",
                "entry_display": "credit 1.2",
                "option_sample_count_5d": 100,
                "option_average_net_R_5d": -0.1,
                "option_profit_factor_5d": 0.8,
                "option_validation_horizon": 5,
                "blockers": "OPTION_NET_EV_NOT_POSITIVE",
                "score": 1.0,
            }
        ],
        [],
        [],
        [],
    )
    assert "Rejected current contracts: **1**" in report
    assert "| credit 1.2 | -0.1 | 0.8 |" in report
    assert "| TRADE_REVIEW |" not in report


def test_short_dated_option_does_not_score_after_expiry():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260108C00330000",
        "expiry": "2026-01-08",
        "ask": 5.0,
        "strategy": "LONG_OPTION_SHORT_DTE",
        "max_horizon": 5,
    }
    history = {
        "AAPL260108C00330000": {
            "2026-01-12": {"bid": 6.5},
        }
    }
    dates = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
    ]
    assert option_outcome(entry, history, dates, 5) is None


def test_short_dated_credit_vertical_uses_defined_risk_and_both_legs():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260115P00300000",
        "expiry": "2026-01-15",
        "ask": 2.10,
        "strategy": "CREDIT_VERTICAL_SHORT_DTE",
        "max_horizon": 5,
        "credit_short_option_symbol": "AAPL260115P00300000",
        "credit_long_option_symbol": "AAPL260115P00290000",
        "entry_credit": 1.00,
        "credit_width": 10.00,
    }
    history = {
        "AAPL260115P00300000": {
            "2026-01-09": {"bid": 0.15, "ask": 0.20},
        },
        "AAPL260115P00290000": {
            "2026-01-09": {"bid": 0.05, "ask": 0.10},
        },
    }
    dates = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
    ]
    outcome = option_outcome(entry, history, dates, 5)
    assert outcome["entry_price"] == 1.0
    assert abs(outcome["exit_price"] - 0.15) < 1e-12
    assert outcome["net_R"] > 0


def test_short_dated_debit_vertical_uses_defined_risk_and_both_legs():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260115C00330000",
        "expiry": "2026-01-15",
        "ask": 5.00,
        "strategy": "DEBIT_VERTICAL_SHORT_DTE",
        "max_horizon": 5,
        "short_option_symbol": "AAPL260115C00340000",
        "short_bid": 2.00,
        "vertical_width": 10.00,
    }
    history = {
        "AAPL260115C00330000": {
            "2026-01-09": {"bid": 7.00, "ask": 7.20},
        },
        "AAPL260115C00340000": {
            "2026-01-09": {"bid": 2.00, "ask": 2.20},
        },
    }
    dates = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
    ]
    outcome = option_outcome(entry, history, dates, 5)
    assert abs(outcome["entry_price"] - 3.0) < 1e-12
    assert abs(outcome["exit_price"] - 4.8) < 1e-12
    assert outcome["net_R"] > 0


def test_long_dated_debit_vertical_uses_defined_risk_and_both_legs():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260618C00330000",
        "expiry": "2026-06-18",
        "ask": 5.00,
        "strategy": "DEBIT_VERTICAL_LONG_DTE",
        "max_horizon": 20,
        "short_option_symbol": "AAPL260618C00340000",
        "short_bid": 2.00,
        "vertical_width": 10.00,
    }
    history = {
        "AAPL260618C00330000": {
            "2026-01-22": {"bid": 7.00, "ask": 7.20},
        },
        "AAPL260618C00340000": {
            "2026-01-22": {"bid": 2.00, "ask": 2.20},
        },
    }
    dates = [f"2026-01-{day:02d}" for day in range(1, 23)]
    outcome = option_outcome(entry, history, dates, 20)
    assert abs(outcome["entry_price"] - 3.0) < 1e-12
    assert abs(outcome["exit_price"] - 4.8) < 1e-12
    assert outcome["net_R"] > 0


def test_entry_selector_keeps_long_dated_contracts_visible(tmp_path):
    date_dir = tmp_path / "2026-01-02"
    date_dir.mkdir()
    fields = ["option_symbol", "bid", "ask", "volume", "open_interest", "premium", "iv", "close"]
    rows = [
        ["AAA260618C00100000", "5.00", "5.20", "1000", "2000", "500000", "0.40", "100"],
        ["AAA260618C00110000", "2.50", "2.70", "900", "1800", "250000", "0.42", "100"],
    ]
    archive = date_dir / "hot-chains-2026-01-02.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        content = ",".join(fields) + "\n"
        content += "\n".join(",".join(row) for row in rows) + "\n"
        handle.writestr("hot-chains-2026-01-02.csv", content)
    quotes, metadata = load_entry_option_quotes(
        tmp_path,
        ["2026-01-02"],
        {"2026-01-02": {("AAA", "bullish")}},
        {"2026-01-02": {"AAA": {"close": 100.0}}},
    )
    variants = quotes[("2026-01-02", "AAA", "bullish")]["variants"]
    strategies = {variant["strategy"] for variant in variants}
    assert "LONG_OPTION_LONG_DTE" in strategies
    assert "DEBIT_VERTICAL_LONG_DTE" in strategies
    assert metadata["entry_option_quote_count"] == 1


def test_fixed_horizon_requires_quote_on_target_session():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260918C00330000",
        "expiry": "2026-09-18",
        "ask": 5.0,
    }
    history = {
        "AAPL260918C00330000": {
            "2026-01-06": {"bid": 6.5},
        }
    }
    dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    assert option_outcome(entry, history, dates, 3) is None


def test_option_coverage_keeps_missing_exits_visible():
    rows = [
        {
            "pattern_family": "TEST",
            "direction": "bullish",
            "horizon": 5,
            "strategy": "LONG_OPTION",
            "signal_role": "forward_setup",
            "status": "SCORED",
        },
        {
            "pattern_family": "TEST",
            "direction": "bullish",
            "horizon": 5,
            "strategy": "LONG_OPTION",
            "signal_role": "forward_setup",
            "status": "MISSING_EXIT_QUOTE",
        },
    ]
    coverage = option_outcome_coverage(rows)
    assert coverage[0]["scored_count"] == 1
    assert coverage[0]["missing_exit_count"] == 1
    assert coverage[0]["coverage_ratio"] == 0.5
    assert coverage[0]["coverage_gate"] == "FAIL"


def test_fixed_horizon_does_not_require_intermediate_quotes():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260918C00330000",
        "expiry": "2026-09-18",
        "ask": 5.0,
    }
    history = {
        "AAPL260918C00330000": {
            "2026-01-07": {"bid": 6.5},
        }
    }
    dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    outcome = option_outcome(entry, history, dates, 3)
    assert outcome["exit_date"] == "2026-01-07"
    assert outcome["net_R"] > 0


def test_future_exit_is_not_counted_as_a_missing_historical_quote():
    dates = ["2026-01-02", "2026-01-05", "2026-01-06"]
    assert option_outcome_status("2026-01-06", 1, dates, None) == "PENDING_FUTURE"
    assert option_outcome_status("2026-01-02", 1, dates, None) == "MISSING_EXIT_QUOTE"


def test_after_expiry_is_ineligible_not_missing_quote():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260108C00330000",
        "expiry": "2026-01-08",
        "ask": 5.0,
    }
    dates = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
    ]
    assert option_outcome_missing_reason(entry, {}, dates, 5) == "TARGET_AFTER_EXPIRY"
    assert option_outcome_status("2026-01-02", 5, dates, None, entry, {}) == "INELIGIBLE_CONTRACT"


def test_option_selection_rejects_contract_that_expires_before_declared_horizon():
    dates = [
        "2026-01-02",
        "2026-01-05",
        "2026-01-06",
        "2026-01-07",
        "2026-01-08",
        "2026-01-09",
        "2026-01-12",
        "2026-01-13",
        "2026-01-14",
        "2026-01-15",
        "2026-01-16",
        "2026-01-19",
        "2026-01-20",
        "2026-01-21",
        "2026-01-22",
        "2026-01-23",
        "2026-01-26",
        "2026-01-27",
        "2026-01-28",
        "2026-01-29",
        "2026-01-30",
    ]
    expiring_before_target = {"expiry": "2026-01-29", "dte": 27}
    target_safe = {"expiry": "2026-02-06", "dte": 35}
    assert not option_quote_is_horizon_safe(expiring_before_target, dates[0], dates, 20)
    assert option_quote_is_horizon_safe(target_safe, dates[0], dates, 20)


def test_explicit_zero_target_bid_is_scored_as_a_loss():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260918C00330000",
        "expiry": "2026-09-18",
        "ask": 5.0,
    }
    history = {
        "AAPL260918C00330000": {
            "2026-01-07": {"bid": 0.0},
        }
    }
    dates = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]
    outcome = option_outcome(entry, history, dates, 3)
    assert outcome["exit_bid"] == 0.0
    assert outcome["net_pnl"] < 0
    assert outcome["net_R"] < 0


def test_iron_condor_uses_all_four_legs_and_defined_risk():
    entry = {
        "date": "2026-01-02",
        "option_symbol": "AAPL260918C00350000",
        "expiry": "2026-09-18",
        "ask": 1.0,
        "strategy": "IRON_CONDOR",
        "entry_credit": 1.5,
        "iron_call_width": 5.0,
        "iron_put_width": 5.0,
        "iron_short_call_option_symbol": "AAPL260918C00350000",
        "iron_long_call_option_symbol": "AAPL260918C00355000",
        "iron_short_put_option_symbol": "AAPL260918P00325000",
        "iron_long_put_option_symbol": "AAPL260918P00320000",
    }
    history = {
        "AAPL260918C00350000": {"2026-01-07": {"bid": 0.3, "ask": 0.4}},
        "AAPL260918C00355000": {"2026-01-07": {"bid": 0.1, "ask": 0.2}},
        "AAPL260918P00325000": {"2026-01-07": {"bid": 0.4, "ask": 0.5}},
        "AAPL260918P00320000": {"2026-01-07": {"bid": 0.1, "ask": 0.2}},
    }
    outcome = option_outcome(
        entry,
        history,
        ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"],
        3,
    )
    assert outcome["exit_date"] == "2026-01-07"
    assert outcome["net_pnl"] == 74.0
    assert outcome["net_R"] > 0


def test_chain_index_keeps_source_report_date_separate_from_quote_date(tmp_path):
    source = tmp_path / "chain-oi-changes-2026-01-05.csv"
    source.write_text(
        "option_symbol,last_date,last_bid,last_ask\n"
        "AAPL260918C00330000,2026-01-02,5.0,5.2\n",
        encoding="utf-8",
    )
    index = tmp_path / "chain-index.jsonl.gz"
    raw_rows, valid_rows = build_chain_source_index(
        index,
        SourceRef(source),
        "2026-01-05",
    )
    rows = list(iter_chain_source_index(index))
    assert (raw_rows, valid_rows) == (1, 1)
    assert rows[0]["quote_date"] == "2026-01-02"
    assert rows[0]["source_date"] == "2026-01-05"
    sqlite_index = tmp_path / "chain-index.sqlite3"
    assert build_chain_sqlite_index(sqlite_index, index, {"AAPL260918C00330000"}) == 1
    with sqlite3.connect(sqlite_index) as connection:
        assert connection.execute("SELECT COUNT(*) FROM quotes").fetchone()[0] == 1
