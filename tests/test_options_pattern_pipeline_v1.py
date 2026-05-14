import zipfile

from uwos.options_pattern_pipeline_v1.core import (
    assign_family_tiers,
    build_daily_snapshot,
    build_validation_splits,
    classify_daily_signals,
    normalize_header,
    parse_option_symbol,
    score_signal_horizon,
    sources_for_date,
    trade_output_row,
)


def test_parse_occ_symbol_with_padded_root():
    parsed = parse_option_symbol("C     260515C00135000")

    assert parsed["ticker"] == "C"
    assert parsed["expiry"] == "2026-05-15"
    assert parsed["option_type"] == "call"
    assert parsed["strike"] == 135.0


def test_hot_chain_duplicate_close_columns_are_disambiguated():
    header = [
        "option_symbol",
        "date",
        "tape_time",
        "volume",
        "open_interest",
        "premium",
        "ask_side_volume",
        "bid_side_volume",
        "floor_volume",
        "high",
        "low",
        "open",
        "close",
        "iv",
        "bid",
        "ask",
        "trades",
        "avg_price",
        "mid_volume",
        "sweep_volume",
        "cross_volume",
        "total_bid_changes",
        "total_ask_changes",
        "stock_multi_leg_volume",
        "neutral_volume",
        "multileg_volume",
        "next_earnings_date",
        "er_time",
        "ticker_option_vol",
        "close",
    ]

    normalized = normalize_header(header, "hot-chains-2026-05-08.csv")

    assert normalized[12] == "option_close"
    assert normalized[29] == "underlying_close"


def test_validation_splits_are_chronological_and_include_required_examples():
    dates = [f"2025-12-{d:02d}" for d in range(19, 32)]
    dates += [f"2026-01-{d:02d}" for d in range(2, 31)]
    dates += [f"2026-02-{d:02d}" for d in range(2, 28)]
    dates += [f"2026-03-{d:02d}" for d in range(2, 32)]

    splits = build_validation_splits(dates, min_month_dates=5)

    names = {s["name"] for s in splits}
    assert "required_dec_2025_to_feb_2026" in names
    assert "required_jan_2026_to_mar_2026" in names
    for split in splits:
        assert max(split["train_dates"]) < min(split["validation_dates"])


def test_unscorable_option_outcome_is_not_a_win():
    signal = {
        "date": "2026-01-02",
        "ticker": "XYZ",
        "direction": "bullish",
        "pattern_family": "BULLISH_FLOW_EXPANSION",
        "market_regime": "MIXED",
        "sector": "Tech",
        "lead_option_symbol": "XYZ260116C00100000",
        "entry_ask": 1.0,
        "entry_bid": 0.9,
        "bid_ask_spread_pct": 0.1,
        "block_reasons": [],
        "close": 100.0,
    }
    snapshots = {
        "2026-01-02": type("S", (), {"option_quotes": {}, "features": {"XYZ": {"close": 100.0}}})(),
        "2026-01-05": type("S", (), {"option_quotes": {}, "features": {}})(),
    }

    row = score_signal_horizon(
        signal,
        snapshots,
        ["2026-01-02", "2026-01-05"],
        "unit",
        "VALIDATION",
        1,
    )

    assert row["status"] == "UNSCORABLE"
    assert row["win"] == 0


def test_managed_long_option_scores_stop_before_same_day_target():
    signal = {
        "date": "2026-01-02",
        "ticker": "XYZ",
        "direction": "bullish",
        "pattern_family": "BULLISH_FLOW_EXPANSION",
        "market_regime": "MIXED",
        "sector": "Tech",
        "lead_option_symbol": "XYZ260116C00100000",
        "strategy_kind": "long_option",
        "entry_ask": 1.0,
        "entry_bid": 0.9,
        "bid_ask_spread_pct": 0.1,
        "block_reasons": [],
        "close": 100.0,
    }
    snapshots = {
        "2026-01-02": type("S", (), {"option_quotes": {}, "features": {"XYZ": {"close": 100.0}}})(),
        "2026-01-05": type(
            "S",
            (),
            {
                "option_quotes": {"XYZ260116C00100000": {"bid": 1.5, "high": 2.2, "low": 0.4}},
                "features": {"XYZ": {"close": 101.0}},
            },
        )(),
    }

    row = score_signal_horizon(
        signal,
        snapshots,
        ["2026-01-02", "2026-01-05"],
        "unit",
        "VALIDATION",
        1,
    )

    assert row["status"] == "SCORED"
    assert row["win"] == 0
    assert row["outcome_note"] == "managed_long_option_stop_hit_conservative"


def test_credit_spread_scores_with_future_leg_quotes():
    signal = {
        "date": "2026-01-02",
        "ticker": "XYZ",
        "direction": "bullish",
        "pattern_family": "VOL_EXPANSION_CATALYST",
        "market_regime": "MIXED",
        "sector": "Tech",
        "lead_option_symbol": "SELL XYZ260116P00095000 / BUY XYZ260116P00090000",
        "strategy_kind": "credit_spread",
        "strategy_type": "Bull Put Credit Spread",
        "legs_json": '[{"action":"SELL","option_symbol":"XYZ260116P00095000"},{"action":"BUY","option_symbol":"XYZ260116P00090000"}]',
        "entry_credit": 1.0,
        "max_risk_per_contract": 401.30,
        "entry_ask": 1.0,
        "entry_bid": 1.0,
        "bid_ask_spread_pct": 0.2,
        "block_reasons": [],
        "close": 100.0,
    }
    snapshots = {
        "2026-01-02": type("S", (), {"option_quotes": {}, "features": {"XYZ": {"close": 100.0}}})(),
        "2026-01-05": type(
            "S",
            (),
            {
                "option_quotes": {
                    "XYZ260116P00095000": {"ask": 0.4},
                    "XYZ260116P00090000": {"bid": 0.1},
                },
                "features": {"XYZ": {"close": 101.0}},
            },
        )(),
    }

    row = score_signal_horizon(
        signal,
        snapshots,
        ["2026-01-02", "2026-01-05"],
        "unit",
        "VALIDATION",
        1,
    )

    assert row["status"] == "SCORED"
    assert row["win"] == 1
    assert row["outcome_note"] == "credit_spread_exit_debit_after_fees"


def test_bot_eod_is_separate_primary_source_when_present(tmp_path):
    date_dir = tmp_path / "2026-05-08"
    date_dir.mkdir()
    with zipfile.ZipFile(date_dir / "bot-eod-report-2026-05-08.zip", "w") as zf:
        zf.writestr("bot-eod-report-2026-05-08.csv", "executed_at,underlying_symbol\n")
    (date_dir / "whale_trades_filtered.csv").write_text("executed_at,underlying_symbol\n", encoding="utf-8")

    sources = sources_for_date(date_dir, "2026-05-08")

    assert sources["bot_eod"]
    assert sources["whale_filtered"]
    assert not sources["option_trades"]


def test_build_snapshot_uses_bot_eod_over_whale_fallback(tmp_path):
    date_dir = tmp_path / "2026-05-08"
    date_dir.mkdir()
    bot_header = (
        "executed_at,underlying_symbol,option_chain_id,side,strike,option_type,expiry,"
        "underlying_price,nbbo_bid,nbbo_ask,ewma_nbbo_bid,ewma_nbbo_ask,price,size,"
        "premium,volume,open_interest,implied_volatility,sector\n"
    )
    bot_row = (
        "2026-05-08T20:00:00Z,XYZ,XYZ260515C00100000,ask,100,call,2026-05-15,"
        "101,1.0,1.2,1.0,1.2,1.1,10,1100,10,100,0.45,Tech\n"
    )
    with zipfile.ZipFile(date_dir / "bot-eod-report-2026-05-08.zip", "w") as zf:
        zf.writestr("bot-eod-report-2026-05-08.csv", bot_header + bot_row)
    (date_dir / "whale_trades_filtered.csv").write_text(
        "executed_at,underlying_symbol,option_type,side,premium\n"
        "2026-05-08T20:00:00Z,XYZ,put,ask,999999\n",
        encoding="utf-8",
    )

    snap = build_daily_snapshot(
        tmp_path,
        "2026-05-08",
        {
            "max_chain_rows_per_day": 10,
            "max_flow_file_mb": 0.0001,
            "bot_eod_cache_dir": str(tmp_path / "cache"),
        },
    )

    xyz = snap.features["XYZ"]
    assert "bot_eod" in xyz["source_flags"]
    assert "option_trades" not in xyz["source_flags"]
    assert xyz["flow_call_ask_premium"] == 1100
    assert xyz["flow_put_ask_premium"] == 0
    assert any(s["reason"] == "bot_eod_present_primary_flow_source" for s in snap.skipped_sources)
    assert snap.counts["bot_eod_rows"] == 1


def test_trade_output_uses_human_readable_long_option_setup():
    row = trade_output_row(
        {
            "classification": "TRADE",
            "ticker": "SNDK",
            "direction": "bullish",
            "pattern_family": "VOL_EXPANSION_CATALYST__BULLISH__LONG_OPTION__TECHNOLOGY",
            "confidence_tier": "PROVEN",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "lead_option_symbol": "SNDK260515C02000000",
            "option_type": "call",
            "strike": 2000.0,
            "expiry": "2026-05-15",
            "entry_range": "7.20-7.30",
            "block_reasons": [],
        }
    )

    assert row["strategy"] == "Long Call Debit"
    assert row["buy_or_sell"] == "BUY"
    assert row["call_or_put"] == "CALL"
    assert row["strike_rates"] == "2000"
    assert row["expiration_date"] == "2026-05-15"
    assert row["trade_setup"] == "BUY CALL SNDK 2000 exp 2026-05-15"
    assert row["occ_symbols"] == "SNDK260515C02000000"


def test_trade_output_uses_human_readable_spread_setup():
    row = trade_output_row(
        {
            "classification": "TRADE",
            "ticker": "XYZ",
            "direction": "bullish",
            "pattern_family": "OI_GAMMA_CONTINUATION__BULLISH__CREDIT_SPREAD__TECHNOLOGY",
            "confidence_tier": "PROVEN",
            "strategy_kind": "credit_spread",
            "strategy_type": "Bull Put Credit Spread",
            "lead_option_symbol": "SELL XYZ260515P00095000 / BUY XYZ260515P00090000",
            "legs_json": (
                '[{"action":"SELL","option_symbol":"XYZ260515P00095000","option_type":"put","strike":95},'
                '{"action":"BUY","option_symbol":"XYZ260515P00090000","option_type":"put","strike":90}]'
            ),
            "expiry": "2026-05-15",
            "entry_credit": 1.0,
            "block_reasons": [],
        }
    )

    assert row["buy_or_sell"] == "SELL / BUY"
    assert row["call_or_put"] == "PUT / PUT"
    assert row["strike_rates"] == "SELL 95 / BUY 90"
    assert row["expiration_date"] == "2026-05-15"
    assert row["trade_setup"] == "SELL PUT XYZ 95 / BUY PUT XYZ 90 exp 2026-05-15"


def test_family_tiers_include_validation_probability_score():
    tiers = assign_family_tiers(
        [
            {
                "pattern_family": "EDGE",
                "signal_count": 40,
                "scored_count": 40,
                "win_count_scored": 28,
                "average_net_r": 0.25,
                "profit_factor": 2.0,
                "worst_losing_streak": 2,
            },
            {
                "pattern_family": "EDGE",
                "signal_count": 40,
                "scored_count": 40,
                "win_count_scored": 24,
                "average_net_r": 0.15,
                "profit_factor": 1.8,
                "worst_losing_streak": 3,
            },
        ],
        [
            {"average_net_r": -0.05, "scored_count": 20},
            {"average_net_r": 0.01, "scored_count": 20},
        ],
    )

    edge = tiers["EDGE"]
    assert edge["validation_success_probability"] == 0.65
    assert edge["validation_failure_probability"] == 0.35
    assert 0.58 < edge["validation_probability_score"] < edge["validation_success_probability"]
    assert "52/80" in edge["probability_evidence"]


def test_classified_trade_carries_probability_fields():
    signal = {
        "date": "2026-05-08",
        "ticker": "XYZ",
        "direction": "bullish",
        "pattern_family": "EDGE",
        "pattern_score": 1.0,
        "classification": "WATCH",
        "block_reasons": [],
        "bid_ask_spread_pct": 0.02,
        "liquidity_volume": 1200,
        "liquidity_open_interest": 1200,
        "earnings_dte": 30,
    }
    snapshot = type("S", (), {"market_regime": {"regime": "RISK_ON"}})()

    rows = classify_daily_signals(
        [signal],
        {
            "EDGE": {
                "confidence_tier": "PROVEN",
                "validation_scored_count": 80,
                "beats_baselines_count": 2,
                "validation_success_probability": 0.65,
                "validation_failure_probability": 0.35,
                "validation_probability_score": 0.59,
                "probability_evidence": "5d OOS scored wins 52/80",
                "validation_note": "unit",
            }
        },
        snapshot,
    )

    assert rows[0]["classification"] == "TRADE"
    assert rows[0]["pattern_success_probability_pct"] == 65.0
    assert rows[0]["pattern_probability_score"] == 59.0
    assert rows[0]["trade_success_probability_pct"] > rows[0]["pattern_success_probability_pct"]
    assert rows[0]["trade_probability_score"] > rows[0]["pattern_probability_score"]


def test_candidate_probability_adjustment_differentiates_same_family_trades():
    signals = [
        {
            "date": "2026-05-08",
            "ticker": "GOOD",
            "direction": "bullish",
            "pattern_family": "EDGE",
            "pattern_score": 5.0,
            "classification": "WATCH",
            "block_reasons": [],
            "bid_ask_spread_pct": 0.0,
            "liquidity_volume": 10000,
            "liquidity_open_interest": 10000,
            "earnings_dte": 60,
        },
        {
            "date": "2026-05-08",
            "ticker": "BAD",
            "direction": "bullish",
            "pattern_family": "EDGE",
            "pattern_score": 1.0,
            "classification": "WATCH",
            "block_reasons": [],
            "bid_ask_spread_pct": 0.30,
            "liquidity_volume": 10,
            "liquidity_open_interest": 10,
            "earnings_dte": 60,
        },
    ]
    snapshot = type("S", (), {"market_regime": {"regime": "RISK_ON"}})()
    tier = {
        "confidence_tier": "PROVEN",
        "validation_scored_count": 80,
        "beats_baselines_count": 2,
        "validation_success_probability": 0.65,
        "validation_failure_probability": 0.35,
        "validation_probability_score": 0.59,
        "probability_evidence": "5d OOS scored wins 52/80",
        "validation_note": "unit",
    }

    rows = classify_daily_signals(signals, {"EDGE": tier}, snapshot)
    by_ticker = {r["ticker"]: r for r in rows}

    assert by_ticker["GOOD"]["trade_success_probability_pct"] > by_ticker["BAD"]["trade_success_probability_pct"]
    assert by_ticker["GOOD"]["trade_probability_score"] > by_ticker["BAD"]["trade_probability_score"]
    assert by_ticker["GOOD"]["pattern_success_probability_pct"] == by_ticker["BAD"]["pattern_success_probability_pct"]
