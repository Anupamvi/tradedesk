import subprocess
import zipfile
from pathlib import Path

import pytest

from uwos.options_pattern_pipeline_v1.macro_geo import (
    SCENARIO_BUCKETS,
    build_macro_geo_bundle,
    build_observability_matrix_rows,
    build_promotion_decision_rows,
    classify_promotion_bucket,
    collect_macro_geo_catalysts,
    decompose_blockers,
)
from uwos.options_pattern_pipeline_v1.core import (
    assign_family_tiers,
    build_artifact_manifest,
    build_catalyst_flow_leaders,
    build_daily_snapshot,
    build_decision_board_rows,
    build_pattern_recommendations,
    build_shadow_ledger_rows,
    build_trade_review_candidates,
    build_validation_splits,
    classify_daily_signals,
    decision_board_fieldnames,
    dedupe_rows_by_ticket,
    empty_validation_bundle,
    generate_signals_for_snapshot,
    normalize_header,
    prepare_decision_rows,
    parse_option_symbol,
    score_signal_horizon,
    source_completeness_for_date,
    sources_for_date,
    trade_output_row,
    pattern_recommendation_output_row,
    catalyst_flow_leader_output_row,
    trade_review_output_row,
    validate_decision_board_rows,
)


class SnapshotStub:
    def __init__(self, features, best_options=None, market_regime=None, signal_date="2026-05-13"):
        self.signal_date = signal_date
        self.features = features
        self.best_options = best_options or {}
        self.option_quotes = {}
        self.market_regime = market_regime or {"regime": "MIXED"}


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


def test_catalyst_flow_leader_rescue_surfaces_top_premium_name_despite_low_ratio():
    feature = {
        "ticker": "AMD",
        "close": 505.0,
        "sector": "Technology",
        "stock_return_1d": 0.08,
        "call_volume_ratio_30d": 0.8,
        "put_volume_ratio_30d": 0.7,
        "premium_bias": 0.02,
        "flow_premium_bias": 0.03,
        "flow_total_premium": 1_600_000_000.0,
        "flow_call_premium_share": 0.81,
        "flow_put_premium_share": 0.19,
        "flow_call_ask_premium_share": 0.81,
        "flow_put_ask_premium_share": 0.19,
        "flow_call_ask_ratio": 0.51,
        "flow_put_ask_ratio": 0.50,
        "hot_total_premium": 900_000_000.0,
        "liquidity_score": 40.0,
        "avg_iv": 0.45,
        "oi_call_diff": 0.0,
        "oi_put_diff": 0.0,
    }
    snapshot = SnapshotStub(
        {"AMD": feature},
        {
            ("AMD", "bullish"): {
                "option_symbol": "AMD260618C00500000",
                "option_type": "call",
                "expiry": "2026-06-18",
                "strike": 500,
                "dte": 23,
                "bid": 39.85,
                "ask": 40.40,
                "mid": 40.125,
                "spread_pct": 0.014,
                "volume": 3000,
                "open_interest": 7000,
                "quote_source": "bot_eod",
            }
        },
        {"regime": "RISK_ON"},
        signal_date="2026-05-26",
    )
    pattern_config = {
        "min_call_volume_ratio": 1.35,
        "min_put_volume_ratio": 1.35,
        "min_hot_premium": 250_000,
        "min_oi_diff": 5_000,
        "max_spread_pct": 0.35,
        "high_iv": 0.75,
        "min_liquidity_score": 8.0,
        "min_ask_side_ratio": 0.52,
        "max_event_dte_without_event_strategy": 2,
    }

    signals = generate_signals_for_snapshot(snapshot, pattern_config, max_signals=1)

    assert signals[0]["ticker"] == "AMD"
    assert signals[0]["base_pattern_family"] == "CATALYST_FLOW_LEADER"
    assert signals[0]["direction"] == "bullish"
    assert signals[0]["entry_range"] == "39.85-40.40"


def test_catalyst_flow_leader_board_keeps_premium_leaders_visible():
    rows = [
        {
            "ticker": "GOOG",
            "direction": "bullish",
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "status": "TRADE_REVIEW",
            "flow_total_premium": 175_000_000.0,
            "hot_total_premium": 120_000_000.0,
            "flow_call_premium_share": 0.81,
            "flow_put_premium_share": 0.19,
            "flow_call_ask_premium_share": 0.72,
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 400,
            "expiry": "2026-06-18",
            "lead_option_symbol": "GOOG260618C00400000",
            "entry_bid": 5.75,
            "entry_ask": 6.10,
            "entry_range": "5.75-6.10",
            "max_risk_per_contract": 610.65,
            "success_probability_pct": 29.12,
            "probability_score": 23.87,
            "expected_R": 0.0307,
            "avg_win_R": 2.4,
            "avg_loss_R": -0.91,
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
        },
        {
            "ticker": "SNDK",
            "direction": "bullish",
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "status": "TRADE_REVIEW",
            "flow_total_premium": 1_896_000_000.0,
            "hot_total_premium": 900_000_000.0,
            "flow_call_premium_share": 0.49,
            "flow_put_premium_share": 0.51,
            "flow_call_ask_premium_share": 0.53,
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 2100,
            "expiry": "2026-06-05",
            "lead_option_symbol": "SNDK260605C02100000",
            "entry_bid": 9.40,
            "entry_ask": 10.70,
            "entry_range": "9.40-10.70",
            "max_risk_per_contract": 1070.65,
            "success_probability_pct": 37.35,
            "probability_score": 34.39,
            "expected_R": 0.4556,
            "avg_win_R": 2.4,
            "avg_loss_R": -0.91,
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
        },
        {
            "ticker": "MSFT",
            "direction": "bullish",
            "base_pattern_family": "OI_GAMMA_CONTINUATION",
            "flow_total_premium": 500_000_000.0,
        },
    ]

    leaders = build_catalyst_flow_leaders(rows)

    assert [row["ticker"] for row in leaders] == ["SNDK", "GOOG"]
    output = catalyst_flow_leader_output_row(leaders[0], 1)
    assert output["trade_setup"] == "BUY CALL SNDK 2100 exp 2026-06-05"
    assert output["entry_limit"] == "debit 9.40-10.70"
    assert output["flow_total_premium"] == 1_896_000_000.0
    assert "Catalyst/flow leader rescue" in output["why_recommended"]


def test_macro_geo_point_in_time_filters_future_captures(tmp_path):
    browser_dir = tmp_path / "2026-05-12" / "browser_text"
    browser_dir.mkdir(parents=True)
    (browser_dir / "browser-text-capture-news-MACRO-LIVE-2026-05-13.txt").write_text(
        "May 13, 2026 live macro context.\n"
        "Premarket tone noted Trump/Xi China talks, trade talks, and AI chip discussion risk. "
        "This is supportive for selected semiconductors and QQQ.\n",
        encoding="utf-8",
    )

    future_for_12 = collect_macro_geo_catalysts(tmp_path, "2026-05-12")
    eligible_for_13 = collect_macro_geo_catalysts(tmp_path, "2026-05-13")

    assert future_for_12
    assert all(not row["as_of_eligible"] for row in future_for_12)
    assert {row["ineligible_reason"] for row in future_for_12} == {"capture_date_after_as_of"}
    assert any(row["as_of_eligible"] for row in eligible_for_13)


def test_macro_geo_extracts_china_trade_ai_chip_catalysts_and_mapping(tmp_path):
    browser_dir = tmp_path / "2026-05-13" / "browser_text"
    browser_dir.mkdir(parents=True)
    (browser_dir / "browser-text-capture-news-SEMIS.txt").write_text(
        "Capture date: 2026-05-13\n"
        "Reuters and AP reported Trump and Xi China summit trade talks with U.S. CEOs. "
        "Nvidia, Micron, Qualcomm, and Intel rebounded on AI chip and semiconductor optimism. "
        "Ticker: MU\n",
        encoding="utf-8",
    )

    catalysts = collect_macro_geo_catalysts(tmp_path, "2026-05-13")
    event_types = {row["event_type"] for row in catalysts}
    mapped = {ticker for row in catalysts for ticker in row["mapped_tickers"] + row["mapped_etfs"]}

    assert "China/US diplomacy" in event_types
    assert "AI chips/semiconductors" in event_types
    assert {"TSLA", "AAPL", "MU", "NVDA", "SMH", "QQQ"} <= mapped


def test_macro_geo_uses_capture_filename_ticker_for_single_name_news(tmp_path):
    browser_dir = tmp_path / "2026-05-26" / "browser_text"
    browser_dir.mkdir(parents=True)
    (browser_dir / "browser-text-capture-news-SNDK-2026-05-26.txt").write_text(
        "SNDK note for the trading desk.\n"
        "SanDisk shares rallied on earnings, guidance, revenue, and storage demand.\n",
        encoding="utf-8",
    )

    catalysts = collect_macro_geo_catalysts(tmp_path, "2026-05-26")
    mapped = {ticker for row in catalysts for ticker in row["mapped_tickers"]}

    assert "SNDK" in mapped


def test_macro_geo_ignores_unrelated_false_positive_headlines(tmp_path):
    browser_dir = tmp_path / "2026-05-13" / "browser_text"
    browser_dir.mkdir(parents=True)
    (browser_dir / "browser-text-capture-news-FOOD.txt").write_text(
        "Capture date: 2026-05-13\n"
        "A classroom discussion covered China history, tariffs in the 1800s, and diplomacy. "
        "The note was about curriculum planning and restaurant menu translations.\n",
        encoding="utf-8",
    )

    assert collect_macro_geo_catalysts(tmp_path, "2026-05-13") == []


def test_macro_geo_uw_confirmation_sector_index_and_watch_rows(tmp_path):
    browser_dir = tmp_path / "2026-05-13" / "browser_text"
    browser_dir.mkdir(parents=True)
    (browser_dir / "browser-text-capture-news-SEMIS.txt").write_text(
        "Capture date: 2026-05-13\n"
        "Semiconductor stocks rallied as Nvidia and Micron AI chip optimism boosted QQQ and SMH.\n",
        encoding="utf-8",
    )
    bullish_feature = {
        "source_flags": ["bot_eod", "hot_chains", "chain_oi"],
        "flow_total_premium": 750000,
        "hot_total_premium": 500000,
        "oi_call_diff": 25000,
        "oi_put_diff": 1000,
        "flow_premium_bias": 0.42,
        "flow_call_ask_ratio": 0.68,
    }
    snapshot = SnapshotStub(
        {
            "NVDA": dict(bullish_feature),
            "SMH": dict(bullish_feature),
        },
        {
            ("NVDA", "bullish"): {
                "bid": 4.9,
                "ask": 5.0,
                "spread_pct": 0.02,
                "volume": 5000,
                "open_interest": 10000,
                "dte": 30,
            },
            ("SMH", "bullish"): {
                "bid": 2.0,
                "ask": 2.02,
                "spread_pct": 0.01,
                "volume": 3000,
                "open_interest": 8000,
                "dte": 30,
            },
        },
    )

    bundle = build_macro_geo_bundle(
        base_dir=tmp_path,
        as_of="2026-05-13",
        snapshots={"2026-05-13": snapshot},
        source_dates=["2026-05-13"],
        daily_rows=[],
        source_complete=True,
        missing_sources=[],
    )
    decisions = {(row["ticker"], row["scenario_bucket"]) for row in bundle["promotion_decisions"]}

    assert ("NVDA", "CATALYST_WATCH") in decisions
    assert ("SMH", "SECTOR_INDEX_CONFIRMED_SETUP") in decisions
    assert any(row["uw_confirmed"] for row in bundle["uw_confirmation"] if row["ticker"] == "NVDA")


def test_macro_geo_promotion_bucket_blocker_decomposition():
    catalyst = {"as_of_eligible": True}
    confirmation = {"uw_confirmed": True, "uw_evidence_found": "bot-EOD options flow"}

    regime_bucket, _ = classify_promotion_bucket(
        catalyst,
        confirmation,
        {"classification": "AVOID", "block_reasons": ["MARKET_REGIME_CONFLICT"]},
        True,
        [],
    )
    validation_bucket, validation_blocker = classify_promotion_bucket(
        catalyst,
        confirmation,
        {
            "classification": "WATCH",
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN", "VALIDATION_EXPECTANCY_NEGATIVE"],
        },
        True,
        [],
    )
    liquidity_bucket, _ = classify_promotion_bucket(
        catalyst,
        confirmation,
        {"classification": "AVOID", "block_reasons": ["BID_ASK_SPREAD_TOO_WIDE"]},
        True,
        [],
    )

    assert regime_bucket == "REGIME_CONFLICTED_SETUP"
    assert validation_bucket == "VALIDATION_BLOCKED_SETUP"
    assert "pattern family not proven" in validation_blocker
    assert liquidity_bucket == "LIQUIDITY_OR_QUOTE_BLOCKED_SETUP"
    assert decompose_blockers(["LIMITED_OUT_OF_SAMPLE_SAMPLE", "DOES_NOT_BEAT_TWO_BASELINES"]) == [
        "sample size too small",
        "does not beat baselines",
    ]


def test_macro_geo_multi_day_continuation_without_future_leakage(tmp_path):
    for d in ("2026-05-11", "2026-05-12", "2026-05-14"):
        browser_dir = tmp_path / d / "browser_text"
        browser_dir.mkdir(parents=True)
        browser_dir.joinpath(f"browser-text-capture-news-SEMIS-{d}.txt").write_text(
            f"Capture date: {d}\n"
            "Semiconductor and AI chip optimism supported Nvidia and Micron, with QQQ confirmation.\n",
            encoding="utf-8",
        )
    weak = {
        "source_flags": ["hot_chains"],
        "hot_total_premium": 120000,
        "oi_call_diff": 2000,
        "flow_premium_bias": 0.2,
    }
    strong = {
        "source_flags": ["bot_eod", "hot_chains", "chain_oi"],
        "flow_total_premium": 600000,
        "hot_total_premium": 400000,
        "oi_call_diff": 20000,
        "flow_premium_bias": 0.3,
    }
    bundle = build_macro_geo_bundle(
        base_dir=tmp_path,
        as_of="2026-05-12",
        snapshots={
            "2026-05-11": SnapshotStub({"NVDA": weak}),
            "2026-05-12": SnapshotStub({"NVDA": strong}),
        },
        source_dates=["2026-05-11", "2026-05-12"],
        daily_rows=[],
        source_complete=True,
        missing_sources=[],
    )

    continuing = [r for r in bundle["promotion_decisions"] if r["scenario_bucket"] == "MULTI_DAY_CONTINUING_CATALYST"]
    assert continuing
    assert "2026-05-14" not in continuing[0]["capture_date"]
    assert "improved" in continuing[0]["uw_evidence_found"]


def test_source_incomplete_handling_lists_missing_files(tmp_path):
    completeness = source_completeness_for_date(tmp_path, "2026-05-14")

    assert completeness["source_complete"] is False
    assert any("date folder" in item for item in completeness["missing_sources"])
    assert any("stock-screener-2026-05-14" in item for item in completeness["missing_sources"])
    assert any("bot-eod-report-2026-05-14" in item for item in completeness["missing_sources"])


def test_observability_matrix_has_every_required_scenario():
    rows = build_observability_matrix_rows("2026-05-13", [])

    assert {row["scenario_name"] for row in rows} == set(SCENARIO_BUCKETS)
    assert all(row["expected_behavior"] for row in rows)


def test_trade_review_board_keeps_reviewable_setups_visible():
    rows = [
        {
            "classification": "WATCH",
            "ticker": "TSLA",
            "direction": "bearish",
            "confidence_tier": "RESEARCH_ONLY",
            "probability_score": 49.97,
            "success_probability_pct": 55.33,
            "pattern_score": 9.0,
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN", "VALIDATION_EXPECTANCY_NEGATIVE"],
            "strategy_kind": "long_option",
            "strategy_type": "Long Put Debit",
            "lead_option_symbol": "TSLA260618P00410000",
            "expiry": "2026-06-18",
            "option_type": "put",
            "strike": 410,
            "entry_range": "20.30-20.45",
            "max_risk_per_contract": 2045.65,
        },
        {
            "classification": "AVOID",
            "ticker": "MSFT",
            "direction": "bullish",
            "confidence_tier": "PROMISING",
            "probability_score": 49.91,
            "success_probability_pct": 52.33,
            "pattern_score": 8.5,
            "block_reasons": ["MARKET_REGIME_CONFLICT", "PATTERN_VALIDATION_NOT_PROVEN"],
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "lead_option_symbol": "MSFT260618C00420000",
            "expiry": "2026-06-18",
            "option_type": "call",
            "strike": 420,
            "entry_range": "16.50-16.75",
            "max_risk_per_contract": 1675.65,
        },
        {
            "classification": "AVOID",
            "ticker": "XYZ",
            "direction": "bullish",
            "confidence_tier": "RESEARCH_ONLY",
            "probability_score": 60.0,
            "success_probability_pct": 65.0,
            "pattern_score": 20.0,
            "block_reasons": ["NO_TRADEABLE_OPTION_QUOTE"],
        },
    ]

    candidates = build_trade_review_candidates(rows)
    output = [trade_review_output_row(row) for row in candidates]

    assert [row["ticker"] for row in output] == ["TSLA", "MSFT"]
    assert output[0]["review_status"] == "TACTICAL_REVIEW"
    assert output[0]["trade_setup"] == "BUY PUT TSLA 410 exp 2026-06-18"
    assert output[1]["review_status"] == "MACRO_CONFLICT_REVIEW"
    assert "regime alignment" in output[1]["promotion_needed"]


def test_ticket_outputs_dedupe_same_contract_family_duplicates():
    base_row = {
        "classification": "WATCH",
        "ticker": "TSLA",
        "direction": "bearish",
        "confidence_tier": "RESEARCH_ONLY",
        "success_probability_pct": 55.33,
        "pattern_score": 9.0,
        "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
        "strategy_kind": "long_option",
        "strategy_type": "Long Put Debit",
        "lead_option_symbol": "TSLA260618P00410000",
        "expiry": "2026-06-18",
        "option_type": "put",
        "strike": 410,
        "entry_range": "20.30-20.45",
    }
    weaker_duplicate = dict(base_row, probability_score=44.90, pattern_family="VOL_EXPANSION_CATALYST")
    stronger_duplicate = dict(base_row, probability_score=49.97, pattern_family="OI_GAMMA_CONTINUATION")
    other_ticket = dict(base_row, ticker="SLV", lead_option_symbol="SLV260618P00070000", strike=70, probability_score=35.0)

    deduped = dedupe_rows_by_ticket([weaker_duplicate, other_ticket, stronger_duplicate])

    assert len(deduped) == 2
    assert deduped[0]["ticker"] == "TSLA"
    assert deduped[0]["pattern_family"] == "OI_GAMMA_CONTINUATION"


def test_macro_promotion_uses_direction_matched_daily_row():
    catalyst = {
        "catalyst_id": "c1",
        "as_of_eligible": True,
        "direction_bias": "bullish",
        "mapped_tickers": ["AAPL"],
        "mapped_etfs": [],
        "event_type": "China/US diplomacy",
    }
    confirmation = {
        "catalyst_id": "c1",
        "ticker": "AAPL",
        "uw_confirmed": True,
        "uw_evidence_found": "bot-EOD options flow; bullish direction; liquid expiry/quote",
        "uw_direction": "bullish",
        "catalyst_direction_bias": "bullish",
        "direction_confirmed": True,
        "sector_etf_confirmation": False,
    }
    daily_rows = [
        {
            "ticker": "AAPL",
            "direction": "bearish",
            "classification": "WATCH",
            "pattern_family": "BEARISH_WATCH_SHOULD_NOT_MATCH",
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
            "probability_score": 60.0,
        },
        {
            "ticker": "AAPL",
            "direction": "bullish",
            "classification": "AVOID",
            "pattern_family": "BULLISH_MATCH",
            "block_reasons": ["MARKET_REGIME_CONFLICT"],
            "probability_score": 45.0,
        },
    ]

    rows = build_promotion_decision_rows([catalyst], [confirmation], daily_rows, True, [], "2026-05-18")

    assert rows[0]["daily_direction"] == "bullish"
    assert rows[0]["pattern_family"] == "BULLISH_MATCH"
    assert rows[0]["scenario_bucket"] == "REGIME_CONFLICTED_SETUP"


def test_macro_promotion_does_not_use_opposite_direction_daily_row():
    catalyst = {
        "catalyst_id": "c1",
        "as_of_eligible": True,
        "direction_bias": "bullish",
        "mapped_tickers": ["AAPL"],
        "mapped_etfs": [],
        "event_type": "China/US diplomacy",
    }
    confirmation = {
        "catalyst_id": "c1",
        "ticker": "AAPL",
        "uw_confirmed": True,
        "uw_evidence_found": "bot-EOD options flow; bullish direction",
        "uw_direction": "bullish",
        "catalyst_direction_bias": "bullish",
        "direction_confirmed": True,
        "sector_etf_confirmation": False,
    }
    daily_rows = [
        {
            "ticker": "AAPL",
            "direction": "bearish",
            "classification": "WATCH",
            "pattern_family": "BEARISH_ONLY",
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
            "probability_score": 60.0,
        }
    ]

    rows = build_promotion_decision_rows([catalyst], [confirmation], daily_rows, True, [], "2026-05-18")

    assert rows[0]["daily_classification"] == ""
    assert rows[0]["daily_direction"] == ""
    assert rows[0]["scenario_bucket"] == "CATALYST_WATCH"


def test_strict_decision_layer_blocks_auto_approval_on_negative_edge():
    daily_rows = [
        {
            "date": "2026-05-18",
            "classification": "TRADE",
            "ticker": "XYZ",
            "direction": "bullish",
            "pattern_family": "EDGE",
            "confidence_tier": "PROVEN",
            "probability_score": 60.0,
            "success_probability_pct": 60.0,
            "pattern_score": 10.0,
            "block_reasons": [],
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "lead_option_symbol": "XYZ260619C00100000",
            "expiry": "2026-06-19",
            "option_type": "call",
            "strike": 100,
            "entry_bid": 1.0,
            "entry_ask": 1.1,
            "entry_range": "1.00-1.10",
            "bid_ask_spread_pct": 0.09,
            "max_risk_per_contract": 110.65,
            "liquidity_volume": 1000,
            "liquidity_open_interest": 1000,
            "dte": 20,
            "quote_source": "bot_eod",
        }
    ]
    validation_bundle = empty_validation_bundle()
    validation_bundle["family_tiers"] = {
        "EDGE": {
            "confidence_tier": "PROVEN",
            "validation_scored_count": 80,
            "validation_win_count": 40,
            "validation_success_probability": 0.5,
            "validation_probability_score": 0.48,
            "validation_average_net_r": -0.05,
            "validation_profit_factor": 0.8,
            "beats_baselines_count": 2,
        }
    }

    rows, _ = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {"risk_config": {"allow_conservative_historical_quote_for_auto": True}},
    )

    assert rows[0]["status"] == "AVOID"
    assert rows[0]["classification"] == "AVOID"
    assert "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS" in rows[0]["block_reasons"]
    assert "PROFIT_FACTOR_BELOW_AUTO_APPROVAL" in rows[0]["block_reasons"]


def test_validated_regime_edge_surfaces_trade_review_instead_of_blanket_avoid():
    daily_rows = [
        {
            "date": "2026-05-19",
            "classification": "WATCH",
            "ticker": "INTC",
            "direction": "bullish",
            "pattern_family": "OI_GAMMA_CONTINUATION__BULLISH__LONG_OPTION__TECHNOLOGY",
            "confidence_tier": "PROMISING",
            "probability_score": 49.0,
            "success_probability_pct": 52.0,
            "pattern_score": 10.0,
            "block_reasons": [
                "PATTERN_VALIDATION_NOT_PROVEN",
                "MARKET_REGIME_CONFLICT",
                "CALIBRATION_SCORE_MISSING_OR_WEAK",
                "CONFIDENCE_BAND_TOO_WEAK",
            ],
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "lead_option_symbol": "INTC260619C00110000",
            "expiry": "2026-06-19",
            "option_type": "call",
            "strike": 110,
            "entry_bid": 10.6,
            "entry_ask": 10.75,
            "entry_range": "10.60-10.75",
            "bid_ask_spread_pct": 0.014,
            "max_risk_per_contract": 1075.65,
            "liquidity_volume": 1000,
            "liquidity_open_interest": 1000,
            "dte": 31,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_OFF",
            "market_regime": "RISK_OFF",
        }
    ]
    validation_bundle = empty_validation_bundle()
    family = "OI_GAMMA_CONTINUATION__BULLISH__LONG_OPTION__TECHNOLOGY"
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROMISING",
            "validation_scored_count": 35,
            "validation_win_count": 22,
            "validation_success_probability": 22 / 35,
            "validation_probability_score": 0.44,
            "validation_average_net_r": 0.20,
            "validation_profit_factor": 2.0,
            "beats_baselines_count": 3,
        }
    }
    validation_bundle["outcomes"] = [
        {
            "split": "cumulative_to_2026-05_holdout",
            "sample": "VALIDATION",
            "horizon": "5d",
            "signal_date": "2026-05-01",
            "pattern_family": family,
            "market_regime": "RISK_OFF",
            "status": "SCORED",
            "net_r": 0.45 if idx < 22 else -0.18,
            "win": int(idx < 22),
        }
        for idx in range(35)
    ]

    rows, _ = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    assert rows[0]["status"] == "TRADE_REVIEW"
    assert rows[0]["classification"] == "WATCH"
    assert rows[0]["edge_review_reason"] == "VALIDATED_FAMILY_AND_REGIME_EDGE_REVIEW"
    assert "MARKET_REGIME_CONFLICT" in rows[0]["block_reasons"]
    assert trade_review_output_row(rows[0])["review_status"] == "VALIDATED_EDGE_REVIEW"
    recommendations = build_pattern_recommendations([], rows)
    assert len(recommendations) == 1
    recommendation = pattern_recommendation_output_row(recommendations[0], 1)
    assert recommendation["recommendation"] == "PATTERN_RECOMMENDATION"
    assert recommendation["entry_limit"] == "debit 10.60-10.75"
    assert recommendation["breakeven_success_probability_pct"] == 28.57
    assert recommendation["edge_vs_breakeven_pct"] > 0
    assert "Validated historical edge" in recommendation["why_recommended"]


def test_decision_board_schema_accepts_no_trade_contract():
    rows = build_decision_board_rows([], "2026-05-18", True, "NO_TRADE", {"daily_report": "x"})

    assert rows[0]["status"] == "NO_TRADE"
    assert "candidate_id" in decision_board_fieldnames()
    assert validate_decision_board_rows(rows) == []


def test_shadow_ledger_tracks_trade_review_rows():
    board_rows = [
        {
            "run_date": "2026-05-18",
            "status": "TRADE_REVIEW",
            "candidate_id": "abc",
            "full_ticket": "BUY CALL XYZ 100 exp 2026-06-19",
            "entry": "debit 1.00-1.10",
            "target": 100,
            "stop": "50% stop",
            "time_stop": "5 trading days",
            "kill_switch_triggered": "",
        }
    ]

    rows = build_shadow_ledger_rows("2026-05-18", board_rows, empty_validation_bundle())

    assert rows[0]["candidate_id"] == "abc"
    assert rows[0]["exit_status"] == "OPEN_SHADOW_PENDING"


def test_artifact_manifest_contains_reproducibility_contract(tmp_path):
    manifest = build_artifact_manifest(
        "2026-05-18",
        tmp_path,
        {"seed": 7, "risk_config_path": "cfg.json", "risk_config_hash": "hash"},
        {
            "command": "python3 -m uwos.options_pattern_pipeline_v1",
            "source_files_for_as_of": [],
            "source_counts_by_date": {},
            "skipped_sources_for_as_of": [],
        },
        {"decision_board_csv": "decision_board.csv"},
        [],
        1.25,
    )

    assert manifest["artifact_schema_version"] == "artifact_manifest_v1"
    assert manifest["deterministic_seed"] == 7
    assert manifest["artifact_paths"]["decision_board_csv"] == "decision_board.csv"


def test_frozen_v1_backup_remains_unchanged_against_baseline_tag():
    if not Path(".git").exists():
        pytest.skip("git metadata not available")

    result = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "options-pattern-pipeline-v1",
            "--",
            "uwos/options_pattern_pipeline_v1_frozen_v1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
