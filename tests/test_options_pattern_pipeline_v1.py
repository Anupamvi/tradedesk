import subprocess
import zipfile
from pathlib import Path

import pytest

from uwos.options_pattern_pipeline_v1.macro_geo import (
    SCENARIO_BUCKETS,
    build_macro_geo_bundle,
    build_observability_matrix_rows,
    classify_promotion_bucket,
    collect_macro_geo_catalysts,
    decompose_blockers,
)
from uwos.options_pattern_pipeline_v1.core import (
    assign_family_tiers,
    build_daily_snapshot,
    build_validation_splits,
    classify_daily_signals,
    normalize_header,
    parse_option_symbol,
    score_signal_horizon,
    source_completeness_for_date,
    sources_for_date,
    trade_output_row,
)


class SnapshotStub:
    def __init__(self, features, best_options=None, market_regime=None):
        self.features = features
        self.best_options = best_options or {}
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
