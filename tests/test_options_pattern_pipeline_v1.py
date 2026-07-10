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
    clean_ticker as clean_macro_ticker,
    decompose_blockers,
)
from uwos.options_pattern_pipeline_v1.core import (
    DEFAULT_RISK_CONFIG,
    GOAL_MAJOR_REQUIRED_TICKERS,
    PIPELINE_RELEASED_AT,
    PIPELINE_VERSION,
    PREVIOUS_PIPELINE_VERSIONS,
    assign_family_tiers,
    balanced_non_ready_trend_rows,
    blocker_text,
    build_artifact_manifest,
    build_signal,
    build_catalyst_flow_leaders,
    build_daily_snapshot,
    build_decision_board_rows,
    build_directional_edge_diagnostic_rows,
    build_directional_scenario_goal_row,
    build_goal_evidence_rows,
    build_pattern_recommendations,
    build_source_ticker_coverage_rows,
    build_shadow_ledger_rows,
    build_scout_call_candidates,
    build_target_ready_candidates,
    build_trade_review_candidates,
    build_ticker_trend_edge_rows,
    build_validation_splits,
    classify_daily_signals,
    daily_trade_decision,
    decision_board_fieldnames,
    dedupe_rows_by_ticket,
    empty_validation_bundle,
    final_verdict,
    generate_signals_for_snapshot,
    goal_evidence_overall_status,
    missed_mover_bucket,
    normalize_header,
    prepare_decision_rows,
    parse_option_symbol,
    resolve_run_verdict,
    run_historical_validation,
    score_signal_horizon,
    select_signal_set,
    source_coverage_quote,
    source_coverage_setup_fields,
    source_complete_dates,
    source_completeness_for_date,
    sources_for_date,
    trade_fieldnames,
    trade_output_row,
    tradeable_gap_quote_eligible,
    trend_edge_strategy_fields,
    ticker_trend_no_edge_reason,
    target_ready_output_row,
    scout_call_output_row,
    pattern_recommendation_output_row,
    catalyst_flow_leader_output_row,
    trade_review_output_row,
    validate_decision_board_rows,
    validation_detail_fieldnames,
)


class SnapshotStub:
    def __init__(self, features, best_options=None, market_regime=None, signal_date="2026-05-13", option_quotes=None):
        self.signal_date = signal_date
        self.features = features
        self.best_options = best_options or {}
        self.option_quotes = option_quotes or {}
        self.market_regime = market_regime or {"regime": "MIXED"}


def test_default_goal_major_required_tickers_match_user_coverage_scope():
    assert set(GOAL_MAJOR_REQUIRED_TICKERS) <= set(DEFAULT_RISK_CONFIG["goal_required_coverage_tickers"])
    assert set(GOAL_MAJOR_REQUIRED_TICKERS) == {
        "AAPL",
        "NVDA",
        "MSFT",
        "GOOG",
        "GOOGL",
        "PLTR",
        "AMD",
        "MU",
        "META",
        "HOOD",
        "NOW",
    }


def test_options_pattern_pipeline_version_retains_previous_live_version():
    assert PIPELINE_VERSION == "options_pattern_pipeline_v1.5-promotion-bridge-20260707-000000"
    assert PREVIOUS_PIPELINE_VERSIONS == (
        "options_pattern_pipeline_v1.2",
        "options_pattern_pipeline_v1.3",
        "options_pattern_pipeline_v1.4",
    )
    assert PIPELINE_RELEASED_AT == "2026-07-07T00:00:00-07:00"


def test_final_verdict_requires_today_actionable_for_production_ready():
    validation_bundle = {
        "family_tiers": {
            "PROVEN_FAMILY": {"confidence_tier": "PROVEN"},
            "PROMISING_FAMILY": {"confidence_tier": "PROMISING"},
        },
        "validation_scorecard": [{"pattern_family": "PROVEN_FAMILY"}],
    }

    assert final_verdict(validation_bundle, [], actionable_rows=[]) == "USABLE_NEEDS_MORE_VALIDATION"
    assert final_verdict(validation_bundle, [], actionable_rows=[{"ticker": "AAPL"}]) == "PRODUCTION_READY"


def test_final_verdict_is_not_yet_proven_without_proven_family_or_actionable():
    validation_bundle = {
        "family_tiers": {
            "PROMISING_FAMILY": {"confidence_tier": "PROMISING"},
            "RESEARCH_FAMILY": {"confidence_tier": "RESEARCH_ONLY"},
        },
        "validation_scorecard": [{"pattern_family": "PROMISING_FAMILY"}],
    }

    assert final_verdict(validation_bundle, [], actionable_rows=[]) == "NOT_YET_PROVEN"


def test_resolve_run_verdict_prefers_written_metadata_verdict(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text('{"verdict": "PRODUCTION_READY"}')

    assert resolve_run_verdict({"family_tiers": {}}, [], {"metadata": str(metadata_path)}) == "PRODUCTION_READY"


def test_resolve_run_verdict_falls_back_when_metadata_is_unreadable(tmp_path):
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text("{")
    validation_bundle = {"family_tiers": {"EDGE": {"confidence_tier": "PROVEN"}}}

    assert (
        resolve_run_verdict(validation_bundle, [], {"metadata": str(metadata_path)})
        == "USABLE_NEEDS_MORE_VALIDATION"
    )


def test_macro_ticker_cleaner_strips_news_filename_prefix():
    assert clean_macro_ticker("NEWS-IBM") == "IBM"
    assert clean_macro_ticker("CAPTURE-CRWD") == "CRWD"


def test_source_coverage_surfaces_high_flow_ticker_that_misses_decision_board():
    snapshot = SnapshotStub(
        features={
            "IBM": {
                "ticker": "IBM",
                "close": 282.0,
                "source_flags": {"bot_eod", "hot_chains", "stock_screener"},
                "flow_total_premium": 94_315_371.0,
                "hot_total_premium": 57_753_450.0,
                "call_premium": 79_222_529.0,
                "put_premium": 15_092_842.0,
                "flow_call_premium_share": 0.84,
                "flow_put_premium_share": 0.16,
                "flow_call_ask_premium_share": 0.51,
                "flow_put_ask_premium_share": 0.08,
                "flow_premium_bias": 0.68,
                "call_volume_ratio_30d": 2.90,
                "put_volume_ratio_30d": 1.06,
                "oi_call_diff": 4500,
                "oi_put_diff": 300,
            }
        },
        best_options={
            ("IBM", "bullish"): {
                "ticker": "IBM",
                "direction": "bullish",
                "option_symbol": "IBM260618C00300000",
                "option_type": "call",
                "expiry": "2026-06-18",
                "strike": 300.0,
                "dte": 15,
                "bid": 4.1,
                "ask": 4.3,
                "mid": 4.2,
                "volume": 900,
                "open_interest": 1200,
                "premium": 387000.0,
                "spread_pct": 0.048,
                "quote_source": "bot_eod",
                "selection_score": 20.0,
            }
        },
    )

    rows = build_source_ticker_coverage_rows(snapshot, {"max_spread_pct": 0.35}, [])

    assert rows[0]["ticker"] == "IBM"
    assert rows[0]["decision_surface_status"] == "NOT_SURFACED"
    assert "below 100M catalyst-flow-leader threshold" in rows[0]["source_gap_reason"]
    assert rows[0]["strategy"] == "Long Call Debit"
    assert rows[0]["call_or_put"] == "CALL"
    assert rows[0]["strike_rates"] == "300"
    assert rows[0]["expiration_date"] == "2026-06-18"
    assert rows[0]["trade_legs"] == "Buy 1 IBM 2026-06-18 300C @ debit 4.10-4.30 limit"


def test_source_coverage_uses_decision_ticket_when_ticker_surfaces():
    snapshot = SnapshotStub(
        features={
            "IBM": {
                "ticker": "IBM",
                "close": 282.0,
                "source_flags": {"bot_eod", "hot_chains", "stock_screener"},
                "flow_total_premium": 94_315_371.0,
                "hot_total_premium": 57_753_450.0,
                "call_premium": 79_222_529.0,
                "put_premium": 15_092_842.0,
                "flow_call_premium_share": 0.84,
                "flow_put_premium_share": 0.16,
                "flow_call_ask_premium_share": 0.51,
                "flow_put_ask_premium_share": 0.08,
                "flow_premium_bias": 0.68,
            }
        },
        best_options={
            ("IBM", "bullish"): {
                "ticker": "IBM",
                "direction": "bullish",
                "option_symbol": "IBM260618C00300000",
                "option_type": "call",
                "expiry": "2026-06-18",
                "strike": 300.0,
                "dte": 15,
                "bid": 4.1,
                "ask": 4.3,
                "mid": 4.2,
                "volume": 900,
                "open_interest": 1200,
                "premium": 387000.0,
                "spread_pct": 0.048,
                "quote_source": "bot_eod",
                "selection_score": 20.0,
            }
        },
    )
    decision_rows = [
        {
            "ticker": "IBM",
            "direction": "bullish",
            "status": "TRADE_REVIEW",
            "classification": "WATCH",
            "pattern_family": "SOURCE_PREMIUM_COVERAGE_RESCUE__BULLISH__LONG_OPTION__TECHNOLOGY",
            "strategy_type": "Long Call Debit",
            "strategy_kind": "long_option",
            "lead_option_symbol": "IBM260618C00260000",
            "option_type": "CALL",
            "strike": 260.0,
            "expiry": "2026-06-18",
            "entry_range": "12.70-12.70",
            "quote_source": "bot_eod",
            "liquidity_volume": 1450,
            "liquidity_open_interest": 2180,
            "bid_ask_spread_pct": 0.0,
            "probability_score": 20.08,
            "expected_R": 0.081568,
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
        }
    ]

    rows = build_source_ticker_coverage_rows(snapshot, {"max_spread_pct": 0.35}, decision_rows)

    assert rows[0]["ticker"] == "IBM"
    assert rows[0]["decision_surface_status"] == "TRADE_REVIEW"
    assert rows[0]["decision_pattern_family"] == "SOURCE_PREMIUM_COVERAGE_RESCUE__BULLISH__LONG_OPTION__TECHNOLOGY"
    assert rows[0]["strategy"] == "Long Call Debit"
    assert rows[0]["call_or_put"] == "CALL"
    assert rows[0]["strike_rates"] == "260"
    assert rows[0]["expiration_date"] == "2026-06-18"
    assert rows[0]["trade_legs"] == "Buy 1 IBM 2026-06-18 260C @ debit 12.70-12.70 limit"
    assert rows[0]["entry_limit"] == "debit 12.70-12.70"
    assert rows[0]["quote_volume"] == 1450


def test_source_coverage_includes_below_threshold_required_ticker():
    snapshot = SnapshotStub(
        features={
            "IBM": {
                "ticker": "IBM",
                "close": 282.0,
                "source_flags": {"bot_eod", "hot_chains"},
                "flow_total_premium": 11_710_000.0,
                "hot_total_premium": 8_000_000.0,
                "flow_call_premium_share": 0.84,
                "flow_put_premium_share": 0.16,
                "flow_call_ask_premium_share": 0.51,
                "flow_put_ask_premium_share": 0.08,
                "flow_premium_bias": 0.68,
            }
        },
        best_options={
            ("IBM", "bullish"): {
                "ticker": "IBM",
                "direction": "bullish",
                "option_symbol": "IBM260618C00300000",
                "option_type": "call",
                "expiry": "2026-06-18",
                "strike": 300.0,
                "dte": 15,
                "bid": 4.1,
                "ask": 4.3,
                "mid": 4.2,
                "volume": 900,
                "open_interest": 1200,
                "premium": 387000.0,
                "spread_pct": 0.048,
                "quote_source": "bot_eod",
                "selection_score": 20.0,
            }
        },
    )

    rows = build_source_ticker_coverage_rows(
        snapshot,
        {"max_spread_pct": 0.35},
        [],
        required_tickers=["IBM"],
    )

    assert len(rows) == 1
    assert rows[0]["ticker"] == "IBM"
    assert rows[0]["decision_surface_status"] == "NOT_SURFACED"
    assert "required coverage ticker below 50000000 high-source threshold" in rows[0]["source_gap_reason"]
    assert "source_total_premium=1.171e+07" in rows[0]["source_gap_reason"]
    assert rows[0]["trade_legs"] == "Buy 1 IBM 2026-06-18 300C @ debit 4.10-4.30 limit"


def test_source_premium_near_miss_generates_validation_candidate():
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
    snapshot = SnapshotStub(
        features={
            "IBM": {
                "date": "2026-05-28",
                "ticker": "IBM",
                "close": 282.0,
                "source_flags": {"bot_eod", "stock_screener"},
                "flow_total_premium": 94_315_371.0,
                "hot_total_premium": 57_753_450.0,
                "call_premium": 79_222_529.0,
                "put_premium": 15_092_842.0,
                "flow_call_premium_share": 0.84,
                "flow_put_premium_share": 0.16,
                "flow_call_ask_premium_share": 0.80,
                "flow_put_ask_premium_share": 0.07,
                "flow_premium_bias": 0.67,
                "call_volume_ratio_30d": 1.1,
                "put_volume_ratio_30d": 0.8,
                "liquidity_score": 25.0,
            }
        },
        best_options={},
        option_quotes={
            "IBM260717C00300000": {
                "ticker": "IBM",
                "direction": "bullish",
                "option_symbol": "IBM260717C00300000",
                "option_type": "call",
                "expiry": "2026-07-17",
                "strike": 300.0,
                "dte": 35,
                "bid": 5.4,
                "ask": 5.5,
                "mid": 5.45,
                "volume": 2600,
                "open_interest": 5594,
                "premium": 1_430_000.0,
                "spread_pct": 0.018,
                "quote_source": "bot_eod",
            }
        },
        signal_date="2026-05-28",
    )

    signals = generate_signals_for_snapshot(snapshot, pattern_config, max_signals=1)

    assert signals[0]["ticker"] == "IBM"
    assert signals[0]["base_pattern_family"] == "SOURCE_PREMIUM_COVERAGE_RESCUE"
    assert signals[0]["entry_range"] == "5.40-5.50"


def test_source_rescue_signals_survive_rank_cap():
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
    features = {
        "LOUD": {
            "date": "2026-05-28",
            "ticker": "LOUD",
            "close": 100.0,
            "source_flags": {"stock_screener"},
            "call_volume_ratio_30d": 80.0,
            "put_volume_ratio_30d": 0.2,
            "premium_bias": 0.50,
            "flow_premium_bias": 0.0,
            "flow_call_ask_ratio": 0.90,
            "hot_total_premium": 100_000_000.0,
            "liquidity_score": 20.0,
        },
        "CRWD": {
            "date": "2026-05-28",
            "ticker": "CRWD",
            "close": 480.0,
            "source_flags": {"bot_eod", "hot_chains", "stock_screener"},
            "flow_total_premium": 110_377_795.0,
            "hot_total_premium": 27_566_565.0,
            "call_premium": 91_551_507.0,
            "put_premium": 18_826_288.0,
            "flow_call_premium_share": 0.82,
            "flow_put_premium_share": 0.18,
            "flow_call_ask_premium_share": 0.86,
            "flow_put_ask_premium_share": 0.06,
            "flow_premium_bias": 0.65,
            "call_volume_ratio_30d": 0.8,
            "put_volume_ratio_30d": 0.7,
            "liquidity_score": 22.0,
        },
    }
    best_options = {
        ("LOUD", "bullish"): {
            "ticker": "LOUD",
            "direction": "bullish",
            "option_symbol": "LOUD260618C00120000",
            "option_type": "call",
            "expiry": "2026-06-18",
            "strike": 120.0,
            "dte": 15,
            "bid": 4.0,
            "ask": 4.2,
            "mid": 4.1,
            "volume": 1000,
            "open_interest": 2000,
            "premium": 420_000.0,
            "spread_pct": 0.049,
            "quote_source": "bot_eod",
        },
        ("CRWD", "bullish"): {
            "ticker": "CRWD",
            "direction": "bullish",
            "option_symbol": "CRWD260618C00820000",
            "option_type": "call",
            "expiry": "2026-06-18",
            "strike": 820.0,
            "dte": 15,
            "bid": 6.95,
            "ask": 8.50,
            "mid": 7.725,
            "volume": 639,
            "open_interest": 935,
            "premium": 543_150.0,
            "spread_pct": 0.2006,
            "quote_source": "bot_eod",
        },
    }
    snapshot = SnapshotStub(features, best_options, signal_date="2026-05-28")

    signals = generate_signals_for_snapshot(snapshot, pattern_config, max_signals=1)

    assert len(signals) == 2
    assert signals[0]["ticker"] == "LOUD"
    assert any(s["ticker"] == "CRWD" and s["base_pattern_family"] == "CATALYST_FLOW_LEADER" for s in signals)


def test_select_signal_set_retains_bearish_scenario_floor_when_bullish_dominates():
    signals = [
        {
            "date": "2026-05-28",
            "ticker": f"BULL{i}",
            "direction": "bullish",
            "pattern_family": "BULLISH_FLOW_EXPANSION",
            "base_pattern_family": "BULLISH_FLOW_EXPANSION",
            "pattern_score": 100.0 - i,
            "hot_total_premium": 1_000_000.0,
        }
        for i in range(30)
    ]
    signals.extend(
        [
            {
                "date": "2026-05-28",
                "ticker": f"BEAR{i}",
                "direction": "bearish",
                "pattern_family": "BEARISH_PUT_FLOW_EXPANSION",
                "base_pattern_family": "BEARISH_PUT_FLOW_EXPANSION",
                "pattern_score": 10.0 - i,
                "hot_total_premium": 500_000.0,
            }
            for i in range(4)
        ]
    )

    selected = select_signal_set(signals, max_signals=10, source_rescue_max_extra=0, tradeable_gap_max_extra=0)

    bearish = [row for row in selected if row["direction"] == "bearish"]
    assert len(bearish) == 4
    assert {row["ticker"] for row in bearish} == {"BEAR0", "BEAR1", "BEAR2", "BEAR3"}


def test_select_signal_set_zero_max_is_uncapped_for_acceptance_runs():
    signals = [
        {
            "date": "2026-05-28",
            "ticker": f"T{i}",
            "direction": "bullish",
            "pattern_family": "BULLISH_FLOW_EXPANSION",
            "base_pattern_family": "BULLISH_FLOW_EXPANSION",
            "strategy_kind": "long_option",
            "pattern_score": 100.0 - i,
            "hot_total_premium": 1_000_000.0 - i,
        }
        for i in range(5)
    ]

    selected = select_signal_set(signals, max_signals=0, source_rescue_max_extra=0, tradeable_gap_max_extra=0)

    assert [row["ticker"] for row in selected] == ["T0", "T1", "T2", "T3", "T4"]


def test_select_signal_set_rescues_opposite_direction_for_same_ticker():
    signals = [
        {
            "date": "2026-05-28",
            "ticker": "AMD",
            "direction": "bullish",
            "pattern_family": "BULLISH_FLOW_EXPANSION",
            "base_pattern_family": "BULLISH_FLOW_EXPANSION",
            "pattern_score": 100.0,
            "hot_total_premium": 1_000_000.0,
        },
        {
            "date": "2026-05-28",
            "ticker": "AMD",
            "direction": "bearish",
            "pattern_family": "TRADEABLE_SOURCE_GAP_RESCUE",
            "base_pattern_family": "TRADEABLE_SOURCE_GAP_RESCUE",
            "pattern_score": 1.0,
            "hot_total_premium": 25_000.0,
            "source_total_premium": 100_000.0,
            "lead_option_symbol": "AMD260618P00500000",
            "bid_ask_spread_pct": 0.10,
            "liquidity_volume": 500,
        },
    ]

    selected = select_signal_set(signals, max_signals=1, source_rescue_max_extra=0, tradeable_gap_max_extra=5)

    assert {row["direction"] for row in selected if row["ticker"] == "AMD"} == {"bullish", "bearish"}


def test_source_rescue_extra_rows_can_be_capped_for_validation():
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

    def quote(ticker, strike):
        return {
            "ticker": ticker,
            "direction": "bullish",
            "option_symbol": f"{ticker}260618C{int(strike * 1000):08d}",
            "option_type": "call",
            "expiry": "2026-06-18",
            "strike": strike,
            "dte": 15,
            "bid": 4.0,
            "ask": 4.2,
            "mid": 4.1,
            "volume": 1000,
            "open_interest": 2000,
            "premium": 420_000.0,
            "spread_pct": 0.049,
            "quote_source": "bot_eod",
        }

    features = {
        "LOUD": {
            "date": "2026-05-28",
            "ticker": "LOUD",
            "close": 100.0,
            "source_flags": {"stock_screener"},
            "call_volume_ratio_30d": 80.0,
            "put_volume_ratio_30d": 0.2,
            "premium_bias": 0.50,
            "flow_premium_bias": 0.0,
            "flow_call_ask_ratio": 0.90,
            "hot_total_premium": 1_000_000.0,
            "liquidity_score": 20.0,
        },
    }
    best_options = {("LOUD", "bullish"): quote("LOUD", 120.0)}
    for idx, ticker in enumerate(("SRC1", "SRC2", "SRC3"), 1):
        features[ticker] = {
            "date": "2026-05-28",
            "ticker": ticker,
            "close": 100.0 + idx,
            "source_flags": {"bot_eod", "hot_chains"},
            "flow_total_premium": 60_000_000.0 + idx,
            "hot_total_premium": 20_000_000.0,
            "call_premium": 35_000_000.0,
            "put_premium": 25_000_000.0,
            "flow_call_premium_share": 0.56,
            "flow_put_premium_share": 0.44,
            "flow_call_ask_premium_share": 0.54,
            "flow_put_ask_premium_share": 0.46,
            "flow_premium_bias": 0.06,
            "call_volume_ratio_30d": 0.8,
            "put_volume_ratio_30d": 0.7,
            "liquidity_score": 18.0,
        }
        best_options[(ticker, "bullish")] = quote(ticker, 120.0 + idx)

    snapshot = SnapshotStub(features, best_options, signal_date="2026-05-28")

    capped = generate_signals_for_snapshot(
        snapshot,
        pattern_config,
        max_signals=1,
        source_rescue_max_extra=1,
        tradeable_gap_max_extra=0,
    )
    uncapped_daily_default = generate_signals_for_snapshot(
        snapshot,
        pattern_config,
        max_signals=1,
        tradeable_gap_max_extra=0,
    )

    assert sum(s["base_pattern_family"] == "SOURCE_PREMIUM_COVERAGE_RESCUE" for s in capped) == 1
    assert sum(s["base_pattern_family"] == "SOURCE_PREMIUM_COVERAGE_RESCUE" for s in uncapped_daily_default) == 3


def test_tradeable_source_gap_rescue_survives_rank_cap():
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
    features = {
        "LOUD": {
            "date": "2026-05-28",
            "ticker": "LOUD",
            "close": 100.0,
            "source_flags": {"stock_screener"},
            "call_volume_ratio_30d": 80.0,
            "put_volume_ratio_30d": 0.2,
            "premium_bias": 0.50,
            "flow_call_ask_ratio": 0.90,
            "hot_total_premium": 100_000_000.0,
            "liquidity_score": 20.0,
        },
            "FUBO": {
                "date": "2026-03-23",
                "ticker": "FUBO",
                "close": 2.20,
                "source_flags": {"hot_chains", "stock_screener"},
                "hot_total_premium": 30_000.0,
                "call_premium": 30_000.0,
                "put_premium": 0.0,
                "call_volume_ratio_30d": 1.30,
                "put_volume_ratio_30d": 0.90,
                "premium_bias": 0.205,
                "hot_call_ask_ratio": 0.20,
                "liquidity_score": 9.0,
        },
    }
    best_options = {
        ("LOUD", "bullish"): {
            "ticker": "LOUD",
            "direction": "bullish",
            "option_symbol": "LOUD260618C00120000",
            "option_type": "call",
            "expiry": "2026-06-18",
            "strike": 120.0,
            "dte": 15,
            "bid": 4.0,
            "ask": 4.2,
            "mid": 4.1,
            "volume": 1000,
            "open_interest": 2000,
            "premium": 420_000.0,
            "spread_pct": 0.049,
            "quote_source": "bot_eod",
        },
        ("FUBO", "bullish"): {
            "ticker": "FUBO",
            "direction": "bullish",
            "option_symbol": "FUBO260515C00001000",
            "option_type": "call",
            "expiry": "2026-05-15",
            "strike": 1.0,
            "dte": 45,
            "bid": 0.20,
            "ask": 0.22,
            "mid": 0.21,
            "volume": 1500,
            "open_interest": 2500,
            "premium": 330_000.0,
            "spread_pct": 0.095,
            "quote_source": "bot_eod",
        },
    }
    snapshot = SnapshotStub(features, best_options, signal_date="2026-03-23")

    signals = generate_signals_for_snapshot(snapshot, pattern_config, max_signals=1)

    assert signals[0]["ticker"] == "LOUD"
    assert any(
        s["ticker"] == "FUBO" and s["base_pattern_family"] == "TRADEABLE_SOURCE_GAP_RESCUE"
        for s in signals
    )


def test_tradeable_source_gap_rescue_falls_back_to_eligible_long_option():
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
    features = {
        "CRWG": {
            "date": "2026-05-04",
            "ticker": "CRWG",
            "close": 5.81,
            "source_flags": {"hot_chains", "stock_screener"},
            "hot_total_premium": 362_473.0,
            "call_premium": 412_000.0,
            "put_premium": 171_186.0,
            "call_volume_ratio_30d": 1.54,
            "put_volume_ratio_30d": 2.17,
            "premium_bias": 0.079,
            "flow_call_premium_share": 0.707,
            "flow_put_premium_share": 0.293,
            "stock_return_1d": 0.1088,
            "liquidity_score": 12.0,
        },
    }
    best_options = {
        ("CRWG", "bullish"): {
            "ticker": "CRWG",
            "direction": "bullish",
            "strategy_kind": "credit_spread",
            "strategy_type": "Bull Put Credit Spread",
            "expiry": "2026-05-15",
            "dte": 11,
            "entry_credit": 0.40,
            "max_risk": 60.0,
            "spread_pct": 0.40,
            "volume": 255,
            "open_interest": 1383,
        },
    }
    long_call = {
        "ticker": "CRWG",
        "direction": "bullish",
        "option_symbol": "CRWG260515C00005000",
        "option_type": "call",
        "expiry": "2026-05-15",
        "strike": 5.0,
        "dte": 11,
        "bid": 0.85,
        "ask": 0.95,
        "mid": 0.90,
        "volume": 800,
        "open_interest": 1200,
        "premium": 760_000.0,
        "spread_pct": 0.111,
        "quote_source": "bot_eod",
    }
    snapshot = SnapshotStub(
        features,
        best_options,
        signal_date="2026-05-04",
        option_quotes={"CRWG260515C00005000": long_call},
    )

    signals = generate_signals_for_snapshot(
        snapshot,
        pattern_config,
        max_signals=1,
        source_rescue_max_extra=0,
        tradeable_gap_max_extra=1,
    )

    gap_signal = next(s for s in signals if s["base_pattern_family"] == "TRADEABLE_SOURCE_GAP_RESCUE")
    assert gap_signal["ticker"] == "CRWG"
    assert gap_signal["lead_option_symbol"] == "CRWG260515C00005000"
    assert gap_signal["strategy_kind"] == "long_option"
    assert not gap_signal["block_reasons"]


def test_generate_signals_keeps_credit_spread_lane_next_to_long_option():
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
    feature = {
        "date": "2026-05-28",
        "ticker": "LANE",
        "close": 100.0,
        "source_flags": {"hot_chains", "stock_screener"},
        "hot_total_premium": 1_200_000.0,
        "call_premium": 900_000.0,
        "put_premium": 300_000.0,
        "call_volume_ratio_30d": 4.0,
        "put_volume_ratio_30d": 1.1,
        "premium_bias": 0.50,
        "hot_call_ask_ratio": 0.90,
        "liquidity_score": 20.0,
    }
    long_call = {
        "ticker": "LANE",
        "direction": "bullish",
        "option_symbol": "LANE260618C00110000",
        "option_type": "call",
        "expiry": "2026-06-18",
        "strike": 110.0,
        "dte": 15,
        "bid": 3.8,
        "ask": 4.0,
        "mid": 3.9,
        "volume": 2000,
        "open_interest": 3000,
        "premium": 800_000.0,
        "spread_pct": 0.051,
        "stock_close": 100.0,
        "quote_source": "bot_eod",
    }
    short_put = {
        "ticker": "LANE",
        "direction": "bearish",
        "option_symbol": "LANE260618P00095000",
        "option_type": "put",
        "expiry": "2026-06-18",
        "strike": 95.0,
        "dte": 15,
        "bid": 2.0,
        "ask": 2.2,
        "mid": 2.1,
        "volume": 1200,
        "open_interest": 1800,
        "premium": 240_000.0,
        "spread_pct": 0.095,
        "stock_close": 100.0,
    }
    long_put = {
        "ticker": "LANE",
        "direction": "bearish",
        "option_symbol": "LANE260618P00090000",
        "option_type": "put",
        "expiry": "2026-06-18",
        "strike": 90.0,
        "dte": 15,
        "bid": 0.8,
        "ask": 0.9,
        "mid": 0.85,
        "volume": 900,
        "open_interest": 1600,
        "premium": 81_000.0,
        "spread_pct": 0.118,
        "stock_close": 100.0,
    }
    snapshot = SnapshotStub(
        {"LANE": feature},
        best_options={("LANE", "bullish"): long_call},
        signal_date="2026-05-28",
        option_quotes={
            long_call["option_symbol"]: long_call,
            short_put["option_symbol"]: short_put,
            long_put["option_symbol"]: long_put,
        },
    )

    signals = generate_signals_for_snapshot(snapshot, pattern_config, max_signals=1)
    lane_signals = [row for row in signals if row["ticker"] == "LANE"]

    assert {row["strategy_kind"] for row in lane_signals} == {"long_option", "credit_spread"}
    spread = next(row for row in lane_signals if row["strategy_kind"] == "credit_spread")
    assert spread["strategy_type"] == "Bull Put Credit Spread"
    assert spread["option_type"] == "put"
    assert "SELL" in spread["legs_json"]


def test_directional_edge_diagnostics_explain_bearish_no_edge_lanes():
    auto = {
        "ticker": "QQQ",
        "status": "AUTO_APPROVED",
        "direction": "bullish",
        "strategy_type": "Bull Put Credit Spread",
        "strategy_kind": "credit_spread",
        "legs_json": '[{"action":"SELL","option_symbol":"QQQ260618P00720000","option_type":"put","strike":720},{"action":"BUY","option_symbol":"QQQ260618P00705000","option_type":"put","strike":705}]',
        "expiry": "2026-06-18",
        "entry_credit": 3.17,
        "expected_R": 0.13,
        "expected_R_per_day": 0.026,
        "probability_score": 61.0,
        "success_probability_pct": 67.0,
        "validation_profit_factor": 13.6,
        "validation_scored_count": 58,
        "beats_baselines_count": 6,
        "block_reasons": [],
    }
    review = {
        "ticker": "CAR",
        "status": "TRADE_REVIEW",
        "direction": "bearish",
        "strategy_type": "Long Put Debit",
        "strategy_kind": "long_option",
        "lead_option_symbol": "CAR260618P00160000",
        "option_type": "put",
        "strike": 160,
        "expiry": "2026-06-18",
        "entry_range": "4.90-6.70",
        "expected_R": 2.19,
        "expected_R_per_day": 0.43,
        "probability_score": 40.0,
        "success_probability_pct": 55.0,
        "validation_profit_factor": 999.0,
        "validation_scored_count": 1,
        "beats_baselines_count": 6,
        "block_reasons": ["LIMITED_OUT_OF_SAMPLE_SAMPLE", "PATTERN_VALIDATION_NOT_PROVEN"],
    }
    avoid = {
        "ticker": "IWM",
        "status": "AVOID",
        "direction": "bearish",
        "strategy_type": "Long Put Debit",
        "strategy_kind": "long_option",
        "lead_option_symbol": "IWM260618P00276000",
        "option_type": "put",
        "strike": 276,
        "expiry": "2026-06-18",
        "entry_range": "1.43-1.45",
        "expected_R": -0.42,
        "expected_R_per_day": -0.08,
        "probability_score": 16.0,
        "success_probability_pct": 38.0,
        "validation_profit_factor": 0.08,
        "validation_scored_count": 33,
        "beats_baselines_count": 0,
        "block_reasons": ["EXPECTED_R_NOT_POSITIVE_AFTER_COSTS", "DOES_NOT_BEAT_TWO_BASELINES"],
    }

    rows = build_directional_edge_diagnostic_rows([auto], [review], [avoid])
    by_lane = {(row["surface_status"], row["direction"], row["strategy"]) for row in rows}

    assert ("AUTO_APPROVED", "bullish", "Bull Put Credit Spread") in by_lane
    assert ("TRADE_REVIEW", "bearish", "Long Put Debit") in by_lane
    assert ("AVOID", "bearish", "Long Put Debit") in by_lane
    bearish_review = next(row for row in rows if row["surface_status"] == "TRADE_REVIEW")
    bearish_avoid = next(row for row in rows if row["surface_status"] == "AVOID")
    assert bearish_review["primary_diagnosis"] == "INSUFFICIENT_VALIDATED_SAMPLE"
    assert "CAR" in bearish_review["top_examples"]
    assert bearish_avoid["primary_diagnosis"] == "NEGATIVE_AVG_EXPECTANCY_AFTER_COSTS"
    assert bearish_avoid["avg_expected_R"] == -0.42


def test_goal_evidence_fails_when_required_high_signal_ticker_disappears():
    snapshot = SnapshotStub(
        features={
            "IBM": {
                "ticker": "IBM",
                "flow_total_premium": 94_315_371.0,
                "hot_total_premium": 57_753_450.0,
                "call_premium": 79_222_529.0,
                "put_premium": 15_092_842.0,
                "flow_call_premium_share": 0.84,
                "flow_put_premium_share": 0.16,
            }
        },
        signal_date="2026-05-28",
    )
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-28",
        snapshot,
        [],
        [],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["IBM"]}},
        {"source_complete": True, "missing_sources": []},
    )

    required_row = next(row for row in rows if row["requirement"] == "known_failure_ticker_surface_audit")
    assert required_row["status"] == "FAIL"
    assert "missing_high_signal=IBM" in required_row["evidence"]
    assert goal_evidence_overall_status(rows) == "FAIL_REQUIREMENTS_REMAIN"


def test_goal_evidence_passes_when_required_ticker_has_below_threshold_coverage():
    snapshot = SnapshotStub(
        features={
            "IBM": {
                "ticker": "IBM",
                "flow_total_premium": 11_710_000.0,
                "hot_total_premium": 8_000_000.0,
            }
        },
        signal_date="2026-05-15",
    )
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }
    coverage_rows = [
        {
            "ticker": "IBM",
            "decision_surface_status": "NOT_SURFACED",
            "source_gap_reason": "required coverage ticker below 50000000 high-source threshold",
        }
    ]

    rows = build_goal_evidence_rows(
        "2026-05-15",
        snapshot,
        [],
        coverage_rows,
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["IBM"]}},
        {"source_complete": True, "missing_sources": []},
    )

    required_row = next(row for row in rows if row["requirement"] == "known_failure_ticker_surface_audit")
    assert required_row["status"] == "PASS"
    assert "covered=IBM" in required_row["evidence"]


def test_goal_evidence_checks_auto_approved_profitability_gates():
    decision_board = [
        {
            "status": "AUTO_APPROVED",
            "ticker": "AMD",
            "full_ticket": "BUY CALL AMD 600 exp 2026-06-18",
            "buy_sell": "BUY",
            "call_put": "CALL",
            "strikes": "600",
            "expiration": "2026-06-18",
            "entry": "debit 11.60-11.65",
            "max_risk": 1165.0,
            "expected_R": 1.12586,
            "expected_R_per_day": 0.225172,
            "probability_score": 54.22,
            "calibrated_probability": 0.619,
            "validation_profit_factor": 1.51,
            "validation_scored_count": 50,
            "beats_baselines_count": 2,
            "baselines_beaten_names": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY",
            "baselines_beaten_details": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY:baseline_avg_R=-0.05,edge_R=0.30,scored=20;BASELINE_NAIVE_UW_FLOW_ONLY:baseline_avg_R=0.01,edge_R=0.24,scored=20",
            "blockers": "",
        }
    ]
    source_coverage_rows = [
        {
            "ticker": "AMD",
            "decision_surface_status": "AUTO_APPROVED",
            "source_gap_reason": "surfaced in decision board",
            "decision_artifact": "actionable_trades.csv",
        }
    ]
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-28",
        SnapshotStub({"AMD": {"ticker": "AMD", "flow_total_premium": 123_000_000.0}}),
        decision_board,
        source_coverage_rows,
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["AMD"]}},
        {"source_complete": True, "missing_sources": []},
    )

    expectancy_row = next(row for row in rows if row["requirement"] == "auto_approved_positive_expectancy_after_costs")
    ticket_row = next(row for row in rows if row["requirement"] == "trade_ready_ticket_fields_present")
    assert expectancy_row["status"] == "PASS"
    assert ticket_row["status"] == "PASS"


def test_goal_evidence_uses_ticker_trend_gates_for_ticker_trend_auto_approval():
    decision_board = [
        {
            "status": "AUTO_APPROVED",
            "ticker": "ASTS",
            "full_ticket": "BUY CALL ASTS 40 exp 2026-06-18",
            "buy_sell": "BUY",
            "call_put": "CALL",
            "strikes": "40",
            "expiration": "2026-06-18",
            "entry": "debit 3.00-3.10",
            "max_risk": 310.0,
            "expected_R": 0.681731,
            "expected_R_per_day": 0.136346,
            "probability_score": 46.13,
            "calibrated_probability": 0.565217,
            "validation_profit_factor": 5.051584,
            "validation_scored_count": 23,
            "beats_baselines_count": 6,
            "baselines_beaten_names": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY",
            "baselines_beaten_details": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY:baseline_avg_R=-0.05,edge_R=0.73,scored=20;BASELINE_NAIVE_UW_FLOW_ONLY:baseline_avg_R=0.01,edge_R=0.67,scored=20",
            "ticker_trend_scope": "ticker_direction_strategy",
            "ticker_trend_scored_count": 23,
            "ticker_trend_win_rate_pct": 56.52,
            "ticker_trend_probability_score_pct": 46.13,
            "ticker_trend_avg_R": 0.681731,
            "ticker_trend_profit_factor": 5.051584,
            "blockers": "",
        }
    ]
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-28",
        SnapshotStub({"ASTS": {"ticker": "ASTS", "flow_total_premium": 217_000_000.0}}),
        decision_board,
        [
            {
                "ticker": "ASTS",
                "decision_surface_status": "AUTO_APPROVED",
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "actionable_trades.csv",
            }
        ],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["ASTS"]}},
        {"source_complete": True, "missing_sources": []},
    )

    expectancy_row = next(row for row in rows if row["requirement"] == "auto_approved_positive_expectancy_after_costs")
    assert expectancy_row["status"] == "PASS"


def test_goal_evidence_fails_auto_approval_when_pattern_is_not_proven():
    decision_board = [
        {
            "status": "AUTO_APPROVED",
            "ticker": "ASTS",
            "full_ticket": "BUY CALL ASTS 40 exp 2026-06-18",
            "buy_sell": "BUY",
            "call_put": "CALL",
            "strikes": "40",
            "expiration": "2026-06-18",
            "entry": "debit 3.00-3.10",
            "max_risk": 310.0,
            "expected_R": 0.681731,
            "expected_R_per_day": 0.136346,
            "probability_score": 56.13,
            "calibrated_probability": 0.565217,
            "validation_profit_factor": 5.051584,
            "validation_scored_count": 40,
            "beats_baselines_count": 6,
            "baselines_beaten_names": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY",
            "baselines_beaten_details": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY:baseline_avg_R=-0.05,edge_R=0.73,scored=20",
            "confidence_tier": "PROMISING",
            "blockers": "PATTERN_VALIDATION_NOT_PROVEN",
        }
    ]
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-28",
        SnapshotStub({"ASTS": {"ticker": "ASTS", "flow_total_premium": 217_000_000.0}}),
        decision_board,
        [
            {
                "ticker": "ASTS",
                "decision_surface_status": "AUTO_APPROVED",
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "actionable_trades.csv",
            }
        ],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["ASTS"]}},
        {"source_complete": True, "missing_sources": []},
    )

    expectancy_row = next(row for row in rows if row["requirement"] == "auto_approved_positive_expectancy_after_costs")
    assert expectancy_row["status"] == "FAIL"
    assert "pattern_not_proven" in expectancy_row["evidence"]


def test_goal_evidence_fails_auto_approval_without_baseline_names():
    decision_board = [
        {
            "status": "AUTO_APPROVED",
            "ticker": "AMD",
            "full_ticket": "BUY CALL AMD 600 exp 2026-06-18",
            "buy_sell": "BUY",
            "call_put": "CALL",
            "strikes": "600",
            "expiration": "2026-06-18",
            "entry": "debit 11.60-11.65",
            "max_risk": 1165.0,
            "expected_R": 1.12586,
            "expected_R_per_day": 0.225172,
            "probability_score": 54.22,
            "calibrated_probability": 0.619,
            "validation_profit_factor": 1.51,
            "validation_scored_count": 50,
            "beats_baselines_count": 2,
            "blockers": "",
        }
    ]
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-28",
        SnapshotStub({"AMD": {"ticker": "AMD", "flow_total_premium": 123_000_000.0}}),
        decision_board,
        [
            {
                "ticker": "AMD",
                "decision_surface_status": "AUTO_APPROVED",
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "actionable_trades.csv",
            }
        ],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["AMD"]}},
        {"source_complete": True, "missing_sources": []},
    )

    expectancy_row = next(row for row in rows if row["requirement"] == "auto_approved_positive_expectancy_after_costs")
    assert expectancy_row["status"] == "FAIL"
    assert "AMD:baseline_names" in expectancy_row["evidence"]


def test_goal_evidence_accepts_insufficient_history_as_quantified_no_edge():
    decision_board = [
        {
            "status": "AVOID",
            "ticker": "AMD",
            "blockers": "LIMITED_OUT_OF_SAMPLE_SAMPLE;PATTERN_VALIDATION_NOT_PROVEN;EXPECTED_R_NOT_POSITIVE_AFTER_COSTS",
        }
    ]
    validation_bundle = empty_validation_bundle()
    validation_bundle.update(
        {
            "validation_history_status": "INSUFFICIENT_SOURCE_COMPLETE_HISTORY",
            "validation_history_reason": "no chronological train/validation split met min_month_dates=5",
            "source_date_count": 8,
            "source_month_date_counts": {"2025-12": 5, "2026-01": 3},
            "min_month_dates": 5,
        }
    )

    rows = build_goal_evidence_rows(
        "2026-01-05",
        SnapshotStub({"AMD": {"ticker": "AMD", "flow_total_premium": 123_000_000.0}}),
        decision_board,
        [
            {
                "ticker": "AMD",
                "decision_surface_status": "AVOID",
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "blocked_candidates.csv",
            }
        ],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["AMD"]}},
        {"source_complete": True, "missing_sources": []},
    )

    validation_row = next(row for row in rows if row["requirement"] == "rolling_oos_backtest_and_baselines_present")
    ticket_row = next(row for row in rows if row["requirement"] == "trade_ready_ticket_fields_present")
    no_edge_row = next(row for row in rows if row["requirement"] == "quantified_no_edge_report_if_no_trade")
    assert validation_row["status"] == "PASS"
    assert "insufficient_source_complete_history_no_edge" in validation_row["evidence"]
    assert ticket_row["status"] == "PASS"
    assert "no review/action ticket required" in ticket_row["evidence"]
    assert no_edge_row["status"] == "PASS"
    assert "LIMITED_OUT_OF_SAMPLE_SAMPLE" in no_edge_row["evidence"]


def test_goal_evidence_still_fails_missing_validation_when_history_was_not_run():
    decision_board = [
        {
            "status": "AVOID",
            "ticker": "AMD",
            "blockers": "PATTERN_VALIDATION_NOT_PROVEN",
        }
    ]

    rows = build_goal_evidence_rows(
        "2026-05-18",
        SnapshotStub({"AMD": {"ticker": "AMD", "flow_total_premium": 123_000_000.0}}),
        decision_board,
        [
            {
                "ticker": "AMD",
                "decision_surface_status": "AVOID",
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "blocked_candidates.csv",
            }
        ],
        [],
        empty_validation_bundle(),
        {"risk_config": {"goal_required_coverage_tickers": ["AMD"]}},
        {"source_complete": True, "missing_sources": []},
    )

    validation_row = next(row for row in rows if row["requirement"] == "rolling_oos_backtest_and_baselines_present")
    ticket_row = next(row for row in rows if row["requirement"] == "trade_ready_ticket_fields_present")
    assert validation_row["status"] == "FAIL"
    assert "no validation splits" in validation_row["evidence"]
    assert ticket_row["status"] == "PASS"


def test_goal_evidence_zero_review_rows_passes_when_blockers_quantify_no_edge():
    decision_board = [
        {
            "status": "AVOID",
            "ticker": "AMD",
            "blockers": "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS;PATTERN_VALIDATION_NOT_PROVEN",
        }
    ]
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }

    rows = build_goal_evidence_rows(
        "2026-04-30",
        SnapshotStub({"AMD": {"ticker": "AMD", "flow_total_premium": 123_000_000.0}}),
        decision_board,
        [
            {
                "ticker": "AMD",
                "decision_surface_status": "AVOID",
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "blocked_candidates.csv",
            }
        ],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["AMD"]}},
        {"source_complete": True, "missing_sources": []},
    )

    ticket_row = next(row for row in rows if row["requirement"] == "trade_ready_ticket_fields_present")
    assert ticket_row["status"] == "PASS"
    assert "reviewable_rows=0" in ticket_row["evidence"]


def test_goal_evidence_accepts_quantified_no_edge_without_forced_trade():
    decision_board = [
        {
            "status": "TRADE_REVIEW",
            "ticker": "AMD",
            "full_ticket": "BUY CALL AMD 500 exp 2026-06-18",
            "buy_sell": "BUY",
            "call_put": "CALL",
            "strikes": "500",
            "expiration": "2026-06-18",
            "entry": "debit 7.65-7.95",
            "max_risk": 795.0,
            "expected_R": -0.10,
            "probability_score": 40.0,
            "blockers": "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS;PATTERN_VALIDATION_NOT_PROVEN",
        }
    ]
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-18",
        SnapshotStub({"AMD": {"ticker": "AMD", "flow_total_premium": 123_000_000.0}}),
        decision_board,
        [
            {
                "ticker": "AMD",
                "decision_surface_status": "TRADE_REVIEW",
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "trade_review_candidates.csv",
            }
        ],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["AMD"]}},
        {"source_complete": True, "missing_sources": []},
    )

    auto_row = next(row for row in rows if row["requirement"] == "auto_approved_positive_expectancy_after_costs")
    no_edge_row = next(row for row in rows if row["requirement"] == "quantified_no_edge_report_if_no_trade")
    assert auto_row["status"] == "PASS"
    assert "no auto-approved rows were emitted" in auto_row["evidence"]
    assert no_edge_row["status"] == "PASS"
    assert "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS" in no_edge_row["evidence"]


def test_goal_evidence_warns_when_no_edge_is_not_quantified():
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-18",
        SnapshotStub({"AMD": {"ticker": "AMD", "flow_total_premium": 123_000_000.0}}),
        [],
        [
            {
                "ticker": "AMD",
                "decision_surface_status": "NOT_SURFACED",
                "source_gap_reason": "explained",
                "decision_artifact": "source_ticker_coverage.csv",
            }
        ],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": ["AMD"]}},
        {"source_complete": True, "missing_sources": []},
    )

    no_edge_row = next(row for row in rows if row["requirement"] == "quantified_no_edge_report_if_no_trade")
    ticket_row = next(row for row in rows if row["requirement"] == "trade_ready_ticket_fields_present")
    assert no_edge_row["status"] == "WARN"
    assert ticket_row["status"] == "WARN"
    assert "blocker counts were not available" in no_edge_row["evidence"]


def test_directional_scenario_goal_passes_when_put_side_is_reviewed_or_covered():
    decision_board = [
        {
            "status": "AUTO_APPROVED",
            "direction": "bullish",
            "call_put": "CALL",
            "full_ticket": "BUY CALL AMD 600 exp 2026-06-18",
        },
        {
            "status": "TRADE_REVIEW",
            "direction": "bearish",
            "call_put": "PUT",
            "full_ticket": "BUY PUT GLD 405 exp 2026-07-17",
        },
        {
            "status": "AVOID",
            "direction": "bearish",
            "call_put": "PUT",
            "full_ticket": "BUY PUT ZS 120 exp 2026-06-18",
        },
    ]
    source_coverage_rows = [
        {
            "ticker": "GLD",
            "direction": "bearish",
            "trade_legs": "Buy 1 GLD 2026-07-17 405P @ debit 8.80-9.15 limit",
        }
    ]

    row = build_directional_scenario_goal_row("2026-05-28", decision_board, source_coverage_rows)

    assert row["status"] == "PASS"
    assert "approved_put=0" in row["evidence"]
    assert "review_put=1" in row["evidence"]
    assert "avoid_put=1" in row["evidence"]
    assert "coverage_bearish=1" in row["evidence"]
    assert "coverage_put=1" in row["evidence"]


def test_directional_scenario_goal_warns_when_auto_day_has_no_put_or_bearish_surface():
    decision_board = [
        {
            "status": "AUTO_APPROVED",
            "direction": "bullish",
            "call_put": "CALL",
            "full_ticket": "BUY CALL AMD 600 exp 2026-06-18",
        }
    ]

    row = build_directional_scenario_goal_row("2026-05-28", decision_board, [])

    assert row["status"] == "WARN"
    assert "approved_call=1" in row["evidence"]
    assert "review_put=0" in row["evidence"]
    assert "coverage_bearish=0" in row["evidence"]


def test_goal_evidence_reports_macro_geo_point_in_time_observability():
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }
    macro_geo_bundle = {
        "summary": {
            "source_complete": True,
            "eligible_catalyst_count": 2,
            "eligible_event_types": ["AI chips/semiconductors", "China/US diplomacy"],
            "future_dated_catalyst_count": 1,
            "future_dated_event_types": ["tariffs"],
            "uw_confirmed_catalyst_count": 1,
            "uw_confirmed_themes": "semiconductors/AI chips",
            "scenario_bucket_counts": {
                "CATALYST_WATCH": 1,
                "POINT_IN_TIME_INELIGIBLE_CATALYST": 1,
                "SECTOR_INDEX_CONFIRMED_SETUP": 1,
            },
        },
        "promotion_decisions": [{"ticker": "NVDA"}, {"ticker": "SMH"}],
        "ticker_map": [{"ticker": "NVDA"}, {"ticker": "SMH"}],
        "uw_confirmation": [{"ticker": "NVDA"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-13",
        SnapshotStub({}),
        [],
        [],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": []}},
        {"source_complete": True, "missing_sources": []},
        macro_geo_bundle=macro_geo_bundle,
    )

    row = next(row for row in rows if row["requirement"] == "macro_geo_point_in_time_observability")
    assert row["status"] == "PASS"
    assert "eligible_catalysts=2" in row["evidence"]
    assert "future_dated_filtered=1" in row["evidence"]
    assert "promotion_rows=2" in row["evidence"]
    assert "ticker_map_rows=2" in row["evidence"]
    assert "uw_confirmation_rows=1" in row["evidence"]
    assert "AI chips/semiconductors" in row["evidence"]
    assert "POINT_IN_TIME_INELIGIBLE_CATALYST:1" in row["evidence"]


def test_goal_evidence_warns_when_macro_geo_artifacts_are_missing():
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }
    macro_geo_bundle = {
        "summary": {
            "source_complete": True,
            "eligible_catalyst_count": 1,
            "eligible_event_types": ["China/US diplomacy"],
            "future_dated_catalyst_count": 0,
            "future_dated_event_types": [],
            "uw_confirmed_catalyst_count": 1,
            "uw_confirmed_themes": "China beta/trade",
            "scenario_bucket_counts": {"CATALYST_WATCH": 1},
        },
        "promotion_decisions": [],
        "ticker_map": [{"ticker": "FXI"}],
        "uw_confirmation": [{"ticker": "FXI"}],
    }

    rows = build_goal_evidence_rows(
        "2026-05-13",
        SnapshotStub({}),
        [],
        [],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": []}},
        {"source_complete": True, "missing_sources": []},
        macro_geo_bundle=macro_geo_bundle,
    )

    row = next(row for row in rows if row["requirement"] == "macro_geo_point_in_time_observability")
    assert row["status"] == "WARN"
    assert "eligible_catalysts=1" in row["evidence"]
    assert "promotion_rows=0" in row["evidence"]


def test_goal_evidence_fails_when_macro_geo_source_is_incomplete():
    validation_bundle = {
        "splits": [{"name": "cumulative_to_2026-05"}],
        "validation_gate_scorecard": [{"pattern_family": "x"}],
        "baseline_comparison": [{"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"}],
    }
    macro_geo_bundle = {
        "summary": {
            "source_complete": False,
            "missing_sources": ["bot_eod"],
            "eligible_catalyst_count": 0,
            "future_dated_catalyst_count": 0,
            "uw_confirmed_catalyst_count": 0,
            "scenario_bucket_counts": {},
        },
    }

    rows = build_goal_evidence_rows(
        "2026-05-13",
        SnapshotStub({}),
        [],
        [],
        [],
        validation_bundle,
        {"risk_config": {"goal_required_coverage_tickers": []}},
        {"source_complete": False, "missing_sources": ["bot_eod"]},
        macro_geo_bundle=macro_geo_bundle,
    )

    row = next(row for row in rows if row["requirement"] == "macro_geo_point_in_time_observability")
    assert row["status"] == "FAIL"
    assert "missing_sources=bot_eod" in row["evidence"]


def test_missed_mover_bucket_separates_untradeable_from_generation_gap():
    assert (
        missed_mover_bucket(
            {"ticker": "COOK", "hot_total_premium": 0.0, "call_volume_ratio_30d": 0.1, "put_volume_ratio_30d": 0.2},
            False,
            {},
            "no_hot_chain_premium;missing_quote_spread",
        )
        == "NOT_OPTION_TRADEABLE_MISSING_QUOTE"
    )
    assert (
        missed_mover_bucket(
            {
                "ticker": "FUBO",
                "hot_total_premium": 274_575.0,
                "call_volume_ratio_30d": 10.1,
                "put_volume_ratio_30d": 1.9,
                "premium_bias": 0.2,
            },
            False,
            {
                "option_symbol": "FUBO260618C00005000",
                "bid": 1.0,
                "ask": 1.1,
                "dte": 30,
                "volume": 1500,
                "open_interest": 2500,
                "spread_pct": 0.095,
            },
            "moved_without_matching_frozen_pattern",
        )
        == "CANDIDATE_GENERATION_GAP"
    )
    assert (
        missed_mover_bucket(
            {"ticker": "CRWG", "hot_total_premium": 362_473.0, "call_volume_ratio_30d": 1.54},
            False,
            {
                "option_symbol": "CRWG260515C00005000",
                "bid": 1.15,
                "ask": 1.60,
                "dte": 11,
                "volume": 800,
                "open_interest": 1200,
                "spread_pct": 0.2857,
            },
            "moved_without_matching_frozen_pattern",
            {
                "base_pattern_family": "TRADEABLE_SOURCE_GAP_RESCUE",
                "direction": "bullish",
                "lead_option_symbol": "CRWG260515C00005000",
                "block_reasons": ["MARKET_REGIME_CONFLICT"],
            },
            "bullish",
        )
        == "GENERATED_BUT_BLOCKED"
    )
    assert (
        missed_mover_bucket(
            {"ticker": "HIMZ", "hot_total_premium": 62_490.0, "call_volume_ratio_30d": 0.45},
            False,
            {
                "option_symbol": "HIMZ260417C00003000",
                "bid": 0.10,
                "ask": 0.15,
                "dte": 30,
                "volume": 1500,
                "open_interest": 2500,
                "spread_pct": 0.20,
            },
            "moved_without_matching_frozen_pattern",
            {
                "base_pattern_family": "TRADEABLE_SOURCE_GAP_RESCUE",
                "direction": "bearish",
                "lead_option_symbol": "HIMZ260417P00002000",
                "block_reasons": [],
            },
            "bullish",
        )
        == "DIRECTION_MISMATCH_NOT_LEAKAGE_SAFE"
    )
    assert (
        missed_mover_bucket(
            {
                "ticker": "FFAI",
                "hot_total_premium": 37_606.0,
                "call_volume_ratio_30d": 1.45,
                "premium_bias": -0.179,
            },
            False,
            {
                "option_symbol": "FFAI260529C00000500",
                "bid": 0.07,
                "ask": 0.08,
                "dte": 39,
                "volume": 1500,
                "open_interest": 2500,
                "spread_pct": 0.1333,
            },
            "moved_without_matching_frozen_pattern",
            {},
            "bullish",
            "bearish",
        )
        == "DIRECTION_MISMATCH_NOT_LEAKAGE_SAFE"
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


def test_historical_validation_marks_sparse_history_without_splits():
    dates = ["2025-12-22", "2025-12-23", "2025-12-24", "2025-12-29", "2025-12-30", "2026-01-05"]

    bundle = run_historical_validation({}, dates, min_month_dates=5, top_candidates_per_day=40, seed=7)

    assert bundle["splits"] == []
    assert bundle["validation_history_status"] == "INSUFFICIENT_SOURCE_COMPLETE_HISTORY"
    assert bundle["source_date_count"] == len(dates)
    assert bundle["source_month_date_counts"] == {"2025-12": 5, "2026-01": 1}


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
    assert row["cost_model"] == "long_option_entry_ask_exit_bid_after_configured_fees"
    assert "round_trip_fees" in validation_detail_fieldnames()
    assert "slippage_dollars" in validation_detail_fieldnames()


def test_long_option_scoring_uses_configured_fees():
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
                "option_quotes": {"XYZ260116C00100000": {"bid": 1.5, "ask": 1.6}},
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
        {"round_trip_long_option_fees": 10.0},
    )

    assert row["status"] == "SCORED"
    assert row["round_trip_fees"] == 10.0
    assert row["opening_fee"] == 5.0
    assert row["entry_slippage"] == pytest.approx(5.0)
    assert row["exit_slippage"] == pytest.approx(5.0)
    assert row["slippage_dollars"] == pytest.approx(10.0)
    assert row["cost_model"] == "long_option_entry_ask_exit_bid_after_configured_fees"
    assert row["slippage_model"] == "configured_extra_slippage_pct_of_entry_and_exit_spreads"
    assert row["net_r"] == pytest.approx((50.0 - 10.0 - 10.0) / 110.0)


def test_build_signal_uses_configured_opening_fee_for_ticket_risk():
    snapshot = SnapshotStub(
        features={},
        market_regime={"regime": "MIXED"},
        signal_date="2026-01-02",
    )
    feature = {
        "ticker": "XYZ",
        "close": 100.0,
        "sector": "Tech",
        "source_flags": set(),
    }
    quote = {
        "option_symbol": "XYZ260116C00100000",
        "strategy_kind": "long_option",
        "option_type": "call",
        "strike": 100.0,
        "expiry": "2026-01-16",
        "dte": 14,
        "bid": 0.95,
        "ask": 1.0,
        "mid": 0.975,
        "volume": 100,
        "open_interest": 100,
        "spread_pct": 0.05,
    }

    row = build_signal(
        snapshot,
        feature,
        "UNIT_PATTERN",
        "bullish",
        1.0,
        ["unit"],
        quote,
        {"max_spread_pct": 0.35, "max_event_dte_without_event_strategy": 2},
        {"round_trip_long_option_fees": 10.0},
    )

    assert row["entry_slippage"] == pytest.approx(2.5)
    assert row["max_risk_per_contract"] == pytest.approx(107.5)
    assert row["target_profit"] == pytest.approx(107.5)


def test_tradeable_gap_quote_eligible_uses_configured_max_risk():
    quote = {
        "strategy_kind": "long_option",
        "ask": 12.0,
        "bid": 11.95,
        "dte": 14,
        "spread_pct": 0.01,
        "volume": 100,
        "open_interest": 100,
    }

    assert not tradeable_gap_quote_eligible(quote, {"max_spread_pct": 0.35}, {"max_risk_per_trade": 1000.0})
    assert tradeable_gap_quote_eligible(quote, {"max_spread_pct": 0.35}, {"max_risk_per_trade": 1500.0})


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
    assert row["outcome_note"] == "managed_long_option_stop_hit_conservative_after_costs_slippage"


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
        "max_risk_per_contract": 411.30,
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
                    "XYZ260116P00095000": {"bid": 0.3, "ask": 0.4},
                    "XYZ260116P00090000": {"bid": 0.1, "ask": 0.2},
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
        {"round_trip_spread_fees": 10.0},
    )

    assert row["status"] == "SCORED"
    assert row["win"] == 1
    assert row["outcome_note"] == "credit_spread_exit_debit_after_costs_slippage"
    assert row["round_trip_fees"] == 10.0
    assert row["opening_fee"] == 5.0
    assert row["entry_slippage"] == pytest.approx(10.0)
    assert row["exit_slippage"] == pytest.approx(10.0)
    assert row["slippage_dollars"] == pytest.approx(20.0)
    assert row["cost_model"] == "credit_spread_entry_credit_exit_debit_after_configured_fees"
    assert row["net_r"] == pytest.approx((70.0 - 10.0 - 20.0) / 411.30)


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
            "status": "AUTO_APPROVED",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "lead_option_symbol": "SNDK260515C02000000",
            "option_type": "call",
            "strike": 2000.0,
            "expiry": "2026-05-15",
            "entry_range": "7.20-7.30",
            "expected_R": 0.42,
            "expected_R_per_day": 0.084,
            "validation_scored_count": 42,
            "validation_profit_factor": 1.7,
            "beats_baselines_count": 3,
            "baselines_beaten_names": "BASELINE_A;BASELINE_B;BASELINE_C",
            "baselines_beaten_details": "BASELINE_A:baseline_avg_R=-0.05,edge_R=0.47,scored=20;BASELINE_B:baseline_avg_R=0.01,edge_R=0.41,scored=20;BASELINE_C:baseline_avg_R=0.10,edge_R=0.32,scored=20",
            "probability_score": 55.0,
            "calibrated_probability": 0.61,
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
    assert row["validation_scored_count"] == 42
    assert row["beats_baselines_count"] == 3
    assert row["baselines_beaten_names"] == "BASELINE_A;BASELINE_B;BASELINE_C"
    assert row["baselines_beaten_details"].startswith("BASELINE_A:baseline_avg_R=-0.05")
    assert "expected_R=0.42" in row["auto_approval_gate_evidence"]
    assert "scored=42" in row["auto_approval_gate_evidence"]
    assert "baselines_beaten=3" in row["auto_approval_gate_evidence"]
    assert "baseline_names=BASELINE_A;BASELINE_B;BASELINE_C" in row["auto_approval_gate_evidence"]
    assert "baseline_edges=BASELINE_A:baseline_avg_R=-0.05" in row["auto_approval_gate_evidence"]


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
            {"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY", "average_net_r": -0.05, "scored_count": 20},
            {"baseline": "BASELINE_NAIVE_UW_FLOW_ONLY", "average_net_r": 0.01, "scored_count": 20},
            {"baseline": "BASELINE_TOO_STRONG", "average_net_r": 0.30, "scored_count": 20},
        ],
    )

    edge = tiers["EDGE"]
    assert edge["validation_success_probability"] == 0.65
    assert edge["validation_failure_probability"] == 0.35
    assert 0.58 < edge["validation_probability_score"] < edge["validation_success_probability"]
    assert "52/80" in edge["probability_evidence"]
    assert edge["beats_baselines_count"] == 2
    assert edge["baselines_beaten_names"] == "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY"
    assert "BASELINE_RANDOM_SAME_DATE_LIQUIDITY:baseline_avg_R=-0.05" in edge["baselines_beaten_details"]


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


def test_ticker_trend_overlay_can_promote_executable_trade():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__TECHNOLOGY"
    daily_rows = [
        {
            "ticker": "AMD",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "classification": "WATCH",
            "block_reasons": [
                "PATTERN_VALIDATION_NOT_PROVEN",
                "PROFIT_FACTOR_BELOW_AUTO_APPROVAL",
                "VALIDATION_EXPECTANCY_NEGATIVE",
                "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS",
                "EXPECTED_R_PER_DAY_NOT_POSITIVE",
                "DOES_NOT_BEAT_TWO_BASELINES",
            ],
            "confidence_tier": "PROMISING",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 600,
            "expiry": "2026-06-18",
            "lead_option_symbol": "AMD260618C00600000",
            "entry_bid": 10.8,
            "entry_ask": 11.25,
            "entry_range": "10.80-11.25",
            "bid_ask_spread_pct": 0.04,
            "max_risk_per_contract": 1125.65,
            "liquidity_volume": 2000,
            "liquidity_open_interest": 3000,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_ON",
            "market_regime": "RISK_ON",
        }
    ]
    outcomes = []
    for idx in range(34):
        win = idx % 3 != 0
        outcomes.append(
            {
                "split": "cumulative_to_2026-05_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                "signal_date": f"2026-05-{idx % 20 + 1:02d}",
                "ticker": "AMD",
                "direction": "bullish",
                "pattern_family": family,
                "market_regime": "RISK_ON",
                "strategy_kind": "long_option",
                "status": "SCORED",
                "net_r": 2.0 if win else -0.45,
                "win": int(win),
            }
        )
    validation_bundle = empty_validation_bundle()
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROMISING",
            "validation_scored_count": 34,
            "validation_win_count": 22,
            "validation_success_probability": 22 / 34,
            "validation_failure_probability": 12 / 34,
            "validation_probability_score": 0.45,
            "validation_average_net_r": 0.1,
            "validation_profit_factor": 1.19,
            "beats_baselines_count": 6,
            "baselines_beaten_names": "BASELINE_FAMILY_ONLY_SHOULD_NOT_SURVIVE_TICKER_TREND",
            "baselines_beaten_details": "BASELINE_FAMILY_ONLY_SHOULD_NOT_SURVIVE_TICKER_TREND:baseline_avg_R=-9,edge_R=9,scored=99",
        }
    }
    validation_bundle["outcomes"] = outcomes
    validation_bundle["baseline_comparison"] = [
        {"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY", "average_net_r": -0.05, "scored_count": 20},
        {"baseline": "BASELINE_NAIVE_UW_FLOW_ONLY", "average_net_r": 0.01, "scored_count": 20},
        {"baseline": "BASELINE_TOO_STRONG", "average_net_r": 1.70, "scored_count": 20},
    ]

    rows, controls = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    assert rows[0]["status"] == "AUTO_APPROVED"
    assert rows[0]["classification"] == "TRADE"
    assert rows[0]["ticker_trend_scope"] == "ticker_direction_strategy"
    assert rows[0]["success_probability_pct"] > 55
    assert rows[0]["expected_R"] > 0
    assert "PATTERN_VALIDATION_NOT_PROVEN" not in rows[0]["block_reasons"]
    assert "PROFIT_FACTOR_BELOW_AUTO_APPROVAL" not in rows[0]["block_reasons"]
    assert "VALIDATION_EXPECTANCY_NEGATIVE" not in rows[0]["block_reasons"]
    assert "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS" not in rows[0]["block_reasons"]
    assert "EXPECTED_R_PER_DAY_NOT_POSITIVE" not in rows[0]["block_reasons"]
    assert "DOES_NOT_BEAT_TWO_BASELINES" not in rows[0]["block_reasons"]
    assert "Not actionable" not in rows[0]["why_actionable_now"]
    assert "all auto-approval gates passed" in rows[0]["why_actionable_now"]
    assert "pattern not proven out-of-sample" not in rows[0]["major_risks"]
    assert "ticker-specific trend validation" in rows[0]["major_risks"]
    trade_row = trade_output_row(rows[0])
    assert "ticker-specific trend validation" in trade_row["historical_evidence_summary"]
    assert "Not actionable" not in trade_row["why_actionable_now"]
    assert trade_row["validation_scored_count"] == rows[0]["validation_scored_count"]
    assert trade_row["beats_baselines_count"] == rows[0]["beats_baselines_count"]
    assert trade_row["beats_baselines_count"] == 2
    assert trade_row["baselines_beaten_names"] == "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY"
    assert "BASELINE_TOO_STRONG" not in trade_row["baselines_beaten_names"]
    assert "BASELINE_FAMILY_ONLY" not in trade_row["baselines_beaten_names"]
    assert "baseline_avg_R=-0.05" in trade_row["baselines_beaten_details"]
    assert "edge_R=1.18" in trade_row["baselines_beaten_details"]
    assert "baselines_beaten=2" in trade_row["auto_approval_gate_evidence"]
    assert "baseline_names=BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY" in trade_row["auto_approval_gate_evidence"]
    assert "baseline_edges=BASELINE_RANDOM_SAME_DATE_LIQUIDITY:baseline_avg_R=-0.05" in trade_row["auto_approval_gate_evidence"]
    assert "ticker_trend_scope=ticker_direction_strategy" in trade_row["auto_approval_gate_evidence"]
    edge_rows = build_ticker_trend_edge_rows(
        controls["ticker_trend_stats"],
        controls["risk_config"],
        controls["baseline_comparison"],
    )
    amd_edge = next(row for row in edge_rows if row["ticker"] == "AMD")
    assert amd_edge["trade_ready_trend"] == "yes"
    assert amd_edge["strategy"] == "Long Call Debit"
    assert amd_edge["strategy_type"] == "Long Call Debit"
    assert amd_edge["call_or_put"] == "CALL"
    assert amd_edge["beats_baselines_count"] == 2
    assert amd_edge["baselines_beaten_names"] == "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY"
    assert controls["run_kill_switches"] == []


def test_proven_soft_calibration_bridge_promotes_complete_positive_ev_ticket():
    family = "SOURCE_PREMIUM_COVERAGE_RESCUE__BULLISH__LONG_OPTION__TECHNOLOGY"
    daily_rows = [
        {
            "ticker": "IBM",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "SOURCE_PREMIUM_COVERAGE_RESCUE",
            "classification": "WATCH",
            "block_reasons": [],
            "confidence_tier": "PROVEN",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 300,
            "expiry": "2026-07-17",
            "lead_option_symbol": "IBM260717C00300000",
            "entry_bid": 9.15,
            "entry_ask": 9.25,
            "entry_range": "9.15-9.25",
            "bid_ask_spread_pct": 0.011,
            "max_risk_per_contract": 930.65,
            "liquidity_volume": 5000,
            "liquidity_open_interest": 8000,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_ON",
            "market_regime": "RISK_ON",
            "success_probability_pct": 53.44,
            "probability_score": 48.51,
            "trade_success_probability_pct": 53.44,
            "trade_probability_score": 48.51,
        }
    ]
    validation_bundle = empty_validation_bundle()
    validation_bundle["splits"] = [{"name": "cumulative_to_2026-06_holdout"}]
    validation_bundle["validation_gate_scorecard"] = [{"pattern_family": family}]
    validation_bundle["baseline_comparison"] = [
        {"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY", "average_net_r": -0.31, "scored_count": 20},
        {"baseline": "BASELINE_NAIVE_UW_FLOW_ONLY", "average_net_r": -0.05, "scored_count": 20},
    ]
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROVEN",
            "validation_scored_count": 99,
            "validation_win_count": 45,
            "validation_success_probability": 45 / 99,
            "validation_failure_probability": 54 / 99,
            "validation_probability_score": 0.40520496556336705,
            "validation_average_net_r": 0.1703507792310215,
            "validation_profit_factor": 1.5409045613372128,
            "beats_baselines_count": 6,
            "baselines_beaten_names": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY",
            "baselines_beaten_details": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY:baseline_avg_R=-0.31,edge_R=0.48,scored=20",
        }
    }

    rows, controls = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    assert rows[0]["status"] == "AUTO_APPROVED"
    assert rows[0]["classification"] == "TRADE"
    assert rows[0]["approval_bridge"] == "PROVEN_SOFT_CALIBRATION"
    assert rows[0]["block_reasons"] == []
    trade_row = trade_output_row(rows[0])
    assert "approval_bridge" in trade_fieldnames()
    assert trade_row["approval_bridge"] == "PROVEN_SOFT_CALIBRATION"
    assert "approval_bridge=PROVEN_SOFT_CALIBRATION" in trade_row["auto_approval_gate_evidence"]

    decision_board = build_decision_board_rows(rows, "2026-07-06", True, "AUTO_APPROVED", {})
    goal_rows = build_goal_evidence_rows(
        "2026-07-06",
        SnapshotStub({"IBM": {"ticker": "IBM", "flow_total_premium": 250_000_000.0}}),
        decision_board,
        [
            {
                "ticker": "IBM",
                "decision_surface_status": "AUTO_APPROVED",
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "actionable_trades.csv",
            }
        ],
        [],
        validation_bundle,
        controls,
        {"source_complete": True, "missing_sources": []},
    )
    auto_gate = next(row for row in goal_rows if row["requirement"] == "auto_approved_positive_expectancy_after_costs")
    assert auto_gate["status"] == "PASS"


def test_trend_edge_strategy_fields_labels_bearish_structures():
    assert trend_edge_strategy_fields("bearish", "long_option") == {
        "strategy": "Long Put Debit",
        "strategy_type": "Long Put Debit",
        "call_or_put": "PUT",
    }
    assert trend_edge_strategy_fields("bearish", "credit_spread") == {
        "strategy": "Bear Call Credit Spread",
        "strategy_type": "Bear Call Credit Spread",
        "call_or_put": "CALL / CALL",
    }


def test_ticker_trend_no_edge_reason_names_failed_gates():
    reason = ticker_trend_no_edge_reason(
        {
            "scored_count": 7,
            "win_rate": 0.40,
            "probability_score": 0.35,
            "avg_R": -0.10,
            "profit_factor": 0.90,
            "drawdown_proxy_r": -9.0,
            "worst_losing_streak": 9,
            "edge_vs_breakeven_pct": -2.0,
            "beats_baselines_count": 1,
        },
        {
            "min_ticker_trend_scored_outcomes": 20,
            "min_ticker_trend_win_rate": 0.55,
            "min_ticker_trend_probability_score": 0.42,
            "min_ticker_trend_expected_r": 0.15,
            "min_ticker_trend_profit_factor": 1.5,
            "max_ticker_trend_drawdown_r": -8.0,
            "max_ticker_trend_losing_streak": 8,
            "min_ticker_trend_breakeven_edge_pct": 5.0,
            "min_baselines_beaten": 2,
        },
    )

    assert "LIMITED_SAMPLE 7/20" in reason
    assert "EXPECTED_R_BELOW_GATE" in reason
    assert "PROFIT_FACTOR_BELOW_GATE" in reason
    assert "BASELINES_BEATEN 1/2" in reason


def test_balanced_non_ready_trend_rows_keeps_bearish_put_and_spread_examples():
    rows = [
        {"ticker": f"BULL{i}", "direction": "bullish", "strategy_kind": "long_option", "trade_ready_trend": "no"}
        for i in range(6)
    ]
    rows.extend(
        [
            {"ticker": "IWM", "direction": "bearish", "strategy_kind": "long_option", "trade_ready_trend": "no"},
            {"ticker": "MSTR", "direction": "bearish", "strategy_kind": "credit_spread", "trade_ready_trend": "no"},
        ]
    )

    selected = balanced_non_ready_trend_rows(rows, limit=5)

    assert ("IWM", "bearish", "long_option") in {
        (row["ticker"], row["direction"], row["strategy_kind"]) for row in selected
    }
    assert ("MSTR", "bearish", "credit_spread") in {
        (row["ticker"], row["direction"], row["strategy_kind"]) for row in selected
    }


def test_ticker_trend_overlay_demotes_when_trend_does_not_beat_baselines():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__TECHNOLOGY"
    daily_rows = [
        {
            "ticker": "AMD",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "classification": "WATCH",
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
            "confidence_tier": "PROMISING",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 600,
            "expiry": "2026-06-18",
            "lead_option_symbol": "AMD260618C00600000",
            "entry_bid": 10.8,
            "entry_ask": 11.25,
            "entry_range": "10.80-11.25",
            "bid_ask_spread_pct": 0.04,
            "max_risk_per_contract": 1125.65,
            "liquidity_volume": 2000,
            "liquidity_open_interest": 3000,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_ON",
            "market_regime": "RISK_ON",
        }
    ]
    outcomes = []
    for idx in range(34):
        win = idx % 3 != 0
        outcomes.append(
            {
                "split": "cumulative_to_2026-05_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                "signal_date": f"2026-05-{idx % 20 + 1:02d}",
                "ticker": "AMD",
                "direction": "bullish",
                "pattern_family": family,
                "market_regime": "RISK_ON",
                "strategy_kind": "long_option",
                "status": "SCORED",
                "net_r": 2.0 if win else -0.45,
                "win": int(win),
            }
        )
    validation_bundle = empty_validation_bundle()
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROMISING",
            "validation_scored_count": 34,
            "validation_win_count": 22,
            "validation_success_probability": 22 / 34,
            "validation_probability_score": 0.45,
            "validation_average_net_r": 0.1,
            "validation_profit_factor": 1.3,
            "beats_baselines_count": 6,
            "baselines_beaten_names": "BASELINE_FAMILY_ONLY_SHOULD_NOT_PASS",
            "baselines_beaten_details": "BASELINE_FAMILY_ONLY_SHOULD_NOT_PASS:baseline_avg_R=-9,edge_R=9,scored=99",
        }
    }
    validation_bundle["outcomes"] = outcomes
    validation_bundle["baseline_comparison"] = [
        {"baseline": "BASELINE_TOO_STRONG_A", "average_net_r": 1.50, "scored_count": 20},
        {"baseline": "BASELINE_TOO_STRONG_B", "average_net_r": 1.60, "scored_count": 20},
    ]

    rows, controls = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    assert rows[0]["ticker_trend_scope"] == "ticker_direction_strategy"
    assert rows[0]["beats_baselines_count"] == 0
    assert rows[0]["baselines_beaten_names"] == ""
    assert "DOES_NOT_BEAT_TWO_BASELINES" in rows[0]["block_reasons"]
    assert rows[0]["status"] != "AUTO_APPROVED"
    edge_rows = build_ticker_trend_edge_rows(
        controls["ticker_trend_stats"],
        controls["risk_config"],
        controls["baseline_comparison"],
    )
    amd_edge = next(row for row in edge_rows if row["ticker"] == "AMD")
    assert amd_edge["trade_ready_trend"] == "no"
    assert amd_edge["beats_baselines_count"] == 0


def test_ticker_trend_overlay_demotes_auto_when_sample_is_below_auto_minimum():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__CONSUMER_CYCLICAL"
    daily_rows = [
        {
            "ticker": "TSLA",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "classification": "WATCH",
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
            "confidence_tier": "PROMISING",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 500,
            "expiry": "2026-06-18",
            "lead_option_symbol": "TSLA260618C00500000",
            "entry_bid": 4.9,
            "entry_ask": 5.0,
            "entry_range": "4.90-5.00",
            "bid_ask_spread_pct": 0.02,
            "max_risk_per_contract": 500.65,
            "liquidity_volume": 2000,
            "liquidity_open_interest": 3000,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_ON",
            "market_regime": "RISK_ON",
            "flow_total_premium": 1_500_000_000.0,
            "hot_total_premium": 1_500_000_000.0,
        }
    ]
    outcomes = []
    for idx in range(18):
        win = idx < 14
        outcomes.append(
            {
                "split": "cumulative_to_2026-05_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                "signal_date": f"2026-05-{idx % 20 + 1:02d}",
                "ticker": "TSLA",
                "direction": "bullish",
                "pattern_family": family,
                "market_regime": "RISK_ON",
                "strategy_kind": "long_option",
                "status": "SCORED",
                "net_r": 2.0 if win else -0.25,
                "win": int(win),
            }
        )
    validation_bundle = empty_validation_bundle()
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROMISING",
            "validation_scored_count": 18,
            "validation_win_count": 14,
            "validation_success_probability": 14 / 18,
            "validation_probability_score": 0.50,
            "validation_average_net_r": 1.5,
            "validation_profit_factor": 10.0,
            "beats_baselines_count": 6,
            "baselines_beaten_names": "BASELINE_FAMILY_ONLY_SHOULD_NOT_OVERRIDE_SAMPLE",
            "baselines_beaten_details": "BASELINE_FAMILY_ONLY_SHOULD_NOT_OVERRIDE_SAMPLE:baseline_avg_R=-9,edge_R=9,scored=99",
        }
    }
    validation_bundle["outcomes"] = outcomes
    validation_bundle["baseline_comparison"] = [
        {"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY", "average_net_r": -0.05, "scored_count": 20},
        {"baseline": "BASELINE_NAIVE_UW_FLOW_ONLY", "average_net_r": 0.01, "scored_count": 20},
    ]

    rows, controls = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    assert rows[0]["ticker_trend_scope"] == "ticker_direction_strategy"
    assert rows[0]["validation_scored_count"] == 18
    assert "LIMITED_OUT_OF_SAMPLE_SAMPLE" in rows[0]["block_reasons"]
    assert rows[0]["status"] == "TRADE_REVIEW"
    goal_rows = build_goal_evidence_rows(
        "2026-05-28",
        SnapshotStub({"TSLA": {"ticker": "TSLA", "flow_total_premium": 1_500_000_000.0}}),
        rows,
        [
            {
                "ticker": "TSLA",
                "decision_surface_status": rows[0]["status"],
                "source_gap_reason": "surfaced in decision board",
                "decision_artifact": "trade_review_candidates.csv",
            }
        ],
        [],
        validation_bundle,
        {"risk_config": controls["risk_config"]},
        {"source_complete": True, "missing_sources": []},
    )
    auto_row = next(row for row in goal_rows if row["requirement"] == "auto_approved_positive_expectancy_after_costs")
    assert auto_row["status"] == "PASS"
    assert "no auto-approved rows were emitted" in auto_row["evidence"]


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


def test_source_complete_dates_requires_options_flow_source(tmp_path):
    complete = tmp_path / "2026-05-14"
    complete.mkdir()
    for prefix in ["stock-screener-", "hot-chains-", "chain-oi-changes-", "bot-eod-report-"]:
        (complete / f"{prefix}2026-05-14.csv").write_text("ticker\nAAA\n", encoding="utf-8")

    missing_flow = tmp_path / "2026-05-15"
    missing_flow.mkdir()
    for prefix in ["stock-screener-", "hot-chains-", "chain-oi-changes-"]:
        (missing_flow / f"{prefix}2026-05-15.csv").write_text("ticker\nBBB\n", encoding="utf-8")

    overlay_like = tmp_path / "2026-05-14-v3-overlay-2026-05-15-live"
    overlay_like.mkdir()
    for prefix in ["stock-screener-", "hot-chains-", "chain-oi-changes-", "bot-eod-report-"]:
        (overlay_like / f"{prefix}2026-05-14.csv").write_text("ticker\nCCC\n", encoding="utf-8")

    assert source_complete_dates(tmp_path) == ["2026-05-14"]


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


def test_target_ready_keeps_risk_labeled_complete_edge_visible():
    row = {
        "status": "AVOID",
        "classification": "AVOID",
        "ticker": "AAPL",
        "direction": "bullish",
        "confidence_tier": "PROVEN",
        "probability_score": 62.0,
        "success_probability_pct": 58.0,
        "pattern_score": 12.0,
        "block_reasons": ["MAX_RISK_EXCEEDS_PER_TRADE_LIMIT"],
        "strategy_kind": "long_option",
        "strategy_type": "Long Call Debit",
        "lead_option_symbol": "AAPL260619C00210000",
        "expiry": "2026-06-19",
        "option_type": "call",
        "strike": 210,
        "entry_range": "7.50-7.70",
        "max_risk_per_contract": 770.65,
        "expected_R": 0.24,
        "expected_R_per_day": 0.012,
        "validation_profit_factor": 1.42,
        "beats_baselines_count": 2,
        "stop_rule": "Close if option loses 50% of debit or thesis breaks.",
    }

    candidates = build_target_ready_candidates([row], DEFAULT_RISK_CONFIG)

    assert candidates == [row]
    assert daily_trade_decision([], [], [row], candidates) == "TARGET_READY"
    output = target_ready_output_row(candidates[0])
    assert output["target_ready_status"] == "TARGET_READY"
    assert output["send_now"] == "no"
    assert output["live_recheck_required"] == "yes"
    assert output["target_debit_credit"] == "debit 7.50-7.70"
    assert output["trade_legs"] == "Buy 1 AAPL 2026-06-19 210C @ debit 7.50-7.70 limit"
    assert output["order_entry_missing_fields"] == ""
    assert "risk_limit_labeled_not_hidden" in output["risk_label"]
    assert "max risk $770.65 exceeds configured trade limit" in output["why_not_send_now"]


def test_target_ready_excludes_incomplete_or_negative_edge_tickets():
    complete_negative_edge = {
        "status": "TRADE_REVIEW",
        "classification": "WATCH",
        "ticker": "NVDA",
        "direction": "bearish",
        "confidence_tier": "PROMISING",
        "probability_score": 55.0,
        "success_probability_pct": 53.0,
        "pattern_score": 10.0,
        "block_reasons": ["EXPECTED_R_NOT_POSITIVE_AFTER_COSTS"],
        "strategy_kind": "long_option",
        "strategy_type": "Long Put Debit",
        "lead_option_symbol": "NVDA260619P00130000",
        "expiry": "2026-06-19",
        "option_type": "put",
        "strike": 130,
        "entry_range": "4.20-4.35",
        "max_risk_per_contract": 435.65,
        "expected_R": -0.02,
        "expected_R_per_day": -0.001,
        "validation_profit_factor": 1.25,
        "beats_baselines_count": 2,
        "stop_rule": "Close if option loses 50% of debit or thesis breaks.",
    }
    incomplete_ticket = {
        "status": "TRADE_REVIEW",
        "classification": "WATCH",
        "ticker": "MSFT",
        "direction": "bullish",
        "confidence_tier": "PROVEN",
        "probability_score": 64.0,
        "success_probability_pct": 59.0,
        "pattern_score": 13.0,
        "block_reasons": ["NO_TRADEABLE_OPTION_QUOTE"],
        "expected_R": 0.31,
        "expected_R_per_day": 0.014,
        "validation_profit_factor": 1.55,
        "beats_baselines_count": 2,
        "stop_rule": "Close if option loses 50% of debit or thesis breaks.",
    }

    candidates = build_target_ready_candidates(
        [complete_negative_edge, incomplete_ticket],
        DEFAULT_RISK_CONFIG,
    )

    assert candidates == []


def test_scout_call_candidates_surface_blocked_bullish_calls_without_approval():
    scout_call = {
        "status": "AVOID",
        "classification": "AVOID",
        "ticker": "NVDA",
        "direction": "bullish",
        "confidence_tier": "PROMISING",
        "probability_score": 37.0,
        "success_probability_pct": 44.0,
        "pattern_score": 11.0,
        "block_reasons": [
            "MARKET_REGIME_CONFLICT",
            "PATTERN_VALIDATION_NOT_PROVEN",
            "CALIBRATION_SCORE_MISSING_OR_WEAK",
        ],
        "strategy_kind": "long_option",
        "strategy_type": "Long Call Debit",
        "lead_option_symbol": "NVDA260717C00200000",
        "expiry": "2026-07-17",
        "option_type": "call",
        "strike": 200,
        "entry_bid": 3.60,
        "entry_ask": 3.80,
        "entry_range": "3.60-3.80",
        "bid_ask_spread_pct": 0.054,
        "max_risk_per_contract": 390.65,
        "expected_R": 0.22,
        "expected_R_per_day": 0.044,
        "validation_profit_factor": 1.66,
        "beats_baselines_count": 2,
        "flow_total_premium": 1_188_031_273.0,
        "flow_call_premium_share": 0.6543,
        "flow_call_ask_premium_share": 0.6602,
    }
    approved_call = dict(scout_call, status="AUTO_APPROVED", classification="TRADE", ticker="AAPL")
    quote_dead_call = dict(
        scout_call,
        ticker="MSFT",
        block_reasons=["NO_TRADEABLE_OPTION_QUOTE", "PATTERN_VALIDATION_NOT_PROVEN"],
    )
    put = dict(
        scout_call,
        ticker="TSLA",
        direction="bearish",
        strategy_type="Long Put Debit",
        lead_option_symbol="TSLA260717P00300000",
        option_type="put",
    )

    candidates = build_scout_call_candidates([quote_dead_call, approved_call, put, scout_call])

    assert candidates == [scout_call]
    output = scout_call_output_row(candidates[0], 1)
    assert output["scout_lane"] == "CONTRARIAN_CALL_REVIEW"
    assert output["contrarian_call_setup"] == "yes"
    assert output["trade_legs"] == "Buy 1 NVDA 2026-07-17 200C @ debit 3.60-3.80 limit"
    assert "mechanically conflicts with bullish" in output["why_not_send_now"]
    assert output["scout_score"] > output["probability_score"]


def test_source_coverage_quote_does_not_cross_direction_unless_allowed():
    bearish_quote = {
        "ticker": "KLAC",
        "direction": "bearish",
        "strategy_kind": "long_option",
        "option_symbol": "KLAC260320P01200000",
        "option_type": "put",
        "strike": 1200,
        "expiry": "2026-03-20",
        "bid": 16.9,
        "ask": 20.3,
        "selection_score": 10.0,
    }
    snapshot = SnapshotStub(
        [{"ticker": "KLAC"}],
        best_options={("KLAC", "bearish"): bearish_quote},
        option_quotes={},
    )

    assert source_coverage_quote(snapshot, {"ticker": "KLAC"}, "bullish", {}) == {}

    fallback = source_coverage_quote(
        snapshot,
        {"ticker": "KLAC"},
        "bullish",
        {},
        allow_opposite_direction_fallback=True,
    )
    setup = source_coverage_setup_fields("KLAC", "bullish", fallback)

    assert setup["strategy"] == "Long Put Debit"
    assert setup["call_or_put"] == "PUT"
    assert "1200P" in setup["trade_legs"]


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


def test_catalyst_flow_leader_without_probability_or_ev_is_avoid_not_review():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__BASIC_MATERIALS"
    daily_rows = [
        {
            "date": "2026-05-26",
            "classification": "WATCH",
            "ticker": "NEM",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "confidence_tier": "RESEARCH_ONLY",
            "pattern_score": 10.0,
            "block_reasons": [],
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "lead_option_symbol": "NEM260618C00120000",
            "expiry": "2026-06-18",
            "option_type": "call",
            "strike": 120,
            "entry_bid": 2.19,
            "entry_ask": 2.34,
            "entry_range": "2.19-2.34",
            "bid_ask_spread_pct": 0.066,
            "max_risk_per_contract": 234.65,
            "liquidity_volume": 269,
            "liquidity_open_interest": 14663,
            "dte": 23,
            "quote_source": "bot_eod",
            "flow_total_premium": 474_900_000.0,
            "hot_total_premium": 474_900_000.0,
        }
    ]

    rows, _ = prepare_decision_rows(
        daily_rows,
        empty_validation_bundle(),
        {"source_complete": True},
        {},
    )

    assert rows[0]["status"] == "AVOID"
    assert rows[0]["classification"] == "AVOID"
    assert "INSUFFICIENT_VALIDATION_FOR_TRADE_REVIEW" in rows[0]["block_reasons"]
    assert rows[0]["expected_R"] is None
    assert rows[0].get("probability_score") is None


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
    assert "regime OOS edge passed review" in recommendation["why_not_auto_approved"]
    assert "manual catalyst confirmation" in recommendation["why_not_auto_approved"]


def test_validated_regime_edge_bridge_promotes_market_conflict_only_ticker_trend():
    family = "SOURCE_PREMIUM_COVERAGE_RESCUE__BULLISH__LONG_OPTION__TECHNOLOGY"
    daily_rows = [
        {
            "date": "2026-07-07",
            "classification": "WATCH",
            "ticker": "HOOD",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "SOURCE_PREMIUM_COVERAGE_RESCUE",
            "confidence_tier": "PROMISING",
            "probability_score": 43.0,
            "success_probability_pct": 51.0,
            "pattern_score": 12.0,
            "block_reasons": ["MARKET_REGIME_CONFLICT"],
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "lead_option_symbol": "HOOD260717C00100000",
            "expiry": "2026-07-17",
            "option_type": "call",
            "strike": 100,
            "entry_bid": 5.05,
            "entry_ask": 5.20,
            "entry_range": "5.05-5.20",
            "bid_ask_spread_pct": 0.029,
            "max_risk_per_contract": 520.65,
            "liquidity_volume": 4200,
            "liquidity_open_interest": 9000,
            "dte": 10,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_OFF",
            "market_regime": "RISK_OFF",
        }
    ]
    hood_outcomes = []
    for idx in range(25):
        win = idx % 2 == 0 or idx == 1
        hood_outcomes.append(
            {
                "split": "cumulative_to_2026-06_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                "signal_date": "2026-06-01",
                "pattern_family": family,
                "ticker": "HOOD",
                "direction": "bullish",
                "strategy_kind": "long_option",
                "market_regime": "RISK_OFF",
                "status": "SCORED",
                "net_r": 1.2 if win else -0.15,
                "win": int(win),
            }
        )
    other_family_outcomes = [
        {
            "split": "cumulative_to_2026-06_holdout",
            "sample": "VALIDATION",
            "horizon": "5d",
            "signal_date": "2026-06-02",
            "pattern_family": family,
            "ticker": "LRCX",
            "direction": "bullish",
            "strategy_kind": "long_option",
            "market_regime": "RISK_ON",
            "status": "SCORED",
            "net_r": 0.65 if idx < 20 else -0.25,
            "win": int(idx < 20),
        }
        for idx in range(35)
    ]
    validation_bundle = empty_validation_bundle()
    validation_bundle["splits"] = [{"name": "cumulative_to_2026-06_holdout"}]
    validation_bundle["validation_gate_scorecard"] = [{"pattern_family": family, "signal_count": 60, "scored_count": 60}]
    validation_bundle["baseline_comparison"] = [
        {"baseline": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY", "average_net_r": -0.10, "scored_count": 20},
        {"baseline": "BASELINE_NAIVE_UW_FLOW_ONLY", "average_net_r": -0.05, "scored_count": 20},
    ]
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROVEN",
            "validation_scored_count": 60,
            "validation_win_count": 34,
            "validation_success_probability": 34 / 60,
            "validation_failure_probability": 26 / 60,
            "validation_probability_score": 0.50,
            "validation_average_net_r": 0.48,
            "validation_profit_factor": 4.9,
            "beats_baselines_count": 3,
            "baselines_beaten_names": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_NAIVE_UW_FLOW_ONLY",
            "baselines_beaten_details": "BASELINE_RANDOM_SAME_DATE_LIQUIDITY:baseline_avg_R=-0.10,edge_R=0.58,scored=20",
        }
    }
    validation_bundle["outcomes"] = hood_outcomes + other_family_outcomes

    rows, controls = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    assert rows[0]["status"] == "AUTO_APPROVED"
    assert rows[0]["classification"] == "TRADE"
    assert rows[0]["approval_bridge"] == "VALIDATED_REGIME_EDGE"
    assert rows[0]["regime_edge_review_passed"] == "yes"
    assert "MARKET_REGIME_CONFLICT" not in rows[0]["block_reasons"]
    assert rows[0]["auto_min_scored_outcomes"] == controls["risk_config"]["min_ticker_trend_scored_outcomes"]
    assert "same-regime OOS evidence bridged" in rows[0]["why_actionable_now"]
    trade_row = trade_output_row(rows[0])
    assert trade_row["approval_bridge"] == "VALIDATED_REGIME_EDGE"
    assert "approval_bridge=VALIDATED_REGIME_EDGE" in trade_row["auto_approval_gate_evidence"]
    assert "ticker_trend_scope=ticker_direction_strategy" in trade_row["auto_approval_gate_evidence"]


def test_blocker_text_explains_probability_thresholds_instead_of_raw_codes():
    row = {
        "block_reasons": ["CALIBRATION_SCORE_MISSING_OR_WEAK", "CONFIDENCE_BAND_TOO_WEAK"],
        "probability_score": 46.77,
        "calibrated_probability": 0.5168,
        "confidence_lower_bound": 0.4163,
        "auto_min_probability_score": 0.50,
        "auto_min_calibrated_probability": 0.50,
        "auto_min_confidence_lower_bound": 0.45,
    }

    text = blocker_text(row)

    assert "probability score 46.77% < 50.00% auto floor" in text
    assert "confidence lower bound 41.63% < 45.00% auto floor" in text
    assert "CALIBRATION_SCORE_MISSING_OR_WEAK" not in text


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
