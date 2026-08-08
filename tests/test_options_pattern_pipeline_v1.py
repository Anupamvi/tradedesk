import subprocess
import sqlite3
import zipfile
from pathlib import Path

import pytest

from uwos.options_pattern_pipeline_v1.macro_geo import (
    SCENARIO_BUCKETS,
    THEME_MAP,
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
    assign_directional_pattern_tiers,
    assign_regime_family_tiers,
    apply_shifted_chain_quotes,
    append_compact_action_board,
    append_compact_candidate_shortlist,
    append_current_pattern_members,
    append_contract_profile_edge_summary,
    append_directional_pattern_summary,
    auto_approved_goal_gate_failures,
    balanced_non_ready_trend_rows,
    backfill_selected_signal_quote_history,
    blocker_text,
    build_artifact_manifest,
    build_signal,
    build_catalyst_flow_leaders,
    build_calibration_metrics,
    build_daily_snapshot,
    build_decision_board_rows,
    build_directional_edge_diagnostic_rows,
    build_directional_outcome_rows,
    build_quote_selection_cache,
    build_current_directional_pattern_candidates,
    build_contract_profile_edge_rows,
    build_directional_scenario_goal_row,
    build_goal_evidence_rows,
    build_pattern_recommendations,
    build_source_ticker_coverage_rows,
    build_shadow_ledger_rows,
    build_scout_call_candidates,
    build_selected_chain_quote_store,
    build_scoring_snapshot,
    build_target_ready_candidates,
    build_theme_flow_leaders,
    build_theme_flow_signal_contexts,
    build_trade_review_candidates,
    build_ticker_trend_edge_rows,
    build_ticker_trend_stats,
    build_validation_splits,
    build_walk_forward_performance_rows,
    classify_daily_signals,
    compact_snapshot_option_quotes,
    conditional_trade_output_row,
    contract_profile_fields,
    daily_trade_decision,
    decision_board_fieldnames,
    dedupe_rows_by_ticket,
    empty_validation_bundle,
    family_has_required_sources,
    final_verdict,
    full_backtest_group_row,
    generate_signals_for_snapshot,
    goal_evidence_overall_status,
    missed_mover_bucket,
    matched_family_permutation_stats,
    new_bot_flow_agg,
    new_feature,
    normalize_header,
    option_quote_from_prior_chain_row,
    payoff_breakeven_probability,
    parse_args,
    prepare_decision_rows,
    parse_option_symbol,
    pattern_recommendation_fieldnames,
    resolve_run_verdict,
    probability_edge_over_breakeven_pct,
    proven_payoff_aware_promotion_eligible,
    run_historical_validation,
    score_signals,
    score_signal_horizon,
    score_signals_from_quote_store,
    sector_momentum_percentiles,
    scoring_session_dates,
    scout_call_fieldnames,
    select_active_tier_info,
    select_baseline_gate_outcomes,
    select_validation_gate_outcomes,
    select_qualified_ticker_trend,
    select_signal_set,
    source_coverage_quote,
    source_coverage_setup_fields,
    source_complete_dates,
    source_completeness_for_date,
    sources_for_date,
    summarize_outcomes,
    trade_fieldnames,
    trade_output_row,
    tradeable_gap_quote_eligible,
    trend_edge_strategy_fields,
    ticker_trend_no_edge_reason,
    ticker_trend_passes,
    target_ready_output_row,
    scout_call_output_row,
    pattern_recommendation_output_row,
    catalyst_flow_leader_output_row,
    theme_flow_leader_output_row,
    trade_review_output_row,
    update_chain_oi_aggregate,
    update_bot_flow_cache_agg,
    update_dark_pool_aggregate,
    update_hot_aggregate,
    finalize_feature,
    validate_decision_board_rows,
    validate_artifact_consistency,
    validation_detail_fieldnames,
)


class SnapshotStub:
    def __init__(self, features, best_options=None, market_regime=None, signal_date="2026-05-13", option_quotes=None):
        self.signal_date = signal_date
        self.features = features
        self.best_options = best_options or {}
        self.option_quotes = option_quotes or {}
        self.market_regime = market_regime or {"regime": "MIXED"}


def test_family_source_guards_and_symmetric_sector_momentum():
    features = {}
    best_options = {}
    for index in range(12):
        ticker = f"T{index:02d}"
        features[ticker] = {
            "ticker": ticker,
            "close": 100.0,
            "sector": "Technology",
            "issue_type": "Common Stock",
            "marketcap": 10_000_000_000.0,
            "pos_52w": index / 11.0,
            "source_flags": {"stock_screener", "hot_chains", "chain_oi"},
            "hot_total_volume": 1000.0,
            "hot_total_premium": 1_000_000.0,
            "total_open_interest": 1000.0,
        }
    features["T11"].update(
        {
            "oi_call_bought_diff": 20_000.0,
            "hot_call_ask_ratio": 0.90,
            "flow_total_premium": 200_000_000.0,
            "flow_call_premium_share": 0.90,
            "flow_put_premium_share": 0.10,
            "flow_call_ask_premium_share": 0.90,
            "flow_premium_bias": 0.80,
        }
    )
    for ticker, option_type, direction, strike in (
        ("T00", "put", "bearish", 95.0),
        ("T00", "call", "bullish", 105.0),
        ("T11", "call", "bullish", 105.0),
        ("T11", "put", "bearish", 95.0),
    ):
        best_options[(ticker, direction)] = {
            "option_symbol": f"{ticker}260918{option_type[0].upper()}00100000",
            "strategy_kind": "long_option",
            "option_type": option_type,
            "direction": direction,
            "expiry": "2026-09-18",
            "dte": 50,
            "strike": strike,
            "bid": 1.0,
            "ask": 1.1,
            "mid": 1.05,
            "volume": 100.0,
            "open_interest": 100.0,
            "premium": 11_000.0,
            "spread_pct": 0.095,
            "stock_close": 100.0,
        }
    snapshot = SnapshotStub(features, best_options, signal_date="2026-07-29")
    config = {
        "min_call_volume_ratio": 2.0,
        "min_put_volume_ratio": 2.0,
        "min_hot_premium": 100_000.0,
        "min_oi_diff": 5_000.0,
        "max_spread_pct": 0.35,
        "high_iv": 0.50,
        "min_liquidity_score": 8.0,
        "min_dark_pool_premium": 5_000_000.0,
        "min_dark_pool_directional_share": 0.60,
        "min_dark_pool_directional_coverage": 0.25,
        "min_ask_side_ratio": 0.52,
        "max_event_dte_without_event_strategy": 2,
    }

    ranks = sector_momentum_percentiles(snapshot)
    strict_sources = {
        "enforce_family_source_requirements": True,
        "enable_sector_momentum_families": True,
    }
    rows = generate_signals_for_snapshot(
        snapshot,
        config,
        max_signals=100,
        risk_config=strict_sources,
    )
    momentum = {
        (row["base_pattern_family"], row["direction"])
        for row in rows
        if row["base_pattern_family"].startswith("SECTOR_MOMENTUM")
    }

    assert ranks["T00"] <= 0.20
    assert ranks["T11"] >= 0.80
    assert momentum == {
        ("SECTOR_MOMENTUM_CONTINUATION", "bullish"),
        ("SECTOR_MOMENTUM_CONTINUATION", "bearish"),
        ("SECTOR_MOMENTUM_REVERSAL", "bullish"),
        ("SECTOR_MOMENTUM_REVERSAL", "bearish"),
    }
    assert any(row["base_pattern_family"] == "OI_GAMMA_CONTINUATION" for row in rows)
    assert not any(row["base_pattern_family"] == "CATALYST_FLOW_LEADER" for row in rows)
    assert not family_has_required_sources(features["T11"], "CATALYST_FLOW_LEADER")

    features["T11"]["source_flags"].add("bot_eod")
    rows_with_bot = generate_signals_for_snapshot(
        snapshot,
        config,
        max_signals=100,
        risk_config=strict_sources,
    )
    assert any(row["base_pattern_family"] == "CATALYST_FLOW_LEADER" for row in rows_with_bot)


def test_chain_oi_unwind_does_not_count_as_new_positioning():
    feature = {
        "oi_total_diff": 0.0,
        "oi_call_diff": 0.0,
        "oi_put_diff": 0.0,
        "oi_call_bought_diff": 0.0,
        "oi_put_bought_diff": 0.0,
        "oi_call_volume": 0.0,
        "oi_put_volume": 0.0,
        "oi_top_diff": 0.0,
        "oi_top_symbol": "",
        "oi_top_direction": "",
        "sector": "",
        "close": None,
        "next_earnings_date": "",
    }
    row = {
        "oi_diff_plain": "-50000",
        "volume": "60000",
        "prev_ask_volume": "50000",
        "prev_bid_volume": "1000",
        "prev_multi_leg_volume": "0",
    }

    update_chain_oi_aggregate(
        feature,
        row,
        {"option_type": "call", "option_symbol": "AAA260821C00100000"},
    )

    assert feature["oi_total_diff"] == 0.0
    assert feature["oi_call_diff"] == 0.0
    assert feature["oi_call_bought_diff"] == 0.0


def test_chain_oi_build_uses_non_multileg_ask_side_conviction():
    feature = {
        "oi_total_diff": 0.0,
        "oi_call_diff": 0.0,
        "oi_put_diff": 0.0,
        "oi_call_bought_diff": 0.0,
        "oi_put_bought_diff": 0.0,
        "oi_call_volume": 0.0,
        "oi_put_volume": 0.0,
        "oi_top_diff": 0.0,
        "oi_top_symbol": "",
        "oi_top_direction": "",
        "sector": "",
        "close": None,
        "next_earnings_date": "",
    }
    row = {
        "oi_diff_plain": "1000",
        "volume": "1500",
        "prev_ask_volume": "900",
        "prev_bid_volume": "300",
        "prev_multi_leg_volume": "400",
    }

    update_chain_oi_aggregate(
        feature,
        row,
        {"option_type": "call", "option_symbol": "AAA260821C00100000"},
    )

    assert feature["oi_call_diff"] == 1000.0
    assert feature["oi_call_bought_diff"] == pytest.approx(750.0)


def test_shifted_chain_quote_uses_only_prior_session_fields():
    parsed = parse_option_symbol("AAA260619C00100000")
    quote = option_quote_from_prior_chain_row(
        {
            "last_date": "2026-05-08",
            "last_bid": "1.00",
            "last_ask": "1.10",
            "last_fill": "1.05",
            "last_oi": "25",
            "curr_oi": "9999",
            "prev_vol": "100",
            "volume": "8888",
            "prev_ask_volume": "60",
            "prev_bid_volume": "30",
            "prev_mid_volume": "10",
            "prev_multi_leg_volume": "20",
            "prev_total_premium": "10500",
            "stock_price": "999",
        },
        parsed,
    )

    assert quote["date"] == "2026-05-08"
    assert quote["open_interest"] == 25
    assert quote["volume"] == 100
    assert quote["stock_close"] is None
    assert quote["quote_source"] == "shifted_chain_oi"

    prior = SnapshotStub(
        features={"AAA": {"close": 100.0}},
        signal_date="2026-05-08",
    )
    following = SnapshotStub(features={}, signal_date="2026-05-11")
    prior.prior_option_quotes = {}
    following.prior_option_quotes = {quote["option_symbol"]: quote}
    prior.counts = {}
    following.counts = {}

    assert apply_shifted_chain_quotes(
        {"2026-05-08": prior, "2026-05-11": following}
    ) == 1
    shifted = prior.option_quotes[quote["option_symbol"]]
    assert shifted["stock_close"] == 100.0
    assert shifted["open_interest"] == 25
    assert following.prior_option_quotes == {}


def test_selective_chain_replay_keeps_only_selected_contract(tmp_path):
    target_dir = tmp_path / "2026-05-08"
    source_dir = tmp_path / "2026-05-11"
    target_dir.mkdir()
    source_dir.mkdir()
    header = (
        "option_symbol,last_date,last_bid,last_ask,last_fill,last_oi,curr_oi,"
        "prev_vol,volume,prev_ask_volume,prev_bid_volume,prev_mid_volume,"
        "prev_multi_leg_volume,prev_total_premium\n"
    )
    rows = (
        "AAA260619C00100000,2026-05-08,1.00,1.10,1.05,25,9999,100,8888,60,30,10,20,10500\n"
        "BBB260619C00100000,2026-05-08,2.00,2.10,2.05,30,9999,200,9999,100,80,20,40,41000\n"
    )
    with zipfile.ZipFile(source_dir / "chain-oi-changes-2026-05-11.zip", "w") as archive:
        archive.writestr("chain-oi-changes-2026-05-11.csv", header + rows)
    target = SnapshotStub(
        features={"AAA": {"close": 100.0}, "BBB": {"close": 200.0}},
        signal_date="2026-05-08",
    )
    target.counts = {}
    target.prior_option_quotes = {}
    run_boundary = SnapshotStub(features={}, signal_date="2026-05-12")
    run_boundary.counts = {}
    run_boundary.prior_option_quotes = {}

    added = backfill_selected_signal_quote_history(
        tmp_path,
        {"2026-05-08": target, "2026-05-12": run_boundary},
        [[{"lead_option_symbol": "AAA260619C00100000", "legs_json": "[]"}]],
    )

    assert added == 1
    assert set(target.option_quotes) == {"AAA260619C00100000"}
    assert target.option_quotes["AAA260619C00100000"]["open_interest"] == 25


def test_disk_quote_store_scores_and_releases_future_marks(tmp_path):
    source_dir = tmp_path / "2026-01-06"
    source_dir.mkdir()
    symbol = "XYZ260320C00100000"
    (source_dir / "chain-oi-changes-2026-01-06.csv").write_text(
        "option_symbol,last_date,last_bid,last_ask,last_fill,last_oi,prev_vol,"
        "prev_ask_volume,prev_bid_volume,prev_mid_volume,prev_multi_leg_volume,"
        "prev_total_premium\n"
        f"{symbol},2026-01-05,1.55,1.60,1.57,100,200,150,30,20,0,31400\n",
        encoding="utf-8",
    )
    snapshots = {
        "2026-01-02": SnapshotStub(
            {"XYZ": {"close": 100.0}},
            signal_date="2026-01-02",
        ),
        "2026-01-05": SnapshotStub(
            {"XYZ": {"close": 101.0}},
            signal_date="2026-01-05",
        ),
        "2026-01-06": SnapshotStub(
            {"XYZ": {"close": 102.0}},
            signal_date="2026-01-06",
        ),
    }
    store_path, count = build_selected_chain_quote_store(
        tmp_path,
        {symbol},
        snapshots.keys(),
    )
    original_mtime = store_path.stat().st_mtime_ns
    reused_path, reused_count = build_selected_chain_quote_store(
        tmp_path,
        {symbol},
        snapshots.keys(),
    )
    signal = {
        "date": "2026-01-02",
        "ticker": "XYZ",
        "direction": "bullish",
        "pattern_family": "SECTOR_MOMENTUM_CONTINUATION__BULLISH__LONG_OPTION__TECHNOLOGY",
        "market_regime": "MIXED",
        "sector": "Technology",
        "lead_option_symbol": symbol,
        "strategy_kind": "long_option",
        "entry_ask": 1.0,
        "entry_bid": 0.95,
        "bid_ask_spread_pct": 0.05,
        "block_reasons": [],
        "close": 100.0,
    }
    connection = sqlite3.connect(store_path)
    try:
        rows = score_signals_from_quote_store(
            [signal],
            snapshots,
            list(snapshots),
            "unit",
            "VALIDATION",
            {
                "validation_horizon_sessions": 2,
                "long_option_profit_target_pct": 0.50,
                "long_option_stop_loss_pct": None,
            },
            connection,
        )
    finally:
        connection.close()

    assert count == 1
    assert reused_path == store_path
    assert reused_count == count
    assert store_path.stat().st_mtime_ns == original_mtime
    assert rows[0]["managed_exit_date"] == "2026-01-05"
    assert rows[0]["managed_exit_price"] == pytest.approx(1.50)
    assert rows[0]["win"] == 1
    assert snapshots["2026-01-05"].option_quotes == {}


def test_partial_date_is_scoring_session_but_not_full_signal_snapshot(tmp_path):
    full = tmp_path / "2026-05-08"
    partial = tmp_path / "2026-05-11"
    full.mkdir()
    partial.mkdir()
    stock_header = "ticker,close,prev_close,high,low,sector,issue_type\n"
    stock_row = "AAA,101,100,102,99,Technology,Common Stock\n"
    for day in (full, partial):
        (day / f"stock-screener-{day.name}.csv").write_text(
            stock_header + stock_row,
            encoding="utf-8",
        )
        (day / f"hot-chains-{day.name}.csv").write_text(
            "option_symbol,date\n",
            encoding="utf-8",
        )
    for prefix in ("chain-oi-changes", "dp-eod-report", "bot-eod-report"):
        (full / f"{prefix}-{full.name}.csv").write_text("ticker\n", encoding="utf-8")

    sessions = scoring_session_dates(tmp_path, "2026-05-11")
    scoring = build_scoring_snapshot(tmp_path, "2026-05-11")

    assert sessions == ["2026-05-08", "2026-05-11"]
    assert source_complete_dates(tmp_path) == ["2026-05-08"]
    assert scoring.features["AAA"]["close"] == 101.0
    assert scoring.option_quotes == {}
    assert scoring.market_regime["regime"] == "SCORING_ONLY"
    assert scoring.skipped_sources == [{"reason": "scoring_session_only_not_signal_eligible"}]


def test_hot_chain_direction_removes_multileg_volume_per_contract():
    feature = new_feature("2026-07-29", "AAA")
    quote = {
        "stock_close": 100.0,
        "volume": 100.0,
        "premium": 10_000.0,
        "sweep_volume": 0.0,
        "multileg_volume": 60.0,
        "iv": 0.40,
        "spread_pct": 0.05,
        "option_type": "call",
        "ask_side_volume": 80.0,
        "bid_side_volume": 20.0,
    }

    update_hot_aggregate(feature, quote, {})
    finalize_feature(feature)

    assert feature["hot_call_ask_volume"] == 50.0
    assert feature["hot_call_bid_volume"] == 0.0
    assert feature["hot_call_ask_ratio"] == 1.0
    assert feature["hot_directional_bias"] == 1.0

    second = dict(quote)
    second.update(
        {
            "option_type": "put",
            "spread_pct": None,
            "ask_side_volume": 20.0,
            "bid_side_volume": 80.0,
        }
    )
    update_hot_aggregate(feature, second, {})
    finalize_feature(feature)
    assert feature["hot_put_ask_volume"] == 0.0
    assert feature["hot_put_bid_volume"] == 50.0
    assert feature["hot_put_ask_ratio"] == 0.0


def test_dark_pool_direction_excludes_invalid_sale_conditions():
    feature = new_feature("2026-07-29", "AAA")
    invalid = {
        "sale_cond_codes": "prior_reference_price",
        "premium": "1000000",
        "size": "10000",
        "price": "101",
        "nbbo_bid": "99",
        "nbbo_ask": "100",
    }
    valid = {
        "sale_cond_codes": "",
        "premium": "500000",
        "size": "5000",
        "price": "101",
        "nbbo_bid": "99",
        "nbbo_ask": "100",
    }

    update_dark_pool_aggregate(feature, invalid)
    update_dark_pool_aggregate(feature, valid)

    assert feature["dp_trade_count"] == 1.0
    assert feature["dp_total_premium"] == 500000.0
    assert feature["dp_above_ask_premium"] == 500000.0


def test_bot_flow_excludes_canceled_and_multileg_rows_from_direction():
    aggregate = new_bot_flow_agg("2026-07-29", "AAA")
    canceled = {
        "canceled": "t",
        "side": "ask",
        "option_type": "call",
        "premium": "1000",
    }
    multileg = {
        "canceled": "f",
        "upstream_condition_detail": "mlet",
        "side": "ask",
        "option_type": "call",
        "premium": "2000",
        "size": "10",
        "vega": "0.5",
        "gamma": "0.1",
    }
    directional = {
        "canceled": "f",
        "upstream_condition_detail": "slan",
        "side": "bid",
        "option_type": "put",
        "premium": "3000",
        "size": "10",
        "vega": "0.5",
        "gamma": "0.1",
    }

    update_bot_flow_cache_agg(aggregate, canceled)
    update_bot_flow_cache_agg(aggregate, multileg)
    update_bot_flow_cache_agg(aggregate, directional)

    assert aggregate["row_count"] == 2
    assert aggregate["flow_gross_premium"] == 5000.0
    assert aggregate["flow_multileg_premium"] == 2000.0
    assert aggregate["flow_total_premium"] == 3000.0
    assert aggregate["flow_put_bid_premium"] == 3000.0
    assert aggregate["flow_call_ask_premium"] == 0.0
    assert aggregate["flow_greek_rows"] == 1


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
    assert PIPELINE_VERSION == "options_pattern_pipeline_v1.17-five-source-managed-selection-20260808-000000"
    assert PREVIOUS_PIPELINE_VERSIONS == (
        "options_pattern_pipeline_v1.2",
        "options_pattern_pipeline_v1.3",
        "options_pattern_pipeline_v1.4",
        "options_pattern_pipeline_v1.5-promotion-bridge-20260707-000000",
        "options_pattern_pipeline_v1.6-theme-flow-observability-20260716-000000",
        "options_pattern_pipeline_v1.7-validation-correctness-20260716-000000",
        "options_pattern_pipeline_v1.8-compact-action-report-20260717-000000",
        "options_pattern_pipeline_v1.9-payoff-aware-approval-20260717-000000",
        "options_pattern_pipeline_v1.10-scoped-recent-trend-20260719-000000",
        "options_pattern_pipeline_v1.11-family-member-validation-20260719-120000",
        "options_pattern_pipeline_v1.12-full-chain-oi-20260719-140000",
        "options_pattern_pipeline_v1.13-two-stage-contract-aware-20260720-090000",
        "options_pattern_pipeline_v1.14-full-history-two-stage-20260720-185300",
        "options_pattern_pipeline_v1.15-final-holdout-audit-20260720-220000",
        "options_pattern_pipeline_v1.16-profile-aware-daily-selection-20260722-000000",
    )
    assert PIPELINE_RELEASED_AT == "2026-08-08T00:00:00-07:00"


def test_compact_action_board_surfaces_one_trade_without_wide_markdown_table():
    trade = {
        "classification": "TRADE",
        "status": "AUTO_APPROVED",
        "candidate_id": "crwd-call",
        "ticker": "CRWD",
        "direction": "bullish",
        "strategy_type": "Long Call Debit",
        "strategy_kind": "long_option",
        "option_type": "CALL",
        "strike": 210,
        "expiry": "2026-08-21",
        "lead_option_symbol": "CRWD260821C00210000",
        "suggested_entry_debit_credit_range": "debit 13.20-13.80",
        "max_risk_per_contract": 1410.65,
        "trade_success_probability_pct": 63.64,
        "trade_failure_probability_pct": 36.36,
        "trade_probability_score": 53.0,
        "expected_R": 0.665979,
        "expected_R_per_day": 0.133196,
        "discovered_pattern_family": "BULLISH_FLOW_EXPANSION",
        "block_reasons": "",
    }
    lines = []

    primary = append_compact_action_board(lines, "2026-07-16", [trade], [trade], [], [])
    report = "\n".join(lines)

    assert primary == trade
    assert "1 actionable ticket for 2026-07-16, from 1 proven pattern family" in report
    assert "### TRADE: CRWD BULLISH" in report
    assert "- Strike(s): 210" in report
    assert "- Entry: debit 13.20-13.80" in report
    assert "|" not in report


def test_compact_action_board_labels_best_watch_when_nothing_is_actionable():
    watch = {
        "classification": "WATCH",
        "status": "TRADE_REVIEW",
        "candidate_id": "meta-call",
        "ticker": "META",
        "direction": "bullish",
        "strategy": "Long Call Debit",
        "buy_or_sell": "BUY",
        "call_or_put": "CALL",
        "strike_rates": "800",
        "expiration_date": "2026-08-21",
        "suggested_entry_debit_credit_range": "debit 9.60-9.85",
        "expected_R": 0.412242,
        "trade_probability_score": 39.82,
        "block_reasons": "CALIBRATION_SCORE_MISSING_OR_WEAK;CONFIDENCE_BAND_TOO_WEAK",
    }
    lines = []

    primary = append_compact_action_board(lines, "2026-07-16", [], [watch], [watch], [watch])
    report = "\n".join(lines)

    assert primary == watch
    assert "No actionable pattern trades for 2026-07-16" in report
    assert "### Best WATCH candidate: META BULLISH" in report
    assert "Why not actionable" in report
    assert "|" not in report


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


def test_final_verdict_reports_promising_family_as_usable_needs_more_validation():
    validation_bundle = {
        "family_tiers": {
            "PROMISING_FAMILY": {"confidence_tier": "PROMISING"},
            "RESEARCH_FAMILY": {"confidence_tier": "RESEARCH_ONLY"},
        },
        "validation_scorecard": [{"pattern_family": "PROMISING_FAMILY"}],
    }

    assert final_verdict(validation_bundle, [], actionable_rows=[]) == "USABLE_NEEDS_MORE_VALIDATION"


def test_final_verdict_is_not_yet_proven_with_research_only_families():
    validation_bundle = {
        "family_tiers": {"RESEARCH_FAMILY": {"confidence_tier": "RESEARCH_ONLY"}},
        "validation_scorecard": [{"pattern_family": "RESEARCH_FAMILY"}],
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


def test_bounded_signal_set_preserves_each_detailed_family_lane():
    signals = [
        {
            "date": "2026-05-28",
            "ticker": f"TECH{i}",
            "direction": "bullish",
            "pattern_family": "FLOW__BULLISH__LONG_OPTION__TECHNOLOGY",
            "base_pattern_family": "FLOW",
            "strategy_kind": "long_option",
            "pattern_score": 100.0 - i,
            "hot_total_premium": 1_000_000.0,
        }
        for i in range(10)
    ]
    signals.append(
        {
            "date": "2026-05-28",
            "ticker": "ENERGY",
            "direction": "bullish",
            "pattern_family": "FLOW__BULLISH__LONG_OPTION__ENERGY",
            "base_pattern_family": "FLOW",
            "strategy_kind": "long_option",
            "pattern_score": 1.0,
            "hot_total_premium": 10_000.0,
        }
    )

    selected = select_signal_set(
        signals,
        max_signals=1,
        source_rescue_max_extra=0,
        tradeable_gap_max_extra=0,
    )

    assert {row["pattern_family"] for row in selected} == {
        "FLOW__BULLISH__LONG_OPTION__TECHNOLOGY",
        "FLOW__BULLISH__LONG_OPTION__ENERGY",
    }


def test_uncapped_validation_keeps_distinct_contract_profiles_but_daily_board_prefers_qualified_profile():
    signals = [
        {
            "date": "2026-05-28",
            "ticker": "XYZ",
            "direction": "bullish",
            "pattern_family": "BULLISH_FLOW_EXPANSION",
            "base_pattern_family": "BULLISH_FLOW_EXPANSION",
            "strategy_kind": "long_option",
            "pattern_score": 10.0,
            "contract_selection_score": 9.0,
            "contract_profile": "LONG_OPTION__DTE_7_13__DEEP_ITM",
        },
        {
            "date": "2026-05-28",
            "ticker": "XYZ",
            "direction": "bullish",
            "pattern_family": "BULLISH_FLOW_EXPANSION",
            "base_pattern_family": "BULLISH_FLOW_EXPANSION",
            "strategy_kind": "long_option",
            "pattern_score": 10.0,
            "contract_selection_score": 8.0,
            "contract_profile": "LONG_OPTION__DTE_14_30__ATM",
            "contract_profile_goal_qualified": "yes",
        },
    ]

    validation_rows = select_signal_set(
        signals,
        max_signals=0,
        source_rescue_max_extra=0,
        tradeable_gap_max_extra=0,
    )
    daily_rows = select_signal_set(
        signals,
        max_signals=40,
        source_rescue_max_extra=0,
        tradeable_gap_max_extra=0,
    )

    assert {row["contract_profile"] for row in validation_rows} == {
        "LONG_OPTION__DTE_7_13__DEEP_ITM",
        "LONG_OPTION__DTE_14_30__ATM",
    }
    assert len(daily_rows) == 1
    assert daily_rows[0]["contract_profile"] == "LONG_OPTION__DTE_14_30__ATM"


def test_quote_cache_keeps_one_liquid_long_option_per_standard_dte_and_moneyness_profile():
    def quote(symbol, dte, strike):
        return {
            "option_symbol": symbol,
            "ticker": "XYZ",
            "direction": "bullish",
            "strategy_kind": "long_option",
            "option_type": "call",
            "expiry": "2026-08-21",
            "dte": dte,
            "strike": strike,
            "ask": 2.0,
            "bid": 1.9,
            "spread_pct": 0.05,
            "volume": 500,
            "open_interest": 1000,
            "premium": 100_000.0,
        }

    snapshot = SnapshotStub(
        {"XYZ": {"ticker": "XYZ", "close": 100.0}},
        option_quotes={
            "near": quote("near", 10, 100),
            "near_otm": quote("near_otm", 10, 110),
            "near_atm_worse": quote("near_atm_worse", 10, 100),
            "standard": quote("standard", 21, 102),
            "medium": quote("medium", 35, 103),
            "long": quote("long", 55, 105),
        },
    )

    cache = build_quote_selection_cache(snapshot, {"max_spread_pct": 0.35}, DEFAULT_RISK_CONFIG)
    frontier = cache["long_option_dte_frontier"][("XYZ", "bullish")]

    assert [quote["option_symbol"] for quote in frontier] == [
        "near",
        "near_otm",
        "standard",
        "medium",
        "long",
    ]


def test_quote_compaction_preserves_deterministic_signal_identities():
    def quote(symbol, direction, option_type, dte, strike, score):
        return {
            "option_symbol": symbol,
            "ticker": "XYZ",
            "direction": direction,
            "strategy_kind": "long_option",
            "option_type": option_type,
            "expiry": "2026-09-18",
            "dte": dte,
            "strike": strike,
            "ask": 1.0 + score / 100.0,
            "bid": 0.95 + score / 100.0,
            "spread_pct": 0.04,
            "volume": 500 + score,
            "open_interest": 1000 + score,
            "premium": 100_000.0,
            "stock_close": 100.0,
        }

    quotes = {
        "C1": quote("XYZ260821C00100000", "bullish", "call", 18, 100, 1),
        "C2": quote("XYZ260918C00105000", "bullish", "call", 45, 105, 2),
        "C_UNUSED": quote("XYZ261218C00200000", "bullish", "call", 100, 200, 0),
        "P1": quote("XYZ260821P00100000", "bearish", "put", 18, 100, 1),
        "P2": quote("XYZ260918P00095000", "bearish", "put", 45, 95, 2),
    }
    feature = {
        "ticker": "XYZ",
        "close": 100.0,
        "sector": "Technology",
        "issue_type": "Common Stock",
        "marketcap": 10_000_000_000.0,
        "pos_52w": 0.95,
        "source_flags": {"stock_screener", "hot_chains", "chain_oi"},
        "hot_total_volume": 1000.0,
        "hot_total_premium": 1_000_000.0,
        "total_open_interest": 1000.0,
    }
    snapshot = SnapshotStub(
        {"XYZ": feature},
        best_options={
            ("XYZ", "bullish"): quotes["C1"],
            ("XYZ", "bearish"): quotes["P1"],
        },
        option_quotes=quotes,
        signal_date="2026-07-29",
    )
    snapshot.counts = {}
    pattern_config = {
        "min_call_volume_ratio": 2.0,
        "min_put_volume_ratio": 2.0,
        "min_hot_premium": 100_000.0,
        "min_oi_diff": 5_000.0,
        "max_spread_pct": 0.35,
        "high_iv": 0.50,
        "min_liquidity_score": 8.0,
        "min_dark_pool_premium": 5_000_000.0,
        "min_dark_pool_directional_share": 0.60,
        "min_dark_pool_directional_coverage": 0.25,
        "min_ask_side_ratio": 0.52,
        "max_event_dte_without_event_strategy": 2,
    }
    risk = {
        "enable_sector_momentum_families": True,
        "enforce_family_source_requirements": True,
        "max_risk_per_trade": 1500.0,
    }

    before = generate_signals_for_snapshot(snapshot, pattern_config, 100, risk_config=risk)
    compact_snapshot_option_quotes(snapshot, risk)
    after = generate_signals_for_snapshot(snapshot, pattern_config, 100, risk_config=risk)
    identity = lambda row: (
        row["base_pattern_family"],
        row["direction"],
        row["strategy_kind"],
        row["contract_profile"],
        row["lead_option_symbol"],
    )

    assert {identity(row) for row in after} == {identity(row) for row in before}
    assert "XYZ261218C00200000" not in snapshot.option_quotes


def test_goal_evidence_overall_status_reports_pass_when_every_requirement_passes():
    assert goal_evidence_overall_status([{"status": "PASS"}, {"status": "PASS"}]) == "GOAL_REQUIREMENTS_PASSED"


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
            "pattern_family": "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__TECHNOLOGY",
            "base_pattern_family": "CATALYST_FLOW_LEADER",
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
            "ticker_trend_scope": "ticker_direction_strategy_pattern",
            "ticker_trend_base_pattern_family": "CATALYST_FLOW_LEADER",
            "ticker_trend_scored_count": 23,
            "ticker_trend_unique_signal_date_count": 23,
            "ticker_trend_win_rate_pct": 56.52,
            "ticker_trend_probability_score_pct": 46.13,
            "ticker_trend_avg_R": 0.681731,
            "ticker_trend_profit_factor": 5.051584,
            "ticker_trend_validation_split_count": 3,
            "ticker_trend_positive_validation_splits": 3,
            "ticker_trend_latest_validation_split_average_net_R": 0.25,
            "ticker_trend_recent_scored_count": 6,
            "ticker_trend_recent_win_rate_pct": 66.67,
            "ticker_trend_recent_avg_R": 0.40,
            "ticker_trend_recent_profit_factor": 2.5,
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

    decayed = dict(decision_board[0])
    decayed.update(
        {
            "ticker_trend_recent_win_rate_pct": 0.0,
            "ticker_trend_recent_avg_R": -0.49,
            "ticker_trend_recent_profit_factor": 0.0,
        }
    )
    failures = auto_approved_goal_gate_failures(decayed, DEFAULT_RISK_CONFIG)
    assert "ticker_trend_recent_win_rate" in failures
    assert "ticker_trend_recent_expected_R" in failures
    assert "ticker_trend_recent_profit_factor" in failures


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
    assert bundle["signal_rows"] == []
    assert bundle["validation_history_status"] == "INSUFFICIENT_SOURCE_COMPLETE_HISTORY"
    assert bundle["source_date_count"] == len(dates)
    assert bundle["source_month_date_counts"] == {"2025-12": 5, "2026-01": 1}


def test_validation_candidate_limit_is_independent_from_daily_board_limit():
    args = parse_args(["--top-candidates-per-day", "40"])

    assert args.top_candidates_per_day == 40
    assert args.validation_top_candidates_per_day == 0
    assert args.missed_mover_audit_days == 20


def test_full_backtest_group_tracks_unsorted_date_bounds():
    row = full_backtest_group_row(
        "pattern_family",
        "TEST",
        [
            {"signal_date": "2026-07-20", "status": "SCORED", "net_r": 0.2},
            {"signal_date": "2026-01-05", "status": "SCORED", "net_r": -0.1},
            {"signal_date": "2026-03-10", "status": "PARTIAL", "net_r": None},
        ],
    )

    assert row["first_signal_date"] == "2026-01-05"
    assert row["last_signal_date"] == "2026-07-20"


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


def test_score_signals_can_limit_horizons_for_baseline_memory_control():
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
        "2026-01-05": type("S", (), {"option_quotes": {}, "features": {}})(),
    }

    rows = score_signals(
        [signal],
        snapshots,
        ["2026-01-02", "2026-01-05"],
        "unit",
        "BASELINE",
        horizons=(1,),
    )

    assert len(rows) == 1
    assert rows[0]["horizon"] == "1d"


def test_score_signals_marks_strategy_specific_primary_horizons():
    dates = [f"D{index:02d}" for index in range(45)]
    snapshots = {
        date: type("S", (), {"option_quotes": {}, "features": {"XYZ": {"close": 100.0}}})()
        for date in dates
    }
    common = {
        "date": dates[0],
        "ticker": "XYZ",
        "direction": "bullish",
        "market_regime": "MIXED",
        "sector": "Tech",
        "entry_ask": 1.0,
        "entry_bid": 0.9,
        "bid_ask_spread_pct": 0.1,
        "block_reasons": [],
        "close": 100.0,
    }
    signals = [
        {
            **common,
            "pattern_family": "LONG_FAMILY",
            "strategy_kind": "long_option",
            "lead_option_symbol": "XYZ260320C00100000",
        },
        {
            **common,
            "pattern_family": "CREDIT_FAMILY",
            "strategy_kind": "credit_spread",
            "entry_credit": 1.0,
            "max_risk_per_contract": 400.0,
            "legs_json": "[]",
        },
    ]

    rows = score_signals(
        signals,
        snapshots,
        dates,
        "cumulative_to_2026-03_holdout",
        "VALIDATION",
        {
            "validation_horizon_sessions": 40,
            "credit_spread_validation_horizon_sessions": 5,
        },
        horizons=None,
    )

    by_family = {row["pattern_family"]: row for row in rows}
    assert by_family["LONG_FAMILY"]["horizon"] == "40d"
    assert by_family["CREDIT_FAMILY"]["horizon"] == "5d"
    assert all(row["primary_validation_horizon"] for row in rows)
    assert len(select_validation_gate_outcomes(rows)) == 2


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


def test_managed_long_option_closes_at_configured_50pct_target_without_stop():
    symbol = "XYZ260320C00100000"
    signal = {
        "date": "2026-01-02",
        "ticker": "XYZ",
        "direction": "bullish",
        "pattern_family": "BULLISH_FLOW_EXPANSION",
        "market_regime": "MIXED",
        "sector": "Tech",
        "lead_option_symbol": symbol,
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
                "option_quotes": {symbol: {"bid": 1.55, "ask": 1.60}},
                "features": {"XYZ": {"close": 101.0}},
            },
        )(),
        "2026-01-06": type(
            "S",
            (),
            {
                "option_quotes": {symbol: {"bid": 0.30, "ask": 0.40, "high": 0.45, "low": 0.20}},
                "features": {"XYZ": {"close": 95.0}},
            },
        )(),
    }

    row = score_signal_horizon(
        signal,
        snapshots,
        ["2026-01-02", "2026-01-05", "2026-01-06"],
        "unit",
        "VALIDATION",
        2,
        {
            "long_option_profit_target_pct": 0.50,
            "long_option_stop_loss_pct": None,
        },
    )

    assert row["status"] == "SCORED"
    assert row["managed_exit_date"] == "2026-01-05"
    assert row["managed_exit_price"] == pytest.approx(1.50)
    assert row["win"] == 1
    assert row["outcome_note"] == "managed_long_option_target_hit_after_costs_slippage"


def test_immature_long_option_cohort_is_censored_before_early_winner_is_counted():
    symbol = "XYZ261218C00100000"
    signal = {
        "date": "2026-07-28",
        "ticker": "XYZ",
        "direction": "bullish",
        "pattern_family": "UNIT_LONG_FAMILY",
        "market_regime": "MIXED",
        "sector": "Tech",
        "lead_option_symbol": symbol,
        "strategy_kind": "long_option",
        "entry_ask": 1.0,
        "entry_bid": 0.9,
        "bid_ask_spread_pct": 0.1,
        "block_reasons": [],
        "close": 100.0,
    }
    snapshots = {
        "2026-07-28": type("S", (), {"option_quotes": {}, "features": {"XYZ": {"close": 100.0}}})(),
        "2026-07-29": type(
            "S",
            (),
            {
                "option_quotes": {symbol: {"bid": 1.60, "ask": 1.65}},
                "features": {"XYZ": {"close": 102.0}},
            },
        )(),
    }

    rows = score_signals(
        [signal],
        snapshots,
        ["2026-07-28", "2026-07-29"],
        "cumulative_to_2026-07_holdout",
        "VALIDATION",
        {
            "validation_horizon_sessions": 40,
            "long_option_profit_target_pct": 0.50,
            "long_option_stop_loss_pct": None,
        },
        horizons=None,
    )
    scorecard = summarize_outcomes(rows, "VALIDATION")

    assert rows[0]["status"] == "CENSORED_OPEN"
    assert rows[0]["net_r"] is None
    assert rows[0]["outcome_note"] == "primary_horizon_not_yet_mature"
    assert scorecard[0]["censored_open_count"] == 1
    assert scorecard[0]["scored_count"] == 0
    assert scorecard[0]["profit_factor"] is None


def test_credit_spread_scores_managed_target_with_future_leg_quotes():
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
    assert row["outcome_note"] == "managed_credit_spread_target_hit_after_costs_slippage"
    assert row["managed_exit_price"] == pytest.approx(0.50)
    assert row["round_trip_fees"] == 10.0
    assert row["opening_fee"] == 5.0
    assert row["entry_slippage"] == pytest.approx(10.0)
    assert row["exit_slippage"] == pytest.approx(10.0)
    assert row["slippage_dollars"] == pytest.approx(20.0)
    assert row["cost_model"] == "credit_spread_entry_credit_exit_debit_after_configured_fees"
    assert row["net_r"] == pytest.approx((50.0 - 10.0 - 20.0) / 411.30)


def test_credit_spread_closes_at_target_before_later_stop():
    signal = {
        "date": "2026-01-02",
        "ticker": "XYZ",
        "direction": "bullish",
        "pattern_family": "OI_GAMMA_CONTINUATION",
        "market_regime": "MIXED",
        "sector": "Tech",
        "strategy_kind": "credit_spread",
        "legs_json": '[{"action":"SELL","option_symbol":"SHORT"},{"action":"BUY","option_symbol":"LONG"}]',
        "entry_credit": 1.0,
        "max_risk_per_contract": 400.0,
        "entry_ask": 1.0,
        "entry_bid": 1.0,
        "bid_ask_spread_pct": 0.0,
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
                    "SHORT": {"bid": 0.40, "ask": 0.45},
                    "LONG": {"bid": 0.05, "ask": 0.10},
                },
                "features": {"XYZ": {"close": 101.0}},
            },
        )(),
        "2026-01-06": type(
            "S",
            (),
            {
                "option_quotes": {
                    "SHORT": {"bid": 2.50, "ask": 2.60},
                    "LONG": {"bid": 0.10, "ask": 0.20},
                },
                "features": {"XYZ": {"close": 90.0}},
            },
        )(),
    }

    row = score_signal_horizon(
        signal,
        snapshots,
        ["2026-01-02", "2026-01-05", "2026-01-06"],
        "unit",
        "VALIDATION",
        2,
        {"round_trip_spread_fees": 0.0, "slippage_pct_of_spread": 0.0},
    )

    assert row["managed_exit_date"] == "2026-01-05"
    assert row["managed_exit_price"] == pytest.approx(0.50)
    assert row["net_r"] == pytest.approx(0.125)
    assert row["win"] == 1


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


def test_dated_whale_fallback_is_discovered_without_crossing_dates(tmp_path):
    date_dir = tmp_path / "2026-01-08"
    date_dir.mkdir()
    dated = date_dir / "whale_trades_filtered-2026-01-08.csv"
    dated.write_text("executed_at,underlying_symbol\n", encoding="utf-8")
    mismatched = date_dir / "whale_trades_filtered-2026-01-09.csv"
    mismatched.write_text("executed_at,underlying_symbol\n", encoding="utf-8")

    sources = sources_for_date(date_dir, "2026-01-08")

    assert [ref.path for ref in sources["whale_filtered"]] == [dated]


def test_legacy_whale_fallback_name_is_discovered_in_its_date_folder(tmp_path):
    date_dir = tmp_path / "2026-01-02"
    date_dir.mkdir()
    legacy = date_dir / "whale_trades_filtered 01-02.csv"
    legacy.write_text("executed_at,underlying_symbol\n", encoding="utf-8")

    sources = sources_for_date(date_dir, "2026-01-02")

    assert [ref.path for ref in sources["whale_filtered"]] == [legacy]


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


def test_bot_quote_refresh_policy_keeps_flow_but_not_bot_only_contracts(tmp_path):
    date_dir = tmp_path / "2026-05-08"
    date_dir.mkdir()
    hot_header = (
        "option_symbol,date,volume,open_interest,premium,ask_side_volume,bid_side_volume,"
        "sweep_volume,multileg_volume,bid,ask,option_close,avg_price,iv\n"
    )
    hot_row = "XYZ260619C00100000,2026-05-08,200,100,20000,150,20,10,0,0.90,1.10,1.00,1.00,0.40\n"
    (date_dir / "hot-chains-2026-05-08.csv").write_text(
        hot_header + hot_row,
        encoding="utf-8",
    )
    bot_header = (
        "executed_at,underlying_symbol,option_chain_id,side,option_type,underlying_price,"
        "nbbo_bid,nbbo_ask,price,size,premium,volume,open_interest,implied_volatility,"
        "upstream_condition_detail,canceled\n"
    )
    bot_rows = (
        "2026-05-08T20:00:00Z,XYZ,XYZ260619C00100000,ask,call,101,1.00,1.20,1.10,200,22000,200,100,0.45,slan,f\n"
        "2026-05-08T20:00:00Z,ABC,ABC260619C00100000,ask,call,100,2.00,2.20,2.10,200,42000,200,100,0.50,slan,f\n"
    )
    (date_dir / "bot-eod-report-2026-05-08.csv").write_text(
        bot_header + bot_rows,
        encoding="utf-8",
    )

    snap = build_daily_snapshot(
        tmp_path,
        "2026-05-08",
        {
            "max_chain_rows_per_day": 0,
            "max_flow_file_mb": 100.0,
            "bot_eod_cache_dir": str(tmp_path / "cache"),
            "risk_config": {"bot_eod_quote_policy": "refresh_existing"},
        },
    )

    assert set(snap.option_quotes) == {"XYZ260619C00100000"}
    assert snap.option_quotes["XYZ260619C00100000"]["bid"] == 1.0
    assert snap.features["XYZ"]["flow_total_premium"] == 22000.0
    assert snap.features["ABC"]["flow_total_premium"] == 42000.0
    assert snap.counts["bot_eod_quote_rows_seen"] == 2
    assert snap.counts["bot_eod_quote_rows"] == 1
    assert snap.counts["bot_eod_quote_rows_not_retained"] == 1


def test_zero_chain_oi_limit_streams_full_export(tmp_path):
    date_dir = tmp_path / "2026-05-08"
    date_dir.mkdir()
    header = "option_symbol,curr_date,oi_diff_plain,volume,stock_price,sector\n"
    rows = "".join(
        [
            "XYZ260515C00100000,2026-05-08,10,100,101,Technology\n",
            "XYZ260515C00105000,2026-05-08,20,200,101,Technology\n",
            "XYZ260515C00110000,2026-05-08,30,300,101,Technology\n",
        ]
    )
    with zipfile.ZipFile(date_dir / "chain-oi-changes-2026-05-08.zip", "w") as zf:
        zf.writestr("chain-oi-changes-2026-05-08.csv", header + rows)

    snap = build_daily_snapshot(
        tmp_path,
        "2026-05-08",
        {
            "max_chain_rows_per_day": 0,
            "max_flow_file_mb": 100.0,
            "bot_eod_cache_dir": str(tmp_path / "cache"),
        },
    )

    assert snap.counts["chain_oi_rows"] == 3
    assert snap.features["XYZ"]["oi_call_diff"] == 60
    assert not any(item["reason"] == "max_chain_rows_per_day" for item in snap.skipped_sources)


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


def test_strict_family_tier_requires_matched_permutation_evidence():
    scorecard = []
    outcomes = []
    for split_index, month in enumerate(("05", "06")):
        split = f"cumulative_to_2026-{month}_holdout"
        scorecard.append(
            {
                "split": split,
                "horizon": "5d",
                "pattern_family": "EDGE",
                "signal_count": 30,
                "scored_count": 30,
                "win_count_scored": 24,
                "unique_signal_date_count": 30,
                "date_cluster_win_count": 24,
                "average_net_r": 0.60,
                "gross_profit_r": 24.0,
                "gross_loss_r": 6.0,
                "profit_factor": 4.0,
                "worst_losing_streak": 1,
            }
        )
        for day in range(1, 31):
            outcomes.append(
                {
                    "split": split,
                    "sample": "VALIDATION",
                    "horizon": "5d",
                    "signal_date": f"2026-{month}-{day:02d}",
                    "pattern_family": "EDGE",
                    "status": "SCORED",
                    "net_r": 1.0 if day <= 24 else -1.0,
                }
            )
    config = {
        "min_proven_family_expected_r": 0.05,
        "min_oos_unique_signal_dates": 20,
        "require_every_validation_split_profitable": True,
        "require_day_clustered_pf_for_proven": True,
        "min_day_clustered_profit_factor_p05": 1.20,
        "day_clustered_bootstrap_iterations": 500,
        "require_matched_permutation_for_proven": True,
        "max_matched_null_p_value": 0.05,
    }
    baselines = [
        {"baseline": "BASELINE_A", "average_net_r": -0.10, "scored_count": 60},
        {"baseline": "BASELINE_B", "average_net_r": 0.00, "scored_count": 60},
    ]

    missing_null = assign_family_tiers(scorecard, baselines, config, outcomes)["EDGE"]
    assert missing_null["confidence_tier"] != "PROVEN"
    assert missing_null["validation_day_clustered_profit_factor_p05"] >= 1.20
    assert missing_null["every_validation_split_profitable"] is True
    assert missing_null["deployment_gate_failures"] == "MATCHED_PERMUTATION_EVIDENCE_MISSING_OR_WEAK"

    for row in scorecard:
        row["matched_null_p_value"] = 0.01
        row["matched_null_coverage"] = 1.0
    with_null = assign_family_tiers(scorecard, baselines, config, outcomes)["EDGE"]
    assert with_null["confidence_tier"] == "PROVEN"
    assert with_null["deployment_gate_failures"] == ""


def test_matched_family_permutation_preserves_date_sector_direction_and_profile():
    outcomes = []
    for day in range(1, 31):
        common = {
            "sample": "VALIDATION",
            "horizon": "5d",
            "signal_date": f"2026-05-{day:02d}",
            "sector": "Technology",
            "direction": "bullish",
            "strategy_kind": "long_option",
            "contract_profile": "LONG_OPTION__DTE_31_45__NEAR_OTM",
            "status": "SCORED",
            "legs_json": "[]",
        }
        outcomes.append(
            {
                **common,
                "ticker": "EDGE",
                "lead_option_symbol": f"EDGE{day:02d}",
                "pattern_family": "EDGE_FAMILY",
                "net_r": 1.0 if day <= 24 else -1.0,
            }
        )
        outcomes.append(
            {
                **common,
                "ticker": "CONTROL",
                "lead_option_symbol": f"CONTROL{day:02d}",
                "pattern_family": "CONTROL_FAMILY",
                "net_r": -1.0,
            }
        )

    stats = matched_family_permutation_stats(outcomes, trials=200, seed=7)

    edge = stats["EDGE_FAMILY"]
    control = stats["CONTROL_FAMILY"]
    assert edge["matched_null_coverage"] == 1.0
    assert edge["matched_null_actual_profit_factor"] == pytest.approx(4.0)
    assert edge["matched_null_median_profit_factor"] == 0.0
    assert edge["matched_null_p_value"] == pytest.approx(1 / 201)
    assert control["matched_null_p_value"] == 1.0


def test_summarize_outcomes_accepts_empty_spread_cells_from_saved_csv():
    rows = summarize_outcomes(
        [
            {
                "split": "cumulative_to_2026-05_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                "pattern_family": "EDGE",
                "status": "SCORED",
                "net_r": "0.25",
                "bid_ask_spread_pct": "",
            }
        ],
        sample="VALIDATION",
    )

    assert rows[0]["average_net_r"] == 0.25
    assert rows[0]["average_bid_ask_spread"] is None


def test_family_tier_uses_pooled_profit_factor_and_not_arbitrary_losing_streak():
    tiers = assign_family_tiers(
        [
            {
                "pattern_family": "EDGE",
                "signal_count": 100,
                "scored_count": 100,
                "win_count_scored": 60,
                "average_net_r": 0.20,
                "gross_profit_r": 30.0,
                "gross_loss_r": 10.0,
                "profit_factor": 3.0,
                "worst_losing_streak": 40,
            },
            {
                "pattern_family": "EDGE",
                "signal_count": 10,
                "scored_count": 10,
                "win_count_scored": 6,
                "average_net_r": 0.10,
                "gross_profit_r": 2.0,
                "gross_loss_r": 2.0,
                "profit_factor": 1.0,
                "worst_losing_streak": 2,
            },
        ],
        [
            {"baseline": "BASELINE_A", "average_net_r": -0.10, "scored_count": 50},
            {"baseline": "BASELINE_B", "average_net_r": 0.00, "scored_count": 50},
        ],
    )

    edge = tiers["EDGE"]
    assert edge["validation_profit_factor"] == pytest.approx(32.0 / 12.0)
    assert edge["confidence_tier"] == "PROVEN"
    assert edge["max_worst_losing_streak"] == 40


def test_family_tier_marks_historical_edge_decayed_when_latest_holdout_is_negative():
    scorecard = []
    for month, avg_r in [("04", 0.40), ("05", 0.30), ("06", 0.20), ("07", -0.10)]:
        scorecard.append(
            {
                "split": f"cumulative_to_2026-{month}_holdout",
                "pattern_family": "EDGE",
                "signal_count": 25,
                "scored_count": 25,
                "win_count_scored": 14,
                "average_net_r": avg_r,
                "gross_profit_r": 20.0 if avg_r > 0 else 5.0,
                "gross_loss_r": 10.0 if avg_r > 0 else 7.0,
                "profit_factor": 2.0 if avg_r > 0 else 5.0 / 7.0,
                "worst_losing_streak": 4,
            }
        )

    tier = assign_family_tiers(
        scorecard,
        [
            {"baseline": "BASELINE_A", "average_net_r": -0.10, "scored_count": 50},
            {"baseline": "BASELINE_B", "average_net_r": 0.00, "scored_count": 50},
        ],
    )["EDGE"]

    assert tier["confidence_tier"] == "DECAYED"
    assert tier["positive_validation_splits"] == 3
    assert tier["latest_validation_split"] == "cumulative_to_2026-07_holdout"
    assert tier["latest_validation_split_average_net_r"] == -0.10


def test_regime_family_tier_can_prove_active_edge_hidden_by_broad_family():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__COMMUNICATION_SERVICES"
    outcomes = []
    for month_index, month in enumerate(("04", "05", "06", "07")):
        positive_split = month_index != 2
        for idx in range(20):
            win = idx < (10 if positive_split else 5)
            outcomes.append(
                {
                    "split": f"cumulative_to_2026-{month}_holdout",
                    "sample": "VALIDATION",
                    "horizon": "5d",
                    "signal_date": f"2026-{month}-{idx + 1:02d}",
                    "ticker": "META",
                    "direction": "bullish",
                    "strategy_kind": "long_option",
                    "pattern_family": family,
                    "market_regime": "RISK_OFF",
                    "status": "SCORED",
                    "net_r": (1.0 if win else -0.5) if positive_split else (0.5 if win else -0.3),
                    "win": int(win),
                }
            )
    baselines = [
        {"baseline": "BASELINE_A", "average_net_r": -0.10, "scored_count": 50},
        {"baseline": "BASELINE_B", "average_net_r": 0.00, "scored_count": 50},
    ]

    tiers = assign_regime_family_tiers(outcomes, baselines)
    tier = tiers[(family, "RISK_OFF")]
    selected = select_active_tier_info(
        family,
        "RISK_OFF",
        {family: {"confidence_tier": "RESEARCH_ONLY"}},
        tiers,
    )

    assert tier["confidence_tier"] == "PROVEN"
    assert tier["pattern_scope"] == "FAMILY_REGIME"
    assert tier["validation_scored_count"] == 80
    assert tier["positive_validation_splits"] == 3
    assert tier["latest_validation_split_average_net_r"] > 0
    assert selected["active_pattern_id"].endswith("MARKET_REGIME_RISK_OFF")


def test_ticker_trend_rejects_pooled_edge_that_breaks_in_recent_splits():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__TECHNOLOGY"
    outcomes = []
    split_shapes = [
        ("cumulative_to_2026-05_holdout", 14, {5, 12}),
        ("cumulative_to_2026-06_holdout", 4, {0, 2, 3}),
        ("cumulative_to_2026-07_holdout", 4, {0, 2, 3}),
    ]
    for split, count, loss_indexes in split_shapes:
        month = split[19:26]
        for idx in range(count):
            win = idx not in loss_indexes
            outcomes.append(
                {
                    "split": split,
                    "sample": "VALIDATION",
                    "horizon": "5d",
                    "signal_date": f"{month}-{idx + 1:02d}",
                    "ticker": "CRWD",
                    "direction": "bullish",
                    "strategy_kind": "long_option",
                    "pattern_family": family,
                    "market_regime": "RISK_OFF",
                    "status": "SCORED",
                    "net_r": (1.5 if count == 14 else 0.2) if win else -0.5,
                    "win": int(win),
                }
            )

    stats = build_ticker_trend_stats(outcomes)[
        ("CRWD", "bullish", "long_option", "CATALYST_FLOW_LEADER")
    ]

    assert stats["win_rate"] > 0.60
    assert stats["avg_r"] > 0.15
    assert stats["profit_factor"] > 1.50
    assert stats["positive_validation_splits"] == 1
    assert stats["latest_validation_split_average_net_r"] < 0
    assert ticker_trend_passes(stats, DEFAULT_RISK_CONFIG) is False


def test_ticker_trend_is_scoped_to_the_current_base_pattern_family():
    catalyst_family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__TECHNOLOGY"
    source_family = "SOURCE_PREMIUM_COVERAGE_RESCUE__BULLISH__LONG_OPTION__TECHNOLOGY"
    outcomes = []
    for idx in range(24):
        win = idx % 4 != 0
        outcomes.append(
            {
                "split": f"cumulative_to_2026-{4 + idx // 8:02d}_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                "signal_date": f"2026-{4 + idx // 8:02d}-{idx % 8 + 1:02d}",
                "ticker": "TQQQ",
                "direction": "bullish",
                "strategy_kind": "long_option",
                "pattern_family": catalyst_family,
                "status": "SCORED",
                "net_r": 1.5 if win else -0.5,
                "win": int(win),
            }
        )
    for idx, net_r in enumerate((1.0, -0.5, 1.0, -0.5), 1):
        outcomes.append(
            {
                "split": "cumulative_to_2026-06_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                "signal_date": f"2026-06-{idx + 10:02d}",
                "ticker": "TQQQ",
                "direction": "bullish",
                "strategy_kind": "long_option",
                "pattern_family": source_family,
                "status": "SCORED",
                "net_r": net_r,
                "win": int(net_r > 0),
            }
        )

    stats = build_ticker_trend_stats(outcomes, DEFAULT_RISK_CONFIG)
    catalyst_key = ("TQQQ", "bullish", "long_option", "CATALYST_FLOW_LEADER")
    source_key = ("TQQQ", "bullish", "long_option", "SOURCE_PREMIUM_COVERAGE_RESCUE")

    assert stats[catalyst_key]["scored_count"] == 24
    assert stats[source_key]["scored_count"] == 4
    assert select_qualified_ticker_trend(
        {
            "ticker": "TQQQ",
            "direction": "bullish",
            "strategy_kind": "long_option",
            "pattern_family": source_family,
            "base_pattern_family": "SOURCE_PREMIUM_COVERAGE_RESCUE",
        },
        stats,
        DEFAULT_RISK_CONFIG,
    ) is None
    qualified = select_qualified_ticker_trend(
        {
            "ticker": "TQQQ",
            "direction": "bullish",
            "strategy_kind": "long_option",
            "pattern_family": catalyst_family,
            "base_pattern_family": "CATALYST_FLOW_LEADER",
        },
        stats,
        DEFAULT_RISK_CONFIG,
    )
    assert qualified is not None
    assert qualified["trend_scope"] == "ticker_direction_strategy_pattern"


def test_ticker_trend_rejects_recent_decay_hidden_by_positive_monthly_split():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__TECHNOLOGY"
    outcomes = []
    for month in (4, 5):
        for idx in range(10):
            win = idx < 7
            outcomes.append(
                {
                    "split": f"cumulative_to_2026-{month:02d}_holdout",
                    "sample": "VALIDATION",
                    "horizon": "5d",
                    "signal_date": f"2026-{month:02d}-{idx + 1:02d}",
                    "ticker": "TQQQ",
                    "direction": "bullish",
                    "strategy_kind": "long_option",
                    "pattern_family": family,
                    "status": "SCORED",
                    "net_r": 1.5 if win else -0.5,
                    "win": int(win),
                }
            )
    for idx in range(10):
        win = idx < 4
        outcomes.append(
            {
                "split": "cumulative_to_2026-06_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                "signal_date": f"2026-06-{idx + 1:02d}",
                "ticker": "TQQQ",
                "direction": "bullish",
                "strategy_kind": "long_option",
                "pattern_family": family,
                "status": "SCORED",
                "net_r": 2.0 if win else -0.5,
                "win": int(win),
            }
        )

    stats = build_ticker_trend_stats(outcomes, DEFAULT_RISK_CONFIG)[
        ("TQQQ", "bullish", "long_option", "CATALYST_FLOW_LEADER")
    ]

    assert stats["win_rate"] >= DEFAULT_RISK_CONFIG["min_ticker_trend_win_rate"]
    assert stats["profit_factor"] >= DEFAULT_RISK_CONFIG["min_ticker_trend_profit_factor"]
    assert stats["positive_validation_splits"] == 3
    assert stats["latest_validation_split_average_net_r"] > 0
    assert stats["recent_scored_count"] == 6
    assert stats["recent_win_count"] == 0
    assert stats["recent_avg_r"] < 0
    assert ticker_trend_passes(stats, DEFAULT_RISK_CONFIG) is False


def test_baseline_gate_outcomes_use_non_overlapping_cumulative_splits():
    rows = [
        {"sample": "BASELINE", "horizon": "5d", "split": "discover_2026-01_validate_2026-03", "id": 1},
        {"sample": "BASELINE", "horizon": "5d", "split": "cumulative_to_2026-03_holdout", "id": 2},
        {"sample": "BASELINE", "horizon": "3d", "split": "cumulative_to_2026-03_holdout", "id": 3},
    ]

    assert [row["id"] for row in select_baseline_gate_outcomes(rows)] == [2]


def test_calibration_is_prequential_instead_of_fitted_on_evaluated_outcomes():
    bundle = {
        "family_tiers": {
            "EDGE": {
                # Deliberately impossible post-hoc value; calibration must not use it.
                "validation_probability_score": 0.99,
            }
        },
        "outcomes": [
            {
                "sample": "VALIDATION",
                "horizon": "5d",
                "split": "cumulative_to_2026-05_holdout",
                "signal_date": "2026-05-01",
                "ticker": "A",
                "pattern_family": "EDGE",
                "status": "SCORED",
                "win": 1,
            },
            {
                "sample": "VALIDATION",
                "horizon": "5d",
                "split": "cumulative_to_2026-05_holdout",
                "signal_date": "2026-05-02",
                "ticker": "B",
                "pattern_family": "EDGE",
                "status": "SCORED",
                "win": 0,
            },
        ],
    }

    metrics = build_calibration_metrics(bundle)

    assert metrics["calibration_method"] == "PREQUENTIAL_BETA_1_1_CANONICAL_WALK_FORWARD"
    assert metrics["brier_score"] == pytest.approx(((0.5 - 1.0) ** 2 + ((2.0 / 3.0) - 0.0) ** 2) / 2.0)


def test_walk_forward_and_shadow_artifacts_do_not_duplicate_overlapping_splits():
    gate_row = {
        "split": "cumulative_to_2026-05_holdout",
        "sample": "VALIDATION",
        "pattern_family": "EDGE",
        "scored_count": 1,
    }
    exploratory_row = {
        **gate_row,
        "split": "discover_2026-03_validate_2026-05",
    }
    bundle = {
        "family_tiers": {"EDGE": {"confidence_tier": "PROMISING"}},
        "validation_scorecard": [gate_row, exploratory_row],
        "validation_gate_scorecard": [gate_row],
        "outcomes": [
            {
                "sample": "VALIDATION",
                "horizon": "5d",
                "split": "cumulative_to_2026-05_holdout",
                "signal_date": "2026-05-01",
                "pattern_family": "EDGE",
                "status": "SCORED",
                "net_r": 0.2,
            },
            {
                "sample": "VALIDATION",
                "horizon": "5d",
                "split": "discover_2026-03_validate_2026-05",
                "signal_date": "2026-05-01",
                "pattern_family": "EDGE",
                "status": "SCORED",
                "net_r": 0.2,
            },
        ],
    }

    assert len(build_walk_forward_performance_rows(bundle)) == 1
    historical = [row for row in build_shadow_ledger_rows("2026-05-10", [], bundle) if row["status"] == "HISTORICAL_SHADOW"]
    assert len(historical) == 1


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
    assert rows[0]["trade_success_probability_pct"] == rows[0]["pattern_success_probability_pct"]
    assert rows[0]["trade_probability_score"] == rows[0]["pattern_probability_score"]
    assert "contract_profile_calibration=pending" in rows[0]["probability_components"]


def test_candidate_probability_does_not_use_unfitted_same_day_execution_heuristics():
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

    assert by_ticker["GOOD"]["trade_success_probability_pct"] == by_ticker["BAD"]["trade_success_probability_pct"]
    assert by_ticker["GOOD"]["trade_probability_score"] == by_ticker["BAD"]["trade_probability_score"]
    assert by_ticker["GOOD"]["pattern_success_probability_pct"] == by_ticker["BAD"]["pattern_success_probability_pct"]


def test_watch_classification_does_not_create_circular_probability_penalty():
    signal = {
        "date": "2026-05-08",
        "ticker": "WATCH",
        "direction": "bullish",
        "pattern_family": "EDGE",
        "pattern_score": 1.0,
        "classification": "WATCH",
        "block_reasons": [],
        "bid_ask_spread_pct": 0.10,
        "liquidity_volume": 100,
        "liquidity_open_interest": 100,
        "earnings_dte": 30,
    }
    snapshot = type("S", (), {"market_regime": {"regime": "RISK_ON"}})()
    tier = {
        "confidence_tier": "PROMISING",
        "validation_scored_count": 80,
        "beats_baselines_count": 2,
        "validation_success_probability": 0.55,
        "validation_failure_probability": 0.45,
        "validation_probability_score": 0.50,
        "validation_note": "unit",
    }

    row = classify_daily_signals([signal], {"EDGE": tier}, snapshot)[0]

    assert row["classification"] == "WATCH"
    assert row["probability_components"] == "family_proxy_only=55.00%; contract_profile_calibration=pending"


def test_contract_profile_fields_bucket_dte_and_directional_moneyness():
    fields = contract_profile_fields(
        "bullish",
        "long_option",
        100.0,
        {"option_type": "call", "strike": 105.0, "dte": 21},
    )

    assert fields["contract_directional_moneyness"] == pytest.approx(0.05)
    assert fields["contract_dte_bucket"] == "DTE_14_30"
    assert fields["contract_moneyness_bucket"] == "NEAR_OTM"
    assert fields["contract_profile"] == "LONG_OPTION__DTE_14_30__NEAR_OTM"


def test_directional_outcomes_dedupe_contract_alternatives_and_cluster_dates():
    family = "EDGE__BULLISH__LONG_OPTION__TECHNOLOGY"
    rows = []
    for split_index, month in enumerate(("04", "05", "06")):
        for date_index in range(5):
            move = 0.01 if date_index < 4 else -0.005
            for ticker_index in range(2):
                for contract_index in range(2):
                    rows.append(
                        {
                            "split": f"cumulative_to_2026-{month}_holdout",
                            "sample": "VALIDATION",
                            "horizon": "5d",
                            "signal_date": f"2026-{month}-{date_index + 1:02d}",
                            "ticker": f"T{ticker_index}",
                            "direction": "bullish",
                            "pattern_family": family,
                            "base_pattern_family": "EDGE",
                            "sector": "Technology",
                            "lead_option_symbol": f"OPT{split_index}{date_index}{ticker_index}{contract_index}",
                            "status": "SCORED",
                            "net_r": 0.5,
                            "stock_proxy_move": move,
                        }
                    )

    directional = build_directional_outcome_rows(rows)
    tiers = assign_directional_pattern_tiers(
        directional,
        [
            {"baseline": "B0", "average_net_r": -0.001, "scored_count": 30},
            {"baseline": "B1", "average_net_r": 0.0, "scored_count": 30},
        ],
    )
    tier = tiers["EDGE__BULLISH__TECHNOLOGY"]

    assert len(directional) == 30
    assert tier["scored_count"] == 30
    assert tier["date_cluster_count"] == 15
    assert tier["date_cluster_win_count"] == 12
    assert tier["confidence_tier"] == "PROVEN_DIRECTIONAL"
    assert tier["probability_score"] < tier["win_rate"]


def test_comparable_contract_profile_can_still_produce_an_approved_ticket():
    family = "EDGE__BULLISH__LONG_OPTION__TECHNOLOGY"
    profile = "LONG_OPTION__DTE_14_30__ATM"
    daily_rows = [
        {
            "date": "2026-07-17",
            "ticker": "XYZ",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "EDGE",
            "sector": "Technology",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 101,
            "expiry": "2026-08-07",
            "dte": 21,
            "underlying_price": 100,
            "contract_reference_strike": 101,
            "contract_directional_moneyness": 0.01,
            "contract_dte_bucket": "DTE_14_30",
            "contract_moneyness_bucket": "ATM",
            "contract_profile": profile,
            "lead_option_symbol": "XYZ260807C00101000",
            "entry_bid": 4.90,
            "entry_ask": 5.00,
            "entry_range": "4.90-5.00",
            "bid_ask_spread_pct": 0.02,
            "max_risk_per_contract": 500.65,
            "liquidity_volume": 2000,
            "liquidity_open_interest": 3000,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_ON",
            "market_regime": "RISK_ON",
            "block_reasons": [],
        }
    ]
    outcomes = []
    for month in ("04", "05", "06", "07"):
        for index in range(10):
            win = index < 8
            outcomes.append(
                {
                    "split": f"cumulative_to_2026-{month}_holdout",
                    "sample": "VALIDATION",
                    "horizon": "5d",
                    "signal_date": f"2026-{month}-{index + 1:02d}",
                    "ticker": f"T{index % 3}",
                    "direction": "bullish",
                    "pattern_family": family,
                    "base_pattern_family": "EDGE",
                    "sector": "Technology",
                    "market_regime": "RISK_ON",
                    "strategy_kind": "long_option",
                    "contract_profile": profile,
                    "status": "SCORED",
                    "net_r": 0.60 if win else -0.20,
                    "win": int(win),
                }
            )
    validation_bundle = empty_validation_bundle()
    validation_bundle["outcomes"] = outcomes
    validation_bundle["baseline_comparison"] = [
        {"baseline": "B0", "average_net_r": -0.10, "scored_count": 40},
        {"baseline": "B1", "average_net_r": 0.0, "scored_count": 40},
    ]
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROVEN",
            "pattern_scope": "FAMILY",
            "validation_scored_count": 40,
            "validation_win_count": 32,
            "validation_success_probability": 0.80,
            "validation_probability_score": 0.70,
            "validation_average_net_r": 0.44,
            "validation_profit_factor": 12.0,
            "beats_baselines_count": 2,
            "baselines_beaten_names": "B0;B1",
            "baselines_beaten_details": "two frozen controls beaten",
        }
    }

    rows, _ = prepare_decision_rows(daily_rows, validation_bundle, {"source_complete": True}, {})
    candidate = rows[0]

    assert candidate["status"] == "AUTO_APPROVED"
    assert candidate["contract_profile_validated"] == "yes"
    assert candidate["contract_profile_scored_count"] == 40
    assert candidate["contract_profile_unique_signal_date_count"] == 40
    assert candidate["confidence_lower_bound"] > 0.70
    assert candidate["expected_R"] == pytest.approx(0.44)
    assert candidate["block_reasons"] == []


def test_contract_profile_goal_edge_requires_every_robustness_gate():
    key = ("EDGE__BULLISH__LONG_OPTION__TECHNOLOGY", "ALL", "LONG_OPTION__DTE_14_30__ATM")
    stats = {
        "scored_count": 40,
        "date_cluster_count": 40,
        "unique_ticker_count": 5,
        "historical_tickers": "A;B;C;D;E",
        "win_rate": 0.80,
        "date_cluster_win_rate": 0.80,
        "date_cluster_probability_score": 0.7298,
        "avg_win_r": 0.60,
        "avg_loss_r": -0.20,
        "avg_r": 0.44,
        "avg_r_without_largest_win": 0.435,
        "profit_factor": 12.0,
        "validation_split_count": 4,
        "positive_validation_splits": 4,
        "latest_validation_split": "cumulative_to_2026-07_holdout",
        "latest_validation_split_average_net_r": 0.40,
        "latest_validation_split_average_net_r_without_largest_win": 0.35,
    }
    baselines = [
        {"baseline": "B0", "average_net_r": -0.10, "scored_count": 40},
        {"baseline": "B1", "average_net_r": 0.0, "scored_count": 40},
    ]

    qualified = build_contract_profile_edge_rows({key: stats}, baselines)[0]
    broken = dict(stats)
    broken["latest_validation_split_average_net_r_without_largest_win"] = -0.01
    rejected = build_contract_profile_edge_rows({key: broken}, baselines)[0]

    assert qualified["qualified_goal_edge"] == "yes"
    assert qualified["confidence_lower_pct"] > qualified["payoff_breakeven_pct"]
    assert qualified["beats_baselines_count"] == 2
    assert rejected["qualified_goal_edge"] == "no"
    assert "LATEST_SPLIT_WITHOUT_BEST_WIN_NOT_POSITIVE" in rejected["qualification_failures"]


def test_contract_profile_goal_edge_requires_the_global_final_holdout():
    june_key = ("JUNE_EDGE", "ALL", "LONG_OPTION__DTE_14_30__ATM")
    july_key = ("JULY_REFERENCE", "ALL", "LONG_OPTION__DTE_14_30__ATM")
    base = {
        "scored_count": 40,
        "date_cluster_count": 40,
        "unique_ticker_count": 5,
        "historical_tickers": "A;B;C;D;E",
        "win_rate": 0.80,
        "date_cluster_win_rate": 0.80,
        "date_cluster_probability_score": 0.7298,
        "avg_win_r": 0.60,
        "avg_loss_r": -0.20,
        "avg_r": 0.44,
        "avg_r_without_largest_win": 0.435,
        "profit_factor": 12.0,
        "validation_split_count": 4,
        "positive_validation_splits": 4,
        "latest_validation_split_average_net_r": 0.40,
        "latest_validation_split_average_net_r_without_largest_win": 0.35,
    }
    june = dict(base, latest_validation_split="cumulative_to_2026-06_holdout")
    july = dict(base, latest_validation_split="cumulative_to_2026-07_holdout")
    baselines = [
        {"baseline": "B0", "average_net_r": -0.10, "scored_count": 40},
        {"baseline": "B1", "average_net_r": 0.0, "scored_count": 40},
    ]

    rows = build_contract_profile_edge_rows(
        {june_key: june, july_key: july},
        baselines,
    )
    by_family = {row["pattern_family"]: row for row in rows}

    assert by_family["JULY_REFERENCE"]["qualified_goal_edge"] == "yes"
    assert by_family["JUNE_EDGE"]["qualified_goal_edge"] == "no"
    assert by_family["JUNE_EDGE"]["required_latest_validation_split"] == "cumulative_to_2026-07_holdout"
    assert "LATEST_SPLIT_NOT_CURRENT_HOLDOUT" in by_family["JUNE_EDGE"]["qualification_failures"]

    only_june = build_contract_profile_edge_rows(
        {june_key: june},
        baselines,
        required_latest_split="cumulative_to_2026-07_holdout",
    )[0]
    assert only_june["qualified_goal_edge"] == "no"
    assert only_june["latest_holdout_present"] == "no"
    assert "LATEST_SPLIT_NOT_CURRENT_HOLDOUT" in only_june["qualification_failures"]


def test_compact_pattern_and_contract_report_sections_render_independently():
    directional = {
        "directional_pattern_family": "EDGE__BULLISH__TECHNOLOGY",
        "confidence_tier": "PROVEN_DIRECTIONAL",
        "scored_count": 40,
        "unique_signal_date_count": 20,
        "average_directional_move": 0.01,
        "average_directional_move_without_largest_win": 0.009,
        "latest_validation_split_average_directional_move": 0.008,
        "profit_factor": 2.0,
        "positive_validation_splits": 4,
        "validation_split_count": 4,
        "probability_score": 0.55,
        "beats_baselines_count": 2,
    }
    profile = {
        "qualified_goal_edge": "yes",
        "pattern_family": "EDGE__BULLISH__LONG_OPTION__TECHNOLOGY",
        "contract_profile": "LONG_OPTION__DTE_14_30__ATM",
        "scored_count": 40,
        "unique_signal_date_count": 20,
        "average_net_R": 0.20,
        "average_net_R_without_largest_win": 0.18,
        "latest_split_average_net_R": 0.15,
        "profit_factor": 1.8,
        "confidence_lower_pct": 55.0,
        "payoff_breakeven_pct": 45.0,
    }
    lines = []

    append_directional_pattern_summary(lines, [directional], [])
    append_contract_profile_edge_summary(lines, [profile], 3)
    report = "\n".join(lines)

    assert "## Directional Patterns Found" in report
    assert "Historical result: 1 proven" in report
    assert "## Contract Implementation Proof" in report
    assert "Qualified profiles: 1 across 1/3 required distinct families" in report


def test_current_pattern_report_does_not_mix_contract_and_family_scopes():
    primary = {
        "active_pattern_id": "EDGE__BULLISH__CREDIT_SPREAD__TECHNOLOGY",
        "ticker": "XYZ",
        "direction": "bullish",
        "status": "TRADE_REVIEW",
        "classification": "WATCH",
        "contract_profile": "CREDIT_SPREAD__DTE_7_13__NEAR_OTM",
        "contract_profile_scored_count": 13,
        "contract_profile_unique_signal_date_count": 10,
        "contract_profile_validation_split_count": 4,
        "contract_profile_avg_R": 0.1683,
        "contract_profile_avg_R_without_largest_win": 0.0949,
        "contract_profile_latest_validation_split": "cumulative_to_2026-07_holdout",
        "contract_profile_latest_validation_split_average_net_R": -0.1627,
        "contract_profile_latest_validation_split_average_net_R_without_largest_win": -0.1627,
        "family_validation_scored_count": 196,
        "family_unique_ticker_count": 7,
        "family_unique_signal_date_count": 52,
        "family_expected_R": -0.1565,
        "family_avg_R_without_largest_win": -0.1649,
        "family_latest_validation_split_average_net_R": -0.1403,
        "family_latest_validation_split_average_net_R_without_largest_win": -0.1403,
        "family_member_scored_count": 20,
        "family_member_avg_R": -0.1,
        "family_member_profit_factor": 0.5,
        "family_member_avg_R_without_largest_win": -0.12,
        "block_reasons": ["FAMILY_MEMBER_HISTORY_NEGATIVE"],
    }
    lines = []

    append_current_pattern_members(lines, primary, ([primary],))
    report = "\n".join(lines)

    assert "13 observations across 10 signal dates and 4 holdouts" in report
    assert "Broad family: 196 observations across 7 tickers and 52 signal dates" in report
    assert "13 observations across 7 tickers and 52 signal dates" not in report


def test_current_pattern_report_counts_pattern_only_members():
    primary = {
        "active_pattern_id": "EDGE__BEARISH__LONG_OPTION__TECHNOLOGY",
        "ticker": "META",
        "direction": "bearish",
        "status": "AVOID",
        "classification": "AVOID",
        "directional_confidence_tier": "PROVEN_DIRECTIONAL",
        "contract_profile": "LONG_OPTION__DTE_31_45__FAR_OTM",
        "block_reasons": ["CONTRACT_PROFILE_NOT_VALIDATED"],
    }
    lines = []

    append_current_pattern_members(lines, primary, ([primary],))
    report = "\n".join(lines)

    assert "0 TRADE, 0 WATCH, 0 AVOID, 1 PATTERN ONLY" in report


def test_directional_pattern_ranking_and_report_primary_are_consistent():
    pattern_only = {
        "status": "AVOID",
        "classification": "AVOID",
        "ticker": "HIGH",
        "direction": "bullish",
        "directional_confidence_tier": "PROVEN_DIRECTIONAL",
        "directional_probability_score_pct": 64.0,
        "directional_win_rate_pct": 70.0,
        "strategy_kind": "long_option",
        "strategy_type": "Long Call Debit",
        "lead_option_symbol": "HIGH260821C00100000",
        "expiry": "2026-08-21",
        "option_type": "call",
        "strike": 100,
        "entry_range": "4.90-5.00",
        "contract_profile_validated": "no",
        "expected_R": None,
        "block_reasons": ["CONTRACT_PROFILE_NOT_VALIDATED"],
    }
    review = {
        "status": "TRADE_REVIEW",
        "classification": "WATCH",
        "ticker": "LOW",
        "direction": "bullish",
        "probability_score": 40.0,
        "success_probability_pct": 55.0,
        "strategy_kind": "long_option",
        "strategy_type": "Long Call Debit",
        "lead_option_symbol": "LOW260821C00100000",
        "expiry": "2026-08-21",
        "option_type": "call",
        "strike": 100,
        "entry_range": "4.90-5.00",
        "contract_profile_validated": "yes",
        "expected_R": 0.10,
        "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN"],
    }

    recommendations = build_pattern_recommendations([], [review], [pattern_only, review])
    lines = []
    primary = append_compact_action_board(lines, "2026-07-17", [], recommendations, [review], [])

    assert recommendations[0]["ticker"] == "HIGH"
    assert primary is recommendations[0]
    assert "Best current proven directional pattern: HIGH BULLISH" in "\n".join(lines)
    assert validate_artifact_consistency(
        [pattern_only, review], [], [review], recommendations, "TRADE_REVIEW"
    ) == []


def test_candidate_shortlist_prioritizes_review_tickets_over_pattern_only_rows():
    pattern_only = {
        "status": "AVOID",
        "classification": "AVOID",
        "ticker": "NEE",
        "direction": "bearish",
        "directional_confidence_tier": "PROVEN_DIRECTIONAL",
        "strategy_type": "Bear Call Credit Spread",
        "strike_rates": "SELL 90 / BUY 95",
        "expiration_date": "2026-08-21",
        "suggested_entry_debit_credit_range": "credit 1.49",
        "block_reasons": ["CONTRACT_PROFILE_NOT_VALIDATED"],
    }
    review = {
        "status": "TRADE_REVIEW",
        "classification": "WATCH",
        "ticker": "HOOD",
        "direction": "bullish",
        "strategy_type": "Long Call Debit",
        "strike_rates": "100",
        "expiration_date": "2026-08-21",
        "suggested_entry_debit_credit_range": "debit 4.90-5.00",
        "expected_R": 0.07,
        "probability_score": 43.9,
        "block_reasons": ["CALIBRATION_SCORE_MISSING_OR_WEAK"],
    }

    lines = []
    append_compact_candidate_shortlist(lines, None, [[pattern_only, review]], 5)
    rendered = "\n".join(lines)

    assert "WATCH: HOOD BULLISH" in rendered
    assert "PATTERN ONLY: NEE" not in rendered


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


def test_cross_ticker_crude_oil_flow_becomes_a_validated_theme_signal():
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
        "USO": {
            "ticker": "USO",
            "close": 121.4,
            "flow_total_premium": 95_705_217.0,
            "flow_call_premium_share": 0.79,
            "flow_put_premium_share": 0.21,
            "flow_call_ask_premium_share": 0.78,
            "flow_put_ask_premium_share": 0.22,
            "flow_premium_bias": 0.58,
            "call_volume_ratio_30d": 0.97,
            "put_volume_ratio_30d": 0.71,
            "liquidity_score": 20.0,
        },
        "BNO": {
            "ticker": "BNO",
            "close": 47.6,
            "flow_total_premium": 6_467_776.0,
            "flow_call_premium_share": 0.92,
            "flow_put_premium_share": 0.08,
        },
        "UCO": {
            "ticker": "UCO",
            "close": 39.94,
            "flow_total_premium": 1_577_599.0,
            "flow_call_premium_share": 0.87,
            "flow_put_premium_share": 0.13,
        },
        "DBO": {
            "ticker": "DBO",
            "close": 20.17,
            "flow_total_premium": 66_081.0,
            "flow_call_premium_share": 0.93,
            "flow_put_premium_share": 0.07,
        },
    }
    snapshot = SnapshotStub(
        features,
        {
            ("USO", "bullish"): {
                "ticker": "USO",
                "direction": "bullish",
                "option_symbol": "USO260821C00125000",
                "option_type": "call",
                "expiry": "2026-08-21",
                "strike": 125.0,
                "dte": 37,
                "bid": 6.9,
                "ask": 7.55,
                "mid": 7.225,
                "spread_pct": 0.09,
                "volume": 3048,
                "open_interest": 4321,
                "quote_source": "bot_eod",
            }
        },
        {"regime": "MIXED"},
        signal_date="2026-07-15",
    )

    contexts = build_theme_flow_signal_contexts(snapshot)
    assert len(contexts) == 1
    assert contexts[0]["theme_key"] == "CRUDE_OIL"
    assert contexts[0]["theme_representative_ticker"] == "USO"
    assert contexts[0]["theme_component_count"] == 3
    assert contexts[0]["theme_aggregate_premium"] > 100_000_000
    assert contexts[0]["theme_call_premium_share"] == pytest.approx(0.7995, abs=0.001)

    signals = generate_signals_for_snapshot(snapshot, pattern_config, max_signals=1)
    theme_signal = next(row for row in signals if row["base_pattern_family"] == "THEME_FLOW_LEADER")
    assert theme_signal["ticker"] == "USO"
    assert theme_signal["direction"] == "bullish"
    assert theme_signal["theme_name"] == "Crude Oil"
    assert theme_signal["theme_component_tickers"] == "USO;BNO;UCO;DBO"
    assert theme_signal["entry_range"] == "6.90-7.55"


def test_cross_ticker_theme_signal_requires_multiple_material_components():
    snapshot = SnapshotStub(
        {
            "USO": {
                "ticker": "USO",
                "close": 121.4,
                "flow_total_premium": 105_000_000.0,
                "flow_call_premium_share": 0.80,
            }
        },
        signal_date="2026-07-15",
    )

    assert build_theme_flow_signal_contexts(snapshot) == []


def test_theme_flow_board_keeps_representative_ticket_and_blockers_visible():
    row = {
        "ticker": "USO",
        "direction": "bullish",
        "base_pattern_family": "THEME_FLOW_LEADER",
        "theme_key": "CRUDE_OIL",
        "theme_name": "Crude Oil",
        "theme_macro_label": "oil/Middle East",
        "theme_aggregate_premium": 103_816_673.0,
        "theme_call_premium_share": 0.7995,
        "theme_put_premium_share": 0.2005,
        "theme_component_count": 3,
        "theme_component_tickers": "USO;BNO;UCO;DBO",
        "theme_component_evidence": "USO $95.7M; BNO $6.5M; UCO $1.6M",
        "classification": "WATCH",
        "status": "TRADE_REVIEW",
        "strategy_kind": "long_option",
        "strategy_type": "Long Call Debit",
        "lead_option_symbol": "USO260821C00125000",
        "option_type": "call",
        "strike": 125.0,
        "expiry": "2026-08-21",
        "entry_bid": 6.9,
        "entry_ask": 7.55,
        "entry_range": "6.90-7.55",
        "max_risk_per_contract": 788.15,
        "success_probability_pct": 38.65,
        "probability_score": 31.88,
        "expected_R": 0.010815,
        "expected_R_per_day": 0.002163,
        "validation_scored_count": 49,
        "validation_profit_factor": 1.63,
        "block_reasons": ["CONFIDENCE_BAND_TOO_WEAK", "PATTERN_VALIDATION_NOT_PROVEN"],
    }

    leaders = build_theme_flow_leaders([row])
    assert leaders == [row]
    output = theme_flow_leader_output_row(leaders[0], 1)
    assert output["representative_ticker"] == "USO"
    assert output["trade_legs"] == "Buy 1 USO 2026-08-21 125C @ debit 6.90-7.55 limit"
    assert "confidence lower bound below auto floor" in output["why_not_auto_approved"]
    assert "auto requires PROVEN family" in output["why_not_auto_approved"]


def test_macro_geo_oil_theme_maps_direct_crude_and_energy_proxies():
    assert {
        "USO",
        "BNO",
        "DBO",
        "UCO",
        "SCO",
        "XLE",
        "XOP",
        "XOM",
        "CVX",
        "COP",
        "OXY",
        "MPC",
        "VLO",
        "SLB",
        "HAL",
    } <= set(THEME_MAP["oil/Middle East"]["tickers"])


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


def test_ticker_trend_overlay_cannot_promote_without_comparable_contract_history():
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
                "split": "cumulative_to_2026-05_holdout" if idx < 17 else "cumulative_to_2026-06_holdout",
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

    assert rows[0]["status"] == "AVOID"
    assert rows[0]["classification"] == "AVOID"
    assert rows[0]["ticker_trend_scope"] == "ticker_direction_strategy_pattern"
    assert rows[0]["success_probability_pct"] is None
    assert rows[0]["expected_R"] is None
    assert "CONTRACT_PROFILE_NOT_VALIDATED" in rows[0]["block_reasons"]
    trade_row = trade_output_row(rows[0])
    assert trade_row["contract_profile_validated"] == "no"
    assert "Not actionable" not in trade_row["why_actionable_now"]
    assert trade_row["validation_scored_count"] == rows[0]["validation_scored_count"]
    assert trade_row["beats_baselines_count"] == rows[0]["beats_baselines_count"]
    assert trade_row["beats_baselines_count"] == 0
    assert trade_row["auto_approval_gate_evidence"] == ""
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


def test_same_day_probability_adjustment_cannot_promote_weak_validated_probability():
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

    assert rows[0]["status"] == "AVOID"
    assert rows[0]["classification"] == "AVOID"
    assert rows[0]["approval_bridge"] == ""
    assert "CONTRACT_PROFILE_NOT_VALIDATED" in rows[0]["block_reasons"]
    assert "CALIBRATION_SCORE_MISSING_OR_WEAK" in rows[0]["block_reasons"]
    assert "CONFIDENCE_BAND_TOO_WEAK" in rows[0]["block_reasons"]
    assert rows[0]["calibrated_probability"] is None
    assert rows[0]["family_calibrated_probability"] == pytest.approx(45 / 99, abs=1e-6)
    trade_row = trade_output_row(rows[0])
    assert "approval_bridge" in trade_fieldnames()
    assert trade_row["approval_bridge"] == ""
    assert trade_row["auto_approval_gate_evidence"] == ""


def test_payoff_aware_probability_math_uses_strategy_breakeven():
    breakeven = payoff_breakeven_probability(1.675895, -0.623328)

    assert breakeven == pytest.approx(0.2711, abs=0.0001)
    assert probability_edge_over_breakeven_pct(0.4504, breakeven) == pytest.approx(17.93, abs=0.01)
    assert probability_edge_over_breakeven_pct(0.3982, breakeven) == pytest.approx(12.71, abs=0.01)
    assert probability_edge_over_breakeven_pct(0.2985009, breakeven) == pytest.approx(2.74, abs=0.01)
    assert payoff_breakeven_probability(None, -0.5) is None
    assert payoff_breakeven_probability(1.0, 0.1) is None


def test_payoff_aware_bridge_requires_strong_edges_and_positive_latest_split():
    row = {"confidence_tier": "PROVEN", "validation_scope": "FAMILY_REGIME"}
    args = {
        "row": row,
        "blockers": {"CALIBRATION_SCORE_MISSING_OR_WEAK", "CONFIDENCE_BAND_TOO_WEAK"},
        "expected_r": 0.182893,
        "expected_r_per_day": 0.036579,
        "validated_success_probability": 27 / 77,
        "confidence_lower": 0.2985009,
        "avg_win_r": 1.675895,
        "avg_loss_r": -0.623328,
        "validation_profit_factor": 1.451858,
        "validation_scored": 77,
        "baselines_beaten": 6,
        "latest_split_average_net_r": 0.7056,
        "family_member_stats": {
            "scored_count": 13,
            "validation_split_count": 4,
            "avg_r_without_largest_win": 0.1416,
            "latest_validation_split_average_net_r_without_largest_win": 1.327,
        },
        "risk_config": DEFAULT_RISK_CONFIG,
    }

    assert proven_payoff_aware_promotion_eligible(**args)
    assert not proven_payoff_aware_promotion_eligible(
        **{**args, "validated_success_probability": 0.32}
    )
    assert not proven_payoff_aware_promotion_eligible(**{**args, "latest_split_average_net_r": -0.01})
    assert not proven_payoff_aware_promotion_eligible(
        **{
            **args,
            "family_member_stats": {
                **args["family_member_stats"],
                "avg_r_without_largest_win": -0.01,
            },
        }
    )
    assert not proven_payoff_aware_promotion_eligible(
        **{**args, "row": {"confidence_tier": "PROVEN", "validation_scope": "FAMILY"}}
    )


def test_proven_payoff_aware_bridge_promotes_meta_like_asymmetric_edge():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__COMMUNICATION_SERVICES"
    daily_rows = [
        {
            "date": "2026-07-16",
            "ticker": "META",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "classification": "WATCH",
            "block_reasons": [],
            "confidence_tier": "RESEARCH_ONLY",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 800,
            "expiry": "2026-08-21",
            "lead_option_symbol": "META260821C00800000",
            "entry_bid": 9.60,
            "entry_ask": 9.85,
            "entry_range": "9.60-9.85",
            "bid_ask_spread_pct": 0.0254,
            "max_risk_per_contract": 998.15,
            "liquidity_volume": 2500,
            "liquidity_open_interest": 5000,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_OFF",
            "market_regime": "RISK_OFF",
            "success_probability_pct": 45.04,
            "failure_probability_pct": 54.96,
            "probability_score": 39.82,
            "trade_success_probability_pct": 45.04,
            "trade_failure_probability_pct": 54.96,
            "trade_probability_score": 39.82,
        }
    ]
    outcomes = []
    split_shapes = [
        ("cumulative_to_2026-04_holdout", 19, 6),
        ("cumulative_to_2026-05_holdout", 19, 6),
        ("cumulative_to_2026-06_holdout", 20, 7),
        ("cumulative_to_2026-07_holdout", 19, 8),
    ]
    for split, count, win_count in split_shapes:
        win_indexes = {round(i * (count - 1) / (win_count - 1)) for i in range(win_count)}
        month = split[19:26]
        for idx in range(count):
            win = idx in win_indexes
            outcomes.append(
                {
                    "split": split,
                    "sample": "VALIDATION",
                    "horizon": "5d",
                    "signal_date": f"{month}-{idx + 1:02d}",
                    "ticker": "META",
                    "direction": "bullish",
                    "strategy_kind": "long_option",
                    "pattern_family": family,
                    "market_regime": "RISK_OFF",
                    "status": "SCORED",
                    "net_r": 1.675895 if win else -0.623328,
                    "win": int(win),
                }
            )

    baseline_names = (
        "BASELINE_RANDOM_SAME_DATE_LIQUIDITY;BASELINE_SPY_QQQ_DIRECTIONAL;"
        "BASELINE_UNUSUAL_VOLUME_ONLY;BASELINE_NAIVE_UW_FLOW_ONLY;"
        "BASELINE_NAIVE_CATALYST_ONLY;BASELINE_FAMILY_PRIOR"
    )
    validation_bundle = empty_validation_bundle()
    validation_bundle["outcomes"] = outcomes
    validation_bundle["family_tiers"] = {family: {"confidence_tier": "RESEARCH_ONLY"}}
    validation_bundle["regime_family_tiers"] = {
        (family, "RISK_OFF"): {
            "confidence_tier": "PROVEN",
            "pattern_scope": "FAMILY_REGIME",
            "market_regime": "RISK_OFF",
            "regime_pattern_id": f"{family}__MARKET_REGIME_RISK_OFF",
            "validation_scored_count": 77,
            "validation_win_count": 27,
            "validation_success_probability": 27 / 77,
            "validation_probability_score": 0.2985009,
            "validation_average_net_r": 0.182893,
            "validation_profit_factor": 1.451858,
            "beats_baselines_count": 6,
            "baselines_beaten_names": baseline_names,
            "baselines_beaten_details": "six historical controls beaten after costs/slippage",
            "latest_validation_split": "cumulative_to_2026-07_holdout",
            "latest_validation_split_average_net_r": 0.344766,
        }
    }

    rows, controls = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {"risk_config": {"kill_switches": {}}},
    )

    candidate = rows[0]
    assert candidate["status"] == "AVOID"
    assert candidate["classification"] == "AVOID"
    assert candidate["approval_bridge"] == ""
    assert "CONTRACT_PROFILE_NOT_VALIDATED" in candidate["block_reasons"]
    assert candidate["expected_R"] is None
    assert candidate["family_expected_R"] == pytest.approx(0.182893, abs=0.0001)
    assert candidate["family_member_scored_count"] == 77
    assert candidate["family_member_payoff_support_passed"] == "yes"

    trade_row = trade_output_row(candidate)
    assert trade_row["approval_bridge"] == ""
    assert trade_row["contract_profile_validated"] == "no"
    assert trade_row["auto_approval_gate_evidence"] == ""

    board = build_decision_board_rows(rows, "2026-07-16", True, "NO_TRADE", {})
    assert auto_approved_goal_gate_failures(board[0], controls["risk_config"])


def test_conditional_trade_ticket_has_spot_target_and_activation_conditions():
    row = {
        "status": "TRADE_REVIEW",
        "classification": "WATCH",
        "ticker": "HOOD",
        "direction": "bullish",
        "strategy_kind": "long_option",
        "strategy_type": "Long Call Debit",
        "lead_option_symbol": "HOOD260918C00100000",
        "option_type": "call",
        "strike": 100.0,
        "expiry": "2026-09-18",
        "entry_bid": 3.95,
        "entry_ask": 4.00,
        "entry_range": "3.95-4.00",
        "max_risk_per_contract": 403.15,
        "underlying_price": 92.50,
        "stop_rule": "No price stop; thesis invalidation only.",
        "time_stop": "Exit after 40 trading sessions.",
        "block_reasons": ["MARKET_REGIME_CONFLICT", "PATTERN_VALIDATION_NOT_PROVEN"],
    }

    ticket = conditional_trade_output_row(row)
    board = build_decision_board_rows([row], "2026-07-31", True, "TRADE_REVIEW", {})

    assert ticket["conditional_status"] == "WAIT_FOR_ACTIVATION"
    assert ticket["color_code"] == "YELLOW"
    assert ticket["send_now"] == "no"
    assert ticket["spot_price"] == 92.50
    assert ticket["target_close_price"] == "SELL TO CLOSE >= 6.00 bid"
    assert "gain regime alignment" in ticket["activation_conditions"]
    assert board[0]["spot_price"] == 92.50


def test_negative_exact_member_history_vetoes_pooled_proven_family():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__COMMUNICATION_SERVICES"
    outcomes = []
    for month in ("06", "07"):
        for index in range(4):
            outcomes.append(
                {
                    "split": f"cumulative_to_2026-{month}_holdout",
                    "sample": "VALIDATION",
                    "horizon": "5d",
                    "signal_date": f"2026-{month}-{index + 1:02d}",
                    "ticker": "GOOGL",
                    "direction": "bullish",
                    "strategy_kind": "long_option",
                    "pattern_family": family,
                    "market_regime": "RISK_ON",
                    "status": "SCORED",
                    "net_r": -0.40,
                    "win": 0,
                }
            )
    validation_bundle = empty_validation_bundle()
    validation_bundle["outcomes"] = outcomes
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROVEN",
            "pattern_scope": "FAMILY",
            "validation_scored_count": 80,
            "validation_win_count": 48,
            "validation_success_probability": 0.60,
            "validation_probability_score": 0.55,
            "validation_average_net_r": 0.20,
            "validation_profit_factor": 2.0,
            "beats_baselines_count": 4,
            "baselines_beaten_names": "A;B;C;D",
            "baselines_beaten_details": "four frozen controls beaten",
        }
    }
    daily_rows = [
        {
            "ticker": "GOOGL",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "option_type": "call",
            "strike": 400,
            "expiry": "2026-08-21",
            "lead_option_symbol": "GOOGL260821C00400000",
            "entry_bid": 4.30,
            "entry_ask": 4.45,
            "bid_ask_spread_pct": 0.034,
            "max_risk_per_contract": 450.65,
            "liquidity_volume": 1000,
            "liquidity_open_interest": 2000,
            "quote_source": "bot_eod",
            "current_market_alignment": "RISK_ON",
            "market_regime": "RISK_ON",
            "success_probability_pct": 65.0,
            "probability_score": 60.0,
        }
    ]

    rows, _ = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    candidate = rows[0]
    assert candidate["status"] == "AVOID"
    assert candidate["classification"] == "AVOID"
    assert "FAMILY_MEMBER_HISTORY_NEGATIVE" in candidate["block_reasons"]
    assert "CONTRACT_PROFILE_NOT_VALIDATED" in candidate["block_reasons"]
    assert candidate["family_member_scored_count"] == 8
    assert candidate["family_member_avg_R"] == pytest.approx(-0.40)
    assert candidate["family_member_latest_validation_split_average_net_R"] == pytest.approx(-0.40)


def test_payoff_aware_fields_are_exported_once_in_specialized_csv_schemas():
    for fields in (trade_fieldnames(), scout_call_fieldnames(), pattern_recommendation_fieldnames()):
        assert fields.count("breakeven_success_probability_pct") == 1
        assert len(fields) == len(set(fields))


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
                "split": "cumulative_to_2026-05_holdout" if idx < 17 else "cumulative_to_2026-06_holdout",
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

    assert rows[0]["ticker_trend_scope"] == "ticker_direction_strategy_pattern"
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
                "split": "cumulative_to_2026-05_holdout" if idx < 9 else "cumulative_to_2026-06_holdout",
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

    assert rows[0]["ticker_trend_scope"] == ""
    assert rows[0]["validation_scored_count"] == 0
    assert rows[0]["family_validation_scored_count"] == 18
    assert "CONTRACT_PROFILE_NOT_VALIDATED" in rows[0]["block_reasons"]
    assert "LIMITED_OUT_OF_SAMPLE_SAMPLE" in rows[0]["block_reasons"]
    assert rows[0]["status"] != "AUTO_APPROVED"
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


def test_source_complete_dates_requires_a_recognized_market_flow_source(tmp_path):
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


def test_source_complete_dates_accepts_dark_pool_flow_with_core_option_sources(tmp_path):
    date_dir = tmp_path / "2026-02-02"
    date_dir.mkdir()
    for prefix in ["stock-screener-", "hot-chains-", "chain-oi-changes-", "dp-eod-report-"]:
        (date_dir / f"{prefix}2026-02-02.csv").write_text("ticker\nAAA\n", encoding="utf-8")

    completeness = source_completeness_for_date(tmp_path, "2026-02-02")

    assert completeness["source_complete"] is True
    assert completeness["present_sources"]["dp_eod"]
    assert source_complete_dates(tmp_path) == ["2026-02-02"]


def test_build_snapshot_aggregates_dark_pool_flow_without_treating_it_as_option_flow(tmp_path):
    date_dir = tmp_path / "2026-02-02"
    date_dir.mkdir()
    header = "ticker,price,nbbo_ask,nbbo_bid,size,premium,date,canceled\n"
    rows = (
        "XYZ,101.0,100.5,100.0,1000,1000000,2026-02-02,f\n"
        "XYZ,99.5,100.5,100.0,500,500000,2026-02-02,f\n"
    )
    with zipfile.ZipFile(date_dir / "dp-eod-report-2026-02-02.zip", "w") as zf:
        zf.writestr("dp-eod-report-2026-02-02.csv", header + rows)

    snapshot = build_daily_snapshot(
        tmp_path,
        "2026-02-02",
        {
            "max_chain_rows_per_day": 0,
            "max_flow_file_mb": 100.0,
            "bot_eod_cache_dir": str(tmp_path / "cache"),
        },
    )
    xyz = snapshot.features["XYZ"]

    assert "dp_eod" in xyz["source_flags"]
    assert "option_trades" not in xyz["source_flags"]
    assert xyz["dp_total_premium"] == 1_500_000
    assert xyz["dp_above_ask_premium_share"] == pytest.approx(2 / 3)
    assert xyz["dp_below_bid_premium_share"] == pytest.approx(1 / 3)
    assert snapshot.counts["dp_eod_rows"] == 2


def test_dark_pool_pattern_is_generated_even_when_direction_conflicts_with_market_regime():
    quote = {
        "ticker": "XYZ",
        "direction": "bullish",
        "option_symbol": "XYZ260320C00105000",
        "option_type": "call",
        "expiry": "2026-03-20",
        "strike": 105.0,
        "dte": 18,
        "bid": 3.9,
        "ask": 4.0,
        "mid": 3.95,
        "volume": 2000,
        "open_interest": 3000,
        "premium": 800_000.0,
        "spread_pct": 0.025,
        "quote_source": "hot_chains",
    }
    snapshot = SnapshotStub(
        {
            "XYZ": {
                "date": "2026-03-02",
                "ticker": "XYZ",
                "close": 100.0,
                "sector": "Technology",
                "source_flags": {"stock_screener", "hot_chains", "chain_oi", "dp_eod"},
                "dp_total_premium": 25_000_000.0,
                "dp_directional_premium_coverage": 0.80,
                "dp_above_ask_premium_share": 0.75,
                "dp_below_bid_premium_share": 0.25,
                "call_volume_ratio_30d": 0.8,
                "put_volume_ratio_30d": 0.8,
                "liquidity_score": 20.0,
            }
        },
        {("XYZ", "bullish"): quote},
        {"regime": "RISK_OFF"},
        signal_date="2026-03-02",
        option_quotes={quote["option_symbol"]: quote},
    )
    pattern_config = {
        "min_call_volume_ratio": 1.35,
        "min_put_volume_ratio": 1.35,
        "min_hot_premium": 250_000.0,
        "min_oi_diff": 5_000.0,
        "max_spread_pct": 0.35,
        "high_iv": 0.75,
        "min_liquidity_score": 8.0,
        "min_ask_side_ratio": 0.52,
        "max_event_dte_without_event_strategy": 2,
        "min_dark_pool_premium": 5_000_000.0,
        "min_dark_pool_directional_share": 0.60,
        "min_dark_pool_directional_coverage": 0.25,
    }

    signals = generate_signals_for_snapshot(snapshot, pattern_config, max_signals=10)
    signal = next(row for row in signals if row["base_pattern_family"] == "DARK_POOL_PRESSURE")

    assert signal["direction"] == "bullish"
    assert signal["dp_total_premium"] == 25_000_000.0
    assert "MARKET_REGIME_CONFLICT" in signal["block_reasons"]


def test_source_complete_dates_accepts_dated_whale_fallback(tmp_path):
    date_dir = tmp_path / "2026-01-08"
    date_dir.mkdir()
    for prefix in ["stock-screener-", "hot-chains-", "chain-oi-changes-"]:
        (date_dir / f"{prefix}2026-01-08.csv").write_text("ticker\nAAA\n", encoding="utf-8")
    (date_dir / "whale_trades_filtered-2026-01-08.csv").write_text(
        "executed_at,underlying_symbol\n",
        encoding="utf-8",
    )

    assert source_complete_dates(tmp_path) == ["2026-01-08"]


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
    assert "establish positive out-of-sample expectancy" in output[0]["promotion_needed"]
    assert "upgrade pattern family to PROVEN" in output[0]["promotion_needed"]
    assert "upgraded to proven" not in output[0]["promotion_needed"]
    assert output[1]["review_status"] == "MACRO_CONFLICT_REVIEW"
    assert "gain regime alignment" in output[1]["promotion_needed"]


def test_target_ready_excludes_avoid_even_when_ticket_is_complete():
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

    assert candidates == []
    assert daily_trade_decision([], [], [row], candidates) == "NO_TRADE"
    output = target_ready_output_row(row)
    assert output["target_ready_status"] == "BLOCKED_NOT_READY"
    assert output["send_now"] == "no"
    assert output["live_recheck_required"] == "yes"
    assert output["target_debit_credit"] == "debit 7.50-7.70"
    assert output["trade_legs"] == "Buy 1 AAPL 2026-06-19 210C @ debit 7.50-7.70 limit"
    assert output["order_entry_missing_fields"] == ""
    assert "risk_limit_labeled_not_hidden" in output["risk_label"]
    assert "max risk $770.65 exceeds configured trade limit" in output["why_not_send_now"]
    consistency_errors = validate_artifact_consistency(
        [row],
        [],
        [],
        [],
        "NO_TRADE",
        target_ready=[row],
    )
    assert "AAPL: AVOID candidate cannot be labeled target ready" in consistency_errors


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


def test_complete_positive_ticket_is_not_trade_review_without_family_proof():
    family = "CATALYST_FLOW_LEADER__BULLISH__LONG_OPTION__TECHNOLOGY"
    daily_rows = [
        {
            "date": "2026-07-31",
            "classification": "WATCH",
            "ticker": "NVDA",
            "direction": "bullish",
            "pattern_family": family,
            "base_pattern_family": "CATALYST_FLOW_LEADER",
            "confidence_tier": "PROMISING",
            "pattern_score": 10.0,
            "block_reasons": ["PATTERN_VALIDATION_NOT_PROVEN", "MARKET_REGIME_CONFLICT"],
            "strategy_kind": "long_option",
            "strategy_type": "Long Call Debit",
            "lead_option_symbol": "NVDA260918C00200000",
            "expiry": "2026-09-18",
            "option_type": "call",
            "strike": 200.0,
            "entry_bid": 13.55,
            "entry_ask": 13.65,
            "entry_range": "13.55-13.65",
            "bid_ask_spread_pct": 0.01,
            "max_risk_per_contract": 1370.65,
            "liquidity_volume": 1000,
            "liquidity_open_interest": 1000,
            "dte": 35,
            "quote_source": "hot_chains",
        }
    ]
    validation_bundle = empty_validation_bundle()
    validation_bundle["family_tiers"] = {
        family: {
            "confidence_tier": "PROMISING",
            "validation_scored_count": 100,
            "validation_win_count": 70,
            "validation_success_probability": 0.70,
            "validation_probability_score": 0.65,
            "validation_average_net_r": 0.10,
            "validation_profit_factor": 1.40,
            "beats_baselines_count": 4,
        }
    }

    rows, _ = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    assert rows[0]["status"] == "AVOID"
    assert rows[0]["classification"] == "AVOID"
    assert "PATTERN_VALIDATION_NOT_PROVEN" in rows[0]["block_reasons"]


def test_validated_regime_edge_does_not_bypass_missing_contract_profile():
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
            "split": "cumulative_to_2026-05_holdout" if idx < 18 else "cumulative_to_2026-06_holdout",
            "sample": "VALIDATION",
            "horizon": "5d",
            "signal_date": "2026-05-01",
            "pattern_family": family,
            "market_regime": "RISK_OFF",
            "status": "SCORED",
            "net_r": 0.45 if (idx % 8) < 5 else -0.18,
            "win": int((idx % 8) < 5),
        }
        for idx in range(35)
    ]

    rows, _ = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        {"source_complete": True},
        {},
    )

    assert rows[0]["status"] == "AVOID"
    assert rows[0]["classification"] == "AVOID"
    assert rows[0]["edge_review_reason"] == ""
    assert "MARKET_REGIME_CONFLICT" in rows[0]["block_reasons"]
    assert "CONTRACT_PROFILE_NOT_VALIDATED" in rows[0]["block_reasons"]
    recommendations = build_pattern_recommendations([], rows)
    assert recommendations == []


def test_validated_regime_edge_bridge_requires_comparable_contract_profile():
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
                "split": "cumulative_to_2026-05_holdout" if idx < 13 else "cumulative_to_2026-06_holdout",
                "sample": "VALIDATION",
                "horizon": "5d",
                    "signal_date": (
                        f"2026-05-{idx + 1:02d}" if idx < 13 else f"2026-06-{idx - 12:02d}"
                    ),
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

    assert rows[0]["status"] == "AVOID"
    assert rows[0]["classification"] == "AVOID"
    assert rows[0]["approval_bridge"] == ""
    assert rows[0]["regime_edge_review_passed"] == "no"
    assert "MARKET_REGIME_CONFLICT" in rows[0]["block_reasons"]
    assert "CONTRACT_PROFILE_NOT_VALIDATED" in rows[0]["block_reasons"]
    assert rows[0]["auto_min_scored_outcomes"] == controls["risk_config"]["min_oos_scored_outcomes"]
    trade_row = trade_output_row(rows[0])
    assert trade_row["approval_bridge"] == ""
    assert trade_row["contract_profile_validated"] == "no"


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
