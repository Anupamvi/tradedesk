import argparse
from types import SimpleNamespace
from pathlib import Path

from scripts import options_pattern_goal_matrix as matrix


def test_resolve_dates_uses_all_source_complete_scope_with_bounds(monkeypatch, tmp_path):
    monkeypatch.setattr(
        matrix,
        "strict_source_complete_dates",
        lambda base_dir: ["2026-01-02", "2026-01-05", "2026-01-05", "2026-01-12"],
    )
    args = argparse.Namespace(
        all_source_complete=True,
        dates=["2026-05-15"],
        from_date="2026-01-03",
        to_date="2026-01-10",
    )

    dates, scope = matrix.resolve_dates(args, tmp_path)

    assert dates == ["2026-01-05"]
    assert scope == "all_source_complete;from=2026-01-03,to=2026-01-10;date_count=1"


def test_resolve_dates_preserves_requested_date_order_and_dedupes(tmp_path):
    args = argparse.Namespace(
        all_source_complete=False,
        dates=["2026-05-20", "2026-05-18", "2026-05-20", "2026-05-22"],
        from_date="2026-05-18",
        to_date="2026-05-20",
    )

    dates, scope = matrix.resolve_dates(args, Path(tmp_path))

    assert dates == ["2026-05-20", "2026-05-18"]
    assert scope == "requested_dates;from=2026-05-18,to=2026-05-20;date_count=2"


def test_matrix_status_names_full_source_complete_scope():
    assert matrix.matrix_status([], [], "all_source_complete;unbounded;date_count=10") == "PASS_SOURCE_COMPLETE_SCOPE"
    assert matrix.matrix_status([], [], "requested_dates;unbounded;date_count=8") == "PASS_DAILY_NOT_GLOBAL"


def test_run_pipeline_uses_shared_bot_eod_cache(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)
    out_dir = tmp_path / "run"
    cache_dir = tmp_path / "runs" / "_cache" / "bot_eod"
    rc = matrix.run_pipeline("pythonX", tmp_path / "base", "2026-05-29", out_dir, cache_dir)

    assert rc == 0
    assert captured["cwd"] == tmp_path / "base"
    cache_flag_index = captured["cmd"].index("--bot-eod-cache-dir")
    assert captured["cmd"][cache_flag_index + 1] == str(cache_dir)


def test_main_uses_explicit_bot_eod_cache_dir_for_run_missing(monkeypatch, tmp_path):
    captured = {}
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    matrix_dir = tmp_path / "matrix"
    runs_root = tmp_path / "runs"
    cache_dir = tmp_path / "warm_cache"

    monkeypatch.setattr(matrix, "resolve_dates", lambda args, base: (["2026-05-29"], "requested_dates;date_count=1"))
    monkeypatch.setattr(matrix, "has_goal_artifacts", lambda out_dir: False)
    monkeypatch.setattr(matrix, "newest_existing_goal_run", lambda root, date: None)

    def fake_run_pipeline(python, base, date, out_dir, bot_eod_cache_dir):
        captured["bot_eod_cache_dir"] = bot_eod_cache_dir
        return 1

    monkeypatch.setattr(matrix, "run_pipeline", fake_run_pipeline)

    rc = matrix.main(
        [
            "--base-dir",
            str(base_dir),
            "--dates",
            "2026-05-29",
            "--runs-root",
            str(runs_root),
            "--matrix-dir",
            str(matrix_dir),
            "--bot-eod-cache-dir",
            str(cache_dir),
            "--run-missing",
        ]
    )

    assert rc == 1
    assert captured["bot_eod_cache_dir"] == cache_dir.resolve()


def test_directional_scenario_gate_fails_when_bearish_source_disappears():
    status, evidence, failed, warned = matrix.directional_scenario_gate(
        {
            "source_bearish": 3,
            "candidate_bearish": 0,
            "candidate_bearish_put_or_spread": 0,
            "trend_bearish": 0,
            "trend_bearish_put_or_spread": 0,
            "auto_bearish": 0,
            "auto_bearish_put_or_spread": 0,
        }
    )

    assert status == "FAIL"
    assert failed == ["directional_scenario_candidate_surface_missing"]
    assert warned == []
    assert "source_bearish=3" in evidence


def test_directional_scenario_gate_warns_when_put_spread_trend_edge_is_missing():
    status, evidence, failed, warned = matrix.directional_scenario_gate(
        {
            "source_bearish": matrix.MIN_BEARISH_SOURCE_ROWS_FOR_SCENARIO_WARN,
            "candidate_bearish": 12,
            "candidate_bearish_put_or_spread": 12,
            "trend_bearish": 2,
            "trend_bearish_put_or_spread": 0,
            "auto_bearish": 0,
            "auto_bearish_put_or_spread": 0,
        }
    )

    assert status == "WARN"
    assert failed == []
    assert warned == ["directional_scenario_put_spread_trend_edge_missing"]
    assert "trend_bearish_put_or_spread=0" in evidence


def test_directional_scenario_gate_labels_missing_trend_as_insufficient_sample_when_scored_below_threshold():
    status, evidence, failed, warned = matrix.directional_scenario_gate(
        {
            "source_bearish": matrix.MIN_BEARISH_SOURCE_ROWS_FOR_SCENARIO_WARN,
            "candidate_bearish": 12,
            "candidate_bearish_put_or_spread": 12,
            "trend_bearish": 0,
            "trend_bearish_put_or_spread": 0,
            "trend_total": 17,
            "validation_gate_bearish_groups": 3,
            "validation_gate_bearish_max_scored": matrix.MIN_TICKER_TREND_EDGE_SCORED - 1,
            "validation_gate_bearish_groups_ge_edge_min": 0,
            "auto_bearish": 0,
            "auto_bearish_put_or_spread": 0,
        }
    )

    assert status == "WARN"
    assert failed == []
    assert warned == ["directional_scenario_trend_edge_insufficient_sample"]
    assert "validation_gate_bearish_max_scored=7" in evidence


def test_directional_scenario_gate_does_not_warn_before_trend_history_exists():
    status, evidence, failed, warned = matrix.directional_scenario_gate(
        {
            "source_bearish": matrix.MIN_BEARISH_SOURCE_ROWS_FOR_SCENARIO_WARN,
            "candidate_bearish": 12,
            "candidate_bearish_put_or_spread": 12,
            "trend_bearish": 0,
            "trend_bearish_put_or_spread": 0,
            "trend_total": 0,
            "auto_bearish": 0,
            "auto_bearish_put_or_spread": 0,
        }
    )

    assert status == "PASS"
    assert failed == []
    assert warned == []
    assert "trend_total=0" in evidence


def test_directional_scenario_metrics_counts_bearish_puts_and_credit_spreads(tmp_path):
    out_dir = tmp_path
    (out_dir / "source_ticker_coverage.csv").write_text(
        "ticker,direction\nA,bearish\nB,bullish\n",
        encoding="utf-8",
    )
    (out_dir / "trade_review_candidates.csv").write_text(
        "ticker,direction,call_or_put,strategy\nA,bearish,PUT,Long Put Debit\nC,bullish,CALL,Long Call Debit\n",
        encoding="utf-8",
    )
    (out_dir / "blocked_candidates.csv").write_text(
        "ticker,direction,call_or_put,strategy\nD,bearish,CALL / CALL,Call Credit Spread\n",
        encoding="utf-8",
    )
    (out_dir / "ticker_trend_edges.csv").write_text(
        "ticker,direction,call_or_put,strategy_kind\nA,bearish,PUT,LONG_OPTION\nD,bearish,CALL / CALL,CREDIT_SPREAD\n",
        encoding="utf-8",
    )

    metrics = matrix.directional_scenario_metrics(out_dir)

    assert metrics["source_bearish"] == 1
    assert metrics["candidate_bearish"] == 2
    assert metrics["candidate_bearish_put_or_spread"] == 2
    assert metrics["trend_bearish"] == 2
    assert metrics["trend_bearish_put_or_spread"] == 2
    assert metrics["trend_total"] == 2


def test_directional_scenario_metrics_infers_bearish_long_option_as_put(tmp_path):
    out_dir = tmp_path
    (out_dir / "source_ticker_coverage.csv").write_text("ticker,direction\nIWM,bearish\n", encoding="utf-8")
    (out_dir / "blocked_candidates.csv").write_text(
        "ticker,direction,call_or_put,strategy\nIWM,bearish,PUT,Long Put Debit\n",
        encoding="utf-8",
    )
    (out_dir / "ticker_trend_edges.csv").write_text(
        "ticker,direction,strategy_kind,trade_ready_trend\nIWM,bearish,long_option,no\n",
        encoding="utf-8",
    )

    metrics = matrix.directional_scenario_metrics(out_dir)

    assert metrics["trend_bearish"] == 1
    assert metrics["trend_bearish_put_or_spread"] == 1


def test_validation_gate_bearish_scores_counts_only_gate_scored_rows(tmp_path):
    path = tmp_path / "validation_details.csv"
    path.write_text(
        "\n".join(
            [
                "split,sample,horizon,ticker,direction,strategy_kind,status,net_r",
                "cumulative_to_2026-04_holdout,VALIDATION,5d,IWM,bearish,long_option,SCORED,-0.2",
                "cumulative_to_2026-04_holdout,VALIDATION,5d,IWM,bearish,long_option,SCORED,0.3",
                "cumulative_to_2026-04_holdout,VALIDATION,3d,IWM,bearish,long_option,SCORED,0.3",
                "cumulative_to_2026-04_holdout,TRAIN,5d,IWM,bearish,long_option,SCORED,0.3",
                "month_2026-04_holdout,VALIDATION,5d,IWM,bearish,long_option,SCORED,0.3",
                "cumulative_to_2026-04_holdout,VALIDATION,5d,SPY,bullish,long_option,SCORED,0.3",
                "cumulative_to_2026-04_holdout,VALIDATION,5d,AAPL,bearish,credit_spread,PARTIAL,",
            ]
        ),
        encoding="utf-8",
    )

    assert matrix.validation_gate_bearish_scores(path) == {("IWM", "long_option"): 2}


def test_portfolio_acceptance_summary_aggregates_auto_trades(tmp_path):
    run_dir = tmp_path / "2026-05-29_run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        '{"risk_config":{"min_ticker_trend_expected_r":0.15,"min_expected_r_per_day":0.0,'
        '"min_ticker_trend_profit_factor":1.5,"min_ticker_trend_scored_outcomes":20,'
        '"min_baselines_beaten":2,"min_ticker_trend_probability_score":0.42,'
        '"min_ticker_trend_win_rate":0.55}}',
        encoding="utf-8",
    )
    (run_dir / "actionable_trades.csv").write_text(
        "\n".join(
            [
                "ticker,direction,strategy,buy_or_sell,call_or_put,strike_rates,expiration_date,"
                "suggested_entry_debit_credit_range,trade_legs,max_risk_per_contract,probability_score,"
                "success_probability_pct,expected_R,expected_R_per_day,validation_profit_factor,"
                "validation_scored_count,beats_baselines_count,baselines_beaten_names,baselines_beaten_details,"
                "ticker_trend_scope,ticker_trend_scored_count,ticker_trend_avg_R,ticker_trend_profit_factor,"
                "ticker_trend_probability_score_pct,ticker_trend_win_rate_pct,calibrated_probability",
                "AMD,bullish,Long Call Debit,BUY,CALL,550,2026-06-05,debit 9.05-9.25,"
                "Buy 1 AMD 2026-06-05 550C @ debit 9.05-9.25 limit,925.65,55.83,66.67,"
                "1.97,0.39,9.73,21,6,BASELINE_A;BASELINE_B,details,ticker_direction_strategy,"
                "21,1.97,9.73,55.83,66.67,0.6667",
            ]
        ),
        encoding="utf-8",
    )

    trade_rows = matrix.build_portfolio_trade_rows([{"date": "2026-05-29", "run_dir": str(run_dir)}])
    summary = matrix.build_portfolio_acceptance_summary([{"date": "2026-05-29"}], trade_rows)

    assert trade_rows[0]["portfolio_gate_status"] == "PASS"
    assert summary["portfolio_status"] == "PASS_WITH_WARNINGS"
    assert summary["trade_count"] == 1
    assert summary["gate_fail_trade_count"] == 0
    assert summary["avg_expected_R"] == 1.97
    assert "AUTO_DIRECTION_CONCENTRATION_NO_BEARISH" in summary["warnings"]


def test_portfolio_acceptance_summary_fails_bad_auto_trade(tmp_path):
    run_dir = tmp_path / "2026-05-29_run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text('{"risk_config":{"min_baselines_beaten":2}}', encoding="utf-8")
    (run_dir / "actionable_trades.csv").write_text(
        "\n".join(
            [
                "ticker,direction,strategy,buy_or_sell,call_or_put,strike_rates,expiration_date,"
                "suggested_entry_debit_credit_range,trade_legs,max_risk_per_contract,probability_score,"
                "success_probability_pct,expected_R,expected_R_per_day,validation_profit_factor,"
                "validation_scored_count,beats_baselines_count,baselines_beaten_names,baselines_beaten_details",
                "BAD,bullish,Long Call Debit,BUY,CALL,10,2026-06-05,debit 1.00-1.05,"
                "Buy 1 BAD 2026-06-05 10C @ debit 1.00-1.05 limit,105,20,30,-0.5,-0.1,0.5,"
                "3,0,,",
            ]
        ),
        encoding="utf-8",
    )

    trade_rows = matrix.build_portfolio_trade_rows([{"date": "2026-05-29", "run_dir": str(run_dir)}])
    summary = matrix.build_portfolio_acceptance_summary([{"date": "2026-05-29"}], trade_rows)

    assert trade_rows[0]["portfolio_gate_status"] == "FAIL"
    assert "expected_R" in trade_rows[0]["portfolio_gate_failures"]
    assert summary["portfolio_status"] == "FAIL"
    assert summary["gate_fail_trade_count"] == 1


def test_scenario_no_edge_rows_aggregate_bearish_put_and_spread_lanes(tmp_path):
    run_dir = tmp_path / "2026-05-20_run"
    run_dir.mkdir()
    (run_dir / "trade_review_candidates.csv").write_text(
        "\n".join(
            [
                "ticker,direction,strategy,call_or_put,expected_R,expected_R_per_day,probability_score,"
                "success_probability_pct,validation_profit_factor,validation_scored_count,beats_baselines_count,"
                "block_reasons,trade_legs",
                "IWM,bearish,Long Put Debit,PUT,-0.20,-0.04,35,45,0.5,12,1,"
                "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS;DOES_NOT_BEAT_TWO_BASELINES,"
                "Buy 1 IWM 2026-06-18 260P @ debit 3.00-3.10 limit",
                "SPY,bearish,Long Put Debit,PUT,0.10,0.02,42,55,1.2,18,2,"
                "LIMITED_OUT_OF_SAMPLE_SAMPLE,"
                "Buy 1 SPY 2026-06-18 700P @ debit 4.00-4.10 limit",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "blocked_candidates.csv").write_text(
        "\n".join(
            [
                "ticker,direction,strategy,call_or_put,expected_R,expected_R_per_day,probability_score,"
                "success_probability_pct,validation_profit_factor,validation_scored_count,beats_baselines_count,"
                "block_reasons,trade_legs",
                "MSTR,bearish,Bear Call Credit Spread,CALL / CALL,-0.30,-0.06,22,37,0.3,8,2,"
                "PROFIT_FACTOR_BELOW_AUTO_APPROVAL;EXPECTED_R_NOT_POSITIVE_AFTER_COSTS,"
                "Sell 1 MSTR 2026-06-18 500C / Buy 1 MSTR 2026-06-18 520C @ net credit 4.00 limit",
            ]
        ),
        encoding="utf-8",
    )

    rows = matrix.build_scenario_no_edge_rows([{"date": "2026-05-20", "run_dir": str(run_dir)}])
    review_put = next(row for row in rows if row["surface_status"] == "REVIEW" and row["call_or_put"] == "PUT")
    avoid_spread = next(row for row in rows if row["surface_status"] == "AVOID")

    assert review_put["candidate_count"] == 2
    assert review_put["distinct_ticker_count"] == 2
    assert review_put["positive_expected_R_count"] == 1
    assert "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS:1" in review_put["top_blockers"]
    assert "SPY" in review_put["top_examples"]
    assert avoid_spread["strategy"] == "Bear Call Credit Spread"
    assert "PROFIT_FACTOR_BELOW_AUTO_APPROVAL:1" in avoid_spread["top_blockers"]


def test_directional_edge_matrix_rows_aggregate_daily_diagnostics(tmp_path):
    run_a = tmp_path / "2026-05-27_run"
    run_b = tmp_path / "2026-05-28_run"
    run_a.mkdir()
    run_b.mkdir()
    header = (
        "surface_status,direction,strategy,call_or_put,primary_diagnosis,date_count,candidate_count,"
        "distinct_ticker_count,positive_expected_R_count,avg_expected_R,max_expected_R,"
        "avg_expected_R_per_day,avg_probability_score,avg_validation_profit_factor,"
        "avg_baselines_beaten,top_blockers,top_examples"
    )
    run_a.joinpath("directional_edge_diagnostics.csv").write_text(
        "\n".join(
            [
                header,
                "TRADE_REVIEW,bearish,Long Put Debit,PUT,INSUFFICIENT_VALIDATED_SAMPLE,1,2,2,1,0.50,2.0,0.10,40,999,6,LIMITED_OUT_OF_SAMPLE_SAMPLE:2,CAR ER=2.0 score=40% legs=Buy 1 CAR 160P",
            ]
        ),
        encoding="utf-8",
    )
    run_b.joinpath("directional_edge_diagnostics.csv").write_text(
        "\n".join(
            [
                header,
                "TRADE_REVIEW,bearish,Long Put Debit,PUT,INSUFFICIENT_VALIDATED_SAMPLE,1,3,3,0,-0.10,0.2,-0.02,20,0.5,1,PATTERN_VALIDATION_NOT_PROVEN:3,IWM ER=-0.4 score=16% legs=Buy 1 IWM 276P",
                "AVOID,bearish,Bear Call Credit Spread,CALL / CALL,NEGATIVE_AVG_EXPECTANCY_AFTER_COSTS,1,4,4,0,-0.60,-0.01,-0.12,10,0.08,0,EXPECTED_R_NOT_POSITIVE_AFTER_COSTS:4,BSX ER=-0.01 legs=Sell 1 BSX 50C / Buy 1 BSX 52C",
            ]
        ),
        encoding="utf-8",
    )

    rows = matrix.build_directional_edge_matrix_rows(
        [
            {"date": "2026-05-27", "run_dir": str(run_a)},
            {"date": "2026-05-28", "run_dir": str(run_b)},
        ]
    )
    review_put = next(row for row in rows if row["surface_status"] == "TRADE_REVIEW")
    avoid_spread = next(row for row in rows if row["surface_status"] == "AVOID")

    assert review_put["date_count"] == 2
    assert review_put["candidate_count"] == 5
    assert review_put["positive_expected_R_count"] == 1
    assert round(review_put["avg_expected_R_weighted"], 2) == 0.14
    assert "LIMITED_OUT_OF_SAMPLE_SAMPLE:2" in review_put["top_blockers"]
    assert "PATTERN_VALIDATION_NOT_PROVEN:3" in review_put["top_blockers"]
    assert "CAR" in review_put["top_examples"]
    assert avoid_spread["primary_diagnosis"] == "NEGATIVE_AVG_EXPECTANCY_AFTER_COSTS"


def test_directional_no_edge_report_explains_zero_auto_direction():
    rows = [
        {
            "surface_status": "AUTO_APPROVED",
            "direction": "bullish",
            "strategy": "Long Call Debit",
            "call_or_put": "CALL",
            "primary_diagnosis": "TRADE_READY_EDGE",
            "candidate_count": 2,
            "positive_expected_R_count": 2,
            "avg_expected_R_weighted": 0.35,
            "max_expected_R": 0.7,
            "avg_probability_score_weighted": 60,
            "avg_validation_profit_factor_weighted": 2.0,
            "avg_baselines_beaten_weighted": 6,
            "top_blockers": "",
            "top_examples": "AAPL ER=0.7 legs=Buy 1 AAPL 310C",
        },
        {
            "surface_status": "TRADE_REVIEW",
            "direction": "bearish",
            "strategy": "Long Put Debit",
            "call_or_put": "PUT",
            "primary_diagnosis": "INSUFFICIENT_VALIDATED_SAMPLE",
            "candidate_count": 5,
            "positive_expected_R_count": 1,
            "avg_expected_R_weighted": 0.14,
            "max_expected_R": 2.0,
            "avg_probability_score_weighted": 25,
            "avg_validation_profit_factor_weighted": 300,
            "avg_baselines_beaten_weighted": 2,
            "top_blockers": "LIMITED_OUT_OF_SAMPLE_SAMPLE:3;PATTERN_VALIDATION_NOT_PROVEN:5",
            "top_examples": "CAR ER=2.0 legs=Buy 1 CAR 160P",
        },
        {
            "surface_status": "AVOID",
            "direction": "bearish",
            "strategy": "Bear Call Credit Spread",
            "call_or_put": "CALL / CALL",
            "primary_diagnosis": "NEGATIVE_AVG_EXPECTANCY_AFTER_COSTS",
            "candidate_count": 4,
            "positive_expected_R_count": 0,
            "avg_expected_R_weighted": -0.6,
            "max_expected_R": -0.01,
            "avg_probability_score_weighted": 10,
            "avg_validation_profit_factor_weighted": 0.08,
            "avg_baselines_beaten_weighted": 0,
            "top_blockers": "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS:4;PATTERN_VALIDATION_NOT_PROVEN:4",
            "top_examples": "BSX ER=-0.01 legs=Sell 1 BSX 50C / Buy 1 BSX 52C",
        },
    ]

    report_rows = matrix.build_directional_no_edge_report_rows(rows)

    assert len(report_rows) == 1
    bearish = report_rows[0]
    assert bearish["direction"] == "bearish"
    assert bearish["primary_no_edge_reason"] == "POSITIVE_REVIEW_EDGE_NOT_VALIDATED"
    assert bearish["auto_approved_candidate_count"] == 0
    assert bearish["non_auto_candidate_count"] == 9
    assert bearish["review_candidate_count"] == 5
    assert bearish["avoid_candidate_count"] == 4
    assert bearish["review_positive_expected_R_count"] == 1
    assert round(bearish["avg_expected_R_weighted"], 3) == -0.189
    assert "LIMITED_OUT_OF_SAMPLE_SAMPLE:3" in bearish["top_blockers"]
    assert "CAR" in bearish["top_examples"]

    markdown = matrix.render_directional_no_edge_report_markdown(
        report_rows,
        {"warnings": "AUTO_DIRECTION_CONCENTRATION_NO_BEARISH"},
    )
    assert "AUTO_DIRECTION_CONCENTRATION_NO_BEARISH" in markdown
    assert "POSITIVE_REVIEW_EDGE_NOT_VALIDATED" in markdown
