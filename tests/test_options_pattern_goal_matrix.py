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
