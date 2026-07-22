from __future__ import annotations

import datetime as dt

import pandas as pd

from codexuw.performance import (
    live_outcome_adjustment,
    load_live_outcome_performance,
    load_recent_performance,
    performance_min_score,
    performance_risk_multiplier,
    summarize_live_outcomes,
    summarize_recent_replay,
)


def test_summarize_recent_replay_marks_degrading_context() -> None:
    detail = pd.DataFrame(
        {
            "asof": ["2026-01-02", "2026-01-03", "2026-01-04"],
            "ticker": ["A", "B", "C"],
            "exact_evaluated": [True, True, True],
            "replay_guard_pass": [True, True, True],
            "exact_win": [False, True, False],
            "pnl_1x": [-100.0, 50.0, -80.0],
        }
    )

    context = summarize_recent_replay(detail, window=3)

    assert context["stance"] == "degrading"
    assert performance_risk_multiplier(context) == 0.75
    assert performance_min_score(context, 5.0) == 5.5


def test_summarize_recent_replay_marks_strong_context() -> None:
    detail = pd.DataFrame(
        {
            "asof": ["2026-01-02", "2026-01-03", "2026-01-04"],
            "ticker": ["A", "B", "C"],
            "exact_evaluated": [True, True, True],
            "replay_guard_pass": [True, True, True],
            "exact_win": [True, True, False],
            "pnl_1x": [90.0, 80.0, -40.0],
        }
    )

    context = summarize_recent_replay(detail, window=3)

    assert context["stance"] == "strong"
    assert context["profit_factor"] == 4.25
    assert context["max_drawdown_1x"] == -40.0
    assert performance_risk_multiplier(context) == 1.0
    assert performance_min_score(context, 5.0) == 5.0


def test_summarize_recent_replay_intersects_decision_and_guard() -> None:
    detail = pd.DataFrame(
        {
            "asof": ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
            "ticker": ["KEPT_WIN", "GUARD_FAIL", "DECISION_FAIL", "KEPT_LOSS"],
            "exact_evaluated": [True, True, True, True],
            "decision_pass": [True, True, False, True],
            "replay_guard_pass": [True, False, True, True],
            "exact_win": [True, True, True, False],
            "pnl_1x": [100.0, 1_000.0, 1_000.0, -40.0],
        }
    )

    context = summarize_recent_replay(detail, window=10)

    assert context["window"] == 2
    assert context["total_pnl_1x"] == 60.0
    assert context["profit_factor"] == 2.5
    assert context["selection_filter"] == "decision_pass+replay_guard_pass"


def test_summarize_recent_replay_excludes_credit_below_current_policy() -> None:
    detail = pd.DataFrame(
        {
            "asof": ["2026-01-02", "2026-01-03"],
            "ticker": ["OLD_POLICY", "CURRENT_POLICY"],
            "direction": ["Bull Put", "Bull Put"],
            "entry_credit_pct_width": [0.20, 0.25],
            "exact_evaluated": [True, True],
            "decision_pass": [True, True],
            "replay_guard_pass": [True, True],
            "exact_win": [True, True],
            "pnl_1x": [1_000.0, 50.0],
        }
    )

    context = summarize_recent_replay(detail, window=20)

    assert context["window"] == 1
    assert context["total_pnl_1x"] == 50.0
    assert context["policy_mismatch_excluded"] == 1


def test_load_recent_performance_prefers_decision_selected_replay(tmp_path) -> None:
    out_dir = tmp_path / "codexuw_audit_decision_select_2026"
    out_dir.mkdir()
    pd.DataFrame(
        {
            "asof": ["2026-01-02", "2026-01-03"],
            "ticker": ["A", "B"],
            "exact_evaluated": [True, True],
            "decision_pass": [True, True],
            "exact_win": [True, True],
            "pnl_1x": [50.0, 60.0],
        }
    ).to_csv(out_dir / "codexuw_replay_detail.csv", index=False)

    context = load_recent_performance(tmp_path, window=2)

    assert context["status"] == "ok"
    assert context["source"].endswith("codexuw_audit_decision_select_2026/codexuw_replay_detail.csv")
    assert context["stance"] == "strong"


def test_load_recent_performance_marks_old_replay_stale(tmp_path) -> None:
    out_dir = tmp_path / "codexuw_audit_decision_select_2026"
    out_dir.mkdir()
    pd.DataFrame(
        {
            "asof": ["2026-01-02", "2026-01-03"],
            "ticker": ["A", "B"],
            "exact_evaluated": [True, True],
            "decision_pass": [True, True],
            "exact_win": [True, True],
            "pnl_1x": [50.0, 60.0],
        }
    ).to_csv(out_dir / "codexuw_replay_detail.csv", index=False)

    context = load_recent_performance(tmp_path, asof=dt.date(2026, 3, 15))

    assert context["status"] == "stale"
    assert context["stance"] == "unavailable"
    assert context["age_days"] > context["max_age_days"]


def test_live_outcome_performance_flags_negative_setup_family(tmp_path) -> None:
    ledger = pd.DataFrame(
        [
            {"report_date": "2026-05-01", "strategy": "Bull Put Credit Spread", "direction": "Bull Put", "realized_pnl": -100.0},
            {"report_date": "2026-05-02", "strategy": "Bear Call Credit Spread", "direction": "Bear Call", "realized_pnl": -50.0},
            {"report_date": "2026-05-03", "strategy": "Bull Put Credit Spread", "direction": "Bull Put", "realized_pnl": 25.0},
        ]
    )

    summary = summarize_live_outcomes(ledger)
    adjustment = live_outcome_adjustment(summary, "Bull Put Credit Spread", "Bull Put")

    assert summary["family_summary"]["credit spreads"]["expectancy"] == "negative"
    assert adjustment["block_execute"] is True


def test_live_outcomes_exclude_hypothetical_pnl_on_unfilled_recommendations() -> None:
    ledger = pd.DataFrame(
        [
            {
                "report_date": "2026-05-01",
                "strategy": "Bull Put Credit Spread",
                "direction": "Bull Put",
                "outcome_status": "NOT_EXECUTED",
                "realized_pnl": 500.0,
            },
            {
                "report_date": "2026-05-02",
                "strategy": "Bull Put Credit Spread",
                "direction": "Bull Put",
                "outcome_status": "CLOSED",
                "actual_fill": 1.25,
                "realized_pnl": 50.0,
            },
        ]
    )

    summary = summarize_live_outcomes(ledger)

    assert summary["realized_outcome_count"] == 1
    assert summary["total_pnl"] == 50.0
    assert summary["excluded_nonexecuted_realized_rows"] == 1


def test_load_live_outcome_performance_prefers_recommendation_ledger(tmp_path) -> None:
    pd.DataFrame(
        [
            {"report_date": "2026-05-01", "strategy": "Bull Call Debit Spread", "direction": "Bull Call", "realized_pnl": 120.0}
        ]
    ).to_csv(tmp_path / "codexuw_recommendation_outcome_ledger.csv", index=False)

    context = load_live_outcome_performance(tmp_path)

    assert context["status"] == "ok"
    assert context["source"].endswith("codexuw_recommendation_outcome_ledger.csv")


def test_load_live_outcome_performance_prefers_v3_ledger_and_is_point_in_time(tmp_path) -> None:
    pd.DataFrame(
        [
            {"report_date": "2026-05-01", "strategy": "Bull Put Credit Spread", "direction": "Bull Put", "realized_pnl": 80.0},
            {"report_date": "2026-05-10", "strategy": "Bull Put Credit Spread", "direction": "Bull Put", "realized_pnl": 5_000.0},
        ]
    ).to_csv(tmp_path / "codexdaily_v3_recommendation_outcome_ledger.csv", index=False)
    pd.DataFrame(
        [{"report_date": "2026-05-01", "strategy": "Bull Call Debit Spread", "direction": "Bull Call", "realized_pnl": 999.0}]
    ).to_csv(tmp_path / "codexuw_recommendation_outcome_ledger.csv", index=False)

    context = load_live_outcome_performance(tmp_path, asof=dt.date(2026, 5, 5))

    assert context["status"] == "ok"
    assert context["source"].endswith("codexdaily_v3_recommendation_outcome_ledger.csv")
    assert context["realized_outcome_count"] == 1
    assert context["avg_pnl"] == 80.0


def test_size_up_requires_global_and_setup_family_live_evidence() -> None:
    ledger = pd.DataFrame(
        [
            {
                "report_date": f"2026-05-{(i % 28) + 1:02d}",
                "strategy": "Bull Put Credit Spread",
                "direction": "Bull Put",
                "realized_pnl": 25.0,
            }
            for i in range(50)
        ]
    )

    summary = summarize_live_outcomes(ledger)
    adjustment = live_outcome_adjustment(summary, "Bull Put Credit Spread", "Bull Put")

    assert summary["size_up_allowed"] is True
    assert adjustment["block_size_up"] is False
    assert live_outcome_adjustment(None, "Bull Put Credit Spread", "Bull Put")["block_size_up"] is True
