from __future__ import annotations

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
    assert performance_risk_multiplier(context) == 1.0
    assert performance_min_score(context, 5.0) == 5.0


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


def test_load_live_outcome_performance_prefers_recommendation_ledger(tmp_path) -> None:
    pd.DataFrame(
        [
            {"report_date": "2026-05-01", "strategy": "Bull Call Debit Spread", "direction": "Bull Call", "realized_pnl": 120.0}
        ]
    ).to_csv(tmp_path / "codexuw_recommendation_outcome_ledger.csv", index=False)

    context = load_live_outcome_performance(tmp_path)

    assert context["status"] == "ok"
    assert context["source"].endswith("codexuw_recommendation_outcome_ledger.csv")
