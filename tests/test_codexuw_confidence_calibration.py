from __future__ import annotations

import datetime as dt

import pandas as pd

from codexuw.confidence_calibration import (
    CONSERVATIVE_CONFIDENCE_STATUS,
    apply_confidence_calibration,
    build_walk_forward_calibration,
    confidence_high_ready,
    wilson_lower_bound,
)


def _history(outcomes: list[int]) -> pd.DataFrame:
    start = dt.date(2026, 1, 2)
    rows = []
    for index, outcome in enumerate(outcomes):
        day = start + dt.timedelta(days=index)
        rows.append(
            {
                "asof": day.isoformat(),
                "exit_day": day.isoformat(),
                "expiry": (day + dt.timedelta(days=14)).isoformat(),
                "ticker": f"T{index}",
                "direction": "Bull Call",
                "regime": "uptrend",
                "exact_evaluated": True,
                "decision_pass": True,
                "replay_guard_pass": True,
                "pnl_1x": 100.0 if outcome else -100.0,
            }
        )
    return pd.DataFrame(rows)


def test_walk_forward_calibration_uses_only_resolved_prior_rows_and_can_pass():
    detail, summary = build_walk_forward_calibration(_history([1] * 60))

    assert len(detail) == 48
    assert detail.iloc[0]["prior_sample_size"] == 12
    assert summary["status"] == "PASS"
    assert summary["brier_score"] < summary["baseline_brier_score"]


def test_unvalidated_probability_is_descriptive_and_cannot_be_high():
    scored = pd.DataFrame(
        [
            {
                "edge_sample_size": 24,
                "edge_win_rate": 0.70,
                "edge_effective_win_rate": 0.69,
                "edge_profit_factor": 1.80,
                "edge_avg_pnl": 40.0,
            }
        ]
    )
    attached = apply_confidence_calibration(
        scored,
        {"status": "FAIL", "brier_score": 0.27, "baseline_brier_score": 0.25},
    )

    assert attached.iloc[0]["confidence_probability_label"] == "descriptive_only"
    assert attached.iloc[0]["confidence_model_tier"] == "medium"
    assert not confidence_high_ready(attached.iloc[0])
    assert 0 < wilson_lower_bound(16, 24) < 1


def test_underprediction_with_good_brier_is_conservative_not_high_confidence() -> None:
    outcomes = [0, 1] * 10 + [1] * 40
    detail, summary = build_walk_forward_calibration(
        _history(outcomes),
        min_predictions=20,
        max_calibration_gap=0.01,
    )
    assert not detail.empty
    assert summary["family_validation"]["Debit"]["status"] == CONSERVATIVE_CONFIDENCE_STATUS

    attached = apply_confidence_calibration(
        pd.DataFrame([{"strategy_kind": "Debit", "direction": "Bull Call", "regime": "uptrend"}]),
        summary,
    )
    row = attached.iloc[0]
    assert row["confidence_calibration_status"] == CONSERVATIVE_CONFIDENCE_STATUS
    assert row["confidence_model_tier"] == "strategy_family_conservative"
    assert row["confidence_probability_label"] == "walk_forward_conservative"
    assert not confidence_high_ready(row)


def test_research_calibration_can_exclude_policy_guard_without_changing_default() -> None:
    history = _history([1] * 60)
    history["replay_guard_pass"] = False

    default_detail, default_summary = build_walk_forward_calibration(history)
    research_detail, research_summary = build_walk_forward_calibration(
        history,
        require_replay_guard=False,
    )

    assert default_detail.empty
    assert default_summary["eligible_history_rows"] == 0
    assert len(research_detail) == 48
    assert research_summary["eligible_history_rows"] == 60
    assert research_summary["require_replay_guard"] is False
