from __future__ import annotations

import datetime as dt

import pandas as pd

from codexuw.confidence_calibration import (
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


def test_walk_forward_calibration_uses_only_exited_prior_rows_and_can_pass():
    detail, summary = build_walk_forward_calibration(_history([1] * 50))

    assert len(detail) == 38
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
