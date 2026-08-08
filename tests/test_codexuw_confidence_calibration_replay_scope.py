from __future__ import annotations

import pandas as pd

from codexuw.confidence_calibration import (
    apply_confidence_calibration,
    build_walk_forward_calibration,
    confidence_high_ready,
)


def _credit_history(rows: int = 72) -> pd.DataFrame:
    start = pd.Timestamp("2026-01-02")
    records = []
    for index in range(rows):
        asof = start + pd.Timedelta(days=index)
        records.append(
            {
                "asof": asof.date().isoformat(),
                "exit_day": (asof + pd.Timedelta(days=1)).date().isoformat(),
                "ticker": f"T{index:03d}",
                "strategy_kind": "Credit",
                "strategy": "Bull Put Credit Spread",
                "direction": "Bull Put",
                "regime": "range",
                "exact_evaluated": True,
                "replay_guard_pass": True,
                "decision_pass": False,
                "exact_win": index % 3 != 0,
            }
        )
    return pd.DataFrame(records)


def test_replay_validated_rows_calibrate_even_when_not_finally_approved() -> None:
    detail, summary = build_walk_forward_calibration(
        _credit_history(),
        asof="2026-04-01",
    )

    assert len(detail) >= 30
    assert summary["eligible_history_rows"] == 72
    assert summary["actionable_history_rows"] == 0
    assert summary["family_validation"]["Credit"]["status"] == "PASS"
    assert summary["high_confidence_available"] is True

    live = pd.DataFrame(
        [
            {
                "ticker": "LIVE",
                "strategy_kind": "Credit",
                "strategy": "Bull Put Credit Spread",
                "direction": "Bull Put",
                "regime": "range",
            }
        ]
    )
    calibrated = apply_confidence_calibration(live, summary)
    row = calibrated.iloc[0]
    assert row["confidence_probability"] >= 0.60
    assert row["confidence_calibration_sample_size"] == 72
    assert row["confidence_probability_source"] == "strategy_family"
    assert confidence_high_ready(row) is True


def test_replay_guard_still_excludes_invalid_outcomes() -> None:
    history = _credit_history()
    history.loc[0, "replay_guard_pass"] = False
    _, summary = build_walk_forward_calibration(history, asof="2026-04-01")
    assert summary["eligible_history_rows"] == 71


def test_early_exit_is_not_confidence_evidence_until_expiry() -> None:
    history = _credit_history(rows=2)
    history["expiry"] = ["2026-03-20", "2026-05-15"]

    _, summary = build_walk_forward_calibration(history, asof="2026-04-01")

    assert summary["eligible_history_rows"] == 1


def test_walk_forward_prior_waits_for_contract_maturity() -> None:
    history = _credit_history(rows=20)
    history["expiry"] = "2026-02-15"

    _, summary = build_walk_forward_calibration(history, asof="2026-03-01")

    assert summary["eligible_history_rows"] == 20
    assert summary["family_validation"]["Credit"]["prediction_count"] == 0
