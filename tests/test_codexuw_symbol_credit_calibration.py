import math

import pandas as pd

from codexuw.symbol_credit_calibration import _calibration_metrics, apply_symbol_credit_calibration


def _resolved_evidence() -> pd.DataFrame:
    signal_days = list(pd.date_range("2026-01-05", periods=7, freq="7D")) + list(
        pd.date_range("2026-08-11", periods=5, freq="2D")
    )
    return pd.DataFrame(
        {
            "_signal_day": signal_days,
            "_exit_day": signal_days,
            "_ticker": [f"T{index}" for index in range(len(signal_days))],
            "stress_pnl_10pct": [50.0] * len(signal_days),
            "_episode_id": range(1, len(signal_days) + 1),
        }
    )


def test_symbol_credit_metrics_require_fresh_history_and_postactivation_results() -> None:
    evidence = _resolved_evidence()
    passed = _calibration_metrics(evidence, pd.Timestamp("2026-08-31"), history_fresh=True)
    stale = _calibration_metrics(evidence, pd.Timestamp("2026-08-31"), history_fresh=False)

    assert passed["status"] == "PASS"
    assert passed["sample_size"] == 12
    assert passed["postactivation_sample_size"] == 5
    assert passed["independent_episode_count"] == 12
    assert math.isinf(passed["stress_profit_factor_10pct"])
    assert stale["status"] == "FAIL"
    assert stale["reason"] == "history_freshness_failed"


def test_symbol_credit_application_uses_direction_regime_group_not_pooled_summary() -> None:
    passed = _calibration_metrics(_resolved_evidence(), pd.Timestamp("2026-08-31"), history_fresh=True)
    failed = {**passed, "status": "FAIL", "reason": "group_failed", "sample_size": 2}
    summary = {
        "version": "test",
        "status": "PASS",
        "groups": {
            "Bull Put|downtrend": passed,
            "Bear Call|uptrend": failed,
        },
    }
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "direction": "Bull Put",
                "strategy": "Bull Put Credit Spread",
                "technical_close": 90.0,
                "technical_sma20": 95.0,
                "technical_sma50": 100.0,
                "technical_return_20d": -0.10,
            },
            {
                "ticker": "BBB",
                "direction": "Bear Call",
                "strategy": "Bear Call Credit Spread",
                "technical_close": 110.0,
                "technical_sma20": 105.0,
                "technical_sma50": 100.0,
                "technical_return_20d": 0.10,
            },
        ]
    )

    def assessor(row, *, live):
        return True, []

    applied = apply_symbol_credit_calibration(frame, summary, assessor=assessor)

    bull = applied.loc[applied["ticker"].eq("AAA")].iloc[0]
    bear = applied.loc[applied["ticker"].eq("BBB")].iloc[0]
    assert bool(bull["symbol_credit_policy_pass"])
    assert bull["symbol_credit_group_key"] == "Bull Put|downtrend"
    assert bull["symbol_credit_calibration_status"] == "PASS"
    assert not bool(bear["symbol_credit_policy_pass"])
    assert bear["symbol_credit_group_key"] == "Bear Call|uptrend"
    assert bear["symbol_credit_calibration_status"] == "FAIL"
