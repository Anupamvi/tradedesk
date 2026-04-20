import datetime as dt
import math
import unittest

import pandas as pd

from uwos import trend_analysis_batch


class TestTrendAnalysisBatch(unittest.TestCase):
    def test_summarize_outcomes_computes_expectancy(self) -> None:
        df = pd.DataFrame(
            [
                {"signal_date": "2026-03-10", "pnl": 100.0, "return_on_risk": 0.25},
                {"signal_date": "2026-03-11", "pnl": -50.0, "return_on_risk": -0.10},
                {"signal_date": "2026-03-11", "pnl": 25.0, "return_on_risk": 0.05},
            ]
        )

        summary = trend_analysis_batch.summarize_outcomes(df)

        self.assertEqual(summary["trades"], 3)
        self.assertEqual(summary["signal_dates"], 2)
        self.assertAlmostEqual(summary["win_rate"], 2 / 3)
        self.assertAlmostEqual(summary["total_pnl"], 75.0)
        self.assertAlmostEqual(summary["profit_factor"], 2.5)
        self.assertLessEqual(summary["max_drawdown"], 0.0)

    def test_summary_verdict_requires_sample_size(self) -> None:
        summary = {
            "trades": 3,
            "avg_pnl": 100.0,
            "profit_factor": math.inf,
            "win_rate": 1.0,
        }

        self.assertEqual(trend_analysis_batch._summary_verdict(summary), "NO_PROOF_LOW_SAMPLE")

    def test_summary_verdict_promotes_positive_large_sample(self) -> None:
        summary = {
            "trades": 25,
            "avg_pnl": 25.0,
            "profit_factor": 1.4,
            "win_rate": 0.52,
        }

        self.assertEqual(trend_analysis_batch._summary_verdict(summary), "PROMOTABLE_WITH_DEFINED_RISK")

    def test_filter_signal_dates_limits_window(self) -> None:
        df = pd.DataFrame(
            [
                {"signal_date": "2026-03-09", "pnl": 1},
                {"signal_date": "2026-03-10", "pnl": 2},
                {"signal_date": "2026-03-11", "pnl": 3},
            ]
        )

        out = trend_analysis_batch._filter_signal_dates(
            df,
            dt.date(2026, 3, 10),
            dt.date(2026, 3, 10),
        )

        self.assertEqual(list(out["pnl"]), [2])

    def test_gap_diagnostics_explain_negative_policy_and_blockers(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "policy": "entry_available_score_gate",
                    "pnl": -100.0,
                    "base_gate_reasons": "backtest UNKNOWN, signals 0",
                    "quality_reject_reasons": "expensive debit: entry 70% of spread width > 50%",
                },
                {
                    "policy": "entry_available_score_gate",
                    "pnl": 25.0,
                    "base_gate_reasons": "",
                    "quality_reject_reasons": "",
                },
            ]
        )

        gaps = trend_analysis_batch.build_gap_diagnostics(df)

        self.assertIn("broad_policy", set(gaps["area"]))
        self.assertTrue(gaps["gap"].astype(str).str.contains("expensive debit").any())
        expensive = gaps[gaps["gap"].astype(str).str.contains("expensive debit")].iloc[0]
        self.assertIn("Repair strikes", expensive["fix"])


if __name__ == "__main__":
    unittest.main()
