import datetime as dt
import tempfile
import unittest
from pathlib import Path

from uwos import options_trend_pipeline_v2 as pipeline


class OptionsTrendPipelineV2Tests(unittest.TestCase):
    def test_inventory_counts_local_uw_artifact_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day_dir = root / "2026-05-01"
            (day_dir / "_unzipped_mode_a").mkdir(parents=True)
            (day_dir / "browser_text").mkdir()
            (day_dir / "stock-screener-2026-05-01.zip").write_text("placeholder", encoding="utf-8")
            (day_dir / "_unzipped_mode_a" / "hot-chains-2026-05-01.csv").write_text(
                "option_symbol,bid,ask\n", encoding="utf-8"
            )
            (day_dir / "bot-eod-report-2026-05-01.zip").write_text("placeholder", encoding="utf-8")
            (day_dir / "browser_text" / "browser-text-capture-news-MACRO-LIVE-2026-05-01.txt").write_text(
                "Fed context", encoding="utf-8"
            )

            inventory, summary = pipeline.inventory_uw_data(root, dt.date(2026, 5, 1))

        self.assertEqual(summary["total_dated_folders"], 1)
        row = inventory.iloc[0]
        self.assertEqual(row["date"], "2026-05-01")
        self.assertGreaterEqual(row["stock_screener_files"], 1)
        self.assertGreaterEqual(row["hot_chains_files"], 1)
        self.assertGreaterEqual(row["bot_eod_report_files"], 1)
        self.assertGreaterEqual(row["browser_text_files"], 1)

    def test_readiness_verdict_requires_actionable_edge(self):
        proof = {
            "verdict": "PROVEN_FOR_ACTIONABLE",
            "v2_avg_net_r": 0.18,
            "v2_profit_factor": 1.4,
            "best_comparison_avg_net_r": 0.05,
        }

        self.assertEqual(
            pipeline.readiness_verdict(proof, {"actionable": 1, "validation_outcomes": 40}),
            "PRODUCTION_READY",
        )
        self.assertEqual(
            pipeline.readiness_verdict(proof, {"actionable": 0, "validation_outcomes": 40}),
            "USABLE_NEEDS_MORE_VALIDATION",
        )
        self.assertEqual(
            pipeline.readiness_verdict({}, {"actionable": 0, "validation_outcomes": 0}),
            "NOT_YET_PROVEN",
        )


if __name__ == "__main__":
    unittest.main()
