import datetime as dt
import json
import unittest
from pathlib import Path

from uwos import trend_analysis_repro_audit as audit


class TestTrendAnalysisReproAudit(unittest.TestCase):
    def test_compare_artifacts_flags_csv_drift(self) -> None:
        with self.subTest("drift"):
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                run_a = root / "a"
                run_b = root / "b"
                run_a.mkdir()
                run_b.mkdir()
                as_of = dt.date(2026, 4, 23)
                lookback = 30
                for name in audit.artifact_names(as_of, lookback):
                    (run_a / name).write_text("col\nsame\n", encoding="utf-8")
                    (run_b / name).write_text("col\nsame\n", encoding="utf-8")
                drift_name = f"trend-analysis-actionable-{audit.suffix(as_of, lookback)}.csv"
                (run_b / drift_name).write_text("col\ndifferent\n", encoding="utf-8")

                rows = audit.compare_artifacts(run_a, run_b, as_of, lookback)
                by_name = {row["artifact"]: row for row in rows}

                self.assertFalse(by_name[drift_name]["same"])
                self.assertTrue(by_name[f"trend-analysis-event-watch-{audit.suffix(as_of, lookback)}.csv"]["same"])

    def test_time_safety_catches_future_data_leaks(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            as_of = dt.date(2026, 4, 23)
            lookback = 2
            quote_csv = run_dir / f"trend-analysis-quote-replay-{audit.suffix(as_of, lookback)}.csv"
            quote_csv.write_text(
                "quote_replay_signal_date,quote_replay_exit_date\n"
                "2026-04-23,2026-04-24\n",
                encoding="utf-8",
            )
            raw_csv = run_dir / f"trend_analysis_raw_{audit.suffix(as_of, lookback)}.csv"
            raw_csv.write_text("ticker,live_validated\nNVDA,true\n", encoding="utf-8")
            meta = {
                "as_of": "2026-04-23",
                "effective_signal_date": "2026-04-24",
                "lookback": 2,
                "trading_days": ["2026-04-22", "2026-04-24"],
                "latest_data_date": "2026-04-24",
                "schwab_enabled": True,
                "open_position_summary": {
                    "position_json": "/tmp/position_data_2026-04-25.json",
                },
                "schwab_actual_summary": {
                    "status": "ok",
                    "audit_as_of": "2026-04-24",
                    "parsed_closed_trades": 2,
                    "parsed_closed_trades_asof": 1,
                },
            }

            failures = audit.time_safety_failures(meta, run_dir, as_of, lookback)

        joined = "\n".join(failures)
        self.assertIn("effective_signal_date 2026-04-24 is after as_of", joined)
        self.assertIn("Schwab live chain enabled", joined)
        self.assertIn("position snapshot 2026-04-25", joined)
        self.assertIn("Schwab actual audit_as_of", joined)
        self.assertIn("quote replay contains future dates", joined)
        self.assertIn("current-Schwab live_validated=true", joined)

    def test_count_failures_checks_metadata_against_csv_rows(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            as_of = dt.date(2026, 4, 23)
            lookback = 30
            for template in audit.COUNT_METADATA_BY_ARTIFACT:
                path = run_dir / template.format(suffix=audit.suffix(as_of, lookback))
                path.write_text("ticker\nAAPL\n", encoding="utf-8")
            metadata = {key: 1 for key in audit.COUNT_METADATA_BY_ARTIFACT.values()}
            metadata["actionable"] = 2

            failures = audit.count_failures(metadata, run_dir, as_of, lookback)

        self.assertEqual(len(failures), 1)
        self.assertIn("actionable", failures[0])

    def test_deterministic_metadata_diff_uses_only_stable_keys(self) -> None:
        base = {
            "as_of": "2026-04-23",
            "lookback": 30,
            "out_dir": "/tmp/a",
            "report": "/tmp/a/report.md",
        }
        other = json.loads(json.dumps(base))
        other["out_dir"] = "/tmp/b"
        other["report"] = "/tmp/b/report.md"

        self.assertEqual(audit.deterministic_metadata_diff(base, other), [])
        other["lookback"] = 31
        self.assertEqual(audit.deterministic_metadata_diff(base, other)[0]["key"], "lookback")


if __name__ == "__main__":
    unittest.main()
