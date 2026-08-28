import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from groki_eq.fill import pnl_dollars, stop_fill, summarize, time_stop_hit
from groki_eq.replay import _may_promote, run_replay
from tests.test_breakout import _bars


class TestFill(unittest.TestCase):
    def test_pnl_and_stop(self):
        self.assertAlmostEqual(pnl_dollars(100.0, 110.0, 10), 100.0)
        self.assertEqual(stop_fill({"open": 90.0, "low": 88.0}, 95.0), 90.0)
        self.assertEqual(stop_fill({"open": 100.0, "low": 94.0}, 95.0), 95.0)
        self.assertIsNone(stop_fill({"open": 100.0, "low": 96.0}, 95.0))
        self.assertTrue(time_stop_hit(15))
        self.assertFalse(time_stop_hit(14))

    def test_summarize(self):
        stats = summarize([100.0, 100.0, -50.0])
        self.assertEqual(stats["n"], 3)
        self.assertEqual(stats["win"], 2)
        self.assertAlmostEqual(stats["pf"], 4.0)

    def test_promote(self):
        self.assertTrue(_may_promote({"n": 30, "pf": 1.2}))
        self.assertFalse(_may_promote({"n": 29, "pf": 2.0}))


class TestReplayOffline(unittest.TestCase):
    def test_writes_status(self):
        spy = _bars(40, breakout=True)
        qqq = _bars(40, breakout=False)
        iwm = _bars(40, breakout=False)
        asof = spy[-1]["date"]
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive"
            out = Path(tmp) / "out"
            with mock.patch("groki_eq.orats.archive_dir", return_value=archive):
                with mock.patch("groki_eq.pipeline.load_universe", return_value=["SPY", "QQQ", "IWM"]):
                    with mock.patch(
                        "groki_eq.replay.ensure_bars",
                        side_effect=lambda ticker, *a, **k: {
                            "bars": {"SPY": spy, "QQQ": qqq, "IWM": iwm}[ticker],
                            "tape": "test",
                            "http": 0,
                            "error": "",
                        },
                    ):
                        report = run_replay(
                            out,
                            spy[0]["date"],
                            asof,
                            max_days=0,
                            token="x",
                        )
            self.assertGreaterEqual(report["days"], 1)
            status = json.loads((out / "sleeve_status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["split"], "2023-01-03")
            self.assertIn("breakout_eq", status["sleeves"])


if __name__ == "__main__":
    unittest.main()
