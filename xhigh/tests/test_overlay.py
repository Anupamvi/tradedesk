import json
import tempfile
import unittest
from pathlib import Path

from xhigh.gates import load_gates
from xhigh.report import overlay_intel, overlay_x, write_run

GATES = load_gates()

CLICK_CREDIT = {
    "ticker": "WMT",
    "last": 106.83,
    "structure": "put_credit",
    "strategy": "SELL 100 P / BUY 95 P",
    "pop_delta": 0.75,
    "conf": 60,
    "credit": 1.25,
    "width": 5.0,
    "short_strike": 100.0,
    "long_strike": 95.0,
    "action": "CLICK",
}


class TestIntelKill(unittest.TestCase):
    def test_kill_survives_write_run_and_xhot(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            write_run(
                dest,
                date="2026-09-18",
                tickets=[dict(CLICK_CREDIT)],
                skip=[],
                watch=[],
                x_queue=["WMT"],
                gates=GATES,
                skips=[],
                manifest={
                    "date": "2026-09-18",
                    "live_schwab": True,
                    "movers": ["WMT"],
                    "shortlist": ["WMT"],
                    "chain_http": 1,
                },
            )
            click = json.loads((dest / "tickets.json").read_text(encoding="utf-8"))
            self.assertEqual(len(click), 1)
            overlay_intel(
                dest,
                {
                    "asof": "2026-09-18",
                    "names": [{"ticker": "WMT", "tag": "Quiet", "kill": True, "conf_delta": 0}],
                },
            )
            click = json.loads((dest / "tickets.json").read_text(encoding="utf-8"))
            watch = json.loads((dest / "watch.json").read_text(encoding="utf-8"))
            self.assertEqual(click, [])
            self.assertEqual(watch[0]["action"], "WATCH")
            self.assertTrue(watch[0].get("intel_kill"))
            overlay_x(dest, {"asof": "2026-09-18", "names": [{"ticker": "WMT", "tag": "Quiet"}]})
            watch = json.loads((dest / "watch.json").read_text(encoding="utf-8"))
            click = json.loads((dest / "tickets.json").read_text(encoding="utf-8"))
            self.assertEqual(click, [])
            self.assertEqual(watch[0]["action"], "WATCH")


class TestEmptyUniverseCopy(unittest.TestCase):
    def test_scanned_empty_still_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp)
            write_run(
                dest,
                date="2026-09-18",
                tickets=[],
                skip=[
                    {
                        "ticker": "NVDA",
                        "structure": "call_debit",
                        "pop_delta": 0.35,
                        "conf": 70,
                        "long_delta": 0.35,
                        "rr": 1.2,
                        "debit": 2.0,
                        "dte": 25,
                        "last": 220.0,
                        "long_strike": 225.0,
                        "short_strike": 235.0,
                    }
                ],
                watch=[],
                x_queue=[],
                gates=GATES,
                skips=[],
                manifest={
                    "date": "2026-09-18",
                    "live_schwab": True,
                    "movers": ["NVDA"],
                    "shortlist": ["NVDA"],
                    "chain_http": 1,
                },
            )
            text = (dest / "recommendation.md").read_text(encoding="utf-8")
            self.assertIn("Empty is valid.", text)
            self.assertNotIn("DATA UNAVAILABLE — universe.", text)


if __name__ == "__main__":
    unittest.main()
