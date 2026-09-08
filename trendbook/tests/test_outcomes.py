import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from trendbook import outcomes


def _flags(ons, start="2025-01-03", px0=100.0):
    day = datetime.strptime(start, "%Y-%m-%d")
    px = px0
    rows = []
    for on in ons:
        px = px * (1.04 if on else 0.97)
        hi = px * 1.02
        rows.append(
            {
                "asof": day.date().isoformat(),
                "close": px,
                "high": hi,
                "on": on,
                "stage": 2 if on else 1,
            }
        )
        day += timedelta(weeks=1)
    return rows


class TestScoreAdd(unittest.TestCase):
    def test_horizon_up_is_capture(self):
        flags = _flags([True] * 12)
        scored = outcomes.score_add(flags, 0, flags[0]["close"])
        self.assertEqual(scored["result"], "CAPTURE")
        self.assertEqual(scored["reason"], "horizon_up")

    def test_first_tick_up_then_down_is_fail(self):
        flags = _flags([True] * 10)
        flags[0]["close"] = 100.0
        flags[1]["close"] = 104.0
        flags[1]["open"] = 100.0
        for row in flags[2:]:
            row["close"] = 90.0
            row["open"] = 90.0
            row["on"] = True
        scored = outcomes.score_add(flags, 0, 100.0)
        self.assertEqual(scored["result"], "FAIL")
        self.assertEqual(scored["reason"], "horizon_down")

    def test_left_campaign_before_high_is_fail(self):
        flags = _flags([True, False, False, False, False, False, False, False, False])
        flags[0]["high"] = 200.0
        flags[0]["close"] = 100.0
        for row in flags[1:]:
            row["high"] = 90.0
            row["close"] = 80.0
        scored = outcomes.score_add(flags, 0, 100.0)
        self.assertEqual(scored["result"], "FAIL")
        self.assertEqual(scored["reason"], "invalidated")

    def test_still_on_but_down_at_week_8_is_fail(self):
        flags = _flags([True] * 10)
        flags[0]["high"] = 1000.0
        flags[0]["close"] = 100.0
        for row in flags[1:]:
            row["high"] = 50.0
            row["close"] = 40.0
            row["on"] = True
        scored = outcomes.score_add(flags, 0, 100.0)
        self.assertEqual(scored["result"], "FAIL")
        self.assertEqual(scored["reason"], "horizon_down")

    def test_wick_then_off_is_fail(self):
        flags = _flags([True, False, False, False, False, False, False, False, False])
        flags[0]["high"] = 100.0
        flags[0]["close"] = 100.0
        flags[1]["high"] = 100.04
        flags[1]["close"] = 70.0
        flags[1]["on"] = False
        scored = outcomes.score_add(flags, 0, 100.0)
        self.assertEqual(scored["result"], "FAIL")
        self.assertEqual(scored["reason"], "invalidated")

    def test_need_eight_weeks_is_open(self):
        flags = _flags([True, True, True])
        flags[0]["high"] = 100.0
        flags[0]["close"] = 100.0
        for row in flags[1:]:
            row["close"] = 95.0
            row["high"] = 96.0
            row["on"] = True
        scored = outcomes.score_add(flags, 0, 100.0)
        self.assertEqual(scored["result"], "OPEN")

    def test_record_skips_hold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            outcomes.record(
                [{"ticker": "AAA", "action": "HOLD", "close": 10, "trend_start": "2025-01-03"}],
                "2025-06-01",
                path=path,
            )
            self.assertEqual(outcomes.load_ledger(path), [])
            outcomes.record(
                [
                    {
                        "ticker": "BBB",
                        "action": "ADD",
                        "close": 10,
                        "trend_start": "2025-01-03",
                        "grade": "A",
                    }
                ],
                "2025-06-01",
                path=path,
            )
            ledger = outcomes.load_ledger(path)
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0]["result"], "OPEN")


if __name__ == "__main__":
    unittest.main()
