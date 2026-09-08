import unittest
from datetime import datetime, timedelta

from trendbook.bars import daily_snapshot, to_weekly
from trendbook.entry import classify_entry
from trendbook.stage import classify_stage
from trendbook.template import eight_point


def _daily(start: str, n: int, start_px: float, drift: float, vol: float = 1.0) -> list:
    day = datetime.strptime(start, "%Y-%m-%d")
    px = start_px
    rows = []
    made = 0
    while made < n:
        if day.weekday() < 5:
            o = px
            c = px * (1.0 + drift)
            h = max(o, c) * 1.01
            lo = min(o, c) * 0.99
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "open": o,
                    "high": h,
                    "low": lo,
                    "close": c,
                    "volume": 1_000_000.0 * vol,
                }
            )
            px = c
            made += 1
        day += timedelta(days=1)
    return rows


class TestStage(unittest.TestCase):
    def test_rising_series_is_stage_2(self):
        bars = _daily("2024-01-02", 400, 50.0, 0.004)
        asof = bars[-1]["date"]
        weekly = to_weekly(bars, asof)
        st = classify_stage(weekly)
        self.assertEqual(st["stage"], 2)
        self.assertTrue(st["rising"])

    def test_falling_series_is_stage_4(self):
        bars = _daily("2024-01-02", 400, 200.0, -0.004)
        asof = bars[-1]["date"]
        weekly = to_weekly(bars, asof)
        st = classify_stage(weekly)
        self.assertEqual(st["stage"], 4)

    def test_need_30_weeks(self):
        bars = _daily("2024-01-02", 40, 50.0, 0.01)
        weekly = to_weekly(bars, bars[-1]["date"])
        st = classify_stage(weekly)
        self.assertIsNone(st["stage"])
        self.assertEqual(st["reason"], "need_30_weeks")


class TestEntry(unittest.TestCase):
    def test_30w_dip_below_20ema_is_pullback(self):
        daily = {
            "close": 100.0,
            "ema20": 102.0,
            "sma50": 110.0,
            "atr14": 4.0,
            "extension_atr": -0.5,
            "range20": 0.12,
            "tightness": 0.6,
        }
        weekly = {"stage": 2, "ma10": 108.0, "ma30": 99.5}
        entry = classify_entry(daily, weekly)
        self.assertEqual(entry["state"], "ready")
        self.assertEqual(entry["reason"], "pullback_30w")

    def test_near_30w_above_20ema_is_not_pullback(self):
        daily = {
            "close": 105.0,
            "ema20": 100.0,
            "sma50": 90.0,
            "atr14": 4.0,
            "extension_atr": 1.25,
            "range20": 0.12,
            "tightness": 0.6,
        }
        weekly = {"stage": 2, "ma10": 92.0, "ma30": 104.0}
        entry = classify_entry(daily, weekly)
        self.assertNotEqual(entry.get("reason"), "pullback_30w")
        self.assertNotEqual(entry["state"], "ready")

    def test_extended_is_not_ready(self):
        bars = _daily("2024-01-02", 260, 50.0, 0.003)
        # Spike last bar far above EMA
        last = dict(bars[-1])
        last["close"] = last["close"] * 1.25
        last["high"] = last["close"]
        bars[-1] = last
        asof = last["date"]
        daily = daily_snapshot(bars, asof)
        weekly = to_weekly(bars, asof)
        st = classify_stage(weekly)
        st["stage"] = 2
        entry = classify_entry(daily, st)
        self.assertEqual(entry["state"], "extended")

    def test_template_counts(self):
        daily = {
            "close": 120.0,
            "sma50": 110.0,
            "sma150": 100.0,
            "sma200": 90.0,
            "sma200_prev_21": 88.0,
            "hi252": 125.0,
            "lo252": 70.0,
        }
        tmpl = eight_point(daily, 85.0)
        self.assertTrue(tmpl["pass"])
        self.assertEqual(tmpl["n_pass"], 8)


if __name__ == "__main__":
    unittest.main()
