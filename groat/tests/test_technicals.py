import unittest

from groat.technicals import atr_wilder, avwap, ema, snapshot, sma
from tests.barsutil import trend_bars


class TestTechnicals(unittest.TestCase):
    def test_sma_ema(self):
        values = [float(i) for i in range(1, 21)]
        self.assertAlmostEqual(sma(values, 5), 18.0)
        self.assertIsNone(sma(values, 50))
        self.assertIsNotNone(ema(values, 5))

    def test_atr_and_snapshot(self):
        bars = trend_bars(220, end="2026-08-26")
        asof = bars[-1]["date"]
        atr = atr_wilder(bars, asof)
        self.assertIsNotNone(atr)
        self.assertGreater(atr, 0)
        vw = avwap(bars, asof, bars[0]["date"])
        self.assertIsNotNone(vw)
        snap = snapshot(bars, asof, bench_bars=bars)
        self.assertTrue(snap["ok"])
        self.assertFalse(snap["stale"])
        self.assertIn(snap["trend"], ("up", "strong_up"))
        self.assertTrue(snap["above_ema20"])

    def test_avwap_none_without_volume(self):
        bars = trend_bars(40, end="2026-08-26")
        for bar in bars:
            bar["volume"] = None
        self.assertIsNone(avwap(bars, bars[-1]["date"], bars[0]["date"]))

    def test_stale_flag(self):
        bars = trend_bars(30, end="2026-08-26")
        snap = snapshot(bars, "2026-08-27")
        self.assertTrue(snap["ok"])
        self.assertTrue(snap["stale"])
        self.assertFalse(snap.get("session_incomplete"))

    def test_incomplete_stub_does_not_replace_structure_close(self):
        complete = trend_bars(60, end="2026-08-26", slope=0.4, volume=2_000_000.0)
        yesterday = complete[-1]
        stub = {
            "date": "2026-08-27",
            "open": yesterday["close"],
            "high": yesterday["close"] + 8.0,
            "low": yesterday["close"] - 0.2,
            "close": yesterday["close"] + 7.5,
            "volume": 80_000.0,
        }
        snap = snapshot(complete + [stub], "2026-08-27")
        self.assertTrue(snap["ok"])
        self.assertFalse(snap["stale"])
        self.assertTrue(snap["session_incomplete"])
        self.assertAlmostEqual(snap["structure_close"], yesterday["close"])
        self.assertAlmostEqual(snap["close"], yesterday["close"])
        self.assertAlmostEqual(snap["live_last"], stub["close"])
        self.assertLess(snap["rvol"], 0.2)
        self.assertGreater(snap["live_last"], snap["close"])


if __name__ == "__main__":
    unittest.main()
