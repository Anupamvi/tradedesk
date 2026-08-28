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


if __name__ == "__main__":
    unittest.main()
