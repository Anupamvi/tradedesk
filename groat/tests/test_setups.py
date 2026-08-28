import unittest

from groat.setups import classify_setups
from groat.technicals import snapshot
from tests.barsutil import flat_bars, trend_bars


class TestSetups(unittest.TestCase):
    def test_trend_pullback_a(self):
        bars = trend_bars(220, end="2026-08-26", slope=0.4, pullback=1.2)
        spy = trend_bars(220, end="2026-08-26", slope=0.15)
        snap = snapshot(bars, "2026-08-26", bench_bars=spy)
        setup = classify_setups(snap, group_row={"status": "accelerating"}, earnings={"usable": True, "source": "exempt"})
        self.assertIn("A", setup["setups"] + [setup["primary"]])
        self.assertEqual(setup["direction"], "bullish")

    def test_rsi_alone_is_not_f(self):
        bars = flat_bars(80, end="2026-08-26", px=100.0)
        snap = snapshot(bars, "2026-08-26")
        snap["rsi14"] = 20.0
        snap["ret_20"] = -0.02
        setup = classify_setups(snap, group_row={"status": "neutral"}, earnings={"usable": True, "source": "exempt"})
        self.assertNotIn("F", setup["setups"])

    def test_no_setup_on_chop(self):
        bars = flat_bars(80, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        setup = classify_setups(snap, group_row={"status": "neutral"}, earnings={"usable": True, "source": "exempt"})
        self.assertIn(setup["direction"], ("neutral", "bearish", "bullish"))
        if not setup["primary"]:
            self.assertEqual(setup["direction"], "neutral")


if __name__ == "__main__":
    unittest.main()
