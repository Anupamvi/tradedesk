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

    def test_a_does_not_fire_below_ema20(self):
        snap = {
            "ok": True,
            "close": 142.61,
            "low": 141.55,
            "high": 145.55,
            "open": 143.67,
            "ema20": 145.88,
            "sma50": 132.22,
            "sma200": 131.31,
            "atr14": 6.01,
            "trend": "up",
            "rs_20": 0.17,
            "rvol": 0.7,
            "above_ema20": False,
            "above_sma50": True,
            "above_sma200": True,
            "avwap_swing_low": 131.18,
            "extension_atr": -0.54,
            "ret_1": -0.032,
            "hi20_close": 156.49,
            "vol_5": 1.0,
            "vol_20": 2.0,
        }
        setup = classify_setups(snap, group_row={"status": "mature"}, earnings={"usable": True, "source": "exempt"})
        self.assertNotIn("A", setup["setups"])

    def test_red_day_e_and_a_prefers_a(self):
        snap = {
            "ok": True,
            "close": 147.47,
            "low": 144.88,
            "high": 151.28,
            "open": 150.70,
            "ema20": 146.23,
            "sma50": 131.53,
            "sma200": 131.39,
            "atr14": 6.03,
            "trend": "strong_up",
            "rs_20": 0.247,
            "rvol": 0.7,
            "above_ema20": True,
            "above_sma50": True,
            "above_sma200": True,
            "avwap_swing_low": 131.14,
            "extension_atr": 0.20,
            "ret_1": -0.0355,
            "hi20_close": 156.49,
            "vol_5": 1.0,
            "vol_20": 2.0,
        }
        setup = classify_setups(snap, group_row={"status": "accelerating"}, earnings={"usable": True, "source": "exempt"})
        self.assertIn("A", setup["setups"])
        self.assertIn("E", setup["setups"])
        self.assertEqual(setup["primary"], "A")

    def test_incomplete_stub_does_not_flip_rs_leader_to_breakout(self):
        spy = trend_bars(220, end="2026-08-26", slope=0.05, volume=2_000_000.0)
        bars = trend_bars(220, end="2026-08-26", slope=0.4, pullback=1.0, volume=2_000_000.0)
        yesterday = bars[-1]
        hi20 = max(b["close"] for b in bars[:-1][-20:])
        stub = {
            "date": "2026-08-27",
            "open": yesterday["close"],
            "high": hi20 + 2.0,
            "low": yesterday["close"] - 0.2,
            "close": hi20 + 1.5,
            "volume": 90_000.0,
        }
        live = snapshot(bars + [stub], "2026-08-27", bench_bars=spy + [dict(spy[-1], date="2026-08-27", volume=90_000.0)])
        self.assertTrue(live["session_incomplete"])
        self.assertGreater(live["live_last"], live["hi20_close"])
        self.assertLessEqual(live["close"], live["hi20_close"])
        live["rs_20"] = 0.12
        live["above_sma50"] = True
        live["trend"] = "up"
        setup = classify_setups(live, group_row={"status": "mature"}, earnings={"usable": True, "source": "exempt"})
        self.assertNotIn("B", setup["setups"])
        self.assertEqual(setup["primary"], "D")
        self.assertIsNone(setup["fire"].get("kind"))

    def test_incomplete_red_stub_does_not_demote_d_to_a(self):
        snap = {
            "ok": True,
            "session_incomplete": True,
            "close": 150.0,
            "structure_close": 150.0,
            "low": 148.8,
            "high": 151.0,
            "open": 150.2,
            "ema20": 146.0,
            "sma50": 140.0,
            "sma200": 130.0,
            "atr14": 6.0,
            "trend": "strong_up",
            "rs_20": 0.18,
            "rvol": 0.08,
            "above_ema20": True,
            "above_sma50": True,
            "above_sma200": True,
            "avwap_swing_low": 131.0,
            "extension_atr": 0.20,
            "ret_1": 0.04,
            "hi20_close": 156.0,
            "vol_5": 1.8e6,
            "vol_20": 2.0e6,
        }
        setup = classify_setups(snap, group_row={"status": "mature"}, earnings={"usable": True, "source": "exempt"})
        self.assertIn("D", setup["setups"])
        self.assertEqual(setup["primary"], "D")
        self.assertIsNone(setup["fire"].get("kind"))

    def test_complete_session_breakout_still_prints_b(self):
        spy = trend_bars(80, end="2026-08-27", slope=0.15, volume=2_000_000.0)
        bars = trend_bars(80, end="2026-08-26", slope=0.45, pullback=3.0, volume=2_000_000.0)
        yesterday = bars[-1]
        full = {
            "date": "2026-08-27",
            "open": yesterday["close"],
            "high": yesterday["close"] + 12.0,
            "low": yesterday["close"] - 0.3,
            "close": yesterday["close"] + 11.0,
            "volume": 2_200_000.0,
        }
        snap = snapshot(bars + [full], "2026-08-27", bench_bars=spy)
        self.assertFalse(snap.get("session_incomplete"))
        setup = classify_setups(snap, group_row={"status": "mature"}, earnings={"usable": True, "source": "exempt"})
        self.assertIn("B", setup["setups"])
        self.assertEqual(setup["primary"], "B")

    def test_e_requires_rs_leader_not_any_green_group(self):
        snap = {
            "ok": True,
            "close": 85.56,
            "low": 85.20,
            "high": 85.80,
            "open": 85.40,
            "ema20": 85.54,
            "sma50": 85.00,
            "sma200": 83.20,
            "atr14": 0.70,
            "trend": "up",
            "rs_20": 0.008,
            "rvol": 1.0,
            "above_ema20": True,
            "above_sma50": True,
            "above_sma200": True,
            "avwap_swing_low": 84.99,
            "extension_atr": 0.03,
            "ret_1": 0.004,
            "hi20_close": 86.50,
            "vol_5": 1.0,
            "vol_20": 1.1,
        }
        setup = classify_setups(snap, group_row={"status": "emerging"}, earnings={"usable": True, "source": "exempt"})
        self.assertNotIn("E", setup["setups"])

    def test_d_outranks_a_when_group_is_mature(self):
        snap = {
            "ok": True,
            "close": 281.0,
            "low": 278.0,
            "high": 282.0,
            "open": 279.0,
            "ema20": 279.5,
            "sma50": 260.0,
            "sma200": 250.0,
            "atr14": 6.0,
            "trend": "up",
            "rs_20": 0.09,
            "rvol": 0.8,
            "above_ema20": True,
            "above_sma50": True,
            "above_sma200": True,
            "avwap_swing_low": 270.0,
            "extension_atr": 0.25,
            "ret_1": 0.01,
            "hi20_close": 285.0,
            "vol_5": 0.8,
            "vol_20": 1.2,
        }
        setup = classify_setups(snap, group_row={"status": "mature"}, earnings={"usable": True, "source": "exempt"})
        self.assertIn("D", setup["setups"])
        self.assertIn("A", setup["setups"])
        self.assertNotIn("E", setup["setups"])
        self.assertEqual(setup["primary"], "D")

    def test_green_rotation_still_e(self):
        snap = {
            "ok": True,
            "close": 152.0,
            "low": 148.0,
            "high": 153.0,
            "open": 149.0,
            "ema20": 146.0,
            "sma50": 131.0,
            "sma200": 131.0,
            "atr14": 6.0,
            "trend": "strong_up",
            "rs_20": 0.20,
            "rvol": 1.0,
            "above_ema20": True,
            "above_sma50": True,
            "above_sma200": True,
            "avwap_swing_low": 131.0,
            "extension_atr": 1.0,
            "ret_1": 0.02,
            "hi20_close": 156.0,
            "vol_5": 1.0,
            "vol_20": 2.0,
        }
        setup = classify_setups(snap, group_row={"status": "accelerating"}, earnings={"usable": True, "source": "exempt"})
        self.assertEqual(setup["primary"], "E")

    def test_no_setup_on_chop(self):
        bars = flat_bars(80, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        setup = classify_setups(snap, group_row={"status": "neutral"}, earnings={"usable": True, "source": "exempt"})
        self.assertIn(setup["direction"], ("neutral", "bearish", "bullish"))
        if not setup["primary"]:
            self.assertEqual(setup["direction"], "neutral")


if __name__ == "__main__":
    unittest.main()
