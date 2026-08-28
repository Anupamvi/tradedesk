import unittest

from groat.gates import trade_park_reason
from groat.replay import run_replay
from tests.barsutil import trend_bars


class TestReplay(unittest.TestCase):
    def test_universe_walk_has_stock_hits(self):
        asof = "2026-08-26"
        spy = trend_bars(220, end=asof, start_px=500, slope=0.3)
        igv = trend_bars(220, end=asof, start_px=80, slope=0.6)
        now = trend_bars(220, end=asof, start_px=90, slope=0.55)
        payload = run_replay(
            asof,
            token="",
            universe=["NOW"],
            bars_by_ticker={"NOW": now, "SPY": spy, "IGV": igv},
            max_strike_http=0,
        )
        self.assertEqual(payload["mode"], "replay")
        self.assertGreaterEqual(payload["overall"]["n"], 1)
        self.assertEqual(payload["strike_http"], 0)
        self.assertIsNotNone(payload["tape_from"])
        self.assertEqual(payload["overall"]["opt_n"], 0)
        parked = {row["primary"] for row in payload["setups"] if (row.get("n") or 0) > 0}
        self.assertNotIn("B", parked)
        self.assertNotIn("C", parked)
        self.assertNotIn("G", parked)


class TestGates(unittest.TestCase):
    def test_park_b_c_g_and_post_rip_e(self):
        self.assertEqual(trade_park_reason("B"), "setup_B_replay_park")
        self.assertEqual(trade_park_reason("C"), "setup_C_replay_park")
        self.assertEqual(trade_park_reason("G"), "setup_G_replay_park")
        self.assertIsNone(trade_park_reason("D"))
        self.assertIsNone(trade_park_reason("A"))
        self.assertEqual(
            trade_park_reason("E", {"ret_1": 0.20, "extension_atr": 1.0}, {}),
            "setup_E_post_rip",
        )
        self.assertIsNone(trade_park_reason("E", {"ret_1": 0.02, "extension_atr": 1.0}, {}))
