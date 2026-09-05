import unittest

from groat.gates import (
    analog_0win_reason,
    analog_fast_stop_reason,
    already_held_same_right_reason,
    apply_already_held_park,
    apply_analog_0win_park,
    apply_below_ema_park,
    apply_same_group_book_park,
    open_trade_verdict,
    stamp_fill_guard,
    trade_park_reason,
)
from groat.replay import run_replay
from tests.barsutil import trend_bars


class TestReplay(unittest.TestCase):
    def test_universe_walk_has_stock_hits(self):
        asof = "2026-08-26"
        spy = trend_bars(220, end=asof, start_px=500, slope=0.3)
        igv = trend_bars(220, end=asof, start_px=80, slope=0.6)
        now = trend_bars(220, end=asof, start_px=90, slope=0.25, pullback=1.2)
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
        self.assertNotIn("H", parked)


class TestGates(unittest.TestCase):
    def test_park_b_c_g_and_post_rip_e(self):
        self.assertEqual(trade_park_reason("B"), "setup_B_replay_park")
        self.assertEqual(trade_park_reason("C"), "setup_C_replay_park")
        self.assertEqual(trade_park_reason("G"), "setup_G_replay_park")
        self.assertEqual(trade_park_reason("H"), "setup_H_replay_park")
        self.assertIsNone(trade_park_reason("D"))
        self.assertIsNone(trade_park_reason("A"))
        self.assertEqual(
            trade_park_reason("E", {"ret_1": 0.20, "extension_atr": 1.0}, {}),
            "setup_E_post_rip",
        )
        self.assertIsNone(trade_park_reason("E", {"ret_1": 0.02, "extension_atr": 1.0}, {}))
        self.assertEqual(
            trade_park_reason("D", {"ret_1": 0.03, "extension_atr": 1.0}, {}),
            "setup_D_post_rip",
        )
        self.assertIsNone(trade_park_reason("D", {"ret_1": 0.02, "extension_atr": 1.0}, {}))

    def test_same_group_book_parks_new_name_not_held_ticker(self):
        xle = {"ticker": "XLE", "action": "TRADE", "group": "energy", "reasons": []}
        cvx = {"ticker": "CVX", "action": "TRADE", "group": "energy", "reasons": []}
        now = {"ticker": "NOW", "action": "TRADE", "group": "software", "reasons": []}
        self.assertEqual(apply_same_group_book_park(xle, {"energy", "software"}, {"CVX", "SHOP"}), "same_group_in_book")
        self.assertEqual(xle["action"], "TRADE")
        self.assertTrue(xle.get("book_group_held"))
        self.assertIsNone(apply_same_group_book_park(cvx, {"energy", "software"}, {"CVX", "SHOP"}))
        self.assertEqual(cvx["action"], "TRADE")
        self.assertEqual(apply_same_group_book_park(now, {"energy", "software"}, {"CVX", "SHOP"}), "same_group_in_book")
        self.assertEqual(now["action"], "TRADE")
        self.assertTrue(now.get("book_group_held"))

    def test_analog_0win_needs_n4_and_no_wins(self):
        self.assertEqual(
            analog_0win_reason({"n": 4, "wins": 0, "avg_r": -0.36}),
            "analog_0win_veto",
        )
        self.assertEqual(
            analog_0win_reason({"stock": {"n": 4, "wins": 0, "losses": 2, "time": 2, "avg_r": -0.36}}),
            "analog_0win_veto",
        )
        self.assertIsNone(analog_0win_reason({"n": 1, "wins": 0, "avg_r": -1.0}))
        self.assertIsNone(analog_0win_reason({"n": 3, "wins": 0, "avg_r": -0.50}))
        self.assertIsNone(analog_0win_reason({"n": 4, "wins": 1, "avg_r": -0.20}))
        self.assertIsNone(analog_0win_reason({"n": 4, "wins": 2, "avg_r": 0.10}))
        self.assertIsNone(analog_0win_reason({"n": 4, "wins": 0, "avg_r": 0.05}))
        self.assertIsNone(analog_0win_reason({"n": 0, "wins": 0, "avg_r": None}))
        self.assertIsNone(analog_0win_reason(None))

    def test_already_held_same_right_not_opposite(self):
        picked = {"instrument": "debit_call_spread", "legs": "BUY 190c / SELL 200c"}
        self.assertEqual(
            already_held_same_right_reason(picked, [{"right": "call", "quantity": 1, "strike": 190}]),
            "already_held_calls",
        )
        self.assertIsNone(
            already_held_same_right_reason(picked, [{"right": "put", "quantity": 1, "strike": 110}])
        )
        self.assertIsNone(
            already_held_same_right_reason(picked, [{"right": "call", "quantity": 0, "strike": 190}])
        )
        self.assertIsNone(already_held_same_right_reason({"instrument": "stock"}, [{"right": "call", "quantity": 1}]))
        self.assertEqual(
            already_held_same_right_reason(
                {"instrument": "debit_put_spread"},
                [{"right": "put", "quantity": -1}],
            ),
            "already_held_puts",
        )

    def test_apply_parks_trade_to_watch_not_ignore(self):
        pltr = {
            "ticker": "PLTR",
            "action": "TRADE",
            "choice": "OPTIONS",
            "same_ticket": False,
            "picked": {"instrument": "debit_call_spread"},
            "schwab_legs": [{"right": "call", "quantity": 1, "expiry": "2026-09-25", "strike": 190.0}],
            "reasons": [],
        }
        self.assertEqual(apply_already_held_park(pltr), "already_held_calls")
        self.assertEqual(pltr["action"], "WATCH")
        self.assertIn("already_held_calls", pltr["reasons"])

        puts_only = {
            "ticker": "PLTR",
            "action": "TRADE",
            "choice": "OPTIONS",
            "same_ticket": False,
            "picked": {"instrument": "debit_call_spread"},
            "schwab_legs": [{"right": "put", "quantity": 1, "strike": 110.0}],
            "reasons": [],
        }
        self.assertIsNone(apply_already_held_park(puts_only))
        self.assertEqual(puts_only["action"], "TRADE")

        open_ticket = {
            "ticker": "CVX",
            "action": "TRADE",
            "choice": "OPTIONS",
            "same_ticket": True,
            "picked": {"instrument": "debit_call_spread"},
            "schwab_legs": [{"right": "call", "quantity": 1}],
            "reasons": [],
        }
        self.assertIsNone(apply_already_held_park(open_ticket))
        self.assertEqual(open_ticket["action"], "TRADE")

        stock = {
            "ticker": "NET",
            "action": "TRADE",
            "choice": "STOCK",
            "picked": {"instrument": "stock"},
            "schwab_legs": [{"right": "call", "quantity": 1}],
            "reasons": [],
        }
        self.assertIsNone(apply_already_held_park(stock))
        self.assertEqual(stock["action"], "TRADE")

        analog_row = {
            "ticker": "PLTR",
            "action": "TRADE",
            "choice": "OPTIONS",
            "evidence": {"stock": {"n": 4, "wins": 0, "avg_r": -0.36}},
            "reasons": [],
        }
        self.assertEqual(apply_analog_0win_park(analog_row), "analog_0win_veto")
        self.assertEqual(analog_row["action"], "WATCH")
        self.assertTrue(analog_row["evidence"]["weak"])

        one_win = {
            "ticker": "NET",
            "action": "TRADE",
            "choice": "STOCK",
            "evidence": {"stock": {"n": 4, "wins": 1, "avg_r": -0.05}},
            "reasons": [],
        }
        self.assertIsNone(apply_analog_0win_park(one_win))
        self.assertEqual(one_win["action"], "TRADE")

    def test_fast_stop_parks_shop_shape_not_net_shape(self):
        self.assertEqual(
            analog_fast_stop_reason({"n": 3, "wins": 1, "avg_r": 0.0, "fast_loss_n": 2}),
            "analog_fast_stop_veto",
        )
        self.assertIsNone(analog_fast_stop_reason({"n": 6, "wins": 1, "avg_r": -0.22, "fast_loss_n": 1}))
        self.assertIsNone(analog_fast_stop_reason({"n": 1, "wins": 0, "avg_r": -1.0, "fast_loss_n": 1}))
        self.assertIsNone(analog_fast_stop_reason({"n": 4, "wins": 2, "avg_r": 1.48, "fast_loss_n": 0}))
        shop = {
            "ticker": "SHOP",
            "action": "TRADE",
            "choice": "OPTIONS",
            "evidence": {"stock": {"n": 3, "wins": 1, "avg_r": 0.0, "fast_loss_n": 2}},
            "reasons": [],
        }
        self.assertEqual(apply_analog_0win_park(shop), "analog_fast_stop_veto")
        self.assertEqual(shop["action"], "WATCH")
        net = {
            "ticker": "NET",
            "action": "TRADE",
            "choice": "STOCK",
            "evidence": {"stock": {"n": 6, "wins": 1, "avg_r": -0.22, "fast_loss_n": 1}},
            "reasons": [],
        }
        self.assertIsNone(apply_analog_0win_park(net))
        self.assertEqual(net["action"], "TRADE")

    def test_below_ema_parks_bullish_options_not_stock(self):
        opt = {
            "ticker": "SHOP",
            "action": "TRADE",
            "choice": "OPTIONS",
            "direction": "bullish",
            "close": 142.61,
            "ema20": 145.88,
            "reasons": [],
        }
        self.assertEqual(apply_below_ema_park(opt), "below_20ema")
        self.assertEqual(opt["action"], "WATCH")
        stock = {
            "ticker": "ADBE",
            "action": "TRADE",
            "choice": "STOCK",
            "direction": "bullish",
            "close": 142.61,
            "ema20": 145.88,
            "reasons": [],
        }
        self.assertIsNone(apply_below_ema_park(stock))
        self.assertEqual(stock["action"], "TRADE")
        above = {
            "ticker": "XOM",
            "action": "TRADE",
            "choice": "OPTIONS",
            "direction": "bullish",
            "close": 162.61,
            "ema20": 159.32,
            "reasons": [],
        }
        self.assertIsNone(apply_below_ema_park(above))
        self.assertEqual(above["action"], "TRADE")

    def test_fill_guard_names_ema_and_debit(self):
        row = {
            "choice": "OPTIONS",
            "direction": "bullish",
            "ema20": 146.23,
            "avwap_swing_low": 131.14,
            "picked": {"target_debit": 4.0, "instrument": "debit_call_spread"},
        }
        stamp_fill_guard(row)
        self.assertIn("146.23", row["fill_note"])
        self.assertIn("4.00", row["fill_note"])
        self.assertIn("20 EMA", row["picked"]["invalidation"])
        self.assertIn("146.23", row["picked"]["invalidation"])
        self.assertEqual(row["fill_guard"]["stock_min"], 146.23)
        self.assertEqual(row["fill_guard"]["debit_max"], 4.0)
        self.assertIn("131.14", row["fill_note"])
        self.assertIn("AVWAP", row["fill_note"])

    def test_review_exits_call_debit_below_ema_even_without_book_stop(self):
        shop = open_trade_verdict(
            {"ticker": "SHOP", "instrument": "debit_call_spread", "direction": "bullish", "entry": 2.84},
            {"close": 142.12, "ema20": 145.83},
        )
        self.assertEqual(shop["verdict"], "EXIT")
        self.assertIn("20 EMA", shop["why"])
        cvx = open_trade_verdict(
            {"ticker": "CVX", "instrument": "debit_call_spread", "direction": "bullish", "stop": 199.27},
            {"close": 208.57, "ema20": 200.13},
        )
        self.assertEqual(cvx["verdict"], "HOLD")
        stock = open_trade_verdict(
            {"ticker": "ADBE", "instrument": "stock", "side": "long", "stop": 266.57},
            {"close": 270.0, "ema20": 280.0},
        )
        self.assertEqual(stock["verdict"], "HOLD")
