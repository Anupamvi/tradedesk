import unittest

from groat.confidence import options_confidence
from groat.picks import desk_picks, score_option_ticket
from groat.xhot import classify_xhot
from groat.setups import classify_setups
from groat.structure import choose, naive_pop_debit_vertical
from groat.technicals import snapshot
from groat.thesis import build_thesis
from tests.barsutil import trend_bars
from tests.test_structure import _strike


class TestFire(unittest.TestCase):
    def test_spike_needs_volume(self):
        bars = trend_bars(80, end="2026-08-26", slope=0.2)
        snap = snapshot(bars, "2026-08-26")
        snap["ret_1"] = 0.04
        snap["rvol"] = 2.0
        snap["extension_atr"] = 1.0
        setup = classify_setups(snap, group_row={"status": "accelerating"}, earnings={"usable": True, "source": "exempt"})
        self.assertIn("H", setup["setups"])
        self.assertEqual(setup["fire"]["kind"], "spike")
        self.assertEqual(setup["lane"], "FIRE")

    def test_big_gap_day_fires_at_rvol_12(self):
        bars = trend_bars(80, end="2026-08-26", slope=0.2)
        snap = snapshot(bars, "2026-08-26")
        snap["ret_1"] = 0.067
        snap["rvol"] = 1.48
        snap["extension_atr"] = 0.2
        setup = classify_setups(snap, group_row={"status": "deteriorating"}, earnings={"usable": True, "source": "exempt"})
        self.assertEqual(setup["fire"]["kind"], "spike")
        modest = dict(snap)
        modest["ret_1"] = 0.04
        modest["rvol"] = 1.40
        setup2 = classify_setups(modest, group_row={"status": "deteriorating"}, earnings={"usable": True, "source": "exempt"})
        self.assertIsNone(setup2["fire"]["kind"])


class TestAllStrategies(unittest.TestCase):
    def test_reviews_cover_stock_and_six_option_structures(self):
        bars = trend_bars(220, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        vol = {"iv30": 18.0, "hv20": 28.0, "vrp": -10.0, "forecast_20d": 26.0}
        earn = {"usable": True, "source": "orats.nextErn", "overlaps_hold": False, "date": "2026-12-01"}
        spot = snap["close"]
        strikes = [
            _strike(spot=spot, strike=spot, delta=0.55),
            _strike(spot=spot, strike=spot * 1.05, delta=0.35, bid=2.4, ask=2.5),
            _strike(spot=spot, strike=spot * 0.95, delta=0.70, bid=7.4, ask=7.55),
        ]
        out = choose(snap, "bullish", vol, strikes, earn, setup={"primary": "A", "chase": False})
        names = [r["strategy"] for r in out.get("reviews") or []]
        for needed in (
            "stock",
            "long_call",
            "long_put",
            "debit_call_spread",
            "debit_put_spread",
            "put_credit_spread",
            "call_credit_spread",
        ):
            self.assertIn(needed, names)
        puts = [r for r in out["reviews"] if r["strategy"] in ("long_put", "debit_put_spread", "call_credit_spread")]
        self.assertTrue(all(r["status"] == "REJECT" for r in puts))
        if out.get("options"):
            self.assertIsNotNone(out["options"].get("target_debit") or out["options"].get("target_credit"))


class TestNaivePop(unittest.TestCase):
    def test_debit_call_interpolates_breakeven(self):
        long_leg = {"strike": 235.0, "delta": 0.40}
        short_leg = {"strike": 240.0, "delta": 0.25}
        pop, note = naive_pop_debit_vertical(long_leg, short_leg, 1.80, "call")
        self.assertIsNotNone(pop)
        # BE 236.80 is 36% of the way from 235 to 240 → 0.40 - 0.15*0.36 = 0.346
        self.assertAlmostEqual(pop, 0.40 - 0.15 * (1.80 / 5.0), places=4)
        self.assertIn("Not a backtested win rate", note)


class TestConfidenceThesis(unittest.TestCase):
    def test_confidence_is_int_not_win_prob(self):
        picked = {
            "ok": True,
            "instrument": "debit_call_spread",
            "target_debit": 3.55,
            "oi": 400,
            "dte": 30,
            "rr": 1.8,
        }
        conf = options_confidence(
            picked,
            {"vrp": -8},
            {"usable": True, "days": 60, "overlaps_hold": False, "source": "web.alphaquery"},
            {"stale": False},
            setup={"primary": "E"},
            x_tag="DATA UNAVAILABLE",
        )
        self.assertIsInstance(conf["conf"], int)
        self.assertLessEqual(conf["conf"], 85)
        self.assertIn("not P(win)", conf["note"])

    def test_thesis_uses_tape(self):
        text = build_thesis(
            {
                "ticker": "PLTR",
                "direction": "bullish",
                "primary": "E",
                "regime": "strong_risk_on",
                "group": "software",
                "group_status": "accelerating",
                "trend": "up",
                "close": 184.88,
                "ema20": 167,
                "sma50": 142,
                "sma200": 151,
                "rs_20": 0.47,
                "rvol": 1.1,
                "choice": "OPTIONS",
                "earnings": {"date": "2026-11-02", "source": "web.alphaquery"},
                "picked": {"invalidation": "close beyond 164"},
            }
        )
        blob = " ".join(text["paragraphs"])
        self.assertIn("PLTR", blob)
        self.assertIn("accelerating", blob)
        self.assertIn("2026-11-02", blob)


class TestDeskPicksAndXhot(unittest.TestCase):
    def test_near_money_beats_crowded_low_delta(self):
        now = {
            "ticker": "NOW",
            "choice": "OPTIONS",
            "naive_pop": 0.43,
            "opt_conf": 70,
            "score": 61,
            "x": "Quiet",
            "ret_1": 0.10,
            "close": 138.43,
            "picked": {"long_strike": 140.0, "delta": 0.20, "instrument": "debit_call_spread", "legs": "BUY 140c"},
        }
        nvda = {
            "ticker": "NVDA",
            "choice": "OPTIONS",
            "naive_pop": 0.37,
            "opt_conf": 60,
            "score": 63,
            "x": "Crowded",
            "ret_1": 0.087,
            "close": 227.98,
            "picked": {"long_strike": 235.0, "delta": 0.08, "instrument": "debit_call_spread", "legs": "BUY 235c"},
        }
        self.assertGreater(score_option_ticket(now), score_option_ticket(nvda))
        picks = desk_picks([now, nvda, {"ticker": "ADBE", "choice": "STOCK", "score": 61, "x": "Quiet"}])
        self.assertEqual(picks["best_options"]["ticker"], "NOW")
        self.assertEqual(picks["best_stock"]["ticker"], "ADBE")
        from groat.picks import render_desk_picks
        from groat.report import render_board, render_report

        text = "\n".join(render_desk_picks(picks))
        self.assertIn("## Desk pick", text)
        self.assertIn("TRADE list: **NOW**", text)
        self.assertIn("Take options: NOW", text)
        self.assertIn("Why this one, not the others", text)
        board = render_board(
            "2026-08-27",
            {"regime": {"regime": "strong_risk_on"}, "trades": [now], "watch": [], "fire": [], "xhot": [], "picks": picks},
        )
        self.assertTrue(board.split("##")[1].strip().startswith("Desk pick"))
        report = render_report(
            "2026-08-27",
            {"regime": {"regime": "strong_risk_on"}, "trades": [now], "watch": [], "fire": [], "xhot": [], "picks": picks, "groups": []},
        )
        self.assertIn("## Desk pick", report.split("# Market regime")[0])

    def test_desk_pick_skips_in_book_for_take_this(self):
        cvx = {
            "ticker": "CVX",
            "choice": "OPTIONS",
            "in_book": True,
            "naive_pop": 0.50,
            "opt_conf": 80,
            "score": 70,
            "x": "Quiet",
            "ret_1": 0.02,
            "close": 210.0,
            "picked": {"long_strike": 210.0, "delta": 0.22, "instrument": "debit_call_spread", "legs": "BUY 210c", "target_debit": 2.73},
        }
        shop = {
            "ticker": "SHOP",
            "choice": "OPTIONS",
            "in_book": False,
            "naive_pop": 0.40,
            "opt_conf": 68,
            "score": 60,
            "x": "Quiet",
            "ret_1": 0.04,
            "close": 147.0,
            "picked": {"long_strike": 150.0, "delta": 0.18, "instrument": "debit_call_spread", "legs": "BUY 150c", "target_debit": 4.00},
        }
        self.assertGreater(score_option_ticket(cvx), score_option_ticket(shop))
        picks = desk_picks([cvx, shop])
        self.assertEqual(picks["best_options"]["ticker"], "SHOP")
        from groat.picks import render_desk_picks

        text = "\n".join(render_desk_picks(picks))
        self.assertIn("Take options: SHOP", text)
        self.assertIn("**CVX**", text)
        self.assertIn("IN BOOK", text)
        self.assertNotIn("Take options: CVX", text)

    def test_desk_pick_prints_do_not_click_band(self):
        shop = {
            "ticker": "SHOP",
            "choice": "OPTIONS",
            "in_book": False,
            "naive_pop": 0.42,
            "opt_conf": 70,
            "score": 55,
            "x": "Informed",
            "close": 147.47,
            "fill_note": "Do not click if last < **146.23** (20 EMA). Do not pay more than debit **4.00**.",
            "picked": {
                "long_strike": 150.0,
                "delta": 0.15,
                "instrument": "debit_call_spread",
                "legs": "BUY 150c",
                "target_debit": 4.00,
                "invalidation": "close back below 20 EMA 146.23 / swing-low AVWAP 131.14",
                "fill_note": "Do not click if last < **146.23** (20 EMA). Do not pay more than debit **4.00**.",
            },
        }
        picks = desk_picks([shop])
        from groat.picks import render_desk_picks

        text = "\n".join(render_desk_picks(picks))
        self.assertIn("Do not click this option", text)
        self.assertIn("146.23", text)
        self.assertIn("4.00", text)

    def test_board_watch_ticket_table_has_setup_strikes_and_skip_icon(self):
        from groat.report import render_board, render_report

        xlp = {
            "ticker": "XLP",
            "action": "WATCH",
            "choice": "OPTIONS",
            "primary": "E",
            "direction": "bullish",
            "close": 85.23,
            "ema20": 85.51,
            "reasons": ["below_trade_score"],
            "fill_guard": {"stock_min": 85.51, "debit_max": 1.68},
            "picked": {
                "instrument": "debit_call_spread",
                "long_strike": 85.0,
                "short_strike": 90.0,
                "expiry": "2026-10-16",
                "target_debit": 1.68,
            },
        }
        shop = {
            "ticker": "SHOP",
            "action": "TRADE",
            "choice": "OPTIONS",
            "primary": "A",
            "direction": "bullish",
            "close": 147.47,
            "ema20": 146.23,
            "fill_guard": {"stock_min": 146.23, "debit_max": 4.00},
            "picked": {
                "instrument": "debit_call_spread",
                "long_strike": 150.0,
                "short_strike": 160.0,
                "expiry": "2026-10-16",
                "target_debit": 4.00,
            },
        }
        built = {
            "regime": {"regime": "weak_risk_on"},
            "trades": [shop],
            "watch": [xlp],
            "fire": [],
            "xhot": [],
            "picks": {},
        }
        board = render_board("2026-09-02", built)
        self.assertNotIn("How to read this", board)
        self.assertIn("| | ticker | setup | ticket | pay | last | click | X |", board)
        self.assertIn("**XLP**", board)
        self.assertIn("Sector rotation", board)
        self.assertIn("Trend pullback", board)
        self.assertIn("RS leader", board)
        self.assertIn("hot group", board)
        self.assertIn("call debit", board)
        self.assertIn("85.0 / 90.0", board)
        self.assertIn("2026-10-16", board)
        self.assertIn("debit 1.68", board)
        self.assertIn("skip if last < **85.51**", board)
        self.assertIn("🔴", board)
        self.assertIn("**SHOP**", board)
        self.assertIn("skip if last < **146.23**", board)
        self.assertIn("🟢", board)
        report = render_report("2026-09-02", built)
        self.assertIn("**XLP**", report)
        self.assertNotIn("# Market regime", report)
        self.assertNotIn("How to read this", report)

    def test_desk_pick_in_book_only_stays_visible(self):
        cvx = {
            "ticker": "CVX",
            "choice": "OPTIONS",
            "in_book": True,
            "naive_pop": 0.50,
            "opt_conf": 80,
            "score": 70,
            "x": "Quiet",
            "close": 210.0,
            "picked": {"long_strike": 210.0, "delta": 0.22, "instrument": "debit_call_spread", "legs": "BUY 210c"},
        }
        picks = desk_picks([cvx])
        self.assertIsNone(picks["best_options"])
        from groat.picks import render_desk_picks

        text = "\n".join(render_desk_picks(picks))
        self.assertIn("in book", text.lower())
        self.assertIn("**CVX**", text)
        self.assertNotIn("Take options: CVX", text)

    def test_xhot_needs_tape(self):
        hot = {"bias": "bullish", "narrative": "loud on X", "tag": "Informed"}
        heat = classify_xhot(hot, {"ret_1": 0.01, "rvol": 1.0, "extension_atr": 1.0})
        self.assertEqual(heat["kind"], "heat_only")
        self.assertFalse(heat["playable"])
        self.assertEqual(heat["move"], "will_rise")
        dip = classify_xhot(hot, {"ret_1": -0.04, "rvol": 1.8, "extension_atr": 0.5})
        self.assertEqual(dip["kind"], "dip")
        self.assertTrue(dip["playable"])
        self.assertEqual(dip["move"], "dipped")
        chase = classify_xhot(hot, {"ret_1": 0.20, "rvol": 2.7, "extension_atr": 3.2})
        self.assertEqual(chase["move"], "will_dip")
        self.assertFalse(chase["playable"])
        dump = classify_xhot({"bias": "bearish", "tag": "Informed"}, {"ret_1": -0.04, "rvol": 1.8, "extension_atr": 0.5})
        self.assertEqual(dump["move"], "will_dip")
        self.assertTrue(dump["playable"])


if __name__ == "__main__":
    unittest.main()

