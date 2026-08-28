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

