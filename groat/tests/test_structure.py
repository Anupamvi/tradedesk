import unittest

from groat.config import DTE_MAX, DTE_MIN, STRIKE_DTE, quote_width_cap
from groat.structure import _iv_flags, choose, debit_spread, quote_ok, stock_plan
from groat.technicals import snapshot
from tests.barsutil import trend_bars


def _strike(dte=45, expiry="2026-10-10", strike=155.0, delta=0.55, bid=4.8, ask=4.95, oi=800, spot=155.0):
    return {
        "strike": strike,
        "dte": dte,
        "expirDate": expiry,
        "stockPrice": spot,
        "spotPrice": spot,
        "delta": delta,
        "gamma": 0.03,
        "theta": -0.04,
        "vega": 0.12,
        "callBidPrice": bid,
        "callAskPrice": ask,
        "callOpenInterest": oi,
        "callVolume": 200,
        "putBidPrice": bid - 0.1,
        "putAskPrice": ask,
        "putOpenInterest": oi,
        "putVolume": 150,
    }


class TestStructure(unittest.TestCase):
    def test_stock_plan_rr(self):
        bars = trend_bars(220, end="2026-08-26", slope=0.3, pullback=0.8)
        snap = snapshot(bars, "2026-08-26")
        plan = stock_plan(snap, "bullish")
        self.assertTrue(plan["ok"])
        self.assertGreaterEqual(plan["rr"], 1.2)
        self.assertGreaterEqual(plan["shares"], 1)

    def test_earnings_unavailable_blocks_options(self):
        bars = trend_bars(220, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        vol = {"iv30": 20.0, "hv20": 28.0, "vrp": -8.0, "forecast_20d": 26.0}
        earn = {"usable": False, "source": "DATA UNAVAILABLE", "overlaps_hold": False, "date": None}
        out = choose(snap, "bullish", vol, [_strike(spot=snap["close"], strike=snap["close"])], earn)
        self.assertEqual(out["options_block"].split("—")[0].strip()[:8], "earnings")
        self.assertIn(out["choice"], ("STOCK", "NO TRADE"))

    def test_cheap_iv_allows_long_when_earnings_clear(self):
        bars = trend_bars(220, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        vol = {"iv30": 18.0, "hv20": 28.0, "vrp": -10.0, "forecast_20d": 26.0}
        earn = {
            "usable": True,
            "source": "orats.nextErn",
            "overlaps_hold": False,
            "date": "2026-12-01",
        }
        spot = snap["close"]
        strikes = [
            _strike(spot=spot, strike=spot, delta=0.55),
            _strike(spot=spot, strike=spot * 1.05, delta=0.35, bid=2.4, ask=2.5),
            _strike(spot=spot, strike=spot * 0.95, delta=0.70, bid=7.4, ask=7.55),
        ]
        out = choose(snap, "bullish", vol, strikes, earn, setup={"primary": "A", "chase": False})
        self.assertIn(out["choice"], ("STOCK", "OPTIONS"))
        self.assertTrue(out["iv_cheap"])

    def test_do_not_invent_vol(self):
        bars = trend_bars(220, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        out = choose(
            snap,
            "bullish",
            {},
            [],
            {"usable": True, "source": "exempt", "overlaps_hold": False, "date": None},
            setup={"primary": "A", "chase": False},
        )
        self.assertTrue(out["vol_missing"])
        self.assertIn(out["choice"], ("STOCK", "NO TRADE"))

    def test_strike_dte_is_hold_window_range(self):
        from groat.config import DTE_WEEKLY_FLOOR

        parts = [int(x) for x in STRIKE_DTE.split(",")]
        self.assertEqual(parts, [DTE_WEEKLY_FLOOR, DTE_MAX])
        self.assertEqual(DTE_MIN, 21)

    def test_quote_width_cap_is_floor_not_ceiling(self):
        self.assertAlmostEqual(quote_width_cap(1.0), 0.20)
        self.assertAlmostEqual(quote_width_cap(7.8), 7.8 * 0.08)

    def test_quote_ok_at_cap_survives_float_width(self):
        self.assertTrue(quote_ok(7.7, 7.9, 800))

    def test_debit_spread_sizes_when_atm_long_cannot(self):
        spot = 185.0
        earn = {
            "usable": True,
            "source": "orats.nextErn",
            "overlaps_hold": False,
            "date": "2026-12-01",
        }
        strikes = [
            _strike(dte=45, expiry="2026-10-16", strike=185, delta=0.50, bid=7.7, ask=7.9, oi=800, spot=spot),
            _strike(dte=45, expiry="2026-10-16", strike=190, delta=0.40, bid=4.8, ask=5.0, oi=600, spot=spot),
            _strike(dte=45, expiry="2026-10-16", strike=195, delta=0.30, bid=3.3, ask=3.5, oi=400, spot=spot),
        ]
        debit = debit_spread(strikes, "bullish", earn)
        self.assertIsNotNone(debit)
        self.assertTrue(debit["ok"])
        self.assertGreaterEqual(debit["contracts"], 1)
        self.assertLessEqual(debit["debit"] * 100, 500)
        self.assertEqual(debit["long_strike"], 190.0)
        self.assertEqual(debit["short_strike"], 195.0)

        bars = trend_bars(220, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        vol = {"iv30": 18.0, "hv20": 28.0, "vrp": -10.0, "forecast_20d": 26.0}
        out = choose(snap, "bullish", vol, strikes, earn, setup={"primary": "E", "chase": False})
        self.assertEqual(out["choice"], "OPTIONS")
        self.assertEqual(out["options"]["instrument"], "debit_call_spread")

    def test_passed_debit_is_the_ticket_even_when_iv_is_fair(self):
        spot = 185.0
        earn = {
            "usable": True,
            "source": "orats.nextErn",
            "overlaps_hold": False,
            "date": "2026-12-01",
        }
        strikes = [
            _strike(dte=45, expiry="2026-10-16", strike=185, delta=0.50, bid=7.7, ask=7.9, oi=800, spot=spot),
            _strike(dte=45, expiry="2026-10-16", strike=190, delta=0.40, bid=4.8, ask=5.0, oi=600, spot=spot),
            _strike(dte=45, expiry="2026-10-16", strike=195, delta=0.30, bid=3.3, ask=3.5, oi=400, spot=spot),
        ]
        bars = trend_bars(220, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        vol = {"iv30": 28.0, "hv20": 26.0, "vrp": 2.0, "forecast_20d": 26.0}
        out = choose(snap, "bullish", vol, strikes, earn, setup={"primary": "E", "chase": False})
        self.assertEqual(out["choice"], "OPTIONS")
        self.assertIsNotNone(out["options"].get("target_debit"))

    def test_iv_flags_vrp_is_xor(self):
        rich, cheap = _iv_flags({"vrp": -12.1, "iv30": 43.4, "forecast_20d": 39.9})
        self.assertFalse(rich)
        self.assertTrue(cheap)
        rich, cheap = _iv_flags({"vrp": 5.0, "iv30": 20.0, "forecast_20d": 30.0})
        self.assertTrue(rich)
        self.assertFalse(cheap)
        rich, cheap = _iv_flags({"vrp": 1.0, "iv30": 24.0, "forecast_20d": 23.0})
        self.assertFalse(rich)
        self.assertFalse(cheap)

    def test_lottery_otm_debit_is_rejected(self):
        spot = 147.0
        earn = {"usable": True, "source": "exempt", "overlaps_hold": False, "date": None}
        far = [
            _strike(dte=45, expiry="2026-10-16", strike=160, delta=0.33, bid=1.4, ask=1.6, oi=400, spot=spot),
            _strike(dte=45, expiry="2026-10-16", strike=165, delta=0.26, bid=0.8, ask=1.0, oi=400, spot=spot),
        ]
        out = debit_spread(far, "bullish", earn)
        self.assertTrue(out)
        self.assertFalse(out.get("ok"))
        self.assertIn("OTM", out.get("reason") or "")

    def test_five_pct_otm_debit_is_rejected_even_with_delta(self):
        spot = 100.0
        earn = {"usable": True, "source": "exempt", "overlaps_hold": False, "date": None}
        far = [
            _strike(dte=45, expiry="2026-10-16", strike=105.5, delta=0.45, bid=2.4, ask=2.6, oi=400, spot=spot),
            _strike(dte=45, expiry="2026-10-16", strike=110.5, delta=0.28, bid=1.1, ask=1.3, oi=400, spot=spot),
        ]
        out = debit_spread(far, "bullish", earn)
        self.assertTrue(out)
        self.assertFalse(out.get("ok"))
        self.assertIn("OTM", out.get("reason") or "")

    def test_near_money_tiny_net_delta_is_rejected(self):
        spot = 275.61
        earn = {"usable": True, "source": "exempt", "overlaps_hold": False, "date": None}
        thin = [
            _strike(dte=28, expiry="2026-10-16", strike=280, delta=0.40, bid=8.4, ask=8.6, oi=400, spot=spot),
            _strike(dte=28, expiry="2026-10-16", strike=290, delta=0.33, bid=4.1, ask=4.3, oi=400, spot=spot),
        ]
        out = debit_spread(thin, "bullish", earn)
        self.assertTrue(out)
        self.assertFalse(out.get("ok"))
        self.assertIn("OTM", out.get("reason") or "")

    def test_pre_earnings_weekly_debit_is_allowed(self):
        spot = 440.0
        earn = {
            "usable": True,
            "source": "orats.nextErn",
            "overlaps_hold": False,
            "date": "2026-10-15",
            "days": 24,
        }
        # 18 DTE expires before Oct 15 — must be usable for TSM-style names.
        strikes = [
            _strike(dte=18, expiry="2026-10-09", strike=445, delta=0.48, bid=7.4, ask=7.6, oi=400, spot=spot),
            _strike(dte=18, expiry="2026-10-09", strike=455, delta=0.36, bid=3.3, ask=3.5, oi=400, spot=spot),
            _strike(dte=45, expiry="2026-11-20", strike=445, delta=0.50, bid=18.0, ask=18.4, oi=400, spot=spot),
            _strike(dte=45, expiry="2026-11-20", strike=455, delta=0.42, bid=12.0, ask=12.4, oi=400, spot=spot),
        ]
        out = debit_spread(strikes, "bullish", earn)
        self.assertTrue(out.get("ok"))
        self.assertEqual(out["expiry"], "2026-10-09")
        self.assertLess(out["expiry"], earn["date"])

    def test_far_earnings_does_not_use_weeklies(self):
        spot = 245.0
        earn = {
            "usable": True,
            "source": "orats.nextErn",
            "overlaps_hold": False,
            "date": "2026-12-01",
            "days": 71,
        }
        weeklies = [
            _strike(dte=11, expiry="2026-10-02", strike=250, delta=0.48, bid=5.4, ask=5.6, oi=400, spot=spot),
            _strike(dte=11, expiry="2026-10-02", strike=260, delta=0.33, bid=1.8, ask=2.0, oi=400, spot=spot),
        ]
        out = debit_spread(weeklies, "bullish", earn)
        self.assertTrue(out)
        self.assertFalse(out.get("ok"))

    def test_not_requested_does_not_claim_priced_structures(self):
        bars = trend_bars(220, end="2026-08-26")
        snap = snapshot(bars, "2026-08-26")
        vol = {"iv30": 20.0, "hv20": 25.0, "vrp": -5.0}
        earn = {"usable": True, "source": "exempt", "overlaps_hold": False, "date": "2026-12-01"}
        out = choose(snap, "bullish", vol, [], earn, setup={"primary": "D", "chase": False}, chain_status="not_requested")
        blob = " ".join(out.get("why") or [])
        self.assertIn("not requested", blob)
        self.assertNotIn("priced option structures", blob)


if __name__ == "__main__":
    unittest.main()
