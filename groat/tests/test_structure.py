import unittest

from groat.structure import choose, stock_plan
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


if __name__ == "__main__":
    unittest.main()
