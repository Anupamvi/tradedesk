import unittest
from datetime import date

from cultra.opportunities import _economics, _plain_leg, _scenario_metrics


def option(symbol, strike, bid, ask, option_type="CALL"):
    return {
        "action": "BUY",
        "ratio": 1,
        "occ_symbol": symbol,
        "expiration": "2026-09-25",
        "strike": strike,
        "option_type": option_type,
        "bid": bid,
        "ask": ask,
        "quote_timestamp": "2026-08-28T20:00:00+00:00",
        "volume": 100,
        "open_interest": 1000,
        "delta_market_heuristic_not_pop": 0.5,
        "relative_spread": (ask - bid) / ((ask + bid) / 2),
    }


class OpportunityTests(unittest.TestCase):
    def test_human_leg_never_needs_occ_identifier(self):
        leg = option("AAPL260925C00200000", 200, 4.0, 4.2)
        text = _plain_leg(leg)
        self.assertEqual("Buy 1x Sep 25 $200 call", text)
        self.assertNotIn("AAPL260925", text)

    def test_finite_risk_economics_and_scenarios_are_deterministic(self):
        long_leg = option("AAPL260925C00200000", 200, 4.0, 4.2)
        short_leg = option("AAPL260925C00210000", 210, 1.9, 2.1)
        short_leg["action"] = "SELL"
        legs = [long_leg, short_leg]
        economics = _economics(legs, 202.0)
        self.assertIsNotNone(economics)
        self.assertGreater(economics["maximum_loss"], 0)
        self.assertGreater(economics["maximum_profit"], 0)
        args = dict(
            ticker="AAPL",
            family="CALL_DEBIT_SPREAD",
            expiry=date(2026, 9, 25),
            spot=202.0,
            legs=legs,
            economic_debit=economics["economic_entry_per_share"],
            drift=0.10,
            volatility=0.30,
        )
        self.assertEqual(_scenario_metrics(**args), _scenario_metrics(**args))


if __name__ == "__main__":
    unittest.main()
