import unittest
from datetime import date, datetime, timezone

from cultra.domain import LegAction, LegQuote, OptionLeg, OptionType
from cultra.economics import EconomicsError, same_expiry_payoff_envelope
from cultra.edge import CostBreakdown, PriceConvention


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)
EXPIRATION = date(2026, 10, 16)


def leg(symbol, action, option_type, strike, ratio=1):
    return OptionLeg(symbol, action, option_type, EXPIRATION, strike, ratio)


class ExactLegEconomicsTests(unittest.TestCase):
    def test_vertical_economics_derive_from_natural_quotes_and_costs(self):
        long_leg = leg("XYZ   261016C00100000", LegAction.BUY, OptionType.CALL, 100.0)
        short_leg = leg("XYZ   261016C00105000", LegAction.SELL, OptionType.CALL, 105.0)
        payoff = same_expiry_payoff_envelope(
            (long_leg, short_leg),
            (
                LegQuote(long_leg.occ_symbol, 1.90, 2.00, NOW),
                LegQuote(short_leg.occ_symbol, 0.90, 1.00, NOW),
            ),
            CostBreakdown(1.30, 0.10, 2.0, model_version="cost-v1"),
        )
        self.assertIs(payoff.price_convention, PriceConvention.DEBIT)
        self.assertAlmostEqual(payoff.executable_price, 1.10)
        self.assertAlmostEqual(payoff.maximum_loss, 113.40)
        self.assertAlmostEqual(payoff.maximum_profit, 386.60)
        self.assertAlmostEqual(payoff.breakevens[0], 101.134)

    def test_uncovered_call_is_structurally_undefined_risk(self):
        naked = leg("XYZ   261016C00105000", LegAction.SELL, OptionType.CALL, 105.0)
        with self.assertRaises(EconomicsError) as caught:
            same_expiry_payoff_envelope(
                (naked,),
                (LegQuote(naked.occ_symbol, 0.90, 1.00, NOW),),
                CostBreakdown(1.0, 0.1, 1.0, model_version="cost-v1"),
            )
        self.assertIn("undefined", str(caught.exception))

    def test_term_structure_requires_pathwise_model(self):
        later = date(2026, 11, 20)
        front = leg("XYZ   261016C00100000", LegAction.SELL, OptionType.CALL, 100.0)
        back = OptionLeg(
            "XYZ   261120C00100000",
            LegAction.BUY,
            OptionType.CALL,
            later,
            100.0,
        )
        with self.assertRaises(EconomicsError) as caught:
            same_expiry_payoff_envelope(
                (back, front),
                (
                    LegQuote(back.occ_symbol, 2.0, 2.1, NOW),
                    LegQuote(front.occ_symbol, 0.9, 1.0, NOW),
                ),
                CostBreakdown(1.0, 0.1, 1.0, model_version="cost-v1"),
            )
        self.assertIn("pathwise", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
