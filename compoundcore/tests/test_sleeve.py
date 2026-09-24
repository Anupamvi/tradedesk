import unittest

from compoundcore.sleeve import (
    TICKER_ORDER,
    band_half_pp,
    bands,
    blended_fee,
    nvda_lookthrough,
    portfolio_rate,
    rates_from_weights,
    smh_crash_hit,
    us_share_of_equities,
    vxus_tax_drag_bps,
    weighted_block,
    weights,
)


class TestSleeve(unittest.TestCase):
    def test_weights_sum_to_one(self):
        for name in ("default", "aggressive"):
            total = sum(weights(name).values())
            self.assertAlmostEqual(total, 1.0, places=12)

    def test_default_weights(self):
        w = weights("default")
        self.assertEqual(
            [w[t] for t in TICKER_ORDER],
            [0.48, 0.10, 0.07, 0.05, 0.20, 0.05, 0.05],
        )

    def test_aggressive_weights(self):
        w = weights("aggressive")
        self.assertEqual(
            [w[t] for t in TICKER_ORDER],
            [0.45, 0.15, 0.10, 0.05, 0.15, 0.05, 0.05],
        )

    def test_blended_fee(self):
        self.assertAlmostEqual(blended_fee("default"), 0.000664, places=9)
        self.assertAlmostEqual(blended_fee("aggressive"), 0.000780, places=9)

    def test_nvda_lookthrough_default(self):
        # 0.48*7.55% + 0.10*16.2% + 0.07*21.94%
        self.assertAlmostEqual(nvda_lookthrough("default"), 0.0677978, places=6)

    def test_us_share_is_overweight_not_cap_neutral(self):
        self.assertAlmostEqual(us_share_of_equities("default"), 70 / 90, places=10)
        self.assertAlmostEqual(us_share_of_equities("aggressive"), 75 / 90, places=10)

    def test_smh_crash_is_single_digit(self):
        self.assertAlmostEqual(smh_crash_hit("default"), -0.0315, places=6)
        self.assertAlmostEqual(smh_crash_hit("aggressive"), -0.045, places=6)

    def test_bands_relative_with_two_pp_floor(self):
        self.assertEqual(band_half_pp(7.0), 2.0)  # 25% of 7 is 1.75, floor 2
        self.assertEqual(band_half_pp(48.0), 12.0)
        self.assertEqual(band_half_pp(5.0), 2.0)
        self.assertEqual(band_half_pp(10.0), 2.5)
        b = bands("default")
        self.assertAlmostEqual(b["SMH"].low, 0.05, places=10)
        self.assertAlmostEqual(b["SMH"].high, 0.09, places=10)
        self.assertAlmostEqual(b["VB"].low, 0.03, places=10)
        self.assertAlmostEqual(b["VB"].high, 0.07, places=10)
        self.assertAlmostEqual(b["VOO"].low, 0.36, places=10)
        self.assertAlmostEqual(b["VOO"].high, 0.60, places=10)
        self.assertAlmostEqual(b["VXUS"].low, 0.15, places=10)
        self.assertAlmostEqual(b["VXUS"].high, 0.25, places=10)

    def test_default_10y_base_matches_weighted_blocks(self):
        self.assertAlmostEqual(weighted_block("default", "base"), 0.05799, places=5)
        self.assertAlmostEqual(portfolio_rate("default", "10y", "base"), 0.058, places=9)
        self.assertAlmostEqual(portfolio_rate("default", "5y", "base"), 0.050, places=9)
        self.assertAlmostEqual(portfolio_rate("default", "5y", "stress"), -0.010, places=9)

    def test_vxus_tax_drag_bps(self):
        low = vxus_tax_drag_bps(0.15, sleeve="default")
        high = vxus_tax_drag_bps(0.24, sleeve="default")
        self.assertAlmostEqual(low, 9.0, places=6)
        self.assertAlmostEqual(high, 14.4, places=6)

    def test_rates_from_weights_matches_blocks(self):
        rates = rates_from_weights(weights("default"))
        self.assertAlmostEqual(rates["10y"]["base"], weighted_block("default", "base"), places=12)
        self.assertAlmostEqual(rates["5y"]["base"], rates["10y"]["base"] - 0.008, places=12)

    def test_unknown_sleeve(self):
        with self.assertRaises(ValueError):
            weights("tweet")
