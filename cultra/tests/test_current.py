import unittest

from cultra.current import CurrentResearchError, _iv_to_realized_ratio


class CurrentRatioTests(unittest.TestCase):
    def test_long_option_value_filter_uses_iv_over_realized(self):
        self.assertAlmostEqual(1.25, _iv_to_realized_ratio(0.25, 0.20))
        self.assertAlmostEqual(0.80, _iv_to_realized_ratio(0.20, 0.25))

    def test_volatility_ratio_rejects_nonpositive_inputs(self):
        with self.assertRaises(CurrentResearchError):
            _iv_to_realized_ratio(0.20, 0.0)


if __name__ == "__main__":
    unittest.main()
