import unittest

from compoundcore.projections import fv_dca, fv_lump, path_table, path_table_from_rates, round_thousands
from compoundcore.sleeve import rates_from_weights, portfolio_rate, weights


class TestProjections(unittest.TestCase):
    def test_lump_matches_published_default_table(self):
        # Contract is the published $k table, not an off-by-one unrounded guess.
        published_k = [
            ("stress", 5, 95),
            ("stress", 10, 105),
            ("bear", 5, 110),
            ("bear", 10, 137),
            ("base", 5, 128),
            ("base", 10, 176),
            ("bull", 5, 157),
            ("bull", 10, 218),
        ]
        for scenario, years, expected_k in published_k:
            horizon = "5y" if years == 5 else "10y"
            annual = portfolio_rate("default", horizon, scenario)
            got = fv_lump(100000, annual, years)
            self.assertEqual(
                round_thousands(got),
                expected_k,
                msg="%s %sy" % (scenario, years),
            )

    def test_fantasy_and_voo_only(self):
        self.assertAlmostEqual(fv_lump(100000, 0.40, 5), 537824, delta=2)
        self.assertAlmostEqual(fv_lump(100000, 0.40, 10), 2892546, delta=5)
        voo = fv_lump(100000, 0.052, 10)
        self.assertEqual(round_thousands(voo), 166)

    def test_dca_matches_published_default_table(self):
        # $100k start + $1,000/month, end-of-month, (1+r)^(1/12)-1
        cases = [
            ("stress", 5, 154000),
            ("stress", 10, 228000),
            ("bear", 5, 173000),
            ("bear", 10, 278000),
            ("base", 5, 195000),
            ("base", 10, 337000),
            ("bull", 5, 233000),
            ("bull", 10, 399000),
        ]
        for scenario, years, rounded in cases:
            horizon = "5y" if years == 5 else "10y"
            annual = portfolio_rate("default", horizon, scenario)
            got = fv_dca(100000, 1000, annual, years)
            self.assertEqual(round_thousands(got) * 1000, rounded, msg="%s %sy" % (scenario, years))

    def test_million_is_ten_times_hundred_k(self):
        table = path_table(1000000, 0, "default")
        small = path_table(100000, 0, "default")
        for horizon in ("5y", "10y"):
            for scenario in ("stress", "bear", "base", "bull"):
                self.assertAlmostEqual(
                    table[horizon][scenario]["nominal"],
                    small[horizon][scenario]["nominal"] * 10,
                    places=4,
                )

    def test_real_base_ten_year(self):
        table = path_table(100000, 0, "default")
        self.assertEqual(round_thousands(table["10y"]["base"]["real"]), 144)
        self.assertEqual(round_thousands(table["10y"]["stress"]["real"]), 86)

    def test_aggressive_left_tail_worse_bull_better(self):
        d = path_table(100000, 0, "default")
        a = path_table(100000, 0, "aggressive")
        self.assertLess(
            a["10y"]["stress"]["nominal"],
            d["10y"]["stress"]["nominal"],
        )
        self.assertGreater(
            a["10y"]["bull"]["nominal"],
            d["10y"]["bull"]["nominal"],
        )

    def test_path_table_from_actual_mix(self):
        mix = dict(weights("default"))
        mix["SMH"] = 0.20
        mix["VOO"] = 0.35
        rates = rates_from_weights(mix)
        table = path_table_from_rates(100000, 0, rates)
        default = path_table(100000, 0, "default")
        self.assertLess(table["10y"]["stress"]["nominal"], default["10y"]["stress"]["nominal"])
        self.assertAlmostEqual(table["10y"]["base"]["annual"], rates["10y"]["base"], places=12)
