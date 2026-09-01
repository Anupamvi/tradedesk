import unittest

from cultra.statistics import two_way_clustered_positive_mean_p_value


class StatisticsV2Tests(unittest.TestCase):
    def test_two_way_positive_mean_test_is_deterministic(self):
        values = tuple(1.0 + (index % 3) * 0.1 for index in range(80))
        tickers = tuple("T%d" % (index % 8) for index in range(80))
        dates = tuple("D%d" % (index // 8) for index in range(80))
        first = two_way_clustered_positive_mean_p_value(
            values, tickers, dates, iterations=500, seed=17
        )
        second = two_way_clustered_positive_mean_p_value(
            values, tickers, dates, iterations=500, seed=17
        )
        self.assertEqual(first, second)
        self.assertLess(first, 0.05)

    def test_nonpositive_mean_cannot_be_significant_positive(self):
        self.assertEqual(
            1.0,
            two_way_clustered_positive_mean_p_value(
                (-1.0, -0.5, 0.0, 0.5),
                ("A", "A", "B", "B"),
                ("D1", "D2", "D1", "D2"),
                iterations=100,
            ),
        )


if __name__ == "__main__":
    unittest.main()
