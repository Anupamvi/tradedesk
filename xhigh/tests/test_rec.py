import unittest

from xhigh.rec import classify, need_line


class TestRec(unittest.TestCase):
    def test_positive_ev_is_click(self):
        row = {"ev_proxy": 39.5, "pop_delta": 0.35, "conf": 70, "structure": "call_debit"}
        self.assertEqual(classify(row), "CLICK")

    def test_negative_ev_is_skip(self):
        row = {"ev_proxy": -112.0, "pop_delta": 0.88, "conf": 60, "structure": "put_credit"}
        self.assertEqual(classify(row), "SKIP")

    def test_csp_is_skip_even_if_high_pop(self):
        row = {"ev_proxy": -2451.0, "pop_delta": 0.87, "conf": 65, "structure": "csp"}
        self.assertEqual(classify(row), "SKIP")

    def test_missing_pop_is_watch(self):
        row = {"ev_proxy": 10, "pop_delta": None, "conf": 70}
        self.assertEqual(classify(row), "WATCH")

    def test_ko_need_line(self):
        row = {
            "structure": "call_debit",
            "long_strike": 89.0,
            "short_strike": 94.0,
            "debit": 1.34,
            "expiry_s": "25 Sep",
        }
        text = need_line(row)
        self.assertIn("90.34", text)
        self.assertIn("94", text)


if __name__ == "__main__":
    unittest.main()
