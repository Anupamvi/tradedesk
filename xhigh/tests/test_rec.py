import unittest

from xhigh.gates import load_gates
from xhigh.rec import classify, need_line
from xhigh.score import csp_annualized, ev_proxy

GATES = load_gates()


class TestRec(unittest.TestCase):
    def test_positive_ev_debit_is_click(self):
        row = {"ev_proxy": 39.5, "pop_delta": 0.35, "conf": 70, "structure": "call_debit"}
        self.assertEqual(classify(row, GATES), "CLICK")

    def test_skinny_put_credit_is_skip(self):
        row = {
            "ev_proxy": -112.0,
            "pop_delta": 0.88,
            "conf": 60,
            "structure": "put_credit",
            "credit": 0.71,
            "width": 15.0,
        }
        self.assertEqual(classify(row, GATES), "SKIP")

    def test_fat_put_credit_can_click(self):
        row = {
            "ev_proxy": -35.0,
            "pop_delta": 0.75,
            "conf": 60,
            "structure": "put_credit",
            "credit": 0.90,
            "width": 5.0,
        }
        self.assertEqual(classify(row, GATES), "CLICK")

    def test_thin_csp_is_skip(self):
        row = {
            "structure": "csp",
            "credit": 1.36,
            "strike": 290.0,
            "dte": 32,
            "pop_delta": 0.88,
            "delta": -0.12,
            "conf": 65,
            "last": 317.0,
        }
        self.assertLess(csp_annualized(row), 0.08)
        self.assertEqual(classify(row, GATES), "SKIP")

    def test_short_dte_csp_clicks_on_annualized(self):
        row = {
            "structure": "csp",
            "credit": 1.25,
            "strike": 200.0,
            "dte": 25,
            "pop_delta": 0.87,
            "delta": -0.13,
            "conf": 60,
            "last": 220.27,
        }
        self.assertGreaterEqual(csp_annualized(row), 0.08)
        self.assertEqual(classify(row, GATES), "CLICK")

    def test_put_credit_clicks_when_six_month_low_blocks_naked_csp(self):
        row = {
            "structure": "put_credit",
            "ev_proxy": -149.0,
            "pop_delta": 0.83,
            "conf": 60,
            "credit": 1.06,
            "width": 15.0,
            "short_strike": 200.0,
            "long_strike": 185.0,
            "last": 220.27,
            "low_126": 164.27,
        }
        self.assertEqual(classify(row, GATES), "CLICK")

    def test_skinny_put_credit_still_skip_with_six_month_low(self):
        row = {
            "structure": "put_credit",
            "ev_proxy": -112.0,
            "pop_delta": 0.88,
            "conf": 60,
            "credit": 0.71,
            "width": 15.0,
            "short_strike": 235.0,
            "last": 260.2,
            "low_126": 199.14,
        }
        self.assertEqual(classify(row, GATES), "SKIP")

    def test_csp_skips_if_six_month_low_through_strike(self):
        row = {
            "structure": "csp",
            "credit": 2.04,
            "strike": 200.0,
            "dte": 39,
            "pop_delta": 0.87,
            "delta": -0.13,
            "conf": 65,
            "last": 220.27,
            "low_126": 140.0,
        }
        self.assertEqual(classify(row, GATES), "SKIP")

    def test_csp_half_price_stress_is_large_loss(self):
        from xhigh.rec import wheel_stress

        px, pnl = wheel_stress({"last": 220.0, "strike": 200.0, "credit": 1.25})
        self.assertAlmostEqual(px, 110.0, places=1)
        self.assertLess(pnl, -8000)

    def test_paid_csp_is_click(self):
        row = {
            "structure": "csp",
            "credit": 2.04,
            "strike": 200.0,
            "dte": 39,
            "pop_delta": 0.87,
            "delta": -0.13,
            "conf": 65,
            "last": 220.27,
        }
        self.assertGreaterEqual(csp_annualized(row), 0.08)
        self.assertEqual(classify(row, GATES), "CLICK")
        self.assertGreater(ev_proxy(row, 0.87), 0)

    def test_csp_ev_is_not_full_strike(self):
        row = {"structure": "csp", "credit": 1.25, "strike": 200.0, "dte": 25, "pop_delta": 0.87}
        ev = ev_proxy(row, 0.87)
        self.assertIsNotNone(ev)
        self.assertLess(ev, 50)
        self.assertGreater(ev, 0)

    def test_missing_pop_is_watch(self):
        row = {"ev_proxy": 10, "pop_delta": None, "conf": 70, "structure": "call_debit"}
        self.assertEqual(classify(row, GATES), "WATCH")

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
