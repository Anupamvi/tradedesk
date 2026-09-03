import unittest

from xhigh.gates import load_gates
from xhigh.rec import classify, decorate, need_line, sort_clicks, why_line
from xhigh.score import csp_annualized, ev_proxy

GATES = load_gates()


class TestRec(unittest.TestCase):
    def test_positive_ev_without_delta_is_watch(self):
        row = {"ev_proxy": 39.5, "pop_delta": 0.35, "conf": 70, "structure": "call_debit"}
        self.assertEqual(classify(row, GATES), "WATCH")

    def test_positive_ev_otm_debit_is_skip(self):
        row = {
            "ev_proxy": 46.1,
            "pop_delta": 0.393,
            "conf": 62,
            "structure": "put_debit",
            "long_delta": -0.393,
            "rr": 2.06,
            "debit": 2.29,
            "max_gain": 4.71,
            "dte": 31,
            "last": 109.18,
            "long_strike": 107.0,
            "short_strike": 100.0,
        }
        self.assertEqual(classify(row, GATES), "SKIP")

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
            "ev_proxy": 20.0,
            "pop_delta": 0.75,
            "conf": 60,
            "structure": "put_credit",
            "credit": 1.25,
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

    def test_call_debit_missing_div_is_watch(self):
        row = {
            "structure": "call_debit",
            "pop_delta": 0.55,
            "conf": 70,
            "long_delta": 0.55,
            "rr": 2.04,
            "debit": 4.93,
            "max_gain": 10.07,
            "dte": 38,
            "last": 100.0,
            "long_strike": 99.0,
            "short_strike": 106.0,
            "asof": "2026-09-02",
            "expiry": "2026-10-16",
        }
        self.assertEqual(classify(row, GATES), "WATCH")

    def test_put_credit_through_dividend_is_skip(self):
        row = {
            "structure": "put_credit",
            "pop_delta": 0.82,
            "conf": 70,
            "credit": 1.53,
            "width": 15.0,
            "short_strike": 200.0,
            "long_strike": 185.0,
            "last": 222.4,
            "asof": "2026-09-02",
            "expiry": "2026-10-16",
            "div_date": "2026-09-10",
        }
        self.assertEqual(classify(row, GATES), "SKIP")
        self.assertIn("ex-div", decorate(row, GATES)["why_s"])

    def test_paid_put_credit_clicks_when_six_month_low_blocks_csp(self):
        row = {
            "structure": "put_credit",
            "ev_proxy": -126.0,
            "pop_delta": 0.79,
            "conf": 60,
            "credit": 1.95,
            "width": 15.0,
            "short_strike": 200.0,
            "long_strike": 185.0,
            "last": 217.87,
            "low_126": 164.27,
        }
        self.assertEqual(classify(row, GATES), "CLICK")

    def test_near_atm_debit_clicks(self):
        row = {
            "structure": "call_debit",
            "ev_proxy": -104.5,
            "pop_delta": 0.55,
            "conf": 70,
            "long_delta": 0.55,
            "rr": 2.04,
            "debit": 4.93,
            "max_gain": 10.07,
            "dte": 38,
            "last": 100.0,
            "long_strike": 99.0,
            "short_strike": 106.0,
            "asof": "2026-09-02",
            "expiry": "2026-10-16",
            "div_date": "2026-11-01",
        }
        self.assertEqual(classify(row, GATES), "CLICK")

    def test_ko_style_debit_is_skip(self):
        row = {
            "structure": "call_debit",
            "pop_delta": 0.35,
            "conf": 70,
            "long_delta": 0.35,
            "rr": 2.8,
            "debit": 1.31,
            "max_gain": 3.69,
            "dte": 25,
            "last": 89.14,
            "long_strike": 89.0,
            "short_strike": 94.0,
            "asof": "2026-08-31",
            "expiry": "2026-09-25",
            "div_date": "2026-09-15",
        }
        self.assertEqual(classify(row, GATES), "SKIP")
        why = decorate(row, GATES)["why_s"]
        self.assertIn("DTE 25", why)
        self.assertIn("ex-div", why)

    def test_call_debit_through_dividend_is_skip(self):
        row = {
            "structure": "call_debit",
            "pop_delta": 0.55,
            "conf": 70,
            "long_delta": 0.55,
            "rr": 2.0,
            "debit": 2.0,
            "dte": 38,
            "last": 100.0,
            "long_strike": 99.0,
            "short_strike": 106.0,
            "asof": "2026-09-02",
            "expiry": "2026-10-09",
            "div_date": "2026-09-15",
        }
        self.assertEqual(classify(row, GATES), "SKIP")

    def test_put_debit_max_below_six_month_low_is_skip(self):
        row = {
            "structure": "put_debit",
            "pop_delta": 0.51,
            "conf": 62,
            "long_delta": -0.51,
            "rr": 2.0,
            "debit": 2.32,
            "dte": 38,
            "last": 105.96,
            "long_strike": 106.0,
            "short_strike": 99.0,
            "low_126": 102.15,
        }
        self.assertEqual(classify(row, GATES), "SKIP")

    def test_otm_long_debit_skips(self):
        row = {
            "structure": "call_debit",
            "ev_proxy": -47.6,
            "pop_delta": 0.10,
            "conf": 60,
            "long_delta": 0.278,
            "rr": 1.94,
            "debit": 0.68,
            "max_gain": 1.32,
            "dte": 38,
            "last": 100.0,
            "long_strike": 103.0,
            "short_strike": 107.0,
            "asof": "2026-09-02",
            "expiry": "2026-10-16",
            "div_date": "2026-11-01",
        }
        self.assertEqual(classify(row, GATES), "SKIP")

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

    def test_fat_credit_skip_blames_pop_not_width(self):
        row = decorate(
            {
                "structure": "call_credit",
                "pop_delta": 0.68,
                "conf": 62,
                "credit": 1.08,
                "width": 5.0,
                "short_delta": 0.32,
            },
            GATES,
        )
        self.assertEqual(row["action"], "SKIP")
        self.assertIn("POP 68%", row["why_s"])
        self.assertNotIn("need ≥10% of width", row["why_s"])

    def test_passing_credit_stays_click_next_to_debit(self):
        debit = decorate(
            {
                "ticker": "NVDA",
                "structure": "call_debit",
                "pop_delta": 0.42,
                "conf": 70,
                "long_delta": 0.415,
                "rr": 2.03,
                "debit": 3.3,
                "max_gain": 6.7,
                "dte": 30,
                "last": 219.94,
                "long_strike": 225.0,
                "short_strike": 235.0,
                "asof": "2026-09-02",
                "expiry": "2026-10-02",
                "div_date": "2026-12-01",
            },
            GATES,
        )
        credit = decorate(
            {
                "ticker": "NVDA",
                "structure": "put_credit",
                "pop_delta": 0.79,
                "conf": 70,
                "credit": 1.87,
                "width": 15.0,
                "short_strike": 200.0,
                "long_strike": 185.0,
                "low_126": 164.27,
            },
            GATES,
        )
        self.assertEqual(debit["action"], "SKIP")
        self.assertEqual(credit["action"], "CLICK")

    def test_debit_skip_why_does_not_blame_passing_rr(self):
        row = decorate(
            {
                "structure": "put_debit",
                "pop_delta": 0.393,
                "conf": 62,
                "long_delta": -0.393,
                "rr": 2.06,
                "debit": 2.29,
                "max_gain": 4.71,
                "dte": 31,
                "last": 109.18,
                "long_strike": 107.0,
                "short_strike": 100.0,
            },
            GATES,
        )
        self.assertEqual(row["action"], "SKIP")
        self.assertIn("|delta| 0.39", row["why_s"])
        self.assertNotIn("R/R 2.1", row["why_s"])

    def test_sort_clicks_small_risk_first(self):
        msft = {"structure": "call_debit", "debit": 11.65, "action": "CLICK"}
        nvda = {"structure": "call_debit", "debit": 3.30, "action": "CLICK"}
        ordered = sort_clicks([msft, nvda])
        self.assertEqual(ordered[0]["debit"], 3.30)

    def test_credit_click_why_does_not_claim_positive_ev(self):
        row = {
            "structure": "call_credit",
            "action": "CLICK",
            "credit": 0.63,
            "width": 6.0,
            "pop_delta": 0.755,
            "ev_proxy": -84.0,
        }
        text = why_line(row)
        self.assertNotIn("EV is positive", text)
        self.assertIn("of width", text)
        self.assertIn("10.5% of width", text)


if __name__ == "__main__":
    unittest.main()
