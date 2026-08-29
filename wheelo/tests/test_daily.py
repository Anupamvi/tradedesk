import unittest

from wheelo.daily import evaluate_cc, evaluate_csp, evaluate_shares


CFG = {"management": {"close_target_pct": 0.5, "dte_roll_threshold": 14}}


class TestDailyMatrix(unittest.TestCase):
    def test_csp_profit_target(self):
        pos = {"ticker": "SOFI", "entry_premium": 1.0, "expiry": "2026-10-16"}
        act = evaluate_csp(pos, 0.40, 40, CFG)
        self.assertEqual(act["action"], "CLOSE")
        self.assertEqual(act["reason"], "profit_target")

    def test_csp_low_dte(self):
        pos = {"ticker": "SOFI", "entry_premium": 1.0}
        self.assertEqual(evaluate_csp(pos, 1.2, 10, CFG)["action"], "ROLL")
        self.assertEqual(evaluate_csp(pos, 0.8, 10, CFG)["action"], "CLOSE")

    def test_csp_closes_before_earnings_in_dte(self):
        pos = {"ticker": "SOFI", "entry_premium": 1.0, "expiry": "2026-09-25"}
        act = evaluate_csp(pos, 0.90, 28, CFG, earn_days=21)
        self.assertEqual(act["action"], "CLOSE")
        self.assertEqual(act["reason"], "earnings_in_dte")
        soon = evaluate_csp(pos, 0.90, 28, CFG, earn_days=5)
        self.assertEqual(soon["reason"], "earnings")

    def test_shares_and_cc(self):
        pos = {"ticker": "SOFI", "cost_basis": 20, "entry_premium": 0.8, "strike": 22}
        self.assertEqual(evaluate_shares(pos, 21, "bullish")["action"], "SELL_CC")
        self.assertEqual(evaluate_cc(pos, 0.3, 20, 21, CFG)["action"], "CLOSE")
        self.assertEqual(evaluate_cc(pos, 0.7, 5, 23, CFG)["action"], "ALLOW_CALL_AWAY")
