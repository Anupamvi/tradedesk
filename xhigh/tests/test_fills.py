import unittest

from xhigh.gates import load_gates
from xhigh.geometry import pick_call_debit, pick_csp

GATES = load_gates()
EARN = {"usable": True, "date": "2026-11-15"}
ASOF = "2026-08-31"


class TestFills(unittest.TestCase):
    def test_csp_credit_is_bid(self):
        puts = [
            {"strike": 198.0, "bid": 2.10, "ask": 2.40, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.18}
        ]
        idea = pick_csp(puts, 220.0, ASOF, EARN, GATES)
        self.assertIsNotNone(idea)
        self.assertEqual(idea["credit"], 2.10)
        self.assertNotEqual(idea["credit"], 2.25)

    def test_debit_is_ask_minus_short_bid(self):
        last = 186.55
        chain = [
            {"strike": 185.0, "bid": 9.50, "ask": 9.80, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.55},
            {"strike": 198.0, "bid": 4.20, "ask": 4.40, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.32},
        ]
        idea = pick_call_debit(chain, last, ASOF, EARN, GATES)
        self.assertIsNotNone(idea)
        self.assertAlmostEqual(idea["debit"], 9.80 - 4.20, places=4)


if __name__ == "__main__":
    unittest.main()
