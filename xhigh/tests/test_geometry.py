import json
import unittest
from pathlib import Path

from xhigh.gates import load_gates
from xhigh.geometry import pick_call_debit, pick_csp, pick_iron_condor, pick_put_credit, spot_from_quote, ticket_legal
from xhigh.score import ev_proxy, pop_delta

GATES = load_gates()
EARN = {"usable": True, "date": "2026-11-15"}
ASOF = "2026-08-31"


class TestSpot(unittest.TestCase):
    def test_last_only(self):
        self.assertEqual(spot_from_quote({"last": 186.55, "bid": 1, "ask": 2}), 186.55)

    def test_bid_is_not_spot(self):
        self.assertIsNone(spot_from_quote({"bid": 186.0, "ask": 187.0, "last": 0}))

    def test_close_after_hours(self):
        self.assertEqual(spot_from_quote({"last": None, "close": 186.55}), 186.55)


class TestCspBand(unittest.TestCase):
    def test_intc_3pct_rejected(self):
        puts = [{"strike": 87.0, "bid": 4.25, "ask": 4.40, "expiry": "2026-10-02", "dte": 32, "side": "put", "delta": -0.45}]
        self.assertIsNone(pick_csp(puts, 89.6927, ASOF, EARN, GATES))
        self.assertFalse(ticket_legal("csp", 89.6927, 87.0, None, GATES))

    def test_nvda_10pct_accepted(self):
        puts = [{"strike": 198.0, "bid": 2.10, "ask": 2.25, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.18}]
        idea = pick_csp(puts, 220.0, ASOF, EARN, GATES)
        self.assertIsNotNone(idea)
        self.assertEqual(idea["strike"], 198.0)
        self.assertAlmostEqual(idea["otm"], 0.10, places=2)


class TestCallDebit(unittest.TestCase):
    def test_pltr_270_vs_186_is_none(self):
        last = 186.55
        lottery = [
            {"strike": 270.0, "bid": 1.00, "ask": 1.20, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.08},
            {"strike": 315.0, "bid": 0.40, "ask": 0.55, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.03},
        ]
        self.assertIsNone(pick_call_debit(lottery, last, ASOF, EARN, GATES))
        self.assertFalse(ticket_legal("call_debit", last, 270.0, 315.0, GATES))

    def test_pltr_near_spot_hits(self):
        last = 186.55
        chain = [
            {"strike": 185.0, "bid": 9.50, "ask": 9.80, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.55},
            {"strike": 198.0, "bid": 4.20, "ask": 4.40, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.32},
            {"strike": 270.0, "bid": 1.00, "ask": 1.20, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.08},
            {"strike": 315.0, "bid": 0.40, "ask": 0.55, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.03},
        ]
        idea = pick_call_debit(chain, last, ASOF, EARN, GATES)
        self.assertIsNotNone(idea)
        self.assertEqual(idea["long_strike"], 185.0)
        self.assertEqual(idea["short_strike"], 198.0)
        self.assertLess(idea["long_strike"] / last, 1.05)
        self.assertLess((idea["short_strike"] - idea["long_strike"]) / last, 0.07)

    def test_wide_msft_vertical_rejected(self):
        last = 507.78
        chain = [
            {"strike": 500.0, "bid": 22.0, "ask": 22.5, "expiry": "2026-10-02", "dte": 32, "side": "call", "delta": 0.55},
            {"strike": 560.0, "bid": 2.7, "ask": 2.9, "expiry": "2026-10-02", "dte": 32, "side": "call", "delta": 0.12},
        ]
        self.assertIsNone(pick_call_debit(chain, last, ASOF, EARN, GATES))


class TestPutCreditAndCondor(unittest.TestCase):
    def test_put_credit_near_atm_rejected(self):
        last = 100.0
        puts = [
            {"strike": 97.0, "bid": 2.0, "ask": 2.1, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.35},
            {"strike": 92.0, "bid": 0.8, "ask": 0.9, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.18},
        ]
        self.assertIsNone(pick_put_credit(puts, last, ASOF, EARN, GATES))

    def test_put_credit_10pct_hits(self):
        last = 100.0
        puts = [
            {"strike": 90.0, "bid": 1.40, "ask": 1.50, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.18},
            {"strike": 84.0, "bid": 0.50, "ask": 0.60, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.08},
        ]
        idea = pick_put_credit(puts, last, ASOF, EARN, GATES)
        self.assertIsNotNone(idea)
        self.assertEqual(idea["short_strike"], 90.0)
        self.assertEqual(idea["long_strike"], 84.0)

    def test_condor_far_call_rejected(self):
        last = 100.0
        puts = [
            {"strike": 90.0, "bid": 1.40, "ask": 1.50, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.18},
            {"strike": 84.0, "bid": 0.50, "ask": 0.60, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.08},
        ]
        calls = [
            {"strike": 140.0, "bid": 0.40, "ask": 0.50, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.05},
            {"strike": 146.0, "bid": 0.10, "ask": 0.20, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.02},
        ]
        self.assertIsNone(pick_iron_condor(puts, calls, last, ASOF, EARN, GATES))

    def test_condor_near_spot_hits(self):
        last = 100.0
        puts = [
            {"strike": 90.0, "bid": 1.40, "ask": 1.50, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.18},
            {"strike": 84.0, "bid": 0.50, "ask": 0.60, "expiry": "2026-10-09", "dte": 39, "side": "put", "delta": -0.08},
        ]
        calls = [
            {"strike": 106.0, "bid": 1.20, "ask": 1.30, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.18},
            {"strike": 112.0, "bid": 0.40, "ask": 0.50, "expiry": "2026-10-09", "dte": 39, "side": "call", "delta": 0.08},
        ]
        idea = pick_iron_condor(puts, calls, last, ASOF, EARN, GATES)
        self.assertIsNotNone(idea)
        self.assertEqual(idea["put_short"], 90.0)
        self.assertEqual(idea["call_short"], 106.0)


class TestScore(unittest.TestCase):
    def test_missing_delta_no_pop(self):
        self.assertIsNone(pop_delta({"structure": "csp", "strike": 200}))

    def test_csp_pop_from_delta(self):
        pop = pop_delta({"structure": "csp", "delta": -0.20})
        self.assertAlmostEqual(pop, 0.80, places=2)

    def test_ev_credit(self):
        idea = {"structure": "put_credit", "credit": 0.80, "width": 6.0}
        ev = ev_proxy(idea, 0.80)
        self.assertIsNotNone(ev)
        self.assertLess(ev, 0)


class TestGatesFile(unittest.TestCase):
    def test_otm_floor_is_eight_percent(self):
        path = Path(__file__).resolve().parent.parent / "configs" / "gates.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(data["csp"]["otm_min"], 0.08)
        self.assertLessEqual(data["call_debit"]["long_otm_max"], 0.04)
        self.assertLessEqual(data["call_debit"]["short_otm_max"], 0.08)
        self.assertLessEqual(data["max_width_frac"], 0.07)
        self.assertNotIn("new_paper", data.get("caps") or {})


if __name__ == "__main__":
    unittest.main()
