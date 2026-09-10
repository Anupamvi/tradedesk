import unittest
from pathlib import Path

from xhigh.geometry import catalog_for_name
from xhigh.pd import PD_NOTE, attach_trade_pd, compute_pd, sort_by_pd
from xhigh.rec import render_recommendation, sort_clicks
from xhigh.report import CLICK_COLS


class TestPdFormula(unittest.TestCase):
    def test_caps_r_at_3(self):
        pack = compute_pd(
            max_loss=100,
            planned_reward=900,
            planned_risk=100,
            liquidity_lots=10,
            quote_age_sec=5,
            spread_frac=0.02,
            size=5,
        )
        self.assertEqual(pack["R_cons"], 3.0)
        self.assertEqual(pack["pd"], 15.0)

    def test_stale_quote_pd_null(self):
        pack = compute_pd(
            max_loss=100,
            planned_reward=200,
            planned_risk=100,
            liquidity_lots=10,
            quote_age_sec=200,
            spread_frac=0.02,
            size=5,
        )
        self.assertIsNone(pack["pd"])
        self.assertEqual(pack["reason"], "DATA UNAVAILABLE")


class TestPdClick(unittest.TestCase):
    def test_pd_next_to_conf_and_sleeves_listed(self):
        debit = {
            "ticker": "MSFT",
            "structure": "call_debit",
            "action": "CLICK",
            "debit": 2.0,
            "max_gain": 8.0,
            "conf": 70,
            "liquidity_lots": 6,
            "spread_frac": 0.03,
            "quote_age_sec": 8,
            "pd_size": 5,
            "last": 400,
            "strategy": "BUY 400 C / SELL 410 C",
            "pop_s": "55%",
        }
        csp = {
            "ticker": "INTC",
            "structure": "csp",
            "action": "CLICK",
            "strike": 20,
            "credit": 0.80,
            "conf": 62,
            "liquidity_lots": 10,
            "spread_frac": 0.04,
            "quote_age_sec": 8,
            "pd_size": 5,
            "last": 23,
            "strategy": "SELL 20 P",
            "pop_s": "80%",
        }
        credit = {
            "ticker": "NVDA",
            "structure": "put_credit",
            "action": "CLICK",
            "credit": 1.5,
            "width": 5.0,
            "conf": 65,
            "liquidity_lots": 8,
            "spread_frac": 0.03,
            "quote_age_sec": 8,
            "pd_size": 5,
            "last": 180,
            "strategy": "SELL 165 P / BUY 160 P",
            "pop_s": "75%",
        }
        attach_trade_pd(debit)
        attach_trade_pd(csp)
        attach_trade_pd(credit)
        self.assertEqual(debit["conf"], 70)
        self.assertIsNotNone(debit["pd"])
        self.assertEqual(debit["r_cons"], 3.0)
        ordered = sort_clicks([csp, debit, credit])
        self.assertEqual(ordered[0]["structure"], "call_debit")
        names = {c[0] for c in CLICK_COLS}
        self.assertIn("conf", names)
        self.assertIn("PD", names)
        rec = "\n".join(render_recommendation("2026-09-09", ordered, [], []))
        self.assertIn(PD_NOTE, rec)
        self.assertIn("wheel 1", rec)
        self.assertIn("swing 1", rec)
        self.assertIn("credit 1", rec)

    def test_catalog_still_lists_all_sleeves(self):
        src = Path(catalog_for_name.__code__.co_filename).read_text(encoding="utf-8")
        for name in ("csp", "put_credit", "call_debit", "call_credit", "put_debit", "iron_condor"):
            self.assertIn('"%s"' % name, src)


if __name__ == "__main__":
    unittest.main()
