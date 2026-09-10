import unittest

from groat.pd import PD_NOTE, attach_trade_pd, compute_pd, sort_by_pd
from groat.report import render_board


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
        self.assertEqual(pack["N"], 5)
        self.assertEqual(pack["L"], 1.0)
        self.assertEqual(pack["pd"], 15.0)

    def test_stale_quote_pd_null(self):
        pack = compute_pd(
            max_loss=100,
            planned_reward=200,
            planned_risk=100,
            liquidity_lots=10,
            quote_age_sec=121,
            spread_frac=0.02,
            size=5,
        )
        self.assertIsNone(pack["pd"])
        self.assertEqual(pack["reason"], "DATA UNAVAILABLE")
        self.assertEqual(pack["N"], 5)

    def test_n_zero(self):
        pack = compute_pd(
            max_loss=1000,
            planned_reward=200,
            planned_risk=1000,
            liquidity_lots=10,
            quote_age_sec=5,
            spread_frac=0.02,
            size=1,
        )
        self.assertEqual(pack["N"], 0)
        self.assertIsNone(pack["pd"])
        self.assertEqual(pack["reason"], "N=0")


class TestPdBoard(unittest.TestCase):
    def _trade(self, ticker, pd, conf, score=70):
        row = {
            "ticker": ticker,
            "action": "TRADE",
            "choice": "OPTIONS",
            "primary": "D",
            "close": 100,
            "opt_conf": conf,
            "pd": pd,
            "pd_n": 2,
            "r_cons": 2.0,
            "pd_l": 1.0,
            "pd_reason": "" if pd is not None else "DATA UNAVAILABLE",
            "picked": {
                "instrument": "debit_call_spread",
                "long_strike": 100,
                "short_strike": 105,
                "expiry": "2026-10-16",
                "target_debit": 2.0,
            },
            "score": score,
        }
        attach_trade_pd(row) if pd is None else row
        return row

    def test_board_pd_next_to_conf_and_sort(self):
        low = {
            "ticker": "AAA",
            "action": "TRADE",
            "choice": "OPTIONS",
            "primary": "D",
            "close": 50,
            "opt_conf": 80,
            "quote_age_sec": 10,
            "spread_frac": 0.02,
            "liquidity_lots": 8,
            "pd_size": 4,
            "picked": {
                "instrument": "debit_call_spread",
                "long_strike": 50,
                "short_strike": 55,
                "expiry": "2026-10-16",
                "target_debit": 1.5,
                "max_loss_1lot": 100,
                "planned_reward": 200,
                "planned_risk": 100,
                "liquidity_lots": 8,
                "spread_frac": 0.02,
                "quote_age_sec": 10,
                "pd_size": 4,
                "contracts": 4,
            },
        }
        high = {
            "ticker": "BBB",
            "action": "TRADE",
            "choice": "OPTIONS",
            "primary": "E",
            "close": 80,
            "opt_conf": 55,
            "quote_age_sec": 10,
            "picked": {
                "instrument": "debit_call_spread",
                "long_strike": 80,
                "short_strike": 85,
                "expiry": "2026-10-16",
                "target_debit": 1.0,
                "max_loss_1lot": 100,
                "planned_reward": 300,
                "planned_risk": 100,
                "liquidity_lots": 10,
                "spread_frac": 0.02,
                "quote_age_sec": 10,
                "pd_size": 5,
                "contracts": 5,
            },
        }
        attach_trade_pd(low)
        attach_trade_pd(high)
        ordered = sort_by_pd([low, high], tie=lambda r: (-(r.get("opt_conf") or 0),))
        self.assertEqual([r["ticker"] for r in ordered], ["BBB", "AAA"])
        self.assertGreater(high["pd"], low["pd"])
        self.assertEqual(low["opt_conf"], 80)
        text = render_board(
            "2026-09-09",
            {"regime": {"regime": "unknown"}, "trades": ordered, "watch": [], "fire": [], "xhot": [], "picks": {}},
        )
        self.assertIn(PD_NOTE, text)
        self.assertIn("| conf | PD | N | R_cons | L |", text)
        self.assertIn("80", text)
        self.assertIn("## WATCH", text)
        # stock vs options still reviewed in structure path; board lists TRADE tickets
        self.assertIn("call debit", text)


if __name__ == "__main__":
    unittest.main()
