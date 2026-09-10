import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from pd_rank import PD_NOTE, attach_structure_pd, compute_pd, sort_by_pd  # noqa: E402


class TestPdFormula(unittest.TestCase):
    def test_caps_r_at_3(self):
        pack = compute_pd(
            max_loss=100,
            planned_reward=800,
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
            max_loss=150,
            planned_reward=200,
            planned_risk=150,
            liquidity_lots=6,
            quote_age_sec=180,
            spread_frac=0.04,
            size=3,
        )
        self.assertIsNone(pack["pd"])
        self.assertEqual(pack["reason"], "DATA UNAVAILABLE")


class TestGrokOptionCard(unittest.TestCase):
    def test_pd_next_to_conf_and_sleeves_listed(self):
        card = (ROOT / "assets" / "daily-card.md").read_text(encoding="utf-8")
        self.assertIn("Conf", card)
        self.assertIn("PD", card)
        self.assertIn("R_cons", card)
        self.assertIn(PD_NOTE, card)
        for sleeve in (
            "Sell put credit",
            "Sell call credit",
            "Sell iron condor",
            "Buy call debit",
            "Buy put debit",
            "Spike",
        ):
            self.assertIn(sleeve, card)

    def test_structure_pd_stamp_keeps_conf(self):
        row = {
            "ok": True,
            "action": "Sell put credit",
            "conf": 79,
            "kind": "credit",
            "pricing": {
                "kind": "credit",
                "net": 1.2,
                "width": 5.0,
                "max_loss_1lot": 380.0,
                "max_profit_1lot": 120.0,
            },
            "short": {"bid": 1.50, "ask": 1.55, "oi": 40, "bid_size": 8, "ask_size": 8},
            "long": {"bid": 0.25, "ask": 0.30, "oi": 20, "bid_size": 5, "ask_size": 5},
            "quote_age_sec": 9,
            "rec_lots": 2,
        }
        attach_structure_pd(row)
        self.assertEqual(row["conf"], 79)
        self.assertIsNotNone(row["pd"])
        stale = dict(row)
        stale["quote_age_sec"] = 400
        stale.pop("pd", None)
        attach_structure_pd(stale)
        self.assertIsNone(stale["pd"])
        a = {"pd": 3.0, "conf": 50}
        b = {"pd": None, "conf": 80, "pd_reason": "DATA UNAVAILABLE"}
        c = {"pd": 9.0, "conf": 40}
        ordered = sort_by_pd([a, b, c], tie=lambda r: (-(r.get("conf") or 0),))
        self.assertEqual([r["pd"] for r in ordered], [9.0, 3.0, None])


if __name__ == "__main__":
    unittest.main()
