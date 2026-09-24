import unittest

from wheelo.pd import PD_NOTE, attach_trade_pd, compute_pd, sort_by_pd
from wheelo.report import render_board


def _cand(ticker, conf, label, **extra):
    row = {
        "ticker": ticker,
        "conf": conf,
        "conf_label": label,
        "spot": 22.0,
        "credit_pct": 0.03,
        "otm_pct": 0.08,
        "x_status": "Quiet",
        "premium": {
            "csp_strike": extra.get("strike", 20.0),
            "csp_bid": extra.get("bid", 0.70),
            "expiry": "2026-10-16",
            "dte": 30,
            "iv_rank": 55,
            "spread_pct": extra.get("spread", 0.03),
            "put_oi": extra.get("oi", 40),
            "put_bid_size": extra.get("size", 10),
            "put_ask_size": extra.get("size", 10),
        },
        "contracts": extra.get("contracts", 2),
        "quote_age_sec": extra.get("age", 10),
    }
    row["premium"].update({k: v for k, v in extra.items() if k in ("quote_date",)})
    return row


class TestPdFormula(unittest.TestCase):
    def test_caps_r_at_3(self):
        pack = compute_pd(
            max_loss=100,
            planned_reward=500,
            planned_risk=100,
            liquidity_lots=8,
            quote_age_sec=4,
            spread_frac=0.02,
            size=5,
        )
        self.assertEqual(pack["R_cons"], 3.0)
        self.assertEqual(pack["pd"], 15.0)

    def test_stale_quote_pd_null(self):
        pack = compute_pd(
            max_loss=80,
            planned_reward=40,
            planned_risk=80,
            liquidity_lots=8,
            quote_age_sec=121,
            spread_frac=0.02,
            size=5,
        )
        self.assertIsNone(pack["pd"])
        self.assertEqual(pack["reason"], "DATA UNAVAILABLE")


class TestPdBoard(unittest.TestCase):
    def test_pd_next_to_conf_watch_unchanged(self):
        cheap = _cand("SOFI", 70, "TRADE", strike=4.0, bid=2.0, oi=20, size=10, contracts=6, age=8)
        rich = _cand("AMAT", 80, "TRADE", strike=200.0, bid=4.0, oi=5, size=5, contracts=1, age=8)
        watch = _cand("PLTR", 55, "WATCH", strike=150.0, bid=3.0)
        no_t = _cand("PYPL", 40, "NO_TRADE", strike=60.0, bid=1.0)
        attach_trade_pd(cheap)
        attach_trade_pd(rich)
        self.assertEqual(cheap["conf"], 70)
        self.assertIsNotNone(cheap["pd"])
        text = render_board(
            "2026-09-09",
            [rich, cheap, watch, no_t],
            35000,
            {"orats_http": 2, "shortlist_a": 2, "shortlist_b": 2, "shortlist_c": 2},
        )
        self.assertIn(PD_NOTE, text)
        self.assertIn("| Conf | Label | PD | N | R_cons | L |", text)
        self.assertIn("AMAT", text.split("**Rotation pick:**")[1].split("## TRADE")[0])
        trade_block = text.split("## TRADE")[1].split("## WATCH")[0]
        self.assertLess(trade_block.index("SOFI"), trade_block.index("AMAT"))
        watch_block = text.split("## WATCH")[1].split("## NO_TRADE")[0]
        self.assertIn("PLTR", watch_block)
        self.assertIn("## NO_TRADE", text)
        self.assertNotIn("| Conf | Label | PD | N | R_cons | L | Ticker | Spot | Put | Bid | Cr% | OTM | Expiry | DTE | IVR | X |\n|------|-------|----|---|--------|---|--------|------|-----|-----|-----|-----|--------|-----|-----|---|\n| 55 |", text.split("## WATCH")[1])


if __name__ == "__main__":
    unittest.main()
