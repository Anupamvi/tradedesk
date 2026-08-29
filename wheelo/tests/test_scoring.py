import unittest

from wheelo.scoring import (
    allocate_capital,
    apply_sentiment,
    assign_tier,
    compute_composite,
    score_premium,
    score_quality,
    stage0_reason,
    tier_score,
)


CFG = {
    "universe": {"min_price": 10, "max_price": 60, "min_option_volume": 500, "min_market_cap_mm": 5000, "max_borrow": 25, "max_spread_pct": 0.20},
    "management": {"earnings_close_days": 7, "dte_target": 30, "sigma_otm": 1.0},
    "scoring": {
        "quality_weight": 0.7,
        "premium_weight": 0.3,
        "quality": {},
        "premium": {},
    },
    "sentiment": {},
    "allocation": {},
}


def _core(**over):
    row = {
        "px": 25.0,
        "mkt_cap": 20000,
        "avg_opt_vol_20d": 2000,
        "borrow30": 3,
        "iv30": 40,
        "iv30_dec": 0.40,
        "iv_pctile_1y": 55,
        "iv_hv": 1.2,
        "next_ern": "2026-12-15",
        "days_to_ern": 100,
        "div_yield": 2.0,
        "beta1y": 1.0,
        "c_vol": 1000,
        "p_vol": 400,
        "confidence": 70,
        "chg_1m": -2.0,
        "chg_1w": 1.0,
        "sector": "Financials",
        "raw": True,
    }
    row.update(over)
    return row


class TestTier(unittest.TestCase):
    def test_higher_and_lower(self):
        self.assertEqual(tier_score(20, 15, 10, 5), 100)
        self.assertEqual(tier_score(0.4, 0.5, 1.0, 2.0, lower_is_better=True), 100)


class TestStage0(unittest.TestCase):
    def test_price_and_earnings(self):
        self.assertIsNone(stage0_reason(_core(), {"last": 25}, "2026-08-28", CFG))
        self.assertEqual(stage0_reason(_core(px=80), None, "2026-08-28", CFG), "price_range")
        open_hi = dict(CFG)
        open_hi["universe"] = dict(CFG["universe"])
        open_hi["universe"]["max_price"] = 0
        self.assertIsNone(stage0_reason(_core(px=80), None, "2026-08-28", open_hi))
        self.assertTrue(stage0_reason(_core(next_ern="2026-09-01", days_to_ern=4), None, "2026-08-28", CFG).startswith("earnings_"))
        self.assertTrue(
            stage0_reason(_core(next_ern="0000-00-00", days_to_ern=0, wks_next_ern=1), None, "2026-08-28", CFG).startswith("earnings_")
        )
        self.assertEqual(stage0_reason(_core(avg_opt_vol_20d=None), None, "2026-08-28", CFG), "DATA UNAVAILABLE option_volume")


class TestQuality(unittest.TestCase):
    def test_orats_only(self):
        qs = score_quality(_core(), None, [], {"ok": False, "error": "yfinance_skipped"}, CFG, "2026-08-28")
        self.assertFalse(qs.disqualified)
        self.assertGreater(qs.composite, 40)
        self.assertEqual(qs.yfinance_note, "yfinance_unavailable")

    def test_yfinance_failure_does_not_disqualify(self):
        qs = score_quality(_core(), {"pe_ratio": 12, "div_yield": 3}, [], {"ok": False, "error": "yfinance_unavailable"}, CFG, "2026-08-28")
        self.assertFalse(qs.disqualified)

    def test_yfinance_de_disqualify_only_when_present(self):
        qs = score_quality(_core(), None, [], {"ok": True, "debt_equity": 4.0, "roe": 20, "fcf_yield": 6}, CFG, "2026-08-28")
        self.assertTrue(qs.disqualified)


class TestPremium(unittest.TestCase):
    def test_uses_bid_rejects_wide(self):
        rows = [
            {
                "ticker": "SOFI",
                "dte": 32,
                "expirDate": "2026-09-18",
                "strike": 23.0,
                "putBidPrice": 0.80,
                "putAskPrice": 0.82,
                "callBidPrice": 0.55,
                "callAskPrice": 0.57,
            }
        ]
        ps = score_premium(rows, _core(), CFG)
        self.assertFalse(ps.rejected)
        self.assertEqual(ps.csp_premium, 0.80)
        self.assertEqual(ps.csp_bid, 0.80)
        self.assertGreater(ps.csp_yield_ann, 20)

    def test_wide_spread(self):
        rows = [
            {
                "ticker": "SOFI",
                "dte": 32,
                "expirDate": "2026-09-18",
                "strike": 23.0,
                "putBidPrice": 0.40,
                "putAskPrice": 0.90,
                "callBidPrice": 0.55,
                "callAskPrice": 0.57,
            }
        ]
        ps = score_premium(rows, _core(), CFG)
        self.assertTrue(ps.rejected)
        self.assertEqual(ps.reject_reason, "no_put_bid")

    def test_skips_zero_bid_strikes(self):
        rows = [
            {
                "ticker": "SOFI",
                "dte": 32,
                "expirDate": "2026-09-18",
                "strike": 5.0,
                "putBidPrice": 0,
                "putAskPrice": 0.40,
                "callBidPrice": 20.0,
                "callAskPrice": 20.2,
            },
            {
                "ticker": "SOFI",
                "dte": 32,
                "expirDate": "2026-09-18",
                "strike": 23.0,
                "putBidPrice": 0.80,
                "putAskPrice": 0.82,
                "callBidPrice": 0.55,
                "callAskPrice": 0.57,
            },
        ]
        ps = score_premium(rows, _core(), CFG)
        self.assertFalse(ps.rejected)
        self.assertEqual(ps.csp_strike, 23.0)
        self.assertEqual(ps.csp_bid, 0.80)

    def test_rejects_penny_credit(self):
        rows = [
            {
                "ticker": "NFLX",
                "dte": 28,
                "expirDate": "2026-09-25",
                "strike": 75.0,
                "putBidPrice": 0.47,
                "putAskPrice": 0.51,
                "callBidPrice": 0.77,
                "callAskPrice": 0.82,
            }
        ]
        core = _core(px=81.62, iv30=29.7, iv30_dec=0.297, iv_pctile_1y=44, iv_hv=0.95)
        ps = score_premium(rows, core, CFG)
        self.assertTrue(ps.rejected)
        self.assertEqual(ps.reject_reason, "credit_too_small")

    def test_rejects_atm_put(self):
        rows = [
            {
                "ticker": "SOFI",
                "dte": 32,
                "expirDate": "2026-09-18",
                "strike": 25.0,
                "putBidPrice": 1.20,
                "putAskPrice": 1.25,
                "callBidPrice": 1.10,
                "callAskPrice": 1.15,
            }
        ]
        ps = score_premium(rows, _core(), CFG)
        self.assertTrue(ps.rejected)

    def test_no_synthetic(self):
        ps = score_premium([], _core(), CFG)
        self.assertTrue(ps.rejected)


class TestOverlay(unittest.TestCase):
    def test_iv_rich_and_x(self):
        sa = apply_sentiment(_core(), {"bias": "bullish", "tag": "Informed"}, "2026-08-28", CFG)
        self.assertGreater(sa.total, 0)
        empty = apply_sentiment(_core(), None, "2026-08-28", CFG)
        self.assertEqual(empty.x_status, "DATA UNAVAILABLE")

    def test_tier(self):
        self.assertEqual(assign_tier(70, CFG), "core")
        self.assertEqual(assign_tier(20, CFG), "excluded")
        self.assertGreater(compute_composite(80, 50, 3, 0.7, 0.3), 60)


class TestAllocate(unittest.TestCase):
    def test_does_not_drop_expensive_trade(self):
        cands = [
            {
                "ticker": "NVDA",
                "tier": "core",
                "conf": 72,
                "conf_label": "TRADE",
                "credit_pct": 0.028,
                "capital_required": 18000,
                "sector": "Technology",
                "premium": {"csp_strike": 180.0},
            },
            {
                "ticker": "HPE",
                "tier": "core",
                "conf": 40,
                "conf_label": "NO_TRADE",
                "credit_pct": 0.030,
                "capital_required": 4600,
                "sector": "Technology",
                "premium": {"csp_strike": 46.0},
            },
        ]
        out = allocate_capital(cands, 35000, CFG)
        by = {c["ticker"]: c for c in out}
        self.assertTrue(by["NVDA"]["allocated"])
        self.assertFalse(by["HPE"]["allocated"])
