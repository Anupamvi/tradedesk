import sys
import unittest
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import schwab_market as sm  # noqa: E402


def leg(strike, delta, bid, ask):
    return {
        "strike": float(strike),
        "delta": float(delta),
        "bid": float(bid),
        "ask": float(ask),
        "quote_time_ms": 1,
        "bid_size": 10,
        "ask_size": 10,
        "oi": 50,
    }


class TestRegimeGates(unittest.TestCase):
    def test_vix_buckets(self):
        self.assertEqual(sm.vix_regime(15.84), "calm")
        self.assertEqual(sm.vix_regime(16.46), "normal")
        self.assertEqual(sm.vix_regime(17.06), "normal")
        self.assertEqual(sm.vix_regime(22.1), "elevated")
        self.assertEqual(sm.vix_regime(31), "crisis")

    def test_normal_matches_calm_geometry(self):
        calm = sm.resolve_gates("calm")
        normal = sm.resolve_gates("normal")
        self.assertEqual(calm["min_frac"], 0.12)
        self.assertEqual(normal["min_frac"], 0.12)
        self.assertEqual(calm["max_delta"], normal["max_delta"])
        self.assertEqual(calm["min_sigma"], normal["min_sigma"])
        self.assertEqual(sm.resolve_gates("elevated")["min_frac"], 0.25)

    def test_short_clears_and_not_or(self):
        g = sm.resolve_gates("normal")
        # Δ 0.207 / 0.84σ — passes cheap-vol AND, fails old Normal OR (0.20Δ or 0.90σ)
        self.assertTrue(sm.short_clears(delta=-0.207, otm=11.8, sigma=14.0, gates=g))
        old = sm.resolve_gates("normal", min_frac=0.20, or_delta=0.20, min_sigma=0.90, max_delta=0.25)
        self.assertFalse(sm.short_clears(delta=-0.207, otm=11.8, sigma=14.0, gates=old))
        # 0.246Δ / 0.76σ hard-clears 0.25 but fails 0.22 AND 0.80σ
        self.assertFalse(sm.short_clears(delta=-0.246, otm=45.0, sigma=59.15, gates=g))
        self.assertFalse(sm.short_clears(delta=-0.26, otm=20.0, sigma=10.0, gates=g))

    def test_friday_expiries_skip_under_14_dte(self):
        days = sm.friday_expiries(date(2026, 9, 15), min_dte=14, max_dte=60)
        self.assertNotIn("2026-09-18", days)
        self.assertIn("2026-10-02", days)
        self.assertIn("2026-10-16", days)
        self.assertIn("2026-10-30", days)
        self.assertNotIn("2026-11-20", days)


class TestCreditPicker(unittest.TestCase):
    def test_put_credit_picks_frac_then_dollars(self):
        # spot 100, sigma 10 → 0.80σ = 8 pts. 90-delta 0.20 put at 90.
        puts = {
            90.0: leg(90, -0.20, 1.40, 1.50),
            85.0: leg(85, -0.12, 0.30, 0.40),
            80.0: leg(80, -0.08, 0.10, 0.20),
        }
        g = sm.resolve_gates("normal")
        hit = sm._credit_put(puts, spot=100.0, sigma=10.0, gates=g)
        self.assertTrue(hit.get("ok"))
        self.assertEqual(hit["short"], 90.0)
        self.assertEqual(hit["long"], 85.0)
        # conservative credit = 1.40 - 0.40 = 1.00 → 0.20 of width 5
        self.assertGreaterEqual(hit["pricing"]["net"], 1.0)

    def test_old_normal_020_empty_on_intc_like_wing(self):
        puts = {
            87.5: leg(87.5, -0.207, 1.80, 1.90),
            82.5: leg(82.5, -0.14, 0.50, 0.60),
            77.5: leg(77.5, -0.09, 0.20, 0.31),
        }
        spot, sigma = 99.2, 14.0
        cheap = sm._credit_put(puts, spot, sigma, sm.resolve_gates("normal"))
        self.assertTrue(cheap.get("ok"), cheap)
        old = sm.resolve_gates("normal", min_frac=0.20, or_delta=0.20, min_sigma=0.90)
        miss = sm._credit_put(puts, spot, sigma, old)
        self.assertFalse(miss.get("ok"), miss)


if __name__ == "__main__":
    unittest.main()
