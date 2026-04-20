import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "uwos" / "setup_likelihood_backtest.py"
SPEC = importlib.util.spec_from_file_location("uwos_setup_likelihood_backtest", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(SPEC)
sys.modules["uwos_setup_likelihood_backtest"] = mod
SPEC.loader.exec_module(mod)


class TestSetupLikelihoodBacktest(unittest.TestCase):
    def test_required_win_rate_iron_condor(self) -> None:
        # Max profit = credit 2.5, max loss = 10 - 2.5 = 7.5 => req win = 75%
        req = mod.required_win_rate_pct("Iron Condor", width=10.0, net=2.5, put_width=10.0, call_width=10.0)
        self.assertAlmostEqual(req, 75.0, places=6)

    def test_breakeven_levels_iron_condor(self) -> None:
        lo, hi = mod.breakeven_levels(
            "Iron Condor",
            long_strike=180.0,
            short_strike=190.0,
            net=2.5,
            short_call_strike=205.0,
        )
        self.assertAlmostEqual(lo, 187.5, places=6)
        self.assertAlmostEqual(hi, 207.5, places=6)

    def test_simulate_setup_iron_condor(self) -> None:
        # Flat path that stays inside both shorts should produce all wins.
        hist = pd.DataFrame(
            {
                "Open": [100, 100, 100, 100, 100],
                "High": [101, 101, 101, 101, 101],
                "Low": [99, 99, 99, 99, 99],
                "Close": [100, 100, 100, 100, 100],
            }
        )
        signals, wins, no_touch = mod.simulate_setup(
            strategy="Iron Condor",
            hist=hist,
            dte=2,
            spot_asof=100.0,
            short_strike=95.0,
            breakeven_level=92.0,
            short_call_strike=105.0,
            upper_breakeven_level=108.0,
        )
        self.assertGreater(signals, 0)
        self.assertEqual(wins, signals)
        self.assertTrue(np.isfinite(no_touch))

    def test_unknown_row_contract(self) -> None:
        row = mod.unknown_row(
            ticker="AAPL",
            strategy="Bull Call Debit",
            expiry="2026-03-20",
            dte=30,
            entry_gate="<= 2.00 db",
            spot_at_signal=190.0,
            required_win_pct=40.0,
            reason="missing_history",
        )
        self.assertEqual(row["verdict"], "UNKNOWN")
        self.assertEqual(row["confidence"], "Unknown")
        self.assertEqual(int(row["signals"]), 0)

    def test_safe_symbol_for_file(self) -> None:
        self.assertEqual(mod.safe_symbol_for_file("BRK/B"), "BRK_B")

    def test_yfinance_symbol_for_class_share(self) -> None:
        self.assertEqual(mod.yfinance_symbol("BRKB"), "BRK-B")
        self.assertEqual(mod.yfinance_symbol("BRK/B"), "BRK-B")
        self.assertEqual(mod.yfinance_symbol("BABA"), "BABA")

    def test_price_conditioning_uses_same_trend_bucket(self) -> None:
        closes = [100 + i for i in range(70)] + [170 - i for i in range(30)]
        hist = pd.DataFrame(
            {
                "Open": closes,
                "High": [c + 1 for c in closes],
                "Low": [c - 1 for c in closes],
                "Close": closes,
            }
        )
        thresholds = mod.context_thresholds(hist)
        up_ctx = mod.price_context_at(hist, 40, thresholds=thresholds)
        down_ctx = mod.price_context_at(hist, 95, thresholds=thresholds)
        profile = {
            "trend_bucket": "up",
            "vol_bucket": "unknown",
            "range_bucket": "unknown",
            "require_range_neutral": False,
        }
        self.assertTrue(mod.context_matches_profile(up_ctx, profile, "same_trend"))
        self.assertFalse(mod.context_matches_profile(down_ctx, profile, "same_trend"))

    def test_unknown_row_includes_base_rate_transparency_columns(self) -> None:
        row = mod.unknown_row(
            ticker="AAPL",
            strategy="Bull Call Debit",
            expiry="2026-03-20",
            dte=30,
            entry_gate="<= 2.00 db",
            spot_at_signal=190.0,
            required_win_pct=40.0,
            reason="missing_history",
        )
        self.assertIn("base_hist_success_pct", row)
        self.assertIn("conditioning_level", row)
        self.assertIn("unsupported_context", row)

    def test_unknown_row_preserves_setup_id_for_precise_merge(self) -> None:
        row = mod.unknown_row(
            ticker="AAPL",
            strategy="Bull Call Debit",
            expiry="2026-03-20",
            dte=30,
            entry_gate="<= 2.00 db",
            spot_at_signal=190.0,
            required_win_pct=40.0,
            reason="missing_history",
            setup_id="AAPL|Bull Call Debit|2026-03-20|190|200|10|<= 2.00 db",
        )
        self.assertEqual(
            row["setup_id"],
            "AAPL|Bull Call Debit|2026-03-20|190|200|10|<= 2.00 db",
        )


if __name__ == "__main__":
    unittest.main()
