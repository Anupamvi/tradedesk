import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "setup_likelihood_backtest.py"
SPEC = importlib.util.spec_from_file_location("setup_likelihood_backtest", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(SPEC)
sys.modules["setup_likelihood_backtest"] = mod
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


if __name__ == "__main__":
    unittest.main()
