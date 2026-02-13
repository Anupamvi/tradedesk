import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCRIPT_PATH = ROOT / "build_live_trade_table.py"
SPEC = importlib.util.spec_from_file_location("build_live_trade_table", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(SPEC)
sys.modules["build_live_trade_table"] = mod
SPEC.loader.exec_module(mod)


class TestBuildLiveTradeTable(unittest.TestCase):
    def test_parse_invalidation_rule_less_than(self) -> None:
        op, level = mod.parse_invalidation_rule("Invalidate if close < 280.00.")
        self.assertEqual(op, "<")
        self.assertAlmostEqual(float(level), 280.0, places=6)

    def test_parse_invalidation_rule_greater_than_or_equal(self) -> None:
        op, level = mod.parse_invalidation_rule("Invalidation: close >= 360")
        self.assertEqual(op, ">=")
        self.assertAlmostEqual(float(level), 360.0, places=6)

    def test_parse_invalidation_rule_missing(self) -> None:
        op, level = mod.parse_invalidation_rule("No invalidation text present")
        self.assertIsNone(op)
        self.assertIsNone(level)

    def test_invalidation_breached_for_less_than_rule(self) -> None:
        self.assertTrue(mod.invalidation_breached("<", 280.0, 277.5))
        self.assertFalse(mod.invalidation_breached("<", 280.0, 281.0))

    def test_invalidation_breached_for_greater_than_rule(self) -> None:
        self.assertTrue(mod.invalidation_breached(">", 360.0, 365.0))
        self.assertFalse(mod.invalidation_breached(">", 360.0, 355.0))

    def test_invalidation_breached_handles_missing_values(self) -> None:
        self.assertIsNone(mod.invalidation_breached("<", 280.0, None))
        self.assertIsNone(mod.invalidation_breached(None, 280.0, 279.0))

    def test_validate_shortlist_columns_condor_ok(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "ticker": "AMZN",
                    "strategy": "Iron Condor",
                    "expiry": "2026-03-20",
                    "net_type": "credit",
                    "entry_gate": ">= 2.50 cr",
                    "width": 10.0,
                    "short_put_leg": "AMZN260320P00190000",
                    "long_put_leg": "AMZN260320P00180000",
                    "short_call_leg": "AMZN260320C00205000",
                    "long_call_leg": "AMZN260320C00215000",
                }
            ]
        )
        mod.validate_shortlist_columns(df)

    def test_validate_shortlist_columns_condor_missing_legs(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "ticker": "AMZN",
                    "strategy": "Iron Condor",
                    "expiry": "2026-03-20",
                    "net_type": "credit",
                    "entry_gate": ">= 2.50 cr",
                    "width": 10.0,
                    "short_put_leg": "",
                    "long_put_leg": "",
                    "short_call_leg": "AMZN260320C00205000",
                    "long_call_leg": "AMZN260320C00215000",
                }
            ]
        )
        with self.assertRaises(ValueError):
            mod.validate_shortlist_columns(df)


if __name__ == "__main__":
    unittest.main()
