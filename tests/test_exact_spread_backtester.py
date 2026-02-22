import datetime as dt
import importlib.util
import sys
import unittest
from pathlib import Path
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "uwos" / "exact_spread_backtester.py"
SPEC = importlib.util.spec_from_file_location("uwos_exact_spread_backtester", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(SPEC)
sys.modules["uwos_exact_spread_backtester"] = mod
SPEC.loader.exec_module(mod)


class TestExactSpreadBacktester(unittest.TestCase):
    def test_occ_roundtrip(self) -> None:
        expiry = dt.date(2026, 3, 20)
        sym = mod.build_occ_symbol("AAPL", expiry, "C", 295.0)
        parsed = mod.parse_occ_symbol(sym)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed[0], "AAPL")
        self.assertEqual(parsed[1], expiry)
        self.assertEqual(parsed[2], "C")
        self.assertAlmostEqual(parsed[3], 295.0, places=3)

    def test_entry_gate_parse(self) -> None:
        op, thr, unit = mod.parse_entry_gate("<= 3.95 db")
        self.assertEqual(op, "<=")
        self.assertAlmostEqual(thr, 3.95, places=6)
        self.assertEqual(unit, "db")

    def test_spread_value_at_expiry(self) -> None:
        long_leg = "AAPL260320C00295000"
        short_leg = "AAPL260320C00305000"
        v = mod._spread_value_at_expiry(long_leg, short_leg, 310.0, net_type="debit")
        self.assertIsNotNone(v)
        assert v is not None
        # Long 295C intrinsic=15, short 305C intrinsic=5, spread=10.
        self.assertAlmostEqual(v, 10.0, places=6)

    def test_spread_value_at_expiry_credit(self) -> None:
        long_leg = "AAPL260320P00240000"
        short_leg = "AAPL260320P00250000"
        v = mod._spread_value_at_expiry(long_leg, short_leg, 230.0, net_type="credit")
        self.assertIsNotNone(v)
        assert v is not None
        # short put intrinsic=20, long put intrinsic=10 => close debit 10.
        self.assertAlmostEqual(v, 10.0, places=6)

    def test_pnl_math_debit_and_credit(self) -> None:
        # Debit spread: pay 2.00, exit value 4.50 -> +250 per contract.
        pnl_debit = mod._pnl_from_spread(2.0, 4.5, "debit", qty=1.0)
        self.assertAlmostEqual(pnl_debit, 250.0, places=6)
        # Credit spread: collect 3.00, exit value 1.25 -> +175 per contract.
        pnl_credit = mod._pnl_from_spread(3.0, 1.25, "credit", qty=1.0)
        self.assertAlmostEqual(pnl_credit, 175.0, places=6)

    def test_max_profit_loss(self) -> None:
        mp, ml = mod.max_profit_max_loss(10.0, 2.5, "debit")
        self.assertAlmostEqual(mp, 750.0, places=6)
        self.assertAlmostEqual(ml, 250.0, places=6)
        mp2, ml2 = mod.max_profit_max_loss(10.0, 3.0, "credit")
        self.assertAlmostEqual(mp2, 300.0, places=6)
        self.assertAlmostEqual(ml2, 700.0, places=6)

    def test_expiry_intrinsic_ignores_input_exit_net(self) -> None:
        class DummyQuoteStore:
            def get_leg_quote(self, *_args, **_kwargs):
                return None

        class DummyCloseStore:
            def get_close_on_or_before(self, *_args, **_kwargs):
                return 110.0

        setups = pd.DataFrame(
            [
                {
                    "trade_id": "T1",
                    "signal_date": dt.date(2020, 1, 2),
                    "ticker": "AAPL",
                    "strategy": "Bear Call Credit",
                    "expiry": dt.date(2020, 1, 17),
                    "short_leg": "AAPL200117C00100000",
                    "long_leg": "AAPL200117C00105000",
                    "short_strike": 100.0,
                    "long_strike": 105.0,
                    "width": 5.0,
                    "net_type": "credit",
                    "qty": 1.0,
                    "entry_gate": "",
                    "entry_net": 2.0,
                    "exit_date": dt.date(2020, 1, 10),
                    "exit_net": 0.25,  # Must be ignored in expiry_intrinsic mode.
                }
            ]
        )

        out = mod.run_backtest(
            setups=setups,
            quote_store=DummyQuoteStore(),
            close_store=DummyCloseStore(),
            entry_source="input_only",
            entry_price_model="conservative",
            exit_mode="expiry_intrinsic",
            exit_price_model="conservative",
            close_lookback_days=7,
        )
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["exit_source"], "expiry_intrinsic")
        self.assertAlmostEqual(float(row["exit_net"]), 5.0, places=6)
        self.assertAlmostEqual(float(row["pnl"]), -300.0, places=6)

    def test_quotes_then_expiry_ignores_input_exit_net_when_quotes_missing(self) -> None:
        class DummyQuoteStore:
            def get_leg_quote(self, *_args, **_kwargs):
                return None

        class DummyCloseStore:
            def get_close_on_or_before(self, *_args, **_kwargs):
                return 90.0

        setups = pd.DataFrame(
            [
                {
                    "trade_id": "T2",
                    "signal_date": dt.date(2020, 1, 2),
                    "ticker": "AAPL",
                    "strategy": "Bull Put Credit",
                    "expiry": dt.date(2020, 1, 17),
                    "short_leg": "AAPL200117P00100000",
                    "long_leg": "AAPL200117P00095000",
                    "short_strike": 100.0,
                    "long_strike": 95.0,
                    "width": 5.0,
                    "net_type": "credit",
                    "qty": 1.0,
                    "entry_gate": "",
                    "entry_net": 2.0,
                    "exit_date": dt.date(2020, 1, 10),
                    "exit_net": 0.10,  # Must be ignored in quotes_then_expiry mode.
                }
            ]
        )

        out = mod.run_backtest(
            setups=setups,
            quote_store=DummyQuoteStore(),
            close_store=DummyCloseStore(),
            entry_source="input_only",
            entry_price_model="conservative",
            exit_mode="quotes_then_expiry",
            exit_price_model="conservative",
            close_lookback_days=7,
        )
        self.assertEqual(len(out), 1)
        row = out.iloc[0]
        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["exit_source"], "expiry_intrinsic")
        self.assertAlmostEqual(float(row["exit_net"]), 5.0, places=6)
        self.assertAlmostEqual(float(row["pnl"]), -300.0, places=6)


if __name__ == "__main__":
    unittest.main()
