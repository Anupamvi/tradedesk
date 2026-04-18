import datetime as dt
import tempfile
import unittest
from pathlib import Path


class TestTradeDeskReport(unittest.TestCase):
    def test_builds_actionable_and_hold_rows(self):
        from uwos.trade_desk import build_recommendations, build_report

        result = {
            "as_of": "2026-04-18T16:00:00Z",
            "positions": [
                {
                    "symbol": "AAPL  260515P00200000",
                    "underlying": "AAPL",
                    "asset_type": "OPTION",
                    "put_call": "PUT",
                    "strike": 200.0,
                    "expiry": "2026-05-15",
                    "qty": -1,
                    "entry_date": "2026-03-17",
                    "greeks": {"delta": -0.10},
                    "underlying_quote": {"last": 240.0},
                    "computed": {
                        "dte": 27,
                        "pct_of_max_profit": 90.0,
                        "unrealized_pnl": 450.0,
                        "unrealized_pnl_pct": 90.0,
                        "theta_pnl_per_day": 4.0,
                    },
                },
                {
                    "symbol": "MSFT",
                    "underlying": "MSFT",
                    "asset_type": "EQUITY",
                    "put_call": "",
                    "strike": None,
                    "expiry": None,
                    "qty": 10,
                    "entry_date": "",
                    "greeks": None,
                    "underlying_quote": {"last": 400.0, "change_pct": 0.2},
                    "computed": {
                        "dte": None,
                        "pct_of_max_profit": None,
                        "unrealized_pnl": 100.0,
                        "unrealized_pnl_pct": 2.5,
                    },
                },
            ],
        }

        rows = build_recommendations(result)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["verdict"], "CLOSE")
        self.assertEqual(rows[0]["action"], "CLOSE")
        self.assertIn("remaining premium is mostly gamma/assignment risk", rows[0]["instruction"])
        self.assertEqual(rows[0]["underlying"], "AAPL")
        self.assertEqual(rows[0]["position"], "Short 1 AAPL 2026-05-15 $200 PUT")

        with tempfile.TemporaryDirectory() as td:
            report = build_report(
                result,
                rows,
                days=90,
                symbol=None,
                json_path=Path(td) / f"position_data_{dt.date.today().isoformat()}.json",
            )
        self.assertIn("TRADE DESK RECOMMENDATIONS", report)
        self.assertIn("Schwab history lookback: 90 days", report)
        self.assertIn("ACTION REQUIRED", report)
        self.assertIn("🟢 KEEP / HOLD", report)
        self.assertIn("yfinance=off", report)
        self.assertIn("🔴 CLOSE", report)
        self.assertIn("AAPL | CLOSE", report)
        self.assertIn("🔴 AAPL | CLOSE", report)
        self.assertIn("mostly gamma/assignment risk", report)
        self.assertIn("Open option legs reviewed: 1", report)
        self.assertIn("Equities/funds omitted: 1", report)
        self.assertIn("Legs: Short 1 AAPL 2026-05-15 $200 PUT", report)
        self.assertNotIn("MSFT", report)
        self.assertNotIn("| verdict |", report)
        self.assertNotIn("Position:", report)
        self.assertNotIn("####", report)
        self.assertNotIn("[CLOSE]", report)

    def test_reviews_vertical_spread_as_one_position(self):
        from uwos.trade_desk import build_recommendations, build_report

        result = {
            "as_of": "2026-04-18T16:00:00Z",
            "positions": [
                {
                    "symbol": "AAPL  260515P00200000",
                    "underlying": "AAPL",
                    "asset_type": "OPTION",
                    "put_call": "PUT",
                    "strike": 200.0,
                    "expiry": "2026-05-15",
                    "qty": -1,
                    "avg_cost": 4.00,
                    "entry_date": "2026-03-17",
                    "greeks": {"delta": -0.70},
                    "underlying_quote": {"last": 198.0},
                    "computed": {
                        "dte": 27,
                        "pct_of_max_profit": -50.0,
                        "unrealized_pnl": -300.0,
                        "unrealized_pnl_pct": -75.0,
                        "max_profit": 400.0,
                        "theta_pnl_per_day": 5.0,
                    },
                },
                {
                    "symbol": "AAPL  260515P00195000",
                    "underlying": "AAPL",
                    "asset_type": "OPTION",
                    "put_call": "PUT",
                    "strike": 195.0,
                    "expiry": "2026-05-15",
                    "qty": 1,
                    "avg_cost": 2.00,
                    "entry_date": "2026-03-17",
                    "greeks": {"delta": -0.55},
                    "underlying_quote": {"last": 198.0},
                    "computed": {
                        "dte": 27,
                        "unrealized_pnl": 125.0,
                        "unrealized_pnl_pct": 62.5,
                        "max_loss": 200.0,
                        "theta_pnl_per_day": -3.0,
                    },
                },
            ],
        }

        rows = build_recommendations(result)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["category"], "Bull Put Credit")
        self.assertEqual(
            rows[0]["position"],
            "Short 1 AAPL 2026-05-15 $200 PUT / Long 1 AAPL 2026-05-15 $195 PUT",
        )
        self.assertIn(rows[0]["verdict"], {"ASSESS", "ROLL"})
        self.assertIn("spread", rows[0]["reason"].lower())
        self.assertTrue(
            "both legs" in rows[0]["reason"].lower()
            or "whole spread" in rows[0]["reason"].lower()
        )

        with tempfile.TemporaryDirectory() as td:
            report = build_report(
                result,
                rows,
                days=90,
                symbol=None,
                json_path=Path(td) / f"position_data_{dt.date.today().isoformat()}.json",
            )
        self.assertIn("Spread groups reviewed: 1", report)
        self.assertIn("act on both legs together", report)
        self.assertIn(
            "Legs: Short 1 AAPL 2026-05-15 $200 PUT / Long 1 AAPL 2026-05-15 $195 PUT",
            report,
        )

    def test_debit_spread_displays_long_leg_first(self):
        from uwos.trade_desk import build_recommendations

        result = {
            "as_of": "2026-04-18T16:00:00Z",
            "positions": [
                {
                    "symbol": "NVDA  260508C00205000",
                    "underlying": "NVDA",
                    "asset_type": "OPTION",
                    "put_call": "CALL",
                    "strike": 205.0,
                    "expiry": "2026-05-08",
                    "qty": -1,
                    "avg_cost": 1.0,
                    "greeks": {"delta": 0.35},
                    "underlying_quote": {"last": 201.0},
                    "computed": {"dte": 20, "unrealized_pnl": -50.0, "max_profit": 100.0},
                },
                {
                    "symbol": "NVDA  260508C00185000",
                    "underlying": "NVDA",
                    "asset_type": "OPTION",
                    "put_call": "CALL",
                    "strike": 185.0,
                    "expiry": "2026-05-08",
                    "qty": 1,
                    "avg_cost": 5.0,
                    "greeks": {"delta": 0.70},
                    "underlying_quote": {"last": 201.0},
                    "computed": {"dte": 20, "unrealized_pnl": 800.0, "max_loss": 500.0},
                },
            ],
        }

        rows = build_recommendations(result)
        self.assertEqual(rows[0]["category"], "Bull Call Debit")
        self.assertEqual(
            rows[0]["position"],
            "Long 1 NVDA 2026-05-08 $185 CALL / Short 1 NVDA 2026-05-08 $205 CALL",
        )

    def test_assess_is_reported_as_set_stop(self):
        from uwos.trade_desk import build_recommendations, build_report

        result = {
            "as_of": "2026-04-18T16:00:00Z",
            "positions": [
                {
                    "symbol": "BABA  260508C00150000",
                    "underlying": "BABA",
                    "asset_type": "OPTION",
                    "put_call": "CALL",
                    "strike": 150.0,
                    "expiry": "2026-05-08",
                    "qty": -1,
                    "avg_cost": 1.0,
                    "greeks": {"delta": 0.29},
                    "underlying_quote": {"last": 141.4},
                    "computed": {"dte": 20, "unrealized_pnl": 20.0, "max_profit": 100.0},
                },
                {
                    "symbol": "BABA  260508C00145000",
                    "underlying": "BABA",
                    "asset_type": "OPTION",
                    "put_call": "CALL",
                    "strike": 145.0,
                    "expiry": "2026-05-08",
                    "qty": 1,
                    "avg_cost": 2.5,
                    "greeks": {"delta": 0.45},
                    "underlying_quote": {"last": 141.4},
                    "computed": {"dte": 20, "unrealized_pnl": -40.0, "max_loss": 250.0},
                },
            ],
        }

        rows = build_recommendations(result)
        self.assertEqual(rows[0]["verdict"], "ASSESS")
        self.assertEqual(rows[0]["action"], "SET STOP")
        self.assertIn("cannot reclaim $145", rows[0]["instruction"])
        self.assertIn("theta/gamma decay is working against the position", rows[0]["instruction"])
        self.assertIn("Monday, April 20, 2026", rows[0]["instruction"])

        with tempfile.TemporaryDirectory() as td:
            report = build_report(
                result,
                rows,
                days=90,
                symbol=None,
                json_path=Path(td) / f"position_data_{dt.date.today().isoformat()}.json",
            )
        self.assertIn("SET STOP", report)
        self.assertIn("🟡 SET STOP", report)
        self.assertIn("Do this: Set a stop", report)
        self.assertNotIn("ASSESS", report)

    def test_profitable_debit_spread_stop_uses_spread_midpoint(self):
        from uwos.trade_desk import build_recommendations

        result = {
            "as_of": "2026-04-18T16:00:00Z",
            "positions": [
                {
                    "symbol": "NVDA  260508C00205000",
                    "underlying": "NVDA",
                    "asset_type": "OPTION",
                    "put_call": "CALL",
                    "strike": 205.0,
                    "expiry": "2026-05-08",
                    "qty": -1,
                    "avg_cost": 1.0,
                    "greeks": {"delta": 0.35},
                    "underlying_quote": {"last": 201.0},
                    "computed": {"dte": 20, "unrealized_pnl": -50.0, "max_profit": 100.0},
                },
                {
                    "symbol": "NVDA  260508C00185000",
                    "underlying": "NVDA",
                    "asset_type": "OPTION",
                    "put_call": "CALL",
                    "strike": 185.0,
                    "expiry": "2026-05-08",
                    "qty": 1,
                    "avg_cost": 5.0,
                    "greeks": {"delta": 0.70},
                    "underlying_quote": {"last": 201.0},
                    "computed": {"dte": 20, "unrealized_pnl": 1030.0, "max_loss": 500.0},
                },
            ],
        }

        rows = build_recommendations(result)
        self.assertEqual(rows[0]["action"], "SET STOP")
        self.assertIn("spread midpoint", rows[0]["instruction"])
        self.assertIn("falls back below $195", rows[0]["instruction"])
        self.assertIn("If it reaches $205", rows[0]["instruction"])
        self.assertNotIn("reclaim $185", rows[0]["instruction"])

    def test_debit_spread_decay_can_make_it_unrecoverable(self):
        from uwos.trade_desk import build_recommendations

        result = {
            "as_of": "2026-04-24T16:00:00Z",
            "positions": [
                {
                    "symbol": "BABA  260508C00150000",
                    "underlying": "BABA",
                    "asset_type": "OPTION",
                    "put_call": "CALL",
                    "strike": 150.0,
                    "expiry": "2026-05-08",
                    "qty": -1,
                    "avg_cost": 1.0,
                    "greeks": {"delta": 0.20},
                    "underlying_quote": {"last": 141.0},
                    "computed": {"dte": 14, "unrealized_pnl": 20.0, "max_profit": 100.0, "gamma_risk": 1.0},
                },
                {
                    "symbol": "BABA  260508C00145000",
                    "underlying": "BABA",
                    "asset_type": "OPTION",
                    "put_call": "CALL",
                    "strike": 145.0,
                    "expiry": "2026-05-08",
                    "qty": 1,
                    "avg_cost": 2.5,
                    "greeks": {"delta": 0.35},
                    "underlying_quote": {"last": 141.0},
                    "computed": {"dte": 14, "unrealized_pnl": -50.0, "max_loss": 250.0, "gamma_risk": 2.0},
                },
            ],
        }

        rows = build_recommendations(result)
        self.assertEqual(rows[0]["action"], "CLOSE")
        self.assertIn("theta decay makes recovery unlikely", rows[0]["reason"])
        self.assertIn("Close the whole spread", rows[0]["instruction"])


if __name__ == "__main__":
    unittest.main()
