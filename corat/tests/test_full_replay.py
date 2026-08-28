from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from corat.cli import main
from corat.config import load_config
from corat.full_replay import (
    authorize_replay,
    build_replay_plan,
    exact_option_cashflow,
    monthly_pnl_summary,
    replay_metrics,
    resolve_underlying_path,
    run_full_replay,
    split_trade,
)
from corat.models import Bar
from corat.orats import FetchBundle


def _row(strike, call_bid=1.0, call_ask=1.2, put_bid=1.0, put_ask=1.2):
    return {
        "tradeDate": "2026-02-03",
        "expirDate": "2026-03-20",
        "strike": strike,
        "callBidPrice": call_bid,
        "callAskPrice": call_ask,
        "putBidPrice": put_bid,
        "putAskPrice": put_ask,
    }


class FakeOfflineReplayClient:
    """Synthetic cache-only provider; any chain request is a test failure."""

    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.run_requests = 0
        self.max_requests = 0
        self._rows = []
        current = date(2023, 12, 1)
        index = 0
        while current <= date(2025, 1, 24):
            if current.weekday() < 5:
                close = 100.0 + index * 0.08
                for ticker in ("AAA", "SPY", "QQQ", "IWM", "DIA", "VIX", "TLT", "UUP", "HYG", "GLD"):
                    self._rows.append(
                        {
                            "ticker": ticker,
                            "tradeDate": current.isoformat(),
                            "open": close - 0.05,
                            "hiPx": close + 0.60,
                            "loPx": close - 0.60,
                            "clsPx": close,
                            "stockVolume": 2_000_000,
                            "updatedAt": current.isoformat() + "T21:00:00Z",
                        }
                    )
                index += 1
            current += timedelta(days=1)

    @staticmethod
    def normalize_tickers(tickers):
        return list(dict.fromkeys(str(value).upper() for value in tickers))

    def usage(self):
        return {"used": 0, "cap": 20000, "left": 20000, "run_requests": 0, "run_left": 0}

    def fetch_dailies(self, tickers, start_date, end_date, batch_size=10):
        del batch_size
        wanted = set(self.normalize_tickers(tickers))
        return FetchBundle(
            rows=[
                row for row in self._rows
                if row["ticker"] in wanted and start_date <= row["tradeDate"] <= end_date
            ]
        )

    def fetch_market_asof(self, family, as_of):
        tickers = ("AAA", "SPY", "QQQ", "IWM", "DIA", "VIX", "TLT", "UUP", "HYG", "GLD")
        if family == "cores":
            return FetchBundle(
                rows=[
                    {
                        "ticker": ticker,
                        "tradeDate": as_of,
                        "assetType": "0",
                        "mktCap": 100_000_000,
                        "pxCls": 125.0,
                        "pxAtmIv": 125.0,
                        "stkVolu": 2_000_000,
                        "avgOptVolu20d": 10_000,
                        "sectorName": "Technology",
                        "bestEtf": "SPY",
                        "orIvXern20d": 0.25,
                        "orHv20d": 0.20,
                        "orFcst20d": 0.21,
                        "orIvFcst20d": 0.24,
                        "nextErn": "2025-03-15",
                        "lastErn": "2024-11-01",
                        "updatedAt": as_of + "T21:00:00Z",
                    }
                    for ticker in tickers
                ]
            )
        if family == "ivrank":
            return FetchBundle(rows=[{"ticker": ticker, "tradeDate": as_of, "iv": 0.25, "ivRank1y": 50} for ticker in tickers])
        return FetchBundle(rows=[{"ticker": ticker, "tradeDate": as_of, "stockPrice": 125.0} for ticker in tickers])

    def fetch_chain(self, *args, **kwargs):
        raise AssertionError("synthetic no-trigger replay unexpectedly requested a chain")

    def fetch_core_history(self, *args, **kwargs):
        raise AssertionError("synthetic no-trigger replay unexpectedly requested core history")

    def fetch_earnings(self, *args, **kwargs):
        raise AssertionError("synthetic no-trigger replay unexpectedly requested earnings")

    def fetch_historical_chain_full(self, *args, **kwargs):
        raise AssertionError("synthetic no-trigger replay unexpectedly requested an exit chain")


class FullReplayPlanningTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_plan_is_pure_and_makes_no_network_request(self):
        with patch("urllib.request.urlopen", side_effect=AssertionError("network called")):
            plan = build_replay_plan(
                self.config,
                "2025-01-02",
                "2025-12-31",
                "2025-06-30",
                "2025-09-30",
                tickers=["NVDA", "AAPL"],
            )
        self.assertEqual(plan["status"], "PLAN_ONLY_NOT_STARTED")
        self.assertEqual(plan["network_requests_made"], 0)
        self.assertFalse(plan["token_read"])
        self.assertGreater(plan["estimates"]["planned_request_ceiling"], 0)

    def test_execute_false_never_constructs_client(self):
        plan = build_replay_plan(
            self.config,
            "2025-01-02",
            "2025-12-31",
            "2025-06-30",
            "2025-09-30",
            tickers=["NVDA"],
        )
        with patch("corat.full_replay.OratsClient", side_effect=AssertionError("client constructed")):
            result = run_full_replay(self.config, "", plan, execute=False)
        self.assertEqual(result["status"], "PLAN_ONLY_NOT_STARTED")

    def test_frozen_plan_rejects_strategy_config_drift(self):
        plan = build_replay_plan(
            self.config,
            "2025-01-02",
            "2025-12-31",
            "2025-06-30",
            "2025-09-30",
            tickers=["NVDA"],
        )
        changed = dict(self.config)
        changed["risk"] = dict(changed["risk"], normal_risk_pct=0.02)
        with self.assertRaisesRegex(ValueError, "strategy config differs"):
            run_full_replay(changed, "", plan, execute=False)

    def test_online_authorization_requires_cap_reserve_and_worst_case_fit(self):
        plan = {"estimates": {"planned_request_ceiling": 100}}
        usage = {"used": 500, "cap": 20000, "left": 19500}
        with self.assertRaisesRegex(ValueError, "request-budget"):
            authorize_replay(plan, usage, True, False, None, 1000, 12000)
        with self.assertRaisesRegex(ValueError, "monthly-reserve"):
            authorize_replay(plan, usage, True, False, 200, None, 12000)
        with self.assertRaisesRegex(ValueError, "confirmed-remaining"):
            authorize_replay(plan, usage, True, False, 200, 1000, None)
        with self.assertRaisesRegex(ValueError, "planned request ceiling"):
            authorize_replay(plan, usage, True, False, 99, 1000, 12000)
        with self.assertRaisesRegex(ValueError, "spendable"):
            authorize_replay(plan, usage, True, False, 19000, 1000, 12000)
        allowed = authorize_replay(plan, usage, True, False, 200, 1000, 12000)
        self.assertTrue(allowed["authorized"])
        self.assertEqual(allowed["network_budget"], 200)
        self.assertEqual(allowed["console_confirmed_remaining"], 12000)

    def test_offline_authorization_has_zero_network_budget(self):
        allowed = authorize_replay({"estimates": {}}, {}, True, True, None, None)
        self.assertEqual(allowed["network_budget"], 0)
        with self.assertRaisesRegex(ValueError, "offline replay request budget"):
            authorize_replay({"estimates": {}}, {}, True, True, 1, None)

    def test_cli_plan_only_does_not_require_token(self):
        output = io.StringIO()
        with patch("corat.cli._require_token", side_effect=AssertionError("token read")), patch(
            "urllib.request.urlopen", side_effect=AssertionError("network called")
        ), redirect_stdout(output):
            code = main(
                [
                    "full-replay",
                    "--start", "2025-01-02",
                    "--end", "2025-12-31",
                    "--train-end", "2025-06-30",
                    "--validation-end", "2025-09-30",
                    "--tickers", "NVDA,AAPL",
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("plan_only_not_started", output.getvalue())
        self.assertIn("orats_requests=0", output.getvalue())

    def test_cache_only_fixture_walks_orchestrator_without_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            universe = root / "universe.csv"
            universe.write_text(
                "ticker,name,sector,theme,kind,sector_etf\nAAA,AAA,Technology,Fixture,equity,SPY\n",
                encoding="utf-8",
            )
            config = dict(self.config)
            config["universe_file"] = str(universe)
            config["output_root"] = str(root / "out")
            config["cache_root"] = str(root / "cache")
            config["state_root"] = str(root / "state")
            config["discovery"] = dict(config["discovery"], dynamic_orats_universe=False)
            plan = build_replay_plan(
                config,
                "2025-01-06",
                "2025-01-10",
                "2025-01-07",
                "2025-01-09",
                tickers=["AAA"],
            )
            with patch("corat.full_replay.OratsClient", FakeOfflineReplayClient), patch(
                "urllib.request.urlopen", side_effect=AssertionError("network called")
            ):
                report = run_full_replay(config, "", plan, execute=True, offline=True)
            self.assertEqual(report["orats_usage"]["run_requests"], 0)
            self.assertFalse(report["production_promotion"])
            self.assertTrue(Path(report["artifacts"]["report"]).is_file())
            self.assertTrue(Path(report["artifacts"]["manifest"]).is_file())


class FullReplayPricingTest(unittest.TestCase):
    def test_long_option_entry_and_exit_cashflows(self):
        rows = [_row(100, call_bid=1.8, call_ask=2.0)]
        option = {
            "expiration": "2026-03-20",
            "debit_credit": "DEBIT",
            "legs": [{"action": "BUY", "option_type": "CALL", "strike": 100, "expiration": "2026-03-20", "quantity": 1}],
        }
        entry, _ = exact_option_cashflow(rows, option, "ENTRY", 0.50, "2026-02-03")
        exit_value, _ = exact_option_cashflow(rows, option, "EXIT", 0.25, "2026-02-03")
        self.assertAlmostEqual(entry, -1.95)
        self.assertAlmostEqual(exit_value, 1.825)

    def test_debit_spread_exact_cashflows(self):
        rows = [
            _row(100, call_bid=8.0, call_ask=8.4),
            _row(110, call_bid=3.0, call_ask=3.4),
        ]
        option = {
            "expiration": "2026-03-20",
            "debit_credit": "DEBIT",
            "legs": [
                {"action": "BUY", "option_type": "CALL", "strike": 100, "expiration": "2026-03-20", "quantity": 1},
                {"action": "SELL", "option_type": "CALL", "strike": 110, "expiration": "2026-03-20", "quantity": 1},
            ],
        }
        entry, _ = exact_option_cashflow(rows, option, "ENTRY", 0.50, "2026-02-03")
        exit_value, _ = exact_option_cashflow(rows, option, "EXIT", 0.25, "2026-02-03")
        self.assertAlmostEqual(entry, -5.2)
        self.assertAlmostEqual(exit_value, 4.7)

    def test_credit_spread_exact_cashflows(self):
        rows = [
            _row(100, put_bid=3.0, put_ask=3.4),
            _row(95, put_bid=1.0, put_ask=1.4),
        ]
        option = {
            "expiration": "2026-03-20",
            "debit_credit": "CREDIT",
            "legs": [
                {"action": "SELL", "option_type": "PUT", "strike": 100, "expiration": "2026-03-20", "quantity": 1},
                {"action": "BUY", "option_type": "PUT", "strike": 95, "expiration": "2026-03-20", "quantity": 1},
            ],
        }
        entry, _ = exact_option_cashflow(rows, option, "ENTRY", 0.50, "2026-02-03")
        exit_value, _ = exact_option_cashflow(rows, option, "EXIT", 0.25, "2026-02-03")
        self.assertAlmostEqual(entry, 1.8)
        self.assertAlmostEqual(exit_value, -2.3)

    def test_missing_exact_leg_is_never_reconstructed(self):
        option = {
            "expiration": "2026-03-20",
            "legs": [
                {"action": "BUY", "option_type": "CALL", "strike": 100, "expiration": "2026-03-20", "quantity": 1},
                {"action": "SELL", "option_type": "CALL", "strike": 110, "expiration": "2026-03-20", "quantity": 1},
            ],
        }
        cashflow, reason = exact_option_cashflow([_row(100)], option, "EXIT", 0.25, "2026-02-03")
        self.assertIsNone(cashflow)
        self.assertIn("exact", reason)


class FullReplayPathAndMetricsTest(unittest.TestCase):
    def test_next_session_zone_required_and_same_day_stop_wins(self):
        no_fill = [Bar("2026-02-03", 110, 112, 109, 111, 1000)]
        result = resolve_underlying_path(no_fill, "BULLISH", 99, 101, 95, 110, 10)
        self.assertFalse(result["filled"])
        both = [
            Bar("2026-02-03", 100, 112, 94, 105, 1000),
            Bar("2026-02-04", 105, 106, 103, 104, 1000),
        ]
        result = resolve_underlying_path(both, "BULLISH", 99, 101, 95, 110, 10)
        self.assertTrue(result["filled"])
        self.assertEqual(result["exit_reason"], "STOP_FIRST_CONSERVATIVE")
        self.assertEqual(result["exit_price"], 95)

    def test_eod_option_entry_begins_exit_monitoring_next_session(self):
        bars = [
            Bar("2026-02-03", 100, 112, 94, 105, 1000),
            Bar("2026-02-04", 105, 109, 102, 108, 1000),
            Bar("2026-02-05", 108, 111, 107, 110, 1000),
        ]
        result = resolve_underlying_path(
            bars,
            "BULLISH",
            99,
            101,
            95,
            110,
            10,
            include_entry_session_for_exit=False,
        )
        self.assertTrue(result["filled"])
        self.assertEqual(result["entry_date"], "2026-02-03")
        self.assertEqual(result["exit_date"], "2026-02-05")
        self.assertEqual(result["exit_reason"], "TARGET_1")

    def test_right_censored_path_is_not_called_a_horizon_exit(self):
        bars = [
            Bar("2026-02-03", 100, 101, 99, 100, 1000),
            Bar("2026-02-04", 100, 102, 99, 101, 1000),
        ]
        result = resolve_underlying_path(bars, "BULLISH", 99, 101, 95, 110, 10)
        self.assertFalse(result["filled"])
        self.assertTrue(result["entry_zone_touched"])
        self.assertIn("right-censored", result["reason"])

    def test_split_embargo_prevents_boundary_leakage(self):
        self.assertEqual(split_trade("2025-06-27", "2025-07-03", "2025-06-30", "2025-09-30"), "EMBARGO_TRAIN_VALIDATION")
        self.assertEqual(split_trade("2025-07-02", "2025-07-10", "2025-06-30", "2025-09-30"), "VALIDATION")
        self.assertEqual(split_trade("2025-09-29", "2025-10-03", "2025-06-30", "2025-09-30"), "EMBARGO_VALIDATION_TEST")
        self.assertEqual(split_trade("2025-10-02", "2025-10-10", "2025-06-30", "2025-09-30"), "TEST")

    def test_metrics_include_uncertainty_drawdown_and_pop_calibration(self):
        rows = [
            {"signal_date": "2025-01-01", "exit_date": "2025-01-10", "unit_pnl_dollars": 100, "return_on_risk": 0.2, "predicted_pop": 0.7},
            {"signal_date": "2025-01-11", "exit_date": "2025-01-20", "unit_pnl_dollars": -50, "return_on_risk": -0.1, "predicted_pop": 0.6},
            {"signal_date": "2025-01-21", "exit_date": "2025-01-30", "unit_pnl_dollars": 25, "return_on_risk": 0.05, "predicted_pop": 0.55},
        ]
        metrics = replay_metrics(rows)
        self.assertEqual(metrics["n"], 3)
        self.assertAlmostEqual(metrics["total_pnl_dollars"], 75)
        self.assertLess(metrics["max_drawdown_dollars"], 0)
        self.assertIsNotNone(metrics["expectancy_lower_95_dollars"])
        self.assertEqual(metrics["pop_calibration_n"], 3)
        self.assertIsNotNone(metrics["brier_score"])

    def test_monthly_summary_includes_zero_trade_months(self):
        rows = [
            {"signal_date": "2025-10-03", "sized_pnl_dollars": 500},
            {"signal_date": "2025-12-04", "sized_pnl_dollars": -100},
        ]
        summary = monthly_pnl_summary(rows, "2025-10-01", "2025-12-31")
        self.assertEqual(summary["months"], 3)
        self.assertEqual(summary["zero_trade_months"], 1)
        self.assertEqual(summary["total_pnl_dollars"], 400)
        self.assertEqual(summary["series"][1]["pnl_dollars"], 0)


if __name__ == "__main__":
    unittest.main()
