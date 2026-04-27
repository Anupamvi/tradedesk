import os
import unittest
from unittest.mock import patch


class TestTradeMonitorConfig(unittest.TestCase):
    def test_ntfy_topic_prefers_cloud_env(self):
        from uwos.trade_monitor import load_notify_config

        with patch.dict(os.environ, {"NTFY_TOPIC": "from-env"}, clear=False):
            with patch("dotenv.dotenv_values", return_value={"NTFY_TOPIC": "from-dotenv"}):
                self.assertEqual(load_notify_config()["ntfy_topic"], "from-env")

    def test_ntfy_topic_falls_back_to_dotenv(self):
        from uwos.trade_monitor import load_notify_config

        with patch.dict(os.environ, {}, clear=True):
            with patch("dotenv.dotenv_values", return_value={"NTFY_TOPIC": "from-dotenv"}):
                self.assertEqual(load_notify_config()["ntfy_topic"], "from-dotenv")

    def test_masked_secret_status_never_returns_value(self):
        from uwos.trade_monitor import _masked_secret_status

        self.assertEqual(_masked_secret_status(""), "unset")
        self.assertEqual(_masked_secret_status("abc"), "set")
        self.assertEqual(_masked_secret_status("secret-topic"), "set (12 chars)")

    def test_trade_ideas_excludes_current_underlyings(self):
        from uwos.trade_monitor import run_trade_ideas_scan

        state = {
            "SPREAD:C:2026-05-15:CALL:C1|C2": {"underlying": "C"},
            "AAPL  260515C00200000": {"underlying": "AAPL"},
            "MSFT": {},
        }
        with patch("uwos.trade_monitor.load_state", return_value=state):
            with patch("uwos.trade_ideas.find_latest_data_dir", return_value=None):
                with patch("uwos.trade_ideas.scan_trade_ideas", return_value=[]) as scan:
                    self.assertEqual(run_trade_ideas_scan(), [])

        exclude = scan.call_args.kwargs["exclude_tickers"]
        self.assertIn("C", exclude)
        self.assertIn("AAPL", exclude)
        self.assertIn("MSFT", exclude)

    def test_after_hours_movement_gate_detects_market_and_watch_moves(self):
        from uwos.trade_monitor import _after_hours_movement_from_quotes

        quotes = {
            "SPY": {"quote": {"netPercentChange": 0.7}},
            "AAPL": {"quote": {"lastPrice": 103, "closePrice": 100}},
            "MSFT": {"quote": {"netPercentChange": 0.5}},
        }
        with patch.dict(os.environ, {
            "AFTER_HOURS_MARKET_MOVE_PCT": "0.6",
            "AFTER_HOURS_WATCH_MOVE_PCT": "2.0",
        }, clear=False):
            should_run, reason = _after_hours_movement_from_quotes(quotes, ["AAPL", "MSFT"])

        self.assertTrue(should_run)
        self.assertIn("SPY +0.7%", reason)
        self.assertIn("AAPL +3.0%", reason)

    def test_after_hours_movement_gate_skips_quiet_quotes(self):
        from uwos.trade_monitor import _after_hours_movement_from_quotes

        quotes = {
            "SPY": {"quote": {"netPercentChange": 0.2}},
            "AAPL": {"quote": {"lastPrice": 101, "closePrice": 100}},
        }
        with patch.dict(os.environ, {
            "AFTER_HOURS_MARKET_MOVE_PCT": "0.6",
            "AFTER_HOURS_WATCH_MOVE_PCT": "2.0",
        }, clear=False):
            should_run, reason = _after_hours_movement_from_quotes(quotes, ["AAPL"])

        self.assertFalse(should_run)
        self.assertIn("after-hours quiet", reason)


if __name__ == "__main__":
    unittest.main()
