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

    def test_phone_notify_config_reads_env(self):
        from uwos.trade_monitor import load_notify_config

        with patch.dict(
            os.environ,
            {
                "NTFY_PHONE_TOPIC": "phone-topic",
                "NTFY_MANUAL_TOPIC": "manual-topic",
                "MANUAL_ALERT_PREFIX": "TRADE WATCH",
                "MANUAL_ALERT_TAGS": "rotating_light,bell",
                "PHONE_NOTIFY_MODE": "both",
                "TWILIO_ACCOUNT_SID": "sid",
                "TWILIO_AUTH_TOKEN": "token",
                "TWILIO_FROM": "+15550000000",
                "SMS_TO": "+15551112222",
            },
            clear=False,
        ):
            with patch("dotenv.dotenv_values", return_value={}):
                cfg = load_notify_config()

        self.assertEqual(cfg["ntfy_phone_topic"], "phone-topic")
        self.assertEqual(cfg["ntfy_manual_topic"], "manual-topic")
        self.assertEqual(cfg["manual_alert_prefix"], "TRADE WATCH")
        self.assertEqual(cfg["manual_alert_tags"], "rotating_light,bell")
        self.assertEqual(cfg["phone_notify_mode"], "both")
        self.assertEqual(cfg["twilio_account_sid"], "sid")
        self.assertEqual(cfg["sms_to"], "+15551112222")

    def test_phone_mode_enabled(self):
        from uwos.trade_monitor import _phone_mode_enabled

        self.assertTrue(_phone_mode_enabled("ntfy", "ntfy"))
        self.assertTrue(_phone_mode_enabled("both", "sms"))
        self.assertTrue(_phone_mode_enabled("ntfy,sms", "sms"))
        self.assertFalse(_phone_mode_enabled("off", "ntfy"))
        self.assertFalse(_phone_mode_enabled("", "sms"))

    def test_masked_secret_status_never_returns_value(self):
        from uwos.trade_monitor import _masked_secret_status

        self.assertEqual(_masked_secret_status(""), "unset")
        self.assertEqual(_masked_secret_status("abc"), "set")
        self.assertEqual(_masked_secret_status("secret-topic"), "set (12 chars)")

    def test_manual_notify_uses_distinct_manual_topic_and_style(self):
        from uwos import trade_monitor

        cfg = {
            "ntfy_server": "https://ntfy.sh",
            "ntfy_topic": "regular-topic",
            "ntfy_token": "",
            "ntfy_phone_topic": "phone-topic",
            "ntfy_manual_topic": "manual-topic",
            "manual_alert_prefix": "MANUAL MONITOR",
            "manual_alert_tags": "rotating_light,warning",
            "phone_notify_mode": "ntfy",
            "twilio_account_sid": "",
            "twilio_auth_token": "",
            "twilio_from": "",
            "sms_to": "",
        }
        with patch("uwos.trade_monitor.load_notify_config", return_value=cfg):
            with patch("uwos.trade_monitor.send_ntfy", return_value=True) as send_ntfy:
                trade_monitor.notify(
                    "[CLOSE] CLOSE: MU",
                    "Trigger hit",
                    priority="high",
                    tags="chart_with_upwards_trend",
                    critical=True,
                    manual=True,
                )

        self.assertEqual(send_ntfy.call_count, 2)
        regular_call, manual_call = send_ntfy.call_args_list
        self.assertEqual(regular_call.args[0], "regular-topic")
        self.assertEqual(regular_call.args[1], "MANUAL MONITOR - [CLOSE] CLOSE: MU")
        self.assertEqual(regular_call.args[3], "urgent")
        self.assertIn("rotating_light", regular_call.args[4])
        self.assertEqual(manual_call.args[0], "manual-topic")
        self.assertEqual(manual_call.args[1], "MANUAL MONITOR - [CLOSE] CLOSE: MU")

    def test_manual_notify_falls_back_to_phone_topic(self):
        from uwos import trade_monitor

        cfg = {
            "ntfy_server": "https://ntfy.sh",
            "ntfy_topic": "regular-topic",
            "ntfy_token": "",
            "ntfy_phone_topic": "phone-topic",
            "ntfy_manual_topic": "",
            "manual_alert_prefix": "MANUAL MONITOR",
            "manual_alert_tags": "rotating_light,warning",
            "phone_notify_mode": "ntfy",
            "twilio_account_sid": "",
            "twilio_auth_token": "",
            "twilio_from": "",
            "sms_to": "",
        }
        with patch("uwos.trade_monitor.load_notify_config", return_value=cfg):
            with patch("uwos.trade_monitor.send_ntfy", return_value=True) as send_ntfy:
                trade_monitor.notify("ROLL: SPY", "Trigger hit", manual=True)

        self.assertEqual(send_ntfy.call_args_list[1].args[0], "phone-topic")

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
