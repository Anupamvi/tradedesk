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

    def test_manual_monitor_triggers_on_spread_close_debit(self):
        from uwos.trade_monitor import evaluate_manual_monitors

        item = {
            "kind": "SPREAD",
            "key": "SPREAD:META:2026-06-18:PUT:short|long",
            "group": {
                "underlying": "META",
                "expiry": "2026-06-18",
                "put_call": "PUT",
                "strategy": "Bull Put Credit",
                "net_type": "credit",
                "qty": 1,
                "width": 10,
                "short_symbol": "short",
                "long_symbol": "long",
                "short_strike": 600,
                "long_strike": 590,
                "short_leg": {
                    "symbol": "short",
                    "qty": -1,
                    "avg_cost": 14.92,
                    "live_quote": {"ask": 17.2},
                    "greeks": {"delta": -0.46},
                    "underlying_quote": {"last": 599},
                    "computed": {"dte": 34, "unrealized_pnl": -210.0, "max_profit": 1492.0},
                },
                "long_leg": {
                    "symbol": "long",
                    "qty": 1,
                    "avg_cost": 11.67,
                    "live_quote": {"bid": 13.1},
                    "greeks": {"delta": -0.33},
                    "underlying_quote": {"last": 599},
                    "computed": {"dte": 34, "unrealized_pnl": 168.0, "max_loss": 1167.0},
                },
            },
        }
        monitor = {
            "id": "META-test",
            "ticker": "META",
            "expiry": "2026-06-18",
            "put_call": "PUT",
            "short_strike": 600,
            "long_strike": 590,
            "critical_spot": 600,
            "critical_close_debit": 4.5,
            "critical_short_delta": 0.45,
        }

        with patch("uwos.trade_monitor.load_manual_monitors", return_value=[monitor]):
            alerts, state = evaluate_manual_monitors([item], {})

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["verdict"], "CLOSE")
        self.assertTrue(alerts[0]["manual_monitor"])
        self.assertIn("Sell $600P / Buy $590P", alerts[0]["legs"])
        self.assertEqual(state["MANUAL:META-test"]["verdict"], "CLOSE")

    def test_manual_monitor_ignores_position_with_blank_symbol(self):
        from uwos.trade_monitor import evaluate_manual_monitors

        item = {
            "kind": "POSITION",
            "key": "",
            "position": {
                "symbol": "",
                "asset_type": "CASH",
                "qty": 0,
                "computed": {},
            },
        }
        with patch("uwos.trade_monitor.load_manual_monitors", return_value=[]):
            alerts, state = evaluate_manual_monitors([item], {})

        self.assertEqual(alerts, [])
        self.assertEqual(state, {})

    def test_manual_monitor_suppresses_generic_matching_spread(self):
        from uwos.trade_monitor import manual_suppressed_position_keys

        item = {
            "kind": "SPREAD",
            "key": "SPREAD:CRM:2026-05-29:CALL:short|long",
            "group": {
                "underlying": "CRM",
                "expiry": "2026-05-29",
                "put_call": "CALL",
                "strategy": "Bear Call Credit",
                "net_type": "credit",
                "qty": 1,
                "width": 5,
                "short_symbol": "short",
                "long_symbol": "long",
                "short_strike": 180,
                "long_strike": 185,
                "short_leg": {
                    "symbol": "short",
                    "qty": -1,
                    "avg_cost": 4.72,
                    "live_quote": {"ask": 6.0},
                    "greeks": {"delta": 0.41},
                    "underlying_quote": {"last": 173},
                    "computed": {"dte": 14, "unrealized_pnl": -30.0, "max_profit": 472.0},
                },
                "long_leg": {
                    "symbol": "long",
                    "qty": 1,
                    "avg_cost": 3.47,
                    "live_quote": {"bid": 4.0},
                    "greeks": {"delta": 0.32},
                    "underlying_quote": {"last": 173},
                    "computed": {"dte": 14, "unrealized_pnl": 0.0, "max_loss": 347.0},
                },
            },
        }
        monitor = {
            "id": "CRM-test",
            "ticker": "CRM",
            "expiry": "2026-05-29",
            "put_call": "CALL",
            "short_strike": 180,
            "long_strike": 185,
        }

        with patch("uwos.trade_monitor.load_manual_monitors", return_value=[monitor]):
            suppressed = manual_suppressed_position_keys([item])

        self.assertEqual(suppressed, {"SPREAD:CRM:2026-05-29:CALL:short|long"})

    def test_manual_monitor_does_not_alert_profit_by_default(self):
        from uwos.trade_monitor import evaluate_manual_monitors

        item = {
            "kind": "POSITION",
            "key": "META 260821P00510000",
            "position": {
                "symbol": "META 260821P00510000",
                "underlying": "META",
                "asset_type": "OPTION",
                "put_call": "PUT",
                "strike": 510,
                "expiry": "2026-08-21",
                "qty": -1,
                "avg_cost": 42.0,
                "live_quote": {"ask": 10.0},
                "greeks": {"delta": -0.15},
                "underlying_quote": {"last": 613},
                "computed": {"dte": 98, "unrealized_pnl": 3200.0, "pct_of_max_profit": 75.0},
            },
        }
        monitor = {
            "id": "META-profit-test",
            "ticker": "META",
            "expiry": "2026-08-21",
            "put_call": "PUT",
            "strike": 510,
            "profit_close_debit": 11.0,
        }

        with patch("uwos.trade_monitor.load_manual_monitors", return_value=[monitor]):
            alerts, state = evaluate_manual_monitors([item], {})

        self.assertEqual(alerts, [])
        self.assertEqual(state["MANUAL:META-profit-test"]["verdict"], "HOLD")

    def test_risk_filter_suppresses_profit_and_allows_failure(self):
        from uwos.trade_monitor import is_risk_management_alert, _suppress_profit_taking_verdict

        self.assertFalse(is_risk_management_alert({
            "verdict": "CLOSE",
            "reason": "90% of max profit — nothing left to harvest",
        }))
        self.assertFalse(is_risk_management_alert({
            "verdict": "CLOSE",
            "reason": "Bull Put Credit 700/690 spread: expiration week with limited profit captured; gamma risk now dominates theta",
        }))
        self.assertTrue(is_risk_management_alert({
            "verdict": "ROLL",
            "reason": "short leg ITM with 7 DTE; roll or close both legs together",
        }))
        verdict, reason, suppressed = _suppress_profit_taking_verdict(
            "ASSESS",
            "equity up +135% — consider trimming",
        )
        self.assertTrue(suppressed)
        self.assertEqual(verdict, "HOLD")
        self.assertIn("risk/failure", reason)


if __name__ == "__main__":
    unittest.main()
