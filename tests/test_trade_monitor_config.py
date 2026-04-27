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


if __name__ == "__main__":
    unittest.main()
