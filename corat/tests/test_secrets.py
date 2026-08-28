import tempfile
import unittest
from pathlib import Path

from corat.secrets import load_env, orats_token, parse_env_text, redact


class SecretsTest(unittest.TestCase):
    def test_parse_env_quotes_and_export(self):
        values = parse_env_text("export ORATS_TOKEN='abc'\nOTHER=two\n# ignored\n")
        self.assertEqual(values["ORATS_TOKEN"], "abc")
        self.assertEqual(values["OTHER"], "two")

    def test_environment_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("ORATS_TOKEN=file\n", encoding="utf-8")
            self.assertEqual(orats_token(root, {"ORATS_TOKEN": "environment"}), "environment")

    def test_redaction_removes_query_and_raw_secret(self):
        secret = "VERYSECRET123"
        text = "https://api.test/x?a=1&token={}&x=2 Authorization={}".format(secret, secret)
        safe = redact(text, secret)
        self.assertNotIn(secret, safe)
        self.assertIn("token=REDACTED", safe)

