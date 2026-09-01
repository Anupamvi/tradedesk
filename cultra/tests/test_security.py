import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cultra.security import SecretError, bootstrap_orats_env, contains_secret, read_named_env_key


class SecurityTests(unittest.TestCase):
    def test_named_key_parser_and_bootstrap_copy_only_requested_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.env"
            source.write_text("ORATS_TOKEN='canary-value'\nUNRELATED=do-not-copy\n", encoding="utf-8")
            os.chmod(source, 0o600)
            destination = root / "cultra.env"
            with mock.patch("cultra.security.CULTRA_ENV_PATH", destination):
                written = bootstrap_orats_env(source)
            self.assertEqual(written, destination)
            self.assertEqual(read_named_env_key(destination, "ORATS_TOKEN"), "canary-value")
            self.assertNotIn("UNRELATED", destination.read_text(encoding="utf-8"))
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)

    def test_bootstrap_api_has_no_arbitrary_destination_surface(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.env"
            source.write_text("ORATS_TOKEN=value\n", encoding="utf-8")
            os.chmod(source, 0o600)
            with self.assertRaises(TypeError):
                bootstrap_orats_env(source, root / "outside.env")

    def test_rejects_non_private_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.env"
            source.write_text("ORATS_TOKEN=value\n", encoding="utf-8")
            os.chmod(source, 0o644)
            with self.assertRaises(SecretError):
                read_named_env_key(source, "ORATS_TOKEN")

    def test_detects_raw_and_encoded_canaries(self):
        self.assertTrue(contains_secret("x abc/123 y", ["abc/123"]))
        self.assertTrue(contains_secret("x abc%2F123 y", ["abc/123"]))
        self.assertFalse(contains_secret("clean", ["abc/123"]))


if __name__ == "__main__":
    unittest.main()
