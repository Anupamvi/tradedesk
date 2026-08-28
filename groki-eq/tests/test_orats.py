import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from groki_eq.cli import main, parse_args
from groki_eq.envload import ORATS_TOKEN_MISSING, load_merged_env, load_orats_token, load_schwab_env
from groki_eq.orats import map_dailies_row, probe_spy_dailies, redact


SECRET = "ORATSSECRETTOKENXYZ"


def _dailies_payload():
    return {
        "data": [
            {
                "ticker": "SPY",
                "tradeDate": "2026-08-26",
                "clsPx": 646.81,
                "hiPx": 648.0,
                "loPx": 644.0,
                "open": 645.0,
            }
        ]
    }


class TestCliFlags(unittest.TestCase):
    def test_token_file_flag_exists(self):
        args = parse_args(["--date", "2026-08-26", "--orats-token-file", "/tmp/x"])
        self.assertEqual(args.orats_token_file, "/tmp/x")

    def test_no_orats_token_argv(self):
        with self.assertRaises(SystemExit):
            parse_args(["--date", "2026-08-26", "--orats-token", "nope"])


class TestRedact(unittest.TestCase):
    def test_query_and_raw(self):
        url = "https://api.orats.io/datav2/hist/dailies?ticker=SPY&token=%s" % SECRET
        self.assertNotIn(SECRET, redact(url, SECRET))
        self.assertIn("token=REDACTED", redact(url, SECRET))


class TestTokenLoad(unittest.TestCase):
    def test_missing_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(load_orats_token(environ={}, code_dir=Path(tmp)))

    def test_file_beats_env_beats_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("ORATS_TOKEN=from-dotenv\n", encoding="utf-8")
            token_path = root / "token.txt"
            token_path.write_text("from-file\n", encoding="utf-8")
            self.assertEqual(
                load_orats_token(
                    token_file=str(token_path),
                    environ={"ORATS_TOKEN": "from-env"},
                    code_dir=root,
                ),
                "from-file",
            )
            self.assertEqual(
                load_orats_token(environ={"ORATS_TOKEN": "from-env"}, code_dir=root),
                "from-env",
            )
            self.assertEqual(load_orats_token(environ={}, code_dir=root), "from-dotenv")

    def test_dotenv_ignores_first_non_token_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "SCHWAB_API_KEY=abc\nORATS_TOKEN=real-token\n",
                encoding="utf-8",
            )
            self.assertEqual(load_orats_token(environ={}, code_dir=root), "real-token")

    def test_merged_env_fills_schwab_from_parent(self):
        env = load_merged_env()
        self.assertTrue((env.get("ORATS_TOKEN") or "").strip())
        self.assertTrue((env.get("SCHWAB_API_KEY") or "").strip())
        self.assertTrue((env.get("SCHWAB_TOKEN_PATH") or "").strip())
        self.assertTrue(Path(env["SCHWAB_TOKEN_PATH"]).is_absolute())
        schwab = load_schwab_env()
        self.assertEqual(schwab.get("SCHWAB_API_KEY"), env.get("SCHWAB_API_KEY"))


class TestFieldMap(unittest.TestCase):
    def test_dailies_keys(self):
        mapped = map_dailies_row(_dailies_payload()["data"][0])
        self.assertEqual(mapped["close"]["key"], "clsPx")
        self.assertEqual(mapped["high"]["key"], "hiPx")
        self.assertEqual(mapped["low"]["key"], "loPx")
        self.assertEqual(mapped["open"]["key"], "open")

    def test_probe_writes_map_without_token(self):
        def getter(path, query, token):
            self.assertEqual(path, "/hist/dailies")
            self.assertEqual(query.get("ticker"), "SPY")
            return 200, _dailies_payload(), ""

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive"
            with mock.patch("groki_eq.orats.archive_dir", return_value=archive):
                out = probe_spy_dailies(SECRET, getter=getter)
            self.assertTrue(out["ok"])
            text = (archive / "field_map.json").read_text(encoding="utf-8")
            self.assertNotIn(SECRET, text)
            payload = json.loads(text)
            self.assertEqual(payload["close"]["key"], "clsPx")


class TestMissingTokenExit(unittest.TestCase):
    def test_exit_2(self):
        with mock.patch("groki_eq.cli.load_orats_token", return_value=None):
            with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
                code = main(["--date", "2026-08-26"])
        self.assertEqual(code, 2)
        self.assertIn("ORATS_TOKEN missing", ORATS_TOKEN_MISSING)


if __name__ == "__main__":
    unittest.main()
