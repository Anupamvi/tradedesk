import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wheelo.cli import parse_args
from wheelo.envload import ORATS_TOKEN_MISSING, load_orats_token
from wheelo.orats import cap_tickers, parse_core, redact, reset_process_http

SECRET = "ORATSSECRETTOKENXYZ"


class TestCliFlags(unittest.TestCase):
    def test_token_file_flag_exists(self):
        args = parse_args(["select", "--date", "2026-08-26", "--orats-token-file", "/tmp/x"])
        self.assertEqual(args.orats_token_file, "/tmp/x")
        self.assertEqual(args.cmd, "select")
        self.assertEqual(args.max_orats_requests, 15)

    def test_positional_date_is_full(self):
        args = parse_args(["2026-08-27"])
        self.assertEqual(args.cmd, "full")
        self.assertEqual(args.date, "2026-08-27")

    def test_analyze_needs_ticker(self):
        with self.assertRaises(SystemExit):
            parse_args(["analyze"])


class TestRedact(unittest.TestCase):
    def test_query_and_raw(self):
        url = "https://api.orats.io/datav2/cores?ticker=SOFI&token=%s" % SECRET
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
                load_orats_token(token_file=str(token_path), environ={"ORATS_TOKEN": "from-env"}, code_dir=root),
                "from-file",
            )
            self.assertEqual(load_orats_token(environ={"ORATS_TOKEN": "from-env"}, code_dir=root), "from-env")
            self.assertEqual(load_orats_token(environ={}, code_dir=root), "from-dotenv")

    def test_missing_message(self):
        self.assertIn("ORATS_TOKEN missing", ORATS_TOKEN_MISSING)


class TestParseCore(unittest.TestCase):
    def test_does_not_invent(self):
        empty = parse_core({})
        self.assertIsNone(empty["px"])
        self.assertIsNone(empty["iv30"])
        self.assertFalse(empty["raw"])
        row = {"ticker": "SOFI", "pxAtmIv": 19.2, "iv30d": 45.0, "mktCap": 18000, "nextErn": "0000-00-00"}
        parsed = parse_core(row)
        self.assertEqual(parsed["px"], 19.2)
        self.assertEqual(parsed["iv30"], 45.0)
        self.assertAlmostEqual(parsed["iv30_dec"], 0.45)
        self.assertTrue(parsed["raw"])


class TestOwnList(unittest.TestCase):
    def test_scan_universe_is_own_list(self):
        from wheelo.config import load_own_list, load_scan_universe

        own = load_own_list()
        scan = load_scan_universe()
        self.assertIn("NVDA", own)
        self.assertIn("COST", own)
        self.assertIn("PLTR", own)
        self.assertIn("MU", own)
        self.assertIn("HOOD", own)
        self.assertIn("SPCX", own)
        self.assertIn("SOFI", own)
        self.assertIn("ORCL", own)
        self.assertNotIn("RIVN", own)
        self.assertNotIn("RIOT", own)
        self.assertNotIn("MSTR", own)
        self.assertNotIn("BMNR", own)
        self.assertEqual(scan, own)


class TestCaps(unittest.TestCase):
    def test_cap_tickers(self):
        names = ["A"] * 5 + ["B", "C"] + ["D"] * 40
        out = cap_tickers(names, 3)
        self.assertEqual(out, ["A", "B", "C"])
