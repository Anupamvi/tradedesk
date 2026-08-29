import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from groat.cli import parse_args
from groat.envload import ORATS_TOKEN_MISSING, load_orats_token, load_merged_env
from groat.orats import fetch_cores, map_cores_row, parse_core, redact

SECRET = "ORATSSECRETTOKENXYZ"


class TestCliFlags(unittest.TestCase):
    def test_token_file_flag_exists(self):
        args = parse_args(["full", "--date", "2026-08-26", "--orats-token-file", "/tmp/x"])
        self.assertEqual(args.orats_token_file, "/tmp/x")
        self.assertEqual(args.cmd, "full")

    def test_no_orats_token_argv(self):
        with self.assertRaises(SystemExit):
            parse_args(["full", "--date", "2026-08-26", "--orats-token", "nope"])

    def test_positional_date_is_full_scan(self):
        args = parse_args(["2026-08-27"])
        self.assertEqual(args.cmd, "full")
        self.assertEqual(args.date, "2026-08-27")

    def test_full_then_date(self):
        args = parse_args(["full", "2026-08-27"])
        self.assertEqual(args.cmd, "full")
        self.assertEqual(args.date, "2026-08-27")
        self.assertIsNone(args.ticker)


class TestRedact(unittest.TestCase):
    def test_query_and_raw(self):
        url = "https://api.orats.io/datav2/cores?ticker=SPY&token=%s" % SECRET
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
            self.assertEqual(load_orats_token(environ={"ORATS_TOKEN": "from-env"}, code_dir=root), "from-env")
            self.assertEqual(load_orats_token(environ={}, code_dir=root), "from-dotenv")

    def test_dotenv_ignores_first_non_token_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("SCHWAB_API_KEY=abc\nORATS_TOKEN=real-token\n", encoding="utf-8")
            self.assertEqual(load_orats_token(environ={}, code_dir=root), "real-token")

    def test_missing_message(self):
        self.assertIn("ORATS_TOKEN missing", ORATS_TOKEN_MISSING)

    def test_merged_env_has_orats_and_schwab(self):
        env = load_merged_env()
        self.assertTrue((env.get("ORATS_TOKEN") or "").strip())
        self.assertTrue((env.get("SCHWAB_API_KEY") or "").strip())
        self.assertTrue((env.get("SCHWAB_TOKEN_PATH") or "").strip())
        self.assertTrue(Path(env["SCHWAB_TOKEN_PATH"]).is_absolute())
        dumped = json.dumps(env)
        # never leak into a serialized dump in this test beyond the env itself
        self.assertIn("ORATS_TOKEN", dumped)


class TestParseCore(unittest.TestCase):
    def test_map_and_parse_do_not_invent(self):
        row = {
            "ticker": "NVDA",
            "tradeDate": "2026-08-26",
            "iv30d": 32.5,
            "orHv20d": 28.0,
            "ivPctile1y": 40,
            "orFcst20d": 30.0,
            "nextErn": "0000-00-00",
        }
        mapped = map_cores_row(row)
        self.assertEqual(mapped["iv30"]["key"], "iv30d")
        self.assertEqual(mapped["hv20"]["key"], "orHv20d")
        parsed = parse_core(row)
        self.assertEqual(parsed["iv30"], 32.5)
        self.assertEqual(parsed["hv20"], 28.0)
        self.assertAlmostEqual(parsed["vrp"], 4.5)
        empty = parse_core({})
        self.assertIsNone(empty["iv30"])
        self.assertIsNone(empty["hv20"])
        self.assertIsNone(empty["vrp"])
        self.assertFalse(empty["raw"])


class TestRefreshBypassesCache(unittest.TestCase):
    def test_fetch_cores_refresh_hits_http(self):
        calls = []

        def getter(path, query, token):
            calls.append(path)
            return 200, {"data": [{"ticker": "SPY", "iv30d": 12.0, "orHv20d": 10.0}]}, ""

        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive"
            (archive / "2026-08-28").mkdir(parents=True)
            (archive / "2026-08-28" / "cores.json").write_text(
                json.dumps({"data": [{"ticker": "SPY", "iv30d": 99.0, "orHv20d": 99.0}]}),
                encoding="utf-8",
            )
            with mock.patch("groat.orats.archive_dir", return_value=archive):
                with mock.patch("groat.orats.can_http", return_value=True):
                    stale = fetch_cores(
                        "2026-08-28", ["SPY"], "tok", "2026-08-28", getter=getter, refresh=False
                    )
                    fresh = fetch_cores(
                        "2026-08-28", ["SPY"], "tok", "2026-08-28", getter=getter, refresh=True
                    )
        self.assertEqual(stale["rows"]["SPY"]["iv30d"], 99.0)
        self.assertEqual(stale["http"], 0)
        self.assertEqual(fresh["rows"]["SPY"]["iv30d"], 12.0)
        self.assertGreaterEqual(fresh["http"], 1)
        self.assertTrue(calls)


class TestTapeRefresh(unittest.TestCase):
    def test_ensure_bars_refresh_bypasses_schwab_cache(self):
        from groat.prices import ensure_bars

        bars = [{"date": "2026-08-28", "open": 1, "high": 2, "low": 1, "close": 1.5, "volume": 10}]
        with mock.patch("groat.prices.load_cached_bars", return_value=[]):
            with mock.patch("groat.schwab.price_history_bars", return_value=bars) as hist:
                out = ensure_bars("SPY", "tok", asof="2026-08-28", live=False, refresh=True)
        self.assertEqual(hist.call_args.kwargs.get("use_cache"), False)
        self.assertEqual(out["tape"], "schwab_history")


if __name__ == "__main__":
    unittest.main()
