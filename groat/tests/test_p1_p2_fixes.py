import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from groat.catalysts import load_catalyst
from groat.picks import score_option_ticket
from groat.xintel import missing_x_tickers


class TestCatalysts(unittest.TestCase):
    def test_missing_stays_unavailable(self):
        out = load_catalyst("news", "2026-09-03", "NOW")
        self.assertEqual(out["summary"], "DATA UNAVAILABLE")

    def test_reads_agent_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "var" / "filings" / "2026-09-03"
            day.mkdir(parents=True)
            (day / "XOM.json").write_text(json.dumps({"summary": "10-Q filed"}), encoding="utf-8")
            with mock.patch("groat.catalysts.CODE_DIR", root):
                out = load_catalyst("filings", "2026-09-03", "XOM")
            self.assertEqual(out["summary"], "10-Q filed")


class TestXhotDoesNotSatisfyXintel(unittest.TestCase):
    def test_xhot_tag_is_not_xintel(self):
        row = {"ticker": "NOW", "x": "DATA UNAVAILABLE", "xhot": {"tag": "Crowded"}}
        self.assertEqual(missing_x_tickers([row]), ["NOW"])


class TestDeskPickNoChaseBonus(unittest.TestCase):
    def test_already_ran_is_a_penalty(self):
        base = {
            "choice": "OPTIONS",
            "naive_pop": 0.40,
            "opt_conf": 70,
            "score": 60,
            "x": "Quiet",
            "close": 145.0,
            "picked": {"long_strike": 146.0, "delta": 0.20, "instrument": "debit_call_spread"},
        }
        quiet = dict(base, ret_1=0.0)
        rip = dict(base, ret_1=0.05)
        self.assertLess(score_option_ticket(rip), score_option_ticket(quiet))


class TestSchwabPositionsError(unittest.TestCase):
    def test_failed_fetch_is_error_not_empty_book(self):
        from groat.schwab import load_positions

        with mock.patch("groat.schwab.schwab_credentials", return_value={"token_path": "x"}):
            with mock.patch("groat.schwab._access_token", return_value="tok"):
                with mock.patch("groat.schwab._get_json", return_value=None):
                    rows, err = load_positions()
        self.assertEqual(rows, [])
        self.assertIn("DATA UNAVAILABLE", err)

    def test_option_underlying_not_occ_as_ticker(self):
        from groat.book import schwab_held_index

        idx = schwab_held_index(
            [
                {
                    "ticker": "XOM",
                    "symbol": "XOM   260925C00165000",
                    "asset": "OPTION",
                    "quantity": 1,
                }
            ]
        )
        self.assertIn("XOM", idx)
        self.assertEqual(idx["XOM"]["legs"][0]["right"], "call")


class TestAnalogHttpBudget(unittest.TestCase):
    def test_allow_http_does_not_pass_exhausted_max_requests(self):
        from groat.evidence import _fetch_analog_strikes

        seen = {}

        def fake_fetch(day, tickers, token, today, getter=None, max_requests=None, dte="", refresh=False):
            seen["max_requests"] = max_requests
            return {"rows": {tickers[0]: [{"strike": 1}]}, "http": 1}

        with mock.patch("groat.evidence.fetch_strikes", side_effect=fake_fetch):
            rows = _fetch_analog_strikes("NOW", "2026-01-02", "tok", "2026-09-03", None, 0, True, [0])
        self.assertIsNone(seen["max_requests"])
        self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
