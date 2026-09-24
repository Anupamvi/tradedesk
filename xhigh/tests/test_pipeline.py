import tempfile
import unittest
from pathlib import Path
from xhigh.cli import parse_args
from xhigh.pipeline import _junk_symbol, build_full


class TestCli(unittest.TestCase):
    def test_date_is_full(self):
        args = parse_args(["2026-08-31"])
        self.assertEqual(args.cmd, "full")
        self.assertEqual(args.date, "2026-08-31")

    def test_analyze(self):
        args = parse_args(["analyze", "nvda", "--date", "2026-08-31"])
        self.assertEqual(args.cmd, "analyze")
        self.assertEqual(args.ticker, "NVDA")


class TestEmptyBoard(unittest.TestCase):
    def test_no_schwab_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = build_full("2026-08-31", out_dir=Path(tmp), no_schwab=True)
            self.assertEqual(info["n_trade"], 0)
            board = Path(info["files"]["board"])
            text = board.read_text(encoding="utf-8")
            self.assertIn("CLICK 0", text)
            self.assertIn("PD sort only — conf unchanged.", text)
            self.assertIn("DATA UNAVAILABLE — universe.", text)
            self.assertIn("not an empty click", text.lower())
            self.assertNotIn("None. Empty is valid.", text)


class TestJunk(unittest.TestCase):
    def test_keeps_liquid_names_ending_r_or_w(self):
        for name in ("UBER", "SMR", "FOUR", "LOW", "HOOD", "BA"):
            self.assertFalse(_junk_symbol(name), name)

    def test_drops_warrants(self):
        for name in ("AAPLW", "XYZWS", "FOOWT"):
            self.assertTrue(_junk_symbol(name), name)


if __name__ == "__main__":
    unittest.main()
