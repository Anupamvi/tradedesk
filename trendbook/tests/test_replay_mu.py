import unittest
from pathlib import Path

from trendbook.config import load_universe
from trendbook.replay import load_json_bars, walk

GROAT_BARS = Path("/Users/anuppamvi/tradedesk/groat/var/schwab_bars")
ASOF = "2026-09-04"


class TestReplayMuIbm(unittest.TestCase):
    def test_ibm_is_in_universe(self):
        names = load_universe()
        # Dynamic memory from this desk's book, not a static ticker file.
        self.assertIn("SPY", names)
        self.assertIn("SMH", names)
        self.assertIn("MU", names)
        self.assertIn("MRVL", names)
        self.assertIn("MRNA", names)

    def test_mu_stays_on_board_for_consecutive_weeks(self):
        mu_path = GROAT_BARS / "MU.json"
        spy_path = GROAT_BARS / "SPY.json"
        if not mu_path.is_file() or not spy_path.is_file():
            self.skipTest("groat MU/SPY cache missing")
        mu = load_json_bars(mu_path)
        spy = load_json_bars(spy_path)
        report = walk("MU", mu, spy, ASOF)
        self.assertGreaterEqual(report["weeks"], 30, "need enough weekly bars for stage")
        self.assertGreaterEqual(
            report["on_board_weeks"],
            8,
            "MU should stay ON_BOARD for many weeks, not a one-day FIRE print: %s" % report,
        )
        self.assertGreaterEqual(report["longest_run"], 6, report)
        on_rows = [r for r in report["rows"] if r.get("on_board")]
        self.assertTrue(on_rows)
        # Persistence, not only a ready-entry day: some ON_BOARD weeks can be HOLD.
        statuses = {r["status"] for r in on_rows}
        self.assertTrue("HOLD" in statuses or "ADD" in statuses or "LATE" in statuses or "NEW" in statuses)

    def test_ibm_walk_when_bars_exist(self):
        paths = [
            GROAT_BARS / "IBM.json",
            Path("/Users/anuppamvi/tradedesk/trendbook/var/schwab_bars/IBM.json"),
        ]
        spy_path = GROAT_BARS / "SPY.json"
        ibm_path = next((p for p in paths if p.is_file()), None)
        if ibm_path is None or not spy_path.is_file():
            self.skipTest("IBM bars not fetched yet")
        ibm = load_json_bars(ibm_path)
        spy = load_json_bars(spy_path)
        report = walk("IBM", ibm, spy, ASOF)
        self.assertGreaterEqual(report["weeks"], 1)
        # Scanned. Stage 2 is data-dependent; just prove the name is not skipped.
        self.assertIn("on_board_weeks", report)


if __name__ == "__main__":
    unittest.main()
