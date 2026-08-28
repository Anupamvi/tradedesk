import tempfile
import unittest
from pathlib import Path

from corat.earnings import fetch_forward_earnings_calendar
from corat.store import write_json


class EarningsCalendarTest(unittest.TestCase):
    def test_cached_forward_calendar_identifies_estimated_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "nasdaq_earnings" / "2026-08-28.json"
            write_json(
                path,
                {
                    "fetched_at_utc": "2026-08-27T22:00:00Z",
                    "payload": {"data": {"rows": [{"symbol": "DELL", "time": "time-after-hours"}]}},
                },
            )
            result = fetch_forward_earnings_calendar(
                "2026-08-27", 1, root, offline=True,
            )
            self.assertEqual(result.dates_by_ticker["DELL"], "2026-08-28")
            self.assertEqual(result.traces[0].source, "NASDAQ EARNINGS CALENDAR (ESTIMATED)")
            self.assertFalse(result.errors)


if __name__ == "__main__":
    unittest.main()
