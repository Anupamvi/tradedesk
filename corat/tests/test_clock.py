import unittest
from unittest import mock

from corat import clock


class ClockTest(unittest.TestCase):
    def test_new_york_clock_uses_named_market_timezone(self):
        class FakeDateTime:
            @classmethod
            def now(cls, zone=None):
                self.assertIsNotNone(zone)
                self.assertEqual(getattr(zone, "key", ""), "America/New_York")
                from datetime import datetime
                return datetime(2026, 8, 27, 23, 59)

        with mock.patch.object(clock, "datetime", FakeDateTime):
            self.assertEqual(clock.today_new_york(), "2026-08-27")


if __name__ == "__main__":
    unittest.main()
