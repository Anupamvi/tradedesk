import json
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from cultra.sessions import (
    SessionCalendarError,
    load_historical_session_calendar,
    session_calendar_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sessions(count=450):
    values = []
    current = date(2024, 1, 2)
    while len(values) < count:
        if current.weekday() < 5:
            values.append(
                {
                    "session_date": current.isoformat(),
                    "close_at": datetime.combine(
                        current, time(16, 0), ZoneInfo("America/New_York")
                    ).isoformat(),
                }
            )
        current += timedelta(days=1)
    return values


class SessionCalendarTests(unittest.TestCase):
    def test_exact_session_calendar_is_hash_bound_and_timezone_aware(self):
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT)) as temporary:
            path = Path(temporary) / "sessions.json"
            payload = session_calendar_payload(
                provider="test-provider",
                source_uri="cultra://test/sessions",
                source_sha256="a" * 64,
                sessions=_sessions(),
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_historical_session_calendar(path)
            self.assertEqual(450, len(loaded.sessions))
            self.assertEqual(payload["calendar_hash"], loaded.calendar_hash)
            self.assertEqual("2024-01-02", loaded.dates[0])

    def test_unknown_fields_naive_timestamps_and_wrong_counts_fail_closed(self):
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT)) as temporary:
            root = Path(temporary)
            base = session_calendar_payload(
                provider="test-provider",
                source_uri="cultra://test/sessions",
                source_sha256="a" * 64,
                sessions=_sessions(),
            )
            extra = dict(base, future_close="known")
            extra_path = root / "extra.json"
            extra_path.write_text(json.dumps(extra), encoding="utf-8")
            with self.assertRaisesRegex(SessionCalendarError, "unfrozen fields"):
                load_historical_session_calendar(extra_path)

            naive_sessions = _sessions()
            naive_sessions[0]["close_at"] = "2024-01-02T16:00:00"
            naive = session_calendar_payload(
                provider="test-provider",
                source_uri="cultra://test/sessions",
                source_sha256="a" * 64,
                sessions=naive_sessions,
            )
            naive_path = root / "naive.json"
            naive_path.write_text(json.dumps(naive), encoding="utf-8")
            with self.assertRaisesRegex(SessionCalendarError, "timezone-aware"):
                load_historical_session_calendar(naive_path)

            short = session_calendar_payload(
                provider="test-provider",
                source_uri="cultra://test/sessions",
                source_sha256="a" * 64,
                sessions=_sessions(449),
            )
            short_path = root / "short.json"
            short_path.write_text(json.dumps(short), encoding="utf-8")
            with self.assertRaisesRegex(SessionCalendarError, "exactly 450"):
                load_historical_session_calendar(short_path)


if __name__ == "__main__":
    unittest.main()
