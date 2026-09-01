import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from cultra.historical_events import (
    HistoricalEventError,
    event_manifest_payload,
    load_historical_event_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HistoricalEventManifestTests(unittest.TestCase):
    def payload(self):
        return event_manifest_payload(
            provider="INDEPENDENT_TEST_SOURCE",
            source_uri="source:test",
            source_sha256="a" * 64,
            coverage_start=date(2024, 1, 1),
            coverage_end=date(2027, 1, 1),
            covered_tickers=("MSFT", "AAPL"),
            records=(
                {
                    "ticker": "AAPL",
                    "event_type": "EARNINGS",
                    "effective_date": "2026-02-01",
                    "observed_at": "2026-01-01T12:00:00Z",
                    "available_at": "2026-01-01T13:00:00Z",
                    "source_event_id": "earnings-1",
                    "status": "CONFIRMED",
                    "cash_amount": None,
                    "split_ratio": None,
                    "adjustment_reference": None,
                },
            ),
        )

    def write(self, value):
        temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "out")
        path = Path(temporary.name) / "events.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return temporary, path

    def test_point_in_time_event_manifest_loads_and_filters_by_availability(self):
        temporary, path = self.write(self.payload())
        try:
            manifest = load_historical_event_manifest(path)
            before = manifest.known_events(
                ticker="AAPL",
                signal_timestamp=datetime(2026, 1, 1, 12, 30, tzinfo=timezone.utc),
                through_date=date(2026, 2, 15),
            )
            after = manifest.known_events(
                ticker="AAPL",
                signal_timestamp=datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc),
                through_date=date(2026, 2, 15),
            )
        finally:
            temporary.cleanup()
        self.assertEqual((), before)
        self.assertEqual(1, len(after))

    def test_unknown_outcome_field_and_hash_tampering_fail_closed(self):
        value = self.payload()
        value["records"][0]["profitable_after_event"] = True
        temporary, path = self.write(value)
        try:
            with self.assertRaises(HistoricalEventError):
                load_historical_event_manifest(path)
        finally:
            temporary.cleanup()

    def test_dividend_without_cash_economics_fails_closed(self):
        value = self.payload()
        value["records"][0]["event_type"] = "DIVIDEND"
        payload = dict(value)
        payload.pop("manifest_hash")
        import hashlib

        value["manifest_hash"] = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()
        temporary, path = self.write(value)
        try:
            with self.assertRaisesRegex(HistoricalEventError, "cash amount"):
                load_historical_event_manifest(path)
        finally:
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
