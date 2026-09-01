import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from cultra.historical_events import EVENT_TYPES
from cultra.prerequisites import (
    HistoricalPrerequisiteError,
    load_historical_prerequisites,
    prepare_historical_prerequisites,
    save_historical_prerequisites,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "out"


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_artifact(root, name):
    path = root / (name + ".source.txt")
    payload = ("independent fixture bytes for %s\n" % name).encode("utf-8")
    path.write_bytes(payload)
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "role": name.upper(),
        "source_uri": "https://%s.example.test/raw" % name,
        "media_type": "text/plain",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_source_fixture(root):
    sessions = []
    current = date(2024, 1, 2)
    while len(sessions) < 450:
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    retrieved_at = datetime(2027, 1, 2, tzinfo=timezone.utc)
    session_source = root / "session_source.json"
    _write(
        session_source,
        {
            "schema": "cultra.market-session-source.v2",
            "provider": "INDEPENDENT_EXCHANGE_CALENDAR",
            "source_uri": "https://calendar.example.test/xnys",
            "retrieved_at": retrieved_at.isoformat(),
            "exchange": "XNYS",
            "timezone": "America/New_York",
            "complete": True,
            "source_artifacts": [_source_artifact(root, "sessions")],
            "sessions": [
                {
                    "session_date": item.isoformat(),
                    "close_at": datetime.combine(
                        item, time(16, 0), ZoneInfo("America/New_York")
                    ).isoformat(),
                }
                for item in sessions
            ],
        },
    )
    selection_dates = (sessions[0], sessions[120], sessions[240], sessions[360])
    snapshots = []
    all_tickers = []
    stock_tickers = []
    ticker_block = {}
    for block, observed_at in enumerate(selection_dates):
        members = []
        for index in range(120):
            ticker = "B%dT%03d" % (block, index)
            asset_type = "STOCK" if index < 110 else "ETF"
            members.append(
                {
                    "ticker": ticker,
                    "asset_type": asset_type,
                    "optionable": True,
                    "sampling_stratum": "S%02d" % (index % 10),
                    "liquidity_rank": index + 1,
                    "classification_status": "VERIFIED_POINT_IN_TIME",
                    "classification_source_roles": ["UNIVERSE"],
                }
            )
            all_tickers.append(ticker)
            ticker_block[ticker] = block
            if asset_type == "STOCK":
                stock_tickers.append(ticker)
        snapshots.append({"observed_at": observed_at.isoformat(), "members": members})
    universe_source = root / "universe_source.json"
    _write(
        universe_source,
        {
            "schema": "cultra.point-in-time-universe-source.v2",
            "provider": "INDEPENDENT_OPTIONABLE_DIRECTORY",
            "source_uri": "https://universe.example.test/archive",
            "retrieved_at": retrieved_at.isoformat(),
            "universe_id": "test-point-in-time-universe",
            "coverage": "US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_ACROSS_2_CBOE_VENUES",
            "point_in_time": True,
            "survivorship_free": True,
            "source_artifacts": [_source_artifact(root, "universe")],
            "snapshots": snapshots,
        },
    )
    records = []
    for index, ticker in enumerate(stock_tickers):
        effective = selection_dates[ticker_block[ticker]]
        records.append(
            {
                "ticker": ticker,
                "event_type": "EARNINGS",
                "effective_date": effective.isoformat(),
                "observed_at": datetime(
                    effective.year,
                    effective.month,
                    effective.day,
                    tzinfo=timezone.utc,
                ).isoformat(),
                "available_at": datetime(
                    effective.year,
                    effective.month,
                    effective.day,
                    1,
                    tzinfo=timezone.utc,
                ).isoformat(),
                "source_event_id": "earnings-%04d" % index,
                "status": "CONFIRMED",
                "cash_amount": None,
                "split_ratio": None,
                "adjustment_reference": None,
            }
        )
    event_source = root / "event_source.json"
    _write(
        event_source,
        {
            "schema": "cultra.historical-event-source.v2",
            "provider": "INDEPENDENT_EVENT_ARCHIVE",
            "source_uri": "https://events.example.test/archive",
            "retrieved_at": retrieved_at.isoformat(),
            "coverage_start": sessions[0].isoformat(),
            "coverage_end": sessions[-1].isoformat(),
            "covered_tickers": sorted(all_tickers),
            "complete_event_types": list(EVENT_TYPES),
            "point_in_time_revisions": True,
            "coverage_attestation": "COMPLETE_FOR_COVERED_TICKERS_AND_EVENT_TYPES",
            "source_artifacts": [_source_artifact(root, "events")],
            "records": records,
        },
    )
    return universe_source, session_source, event_source


class HistoricalPrerequisiteTests(unittest.TestCase):
    def test_sources_are_normalized_hash_bound_and_reproducible(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            universe, sessions, events = build_source_fixture(root)
            prepared = prepare_historical_prerequisites(
                input_set_id="test-inputs-v1",
                universe_source_path=universe,
                session_source_path=sessions,
                event_source_path=events,
            )
            self.assertEqual(40, len(prepared.sampled_symbols))
            self.assertEqual(4, len(prepared.selection_dates))
            freeze_path = save_historical_prerequisites(root / "frozen", prepared)
            frozen = load_historical_prerequisites(freeze_path)
            self.assertEqual("test-inputs-v1", frozen.input_set_id)
            self.assertFalse(frozen.payload["network_attempted"])
            self.assertFalse(frozen.payload["orats_source_used"])
            self.assertTrue(frozen.payload["raw_sources_hash_bound"])

    def test_raw_source_drift_fails_reproduction(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            universe, sessions, events = build_source_fixture(root)
            prepared = prepare_historical_prerequisites(
                input_set_id="test-drift-v1",
                universe_source_path=universe,
                session_source_path=sessions,
                event_source_path=events,
            )
            freeze_path = save_historical_prerequisites(root / "frozen", prepared)
            raw = json.loads(events.read_text(encoding="utf-8"))
            raw["source_uri"] = "https://events.example.test/changed"
            _write(events, raw)
            with self.assertRaisesRegex(HistoricalPrerequisiteError, "source changed"):
                load_historical_prerequisites(freeze_path)

    def test_preserved_provider_byte_drift_fails_reproduction(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            universe, sessions, events = build_source_fixture(root)
            prepared = prepare_historical_prerequisites(
                input_set_id="test-provider-byte-drift-v2",
                universe_source_path=universe,
                session_source_path=sessions,
                event_source_path=events,
            )
            freeze_path = save_historical_prerequisites(root / "frozen", prepared)
            raw = json.loads(events.read_text(encoding="utf-8"))
            artifact_path = PROJECT_ROOT / raw["source_artifacts"][0]["path"]
            artifact_path.write_text("changed provider bytes\n", encoding="utf-8")
            with self.assertRaisesRegex(HistoricalPrerequisiteError, "artifact changed"):
                load_historical_prerequisites(freeze_path)

    def test_wrapper_only_v1_schema_is_rejected(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            universe, sessions, events = build_source_fixture(root)
            raw = json.loads(sessions.read_text(encoding="utf-8"))
            raw["schema"] = "cultra.market-session-source.v1"
            raw.pop("source_artifacts")
            _write(sessions, raw)
            with self.assertRaisesRegex(HistoricalPrerequisiteError, "unsupported"):
                prepare_historical_prerequisites(
                    input_set_id="test-wrapper-only-v1",
                    universe_source_path=universe,
                    session_source_path=sessions,
                    event_source_path=events,
                )

    def test_current_only_universe_and_orats_sources_are_rejected(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            universe, sessions, events = build_source_fixture(root)
            raw = json.loads(universe.read_text(encoding="utf-8"))
            raw["snapshots"] = raw["snapshots"][-1:]
            _write(universe, raw)
            with self.assertRaisesRegex(HistoricalPrerequisiteError, "four exact"):
                prepare_historical_prerequisites(
                    input_set_id="test-current-only",
                    universe_source_path=universe,
                    session_source_path=sessions,
                    event_source_path=events,
                )
            universe, sessions, events = build_source_fixture(root)
            raw = json.loads(sessions.read_text(encoding="utf-8"))
            raw["provider"] = "ORATS"
            _write(sessions, raw)
            with self.assertRaisesRegex(HistoricalPrerequisiteError, "independent of ORATS"):
                prepare_historical_prerequisites(
                    input_set_id="test-orats-source",
                    universe_source_path=universe,
                    session_source_path=sessions,
                    event_source_path=events,
                )

    def test_empty_event_attestation_cannot_cover_sampled_stocks(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            universe, sessions, events = build_source_fixture(root)
            raw = json.loads(events.read_text(encoding="utf-8"))
            raw["records"] = []
            _write(events, raw)
            with self.assertRaisesRegex(HistoricalPrerequisiteError, "no earnings evidence"):
                prepare_historical_prerequisites(
                    input_set_id="test-empty-events",
                    universe_source_path=universe,
                    session_source_path=sessions,
                    event_source_path=events,
                )


if __name__ == "__main__":
    unittest.main()
