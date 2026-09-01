import hashlib
import json
import tempfile
import unittest
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from cultra.backfill import BackfillError, execute_chain_backfill
from cultra.campaign import (
    CampaignFreezeError,
    build_historical_campaign_freeze,
    load_historical_campaign_freeze,
    save_historical_campaign_freeze,
)
from cultra.historical_events import EVENT_TYPES
from cultra.prerequisites import (
    prepare_historical_prerequisites,
    save_historical_prerequisites,
)
from cultra.requesting import RequestPlan


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "out"


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_artifact(root, name):
    path = root / (name + ".source.txt")
    payload = ("independent campaign fixture bytes for %s\n" % name).encode("utf-8")
    path.write_bytes(payload)
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "role": name.upper(),
        "source_uri": "https://%s.example.test/raw" % name,
        "media_type": "text/plain",
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _fixture(root):
    values = []
    current = date(2024, 1, 2)
    while len(values) < 450:
        if current.weekday() < 5:
            values.append(current)
        current += timedelta(days=1)
    sessions = tuple(values)
    selection_dates = (sessions[0], sessions[120], sessions[240], sessions[360])
    snapshots = []
    all_tickers = []
    stock_tickers = []
    ticker_blocks = {}
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
            ticker_blocks[ticker] = block
            if asset_type == "STOCK":
                stock_tickers.append(ticker)
        snapshots.append({"observed_at": observed_at.isoformat(), "members": members})
    retrieved_at = datetime(2027, 1, 2, tzinfo=ZoneInfo("UTC"))
    universe_path = root / "universe_source.json"
    _write_json(
        universe_path,
        {
            "schema": "cultra.point-in-time-universe-source.v2",
            "universe_id": "campaign-test-universe",
            "provider": "independent-test-source",
            "source_uri": "https://universe.example.test/archive",
            "retrieved_at": retrieved_at.isoformat(),
            "coverage": "US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_ACROSS_2_CBOE_VENUES",
            "point_in_time": True,
            "survivorship_free": True,
            "source_artifacts": [_source_artifact(root, "universe")],
            "snapshots": snapshots,
        },
    )
    session_path = root / "session_source.json"
    _write_json(
        session_path,
        {
            "schema": "cultra.market-session-source.v2",
            "provider": "independent-test-source",
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
    records = []
    for index, ticker in enumerate(stock_tickers):
        effective = selection_dates[ticker_blocks[ticker]]
        records.append(
            {
                "ticker": ticker,
                "event_type": "EARNINGS",
                "effective_date": effective.isoformat(),
                "observed_at": datetime.combine(
                    effective, time(0, 0), ZoneInfo("UTC")
                ).isoformat(),
                "available_at": datetime.combine(
                    effective, time(1, 0), ZoneInfo("UTC")
                ).isoformat(),
                "source_event_id": "earnings-%04d" % index,
                "status": "CONFIRMED",
                "cash_amount": None,
                "split_ratio": None,
                "adjustment_reference": None,
            }
        )
    events_path = root / "event_source.json"
    _write_json(
        events_path,
        {
            "schema": "cultra.historical-event-source.v2",
            "provider": "independent-test-source",
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
    prepared = prepare_historical_prerequisites(
        input_set_id="campaign-test-inputs",
        universe_source_path=universe_path,
        session_source_path=session_path,
        event_source_path=events_path,
    )
    prerequisite_path = save_historical_prerequisites(root / "prerequisites", prepared)
    return prerequisite_path, events_path


class CampaignFreezeTests(unittest.TestCase):
    def test_complete_campaign_freeze_reproduces_all_requests_and_slices(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            prerequisites, _ = _fixture(root)
            campaign = build_historical_campaign_freeze(
                campaign_id="test-campaign-v2",
                prerequisite_freeze_path=prerequisites,
            )
            self.assertEqual(474, campaign.payload["request_campaign"]["expected_attempts"])
            self.assertEqual(
                [90, 90, 90, 90, 90, 24],
                campaign.payload["request_campaign"]["slice_attempts"],
            )
            self.assertFalse(campaign.payload["network_attempted"])
            freeze_path = save_historical_campaign_freeze(root / "campaign", campaign)
            loaded = load_historical_campaign_freeze(freeze_path)
            self.assertEqual(campaign.payload, loaded.payload)
            self.assertEqual(
                [item.plan_hash for item in campaign.slices],
                [item.plan_hash for item in loaded.slices],
            )

    def test_source_drift_and_nonfrozen_execution_plan_fail_before_network(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            prerequisites, events = _fixture(root)
            campaign = build_historical_campaign_freeze(
                campaign_id="test-campaign-drift",
                prerequisite_freeze_path=prerequisites,
            )
            freeze_path = save_historical_campaign_freeze(root / "campaign", campaign)
            original = campaign.slices[0]
            altered = RequestPlan(
                run_id=original.run_id,
                run_type=original.run_type,
                requests=tuple(reversed(original.requests)),
                target=original.target,
                hard_cap=original.hard_cap,
                retry_reserve=original.retry_reserve,
                campaign_id=original.campaign_id,
                campaign_hard_cap=original.campaign_hard_cap,
            )
            with self.assertRaisesRegex(BackfillError, "immutable campaign slice"):
                execute_chain_backfill(
                    altered,
                    output_root=root,
                    campaign_freeze_path=freeze_path,
                    slice_index=0,
                )
            raw = json.loads(events.read_text(encoding="utf-8"))
            raw["source_uri"] = "cultra://test/events-drifted"
            _write_json(events, raw)
            with self.assertRaisesRegex(CampaignFreezeError, "source changed"):
                load_historical_campaign_freeze(freeze_path)


if __name__ == "__main__":
    unittest.main()
