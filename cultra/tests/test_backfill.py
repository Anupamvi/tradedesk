import json
import tempfile
import unittest
from pathlib import Path

from cultra.backfill import (
    BackfillError,
    _private_json_once,
    build_broad_cohort_backfill_plan,
    build_chain_backfill_plan,
    _partition_manifest_payload,
    load_recent_sessions,
    load_validation_config,
    plan_chain_slices,
)
from cultra.requesting import Endpoint, RequestPlan, RunType, make_planned_request


class HistoricalBackfillPlanningTests(unittest.TestCase):
    def setUp(self):
        self.config = load_validation_config()
        self.sessions = tuple(
            "2025-01-%02d" % day for day in range(1, 10)
        )

    def test_legacy_date_grid_plan_is_disabled(self):
        with self.assertRaisesRegex(BackfillError, "date-grid backfill is disabled"):
            build_chain_backfill_plan(
                run_id="backfill-test",
                sessions=self.sessions,
                slice_index=1,
                slice_size=4,
                config=self.config,
            )

    def test_empty_and_oversized_slices_fail_closed(self):
        with self.assertRaises(BackfillError):
            build_chain_backfill_plan(
                run_id="empty",
                sessions=self.sessions,
                slice_index=10,
                slice_size=4,
                config=self.config,
            )
        with self.assertRaises(BackfillError):
            plan_chain_slices(self.sessions, slice_size=91)

    def test_session_calendar_requires_450_unique_sorted_dates(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calendar.json"
            path.write_text(json.dumps({"recent_sessions": list(self.sessions)}))
            with self.assertRaises(BackfillError):
                load_recent_sessions(path)

    def test_legacy_broad_date_grid_plan_is_disabled(self):
        with self.assertRaisesRegex(BackfillError, "broad date-grid backfill is disabled"):
            build_broad_cohort_backfill_plan(
                run_id="broad-cohort",
                sessions=("2025-03-03",),
                symbols=tuple("T%02d" % index for index in range(20)),
                slice_index=0,
                sessions_per_slice=1,
            )

    def test_partition_manifest_reconciles_multiple_batches_per_date(self):
        requests = []
        for trade_date in ("2025-03-03", "2025-03-04"):
            for batch_index in range(2):
                requests.append(
                    make_planned_request(
                        logical_request_id="%s-%d" % (trade_date, batch_index),
                        endpoint=Endpoint.HIST_STRIKES,
                        run_type=RunType.HISTORICAL_BACKFILL,
                        entities=("T%02d" % batch_index,),
                        fields=("ticker", "tradeDate"),
                        field_profile="HIST_ROTATING_COHORT_CHAIN_V2",
                        purpose="partition reconciliation",
                        expected_vintage=trade_date,
                        expected_rows=100,
                        expected_bytes=1000,
                        retry_limit=0,
                        params={
                            "tradeDate": trade_date,
                            "dte": "20,180",
                            "delta": "0.01,0.99",
                        },
                    )
                )
        plan = RequestPlan(
            run_id="optimized-reconcile",
            run_type=RunType.HISTORICAL_BACKFILL,
            requests=tuple(requests),
            target=4,
            hard_cap=90,
            retry_reserve=0,
            campaign_id="optimized-campaign-slice-00",
            campaign_hard_cap=90,
        )
        completed = {
            item.logical_request_id: {
                "logical_request_id": item.logical_request_id,
                "trade_date": item.expected_vintage,
                "entities": list(item.entities),
            }
            for item in plan.requests
        }
        manifest = _partition_manifest_payload(plan, completed, {}, {})
        self.assertTrue(manifest["complete"])
        self.assertEqual(4, len(manifest["completed_requests"]))
        self.assertEqual(2, len(manifest["date_coverage"]))
        self.assertTrue(
            all(
                len(item["completed_request_ids"]) == 2
                for item in manifest["date_coverage"].values()
            )
        )

    def test_checkpoint_identity_is_idempotent_but_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.json"
            _private_json_once(path, {"plan_hash": "abc"})
            _private_json_once(path, {"plan_hash": "abc"})
            with self.assertRaisesRegex(BackfillError, "drifted"):
                _private_json_once(path, {"plan_hash": "changed"})
            self.assertEqual({"plan_hash": "abc"}, json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
