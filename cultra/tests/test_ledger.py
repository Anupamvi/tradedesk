import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from cultra.ledger import (
    ORATS_ACCOUNT_SCOPE,
    ActiveRunError,
    BudgetExhausted,
    CircuitOpen,
    LedgerUnavailable,
    PermitStateError,
    RequestLedger,
)
from cultra.requesting import (
    Endpoint,
    PlanningError,
    RequestPlan,
    RunType,
    make_planned_request,
)


def planned(identifier, ticker, retry_limit=0):
    return make_planned_request(
        logical_request_id=identifier,
        endpoint=Endpoint.CORES,
        run_type=RunType.EOD,
        entities=[ticker],
        fields=["ticker", "tradeDate"],
        field_profile="CORE_V1",
        purpose="offline ledger test",
        expected_vintage="2026-08-29",
        expected_rows=1,
        expected_bytes=100,
        retry_limit=retry_limit,
    )


def plan(run_id="ledger-run", retry=True):
    del retry
    requests = (
        planned("request-one", "AAPL", 0),
        planned("request-two", "MSFT", 0),
    )
    return RequestPlan(
        run_id=run_id,
        run_type=RunType.EOD,
        requests=requests,
        target=2,
        hard_cap=2,
        retry_reserve=0,
    )


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.allowed_root = Path(self.temp.name) / "cultra-ledger-root"
        self.allowed_root.mkdir()
        self.root_patch = mock.patch(
            "cultra.ledger.CULTRA_LEDGER_ROOT", self.allowed_root
        )
        self.root_patch.start()
        self.path = self.allowed_root / "ledger.sqlite3"

    def tearDown(self):
        self.root_patch.stop()
        self.temp.cleanup()

    def test_ledger_path_must_be_cultra_local_and_sqlite(self):
        with self.assertRaises(LedgerUnavailable):
            RequestLedger(Path(self.temp.name) / "outside.sqlite3")
        with self.assertRaises(LedgerUnavailable):
            RequestLedger(self.allowed_root / "ledger.db")

    def test_permit_is_committed_before_send_and_survives_restart(self):
        request_plan = plan()
        ledger = RequestLedger(self.path)
        ledger.start_run(request_plan)
        permit = ledger.reserve_attempt(request_plan.run_id, "request-one")
        self.assertEqual(1, permit.network_attempt_number)
        restarted = RequestLedger(self.path)
        self.assertEqual(1, restarted.summary(request_plan.run_id)["charged_attempts"])
        self.assertEqual({"reserved": 1}, restarted.summary(request_plan.run_id)["attempt_states"])
        with self.assertRaises(PermitStateError):
            restarted.reserve_attempt(request_plan.run_id, "request-one")

    def test_aborted_exact_plan_can_resume_but_charged_permit_cannot_be_reused(self):
        request_plan = plan("resume-exact")
        ledger = RequestLedger(self.path)
        ledger.start_run(request_plan)
        ledger.reserve_attempt(request_plan.run_id, "request-one")
        ledger.finish_run(request_plan.run_id, aborted=True)
        ledger.reactivate_aborted_run(request_plan)
        self.assertEqual("active", ledger.summary(request_plan.run_id)["state"])
        with self.assertRaises(PermitStateError):
            ledger.reserve_attempt(request_plan.run_id, "request-one")
        second = ledger.reserve_attempt(request_plan.run_id, "request-two")
        self.assertEqual(2, second.network_attempt_number)

        ledger.finish_run(request_plan.run_id, aborted=True)
        changed = plan("different-plan-id")
        with self.assertRaises(ActiveRunError):
            ledger.reactivate_aborted_run(changed)

    def test_confirmed_completed_state_and_export(self):
        request_plan = plan(retry=False)
        ledger = RequestLedger(self.path)
        ledger.start_run(request_plan)
        permit = ledger.reserve_attempt(request_plan.run_id, "request-one")
        ledger.mark_indeterminate(permit)
        ledger.mark_confirmed(
            permit,
            status_code=200,
            rows_returned=1,
            bytes_returned=12,
            duration_ms=2.5,
            provider_trade_date="2026-08-29",
        )
        ledger.mark_completed(permit, outcome_code="SUCCESS")
        summary = ledger.summary(request_plan.run_id)
        self.assertEqual({"completed": 1}, summary["attempt_states"])
        artifact = ledger.export(request_plan.run_id, Path(self.temp.name) / "ledger.json")
        text = artifact.read_text()
        self.assertIn('"network_attempt_number": 1', text)
        self.assertNotIn("token", text.lower())
        payload = json.loads(text)
        self.assertEqual("CULTRA_LEDGER_EXPORT_V2", payload["schema_version"])
        self.assertEqual(1, payload["summary"]["actual_logical_requests"])
        self.assertEqual(2, payload["summary"]["planned_logical_requests"])
        self.assertEqual(1, payload["summary"]["outbound_http_attempts"])
        self.assertEqual(0, payload["summary"]["redirects"])
        self.assertEqual(1, payload["summary"]["symbols_requested"])
        self.assertEqual(0, payload["summary"]["contracts_requested"])
        self.assertEqual(1, payload["summary"]["rows_downloaded"])
        self.assertEqual(12, payload["summary"]["total_response_bytes"])
        self.assertEqual(1, payload["summary"]["rows_returned"])
        self.assertEqual(12, payload["summary"]["bytes_returned"])
        self.assertEqual(1, payload["attempts"][0]["entity_count"])
        self.assertEqual("CORE_V1", payload["attempts"][0]["field_profile"])
        self.assertEqual("MISS", payload["attempts"][0]["cache_status"])
        self.assertFalse(payload["attempts"][0]["cache_hit"])
        self.assertEqual(1, payload["attempts"][0]["symbol_count"])
        self.assertEqual(0, payload["attempts"][0]["contract_count"])
        self.assertEqual(1, payload["attempts"][0]["budget_remaining"])

    def test_hard_cap_blocks_next_attempt(self):
        request_plan = plan()
        ledger = RequestLedger(self.path)
        ledger.start_run(request_plan)
        ledger.reserve_attempt(request_plan.run_id, "request-one")
        ledger.reserve_attempt(request_plan.run_id, "request-two")
        with self.assertRaises(BudgetExhausted):
            ledger.reserve_attempt(request_plan.run_id, "request-two")
        self.assertEqual(0, ledger.summary(request_plan.run_id)["remaining"])

    def test_admitted_attempt_ceiling_is_stricter_than_hard_cap(self):
        base = plan("admitted")
        request_plan = RequestPlan(
            run_id=base.run_id,
            run_type=base.run_type,
            requests=base.requests,
            target=base.target,
            hard_cap=99,
            retry_reserve=base.retry_reserve,
        )
        ledger = RequestLedger(self.path)
        ledger.start_run(request_plan)
        ledger.reserve_attempt(request_plan.run_id, "request-one")
        ledger.reserve_attempt(request_plan.run_id, "request-two")
        with self.assertRaises(BudgetExhausted):
            ledger.reserve_attempt(request_plan.run_id, "request-two")
        summary = ledger.summary(request_plan.run_id)
        self.assertEqual(99, summary["hard_cap"])
        self.assertEqual(2, summary["admitted_attempt_cap"])
        self.assertEqual(0, summary["remaining"])

    def test_concurrent_final_permit_has_one_winner(self):
        requests = (
            planned("request-one", "AAPL"),
            planned("request-two", "MSFT"),
            planned("request-three", "NVDA"),
        )
        request_plan = RequestPlan(
            run_id="concurrent-final",
            run_type=RunType.EOD,
            requests=requests,
            target=3,
            hard_cap=3,
            retry_reserve=0,
        )
        ledger = RequestLedger(self.path)
        ledger.start_run(request_plan)
        ledger.reserve_attempt(request_plan.run_id, "request-one")
        ledger.reserve_attempt(request_plan.run_id, "request-two")
        barrier = threading.Barrier(17)
        winners = []
        failures = []

        def race():
            barrier.wait()
            try:
                winners.append(
                    ledger.reserve_attempt(request_plan.run_id, "request-three")
                )
            except (BudgetExhausted, PermitStateError):
                failures.append(1)

        threads = [threading.Thread(target=race) for _ in range(16)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(1, len(winners))
        self.assertEqual(15, len(failures))
        self.assertEqual(3, ledger.summary(request_plan.run_id)["charged_attempts"])

    def test_one_active_run_per_credential(self):
        first = plan("first", retry=False)
        second = plan("second", retry=False)
        ledger = RequestLedger(self.path)
        ledger.start_run(first)
        with self.assertRaises(ActiveRunError):
            ledger.start_run(second)
        ledger.finish_run(first.run_id)
        ledger.start_run(second)

    def test_campaign_cap_is_cumulative_across_completed_runs(self):
        ledger = RequestLedger(self.path)

        def campaign_plan(run_id, ticker):
            item = make_planned_request(
                logical_request_id="request-" + run_id,
                endpoint=Endpoint.HIST_CORES,
                run_type=RunType.HISTORICAL_BACKFILL,
                entities=(ticker,),
                fields=("ticker", "tradeDate"),
                field_profile="HIST_CORE_V1",
                purpose="cumulative campaign test",
                expected_vintage="2026-08-29",
                expected_rows=100,
                expected_bytes=1_000,
                retry_limit=0,
            )
            return RequestPlan(
                run_id=run_id,
                run_type=RunType.HISTORICAL_BACKFILL,
                requests=(item,),
                target=1,
                hard_cap=1,
                retry_reserve=0,
                campaign_id="historical-campaign",
                campaign_hard_cap=2,
            )

        first = campaign_plan("phase-one", "AAPL")
        second = campaign_plan("phase-two", "MSFT")
        third = campaign_plan("phase-three", "NVDA")
        ledger.start_run(first)
        ledger.reserve_attempt(first.run_id, first.requests[0].logical_request_id)
        ledger.finish_run(first.run_id)
        ledger.start_run(second)
        ledger.reserve_attempt(second.run_id, second.requests[0].logical_request_id)
        ledger.finish_run(second.run_id)
        self.assertEqual(
            {"campaign_id": "historical-campaign", "hard_cap": 2,
             "charged_attempts": 2, "remaining": 0},
            ledger.campaign_summary("historical-campaign"),
        )
        with self.assertRaises(BudgetExhausted):
            ledger.start_run(third)

    def test_account_scope_is_fixed_and_caller_cannot_supply_one(self):
        ledger = RequestLedger(self.path)
        request_plan = plan("fixed-scope", retry=False)
        ledger.start_run(request_plan)
        self.assertEqual(
            ORATS_ACCOUNT_SCOPE, ledger.summary(request_plan.run_id)["account_scope"]
        )
        with self.assertRaises(TypeError):
            ledger.start_run(plan("invalid-scope", retry=False), credential_scope="caller")

    def test_oversized_eod_response_reservations_fail_before_send(self):
        requests = tuple(
            planned("request-%02d" % index, "T%03d" % index)
            for index in range(11)
        )
        with self.assertRaisesRegex(PlanningError, "response-byte reservations"):
            RequestPlan(
                run_id="response-budget",
                run_type=RunType.EOD,
                requests=requests,
                target=18,
                hard_cap=11,
                retry_reserve=0,
            )

    def test_provider_circuit_opens_after_three_consecutive_failures(self):
        request_plan = plan("provider-circuit", retry=False)
        ledger = RequestLedger(self.path)
        ledger.start_run(request_plan)
        ledger.record_provider_result(request_plan.run_id, success=False)
        ledger.record_provider_result(request_plan.run_id, success=False)
        self.assertEqual("closed", ledger.summary(request_plan.run_id)["provider_circuit_state"])
        ledger.record_provider_result(request_plan.run_id, success=False)
        summary = ledger.summary(request_plan.run_id)
        self.assertEqual("open", summary["provider_circuit_state"])
        self.assertEqual(3, summary["consecutive_provider_failures"])
        with self.assertRaises(CircuitOpen):
            ledger.reserve_attempt(request_plan.run_id, "request-one")

    def test_schema_structurally_rejects_attempt_number_100(self):
        request_plan = plan(retry=False)
        ledger = RequestLedger(self.path)
        ledger.start_run(request_plan)
        connection = sqlite3.connect(str(self.path))
        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO attempts(
                    run_id, logical_request_id, network_attempt_number,
                    retry_number, endpoint, method, request_fingerprint,
                    response_byte_reservation, state, reserved_at
                ) SELECT run_id, logical_request_id, 100, 0, endpoint, method,
                         request_fingerprint, max_response_bytes, 'reserved', 0
                  FROM planned_requests LIMIT 1
                """
            )
        connection.close()

    def test_replaced_ledger_fails_closed(self):
        ledger = RequestLedger(self.path)
        replacement = self.path.with_suffix(".replacement")
        os.replace(self.path, replacement)
        self.path.write_bytes(b"not sqlite")
        with self.assertRaises(LedgerUnavailable):
            ledger.assert_healthy()

    def test_v1_schema_is_migrated_in_place(self):
        connection = sqlite3.connect(str(self.path))
        connection.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                credential_key TEXT NOT NULL,
                run_type TEXT NOT NULL,
                plan_hash TEXT NOT NULL,
                hard_cap INTEGER NOT NULL,
                target INTEGER NOT NULL,
                retry_reserve INTEGER NOT NULL,
                state TEXT NOT NULL,
                started_at REAL NOT NULL,
                ended_at REAL
            );
            CREATE TABLE planned_requests (
                run_id TEXT NOT NULL,
                logical_request_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                retry_limit INTEGER NOT NULL,
                contingency INTEGER NOT NULL,
                PRIMARY KEY(run_id, logical_request_id)
            );
            CREATE TABLE attempts (
                permit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                logical_request_id TEXT NOT NULL,
                network_attempt_number INTEGER NOT NULL,
                retry_number INTEGER NOT NULL,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                request_fingerprint TEXT NOT NULL,
                state TEXT NOT NULL,
                reserved_at REAL NOT NULL,
                send_started_at REAL,
                confirmed_at REAL,
                completed_at REAL,
                status_code INTEGER,
                rows_returned INTEGER,
                bytes_returned INTEGER,
                duration_ms REAL,
                provider_trade_date TEXT,
                updated_at_min TEXT,
                updated_at_max TEXT,
                outcome_code TEXT
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta VALUES ('schema_version', 'CULTRA_LEDGER_V1');
            """
        )
        connection.close()
        RequestLedger(self.path)
        connection = sqlite3.connect(str(self.path))
        run_columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        request_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(planned_requests)")
        }
        version = connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        connection.close()
        self.assertIn("admitted_attempt_cap", run_columns)
        self.assertIn("response_byte_cap", run_columns)
        self.assertIn("circuit_state", run_columns)
        self.assertIn("entity_count", request_columns)
        self.assertIn("field_profile", request_columns)
        self.assertIn("max_response_bytes", request_columns)
        self.assertIn("campaign_id", run_columns)
        self.assertEqual("CULTRA_LEDGER_V4", version)


if __name__ == "__main__":
    unittest.main()
