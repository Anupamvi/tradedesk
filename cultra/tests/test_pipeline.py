import json
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from cultra.artifacts import verify_manifest
from cultra.cli import build_parser
from cultra.pipeline import (
    CultraPipeline,
    LiveExecutionDisabled,
    PipelineError,
    PipelineInputs,
    PipelineRunConfig,
    reference_request_budget,
    run_doctor,
)
from cultra.reports import CandidateRow


class PipelineTests(unittest.TestCase):
    def test_doctor_pass_is_explicitly_not_production_or_profit_evidence(self):
        payload = run_doctor().to_dict()
        self.assertEqual("OFFLINE_ENGINEERING_ONLY", payload["scope"])
        self.assertFalse(payload["production_ready"])
        self.assertFalse(payload["manual_ticket_enabled"])
        self.assertEqual("UNPROVEN", payload["profit_confidence"])

    def test_initial_run_is_unproven_zero_request_zero_ticket_and_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = CultraPipeline().run(
                PipelineRunConfig(
                    as_of=date(2026, 8, 30),
                    output_root=Path(temporary),
                    run_id="initial-offline",
                    created_at=datetime(2026, 8, 30, 20, tzinfo=timezone.utc),
                )
            )
            self.assertEqual(result.ticket_count, 0)
            self.assertEqual(verify_manifest(result.run_dir), ())
            board = result.board_path.read_text()
            self.assertIn("Overall status: `UNPROVEN`", board)
            self.assertIn("No manual-review tickets", board)
            request_plan = json.loads((result.run_dir / "request_plan.json").read_text())
            self.assertEqual(request_plan["logical_count"], 0)
            self.assertEqual(request_plan["charged_attempts"], 0)
            self.assertFalse(request_plan["network_attempted"])
            self.assertEqual(len(request_plan["plan_hash"]), 64)
            manifest = json.loads((result.run_dir / "manifest.json").read_text())
            self.assertFalse(manifest["metadata"]["order_submission_surface"])
            self.assertFalse(manifest["metadata"]["network_attempted"])
            self.assertEqual(manifest["request_plan_id"], request_plan["plan_hash"])
            self.assertRegex(manifest["source_fingerprint"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                set(manifest["field_profile_statuses"].values()),
                {"DOCUMENTED_NOT_PROBED"},
            )
            required = {
                "orats_request_plan.json",
                "orats_request_ledger.json",
                "orats_cache_report.json",
                "orats_data_vintage_manifest.json",
                "data_health.md",
                "quotes.json",
                "candidates.json",
                "pop_calculations.json",
                "edge_calculations.json",
                "promotion_decisions.json",
                "model_artifacts.json",
                "field_profiles.json",
            }
            self.assertTrue(
                required.issubset({item.path for item in result.manifest.artifacts})
            )
            data_health = (result.run_dir / "data_health.md").read_text()
            self.assertIn("Actual outbound HTTP attempts: 0", data_health)
            self.assertIn("zero-request offline run", data_health)
            field_profiles = json.loads(
                (result.run_dir / "field_profiles.json").read_text()
            )
            self.assertEqual(
                set(field_profiles),
                {
                    "CORE_SCREEN_V1",
                    "SUMMARY_ENRICH_V1",
                    "MONEY_IMPLIED_V1",
                    "MONEY_FORECAST_V1",
                    "EXACT_OPTION_V1",
                },
            )
            self.assertTrue(
                all(
                    item["status"] == "DOCUMENTED_NOT_PROBED"
                    for item in field_profiles.values()
                )
            )
            evidence = json.loads(
                (result.run_dir / "strategy_evidence.json").read_text()
            )
            self.assertEqual(len(evidence), len(manifest["strategy_states"]))
            self.assertTrue(all(item["state"] == "UNPROVEN" for item in evidence))

    def test_live_flag_fails_before_creating_an_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(LiveExecutionDisabled):
                CultraPipeline().run(
                    PipelineRunConfig(
                        as_of=date(2026, 8, 30),
                        output_root=Path(temporary),
                        run_id="must-not-exist",
                        execute_orats=True,
                    )
                )
            self.assertFalse((Path(temporary) / "must-not-exist").exists())

    def test_all_candidates_remain_in_outputs_without_top_n(self):
        values = tuple(
            CandidateRow(
                "candidate-%03d" % index,
                "SYM%d" % index,
                "LONG_CALL",
                "still visible",
                "WATCHLIST",
                index / 1000.0,
            )
            for index in range(70)
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = CultraPipeline().run(
                PipelineRunConfig(
                    as_of=date(2026, 8, 30),
                    output_root=Path(temporary),
                    run_id="uncapped",
                ),
                PipelineInputs(watchlist=values),
            )
            payload = json.loads((result.run_dir / "watchlist.json").read_text())
            self.assertEqual(len(payload), 70)
            self.assertIn("candidate-069", result.board_path.read_text())

    def test_ineligible_ticket_fails_closed(self):
        edge = SimpleNamespace(
            net_expected_value=10.0,
            conservative_net_expected_value=2.0,
            maximum_loss=100.0,
        )
        bad_ticket = SimpleNamespace(
            evidence_state="SHADOW_PASS",
            quantity="USER DETERMINED",
            edge=edge,
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(PipelineError):
                CultraPipeline().run(
                    PipelineRunConfig(
                        as_of=date(2026, 8, 30),
                        output_root=Path(temporary),
                        run_id="bad-ticket",
                        overall_status="EVIDENCE_GATED_ACTIVE",
                    ),
                    PipelineInputs(tickets=(bad_ticket,)),
                )
            self.assertFalse((Path(temporary) / "bad-ticket").exists())

    def test_reference_request_budget_is_conservative_and_credential_free(self):
        budget = reference_request_budget()
        self.assertEqual(budget["target_logical_requests"], 25)
        self.assertEqual(budget["base_logical_request_ceiling"], 60)
        self.assertEqual(budget["logical_request_ceiling"], 60)
        self.assertEqual(budget["maximum_retry_reserve"], 0)
        self.assertEqual(budget["maximum_planned_charged_attempts"], 60)
        self.assertLess(budget["maximum_planned_charged_attempts"], 100)
        self.assertFalse(budget["credential_loaded"])
        self.assertFalse(budget["network_attempted"])

    def test_failed_family_labeled_promotion_decision_reaches_board(self):
        decision = {
            "strategy_family": "LONG_CALL",
            "current_state": "UNPROVEN",
            "target_state": "RESEARCH_PASS",
            "passed": False,
            "reasons": ["training expectancy is not positive"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            result = CultraPipeline().run(
                PipelineRunConfig(
                    as_of=date(2026, 8, 30),
                    output_root=Path(temporary),
                    run_id="rejected-family",
                ),
                PipelineInputs(promotion_decisions=(decision,)),
            )
            board = result.board_path.read_text()
            self.assertIn(
                "LONG_CALL (`REJECTED`): training expectancy is not positive",
                board,
            )
            self.assertEqual(
                result.manifest.strategy_states["LONG_CALL"],
                "REJECTED",
            )

    def test_cli_has_no_secret_destination_override(self):
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "secrets-bootstrap",
                    "--source",
                    "/tmp/source.env",
                    "--destination",
                    "/tmp/forbidden.env",
                ]
            )


if __name__ == "__main__":
    unittest.main()
