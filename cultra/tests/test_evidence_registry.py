import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from cultra.catalog import CATALOG_VERSION
from cultra.evidence_registry import (
    DEFAULT_EVIDENCE_ROOT,
    EvidenceMutationError,
    EvidencePartitions,
    EvidenceRegistry,
    EvidenceRetestError,
    FrozenEvidenceIdentity,
    HoldoutReuseError,
    RegistryState,
    TerminalEvidenceError,
)


NOW = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)


def identity(**changes):
    values = dict(
        strategy_family="CALL_DEBIT_VERTICAL",
        catalog_version=CATALOG_VERSION,
        hypothesis_fingerprint="sha256:hypothesis-v1",
        cost_model_version="cost-v1",
        exit_policy_version="exit-v1",
        pop_model_version="pop-v1",
        pop_model_artifact_id="sha256:model-v1",
        model_frozen_at=NOW - timedelta(minutes=1),
    )
    values.update(changes)
    return FrozenEvidenceIdentity(**values)


def partitions():
    return EvidencePartitions(
        training_observation_ids=("train-1", "train-2"),
        validation_observation_ids=("validation-1", "validation-2"),
        holdout_observation_ids=("holdout-1", "holdout-2"),
    )


class EvidenceRegistryTests(unittest.TestCase):
    def setUp(self):
        DEFAULT_EVIDENCE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=str(DEFAULT_EVIDENCE_ROOT))
        self.path = Path(self.temporary.name) / "evidence.sqlite3"

    def tearDown(self):
        self.temporary.cleanup()

    def _registered(self):
        registry = EvidenceRegistry(self.path)
        registry.register(identity(), partitions(), now=NOW)
        return registry

    def _validation_pass(self, registry):
        fingerprint = partitions().development_fingerprint
        registry.advance_development(
            "CALL_DEBIT_VERTICAL", RegistryState.RESEARCH_PASS, fingerprint, now=NOW
        )
        registry.advance_development(
            "CALL_DEBIT_VERTICAL",
            RegistryState.VALIDATION_PASS,
            fingerprint,
            now=NOW + timedelta(seconds=1),
        )

    def test_partitions_reject_overlap_before_registry_write(self):
        with self.assertRaises(ValueError):
            EvidencePartitions(
                training_observation_ids=("shared",),
                validation_observation_ids=("shared",),
                holdout_observation_ids=("holdout",),
            )
        self.assertFalse(self.path.exists())

    def test_registry_is_private_and_frozen_identity_is_idempotent_only(self):
        with self._registered() as registry:
            self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
            self.assertIs(registry.get("CALL_DEBIT_VERTICAL").state, RegistryState.UNPROVEN)
            registry.register(identity(), partitions(), now=NOW)
            with self.assertRaises(EvidenceMutationError):
                registry.register(
                    identity(cost_model_version="cost-v2"), partitions(), now=NOW
                )
            with self.assertRaises(EvidenceMutationError):
                registry.assert_identity(identity(pop_model_artifact_id="sha256:other"))

    def test_same_development_evidence_cannot_be_retested(self):
        with self._registered() as registry:
            fingerprint = partitions().development_fingerprint
            registry.advance_development(
                "CALL_DEBIT_VERTICAL",
                RegistryState.RESEARCH_PASS,
                fingerprint,
                now=NOW,
            )
            with self.assertRaises(EvidenceRetestError):
                registry.advance_development(
                    "CALL_DEBIT_VERTICAL",
                    RegistryState.RESEARCH_PASS,
                    fingerprint,
                    now=NOW,
                )

    def test_holdout_is_consumed_once_across_restart(self):
        registry = self._registered()
        self._validation_pass(registry)
        result = registry.consume_holdout(
            "CALL_DEBIT_VERTICAL",
            partitions().holdout_fingerprint,
            passed=True,
            now=NOW + timedelta(seconds=2),
        )
        self.assertIs(result.state, RegistryState.HOLDOUT_PASS)
        self.assertTrue(result.holdout_consumed)
        registry.close()

        with EvidenceRegistry(self.path) as reopened:
            with self.assertRaises(HoldoutReuseError):
                reopened.consume_holdout(
                    "CALL_DEBIT_VERTICAL",
                    partitions().holdout_fingerprint,
                    passed=True,
                    now=NOW + timedelta(seconds=3),
                )

    def test_holdout_batch_preflight_is_atomic(self):
        second_identity = identity(strategy_family="LONG_CALL")
        second_partitions = EvidencePartitions(
            training_observation_ids=("second-train-1",),
            validation_observation_ids=("second-validation-1",),
            holdout_observation_ids=("second-holdout-1",),
        )
        with self._registered() as registry:
            registry.register(second_identity, second_partitions, now=NOW)
            self._validation_pass(registry)
            registry.advance_development(
                "LONG_CALL",
                RegistryState.RESEARCH_PASS,
                second_partitions.development_fingerprint,
                now=NOW,
            )
            registry.advance_development(
                "LONG_CALL",
                RegistryState.VALIDATION_PASS,
                second_partitions.development_fingerprint,
                now=NOW + timedelta(seconds=1),
            )
            with self.assertRaises(EvidenceMutationError):
                registry.consume_holdout_batch(
                    (
                        (
                            "CALL_DEBIT_VERTICAL",
                            partitions().holdout_fingerprint,
                            True,
                        ),
                        ("LONG_CALL", "sha256:wrong", False),
                    ),
                    now=NOW + timedelta(seconds=2),
                )
            self.assertFalse(
                registry.get("CALL_DEBIT_VERTICAL").holdout_consumed
            )
            self.assertFalse(registry.get("LONG_CALL").holdout_consumed)

            results = registry.consume_holdout_batch(
                (
                    (
                        "CALL_DEBIT_VERTICAL",
                        partitions().holdout_fingerprint,
                        True,
                    ),
                    ("LONG_CALL", second_partitions.holdout_fingerprint, False),
                ),
                now=NOW + timedelta(seconds=3),
            )
            self.assertIs(
                results["CALL_DEBIT_VERTICAL"].state,
                RegistryState.HOLDOUT_PASS,
            )
            self.assertIs(results["LONG_CALL"].state, RegistryState.REJECTED)

    def test_holdout_pass_can_enable_and_shadow_failure_revokes(self):
        with self._registered() as registry:
            self._validation_pass(registry)
            registry.consume_holdout(
                "CALL_DEBIT_VERTICAL",
                partitions().holdout_fingerprint,
                passed=True,
                now=NOW + timedelta(seconds=2),
            )
            enabled = registry.enable_manual_tickets(
                "CALL_DEBIT_VERTICAL", now=NOW + timedelta(seconds=3)
            )
            self.assertIs(enabled.state, RegistryState.MANUAL_TICKET_ENABLED)
            revoked = registry.record_shadow(
                "CALL_DEBIT_VERTICAL",
                "sha256:forward-monitor-1",
                passed=False,
                now=NOW + timedelta(seconds=4),
            )
            self.assertIs(revoked.state, RegistryState.REJECTED)
            self.assertEqual("SHADOW", revoked.failure_stage)

    def test_changed_holdout_partition_is_rejected_without_consuming(self):
        with self._registered() as registry:
            self._validation_pass(registry)
            with self.assertRaises(EvidenceMutationError):
                registry.consume_holdout(
                    "CALL_DEBIT_VERTICAL",
                    "sha256:different-holdout",
                    passed=True,
                    now=NOW + timedelta(seconds=2),
                )
            self.assertFalse(registry.get("CALL_DEBIT_VERTICAL").holdout_consumed)

    def test_failed_holdout_is_terminal_and_cannot_be_fixed_or_retried(self):
        registry = self._registered()
        self._validation_pass(registry)
        failed = registry.consume_holdout(
            "CALL_DEBIT_VERTICAL",
            partitions().holdout_fingerprint,
            passed=False,
            now=NOW + timedelta(seconds=2),
        )
        self.assertIs(failed.state, RegistryState.REJECTED)
        self.assertEqual(failed.failure_stage, "HOLDOUT")
        registry.close()
        with EvidenceRegistry(self.path) as reopened:
            with self.assertRaises(TerminalEvidenceError):
                reopened.consume_holdout(
                    "CALL_DEBIT_VERTICAL",
                    partitions().holdout_fingerprint,
                    passed=True,
                    now=NOW + timedelta(seconds=3),
                )
            with self.assertRaises(EvidenceMutationError):
                reopened.register(
                    identity(model_frozen_at=NOW, pop_model_version="fixed-after-fail"),
                    partitions(),
                    now=NOW + timedelta(seconds=3),
                )

    def test_failed_shadow_is_terminal_and_cannot_be_retried(self):
        with self._registered() as registry:
            self._validation_pass(registry)
            registry.consume_holdout(
                "CALL_DEBIT_VERTICAL",
                partitions().holdout_fingerprint,
                passed=True,
                now=NOW + timedelta(seconds=2),
            )
            failed = registry.record_shadow(
                "CALL_DEBIT_VERTICAL",
                "sha256:shadow-1",
                passed=False,
                now=NOW + timedelta(seconds=3),
            )
            self.assertIs(failed.state, RegistryState.REJECTED)
            self.assertEqual(failed.failure_stage, "SHADOW")
            with self.assertRaises(TerminalEvidenceError):
                registry.record_shadow(
                    "CALL_DEBIT_VERTICAL",
                    "sha256:shadow-fixed",
                    passed=True,
                    now=NOW + timedelta(seconds=4),
                )

    def test_failed_development_gate_is_terminal(self):
        with self._registered() as registry:
            failed = registry.reject_development(
                "CALL_DEBIT_VERTICAL",
                RegistryState.RESEARCH_PASS,
                partitions().development_fingerprint,
                now=NOW,
            )
            self.assertIs(failed.state, RegistryState.REJECTED)
            self.assertEqual("RESEARCH_PASS", failed.failure_stage)
            with self.assertRaises(TerminalEvidenceError):
                registry.reject_development(
                    "CALL_DEBIT_VERTICAL",
                    RegistryState.RESEARCH_PASS,
                    partitions().development_fingerprint,
                    now=NOW + timedelta(seconds=1),
                )

    def test_model_must_be_frozen_before_registration(self):
        with EvidenceRegistry(self.path) as registry:
            with self.assertRaises(ValueError):
                registry.register(
                    identity(model_frozen_at=NOW + timedelta(seconds=1)),
                    partitions(),
                    now=NOW,
                )

    def test_registry_path_cannot_escape_cultra(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                EvidenceRegistry(Path(temporary) / "outside.sqlite3")


if __name__ == "__main__":
    unittest.main()
