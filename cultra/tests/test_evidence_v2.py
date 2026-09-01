import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from cultra.catalog import CATALOG_VERSION
from cultra.evidence_registry import (
    DEFAULT_EVIDENCE_ROOT,
    EvidencePartitions,
    EvidenceRegistry,
    FrozenEvidenceIdentity,
    RegistryState,
)
from cultra.evidence_v2 import load_holdout_pass_family_evidence
from cultra.hypotheses import (
    FROZEN_HYPOTHESIS_REGISTRY,
    HYPOTHESIS_REGISTRY_HASH,
)
from cultra.modeling_v2 import MODEL_ARTIFACT_SCHEMA, MODEL_VERSION, _payload_hash
from cultra.protocol import historical_protocol_hash
from cultra.validation import validate_holdout_pass


def _write(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvidenceV2Tests(unittest.TestCase):
    def test_committed_holdout_artifacts_construct_ticket_evidence(self):
        DEFAULT_EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(DEFAULT_EVIDENCE_ROOT)) as temporary:
            root = Path(temporary)
            definition = next(
                item
                for item in FROZEN_HYPOTHESIS_REGISTRY
                if item.strategy_id == "CALL_DEBIT_VERTICAL"
                and item.holding_sessions == 40
            )
            period = {
                "start": "2024-01-02",
                "end": "2024-03-29",
                "selected_resolved_trades": 150,
                "ticker_date_clusters": 150,
                "net_expectancy_dollars": 12.0,
                "lower_net_expectancy_dollars_95": 2.0,
            }
            selected_model = {
                "hypothesis_id": definition.hypothesis_id,
                "strategy_id": definition.strategy_id,
                "state": "VALIDATION_PASS",
                "selection_model_validation": {
                    "training_period": period,
                    "validation_period": dict(
                        period, start="2024-06-03", end="2024-08-30"
                    ),
                },
            }
            hypotheses = []
            for item in FROZEN_HYPOTHESIS_REGISTRY:
                hypotheses.append(
                    selected_model
                    if item.hypothesis_id == definition.hypothesis_id
                    else {
                        "hypothesis_id": item.hypothesis_id,
                        "strategy_id": item.strategy_id,
                        "state": "REJECT_RESEARCH",
                    }
                )
            model_path = root / "models.json"
            models = {
                "schema": MODEL_ARTIFACT_SCHEMA,
                "model_version": MODEL_VERSION,
                "model_frozen_at": "2026-07-31T16:00:00+00:00",
                "hypothesis_registry_hash": HYPOTHESIS_REGISTRY_HASH,
                "protocol_hash": historical_protocol_hash(),
                "holdout_outcomes_read": False,
                "hypotheses": hypotheses,
            }
            _write(model_path, models)
            _write(
                model_path.with_suffix(".json.manifest.json"),
                {
                    "schema": "cultra.frozen-models-v2-manifest.v1",
                    "artifact": str(model_path.resolve()),
                    "artifact_bytes": model_path.stat().st_size,
                    "artifact_sha256": _sha(model_path),
                },
            )

            registry_path = root / "registry.sqlite3"
            frozen_at = datetime(2026, 7, 31, 16, 0, tzinfo=timezone.utc)
            evaluated_at = datetime(2026, 8, 1, 16, 0, tzinfo=timezone.utc)
            partitions = EvidencePartitions(("tr",), ("va",), ("ho",))
            with EvidenceRegistry(registry_path) as registry:
                registry.register(
                    FrozenEvidenceIdentity(
                        strategy_family=definition.hypothesis_id,
                        catalog_version=CATALOG_VERSION,
                        hypothesis_fingerprint="h" * 64,
                        cost_model_version="CULTRA_COSTS_V2",
                        exit_policy_version=definition.exit_policy,
                        pop_model_version=MODEL_VERSION,
                        pop_model_artifact_id=_payload_hash(selected_model),
                        model_frozen_at=frozen_at,
                    ),
                    partitions,
                    now=frozen_at,
                )
                registry.advance_development(
                    definition.hypothesis_id,
                    RegistryState.RESEARCH_PASS,
                    partitions.development_fingerprint,
                    now=frozen_at,
                )
                registry.advance_development(
                    definition.hypothesis_id,
                    RegistryState.VALIDATION_PASS,
                    partitions.development_fingerprint,
                    now=frozen_at,
                )
                registry.consume_holdout(
                    definition.hypothesis_id,
                    partitions.holdout_fingerprint,
                    passed=True,
                    now=evaluated_at,
                )

            calibration = {
                target: {
                    "brier": 0.18,
                    "development_base_rate_brier": 0.24,
                    "expected_calibration_error": 0.03,
                    "positive_events": 60,
                    "negative_events": 60,
                }
                for target in ("POP_NET", "P_TARGET", "P_STOP", "P_MAX_LOSS")
            }
            holdout_path = root / "holdout.json"
            result = {
                "state": "HOLDOUT_PASS",
                "reasons": [],
                "net_expectancy_dollars": 14.0,
                "selected_resolved_trades": 120,
                "ticker_date_clusters": 120,
                "unresolved_selected_worst_case_count": 0,
                "unresolved_worst_case_expectancy_dollars": 10.0,
                "bootstrap": {
                    "lower_net_pnl_dollars": 3.0,
                    "confidence": 0.95,
                },
                "holdout_period": {
                    "start": "2024-12-02",
                    "end": "2025-04-11",
                },
                "holm_adjusted_p_value": 0.01,
                "holm_family_size": len(FROZEN_HYPOTHESIS_REGISTRY),
                "ticker_profit_concentration": 0.15,
                "calendar_profit_concentration": 0.18,
                "calibration": calibration,
            }
            holdout = {
                "schema": "cultra.holdout-results-v2.v1",
                "prepared_at": evaluated_at.isoformat(),
                "model_artifact_sha256": _sha(model_path),
                "results": {definition.hypothesis_id: result},
            }
            _write(holdout_path, holdout)
            _write(
                holdout_path.with_suffix(".json.manifest.json"),
                {
                    "schema": "cultra.holdout-results-v2-manifest.v1",
                    "result": str(holdout_path.resolve()),
                    "result_bytes": holdout_path.stat().st_size,
                    "result_sha256": _sha(holdout_path),
                    "model_artifact_sha256": _sha(model_path),
                },
            )
            _write(
                holdout_path.with_suffix(".json.registry.json"),
                {
                    "schema": "cultra.holdout-registry-commit-v2.v1",
                    "holdout_result_sha256": _sha(holdout_path),
                    "evidence_registry": str(registry_path.resolve()),
                    "committed_states": {
                        definition.hypothesis_id: "HOLDOUT_PASS"
                    },
                },
            )
            evidence = load_holdout_pass_family_evidence(
                hypothesis_id=definition.hypothesis_id,
                model_artifact_path=model_path,
                holdout_result_path=holdout_path,
                evidence_registry_path=registry_path,
            )
            self.assertEqual(definition.hypothesis_id, evidence.strategy_family)
            self.assertEqual(120, evidence.holdout.resolved_trades)
            self.assertEqual((), validate_holdout_pass(evidence))
            self.assertGreater(evidence.evidence_expires_at, evaluated_at)


if __name__ == "__main__":
    unittest.main()
