import hashlib
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cultra.current_v2 import CurrentV2Error, build_current_manual_ticket_v2
from cultra.domain import EvidenceState
from cultra.modeling_v2 import FrozenLinearModel, MODEL_VERSION, TARGETS
from tests.test_tickets import NOW, candidate, evidence


def _linear(kind, intercept):
    return FrozenLinearModel(
        kind=kind,
        feature_names=("x",),
        means=(0.0,),
        scales=(1.0,),
        coefficients=(intercept, 0.0),
        l2=1.0,
        sample_size=200,
    ).to_dict()


def _models(return_intercept=0.20):
    base = candidate(EvidenceState.HOLDOUT_PASS)
    calibrated = {
        "POP_NET": 0.60,
        "P_TARGET": 0.45,
        "P_STOP": 0.40,
        "P_MAX_LOSS": 0.10,
    }
    probability_models = {}
    for target in TARGETS:
        probability_models[target] = {
            "raw_model": _linear("LOGISTIC", 0.0),
            "calibrator": {
                "kind": "ISOTONIC",
                "x_thresholds": [0.0, 1.0],
                "y_values": [calibrated[target], calibrated[target]],
                "sample_size": 200,
            },
            "calibration_period": {
                "start": "2025-01-01",
                "end": "2026-06-30",
            },
        }
    joint_targets = {
        target: {
            "wilson_95_lower": max(0.0, value - 0.10),
            "wilson_95_upper": min(1.0, value + 0.10),
        }
        for target, value in calibrated.items()
    }
    scenario_returns = {
        "TARGET": 1.20,
        "TIME_PROFIT": 0.25,
        "STOP": -0.35,
        "MAX_LOSS": -1.00,
        "TIME_LOSS": -0.20,
    }
    hypothesis = {
        "hypothesis_id": base.hypothesis_id,
        "state": "VALIDATION_PASS",
        "return_model": _linear("RIDGE", return_intercept),
        "return_model_uncertainty": {
            "conservative_return_on_risk_offset": -0.05
        },
        "probability_models": probability_models,
        "joint_calibration_bins": [
            {"bin_index": 6, "sample_size": 150, "targets": joint_targets}
        ],
        "scenario_return_profile": {
            key: {"sample_size": 30, "mean_net_return_on_risk": value}
            for key, value in scenario_returns.items()
        },
    }
    return {
        "model_version": MODEL_VERSION,
        "hypotheses": [hypothesis],
    }


def _artifact_id(models):
    payload = models["hypotheses"][0]
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


class CurrentV2Tests(unittest.TestCase):
    def _build(self, models, supplied_evidence=None):
        base = candidate(EvidenceState.HOLDOUT_PASS)
        current_evidence = supplied_evidence or replace(
            evidence(EvidenceState.HOLDOUT_PASS),
            shadow=None,
            cost_model_version="CULTRA_COSTS_V2",
            model_version=MODEL_VERSION,
            pop_model_artifact_id=_artifact_id(models),
        )
        with patch(
            "cultra.current_v2.load_frozen_models_v2", return_value=models
        ):
            return build_current_manual_ticket_v2(
                model_artifact_path=Path("unused-in-unit-test.json"),
                hypothesis_id=base.hypothesis_id,
                features={"x": 1.0},
                evidence=current_evidence,
                candidate_id="current-v2-001",
                symbol=base.symbol,
                thesis="Frozen V2 signal with exact current economics",
                signal="DIRECTIONAL_COMPOSITE_V1",
                legs=base.legs,
                leg_quotes=base.leg_quotes,
                underlying_quote=base.underlying_quote,
                orats_snapshot_id=base.orats_snapshot_id,
                provider_trade_date=base.provider_trade_date,
                analytical_fields=base.analytical_fields,
                snapshot_manifest=base.snapshot_manifest,
                field_profile=base.field_profile,
                event_evidence=base.event_evidence,
                invalidation="Frozen directional feature changes sign",
                now=NOW,
            )

    def test_verified_v2_model_and_holdout_evidence_build_exact_ticket(self):
        models = _models()
        ticket = self._build(models)
        self.assertEqual("MANUAL_TICKET_ENABLED", ticket.evidence_state.value)
        self.assertEqual("USER DETERMINED", ticket.quantity)
        self.assertEqual("CULTRA_COSTS_V2", ticket.edge.costs.model_version)
        self.assertAlmostEqual(116.72, ticket.edge.maximum_loss)
        self.assertGreater(ticket.edge.net_expected_value, 0.0)
        self.assertGreater(ticket.edge.conservative_net_expected_value, 0.0)
        self.assertEqual((("x", 1.0),), ticket.model_calculation.features)
        self.assertEqual(
            _artifact_id(models), ticket.model_calculation.model_artifact_id
        )
        self.assertEqual(64, len(ticket.model_calculation.calculation_id))
        self.assertAlmostEqual(0.60, ticket.probabilities.pop_net.point)
        self.assertAlmostEqual(0.45, ticket.probabilities.p_target.point)
        self.assertAlmostEqual(0.40, ticket.probabilities.p_stop.point)
        self.assertAlmostEqual(0.10, ticket.probabilities.p_max_loss.point)

    def test_model_artifact_must_equal_holdout_registry_identity(self):
        models = _models()
        mismatched = replace(
            evidence(EvidenceState.HOLDOUT_PASS),
            shadow=None,
            cost_model_version="CULTRA_COSTS_V2",
            model_version=MODEL_VERSION,
            pop_model_artifact_id="f" * 64,
        )
        with self.assertRaisesRegex(CurrentV2Error, "does not match holdout"):
            self._build(models, mismatched)

    def test_negative_current_selection_model_cannot_become_ticket(self):
        with self.assertRaisesRegex(CurrentV2Error, "positive conservative"):
            self._build(_models(return_intercept=-0.10))


if __name__ == "__main__":
    unittest.main()
