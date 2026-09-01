import json
import unittest

from cultra.holdout_v2 import evaluate_frozen_hypothesis_holdout
from cultra.modeling_v2 import FrozenLinearModel, TARGETS
from cultra.protocol import load_historical_campaign_protocol


def frozen_model(kind, intercept):
    return FrozenLinearModel(
        kind=kind,
        feature_names=("x",),
        means=(0.0,),
        scales=(1.0,),
        coefficients=(intercept, 0.0),
        l2=1.0,
        sample_size=200,
    ).to_dict()


def model_artifact():
    probability = {
        "raw_model": frozen_model("LOGISTIC", 0.0),
        "calibrator": {
            "kind": "ISOTONIC",
            "x_thresholds": [0.0, 1.0],
            "y_values": [0.5, 0.5],
            "sample_size": 200,
        },
        "development_base_rate": 0.5,
    }
    return {
        "hypothesis_id": "TEST_HYPOTHESIS",
        "strategy_id": "LONG_CALL",
        "return_model": frozen_model("RIDGE", 0.10),
        "probability_models": {target: dict(probability) for target in TARGETS},
    }


def resolved_row(index):
    positive = index % 3 != 0
    outcome = {
        "net_pnl": 100.0 if positive else -20.0,
        "risk_reference": 100.0,
        "target_hit": positive,
        "stop_hit": not positive,
        "max_loss_hit": index % 5 == 0,
    }
    return {
        "record_id": "row-%03d" % index,
        "ticker": "T%02d" % (index % 10),
        "signal_date": "2026-%02d-%02d" % (1 + (index // 84), 1 + (index % 28)),
        "status": "RESOLVED",
        "selection_json": "{}",
        "features_json": json.dumps({"x": float(index)}),
        "risk_json": json.dumps({"risk_reference": 100.0}),
        "outcome_json": json.dumps(outcome),
    }


class HoldoutV2Tests(unittest.TestCase):
    def test_every_positive_ev_candidate_is_retained_without_top_n(self):
        rows = tuple(resolved_row(index) for index in range(120))
        result = evaluate_frozen_hypothesis_holdout(
            rows,
            model_artifact(),
            load_historical_campaign_protocol(),
            seed=7,
        )
        self.assertEqual(120, result["geometrically_executable_rows"])
        self.assertEqual(120, result["selected_resolved_trades"])
        self.assertEqual(120, len(result["selected_observations"]))

    def test_missing_features_cannot_improve_resolution_or_expectancy(self):
        rows = [resolved_row(index) for index in range(120)]
        rows[0] = dict(rows[0], status="DATA_UNAVAILABLE", features_json=None, outcome_json=None)
        result = evaluate_frozen_hypothesis_holdout(
            tuple(rows),
            model_artifact(),
            load_historical_campaign_protocol(),
            seed=9,
        )
        self.assertEqual(119 / 120, result["resolution_rate"])
        self.assertEqual(1, result["unknown_feature_rows_charged_as_selected_worst_case"])
        self.assertEqual(1, result["unresolved_selected_worst_case_count"])


if __name__ == "__main__":
    unittest.main()
