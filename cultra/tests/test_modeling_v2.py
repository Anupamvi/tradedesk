import unittest
from datetime import date, timedelta

from cultra.modeling_v2 import (
    DevelopmentObservation,
    FrozenLinearModel,
    ModelingV2Error,
    TARGETS,
    _folds,
    _oof_predictions,
    fit_linear_model,
    frozen_calendar_split,
    coherent_exit_probabilities,
    score_current_candidate_v2,
)


class ModelingV2Tests(unittest.TestCase):
    @staticmethod
    def _joint_bin():
        return {
            "bin_index": 5,
            "sample_size": 50,
            "targets": {
                target: {
                    "wilson_95_lower": 0.35,
                    "wilson_95_upper": 0.65,
                }
                for target in TARGETS
            },
        }

    @staticmethod
    def _scenario_profile():
        values = {
            "TARGET": 2.0,
            "TIME_PROFIT": 0.2,
            "STOP": -0.35,
            "MAX_LOSS": -1.0,
            "TIME_LOSS": -0.15,
        }
        return {
            key: {"sample_size": 25, "mean_net_return_on_risk": value}
            for key, value in values.items()
        }

    def test_ridge_and_logistic_models_are_deterministic_and_feature_bound(self):
        rows = tuple(
            {"x": float(index), "z": float(index % 3)} for index in range(1, 41)
        )
        ridge = fit_linear_model(
            rows,
            tuple(0.5 * row["x"] - 0.25 * row["z"] for row in rows),
            ("x", "z"),
            kind="RIDGE",
            l2=0.1,
        )
        repeated = fit_linear_model(
            rows,
            tuple(0.5 * row["x"] - 0.25 * row["z"] for row in rows),
            ("x", "z"),
            kind="RIDGE",
            l2=0.1,
        )
        self.assertEqual(ridge, repeated)
        self.assertGreater(ridge.predict_one({"x": 35.0, "z": 0.0}), 0.0)

        logistic = fit_linear_model(
            rows,
            tuple(float(row["x"] >= 20.0) for row in rows),
            ("x", "z"),
            kind="LOGISTIC",
            l2=1.0,
        )
        self.assertLess(logistic.predict_one({"x": 5.0, "z": 0.0}), 0.5)
        self.assertGreater(logistic.predict_one({"x": 35.0, "z": 0.0}), 0.5)
        with self.assertRaisesRegex(ModelingV2Error, "feature is missing"):
            logistic.predict_one({"x": 1.0})

    def test_calendar_split_seals_final_twenty_percent_after_sixty_session_embargo(self):
        start = date(2024, 1, 1)
        sessions = tuple(start + timedelta(days=index) for index in range(450))
        split = frozen_calendar_split(sessions)
        self.assertEqual(59, len(split["research"]))
        self.assertEqual(59, len(split["tuning"]))
        self.assertEqual(59, len(split["validation"]))
        self.assertEqual(61, len(split["development_embargo_1"]))
        self.assertEqual(61, len(split["development_embargo_2"]))
        self.assertEqual(61, len(split["final_embargo"]))
        self.assertEqual(90, len(split["holdout"]))
        self.assertEqual(29, len(split["holdout_signal"]))
        self.assertEqual(61, len(split["holdout_path"]))
        self.assertEqual(118, len(split["oof"]))
        self.assertLess(split["research"][-1], split["tuning"][0])
        self.assertLess(split["tuning"][-1], split["validation"][0])
        self.assertLess(split["final_embargo"][-1], split["holdout"][0])
        folds = _folds(sessions)
        self.assertEqual(2, len(folds))
        self.assertEqual(split["tuning"], folds[0][1])
        self.assertEqual(split["validation"], folds[1][1])
        self.assertTrue(all(training and validation for training, validation in folds))

    def test_oof_predictions_use_only_real_cohort_signal_windows(self):
        start = date(2024, 1, 1)
        sessions = tuple(start + timedelta(days=index) for index in range(450))
        split = frozen_calendar_split(sessions)
        observations = []
        for index, session in enumerate(
            split["research"] + split["tuning"] + split["validation"]
        ):
            observations.append(
                DevelopmentObservation(
                    record_id="row-%03d" % index,
                    ticker="T%02d" % (index % 10),
                    signal_date=session,
                    features={"x": float(index % 17)},
                    net_pnl=10.0 if index % 3 else -4.0,
                    return_on_risk=0.10 if index % 3 else -0.04,
                    targets={target: int(index % 3 != 0) for target in TARGETS},
                )
            )
        predictions = _oof_predictions(
            tuple(observations),
            _folds(sessions),
            ("x",),
            target_name=None,
            kind="RIDGE",
            l2=1.0,
            policy={
                "maximum_newton_iterations": 100,
                "convergence_tolerance": 1e-8,
            },
        )
        self.assertEqual(118, len(predictions))
        predicted_dates = {
            item.signal_date for item in observations if item.record_id in predictions
        }
        self.assertEqual(set(split["tuning"] + split["validation"]), predicted_dates)

    def test_calendar_split_rejects_any_non_frozen_size(self):
        with self.assertRaisesRegex(ModelingV2Error, "450"):
            frozen_calendar_split(tuple(date(2024, 1, 1) + timedelta(days=i) for i in range(449)))

    def test_current_scoring_requires_positive_point_and_conservative_ev(self):
        ridge = FrozenLinearModel(
            "RIDGE", ("x",), (0.0,), (1.0,), (0.20, 0.0), 1.0, 100
        ).to_dict()
        logistic = FrozenLinearModel(
            "LOGISTIC", ("x",), (0.0,), (1.0,), (0.0, 0.0), 1.0, 100
        ).to_dict()
        probability = {
            "raw_model": logistic,
            "calibrator": {
                "kind": "ISOTONIC",
                "x_thresholds": [0.0, 1.0],
                "y_values": [0.5, 0.5],
                "sample_size": 100,
            },
            "calibration_period": {"start": "2025-01-01", "end": "2025-06-30"},
            "calibration_bins": [
                {
                    "bin_index": 5,
                    "sample_size": 50,
                    "wilson_95_lower": 0.35,
                    "wilson_95_upper": 0.65,
                }
            ],
        }
        models = {
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "state": "VALIDATION_PASS",
                    "return_model": ridge,
                    "return_model_uncertainty": {
                        "conservative_return_on_risk_offset": -0.05
                    },
                    "probability_models": {
                        target: dict(probability) for target in TARGETS
                    },
                    "joint_calibration_bins": [self._joint_bin()],
                    "scenario_return_profile": self._scenario_profile(),
                }
            ]
        }
        result = score_current_candidate_v2(
            models,
            hypothesis_id="H1",
            features={"x": 1.0},
            finite_maximum_loss=1000.0,
        )
        self.assertAlmostEqual(500.0, result["point_net_ev_dollars"])
        self.assertAlmostEqual(475.0, result["conservative_net_ev_dollars"])
        self.assertAlmostEqual(
            0.20, result["selection_model_point_return_on_maximum_loss"]
        )
        self.assertTrue(result["model_candidate_eligible"])
        self.assertFalse(result["manual_ticket_ready"])

    def test_current_conservative_ev_cannot_exceed_point_ev(self):
        ridge = FrozenLinearModel(
            "RIDGE", ("x",), (0.0,), (1.0,), (0.20, 0.0), 1.0, 100
        ).to_dict()
        logistic = FrozenLinearModel(
            "LOGISTIC", ("x",), (0.0,), (1.0,), (0.0, 0.0), 1.0, 100
        ).to_dict()
        probability = {
            "raw_model": logistic,
            "calibrator": {
                "kind": "ISOTONIC",
                "x_thresholds": [0.0, 1.0],
                "y_values": [0.5, 0.5],
                "sample_size": 100,
            },
            "calibration_period": {"start": "2025-01-01", "end": "2025-06-30"},
            "calibration_bins": [
                {
                    "bin_index": 5,
                    "sample_size": 50,
                    "wilson_95_lower": 0.35,
                    "wilson_95_upper": 0.65,
                }
            ],
        }
        models = {
            "hypotheses": [
                {
                    "hypothesis_id": "H1",
                    "state": "VALIDATION_PASS",
                    "return_model": ridge,
                    "return_model_uncertainty": {
                        "conservative_return_on_risk_offset": 0.05
                    },
                    "probability_models": {
                        target: dict(probability) for target in TARGETS
                    },
                    "joint_calibration_bins": [self._joint_bin()],
                    "scenario_return_profile": self._scenario_profile(),
                }
            ]
        }
        with self.assertRaisesRegex(ModelingV2Error, "conservative offset"):
            score_current_candidate_v2(
                models,
                hypothesis_id="H1",
                features={"x": 1.0},
                finite_maximum_loss=1000.0,
            )

    def test_probability_projection_produces_one_coherent_exit_distribution(self):
        result = coherent_exit_probabilities(
            {
                "POP_NET": 0.80,
                "P_TARGET": 0.90,
                "P_STOP": 0.70,
                "P_MAX_LOSS": 0.85,
            }
        )
        categories = result["categories"]
        metrics = result["metrics"]
        self.assertAlmostEqual(1.0, sum(categories.values()))
        self.assertLessEqual(metrics["P_TARGET"], metrics["POP_NET"])
        self.assertLessEqual(metrics["P_MAX_LOSS"], metrics["P_STOP"])
        self.assertLessEqual(metrics["POP_NET"] + metrics["P_STOP"], 1.0 + 1e-12)


if __name__ == "__main__":
    unittest.main()
