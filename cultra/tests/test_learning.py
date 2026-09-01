import copy
import json
import unittest
from datetime import date, timedelta

from cultra.learning import (
    build_walk_forward_models,
    fit_logistic,
    overlap_exposure_clusters,
    public_model_evidence,
)
from cultra.patterns import CONFIG_PATH


def row(record_id, ticker, entry_date, exit_date, momentum, won):
    return {
        "record_id": record_id,
        "ticker": ticker,
        "strategy_family": "CALL_DEBIT_VERTICAL",
        "entry_date": entry_date,
        "exit_date": exit_date,
        "momentum_20": momentum,
        "realized_volatility_20": 0.20,
        "smv_vol": 0.18,
        "relative_spread": 0.05,
        "entry_debit": 300.0,
        "maximum_loss": 320.0,
        "maximum_profit": 680.0,
        "net_pnl": 160.0 if won else -128.0,
        "target_hit": bool(won),
        "stop_hit": not bool(won),
        "max_loss_hit": False,
    }


class LearningTests(unittest.TestCase):
    def test_multivariate_logistic_learns_entry_feature_instead_of_formula(self):
        rows = tuple(
            row(
                "r%03d" % index,
                "SPY",
                "2026-01-%02d" % (index % 28 + 1),
                "2026-02-%02d" % (index % 28 + 1),
                0.02 if index < 50 else 0.20,
                index >= 50,
            )
            for index in range(100)
        )
        model = fit_logistic(
            rows,
            "POP_NET",
            l2=2.0,
            maximum_iterations=60,
            tolerance=1e-8,
        )
        self.assertTrue(model.converged)
        self.assertLess(model.predict(rows[0]), 0.25)
        self.assertGreater(model.predict(rows[-1]), 0.75)

    def test_overlapping_daily_positions_collapse_to_exposure_episodes(self):
        sessions = {
            (date(2026, 1, 1) + timedelta(days=index)).isoformat(): index
            for index in range(40)
        }
        rows = (
            row("a", "QQQ", "2026-01-01", "2026-01-10", 0.1, True),
            row("b", "QQQ", "2026-01-05", "2026-01-14", 0.1, True),
            row("c", "QQQ", "2026-01-20", "2026-01-25", 0.1, False),
            row("d", "SPY", "2026-01-05", "2026-01-14", 0.1, True),
        )
        clusters = overlap_exposure_clusters(rows, sessions)
        self.assertEqual(clusters["a"], clusters["b"])
        self.assertNotEqual(clusters["b"], clusters["c"])
        self.assertNotEqual(clusters["a"], clusters["d"])
        self.assertEqual(3, len(set(clusters.values())))

    def test_walk_forward_model_uses_embargo_and_can_beat_baseline(self):
        start = date(2025, 1, 1)
        sessions = tuple((start + timedelta(days=index)).isoformat() for index in range(320))
        values = []
        for index in range(20, 300, 2):
            strong = (index // 2) % 2 == 0
            values.append(
                row(
                    "r%03d" % index,
                    "T%02d" % ((index // 20) % 10),
                    sessions[index],
                    sessions[index + 5],
                    0.20 if strong else 0.02,
                    strong,
                )
            )
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["families"] = {"CALL_DEBIT_VERTICAL": config["families"]["CALL_DEBIT_VERTICAL"]}
        config["learning_policy"].update(
            {
                "minimum_training_sessions": 60,
                "validation_sessions": 30,
                "step_sessions": 30,
                "embargo_sessions": 20,
                "bootstrap_iterations": 200,
                "minimum_oof_observations": 20,
                "minimum_independent_exposure_clusters": 2,
            }
        )
        result = build_walk_forward_models(values, sessions, config)[
            "CALL_DEBIT_VERTICAL"
        ]
        metrics = result["metrics"]
        self.assertTrue(result["folds"])
        self.assertTrue(all(item["embargo_sessions"] == 20 for item in result["folds"]))
        self.assertLess(
            metrics["probabilities"]["POP_NET"]["oof_brier"],
            metrics["probabilities"]["POP_NET"]["base_rate_brier"],
        )
        calibration = result["calibration"]["targets"]["POP_NET"]
        self.assertEqual(
            "DEVELOPMENT_CALIBRATED_NOT_HOLDOUT_VALIDATED",
            calibration["status"],
        )
        self.assertIn(calibration["selected_method"], {"logistic", "isotonic"})
        self.assertTrue(calibration["folds"])
        self.assertTrue(
            all(
                item["validation_start_session_index"]
                - item["training_end_session_index"]
                > item["embargo_sessions"]
                for item in calibration["folds"]
            )
        )
        public = public_model_evidence({"CALL_DEBIT_VERTICAL": result})[
            "CALL_DEBIT_VERTICAL"
        ]
        self.assertNotIn("_runtime_calibrators", public["calibration"])
        self.assertNotIn(
            "selected_oof_predictions",
            public["calibration"]["targets"]["POP_NET"],
        )


if __name__ == "__main__":
    unittest.main()
