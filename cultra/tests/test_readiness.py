import sqlite3
import tempfile
import unittest
from pathlib import Path

from cultra.readiness import assess_production_readiness


class ProductionReadinessTests(unittest.TestCase):
    def test_etf_history_partial_chains_and_failed_models_are_explicit_blockers(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "history.sqlite3"
            connection = sqlite3.connect(str(database))
            connection.executescript(
                """
                CREATE TABLE sessions(trade_date TEXT);
                CREATE TABLE chains(trade_date TEXT, ticker TEXT);
                INSERT INTO sessions VALUES ('2026-01-02');
                INSERT INTO chains VALUES ('2026-01-02', 'SPY');
                """
            )
            connection.commit()
            connection.close()
            screen = {
                "quotes": [{"ticker": "AAPL"}],
                "admitted": [{"ticker": "AAPL"}],
                "budget_unresolved": [{"ticker": "MSFT"}],
            }
            history = {"rows": [{"ticker": "AAPL"}]}
            orats = {"rows": [{"ticker": "AAPL"}]}
            chains = {
                "chains": [{"ticker": "AAPL"}],
                "error_count": 0,
            }
            selection = {"selected_symbols": ["AAPL", "MSFT"]}
            config = {
                "families": {"LONG_CALL": {}},
                "historical_domain": {
                    "universe": ["SPY"],
                    "prior_holdout_status": "INVALIDATED_EXPOSED_AS_DEVELOPMENT_DATA",
                },
            }
            models = {
                "LONG_CALL": {
                    "metrics": {
                        "pop_gate_pass": False,
                        "pop_gate_reasons": ["POP_ECE_EXCEEDS_LIMIT"],
                        "ev_gate_pass": False,
                        "ev_gate_reasons": ["EV_MODEL_DOES_NOT_BEAT_BASE_MEAN_MSE"],
                        "probabilities": {
                            "POP_NET": {
                                "status": "DEVELOPMENT_CALIBRATED_NOT_HOLDOUT_VALIDATED",
                                "selected_method": "isotonic",
                                "oof_observations": 20,
                                "oof_brier": 0.30,
                                "base_rate_brier": 0.25,
                                "expected_calibration_error": 0.20,
                            }
                        },
                        "return_model": {
                            "oof_mse": 0.4,
                            "base_mean_mse": 0.3,
                            "selected_oof_95_lower_return_on_risk": -0.1,
                        },
                    }
                }
            }
            candidates = [
                {
                    "candidate_id": "aapl-call",
                    "ticker": "AAPL",
                    "strategy_family": "LONG_CALL",
                    "economics": {
                        "maximum_loss": 100.0,
                        "commissions_and_fees": 1.36,
                        "modeled_round_trip_slippage": 2.0,
                    },
                }
            ]
            result = assess_production_readiness(
                screen=screen,
                history=history,
                orats=orats,
                chains=chains,
                selection=selection,
                config=config,
                models=models,
                candidates=candidates,
                confirmed_events={},
                database=database,
            )
        blockers = {
            item["check_id"]
            for item in result["checks"]
            if item["status"] == "BLOCKED"
        }
        self.assertEqual("BLOCKED", result["status"])
        self.assertFalse(result["historically_validated_action_enabled"])
        self.assertIn("HISTORICAL_DOMAIN_MATCH", blockers)
        self.assertIn("CALIBRATED_POP_GATE", blockers)
        self.assertIn("CONSERVATIVE_EDGE_GATE", blockers)
        self.assertIn("NEW_UNTOUCHED_HOLDOUT", blockers)
        self.assertIn("EXACT_EVENT_COVERAGE", blockers)
        self.assertIn("NOT_A_90_DAY_WAIT_GATE", result["shadow_policy"])


if __name__ == "__main__":
    unittest.main()
