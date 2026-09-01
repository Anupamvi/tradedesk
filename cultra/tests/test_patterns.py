import unittest
from datetime import date

from cultra.patterns import (
    _build_manual_research_actions,
    _economics,
    _universe_disposition,
)


class PatternPipelineTests(unittest.TestCase):
    @staticmethod
    def _confirmed_event_outside_window():
        return {
            "XYZ": {
                "date": "2026-10-15",
                "market_timing": "AFTER_MARKET_CLOSE",
                "source": "COMPANY_INVESTOR_RELATIONS",
                "source_url": "https://example.test/earnings",
            }
        }

    @staticmethod
    def _add_validated_evidence(candidate):
        candidate["evidence_state"] = "HOLDOUT_PASS"
        shared = {
            "status": "AVAILABLE_CALIBRATED_HOLDOUT_VALIDATED",
            "interval_95": [0.52, 0.68],
            "confidence_level": 0.95,
            "sample_size": 140,
            "calibration_period": {"start": "2025-01-02", "end": "2026-03-31"},
            "model_version": "model-v1",
        }
        candidate["POP_net"] = dict(shared, point=0.61)
        candidate["P_target"] = dict(shared, point=0.55)
        candidate["P_stop"] = dict(shared, point=0.30, interval_95=[0.22, 0.38])
        candidate["P_max_loss"] = dict(
            shared, point=0.08, interval_95=[0.04, 0.13]
        )
        candidate["net_edge"] = {
            "status": "AVAILABLE_VALIDATED",
            "point": 42.0,
            "conservative": 8.0,
        }
        return candidate

    @staticmethod
    def _action_candidate(family, candidate_id, maximum_profit):
        legs = [
            {
                "occ_symbol": candidate_id + "-long",
                "quote_timestamp": "2026-08-28T20:00:00Z",
                "strike": 100.0,
            }
        ]
        if "VERTICAL" in family:
            legs.append(
                {
                    "occ_symbol": candidate_id + "-short",
                    "quote_timestamp": "2026-08-28T20:00:00Z",
                    "strike": 110.0,
                }
            )
        return {
            "candidate_id": candidate_id,
            "ticker": "XYZ",
            "direction": "BULLISH",
            "strategy_family": family,
            "human_legs": family,
            "legs": legs,
            "expiration": "2026-10-16",
            "underlying_quote": {
                "last": 100.0,
                "bid": 99.9,
                "ask": 100.1,
                "timestamp": "2026-08-28T20:00:00Z",
            },
            "signal": {
                "momentum_20": 0.10,
                "momentum_60": 0.20,
                "weeks_to_next_earnings": 8.0,
            },
            "economics": {
                "proposed_limit_debit_per_share": 4.0,
                "entry_debit_before_costs": 400.0,
                "modeled_round_trip_slippage": 18.64,
                "commissions_and_fees": 1.36,
                "maximum_loss": 420.0,
                "maximum_profit": maximum_profit,
                "reward_to_risk": (
                    None if maximum_profit is None else maximum_profit / 420.0
                ),
                "breakevens_at_expiration": [104.2],
                "target_pnl": 210.0,
                "stop_pnl": -168.0,
                "time_exit_sessions": 20,
            },
            "development_model_diagnostics_not_pop_or_edge": {
                "probabilities": {"POP_NET": 0.99},
                "predicted_net_dollars": -9999.0,
            },
            "POP_net": {
                "status": "UNAVAILABLE_OUT_OF_DOMAIN_AND_MODEL_GATE",
                "point": None,
                "reason": ["POP_DOES_NOT_BEAT_BASE_RATE_BRIER"],
            },
            "net_edge": {
                "status": "UNAVAILABLE_AS_VALIDATED_EDGE",
                "point": None,
                "conservative": None,
                "reason": ["EV_MODEL_DOES_NOT_BEAT_BASE_MEAN_MSE"],
            },
            "disposition": "WATCHLIST_EV_MODEL_GATE_FAILED",
            "chain_coverage_provenance": "TEST_PARTIAL_CHAIN_INPUT",
        }

    def test_current_structure_economics_are_finite_and_costed(self):
        legs = (
            {
                "action": "BUY",
                "ratio": 1,
                "occ_symbol": "SPY261016C00750000",
                "expiration": "2026-10-16",
                "strike": 750.0,
                "option_type": "CALL",
                "bid": 20.0,
                "ask": 20.5,
            },
            {
                "action": "SELL",
                "ratio": 1,
                "occ_symbol": "SPY261016C00760000",
                "expiration": "2026-10-16",
                "strike": 760.0,
                "option_type": "CALL",
                "bid": 15.0,
                "ask": 15.4,
            },
        )
        costs = {
            "commission_per_contract_per_side": 0.65,
            "fee_per_contract_per_side": 0.03,
            "additional_slippage_fraction_of_spread": 0.10,
            "minimum_slippage_per_share_per_leg_per_side": 0.01,
            "contract_multiplier": 100,
        }
        exits = {
            "profit_target_fraction_of_maximum_loss": 0.50,
            "stop_loss_fraction_of_maximum_loss": 0.40,
            "time_exit_sessions": 20,
        }
        result = _economics("CALL_DEBIT_VERTICAL", legs, costs, exits)
        self.assertGreater(result["maximum_loss"], 550.0)
        self.assertGreater(result["maximum_profit"], 0.0)
        self.assertAlmostEqual(5.5, result["natural_debit_per_share"])

    def test_every_universe_symbol_gets_exactly_one_disposition(self):
        maps = {
            "universe": {name: {"ticker": name, "name": name} for name in "ABCDE"},
            "admitted": {name: {"ticker": name} for name in "CDE"},
            "legacy_local": {"A": {"ticker": "A", "reasons": ["legacy"]}},
            "orats_budget": {"B": {"ticker": "B", "reason": "budget"}},
            "screen_unavailable": {},
            "history": {name: {"ticker": name} for name in "CDE"},
            "orats": {"D": {"ticker": "D"}, "E": {"ticker": "E"}},
            "chains": {"E": {"ticker": "E"}},
        }
        rows = _universe_disposition(maps, {"E": "EVALUATED_RESEARCH_ONLY"})
        self.assertEqual(5, len(rows))
        self.assertEqual(5, len({item["ticker"] for item in rows}))
        dispositions = {item["ticker"]: item["disposition"] for item in rows}
        self.assertEqual("NOT_FULLY_EVALUATED_LEGACY_LOCAL_SCREEN", dispositions["A"])
        self.assertEqual("NOT_FULLY_EVALUATED_ORATS_BUDGET", dispositions["B"])
        self.assertEqual("DATA_UNAVAILABLE_ORATS_CORE", dispositions["C"])
        self.assertEqual("NOT_FULLY_EVALUATED_CHAIN_NOT_COLLECTED", dispositions["D"])
        self.assertEqual("EVALUATED_RESEARCH_ONLY", dispositions["E"])

    def test_candidate_is_retained_but_failed_profit_evidence_blocks_action(self):
        policy = {
            "structure_preference": [
                "CALL_DEBIT_VERTICAL",
                "PUT_DEBIT_VERTICAL",
                "LONG_CALL",
                "LONG_PUT",
            ],
            "require_20_and_60_session_direction_alignment": True,
            "earnings_confirmation_window_weeks": 4,
        }
        long_call = self._action_candidate("LONG_CALL", "long", None)
        vertical = self._action_candidate("CALL_DEBIT_VERTICAL", "vertical", 580.0)
        actions, exclusions = _build_manual_research_actions(
            [long_call, vertical],
            date(2026, 8, 28),
            policy,
            self._confirmed_event_outside_window(),
        )
        self.assertEqual((), exclusions)
        self.assertEqual(1, len(actions))
        action = actions[0]
        self.assertEqual("vertical", action["source_candidate_id"])
        self.assertFalse(action["failed_model_outputs_used"])
        self.assertEqual("2026-08-31", action["next_review_date"])
        self.assertEqual("PRESERVED_EXACT_TRADE_CANDIDATE", action["candidate_list_status"])
        self.assertEqual("🔵", action["candidate_symbol"])
        self.assertEqual("RED", action["action_color"])
        self.assertEqual("BLOCKED_PROFIT_EVIDENCE_GATE", action["action"])
        self.assertEqual("GREEN", action["payoff_geometry_color"])
        self.assertFalse(action["profit_evidence_gate_passed"])
        self.assertIn(
            "CALIBRATED_POP_UNAVAILABLE", action["profit_evidence_gate_reasons"]
        )
        self.assertEqual(
            "WATCHLIST_EV_MODEL_GATE_FAILED",
            action["admission_audit"]["upstream_candidate_disposition"],
        )
        self.assertEqual(6.30, action["exit"]["target_structure_value_per_share"])
        self.assertEqual(2.52, action["exit"]["stop_structure_value_per_share"])
        self.assertIsNone(action["POP_net"]["point"])
        self.assertIsNone(action["validated_net_edge"]["point"])

    def test_validated_pop_and_positive_conservative_edge_can_enter(self):
        policy = {
            "structure_preference": ["CALL_DEBIT_VERTICAL"],
            "require_20_and_60_session_direction_alignment": True,
            "earnings_confirmation_window_weeks": 4,
        }
        candidate = self._action_candidate(
            "CALL_DEBIT_VERTICAL", "vertical", 580.0
        )
        self._add_validated_evidence(candidate)
        actions, exclusions = _build_manual_research_actions(
            [candidate],
            date(2026, 8, 28),
            policy,
            self._confirmed_event_outside_window(),
        )
        self.assertEqual((), exclusions)
        self.assertEqual("ENTER_AT_OR_BELOW_LIMIT", actions[0]["action"])
        self.assertEqual("GREEN", actions[0]["action_color"])
        self.assertTrue(actions[0]["profit_evidence_gate_passed"])

    def test_nonnull_pop_and_edge_cannot_bypass_holdout(self):
        policy = {
            "structure_preference": ["CALL_DEBIT_VERTICAL"],
            "require_20_and_60_session_direction_alignment": True,
            "earnings_confirmation_window_weeks": 4,
        }
        candidate = self._action_candidate(
            "CALL_DEBIT_VERTICAL", "vertical", 580.0
        )
        self._add_validated_evidence(candidate)
        candidate["evidence_state"] = "RESEARCH_PASS"
        actions, exclusions = _build_manual_research_actions(
            [candidate],
            date(2026, 8, 28),
            policy,
            self._confirmed_event_outside_window(),
        )
        self.assertEqual((), exclusions)
        self.assertEqual("BLOCKED_PROFIT_EVIDENCE_GATE", actions[0]["action"])
        self.assertIn(
            "UNTOUCHED_HOLDOUT_NOT_PASSED",
            actions[0]["profit_evidence_gate_reasons"],
        )

    def test_missing_authoritative_event_record_is_red_avoid(self):
        policy = {
            "structure_preference": ["CALL_DEBIT_VERTICAL"],
            "require_20_and_60_session_direction_alignment": True,
            "earnings_confirmation_window_weeks": 4,
        }
        candidate = self._action_candidate(
            "CALL_DEBIT_VERTICAL", "vertical", 580.0
        )
        self._add_validated_evidence(candidate)
        actions, exclusions = _build_manual_research_actions(
            [candidate], date(2026, 8, 28), policy, {}
        )
        self.assertEqual((), exclusions)
        self.assertEqual("AVOID_EVENT_DATE_UNAVAILABLE", actions[0]["action"])
        self.assertEqual(
            "EVENT_DATE_UNVERIFIED_AVOID", actions[0]["event_gate"]["status"]
        )
        self.assertTrue(actions[0]["profit_evidence_gate_passed"])
        self.assertIsNone(actions[0]["entry"]["maximum_net_debit_per_share"])

    def test_stale_event_record_cannot_clear_next_earnings_gate(self):
        policy = {
            "structure_preference": ["CALL_DEBIT_VERTICAL"],
            "require_20_and_60_session_direction_alignment": True,
            "earnings_confirmation_window_weeks": 4,
        }
        candidate = self._action_candidate(
            "CALL_DEBIT_VERTICAL", "vertical", 580.0
        )
        self._add_validated_evidence(candidate)
        stale = self._confirmed_event_outside_window()
        stale["XYZ"]["date"] = "2026-08-01"
        actions, exclusions = _build_manual_research_actions(
            [candidate], date(2026, 8, 28), policy, stale
        )
        self.assertEqual((), exclusions)
        self.assertEqual("AVOID_STALE_EVENT_RECORD", actions[0]["action"])
        self.assertEqual(
            "STALE_EVENT_RECORD_AVOID", actions[0]["event_gate"]["status"]
        )

    def test_action_fails_closed_when_direction_is_not_aligned(self):
        policy = {
            "structure_preference": ["LONG_CALL"],
            "require_20_and_60_session_direction_alignment": True,
            "earnings_confirmation_window_weeks": 4,
        }
        candidate = self._action_candidate("LONG_CALL", "long", None)
        candidate["signal"]["momentum_60"] = -0.10
        actions, exclusions = _build_manual_research_actions(
            [candidate], date(2026, 8, 28), policy, {}
        )
        self.assertEqual((), actions)
        self.assertEqual("20_AND_60_SESSION_DIRECTION_NOT_ALIGNED", exclusions[0]["reason"])

    def test_confirmed_near_earnings_is_a_red_avoid_with_exact_date(self):
        policy = {
            "structure_preference": ["CALL_DEBIT_VERTICAL"],
            "require_20_and_60_session_direction_alignment": True,
            "earnings_confirmation_window_weeks": 4,
        }
        candidate = self._action_candidate("CALL_DEBIT_VERTICAL", "vertical", 580.0)
        candidate["signal"]["weeks_to_next_earnings"] = 0.0
        actions, exclusions = _build_manual_research_actions(
            [candidate],
            date(2026, 8, 28),
            policy,
            {
                "XYZ": {
                    "date": "2026-09-01",
                    "market_timing": "AFTER_MARKET_CLOSE",
                    "source": "COMPANY_INVESTOR_RELATIONS",
                    "source_url": "https://example.test/earnings",
                }
            },
        )
        self.assertEqual((), exclusions)
        action = actions[0]
        self.assertEqual("RED", action["action_color"])
        self.assertEqual("GREEN", action["payoff_geometry_color"])
        self.assertEqual("AVOID_UNTIL_POST_EARNINGS", action["action"])
        self.assertEqual("2026-09-01", action["event_gate"]["confirmed_date"])
        self.assertEqual("2026-09-02", action["next_review_date"])
        self.assertIsNone(action["entry"]["maximum_net_debit_per_share"])


if __name__ == "__main__":
    unittest.main()
