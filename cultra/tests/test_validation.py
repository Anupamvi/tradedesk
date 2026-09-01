import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from cultra.catalog import CATALOG_VERSION, FROZEN_STRATEGY_CATALOG
from cultra.domain import EvidenceState, FamilyEvidence, HistoricalObservation, PeriodEvidence
from cultra.hypotheses import (
    FROZEN_HYPOTHESIS_COUNT,
    HYPOTHESIS_REGISTRY_HASH,
    HYPOTHESIS_REGISTRY_VERSION,
)
from cultra.statistics import (
    clustered_bootstrap_mean_ci,
    contribution_concentration,
    holm_adjust,
    two_way_clustered_bootstrap_mean_ci,
)
from cultra.validation import (
    PromotionPolicy,
    assert_transition,
    chronological_split,
    evaluate_promotion,
    promote_evidence,
    walk_forward_development_splits,
    walk_forward_splits,
)


def period(
    name,
    start_day,
    end_day,
    expectancy=5.0,
    lower=1.0,
    trades=150,
    clusters=50,
    confidence=0.95,
):
    return PeriodEvidence(
        name,
        expectancy,
        lower,
        trades,
        clusters,
        start_day,
        end_day,
        confidence,
    )


def passing_evidence(state=EvidenceState.UNPROVEN):
    return FamilyEvidence(
        strategy_family="CALL_DEBIT_VERTICAL",
        state=state,
        training=period("training", date(2020, 1, 1), date(2021, 1, 1)),
        validation=period("validation", date(2021, 6, 1), date(2022, 1, 1)),
        holdout=period("holdout", date(2022, 6, 1), date(2023, 1, 1)),
        shadow=period(
            "shadow",
            date(2023, 2, 1),
            date(2023, 7, 1),
            trades=40,
            confidence=0.90,
        ),
        holm_adjusted_p_value=0.01,
        holm_family_size=FROZEN_HYPOTHESIS_COUNT,
        holm_catalog_version=HYPOTHESIS_REGISTRY_VERSION,
        max_contribution_fraction=0.15,
        contribution_dimensions=("calendar_period", "ticker"),
        pop_ece=0.03,
        pop_brier_score=0.18,
        base_rate_brier_score=0.23,
        cost_model_version="cost-v1",
        model_version="pop-v1",
        pop_model_artifact_id="b" * 64,
        frozen_catalog_version=CATALOG_VERSION,
        frozen_exit_policy="exit-v1",
        holdout_consumed_once=True,
        shadow_calendar_days=100,
        hypothesis_registry_hash=HYPOTHESIS_REGISTRY_HASH,
        timing_policy_version="SIGNAL_CLOSE_T_ENTRY_T_PLUS_1_V1",
        universe_policy_version="POINT_IN_TIME_ROTATING_COHORT_V1",
        model_frozen_at=datetime(2022, 5, 1, tzinfo=timezone.utc),
        holdout_evaluated_at=datetime(2023, 1, 2, tzinfo=timezone.utc),
        evidence_expires_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
        holdout_resolved_candidates=150,
        holdout_unresolved_candidates=0,
        unresolved_worst_case_expectancy=1.0,
        probability_event_counts=(
            ("POP_NET", 90, 60),
            ("P_TARGET", 65, 85),
            ("P_STOP", 60, 90),
            ("P_MAX_LOSS", 20, 130),
        ),
        two_way_clustered=True,
        point_in_time_membership=True,
        next_session_entry=True,
        holdout_registry_receipt="c" * 64,
    )


class ChronologicalValidationTests(unittest.TestCase):
    def test_final_twenty_percent_and_two_sixty_session_embargoes(self):
        first = date(2020, 1, 1)
        observations = tuple(
            HistoricalObservation("id-%03d" % index, first + timedelta(days=index), "C-%03d" % index, 1.0)
            for index in range(400)
        )
        split = chronological_split(observations)
        self.assertEqual(len(split.training), 120)
        self.assertEqual(len(split.validation), 80)
        self.assertEqual(len(split.holdout), 80)
        self.assertEqual(len(split.embargoed), 120)
        self.assertEqual(split.holdout[0].session_date, first + timedelta(days=320))
        self.assertEqual(split.embargo_sessions, 60)

    def test_same_session_observations_never_cross_partitions(self):
        first = date(2020, 1, 1)
        observations = []
        for index in range(400):
            session = first + timedelta(days=index)
            observations.append(HistoricalObservation("a-%d" % index, session, "a", 1.0))
            observations.append(HistoricalObservation("b-%d" % index, session, "b", -0.5))
        split = chronological_split(observations)
        membership = {}
        for name in ("training", "validation", "holdout", "embargoed"):
            for observation in getattr(split, name):
                previous = membership.setdefault(observation.session_date, name)
                self.assertEqual(previous, name)

    def test_insufficient_history_fails_instead_of_shrinking_embargo(self):
        first = date(2020, 1, 1)
        observations = tuple(
            HistoricalObservation(str(index), first + timedelta(days=index), "x", 1.0)
            for index in range(100)
        )
        with self.assertRaises(ValueError):
            chronological_split(observations)

    def test_walk_forward_is_expanding_deterministic_and_seals_final_holdout(self):
        first = date(2018, 1, 1)
        observations = tuple(
            HistoricalObservation(
                "wf-%03d" % index,
                first + timedelta(days=index),
                "cluster-%03d" % index,
                1.0,
            )
            for index in range(500)
        )
        first_plan = walk_forward_splits(observations)
        second_plan = walk_forward_splits(tuple(reversed(observations)))
        self.assertEqual(first_plan, second_plan)
        self.assertEqual(len(first_plan.final_holdout), 100)
        self.assertEqual(len(first_plan.final_holdout_embargoed), 60)
        self.assertEqual(len(first_plan.folds), 8)
        prior_training = 0
        validation_ids = set()
        for fold in first_plan.folds:
            self.assertEqual(len({item.session_date for item in fold.embargoed}), 60)
            self.assertGreater(len(fold.training), prior_training)
            prior_training = len(fold.training)
            current = {item.observation_id for item in fold.validation}
            self.assertFalse(current & validation_ids)
            validation_ids.update(current)
            self.assertLess(
                max(item.session_date for item in fold.training),
                min(item.session_date for item in fold.embargoed),
            )
            self.assertLess(
                max(item.session_date for item in fold.embargoed),
                min(item.session_date for item in fold.validation),
            )

    def test_walk_forward_rejects_overlapping_validation_windows(self):
        first = date(2020, 1, 1)
        observations = tuple(
            HistoricalObservation(str(index), first + timedelta(days=index), "x", 1.0)
            for index in range(240)
        )
        with self.assertRaises(ValueError):
            walk_forward_development_splits(
                observations,
                min_training_sessions=80,
                validation_sessions=20,
                embargo_sessions=60,
                step_sessions=10,
            )


class StatisticsAndPromotionTests(unittest.TestCase):
    def test_cluster_bootstrap_is_deterministic_and_clustered(self):
        values = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        clusters = ("A", "A", "B", "B", "C", "C")
        first = clustered_bootstrap_mean_ci(values, clusters, iterations=1000, seed=17)
        second = clustered_bootstrap_mean_ci(values, clusters, iterations=1000, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first.cluster_count, 3)
        self.assertGreater(first.lower, 0.0)

    def test_two_way_bootstrap_is_deterministic_across_ticker_and_date(self):
        values = (1.0, 2.0, 1.5, 2.5, 1.25, 2.25, 1.75, 2.75)
        tickers = ("A", "A", "B", "B", "C", "C", "D", "D")
        dates = ("D1", "D2", "D1", "D2", "D1", "D2", "D1", "D2")
        first = two_way_clustered_bootstrap_mean_ci(
            values, tickers, dates, iterations=1000, seed=23
        )
        second = two_way_clustered_bootstrap_mean_ci(
            values, tickers, dates, iterations=1000, seed=23
        )
        self.assertEqual(first, second)
        self.assertEqual(4, first.first_cluster_count)
        self.assertEqual(2, first.second_cluster_count)
        self.assertEqual(8, first.joint_cluster_count)
        self.assertGreater(first.lower, 0.0)

    def test_holm_and_contribution_concentration(self):
        adjusted = holm_adjust((0.01, 0.04, 0.03))
        self.assertEqual(adjusted, (0.03, 0.06, 0.06))
        concentration = contribution_concentration(
            (60.0, 40.0, -100.0), ("A", "B", "C")
        )
        self.assertEqual(concentration.max_cluster, "A")
        self.assertAlmostEqual(concentration.max_fraction, 0.6)

    def test_state_machine_prevents_skips_and_terminal_reentry(self):
        assert_transition(EvidenceState.UNPROVEN, EvidenceState.RESEARCH_PASS)
        with self.assertRaises(ValueError):
            assert_transition(EvidenceState.UNPROVEN, EvidenceState.HOLDOUT_PASS)
        with self.assertRaises(ValueError):
            assert_transition(
                EvidenceState.MANUAL_TICKET_ENABLED, EvidenceState.MANUAL_TICKET_ENABLED
            )

    def test_all_states_promote_only_one_step(self):
        evidence = passing_evidence(EvidenceState.UNPROVEN)
        for expected in (
            EvidenceState.RESEARCH_PASS,
            EvidenceState.VALIDATION_PASS,
            EvidenceState.HOLDOUT_PASS,
            EvidenceState.MANUAL_TICKET_ENABLED,
        ):
            decision = evaluate_promotion(evidence)
            self.assertTrue(decision.passed, decision.reasons)
            self.assertIs(decision.target_state, expected)
            evidence = promote_evidence(evidence)
            self.assertIs(evidence.state, expected)

    def test_shadow_is_optional_monitoring_not_a_manual_action_delay(self):
        evidence = passing_evidence(EvidenceState.HOLDOUT_PASS)
        direct = evaluate_promotion(evidence)
        self.assertTrue(direct.passed, direct.reasons)
        self.assertIs(direct.target_state, EvidenceState.MANUAL_TICKET_ENABLED)
        shadow = evaluate_promotion(evidence, EvidenceState.SHADOW_PASS)
        self.assertTrue(shadow.passed, shadow.reasons)

    def test_holdout_and_shadow_gates_are_cumulative_and_fail_closed(self):
        weak = replace(
            passing_evidence(EvidenceState.VALIDATION_PASS),
            holdout=period(
                "holdout",
                date(2022, 6, 1),
                date(2023, 1, 1),
                expectancy=5.0,
                lower=-0.01,
                trades=99,
                clusters=39,
            ),
            holm_adjusted_p_value=0.051,
            max_contribution_fraction=0.21,
        )
        decision = evaluate_promotion(weak)
        self.assertFalse(decision.passed)
        joined = " ".join(decision.reasons)
        self.assertIn("lower confidence", joined)
        self.assertIn("100", joined)
        self.assertIn("40", joined)
        self.assertIn("Holm", joined)
        self.assertIn("concentration", joined)

        weak_shadow = replace(
            passing_evidence(EvidenceState.HOLDOUT_PASS),
            shadow_calendar_days=89,
            pop_ece=0.051,
            pop_brier_score=0.25,
        )
        decision = evaluate_promotion(weak_shadow, EvidenceState.SHADOW_PASS)
        self.assertFalse(decision.passed)
        self.assertIn("calendar days", " ".join(decision.reasons))
        self.assertIn("calibration", " ".join(decision.reasons))
        self.assertIn("Brier", " ".join(decision.reasons))


if __name__ == "__main__":
    unittest.main()
