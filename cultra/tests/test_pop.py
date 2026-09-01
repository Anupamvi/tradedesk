import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from cultra.pop import (
    OOFPOPObservation,
    POPBucketIdentity,
    ProbabilityTarget,
    build_probability_bundle,
    build_oof_pop_model,
)


FROZEN_AT = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)


def bucket(target=ProbabilityTarget.POP_NET):
    return POPBucketIdentity(
        strategy_family="CALL_DEBIT_VERTICAL",
        regime_id="iv-premium-and-uptrend",
        target=target,
        bucket_version="bucket-v1",
    )


def observations(selected_bucket=None, count=240):
    selected_bucket = selected_bucket or bucket()
    first = date(2020, 1, 1)
    result = []
    for index in range(count):
        raw = ((index % 10) + 0.5) / 10.0
        outcome = int(raw >= 0.55)
        if index % 17 == 0:
            outcome = 1 - outcome
        result.append(
            OOFPOPObservation(
                observation_id="oof-%03d" % index,
                session_date=first + timedelta(days=index),
                bucket_id=selected_bucket.bucket_id,
                raw_probability=raw,
                outcome=outcome,
            )
        )
    return tuple(result)


def build(rows=None):
    selected_bucket = bucket()
    return build_oof_pop_model(
        rows or observations(selected_bucket),
        selected_bucket,
        model_version="pop-v1",
        holdout_start=date(2021, 1, 1),
        model_frozen_at=FROZEN_AT,
        min_training_sessions=80,
        validation_sessions=20,
        embargo_sessions=60,
    )


class OOFPOPTests(unittest.TestCase):
    def test_artifact_is_deterministic_verified_and_bucket_identified(self):
        first = build()
        second = build(tuple(reversed(observations())))
        self.assertEqual(first, second)
        self.assertTrue(first.verify())
        self.assertEqual(first.bucket.bucket_id, bucket().bucket_id)
        self.assertEqual(first.interval.method, "WILSON_SCORE")
        self.assertEqual(first.interval.confidence, 0.95)
        self.assertEqual(first.interval.sample_size, len(first.oof_predictions))
        self.assertTrue(all(item.session_date < first.holdout_start for item in first.oof_predictions))
        self.assertTrue(all(item.embargo_sessions == 60 for item in first.fold_metrics))
        self.assertIn(first.selected_method, ("logistic", "isotonic"))
        self.assertEqual(first.selected_method, first.frozen_calibrator.method)

    def test_oof_validation_observations_are_unique_across_folds(self):
        artifact = build()
        identifiers = [item.observation_id for item in artifact.oof_predictions]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertGreaterEqual(len(artifact.fold_metrics), 2)
        for fold in artifact.fold_metrics:
            self.assertLess(fold.training_end, fold.validation_start)

    def test_sparse_strategy_uses_complete_market_calendar_for_embargo(self):
        selected_bucket = bucket()
        first = date(2020, 1, 1)
        calendar = tuple(first + timedelta(days=index) for index in range(240))
        sparse = tuple(
            OOFPOPObservation(
                observation_id="sparse-%03d" % index,
                session_date=calendar[index],
                bucket_id=selected_bucket.bucket_id,
                raw_probability=0.70 if index % 4 == 0 else 0.30,
                outcome=1 if index % 4 == 0 else 0,
            )
            for index in range(0, 240, 2)
        )
        # Only 120 trade dates exist, so trade-date counting cannot form an
        # 80 + 60 + 20 fold. The complete 240-session calendar can.
        artifact = build_oof_pop_model(
            sparse,
            selected_bucket,
            model_version="pop-v1-calendar",
            holdout_start=date(2021, 1, 1),
            model_frozen_at=FROZEN_AT,
            min_training_sessions=80,
            validation_sessions=20,
            embargo_sessions=60,
            session_calendar=calendar,
        )
        self.assertTrue(artifact.verify())
        self.assertGreaterEqual(len(artifact.fold_metrics), 2)
        self.assertTrue(
            all(item.embargo_sessions == 60 for item in artifact.fold_metrics)
        )

    def test_holdout_or_cross_bucket_observation_is_rejected(self):
        rows = observations()
        leaked = replace(rows[-1], session_date=date(2021, 1, 1))
        with self.assertRaises(ValueError):
            build(rows[:-1] + (leaked,))

        wrong_bucket = replace(rows[-1], bucket_id=bucket(ProbabilityTarget.P_STOP).bucket_id)
        with self.assertRaises(ValueError):
            build(rows[:-1] + (wrong_bucket,))

    def test_artifact_mutation_breaks_content_identity(self):
        artifact = build()
        with self.assertRaises(ValueError):
            replace(artifact, oof_brier_score=artifact.oof_brier_score + 0.01)
        payload = artifact.to_dict()
        self.assertEqual(payload["artifact_id"], artifact.artifact_id)
        self.assertEqual(payload["bucket"]["bucket_id"], bucket().bucket_id)

    def test_duplicate_observation_cannot_enter_oof_workflow(self):
        rows = observations()
        with self.assertRaises(ValueError):
            build(rows + (rows[0],))

    def test_frozen_calibrator_reproduces_probabilities(self):
        artifact = build()
        first = artifact.frozen_calibrator.predict_one(0.42)
        second = artifact.frozen_calibrator.predict_one(0.42)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 1.0)

    def test_four_target_artifacts_build_ticket_probability_bundle(self):
        artifacts = {}
        for target in ProbabilityTarget:
            selected_bucket = bucket(target)
            artifacts[target] = build_oof_pop_model(
                observations(selected_bucket),
                selected_bucket,
                model_version="pop-v1",
                holdout_start=date(2021, 1, 1),
                model_frozen_at=FROZEN_AT,
                min_training_sessions=80,
                validation_sessions=20,
                embargo_sessions=60,
            )
        result = build_probability_bundle(
            artifacts,
            {
                ProbabilityTarget.POP_NET: 0.60,
                ProbabilityTarget.P_TARGET: 0.45,
                ProbabilityTarget.P_STOP: 0.40,
                ProbabilityTarget.P_MAX_LOSS: 0.10,
            },
        )
        self.assertEqual(result.pop_net.target_name, "POP_NET")
        self.assertEqual(result.p_target.target_name, "P_TARGET")
        self.assertEqual(result.p_stop.target_name, "P_STOP")
        self.assertEqual(result.p_max_loss.target_name, "P_MAX_LOSS")
        self.assertTrue(result.pop_net.artifact_id.startswith("sha256:"))
        self.assertEqual(result.pop_net.interval_method, "WILSON_SCORE_PREDICTED_COUNT")


if __name__ == "__main__":
    unittest.main()
