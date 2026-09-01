import unittest

from cultra.calibration import (
    IsotonicCalibrator,
    LogisticCalibrator,
    brier_score,
    choose_calibrator,
    empirical_probability_interval,
    expected_calibration_error,
    unconditional_brier_score,
    wilson_interval,
)


class CalibrationTests(unittest.TestCase):
    def test_brier_and_ece_have_known_values(self):
        predictions = (0.1, 0.2, 0.8, 0.9)
        outcomes = (0, 0, 1, 1)
        self.assertAlmostEqual(brier_score(predictions, outcomes), 0.025)
        self.assertAlmostEqual(expected_calibration_error(predictions, outcomes, bins=2), 0.15)
        self.assertAlmostEqual(unconditional_brier_score(outcomes, 0.5), 0.25)

    def test_wilson_and_empirical_intervals(self):
        lower, upper = wilson_interval(50, 100)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        self.assertAlmostEqual(lower, 1.0 - upper)
        self.assertEqual(empirical_probability_interval((0, 1) * 50), (lower, upper))
        with self.assertRaises(ValueError):
            wilson_interval(2, 1)

    def test_isotonic_pava_is_monotone_and_deterministic(self):
        scores = (0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9)
        outcomes = (0, 1, 0, 0, 1, 0, 1, 1, 0, 1)
        first = IsotonicCalibrator.fit(scores, outcomes)
        second = IsotonicCalibrator.fit(scores, outcomes)
        self.assertEqual(first, second)
        predictions = first.predict(tuple(index / 100 for index in range(101)))
        self.assertTrue(all(left <= right for left, right in zip(predictions, predictions[1:])))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in predictions))

    def test_logistic_calibration_is_deterministic_and_order_preserving(self):
        scores = tuple((index + 1) / 21 for index in range(20))
        outcomes = tuple(0 if index < 10 else 1 for index in range(20))
        first = LogisticCalibrator.fit(scores, outcomes)
        second = LogisticCalibrator.fit(scores, outcomes)
        self.assertEqual(first, second)
        predictions = first.predict(scores)
        self.assertLess(predictions[0], predictions[-1])
        self.assertTrue(all(0.0 <= value <= 1.0 for value in predictions))

    def test_selection_uses_validation_brier_and_is_reproducible(self):
        training_scores = tuple(index / 20 for index in range(1, 20))
        training_outcomes = tuple(1 if value > 0.55 else 0 for value in training_scores)
        validation_scores = (0.15, 0.35, 0.45, 0.65, 0.75, 0.9)
        validation_outcomes = (0, 0, 0, 1, 1, 1)
        selection = choose_calibrator(
            training_scores,
            training_outcomes,
            validation_scores,
            validation_outcomes,
            "pop-v1",
        )
        repeated = choose_calibrator(
            training_scores,
            training_outcomes,
            validation_scores,
            validation_outcomes,
            "pop-v1",
        )
        self.assertEqual(selection, repeated)
        self.assertIn(selection.name, ("logistic", "isotonic"))
        self.assertAlmostEqual(
            selection.validation_brier,
            min(selection.logistic_validation_brier, selection.isotonic_validation_brier),
        )

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            LogisticCalibrator.fit((0.5,), (2,))
        with self.assertRaises(ValueError):
            IsotonicCalibrator.fit((float("nan"),), (1,))
        with self.assertRaises(ValueError):
            expected_calibration_error((0.5,), (1,), bins=0)


if __name__ == "__main__":
    unittest.main()

