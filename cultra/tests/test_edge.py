import dataclasses
import unittest

from cultra.domain import Scenario
from cultra.edge import CostBreakdown, EdgeEstimate, PriceConvention, compute_edge


class EdgeTests(unittest.TestCase):
    def test_costs_and_point_conservative_ev_are_reproducible(self):
        costs = CostBreakdown(1.0, 0.5, 2.0, assignment_exercise=0.5, early_exit=1.0)
        self.assertEqual(costs.total, 5.0)
        edge = compute_edge(
            scenarios=(Scenario("profit", 0.6, 100.0), Scenario("loss", 0.4, -50.0)),
            conservative_scenarios=(
                Scenario("profit", 0.5, 100.0),
                Scenario("loss", 0.5, -50.0),
            ),
            maximum_loss=100.0,
            costs=costs,
            model_fair_price=1.25,
            executable_limit_price=1.10,
            price_convention=PriceConvention.DEBIT,
            maximum_profit=200.0,
            breakevens=(101.1,),
            target_pnl=50.0,
            stop_pnl=-35.0,
            expected_shortfall=60.0,
            adverse_gap_stress_loss=100.0,
        )
        self.assertAlmostEqual(edge.gross_expected_value, 40.0)
        self.assertAlmostEqual(edge.net_expected_value, 35.0)
        self.assertAlmostEqual(edge.conservative_net_expected_value, 20.0)
        self.assertAlmostEqual(edge.expected_return_on_max_loss, 0.35)
        self.assertAlmostEqual(edge.ranking_score, 0.20)
        self.assertTrue(edge.is_positive)
        self.assertEqual(2, len(edge.point_scenarios))
        self.assertEqual(2, len(edge.conservative_scenarios))
        self.assertAlmostEqual(edge.conservative_gross_expected_value, 25.0)
        with self.assertRaises(ValueError):
            dataclasses.replace(edge, net_expected_value=edge.net_expected_value + 1.0)

    def test_no_universal_pop_or_expected_dollar_threshold(self):
        edge = compute_edge(
            (Scenario("up", 0.01, 200.0), Scenario("down", 0.99, -1.0)),
            (Scenario("up", 0.01, 150.0), Scenario("down", 0.99, -1.0)),
            maximum_loss=10.0,
            costs=CostBreakdown(0.0, 0.0, 0.0),
            model_fair_price=0.02,
            executable_limit_price=0.01,
            price_convention=PriceConvention.DEBIT,
        )
        self.assertGreater(edge.net_expected_value, 0.0)
        self.assertGreater(edge.conservative_net_expected_value, 0.0)
        self.assertTrue(edge.is_positive)

    def test_conservative_distribution_is_mandatory_and_normalized(self):
        kwargs = dict(
            maximum_loss=100.0,
            costs=CostBreakdown(0.0, 0.0, 0.0),
            model_fair_price=1.0,
            executable_limit_price=1.0,
            price_convention=PriceConvention.DEBIT,
        )
        with self.assertRaises(ValueError):
            compute_edge((Scenario("x", 1.0, 1.0),), (), **kwargs)
        with self.assertRaises(ValueError):
            compute_edge(
                (Scenario("x", 0.8, 1.0),),
                (Scenario("x", 1.0, 1.0),),
                **kwargs,
            )

    def test_finite_positive_max_loss_is_structural(self):
        with self.assertRaises(ValueError):
            compute_edge(
                (Scenario("x", 1.0, 1.0),),
                (Scenario("x", 1.0, 1.0),),
                maximum_loss=float("inf"),
                costs=CostBreakdown(0.0, 0.0, 0.0),
                model_fair_price=1.0,
                executable_limit_price=1.0,
                price_convention=PriceConvention.DEBIT,
            )
        with self.assertRaises(ValueError):
            CostBreakdown(-1.0, 0.0, 0.0)

    def test_edge_dataclass_detects_inconsistent_return(self):
        with self.assertRaises(ValueError):
            EdgeEstimate(
                10.0,
                9.0,
                8.0,
                999.0,
                0.08,
                1.0,
                1.0,
                PriceConvention.DEBIT,
                20.0,
                100.0,
                (),
                5.0,
                -2.0,
                3.0,
                10.0,
                CostBreakdown(1.0, 0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
