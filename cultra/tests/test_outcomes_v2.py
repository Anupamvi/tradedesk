import sqlite3
import unittest
from datetime import date, datetime, timezone

from cultra.domain import LegAction, LegQuote, OptionLeg, OptionType
from cultra.hypotheses import FROZEN_HYPOTHESIS_REGISTRY
from cultra.outcomes_v2 import _early_exercise_risk, _write_candidate, historical_costs
from cultra.structures import ContractQuote, SelectedStructure


class HistoricalOutcomeV2Tests(unittest.TestCase):
    def test_round_trip_costs_include_spread_slippage_without_portfolio_inputs(self):
        leg = OptionLeg(
            "AAPL261218C00200000",
            LegAction.BUY,
            OptionType.CALL,
            date(2026, 12, 18),
            200.0,
        )
        quote = LegQuote(
            leg.occ_symbol,
            4.80,
            5.20,
            datetime(2026, 8, 28, 20, tzinfo=timezone.utc),
        )
        hypothesis = next(
            item
            for item in FROZEN_HYPOTHESIS_REGISTRY
            if item.strategy_id == "LONG_CALL" and item.holding_sessions == 20
        )
        selection = SelectedStructure(
            hypothesis_id=hypothesis.hypothesis_id,
            strategy_id=hypothesis.strategy_id,
            holding_sessions=20,
            template_hash=hypothesis.structure_template_hash,
            legs=(leg,),
            entry_quotes=(quote,),
            entry_snapshot_ids=("snapshot",),
            target_call_deltas=(0.55,),
            signed_entry_debit=5.20,
        )
        costs = historical_costs(
            selection,
            {
                "version": "cost-v2",
                "commission_per_contract_per_side": 0.65,
                "fee_per_contract_per_side": 0.03,
                "slippage_fraction_of_quoted_spread_per_side": 0.10,
                "minimum_slippage_dollars_per_contract_per_side": 0.01,
            },
        )
        self.assertAlmostEqual(1.30, costs.commissions)
        self.assertAlmostEqual(0.06, costs.fees)
        self.assertAlmostEqual(40.0, costs.spread_reference)
        self.assertAlmostEqual(8.0, costs.slippage)

    def test_candidate_ledger_retains_geometry_for_missing_features(self):
        connection = sqlite3.connect(":memory:")
        connection.execute(
            "CREATE TABLE candidate_ledger(%s)"
            % ",".join("c%d" % index for index in range(20))
        )
        hypothesis = FROZEN_HYPOTHESIS_REGISTRY[0]
        _write_candidate(
            connection,
            record_id="r1",
            hypothesis=hypothesis,
            ticker="AAPL",
            signal_date=date(2026, 8, 27),
            signal_close_at=datetime(2026, 8, 27, 20, tzinfo=timezone.utc),
            entry_date=date(2026, 8, 28),
            planned_exit_date=date(2026, 9, 25),
            status="DATA_UNAVAILABLE",
            reason="SIGNAL_FEATURES_UNAVAILABLE",
            selection={"legs": [{"occ_symbol": "AAPL261218C00200000"}]},
            costs={"total": 3.0},
            risk={"risk_reference": 500.0, "maximum_loss": 500.0},
        )
        row = connection.execute("SELECT * FROM candidate_ledger").fetchone()
        connection.close()
        self.assertEqual(20, len(row))
        self.assertEqual("DATA_UNAVAILABLE", row[11])
        self.assertEqual("SIGNAL_FEATURES_UNAVAILABLE", row[12])
        self.assertIsNotNone(row[13])
        self.assertIsNone(row[14])
        self.assertIsNotNone(row[15])
        self.assertIsNotNone(row[16])

    def test_short_option_early_exercise_risk_fails_closed(self):
        leg = OptionLeg(
            "AAPL261218P00200000",
            LegAction.SELL,
            OptionType.PUT,
            date(2026, 12, 18),
            200.0,
        )
        quote = LegQuote(
            leg.occ_symbol,
            10.00,
            10.04,
            datetime(2026, 8, 28, 20, tzinfo=timezone.utc),
        )
        hypothesis = next(
            item
            for item in FROZEN_HYPOTHESIS_REGISTRY
            if item.strategy_id == "NAKED_PUT" and item.holding_sessions == 20
        )
        selection = SelectedStructure(
            hypothesis_id=hypothesis.hypothesis_id,
            strategy_id=hypothesis.strategy_id,
            holding_sessions=20,
            template_hash=hypothesis.structure_template_hash,
            legs=(leg,),
            entry_quotes=(quote,),
            entry_snapshot_ids=("snapshot",),
            target_call_deltas=(0.65,),
            signed_entry_debit=-10.00,
        )
        contract = ContractQuote(
            ticker="AAPL",
            trade_date=date(2026, 8, 28),
            expiration=date(2026, 12, 18),
            dte=112,
            strike=200.0,
            call_delta=0.10,
            call_bid=0.10,
            call_ask=0.12,
            put_bid=10.00,
            put_ask=10.04,
            call_open_interest=100,
            put_open_interest=100,
            observed_at=datetime(2026, 8, 28, 20, tzinfo=timezone.utc),
            snapshot_id="snapshot",
            stock_price=190.0,
        )
        self.assertEqual(
            "EARLY_EXERCISE_RISK_SHORT_PUT",
            _early_exercise_risk(selection, (contract,)),
        )


if __name__ == "__main__":
    unittest.main()
