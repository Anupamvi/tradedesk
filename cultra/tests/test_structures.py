import unittest
from datetime import date, datetime, timedelta, timezone

from cultra.catalog import FROZEN_STRATEGY_CATALOG
from cultra.domain import LegAction, LegQuote
from cultra.edge import CostBreakdown
from cultra.hypotheses import FROZEN_HYPOTHESIS_REGISTRY
from cultra.structures import (
    ContractQuote,
    RiskClass,
    STRUCTURE_TEMPLATE_REGISTRY_HASH,
    StructureError,
    get_structure_template,
    resolve_historical_structure_path,
    select_frozen_structure,
    structure_risk_envelope,
)


TRADE_DATE = date(2026, 1, 1)
OBSERVED_AT = datetime(2026, 1, 1, 21, tzinfo=timezone.utc)


def synthetic_chain():
    rows = []
    for dte in (49, 91):
        expiration = TRADE_DATE + timedelta(days=dte)
        for strike in range(60, 141, 5):
            call_delta = (150.0 - strike) / 100.0
            call_mid = call_delta * 10.0 + dte / 100.0
            put_mid = (1.0 - call_delta) * 10.0 + dte / 100.0
            rows.append(
                ContractQuote(
                    ticker="XYZ",
                    trade_date=TRADE_DATE,
                    expiration=expiration,
                    dte=dte,
                    strike=float(strike),
                    call_delta=call_delta,
                    call_bid=call_mid - 0.05,
                    call_ask=call_mid + 0.05,
                    put_bid=put_mid - 0.05,
                    put_ask=put_mid + 0.05,
                    call_open_interest=1000,
                    put_open_interest=1000,
                    observed_at=OBSERVED_AT,
                    snapshot_id="snapshot-%d" % dte,
                )
            )
    return tuple(rows)


def hypothesis(strategy_id):
    return next(
        item
        for item in FROZEN_HYPOTHESIS_REGISTRY
        if item.strategy_id == strategy_id and item.holding_sessions == 20
    )


class FrozenStructureEngineTests(unittest.TestCase):
    def setUp(self):
        self.chain = synthetic_chain()
        self.costs = CostBreakdown(1.30, 0.06, 2.0, model_version="cost-v2")

    def select(self, strategy_id, values=None):
        frozen = hypothesis(strategy_id)
        return select_frozen_structure(
            hypothesis_id=frozen.hypothesis_id,
            strategy_id=strategy_id,
            holding_sessions=20,
            contracts=self.chain if values is None else values,
        )

    def test_every_catalog_family_has_a_deterministic_exact_structure(self):
        first = {}
        for definition in FROZEN_STRATEGY_CATALOG:
            selection = self.select(definition.strategy_id)
            repeated = self.select(definition.strategy_id, tuple(reversed(self.chain)))
            self.assertEqual(selection, repeated)
            self.assertEqual(definition.leg_count, len(selection.legs))
            self.assertEqual(
                get_structure_template(definition.strategy_id).template_hash,
                selection.template_hash,
            )
            self.assertEqual(
                len(selection.legs), len({item.occ_symbol for item in selection.legs})
            )
            first[definition.strategy_id] = selection
        self.assertEqual(30, len(first))
        self.assertEqual(64, len(STRUCTURE_TEMPLATE_REGISTRY_HASH))

    def test_defined_and_research_only_risk_classes_fail_closed(self):
        for definition in FROZEN_STRATEGY_CATALOG:
            selection = self.select(definition.strategy_id)
            envelope = structure_risk_envelope(selection, self.costs)
            if definition.defined_risk_by_construction:
                self.assertIsNotNone(envelope, definition.strategy_id)
                self.assertGreater(envelope.maximum_loss, 0.0)
            else:
                self.assertIsNone(envelope, definition.strategy_id)
                self.assertIs(
                    RiskClass.UNDEFINED_RESEARCH_ONLY,
                    get_structure_template(definition.strategy_id).risk_class,
                )

    def test_term_structures_use_later_bought_expiry_and_finite_debit_bound(self):
        for strategy_id in ("CALL_DIAGONAL", "PUT_DIAGONAL", "CALL_CALENDAR", "PUT_CALENDAR"):
            selection = self.select(strategy_id)
            bought = [item for item in selection.legs if item.action is LegAction.BUY]
            sold = [item for item in selection.legs if item.action is LegAction.SELL]
            self.assertEqual(1, len(bought))
            self.assertEqual(1, len(sold))
            self.assertGreater(bought[0].expiration, sold[0].expiration)
            self.assertGreater(selection.signed_entry_debit, 0.0)
            self.assertGreater(structure_risk_envelope(selection, self.costs).maximum_loss, 0.0)

    def test_exact_path_cannot_substitute_or_drop_a_selected_contract(self):
        selection = self.select("IRON_CONDOR")
        incomplete = tuple(selection.entry_quotes[:-1])
        path = tuple(
            (TRADE_DATE + timedelta(days=index + 1), incomplete)
            for index in range(20)
        )
        with self.assertRaisesRegex(StructureError, "every frozen exact leg"):
            resolve_historical_structure_path(selection, path, self.costs)

    def test_first_adverse_mark_stops_even_if_later_marks_recover(self):
        selection = self.select("LONG_CALL")
        leg = selection.legs[0]
        adverse = (LegQuote(leg.occ_symbol, 0.0, 0.01, OBSERVED_AT + timedelta(days=1)),)
        favorable = (LegQuote(leg.occ_symbol, 20.0, 20.1, OBSERVED_AT + timedelta(days=2)),)
        path = [(TRADE_DATE + timedelta(days=1), adverse)]
        path.extend(
            (TRADE_DATE + timedelta(days=index), favorable)
            for index in range(2, 21)
        )
        outcome = resolve_historical_structure_path(selection, path, self.costs)
        self.assertEqual("STOP", outcome.exit_reason)
        self.assertEqual(1, outcome.holding_sessions)
        self.assertTrue(outcome.stop_hit)

    def test_selected_expiration_remains_in_the_chain_window_through_exit(self):
        frozen = hypothesis("LONG_CALL")
        selection = select_frozen_structure(
            hypothesis_id=frozen.hypothesis_id,
            strategy_id=frozen.strategy_id,
            holding_sessions=20,
            contracts=self.chain,
            required_path_end=TRADE_DATE + timedelta(days=60),
        )
        self.assertGreaterEqual(
            (selection.legs[0].expiration - (TRADE_DATE + timedelta(days=60))).days,
            20,
        )
        with self.assertRaisesRegex(StructureError, "frozen structure geometry"):
            select_frozen_structure(
                hypothesis_id=frozen.hypothesis_id,
                strategy_id=frozen.strategy_id,
                holding_sessions=20,
                contracts=self.chain,
                required_path_end=TRADE_DATE + timedelta(days=72),
            )


if __name__ == "__main__":
    unittest.main()
