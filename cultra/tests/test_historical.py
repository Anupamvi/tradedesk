import unittest
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

from cultra.domain import LegAction, OptionLeg, OptionType
from cultra.historical import (
    AmbiguityResolution,
    CorporateActionReview,
    ExitReason,
    FrozenExitPolicy,
    HistoricalCostInputs,
    HistoricalExitPath,
    HistoricalFeature,
    HistoricalLegSnapshot,
    HistoricalTradeRecord,
    HistoricalValidationError,
    ObservationOrigin,
    OptionGreeks,
    historical_validation_errors,
    validate_historical_trade,
)


UTC = timezone.utc
SIGNAL = datetime(2026, 1, 2, 15, 0, tzinfo=UTC)
ENTRY = datetime(2026, 1, 3, 15, 5, tzinfo=UTC)
EXIT = datetime(2026, 2, 2, 15, 5, tzinfo=UTC)
MARKET_SESSIONS = tuple(SIGNAL.date() + timedelta(days=index) for index in range(62))


def greeks(delta):
    return OptionGreeks(delta, 0.02, -0.03, 0.12, 0.01, 0.30)


def snapshot(leg, decision, delta, origin=ObservationOrigin.CONTEMPORANEOUS):
    return HistoricalLegSnapshot(
        occ_symbol=leg.occ_symbol,
        bid=1.90,
        ask=2.00,
        greeks=greeks(delta),
        observed_at=decision - timedelta(minutes=2),
        available_at=decision - timedelta(minutes=1),
        origin=origin,
        source_snapshot_id="snapshot:" + leg.occ_symbol,
    )


def valid_record():
    expiration = date(2026, 3, 20)
    long_leg = OptionLeg(
        "XYZ   260320C00100000", LegAction.BUY, OptionType.CALL, expiration, 100.0
    )
    short_leg = OptionLeg(
        "XYZ   260320C00105000", LegAction.SELL, OptionType.CALL, expiration, 105.0
    )
    costs = HistoricalCostInputs(1.30, 0.20, 1.00, 1.00, 0.00, 0.00, 0.50)
    return HistoricalTradeRecord(
        record_id="hist-001",
        symbol="XYZ",
        strategy_id="CALL_DEBIT_VERTICAL",
        signal_timestamp=SIGNAL,
        entry_decision_timestamp=ENTRY,
        exit_decision_timestamp=EXIT,
        expected_legs=(long_leg, short_leg),
        entry_snapshots=(
            snapshot(long_leg, ENTRY, 0.55),
            snapshot(short_leg, ENTRY, 0.35),
        ),
        exit_snapshots=(
            snapshot(long_leg, EXIT, 0.70),
            snapshot(short_leg, EXIT, 0.50),
        ),
        features=(
            HistoricalFeature(
                "forecast_iv_spread",
                0.08,
                SIGNAL - timedelta(hours=2),
                SIGNAL - timedelta(hours=1),
                ObservationOrigin.CONTEMPORANEOUS,
                "features:2026-01-02",
            ),
        ),
        corporate_action_review=CorporateActionReview(
            reviewed=True,
            reviewed_at=EXIT + timedelta(hours=1),
            source="historical corporate-actions ledger",
            relevant_action=False,
            adjustment_details=None,
            exact_contracts_verified=True,
        ),
        exit_policy=FrozenExitPolicy(
            policy_id="call-debit-exit",
            version="v1",
            time_exit_sessions=40,
            profit_target_return=0.50,
            stop_loss_return=0.35,
            ambiguity_resolution=AmbiguityResolution.WORST_CASE,
            frozen_at=SIGNAL - timedelta(days=30),
            is_frozen=True,
        ),
        costs=costs,
        exit_path=HistoricalExitPath(
            reason=ExitReason.TARGET,
            target_hit_session=EXIT.date(),
            stop_hit_session=None,
            ambiguity_resolution_applied=None,
        ),
        holding_sessions=30,
        gross_pnl=-20.0,
        net_pnl=-20.0 - costs.total,
        market_sessions=MARKET_SESSIONS,
    )


class HistoricalSchemaTests(unittest.TestCase):
    def test_complete_exact_leg_record_validates(self):
        record = valid_record()
        self.assertEqual(historical_validation_errors(record), ())
        self.assertIs(validate_historical_trade(record), record)

    def test_missing_entry_or_exit_leg_is_rejected(self):
        record = valid_record()
        for changed in (
            replace(record, entry_snapshots=record.entry_snapshots[:1]),
            replace(record, exit_snapshots=record.exit_snapshots[:1]),
        ):
            with self.assertRaises(HistoricalValidationError) as caught:
                validate_historical_trade(changed)
            self.assertIn("every exact expected OCC contract", str(caught.exception))

    def test_mismatched_or_substituted_contract_is_rejected(self):
        record = valid_record()
        mismatched_snapshot = replace(
            record.entry_snapshots[0], occ_symbol="XYZ   260320C00110000"
        )
        mismatched = replace(
            record,
            entry_snapshots=(mismatched_snapshot, record.entry_snapshots[1]),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(mismatched)
        self.assertIn("exact expected OCC contract", str(caught.exception))

        substituted_snapshot = replace(
            record.exit_snapshots[0], origin=ObservationOrigin.SUBSTITUTED
        )
        substituted = replace(
            record,
            exit_snapshots=(substituted_snapshot, record.exit_snapshots[1]),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(substituted)
        self.assertIn("reconstructed or substituted", str(caught.exception))

    def test_future_or_reconstructed_signal_feature_is_rejected(self):
        record = valid_record()
        leaked_feature = replace(
            record.features[0],
            observed_at=SIGNAL + timedelta(minutes=1),
            available_at=SIGNAL + timedelta(minutes=2),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(replace(record, features=(leaked_feature,)))
        self.assertIn("leaks data unavailable", str(caught.exception))

        reconstructed = replace(
            record.features[0], origin=ObservationOrigin.RECONSTRUCTED
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(replace(record, features=(reconstructed,)))
        self.assertIn("reconstructed or substituted", str(caught.exception))

    def test_same_session_or_nonconsecutive_entry_is_rejected(self):
        record = valid_record()
        same_day = replace(
            record,
            entry_decision_timestamp=SIGNAL + timedelta(minutes=5),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(same_day)
        self.assertIn("next market session", str(caught.exception))

        skipped = replace(
            record,
            entry_decision_timestamp=ENTRY + timedelta(days=1),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(skipped)
        self.assertIn("exactly T+1", str(caught.exception))

    def test_missing_greeks_and_noncontemporaneous_quotes_are_rejected(self):
        record = valid_record()
        missing_greeks = replace(record.entry_snapshots[0], greeks=None)
        stale_exit = replace(
            record.exit_snapshots[0],
            observed_at=EXIT - timedelta(hours=1),
            available_at=EXIT - timedelta(minutes=1),
        )
        changed = replace(
            record,
            entry_snapshots=(missing_greeks, record.entry_snapshots[1]),
            exit_snapshots=(stale_exit, record.exit_snapshots[1]),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(changed)
        self.assertIn("missing contemporaneous Greeks", str(caught.exception))
        self.assertIn("not contemporaneous", str(caught.exception))

    def test_unreviewed_or_unverified_corporate_actions_are_rejected(self):
        record = valid_record()
        changed = replace(
            record,
            corporate_action_review=replace(
                record.corporate_action_review,
                reviewed=False,
                relevant_action=True,
                adjustment_details=None,
                exact_contracts_verified=False,
            ),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(changed)
        message = str(caught.exception)
        self.assertIn("not reviewed", message)
        self.assertIn("did not verify exact OCC contracts", message)
        self.assertIn("lacks adjustment details", message)

    def test_unfrozen_or_post_signal_policy_and_bad_window_are_rejected(self):
        record = valid_record()
        changed = replace(
            record,
            exit_policy=replace(
                record.exit_policy,
                is_frozen=False,
                frozen_at=SIGNAL + timedelta(minutes=1),
                time_exit_sessions=61,
                ambiguity_resolution=None,
            ),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(changed)
        message = str(caught.exception)
        self.assertIn("not frozen", message)
        self.assertIn("between 20 and 60", message)
        self.assertIn("after the signal", message)
        self.assertIn("ambiguity ordering", message)

    def test_ambiguous_stop_target_order_must_match_frozen_policy(self):
        record = valid_record()
        ambiguous = replace(
            record,
            exit_path=HistoricalExitPath(
                reason=ExitReason.TARGET,
                target_hit_session=EXIT.date(),
                stop_hit_session=EXIT.date(),
                ambiguity_resolution_applied=None,
            ),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(ambiguous)
        self.assertIn("does not match frozen policy", str(caught.exception))

        resolved = replace(
            record,
            exit_path=HistoricalExitPath(
                reason=ExitReason.STOP,
                target_hit_session=EXIT.date(),
                stop_hit_session=EXIT.date(),
                ambiguity_resolution_applied=AmbiguityResolution.WORST_CASE,
            ),
        )
        self.assertEqual(historical_validation_errors(resolved), ())

    def test_incomplete_costs_and_unreconciled_net_pnl_are_rejected(self):
        record = valid_record()
        incomplete = replace(
            record,
            costs=replace(record.costs, exit_slippage=None),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(incomplete)
        self.assertIn("cost inputs are incomplete", str(caught.exception))

        unreconciled = replace(record, net_pnl=record.net_pnl + 1.0)
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(unreconciled)
        self.assertIn("does not reconcile", str(caught.exception))

    def test_time_exit_must_equal_frozen_session_count(self):
        record = valid_record()
        changed = replace(
            record,
            exit_path=replace(record.exit_path, reason=ExitReason.TIME),
            holding_sessions=39,
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(changed)
        self.assertIn("frozen session count", str(caught.exception))

    def test_occ_root_and_executable_side_gross_pnl_are_reconciled(self):
        record = valid_record()
        wrong_root = replace(
            record,
            expected_legs=tuple(
                replace(
                    leg,
                    occ_symbol=leg.occ_symbol.replace("XYZ", "AAPL"),
                )
                for leg in record.expected_legs
            ),
            entry_snapshots=tuple(
                replace(
                    item,
                    occ_symbol=item.occ_symbol.replace("XYZ", "AAPL"),
                )
                for item in record.entry_snapshots
            ),
            exit_snapshots=tuple(
                replace(
                    item,
                    occ_symbol=item.occ_symbol.replace("XYZ", "AAPL"),
                )
                for item in record.exit_snapshots
            ),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(wrong_root)
        self.assertIn("root does not match", str(caught.exception))

        fabricated = replace(record, gross_pnl=50.0, net_pnl=50.0 - record.costs.total)
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(fabricated)
        self.assertIn("executable entry/exit quote sides", str(caught.exception))

    def test_spread_dependent_slippage_and_multiplier_are_required(self):
        record = valid_record()
        no_slippage = replace(
            record,
            costs=replace(record.costs, entry_slippage=0.0),
            net_pnl=record.gross_pnl
            - replace(record.costs, entry_slippage=0.0).total,
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(no_slippage)
        self.assertIn("entry slippage must be positive", str(caught.exception))

        changed_multiplier = replace(
            record,
            exit_snapshots=(
                replace(record.exit_snapshots[0], contract_multiplier=50.0),
                record.exit_snapshots[1],
            ),
        )
        with self.assertRaises(HistoricalValidationError) as caught:
            validate_historical_trade(changed_multiplier)
        self.assertIn("cannot be reproduced", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
