import json
import unittest
from dataclasses import fields, replace
from datetime import date, datetime, timedelta, timezone

from cultra.cache import SnapshotManifest
from cultra.catalog import CATALOG_VERSION, FROZEN_STRATEGY_CATALOG
from cultra.domain import (
    EntryExitPolicy,
    EvidenceState,
    FamilyEvidence,
    LegAction,
    LegQuote,
    OptionLeg,
    OptionType,
    PeriodEvidence,
    ProbabilityBundle,
    ProbabilityEstimate,
    Scenario,
    ScenarioOutcome,
    UnderlyingQuote,
)
from cultra.edge import CostBreakdown, PriceConvention, compute_edge
from cultra.hypotheses import (
    FROZEN_HYPOTHESIS_COUNT,
    FROZEN_HYPOTHESIS_REGISTRY,
    HYPOTHESIS_REGISTRY_HASH,
    HYPOTHESIS_REGISTRY_VERSION,
)
from cultra.tickets import (
    CurrentModelCalculation,
    EventEvidence,
    QUANTITY_USER_DETERMINED,
    ManualTicket,
    TicketFieldProfile,
    TicketCandidate,
    TicketRejection,
    build_manual_ticket,
    revalidate_manual_ticket,
)


NOW = datetime(2026, 8, 30, 20, 0, tzinfo=timezone.utc)


def hypothesis_id(strategy):
    return next(
        item.hypothesis_id
        for item in FROZEN_HYPOTHESIS_REGISTRY
        if item.strategy_id == strategy and item.holding_sessions == 40
    )


def period(name, start, end, trades=150, clusters=50, confidence=0.95):
    return PeriodEvidence(
        name, 10.0, 2.0, trades, clusters, start, end, confidence
    )


def evidence(state=EvidenceState.SHADOW_PASS, strategy="CALL_DEBIT_VERTICAL"):
    return FamilyEvidence(
        strategy_family=hypothesis_id(strategy),
        state=state,
        training=period("training", date(2020, 1, 1), date(2021, 1, 1)),
        validation=period("validation", date(2021, 6, 1), date(2022, 1, 1)),
        holdout=period("holdout", date(2022, 6, 1), date(2023, 1, 1)),
        shadow=period(
            "shadow",
            date(2023, 2, 1),
            date(2023, 6, 1),
            40,
            30,
            0.90,
        ),
        holm_adjusted_p_value=0.01,
        holm_family_size=FROZEN_HYPOTHESIS_COUNT,
        holm_catalog_version=HYPOTHESIS_REGISTRY_VERSION,
        max_contribution_fraction=0.15,
        contribution_dimensions=("calendar_period", "ticker"),
        pop_ece=0.03,
        pop_brier_score=0.18,
        base_rate_brier_score=0.24,
        cost_model_version="cost-v1",
        model_version="pop-v1",
        pop_model_artifact_id="b" * 64,
        frozen_catalog_version=CATALOG_VERSION,
        frozen_exit_policy="STOP_FIRST_TARGET_STOP_TIME_H40_V1",
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


def probabilities(sample_size=150):
    def item(point, target):
        return ProbabilityEstimate(
            point,
            max(0.0, point - 0.1),
            min(1.0, point + 0.1),
            sample_size,
            "pop-v1",
            date(2025, 1, 1),
            date(2026, 6, 30),
            0.95,
            "WILSON",
            "CALL_DEBIT_VERTICAL|trend_iv",
            "b" * 64,
            target,
        )

    return ProbabilityBundle(
        item(0.60, "POP_NET"),
        item(0.45, "P_TARGET"),
        item(0.40, "P_STOP"),
        item(0.10, "P_MAX_LOSS"),
    )


def model_calculation(strategy="CALL_DEBIT_VERTICAL"):
    category_probabilities = (
        ("MAX_LOSS", 0.10),
        ("STOP", 0.30),
        ("TARGET", 0.45),
        ("TIME_LOSS", 0.00),
        ("TIME_PROFIT", 0.15),
    )
    point_returns = (
        ("MAX_LOSS", -1.00),
        ("STOP", -0.30),
        ("TARGET", 1.00),
        ("TIME_LOSS", -0.20),
        ("TIME_PROFIT", 0.30),
    )
    conservative_returns = (
        ("MAX_LOSS", -1.00),
        ("STOP", -0.40),
        ("TARGET", 0.90),
        ("TIME_LOSS", -0.30),
        ("TIME_PROFIT", 0.20),
    )
    return CurrentModelCalculation(
        calculation_version="test-current-model-v1",
        hypothesis_id=hypothesis_id(strategy),
        model_version="pop-v1",
        model_artifact_id="b" * 64,
        features=(("x", 1.0),),
        selection_point_return_on_max_loss=0.20,
        selection_conservative_return_on_max_loss=0.15,
        scenario_point_return_on_max_loss=0.305,
        scenario_conservative_return_on_max_loss=0.215,
        probability_projection_l1_distance=0.0,
        joint_exit_probabilities=category_probabilities,
        scenario_net_returns_on_risk=point_returns,
        conservative_scenario_net_returns_on_risk=conservative_returns,
    )


def positive_edge(limit=1.10):
    return compute_edge(
        (
            Scenario("target", 0.45, 150.0, ScenarioOutcome.TARGET),
            Scenario("time_profit", 0.15, 50.0, ScenarioOutcome.TIME_PROFIT),
            Scenario("stop", 0.30, -40.0, ScenarioOutcome.STOP),
            Scenario("max_loss", 0.10, -110.0, ScenarioOutcome.MAX_LOSS),
        ),
        (
            Scenario("target", 0.35, 140.0, ScenarioOutcome.TARGET),
            Scenario("time_profit", 0.15, 40.0, ScenarioOutcome.TIME_PROFIT),
            Scenario("stop", 0.35, -50.0, ScenarioOutcome.STOP),
            Scenario("max_loss", 0.15, -113.4, ScenarioOutcome.MAX_LOSS),
        ),
        maximum_loss=113.4,
        costs=CostBreakdown(
            1.30,
            0.10,
            2.0,
            model_version="cost-v1",
            spread_reference=20.0,
        ),
        model_fair_price=1.30,
        executable_limit_price=limit,
        price_convention=PriceConvention.DEBIT,
        maximum_profit=386.6,
        breakevens=(101.134,),
        target_pnl=55.0,
        stop_pnl=-38.5,
        expected_shortfall=113.4,
        adverse_gap_stress_loss=113.4,
    )


def snapshot_manifest(legs):
    symbols = tuple(sorted(leg.occ_symbol for leg in legs))
    return SnapshotManifest(
        snapshot_id="a" * 64,
        cache_key="c" * 64,
        request_fingerprint="d" * 64,
        expectation_id="e" * 64,
        endpoint="/datav2/strikes/options",
        method="GET",
        publication_cycle="EOD_DELAYED",
        expected_trade_date="2026-08-28",
        provider_trade_dates=("2026-08-28",),
        updated_at_min="2026-08-28T20:00:00Z",
        updated_at_max="2026-08-28T20:00:00Z",
        field_profile="EXACT_OPTION_V1",
        schema_version="ORATS_NORMALIZED_V1",
        representation="json",
        requested_entities=symbols,
        returned_entities=symbols,
        missing_entities=(),
        raw_sha256="f" * 64,
        raw_bytes=1000,
        row_count=len(symbols),
        created_at=1.0,
    )


def field_profile():
    return TicketFieldProfile(
        profile_id="EXACT_OPTION_V1",
        schema_version="ORATS_NORMALIZED_V1",
        fields=(
            "confidence",
            "forecastVol",
            "iv",
            "optionSymbol",
            "ticker",
            "tradeDate",
            "updatedAt",
        ),
        concept_mapping=(
            ("confidence", "confidence"),
            ("forecast_volatility", "forecastVol"),
            ("implied_volatility", "iv"),
        ),
    )


def candidate(
    state=EvidenceState.SHADOW_PASS,
    quote_time=NOW - timedelta(minutes=1),
    candidate_edge=None,
):
    expiration = date(2026, 10, 16)
    long_leg = OptionLeg(
        "XYZ   261016C00100000", LegAction.BUY, OptionType.CALL, expiration, 100.0
    )
    short_leg = OptionLeg(
        "XYZ   261016C00105000", LegAction.SELL, OptionType.CALL, expiration, 105.0
    )
    event_sessions = tuple(date(2026, 8, 31) + timedelta(days=index) for index in range(40))
    return TicketCandidate(
        candidate_id="cand-001",
        symbol="XYZ",
        thesis="Frozen directional signal",
        signal="trend-and-IV-v1",
        strategy_id="CALL_DEBIT_VERTICAL",
        hypothesis_id=hypothesis_id("CALL_DEBIT_VERTICAL"),
        evidence=evidence(state),
        legs=(long_leg, short_leg),
        leg_quotes=(
            LegQuote(long_leg.occ_symbol, 1.90, 2.00, quote_time),
            LegQuote(short_leg.occ_symbol, 0.90, 1.00, quote_time),
        ),
        underlying_quote=UnderlyingQuote("XYZ", 99.95, 100.05, quote_time),
        orats_snapshot_id="a" * 64,
        provider_trade_date=date(2026, 8, 28),
        analytical_fields=("confidence", "forecastVol", "iv"),
        probabilities=probabilities(),
        edge=candidate_edge or positive_edge(),
        policy=EntryExitPolicy(
            "enter only at stated limit",
            "exit at +50%",
            "exit at -35%",
            "exit after 40 sessions",
            "signal no longer valid",
            "close before exercise; manual review if assigned",
            date(2026, 8, 31),
            "STOP_FIRST_TARGET_STOP_TIME_H40_V1",
            40,
        ),
        event_evidence=EventEvidence(
            asset_type="STOCK",
            source="SCHWAB_AND_COMPANY_IR_FROZEN_V1",
            source_timestamp=NOW - timedelta(days=1),
            holding_window_start=event_sessions[0],
            holding_window_end=event_sessions[-1],
            market_sessions=event_sessions,
            earnings_date=date(2026, 10, 20),
            dividend_dates=(),
            status="CLEAR",
            artifact_id="d" * 64,
        ),
        model_calculation=model_calculation(),
        snapshot_manifest=snapshot_manifest((long_leg, short_leg)),
        field_profile=field_profile(),
    )


class TicketTests(unittest.TestCase):
    def test_complete_shadow_pass_candidate_builds_manual_ticket(self):
        ticket = build_manual_ticket(candidate(), NOW)
        self.assertIsInstance(ticket, ManualTicket)
        self.assertIs(ticket.evidence_state, EvidenceState.MANUAL_TICKET_ENABLED)
        self.assertEqual(ticket.quantity, QUANTITY_USER_DETERMINED)
        self.assertAlmostEqual(ticket.ranking_score, ticket.edge.ranking_score)
        payload = ticket.to_dict()
        self.assertEqual(payload["quantity"], "USER DETERMINED")
        self.assertEqual(
            payload["hypothesis_id"], hypothesis_id("CALL_DEBIT_VERTICAL")
        )
        self.assertEqual(payload["probabilities"]["pop_net"]["point"], 0.60)
        json.dumps(payload)  # Every field is machine-artifact safe.
        self.assertIs(revalidate_manual_ticket(ticket, NOW), ticket)

        tampered = replace(ticket, ranking_score=ticket.ranking_score + 1.0)
        with self.assertRaises(TicketRejection):
            revalidate_manual_ticket(tampered, NOW)

    def test_complete_holdout_pass_does_not_wait_for_shadow(self):
        ticket = build_manual_ticket(candidate(EvidenceState.HOLDOUT_PASS), NOW)
        self.assertIsInstance(ticket, ManualTicket)
        self.assertIs(ticket.evidence.state, EvidenceState.HOLDOUT_PASS)
        self.assertIs(ticket.evidence_state, EvidenceState.MANUAL_TICKET_ENABLED)

    def test_ticket_contract_has_no_portfolio_or_top_n_gate(self):
        prohibited = {
            "nav",
            "portfolio_value",
            "buying_power",
            "position_size",
            "sector_limit",
            "top_n",
            "quantity_number",
        }
        self.assertFalse(prohibited & {field.name for field in fields(TicketCandidate)})
        self.assertFalse(prohibited & {field.name for field in fields(ManualTicket)})

    def test_pre_holdout_state_and_weak_pop_evidence_are_rejected(self):
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(candidate(EvidenceState.VALIDATION_PASS), NOW)
        self.assertIn("HOLDOUT_PASS", str(caught.exception))

        weak = replace(candidate(), probabilities=probabilities(sample_size=99))
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(weak, NOW)
        self.assertIn("100 observations", str(caught.exception))

    def test_stale_missing_and_non_schwab_quotes_are_rejected(self):
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(candidate(quote_time=NOW - timedelta(minutes=6)), NOW)
        self.assertIn("stale", str(caught.exception))

        missing = replace(candidate(), leg_quotes=candidate().leg_quotes[:1])
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(missing, NOW)
        self.assertIn("quotes must match", str(caught.exception))

        wrong_source = replace(candidate(), quote_source="ORATS")
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(wrong_source, NOW)
        self.assertIn("SCHWAB", str(caught.exception))

    def test_positive_point_and_conservative_edge_are_both_required(self):
        weak_edge = compute_edge(
            (Scenario("profit", 0.7, 100.0), Scenario("loss", 0.3, -100.0)),
            (Scenario("profit", 0.4, 100.0), Scenario("loss", 0.6, -100.0)),
            maximum_loss=110.0,
            costs=CostBreakdown(1.0, 0.0, 0.0),
            model_fair_price=1.2,
            executable_limit_price=1.1,
            price_convention=PriceConvention.DEBIT,
        )
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(candidate(candidate_edge=weak_edge), NOW)
        self.assertIn("conservative net EV", str(caught.exception))

    def test_arbitrary_leg_geometry_and_non_natural_limit_are_rejected(self):
        original = candidate()
        reversed_long = replace(
            original.legs[0],
            occ_symbol="XYZ   261016C00110000",
            strike=110.0,
        )
        bad_geometry = replace(original, legs=(reversed_long, original.legs[1]))
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(bad_geometry, NOW)
        self.assertIn("bought strike below", str(caught.exception))

        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(candidate(candidate_edge=positive_edge(limit=1.50)), NOW)
        self.assertIn("natural price", str(caught.exception))

    def test_undefined_risk_catalog_structure_cannot_be_promoted(self):
        naked_leg = OptionLeg(
            "XYZ   261016C00105000",
            LegAction.SELL,
            OptionType.CALL,
            date(2026, 10, 16),
            105.0,
        )
        base = candidate()
        naked = replace(
            base,
            strategy_id="NAKED_CALL",
            hypothesis_id=hypothesis_id("NAKED_CALL"),
            evidence=evidence(strategy="NAKED_CALL"),
            legs=(naked_leg,),
            leg_quotes=(LegQuote(naked_leg.occ_symbol, 0.9, 1.0, NOW - timedelta(minutes=1)),),
            edge=replace(base.edge, price_convention=PriceConvention.CREDIT, executable_limit_price=0.9),
        )
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(naked, NOW)
        self.assertIn("research-only undefined-risk", str(caught.exception))

    def test_snapshot_profile_and_exit_policy_must_be_bound(self):
        unbound = replace(
            candidate(),
            snapshot_manifest=None,
            field_profile=None,
            model_calculation=None,
        )
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(unbound, NOW)
        self.assertIn("snapshot manifest is required", str(caught.exception))
        self.assertIn("field profile is required", str(caught.exception))
        self.assertIn("current model feature/score", str(caught.exception))

        changed_policy = replace(
            candidate(),
            policy=replace(candidate().policy, policy_version="invented-after-shadow"),
        )
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(changed_policy, NOW)
        self.assertIn("does not match frozen", str(caught.exception))

    def test_ticket_is_bound_to_exact_horizon_hypothesis(self):
        wrong_horizon = next(
            item.hypothesis_id
            for item in FROZEN_HYPOTHESIS_REGISTRY
            if item.strategy_id == "CALL_DEBIT_VERTICAL"
            and item.holding_sessions == 20
        )
        mismatched = replace(candidate(), hypothesis_id=wrong_horizon)
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(mismatched, NOW)
        message = str(caught.exception)
        self.assertIn("holding horizon", message)
        self.assertIn("family evidence", message)

    def test_pop_must_reconcile_to_saved_scenarios(self):
        base = candidate()

        def item(point, source):
            return replace(
                source,
                point=point,
                lower=max(0.0, point - 0.01),
                upper=min(1.0, point + 0.01),
            )

        incoherent = replace(
            base,
            probabilities=ProbabilityBundle(
                item(0.99, base.probabilities.pop_net),
                item(0.98, base.probabilities.p_target),
                item(0.98, base.probabilities.p_stop),
                item(0.97, base.probabilities.p_max_loss),
            ),
        )
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(incoherent, NOW)
        message = str(caught.exception)
        self.assertIn("POP_net does not reconcile", message)
        self.assertIn("P_max_loss does not reconcile", message)

    def test_self_asserted_payoff_fields_cannot_override_exact_legs(self):
        base = candidate()
        fabricated = compute_edge(
            base.edge.point_scenarios,
            base.edge.conservative_scenarios,
            maximum_loss=1.0,
            costs=base.edge.costs,
            model_fair_price=base.edge.model_fair_price,
            executable_limit_price=base.edge.executable_limit_price,
            price_convention=base.edge.price_convention,
            maximum_profit=9999.0,
            breakevens=(),
            target_pnl=0.0,
            stop_pnl=0.0,
            expected_shortfall=0.0,
            adverse_gap_stress_loss=0.0,
        )
        with self.assertRaises(TicketRejection) as caught:
            build_manual_ticket(replace(base, edge=fabricated), NOW)
        message = str(caught.exception)
        self.assertIn("maximum loss does not reconcile", message)
        self.assertIn("breakeven", message)
        self.assertIn("target P/L", message)
        self.assertIn("expected shortfall", message)

    def test_conservative_edge_cannot_be_more_optimistic(self):
        point = (
            Scenario("profit", 0.6, 100.0, ScenarioOutcome.TARGET),
            Scenario("loss", 0.4, -100.0, ScenarioOutcome.STOP),
        )
        optimistic = (
            Scenario("profit", 0.8, 101.0, ScenarioOutcome.TARGET),
            Scenario("loss", 0.2, -90.0, ScenarioOutcome.STOP),
        )
        with self.assertRaises(ValueError):
            compute_edge(
                point,
                optimistic,
                maximum_loss=110.0,
                costs=CostBreakdown(1.0, 0.1, 1.0),
                model_fair_price=1.2,
                executable_limit_price=1.1,
                price_convention=PriceConvention.DEBIT,
            )


if __name__ == "__main__":
    unittest.main()
