import unittest
from datetime import date, datetime, timezone

from cultra.catalog import (
    CATALOG_VERSION,
    FROZEN_STRATEGY_CATALOG,
    StrategyCategory,
    get_strategy,
    iter_research_only,
    iter_ticket_eligible,
)
from cultra.domain import (
    EvidenceState,
    LegAction,
    LegQuote,
    OptionLeg,
    OptionType,
    ProbabilityBundle,
    ProbabilityEstimate,
    exact_quote_map,
)


class DomainTests(unittest.TestCase):
    def test_option_leg_and_quote_are_exact_and_validated(self):
        leg = OptionLeg(
            "AAPL  261218C00200000",
            LegAction.BUY,
            OptionType.CALL,
            date(2026, 12, 18),
            200.0,
        )
        quote = LegQuote(
            leg.occ_symbol,
            2.0,
            2.2,
            datetime(2026, 8, 30, 16, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(quote.midpoint, 2.1)
        self.assertAlmostEqual(quote.spread, 0.2)
        self.assertIs(exact_quote_map((quote,))[leg.occ_symbol], quote)

    def test_domain_rejects_invalid_geometry_and_quotes(self):
        with self.assertRaises(ValueError):
            OptionLeg("X", LegAction.BUY, OptionType.CALL, date(2026, 1, 1), 0.0)
        with self.assertRaises(ValueError):
            OptionLeg("X", LegAction.BUY, OptionType.CALL, date(2026, 1, 1), 1.0, 0)
        with self.assertRaises(ValueError):
            LegQuote(
                "XYZ   261218C00100000",
                2.0,
                1.0,
                datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        with self.assertRaises(ValueError):
            LegQuote("XYZ   261218C00100000", 1.0, 2.0, datetime(2026, 1, 1))
        timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
        quote = LegQuote("XYZ   261218C00100000", 1.0, 2.0, timestamp)
        with self.assertRaises(ValueError):
            exact_quote_map((quote, quote))

    def test_occ_identity_must_match_leg_fields(self):
        with self.assertRaises(ValueError):
            OptionLeg(
                "XYZ   261218P00100000",
                LegAction.BUY,
                OptionType.CALL,
                date(2026, 12, 18),
                100.0,
            )
        with self.assertRaises(ValueError):
            OptionLeg(
                "XYZ   261218C00105000",
                LegAction.BUY,
                OptionType.CALL,
                date(2026, 12, 18),
                100.0,
            )

    def test_probability_bundle_requires_one_model_version(self):
        def estimate(version):
            return ProbabilityEstimate(
                0.6,
                0.5,
                0.7,
                100,
                version,
                date(2025, 1, 1),
                date(2025, 12, 31),
            )

        ProbabilityBundle(estimate("v1"), estimate("v1"), estimate("v1"), estimate("v1"))
        with self.assertRaises(ValueError):
            ProbabilityBundle(estimate("v1"), estimate("v2"), estimate("v1"), estimate("v1"))
        with self.assertRaises(ValueError):
            ProbabilityEstimate(0.6, 0.7, 0.8, 100, "v1", date(2025, 1, 1), date(2025, 2, 1))

    def test_evidence_states_are_explicit_not_scores(self):
        self.assertEqual(
            [state.value for state in EvidenceState],
            [
                "UNPROVEN",
                "RESEARCH_PASS",
                "VALIDATION_PASS",
                "HOLDOUT_PASS",
                "SHADOW_PASS",
                "MANUAL_TICKET_ENABLED",
            ],
        )

    def test_frozen_catalog_is_finite_unique_and_complete(self):
        self.assertEqual(CATALOG_VERSION, "cultra-options-catalog-v1")
        identifiers = [item.strategy_id for item in FROZEN_STRATEGY_CATALOG]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertGreaterEqual(len(identifiers), 25)
        self.assertEqual(
            {item.category for item in FROZEN_STRATEGY_CATALOG}, set(StrategyCategory)
        )
        for definition in FROZEN_STRATEGY_CATALOG:
            self.assertGreaterEqual(definition.holding_sessions_min, 20)
            self.assertLessEqual(definition.holding_sessions_max, 60)
        self.assertIs(get_strategy("IRON_CONDOR"), get_strategy("IRON_CONDOR"))
        with self.assertRaises(KeyError):
            get_strategy("INVENTED_AFTER_HOLDOUT")

    def test_undefined_risk_is_research_only_and_capped_variants_exist(self):
        research_only = {item.strategy_id for item in iter_research_only()}
        eligible = {item.strategy_id for item in iter_ticket_eligible()}
        self.assertTrue({"NAKED_CALL", "NAKED_PUT", "SHORT_STRANGLE"} <= research_only)
        self.assertTrue(
            {"WING_CAPPED_SHORT_STRANGLE", "WING_CAPPED_CALL_RATIO"} <= eligible
        )
        self.assertTrue(
            all(get_strategy(item).defined_risk_by_construction for item in eligible)
        )


if __name__ == "__main__":
    unittest.main()
