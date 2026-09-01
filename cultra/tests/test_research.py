import unittest

from cultra.research import (
    ChainQuote,
    ResearchError,
    _global_split_dates,
    _occ_symbol,
    _relative_spread,
    effective_validation_config,
    run_historical_validation,
)


class HistoricalResearchTests(unittest.TestCase):
    def test_pre_outcome_amendment_is_narrow_and_reproducible(self):
        config = effective_validation_config()
        self.assertEqual("CULTRA_HISTORICAL_VALIDATION_V1_1", config["effective_version"])
        self.assertEqual([52, 60], config["data"]["entry_dte"])
        self.assertEqual(56, config["data"]["preferred_dte"])
        self.assertEqual(20, config["exit_policy"]["time_exit_sessions"])
        self.assertEqual(60, config["split_policy"]["embargo_sessions"])
        self.assertEqual(0.20, config["split_policy"]["holdout_fraction"])

    def test_global_calendar_split_has_both_sixty_session_embargoes(self):
        sessions = tuple("2025-%03d" % index for index in range(450))
        # The split function validates ISO dates, so use the real frozen calendar.
        from cultra.backfill import load_recent_sessions

        split = _global_split_dates(load_recent_sessions())
        self.assertEqual(126, len(split["training"]))
        self.assertEqual(82, len(split["validation"]))
        self.assertEqual(82, len(split["holdout"]))
        self.assertEqual(120, len(split["embargoed"]))

    def test_occ_and_spread_are_exact(self):
        self.assertEqual("SPY260918C00750000", _occ_symbol("SPY", "2026-09-18", "CALL", 750.0))
        quote = ChainQuote(
            trade_date="2026-08-28",
            ticker="SPY",
            expiry="2026-09-18",
            strike=750.0,
            dte=21,
            stock_price=769.0,
            call_bid=25.0,
            call_ask=25.5,
            put_bid=5.0,
            put_ask=5.2,
            smv_vol=0.15,
            delta=0.6,
            call_open_interest=1000,
            put_open_interest=1000,
            updated_at="2026-08-28T20:45:00Z",
            snapshot_id="a" * 64,
        )
        self.assertAlmostEqual(0.5 / 25.25, _relative_spread(quote, "CALL"))

    def test_exposed_v1_holdout_is_permanently_fail_closed(self):
        with self.assertRaisesRegex(ResearchError, "HOLDOUT INVALIDATED"):
            run_historical_validation()


if __name__ == "__main__":
    unittest.main()
