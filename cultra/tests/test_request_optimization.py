import hashlib
import json
import unittest
from datetime import date, timedelta

from cultra.request_optimization import (
    ExactStrikeHistoryKey,
    RotatingCohortPolicy,
    SignalDateKey,
    build_bulk_history_feature_plan,
    build_exact_strike_history_plan,
    build_rotating_cohort_requests,
    build_rotating_cohort_slices,
    build_signal_entry_plan,
    daily_request_budget,
    historical_campaign_forecast,
    rotating_cohort_campaign_forecast,
)
from cultra.requesting import Endpoint, PlanningError


def tickers(count):
    return tuple("T%03d" % index for index in range(count))


def sessions(count=450):
    start = date(2025, 1, 1)
    return tuple((start + timedelta(days=index)).isoformat() for index in range(count))


def cohort_manifest(values):
    blocks = []
    for block_index, offset in enumerate(range(0, len(values), 120)):
        block = values[offset : offset + 120]
        eligible = max(0, len(block) - 61)
        symbols = tickers(40)[block_index * 10 : block_index * 10 + 10]
        blocks.append(
            {
                "block_index": block_index,
                "selection_date": block[0],
                "block_start": block[0],
                "block_end": block[-1],
                "required_coverage_through": block[-1],
                "eligible_signal_session_count": eligible,
                "last_eligible_signal_date": block[eligible - 1] if eligible else None,
                "future_membership_used_for_selection": False,
                "tickers": list(symbols),
                "strata": ["STOCK"] * 10,
            }
        )
    payload = {
        "schema": "cultra.rotating-historical-cohorts.v1",
        "selection_policy": "POINT_IN_TIME_STRATIFIED_DETERMINISTIC_SAMPLE",
        "daily_production_universe_cap": None,
        "research_sample_is_not_a_ticket_output_cap": True,
        "universe_id": "test-universe",
        "universe_fingerprint": "a" * 64,
        "session_start": values[0],
        "session_end": values[-1],
        "session_count": len(values),
        "cohort_size": 10,
        "block_sessions": 120,
        "maximum_holding_sessions": 60,
        "minimum_point_in_time_universe": 100,
        "minimum_stock_fraction": 0.8,
        "transition_policy": "CENSOR_ENTRIES_BEFORE_COHORT_ROTATION",
        "blocks": blocks,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return dict(payload, freeze_hash=hashlib.sha256(encoded).hexdigest())


class RequestOptimizationTests(unittest.TestCase):
    def test_daily_cold_and_warm_budgets_are_exact(self):
        current = daily_request_budget(
            core_symbols=254,
            summary_symbols=120,
            monies_symbols=40,
            exact_contracts=250,
        )
        self.assertEqual(49, current["logical_requests"])
        self.assertTrue(current["admissible"])
        self.assertEqual(0, current["same_vintage_warm_attempts"])
        absolute = daily_request_budget(
            core_symbols=600,
            summary_symbols=120,
            monies_symbols=40,
            exact_contracts=250,
        )
        self.assertEqual(83, absolute["worst_charged_attempts"])
        self.assertFalse(absolute["admissible"])
        self.assertEqual(23, absolute["requests_over_cap"])

    def test_complete_base_campaign_is_474_not_a_full_universe_grid(self):
        forecast = rotating_cohort_campaign_forecast(
            RotatingCohortPolicy(eligible_symbols=254)
        )
        self.assertEqual(20, forecast["requests"]["historical_core"])
        self.assertEqual(450, forecast["requests"]["historical_chain_total"])
        self.assertEqual(4, forecast["requests"]["split_history"])
        self.assertEqual(474, forecast["requests"]["cold_cache_total"])
        self.assertEqual(6, forecast["slicing"]["slice_count"])
        self.assertEqual([90, 90, 90, 90, 90, 24], forecast["slicing"]["exact_slice_attempts"])
        self.assertEqual(474, forecast["slicing"]["initial_campaign_max_actual_attempts"])
        self.assertEqual(540, forecast["slicing"]["sum_of_generic_slice_caps"])
        self.assertEqual(66, forecast["slicing"]["unused_cap_capacity_not_authorized"])
        self.assertEqual(2, forecast["historical_core_ticker_batch_size"])
        self.assertEqual(10, forecast["historical_chain_ticker_batch_size"])
        self.assertEqual(183, forecast["requests"]["optional_continuous_entry_extension_chain_batches"])
        self.assertEqual(2060, forecast["research_capacity"]["maximum_horizon_ticker_date_candidates_before_signals"])

    def test_exact_requests_and_slices_reconcile_to_forecast(self):
        dates = sessions()
        manifest = cohort_manifest(dates)
        requests = build_rotating_cohort_requests(
            eligible_symbols=tickers(254),
            sessions=dates,
            cohort_manifest=manifest,
            through_date=dates[-1],
        )
        self.assertEqual(474, len(requests))
        counts = {}
        for item in requests:
            counts[item.endpoint] = counts.get(item.endpoint, 0) + 1
        self.assertEqual(20, counts[Endpoint.HIST_CORES])
        self.assertEqual(450, counts[Endpoint.HIST_STRIKES])
        self.assertEqual(4, counts[Endpoint.HIST_SPLITS])
        plans = build_rotating_cohort_slices(
            campaign_id="campaign-v2",
            eligible_symbols=tickers(254),
            sessions=dates,
            cohort_manifest=manifest,
            through_date=dates[-1],
        )
        self.assertEqual(6, len(plans))
        self.assertEqual([90, 90, 90, 90, 90, 24], [item.logical_count for item in plans])
        self.assertTrue(all(item.retry_reserve == 0 for item in plans))
        self.assertEqual(474, sum(item.logical_count for item in plans))
        chain_requests = tuple(
            item for item in requests if item.endpoint is Endpoint.HIST_STRIKES
        )
        self.assertTrue(
            all(dict(item.params)["delta"] == "0,1" for item in chain_requests)
        )

    def test_future_membership_or_signal_cutoff_tampering_fails_closed(self):
        dates = sessions()
        manifest = cohort_manifest(dates)
        manifest["blocks"][0]["future_membership_used_for_selection"] = True
        payload = dict(manifest)
        payload.pop("freeze_hash")
        manifest["freeze_hash"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        with self.assertRaisesRegex(PlanningError, "future-membership"):
            build_rotating_cohort_requests(
                eligible_symbols=tickers(254),
                sessions=dates,
                cohort_manifest=manifest,
                through_date=dates[-1],
            )

    def test_superseded_n_plus_one_planners_are_disabled(self):
        with self.assertRaisesRegex(PlanningError, "rotating-cohort"):
            historical_campaign_forecast(symbols=tickers(254))
        with self.assertRaisesRegex(PlanningError, "standalone broad"):
            build_bulk_history_feature_plan(
                run_id="old",
                campaign_id="old",
                symbols=tickers(254),
                through_date="2026-08-28",
            )
        with self.assertRaisesRegex(PlanningError, "request-per-signal"):
            build_signal_entry_plan(
                run_id="old",
                campaign_id="old",
                signal_dates=(SignalDateKey("2026-06-01", "AAPL"),),
            )
        with self.assertRaisesRegex(PlanningError, "request-per-exact-strike"):
            build_exact_strike_history_plan(
                run_id="old",
                campaign_id="old",
                exact_strikes=(ExactStrikeHistoryKey("AAPL", "2026-07-17", 200),),
                through_date="2026-08-28",
            )


if __name__ == "__main__":
    unittest.main()
