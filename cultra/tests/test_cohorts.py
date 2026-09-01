import json
import tempfile
import unittest
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

from cultra.cohorts import (
    CohortError,
    PointInTimeMember,
    PointInTimeUniverse,
    eligible_members,
    freeze_rotating_cohorts,
    load_point_in_time_universe,
)


def session_dates():
    start = date(2025, 1, 1)
    return tuple(start + timedelta(days=index) for index in range(450))


def universe():
    sessions = session_dates()
    selection_dates = (sessions[0], sessions[120], sessions[240], sessions[360])
    members = []
    for block, observed_at in enumerate(selection_dates):
        for index in range(120):
            members.append(
                PointInTimeMember(
                    ticker="B%dT%03d" % (block, index),
                    asset_type="STOCK" if index < 110 else "ETF",
                    eligible_from=observed_at,
                    eligible_through=sessions[-1],
                    observed_at=observed_at,
                    optionable=True,
                    sampling_stratum="S%02d" % (index % 10),
                    liquidity_rank=index + 1,
                )
            )
    return PointInTimeUniverse(
        universe_id="pit-v1",
        provider="independent-source",
        source_uri="cultra://frozen/pit-source",
        source_sha256="a" * 64,
        coverage="US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_ACROSS_2_CBOE_VENUES",
        members=tuple(members),
    )


class CohortTests(unittest.TestCase):
    def test_exact_selection_date_snapshot_is_required(self):
        item = universe()
        selection = session_dates()[120]
        values = eligible_members(item, selection_date=selection)
        self.assertEqual(120, len(values))
        self.assertTrue(all(member.observed_at == selection for member in values))
        self.assertFalse(eligible_members(item, selection_date=selection + timedelta(days=1)))

    def test_freeze_is_disjoint_stock_relevant_and_censors_boundaries(self):
        manifest = freeze_rotating_cohorts(universe(), session_dates())
        self.assertEqual(4, len(manifest["blocks"]))
        sampled = [ticker for block in manifest["blocks"] for ticker in block["tickers"]]
        self.assertEqual(40, len(sampled))
        self.assertEqual(40, len(set(sampled)))
        self.assertEqual([59, 59, 59, 29], [block["eligible_signal_session_count"] for block in manifest["blocks"]])
        self.assertEqual("CENSOR_ENTRIES_BEFORE_COHORT_ROTATION", manifest["transition_policy"])
        self.assertTrue(all(block["required_coverage_through"] == block["block_end"] for block in manifest["blocks"]))
        self.assertTrue(manifest["stock_floor_enforced_during_selection"])

    def test_stock_floor_is_enforced_during_selection_not_checked_afterward(self):
        sessions = session_dates()[:120]
        observed_at = sessions[0]
        members = tuple(
            PointInTimeMember(
                ticker="T%03d" % index,
                asset_type="ETF" if index < 20 else "STOCK",
                eligible_from=observed_at,
                eligible_through=sessions[-1],
                observed_at=observed_at,
                optionable=True,
                sampling_stratum="S%02d" % (index % 10),
                liquidity_rank=index + 1,
            )
            for index in range(120)
        )
        item = PointInTimeUniverse(
            universe_id="stock-floor-test",
            provider="independent-source",
            source_uri="cultra://frozen/stock-floor-test",
            source_sha256="c" * 64,
            coverage="US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_ACROSS_2_CBOE_VENUES",
            members=members,
        )
        manifest = freeze_rotating_cohorts(item, sessions)
        by_ticker = {member.ticker: member for member in members}
        selected = manifest["blocks"][0]["tickers"]
        self.assertGreaterEqual(
            sum(by_ticker[ticker].asset_type == "STOCK" for ticker in selected),
            8,
        )

    def test_broad_population_can_preserve_unresolved_names_without_selecting_them(self):
        sessions = session_dates()[:120]
        observed_at = sessions[0]
        members = tuple(
            PointInTimeMember(
                ticker="U%03d" % index,
                asset_type="STOCK" if index < 10 else "UNRESOLVED_STOCK_OR_ETP",
                eligible_from=observed_at,
                eligible_through=sessions[-1],
                observed_at=observed_at,
                optionable=True,
                sampling_stratum="S%02d" % (index % 10),
                liquidity_rank=index + 1,
            )
            for index in range(120)
        )
        item = PointInTimeUniverse(
            universe_id="unresolved-preservation-test",
            provider="independent-source",
            source_uri="cultra://frozen/unresolved-preservation-test",
            source_sha256="d" * 64,
            coverage="US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_ACROSS_2_CBOE_VENUES",
            members=members,
        )
        manifest = freeze_rotating_cohorts(item, sessions)
        block = manifest["blocks"][0]
        self.assertEqual(120, block["point_in_time_population_count"])
        self.assertEqual(10, block["resolved_classification_count"])
        self.assertTrue(all(int(ticker[1:]) < 10 for ticker in block["tickers"]))

    def test_future_or_stale_snapshot_and_outcome_fields_are_rejected(self):
        item = universe()
        first = item.members[0]
        with self.assertRaises(CohortError):
            replace(first, observed_at=first.eligible_from - timedelta(days=1))

        with tempfile.TemporaryDirectory(dir=str(Path(__file__).resolve().parents[1])) as temporary:
            path = Path(temporary) / "universe.json"
            payload = {
                "schema": "cultra.point-in-time-universe.v1",
                "universe_id": "pit",
                "provider": "source",
                "source_uri": "cultra://source",
                "source_sha256": "b" * 64,
                "coverage": "US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_ACROSS_2_CBOE_VENUES",
                "members": [
                    {
                        "ticker": "AAPL",
                        "asset_type": "STOCK",
                        "eligible_from": "2025-01-01",
                        "eligible_through": "2025-12-31",
                        "observed_at": "2025-01-01",
                        "optionable": True,
                        "sampling_stratum": "TECH",
                        "liquidity_rank": 1,
                        "future_profit": 12.0,
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(CohortError, "unfrozen fields"):
                load_point_in_time_universe(path)


if __name__ == "__main__":
    unittest.main()
