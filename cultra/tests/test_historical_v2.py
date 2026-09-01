import unittest

from cultra.historical_v2 import (
    HistoricalV2Error,
    normalize_chain_row,
    normalize_core_row,
    normalize_split_row,
)
from cultra.request_optimization import (
    HISTORICAL_CORE_FIELDS,
    HISTORICAL_SPLIT_FIELDS,
    HISTORICAL_STRIKE_FIELDS,
)
from cultra.requesting import Endpoint, RunType, make_planned_request


def _request(endpoint, fields, profile, params=None):
    return make_planned_request(
        logical_request_id="normalize-%s" % endpoint.name.lower(),
        endpoint=endpoint,
        run_type=RunType.HISTORICAL_BACKFILL,
        entities=("AAPL",),
        fields=fields,
        field_profile=profile,
        purpose="strict V2 normalizer test",
        expected_vintage="2026-08-28",
        expected_rows=1000,
        expected_bytes=1000,
        retry_limit=0,
        params=params,
    )


class HistoricalV2NormalizerTests(unittest.TestCase):
    def test_core_chain_and_split_rows_are_strictly_normalized(self):
        core_request = _request(
            Endpoint.HIST_CORES, HISTORICAL_CORE_FIELDS, "HIST_CORE_SIGNAL_V3"
        )
        core = {field: "0.2" for field in HISTORICAL_CORE_FIELDS}
        core.update(
            ticker="AAPL",
            tradeDate="2026-08-28",
            updatedAt="2026-08-28T22:00:00Z",
            priorCls="200",
        )
        normalized_core = normalize_core_row(core, core_request, "a" * 64)
        self.assertEqual("AAPL", normalized_core["ticker"])
        self.assertEqual(200.0, normalized_core["priorCls"])

        chain_request = _request(
            Endpoint.HIST_STRIKES,
            HISTORICAL_STRIKE_FIELDS,
            "HIST_ROTATING_COHORT_CHAIN_V2",
            params={"tradeDate": "2026-08-28", "dte": "20,180", "delta": "0,1"},
        )
        chain = {field: "0.2" for field in HISTORICAL_STRIKE_FIELDS}
        chain.update(
            ticker="AAPL",
            tradeDate="2026-08-28",
            expirDate="2026-10-02",
            updatedAt="2026-08-28T22:00:00Z",
            dte="35",
            strike="200",
            stockPrice="202",
            delta="0.55",
            callBidPrice="4.90",
            callAskPrice="5.10",
            putBidPrice="2.90",
            putAskPrice="3.10",
            callOpenInterest="100",
            putOpenInterest="120",
            callVolume="10",
            putVolume="12",
        )
        normalized_chain = normalize_chain_row(chain, chain_request, "b" * 64)
        self.assertEqual(35, normalized_chain["dte"])
        self.assertEqual(5.10, normalized_chain["call_ask"])

        split_request = _request(
            Endpoint.HIST_SPLITS, HISTORICAL_SPLIT_FIELDS, "HIST_SPLITS_V2"
        )
        split = normalize_split_row(
            {"ticker": "AAPL", "splitDate": "2026-06-01", "divisor": "4"},
            split_request,
            "c" * 64,
        )
        self.assertEqual(4.0, split["divisor"])

    def test_unknown_fields_nonfinite_numbers_and_dte_drift_fail_closed(self):
        core_request = _request(
            Endpoint.HIST_CORES, HISTORICAL_CORE_FIELDS, "HIST_CORE_SIGNAL_V3"
        )
        core = {field: "0.2" for field in HISTORICAL_CORE_FIELDS}
        core.update(
            ticker="AAPL",
            tradeDate="2026-08-28",
            updatedAt="2026-08-28T22:00:00Z",
            priorCls="NaN",
            futureProfit=20,
        )
        with self.assertRaises(HistoricalV2Error):
            normalize_core_row(core, core_request, "a" * 64)

        chain_request = _request(
            Endpoint.HIST_STRIKES,
            HISTORICAL_STRIKE_FIELDS,
            "HIST_ROTATING_COHORT_CHAIN_V2",
            params={"tradeDate": "2026-08-28", "dte": "20,180", "delta": "0,1"},
        )
        chain = {field: "0.2" for field in HISTORICAL_STRIKE_FIELDS}
        chain.update(
            ticker="AAPL",
            tradeDate="2026-08-28",
            expirDate="2026-10-02",
            updatedAt="2026-08-28T22:00:00Z",
            dte="30",
            strike="200",
            stockPrice="202",
            delta="0.55",
            callOpenInterest="100",
            putOpenInterest="120",
            callVolume="10",
            putVolume="12",
        )
        with self.assertRaisesRegex(HistoricalV2Error, "does not reconcile"):
            normalize_chain_row(chain, chain_request, "b" * 64)

    def test_future_revised_timestamp_cannot_enter_historical_evidence(self):
        request = _request(
            Endpoint.HIST_CORES, HISTORICAL_CORE_FIELDS, "HIST_CORE_SIGNAL_V3"
        )
        core = {field: "0.2" for field in HISTORICAL_CORE_FIELDS}
        core.update(
            ticker="AAPL",
            tradeDate="2026-08-28",
            updatedAt="2026-09-15T22:00:00Z",
            priorCls="200",
        )
        with self.assertRaisesRegex(HistoricalV2Error, "not contemporaneous"):
            normalize_core_row(core, request, "a" * 64)


if __name__ == "__main__":
    unittest.main()
