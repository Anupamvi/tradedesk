import json
import tempfile
import unittest
from pathlib import Path

from cultra.eod import (
    CORE_SCREEN_FIELDS,
    build_core_plan,
    decode_rows,
    select_chain_finalists,
)


class BroadEodTests(unittest.TestCase):
    def test_full_core_profile_batches_eighty_symbols_into_eight_requests(self):
        symbols = ["T%02d" % index for index in range(80)]
        plan = build_core_plan(
            run_id="test-full-core",
            symbols=symbols,
            expected_vintage="2026-08-28",
            fields=CORE_SCREEN_FIELDS,
        )
        self.assertEqual(8, plan.logical_count)
        self.assertEqual(80, sum(len(item.entities) for item in plan.requests))
        self.assertLessEqual(plan.worst_charged_attempts, 49)
        self.assertIn("wksNextErn", plan.requests[0].fields)
        self.assertIn("fcstR2", plan.requests[0].fields)

    def test_full_liquidity_eligible_universe_fits_below_request_100(self):
        symbols = ["T%03d" % index for index in range(503)]
        plan = build_core_plan(
            run_id="test-broad-core",
            symbols=symbols,
            expected_vintage="2026-08-28",
            fields=CORE_SCREEN_FIELDS,
        )
        self.assertEqual(51, plan.logical_count)
        self.assertEqual(503, sum(len(item.entities) for item in plan.requests))
        self.assertLess(plan.worst_charged_attempts, 100)

    def test_response_decoder_preserves_provider_rows(self):
        raw = json.dumps(
            {"data": [{"ticker": "AAPL", "tradeDate": "2026-08-28"}]}
        ).encode("utf-8")
        self.assertEqual("AAPL", decode_rows(raw)[0]["ticker"])

    def test_chain_selection_has_no_default_top_n_cap(self):
        history = {"rows": []}
        analytics = {"rows": []}
        for index in range(45):
            ticker = "T%02d" % index
            history["rows"].append({"ticker": ticker, "trend_score": index / 100.0})
            analytics["rows"].append(
                {
                    "ticker": ticker,
                    "iv30d": 20.0,
                    "orFcst20d": 21.0,
                    "confidence": 80.0,
                    "atmIvM1": 20.0,
                    "atmFcstIvM1": 21.0,
                    "fcstR2": 0.5,
                    "cOi": 1000,
                    "pOi": 1000,
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.json"
            analytics_path = Path(directory) / "orats.json"
            history_path.write_text(json.dumps(history), encoding="utf-8")
            analytics_path.write_text(json.dumps(analytics), encoding="utf-8")
            result = select_chain_finalists(
                history_screen=history_path, orats_enrichment=analytics_path
            )
        self.assertEqual(45, len(result["selected_symbols"]))
        self.assertEqual([], result["budget_unresolved"])
        self.assertIsNone(result["capacity"])
        self.assertEqual(64, len(result["input_fingerprints"]["history_screen_sha256"]))


if __name__ == "__main__":
    unittest.main()
