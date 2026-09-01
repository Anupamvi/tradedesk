import unittest

from cultra.hypotheses import HYPOTHESIS_REGISTRY_HASH
from cultra.protocol import (
    build_campaign_freeze_receipt,
    historical_protocol_hash,
    load_historical_campaign_protocol,
)
from cultra.structures import STRUCTURE_TEMPLATE_REGISTRY_HASH


class HistoricalProtocolTests(unittest.TestCase):
    def test_canonical_protocol_has_no_named_list_and_reconciles_474(self):
        value = load_historical_campaign_protocol()
        self.assertIsNone(value["scope"]["named_universe"])
        self.assertIsNone(value["scope"]["ticket_output_cap"])
        self.assertEqual(474, value["acquisition"]["expected_cold_attempts"])
        self.assertEqual(2, value["acquisition"]["historical_core_ticker_batch_size"])
        self.assertEqual(183, value["acquisition"]["optional_continuous_entry_extension_attempts"])
        self.assertEqual(90, value["promotion_policy"]["holm_family_size"])
        self.assertEqual(59, value["split_policy"]["validation_sessions"])
        self.assertEqual(
            [59, 59, 59],
            [
                item["signal_sessions"]
                for item in value["split_policy"]["development_signal_windows"]
            ],
        )
        self.assertEqual(
            5,
            value["promotion_policy"]["calendar_concentration_period_sessions"],
        )
        self.assertEqual(HYPOTHESIS_REGISTRY_HASH, value["hypothesis_registry"]["registry_hash"])
        self.assertEqual(
            STRUCTURE_TEMPLATE_REGISTRY_HASH,
            value["hypothesis_registry"]["structure_template_registry_hash"],
        )
        self.assertFalse(value["learning_policy"]["outcome_dependent_prescreen_allowed"])
        self.assertEqual(64, len(historical_protocol_hash()))

    def test_freeze_receipt_binds_protocol_cohort_sessions_and_events(self):
        receipt = build_campaign_freeze_receipt(
            cohort_manifest={
                "schema": "cultra.rotating-historical-cohorts.v1",
                "freeze_hash": "a" * 64,
                "universe_fingerprint": "b" * 64,
            },
            session_calendar_sha256="c" * 64,
            event_manifest_sha256="d" * 64,
            prerequisite_freeze_sha256="e" * 64,
        )
        self.assertEqual(64, len(receipt["receipt_hash"]))
        self.assertEqual(HYPOTHESIS_REGISTRY_HASH, receipt["hypothesis_registry_hash"])
        self.assertEqual("e" * 64, receipt["prerequisite_freeze_sha256"])


if __name__ == "__main__":
    unittest.main()
