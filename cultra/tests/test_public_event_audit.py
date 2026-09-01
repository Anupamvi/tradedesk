import unittest
from datetime import date

from cultra.public_event_audit import (
    PublicEventAuditError,
    _candidate_only_status,
    _dividend_rows,
    _sec_financial_events,
    _successors,
    _title_mentions_option_symbol,
)


class PublicEventAuditTests(unittest.TestCase):
    def test_targeted_candidate_presence_never_becomes_complete_history(self):
        self.assertEqual(
            "BLOCKED_EARNINGS_CANDIDATES_PRESENT_COMPLETENESS_UNATTESTED",
            _candidate_only_status("EARNINGS", 3),
        )
        self.assertEqual(
            "BLOCKED_DIVIDEND_CANDIDATES_PRESENT_COMPLETENESS_UNATTESTED",
            _candidate_only_status("DIVIDEND", 1),
        )
        self.assertEqual(
            "BLOCKED_NO_FINANCIAL_EVENT_EVIDENCE",
            _candidate_only_status("EARNINGS", 0),
        )
        with self.assertRaises(PublicEventAuditError):
            _candidate_only_status("SPLIT", 1)

    def test_occ_symbol_matching_does_not_match_one_letter_company_prose(self):
        self.assertTrue(
            _title_mentions_option_symbol(
                "Example Corp - Contract Adjustment Option Symbol: T New Symbol: T1",
                "T",
            )
        )
        self.assertFalse(
            _title_mentions_option_symbol(
                "AT&T Inc. - Contract Adjustment Option Symbol: OTHER",
                "T",
            )
        )
        self.assertFalse(
            _title_mentions_option_symbol(
                "SentinelOne S transaction Option Symbol: OTHER", "S"
            )
        )

    def test_successor_mapping_uses_becomes_not_adjusted_option_suffix(self):
        records = (
            {
                "title": "Option Symbol: 08/07/2025 - PARA remains PARA 08/08/2025 - PARA becomes PSKY"
            },
            {
                "title": "Reverse Split Option Symbol: MSTZ New Symbol: MSTZ1"
            },
        )
        self.assertEqual(
            {"PARA": ("PSKY",)},
            _successors(records, {"PARA", "MSTZ"}),
        )

    def test_sec_item_202_and_named_foreign_results_are_conservative_events(self):
        domestic = {
            "filings": {
                "recent": {
                    "form": ["8-K", "8-K"],
                    "filingDate": ["2025-01-10", "2025-01-11"],
                    "accessionNumber": ["one", "two"],
                    "primaryDocument": ["one.htm", "two.htm"],
                    "items": ["2.02,9.01", "5.02"],
                    "primaryDocDescription": ["", ""],
                    "acceptanceDateTime": [
                        "2025-01-10T21:05:00.000Z",
                        "2025-01-11T21:05:00.000Z",
                    ],
                }
            }
        }
        events = _sec_financial_events(
            ticker="AAA",
            submission=domestic,
            start=date(2025, 1, 1),
            end=date(2025, 1, 31),
        )
        self.assertEqual(1, len(events))
        self.assertEqual("SEC_8K_ITEM_2_02", events[0]["source"])
        self.assertEqual(
            "CONSERVATIVE_WHOLE_SESSION_BLACKOUT", events[0]["timing_policy"]
        )

        foreign = {
            "filings": {
                "recent": {
                    "form": ["6-K"],
                    "filingDate": ["2025-02-13"],
                    "accessionNumber": ["three"],
                    "primaryDocument": ["results.htm"],
                    "items": [""],
                    "primaryDocDescription": ["FINAL RESULTS"],
                    "acceptanceDateTime": ["2025-02-13T16:34:58.000Z"],
                }
            }
        }
        events = _sec_financial_events(
            ticker="UL",
            submission=foreign,
            start=date(2025, 1, 1),
            end=date(2025, 3, 1),
        )
        self.assertEqual(1, len(events))
        self.assertEqual("SEC_6K_FINANCIAL_DESCRIPTION", events[0]["source"])

    def test_unavailable_dividend_is_not_rewritten_as_no_dividend(self):
        unavailable = {
            "data": {
                "dividends": {"rows": None},
            },
            "message": "Dividend History is unavailable",
            "status": {"rCode": 200},
        }
        self.assertEqual((), _dividend_rows(unavailable, ticker="AAA"))

        malformed = {
            "data": {
                "dividends": {
                    "rows": [
                        {
                            "exOrEffDate": "01/02/2025",
                            "type": "Cash",
                            "amount": "$0.00",
                            "currency": "USD",
                        }
                    ]
                }
            },
            "message": None,
            "status": {"rCode": 200},
        }
        with self.assertRaises(PublicEventAuditError):
            _dividend_rows(malformed, ticker="AAA")


if __name__ == "__main__":
    unittest.main()
