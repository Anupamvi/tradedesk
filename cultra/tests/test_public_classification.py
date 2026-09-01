import unittest
from datetime import date

from cultra.public_classification import _classification


def _submission(cik, ticker, exchange, form, filing_date, document, entity_type="operating"):
    return {
        "cik": "%010d" % cik,
        "entityType": entity_type,
        "tickers": [ticker] if ticker else [],
        "exchanges": [exchange] if ticker else [],
        "filings": {
            "recent": {
                "form": [form],
                "filingDate": [filing_date],
                "reportDate": [filing_date],
                "accessionNumber": ["0000000000-24-000001"],
                "primaryDocument": [document],
            }
        },
    }


def _classify(
    ticker,
    *,
    current_map,
    flags,
    submissions,
    adjustments=(),
):
    return _classification(
        ticker=ticker,
        selection_date=date(2024, 11, 11),
        cboe_role="CBOE_ALL_SYMBOLS_DAILY_VOLUME_2024-11-11",
        current_map=current_map,
        nasdaq_flags=flags,
        submissions=submissions,
        adjustment_records=adjustments,
        occ_roles=("OCC_CONTRACT_ADJUSTMENT_OPTIONS_INDEX_TEST",),
        sec_current_role="SEC_CURRENT_COMPANY_TICKER_EXCHANGE_REFERENCE",
    )


class PublicClassificationTests(unittest.TestCase):
    def test_current_identity_requires_a_pre_date_sec_filing(self):
        submission = _submission(
            1,
            "AAA",
            "Nasdaq",
            "10-Q",
            "2024-10-01",
            "aaa-20240930.htm",
        )
        result = _classify(
            "AAA",
            current_map={"AAA": ((1, "Nasdaq"),)},
            flags={"AAA": ("N", "NASDAQ_CURRENT_LISTED_SYMBOL_DIRECTORY")},
            submissions={1: submission},
        )
        self.assertEqual("STOCK", result["asset_type"])
        self.assertEqual("VERIFIED_POINT_IN_TIME", result["classification_status"])

        future_only = _submission(
            1,
            "AAA",
            "Nasdaq",
            "10-Q",
            "2024-12-01",
            "aaa-20240930.htm",
        )
        unresolved = _classify(
            "AAA",
            current_map={"AAA": ((1, "Nasdaq"),)},
            flags={"AAA": ("N", "NASDAQ_CURRENT_LISTED_SYMBOL_DIRECTORY")},
            submissions={1: future_only},
        )
        self.assertEqual("UNRESOLVED_STOCK_OR_ETP", unresolved["asset_type"])

    def test_historical_document_identity_recovers_a_delisted_stock(self):
        historical = _submission(
            2,
            "",
            "",
            "10-Q",
            "2024-10-01",
            "old-20240930.htm",
        )
        result = _classify(
            "OLD",
            current_map={},
            flags={},
            submissions={2: historical},
        )
        self.assertEqual("STOCK", result["asset_type"])
        self.assertEqual(
            "HISTORICAL_SEC_DOCUMENT_PREFIX_WITH_CBOE_AND_OCC_IDENTITY",
            result["classification_method"],
        )

    def test_symbol_reuse_ignores_the_current_cik_and_uses_historical_filing(self):
        current_reuse = _submission(
            3,
            "PARA",
            "Nasdaq",
            "10-K",
            "2024-10-01",
            "different-20240930.htm",
        )
        historical = _submission(
            4,
            "",
            "",
            "10-K",
            "2024-10-01",
            "para-20240930.htm",
        )
        result = _classify(
            "PARA",
            current_map={"PARA": ((3, "Nasdaq"),)},
            flags={"PARA": ("N", "NASDAQ_CURRENT_LISTED_SYMBOL_DIRECTORY")},
            submissions={3: current_reuse, 4: historical},
            adjustments=(
                {
                    "post_date": "2025-08-07",
                    "memo_number": "1",
                    "title": "Option Symbol: PARA becomes PSKY",
                },
            ),
        )
        self.assertEqual(4, result["cik"])
        self.assertEqual("STOCK", result["asset_type"])

    def test_current_etf_and_out_of_scope_etn_are_not_stocks(self):
        etf = _classify(
            "FUND",
            current_map={},
            flags={"FUND": ("Y", "NASDAQ_CURRENT_LISTED_SYMBOL_DIRECTORY")},
            submissions={},
        )
        self.assertEqual("ETF", etf["asset_type"])

        etn_submission = _submission(
            5,
            "NOTE",
            "CBOE",
            "20-F",
            "2024-10-01",
            "issuer-20240930.htm",
            entity_type="other",
        )
        etn = _classify(
            "NOTE",
            current_map={"NOTE": ((5, "CBOE"),)},
            flags={"NOTE": ("N", "NASDAQ_CURRENT_OTHER_LISTED_SYMBOL_DIRECTORY")},
            submissions={5: etn_submission},
        )
        self.assertEqual("INELIGIBLE_OTHER_SECURITY", etn["asset_type"])


if __name__ == "__main__":
    unittest.main()
