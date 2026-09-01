import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from cultra import public_history_sources as sources
from cultra.public_history_sources import (
    PublicHistorySourceError,
    analyze_public_history_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PARENT = PROJECT_ROOT / "var" / "historical" / "public_sources"


def _private_write(path, payload, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        path.write_bytes(payload)
    else:
        path.write_text(payload, encoding="utf-8")
    os.chmod(path, 0o600)


def _build_fixture(root):
    for selection, spec in zip(sources.SELECTION_DATES, sources._cboe_specs()):
        path = root / spec.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(sources._CBOE_FIELDS)
            date_text = selection.strftime("%Y/%m/%d")
            writer.writerow([date_text, "AAA", "AAA", "S", "CBOE", "600"])
            writer.writerow([date_text, "AAA", "AAA", "S", "BATS", "500"])
            writer.writerow([date_text, "ZZZ", "ZZZ", "S", "CBOE", "700"])
            writer.writerow([date_text, "ZZZ", "ZZZ", "S", "BATS", "400"])
            writer.writerow([date_text, "SPX", "SPX", "I", "CBOE", "1000"])
        os.chmod(path, 0o600)

    fake_pdf = b"%PDF-1.7\n" + (b"x" * 1200) + b"\n%%EOF\n"
    for relative in (
        "nyse_calendars/nyse_2024_trading_calendar.pdf",
        "nyse_calendars/nyse_2025_trading_calendar.pdf",
        "nyse_calendars/nyse_2026_trading_calendar.pdf",
        "nyse_calendars/nyse_2025_national_day_of_mourning.pdf",
    ):
        _private_write(root / relative, fake_pdf, binary=True)

    for selection in sources.SELECTION_DATES:
        _private_write(
            root
            / "occ_dlp_tombstones"
            / ("occ_dlp_%s.txt" % selection.isoformat()),
            "File requested does not exist.\n",
        )

    for index, (start, end) in enumerate(sources._OCC_SLICES, start=1):
        path = (
            root
            / "occ_info_memos"
            / (
                "occ_contract_adjustments_%s_to_%s.csv"
                % (start.isoformat(), end.isoformat())
            )
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(sources._OCC_FIELDS)
            writer.writerow(
                [
                    str(55000 + index),
                    start.strftime("%b-%d-%Y"),
                    "",
                    "Example Contract Adjustment Option Symbol: AAA",
                ]
            )
        os.chmod(path, 0o600)

    _private_write(
        root / "reference_current" / "sec_company_tickers_exchange.json",
        json.dumps(
            {
                "fields": ["cik", "name", "ticker", "exchange"],
                "data": [[1, "Example Inc.", "AAA", "Nasdaq"]],
            }
        ),
    )
    _private_write(
        root / "reference_current" / "nasdaqlisted.txt",
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
        "AAA|Example Inc.|Q|N|N|100|N|N\n"
        "File Creation Time: 0831202610:01|||||||\n",
    )
    _private_write(
        root / "reference_current" / "otherlisted.txt",
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "ZZZ|Example ETF|N|ZZZ|Y|100|N|ZZZ\n"
        "File Creation Time: 0831202610:01||||||\n",
    )


class PublicHistorySourceTests(unittest.TestCase):
    def setUp(self):
        SOURCE_PARENT.mkdir(parents=True, exist_ok=True)

    def test_public_sources_are_broad_discovery_not_a_false_freeze(self):
        with tempfile.TemporaryDirectory(dir=str(SOURCE_PARENT)) as temporary:
            root = Path(temporary)
            _build_fixture(root)
            result = analyze_public_history_sources(
                root,
                minimum_security_underlyings=2,
                minimum_liquid_candidates=2,
                require_complete_month=False,
            )
            self.assertEqual("PARTIAL_NOT_FREEZEABLE", result.audit["status"])
            self.assertEqual("UNPROVEN", result.audit["profit_confidence"])
            self.assertEqual(0, result.audit["orats_attempts"])
            self.assertEqual(450, result.calendar["session_count"])
            self.assertEqual(8, result.adjustment_index["memo_count"])
            self.assertEqual(
                [2, 2, 2, 2],
                [item["security_underlying_count"] for item in result.discovery["snapshots"]],
            )
            self.assertEqual(
                [2, 2, 2, 2],
                [item["liquid_candidate_count"] for item in result.discovery["snapshots"]],
            )
            self.assertIsNone(
                result.discovery["liquidity_policy"]["fixed_candidate_count"]
            )
            self.assertEqual(
                [2, 2, 2, 2],
                [
                    item["queue_count"]
                    for item in result.classification_queue["snapshots"]
                ],
            )
            self.assertFalse(result.classification_queue["queue_truncated"])
            self.assertEqual(
                "UNRESOLVED_STOCK_OR_ETP",
                result.discovery["asset_classification_status"],
            )
            self.assertFalse(
                result.discovery["current_reference_files_used_for_historical_classification"]
            )
            self.assertIn("POINT_IN_TIME_ASSET_TYPE", result.markdown)

    def test_transport_header_or_any_unexpected_file_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=str(SOURCE_PARENT)) as temporary:
            root = Path(temporary)
            _build_fixture(root)
            _private_write(root / "cboe_volume" / "download.csv.headers", "set-cookie: x=y\n")
            with self.assertRaisesRegex(PublicHistorySourceError, "inventory mismatch"):
                analyze_public_history_sources(
                    root,
                    minimum_security_underlyings=2,
                    minimum_liquid_candidates=2,
                    require_complete_month=False,
                )

    def test_duplicate_occ_memo_across_slices_fails_closed(self):
        with tempfile.TemporaryDirectory(dir=str(SOURCE_PARENT)) as temporary:
            root = Path(temporary)
            _build_fixture(root)
            start, end = sources._OCC_SLICES[1]
            path = (
                root
                / "occ_info_memos"
                / (
                    "occ_contract_adjustments_%s_to_%s.csv"
                    % (start.isoformat(), end.isoformat())
                )
            )
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.reader(handle))
            rows[1][0] = "55001"
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)
            with self.assertRaisesRegex(PublicHistorySourceError, "duplicated"):
                analyze_public_history_sources(
                    root,
                    minimum_security_underlyings=2,
                    minimum_liquid_candidates=2,
                    require_complete_month=False,
                )

    def test_incomplete_cboe_month_is_rejected_by_production_policy(self):
        with tempfile.TemporaryDirectory(dir=str(SOURCE_PARENT)) as temporary:
            root = Path(temporary)
            _build_fixture(root)
            with self.assertRaisesRegex(PublicHistorySourceError, "month is incomplete"):
                analyze_public_history_sources(
                    root,
                    minimum_security_underlyings=2,
                    minimum_liquid_candidates=2,
                    require_complete_month=True,
                )


if __name__ == "__main__":
    unittest.main()
