import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from cultra.universe import (
    PROJECT_ROOT,
    fetch_finalist_chains,
    load_spy_holdings,
    local_screen,
    rebuild_broad_screen_offline,
)


class BroadUniverseTests(unittest.TestCase):
    def test_exact_chain_fetch_has_no_arbitrary_forty_symbol_cap(self):
        class FakeBoundary:
            def option_chain(self, symbol, *, from_date, to_date):
                stamp = datetime(2026, 8, 28, 20, 0, tzinfo=timezone.utc)
                return SimpleNamespace(
                    underlying_quote=SimpleNamespace(
                        bid=99.9, ask=100.1, last=100.0, timestamp=stamp
                    ),
                    timestamp=stamp,
                    contracts=(),
                )

        symbols = ["T%02d" % index for index in range(41)]
        with TemporaryDirectory(dir=str(PROJECT_ROOT / "out")) as temporary:
            output = Path(temporary) / "chains.json"
            result = fetch_finalist_chains(
                symbols=symbols,
                output_path=output,
                from_date=datetime(2026, 9, 1).date(),
                to_date=datetime(2026, 11, 30).date(),
                workers=4,
                provider_factory=FakeBoundary,
            )
        self.assertEqual(41, result["resolved_count"])
        self.assertEqual(41, len(result["requested"]))

    def test_decision_refresh_is_schwab_complete_and_has_no_order_surface(self):
        class FakeBoundary:
            def option_chain(self, symbol, *, from_date, to_date):
                stamp = datetime(2026, 8, 31, 14, 35, tzinfo=timezone.utc)
                return SimpleNamespace(
                    underlying_quote=SimpleNamespace(
                        bid=99.9, ask=100.1, last=100.0, timestamp=stamp
                    ),
                    timestamp=stamp,
                    contracts=(),
                )

        with TemporaryDirectory(dir=str(PROJECT_ROOT / "out")) as temporary:
            result = fetch_finalist_chains(
                symbols=("AAA", "BBB"),
                output_path=Path(temporary) / "decision.json",
                from_date=datetime(2026, 9, 1).date(),
                to_date=datetime(2026, 11, 30).date(),
                provider_factory=FakeBoundary,
                decision_refresh=True,
            )
        refresh = result["decision_quote_refresh"]
        self.assertTrue(refresh["complete"])
        self.assertEqual("SCHWAB", refresh["source"])
        self.assertEqual("MARKET_OPEN_DECISION", refresh["purpose"])
        self.assertFalse(refresh["broker_order_surface"])

    def test_official_spy_workbook_yields_broad_unique_equity_universe(self):
        value = load_spy_holdings()
        tickers = [item["ticker"] for item in value["holdings"]]
        self.assertEqual(503, value["constituent_count"])
        self.assertEqual(len(tickers), len(set(tickers)))
        self.assertIn("NVDA", tickers)
        self.assertIn("PSKY", tickers)
        self.assertEqual("BRK/B", next(item for item in value["holdings"] if item["ticker"] == "BRK.B")["schwab_symbol"])

    def test_local_screen_preserves_budget_unresolved_and_rejections(self):
        def row(ticker, price, dollar_volume, spread, score_move=0.0):
            return {
                "ticker": ticker,
                "last": price,
                "dollar_volume": dollar_volume,
                "spread_fraction": spread,
                "net_percent_change": score_move,
                "week52_position": 0.9,
            }

        result = local_screen(
            (
                row("AAA", 100, 500_000_000, 0.001, 2.0),
                row("BBB", 100, 400_000_000, 0.001, 1.0),
                row("CCC", 5, 500_000_000, 0.001),
            ),
            orats_capacity=1,
        )
        self.assertEqual(["AAA"], [item["ticker"] for item in result["admitted"]])
        self.assertEqual("NOT_FULLY_EVALUATED_BUDGET", result["budget_unresolved"][0]["disposition"])
        self.assertEqual("LOCAL_SCREEN_REJECT", result["locally_rejected"][0]["disposition"])

    def test_local_screen_default_has_no_arbitrary_capacity(self):
        rows = tuple(
            {
                "ticker": "T%03d" % index,
                "last": 100.0,
                "dollar_volume": 100_000_000.0,
                "spread_fraction": 0.001,
                "net_percent_change": 0.0,
                "week52_position": 0.5,
            }
            for index in range(125)
        )
        result = local_screen(rows)
        self.assertEqual(125, len(result["admitted"]))
        self.assertEqual((), result["budget_unresolved"])

    def test_offline_rebuild_removes_saved_capacity_without_network(self):
        rows = [
            {
                "ticker": "AAA",
                "last": 100.0,
                "dollar_volume": 100_000_000.0,
                "spread_fraction": 0.001,
                "net_percent_change": 0.0,
                "week52_position": 0.5,
            },
            {
                "ticker": "BBB",
                "last": 101.0,
                "dollar_volume": 90_000_000.0,
                "spread_fraction": 0.001,
                "net_percent_change": 0.0,
                "week52_position": 0.5,
            },
        ]
        with TemporaryDirectory(dir=str(PROJECT_ROOT / "out")) as temporary:
            root = Path(temporary)
            source = root / "source.json"
            output = root / "output.json"
            source.write_text(
                __import__("json").dumps(
                    {
                        "schema": "cultra.broad-schwab-screen.v1",
                        "generated_at": "2026-08-30T00:00:00+00:00",
                        "universe": {"constituent_count": 2},
                        "quotes": rows,
                        "data_unavailable": [],
                        "admitted": [rows[0]],
                        "budget_unresolved": [rows[1]],
                    }
                ),
                encoding="utf-8",
            )
            rebuilt = rebuild_broad_screen_offline(
                source_path=source, output_path=output
            )
            self.assertEqual(2, rebuilt["counts"]["orats_admitted"])
            self.assertEqual(0, rebuilt["counts"]["budget_unresolved"])
            self.assertFalse(rebuilt["offline_rebuild"]["network_attempted"])
            self.assertEqual(["AAA", "BBB"], sorted(rebuilt["orats_admitted_symbols"]))


if __name__ == "__main__":
    unittest.main()
