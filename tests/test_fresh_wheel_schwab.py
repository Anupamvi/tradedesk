from __future__ import annotations

import csv
import datetime as dt
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from uwos.fresh_wheel_schwab import (
    Position,
    UniverseRow,
    WheelConfig,
    allocate_contracts,
    analyze_symbol,
    build_universe,
    iter_option_contracts,
    latest_usable_uw_folder,
    sanitize_error,
    write_outputs,
)


def _contract(
    *,
    symbol: str,
    strike: float,
    right: str,
    expiry: str = "2026-06-18",
    bid: float = 5.0,
    ask: float = 5.4,
    delta: float = -0.2,
    open_interest: int = 800,
    volume: int = 50,
) -> dict:
    return {
        "symbol": symbol,
        "strikePrice": strike,
        "bid": bid,
        "ask": ask,
        "mark": (bid + ask) / 2.0,
        "delta": delta,
        "volatility": 32.0,
        "openInterest": open_interest,
        "totalVolume": volume,
        "putCall": "PUT" if right == "P" else "CALL",
    }


def _chain(root: str = "AMZN", spot: float = 280.0, put_strike: float = 255.0, call_strike: float = 310.0) -> dict:
    return {
        "underlyingPrice": spot,
        "putExpDateMap": {
            "2026-06-18:49": {
                f"{put_strike:.1f}": [
                    _contract(
                        symbol=f"{root:<6}260618P{int(put_strike * 1000):08d}",
                        strike=put_strike,
                        right="P",
                        delta=-0.22,
                    )
                ]
            }
        },
        "callExpDateMap": {
            "2026-06-18:49": {
                f"{call_strike:.1f}": [
                    _contract(
                        symbol=f"{root:<6}260618C{int(call_strike * 1000):08d}",
                        strike=call_strike,
                        right="C",
                        bid=2.1,
                        ask=2.25,
                        delta=0.20,
                    )
                ]
            },
            "2026-11-20:204": {
                "220.0": [_contract(symbol=f"{root:<6}261120C00220000", strike=220.0, right="C", bid=68.0, ask=69.0, delta=0.78)]
            },
        },
    }


class FakeSchwabService:
    def __init__(self, chain: dict | None = None) -> None:
        self.chain = chain or _chain()
        self.requested_symbols: list[str] = []
        self.last_kwargs: dict = {}

    def get_option_chain(self, symbol: str, **kwargs):
        self.requested_symbols.append(symbol)
        self.last_kwargs = kwargs
        return self.chain


def _universe_row(**overrides) -> UniverseRow:
    base = {
        "ticker": "AMZN",
        "full_name": "Amazon.com Inc",
        "sector": "Consumer Cyclical",
        "issue_type": "Common Stock",
        "close": 280.0,
        "marketcap": 1_800_000_000_000.0,
        "avg30_volume": 20_000_000.0,
        "total_open_interest": 2_000_000.0,
        "next_earnings": dt.date(2026, 7, 30),
        "quality_score": 92.0,
        "flow_score": 62.0,
        "thesis": "AWS plus commerce scale",
        "tier": 1,
        "reasons": ["test"],
    }
    base.update(overrides)
    return UniverseRow(**base)


class FreshWheelSchwabTests(unittest.TestCase):
    def test_latest_usable_uw_folder_skips_empty_newer_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            empty = root / "2026-05-01"
            usable = root / "2026-04-30"
            empty.mkdir()
            usable.mkdir()
            (usable / "stock-screener-2026-04-30.csv").write_text("ticker,close\nAMZN,280\n", encoding="utf-8")
            (usable / "hot-chains-2026-04-30.csv").write_text("option_symbol,volume\nAMZN260618P00255000,100\n", encoding="utf-8")

            self.assertEqual(latest_usable_uw_folder(root), usable)

    def test_iter_option_contracts_reads_schwab_chain_shape(self) -> None:
        contracts = list(iter_option_contracts(_chain(), "P", dt.date(2026, 4, 30), 0.04, 280.0))

        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0].symbol, "AMZN  260618P00255000")
        self.assertEqual(contracts[0].expiry, dt.date(2026, 6, 18))
        self.assertAlmostEqual(contracts[0].mid, 5.2)

    def test_analyze_symbol_uses_schwab_chain_for_csp_alert_or_entry(self) -> None:
        service = FakeSchwabService()
        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
            action, _ = analyze_symbol(
                row=_universe_row(),
                service=service,
                quote={},
                position=None,
                asof=dt.date(2026, 4, 30),
                config=WheelConfig(account_size=250_000, min_option_volume=1),
                out_dir=Path(tempfile.gettempdir()),
            )

        self.assertEqual(service.requested_symbols, ["AMZN"])
        self.assertEqual(service.last_kwargs["from_date"], dt.date(2026, 5, 3))
        self.assertIn(action.action, {"OPEN_CSP", "SET_CSP_ALERT"})
        self.assertEqual(action.option_symbol, "AMZN  260618P00255000")
        self.assertEqual(action.limit_price, 5.0)

    def test_sanitize_error_redacts_schwab_api_key(self) -> None:
        text = sanitize_error("https://api.schwabapi.com/marketdata/v1/chains?apikey=SECRET123&symbol=MSFT")

        self.assertIn("apikey=REDACTED", text)
        self.assertNotIn("SECRET123", text)

    def test_analyze_symbol_prefers_covered_call_when_schwab_position_has_shares(self) -> None:
        action, _ = analyze_symbol(
            row=_universe_row(),
            service=FakeSchwabService(),
            quote={},
            position=Position(symbol="AMZN", shares=200, avg_cost=240.0),
            asof=dt.date(2026, 4, 30),
            config=WheelConfig(account_size=250_000, min_option_volume=1, enable_covered_strangles=False),
            out_dir=Path(tempfile.gettempdir()),
        )

        self.assertEqual(action.action, "SELL_COVERED_CALL")
        self.assertEqual(action.contracts, 2)
        self.assertEqual(action.option_symbol, "AMZN  260618C00310000")

    def test_analyze_symbol_prefers_covered_strangle_when_shares_and_put_budget_exist(self) -> None:
        action, _ = analyze_symbol(
            row=_universe_row(),
            service=FakeSchwabService(),
            quote={},
            position=Position(symbol="AMZN", shares=200, avg_cost=240.0),
            asof=dt.date(2026, 4, 30),
            config=WheelConfig(account_size=250_000, min_option_volume=1),
            out_dir=Path(tempfile.gettempdir()),
        )

        self.assertEqual(action.action, "SELL_COVERED_STRANGLE")
        self.assertEqual(action.option_symbol, "AMZN  260618C00310000")
        self.assertEqual(action.paired_option_symbol, "AMZN  260618P00255000")
        self.assertEqual(action.paired_strike, 255.0)

    def test_analyze_symbol_can_pair_csp_with_budgeted_upside_call(self) -> None:
        chain = _chain(call_strike=340.0)
        chain["callExpDateMap"]["2026-06-18:49"]["340.0"][0].update(
            {"symbol": "AMZN  260618C00340000", "bid": 0.9, "ask": 1.0, "mark": 0.95, "delta": 0.16}
        )
        action, _ = analyze_symbol(
            row=_universe_row(flow_score=70.0, next_earnings=dt.date(2027, 1, 30)),
            service=FakeSchwabService(chain),
            quote={},
            position=None,
            asof=dt.date(2026, 4, 30),
            config=WheelConfig(account_size=250_000, min_option_volume=1),
            out_dir=Path(tempfile.gettempdir()),
        )

        self.assertEqual(action.action, "OPEN_CSP_WITH_CALL_OVERLAY")
        self.assertEqual(action.long_option_symbol, "AMZN  260618C00340000")
        self.assertEqual(action.long_limit_price, 1.0)

    def test_analyze_symbol_can_build_leaps_covered_strangle(self) -> None:
        chain = _chain()
        chain["callExpDateMap"]["2026-06-18:49"]["310.0"][0].update(
            {"bid": 7.0, "ask": 7.4, "mark": 7.2, "delta": 0.20}
        )
        chain["callExpDateMap"]["2026-11-20:204"]["220.0"][0].update(
            {"bid": 65.0, "ask": 65.2, "mark": 65.1, "delta": 0.78}
        )

        action, _ = analyze_symbol(
            row=_universe_row(flow_score=70.0, next_earnings=dt.date(2027, 1, 30)),
            service=FakeSchwabService(chain),
            quote={},
            position=None,
            asof=dt.date(2026, 4, 30),
            config=WheelConfig(account_size=250_000, min_option_volume=1),
            out_dir=Path(tempfile.gettempdir()),
        )

        self.assertEqual(action.action, "OPEN_LEAPS_COVERED_STRANGLE")
        self.assertEqual(action.long_option_symbol, "AMZN  261120C00220000")
        self.assertEqual(action.option_symbol, "AMZN  260618C00310000")
        self.assertEqual(action.paired_option_symbol, "AMZN  260618P00255000")

    def test_allocate_contracts_respects_account_size(self) -> None:
        action, _ = analyze_symbol(
            row=_universe_row(),
            service=FakeSchwabService(),
            quote={},
            position=None,
            asof=dt.date(2026, 4, 30),
            config=WheelConfig(account_size=100_000, min_option_volume=1),
            out_dir=Path(tempfile.gettempdir()),
        )
        action.action = "OPEN_CSP"

        allocate_contracts([action], WheelConfig(account_size=100_000))

        self.assertEqual(action.contracts, 0)
        self.assertEqual(action.action, "SET_CSP_ALERT")
        self.assertIn("cash-buffer", "; ".join(action.blockers))

    def test_tactical_sleeve_promotes_high_premium_non_core_csp(self) -> None:
        chain = _chain(root="MRVL", spot=165.0, put_strike=150.0, call_strike=190.0)
        chain["putExpDateMap"]["2026-06-18:49"]["150.0"][0].update(
            {"bid": 7.8, "ask": 8.0, "mark": 7.9, "delta": -0.22}
        )
        row = _universe_row(
            ticker="MRVL",
            full_name="Marvell Technology Inc",
            close=165.0,
            marketcap=70_000_000_000.0,
            avg30_volume=12_000_000.0,
            total_open_interest=700_000.0,
            next_earnings=dt.date(2026, 8, 30),
            quality_score=45.0,
            flow_score=58.0,
            thesis="tactical premium/flow candidate",
            tier=4,
            total_premium=800_000_000.0,
            iv30d=0.55,
            tactical_score=76.0,
        )
        config = WheelConfig(
            account_size=400_000,
            min_option_volume=1,
            enable_csp_call_overlay=False,
            enable_leaps_covered_strangles=False,
        )

        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
            action, _ = analyze_symbol(
                row=row,
                service=FakeSchwabService(chain),
                quote={},
                position=None,
                asof=dt.date(2026, 5, 1),
                config=config,
                out_dir=Path(tempfile.gettempdir()),
            )
            allocate_contracts([action], config)

        self.assertEqual(action.action, "OPEN_TACTICAL_CSP")
        self.assertEqual(action.sleeve, "tactical")
        self.assertEqual(action.contracts, 1)
        self.assertEqual(action.cash_required, 15_000.0)
        self.assertEqual(action.estimated_credit, 780.0)

    def test_analyze_symbol_blocks_replay_tail_loss_csp_names(self) -> None:
        action, _ = analyze_symbol(
            row=_universe_row(ticker="ORCL", full_name="Oracle Corp", close=180.0, quality_score=92.0),
            service=FakeSchwabService(_chain(root="ORCL", spot=180.0, put_strike=160.0, call_strike=200.0)),
            quote={},
            position=None,
            asof=dt.date(2026, 4, 30),
            config=WheelConfig(account_size=250_000, min_option_volume=1),
            out_dir=Path(tempfile.gettempdir()),
        )

        self.assertEqual(action.action, "WATCH_ONLY")
        self.assertIn("replay block", "; ".join(action.blockers))

    def test_build_universe_from_uw_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "2026-04-30"
            base.mkdir()
            screener_csv = (
                "date,ticker,full_name,sector,issue_type,is_index,close,marketcap,avg30_volume,total_open_interest,"
                "bullish_premium,bearish_premium,next_earnings_date\n"
                "2026-04-30,AMZN,Amazon.com Inc,Consumer Cyclical,Common Stock,f,280,1800000000000,20000000,2000000,1000000,500000,2026-07-30\n"
                "2026-04-30,JUNK,Junk Inc,Other,Common Stock,f,5,1000000,1000,10,0,0,\n"
            )
            with zipfile.ZipFile(base / "stock-screener-2026-04-30.zip", "w") as zf:
                zf.writestr("stock-screener-2026-04-30.csv", screener_csv)
            with zipfile.ZipFile(base / "hot-chains-2026-04-30.zip", "w") as zf:
                zf.writestr("hot-chains-2026-04-30.csv", "option_symbol,volume,open_interest,premium,ask_side_volume,bid_side_volume\nAMZN260618P00255000,100,500,100000,80,20\n")

            universe = build_universe(base, WheelConfig(max_symbols=5))

        self.assertEqual([row.ticker for row in universe], ["AMZN"])
        self.assertGreater(universe[0].quality_score, 80.0)

    def test_write_outputs_manifest_records_schwab_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            action, _ = analyze_symbol(
                row=_universe_row(),
                service=FakeSchwabService(),
                quote={},
                position=None,
                asof=dt.date(2026, 4, 30),
                config=WheelConfig(account_size=250_000, min_option_volume=1),
                out_dir=out,
            )
            outputs = write_outputs(
                out,
                dt.date(2026, 4, 30),
                Path("/tmp/2026-04-30"),
                [_universe_row()],
                [action],
                "skipped",
                {},
                WheelConfig(),
            )
            manifest = outputs["manifest"].read_text(encoding="utf-8")
            report = outputs["report"].read_text(encoding="utf-8")

        self.assertIn('"live_source": "Schwab API"', manifest)
        self.assertIn('"yahoo_yfinance_used": false', manifest)
        self.assertIn("## Weekly Focus", report)
        self.assertIn("## Action Board", report)
        self.assertIn("| Status | Ticker | Type | Exp | Strike | Trade / Trigger |", report)
        self.assertRegex(report, "🟢 STRONG|🔵 SECONDARY|🟡 ALERT|🟠 WAIT|🔴 AVOID")


if __name__ == "__main__":
    unittest.main()
