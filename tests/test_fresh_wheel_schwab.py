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
    WheelAction,
    allocate_contracts,
    analyze_symbol,
    apply_calendar_seasonality,
    build_universe,
    find_export,
    iter_option_contracts,
    latest_usable_uw_folder,
    load_chain_oi_overlay,
    same_calendar_window_returns,
    sanitize_error,
    summarize_calendar_seasonality,
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
    @staticmethod
    def _seasonality_candles(returns_pct: list[float]) -> list[dict]:
        candles: list[dict] = []
        for index, return_pct in enumerate(returns_pct):
            year = 2025 - index
            for day, close in (
                (dt.date(year, 8, 20), 100.0),
                (dt.date(year, 9, 20), 100.0 * (1.0 + return_pct / 100.0)),
            ):
                timestamp = dt.datetime.combine(day, dt.time(20), tzinfo=dt.timezone.utc)
                candles.append({"datetime": int(timestamp.timestamp() * 1000), "close": close})
        return candles

    def test_calendar_seasonality_uses_same_prior_year_holding_window(self) -> None:
        candles = self._seasonality_candles([6.0, 5.0, 4.0, 3.0, 2.0, 2.0, 1.5, 1.0, -1.0, -2.0])
        config = WheelConfig(seasonality_min_years=5)

        returns = same_calendar_window_returns(
            candles,
            dt.date(2026, 8, 20),
            dt.date(2026, 9, 20),
            lookback_years=10,
        )
        evidence = summarize_calendar_seasonality(
            candles,
            dt.date(2026, 8, 20),
            dt.date(2026, 9, 20),
            config,
        )

        self.assertEqual(len(returns), 10)
        self.assertEqual(evidence.bias, "positive")
        self.assertEqual(evidence.positive_rate_pct, 80.0)
        self.assertGreater(evidence.directional_adjustment, 0.0)

    def test_calendar_seasonality_helps_csp_but_penalizes_call_away_risk(self) -> None:
        candles = self._seasonality_candles([6.0, 5.0, 4.0, 3.0, 2.0, 2.0, 1.5, 1.0, -1.0, -2.0])
        config = WheelConfig(seasonality_min_years=5)
        base = {
            "ticker": "AMZN",
            "confidence": 80.0,
            "spot": 200.0,
            "quality_score": 85.0,
            "flow_score": 60.0,
            "expiry": dt.date(2026, 9, 20),
        }
        csp = WheelAction(action="OPEN_CSP", **base)
        covered_call = WheelAction(action="SELL_COVERED_CALL", **base)

        apply_calendar_seasonality(csp, candles, dt.date(2026, 8, 20), config)
        apply_calendar_seasonality(covered_call, candles, dt.date(2026, 8, 20), config)

        self.assertGreater(csp.confidence, 80.0)
        self.assertGreater(csp.seasonality_confidence_adjustment, 0.0)
        self.assertLess(covered_call.confidence, 80.0)
        self.assertLess(covered_call.seasonality_confidence_adjustment, 0.0)
        self.assertIn("call-away risk", "; ".join(covered_call.reasons))

        first_confidence = csp.confidence
        apply_calendar_seasonality(csp, candles, dt.date(2026, 8, 20), config)
        self.assertEqual(csp.confidence, first_confidence)
        self.assertEqual(sum(reason.startswith("calendar seasonality ") for reason in csp.reasons), 1)

        csp.action = "SELL_COVERED_CALL"
        apply_calendar_seasonality(csp, candles, dt.date(2026, 8, 20), config)
        self.assertLess(csp.confidence, 80.0)
        self.assertIn("call-away risk", "; ".join(csp.reasons))

    def test_find_export_prefers_file_matching_folder_date(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "2026-07-15"
            base.mkdir()
            older = base / "stock-screener-2026-07-14.zip"
            current = base / "stock-screener-2026-07-15.zip"
            older.touch()
            current.touch()

            self.assertEqual(find_export(base, "stock-screener-"), current)

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
        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
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
        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
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

    def test_covered_strangle_cash_fallback_clears_abandoned_put_leg(self) -> None:
        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
            action, _ = analyze_symbol(
                row=_universe_row(),
                service=FakeSchwabService(),
                quote={},
                position=Position(symbol="AMZN", shares=200, avg_cost=240.0),
                asof=dt.date(2026, 4, 30),
                config=WheelConfig(account_size=100_000, min_option_volume=1),
                out_dir=Path(tempfile.gettempdir()),
            )
            allocate_contracts([action], WheelConfig(account_size=100_000))

        self.assertEqual(action.action, "SELL_COVERED_CALL")
        self.assertEqual(action.option_symbol, "AMZN  260618C00310000")
        self.assertEqual(action.paired_option_symbol, "")
        self.assertIsNone(action.paired_expiry)
        self.assertIsNone(action.paired_strike)
        self.assertIsNone(action.paired_limit_price)
        self.assertNotIn("covered strangle:", "; ".join(action.reasons))

    def test_analyze_symbol_can_pair_csp_with_budgeted_upside_call(self) -> None:
        chain = _chain(call_strike=340.0)
        chain["callExpDateMap"]["2026-06-18:49"]["340.0"][0].update(
            {"symbol": "AMZN  260618C00340000", "bid": 0.9, "ask": 1.0, "mark": 0.95, "delta": 0.16}
        )
        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
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

        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
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
        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
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
        with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
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
                "bullish_premium,bearish_premium,next_earnings_date,iv30d\n"
                "2026-04-30,AMZN,Amazon.com Inc,Consumer Cyclical,Common Stock,f,280,1800000000000,20000000,2000000,1000000,500000,2026-07-30,0.40\n"
                "2026-04-30,JUNK,Junk Inc,Other,Common Stock,f,5,1000000,1000,10,0,0,,0.40\n"
            )
            with zipfile.ZipFile(base / "stock-screener-2026-04-30.zip", "w") as zf:
                zf.writestr("stock-screener-2026-04-30.csv", screener_csv)
            with zipfile.ZipFile(base / "hot-chains-2026-04-30.zip", "w") as zf:
                zf.writestr("hot-chains-2026-04-30.csv", "option_symbol,volume,open_interest,premium,ask_side_volume,bid_side_volume\nAMZN260618P00255000,100,500,100000,80,20\n")

            universe = build_universe(base, WheelConfig(max_symbols=5))

        self.assertEqual([row.ticker for row in universe], ["AMZN"])
        self.assertGreater(universe[0].quality_score, 80.0)

    def test_chain_oi_overlay_validates_date_and_influences_flow_score(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "2026-08-14"
            base.mkdir()
            screener_csv = (
                "date,ticker,full_name,sector,issue_type,is_index,close,marketcap,avg30_volume,total_open_interest,"
                "bullish_premium,bearish_premium,next_earnings_date,iv30d\n"
                "2026-08-14,IBM,International Business Machines,Technology,Common Stock,f,250,250000000000,"
                "5000000,500000,1000000,1000000,2026-10-21,0.40\n"
            )
            with zipfile.ZipFile(base / "stock-screener-2026-08-14.zip", "w") as zf:
                zf.writestr("stock-screener-2026-08-14.csv", screener_csv)
            with zipfile.ZipFile(base / "hot-chains-2026-08-14.zip", "w") as zf:
                zf.writestr(
                    "hot-chains-2026-08-14.csv",
                    "option_symbol,volume,open_interest,premium,ask_side_volume,bid_side_volume\n",
                )
            overlay_path = Path(tmpdir) / "chain-oi-changes-2026-08-17.csv"
            overlay_path.write_text(
                "option_symbol,underlying_symbol,oi_diff_plain,curr_oi,volume,last_date,curr_date,"
                "prev_total_premium,prev_ask_volume,prev_bid_volume,avg_price\n"
                "IBM261218C00300000,IBM,50000,60000,50000,2026-08-14,2026-08-17,1000000,50000,0,0.20\n",
                encoding="utf-8",
            )

            overlay, metadata = load_chain_oi_overlay(overlay_path, dt.date(2026, 8, 14))
            universe = build_universe(base, WheelConfig(max_symbols=1), chain_oi_overlay=overlay)

            self.assertEqual(metadata["base_dates"], ["2026-08-14"])
            self.assertEqual(metadata["overlay_dates"], ["2026-08-17"])
            self.assertEqual(universe[0].overlay_positive_oi_change, 50000.0)
            self.assertEqual(universe[0].overlay_directional_bias, 1.0)
            self.assertGreater(universe[0].flow_score, 50.0)
            with self.assertRaisesRegex(ValueError, "does not compare from wheel base date"):
                load_chain_oi_overlay(overlay_path, dt.date(2026, 8, 13))

    def test_build_universe_uses_objective_lanes_for_high_premium_ibm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "2026-07-14"
            base.mkdir()
            screener_csv = (
                "date,ticker,full_name,sector,issue_type,is_index,close,marketcap,avg30_volume,total_open_interest,"
                "bullish_premium,bearish_premium,next_earnings_date,iv30d\n"
                "2026-07-14,QUALITY,Quality Inc,Technology,Common Stock,f,300,1500000000000,25000000,3000000,60000000,40000000,2026-09-01,0.35\n"
                "2026-07-14,TACT,Tactical Inc,Technology,Common Stock,f,100,50000000000,15000000,800000,900000000,100000000,2026-09-01,1.20\n"
                "2026-07-14,IBM,International Business Machines,Technology,Common Stock,f,216,272000000000,8000000,565000,197000000,253000000,2026-07-22,0.59\n"
            )
            with zipfile.ZipFile(base / "stock-screener-2026-07-14.zip", "w") as zf:
                zf.writestr("stock-screener-2026-07-14.csv", screener_csv)
            with zipfile.ZipFile(base / "hot-chains-2026-07-14.zip", "w") as zf:
                zf.writestr(
                    "hot-chains-2026-07-14.csv",
                    "option_symbol,volume,open_interest,premium,ask_side_volume,bid_side_volume\n",
                )

            universe = build_universe(base, WheelConfig(max_symbols=3))

        self.assertEqual([row.ticker for row in universe], ["QUALITY", "TACT", "IBM"])
        ibm = universe[-1]
        self.assertEqual(ibm.selection_lane, "premium")
        self.assertLess(ibm.flow_score, WheelConfig().tactical_min_flow_score)
        self.assertNotIn("curated", "; ".join(ibm.reasons).lower())

    def test_build_universe_appends_held_round_lot_outside_candidate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "2026-07-14"
            base.mkdir()
            screener_csv = (
                "date,ticker,full_name,sector,issue_type,is_index,close,marketcap,avg30_volume,total_open_interest,"
                "bullish_premium,bearish_premium,next_earnings_date,iv30d\n"
                "2026-07-14,TOP,Top Inc,Technology,Common Stock,f,300,1500000000000,25000000,3000000,60000000,40000000,2026-09-01,0.35\n"
                "2026-07-14,HELD,Held Inc,Technology,Common Stock,f,10,1000000000,500000,10000,100000,100000,2026-09-01,0.50\n"
            )
            with zipfile.ZipFile(base / "stock-screener-2026-07-14.zip", "w") as zf:
                zf.writestr("stock-screener-2026-07-14.csv", screener_csv)
            with zipfile.ZipFile(base / "hot-chains-2026-07-14.zip", "w") as zf:
                zf.writestr(
                    "hot-chains-2026-07-14.csv",
                    "option_symbol,volume,open_interest,premium,ask_side_volume,bid_side_volume\n",
                )

            universe = build_universe(base, WheelConfig(max_symbols=1), position_symbols={"HELD"})

        self.assertEqual([row.ticker for row in universe], ["TOP", "HELD"])
        self.assertEqual(universe[-1].selection_lane, "position")
        self.assertIn("outside candidate limit", "; ".join(universe[-1].reasons))

    def test_build_universe_appends_user_requested_symbol_outside_candidate_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "2026-07-16"
            base.mkdir()
            screener_csv = (
                "date,ticker,full_name,sector,issue_type,is_index,close,marketcap,avg30_volume,total_open_interest,"
                "bullish_premium,bearish_premium,next_earnings_date,iv30d\n"
                "2026-07-16,TOP,Top Inc,Technology,Common Stock,f,300,1500000000000,25000000,3000000,60000000,40000000,2026-09-01,0.35\n"
                "2026-07-16,IBM,International Business Machines,Technology,Common Stock,f,215,272000000000,8000000,565000,1000000,1000000,2026-07-22,0.59\n"
            )
            with zipfile.ZipFile(base / "stock-screener-2026-07-16.zip", "w") as zf:
                zf.writestr("stock-screener-2026-07-16.csv", screener_csv)
            with zipfile.ZipFile(base / "hot-chains-2026-07-16.zip", "w") as zf:
                zf.writestr(
                    "hot-chains-2026-07-16.csv",
                    "option_symbol,volume,open_interest,premium,ask_side_volume,bid_side_volume\n",
                )

            universe = build_universe(base, WheelConfig(max_symbols=1), include_symbols={"IBM"})

        self.assertEqual([row.ticker for row in universe], ["TOP", "IBM"])
        self.assertEqual(universe[-1].selection_lane, "requested")
        self.assertIn("user-requested", "; ".join(universe[-1].reasons))

    def test_write_outputs_manifest_records_schwab_only_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            with patch("uwos.fresh_wheel_schwab._today", return_value=dt.date(2026, 5, 3)):
                action, _ = analyze_symbol(
                    row=_universe_row(),
                    service=FakeSchwabService(),
                    quote={},
                    position=None,
                    asof=dt.date(2026, 4, 30),
                    config=WheelConfig(account_size=250_000, min_option_volume=1),
                    out_dir=out,
                )
                allocate_contracts([action], WheelConfig(account_size=250_000))
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
            tickets = outputs["tickets"].read_text(encoding="utf-8")

        self.assertIn('"live_source": "Schwab API"', manifest)
        self.assertIn('"pipeline_version": "fresh-wheel-v1.2-calendar-seasonality-20260819"', manifest)
        self.assertIn('"yahoo_yfinance_used": false', manifest)
        self.assertIn('"calendar_seasonality"', manifest)
        self.assertIn("fresh-wheel-v1.2-calendar-seasonality-20260819", report)
        self.assertIn("## Calendar Seasonality", report)
        self.assertIn("## Weekly Focus", report)
        self.assertIn("## Action Board", report)
        self.assertIn("| Status | Ticker | Type | Exp | Strike | Trade / Trigger |", report)
        self.assertIn("Sell Jun 18, 2026 $255 put", report)
        self.assertIn("$5.00+ credit", report)
        self.assertNotIn("260618P00255000", report)
        self.assertRegex(report, "🟢 STRONG|🔵 SECONDARY|🟡 ALERT|🟠 WAIT|🔴 AVOID")
        self.assertIn("# Wheel Execution Tickets", tickets)
        self.assertIn("Entry: Sell Jun 18, 2026 $255 put", tickets)
        self.assertIn("Exit/OCO: Not generated", tickets)
        self.assertNotIn("260618P00255000", tickets)


if __name__ == "__main__":
    unittest.main()
