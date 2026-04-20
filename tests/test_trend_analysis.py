import argparse
import datetime as dt
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from uwos import swing_trend_pipeline
from uwos import trend_analysis


class TestTrendAnalysisWrapper(unittest.TestCase):
    def test_resolve_invocation_accepts_date_then_lookback(self) -> None:
        args = argparse.Namespace(
            as_of="",
            lookback=None,
            tokens=["2026-04-17", "90"],
        )
        as_of, lookback = trend_analysis.resolve_invocation(args)
        self.assertEqual(as_of, dt.date(2026, 4, 17))
        self.assertEqual(lookback, 90)

    def test_resolve_invocation_rolls_weekend_back(self) -> None:
        args = argparse.Namespace(
            as_of="",
            lookback=None,
            tokens=["2026-04-18"],
        )
        as_of, lookback = trend_analysis.resolve_invocation(args)
        self.assertEqual(as_of, dt.date(2026, 4, 17))
        self.assertEqual(lookback, trend_analysis.DEFAULT_LOOKBACK)

    def test_discovery_counts_usable_market_days_not_dated_folders(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for day in (
                "2026-03-31",
                "2026-04-01",
                "2026-04-02",
                "2026-04-03",
                "2026-04-04",
                "2026-04-05",
                "2026-04-06",
            ):
                day_dir = root / day
                day_dir.mkdir()
                if day != "2026-04-01":
                    (day_dir / f"stock-screener-{day}.csv").write_text(
                        "ticker,close\nAAPL,100\n",
                        encoding="utf-8",
                    )

            days = swing_trend_pipeline.discover_trading_days(
                root,
                3,
                dt.date(2026, 4, 6),
            )

        self.assertEqual(
            [d.isoformat() for d, _ in days],
            ["2026-04-02", "2026-04-03", "2026-04-06"],
        )

    def test_trade_repair_generates_pre_earnings_variant(self) -> None:
        score = swing_trend_pipeline.SwingScore(
            ticker="SOFI",
            recommended_strategy="Bear Put Debit",
            target_expiry="2026-05-22",
            target_dte=50,
            long_strike=15.0,
            short_strike=12.5,
            spread_width=2.5,
            strike_setup="Buy 15P / Sell 12.5P",
            cost_type="debit",
            est_cost=0.13,
            direction="bearish",
        )
        signals = swing_trend_pipeline.SwingSignals(
            ticker="SOFI",
            latest_date=dt.date(2026, 4, 2),
            latest_close=14.7,
            next_earnings_date=dt.date(2026, 4, 29),
            price_direction="bearish",
            flow_direction="bearish",
        )

        variants = swing_trend_pipeline.generate_trade_repair_variants(
            [score],
            {"SOFI": signals},
            {"filters": {"earnings_buffer_days": 3}},
        )

        pre_earn = [v for v in variants if v.variant_tag == "pre_earnings"]
        self.assertEqual(len(pre_earn), 1)
        self.assertEqual(pre_earn[0].target_expiry, "2026-04-24")
        self.assertTrue(pre_earn[0].earnings_safe)
        self.assertEqual(pre_earn[0].repair_source, "expiry moved before earnings window")

    def test_earnings_safety_allows_expiry_on_buffer_boundary(self) -> None:
        score = swing_trend_pipeline.SwingScore(
            ticker="PBR",
            target_expiry="2026-05-08",
        )
        signals = swing_trend_pipeline.SwingSignals(
            ticker="PBR",
            latest_date=dt.date(2026, 4, 2),
            next_earnings_date=dt.date(2026, 5, 11),
        )

        swing_trend_pipeline._check_earnings_safety(
            signals,
            score,
            {"filters": {"earnings_buffer_days": 3}},
        )

        self.assertTrue(score.earnings_safe)
        self.assertEqual(score.earnings_label, "")

    def test_split_actionable_requires_backtest_pass(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "PASS",
                    "swing_score": 70,
                    "backtest_verdict": "PASS",
                    "edge_pct": 6.5,
                    "backtest_signals": 150,
                },
                {
                    "ticker": "FAIL",
                    "swing_score": 90,
                    "backtest_verdict": "FAIL",
                    "edge_pct": 12.0,
                    "backtest_signals": 200,
                },
                {
                    "ticker": "THIN",
                    "swing_score": 80,
                    "backtest_verdict": "PASS",
                    "edge_pct": 5.0,
                    "backtest_signals": 30,
                },
            ]
        )
        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=False,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
        )
        self.assertEqual(list(actionable["ticker"]), ["PASS"])
        self.assertEqual(set(patterns["ticker"]), {"FAIL", "THIN"})

    def test_split_actionable_requires_min_swing_score_when_configured(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "STRONG",
                    "swing_score": 62,
                    "backtest_verdict": "PASS",
                    "edge_pct": 5.0,
                    "backtest_signals": 150,
                },
                {
                    "ticker": "WEAK",
                    "swing_score": 54,
                    "backtest_verdict": "PASS",
                    "edge_pct": 20.0,
                    "backtest_signals": 200,
                },
            ]
        )
        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=False,
            min_edge=0.0,
            min_signals=100,
            min_swing_score=60.0,
            allow_low_sample=False,
        )
        self.assertEqual(list(actionable["ticker"]), ["STRONG"])
        self.assertEqual(list(patterns["ticker"]), ["WEAK"])
        self.assertIn("swing score", patterns.iloc[0]["base_gate_reasons"])

    def test_split_actionable_dedupes_equivalent_live_structures(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "BABA",
                    "strategy": "Bull Call Debit",
                    "variant_tag": "pre_earnings",
                    "target_expiry": "2026-05-15",
                    "live_strike_setup": "Buy 140C / Sell 150C",
                    "strike_setup": "Buy 140C / Sell 150C",
                    "swing_score": 70,
                    "backtest_verdict": "PASS",
                    "edge_pct": 2.0,
                    "backtest_signals": 190,
                },
                {
                    "ticker": "BABA",
                    "strategy": "Bull Call Debit",
                    "variant_tag": "listed_expiry",
                    "target_expiry": "2026-05-15",
                    "live_strike_setup": "Buy 140C / Sell 150C",
                    "strike_setup": "Buy 140C / Sell 150C",
                    "swing_score": 70,
                    "backtest_verdict": "PASS",
                    "edge_pct": 2.0,
                    "backtest_signals": 190,
                },
            ]
        )
        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=False,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
        )
        self.assertEqual(len(actionable), 1)
        self.assertTrue(patterns.empty)

    def test_candidate_shortlist_allows_labeled_price_divergence(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "GOOD",
                    "variant_tag": "base",
                    "direction": "bullish",
                    "swing_score": 66,
                    "days_observed": 30,
                    "price_direction": "bullish",
                    "price_trend": 80,
                    "flow_direction": "bullish",
                    "flow_persistence": 70,
                    "oi_direction": "bullish",
                    "oi_momentum": 65,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 72,
                    "whale_appearances": 8,
                    "base_gate_pass": False,
                    "backtest_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quality_gate_pass": False,
                    "edge_pct": 8.0,
                    "backtest_signals": 120,
                    "backtest_verdict": "PASS",
                },
                {
                    "ticker": "DIVERGE",
                    "variant_tag": "base",
                    "direction": "bullish",
                    "swing_score": 80,
                    "days_observed": 30,
                    "price_direction": "bearish",
                    "price_trend": 80,
                    "flow_direction": "bullish",
                    "flow_persistence": 80,
                    "oi_direction": "bullish",
                    "oi_momentum": 80,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 80,
                    "whale_appearances": 20,
                    "backtest_gate_pass": True,
                    "schwab_gate_pass": False,
                    "quality_gate_pass": False,
                    "edge_pct": 11.0,
                    "backtest_signals": 150,
                    "backtest_verdict": "PASS",
                },
                {
                    "ticker": "TOO_THIN",
                    "variant_tag": "base",
                    "direction": "bullish",
                    "swing_score": 80,
                    "days_observed": 30,
                    "price_direction": "bearish",
                    "price_trend": 80,
                    "flow_direction": "bullish",
                    "flow_persistence": 80,
                    "oi_direction": "bullish",
                    "oi_momentum": 80,
                    "backtest_gate_pass": False,
                    "schwab_gate_pass": True,
                    "quality_gate_pass": False,
                    "edge_pct": -12.0,
                    "backtest_signals": 150,
                    "backtest_verdict": "FAIL",
                },
                {
                    "ticker": "CONFLICT",
                    "variant_tag": "base",
                    "direction": "bearish",
                    "swing_score": 70,
                    "days_observed": 30,
                    "price_direction": "bearish",
                    "price_trend": 80,
                    "flow_direction": "bearish",
                    "flow_persistence": 80,
                    "oi_direction": "bullish",
                    "oi_momentum": 80,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 80,
                    "whale_appearances": 20,
                    "backtest_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quality_gate_pass": False,
                    "edge_pct": 8.0,
                    "backtest_signals": 120,
                    "backtest_verdict": "PASS",
                },
            ]
        )

        shortlist = trend_analysis.build_candidate_shortlist(candidates, top_n=5)

        self.assertEqual(set(shortlist["ticker"]), {"GOOD", "DIVERGE"})
        diverge = shortlist[shortlist["ticker"].eq("DIVERGE")].iloc[0]
        self.assertIn("price divergence", diverge["candidate_conflicts"])
        self.assertIn("Treat as divergence", diverge["candidate_next_step"])
        good = shortlist[shortlist["ticker"].eq("GOOD")].iloc[0]
        self.assertIn("price trend bullish", good["candidate_confirmations"])

    def test_no_backtest_means_patterns_only(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "swing_score": 70,
                    "backtest_verdict": "PASS",
                    "edge_pct": 6.5,
                    "backtest_signals": 150,
                }
            ]
        )
        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=False,
            schwab_enabled=False,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
        )
        self.assertTrue(actionable.empty)
        self.assertEqual(list(patterns["ticker"]), ["AAPL"])

    def test_schwab_enabled_requires_live_validation(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "LIVE",
                    "strategy": "Bull Put Credit",
                    "swing_score": 70,
                    "backtest_verdict": "PASS",
                    "edge_pct": 6.5,
                    "backtest_signals": 150,
                    "live_validated": True,
                    "earnings_safe": True,
                    "live_bid_ask_width": 0.10,
                    "live_spread_cost": 1.00,
                    "spread_width": 5.00,
                    "short_delta_live": -0.20,
                },
                {
                    "ticker": "STALE",
                    "strategy": "Bull Put Credit",
                    "swing_score": 80,
                    "backtest_verdict": "PASS",
                    "edge_pct": 9.0,
                    "backtest_signals": 200,
                    "live_validated": False,
                    "earnings_safe": True,
                },
            ]
        )
        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=True,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
        )
        self.assertEqual(list(actionable["ticker"]), ["LIVE"])
        self.assertEqual(list(patterns["ticker"]), ["STALE"])

    def test_tradeability_gates_block_earnings_liquidity_and_delta(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "OK",
                    "strategy": "Bull Put Credit",
                    "swing_score": 70,
                    "backtest_verdict": "PASS",
                    "edge_pct": 6.5,
                    "backtest_signals": 150,
                    "live_validated": True,
                    "earnings_safe": True,
                    "live_bid_ask_width": 0.10,
                    "live_spread_cost": 1.00,
                    "spread_width": 5.00,
                    "short_delta_live": -0.20,
                },
                {
                    "ticker": "EARN",
                    "strategy": "Bull Put Credit",
                    "swing_score": 90,
                    "backtest_verdict": "PASS",
                    "edge_pct": 12.0,
                    "backtest_signals": 200,
                    "live_validated": True,
                    "earnings_safe": False,
                    "earnings_label": "EARNINGS 2026-04-29",
                    "live_bid_ask_width": 0.10,
                    "live_spread_cost": 1.00,
                    "spread_width": 5.00,
                    "short_delta_live": -0.20,
                },
                {
                    "ticker": "WIDE",
                    "strategy": "Bull Put Credit",
                    "swing_score": 85,
                    "backtest_verdict": "PASS",
                    "edge_pct": 10.0,
                    "backtest_signals": 200,
                    "live_validated": True,
                    "earnings_safe": True,
                    "live_bid_ask_width": 0.60,
                    "live_spread_cost": 1.00,
                    "spread_width": 5.00,
                    "short_delta_live": -0.20,
                },
                {
                    "ticker": "DELTA",
                    "strategy": "Bull Put Credit",
                    "swing_score": 80,
                    "backtest_verdict": "PASS",
                    "edge_pct": 9.0,
                    "backtest_signals": 200,
                    "live_validated": True,
                    "earnings_safe": True,
                    "live_bid_ask_width": 0.10,
                    "live_spread_cost": 1.00,
                    "spread_width": 5.00,
                    "short_delta_live": -0.42,
                },
            ]
        )
        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=True,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
        )
        self.assertEqual(list(actionable["ticker"]), ["OK"])
        reasons = dict(zip(patterns["ticker"], patterns["quality_reject_reasons"]))
        self.assertIn("earnings", reasons["EARN"])
        self.assertIn("wide market", reasons["WIDE"])
        self.assertIn("short delta", reasons["DELTA"])

    def test_professional_quality_gate_blocks_lotto_and_flow_conflict(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 72,
                    "latest_close": 210.00,
                    "price_direction": "bullish",
                    "price_trend": 72.0,
                    "flow_direction": "bullish",
                    "flow_persistence": 74.0,
                    "whale_appearances": 18,
                    "backtest_verdict": "PASS",
                    "edge_pct": 6.5,
                    "backtest_signals": 160,
                    "quote_replay_verdict": "ENTRY_OK",
                    "live_validated": True,
                    "earnings_safe": True,
                    "live_bid_ask_width": 0.08,
                    "live_spread_cost": 2.10,
                    "spread_width": 5.00,
                    "long_strike": 210.0,
                },
                {
                    "ticker": "NIO",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 61.2,
                    "latest_close": 6.83,
                    "flow_direction": "bearish",
                    "whale_appearances": 5,
                    "backtest_verdict": "PASS",
                    "edge_pct": 17.5,
                    "backtest_signals": 164,
                    "quote_replay_verdict": "ENTRY_OK",
                    "live_validated": True,
                    "earnings_safe": True,
                    "live_bid_ask_width": 0.05,
                    "live_spread_cost": 0.28,
                    "spread_width": 2.50,
                },
                {
                    "ticker": "HOOD",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 78.4,
                    "days_observed": 5,
                    "latest_close": 90.75,
                    "price_direction": "bullish",
                    "price_trend": 79.0,
                    "flow_direction": "bullish",
                    "flow_persistence": 86.0,
                    "whale_appearances": 4,
                    "backtest_verdict": "PASS",
                    "edge_pct": 17.4,
                    "backtest_signals": 102,
                    "quote_replay_verdict": "ENTRY_OK",
                    "live_validated": True,
                    "earnings_safe": True,
                    "live_bid_ask_width": 0.08,
                    "live_spread_cost": 2.87,
                    "spread_width": 10.00,
                    "long_strike": 90.0,
                },
            ]
        )

        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=True,
            quote_replay_mode="gate",
            min_edge=0.0,
            min_signals=100,
            min_swing_score=60.0,
            allow_low_sample=False,
            min_underlying_price=20.0,
            min_debit_spread_price=0.75,
            min_whale_appearances=8,
        )

        self.assertEqual(list(actionable["ticker"]), ["HOOD", "AAPL"])
        self.assertEqual(list(patterns["ticker"]), ["NIO"])
        reasons = patterns.iloc[0]["quality_reject_reasons"]
        self.assertIn("lotto underlying", reasons)
        self.assertIn("lotto debit", reasons)
        self.assertIn("flow conflict", reasons)
        self.assertIn("thin institutional confirmation", reasons)

    def test_directional_debit_gate_blocks_weak_confirmation_and_chase(self) -> None:
        base = {
            "strategy": "Bull Call Debit",
            "direction": "bullish",
            "swing_score": 72,
            "latest_close": 100.00,
            "price_direction": "bullish",
            "price_trend": 72.0,
            "flow_direction": "bullish",
            "flow_persistence": 72.0,
            "whale_appearances": 20,
            "backtest_verdict": "PASS",
            "edge_pct": 8.0,
            "backtest_signals": 160,
            "live_validated": True,
            "earnings_safe": True,
            "live_spread_cost": 4.00,
            "spread_width": 10.00,
            "long_strike": 100.0,
        }
        rows = []
        for ticker, overrides in (
            ("OK", {}),
            ("WEAK_FLOW", {"flow_persistence": 55.0}),
            ("MIXED_FLOW", {"flow_direction": "mixed"}),
            ("WEAK_PRICE", {"price_trend": 55.0}),
            ("RANGE", {"price_direction": "range_bound"}),
            ("EXPENSIVE", {"live_spread_cost": 6.00}),
            ("OTM", {"long_strike": 104.0}),
        ):
            row = dict(base)
            row["ticker"] = ticker
            row.update(overrides)
            rows.append(row)

        actionable, patterns = trend_analysis.split_actionable_candidates(
            pd.DataFrame(rows),
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=False,
            min_edge=0.0,
            min_signals=100,
            min_swing_score=60.0,
            allow_low_sample=False,
        )

        self.assertEqual(list(actionable["ticker"]), ["OK"])
        reasons = dict(zip(patterns["ticker"], patterns["quality_reject_reasons"]))
        self.assertIn("weak directional flow", reasons["WEAK_FLOW"])
        self.assertIn("directional flow not confirming", reasons["MIXED_FLOW"])
        self.assertIn("weak directional price trend", reasons["WEAK_PRICE"])
        self.assertIn("directional price not confirming", reasons["RANGE"])
        self.assertIn("expensive debit", reasons["EXPENSIVE"])
        self.assertIn("long strike too far OTM", reasons["OTM"])

    def test_max_conviction_requires_actionable_alignment(self) -> None:
        actionable = pd.DataFrame(
            [
                {
                    "ticker": "HOOD",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "variant_tag": "pre_earnings",
                    "target_expiry": "2026-04-24",
                    "live_strike_setup": "Buy 90C / Sell 100C",
                    "swing_score": 78.4,
                    "days_observed": 5,
                    "latest_close": 90.75,
                    "price_direction": "bullish",
                    "flow_direction": "bullish",
                    "oi_direction": "bullish",
                    "dp_direction": "accumulation",
                    "whale_appearances": 4,
                    "backtest_verdict": "PASS",
                    "edge_pct": 17.4,
                    "backtest_signals": 102,
                    "quote_replay_verdict": "ENTRY_OK",
                    "live_validated": True,
                    "live_bid_ask_width": 0.08,
                    "live_spread_cost": 2.87,
                    "spread_width": 10.00,
                },
                {
                    "ticker": "WEAK",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 78.0,
                    "days_observed": 5,
                    "price_direction": "range_bound",
                    "flow_direction": "bullish",
                    "oi_direction": "bullish",
                    "dp_direction": "accumulation",
                    "whale_appearances": 4,
                    "backtest_verdict": "PASS",
                    "edge_pct": 20.0,
                    "backtest_signals": 120,
                    "live_bid_ask_width": 0.08,
                    "live_spread_cost": 2.87,
                    "spread_width": 10.00,
                },
            ]
        )

        max_conviction = trend_analysis.build_max_conviction(actionable, top_n=10)

        self.assertEqual(list(max_conviction["ticker"]), ["HOOD"])
        self.assertEqual(max_conviction.iloc[0]["position_size_tier"], "MAX_PLANNED_RISK")

    def test_candidate_shortlist_excludes_hard_lotto_rejects(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NIO",
                    "variant_tag": "base",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 72,
                    "days_observed": 30,
                    "latest_close": 6.83,
                    "price_direction": "bullish",
                    "price_trend": 80,
                    "flow_direction": "bearish",
                    "flow_persistence": 80,
                    "oi_direction": "bullish",
                    "oi_momentum": 80,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 80,
                    "whale_appearances": 12,
                    "backtest_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": False,
                    "quality_gate_pass": False,
                    "quality_reject_reasons": (
                        "lotto underlying: stock $6.83 < min $20.00; "
                        "flow conflict: bearish flow vs bullish trade"
                    ),
                    "actionability_reject_reasons": (
                        "quote replay UNAVAILABLE; "
                        "lotto underlying: stock $6.83 < min $20.00; "
                        "flow conflict: bearish flow vs bullish trade"
                    ),
                    "edge_pct": 17.5,
                    "backtest_signals": 164,
                    "backtest_verdict": "PASS",
                }
            ]
        )

        shortlist = trend_analysis.build_candidate_shortlist(candidates, top_n=5)

        self.assertTrue(shortlist.empty)

    def test_trade_workup_surfaces_quality_low_sample_without_actionable_label(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NVDA",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "variant_tag": "pre_earnings",
                    "target_expiry": "2026-05-15",
                    "live_strike_setup": "Buy 210C / Sell 220C",
                    "swing_score": 67.4,
                    "latest_close": 201.68,
                    "flow_direction": "bullish",
                    "whale_appearances": 30,
                    "backtest_verdict": "LOW_SAMPLE",
                    "edge_pct": 20.5,
                    "backtest_signals": 77,
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "quality_reject_reasons": "",
                    "actionability_reject_reasons": "backtest LOW_SAMPLE, edge 20.5%, signals 77",
                },
                {
                    "ticker": "NIO",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "variant_tag": "pre_earnings",
                    "target_expiry": "2026-05-29",
                    "live_strike_setup": "Buy 7.5C / Sell 10C",
                    "swing_score": 61.2,
                    "latest_close": 6.83,
                    "flow_direction": "bearish",
                    "whale_appearances": 5,
                    "backtest_verdict": "PASS",
                    "edge_pct": 17.5,
                    "backtest_signals": 164,
                    "quality_gate_pass": False,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "quality_reject_reasons": (
                        "lotto underlying: stock $6.83 < min $20.00; "
                        "flow conflict: bearish flow vs bullish trade"
                    ),
                    "actionability_reject_reasons": (
                        "lotto underlying: stock $6.83 < min $20.00; "
                        "flow conflict: bearish flow vs bullish trade"
                    ),
                },
                {
                    "ticker": "QUOTE",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 70.0,
                    "latest_close": 120.0,
                    "flow_direction": "bullish",
                    "whale_appearances": 20,
                    "backtest_verdict": "LOW_SAMPLE",
                    "edge_pct": 15.0,
                    "backtest_signals": 90,
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": False,
                    "quality_reject_reasons": "",
                    "actionability_reject_reasons": "quote replay UNAVAILABLE",
                },
            ]
        )

        workups = trend_analysis.build_trade_workups(
            candidates,
            top_n=10,
            min_edge=0.0,
            min_signals=100,
            min_workup_signals=50,
            min_swing_score=60.0,
        )

        self.assertEqual(list(workups["ticker"]), ["NVDA"])
        self.assertIn("LOW_SAMPLE", workups.iloc[0]["workup_reason"])
        self.assertIn("not size as Actionable Now", workups.iloc[0]["workup_next_step"])

    def test_current_trade_setups_surface_workable_non_order_tickets(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NBIS",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 69.2,
                    "latest_close": 157.14,
                    "live_long_strike": 160.0,
                    "long_strike": 160.0,
                    "spread_width": 10.0,
                    "edge_pct": 35.5,
                    "backtest_signals": 63,
                    "backtest_verdict": "LOW_SAMPLE",
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "actionability_reject_reasons": "backtest LOW_SAMPLE, edge 35.5%, signals 63",
                },
                {
                    "ticker": "MRVL",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 66.3,
                    "latest_close": 139.69,
                    "live_long_strike": 140.0,
                    "long_strike": 140.0,
                    "spread_width": 10.0,
                    "edge_pct": 12.2,
                    "backtest_signals": 201,
                    "backtest_verdict": "PASS",
                    "base_gate_pass": True,
                    "quality_gate_pass": False,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "quality_reject_reasons": "wide market: bid/ask 36% of spread price",
                },
                {
                    "ticker": "NIO",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "swing_score": 80.0,
                    "latest_close": 6.83,
                    "edge_pct": 17.5,
                    "backtest_signals": 164,
                    "backtest_verdict": "PASS",
                    "base_gate_pass": True,
                    "quality_gate_pass": False,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "quality_reject_reasons": "lotto underlying: stock $6.83 < min $20.00",
                },
            ]
        )

        setups = trend_analysis.build_current_trade_setups(
            candidates,
            top_n=10,
            min_edge=0.0,
            min_signals=100,
            min_workup_signals=50,
            min_swing_score=60.0,
        )

        self.assertEqual(list(setups["ticker"]), ["NBIS", "MRVL"])
        self.assertEqual(list(setups["setup_tier"]), ["TRADE_SETUP", "REBUILD"])
        self.assertIn("holds above", setups.iloc[0]["setup_entry_trigger"])

    def test_volatile_gex_blocks_iron_condor(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "IC",
                    "strategy": "Iron Condor",
                    "swing_score": 70,
                    "backtest_verdict": "PASS",
                    "edge_pct": 6.5,
                    "backtest_signals": 150,
                    "live_validated": True,
                    "earnings_safe": True,
                    "live_bid_ask_width": 0.20,
                    "live_spread_cost": 4.00,
                    "spread_width": 10.00,
                    "short_put_delta_live": -0.20,
                    "short_call_delta_live": 0.22,
                    "gex_regime": "volatile",
                }
            ]
        )
        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=True,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
        )
        self.assertTrue(actionable.empty)
        self.assertIn("volatile GEX", str(patterns.iloc[0]["quality_reject_reasons"]))

    def test_watchlist_trigger_explains_near_miss(self) -> None:
        row = pd.Series(
            {
                "ticker": "META",
                "strategy": "Iron Condor",
                "backtest_verdict": "PASS",
                "actionability_reject_reasons": "earnings in trade window",
                "quality_reject_reasons": "earnings in trade window",
            }
        )
        self.assertIn("Recheck after earnings", trend_analysis._watchlist_trigger(row))

    def test_watchlist_trigger_prioritizes_missing_quote_replay(self) -> None:
        row = pd.Series(
            {
                "ticker": "OXY",
                "strategy": "Bull Call Debit",
                "backtest_verdict": "PASS",
                "actionability_reject_reasons": (
                    "quote replay UNAVAILABLE, status skipped_missing_entry: "
                    "missing_entry_net_or_quotes; earnings in trade window"
                ),
                "quality_reject_reasons": "earnings in trade window",
            }
        )
        self.assertIn("both legs", trend_analysis._watchlist_trigger(row))

    def test_report_blocks_put_trade_setup_first(self) -> None:
        actionable = pd.DataFrame(
            [
                {
                    "ticker": "HOOD",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "target_expiry": "2026-04-24",
                    "strike_setup": "Buy 90C / Sell 100C",
                    "live_strike_setup": "Buy 90C / Sell 100C (10w, $2.87 debit)",
                    "live_validated": True,
                    "swing_score": 78.4,
                    "edge_pct": 17.4,
                    "backtest_signals": 102,
                }
            ]
        )

        block = "\n".join(
            trend_analysis._actionable_blocks(
                actionable,
                schwab_enabled=True,
                backtest_enabled=True,
            )
        )

        self.assertIn(
            "**Trade setup:** Bull Call Debit | Buy 90C / Sell 100C (10w, $2.87 debit) | exp 2026-04-24",
            block,
        )
        self.assertLess(block.index("**Trade setup:**"), block.index("**Variant:**"))

    def test_walk_forward_dates_need_prior_lookback_and_future_horizon(self) -> None:
        days = [
            (dt.date(2026, 1, 1) + dt.timedelta(days=i), Path(f"/tmp/{i}"))
            for i in range(10)
        ]

        selected = trend_analysis._walk_forward_signal_dates(
            days,
            lookback=3,
            as_of=dt.date(2026, 1, 10),
            samples=2,
            max_horizon=2,
        )

        self.assertEqual(
            selected,
            [dt.date(2026, 1, 7), dt.date(2026, 1, 8)],
        )

    def test_signal_dates_are_horizon_specific(self) -> None:
        days = [
            (dt.date(2026, 1, 1) + dt.timedelta(days=i), Path(f"/tmp/{i}"))
            for i in range(12)
        ]

        selected = trend_analysis._signal_dates_by_horizon(
            days,
            lookback=3,
            as_of=dt.date(2026, 1, 12),
            samples=20,
            horizons=[1, 4],
        )

        self.assertGreater(len(selected[1]), len(selected[4]))
        self.assertIn(dt.date(2026, 1, 11), selected[1])
        self.assertNotIn(dt.date(2026, 1, 11), selected[4])

    def test_walk_forward_confidence_text_uses_completed_outcomes(self) -> None:
        outcomes = pd.DataFrame(
            [
                {"pnl": 120.0, "return_on_risk": 0.24},
                {"pnl": -40.0, "return_on_risk": -0.08},
                {"pnl": 20.0, "return_on_risk": 0.04},
            ]
        )

        text = trend_analysis._walk_forward_confidence_text(outcomes)

        self.assertIn("low sample", text)
        self.assertIn("3 outcomes", text)
        self.assertIn("hit rate 67%", text)

    def test_research_confidence_requires_supportive_bucket(self) -> None:
        negative = pd.DataFrame(
            [
                {"policy": "professional_quality_gate", "verdict": "negative"},
                {"policy": "backtest_pass_sample_gate", "verdict": "mixed"},
            ]
        )
        supportive = pd.DataFrame(
            [
                {"policy": "professional_quality_gate", "verdict": "supportive"},
            ]
        )

        self.assertFalse(trend_analysis._research_confidence_supportive(negative))
        self.assertTrue(trend_analysis._research_confidence_supportive(supportive))

    def test_research_summary_requires_unique_setups_not_repeated_horizons(self) -> None:
        rows = []
        for setup_idx in range(8):
            for horizon in (5, 10, 20):
                rows.append(
                    {
                        "policy": "professional_quality_gate",
                        "signal_date": "2026-02-01",
                        "ticker": f"T{setup_idx}",
                        "trade_setup": f"setup-{setup_idx}",
                        "horizon_market_days": horizon,
                        "pnl": 100.0,
                        "return_on_risk": 0.25,
                    }
                )

        summary = trend_analysis._research_summary_from_outcomes(pd.DataFrame(rows))
        horizon = trend_analysis._research_summary_by_horizon_from_outcomes(pd.DataFrame(rows))

        self.assertEqual(int(summary.iloc[0]["outcomes"]), 24)
        self.assertEqual(int(summary.iloc[0]["unique_setups"]), 8)
        self.assertEqual(summary.iloc[0]["verdict"], "low_sample")
        self.assertFalse(trend_analysis._research_confidence_supportive(summary, horizon))

    def test_research_confidence_can_use_supportive_horizon_bucket(self) -> None:
        rows = []
        for setup_idx in range(20):
            rows.append(
                {
                    "policy": "professional_quality_gate",
                    "signal_date": f"2026-02-{setup_idx + 1:02d}",
                    "ticker": f"T{setup_idx}",
                    "trade_setup": f"setup-{setup_idx}",
                    "horizon_market_days": 10,
                    "pnl": 100.0,
                    "return_on_risk": 0.20,
                }
            )

        outcomes = pd.DataFrame(rows)
        summary = trend_analysis._research_summary_from_outcomes(outcomes)
        horizon = trend_analysis._research_summary_by_horizon_from_outcomes(outcomes)

        self.assertEqual(summary.iloc[0]["verdict"], "supportive")
        self.assertEqual(horizon.iloc[0]["verdict"], "supportive")
        self.assertTrue(trend_analysis._research_confidence_supportive(summary, horizon))

    def test_strategy_family_audit_requires_train_and_validation_profit(self) -> None:
        rows = []
        for setup_idx in range(30):
            signal_day = dt.date(2026, 1, 1) + dt.timedelta(days=setup_idx)
            rows.append(
                {
                    "policy": "entry_available_score_gate",
                    "signal_date": signal_day.isoformat(),
                    "horizon_market_days": 10,
                    "ticker": f"E{setup_idx}",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": f"setup-{setup_idx}",
                    "swing_score": 66.0,
                    "price_direction": "bullish",
                    "price_trend": 68.0,
                    "flow_direction": "bullish",
                    "flow_persistence": 70.0,
                    "sector": "Energy",
                    "pnl": 120.0 if setup_idx < 27 else -40.0,
                    "return_on_risk": 0.20,
                }
            )

        audit = trend_analysis._strategy_family_audit_from_outcomes(pd.DataFrame(rows))
        row = audit[audit["family"].eq("bull_energy_materials_debit")].iloc[0]

        self.assertEqual(row["verdict"], "promotable")
        self.assertEqual(int(row["validation_unique_setups"]), 9)

    def test_strategy_family_gate_blocks_unpromoted_candidate(self) -> None:
        candidate = pd.DataFrame(
            [
                {
                    "ticker": "MRVL",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "swing_score": 66.0,
                    "price_direction": "bullish",
                    "price_trend": 85.0,
                    "flow_direction": "bullish",
                    "flow_persistence": 56.0,
                    "sector": "Technology",
                    "quote_replay_verdict": "ENTRY_OK",
                    "live_validated": True,
                    "backtest_verdict": "PASS",
                    "edge_pct": 10.0,
                    "backtest_signals": 150,
                }
            ]
        )
        audit = pd.DataFrame(
            [
                {
                    "family": "growth_bull_call_debit",
                    "horizon_market_days": 10,
                    "validation_unique_setups": 10,
                    "validation_hit_rate": 0.20,
                    "validation_avg_pnl": -100.0,
                    "validation_profit_factor": 0.5,
                    "verdict": "validation_negative",
                }
            ]
        )

        annotated = trend_analysis.annotate_strategy_family_gate(candidate, audit)
        actionable, patterns = trend_analysis.split_actionable_candidates(
            annotated,
            top_n=1,
            backtest_enabled=True,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
            schwab_enabled=True,
            quote_replay_mode="gate",
            min_swing_score=60.0,
        )

        self.assertTrue(actionable.empty)
        self.assertIn("strategy family audit not promotable", patterns.iloc[0]["base_gate_reasons"])

    def test_ticker_playbook_audit_can_promote_narrow_setup(self) -> None:
        rows = []
        for setup_idx in range(20):
            signal_day = dt.date(2026, 1, 1) + dt.timedelta(days=setup_idx)
            rows.append(
                {
                    "policy": "entry_available_score_gate",
                    "signal_date": signal_day.isoformat(),
                    "horizon_market_days": 20,
                    "ticker": "NVDA",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": f"NVDA setup {setup_idx}",
                    "pnl": 80.0 if setup_idx < 16 else 220.0,
                    "return_on_risk": 0.20,
                }
            )

        audit = trend_analysis._ticker_playbook_audit_from_outcomes(pd.DataFrame(rows))
        row = audit[audit["ticker"].eq("NVDA")].iloc[0]

        self.assertEqual(row["verdict"], "promotable")
        self.assertEqual(int(row["validation_unique_setups"]), 6)

    def test_ticker_playbook_support_can_override_broad_family_block(self) -> None:
        candidate = pd.DataFrame(
            [
                {
                    "ticker": "NBIS",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "swing_score": 69.0,
                    "backtest_verdict": "LOW_SAMPLE",
                    "edge_pct": 29.0,
                    "backtest_signals": 64,
                    "latest_close": 157.0,
                    "long_strike": 160.0,
                    "short_strike": 170.0,
                    "spread_width": 10.0,
                    "live_spread_cost": 3.75,
                    "live_bid_ask_width": 0.20,
                    "price_direction": "bullish",
                    "price_trend": 70.0,
                    "flow_direction": "bullish",
                    "flow_persistence": 72.0,
                    "live_validated": True,
                    "quote_replay_verdict": "ENTRY_OK",
                    "strategy_family_gate_pass": False,
                    "strategy_family_verdict": "negative",
                    "strategy_family": "growth_bull_call_debit",
                    "ticker_playbook_gate_pass": True,
                    "ticker_playbook_verdict": "promotable",
                    "ticker_playbook_summary": "promotable, 20d, validation 7 setups, hit 71%, avg $107.14, PF 2.42",
                }
            ]
        )

        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidate,
            top_n=1,
            backtest_enabled=True,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
            schwab_enabled=True,
            quote_replay_mode="gate",
            min_swing_score=60.0,
        )

        self.assertEqual(list(actionable["ticker"]), ["NBIS"])
        self.assertTrue(patterns.empty)


if __name__ == "__main__":
    unittest.main()
