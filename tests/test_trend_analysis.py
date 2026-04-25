import argparse
import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from uwos import swing_trend_pipeline
from uwos import trend_analysis
from uwos import trend_missed_mover_audit


class TestTrendAnalysisWrapper(unittest.TestCase):
    class _FakeQuoteStore:
        def __init__(self, quotes=None, dates=None) -> None:
            if quotes is None:
                self.quotes = {}
            elif isinstance(quotes, dict):
                self.quotes = dict(quotes)
            else:
                self.quotes = {(None, symbol): object() for symbol in quotes}
            self.dates = list(dates or [])

        def get_leg_quote(self, asof, symbol):
            return self.quotes.get((asof, symbol), self.quotes.get((None, symbol)))

        def available_dates(self):
            return list(self.dates)

    def test_default_single_date_scan_skips_walk_forward_replay(self) -> None:
        args = trend_analysis.parse_args(["2026-04-20", "30"])

        self.assertEqual(args.walk_forward_samples, 0)

    def test_default_invocation_uses_l30_lookback(self) -> None:
        args = trend_analysis.parse_args(["2026-04-23"])
        as_of, lookback = trend_analysis.resolve_invocation(args)

        self.assertEqual(as_of, dt.date(2026, 4, 23))
        self.assertEqual(lookback, 30)

    def test_event_momentum_watch_surfaces_intc_like_catalyst_spike(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "INTC",
                    "direction": "bullish",
                    "strategy": "Bull Put Credit",
                    "strike_setup": "Sell 40P / Buy 35P (5w, ~$0.75 credit)",
                    "live_strike_setup": "Sell 57P / Buy 50P (7w, $0.82 credit)",
                    "live_validated": True,
                    "target_expiry": "2026-06-05",
                    "latest_close": 66.78,
                    "live_spot": 81.85,
                    "swing_score": 52.5,
                    "price_direction": "bullish",
                    "price_trend": 70.3,
                    "flow_direction": "bullish",
                    "flow_persistence": 56.0,
                    "oi_direction": "mixed",
                    "oi_momentum": 40.1,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 44.0,
                    "whale_consensus": 97.3,
                    "whale_appearances": 29,
                    "days_observed": 30,
                    "latest_iv_rank": 100.0,
                    "next_earnings_date": "2026-04-23",
                    "earnings_label": "EARNINGS 2026-04-23 (0d away, 43d before expiry)",
                    "actionability_reject_reasons": (
                        "swing score 52.5 < min 60.0; "
                        "earnings in trade window: EARNINGS 2026-04-23"
                    ),
                    "quality_reject_reasons": "wide market: bid/ask 270% of spread price",
                }
            ]
        )

        watch = trend_analysis.build_event_momentum_watch(candidates, top_n=5)

        self.assertEqual(watch.iloc[0]["ticker"], "INTC")
        self.assertEqual(watch.iloc[0]["event_watch_status"], "WATCH ONLY")
        self.assertGreaterEqual(float(watch.iloc[0]["event_watch_score"]), 65.0)
        self.assertIn("live spot $81.85 vs latest UW close $66.78", watch.iloc[0]["event_watch_catalysts"])
        self.assertIn("Do not chase stale strikes", watch.iloc[0]["event_watch_trigger"])
        self.assertEqual(watch.iloc[0]["strike_setup"], "Sell 57P / Buy 50P (7w, $0.82 credit)")

    def test_event_momentum_watch_flips_stale_direction_to_live_reaction(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "DROP",
                    "direction": "bullish",
                    "strategy": "Bull Put Credit",
                    "strike_setup": "Sell 95P / Buy 90P (5w, ~$0.80 credit)",
                    "target_expiry": "2026-06-05",
                    "latest_close": 100.0,
                    "live_spot": 78.0,
                    "swing_score": 61.0,
                    "price_direction": "bullish",
                    "price_trend": 72.0,
                    "flow_direction": "bullish",
                    "flow_persistence": 70.0,
                    "oi_direction": "bullish",
                    "oi_momentum": 60.0,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 65.0,
                    "whale_consensus": 90.0,
                    "whale_appearances": 12,
                    "days_observed": 30,
                    "latest_iv_rank": 95.0,
                    "next_earnings_date": "2026-04-23",
                    "earnings_label": "EARNINGS 2026-04-23 (0d away, 43d before expiry)",
                    "direction_note": "event direction pending",
                    "actionability_reject_reasons": "earnings in trade window",
                }
            ]
        )

        watch = trend_analysis.build_event_momentum_watch(candidates, top_n=5)

        self.assertEqual(watch.iloc[0]["direction"], "bearish")
        self.assertEqual(watch.iloc[0]["strategy"], "Bear Call Credit")
        self.assertIn("C / Buy", watch.iloc[0]["strike_setup"])
        self.assertIn("Reaction direction changed from bullish to bearish", watch.iloc[0]["event_watch_reason"])

    def test_event_momentum_watch_flips_on_latest_day_reversal(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "REV",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "strike_setup": "Buy 100C / Sell 110C (10w, ~$4.00 debit)",
                    "target_expiry": "2026-06-05",
                    "latest_close": 100.0,
                    "swing_score": 63.0,
                    "price_direction": "bullish",
                    "price_trend": 72.0,
                    "latest_return_pct": -0.045,
                    "latest_return_direction": "bearish",
                    "flow_direction": "bullish",
                    "flow_persistence": 70.0,
                    "oi_direction": "bullish",
                    "oi_momentum": 60.0,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 65.0,
                    "whale_consensus": 90.0,
                    "whale_appearances": 12,
                    "days_observed": 30,
                    "latest_iv_rank": 50.0,
                    "next_earnings_date": "2026-05-07",
                    "earnings_label": "EARNINGS 2026-05-07 (14d away, 15d before expiry)",
                    "direction_note": "latest-day reversal conflict",
                    "actionability_reject_reasons": "latest-day reversal conflict",
                }
            ]
        )

        watch = trend_analysis.build_event_momentum_watch(candidates, top_n=5)

        self.assertEqual(watch.iloc[0]["direction"], "bearish")
        self.assertEqual(watch.iloc[0]["strategy"], "Bear Put Debit")
        self.assertIn("P / Sell", watch.iloc[0]["strike_setup"])

    def test_event_direction_pending_blocks_actionability(self) -> None:
        row = pd.Series(
            {
                "ticker": "EARN",
                "direction": "bullish",
                "strategy": "Bull Call Debit",
                "latest_close": 100.0,
                "direction_status": "event_pending",
                "direction_note": "event direction pending: earnings tomorrow",
                "earnings_safe": True,
                "live_bid_ask_width": 0.05,
                "live_spread_cost": 2.0,
                "spread_width": 5.0,
                "live_long_strike": 100.0,
                "price_direction": "bullish",
                "price_trend": 75.0,
                "flow_direction": "bullish",
                "flow_persistence": 75.0,
                "whale_appearances": 10,
                "days_observed": 30,
            }
        )

        reasons = trend_analysis.quality_gate_reasons(
            row,
            schwab_enabled=False,
            allow_earnings_risk=False,
            allow_volatile_ic=False,
            allow_flow_conflict=False,
            max_bid_ask_to_price_pct=0.30,
            max_bid_ask_to_width_pct=0.10,
            max_short_delta=0.30,
            min_underlying_price=20.0,
            min_debit_spread_price=0.75,
            min_whale_appearances=8,
        )

        self.assertIn("event direction pending: earnings tomorrow", reasons)

    def test_direction_scoring_marks_pre_event_trend_as_pending(self) -> None:
        sig = swing_trend_pipeline.SwingSignals(
            ticker="EARN",
            latest_date=dt.date(2026, 4, 22),
            next_earnings_date=dt.date(2026, 4, 23),
            n_days_observed=30,
            latest_close=100.0,
            latest_return_pct=0.004,
            latest_return_direction="neutral",
            price_direction="bullish",
            price_r_squared=0.80,
            flow_direction="bullish",
            flow_consistency=0.90,
            hot_flow_direction="bullish",
            hot_flow_consistency=0.90,
            pcr_direction="declining",
            oi_direction="bullish",
            oi_consistency=0.70,
            dp_direction="accumulation",
            dp_consistency=0.70,
            latest_iv_rank=70.0,
        )
        cfg = {
            "scoring": {
                "weights": {
                    "flow_persistence": 0.30,
                    "oi_momentum": 0.20,
                    "iv_regime": 0.15,
                    "price_trend": 0.15,
                    "whale_consensus": 0.10,
                    "dp_confirmation": 0.10,
                },
                "direction_inference": {
                    "event_direction_guard_days": 1,
                    "event_same_day_reaction_min_abs": 0.03,
                },
            }
        }

        score = swing_trend_pipeline.score_ticker(sig, cfg)

        self.assertEqual(score.direction, "bullish")
        self.assertEqual(score.direction_status, "event_pending")
        self.assertIn("wait for post-event", score.direction_note)

    def test_event_momentum_watch_does_not_promote_actionable_trade(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "INTC",
                    "direction": "bullish",
                    "strategy": "Bull Put Credit",
                    "strike_setup": "Sell 40P / Buy 35P (5w, ~$0.75 credit)",
                    "target_expiry": "2026-06-05",
                    "latest_close": 66.78,
                    "live_spot": 81.85,
                    "swing_score": 52.5,
                    "price_direction": "bullish",
                    "price_trend": 70.3,
                    "flow_direction": "bullish",
                    "flow_persistence": 56.0,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 44.0,
                    "whale_consensus": 97.3,
                    "whale_appearances": 29,
                    "days_observed": 30,
                    "latest_iv_rank": 100.0,
                    "earnings_label": "EARNINGS 2026-04-23 (0d away, 43d before expiry)",
                    "backtest_verdict": "",
                    "edge_pct": "",
                    "backtest_signals": "",
                    "live_validated": True,
                    "quote_replay_verdict": "UNAVAILABLE",
                    "live_bid_ask_width": 1.68,
                    "live_spread_cost": 0.88,
                    "spread_width": 4.0,
                    "earnings_safe": False,
                    "schwab_actual_strategy_verdict": "negative",
                }
            ]
        )

        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=5,
            backtest_enabled=True,
            schwab_enabled=True,
            quote_replay_mode="gate",
            min_edge=0.0,
            min_signals=100,
            min_swing_score=60.0,
            allow_low_sample=False,
            allow_earnings_risk=False,
            max_bid_ask_to_price_pct=0.30,
            max_bid_ask_to_width_pct=0.10,
            max_short_delta=0.30,
            min_underlying_price=20.0,
            min_debit_spread_price=0.75,
            min_whale_appearances=8,
        )
        watch = trend_analysis.build_event_momentum_watch(patterns, top_n=5)

        self.assertTrue(actionable.empty)
        self.assertEqual(watch.iloc[0]["ticker"], "INTC")
        self.assertFalse(bool(patterns.iloc[0]["base_gate_pass"]))

    def test_missed_mover_audit_classifies_visibility_without_changing_gates(self) -> None:
        frames = {
            "actionable": pd.DataFrame(columns=["ticker"]),
            "current_setup": pd.DataFrame(columns=["ticker"]),
            "event_watch": pd.DataFrame([{"ticker": "INTC", "event_watch_score": 82.2}]),
            "trade_workup": pd.DataFrame(columns=["ticker"]),
            "candidate_shortlist": pd.DataFrame(columns=["ticker"]),
            "proven_ticket": pd.DataFrame(columns=["ticker"]),
            "pattern": pd.DataFrame(columns=["ticker"]),
            "raw": pd.DataFrame([{"ticker": "INTC"}, {"ticker": "NVDA"}]),
        }

        coverage, source, row = trend_missed_mover_audit.classify_coverage(frames, "INTC")
        raw_only, _, raw_row = trend_missed_mover_audit.classify_coverage(frames, "NVDA")
        missed, _, missed_row = trend_missed_mover_audit.classify_coverage(frames, "AAPL")

        self.assertEqual(coverage, "event_watch")
        self.assertEqual(source, "event_watch")
        self.assertEqual(row["ticker"], "INTC")
        self.assertEqual(raw_only, "raw_only")
        self.assertEqual(raw_row["ticker"], "NVDA")
        self.assertEqual(missed, "missed")
        self.assertIsNone(missed_row)

    def test_universe_radar_appends_non_actionable_lower_ranked_tickers(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NVDA",
                    "strategy": "Bull Call Debit",
                    "swing_score": 80.0,
                    "live_validated": True,
                }
            ]
        )
        radar = pd.DataFrame(
            [
                {
                    "ticker": "NVDA",
                    "strategy": "Bull Call Debit",
                    "swing_score": 80.0,
                },
                {
                    "ticker": "INTC",
                    "strategy": "Bull Put Credit",
                    "swing_score": 52.5,
                },
            ]
        )

        merged = trend_analysis.merge_universe_radar_candidates(candidates, radar)
        intc = merged[merged["ticker"].eq("INTC")].iloc[0]

        self.assertEqual(len(merged), 2)
        self.assertEqual(intc["variant_tag"], "radar_only")
        self.assertTrue(bool(intc["radar_only"]))
        self.assertFalse(bool(intc["live_validated"]))
        self.assertIn("never treat it as an order ticket", intc["radar_note"])

    def test_ticker_universe_includes_latest_day_catalyst_supplement(self) -> None:
        cfg = {
            "filters": {
                "exclude_etfs": True,
                "exclude_indices": True,
                "min_market_cap": 2_000_000_000,
                "min_avg30_volume": 500_000,
                "max_tickers_to_score": 1,
                "max_latest_day_tickers": 1,
                "max_earnings_tickers": 5,
                "max_catalyst_tickers": 0,
                "max_total_tickers_to_score": 10,
                "catalyst_min_market_cap": 1_000_000_000,
                "catalyst_min_avg30_volume": 250_000,
                "catalyst_earnings_days": 2,
                "catalyst_iv_rank_floor": 70,
            }
        }
        base = {
            "issue_type": "Common Stock",
            "is_index": "f",
            "avg30_volume": 600_000,
            "total_volume": 900_000,
            "put_volume": 0,
            "call_volume": 0,
            "bullish_premium": 0,
            "bearish_premium": 0,
            "next_earnings_date": "",
            "iv_rank": 20,
        }
        screeners = {
            dt.date(2026, 4, 21): pd.DataFrame(
                [
                    {**base, "ticker": "NVDA", "marketcap": 3_000_000_000, "call_volume": 1_000_000},
                    {**base, "ticker": "ASGN", "marketcap": 1_500_000_000, "put_volume": 10},
                ]
            ),
            dt.date(2026, 4, 22): pd.DataFrame(
                [
                    {**base, "ticker": "NVDA", "marketcap": 3_000_000_000, "call_volume": 1_000_000},
                    {
                        **base,
                        "ticker": "ASGN",
                        "marketcap": 1_500_000_000,
                        "avg30_volume": 300_000,
                        "put_volume": 100,
                        "next_earnings_date": "2026-04-22",
                        "iv_rank": 95,
                    },
                ]
            ),
        }

        universe = swing_trend_pipeline.build_ticker_universe(screeners, cfg)

        self.assertIn("NVDA", universe)
        self.assertIn("ASGN", universe)

    def test_quote_replay_can_force_entry_only_for_live_scan(self) -> None:
        from uwos import trend_quote_replay
        from uwos.exact_spread_backtester import LegQuote, build_occ_symbol

        signal_date = dt.date(2026, 4, 21)
        expiry = dt.date(2026, 5, 15)
        long_leg = build_occ_symbol("NFLX", expiry, "C", 100.0)
        short_leg = build_occ_symbol("NFLX", expiry, "C", 110.0)
        q = LegQuote(bid=2.0, ask=2.2, mid=2.1, volume=10, open_interest=20, source_kind="test")
        store = self._FakeQuoteStore(
            quotes={
                (signal_date, long_leg): q,
                (signal_date, short_leg): q,
            },
            dates=[signal_date, dt.date(2026, 4, 22)],
        )

        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "strategy": "Bull Call Debit",
                    "target_expiry": expiry.isoformat(),
                    "long_strike": 100.0,
                    "short_strike": 110.0,
                    "cost_type": "debit",
                }
            ]
        )

        annotated, _ = trend_quote_replay.annotate_quote_replay(
            candidates,
            root=Path("/tmp"),
            signal_date=signal_date,
            mode="gate",
            exit_date_override=signal_date,
            quote_store=store,
        )

        self.assertEqual(annotated.iloc[0]["quote_replay_verdict"], "ENTRY_OK")

    def test_no_trade_diagnostics_show_blocked_pass_rows(self) -> None:
        patterns = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "strategy": "Bull Call Debit",
                    "target_expiry": "2026-05-22",
                    "strike_setup": "Buy 100C / Sell 110C",
                    "backtest_verdict": "PASS",
                    "edge_pct": 20.5,
                    "backtest_signals": 111,
                    "swing_score": 72.8,
                    "base_gate_pass": False,
                    "quality_gate_pass": False,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "base_gate_reasons": "open position conflict: existing option exposure in NFLX",
                    "quality_reject_reasons": "flow conflict: bearish flow vs bullish trade",
                    "actionability_reject_reasons": "open position conflict; flow conflict",
                }
            ]
        )
        lines = trend_analysis._no_trade_diagnostic_lines(
            candidates=patterns,
            actionable=pd.DataFrame(),
            patterns=patterns,
            current_setups=pd.DataFrame(),
            trade_workups=pd.DataFrame(),
            min_edge=0.0,
            min_signals=100,
            min_workup_signals=50,
        )
        text = "\n".join(lines)

        self.assertIn("Backtest PASS rows blocked before action", text)
        self.assertIn("NFLX", text)
        self.assertIn("open position conflict", text)

    def test_final_trade_output_surfaces_proven_lane_tickets(self) -> None:
        current = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "strategy": "Bull Call Debit",
                    "target_expiry": "2026-06-18",
                    "strike_setup": "Buy 93C / Sell 103C",
                    "live_strike_setup": "Buy 93C / Sell 103C (10w, $3.54 debit)",
                    "live_validated": True,
                    "setup_entry_trigger": "Enter only above 95.00 and keep debit <= 3.75.",
                }
            ]
        )
        proven_tickets = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "strategy": "Bull Call Debit",
                    "target_expiry": "2026-06-18",
                    "strike_setup": "Buy 93C / Sell 103C",
                    "live_strike_setup": "Buy 93C / Sell 103C (10w, $3.54 debit)",
                    "live_validated": True,
                    "proven_ticket_status": "DO NOT OPEN",
                    "proven_ticket_what_to_do": "Do not add a duplicate or conflicting position until the existing exposure is closed or justified.",
                    "proven_ticket_why": "open position conflict: existing option exposure in NFLX; flow conflict: bearish flow vs bullish trade",
                    "proven_ticket_next": "Only reconsider after the blocking condition clears and the setup is rerun.",
                }
            ]
        )

        lines = trend_analysis._final_trade_ticket_lines(
            proven_tickets=proven_tickets,
            current_setups=current,
        )
        text = "\n".join(lines)

        self.assertIn("### Make Now", text)
        self.assertIn("### Probe", text)
        self.assertIn("### Watch / Avoid", text)
        self.assertIn("DO NOT OPEN", text)
        self.assertIn("NFLX", text)
        self.assertIn("Enter only above 95.00", text)
        self.assertIn("Enter only when", text)
        self.assertIn("Blocking gates", text)
        self.assertNotIn("WULF", text)
        self.assertLess(text.index("### Make Now"), text.index("### Probe"))
        self.assertLess(text.index("### Probe"), text.index("### Watch / Avoid"))

    def test_watch_only_tickets_show_specific_entry_conditions(self) -> None:
        proven_tickets = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "target_expiry": "2026-05-22",
                    "live_strike_setup": "Buy 280C / Sell 290C (10w, $3.00 debit)",
                    "live_validated": True,
                    "latest_close": 273.43,
                    "live_long_strike": 280.0,
                    "live_short_strike": 290.0,
                    "spread_width": 10.0,
                    "flow_persistence": 56.7,
                    "price_trend": 61.2,
                    "next_earnings_date": "2026-04-30",
                    "backtest_verdict": "LOW_SAMPLE",
                    "edge_pct": 10.9,
                    "backtest_signals": 66,
                    "schwab_actual_playbook_summary": "2 Schwab closed trade(s), hit 50%, avg $15.00",
                    "batch_playbook_summary": "rolling batch proof insufficient_forward, avg -",
                    "proven_ticket_status": "WATCH ONLY",
                    "proven_ticket_what_to_do": "Watch only: Schwab history supports the playbook, but the current entry gates below are not clean yet.",
                    "proven_ticket_why": (
                        "market regime conflict: risk_off conflict for bullish Bull Call Debit; "
                        "batch proof playbook gate blocked: rolling batch proof insufficient_forward, avg -; "
                        "directional flow not confirming: mixed vs bullish trade; "
                        "weak directional flow: 56.7 < min 60.0; "
                        "long strike too far OTM: 2% > 2%; "
                        "earnings in trade window: EARNINGS 2026-04-30"
                    ),
                    "proven_ticket_next": "Only reconsider after the blocking condition clears and the setup is rerun.",
                }
            ]
        )

        lines = trend_analysis._final_trade_ticket_lines(
            proven_tickets=proven_tickets,
            current_setups=pd.DataFrame(),
        )
        text = "\n".join(lines)

        self.assertIn("AAPL - `WATCH ONLY`", text)
        self.assertIn("Options flow must confirm bullish direction and score >= 60 from 56.7", text)
        self.assertIn("Underlying should close above the long call strike $280.00", text)
        self.assertIn("Wait until after 2026-04-30 earnings", text)

    def test_build_proven_playbook_tickets_suppresses_unproven_research_names(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "WULF",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "strike_setup": "Buy 20C / Sell 22.5C",
                    "target_expiry": "2026-05-01",
                    "swing_score": 63.8,
                    "edge_pct": 35.5,
                    "backtest_signals": 83,
                    "backtest_verdict": "LOW_SAMPLE",
                    "batch_playbook_verdict": "no_match",
                    "base_gate_pass": False,
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                },
                {
                    "ticker": "NFLX",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "strike_setup": "Buy 93C / Sell 103C",
                    "target_expiry": "2026-06-18",
                    "swing_score": 72.8,
                    "edge_pct": 17.8,
                    "backtest_signals": 109,
                    "backtest_verdict": "PASS",
                    "batch_playbook_verdict": "supportive",
                    "base_gate_pass": False,
                    "quality_gate_pass": False,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "actionability_reject_reasons": "open position conflict: existing option exposure in NFLX; flow conflict: bearish flow vs bullish trade",
                    "quality_reject_reasons": "flow conflict: bearish flow vs bullish trade",
                },
            ]
        )

        tickets = trend_analysis.build_proven_playbook_tickets(
            candidates,
            top_n=5,
            min_edge=0.0,
            min_signals=100,
            min_workup_signals=50,
        )

        self.assertEqual(list(tickets["ticker"]), ["NFLX"])
        self.assertEqual(tickets.iloc[0]["proven_ticket_status"], "DO NOT OPEN")

    def test_build_proven_playbook_tickets_can_surface_family_supported_ticket(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NVDA",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "strike_setup": "Buy 110C / Sell 120C",
                    "target_expiry": "2026-04-10",
                    "swing_score": 62.7,
                    "edge_pct": 5.3,
                    "backtest_signals": 119,
                    "backtest_verdict": "PASS",
                    "batch_playbook_verdict": "no_match",
                    "batch_family_verdict": "supportive",
                    "base_gate_pass": True,
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "position_size_tier": "STARTER_RISK",
                }
            ]
        )

        tickets = trend_analysis.build_proven_playbook_tickets(
            candidates,
            top_n=5,
            min_edge=0.0,
            min_signals=100,
            min_workup_signals=50,
        )

        self.assertEqual(list(tickets["ticker"]), ["NVDA"])
        self.assertEqual(tickets.iloc[0]["proven_ticket_status"], "MAKE NOW")
        self.assertIn("strategy family", tickets.iloc[0]["proven_ticket_why"].lower())

    def test_build_proven_playbook_tickets_can_surface_broker_supported_ticket(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "strike_setup": "Buy 90C / Sell 100C",
                    "target_expiry": "2026-05-08",
                    "swing_score": 67.0,
                    "edge_pct": 12.0,
                    "backtest_signals": 108,
                    "backtest_verdict": "PASS",
                    "batch_playbook_verdict": "no_match",
                    "batch_family_verdict": "no_match",
                    "schwab_actual_playbook_supportive": True,
                    "schwab_actual_playbook_verdict": "emerging_positive",
                    "base_gate_pass": True,
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "position_size_tier": "STARTER_RISK",
                }
            ]
        )

        tickets = trend_analysis.build_proven_playbook_tickets(
            candidates,
            top_n=5,
            min_edge=0.0,
            min_signals=100,
            min_workup_signals=50,
        )

        self.assertEqual(list(tickets["ticker"]), ["NFLX"])
        self.assertEqual(tickets.iloc[0]["proven_ticket_status"], "MAKE NOW")
        self.assertIn("schwab", tickets.iloc[0]["proven_ticket_why"].lower())

    def test_current_setup_pool_prefers_broker_supported_candidates(self) -> None:
        annotated = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "schwab_actual_playbook_supportive": True,
                },
                {
                    "ticker": "WULF",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "schwab_actual_playbook_supportive": False,
                },
            ]
        )

        proven = annotated[trend_analysis._proven_playbook_mask(annotated)].copy()
        pool = trend_analysis._current_setup_reporting_pool(annotated, proven)

        self.assertEqual(list(pool["ticker"]), ["NFLX"])

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

    def test_chain_optimizer_prefers_listed_expiry_with_local_quote_coverage(self) -> None:
        score = swing_trend_pipeline.SwingScore(
            ticker="NFLX",
            recommended_strategy="Bull Call Debit",
            target_expiry="2026-06-05",
            target_dte=46,
            long_strike=100.0,
            short_strike=110.0,
            spread_width=10.0,
            cost_type="debit",
            est_cost=3.35,
        )
        signals = swing_trend_pipeline.SwingSignals(
            ticker="NFLX",
            latest_date=dt.date(2026, 4, 20),
            latest_close=97.0,
        )
        quote_store = self._FakeQuoteStore(
            {
                swing_trend_pipeline._build_occ_symbol("NFLX", dt.date(2026, 5, 22), "C", 100.0),
                swing_trend_pipeline._build_occ_symbol("NFLX", dt.date(2026, 5, 22), "C", 110.0),
            }
        )
        chain_map = {
            "2026-05-22": {
                "C": {
                    90.0: {"bid": 7.7, "ask": 8.1, "delta": 0.72},
                    100.0: {"bid": 4.0, "ask": 4.2, "delta": 0.56},
                    110.0: {"bid": 1.9, "ask": 2.1, "delta": 0.29},
                },
                "P": {},
            },
            "2026-06-19": {
                "C": {
                    100.0: {"bid": 5.1, "ask": 6.3, "delta": 0.58},
                    110.0: {"bid": 2.9, "ask": 4.1, "delta": 0.34},
                },
                "P": {},
            },
        }

        candidate = swing_trend_pipeline._optimize_score_with_live_chain(
            score,
            signals,
            chain_map,
            spot=97.0,
            cfg={
                "strategy_selection": {
                    "bull_low_iv": {
                        "strategy": "Bull Call Debit",
                        "dte_range": [21, 70],
                        "target_dte": 42,
                    }
                },
                "filters": {"earnings_buffer_days": 3},
            },
            quote_store=quote_store,
            max_candidates=4,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["target_expiry"], "2026-05-22")
        self.assertEqual(candidate["live_strike_setup"], "Buy 100C / Sell 110C (10w, $2.10 debit)")
        self.assertEqual(candidate["quote_hits"], 2)

    def test_chain_optimizer_avoids_negative_credit_spreads(self) -> None:
        score = swing_trend_pipeline.SwingScore(
            ticker="BAC",
            recommended_strategy="Bull Put Credit",
            target_expiry="2026-06-05",
            target_dte=42,
            long_strike=50.0,
            short_strike=55.0,
            spread_width=5.0,
            cost_type="credit",
            est_cost=1.20,
        )
        signals = swing_trend_pipeline.SwingSignals(
            ticker="BAC",
            latest_date=dt.date(2026, 4, 20),
            latest_close=58.0,
        )
        chain_map = {
            "2026-05-22": {
                "P": {
                    45.0: {"bid": 0.20, "ask": 0.30, "delta": -0.06},
                    50.0: {"bid": 0.75, "ask": 0.90, "delta": -0.12},
                    55.0: {"bid": 1.85, "ask": 2.05, "delta": -0.24},
                },
                "C": {},
            }
        }

        candidate = swing_trend_pipeline._optimize_score_with_live_chain(
            score,
            signals,
            chain_map,
            spot=58.0,
            cfg={
                "strategy_selection": {
                    "bull_high_iv": {
                        "strategy": "Bull Put Credit",
                        "dte_range": [28, 56],
                        "target_dte": 42,
                    }
                }
            },
            max_candidates=3,
        )

        self.assertIsNotNone(candidate)
        self.assertGreater(candidate["live_spread_cost"], 0.0)
        self.assertEqual(candidate["live_strike_setup"], "Sell 55P / Buy 50P (5w, $1.12 credit)")

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

    def test_supported_bullish_reversal_debit_can_skip_breakout_price_checks(self) -> None:
        row = pd.Series(
            {
                "ticker": "NFLX",
                "strategy": "Bull Call Debit",
                "direction": "bullish",
                "swing_score": 63.4,
                "latest_close": 100.0,
                "price_direction": "range_bound",
                "price_trend": 22.0,
                "flow_direction": "mixed",
                "flow_persistence": 58.0,
                "dp_direction": "accumulation",
                "dp_confirmation": 80.0,
                "whale_appearances": 20,
                "ticker_playbook_gate_pass": True,
                "earnings_safe": True,
                "live_spread_cost": 4.20,
                "spread_width": 10.0,
                "long_strike": 101.0,
            }
        )

        reasons = trend_analysis.quality_gate_reasons(
            row,
            schwab_enabled=False,
            allow_earnings_risk=False,
            allow_volatile_ic=False,
            allow_flow_conflict=False,
            max_bid_ask_to_price_pct=0.30,
            max_bid_ask_to_width_pct=0.10,
            max_short_delta=0.30,
            min_underlying_price=20.0,
            min_debit_spread_price=0.75,
            min_whale_appearances=8,
        )

        self.assertNotIn("directional price not confirming", reasons)
        self.assertNotIn("weak directional price trend", reasons)
        self.assertNotIn("directional flow not confirming", reasons)
        self.assertNotIn("weak directional flow", reasons)
        self.assertIn("bull_reversal_call_debit", trend_analysis._strategy_family_labels(row))

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

        tracker_summary = {
            "completed": 30,
            "hit_rate": 0.61,
            "avg_pnl": 42.0,
        }

        max_conviction = trend_analysis.build_max_conviction(
            actionable,
            top_n=10,
            tracker_summary=tracker_summary,
        )

        self.assertEqual(list(max_conviction["ticker"]), ["HOOD"])
        self.assertEqual(max_conviction.iloc[0]["position_size_tier"], "MAX_PLANNED_RISK")

    def test_build_max_conviction_blocks_without_tracker_proof(self) -> None:
        actionable = pd.DataFrame(
            [
                {
                    "ticker": "HOOD",
                    "strategy": "Bull Put Credit",
                    "direction": "bullish",
                    "swing_score": 84.0,
                    "days_observed": 5,
                    "price_direction": "bullish",
                    "flow_direction": "bullish",
                    "oi_direction": "bullish",
                    "dp_direction": "accumulation",
                    "whale_appearances": 4,
                    "backtest_verdict": "PASS",
                    "edge_pct": 22.0,
                    "backtest_signals": 140,
                    "quote_replay_verdict": "ENTRY_OK",
                    "live_validated": True,
                    "live_bid_ask_width": 0.05,
                    "live_spread_cost": 1.25,
                    "spread_width": 5.0,
                }
            ]
        )

        max_conviction = trend_analysis.build_max_conviction(actionable, top_n=5)

        self.assertTrue(max_conviction.empty)
        self.assertIn("ticker", max_conviction.columns)

    def test_calibrate_trade_tracking_execution_matches_schwab_spread_fills(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tracker_csv = root / "tracker.csv"
            schwab_csv = root / "schwab.csv"
            tracker = pd.DataFrame(
                [
                    {
                        "trade_id": "2026-04-20|NFLX|bullish|Bull Call Debit|2026-04-24|Buy 90C / Sell 100C",
                        "status": "OPEN_TRACKED",
                        "signal_date": "2026-04-20",
                        "last_seen_as_of": "2026-04-20",
                        "ticker": "NFLX",
                        "direction": "bullish",
                        "strategy": "Bull Call Debit",
                        "variant_tag": "base",
                        "trade_setup": "Bull Call Debit | Buy 90C / Sell 100C | exp 2026-04-24",
                        "target_expiry": "2026-04-24",
                        "long_strike": 90.0,
                        "short_strike": 100.0,
                        "spread_width": 10.0,
                        "cost_type": "debit",
                        "entry_price": 2.00,
                        "max_risk": 200.0,
                        "max_profit": 800.0,
                        "position_size_tier": "STARTER_RISK",
                        "position_size_guidance": "starter",
                        "entry_trigger": "enter on confirmation",
                        "source_report": "report.md",
                        "expected_entry_price": 2.00,
                        "expected_exit_price": 3.00,
                    }
                ]
            )
            for col in trend_analysis.TRACKING_COLUMNS:
                if col not in tracker.columns:
                    tracker[col] = ""
            tracker = tracker[trend_analysis.TRACKING_COLUMNS]
            tracker.to_csv(tracker_csv, index=False)

            schwab_csv.write_text(
                "\n".join(
                    [
                        "Date,Action,Symbol,Quantity,Price,Amount",
                        "2026-04-20,Buy to Open,\"NFLX 4/24/2026 90 C\",1,3.00,($300.00)",
                        "2026-04-20,Sell to Open,\"NFLX 4/24/2026 100 C\",1,1.00,$100.00",
                        "2026-04-23,Sell to Close,\"NFLX 4/24/2026 90 C\",1,5.00,$500.00",
                        "2026-04-23,Buy to Close,\"NFLX 4/24/2026 100 C\",1,2.00,($200.00)",
                    ]
                ),
                encoding="utf-8",
            )

            summary = trend_analysis.calibrate_trade_tracking_execution(
                tracker_csv,
                schwab_transactions_csv=schwab_csv,
                enabled=True,
            )
            out = pd.read_csv(tracker_csv, low_memory=False)

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["matched"], 1)
        self.assertAlmostEqual(float(out.loc[0, "actual_entry_price"]), 2.0)
        self.assertAlmostEqual(float(out.loc[0, "actual_exit_price"]), 3.0)
        self.assertAlmostEqual(float(out.loc[0, "actual_pnl"]), 100.0)
        self.assertEqual(str(out.loc[0, "execution_match_status"]), "closed")

    def test_load_schwab_closed_trade_history_parses_vertical_order_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            db_path = root / "schwab.sqlite"
            con = sqlite3.connect(str(db_path))
            con.execute(
                """
                create table raw_orders (
                    account_hash text,
                    record_key text,
                    order_id text,
                    entered_time text,
                    status text,
                    payload text,
                    pulled_at text
                )
                """
            )
            entry_payload = {
                "orderId": 111,
                "price": 2.15,
                "filledQuantity": 1.0,
                "orderLegCollection": [
                    {
                        "instruction": "BUY_TO_OPEN",
                        "instrument": {"symbol": "NFLX  260320C00085000"},
                    },
                    {
                        "instruction": "SELL_TO_OPEN",
                        "instrument": {"symbol": "NFLX  260320C00095000"},
                    },
                ],
            }
            exit_payload = {
                "orderId": 222,
                "price": 4.75,
                "filledQuantity": 1.0,
                "orderLegCollection": [],
            }
            con.execute(
                "insert into raw_orders values (?,?,?,?,?,?,?)",
                ("acct", "rk1", "111", "", "FILLED", json.dumps(entry_payload), "2026-04-20T00:00:00+00:00"),
            )
            con.execute(
                "insert into raw_orders values (?,?,?,?,?,?,?)",
                ("acct", "rk2", "222", "", "FILLED", json.dumps(exit_payload), "2026-04-24T00:00:00+00:00"),
            )
            con.commit()
            con.close()

            report_json = root / "schwab_positions_2026-04-20.json"
            report_json.write_text(
                json.dumps(
                    {
                        "accounts": [
                            {
                                "closed_trades": [
                                    {
                                        "engine_trade_id": "T1",
                                        "ticker": "NFLX",
                                        "strategy": "vertical_spread",
                                        "expiry": "2026-03-20",
                                        "opened_at": "2026-03-10T14:00:00+00:00",
                                        "closed_at": "2026-03-18T14:00:00+00:00",
                                        "quantity": 1.0,
                                        "realized_pnl": 260.0,
                                        "entry_order_ids": ["111"],
                                        "exit_order_ids": ["222"],
                                        "track": "UNKNOWN",
                                        "source_group": "order:111",
                                    }
                                ],
                                "activity_pull": {"state_db": str(db_path)},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            history, summary = trend_analysis.load_schwab_closed_trade_history(report_json)

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["parsed_closed_trades"], 1)
        self.assertEqual(history.iloc[0]["strategy"], "Bull Call Debit")
        self.assertEqual(history.iloc[0]["direction"], "bullish")
        self.assertEqual(history.iloc[0]["strike_setup"], "Buy 85C / Sell 95C")
        self.assertAlmostEqual(float(history.iloc[0]["spread_width"]), 10.0)
        self.assertAlmostEqual(float(history.iloc[0]["target_dte"]), 10.0)
        self.assertEqual(history.iloc[0]["target_dte_bucket"], "00-14")
        self.assertEqual(history.iloc[0]["spread_width_bucket"], "10w")
        self.assertAlmostEqual(float(history.iloc[0]["entry_net"]), 2.15)
        self.assertAlmostEqual(float(history.iloc[0]["exit_net"]), 4.75)

    def test_annotate_schwab_actual_evidence_marks_supportive_and_negative(self) -> None:
        closed = pd.DataFrame(
            [
                {"ticker": "NFLX", "direction": "bullish", "strategy": "Bull Call Debit", "realized_pnl": 200.0, "hold_days": 7},
                {"ticker": "NFLX", "direction": "bullish", "strategy": "Bull Call Debit", "realized_pnl": 150.0, "hold_days": 6},
                {"ticker": "NFLX", "direction": "bullish", "strategy": "Bull Call Debit", "realized_pnl": 100.0, "hold_days": 5},
                {"ticker": "AAPL", "direction": "bullish", "strategy": "Bull Put Credit", "realized_pnl": -120.0, "hold_days": 10},
                {"ticker": "AAPL", "direction": "bullish", "strategy": "Bull Put Credit", "realized_pnl": -80.0, "hold_days": 8},
            ]
        )
        playbook_audit = trend_analysis.build_schwab_actual_playbook_audit(closed)
        strategy_audit = trend_analysis.build_schwab_actual_strategy_audit(closed)
        candidates = pd.DataFrame(
            [
                {"ticker": "NFLX", "direction": "bullish", "strategy": "Bull Call Debit"},
                {"ticker": "AAPL", "direction": "bullish", "strategy": "Bull Put Credit"},
            ]
        )

        annotated = trend_analysis.annotate_schwab_actual_evidence(
            candidates,
            playbook_audit,
            strategy_audit,
        )

        self.assertTrue(bool(annotated.loc[0, "schwab_actual_playbook_supportive"]))
        self.assertEqual(annotated.loc[0, "schwab_actual_playbook_verdict"], "supportive")
        self.assertEqual(annotated.loc[1, "schwab_actual_playbook_verdict"], "negative")
        self.assertEqual(annotated.loc[1, "schwab_actual_strategy_verdict"], "negative")

    def test_build_schwab_actual_shape_audit_matches_exact_candidate_shape(self) -> None:
        closed = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "option_right": "C",
                    "target_dte": 33,
                    "target_dte_bucket": "29-45",
                    "spread_width": 10.0,
                    "spread_width_bucket": "10w",
                    "cost_type": "debit",
                    "realized_pnl": 220.0,
                    "hold_days": 8,
                },
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "option_right": "C",
                    "target_dte": 35,
                    "target_dte_bucket": "29-45",
                    "spread_width": 10.0,
                    "spread_width_bucket": "10w",
                    "cost_type": "debit",
                    "realized_pnl": 180.0,
                    "hold_days": 7,
                },
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "option_right": "C",
                    "target_dte": 41,
                    "target_dte_bucket": "29-45",
                    "spread_width": 10.0,
                    "spread_width_bucket": "10w",
                    "cost_type": "debit",
                    "realized_pnl": 140.0,
                    "hold_days": 9,
                },
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "option_right": "C",
                    "target_dte": 12,
                    "target_dte_bucket": "00-14",
                    "spread_width": 5.0,
                    "spread_width_bucket": "5w",
                    "cost_type": "debit",
                    "realized_pnl": -75.0,
                    "hold_days": 3,
                },
            ]
        )
        shape_audit = trend_analysis.build_schwab_actual_shape_audit(closed)
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "option_right": "C",
                    "target_dte": 38,
                    "spread_width": 10.0,
                    "cost_type": "debit",
                },
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "option_right": "C",
                    "target_dte": 10,
                    "spread_width": 5.0,
                    "cost_type": "debit",
                },
            ]
        )

        annotated = trend_analysis.annotate_schwab_actual_evidence(
            candidates,
            pd.DataFrame(),
            pd.DataFrame(),
            shape_audit,
        )

        self.assertEqual(annotated.loc[0, "schwab_actual_shape_verdict"], "supportive")
        self.assertTrue(bool(annotated.loc[0, "schwab_actual_shape_supportive"]))
        self.assertIn("DTE 29-45", annotated.loc[0, "schwab_actual_shape_summary"])
        self.assertIn("width 10w", annotated.loc[0, "schwab_actual_shape_summary"])
        self.assertEqual(annotated.loc[1, "schwab_actual_shape_verdict"], "negative_low_sample")

    def test_annotate_schwab_actual_evidence_asof_avoids_future_closed_trades(self) -> None:
        closed = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "realized_pnl": 200.0,
                    "hold_days": 7,
                    "closed_at": "2026-03-10T14:00:00+00:00",
                },
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "realized_pnl": 180.0,
                    "hold_days": 6,
                    "closed_at": "2026-03-20T14:00:00+00:00",
                },
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "realized_pnl": 160.0,
                    "hold_days": 5,
                    "closed_at": "2026-04-05T14:00:00+00:00",
                },
            ]
        )
        candidates = pd.DataFrame(
            [
                {"ticker": "NFLX", "direction": "bullish", "strategy": "Bull Call Debit"},
            ]
        )

        annotated_early = trend_analysis.annotate_schwab_actual_evidence_asof(
            candidates,
            closed,
            as_of=dt.date(2026, 3, 15),
        )
        annotated_late = trend_analysis.annotate_schwab_actual_evidence_asof(
            candidates,
            closed,
            as_of=dt.date(2026, 4, 10),
        )

        self.assertEqual(annotated_early.loc[0, "schwab_actual_playbook_verdict"], "positive_low_sample")
        self.assertIn("1 Schwab closed trade(s)", annotated_early.loc[0, "schwab_actual_playbook_summary"])
        self.assertEqual(annotated_late.loc[0, "schwab_actual_playbook_verdict"], "supportive")
        self.assertIn("3 Schwab closed trade(s)", annotated_late.loc[0, "schwab_actual_playbook_summary"])

    def test_calibrate_trade_tracking_execution_matches_schwab_report_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tracker_csv = root / "tracker.csv"
            db_path = root / "schwab.sqlite"
            con = sqlite3.connect(str(db_path))
            con.execute(
                """
                create table raw_orders (
                    account_hash text,
                    record_key text,
                    order_id text,
                    entered_time text,
                    status text,
                    payload text,
                    pulled_at text
                )
                """
            )
            entry_payload = {
                "orderId": 111,
                "price": 2.00,
                "filledQuantity": 1.0,
                "orderLegCollection": [
                    {"instruction": "BUY_TO_OPEN", "instrument": {"symbol": "NFLX  260424C00090000"}},
                    {"instruction": "SELL_TO_OPEN", "instrument": {"symbol": "NFLX  260424C00100000"}},
                ],
            }
            exit_payload = {"orderId": 222, "price": 3.10, "filledQuantity": 1.0, "orderLegCollection": []}
            con.execute(
                "insert into raw_orders values (?,?,?,?,?,?,?)",
                ("acct", "rk1", "111", "", "FILLED", json.dumps(entry_payload), "2026-04-20T00:00:00+00:00"),
            )
            con.execute(
                "insert into raw_orders values (?,?,?,?,?,?,?)",
                ("acct", "rk2", "222", "", "FILLED", json.dumps(exit_payload), "2026-04-23T00:00:00+00:00"),
            )
            con.commit()
            con.close()

            report_json = root / "schwab_positions_2026-04-20.json"
            report_json.write_text(
                json.dumps(
                    {
                        "accounts": [
                            {
                                "closed_trades": [
                                    {
                                        "engine_trade_id": "T1",
                                        "ticker": "NFLX",
                                        "strategy": "vertical_spread",
                                        "expiry": "2026-04-24",
                                        "opened_at": "2026-04-20T14:00:00+00:00",
                                        "closed_at": "2026-04-23T14:00:00+00:00",
                                        "quantity": 1.0,
                                        "realized_pnl": 110.0,
                                        "entry_order_ids": ["111"],
                                        "exit_order_ids": ["222"],
                                        "track": "UNKNOWN",
                                        "source_group": "order:111",
                                    }
                                ],
                                "activity_pull": {"state_db": str(db_path)},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            tracker = pd.DataFrame(
                [
                    {
                        "trade_id": "id1",
                        "engine_trade_id": "T1",
                        "track": "TREND",
                        "status": "OPEN_TRACKED",
                        "signal_date": "2026-04-20",
                        "last_seen_as_of": "2026-04-20",
                        "ticker": "NFLX",
                        "direction": "bullish",
                        "strategy": "Bull Call Debit",
                        "variant_tag": "base",
                        "trade_setup": "Bull Call Debit | Buy 90C / Sell 100C | exp 2026-04-24",
                        "target_expiry": "2026-04-24",
                        "long_strike": 90.0,
                        "short_strike": 100.0,
                        "spread_width": 10.0,
                        "cost_type": "debit",
                        "entry_price": 2.00,
                        "max_risk": 200.0,
                        "max_profit": 800.0,
                        "position_size_tier": "STARTER_RISK",
                        "position_size_guidance": "starter",
                        "entry_trigger": "enter on confirmation",
                        "source_report": "report.md",
                        "expected_entry_price": 2.00,
                        "expected_exit_price": 3.00,
                    }
                ]
            )
            for col in trend_analysis.TRACKING_COLUMNS:
                if col not in tracker.columns:
                    tracker[col] = ""
            tracker = tracker[trend_analysis.TRACKING_COLUMNS]
            tracker.to_csv(tracker_csv, index=False)

            summary = trend_analysis.calibrate_trade_tracking_execution(
                tracker_csv,
                schwab_transactions_csv=None,
                schwab_report_json=report_json,
                enabled=True,
            )
            out = pd.read_csv(tracker_csv, low_memory=False)

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["matched"], 1)
        self.assertEqual(summary["source"], "schwab_report_json")
        self.assertEqual(summary["exact_engine_matches"], 1)
        self.assertAlmostEqual(float(out.loc[0, "actual_entry_price"]), 2.0)
        self.assertAlmostEqual(float(out.loc[0, "actual_exit_price"]), 3.1)
        self.assertAlmostEqual(float(out.loc[0, "actual_pnl"]), 110.0)
        self.assertEqual(str(out.loc[0, "execution_match_status"]), "closed")
        self.assertEqual(str(out.loc[0, "execution_match_trade_id"]), "T1")

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

    def test_tactical_probe_candidates_surface_live_quote_positive_tickets(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "WULF",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "target_expiry": "2026-05-01",
                    "live_strike_setup": "Buy 20C / Sell 22.5C (2.5w, $0.80 debit)",
                    "live_validated": True,
                    "swing_score": 60.2,
                    "edge_pct": 23.8,
                    "backtest_signals": 86,
                    "backtest_verdict": "LOW_SAMPLE",
                    "base_gate_pass": False,
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "base_gate_reasons": (
                        "backtest LOW_SAMPLE, edge 23.8%, signals 86; "
                        "market regime conflict: risk_off conflict for bullish Bull Call Debit; "
                        "batch proof playbook gate blocked: no supportive prior-only batch-proof playbook match"
                    ),
                    "actionability_reject_reasons": (
                        "backtest LOW_SAMPLE, edge 23.8%, signals 86; "
                        "market regime conflict: risk_off conflict for bullish Bull Call Debit; "
                        "batch proof playbook gate blocked: no supportive prior-only batch-proof playbook match"
                    ),
                    "quality_reject_reasons": "",
                },
                {
                    "ticker": "NVDA",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "target_expiry": "2026-05-29",
                    "live_strike_setup": "Buy 190C / Sell 200C",
                    "live_validated": True,
                    "swing_score": 62.0,
                    "edge_pct": 1.2,
                    "backtest_signals": 76,
                    "backtest_verdict": "LOW_SAMPLE",
                    "base_gate_pass": False,
                    "quality_gate_pass": False,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "actionability_reject_reasons": "open position conflict: existing option exposure in NVDA",
                    "quality_reject_reasons": "",
                },
            ]
        )

        probes = trend_analysis.build_tactical_probe_candidates(
            candidates,
            top_n=5,
            min_edge=0.0,
            min_signals=100,
            min_workup_signals=50,
            min_swing_score=60.0,
        )

        self.assertEqual(list(probes["ticker"]), ["WULF"])
        self.assertEqual(probes.iloc[0]["tactical_probe_status"], "TACTICAL PROBE")
        self.assertIn("max 0.25R", probes.iloc[0]["tactical_probe_what_to_do"])
        self.assertIn("LOW_SAMPLE", probes.iloc[0]["tactical_probe_why"])

    def test_final_ticket_sheet_prints_tactical_probe_exact_legs(self) -> None:
        tactical = pd.DataFrame(
            [
                {
                    "ticker": "WULF",
                    "strategy": "Bull Call Debit",
                    "direction": "bullish",
                    "target_expiry": "2026-05-01",
                    "live_strike_setup": "Buy 20C / Sell 22.5C (2.5w, $0.80 debit)",
                    "live_validated": True,
                    "latest_close": 20.85,
                    "live_long_strike": 20.0,
                    "live_short_strike": 22.5,
                    "spread_width": 2.5,
                    "flow_persistence": 64.0,
                    "price_trend": 63.0,
                    "backtest_verdict": "LOW_SAMPLE",
                    "edge_pct": 23.8,
                    "backtest_signals": 86,
                    "actionability_reject_reasons": (
                        "market regime conflict: risk_off conflict for bullish Bull Call Debit; "
                        "batch proof playbook gate blocked: no supportive prior-only batch-proof playbook match"
                    ),
                    "tactical_probe_status": "TACTICAL PROBE",
                    "tactical_probe_what_to_do": "Smallest defined-risk probe only, max 0.25R.",
                    "tactical_probe_why": (
                        "needs more analog history: LOW_SAMPLE, edge 23.8%, signals 86 < actionable min 100; "
                        "tactical blockers: batch proof playbook gate blocked"
                    ),
                    "tactical_probe_next": "Reprice both legs live.",
                }
            ]
        )

        lines = trend_analysis._final_trade_ticket_lines(
            proven_tickets=pd.DataFrame(),
            current_setups=pd.DataFrame(),
            tactical_probes=tactical,
        )
        text = "\n".join(lines)

        self.assertIn("WULF - `TACTICAL PROBE`", text)
        self.assertIn("Bull Call Debit | Buy 20C / Sell 22.5C", text)
        self.assertIn("Because prior-only playbook proof is not supportive yet", text)
        self.assertIn("Market regime must clear risk-off", text)

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

    def test_current_setup_pool_falls_back_when_no_proven_lane_matches(self) -> None:
        annotated = pd.DataFrame(
            [
                {
                    "ticker": "NBIS",
                    "strategy": "Bull Put Credit",
                    "direction": "bullish",
                    "swing_score": 61.2,
                    "edge_pct": 4.7,
                    "backtest_signals": 65,
                    "backtest_verdict": "LOW_SAMPLE",
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                }
            ]
        )

        pool = trend_analysis._current_setup_reporting_pool(annotated, pd.DataFrame())
        setups = trend_analysis.build_current_trade_setups(
            pool,
            top_n=10,
            min_edge=0.0,
            min_signals=100,
            min_workup_signals=50,
            min_swing_score=60.0,
        )

        self.assertEqual(list(setups["ticker"]), ["NBIS"])
        self.assertEqual(setups.iloc[0]["setup_tier"], "TRADE_SETUP")

    def test_current_setup_pool_does_not_hide_non_proven_workbench_rows(self) -> None:
        annotated = pd.DataFrame(
            [
                {
                    "ticker": "NVDA",
                    "strategy": "Bull Call Debit",
                    "batch_playbook_verdict": "insufficient_forward",
                    "schwab_actual_playbook_supportive": True,
                },
                {
                    "ticker": "WULF",
                    "strategy": "Bull Call Debit",
                    "batch_playbook_verdict": "no_match",
                    "schwab_actual_playbook_supportive": False,
                },
            ]
        )
        proven = annotated.iloc[[0]].copy()

        pool = trend_analysis._current_setup_reporting_pool(annotated, proven)

        self.assertEqual(set(pool["ticker"]), {"NVDA", "WULF"})

    def test_current_trade_setups_do_not_surface_backtest_fail_as_research_blocked(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "NKE",
                    "strategy": "Bear Put Debit",
                    "direction": "bearish",
                    "swing_score": 62.4,
                    "latest_close": 44.33,
                    "live_long_strike": 45.0,
                    "long_strike": 45.0,
                    "spread_width": 5.0,
                    "edge_pct": -14.0,
                    "backtest_signals": 57,
                    "backtest_verdict": "FAIL",
                    "quality_gate_pass": True,
                    "schwab_gate_pass": True,
                    "quote_replay_gate_pass": True,
                    "actionability_reject_reasons": "strategy family audit not promotable: no matching family (no_audit)",
                }
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

        self.assertTrue(setups.empty)

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

    def test_strategy_family_audit_can_promote_bullish_reversal_family(self) -> None:
        rows = []
        for setup_idx in range(30):
            signal_day = dt.date(2026, 1, 1) + dt.timedelta(days=setup_idx)
            rows.append(
                {
                    "policy": "entry_available_score_gate",
                    "signal_date": signal_day.isoformat(),
                    "horizon_market_days": 10,
                    "ticker": f"R{setup_idx}",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": f"reversal-{setup_idx}",
                    "swing_score": 63.0,
                    "price_direction": "range_bound" if setup_idx % 2 == 0 else "bearish",
                    "price_trend": 24.0,
                    "flow_direction": "bullish" if setup_idx % 3 else "mixed",
                    "flow_persistence": 58.0,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 80.0,
                    "whale_appearances": 16,
                    "pnl": 180.0 if setup_idx < 27 else 90.0,
                    "return_on_risk": 0.22,
                }
            )

        audit = trend_analysis._strategy_family_audit_from_outcomes(pd.DataFrame(rows))
        row = audit[audit["family"].eq("bull_reversal_call_debit")].iloc[0]

        self.assertEqual(row["verdict"], "promotable")
        self.assertEqual(int(row["validation_unique_setups"]), 9)

    def test_rolling_strategy_family_audit_supports_reversal_family(self) -> None:
        rows = []
        for setup_idx in range(30):
            signal_day = dt.date(2026, 1, 1) + dt.timedelta(days=setup_idx)
            rows.append(
                {
                    "policy": "entry_available_score_gate",
                    "signal_date": signal_day.isoformat(),
                    "horizon_market_days": 10,
                    "ticker": f"R{setup_idx}",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": f"reversal-{setup_idx}",
                    "swing_score": 63.0,
                    "price_direction": "range_bound" if setup_idx % 2 == 0 else "bearish",
                    "price_trend": 24.0,
                    "flow_direction": "bullish" if setup_idx % 3 else "mixed",
                    "flow_persistence": 58.0,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 80.0,
                    "whale_appearances": 16,
                    "pnl": 180.0 if setup_idx < 27 else 90.0,
                    "return_on_risk": 0.22,
                }
            )

        outcomes = pd.DataFrame(rows)
        audit = trend_analysis._strategy_family_audit_from_outcomes(outcomes)
        rolling = trend_analysis._rolling_strategy_family_audit_from_outcomes(outcomes, audit)
        row = rolling[rolling["family"].eq("bull_reversal_call_debit")].iloc[0]

        self.assertIn(row["verdict"], {"supportive", "emerging_forward"})
        self.assertGreater(int(row["forward_tests"]), 0)

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

    def test_research_gate_passes_with_supportive_batch_family(self) -> None:
        row = pd.Series(
            {
                "batch_playbook_gate_pass": False,
                "batch_family_gate_pass": True,
                "strategy_family_gate_pass": False,
                "ticker_playbook_gate_pass": False,
                "rolling_playbook_gate_pass": True,
                "schwab_actual_playbook_supportive": False,
            }
        )

        self.assertTrue(trend_analysis.research_gate_passes(row))

    def test_research_gate_passes_with_supportive_schwab_shape(self) -> None:
        row = pd.Series(
            {
                "batch_playbook_gate_pass": False,
                "batch_family_gate_pass": False,
                "strategy_family_gate_pass": False,
                "ticker_playbook_gate_pass": False,
                "rolling_playbook_gate_pass": True,
                "schwab_actual_playbook_supportive": False,
                "schwab_actual_shape_supportive": True,
            }
        )

        self.assertTrue(trend_analysis.research_gate_passes(row))

    def test_batch_no_match_does_not_veto_broker_supported_base_gate(self) -> None:
        row = pd.Series(
            {
                "ticker": "AAPL",
                "strategy": "Bull Call Debit",
                "direction": "bullish",
                "swing_score": 64.0,
                "backtest_verdict": "LOW_SAMPLE",
                "edge_pct": 8.0,
                "backtest_signals": 66,
                "live_validated": True,
                "quote_replay_verdict": "ENTRY_OK",
                "market_regime_gate_pass": True,
                "open_position_gate_pass": True,
                "batch_playbook_gate_pass": False,
                "batch_playbook_verdict": "insufficient_forward",
                "batch_playbook_summary": "rolling batch proof insufficient_forward, avg -",
                "batch_family_gate_pass": False,
                "schwab_actual_playbook_supportive": True,
                "schwab_actual_shape_supportive": False,
            }
        )

        reasons = trend_analysis.base_gate_reasons(
            row,
            backtest_enabled=True,
            schwab_enabled=True,
            quote_replay_mode="gate",
            min_edge=0.0,
            min_signals=100,
            min_swing_score=60.0,
            allow_low_sample=False,
        )

        self.assertFalse(any("batch proof playbook gate blocked" in reason for reason in reasons))

    def test_negative_batch_playbook_still_blocks_broker_supported_base_gate(self) -> None:
        row = pd.Series(
            {
                "ticker": "CVX",
                "strategy": "Bull Call Debit",
                "direction": "bullish",
                "swing_score": 64.0,
                "backtest_verdict": "LOW_SAMPLE",
                "edge_pct": 8.0,
                "backtest_signals": 66,
                "live_validated": True,
                "quote_replay_verdict": "ENTRY_OK",
                "market_regime_gate_pass": True,
                "open_position_gate_pass": True,
                "batch_playbook_gate_pass": False,
                "batch_playbook_verdict": "negative",
                "batch_playbook_summary": "rolling batch proof negative, avg $-96.60",
                "batch_family_gate_pass": False,
                "schwab_actual_playbook_supportive": True,
                "schwab_actual_shape_supportive": False,
            }
        )

        reasons = trend_analysis.base_gate_reasons(
            row,
            backtest_enabled=True,
            schwab_enabled=True,
            quote_replay_mode="gate",
            min_edge=0.0,
            min_signals=100,
            min_swing_score=60.0,
            allow_low_sample=False,
        )

        self.assertTrue(any("batch proof playbook gate blocked" in reason for reason in reasons))

    def test_batch_playbook_support_counts_as_historical_support(self) -> None:
        row = pd.Series(
            {
                "batch_playbook_gate_pass": True,
                "backtest_verdict": "UNKNOWN",
                "edge_pct": float("nan"),
                "backtest_signals": 0,
            }
        )

        self.assertTrue(
            trend_analysis.historical_support_passes(
                row,
                min_edge=0.0,
                min_signals=100,
                allow_low_sample=False,
            )
        )

    def test_batch_playbook_support_does_not_override_backtest_fail(self) -> None:
        row = pd.Series(
            {
                "batch_playbook_gate_pass": True,
                "backtest_verdict": "FAIL",
                "edge_pct": -8.0,
                "backtest_signals": 140,
            }
        )

        self.assertFalse(
            trend_analysis.historical_support_passes(
                row,
                min_edge=0.0,
                min_signals=100,
                allow_low_sample=False,
            )
        )

    def test_exact_batch_playbook_overrides_broad_schwab_strategy_negative(self) -> None:
        row = pd.Series(
            {
                "ticker": "NFLX",
                "strategy": "Bull Call Debit",
                "direction": "bullish",
                "latest_close": 100.0,
                "whale_appearances": 30,
                "price_direction": "bullish",
                "price_trend": 65.0,
                "flow_direction": "bullish",
                "flow_persistence": 65.0,
                "spread_width": 10.0,
                "live_spread_cost": 3.00,
                "long_strike": 100.0,
                "earnings_safe": True,
                "live_bid_ask_width": 0.10,
                "schwab_actual_strategy_verdict": "negative",
                "schwab_actual_strategy_summary": "23 Schwab Bull Call Debit trade(s), avg -$12.04",
                "batch_playbook_gate_pass": True,
            }
        )

        reasons = trend_analysis.quality_gate_reasons(
            row,
            schwab_enabled=True,
            allow_earnings_risk=False,
            allow_volatile_ic=False,
            allow_flow_conflict=False,
            max_bid_ask_to_price_pct=0.35,
            max_bid_ask_to_width_pct=0.15,
            max_short_delta=0.30,
            min_underlying_price=20.0,
            min_debit_spread_price=0.75,
            min_whale_appearances=8,
        )

        self.assertFalse(any("actual Schwab strategy audit negative" in reason for reason in reasons))

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

    def test_ticker_playbook_audit_dedupes_same_day_variants_without_hindsight(self) -> None:
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
                    "trade_setup": f"higher-score setup {setup_idx}",
                    "swing_score": 80.0,
                    "edge_pct": 10.0,
                    "backtest_signals": 120,
                    "backtest_verdict": "PASS",
                    "pnl": -100.0,
                }
            )
            rows.append(
                {
                    "policy": "entry_available_score_gate",
                    "signal_date": signal_day.isoformat(),
                    "horizon_market_days": 20,
                    "ticker": "NVDA",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": f"lower-score winner {setup_idx}",
                    "swing_score": 60.0,
                    "edge_pct": 0.0,
                    "backtest_signals": 0,
                    "backtest_verdict": "UNKNOWN",
                    "pnl": 500.0,
                }
            )

        audit = trend_analysis._ticker_playbook_audit_from_outcomes(pd.DataFrame(rows))
        row = audit[audit["ticker"].eq("NVDA")].iloc[0]

        self.assertEqual(int(row["overall_outcomes"]), 20)
        self.assertEqual(float(row["overall_avg_pnl"]), -100.0)
        self.assertEqual(row["verdict"], "validation_negative")

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

    def test_rolling_ticker_playbook_forward_validation_flags_negative(self) -> None:
        rows = []
        for setup_idx in range(21):
            signal_day = dt.date(2026, 1, 1) + dt.timedelta(days=setup_idx)
            rows.append(
                {
                    "policy": "entry_available_score_gate",
                    "signal_date": signal_day.isoformat(),
                    "horizon_market_days": 20,
                    "ticker": "XYZ",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": f"XYZ setup {setup_idx}",
                    "pnl": 100.0 if setup_idx < 16 else -150.0,
                    "return_on_risk": 0.20,
                }
            )
        playbook_audit = pd.DataFrame(
            [
                {
                    "ticker": "XYZ",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "horizon_market_days": 20,
                    "verdict": "promotable",
                }
            ]
        )

        rolling = trend_analysis._rolling_ticker_playbook_audit_from_outcomes(
            pd.DataFrame(rows),
            playbook_audit,
        )
        row = rolling[rolling["ticker"].eq("XYZ")].iloc[0]

        self.assertEqual(row["verdict"], "negative")
        self.assertGreaterEqual(int(row["forward_tests"]), 3)
        self.assertLess(float(row["forward_avg_pnl"]), 0)

    def test_rolling_ticker_playbook_forward_validation_flags_recent_decay(self) -> None:
        rows = []
        for setup_idx in range(28):
            signal_day = dt.date(2026, 1, 1) + dt.timedelta(days=setup_idx)
            if setup_idx < 20:
                pnl = 100.0
            elif setup_idx < 23:
                pnl = 1000.0
            else:
                pnl = -200.0
            rows.append(
                {
                    "policy": "entry_available_score_gate",
                    "signal_date": signal_day.isoformat(),
                    "horizon_market_days": 20,
                    "ticker": "CVX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": f"CVX setup {setup_idx}",
                    "pnl": pnl,
                    "return_on_risk": 0.20,
                }
            )
        playbook_audit = pd.DataFrame(
            [
                {
                    "ticker": "CVX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "horizon_market_days": 20,
                    "verdict": "promotable",
                }
            ]
        )

        rolling = trend_analysis._rolling_ticker_playbook_audit_from_outcomes(
            pd.DataFrame(rows),
            playbook_audit,
        )
        row = rolling[rolling["ticker"].eq("CVX")].iloc[0]
        candidate = pd.DataFrame(
            [
                {
                    "ticker": "CVX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "ticker_playbook_horizon": 20,
                }
            ]
        )
        annotated = trend_analysis.annotate_rolling_playbook_gate(candidate, rolling)

        self.assertEqual(row["verdict"], "decaying")
        self.assertLess(float(row["recent_forward_avg_pnl"]), 0)
        self.assertFalse(bool(annotated.iloc[0]["rolling_playbook_gate_pass"]))

    def test_annotate_batch_family_gate_marks_supportive_match(self) -> None:
        candidate = pd.DataFrame(
            [
                {
                    "ticker": "NVDA",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "swing_score": 62.7,
                    "price_direction": "range_bound",
                    "price_trend": 24.0,
                    "flow_direction": "bullish",
                    "flow_persistence": 58.0,
                    "dp_direction": "accumulation",
                    "dp_confirmation": 80.0,
                    "whale_appearances": 16,
                }
            ]
        )
        proof_gate = {
            "enabled": True,
            "supported_families": {
                "bull_reversal_call_debit": {
                    "family": "bull_reversal_call_debit",
                    "horizon_market_days": 20,
                    "forward_tests": 4,
                    "forward_dates": 4,
                    "forward_hit_rate": 0.75,
                    "forward_avg_pnl": 210.0,
                    "forward_profit_factor": 2.1,
                    "recent_forward_tests": 4,
                    "recent_forward_hit_rate": 0.75,
                    "recent_forward_avg_pnl": 210.0,
                    "recent_forward_profit_factor": 2.1,
                    "first_forward_date": "2026-03-12",
                    "last_forward_date": "2026-03-23",
                    "verdict": "supportive",
                }
            },
            "blocked_families": {},
        }

        annotated = trend_analysis.annotate_batch_family_gate(candidate, proof_gate)

        self.assertEqual(annotated.iloc[0]["batch_family_verdict"], "supportive")
        self.assertTrue(bool(annotated.iloc[0]["batch_family_gate_pass"]))
        self.assertIn("4 forward tests", annotated.iloc[0]["batch_family_summary"])

    def test_regime_filter_blocks_bullish_debit_in_risk_off(self) -> None:
        candidate = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "swing_score": 70.0,
                    "backtest_verdict": "PASS",
                    "edge_pct": 8.0,
                    "backtest_signals": 150,
                }
            ]
        )

        annotated = trend_analysis.annotate_regime_filter(
            candidate,
            {"regime": "risk_off", "reason": "breadth weak"},
        )
        actionable, patterns = trend_analysis.split_actionable_candidates(
            annotated,
            top_n=1,
            backtest_enabled=True,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
            schwab_enabled=False,
            quote_replay_mode="off",
            min_swing_score=60.0,
        )

        self.assertTrue(actionable.empty)
        self.assertIn("market regime conflict", patterns.iloc[0]["base_gate_reasons"])

    def test_historical_regime_filter_uses_signal_date_screener(self) -> None:
        candidate = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day = dt.date(2026, 3, 24)
            day_dir = root / day.isoformat()
            day_dir.mkdir()
            (day_dir / "stock-screener.csv").write_text(
                "\n".join(
                    [
                        "ticker,close,prev_close,issue_type,put_call_ratio,iv_rank",
                        "SPY,99,100,ETF,1.30,40",
                        "QQQ,98,100,ETF,1.20,45",
                        "AAA,9,10,common stock,,",
                        "BBB,8,10,common stock,,",
                        "CCC,12,10,common stock,,",
                    ]
                ),
                encoding="utf-8",
            )

            annotated, regime = trend_analysis.annotate_historical_regime_filter(
                candidate,
                root=root,
                all_days=[(day, day_dir)],
                signal_date=day,
                lookback=30,
            )

        self.assertEqual(regime["regime"], "risk_off")
        self.assertFalse(bool(annotated.iloc[0]["market_regime_gate_pass"]))
        self.assertIn("risk_off conflict", annotated.iloc[0]["market_regime_summary"])

    def test_open_position_awareness_blocks_existing_underlying(self) -> None:
        candidate = pd.DataFrame(
            [
                {
                    "ticker": "NVDA",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "swing_score": 70.0,
                    "backtest_verdict": "PASS",
                    "edge_pct": 8.0,
                    "backtest_signals": 150,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            position_json = Path(td) / "position_data_2026-04-17.json"
            position_json.write_text(
                '{"positions":[{"asset_type":"OPTION","underlying":"NVDA","symbol":"NVDA 260515C00200000","put_call":"CALL","qty":1}]}',
                encoding="utf-8",
            )
            annotated, summary = trend_analysis.annotate_open_position_awareness(candidate, position_json)

        actionable, patterns = trend_analysis.split_actionable_candidates(
            annotated,
            top_n=1,
            backtest_enabled=True,
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
            schwab_enabled=False,
            quote_replay_mode="off",
            min_swing_score=60.0,
        )

        self.assertEqual(summary["open_underlyings"], 1)
        self.assertTrue(actionable.empty)
        self.assertIn("open position conflict", patterns.iloc[0]["base_gate_reasons"])

    def test_position_sizing_starter_for_low_sample_playbook(self) -> None:
        actionable = pd.DataFrame(
            [
                {
                    "ticker": "NBIS",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "swing_score": 69.0,
                    "backtest_verdict": "LOW_SAMPLE",
                    "edge_pct": 29.0,
                    "backtest_signals": 64,
                    "ticker_playbook_gate_pass": True,
                    "ticker_playbook_validation_setups": 7,
                    "rolling_playbook_verdict": "insufficient_forward",
                }
            ]
        )

        sized = trend_analysis.annotate_position_sizing(actionable)

        self.assertEqual(sized.iloc[0]["position_size_tier"], "STARTER_RISK")
        self.assertEqual(float(sized.iloc[0]["max_planned_risk_units"]), 0.25)
        self.assertIn("Starter", sized.iloc[0]["position_size_guidance"])

    def test_trade_tracking_appends_unique_actionable_trade(self) -> None:
        actionable = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "variant_tag": "base",
                    "strike_setup": "Buy 200C / Sell 210C",
                    "target_expiry": "2026-05-15",
                    "live_spread_cost": 3.0,
                    "spread_width": 10.0,
                    "cost_type": "debit",
                    "position_size_tier": "STARTER_RISK",
                    "position_size_guidance": "Starter only",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            tracking_csv = Path(td) / "tracker.csv"
            report_path = Path(td) / "report.md"
            first = trend_analysis.update_trade_tracking(
                actionable,
                tracking_csv,
                report_path=report_path,
                as_of=dt.date(2026, 4, 17),
                enabled=True,
            )
            second = trend_analysis.update_trade_tracking(
                actionable,
                tracking_csv,
                report_path=report_path,
                as_of=dt.date(2026, 4, 17),
                enabled=True,
            )
            tracker = pd.read_csv(tracking_csv)

        self.assertEqual(first["added"], 1)
        self.assertEqual(second["added"], 0)
        self.assertEqual(len(first["new_engine_trade_ids"]), 1)
        self.assertEqual(len(tracker), 1)
        self.assertEqual(tracker.iloc[0]["ticker"], "AAPL")
        self.assertTrue(str(tracker.iloc[0]["engine_trade_id"]).startswith("TREND-AAPL-20260417-"))
        self.assertEqual(str(tracker.iloc[0]["track"]), "TREND")
        self.assertEqual(str(tracker.iloc[0]["registration_status"]), "PENDING_REGISTRATION")
        intended_legs = json.loads(tracker.iloc[0]["intended_legs_json"])
        self.assertEqual(len(intended_legs), 2)
        self.assertEqual(intended_legs[0]["instruction"], "BUY_TO_OPEN")
        self.assertEqual(intended_legs[1]["instruction"], "SELL_TO_OPEN")

    def test_register_tracked_trade_open_persists_metadata_and_tracker_linkage(self) -> None:
        actionable = pd.DataFrame(
            [
                {
                    "ticker": "NFLX",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "variant_tag": "base",
                    "strike_setup": "Buy 90C / Sell 100C",
                    "target_expiry": "2026-05-15",
                    "live_spread_cost": 2.5,
                    "spread_width": 10.0,
                    "cost_type": "debit",
                    "position_size_tier": "STARTER_RISK",
                    "position_size_guidance": "Starter only",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            tracking_csv = root / "tracker.csv"
            report_path = root / "report.md"
            state_dir = root / "state"
            state_db = root / "state.sqlite"
            trend_analysis.update_trade_tracking(
                actionable,
                tracking_csv,
                report_path=report_path,
                as_of=dt.date(2026, 4, 20),
                enabled=True,
            )
            tracker = pd.read_csv(tracking_csv, low_memory=False)
            engine_trade_id = str(tracker.loc[0, "engine_trade_id"])

            summary = trend_analysis.register_tracked_trade_open(
                tracking_csv,
                trade_ref=engine_trade_id,
                broker_order_id="123456789",
                account_hash="acct-1",
                opened_at="2026-04-20T10:15:00-07:00",
                state_dir=state_dir,
                state_db=state_db,
            )
            out = pd.read_csv(tracking_csv, low_memory=False)

            con = sqlite3.connect(str(state_db))
            row = con.execute(
                "select engine_trade_id, broker_order_id, track, ticker, strategy, expiry, intended_legs from open_trade_metadata"
            ).fetchone()
            con.close()

        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["engine_trade_id"], engine_trade_id)
        self.assertEqual(str(out.loc[0, "broker_order_id"]), "123456789")
        self.assertEqual(str(out.loc[0, "registration_status"]), "REGISTERED")
        self.assertEqual(row[0], engine_trade_id)
        self.assertEqual(row[1], "123456789")
        self.assertEqual(row[2], "TREND")
        self.assertEqual(row[3], "NFLX")
        self.assertEqual(row[4], "Bull Call Debit")
        intended = json.loads(row[6])
        self.assertEqual(len(intended), 2)
        self.assertEqual(intended[0]["instruction"], "BUY_TO_OPEN")

    def test_trade_tracking_refreshes_outcomes_from_replay(self) -> None:
        actionable = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "variant_tag": "base",
                    "strike_setup": "Buy 200C / Sell 210C",
                    "target_expiry": "2026-05-15",
                    "live_spread_cost": 3.0,
                    "spread_width": 10.0,
                    "cost_type": "debit",
                    "position_size_tier": "STARTER_RISK",
                    "position_size_guidance": "Starter only",
                }
            ]
        )

        def fake_replay(candidates: pd.DataFrame, **_: object):
            annotated = candidates.copy()
            annotated["quote_replay_exit_net"] = 6.0
            annotated["quote_replay_entry_net"] = 3.0
            annotated["quote_replay_exit_date"] = "2026-04-10"
            annotated["quote_replay_final"] = False
            annotated["quote_replay_status"] = "completed"
            annotated["quote_replay_reason"] = "ok"
            annotated["quote_replay_days_held"] = 9
            return annotated, pd.DataFrame()

        with tempfile.TemporaryDirectory() as td:
            tracking_csv = Path(td) / "tracker.csv"
            report_path = Path(td) / "report.md"
            trend_analysis.update_trade_tracking(
                actionable,
                tracking_csv,
                report_path=report_path,
                as_of=dt.date(2026, 4, 1),
                enabled=True,
            )
            summary = trend_analysis.refresh_trade_tracking_outcomes(
                tracking_csv,
                root=Path(td),
                as_of=dt.date(2026, 4, 10),
                enabled=True,
                replay_fn=fake_replay,
            )
            tracker = pd.read_csv(tracking_csv)

        self.assertEqual(summary["updated"], 1)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(tracker.iloc[0]["status"], "OPEN_WIN")
        self.assertEqual(tracker.iloc[0]["outcome_verdict"], "PARTIAL_WIN")
        self.assertAlmostEqual(float(tracker.iloc[0]["outcome_pnl"]), 300.0)


if __name__ == "__main__":
    unittest.main()
