import datetime as dt
import math
import unittest

import pandas as pd

from uwos import trend_analysis_v2 as v2


class TrendAnalysisV2Tests(unittest.TestCase):
    def test_parse_occ_symbol(self):
        parsed = v2.parse_occ("AAPL260515C00190000")

        self.assertIsNotNone(parsed)
        ticker, expiry, right, strike = parsed
        self.assertEqual(ticker, "AAPL")
        self.assertEqual(expiry, dt.date(2026, 5, 15))
        self.assertEqual(right, "C")
        self.assertEqual(strike, 190.0)

    def test_build_debit_structure_selects_tradeable_bull_call(self):
        as_of = dt.date(2026, 5, 1)
        expiry = dt.date(2026, 5, 29)
        options = pd.DataFrame(
            [
                {
                    "ticker": "XYZ",
                    "expiry": expiry,
                    "dte": 28,
                    "right": "C",
                    "strike": 100.0,
                    "option_symbol": "XYZ260529C00100000",
                    "bid": 5.00,
                    "ask": 5.20,
                    "volume": 250,
                    "open_interest": 1200,
                },
                {
                    "ticker": "XYZ",
                    "expiry": expiry,
                    "dte": 28,
                    "right": "C",
                    "strike": 110.0,
                    "option_symbol": "XYZ260529C00110000",
                    "bid": 2.10,
                    "ask": 2.30,
                    "volume": 200,
                    "open_interest": 900,
                },
            ]
        )

        structure, blockers = v2.build_debit_structure(
            options,
            ticker="XYZ",
            spot=101.0,
            direction="bullish",
            as_of=as_of,
        )

        self.assertEqual(blockers, [])
        self.assertIsNotNone(structure)
        self.assertEqual(structure.strategy, "Bull Call Debit")
        self.assertEqual(structure.long_symbol, "XYZ260529C00100000")
        self.assertEqual(structure.short_symbol, "XYZ260529C00110000")
        self.assertAlmostEqual(structure.entry_net, 3.10)
        self.assertEqual(structure.quote_sanity, "ok")

    def test_build_debit_structure_allows_liquid_weekly_momentum_spread(self):
        as_of = dt.date(2026, 5, 1)
        expiry = dt.date(2026, 5, 8)
        options = pd.DataFrame(
            [
                {
                    "ticker": "XYZ",
                    "expiry": expiry,
                    "dte": 7,
                    "right": "C",
                    "strike": 100.0,
                    "option_symbol": "XYZ260508C00100000",
                    "bid": 3.10,
                    "ask": 3.30,
                    "volume": 500,
                    "open_interest": 1500,
                },
                {
                    "ticker": "XYZ",
                    "expiry": expiry,
                    "dte": 7,
                    "right": "C",
                    "strike": 105.0,
                    "option_symbol": "XYZ260508C00105000",
                    "bid": 1.20,
                    "ask": 1.35,
                    "volume": 400,
                    "open_interest": 1000,
                },
            ]
        )

        structure, blockers = v2.build_debit_structure(
            options,
            ticker="XYZ",
            spot=101.0,
            direction="bullish",
            as_of=as_of,
        )

        self.assertEqual(blockers, [])
        self.assertIsNotNone(structure)
        self.assertEqual(structure.expiry, expiry)
        self.assertAlmostEqual(structure.entry_net, 2.10)

    def test_constructive_tape_with_high_index_pcr_is_not_risk_off(self):
        as_of = dt.date(2026, 5, 6)

        class Cache:
            def screener(self, day):
                rows = [
                    {
                        "ticker": "SPY",
                        "close": 110.0 if day == as_of else 100.0,
                        "prev_close": 109.0,
                        "put_call_ratio": 1.5,
                        "issue_type": "ETF",
                    },
                    {
                        "ticker": "QQQ",
                        "close": 112.0 if day == as_of else 100.0,
                        "prev_close": 111.0,
                        "put_call_ratio": 1.4,
                        "issue_type": "ETF",
                    },
                    {
                        "ticker": "IWM",
                        "close": 105.0 if day == as_of else 100.0,
                        "prev_close": 104.0,
                        "put_call_ratio": 1.2,
                        "issue_type": "ETF",
                    },
                    {
                        "ticker": "VIX",
                        "close": 16.0,
                        "prev_close": 17.0,
                        "issue_type": "Index",
                    },
                ]
                for i in range(8):
                    rows.append(
                        {
                            "ticker": f"UP{i}",
                            "close": 50.0 + i,
                            "prev_close": 49.0 + i,
                            "issue_type": "Common Stock",
                            "sector": "Technology",
                        }
                    )
                for i in range(2):
                    rows.append(
                        {
                            "ticker": f"DN{i}",
                            "close": 40.0 + i,
                            "prev_close": 41.0 + i,
                            "issue_type": "Common Stock",
                            "sector": "Industrials",
                        }
                    )
                return pd.DataFrame(rows)

        summary, _ = v2.market_regime_summary(
            Cache(),
            [
                dt.date(2026, 4, 28),
                dt.date(2026, 4, 29),
                dt.date(2026, 4, 30),
                dt.date(2026, 5, 1),
                dt.date(2026, 5, 4),
                as_of,
            ],
            as_of,
            lookback=6,
        )

        self.assertEqual(summary["regime"], "risk_on")

    def test_score_candidate_outcomes_uses_later_option_quotes_before_intrinsic_proxy(self):
        signal = dt.date(2026, 5, 1)
        exit_day = dt.date(2026, 5, 4)
        expiry = dt.date(2026, 5, 29)
        rows = pd.DataFrame(
            [
                {
                    "as_of": signal.isoformat(),
                    "ticker": "XYZ",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": "Buy 100C / Sell 105C",
                    "expiry": expiry.isoformat(),
                    "long_symbol": "XYZ260529C00100000",
                    "short_symbol": "XYZ260529C00105000",
                    "long_strike": 100.0,
                    "short_strike": 105.0,
                    "width": 5.0,
                    "entry_net": 1.00,
                    "max_risk": 100.0,
                }
            ]
        )

        class Cache:
            def option_snapshot(self, day):
                if day != exit_day:
                    return pd.DataFrame()
                return pd.DataFrame(
                    [
                        {"option_symbol": "XYZ260529C00100000", "bid": 3.00, "ask": 3.20},
                        {"option_symbol": "XYZ260529C00105000", "bid": 0.90, "ask": 1.00},
                    ]
                )

            def screener(self, day):
                return pd.DataFrame([{"ticker": "XYZ", "close": 101.0}])

        outcomes = v2.score_candidate_outcomes(
            rows,
            cache=Cache(),
            all_days=[signal, exit_day],
            signal_date=signal,
            horizons=[1],
            baseline="trend_v2",
            tier="actionable_gate",
        )
        row = outcomes.iloc[0]

        self.assertEqual(row["exit_source"], "exit_quotes_conservative")
        self.assertEqual(row["score_status"], "SCORED")
        self.assertAlmostEqual(row["exit_after_slippage"], 2.0)
        self.assertGreater(row["net_r"], 0.9)

    def test_score_candidate_outcomes_marks_intrinsic_proxy_as_partial(self):
        signal = dt.date(2026, 5, 1)
        exit_day = dt.date(2026, 5, 4)
        expiry = dt.date(2026, 5, 29)
        rows = pd.DataFrame(
            [
                {
                    "as_of": signal.isoformat(),
                    "ticker": "XYZ",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "trade_setup": "Buy 100C / Sell 105C",
                    "expiry": expiry.isoformat(),
                    "long_symbol": "XYZ260529C00100000",
                    "short_symbol": "XYZ260529C00105000",
                    "long_strike": 100.0,
                    "short_strike": 105.0,
                    "width": 5.0,
                    "entry_net": 1.00,
                    "max_risk": 100.0,
                }
            ]
        )

        class Cache:
            def option_snapshot(self, day):
                return pd.DataFrame()

            def screener(self, day):
                return pd.DataFrame([{"ticker": "XYZ", "close": 104.0}])

        outcomes = v2.score_candidate_outcomes(
            rows,
            cache=Cache(),
            all_days=[signal, exit_day],
            signal_date=signal,
            horizons=[1],
            baseline="trend_v2",
            tier="actionable_gate",
            max_exit_date=exit_day,
        )
        row = outcomes.iloc[0]

        self.assertEqual(row["exit_source"], "intrinsic_proxy_missing_exit_quote")
        self.assertEqual(row["score_status"], "PARTIAL")
        self.assertGreater(row["net_r"], 0)

    def test_score_candidate_outcomes_marks_cutoff_exits_unscorable(self):
        signal = dt.date(2026, 5, 1)
        exit_day = dt.date(2026, 5, 4)
        rows = pd.DataFrame(
            [
                {
                    "as_of": signal.isoformat(),
                    "ticker": "XYZ",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "expiry": "2026-05-29",
                    "long_strike": 100.0,
                    "short_strike": 105.0,
                    "width": 5.0,
                    "entry_net": 1.00,
                    "max_risk": 100.0,
                }
            ]
        )

        class Cache:
            def screener(self, day):
                return pd.DataFrame([{"ticker": "XYZ", "close": 104.0}])

            def option_snapshot(self, day):
                return pd.DataFrame()

        outcomes = v2.score_candidate_outcomes(
            rows,
            cache=Cache(),
            all_days=[signal, exit_day],
            signal_date=signal,
            horizons=[1],
            baseline="trend_v2",
            tier="actionable_gate",
            max_exit_date=signal,
        )
        row = outcomes.iloc[0]

        self.assertEqual(row["score_status"], "UNSCORABLE")
        self.assertEqual(row["score_status_reason"], "exit_after_validation_cutoff")
        self.assertTrue(math.isnan(row["net_r"]))

    def test_scorecard_reports_net_r_metrics_and_blockers(self):
        outcomes = pd.DataFrame(
            [
                {"baseline": "trend_v2", "tier": "actionable_gate", "horizon": 5, "net_r": 0.50, "block_reasons": ""},
                {"baseline": "trend_v2", "tier": "actionable_gate", "horizon": 5, "net_r": -0.20, "block_reasons": "wide_quote"},
                {"baseline": "trend_v2", "tier": "actionable_gate", "horizon": 5, "net_r": -0.10, "block_reasons": ""},
                {"baseline": "trend_v2", "tier": "actionable_gate", "horizon": 5, "net_r": 0.30, "block_reasons": ""},
            ]
        )

        scorecard = v2.summarize_scorecard(outcomes)
        row = scorecard.iloc[0]

        self.assertEqual(row["signal_count"], 4)
        self.assertEqual(row["scored_count"], 4)
        self.assertEqual(row["unscorable_count"], 0)
        self.assertAlmostEqual(row["win_rate"], 0.5)
        self.assertAlmostEqual(row["avg_net_r"], 0.125)
        self.assertGreater(row["profit_factor"], 2.0)
        self.assertEqual(row["worst_losing_streak"], 2)
        self.assertAlmostEqual(row["blocked_pct"], 0.25)
        self.assertIn("wide_quote:1", row["top_block_reasons"])

    def test_empty_scorecard_keeps_required_columns(self):
        scorecard = v2.summarize_scorecard(pd.DataFrame())

        self.assertEqual(len(scorecard), 0)
        self.assertIn("baseline", scorecard.columns)
        self.assertIn("avg_net_r", scorecard.columns)

    def test_actionable_gate_requires_out_of_sample_edge(self):
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "XYZ",
                    "score": 72.0,
                    "entry_net": 3.10,
                    "block_reasons": "",
                    "live_chain_quote_sanity": "available",
                }
            ]
        )
        weak_scorecard = pd.DataFrame(
            [
                {
                    "baseline": "trend_v2",
                    "tier": "actionable_gate",
                    "horizon": 5,
                    "signal_count": 12,
                    "avg_net_r": 0.01,
                    "profit_factor": 1.20,
                },
                {
                    "baseline": "random_same_date_liquidity",
                    "tier": "baseline",
                    "horizon": 5,
                    "signal_count": 12,
                    "avg_net_r": -0.05,
                    "profit_factor": 0.70,
                },
            ]
        )

        classified, proof = v2.classify_current_candidates(candidates, weak_scorecard, primary_horizon=5)

        self.assertEqual(proof["verdict"], "NO_PROVEN_EDGE_FOR_ACTIONABLE")
        self.assertEqual(classified.iloc[0]["classification"], "WATCH")

        strong_scorecard = weak_scorecard.copy()
        strong_scorecard.loc[0, "avg_net_r"] = 0.18
        strong_scorecard.loc[0, "profit_factor"] = 1.60
        classified, proof = v2.classify_current_candidates(candidates, strong_scorecard, primary_horizon=5)

        self.assertEqual(proof["verdict"], "PROVEN_FOR_ACTIONABLE")
        self.assertEqual(classified.iloc[0]["classification"], "TRADE")

    def test_validation_supported_playbook_can_promote_or_watch_gate_current_rows(self):
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "MCHP",
                    "sector": "Technology",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "score": 58.5,
                    "ret_1d_pct": 1.1,
                    "ret_5d_pct": 5.4,
                    "flow_bias": 0.3,
                    "entry_net": 1.2,
                    "block_reasons": "",
                    "live_chain_quote_sanity": "historical_not_required",
                },
                {
                    "ticker": "GFS",
                    "sector": "Technology",
                    "direction": "bullish",
                    "strategy": "Bull Call Debit",
                    "score": 58.1,
                    "ret_1d_pct": 0.5,
                    "ret_5d_pct": 5.1,
                    "flow_bias": 0.4,
                    "entry_net": 1.9,
                    "block_reasons": "earnings_before_expiry:2026-05-05",
                    "live_chain_quote_sanity": "historical_not_required",
                },
            ]
        )
        scorecard = pd.DataFrame(
            [
                {
                    "baseline": "trend_v2",
                    "tier": v2.TECH_BULLISH_DRIFT_PLAYBOOK,
                    "horizon": 5,
                    "signal_count": 8,
                    "avg_net_r": 0.30,
                    "profit_factor": 1.8,
                }
            ]
        )

        classified, proof = v2.classify_current_candidates(candidates, scorecard, primary_horizon=5)

        self.assertEqual(proof["verdict"], "PROVEN_PLAYBOOKS_FOR_ACTIONABLE")
        self.assertEqual(classified.iloc[0]["classification"], "TRADE")
        self.assertEqual(classified.iloc[1]["classification"], "WATCH")
        self.assertIn(v2.TECH_BULLISH_DRIFT_PLAYBOOK, classified.iloc[0]["validation_playbook"])
        self.assertIn("earnings_before_expiry", classified.iloc[1]["classification_reason"])


if __name__ == "__main__":
    unittest.main()
