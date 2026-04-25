import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from uwos import browser_text_capture
from uwos import sentiment_pipeline


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else ["ticker"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


class TestSentimentPipeline(unittest.TestCase):
    def test_text_sentiment_scores_finance_phrases(self):
        bullish, bull_hits = sentiment_pipeline.text_sentiment_score(
            "Analysts upgraded NFLX after strong demand; call buying and breakout momentum."
        )
        bearish, bear_hits = sentiment_pipeline.text_sentiment_score(
            "AAL saw put buying after weak demand and a price target cut."
        )

        self.assertGreater(bullish, 50)
        self.assertLess(bearish, -50)
        self.assertTrue(any("call buying" in hit for hit in bull_hits))
        self.assertTrue(any("put buying" in hit for hit in bear_hits))

    def test_iran_war_theme_maps_energy_bullish_airlines_bearish(self):
        xom = sentiment_pipeline.theme_score("XOM", "Energy", "Iran war Strait of Hormuz oil shock")
        dal = sentiment_pipeline.theme_score("DAL", "Consumer Cyclical", "Iran war Strait of Hormuz oil shock")

        self.assertGreater(xom["score"], 50)
        self.assertLess(dal["score"], -50)

    def test_schwab_auth_wall_text_is_not_sentiment_evidence(self):
        self.assertTrue(
            sentiment_pipeline.is_auth_wall_text(
                "schwab",
                "Enter Security Code. A security code has been sent. Use a different method. Log In.",
            )
        )
        self.assertFalse(
            sentiment_pipeline.is_auth_wall_text(
                "schwab",
                "NFLX upgraded after strong demand; options traders expect a large move.",
            )
        )

    def test_browser_capture_url_builders(self):
        self.assertIn("x.com/search", browser_text_capture.build_url("x", "NFLX stock options"))
        self.assertIn("reddit.com/search", browser_text_capture.build_url("reddit", "Iran war stocks"))
        self.assertIn("news.google.com/search", browser_text_capture.build_url("news", "macro VIX"))
        self.assertIn("client.schwab.com/app/research/#/stocks/NFLX/news", browser_text_capture.build_url("schwab", "NFLX"))

    @patch("uwos.schwab_auth.SchwabLiveDataService.connect")
    def test_schwab_news_best_effort_normalizes_items(self, mock_connect):
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

        response = MagicMock()
        response.status_code = 200
        response.json.return_value = {
            "news": [
                {
                    "headline": "NFLX upgraded after strong demand",
                    "summary": "Analyst notes streaming momentum.",
                    "provider": "Schwab News",
                    "symbols": ["NFLX"],
                    "publishedDate": "2026-04-17T12:00:00Z",
                }
            ]
        }
        client = MagicMock()
        client._get_request.return_value = response
        mock_connect.return_value = client

        svc = SchwabLiveDataService(SchwabAuthConfig(api_key="x", app_secret="y"))
        news = svc.get_news(["NFLX"], limit=5)

        self.assertEqual(news["status"], "ok")
        self.assertEqual(news["items"][0]["headline"], "NFLX upgraded after strong demand")
        self.assertEqual(news["items"][0]["symbols"], ["NFLX"])
        client._get_request.assert_called()

    def test_pipeline_scores_ticker_and_links_existing_trade(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day = root / "2026-04-17"
            uz = day / "_unzipped_mode_a"
            write_csv(
                uz / "stock-screener-2026-04-17.csv",
                [
                    {
                        "date": "2026-04-17",
                        "ticker": "SPY",
                        "bullish_premium": "1000000",
                        "bearish_premium": "900000",
                        "put_call_ratio": "0.9",
                        "sector": "ETF",
                        "full_name": "SPDR S&P 500 ETF",
                        "issue_type": "ETF",
                        "close": "510",
                        "prev_close": "505",
                        "week_52_high": "520",
                        "week_52_low": "410",
                    },
                    {
                        "date": "2026-04-17",
                        "ticker": "QQQ",
                        "bullish_premium": "800000",
                        "bearish_premium": "700000",
                        "put_call_ratio": "0.9",
                        "sector": "ETF",
                        "full_name": "Nasdaq ETF",
                        "issue_type": "ETF",
                        "close": "430",
                        "prev_close": "425",
                        "week_52_high": "440",
                        "week_52_low": "330",
                    },
                    {
                        "date": "2026-04-17",
                        "ticker": "NFLX",
                        "bullish_premium": "9000000",
                        "bearish_premium": "1500000",
                        "put_call_ratio": "0.55",
                        "sector": "Communication Services",
                        "full_name": "Netflix Inc",
                        "issue_type": "Common Stock",
                        "close": "1000",
                        "prev_close": "965",
                        "week_52_high": "1010",
                        "week_52_low": "600",
                    },
                ],
            )
            write_csv(
                uz / "hot-chains-2026-04-17.csv",
                [
                    {
                        "option_symbol": "NFLX260515C01000000",
                        "premium": "2500000",
                        "ask_side_volume": "900",
                        "bid_side_volume": "100",
                        "volume": "1000",
                        "avg_price": "25",
                    }
                ],
            )
            write_csv(
                uz / "chain-oi-changes-2026-04-17.csv",
                [
                    {
                        "option_symbol": "NFLX260515C01050000",
                        "underlying_symbol": "NFLX",
                        "prev_total_premium": "1000000",
                        "prev_ask_volume": "700",
                        "prev_bid_volume": "100",
                        "oi_diff_plain": "5000",
                        "curr_vol": "800",
                        "avg_price": "12.5",
                    }
                ],
            )
            write_csv(
                day / "browser-text-capture-nflx-2026-04-17.csv",
                [
                    {
                        "captured_at": "2026-04-17T20:00:00Z",
                        "source": "news",
                        "query": "NFLX stock options",
                        "url": "https://example.test",
                        "text_file": "",
                        "screenshot_file": "",
                        "char_count": "120",
                        "text": "NFLX upgraded as strong demand drives breakout momentum; traders highlight call buying.",
                    }
                ],
            )
            write_csv(
                root / "out" / "trend_analysis" / "trend-analysis-actionable-2026-04-17-L20.csv",
                [
                    {
                        "ticker": "NFLX",
                        "direction": "bullish",
                        "strategy": "Bull Call Debit",
                        "strike_setup": "Buy 1000C / Sell 1030C",
                        "target_expiry": "2026-05-15",
                        "position_size_tier": "STARTER_RISK",
                        "swing_score": "75",
                    }
                ],
            )
            write_csv(
                root
                / "out"
                / "trend_analysis_batch"
                / "trend-analysis-batch-rolling-ticker-playbook-audit-2026-04-01_2026-04-17-L1.csv",
                [
                    {
                        "ticker": "NFLX",
                        "direction": "bullish",
                        "strategy": "Bull Call Debit",
                        "horizon_market_days": "20",
                        "forward_tests": "4",
                        "forward_dates": "4",
                        "forward_hit_rate": "1.0",
                        "forward_avg_pnl": "248.25",
                        "forward_profit_factor": "inf",
                        "forward_worst_pnl": "189",
                        "recent_forward_tests": "4",
                        "recent_forward_hit_rate": "1.0",
                        "recent_forward_avg_pnl": "248.25",
                        "recent_forward_profit_factor": "inf",
                        "first_forward_date": "2026-03-16",
                        "last_forward_date": "2026-03-19",
                        "verdict": "supportive",
                    }
                ],
            )

            result = sentiment_pipeline.run(
                [
                    "NFLX",
                    "--root-dir",
                    str(root),
                    "--as-of",
                    "2026-04-17",
                    "--lookback",
                    "1",
                    "--top",
                    "5",
                    "--no-schwab-news",
                ]
            )
            rows = sentiment_pipeline.read_csv_rows(result["scores_csv"])

        self.assertEqual(rows[0]["ticker"], "NFLX")
        self.assertEqual(rows[0]["direction"], "bullish")
        self.assertGreater(float(rows[0]["sentiment_score"]), 40)
        self.assertEqual(rows[0]["trade_status"], "ACTIONABLE")
        self.assertEqual(rows[0]["proof_status"], "PROOF_SUPPORTED")
        self.assertIn("Bull Call Debit", rows[0]["trade_summary"])

    def test_pipeline_blocks_unproven_trade_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day = root / "2026-04-17"
            uz = day / "_unzipped_mode_a"
            write_csv(
                uz / "stock-screener-2026-04-17.csv",
                [
                    {
                        "date": "2026-04-17",
                        "ticker": "SPY",
                        "bullish_premium": "1000000",
                        "bearish_premium": "900000",
                        "put_call_ratio": "0.9",
                        "sector": "ETF",
                        "full_name": "SPDR S&P 500 ETF",
                        "issue_type": "ETF",
                        "close": "510",
                        "prev_close": "505",
                        "week_52_high": "520",
                        "week_52_low": "410",
                    },
                    {
                        "date": "2026-04-17",
                        "ticker": "QQQ",
                        "bullish_premium": "800000",
                        "bearish_premium": "700000",
                        "put_call_ratio": "0.9",
                        "sector": "ETF",
                        "full_name": "Nasdaq ETF",
                        "issue_type": "ETF",
                        "close": "430",
                        "prev_close": "425",
                        "week_52_high": "440",
                        "week_52_low": "330",
                    },
                    {
                        "date": "2026-04-17",
                        "ticker": "CVX",
                        "bullish_premium": "5000000",
                        "bearish_premium": "1000000",
                        "put_call_ratio": "0.5",
                        "sector": "Energy",
                        "full_name": "Chevron Corp",
                        "issue_type": "Common Stock",
                        "close": "160",
                        "prev_close": "155",
                        "week_52_high": "170",
                        "week_52_low": "130",
                    },
                ],
            )
            write_csv(
                root / "out" / "trend_analysis" / "trend-analysis-actionable-2026-04-17-L1.csv",
                [
                    {
                        "ticker": "CVX",
                        "direction": "bullish",
                        "strategy": "Bull Call Debit",
                        "strike_setup": "Buy 160C / Sell 165C",
                        "target_expiry": "2026-05-15",
                        "position_size_tier": "STARTER_RISK",
                        "swing_score": "70",
                    }
                ],
            )
            write_csv(
                root
                / "out"
                / "trend_analysis_batch"
                / "trend-analysis-batch-rolling-ticker-playbook-audit-2026-04-01_2026-04-17-L1.csv",
                [
                    {
                        "ticker": "CVX",
                        "direction": "bullish",
                        "strategy": "Bull Call Debit",
                        "horizon_market_days": "20",
                        "forward_tests": "6",
                        "forward_dates": "6",
                        "forward_hit_rate": "0.5",
                        "forward_avg_pnl": "-20.5",
                        "forward_profit_factor": "0.77",
                        "forward_worst_pnl": "-280",
                        "recent_forward_tests": "5",
                        "recent_forward_hit_rate": "0.4",
                        "recent_forward_avg_pnl": "-96.6",
                        "recent_forward_profit_factor": "0.09",
                        "first_forward_date": "2026-03-02",
                        "last_forward_date": "2026-03-11",
                        "verdict": "negative",
                    }
                ],
            )

            result = sentiment_pipeline.run(
                [
                    "CVX",
                    "--root-dir",
                    str(root),
                    "--as-of",
                    "2026-04-17",
                    "--lookback",
                    "1",
                    "--top",
                    "5",
                    "--no-schwab-news",
                ]
            )
            rows = sentiment_pipeline.read_csv_rows(result["scores_csv"])

        self.assertEqual(rows[0]["ticker"], "CVX")
        self.assertEqual(rows[0]["trade_status"], "BATCH_BLOCKED")
        self.assertEqual(rows[0]["trade_original_status"], "ACTIONABLE")
        self.assertEqual(rows[0]["proof_status"], "BATCH_BLOCKED")
        self.assertIn("rolling verdict negative", rows[0]["proof_summary"])

    @patch("uwos.sentiment_pipeline.trend_analysis.run")
    def test_pipeline_can_refresh_trend_artifacts_before_scoring(self, mock_trend_run):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            day = root / "2026-04-17"
            uz = day / "_unzipped_mode_a"
            write_csv(
                uz / "stock-screener-2026-04-17.csv",
                [
                    {
                        "date": "2026-04-17",
                        "ticker": "SPY",
                        "bullish_premium": "1000000",
                        "bearish_premium": "900000",
                        "put_call_ratio": "0.9",
                        "sector": "ETF",
                        "full_name": "SPDR S&P 500 ETF",
                        "issue_type": "ETF",
                        "close": "510",
                        "prev_close": "505",
                        "week_52_high": "520",
                        "week_52_low": "410",
                    },
                    {
                        "date": "2026-04-17",
                        "ticker": "QQQ",
                        "bullish_premium": "800000",
                        "bearish_premium": "700000",
                        "put_call_ratio": "0.9",
                        "sector": "ETF",
                        "full_name": "Nasdaq ETF",
                        "issue_type": "ETF",
                        "close": "430",
                        "prev_close": "425",
                        "week_52_high": "440",
                        "week_52_low": "330",
                    },
                    {
                        "date": "2026-04-17",
                        "ticker": "NFLX",
                        "bullish_premium": "9000000",
                        "bearish_premium": "1500000",
                        "put_call_ratio": "0.55",
                        "sector": "Communication Services",
                        "full_name": "Netflix Inc",
                        "issue_type": "Common Stock",
                        "close": "1000",
                        "prev_close": "965",
                        "week_52_high": "1010",
                        "week_52_low": "600",
                    },
                ],
            )

            def _write_trend_outputs(argv):
                out_dir = root / "out" / "trend_analysis"
                trade_csv = out_dir / "trend-analysis-current-setups-2026-04-17-L1.csv"
                write_csv(
                    trade_csv,
                    [
                        {
                            "ticker": "NFLX",
                            "direction": "bullish",
                            "strategy": "Bull Call Debit",
                            "live_strike_setup": "Buy 1000C / Sell 1030C",
                            "target_expiry": "2026-05-15",
                            "setup_tier": "ORDER_READY",
                            "swing_score": "75",
                        }
                    ],
                )
                return {
                    "report": out_dir / "trend-analysis-2026-04-17-L1.md",
                    "actionable_csv": out_dir / "trend-analysis-actionable-2026-04-17-L1.csv",
                    "trade_workup_csv": out_dir / "trend-analysis-trade-workups-2026-04-17-L1.csv",
                    "current_setups_csv": trade_csv,
                    "metadata": out_dir / "trend-analysis-metadata-2026-04-17-L1.json",
                }

            mock_trend_run.side_effect = _write_trend_outputs
            result = sentiment_pipeline.run(
                [
                    "NFLX",
                    "--root-dir",
                    str(root),
                    "--as-of",
                    "2026-04-17",
                    "--lookback",
                    "1",
                    "--top",
                    "5",
                    "--no-schwab-news",
                    "--no-batch-proof-gate",
                    "--run-trend-analysis",
                    "--trend-no-schwab",
                ]
            )
            rows = sentiment_pipeline.read_csv_rows(result["scores_csv"])

        self.assertEqual(rows[0]["trade_status"], "CURRENT_SETUP")
        self.assertIn("Bull Call Debit", rows[0]["trade_summary"])
        mock_trend_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
