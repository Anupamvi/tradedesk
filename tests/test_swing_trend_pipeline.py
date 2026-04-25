import unittest
import datetime as dt
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

from uwos import swing_trend_pipeline as swing


def _signals(**overrides):
    base = dict(
        ticker="TEST",
        n_days_observed=5,
        latest_close=100.0,
        price_direction="bullish",
        price_r_squared=0.70,
        flow_direction="bullish",
        flow_consistency=0.80,
        hot_flow_direction="mixed",
        hot_flow_consistency=0.50,
        pcr_direction="stable",
        oi_direction="bullish",
        oi_consistency=0.85,
        oi_momentum_slope=0.10,
        top_call_strike=105.0,
        top_put_strike=95.0,
        iv_level="low",
        iv_regime="stable",
        latest_iv_rank=30.0,
        volume_surge_days=1,
        whale_appearances=3,
        dp_direction="accumulation",
        dp_consistency=0.80,
    )
    base.update(overrides)
    return swing.SwingSignals(**base)


def _whale_cfg():
    return {
        "gates": {
            "exclude_etfs": True,
            "exclude_issue_types": ["ETF"],
            "min_credit_pct_width": 0.30,
            "max_credit_pct_width": 0.55,
            "max_debit_pct_width": 0.45,
            "min_leg_open_interest": 100,
            "max_strike_distance_pct": 0.80,
            "width_tiers": [
                {"min_price": 0, "max_price": 25, "default_width": 2.5},
                {"min_price": 25, "max_price": 150, "default_width": 10},
            ],
        },
        "shield": {"dte_range": [28, 56], "use_anchor_whitelist": False},
        "fire": {"dte_range": [21, 70]},
    }


def _write_bot_eod(day_dir: Path, date_str: str, symbol: str) -> None:
    csv_path = day_dir / f"bot-eod-report-{date_str}.csv"
    pd.DataFrame(
        [
            {
                "executed_at": f"{date_str} 13:30:00+00",
                "underlying_symbol": symbol,
                "side": "ask",
                "strike": 105,
                "option_type": "call",
                "expiry": "2026-05-15",
                "underlying_price": 100,
                "price": 2.0,
                "size": 10,
                "premium": 2000,
                "open_interest": 500,
                "implied_volatility": 0.2,
                "delta": 0.3,
                "equity_type": "Common Stock",
            }
        ]
    ).to_csv(csv_path, index=False)
    with zipfile.ZipFile(day_dir / f"bot-eod-report-{date_str}.zip", "w") as zf:
        zf.write(csv_path, arcname=csv_path.name)


class TestSwingTrendPipelineDirection(unittest.TestCase):
    def test_run_pipeline_keeps_latest_day_short_history_ticker_in_radar_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dates = [dt.date(2026, 4, 20), dt.date(2026, 4, 21), dt.date(2026, 4, 22)]
            base_row = {
                "issue_type": "Common Stock",
                "is_index": "f",
                "marketcap": 3_000_000_000,
                "avg30_volume": 1_000_000,
                "total_volume": 2_000_000,
                "call_volume": 100,
                "put_volume": 50,
                "bullish_premium": 1000,
                "bearish_premium": 500,
                "iv_rank": 50,
                "next_earnings_date": "",
            }
            for i, day in enumerate(dates):
                day_dir = root / day.isoformat()
                day_dir.mkdir()
                rows = [{**base_row, "ticker": "BASE", "close": 100 + i}]
                if day == dates[-1]:
                    rows.append({**base_row, "ticker": "NEW", "close": 42.0, "call_volume": 500})
                pd.DataFrame(rows).to_csv(day_dir / f"stock-screener-{day.isoformat()}.csv", index=False)

            cfg = {
                "pipeline": {"min_days_required": 3},
                "filters": {
                    "exclude_etfs": True,
                    "exclude_indices": True,
                    "min_market_cap": 0,
                    "min_avg30_volume": 0,
                    "max_tickers_to_score": 20,
                    "max_latest_day_tickers": 20,
                    "max_total_tickers_to_score": 20,
                },
                "schwab_validation": {"enabled": False},
                "backtest": {"enabled": False},
                "output": {
                    "report_md_name": "report.md",
                    "shortlist_csv_name": "shortlist.csv",
                    "min_composite_score": 0,
                },
            }

            _, signals = swing.run_pipeline(
                cfg=cfg,
                root=root,
                lookback=3,
                as_of=dates[-1],
                out_dir=root / "out",
                max_recommendations=20,
            )

            self.assertIn("BASE", signals)
            self.assertIn("NEW", signals)
            self.assertEqual(signals["NEW"].n_days_observed, 1)

    def test_conflicting_price_and_flow_stays_neutral_without_extra_confirmation(self) -> None:
        sig = _signals(
            price_direction="bullish",
            flow_direction="bearish",
            hot_flow_direction="mixed",
            oi_direction="bullish",
            pcr_direction="stable",
        )

        score = swing.score_ticker(sig, {})

        self.assertEqual(score.direction, "neutral")
        self.assertGreater(score.direction_bull_score, 1.0)
        self.assertGreater(score.direction_bear_score, 0.5)

    def test_bearish_price_and_flow_can_win_even_if_oi_is_bullish(self) -> None:
        sig = _signals(
            price_direction="bearish",
            flow_direction="bearish",
            oi_direction="bullish",
            dp_direction="neutral",
            hot_flow_direction="mixed",
        )

        score = swing.score_ticker(sig, {})

        self.assertEqual(score.direction, "bearish")
        self.assertGreater(score.direction_bear_score, score.direction_bull_score)

    def test_range_bound_bullish_flow_and_accumulation_can_form_reversal_bull_bias(self) -> None:
        sig = _signals(
            price_direction="range_bound",
            flow_direction="bullish",
            hot_flow_direction="bearish",
            oi_direction="bullish",
            dp_direction="accumulation",
            dp_consistency=0.80,
        )

        score = swing.score_ticker(sig, {})

        self.assertEqual(score.direction, "bullish")
        self.assertGreater(score.direction_bull_score, score.direction_bear_score)

    def test_dp_accumulation_scores_higher_for_reversal_than_for_extended_bullish_trend(self) -> None:
        reversal = swing.score_ticker(
            _signals(
                price_direction="range_bound",
                flow_direction="bullish",
                dp_direction="accumulation",
                pcr_direction="declining",
            ),
            {},
        )
        extended = swing.score_ticker(
            _signals(
                price_direction="bullish",
                flow_direction="bullish",
                dp_direction="accumulation",
                pcr_direction="declining",
            ),
            {},
        )

        self.assertGreater(reversal.dp_confirmation_score, extended.dp_confirmation_score)

    def test_chain_alias_fallback_supports_class_share_symbol(self) -> None:
        class _FakeService:
            def __init__(self) -> None:
                self.calls = []

            def get_option_chain(self, *, symbol, **kwargs):
                self.calls.append(symbol)
                if symbol == "BRKB":
                    raise RuntimeError("400 for BRKB")
                if symbol == "BRK/B":
                    return {"status": "SUCCESS", "underlying": {"mark": 500.0}}
                raise RuntimeError(symbol)

        service = _FakeService()
        chain, used_symbol, exc, attempted = swing._fetch_option_chain_with_alias_fallback(
            service,
            ticker="BRKB",
            from_date=swing.dt.date(2026, 5, 19),
            to_date=swing.dt.date(2026, 6, 8),
        )

        self.assertEqual(attempted, ["BRKB", "BRK/B"])
        self.assertEqual(service.calls, ["BRKB", "BRK/B"])
        self.assertEqual(used_symbol, "BRK/B")
        self.assertIsNone(exc)
        self.assertEqual(chain["status"], "SUCCESS")

    def test_chain_alias_fallback_does_not_invent_alias_for_plain_a_ticker(self) -> None:
        self.assertEqual(
            swing._ticker_chain_candidates("NVDA"),
            ["NVDA"],
        )

    def test_whale_mentions_prefer_bot_eod_over_legacy_markdown(self) -> None:
        date_str = "2026-04-23"
        day = swing.dt.date.fromisoformat(date_str)
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp)
            _write_bot_eod(day_dir, date_str, "AAPL")
            (day_dir / f"whale-{date_str}.md").write_text(
                "\n".join(
                    [
                        "## Top Symbols by Total Premium (Yes-Prime)",
                        "| underlying_symbol | count | total_premium |",
                        "| --- | --- | --- |",
                        "| TSLA | 10 | 1000000 |",
                    ]
                ),
                encoding="utf-8",
            )

            mentions = swing.load_whale_mentions(
                [(day, day_dir)],
                {"AAPL", "TSLA"},
                cfg=_whale_cfg(),
            )

        self.assertEqual(mentions[day], {"AAPL"})

    def test_whale_mentions_use_legacy_markdown_when_bot_eod_is_missing(self) -> None:
        date_str = "2026-04-23"
        day = swing.dt.date.fromisoformat(date_str)
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp)
            (day_dir / f"whale-{date_str}.md").write_text(
                "\n".join(
                    [
                        "## Top Symbols by Total Premium (Yes-Prime)",
                        "| underlying_symbol | count | total_premium |",
                        "| --- | --- | --- |",
                        "| TSLA | 10 | 1000000 |",
                    ]
                ),
                encoding="utf-8",
            )

            mentions = swing.load_whale_mentions(
                [(day, day_dir)],
                {"AAPL", "TSLA"},
                cfg=_whale_cfg(),
            )

        self.assertEqual(mentions[day], {"TSLA"})

    def test_whale_mentions_legacy_fallback_ignores_wrong_date_bot_eod(self) -> None:
        date_str = "2026-04-23"
        day = swing.dt.date.fromisoformat(date_str)
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp)
            _write_bot_eod(day_dir, "2026-04-22", "AAPL")
            (day_dir / f"whale-{date_str}.md").write_text(
                "\n".join(
                    [
                        "## Top Symbols by Total Premium (Yes-Prime)",
                        "| underlying_symbol | count | total_premium |",
                        "| --- | --- | --- |",
                        "| TSLA | 10 | 1000000 |",
                    ]
                ),
                encoding="utf-8",
            )

            mentions = swing.load_whale_mentions(
                [(day, day_dir)],
                {"AAPL", "TSLA"},
                cfg=_whale_cfg(),
            )

        self.assertEqual(mentions[day], {"TSLA"})

    def test_whale_mentions_ignores_corrupt_cached_summary_and_uses_bot_eod(self) -> None:
        date_str = "2026-04-23"
        day = swing.dt.date.fromisoformat(date_str)
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp)
            (day_dir / f"whale-symbol-summary-{date_str}.csv").write_text(
                '"unterminated\n',
                encoding="utf-8",
            )
            _write_bot_eod(day_dir, date_str, "AAPL")

            mentions = swing.load_whale_mentions(
                [(day, day_dir)],
                {"AAPL", "TSLA"},
                cfg=_whale_cfg(),
            )

        self.assertEqual(mentions[day], {"AAPL"})

    def test_whale_mentions_do_not_trust_nonempty_cache_when_bot_eod_exists(self) -> None:
        date_str = "2026-04-23"
        day = swing.dt.date.fromisoformat(date_str)
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp)
            (day_dir / f"whale-symbol-summary-{date_str}.csv").write_text(
                "underlying_symbol,count,total_premium\nAAPL,1,100\n",
                encoding="utf-8",
            )
            _write_bot_eod(day_dir, date_str, "MSFT")

            mentions = swing.load_whale_mentions(
                [(day, day_dir)],
                {"AAPL", "MSFT"},
                cfg=_whale_cfg(),
            )

        self.assertEqual(mentions[day], {"MSFT"})

    def test_whale_mentions_do_not_fallback_to_markdown_when_bot_eod_exists(self) -> None:
        date_str = "2026-04-23"
        day = swing.dt.date.fromisoformat(date_str)
        with tempfile.TemporaryDirectory() as tmp:
            day_dir = Path(tmp)
            _write_bot_eod(day_dir, date_str, "MSFT")
            (day_dir / f"whale-{date_str}.md").write_text(
                "\n".join(
                    [
                        "## Top Symbols by Total Premium (Yes-Prime)",
                        "| underlying_symbol | count | total_premium |",
                        "| --- | --- | --- |",
                        "| TSLA | 10 | 1000000 |",
                    ]
                ),
                encoding="utf-8",
            )

            mentions = swing.load_whale_mentions(
                [(day, day_dir)],
                {"AAPL", "TSLA"},
                cfg=_whale_cfg(),
            )

        self.assertNotIn(day, mentions)


if __name__ == "__main__":
    unittest.main()
