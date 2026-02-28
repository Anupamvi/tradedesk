import argparse
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from uwos import historical_trend_pipeline as trend


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _base_live_rows() -> list[dict]:
    return [
        {
            "ticker": "AAA",
            "strategy": "Bull Call Debit",
            "expiry": "2026-03-20",
            "entry_gate": "<= 1.20 db",
            "conviction": 80,
            "track": "FIRE",
            "gate_pass_live": True,
            "is_final_live_valid": True,
            "live_max_profit": 300.0,
            "live_max_loss": 100.0,
        },
        {
            "ticker": "BBB",
            "strategy": "Bear Put Debit",
            "expiry": "2026-03-20",
            "entry_gate": "<= 2.00 db",
            "conviction": 72,
            "track": "FIRE",
            "gate_pass_live": True,
            "is_final_live_valid": True,
            "live_max_profit": 500.0,
            "live_max_loss": 100.0,
        },
    ]


def _base_setup_rows() -> list[dict]:
    return [
        {
            "ticker": "AAA",
            "strategy": "Bull Call Debit",
            "expiry": "2026-03-20",
            "entry_gate": "<= 1.20 db",
            "hist_success_pct": 60.0,
            "edge_pct": 20.0,
            "verdict": "PASS",
        },
        {
            "ticker": "BBB",
            "strategy": "Bear Put Debit",
            "expiry": "2026-03-20",
            "entry_gate": "<= 2.00 db",
            "hist_success_pct": 50.0,
            "edge_pct": 10.0,
            "verdict": "PASS",
        },
    ]


def _write_run(
    run_dir: Path,
    date_txt: str,
    *,
    approved_rows: int,
    final_rows: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    live_csv = run_dir / f"live_trade_table_{date_txt}.csv"
    final_csv = run_dir / f"live_trade_table_{date_txt}_final.csv"
    setup_csv = run_dir / f"setup_likelihood_{date_txt}.csv"
    dropped_csv = run_dir / f"dropped_trades_{date_txt}.csv"
    output_md = run_dir / f"anu-expert-trade-table-{date_txt}.md"

    _write_csv(live_csv, _base_live_rows())
    _write_csv(final_csv, _base_live_rows()[:1])
    _write_csv(setup_csv, _base_setup_rows())
    _write_csv(
        dropped_csv,
        [
            {"ticker": "CCC", "strategy": "Bull Call Debit", "expiry": "2026-03-20", "stage": "final", "drop_reason": "final_top_limit"},
            {"ticker": "DDD", "strategy": "Bull Call Debit", "expiry": "2026-03-20", "stage": "final", "drop_reason": "final_top_limit"},
        ],
    )
    output_md.write_text("# dummy", encoding="utf-8")

    manifest = {
        "asof_date": date_txt,
        "output_md": str(output_md),
        "artifacts": {
            "live_csv": str(live_csv),
            "live_final_csv": str(final_csv),
            "likelihood_csv": str(setup_csv),
            "dropped_csv": str(dropped_csv),
        },
        "counts": {
            "approved_rows": approved_rows,
            "watch_rows": max(0, final_rows - approved_rows),
            "final_output_rows": final_rows,
            "stage1_shortlist_rows": 40,
            "stage2_live_rows": 40,
        },
        "settings": {"top_trades_requested": final_rows, "pretrade_caps_status": "disabled"},
    }
    (run_dir / f"run_manifest_{date_txt}.json").write_text(json.dumps(manifest), encoding="utf-8")


class TestHistoricalTrendPipeline(unittest.TestCase):
    def test_discovery_handles_both_folder_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "replay_compare"
            _write_run(
                root / "2026-02-01" / "alpha",
                "2026-02-01",
                approved_rows=3,
                final_rows=5,
            )
            _write_run(
                root / "beta" / "2026-02-02",
                "2026-02-02",
                approved_rows=2,
                final_rows=5,
            )
            bundles = trend.discover_run_bundles(root)
            self.assertEqual(len(bundles), 2)
            variants = sorted([b.variant for b in bundles])
            self.assertEqual(variants, ["alpha", "beta"])

    def test_pipeline_outputs_daily_weekly_and_proxy_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "replay_compare"
            out_dir = Path(td) / "trend_out"
            _write_run(
                root / "2026-02-01" / "alpha",
                "2026-02-01",
                approved_rows=3,
                final_rows=5,
            )
            _write_run(
                root / "beta" / "2026-02-02",
                "2026-02-02",
                approved_rows=2,
                final_rows=5,
            )

            args = argparse.Namespace(
                search_root=str(root),
                out_dir=str(out_dir),
                start_date="",
                end_date="",
                lookback_days=0,
                variants=[],
                min_ticker_days=1,
                link_style="relative",
                recommendation_variant="",
                recommendation_top_n=10,
                recommendation_min_days=1,
                recommendation_min_proxy_pf=0.0,
            )
            paths = trend.run_pipeline(args)
            self.assertTrue(paths["daily_variant_metrics_csv"].exists())
            self.assertTrue(paths["weekly_variant_metrics_csv"].exists())
            self.assertTrue(paths["ticker_persistence_csv"].exists())
            self.assertTrue(paths["summary_md"].exists())
            self.assertTrue(paths["final_trade_recommendations_csv"].exists())
            self.assertTrue(paths["final_trade_recommendations_md"].exists())

            daily = pd.read_csv(paths["daily_variant_metrics_csv"])
            self.assertEqual(len(daily), 2)
            alpha = daily[daily["variant"] == "alpha"].iloc[0]
            self.assertEqual(int(alpha["approved_rows_manifest"]), 3)
            self.assertAlmostEqual(float(alpha["approval_rate_manifest"]), 0.6, places=6)

            # Expected proxy:
            # AAA: p=0.60 => gp=180, gl=40
            # BBB: p=0.50 => gp=250, gl=50
            # PF = 430 / 90
            self.assertAlmostEqual(float(alpha["proxy_expected_gross_profit"]), 430.0, places=6)
            self.assertAlmostEqual(float(alpha["proxy_expected_gross_loss"]), 90.0, places=6)
            self.assertAlmostEqual(float(alpha["proxy_pf"]), 430.0 / 90.0, places=6)

            weekly = pd.read_csv(paths["weekly_variant_metrics_csv"])
            self.assertEqual(set(weekly["variant"]), {"alpha", "beta"})

            tickers = pd.read_csv(paths["ticker_persistence_csv"])
            self.assertTrue((tickers["days_present"] >= 1).all())

            recs = pd.read_csv(paths["final_trade_recommendations_csv"])
            self.assertGreaterEqual(len(recs), 1)
            self.assertIn("book_recommendation", recs.columns)
            self.assertIn("track_label", recs.columns)

    def test_lookback_days_filters_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "replay_compare"
            out_dir = Path(td) / "trend_out"
            _write_run(
                root / "2026-02-01" / "alpha",
                "2026-02-01",
                approved_rows=3,
                final_rows=5,
            )
            _write_run(
                root / "2026-02-03" / "alpha",
                "2026-02-03",
                approved_rows=4,
                final_rows=5,
            )
            args = argparse.Namespace(
                search_root=str(root),
                out_dir=str(out_dir),
                start_date="",
                end_date="",
                lookback_days=1,
                variants=[],
                min_ticker_days=1,
                link_style="relative",
                recommendation_variant="",
                recommendation_top_n=10,
                recommendation_min_days=1,
                recommendation_min_proxy_pf=0.0,
            )
            paths = trend.run_pipeline(args)
            daily = pd.read_csv(paths["daily_variant_metrics_csv"])
            self.assertEqual(len(daily), 1)
            self.assertEqual(str(daily.iloc[0]["trade_date"])[:10], "2026-02-03")

    def test_extract_output_md_ranked_rows_ignores_reason_tables(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            md_path = Path(td) / "sample.md"
            md_path.write_text(
                "\n".join(
                    [
                        "## Anu Expert Trade Table",
                        "### Core Book",
                        "#### Approved - FIRE",
                        "| # | Ticker | Action | Strategy Type | Strike Setup | Expiry | Conviction % | Execution Book |",
                        "|---|---|---|---|---|---|---|---|",
                        "| 1 | AAA | FIRE | Bull Call Debit | Buy 10.00C / Sell 12.00C (2.00w) | 2026-03-20 | 80% | Core |",
                        "### Watch Book",
                        "#### Watch Only - FIRE",
                        "| # | Ticker | Action | Strategy Type | Strike Setup | Expiry | Conviction % | Execution Book |",
                        "|---|---|---|---|---|---|---|---|",
                        "| 2 | BBB | FIRE | Bull Call Debit | Buy 20.00C / Sell 22.00C (2.00w) | 2026-03-20 | 70% | Watch |",
                        "## Watch Only Reason Tables",
                        "### Live Entry Gate Miss",
                        "| # | Ticker | Strategy Type | Expiry | Conviction % |",
                        "|---|---|---|---|---|",
                        "| 2 | BBB | Bull Call Debit | 2026-03-20 | 70% |",
                    ]
                ),
                encoding="utf-8",
            )
            rows = trend._extract_output_md_ranked_rows(md_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(int(rows["rank_md"].nunique()), 2)
            self.assertEqual(float(rows.loc[rows["rank_md"] == 1, "long_strike_md"].iloc[0]), 10.0)
            self.assertEqual(float(rows.loc[rows["rank_md"] == 1, "short_strike_md"].iloc[0]), 12.0)

    def test_select_source_rows_by_output_md_matches_strikes(self) -> None:
        source = pd.DataFrame(
            [
                {
                    "ticker": "PYPL",
                    "strategy": "Bull Call Debit",
                    "track": "FIRE",
                    "expiry": "2026-03-20",
                    "conviction": 72,
                    "is_final_live_valid": True,
                    "gate_pass_live": True,
                    "optimal_stage1": "Yes-Good",
                    "long_strike": 47.0,
                    "short_strike": 52.0,
                },
                {
                    "ticker": "PYPL",
                    "strategy": "Bull Call Debit",
                    "track": "FIRE",
                    "expiry": "2026-03-20",
                    "conviction": 72,
                    "is_final_live_valid": True,
                    "gate_pass_live": True,
                    "optimal_stage1": "Watch Only",
                    "long_strike": 47.5,
                    "short_strike": 52.0,
                },
            ]
        )
        md_rows = pd.DataFrame(
            [
                {
                    "rank_md": 17,
                    "ticker": "PYPL",
                    "strategy": "Bull Call Debit",
                    "track_md": "FIRE",
                    "execution_book_md": "Watch",
                    "conviction_md": 72.0,
                    "expiry_md": "2026-03-20",
                    "short_strike_md": 52.0,
                    "long_strike_md": 47.5,
                    "short_put_strike_md": float("nan"),
                    "long_put_strike_md": float("nan"),
                    "short_call_strike_md": 52.0,
                    "long_call_strike_md": 47.5,
                }
            ]
        )
        selected = trend._select_source_rows_by_output_md(source, md_rows)
        self.assertEqual(len(selected), 1)
        self.assertEqual(float(selected.iloc[0]["long_strike"]), 47.5)
        self.assertEqual(str(selected.iloc[0]["optimal_stage1"]), "Watch Only")


if __name__ == "__main__":
    unittest.main()
