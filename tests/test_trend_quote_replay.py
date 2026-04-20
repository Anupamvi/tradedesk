import datetime as dt
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd

from uwos import trend_analysis
from uwos import trend_quote_replay
from uwos.exact_spread_backtester import build_occ_symbol


def _write_hot_chain_zip(root: Path, day: dt.date, rows: list[dict[str, object]]) -> None:
    day_dir = root / day.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    buf = io.StringIO()
    frame.to_csv(buf, index=False)
    zip_path = day_dir / f"hot-chains-{day.isoformat()}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"hot-chains-{day.isoformat()}.csv", buf.getvalue())


class TestTrendQuoteReplay(unittest.TestCase):
    def test_annotates_partial_pass_from_daily_option_quotes(self) -> None:
        signal_date = dt.date(2026, 1, 2)
        exit_date = dt.date(2026, 1, 5)
        expiry = dt.date(2026, 1, 16)
        long_leg = build_occ_symbol("AAPL", expiry, "C", 100.0)
        short_leg = build_occ_symbol("AAPL", expiry, "C", 105.0)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_hot_chain_zip(
                root,
                signal_date,
                [
                    {
                        "option_symbol": long_leg,
                        "date": signal_date.isoformat(),
                        "bid": 1.90,
                        "ask": 2.00,
                        "volume": 100,
                        "open_interest": 1000,
                    },
                    {
                        "option_symbol": short_leg,
                        "date": signal_date.isoformat(),
                        "bid": 0.50,
                        "ask": 0.60,
                        "volume": 100,
                        "open_interest": 1000,
                    },
                ],
            )
            _write_hot_chain_zip(
                root,
                exit_date,
                [
                    {
                        "option_symbol": long_leg,
                        "date": exit_date.isoformat(),
                        "bid": 3.00,
                        "ask": 3.10,
                        "volume": 100,
                        "open_interest": 1000,
                    },
                    {
                        "option_symbol": short_leg,
                        "date": exit_date.isoformat(),
                        "bid": 0.30,
                        "ask": 0.40,
                        "volume": 100,
                        "open_interest": 1000,
                    },
                ],
            )

            candidates = pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "strategy": "Bull Call Debit",
                        "target_expiry": expiry.isoformat(),
                        "long_strike": 100.0,
                        "short_strike": 105.0,
                        "cost_type": "debit",
                    }
                ]
            )

            annotated, replay = trend_quote_replay.annotate_quote_replay(
                candidates,
                root=root,
                signal_date=signal_date,
                mode="gate",
            )

        self.assertEqual(len(replay), 1)
        row = annotated.iloc[0]
        self.assertEqual(row["quote_replay_verdict"], "PARTIAL_PASS")
        self.assertEqual(row["quote_replay_status"], "completed")
        self.assertEqual(row["quote_replay_exit_date"], exit_date.isoformat())
        self.assertAlmostEqual(float(row["quote_replay_entry_net"]), 1.50, places=6)
        self.assertAlmostEqual(float(row["quote_replay_exit_net"]), 2.60, places=6)
        self.assertAlmostEqual(float(row["quote_replay_pnl"]), 110.0, places=6)

    def test_exit_date_override_scores_requested_horizon(self) -> None:
        signal_date = dt.date(2026, 1, 2)
        early_exit = dt.date(2026, 1, 5)
        later_exit = dt.date(2026, 1, 9)
        expiry = dt.date(2026, 1, 16)
        long_leg = build_occ_symbol("AAPL", expiry, "C", 100.0)
        short_leg = build_occ_symbol("AAPL", expiry, "C", 105.0)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_hot_chain_zip(
                root,
                signal_date,
                [
                    {"option_symbol": long_leg, "date": signal_date.isoformat(), "bid": 1.90, "ask": 2.00, "volume": 100, "open_interest": 1000},
                    {"option_symbol": short_leg, "date": signal_date.isoformat(), "bid": 0.50, "ask": 0.60, "volume": 100, "open_interest": 1000},
                ],
            )
            _write_hot_chain_zip(
                root,
                early_exit,
                [
                    {"option_symbol": long_leg, "date": early_exit.isoformat(), "bid": 3.00, "ask": 3.10, "volume": 100, "open_interest": 1000},
                    {"option_symbol": short_leg, "date": early_exit.isoformat(), "bid": 0.30, "ask": 0.40, "volume": 100, "open_interest": 1000},
                ],
            )
            _write_hot_chain_zip(
                root,
                later_exit,
                [
                    {"option_symbol": long_leg, "date": later_exit.isoformat(), "bid": 0.80, "ask": 0.90, "volume": 100, "open_interest": 1000},
                    {"option_symbol": short_leg, "date": later_exit.isoformat(), "bid": 0.20, "ask": 0.30, "volume": 100, "open_interest": 1000},
                ],
            )

            candidates = pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "strategy": "Bull Call Debit",
                        "target_expiry": expiry.isoformat(),
                        "long_strike": 100.0,
                        "short_strike": 105.0,
                        "cost_type": "debit",
                    }
                ]
            )

            annotated, _ = trend_quote_replay.annotate_quote_replay(
                candidates,
                root=root,
                signal_date=signal_date,
                mode="gate",
                exit_date_override=early_exit,
            )

        row = annotated.iloc[0]
        self.assertEqual(row["quote_replay_exit_date"], early_exit.isoformat())
        self.assertEqual(row["quote_replay_verdict"], "PARTIAL_PASS")
        self.assertAlmostEqual(float(row["quote_replay_exit_net"]), 2.60, places=6)

    def test_quote_replay_gate_requires_passing_two_leg_replay(self) -> None:
        candidates = pd.DataFrame(
            [
                {
                    "ticker": "PASS",
                    "swing_score": 70,
                    "backtest_verdict": "PASS",
                    "edge_pct": 5.0,
                    "backtest_signals": 200,
                    "quote_replay_verdict": "PARTIAL_PASS",
                },
                {
                    "ticker": "MISS",
                    "swing_score": 80,
                    "backtest_verdict": "PASS",
                    "edge_pct": 8.0,
                    "backtest_signals": 200,
                    "quote_replay_verdict": "UNAVAILABLE",
                    "quote_replay_status": "skipped_missing_entry",
                    "quote_replay_reason": "missing_entry_net_or_quotes",
                },
            ]
        )

        actionable, patterns = trend_analysis.split_actionable_candidates(
            candidates,
            top_n=10,
            backtest_enabled=True,
            schwab_enabled=False,
            quote_replay_mode="gate",
            min_edge=0.0,
            min_signals=100,
            allow_low_sample=False,
        )

        self.assertEqual(list(actionable["ticker"]), ["PASS"])
        self.assertEqual(list(patterns["ticker"]), ["MISS"])
        self.assertIn("quote replay UNAVAILABLE", patterns.iloc[0]["base_gate_reasons"])

    def test_replay_rejects_impossible_vertical_quote_economics(self) -> None:
        signal_date = dt.date(2026, 1, 2)
        exit_date = dt.date(2026, 1, 5)
        expiry = dt.date(2026, 1, 16)
        long_leg = build_occ_symbol("AAPL", expiry, "C", 100.0)
        short_leg = build_occ_symbol("AAPL", expiry, "C", 105.0)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_hot_chain_zip(
                root,
                signal_date,
                [
                    {"option_symbol": long_leg, "date": signal_date.isoformat(), "bid": 7.90, "ask": 8.00, "volume": 100, "open_interest": 1000},
                    {"option_symbol": short_leg, "date": signal_date.isoformat(), "bid": 1.00, "ask": 1.10, "volume": 100, "open_interest": 1000},
                ],
            )
            _write_hot_chain_zip(
                root,
                exit_date,
                [
                    {"option_symbol": long_leg, "date": exit_date.isoformat(), "bid": 3.00, "ask": 3.10, "volume": 100, "open_interest": 1000},
                    {"option_symbol": short_leg, "date": exit_date.isoformat(), "bid": 0.30, "ask": 0.40, "volume": 100, "open_interest": 1000},
                ],
            )

            candidates = pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "strategy": "Bull Call Debit",
                        "target_expiry": expiry.isoformat(),
                        "long_strike": 100.0,
                        "short_strike": 105.0,
                        "cost_type": "debit",
                    }
                ]
            )

            annotated, _ = trend_quote_replay.annotate_quote_replay(
                candidates,
                root=root,
                signal_date=signal_date,
                mode="gate",
            )

        row = annotated.iloc[0]
        self.assertEqual(row["quote_replay_verdict"], "UNAVAILABLE")
        self.assertEqual(row["quote_replay_status"], "skipped_invalid_entry_economics")
        self.assertEqual(row["quote_replay_reason"], "entry_net_exceeds_width")

    def test_missing_vertical_entry_names_missing_leg(self) -> None:
        signal_date = dt.date(2026, 1, 2)
        exit_date = dt.date(2026, 1, 5)
        expiry = dt.date(2026, 1, 16)
        long_leg = build_occ_symbol("AAPL", expiry, "C", 100.0)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_hot_chain_zip(
                root,
                signal_date,
                [
                    {
                        "option_symbol": long_leg,
                        "date": signal_date.isoformat(),
                        "bid": 1.90,
                        "ask": 2.00,
                        "volume": 100,
                        "open_interest": 1000,
                    },
                ],
            )
            _write_hot_chain_zip(root, exit_date, [])

            candidates = pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "strategy": "Bull Call Debit",
                        "target_expiry": expiry.isoformat(),
                        "long_strike": 100.0,
                        "short_strike": 105.0,
                        "cost_type": "debit",
                    }
                ]
            )

            annotated, _ = trend_quote_replay.annotate_quote_replay(
                candidates,
                root=root,
                signal_date=signal_date,
                mode="gate",
            )

        row = annotated.iloc[0]
        self.assertEqual(row["quote_replay_status"], "skipped_missing_entry")
        self.assertIn("short", row["quote_replay_reason"])
        self.assertIn(build_occ_symbol("AAPL", expiry, "C", 105.0), row["quote_replay_reason"])

    def test_latest_date_with_entry_quotes_passes_as_entry_ok(self) -> None:
        signal_date = dt.date(2026, 1, 2)
        expiry = dt.date(2026, 1, 16)
        long_leg = build_occ_symbol("AAPL", expiry, "C", 100.0)
        short_leg = build_occ_symbol("AAPL", expiry, "C", 105.0)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_hot_chain_zip(
                root,
                signal_date,
                [
                    {
                        "option_symbol": long_leg,
                        "date": signal_date.isoformat(),
                        "bid": 1.90,
                        "ask": 2.00,
                        "volume": 100,
                        "open_interest": 1000,
                    },
                    {
                        "option_symbol": short_leg,
                        "date": signal_date.isoformat(),
                        "bid": 0.50,
                        "ask": 0.60,
                        "volume": 100,
                        "open_interest": 1000,
                    },
                ],
            )

            candidates = pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "strategy": "Bull Call Debit",
                        "target_expiry": expiry.isoformat(),
                        "long_strike": 100.0,
                        "short_strike": 105.0,
                        "cost_type": "debit",
                    }
                ]
            )

            annotated, replay = trend_quote_replay.annotate_quote_replay(
                candidates,
                root=root,
                signal_date=signal_date,
                mode="gate",
            )

        row = annotated.iloc[0]
        self.assertEqual(row["quote_replay_verdict"], "ENTRY_OK")
        self.assertEqual(row["quote_replay_status"], "entry_only_no_later_snapshot")
        self.assertTrue(trend_quote_replay.quote_replay_passes(row))
        self.assertEqual(len(replay), 1)
        self.assertAlmostEqual(float(row["quote_replay_entry_net"]), 1.50, places=6)

    def test_iron_condor_quote_replay_uses_all_four_legs(self) -> None:
        signal_date = dt.date(2026, 1, 2)
        exit_date = dt.date(2026, 1, 5)
        expiry = dt.date(2026, 1, 16)
        symbols = {
            "ps": build_occ_symbol("AAPL", expiry, "P", 95.0),
            "pl": build_occ_symbol("AAPL", expiry, "P", 90.0),
            "cs": build_occ_symbol("AAPL", expiry, "C", 105.0),
            "cl": build_occ_symbol("AAPL", expiry, "C", 110.0),
        }

        def row(symbol: str, day: dt.date, bid: float, ask: float) -> dict[str, object]:
            return {
                "option_symbol": symbol,
                "date": day.isoformat(),
                "bid": bid,
                "ask": ask,
                "volume": 100,
                "open_interest": 1000,
            }

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _write_hot_chain_zip(
                root,
                signal_date,
                [
                    row(symbols["ps"], signal_date, 1.20, 1.30),
                    row(symbols["pl"], signal_date, 0.30, 0.40),
                    row(symbols["cs"], signal_date, 1.10, 1.20),
                    row(symbols["cl"], signal_date, 0.20, 0.30),
                ],
            )
            _write_hot_chain_zip(
                root,
                exit_date,
                [
                    row(symbols["ps"], exit_date, 0.40, 0.50),
                    row(symbols["pl"], exit_date, 0.10, 0.15),
                    row(symbols["cs"], exit_date, 0.30, 0.40),
                    row(symbols["cl"], exit_date, 0.05, 0.10),
                ],
            )

            candidates = pd.DataFrame(
                [
                    {
                        "ticker": "AAPL",
                        "strategy": "Iron Condor",
                        "target_expiry": expiry.isoformat(),
                        "short_strike": 95.0,
                        "long_strike": 105.0,
                        "spread_width": 5.0,
                        "cost_type": "credit",
                    }
                ]
            )

            annotated, replay = trend_quote_replay.annotate_quote_replay(
                candidates,
                root=root,
                signal_date=signal_date,
                mode="gate",
            )

        self.assertEqual(len(replay), 1)
        row_out = annotated.iloc[0]
        self.assertEqual(row_out["quote_replay_status"], "completed")
        self.assertEqual(row_out["quote_replay_verdict"], "PARTIAL_PASS")
        self.assertAlmostEqual(float(row_out["quote_replay_entry_net"]), 1.60, places=6)
        self.assertAlmostEqual(float(row_out["quote_replay_exit_net"]), 0.75, places=6)
        self.assertAlmostEqual(float(row_out["quote_replay_pnl"]), 85.0, places=6)


if __name__ == "__main__":
    unittest.main()
