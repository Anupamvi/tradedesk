from __future__ import annotations

import datetime as dt
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from uwos.fresh_wheel_replay import (
    ReplaySignal,
    generate_signals_for_session,
    iter_sessions,
    load_uw_option_contracts,
    run_replay,
    score_signal,
)
from uwos.fresh_wheel_schwab import WheelConfig


def _write_zip_csv(path: Path, member: str, text: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, text)


def _write_session(root: Path, day: str, *, option_mark: float = 5.0, spot: float = 280.0) -> Path:
    base = root / day
    base.mkdir(parents=True, exist_ok=True)
    screener = (
        "date,ticker,full_name,sector,issue_type,is_index,close,marketcap,avg30_volume,total_open_interest,"
        "bullish_premium,bearish_premium,next_earnings_date\n"
        f"{day},AMZN,Amazon.com Inc,Consumer Cyclical,Common Stock,f,{spot},1800000000000,20000000,2000000,"
        "1000000,250000,2026-07-30\n"
    )
    chain = (
        "option_symbol,bid,ask,close,underlying_price,volume,open_interest,premium,iv,ask_side_volume,bid_side_volume\n"
        f"AMZN260618P00255000,{option_mark - 0.05:.2f},{option_mark + 0.05:.2f},{option_mark:.2f},{spot},"
        "100,800,50000,42,80,20\n"
    )
    _write_zip_csv(base / f"stock-screener-{day}.zip", f"stock-screener-{day}.csv", screener)
    _write_zip_csv(base / f"hot-chains-{day}.zip", f"hot-chains-{day}.csv", chain)
    return base


class FreshWheelReplayTests(unittest.TestCase):
    def test_iter_sessions_requires_screener_and_hot_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "2026-04-29").mkdir()
            _write_session(root, "2026-04-30")

            sessions = iter_sessions(root, dt.date(2026, 4, 1), dt.date(2026, 4, 30))

        self.assertEqual([(day, path.name) for day, path in sessions], [(dt.date(2026, 4, 30), "2026-04-30")])

    def test_load_uw_option_contracts_parses_zipped_hot_chains(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _write_session(Path(tmpdir), "2026-04-30")

            contracts = load_uw_option_contracts(
                base,
                dt.date(2026, 4, 30),
                ["AMZN"],
                WheelConfig(min_option_volume=1),
            )

        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0].symbol, "AMZN260618P00255000")
        self.assertEqual(contracts[0].expiry, dt.date(2026, 6, 18))
        self.assertEqual(contracts[0].right, "P")
        self.assertAlmostEqual(contracts[0].mid, 5.0)

    def test_generate_signals_for_session_uses_fresh_selector(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = _write_session(Path(tmpdir), "2026-04-30")

            signals = generate_signals_for_session(
                base,
                dt.date(2026, 4, 30),
                WheelConfig(max_symbols=5, min_option_volume=1),
            )

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].ticker, "AMZN")
        self.assertIn(signals[0].action, {"OPEN_CSP", "SET_CSP_ALERT"})
        self.assertEqual(signals[0].option_symbol, "AMZN260618P00255000")

    def test_score_signal_marks_50pct_credit_capture_hit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_session(root, "2026-04-30", option_mark=5.0)
            _write_session(root, "2026-05-01", option_mark=2.4)
            sessions = iter_sessions(root, None, None)
            signal = ReplaySignal(
                signal_date=dt.date(2026, 4, 30),
                ticker="AMZN",
                action="OPEN_CSP",
                confidence=88.0,
                spot=280.0,
                quality_score=92.0,
                flow_score=60.0,
                option_symbol="AMZN260618P00255000",
                expiry=dt.date(2026, 6, 18),
                strike=255.0,
                entry_credit=5.0,
                dte=49,
                alert_price=None,
                entry_date=dt.date(2026, 4, 30),
            )

            scored = score_signal(signal, sessions, 0, alert_window_days=3, management_window_days=20)

        self.assertEqual(scored.outcome_status, "scored")
        self.assertTrue(scored.hit_50pct_target)
        self.assertEqual(scored.exit_reason, "hit_50pct_target")
        self.assertAlmostEqual(scored.pnl_per_contract or 0.0, 260.0)

    def test_run_replay_writes_manifest_without_live_schwab(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_session(root, "2026-04-30", option_mark=5.0)
            _write_session(root, "2026-05-01", option_mark=2.4)
            out_dir = root / "out"

            _, metrics, outputs = run_replay(
                data_root=root,
                start=dt.date(2026, 4, 30),
                end=dt.date(2026, 5, 1),
                out_dir=out_dir,
                config=WheelConfig(max_symbols=5, min_option_volume=1),
                max_signals_per_session=2,
                alert_window_days=3,
                management_window_days=20,
            )
            manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))

        self.assertEqual(metrics.signals, 1)
        self.assertFalse(manifest["schwab_live_used_for_replay"])
        self.assertEqual(manifest["historical_quote_source"], "local_uw_hot_chains")


if __name__ == "__main__":
    unittest.main()
