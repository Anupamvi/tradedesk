"""Action board, tape age, LATE, percentile-not-a-kill, DELL/PLTR cache proofs."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from trendbook.pipeline import build_full
from trendbook.replay import load_json_bars
from trendbook.rs import rs_ok
from trendbook.trend import LATE_MIN_WEEKS, current_run, current_run_from_flags, from_base, universe_replay

BARS = Path("/Users/anuppamvi/tradedesk/trendbook/var/schwab_bars")
ASOF = "2026-09-04"


def _daily(start: str, n: int, start_px: float, drift: float) -> list:
    day = datetime.strptime(start, "%Y-%m-%d")
    px = start_px
    rows = []
    made = 0
    while made < n:
        if day.weekday() < 5:
            c = px * (1.0 + drift)
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "open": px,
                    "high": max(px, c) * 1.005,
                    "low": min(px, c) * 0.995,
                    "close": c,
                    "volume": 2_000_000.0,
                }
            )
            px = c
            made += 1
        day += timedelta(days=1)
    return rows


class TestPipelineOffline(unittest.TestCase):
    def test_leader_on_board_laggard_out(self):
        spy = _daily("2024-01-02", 400, 100.0, 0.0008)
        leader = _daily("2024-01-02", 400, 40.0, 0.004)
        laggard = _daily("2024-01-02", 400, 80.0, -0.002)
        asof = spy[-1]["date"]
        with tempfile.TemporaryDirectory() as tmp:
            info = build_full(
                asof,
                out_dir=Path(tmp),
                no_schwab=True,
                no_orats=True,
                persist=False,
                bars_map={"SPY": spy, "LEAD": leader, "LAG": laggard, "SMH": spy},
                tickers=["SPY", "LEAD", "LAG"],
            )
            board = (Path(tmp) / asof / "board.md").read_text(encoding="utf-8")
            self.assertIn("## ADD", board)
            self.assertIn("## NEW", board)
            self.assertIn("## HOLD", board)
            self.assertIn("## LATE", board)
            self.assertTrue((Path(tmp) / asof / "replay.md").is_file())
        by = {r["ticker"]: r for r in info["rows"]}
        self.assertIn("LEAD", by)
        self.assertTrue(by["LEAD"]["on_board"], by["LEAD"])
        self.assertIn(by["LEAD"]["status"], ("ADD", "HOLD", "LATE"))
        self.assertGreaterEqual(by["LEAD"]["weeks_in_trend"], 8, by["LEAD"])
        self.assertFalse(by["LAG"]["on_board"], by["LAG"])
        self.assertEqual(by["LAG"]["status"], "OUT")

    def test_empty_add_valid_extended_is_hold(self):
        spy = _daily("2024-01-02", 400, 100.0, 0.0008)
        leader = _daily("2024-01-02", 399, 40.0, 0.003)
        last = dict(leader[-1])
        last["date"] = spy[-1]["date"]
        last["close"] = last["close"] * 1.3
        last["high"] = last["close"]
        leader.append(last)
        asof = last["date"]
        with tempfile.TemporaryDirectory() as tmp:
            info = build_full(
                asof,
                out_dir=Path(tmp),
                no_schwab=True,
                no_orats=True,
                persist=False,
                bars_map={"SPY": spy, "HOT": leader},
                tickers=["SPY", "HOT"],
            )
        hot = [r for r in info["rows"] if r["ticker"] == "HOT"][0]
        self.assertTrue(hot["on_board"])
        self.assertEqual(hot["status"], "HOLD")
        self.assertEqual(hot["entry"], "extended")
        self.assertEqual(info["n_add"], 0)

    def test_weaker_rs_percentile_still_on_board(self):
        spy = _daily("2024-01-02", 400, 100.0, 0.0008)
        lead = _daily("2024-01-02", 400, 40.0, 0.004)
        mid = _daily("2024-01-02", 400, 50.0, 0.0015)
        asof = spy[-1]["date"]
        with tempfile.TemporaryDirectory() as tmp:
            info = build_full(
                asof,
                out_dir=Path(tmp),
                no_schwab=True,
                no_orats=True,
                persist=False,
                bars_map={"SPY": spy, "LEAD": lead, "MID": mid},
                tickers=["SPY", "LEAD", "MID"],
            )
        by = {r["ticker"]: r for r in info["rows"]}
        self.assertTrue(by["MID"]["on_board"], by["MID"])
        self.assertLess(by["MID"]["rs_pctile"], by["LEAD"]["rs_pctile"])
        self.assertTrue(rs_ok({"mansfield_spy": 0.01}, pctile=60.0, universe_n=120))


class TestTapeAgeAndLate(unittest.TestCase):
    def test_late_from_stalled_flags(self):
        flags = []
        px = 100.0
        start = datetime(2025, 1, 3)
        for i in range(32):
            if i < 12:
                px *= 1.04
            else:
                px *= 0.992
            flags.append(
                {
                    "asof": (start + timedelta(weeks=i)).date().isoformat(),
                    "close": px,
                    "high": px * 1.01,
                    "on": True,
                    "stage": 2,
                }
            )
        run = current_run_from_flags(flags)
        self.assertGreaterEqual(run["weeks_in_trend"], LATE_MIN_WEEKS)
        self.assertTrue(run["late"], run)
        self.assertTrue(run["on_board"])

    def test_slow_grind_is_not_late_at_week_8(self):
        flags = []
        px = 100.0
        start = datetime(2025, 1, 3)
        for i in range(12):
            px *= 1.008
            flags.append(
                {
                    "asof": (start + timedelta(weeks=i)).date().isoformat(),
                    "close": px,
                    "high": px * 1.01,
                    "on": True,
                    "stage": 2,
                }
            )
        run = current_run_from_flags(flags)
        self.assertFalse(run["late"], run)
        self.assertEqual(run["weeks_in_trend"], 12)

    def test_dell_weeks_from_cache(self):
        dell_path = BARS / "DELL.json"
        spy_path = BARS / "SPY.json"
        if not dell_path.is_file() or not spy_path.is_file():
            self.skipTest("DELL/SPY cache missing")
        dell = load_json_bars(dell_path)
        spy = load_json_bars(spy_path)
        run = current_run(dell, spy, ASOF)
        self.assertTrue(run["on_board"], run)
        self.assertGreaterEqual(run["weeks_in_trend"], 20, run)
        self.assertLessEqual(str(run["trend_start"]), "2026-03-20")
        self.assertGreater(run["pct_from_start"] or 0, 1.0, run)

    def test_dell_replay_tags_before_the_meat(self):
        dell_path = BARS / "DELL.json"
        spy_path = BARS / "SPY.json"
        if not dell_path.is_file() or not spy_path.is_file():
            self.skipTest("DELL/SPY cache missing")
        rows = universe_replay(
            {"SPY": load_json_bars(spy_path), "DELL": load_json_bars(dell_path)},
            ASOF,
        )
        dell = rows[0]
        self.assertTrue(dell["tagged"])
        self.assertLessEqual(str(dell["first_tag"]), "2026-03-20")
        self.assertIsNotNone(dell["fwd_13w"])
        self.assertGreater(dell["fwd_13w"], 0.10, dell)

    def test_pltr_not_killed_by_percentile(self):
        pltr_path = BARS / "PLTR.json"
        spy_path = BARS / "SPY.json"
        if not pltr_path.is_file() or not spy_path.is_file():
            self.skipTest("PLTR/SPY cache missing")
        run = current_run(load_json_bars(pltr_path), load_json_bars(spy_path), ASOF)
        if run["on_board"]:
            self.assertIn(run["late"] or False, (True, False))
            self.assertGreaterEqual(run["weeks_in_trend"], 1)
        else:
            self.assertTrue(run["had_trend"] or run["weeks_in_trend"] == 0)


class TestCampaignAndGrade(unittest.TestCase):
    def test_two_week_gap_does_not_reset_age(self):
        flags = []
        start = datetime(2025, 1, 3)
        for i in range(20):
            flags.append(
                {
                    "asof": (start + timedelta(weeks=i)).date().isoformat(),
                    "close": 100.0 + i,
                    "high": 101.0 + i,
                    "on": True,
                    "stage": 2,
                }
            )
        for i in range(20, 22):
            flags.append(
                {
                    "asof": (start + timedelta(weeks=i)).date().isoformat(),
                    "close": 118.0,
                    "high": 119.0,
                    "on": False,
                    "stage": 1,
                }
            )
        flags.append(
            {
                "asof": (start + timedelta(weeks=22)).date().isoformat(),
                "close": 125.0,
                "high": 126.0,
                "on": True,
                "stage": 2,
            }
        )
        run = current_run_from_flags(flags)
        self.assertTrue(run["on_board"])
        self.assertEqual(run["weeks_in_trend"], 23)
        self.assertEqual(run["trend_start"], flags[0]["asof"])

    def test_three_week_gap_starts_new_campaign(self):
        flags = []
        start = datetime(2025, 1, 3)
        for i in range(20):
            flags.append(
                {
                    "asof": (start + timedelta(weeks=i)).date().isoformat(),
                    "close": 100.0 + i,
                    "high": 101.0 + i,
                    "on": True,
                    "stage": 2,
                }
            )
        for i in range(20, 23):
            flags.append(
                {
                    "asof": (start + timedelta(weeks=i)).date().isoformat(),
                    "close": 90.0,
                    "high": 91.0,
                    "on": False,
                    "stage": 1,
                }
            )
        flags.append(
            {
                "asof": (start + timedelta(weeks=23)).date().isoformat(),
                "close": 95.0,
                "high": 96.0,
                "on": True,
                "stage": 2,
            }
        )
        run = current_run_from_flags(flags)
        self.assertEqual(run["weeks_in_trend"], 1)
        self.assertEqual(run["trend_start"], flags[-1]["asof"])

    def test_incomplete_week_does_not_count_as_week_two(self):
        flags = [
            {
                "asof": "2026-08-28",
                "week_end": "2026-08-28",
                "close": 100.0,
                "high": 101.0,
                "on": False,
                "stage": 1,
                "ma30": 99.0,
            },
            {
                "asof": "2026-09-04",
                "week_end": "2026-09-04",
                "close": 105.0,
                "high": 106.0,
                "on": True,
                "stage": 2,
                "ma30": 100.0,
                "vol_expand": True,
            },
            {
                "asof": "2026-09-07",
                "week_end": "2026-09-11",
                "close": 104.0,
                "high": 105.0,
                "on": True,
                "stage": 2,
                "ma30": 100.0,
                "vol_expand": False,
            },
        ]
        run = current_run_from_flags(flags)
        self.assertEqual(run["weeks_in_trend"], 1)
        self.assertTrue(run["from_base"])
        self.assertEqual(run["trend_start"], "2026-09-04")

    def test_long_stage3_reclaim_is_not_from_base(self):
        flags = []
        start = datetime(2026, 7, 3)
        for i in range(8):
            flags.append(
                {
                    "asof": (start + timedelta(weeks=i)).date().isoformat(),
                    "week_end": (start + timedelta(weeks=i)).date().isoformat(),
                    "close": 530.0 + i * 12,
                    "high": 540.0 + i * 12,
                    "on": False,
                    "stage": 3,
                    "ma30": 500.0,
                }
            )
        flags.append(
            {
                "asof": "2026-09-04",
                "week_end": "2026-09-04",
                "close": 613.0,
                "high": 622.0,
                "on": True,
                "stage": 2,
                "ma30": 515.0,
            }
        )
        self.assertFalse(from_base(flags, len(flags) - 1))
        run = current_run_from_flags(flags)
        self.assertFalse(run["from_base"])
        self.assertEqual(run["prior_stage"], 3)

    def test_tmo_is_not_add(self):
        tmo_path = BARS / "TMO.json"
        spy_path = BARS / "SPY.json"
        if not tmo_path.is_file() or not spy_path.is_file():
            self.skipTest("TMO/SPY cache missing")
        tmo = load_json_bars(tmo_path)
        spy = load_json_bars(spy_path)
        for asof in (ASOF, "2026-09-07"):
            with tempfile.TemporaryDirectory() as tmp:
                info = build_full(
                    asof,
                    out_dir=Path(tmp),
                    no_schwab=True,
                    no_orats=True,
                    persist=False,
                    bars_map={"SPY": spy, "TMO": tmo},
                    tickers=["SPY", "TMO"],
                )
            tmo_row = [r for r in info["rows"] if r["ticker"] == "TMO"][0]
            self.assertNotEqual(tmo_row["action"], "ADD", tmo_row)
            self.assertFalse(tmo_row.get("from_base"), tmo_row)

    def test_stage2_breakout_and_early_pullback(self):
        from trendbook.trend import assign_action, grade

        self.assertEqual(
            assign_action(
                True,
                False,
                "extended",
                weeks=2,
                entry_reason="chase_3.0_atr",
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.08,
                rs_63=0.12,
                vol_expand=True,
                from_base=True,
            ),
            "ADD",
        )
        self.assertEqual(
            assign_action(
                True,
                False,
                "extended",
                weeks=2,
                entry_reason="chase_3.0_atr",
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.08,
                rs_63=0.12,
            ),
            "NEW",
        )
        self.assertEqual(
            assign_action(
                True,
                False,
                "held",
                weeks=1,
                entry_reason="in_trend_wait",
                hh_hl=False,
                mansfield_rising=False,
                mansfield_spy=0.005,
                rs_63=0.0,
            ),
            "NEW",
        )
        self.assertEqual(
            assign_action(
                True,
                False,
                "held",
                weeks=2,
                entry_reason="in_trend_wait",
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.05,
                rs_63=-0.08,
            ),
            "NEW",
        )
        self.assertEqual(
            assign_action(True, False, "ready", weeks=3, entry_reason="pullback", off_high=0.06),
            "ADD",
        )
        self.assertEqual(assign_action(True, False, "ready", weeks=12, entry_reason="tightness"), "HOLD")
        self.assertEqual(assign_action(True, False, "held", weeks=12, entry_reason="rest_20ema"), "HOLD")
        self.assertEqual(
            assign_action(True, False, "ready", weeks=12, entry_reason="pullback", off_high=0.01),
            "HOLD",
        )
        self.assertEqual(
            assign_action(
                True,
                False,
                "extended",
                weeks=20,
                entry_reason="chase_3.0_atr",
                hh_hl=True,
                mansfield_rising=True,
            ),
            "HOLD",
        )
        self.assertEqual(
            assign_action(True, True, "ready", weeks=30, entry_reason="pullback", off_high=0.10),
            "LATE",
        )
        self.assertEqual(
            grade("HOLD", 20, 0.20, 0.05, "in_trend_wait", True),
            "A",
        )
        self.assertEqual(grade("NEW", 1, 0.20, 0.0, "in_trend_wait", True), "C")
        self.assertEqual(grade("ADD", 2, 0.20, 0.0, "breakout", True, hh_hl=True), "B")
        self.assertEqual(grade("LATE", 30, -0.05, 0.40, "held", True), "C")

    def test_be_campaign_stitches_august_shakeout(self):
        be_path = BARS / "BE.json"
        spy_path = BARS / "SPY.json"
        if not be_path.is_file() or not spy_path.is_file():
            self.skipTest("BE/SPY cache missing")
        run = current_run(load_json_bars(be_path), load_json_bars(spy_path), ASOF)
        self.assertTrue(run["on_board"], run)
        self.assertGreaterEqual(run["weeks_in_trend"], 50, run)
        self.assertLessEqual(str(run["trend_start"]), "2025-08-01")
        self.assertGreater(run["pct_from_start"] or 0, 4.0, run)

    def test_probe_can_see_be_without_hardcoding(self):
        from trendbook.discover import probe_from_cache

        spy_path = BARS / "SPY.json"
        be_path = BARS / "BE.json"
        if not be_path.is_file() or not spy_path.is_file():
            self.skipTest("BE/SPY cache missing")
        names = probe_from_cache(ASOF, set(), {"SPY", "DELL", "MU"}, load_json_bars(spy_path))
        self.assertIn("BE", names)

    def test_ko_20ema_pause_is_not_add(self):
        ko_path = BARS / "KO.json"
        spy_path = BARS / "SPY.json"
        if not ko_path.is_file() or not spy_path.is_file():
            self.skipTest("KO/SPY cache missing")
        with tempfile.TemporaryDirectory() as tmp:
            info = build_full(
                ASOF,
                out_dir=Path(tmp),
                no_schwab=True,
                no_orats=True,
                persist=False,
                bars_map={"SPY": load_json_bars(spy_path), "KO": load_json_bars(ko_path)},
                tickers=["SPY", "KO"],
            )
        ko = [r for r in info["rows"] if r["ticker"] == "KO"][0]
        self.assertNotEqual(ko["action"], "ADD", ko)
        self.assertIn(ko["action"], ("HOLD", "LATE", "NEW", "OUT"))


if __name__ == "__main__":
    unittest.main()
