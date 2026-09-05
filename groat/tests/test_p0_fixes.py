import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

from groat.dates import session_phase
from groat.gates import (
    analog_persist_reason,
    apply_analog_persist_park,
    apply_crowded_park,
    apply_freshness_park,
    apply_regime_trade_block,
    apply_same_group_book_park,
    stamp_fill_guard,
)
from groat.picks import desk_picks
from groat.persist import (
    copy_session_artifacts,
    extract_prior_analog,
    extract_prior_trades,
    load_prior_payload,
    load_prior_state,
)
from groat.regime import classify
from groat.pipeline import build_full


try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


def _et(hour, minute, day="2026-09-03"):
    y, m, d = [int(x) for x in day.split("-")]
    if ZoneInfo is None:
        return datetime(y, m, d, hour, minute)
    return datetime(y, m, d, hour, minute, tzinfo=ZoneInfo("America/New_York"))


def _trade(ticker="NOW", primary="D", **extra):
    row = {
        "ticker": ticker,
        "primary": primary,
        "action": "TRADE",
        "choice": "OPTIONS",
        "group_status": "accelerating",
        "ret_1": 0.03,
        "extension_atr": 1.2,
        "close": 145.0,
        "ema20": 138.0,
        "atr14": 4.0,
        "reasons": [],
        "x": "Quiet",
        "in_book": False,
        "picked": {
            "instrument": "debit_call_spread",
            "long_strike": 146.0,
            "delta": 0.22,
            "target_debit": 3.4,
        },
    }
    row.update(extra)
    return row


class TestSessionPhase(unittest.TestCase):
    def test_open_rth_close_and_historical(self):
        self.assertEqual(session_phase("2026-09-03", "2026-09-03", now=_et(9, 30)), "open")
        self.assertEqual(session_phase("2026-09-03", "2026-09-03", now=_et(10, 15)), "rth")
        self.assertEqual(session_phase("2026-09-03", "2026-09-03", now=_et(16, 0)), "close")
        self.assertEqual(session_phase("2026-09-02", "2026-09-03", now=_et(10, 15)), "close")


class TestFreshness(unittest.TestCase):
    def test_three_day_now_xom_does_not_reprint_without_pullback(self):
        prior = [
            {"ticker": "NOW", "primary": "D", "group_status": "accelerating"},
            {"ticker": "XOM", "primary": "D", "group_status": "accelerating"},
        ]
        now = _trade(
            "NOW",
            ret_1=0.03,
            extension_atr=1.2,
            close=145.0,
            ema20=138.0,
            atr14=4.0,
        )
        xom = _trade(
            "XOM",
            ret_1=-0.012,
            extension_atr=1.1,
            close=162.58,
            ema20=159.0,
            atr14=2.0,
        )
        self.assertEqual(apply_freshness_park(now, prior), "already_recommended")
        self.assertEqual(now["action"], "WATCH")
        self.assertIsNone(apply_freshness_park(xom, prior))
        self.assertEqual(xom["action"], "TRADE")

    def test_skipped_quiet_ticket_reprints(self):
        prior = [{"ticker": "XOM", "primary": "E", "group_status": "accelerating"}]
        row = _trade(
            "XOM",
            primary="E",
            ret_1=0.0,
            extension_atr=0.55,
            close=162.21,
            ema20=160.35,
            atr14=3.4,
            reasons=[],
        )
        self.assertIsNone(apply_freshness_park(row, prior))
        self.assertEqual(row["action"], "TRADE")

    def test_pullback_or_group_change_may_reprint(self):
        prior = [{"ticker": "NOW", "primary": "D", "group_status": "accelerating"}]
        dip = _trade("NOW", ret_1=-0.03, extension_atr=0.2, reasons=[])
        self.assertIsNone(apply_freshness_park(dip, prior))
        self.assertEqual(dip["action"], "TRADE")
        rotated = _trade("NOW", group_status="deteriorating", reasons=[])
        self.assertIsNone(apply_freshness_park(rotated, prior))
        self.assertEqual(rotated["action"], "TRADE")

    def test_red_day_into_avwap_may_reprint(self):
        prior = [{"ticker": "NOW", "primary": "D", "group_status": "accelerating"}]
        row = _trade(
            "NOW",
            ret_1=-0.02,
            extension_atr=1.4,
            close=146.0,
            ema20=138.0,
            atr14=4.0,
            avwap_swing_low=145.5,
            reasons=[],
        )
        self.assertIsNone(apply_freshness_park(row, prior))
        self.assertEqual(row["action"], "TRADE")

    def test_missing_group_status_does_not_count_as_rotation(self):
        prior = [{"ticker": "NOW", "primary": "D", "group_status": "accelerating"}]
        row = _trade("NOW", group_status="DATA UNAVAILABLE", reasons=[])
        self.assertEqual(apply_freshness_park(row, prior), "already_recommended")
        self.assertEqual(row["action"], "WATCH")


class TestCrowdedAndDeskPick(unittest.TestCase):
    def test_crowded_parks_without_dip(self):
        row = _trade("NOW", x="Crowded", ret_1=0.01, rvol=0.8)
        self.assertEqual(apply_crowded_park(row), "crowded_no_dip")
        self.assertEqual(row["action"], "WATCH")

    def test_desk_pick_is_none_on_crowded_or_lottery_leftover(self):
        crowded = _trade(
            "NOW",
            x="Crowded",
            picked={"instrument": "debit_call_spread", "long_strike": 150.0, "delta": 0.18},
            close=145.0,
        )
        out = desk_picks([crowded])
        self.assertIsNone(out["best_options"])
        self.assertIn("No desk pick", out["none_note"])
        lottery = _trade(
            "NOW",
            x="Quiet",
            picked={"instrument": "debit_call_spread", "long_strike": 160.0, "delta": 0.18},
            close=145.0,
        )
        out2 = desk_picks([lottery])
        self.assertIsNone(out2["best_options"])

    def test_open_group_is_caveat_not_a_skipped_take(self):
        xom = _trade(
            "XOM",
            x="Informed",
            group="energy",
            book_group_held=True,
            book_group_note="CAVEAT: energy",
        )
        out = desk_picks([xom])
        self.assertEqual(out["best_options"]["ticker"], "XOM")
        self.assertEqual(out["caution"][0]["ticker"], "XOM")


class TestAnalogPersist(unittest.TestCase):
    def test_n_shrink_keeps_veto(self):
        row = _trade(
            "XOM",
            evidence={"stock": {"n": 2, "wins": 0, "avg_r": -0.20}},
        )
        prior = {
            ("XOM", "D"): {
                "veto": "analog_0win_veto",
                "n": 4,
                "wins": 0,
                "avg_r": -0.36,
                "stock": {"n": 4, "wins": 0, "avg_r": -0.36},
            }
        }
        self.assertEqual(analog_persist_reason(row, prior), "analog_0win_veto")
        self.assertEqual(apply_analog_persist_park(row, prior), "analog_0win_veto")
        self.assertEqual(row["action"], "WATCH")
        self.assertTrue(row["evidence"].get("analog_persist"))

    def test_same_n_without_live_veto_does_not_persist(self):
        row = _trade("XOM", evidence={"stock": {"n": 4, "wins": 0, "avg_r": 0.05}})
        prior = {
            ("XOM", "D"): {
                "veto": "analog_0win_veto",
                "stock": {"n": 4, "wins": 0, "avg_r": 0.05},
            }
        }
        self.assertIsNone(analog_persist_reason(row, prior))


class TestRegimeAndIncomplete(unittest.TestCase):
    def test_strong_up_quiet_thrust_is_not_unknown(self):
        spy = {
            "ok": True,
            "trend": "strong_up",
            "ret_20": 0.0055,
            "ret_5": 0.002,
            "close": 500.0,
            "ema20": 498.0,
            "sma50": 490.0,
            "sma200": 480.0,
            "above_ema20": True,
        }
        with mock.patch("groat.regime.snapshot", return_value=spy):
            out = classify("2026-09-03", {"SPY": [{}]}, universe_snaps=[])
        self.assertEqual(out["regime"], "weak_risk_on")
        self.assertNotEqual(out["regime"], "unknown")

    def test_unknown_and_incomplete_block_trade(self):
        unknown = _trade("NOW")
        self.assertEqual(apply_regime_trade_block(unknown, "unknown", False), "regime_unknown")
        self.assertEqual(unknown["action"], "WATCH")
        incomplete = _trade("XOM")
        self.assertEqual(apply_regime_trade_block(incomplete, "weak_risk_on", True), "session_incomplete")
        self.assertEqual(incomplete["action"], "WATCH")

    def test_build_full_open_session_blocks_new_trade(self):
        from tests.barsutil import trend_bars

        asof = "2026-08-26"
        bars = {
            "SPY": trend_bars(220, end=asof, start_px=500, slope=0.4),
            "QQQ": trend_bars(220, end=asof, start_px=400, slope=0.5),
            "IWM": trend_bars(220, end=asof, start_px=200, slope=0.2),
            "DIA": trend_bars(220, end=asof, start_px=380, slope=0.3),
            "NVDA": trend_bars(220, end=asof, start_px=120, slope=0.7, pullback=1.6),
        }
        cores = {
            k: {
                "ticker": k,
                "tradeDate": asof,
                "iv30d": 22.0,
                "orHv20d": 30.0,
                "ivPctile1y": 35,
                "orFcst20d": 28.0,
                "nextErn": "2026-12-01",
                "daysToNextErn": 90,
                "lastErn": "2026-05-20",
            }
            for k in bars
        }
        built = build_full(
            asof,
            token="secret",
            today="2099-01-01",
            live=False,
            universe=list(bars),
            bars_by_ticker=bars,
            cores_by_ticker=cores,
            strikes_by_ticker={},
            session="open",
            prior_trades=[],
            prior_analog={},
        )
        self.assertTrue(built["session_incomplete"])
        self.assertEqual(built["trade_count"], 0)

    def test_build_full_rth_does_not_park_session_incomplete(self):
        from tests.barsutil import trend_bars

        asof = "2026-08-26"
        bars = {
            "SPY": trend_bars(220, end=asof, start_px=500, slope=0.4),
            "QQQ": trend_bars(220, end=asof, start_px=400, slope=0.5),
            "IWM": trend_bars(220, end=asof, start_px=200, slope=0.2),
            "DIA": trend_bars(220, end=asof, start_px=380, slope=0.3),
            "NVDA": trend_bars(220, end=asof, start_px=120, slope=0.7, pullback=1.6),
        }
        cores = {
            k: {
                "ticker": k,
                "tradeDate": asof,
                "iv30d": 22.0,
                "orHv20d": 30.0,
                "ivPctile1y": 35,
                "orFcst20d": 28.0,
                "nextErn": "2026-12-01",
                "daysToNextErn": 90,
                "lastErn": "2026-05-20",
            }
            for k in bars
        }
        built = build_full(
            asof,
            token="secret",
            today="2099-01-01",
            live=False,
            universe=list(bars),
            bars_by_ticker=bars,
            cores_by_ticker=cores,
            strikes_by_ticker={},
            session="rth",
            prior_trades=[],
            prior_analog={},
        )
        parked = []
        for row in built.get("candidates") or []:
            parked.extend(row.get("reasons") or [])
        self.assertNotIn("session_incomplete", parked)


class TestSameGroupParks(unittest.TestCase):
    def test_xom_parks_when_cvx_energy_is_open(self):
        xom = _trade("XOM", group="energy")
        self.assertEqual(apply_same_group_book_park(xom, {"energy"}, {"CVX"}), "same_group_in_book")
        self.assertEqual(xom["action"], "TRADE")
        self.assertTrue(xom.get("book_group_held"))
        self.assertIn("CAVEAT", xom.get("book_group_note") or "")


class TestFillGuardChase(unittest.TestCase):
    def test_already_ran_today_named_on_guard(self):
        row = {
            "choice": "OPTIONS",
            "direction": "bullish",
            "ema20": 146.23,
            "ret_1": 0.04,
            "picked": {"target_debit": 4.0, "instrument": "debit_call_spread"},
        }
        stamp_fill_guard(row)
        self.assertIn("Already ran today", row["fill_note"])


class TestPersistSession(unittest.TestCase):
    def _trade_payload(self, ticker="NOW", action="TRADE", **extra):
        row = {
            "ticker": ticker,
            "primary": "D",
            "action": action,
            "group_status": "accelerating",
            "evidence": {
                "analog_veto": "analog_0win_veto",
                "analog_dates": ["2026-04-01"],
                "stock": {"n": 4, "wins": 0, "avg_r": -0.4},
            },
        }
        row.update(extra)
        return {"candidates": [row]}

    def test_copy_and_load_yesterday_close_for_morning(self):
        payload = self._trade_payload()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            day = root / "2026-09-03"
            day.mkdir()
            (day / "candidates.json").write_text(json.dumps(payload), encoding="utf-8")
            copy_session_artifacts(day, "open")
            self.assertTrue((day / "open" / "candidates.json").is_file())
            yday = root / "2026-09-02"
            yday.mkdir()
            (yday / "close").mkdir()
            (yday / "close" / "candidates.json").write_text(json.dumps(payload), encoding="utf-8")
            morning = load_prior_payload(root, "2026-09-03", session="open")
            self.assertEqual(extract_prior_trades(morning)[0]["ticker"], "NOW")
            analog = extract_prior_analog(morning)
            self.assertEqual(analog[("NOW", "D")]["dates"], ["2026-04-01"])

    def test_incomplete_morning_does_not_wipe_yesterday_trades(self):
        yday_payload = self._trade_payload("NOW")
        morning_payload = {
            "session": "open",
            "session_incomplete": True,
            "candidates": [
                {
                    "ticker": "NOW",
                    "primary": "D",
                    "action": "WATCH",
                    "group_status": "accelerating",
                    "reasons": ["session_incomplete"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yday = root / "2026-09-02"
            (yday / "close").mkdir(parents=True)
            (yday / "close" / "candidates.json").write_text(json.dumps(yday_payload), encoding="utf-8")
            day = root / "2026-09-03"
            day.mkdir()
            (day / "candidates.json").write_text(json.dumps(morning_payload), encoding="utf-8")
            copy_session_artifacts(day, "open")
            trades, analog = load_prior_state(root, "2026-09-03", session="close")
            self.assertEqual([t["ticker"] for t in trades], ["NOW"])
            self.assertEqual(analog[("NOW", "D")]["veto"], "analog_0win_veto")

    def test_close_rerun_does_not_use_same_evening_board(self):
        yday = self._trade_payload("NOW")
        tonight = self._trade_payload("ADBE")
        tonight["session"] = "close"
        tonight["session_incomplete"] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "2026-09-02" / "close").mkdir(parents=True)
            (root / "2026-09-02" / "close" / "candidates.json").write_text(
                json.dumps(yday), encoding="utf-8"
            )
            day = root / "2026-09-03"
            day.mkdir()
            (day / "candidates.json").write_text(json.dumps(tonight), encoding="utf-8")
            trades, _analog = load_prior_state(root, "2026-09-03", session="close")
            self.assertEqual([t["ticker"] for t in trades], ["NOW"])
            self.assertNotIn("ADBE", [t["ticker"] for t in trades])

    def test_walks_past_morning_only_day(self):
        older = self._trade_payload("XOM")
        morning_only = {
            "session": "open",
            "session_incomplete": True,
            "candidates": [
                {
                    "ticker": "NOW",
                    "primary": "D",
                    "action": "WATCH",
                    "reasons": ["session_incomplete"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d2 = root / "2026-09-02"
            (d2 / "close").mkdir(parents=True)
            (d2 / "close" / "candidates.json").write_text(json.dumps(older), encoding="utf-8")
            d3 = root / "2026-09-03"
            d3.mkdir()
            (d3 / "candidates.json").write_text(json.dumps(morning_only), encoding="utf-8")
            copy_session_artifacts(d3, "open")
            trades, _analog = load_prior_state(root, "2026-09-04", session="open")
            self.assertEqual([t["ticker"] for t in trades], ["XOM"])


if __name__ == "__main__":
    unittest.main()
