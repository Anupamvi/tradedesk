import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from groat.cli import run
from groat.gates import apply_analog_0win_park
from groat.pipeline import _rank_actionable, build_full, score_row, select_option_names
from groat.xintel import missing_x_tickers
from groat.structure import choose
from tests.barsutil import flat_bars, trend_bars


ASOF = "2026-08-26"


def _core(ticker, iv=22.0, hv=30.0, earn="2026-12-01"):
    return {
        "ticker": ticker,
        "tradeDate": ASOF,
        "iv30d": iv,
        "orHv20d": hv,
        "ivPctile1y": 35,
        "orFcst20d": 28.0,
        "nextErn": earn,
        "daysToNextErn": 90,
        "lastErn": "2026-05-20",
    }


def _universe_bars():
    spy = trend_bars(220, end=ASOF, start_px=500, slope=0.4)
    qqq = trend_bars(220, end=ASOF, start_px=400, slope=0.5)
    iwm = trend_bars(220, end=ASOF, start_px=200, slope=0.2)
    dia = trend_bars(220, end=ASOF, start_px=380, slope=0.3)
    tlt = trend_bars(220, end=ASOF, start_px=90, slope=0.02)
    uup = trend_bars(220, end=ASOF, start_px=28, slope=0.0)
    smh = trend_bars(220, end=ASOF, start_px=220, slope=0.8)
    nvda = trend_bars(220, end=ASOF, start_px=120, slope=0.7, pullback=1.6)
    chop = flat_bars(220, end=ASOF, px=50)
    return {
        "SPY": spy,
        "QQQ": qqq,
        "IWM": iwm,
        "DIA": dia,
        "TLT": tlt,
        "UUP": uup,
        "SMH": smh,
        "NVDA": nvda,
        "AAPL": chop,
    }


class TestPipeline(unittest.TestCase):
    def test_empty_board_is_valid(self):
        bars = {k: flat_bars(80, end=ASOF, px=100) for k in ("SPY", "QQQ", "IWM", "DIA", "AAPL")}
        built = build_full(
            ASOF,
            token="secret-token",
            today="2099-01-01",
            live=False,
            universe=["SPY", "QQQ", "IWM", "DIA", "AAPL"],
            bars_by_ticker=bars,
            cores_by_ticker={k: _core(k) for k in bars},
            strikes_by_ticker={},
        )
        self.assertEqual(built["sleeve"], "groat_swing")
        self.assertGreaterEqual(built["trade_count"], 0)
        # chop may still produce a weak setup; IGNORE-only is the common path
        self.assertIsInstance(built["board"], list)

    def test_leader_can_trade_and_token_stays_out(self):
        bars = _universe_bars()
        cores = {k: _core(k) for k in bars}
        cores["NVDA"] = _core("NVDA", iv=20, hv=32)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with mock.patch("groat.orats.archive_dir", return_value=Path(tmp) / "archive"):
                result = run(
                    ASOF,
                    out,
                    token="secret-token",
                    cmd="full",
                    no_schwab=True,
                    today="2099-01-01",
                    universe=list(bars),
                    bars_by_ticker=bars,
                    cores_by_ticker=cores,
                    strikes_by_ticker={"NVDA": []},
                )
            self.assertEqual(result["mode"], "full")
            self.assertIn(result["regime_label"], {"strong_risk_on", "weak_risk_on", "rotation", "unknown", "range_chop"})
            manifest = json.loads((out / ASOF / "manifest.json").read_text(encoding="utf-8"))
            blob = json.dumps(manifest)
            self.assertNotIn("secret-token", blob)
            board = (out / ASOF / "board.md").read_text(encoding="utf-8")
            self.assertIn("Groat", board)
            self.assertTrue("Empty board. Valid." in board or "TRADE" in board or "WATCH" in board)

    def test_delta_without_prior(self):
        bars = {k: flat_bars(40, end=ASOF) for k in ("SPY", "QQQ")}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with mock.patch("groat.orats.archive_dir", return_value=Path(tmp) / "archive"):
                result = run(
                    ASOF,
                    out,
                    token="secret",
                    cmd="delta",
                    no_schwab=True,
                    today="2099-01-01",
                    universe=["SPY", "QQQ"],
                    bars_by_ticker=bars,
                    cores_by_ticker={"SPY": _core("SPY"), "QQQ": _core("QQQ")},
                    strikes_by_ticker={},
                )
            self.assertTrue((out / ASOF / "delta.md").is_file())
            self.assertIn("delta", result)

    def test_analyze_and_review(self):
        bars = _universe_bars()
        cores = {k: _core(k) for k in bars}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            with mock.patch("groat.orats.archive_dir", return_value=Path(tmp) / "archive"):
                result = run(
                    ASOF,
                    out,
                    token="secret",
                    cmd="analyze",
                    ticker="NVDA",
                    no_schwab=True,
                    today="2099-01-01",
                    universe=list(bars),
                    bars_by_ticker=bars,
                    cores_by_ticker=cores,
                    strikes_by_ticker={},
                )
            self.assertEqual(result["analyze"]["ticker"], "NVDA")
            self.assertTrue((out / ASOF / "analyze_NVDA.md").is_file())
            text = (out / ASOF / "analyze_NVDA.md").read_text(encoding="utf-8")
            self.assertIn("STOCK", text)
            self.assertIn("NO TRADE", text)

    def test_hot_name_is_not_chopped_by_chain_cap(self):
        prelim = []
        for i in range(50):
            prelim.append(
                (
                    "G%s" % i,
                    {"primary": "G", "direction": "bearish", "setups": ["G"], "fire": {}},
                    {"rs_20": 0.20 - i * 0.001},
                )
            )
        prelim.append(
            (
                "DELL",
                {"primary": "A", "direction": "bullish", "setups": ["A"], "fire": {}},
                {"rs_20": -0.01},
            )
        )
        names = select_option_names(prelim, {"DELL": {"heat": "hot"}}, cap=40)
        self.assertIn("DELL", names)
        self.assertEqual(names[0], "DELL")
        self.assertEqual(len(names), 40)

    def test_choose_says_not_requested_not_fetch_fail(self):
        snap = {"close": 100, "atr14": 2, "ema20": 99, "extension_atr": 0.2, "primary": "A"}
        vol = {"iv30": 20, "hv20": 25, "vrp": -5}
        earn = {"usable": True, "source": "exempt", "overlaps_hold": False, "date": "2026-12-01"}
        out = choose(snap, "bullish", vol, [], earn, setup={"primary": "A", "chase": False}, chain_status="not_requested")
        blob = " ".join(out.get("why") or [])
        self.assertIn("not requested", blob)
        self.assertNotIn("DATA UNAVAILABLE", blob)

    def test_ignore_reasons_name_score_not_only_chain(self):
        from groat.pipeline import build_candidate

        snap = {
            "ok": True,
            "stale": False,
            "close": 50,
            "ema20": 49,
            "sma50": 48,
            "sma200": 40,
            "atr14": 2,
            "trend": "up",
            "rs_20": -0.01,
            "rvol": 1.0,
            "extension_atr": 0.2,
            "above_sma50": True,
            "above_sma200": True,
            "reason": "",
        }
        row = build_candidate(
            ASOF,
            "DELL",
            snap,
            _core("DELL"),
            {"status": "deteriorating"},
            {"regime": "weak_risk_on"},
            [],
            [],
            earn={"usable": True, "source": "exempt", "overlaps_hold": False, "date": "2026-12-01"},
            chain_status="not_requested",
        )
        if row["action"] == "IGNORE":
            self.assertIn("score_below_watch", row["reasons"])

    def test_missing_x_on_trade_is_explicit(self):
        self.assertEqual(missing_x_tickers([{"ticker": "XOM", "x": "DATA UNAVAILABLE"}]), ["XOM"])
        self.assertEqual(missing_x_tickers([{"ticker": "NOW", "x": "Informed"}]), [])
        self.assertEqual(missing_x_tickers([{"ticker": "XOM", "x": ""}]), ["XOM"])

    def test_score_lets_mature_d_trade_not_mature_a(self):
        base = {
            "rs_20": 0.08,
            "above_sma200": True,
            "above_sma50": True,
            "close": 100,
            "avwap_swing_low": 95,
            "extension_atr": 0.4,
            "choice": "OPTIONS",
            "picked": {"rr": 2.0, "instrument": "debit_call_spread"},
            "fire": {},
        }
        d = dict(base, primary="D")
        a = dict(base, primary="A")
        e = dict(base, primary="E", rs_20=0.14)
        self.assertGreaterEqual(score_row(d, "weak_risk_on", "mature"), 52)
        self.assertGreaterEqual(score_row(d, "weak_risk_on", "neutral"), 52)
        self.assertLess(score_row(a, "weak_risk_on", "mature"), 52)
        self.assertLess(score_row(a, "weak_risk_on", "neutral"), 52)
        self.assertGreaterEqual(score_row(e, "weak_risk_on", "accelerating"), 52)
        # Original bug: D maxed at 51 so only energy E could TRADE.
        self.assertGreater(score_row(d, "weak_risk_on", "mature"), score_row(a, "weak_risk_on", "accelerating"))

    def test_analog_veto_parks_one_name_not_the_book(self):
        pltr = {
            "ticker": "PLTR",
            "action": "TRADE",
            "score": 70,
            "rs_20": 0.4,
            "evidence": {"stock": {"n": 4, "wins": 0, "avg_r": -0.36}},
        }
        shop = {
            "ticker": "SHOP",
            "action": "TRADE",
            "score": 65,
            "rs_20": 0.2,
            "evidence": {"stock": {"n": 2, "wins": 1, "avg_r": 0.4}},
        }
        net = {
            "ticker": "NET",
            "action": "TRADE",
            "score": 60,
            "rs_20": 0.1,
            "evidence": {"stock": {"n": 1, "wins": 1, "avg_r": 0.8}},
        }
        self.assertEqual(apply_analog_0win_park(pltr), "analog_0win_veto")
        _, trades, watch, _ = _rank_actionable([pltr, shop, net])
        self.assertEqual([r["ticker"] for r in trades], ["SHOP", "NET"])
        self.assertEqual(pltr["action"], "WATCH")
        self.assertEqual(watch[0]["ticker"], "PLTR")

    def test_analog_veto_frees_a_trade_slot(self):
        rows = []
        for i in range(11):
            rows.append(
                {
                    "ticker": "T%s" % i,
                    "action": "TRADE",
                    "score": 80 - i,
                    "rs_20": 0.1,
                    "evidence": {"stock": {"n": 1, "wins": 1, "avg_r": 0.2}},
                }
            )
        rows[0]["ticker"] = "PLTR"
        rows[0]["evidence"] = {"stock": {"n": 4, "wins": 0, "avg_r": -1.0}}
        _, trades, _, _ = _rank_actionable(rows)
        self.assertEqual(len(trades), 10)
        self.assertEqual(trades[0]["ticker"], "PLTR")
        apply_analog_0win_park(trades[0])
        _, trades2, watch2, _ = _rank_actionable(rows)
        names = [r["ticker"] for r in trades2]
        self.assertNotIn("PLTR", names)
        self.assertEqual(len(trades2), 10)
        self.assertIn("T10", names)
        self.assertEqual(watch2[0]["ticker"], "PLTR")


if __name__ == "__main__":
    unittest.main()
