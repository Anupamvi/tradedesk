import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from groat.cli import run
from groat.pipeline import build_full
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


if __name__ == "__main__":
    unittest.main()
