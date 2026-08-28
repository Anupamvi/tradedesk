import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest import mock

from groki_eq.breakout import atr_wilder, is_breakout, pct_above, prior_high, share_count
from groki_eq.pipeline import build_day, pick_one


def _bars(n=40, end="2026-08-26", close0=100.0, breakout=False):
    last = date.fromisoformat(end)
    out = []
    for i in range(n):
        day = last - timedelta(days=n - 1 - i)
        while day.weekday() >= 5:
            day -= timedelta(days=1)
            # keep weekdays by walking forward from a start instead
        close = close0
        out.append({"date": None, "open": close, "high": close + 1, "low": close - 1, "close": close, "_i": i})
    # rebuild dates as consecutive weekdays ending at end
    days = []
    d = last
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    days = list(reversed(days))
    bars = []
    for i, day in enumerate(days):
        close = close0
        if breakout and i == n - 1:
            close = close0 + 5
        bars.append(
            {
                "date": day.isoformat(),
                "open": close0,
                "high": close + 1.0,
                "low": close0 - 1.0,
                "close": close,
            }
        )
    return bars


class TestMath(unittest.TestCase):
    def test_breakout_strict(self):
        self.assertTrue(is_breakout(101.0, 100.0))
        self.assertFalse(is_breakout(100.0, 100.0))
        self.assertAlmostEqual(pct_above(110.0, 100.0), 0.10)

    def test_prior_high_excludes_asof(self):
        bars = _bars(25, breakout=True)
        high = prior_high(bars, bars[-1]["date"])
        self.assertEqual(high, 100.0)
        self.assertTrue(is_breakout(bars[-1]["close"], high))

    def test_atr_and_shares(self):
        bars = _bars(30)
        atr = atr_wilder(bars, bars[-1]["date"])
        self.assertIsNotNone(atr)
        self.assertGreater(atr, 0)
        self.assertGreaterEqual(share_count(atr), 1)

    def test_rank_pct_then_spy(self):
        a = {"ticker": "IWM", "pct_above": 0.02}
        b = {"ticker": "SPY", "pct_above": 0.02}
        c = {"ticker": "QQQ", "pct_above": 0.03}
        self.assertEqual(pick_one([a, b, c])["ticker"], "QQQ")
        self.assertEqual(pick_one([a, b])["ticker"], "SPY")


class TestDay(unittest.TestCase):
    def test_one_execute_and_week_cap(self):
        spy = _bars(40, breakout=True)
        qqq = _bars(40, breakout=False)
        iwm = _bars(40, breakout=False)
        asof = spy[-1]["date"]
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive"
            out = Path(tmp) / "out"
            with mock.patch("groki_eq.orats.archive_dir", return_value=archive):
                result = build_day(
                    asof,
                    "x",
                    out_dir=out,
                    liquid=["SPY", "QQQ", "IWM"],
                    book={"open": [], "week_entries": {}},
                    schwab_bars={"SPY": spy, "QQQ": qqq, "IWM": iwm},
                )
                self.assertEqual(result["selector"], "breakout_eq")
                self.assertEqual(result["execute_count"], 1)
                self.assertEqual(result["execute_ticker"], "SPY")
                result2 = build_day(
                    asof,
                    "x",
                    out_dir=out,
                    liquid=["SPY", "QQQ", "IWM"],
                    schwab_bars={"SPY": spy, "QQQ": qqq, "IWM": iwm},
                )
            self.assertEqual(result2["execute_count"], 0)
            self.assertTrue(
                any(
                    "week_cap" in str(r.get("reasons") or "") or "already_open" in str(r.get("reasons") or "")
                    for r in result2["board"] + result2["rejections"]
                )
            )


if __name__ == "__main__":
    unittest.main()
