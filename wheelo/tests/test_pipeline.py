import tempfile
import unittest
from pathlib import Path
from unittest import mock

from wheelo import orats as orats_mod
from wheelo import xhot as xhot_mod
from wheelo.orats import reset_process_http
from wheelo.pipeline import build_daily, build_select, run_pipeline


def _core_row(ticker, px=22.0):
    return {
        "ticker": ticker,
        "tradeDate": "2026-08-28",
        "pxAtmIv": px,
        "mktCap": 25000,
        "avgOptVolu20d": 3000,
        "borrow30": 2.0,
        "iv30d": 38.0,
        "ivPctile1y": 58,
        "ivHvXernRatio": 1.15,
        "nextErn": "2026-12-20",
        "daysToNextErn": 114,
        "divYield": 1.8,
        "beta1y": 1.05,
        "cVolu": 800,
        "pVolu": 400,
        "confidence": 72,
        "stkPxChng1wk": 1.0,
        "stkPxChng1m": -3.0,
        "stkPxChng1y": 12.0,
        "orHv20d": 32.0,
        "sectorName": "Financials",
    }


def _strike_row(ticker, px=22.0):
    return {
        "ticker": ticker,
        "dte": 30,
        "expirDate": "2026-09-25",
        "strike": round(px * 0.92, 2),
        "putBidPrice": 0.70,
        "putAskPrice": 0.72,
        "callBidPrice": 0.50,
        "callAskPrice": 0.52,
        "stockPrice": px,
    }


class FakeOrats:
    def __init__(self):
        self.calls = []

    def __call__(self, path, query, token):
        self.calls.append((path, query.get("ticker") or "", query.get("fields") or ""))
        if "dailies" in path or path.endswith("/tickers"):
            raise AssertionError("forbidden ORATS path %s" % path)
        tickers = [t for t in (query.get("ticker") or "").split(",") if t]
        if "cores" in path:
            return 200, {"data": [_core_row(t) for t in tickers]}, ""
        if "strikes" in path:
            return 200, {"data": [_strike_row(t) for t in tickers]}, ""
        return 200, {"data": []}, ""


class TestFunnel(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(orats_mod, "CODE_DIR", self.root),
            mock.patch.object(xhot_mod, "CODE_DIR", self.root),
        ]
        for p in self.patches:
            p.start()
        reset_process_http()
        self.fake = FakeOrats()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def _quotes(self, n=80):
        out = {}
        for i in range(n):
            name = "T%02d" % i
            px = 15 + (i % 40)
            if i >= 70:
                px = 90
            out[name] = {"ticker": name, "last": float(px), "volume": 1000.0 + i}
        return out

    def test_own_list_keeps_expensive_growth_names(self):
        from wheelo.config import load_json_config
        from wheelo.pipeline import _price_shortlist

        cfg = load_json_config()
        quotes = {
            "NVDA": {"last": 227.0, "volume": 9000000},
            "PYPL": {"last": 54.0, "volume": 1000000},
        }
        names = _price_shortlist(["NVDA", "PYPL"], quotes, cfg, {}, 40, 35000)
        self.assertIn("NVDA", names)
        self.assertIn("PYPL", names)

    def test_priority_beats_volume(self):
        from wheelo.config import load_json_config
        from wheelo.pipeline import _price_shortlist

        cfg = load_json_config()
        quotes = {
            "PLTR": {"last": 170.0, "volume": 100},
            "PYPL": {"last": 54.0, "volume": 9000000},
        }
        names = _price_shortlist(["PLTR", "PYPL"], quotes, cfg, {}, 1, 35000)
        self.assertEqual(names, ["PLTR"])

    def test_cores_only_after_shortlist(self):
        uni = ["T%02d" % i for i in range(80)]
        quotes = self._quotes(80)
        built = build_select(
            "2026-08-28",
            "tok",
            35000,
            today="2026-08-28",
            live=True,
            getter=self.fake,
            max_requests=15,
            universe=uni,
            quotes=quotes,
            history_fn=lambda t, d: [],
            chain_fn=lambda *a, **k: None,
            yfinance_fn=lambda t: {"ok": False, "error": "yfinance_skipped"},
        )
        core_tickers = []
        strike_tickers = []
        for path, tickers, _fields in self.fake.calls:
            names = [t for t in tickers.split(",") if t]
            if "cores" in path:
                core_tickers.extend(names)
            if "strikes" in path:
                strike_tickers.extend(names)
        self.assertLessEqual(len(set(core_tickers)), 80)
        self.assertLessEqual(len(set(strike_tickers)), 25)
        self.assertLessEqual(len(self.fake.calls), 15)
        self.assertLessEqual(built["manifest"]["orats_http"], 12)
        self.assertGreater(built["manifest"]["shortlist_a"], 0)
        self.assertLessEqual(built["manifest"]["shortlist_a"], 80)
        self.assertLessEqual(built["manifest"]["shortlist_c"], 25)
        self.assertNotIn("/hist/dailies", "".join(c[0] for c in self.fake.calls))
        self.assertFalse(any(c[1] == "" for c in self.fake.calls))

    def test_live_second_run_refetches(self):
        uni = ["T%02d" % i for i in range(20)]
        quotes = self._quotes(20)
        kwargs = dict(
            asof="2026-08-28",
            token="tok",
            capital=35000,
            today="2026-08-28",
            live=True,
            getter=self.fake,
            max_requests=15,
            universe=uni,
            quotes=quotes,
            history_fn=lambda t, d: [],
            chain_fn=lambda *a, **k: None,
            yfinance_fn=lambda t: {"ok": False, "error": "yfinance_skipped"},
        )
        first = build_select(**kwargs)
        n1 = first["manifest"]["orats_http"]
        self.assertGreater(n1, 0)
        reset_process_http()
        self.fake.calls = []
        second = build_select(**kwargs)
        self.assertGreater(second["manifest"]["orats_http"], 0)
        self.assertTrue(self.fake.calls)
        self.assertEqual(second["manifest"]["cache_hits"], 0)

    def test_today_refetches_even_without_schwab_live(self):
        uni = ["T00", "T01"]
        quotes = self._quotes(20)
        kwargs = dict(
            asof="2026-08-28",
            token="tok",
            capital=35000,
            today="2026-08-28",
            live=False,
            getter=self.fake,
            max_requests=15,
            universe=uni,
            quotes=quotes,
            history_fn=lambda t, d: [],
            chain_fn=lambda *a, **k: None,
            yfinance_fn=lambda t: {"ok": False, "error": "yfinance_skipped"},
        )
        build_select(**kwargs)
        reset_process_http()
        self.fake.calls = []
        second = build_select(**kwargs)
        self.assertGreater(second["manifest"]["orats_http"], 0)
        self.assertTrue(any("cores" in c[0] for c in self.fake.calls))

    def test_asof_hist_uses_disk_cache(self):
        uni = ["T00", "T01"]
        kwargs = dict(
            asof="2026-08-20",
            token="tok",
            capital=35000,
            today="2026-08-28",
            live=False,
            getter=self.fake,
            max_requests=15,
            universe=uni,
            quotes={},
            history_fn=lambda t, d: [],
            chain_fn=lambda *a, **k: None,
            yfinance_fn=lambda t: {"ok": False, "error": "yfinance_skipped"},
        )
        first = build_select(**kwargs)
        self.assertGreater(first["manifest"]["orats_http"], 0)
        reset_process_http()
        self.fake.calls = []
        second = build_select(**kwargs)
        self.assertEqual(second["manifest"]["orats_http"], 0)
        self.assertEqual(self.fake.calls, [])

    def test_hist_cores_when_asof_not_today(self):
        uni = ["T00", "T01"]
        build_select(
            "2026-08-20",
            "tok",
            35000,
            today="2026-08-28",
            live=False,
            getter=self.fake,
            max_requests=15,
            universe=uni,
            quotes={},
            history_fn=lambda t, d: [],
            chain_fn=lambda *a, **k: None,
            yfinance_fn=lambda t: {"ok": False, "error": "yfinance_skipped"},
        )
        paths = [c[0] for c in self.fake.calls]
        self.assertTrue(any(p == "/hist/cores" for p in paths))
        self.assertFalse(any(p == "/cores" for p in paths))
        self.assertTrue(any(p == "/hist/strikes" for p in paths) or any(p == "/hist/cores" for p in paths))

    def test_budget_cap(self):
        uni = ["T%02d" % i for i in range(20)]
        built = build_select(
            "2026-08-28",
            "tok",
            35000,
            today="2026-08-28",
            live=True,
            getter=self.fake,
            max_requests=0,
            universe=uni,
            quotes=self._quotes(20),
            history_fn=lambda t, d: [],
            chain_fn=lambda *a, **k: None,
            yfinance_fn=lambda t: {"ok": False, "error": "yfinance_skipped"},
        )
        self.assertEqual(built["manifest"]["error"], "orats_budget")
        self.assertEqual(built["candidates"], [])


class TestDailyNoOratsWhenSchwab(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(orats_mod, "CODE_DIR", self.root),
            mock.patch.object(xhot_mod, "CODE_DIR", self.root),
        ]
        for p in self.patches:
            p.start()
        reset_process_http()
        self.fake = FakeOrats()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_marks_skip_orats(self):
        book = Path(self.tmp.name) / "book.json"
        book.write_text(
            '{"positions":[{"ticker":"SOFI","phase":"csp","entry_premium":1.0,"expiry":"2026-10-16","strike":18}]}',
            encoding="utf-8",
        )
        built = build_daily(
            "2026-08-28",
            "tok",
            today="2026-08-28",
            live=True,
            getter=self.fake,
            book_path=book,
            marks={"SOFI": {"bid": 0.4, "ask": 0.42}},
            spots={"SOFI": 19.0},
            cores_by_ticker={"SOFI": _core_row("SOFI", 19)},
        )
        self.assertEqual(built["orats_http"], 0)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(built["actions"][0]["action"], "CLOSE")


class TestRunPipelineArtifacts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(orats_mod, "CODE_DIR", self.root),
            mock.patch.object(xhot_mod, "CODE_DIR", self.root),
        ]
        for p in self.patches:
            p.start()
        reset_process_http()
        self.fake = FakeOrats()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def test_writes_board(self):
        out = Path(self.tmp.name) / "out"
        uni = ["T00", "T01"]
        quotes = {
            "T00": {"ticker": "T00", "last": 22.0, "volume": 5000},
            "T01": {"ticker": "T01", "last": 24.0, "volume": 4000},
        }
        info = run_pipeline(
            "select",
            "2026-08-28",
            "tok",
            35000,
            out_dir=out,
            live_schwab=True,
            getter=self.fake,
            universe=uni,
            quotes=quotes,
            history_fn=lambda t, d: [],
            chain_fn=lambda *a, **k: None,
            yfinance_fn=lambda t: {"ok": False, "error": "yfinance_skipped"},
            today="2026-08-28",
        )
        day = Path(info["out_dir"])
        self.assertTrue((day / "board.md").is_file())
        self.assertTrue((day / "manifest.json").is_file())
        self.assertTrue((day / "rejections.csv").is_file())

    def test_select_clears_leftover_hot_json(self):
        from wheelo.xhot import hot_path, write_hot

        write_hot(
            "2026-08-28",
            [{"ticker": "T00", "tag": "Crowded", "bias": "bearish", "posts_24h": 9, "narrative": "stale"}],
        )
        self.assertTrue(hot_path("2026-08-28").is_file())
        uni = ["T00", "T01"]
        quotes = {
            "T00": {"ticker": "T00", "last": 22.0, "volume": 5000},
            "T01": {"ticker": "T01", "last": 24.0, "volume": 4000},
        }
        built = build_select(
            "2026-08-28",
            "tok",
            35000,
            today="2026-08-28",
            live=True,
            getter=self.fake,
            universe=uni,
            quotes=quotes,
            history_fn=lambda t, d: [],
            chain_fn=lambda *a, **k: None,
            yfinance_fn=lambda t: {"ok": False, "error": "yfinance_skipped"},
        )
        self.assertFalse(hot_path("2026-08-28").is_file())
        for cand in built.get("candidates") or []:
            self.assertEqual(cand.get("x_status"), "DATA UNAVAILABLE")

    def test_xhot_overlay_does_not_call_orats(self):
        from wheelo import xhot as xhot_mod
        from wheelo.pipeline import overlay_x_artifacts
        from wheelo.xhot import write_hot

        out = Path(self.tmp.name) / "out"
        day = out / "2026-08-31"
        day.mkdir(parents=True)
        (day / "candidates.json").write_text(
            '[{"ticker":"AMAT","conf":78,"conf_label":"TRADE","conf_drivers":["X DATA UNAVAILABLE"],"x_status":"DATA UNAVAILABLE","premium":{"csp_strike":425,"csp_bid":11,"expiry":"2026-10-02","dte":32,"iv_rank":49}}]',
            encoding="utf-8",
        )
        (day / "manifest.json").write_text('{"orats_http":11,"shortlist_a":80,"shortlist_b":78,"shortlist_c":25}', encoding="utf-8")
        with mock.patch.object(xhot_mod, "CODE_DIR", self.root):
            write_hot(
                "2026-08-31",
                [{"ticker": "AMAT", "tag": "Informed", "bias": "bullish", "posts_24h": 6, "narrative": "WFE"}],
            )
            overlay_x_artifacts("2026-08-31", out_dir=out)
        board = (day / "board.md").read_text(encoding="utf-8")
        self.assertIn("Informed", board)
        self.assertIn("| Informed |", board)

    def test_overlay_ignores_stale_hot_json(self):
        import os
        import time

        from wheelo import xhot as xhot_local
        from wheelo.pipeline import overlay_x_artifacts
        from wheelo.xhot import write_hot

        out = Path(self.tmp.name) / "out"
        day = out / "2026-08-31"
        day.mkdir(parents=True)
        (day / "candidates.json").write_text(
            '[{"ticker":"AMAT","conf":78,"conf_label":"TRADE","x_status":"DATA UNAVAILABLE","premium":{"csp_strike":425,"csp_bid":11,"expiry":"2026-10-02","dte":32,"iv_rank":49}}]',
            encoding="utf-8",
        )
        (day / "manifest.json").write_text('{"orats_http":11}', encoding="utf-8")
        with mock.patch.object(xhot_local, "CODE_DIR", self.root):
            path = write_hot(
                "2026-08-31",
                [{"ticker": "AMAT", "tag": "Informed", "bias": "bullish", "posts_24h": 6, "narrative": "stale"}],
            )
            past = time.time() - 120
            os.utime(path, (past, past))
            overlay_x_artifacts("2026-08-31", out_dir=out)
        board = (day / "board.md").read_text(encoding="utf-8")
        self.assertIn("DATA UNAVAILABLE", board)
        self.assertNotIn("| Informed |", board)
