import unittest

from groat.evidence import attach_evidence, options_proxy, walk_stock_plan
from groat.rotation import classify_group
from groat.setups import classify_setups
from groat.structure import stock_plan
from groat.technicals import snapshot
from tests.barsutil import trend_bars


class TestWalkStock(unittest.TestCase):
    def test_stop_then_target_counts_as_loss(self):
        bars = trend_bars(40, end="2026-08-26", start_px=100, slope=0.2)
        start = bars[20]["date"]
        plan = {"side": "long", "entry": bars[20]["close"], "stop": bars[20]["close"] + 50, "target": bars[20]["close"] + 1}
        # stop above entry makes no sense for long; force a red bar
        bars[21]["low"] = plan["entry"] - 10
        bars[21]["high"] = plan["entry"] + 10
        plan["stop"] = plan["entry"] - 2
        plan["target"] = plan["entry"] + 4
        out = walk_stock_plan(bars, start, plan, hold=10)
        self.assertEqual(out["result"], "loss")
        self.assertEqual(out["r"], -1.0)

    def test_target_hits_win(self):
        bars = trend_bars(40, end="2026-08-26", start_px=100, slope=1.0)
        start = bars[20]["date"]
        entry = bars[20]["close"]
        plan = {"side": "long", "entry": entry, "stop": entry - 20, "target": entry + 2}
        out = walk_stock_plan(bars, start, plan, hold=10)
        self.assertEqual(out["result"], "win")
        self.assertGreater(out["r"], 0)


class TestOptionsProxy(unittest.TestCase):
    def test_clamps_debit_loss(self):
        picked = {"target_debit": 2.0, "width": 5.0, "delta": 0.20, "theta": 0.0, "breakeven": 142.0}
        out = options_proxy(picked, 140.0, 100.0, 5, "bullish")
        self.assertTrue(out["priced"])
        self.assertEqual(out["pnl"], -2.0)

    def test_clamps_debit_max_gain(self):
        picked = {"target_debit": 2.0, "width": 5.0, "delta": 0.50, "theta": 0.0, "breakeven": 142.0}
        out = options_proxy(picked, 140.0, 160.0, 5, "bullish")
        self.assertEqual(out["pnl"], 3.0)
        self.assertTrue(out["be_hit"])


class TestAttachEvidence(unittest.TestCase):
    def test_same_setup_analogs_on_trend(self):
        asof = "2026-08-26"
        spy = trend_bars(220, end=asof, start_px=500, slope=0.3)
        igv = trend_bars(220, end=asof, start_px=80, slope=0.6)
        now = trend_bars(220, end=asof, start_px=90, slope=0.55)
        snap = snapshot(now, asof, bench_bars=spy)
        igv_snap = snapshot(igv, asof, bench_bars=spy)
        accel = None
        if igv_snap.get("rs_20") is not None and igv_snap.get("rs_60") is not None:
            accel = igv_snap["rs_20"] - igv_snap["rs_60"]
        group_row = {
            "etf": "IGV",
            "group": "software",
            "status": classify_group(igv_snap.get("rs_20"), igv_snap.get("rs_60"), accel, str(igv_snap.get("trend") or "")),
            "ok": True,
        }
        setup = classify_setups(snap, group_row=group_row, earnings={"usable": True, "source": "exempt"}, bars=now)
        primary = setup.get("primary") or "D"
        trades = [
            {
                "ticker": "NOW",
                "primary": primary,
                "direction": setup.get("direction") or "bullish",
                "choice": "STOCK",
                "picked": {"instrument": "stock"},
            }
        ]
        picks = {"best_stock": trades[0], "best_options": None}
        ev = attach_evidence(
            asof,
            trades,
            picks,
            {"NOW": now, "SPY": spy, "IGV": igv},
            hist_e={},
            cores={},
            token="",
            today="2099-01-01",
            allow_orats_http=False,
        )
        row = ev["rows"][0]
        self.assertEqual(row["ticker"], "NOW")
        self.assertGreaterEqual(row["stock"]["n"], 1)
        self.assertIsNotNone(row["stock"]["avg_r"])
        self.assertEqual(row["options"]["n"], 0)

    def test_clustered_dates_are_not_independent(self):
        from groat.evidence import _overlaps

        bars = trend_bars(40, end="2026-08-26")
        taken = [{"date": bars[25]["date"], "exit_date": bars[35]["date"]}]
        self.assertTrue(_overlaps(bars[24]["date"], taken, bars, gap=15))
        self.assertFalse(_overlaps(bars[5]["date"], taken, bars, gap=15))

    def test_no_http_when_injected(self):
        asof = "2026-08-26"
        bars = trend_bars(80, end=asof)
        ev = attach_evidence(
            asof,
            [{"ticker": "AAPL", "primary": "A", "direction": "bullish", "choice": "OPTIONS", "picked": {"instrument": "debit_call_spread"}}],
            {},
            {"AAPL": bars, "SPY": bars},
            allow_orats_http=False,
            token="secret-token",
        )
        self.assertEqual(ev["http"], 0)


if __name__ == "__main__":
    unittest.main()
