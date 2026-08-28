import unittest

from corat.technical import append_core_spot, avwap_levels, technical_snapshot
from tests.helpers import trend_bars


class TechnicalTest(unittest.TestCase):
    def test_snapshot_ignores_future_bars(self):
        bars = trend_bars(count=260)
        as_of = bars[-10].date
        first = technical_snapshot("AAA", bars, as_of)
        mutated = list(bars)
        future = mutated[-1]
        mutated[-1] = type(future)(future.date, 999, 1000, 998, 999, 99_000_000, True, future.updated_at, future.source)
        second = technical_snapshot("AAA", mutated, as_of)
        self.assertAlmostEqual(first.price, second.price)
        self.assertAlmostEqual(first.ema20, second.ema20)

    def test_avwap_anchors_are_event_based(self):
        bars = trend_bars(count=260)
        levels = avwap_levels(bars, last_earnings_date=bars[-30].date)
        reasons = {level.anchor_reason for level in levels}
        self.assertIn("beginning of year", reasons)
        self.assertIn("most recent earnings", reasons)

    def test_core_spot_is_marked_partial(self):
        bars = trend_bars(count=20)
        day = "2026-08-27"
        updated = append_core_spot(bars, {"tradeDate": day, "pxCls": 150, "priorCls": 148, "stkVolu": 1000}, day)
        self.assertEqual(updated[-1].date, day)
        self.assertFalse(updated[-1].complete)

    def test_snapshot_computes_liquidity(self):
        bars = trend_bars(count=260)
        result = technical_snapshot("AAA", bars, bars[-1].date)
        self.assertGreater(result.average_dollar_volume_20d, 25_000_000)
        self.assertIsNotNone(result.atr14)

