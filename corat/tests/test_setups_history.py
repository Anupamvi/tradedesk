import unittest
from unittest import mock

from corat.history import _match_setup, analyze_analogues
from corat.models import Bar
from corat.setups import detect_setups
from corat.technical import technical_snapshot
from tests.helpers import trend_bars


class SetupHistoryTest(unittest.TestCase):
    def test_breakout_detected_from_price_and_volume(self):
        bars = trend_bars(count=260, breakout=True)
        snap = technical_snapshot("AAA", bars, bars[-1].date)
        signals = detect_setups(snap, bars, snap, snap, "ACCELERATING LEADER")
        names = {signal.name for signal in signals}
        self.assertIn("BREAKOUT + CONFIRMATION", names)

    def test_no_future_changes_analogue_stats(self):
        bars = trend_bars(count=360, breakout=True)
        as_of = bars[-20].date
        kwargs = dict(
            setup_name="RELATIVE-STRENGTH LEADER", direction="BULLISH", spy_bars=bars,
            as_of=as_of, horizons=[1, 3, 5, 10, 20], primary_horizon=10,
            minimum_sample=2, maximum_sample=200, signal_spacing=5,
        )
        first = analyze_analogues(bars=bars, **kwargs)
        mutated = list(bars)
        last = mutated[-1]
        mutated[-1] = type(last)(last.date, 10, 11, 9, 10, last.volume, True, last.updated_at, last.source)
        second = analyze_analogues(bars=mutated, **kwargs)
        self.assertEqual(first.sample_size, second.sample_size)
        self.assertEqual(first.expectancy, second.expectancy)

    def test_split_bounds_signal_dates(self):
        bars = trend_bars(count=360, breakout=True)
        start = bars[250].date
        result = analyze_analogues(
            "RELATIVE-STRENGTH LEADER", "BULLISH", bars, bars, bars[-1].date,
            [1, 3, 5, 10, 20], 10, 1, 200, 5, signal_start_date=start,
        )
        self.assertTrue(all(value >= start for value in result.signal_dates))

    def test_primary_horizon_forces_nonoverlapping_analogue_signals(self):
        bars = trend_bars(count=420, breakout=True)
        with mock.patch("corat.history._match_setup", return_value=True):
            result = analyze_analogues(
                "RELATIVE-STRENGTH LEADER", "BULLISH", bars, bars, bars[-1].date,
                [1, 3, 5, 10, 20], 10, 1, 500, 1,
            )
        index_by_date = {bar.date: index for index, bar in enumerate(bars)}
        indices = [index_by_date[value] for value in result.signal_dates]
        self.assertTrue(all(right - left >= 10 for left, right in zip(indices, indices[1:])))

    def test_post_earnings_drift_has_a_historical_matcher(self):
        bars = []
        price = 100.0
        for index in range(100):
            if index == 80:
                price *= 1.05
            else:
                price *= 1.002
            day = "2026-{:02d}-{:02d}".format(1 + index // 28, 1 + index % 28)
            bars.append(Bar(day, price - 0.2, price + 0.3, price - 0.4, price, 1_000_000, True))
        last = bars[-1]
        bars[-1] = Bar(last.date, last.open, last.high + 1.0, last.low, bars[-2].high + 0.5, 1_500_000, True)
        event = {"earnDate": bars[80].date, "anncTod": "900"}
        self.assertTrue(_match_setup("POST-EARNINGS DRIFT", bars, bars, earnings_events=[event]))
