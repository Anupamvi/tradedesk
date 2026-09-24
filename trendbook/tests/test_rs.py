import unittest

from trendbook.bars import volume_expand
from trendbook.config import LOOKBACK_DAYS
from trendbook.discover import dynamic_universe, infra_tickers, quote_is_52w_high
from trendbook.rs import ols_beta, residual_return
from trendbook.trend import assign_action


def _series(start, n, drift):
    px = start
    out = []
    for _ in range(n):
        px = px * (1.0 + drift)
        out.append(px)
    return out


class TestResidual(unittest.TestCase):
    def test_high_beta_matching_market_has_small_residual(self):
        spy = [100.0]
        for i in range(1, 200):
            bump = 0.004 if i % 3 == 0 else (-0.002 if i % 3 == 1 else 0.001)
            spy.append(spy[-1] * (1.0 + bump))
        # 2x the daily move of SPY → beta ~2, raw excess large, residual ~0
        stock = [100.0]
        for i in range(1, 200):
            r = spy[i] / spy[i - 1] - 1.0
            stock.append(stock[-1] * (1.0 + 2.0 * r))
        beta = ols_beta(stock, spy, 126)
        resid = residual_return(stock, spy, 126)
        raw = (stock[-1] / stock[-1 - 126] - 1.0) - (spy[-1] / spy[-1 - 126] - 1.0)
        self.assertIsNotNone(beta)
        self.assertGreater(beta, 1.7)
        self.assertLess(beta, 2.3)
        self.assertGreater(raw, 0.05)
        self.assertLess(abs(resid), 0.03)

    def test_lookback_covers_a_bear_year(self):
        self.assertGreaterEqual(LOOKBACK_DAYS, 1825)


class TestVolumeExpand(unittest.TestCase):
    def test_spike_is_expand(self):
        weekly = [{"volume": 1_000_000.0} for _ in range(12)]
        weekly.append({"volume": 2_000_000.0})
        self.assertTrue(volume_expand(weekly))

    def test_quiet_week_is_not_expand(self):
        weekly = [{"volume": 1_000_000.0} for _ in range(13)]
        self.assertFalse(volume_expand(weekly))


class TestDynamicUniverse(unittest.TestCase):
    def test_does_not_read_static_txt(self):
        from pathlib import Path
        static = Path("/Users/anuppamvi/tradedesk/trendbook/configs/universe.txt")
        names = dynamic_universe("2026-09-07", live=False)
        self.assertTrue(any(n in names for n in infra_tickers()))
        self.assertIn("SPY", names)
        # Memory/book can include MU without a static list.
        self.assertIn("MU", names)
        if static.is_file():
            text = static.read_text(encoding="utf-8")
            self.assertTrue(text.strip().startswith("#"))


class TestLiveHighQuote(unittest.TestCase):
    def test_near_52w_high(self):
        self.assertTrue(quote_is_52w_high({"last": 99.0, "high52": 100.0}))
        self.assertFalse(quote_is_52w_high({"last": 90.0, "high52": 100.0}))
        self.assertFalse(quote_is_52w_high({"last": 5.0, "high52": 5.1}))


class TestBreakoutVolume(unittest.TestCase):
    def test_quiet_break_week_is_never_add(self):
        kwargs = dict(
            entry_reason="in_trend_wait",
            hh_hl=True,
            mansfield_rising=True,
            mansfield_spy=0.10,
            residual_63=0.10,
            break_vol_expand=False,
            from_base=True,
        )
        self.assertEqual(assign_action(True, False, "held", weeks=1, **kwargs), "NEW")
        self.assertEqual(assign_action(True, False, "held", weeks=3, **kwargs), "NEW")

    def test_already_bought_is_hold(self):
        self.assertEqual(
            assign_action(
                True,
                False,
                "extended",
                weeks=3,
                entry_reason="chase_3.0_atr",
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.10,
                residual_63=0.10,
                vol_expand=True,
                from_base=True,
                already_bought=True,
            ),
            "HOLD",
        )

    def test_regime_off_blocks_add(self):
        self.assertEqual(
            assign_action(
                True,
                False,
                "extended",
                weeks=2,
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.10,
                residual_63=0.10,
                vol_expand=True,
                from_base=True,
                regime_risk="off",
            ),
            "HOLD",
        )

    def test_pullback_needs_nonnegative_residual(self):
        self.assertEqual(
            assign_action(
                True,
                False,
                "ready",
                weeks=6,
                entry_reason="pullback",
                off_high=0.08,
                residual_63=-0.04,
            ),
            "HOLD",
        )

    def test_week1_without_volume_is_not_breakout(self):
        self.assertEqual(
            assign_action(
                True,
                False,
                "extended",
                weeks=1,
                entry_reason="chase_3.0_atr",
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.10,
                rs_63=0.12,
                residual_63=0.08,
                vol_expand=False,
                from_base=True,
            ),
            "NEW",
        )
        self.assertEqual(
            assign_action(
                True,
                False,
                "extended",
                weeks=1,
                entry_reason="chase_3.0_atr",
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.10,
                rs_63=0.12,
                residual_63=0.08,
                vol_expand=True,
                from_base=True,
            ),
            "ADD",
        )

    def test_stage3_reclaim_is_not_breakout(self):
        self.assertEqual(
            assign_action(
                True,
                False,
                "held",
                weeks=2,
                entry_reason="in_trend_wait",
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.10,
                residual_63=0.31,
                vol_expand=True,
                from_base=False,
            ),
            "NEW",
        )

    def test_negative_residual_blocks_breakout(self):
        self.assertEqual(
            assign_action(
                True,
                False,
                "held",
                weeks=2,
                entry_reason="in_trend_wait",
                hh_hl=True,
                mansfield_rising=True,
                mansfield_spy=0.10,
                rs_63=0.20,
                residual_63=-0.05,
                vol_expand=True,
                from_base=True,
            ),
            "NEW",
        )


if __name__ == "__main__":
    unittest.main()
