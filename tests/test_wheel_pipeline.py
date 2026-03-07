import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "uwos" / "wheel_pipeline.py"
SPEC = importlib.util.spec_from_file_location("uwos_wheel_pipeline", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(SPEC)
sys.modules["uwos_wheel_pipeline"] = mod
SPEC.loader.exec_module(mod)


class TestTierScore(unittest.TestCase):
    """Tests for tier_score() helper — maps a metric to 0-100 via thresholds."""

    # --- higher_is_better (default) ---

    def test_higher_excellent(self) -> None:
        # ROE 20 >= excellent(15) => 100
        score = mod.tier_score(20.0, excellent=15, good=10, fair=5)
        self.assertEqual(score, 100)

    def test_higher_good(self) -> None:
        # ROE 12 >= good(10) but < excellent(15) => 75
        score = mod.tier_score(12.0, excellent=15, good=10, fair=5)
        self.assertEqual(score, 75)

    def test_higher_fair(self) -> None:
        # ROE 7 >= fair(5) but < good(10) => 50
        score = mod.tier_score(7.0, excellent=15, good=10, fair=5)
        self.assertEqual(score, 50)

    def test_higher_below(self) -> None:
        # ROE 3 < fair(5) => 25
        score = mod.tier_score(3.0, excellent=15, good=10, fair=5)
        self.assertEqual(score, 25)

    # --- lower_is_better ---

    def test_lower_excellent(self) -> None:
        # D/E 0.3 <= excellent(0.5) => 100
        score = mod.tier_score(0.3, excellent=0.5, good=1.0, fair=2.0, lower_is_better=True)
        self.assertEqual(score, 100)

    def test_lower_good(self) -> None:
        # D/E 0.7 <= good(1.0) but > excellent(0.5) => 75
        score = mod.tier_score(0.7, excellent=0.5, good=1.0, fair=2.0, lower_is_better=True)
        self.assertEqual(score, 75)

    def test_lower_fair(self) -> None:
        # D/E 1.5 <= fair(2.0) but > good(1.0) => 50
        score = mod.tier_score(1.5, excellent=0.5, good=1.0, fair=2.0, lower_is_better=True)
        self.assertEqual(score, 50)

    def test_lower_below(self) -> None:
        # D/E 2.5 > fair(2.0) => 25
        score = mod.tier_score(2.5, excellent=0.5, good=1.0, fair=2.0, lower_is_better=True)
        self.assertEqual(score, 25)

    # --- edge cases: exact boundary values ---

    def test_exact_excellent_boundary_higher(self) -> None:
        score = mod.tier_score(15.0, excellent=15, good=10, fair=5)
        self.assertEqual(score, 100)

    def test_exact_good_boundary_higher(self) -> None:
        score = mod.tier_score(10.0, excellent=15, good=10, fair=5)
        self.assertEqual(score, 75)

    def test_exact_fair_boundary_higher(self) -> None:
        score = mod.tier_score(5.0, excellent=15, good=10, fair=5)
        self.assertEqual(score, 50)

    def test_exact_excellent_boundary_lower(self) -> None:
        score = mod.tier_score(0.5, excellent=0.5, good=1.0, fair=2.0, lower_is_better=True)
        self.assertEqual(score, 100)


class TestDataclasses(unittest.TestCase):
    """Tests for dataclass construction and defaults."""

    def test_quality_score_defaults(self) -> None:
        qs = mod.QualityScore()
        self.assertEqual(qs.composite, 0.0)
        self.assertFalse(qs.disqualified)
        self.assertEqual(qs.disqualify_reason, "")

    def test_quality_score_construction(self) -> None:
        qs = mod.QualityScore(roe=18.0, roe_score=100, composite=85.0)
        self.assertEqual(qs.roe, 18.0)
        self.assertEqual(qs.roe_score, 100)
        self.assertEqual(qs.composite, 85.0)

    def test_premium_score_defaults(self) -> None:
        ps = mod.PremiumScore()
        self.assertEqual(ps.composite, 0.0)
        self.assertEqual(ps.csp_yield_ann, 0.0)

    def test_premium_score_construction(self) -> None:
        ps = mod.PremiumScore(csp_yield_ann=35.0, csp_yield_score=75, iv_rank=55.0)
        self.assertEqual(ps.csp_yield_ann, 35.0)
        self.assertEqual(ps.csp_yield_score, 75)
        self.assertEqual(ps.iv_rank, 55.0)

    def test_sentiment_adjustment_defaults(self) -> None:
        sa = mod.SentimentAdjustment()
        self.assertEqual(sa.total, 0.0)
        self.assertIsNone(sa.earnings_days_away)
        self.assertEqual(sa.notes, [])

    def test_sentiment_adjustment_notes_isolation(self) -> None:
        """Ensure mutable default (list) is not shared between instances."""
        sa1 = mod.SentimentAdjustment()
        sa2 = mod.SentimentAdjustment()
        sa1.notes.append("test")
        self.assertEqual(len(sa2.notes), 0)

    def test_wheel_candidate_defaults(self) -> None:
        wc = mod.WheelCandidate()
        self.assertEqual(wc.ticker, "")
        self.assertEqual(wc.composite, 0.0)
        self.assertEqual(wc.tier, "")
        self.assertFalse(wc.live_validated)
        self.assertEqual(wc.notes, [])

    def test_wheel_position_defaults(self) -> None:
        wp = mod.WheelPosition()
        self.assertEqual(wp.ticker, "")
        self.assertEqual(wp.phase, "csp")
        self.assertEqual(wp.wheel_cycles, 0)

    def test_daily_action_defaults(self) -> None:
        da = mod.DailyAction()
        self.assertEqual(da.ticker, "")
        self.assertEqual(da.action, "")
        self.assertEqual(da.icon, "")


class TestFilterUniverse(unittest.TestCase):
    """Tests for filter_universe() — filters DataFrame by price, mcap, option volume."""

    def _make_cfg(self) -> dict:
        return {
            "universe": {
                "min_price": 10,
                "max_price": 60,
                "min_option_volume": 500,
                "min_market_cap_b": 2.0,
            }
        }

    def test_filters_by_price(self) -> None:
        import pandas as pd
        df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D"],
            "close": [5.0, 25.0, 45.0, 80.0],
            "option_volume": [1000, 1000, 1000, 1000],
            "market_cap": [5e9, 5e9, 5e9, 5e9],
        })
        result = mod.filter_universe(df, self._make_cfg())
        self.assertListEqual(list(result["ticker"]), ["B", "C"])

    def test_filters_by_market_cap(self) -> None:
        import pandas as pd
        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "close": [30.0, 30.0, 30.0],
            "option_volume": [1000, 1000, 1000],
            "market_cap": [1e9, 2e9, 10e9],
        })
        result = mod.filter_universe(df, self._make_cfg())
        self.assertListEqual(list(result["ticker"]), ["B", "C"])

    def test_filters_by_option_volume(self) -> None:
        import pandas as pd
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "close": [30.0, 30.0],
            "option_volume": [100, 1000],
            "market_cap": [5e9, 5e9],
        })
        result = mod.filter_universe(df, self._make_cfg())
        self.assertListEqual(list(result["ticker"]), ["B"])


class TestScoreQuality(unittest.TestCase):
    """Tests for score_quality() — fundamental quality scoring."""

    def _make_cfg(self) -> dict:
        import yaml
        config_path = Path(__file__).resolve().parents[1] / "uwos" / "wheel_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_high_quality_stock(self) -> None:
        cfg = self._make_cfg()
        fundamentals = {
            "roe": 20, "debt_equity": 0.3, "rev_growth_yoy": 18,
            "fcf_yield": 6, "pe_ratio": 12, "earnings_beats": 4,
            "mean_reversion_rate": 80,
        }
        qs = mod.score_quality(fundamentals, cfg)
        self.assertFalse(qs.disqualified)
        self.assertGreater(qs.composite, 85)

    def test_disqualified_high_debt(self) -> None:
        cfg = self._make_cfg()
        fundamentals = {
            "roe": 15, "debt_equity": 3.5, "rev_growth_yoy": 10,
            "fcf_yield": 4, "pe_ratio": 20, "earnings_beats": 3,
            "mean_reversion_rate": 60,
        }
        qs = mod.score_quality(fundamentals, cfg)
        self.assertTrue(qs.disqualified)

    def test_low_quality_stock(self) -> None:
        cfg = self._make_cfg()
        fundamentals = {
            "roe": -5, "debt_equity": 2.5, "rev_growth_yoy": -2,
            "fcf_yield": 0.5, "pe_ratio": 50, "earnings_beats": 0,
            "mean_reversion_rate": 10,
        }
        qs = mod.score_quality(fundamentals, cfg)
        self.assertLess(qs.composite, 35)

    def test_negative_pe_handling(self) -> None:
        cfg = self._make_cfg()
        fundamentals = {
            "roe": 10, "debt_equity": 1.0, "rev_growth_yoy": 5,
            "fcf_yield": 3, "pe_ratio": -10, "earnings_beats": 2,
            "mean_reversion_rate": 50,
        }
        qs = mod.score_quality(fundamentals, cfg)
        self.assertEqual(qs.pe_score, 10)


class TestConfigLoad(unittest.TestCase):
    """Verify the YAML config loads and has expected keys."""

    def test_config_loads(self) -> None:
        import yaml
        config_path = Path(__file__).resolve().parents[1] / "uwos" / "wheel_config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.assertIn("pipeline", cfg)
        self.assertIn("universe", cfg)
        self.assertIn("scoring", cfg)
        self.assertIn("sentiment", cfg)
        self.assertIn("allocation", cfg)
        self.assertIn("management", cfg)
        self.assertIn("schwab_validation", cfg)
        self.assertIn("output", cfg)

    def test_scoring_weights_sum(self) -> None:
        import yaml
        config_path = Path(__file__).resolve().parents[1] / "uwos" / "wheel_config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        scoring = cfg["scoring"]
        self.assertAlmostEqual(
            scoring["quality_weight"] + scoring["premium_weight"], 1.0, places=6
        )


class TestComputeMeanReversion(unittest.TestCase):
    """Tests for compute_mean_reversion() — drawdown recovery rate."""

    def _make_uptrend(self, n: int = 100) -> "pd.DataFrame":
        """Create a steady uptrend with no drawdowns."""
        import pandas as pd
        import numpy as np
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        closes = 100 + np.arange(n) * 0.5  # steady rise
        return pd.DataFrame({"Close": closes}, index=dates)

    def test_perfect_recovery(self) -> None:
        """Two drawdowns that recover quickly should yield rate >= 75%."""
        import pandas as pd
        import numpy as np
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        prices = np.full(n, 100.0)
        # First drawdown at day 30: drop 15%, recover by day 45
        for i in range(30, 40):
            prices[i] = 100 - (15 * (i - 29) / 10)
        for i in range(40, 50):
            prices[i] = 85 + (15 * (i - 39) / 10)
        # Second drawdown at day 100: drop 12%, recover by day 115
        for i in range(100, 108):
            prices[i] = 100 - (12 * (i - 99) / 8)
        for i in range(108, 118):
            prices[i] = 88 + (12 * (i - 107) / 10)
        df = pd.DataFrame({"Close": prices}, index=dates)
        rate = mod.compute_mean_reversion(df, drawdown_pct=10, recovery_days=30)
        self.assertGreaterEqual(rate, 75.0)

    def test_no_drawdowns(self) -> None:
        """Steady uptrend with no drawdowns should return 100.0."""
        df = self._make_uptrend(100)
        rate = mod.compute_mean_reversion(df, drawdown_pct=10, recovery_days=30)
        self.assertEqual(rate, 100.0)

    def test_insufficient_data(self) -> None:
        """Less than 30 rows should return 50.0."""
        import pandas as pd
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        df = pd.DataFrame({"Close": [100.0] * 20}, index=dates)
        rate = mod.compute_mean_reversion(df, drawdown_pct=10, recovery_days=30)
        self.assertEqual(rate, 50.0)

    def test_no_recovery(self) -> None:
        """Drawdowns that never recover should yield a low rate."""
        import pandas as pd
        import numpy as np
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Start at 100, drop to 80 at day 20, stay at 80 forever
        prices = np.full(n, 80.0)
        for i in range(20):
            prices[i] = 100 - (i * 1.0)
        df = pd.DataFrame({"Close": prices}, index=dates)
        rate = mod.compute_mean_reversion(df, drawdown_pct=10, recovery_days=30)
        self.assertLessEqual(rate, 25.0)


if __name__ == "__main__":
    unittest.main()
