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
            "earnings_growth_est": 25, "analyst_upside": 22,
            "institutional_pct": 75,
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


class TestComputeSigmaStrike(unittest.TestCase):
    """Tests for compute_sigma_strike() — Black-Scholes-style strike placement."""

    def test_put_strike_below_spot(self) -> None:
        strike = mod.compute_sigma_strike(spot=50, iv=0.30, dte=30, side="put", sigma=1.0)
        self.assertLess(strike, 50)
        self.assertGreater(strike, 40)

    def test_call_strike_above_spot(self) -> None:
        strike = mod.compute_sigma_strike(spot=50, iv=0.30, dte=30, side="call", sigma=1.0)
        self.assertGreater(strike, 50)
        self.assertLess(strike, 60)

    def test_zero_iv(self) -> None:
        strike = mod.compute_sigma_strike(spot=50, iv=0, dte=30, side="put", sigma=1.0)
        self.assertEqual(strike, 50)


class TestScorePremium(unittest.TestCase):
    """Tests for score_premium() — option premium attractiveness scoring."""

    def _make_cfg(self) -> dict:
        import yaml
        config_path = Path(__file__).resolve().parents[1] / "uwos" / "wheel_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_high_premium_stock(self) -> None:
        cfg = self._make_cfg()
        chain_data = {
            "csp_premium": 2.0, "csp_strike": 17, "cc_premium": 1.5,
            "cc_strike": 21, "spot": 19, "iv_rank": 55, "spread_pct": 1.5, "dte": 30,
        }
        ps = mod.score_premium(chain_data, cfg)
        self.assertGreater(ps.composite, 70)

    def test_low_premium_stock(self) -> None:
        cfg = self._make_cfg()
        chain_data = {
            "csp_premium": 0.05, "csp_strike": 50, "cc_premium": 0.03,
            "cc_strike": 55, "spot": 52, "iv_rank": 10, "spread_pct": 8.0, "dte": 45,
        }
        ps = mod.score_premium(chain_data, cfg)
        self.assertLess(ps.composite, 40)

    def test_zero_dte_defaults(self) -> None:
        cfg = self._make_cfg()
        chain_data = {
            "csp_premium": 1.0, "csp_strike": 20, "cc_premium": 0.8,
            "cc_strike": 22, "spot": 21, "iv_rank": 40, "spread_pct": 3.0, "dte": 0,
        }
        ps = mod.score_premium(chain_data, cfg)
        # Should not crash and should use default dte of 30
        self.assertGreater(ps.composite, 0)


class TestSentiment(unittest.TestCase):
    """Tests for apply_sentiment() — sentiment overlay adjustments."""

    def _make_cfg(self) -> dict:
        import yaml
        config_path = Path(__file__).resolve().parents[1] / "uwos" / "wheel_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_bullish_boost(self) -> None:
        cfg = self._make_cfg()
        sa = mod.apply_sentiment(
            swing_direction="bullish", swing_verdict="PASS",
            whale_score=75, dp_bearish=False, earnings_days=None,
            oi_confirms=True, cfg=cfg,
        )
        self.assertGreater(sa.total, 0)
        self.assertLessEqual(sa.total, 10)

    def test_bearish_penalty(self) -> None:
        cfg = self._make_cfg()
        sa = mod.apply_sentiment(
            swing_direction="bearish", swing_verdict="FAIL",
            whale_score=30, dp_bearish=True, earnings_days=10,
            oi_confirms=False, cfg=cfg,
        )
        self.assertLess(sa.total, 0)
        self.assertGreaterEqual(sa.total, -10)

    def test_neutral_no_adjustment(self) -> None:
        cfg = self._make_cfg()
        sa = mod.apply_sentiment(
            swing_direction="", swing_verdict="",
            whale_score=30, dp_bearish=False, earnings_days=None,
            oi_confirms=False, cfg=cfg,
        )
        self.assertEqual(sa.total, 0)

    def test_clamping(self) -> None:
        cfg = self._make_cfg()
        sa = mod.apply_sentiment(
            swing_direction="bullish", swing_verdict="PASS",
            whale_score=80, dp_bearish=False, earnings_days=None,
            oi_confirms=True, cfg=cfg,
        )
        self.assertEqual(sa.total, 10)


class TestComposite(unittest.TestCase):
    """Tests for compute_composite() — weighted composite scoring."""

    def test_composite_weighting(self) -> None:
        result = mod.compute_composite(quality=80, premium=60, sentiment=0)
        self.assertAlmostEqual(result, 74.0, places=1)

    def test_with_sentiment(self) -> None:
        result = mod.compute_composite(quality=80, premium=60, sentiment=5)
        self.assertAlmostEqual(result, 79.0, places=1)

    def test_clamp_floor(self) -> None:
        result = mod.compute_composite(quality=5, premium=5, sentiment=-10)
        self.assertGreaterEqual(result, 0)

    def test_clamp_ceiling(self) -> None:
        result = mod.compute_composite(quality=100, premium=100, sentiment=10)
        self.assertLessEqual(result, 100)


class TestAssignTier(unittest.TestCase):
    """Tests for assign_tier() — composite-to-tier mapping."""

    def _make_cfg(self) -> dict:
        import yaml
        config_path = Path(__file__).resolve().parents[1] / "uwos" / "wheel_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_core(self) -> None:
        cfg = self._make_cfg()
        self.assertEqual(mod.assign_tier(65, cfg), "core")

    def test_aggressive(self) -> None:
        cfg = self._make_cfg()
        self.assertEqual(mod.assign_tier(50, cfg), "aggressive")

    def test_watchlist(self) -> None:
        cfg = self._make_cfg()
        self.assertEqual(mod.assign_tier(40, cfg), "watchlist")

    def test_excluded(self) -> None:
        cfg = self._make_cfg()
        self.assertEqual(mod.assign_tier(30, cfg), "excluded")

    def test_boundary_core(self) -> None:
        cfg = self._make_cfg()
        self.assertEqual(mod.assign_tier(60, cfg), "core")


class TestAllocateCapital(unittest.TestCase):
    """Tests for allocate_capital() — capital allocation with position limits."""

    def _make_cfg(self) -> dict:
        return {
            "allocation": {
                "max_deployed_pct": 0.65,
                "max_single_name_pct": 0.25,
                "max_positions": 5,
            }
        }

    def _make_candidate(self, ticker: str, composite: float, csp_strike: float) -> "mod.WheelCandidate":
        wc = mod.WheelCandidate(ticker=ticker, composite=composite)
        wc.premium = mod.PremiumScore(csp_strike=csp_strike)
        return wc

    def test_basic_allocation(self) -> None:
        """2 candidates, $35K capital → both get contracts, total <= 65% of capital."""
        cfg = self._make_cfg()
        c1 = self._make_candidate("AAPL", 80, 45.0)
        c2 = self._make_candidate("MSFT", 70, 30.0)
        result = mod.allocate_capital([c1, c2], 35000, cfg)
        self.assertEqual(len(result), 2)
        total = sum(c.capital_required for c in result)
        self.assertLessEqual(total, 35000 * 0.65)
        for c in result:
            self.assertGreater(c.max_contracts, 0)
            self.assertGreater(c.capital_required, 0)

    def test_single_name_limit(self) -> None:
        """1 expensive candidate → capital_required <= 25% of total."""
        cfg = self._make_cfg()
        c1 = self._make_candidate("TSLA", 90, 50.0)
        result = mod.allocate_capital([c1], 35000, cfg)
        self.assertEqual(len(result), 1)
        self.assertLessEqual(result[0].capital_required, 35000 * 0.25)

    def test_excludes_zero_strike(self) -> None:
        """Candidate with csp_strike=0 → excluded from result."""
        cfg = self._make_cfg()
        c1 = self._make_candidate("BAD", 90, 0.0)
        c2 = self._make_candidate("GOOD", 80, 20.0)
        result = mod.allocate_capital([c1, c2], 35000, cfg)
        tickers = [c.ticker for c in result]
        self.assertNotIn("BAD", tickers)
        self.assertIn("GOOD", tickers)

    def test_max_positions_limit(self) -> None:
        """6 candidates but max_positions=5 → only 5 returned."""
        cfg = self._make_cfg()
        candidates = [self._make_candidate(f"T{i}", 90 - i, 15.0) for i in range(6)]
        result = mod.allocate_capital(candidates, 100000, cfg)
        self.assertLessEqual(len(result), 5)

    def test_sorts_by_composite(self) -> None:
        """Lower composite candidate should not get priority over higher."""
        cfg = self._make_cfg()
        c_low = self._make_candidate("LOW", 50, 20.0)
        c_high = self._make_candidate("HIGH", 90, 20.0)
        result = mod.allocate_capital([c_low, c_high], 35000, cfg)
        self.assertEqual(result[0].ticker, "HIGH")


class TestExtractChainData(unittest.TestCase):
    """Tests for extract_chain_data() — Schwab chain data extraction."""

    def _make_chain(self) -> dict:
        return {
            "underlying": {"mark": 40.0},
            "putExpDateMap": {
                "2026-04-17:42": {
                    "38.0": [{"bid": 0.80, "ask": 0.90, "mark": 0.85,
                              "openInterest": 500, "totalVolume": 100}],
                    "36.0": [{"bid": 0.40, "ask": 0.50, "mark": 0.45,
                              "openInterest": 200, "totalVolume": 50}],
                }
            },
            "callExpDateMap": {
                "2026-04-17:42": {
                    "42.0": [{"bid": 0.70, "ask": 0.80, "mark": 0.75,
                              "openInterest": 400, "totalVolume": 80}],
                    "44.0": [{"bid": 0.30, "ask": 0.40, "mark": 0.35,
                              "openInterest": 150, "totalVolume": 30}],
                }
            },
        }

    def test_extracts_csp_and_cc(self) -> None:
        chain = self._make_chain()
        result = mod.extract_chain_data(chain, spot=40.0, iv=0.30, dte_target=30, sigma=1.0)
        self.assertGreater(result["csp_premium"], 0)
        self.assertGreater(result["cc_premium"], 0)
        self.assertEqual(result["dte"], 42)
        self.assertIn(result["csp_strike"], [38.0, 36.0])
        self.assertIn(result["cc_strike"], [42.0, 44.0])

    def test_empty_chain(self) -> None:
        chain = {
            "underlying": {"mark": 40.0},
            "putExpDateMap": {},
            "callExpDateMap": {},
        }
        result = mod.extract_chain_data(chain, spot=40.0, iv=0.30, dte_target=30, sigma=1.0)
        self.assertEqual(result["csp_premium"], 0.0)
        self.assertEqual(result["cc_premium"], 0.0)
        self.assertEqual(result["csp_strike"], 0.0)
        self.assertEqual(result["cc_strike"], 0.0)

    def test_picks_closest_expiry(self) -> None:
        chain = {
            "underlying": {"mark": 40.0},
            "putExpDateMap": {
                "2026-03-25:18": {
                    "38.0": [{"bid": 0.50, "ask": 0.60, "mark": 0.55,
                              "openInterest": 100, "totalVolume": 20}],
                },
                "2026-04-13:37": {
                    "38.0": [{"bid": 0.90, "ask": 1.00, "mark": 0.95,
                              "openInterest": 300, "totalVolume": 60}],
                },
            },
            "callExpDateMap": {
                "2026-03-25:18": {
                    "42.0": [{"bid": 0.40, "ask": 0.50, "mark": 0.45,
                              "openInterest": 80, "totalVolume": 15}],
                },
                "2026-04-13:37": {
                    "42.0": [{"bid": 0.80, "ask": 0.90, "mark": 0.85,
                              "openInterest": 250, "totalVolume": 50}],
                },
            },
        }
        result = mod.extract_chain_data(chain, spot=40.0, iv=0.30, dte_target=30, sigma=1.0)
        # 37d is closer to 30 than 18d (|37-30|=7 < |18-30|=12)
        self.assertEqual(result["dte"], 37)
        self.assertGreater(result["csp_premium"], 0)


class TestPositionTracker(unittest.TestCase):
    """Tests for PositionTracker — JSON-backed position persistence."""

    def _tmp_path(self) -> Path:
        import tempfile
        return Path(tempfile.mkdtemp()) / "positions.json"

    def _make_position(self, ticker="AAPL", phase="csp") -> "mod.WheelPosition":
        return mod.WheelPosition(
            ticker=ticker, tier="core", phase=phase,
            entry_date="2026-03-07", strike=45.0, expiry="2026-04-17",
            contracts=1, shares=0, entry_premium=0.85,
            cost_basis=0.0, capital_reserved=4500.0,
            cumulative_premium=0.85, assignment_count=0, wheel_cycles=0,
        )

    def test_load_empty(self) -> None:
        """Non-existent path => empty positions list."""
        p = self._tmp_path()
        tracker = mod.PositionTracker(p)
        self.assertEqual(len(tracker.positions), 0)
        self.assertEqual(len(tracker.premium_journal), 0)

    def test_add_and_save_and_reload(self) -> None:
        """Add position, save, reload from same path => position preserved."""
        p = self._tmp_path()
        tracker = mod.PositionTracker(p)
        pos = self._make_position()
        tracker.add_position(pos)
        tracker.save()

        tracker2 = mod.PositionTracker(p)
        self.assertEqual(len(tracker2.positions), 1)
        self.assertEqual(tracker2.positions[0].ticker, "AAPL")
        self.assertEqual(tracker2.positions[0].strike, 45.0)
        self.assertEqual(tracker2.positions[0].phase, "csp")

    def test_remove_position(self) -> None:
        """Add 2 positions, remove 1 => 1 remains."""
        p = self._tmp_path()
        tracker = mod.PositionTracker(p)
        tracker.add_position(self._make_position("AAPL", "csp"))
        tracker.add_position(self._make_position("MSFT", "shares"))
        self.assertEqual(len(tracker.positions), 2)
        tracker.remove_position("AAPL", "csp")
        self.assertEqual(len(tracker.positions), 1)
        self.assertEqual(tracker.positions[0].ticker, "MSFT")

    def test_log_premium(self) -> None:
        """Log an entry => journal length increases."""
        p = self._tmp_path()
        tracker = mod.PositionTracker(p)
        tracker.log_premium("2026-03-07", "AAPL", "SELL_CSP", 0.85, notes="initial")
        self.assertEqual(len(tracker.premium_journal), 1)
        self.assertEqual(tracker.premium_journal[0]["ticker"], "AAPL")
        self.assertEqual(tracker.premium_journal[0]["amount"], 0.85)


class TestDailyManager(unittest.TestCase):
    """Tests for DailyManager — daily decision matrix for wheel positions."""

    def _make_cfg(self) -> dict:
        return {
            "management": {
                "close_target_pct": 0.50,
                "dte_target": 30,
                "dte_roll_threshold": 14,
                "sigma_otm": 1.0,
                "max_unrealized_loss_pct": -0.50,
                "max_consecutive_assignments": 2,
                "sector_concentration_limit": 0.40,
                "earnings_close_days": 7,
            }
        }

    def _make_csp_pos(self, entry_premium=0.85) -> "mod.WheelPosition":
        return mod.WheelPosition(
            ticker="AAPL", phase="csp", entry_premium=entry_premium,
            strike=45.0, expiry="2026-04-17", contracts=1,
            capital_reserved=4500.0,
        )

    def _make_shares_pos(self) -> "mod.WheelPosition":
        return mod.WheelPosition(
            ticker="AAPL", phase="shares", strike=45.0,
            shares=100, cost_basis=45.0, capital_reserved=4500.0,
        )

    def _make_cc_pos(self, entry_premium=0.90, strike=48.0) -> "mod.WheelPosition":
        return mod.WheelPosition(
            ticker="AAPL", phase="cc", entry_premium=entry_premium,
            strike=strike, expiry="2026-04-17", contracts=1,
            capital_reserved=4500.0,
        )

    # --- CSP tests ---

    def test_csp_close_at_50pct(self) -> None:
        """entry=0.85, current=0.40 => pnl_pct ~53% >= 50% => CLOSE."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_csp_pos(entry_premium=0.85)
        action = mgr.evaluate_csp(pos, current_premium=0.40, dte=25)
        self.assertEqual(action.action, "CLOSE")
        self.assertGreater(action.pnl_pct, 0.50)

    def test_csp_hold_not_target(self) -> None:
        """entry=0.85, current=0.70 => pnl_pct ~18% < 50% => HOLD."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_csp_pos(entry_premium=0.85)
        action = mgr.evaluate_csp(pos, current_premium=0.70, dte=25)
        self.assertEqual(action.action, "HOLD")

    def test_csp_roll_low_dte_losing(self) -> None:
        """entry=0.95, current=1.20, dte=10 => losing + low DTE => ROLL."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_csp_pos(entry_premium=0.95)
        action = mgr.evaluate_csp(pos, current_premium=1.20, dte=10)
        self.assertEqual(action.action, "ROLL")

    def test_csp_close_low_dte_winning(self) -> None:
        """entry=0.85, current=0.30, dte=10 => winning + low DTE => CLOSE."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_csp_pos(entry_premium=0.85)
        action = mgr.evaluate_csp(pos, current_premium=0.30, dte=10)
        self.assertEqual(action.action, "CLOSE")

    # --- Shares tests ---

    def test_shares_bullish(self) -> None:
        """Bullish signal => SELL_CC."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_shares_pos()
        action = mgr.evaluate_shares(pos, spot=46.0, signal="bullish")
        self.assertEqual(action.action, "SELL_CC")

    def test_shares_bearish(self) -> None:
        """Bearish signal => SELL_CC."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_shares_pos()
        action = mgr.evaluate_shares(pos, spot=44.0, signal="bearish")
        self.assertEqual(action.action, "SELL_CC")

    # --- CC tests ---

    def test_cc_close_at_target(self) -> None:
        """entry=0.90, current=0.40 => pnl_pct ~56% >= 50% => CLOSE."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_cc_pos(entry_premium=0.90, strike=48.0)
        action = mgr.evaluate_cc(pos, current_premium=0.40, dte=20, spot=46.0)
        self.assertEqual(action.action, "CLOSE")

    def test_cc_called_away(self) -> None:
        """spot >= strike, dte=5 => ALLOW_CALL_AWAY."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_cc_pos(entry_premium=0.90, strike=48.0)
        action = mgr.evaluate_cc(pos, current_premium=1.50, dte=5, spot=49.0)
        self.assertEqual(action.action, "ALLOW_CALL_AWAY")

    def test_cc_hold(self) -> None:
        """Profitable but not at target => HOLD."""
        mgr = mod.DailyManager(self._make_cfg())
        pos = self._make_cc_pos(entry_premium=0.90, strike=48.0)
        action = mgr.evaluate_cc(pos, current_premium=0.60, dte=20, spot=46.0)
        self.assertEqual(action.action, "HOLD")


class TestGenerateSelectReport(unittest.TestCase):
    """Tests for generate_select_report() — weekly wheel selection markdown."""

    def _make_candidate(self, ticker, tier, composite, csp_strike, csp_premium,
                        dte=42, expiry="2026-04-17", max_contracts=1,
                        capital_required=3800.0):
        wc = mod.WheelCandidate(
            ticker=ticker, tier=tier, composite=composite, dte=dte,
            expiry=expiry, max_contracts=max_contracts,
            capital_required=capital_required,
            action=f"Sell ${csp_strike}P",
        )
        wc.premium = mod.PremiumScore(
            csp_strike=csp_strike, csp_premium=csp_premium,
            csp_yield_ann=27.3, csp_yield_score=75,
            composite=54.0,
        )
        wc.quality = mod.QualityScore(composite=75.0)
        return wc

    def test_contains_header(self) -> None:
        candidates = [self._make_candidate("BP", "core", 68.7, 38.0, 0.85)]
        output = mod.generate_select_report(candidates, 35000, "2026-03-07", {})
        self.assertIn("Wheel Selection Report", output)

    def test_contains_capital_line(self) -> None:
        candidates = [self._make_candidate("BP", "core", 68.7, 38.0, 0.85)]
        output = mod.generate_select_report(candidates, 35000, "2026-03-07", {})
        self.assertIn("Capital: $", output)

    def test_contains_tier_tables(self) -> None:
        candidates = [
            self._make_candidate("BP", "core", 68.7, 38.0, 0.85),
            self._make_candidate("RIOT", "aggressive", 55.0, 12.0, 0.45,
                                 capital_required=1200.0),
        ]
        output = mod.generate_select_report(candidates, 35000, "2026-03-07", {})
        self.assertIn("CORE", output)
        self.assertIn("AGGRESSIVE", output)


class TestGenerateDailyReport(unittest.TestCase):
    """Tests for generate_daily_report() — daily management markdown."""

    def _make_action(self, ticker="BP", action="CLOSE", phase="csp",
                     detail="Buy back $38P at $0.40", pnl_pct=0.53,
                     signal="neutral"):
        return mod.DailyAction(
            ticker=ticker, phase=phase, action=action,
            icon="V", detail=detail, pnl_pct=pnl_pct,
            current_premium=0.40, signal=signal, reason="profit_target",
        )

    def _make_position(self, ticker="BP", phase="csp"):
        return mod.WheelPosition(
            ticker=ticker, phase=phase, strike=38.0,
            expiry="2026-04-17", contracts=1, entry_premium=0.85,
            capital_reserved=3800.0,
        )

    def test_contains_header(self) -> None:
        output = mod.generate_daily_report(
            actions=[], positions=[], capital=35000,
            as_of="2026-03-07", journal=[],
        )
        self.assertIn("Wheel Daily", output)

    def test_contains_actions(self) -> None:
        actions = [self._make_action("BP", "CLOSE")]
        output = mod.generate_daily_report(
            actions=actions, positions=[self._make_position()],
            capital=35000, as_of="2026-03-07", journal=[],
        )
        self.assertIn("CLOSE", output)

    def test_contains_risk_dashboard(self) -> None:
        output = mod.generate_daily_report(
            actions=[], positions=[], capital=35000,
            as_of="2026-03-07", journal=[],
        )
        self.assertIn("Risk Dashboard", output)


if __name__ == "__main__":
    unittest.main()
