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


if __name__ == "__main__":
    unittest.main()
