import json
import tempfile
import unittest
from pathlib import Path

from cultra.catalog import CATALOG_VERSION, FROZEN_STRATEGY_CATALOG
from cultra.config import ConfigurationError, load_evidence_policy, load_strategy_catalog


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    def test_shipped_evidence_policy_preserves_user_control(self):
        policy = load_evidence_policy(ROOT / "configs" / "evidence_policy.v1.json")
        self.assertFalse(policy.portfolio_gates_enabled)
        self.assertIsNone(policy.output_top_n_cap)
        self.assertEqual(policy.quantity_policy, "USER DETERMINED")
        self.assertTrue(policy.manual_ticket_requires_finite_max_loss)
        self.assertEqual("HOLDOUT_PASS", policy.manual_action_minimum_evidence_state)
        self.assertEqual(
            "CONTINUOUS_NONBLOCKING_REVOCATION_MONITOR",
            policy.prospective_shadow_mode,
        )
        self.assertEqual(0, policy.shadow_minimum_calendar_days_before_action)

    def test_shipped_strategy_catalog_is_frozen_and_unique(self):
        catalog = load_strategy_catalog(ROOT / "configs" / "strategy_catalog.v1.json")
        self.assertEqual(catalog.catalog_version, CATALOG_VERSION)
        self.assertEqual(len(catalog.families), len(set(catalog.families)))
        self.assertEqual(
            catalog.families,
            [item.strategy_id for item in FROZEN_STRATEGY_CATALOG],
        )
        self.assertIn("NAKED_CALL", catalog.families)
        self.assertIn("IRON_CONDOR", catalog.families)

    def test_rejects_strategy_catalog_drift(self):
        raw = json.loads((ROOT / "configs" / "strategy_catalog.v1.json").read_text())
        raw["families"] = raw["families"][:-1]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad-catalog.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ConfigurationError):
                load_strategy_catalog(path)

    def test_rejects_portfolio_gate(self):
        raw = json.loads((ROOT / "configs" / "evidence_policy.v1.json").read_text())
        raw["portfolio_gates_enabled"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ConfigurationError):
                load_evidence_policy(path)


if __name__ == "__main__":
    unittest.main()
