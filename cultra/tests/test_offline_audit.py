import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cultra.offline_audit import OUT_ROOT, request_feasibility, verify_offline_audit


class OfflineAuditTests(unittest.TestCase):
    def test_censored_rotating_campaign_replaces_full_universe_date_grid(self):
        result = request_feasibility(
            symbol_counts=(254,),
            batch_size=10,
            minimum_training_sessions=120,
            embargo_sessions=60,
            validation_sessions=40,
            holdout_fraction=0.20,
            reference_sessions=450,
            attempt_cap=90,
        )
        self.assertEqual(275, result["minimum_total_sessions"])
        scenario = result["scenarios"][0]
        self.assertEqual(11700, scenario["rejected_full_universe_date_grid_attempts"])
        self.assertEqual(474, scenario["expected_campaign_attempts"])
        self.assertEqual(6, scenario["slice_count"])
        campaign = scenario["rotating_cohort_campaign"]
        self.assertEqual(20, campaign["requests"]["historical_core"])
        self.assertEqual(450, campaign["requests"]["historical_chain_total"])
        self.assertEqual(4, campaign["requests"]["split_history"])
        self.assertFalse(campaign["execution_authorized"])

    def test_saved_audit_verifier_detects_artifact_drift(self):
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(OUT_ROOT)) as temporary:
            root = Path(temporary)
            run_id = root.name
            source = Path(__file__).resolve()
            audit_path = root / "offline_audit.json"
            board_path = root / "OFFLINE_AUDIT.md"
            audit_path.write_text(
                json.dumps(
                    {
                        "schema": "cultra.offline-audit.v7",
                        "run_id": run_id,
                        "network_attempted": False,
                    }
                )
            )
            board_path.write_text("network requests: 0\n")
            digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "cultra.offline-audit-manifest.v7",
                        "run_id": run_id,
                        "network_attempted": False,
                        "inputs": [
                            {
                                "path": str(source),
                                "bytes": source.stat().st_size,
                                "sha256": digest(source),
                            }
                        ],
                        "artifacts": [
                            {
                                "path": path.name,
                                "bytes": path.stat().st_size,
                                "sha256": digest(path),
                            }
                            for path in (audit_path, board_path)
                        ],
                    }
                )
            )
            self.assertEqual((), verify_offline_audit(root))
            board_path.write_text("changed\n")
            self.assertTrue(verify_offline_audit(root))


if __name__ == "__main__":
    unittest.main()
