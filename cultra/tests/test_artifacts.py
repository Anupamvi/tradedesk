import json
import os
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from cultra.artifacts import (
    ArtifactError,
    ArtifactWriter,
    assert_secret_free_bytes,
    canonical_json_bytes,
    source_fingerprint,
    verify_manifest,
)


class ArtifactWriterTests(unittest.TestCase):
    def test_writes_private_immutable_reconciled_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ArtifactWriter(Path(temporary), "run-001")
            writer.write_json("evidence/input.json", {"b": 2, "a": 1})
            writer.write_text("daily_board.md", "# UNPROVEN\n", "text/markdown")
            manifest = writer.finalize(
                as_of=date(2026, 8, 30),
                overall_status="UNPROVEN",
                created_at=datetime(2026, 8, 30, 18, tzinfo=timezone.utc),
                metadata={"order_submission_surface": False},
            )

            run_dir = Path(temporary) / "run-001"
            self.assertEqual(verify_manifest(run_dir), ())
            self.assertEqual([item.path for item in manifest.artifacts], [
                "daily_board.md",
                "evidence/input.json",
            ])
            self.assertEqual(os.stat(run_dir / "daily_board.md").st_mode & 0o777, 0o600)
            payload = json.loads((run_dir / "manifest.json").read_text())
            self.assertEqual(payload["schema"], "cultra.run-manifest.v1")
            self.assertEqual(payload["overall_status"], "UNPROVEN")
            self.assertRegex(payload["source_fingerprint"], r"^[0-9a-f]{64}$")

            with self.assertRaises(ArtifactError):
                writer.write_text("late.txt", "not allowed")

    def test_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ArtifactWriter(Path(temporary), "run-tamper")
            writer.write_text("board.md", "original")
            writer.finalize(as_of=date(2026, 8, 30), overall_status="UNPROVEN")
            (Path(temporary) / "run-tamper" / "board.md").write_text("changed")
            errors = verify_manifest(Path(temporary) / "run-tamper")
            self.assertTrue(any("mismatch" in item for item in errors))

    def test_detects_unlisted_or_manifest_removed_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ArtifactWriter(Path(temporary), "reconcile")
            writer.write_json("one.json", {"value": 1})
            writer.finalize(as_of=date(2026, 8, 30), overall_status="UNPROVEN")
            extra = writer.run_dir / "extra.json"
            extra.write_text("{}\n", encoding="utf-8")
            errors = verify_manifest(writer.run_dir)
            self.assertTrue(any("unlisted artifact" in error for error in errors))

            extra.unlink()
            manifest_path = writer.run_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifacts"] = []
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            errors = verify_manifest(writer.run_dir)
            self.assertTrue(any("unlisted artifact" in error for error in errors))

    def test_rejects_escape_overwrite_and_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = ArtifactWriter(Path(temporary), "safe-run")
            with self.assertRaises(ArtifactError):
                writer.write_text("../escape.txt", "x")
            writer.write_text("once.txt", "x")
            with self.assertRaises(ArtifactError):
                writer.write_text("once.txt", "y")
            with self.assertRaises(ArtifactError):
                writer.write_json("credential.json", {"token": "not-for-artifacts"})
            with self.assertRaises(ArtifactError):
                writer.write_text("credential.txt", "ORATS_TOKEN=not-for-artifacts")

    def test_canonical_json_is_stable_and_rejects_naive_time(self):
        left = canonical_json_bytes({"z": [2, 1], "a": {"c", "b"}})
        right = canonical_json_bytes({"a": {"b", "c"}, "z": [2, 1]})
        self.assertEqual(left, right)
        with self.assertRaises(ArtifactError):
            canonical_json_bytes({"timestamp": datetime(2026, 8, 30, 12)})

    def test_rejects_query_json_and_raw_byte_credentials_without_prose_false_positives(self):
        for value in (
            "https://api.orats.io/data?token=canary",
            "https://api.orats.io/data?apikey=canary",
            "authorization=canary",
            '{"access_token":"canary"}',
        ):
            with self.subTest(value=value):
                with self.assertRaises(ArtifactError):
                    assert_secret_free_bytes(value.encode("utf-8"))
        canonical_json_bytes({"note": "token budget and API key documentation"})
        canonical_json_bytes({"url": "https://example.invalid?token=[REDACTED]"})

        with tempfile.TemporaryDirectory() as temporary:
            writer = ArtifactWriter(Path(temporary), "raw-secret")
            with self.assertRaises(ArtifactError):
                writer.write_bytes(
                    "raw.bin",
                    b"prefix?token=canary",
                    "application/octet-stream",
                )
            writer.write_bytes(
                "safe.bin",
                b"binary token documentation without an assignment",
                "application/octet-stream",
            )

    def test_source_fingerprint_changes_with_runtime_source_or_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cultra").mkdir()
            (root / "configs").mkdir()
            source = root / "cultra" / "module.py"
            config = root / "configs" / "policy.json"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            config.write_text('{"version": 1}\n', encoding="utf-8")
            first = source_fingerprint(root)
            source.write_text("VALUE = 2\n", encoding="utf-8")
            second = source_fingerprint(root)
            self.assertRegex(first, r"^[0-9a-f]{64}$")
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
