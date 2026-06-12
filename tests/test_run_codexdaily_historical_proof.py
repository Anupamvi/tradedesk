from __future__ import annotations

import json

from scripts import run_codexdaily_historical_proof as proof


def test_validate_manifest_requires_current_visibility_policy(tmp_path) -> None:
    manifest = tmp_path / "codexdaily_v3_manifest_2026-05-01.json"
    manifest.write_text(json.dumps({"visible_signal_policy": {"active_execute_cap": 1}}), encoding="utf-8")

    status, note = proof.validate_manifest(manifest, "v3")

    assert status == "FAIL"
    assert "active_execute_cap=1" == note


def test_validate_manifest_accepts_v3_uncapped_policy(tmp_path) -> None:
    manifest = tmp_path / "codexdaily_v3_manifest_2026-05-01.json"
    manifest.write_text(
        json.dumps(
            {
                "visible_signal_policy": {
                    "active_execute_cap": None,
                    "risk_caps_size_and_label_only": True,
                }
            }
        ),
        encoding="utf-8",
    )

    status, note = proof.validate_manifest(manifest, "v3")

    assert status == "PASS"
    assert "uncapped visibility" in note


def test_manifest_path_uses_pipeline_subdirectory(tmp_path) -> None:
    path = proof.manifest_path(tmp_path, "v4", "2026-05-01")

    assert path == tmp_path / "v4" / "codexdaily_v4_2026-05-01" / "codexdaily_v4_manifest_2026-05-01.json"


def test_default_proof_scope_is_uncapped() -> None:
    args = proof.parse_args([])

    status, note, config = proof.proof_scope(args)

    assert status == "FULL"
    assert "uncapped" in note
    assert config["bot_max_rows"] == 0
    assert config["max_tickers"] == 0
    assert config["max_candidates"] == 0


def test_proof_scope_marks_smoke_caps() -> None:
    args = proof.parse_args(["--bot-max-rows", "100", "--max-tickers", "8", "--max-candidates", "8"])

    status, note, _config = proof.proof_scope(args)

    assert status == "CAPPED"
    assert "bot_max_rows=100" in note
    assert "max_tickers=8" in note
    assert "max_candidates=8" in note
