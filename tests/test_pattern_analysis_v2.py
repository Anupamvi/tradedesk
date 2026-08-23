from pathlib import Path

import pytest

from uwos.pattern_analysis_v2 import PIPELINE_VERSION
from uwos.pattern_analysis_v2.core import parse_args
from uwos.pattern_analysis_v2 import core as pattern_v2
from uwos.options_pattern_pipeline_v1 import core as shared_engine


def test_pattern_analysis_v2_defaults_to_v2_output_namespace(tmp_path):
    args = parse_args(
        [
            "--base-dir",
            str(tmp_path),
            "--as-of",
            "2026-05-18",
        ]
    )

    assert PIPELINE_VERSION == "pattern_analysis_v2.12-family-sources-symmetric-momentum-20260803"
    assert Path(args.out_dir) == tmp_path / "out" / "pattern_analysis_v2" / "2026-05-18"


def test_pattern_analysis_v2_respects_explicit_out_dir(tmp_path):
    explicit = tmp_path / "custom"
    args = parse_args(
        [
            "--base-dir",
            str(tmp_path),
            "--as-of",
            "2026-05-18",
            "--out-dir",
            str(explicit),
        ]
    )

    assert Path(args.out_dir) == explicit


def test_pattern_analysis_v2_restores_shared_engine_version(monkeypatch):
    original = shared_engine.PIPELINE_VERSION
    observed = []

    def fake_run_pipeline(args):
        observed.append(shared_engine.PIPELINE_VERSION)
        return "done"

    monkeypatch.setattr(shared_engine, "run_pipeline", fake_run_pipeline)

    assert pattern_v2.run_pipeline(object()) == "done"
    assert observed == [pattern_v2.PIPELINE_VERSION]
    assert shared_engine.PIPELINE_VERSION == original


def test_pattern_analysis_v2_latest_fails_cleanly_without_source_complete_dates(tmp_path):
    with pytest.raises(ValueError, match="no source-complete UW dates"):
        parse_args(["--base-dir", str(tmp_path), "--as-of", "latest"])
