from pathlib import Path

import pytest

from uwos.pattern_analysis_v2 import PIPELINE_VERSION
from uwos.pattern_analysis_v2.core import parse_args
from uwos.pattern_analysis_v2 import core as pattern_v2


def test_pattern_analysis_v2_defaults_to_v2_output_namespace(tmp_path):
    args = parse_args(
        [
            "--base-dir",
            str(tmp_path),
            "--as-of",
            "2026-05-18",
        ]
    )

    assert PIPELINE_VERSION == "pattern_analysis_v2.20-price-first-managed-rebuild-20260823"
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


def test_pattern_analysis_v2_accepts_explicit_validated_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    args = parse_args(
        [
            "--base-dir",
            str(tmp_path),
            "--as-of",
            "2026-05-18",
            "--cache-dir",
            str(cache_dir),
        ]
    )

    assert Path(args.cache_dir) == cache_dir


def test_pattern_analysis_v2_runs_independent_engine(monkeypatch):
    observed = []

    def fake_run_pipeline(args):
        observed.append(args)
        return "done"

    monkeypatch.setattr(pattern_v2.engine, "run_pipeline", fake_run_pipeline)

    args = object()
    assert pattern_v2.run_pipeline(args) == "done"
    assert observed == [args]


def test_pattern_analysis_v2_latest_fails_cleanly_without_source_complete_dates(tmp_path):
    with pytest.raises(ValueError, match="no source-complete UW dates"):
        parse_args(["--base-dir", str(tmp_path), "--as-of", "latest"])
