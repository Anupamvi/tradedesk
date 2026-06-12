from pathlib import Path

from uwos.pattern_analysis_v2 import PIPELINE_VERSION
from uwos.pattern_analysis_v2.core import parse_args


def test_pattern_analysis_v2_defaults_to_v2_output_namespace(tmp_path):
    args = parse_args(
        [
            "--base-dir",
            str(tmp_path),
            "--as-of",
            "2026-05-18",
        ]
    )

    assert PIPELINE_VERSION == "pattern_analysis_v2.0"
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
