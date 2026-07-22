"""Pattern Analysis V2 wrapper around the hardened options-pattern engine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from uwos.options_pattern_pipeline_v1 import core as engine

PIPELINE_VERSION = "pattern_analysis_v2.10-profile-aware-daily-selection-20260722"
DEFAULT_OUTPUT_NAMESPACE = "pattern_analysis_v2"


def parse_args(argv: Optional[Sequence[str]] = None):
    args = engine.parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve()
    requested = str(args.as_of)
    if requested.lower() == "latest":
        available_dates = engine.source_complete_dates(base_dir)
        if not available_dates:
            raise ValueError(f"no source-complete UW dates found under {base_dir}")
        resolved_as_of = available_dates[-1]
        args.as_of = resolved_as_of
    else:
        resolved_as_of = engine.require_date(requested)
    if not args.out_dir:
        args.out_dir = str(base_dir / "out" / DEFAULT_OUTPUT_NAMESPACE / resolved_as_of)
    return args


def run_pipeline(args):
    previous_version = engine.PIPELINE_VERSION
    engine.PIPELINE_VERSION = PIPELINE_VERSION
    try:
        return engine.run_pipeline(args)
    finally:
        # V2 currently shares the hardened engine module with V1. Avoid leaking
        # the V2 artifact version into a later V1 run in the same Python process.
        engine.PIPELINE_VERSION = previous_version


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0
