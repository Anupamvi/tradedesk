"""Pattern Analysis V2 wrapper around the hardened options-pattern engine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from uwos.options_pattern_pipeline_v1 import core as engine

PIPELINE_VERSION = "pattern_analysis_v2.0"
DEFAULT_OUTPUT_NAMESPACE = "pattern_analysis_v2"


def parse_args(argv: Optional[Sequence[str]] = None):
    args = engine.parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve()
    requested = str(args.as_of)
    resolved_as_of = engine.source_complete_dates(base_dir)[-1] if requested.lower() == "latest" else engine.require_date(requested)
    if not args.out_dir:
        args.out_dir = str(base_dir / "out" / DEFAULT_OUTPUT_NAMESPACE / resolved_as_of)
    return args


def run_pipeline(args):
    engine.PIPELINE_VERSION = PIPELINE_VERSION
    return engine.run_pipeline(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0
