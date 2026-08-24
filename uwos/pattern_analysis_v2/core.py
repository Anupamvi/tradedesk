"""Public CLI contract for the independent Pattern Analysis V2 engine."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from . import PIPELINE_VERSION
from . import engine

DEFAULT_OUTPUT_NAMESPACE = "pattern_analysis_v2"


def source_complete_dates(base_dir: Path):
    """Return dates with price and option sources required by the V2 engine."""

    complete = []
    for signal_date in engine.date_dirs(base_dir):
        date_dir = base_dir / signal_date
        if all(
            engine.find_source(date_dir, prefix, signal_date)
            for prefix in ("stock-screener", "hot-chains", "chain-oi-changes")
        ):
            complete.append(signal_date)
    return complete


def parse_args(argv: Optional[Sequence[str]] = None):
    args = engine.parse_args(argv)
    base_dir = Path(args.base_dir).expanduser().resolve()
    requested = str(args.as_of)
    if requested.lower() == "latest":
        available_dates = source_complete_dates(base_dir)
        if not available_dates:
            raise ValueError(f"no source-complete UW dates found under {base_dir}")
        resolved_as_of = available_dates[-1]
        args.as_of = resolved_as_of
    else:
        engine.parse_date(requested)
        resolved_as_of = requested
    if not args.out_dir:
        args.out_dir = str(base_dir / "out" / DEFAULT_OUTPUT_NAMESPACE / resolved_as_of)
    return args


def run_pipeline(args):
    return engine.run_pipeline(args)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0
