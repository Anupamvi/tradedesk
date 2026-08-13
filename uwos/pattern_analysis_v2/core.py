"""Pattern Analysis V2 wrapper around the hardened options-pattern engine."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
from typing import Optional, Sequence

from uwos.options_pattern_pipeline_v1 import core as engine

PIPELINE_VERSION = "pattern_analysis_v2.12-family-sources-symmetric-momentum-20260803"
DEFAULT_OUTPUT_NAMESPACE = "pattern_analysis_v2"
V2_VALIDATION_TOP_CANDIDATES_PER_DAY = 500
V2_RISK_OVERRIDES = {
    "expected_hold_days": 40,
    "validation_horizon_sessions": 40,
    "long_option_profit_target_pct": 0.50,
    "long_option_stop_loss_pct": None,
    "use_shifted_chain_quotes": True,
    "bot_eod_quote_policy": "refresh_existing",
    "enforce_family_source_requirements": True,
    "enable_sector_momentum_families": True,
    "compact_snapshot_option_quotes": True,
    "min_oos_unique_signal_dates": 20,
    "require_every_validation_split_profitable": True,
    "require_day_clustered_pf_for_proven": True,
    "min_day_clustered_profit_factor_p05": 1.20,
    "require_matched_permutation_for_proven": True,
    "max_matched_null_p_value": 0.05,
    "matched_permutation_trials": 1000,
    "min_matched_null_coverage": 0.80,
}
CORE_SIGNAL_SOURCES = (
    "stock_screener",
    "hot_chains",
    "chain_oi",
)
ALL_PRIMARY_SOURCES = CORE_SIGNAL_SOURCES + (
    "dp_eod",
    "bot_eod",
)
ENGINE_SOURCE_COMPLETENESS_FOR_DATE = engine.source_completeness_for_date


def source_complete_dates(base_dir: Path):
    """Return dates with the core price, contract, and OI execution sources."""

    complete = []
    for signal_date in engine.list_date_dirs(base_dir):
        sources = engine.sources_for_date(base_dir / signal_date, signal_date)
        if all(sources.get(key) for key in CORE_SIGNAL_SOURCES):
            complete.append(signal_date)
    return complete


def source_completeness_for_date(base_dir: Path, signal_date: str):
    completeness = ENGINE_SOURCE_COMPLETENESS_FOR_DATE(base_dir, signal_date)
    sources = engine.sources_for_date(base_dir / signal_date, signal_date)
    missing = []
    labels = {
        "stock_screener": "stock-screener",
        "hot_chains": "hot-chains",
        "chain_oi": "chain-oi-changes",
        "dp_eod": "dp-eod-report",
        "bot_eod": "bot-eod-report",
    }
    for key in CORE_SIGNAL_SOURCES:
        if not sources.get(key):
            message = f"{labels[key]} core source required by Pattern Analysis V2 for {signal_date}"
            if message not in missing:
                missing.append(message)
    completeness["source_complete"] = not missing
    completeness["missing_sources"] = missing
    completeness["all_five_present"] = all(sources.get(key) for key in ALL_PRIMARY_SOURCES)
    completeness["source_policy"] = "core execution dates plus family-specific optional sources"
    return completeness


def parse_args(argv: Optional[Sequence[str]] = None):
    args = engine.parse_args(argv)
    if int(args.validation_top_candidates_per_day or 0) <= 0:
        args.validation_top_candidates_per_day = V2_VALIDATION_TOP_CANDIDATES_PER_DAY
    base_dir = Path(args.base_dir).expanduser().resolve()
    requested = str(args.as_of)
    if requested.lower() == "latest":
        available_dates = source_complete_dates(base_dir)
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
    previous_source_complete_dates = engine.source_complete_dates
    previous_source_completeness_for_date = engine.source_completeness_for_date
    previous_load_risk_config = engine.load_risk_config

    def load_v2_risk_config(config_arg, base_dir):
        path, config, _ = previous_load_risk_config(config_arg, base_dir)
        config = engine.deep_merge_dicts(config, V2_RISK_OVERRIDES)
        config_hash = hashlib.sha256(engine.stable_json(config).encode("utf-8")).hexdigest()
        return path, config, config_hash

    engine.PIPELINE_VERSION = PIPELINE_VERSION
    engine.source_complete_dates = source_complete_dates
    engine.source_completeness_for_date = source_completeness_for_date
    engine.load_risk_config = load_v2_risk_config
    try:
        result = engine.run_pipeline(args)
        if isinstance(result, dict) and result.get("out_dir"):
            from .external_context import build_external_context
            from .research_registry import build_pattern_registry
            from .source_audit import build_primary_source_audit

            out_dir = Path(str(result["out_dir"]))
            base_dir = Path(str(args.base_dir)).expanduser().resolve()
            as_of = str(result.get("as_of") or args.as_of)
            supplemental = {}
            supplemental.update(build_primary_source_audit(base_dir, out_dir, as_of))
            supplemental.update(build_pattern_registry(base_dir, out_dir, as_of))
            supplemental.update(build_external_context(base_dir, out_dir, as_of))
            result.setdefault("output_paths", {}).update(supplemental)
            register_supplemental_artifacts(result.get("output_paths", {}), supplemental)
        return result
    finally:
        # V2 currently shares the hardened engine module with V1. Avoid leaking
        # the V2 artifact version into a later V1 run in the same Python process.
        engine.PIPELINE_VERSION = previous_version
        engine.source_complete_dates = previous_source_complete_dates
        engine.source_completeness_for_date = previous_source_completeness_for_date
        engine.load_risk_config = previous_load_risk_config


def register_supplemental_artifacts(output_paths, supplemental):
    manifest_value = output_paths.get("artifact_manifest") if isinstance(output_paths, dict) else None
    if not manifest_value:
        return
    manifest_path = Path(str(manifest_value))
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("artifact_paths", {}).update(supplemental)
    records = {}
    for name, value in supplemental.items():
        path = Path(str(value))
        payload = path.read_bytes()
        records[name] = {
            "path": str(path),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    manifest["v2_supplemental_artifacts"] = records
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0
