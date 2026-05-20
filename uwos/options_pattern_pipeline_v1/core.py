"""Production-oriented, leakage-aware options-pattern pipeline.

This module is intentionally separate from the older trend pipelines. It reads
dated source-like Unusual Whales exports directly, learns simple frozen pattern
definitions on chronological training windows, validates those definitions on
later windows, and emits daily trade-review artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from .macro_geo import (
    build_macro_geo_bundle,
    decompose_blockers,
    macro_geo_confirmation_fieldnames,
    macro_geo_promotion_fieldnames,
    macro_geo_ticker_map_fieldnames,
    render_missed_pattern_audit,
    render_pattern_observability_matrix,
)

PIPELINE_VERSION = "options_pattern_pipeline_v1.2"
ARTIFACT_SCHEMA_VERSION = "options_pattern_artifacts_v1"
DECISION_BOARD_SCHEMA_VERSION = "decision_board_v1"
MANIFEST_SCHEMA_VERSION = "artifact_manifest_v1"
DEFAULT_SEED = 20260510
HORIZONS = (1, 3, 5, 10, 20)
INDEX_TICKERS = {"SPY", "QQQ", "IWM"}
SOURCE_PREFIXES = (
    "stock-screener",
    "hot-chains",
    "chain-oi-changes",
    "option-trades",
    "bot-eod-report",
)
BOT_EOD_CACHE_SCHEMA_VERSION = 1
BOT_EOD_QUOTE_MIN_PREMIUM = 10_000.0
BOT_EOD_QUOTE_MIN_VOLUME = 100.0
GENERATED_OR_OLD_ARTIFACT_MARKERS = (
    "_trend_cache",
    "morning-watch-setups",
    "anu-expert-trade-table",
    "shortlist",
    "trend",
    "candidate",
    "rejection",
    "watchlist",
)
HARD_TRADE_BLOCKERS = {
    "NO_TRADEABLE_OPTION_QUOTE",
    "MISSING_BID_ASK_SPREAD",
    "BID_ASK_SPREAD_TOO_WIDE",
    "MISSING_ENTRY_CREDIT",
    "MISSING_ENTRY_ASK",
    "OPTION_LIQUIDITY_TOO_LOW",
    "DTE_TOO_SHORT_FOR_VALIDATION_HORIZONS",
}
REGIME_TRADE_BLOCKERS = {"MARKET_REGIME_CONFLICT"}
VALIDATION_TRADE_BLOCKERS = {
    "PATTERN_VALIDATION_NOT_PROVEN",
    "LIMITED_OUT_OF_SAMPLE_SAMPLE",
    "DOES_NOT_BEAT_TWO_BASELINES",
    "VALIDATION_EXPECTANCY_NEGATIVE",
}
EV_TRADE_BLOCKERS = {
    "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS",
    "EXPECTED_R_PER_DAY_NOT_POSITIVE",
    "PROFIT_FACTOR_BELOW_AUTO_APPROVAL",
    "CALIBRATION_SCORE_MISSING_OR_WEAK",
    "CONFIDENCE_BAND_TOO_WEAK",
}
RISK_TRADE_BLOCKERS = {
    "MAX_RISK_EXCEEDS_PER_TRADE_LIMIT",
    "CAPACITY_TOO_SMALL_FOR_INTENDED_SIZE",
    "EXCLUDED_TICKER_OR_INSTRUMENT",
    "MANUAL_ONLY_TICKER_OR_INSTRUMENT",
    "HIGH_NOTIONAL_INDEX_GUARDRAIL",
}
QUOTE_TRADE_BLOCKERS = {
    "LIVE_QUOTE_VALIDATION_SKIPPED",
    "LIVE_QUOTE_VALIDATION_FAILED",
    "QUOTE_COVERAGE_TOO_WEAK",
    "STALE_OR_HISTORICAL_QUOTE_ONLY",
}
KILL_SWITCH_PREFIX = "KILL_SWITCH_"
DEFAULT_RISK_CONFIG = {
    "account_risk_budget": 15000.0,
    "max_risk_per_trade": 1500.0,
    "max_daily_risk": 5000.0,
    "max_weekly_risk": 10000.0,
    "max_ticker_exposure": 3000.0,
    "max_sector_theme_exposure": 6000.0,
    "max_correlated_exposure": 7500.0,
    "max_buying_power_usage_pct": 0.35,
    "max_contracts_per_trade": 10,
    "min_expected_r": 0.0,
    "min_expected_r_per_day": 0.0,
    "min_profit_factor": 1.20,
    "min_oos_scored_outcomes": 30,
    "min_baselines_beaten": 2,
    "min_probability_score": 0.50,
    "min_calibrated_probability": 0.50,
    "min_confidence_lower_bound": 0.45,
    "max_bid_ask_spread_pct": 0.35,
    "max_credit_spread_spread_pct": 0.65,
    "min_option_volume": 50,
    "min_open_interest": 25,
    "min_quote_coverage": 0.25,
    "max_unscorable_rate": 0.75,
    "max_validation_drawdown_r": -12.0,
    "max_calibration_brier": 0.35,
    "min_edge_review_expected_r": 0.05,
    "min_edge_review_profit_factor": 1.30,
    "min_edge_review_scored_outcomes": 30,
    "min_edge_review_quote_coverage": 0.50,
    "min_edge_review_baselines_beaten": 2,
    "min_regime_edge_review_expected_r": 0.05,
    "min_regime_edge_review_profit_factor": 1.20,
    "min_regime_edge_review_scored_outcomes": 20,
    "contract_fee": 0.65,
    "round_trip_long_option_fees": 1.30,
    "round_trip_spread_fees": 2.60,
    "slippage_pct_of_spread": 0.50,
    "expected_hold_days": 5,
    "live_quote_required_for_auto": False,
    "allow_conservative_historical_quote_for_auto": True,
    "manual_only_tickers": ["SPX", "SPXW", "NDX", "RUT"],
    "manual_only_instruments": ["AM_SETTLED_INDEX_OPTIONS", "PM_SETTLED_INDEX_OPTIONS"],
    "excluded_tickers": [],
    "excluded_instruments": [],
    "high_notional_index_tickers": ["SPX", "SPXW", "NDX", "RUT"],
    "option_approval_constraints": {
        "defined_risk_only": True,
        "no_naked_short_options": True,
        "manual_review_for_index_options": True,
    },
    "kill_switches": {
        "source_incomplete": True,
        "low_quote_coverage": True,
        "high_unscorable_rate": True,
        "calibration_fails": True,
        "validation_drawdown_breached": True,
        "artifact_schema_validation_fails": True,
        "decision_board_missing_required_fields": True,
        "live_quote_validation_required": False,
        "shadow_pl_deteriorates": False,
        "volatility_shock": False,
        "max_risk_used": False,
        "too_many_correlated_candidates": False,
    },
    "artifact_retention": {
        "max_artifact_size_mb": 200,
        "preserve_rerun_suffixes": True,
        "canonical_overwrite_allowed": True,
    },
}


@dataclass(frozen=True)
class SourceRef:
    path: Path
    member: Optional[str] = None

    @property
    def label(self) -> str:
        if self.member:
            return f"{self.path}::{self.member}"
        return str(self.path)

    @property
    def name(self) -> str:
        return self.member or self.path.name

    @property
    def suffix(self) -> str:
        return Path(self.name).suffix.lower()


@dataclass
class Snapshot:
    signal_date: str
    source_files: List[str]
    skipped_sources: List[Dict[str, Any]]
    features: Dict[str, Dict[str, Any]]
    option_quotes: Dict[str, Dict[str, Any]]
    best_options: Dict[Tuple[str, str], Dict[str, Any]]
    market_regime: Dict[str, Any]
    counts: Dict[str, int]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python3 -m uwos.options_pattern_pipeline_v1",
        description=(
            "Build leakage-aware options-pattern reports from local dated "
            "Unusual Whales source exports."
        ),
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root containing YYYY-MM-DD UW folders. Default: current directory.",
    )
    parser.add_argument(
        "--as-of",
        default="latest",
        help="Signal date to run, or 'latest' for the latest source-complete UW date.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory. Default: "
            "<base-dir>/out/options_pattern_pipeline_v1/<as-of>."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Deterministic seed for random baselines.",
    )
    parser.add_argument(
        "--max-chain-rows-per-day",
        type=int,
        default=5000,
        help=(
            "Maximum chain-oi rows to stream per date. Files are ranked exports; "
            "the cap keeps full-history validation practical."
        ),
    )
    parser.add_argument(
        "--max-flow-file-mb",
        type=float,
        default=100.0,
        help="Skip non-bot option-trade/whale CSV feature reads above this size. Bot EOD is always used when present.",
    )
    parser.add_argument(
        "--bot-eod-cache-dir",
        default=None,
        help=(
            "Directory for derived bot-eod-report flow/quote caches. Default: "
            "<base-dir>/out/options_pattern_pipeline_v1/cache/bot_eod."
        ),
    )
    parser.add_argument(
        "--top-candidates-per-day",
        type=int,
        default=40,
        help="Maximum discovered pattern candidates retained per date.",
    )
    parser.add_argument(
        "--min-month-dates",
        type=int,
        default=5,
        help="Minimum source-complete dates required in a train/validation month.",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Generate the daily report without historical validation.",
    )
    parser.add_argument(
        "--risk-config",
        default=None,
        help=(
            "JSON risk/gating config. Default: "
            "<base-dir>/configs/options_pattern_pipeline_v1_hardening.json when present, "
            "otherwise built-in conservative defaults."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(args)
    return 0


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    started_perf = time.perf_counter()
    base_dir = Path(args.base_dir).expanduser().resolve()
    inventory_rows = inventory_source_data(base_dir)
    source_dates = source_complete_dates(base_dir)
    requested_latest = str(args.as_of).lower() == "latest"
    if not source_dates and requested_latest:
        raise SystemExit(f"No source-complete UW date folders found under {base_dir}")

    if requested_latest:
        as_of = source_dates[-1]
    else:
        as_of = require_date(args.as_of)
        if as_of not in source_dates:
            out_dir = (
                Path(args.out_dir).expanduser().resolve()
                if args.out_dir
                else base_dir / "out" / "options_pattern_pipeline_v1" / as_of
            )
            bot_eod_cache_dir = (
                Path(args.bot_eod_cache_dir).expanduser().resolve()
                if args.bot_eod_cache_dir
                else base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
            )
            config = base_run_config(args, base_dir, as_of, bot_eod_cache_dir)
            completeness = source_completeness_for_date(base_dir, as_of)
            macro_geo_bundle = build_macro_geo_bundle(
                base_dir=base_dir,
                as_of=as_of,
                snapshots={},
                source_dates=[d for d in source_dates if d <= as_of],
                daily_rows=[],
                source_complete=False,
                missing_sources=completeness["missing_sources"],
            )
            output_paths = write_source_incomplete_outputs(
                out_dir=out_dir,
                base_dir=base_dir,
                as_of=as_of,
                config=config,
                inventory_rows=inventory_rows,
                completeness=completeness,
                macro_geo_bundle=macro_geo_bundle,
                runtime_seconds=time.perf_counter() - started_perf,
            )
            return {
                "as_of": as_of,
                "out_dir": str(out_dir),
                "output_paths": output_paths,
                "verdict": "BLOCKED_SOURCE_INCOMPLETE",
            }

    usable_dates = [d for d in source_dates if d <= as_of]
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else base_dir / "out" / "options_pattern_pipeline_v1" / as_of
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    bot_eod_cache_dir = (
        Path(args.bot_eod_cache_dir).expanduser().resolve()
        if args.bot_eod_cache_dir
        else base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "bot_eod"
    )

    config = base_run_config(args, base_dir, as_of, bot_eod_cache_dir)

    snapshot_dates = [as_of] if args.no_validation else usable_dates
    snapshots: Dict[str, Snapshot] = {}
    for d in snapshot_dates:
        snapshots[d] = build_daily_snapshot(base_dir, d, config)

    validation_bundle: Dict[str, Any]
    if args.no_validation:
        validation_bundle = empty_validation_bundle()
    else:
        validation_bundle = run_historical_validation(
            snapshots=snapshots,
            source_dates=usable_dates,
            min_month_dates=args.min_month_dates,
            top_candidates_per_day=args.top_candidates_per_day,
            seed=args.seed,
        )

    prior_dates = [d for d in snapshot_dates if d < as_of]
    if prior_dates:
        daily_pattern_config = learn_pattern_config([snapshots[d] for d in prior_dates])
    else:
        daily_pattern_config = learn_pattern_config([snapshots[as_of]])

    daily_signals = generate_signals_for_snapshot(
        snapshots[as_of],
        daily_pattern_config,
        max_signals=args.top_candidates_per_day,
    )
    daily_rows = classify_daily_signals(
        daily_signals,
        validation_bundle["family_tiers"],
        snapshots[as_of],
    )

    missed_rows = [] if args.no_validation else run_missed_mover_audit(snapshots, usable_dates, as_of)
    sentiment_summary = build_sentiment_news_summary(base_dir / as_of, as_of)
    completeness = source_completeness_for_date(base_dir, as_of)
    macro_geo_bundle = build_macro_geo_bundle(
        base_dir=base_dir,
        as_of=as_of,
        snapshots=snapshots,
        source_dates=usable_dates,
        daily_rows=daily_rows,
        source_complete=bool(completeness["source_complete"]),
        missing_sources=completeness["missing_sources"],
    )

    output_paths = write_outputs(
        out_dir=out_dir,
        base_dir=base_dir,
        as_of=as_of,
        config=config,
        inventory_rows=inventory_rows,
        snapshots=snapshots,
        validation_bundle=validation_bundle,
        daily_pattern_config=daily_pattern_config,
        daily_rows=daily_rows,
        missed_rows=missed_rows,
        sentiment_summary=sentiment_summary,
        source_completeness=completeness,
        macro_geo_bundle=macro_geo_bundle,
        runtime_seconds=time.perf_counter() - started_perf,
    )
    return {
        "as_of": as_of,
        "out_dir": str(out_dir),
        "output_paths": output_paths,
        "verdict": final_verdict(validation_bundle, daily_rows),
    }


def base_run_config(
    args: argparse.Namespace,
    base_dir: Path,
    as_of: str,
    bot_eod_cache_dir: Path,
) -> Dict[str, Any]:
    risk_config_path, risk_config, risk_config_hash = load_risk_config(args.risk_config, base_dir)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "base_dir": str(base_dir),
        "as_of": as_of,
        "seed": args.seed,
        "max_chain_rows_per_day": args.max_chain_rows_per_day,
        "max_flow_file_mb": args.max_flow_file_mb,
        "bot_eod_cache_dir": str(bot_eod_cache_dir),
        "top_candidates_per_day": args.top_candidates_per_day,
        "horizons": list(HORIZONS),
        "risk_config_path": str(risk_config_path) if risk_config_path else "",
        "risk_config_hash": risk_config_hash,
        "risk_config": risk_config,
        "input_policy": (
            "Reads dated UW source-like exports only. Ignores prior trend pipeline "
            "scores, gates, candidates, rejection labels, generated reports, and "
            "morning watchlists as feature inputs."
        ),
    }


def load_risk_config(config_arg: Optional[str], base_dir: Path) -> Tuple[Optional[Path], Dict[str, Any], str]:
    default_path = base_dir / "configs" / "options_pattern_pipeline_v1_hardening.json"
    path = Path(config_arg).expanduser().resolve() if config_arg else default_path
    config = dict(DEFAULT_RISK_CONFIG)
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Could not read risk config {path}: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SystemExit(f"Risk config {path} must be a JSON object")
        config = deep_merge_dicts(config, loaded)
        used_path: Optional[Path] = path
    elif config_arg:
        raise SystemExit(f"Risk config does not exist: {path}")
    else:
        used_path = None
    config_hash = hashlib.sha256(stable_json(config).encode("utf-8")).hexdigest()
    return used_path, config, config_hash


def deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge_dicts(out[key], value)
        else:
            out[key] = value
    return out


def source_completeness_for_date(base_dir: Path, signal_date: str) -> Dict[str, Any]:
    date_dir = base_dir / signal_date
    missing: List[str] = []
    present: Dict[str, List[str]] = {
        "stock_screener": [],
        "hot_chains": [],
        "chain_oi": [],
        "bot_eod": [],
        "option_trades": [],
        "whale_filtered": [],
    }
    if not date_dir.exists():
        missing.append(f"date folder: {date_dir}")
        missing.extend(
            [
                f"stock-screener source: expected stock-screener-{signal_date}.csv or .zip under {date_dir}",
                f"hot-chains source: expected hot-chains-{signal_date}.csv or .zip under {date_dir}",
                f"chain-oi-changes source: expected chain-oi-changes-{signal_date}.csv or .zip under {date_dir}",
                f"options flow source: expected bot-eod-report-{signal_date}.csv/.zip, option-trades-{signal_date}.csv/.zip, or whale_trades_filtered.csv under {date_dir}",
            ]
        )
        return {"source_complete": False, "missing_sources": missing, "present_sources": present}

    sources = sources_for_date(date_dir, signal_date)
    for key in present:
        present[key] = [ref.label for ref in sources.get(key, [])]
    required = {
        "stock_screener": f"stock-screener source: expected stock-screener-{signal_date}.csv or .zip under {date_dir}",
        "hot_chains": f"hot-chains source: expected hot-chains-{signal_date}.csv or .zip under {date_dir}",
        "chain_oi": f"chain-oi-changes source: expected chain-oi-changes-{signal_date}.csv or .zip under {date_dir}",
    }
    for key, message in required.items():
        if not sources.get(key):
            missing.append(message)
    if not (sources.get("bot_eod") or sources.get("option_trades") or sources.get("whale_filtered")):
        missing.append(
            f"options flow source: expected bot-eod-report-{signal_date}.csv/.zip, "
            f"option-trades-{signal_date}.csv/.zip, or whale_trades_filtered.csv under {date_dir}"
        )
    return {"source_complete": not missing, "missing_sources": missing, "present_sources": present}


def require_date(value: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(value)):
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM-DD, got {value!r}")
    return str(value)


def source_complete_dates(base_dir: Path) -> List[str]:
    out: List[str] = []
    for d in list_date_dirs(base_dir):
        sources = sources_for_date(base_dir / d, d)
        if sources.get("stock_screener") and sources.get("hot_chains") and sources.get("chain_oi"):
            out.append(d)
    return sorted(out)


def list_date_dirs(base_dir: Path) -> List[str]:
    if not base_dir.exists():
        return []
    return sorted(
        p.name
        for p in base_dir.iterdir()
        if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name)
    )


def inventory_source_data(base_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for d in list_date_dirs(base_dir):
        date_dir = base_dir / d
        for path in sorted(date_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(base_dir)
            lowered = str(rel).lower()
            ignored = any(marker in lowered for marker in GENERATED_OR_OLD_ARTIFACT_MARKERS)
            source_like = is_source_like_path(path)
            rows.append(
                {
                    "date": d,
                    "relative_path": str(rel),
                    "absolute_path": str(path.resolve()),
                    "bytes": path.stat().st_size,
                    "source_like": source_like,
                    "used_by_features": source_like and not ignored,
                    "ignored_reason": "old_or_generated_artifact" if ignored else "",
                }
            )
    return rows


def is_source_like_path(path: Path) -> bool:
    lowered = str(path).lower()
    if any(marker in lowered for marker in GENERATED_OR_OLD_ARTIFACT_MARKERS):
        return False
    name = path.name.lower()
    if path.suffix.lower() == ".zip" and any(name.startswith(p) for p in SOURCE_PREFIXES):
        return True
    if path.suffix.lower() == ".csv" and any(name.startswith(p) for p in SOURCE_PREFIXES):
        return True
    if name == "whale_trades_filtered.csv":
        return True
    if "enrichments/uw/uw_gex_raw" in lowered and path.suffix.lower() == ".json":
        return True
    if "/browser_text/" in lowered and path.suffix.lower() == ".txt":
        return True
    return False


def sources_for_date(date_dir: Path, signal_date: str) -> Dict[str, List[SourceRef]]:
    return {
        "stock_screener": find_csv_sources(date_dir, "stock-screener", signal_date, exact=True),
        "hot_chains": find_csv_sources(date_dir, "hot-chains", signal_date, exact=True),
        "chain_oi": find_csv_sources(date_dir, "chain-oi-changes", signal_date, exact=True),
        "bot_eod": find_csv_sources(date_dir, "bot-eod-report", signal_date, exact=True),
        "option_trades": find_csv_sources(date_dir, "option-trades", signal_date, exact=True),
        "whale_filtered": [SourceRef(date_dir / "whale_trades_filtered.csv")]
        if (date_dir / "whale_trades_filtered.csv").exists()
        else [],
    }


def find_csv_sources(date_dir: Path, prefix: str, signal_date: str, exact: bool) -> List[SourceRef]:
    sources: List[SourceRef] = []
    unzipped = date_dir / "_unzipped_mode_a"
    if unzipped.exists():
        for p in sorted(unzipped.glob(f"{prefix}*.csv")):
            if not exact or extract_date_from_name(p.name) == signal_date:
                sources.append(SourceRef(p))
    for p in sorted(date_dir.glob(f"{prefix}*.csv")):
        if not exact or extract_date_from_name(p.name) == signal_date:
            sources.append(SourceRef(p))
    if sources:
        return dedupe_sources(sources)
    for zpath in sorted(date_dir.glob(f"{prefix}*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                for member in sorted(zf.namelist()):
                    if not member.lower().endswith(".csv"):
                        continue
                    if not exact or extract_date_from_name(Path(member).name) == signal_date:
                        sources.append(SourceRef(zpath, member))
        except zipfile.BadZipFile:
            continue
    return dedupe_sources(sources)


def dedupe_sources(sources: Sequence[SourceRef]) -> List[SourceRef]:
    seen: set[str] = set()
    out: List[SourceRef] = []
    for ref in sources:
        key = ref.label
        if key not in seen:
            seen.add(key)
            out.append(ref)
    return out


def extract_date_from_name(name: str) -> Optional[str]:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", name)
    return m.group(1) if m else None


def build_daily_snapshot(base_dir: Path, signal_date: str, config: Mapping[str, Any]) -> Snapshot:
    date_dir = base_dir / signal_date
    sources = sources_for_date(date_dir, signal_date)
    source_files: List[str] = []
    skipped_sources: List[Dict[str, Any]] = []
    counts: Counter[str] = Counter()
    features: Dict[str, Dict[str, Any]] = {}
    option_quotes: Dict[str, Dict[str, Any]] = {}
    best_options: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def feature_for(ticker: str) -> Dict[str, Any]:
        ticker = clean_ticker(ticker)
        if ticker not in features:
            features[ticker] = new_feature(signal_date, ticker)
        return features[ticker]

    for ref in sources["stock_screener"]:
        source_files.append(ref.label)
        for row in iter_csv_dicts(ref):
            ticker = clean_ticker(row.get("ticker", ""))
            if not ticker:
                continue
            f = feature_for(ticker)
            counts["stock_screener_rows"] += 1
            f["source_flags"].add("stock_screener")
            f["sector"] = row.get("sector") or f.get("sector") or ""
            f["issue_type"] = row.get("issue_type") or f.get("issue_type") or ""
            f["next_earnings_date"] = row.get("next_earnings_date") or f.get("next_earnings_date") or ""
            f["er_time"] = row.get("er_time") or f.get("er_time") or ""
            for src, dst in (
                ("close", "close"),
                ("prev_close", "prev_close"),
                ("high", "high"),
                ("low", "low"),
                ("total_volume", "stock_volume"),
                ("avg30_volume", "avg30_stock_volume"),
                ("call_volume", "call_volume"),
                ("put_volume", "put_volume"),
                ("call_premium", "call_premium"),
                ("put_premium", "put_premium"),
                ("bullish_premium", "bullish_premium"),
                ("bearish_premium", "bearish_premium"),
                ("net_call_premium", "net_call_premium"),
                ("net_put_premium", "net_put_premium"),
                ("avg_30_day_call_volume", "avg30_call_volume"),
                ("avg_30_day_put_volume", "avg30_put_volume"),
                ("call_volume_ask_side", "call_volume_ask_side"),
                ("call_volume_bid_side", "call_volume_bid_side"),
                ("put_volume_ask_side", "put_volume_ask_side"),
                ("put_volume_bid_side", "put_volume_bid_side"),
                ("total_open_interest", "total_open_interest"),
            ):
                val = num(row.get(src))
                if val is not None:
                    f[dst] = val

    for ref in sources["hot_chains"]:
        source_files.append(ref.label)
        for row in iter_csv_dicts(ref):
            if row.get("date") and row.get("date") != signal_date:
                continue
            parsed = parse_option_symbol(row.get("option_symbol", ""))
            if not parsed:
                continue
            ticker = parsed["ticker"]
            f = feature_for(ticker)
            counts["hot_chain_rows"] += 1
            f["source_flags"].add("hot_chains")
            if row.get("sector") and not f.get("sector"):
                f["sector"] = row.get("sector") or ""
            if row.get("next_earnings_date") and not f.get("next_earnings_date"):
                f["next_earnings_date"] = row.get("next_earnings_date") or ""
            if row.get("er_time") and not f.get("er_time"):
                f["er_time"] = row.get("er_time") or ""
            q = option_quote_from_hot_row(row, parsed, signal_date)
            option_quotes[q["option_symbol"]] = q
            update_hot_aggregate(f, q, row)
            maybe_set_best_option(best_options, q)

    for ref in sources["chain_oi"]:
        source_files.append(ref.label)
        for idx, row in enumerate(iter_csv_dicts(ref)):
            if idx >= int(config["max_chain_rows_per_day"]):
                skipped_sources.append(
                    {
                        "source": ref.label,
                        "reason": "max_chain_rows_per_day",
                        "limit": int(config["max_chain_rows_per_day"]),
                    }
                )
                break
            if row.get("curr_date") and row.get("curr_date") != signal_date:
                continue
            parsed = parse_option_symbol(row.get("option_symbol", ""))
            if not parsed:
                continue
            ticker = parsed["ticker"]
            f = feature_for(ticker)
            counts["chain_oi_rows"] += 1
            f["source_flags"].add("chain_oi")
            update_chain_oi_aggregate(f, row, parsed)

    bot_refs = list(sources["bot_eod"])
    fallback_refs = list(sources["option_trades"]) + list(sources["whale_filtered"])
    if bot_refs and fallback_refs:
        for ref in fallback_refs:
            skipped_sources.append(
                {
                    "source": ref.label,
                    "reason": "bot_eod_present_primary_flow_source",
                }
            )
    if bot_refs:
        for ref in bot_refs:
            source_files.append(ref.label)
        bot_cache = load_or_build_bot_eod_cache(bot_refs, signal_date, config)
        source_files.extend(bot_cache.get("cache_files", []))
        counts["bot_eod_cache_hit"] += int(bool(bot_cache.get("cache_hit")))
        counts["bot_eod_cache_built"] += int(bool(bot_cache.get("cache_built")))
        for row in bot_cache["flow_rows"]:
            ticker = clean_ticker(row.get("ticker", ""))
            if not ticker:
                continue
            f = feature_for(ticker)
            f["source_flags"].add("bot_eod")
            apply_bot_flow_aggregate(f, row)
            counts["bot_eod_rows"] += int(num(row.get("row_count")) or 0)
            counts["bot_eod_ticker_rows"] += 1
        for quote in bot_cache["quote_rows"]:
            symbol = str(quote.get("option_symbol") or "")
            if not symbol:
                continue
            option_quotes[symbol] = quote
            maybe_set_best_option(best_options, quote)
            counts["bot_eod_quote_rows"] += 1
    else:
        for ref in fallback_refs:
            size_mb = source_size_mb(ref)
            if size_mb > float(config["max_flow_file_mb"]):
                skipped_sources.append(
                    {
                        "source": ref.label,
                        "reason": "flow_file_too_large",
                        "size_mb": round(size_mb, 2),
                        "limit_mb": float(config["max_flow_file_mb"]),
                    }
                )
                continue
            source_files.append(ref.label)
            for row in iter_csv_dicts(ref):
                ticker = clean_ticker(row.get("underlying_symbol", ""))
                if not ticker:
                    continue
                f = feature_for(ticker)
                counts["option_trade_rows"] += 1
                f["source_flags"].add("option_trades")
                update_option_trade_aggregate(f, row)

    add_best_vertical_spreads(best_options, option_quotes)

    apply_gex_context(date_dir, signal_date, features, source_files, skipped_sources, counts)

    for f in features.values():
        finalize_feature(f)

    market_regime = compute_market_regime(signal_date, features)
    return Snapshot(
        signal_date=signal_date,
        source_files=source_files,
        skipped_sources=skipped_sources,
        features=features,
        option_quotes=option_quotes,
        best_options=best_options,
        market_regime=market_regime,
        counts=dict(counts),
    )


def source_size_mb(ref: SourceRef) -> float:
    if ref.member:
        try:
            with zipfile.ZipFile(ref.path) as zf:
                info = zf.getinfo(ref.member)
                return info.file_size / 1_000_000.0
        except Exception:
            return 0.0
    return ref.path.stat().st_size / 1_000_000.0


def is_bot_eod_ref(ref: SourceRef) -> bool:
    return "bot-eod-report" in ref.name.lower() or "bot-eod-report" in ref.path.name.lower()


def load_or_build_bot_eod_cache(
    refs: Sequence[SourceRef],
    signal_date: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    paths = bot_eod_cache_paths(signal_date, config)
    fingerprints = [source_fingerprint(ref) for ref in refs]
    if paths["meta"].exists() and paths["flow"].exists() and paths["quotes"].exists():
        try:
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if (
            meta.get("schema_version") == BOT_EOD_CACHE_SCHEMA_VERSION
            and meta.get("signal_date") == signal_date
            and meta.get("source_fingerprints") == fingerprints
        ):
            return {
                "flow_rows": [coerce_bot_flow_cache_row(r) for r in read_csv_rows(paths["flow"])],
                "quote_rows": [coerce_bot_quote_cache_row(r) for r in read_csv_rows(paths["quotes"])],
                "cache_files": [str(paths["flow"].resolve()), str(paths["quotes"].resolve()), str(paths["meta"].resolve())],
                "cache_hit": True,
                "cache_built": False,
            }

    flow_rows, quote_rows, raw_row_count = build_bot_eod_cache_rows(refs, signal_date)
    paths["flow"].parent.mkdir(parents=True, exist_ok=True)
    write_csv(paths["flow"], flow_rows, bot_flow_cache_fieldnames())
    write_csv(paths["quotes"], quote_rows, bot_quote_cache_fieldnames())
    write_json(
        paths["meta"],
        {
            "schema_version": BOT_EOD_CACHE_SCHEMA_VERSION,
            "signal_date": signal_date,
            "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "source_fingerprints": fingerprints,
            "raw_row_count": raw_row_count,
            "flow_ticker_count": len(flow_rows),
            "quote_row_count": len(quote_rows),
            "quote_materiality": {
                "min_premium": BOT_EOD_QUOTE_MIN_PREMIUM,
                "min_volume": BOT_EOD_QUOTE_MIN_VOLUME,
            },
        },
    )
    return {
        "flow_rows": flow_rows,
        "quote_rows": quote_rows,
        "cache_files": [str(paths["flow"].resolve()), str(paths["quotes"].resolve()), str(paths["meta"].resolve())],
        "cache_hit": False,
        "cache_built": True,
    }


def bot_eod_cache_paths(signal_date: str, config: Mapping[str, Any]) -> Dict[str, Path]:
    base = Path(str(config.get("bot_eod_cache_dir") or ".")).expanduser().resolve()
    return {
        "flow": base / f"bot_eod_flow_by_ticker_{signal_date}.csv",
        "quotes": base / f"bot_eod_quotes_{signal_date}.csv",
        "meta": base / f"bot_eod_cache_{signal_date}.json",
    }


def source_fingerprint(ref: SourceRef) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "label": ref.label,
        "path": str(ref.path.resolve()),
        "member": ref.member or "",
        "path_bytes": ref.path.stat().st_size if ref.path.exists() else None,
        "path_mtime_ns": ref.path.stat().st_mtime_ns if ref.path.exists() else None,
    }
    if ref.member:
        try:
            with zipfile.ZipFile(ref.path) as zf:
                info = zf.getinfo(ref.member)
            row.update({"member_bytes": info.file_size, "member_crc": info.CRC})
        except Exception:
            row.update({"member_bytes": None, "member_crc": None})
    return row


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def build_bot_eod_cache_rows(
    refs: Sequence[SourceRef],
    signal_date: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    flow_aggs: Dict[str, Dict[str, Any]] = {}
    quote_rows: Dict[str, Dict[str, Any]] = {}
    raw_row_count = 0
    for ref in refs:
        for row in iter_csv_dicts(ref):
            raw_row_count += 1
            ticker = clean_ticker(row.get("underlying_symbol", ""))
            if not ticker:
                continue
            agg = flow_aggs.setdefault(ticker, new_bot_flow_agg(signal_date, ticker))
            update_bot_flow_cache_agg(agg, row)
            quote = option_quote_from_bot_trade_row(row, signal_date)
            if quote:
                quote_rows[quote["option_symbol"]] = merge_bot_quote(quote_rows.get(quote["option_symbol"]), quote)
    return (
        sorted(flow_aggs.values(), key=lambda r: (-float(r.get("flow_total_premium") or 0.0), str(r.get("ticker")))),
        sorted(quote_rows.values(), key=lambda r: (-float(r.get("premium") or 0.0), str(r.get("option_symbol")))),
        raw_row_count,
    )


def new_bot_flow_agg(signal_date: str, ticker: str) -> Dict[str, Any]:
    return {
        "date": signal_date,
        "ticker": ticker,
        "row_count": 0,
        "sector": "",
        "underlying_price_last": None,
        "flow_call_ask_premium": 0.0,
        "flow_put_ask_premium": 0.0,
        "flow_call_bid_premium": 0.0,
        "flow_put_bid_premium": 0.0,
        "flow_total_premium": 0.0,
        "flow_call_trade_count": 0,
        "flow_put_trade_count": 0,
    }


def update_bot_flow_cache_agg(agg: Dict[str, Any], row: Mapping[str, str]) -> None:
    premium = option_trade_premium(row)
    option_type = str(row.get("option_type") or "").lower()
    side = str(row.get("side") or "").lower()
    agg["row_count"] += 1
    agg["flow_total_premium"] += premium
    if row.get("sector") and not agg.get("sector"):
        agg["sector"] = row.get("sector") or ""
    underlying = num(row.get("underlying_price"))
    if underlying and underlying > 0:
        agg["underlying_price_last"] = underlying
    if option_type == "call":
        agg["flow_call_trade_count"] += 1
        if side == "ask":
            agg["flow_call_ask_premium"] += premium
        elif side == "bid":
            agg["flow_call_bid_premium"] += premium
    elif option_type == "put":
        agg["flow_put_trade_count"] += 1
        if side == "ask":
            agg["flow_put_ask_premium"] += premium
        elif side == "bid":
            agg["flow_put_bid_premium"] += premium


def option_trade_premium(row: Mapping[str, Any]) -> float:
    premium = abs(num(row.get("premium")) or 0.0)
    if premium <= 0:
        price = num(row.get("price")) or 0.0
        size = num(row.get("size")) or 0.0
        premium = abs(price * size * 100.0)
    return premium


def apply_bot_flow_aggregate(f: Dict[str, Any], row: Mapping[str, Any]) -> None:
    if row.get("sector") and not f.get("sector"):
        f["sector"] = str(row.get("sector") or "")
    underlying = num(row.get("underlying_price_last"))
    if underlying and not f.get("close"):
        f["close"] = underlying
    for key in (
        "flow_call_ask_premium",
        "flow_put_ask_premium",
        "flow_call_bid_premium",
        "flow_put_bid_premium",
        "flow_total_premium",
        "flow_call_trade_count",
        "flow_put_trade_count",
    ):
        f[key] = (f.get(key) or 0.0) + (num(row.get(key)) or 0.0)


def option_quote_from_bot_trade_row(row: Mapping[str, str], signal_date: str) -> Optional[Dict[str, Any]]:
    parsed = parse_option_symbol(row.get("option_chain_id", ""))
    if not parsed:
        return None
    bid = first_positive([row.get("nbbo_bid"), row.get("ewma_nbbo_bid")]) or 0.0
    ask = first_positive([row.get("nbbo_ask"), row.get("ewma_nbbo_ask")]) or 0.0
    price = num(row.get("price"))
    size = num(row.get("size")) or 0.0
    premium = option_trade_premium(row)
    volume = num(row.get("volume")) or size
    open_interest = num(row.get("open_interest")) or 0.0
    if premium < BOT_EOD_QUOTE_MIN_PREMIUM and volume < BOT_EOD_QUOTE_MIN_VOLUME:
        return None
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else price
    reference = mid or price or ask or bid
    spread = ask - bid if ask > 0 and bid >= 0 else None
    spread_pct = spread / reference if spread is not None and reference and reference > 0 else None
    return {
        "date": signal_date,
        "ticker": parsed["ticker"],
        "option_symbol": parsed["option_symbol"],
        "option_type": parsed["option_type"],
        "direction": "bullish" if parsed["option_type"] == "call" else "bearish",
        "expiry": parsed["expiry"],
        "strike": parsed["strike"],
        "dte": trading_day_delta(signal_date, parsed["expiry"]),
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "high": None,
        "low": None,
        "option_close": price,
        "avg_price": price,
        "stock_close": num(row.get("underlying_price")),
        "volume": volume,
        "open_interest": open_interest,
        "premium": premium,
        "iv": num(row.get("implied_volatility")),
        "spread": spread,
        "spread_pct": spread_pct,
        "ask_side_volume": size if str(row.get("side") or "").lower() == "ask" else 0.0,
        "bid_side_volume": size if str(row.get("side") or "").lower() == "bid" else 0.0,
        "sweep_volume": 0.0,
        "multileg_volume": 0.0,
        "quote_source": "bot_eod",
    }


def merge_bot_quote(existing: Optional[Dict[str, Any]], quote: Dict[str, Any]) -> Dict[str, Any]:
    if not existing:
        return dict(quote)
    merged = dict(existing)
    merged["premium"] = (num(existing.get("premium")) or 0.0) + (num(quote.get("premium")) or 0.0)
    merged["volume"] = max(num(existing.get("volume")) or 0.0, num(quote.get("volume")) or 0.0)
    merged["open_interest"] = max(num(existing.get("open_interest")) or 0.0, num(quote.get("open_interest")) or 0.0)
    merged["ask_side_volume"] = (num(existing.get("ask_side_volume")) or 0.0) + (num(quote.get("ask_side_volume")) or 0.0)
    merged["bid_side_volume"] = (num(existing.get("bid_side_volume")) or 0.0) + (num(quote.get("bid_side_volume")) or 0.0)
    if (num(quote.get("bid")) or 0.0) > 0 or (num(quote.get("ask")) or 0.0) > 0:
        for key in ("bid", "ask", "mid", "option_close", "avg_price", "stock_close", "iv", "spread", "spread_pct"):
            if quote.get(key) not in (None, ""):
                merged[key] = quote.get(key)
    return merged


def coerce_bot_flow_cache_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key in bot_flow_cache_fieldnames():
        if key not in {"date", "ticker", "sector"}:
            out[key] = num(out.get(key)) or 0.0
    return out


def coerce_bot_quote_cache_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    for key in bot_quote_cache_fieldnames():
        if key not in {"date", "ticker", "option_symbol", "option_type", "direction", "expiry", "quote_source"}:
            out[key] = num(out.get(key))
            if out[key] is None and key in {
                "bid",
                "ask",
                "volume",
                "open_interest",
                "premium",
                "ask_side_volume",
                "bid_side_volume",
                "sweep_volume",
                "multileg_volume",
            }:
                out[key] = 0.0
    return out


def bot_flow_cache_fieldnames() -> List[str]:
    return [
        "date",
        "ticker",
        "row_count",
        "sector",
        "underlying_price_last",
        "flow_call_ask_premium",
        "flow_put_ask_premium",
        "flow_call_bid_premium",
        "flow_put_bid_premium",
        "flow_total_premium",
        "flow_call_trade_count",
        "flow_put_trade_count",
    ]


def bot_quote_cache_fieldnames() -> List[str]:
    return [
        "date",
        "ticker",
        "option_symbol",
        "option_type",
        "direction",
        "expiry",
        "strike",
        "dte",
        "bid",
        "ask",
        "mid",
        "high",
        "low",
        "option_close",
        "avg_price",
        "stock_close",
        "volume",
        "open_interest",
        "premium",
        "iv",
        "spread",
        "spread_pct",
        "ask_side_volume",
        "bid_side_volume",
        "sweep_volume",
        "multileg_volume",
        "quote_source",
    ]


@contextmanager
def open_text_source(ref: SourceRef) -> Iterator[Any]:
    if ref.member:
        with zipfile.ZipFile(ref.path) as zf:
            with zf.open(ref.member) as fh:
                import io

                yield io.TextIOWrapper(fh, encoding="utf-8-sig", errors="replace", newline="")
    else:
        with ref.path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
            yield fh


def iter_csv_dicts(ref: SourceRef) -> Iterator[Dict[str, str]]:
    with open_text_source(ref) as fh:
        reader = csv.reader(fh)
        try:
            raw_header = next(reader)
        except StopIteration:
            return
        header = normalize_header(raw_header, ref.name)
        width = len(header)
        for values in reader:
            if len(values) < width:
                values = values + [""] * (width - len(values))
            elif len(values) > width:
                values = values[:width]
            yield dict(zip(header, values))


def normalize_header(raw_header: Sequence[str], source_name: str) -> List[str]:
    out: List[str] = []
    counts: Counter[str] = Counter()
    is_hot = "hot-chains" in source_name.lower()
    for idx, raw in enumerate(raw_header):
        name = (raw or f"col_{idx}").strip()
        if is_hot and name == "close":
            if idx == 12:
                name = "option_close"
            elif idx == 29:
                name = "underlying_close"
        counts[name] += 1
        if counts[name] > 1:
            name = f"{name}_{counts[name]}"
        out.append(name)
    return out


def new_feature(signal_date: str, ticker: str) -> Dict[str, Any]:
    return {
        "date": signal_date,
        "ticker": ticker,
        "source_flags": set(),
        "sector": "",
        "issue_type": "",
        "next_earnings_date": "",
        "er_time": "",
        "gex_available": 0,
        "gex_capture_ok_point_in_time": 0,
        "close": None,
        "prev_close": None,
        "high": None,
        "low": None,
        "stock_volume": 0.0,
        "avg30_stock_volume": 0.0,
        "call_volume": 0.0,
        "put_volume": 0.0,
        "call_premium": 0.0,
        "put_premium": 0.0,
        "bullish_premium": 0.0,
        "bearish_premium": 0.0,
        "net_call_premium": 0.0,
        "net_put_premium": 0.0,
        "avg30_call_volume": 0.0,
        "avg30_put_volume": 0.0,
        "call_volume_ask_side": 0.0,
        "call_volume_bid_side": 0.0,
        "put_volume_ask_side": 0.0,
        "put_volume_bid_side": 0.0,
        "total_open_interest": 0.0,
        "hot_chain_count": 0,
        "hot_total_volume": 0.0,
        "hot_total_premium": 0.0,
        "hot_call_volume": 0.0,
        "hot_put_volume": 0.0,
        "hot_call_premium": 0.0,
        "hot_put_premium": 0.0,
        "hot_call_ask_volume": 0.0,
        "hot_call_bid_volume": 0.0,
        "hot_put_ask_volume": 0.0,
        "hot_put_bid_volume": 0.0,
        "hot_sweep_volume": 0.0,
        "hot_multileg_volume": 0.0,
        "hot_iv_weighted_sum": 0.0,
        "hot_iv_weight": 0.0,
        "hot_spread_weighted_sum": 0.0,
        "hot_spread_weight": 0.0,
        "min_spread_pct": None,
        "oi_call_diff": 0.0,
        "oi_put_diff": 0.0,
        "oi_total_diff": 0.0,
        "oi_call_volume": 0.0,
        "oi_put_volume": 0.0,
        "oi_top_symbol": "",
        "oi_top_diff": 0.0,
        "oi_top_direction": "",
        "flow_call_ask_premium": 0.0,
        "flow_put_ask_premium": 0.0,
        "flow_call_bid_premium": 0.0,
        "flow_put_bid_premium": 0.0,
        "flow_total_premium": 0.0,
        "flow_call_trade_count": 0.0,
        "flow_put_trade_count": 0.0,
    }


def option_quote_from_hot_row(row: Mapping[str, str], parsed: Mapping[str, Any], signal_date: str) -> Dict[str, Any]:
    bid = num(row.get("bid")) or 0.0
    ask = num(row.get("ask")) or 0.0
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else None
    option_close = num(row.get("option_close"))
    avg_price = num(row.get("avg_price"))
    stock_close = num(row.get("underlying_close"))
    volume = num(row.get("volume")) or 0.0
    open_interest = num(row.get("open_interest")) or 0.0
    premium = num(row.get("premium")) or 0.0
    spread = ask - bid if ask > 0 and bid >= 0 else None
    reference_price = mid or option_close or avg_price or ask or bid
    spread_pct = spread / reference_price if spread is not None and reference_price and reference_price > 0 else None
    expiry = parsed["expiry"]
    dte = trading_day_delta(signal_date, expiry)
    direction = "bullish" if parsed["option_type"] == "call" else "bearish"
    return {
        "date": signal_date,
        "ticker": parsed["ticker"],
        "option_symbol": parsed["option_symbol"],
        "option_type": parsed["option_type"],
        "direction": direction,
        "expiry": expiry,
        "strike": parsed["strike"],
        "dte": dte,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "high": num(row.get("high")),
        "low": num(row.get("low")),
        "option_close": option_close,
        "avg_price": avg_price,
        "stock_close": stock_close,
        "volume": volume,
        "open_interest": open_interest,
        "premium": premium,
        "iv": num(row.get("iv")),
        "spread": spread,
        "spread_pct": spread_pct,
        "ask_side_volume": num(row.get("ask_side_volume")) or 0.0,
        "bid_side_volume": num(row.get("bid_side_volume")) or 0.0,
        "sweep_volume": num(row.get("sweep_volume")) or 0.0,
        "multileg_volume": num(row.get("multileg_volume")) or 0.0,
    }


def update_hot_aggregate(f: Dict[str, Any], q: Mapping[str, Any], row: Mapping[str, str]) -> None:
    if q.get("stock_close") and not f.get("close"):
        f["close"] = q["stock_close"]
    if row.get("chain_prev_close"):
        prev = num(row.get("chain_prev_close"))
        if prev and not f.get("prev_close"):
            f["prev_close"] = prev
    f["hot_chain_count"] += 1
    f["hot_total_volume"] += q["volume"]
    f["hot_total_premium"] += q["premium"]
    f["hot_sweep_volume"] += q["sweep_volume"]
    f["hot_multileg_volume"] += q["multileg_volume"]
    if q.get("iv") is not None and q["volume"] > 0:
        f["hot_iv_weighted_sum"] += q["iv"] * q["volume"]
        f["hot_iv_weight"] += q["volume"]
    if q.get("spread_pct") is not None and q["volume"] > 0:
        f["hot_spread_weighted_sum"] += q["spread_pct"] * q["volume"]
        f["hot_spread_weight"] += q["volume"]
        if f["min_spread_pct"] is None or q["spread_pct"] < f["min_spread_pct"]:
            f["min_spread_pct"] = q["spread_pct"]
    if q["option_type"] == "call":
        f["hot_call_volume"] += q["volume"]
        f["hot_call_premium"] += q["premium"]
        f["hot_call_ask_volume"] += q["ask_side_volume"]
        f["hot_call_bid_volume"] += q["bid_side_volume"]
    else:
        f["hot_put_volume"] += q["volume"]
        f["hot_put_premium"] += q["premium"]
        f["hot_put_ask_volume"] += q["ask_side_volume"]
        f["hot_put_bid_volume"] += q["bid_side_volume"]


def maybe_set_best_option(best_options: Dict[Tuple[str, str], Dict[str, Any]], q: Dict[str, Any]) -> None:
    if q["ask"] <= 0 or q["bid"] < 0:
        return
    if q["dte"] is None or q["dte"] < 7 or q["dte"] > 70:
        return
    if q["volume"] < 50 or q["open_interest"] < 25:
        return
    spread_penalty = 1.0 + 8.0 * max(q.get("spread_pct") or 0.0, 0.0)
    score = (
        math.log1p(q["volume"])
        + math.log1p(q["open_interest"])
        + math.log1p(max(q["premium"], 0.0) / 1000.0)
    ) / spread_penalty
    candidate = dict(q)
    candidate["selection_score"] = score
    key = (q["ticker"], q["direction"])
    if key not in best_options or score > best_options[key].get("selection_score", -1):
        best_options[key] = candidate


def add_best_vertical_spreads(
    best_options: Dict[Tuple[str, str], Dict[str, Any]],
    option_quotes: Mapping[str, Mapping[str, Any]],
) -> None:
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for quote in option_quotes.values():
        if quote.get("ask", 0.0) <= 0 or quote.get("bid", 0.0) <= 0:
            continue
        if quote.get("dte") is None or quote["dte"] < 7 or quote["dte"] > 70:
            continue
        if quote.get("volume", 0.0) < 25 or quote.get("open_interest", 0.0) < 10:
            continue
        if quote.get("spread_pct") is not None and quote["spread_pct"] > 0.40:
            continue
        grouped[(quote["ticker"], quote["expiry"], quote["option_type"])].append(quote)

    for (ticker, expiry, option_type), quotes in grouped.items():
        stock = first_positive(q.get("stock_close") for q in quotes)
        if not stock:
            continue
        quotes = sorted(quotes, key=lambda q: q["strike"])
        if option_type == "put":
            spread = best_credit_spread(
                ticker=ticker,
                expiry=expiry,
                direction="bullish",
                short_candidates=[q for q in quotes if q["strike"] < stock],
                long_candidates=quotes,
                prefer_lower_long=True,
            )
        else:
            spread = best_credit_spread(
                ticker=ticker,
                expiry=expiry,
                direction="bearish",
                short_candidates=[q for q in quotes if q["strike"] > stock],
                long_candidates=quotes,
                prefer_lower_long=False,
            )
        if not spread:
            continue
        key = (ticker, spread["direction"])
        prior = best_options.get(key, {})
        if spread["selection_score"] > prior.get("selection_score", -1) * 0.85:
            best_options[key] = spread


def best_credit_spread(
    ticker: str,
    expiry: str,
    direction: str,
    short_candidates: Sequence[Mapping[str, Any]],
    long_candidates: Sequence[Mapping[str, Any]],
    prefer_lower_long: bool,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for short in short_candidates:
        for long in long_candidates:
            if short["expiry"] != long["expiry"] or short["option_type"] != long["option_type"]:
                continue
            width = short["strike"] - long["strike"] if prefer_lower_long else long["strike"] - short["strike"]
            if width <= 0:
                continue
            stock = short.get("stock_close") or 0.0
            if width > max(25.0, stock * 0.12):
                continue
            credit = (short.get("bid") or 0.0) - (long.get("ask") or 0.0)
            if credit <= 0:
                continue
            credit_to_width = credit / width
            if credit_to_width < 0.12 or credit_to_width > 0.75:
                continue
            combined_spread = ((short.get("ask") or 0.0) - (short.get("bid") or 0.0)) + (
                (long.get("ask") or 0.0) - (long.get("bid") or 0.0)
            )
            if combined_spread / max(credit, 0.01) > 0.85:
                continue
            min_volume = min(short.get("volume") or 0.0, long.get("volume") or 0.0)
            min_oi = min(short.get("open_interest") or 0.0, long.get("open_interest") or 0.0)
            if min_volume < 25 or min_oi < 10:
                continue
            max_risk = (width - credit) * 100.0 + 1.30
            if max_risk <= 0:
                continue
            score = (
                math.log1p(min_volume)
                + math.log1p(min_oi)
                + credit_to_width * 5.0
                - combined_spread / max(credit, 0.01)
            )
            candidate = {
                "strategy_kind": "credit_spread",
                "ticker": ticker,
                "direction": direction,
                "option_symbol": f"SELL {short['option_symbol']} / BUY {long['option_symbol']}",
                "option_type": short["option_type"],
                "strategy_type": "Bull Put Credit Spread" if direction == "bullish" else "Bear Call Credit Spread",
                "expiry": expiry,
                "dte": short.get("dte"),
                "strike": short["strike"],
                "long_strike": long["strike"],
                "spread_width": width,
                "entry_credit": credit,
                "max_risk": max_risk,
                "bid": credit,
                "ask": credit,
                "mid": credit,
                "spread": combined_spread,
                "spread_pct": combined_spread / max(credit, 0.01),
                "volume": min_volume,
                "open_interest": min_oi,
                "premium": credit * min_volume * 100.0,
                "iv": max(short.get("iv") or 0.0, long.get("iv") or 0.0),
                "selection_score": score,
                "legs": [
                    {
                        "action": "SELL",
                        "option_symbol": short["option_symbol"],
                        "option_type": short["option_type"],
                        "strike": short["strike"],
                        "bid": short["bid"],
                        "ask": short["ask"],
                    },
                    {
                        "action": "BUY",
                        "option_symbol": long["option_symbol"],
                        "option_type": long["option_type"],
                        "strike": long["strike"],
                        "bid": long["bid"],
                        "ask": long["ask"],
                    },
                ],
            }
            if best is None or score > best["selection_score"]:
                best = candidate
    return best


def update_chain_oi_aggregate(f: Dict[str, Any], row: Mapping[str, str], parsed: Mapping[str, Any]) -> None:
    oi_diff = abs(num(row.get("oi_diff_plain")) or 0.0)
    volume = num(row.get("volume")) or 0.0
    f["oi_total_diff"] += oi_diff
    if parsed["option_type"] == "call":
        f["oi_call_diff"] += oi_diff
        f["oi_call_volume"] += volume
        direction = "bullish"
    else:
        f["oi_put_diff"] += oi_diff
        f["oi_put_volume"] += volume
        direction = "bearish"
    if oi_diff > f["oi_top_diff"]:
        f["oi_top_diff"] = oi_diff
        f["oi_top_symbol"] = parsed["option_symbol"]
        f["oi_top_direction"] = direction
    if row.get("sector") and not f.get("sector"):
        f["sector"] = row.get("sector") or ""
    stock_price = num(row.get("stock_price"))
    if stock_price and not f.get("close"):
        f["close"] = stock_price
    if row.get("next_earnings_date") and not f.get("next_earnings_date"):
        f["next_earnings_date"] = row.get("next_earnings_date") or ""


def update_option_trade_aggregate(f: Dict[str, Any], row: Mapping[str, str]) -> None:
    premium = option_trade_premium(row)
    option_type = str(row.get("option_type") or "").lower()
    side = str(row.get("side") or "").lower()
    f["flow_total_premium"] += premium
    if option_type == "call":
        f["flow_call_trade_count"] += 1
        if side == "ask":
            f["flow_call_ask_premium"] += premium
        elif side == "bid":
            f["flow_call_bid_premium"] += premium
    elif option_type == "put":
        f["flow_put_trade_count"] += 1
        if side == "ask":
            f["flow_put_ask_premium"] += premium
        elif side == "bid":
            f["flow_put_bid_premium"] += premium


def maybe_update_quote_from_bot_trade(
    option_quotes: Dict[str, Dict[str, Any]],
    row: Mapping[str, str],
    signal_date: str,
) -> None:
    parsed = parse_option_symbol(row.get("option_chain_id", ""))
    if not parsed:
        return
    bid = first_positive([row.get("nbbo_bid"), row.get("ewma_nbbo_bid")]) or 0.0
    ask = first_positive([row.get("nbbo_ask"), row.get("ewma_nbbo_ask")]) or 0.0
    price = num(row.get("price"))
    size = num(row.get("size")) or 0.0
    premium = abs(num(row.get("premium")) or ((price or 0.0) * size * 100.0))
    volume = num(row.get("volume")) or 0.0
    open_interest = num(row.get("open_interest")) or 0.0
    symbol = parsed["option_symbol"]
    existing = option_quotes.get(symbol)
    # Bot EOD can be enormous. Always refresh quotes already selected from hot
    # chains, and keep new bot-only quotes only when the trade is material.
    if existing is None and premium < 10_000.0 and volume < 100:
        return
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else price
    reference = mid or price or ask or bid
    spread = ask - bid if ask > 0 and bid >= 0 else None
    spread_pct = spread / reference if spread is not None and reference and reference > 0 else None
    q = dict(existing or {})
    q.update(
        {
            "date": signal_date,
            "ticker": parsed["ticker"],
            "option_symbol": symbol,
            "option_type": parsed["option_type"],
            "direction": "bullish" if parsed["option_type"] == "call" else "bearish",
            "expiry": parsed["expiry"],
            "strike": parsed["strike"],
            "dte": trading_day_delta(signal_date, parsed["expiry"]),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "stock_close": num(row.get("underlying_price")) or q.get("stock_close"),
            "volume": max(volume, q.get("volume") or 0.0),
            "open_interest": max(open_interest, q.get("open_interest") or 0.0),
            "premium": max(premium, q.get("premium") or 0.0),
            "iv": num(row.get("implied_volatility")) or q.get("iv"),
            "spread": spread,
            "spread_pct": spread_pct,
            "quote_source": "bot_eod",
        }
    )
    option_quotes[symbol] = q


def apply_gex_context(
    date_dir: Path,
    signal_date: str,
    features: Dict[str, Dict[str, Any]],
    source_files: List[str],
    skipped_sources: List[Dict[str, Any]],
    counts: Counter[str],
) -> None:
    gex_dir = date_dir / "enrichments" / "uw" / "uw_gex_raw"
    if not gex_dir.exists():
        return
    for path in sorted(gex_dir.glob(f"uw_gex_{signal_date}_*.json")):
        counts["gex_files_seen"] += 1
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:
            skipped_sources.append({"source": str(path), "reason": f"gex_parse_error:{exc}"})
            continue
        ticker = clean_ticker(str(data.get("ticker") or path.stem.split("_")[-1]))
        if not ticker:
            continue
        f = features.setdefault(ticker, new_feature(signal_date, ticker))
        f["source_flags"].add("gex")
        f["gex_available"] = 1
        captured = str(data.get("captured_utc") or "")
        captured_date = captured[:10] if re.match(r"\d{4}-\d{2}-\d{2}", captured) else ""
        if captured_date and captured_date <= signal_date and data.get("collection_ok"):
            f["gex_capture_ok_point_in_time"] = 1
            source_files.append(str(path.resolve()))
        else:
            skipped_sources.append(
                {
                    "source": str(path.resolve()),
                    "reason": "gex_capture_after_signal_date_or_missing_capture_time",
                    "captured_utc": captured,
                }
            )


def finalize_feature(f: Dict[str, Any]) -> None:
    f["source_flags"] = sorted(f["source_flags"])
    close = f.get("close")
    prev = f.get("prev_close")
    f["stock_return_1d"] = pct_change(close, prev)
    f["call_volume_ratio_30d"] = safe_div(f.get("call_volume"), f.get("avg30_call_volume"))
    f["put_volume_ratio_30d"] = safe_div(f.get("put_volume"), f.get("avg30_put_volume"))
    f["stock_volume_ratio_30d"] = safe_div(f.get("stock_volume"), f.get("avg30_stock_volume"))
    f["put_call_ratio"] = safe_div(f.get("put_volume"), f.get("call_volume"))
    total_premium = (f.get("bullish_premium") or 0.0) + (f.get("bearish_premium") or 0.0)
    f["premium_bias"] = safe_div(
        (f.get("bullish_premium") or 0.0) - (f.get("bearish_premium") or 0.0),
        total_premium,
    )
    f["hot_call_ratio"] = safe_div(f.get("hot_call_volume"), f.get("hot_total_volume"))
    f["hot_put_ratio"] = safe_div(f.get("hot_put_volume"), f.get("hot_total_volume"))
    f["hot_call_ask_ratio"] = safe_div(
        f.get("hot_call_ask_volume"),
        (f.get("hot_call_ask_volume") or 0.0) + (f.get("hot_call_bid_volume") or 0.0),
    )
    f["hot_put_ask_ratio"] = safe_div(
        f.get("hot_put_ask_volume"),
        (f.get("hot_put_ask_volume") or 0.0) + (f.get("hot_put_bid_volume") or 0.0),
    )
    flow_bullish = (f.get("flow_call_ask_premium") or 0.0) + (f.get("flow_put_bid_premium") or 0.0)
    flow_bearish = (f.get("flow_put_ask_premium") or 0.0) + (f.get("flow_call_bid_premium") or 0.0)
    f["flow_premium_bias"] = safe_div(flow_bullish - flow_bearish, flow_bullish + flow_bearish)
    f["flow_call_ask_ratio"] = safe_div(
        f.get("flow_call_ask_premium"),
        (f.get("flow_call_ask_premium") or 0.0) + (f.get("flow_call_bid_premium") or 0.0),
    )
    f["flow_put_ask_ratio"] = safe_div(
        f.get("flow_put_ask_premium"),
        (f.get("flow_put_ask_premium") or 0.0) + (f.get("flow_put_bid_premium") or 0.0),
    )
    f["avg_iv"] = safe_div(f.get("hot_iv_weighted_sum"), f.get("hot_iv_weight"))
    f["avg_spread_pct"] = safe_div(f.get("hot_spread_weighted_sum"), f.get("hot_spread_weight"))
    f["liquidity_score"] = (
        math.log1p(f.get("hot_total_volume") or 0.0)
        + math.log1p((f.get("hot_total_premium") or 0.0) / 1000.0)
        + math.log1p(f.get("total_open_interest") or 0.0)
    )
    f["earnings_dte"] = calendar_day_delta(f["date"], f.get("next_earnings_date") or "")


def compute_market_regime(signal_date: str, features: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    index_returns = {
        ticker: features.get(ticker, {}).get("stock_return_1d")
        for ticker in sorted(INDEX_TICKERS)
        if features.get(ticker, {}).get("stock_return_1d") is not None
    }
    index_pcr = {
        ticker: features.get(ticker, {}).get("put_call_ratio")
        for ticker in sorted(INDEX_TICKERS)
        if features.get(ticker, {}).get("put_call_ratio") is not None
    }
    returns = [v for v in index_returns.values() if v is not None]
    avg_index_return = statistics.fmean(returns) if returns else None
    all_returns = [
        f.get("stock_return_1d")
        for f in features.values()
        if f.get("stock_return_1d") is not None and f.get("close")
    ]
    breadth_positive = safe_div(sum(1 for r in all_returns if r > 0), len(all_returns))
    spy_pcr = index_pcr.get("SPY")
    if avg_index_return is None:
        regime = "UNKNOWN"
    elif avg_index_return > 0.003 and (breadth_positive or 0) >= 0.52 and (spy_pcr or 1.0) < 1.35:
        regime = "RISK_ON"
    elif avg_index_return < -0.003 or (breadth_positive is not None and breadth_positive < 0.43) or (spy_pcr or 0) > 1.55:
        regime = "RISK_OFF"
    else:
        regime = "MIXED"
    sector_scores: Dict[str, List[float]] = defaultdict(list)
    for f in features.values():
        if f.get("sector") and f.get("stock_return_1d") is not None:
            sector_scores[str(f["sector"])].append(float(f["stock_return_1d"]))
    top_sectors = sorted(
        (
            {"sector": k, "avg_return": round(statistics.fmean(v), 5), "count": len(v)}
            for k, v in sector_scores.items()
            if len(v) >= 3
        ),
        key=lambda x: x["avg_return"],
        reverse=True,
    )[:5]
    return {
        "date": signal_date,
        "regime": regime,
        "avg_index_return": avg_index_return,
        "breadth_positive_pct": breadth_positive,
        "index_returns": index_returns,
        "index_put_call_ratio": index_pcr,
        "vix_context": "VIX not present in local UW stock-screener source for this date",
        "top_sectors": top_sectors,
        "source": "local_uw_stock_screener_and_hot_chains",
    }


def learn_pattern_config(snapshots: Sequence[Snapshot]) -> Dict[str, Any]:
    call_ratios: List[float] = []
    put_ratios: List[float] = []
    premiums: List[float] = []
    oi_diffs: List[float] = []
    spreads: List[float] = []
    ivs: List[float] = []
    liquidity: List[float] = []
    for snap in snapshots:
        for f in snap.features.values():
            if f.get("call_volume_ratio_30d"):
                call_ratios.append(float(f["call_volume_ratio_30d"]))
            if f.get("put_volume_ratio_30d"):
                put_ratios.append(float(f["put_volume_ratio_30d"]))
            total_signal_premium = max(float(f.get("hot_total_premium") or 0.0), float(f.get("flow_total_premium") or 0.0))
            if total_signal_premium:
                premiums.append(total_signal_premium)
            oi = max(float(f.get("oi_call_diff") or 0.0), float(f.get("oi_put_diff") or 0.0))
            if oi:
                oi_diffs.append(oi)
            if f.get("avg_spread_pct") is not None and f["avg_spread_pct"] > 0:
                spreads.append(float(f["avg_spread_pct"]))
            if f.get("avg_iv"):
                ivs.append(float(f["avg_iv"]))
            if f.get("liquidity_score"):
                liquidity.append(float(f["liquidity_score"]))
    return {
        "trained_on_dates": [s.signal_date for s in snapshots],
        "min_call_volume_ratio": max(1.35, quantile(call_ratios, 0.78, default=1.8)),
        "min_put_volume_ratio": max(1.35, quantile(put_ratios, 0.78, default=1.8)),
        "min_hot_premium": max(100_000.0, quantile(premiums, 0.70, default=250_000.0)),
        "min_oi_diff": max(5_000.0, quantile(oi_diffs, 0.80, default=25_000.0)),
        "max_spread_pct": min(0.35, max(0.08, quantile(spreads, 0.75, default=0.25))),
        "high_iv": max(0.20, quantile(ivs, 0.75, default=0.45)),
        "min_liquidity_score": max(8.0, quantile(liquidity, 0.45, default=10.0)),
        "min_ask_side_ratio": 0.52,
        "max_event_dte_without_event_strategy": 2,
        "discovery_method": "training_window_quantiles_with_fixed_risk_floors",
    }


def generate_signals_for_snapshot(
    snapshot: Snapshot,
    pattern_config: Mapping[str, Any],
    max_signals: int,
) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    market = snapshot.market_regime.get("regime", "UNKNOWN")
    for ticker, f in snapshot.features.items():
        if not f.get("close") or f.get("close") <= 0:
            continue
        if ticker.startswith("^"):
            continue
        candidates: List[Tuple[str, str, float, List[str]]] = []
        call_ratio = f.get("call_volume_ratio_30d") or 0.0
        put_ratio = f.get("put_volume_ratio_30d") or 0.0
        screen_premium_bias = f.get("premium_bias") or 0.0
        flow_premium_bias = f.get("flow_premium_bias") or 0.0
        premium_bias = flow_premium_bias if abs(flow_premium_bias) > abs(screen_premium_bias) else screen_premium_bias
        hot_call_ask = max(f.get("hot_call_ask_ratio") or 0.0, f.get("flow_call_ask_ratio") or 0.0)
        hot_put_ask = max(f.get("hot_put_ask_ratio") or 0.0, f.get("flow_put_ask_ratio") or 0.0)
        stock_ret = f.get("stock_return_1d") or 0.0
        hot_premium = max(f.get("hot_total_premium") or 0.0, f.get("flow_total_premium") or 0.0)
        liquidity_score = f.get("liquidity_score") or 0.0
        avg_iv = f.get("avg_iv") or 0.0

        if (
            call_ratio >= pattern_config["min_call_volume_ratio"]
            and premium_bias > 0.03
            and hot_call_ask >= pattern_config["min_ask_side_ratio"]
            and market != "RISK_OFF"
        ):
            score = (
                zish(call_ratio, pattern_config["min_call_volume_ratio"])
                + zish(hot_premium, pattern_config["min_hot_premium"])
                + 2.0 * premium_bias
                + max(stock_ret, -0.02)
            )
            candidates.append(
                (
                    "BULLISH_FLOW_EXPANSION",
                    "bullish",
                    score,
                    [
                        "call volume expansion versus 30-day average",
                        "bullish premium bias",
                        "ask-side call participation",
                    ],
                )
            )

        if (
            put_ratio >= pattern_config["min_put_volume_ratio"]
            and premium_bias < -0.03
            and hot_put_ask >= pattern_config["min_ask_side_ratio"]
            and market != "RISK_ON"
        ):
            score = (
                zish(put_ratio, pattern_config["min_put_volume_ratio"])
                + zish(hot_premium, pattern_config["min_hot_premium"])
                + 2.0 * abs(premium_bias)
                + max(-stock_ret, -0.02)
            )
            candidates.append(
                (
                    "BEARISH_PUT_FLOW_EXPANSION",
                    "bearish",
                    score,
                    [
                        "put volume expansion versus 30-day average",
                        "bearish premium bias",
                        "ask-side put participation",
                    ],
                )
            )

        if f.get("oi_call_diff", 0.0) >= pattern_config["min_oi_diff"] and hot_call_ask >= 0.50:
            score = zish(f["oi_call_diff"], pattern_config["min_oi_diff"]) + zish(
                liquidity_score, pattern_config["min_liquidity_score"]
            )
            candidates.append(
                (
                    "OI_GAMMA_CONTINUATION",
                    "bullish",
                    score,
                    ["large call open-interest change", "supportive call hot-chain pressure"],
                )
            )
        if f.get("oi_put_diff", 0.0) >= pattern_config["min_oi_diff"] and hot_put_ask >= 0.50:
            score = zish(f["oi_put_diff"], pattern_config["min_oi_diff"]) + zish(
                liquidity_score, pattern_config["min_liquidity_score"]
            )
            candidates.append(
                (
                    "OI_GAMMA_CONTINUATION",
                    "bearish",
                    score,
                    ["large put open-interest change", "supportive put hot-chain pressure"],
                )
            )

        if avg_iv >= pattern_config["high_iv"] and hot_premium >= pattern_config["min_hot_premium"]:
            direction = "bullish" if premium_bias >= 0 else "bearish"
            score = zish(avg_iv, pattern_config["high_iv"]) + zish(
                hot_premium, pattern_config["min_hot_premium"]
            )
            candidates.append(
                (
                    "VOL_EXPANSION_CATALYST",
                    direction,
                    score,
                    ["elevated IV", "large options premium concentration"],
                )
            )

        if ticker in INDEX_TICKERS and put_ratio >= pattern_config["min_put_volume_ratio"]:
            score = zish(put_ratio, pattern_config["min_put_volume_ratio"]) + max(
                f.get("put_call_ratio") or 0.0, 0.0
            )
            candidates.append(
                (
                    "INDEX_HEDGE_PRESSURE",
                    "bearish",
                    score,
                    ["index put-volume pressure", "market hedge demand"],
                )
            )

        for family, direction, score, reasons in candidates:
            quote = snapshot.best_options.get((ticker, direction))
            signals.append(build_signal(snapshot, f, family, direction, score, reasons, quote, pattern_config))

    signals.sort(key=lambda x: (x["pattern_score"], x.get("hot_total_premium", 0.0)), reverse=True)
    return signals[:max_signals]


def build_signal(
    snapshot: Snapshot,
    f: Mapping[str, Any],
    family: str,
    direction: str,
    score: float,
    reasons: Sequence[str],
    quote: Optional[Mapping[str, Any]],
    pattern_config: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: List[str] = []
    if quote is None:
        blockers.append("NO_TRADEABLE_OPTION_QUOTE")
        quote = {}
    strategy_kind = quote.get("strategy_kind", "long_option")
    detailed_family = detailed_pattern_family(family, direction, strategy_kind, f.get("sector") or "")
    spread_pct = quote.get("spread_pct")
    if spread_pct is None:
        blockers.append("MISSING_BID_ASK_SPREAD")
    elif strategy_kind == "credit_spread" and spread_pct > 0.65:
        blockers.append("BID_ASK_SPREAD_TOO_WIDE")
    elif strategy_kind != "credit_spread" and spread_pct > pattern_config["max_spread_pct"]:
        blockers.append("BID_ASK_SPREAD_TOO_WIDE")
    if strategy_kind == "credit_spread" and quote.get("entry_credit", 0.0) <= 0:
        blockers.append("MISSING_ENTRY_CREDIT")
    elif strategy_kind != "credit_spread" and quote.get("ask", 0.0) <= 0:
        blockers.append("MISSING_ENTRY_ASK")
    if quote.get("volume", 0.0) < 50 or quote.get("open_interest", 0.0) < 25:
        blockers.append("OPTION_LIQUIDITY_TOO_LOW")
    if quote.get("dte") is None or quote.get("dte", 0) < 7:
        blockers.append("DTE_TOO_SHORT_FOR_VALIDATION_HORIZONS")
    if (
        f.get("earnings_dte") is not None
        and f.get("earnings_dte") >= 0
        and f.get("earnings_dte") <= pattern_config["max_event_dte_without_event_strategy"]
        and family != "VOL_EXPANSION_CATALYST"
    ):
        blockers.append("NEAR_TERM_EARNINGS_EVENT_RISK")
    if snapshot.market_regime.get("regime") == "RISK_OFF" and direction == "bullish":
        blockers.append("MARKET_REGIME_CONFLICT")
    if snapshot.market_regime.get("regime") == "RISK_ON" and direction == "bearish" and family != "INDEX_HEDGE_PRESSURE":
        blockers.append("MARKET_REGIME_CONFLICT")

    entry_ask = quote.get("ask")
    entry_bid = quote.get("bid")
    entry_mid = quote.get("mid")
    entry_credit = quote.get("entry_credit")
    max_risk = quote.get("max_risk") if strategy_kind == "credit_spread" else (entry_ask * 100.0 + 0.65 if entry_ask else None)
    strategy_type = quote.get("strategy_type") or ("Long Call Debit" if direction == "bullish" else "Long Put Debit")
    if strategy_kind == "credit_spread":
        entry_range = f"credit {entry_credit:.2f}" if entry_credit is not None else ""
        target_profit = (entry_credit * 50.0) if entry_credit else None
        stop_rule = "Exit if spread debit reaches 2x entry credit or short strike is breached."
        time_stop = "Exit after 5 trading days or at 50% credit capture, whichever comes first."
        invalidation_rule = "Short strike breach, event-risk surprise, or regime flip against spread direction."
        review_close_rule = "Review daily; close at 50% credit capture, 2x credit stop, or two sessions before expiration."
    else:
        entry_range = format_entry_range(entry_bid, entry_ask)
        target_profit = max_risk if max_risk else None
        stop_rule = "Exit if option bid loses 50% from entry debit or thesis invalidates."
        time_stop = "Exit after 5 trading days unless target/stop triggers first."
        invalidation_rule = "Underlying closes through thesis invalidation level or catalyst/macro direction reverses."
        review_close_rule = "Review daily; close at 100% option gain, 50% loss, time stop, or catalyst invalidation."
    return {
        "date": snapshot.signal_date,
        "ticker": f["ticker"],
        "direction": direction,
        "pattern_family": detailed_family,
        "base_pattern_family": family,
        "pattern_score": round(score, 6),
        "classification": "BLOCKED" if blockers else "WATCH",
        "block_reasons": blockers,
        "reason_summary": "; ".join(reasons),
        "market_regime": snapshot.market_regime.get("regime", "UNKNOWN"),
        "sector": f.get("sector") or "",
        "close": f.get("close"),
        "stock_return_1d": f.get("stock_return_1d"),
        "call_volume_ratio_30d": f.get("call_volume_ratio_30d"),
        "put_volume_ratio_30d": f.get("put_volume_ratio_30d"),
        "premium_bias": f.get("premium_bias"),
        "hot_total_premium": max(f.get("hot_total_premium") or 0.0, f.get("flow_total_premium") or 0.0),
        "flow_total_premium": f.get("flow_total_premium"),
        "flow_premium_bias": f.get("flow_premium_bias"),
        "avg_iv": f.get("avg_iv"),
        "quote_iv": quote.get("iv"),
        "avg_spread_pct": f.get("avg_spread_pct"),
        "lead_option_symbol": quote.get("option_symbol", ""),
        "quote_source": quote.get("quote_source", ""),
        "strategy_kind": strategy_kind,
        "legs_json": stable_json(quote.get("legs", [])),
        "strategy_type": strategy_type,
        "option_type": quote.get("option_type", "call" if direction == "bullish" else "put"),
        "strike": quote.get("strike"),
        "long_strike": quote.get("long_strike"),
        "spread_width": quote.get("spread_width"),
        "expiry": quote.get("expiry", ""),
        "dte": quote.get("dte"),
        "entry_bid": entry_bid,
        "entry_mid": entry_mid,
        "entry_ask": entry_ask,
        "entry_credit": entry_credit,
        "entry_range": entry_range,
        "max_risk_per_contract": max_risk,
        "target_profit": target_profit,
        "stop_rule": stop_rule,
        "time_stop": time_stop,
        "invalidation_rule": invalidation_rule,
        "review_close_rule": review_close_rule,
        "expected_hold_days": 5,
        "position_size_tier": "RISK_CAPPED_RESEARCH" if blockers else "STANDARD_REVIEW",
        "liquidity_volume": quote.get("volume"),
        "liquidity_open_interest": quote.get("open_interest"),
        "bid_ask_spread_pct": spread_pct,
        "earnings_dte": f.get("earnings_dte"),
        "next_earnings_date": f.get("next_earnings_date") or "",
    }


def detailed_pattern_family(base_family: str, direction: str, strategy_kind: str, sector: str) -> str:
    if str(base_family).startswith("BASELINE_"):
        return base_family
    return "__".join(
        [
            str(base_family),
            clean_family_part(direction),
            clean_family_part(strategy_kind),
            clean_family_part(sector or "NO_SECTOR"),
        ]
    )


def clean_family_part(value: str) -> str:
    text = re.sub(r"[^A-Z0-9]+", "_", str(value or "").upper()).strip("_")
    return text or "NA"


def classify_daily_signals(
    signals: Sequence[Dict[str, Any]],
    family_tiers: Mapping[str, Mapping[str, Any]],
    snapshot: Snapshot,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for signal in signals:
        row = dict(signal)
        tier_info = family_tiers.get(
            signal["pattern_family"],
            {
                "confidence_tier": "RESEARCH_ONLY",
                "validation_note": "Pattern family has no completed validation evidence.",
                "beats_baselines_count": 0,
            },
        )
        blockers = list(row.get("block_reasons") or [])
        tier = tier_info.get("confidence_tier", "RESEARCH_ONLY")
        if tier != "PROVEN":
            blockers.append("PATTERN_VALIDATION_NOT_PROVEN")
        if int(tier_info.get("validation_scored_count") or 0) < 30:
            blockers.append("LIMITED_OUT_OF_SAMPLE_SAMPLE")
        if int(tier_info.get("beats_baselines_count") or 0) < 2:
            blockers.append("DOES_NOT_BEAT_TWO_BASELINES")
        validation_avg_r = num(tier_info.get("validation_average_net_r"))
        if validation_avg_r is not None and validation_avg_r <= 0:
            blockers.append("VALIDATION_EXPECTANCY_NEGATIVE")
        validation_pf = num(tier_info.get("validation_profit_factor"))
        if validation_pf is not None and validation_pf < 1.20:
            blockers.append("PROFIT_FACTOR_BELOW_AUTO_APPROVAL")
        blockers = sorted(set(blockers))
        if tier == "PROVEN" and not blockers:
            classification = "TRADE"
        elif "NO_TRADEABLE_OPTION_QUOTE" in blockers or "BID_ASK_SPREAD_TOO_WIDE" in blockers:
            classification = "AVOID"
        elif signal["classification"] == "BLOCKED":
            classification = "AVOID"
        else:
            classification = "WATCH"
        row["classification"] = classification
        row["confidence_tier"] = tier
        row["validation_note"] = tier_info.get("validation_note", "")
        row["_base_success_probability"] = tier_info.get("validation_success_probability")
        row["_base_failure_probability"] = tier_info.get("validation_failure_probability")
        row["_base_probability_score"] = tier_info.get("validation_probability_score")
        row["pattern_success_probability_pct"] = pct_value(tier_info.get("validation_success_probability"))
        row["pattern_failure_probability_pct"] = pct_value(tier_info.get("validation_failure_probability"))
        row["pattern_probability_score"] = pct_value(tier_info.get("validation_probability_score"))
        row["probability_evidence"] = tier_info.get("probability_evidence", "")
        row["block_reasons"] = blockers
        row["blocker_categories"] = decompose_blockers(blockers)
        row["current_market_alignment"] = snapshot.market_regime.get("regime", "UNKNOWN")
        row["why_actionable_now"] = (
            "Passes validation, liquidity, quote, risk, event, and regime checks."
            if classification == "TRADE"
            else "Not actionable because " + "; ".join(decompose_blockers(blockers)[:5])
        )
        row["major_risks"] = daily_major_risks(row)
        rows.append(row)
    apply_candidate_probability_adjustments(rows)
    rows.sort(
        key=lambda r: (
            classification_rank(r["classification"]),
            num(r.get("trade_probability_score")) or -1.0,
            r["pattern_score"],
        ),
        reverse=True,
    )
    return rows


def classification_rank(value: str) -> int:
    return {"TRADE": 3, "WATCH": 2, "AVOID": 1, "BLOCKED": 0}.get(value, 0)


def apply_candidate_probability_adjustments(rows: Sequence[Dict[str, Any]]) -> None:
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[str(row.get("pattern_family") or "")].append(row)
    percentiles: Dict[int, float] = {}
    for family_rows in by_family.values():
        scored = sorted(
            ((num(row.get("pattern_score")) or 0.0, idx) for idx, row in enumerate(family_rows)),
            key=lambda item: item[0],
        )
        denom = max(len(scored) - 1, 1)
        for rank, (_, local_idx) in enumerate(scored):
            percentiles[id(family_rows[local_idx])] = rank / denom if len(scored) > 1 else 0.5

    for row in rows:
        base_success = num(row.get("_base_success_probability"))
        base_failure = num(row.get("_base_failure_probability"))
        base_score = num(row.get("_base_probability_score"))
        if base_success is None:
            row["success_probability_pct"] = None
            row["failure_probability_pct"] = None
            row["probability_score"] = None
            row["trade_success_probability_pct"] = None
            row["trade_failure_probability_pct"] = None
            row["trade_probability_score"] = None
            row["probability_components"] = "No validated family probability."
            continue
        percentile = percentiles.get(id(row), 0.5)
        signal_adj = (percentile - 0.5) * 0.08
        spread_adj = spread_probability_adjustment(row.get("bid_ask_spread_pct"))
        liquidity_adj = liquidity_probability_adjustment(row.get("liquidity_volume"), row.get("liquidity_open_interest"))
        event_adj = event_probability_adjustment(row.get("earnings_dte"))
        blocker_adj = -0.04 if row.get("classification") != "TRADE" else 0.0
        total_adj = signal_adj + spread_adj + liquidity_adj + event_adj + blocker_adj
        trade_success = clamp(base_success + total_adj, 0.01, 0.90)
        trade_score = clamp((base_score if base_score is not None else base_success) + total_adj, 0.0, 0.90)
        row["success_probability_pct"] = pct_value(trade_success)
        row["failure_probability_pct"] = pct_value(1.0 - trade_success if base_failure is not None else None)
        row["probability_score"] = pct_value(trade_score)
        row["trade_success_probability_pct"] = row["success_probability_pct"]
        row["trade_failure_probability_pct"] = row["failure_probability_pct"]
        row["trade_probability_score"] = row["probability_score"]
        row["probability_components"] = (
            f"pattern_base={fmt_pct(base_success)}; "
            f"signal_rank_adj={fmt_signed_pct(signal_adj)}; "
            f"spread_adj={fmt_signed_pct(spread_adj)}; "
            f"liquidity_adj={fmt_signed_pct(liquidity_adj)}; "
            f"event_adj={fmt_signed_pct(event_adj)}; "
            f"blocker_adj={fmt_signed_pct(blocker_adj)}"
        )


def spread_probability_adjustment(spread_pct: Any) -> float:
    spread = num(spread_pct)
    if spread is None:
        return -0.03
    return clamp(0.04 - 0.40 * max(spread, 0.0), -0.06, 0.04)


def liquidity_probability_adjustment(volume: Any, open_interest: Any) -> float:
    vol = max(num(volume) or 0.0, 0.0)
    oi = max(num(open_interest) or 0.0, 0.0)
    score = math.log1p(vol) + math.log1p(oi)
    return clamp((score - 14.0) * 0.006, -0.03, 0.03)


def event_probability_adjustment(earnings_dte: Any) -> float:
    dte = num(earnings_dte)
    if dte is None or dte < 0:
        return 0.0
    if dte <= 2:
        return -0.06
    if dte <= 7:
        return -0.025
    return 0.0


def daily_major_risks(row: Mapping[str, Any]) -> str:
    risks = []
    if row.get("next_earnings_date"):
        risks.append(f"earnings/event date {row['next_earnings_date']}")
    if row.get("bid_ask_spread_pct") is not None and row["bid_ask_spread_pct"] > 0.15:
        risks.append("wide option spread")
    if row.get("confidence_tier") != "PROVEN":
        risks.append("pattern not proven out-of-sample")
    if row.get("market_regime") in {"MIXED", "UNKNOWN"}:
        risks.append("uncertain market regime")
    return "; ".join(risks) if risks else "defined-risk option debit can still expire worthless"


def run_historical_validation(
    snapshots: Mapping[str, Snapshot],
    source_dates: Sequence[str],
    min_month_dates: int,
    top_candidates_per_day: int,
    seed: int,
) -> Dict[str, Any]:
    splits = build_validation_splits(source_dates, min_month_dates)
    all_signal_rows: List[Dict[str, Any]] = []
    all_outcomes: List[Dict[str, Any]] = []
    all_baseline_outcomes: List[Dict[str, Any]] = []
    pattern_definitions: List[Dict[str, Any]] = []
    for split in splits:
        train_snaps = [snapshots[d] for d in split["train_dates"]]
        validation_snaps = [snapshots[d] for d in split["validation_dates"]]
        pattern_config = learn_pattern_config(train_snaps)
        pattern_definitions.append(
            {
                "split": split["name"],
                "train_start": split["train_dates"][0],
                "train_end": split["train_dates"][-1],
                "validation_start": split["validation_dates"][0],
                "validation_end": split["validation_dates"][-1],
                "pattern_config_json": stable_json(pattern_config),
            }
        )
        train_signals = []
        for snap in train_snaps:
            train_signals.extend(generate_signals_for_snapshot(snap, pattern_config, top_candidates_per_day))
        validation_signals = []
        for snap in validation_snaps:
            validation_signals.extend(generate_signals_for_snapshot(snap, pattern_config, top_candidates_per_day))
        for sig in train_signals:
            sig = dict(sig)
            sig["split"] = split["name"]
            sig["sample"] = "TRAIN"
            all_signal_rows.append(flatten_signal(sig))
        for sig in validation_signals:
            sig = dict(sig)
            sig["split"] = split["name"]
            sig["sample"] = "VALIDATION"
            all_signal_rows.append(flatten_signal(sig))
        all_outcomes.extend(score_signals(train_signals, snapshots, source_dates, split["name"], "TRAIN"))
        all_outcomes.extend(score_signals(validation_signals, snapshots, source_dates, split["name"], "VALIDATION"))
        baseline_signals = generate_baseline_signals(
            validation_signals,
            validation_snaps,
            pattern_config,
            top_candidates_per_day,
            seed,
            split["name"],
        )
        all_baseline_outcomes.extend(
            score_signals(baseline_signals, snapshots, source_dates, split["name"], "BASELINE")
        )

    validation_scorecard = summarize_outcomes(all_outcomes, sample="VALIDATION")
    validation_gate_outcomes = select_validation_gate_outcomes(all_outcomes)
    validation_gate_scorecard = summarize_outcomes(validation_gate_outcomes, sample="VALIDATION")
    train_scorecard = summarize_outcomes(all_outcomes, sample="TRAIN")
    baseline_comparison = summarize_baselines(all_baseline_outcomes)
    family_tiers = assign_family_tiers(validation_gate_scorecard, baseline_comparison)
    regime_sector = summarize_regime_sector(all_outcomes)
    return {
        "splits": splits,
        "pattern_definitions": pattern_definitions,
        "signal_rows": all_signal_rows,
        "outcomes": all_outcomes,
        "baseline_outcomes": all_baseline_outcomes,
        "validation_scorecard": validation_scorecard,
        "validation_gate_scorecard": validation_gate_scorecard,
        "train_scorecard": train_scorecard,
        "baseline_comparison": baseline_comparison,
        "family_tiers": family_tiers,
        "regime_sector": regime_sector,
    }


def empty_validation_bundle() -> Dict[str, Any]:
    return {
        "splits": [],
        "pattern_definitions": [],
        "signal_rows": [],
        "outcomes": [],
        "baseline_outcomes": [],
        "validation_scorecard": [],
        "validation_gate_scorecard": [],
        "train_scorecard": [],
        "baseline_comparison": [],
        "family_tiers": {},
        "regime_sector": [],
    }


def build_validation_splits(source_dates: Sequence[str], min_month_dates: int = 5) -> List[Dict[str, Any]]:
    by_month: Dict[str, List[str]] = defaultdict(list)
    for d in source_dates:
        by_month[d[:7]].append(d)
    months = sorted(by_month)
    splits: List[Dict[str, Any]] = []

    def add_split(train_month: str, validation_month: str, explicit_name: Optional[str] = None) -> None:
        train_dates = sorted(by_month.get(train_month, []))
        validation_dates = sorted(by_month.get(validation_month, []))
        if len(train_dates) < min_month_dates or len(validation_dates) < min_month_dates:
            return
        if train_dates[-1] >= validation_dates[0]:
            return
        name = explicit_name or f"discover_{train_month}_validate_{validation_month}"
        key = (train_month, validation_month)
        if key in {(s["train_month"], s["validation_month"]) for s in splits}:
            return
        splits.append(
            {
                "name": name,
                "train_month": train_month,
                "validation_month": validation_month,
                "train_dates": train_dates,
                "validation_dates": validation_dates,
            }
        )

    add_split("2025-12", "2026-02", "required_dec_2025_to_feb_2026")
    add_split("2026-01", "2026-03", "required_jan_2026_to_mar_2026")
    for idx, train_month in enumerate(months):
        if idx + 2 < len(months):
            add_split(train_month, months[idx + 2])
    for validation_month in months[2:]:
        train_dates: List[str] = []
        for month in months:
            if month >= validation_month:
                break
            train_dates.extend(sorted(by_month.get(month, [])))
        validation_dates = sorted(by_month.get(validation_month, []))
        if len(train_dates) >= min_month_dates * 2 and len(validation_dates) >= min_month_dates:
            splits.append(
                {
                    "name": f"cumulative_to_{validation_month}_holdout",
                    "train_month": f"{months[0]}..{months[months.index(validation_month) - 1]}",
                    "validation_month": validation_month,
                    "train_dates": train_dates,
                    "validation_dates": validation_dates,
                }
            )
    return splits


def score_signals(
    signals: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Snapshot],
    source_dates: Sequence[str],
    split_name: str,
    sample: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ordered_dates = list(source_dates)
    for signal in signals:
        for horizon in HORIZONS:
            rows.append(score_signal_horizon(signal, snapshots, ordered_dates, split_name, sample, horizon))
    return rows


def score_signal_horizon(
    signal: Mapping[str, Any],
    snapshots: Mapping[str, Snapshot],
    ordered_dates: Sequence[str],
    split_name: str,
    sample: str,
    horizon: int,
) -> Dict[str, Any]:
    signal_date = signal["date"]
    target_date = nth_future_date(ordered_dates, signal_date, horizon)
    base = {
        "split": split_name,
        "sample": sample,
        "horizon": f"{horizon}d",
        "signal_date": signal_date,
        "target_date": target_date or "",
        "ticker": signal["ticker"],
        "direction": signal["direction"],
        "pattern_family": signal["pattern_family"],
        "market_regime": signal.get("market_regime", ""),
        "sector": signal.get("sector", ""),
        "lead_option_symbol": signal.get("lead_option_symbol", ""),
        "strategy_kind": signal.get("strategy_kind", "long_option"),
        "strategy_type": signal.get("strategy_type", ""),
        "legs_json": signal.get("legs_json", ""),
        "entry_credit": signal.get("entry_credit"),
        "entry_ask": signal.get("entry_ask"),
        "entry_bid": signal.get("entry_bid"),
        "bid_ask_spread_pct": signal.get("bid_ask_spread_pct"),
        "blocked": bool(signal.get("block_reasons")),
        "block_reasons": ";".join(signal.get("block_reasons") or []),
        "status": "UNSCORABLE",
        "net_r": None,
        "win": 0,
        "outcome_note": "",
    }
    if not target_date:
        base["outcome_note"] = "not_enough_future_dates"
        return base
    if signal.get("strategy_kind") == "credit_spread":
        entry_credit = signal.get("entry_credit")
        max_risk = signal.get("max_risk_per_contract")
        if not entry_credit or entry_credit <= 0 or not max_risk or max_risk <= 0:
            base["outcome_note"] = "missing_entry_credit_or_risk"
            return base
        legs = parse_legs(signal.get("legs_json", ""))
        if len(legs) != 2:
            base["outcome_note"] = "missing_spread_legs"
            return base
        short_leg = next((leg for leg in legs if leg.get("action") == "SELL"), None)
        long_leg = next((leg for leg in legs if leg.get("action") == "BUY"), None)
        if not short_leg or not long_leg:
            base["outcome_note"] = "invalid_spread_legs"
            return base
        future_short = snapshots[target_date].option_quotes.get(short_leg["option_symbol"])
        future_long = snapshots[target_date].option_quotes.get(long_leg["option_symbol"])
        if future_short and future_long and future_short.get("ask", 0.0) > 0 and future_long.get("bid", 0.0) >= 0:
            exit_debit = max(0.0, future_short["ask"] - future_long["bid"])
            net_dollars = (entry_credit - exit_debit) * 100.0 - 2.60
            net_r = net_dollars / max_risk if max_risk > 0 else None
            base.update(
                {
                    "status": "SCORED",
                    "exit_debit": exit_debit,
                    "net_r": net_r,
                    "win": int(net_r is not None and net_r > 0),
                    "outcome_note": "credit_spread_exit_debit_after_fees",
                }
            )
            return base
        current_close = signal.get("close")
        future_close = snapshots[target_date].features.get(signal["ticker"], {}).get("close")
        if current_close and future_close:
            raw_move = (future_close - current_close) / current_close
            directional_move = raw_move if signal["direction"] == "bullish" else -raw_move
            base.update(
                {
                    "status": "PARTIAL",
                    "stock_proxy_move": directional_move,
                    "win": 0,
                    "outcome_note": "spread_quote_missing_stock_proxy_not_counted_as_win",
                }
            )
            return base
        base["outcome_note"] = "future_spread_leg_quotes_missing"
        return base
    entry = signal.get("entry_ask")
    if not entry or entry <= 0:
        base["outcome_note"] = "missing_entry_debit"
        return base
    symbol = signal.get("lead_option_symbol") or ""
    managed = score_managed_long_option(signal, snapshots, ordered_dates, horizon, entry, symbol)
    if managed:
        base.update(managed)
        return base
    future_quote = snapshots[target_date].option_quotes.get(symbol)
    if future_quote and future_quote.get("bid", 0.0) > 0:
        exit_value = future_quote["bid"]
        net_dollars = (exit_value - entry) * 100.0 - 1.30
        risk_dollars = entry * 100.0 + 0.65
        net_r = net_dollars / risk_dollars if risk_dollars > 0 else None
        base.update(
            {
                "status": "SCORED",
                "exit_bid": exit_value,
                "net_r": net_r,
                "win": int(net_r is not None and net_r > 0),
                "outcome_note": "option_bid_to_entry_ask_after_fees",
            }
        )
        return base
    if future_quote and (future_quote.get("mid") or future_quote.get("option_close")):
        exit_value = future_quote.get("mid") or future_quote.get("option_close")
        net_dollars = (exit_value - entry) * 100.0 - 1.30
        risk_dollars = entry * 100.0 + 0.65
        net_r = net_dollars / risk_dollars if risk_dollars > 0 else None
        base.update(
            {
                "status": "PARTIAL",
                "exit_proxy": exit_value,
                "net_r": net_r,
                "win": 0,
                "outcome_note": "option_mid_or_close_proxy_not_counted_as_win",
            }
        )
        return base
    current_close = signal.get("close")
    future_close = snapshots[target_date].features.get(signal["ticker"], {}).get("close")
    if current_close and future_close:
        raw_move = (future_close - current_close) / current_close
        directional_move = raw_move if signal["direction"] == "bullish" else -raw_move
        base.update(
            {
                "status": "PARTIAL",
                "stock_proxy_move": directional_move,
                "win": 0,
                "outcome_note": "stock_only_proxy_no_option_quote_history_not_counted_as_win",
            }
        )
        return base
    base["outcome_note"] = "future_option_and_stock_data_missing"
    return base


def score_managed_long_option(
    signal: Mapping[str, Any],
    snapshots: Mapping[str, Snapshot],
    ordered_dates: Sequence[str],
    horizon: int,
    entry: float,
    symbol: str,
) -> Optional[Dict[str, Any]]:
    if not symbol:
        return None
    signal_date = signal["date"]
    try:
        start_idx = ordered_dates.index(signal_date)
    except ValueError:
        return None
    target_price = entry * 2.0
    stop_price = entry * 0.50
    risk_dollars = entry * 100.0 + 0.65
    saw_real_quote = False
    last_bid: Optional[float] = None
    last_date = ""
    for idx in range(start_idx + 1, min(len(ordered_dates), start_idx + horizon + 1)):
        d = ordered_dates[idx]
        quote = snapshots[d].option_quotes.get(symbol)
        if not quote:
            continue
        high = quote.get("high")
        low = quote.get("low")
        bid = quote.get("bid")
        if bid is not None and bid > 0:
            saw_real_quote = True
            last_bid = bid
            last_date = d
        if high is None and low is None:
            continue
        saw_real_quote = True
        # Conservative same-day ordering: if target and stop are both inside the
        # daily range, assume the stop was hit first.
        if low is not None and low <= stop_price:
            net_dollars = (stop_price - entry) * 100.0 - 1.30
            return {
                "status": "SCORED",
                "managed_exit_date": d,
                "managed_exit_price": stop_price,
                "net_r": net_dollars / risk_dollars if risk_dollars > 0 else None,
                "win": 0,
                "outcome_note": "managed_long_option_stop_hit_conservative",
            }
        if high is not None and high >= target_price:
            net_dollars = (target_price - entry) * 100.0 - 1.30
            net_r = net_dollars / risk_dollars if risk_dollars > 0 else None
            return {
                "status": "SCORED",
                "managed_exit_date": d,
                "managed_exit_price": target_price,
                "net_r": net_r,
                "win": int(net_r is not None and net_r > 0),
                "outcome_note": "managed_long_option_target_hit",
            }
    if saw_real_quote and last_bid is not None:
        net_dollars = (last_bid - entry) * 100.0 - 1.30
        net_r = net_dollars / risk_dollars if risk_dollars > 0 else None
        return {
            "status": "SCORED",
            "managed_exit_date": last_date,
            "managed_exit_price": last_bid,
            "exit_bid": last_bid,
            "net_r": net_r,
            "win": int(net_r is not None and net_r > 0),
            "outcome_note": "managed_long_option_horizon_bid_exit",
        }
    return None


def nth_future_date(ordered_dates: Sequence[str], signal_date: str, horizon: int) -> Optional[str]:
    try:
        idx = ordered_dates.index(signal_date)
    except ValueError:
        return None
    target_idx = idx + horizon
    if target_idx >= len(ordered_dates):
        return None
    return ordered_dates[target_idx]


def select_validation_gate_outcomes(outcomes: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Use each validation month once for daily gates and backtest summaries."""
    validation_5d = [
        o
        for o in outcomes
        if o.get("sample") == "VALIDATION" and o.get("horizon") == "5d"
    ]
    cumulative = [
        o
        for o in validation_5d
        if str(o.get("split") or "").startswith("cumulative_to_")
    ]
    return cumulative or validation_5d


def summarize_outcomes(outcomes: Sequence[Mapping[str, Any]], sample: str) -> List[Dict[str, Any]]:
    primary = [o for o in outcomes if o.get("sample") == sample and o.get("horizon") == "5d"]
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for o in primary:
        grouped[(str(o.get("split")), str(o.get("pattern_family")))].append(o)
    rows: List[Dict[str, Any]] = []
    for (split, family), items in sorted(grouped.items()):
        scored = [o for o in items if o.get("status") == "SCORED" and o.get("net_r") is not None]
        partial = [o for o in items if o.get("status") == "PARTIAL"]
        unscorable = [o for o in items if o.get("status") == "UNSCORABLE"]
        net_rs = [float(o["net_r"]) for o in scored]
        positives = sum(r for r in net_rs if r > 0)
        negatives = abs(sum(r for r in net_rs if r < 0))
        win_count = sum(1 for r in net_rs if r > 0)
        block_counter = Counter()
        spreads: List[float] = []
        for o in items:
            for reason in str(o.get("block_reasons") or "").split(";"):
                if reason:
                    block_counter[reason] += 1
            if o.get("bid_ask_spread_pct") is not None:
                spreads.append(float(o["bid_ask_spread_pct"]))
        rows.append(
            {
                "split": split,
                "sample": sample,
                "pattern_family": family,
                "signal_count": len(items),
                "scored_count": len(scored),
                "partial_count": len(partial),
                "unscorable_count": len(unscorable),
                "win_count_scored": win_count,
                "win_rate_scored": safe_div(win_count, len(scored)),
                "win_rate_all_counting_unscorable_losses": safe_div(win_count, len(items)),
                "average_net_r": statistics.fmean(net_rs) if net_rs else None,
                "median_net_r": statistics.median(net_rs) if net_rs else None,
                "profit_factor": positives / negatives if negatives > 0 else (None if positives == 0 else 999.0),
                "worst_losing_streak": worst_losing_streak(net_rs),
                "drawdown_proxy_r": drawdown_proxy(net_rs),
                "tradeable_with_real_quotes_pct": safe_div(len(scored), len(items)),
                "average_bid_ask_spread": statistics.fmean(spreads) if spreads else None,
                "blocked_pct": safe_div(sum(1 for o in items if o.get("blocked")), len(items)),
                "top_block_reasons": "; ".join(f"{k}:{v}" for k, v in block_counter.most_common(5)),
            }
        )
    return rows


def summarize_baselines(baseline_outcomes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    primary = [o for o in baseline_outcomes if o.get("horizon") == "5d"]
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for o in primary:
        grouped[str(o.get("pattern_family"))].append(o)
    rows: List[Dict[str, Any]] = []
    for name, items in sorted(grouped.items()):
        scored = [o for o in items if o.get("status") == "SCORED" and o.get("net_r") is not None]
        net_rs = [float(o["net_r"]) for o in scored]
        rows.append(
            {
                "baseline": name,
                "seed": DEFAULT_SEED if "RANDOM" in name else "",
                "signal_count": len(items),
                "scored_count": len(scored),
                "win_rate_scored": safe_div(sum(1 for r in net_rs if r > 0), len(scored)),
                "average_net_r": statistics.fmean(net_rs) if net_rs else None,
                "median_net_r": statistics.median(net_rs) if net_rs else None,
                "note": baseline_note(name),
            }
        )
    return rows


def assign_family_tiers(
    validation_scorecard: Sequence[Mapping[str, Any]],
    baseline_comparison: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in validation_scorecard:
        by_family[str(row["pattern_family"])].append(row)
    baseline_avgs = [
        float(b["average_net_r"])
        for b in baseline_comparison
        if b.get("average_net_r") is not None and int(b.get("scored_count") or 0) > 0
    ]
    tiers: Dict[str, Dict[str, Any]] = {}
    for family, rows in by_family.items():
        scored = sum(int(r.get("scored_count") or 0) for r in rows)
        signals = sum(int(r.get("signal_count") or 0) for r in rows)
        wins = sum(win_count_from_scorecard_row(r) for r in rows)
        success_probability = safe_div(wins, scored)
        failure_probability = 1.0 - success_probability if success_probability is not None else None
        probability_score = wilson_lower_bound(wins, scored)
        weighted = [
            (float(r["average_net_r"]), int(r.get("scored_count") or 0))
            for r in rows
            if r.get("average_net_r") is not None and int(r.get("scored_count") or 0) > 0
        ]
        avg_r = safe_div(sum(avg * n for avg, n in weighted), sum(n for _, n in weighted))
        split_avgs = [
            float(r["average_net_r"])
            for r in rows
            if r.get("average_net_r") is not None and int(r.get("scored_count") or 0) > 0
        ]
        positive_splits = sum(1 for avg in split_avgs if avg > 0)
        split_count = len(split_avgs)
        worst_split_avg = min(split_avgs) if split_avgs else None
        max_losing_streak = max((int(r.get("worst_losing_streak") or 0) for r in rows), default=0)
        profit_factors = [
            float(r["profit_factor"])
            for r in rows
            if r.get("profit_factor") not in (None, "") and float(r["profit_factor"]) < 999
        ]
        pf = statistics.fmean(profit_factors) if profit_factors else None
        beats = sum(1 for b in baseline_avgs if avg_r is not None and avg_r > b)
        split_consistent = (
            split_count >= 2
            and positive_splits >= math.ceil(split_count * 0.75)
            and (worst_split_avg is None or worst_split_avg > -0.05)
        )
        drawdown_ok = max_losing_streak <= 8
        if (
            scored >= 50
            and avg_r is not None
            and avg_r > 0
            and (pf or 0) >= 1.2
            and beats >= 2
            and split_consistent
            and drawdown_ok
        ):
            tier = "PROVEN"
            note = "Positive out-of-sample expectancy with enough scored samples, baseline edge, and split consistency."
        elif scored >= 10 and avg_r is not None and avg_r > 0 and beats >= 1:
            tier = "PROMISING"
            note = "Positive pooled validation evidence, but split consistency, drawdown, or sample quality is not production-grade."
        elif signals == 0:
            tier = "BLOCKED"
            note = "No validation signals."
        else:
            tier = "RESEARCH_ONLY"
            note = "Insufficient positive out-of-sample evidence; do not treat as actionable."
        tiers[family] = {
            "pattern_family": family,
            "confidence_tier": tier,
            "validation_signal_count": signals,
            "validation_scored_count": scored,
            "validation_win_count": wins,
            "validation_success_probability": success_probability,
            "validation_failure_probability": failure_probability,
            "validation_probability_score": probability_score,
            "validation_average_net_r": avg_r,
            "validation_profit_factor": pf,
            "beats_baselines_count": beats,
            "validation_split_count": split_count,
            "positive_validation_splits": positive_splits,
            "worst_split_average_net_r": worst_split_avg,
            "max_worst_losing_streak": max_losing_streak,
            "probability_evidence": probability_evidence(wins, scored, success_probability, probability_score),
            "validation_note": note,
        }
    return tiers


def win_count_from_scorecard_row(row: Mapping[str, Any]) -> int:
    explicit = num(row.get("win_count_scored"))
    if explicit is not None:
        return int(round(explicit))
    scored = int(row.get("scored_count") or 0)
    win_rate = num(row.get("win_rate_scored"))
    if scored <= 0 or win_rate is None:
        return 0
    return int(round(scored * win_rate))


def wilson_lower_bound(wins: int, total: int, z: float = 1.0) -> Optional[float]:
    if total <= 0:
        return None
    phat = wins / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return max(0.0, min(1.0, (centre - margin) / denom))


def probability_evidence(
    wins: int,
    scored: int,
    success_probability: Optional[float],
    probability_score: Optional[float],
) -> str:
    if scored <= 0 or success_probability is None:
        return "No scored out-of-sample option outcomes."
    return (
        f"5d OOS scored wins {wins}/{scored}; empirical success {fmt_pct(success_probability)}; "
        f"sample-adjusted probability score {fmt_pct(probability_score)}."
    )


def summarize_regime_sector(outcomes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for dimension in ("market_regime", "sector", "ticker"):
        grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = defaultdict(list)
        for o in outcomes:
            if o.get("sample") != "VALIDATION" or o.get("horizon") != "5d":
                continue
            grouped[(str(o.get("pattern_family")), dimension, str(o.get(dimension) or ""))].append(o)
        for (family, dim, value), items in sorted(grouped.items()):
            if len(items) < 3 and dim != "market_regime":
                continue
            scored = [o for o in items if o.get("status") == "SCORED" and o.get("net_r") is not None]
            net_rs = [float(o["net_r"]) for o in scored]
            rows.append(
                {
                    "pattern_family": family,
                    "dimension": dim,
                    "value": value,
                    "signal_count": len(items),
                    "scored_count": len(scored),
                    "average_net_r": statistics.fmean(net_rs) if net_rs else None,
                    "win_rate_scored": safe_div(sum(1 for r in net_rs if r > 0), len(scored)),
                }
            )
    return rows


def generate_baseline_signals(
    validation_signals: Sequence[Mapping[str, Any]],
    validation_snaps: Sequence[Snapshot],
    pattern_config: Mapping[str, Any],
    top_candidates_per_day: int,
    seed: int,
    split_name: str,
) -> List[Dict[str, Any]]:
    signals: List[Dict[str, Any]] = []
    by_date_real = Counter(str(s["date"]) for s in validation_signals)
    for snap in validation_snaps:
        per_date_target = max(3, min(top_candidates_per_day, by_date_real.get(snap.signal_date, 5)))
        signals.extend(unusual_volume_baseline(snap, pattern_config, per_date_target))
        signals.extend(uw_flow_only_baseline(snap, pattern_config, per_date_target))
        signals.extend(naive_catalyst_only_baseline(snap, pattern_config, per_date_target))
        signals.extend(price_momentum_baseline(snap, pattern_config, per_date_target))
        signals.extend(index_directional_baseline(snap, pattern_config))
        signals.extend(random_same_liquidity_baseline(snap, validation_signals, pattern_config, per_date_target, seed, split_name))
    return signals


def unusual_volume_baseline(snap: Snapshot, pattern_config: Mapping[str, Any], limit: int) -> List[Dict[str, Any]]:
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for f in snap.features.values():
        direction = "bullish" if (f.get("call_volume_ratio_30d") or 0) >= (f.get("put_volume_ratio_30d") or 0) else "bearish"
        ratio = max(f.get("call_volume_ratio_30d") or 0.0, f.get("put_volume_ratio_30d") or 0.0)
        quote = snap.best_options.get((f["ticker"], direction))
        if quote:
            sig = build_signal(snap, f, "BASELINE_UNUSUAL_VOLUME_ONLY", direction, ratio, ["highest options volume ratio"], quote, pattern_config)
            rows.append((ratio, sig))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows[:limit]]


def uw_flow_only_baseline(snap: Snapshot, pattern_config: Mapping[str, Any], limit: int) -> List[Dict[str, Any]]:
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for f in snap.features.values():
        flow_total = f.get("flow_total_premium") or 0.0
        if flow_total <= 0:
            continue
        bias = f.get("flow_premium_bias") or f.get("premium_bias") or 0.0
        direction = "bullish" if bias >= 0 else "bearish"
        quote = snap.best_options.get((f["ticker"], direction))
        if quote:
            sig = build_signal(snap, f, "BASELINE_NAIVE_UW_FLOW_ONLY", direction, flow_total, ["naive UW flow premium only"], quote, pattern_config)
            rows.append((flow_total, sig))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows[:limit]]


def naive_catalyst_only_baseline(snap: Snapshot, pattern_config: Mapping[str, Any], limit: int) -> List[Dict[str, Any]]:
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for f in snap.features.values():
        earnings_dte = f.get("earnings_dte")
        premium = max(f.get("hot_total_premium") or 0.0, f.get("flow_total_premium") or 0.0)
        if earnings_dte is None and premium <= 0:
            continue
        catalyst_score = premium
        if earnings_dte is not None and 0 <= earnings_dte <= 7:
            catalyst_score += 500_000.0 / (1.0 + earnings_dte)
        if catalyst_score <= 0:
            continue
        direction = "bullish" if (f.get("flow_premium_bias") or f.get("premium_bias") or 0.0) >= 0 else "bearish"
        quote = snap.best_options.get((f["ticker"], direction))
        if quote:
            sig = build_signal(snap, f, "BASELINE_NAIVE_CATALYST_ONLY", direction, catalyst_score, ["naive catalyst/event proximity only"], quote, pattern_config)
            rows.append((catalyst_score, sig))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows[:limit]]


def price_momentum_baseline(snap: Snapshot, pattern_config: Mapping[str, Any], limit: int) -> List[Dict[str, Any]]:
    rows: List[Tuple[float, Dict[str, Any]]] = []
    for f in snap.features.values():
        ret = f.get("stock_return_1d")
        if ret is None:
            continue
        direction = "bullish" if ret >= 0 else "bearish"
        quote = snap.best_options.get((f["ticker"], direction))
        if quote:
            sig = build_signal(snap, f, "BASELINE_SIMPLE_PRICE_MOMENTUM", direction, abs(ret), ["same-day price momentum"], quote, pattern_config)
            rows.append((abs(ret), sig))
    rows.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in rows[:limit]]


def index_directional_baseline(snap: Snapshot, pattern_config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ticker in ("SPY", "QQQ"):
        f = snap.features.get(ticker)
        if not f:
            continue
        ret = f.get("stock_return_1d") or 0.0
        direction = "bullish" if ret >= 0 else "bearish"
        quote = snap.best_options.get((ticker, direction))
        if quote:
            rows.append(
                build_signal(
                    snap,
                    f,
                    "BASELINE_SPY_QQQ_DIRECTIONAL",
                    direction,
                    abs(ret),
                    ["index direction baseline"],
                    quote,
                    pattern_config,
                )
            )
    return rows


def random_same_liquidity_baseline(
    snap: Snapshot,
    validation_signals: Sequence[Mapping[str, Any]],
    pattern_config: Mapping[str, Any],
    limit: int,
    seed: int,
    split_name: str,
) -> List[Dict[str, Any]]:
    universe: List[Tuple[float, Dict[str, Any], str]] = []
    for f in snap.features.values():
        for direction in ("bullish", "bearish"):
            quote = snap.best_options.get((f["ticker"], direction))
            if quote:
                universe.append((float(f.get("liquidity_score") or 0.0), f, direction))
    if not universe:
        return []
    same_date_real = [s for s in validation_signals if s["date"] == snap.signal_date]
    rng = random.Random(stable_seed(f"{seed}:{split_name}:{snap.signal_date}:random_liquidity"))
    rows: List[Dict[str, Any]] = []
    if same_date_real:
        sorted_universe = sorted(universe, key=lambda x: x[0])
        for sig in same_date_real[:limit]:
            target_liq = float(sig.get("pattern_score") or 0.0)
            direction = sig.get("direction", "bullish")
            matches = [u for u in sorted_universe if u[2] == direction] or sorted_universe
            idx = min(range(len(matches)), key=lambda i: abs(matches[i][0] - target_liq))
            lo = max(0, idx - 5)
            hi = min(len(matches), idx + 6)
            _, f, chosen_direction = rng.choice(matches[lo:hi])
            quote = snap.best_options.get((f["ticker"], chosen_direction))
            if quote:
                rows.append(
                    build_signal(
                        snap,
                        f,
                        "BASELINE_RANDOM_SAME_DATE_LIQUIDITY",
                        chosen_direction,
                        float(f.get("liquidity_score") or 0.0),
                        ["deterministic random same-date liquidity baseline"],
                        quote,
                        pattern_config,
                    )
                )
    else:
        for _, f, direction in rng.sample(universe, min(limit, len(universe))):
            quote = snap.best_options.get((f["ticker"], direction))
            if quote:
                rows.append(
                    build_signal(
                        snap,
                        f,
                        "BASELINE_RANDOM_SAME_DATE_LIQUIDITY",
                        direction,
                        float(f.get("liquidity_score") or 0.0),
                        ["deterministic random same-date liquidity baseline"],
                        quote,
                        pattern_config,
                    )
                )
    return rows[:limit]


def run_missed_mover_audit(
    snapshots: Mapping[str, Snapshot],
    source_dates: Sequence[str],
    as_of: str,
    per_date: int = 5,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    usable_dates = [d for d in source_dates if d < as_of]
    for d in usable_dates:
        next_d = nth_future_date(source_dates, d, 1)
        if not next_d:
            continue
        current = snapshots[d]
        future = snapshots[next_d]
        movers: List[Tuple[float, Dict[str, Any], float]] = []
        for ticker, f in current.features.items():
            close = f.get("close")
            future_close = future.features.get(ticker, {}).get("close")
            if not close or not future_close:
                continue
            move = (future_close - close) / close
            movers.append((abs(move), f, move))
        movers.sort(key=lambda x: x[0], reverse=True)
        cfg = learn_pattern_config([current])
        signals = generate_signals_for_snapshot(current, cfg, max_signals=50)
        signal_tickers = {s["ticker"] for s in signals}
        for abs_move, f, move in movers[:per_date]:
            ticker = f["ticker"]
            rows.append(
                {
                    "signal_date": d,
                    "next_date": next_d,
                    "ticker": ticker,
                    "next_day_stock_move": move,
                    "abs_next_day_stock_move": abs_move,
                    "sector": f.get("sector") or "",
                    "was_flagged_by_new_pipeline": ticker in signal_tickers,
                    "likely_miss_reason": missed_reason(f, ticker in signal_tickers),
                    "call_volume_ratio_30d": f.get("call_volume_ratio_30d"),
                    "put_volume_ratio_30d": f.get("put_volume_ratio_30d"),
                    "premium_bias": f.get("premium_bias"),
                    "hot_total_premium": f.get("hot_total_premium"),
                }
            )
    rows.sort(key=lambda r: (r["signal_date"], r["abs_next_day_stock_move"]), reverse=True)
    return rows[:250]


def missed_reason(f: Mapping[str, Any], flagged: bool) -> str:
    if flagged:
        return "flagged_by_pattern_pipeline"
    reasons = []
    if not f.get("hot_total_premium"):
        reasons.append("no_hot_chain_premium")
    if max(f.get("call_volume_ratio_30d") or 0, f.get("put_volume_ratio_30d") or 0) < 1.25:
        reasons.append("no_unusual_volume_expansion")
    if abs(f.get("premium_bias") or 0) < 0.03:
        reasons.append("weak_premium_bias")
    if f.get("avg_spread_pct") is None:
        reasons.append("missing_quote_spread")
    return ";".join(reasons) if reasons else "moved_without_matching_frozen_pattern"


def build_sentiment_news_summary(date_dir: Path, as_of: str) -> Dict[str, Any]:
    browser_dir = date_dir / "browser_text"
    used: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    keyword_counts: Counter[str] = Counter()
    keywords = (
        "fed",
        "cpi",
        "jobs",
        "rates",
        "yield",
        "dollar",
        "oil",
        "war",
        "tariff",
        "sanction",
        "earnings",
        "guidance",
        "analyst",
        "regulatory",
        "election",
        "credit",
        "vix",
    )
    if not browser_dir.exists():
        return {
            "used_sources": [],
            "skipped_sources": [],
            "summary": "No local browser/news/social captures found for this date.",
        }
    for path in sorted(browser_dir.glob("*.txt")):
        capture_date = extract_date_from_name(path.name) or ""
        if capture_date and capture_date > as_of:
            skipped.append(
                {
                    "path": str(path.resolve()),
                    "reason": "capture_timestamp_after_signal_date",
                    "capture_date": capture_date,
                }
            )
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[:20_000]
        lowered = text.lower()
        for keyword in keywords:
            keyword_counts[keyword] += lowered.count(keyword)
        urls = sorted(set(re.findall(r"https?://[^\s)>\"]+", text)))[:5]
        used.append(
            {
                "path": str(path.resolve()),
                "capture_date": capture_date,
                "urls": urls,
                "bytes_read": min(len(text), 20_000),
            }
        )
    if used:
        top = ", ".join(f"{k}:{v}" for k, v in keyword_counts.most_common(8) if v)
        summary = f"Point-in-time local captures used. Dominant keyword counts: {top or 'none'}."
    else:
        summary = "No browser/news/social captures passed point-in-time timestamp checks."
    return {"used_sources": used, "skipped_sources": skipped, "keyword_counts": dict(keyword_counts), "summary": summary}


def prepare_decision_rows(
    daily_rows: Sequence[Mapping[str, Any]],
    validation_bundle: Mapping[str, Any],
    source_completeness: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    risk_config = dict(config.get("risk_config") or DEFAULT_RISK_CONFIG)
    family_stats = build_family_outcome_stats(validation_bundle.get("outcomes", []))
    regime_edge_stats = build_regime_edge_stats(validation_bundle.get("outcomes", []))
    calibration_metrics = build_calibration_metrics(validation_bundle)
    run_kill_switches = build_run_kill_switches(
        validation_bundle=validation_bundle,
        source_completeness=source_completeness,
        risk_config=risk_config,
        calibration_metrics=calibration_metrics,
    )
    family_tiers = validation_bundle.get("family_tiers", {})
    enriched = [
        enrich_decision_row(
            row,
            family_tiers.get(str(row.get("pattern_family") or ""), {}),
            family_stats.get(str(row.get("pattern_family") or ""), {}),
            regime_edge_stats.get(
                (
                    str(row.get("pattern_family") or ""),
                    str(row.get("current_market_alignment") or row.get("market_regime") or "UNKNOWN"),
                ),
                {},
            ),
            risk_config,
            run_kill_switches,
        )
        for row in daily_rows
    ]
    return enriched, {
        "risk_config": risk_config,
        "family_stats": family_stats,
        "regime_edge_stats": regime_edge_stats,
        "calibration_metrics": calibration_metrics,
        "run_kill_switches": run_kill_switches,
    }


def build_family_outcome_stats(outcomes: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for outcome in select_validation_gate_outcomes(outcomes):
        grouped[str(outcome.get("pattern_family") or "")].append(outcome)
    stats: Dict[str, Dict[str, Any]] = {}
    for family, rows in grouped.items():
        scored = [r for r in rows if r.get("status") == "SCORED" and r.get("net_r") is not None]
        net_rs = [float(r["net_r"]) for r in scored]
        wins = [r for r in net_rs if r > 0]
        losses = [r for r in net_rs if r <= 0]
        positives = sum(wins)
        negatives = abs(sum(losses))
        stats[family] = {
            "pattern_family": family,
            "signal_count": len(rows),
            "scored_count": len(scored),
            "partial_count": sum(1 for r in rows if r.get("status") == "PARTIAL"),
            "unscorable_count": sum(1 for r in rows if r.get("status") == "UNSCORABLE"),
            "win_count": len(wins),
            "win_rate": safe_div(len(wins), len(scored)),
            "avg_r": statistics.fmean(net_rs) if net_rs else None,
            "median_r": statistics.median(net_rs) if net_rs else None,
            "avg_win_r": statistics.fmean(wins) if wins else None,
            "avg_loss_r": statistics.fmean(losses) if losses else None,
            "payoff_ratio": (statistics.fmean(wins) / abs(statistics.fmean(losses))) if wins and losses else None,
            "profit_factor": positives / negatives if negatives > 0 else (None if positives == 0 else 999.0),
            "drawdown_proxy_r": drawdown_proxy(net_rs),
            "worst_losing_streak": worst_losing_streak(net_rs),
            "quote_coverage": safe_div(len(scored), len(rows)),
        }
    return stats


def build_regime_edge_stats(outcomes: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for outcome in select_validation_gate_outcomes(outcomes):
        key = (str(outcome.get("pattern_family") or ""), str(outcome.get("market_regime") or "UNKNOWN"))
        grouped[key].append(outcome)
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, rows in grouped.items():
        scored = [r for r in rows if r.get("status") == "SCORED" and r.get("net_r") is not None]
        net_rs = [float(r["net_r"]) for r in scored]
        positives = sum(r for r in net_rs if r > 0)
        negatives = abs(sum(r for r in net_rs if r < 0))
        stats[key] = {
            "signal_count": len(rows),
            "scored_count": len(scored),
            "win_rate": safe_div(sum(1 for r in net_rs if r > 0), len(scored)),
            "avg_r": statistics.fmean(net_rs) if net_rs else None,
            "profit_factor": positives / negatives if negatives > 0 else (None if positives == 0 else 999.0),
            "quote_coverage": safe_div(len(scored), len(rows)),
        }
    return stats


def build_calibration_metrics(validation_bundle: Mapping[str, Any]) -> Dict[str, Any]:
    family_tiers = validation_bundle.get("family_tiers", {})
    buckets: Dict[str, Dict[str, Any]] = {}
    scored_total = 0
    brier_terms: List[float] = []
    for outcome in select_validation_gate_outcomes(validation_bundle.get("outcomes", [])):
        if outcome.get("status") != "SCORED" or outcome.get("win") in (None, ""):
            continue
        tier = family_tiers.get(str(outcome.get("pattern_family") or ""), {})
        p = num(tier.get("validation_probability_score"))
        if p is None:
            p = num(tier.get("validation_success_probability"))
        if p is None:
            continue
        p = clamp(p, 0.0, 1.0)
        win = 1.0 if int(outcome.get("win") or 0) else 0.0
        brier_terms.append((p - win) ** 2)
        scored_total += 1
        lo = int(math.floor(p * 10.0)) * 10
        hi = min(lo + 10, 100)
        key = f"{lo:02d}-{hi:02d}%"
        bucket = buckets.setdefault(key, {"bucket": key, "count": 0, "predicted_sum": 0.0, "wins": 0})
        bucket["count"] += 1
        bucket["predicted_sum"] += p
        bucket["wins"] += int(win)
    rows: List[Dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        rows.append(
            {
                "bucket": key,
                "count": bucket["count"],
                "avg_predicted_probability": safe_div(bucket["predicted_sum"], bucket["count"]),
                "observed_win_rate": safe_div(bucket["wins"], bucket["count"]),
                "wins": bucket["wins"],
            }
        )
    brier = statistics.fmean(brier_terms) if brier_terms else None
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "scored_prediction_count": scored_total,
        "brier_score": brier,
        "reliability_buckets": rows,
        "status": "AVAILABLE" if brier is not None else "NOT_AVAILABLE",
    }


def build_run_kill_switches(
    validation_bundle: Mapping[str, Any],
    source_completeness: Mapping[str, Any],
    risk_config: Mapping[str, Any],
    calibration_metrics: Mapping[str, Any],
) -> List[str]:
    switches: List[str] = []
    kill_cfg = risk_config.get("kill_switches") or {}
    if kill_cfg.get("source_incomplete", True) and not source_completeness.get("source_complete"):
        switches.append("KILL_SWITCH_SOURCE_INCOMPLETE")
    scorecard = validation_bundle.get("validation_gate_scorecard") or validation_bundle.get("validation_scorecard", [])
    signals = sum(int(r.get("signal_count") or 0) for r in scorecard)
    scored = sum(int(r.get("scored_count") or 0) for r in scorecard)
    unscorable = sum(int(r.get("unscorable_count") or 0) for r in scorecard)
    quote_coverage = safe_div(scored, signals)
    unscorable_rate = safe_div(unscorable, signals)
    if (
        kill_cfg.get("low_quote_coverage", True)
        and signals
        and quote_coverage is not None
        and quote_coverage < float(risk_config.get("min_quote_coverage", 0.25))
    ):
        switches.append("KILL_SWITCH_LOW_QUOTE_COVERAGE")
    if (
        kill_cfg.get("high_unscorable_rate", True)
        and signals
        and unscorable_rate is not None
        and unscorable_rate > float(risk_config.get("max_unscorable_rate", 0.75))
    ):
        switches.append("KILL_SWITCH_HIGH_UNSCORABLE_RATE")
    drawdowns = [float(r["drawdown_proxy_r"]) for r in scorecard if r.get("drawdown_proxy_r") not in (None, "")]
    if (
        kill_cfg.get("validation_drawdown_breached", True)
        and drawdowns
        and min(drawdowns) < float(risk_config.get("max_validation_drawdown_r", -12.0))
    ):
        switches.append("KILL_SWITCH_VALIDATION_DRAWDOWN_BREACHED")
    brier = num(calibration_metrics.get("brier_score"))
    if (
        kill_cfg.get("calibration_fails", True)
        and brier is not None
        and brier > float(risk_config.get("max_calibration_brier", 0.35))
    ):
        switches.append("KILL_SWITCH_CALIBRATION_FAILS")
    return sorted(set(switches))


def hard_review_blockers(blockers: Iterable[str]) -> set[str]:
    blocker_set = set(blockers)
    hard_kill_switches = {
        "KILL_SWITCH_SOURCE_INCOMPLETE",
        "KILL_SWITCH_ARTIFACT_SCHEMA_VALIDATION_FAILS",
    }
    return blocker_set & (
        HARD_TRADE_BLOCKERS
        | RISK_TRADE_BLOCKERS
        | QUOTE_TRADE_BLOCKERS
        | hard_kill_switches
        | {
            "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS",
            "EXPECTED_R_PER_DAY_NOT_POSITIVE",
            "VALIDATION_EXPECTANCY_NEGATIVE",
        }
    )


def qualifies_for_validated_edge_review(
    blockers: Iterable[str],
    has_ticket: bool,
    family_stats: Mapping[str, Any],
    regime_edge_stats: Mapping[str, Any],
    risk_config: Mapping[str, Any],
    current_regime: str,
    baselines_beaten: int,
) -> Tuple[bool, str, str]:
    blocker_set = set(blockers)
    if not has_ticket or hard_review_blockers(blocker_set):
        return False, "", ""

    scored = int(num(family_stats.get("scored_count")) or 0)
    avg_r = num(family_stats.get("avg_r"))
    pf = num(family_stats.get("profit_factor"))
    quote_coverage = num(family_stats.get("quote_coverage"))
    if (
        scored < int(risk_config.get("min_edge_review_scored_outcomes", 30))
        or avg_r is None
        or avg_r <= float(risk_config.get("min_edge_review_expected_r", 0.05))
        or pf is None
        or pf < float(risk_config.get("min_edge_review_profit_factor", 1.30))
        or quote_coverage is None
        or quote_coverage < float(risk_config.get("min_edge_review_quote_coverage", 0.50))
        or baselines_beaten < int(risk_config.get("min_edge_review_baselines_beaten", 2))
    ):
        return False, "", ""

    family_text = (
        f"family scored={scored}, avg_R={fmt_num(avg_r)}, "
        f"PF={fmt_num(pf)}, quote_coverage={fmt_pct(quote_coverage)}"
    )
    if "MARKET_REGIME_CONFLICT" not in blocker_set:
        return True, "VALIDATED_FAMILY_EDGE_REVIEW", family_text

    regime_scored = int(num(regime_edge_stats.get("scored_count")) or 0)
    regime_avg_r = num(regime_edge_stats.get("avg_r"))
    regime_pf = num(regime_edge_stats.get("profit_factor"))
    if (
        regime_scored >= int(risk_config.get("min_regime_edge_review_scored_outcomes", 20))
        and regime_avg_r is not None
        and regime_avg_r > float(risk_config.get("min_regime_edge_review_expected_r", 0.05))
        and regime_pf is not None
        and regime_pf >= float(risk_config.get("min_regime_edge_review_profit_factor", 1.20))
    ):
        regime_text = (
            f"{current_regime or 'UNKNOWN'} regime scored={regime_scored}, "
            f"avg_R={fmt_num(regime_avg_r)}, PF={fmt_num(regime_pf)}"
        )
        return True, "VALIDATED_FAMILY_AND_REGIME_EDGE_REVIEW", f"{family_text}; {regime_text}"

    return False, "", family_text


def soft_review_candidate(row: Mapping[str, Any]) -> bool:
    tier = str(row.get("confidence_tier") or "")
    score = num(row.get("probability_score")) or 0.0
    success = num(row.get("success_probability_pct")) or 0.0
    return tier in {"PROMISING", "PROVEN"} or score >= 45.0 or success >= 55.0


def enrich_decision_row(
    row: Mapping[str, Any],
    tier_info: Mapping[str, Any],
    family_stats: Mapping[str, Any],
    regime_edge_stats: Mapping[str, Any],
    risk_config: Mapping[str, Any],
    run_kill_switches: Sequence[str],
) -> Dict[str, Any]:
    enriched = dict(row)
    blockers = set(enriched.get("block_reasons") or [])
    setup = trade_setup_fields(enriched)
    p = probability_decimal(enriched)
    if p is None:
        p = num(tier_info.get("validation_probability_score"))
    if p is None:
        p = num(tier_info.get("validation_success_probability"))
    calibrated_probability = clamp(p, 0.0, 1.0) if p is not None else None
    avg_win = num(family_stats.get("avg_win_r"))
    avg_loss = num(family_stats.get("avg_loss_r"))
    expected_r = None
    if calibrated_probability is not None and avg_win is not None and avg_loss is not None:
        expected_r = calibrated_probability * avg_win + (1.0 - calibrated_probability) * avg_loss
    if expected_r is None:
        expected_r = num(tier_info.get("validation_average_net_r")) or num(family_stats.get("avg_r"))
    expected_hold = int(num(enriched.get("expected_hold_days")) or int(risk_config.get("expected_hold_days", 5)))
    expected_r_per_day = safe_div(expected_r, max(expected_hold, 1))
    validation_profit_factor = num(tier_info.get("validation_profit_factor")) or num(family_stats.get("profit_factor"))
    validation_scored = int(num(tier_info.get("validation_scored_count")) or num(family_stats.get("scored_count")) or 0)
    baselines_beaten = int(num(tier_info.get("beats_baselines_count")) or 0)
    probability_score = probability_decimal_from_pct(enriched.get("probability_score"))
    confidence_lower = num(tier_info.get("validation_probability_score"))
    confidence_upper = wilson_upper_bound(
        int(num(tier_info.get("validation_win_count")) or num(family_stats.get("win_count")) or 0),
        validation_scored,
    )
    capacity = estimate_capacity_contracts(enriched, risk_config)
    quote_validation_status = live_quote_validation_status(enriched, risk_config)
    fill = fill_cost_fields(enriched, risk_config)
    max_risk = num(enriched.get("max_risk_per_contract"))
    ticker = str(enriched.get("ticker") or "")

    if expected_r is None or expected_r <= float(risk_config.get("min_expected_r", 0.0)):
        blockers.add("EXPECTED_R_NOT_POSITIVE_AFTER_COSTS")
    if expected_r_per_day is None or expected_r_per_day <= float(risk_config.get("min_expected_r_per_day", 0.0)):
        blockers.add("EXPECTED_R_PER_DAY_NOT_POSITIVE")
    if validation_profit_factor is None or validation_profit_factor < float(risk_config.get("min_profit_factor", 1.2)):
        blockers.add("PROFIT_FACTOR_BELOW_AUTO_APPROVAL")
    if validation_scored < int(risk_config.get("min_oos_scored_outcomes", 30)):
        blockers.add("LIMITED_OUT_OF_SAMPLE_SAMPLE")
    if baselines_beaten < int(risk_config.get("min_baselines_beaten", 2)):
        blockers.add("DOES_NOT_BEAT_TWO_BASELINES")
    if probability_score is None or probability_score < float(risk_config.get("min_probability_score", 0.50)):
        blockers.add("CALIBRATION_SCORE_MISSING_OR_WEAK")
    if calibrated_probability is None or calibrated_probability < float(risk_config.get("min_calibrated_probability", 0.50)):
        blockers.add("CALIBRATION_SCORE_MISSING_OR_WEAK")
    if confidence_lower is None or confidence_lower < float(risk_config.get("min_confidence_lower_bound", 0.45)):
        blockers.add("CONFIDENCE_BAND_TOO_WEAK")
    if max_risk is not None and max_risk > float(risk_config.get("max_risk_per_trade", 1500.0)):
        blockers.add("MAX_RISK_EXCEEDS_PER_TRADE_LIMIT")
    if capacity < 1:
        blockers.add("CAPACITY_TOO_SMALL_FOR_INTENDED_SIZE")
    if ticker in set(risk_config.get("excluded_tickers") or []):
        blockers.add("EXCLUDED_TICKER_OR_INSTRUMENT")
    if ticker in set(risk_config.get("manual_only_tickers") or []):
        blockers.add("MANUAL_ONLY_TICKER_OR_INSTRUMENT")
    if ticker in set(risk_config.get("high_notional_index_tickers") or []) and max_risk and max_risk > float(risk_config.get("max_risk_per_trade", 1500.0)) * 0.5:
        blockers.add("HIGH_NOTIONAL_INDEX_GUARDRAIL")
    if quote_validation_status.startswith("FAILED"):
        blockers.add("LIVE_QUOTE_VALIDATION_FAILED")
    elif quote_validation_status.startswith("SKIPPED") and risk_config.get("live_quote_required_for_auto", False):
        blockers.add("LIVE_QUOTE_VALIDATION_SKIPPED")
    for switch in run_kill_switches:
        blockers.add(switch)

    has_ticket = "No complete" not in str(setup.get("trade_setup") or "")
    edge_review, edge_review_reason, edge_review_evidence = qualifies_for_validated_edge_review(
        blockers,
        has_ticket,
        family_stats,
        regime_edge_stats,
        risk_config,
        str(enriched.get("current_market_alignment") or enriched.get("market_regime") or "UNKNOWN"),
        baselines_beaten,
    )
    hard_or_reject = hard_review_blockers(blockers)
    if not blockers and enriched.get("classification") == "TRADE":
        status = "AUTO_APPROVED"
    elif edge_review:
        status = "TRADE_REVIEW"
    elif hard_or_reject:
        status = "AVOID"
    elif has_ticket and soft_review_candidate(enriched):
        status = "TRADE_REVIEW"
    else:
        status = "AVOID"
    normalized_classification = (
        "TRADE" if status == "AUTO_APPROVED" else "WATCH" if status == "TRADE_REVIEW" else "AVOID"
    )

    enriched.update(
        {
            "status": status,
            "classification": normalized_classification,
            "candidate_id": stable_candidate_id(enriched),
            "block_reasons": sorted(blockers),
            "expected_R": round(expected_r, 6) if expected_r is not None else None,
            "expected_R_per_day": round(expected_r_per_day, 6) if expected_r_per_day is not None else None,
            "avg_win_R": avg_win,
            "avg_loss_R": avg_loss,
            "payoff_ratio": family_stats.get("payoff_ratio"),
            "validation_profit_factor": validation_profit_factor,
            "validation_scored_count": validation_scored,
            "beats_baselines_count": baselines_beaten,
            "calibrated_probability": round(calibrated_probability, 6) if calibrated_probability is not None else None,
            "confidence_lower_bound": confidence_lower,
            "confidence_upper_bound": confidence_upper,
            "confidence_uncertainty_band": confidence_band_text(confidence_lower, confidence_upper),
            "capacity_estimate_contracts": capacity,
            "live_quote_validation_status": quote_validation_status,
            "fill_model": fill["fill_model"],
            "slippage_estimate": fill["slippage_estimate"],
            "fees_commissions": fill["fees_commissions"],
            "entry_debit_credit": setup.get("entry_range"),
            "full_ticket": setup.get("trade_setup"),
            "all_legs": setup.get("occ_symbols"),
            "expected_hold": f"{expected_hold} trading days",
            "invalidation": enriched.get("invalidation_rule") or "",
            "review_close_rule": enriched.get("review_close_rule") or "",
            "iv_volatility_context": volatility_context(enriched),
            "event_risk": event_risk_text(enriched),
            "regime_alignment": regime_alignment_text(enriched),
            "uw_evidence": uw_evidence_text(enriched),
            "kill_switch_triggered": ";".join(run_kill_switches),
            "edge_review_reason": edge_review_reason,
            "edge_review_evidence": edge_review_evidence,
        }
    )
    enriched["blocker_categories"] = decompose_blockers(enriched["block_reasons"])
    enriched["promotion_requirements"] = promotion_needed_text(enriched)
    return enriched


def probability_decimal(row: Mapping[str, Any]) -> Optional[float]:
    for key in ("trade_success_probability_pct", "success_probability_pct", "pattern_success_probability_pct"):
        parsed = probability_decimal_from_pct(row.get(key))
        if parsed is not None:
            return parsed
    return None


def probability_decimal_from_pct(value: Any) -> Optional[float]:
    parsed = num(value)
    if parsed is None:
        return None
    if parsed > 1.0:
        return clamp(parsed / 100.0, 0.0, 1.0)
    return clamp(parsed, 0.0, 1.0)


def wilson_upper_bound(wins: int, total: int, z: float = 1.0) -> Optional[float]:
    if total <= 0:
        return None
    phat = wins / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return max(0.0, min(1.0, (centre + margin) / denom))


def estimate_capacity_contracts(row: Mapping[str, Any], risk_config: Mapping[str, Any]) -> int:
    volume = max(num(row.get("liquidity_volume")) or 0.0, 0.0)
    oi = max(num(row.get("liquidity_open_interest")) or 0.0, 0.0)
    max_contracts = int(risk_config.get("max_contracts_per_trade", 10) or 10)
    if volume <= 0 and oi <= 0:
        return 0
    capacity = max(1, int(min(max_contracts, max(volume * 0.02, 0.0), max(oi * 0.01, 0.0) if oi else max_contracts)))
    return capacity


def live_quote_validation_status(row: Mapping[str, Any], risk_config: Mapping[str, Any]) -> str:
    bid = num(row.get("entry_bid"))
    ask = num(row.get("entry_ask"))
    spread_pct = num(row.get("bid_ask_spread_pct"))
    max_spread = (
        float(risk_config.get("max_credit_spread_spread_pct", 0.65))
        if row.get("strategy_kind") == "credit_spread"
        else float(risk_config.get("max_bid_ask_spread_pct", 0.35))
    )
    if ask is None and row.get("strategy_kind") != "credit_spread":
        return "FAILED_NO_COMPLETE_QUOTE"
    if spread_pct is None:
        return "FAILED_MISSING_SPREAD"
    if spread_pct > max_spread:
        return "FAILED_SPREAD_TOO_WIDE"
    source = str(row.get("quote_source") or "")
    if source in {"bot_eod", "hot_chains"} and risk_config.get("allow_conservative_historical_quote_for_auto", True):
        return "CONSERVATIVE_HISTORICAL_QUOTE_ASSUMPTION"
    if bid is not None or ask is not None:
        return "SKIPPED_NO_SCHWAB_LIVE_CHAIN_HISTORICAL_QUOTE_ONLY"
    return "FAILED_NO_COMPLETE_QUOTE"


def fill_cost_fields(row: Mapping[str, Any], risk_config: Mapping[str, Any]) -> Dict[str, Any]:
    spread = num(row.get("entry_ask")) - num(row.get("entry_bid")) if num(row.get("entry_ask")) is not None and num(row.get("entry_bid")) is not None else None
    slippage = None
    if spread is not None:
        slippage = max(0.0, spread) * 100.0 * float(risk_config.get("slippage_pct_of_spread", 0.5))
    fees = (
        float(risk_config.get("round_trip_spread_fees", 2.60))
        if row.get("strategy_kind") == "credit_spread"
        else float(risk_config.get("round_trip_long_option_fees", 1.30))
    )
    fill_model = "conservative_credit" if row.get("strategy_kind") == "credit_spread" else "conservative_debit_at_ask"
    return {"fill_model": fill_model, "slippage_estimate": slippage, "fees_commissions": fees}


def stable_candidate_id(row: Mapping[str, Any]) -> str:
    setup = trade_setup_fields(row)
    key = stable_json(
        {
            "date": row.get("date"),
            "ticker": row.get("ticker"),
            "direction": row.get("direction"),
            "ticket": setup.get("trade_setup"),
            "entry": setup.get("entry_range"),
            "family": row.get("pattern_family"),
        }
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def confidence_band_text(lower: Any, upper: Any) -> str:
    lo = num(lower)
    hi = num(upper)
    if lo is None and hi is None:
        return "n/a"
    return f"{fmt_pct(lo)} to {fmt_pct(hi)}"


def volatility_context(row: Mapping[str, Any]) -> str:
    iv = num(row.get("quote_iv")) or num(row.get("avg_iv"))
    if iv is None:
        return "IV unavailable in local quote history"
    if iv >= 0.75:
        label = "elevated IV / IV-crush risk"
    elif iv >= 0.40:
        label = "moderate-to-high IV"
    else:
        label = "normal/lower IV"
    return f"{label}: {fmt_pct(iv)}"


def event_risk_text(row: Mapping[str, Any]) -> str:
    dte = num(row.get("earnings_dte"))
    next_date = row.get("next_earnings_date") or ""
    if dte is None:
        return "No near-term earnings date in local source"
    if dte < 0:
        return f"Earnings date already passed ({next_date})"
    if dte <= 2:
        return f"Near-term earnings/event risk in {int(dte)} calendar days ({next_date})"
    return f"Earnings/event risk outside immediate window ({int(dte)} calendar days)"


def regime_alignment_text(row: Mapping[str, Any]) -> str:
    if "MARKET_REGIME_CONFLICT" in set(row.get("block_reasons") or []):
        return f"conflicts with {row.get('current_market_alignment') or row.get('market_regime') or 'UNKNOWN'}"
    return f"aligned/acceptable in {row.get('current_market_alignment') or row.get('market_regime') or 'UNKNOWN'}"


def uw_evidence_text(row: Mapping[str, Any]) -> str:
    return (
        f"{row.get('reason_summary') or 'pattern signal'}; "
        f"premium={fmt_num(row.get('hot_total_premium'))}; "
        f"flow_bias={fmt_num(row.get('flow_premium_bias'))}; "
        f"call_ratio={fmt_num(row.get('call_volume_ratio_30d'))}; "
        f"put_ratio={fmt_num(row.get('put_volume_ratio_30d'))}"
    )


def build_decision_board_rows(
    rows: Sequence[Mapping[str, Any]],
    as_of: str,
    source_complete: bool,
    daily_decision: str,
    artifact_paths: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    artifact_paths_text = stable_json(artifact_paths or {})
    board: List[Dict[str, Any]] = []
    for row in rows:
        setup = trade_setup_fields(row)
        board.append(
            {
                "run_date": as_of,
                "pipeline_version": PIPELINE_VERSION,
                "artifact_schema_version": DECISION_BOARD_SCHEMA_VERSION,
                "daily_trade_decision": daily_decision,
                "source_complete": bool(source_complete),
                "candidate_id": row.get("candidate_id") or stable_candidate_id(row),
                "status": row.get("status") or "TRADE_REVIEW",
                "ticker": row.get("ticker"),
                "direction": row.get("direction"),
                "strategy": setup.get("strategy"),
                "full_ticket": setup.get("trade_setup"),
                "all_legs": setup.get("occ_symbols"),
                "buy_sell": setup.get("buy_or_sell"),
                "call_put": setup.get("call_or_put"),
                "strikes": setup.get("strike_rates"),
                "expiration": setup.get("expiration_date"),
                "entry": setup.get("entry_range"),
                "entry_debit_credit": row.get("entry_debit_credit") or setup.get("entry_range"),
                "bid": row.get("entry_bid"),
                "ask": row.get("entry_ask"),
                "spread": row.get("bid_ask_spread_pct"),
                "fill_model": row.get("fill_model"),
                "slippage": row.get("slippage_estimate"),
                "fees_commissions": row.get("fees_commissions"),
                "max_risk": row.get("max_risk_per_contract"),
                "target": row.get("target_profit"),
                "stop": row.get("stop_rule"),
                "time_stop": row.get("time_stop"),
                "invalidation": row.get("invalidation"),
                "review_close_rule": row.get("review_close_rule"),
                "expected_hold": row.get("expected_hold"),
                "expected_R": row.get("expected_R"),
                "expected_R_per_day": row.get("expected_R_per_day"),
                "avg_win_R": row.get("avg_win_R"),
                "avg_loss_R": row.get("avg_loss_R"),
                "payoff_ratio": row.get("payoff_ratio"),
                "probability_score": row.get("probability_score"),
                "calibrated_probability": row.get("calibrated_probability"),
                "validation_tier": row.get("confidence_tier"),
                "validation_scored_count": row.get("validation_scored_count"),
                "validation_profit_factor": row.get("validation_profit_factor"),
                "beats_baselines_count": row.get("beats_baselines_count"),
                "confidence_uncertainty_band": row.get("confidence_uncertainty_band"),
                "volume": row.get("liquidity_volume"),
                "open_interest": row.get("liquidity_open_interest"),
                "capacity_estimate": row.get("capacity_estimate_contracts"),
                "DTE": row.get("dte"),
                "IV_volatility_context": row.get("iv_volatility_context"),
                "event_risk": row.get("event_risk"),
                "regime_alignment": row.get("regime_alignment"),
                "live_quote_validation_status": row.get("live_quote_validation_status"),
                "catalyst_thesis": row.get("reason_summary"),
                "UW_evidence": row.get("uw_evidence"),
                "edge_review_reason": row.get("edge_review_reason"),
                "edge_review_evidence": row.get("edge_review_evidence"),
                "blockers": ";".join(row.get("block_reasons") or []),
                "promotion_requirements": row.get("promotion_requirements") or promotion_needed_text(row),
                "kill_switch_triggered": row.get("kill_switch_triggered") or "",
                "artifact_paths": artifact_paths_text,
            }
        )
    if not board:
        board.append(no_trade_decision_board_row(as_of, source_complete, daily_decision, artifact_paths_text))
    return board


def no_trade_decision_board_row(
    as_of: str,
    source_complete: bool,
    daily_decision: str,
    artifact_paths_text: str,
) -> Dict[str, Any]:
    return {
        "run_date": as_of,
        "pipeline_version": PIPELINE_VERSION,
        "artifact_schema_version": DECISION_BOARD_SCHEMA_VERSION,
        "daily_trade_decision": daily_decision or "NO_TRADE",
        "source_complete": bool(source_complete),
        "candidate_id": f"NO_TRADE_{as_of}",
        "status": "NO_TRADE",
        "ticker": "",
        "direction": "",
        "full_ticket": "No candidate cleared required evidence, execution, risk, and validation gates.",
        "entry": "",
        "max_risk": "",
        "expected_R": "",
        "expected_R_per_day": "",
        "probability_score": "",
        "calibrated_probability": "",
        "validation_tier": "",
        "blockers": "NO_REVIEWABLE_TICKETS",
        "promotion_requirements": "new source-complete evidence, valid quote, positive EV, validation edge, and no kill-switches",
        "kill_switch_triggered": "",
        "artifact_paths": artifact_paths_text,
    }


def validate_decision_board_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    required = [
        "run_date",
        "pipeline_version",
        "artifact_schema_version",
        "daily_trade_decision",
        "source_complete",
        "candidate_id",
        "status",
        "ticker",
        "direction",
        "full_ticket",
        "entry",
        "max_risk",
        "expected_R",
        "expected_R_per_day",
        "probability_score",
        "calibrated_probability",
        "validation_tier",
        "blockers",
        "promotion_requirements",
        "kill_switch_triggered",
        "artifact_paths",
    ]
    allowed_statuses = {"AUTO_APPROVED", "TRADE_REVIEW", "AVOID", "NO_TRADE"}
    for idx, row in enumerate(rows, 1):
        for field in required:
            if field not in row:
                errors.append(f"row {idx} missing {field}")
        status = row.get("status")
        if status not in allowed_statuses:
            errors.append(f"row {idx} invalid status {status!r}")
        if row.get("status") != "NO_TRADE" and not row.get("candidate_id"):
            errors.append(f"row {idx} missing candidate_id")
    return errors


def decision_board_fieldnames() -> List[str]:
    return [
        "run_date",
        "pipeline_version",
        "artifact_schema_version",
        "daily_trade_decision",
        "source_complete",
        "candidate_id",
        "status",
        "ticker",
        "direction",
        "strategy",
        "full_ticket",
        "all_legs",
        "buy_sell",
        "call_put",
        "strikes",
        "expiration",
        "entry",
        "entry_debit_credit",
        "bid",
        "ask",
        "spread",
        "fill_model",
        "slippage",
        "fees_commissions",
        "max_risk",
        "target",
        "stop",
        "time_stop",
        "invalidation",
        "review_close_rule",
        "expected_hold",
        "expected_R",
        "expected_R_per_day",
        "avg_win_R",
        "avg_loss_R",
        "payoff_ratio",
        "probability_score",
        "calibrated_probability",
        "validation_tier",
        "validation_scored_count",
        "validation_profit_factor",
        "beats_baselines_count",
        "confidence_uncertainty_band",
        "volume",
        "open_interest",
        "capacity_estimate",
        "DTE",
        "IV_volatility_context",
        "event_risk",
        "regime_alignment",
        "live_quote_validation_status",
        "catalyst_thesis",
        "UW_evidence",
        "edge_review_reason",
        "edge_review_evidence",
        "blockers",
        "promotion_requirements",
        "kill_switch_triggered",
        "artifact_paths",
    ]


def build_walk_forward_performance_rows(validation_bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    family_tiers = validation_bundle.get("family_tiers", {})
    rows: List[Dict[str, Any]] = []
    for row in validation_bundle.get("validation_scorecard", []):
        tier = family_tiers.get(str(row.get("pattern_family") or ""), {})
        rows.append(
            {
                "split": row.get("split"),
                "sample": row.get("sample"),
                "pattern_family": row.get("pattern_family"),
                "validation_tier": tier.get("confidence_tier", ""),
                "net_R_after_fees_slippage": row.get("average_net_r"),
                "expected_R": tier.get("validation_average_net_r"),
                "expected_R_per_day": safe_div(tier.get("validation_average_net_r"), 5),
                "win_rate": row.get("win_rate_scored"),
                "avg_R": row.get("average_net_r"),
                "median_R": row.get("median_net_r"),
                "profit_factor": row.get("profit_factor"),
                "drawdown_proxy": row.get("drawdown_proxy_r"),
                "losing_streak": row.get("worst_losing_streak"),
                "brier_calibration_score": "",
                "scored_count": row.get("scored_count"),
                "unscorable_count": row.get("unscorable_count"),
                "quote_coverage": row.get("tradeable_with_real_quotes_pct"),
                "blocker_distribution": row.get("top_block_reasons"),
                "baselines_beaten": tier.get("beats_baselines_count", ""),
            }
        )
    return rows


def walk_forward_fieldnames() -> List[str]:
    return [
        "split",
        "sample",
        "pattern_family",
        "validation_tier",
        "net_R_after_fees_slippage",
        "expected_R",
        "expected_R_per_day",
        "win_rate",
        "avg_R",
        "median_R",
        "profit_factor",
        "drawdown_proxy",
        "losing_streak",
        "brier_calibration_score",
        "scored_count",
        "unscorable_count",
        "quote_coverage",
        "blocker_distribution",
        "baselines_beaten",
    ]


def build_threshold_sensitivity_rows(
    validation_bundle: Mapping[str, Any],
    risk_config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    tiers = list((validation_bundle.get("family_tiers") or {}).values())
    definitions = [
        {
            "threshold_set": "OLD_DAILY_TRADE_GATE",
            "min_oos_scored": 20,
            "min_profit_factor": 1.0,
            "min_avg_R": 0.0,
            "min_baselines_beaten": 1,
            "notes": "Approximate legacy daily promotion pressure before strict acceptance gate.",
        },
        {
            "threshold_set": "NEW_AUTO_APPROVED_GATE",
            "min_oos_scored": int(risk_config.get("min_oos_scored_outcomes", 30)),
            "min_profit_factor": float(risk_config.get("min_profit_factor", 1.2)),
            "min_avg_R": float(risk_config.get("min_expected_r", 0.0)),
            "min_baselines_beaten": int(risk_config.get("min_baselines_beaten", 2)),
            "notes": "Configured strict gate used for AUTO_APPROVED.",
        },
        {
            "threshold_set": "STRICT_EDGE_GATE",
            "min_oos_scored": 50,
            "min_profit_factor": 1.5,
            "min_avg_R": 0.05,
            "min_baselines_beaten": 3,
            "notes": "Stress threshold for sensitivity review.",
        },
    ]
    rows: List[Dict[str, Any]] = []
    for definition in definitions:
        passing = [
            t
            for t in tiers
            if int(t.get("validation_scored_count") or 0) >= definition["min_oos_scored"]
            and (num(t.get("validation_profit_factor")) or 0.0) >= definition["min_profit_factor"]
            and (num(t.get("validation_average_net_r")) or -999.0) > definition["min_avg_R"]
            and int(t.get("beats_baselines_count") or 0) >= definition["min_baselines_beaten"]
        ]
        rows.append(
            {
                **definition,
                "passing_family_count": len(passing),
                "passing_families": ";".join(str(t.get("pattern_family") or "") for t in passing[:25]),
                "candidate_family_count": len(tiers),
            }
        )
    return rows


def threshold_sensitivity_fieldnames() -> List[str]:
    return [
        "threshold_set",
        "min_oos_scored",
        "min_profit_factor",
        "min_avg_R",
        "min_baselines_beaten",
        "passing_family_count",
        "passing_families",
        "candidate_family_count",
        "notes",
    ]


def build_ablation_attribution_rows(validation_bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    baseline_by_name = {
        str(r.get("baseline") or ""): r for r in validation_bundle.get("baseline_comparison", [])
    }
    scorecard = validation_bundle.get("validation_scorecard", [])
    combined_avg = statistics.fmean(
        [float(r["average_net_r"]) for r in scorecard if r.get("average_net_r") not in (None, "")]
    ) if any(r.get("average_net_r") not in (None, "") for r in scorecard) else None
    rows = []
    mappings = [
        ("UW flow only", "BASELINE_NAIVE_UW_FLOW_ONLY"),
        ("macro only", "BASELINE_NAIVE_CATALYST_ONLY"),
        ("validation only", "validation_scorecard_pre_daily_gate"),
        ("regime filter", "MARKET_REGIME_CONFLICT blocker impact"),
        ("liquidity filter", "BASELINE_RANDOM_SAME_DATE_LIQUIDITY"),
        ("volatility filter", "VOL_EXPANSION_CATALYST families"),
        ("fill/slippage filter", "SCORED option exits after fees"),
        ("combined model", "pattern families after all gates"),
    ]
    for component, reference in mappings:
        baseline = baseline_by_name.get(reference, {})
        rows.append(
            {
                "component": component,
                "reference": reference,
                "average_net_R": baseline.get("average_net_r", combined_avg if component == "combined model" else ""),
                "scored_count": baseline.get("scored_count", ""),
                "interpretation": ablation_interpretation(component, baseline, combined_avg),
            }
        )
    return rows


def ablation_interpretation(component: str, baseline: Mapping[str, Any], combined_avg: Optional[float]) -> str:
    if component == "combined model":
        return "Combined model after discovery, validation, quote/liquidity, regime, cost, and risk gates."
    if baseline:
        return "Baseline scored directly against historical option outcomes."
    return "Component is represented as a documented gate/filter; no separate standalone P/L model is promoted."


def ablation_fieldnames() -> List[str]:
    return ["component", "reference", "average_net_R", "scored_count", "interpretation"]


def build_full_backtest_by_family(outcomes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for outcome in select_validation_gate_outcomes(outcomes):
        grouped[str(outcome.get("pattern_family") or "")].append(outcome)
    rows = [full_backtest_group_row("pattern_family", family, items) for family, items in grouped.items()]
    rows.sort(
        key=lambda r: (
            int(r.get("scored_count") or 0) >= 30,
            num(r.get("avg_R")) or -999.0,
            num(r.get("profit_factor")) or -999.0,
        ),
        reverse=True,
    )
    return rows


def build_full_backtest_by_month(outcomes: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for outcome in select_validation_gate_outcomes(outcomes):
        month = str(outcome.get("signal_date") or "")[:7] or "UNKNOWN"
        grouped[month].append(outcome)
    return [full_backtest_group_row("month", month, items) for month, items in sorted(grouped.items())]


def full_backtest_group_row(key_name: str, key_value: str, items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    scored = [o for o in items if o.get("status") == "SCORED" and num(o.get("net_r")) is not None]
    partial = [o for o in items if o.get("status") == "PARTIAL"]
    unscorable = [o for o in items if o.get("status") == "UNSCORABLE"]
    net_rs = [float(o["net_r"]) for o in scored]
    positives = sum(r for r in net_rs if r > 0)
    negatives = abs(sum(r for r in net_rs if r < 0))
    dates = sorted(str(o.get("signal_date") or "") for o in items if o.get("signal_date"))
    return {
        key_name: key_value,
        "signal_count": len(items),
        "scored_count": len(scored),
        "partial_count": len(partial),
        "unscorable_count": len(unscorable),
        "quote_coverage": safe_div(len(scored), len(items)),
        "win_rate": safe_div(sum(1 for r in net_rs if r > 0), len(scored)),
        "avg_R": statistics.fmean(net_rs) if net_rs else None,
        "median_R": statistics.median(net_rs) if net_rs else None,
        "profit_factor": positives / negatives if negatives > 0 else (None if positives == 0 else 999.0),
        "gross_R": sum(net_rs) if net_rs else None,
        "drawdown_proxy": drawdown_proxy(net_rs),
        "max_losing_streak": worst_losing_streak(net_rs),
        "first_signal_date": dates[0] if dates else "",
        "last_signal_date": dates[-1] if dates else "",
    }


def full_backtest_family_fieldnames() -> List[str]:
    return [
        "pattern_family",
        "signal_count",
        "scored_count",
        "partial_count",
        "unscorable_count",
        "quote_coverage",
        "win_rate",
        "avg_R",
        "median_R",
        "profit_factor",
        "gross_R",
        "drawdown_proxy",
        "max_losing_streak",
        "first_signal_date",
        "last_signal_date",
    ]


def full_backtest_month_fieldnames() -> List[str]:
    return [
        "month",
        "signal_count",
        "scored_count",
        "partial_count",
        "unscorable_count",
        "quote_coverage",
        "win_rate",
        "avg_R",
        "median_R",
        "profit_factor",
        "gross_R",
        "drawdown_proxy",
        "max_losing_streak",
        "first_signal_date",
        "last_signal_date",
    ]


def render_full_backtest_summary(
    as_of: str,
    outcomes: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]],
    month_rows: Sequence[Mapping[str, Any]],
) -> str:
    gate_outcomes = select_validation_gate_outcomes(outcomes)
    overall = full_backtest_group_row("scope", "canonical_cumulative_walk_forward", gate_outcomes)
    dates = [str(o.get("signal_date") or "") for o in gate_outcomes if o.get("signal_date")]
    top = [r for r in family_rows if int(r.get("scored_count") or 0) >= 30 and (num(r.get("avg_R")) or -999.0) > 0]
    bottom = [r for r in family_rows if int(r.get("scored_count") or 0) >= 30 and (num(r.get("avg_R")) or 999.0) < 0]
    bottom = sorted(bottom, key=lambda r: num(r.get("avg_R")) or 999.0)[:5]
    lines = [
        f"# Full Backtest Summary - {as_of}",
        "",
        "Method: canonical cumulative walk-forward holdouts only. Each validation month is counted once; overlapping exploratory validation splits are not double-counted in this summary or in the daily approval gate.",
        "",
        "## Overall",
        f"- Validation dates: {(min(dates) if dates else 'n/a')} through {(max(dates) if dates else 'n/a')}",
        f"- Signals: {overall.get('signal_count', 0)}",
        f"- Scored: {overall.get('scored_count', 0)}",
        f"- Quote coverage: {fmt_pct(overall.get('quote_coverage'))}",
        f"- Win rate: {fmt_pct(overall.get('win_rate'))}",
        f"- Avg R: {fmt_num(overall.get('avg_R'))}",
        f"- Median R: {fmt_num(overall.get('median_R'))}",
        f"- Profit factor: {fmt_num(overall.get('profit_factor'))}",
        f"- Gross R: {fmt_num(overall.get('gross_R'))}",
        f"- Drawdown proxy: {fmt_num(overall.get('drawdown_proxy'))}",
        f"- Max losing streak: {overall.get('max_losing_streak', 0)}",
        "",
        "## Monthly",
        "| Month | Signals | Scored | Quote Coverage | Win Rate | Avg R | Profit Factor | Gross R |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in month_rows:
        lines.append(
            f"| {row.get('month')} | {row.get('signal_count')} | {row.get('scored_count')} | "
            f"{fmt_pct(row.get('quote_coverage'))} | {fmt_pct(row.get('win_rate'))} | "
            f"{fmt_num(row.get('avg_R'))} | {fmt_num(row.get('profit_factor'))} | {fmt_num(row.get('gross_R'))} |"
        )
    lines.extend(["", "## Best Families With At Least 30 Scored Outcomes"])
    if top:
        for row in top[:5]:
            lines.append(
                f"- {row.get('pattern_family')}: scored {row.get('scored_count')}, "
                f"win {fmt_pct(row.get('win_rate'))}, avg R {fmt_num(row.get('avg_R'))}, "
                f"PF {fmt_num(row.get('profit_factor'))}, gross R {fmt_num(row.get('gross_R'))}"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "## Worst Families With At Least 30 Scored Outcomes"])
    if bottom:
        for row in bottom:
            lines.append(
                f"- {row.get('pattern_family')}: scored {row.get('scored_count')}, "
                f"win {fmt_pct(row.get('win_rate'))}, avg R {fmt_num(row.get('avg_R'))}, "
                f"PF {fmt_num(row.get('profit_factor'))}, gross R {fmt_num(row.get('gross_R'))}"
            )
    else:
        lines.append("- None.")
    return "\n".join(lines)


def render_calibration_summary(as_of: str, metrics: Mapping[str, Any]) -> str:
    lines = [
        f"# Calibration Summary - {as_of}",
        "",
        f"- Artifact schema: {ARTIFACT_SCHEMA_VERSION}",
        f"- Status: {metrics.get('status')}",
        f"- Scored predictions: {metrics.get('scored_prediction_count', 0)}",
        f"- Brier score: {fmt_num(metrics.get('brier_score'))}",
        "",
        "## Reliability Buckets",
        "| Bucket | Count | Avg Predicted | Observed Win Rate | Wins |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in metrics.get("reliability_buckets", []):
        lines.append(
            f"| {row['bucket']} | {row['count']} | {fmt_pct(row.get('avg_predicted_probability'))} | "
            f"{fmt_pct(row.get('observed_win_rate'))} | {row['wins']} |"
        )
    if not metrics.get("reliability_buckets"):
        lines.append("| n/a | 0 | n/a | n/a | 0 |")
    lines.extend(
        [
            "",
            "Calibration is used as an approval gate. Missing or weak calibration keeps candidates out of AUTO_APPROVED.",
        ]
    )
    return "\n".join(lines)


def build_shadow_ledger_rows(
    as_of: str,
    decision_rows: Sequence[Mapping[str, Any]],
    validation_bundle: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in decision_rows:
        if row.get("status") not in {"AUTO_APPROVED", "TRADE_REVIEW"}:
            continue
        rows.append(
            {
                "recommendation_date": as_of,
                "candidate_id": row.get("candidate_id"),
                "status": row.get("status"),
                "full_ticket": row.get("full_ticket"),
                "entry_assumption": row.get("entry"),
                "target": row.get("target"),
                "stop": row.get("stop"),
                "time_stop": row.get("time_stop"),
                "max_favorable_excursion": "",
                "max_adverse_excursion": "",
                "exit_status": "OPEN_SHADOW_PENDING",
                "net_R_after_costs": "",
                "days_held": "",
                "catalyst_thesis_validity": "pending",
                "later_kill_switch_would_have_skipped": row.get("kill_switch_triggered"),
            }
        )
    for outcome in validation_bundle.get("outcomes", []):
        if outcome.get("sample") != "VALIDATION" or outcome.get("horizon") != "5d":
            continue
        rows.append(
            {
                "recommendation_date": outcome.get("signal_date"),
                "candidate_id": hashlib.sha256(stable_json(outcome).encode("utf-8")).hexdigest()[:16],
                "status": "HISTORICAL_SHADOW",
                "full_ticket": outcome.get("lead_option_symbol"),
                "entry_assumption": outcome.get("entry_credit") or outcome.get("entry_ask"),
                "target": "",
                "stop": "",
                "time_stop": "5d validation horizon",
                "max_favorable_excursion": "",
                "max_adverse_excursion": "",
                "exit_status": outcome.get("status"),
                "net_R_after_costs": outcome.get("net_r"),
                "days_held": "5",
                "catalyst_thesis_validity": "historical_outcome_only",
                "later_kill_switch_would_have_skipped": "not_replayed",
            }
        )
    return rows


def shadow_ledger_fieldnames() -> List[str]:
    return [
        "recommendation_date",
        "candidate_id",
        "status",
        "full_ticket",
        "entry_assumption",
        "target",
        "stop",
        "time_stop",
        "max_favorable_excursion",
        "max_adverse_excursion",
        "exit_status",
        "net_R_after_costs",
        "days_held",
        "catalyst_thesis_validity",
        "later_kill_switch_would_have_skipped",
    ]


def render_shadow_summary(as_of: str, rows: Sequence[Mapping[str, Any]]) -> str:
    historical = [r for r in rows if r.get("status") == "HISTORICAL_SHADOW" and r.get("net_R_after_costs") not in (None, "")]
    net_rs = [float(r["net_R_after_costs"]) for r in historical if num(r.get("net_R_after_costs")) is not None]
    pending = [r for r in rows if r.get("exit_status") == "OPEN_SHADOW_PENDING"]
    lines = [
        f"# Shadow Outcome Summary - {as_of}",
        "",
        f"- Open AUTO_APPROVED/TRADE_REVIEW shadow rows: {len(pending)}",
        f"- Historical validation shadow rows: {len(historical)}",
        f"- Average net R after costs: {fmt_num(statistics.fmean(net_rs) if net_rs else None)}",
        f"- Median net R after costs: {fmt_num(statistics.median(net_rs) if net_rs else None)}",
        f"- Profit factor: {fmt_num(profit_factor_from_rs(net_rs))}",
        f"- Worst losing streak: {worst_losing_streak(net_rs) if net_rs else 0}",
        "",
        "Rows are shadow records only. This pipeline does not place trades or orders.",
    ]
    return "\n".join(lines)


def profit_factor_from_rs(net_rs: Sequence[float]) -> Optional[float]:
    positives = sum(r for r in net_rs if r > 0)
    negatives = abs(sum(r for r in net_rs if r < 0))
    if negatives > 0:
        return positives / negatives
    if positives > 0:
        return 999.0
    return None


def build_artifact_manifest(
    as_of: str,
    base_dir: Path,
    config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    paths: Mapping[str, str],
    schema_errors: Sequence[str],
    runtime_seconds: float,
) -> Dict[str, Any]:
    source_files = metadata.get("source_files_for_as_of") or []
    return {
        "run_date": as_of,
        "pipeline_version": PIPELINE_VERSION,
        "artifact_schema_version": MANIFEST_SCHEMA_VERSION,
        "git_sha": current_git_sha(base_dir),
        "command": metadata.get("command"),
        "config_path": config.get("risk_config_path"),
        "config_hash": config.get("risk_config_hash"),
        "deterministic_seed": config.get("seed"),
        "source_files_used": source_files,
        "source_fingerprints": [fingerprint_source_label(label) for label in source_files],
        "cache_status": {
            "hit": any((counts or {}).get("bot_eod_cache_hit") for counts in (metadata.get("source_counts_by_date") or {}).values()),
            "built": any((counts or {}).get("bot_eod_cache_built") for counts in (metadata.get("source_counts_by_date") or {}).values()),
        },
        "artifact_paths": dict(paths),
        "artifact_schema_versions": {
            "decision_board": DECISION_BOARD_SCHEMA_VERSION,
            "manifest": MANIFEST_SCHEMA_VERSION,
            "general": ARTIFACT_SCHEMA_VERSION,
        },
        "runtime_seconds": round(runtime_seconds, 3),
        "warning_count": len(metadata.get("skipped_sources_for_as_of") or []),
        "error_count": len(schema_errors),
        "schema_errors": list(schema_errors),
        "no_trade_order_placement": True,
    }


def current_git_sha(base_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=base_dir,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return "UNKNOWN"
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def fingerprint_source_label(label: str) -> Dict[str, Any]:
    path_text, _, member = str(label).partition("::")
    path = Path(path_text)
    row: Dict[str, Any] = {"label": label, "path": path_text, "member": member}
    if not path.exists():
        row["exists"] = False
        return row
    row.update({"exists": True, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    try:
        if member:
            with zipfile.ZipFile(path) as zf:
                info = zf.getinfo(member)
            row.update({"member_bytes": info.file_size, "member_crc": info.CRC})
        else:
            h = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    h.update(chunk)
            row["sha256"] = h.hexdigest()
    except Exception as exc:
        row["fingerprint_error"] = str(exc)
    return row


def render_profitability_audit(
    as_of: str,
    metadata: Mapping[str, Any],
    decision_rows: Sequence[Mapping[str, Any]],
    validation_bundle: Mapping[str, Any],
    threshold_rows: Sequence[Mapping[str, Any]],
    calibration_metrics: Mapping[str, Any],
    run_controls: Mapping[str, Any],
) -> str:
    counts = Counter(str(r.get("status") or "") for r in decision_rows)
    blockers = Counter()
    for row in decision_rows:
        for blocker in str(row.get("blockers") or "").split(";"):
            if blocker:
                blockers[blocker] += 1
    lines = [
        f"# Profitability Audit - {as_of}",
        "",
        "## Old Pipeline Failure Modes",
        "- Trade-looking rows could be promoted without a single status vocabulary tied to EV, quote quality, calibration, risk, and kill-switches.",
        "- Pretty markdown could hide duplicated tickets, incomplete CSV contracts, weak OOS samples, and missing shadow outcome tracking.",
        "- Baseline comparison existed, but it was not packaged with threshold sensitivity, calibration, manifest, and decision-board schema validation.",
        "",
        "## Root Causes",
        "- Discovery, validation, reporting, and approval were too tightly implied by the old `TRADE` label.",
        "- Realistic execution assumptions were scattered across outcome scoring instead of repeated on every current ticket.",
        "- Run reproducibility existed in metadata, but not as a standalone artifact manifest with schema versions and source fingerprints.",
        "",
        "## Fixes Applied",
        "- Added strict statuses: AUTO_APPROVED, TRADE_REVIEW, AVOID, and NO_TRADE.",
        "- Added expected R, expected R/day, calibrated probability, confidence band, profit-factor, capacity, risk, fill, fees, slippage, lifecycle, and kill-switch gates.",
        "- Added decision board JSON/CSV, artifact manifest, walk-forward performance, threshold sensitivity, calibration summary, shadow ledger, shadow summary, and this audit.",
        "- Kept frozen V1 untouched and preserved the existing source-first validation engine.",
        "",
        "## Old vs New Behavior",
        f"- New AUTO_APPROVED rows: {counts.get('AUTO_APPROVED', 0)}.",
        f"- New TRADE_REVIEW rows: {counts.get('TRADE_REVIEW', 0)}.",
        f"- New AVOID rows: {counts.get('AVOID', 0)}.",
        f"- New NO_TRADE rows: {counts.get('NO_TRADE', 0)}.",
        f"- Daily trade decision: {metadata.get('daily_trade_decision')}.",
        "",
        "## Examples From Required Dates",
        f"- This artifact run date: {as_of}. Required reruns must include 2026-05-15 and 2026-05-18; compare their decision boards and manifests side by side.",
        "- 2026-05-15: inspect `decision_board.csv`, `daily_report_2026-05-15.md`, and `artifact_manifest.json` for final counts and blockers.",
        "- 2026-05-18: inspect `decision_board.csv`, `daily_report_2026-05-18.md`, and `artifact_manifest.json` for final counts and blockers.",
        "",
        "## Why Each Gate Exists",
        "- EV and expected R/day prevent attractive narratives from outranking negative expectancy after fees and slippage.",
        "- Profit factor, OOS count, calibration, and baselines reduce one-off luck and overfit promotion.",
        "- Quote, spread, volume, OI, capacity, and max-risk gates prevent untradeable or oversized tickets.",
        "- Macro, catalyst, event, volatility, and regime gates prevent directionally inconsistent tickets from being auto-approved.",
        "- Kill-switches stop promotion when source, quote, validation, calibration, schema, or shadow evidence degrades.",
        "",
        "## What Got Better",
        "- Every current setup is ticket-first and has explicit blocker/promotion requirements.",
        "- CSV and JSON artifacts are schema-versioned and deduped by ticket key.",
        "- Validation evidence now flows into current expected-R fields and shadow tracking.",
        "",
        "## What Got Worse",
        "- AUTO_APPROVED is deliberately rarer. Some setups that used to appear actionable now require review because strict evidence is missing.",
        "- The pipeline can be more conservative when live quote validation is unavailable.",
        "",
        "## Validation And Calibration",
        f"- Validation splits: {len(validation_bundle.get('splits', []))}.",
        f"- Calibration status: {calibration_metrics.get('status')} with Brier score {fmt_num(calibration_metrics.get('brier_score'))}.",
        f"- Run kill-switches: {', '.join(run_controls.get('run_kill_switches') or []) or 'none'}.",
        "",
        "## Top Blockers",
    ]
    for blocker, count in blockers.most_common(12):
        lines.append(f"- {blocker}: {count}")
    if not blockers:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Threshold Sensitivity",
        ]
    )
    for row in threshold_rows:
        lines.append(
            f"- {row['threshold_set']}: {row['passing_family_count']}/{row['candidate_family_count']} families pass "
            f"(min scored {row['min_oos_scored']}, PF {row['min_profit_factor']}, baselines {row['min_baselines_beaten']})."
        )
    lines.extend(
        [
            "",
            "## Known Limitations",
            "- This does not guarantee profit; it only states whether the historical evidence supports positive expected edge under documented assumptions.",
            "- Schwab/live-chain validation is not performed by this local pattern package unless a future integration supplies live quotes.",
            "- Shadow outcomes are strongest for dates with future option quote coverage; sparse quote history remains a limiting factor.",
            "- Corporate-action, delisting, and symbol-change handling is source-dependent and remains a data-risk item.",
        ]
    )
    return "\n".join(lines)


def render_runbook(as_of: str, config: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Options Pattern Pipeline Runbook",
            "",
            "## Run Command",
            "```bash",
            "python3 -m uwos.options_pattern_pipeline_v1 \\",
            "  --base-dir /Users/anuppamvi/uw_root/tradedesk \\",
            f"  --as-of {as_of} \\",
            f"  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/options_pattern_pipeline_v1/{as_of}",
            "```",
            "",
            "## Artifact Meanings",
            "- `decision_board.csv/json`: ticket-first status board and schema contract.",
            "- `artifact_manifest.json`: reproducibility metadata, git SHA, source fingerprints, config hash, and artifact paths.",
            "- `walk_forward_performance.csv`: validation performance after fees/slippage.",
            "- `threshold_sensitivity.csv`: old-vs-new approval threshold comparison.",
            "- `calibration_summary.md`: Brier score and reliability buckets.",
            "- `shadow_recommendation_ledger.csv`: open and historical shadow rows.",
            "",
            "## Status Meanings",
            "- AUTO_APPROVED: all configured EV, validation, execution, risk, lifecycle, macro, and kill-switch gates passed.",
            "- TRADE_REVIEW: a complete ticket exists but one or more promotion requirements remain.",
            "- AVOID: execution, risk, EV, quote, validation, or data quality makes the ticket unsuitable.",
            "- NO_TRADE: no reviewable ticket exists for the run.",
            "",
            "## Before Trusting A Recommendation",
            "- Inspect `decision_board.csv` first, then the daily report.",
            "- Confirm live quote, spread, liquidity, event risk, and portfolio exposure manually before any real order.",
            "- Review shadow ledger deterioration and kill-switches.",
            "- Never treat historical edge as guaranteed profit.",
            "",
            "## Source Incomplete Handling",
            "- If source completeness is false, the kill-switch prevents AUTO_APPROVED and the board falls to NO_TRADE/AVOID.",
            "",
            "## Rollback",
            "- Restore behavior from `uwos/options_pattern_pipeline_v1_frozen_v1/` only if explicitly requested.",
            "- Verify frozen V1 remains unchanged with `git diff --exit-code options-pattern-pipeline-v1 -- uwos/options_pattern_pipeline_v1_frozen_v1`.",
            "",
            "## GitHub Sync",
            "- Commit only live pipeline, tests, configs, docs, and generated acceptance artifacts.",
            "- Do not commit secrets, account numbers, Schwab credentials, tokens, or unrelated user changes.",
            "",
            f"Risk config hash: `{config.get('risk_config_hash')}`",
        ]
    )


def write_outputs(
    out_dir: Path,
    base_dir: Path,
    as_of: str,
    config: Mapping[str, Any],
    inventory_rows: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Snapshot],
    validation_bundle: Mapping[str, Any],
    daily_pattern_config: Mapping[str, Any],
    daily_rows: Sequence[Mapping[str, Any]],
    missed_rows: Sequence[Mapping[str, Any]],
    sentiment_summary: Mapping[str, Any],
    source_completeness: Mapping[str, Any],
    macro_geo_bundle: Mapping[str, Any],
    runtime_seconds: float,
) -> Dict[str, str]:
    decision_enriched_rows, run_controls = prepare_decision_rows(
        daily_rows,
        validation_bundle,
        source_completeness,
        config,
    )
    actionable = dedupe_rows_by_ticket([r for r in decision_enriched_rows if r["status"] == "AUTO_APPROVED"])
    trade_review = build_trade_review_candidates(decision_enriched_rows)
    watch = dedupe_rows_by_ticket([r for r in decision_enriched_rows if r["status"] == "TRADE_REVIEW" and r.get("classification") == "WATCH"])
    blocked = dedupe_rows_by_ticket([r for r in decision_enriched_rows if r["status"] == "AVOID"])

    discovered_rows = []
    for family, tier in sorted(validation_bundle["family_tiers"].items()):
        row = dict(tier)
        row["daily_pattern_config_json"] = stable_json(daily_pattern_config)
        discovered_rows.append(row)

    daily_decision = daily_trade_decision(actionable, trade_review, blocked)
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "command": " ".join(sys.argv),
        "base_dir": str(base_dir),
        "as_of": as_of,
        "output_dir": str(out_dir),
        "deterministic_seed": config["seed"],
        "source_dates_used": sorted(snapshots),
        "source_files_by_date": {d: snapshots[d].source_files for d in sorted(snapshots)},
        "skipped_sources_by_date": {d: snapshots[d].skipped_sources for d in sorted(snapshots)},
        "source_counts_by_date": {d: snapshots[d].counts for d in sorted(snapshots)},
        "source_files_for_as_of": snapshots[as_of].source_files,
        "skipped_sources_for_as_of": snapshots[as_of].skipped_sources,
        "validation_splits": validation_bundle["splits"],
        "daily_pattern_config": daily_pattern_config,
        "input_policy": config["input_policy"],
        "source_completeness": source_completeness,
        "macro_geo_summary": macro_geo_bundle.get("summary", {}),
        "daily_trade_decision": daily_decision,
        "run_kill_switches": run_controls["run_kill_switches"],
        "candidate_counts": {
            "auto_approved": len(actionable),
            "trade_review_candidates": len(trade_review),
            "avoid": len(blocked),
            "watchlist_research_setups": len(watch),
            "blocked_candidates": len(blocked),
        },
        "reproducibility": {
            "python": sys.version,
            "random_seed": config["seed"],
            "max_chain_rows_per_day": config["max_chain_rows_per_day"],
            "max_flow_file_mb": config["max_flow_file_mb"],
            "bot_eod_cache_dir": config["bot_eod_cache_dir"],
            "risk_config_path": config.get("risk_config_path"),
            "risk_config_hash": config.get("risk_config_hash"),
        },
        "risk_config": config.get("risk_config", {}),
        "verdict": final_verdict(validation_bundle, daily_rows),
    }

    paths = {
        "daily_report": str(out_dir / f"daily_report_{as_of}.md"),
        "actionable_trades": str(out_dir / "actionable_trades.csv"),
        "watchlist_research_setups": str(out_dir / "watchlist_research_setups.csv"),
        "blocked_candidates": str(out_dir / "blocked_candidates.csv"),
        "discovered_pattern_families": str(out_dir / "discovered_pattern_families.csv"),
        "market_regime_summary": str(out_dir / "market_regime_summary.json"),
        "sentiment_news_summary": str(out_dir / "sentiment_news_summary.json"),
        "validation_scorecard": str(out_dir / "validation_scorecard.csv"),
        "validation_gate_scorecard": str(out_dir / "validation_gate_scorecard.csv"),
        "train_scorecard": str(out_dir / "train_scorecard.csv"),
        "validation_details": str(out_dir / "validation_details.csv"),
        "baseline_comparison": str(out_dir / "baseline_comparison.csv"),
        "missed_mover_audit": str(out_dir / "missed_mover_audit.csv"),
        "macro_geo_catalysts": str(out_dir / "macro_geo_catalysts.json"),
        "macro_geo_ticker_map": str(out_dir / "macro_geo_ticker_map.csv"),
        "macro_geo_uw_confirmation": str(out_dir / "macro_geo_uw_confirmation.csv"),
        "macro_geo_promotion_decisions": str(out_dir / "macro_geo_promotion_decisions.csv"),
        "trade_review_candidates": str(out_dir / "trade_review_candidates.csv"),
        "decision_board_json": str(out_dir / "decision_board.json"),
        "decision_board_csv": str(out_dir / "decision_board.csv"),
        "artifact_manifest": str(out_dir / "artifact_manifest.json"),
        "full_backtest_summary": str(out_dir / "full_backtest_summary.md"),
        "full_backtest_by_family": str(out_dir / "full_backtest_by_family.csv"),
        "full_backtest_by_month": str(out_dir / "full_backtest_by_month.csv"),
        "walk_forward_performance": str(out_dir / "walk_forward_performance.csv"),
        "threshold_sensitivity": str(out_dir / "threshold_sensitivity.csv"),
        "calibration_summary": str(out_dir / "calibration_summary.md"),
        "profitability_audit": str(out_dir / "profitability_audit.md"),
        "shadow_recommendation_ledger": str(out_dir / "shadow_recommendation_ledger.csv"),
        "shadow_outcome_summary": str(out_dir / "shadow_outcome_summary.md"),
        "ablation_attribution": str(out_dir / "ablation_attribution.csv"),
        "pattern_pipeline_runbook": str(out_dir / "pattern_pipeline_runbook.md"),
        "pattern_observability_matrix": str(out_dir / "pattern_observability_matrix.md"),
        "missed_pattern_audit": str(out_dir / "missed_pattern_audit.md"),
        "inventory": str(out_dir / "source_inventory.csv"),
        "metadata": str(out_dir / "metadata.json"),
        "regime_sector_validation": str(out_dir / "validation_by_regime_sector_ticker.csv"),
    }

    decision_board_source_rows = dedupe_rows_by_ticket(decision_enriched_rows)
    decision_board = build_decision_board_rows(
        decision_board_source_rows,
        as_of,
        bool(source_completeness.get("source_complete")),
        daily_decision,
        paths,
    )
    schema_errors = validate_decision_board_rows(decision_board)
    if schema_errors and "KILL_SWITCH_ARTIFACT_SCHEMA_VALIDATION_FAILS" not in metadata["run_kill_switches"]:
        metadata["run_kill_switches"] = sorted(set(metadata["run_kill_switches"] + ["KILL_SWITCH_ARTIFACT_SCHEMA_VALIDATION_FAILS"]))
    walk_forward_rows = build_walk_forward_performance_rows(validation_bundle)
    threshold_rows = build_threshold_sensitivity_rows(validation_bundle, run_controls["risk_config"])
    ablation_rows = build_ablation_attribution_rows(validation_bundle)
    shadow_rows = build_shadow_ledger_rows(as_of, decision_board, validation_bundle)
    backtest_family_rows = build_full_backtest_by_family(validation_bundle.get("outcomes", []))
    backtest_month_rows = build_full_backtest_by_month(validation_bundle.get("outcomes", []))

    write_csv(Path(paths["actionable_trades"]), [trade_output_row(r) for r in actionable], trade_fieldnames())
    write_csv(Path(paths["watchlist_research_setups"]), [trade_output_row(r) for r in watch], trade_fieldnames())
    write_csv(Path(paths["blocked_candidates"]), [blocked_output_row(r) for r in blocked], blocked_fieldnames())
    write_csv(Path(paths["trade_review_candidates"]), [trade_review_output_row(r) for r in trade_review], trade_review_fieldnames())
    write_json(
        Path(paths["decision_board_json"]),
        {
            "schema_version": DECISION_BOARD_SCHEMA_VERSION,
            "run_date": as_of,
            "daily_trade_decision": daily_decision,
            "validation_errors": schema_errors,
            "rows": decision_board,
        },
    )
    write_csv(Path(paths["decision_board_csv"]), decision_board, decision_board_fieldnames())
    write_csv(Path(paths["walk_forward_performance"]), walk_forward_rows, walk_forward_fieldnames())
    write_csv(Path(paths["threshold_sensitivity"]), threshold_rows, threshold_sensitivity_fieldnames())
    write_csv(Path(paths["ablation_attribution"]), ablation_rows, ablation_fieldnames())
    write_csv(Path(paths["validation_gate_scorecard"]), validation_bundle.get("validation_gate_scorecard", []), scorecard_fieldnames())
    write_csv(Path(paths["full_backtest_by_family"]), backtest_family_rows, full_backtest_family_fieldnames())
    write_csv(Path(paths["full_backtest_by_month"]), backtest_month_rows, full_backtest_month_fieldnames())
    Path(paths["full_backtest_summary"]).write_text(
        render_full_backtest_summary(as_of, validation_bundle.get("outcomes", []), backtest_family_rows, backtest_month_rows),
        encoding="utf-8",
    )
    Path(paths["calibration_summary"]).write_text(
        render_calibration_summary(as_of, run_controls["calibration_metrics"]),
        encoding="utf-8",
    )
    Path(paths["profitability_audit"]).write_text(
        render_profitability_audit(
            as_of,
            metadata,
            decision_board,
            validation_bundle,
            threshold_rows,
            run_controls["calibration_metrics"],
            run_controls,
        ),
        encoding="utf-8",
    )
    write_csv(Path(paths["shadow_recommendation_ledger"]), shadow_rows, shadow_ledger_fieldnames())
    Path(paths["shadow_outcome_summary"]).write_text(render_shadow_summary(as_of, shadow_rows), encoding="utf-8")
    Path(paths["pattern_pipeline_runbook"]).write_text(render_runbook(as_of, config), encoding="utf-8")
    write_csv(Path(paths["discovered_pattern_families"]), discovered_rows, discovered_fieldnames())
    write_json(Path(paths["market_regime_summary"]), snapshots[as_of].market_regime)
    write_json(Path(paths["sentiment_news_summary"]), sentiment_summary)
    write_csv(Path(paths["validation_scorecard"]), validation_bundle["validation_scorecard"], scorecard_fieldnames())
    write_csv(Path(paths["train_scorecard"]), validation_bundle["train_scorecard"], scorecard_fieldnames())
    write_csv(Path(paths["validation_details"]), validation_bundle["outcomes"], validation_detail_fieldnames())
    write_csv(Path(paths["baseline_comparison"]), validation_bundle["baseline_comparison"], baseline_fieldnames())
    write_csv(Path(paths["missed_mover_audit"]), missed_rows, missed_fieldnames())
    write_json(Path(paths["macro_geo_catalysts"]), macro_geo_bundle.get("catalysts", []))
    write_csv(Path(paths["macro_geo_ticker_map"]), macro_geo_bundle.get("ticker_map", []), macro_geo_ticker_map_fieldnames())
    write_csv(
        Path(paths["macro_geo_uw_confirmation"]),
        macro_geo_bundle.get("uw_confirmation", []),
        macro_geo_confirmation_fieldnames(),
    )
    write_csv(
        Path(paths["macro_geo_promotion_decisions"]),
        macro_geo_bundle.get("promotion_decisions", []),
        macro_geo_promotion_fieldnames(),
    )
    Path(paths["pattern_observability_matrix"]).write_text(
        render_pattern_observability_matrix(macro_geo_bundle.get("observability_rows", [])),
        encoding="utf-8",
    )
    Path(paths["missed_pattern_audit"]).write_text(
        render_missed_pattern_audit(
            macro_geo_bundle.get("missed_pattern_rows", []),
            source_completeness.get("missing_sources", []),
        ),
        encoding="utf-8",
    )
    write_csv(Path(paths["inventory"]), inventory_rows, inventory_fieldnames())
    write_csv(Path(paths["regime_sector_validation"]), validation_bundle["regime_sector"], regime_sector_fieldnames())
    write_json(Path(paths["metadata"]), metadata)
    write_json(
        Path(paths["artifact_manifest"]),
        build_artifact_manifest(
            as_of,
            base_dir,
            config,
            metadata,
            paths,
            schema_errors,
            runtime_seconds,
        ),
    )
    Path(paths["daily_report"]).write_text(
        render_daily_report(
            as_of,
            snapshots[as_of],
            actionable,
            trade_review,
            watch,
            blocked,
            discovered_rows,
            validation_bundle,
            missed_rows,
            sentiment_summary,
            source_completeness,
            macro_geo_bundle,
            metadata,
        ),
        encoding="utf-8",
    )
    return paths


def write_source_incomplete_outputs(
    out_dir: Path,
    base_dir: Path,
    as_of: str,
    config: Mapping[str, Any],
    inventory_rows: Sequence[Mapping[str, Any]],
    completeness: Mapping[str, Any],
    macro_geo_bundle: Mapping[str, Any],
    runtime_seconds: float,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "command": " ".join(sys.argv),
        "base_dir": str(base_dir),
        "as_of": as_of,
        "output_dir": str(out_dir),
        "deterministic_seed": config["seed"],
        "source_dates_used": [],
        "source_files_by_date": {},
        "skipped_sources_by_date": {},
        "source_counts_by_date": {},
        "source_files_for_as_of": [],
        "skipped_sources_for_as_of": [],
        "validation_splits": [],
        "daily_pattern_config": {},
        "input_policy": config["input_policy"],
        "source_completeness": completeness,
        "macro_geo_summary": macro_geo_bundle.get("summary", {}),
        "daily_trade_decision": "NO_TRADE",
        "run_kill_switches": ["KILL_SWITCH_SOURCE_INCOMPLETE"],
        "risk_config": config.get("risk_config", {}),
        "verdict": "BLOCKED_SOURCE_INCOMPLETE",
    }
    paths = {
        "daily_report": str(out_dir / f"daily_report_{as_of}.md"),
        "actionable_trades": str(out_dir / "actionable_trades.csv"),
        "watchlist_research_setups": str(out_dir / "watchlist_research_setups.csv"),
        "blocked_candidates": str(out_dir / "blocked_candidates.csv"),
        "discovered_pattern_families": str(out_dir / "discovered_pattern_families.csv"),
        "market_regime_summary": str(out_dir / "market_regime_summary.json"),
        "sentiment_news_summary": str(out_dir / "sentiment_news_summary.json"),
        "validation_scorecard": str(out_dir / "validation_scorecard.csv"),
        "validation_gate_scorecard": str(out_dir / "validation_gate_scorecard.csv"),
        "train_scorecard": str(out_dir / "train_scorecard.csv"),
        "validation_details": str(out_dir / "validation_details.csv"),
        "baseline_comparison": str(out_dir / "baseline_comparison.csv"),
        "missed_mover_audit": str(out_dir / "missed_mover_audit.csv"),
        "macro_geo_catalysts": str(out_dir / "macro_geo_catalysts.json"),
        "macro_geo_ticker_map": str(out_dir / "macro_geo_ticker_map.csv"),
        "macro_geo_uw_confirmation": str(out_dir / "macro_geo_uw_confirmation.csv"),
        "macro_geo_promotion_decisions": str(out_dir / "macro_geo_promotion_decisions.csv"),
        "trade_review_candidates": str(out_dir / "trade_review_candidates.csv"),
        "decision_board_json": str(out_dir / "decision_board.json"),
        "decision_board_csv": str(out_dir / "decision_board.csv"),
        "artifact_manifest": str(out_dir / "artifact_manifest.json"),
        "full_backtest_summary": str(out_dir / "full_backtest_summary.md"),
        "full_backtest_by_family": str(out_dir / "full_backtest_by_family.csv"),
        "full_backtest_by_month": str(out_dir / "full_backtest_by_month.csv"),
        "walk_forward_performance": str(out_dir / "walk_forward_performance.csv"),
        "threshold_sensitivity": str(out_dir / "threshold_sensitivity.csv"),
        "calibration_summary": str(out_dir / "calibration_summary.md"),
        "profitability_audit": str(out_dir / "profitability_audit.md"),
        "shadow_recommendation_ledger": str(out_dir / "shadow_recommendation_ledger.csv"),
        "shadow_outcome_summary": str(out_dir / "shadow_outcome_summary.md"),
        "ablation_attribution": str(out_dir / "ablation_attribution.csv"),
        "pattern_pipeline_runbook": str(out_dir / "pattern_pipeline_runbook.md"),
        "pattern_observability_matrix": str(out_dir / "pattern_observability_matrix.md"),
        "missed_pattern_audit": str(out_dir / "missed_pattern_audit.md"),
        "inventory": str(out_dir / "source_inventory.csv"),
        "metadata": str(out_dir / "metadata.json"),
        "regime_sector_validation": str(out_dir / "validation_by_regime_sector_ticker.csv"),
    }
    write_csv(Path(paths["actionable_trades"]), [], trade_fieldnames())
    write_csv(Path(paths["watchlist_research_setups"]), [], trade_fieldnames())
    write_csv(Path(paths["blocked_candidates"]), [], blocked_fieldnames())
    write_csv(Path(paths["trade_review_candidates"]), [], trade_review_fieldnames())
    write_csv(Path(paths["discovered_pattern_families"]), [], discovered_fieldnames())
    write_json(Path(paths["market_regime_summary"]), {"date": as_of, "regime": "UNKNOWN", "source": "source_incomplete"})
    write_json(
        Path(paths["sentiment_news_summary"]),
        {"summary": "Source incomplete; no single-date sentiment summary was built.", "used_sources": [], "skipped_sources": []},
    )
    write_csv(Path(paths["validation_scorecard"]), [], scorecard_fieldnames())
    write_csv(Path(paths["validation_gate_scorecard"]), [], scorecard_fieldnames())
    write_csv(Path(paths["train_scorecard"]), [], scorecard_fieldnames())
    write_csv(Path(paths["validation_details"]), [], validation_detail_fieldnames())
    write_csv(Path(paths["baseline_comparison"]), [], baseline_fieldnames())
    write_csv(Path(paths["missed_mover_audit"]), [], missed_fieldnames())
    decision_board = build_decision_board_rows([], as_of, False, "NO_TRADE", paths)
    if decision_board:
        decision_board[0]["kill_switch_triggered"] = "KILL_SWITCH_SOURCE_INCOMPLETE"
        decision_board[0]["blockers"] = "KILL_SWITCH_SOURCE_INCOMPLETE;NO_REVIEWABLE_TICKETS"
    schema_errors = validate_decision_board_rows(decision_board)
    write_json(
        Path(paths["decision_board_json"]),
        {
            "schema_version": DECISION_BOARD_SCHEMA_VERSION,
            "run_date": as_of,
            "daily_trade_decision": "NO_TRADE",
            "validation_errors": schema_errors,
            "rows": decision_board,
        },
    )
    write_csv(Path(paths["decision_board_csv"]), decision_board, decision_board_fieldnames())
    write_csv(Path(paths["full_backtest_by_family"]), [], full_backtest_family_fieldnames())
    write_csv(Path(paths["full_backtest_by_month"]), [], full_backtest_month_fieldnames())
    Path(paths["full_backtest_summary"]).write_text(
        render_full_backtest_summary(as_of, [], [], []),
        encoding="utf-8",
    )
    write_csv(Path(paths["walk_forward_performance"]), [], walk_forward_fieldnames())
    write_csv(Path(paths["threshold_sensitivity"]), build_threshold_sensitivity_rows(empty_validation_bundle(), config.get("risk_config", {})), threshold_sensitivity_fieldnames())
    write_csv(Path(paths["ablation_attribution"]), [], ablation_fieldnames())
    Path(paths["calibration_summary"]).write_text(
        render_calibration_summary(as_of, build_calibration_metrics(empty_validation_bundle())),
        encoding="utf-8",
    )
    Path(paths["profitability_audit"]).write_text(
        render_profitability_audit(
            as_of,
            metadata,
            decision_board,
            empty_validation_bundle(),
            build_threshold_sensitivity_rows(empty_validation_bundle(), config.get("risk_config", {})),
            build_calibration_metrics(empty_validation_bundle()),
            {"run_kill_switches": ["KILL_SWITCH_SOURCE_INCOMPLETE"]},
        ),
        encoding="utf-8",
    )
    write_csv(Path(paths["shadow_recommendation_ledger"]), [], shadow_ledger_fieldnames())
    Path(paths["shadow_outcome_summary"]).write_text(render_shadow_summary(as_of, []), encoding="utf-8")
    Path(paths["pattern_pipeline_runbook"]).write_text(render_runbook(as_of, config), encoding="utf-8")
    write_json(Path(paths["macro_geo_catalysts"]), macro_geo_bundle.get("catalysts", []))
    write_csv(Path(paths["macro_geo_ticker_map"]), macro_geo_bundle.get("ticker_map", []), macro_geo_ticker_map_fieldnames())
    write_csv(
        Path(paths["macro_geo_uw_confirmation"]),
        macro_geo_bundle.get("uw_confirmation", []),
        macro_geo_confirmation_fieldnames(),
    )
    write_csv(
        Path(paths["macro_geo_promotion_decisions"]),
        macro_geo_bundle.get("promotion_decisions", []),
        macro_geo_promotion_fieldnames(),
    )
    Path(paths["pattern_observability_matrix"]).write_text(
        render_pattern_observability_matrix(macro_geo_bundle.get("observability_rows", [])),
        encoding="utf-8",
    )
    Path(paths["missed_pattern_audit"]).write_text(
        render_missed_pattern_audit(
            macro_geo_bundle.get("missed_pattern_rows", []),
            completeness.get("missing_sources", []),
        ),
        encoding="utf-8",
    )
    write_csv(Path(paths["inventory"]), inventory_rows, inventory_fieldnames())
    write_csv(Path(paths["regime_sector_validation"]), [], regime_sector_fieldnames())
    write_json(Path(paths["metadata"]), metadata)
    write_json(
        Path(paths["artifact_manifest"]),
        build_artifact_manifest(
            as_of,
            base_dir,
            config,
            metadata,
            paths,
            schema_errors,
            runtime_seconds,
        ),
    )
    Path(paths["daily_report"]).write_text(
        render_source_incomplete_report(as_of, completeness, macro_geo_bundle, metadata),
        encoding="utf-8",
    )
    return paths


def trade_output_row(r: Mapping[str, Any]) -> Dict[str, Any]:
    setup = trade_setup_fields(r)
    return {
        "classification": r.get("classification"),
        "status": r.get("status"),
        "candidate_id": r.get("candidate_id"),
        "ticker": r.get("ticker"),
        "direction": r.get("direction"),
        "discovered_pattern_family": r.get("pattern_family"),
        "confidence_tier": r.get("confidence_tier"),
        "pattern_success_probability_pct": r.get("pattern_success_probability_pct"),
        "pattern_failure_probability_pct": r.get("pattern_failure_probability_pct"),
        "pattern_probability_score": r.get("pattern_probability_score"),
        "trade_success_probability_pct": r.get("trade_success_probability_pct"),
        "trade_failure_probability_pct": r.get("trade_failure_probability_pct"),
        "trade_probability_score": r.get("trade_probability_score"),
        "success_probability_pct": r.get("success_probability_pct"),
        "failure_probability_pct": r.get("failure_probability_pct"),
        "probability_score": r.get("probability_score"),
        "probability_evidence": r.get("probability_evidence"),
        "probability_components": r.get("probability_components"),
        "strategy": setup["strategy"],
        "buy_or_sell": setup["buy_or_sell"],
        "call_or_put": setup["call_or_put"],
        "strike_rates": setup["strike_rates"],
        "expiration_date": setup["expiration_date"],
        "trade_setup": setup["trade_setup"],
        "occ_symbols": setup["occ_symbols"],
        "suggested_entry_debit_credit_range": setup["entry_range"],
        "fill_model": r.get("fill_model"),
        "slippage_estimate": r.get("slippage_estimate"),
        "fees_commissions": r.get("fees_commissions"),
        "max_risk_per_contract": r.get("max_risk_per_contract"),
        "target_profit": r.get("target_profit"),
        "stop_invalidation_rule": r.get("stop_rule"),
        "time_stop": r.get("time_stop"),
        "invalidation": r.get("invalidation"),
        "review_close_rule": r.get("review_close_rule"),
        "expected_hold": r.get("expected_hold"),
        "expected_R": r.get("expected_R"),
        "expected_R_per_day": r.get("expected_R_per_day"),
        "avg_win_R": r.get("avg_win_R"),
        "avg_loss_R": r.get("avg_loss_R"),
        "payoff_ratio": r.get("payoff_ratio"),
        "calibrated_probability": r.get("calibrated_probability"),
        "validation_profit_factor": r.get("validation_profit_factor"),
        "capacity_estimate_contracts": r.get("capacity_estimate_contracts"),
        "live_quote_validation_status": r.get("live_quote_validation_status"),
        "position_size_tier": r.get("position_size_tier"),
        "catalyst_thesis": r.get("reason_summary"),
        "historical_evidence_summary": r.get("validation_note"),
        "edge_review_reason": r.get("edge_review_reason"),
        "edge_review_evidence": r.get("edge_review_evidence"),
        "current_market_regime_alignment": r.get("current_market_alignment"),
        "liquidity_quote_sanity": quote_sanity_text(r),
        "major_risks": r.get("major_risks"),
        "why_actionable_now": r.get("why_actionable_now"),
        "blocker_categories": ";".join(r.get("blocker_categories") or []),
        "block_reasons": ";".join(r.get("block_reasons") or []),
    }


def daily_trade_decision(
    actionable: Sequence[Mapping[str, Any]],
    trade_review: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]] = (),
) -> str:
    if actionable:
        return "AUTO_APPROVED"
    if trade_review:
        return "TRADE_REVIEW"
    return "NO_TRADE"


def build_trade_review_candidates(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    candidates_by_ticket: Dict[Tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("status") == "AUTO_APPROVED":
            continue
        if row.get("status") not in (None, "TRADE_REVIEW"):
            continue
        if trade_review_status(row) in {"NO_TRADE_NO_QUOTE", "NO_TRADE_BAD_CONTRACT"}:
            continue
        setup = trade_setup_fields(row)
        key = (
            str(row.get("ticker") or ""),
            str(row.get("direction") or ""),
            str(setup.get("trade_setup") or ""),
            str(setup.get("entry_range") or ""),
        )
        current = candidates_by_ticket.get(key)
        if current is None or trade_review_sort_key(row) > trade_review_sort_key(current):
            candidates_by_ticket[key] = row
    candidates = list(candidates_by_ticket.values())
    candidates.sort(
        key=trade_review_sort_key,
        reverse=True,
    )
    return candidates


def dedupe_rows_by_ticket(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    best_by_ticket: Dict[Tuple[str, str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = trade_ticket_key(row)
        current = best_by_ticket.get(key)
        if current is None or trade_review_sort_key(row) > trade_review_sort_key(current):
            best_by_ticket[key] = row
    out = list(best_by_ticket.values())
    out.sort(key=trade_review_sort_key, reverse=True)
    return out


def trade_ticket_key(row: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    setup = trade_setup_fields(row)
    return (
        str(row.get("ticker") or ""),
        str(row.get("direction") or ""),
        str(setup.get("trade_setup") or ""),
        str(setup.get("entry_range") or ""),
    )


def trade_review_sort_key(row: Mapping[str, Any]) -> Tuple[int, float, float, float]:
    return (
        trade_review_rank(trade_review_status(row)),
        num(row.get("probability_score")) or -1.0,
        num(row.get("success_probability_pct")) or -1.0,
        num(row.get("pattern_score")) or -1.0,
    )


def hard_trade_blockers(row: Mapping[str, Any]) -> List[str]:
    blockers = row.get("block_reasons") or []
    return [b for b in blockers if b in HARD_TRADE_BLOCKERS]


def trade_review_status(row: Mapping[str, Any]) -> str:
    if row.get("edge_review_reason"):
        return "VALIDATED_EDGE_REVIEW"
    blockers = set(row.get("block_reasons") or [])
    if "NO_TRADEABLE_OPTION_QUOTE" in blockers:
        return "NO_TRADE_NO_QUOTE"
    if blockers & HARD_TRADE_BLOCKERS:
        return "NO_TRADE_BAD_CONTRACT"
    if blockers & EV_TRADE_BLOCKERS:
        return "EV_REVIEW"
    if blockers & RISK_TRADE_BLOCKERS:
        return "RISK_REVIEW"
    if any(str(b).startswith(KILL_SWITCH_PREFIX) for b in blockers):
        return "KILL_SWITCH_REVIEW"
    score = num(row.get("probability_score")) or 0.0
    success = num(row.get("success_probability_pct")) or 0.0
    tier = str(row.get("confidence_tier") or "")
    if blockers & REGIME_TRADE_BLOCKERS:
        return "MACRO_CONFLICT_REVIEW"
    if tier == "PROMISING" and (score >= 40.0 or success >= 50.0):
        return "PROMISING_REVIEW"
    if score >= 45.0 or success >= 55.0:
        return "TACTICAL_REVIEW"
    if blockers & VALIDATION_TRADE_BLOCKERS:
        return "VALIDATION_REVIEW"
    return "RESEARCH_REVIEW"


def trade_review_rank(status: str) -> int:
    return {
        "VALIDATED_EDGE_REVIEW": 6,
        "TACTICAL_REVIEW": 5,
        "PROMISING_REVIEW": 4,
        "MACRO_CONFLICT_REVIEW": 3,
        "EV_REVIEW": 3,
        "RISK_REVIEW": 3,
        "KILL_SWITCH_REVIEW": 3,
        "VALIDATION_REVIEW": 2,
        "RESEARCH_REVIEW": 1,
        "NO_TRADE_BAD_CONTRACT": 0,
        "NO_TRADE_NO_QUOTE": 0,
    }.get(status, 0)


def promotion_needed_text(row: Mapping[str, Any]) -> str:
    blockers = set(row.get("block_reasons") or [])
    needs: List[str] = []
    edge_review_reason = str(row.get("edge_review_reason") or "")
    if blockers & REGIME_TRADE_BLOCKERS and edge_review_reason:
        needs.append("confirm current catalyst still matches validated regime edge")
    elif blockers & REGIME_TRADE_BLOCKERS:
        needs.append("regime alignment or explicit hedge thesis")
    if "VALIDATION_EXPECTANCY_NEGATIVE" in blockers:
        needs.append("positive out-of-sample expectancy")
    if "PATTERN_VALIDATION_NOT_PROVEN" in blockers:
        needs.append("pattern family upgraded to proven")
    if "LIMITED_OUT_OF_SAMPLE_SAMPLE" in blockers:
        needs.append("larger scored OOS sample")
    if "DOES_NOT_BEAT_TWO_BASELINES" in blockers:
        needs.append("edge over at least two baselines")
    if "PROFIT_FACTOR_BELOW_AUTO_APPROVAL" in blockers:
        needs.append("profit factor at or above approval threshold")
    if "EXPECTED_R_NOT_POSITIVE_AFTER_COSTS" in blockers:
        needs.append("positive expected R after costs")
    if "EXPECTED_R_PER_DAY_NOT_POSITIVE" in blockers:
        needs.append("positive expected R per day")
    if "CALIBRATION_SCORE_MISSING_OR_WEAK" in blockers or "CONFIDENCE_BAND_TOO_WEAK" in blockers:
        needs.append("better calibration/confidence band")
    if "MAX_RISK_EXCEEDS_PER_TRADE_LIMIT" in blockers:
        needs.append("smaller defined-risk ticket")
    if any(str(b).startswith(KILL_SWITCH_PREFIX) for b in blockers):
        needs.append("kill-switch cleared")
    if "NEAR_TERM_EARNINGS_EVENT_RISK" in blockers:
        needs.append("earnings/event-risk plan")
    if blockers & HARD_TRADE_BLOCKERS:
        needs.append("clean quote/liquidity/DTE")
    return "; ".join(needs[:5]) or "manual desk review"


def trade_review_output_row(r: Mapping[str, Any]) -> Dict[str, Any]:
    row = trade_output_row(r)
    row.update(
        {
            "review_status": trade_review_status(r),
            "review_priority": trade_review_rank(trade_review_status(r)),
            "promotion_needed": promotion_needed_text(r),
            "hard_blockers": ";".join(hard_trade_blockers(r)),
        }
    )
    return row


def trade_setup_fields(r: Mapping[str, Any]) -> Dict[str, str]:
    strategy = str(r.get("strategy_type") or "")
    strategy_kind = str(r.get("strategy_kind") or "long_option")
    ticker = str(r.get("ticker") or "")
    expiry = str(r.get("expiry") or "")
    if strategy_kind == "credit_spread":
        return spread_trade_setup_fields(r, strategy, ticker, expiry)
    symbol = str(r.get("lead_option_symbol") or "")
    parsed = parse_option_symbol(symbol)
    option_type = str(r.get("option_type") or (parsed or {}).get("option_type") or "").upper()
    strike = r.get("strike")
    if strike in (None, "") and parsed:
        strike = parsed.get("strike")
    action = "BUY"
    strike_text = format_strike(strike)
    if not option_type or not strike_text or not expiry:
        trade_setup = "No complete option ticket - missing quote/strike/expiry"
    else:
        trade_setup = f"{action} {option_type} {ticker} {strike_text} exp {expiry}".strip()
    return {
        "strategy": strategy,
        "buy_or_sell": action,
        "call_or_put": option_type,
        "strike_rates": strike_text,
        "expiration_date": expiry,
        "trade_setup": trade_setup,
        "occ_symbols": symbol,
        "entry_range": format_entry_for_output(r),
    }


def spread_trade_setup_fields(
    r: Mapping[str, Any],
    strategy: str,
    ticker: str,
    expiry: str,
) -> Dict[str, str]:
    legs = parse_legs(r.get("legs_json", ""))
    parts: List[str] = []
    strike_parts: List[str] = []
    actions: List[str] = []
    option_types: List[str] = []
    occ_symbols: List[str] = []
    for leg in legs:
        action = str(leg.get("action") or "").upper()
        symbol = str(leg.get("option_symbol") or "")
        parsed = parse_option_symbol(symbol)
        option_type = str(leg.get("option_type") or (parsed or {}).get("option_type") or "").upper()
        strike = leg.get("strike")
        if strike in (None, "") and parsed:
            strike = parsed.get("strike")
        strike_text = format_strike(strike)
        if action:
            actions.append(action)
        if option_type:
            option_types.append(option_type)
        if symbol:
            occ_symbols.append(symbol)
        parts.append(f"{action} {option_type} {ticker} {strike_text}".strip())
        strike_parts.append(f"{action} {strike_text}".strip())
    trade_setup = " / ".join(parts)
    if not trade_setup:
        trade_setup = "No complete spread ticket - missing legs/quote"
    elif expiry:
        trade_setup = f"{trade_setup} exp {expiry}"
    return {
        "strategy": strategy,
        "buy_or_sell": " / ".join(actions),
        "call_or_put": " / ".join(option_types),
        "strike_rates": " / ".join(strike_parts),
        "expiration_date": expiry,
        "trade_setup": trade_setup,
        "occ_symbols": " / ".join(occ_symbols),
        "entry_range": format_entry_for_output(r),
    }


def format_entry_for_output(r: Mapping[str, Any]) -> str:
    if r.get("strategy_kind") == "credit_spread":
        credit = num(r.get("entry_credit"))
        return f"credit {credit:.2f}" if credit is not None else ""
    entry = str(r.get("entry_range") or "")
    return f"debit {entry}" if entry else ""


def format_strike(value: Any) -> str:
    parsed = num(value)
    if parsed is None:
        return ""
    text = f"{parsed:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def blocked_output_row(r: Mapping[str, Any]) -> Dict[str, Any]:
    row = trade_output_row(r)
    row.update(
        {
            "pattern_score": r.get("pattern_score"),
            "bid_ask_spread_pct": r.get("bid_ask_spread_pct"),
            "liquidity_volume": r.get("liquidity_volume"),
            "liquidity_open_interest": r.get("liquidity_open_interest"),
        }
    )
    return row


def quote_sanity_text(r: Mapping[str, Any]) -> str:
    return (
        f"bid={fmt_num(r.get('entry_bid'))} ask={fmt_num(r.get('entry_ask'))} "
        f"spread_pct={fmt_num(r.get('bid_ask_spread_pct'))} "
        f"volume={fmt_num(r.get('liquidity_volume'))} oi={fmt_num(r.get('liquidity_open_interest'))}"
    )


def append_run_observability_summary(
    lines: List[str],
    actionable: Sequence[Mapping[str, Any]],
    source_completeness: Mapping[str, Any],
    macro_geo_bundle: Mapping[str, Any],
) -> None:
    summary = macro_geo_bundle.get("summary", {})
    source_complete = bool(source_completeness.get("source_complete"))
    lines.append("## Run Observability Summary")
    lines.append(f"- Source data complete: {'yes' if source_complete else 'no'}.")
    if source_completeness.get("missing_sources"):
        for missing in source_completeness.get("missing_sources", [])[:8]:
            lines.append(f"- Missing source data: {missing}")
    lines.append(f"- Eligible catalysts existed: {'yes' if summary.get('eligible_catalyst_count') else 'no'}.")
    lines.append(
        f"- Future-dated catalysts skipped: {'yes' if summary.get('future_dated_catalyst_count') else 'no'} "
        f"({summary.get('future_dated_catalyst_count', 0)})."
    )
    lines.append(
        f"- UW confirmed catalyst themes: {summary.get('uw_confirmed_themes') or 'none'}."
    )
    if actionable:
        lines.append(f"- Approved trades: {len(actionable)}.")
    else:
        reason = summary.get("primary_no_trade_reason") or "no setup cleared validation, quote, liquidity, regime, and event checks"
        lines.append(f"- Approved trades: 0 because {reason}.")
    lines.append(f"- Watch/blocked names that matter: {summary.get('watch_or_blocked_names') or 'none'}.")


def markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("\n", " ").replace("|", "/").strip()


def money_text(value: Any) -> str:
    parsed = num(value)
    if parsed is None:
        return ""
    return f"${parsed:,.2f}"


def pct_text(value: Any) -> str:
    parsed = num(value)
    if parsed is None:
        return ""
    return f"{f'{parsed:.2f}'.rstrip('0').rstrip('.')}%"


def count_text(value: Any) -> str:
    parsed = num(value)
    if parsed is None:
        return "n/a"
    return f"{int(parsed):,}"


def strategy_plain_english(strategy: str) -> str:
    normalized = str(strategy or "").strip().lower()
    if normalized == "long put debit":
        return "Buy a put; bearish; max loss is the debit paid."
    if normalized == "long call debit":
        return "Buy a call; bullish; max loss is the debit paid."
    if normalized == "bear call credit spread":
        return "Sell lower-strike call, buy higher-strike call; bearish/neutral credit spread."
    if normalized == "bull put credit spread":
        return "Sell higher-strike put, buy lower-strike put; bullish/neutral credit spread."
    if normalized:
        return "Option structure shown in the ticket legs."
    return ""


def blocker_text(row: Mapping[str, Any]) -> str:
    blockers = row.get("blocker_categories") or row.get("block_reasons") or []
    if isinstance(blockers, str):
        parts = [p.strip() for p in blockers.split(";") if p.strip()]
    else:
        parts = [str(p).strip() for p in blockers if str(p).strip()]
    return "; ".join(parts[:5])


def ticket_row(
    row: Mapping[str, Any],
    status: str,
    index: int,
) -> str:
    setup = trade_setup_fields(row)
    why = "All configured auto-approval gates passed" if status == "AUTO_APPROVED" else blocker_text(row) or row.get("why_actionable_now") or ""
    return (
        f"| {index} | {status} | {markdown_cell(row.get('ticker'))} | {markdown_cell(row.get('direction'))} | "
        f"{markdown_cell(setup.get('trade_setup'))} | {markdown_cell(setup.get('strategy'))} | "
        f"{markdown_cell(strategy_plain_english(setup.get('strategy', '')))} | {markdown_cell(setup.get('entry_range'))} | "
        f"{money_text(row.get('max_risk_per_contract'))} | {pct_text(row.get('success_probability_pct'))} | "
        f"{pct_text(row.get('probability_score'))} | {markdown_cell(why)} |"
    )


def append_ticket_table(
    lines: List[str],
    title: str,
    rows: Sequence[Mapping[str, Any]],
    status: str,
    limit: int,
    empty_text: str,
) -> None:
    lines.append(f"## {title}")
    if not rows:
        lines.append(f"- {empty_text}")
        lines.append("")
        return
    lines.append("| # | Status | Ticker | Bias | Full Ticket | Strategy | Meaning | Entry | Max Risk | Success | Score | Why |")
    lines.append("|---:|---|---|---|---|---|---|---:|---:|---:|---:|---|")
    for idx, row in enumerate(rows[:limit], 1):
        lines.append(ticket_row(row, status, idx))
    lines.append("")


def trade_review_row(row: Mapping[str, Any], index: int) -> str:
    setup = trade_setup_fields(row)
    edge_evidence = row.get("edge_review_evidence") or f"expected R {fmt_num(row.get('expected_R'))}; PF {fmt_num(row.get('validation_profit_factor'))}"
    return (
        f"| {index} | {markdown_cell(trade_review_status(row))} | {markdown_cell(row.get('ticker'))} | "
        f"{markdown_cell(row.get('direction'))} | {markdown_cell(setup.get('trade_setup'))} | "
        f"{markdown_cell(setup.get('entry_range'))} | {money_text(row.get('max_risk_per_contract'))} | "
        f"{pct_text(row.get('success_probability_pct'))} | {pct_text(row.get('probability_score'))} | "
        f"{markdown_cell(edge_evidence)} | {markdown_cell(blocker_text(row))} | {markdown_cell(promotion_needed_text(row))} |"
    )


def append_trade_review_table(
    lines: List[str],
    rows: Sequence[Mapping[str, Any]],
    limit: int = 12,
) -> None:
    lines.append("## Trade-Review Board")
    lines.append("- Not auto-approved. This is the ranked desk-review board so a zero-TRADE day still shows the best live setups.")
    if not rows:
        lines.append("- No reviewable tickets with complete quotes/liquidity.")
        lines.append("")
        return
    lines.append("| # | Review | Ticker | Bias | Full Ticket | Entry | Max Risk | Success | Score | Edge Evidence | What's Wrong | Promotion Needed |")
    lines.append("|---:|---|---|---|---|---:|---:|---:|---:|---|---|---|")
    for idx, row in enumerate(rows[:limit], 1):
        lines.append(trade_review_row(row, idx))
    lines.append("")


def append_strategy_glossary(lines: List[str], rows: Sequence[Mapping[str, Any]]) -> None:
    strategies: Dict[str, str] = {}
    for row in rows:
        setup = trade_setup_fields(row)
        strategy = str(setup.get("strategy") or "").strip()
        if strategy and strategy not in strategies:
            strategies[strategy] = strategy_plain_english(strategy)
    if not strategies:
        return
    lines.append("## Strategy Glossary")
    for strategy, meaning in sorted(strategies.items()):
        lines.append(f"- {strategy}: {meaning}")
    lines.append("")


def validation_snapshot_rows(discovered_rows: Sequence[Mapping[str, Any]], limit: int = 12) -> List[Mapping[str, Any]]:
    tier_rank = {"PROVEN": 3, "PROMISING": 2, "RESEARCH_ONLY": 1}
    rows = [
        row
        for row in discovered_rows
        if str(row.get("confidence_tier") or "") in {"PROVEN", "PROMISING"}
    ]
    rows.sort(
        key=lambda row: (
            tier_rank.get(str(row.get("confidence_tier") or ""), 0),
            num(row.get("validation_probability_score")) or -1.0,
            num(row.get("validation_average_net_r")) or -999.0,
        ),
        reverse=True,
    )
    return rows[:limit]


def render_source_incomplete_report(
    as_of: str,
    completeness: Mapping[str, Any],
    macro_geo_bundle: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    lines.append(f"# Options Pattern Pipeline v1 Daily Report - {as_of}")
    lines.append("")
    lines.append("Final pipeline verdict: **BLOCKED_SOURCE_INCOMPLETE**")
    lines.append("")
    append_run_observability_summary(lines, [], completeness, macro_geo_bundle)
    lines.append("")
    lines.append("## Source Incomplete")
    lines.append("The run was not promoted because required point-in-time UW source files are missing.")
    for missing in completeness.get("missing_sources", []):
        lines.append(f"- {missing}")
    lines.append("")
    lines.append("## Catalyst Audit")
    summary = macro_geo_bundle.get("summary", {})
    lines.append(f"- Eligible catalysts: {', '.join(summary.get('eligible_event_types') or []) or 'none'}.")
    lines.append(
        f"- Future-dated catalysts skipped: {summary.get('future_dated_catalyst_count', 0)} "
        f"({', '.join(summary.get('future_dated_event_types') or []) or 'none'})."
    )
    lines.append(f"- UW-confirmed catalyst themes: {summary.get('uw_confirmed_themes') or 'none'}.")
    lines.append("")
    lines.append("## Scenario Buckets")
    for row in macro_geo_bundle.get("promotion_decisions", [])[:20]:
        blocker = f" | blocker: {row.get('promotion_blocker')}" if row.get("promotion_blocker") else ""
        lines.append(f"- {row.get('scenario_bucket')} {row.get('ticker') or 'run'} {row.get('event_type') or ''}{blocker}")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append(f"- Deterministic seed: {metadata['deterministic_seed']}")
    lines.append("- No pattern validation or trade promotion was run because source data was incomplete.")
    lines.append("")
    return "\n".join(lines)


def render_daily_report(
    as_of: str,
    snapshot: Snapshot,
    actionable: Sequence[Mapping[str, Any]],
    trade_review: Sequence[Mapping[str, Any]],
    watch: Sequence[Mapping[str, Any]],
    blocked: Sequence[Mapping[str, Any]],
    discovered_rows: Sequence[Mapping[str, Any]],
    validation_bundle: Mapping[str, Any],
    missed_rows: Sequence[Mapping[str, Any]],
    sentiment_summary: Mapping[str, Any],
    source_completeness: Mapping[str, Any],
    macro_geo_bundle: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    verdict = metadata["verdict"]
    macro_summary = macro_geo_bundle.get("summary", {})
    no_trade_reason = macro_summary.get("primary_no_trade_reason") or (
        "No setup cleared validation, quote, liquidity, event, and regime checks."
    )
    source_counts = (metadata.get("source_counts_by_date") or {}).get(as_of, {})
    lines.append(f"# Options Pattern Pipeline v1 Daily Report - {as_of}")
    lines.append("")
    lines.append(f"Final pipeline verdict: **{verdict}**")
    lines.append(f"Daily trade decision: **{metadata.get('daily_trade_decision', 'UNKNOWN')}**")
    lines.append("")
    regime = snapshot.market_regime
    lines.append("## Decision Summary")
    lines.append(f"- Approved trades: {len(actionable)}.")
    lines.append(f"- Trade-review candidates: {len(trade_review)}.")
    if not actionable:
        lines.append(f"- No-trade reason: {no_trade_reason}")
    lines.append(
        f"- Regime: {regime.get('regime')} | index return {fmt_pct(regime.get('avg_index_return'))} | "
        f"positive breadth {fmt_pct(regime.get('breadth_positive_pct'))}."
    )
    lines.append(f"- Source data complete: {'yes' if source_completeness.get('source_complete') else 'no'}.")
    lines.append(
        f"- Bot-EOD: {count_text(source_counts.get('bot_eod_rows'))} flow rows, "
        f"{count_text(source_counts.get('bot_eod_quote_rows'))} quote rows, "
        f"cache {'hit' if source_counts.get('bot_eod_cache_hit') else 'built' if source_counts.get('bot_eod_cache_built') else 'n/a'}."
    )
    lines.append(
        f"- Catalyst buckets: {macro_summary.get('primary_no_trade_reason') or 'see Macro/Catalyst Read'}."
    )
    lines.append("")

    append_ticket_table(lines, "AUTO_APPROVED Trade Tickets", actionable, "AUTO_APPROVED", 10, "No auto-approved trade tickets.")
    append_trade_review_table(lines, trade_review, 12)
    append_ticket_table(lines, "Trade-Review Watch Tickets", watch, "TRADE_REVIEW", 15, "No review watch tickets passed the daily screens.")
    append_ticket_table(lines, "Avoid / Blocked Tickets", blocked, "AVOID", 15, "No blocked tickets after daily classification.")
    append_strategy_glossary(lines, list(actionable[:10]) + list(trade_review[:12]) + list(watch[:15]) + list(blocked[:15]))

    lines.append("## Macro / Catalyst Read")
    eligible_types = macro_summary.get("eligible_event_types") or []
    lines.append(f"- Eligible catalysts: {', '.join(eligible_types[:12]) or 'none'}.")
    lines.append(f"- UW-confirmed themes: {macro_summary.get('uw_confirmed_themes') or 'none'}.")
    lines.append(
        f"- Scenario counts: {macro_summary.get('primary_no_trade_reason') or 'not provided'}."
    )
    lines.append(f"- News/capture note: {sentiment_summary.get('summary')}")
    if sentiment_summary.get("used_sources"):
        for src in sentiment_summary["used_sources"][:3]:
            urls = ", ".join(src.get("urls") or [])
            lines.append(f"- Capture: `{src['path']}`" + (f" | {urls}" if urls else ""))
    lines.append("")
    lines.append("## Catalyst Promotion / Scenario Buckets")
    promotion_rows = macro_geo_bundle.get("promotion_decisions", [])
    if promotion_rows:
        for row in promotion_rows[:12]:
            ticker = row.get("ticker") or "run"
            blocker = f" | blocker: {row.get('promotion_blocker')}" if row.get("promotion_blocker") else ""
            evidence = f" | UW: {row.get('uw_evidence_found')}" if row.get("uw_evidence_found") else ""
            lines.append(
                f"- {row.get('scenario_bucket')} {ticker} {row.get('event_type') or ''}{evidence}{blocker}"
            )
    else:
        lines.append("- No macro/geopolitical scenario rows were produced.")
    lines.append("")
    lines.append("## Validation Snapshot")
    compact_validation_rows = validation_snapshot_rows(discovered_rows)
    if compact_validation_rows:
        for row in compact_validation_rows:
            lines.append(
                f"- {row['pattern_family']}: {row['confidence_tier']} | "
                f"success {fmt_pct(row.get('validation_success_probability'))}, "
                f"score {fmt_pct(row.get('validation_probability_score'))}, "
                f"avg R {fmt_num(row.get('validation_average_net_r'))}, "
                f"scored {row.get('validation_scored_count')}, "
                f"beats {row.get('beats_baselines_count')} baselines."
            )
    else:
        lines.append("- No proven or promising pattern families were produced.")
    lines.append("- Full validation detail is in `validation_scorecard.csv` and `discovered_pattern_families.csv`.")
    lines.append("")
    lines.append("## Baseline Comparison")
    for row in validation_bundle.get("baseline_comparison", []):
        lines.append(
            f"- {row['baseline']}: avg R {fmt_num(row.get('average_net_r'))}, "
            f"scored {row.get('scored_count')}, win rate {fmt_pct(row.get('win_rate_scored'))}"
        )
    lines.append("")
    lines.append("## Missed-Mover Lessons")
    for row in missed_rows[:10]:
        lines.append(
            f"- {row['signal_date']} {row['ticker']} next-day move {fmt_pct(row['next_day_stock_move'])}: "
            f"{row['likely_miss_reason']}"
        )
    if not missed_rows:
        lines.append("- Missed-mover audit had no scorable next-day stock moves.")
    lines.append("")
    lines.append("## Source / Reproducibility")
    lines.append(f"- Source dates used: {metadata['source_dates_used'][0]} through {metadata['source_dates_used'][-1]}")
    lines.append(f"- Validation splits: {len(metadata['validation_splits'])}")
    lines.append(f"- Deterministic seed: {metadata['deterministic_seed']}")
    lines.append("- Unscorable option outcomes are not counted as wins.")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_value(row.get(k)) for k in fieldnames})


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=json_default) + "\n", encoding="utf-8")


def trade_fieldnames() -> List[str]:
    return [
        "classification",
        "status",
        "candidate_id",
        "ticker",
        "direction",
        "discovered_pattern_family",
        "confidence_tier",
        "pattern_success_probability_pct",
        "pattern_failure_probability_pct",
        "pattern_probability_score",
        "trade_success_probability_pct",
        "trade_failure_probability_pct",
        "trade_probability_score",
        "success_probability_pct",
        "failure_probability_pct",
        "probability_score",
        "probability_evidence",
        "probability_components",
        "strategy",
        "buy_or_sell",
        "call_or_put",
        "strike_rates",
        "expiration_date",
        "trade_setup",
        "occ_symbols",
        "suggested_entry_debit_credit_range",
        "fill_model",
        "slippage_estimate",
        "fees_commissions",
        "max_risk_per_contract",
        "target_profit",
        "stop_invalidation_rule",
        "time_stop",
        "invalidation",
        "review_close_rule",
        "expected_hold",
        "expected_R",
        "expected_R_per_day",
        "avg_win_R",
        "avg_loss_R",
        "payoff_ratio",
        "calibrated_probability",
        "validation_profit_factor",
        "capacity_estimate_contracts",
        "live_quote_validation_status",
        "position_size_tier",
        "catalyst_thesis",
        "historical_evidence_summary",
        "edge_review_reason",
        "edge_review_evidence",
        "current_market_regime_alignment",
        "liquidity_quote_sanity",
        "major_risks",
        "why_actionable_now",
        "blocker_categories",
        "block_reasons",
    ]


def blocked_fieldnames() -> List[str]:
    return trade_fieldnames() + ["pattern_score", "bid_ask_spread_pct", "liquidity_volume", "liquidity_open_interest"]


def trade_review_fieldnames() -> List[str]:
    return [
        "review_status",
        "review_priority",
        "promotion_needed",
        "hard_blockers",
    ] + trade_fieldnames()


def discovered_fieldnames() -> List[str]:
    return [
        "pattern_family",
        "confidence_tier",
        "validation_signal_count",
        "validation_scored_count",
        "validation_win_count",
        "validation_success_probability",
        "validation_failure_probability",
        "validation_probability_score",
        "validation_average_net_r",
        "validation_profit_factor",
        "beats_baselines_count",
        "validation_split_count",
        "positive_validation_splits",
        "worst_split_average_net_r",
        "max_worst_losing_streak",
        "probability_evidence",
        "validation_note",
        "daily_pattern_config_json",
    ]


def scorecard_fieldnames() -> List[str]:
    return [
        "split",
        "sample",
        "pattern_family",
        "signal_count",
        "scored_count",
        "partial_count",
        "unscorable_count",
        "win_count_scored",
        "win_rate_scored",
        "win_rate_all_counting_unscorable_losses",
        "average_net_r",
        "median_net_r",
        "profit_factor",
        "worst_losing_streak",
        "drawdown_proxy_r",
        "tradeable_with_real_quotes_pct",
        "average_bid_ask_spread",
        "blocked_pct",
        "top_block_reasons",
    ]


def validation_detail_fieldnames() -> List[str]:
    return [
        "split",
        "sample",
        "horizon",
        "signal_date",
        "target_date",
        "ticker",
        "direction",
        "pattern_family",
        "market_regime",
        "sector",
        "lead_option_symbol",
        "strategy_kind",
        "strategy_type",
        "legs_json",
        "entry_credit",
        "entry_ask",
        "entry_bid",
        "bid_ask_spread_pct",
        "blocked",
        "block_reasons",
        "status",
        "net_r",
        "win",
        "outcome_note",
        "exit_bid",
        "exit_debit",
        "exit_proxy",
        "managed_exit_date",
        "managed_exit_price",
        "stock_proxy_move",
    ]


def baseline_fieldnames() -> List[str]:
    return ["baseline", "seed", "signal_count", "scored_count", "win_rate_scored", "average_net_r", "median_net_r", "note"]


def missed_fieldnames() -> List[str]:
    return [
        "signal_date",
        "next_date",
        "ticker",
        "next_day_stock_move",
        "abs_next_day_stock_move",
        "sector",
        "was_flagged_by_new_pipeline",
        "likely_miss_reason",
        "call_volume_ratio_30d",
        "put_volume_ratio_30d",
        "premium_bias",
        "hot_total_premium",
    ]


def inventory_fieldnames() -> List[str]:
    return ["date", "relative_path", "absolute_path", "bytes", "source_like", "used_by_features", "ignored_reason"]


def regime_sector_fieldnames() -> List[str]:
    return ["pattern_family", "dimension", "value", "signal_count", "scored_count", "average_net_r", "win_rate_scored"]


def flatten_signal(sig: Mapping[str, Any]) -> Dict[str, Any]:
    row = dict(sig)
    if isinstance(row.get("block_reasons"), list):
        row["block_reasons"] = ";".join(row["block_reasons"])
    return row


def final_verdict(validation_bundle: Mapping[str, Any], daily_rows: Sequence[Mapping[str, Any]]) -> str:
    tiers = [v.get("confidence_tier") for v in validation_bundle.get("family_tiers", {}).values()]
    if any(t == "PROVEN" for t in tiers):
        return "PRODUCTION_READY"
    if any(t == "PROMISING" for t in tiers) or validation_bundle.get("validation_scorecard"):
        return "USABLE_NEEDS_MORE_VALIDATION" if any(t == "PROMISING" for t in tiers) else "NOT_YET_PROVEN"
    return "NOT_YET_PROVEN"


def parse_option_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    raw = str(symbol or "").upper()
    compact = re.sub(r"\s+", "", raw)
    m = re.match(r"^(.+?)(\d{6})([CP])(\d{8})$", compact)
    if not m:
        return None
    root, yymmdd, cp, strike_raw = m.groups()
    try:
        yy = int(yymmdd[:2])
        year = 2000 + yy if yy < 80 else 1900 + yy
        expiry = date(year, int(yymmdd[2:4]), int(yymmdd[4:6])).isoformat()
        strike = int(strike_raw) / 1000.0
    except ValueError:
        return None
    return {
        "option_symbol": compact,
        "ticker": clean_ticker(root),
        "expiry": expiry,
        "option_type": "call" if cp == "C" else "put",
        "strike": strike,
    }


def clean_ticker(value: str) -> str:
    return re.sub(r"[^A-Z0-9.\-]", "", str(value or "").upper())


def num(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isnan(value) or math.isinf(value):
            return None
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        val = float(text)
    except ValueError:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def safe_div(numer: Any, denom: Any) -> Optional[float]:
    n = num(numer)
    d = num(denom)
    if n is None or d in (None, 0):
        return None
    return n / d


def first_positive(values: Iterable[Any]) -> Optional[float]:
    for value in values:
        parsed = num(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def pct_change(current: Any, previous: Any) -> Optional[float]:
    c = num(current)
    p = num(previous)
    if c is None or p in (None, 0):
        return None
    return (c - p) / p


def quantile(values: Sequence[float], q: float, default: float) -> float:
    clean = sorted(v for v in values if v is not None and not math.isnan(v))
    if not clean:
        return default
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[int(pos)]
    frac = pos - lo
    return clean[lo] * (1 - frac) + clean[hi] * frac


def zish(value: Any, threshold: Any) -> float:
    v = num(value) or 0.0
    t = num(threshold) or 1.0
    if t <= 0:
        return 0.0
    return max(0.0, min(5.0, (v / t) - 1.0))


def trading_day_delta(start: str, end: str) -> Optional[int]:
    try:
        start_d = date.fromisoformat(start)
        end_d = date.fromisoformat(end)
    except ValueError:
        return None
    return max(0, (end_d - start_d).days)


def calendar_day_delta(start: str, end: str) -> Optional[int]:
    if not end:
        return None
    try:
        return (date.fromisoformat(end[:10]) - date.fromisoformat(start)).days
    except ValueError:
        return None


def format_entry_range(bid: Any, ask: Any) -> str:
    b = num(bid)
    a = num(ask)
    if b is None and a is None:
        return ""
    if b is None:
        return f"up to {a:.2f}"
    if a is None:
        return f"{b:.2f}+"
    return f"{b:.2f}-{a:.2f}"


def worst_losing_streak(net_rs: Sequence[float]) -> int:
    worst = 0
    current = 0
    for r in net_rs:
        if r <= 0:
            current += 1
            worst = max(worst, current)
        else:
            current = 0
    return worst


def drawdown_proxy(net_rs: Sequence[float]) -> Optional[float]:
    if not net_rs:
        return None
    equity = 0.0
    peak = 0.0
    worst = 0.0
    for r in net_rs:
        equity += r
        peak = max(peak, equity)
        worst = min(worst, equity - peak)
    return worst


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=json_default)


def parse_legs(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return [dict(v) for v in value if isinstance(v, Mapping)]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [dict(v) for v in parsed if isinstance(v, Mapping)]


def json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        return ";".join(str(v) for v in value)
    if isinstance(value, dict):
        return stable_json(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return value
    if value is None:
        return ""
    return value


def fmt_num(value: Any) -> str:
    v = num(value)
    if v is None:
        return "n/a"
    return f"{v:.4g}"


def pct_value(value: Any) -> Optional[float]:
    v = num(value)
    if v is None:
        return None
    return round(v * 100.0, 2)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def fmt_signed_pct(value: Any) -> str:
    v = num(value)
    if v is None:
        return "n/a"
    return f"{v * 100.0:+.2f}%"


def fmt_pct(value: Any) -> str:
    v = num(value)
    if v is None:
        return "n/a"
    return f"{v * 100:.2f}%"


def baseline_note(name: str) -> str:
    notes = {
        "BASELINE_RANDOM_SAME_DATE_LIQUIDITY": "Deterministic random ticker/option baseline from same date and approximate liquidity.",
        "BASELINE_SPY_QQQ_DIRECTIONAL": "Simple SPY/QQQ same-day direction option baseline.",
        "BASELINE_UNUSUAL_VOLUME_ONLY": "Ranks options volume expansion without full pattern/risk gates.",
        "BASELINE_NAIVE_UW_FLOW_ONLY": "Ranks raw UW flow premium/bias without full pattern/risk gates.",
        "BASELINE_NAIVE_CATALYST_ONLY": "Ranks naive catalyst/event proximity and premium concentration without full pattern/risk gates.",
        "BASELINE_SIMPLE_PRICE_MOMENTUM": "Ranks same-day stock momentum with available option quotes.",
    }
    return notes.get(name, "")
