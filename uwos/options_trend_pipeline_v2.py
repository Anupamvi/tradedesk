#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from uwos import trend_analysis_v2 as engine


DEFAULT_ROOT = engine.DEFAULT_ROOT
DEFAULT_LOOKBACK = engine.DEFAULT_LOOKBACK
DEFAULT_VALIDATE_DAYS = engine.DEFAULT_VALIDATE_DAYS
DEFAULT_HORIZONS = engine.DEFAULT_HORIZONS

CANONICAL_ARTIFACT_NAMES = {
    "report": "daily_report.md",
    "actionable_csv": "actionable_trades.csv",
    "watchlist_csv": "watchlist_research_setups.csv",
    "blocked_csv": "blocked_candidates.csv",
    "candidates_csv": "all_candidates.csv",
    "regime_csv": "market_regime.csv",
    "sector_rotation_csv": "sector_rotation.csv",
    "sentiment_news_csv": "sentiment_news_summary.csv",
    "validation_scorecard_csv": "validation_scorecard.csv",
    "validation_outcomes_csv": "validation_outcomes.csv",
    "missed_movers_csv": "missed_mover_audit.csv",
    "metadata_json": "engine_metadata.json",
}

REQUIRED_BASELINES = {
    "legacy_trend_analysis": "current_old_pipeline",
    "random_same_date_liquidity": "random_same_date_same_liquidity",
    "spy_qqq_directional": "spy_qqq_directional",
    "unusual_options_volume_only": "unusual_options_volume_only",
    "missed_mover_hindsight": "top_mover_hindsight_benchmark",
}

UW_ARTIFACT_PATTERNS = {
    "stock_screener": ("stock-screener*", "_unzipped_mode_a/stock-screener*"),
    "hot_chains": ("hot-chains*", "_unzipped_mode_a/hot-chains*"),
    "chain_oi_changes": ("chain-oi-changes*", "_unzipped_mode_a/chain-oi-changes*"),
    "dp_eod_report": ("dp-eod-report*", "_unzipped_mode_a/dp-eod-report*"),
    "bot_eod_report": ("bot-eod-report*", "_unzipped_mode_a/bot-eod-report*"),
    "whale_institutional": ("*whale*", "*institutional*", "_trend_cache/trend-whale-symbols*"),
    "gex_gamma": ("*gex*", "*GEX*", "*gamma*", "*Gamma*"),
    "browser_text": ("browser_text/*.txt",),
    "trend_cache": ("_trend_cache/*",),
    "candidate_research": ("*candidate*.csv", "*watch*.csv", "*setup*.csv", "*recommendation*.csv"),
    "rejections": ("*reject*.csv", "*dropped*.csv", "*blocked*.csv"),
    "post_trade_trackers": ("*tracker*.csv", "*post-trade*.csv", "*post_trade*.csv"),
}


def _parse_horizons(text: str) -> List[int]:
    return engine._parse_horizons(text)


def _dated_dirs(root: Path, as_of: Optional[dt.date] = None) -> List[Tuple[dt.date, Path]]:
    rows: List[Tuple[dt.date, Path]] = []
    for path in Path(root).iterdir():
        if not path.is_dir() or not engine.DATE_RE.match(path.name):
            continue
        day = dt.date.fromisoformat(path.name)
        if as_of is None or day <= as_of:
            rows.append((day, path))
    return sorted(rows)


def _matches(day_dir: Path, patterns: Sequence[str]) -> List[str]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(path for path in day_dir.glob(pattern) if path.is_file())
    return sorted({str(path.resolve()) for path in paths})


def inventory_uw_data(root: Path, as_of: Optional[dt.date] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    root = Path(root).expanduser().resolve()
    try:
        usable_market_days = set(engine.available_market_days(root))
    except Exception:
        usable_market_days = set()

    rows: List[Dict[str, Any]] = []
    for day, day_dir in _dated_dirs(root, as_of):
        category_files = {name: _matches(day_dir, patterns) for name, patterns in UW_ARTIFACT_PATTERNS.items()}
        all_files = sorted({path for files in category_files.values() for path in files})
        row: Dict[str, Any] = {
            "date": day.isoformat(),
            "usable_market_day": day in usable_market_days,
            "file_count": sum(1 for path in day_dir.rglob("*") if path.is_file()),
            "recognized_file_count": len(all_files),
            "recognized_categories": ";".join(name for name, files in category_files.items() if files),
            "source_files_sample": ";".join(all_files[:60]),
        }
        for name, files in category_files.items():
            row[f"{name}_files"] = len(files)
        rows.append(row)

    df = pd.DataFrame(rows)
    category_day_counts = {
        name: int(df[f"{name}_files"].gt(0).sum()) if not df.empty and f"{name}_files" in df.columns else 0
        for name in UW_ARTIFACT_PATTERNS
    }
    category_file_counts = {
        name: int(df[f"{name}_files"].sum()) if not df.empty and f"{name}_files" in df.columns else 0
        for name in UW_ARTIFACT_PATTERNS
    }
    summary = {
        "root_dir": str(root),
        "inventory_as_of": as_of.isoformat() if as_of else "",
        "total_dated_folders": int(len(df)),
        "usable_market_days": int(df["usable_market_day"].sum()) if not df.empty else 0,
        "first_dated_folder": str(df["date"].min()) if not df.empty else "",
        "latest_dated_folder": str(df["date"].max()) if not df.empty else "",
        "latest_usable_market_day": max(usable_market_days).isoformat() if usable_market_days else "",
        "category_day_counts": category_day_counts,
        "category_file_counts": category_file_counts,
        "artifact_policy": "Local UW data is primary; browser_text/news captures are enrichment only.",
    }
    return df, summary


def source_files_for_date(root: Path, as_of: dt.date, limit: int = 250) -> List[str]:
    day_dir = Path(root).expanduser().resolve() / as_of.isoformat()
    if not day_dir.is_dir():
        return []
    return sorted(str(path.resolve()) for path in day_dir.rglob("*") if path.is_file())[:limit]


def readiness_verdict(proof: Dict[str, Any], counts: Dict[str, Any]) -> str:
    actionable = int(counts.get("actionable", 0) or 0)
    outcomes = int(counts.get("validation_outcomes", 0) or 0)
    proof_verdict = str(proof.get("verdict", "") or "").upper()
    v2_avg = engine._fnum(proof.get("v2_avg_net_r"))
    comparison = engine._fnum(proof.get("best_comparison_avg_net_r"))
    pf = engine._fnum(proof.get("v2_profit_factor"))
    if (
        proof_verdict == "PROVEN_FOR_ACTIONABLE"
        and actionable > 0
        and math.isfinite(v2_avg)
        and v2_avg > max(0.0, comparison if math.isfinite(comparison) else 0.0)
        and math.isfinite(pf)
        and pf >= engine.DEFAULT_MIN_VALIDATION_PROFIT_FACTOR
    ):
        return "PRODUCTION_READY"
    if outcomes > 0:
        return "USABLE_NEEDS_MORE_VALIDATION"
    return "NOT_YET_PROVEN"


def build_baseline_comparison(
    scorecard_csv: Path,
    *,
    out_dir: Path,
    primary_horizon: int,
) -> Tuple[Path, Path, Dict[str, Any]]:
    columns = [
        "baseline",
        "baseline_role",
        "tier",
        "horizon",
        "signal_count",
        "scored_count",
        "partial_count",
        "unscorable_count",
        "win_rate",
        "avg_net_r",
        "median_net_r",
        "profit_factor",
        "worst_losing_streak",
        "drawdown_proxy_r",
        "tradeable_with_real_quotes_pct",
        "avg_bid_ask_spread",
        "blocked_pct",
        "beats_best_comparison_at_primary_horizon",
    ]
    if scorecard_csv.exists():
        scorecard = pd.read_csv(scorecard_csv, low_memory=False)
    else:
        scorecard = pd.DataFrame()
    if scorecard.empty:
        comparison = pd.DataFrame(columns=columns)
    else:
        comparison = scorecard.copy()
        comparison["baseline_role"] = comparison["baseline"].map(REQUIRED_BASELINES).fillna("candidate_pipeline")
        comparison["_horizon"] = pd.to_numeric(comparison.get("horizon"), errors="coerce")
        primary = comparison[comparison["_horizon"].eq(int(primary_horizon))].copy()
        peers = primary[~primary["baseline"].astype(str).eq("trend_v2")]
        best_peer = pd.to_numeric(peers.get("avg_net_r", pd.Series(dtype=float)), errors="coerce").max()
        if not math.isfinite(engine._fnum(best_peer)):
            best_peer = math.nan
        comparison["beats_best_comparison_at_primary_horizon"] = False
        trend_mask = comparison["baseline"].astype(str).eq("trend_v2") & comparison["_horizon"].eq(int(primary_horizon))
        comparison.loc[trend_mask, "beats_best_comparison_at_primary_horizon"] = pd.to_numeric(
            comparison.loc[trend_mask, "avg_net_r"], errors="coerce"
        ).gt(best_peer if math.isfinite(engine._fnum(best_peer)) else -math.inf)
        comparison = comparison.drop(columns=["_horizon"])
        for column in columns:
            if column not in comparison.columns:
                comparison[column] = math.nan if column not in {"baseline", "baseline_role", "tier"} else ""
        comparison = comparison[columns]

    csv_path = out_dir / "baseline_comparison_report.csv"
    md_path = out_dir / "baseline_comparison_report.md"
    comparison.to_csv(csv_path, index=False)
    if comparison.empty:
        md = "# Baseline Comparison Report\n\n_No validation rows were available._\n"
    else:
        md = "# Baseline Comparison Report\n\n"
        md += comparison.sort_values(["horizon", "baseline", "tier"]).to_markdown(index=False)
        md += "\n"
    md_path.write_text(md, encoding="utf-8")

    summary = {
        "primary_horizon": int(primary_horizon),
        "required_baselines": REQUIRED_BASELINES,
        "rows": int(len(comparison)),
        "trend_v2_beats_best_comparison_at_primary_horizon": bool(
            comparison.get("beats_best_comparison_at_primary_horizon", pd.Series(dtype=bool)).fillna(False).any()
        ),
    }
    return csv_path, md_path, summary


def _copy_artifacts(engine_paths: Dict[str, Path], out_dir: Path) -> Dict[str, str]:
    copied: Dict[str, str] = {}
    for key, filename in CANONICAL_ARTIFACT_NAMES.items():
        src = Path(engine_paths[key])
        dst = out_dir / filename
        if src.exists():
            shutil.copy2(src, dst)
            copied[filename] = str(dst.resolve())
    return copied


def _write_manifest(
    *,
    root: Path,
    as_of: dt.date,
    out_dir: Path,
    args: argparse.Namespace,
    engine_result: Dict[str, Any],
    copied: Dict[str, str],
    inventory_summary: Dict[str, Any],
    baseline_summary: Dict[str, Any],
) -> Path:
    engine_metadata_path = out_dir / CANONICAL_ARTIFACT_NAMES["metadata_json"]
    engine_metadata: Dict[str, Any] = {}
    if engine_metadata_path.exists():
        try:
            engine_metadata = json.loads(engine_metadata_path.read_text(encoding="utf-8"))
        except Exception:
            engine_metadata = {}

    counts = dict(engine_metadata.get("counts", {}))
    proof = dict(engine_result.get("proof", engine_metadata.get("proof", {})))
    verdict = readiness_verdict(proof, counts)
    command = (
        f"python3 -m uwos.options_trend_pipeline_v2 --base-dir {root} --as-of {as_of.isoformat()} "
        f"--out-dir {out_dir}"
    )
    metadata = {
        "command": command,
        "latest_date_command": f"python3 -m uwos.options_trend_pipeline_v2 --base-dir {root}",
        "historical_date_command": (
            f"python3 -m uwos.options_trend_pipeline_v2 --base-dir {root} --as-of YYYY-MM-DD "
            f"--out-dir {root / 'out' / 'options_trend_pipeline_v2' / 'YYYY-MM-DD'}"
        ),
        "root_dir": str(root),
        "out_dir": str(out_dir),
        "as_of": as_of.isoformat(),
        "lookback": int(args.lookback),
        "validate_days": int(args.validate_days),
        "horizons": _parse_horizons(args.horizons),
        "pipeline_module": "uwos.options_trend_pipeline_v2",
        "engine_module": "uwos.trend_analysis_v2",
        "readiness_verdict": verdict,
        "proof": proof,
        "counts": counts,
        "data_inventory": inventory_summary,
        "baseline_comparison": baseline_summary,
        "deterministic_seeds": {
            "random_same_date_liquidity": "sha256(signal_date|ticker) first 16 hex digits",
        },
        "historical_leakage_controls": {
            "signal_generation": "Only dated local UW files on or before the signal date are read.",
            "validation_cutoff": "Validation outcomes with exit dates after the report as-of date are UNSCORABLE.",
            "current_social_news_on_old_signals": "Not used; local browser_text is report enrichment only.",
            "uses_current_schwab_for_historical_validation": False,
        },
        "quote_outcome_integrity": {
            "SCORED": "Exit used later local option quotes or expiry intrinsic value.",
            "PARTIAL": "Exit used stock/intrinsic proxy before expiry because later option quotes were incomplete.",
            "UNSCORABLE": "Missing entry/exit quote, underlying, or validation cutoff prevented scoring.",
        },
        "existing_pipeline_assessment": {
            "reused_reliable_pieces": [
                "local UW dated-folder discovery and CSV/zip readers from swing_trend_pipeline",
                "hot-chain OCC quote parsing and vertical-spread construction",
                "local bot-EOD whale/institutional premium extraction",
                "legacy trend_analysis outputs as comparison baseline only",
                "local browser_text captures as untrusted enrichment only",
            ],
            "replaced_or_gated_pieces": [
                "legacy trend recommendations cannot approve trades; they are scored as old-pipeline baseline",
                "historical validation cannot consume current Schwab chains or current web/social sentiment",
                "incomplete outcomes are labeled PARTIAL or UNSCORABLE and are not counted as wins",
            ],
        },
        "source_files_latest_date": source_files_for_date(root, as_of),
        "artifacts": copied,
        "engine_metadata": engine_metadata,
    }
    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(engine._json_safe(metadata), indent=2, sort_keys=True), encoding="utf-8")
    return metadata_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production daily UW options-trade-idea pipeline with validation, baselines, and missed-mover audit."
    )
    parser.add_argument("--base-dir", default=str(DEFAULT_ROOT), help="Root tradedesk directory containing dated UW folders.")
    parser.add_argument("--as-of", default="", help="YYYY-MM-DD. Default: latest usable local UW date.")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: out/options_trend_pipeline_v2/YYYY-MM-DD.")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--validate-days", type=int, default=DEFAULT_VALIDATE_DAYS)
    parser.add_argument("--horizons", default=",".join(str(h) for h in DEFAULT_HORIZONS))
    parser.add_argument("--max-daily-rows", type=int, default=engine.DEFAULT_MAX_DAILY_ROWS)
    parser.add_argument("--min-validation-samples", type=int, default=engine.DEFAULT_MIN_VALIDATION_SAMPLES)
    parser.add_argument("--whale-lookback-days", type=int, default=1)
    parser.add_argument("--no-whales", action="store_true")
    return parser.parse_args(list(argv) if argv is not None else None)


def run(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    root = Path(args.base_dir).expanduser().resolve()
    as_of = engine.resolve_as_of(root, args.as_of or None)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else root / "out" / "options_trend_pipeline_v2" / as_of.isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)

    engine_args = [
        as_of.isoformat(),
        "--root-dir",
        str(root),
        "--out-dir",
        str(out_dir),
        "--lookback",
        str(int(args.lookback)),
        "--validate-days",
        str(int(args.validate_days)),
        "--horizons",
        str(args.horizons),
        "--max-daily-rows",
        str(int(args.max_daily_rows)),
        "--min-validation-samples",
        str(int(args.min_validation_samples)),
        "--whale-lookback-days",
        str(int(args.whale_lookback_days)),
    ]
    if args.no_whales:
        engine_args.append("--no-whales")

    engine_result = engine.run(engine_args)
    engine_paths = {key: Path(value) for key, value in engine_result["paths"].items()}
    copied = _copy_artifacts(engine_paths, out_dir)

    inventory_df, inventory_summary = inventory_uw_data(root, as_of)
    inventory_csv = out_dir / "data_inventory.csv"
    inventory_json = out_dir / "data_inventory_summary.json"
    inventory_df.to_csv(inventory_csv, index=False)
    inventory_json.write_text(json.dumps(engine._json_safe(inventory_summary), indent=2, sort_keys=True), encoding="utf-8")
    copied["data_inventory.csv"] = str(inventory_csv.resolve())
    copied["data_inventory_summary.json"] = str(inventory_json.resolve())

    primary_horizon = 5 if 5 in _parse_horizons(args.horizons) else _parse_horizons(args.horizons)[0]
    baseline_csv, baseline_md, baseline_summary = build_baseline_comparison(
        out_dir / "validation_scorecard.csv",
        out_dir=out_dir,
        primary_horizon=primary_horizon,
    )
    copied["baseline_comparison_report.csv"] = str(baseline_csv.resolve())
    copied["baseline_comparison_report.md"] = str(baseline_md.resolve())

    metadata_path = _write_manifest(
        root=root,
        as_of=as_of,
        out_dir=out_dir,
        args=args,
        engine_result=engine_result,
        copied=copied,
        inventory_summary=inventory_summary,
        baseline_summary=baseline_summary,
    )
    copied["metadata.json"] = str(metadata_path.resolve())

    print(f"Wrote canonical output dir: {out_dir}")
    print(f"Wrote: {out_dir / 'daily_report.md'}")
    print(f"Wrote: {metadata_path}")
    return {
        "as_of": as_of,
        "out_dir": out_dir,
        "paths": copied,
        "metadata_path": metadata_path,
        "engine_result": engine_result,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
