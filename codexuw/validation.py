from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .data import find_export, infer_asof_date
from .opportunity import PIPELINE_NAME_V3, PIPELINE_VERSION_V3


Runner = Callable[[Path, Path], dict[str, Any]]


def _source_complete(folder: Path) -> bool:
    try:
        find_export(folder, "stock-screener-")
        find_export(folder, "hot-chains-")
        find_export(folder, "bot-eod-report-")
        return True
    except FileNotFoundError:
        return False


def select_systematic_date_folders(root: Path, *, as_of: dt.date | None = None, latest_n: int = 5) -> list[Path]:
    dated: dict[dt.date, Path] = {}
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            day = infer_asof_date(child)
        except ValueError:
            continue
        if as_of is not None and day > as_of:
            continue
        if not _source_complete(child):
            continue
        existing = dated.get(day)
        if existing is None or child.name == str(day):
            dated[day] = child
    return [path for _, path in sorted(dated.items(), key=lambda item: item[0], reverse=True)[:latest_n]]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _v2_manifest(root: Path, day: dt.date) -> dict[str, Any]:
    return _read_json(root / "out" / f"codexdaily_v2_{day}" / f"codexuw_manifest_{day}.json")


def compare_v3_vs_v2(v3_manifest: dict[str, Any], v2_manifest: dict[str, Any]) -> dict[str, Any]:
    v3_counts = v3_manifest.get("opportunity_counts") or {}
    v2_funnel = v2_manifest.get("funnel") or {}
    liquidity = v3_manifest.get("liquidity_shift") or {}
    return {
        "v2_execute_count": int(v2_manifest.get("execute_rows", v2_funnel.get("final_trade_rows", 0)) or 0),
        "v3_execute_count": int(v3_counts.get("execute", v3_manifest.get("execute_rows", 0)) or 0),
        "v3_scout_count": int(v3_counts.get("scout", v3_manifest.get("scout_rows", 0)) or 0),
        "v3_lane_coverage": ",".join(sorted((v3_manifest.get("lane_coverage") or {}).keys())),
        "v2_live_data_failures": len(((v2_manifest.get("run_provenance") or {}).get("schwab_snapshot") or {}).get("errors") or {}),
        "v3_live_data_failures": len(((v3_manifest.get("reproducibility") or {}).get("schwab_snapshot") or {}).get("errors") or {}),
        "v3_overlay_behavior": "available via codexuw.daily_v3 overlay",
        "v3_ledger_coverage": v3_manifest.get("recommendation_ledger", ""),
        "v3_no_trade_audit_classification": (v3_manifest.get("no_trade_audit") or {}).get("classification", ""),
        "v3_liquidity_shift_signals": int(liquidity.get("flow_velocity_signals", 0) or 0),
        "v3_flow_velocity_candidates": int(liquidity.get("flow_velocity_rows", 0) or 0),
        "v3_zero_dte_index_signals": int(liquidity.get("zero_dte_index_signal_count", 0) or 0),
    }


def _top_rejection_reasons(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        df = pd.read_csv(path)
    except Exception:
        return ""
    if df.empty:
        return ""
    reason_col = "reason" if "reason" in df.columns else "decision_reason" if "decision_reason" in df.columns else ""
    if not reason_col:
        return ""
    count_col = "count" if "count" in df.columns else ""
    rows = []
    for _, row in df.head(5).iterrows():
        reason = str(row.get(reason_col) or "").strip()
        if not reason:
            continue
        if count_col:
            rows.append(f"{reason}({row.get(count_col)})")
        else:
            rows.append(reason)
    return "; ".join(rows)


def run_validation_harness(
    *,
    root: Path,
    out_dir: Path,
    asof: dt.date,
    latest_n: int,
    runner: Runner | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = select_systematic_date_folders(root, as_of=asof, latest_n=latest_n)
    rows: list[dict[str, Any]] = []
    run_results: list[dict[str, Any]] = []
    for folder in selected:
        day = infer_asof_date(folder)
        run_out = out_dir / f"codexdaily_v3_{day}" if runner is not None else root / "out" / f"codexdaily_v3_{day}"
        if runner is not None:
            result = runner(folder, run_out)
            run_results.append(result)
            v3_manifest = result
        else:
            v3_manifest = _read_json(run_out / f"codexdaily_v3_manifest_{day}.json")
        v2_manifest = _v2_manifest(root, day)
        compare = compare_v3_vs_v2(v3_manifest, v2_manifest)
        compare["v3_top_rejected_candidates"] = _top_rejection_reasons(run_out / f"codexdaily_v3_rejections_{day}.csv")
        compare["v2_top_rejected_candidates"] = _top_rejection_reasons(
            root / "out" / f"codexdaily_v2_{day}" / f"codexuw_rejections_{day}.csv"
        )
        rows.append(
            {
                "date": str(day),
                "folder": str(folder),
                "v3_out_dir": str(run_out),
                **compare,
            }
        )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / f"codexdaily_v3_validation_summary_{asof}.csv"
    summary.to_csv(summary_path, index=False)
    manifest = {
        "pipeline_name": PIPELINE_NAME_V3,
        "pipeline_version": PIPELINE_VERSION_V3,
        "run_mode": "validation",
        "asof": str(asof),
        "latest_n": latest_n,
        "selection_method": "latest source-complete dated folders at or before asof",
        "selected_dates": [str(infer_asof_date(path)) for path in selected],
        "summary_csv": str(summary_path),
        "run_result_count": len(run_results),
        "comparisons": rows,
    }
    manifest_path = out_dir / f"codexdaily_v3_validation_manifest_{asof}.json"
    report_path = out_dir / f"codexdaily_v3_validation_report_{asof}.md"
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        f"# {PIPELINE_NAME_V3} Validation Harness - {asof}",
        "",
        "## First Screen",
        "",
        "| Item | Value |",
        "|:--|:--|",
        f"| Pipeline | {PIPELINE_NAME_V3} |",
        "| Run mode | Validation harness |",
        f"| Date selection | latest {latest_n} source-complete folders at or before {asof} |",
        f"| Selected dates | {', '.join(manifest['selected_dates'])} |",
        "",
        "## V3 vs V2",
        "",
        summary.to_markdown(index=False) if not summary.empty else "_No source-complete dated folders selected._",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return manifest
