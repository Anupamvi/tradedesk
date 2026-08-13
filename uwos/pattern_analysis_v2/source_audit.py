"""Per-date audit of the five canonical Pattern Analysis V2 UW inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from uwos.options_pattern_pipeline_v1 import core as engine

PRIMARY_SOURCES = (
    ("stock_screener", "stock-screener"),
    ("hot_chains", "hot-chains"),
    ("chain_oi", "chain-oi-changes"),
    ("dp_eod", "dp-eod-report"),
    ("bot_eod", "bot-eod-report"),
)
CORE_SIGNAL_KEYS = {"stock_screener", "hot_chains", "chain_oi"}


def build_primary_source_audit(base_dir: Path, out_dir: Path, as_of: str) -> Dict[str, str]:
    rows: List[Dict[str, Any]] = []
    for signal_date in engine.list_date_dirs(base_dir):
        if signal_date > as_of:
            continue
        sources = engine.sources_for_date(base_dir / signal_date, signal_date)
        row: Dict[str, Any] = {"date": signal_date}
        missing = []
        missing_core = []
        total_bytes = 0
        for key, label in PRIMARY_SOURCES:
            refs = list(sources.get(key) or [])
            row[f"{key}_present"] = bool(refs)
            row[f"{key}_archive_count"] = len(refs)
            source_bytes = sum(ref.path.stat().st_size for ref in refs if ref.path.exists())
            row[f"{key}_compressed_bytes"] = source_bytes
            total_bytes += source_bytes
            if not refs:
                missing.append(label)
                if key in CORE_SIGNAL_KEYS:
                    missing_core.append(label)
        row["all_five_present"] = not missing
        row["core_signal_eligible"] = not missing_core
        row["included_by_v2"] = not missing_core
        row["bot_flow_family_eligible"] = bool(sources.get("bot_eod"))
        row["dark_pool_family_eligible"] = bool(sources.get("dp_eod"))
        row["missing_sources"] = ";".join(missing)
        row["missing_core_sources"] = ";".join(missing_core)
        row["compressed_bytes_all_sources"] = total_bytes
        rows.append(row)

    audit_path = out_dir / "primary_source_coverage.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(audit_path, index=False)
    complete = frame[frame["all_five_present"].astype(bool)] if not frame.empty else frame
    core_eligible = frame[frame["core_signal_eligible"].astype(bool)] if not frame.empty else frame
    summary = {
        "as_of": as_of,
        "dated_folders_audited": len(frame),
        "core_signal_dates": len(core_eligible),
        "all_five_source_dates": len(complete),
        "excluded_missing_core_dates": len(frame) - len(core_eligible),
        "core_dates_missing_optional_source": int(
            (core_eligible["all_five_present"].astype(bool) == False).sum()
        ) if len(core_eligible) else 0,
        "first_core_signal_date": str(core_eligible["date"].min()) if len(core_eligible) else None,
        "last_core_signal_date": str(core_eligible["date"].max()) if len(core_eligible) else None,
        "first_all_five_date": str(complete["date"].min()) if len(complete) else None,
        "last_all_five_date": str(complete["date"].max()) if len(complete) else None,
        "compressed_bytes_all_five_dates": int(
            complete["compressed_bytes_all_sources"].sum()
        ) if len(complete) else 0,
        "required_sources": [label for _, label in PRIMARY_SOURCES],
        "core_required_sources": [
            label for key, label in PRIMARY_SOURCES if key in CORE_SIGNAL_KEYS
        ],
        "optional_family_sources": ["dp-eod-report", "bot-eod-report"],
    }
    summary_path = out_dir / "primary_source_coverage_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return {
        "primary_source_coverage": str(audit_path),
        "primary_source_coverage_summary": str(summary_path),
    }
