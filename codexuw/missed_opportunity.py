from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import pandas as pd


MISS_COLUMNS = [
    "asof",
    "ticker",
    "lane",
    "status",
    "strategy",
    "realized_pnl",
    "mfe",
    "thesis_worked",
    "classification",
    "reason",
]


def classify_missed_opportunity(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    status = str(row.get("status") or row.get("Status") or "")
    lane = str(row.get("lane") or row.get("Lane") or "")
    reason = str(row.get("reason_for_win_loss") or row.get("outcome_note") or row.get("monitor_trigger") or "")
    pnl = pd.to_numeric(pd.Series([row.get("realized_pnl")]), errors="coerce").iloc[0]
    mfe = pd.to_numeric(pd.Series([row.get("mfe")]), errors="coerce").iloc[0]
    thesis_worked = str(row.get("thesis_worked") or "").strip().lower() in {"true", "yes", "1", "worked"}
    worked = bool((pd.notna(pnl) and pnl > 0) or (pd.notna(mfe) and mfe > 0) or thesis_worked)
    if not worked:
        return "correct risk avoidance", "no later positive outcome recorded"
    lower_reason = reason.lower()
    if "data" in lower_reason or "news" in lower_reason or "unconfirmed" in lower_reason:
        return "bad data/news gap", reason or "later outcome worked after data/news blocker"
    if "wheel" in lane.lower() or "debit" in lane.lower() or "index" in lane.lower():
        return "missing strategy coverage", reason or "later outcome worked in non-Execute lane"
    if "research" in lane.lower() or "avoid" in status.lower() or "research" in status.lower():
        return "over-filtering", reason or "later positive outcome was not promoted"
    return "correct risk avoidance", reason or "later outcome worked but was not a rejected opportunity"


def build_missed_opportunity_audit(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=MISS_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in ledger.iterrows():
        status = str(row.get("status") or "")
        if "Execute" in status:
            continue
        classification, reason = classify_missed_opportunity(row)
        if classification == "correct risk avoidance" and reason == "no later positive outcome recorded":
            continue
        rows.append(
            {
                "asof": row.get("asof") or row.get("report_date"),
                "ticker": row.get("ticker"),
                "lane": row.get("lane"),
                "status": row.get("status"),
                "strategy": row.get("strategy"),
                "realized_pnl": row.get("realized_pnl"),
                "mfe": row.get("mfe"),
                "thesis_worked": row.get("thesis_worked"),
                "classification": classification,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=MISS_COLUMNS)


def write_missed_opportunity_audit(out_dir: Path, asof: dt.date, ledger: pd.DataFrame) -> tuple[Path, Path, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = build_missed_opportunity_audit(ledger)
    csv_path = out_dir / f"codexdaily_v3_missed_opportunity_audit_{asof}.csv"
    audit.to_csv(csv_path, index=False)
    summary = {
        "status": "ok",
        "rows": int(len(audit)),
        "classifications": audit["classification"].value_counts().to_dict() if not audit.empty else {},
        "csv": str(csv_path),
    }
    json_path = out_dir / f"codexdaily_v3_missed_opportunity_audit_{asof}.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path, summary
