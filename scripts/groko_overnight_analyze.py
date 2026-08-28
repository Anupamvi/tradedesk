#!/usr/bin/env python3
"""Summarize a Groko replay pin and a live run. Does not import Codex P&L as proof."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def pf_metrics(df: pd.DataFrame, pnl_col: str = "pnl_1x") -> dict:
    if df.empty or pnl_col not in df.columns:
        return {"n": 0, "wins": 0, "win_rate": None, "pf": None, "pnl": 0.0}
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
    wins = int((pnl > 0).sum())
    gp = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
    gl = float(-pnl[pnl < 0].sum()) if not pnl.empty else 0.0
    pf = gp / gl if gl > 0 else (math.inf if gp > 0 else None)
    return {
        "n": int(len(pnl)),
        "wins": wins,
        "win_rate": round(wins / len(pnl), 4) if len(pnl) else None,
        "pf": None if pf is None else (round(pf, 3) if math.isfinite(pf) else "inf"),
        "pnl": round(float(pnl.sum()), 2),
    }


def summarize_replay(replay_dir: Path, split_day: str) -> dict:
    detail = load_frame(replay_dir / "groko_replay_detail.csv")
    manifest_path = replay_dir / "groko_replay_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    pin = {}
    for pin_path in (
        Path("/Users/anuppamvi/uw_root/tradedesk/knowledge/groko_replay_pin.json"),
        Path("/Users/anuppamvi/tradedesk/knowledge/groko_replay_pin.json"),
    ):
        if pin_path.exists():
            pin = json.loads(pin_path.read_text())
            pin["_path"] = str(pin_path)
            break

    out: dict = {
        "replay_dir": str(replay_dir),
        "manifest_producer": manifest.get("producer"),
        "manifest_schema": manifest.get("schema_version"),
        "pipeline_version": manifest.get("pipeline_version"),
        "history_days": manifest.get("days"),
        "pin": {
            k: pin.get(k)
            for k in ("producer", "schema_version", "split_day", "replay_detail_path", "_path")
        },
        "row_count": int(len(detail)),
        "borrowed_codex_pnl": False,
    }
    if detail.empty:
        out["error"] = "missing groko replay detail"
        return out

    split = pd.Timestamp(split_day)
    evaluated = detail.copy()
    if "exact_evaluated" in evaluated.columns:
        evaluated = evaluated[as_bool(evaluated["exact_evaluated"])]
    if "pnl_1x" in evaluated.columns:
        evaluated = evaluated[pd.to_numeric(evaluated["pnl_1x"], errors="coerce").notna()]

    if "entry_type" in evaluated.columns:
        credit = evaluated[evaluated["entry_type"].astype(str).str.upper().eq("CREDIT")]
        debit = evaluated[evaluated["entry_type"].astype(str).str.upper().eq("DEBIT")]
    else:
        credit = evaluated
        debit = evaluated.iloc[0:0]

    if "next_session_reprice_approved" in evaluated.columns:
        selected = evaluated[as_bool(evaluated["next_session_reprice_approved"])]
    elif "decision_pass" in evaluated.columns:
        selected = evaluated[as_bool(evaluated["decision_pass"])]
    else:
        selected = evaluated.iloc[0:0]

    selected_asof = pd.to_datetime(selected.get("asof"), errors="coerce")
    out["evaluated"] = pf_metrics(evaluated)
    out["credit_evaluated"] = pf_metrics(credit)
    out["debit_evaluated"] = pf_metrics(debit)
    out["reprice_approved"] = pf_metrics(selected)
    out["reprice_approved_train"] = pf_metrics(selected[selected_asof.lt(split)])
    out["reprice_approved_test"] = pf_metrics(selected[selected_asof.ge(split)])
    if "selector_policy_status" in evaluated.columns:
        sel = evaluated[evaluated["selector_policy_status"].astype(str).str.upper().eq("PASS")]
        sel_asof = pd.to_datetime(sel.get("asof"), errors="coerce")
        out["selector_pass"] = pf_metrics(sel)
        out["selector_pass_train"] = pf_metrics(sel[sel_asof.lt(split)])
        out["selector_pass_test"] = pf_metrics(sel[sel_asof.ge(split)])
        if "strategy_route" in sel.columns:
            out["selector_pass_by_route"] = {
                str(route): pf_metrics(group)
                for route, group in sel.groupby(sel["strategy_route"].astype(str))
            }
    if "dte" in selected.columns:
        dte = pd.to_numeric(selected["dte"], errors="coerce")
        out["reprice_approved_dte_28_45"] = pf_metrics(selected[dte.between(28, 45)])
        out["reprice_approved_dte_11_27"] = pf_metrics(selected[dte.between(11, 27)])
    return out


def summarize_live(run_dir: Path) -> dict:
    if not run_dir.exists():
        return {"error": f"missing live dir {run_dir}"}
    manifests = sorted(run_dir.glob("groko_manifest_*.json"))
    reports = sorted(run_dir.glob("groko_report_*.md"))
    tickets = load_frame(run_dir / "green_trade_tickets.csv")
    if tickets.empty:
        tickets = load_frame(run_dir / "trade_tickets.csv")
    board = load_frame(run_dir / "decision_board.csv")
    manifest = json.loads(manifests[-1].read_text()) if manifests else {}
    green = tickets
    if not tickets.empty and "ready_to_enter" in tickets.columns:
        green = tickets[as_bool(tickets["ready_to_enter"])]
    return {
        "run_dir": str(run_dir),
        "pipeline_name": manifest.get("pipeline_name"),
        "pipeline_version": manifest.get("pipeline_version"),
        "manifest": str(manifests[-1]) if manifests else "",
        "report": str(reports[-1]) if reports else "",
        "ticket_rows": int(len(tickets)),
        "green_ready_rows": int(len(green)),
        "decision_board_rows": int(len(board)),
        "codex_artifact_names_in_run": sorted(p.name for p in run_dir.glob("codexuw_*")),
        "groko_artifact_names": sorted(p.name for p in run_dir.glob("groko_*")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-dir", required=True, type=Path)
    parser.add_argument("--live-dir", required=True, type=Path)
    parser.add_argument("--split-day", default="2026-05-01")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    payload = {
        "replay": summarize_replay(args.replay_dir, args.split_day),
        "live": summarize_live(args.live_dir),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
