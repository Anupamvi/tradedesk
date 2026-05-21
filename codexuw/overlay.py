from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .data import dte_from_expiry, read_csv_export, safe_float
from .engine import apply_oi_carryover, assign_trade_statuses, build_entry_watchlist
from .occ import parse_occ_symbol
from .opportunity import PIPELINE_NAME_V3, PIPELINE_VERSION_V3, build_opportunity_board, write_recommendation_ledger


def infer_date_from_name(path: Path) -> dt.date | None:
    match = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), "%Y-%m-%d").date()


def load_chain_oi_file(path: Path, *, asof: dt.date) -> pd.DataFrame:
    df = read_csv_export(path)
    parsed = df["option_symbol"].map(parse_occ_symbol)
    df["ticker"] = parsed.map(lambda x: x.root if x else df.get("underlying_symbol", ""))
    df["expiry_dt"] = parsed.map(lambda x: x.expiry if x else pd.NaT)
    df["right"] = parsed.map(lambda x: x.right if x else "")
    df["strike"] = parsed.map(lambda x: x.strike if x else math.nan)
    df["dte"] = df["expiry_dt"].map(lambda x: dte_from_expiry(x, asof))
    for col in [
        "oi_diff_plain",
        "oi_change",
        "curr_oi",
        "last_oi",
        "volume",
        "prev_total_premium",
        "prev_bid_volume",
        "prev_ask_volume",
        "prev_multi_leg_volume",
        "prev_stock_multi_leg_volume",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.attrs["source_path"] = str(path)
    return df


def _candidate_key(row: pd.Series) -> str:
    return "|".join(
        [
            str(row.get("ticker") or ""),
            str(row.get("strategy") or row.get("direction") or ""),
            str(row.get("expiry") or ""),
            str(row.get("short_leg") or row.get("sell_leg") or ""),
            str(row.get("long_leg") or row.get("buy_leg") or ""),
        ]
    )


def _recommendation_rank(status: object) -> int:
    text = str(status or "")
    if text == "Execute":
        return 4
    if text == "Watch":
        return 3
    if text == "Research":
        return 2
    if text == "Avoid":
        return 1
    return 0


def compare_overlay_changes(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    before = before.copy()
    after = after.copy()
    before["_key"] = before.apply(_candidate_key, axis=1)
    after["_key"] = after.apply(_candidate_key, axis=1)
    bmap = before.set_index("_key", drop=False)
    amap = after.set_index("_key", drop=False)
    keys = sorted(set(bmap.index) | set(amap.index))
    rows: list[dict[str, Any]] = []
    for key in keys:
        if key in bmap.index:
            old_lookup = bmap.loc[key]
            old = old_lookup.iloc[0] if isinstance(old_lookup, pd.DataFrame) else old_lookup
        else:
            old = pd.Series(dtype=object)
        if key in amap.index:
            new_lookup = amap.loc[key]
            new = new_lookup.iloc[0] if isinstance(new_lookup, pd.DataFrame) else new_lookup
        else:
            new = pd.Series(dtype=object)
        old_status = str(old.get("trade_status") or "") if not old.empty else ""
        new_status = str(new.get("trade_status") or "") if not new.empty else ""
        old_oi = str(old.get("oi_carryover_status") or "") if not old.empty else ""
        new_oi = str(new.get("oi_carryover_status") or "") if not new.empty else ""
        old_reason = str(old.get("trade_status_reason") or old.get("primary_blocker") or "") if not old.empty else ""
        new_reason = str(new.get("trade_status_reason") or new.get("primary_blocker") or "") if not new.empty else ""
        old_mid = safe_float(old.get("mid_credit"), safe_float(old.get("mid_debit"))) if not old.empty else math.nan
        new_mid = safe_float(new.get("mid_credit"), safe_float(new.get("mid_debit"))) if not new.empty else math.nan
        live_changed = (
            "not refreshed"
            if (math.isfinite(old_mid) and math.isfinite(new_mid) and abs(old_mid - new_mid) < 0.005)
            else "changed" if math.isfinite(old_mid) and math.isfinite(new_mid) else "not available"
        )
        if old_status == new_status and old_oi == new_oi and live_changed == "not refreshed":
            continue
        rank_delta = _recommendation_rank(new_status) - _recommendation_rank(old_status)
        if rank_delta > 0:
            rec_change = "upgraded"
        elif rank_delta < 0:
            rec_change = "downgraded"
        elif old_status != new_status:
            rec_change = "changed"
        else:
            rec_change = "unchanged_status"
        rows.append(
            {
                "candidate_key": key,
                "ticker": str((new if not new.empty else old).get("ticker") or ""),
                "trade": str((new if not new.empty else old).get("strategy") or (new if not new.empty else old).get("direction") or ""),
                "previous_status": old_status or "new",
                "new_status": new_status or "removed",
                "previous_oi_support": old_oi,
                "new_oi_support_or_conflict": new_oi,
                "recommendation_change": rec_change,
                "removed_trade": bool(new.empty),
                "changed_live_pricing": live_changed,
                "exact_reason": new_reason or old_reason or "OI overlay changed candidate context",
            }
        )
    return pd.DataFrame(rows)


def _read_prior_scored(prior_out_dir: Path, asof: dt.date) -> pd.DataFrame:
    candidates = [
        prior_out_dir / f"codexdaily_v3_scored_{asof}.csv",
        prior_out_dir / f"codexuw_scored_{asof}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"No prior scored CSV found in {prior_out_dir} for {asof}")


def run_overlay(
    *,
    prior_out_dir: Path,
    overlay_file: Path,
    out_dir: Path,
    asof: dt.date,
    overlay_date: dt.date | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_date = overlay_date or infer_date_from_name(overlay_file) or asof
    before = _read_prior_scored(prior_out_dir, asof)
    chain_oi = load_chain_oi_file(overlay_file, asof=asof)
    after = apply_oi_carryover(before, chain_oi)
    after = assign_trade_statuses(after, index_income_mode="primary")
    watch = build_entry_watchlist(after)
    final = after[after["trade_status"].astype(str).eq("Execute")].copy() if "trade_status" in after.columns else after.iloc[0:0].copy()
    board = build_opportunity_board(scored=after, final=final, watchlist=watch, portfolio=None)
    changes = compare_overlay_changes(before, after)

    after.to_csv(out_dir / f"codexdaily_v3_overlay_scored_{asof}_{overlay_date}.csv", index=False)
    watch.to_csv(out_dir / f"codexdaily_v3_overlay_watchlist_{asof}_{overlay_date}.csv", index=False)
    board.to_csv(out_dir / f"codexdaily_v3_overlay_opportunity_board_{asof}_{overlay_date}.csv", index=False)
    changes.to_csv(out_dir / f"codexdaily_v3_overlay_changes_{asof}_{overlay_date}.csv", index=False)
    ledger_path, global_ledger_path = write_recommendation_ledger(out_dir, asof, board)

    manifest = {
        "pipeline_name": PIPELINE_NAME_V3,
        "pipeline_version": PIPELINE_VERSION_V3,
        "run_mode": "overlay",
        "asof": str(asof),
        "overlay_date": str(overlay_date),
        "prior_out_dir": str(prior_out_dir),
        "overlay_file": str(overlay_file),
        "changed_candidate_rows": int(len(changes)),
        "execute_rows": int(board["Status"].astype(str).str.contains("Execute", regex=False).sum()) if not board.empty else 0,
        "scout_rows": int(board["Status"].astype(str).str.contains("Scout", regex=False).sum()) if not board.empty else 0,
        "recommendation_ledger": str(ledger_path),
        "global_recommendation_ledger": str(global_ledger_path),
    }
    manifest_path = out_dir / f"codexdaily_v3_overlay_manifest_{asof}_{overlay_date}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    report_path = out_dir / f"codexdaily_v3_overlay_report_{asof}_{overlay_date}.md"
    lines = [
        f"# {PIPELINE_NAME_V3} Overlay Report - {asof} with {overlay_date}",
        "",
        "## First Screen",
        "",
        "| Item | Value |",
        "|:--|:--|",
        f"| Pipeline | {PIPELINE_NAME_V3} |",
        "| Run mode | Overlay |",
        f"| Prior run | {prior_out_dir} |",
        f"| Overlay input | {overlay_file} |",
        f"| Changed candidates | {len(changes)} |",
        "",
        "## Opportunity Board",
        "",
        board[[c for c in board.columns if c in [
            "Lane",
            "Status",
            "Ticker",
            "Trade",
            "Expiry",
            "Entry limit",
            "Live mid/natural",
            "Required confirmation",
            "Why Execute, Scout, Research, or Avoid",
        ]]].to_markdown(index=False) if not board.empty else "_No candidates._",
        "",
        "## What Changed",
        "",
        changes.to_markdown(index=False) if not changes.empty else "_No recommendation status or OI-support changes._",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["report_path"] = str(report_path)
    return manifest
