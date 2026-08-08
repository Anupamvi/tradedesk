"""Consolidate Pattern Analysis V2 research into one honest ranking registry.

Ranking is descriptive, not a confidence score. A high rank means the observed
out-of-sample economics are better than the rows below it. Deployment still
requires every explicit gate: positive train/test economics, every fold
profitable, a date/sector/count-matched null, day-clustered PF p05 >= 1.2, and
executable option replay after costs.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

REGISTRY_FIELDS = [
    "rank",
    "trade_pattern_rank",
    "research_rank",
    "rank_scope",
    "pattern_id",
    "pattern_name",
    "pattern_class",
    "pattern_scope",
    "source_inputs",
    "direction",
    "strategy",
    "sample_train",
    "sample_test",
    "train_profit_factor",
    "test_profit_factor",
    "test_average_r",
    "test_win_rate",
    "positive_folds",
    "fold_count",
    "matched_null_profit_factor",
    "matched_null_p_value",
    "matched_null_coverage",
    "clustered_pf_p05",
    "option_replay",
    "all_five_sources",
    "deployment_ready",
    "status",
    "ranking_basis",
    "deployment_gate_failures",
    "rejection_reason",
    "evidence_artifact",
]


def _num(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value).strip()


def _pf(values: Iterable[float]) -> Optional[float]:
    array = np.asarray(list(values), dtype=float)
    gains = array[array > 0].sum()
    losses = -array[array < 0].sum()
    if losses > 0:
        return float(gains / losses)
    return 999.0 if gains > 0 else None


def _frame(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _base_row(**values: Any) -> Dict[str, Any]:
    row = {field: "" for field in REGISTRY_FIELDS}
    row.update(
        {
            "deployment_ready": False,
            "option_replay": "no",
            "all_five_sources": "unknown",
            "ranking_basis": "observed OOS economics; not a per-trade probability",
        }
    )
    row.update(values)
    return row


def _engine_rows(out_dir: Path) -> List[Dict[str, Any]]:
    discovered = _frame(out_dir / "discovered_pattern_families.csv")
    rows: List[Dict[str, Any]] = []
    for item in discovered.to_dict("records"):
        family_id = str(item.get("pattern_family") or "UNKNOWN")
        family_parts = family_id.split("__")
        strategy = family_parts[2] if len(family_parts) >= 3 else ""
        pattern_scope = _text(item.get("pattern_scope")) or "FAMILY"
        scored = int(_num(item.get("validation_scored_count")) or 0)
        avg_r = _num(item.get("validation_average_net_r"))
        profit_factor = _num(item.get("validation_profit_factor"))
        positive_folds = int(_num(item.get("positive_validation_splits")) or 0)
        fold_count = int(_num(item.get("validation_split_count")) or 0)
        clustered_p05 = _num(item.get("validation_day_clustered_profit_factor_p05"))
        matched_null_p = _num(item.get("matched_null_p_value"))
        matched_null_coverage = _num(item.get("matched_null_coverage"))
        matched_null_pf = _num(item.get("matched_null_median_profit_factor"))
        gate_failures = _text(item.get("deployment_gate_failures"))
        tier = _text(item.get("confidence_tier")).upper()
        hard_blocked_long_vol = strategy == "LONG_STRANGLE"
        deployment_ready = tier == "PROVEN" and not gate_failures and not hard_blocked_long_vol
        if hard_blocked_long_vol:
            status = "RESEARCH_ONLY_LONG_VOL_HARD_BLOCK"
            reason = "long-vol remains non-executable until ask-to-bid option replay clears its dedicated gate"
        elif deployment_ready:
            status = "DEPLOYMENT_READY"
            reason = ""
        elif gate_failures:
            status = "REJECTED_DEPLOYMENT_GATE"
            reason = gate_failures
        elif scored < 30:
            status = "REJECTED_LOW_SAMPLE"
            reason = f"only {scored} scored OOS outcomes; minimum is 30"
        elif avg_r is None or avg_r <= 0 or profit_factor is None or profit_factor < 1.2:
            status = "REJECTED_NEGATIVE_OR_WEAK"
            reason = "OOS expectancy is not positive with profit factor >= 1.2"
        elif fold_count <= 0 or positive_folds < fold_count:
            status = "REJECTED_FOLD_FAILURE"
            reason = f"profitable in {positive_folds}/{fold_count} validation folds"
        else:
            status = "RESEARCH_ONLY_NOT_PROVEN"
            reason = _text(item.get("validation_note")) or "family did not clear the core proof gate"
        rows.append(
            _base_row(
                pattern_id=family_id,
                pattern_name=str(item.get("base_pattern_family") or item.get("pattern_family") or "UNKNOWN"),
                pattern_class="V2_PREDECLARED_FAMILY",
                pattern_scope=pattern_scope,
                rank_scope="TRADE_PATTERN",
                source_inputs="all five canonical UW feeds",
                direction=family_parts[1] if len(family_parts) >= 2 else "",
                strategy=strategy,
                sample_test=scored,
                test_profit_factor=profit_factor,
                test_average_r=avg_r,
                test_win_rate=_num(item.get("validation_success_probability")),
                positive_folds=positive_folds,
                fold_count=fold_count,
                matched_null_profit_factor=matched_null_pf,
                matched_null_p_value=matched_null_p,
                matched_null_coverage=matched_null_coverage,
                clustered_pf_p05=clustered_p05,
                option_replay="yes",
                all_five_sources="yes",
                deployment_ready=deployment_ready,
                status=status,
                deployment_gate_failures=gate_failures,
                rejection_reason=reason,
                evidence_artifact=str(out_dir / "discovered_pattern_families.csv"),
            )
        )
    return rows


def _opening_flow_rows(base_dir: Path) -> List[Dict[str, Any]]:
    trades_path = base_dir / "out" / "opening_flow_option_trades.csv"
    null_path = base_dir / "out" / "opening_flow_option_permutations.csv"
    trades = _frame(trades_path)
    null = _frame(null_path)
    rows: List[Dict[str, Any]] = []
    required = {"direction", "signal_date", "pnl", "return_on_cost"}
    if trades.empty or not required <= set(trades.columns):
        return rows
    for direction in sorted(trades["direction"].dropna().astype(str).unique()):
        direction_rows = trades[trades["direction"].astype(str).eq(direction)].copy()
        direction_rows["sample"] = np.where(
            direction_rows["signal_date"].astype(str) >= "2026-04-14", "TEST", "TRAIN"
        )
        train = direction_rows[direction_rows["sample"].eq("TRAIN")]
        test = direction_rows[direction_rows["sample"].eq("TEST")]
        train_pf = _pf(train["pnl"])
        test_pf = _pf(test["pnl"])
        null_block = null[
            null.get("direction", pd.Series("", index=null.index)).astype(str).eq(direction)
            & null.get("sample", pd.Series("", index=null.index)).astype(str).eq("TEST")
        ] if not null.empty else pd.DataFrame()
        null_pf = _num(null_block.get("pf", pd.Series(dtype=float)).median()) if not null_block.empty else None
        p_value = (
            float((pd.to_numeric(null_block["pf"], errors="coerce") >= test_pf).mean())
            if test_pf is not None and not null_block.empty and "pf" in null_block
            else None
        )
        passes = (
            len(train) >= 30
            and len(test) >= 15
            and (train_pf or 0) >= 1.2
            and (test_pf or 0) >= 1.2
            and p_value is not None
            and p_value <= 0.05
        )
        status = "RESEARCH_ONLY_NEEDS_CLUSTERED_FOLDS" if passes else "REJECTED_EXECUTABLE_REPLAY"
        reason = (
            "passes train/test and matched null but lacks clustered/fold proof"
            if passes
            else "managed option replay failed train/test profitability or its matched null"
        )
        rows.append(
            _base_row(
                pattern_id=f"BUYER_TO_OPEN_{direction.upper()}",
                pattern_name="Buyer-to-open OI direction",
                pattern_class="LITERATURE_GROUNDED_RESEARCH",
                pattern_scope="RESEARCH_HYPOTHESIS",
                rank_scope="RESEARCH_CONTEXT",
                source_inputs="chain-oi-changes + shifted option quotes",
                direction=direction,
                strategy="long option",
                sample_train=len(train),
                sample_test=len(test),
                train_profit_factor=train_pf,
                test_profit_factor=test_pf,
                test_average_r=_num(test["return_on_cost"].mean()) if len(test) else None,
                test_win_rate=_num(test["pnl"].gt(0).mean()) if len(test) else None,
                matched_null_profit_factor=null_pf,
                matched_null_p_value=p_value,
                option_replay="yes",
                all_five_sources="no; dedicated historical panel",
                status=status,
                rejection_reason=reason,
                evidence_artifact=f"{trades_path}; {null_path}",
            )
        )
    return rows


def _research_summary_rows(base_dir: Path) -> List[Dict[str, Any]]:
    out = base_dir / "out"
    rows: List[Dict[str, Any]] = []

    fdr_path = out / "fdr_feature_scan.csv"
    fdr = _frame(fdr_path)
    if not fdr.empty:
        selected = int(fdr.get("train_fdr_selected", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        confirmed = int(fdr.get("test_bonferroni_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        rows.append(
            _base_row(
                pattern_id="FDR_SINGLE_FEATURE_SCREEN",
                pattern_name="All single-feature directional screens",
                pattern_class="MULTIPLE_TESTING_AUDIT",
                pattern_scope="RESEARCH_AUDIT",
                rank_scope="RESEARCH_CONTEXT",
                source_inputs="all five UW derived feature panel",
                sample_train=int(fdr.get("train_days", pd.Series(dtype=float)).max() or 0),
                sample_test=int(fdr.get("test_days", pd.Series(dtype=float)).max() or 0),
                option_replay="no; stock proxy",
                all_five_sources="mixed historical coverage",
                status="REJECTED_NO_FDR_DISCOVERY" if confirmed == 0 else "RESEARCH_ONLY_FDR_DISCOVERY",
                rejection_reason=f"{selected} train discoveries and {confirmed} untouched-test confirmations across {len(fdr)} tests",
                evidence_artifact=str(fdr_path),
            )
        )

    model_path = out / "walk_forward_flow_model.csv"
    model = _frame(model_path)
    if not model.empty and "net_nw_t" in model:
        best = model.sort_values("net_nw_t", ascending=False).iloc[0]
        t_stat = _num(best.get("net_nw_t"))
        spread = _num(best.get("net_spread"))
        status = "RESEARCH_ONLY_STOCK_PROXY" if (spread or 0) > 0 and (t_stat or 0) >= 2 else "REJECTED_PURGED_MODEL"
        rows.append(
            _base_row(
                pattern_id="PURGED_MULTI_FEATURE_RIDGE",
                pattern_name="Purged all-feed cross-sectional model",
                pattern_class="MULTIVARIATE_MODEL",
                pattern_scope="RESEARCH_HYPOTHESIS",
                rank_scope="RESEARCH_CONTEXT",
                source_inputs="all five UW derived feature panel",
                strategy=str(best.get("portfolio") or ""),
                sample_test=int(_num(best.get("days")) or 0),
                test_average_r=spread,
                option_replay="no; stock proxy",
                all_five_sources="mixed historical coverage",
                status=status,
                rejection_reason=f"best net Newey-West t={t_stat}; requires t>=2 plus executable option replay",
                evidence_artifact=str(model_path),
            )
        )

    vol_path = out / "volatility_spread_validation.csv"
    vol = _frame(vol_path)
    if not vol.empty:
        test = vol[(vol.get("sample") == "TEST") & (vol.get("horizon") == "5d")]
        train = vol[(vol.get("sample") == "TRAIN") & (vol.get("horizon") == "5d")]
        if len(test):
            rows.append(
                _base_row(
                    pattern_id="MATCHED_CALL_PUT_IV_SPREAD",
                    pattern_name="Matched call-put implied-volatility spread",
                    pattern_class="LITERATURE_GROUNDED_RESEARCH",
                    pattern_scope="RESEARCH_HYPOTHESIS",
                    rank_scope="RESEARCH_CONTEXT",
                    source_inputs="bot-EOD full-tape quotes",
                    direction="cross-sectional",
                    strategy="stock proxy",
                    sample_train=int(_num(train.iloc[0].get("days")) or 0) if len(train) else 0,
                    sample_test=int(_num(test.iloc[0].get("days")) or 0),
                    test_average_r=_num(test.iloc[0].get("net_spread")),
                    option_replay="no; stock proxy",
                    all_five_sources="bot-EOD + prices",
                    status="REJECTED_OOS_SIGN_FLIP",
                    rejection_reason="positive training spread reversed negative in untouched test",
                    evidence_artifact=str(vol_path),
                )
            )

    vega_path = out / "vega_demand_validation.csv"
    vega = _frame(vega_path)
    if not vega.empty:
        test = vega[vega.get("sample").eq("TEST")]
        best = test.sort_values("nw_t", ascending=False).iloc[0] if len(test) else None
        rows.append(
            _base_row(
                pattern_id="CUSTOMER_VEGA_MOVE_MAGNITUDE",
                pattern_name="Customer vega demand predicts move magnitude",
                pattern_class="VOLATILITY_CONTEXT",
                pattern_scope="CONTEXT_ONLY",
                rank_scope="RESEARCH_CONTEXT",
                source_inputs="bot-EOD signed greeks + stock screener implied move",
                direction="neutral magnitude",
                strategy="context only",
                sample_test=int(_num(best.get("days")) or 0) if best is not None else 0,
                test_average_r=_num(best.get("spread")) if best is not None else None,
                option_replay="no; long-vol option replay failed separately",
                all_five_sources="bot-EOD + stock screener",
                status="RESEARCH_ONLY_NON_EXECUTABLE",
                rejection_reason="predicts realized move magnitude but has not produced profitable ask-to-bid long-vol options",
                evidence_artifact=str(vega_path),
            )
        )

    symmetric_path = out / "symmetric_direction_test.csv"
    if symmetric_path.exists():
        rows.append(
            _base_row(
                pattern_id="LEGACY_TECH_WEAK_MOMENTUM_PUTS",
                pattern_name="Technology weak-momentum long puts",
                pattern_class="LEGACY_RESEARCH_LANE",
                pattern_scope="INVALIDATED_LEGACY",
                rank_scope="RESEARCH_CONTEXT",
                source_inputs="partial-date panel + chain/OI quotes",
                direction="bearish",
                strategy="long put",
                option_replay="yes, but invalid input timing",
                all_five_sources="no",
                status="INVALIDATED_REQUIRES_CLEAN_REPLAY",
                rejection_reason=(
                    "legacy replay admitted dates without all five files and selected contracts using curr_oi "
                    "from the following file; rerun with shifted last_oi before use"
                ),
                evidence_artifact=str(symmetric_path),
            )
        )
    return rows


def _sort_key(row: Dict[str, Any]) -> tuple:
    status = str(row.get("status") or "")
    status_rank = (
        5 if row.get("deployment_ready") else
        4 if status.startswith("RESEARCH_ONLY") else
        2 if status.startswith("REJECTED_LOW_SAMPLE") else
        1 if status.startswith("REJECTED") else
        0
    )
    return (
        status_rank,
        1 if str(row.get("pattern_scope") or "") == "FAMILY" else 0,
        1 if int(_num(row.get("sample_test")) or 0) >= 20 else 0,
        _num(row.get("test_average_r")) or -999.0,
        min(_num(row.get("test_profit_factor")) or -999.0, 20.0),
        int(_num(row.get("sample_test")) or 0),
    )


def build_pattern_registry(base_dir: Path, out_dir: Path, as_of: str) -> Dict[str, str]:
    rows = _engine_rows(out_dir)
    rows.extend(_opening_flow_rows(base_dir))
    rows.extend(_research_summary_rows(base_dir))
    rows.sort(key=_sort_key, reverse=True)
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    trade_rows = [row for row in rows if row.get("rank_scope") == "TRADE_PATTERN"]
    trade_rows.sort(key=_sort_key, reverse=True)
    for index, row in enumerate(trade_rows, 1):
        row["trade_pattern_rank"] = index
    research_rows = [row for row in rows if row.get("rank_scope") == "RESEARCH_CONTEXT"]
    research_rows.sort(key=_sort_key, reverse=True)
    for index, row in enumerate(research_rows, 1):
        row["research_rank"] = index

    broad_adequate = [
        row
        for row in trade_rows
        if row.get("pattern_scope") == "FAMILY"
        and int(_num(row.get("sample_test")) or 0) >= 20
    ]
    broad_adequate.sort(
        key=lambda row: (
            _num(row.get("test_average_r")) or -999.0,
            min(_num(row.get("test_profit_factor")) or -999.0, 20.0),
            int(_num(row.get("sample_test")) or 0),
        ),
        reverse=True,
    )

    registry_path = out_dir / "pattern_registry.csv"
    pd.DataFrame(rows, columns=REGISTRY_FIELDS).to_csv(registry_path, index=False)
    summary = {
        "as_of": as_of,
        "ranking_is_probability": False,
        "ranking_basis": "observed OOS economics; all deployment gates remain separate",
        "patterns_ranked": len(rows),
        "deployment_ready_count": sum(bool(row.get("deployment_ready")) for row in rows),
        "best_trade_pattern_adequate_sample": broad_adequate[0] if broad_adequate else None,
        "worst_trade_pattern_adequate_sample": broad_adequate[-1] if broad_adequate else None,
        "best_research_context": research_rows[0] if research_rows else None,
        "adequate_sample_definition": "broad FAMILY scope with at least 20 scored OOS outcomes",
        "status_counts": pd.Series([row.get("status") for row in rows]).value_counts().to_dict() if rows else {},
    }
    summary_path = out_dir / "pattern_ranking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return {
        "pattern_registry": str(registry_path),
        "pattern_ranking_summary": str(summary_path),
    }
