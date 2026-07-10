from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float


def setup_family(strategy: object, direction: object = "") -> str:
    text = f"{strategy or ''} {direction or ''}".lower()
    if "earn" in text:
        return "earnings-risk trades"
    if "hedge" in text or "collar" in text:
        return "hedges/rolls"
    if "covered" in text or "income" in text or "cash-secured" in text or "csp" in text:
        return "portfolio income"
    if "debit" in text or "bull call" in text or "bear put" in text:
        return "debit spreads"
    if "credit" in text or "bull put" in text or "bear call" in text:
        return "credit spreads"
    if "roll" in text:
        return "hedges/rolls"
    return "other"


def summarize_recent_replay(detail: pd.DataFrame, *, window: int = 20) -> dict[str, Any]:
    if detail.empty:
        return {"status": "unavailable", "reason": "empty_replay_detail"}
    df = detail.copy()
    for col in ["exact_evaluated"]:
        if col not in df.columns:
            return {"status": "unavailable", "reason": f"missing_{col}"}
    selection_col = "decision_pass" if "decision_pass" in df.columns else "replay_guard_pass"
    if selection_col not in df.columns:
        return {"status": "unavailable", "reason": f"missing_{selection_col}"}
    exact_mask = df["exact_evaluated"].astype(str).str.lower().eq("true")
    selected_mask = df[selection_col].astype(str).str.lower().eq("true")
    df = df[exact_mask & selected_mask].copy()
    if df.empty:
        return {"status": "unavailable", "reason": f"no_{selection_col}_replay_trades"}
    if "asof" in df.columns:
        df["asof"] = pd.to_datetime(df["asof"], errors="coerce")
        df = df.sort_values(["asof", "ticker"] if "ticker" in df.columns else ["asof"])
    recent = df.tail(window).copy()
    win_rate = float(recent["exact_win"].mean()) if "exact_win" in recent.columns else None
    avg_pnl = float(pd.to_numeric(recent["pnl_1x"], errors="coerce").mean()) if "pnl_1x" in recent.columns else None
    total_pnl = float(pd.to_numeric(recent["pnl_1x"], errors="coerce").sum()) if "pnl_1x" in recent.columns else None
    if avg_pnl is None or win_rate is None:
        stance = "neutral"
    elif avg_pnl < 0 or win_rate < 0.55:
        stance = "degrading"
    elif avg_pnl > 0 and win_rate >= 0.60:
        stance = "strong"
    else:
        stance = "neutral"
    return {
        "status": "ok",
        "stance": stance,
        "window": int(len(recent)),
        "win_rate": win_rate,
        "avg_pnl_1x": avg_pnl,
        "total_pnl_1x": total_pnl,
        "latest_asof": str(recent["asof"].max().date()) if "asof" in recent.columns and pd.notna(recent["asof"].max()) else "",
    }


def load_recent_performance(
    out_root: Path,
    *,
    window: int = 20,
    asof: dt.date | None = None,
    history_namespace: str | None = None,
) -> dict[str, Any]:
    if history_namespace:
        from .edge_model import load_replay_edge_history

        detail = load_replay_edge_history(
            out_root,
            asof=asof or dt.date.max,
            history_namespace=history_namespace,
        )
        if detail.empty:
            return {"status": "unavailable", "reason": "namespaced_replay_history_unavailable"}
        summary = summarize_recent_replay(detail, window=window)
        summary["history_namespace"] = history_namespace
        sources = sorted(set(detail.get("edge_source_file", pd.Series(dtype=str)).dropna().astype(str)))
        summary["source"] = sources[-1] if sources else ""
        return summary

    patterns = [
        "codexuw_audit_decision_select_*/codexuw_replay_detail.csv",
        "codexuw_replay_*decision*/codexuw_replay_detail.csv",
        "codexuw_replay_2026_full_available*/codexuw_replay_detail.csv",
        "codexuw_replay_*_guard_v*/codexuw_replay_detail.csv",
    ]
    candidates = []
    seen = set()
    for pattern in patterns:
        matches = sorted(out_root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in matches:
            if path not in seen:
                candidates.append(path)
                seen.add(path)
        if candidates:
            break
    if not candidates:
        return {"status": "unavailable", "reason": "no_replay_detail_found"}
    path = candidates[0]
    try:
        detail = pd.read_csv(path)
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc), "source": str(path)}
    summary = summarize_recent_replay(detail, window=window)
    summary["source"] = str(path)
    return summary


def summarize_live_outcomes(ledger: pd.DataFrame, *, window: int = 30) -> dict[str, Any]:
    if ledger.empty:
        return {"status": "unavailable", "reason": "empty_live_outcome_ledger"}
    df = ledger.copy()
    if "realized_pnl" not in df.columns:
        return {"status": "unavailable", "reason": "missing_realized_pnl"}
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    realized = df[df["realized_pnl"].notna()].copy()
    if realized.empty:
        return {
            "status": "unavailable",
            "reason": "no_realized_live_outcomes",
            "ledger_rows": int(len(df)),
        }
    date_col = "report_date" if "report_date" in realized.columns else "asof" if "asof" in realized.columns else ""
    if date_col:
        realized[date_col] = pd.to_datetime(realized[date_col], errors="coerce")
        realized = realized.sort_values(date_col)
    recent = realized.tail(window).copy()
    if "setup_family" not in recent.columns:
        recent["setup_family"] = recent.apply(lambda row: setup_family(row.get("strategy"), row.get("direction")), axis=1)
    family_summary: dict[str, Any] = {}
    for family, part in recent.groupby("setup_family"):
        pnl = pd.to_numeric(part["realized_pnl"], errors="coerce").dropna()
        if pnl.empty:
            continue
        wins = int((pnl > 0).sum())
        outcomes = int(len(pnl))
        avg_pnl = float(pnl.mean())
        total_pnl = float(pnl.sum())
        expectancy = "negative" if outcomes >= 3 and avg_pnl < 0 else "positive" if outcomes >= 3 and avg_pnl > 0 else "insufficient"
        family_summary[str(family)] = {
            "outcomes": outcomes,
            "wins": wins,
            "win_rate": wins / outcomes if outcomes else math.nan,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "expectancy": expectancy,
        }
    win_rate = float((recent["realized_pnl"] > 0).mean()) if not recent.empty else math.nan
    latest = ""
    if date_col and recent[date_col].notna().any():
        latest = str(recent[date_col].max().date())
    return {
        "status": "ok",
        "window": int(len(recent)),
        "win_rate": win_rate,
        "avg_pnl": float(recent["realized_pnl"].mean()),
        "total_pnl": float(recent["realized_pnl"].sum()),
        "latest_report_date": latest,
        "family_summary": family_summary,
    }


def load_live_outcome_performance(out_root: Path, *, window: int = 30) -> dict[str, Any]:
    candidates = [
        out_root / "codexuw_recommendation_outcome_ledger.csv",
        out_root / "codexuw_execute_outcome_ledger.csv",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {"status": "unavailable", "reason": "no_live_outcome_ledger_found"}
    path = existing[0]
    try:
        ledger = pd.read_csv(path)
    except Exception as exc:
        return {"status": "unavailable", "reason": str(exc), "source": str(path)}
    summary = summarize_live_outcomes(ledger, window=window)
    summary["source"] = str(path)
    try:
        summary["ledger_mtime_utc"] = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
    except OSError:
        pass
    return summary


def live_outcome_adjustment(context: dict[str, Any] | None, strategy: object, direction: object = "") -> dict[str, Any]:
    family = setup_family(strategy, direction)
    if not context or context.get("status") != "ok":
        return {"family": family, "status": "unavailable", "score_penalty": 0.0, "block_execute": False}
    family_summary = (context.get("family_summary") or {}).get(family, {})
    expectancy = family_summary.get("expectancy")
    if expectancy == "negative":
        return {
            "family": family,
            "status": "negative_expectancy",
            "score_penalty": 1.5,
            "block_execute": True,
            "summary": family_summary,
        }
    return {
        "family": family,
        "status": expectancy or "insufficient",
        "score_penalty": 0.0,
        "block_execute": False,
        "summary": family_summary,
    }


def performance_risk_multiplier(context: dict[str, Any] | None) -> float:
    if not context or context.get("status") != "ok":
        return 1.0
    return 0.75 if context.get("stance") == "degrading" else 1.0


def performance_min_score(context: dict[str, Any] | None, base: float) -> float:
    if not context or context.get("status") != "ok":
        return base
    return max(base, 5.5) if context.get("stance") == "degrading" else base
