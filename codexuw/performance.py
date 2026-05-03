from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float


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


def load_recent_performance(out_root: Path, *, window: int = 20) -> dict[str, Any]:
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


def performance_risk_multiplier(context: dict[str, Any] | None) -> float:
    if not context or context.get("status") != "ok":
        return 1.0
    return 0.75 if context.get("stance") == "degrading" else 1.0


def performance_min_score(context: dict[str, Any] | None, base: float) -> float:
    if not context or context.get("status") != "ok":
        return base
    return max(base, 5.5) if context.get("stance") == "degrading" else base
