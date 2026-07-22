from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .credit_policy import MAX_CREDIT_PCT_WIDTH, MIN_CREDIT_PCT_WIDTH
from .data import safe_float


MAX_REPLAY_AGE_DAYS = 30
MIN_LIVE_OUTCOMES_FOR_SIZE_UP = 50
MIN_LIVE_FAMILY_OUTCOMES_FOR_SIZE_UP = 12
MIN_LIVE_PROFIT_FACTOR_FOR_SIZE_UP = 1.25


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
    selection_cols = [col for col in ["decision_pass", "replay_guard_pass"] if col in df.columns]
    if not selection_cols:
        return {"status": "unavailable", "reason": "missing_decision_and_replay_guard"}
    exact_mask = df["exact_evaluated"].astype(str).str.lower().eq("true")
    selected_mask = pd.Series(True, index=df.index)
    for col in selection_cols:
        selected_mask &= df[col].astype(str).str.lower().eq("true")
    df = df[exact_mask & selected_mask].copy()
    if df.empty:
        return {"status": "unavailable", "reason": "no_guarded_decision_replay_trades"}
    policy_mismatch_excluded = 0
    if "entry_credit_pct_width" in df.columns:
        credit_pct = pd.to_numeric(df["entry_credit_pct_width"], errors="coerce")
        direction = df.get("direction", pd.Series("", index=df.index)).astype(str)
        strategy = df.get("strategy", pd.Series("", index=df.index)).astype(str)
        credit_rows = direction.isin({"Bull Put", "Bear Call"}) | strategy.str.contains("Credit", case=False, regex=False)
        if not credit_rows.any():
            credit_rows = credit_pct.notna()
        policy_match = (~credit_rows) | credit_pct.between(
            MIN_CREDIT_PCT_WIDTH,
            MAX_CREDIT_PCT_WIDTH,
            inclusive="both",
        )
        policy_mismatch_excluded = int((~policy_match).sum())
        df = df[policy_match].copy()
    if df.empty:
        return {
            "status": "unavailable",
            "reason": "no_current_credit_policy_replay_trades",
            "policy_min_credit_pct_width": MIN_CREDIT_PCT_WIDTH,
            "policy_mismatch_excluded": policy_mismatch_excluded,
        }
    if "asof" in df.columns:
        df["asof"] = pd.to_datetime(df["asof"], errors="coerce")
        df = df.sort_values(["asof", "ticker"] if "ticker" in df.columns else ["asof"])
    recent = df.tail(window).copy()
    win_rate = float(recent["exact_win"].mean()) if "exact_win" in recent.columns else None
    avg_pnl = float(pd.to_numeric(recent["pnl_1x"], errors="coerce").mean()) if "pnl_1x" in recent.columns else None
    total_pnl = float(pd.to_numeric(recent["pnl_1x"], errors="coerce").sum()) if "pnl_1x" in recent.columns else None
    pnl = pd.to_numeric(recent.get("pnl_1x"), errors="coerce").dropna()
    gross_profit = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
    gross_loss = float(-pnl[pnl < 0].sum()) if not pnl.empty else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else None)
    equity = pd.concat([pd.Series([0.0]), pnl.cumsum()], ignore_index=True)
    max_drawdown = float((equity - equity.cummax()).min()) if not equity.empty else 0.0
    if avg_pnl is None or win_rate is None:
        stance = "neutral"
    elif avg_pnl < 0 or (profit_factor is not None and profit_factor < 1.0):
        stance = "degrading"
    elif avg_pnl > 0 and profit_factor is not None and profit_factor >= 1.25 and win_rate >= 0.55:
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
        "profit_factor": profit_factor,
        "max_drawdown_1x": max_drawdown,
        "selection_filter": "+".join(selection_cols),
        "policy_min_credit_pct_width": MIN_CREDIT_PCT_WIDTH,
        "policy_mismatch_excluded": policy_mismatch_excluded,
        "latest_asof": str(recent["asof"].max().date()) if "asof" in recent.columns and pd.notna(recent["asof"].max()) else "",
    }


def _apply_replay_freshness(
    summary: dict[str, Any],
    *,
    asof: dt.date | None,
    max_age_days: int,
) -> dict[str, Any]:
    if asof is None or summary.get("status") != "ok":
        return summary
    latest = pd.to_datetime(summary.get("latest_asof"), errors="coerce")
    if pd.isna(latest):
        summary.update({"status": "unavailable", "stance": "unavailable", "reason": "missing_replay_latest_asof"})
        return summary
    age_days = max(0, (asof - latest.date()).days)
    summary["age_days"] = age_days
    summary["max_age_days"] = max_age_days
    if age_days > max_age_days:
        summary["prior_stance"] = summary.get("stance")
        summary["status"] = "stale"
        summary["stance"] = "unavailable"
        summary["reason"] = f"replay_is_{age_days}_days_old; maximum_allowed_is_{max_age_days}"
    return summary


def _filter_replay_point_in_time(detail: pd.DataFrame, asof: dt.date | None) -> pd.DataFrame:
    if asof is None or detail.empty or "asof" not in detail.columns:
        return detail
    cutoff = pd.Timestamp(asof)
    source_day = pd.to_datetime(detail["asof"], errors="coerce")
    mask = source_day.lt(cutoff)
    if "exit_day" in detail.columns:
        exit_day = pd.to_datetime(detail["exit_day"], errors="coerce")
        mask &= exit_day.lt(cutoff)
    return detail[mask].copy()


def load_recent_performance(
    out_root: Path,
    *,
    window: int = 20,
    asof: dt.date | None = None,
    history_namespace: str | None = None,
    max_age_days: int = MAX_REPLAY_AGE_DAYS,
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
        detail = _filter_replay_point_in_time(detail, asof)
        summary = summarize_recent_replay(detail, window=window)
        summary["history_namespace"] = history_namespace
        sources = sorted(set(detail.get("edge_source_file", pd.Series(dtype=str)).dropna().astype(str)))
        summary["source"] = sources[-1] if sources else ""
        return _apply_replay_freshness(summary, asof=asof, max_age_days=max_age_days)

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
    detail = _filter_replay_point_in_time(detail, asof)
    summary = summarize_recent_replay(detail, window=window)
    summary["source"] = str(path)
    return _apply_replay_freshness(summary, asof=asof, max_age_days=max_age_days)


def summarize_live_outcomes(ledger: pd.DataFrame, *, window: int = MIN_LIVE_OUTCOMES_FOR_SIZE_UP) -> dict[str, Any]:
    if ledger.empty:
        return {"status": "unavailable", "reason": "empty_live_outcome_ledger"}
    df = ledger.copy()
    if "realized_pnl" not in df.columns:
        return {"status": "unavailable", "reason": "missing_realized_pnl"}
    df["realized_pnl"] = pd.to_numeric(df["realized_pnl"], errors="coerce")
    realized = df[df["realized_pnl"].notna()].copy()
    excluded_nonexecuted = 0
    if not realized.empty and "outcome_status" in realized.columns:
        generated_statuses = {"OPEN_REVIEW_REQUIRED", "CONDITIONAL_NOT_FILLED", "NOT_EXECUTED"}
        statuses = realized["outcome_status"].fillna("").astype(str).str.strip()
        terminal = statuses.ne("") & ~statuses.isin(generated_statuses)
        if "actual_fill" in realized.columns:
            actual_fill = pd.to_numeric(realized["actual_fill"], errors="coerce").notna()
        else:
            actual_fill = pd.Series(False, index=realized.index)
        eligible = terminal | actual_fill
        excluded_nonexecuted = int((~eligible).sum())
        realized = realized[eligible].copy()
    if realized.empty:
        return {
            "status": "unavailable",
            "reason": "no_closed_filled_live_outcomes",
            "ledger_rows": int(len(df)),
            "excluded_nonexecuted_realized_rows": excluded_nonexecuted,
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
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else math.nan)
        expectancy = "negative" if outcomes >= 3 and avg_pnl < 0 else "positive" if outcomes >= 3 and avg_pnl > 0 else "insufficient"
        family_summary[str(family)] = {
            "outcomes": outcomes,
            "wins": wins,
            "win_rate": wins / outcomes if outcomes else math.nan,
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
        }
    win_rate = float((recent["realized_pnl"] > 0).mean()) if not recent.empty else math.nan
    pnl = pd.to_numeric(recent["realized_pnl"], errors="coerce").dropna()
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else math.nan)
    equity = pd.concat([pd.Series([0.0]), pnl.cumsum()], ignore_index=True)
    max_drawdown = float((equity - equity.cummax()).min()) if not equity.empty else 0.0
    avg_pnl = float(pnl.mean()) if not pnl.empty else math.nan
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if not wins.empty else math.nan
    avg_loss = float(losses.mean()) if not losses.empty else math.nan
    size_up_allowed = bool(
        len(pnl) >= MIN_LIVE_OUTCOMES_FOR_SIZE_UP
        and math.isfinite(avg_pnl)
        and avg_pnl > 0
        and (math.isinf(profit_factor) or profit_factor >= MIN_LIVE_PROFIT_FACTOR_FOR_SIZE_UP)
    )
    latest = ""
    if date_col and recent[date_col].notna().any():
        latest = str(recent[date_col].max().date())
    return {
        "status": "ok",
        "window": int(len(recent)),
        "realized_outcome_count": int(len(realized)),
        "excluded_nonexecuted_realized_rows": excluded_nonexecuted,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "total_pnl": float(pnl.sum()),
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "minimum_outcomes_for_size_up": MIN_LIVE_OUTCOMES_FOR_SIZE_UP,
        "size_up_allowed": size_up_allowed,
        "latest_report_date": latest,
        "family_summary": family_summary,
    }


def load_live_outcome_performance(
    out_root: Path,
    *,
    window: int = MIN_LIVE_OUTCOMES_FOR_SIZE_UP,
    asof: dt.date | None = None,
) -> dict[str, Any]:
    candidates = [
        out_root / "codexdaily_v3_recommendation_outcome_ledger.csv",
        out_root / "codexuw_recommendation_outcome_ledger.csv",
        out_root / "codexuw_execute_outcome_ledger.csv",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return {"status": "unavailable", "reason": "no_live_outcome_ledger_found"}
    attempts: list[dict[str, Any]] = []
    for path in existing:
        try:
            ledger = pd.read_csv(path)
        except Exception as exc:
            attempts.append({"source": str(path), "status": "unavailable", "reason": str(exc), "ledger_rows": 0})
            continue
        if asof is not None:
            date_col = "report_date" if "report_date" in ledger.columns else "asof" if "asof" in ledger.columns else ""
            if date_col:
                report_day = pd.to_datetime(ledger[date_col], errors="coerce")
                ledger = ledger[report_day.lt(pd.Timestamp(asof))].copy()
        summary = summarize_live_outcomes(ledger, window=window)
        summary["source"] = str(path)
        try:
            summary["ledger_mtime_utc"] = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
        except OSError:
            pass
        if summary.get("status") == "ok":
            summary["sources_checked"] = [str(item) for item in existing]
            return summary
        attempts.append(summary)
    return {
        "status": "unavailable",
        "reason": "no_closed_filled_live_outcomes",
        "ledger_rows": int(sum(safe_float(item.get("ledger_rows"), 0.0) for item in attempts)),
        "source": str(existing[0]),
        "sources_checked": [str(item) for item in existing],
        "attempts": attempts,
        "size_up_allowed": False,
        "minimum_outcomes_for_size_up": MIN_LIVE_OUTCOMES_FOR_SIZE_UP,
    }


def live_outcome_adjustment(context: dict[str, Any] | None, strategy: object, direction: object = "") -> dict[str, Any]:
    family = setup_family(strategy, direction)
    if not context or context.get("status") != "ok":
        return {
            "family": family,
            "status": "unavailable",
            "score_penalty": 0.0,
            "block_execute": False,
            "block_size_up": True,
        }
    family_summary = (context.get("family_summary") or {}).get(family, {})
    expectancy = family_summary.get("expectancy")
    if expectancy == "negative":
        return {
            "family": family,
            "status": "negative_expectancy",
            "score_penalty": 1.5,
            "block_execute": True,
            "block_size_up": True,
            "summary": family_summary,
        }
    family_outcomes = int(safe_float(family_summary.get("outcomes"), 0.0))
    family_avg = safe_float(family_summary.get("avg_pnl"), math.nan)
    try:
        family_pf = float(family_summary.get("profit_factor"))
    except (TypeError, ValueError):
        family_pf = math.nan
    family_size_up_allowed = bool(
        context.get("size_up_allowed")
        and family_outcomes >= MIN_LIVE_FAMILY_OUTCOMES_FOR_SIZE_UP
        and math.isfinite(family_avg)
        and family_avg > 0
        and (math.isinf(family_pf) or (math.isfinite(family_pf) and family_pf >= MIN_LIVE_PROFIT_FACTOR_FOR_SIZE_UP))
    )
    return {
        "family": family,
        "status": expectancy or "insufficient",
        "score_penalty": 0.0,
        "block_execute": False,
        "block_size_up": not family_size_up_allowed,
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
