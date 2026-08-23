from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float
from .replay import (
    _decision_sort_score,
    _expiry_spread_value,
    _quote_lookup,
    _spread_mid_debit,
    _spread_mid_value,
    dated_folders,
    future_close,
    load_close_history,
    load_hot_history,
)


GOAL_SHADOW_SCHEMA_VERSION = "codexuw.goal_shadow.v1"
GOAL_SHADOW_POLICY_ID = "n1_fillable_dp0.20_oi-none_hist0.0_prior8_model0.0"
GOAL_SHADOW_DP_WEIGHT = 0.20
GOAL_SHADOW_LEDGER_NAME = "codexdaily_v4_goal_shadow_ledger.csv"
RESOLVED_OUTCOME_STATES = {"RESOLVED_WIN", "RESOLVED_LOSS", "RESOLVED_FLAT"}

SHADOW_COLUMNS = [
    "schema_version",
    "policy_id",
    "producer",
    "generated_at_utc",
    "asof",
    "shadow_rank",
    "shadow_status",
    "shadow_only",
    "execution_eligible",
    "no_order_placement",
    "ticker",
    "strategy",
    "strategy_kind",
    "direction",
    "regime",
    "expiry",
    "dte",
    "short_leg_eod",
    "long_leg_eod",
    "short_strike_eod",
    "long_strike_eod",
    "stock_price_eod",
    "entry_side",
    "entry_price",
    "entry_mid",
    "entry_natural",
    "entry_width",
    "entry_pct_width",
    "entry_quote_width_pct",
    "max_loss",
    "target_exit_value",
    "stop_exit_value",
    "goal_score",
    "option_flow_bias",
    "dp_flow_bias",
    "dp_directional_ratio",
    "effective_flow_bias",
    "flow_quality",
    "oi_carryover_status",
    "bot_flow_source_status",
    "dark_pool_source_status",
    "source_scored_file",
    "outcome_status",
    "outcome_last_checked",
    "exit_day",
    "exit_reason",
    "exit_value",
    "pnl_1x",
    "return_on_risk",
    "exact_win",
    "outcome_note",
]


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _clean(value).lower() in {"1", "true", "yes", "y"}


def _strategy_kind(row: pd.Series | dict[str, Any]) -> str:
    explicit = _clean(row.get("strategy_kind")).title()
    if explicit in {"Credit", "Debit"}:
        return explicit
    strategy = _clean(row.get("strategy")).lower()
    direction = _clean(row.get("direction"))
    return "Credit" if "credit" in strategy or direction in {"Bull Put", "Bear Call"} else "Debit"


def _first_number(row: pd.Series | dict[str, Any], names: list[str]) -> float:
    for name in names:
        value = safe_float(row.get(name))
        if math.isfinite(value):
            return value
    return math.nan


def _dp_source_status(row: pd.Series | dict[str, Any]) -> str:
    return _clean(row.get("dark_pool_source_status") or row.get("dp_source_status"))


def _dp_loaded(row: pd.Series | dict[str, Any]) -> bool:
    status = _dp_source_status(row).lower()
    return status.startswith("dp_eod") or status == "loaded"


def _effective_flow_bias(row: pd.Series | dict[str, Any]) -> tuple[float, float, float]:
    option_bias = _first_number(row, ["option_flow_bias", "combined_flow_bias", "flow_bias"])
    if not math.isfinite(option_bias):
        option_bias = 0.0
    dp_bias = safe_float(row.get("dp_flow_bias"))
    dp_ratio = safe_float(row.get("dp_directional_ratio"))
    if not math.isfinite(dp_ratio):
        total = safe_float(row.get("dp_total_premium"))
        prints = safe_float(row.get("dp_prints"), safe_float(row.get("dp_trades")))
        if math.isfinite(total) and total > 0 and math.isfinite(prints) and prints > 0:
            dp_ratio = 1.0
    if not _dp_loaded(row) or not math.isfinite(dp_bias) or not math.isfinite(dp_ratio) or dp_ratio < 0.25:
        return option_bias, dp_bias, math.nan
    effective = option_bias * (1.0 - GOAL_SHADOW_DP_WEIGHT) + dp_bias * GOAL_SHADOW_DP_WEIGHT
    return effective, dp_bias, dp_ratio


def _normalized_policy_row(row: pd.Series) -> pd.Series:
    out = row.copy()
    kind = _strategy_kind(row)
    effective, dp_bias, dp_ratio = _effective_flow_bias(row)
    out["strategy_kind"] = kind
    out["entry_width"] = _first_number(row, ["entry_width", "spread_width", "preferred_width"])
    out["entry_credit"] = _first_number(row, ["natural_credit", "credit", "entry_credit"])
    out["entry_debit"] = _first_number(row, ["natural_debit", "debit", "entry_debit"])
    out["entry_credit_pct_width"] = _first_number(row, ["entry_credit_pct_width", "credit_pct_width"])
    out["entry_debit_pct_width"] = _first_number(row, ["entry_debit_pct_width", "debit_pct_width"])
    out["entry_quote_width_pct"] = _first_number(row, ["entry_quote_width_pct", "quote_width_pct"])
    out["stock_price_eod"] = _first_number(row, ["stock_price_eod", "underlying_price", "close"])
    out["short_strike_eod"] = _first_number(row, ["short_strike_eod", "short_strike"])
    out["long_strike_eod"] = _first_number(row, ["long_strike_eod", "long_strike"])
    out["regime"] = _clean(row.get("regime") or row.get("regime_trend"))
    out["combined_flow_bias"] = effective
    out["option_flow_bias"] = _first_number(row, ["option_flow_bias", "combined_flow_bias", "flow_bias"])
    out["dp_flow_bias"] = dp_bias
    out["dp_directional_ratio"] = dp_ratio
    return out


def _shadow_eligibility_reason(row: pd.Series) -> str:
    if _clean(row.get("live_status")).upper() != "PASS":
        return "live_quote_not_pass"
    if not _dp_loaded(row):
        return "dark_pool_source_not_loaded"
    effective, dp_bias, dp_ratio = _effective_flow_bias(row)
    if not all(math.isfinite(value) for value in (effective, dp_bias, dp_ratio)):
        return "dark_pool_direction_unavailable"
    blockers = ";".join(
        [
            _clean(row.get("hard_rejects")),
            _clean(row.get("penalties")),
            _clean(row.get("trade_status_reason")),
        ]
    ).lower()
    if _truthy(row.get("earnings_before_expiry")) or "earnings_crosses_expiry" in blockers or "earnings through expiry" in blockers:
        return "earnings_crosses_expiry"
    if not _clean(row.get("short_leg_eod")) or not _clean(row.get("long_leg_eod")):
        return "missing_exact_legs"
    if pd.isna(pd.to_datetime(row.get("expiry"), errors="coerce")):
        return "missing_expiry"
    normalized = _normalized_policy_row(row)
    width = safe_float(normalized.get("entry_width"))
    quote_width = safe_float(normalized.get("entry_quote_width_pct"))
    entry_field = "entry_credit" if _strategy_kind(normalized) == "Credit" else "entry_debit"
    entry = safe_float(normalized.get(entry_field))
    if not math.isfinite(width) or width <= 0 or not math.isfinite(entry) or not 0 < entry < width:
        return "invalid_entry_economics"
    if not math.isfinite(quote_width) or quote_width > 0.80:
        return "entry_quote_width_above_0.80"
    return "eligible"


def build_goal_shadow_candidates(
    scored: pd.DataFrame,
    *,
    asof: dt.date,
    source_scored_file: str = "",
) -> pd.DataFrame:
    """Return exactly one research-only candidate for the fixed prospective policy."""

    if scored is None or scored.empty:
        return pd.DataFrame(columns=SHADOW_COLUMNS)
    eligible_rows: list[pd.Series] = []
    for _, source in scored.iterrows():
        if _shadow_eligibility_reason(source) != "eligible":
            continue
        row = _normalized_policy_row(source)
        row["_goal_score"] = _decision_sort_score(row)
        eligible_rows.append(row)
    if not eligible_rows:
        return pd.DataFrame(columns=SHADOW_COLUMNS)
    ranked = pd.DataFrame(eligible_rows).sort_values(
        ["_goal_score", "flow_total_premium", "ticker"],
        ascending=[False, False, True],
        na_position="last",
    )
    row = ranked.iloc[0]
    kind = _strategy_kind(row)
    width = safe_float(row.get("entry_width"))
    if kind == "Credit":
        entry_price = safe_float(row.get("entry_credit"))
        entry_mid = _first_number(row, ["mid_credit", "entry_mid_credit"])
        entry_natural = _first_number(row, ["natural_credit", "entry_natural_credit", "credit"])
        entry_pct = safe_float(row.get("entry_credit_pct_width"))
        target_exit = entry_price * 0.40
        stop_exit = entry_price * 2.0
        entry_side = "CREDIT"
    else:
        entry_price = safe_float(row.get("entry_debit"))
        entry_mid = _first_number(row, ["mid_debit", "entry_mid_debit"])
        entry_natural = _first_number(row, ["natural_debit", "entry_natural_debit", "debit"])
        entry_pct = safe_float(row.get("entry_debit_pct_width"))
        target_exit = min(width, entry_price * 1.60)
        stop_exit = entry_price * 0.50
        entry_side = "DEBIT"
    record = {
        "schema_version": GOAL_SHADOW_SCHEMA_VERSION,
        "policy_id": GOAL_SHADOW_POLICY_ID,
        "producer": "codexuw.goal_shadow",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "asof": asof.isoformat(),
        "shadow_rank": 1,
        "shadow_status": "SHADOW_ONLY",
        "shadow_only": True,
        "execution_eligible": False,
        "no_order_placement": True,
        "ticker": _clean(row.get("ticker")).upper(),
        "strategy": _clean(row.get("strategy")),
        "strategy_kind": kind,
        "direction": _clean(row.get("direction")),
        "regime": _clean(row.get("regime") or row.get("regime_trend")),
        "expiry": pd.to_datetime(row.get("expiry"), errors="coerce").date().isoformat(),
        "dte": safe_float(row.get("dte")),
        "short_leg_eod": _clean(row.get("short_leg_eod")),
        "long_leg_eod": _clean(row.get("long_leg_eod")),
        "short_strike_eod": safe_float(row.get("short_strike_eod")),
        "long_strike_eod": safe_float(row.get("long_strike_eod")),
        "stock_price_eod": safe_float(row.get("stock_price_eod")),
        "entry_side": entry_side,
        "entry_price": entry_price,
        "entry_mid": entry_mid,
        "entry_natural": entry_natural,
        "entry_width": width,
        "entry_pct_width": entry_pct,
        "entry_quote_width_pct": safe_float(row.get("entry_quote_width_pct")),
        "max_loss": safe_float(row.get("max_loss")),
        "target_exit_value": target_exit,
        "stop_exit_value": stop_exit,
        "goal_score": safe_float(row.get("_goal_score")),
        "option_flow_bias": safe_float(row.get("option_flow_bias")),
        "dp_flow_bias": safe_float(row.get("dp_flow_bias")),
        "dp_directional_ratio": safe_float(row.get("dp_directional_ratio")),
        "effective_flow_bias": safe_float(row.get("combined_flow_bias")),
        "flow_quality": _clean(row.get("flow_quality")),
        "oi_carryover_status": _clean(row.get("oi_carryover_status")),
        "bot_flow_source_status": _clean(row.get("bot_flow_source_status")),
        "dark_pool_source_status": _dp_source_status(row),
        "source_scored_file": source_scored_file,
        "outcome_status": "PENDING",
        "outcome_last_checked": "",
        "exit_day": "",
        "exit_reason": "",
        "exit_value": math.nan,
        "pnl_1x": math.nan,
        "return_on_risk": math.nan,
        "exact_win": "",
        "outcome_note": "awaiting future point-in-time quote history",
    }
    return pd.DataFrame([record], columns=SHADOW_COLUMNS)


def _ledger_key(frame: pd.DataFrame) -> pd.Series:
    columns = ["policy_id", "asof", "ticker", "direction", "expiry", "short_strike_eod", "long_strike_eod"]
    return frame.reindex(columns=columns, fill_value="").astype(str).agg("|".join, axis=1)


def update_goal_shadow_ledger(ledger_path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    existing = (
        pd.read_csv(ledger_path, low_memory=False).reindex(columns=SHADOW_COLUMNS)
        if ledger_path.exists()
        else pd.DataFrame(columns=SHADOW_COLUMNS)
    )
    incoming = (
        new_rows.reindex(columns=SHADOW_COLUMNS).copy()
        if new_rows is not None and not new_rows.empty
        else pd.DataFrame(columns=SHADOW_COLUMNS)
    )
    if not existing.empty:
        existing["_key"] = _ledger_key(existing)
        existing["_resolved"] = existing["outcome_status"].astype(str).isin(RESOLVED_OUTCOME_STATES).astype(int)
        existing = (
            existing.sort_values(
                ["_key", "_resolved", "generated_at_utc"],
                ascending=[True, False, True],
            )
            .drop_duplicates("_key", keep="first")
            .drop(columns=["_resolved"])
        )
    if not incoming.empty:
        incoming["_key"] = _ledger_key(incoming)
        incoming = incoming.drop_duplicates("_key", keep="first")
        if not existing.empty:
            incoming = incoming.loc[~incoming["_key"].isin(set(existing["_key"]))]
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    combined = combined.drop(columns=["_key"], errors="ignore")
    combined = combined.reindex(columns=SHADOW_COLUMNS)
    combined = combined.reset_index(drop=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ledger_path, index=False)
    return combined


def write_goal_shadow_outputs(
    scored: pd.DataFrame,
    *,
    out_dir: Path,
    asof: dt.date,
    source_scored_file: str = "",
    root: Path | None = None,
    resolve_through_date: dt.date | None = None,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    shadow = build_goal_shadow_candidates(scored, asof=asof, source_scored_file=source_scored_file)
    run_path = out_dir / f"codexdaily_v4_goal_shadow_{asof}.csv"
    ledger_path = out_dir.parent / GOAL_SHADOW_LEDGER_NAME
    shadow.to_csv(run_path, index=False)
    ledger = update_goal_shadow_ledger(ledger_path, shadow)
    if root is not None and resolve_through_date is not None:
        ledger = resolve_goal_shadow_ledger(
            root=root,
            ledger_path=ledger_path,
            through_date=resolve_through_date,
        )
    summary = {
        "policy_id": GOAL_SHADOW_POLICY_ID,
        "shadow_only": True,
        "execution_eligible": False,
        "selected_rows": int(len(shadow)),
        "pending_ledger_rows": int(ledger["outcome_status"].astype(str).eq("PENDING").sum()) if not ledger.empty else 0,
        "resolved_ledger_rows": int(ledger["outcome_status"].astype(str).isin(RESOLVED_OUTCOME_STATES).sum()) if not ledger.empty else 0,
        "pending_count": int(ledger["outcome_status"].astype(str).eq("PENDING").sum()) if not ledger.empty else 0,
        "resolved_count": int(ledger["outcome_status"].astype(str).isin(RESOLVED_OUTCOME_STATES).sum()) if not ledger.empty else 0,
        "run_artifact": str(run_path),
        "central_ledger": str(ledger_path),
    }
    paths = {"goal_shadow": str(run_path), "goal_shadow_ledger": str(ledger_path)}
    try:
        from .daily_shadow_books import write_daily_shadow_outputs

        research_paths, research_summary = write_daily_shadow_outputs(
            scored,
            out_dir=out_dir,
            root=root or out_dir.parent.parent,
            asof=asof,
            source_scored_file=source_scored_file,
        )
        paths.update(research_paths)
        summary["research_shadow_books"] = research_summary
    except Exception as exc:
        error_path = out_dir / f"codexdaily_v4_research_shadow_error_{asof}.json"
        error_payload = {
            "status": "ERROR",
            "execution_impact": "none",
            "shadow_only": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        error_path.write_text(json.dumps(error_payload, indent=2) + "\n", encoding="utf-8")
        paths["research_shadow_error"] = str(error_path)
        summary["research_shadow_books"] = error_payload
    return shadow, paths, summary


def _simulate_locked_spread_exit(
    row: pd.Series,
    close_history: dict[dt.date, pd.DataFrame],
    quote_history: dict[dt.date, dict[str, dict[str, float | str]]],
    *,
    through_date: dt.date,
    slippage_pct: float,
    profit_take_pct: float,
    stop_loss_mult: float | None,
) -> dict[str, Any]:
    asof = row.get("asof")
    expiry = row.get("expiry")
    if not isinstance(asof, dt.date) or not isinstance(expiry, dt.date):
        return {"exact_evaluated": False, "exact_reason": "missing_asof_or_expiry"}
    entry_price = safe_float(row.get("entry_price"))
    width = safe_float(row.get("entry_width"))
    if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(width) or width <= 0:
        return {"exact_evaluated": False, "exact_reason": "invalid_locked_entry_economics"}

    kind = _strategy_kind(row)
    if kind == "Credit":
        target_value = safe_float(row.get("target_exit_value"), entry_price * (1.0 - profit_take_pct))
        stop_value = (
            safe_float(row.get("stop_exit_value"), entry_price * stop_loss_mult)
            if stop_loss_mult is not None
            else math.nan
        )
        risk = max(width - entry_price, 0.01)
    else:
        target_value = safe_float(row.get("target_exit_value"), min(width, entry_price * (1.0 + profit_take_pct)))
        stop_value = (
            safe_float(row.get("stop_exit_value"), entry_price / max(stop_loss_mult, 1.0))
            if stop_loss_mult is not None
            else math.nan
        )
        risk = max(entry_price, 0.01)

    quote_days_seen = 0
    for day in sorted(day for day in quote_history if asof < day <= min(expiry, through_date)):
        if kind == "Credit":
            mark = _spread_mid_debit(row, quote_history[day])
            exit_value = min(width, mark * (1.0 + slippage_pct)) if math.isfinite(mark) else math.nan
            target_hit = math.isfinite(exit_value) and exit_value <= target_value
            stop_hit = math.isfinite(exit_value) and math.isfinite(stop_value) and exit_value >= stop_value
        else:
            mark = _spread_mid_value(row, quote_history[day])
            exit_value = max(0.0, mark * (1.0 - slippage_pct)) if math.isfinite(mark) else math.nan
            target_hit = math.isfinite(exit_value) and exit_value >= target_value
            stop_hit = math.isfinite(exit_value) and math.isfinite(stop_value) and exit_value <= stop_value
        if not math.isfinite(exit_value):
            continue
        quote_days_seen += 1
        if not target_hit and not stop_hit:
            continue
        pnl_per_share = entry_price - exit_value if kind == "Credit" else exit_value - entry_price
        return {
            "exact_evaluated": True,
            "exit_day": day,
            "exit_reason": "profit_target" if target_hit else "stop_loss",
            "exit_value": exit_value,
            "target_exit_value": target_value,
            "stop_exit_value": stop_value,
            "pnl_1x": pnl_per_share * 100.0,
            "return_on_risk": pnl_per_share / risk,
            "exact_win": pnl_per_share > 0,
            "quote_days_seen": quote_days_seen,
        }

    if through_date >= expiry:
        eval_day, close = future_close(close_history, _clean(row.get("ticker")).upper(), expiry)
        if eval_day is not None and eval_day <= through_date and math.isfinite(close):
            exit_value = _expiry_spread_value(row, close)
            pnl_per_share = entry_price - exit_value if kind == "Credit" else exit_value - entry_price
            return {
                "exact_evaluated": True,
                "exit_day": eval_day,
                "exit_reason": "expiry_settlement",
                "exit_value": exit_value,
                "target_exit_value": target_value,
                "stop_exit_value": stop_value,
                "pnl_1x": pnl_per_share * 100.0,
                "return_on_risk": pnl_per_share / risk,
                "exact_win": pnl_per_share > 0,
                "quote_days_seen": quote_days_seen,
            }
    return {
        "exact_evaluated": False,
        "exact_reason": "awaiting_future_quote_or_expiry_close",
        "quote_days_seen": quote_days_seen,
    }


def resolve_goal_shadow_ledger(
    *,
    root: Path,
    ledger_path: Path,
    through_date: dt.date,
    slippage_pct: float = 0.10,
    profit_take_pct: float = 0.50,
    stop_loss_mult: float | None = None,
) -> pd.DataFrame:
    ledger = pd.read_csv(ledger_path, low_memory=False) if ledger_path.exists() else pd.DataFrame(columns=SHADOW_COLUMNS)
    if ledger.empty:
        return ledger
    for column in ["outcome_status", "outcome_last_checked", "exit_day", "exit_reason", "exact_win", "outcome_note"]:
        ledger[column] = ledger[column].astype(object)
    pending_asof = pd.to_datetime(
        ledger.loc[~ledger["outcome_status"].astype(str).isin(RESOLVED_OUTCOME_STATES), "asof"],
        errors="coerce",
    ).dropna()
    history_start = pending_asof.min().date() if not pending_asof.empty else through_date
    folders = dated_folders(root, history_start, through_date)
    close_history = load_close_history(folders)
    hot_history = load_hot_history(folders)
    quote_history = {day: _quote_lookup(hot) for day, hot in hot_history.items()}
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    for index, source in ledger.iterrows():
        if _clean(source.get("outcome_status")) in RESOLVED_OUTCOME_STATES:
            continue
        row = source.copy()
        row["asof"] = pd.to_datetime(source.get("asof"), errors="coerce")
        row["expiry"] = pd.to_datetime(source.get("expiry"), errors="coerce")
        if pd.isna(row["asof"]) or pd.isna(row["expiry"]):
            ledger.at[index, "outcome_last_checked"] = now
            ledger.at[index, "outcome_note"] = "invalid asof/expiry in shadow ledger"
            continue
        row["asof"] = row["asof"].date()
        row["expiry"] = row["expiry"].date()
        result = _simulate_locked_spread_exit(
            row,
            close_history,
            quote_history,
            through_date=through_date,
            slippage_pct=slippage_pct,
            profit_take_pct=profit_take_pct,
            stop_loss_mult=stop_loss_mult,
        )
        ledger.at[index, "outcome_last_checked"] = now
        if not _truthy(result.get("exact_evaluated")):
            ledger.at[index, "outcome_note"] = _clean(result.get("exact_reason")) or "future outcome not yet observable"
            continue
        pnl = safe_float(result.get("pnl_1x"), 0.0)
        ledger.at[index, "outcome_status"] = "RESOLVED_WIN" if pnl > 0 else "RESOLVED_LOSS" if pnl < 0 else "RESOLVED_FLAT"
        ledger.at[index, "exit_day"] = _clean(result.get("exit_day"))
        ledger.at[index, "exit_reason"] = _clean(result.get("exit_reason"))
        ledger.at[index, "exit_value"] = safe_float(result.get("exit_value"), safe_float(result.get("exit_debit")))
        ledger.at[index, "pnl_1x"] = pnl
        ledger.at[index, "return_on_risk"] = safe_float(result.get("return_on_risk"))
        ledger.at[index, "exact_win"] = pnl > 0
        ledger.at[index, "outcome_note"] = "resolved from point-in-time UW hot-chain/stock-close history"
    ledger = ledger.reindex(columns=SHADOW_COLUMNS)
    ledger.to_csv(ledger_path, index=False)
    return ledger


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit or resolve the fixed V4 goal shadow policy.")
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--scored", required=True)
    emit.add_argument("--out-dir", required=True)
    emit.add_argument("--as-of", required=True)
    emit.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    resolve = sub.add_parser("resolve")
    resolve.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    resolve.add_argument("--ledger", required=True)
    resolve.add_argument("--through-date", required=True)
    args = parser.parse_args(argv)
    if args.command == "emit":
        scored_path = Path(args.scored)
        shadow, paths, summary = write_goal_shadow_outputs(
            pd.read_csv(scored_path, low_memory=False),
            out_dir=Path(args.out_dir),
            asof=_parse_date(args.as_of),
            source_scored_file=str(scored_path),
            root=Path(args.root),
            resolve_through_date=_parse_date(args.as_of),
        )
        print(json.dumps({"summary": summary, "selected": shadow.to_dict(orient="records"), "artifacts": paths}, indent=2, default=str))
        return
    ledger = resolve_goal_shadow_ledger(
        root=Path(args.root),
        ledger_path=Path(args.ledger),
        through_date=_parse_date(args.through_date),
    )
    print(json.dumps({"rows": int(len(ledger)), "outcomes": ledger["outcome_status"].value_counts().to_dict() if not ledger.empty else {}}, indent=2))


if __name__ == "__main__":
    main()
