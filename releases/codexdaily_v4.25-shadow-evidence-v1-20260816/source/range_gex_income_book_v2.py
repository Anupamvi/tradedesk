from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import range_gex_income_book as legacy


POLICY_VERSION = "range-gex-income-shadow-v2-point-in-time-joint-payoff-20260816"
RANGE_LEDGER_NAME = "codexdaily_v4_range_gex_shadow_ledger.csv"
RANGE_LEDGER_COLUMNS = [
    "policy_version", "signal_date", "generated_at_utc", "structure", "ticker",
    "expiry", "legs", "entry_value", "max_loss_1x", "shadow_only",
    "execution_authorized", "outcome_status", "outcome_note",
]


def _series(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    return frame[name] if name in frame.columns else pd.Series(default, index=frame.index)


def _num(frame: pd.DataFrame, name: str, default: Any = np.nan) -> pd.Series:
    return pd.to_numeric(_series(frame, name, default), errors="coerce")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def derive_strict_gex_features(summary: pd.DataFrame, strikes: pd.DataFrame) -> pd.DataFrame:
    """Require the requested UW date and capture market date to match the signal date."""
    base = legacy.derive_gex_features(summary, strikes)
    if base.empty:
        return base
    timing = summary.copy()
    timing["asof"] = pd.to_datetime(timing["date"], errors="coerce").dt.normalize()
    timing["ticker"] = timing["ticker"].astype(str).str.upper()
    captured = pd.to_datetime(timing.get("captured_utc"), errors="coerce", utc=True)
    uw_time = pd.to_datetime(timing.get("uw_time"), errors="coerce", utc=True)
    timing["gex_capture_market_date"] = captured.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    timing["gex_uw_date"] = uw_time.dt.tz_localize(None).dt.normalize()
    timing["gex_source_point_in_time"] = (
        timing["gex_capture_market_date"].eq(timing["asof"])
        & timing["gex_uw_date"].eq(timing["asof"])
    )
    timing = timing.sort_values("captured_utc", na_position="first").drop_duplicates(["asof", "ticker"], keep="last")
    keep = ["asof", "ticker", "gex_capture_market_date", "gex_uw_date", "gex_source_point_in_time"]
    out = base.drop(columns=["gex_capture_timing"], errors="ignore").merge(
        timing[keep], on=["asof", "ticker"], how="left", validate="one_to_one"
    )
    out["gex_source_point_in_time"] = out["gex_source_point_in_time"].fillna(False).astype(bool)
    out["gex_capture_timing"] = np.where(out["gex_source_point_in_time"], "point_in_time", "historical_api_reconstruction")
    out["gex_wall_span_pct"] = (out["gex_call_wall"] - out["gex_put_wall"]) / out["gex_spot"].replace(0, np.nan)
    out["gex_net_magnitude_per_spot"] = out["gamma_oi_per_1pct"].abs() / out["gex_spot"].replace(0, np.nan)
    return out.sort_values(["asof", "ticker"]).reset_index(drop=True)


def load_strict_historical_gex(root: Path) -> pd.DataFrame:
    summaries = [pd.read_csv(path, low_memory=False) for path in sorted(root.glob("20??-??-??/enrichments/uw/uw_gex_summary_*.csv"))]
    strikes = [pd.read_csv(path, low_memory=False) for path in sorted(root.glob("20??-??-??/enrichments/uw/uw_gex_strikes_*.csv"))]
    if not summaries or not strikes:
        return pd.DataFrame()
    return derive_strict_gex_features(pd.concat(summaries, ignore_index=True), pd.concat(strikes, ignore_index=True))


def load_gex_for_date(root: Path, asof: dt.date) -> pd.DataFrame:
    folder = root / str(asof) / "enrichments" / "uw"
    summary_path = folder / f"uw_gex_summary_{asof}.csv"
    strikes_path = folder / f"uw_gex_strikes_{asof}.csv"
    if not summary_path.exists() or not strikes_path.exists():
        return pd.DataFrame()
    features = derive_strict_gex_features(
        pd.read_csv(summary_path, low_memory=False),
        pd.read_csv(strikes_path, low_memory=False),
    )
    return features[features["gex_source_point_in_time"]].copy()


def evaluate_live_verticals(scored: pd.DataFrame, gex: pd.DataFrame, *, asof: dt.date) -> pd.DataFrame:
    source = scored[_series(scored, "strategy", "").astype(str).isin({"Bull Put Credit Spread", "Bear Call Credit Spread"})].copy()
    if source.empty:
        return source
    source["asof"] = pd.Timestamp(asof)
    source["ticker"] = source["ticker"].astype(str).str.upper()
    source["entry_credit"] = _num(source, "natural_credit")
    source["entry_width"] = _num(source, "spread_width")
    source["entry_credit_pct_width"] = source["entry_credit"] / source["entry_width"].replace(0, np.nan)
    source["entry_quote_width_pct"] = _num(source, "quote_width_pct")
    source["entry_dte"] = _num(source, "dte")
    source["short_contract"] = _series(source, "short_leg", "")
    source["long_contract"] = _series(source, "long_leg", "")
    source["short_strike_live"] = _num(source, "short_strike")
    source["long_strike_live"] = _num(source, "long_strike")
    source["symbol_regime"] = _series(source, "symbol_regime_trend", "").fillna("").astype(str)
    source["symbol_regime"] = source["symbol_regime"].where(source["symbol_regime"].str.len().gt(0), _series(source, "regime_trend", "range").astype(str))
    merged = source.merge(gex, on=["asof", "ticker"], how="inner", validate="many_to_one")
    if merged.empty:
        return merged
    expiry = pd.to_datetime(merged["expiry"], errors="coerce")
    earnings = pd.to_datetime(_series(merged, "next_earnings_dt", ""), errors="coerce")
    merged["earnings_known"] = earnings.notna()
    merged["earnings_crosses"] = earnings.le(expiry)
    merged["flow_not_contra"] = (
        merged["strategy"].eq("Bull Put Credit Spread") & _num(merged, "combined_flow_bias").ge(-0.05)
    ) | (
        merged["strategy"].eq("Bear Call Credit Spread") & _num(merged, "combined_flow_bias").le(0.05)
    )
    merged["short_outside_wall"] = (
        merged["strategy"].eq("Bull Put Credit Spread") & merged["short_strike_live"].le(merged["gex_put_wall"])
    ) | (
        merged["strategy"].eq("Bear Call Credit Spread") & merged["short_strike_live"].ge(merged["gex_call_wall"])
    )
    merged["gex_spot_between_walls"] = merged["gex_put_wall"].lt(merged["gex_spot"]) & merged["gex_spot"].lt(merged["gex_call_wall"])
    checks = {
        "live_price": _series(merged, "live_status", "").astype(str).eq("PASS") & _series(merged, "regular_session_quote", True).map(_truthy),
        "defined_risk": merged["entry_credit"].gt(0) & merged["entry_width"].gt(merged["entry_credit"]),
        "credit_quality": merged["entry_credit_pct_width"].between(0.15, 0.45) & merged["entry_quote_width_pct"].le(0.35),
        "dte": merged["entry_dte"].between(21, 44),
        "expected_move": _num(merged, "expected_move_ratio").le(0.90),
        "iv_over_hv": _num(merged, "iv_hv_ratio").between(1.05, 2.50),
        "earnings": merged["earnings_known"] & ~merged["earnings_crosses"],
        "range_regime": merged["symbol_regime"].eq("range"),
        "positive_gamma": merged["gamma_oi_per_1pct"].gt(0),
        "between_walls": merged["gex_spot_between_walls"],
        "wall_span": merged["gex_wall_span_pct"].ge(0.02),
        "wall_concentration": merged["gex_wall_concentration"].ge(0.15),
        "short_outside_wall": merged["short_outside_wall"],
        "flow_not_contra": merged["flow_not_contra"],
        "source_timing": merged["gex_source_point_in_time"],
    }
    merged["range_gex_qualified"] = True
    for passed in checks.values():
        merged["range_gex_qualified"] &= passed.fillna(False)
    reasons: list[str] = []
    for index in merged.index:
        failed = [name for name, passed in checks.items() if not bool(passed.loc[index])]
        reasons.append(";".join(failed))
    merged["range_gex_blockers"] = reasons
    merged["range_gex_rank"] = (
        2.0 * merged["entry_credit_pct_width"]
        - 0.8 * _num(merged, "expected_move_ratio")
        - 0.5 * merged["entry_quote_width_pct"]
        + 0.5 * merged["gex_wall_concentration"]
        + 0.15 * merged["gex_wall_span_pct"].clip(0, 0.25)
    )
    merged["policy_version"] = POLICY_VERSION
    merged["shadow_only"] = True
    merged["execution_authorized"] = False
    merged["no_order_placement"] = True
    return merged.sort_values(["range_gex_qualified", "range_gex_rank"], ascending=[False, False]).reset_index(drop=True)


def select_live_range_book(evaluated: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    qualified = evaluated[evaluated.get("range_gex_qualified", False)].copy()
    if qualified.empty:
        return qualified, qualified
    vertical = qualified.sort_values("range_gex_rank", ascending=False).head(1).copy()
    components = qualified.sort_values("range_gex_rank", ascending=False).drop_duplicates(["ticker", "expiry", "strategy"])
    puts = components[components["strategy"].eq("Bull Put Credit Spread")]
    calls = components[components["strategy"].eq("Bear Call Credit Spread")]
    pairs = puts.merge(calls, on=["asof", "ticker", "expiry"], suffixes=("_put", "_call"), how="inner")
    if pairs.empty:
        return vertical, pairs
    pairs["total_credit"] = pairs["entry_credit_put"] + pairs["entry_credit_call"]
    pairs["max_wing_width"] = pairs[["entry_width_put", "entry_width_call"]].max(axis=1)
    pairs["max_loss_1x"] = (pairs["max_wing_width"] - pairs["total_credit"]) * 100.0
    pairs["credit_pct_width"] = pairs["total_credit"] / pairs["max_wing_width"]
    pairs = pairs[pairs["credit_pct_width"].between(0.25, 0.70) & pairs["max_loss_1x"].gt(0)].copy()
    pairs["range_gex_rank"] = pairs["range_gex_rank_put"] + pairs["range_gex_rank_call"]
    pairs["policy_version"] = POLICY_VERSION
    pairs["shadow_only"] = True
    pairs["execution_authorized"] = False
    pairs["no_order_placement"] = True
    return vertical, pairs.sort_values("range_gex_rank", ascending=False).head(1).copy()


def _quote_mid(lookup: dict[str, dict[str, Any]], contract: Any) -> float:
    key = str(contract or "").strip()
    row = lookup.get(key) or lookup.get(key.replace(" ", "")) or {}
    try:
        return float(row.get("mid"))
    except (TypeError, ValueError):
        return math.nan


def simulate_joint_condor_exit(
    row: pd.Series,
    *,
    quote_history: dict[dt.date, dict[str, dict[str, Any]]],
    close_history: dict[dt.date, pd.DataFrame],
    through_date: dt.date,
    profit_take_pct: float = 0.50,
    stop_loss_mult: float = 2.0,
) -> dict[str, Any]:
    from . import goal_shadow

    entry_day = pd.to_datetime(row.get("entry_day"), errors="coerce")
    expiry = pd.to_datetime(row.get("expiry"), errors="coerce")
    credit = float(row.get("total_credit", math.nan))
    width = float(row.get("max_wing_width", math.nan))
    if pd.isna(entry_day) or pd.isna(expiry) or not (credit > 0 and width > credit):
        return {"exact_evaluated": False, "exact_reason": "invalid_joint_condor_entry"}
    entry_date, expiry_date = entry_day.date(), expiry.date()
    contracts = [row.get("short_contract_put"), row.get("long_contract_put"), row.get("short_contract_call"), row.get("long_contract_call")]
    target, stop = credit * (1.0 - profit_take_pct), min(width, credit * stop_loss_mult)
    for day in sorted(day for day in quote_history if entry_date < day <= min(expiry_date, through_date)):
        mids = [_quote_mid(quote_history[day], contract) for contract in contracts]
        if not all(math.isfinite(value) for value in mids):
            continue
        debit = min(width, max(0.0, mids[0] - mids[1] + mids[2] - mids[3]))
        if debit > target and debit < stop:
            continue
        pnl = (credit - debit) * 100.0
        return {"exact_evaluated": True, "exit_day": day, "exit_reason": "profit_target" if debit <= target else "stop_loss", "exit_value": debit, "pnl_1x": pnl, "return_on_risk": pnl / ((width - credit) * 100.0)}
    if through_date >= expiry_date:
        eval_day, close = goal_shadow.future_close(close_history, str(row.get("ticker", "")).upper(), expiry_date)
        if eval_day is not None and math.isfinite(close):
            sp, lp = float(row["short_strike_live_put"]), float(row["long_strike_live_put"])
            sc, lc = float(row["short_strike_live_call"]), float(row["long_strike_live_call"])
            debit = min(width, max(0.0, max(sp - close, 0.0) - max(lp - close, 0.0) + max(close - sc, 0.0) - max(close - lc, 0.0)))
            pnl = (credit - debit) * 100.0
            return {"exact_evaluated": True, "exit_day": eval_day, "exit_reason": "expiry_settlement", "exit_value": debit, "pnl_1x": pnl, "return_on_risk": pnl / ((width - credit) * 100.0)}
    return {"exact_evaluated": False, "exact_reason": "awaiting_joint_four_leg_quotes_or_expiry"}


def _range_ledger_rows(vertical: pd.DataFrame, condor: pd.DataFrame, *, asof: dt.date) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    for _, row in vertical.iterrows():
        rows.append({"policy_version": POLICY_VERSION, "signal_date": str(asof), "generated_at_utc": now, "structure": row["strategy"], "ticker": row["ticker"], "expiry": row["expiry"], "legs": f"SELL {row['short_contract']} | BUY {row['long_contract']}", "entry_value": row["entry_credit"], "max_loss_1x": (row["entry_width"] - row["entry_credit"]) * 100.0, "shadow_only": True, "execution_authorized": False, "outcome_status": "PENDING", "outcome_note": "point-in-time GEX shadow; not an order"})
    for _, row in condor.iterrows():
        rows.append({"policy_version": POLICY_VERSION, "signal_date": str(asof), "generated_at_utc": now, "structure": "Iron Condor", "ticker": row["ticker"], "expiry": row["expiry"], "legs": f"SELL {row['short_contract_put']} | BUY {row['long_contract_put']} | SELL {row['short_contract_call']} | BUY {row['long_contract_call']}", "entry_value": row["total_credit"], "max_loss_1x": row["max_loss_1x"], "shadow_only": True, "execution_authorized": False, "outcome_status": "PENDING", "outcome_note": "joint four-leg outcome required; not an order"})
    return pd.DataFrame(rows).reindex(columns=RANGE_LEDGER_COLUMNS)


def _update_range_ledger(path: Path, incoming: pd.DataFrame) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path, low_memory=False).reindex(columns=RANGE_LEDGER_COLUMNS)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=RANGE_LEDGER_COLUMNS)
    else:
        existing = pd.DataFrame(columns=RANGE_LEDGER_COLUMNS)
    incoming = incoming.reindex(columns=RANGE_LEDGER_COLUMNS)
    if existing.empty:
        combined = incoming.copy()
    elif incoming.empty:
        combined = existing.copy()
    else:
        combined = pd.concat([existing, incoming], ignore_index=True)
    if not combined.empty:
        key = ["policy_version", "signal_date", "structure", "ticker", "expiry", "legs"]
        combined = combined.sort_values("generated_at_utc").drop_duplicates(key, keep="first")
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined


def write_prospective_outputs(scored: pd.DataFrame, *, out_dir: Path, root: Path, asof: dt.date) -> tuple[dict[str, str], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated_path = out_dir / f"codexdaily_v4_range_gex_shadow_candidates_{asof}.csv"
    vertical_path = out_dir / f"codexdaily_v4_range_gex_shadow_vertical_{asof}.csv"
    condor_path = out_dir / f"codexdaily_v4_range_gex_shadow_condor_{asof}.csv"
    status_path = out_dir / f"codexdaily_v4_range_gex_shadow_status_{asof}.json"
    ledger_path = out_dir.parent / RANGE_LEDGER_NAME
    gex = load_gex_for_date(root, asof)
    if gex.empty:
        evaluated, vertical, condor = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        status = "MISSING_POINT_IN_TIME_GEX"
        reason = "same-session UW summary and strike-wall capture was not available; reconstructed GEX is prohibited"
    else:
        evaluated = evaluate_live_verticals(scored, gex, asof=asof)
        vertical, condor = select_live_range_book(evaluated)
        status = "SHADOW_ACTIVE"
        reason = "point-in-time GEX candidates evaluated; execution authority intentionally disabled"
    evaluated.to_csv(evaluated_path, index=False)
    vertical.to_csv(vertical_path, index=False)
    condor.to_csv(condor_path, index=False)
    ledger = _update_range_ledger(ledger_path, _range_ledger_rows(vertical, condor, asof=asof))
    summary = {"policy_version": POLICY_VERSION, "status": status, "reason": reason, "shadow_only": True, "execution_authorized": False, "gex_rows": int(len(gex)), "evaluated_rows": int(len(evaluated)), "qualified_rows": int(evaluated.get("range_gex_qualified", pd.Series(dtype=bool)).sum()) if not evaluated.empty else 0, "selected_vertical_rows": int(len(vertical)), "selected_condor_rows": int(len(condor)), "ledger_rows": int(len(ledger))}
    status_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return {"range_gex_shadow_candidates": str(evaluated_path), "range_gex_shadow_vertical": str(vertical_path), "range_gex_shadow_condor": str(condor_path), "range_gex_shadow_status": str(status_path), "range_gex_shadow_ledger": str(ledger_path)}, summary


def run_historical_research(*, root: Path, replay_path: Path, out_dir: Path, cutoff: pd.Timestamp) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    gex_all = load_strict_historical_gex(root)
    gex = gex_all[gex_all["gex_source_point_in_time"]].copy() if not gex_all.empty else gex_all
    replay = pd.read_csv(replay_path, low_memory=False)
    enriched = legacy.enrich_replay(replay, gex) if not gex.empty else pd.DataFrame()
    vertical_qualified, vertical_selected = legacy.build_vertical_shadow(enriched) if not enriched.empty else (pd.DataFrame(), pd.DataFrame())
    if not vertical_qualified.empty:
        vertical_qualified = vertical_qualified[pd.to_numeric(vertical_qualified["iv_hv_ratio"], errors="coerce").between(1.05, 2.50)].copy()
        vertical_selected = vertical_qualified.sort_values(["asof", "range_gex_rank"], ascending=[True, False]).groupby("asof", as_index=False).head(1)
    pairs, _, _ = legacy.build_condor_shadow(enriched) if not enriched.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    condor_selected = pd.DataFrame()
    if not pairs.empty:
        policy = pairs["gex_capture_timing"].eq("point_in_time") & pairs["iv_hv"].between(1.05, 2.50) & pairs["gamma_oi_per_1pct"].gt(0) & pairs["gex_spot_between_walls"] & pairs["shorts_outside_walls"] & pairs["credit_pct_width"].between(0.25, 0.70)
        condor_selected = pairs[policy].sort_values(["asof", "credit_pct_width"], ascending=[True, False]).groupby("asof", as_index=False).head(1).copy()
        if not condor_selected.empty:
            from . import goal_shadow
            start = pd.to_datetime(condor_selected["entry_day"]).min().date()
            through = max(path.name for path in root.glob("20??-??-??") if path.is_dir())
            through_date = dt.date.fromisoformat(through)
            folders = goal_shadow.dated_folders(root, start, through_date)
            close_history = goal_shadow.load_close_history(folders)
            hot_history = goal_shadow.load_hot_history(folders)
            quote_history = {day: goal_shadow._quote_lookup(frame) for day, frame in hot_history.items()}
            outcomes = [simulate_joint_condor_exit(row, quote_history=quote_history, close_history=close_history, through_date=through_date) for _, row in condor_selected.iterrows()]
            condor_selected["exact_evaluated"] = [bool(item.get("exact_evaluated")) for item in outcomes]
            condor_selected["pnl_1x"] = [item.get("pnl_1x", np.nan) for item in outcomes]
            condor_selected["exit_day"] = [item.get("exit_day", "") for item in outcomes]
            condor_selected["exit_reason"] = [item.get("exit_reason", item.get("exact_reason", "")) for item in outcomes]
            condor_selected = condor_selected[condor_selected["exact_evaluated"]].copy()
    vertical_summary, vertical_metrics, vertical_monthly = legacy.evaluate_shadow_book(vertical_selected, cutoff=cutoff, credit_column="entry_credit")
    condor_summary, condor_metrics, condor_monthly = legacy.evaluate_shadow_book(condor_selected, cutoff=cutoff, credit_column="total_credit")
    for summary in (vertical_summary, condor_summary):
        summary["policy_version"] = POLICY_VERSION
        summary["execution_authorized"] = False
    overall = {"policy_version": POLICY_VERSION, "status": "RESEARCH_ONLY", "execution_authorized": False, "reason": "strict point-in-time and independent promotion gates remain mandatory", "cutoff": str(cutoff.date()), "all_gex_rows": int(len(gex_all)), "point_in_time_gex_rows": int(len(gex)), "point_in_time_gex_dates": int(gex["asof"].nunique()) if not gex.empty else 0, "joined_replay_rows": int(len(enriched)), "vertical": vertical_summary, "condor": condor_summary}
    artifacts = {"strict_gex_features": gex_all, "range_gex_vertical_qualified": vertical_qualified, "range_gex_vertical_selected": vertical_selected, "range_gex_vertical_metrics": vertical_metrics, "range_gex_vertical_monthly": vertical_monthly, "range_gex_condor_selected_joint": condor_selected, "range_gex_condor_metrics": condor_metrics, "range_gex_condor_monthly": condor_monthly}
    for name, frame in artifacts.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "range_gex_v2_validation.json").write_text(json.dumps(overall, indent=2, default=str) + "\n", encoding="utf-8")
    return overall
