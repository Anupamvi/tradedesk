from __future__ import annotations

from typing import Any

import pandas as pd

from .payoff_calibration import PROBATIONARY_PAYOFF_STATUS
from .strategy_builder import generic_strategy_keys, historical_scope_for_strategy


STRATEGY_SPECS: tuple[dict[str, Any], ...] = (
    {"strategy_key": "long_call", "display_name": "Long Call", "category": "directional", "outlook": "bullish", "legs": 1, "risk_profile": "defined", "research_support": True, "live_builder": False},
    {"strategy_key": "long_put", "display_name": "Long Put", "category": "directional", "outlook": "bearish", "legs": 1, "risk_profile": "defined", "research_support": True, "live_builder": False},
    {"strategy_key": "covered_call", "display_name": "Covered Call", "category": "income", "outlook": "neutral_bullish", "legs": 2, "risk_profile": "stock_backed", "research_support": False, "live_builder": False},
    {"strategy_key": "cash_secured_put", "display_name": "Cash-Secured Put", "category": "income", "outlook": "neutral_bullish", "legs": 1, "risk_profile": "cash_secured", "research_support": False, "live_builder": False},
    {"strategy_key": "protective_put", "display_name": "Protective Put", "category": "hedge", "outlook": "bullish_with_floor", "legs": 2, "risk_profile": "stock_backed", "research_support": False, "live_builder": False},
    {"strategy_key": "collar", "display_name": "Collar", "category": "hedge", "outlook": "bullish_with_bounded_range", "legs": 3, "risk_profile": "stock_backed", "research_support": False, "live_builder": False},
    {"strategy_key": "bull_call_debit_vertical", "display_name": "Bull Call Debit Vertical", "category": "vertical", "outlook": "bullish", "legs": 2, "risk_profile": "defined", "research_support": True, "live_builder": True, "direction": "Bull Call", "confidence_family": "Debit"},
    {"strategy_key": "bear_put_debit_vertical", "display_name": "Bear Put Debit Vertical", "category": "vertical", "outlook": "bearish", "legs": 2, "risk_profile": "defined", "research_support": True, "live_builder": True, "direction": "Bear Put", "confidence_family": "Debit"},
    {"strategy_key": "bull_put_credit_vertical", "display_name": "Bull Put Credit Vertical", "category": "vertical", "outlook": "neutral_bullish", "legs": 2, "risk_profile": "defined", "research_support": True, "live_builder": True, "direction": "Bull Put", "confidence_family": "Credit"},
    {"strategy_key": "bear_call_credit_vertical", "display_name": "Bear Call Credit Vertical", "category": "vertical", "outlook": "neutral_bearish", "legs": 2, "risk_profile": "defined", "research_support": True, "live_builder": True, "direction": "Bear Call", "confidence_family": "Credit"},
    {"strategy_key": "long_straddle", "display_name": "Long Straddle", "category": "volatility", "outlook": "large_move", "legs": 2, "risk_profile": "defined", "research_support": True, "live_builder": False},
    {"strategy_key": "short_straddle", "display_name": "Short Straddle", "category": "volatility", "outlook": "range", "legs": 2, "risk_profile": "undefined", "research_support": False, "live_builder": False},
    {"strategy_key": "long_strangle", "display_name": "Long Strangle", "category": "volatility", "outlook": "large_move", "legs": 2, "risk_profile": "defined", "research_support": True, "live_builder": False},
    {"strategy_key": "short_strangle", "display_name": "Short Strangle", "category": "volatility", "outlook": "range", "legs": 2, "risk_profile": "undefined", "research_support": False, "live_builder": False},
    {"strategy_key": "iron_condor", "display_name": "Iron Condor", "category": "range", "outlook": "range", "legs": 4, "risk_profile": "defined", "research_support": True, "live_builder": False},
    {"strategy_key": "iron_butterfly", "display_name": "Iron Butterfly", "category": "range", "outlook": "range", "legs": 4, "risk_profile": "defined", "research_support": True, "live_builder": False},
    {"strategy_key": "call_butterfly", "display_name": "Call Butterfly", "category": "butterfly", "outlook": "targeted_bullish", "legs": 3, "risk_profile": "defined", "research_support": True, "live_builder": False},
    {"strategy_key": "put_butterfly", "display_name": "Put Butterfly", "category": "butterfly", "outlook": "targeted_bearish", "legs": 3, "risk_profile": "defined", "research_support": True, "live_builder": False},
    {"strategy_key": "call_broken_wing_butterfly", "display_name": "Call Broken-Wing Butterfly", "category": "butterfly", "outlook": "targeted_bullish", "legs": 3, "risk_profile": "defined", "research_support": False, "live_builder": False},
    {"strategy_key": "put_broken_wing_butterfly", "display_name": "Put Broken-Wing Butterfly", "category": "butterfly", "outlook": "targeted_bearish", "legs": 3, "risk_profile": "defined", "research_support": False, "live_builder": False},
    {"strategy_key": "call_calendar", "display_name": "Call Calendar", "category": "time_spread", "outlook": "targeted_bullish", "legs": 2, "risk_profile": "defined", "research_support": False, "live_builder": False},
    {"strategy_key": "put_calendar", "display_name": "Put Calendar", "category": "time_spread", "outlook": "targeted_bearish", "legs": 2, "risk_profile": "defined", "research_support": False, "live_builder": False},
    {"strategy_key": "call_diagonal", "display_name": "Call Diagonal", "category": "time_spread", "outlook": "bullish", "legs": 2, "risk_profile": "defined", "research_support": False, "live_builder": False},
    {"strategy_key": "put_diagonal", "display_name": "Put Diagonal", "category": "time_spread", "outlook": "bearish", "legs": 2, "risk_profile": "defined", "research_support": False, "live_builder": False},
    {"strategy_key": "call_ratio_spread", "display_name": "Call Ratio Spread", "category": "ratio", "outlook": "targeted_bullish", "legs": 2, "risk_profile": "potentially_undefined", "research_support": False, "live_builder": False},
    {"strategy_key": "put_ratio_spread", "display_name": "Put Ratio Spread", "category": "ratio", "outlook": "targeted_bearish", "legs": 2, "risk_profile": "potentially_undefined", "research_support": False, "live_builder": False},
    {"strategy_key": "call_ratio_backspread", "display_name": "Call Ratio Backspread", "category": "ratio", "outlook": "large_bullish_move", "legs": 2, "risk_profile": "defined_or_limited", "research_support": True, "live_builder": False},
    {"strategy_key": "put_ratio_backspread", "display_name": "Put Ratio Backspread", "category": "ratio", "outlook": "large_bearish_move", "legs": 2, "risk_profile": "defined_or_limited", "research_support": True, "live_builder": False},
    {"strategy_key": "jade_lizard", "display_name": "Jade Lizard", "category": "income", "outlook": "neutral_bullish", "legs": 3, "risk_profile": "one_sided_undefined", "research_support": False, "live_builder": False},
    {"strategy_key": "reverse_jade_lizard", "display_name": "Reverse Jade Lizard", "category": "income", "outlook": "neutral_bearish", "legs": 3, "risk_profile": "one_sided_undefined", "research_support": False, "live_builder": False},
    {"strategy_key": "covered_strangle", "display_name": "Covered Strangle", "category": "income", "outlook": "range_bullish", "legs": 3, "risk_profile": "stock_and_cash_backed", "research_support": False, "live_builder": False},
    {"strategy_key": "wheel", "display_name": "Wheel", "category": "income_lifecycle", "outlook": "neutral_bullish", "legs": 1, "risk_profile": "cash_or_stock_backed", "research_support": False, "live_builder": False},
)

STRATEGY_SPECS = tuple(
    {
        **spec,
        "specialized_execution_builder": bool(spec.get("live_builder")),
        "live_builder": True,
        "research_support": historical_scope_for_strategy(str(spec["strategy_key"])) != "unavailable",
        "historical_scope": historical_scope_for_strategy(str(spec["strategy_key"])),
    }
    for spec in STRATEGY_SPECS
)


DIRECTION_TO_KEY = {
    "Bull Call": "bull_call_debit_vertical",
    "Bear Put": "bear_put_debit_vertical",
    "Bull Put": "bull_put_credit_vertical",
    "Bear Call": "bear_call_credit_vertical",
}


def _payoff_status(groups: pd.DataFrame, direction: str) -> str:
    if groups is None or groups.empty or "direction" not in groups.columns:
        return "INSUFFICIENT"
    rows = groups[groups["direction"].astype(str).eq(direction)]
    statuses = set(rows.get("payoff_calibration_status", pd.Series(dtype=str)).astype(str).str.upper())
    if "PASS" in statuses:
        return "PASS"
    if PROBATIONARY_PAYOFF_STATUS in statuses:
        return PROBATIONARY_PAYOFF_STATUS
    if "VETO" in statuses:
        return "VETO"
    return "INSUFFICIENT"


def build_strategy_registry(
    *,
    payoff_summary: dict[str, Any] | None,
    payoff_groups: pd.DataFrame,
    confidence_summary: dict[str, Any] | None,
    strategy_validation: pd.DataFrame | None = None,
) -> pd.DataFrame:
    payoff_summary = payoff_summary or {}
    confidence_summary = confidence_summary or {}
    family_validation = confidence_summary.get("family_validation", {}) or {}
    test_bypass = str(payoff_summary.get("status", "")).upper() == "TEST_BYPASS"
    generic_live = set(generic_strategy_keys())
    validation = strategy_validation if strategy_validation is not None else pd.DataFrame()
    validation_lookup: dict[str, dict[str, Any]] = {}
    if not validation.empty and "strategy" in validation.columns:
        for strategy, part in validation.groupby(validation["strategy"].astype(str)):
            release = part.get("release_status", pd.Series("REJECTED", index=part.index)).astype(str).str.upper()
            validated = part[release.eq("VALIDATED")]
            best = (
                validated.sort_values(
                    ["holm_adjusted_joint_p", "clustered_pf_p05"],
                    ascending=[True, False],
                ).iloc[0]
                if not validated.empty
                else part.sort_values(
                    ["holm_adjusted_joint_p", "clustered_pf_p05"],
                    ascending=[True, False],
                    na_position="last",
                ).iloc[0]
            )
            validation_lookup[strategy] = {
                "status": "VALIDATED" if not validated.empty else "REJECTED",
                "scope": str(best.get("scope") or ""),
                "scope_value": str(best.get("scope_value") or ""),
                "clustered_pf_p05": best.get("clustered_pf_p05"),
                "holm_adjusted_joint_p": best.get("holm_adjusted_joint_p"),
            }
    rows: list[dict[str, Any]] = []
    for spec in STRATEGY_SPECS:
        row = dict(spec)
        row["live_builder"] = bool(spec.get("live_builder") or spec.get("strategy_key") in generic_live)
        row["research_support"] = historical_scope_for_strategy(str(spec.get("strategy_key"))) != "unavailable"
        row["historical_scope"] = historical_scope_for_strategy(str(spec.get("strategy_key")))
        validation_result = validation_lookup.get(str(spec.get("strategy_key")), {})
        row["strategy_validation_status"] = validation_result.get("status", "NOT_EVALUATED")
        row["strategy_validation_scope"] = validation_result.get("scope", "")
        row["strategy_validation_scope_value"] = validation_result.get("scope_value", "")
        row["strategy_validation_clustered_pf_p05"] = validation_result.get("clustered_pf_p05")
        row["strategy_validation_holm_p"] = validation_result.get("holm_adjusted_joint_p")
        direction = str(spec.get("direction", ""))
        confidence_family = str(spec.get("confidence_family", ""))
        payoff_status = _payoff_status(payoff_groups, direction) if direction else "NOT_APPLICABLE"
        confidence_status = (
            str((family_validation.get(confidence_family, {}) or {}).get("status", "INSUFFICIENT")).upper()
            if confidence_family
            else "NOT_APPLICABLE"
        )
        authorized = bool(
            spec.get("specialized_execution_builder")
            and ((payoff_status == "PASS" and confidence_status == "PASS") or test_bypass)
        )
        probationary_candidate = bool(
            spec.get("specialized_execution_builder")
            and payoff_status == PROBATIONARY_PAYOFF_STATUS
            and confidence_family == "Credit"
            and confidence_status in {"PASS", "CONSERVATIVE"}
        )
        probationary_authorized = probationary_candidate
        if authorized:
            pipeline_status = "PRODUCTION"
            reason = "live builder, maturity-safe payoff evidence, and calibrated confidence pass"
        elif probationary_candidate:
            pipeline_status = "PROBATIONARY"
            reason = "one-contract pilot only; payoff route passes stress/OOS gates and awaits post-activation outcomes before scaling"
        elif spec.get("specialized_execution_builder"):
            pipeline_status = "PROSPECTIVE"
            reason = f"live builder exists; payoff={payoff_status}; confidence={confidence_status}"
        elif row["strategy_validation_status"] == "VALIDATED":
            pipeline_status = "PROSPECTIVE"
            reason = (
                f"live and historical model={row['historical_scope']} pass search-wide validation; "
                "strategy-specific confidence and post-activation calibration are still absent"
            )
        elif row.get("live_builder") and row.get("research_support"):
            pipeline_status = "RESEARCH_ONLY"
            reason = (
                f"live builder and historical model={row['historical_scope']} exist; "
                f"search-wide release validation={row['strategy_validation_status']}"
            )
        elif row.get("live_builder"):
            pipeline_status = "UNTESTED_DATA_GAP"
            reason = "live builder exists; point-in-time historical construction and release validation are absent"
        elif spec.get("research_support"):
            pipeline_status = "RESEARCH_ONLY"
            reason = "historical construction exists; search-wide release authority and/or live construction is absent"
        else:
            pipeline_status = "UNTESTED_DATA_GAP"
            reason = "no point-in-time historical and live construction parity yet"
        row.update(
            {
                "payoff_evidence_status": payoff_status,
                "confidence_evidence_status": confidence_status,
                "pipeline_status": pipeline_status,
                "execution_authorized": authorized,
                "probationary_execution_authorized": probationary_authorized,
                "status_reason": reason,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def strategy_key_for_row(row: pd.Series | dict[str, Any]) -> str:
    explicit = str(row.get("strategy_registry_key") or row.get("strategy_key") or "").strip()
    if explicit:
        return explicit
    direction = str(row.get("direction", "")).strip()
    if direction in DIRECTION_TO_KEY:
        return DIRECTION_TO_KEY[direction]
    strategy = str(row.get("strategy", "")).strip().lower()
    for spec in STRATEGY_SPECS:
        key = str(spec["strategy_key"])
        if key.replace("_", " ") in strategy:
            return key
    return "unregistered"


def apply_strategy_registry_gate(scored: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else pd.DataFrame()
    out = scored.copy()
    lookup = registry.set_index("strategy_key").to_dict(orient="index") if not registry.empty else {}
    for index, row in out.iterrows():
        key = strategy_key_for_row(row)
        record = lookup.get(key, {})
        production_authorized = bool(record.get("execution_authorized", False))
        probationary_authorized = bool(record.get("probationary_execution_authorized", False))
        pilot_row = "pilot" in str(row.get("trade_tier", "")).lower()
        medium_debit_authorized = str(row.get("v4_execution_authority", "")) == "validated_medium_debit_one_lot"
        symbol_credit_authorized = str(row.get("v4_execution_authority", "")) == "symbol_regime_credit_one_lot"
        walk_forward_credit_authorized = str(row.get("v4_execution_authority", "")) == "walk_forward_credit_one_lot"
        authorized = (
            production_authorized
            or (probationary_authorized and pilot_row)
            or medium_debit_authorized
            or symbol_credit_authorized
            or walk_forward_credit_authorized
        )
        out.at[index, "strategy_registry_key"] = key
        out.at[index, "strategy_registry_status"] = record.get("pipeline_status", "UNREGISTERED")
        out.at[index, "strategy_historical_scope"] = record.get("historical_scope", "unavailable")
        out.at[index, "strategy_validation_status"] = record.get("strategy_validation_status", "NOT_EVALUATED")
        out.at[index, "strategy_validation_scope"] = record.get("strategy_validation_scope", "")
        out.at[index, "strategy_validation_scope_value"] = record.get("strategy_validation_scope_value", "")
        out.at[index, "strategy_validation_clustered_pf_p05"] = record.get("strategy_validation_clustered_pf_p05")
        out.at[index, "strategy_validation_holm_p"] = record.get("strategy_validation_holm_p")
        out.at[index, "strategy_execution_authorized"] = authorized
        out.at[index, "strategy_execution_authority"] = (
            "production"
            if production_authorized
            else "symbol_regime_credit_one_lot"
            if symbol_credit_authorized
            else "walk_forward_credit_one_lot"
            if walk_forward_credit_authorized
            else "validated_medium_debit_one_lot"
            if medium_debit_authorized
            else "probationary_one_lot"
            if authorized
            else "none"
        )
        out.at[index, "strategy_registry_reason"] = record.get("status_reason", "strategy is not registered")
        if str(row.get("trade_status", "")) == "Execute" and not authorized:
            out.at[index, "trade_status"] = "Research"
            out.at[index, "trade_tier"] = "strategy-registry-blocked"
            out.at[index, "trade_status_reason"] = f"strategy registry blocked execution: {key}"
            out.at[index, "primary_blocker"] = f"strategy_not_production:{key}"
            out.at[index, "decision_eligible"] = False
        elif str(row.get("trade_status", "")) == "Execute" and probationary_authorized and pilot_row:
            out.at[index, "contracts"] = 1
            out.at[index, "strategy_registry_reason"] = (
                "one-contract probationary execution; post-activation outcomes required before scaling"
            )
        elif str(row.get("trade_status", "")) == "Execute" and medium_debit_authorized:
            out.at[index, "contracts"] = 1
            out.at[index, "strategy_registry_reason"] = (
                "one-contract validated medium-debit sleeve; route-specific replay evidence controls authority"
            )
        elif str(row.get("trade_status", "")) == "Execute" and symbol_credit_authorized:
            out.at[index, "contracts"] = 1
            out.at[index, "strategy_registry_reason"] = (
                "one-contract symbol-trend credit pilot; post-activation outcomes required before scaling"
            )
        elif walk_forward_credit_authorized:
            out.at[index, "contracts"] = 1
            out.at[index, "strategy_registry_reason"] = (
                "one-contract maturity-safe walk-forward credit authority; High confidence unavailable"
            )
    return out
