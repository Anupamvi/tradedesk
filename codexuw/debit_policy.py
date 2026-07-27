from __future__ import annotations

import math
from typing import Any

from .data import safe_float


DEBIT_POLICY_VERSION = "debit-v3.0-bull-call-only"


# Bear Put was removed after the regenerated replay base showed it losing in
# every out-of-sample fold and under every macro conditioning tried:
#
#   Bear Put, any condition ... n=303  PF 0.855  -$3,500  OOS 0/4 folds
#   Bear Put + low market IV ... n=165  PF 1.002    +$29  OOS 1/4 folds
#
# There is no slice of it that pays. An unknown direction falls through to
# "unsupported_debit_direction" below, so removing the key disables the family.
DEBIT_POLICY = {
    "Bull Call": {
        "allowed_regimes": {"uptrend"},
        "allowed_dte_ranges": ((7, 10), (22, 45)),
        "max_debit_pct_width": 0.45,
        "min_reward_risk": 1.25,
        "min_expected_move_ratio": 1.00,
        "min_flow_alignment": 0.20,
        "max_quote_width_pct": 0.35,
        "max_iv_rank": 55.0,
    },
}


def _clean(value: object) -> str:
    return str(value or "").strip()


def _direction_sign(direction: str) -> int:
    if direction == "Bull Call":
        return 1
    if direction == "Bear Put":
        return -1
    return 0


def _has_full_bot_flow(value: object) -> bool:
    return _clean(value).lower() in {"bot_eod_loaded", "bot_eod_split_bundle_loaded"}


def _flow_alignment(row: dict[str, Any] | Any) -> float:
    bias = safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), math.nan))
    sign = _direction_sign(_clean(row.get("direction")))
    return bias * sign if math.isfinite(bias) and sign else math.nan


def assess_debit_spread(
    row: dict[str, Any] | Any,
    *,
    live: bool,
    expected_move_ratio: float | None = None,
    flow_alignment: float | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate a debit spread using only information available at entry."""
    direction = _clean(row.get("direction"))
    policy = DEBIT_POLICY.get(direction)
    if policy is None:
        return False, ["unsupported_debit_direction"]

    reasons: list[str] = []
    source = _clean(row.get("bot_flow_source_status")).lower()
    flow_quality = _clean(row.get("flow_quality")).lower()
    full_bot_flow = _has_full_bot_flow(source)
    directional_contract_flow = flow_quality == "directional"
    if not full_bot_flow and not directional_contract_flow:
        reasons.append("side_aware_bot_or_directional_contract_flow_required")

    regime = _clean(row.get("regime_trend") or row.get("regime") or row.get("trend")).lower()
    if live or regime:
        if regime not in policy["allowed_regimes"]:
            reasons.append(f"regime_not_aligned:{regime or 'unknown'}")

    dte = safe_float(row.get("dte"))
    allowed_dte_ranges = policy["allowed_dte_ranges"]
    if not math.isfinite(dte) or not any(low <= dte <= high for low, high in allowed_dte_ranges):
        allowed = "_or_".join(f"{low}_{high}" for low, high in allowed_dte_ranges)
        reasons.append(f"dte_outside_{allowed}")

    debit_pct = safe_float(row.get("entry_debit_pct_width"), safe_float(row.get("debit_pct_width")))
    if not math.isfinite(debit_pct) or debit_pct <= 0 or debit_pct > policy["max_debit_pct_width"]:
        reasons.append(f"debit_pct_width_above_{policy['max_debit_pct_width']:.2f}")

    reward_risk = safe_float(row.get("reward_risk"))
    if not math.isfinite(reward_risk) or reward_risk < policy["min_reward_risk"]:
        reasons.append(f"reward_risk_below_{policy['min_reward_risk']:.2f}")

    ratio = safe_float(expected_move_ratio, safe_float(row.get("expected_move_ratio")))
    if not math.isfinite(ratio) or ratio < policy["min_expected_move_ratio"]:
        reasons.append(f"breakeven_expected_move_ratio_below_{policy['min_expected_move_ratio']:.2f}")

    align = safe_float(flow_alignment, _flow_alignment(row))
    if not math.isfinite(align) or align < policy["min_flow_alignment"]:
        reasons.append(f"flow_alignment_below_{policy['min_flow_alignment']:.2f}")

    quote_width = safe_float(row.get("entry_quote_width_pct"), safe_float(row.get("quote_width_pct")))
    if not math.isfinite(quote_width) or quote_width > policy["max_quote_width_pct"]:
        reasons.append(f"quote_width_above_{policy['max_quote_width_pct']:.2f}")

    iv_rank = safe_float(row.get("iv_rank"))
    if live and not math.isfinite(iv_rank):
        reasons.append("iv_rank_missing")
    elif math.isfinite(iv_rank) and iv_rank > policy["max_iv_rank"]:
        reasons.append(f"iv_rank_above_{int(policy['max_iv_rank'])}")

    if flow_quality in {"spread_leg", "mixed", "ambiguous", "weak_or_ambiguous"}:
        reasons.append(f"contract_flow_{flow_quality}")

    if live:
        oi_status = _clean(row.get("oi_carryover_status")).lower()
        if oi_status not in {"supportive", "matched_unconfirmed"}:
            reasons.append(f"exact_leg_oi_not_confirmed:{oi_status or 'unknown'}")
        sample = safe_float(row.get("edge_sample_size"))
        profit_factor = safe_float(row.get("edge_profit_factor"))
        avg_pnl = safe_float(row.get("edge_avg_pnl"))
        if not math.isfinite(sample) or sample < 12:
            reasons.append("debit_edge_sample_below_12")
        if not math.isfinite(profit_factor) or profit_factor < 1.25:
            reasons.append("debit_edge_pf_below_1.25")
        if not math.isfinite(avg_pnl) or avg_pnl <= 0:
            reasons.append("debit_edge_avg_pnl_not_positive")

    return not reasons, reasons


def debit_spread_confidence(
    row: dict[str, Any] | Any,
    *,
    live: bool,
    expected_move_ratio: float | None = None,
    flow_alignment: float | None = None,
) -> tuple[str, list[str]]:
    """Return reject, medium, or high without using future outcomes."""
    ok, reasons = assess_debit_spread(
        row,
        live=live,
        expected_move_ratio=expected_move_ratio,
        flow_alignment=flow_alignment,
    )
    if not ok:
        return "reject", reasons
    if not live:
        return "qualified", []

    high_reasons: list[str] = []
    if not _has_full_bot_flow(row.get("bot_flow_source_status")):
        high_reasons.append("high_requires_full_bot_flow")
    if _clean(row.get("flow_quality")).lower() != "directional":
        high_reasons.append("high_requires_directional_contract_flow")
    if _clean(row.get("oi_carryover_status")).lower() != "supportive":
        high_reasons.append("high_requires_supportive_exact_leg_oi")
    sample = safe_float(row.get("edge_sample_size"))
    profit_factor = safe_float(row.get("edge_profit_factor"))
    if not math.isfinite(sample) or sample < 20:
        high_reasons.append("high_requires_edge_sample_20")
    if not math.isfinite(profit_factor) or profit_factor < 1.35:
        high_reasons.append("high_requires_edge_pf_1.35")
    if _clean(row.get("edge_match_level")).lower() not in {"exact", "ticker_direction", "strategy_regime"}:
        high_reasons.append("high_requires_specific_edge_match")
    quote_width = safe_float(row.get("entry_quote_width_pct"), safe_float(row.get("quote_width_pct")))
    if not math.isfinite(quote_width) or quote_width > 0.20:
        high_reasons.append("high_requires_quote_width_0.20")
    return ("high", []) if not high_reasons else ("medium", high_reasons)
