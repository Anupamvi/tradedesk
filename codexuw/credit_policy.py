from __future__ import annotations

import math
from typing import Any

from .data import safe_float


CREDIT_POLICY_VERSION = "credit-v1.4-regime-contract"

MIN_DTE = 7
MAX_DTE = 45
MIN_CREDIT_PCT_WIDTH = 0.25
MIN_WATCH_CREDIT_PCT_WIDTH = 0.20
MAX_CREDIT_PCT_WIDTH = 0.30
MIN_FLOW_ALIGNMENT = 0.10
MAX_QUOTE_WIDTH_PCT = 0.35
MIN_IV_HV_RATIO = 0.90
MIN_DISTANCE_EXPECTED_MOVE_RATIO = 0.75
ALLOWED_REGIMES = {
    "Bull Put": {"uptrend"},
    "Bear Call": {"downtrend"},
}


def _clean(value: object) -> str:
    return str(value or "").strip()


def _direction_sign(direction: str) -> int:
    if direction == "Bull Put":
        return 1
    if direction == "Bear Call":
        return -1
    return 0


def _flow_alignment(row: dict[str, Any] | Any) -> float:
    bias = safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), math.nan))
    sign = _direction_sign(_clean(row.get("direction")))
    return bias * sign if math.isfinite(bias) and sign else math.nan


def _regime(row: dict[str, Any] | Any) -> str:
    return _clean(row.get("regime_trend") or row.get("regime")).lower()


def _iv_hv_ratio(row: dict[str, Any] | Any) -> float:
    explicit = safe_float(row.get("iv_hv_ratio"))
    if math.isfinite(explicit) and explicit > 0:
        return explicit
    implied = safe_float(row.get("iv30d"))
    realized = safe_float(row.get("realized_volatility_30d"), safe_float(row.get("volatility")))
    return implied / realized if math.isfinite(implied) and math.isfinite(realized) and realized > 0 else math.nan


def credit_spread_edge_lane(
    row: dict[str, Any] | Any,
    *,
    expected_move_ratio: float | None = None,
) -> str:
    ratio = safe_float(expected_move_ratio, safe_float(row.get("expected_move_ratio")))
    iv_hv = _iv_hv_ratio(row)
    volatility_edge = math.isfinite(iv_hv) and iv_hv >= MIN_IV_HV_RATIO
    distance_edge = math.isfinite(ratio) and ratio >= MIN_DISTANCE_EXPECTED_MOVE_RATIO
    if not distance_edge:
        return "none"
    return "volatility_and_distance" if volatility_edge else "distance_buffer"


def assess_credit_spread(
    row: dict[str, Any] | Any,
    *,
    live: bool,
    expected_move_ratio: float | None = None,
    flow_alignment: float | None = None,
) -> tuple[bool, list[str]]:
    """Evaluate a vertical credit spread using entry-time evidence only."""
    direction = _clean(row.get("direction"))
    if direction not in {"Bull Put", "Bear Call"}:
        return False, ["unsupported_credit_direction"]

    reasons: list[str] = []
    regime = _regime(row)
    if regime not in ALLOWED_REGIMES[direction]:
        reasons.append(f"credit_regime_not_aligned:{direction}:{regime or 'unknown'}")
    dte = safe_float(row.get("dte"))
    if not math.isfinite(dte) or not MIN_DTE <= dte <= MAX_DTE:
        reasons.append(f"dte_outside_{MIN_DTE}_{MAX_DTE}")

    credit_pct = safe_float(row.get("entry_credit_pct_width"), safe_float(row.get("credit_pct_width")))
    if not math.isfinite(credit_pct) or not MIN_CREDIT_PCT_WIDTH <= credit_pct <= MAX_CREDIT_PCT_WIDTH:
        reasons.append(f"credit_pct_width_outside_{MIN_CREDIT_PCT_WIDTH:.2f}_{MAX_CREDIT_PCT_WIDTH:.2f}")

    align = safe_float(flow_alignment, _flow_alignment(row))
    if not math.isfinite(align) or align < MIN_FLOW_ALIGNMENT:
        reasons.append(f"flow_alignment_below_{MIN_FLOW_ALIGNMENT:.2f}")

    quote_width = safe_float(row.get("entry_quote_width_pct"), safe_float(row.get("quote_width_pct")))
    if not math.isfinite(quote_width) or quote_width > MAX_QUOTE_WIDTH_PCT:
        reasons.append(f"quote_width_above_{MAX_QUOTE_WIDTH_PCT:.2f}")

    if _clean(row.get("flow_quality")).lower() == "hedge":
        reasons.append("contract_flow_hedge")

    lane = credit_spread_edge_lane(row, expected_move_ratio=expected_move_ratio)
    if lane == "none":
        reasons.append("credit_short_strike_inside_distance_buffer")

    if live:
        oi_status = _clean(row.get("oi_carryover_status")).lower()
        if oi_status not in {"supportive", "matched_unconfirmed"}:
            reasons.append(f"exact_leg_oi_not_confirmed:{oi_status or 'unknown'}")
        sample = safe_float(row.get("edge_sample_size"))
        profit_factor = safe_float(row.get("edge_profit_factor"))
        avg_pnl = safe_float(row.get("edge_avg_pnl"))
        if not math.isfinite(sample) or sample < 12:
            reasons.append("credit_edge_sample_below_12")
        if not math.isfinite(profit_factor) or profit_factor < 1.25:
            reasons.append("credit_edge_pf_below_1.25")
        if not math.isfinite(avg_pnl) or avg_pnl <= 0:
            reasons.append("credit_edge_avg_pnl_not_positive")

    return not reasons, reasons


def credit_spread_confidence(
    row: dict[str, Any] | Any,
    *,
    live: bool,
    expected_move_ratio: float | None = None,
    flow_alignment: float | None = None,
) -> tuple[str, list[str]]:
    ok, reasons = assess_credit_spread(
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
    iv_hv = _iv_hv_ratio(row)
    ratio = safe_float(expected_move_ratio, safe_float(row.get("expected_move_ratio")))
    if not (math.isfinite(ratio) and ratio >= 1.0):
        high_reasons.append("high_requires_distance_edge_1.00")
    if _clean(row.get("bot_flow_source_status")).lower() != "bot_eod_loaded":
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
