from __future__ import annotations

import math
from typing import Any

from .data import safe_float


CREDIT_POLICY_VERSION = "credit-v4.0-regime-map-validated-rv"

# Thresholds below are the ones that survived rolling out-of-sample validation
# on the regenerated 2026 replay base under the corrected exit policy (take
# profit at 50% of credit, no stop). See scripts/strategy_policy_matrix.py.
#
#   guard core (DTE>=28, credit 25-30% of width, IV rank>=30)
#                                           ... n=164  PF 1.484  +$4,430
#   contrarian regime map (see ALLOWED_REGIMES)
#                                           ... n=600  PF 1.196  +$7,110  OOS 3/4
#
# Two former gates were removed because the data showed they were harmful, not
# merely strict:
#   * MIN_DISTANCE_EXPECTED_MOVE_RATIO 0.75 -- jointly unsatisfiable with the
#     credit band (corr(credit%, expected-move ratio) = -0.734; 0 rows in the
#     validated core satisfied it) AND anti-predictive on its own
#     (expected-move-ranked selection scored PF 0.818 vs 0.935 for no selection).
#   * A short-dated floor of 7 DTE -- the sub-28-DTE population is where the
#     edge is destroyed by gamma; lifting the floor to 28 is a tightening.
MIN_DTE = 28
MAX_DTE = 45
MIN_CREDIT_PCT_WIDTH = 0.25
MIN_WATCH_CREDIT_PCT_WIDTH = 0.20
MAX_CREDIT_PCT_WIDTH = 0.30
MAX_QUOTE_WIDTH_PCT = 0.35
MIN_FLOW_ALIGNMENT = 0.10

# ---------------------------------------------------------------------------
# IV/HV richness -- retained at the long-standing 0.90, NOT raised.
#
# This was very nearly shipped at 1.30 on the strength of the underlying panel
# (857,328 ticker-sessions), where "capture" -- the fraction of sold implied vol
# that never realises over the next 21 sessions -- is cleanly monotone in IV/RV:
#
#   IV/RV >= 0.90  keeps 71.5%  capture +0.057  win 66.3%
#   IV/RV >= 1.20  keeps 30.6%  capture +0.135  win 74.6%
#   IV/RV >= 1.30  keeps 21.3%  capture +0.174  win 78.2%
#   IV/RV >= 1.50  keeps 10.6%  capture +0.278  win 84.8%
#
# That result is real but it does not belong to this strategy. Capture is the
# payoff of a variance swap: continuous, symmetric, proportional to the vol
# miss. A defined-risk vertical caps the gain at the credit and the loss at
# (width - credit), roughly 3x the credit, so it monetises almost none of it.
# Replaying the same threshold against actual vertical P&L inside the regime map
# (573 trades, 72 sessions, resampling whole sessions since same-day trades
# share regime and macro shocks):
#
#   no threshold   n=573  PF 1.22   avg +13.1
#   >= 0.90        n=391  PF 1.13   avg  +8.4   delta -4.8  p(no gain) 0.811
#   >= 1.00        n=309  PF 1.06   avg  +3.7   delta -9.4  p(no gain) 0.887
#   >= 1.15        n=186  PF 1.33   avg +16.2   delta +2.8  p(no gain) 0.396
#   >= 1.30        n=101  PF 1.37   avg +16.6   delta +3.2  p(no gain) 0.407
#   >= 1.50        n= 39  PF 1.46   avg +18.1   delta +2.9  p(no gain) 0.421
#
# No threshold separates from no-threshold, every interval spans zero, the PF
# sweep is non-monotone (1.10 -> 1.08, 1.15 -> 1.33, 1.20 -> 1.67, 1.30 -> 1.37,
# 1.40 -> 1.85, 1.50 -> 1.46) and the single best bucket is the CHEAPEST one
# (ratio < 0.80: n=101, PF 1.62). Raising the bar to 1.30 would have cut trade
# count 82% and left 65% of sessions with nothing to trade, in exchange for noise.
#
# The unconditional sweep is even more misleading: it says richness is toxic
# (PF 0.94 -> 0.63) purely because 71% of rich candidates sit in regimes the map
# already blocks, which is where the losses live. Neither direction of that
# result survives conditioning, so IV/HV is kept as the pre-existing sanity
# bound and the regime map does the real work.
MIN_IV_HV_RATIO = 0.90
MIN_IV_RANK = 30.0

# Realised-vol floor. IV/HV is a ratio, so a near-zero denominator manufactures
# enormous readings that are estimation noise rather than premium. Without this
# floor the richest names on 2026-07-24 were short-duration bond and cash ETFs
# -- ICSH, BOXX, JPST, MINT -- with realised vol of 0.3% to 3.8% and ratios of
# 6-12x. Their implied vol averages 0.088, so the credit collected is negligible
# and transaction costs dominate whatever edge exists.
#
# This is an artefact guard, not a performance filter: inside the regime map it
# removes exactly one of 574 replayed trades. It exists so that a degenerate
# denominator can never manufacture a candidate.
MIN_REALIZED_VOL = 0.15

# NOTE: an earnings-in-window exclusion (MAX_DTE_EARNINGS_EXCLUSION = 21) was
# tested here and deliberately NOT adopted. The capture proxy liked it, lifting
# top-quintile capture from +0.171 to +0.206, but on replayed vertical P&L
# earnings names are the better half, not the worse one: inside the regime map
# they run n=154, win 81.2%, PF 1.27 against a +11.6 average for everything
# else. Excluding them is neutral at 21 days (delta -1.7) and significantly
# harmful at 7 days (delta -5.2, 90% CI [-9.8, -1.0]). The genuinely dangerous
# case -- an event landing after the position is opened but before it expires --
# is already a hard reject via `earnings_crosses_expiry`.

# ---------------------------------------------------------------------------
# The 11-27 DTE dead zone.
#
# `MIN_DTE`/`MAX_DTE` above band the primary lane at 28-45. The high-conviction
# decision lanes in engine.py deliberately do NOT call `assess_credit_spread`,
# so they were unbanded on the short end and were writing trades straight into
# the worst duration bucket in the book. Duration splits the credit population
# into three clearly separated regions, and the split replicates on three
# independent slices:
#
#   slice                          0-10 DTE      11-27 DTE       28-45 DTE
#   full credit book (n=2,483)     PF 1.34       PF 0.71-0.80    PF 1.24
#                                                (-$19,327)
#   secondary sleeve (n=127)       PF 2.90       PF 0.72         PF 2.08
#                                  (+$1,572)     (-$1,728)       (+$1,591)
#   primary, map-blocked (n=141)   PF 2.42       PF 0.63         PF 1.54
#                                  (+$1,178)     (-$2,487)       (+$1,581)
#
# The two ends are profitable for different and non-competing reasons: 0-10 DTE
# is fast theta on a position that is closed or expires before a trend can
# develop, and 28-45 DTE is the validated premium-selling core. The middle band
# gets the worst of both -- enough time for the underlying to travel through the
# short strike, not enough theta per day to be paid for the exposure.
#
# Excluding 11-27 from the sleeve, resampling whole sessions:
#
#   sleeve as-is             n=127  win 79.5%  PF 1.17
#   sleeve minus dead zone   n= 64  win 87.5%  PF 2.38
#   delta +38.1 per trade, 90% CI [+10.2, +68.8], p(no gain) 0.012
#
# This is the only change tested this cycle that clears p < 0.05, and it is a
# pure tightening -- it removes trades, never adds them. Note it is a BAND, not
# a floor: raising the floor to `MIN_DTE` would also delete the 0-10 bucket,
# which is the single best-performing region of the book.
DEAD_ZONE_DTE_MIN = 11
DEAD_ZONE_DTE_MAX = 27


def in_dte_dead_zone(dte: Any) -> bool:
    """True when `dte` falls in the validated 11-27 day loss band."""

    value = safe_float(dte)
    if not math.isfinite(value):
        return False
    return DEAD_ZONE_DTE_MIN <= value <= DEAD_ZONE_DTE_MAX


# Credit verticals earn in the regime OPPOSITE to the one they lean on. Selling
# premium *into* a trend leaves you short the tail that is actively moving
# against you; selling it *after* the move collects elevated IV on mean
# reversion. Measured on all 3,258 truncation-corrected replay outcomes:
#
#   contrarian (BP>downtrend, BC>uptrend) ... n=600  PF 1.196  +$7,110  OOS 3/4
#   trend-following (BP>uptrend, BC>down) ... n=874  PF 0.894  -$6,987  OOS 1/4
#
# The trend-following map shipped previously and was inverted: it permitted the
# two losing pairings and blocked the two profitable ones, a $14,097 swing. It
# also blocked Credit|Bear Call|uptrend, the only lane the payoff walk-forward
# independently validated, which is why the daily run produced zero trades.
# Range days are excluded for both directions (BC|range PF 0.814, BP|range 0.856).
ALLOWED_REGIMES = {
    "Bull Put": {"downtrend"},
    "Bear Call": {"uptrend"},
}

# Exit contract for credit verticals. Validated against a 13-point exit grid on
# 3,395 replayed trades: every configuration carrying a hard stop underperformed
# the same configuration without one, in-sample and out-of-sample alike.
# Risk stays fully defined by the spread width, which is unchanged.
PROFIT_TAKE_PCT = 0.50
USE_HARD_STOP = False


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


def _iv_rank(row: dict[str, Any] | Any) -> float:
    return safe_float(row.get("iv_rank"))


def credit_spread_edge_lane(
    row: dict[str, Any] | Any,
    *,
    expected_move_ratio: float | None = None,
) -> str:
    """Classify the source of edge for a credit vertical.

    Edge is priced richness, not strike distance. Distance and credit are two
    readings of the same short-strike delta, so requiring both independently
    double-counts one variable; the credit band already fixes the position on
    that axis.

    What remains informative is whether implied vol is rich *against the name's
    own realised vol*. That is the variance risk premium, and it is what a
    credit vertical actually harvests -- the validated lane wins 89% of the
    time, which is a premium-capture signature, not directional skill.
    """
    iv_hv = _iv_hv_ratio(row)
    iv_rank = _iv_rank(row)
    realized = safe_float(row.get("realized_volatility_30d"))
    # a near-zero denominator makes the ratio meaningless, not attractive
    if math.isfinite(realized) and realized < MIN_REALIZED_VOL:
        return "none"
    volatility_edge = math.isfinite(iv_hv) and iv_hv >= MIN_IV_HV_RATIO
    rank_edge = math.isfinite(iv_rank) and iv_rank >= MIN_IV_RANK
    if not volatility_edge:
        return "none"
    return "volatility_and_rank" if rank_edge else "volatility_premium"


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

    # Flow alignment is retained as a gate at its long-standing 0.10, but it is
    # a conservative bound rather than an edge. Measured as a daily
    # cross-sectional information coefficient against 21-day forward returns on
    # the 2026 panel, every directional flow feature is indistinguishable from
    # noise and carries the wrong sign: net_prem_dir t=-1.15, bull_prem_ratio
    # t=-1.35, side_net_pressure t=-0.31, side_bull_ratio t=-0.36,
    # call_ask_ratio t=-0.35. Of 46 candidate features only one survived a
    # Benjamini-Hochberg correction at the 21-day horizon, and none at 5 days.
    # Rebuilding the same test directly from the bot-eod option tape (66
    # features, 135,371 ticker-sessions) reproduced it: six features cleared
    # BH, all six were U-shaped across quintiles with month-to-month sign flips,
    # and none survived at the 5-day horizon.
    #
    # On replayed P&L inside the regime map the gate is neutral rather than
    # useful (n=297, PF 1.26, delta +1.4, 90% CI [-12.5, +16.1]), so it is kept
    # only because dropping it would loosen entry on no evidence. It must NOT be
    # used for ranking -- see _decision_sort_score.
    align = flow_alignment if flow_alignment is not None else _flow_alignment(row)
    if not math.isfinite(align) or align < MIN_FLOW_ALIGNMENT:
        reasons.append(f"flow_alignment_below_{MIN_FLOW_ALIGNMENT:.2f}")

    quote_width = safe_float(row.get("entry_quote_width_pct"), safe_float(row.get("quote_width_pct")))
    if not math.isfinite(quote_width) or quote_width > MAX_QUOTE_WIDTH_PCT:
        reasons.append(f"quote_width_above_{MAX_QUOTE_WIDTH_PCT:.2f}")

    if _clean(row.get("flow_quality")).lower() == "hedge":
        reasons.append("contract_flow_hedge")

    lane = credit_spread_edge_lane(row, expected_move_ratio=expected_move_ratio)
    if lane == "none":
        realized = safe_float(row.get("realized_volatility_30d"))
        if math.isfinite(realized) and realized < MIN_REALIZED_VOL:
            reasons.append(f"realized_vol_below_{MIN_REALIZED_VOL:.2f}")
        else:
            reasons.append(f"iv_hv_ratio_below_{MIN_IV_HV_RATIO:.2f}")

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
    iv_rank = _iv_rank(row)
    if not (math.isfinite(iv_rank) and iv_rank >= 40.0):
        high_reasons.append("high_requires_iv_rank_40")
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
