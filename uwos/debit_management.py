#!/usr/bin/env python3
"""Shared management policy for long options and defined-risk debit spreads."""
from __future__ import annotations

from typing import Optional, Tuple


DEBIT_REVIEW_LOSS_PCT = 40.0
DEBIT_HARD_EXIT_LOSS_PCT = 60.0
DEBIT_NEAR_OTM_PCT = 3.0
DEBIT_DEEP_OTM_PCT = 5.0
DEBIT_DECAY_DTE = 35.0
DEBIT_EXPIRY_DTE = 14.0


def evaluate_debit_management(
    *,
    otm_pct: Optional[float],
    dte: Optional[float],
    loss_pct: Optional[float],
    trend_against_thesis: Optional[bool],
) -> Tuple[str, str]:
    """Return HOLD, ASSESS, or CLOSE from observable debit-position risk.

    ``loss_pct`` is a positive loss magnitude. For spreads, callers should use
    the executable whole-spread close credit rather than adding leg marks.
    Missing trend data never creates an artificial time or strike deadline.
    """
    otm = max(float(otm_pct or 0.0), 0.0)
    loss = max(float(loss_pct or 0.0), 0.0)
    days = float(dte) if dte is not None else None

    if loss >= DEBIT_HARD_EXIT_LOSS_PCT:
        return "CLOSE", f"executable debit loss is {loss:.0f}%, at/through the {DEBIT_HARD_EXIT_LOSS_PCT:.0f}% hard exit"
    if days is not None and days <= DEBIT_EXPIRY_DTE and otm > 0:
        return "CLOSE", f"long strike is OTM {otm:.1f}% with {days:.0f} DTE; expiration-window theta risk is now dominant"
    if days is not None and days < DEBIT_DECAY_DTE and otm > DEBIT_DEEP_OTM_PCT:
        return "CLOSE", f"long strike is OTM {otm:.1f}% with {days:.0f} DTE, beyond the {DEBIT_DEEP_OTM_PCT:.0f}% recovery boundary"
    if loss >= DEBIT_REVIEW_LOSS_PCT:
        return "ASSESS", f"executable debit loss is {loss:.0f}%, at/through the {DEBIT_REVIEW_LOSS_PCT:.0f}% review trigger"
    if (
        days is not None
        and days < DEBIT_DECAY_DTE
        and DEBIT_NEAR_OTM_PCT <= otm <= DEBIT_DEEP_OTM_PCT
        and trend_against_thesis is True
    ):
        return "ASSESS", f"long strike is OTM {otm:.1f}% with {days:.0f} DTE and price trend is moving against the thesis"

    trend_text = (
        "trend supports the thesis"
        if trend_against_thesis is False
        else "trend is unavailable, so use price and debit-loss triggers"
    )
    return "HOLD", f"long strike is OTM {otm:.1f}%, debit loss is {loss:.0f}%, and {trend_text}"
