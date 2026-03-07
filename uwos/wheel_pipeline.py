#!/usr/bin/env python3
"""
wheel_pipeline.py -- Wheel strategy selection and management pipeline.

Scores stocks on fundamental quality and option premium attractiveness,
applies sentiment adjustments, validates with Schwab live quotes, and
produces actionable wheel trade recommendations (CSP -> shares -> CC cycle).

Usage:
  python -m uwos.wheel_pipeline --date 2026-03-07
  python -m uwos.wheel_pipeline --date 2026-03-07 --no-schwab
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helper: tier_score
# ---------------------------------------------------------------------------

def tier_score(
    value: float,
    excellent: float,
    good: float,
    fair: float,
    lower_is_better: bool = False,
) -> int:
    """Map a metric *value* to a 0-100 score using tier thresholds.

    Returns:
        100 if value reaches the *excellent* threshold,
         75 if it reaches *good*,
         50 if it reaches *fair*,
         25 otherwise.

    When *lower_is_better* is True the comparison direction is reversed
    (e.g. debt-to-equity where smaller is better).
    """
    if lower_is_better:
        if value <= excellent:
            return 100
        if value <= good:
            return 75
        if value <= fair:
            return 50
        return 25
    else:
        if value >= excellent:
            return 100
        if value >= good:
            return 75
        if value >= fair:
            return 50
        return 25


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """Fundamental quality assessment of a wheel candidate."""
    roe: float = 0.0
    roe_score: int = 0
    debt_equity: float = 0.0
    debt_equity_score: int = 0
    rev_growth_yoy: float = 0.0
    rev_growth_score: int = 0
    fcf_yield: float = 0.0
    fcf_yield_score: int = 0
    pe_ratio: float = 0.0
    pe_score: int = 0
    earnings_beats: int = 0
    stability_score: int = 0
    mean_reversion_rate: float = 0.0
    mean_reversion_score: int = 0
    composite: float = 0.0
    disqualified: bool = False
    disqualify_reason: str = ""


@dataclass
class PremiumScore:
    """Option premium attractiveness assessment."""
    csp_strike: float = 0.0
    csp_premium: float = 0.0
    csp_yield_ann: float = 0.0
    csp_yield_score: int = 0
    cc_strike: float = 0.0
    cc_premium: float = 0.0
    cc_yield_ann: float = 0.0
    cc_yield_score: int = 0
    iv_rank: float = 0.0
    iv_rank_score: int = 0
    spread_pct: float = 0.0
    spread_score: int = 0
    composite: float = 0.0


@dataclass
class SentimentAdjustment:
    """Sentiment-based adjustment to the composite score."""
    swing_trend_adj: float = 0.0
    whale_adj: float = 0.0
    dp_adj: float = 0.0
    earnings_adj: float = 0.0
    oi_adj: float = 0.0
    total: float = 0.0
    swing_trend_direction: str = ""
    earnings_days_away: Optional[int] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class WheelCandidate:
    """A scored candidate for wheel trading."""
    ticker: str = ""
    spot: float = 0.0
    sector: str = ""
    market_cap_b: float = 0.0
    quality: QualityScore = field(default_factory=QualityScore)
    premium: PremiumScore = field(default_factory=PremiumScore)
    sentiment: SentimentAdjustment = field(default_factory=SentimentAdjustment)
    composite_raw: float = 0.0
    composite: float = 0.0
    tier: str = ""
    capital_required: float = 0.0
    max_contracts: int = 0
    action: str = ""
    expiry: str = ""
    dte: int = 0
    notes: List[str] = field(default_factory=list)
    live_validated: bool = False
    live_spot: float = 0.0
    live_csp_strike: float = 0.0
    live_csp_premium: float = 0.0
    live_csp_bid_ask: Tuple[float, float] = (0.0, 0.0)
    live_cc_strike: float = 0.0
    live_cc_premium: float = 0.0
    live_cc_bid_ask: Tuple[float, float] = (0.0, 0.0)


@dataclass
class WheelPosition:
    """Tracks an active wheel position through its lifecycle."""
    ticker: str = ""
    tier: str = ""
    phase: str = "csp"
    entry_date: str = ""
    strike: float = 0.0
    expiry: str = ""
    contracts: int = 0
    shares: int = 0
    entry_premium: float = 0.0
    cost_basis: float = 0.0
    capital_reserved: float = 0.0
    cumulative_premium: float = 0.0
    assignment_count: int = 0
    wheel_cycles: int = 0


@dataclass
class DailyAction:
    """A recommended daily action for an existing wheel position."""
    ticker: str = ""
    phase: str = ""
    action: str = ""
    icon: str = ""
    detail: str = ""
    pnl_pct: float = 0.0
    current_premium: float = 0.0
    signal: str = ""
    reason: str = ""
