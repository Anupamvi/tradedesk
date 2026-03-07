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


# ---------------------------------------------------------------------------
# Universe Filter
# ---------------------------------------------------------------------------

def filter_universe(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filter a DataFrame of stock candidates by price, market-cap, and option volume.

    Parameters
    ----------
    df : DataFrame with at least columns: ticker, close, option_volume, market_cap
    cfg : config dict containing a ``universe`` section with thresholds

    Returns
    -------
    Filtered DataFrame with reset index.
    """
    u = cfg["universe"]
    mask = (
        (df["close"] >= u["min_price"])
        & (df["close"] <= u["max_price"])
        & (df["option_volume"] >= u["min_option_volume"])
        & (df["market_cap"] >= u["min_market_cap_b"] * 1e9)
    )
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quality Scorer
# ---------------------------------------------------------------------------

def score_quality(fundamentals: dict, cfg: dict) -> QualityScore:
    """Score a stock's fundamental quality for wheel suitability.

    Parameters
    ----------
    fundamentals : dict with keys roe, debt_equity, rev_growth_yoy, fcf_yield,
                   pe_ratio, earnings_beats, mean_reversion_rate
    cfg : full config dict (uses ``scoring.quality`` section)

    Returns
    -------
    QualityScore dataclass with sub-scores and weighted composite.
    """
    q = cfg["scoring"]["quality"]
    qs = QualityScore()

    # --- raw values ---
    qs.roe = fundamentals["roe"]
    qs.debt_equity = fundamentals["debt_equity"]
    qs.rev_growth_yoy = fundamentals["rev_growth_yoy"]
    qs.fcf_yield = fundamentals["fcf_yield"]
    qs.pe_ratio = fundamentals["pe_ratio"]
    qs.earnings_beats = fundamentals["earnings_beats"]
    qs.mean_reversion_rate = fundamentals["mean_reversion_rate"]

    # --- disqualification check ---
    if qs.debt_equity > q["de_disqualify"]:
        qs.disqualified = True
        qs.disqualify_reason = f"D/E {qs.debt_equity:.2f} > {q['de_disqualify']}"

    # --- sub-scores ---
    # Profitability (ROE): negative ROE = 0 score
    if qs.roe < 0:
        qs.roe_score = 0
    else:
        qs.roe_score = tier_score(qs.roe, excellent=q["roe_excellent"],
                                  good=q["roe_good"], fair=q["roe_fair"])

    # Balance sheet (D/E): lower is better
    qs.debt_equity_score = tier_score(qs.debt_equity, excellent=q["de_excellent"],
                                      good=q["de_good"], fair=q["de_fair"],
                                      lower_is_better=True)

    # Growth
    qs.rev_growth_score = tier_score(qs.rev_growth_yoy, excellent=q["growth_excellent"],
                                     good=q["growth_good"], fair=q["growth_fair"])

    # Cash flow
    qs.fcf_yield_score = tier_score(qs.fcf_yield, excellent=q["fcf_excellent"],
                                    good=q["fcf_good"], fair=q["fcf_fair"])

    # Valuation (P/E): negative P/E = 10 score; otherwise lower is better
    if qs.pe_ratio < 0:
        qs.pe_score = 10
    else:
        qs.pe_score = tier_score(qs.pe_ratio, excellent=q["pe_excellent"],
                                 good=q["pe_good"], fair=q["pe_fair"],
                                 lower_is_better=True)

    # Stability (earnings beats out of 4)
    qs.stability_score = tier_score(qs.earnings_beats, excellent=4, good=3, fair=2)

    # Mean reversion
    qs.mean_reversion_score = tier_score(qs.mean_reversion_rate,
                                         excellent=q["mr_excellent"],
                                         good=q["mr_good"], fair=q["mr_fair"])

    # --- weighted composite ---
    qs.composite = (
        q["profitability_weight"] * qs.roe_score
        + q["balance_sheet_weight"] * qs.debt_equity_score
        + q["growth_weight"] * qs.rev_growth_score
        + q["cash_flow_weight"] * qs.fcf_yield_score
        + q["valuation_weight"] * qs.pe_score
        + q["stability_weight"] * qs.stability_score
        + q["mean_reversion_weight"] * qs.mean_reversion_score
    )

    return qs
