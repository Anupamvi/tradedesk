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
import dataclasses
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
import yfinance as yf


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
    # Filter by issue_type (eliminate ETFs, ADRs, structured products)
    allowed = u.get("allowed_issue_types")
    if allowed and "issue_type" in df.columns:
        mask = mask & df["issue_type"].isin(allowed)
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


# ---------------------------------------------------------------------------
# Mean Reversion Calculator
# ---------------------------------------------------------------------------

def compute_mean_reversion(
    price_df: pd.DataFrame,
    drawdown_pct: float = 10,
    recovery_days: int = 30,
) -> float:
    """Compute the percentage of drawdowns that recovered within *recovery_days*.

    Parameters
    ----------
    price_df : DataFrame with a ``Close`` column (like yfinance history output).
    drawdown_pct : minimum drawdown depth (%) from rolling peak to count.
    recovery_days : number of trading days allowed for recovery.

    Returns
    -------
    Float 0-100 representing the recovery rate.
    - 100.0 if no qualifying drawdowns are found (steady uptrend).
    - 50.0 if fewer than 30 rows of data.
    """
    if len(price_df) < 30:
        return 50.0

    closes = price_df["Close"].values.astype(float)
    n = len(closes)

    # Rolling peak (cumulative max)
    rolling_peak = np.maximum.accumulate(closes)

    # Drawdown percentage at each point
    dd_pct = (rolling_peak - closes) / rolling_peak * 100

    # Find drawdown events: first bar where dd crosses the threshold
    threshold = drawdown_pct
    in_drawdown = False
    drawdown_events: list[dict] = []

    for i in range(n):
        if not in_drawdown and dd_pct[i] >= threshold:
            in_drawdown = True
            peak_val = rolling_peak[i]
            drawdown_events.append({"start": i, "peak": peak_val})
        elif in_drawdown:
            # Drawdown ends when price recovers to prior peak
            if closes[i] >= drawdown_events[-1]["peak"]:
                drawdown_events[-1]["recovered_at"] = i
                in_drawdown = False

    if not drawdown_events:
        return 100.0

    # Count recoveries within the allowed window
    total = len(drawdown_events)
    recovered = 0
    for evt in drawdown_events:
        if "recovered_at" in evt:
            days_to_recover = evt["recovered_at"] - evt["start"]
            if days_to_recover <= recovery_days:
                recovered += 1

    return (recovered / total) * 100.0


# ---------------------------------------------------------------------------
# Fundamentals Fetcher
# ---------------------------------------------------------------------------

def fetch_fundamentals(ticker: str, schwab_quote: Optional[dict] = None) -> dict:
    """Fetch fundamental data for a ticker using yfinance.

    Parameters
    ----------
    ticker : stock ticker symbol.
    schwab_quote : optional Schwab quote dict; if provided, overrides P/E ratio.

    Returns
    -------
    Dict with keys: roe, debt_equity, rev_growth_yoy, fcf_yield, pe_ratio,
    earnings_beats, mean_reversion_rate, sector, market_cap.
    """
    defaults = {
        "roe": 0.0,
        "debt_equity": 999.0,
        "rev_growth_yoy": 0.0,
        "fcf_yield": 0.0,
        "pe_ratio": 0.0,
        "earnings_beats": 0,
        "mean_reversion_rate": 50.0,
        "sector": "Unknown",
        "market_cap": 0.0,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # --- ROE ---
        roe = info.get("returnOnEquity")
        defaults["roe"] = (roe * 100) if roe is not None else 0.0

        # --- Debt/Equity ---
        de = info.get("debtToEquity")
        defaults["debt_equity"] = (de / 100) if de is not None else 999.0

        # --- Revenue growth YoY ---
        rg = info.get("revenueGrowth")
        defaults["rev_growth_yoy"] = (rg * 100) if rg is not None else 0.0

        # --- FCF yield ---
        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")
        if fcf is not None and mcap and mcap > 0:
            defaults["fcf_yield"] = (fcf / mcap) * 100
        else:
            defaults["fcf_yield"] = 0.0

        # --- P/E ratio ---
        pe = info.get("trailingPE")
        defaults["pe_ratio"] = pe if pe is not None else 0.0

        # Override P/E from Schwab if provided
        if schwab_quote is not None:
            schwab_pe = schwab_quote.get("peRatio") or schwab_quote.get("pe_ratio")
            if schwab_pe is not None:
                defaults["pe_ratio"] = float(schwab_pe)

        # --- Sector & market cap ---
        defaults["sector"] = info.get("sector", "Unknown")
        defaults["market_cap"] = float(mcap) if mcap else 0.0

        # --- Earnings beats (last 4 quarters) ---
        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is not None and len(earnings_dates) > 0:
                # Filter to rows that have surprise data
                surprise_col = None
                for col in earnings_dates.columns:
                    if "surprise" in col.lower():
                        surprise_col = col
                        break
                if surprise_col is not None:
                    recent = earnings_dates.head(4)
                    beats = (recent[surprise_col].dropna() > 0).sum()
                    defaults["earnings_beats"] = int(beats)
        except Exception:
            pass

        # --- Mean reversion rate (2-year history) ---
        try:
            hist = stock.history(period="2y")
            if hist is not None and len(hist) > 0:
                defaults["mean_reversion_rate"] = compute_mean_reversion(hist)
        except Exception:
            pass

    except Exception:
        pass

    return defaults


# ---------------------------------------------------------------------------
# Sigma Strike Calculator
# ---------------------------------------------------------------------------

def compute_sigma_strike(
    spot: float,
    iv: float,
    dte: int,
    side: str = "put",
    sigma: float = 1.0,
) -> float:
    """Compute a strike price at *sigma* standard deviations from *spot*.

    Uses a Black-Scholes-style expected move estimate:
        move = spot * iv * sqrt(dte / 365) * sigma

    Parameters
    ----------
    spot : current stock price.
    iv : implied volatility (annualised, as a decimal, e.g. 0.30 for 30%).
    dte : days to expiration.
    side : ``"put"`` (strike below spot) or ``"call"`` (strike above spot).
    sigma : number of standard deviations.

    Returns
    -------
    Strike price rounded to 2 decimals.
    """
    move = spot * iv * math.sqrt(dte / 365) * sigma
    if side == "call":
        return round(spot + move, 2)
    return round(spot - move, 2)


# ---------------------------------------------------------------------------
# Schwab Chain Data Extraction
# ---------------------------------------------------------------------------

def extract_chain_data(
    chain_payload: dict,
    spot: float,
    iv: float,
    dte_target: int = 30,
    sigma: float = 1.0,
) -> dict:
    """Extract CSP and CC pricing from a Schwab option chain payload.

    Parameters
    ----------
    chain_payload : raw JSON dict from Schwab ``get_option_chain``.
    spot : current stock price.
    iv : implied volatility (annualised, decimal).
    dte_target : desired days to expiration.
    sigma : number of standard deviations for strike placement.

    Returns
    -------
    Dict with keys: csp_strike, csp_premium, cc_strike, cc_premium,
    spot, iv_rank, spread_pct, dte.
    """
    zeros = {
        "csp_strike": 0.0, "csp_premium": 0.0,
        "cc_strike": 0.0, "cc_premium": 0.0,
        "spot": spot, "iv_rank": 0, "spread_pct": 0.0, "dte": 0,
    }

    put_map = chain_payload.get("putExpDateMap", {})
    call_map = chain_payload.get("callExpDateMap", {})

    if not put_map and not call_map:
        return zeros

    # --- helper: pick the expiry key closest to dte_target ---
    def _pick_expiry(exp_map: dict) -> Tuple[str, int]:
        """Return (expiry_key, dte) for the expiry closest to dte_target."""
        best_key, best_dte, best_diff = "", 0, float("inf")
        for key in exp_map:
            parts = key.split(":")
            dte_val = int(parts[-1]) if len(parts) == 2 else 0
            diff = abs(dte_val - dte_target)
            if diff < best_diff:
                best_key, best_dte, best_diff = key, dte_val, diff
        return best_key, best_dte

    # --- helper: pick the strike closest to target within an expiry ---
    def _pick_strike(strike_map: dict, target_strike: float) -> Tuple[float, dict]:
        """Return (strike, contract_dict) for the strike closest to target."""
        best_strike, best_diff, best_contract = 0.0, float("inf"), {}
        for strike_str, contracts in strike_map.items():
            strike_val = float(strike_str)
            diff = abs(strike_val - target_strike)
            if diff < best_diff:
                best_strike = strike_val
                best_diff = diff
                best_contract = contracts[0] if contracts else {}
        return best_strike, best_contract

    # --- CSP (put side) ---
    csp_strike, csp_premium, csp_spread = 0.0, 0.0, 0.0
    put_dte = 0
    if put_map:
        put_expiry_key, put_dte = _pick_expiry(put_map)
        if put_expiry_key:
            target_put = compute_sigma_strike(spot, iv, put_dte or dte_target, side="put", sigma=sigma)
            csp_strike, contract = _pick_strike(put_map[put_expiry_key], target_put)
            if contract:
                bid = contract.get("bid", 0.0)
                ask = contract.get("ask", 0.0)
                csp_premium = (bid + ask) / 2.0
                mid = csp_premium if csp_premium > 0 else 1.0
                csp_spread = (ask - bid) / mid * 100.0 if mid else 0.0

    # --- CC (call side) ---
    cc_strike, cc_premium, cc_spread = 0.0, 0.0, 0.0
    call_dte = 0
    if call_map:
        call_expiry_key, call_dte = _pick_expiry(call_map)
        if call_expiry_key:
            target_call = compute_sigma_strike(spot, iv, call_dte or dte_target, side="call", sigma=sigma)
            cc_strike, contract = _pick_strike(call_map[call_expiry_key], target_call)
            if contract:
                bid = contract.get("bid", 0.0)
                ask = contract.get("ask", 0.0)
                cc_premium = (bid + ask) / 2.0
                mid = cc_premium if cc_premium > 0 else 1.0
                cc_spread = (ask - bid) / mid * 100.0 if mid else 0.0

    # Average spread %
    spreads = [s for s in [csp_spread, cc_spread] if s > 0]
    avg_spread = sum(spreads) / len(spreads) if spreads else 0.0

    # Use the put expiry DTE as canonical (or call if no puts)
    dte = put_dte or call_dte

    return {
        "csp_strike": csp_strike,
        "csp_premium": csp_premium,
        "cc_strike": cc_strike,
        "cc_premium": cc_premium,
        "spot": spot,
        "iv_rank": 0,
        "spread_pct": round(avg_spread, 2),
        "dte": dte,
    }


# ---------------------------------------------------------------------------
# Premium Scorer
# ---------------------------------------------------------------------------

def score_premium(chain_data: dict, cfg: dict) -> PremiumScore:
    """Score option premium attractiveness for a wheel candidate.

    Parameters
    ----------
    chain_data : dict with keys csp_premium, csp_strike, cc_premium, cc_strike,
                 spot, iv_rank, spread_pct, dte.
    cfg : full config dict (uses ``scoring.premium`` section).

    Returns
    -------
    PremiumScore dataclass with sub-scores and weighted composite.
    """
    p = cfg["scoring"]["premium"]
    ps = PremiumScore()

    dte = chain_data["dte"]
    if dte <= 0:
        dte = 30

    ps.csp_strike = chain_data["csp_strike"]
    ps.csp_premium = chain_data["csp_premium"]
    ps.cc_strike = chain_data.get("cc_strike", 0.0)
    ps.cc_premium = chain_data.get("cc_premium", 0.0)
    ps.iv_rank = chain_data["iv_rank"]
    ps.spread_pct = chain_data["spread_pct"]

    # --- annualised yields ---
    ps.csp_yield_ann = (ps.csp_premium / ps.csp_strike) * (365 / dte) * 100 if ps.csp_strike else 0.0
    spot = chain_data["spot"]
    ps.cc_yield_ann = (ps.cc_premium / spot) * (365 / dte) * 100 if spot else 0.0

    # --- sub-scores ---
    # CSP yield
    if ps.csp_yield_ann < p["csp_low"]:
        ps.csp_yield_score = 25
    else:
        ps.csp_yield_score = tier_score(ps.csp_yield_ann,
                                         excellent=p["csp_excellent"],
                                         good=p["csp_good"],
                                         fair=p["csp_fair"])

    # CC yield
    if ps.cc_yield_ann < p["cc_low"]:
        ps.cc_yield_score = 25
    else:
        ps.cc_yield_score = tier_score(ps.cc_yield_ann,
                                        excellent=p["cc_excellent"],
                                        good=p["cc_good"],
                                        fair=p["cc_fair"])

    # IV rank
    ps.iv_rank_score = tier_score(ps.iv_rank,
                                   excellent=p["ivr_excellent"],
                                   good=p["ivr_good"],
                                   fair=p["ivr_fair"])

    # Spread quality (lower is better — tight spreads are good)
    ps.spread_score = tier_score(ps.spread_pct,
                                  excellent=p["spread_excellent"],
                                  good=p["spread_good"],
                                  fair=p["spread_fair"],
                                  lower_is_better=True)

    # --- weighted composite ---
    ps.composite = (
        p["csp_yield_weight"] * ps.csp_yield_score
        + p["cc_yield_weight"] * ps.cc_yield_score
        + p["iv_rank_weight"] * ps.iv_rank_score
        + p["spread_quality_weight"] * ps.spread_score
    )

    return ps


# ---------------------------------------------------------------------------
# Sentiment Overlay
# ---------------------------------------------------------------------------

def apply_sentiment(
    *,
    swing_direction: str,
    swing_verdict: str,
    whale_score: float,
    dp_bearish: bool,
    earnings_days: Optional[int],
    oi_confirms: bool,
    cfg: dict,
) -> SentimentAdjustment:
    """Apply sentiment-based adjustments and return a SentimentAdjustment.

    Parameters
    ----------
    swing_direction : "bullish", "bearish", or "" (empty).
    swing_verdict : "PASS", "FAIL", or "" (empty).
    whale_score : 0-100 whale accumulation score.
    dp_bearish : True if dark-pool flow is bearish.
    earnings_days : days until next earnings, or None.
    oi_confirms : True if open-interest confirms directional thesis.
    cfg : full config dict (uses ``sentiment`` section).

    Returns
    -------
    SentimentAdjustment dataclass with itemised adjustments and clamped total.
    """
    s = cfg["sentiment"]
    sa = SentimentAdjustment()
    sa.swing_trend_direction = swing_direction
    sa.earnings_days_away = earnings_days

    # Swing trend
    if swing_direction == "bullish" and swing_verdict == "PASS":
        sa.swing_trend_adj = s["swing_trend_pass_bullish"]
        sa.notes.append(f"Swing trend bullish PASS: +{sa.swing_trend_adj}")
    elif swing_direction == "bearish" or swing_verdict == "FAIL":
        sa.swing_trend_adj = s["swing_trend_fail_bearish"]
        sa.notes.append(f"Swing trend bearish/FAIL: {sa.swing_trend_adj}")

    # Whale accumulation
    if whale_score >= 70:
        sa.whale_adj = s["whale_accumulation_boost"]
        sa.notes.append(f"Whale accumulation (score {whale_score}): +{sa.whale_adj}")

    # Dark pool bearish
    if dp_bearish:
        sa.dp_adj = s["dp_bearish_penalty"]
        sa.notes.append(f"Dark-pool bearish: {sa.dp_adj}")

    # Earnings proximity
    if earnings_days is not None and earnings_days <= 14:
        sa.earnings_adj = s["earnings_within_14d_penalty"]
        sa.notes.append(f"Earnings in {earnings_days}d: {sa.earnings_adj}")

    # OI confirmation
    if oi_confirms:
        sa.oi_adj = s["oi_confirms_direction"]
        sa.notes.append(f"OI confirms direction: +{sa.oi_adj}")

    raw = sa.swing_trend_adj + sa.whale_adj + sa.dp_adj + sa.earnings_adj + sa.oi_adj
    max_adj = s["max_adjustment"]
    sa.total = max(-max_adj, min(max_adj, raw))

    return sa


# ---------------------------------------------------------------------------
# Composite Scoring
# ---------------------------------------------------------------------------

def compute_composite(
    quality: float,
    premium: float,
    sentiment: float,
    quality_weight: float = 0.7,
    premium_weight: float = 0.3,
) -> float:
    """Compute the final composite score, clamped to 0-100.

    Parameters
    ----------
    quality : quality composite score (0-100).
    premium : premium composite score (0-100).
    sentiment : sentiment adjustment (can be negative).
    quality_weight : weight for quality score.
    premium_weight : weight for premium score.

    Returns
    -------
    Float composite score clamped to [0, 100].
    """
    raw = quality * quality_weight + premium * premium_weight + sentiment
    return max(0.0, min(100.0, raw))


# ---------------------------------------------------------------------------
# Tier Assignment
# ---------------------------------------------------------------------------

def assign_tier(composite: float, cfg: dict) -> str:
    """Map a composite score to a tier label.

    Parameters
    ----------
    composite : composite score (0-100).
    cfg : full config dict (uses ``allocation`` section thresholds).

    Returns
    -------
    One of "core", "aggressive", "watchlist", or "excluded".
    """
    a = cfg["allocation"]
    if composite >= a["min_composite_core"]:
        return "core"
    if composite >= a["min_composite_aggressive"]:
        return "aggressive"
    if composite >= a["min_composite_watchlist"]:
        return "watchlist"
    return "excluded"


# ---------------------------------------------------------------------------
# Capital Allocator
# ---------------------------------------------------------------------------

def allocate_capital(
    candidates: List[WheelCandidate],
    capital: float,
    cfg: dict,
) -> List[WheelCandidate]:
    """Allocate capital across wheel candidates respecting position limits.

    Parameters
    ----------
    candidates : list of WheelCandidate objects to consider.
    capital : total available capital (float).
    cfg : config dict (uses ``allocation`` section).

    Returns
    -------
    List of WheelCandidate objects that received an allocation, sorted by
    composite score descending.  Each returned candidate has ``max_contracts``
    and ``capital_required`` populated.
    """
    a = cfg["allocation"]
    max_deployed_pct = a["max_deployed_pct"]
    max_single_name_pct = a["max_single_name_pct"]
    max_positions = a["max_positions"]

    max_deploy = capital * max_deployed_pct
    max_single = capital * max_single_name_pct

    # Sort by composite descending
    sorted_candidates = sorted(candidates, key=lambda c: c.composite, reverse=True)

    # Filter out candidates with no valid strike
    sorted_candidates = [c for c in sorted_candidates if c.premium.csp_strike > 0]

    sector_limit = a.get("sector_concentration_limit", 0.40)
    max_per_sector = capital * sector_limit

    allocated: List[WheelCandidate] = []
    total_deployed = 0.0
    sector_deployed: Dict[str, float] = defaultdict(float)

    for cand in sorted_candidates:
        if len(allocated) >= max_positions:
            break

        per_contract = cand.premium.csp_strike * 100
        remaining = max_deploy - total_deployed

        if remaining < per_contract:
            break

        # Sector concentration check
        sector = cand.sector or "Unknown"
        sector_remaining = max_per_sector - sector_deployed[sector]
        if sector_remaining < per_contract:
            continue  # skip this candidate, sector full

        max_by_single = int(max_single / per_contract)
        max_by_remaining = int(remaining / per_contract)
        max_by_sector = int(sector_remaining / per_contract)
        contracts = min(max_by_single, max_by_remaining, max_by_sector)

        if contracts < 1:
            continue

        cand.max_contracts = contracts
        cand.capital_required = contracts * per_contract
        total_deployed += cand.capital_required
        sector_deployed[sector] += cand.capital_required
        allocated.append(cand)

    return allocated


# ---------------------------------------------------------------------------
# Position Tracker
# ---------------------------------------------------------------------------

class PositionTracker:
    """JSON-backed tracker for active wheel positions and premium journal."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.positions: List[WheelPosition] = []
        self.premium_journal: List[dict] = []
        if path.exists():
            self.load()

    def load(self) -> None:
        """Read JSON file and hydrate positions and journal."""
        with open(self.path, "r") as f:
            data = json.load(f)
        self.positions = [
            WheelPosition(**pos) for pos in data.get("positions", [])
        ]
        self.premium_journal = data.get("premium_journal", [])

    def save(self) -> None:
        """Persist positions and journal to JSON with a last_updated timestamp."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_updated": dt.datetime.now().isoformat(),
            "positions": [dataclasses.asdict(p) for p in self.positions],
            "premium_journal": self.premium_journal,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def add_position(self, pos: WheelPosition) -> None:
        """Append a position to the tracked list."""
        self.positions.append(pos)

    def remove_position(self, ticker: str, phase: str) -> None:
        """Remove the first position matching *ticker* and *phase*."""
        self.positions = [
            p for p in self.positions
            if not (p.ticker == ticker and p.phase == phase)
        ]

    def log_premium(self, date: str, ticker: str, action: str, amount: float,
                    notes: str = "") -> None:
        """Append a premium journal entry."""
        self.premium_journal.append({
            "date": date,
            "ticker": ticker,
            "action": action,
            "amount": amount,
            "notes": notes,
        })

    @property
    def total_capital_reserved(self) -> float:
        """Sum of capital_reserved across all tracked positions."""
        return sum(p.capital_reserved for p in self.positions)


# ---------------------------------------------------------------------------
# Daily Manager
# ---------------------------------------------------------------------------

class DailyManager:
    """Daily decision matrix for managing active wheel positions."""

    def __init__(self, cfg: Dict) -> None:
        self.mgmt = cfg["management"]

    def evaluate_csp(self, pos: WheelPosition, current_premium: float,
                     dte: int, signal: str = "neutral") -> DailyAction:
        """Evaluate a cash-secured put position and recommend an action."""
        pnl_pct = (pos.entry_premium - current_premium) / pos.entry_premium
        close_target = self.mgmt["close_target_pct"]
        dte_threshold = self.mgmt["dte_roll_threshold"]

        action = DailyAction(
            ticker=pos.ticker, phase="csp",
            pnl_pct=pnl_pct, current_premium=current_premium, signal=signal,
        )

        if pnl_pct >= close_target:
            action.action = "CLOSE"
            action.icon = "V"
            action.detail = f"Target reached ({pnl_pct:.0%} profit). Close to lock in gains."
            action.reason = "profit_target"
        elif dte <= dte_threshold and pnl_pct < 0:
            action.action = "ROLL"
            action.icon = "R"
            action.detail = f"Losing ({pnl_pct:.0%}) with {dte} DTE. Roll out for more time."
            action.reason = "low_dte_losing"
        elif dte <= dte_threshold and pnl_pct > 0:
            action.action = "CLOSE"
            action.icon = "V"
            action.detail = f"Winning ({pnl_pct:.0%}) with {dte} DTE. Close before expiry."
            action.reason = "low_dte_winning"
        else:
            action.action = "HOLD"
            action.icon = "H"
            action.detail = f"P&L {pnl_pct:.0%}, {dte} DTE. No action needed."
            action.reason = "hold"

        return action

    def evaluate_shares(self, pos: WheelPosition, spot: float,
                        signal: str = "neutral") -> DailyAction:
        """Evaluate a shares position and recommend covered-call parameters."""
        action = DailyAction(
            ticker=pos.ticker, phase="shares", signal=signal,
            action="SELL_CC",
        )

        if signal == "bullish":
            action.icon = "C"
            action.detail = "Bullish signal: sell CC at 0.5 sigma (aggressive, higher strike)."
            action.reason = "bullish_cc"
        elif signal == "bearish":
            action.icon = "C"
            action.detail = "Bearish signal: sell CC at ATM/ITM for maximum premium."
            action.reason = "bearish_cc"
        else:
            action.icon = "C"
            action.detail = "Neutral: sell CC at 1 sigma, 30 DTE."
            action.reason = "neutral_cc"

        return action

    def evaluate_cc(self, pos: WheelPosition, current_premium: float,
                    dte: int, spot: float) -> DailyAction:
        """Evaluate a covered-call position and recommend an action."""
        pnl_pct = (pos.entry_premium - current_premium) / pos.entry_premium
        close_target = self.mgmt["close_target_pct"]

        action = DailyAction(
            ticker=pos.ticker, phase="cc",
            pnl_pct=pnl_pct, current_premium=current_premium,
        )

        if pnl_pct >= close_target:
            action.action = "CLOSE"
            action.icon = "V"
            action.detail = f"CC target reached ({pnl_pct:.0%} profit). Close and re-sell higher."
            action.reason = "profit_target"
        elif spot >= pos.strike and dte <= 7:
            action.action = "ALLOW_CALL_AWAY"
            action.icon = "A"
            action.detail = f"Spot ${spot:.2f} >= strike ${pos.strike:.2f} with {dte} DTE. Allow assignment."
            action.reason = "called_away"
        else:
            action.action = "HOLD"
            action.icon = "H"
            action.detail = f"CC P&L {pnl_pct:.0%}, {dte} DTE. Hold."
            action.reason = "hold"

        return action


# ---------------------------------------------------------------------------
# Report Writers (Markdown Output)
# ---------------------------------------------------------------------------

_TIER_ICONS = {"core": "GREEN", "aggressive": "RED", "watchlist": "WHITE"}
_TIER_ORDER = ["core", "aggressive", "watchlist"]


def generate_select_report(
    candidates: List[WheelCandidate],
    capital: float,
    as_of: str,
    cfg: dict,
) -> str:
    """Generate the weekly wheel selection report in Markdown.

    Parameters
    ----------
    candidates : scored and allocated WheelCandidate list.
    capital : total available capital.
    as_of : report date string (YYYY-MM-DD).
    cfg : config dict (unused for now, reserved for future formatting knobs).

    Returns
    -------
    Markdown string with tier tables and capital allocation summary.
    """
    by_tier: Dict[str, List[WheelCandidate]] = defaultdict(list)
    for c in candidates:
        by_tier[c.tier].append(c)

    total_allocated = sum(c.capital_required for c in candidates)
    reserve = capital - total_allocated
    alloc_pct = (total_allocated / capital * 100) if capital else 0

    lines: List[str] = []
    lines.append(f"# Wheel Selection Report --- {as_of}")
    lines.append("")
    lines.append(
        f"Capital: ${capital:,.0f} | Allocated: ${total_allocated:,.0f} "
        f"({alloc_pct:.0f}%) | Reserve: ${reserve:,.0f} ({100 - alloc_pct:.0f}%)"
    )
    lines.append("")

    header = (
        "| # | Ticker | Phase | Action | Strike | Expiry | DTE | Premium "
        "| Ann. Yield | Quality | Premium Scr | Composite | Capital Req "
        "| Contracts | Notes |"
    )
    sep = (
        "|---|--------|-------|--------|--------|--------|-----|---------|"
        "------------|---------|-------------|-----------|-------------|"
        "-----------|-------|"
    )

    for tier_key in _TIER_ORDER:
        tier_candidates = by_tier.get(tier_key, [])
        if not tier_candidates:
            continue
        icon = _TIER_ICONS.get(tier_key, tier_key.upper())
        label = tier_key.upper()
        lines.append(f"### {icon} {label} WHEEL")
        lines.append(header)
        lines.append(sep)
        for idx, c in enumerate(tier_candidates, 1):
            notes_str = "; ".join(c.notes) if c.notes else ""
            lines.append(
                f"| {idx} | {c.ticker} | CSP | {c.action} "
                f"| ${c.premium.csp_strike:.2f} | {c.expiry} | {c.dte} "
                f"| ${c.premium.csp_premium:.2f} | {c.premium.csp_yield_ann:.1f}% "
                f"| {c.quality.composite:.0f} | {c.premium.composite:.0f} "
                f"| {c.composite:.1f} | ${c.capital_required:,.0f} "
                f"| {c.max_contracts} | {notes_str} |"
            )
        lines.append("")

    # Capital allocation summary
    lines.append("## Capital Allocation")
    lines.append("| Tier | Ticker | Capital | % of Book | Contracts |")
    lines.append("|------|--------|---------|-----------|-----------|")
    for tier_key in _TIER_ORDER:
        for c in by_tier.get(tier_key, []):
            icon = _TIER_ICONS.get(tier_key, tier_key.upper())
            label = tier_key.upper()
            pct = (c.capital_required / capital * 100) if capital else 0
            lines.append(
                f"| {icon} {label} | {c.ticker} | ${c.capital_required:,.0f} "
                f"| {pct:.0f}% | {c.max_contracts} |"
            )
    lines.append(
        f"| CASH | Reserve | ${reserve:,.0f} "
        f"| {(reserve / capital * 100) if capital else 0:.0f}% | --- |"
    )
    lines.append("")

    return "\n".join(lines)


def generate_daily_report(
    actions: List[DailyAction],
    positions: List[WheelPosition],
    capital: float,
    as_of: str,
    journal: List[dict],
) -> str:
    """Generate the daily wheel management report in Markdown.

    Parameters
    ----------
    actions : list of DailyAction recommendations for today.
    positions : list of active WheelPosition objects.
    capital : total available capital.
    as_of : report date string (YYYY-MM-DD).
    journal : premium journal entries (list of dicts with 'amount' key).

    Returns
    -------
    Markdown string with positions, actions, premium journal, risk dashboard.
    """
    lines: List[str] = []
    lines.append(f"# Wheel Daily --- {as_of}")
    lines.append("")

    # Active Positions
    lines.append("## Active Positions")
    lines.append(
        "| Ticker | Phase | Position | Strike | Expiry | P/L % | Signal | Action |"
    )
    lines.append(
        "|--------|-------|----------|--------|--------|-------|--------|--------|"
    )
    action_map: Dict[str, DailyAction] = {}
    for a in actions:
        action_map[f"{a.ticker}:{a.phase}"] = a

    for pos in positions:
        key = f"{pos.ticker}:{pos.phase}"
        act = action_map.get(key)
        pnl_str = f"{act.pnl_pct:.0%}" if act else "---"
        sig_str = act.signal if act else "---"
        act_str = act.action if act else "HOLD"
        position_str = (
            f"{pos.contracts}x" if pos.phase in ("csp", "cc")
            else f"{pos.shares} shares"
        )
        lines.append(
            f"| {pos.ticker} | {pos.phase.upper()} | {position_str} "
            f"| ${pos.strike:.2f} | {pos.expiry} | {pnl_str} "
            f"| {sig_str} | {act_str} |"
        )
    lines.append("")

    # Actions Today (non-HOLD only)
    non_hold = [a for a in actions if a.action != "HOLD"]
    lines.append("## Actions Today")
    if non_hold:
        for idx, a in enumerate(non_hold, 1):
            lines.append(f"{idx}. **{a.action}** {a.ticker} --- {a.detail}")
    else:
        lines.append("No actions required today.")
    lines.append("")

    # Premium Journal
    total_premium = sum(e.get("amount", 0) for e in journal)
    lines.append("## Premium Journal")
    lines.append(f"Total realized premium: **${total_premium:,.2f}**")
    lines.append("")

    # Risk Dashboard
    total_reserved = sum(p.capital_reserved for p in positions)
    deployed_pct = (total_reserved / capital * 100) if capital else 0
    active_count = len(positions)
    deployed_status = "OK" if deployed_pct <= 65 else "WARN"
    positions_status = "OK" if active_count <= 5 else "WARN"

    lines.append("## Risk Dashboard")
    lines.append("| Metric | Value | Status |")
    lines.append("|--------|-------|--------|")
    lines.append(f"| Capital deployed | {deployed_pct:.0f}% | {deployed_status} |")
    lines.append(f"| Active positions | {active_count} | {positions_status} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline Helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: Path) -> Dict:
    """Load and return the YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_screener(base_dir: Path) -> pd.DataFrame:
    """Find and load stock-screener-*.csv from *base_dir*.

    Normalises column names to lowercase with underscores.
    Returns an empty DataFrame if no screener file is found.
    """
    import glob as _glob

    pattern = str(base_dir / "stock-screener-*.csv")
    matches = sorted(_glob.glob(pattern))
    if not matches:
        return pd.DataFrame()

    df = pd.read_csv(matches[-1])
    # normalise column names: lowercase, spaces/dashes to underscores
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("-", "_")
        for c in df.columns
    ]
    # Derive missing columns expected by filter_universe:
    # 'option_volume' from call_volume + put_volume
    if "option_volume" not in df.columns:
        cv = pd.to_numeric(df.get("call_volume", 0), errors="coerce").fillna(0)
        pv = pd.to_numeric(df.get("put_volume", 0), errors="coerce").fillna(0)
        df["option_volume"] = cv + pv
    # 'market_cap' from 'marketcap' (screener uses no underscore)
    if "market_cap" not in df.columns and "marketcap" in df.columns:
        df["market_cap"] = pd.to_numeric(df["marketcap"], errors="coerce").fillna(0)
    # Ensure 'close' is numeric
    if "close" in df.columns:
        df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0)
    return df


def _load_swing_trend_signals(out_dir: Path, as_of: str) -> Dict[str, Dict]:
    """Load swing-trend shortlist CSVs and return a dict keyed by ticker.

    Checks L5 and L30 lookback files under ``out_dir/swing_trend/``.
    For each ticker extracts direction, verdict, whale_score, oi_direction.
    """
    swing_dir = out_dir / "swing_trend"
    signals: Dict[str, Dict] = {}
    if not swing_dir.exists():
        return signals

    for lookback in ("L5", "L30"):
        csv_path = swing_dir / f"swing_trend_shortlist_{as_of}-{lookback}.csv"
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
            # normalise columns
            df.columns = [
                c.strip().lower().replace(" ", "_").replace("-", "_")
                for c in df.columns
            ]
            ticker_col = "ticker" if "ticker" in df.columns else "symbol"
            if ticker_col not in df.columns:
                continue
            for _, row in df.iterrows():
                tkr = str(row.get(ticker_col, "")).strip().upper()
                if not tkr:
                    continue
                # later lookback overwrites earlier -- L30 overrides L5
                signals[tkr] = {
                    "direction": str(row.get("direction", "")).lower(),
                    "verdict": str(row.get("verdict", "")).upper(),
                    "whale_score": float(row.get("whale_score", 0)),
                    "oi_direction": str(row.get("oi_direction", "")),
                }
        except Exception:
            continue

    return signals


# ---------------------------------------------------------------------------
# Pipeline Orchestration
# ---------------------------------------------------------------------------

def run_select(
    cfg: Dict,
    capital: float,
    base_dir: Path,
    out_dir: Path,
    as_of: str,
    no_schwab: bool = False,
) -> List[WheelCandidate]:
    """Run the wheel selection pipeline end-to-end.

    Returns a list of scored, allocated WheelCandidate objects.
    """
    _err = lambda msg: print(msg, file=sys.stderr)

    # --- Schwab service (optional) ---
    schwab_svc = None
    if not no_schwab:
        try:
            from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
            auth_cfg = SchwabAuthConfig.from_env()
            schwab_svc = SchwabLiveDataService(auth_cfg)
            _err("[wheel] Schwab live data enabled")
        except Exception as exc:
            _err(f"[wheel] Schwab unavailable ({exc}), continuing without live data")
            schwab_svc = None

    # --- Stage 0: load screener & filter universe ---
    _err("[wheel] Stage 0: loading screener ...")
    screener_df = _load_screener(base_dir)
    if screener_df.empty:
        _err("[wheel] No screener data found -- nothing to do")
        return []

    universe_df = filter_universe(screener_df, cfg)
    tickers = universe_df["ticker"].tolist()
    _err(f"[wheel] Universe: {len(tickers)} tickers after filter")

    if not tickers:
        _err("[wheel] Universe is empty after filtering")
        return []

    # --- Stage 1: fundamentals + quality scoring ---
    _err("[wheel] Stage 1: fundamentals & quality scoring ...")
    quality_map: Dict[str, Tuple[QualityScore, dict]] = {}
    spots: Dict[str, float] = {}

    for tkr in tickers:
        # Schwab quote (optional)
        schwab_quote = None
        if schwab_svc is not None:
            try:
                quotes = schwab_svc.get_quotes([tkr])
                schwab_quote = quotes.get(tkr, {})
                last_price = schwab_quote.get("lastPrice") or schwab_quote.get("mark")
                if last_price:
                    spots[tkr] = float(last_price)
            except Exception:
                pass

        # Use screener close if no live spot
        if tkr not in spots:
            row = universe_df.loc[universe_df["ticker"] == tkr]
            if not row.empty:
                spots[tkr] = float(row.iloc[0].get("close", 0))

        fundamentals = fetch_fundamentals(tkr, schwab_quote=schwab_quote)
        qs = score_quality(fundamentals, cfg)
        if qs.disqualified:
            _err(f"  {tkr}: disqualified -- {qs.disqualify_reason}")
            continue
        quality_map[tkr] = (qs, fundamentals)

    surviving = list(quality_map.keys())
    _err(f"[wheel] Stage 1 survivors: {len(surviving)}")

    # --- Stage 2: option chain + premium scoring ---
    _err("[wheel] Stage 2: option chain & premium scoring ...")
    dte_target = cfg["management"]["dte_target"]
    sigma_otm = cfg["management"]["sigma_otm"]

    candidates: List[WheelCandidate] = []

    for tkr in surviving:
        qs, fundamentals = quality_map[tkr]
        spot = spots.get(tkr, 0)
        if spot <= 0:
            continue

        # default IV estimate (30% if unknown)
        iv = 0.30

        chain_data = None
        if schwab_svc is not None:
            try:
                chain_payload = schwab_svc.get_option_chain(tkr)
                chain_data = extract_chain_data(
                    chain_payload, spot, iv,
                    dte_target=dte_target, sigma=sigma_otm,
                )
            except Exception:
                pass

        if chain_data is None:
            # Synthetic chain data when Schwab is unavailable
            csp_strike = compute_sigma_strike(spot, iv, dte_target, side="put", sigma=sigma_otm)
            cc_strike = compute_sigma_strike(spot, iv, dte_target, side="call", sigma=sigma_otm)
            chain_data = {
                "csp_strike": csp_strike,
                "csp_premium": round(spot * 0.02, 2),
                "cc_strike": cc_strike,
                "cc_premium": round(spot * 0.015, 2),
                "spot": spot,
                "iv_rank": 0,
                "spread_pct": 0.0,
                "dte": dte_target,
            }

        ps = score_premium(chain_data, cfg)

        cand = WheelCandidate(
            ticker=tkr,
            spot=spot,
            sector=fundamentals.get("sector", "Unknown"),
            market_cap_b=fundamentals.get("market_cap", 0) / 1e9,
            quality=qs,
            premium=ps,
            action="SELL_CSP",
            dte=chain_data["dte"] or dte_target,
        )
        candidates.append(cand)

    _err(f"[wheel] Stage 2 candidates: {len(candidates)}")

    # --- Stage 3: sentiment overlay ---
    _err("[wheel] Stage 3: sentiment overlay ...")
    swing_signals = _load_swing_trend_signals(out_dir, as_of)
    _err(f"  Swing trend signals loaded for {len(swing_signals)} tickers")

    qw = cfg["scoring"]["quality_weight"]
    pw = cfg["scoring"]["premium_weight"]

    final: List[WheelCandidate] = []
    for cand in candidates:
        sig = swing_signals.get(cand.ticker, {})
        direction = sig.get("direction", "")
        verdict = sig.get("verdict", "")
        whale_score = sig.get("whale_score", 0)
        oi_dir = sig.get("oi_direction", "")

        sa = apply_sentiment(
            swing_direction=direction,
            swing_verdict=verdict,
            whale_score=whale_score,
            dp_bearish=False,
            earnings_days=None,
            oi_confirms=(oi_dir == direction and direction != ""),
            cfg=cfg,
        )
        cand.sentiment = sa
        cand.composite_raw = compute_composite(
            cand.quality.composite, cand.premium.composite, 0.0,
            quality_weight=qw, premium_weight=pw,
        )
        cand.composite = compute_composite(
            cand.quality.composite, cand.premium.composite, sa.total,
            quality_weight=qw, premium_weight=pw,
        )
        cand.tier = assign_tier(cand.composite, cfg)
        if cand.tier == "excluded":
            continue

        # Compute target expiry -- snap to nearest Friday
        as_of_date = dt.date.fromisoformat(as_of)
        target_date = as_of_date + dt.timedelta(days=cand.dte)
        # Snap to Friday (weekday 4)
        days_until_friday = (4 - target_date.weekday()) % 7
        expiry_date = target_date + dt.timedelta(days=days_until_friday)
        cand.expiry = expiry_date.isoformat()

        if sa.notes:
            cand.notes.extend(sa.notes)

        final.append(cand)

    _err(f"[wheel] Non-excluded candidates: {len(final)}")

    # --- Allocate capital ---
    allocated = allocate_capital(final, capital, cfg)
    _err(f"[wheel] Allocated positions: {len(allocated)}")

    # --- Write report ---
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    report_name = cfg.get("output", {}).get(
        "report_md_name", "wheel-select-{date}.md"
    ).format(date=as_of)
    report_path = out_dir_path / report_name
    report_md = generate_select_report(allocated, capital, as_of, cfg)
    report_path.write_text(report_md, encoding="utf-8")
    _err(f"[wheel] Report written: {report_path}")

    return allocated


def run_daily(
    cfg: Dict,
    out_dir: Path,
    as_of: str,
    no_schwab: bool = False,
) -> List[DailyAction]:
    """Run the daily wheel management pipeline.

    Loads positions from wheel_positions.json, evaluates each, and writes
    a daily report.  Returns a list of DailyAction recommendations.
    """
    _err = lambda msg: print(msg, file=sys.stderr)

    # --- Locate positions file ---
    positions_file = Path(cfg["pipeline"]["root_dir"]) / cfg["pipeline"]["positions_file"]
    if not positions_file.exists():
        _err(f"[wheel-daily] No positions file found ({positions_file})")
        return []

    tracker = PositionTracker(positions_file)
    if not tracker.positions:
        _err("[wheel-daily] No active positions -- nothing to do")
        return []

    _err(f"[wheel-daily] Loaded {len(tracker.positions)} position(s)")

    # --- Optional Schwab quotes ---
    schwab_svc = None
    live_quotes: Dict[str, dict] = {}
    if not no_schwab:
        try:
            from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
            auth_cfg = SchwabAuthConfig.from_env()
            schwab_svc = SchwabLiveDataService(auth_cfg)
            tickers = list({p.ticker for p in tracker.positions})
            live_quotes = schwab_svc.get_quotes(tickers)
            _err(f"[wheel-daily] Fetched live quotes for {len(live_quotes)} tickers")
        except Exception as exc:
            _err(f"[wheel-daily] Schwab unavailable ({exc}), using estimates")

    # --- Evaluate each position ---
    mgr = DailyManager(cfg)
    as_of_date = dt.date.fromisoformat(as_of)
    actions: List[DailyAction] = []

    for pos in tracker.positions:
        # DTE remaining
        try:
            expiry_date = dt.date.fromisoformat(pos.expiry)
            dte_remaining = (expiry_date - as_of_date).days
        except (ValueError, TypeError):
            dte_remaining = 30

        # Current premium estimate
        current_premium = pos.entry_premium * 0.5  # placeholder
        spot = pos.strike  # default

        if pos.ticker in live_quotes:
            q = live_quotes[pos.ticker]
            last = q.get("lastPrice") or q.get("mark")
            if last:
                spot = float(last)

        if pos.phase == "csp":
            action = mgr.evaluate_csp(pos, current_premium, dte_remaining)
        elif pos.phase == "shares":
            action = mgr.evaluate_shares(pos, spot)
        elif pos.phase == "cc":
            action = mgr.evaluate_cc(pos, current_premium, dte_remaining, spot)
        else:
            action = DailyAction(
                ticker=pos.ticker, phase=pos.phase,
                action="HOLD", icon="H",
                detail=f"Unknown phase '{pos.phase}' -- holding.",
            )

        actions.append(action)

    # --- Write daily report ---
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    daily_name = cfg.get("output", {}).get(
        "daily_md_name", "wheel-daily-{date}.md"
    ).format(date=as_of)
    daily_path = out_dir_path / daily_name
    capital = cfg["pipeline"].get("default_capital", 35000)
    report_md = generate_daily_report(
        actions, tracker.positions, capital, as_of, tracker.premium_journal,
    )
    daily_path.write_text(report_md, encoding="utf-8")
    _err(f"[wheel-daily] Report written: {daily_path}")

    return actions


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for the wheel pipeline."""
    parser = argparse.ArgumentParser(
        description="Wheel strategy selection and daily management pipeline.",
    )
    parser.add_argument(
        "--mode", choices=["select", "daily", "full"], default="select",
        help="Pipeline mode: select (new candidates), daily (manage positions), "
             "full (both). Default: select.",
    )
    parser.add_argument(
        "--capital", type=float, default=None,
        help="Total available capital. Default: from config pipeline.default_capital.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. Default: <module_dir>/wheel_config.yaml.",
    )
    parser.add_argument(
        "--base-dir", type=str, default=None,
        help="Base directory with screener data. Default: <root_dir>/<as_of>.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for reports. Default: from config pipeline.output_dir.",
    )
    parser.add_argument(
        "--as-of", type=str, default=None,
        help="Pipeline date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--no-schwab", action="store_true",
        help="Disable Schwab live data (use synthetic estimates).",
    )

    args = parser.parse_args()

    # --- Config ---
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).parent / "wheel_config.yaml"
    cfg = _load_config(config_path)

    # --- as_of ---
    as_of = args.as_of or dt.date.today().isoformat()

    # --- capital ---
    capital = args.capital or cfg["pipeline"]["default_capital"]

    # --- base_dir ---
    root_dir = Path(cfg["pipeline"]["root_dir"])
    base_dir = Path(args.base_dir) if args.base_dir else root_dir / as_of

    # --- out_dir ---
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["pipeline"]["output_dir"])

    # --- no_schwab override ---
    if args.no_schwab:
        cfg.setdefault("schwab_validation", {})["enabled"] = False

    print(f"[wheel] mode={args.mode}  as_of={as_of}  capital=${capital:,.0f}", file=sys.stderr)
    print(f"[wheel] base_dir={base_dir}  out_dir={out_dir}", file=sys.stderr)

    # --- Run ---
    if args.mode in ("select", "full"):
        candidates = run_select(
            cfg, capital, base_dir, out_dir, as_of,
            no_schwab=args.no_schwab,
        )
        print(f"[wheel] Selection complete: {len(candidates)} candidates", file=sys.stderr)

    if args.mode in ("daily", "full"):
        actions = run_daily(cfg, out_dir, as_of, no_schwab=args.no_schwab)
        print(f"[wheel] Daily complete: {len(actions)} actions", file=sys.stderr)


if __name__ == "__main__":
    main()