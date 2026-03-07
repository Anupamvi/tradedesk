# Wheel Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a wheel trading pipeline (`uwos/wheel_pipeline.py`) that scores stocks for wheel suitability and manages daily positions.

**Architecture:** Python module following existing uwos patterns (dataclasses, YAML config, Schwab integration, markdown output). Two modes: `select` (weekly candidate scoring) and `daily` (position management). Invoked via `/wheel` Claude skill.

**Tech Stack:** Python 3.11+, pandas, numpy, yfinance, schwab-py, PyYAML (all existing deps)

**Design Doc:** `docs/plans/2026-03-07-wheel-pipeline-design.md`

---

## Task 1: Config YAML + Dataclasses

**Files:**
- Create: `uwos/wheel_config.yaml`
- Create: `uwos/wheel_pipeline.py` (initial scaffold with dataclasses only)
- Test: `tests/test_wheel_pipeline.py`

**Step 1: Create wheel_config.yaml**

```yaml
# Wheel Pipeline Configuration
pipeline:
  root_dir: "c:\\uw_root"
  output_dir: "c:\\uw_root\\out\\wheel"
  default_capital: 35000
  positions_file: "wheel_positions.json"
  premium_journal_file: "wheel_premium_journal.csv"

universe:
  min_price: 10
  max_price: 60
  min_option_volume: 500
  max_option_spread_pct: 0.05
  min_market_cap_b: 2.0
  exclude_etfs: true

scoring:
  quality_weight: 0.70
  premium_weight: 0.30
  quality:
    profitability_weight: 0.25
    balance_sheet_weight: 0.15
    growth_weight: 0.175
    cash_flow_weight: 0.175
    valuation_weight: 0.05
    stability_weight: 0.10
    mean_reversion_weight: 0.10
    # Profitability (ROE)
    roe_excellent: 15
    roe_good: 10
    roe_fair: 5
    # Balance Sheet (D/E)
    de_excellent: 0.5
    de_good: 1.0
    de_fair: 2.0
    de_disqualify: 3.0
    # Growth (Rev YoY %)
    growth_excellent: 15
    growth_good: 5
    growth_fair: 0
    # Cash Flow (FCF yield %)
    fcf_excellent: 5
    fcf_good: 3
    fcf_fair: 1
    # Valuation (P/E)
    pe_excellent: 15
    pe_good: 25
    pe_fair: 40
    # Stability (earnings beats out of 4)
    # Mean Reversion (recovery rate %)
    mr_excellent: 75
    mr_good: 50
    mr_fair: 25
    mr_lookback_years: 2
    mr_drawdown_pct: 10
    mr_recovery_days: 30
  premium:
    csp_yield_weight: 0.35
    cc_yield_weight: 0.25
    iv_rank_weight: 0.20
    spread_quality_weight: 0.20
    # CSP yield annualized %
    csp_excellent: 40
    csp_good: 30
    csp_fair: 20
    csp_low: 10
    # CC yield annualized %
    cc_excellent: 30
    cc_good: 20
    cc_fair: 10
    cc_low: 5
    # IV Rank %
    ivr_excellent: 60
    ivr_good: 40
    ivr_fair: 20
    # Spread quality (bid/ask as % of mid)
    spread_excellent: 2
    spread_good: 4
    spread_fair: 6

sentiment:
  swing_trend_pass_bullish: 5
  swing_trend_fail_bearish: -5
  whale_accumulation_boost: 3
  dp_bearish_penalty: -3
  earnings_within_14d_penalty: -5
  oi_confirms_direction: 2
  max_adjustment: 10

allocation:
  max_deployed_pct: 0.65
  max_single_name_pct: 0.25
  max_positions: 5
  core_split: 0.60
  aggressive_split: 0.40
  min_composite_core: 60
  min_composite_aggressive: 45
  min_composite_watchlist: 35

management:
  close_target_pct: 0.50
  dte_target: 30
  dte_roll_threshold: 14
  sigma_otm: 1.0
  max_unrealized_loss_pct: -0.50
  max_consecutive_assignments: 2
  sector_concentration_limit: 0.40
  earnings_close_days: 7

schwab_validation:
  enabled: true
  max_strike_snap_pct: 0.03

output:
  max_candidates: 8
  report_md_name: "wheel-select-{date}.md"
  daily_md_name: "wheel-daily-{date}.md"
```

**Step 2: Create initial wheel_pipeline.py with dataclasses**

```python
"""Wheel trading pipeline — candidate selection and daily position management.

Usage:
    python -m uwos.wheel_pipeline --mode select --capital 35000
    python -m uwos.wheel_pipeline --mode daily
    python -m uwos.wheel_pipeline --mode full --capital 35000
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QualityScore:
    """Ownership quality sub-scores (70% of composite)."""
    roe: float = 0.0                # Return on equity %
    roe_score: float = 0.0
    debt_equity: float = 0.0
    debt_equity_score: float = 0.0
    rev_growth_yoy: float = 0.0
    rev_growth_score: float = 0.0
    fcf_yield: float = 0.0
    fcf_yield_score: float = 0.0
    pe_ratio: float = 0.0
    pe_score: float = 0.0
    earnings_beats: int = 0         # out of 4
    stability_score: float = 0.0
    mean_reversion_rate: float = 0.0  # % of drawdowns that recovered
    mean_reversion_score: float = 0.0
    composite: float = 0.0         # weighted sum 0-100
    disqualified: bool = False
    disqualify_reason: str = ""


@dataclass
class PremiumScore:
    """Premium yield sub-scores (30% of composite)."""
    csp_strike: float = 0.0
    csp_premium: float = 0.0
    csp_yield_ann: float = 0.0     # annualized %
    csp_yield_score: float = 0.0
    cc_strike: float = 0.0
    cc_premium: float = 0.0
    cc_yield_ann: float = 0.0
    cc_yield_score: float = 0.0
    iv_rank: float = 0.0
    iv_rank_score: float = 0.0
    spread_pct: float = 0.0        # bid-ask as % of mid
    spread_score: float = 0.0
    composite: float = 0.0


@dataclass
class SentimentAdjustment:
    """Sentiment overlay adjustments."""
    swing_trend_adj: float = 0.0
    whale_adj: float = 0.0
    dp_adj: float = 0.0
    earnings_adj: float = 0.0
    oi_adj: float = 0.0
    total: float = 0.0            # clamped to +/- max
    swing_trend_direction: str = ""
    earnings_days_away: Optional[int] = None
    notes: List[str] = field(default_factory=list)


@dataclass
class WheelCandidate:
    """Complete scored wheel candidate."""
    ticker: str = ""
    spot: float = 0.0
    sector: str = ""
    market_cap_b: float = 0.0
    quality: QualityScore = field(default_factory=QualityScore)
    premium: PremiumScore = field(default_factory=PremiumScore)
    sentiment: SentimentAdjustment = field(default_factory=SentimentAdjustment)
    composite_raw: float = 0.0     # quality*0.7 + premium*0.3
    composite: float = 0.0        # after sentiment adjustment
    tier: str = ""                 # core / tactical / aggressive / watchlist
    capital_required: float = 0.0  # strike * 100 * contracts
    max_contracts: int = 0
    action: str = ""               # e.g. "Sell $38P Apr 17"
    expiry: str = ""
    dte: int = 0
    notes: str = ""

    # Schwab live validation
    live_validated: bool = False
    live_spot: float = 0.0
    live_csp_strike: float = 0.0
    live_csp_premium: float = 0.0
    live_csp_bid_ask: str = ""
    live_cc_strike: float = 0.0
    live_cc_premium: float = 0.0
    live_cc_bid_ask: str = ""


@dataclass
class WheelPosition:
    """Tracked wheel position (persisted in JSON)."""
    ticker: str = ""
    tier: str = ""
    phase: str = ""                # csp / shares / cc
    entry_date: str = ""
    strike: float = 0.0
    expiry: str = ""
    contracts: int = 0
    shares: int = 0
    entry_premium: float = 0.0
    cost_basis: float = 0.0       # for shares phase
    capital_reserved: float = 0.0
    cumulative_premium: float = 0.0
    assignment_count: int = 0
    wheel_cycles: int = 0


@dataclass
class DailyAction:
    """Recommended action from daily manager."""
    ticker: str = ""
    phase: str = ""
    action: str = ""               # CLOSE, HOLD, ROLL, SELL_CC, NEW_CSP, etc.
    icon: str = ""                 # emoji for display
    detail: str = ""               # e.g. "Buy back $38P at $0.40"
    pnl_pct: float = 0.0
    current_premium: float = 0.0
    signal: str = ""               # Bullish / Bearish / Neutral
    reason: str = ""


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def tier_score(value: float, excellent: float, good: float, fair: float,
               *, higher_is_better: bool = True, floor: float = 25.0) -> float:
    """Map a metric to a 0-100 score using tier thresholds."""
    if higher_is_better:
        if value >= excellent:
            return 100.0
        if value >= good:
            return 75.0
        if value >= fair:
            return 50.0
        return floor
    else:  # lower is better (e.g. P/E, D/E)
        if value <= excellent:
            return 100.0
        if value <= good:
            return 75.0
        if value <= fair:
            return 50.0
        return floor
```

**Step 3: Write initial test**

Create `tests/test_wheel_pipeline.py`:

```python
"""Tests for wheel_pipeline module."""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

# Load module dynamically (same pattern as test_setup_likelihood_backtest.py)
_mod_path = Path(__file__).resolve().parent.parent / "uwos" / "wheel_pipeline.py"
_spec = importlib.util.spec_from_file_location("wheel_pipeline", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

tier_score = _mod.tier_score
QualityScore = _mod.QualityScore
PremiumScore = _mod.PremiumScore
WheelCandidate = _mod.WheelCandidate


class TestTierScore(unittest.TestCase):
    """Test the tier_score helper."""

    def test_higher_is_better_excellent(self):
        self.assertEqual(tier_score(20, 15, 10, 5, higher_is_better=True), 100.0)

    def test_higher_is_better_good(self):
        self.assertEqual(tier_score(12, 15, 10, 5, higher_is_better=True), 75.0)

    def test_higher_is_better_fair(self):
        self.assertEqual(tier_score(7, 15, 10, 5, higher_is_better=True), 50.0)

    def test_higher_is_better_below(self):
        self.assertEqual(tier_score(3, 15, 10, 5, higher_is_better=True), 25.0)

    def test_lower_is_better_excellent(self):
        self.assertEqual(tier_score(10, 15, 25, 40, higher_is_better=False), 100.0)

    def test_lower_is_better_good(self):
        self.assertEqual(tier_score(20, 15, 25, 40, higher_is_better=False), 75.0)

    def test_lower_is_better_fair(self):
        self.assertEqual(tier_score(30, 15, 25, 40, higher_is_better=False), 50.0)

    def test_lower_is_better_above(self):
        self.assertEqual(tier_score(50, 15, 25, 40, higher_is_better=False), 25.0)


class TestDataclasses(unittest.TestCase):
    """Test dataclass defaults and construction."""

    def test_quality_score_defaults(self):
        q = QualityScore()
        self.assertEqual(q.composite, 0.0)
        self.assertFalse(q.disqualified)

    def test_wheel_candidate_defaults(self):
        c = WheelCandidate(ticker="BP", spot=40.0)
        self.assertEqual(c.ticker, "BP")
        self.assertEqual(c.tier, "")
        self.assertFalse(c.live_validated)


if __name__ == "__main__":
    unittest.main()
```

**Step 4: Run tests**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add uwos/wheel_config.yaml uwos/wheel_pipeline.py tests/test_wheel_pipeline.py
git commit -m "feat(wheel): add config, dataclasses, and tier_score helper"
```

---

## Task 2: Universe Filter + Quality Scorer

**Files:**
- Modify: `uwos/wheel_pipeline.py`
- Modify: `tests/test_wheel_pipeline.py`

**Step 1: Write failing tests for quality scorer**

Add to `tests/test_wheel_pipeline.py`:

```python
score_quality = _mod.score_quality
filter_universe = _mod.filter_universe


class TestFilterUniverse(unittest.TestCase):
    """Test universe filter."""

    def test_filters_by_price(self):
        df = pd.DataFrame({
            "ticker": ["A", "B", "C"],
            "close": [5.0, 30.0, 100.0],
            "option_volume": [1000, 1000, 1000],
            "market_cap": [5e9, 5e9, 5e9],
        })
        cfg = {"universe": {"min_price": 10, "max_price": 60,
                            "min_option_volume": 500, "min_market_cap_b": 2.0}}
        result = filter_universe(df, cfg)
        self.assertEqual(list(result["ticker"]), ["B"])

    def test_filters_by_market_cap(self):
        df = pd.DataFrame({
            "ticker": ["A", "B"],
            "close": [30.0, 30.0],
            "option_volume": [1000, 1000],
            "market_cap": [1e9, 5e9],
        })
        cfg = {"universe": {"min_price": 10, "max_price": 60,
                            "min_option_volume": 500, "min_market_cap_b": 2.0}}
        result = filter_universe(df, cfg)
        self.assertEqual(list(result["ticker"]), ["B"])


class TestScoreQuality(unittest.TestCase):
    """Test ownership quality scoring."""

    def test_high_quality_stock(self):
        fundamentals = {
            "roe": 20.0, "debt_equity": 0.3, "rev_growth_yoy": 18.0,
            "fcf_yield": 6.0, "pe_ratio": 12.0, "earnings_beats": 4,
            "mean_reversion_rate": 80.0,
        }
        cfg = yaml.safe_load(Path(_mod_path).parent.joinpath(
            "wheel_config.yaml").read_text(encoding="utf-8"))
        q = score_quality(fundamentals, cfg)
        self.assertGreater(q.composite, 85)
        self.assertFalse(q.disqualified)

    def test_disqualified_high_debt(self):
        fundamentals = {
            "roe": 5.0, "debt_equity": 3.5, "rev_growth_yoy": 2.0,
            "fcf_yield": 1.0, "pe_ratio": 30.0, "earnings_beats": 2,
            "mean_reversion_rate": 40.0,
        }
        cfg = yaml.safe_load(Path(_mod_path).parent.joinpath(
            "wheel_config.yaml").read_text(encoding="utf-8"))
        q = score_quality(fundamentals, cfg)
        self.assertTrue(q.disqualified)

    def test_low_quality_stock(self):
        fundamentals = {
            "roe": -5.0, "debt_equity": 2.5, "rev_growth_yoy": -3.0,
            "fcf_yield": 0.5, "pe_ratio": -1.0, "earnings_beats": 1,
            "mean_reversion_rate": 20.0,
        }
        cfg = yaml.safe_load(Path(_mod_path).parent.joinpath(
            "wheel_config.yaml").read_text(encoding="utf-8"))
        q = score_quality(fundamentals, cfg)
        self.assertLess(q.composite, 35)
```

**Step 2: Run tests — verify they fail**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py::TestFilterUniverse -v`
Expected: FAIL — `filter_universe` not defined

**Step 3: Implement filter_universe and score_quality**

Add to `uwos/wheel_pipeline.py`:

```python
def filter_universe(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Filter tickers to wheelable universe based on price, volume, market cap."""
    u = cfg.get("universe", {})
    mask = (
        (df["close"] >= u.get("min_price", 10))
        & (df["close"] <= u.get("max_price", 60))
        & (df["option_volume"] >= u.get("min_option_volume", 500))
        & (df["market_cap"] >= u.get("min_market_cap_b", 2.0) * 1e9)
    )
    return df.loc[mask].reset_index(drop=True)


def score_quality(fundamentals: Dict[str, float], cfg: Dict) -> QualityScore:
    """Compute ownership quality score from fundamental data."""
    qcfg = cfg.get("scoring", {}).get("quality", {})
    q = QualityScore()

    # Profitability (ROE)
    q.roe = fundamentals.get("roe", 0.0)
    q.roe_score = tier_score(q.roe, qcfg.get("roe_excellent", 15),
                             qcfg.get("roe_good", 10), qcfg.get("roe_fair", 5))
    if q.roe < 0:
        q.roe_score = 0.0

    # Balance Sheet (D/E)
    q.debt_equity = fundamentals.get("debt_equity", 0.0)
    if q.debt_equity > qcfg.get("de_disqualify", 3.0):
        q.disqualified = True
        q.disqualify_reason = f"D/E {q.debt_equity:.1f} > {qcfg.get('de_disqualify', 3.0)}"
    q.debt_equity_score = tier_score(q.debt_equity, qcfg.get("de_excellent", 0.5),
                                     qcfg.get("de_good", 1.0), qcfg.get("de_fair", 2.0),
                                     higher_is_better=False)

    # Growth (Rev YoY)
    q.rev_growth_yoy = fundamentals.get("rev_growth_yoy", 0.0)
    q.rev_growth_score = tier_score(q.rev_growth_yoy, qcfg.get("growth_excellent", 15),
                                    qcfg.get("growth_good", 5), qcfg.get("growth_fair", 0))

    # Cash Flow (FCF yield)
    q.fcf_yield = fundamentals.get("fcf_yield", 0.0)
    q.fcf_yield_score = tier_score(q.fcf_yield, qcfg.get("fcf_excellent", 5),
                                   qcfg.get("fcf_good", 3), qcfg.get("fcf_fair", 1))

    # Valuation (P/E)
    q.pe_ratio = fundamentals.get("pe_ratio", 0.0)
    if q.pe_ratio < 0:
        q.pe_score = 10.0  # negative P/E = unprofitable
    else:
        q.pe_score = tier_score(q.pe_ratio, qcfg.get("pe_excellent", 15),
                                qcfg.get("pe_good", 25), qcfg.get("pe_fair", 40),
                                higher_is_better=False)

    # Stability (earnings beats)
    q.earnings_beats = int(fundamentals.get("earnings_beats", 0))
    q.stability_score = {4: 100.0, 3: 75.0, 2: 50.0}.get(q.earnings_beats, 25.0)

    # Mean Reversion
    q.mean_reversion_rate = fundamentals.get("mean_reversion_rate", 0.0)
    q.mean_reversion_score = tier_score(q.mean_reversion_rate,
                                        qcfg.get("mr_excellent", 75),
                                        qcfg.get("mr_good", 50),
                                        qcfg.get("mr_fair", 25))

    # Weighted composite
    q.composite = (
        q.roe_score * qcfg.get("profitability_weight", 0.25)
        + q.debt_equity_score * qcfg.get("balance_sheet_weight", 0.15)
        + q.rev_growth_score * qcfg.get("growth_weight", 0.175)
        + q.fcf_yield_score * qcfg.get("cash_flow_weight", 0.175)
        + q.pe_score * qcfg.get("valuation_weight", 0.05)
        + q.stability_score * qcfg.get("stability_weight", 0.10)
        + q.mean_reversion_score * qcfg.get("mean_reversion_weight", 0.10)
    )
    return q
```

**Step 4: Run tests — verify they pass**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add uwos/wheel_pipeline.py tests/test_wheel_pipeline.py
git commit -m "feat(wheel): add universe filter and quality scorer"
```

---

## Task 3: Fundamentals Fetcher (yfinance + Schwab)

**Files:**
- Modify: `uwos/wheel_pipeline.py`
- Modify: `tests/test_wheel_pipeline.py`

**Step 1: Write failing test**

```python
fetch_fundamentals = _mod.fetch_fundamentals
compute_mean_reversion = _mod.compute_mean_reversion


class TestComputeMeanReversion(unittest.TestCase):
    """Test mean reversion calculation from price history."""

    def test_perfect_recovery(self):
        # Stock drops 15% then recovers fully within 10 days, twice
        dates = pd.date_range("2024-01-01", periods=100)
        prices = np.ones(100) * 50.0
        # First drawdown at day 20
        prices[20:25] = 42.0   # -16%
        prices[25:30] = 50.0   # recovered
        # Second drawdown at day 60
        prices[60:65] = 42.0
        prices[65:70] = 50.0
        df = pd.DataFrame({"Close": prices, "High": prices * 1.01,
                           "Low": prices * 0.99}, index=dates)
        rate = compute_mean_reversion(df, drawdown_pct=10, recovery_days=30)
        self.assertGreaterEqual(rate, 75.0)

    def test_no_drawdowns(self):
        dates = pd.date_range("2024-01-01", periods=100)
        prices = np.linspace(50, 60, 100)  # steady uptrend
        df = pd.DataFrame({"Close": prices, "High": prices * 1.01,
                           "Low": prices * 0.99}, index=dates)
        rate = compute_mean_reversion(df, drawdown_pct=10, recovery_days=30)
        self.assertEqual(rate, 100.0)  # no drawdowns = perfect score
```

**Step 2: Implement compute_mean_reversion and fetch_fundamentals**

Add to `uwos/wheel_pipeline.py`:

```python
import yfinance as yf


def compute_mean_reversion(price_df: pd.DataFrame, drawdown_pct: float = 10,
                           recovery_days: int = 30) -> float:
    """Compute % of drawdowns >drawdown_pct that recovered within recovery_days.

    Returns 100.0 if no drawdowns occurred (steady uptrend = perfect wheel stock).
    """
    closes = price_df["Close"].values
    n = len(closes)
    if n < 30:
        return 50.0  # insufficient data

    # Find rolling max and drawdowns from peak
    peak = closes[0]
    drawdowns = []  # list of (start_idx, trough_idx, peak_price)
    in_drawdown = False
    dd_start = 0

    for i in range(1, n):
        if closes[i] > peak:
            if in_drawdown:
                # recovered — record
                drawdowns.append((dd_start, i, True))
                in_drawdown = False
            peak = closes[i]
        elif (peak - closes[i]) / peak * 100 >= drawdown_pct:
            if not in_drawdown:
                dd_start = i
                in_drawdown = True

    if in_drawdown:
        drawdowns.append((dd_start, n - 1, False))

    if not drawdowns:
        return 100.0  # no significant drawdowns

    recovered_in_time = 0
    total = len(drawdowns)
    for start_idx, end_idx, did_recover in drawdowns:
        if did_recover and (end_idx - start_idx) <= recovery_days:
            recovered_in_time += 1

    return (recovered_in_time / total) * 100.0


def fetch_fundamentals(ticker: str, schwab_quote: Optional[Dict] = None) -> Dict[str, float]:
    """Fetch fundamental data from yfinance + Schwab quote.

    Returns dict with keys: roe, debt_equity, rev_growth_yoy, fcf_yield,
    pe_ratio, earnings_beats, mean_reversion_rate, sector, market_cap.
    """
    result: Dict[str, Any] = {
        "roe": 0.0, "debt_equity": 0.0, "rev_growth_yoy": 0.0,
        "fcf_yield": 0.0, "pe_ratio": 0.0, "earnings_beats": 0,
        "mean_reversion_rate": 50.0, "sector": "", "market_cap": 0.0,
    }

    try:
        tk = yf.Ticker(ticker)
        info = tk.info or {}

        # From yfinance
        result["roe"] = (info.get("returnOnEquity") or 0.0) * 100
        result["debt_equity"] = info.get("debtToEquity", 0.0) / 100 if info.get("debtToEquity") else 0.0
        result["fcf_yield"] = 0.0
        fcf = info.get("freeCashflow", 0)
        mcap = info.get("marketCap", 0)
        if mcap and mcap > 0 and fcf:
            result["fcf_yield"] = (fcf / mcap) * 100
        result["sector"] = info.get("sector", "")
        result["market_cap"] = mcap

        # Revenue growth YoY
        rev_growth = info.get("revenueGrowth")
        result["rev_growth_yoy"] = (rev_growth or 0.0) * 100

        # Earnings beats — check last 4 quarters
        try:
            earnings = tk.earnings_dates
            if earnings is not None and len(earnings) >= 4:
                surprise = earnings.get("Surprise(%)")
                if surprise is not None:
                    recent = surprise.dropna().head(4)
                    result["earnings_beats"] = int((recent > 0).sum())
        except Exception:
            result["earnings_beats"] = 2  # default if unavailable

        # Mean reversion from 2-year price history
        try:
            hist = tk.history(period="2y")
            if hist is not None and len(hist) > 60:
                result["mean_reversion_rate"] = compute_mean_reversion(hist)
        except Exception:
            pass

    except Exception as exc:
        print(f"  [warn] yfinance fetch failed for {ticker}: {exc}", file=sys.stderr)

    # Override P/E and dividend from Schwab if available (more reliable)
    if schwab_quote:
        fund = schwab_quote.get("fundamental", {})
        if fund.get("peRatio"):
            result["pe_ratio"] = fund["peRatio"]
        elif result.get("pe_ratio", 0) == 0:
            result["pe_ratio"] = (schwab_quote.get("quote", {}).get("peRatio") or 0.0)

    return result
```

**Step 3: Run tests**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add uwos/wheel_pipeline.py tests/test_wheel_pipeline.py
git commit -m "feat(wheel): add fundamentals fetcher and mean reversion calculator"
```

---

## Task 4: Premium Scorer (Schwab Chains)

**Files:**
- Modify: `uwos/wheel_pipeline.py`
- Modify: `tests/test_wheel_pipeline.py`

**Step 1: Write failing test**

```python
score_premium = _mod.score_premium
compute_sigma_strike = _mod.compute_sigma_strike


class TestComputeSigmaStrike(unittest.TestCase):
    def test_put_strike_below_spot(self):
        # spot=50, iv=30%, dte=30 → 1-sigma put strike < 50
        strike = compute_sigma_strike(50.0, 0.30, 30, side="put", sigma=1.0)
        self.assertLess(strike, 50.0)
        self.assertGreater(strike, 40.0)

    def test_call_strike_above_spot(self):
        strike = compute_sigma_strike(50.0, 0.30, 30, side="call", sigma=1.0)
        self.assertGreater(strike, 50.0)
        self.assertLess(strike, 60.0)


class TestScorePremium(unittest.TestCase):
    def test_high_premium_stock(self):
        chain_data = {
            "csp_premium": 2.0, "csp_strike": 17.0,
            "cc_premium": 1.5, "cc_strike": 21.0,
            "spot": 19.0, "iv_rank": 55.0,
            "spread_pct": 1.5, "dte": 30,
        }
        cfg = yaml.safe_load(Path(_mod_path).parent.joinpath(
            "wheel_config.yaml").read_text(encoding="utf-8"))
        p = score_premium(chain_data, cfg)
        self.assertGreater(p.composite, 70)

    def test_low_premium_stock(self):
        chain_data = {
            "csp_premium": 0.20, "csp_strike": 40.0,
            "cc_premium": 0.15, "cc_strike": 44.0,
            "spot": 42.0, "iv_rank": 15.0,
            "spread_pct": 5.5, "dte": 30,
        }
        cfg = yaml.safe_load(Path(_mod_path).parent.joinpath(
            "wheel_config.yaml").read_text(encoding="utf-8"))
        p = score_premium(chain_data, cfg)
        self.assertLess(p.composite, 40)
```

**Step 2: Implement**

Add to `uwos/wheel_pipeline.py`:

```python
def compute_sigma_strike(spot: float, iv: float, dte: int,
                         side: str = "put", sigma: float = 1.0) -> float:
    """Compute strike at N sigma from spot using IV and DTE."""
    move = spot * iv * math.sqrt(dte / 365.0) * sigma
    if side == "put":
        return round(spot - move, 2)
    return round(spot + move, 2)


def score_premium(chain_data: Dict[str, float], cfg: Dict) -> PremiumScore:
    """Compute premium yield score from option chain data."""
    pcfg = cfg.get("scoring", {}).get("premium", {})
    p = PremiumScore()
    dte = chain_data.get("dte", 30)
    if dte <= 0:
        dte = 30

    # CSP yield
    p.csp_strike = chain_data.get("csp_strike", 0)
    p.csp_premium = chain_data.get("csp_premium", 0)
    if p.csp_strike > 0:
        p.csp_yield_ann = (p.csp_premium / p.csp_strike) * (365 / dte) * 100
    p.csp_yield_score = tier_score(p.csp_yield_ann, pcfg.get("csp_excellent", 40),
                                   pcfg.get("csp_good", 30), pcfg.get("csp_fair", 20))
    if p.csp_yield_ann < pcfg.get("csp_low", 10):
        p.csp_yield_score = 25.0

    # CC yield
    p.cc_strike = chain_data.get("cc_strike", 0)
    p.cc_premium = chain_data.get("cc_premium", 0)
    spot = chain_data.get("spot", 0)
    if spot > 0:
        p.cc_yield_ann = (p.cc_premium / spot) * (365 / dte) * 100
    p.cc_yield_score = tier_score(p.cc_yield_ann, pcfg.get("cc_excellent", 30),
                                  pcfg.get("cc_good", 20), pcfg.get("cc_fair", 10))
    if p.cc_yield_ann < pcfg.get("cc_low", 5):
        p.cc_yield_score = 25.0

    # IV Rank
    p.iv_rank = chain_data.get("iv_rank", 0)
    p.iv_rank_score = tier_score(p.iv_rank, pcfg.get("ivr_excellent", 60),
                                 pcfg.get("ivr_good", 40), pcfg.get("ivr_fair", 20))

    # Spread quality
    p.spread_pct = chain_data.get("spread_pct", 0)
    p.spread_score = tier_score(p.spread_pct, pcfg.get("spread_excellent", 2),
                                pcfg.get("spread_good", 4), pcfg.get("spread_fair", 6),
                                higher_is_better=False)

    # Weighted composite
    p.composite = (
        p.csp_yield_score * pcfg.get("csp_yield_weight", 0.35)
        + p.cc_yield_score * pcfg.get("cc_yield_weight", 0.25)
        + p.iv_rank_score * pcfg.get("iv_rank_weight", 0.20)
        + p.spread_score * pcfg.get("spread_quality_weight", 0.20)
    )
    return p
```

**Step 3: Run tests**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add uwos/wheel_pipeline.py tests/test_wheel_pipeline.py
git commit -m "feat(wheel): add premium scorer with sigma strike calculation"
```

---

## Task 5: Sentiment Overlay + Composite Scoring

**Files:**
- Modify: `uwos/wheel_pipeline.py`
- Modify: `tests/test_wheel_pipeline.py`

**Step 1: Write failing test**

```python
apply_sentiment = _mod.apply_sentiment
compute_composite = _mod.compute_composite
assign_tier = _mod.assign_tier


class TestSentiment(unittest.TestCase):
    def test_bullish_boost(self):
        adj = apply_sentiment(
            swing_direction="bullish", swing_verdict="PASS",
            whale_score=75, dp_bearish=False, earnings_days=30,
            oi_confirms=True, cfg={"sentiment": {"swing_trend_pass_bullish": 5,
            "whale_accumulation_boost": 3, "oi_confirms_direction": 2,
            "max_adjustment": 10}})
        self.assertGreater(adj.total, 0)
        self.assertLessEqual(adj.total, 10)

    def test_bearish_penalty(self):
        adj = apply_sentiment(
            swing_direction="bearish", swing_verdict="FAIL",
            whale_score=20, dp_bearish=True, earnings_days=10,
            oi_confirms=False, cfg={"sentiment": {"swing_trend_fail_bearish": -5,
            "dp_bearish_penalty": -3, "earnings_within_14d_penalty": -5,
            "max_adjustment": 10}})
        self.assertLess(adj.total, 0)
        self.assertGreaterEqual(adj.total, -10)


class TestComposite(unittest.TestCase):
    def test_composite_weighting(self):
        score = compute_composite(quality=80, premium=60, sentiment=0,
                                  quality_weight=0.7, premium_weight=0.3)
        self.assertAlmostEqual(score, 74.0)

    def test_sentiment_clamped(self):
        score = compute_composite(quality=80, premium=60, sentiment=15,
                                  quality_weight=0.7, premium_weight=0.3)
        # sentiment adds to composite but doesn't exceed 100
        self.assertLessEqual(score, 100.0)


class TestAssignTier(unittest.TestCase):
    def test_core(self):
        self.assertEqual(assign_tier(65, {"allocation": {
            "min_composite_core": 60, "min_composite_aggressive": 45}}), "core")

    def test_aggressive(self):
        self.assertEqual(assign_tier(50, {"allocation": {
            "min_composite_core": 60, "min_composite_aggressive": 45}}), "aggressive")

    def test_watchlist(self):
        self.assertEqual(assign_tier(40, {"allocation": {
            "min_composite_core": 60, "min_composite_aggressive": 45,
            "min_composite_watchlist": 35}}), "watchlist")

    def test_excluded(self):
        self.assertEqual(assign_tier(30, {"allocation": {
            "min_composite_core": 60, "min_composite_aggressive": 45,
            "min_composite_watchlist": 35}}), "excluded")
```

**Step 2: Implement**

Add to `uwos/wheel_pipeline.py`:

```python
def apply_sentiment(*, swing_direction: str = "", swing_verdict: str = "",
                    whale_score: float = 0, dp_bearish: bool = False,
                    earnings_days: Optional[int] = None,
                    oi_confirms: bool = False,
                    cfg: Dict) -> SentimentAdjustment:
    """Apply sentiment overlay adjustments."""
    scfg = cfg.get("sentiment", {})
    adj = SentimentAdjustment()
    adj.swing_trend_direction = swing_direction

    if swing_direction == "bullish" and swing_verdict == "PASS":
        adj.swing_trend_adj = scfg.get("swing_trend_pass_bullish", 5)
        adj.notes.append(f"Swing trend bullish PASS (+{adj.swing_trend_adj})")
    elif swing_direction == "bearish" or swing_verdict == "FAIL":
        adj.swing_trend_adj = scfg.get("swing_trend_fail_bearish", -5)
        adj.notes.append(f"Swing trend bearish/FAIL ({adj.swing_trend_adj})")

    if whale_score >= 70:
        adj.whale_adj = scfg.get("whale_accumulation_boost", 3)
        adj.notes.append(f"Whale accumulation ({adj.whale_adj:+.0f})")

    if dp_bearish:
        adj.dp_adj = scfg.get("dp_bearish_penalty", -3)
        adj.notes.append(f"DP bearish divergence ({adj.dp_adj})")

    if earnings_days is not None:
        adj.earnings_days_away = earnings_days
        if earnings_days <= 14:
            adj.earnings_adj = scfg.get("earnings_within_14d_penalty", -5)
            adj.notes.append(f"Earnings in {earnings_days}d ({adj.earnings_adj})")

    if oi_confirms:
        adj.oi_adj = scfg.get("oi_confirms_direction", 2)
        adj.notes.append(f"OI confirms direction ({adj.oi_adj:+.0f})")

    raw = adj.swing_trend_adj + adj.whale_adj + adj.dp_adj + adj.earnings_adj + adj.oi_adj
    cap = scfg.get("max_adjustment", 10)
    adj.total = max(-cap, min(cap, raw))
    return adj


def compute_composite(quality: float, premium: float, sentiment: float,
                      quality_weight: float = 0.7, premium_weight: float = 0.3) -> float:
    """Compute final composite score."""
    raw = quality * quality_weight + premium * premium_weight + sentiment
    return max(0.0, min(100.0, raw))


def assign_tier(composite: float, cfg: Dict) -> str:
    """Assign tier based on composite score."""
    alloc = cfg.get("allocation", {})
    if composite >= alloc.get("min_composite_core", 60):
        return "core"
    if composite >= alloc.get("min_composite_aggressive", 45):
        return "aggressive"
    if composite >= alloc.get("min_composite_watchlist", 35):
        return "watchlist"
    return "excluded"
```

**Step 3: Run tests**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add uwos/wheel_pipeline.py tests/test_wheel_pipeline.py
git commit -m "feat(wheel): add sentiment overlay, composite scoring, and tier assignment"
```

---

## Task 6: Capital Allocator

**Files:**
- Modify: `uwos/wheel_pipeline.py`
- Modify: `tests/test_wheel_pipeline.py`

**Step 1: Write failing test**

```python
allocate_capital = _mod.allocate_capital


class TestAllocateCapital(unittest.TestCase):
    def test_basic_allocation(self):
        candidates = [
            WheelCandidate(ticker="BP", spot=40, composite=70, tier="core",
                           premium=PremiumScore(csp_strike=38)),
            WheelCandidate(ticker="SOFI", spot=19, composite=55, tier="aggressive",
                           premium=PremiumScore(csp_strike=17)),
        ]
        cfg = {"allocation": {"max_deployed_pct": 0.65, "max_single_name_pct": 0.25,
                              "max_positions": 5, "core_split": 0.6, "aggressive_split": 0.4}}
        result = allocate_capital(candidates, 35000, cfg)
        self.assertTrue(all(c.max_contracts >= 1 for c in result))
        total = sum(c.capital_required for c in result)
        self.assertLessEqual(total, 35000 * 0.65)

    def test_single_name_limit(self):
        candidates = [
            WheelCandidate(ticker="OXY", spot=53, composite=65, tier="core",
                           premium=PremiumScore(csp_strike=50)),
        ]
        cfg = {"allocation": {"max_deployed_pct": 0.65, "max_single_name_pct": 0.25,
                              "max_positions": 5}}
        result = allocate_capital(candidates, 35000, cfg)
        self.assertLessEqual(result[0].capital_required, 35000 * 0.25)

    def test_excludes_zero_strike(self):
        candidates = [
            WheelCandidate(ticker="BAD", spot=30, composite=50, tier="aggressive",
                           premium=PremiumScore(csp_strike=0)),
        ]
        cfg = {"allocation": {"max_deployed_pct": 0.65, "max_single_name_pct": 0.25,
                              "max_positions": 5}}
        result = allocate_capital(candidates, 35000, cfg)
        self.assertEqual(len(result), 0)
```

**Step 2: Implement**

Add to `uwos/wheel_pipeline.py`:

```python
def allocate_capital(candidates: List[WheelCandidate], capital: float,
                     cfg: Dict) -> List[WheelCandidate]:
    """Allocate capital to wheel candidates respecting limits."""
    alloc = cfg.get("allocation", {})
    max_deploy = capital * alloc.get("max_deployed_pct", 0.65)
    max_single = capital * alloc.get("max_single_name_pct", 0.25)
    max_pos = alloc.get("max_positions", 5)

    # Sort by composite descending
    ranked = sorted([c for c in candidates if c.premium.csp_strike > 0],
                    key=lambda c: c.composite, reverse=True)

    allocated = []
    total_deployed = 0.0

    for c in ranked:
        if len(allocated) >= max_pos:
            break
        per_contract = c.premium.csp_strike * 100
        if per_contract <= 0:
            continue
        max_by_single = int(max_single / per_contract)
        max_by_remaining = int((max_deploy - total_deployed) / per_contract)
        contracts = max(0, min(max_by_single, max_by_remaining))
        if contracts < 1:
            continue
        c.max_contracts = contracts
        c.capital_required = contracts * per_contract
        total_deployed += c.capital_required
        allocated.append(c)

    return allocated
```

**Step 3: Run tests, commit**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`

```bash
git add uwos/wheel_pipeline.py tests/test_wheel_pipeline.py
git commit -m "feat(wheel): add capital allocator with position limits"
```

---

## Task 7: Schwab Chain Data Extraction

**Files:**
- Modify: `uwos/wheel_pipeline.py`
- Modify: `tests/test_wheel_pipeline.py`

**Step 1: Write failing test**

```python
extract_chain_data = _mod.extract_chain_data


class TestExtractChainData(unittest.TestCase):
    def test_extracts_csp_and_cc(self):
        """Test extraction from a mock Schwab chain payload."""
        # Minimal mock of Schwab option chain structure
        chain = {
            "underlying": {"mark": 40.0},
            "putExpDateMap": {
                "2026-04-17:42": {
                    "38.0": [{"bid": 0.80, "ask": 0.90, "mark": 0.85,
                              "openInterest": 500, "totalVolume": 100}],
                }
            },
            "callExpDateMap": {
                "2026-04-17:42": {
                    "42.0": [{"bid": 0.70, "ask": 0.80, "mark": 0.75,
                              "openInterest": 400, "totalVolume": 80}],
                }
            },
        }
        data = extract_chain_data(chain, spot=40.0, iv=0.30, dte_target=30,
                                  sigma=1.0)
        self.assertGreater(data["csp_premium"], 0)
        self.assertGreater(data["cc_premium"], 0)
        self.assertEqual(data["dte"], 42)
```

**Step 2: Implement**

Add to `uwos/wheel_pipeline.py`:

```python
def extract_chain_data(chain_payload: Dict, spot: float, iv: float,
                       dte_target: int = 30, sigma: float = 1.0) -> Dict[str, float]:
    """Extract CSP and CC pricing from Schwab chain payload.

    Finds the expiry closest to dte_target, then the strikes closest to
    1-sigma OTM for puts and calls.
    """
    result = {"csp_strike": 0, "csp_premium": 0, "cc_strike": 0,
              "cc_premium": 0, "spot": spot, "iv_rank": 0, "spread_pct": 0,
              "dte": dte_target}

    target_put = compute_sigma_strike(spot, iv, dte_target, "put", sigma)
    target_call = compute_sigma_strike(spot, iv, dte_target, "call", sigma)

    def _best_contract(exp_map: Dict, target_strike: float) -> Tuple[float, float, float, int]:
        """Find closest strike and expiry to targets. Returns (strike, mid, spread_pct, dte)."""
        best_strike, best_mid, best_spread, best_dte = 0, 0, 999, dte_target
        best_dte_diff = 9999
        for exp_key, strikes in exp_map.items():
            parts = exp_key.split(":")
            dte_val = int(parts[1]) if len(parts) > 1 else dte_target
            dte_diff = abs(dte_val - dte_target)
            if dte_diff > best_dte_diff + 14:
                continue  # skip expiries too far from target
            for strike_str, contracts in strikes.items():
                s = float(strike_str)
                c = contracts[0] if contracts else {}
                bid = c.get("bid", 0) or 0
                ask = c.get("ask", 0) or 0
                mid = (bid + ask) / 2 if (bid + ask) > 0 else c.get("mark", 0)
                if mid <= 0:
                    continue
                spread = (ask - bid) / mid * 100 if mid > 0 else 999
                dist = abs(s - target_strike)
                # Prefer closest DTE first, then closest strike
                if (dte_diff < best_dte_diff) or \
                   (dte_diff == best_dte_diff and dist < abs(best_strike - target_strike)):
                    best_strike = s
                    best_mid = mid
                    best_spread = spread
                    best_dte = dte_val
                    best_dte_diff = dte_diff
        return best_strike, best_mid, best_spread, best_dte

    put_map = chain_payload.get("putExpDateMap", {})
    call_map = chain_payload.get("callExpDateMap", {})

    if put_map:
        strike, mid, spread, dte = _best_contract(put_map, target_put)
        result["csp_strike"] = strike
        result["csp_premium"] = mid
        result["spread_pct"] = spread
        result["dte"] = dte

    if call_map:
        strike, mid, spread_c, dte = _best_contract(call_map, target_call)
        result["cc_strike"] = strike
        result["cc_premium"] = mid
        # average spread of put and call
        if result["spread_pct"] > 0:
            result["spread_pct"] = (result["spread_pct"] + spread_c) / 2
        else:
            result["spread_pct"] = spread_c

    return result
```

**Step 3: Run tests, commit**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`

```bash
git add uwos/wheel_pipeline.py tests/test_wheel_pipeline.py
git commit -m "feat(wheel): add Schwab chain data extraction for CSP/CC pricing"
```

---

## Task 8: Position Tracker + Daily Manager

**Files:**
- Modify: `uwos/wheel_pipeline.py`
- Modify: `tests/test_wheel_pipeline.py`

**Step 1: Write failing tests**

```python
PositionTracker = _mod.PositionTracker


class TestPositionTracker(unittest.TestCase):
    def test_load_empty(self):
        tracker = PositionTracker(Path("/tmp/test_wheel_pos.json"))
        self.assertEqual(len(tracker.positions), 0)

    def test_add_and_save(self):
        import tempfile
        p = Path(tempfile.mktemp(suffix=".json"))
        try:
            tracker = PositionTracker(p)
            pos = WheelPosition(ticker="BP", tier="core", phase="csp",
                                entry_date="2026-03-07", strike=38.0,
                                expiry="2026-04-17", contracts=2,
                                entry_premium=0.85, capital_reserved=7600)
            tracker.add_position(pos)
            tracker.save()
            # Reload
            tracker2 = PositionTracker(p)
            tracker2.load()
            self.assertEqual(len(tracker2.positions), 1)
            self.assertEqual(tracker2.positions[0].ticker, "BP")
        finally:
            if p.exists():
                p.unlink()


DailyManager = _mod.DailyManager


class TestDailyManager(unittest.TestCase):
    def test_close_at_50pct(self):
        pos = WheelPosition(ticker="BP", phase="csp", strike=38.0,
                            entry_premium=0.85, contracts=2)
        mgr = DailyManager(cfg={"management": {"close_target_pct": 0.50,
                                                "dte_roll_threshold": 14}})
        action = mgr.evaluate_csp(pos, current_premium=0.40, dte=30,
                                  signal="bullish")
        self.assertEqual(action.action, "CLOSE")

    def test_hold_if_not_target(self):
        pos = WheelPosition(ticker="BP", phase="csp", strike=38.0,
                            entry_premium=0.85, contracts=2)
        mgr = DailyManager(cfg={"management": {"close_target_pct": 0.50,
                                                "dte_roll_threshold": 14}})
        action = mgr.evaluate_csp(pos, current_premium=0.70, dte=30,
                                  signal="bullish")
        self.assertEqual(action.action, "HOLD")

    def test_roll_when_dte_low_and_losing(self):
        pos = WheelPosition(ticker="INTC", phase="csp", strike=42.0,
                            entry_premium=0.95, contracts=1)
        mgr = DailyManager(cfg={"management": {"close_target_pct": 0.50,
                                                "dte_roll_threshold": 14}})
        action = mgr.evaluate_csp(pos, current_premium=1.20, dte=10,
                                  signal="neutral")
        self.assertEqual(action.action, "ROLL")
```

**Step 2: Implement**

Add to `uwos/wheel_pipeline.py`:

```python
class PositionTracker:
    """Read/write wheel positions from JSON file."""

    def __init__(self, path: Path):
        self.path = path
        self.positions: List[WheelPosition] = []
        self.journal: List[Dict] = []
        if path.exists():
            self.load()

    def load(self):
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.positions = [WheelPosition(**p) for p in data.get("positions", [])]
        self.journal = data.get("premium_journal", [])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from dataclasses import asdict
        data = {
            "last_updated": dt.datetime.now().isoformat(),
            "positions": [asdict(p) for p in self.positions],
            "premium_journal": self.journal,
        }
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def add_position(self, pos: WheelPosition):
        self.positions.append(pos)

    def remove_position(self, ticker: str, phase: str):
        self.positions = [p for p in self.positions
                          if not (p.ticker == ticker and p.phase == phase)]

    def log_premium(self, date: str, ticker: str, action: str,
                    amount: float, notes: str = ""):
        self.journal.append({"date": date, "ticker": ticker, "action": action,
                             "premium_realized": amount, "notes": notes})

    @property
    def total_capital_reserved(self) -> float:
        return sum(p.capital_reserved for p in self.positions)


class DailyManager:
    """Evaluate positions and generate daily actions."""

    def __init__(self, cfg: Dict):
        self.cfg = cfg.get("management", {})

    def evaluate_csp(self, pos: WheelPosition, current_premium: float,
                     dte: int, signal: str = "neutral") -> DailyAction:
        """Evaluate a CSP position and recommend action."""
        action = DailyAction(ticker=pos.ticker, phase="csp")
        close_target = self.cfg.get("close_target_pct", 0.50)
        roll_dte = self.cfg.get("dte_roll_threshold", 14)

        pnl_pct = (pos.entry_premium - current_premium) / pos.entry_premium \
            if pos.entry_premium > 0 else 0

        action.pnl_pct = pnl_pct
        action.current_premium = current_premium
        action.signal = signal

        if pnl_pct >= close_target:
            action.action = "CLOSE"
            action.icon = "CLOSE"
            action.detail = (f"Buy back ${pos.strike:.0f}P at ${current_premium:.2f} "
                             f"({pnl_pct:.0%} profit)")
            action.reason = f"Hit {close_target:.0%} profit target"
        elif dte <= roll_dte and pnl_pct < 0:
            action.action = "ROLL"
            action.icon = "ROLL"
            action.detail = f"Roll ${pos.strike:.0f}P out +30 DTE for credit"
            action.reason = f"DTE {dte} <= {roll_dte} and position losing"
        elif dte <= roll_dte and pnl_pct > 0:
            action.action = "CLOSE"
            action.icon = "CLOSE"
            action.detail = f"Close ${pos.strike:.0f}P — expiry approaching"
            action.reason = f"DTE {dte} <= {roll_dte}, profitable"
        else:
            action.action = "HOLD"
            action.icon = "HOLD"
            action.detail = f"${pos.strike:.0f}P at ${current_premium:.2f} ({pnl_pct:+.0%})"
            action.reason = "On track"

        return action

    def evaluate_shares(self, pos: WheelPosition, spot: float,
                        signal: str = "neutral", cfg: Dict = None) -> DailyAction:
        """Evaluate assigned shares and recommend CC action."""
        action = DailyAction(ticker=pos.ticker, phase="shares")
        pnl_pct = (spot - pos.cost_basis) / pos.cost_basis if pos.cost_basis > 0 else 0
        action.pnl_pct = pnl_pct
        action.signal = signal

        if signal == "bullish":
            action.action = "SELL_CC"
            action.detail = f"Sell CC — aggressive strike (0.5 sigma OTM)"
            action.reason = "Bullish signal — tighter strike for more premium"
        elif signal == "bearish":
            action.action = "SELL_CC"
            action.detail = f"Sell CC — ATM or ITM to accelerate exit"
            action.reason = "Bearish signal — prioritize premium over upside"
        else:
            action.action = "SELL_CC"
            action.detail = f"Sell CC — 1 sigma OTM, 30 DTE"
            action.reason = "Neutral signal — standard CC placement"

        action.icon = "SELL_CC"
        return action

    def evaluate_cc(self, pos: WheelPosition, current_premium: float,
                    dte: int, spot: float) -> DailyAction:
        """Evaluate a covered call position."""
        action = DailyAction(ticker=pos.ticker, phase="cc")
        close_target = self.cfg.get("close_target_pct", 0.50)
        pnl_pct = (pos.entry_premium - current_premium) / pos.entry_premium \
            if pos.entry_premium > 0 else 0

        action.pnl_pct = pnl_pct
        action.current_premium = current_premium

        if pnl_pct >= close_target:
            action.action = "CLOSE"
            action.icon = "CLOSE"
            action.detail = f"Buy back ${pos.strike:.0f}C, re-sell higher"
            action.reason = f"Hit {close_target:.0%} profit target"
        elif spot >= pos.strike and dte <= 7:
            action.action = "ALLOW_CALL_AWAY"
            action.icon = "CALLED_AWAY"
            action.detail = f"Shares called away at ${pos.strike:.0f} — restart wheel"
            action.reason = "ITM near expiry"
        else:
            action.action = "HOLD"
            action.icon = "HOLD"
            action.detail = f"${pos.strike:.0f}C at ${current_premium:.2f} ({pnl_pct:+.0%})"
            action.reason = "On track"

        return action
```

**Step 3: Run tests, commit**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`

```bash
git add uwos/wheel_pipeline.py tests/test_wheel_pipeline.py
git commit -m "feat(wheel): add position tracker and daily manager with decision matrix"
```

---

## Task 9: Report Writer (Markdown Output)

**Files:**
- Modify: `uwos/wheel_pipeline.py`

**Step 1: Implement generate_select_report**

Add to `uwos/wheel_pipeline.py`:

```python
def generate_select_report(candidates: List[WheelCandidate], capital: float,
                           as_of: str, cfg: Dict) -> str:
    """Generate wheel selection report markdown."""
    lines = [f"# Wheel Selection Report --- {as_of}\n"]
    deployed = sum(c.capital_required for c in candidates if c.tier != "watchlist")
    reserve = capital - deployed
    lines.append(f"Capital: ${capital:,.0f} | Allocated: ${deployed:,.0f} "
                 f"({deployed/capital:.0%}) | Reserve: ${reserve:,.0f} "
                 f"({reserve/capital:.0%})\n")

    tier_map = {"core": ("CORE WHEEL", "GREEN"),
                "aggressive": ("AGGRESSIVE WHEEL", "RED"),
                "watchlist": ("WATCHLIST", "WHITE")}
    tier_icons = {"core": "GREEN", "aggressive": "RED", "watchlist": "WHITE"}

    for tier_key in ["core", "aggressive", "watchlist"]:
        tier_cands = [c for c in candidates if c.tier == tier_key]
        if not tier_cands:
            continue
        label, color = tier_map[tier_key]
        lines.append(f"\n### {color} {label}")
        lines.append("| # | Ticker | Phase | Action | Strike | Expiry | DTE | "
                     "Premium | Ann. Yield | Quality | Premium Scr | Composite | "
                     "Capital Req | Contracts | Notes |")
        lines.append("|---|--------|-------|--------|--------|--------|-----|"
                     "---------|------------|---------|-------------|-----------|"
                     "-------------|-----------|-------|")
        for i, c in enumerate(tier_cands, 1):
            ann_yield = c.premium.csp_yield_ann
            lines.append(
                f"| {i} | {c.ticker} | CSP | Sell ${c.premium.csp_strike:.0f}P | "
                f"${c.premium.csp_strike:.2f} | {c.expiry} | {c.dte} | "
                f"${c.premium.csp_premium:.2f} | {ann_yield:.1f}% | "
                f"{c.quality.composite:.0f} | {c.premium.composite:.0f} | "
                f"{c.composite:.1f} | ${c.capital_required:,.0f} | "
                f"{c.max_contracts} | {c.notes} |"
            )

    # Capital allocation summary
    lines.append("\n## Capital Allocation")
    lines.append("| Tier | Ticker | Capital | % of Book | Contracts |")
    lines.append("|------|--------|---------|-----------|-----------|")
    for c in candidates:
        if c.tier == "watchlist":
            continue
        pct = c.capital_required / capital * 100 if capital > 0 else 0
        icon = tier_icons.get(c.tier, "")
        lines.append(f"| {icon} {c.tier.title()} | {c.ticker} | "
                     f"${c.capital_required:,.0f} | {pct:.0f}% | {c.max_contracts} |")
    lines.append(f"| CASH | Reserve | ${reserve:,.0f} | "
                 f"{reserve/capital*100:.0f}% | --- |")

    return "\n".join(lines)


def generate_daily_report(actions: List[DailyAction], positions: List[WheelPosition],
                          capital: float, as_of: str, journal: List[Dict]) -> str:
    """Generate daily wheel management report markdown."""
    lines = [f"# Wheel Daily --- {as_of}\n"]

    # Active positions table
    lines.append("## Active Positions")
    lines.append("| Ticker | Phase | Position | Strike | Expiry | P/L % | "
                 "Signal | Action |")
    lines.append("|--------|-------|----------|--------|--------|-------|"
                 "--------|--------|")
    for a in actions:
        lines.append(f"| {a.icon} {a.ticker} | {a.phase.upper()} | "
                     f"{a.detail} | --- | --- | {a.pnl_pct:+.0%} | "
                     f"{a.signal} | {a.action} |")

    # Actions today
    action_items = [a for a in actions if a.action != "HOLD"]
    if action_items:
        lines.append("\n## Actions Today")
        for i, a in enumerate(action_items, 1):
            lines.append(f"{i}. **{a.action}** {a.ticker} --- {a.detail} "
                         f"({a.reason})")

    # Premium journal summary
    if journal:
        total_realized = sum(j.get("premium_realized", 0) for j in journal)
        lines.append("\n## Premium Journal")
        lines.append(f"Total realized premium: **${total_realized:,.2f}**")

    # Risk dashboard
    total_reserved = sum(p.capital_reserved for p in positions)
    lines.append("\n## Risk Dashboard")
    lines.append("| Metric | Value | Status |")
    lines.append("|--------|-------|--------|")
    deployed_pct = total_reserved / capital if capital > 0 else 0
    status = "OK" if deployed_pct < 0.65 else "WARNING"
    lines.append(f"| Capital deployed | {deployed_pct:.0%} | {status} |")
    lines.append(f"| Active positions | {len(positions)} | "
                 f"{'OK' if len(positions) <= 5 else 'WARNING'} |")

    # Sector concentration
    sectors = defaultdict(float)
    for p in positions:
        sectors["Unknown"] += p.capital_reserved  # simplified
    if sectors:
        max_sector_pct = max(sectors.values()) / capital if capital > 0 else 0
        status = "OK" if max_sector_pct < 0.40 else "WARNING"
        lines.append(f"| Max sector concentration | {max_sector_pct:.0%} | {status} |")

    return "\n".join(lines)
```

**Step 2: Commit**

```bash
git add uwos/wheel_pipeline.py
git commit -m "feat(wheel): add markdown report generators for select and daily modes"
```

---

## Task 10: Pipeline Orchestration + CLI

**Files:**
- Modify: `uwos/wheel_pipeline.py`

**Step 1: Implement run_select, run_daily, and main()**

Add to `uwos/wheel_pipeline.py`:

```python
def _load_config(config_path: Path) -> Dict:
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def _load_screener(base_dir: Path) -> pd.DataFrame:
    """Load stock-screener CSV from base_dir."""
    matches = sorted(base_dir.glob("stock-screener-*.csv"))
    if not matches:
        print(f"  [warn] No stock-screener CSV in {base_dir}", file=sys.stderr)
        return pd.DataFrame()
    df = pd.read_csv(matches[-1])
    # Normalize column names
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower().replace(" ", "_")
        col_map[c] = cl
    df.rename(columns=col_map, inplace=True)
    return df


def _load_swing_trend_signals(out_dir: Path, as_of: str) -> Dict[str, Dict]:
    """Load latest swing trend report for sentiment overlay."""
    signals = {}
    for lookback in [5, 30]:
        path = out_dir / "swing_trend" / f"swing_trend_shortlist_{as_of}-L{lookback}.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                ticker = row.get("ticker", "")
                if ticker and ticker not in signals:
                    signals[ticker] = {
                        "direction": row.get("price_direction", ""),
                        "verdict": row.get("backtest_verdict", ""),
                        "whale_score": row.get("whale_consensus", 0),
                        "oi_direction": row.get("oi_direction", ""),
                    }
        except Exception:
            pass
    return signals


def run_select(cfg: Dict, capital: float, base_dir: Path, out_dir: Path,
               as_of: str, no_schwab: bool = False) -> List[WheelCandidate]:
    """Run wheel selection pipeline."""
    print(f"=== Wheel Selection Pipeline ({as_of}) ===", file=sys.stderr)
    print(f"  Capital: ${capital:,.0f}", file=sys.stderr)

    # Stage 0: Load and filter universe
    print("  Stage 0: Universe filter...", file=sys.stderr)
    screener = _load_screener(base_dir)
    if screener.empty:
        print("  [error] No screener data. Aborting.", file=sys.stderr)
        return []
    universe = filter_universe(screener, cfg)
    print(f"    {len(universe)} tickers pass universe filter", file=sys.stderr)

    # Stage 1: Quality scoring
    print("  Stage 1: Quality scoring (yfinance + Schwab)...", file=sys.stderr)
    schwab_svc = None
    quotes_payload = {}
    if not no_schwab:
        try:
            from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
            auth = SchwabAuthConfig.from_env()
            schwab_svc = SchwabLiveDataService(auth)
            schwab_svc.connect()
            tickers = list(universe["ticker"])
            quotes_payload = schwab_svc.get_quotes(tickers) or {}
            print(f"    Schwab quotes fetched for {len(quotes_payload)} tickers",
                  file=sys.stderr)
        except Exception as exc:
            print(f"    [warn] Schwab fetch failed: {exc}", file=sys.stderr)

    candidates = []
    for _, row in universe.iterrows():
        ticker = row["ticker"]
        print(f"    Scoring {ticker}...", file=sys.stderr)
        schwab_quote = quotes_payload.get(ticker, {})
        fundamentals = fetch_fundamentals(ticker, schwab_quote)

        quality = score_quality(fundamentals, cfg)
        if quality.disqualified:
            print(f"      {ticker} disqualified: {quality.disqualify_reason}",
                  file=sys.stderr)
            continue

        # Stage 2: Premium scoring
        chain_data = {"csp_strike": 0, "csp_premium": 0, "cc_strike": 0,
                      "cc_premium": 0, "spot": row.get("close", 0),
                      "iv_rank": 0, "spread_pct": 5, "dte": cfg.get(
                          "management", {}).get("dte_target", 30)}

        if schwab_svc and not no_schwab:
            try:
                chain = schwab_svc.get_option_chain(ticker, strike_count=10)
                if chain:
                    iv = row.get("iv30d", 0.30) if "iv30d" in row else 0.30
                    chain_data = extract_chain_data(
                        chain, spot=row.get("close", 0), iv=iv,
                        dte_target=cfg.get("management", {}).get("dte_target", 30),
                        sigma=cfg.get("management", {}).get("sigma_otm", 1.0))
                    chain_data["iv_rank"] = row.get("iv_rank", 0) if "iv_rank" in row.index else 0
            except Exception as exc:
                print(f"      [warn] Chain fetch failed for {ticker}: {exc}",
                      file=sys.stderr)

        premium = score_premium(chain_data, cfg)

        # Stage 3: Sentiment overlay
        swing_signals = _load_swing_trend_signals(
            Path(cfg.get("pipeline", {}).get("root_dir", r"c:\uw_root")) / "out",
            as_of)
        swing = swing_signals.get(ticker, {})
        sentiment = apply_sentiment(
            swing_direction=swing.get("direction", ""),
            swing_verdict=swing.get("verdict", ""),
            whale_score=swing.get("whale_score", 0),
            dp_bearish=False,
            earnings_days=None,
            oi_confirms=swing.get("oi_direction", "") == "bullish",
            cfg=cfg)

        # Composite
        scfg = cfg.get("scoring", {})
        composite_raw = compute_composite(
            quality.composite, premium.composite, 0,
            scfg.get("quality_weight", 0.7), scfg.get("premium_weight", 0.3))
        composite = compute_composite(
            quality.composite, premium.composite, sentiment.total,
            scfg.get("quality_weight", 0.7), scfg.get("premium_weight", 0.3))
        tier = assign_tier(composite, cfg)

        if tier == "excluded":
            continue

        # Target expiry
        target_dte = cfg.get("management", {}).get("dte_target", 30)
        expiry_date = dt.date.fromisoformat(as_of) + dt.timedelta(days=target_dte)
        # Snap to next Friday
        days_to_fri = (4 - expiry_date.weekday()) % 7
        expiry_date += dt.timedelta(days=days_to_fri)

        c = WheelCandidate(
            ticker=ticker, spot=row.get("close", 0),
            sector=fundamentals.get("sector", ""),
            market_cap_b=fundamentals.get("market_cap", 0) / 1e9,
            quality=quality, premium=premium, sentiment=sentiment,
            composite_raw=composite_raw, composite=composite, tier=tier,
            expiry=expiry_date.isoformat(), dte=chain_data.get("dte", target_dte),
            action=f"Sell ${chain_data.get('csp_strike', 0):.0f}P {expiry_date}",
            notes="; ".join(sentiment.notes) if sentiment.notes else "",
        )
        candidates.append(c)

    # Allocate capital
    print("  Allocating capital...", file=sys.stderr)
    candidates = allocate_capital(candidates, capital, cfg)
    # Re-add watchlist candidates (those not allocated)
    print(f"  {len(candidates)} candidates selected", file=sys.stderr)

    # Write output
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = generate_select_report(candidates, capital, as_of, cfg)
    report_path = out_dir / f"wheel-select-{as_of}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  Wrote: {report_path}", file=sys.stderr)

    return candidates


def run_daily(cfg: Dict, out_dir: Path, as_of: str,
              no_schwab: bool = False) -> List[DailyAction]:
    """Run daily wheel position management."""
    print(f"=== Wheel Daily Manager ({as_of}) ===", file=sys.stderr)

    pos_path = out_dir / cfg.get("pipeline", {}).get("positions_file",
                                                      "wheel_positions.json")
    tracker = PositionTracker(pos_path)
    if not tracker.positions:
        print("  No active positions. Run 'select' mode first.", file=sys.stderr)
        return []

    capital = cfg.get("pipeline", {}).get("default_capital", 35000)
    mgr = DailyManager(cfg)
    actions = []

    # Fetch live quotes if Schwab available
    quotes = {}
    if not no_schwab:
        try:
            from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
            auth = SchwabAuthConfig.from_env()
            svc = SchwabLiveDataService(auth)
            svc.connect()
            tickers = [p.ticker for p in tracker.positions]
            quotes = svc.get_quotes(tickers) or {}
        except Exception as exc:
            print(f"  [warn] Schwab quotes failed: {exc}", file=sys.stderr)

    for pos in tracker.positions:
        q = quotes.get(pos.ticker, {})
        spot = q.get("quote", {}).get("mark", 0) or q.get("quote", {}).get("lastPrice", 0)

        if pos.phase == "csp":
            # Estimate current premium (simplified — ideally from chain)
            current = pos.entry_premium * 0.5  # placeholder
            dte_remaining = 30  # placeholder
            if pos.expiry:
                try:
                    exp = dt.date.fromisoformat(pos.expiry)
                    dte_remaining = (exp - dt.date.fromisoformat(as_of)).days
                except ValueError:
                    pass
            action = mgr.evaluate_csp(pos, current, dte_remaining)
            actions.append(action)

        elif pos.phase == "shares":
            action = mgr.evaluate_shares(pos, spot)
            actions.append(action)

        elif pos.phase == "cc":
            current = pos.entry_premium * 0.5  # placeholder
            dte_remaining = 30
            if pos.expiry:
                try:
                    exp = dt.date.fromisoformat(pos.expiry)
                    dte_remaining = (exp - dt.date.fromisoformat(as_of)).days
                except ValueError:
                    pass
            action = mgr.evaluate_cc(pos, current, dte_remaining, spot)
            actions.append(action)

    # Write daily report
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = generate_daily_report(actions, tracker.positions, capital,
                                      as_of, tracker.journal)
    report_path = out_dir / f"wheel-daily-{as_of}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"  Wrote: {report_path}", file=sys.stderr)

    return actions


def main():
    ap = argparse.ArgumentParser(description="Wheel trading pipeline")
    ap.add_argument("--mode", choices=["select", "daily", "full"], default="full",
                    help="Pipeline mode: select (weekly), daily (manage), full (both)")
    ap.add_argument("--capital", type=float, default=None,
                    help="Total capital for wheel allocation")
    ap.add_argument("--config", default=str(Path(__file__).parent / "wheel_config.yaml"),
                    help="Config YAML path")
    ap.add_argument("--base-dir", default=None,
                    help="Base data directory (dated folder)")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory")
    ap.add_argument("--as-of", default=None,
                    help="As-of date YYYY-MM-DD (default: today)")
    ap.add_argument("--no-schwab", action="store_true",
                    help="Skip Schwab live validation")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    as_of = args.as_of or dt.date.today().isoformat()
    capital = args.capital or cfg.get("pipeline", {}).get("default_capital", 35000)
    root = Path(cfg.get("pipeline", {}).get("root_dir", r"c:\uw_root"))
    base_dir = Path(args.base_dir) if args.base_dir else root / as_of
    out_dir = Path(args.out_dir) if args.out_dir else \
        Path(cfg.get("pipeline", {}).get("output_dir", r"c:\uw_root\out\wheel"))

    if args.mode in ("select", "full"):
        run_select(cfg, capital, base_dir, out_dir, as_of, args.no_schwab)

    if args.mode in ("daily", "full"):
        run_daily(cfg, out_dir, as_of, args.no_schwab)


if __name__ == "__main__":
    main()
```

**Step 2: Run tests and smoke test**

Run: `cd /c/uw_root && python -m pytest tests/test_wheel_pipeline.py -v`
Run: `cd /c/uw_root && python -m uwos.wheel_pipeline --help`

**Step 3: Commit**

```bash
git add uwos/wheel_pipeline.py
git commit -m "feat(wheel): add pipeline orchestration and CLI entry point"
```

---

## Task 11: Claude Code Skill

**Files:**
- Create: `.claude/skills/wheel.md`

**Step 1: Create the skill file**

```markdown
---
name: wheel
description: Run the wheel trading pipeline (candidate selection + daily position management). Use when user asks to run wheel analysis, wheel trades, or manage wheel positions.
---

## Wheel Trading Pipeline

Run the signal-driven wheel strategy pipeline.

### Usage
- `/wheel` — Full run (select + daily) with defaults
- `/wheel select` — Weekly candidate selection only
- `/wheel daily` — Daily position management only
- `/wheel select 50000` — Selection with custom capital

### Execution

1. Parse user args. Default: mode=full, capital=35000, as-of=today
2. Determine base-dir from as-of date: `c:/uw_root/{as-of}/`
3. Run the pipeline:

```bash
python -m uwos.wheel_pipeline --mode {mode} --capital {capital} --base-dir "c:/uw_root/{as_of}" --out-dir "c:/uw_root/out/wheel" --as-of {as_of}
```

4. Read the output files and present to user:
   - Selection: `c:/uw_root/out/wheel/wheel-select-{date}.md`
   - Daily: `c:/uw_root/out/wheel/wheel-daily-{date}.md`

5. Add market context commentary interpreting the results

6. **ALWAYS** provide clickable file links:
   - Report: `c:\uw_root\out\wheel\wheel-select-{date}.md`
   - Daily: `c:\uw_root\out\wheel\wheel-daily-{date}.md`
   - Positions: `c:\uw_root\out\wheel\wheel_positions.json`

### Design Doc
See `docs/plans/2026-03-07-wheel-pipeline-design.md` for full scoring model, decision matrix, and risk controls.
```

**Step 2: Commit**

```bash
git add .claude/skills/wheel.md
git commit -m "feat(wheel): add /wheel Claude Code skill"
```

---

## Task 12: Integration Test — End-to-End Smoke Run

**Step 1: Run selection with --no-schwab on today's data**

```bash
cd /c/uw_root && python -m uwos.wheel_pipeline --mode select --capital 35000 --base-dir "c:/uw_root/2026-03-06" --out-dir "c:/uw_root/out/wheel" --as-of 2026-03-07 --no-schwab
```

Expected: Generates `out/wheel/wheel-select-2026-03-07.md` with ranked candidates

**Step 2: Verify output file exists and contains expected sections**

Check for: Capital line, CORE/AGGRESSIVE/WATCHLIST tables, Capital Allocation table

**Step 3: Run with Schwab live**

```bash
cd /c/uw_root && python -m uwos.wheel_pipeline --mode select --capital 35000 --base-dir "c:/uw_root/2026-03-06" --out-dir "c:/uw_root/out/wheel" --as-of 2026-03-07
```

**Step 4: Fix any issues found in smoke test**

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(wheel): complete wheel pipeline v1 with selection, daily management, and skill"
```

---

## Summary of All Files

| File | Action | Task |
|---|---|---|
| `uwos/wheel_config.yaml` | Create | 1 |
| `uwos/wheel_pipeline.py` | Create + iterate | 1-10 |
| `tests/test_wheel_pipeline.py` | Create + iterate | 1-8 |
| `.claude/skills/wheel.md` | Create | 11 |
| `docs/plans/2026-03-07-wheel-pipeline-design.md` | Already exists | Reference |
| `docs/plans/2026-03-07-wheel-pipeline-impl.md` | This file | Reference |

## Dependencies (all existing)

- `schwab-py>=1.5.0` (uwos/schwab_auth.py)
- `yfinance` (uwos/setup_likelihood_backtest.py)
- `pandas`, `numpy`, `PyYAML` (all existing)
