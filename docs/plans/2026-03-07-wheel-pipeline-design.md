# Wheel Pipeline Design Document

**Date:** 2026-03-07
**Status:** Approved
**Author:** Claude Opus 4.6 + Anu

---

## 1. Overview

A signal-driven wheel trading system for a $25-50K account. Two components:

1. **Wheel Selection Pipeline** (`--mode select`) — Weekly scoring of wheel candidates using ownership quality (70%) + premium yield (30%)
2. **Wheel Daily Manager** (`--mode daily`) — Daily position management with action recommendations

Invoked via Claude Code skill: `/wheel`

### Design Principles

- **Ownership-first (70/30):** Filter hard on quality, rank survivors by premium yield
- **Hybrid book:** Core stable names (60%) + aggressive premium names (40%)
- **Daily monitoring:** Active management with 50% profit targets, rolling, and regime-aware strike selection
- **Pipeline-integrated:** Uses existing UW daily data files + swing trend signals as sentiment overlay
- **Deterministic scoring:** Python module for reproducible scores; Claude skill for interpretation layer

---

## 2. Architecture

```
/wheel (Claude Skill)
  |
  +-- python -m uwos.wheel_pipeline --mode select --capital 35000
  |     |
  |     +-- Stage 0: Universe Filter (price, liquidity, market cap)
  |     +-- Stage 1: Ownership Quality Score (70%)
  |     |     +-- Schwab API: P/E, EPS, divYield, 52wk range
  |     |     +-- yfinance: FCF yield, D/E, ROE, margins, rev growth, earnings beats
  |     |     +-- yfinance: Mean reversion score (drawdown recovery analysis)
  |     |
  |     +-- Stage 2: Premium Yield Score (30%)
  |     |     +-- Schwab chains: CSP yield, CC yield, IV rank, spread quality
  |     |
  |     +-- Stage 3: Sentiment Overlay (boost/penalize)
  |     |     +-- UW daily files: whale, dark pool, OI changes
  |     |     +-- Swing trend signal (if available for ticker)
  |     |     +-- Earnings proximity warning
  |     |
  |     +-- Output: Ranked candidates with capital allocation
  |
  +-- python -m uwos.wheel_pipeline --mode daily
  |     |
  |     +-- Read wheel_positions.json
  |     +-- Fetch live quotes from Schwab
  |     +-- Apply decision matrix per position
  |     +-- Output: Action recommendations + premium journal
  |
  +-- Claude interprets output, adds market context, returns .md links
```

### Data Sources

| Data Layer | Source | Access Method |
|---|---|---|
| Fundamentals (P/E, EPS, divYield) | Schwab API `get_quotes` | Programmatic (existing) |
| Deep Fundamentals (FCF, D/E, ROE, margins, rev growth) | yfinance | Programmatic (existing) |
| Options chains (IV, Greeks, bid/ask, OI) | Schwab API `get_option_chain` | Programmatic (existing) |
| Price history (mean reversion calc) | yfinance | Programmatic (existing) |
| Flow/sentiment (whale, dark pool, OI) | UW dashboard CSV exports | Already ingested daily |
| Swing trend signals | uwos.swing_trend_pipeline output | Read from out/swing_trend/ |

### Future Data Enhancements (not in v1)

| Endpoint | Status | Potential Use |
|---|---|---|
| Schwab `get_instruments` | Not yet used | Sector search, dynamic universe building |
| Schwab `get_movers` | Not yet used | Real-time high-IV candidate discovery |
| Schwab `get_price_history` | Not yet used | Replace yfinance for consistency |
| Schwab `get_account` | Not yet used | Auto-read buying power for position sizing |

---

## 3. Universe Filter (Stage 0)

Hard filters applied before scoring. Removes stocks that cannot be wheeled at this capital level.

| Filter | Threshold | Rationale |
|---|---|---|
| Price range | $10 - $60 | CSP-affordable at $25-50K capital |
| Options avg daily volume | > 500 contracts | Liquidity for clean execution |
| Option bid/ask spread | < 5% of mid | Execution cost control |
| Market cap | > $2B | No penny stocks / micro-caps |
| Options chain available | Must exist on Schwab | Required for wheel mechanics |

**Universe source:** Start from the existing UW stock-screener CSV (~200 tickers), apply filters. This keeps the universe aligned with the tickers the daily pipeline already tracks.

---

## 4. Ownership Quality Score (Stage 1) — 70% of Composite

Scored 0-100. Measures "would I hold this through a 20% drawdown?"

| Sub-Score | Weight | Source | Scoring |
|---|---|---|---|
| Profitability (ROE) | 25% | yfinance | >15% = 100, 10-15 = 75, 5-10 = 50, <5 = 25, negative = 0 |
| Balance Sheet (D/E) | 15% | yfinance | <0.5 = 100, 0.5-1.0 = 75, 1.0-2.0 = 50, >2.0 = 25 |
| Growth (Rev YoY) | 17.5% | yfinance | >15% = 100, 5-15 = 75, 0-5 = 50, negative = 25 |
| Cash Flow (FCF yield) | 17.5% | yfinance | >5% = 100, 3-5 = 75, 1-3 = 50, <1 = 25 |
| Valuation (P/E) | 5% | Schwab | <15 = 100, 15-25 = 75, 25-40 = 50, >40 = 25, negative = 10 |
| Stability (earnings beats) | 10% | yfinance | 4/4 = 100, 3/4 = 75, 2/4 = 50, <2 = 25 |
| Mean Reversion | 10% | yfinance price history | Recovery rate of 10%+ drawdowns within 30 trading days over 2yr. >75% = 100, 50-75 = 75, 25-50 = 50, <25 = 25 |

**Hard disqualifiers (score = 0, excluded):**
- Negative FCF for 2+ consecutive years
- Debt/Equity > 3.0
- Fails Stage 0 filters

---

## 5. Premium Yield Score (Stage 2) — 30% of Composite

Scored 0-100. Measures "how much premium can I harvest?"

| Sub-Score | Weight | Source | Scoring |
|---|---|---|---|
| CSP Yield (annualized) | 35% | Schwab chain | >40% = 100, 30-40 = 85, 20-30 = 70, 10-20 = 50, <10 = 25 |
| CC Yield (annualized) | 25% | Schwab chain | >30% = 100, 20-30 = 85, 10-20 = 70, 5-10 = 50, <5 = 25 |
| IV Rank | 20% | Schwab + yfinance | >60% = 100, 40-60 = 75, 20-40 = 50, <20 = 25 |
| Spread Quality | 20% | Schwab chain | <2% of mid = 100, 2-4 = 75, 4-6 = 50, >6 = 25 |

**Yield calculation:**
```
CSP Yield (ann.) = (put_premium / strike) * (365 / DTE) * 100
CC Yield (ann.)  = (call_premium / spot) * (365 / DTE) * 100
```

Strike selection for scoring: 1-sigma OTM (1 standard deviation from spot using current IV).
```
CSP strike = spot * (1 - IV * sqrt(DTE/365))
CC strike  = spot * (1 + IV * sqrt(DTE/365))
```

---

## 6. Sentiment Overlay (Stage 3)

Not a separate score. Applies +/- adjustments to the composite score.

| Signal | Source | Adjustment |
|---|---|---|
| Swing trend PASS + Bullish | swing_trend output | +5 points |
| Swing trend FAIL or Bearish | swing_trend output | -5 points |
| Strong whale accumulation (>70 whale score) | UW whale file | +3 points |
| Dark pool bearish divergence | UW dp-eod-report | -3 points |
| Earnings within 14 days | yfinance earnings calendar | -5 points + warning flag |
| OI buildup confirms direction | UW chain-oi-changes | +2 points |

Max adjustment: +/- 10 points. Sentiment shifts ranking order but cannot promote a low-quality name into the core book.

---

## 7. Capital Allocation Rules

| Rule | Threshold |
|---|---|
| Max capital deployed | 65% of total (35% cash reserve) |
| Max single-name exposure | 25% of total capital |
| Max concurrent positions | 5 |
| Core/Aggressive split | 60% / 40% of deployed capital |
| Min composite score for Core | 60 |
| Min composite score for Aggressive | 45 |
| Min composite score for Watchlist | 35 |

**Allocation algorithm:**
1. Rank candidates by composite score descending
2. Assign top candidates to Core until 60% of deployed capital filled
3. Assign next candidates to Aggressive until 40% filled
4. Remaining candidates go to Watchlist
5. Never exceed max single-name or total deployment limits
6. Report unused capital as cash reserve

---

## 8. Daily Management Decision Matrix

The daily manager reads `wheel_positions.json` and applies these rules:

### CSP Phase (Short Put Open)

| Condition | Action |
|---|---|
| P/L >= 50% of max profit | CLOSE — buy back, re-enter if signal still bullish |
| DTE <= 14 and P/L > 0 | CLOSE — avoid gamma risk near expiry |
| DTE <= 14 and P/L < 0 | ROLL — same strike, +30 DTE, collect credit |
| Swing trend flips bearish | ROLL DOWN — lower strike, same/later expiry |
| Approaching assignment (ITM, DTE < 7) | ALLOW ASSIGNMENT — if quality score > 55. Otherwise ROLL DOWN. |
| Earnings within 7 days | CLOSE or ROLL past earnings |

### Shares Phase (Assigned, Holding Stock)

| Condition | Action |
|---|---|
| Swing trend bullish | SELL CC — aggressive strike (0.5 sigma OTM), shorter DTE |
| Swing trend neutral | SELL CC — 1 sigma OTM, 30 DTE |
| Swing trend bearish | SELL CC — ATM or slight ITM to accelerate exit |
| Stock drops >10% from assignment | SELL CC at cost basis, flag for review |
| Stock gaps up >10% | Consider selling shares if quality score < 50 |

### CC Phase (Covered Call Open)

| Condition | Action |
|---|---|
| P/L >= 50% of max profit | CLOSE — re-sell higher/later CC |
| DTE <= 14 and OTM | LET EXPIRE or close for pennies |
| Stock rallied through strike | ALLOW CALL AWAY — restart wheel with new CSP |
| Stock pulled back, CC near worthless | CLOSE early, re-sell at lower strike for more premium |

### 50% Profit Target Rationale

TJ rolls and collects continuously. At our capital level, closing at 50% and re-entering:
- Frees capital faster for new premium cycles
- Reduces time exposure to adverse moves
- Compounds faster: two 50% wins > one 100% win in same timeframe

---

## 9. Position Tracker Schema

File: `out/wheel/wheel_positions.json`

```json
{
  "last_updated": "2026-03-07T16:30:00",
  "capital_total": 35000,
  "positions": [
    {
      "ticker": "BP",
      "tier": "core",
      "phase": "csp",
      "entry_date": "2026-03-03",
      "strike": 38.00,
      "expiry": "2026-04-17",
      "contracts": 2,
      "entry_premium": 0.85,
      "current_premium": 0.40,
      "capital_reserved": 7600,
      "cumulative_premium": 170.00,
      "assignment_count": 0,
      "wheel_cycles_completed": 0
    }
  ],
  "premium_journal": [
    {
      "date": "2026-03-07",
      "ticker": "BP",
      "action": "close_csp",
      "premium_realized": 90.00,
      "notes": "Closed at 53% profit"
    }
  ]
}
```

---

## 10. Output Files

| File | Frequency | Content |
|---|---|---|
| `out/wheel/wheel-select-YYYY-MM-DD.md` | Weekly (or on-demand) | Ranked candidates, scores, capital allocation |
| `out/wheel/wheel-daily-YYYY-MM-DD.md` | Daily | Position status, actions, premium journal, risk dashboard |
| `out/wheel/wheel_positions.json` | Persistent (updated daily) | Position tracker |
| `out/wheel/wheel_premium_journal.csv` | Append-only | Cumulative premium log |

---

## 11. Skill Invocation

**Skill name:** `wheel`

**Usage:**
```
/wheel                    # Full run: select + daily, defaults
/wheel select             # Weekly candidate selection only
/wheel daily              # Daily position management only
/wheel select 50000       # Selection with $50K capital
```

**Under the hood:**
```bash
# Selection mode
python -m uwos.wheel_pipeline --mode select --capital 35000 --out-dir "c:/uw_root/out/wheel"

# Daily mode
python -m uwos.wheel_pipeline --mode daily --out-dir "c:/uw_root/out/wheel"
```

**Config file:** `uwos/wheel_config.yaml`
```yaml
capital: 35000
max_positions: 5
max_single_name_pct: 0.25
cash_reserve_pct: 0.35
core_aggressive_split: [0.60, 0.40]
close_target_pct: 0.50
dte_target: 30
dte_roll_threshold: 14
min_composite_core: 60
min_composite_aggressive: 45
min_composite_watchlist: 35
sigma_otm: 1.0
positions_file: "out/wheel/wheel_positions.json"
out_dir: "out/wheel"
```

---

## 12. Output Format

### Selection Report Color Coding

| Color | Tier | Meaning |
|---|---|---|
| GREEN | Core | Ownership-quality passes, stable wheel candidate |
| YELLOW | Tactical | Balanced quality + premium, moderate risk |
| RED | Aggressive | Premium-driven, lower quality, higher risk |
| WHITE | Watchlist | Does not qualify yet, monitor for improvement |

### Daily Report Action Icons

| Icon | Action |
|---|---|
| CLOSE | Hit profit target or risk trigger — buy back |
| HOLD | On track, no action needed |
| ROLL | Roll to new strike/expiry |
| SELL CC | Assigned shares — initiate covered call |
| ASSIGNED | Put exercised, shares acquired |
| CALLED AWAY | Call exercised, shares sold — restart wheel |
| NEW CSP | Enter new cash-secured put |
| WARNING | Risk flag (earnings, signal flip, concentration) |

---

## 13. Risk Controls

| Control | Limit | Action if Breached |
|---|---|---|
| Total deployment | 65% of capital | Block new entries |
| Single-name concentration | 25% of capital | Block additional contracts |
| Sector concentration | 40% of capital | Warn, suggest diversification |
| Max unrealized loss per position | -50% of premium received | Force review, consider closing |
| Earnings within 7 days | Flag | Close or roll past earnings |
| Swing trend flip to bearish | Flag | Roll CSP down or close |
| Consecutive assignment on same name | 2x | Pause wheeling that name, review thesis |

---

## 14. Implementation Plan

### Module: `uwos/wheel_pipeline.py`

| Component | Description |
|---|---|
| `WheelSelector` | Stage 0-3 scoring engine |
| `WheelDailyManager` | Position tracker + decision matrix |
| `QualityScorer` | Ownership quality sub-scores (yfinance + Schwab) |
| `PremiumScorer` | Premium yield sub-scores (Schwab chains) |
| `SentimentOverlay` | Boost/penalize from UW files + swing trend |
| `CapitalAllocator` | Fits candidates to capital budget |
| `PositionTracker` | Read/write wheel_positions.json |
| `PremiumJournal` | Append-only premium log |
| `WheelReportWriter` | Markdown output generation |

### Dependencies (all existing)

- `uwos.schwab_auth.SchwabLiveDataService` — Schwab API
- `yfinance` — deep fundamentals + price history
- `uwos.swing_trend_pipeline` — swing trend signal reads
- Existing UW CSV files in daily data directories

### Skill: `.claude/skills/wheel.md`

Claude Code skill that:
1. Parses user args (mode, capital)
2. Invokes `python -m uwos.wheel_pipeline`
3. Reads output .md files
4. Adds market context and interpretation
5. Returns clickable file links

---

## 15. Appendix: TJ Framework Mapping

How this design maps to TJ's strategies at $25-50K scale:

| TJ Concept | Our Implementation |
|---|---|
| "Rich Man's Covered Call" (LEAP put + short calls) | Standard wheel (CSP + CC cycle) — LEAPs need too much margin at our size |
| "Premium Cash Grab" — sell with no intent to own | Aggressive tier: high-IV names, tight roll rules |
| "This or That" — inverse correlation hedge | Sector diversification + max concentration limits |
| Cross-funding positions with premium | Premium journal tracks cumulative income; can fund new CSP entries |
| Rolling down call strikes in drawdowns | Daily manager: swing trend bearish -> roll CC to lower strike |
| Covered calls on drawdown positions | Shares phase: sell CC at cost basis to generate income while waiting |
| Strike selection = conviction level | Quality score determines strike distance: higher quality = wider strikes (more conviction) |
| Multi-timeframe layering | Selection (weekly) + management (daily) + signals (swing trend L5/L30) |
