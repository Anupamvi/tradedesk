# Trade Analysis Agent Design Document

**Date:** 2026-03-07
**Status:** Approved
**Author:** Claude Opus 4.6 + Anu

---

## 1. Overview

An intelligent trade analysis agent invoked via `/tradehistory` that fetches open positions from Schwab, enriches them with live market data, quantitative risk metrics, news, macro events, and X sentiment, then provides per-position verdicts (HOLD / CLOSE / ROLL / other) biased toward patience and profit maximization.

### Core Philosophy

> **"Close when the thesis breaks or the math inverts, not at an arbitrary profit target."**

The agent does not rush to close at 50%. For each position it answers: "Is the expected value of holding for another week positive?" If yes → HOLD. If no → what's the best exit?

### Design Principles

- **Patience-biased:** Default is HOLD unless a concrete reason exists to act
- **Data-driven verdicts:** Every recommendation backed by quantitative metrics + qualitative research
- **Python for data, Claude for reasoning:** Python module collects/computes; Claude skill interprets and recommends
- **Open positions only:** Full deep analysis for positions you can act on; closed trades get a P&L summary

---

## 2. Architecture

```
/tradehistory [days] [symbol]
  |
  +-- Phase 1: Data Collection (Python module)
  |     |
  |     +-- Schwab get_account(positions) --> current open positions + unrealized P&L
  |     +-- Schwab get_trade_history() --> entry details (date, price, premium)
  |     +-- Schwab get_quotes() --> live underlying prices
  |     +-- Schwab get_option_chain() --> live Greeks, IV, bid/ask for option positions
  |     +-- yfinance --> earnings date, IV history (for IV rank), realized vol, sector,
  |     |               support/resistance levels, SPY correlation
  |     +-- Computed risk metrics (see Section 4)
  |     +-- Output: position_data_{date}.json
  |
  +-- Phase 2: Research (Claude skill, parallel agents)
  |     |
  |     +-- Per-ticker WebSearch --> recent news, earnings dates, analyst actions
  |     +-- Per-ticker WebSearch --> X/Twitter sentiment (best-effort)
  |     +-- Macro WebSearch --> Fed, CPI, jobs, VIX, market regime
  |     +-- Output: research context fed into analysis
  |
  +-- Phase 3: Analysis & Recommendation (Claude skill)
        |
        +-- Per-position: verdict card with metrics + research + recommendation
        +-- Portfolio-level: concentration, correlation, macro alignment
        +-- Output: trade-analysis-{date}.md + clickable links
```

### Data Sources

| Data Layer | Source | Access Method |
|---|---|---|
| Current positions | Schwab `get_account` | New API method (to add) |
| Trade history | Schwab `get_transactions` | Existing (just built) |
| Live quotes | Schwab `get_quotes` | Existing |
| Option chains + Greeks | Schwab `get_option_chain` | Existing |
| Earnings dates | yfinance | Programmatic |
| IV history (IV rank) | yfinance 1yr price history | Programmatic |
| Realized volatility | yfinance 20-day HV | Programmatic |
| Support/resistance | yfinance price history (MAs, recent highs/lows) | Programmatic |
| SPY correlation | yfinance 20-day correlation | Programmatic |
| Sector | yfinance `.info["sector"]` | Programmatic |
| News | WebSearch | Claude skill |
| X sentiment | WebSearch (best-effort) | Claude skill |
| Macro events | WebSearch | Claude skill |

---

## 3. Phase 1 — Python Data Collection Module

**Module:** `uwos/schwab_position_analyzer.py`

### New Schwab API method needed

Add `get_account_positions()` to `SchwabLiveDataService` using `client.get_account(account_hash, fields=['positions'])`.

### Data collected per open position

| Data | Source | Fields |
|------|--------|--------|
| Current positions | `get_account(positions)` | symbol, qty, market value, avg cost, unrealized P&L, asset type |
| Entry details | `get_trade_history()` match | entry date, entry price, original premium |
| Live underlying quote | `get_quotes()` | last, bid, ask, change%, volume |
| Live option chain | `get_option_chain()` | current bid/ask/mark, delta, gamma, theta, vega, IV, OI |
| Earnings date | yfinance | next earnings date, days until earnings |
| IV rank | yfinance 1yr IV history vs current Schwab IV | percentile rank |
| Historical vol | yfinance 20-day realized vol | HV for IV vs HV comparison |
| Support/resistance | yfinance price history | 200-day MA, 50-day MA, recent swing high/low |
| SPY correlation | yfinance 20-day returns correlation | portfolio correlation risk |
| Sector | yfinance | sector classification for concentration check |

### Output schema: `out/trade_analysis/position_data_{date}.json`

```json
{
  "as_of": "2026-03-07T16:30:00",
  "account_summary": {
    "total_value": 45000,
    "cash": 15000,
    "positions_value": 30000
  },
  "positions": [
    {
      "symbol": "AAPL 260417P00200000",
      "underlying": "AAPL",
      "asset_type": "OPTION",
      "put_call": "PUT",
      "strike": 200.0,
      "expiry": "2026-04-17",
      "dte": 41,
      "qty": -2,
      "avg_cost": 3.50,
      "market_value": -480.00,
      "unrealized_pnl": 220.00,
      "unrealized_pnl_pct": 31.4,
      "entry_date": "2026-02-20",
      "live_quote": { "bid": 2.30, "ask": 2.50, "mark": 2.40, "last": 2.40 },
      "greeks": { "delta": -0.25, "gamma": 0.02, "theta": -0.05, "vega": 0.12, "iv": 0.32 },
      "underlying_quote": { "last": 218.50, "change_pct": -0.8 },
      "computed": {
        "theta_pnl_per_day": 12.00,
        "gamma_risk": 0.02,
        "vega_exposure": 24.00,
        "breakeven": 196.50,
        "distance_to_breakeven_pct": 10.1,
        "prob_itm": 0.25,
        "prob_profit": 0.75,
        "max_profit": 700.00,
        "max_loss": 39300.00,
        "risk_reward_ratio": 56.1,
        "theta_risk_ratio": 0.0003,
        "pct_of_max_profit": 31.4,
        "days_held": 15,
        "iv_rank": 62,
        "iv_percentile": 65,
        "iv_vs_hv_spread": 7.0,
        "hv_20d": 0.25,
        "bid_ask_spread_pct": 2.1,
        "open_interest_at_strike": 3500,
        "volume_at_strike": 450,
        "spy_correlation_20d": 0.72,
        "sector": "Technology",
        "support_levels": [210.0, 205.0],
        "resistance_levels": [225.0, 230.0],
        "ma_50d": 215.0,
        "ma_200d": 210.0,
        "earnings_date": "2026-04-24",
        "days_to_earnings": 48
      }
    }
  ]
}
```

---

## 4. Computed Risk Metrics

All calculated by the Python module from Schwab + yfinance data:

| Metric | Calculation | Why it matters |
|--------|-------------|----------------|
| Theta decay $/day | `theta * qty * 100` | Daily time decay P&L |
| Gamma risk | `gamma * qty * 100` | Delta acceleration on $1 move |
| Vega exposure $ | `vega * qty * 100` | P&L per 1% IV change |
| Breakeven price | Strike +/- entry premium | Where the underlying kills the trade |
| Distance to breakeven % | `(underlying - breakeven) / underlying` | Buffer before trouble |
| Probability ITM | `abs(delta)` | Chance option expires ITM |
| Probability of profit | Credit: `1 - abs(delta)`, Debit: `abs(delta)` | Chance trade is profitable at expiry |
| Max profit | Entry premium * qty * 100 (credits) | Best case |
| Max loss | Strike * qty * 100 (naked), width * qty * 100 (spreads) | Worst case |
| Risk/reward ratio | `max_loss / max_profit` | Payoff skew |
| Theta/risk ratio | `daily_theta_pnl / max_loss` | Earning rate relative to risk |
| % of max profit reached | `unrealized_pnl / max_profit` | How close to target |
| Days held | `today - entry_date` | Time in trade |
| IV Rank | Current IV percentile vs 1yr range | Is premium rich or cheap? |
| IV vs HV spread | `schwab_iv - yfinance_hv_20d` | Overpriced vs underpriced vol |
| Bid/ask spread % | `(ask - bid) / mark * 100` | Exit execution cost |
| OI + Volume at strike | Schwab chain | Exit liquidity |
| SPY correlation (20d) | yfinance returns correlation | Portfolio correlation risk |
| Support/resistance | 50-day MA, 200-day MA, swing high/low | Key technical levels |
| Earnings proximity | yfinance earnings calendar | Binary event risk |

---

## 5. Phase 2 — Research (Claude Skill)

Performed by Claude using WebSearch, parallelized per unique underlying ticker:

### Per-ticker research
- **News:** "AAPL stock news last 7 days" — recent headlines, analyst upgrades/downgrades, SEC filings
- **X Sentiment:** "AAPL stock Twitter/X sentiment" — bullish/bearish themes, retail sentiment

### Macro research (once per run)
- **Fed/rates:** upcoming FOMC, rate expectations
- **Economic data:** CPI, jobs, GDP — anything in the next 2 weeks
- **VIX/market regime:** current VIX level, trend, risk-on vs risk-off

---

## 6. Phase 3 — Verdict Framework (Patience-Biased)

### Decision Matrix

| Signal | Verdict | Rationale |
|--------|---------|-----------|
| Theta strong, no catalyst, IV stable/elevated | **HOLD — let it work** | Time is on your side |
| 50%+ profit, high IV rank, strong theta, no earnings | **HOLD — target 65-80%** | Premium still rich, more to harvest |
| 50%+ profit, IV crushed or theta slowing (DTE < 10) | **CLOSE — diminishing returns** | Remaining premium not worth gamma risk |
| Underlying moved against, still OTM, good DTE | **HOLD — time heals** | Don't panic-close when theta still working |
| Underlying moved against, near/ITM, DTE < 14 | **ROLL — extend duration** | Buy time, collect credit if possible |
| Earnings within 7 days, at profit | **CLOSE or ROLL past** | Binary event, protect gains |
| Earnings within 7 days, at loss | **ASSESS** | May not be worth eating loss pre-earnings |
| IV expanding, short vol position | **HOLD if thesis intact** | IV expansion temporary if fundamentals solid |
| News/macro adverse + fundamentals deteriorating | **CLOSE — thesis broken** | Only close when the reason for the trade changes |
| Near max profit (>85%) | **CLOSE — nothing left** | Risk/reward inverted |

### Key principle

For each position, the analysis must answer: **"Is the expected value of holding for another week positive?"**

- If yes → HOLD, with a note on what would change the verdict
- If no → recommend the best exit: CLOSE, ROLL (with specific strike/expiry/credit), or CONVERT

### Roll recommendations must include specifics
- Target expiry and DTE
- Target strike (same, lower, higher)
- Expected credit/debit of the roll
- New probability of profit after roll

---

## 7. Output Format

### Per-position card

```
### AAPL — Short Put $200 | Apr 17 | 41 DTE
**Status:** +$220 (+31.4%) | Prob Profit: 75% | Theta: +$12/day

| Metric | Value | Signal |
|--------|-------|--------|
| P&L | +31.4% of max | On track |
| Delta | -0.25 | Comfortable OTM |
| Theta/day | +$12 | Strong decay |
| Gamma risk | Low (0.02) | Not accelerating |
| IV Rank | 62nd pctl | Premium still rich |
| IV vs HV | IV 32% > HV 25% | Overpriced vol — good for sellers |
| Breakeven | $196.50 (10.1% away) | Wide buffer |
| Earnings | Apr 24 (48 days, after expiry) | No overlap |
| Bid/Ask | 2.1% | Clean exit available |
| Support | $210 (200d MA) | Well above strike |

**News:** [2-3 bullet summary]
**Macro:** [relevant macro context]
**X Sentiment:** [bullish/bearish/mixed + key themes]

**VERDICT: HOLD — target 65-80% profit**
[2-3 sentence reasoning explaining WHY based on the data above]
```

### Portfolio summary (at end of report)

- Total open positions, total unrealized P&L
- Sector concentration breakdown
- SPY correlation risk assessment
- Macro regime alignment
- Overall portfolio health score (1-10)

### Output files

| File | Path | Content |
|------|------|---------|
| Analysis report | `out/trade_analysis/trade-analysis-{date}.md` | Full per-position analysis + portfolio summary |
| Raw position data | `out/trade_analysis/position_data_{date}.json` | All collected data for audit/debugging |

---

## 8. Skill Invocation

**Skill name:** `trade-history`

**Usage:**
```
/tradehistory              # Analyze all open positions (90-day history for entry matching)
/tradehistory 60           # Use 60-day window for entry matching
/tradehistory 90 AAPL      # Analyze only AAPL positions
```

**Under the hood:**
```bash
# Phase 1: Python data collection
python -m uwos.schwab_position_analyzer --days 90 --out-dir "c:/uw_root/out/trade_analysis"

# Phase 2-3: Claude reads JSON, does research, writes analysis
# (handled by the skill itself)
```

---

## 9. Implementation Components

### Python module: `uwos/schwab_position_analyzer.py`

| Component | Description |
|-----------|-------------|
| `get_account_positions()` | New Schwab API method — fetch current positions |
| `match_entry_details()` | Match open positions to trade history for entry date/price |
| `fetch_option_enrichment()` | Get live chain data + Greeks for option positions |
| `compute_risk_metrics()` | Calculate all derived metrics (Section 4) |
| `fetch_yfinance_context()` | Earnings, IV history, HV, support/resistance, correlation, sector |
| `build_position_data()` | Assemble final JSON output |

### Schwab API addition: `schwab_auth.py`

| Method | Description |
|--------|-------------|
| `get_account_positions()` | Call `client.get_account(hash, fields=['positions'])` |

### Skill: `.claude/skills/trade-history/SKILL.md`

Updated skill that:
1. Invokes Python module (Phase 1)
2. Reads position_data JSON
3. Runs parallel WebSearch agents per ticker (Phase 2)
4. Generates per-position verdict cards (Phase 3)
5. Writes trade-analysis-{date}.md
6. Returns clickable file links

### Dependencies (all existing)

- `uwos.schwab_auth.SchwabLiveDataService` — Schwab API
- `yfinance` — fundamentals, price history, earnings
- `WebSearch` — news, X sentiment, macro (Claude tool)

---

## 10. Error Handling

| Error | Action |
|-------|--------|
| Schwab auth failed | Instruct user to re-auth in terminal: `python -m uwos.schwab_position_analyzer --manual-auth` |
| No open positions | Report cleanly — "No open positions found" |
| yfinance data missing for a ticker | Skip that enrichment, note it in output |
| WebSearch fails | Skip research for that ticker, note it |
| Option chain unavailable (e.g., equity position) | Analyze as equity with simpler metrics (P&L, support/resistance, news) |
