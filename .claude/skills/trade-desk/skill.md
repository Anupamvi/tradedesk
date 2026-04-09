# Trade Desk Agent

Full-spectrum trading agent: manage open positions, discover dip-buy opportunities, and run automated surveillance with push notifications.

Three modes:
1. **Position Monitor** — HOLD/CLOSE/ROLL verdicts on open Schwab positions (credit-patience / debit-urgency)
2. **Dip Scanner** — smart screener for quality stocks experiencing meaningful drops (4-layer scoring: quality + drop + context + recovery)
3. **Automated Alerts** — 30-min scheduled monitor with ntfy push + SMS for CLOSE/ROLL/BUY signals

Analyze open Schwab positions with live market data, risk metrics, news, macro events, X sentiment, and per-position HOLD/CLOSE/ROLL verdicts biased toward patience and profit maximization.

## Core Philosophy

> "Close when the thesis breaks or the math inverts, not at an arbitrary profit target."

For each position, answer: "Is the expected value of holding for another week positive?" If yes, HOLD. If no, what's the best exit?

**Credit trades (short premium):** Patience is a strategy. Time decay works in your favor — theta erodes the position toward zero. HOLD through temporary adverse moves if the underlying stays OTM with adequate DTE.

**Debit trades (long premium / debit spreads):** Time is the enemy. Every day costs you theta. Do NOT apply credit-trade patience rules to debit positions. Act faster. A debit spread that is OTM and trending away is not "healing" — it is bleeding.

## Parameters

- **days** (optional): History lookback for entry matching. Default: 90.
- **symbol** (optional): Filter to a specific underlying (e.g., AAPL).

Examples:
- `/tradehistory` — analyze all open positions
- `/tradehistory 60` — 60-day lookback
- `/tradehistory 90 AAPL` — AAPL positions only

## Execution Steps

### Phase 1: Data Collection

Run the Python analyzer to fetch positions + Schwab data + yfinance context:

```bash
cd "c:/uw_root" && python -m uwos.schwab_position_analyzer --days {days} [--symbol {symbol}]
```

Timeout: 180000ms (3 min). If Schwab auth fails, tell user to run with `--manual-auth` in their terminal.

Read the output JSON from `c:/uw_root/out/trade_analysis/position_data_{date}.json`.

If no open positions found, report that clearly and stop.

### Phase 2: Research (parallel per unique underlying)

For each unique underlying ticker in the positions, use the Agent tool to run parallel WebSearch agents:

1. **News search:** `"{ticker} stock news last 7 days"` — headlines, analyst actions, SEC filings
2. **X sentiment search:** `"{ticker} stock Twitter sentiment"` — bullish/bearish themes
3. **Macro search (once, not per ticker):** `"stock market macro outlook Fed CPI VIX this week"` — rates, economic data, market regime

Use `subagent_type: "general-purpose"` with WebSearch tool for each. Run them in parallel.

### Phase 3: Analysis & Verdicts

For each position, produce a verdict card combining Phase 1 data + Phase 2 research.

## Verdict Decision Matrix

### Credit Trades (short puts, short calls, iron condors, credit spreads) — patience-biased

| Signal | Verdict | Rationale |
|--------|---------|-----------|
| Theta strong, no catalyst, IV stable/elevated | **HOLD — let it work** | Time is on your side |
| 50%+ profit, high IV rank, strong theta, no earnings | **HOLD — target 65-80%** | Premium still rich, more to harvest |
| 50%+ profit, IV crushed or theta slowing (DTE < 10) | **CLOSE — diminishing returns** | Gamma risk not worth it |
| Underlying against you, still OTM, good DTE | **HOLD — time heals** | Don't panic-close when theta is working |
| Underlying against you, near/ITM, DTE < 14 | **ROLL — extend duration** | Buy time + collect credit |
| Earnings within 7 days, at profit | **CLOSE or ROLL past** | Binary event risk |
| Earnings within 7 days, at loss | **ASSESS — hold if OTM with buffer** | Evaluate on merits, may not be worth eating the loss |
| IV expanding, short vol position | **HOLD if thesis intact** | IV expansion is temporary if fundamentals solid |
| News/macro adverse + fundamentals deteriorating | **CLOSE — thesis broken** | Only close when the reason for the trade changes |
| Near max profit (>85%) | **CLOSE — nothing left** | Risk/reward inverted |

### Debit Trades (long calls, long puts, debit spreads, LEAPS) — urgency-biased

**First, always check:** Is the long leg ITM, ATM, or OTM? Debit trades need the underlying to MOVE in your favor. Patience without movement is just bleeding.

| Signal | Verdict | Rationale |
|--------|---------|-----------|
| Long leg ITM or ATM, trend in your favor, DTE > 21 | **HOLD — thesis intact** | Position is working, give it room |
| Long leg OTM by < 3%, DTE > 21, underlying flat | **HOLD with stop** — set a price stop | Marginal situation; need movement soon |
| Long leg OTM by 3-5%, DTE < 35, underlying trending away | **ASSESS — close unless catalyst imminent** | Time + direction working against you |
| Long leg OTM by > 5%, DTE < 35, any trend | **CLOSE — thesis broken** | Math is inverted; can't recover in time |
| Long leg OTM by any amount, DTE < 14 | **CLOSE — theta acceleration kills it** | Last-week theta decay is brutal on long premium |
| Known binary event (FOMC, earnings, CPI) within 48h, long leg OTM | **CLOSE or hedge** | Binary event won't help an already-losing debit position |
| Long leg OTM, position down > 40% of premium paid | **ASSESS immediately** | Escalate to CLOSE unless strong catalyst within 5 days |
| Long leg OTM, position down > 60% of premium paid | **CLOSE — cut losses** | Expected value is negative; remaining premium has low chance of recovery |
| Underlying trending strongly in your favor, long leg approaching ITM | **HOLD — let it run** | This is the scenario you paid for |
| Near max profit (long spread at 80%+) | **CLOSE — nothing left to gain** | Take the win |

## Output Format

### Per-Position Verdict Card

**Before writing each verdict, classify the position:**
- **CREDIT** (short premium): use credit decision matrix — patience-biased
- **DEBIT** (long premium / debit spread): use debit decision matrix — urgency-biased
- **EQUITY**: use simpler metrics (P&L, trend, news)

```
### {UNDERLYING} — {CREDIT|DEBIT} | {Short/Long} {Put/Call} ${strike} | {expiry} | {DTE} DTE
**Status:** {+/-$P&L} ({pnl%}) | Prob Profit: {prob}% | Theta: {+/-$}/day
**Long leg vs underlying:** {ITM / OTM by X%} — {trending toward / away from strike}

| Metric | Value | Signal |
|--------|-------|--------|
| P&L | {pnl%} of max | {assessment} |
| Delta | {delta} | {OTM/ATM/ITM assessment} |
| Theta/day | {$value} | {Strong/Weak/Bleeding} |
| Gamma risk | {value} | {Low/Medium/High} |
| IV Rank | {pctl} | {Premium rich/fair/cheap} |
| IV vs HV | {spread} | {Overpriced/Fair/Underpriced vol} |
| Breakeven | ${price} ({dist}% away) | {Wide/Narrow/Breached buffer} |
| Earnings | {date} ({days} days) | {No overlap/Caution/Danger} |
| Bid/Ask | {spread%} | {Clean/Acceptable/Wide exit} |
| Support | {levels} | {Above/Near/Below strike} |

**News:** [2-3 bullet summary of recent news]
**Macro:** [relevant macro context — Fed, sector rotation, VIX]
**X Sentiment:** [bullish/bearish/mixed + key themes]

**VERDICT: {HOLD/CLOSE/ROLL/ASSESS} — {one-line reason}**
{2-3 sentence reasoning based on the data. What would change the verdict.}
```

For ROLL verdicts, include specifics:
- Roll to which expiry (target DTE)
- Roll to which strike (same, lower, higher)
- Expected credit/debit if estimatable from the chain data
- New probability of profit after roll

### Portfolio Summary (at end of report)

- Total open positions, total unrealized P&L
- Sector concentration breakdown
- SPY correlation risk (are all positions correlated?)
- Macro regime alignment
- Overall portfolio health score (1-10)

### Output Files (ALWAYS include — clickable links)

```
- Analysis report: [trade-analysis-{date}.md](out/trade_analysis/trade-analysis-{date}.md)
- Position data: [position_data_{date}.json](out/trade_analysis/position_data_{date}.json)
```

## Report Writing

After generating all verdict cards, write the full analysis to:
`c:/uw_root/out/trade_analysis/trade-analysis-{date}.md`

Use the Write tool to create the file, then present clickable links to the user.

## Error Handling

- Schwab auth failed: tell user to re-auth in terminal: `python -m uwos.schwab_position_analyzer --manual-auth`
- No open positions: report clearly — "No open positions found"
- yfinance data missing: skip that enrichment, note in output
- WebSearch fails: skip research for that ticker, note it
- Equity positions (no Greeks): analyze with simpler metrics (P&L, support/resistance, news only)
