# Daily Options Pipeline Runbook

This runbook is written as simple commands you can give Codex.

You do not need to remember Python commands. Tell Codex what to do in plain English, and Codex should run the correct pipeline command, inspect the output, and summarize the decision.

## The Operating Rhythm

```text
After market close:
  Download UW files.
  Tell Codex to generate tomorrow's candidates.

Before next market entry:
  Tell Codex to live-check entries.
  Enter only trades that pass live Schwab pricing.

During trade life:
  Tell Codex to review open positions.

Weekly:
  Tell Codex to audit the pipeline.
```

## Commands To Give Codex

### 1. Generate Tomorrow's Candidates

Use this after market close, once that day's UW files are downloaded into the dated folder.

Say:

```text
run daily pipeline for 2026-MM-DD
```

Example:

```text
run daily pipeline for 2026-04-20
```

What Codex should do:

- Use the UW files in `/Users/anuppamvi/uw_root/tradedesk/2026-04-20`.
- Collect or refresh UW GEX if needed.
- Generate the candidate trade Markdown.
- Generate the machine-readable decision CSV.
- Summarize Core, Tactical, and Watch trades.

Expected answer from Codex:

```text
Core: X
Tactical: Y
Watch: Z

Approved trades:
...

Output file:
...
```

### 2. Live-Check Entries Before Trading

Use this the next market session before placing trades.

Say:

```text
run live entry check from 2026-MM-DD
```

Example for Monday using Friday data:

```text
run live entry check from 2026-04-17
```

What Codex should do:

- Use the prior-session UW candidate folder.
- Use live Schwab quotes for current option bid/ask.
- Rebuild the recommendation table with live entry prices.
- Say `ENTER`, `SKIP`, or `WAIT` for each approved candidate.

Expected answer from Codex:

```text
CCL: ENTER only if debit <= 0.91
MRVL: SKIP because live debit is above gate
MU: WAIT because spread is too wide
```

Important:

- Do not use stale prior-day option prices for entry.
- If Schwab live pricing fails, do not trade.
- If the live gate fails, skip.
- If the spread is too wide, skip or wait.

### 3. Review Open Trades

Use this after positions are entered.

Say:

```text
review open Schwab option positions
```

What Codex should do:

- Read current Schwab positions.
- Match them to the pipeline plan where possible.
- Recommend one of:

```text
HOLD
CLOSE
ROLL
SET STOP
```

Expected answer from Codex:

```text
MU Iron Condor: HOLD
Reason: still inside range, credit decay working.

CCL Bull Call Debit: SET STOP
Reason: underlying close is near invalidation.
```

### 4. Run Weekly Audit

Use this once a week, usually after Friday close or over the weekend.

Say:

```text
run weekly daily-pipeline audit
```

What Codex should do:

- Replay all available dated folders.
- Use available UW GEX enrichment.
- Backtest completed recommendations.
- Compare approved trades vs rejected trades.
- Report win rate, net P/L, profit factor, and failure reasons.
- Identify rule bugs or overfitting risks.

Expected answer from Codex:

```text
Approved completed trades: X
Wins / losses: X / Y
Win rate: X%
Net P/L: $X
Profit factor: X

New issues found:
...
```

### 5. Check Schwab Token

Use this before market open if there is any doubt about Schwab auth.

Say:

```text
check Schwab token
```

What Codex should do:

- Check token age and approximate refresh expiry.
- Run a Schwab quote smoke test.
- Tell you if renewal is needed.

If renewal is needed, say:

```text
renew Schwab token
```

What Codex should do:

- Open Schwab login in the browser.
- Wait for your MFA/approval.
- Capture the callback automatically if possible.
- Confirm the new token window.

## Monday 2026-04-20 Plan

For Monday, we are using Friday `2026-04-17` UW data.

Say:

```text
run live entry check from 2026-04-17
```

Current Friday-derived Tactical candidates:

| Ticker | Strategy | Gate |
|---|---|---|
| CCL | Bull Call Debit | enter only if debit `<= 0.91` |
| MRVL | Bull Call Debit | enter only if debit `<= 4.20` |
| MU | Iron Condor | enter only if credit `>= 4.10` |

Expected decision style:

```text
CCL: ENTER / SKIP / WAIT
MRVL: ENTER / SKIP / WAIT
MU: ENTER / SKIP / WAIT
```

Use Tactical size only.

## Non-Negotiable Rules

- Do not trade Watch rows.
- Do not enter if live Schwab pricing fails.
- Do not enter if the live entry gate fails.
- Do not manually override the gate.
- Do not size Tactical trades as Core trades.
- Do not average down.
- Do not mix trend-analysis into daily-pipeline runs.
- Do not treat historical replay output as live entry approval.

## Sizing Guide

```text
Core:      normal risk size
Tactical: 0.25x to 0.50x normal risk size
Watch:    no trade
```

If there are no Core trades, do not force one.

## Current Confidence Benchmark

Latest fixed full-window replay:

```text
Usable replay days: 80
Candidate rows tested: 5,286
Approved completed trades: 40
Wins / losses: 28 / 12
Win rate: 70.0%
Net P/L: +$14,210
Profit factor: 6.50
Rejected completed trades: 4,281
Rejected win rate: 32.2%
Rejected net P/L: -$111,255
Rejected profit factor: 0.86
```

Confidence:

```text
Decision-support pipeline: 74 / 100
Reduced-size Tactical execution: 72 / 100
Blind auto-trading: 55 / 100
```

Use the pipeline as disciplined decision support, not blind auto-trading.

