# Daily Options Pipeline Runbook

This runbook is written as simple commands you can give Codex.

You do not need to remember Python commands. Tell Codex what to do in plain English, and Codex should run the correct pipeline command, inspect the output, and summarize the decision.

This runbook does not make the pipeline profitable by itself. It helps by keeping the operating workflow consistent:

- it prevents mixing historical replay with live entry runs
- it keeps the same validated output path every day
- it reduces avoidable mistakes like stale inputs, missing unzip, or wrong run mode

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
- Summarize Core, Tactical, Scout, and Watch trades.
- Write the report to `/Users/anuppamvi/uw_root/tradedesk/out/daily_pipeline_YYYY-MM-DD/anu-expert-trade-table-YYYY-MM-DD.md`.
- Surface missing external-scanner or morning-watch setups so they can be evaluated by the same daily gates instead of disappearing silently.

Expected answer from Codex:

```text
Core: X
Tactical: Y
Scout: Z
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

## Non-Negotiable Rules

- Do not trade Watch rows.
- Do not enter if live Schwab pricing fails.
- Do not enter if the live entry gate fails.
- Do not manually override the gate.
- Do not size Tactical trades as Core trades.
- Do not size Scout trades as Tactical or Core trades.
- Do not average down.
- Do not mix trend-analysis into daily-pipeline runs.
- Do not treat historical replay output as live entry approval.
- Do not promote a trade just because an external scanner likes it; feed it into the daily candidate book and let the daily rules approve, Scout, or reject it.

## Sizing Guide

```text
Core:      normal risk size
Tactical: 0.25x to 0.50x normal risk size
Scout:    0.10x to 0.25x normal risk size
Watch:    no trade
```

If there are no Core trades, do not force one.

Scout is a controlled pilot tier for high-quality near-misses. It is useful when the setup has enough live price/edge/contract evidence to watch actively, but not enough quality to become Tactical or Core.

## Skip-Streak Escalation

If the daily pipeline produces no Core, Tactical, or Scout trades for three market days in a row, do not loosen the gates by hand. Ask Codex:

```text
run near-miss audit on YYYY-MM-DD
```

Then ask:

```text
run daily-pipeline replay audit for the last two weeks
```

The escalation should identify whether the skip streak is caused by market-quality issues or by a pipeline bug. The audit must specifically check Stage-1 flow blockers, GEX availability, IV, Schwab live pricing, external scanner coverage, and whether high-EV structures were dropped before approval.

## Current Confidence Benchmark

Latest validated full-window replay after the high-IV directional SHIELD fix:

```text
Market days replayed: 74
Days with approved trades: 47
Days with zero approved trades: 27
Completed approved trades: 66
Win rate: 64.6%
Net P/L: +$16,177
Profit factor: 4.54
Average approved trades per day: 1.07
Median approved trades per day: 1
```

Confidence:

```text
Decision-support pipeline: 70 / 100
Reduced-size Tactical execution: 68 / 100
Blind auto-trading: 45 / 100
```

Use replay metrics as a guardrail, not a promise. No policy change should stay unless old-data replay shows it does not degrade profit factor, win quality, or trade discipline.

Use the pipeline as disciplined decision support, not blind auto-trading.

Recent validated examples:

```text
2026-04-17 historical replay: 2 Tactical FIRE trades (CCL, MRVL)
2026-04-20 live/current-day run: SKIP
2026-04-21 live/current-day run: SKIP
```

Today a SKIP does not mean the pipeline is dead. The validated replay shows approvals on 47 of 74 market days, so a no-trade day is normal when flow, likelihood, and live entry quality do not line up.
