---
name: wheelo
description: >
  Run the Wheelo cash-secured put / covered-call desk. Triggers include wheelo,
  wheelo select, wheelo daily, wheelo full, wheelo YYYY-MM-DD, wheelo analyze TICKER,
  wheelo review, /wheelo. Independent of groat, groko, and uwos.wheel_pipeline.
  Schwab shortlist first, then ORATS delayed. No order placement.
---

# Wheelo

CODE=`/Users/anuppamvi/tradedesk/wheelo`

Wheelo sells CSPs only on `configs/own_list.txt` (holdable multi-year growth). Slightly OTM. The user sizes cash; do not lecture sleeve math or drop a name because 1 lot is expensive. Thesis break → sell, not roll forever. Empty TRADE board is valid. The user clicks every Schwab order. Never submit, cancel, or replace.

Never invent ORATS numbers, Schwab quotes, or X posts. Missing source → **DATA UNAVAILABLE**.

**conf** is structure/research quality 0-85, not P(win). Labels: TRADE / WATCH / NO_TRADE. Hard fails (NO_TRADE): no bid, credit < 1.5% of strike, ATM (<2% OTM), cheap vol (IV/HV < 0.90 and IVR < 50), earnings ≤7d, earnings inside the put's DTE (plus 3d buffer), earnings unknown. Lead the reply with the rotation pick (highest-conf TRADE). Do not casually omit `universe.priority` names (PLTR, MU, HOOD, SPCX, SOFI, ORCL, …); if a keeper has no quote or fails a gate, say why.

## Parse

| User says | CMD | DATE |
|---|---|---|
| `wheelo` / `wheelo full` / `wheelo today` | full | today America/New_York |
| `wheelo YYYY-MM-DD` | full | that date |
| `wheelo select` | select | today unless a date is given |
| `wheelo daily` | daily | today unless a date is given |
| `wheelo analyze SOFI` | analyze SOFI | today unless a date is given |
| `wheelo review` | review | today unless a date is given |

## Run (agent runs this)

From CODE, timeout 180000ms.

```bash
python3 -m wheelo CMD --date DATE --capital 35000
```

`python3 -m wheelo DATE` is `full --date DATE`. Default `--max-orats-requests 15`. Add `--no-schwab` only if asked. Add `--yfinance` only if asked (rate-limits).

If `ORATS_TOKEN` is missing (exit 2), tell the user to edit `CODE/.env` or `read -s ORATS_TOKEN && export ORATS_TOKEN`. Never print it.

Schwab: `CODE/.env` then `/Users/anuppamvi/tradedesk/.env`. Live Schwab when `--date` is today.

## ORATS budget

Do not cores/strikes the universe file. Schwab quotes first, then ORATS on ≤80 / ≤20 names. **Today / live runs always refetch Schwab and ORATS** (disk write is audit only). Historical as-of may reuse `/hist` cache. If the run prints `error=orats_budget`, stop; do not retry with a higher cap unless the user asks.

Delayed `nextErn` is often `0000-00-00`. Use `wksNextErn` (~7d per week) for the earnings gate.

## X after the scan

Read `x_queue.json`. Search X for those tickers plus the rotation pick. Write `CODE/var/xhot/DATE/hot.json`:

```json
{"asof":"DATE","source":"x_keyword_search","names":[
  {"ticker":"SOFI","heat":"hot","bias":"bullish","posts_24h":20,"narrative":"...","tag":"Informed"}
]}
```

Bias: `bullish` / `bearish` / `unknown`. Tag: Quiet|Informed|Crowded. Missing X → **DATA UNAVAILABLE**. Do not change ORATS/Schwab numbers from X. Do not burn a second ORATS run just to overlay X.

## Reply shape

```
**Wheelo DATE** | rotation {TICKER} | conf {n} TRADE | orats_http {n}

{one sentence: why this name — thesis + spot + put/expiry/bid + Cr% + OTM + IVR}

{board table from board.md}

Keepers missing a quote or TRADE label: {ticker — reason}. Do not skip silently.

### Files
- Board: [board.md](/Users/anuppamvi/tradedesk/wheelo/out/wheelo/DATE/board.md)
- Report: [report.md](/Users/anuppamvi/tradedesk/wheelo/out/wheelo/DATE/report.md)
- Daily: [daily.md](/Users/anuppamvi/tradedesk/wheelo/out/wheelo/DATE/daily.md)
```

0 TRADE rows is valid. Credits are put **bid**, not mid. Do not overlay covered calls on protected core holdings. Do not open with cash-sleeve accounting.
