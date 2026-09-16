---
name: trendbook
description: >
  Run the independent weekly Stage 2 / relative-strength trend book.
  Triggers include trendbook, trend book, weekly trend, trendbook YYYY-MM-DD,
  trendbook full, trendbook today, trendbook analyze TICKER, trendbook replay.
  Independent of groat, xhigh, grok-option, and UW trend-analysis. No order placement.
---

# trendbook

CODE=`/Users/anuppamvi/tradedesk/trendbook`

Weekly trend book: Weinstein Stage 2 + beating SPY. Buys a breakout from a base, or the first early pullback. A Stage 3 MA-turn after the stock already ran is not a buy. Stock first. Empty **ADD** is valid. Never submit, cancel, or replace an order.

Do not treat `groat`, `RUN FULL SCAN`, `xhigh`, `grok-option`, `trend-analysis`, or `pattern` as this desk.

Never invent ORATS, Schwab, X, or earnings numbers. Missing source → **DATA UNAVAILABLE**. No Unusual Whales. No X API.

## Parse

| User says | CMD | DATE |
|---|---|---|
| `trendbook` / `trend book` / `weekly trend` / `trendbook full` / `trendbook today` | full | today America/New_York |
| `trendbook YYYY-MM-DD` | full | that date |
| `trendbook analyze IBM` | analyze IBM | today unless a date is given |
| `trendbook replay` | replay | today unless a date is given |

## Run

From CODE, timeout 600000ms.

```bash
python3 -m trendbook full --date DATE
```

Analyze: `python3 -m trendbook analyze TICKER --date DATE`.

Replay (full-universe first-tag then 4/8/13w forward): `python3 -m trendbook replay --date DATE`.

Schwab uses `SCHWAB_TOKEN_PATH` from tradedesk `.env` (do not copy the token into this folder). ORATS from `xhigh/.env` then `groat/.env` then tradedesk `.env`. If `orats=missing`, say so. Do not print the token.

## Reply

```
**Trendbook DATE** | SPY stage {n} | ADD {n} · NEW {n} · HOLD {n} · LATE {n} · OUT {n}

{ADD table or: No ADD rows. Prefer grade A then B.}

### Files
- Board: [board.md](/Users/anuppamvi/tradedesk/trendbook/out/trendbook/DATE/board.md)
- Replay: [replay.md](/Users/anuppamvi/tradedesk/trendbook/out/trendbook/DATE/replay.md)
- Outcomes: [outcomes.md](/Users/anuppamvi/tradedesk/trendbook/out/trendbook/DATE/outcomes.md)
- Evidence: [evidence.md](/Users/anuppamvi/tradedesk/trendbook/out/trendbook/DATE/evidence.md)
```

0 ADD is valid. ADD is a buy, not a tag. Breakout ADD = first Stage 2 ticket from a base (prior Stage 1/4, or a 1–2 week Stage 3 poke still within 10% of the 30-week) with break-week volume expansion, residual 3m after beta ≥ 0, not lagging sector. A Stage 3 MA-turn after the stock already ran is NEW. Incomplete mid-week bars do not create a second week. Pullback ADD = first 10-week / 50-day / 30-week dip with residual ≥ 0. A 20 EMA pause near highs is HOLD. One ADD per campaign. NEW = weak tag. LATE = old and dying. Grade A/B/C is structure, not P(win). RS percentile is never a kill switch. Probe 52-week highs / large RS so names off the static list can appear as NEW. Options overlay on ADD only when IV is cheap. Outcomes score ADD only. HOLD/NEW/LATE are not trades.
