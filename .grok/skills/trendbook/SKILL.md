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

Weekly trend book: Weinstein Stage 2 + beating SPY. Buys the breakout (including a young spike) and the first early pullback. Stock first. Empty **ADD** is valid. Never submit, cancel, or replace an order.

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

Universe is generated each run (`var/universe.json`): this desk's Stage-2 memory, live 52-week highs, Schwab movers. Not `configs/universe.txt`, not groat/xhigh. Empty ADD is valid; a typical week should print 1–2 genuine first tickets when the tape has them. One ADD per campaign. ADD = Stage 2 breakout (weeks 1–4, break-week volume not contracted, residual 3m after beta ≥ 0, not lagging sector) or the first early pullback with residual ≥ 0. No new ADD if SPY is not Stage 2; tight regime (UUP Stage 2 + TLT Stage 4) is half size, blocked only if residual is negative. Earnings inside 10 days is HOLD. Stop = weekly close below the 30-week. NEW = weak tag. LATE = old and dying. Grade A/B/C is structure, not P(win). Outcomes: next-week open fill, hold 8 weeks or two off weeks; CAPTURE = return > 0. A wick is not a win. HOLD/NEW/LATE are not trades.
