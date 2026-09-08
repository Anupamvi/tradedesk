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

Weekly trend book: Weinstein stage + beating SPY + campaign age with 1–2 week hysteresis. Stock first. Empty **ADD** is valid. Never submit, cancel, or replace an order.

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

0 ADD is valid. ADD only on a 10-week or 50-day pullback in a name beating SPY by ≥15% over 6 months, and at least 8% off the campaign high. A 20 EMA pause near highs is HOLD, not a buy. NEW = just tagged. Grade A/B/C is structure, not P(win). Tightness-only is HOLD. RS percentile is never a kill switch. Probe 52-week highs / large RS so names off the static list can appear as NEW. Options overlay on ADD only when IV is cheap. Outcomes score ADD only: CAPTURE = new high within 8 weeks or still on-board at week 8; FAIL = left Stage 2 before a new high. HOLD/NEW/LATE are not trades.
