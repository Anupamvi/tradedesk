---
name: groat
description: >
  Run the Groat swing-trading research desk for a session date. Triggers include
  groat, gorat, groat YYYY-MM-DD, gorat 2026-08-27, groat 2026-08-27, groat full,
  groat today, groat morning, groat replay, groat replay YYYY-MM-DD, RUN FULL SCAN, RUN DELTA SCAN, ANALYZE TICKER,
  groat analyze, REVIEW OPEN TRADES, groat review, /groat. Independent of groki,
  groko, and Codex Daily. Stock first, then options. No order placement.
---

# Groat

CODE=`/Users/anuppamvi/tradedesk/groat`

Groat finds a **small** number of high-quality **stock and options** swing trades. Empty board is valid. The user clicks every Schwab order. Never submit, cancel, or replace.

Never invent ORATS numbers, prices, X posts, news, or win probabilities. Missing source → **DATA UNAVAILABLE**.

## Parse the date (do this first)

| User says | DATE |
|---|---|
| `groat 2026-08-27` / `gorat 2026-08-27` / `groat full 2026-08-27` | that YYYY-MM-DD |
| `groat` / `groat full` / `groat today` / `RUN FULL SCAN` (no date) | today America/New_York |
| `groat morning` / `groat revalidate` | today America/New_York (open recheck) |
| `groat delta` | delta; DATE as above |
| `groat analyze NVDA` / `ANALYZE NVDA` | analyze; DATE as above |
| `groat review` | review; DATE as above |
| `groat replay` / `groat replay 2026-08-27` | replay; DATE as above |

Typo `gorat` = `groat`.

## Run (agent runs this; do not tell the user to type it)

From CODE, timeout 600000ms.

```bash
python3 -m groat full --date DATE
```

`python3 -m groat DATE` is the same as `full --date DATE`.

```bash
python3 -m groat delta --date DATE
python3 -m groat analyze TICKER --date DATE
python3 -m groat review --date DATE
python3 -m groat replay --date DATE --option-slices 3 --max-strike-http 40
```

`groat replay` / `groat replay YYYY-MM-DD` is **Python on cached tape**, not an LLM mode. Surviving setups only (park B/C/G; park post-rip E). Stock walk is free; `--option-slices N --max-strike-http 40` prices options on those hits only.

If `ORATS_TOKEN` is missing (exit 2), tell the user to edit `CODE/.env` or run `read -s ORATS_TOKEN && export ORATS_TOKEN`. Do not ask them to paste the token. Never print it.

Schwab: `CODE/.env` then `/Users/anuppamvi/tradedesk/.env`. Live Schwab when `--date` is today.

## Evening vs morning (when they ask about placing last night’s trades)

**Do not place last evening’s option ticket blindly at 9:30 ET.**

- **Evening (16:30–18:00 ET)** is the **daily list**: full session 1d/rvol/FIRE, Delayed ORATS cores/strikes for that close, analog evidence. That is the default daily `groat`.
- **Overnight / open can change the setup materially:** gap through stop or AVWAP, IV/ask on the debit, chase if the name rips again, news/AH earnings. Stock thesis often survives a quiet open. **Option fills from last night are stale.**
- **Morning:** if they want to **click orders**, re-run `groat full --date TODAY` after **~9:45–10:15 ET** (open auction done). Then work **today’s** debit/credit, not last night’s print. Skip names that gapped >1 ATR against the stop or are now extended >2.5 ATR.
- Do **not** make morning the only daily run. Delayed ORATS + incomplete 1d bars at 9:30 will mis-rank FIRE/X-HOT.

Cadence: evening full scan → watchlist. Next session: morning revalidate only if placing.

## X before / after the Python scan

**X-HOT (conversation first).** Search X for what is loud *now*. Write `CODE/var/xhot/DATE/hot.json`:

```json
{"asof":"DATE","source":"x_keyword_search","names":[
  {"ticker":"NVDA","heat":"hot","bias":"bullish","posts_24h":80,"narrative":"...","tag":"Crowded"}
]}
```

Bias: `bullish` / `bearish` / `unknown`. Downweight pumps. Re-run Groat so tape can label **dipped** / **will_rise** / **will_dip**. Heat without a trigger stays Watch.

**FIRE** is tape first; X confirms or vetoes.

After the scan:

1. Read `board.md` — **Desk pick first**, then **Evidence**.
2. X on `x_queue.json`. Write `CODE/var/xintel/DATE/TICKER.json` (`tag` Quiet|Informed|Crowded). Re-run if tags were missing.
3. Missing X → **DATA UNAVAILABLE**. Do not invent posts. Do not change ORATS/price/debit/credit numbers.

## Reply shape

```
**Groat DATE** | regime {label} | TRADE {n} | WATCH {n}

{Desk pick}

{TRADE index}

{FIRE}

{X-HOT}

{one card per TRADE}

### Files
- Board: [board.md](/Users/anuppamvi/tradedesk/groat/out/groat/DATE/board.md)
- Report: [report.md](/Users/anuppamvi/tradedesk/groat/out/groat/DATE/report.md)
- Regime: [regime.md](/Users/anuppamvi/tradedesk/groat/out/groat/DATE/regime.md)
- Evidence: [evidence.md](/Users/anuppamvi/tradedesk/groat/out/groat/DATE/evidence.md)
```

0 TRADE rows is valid. Do not loosen gates to fill the table.

## Hard rules

- Underlying thesis first. Options serve the thesis.
- Ordinary options: do not hold through earnings. Missing earnings date → options rejected.
- EVENT TRADE — EARNINGS is never auto-selected.
- DTE ~21–75. No 0DTE.
- Conservative fills: debit at ask, credit at short bid − long ask. Never mid. Report **target debit/credit**.
- Review stock, long call, long put, call debit, put debit, put credit, call credit, then shortlist.
- **conf** is structure quality 0–85, not P(win).
- Prefer 2:1 R/R. Risk 0.5–1% of the 50k research account.
- Do not chase >2.5 ATR above 20 EMA.
- Do not import other desks as the execute path.
