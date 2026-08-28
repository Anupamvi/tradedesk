---
name: groat
description: >
  Run the Groat swing-trading research desk. Triggers include groat, groat full,
  groat YYYY-MM-DD, RUN FULL SCAN, RUN DELTA SCAN, ANALYZE TICKER, groat analyze,
  REVIEW OPEN TRADES, groat review, /groat. Independent of groki, groko, and Codex Daily.
  Stock first, then options. No order placement.
---

# Groat

CODE=`/Users/anuppamvi/tradedesk/groat`

Groat finds a **small** number of high-quality **stock and options** swing trades with positive expected value. It does not force a trade every day. Empty board is valid.

The user clicks every Schwab order. Never submit, cancel, or replace.

Never invent ORATS numbers, prices, X posts, news, or win probabilities. Missing source → **DATA UNAVAILABLE**.

## Date

- `groat YYYY-MM-DD` / `groat full YYYY-MM-DD` → that date
- otherwise today America/New_York

## Mode

| User says | Mode |
|---|---|
| groat, groat full, RUN FULL SCAN | `full` |
| groat delta, RUN DELTA SCAN | `delta` |
| ANALYZE TICKER, groat analyze NVDA | `analyze` + ticker |
| REVIEW OPEN TRADES, groat review | `review` |

## Run

From CODE, timeout 600000ms. Do not tell the user to type this.

Full scan:

```bash
python3 -m groat full --date DATE
```

Delta:

```bash
python3 -m groat delta --date DATE
```

Analyze:

```bash
python3 -m groat analyze TICKER --date DATE
```

Review:

```bash
python3 -m groat review --date DATE
```

If `ORATS_TOKEN` is missing (exit 2), tell the user to edit `CODE/.env` or run `read -s ORATS_TOKEN && export ORATS_TOKEN`. Do not ask them to paste the token. Never print it.

Schwab credentials come from `CODE/.env` then `/Users/anuppamvi/tradedesk/.env`. Live Schwab is used when `--date` is today.

## X before / after the Python scan

**X-HOT (conversation first).** Search X for what is loud *now*: `$TICKER`, earnings, unusual volume, dip, squeeze, “gap”, mega-cap prints. Goal: names that **dipped**, **will_rise**, or **will_dip** so a swing/spike is tradeable. Write `CODE/var/xhot/DATE/hot.json`:

```json
{"asof":"DATE","source":"x_keyword_search","names":[
  {"ticker":"NVDA","heat":"hot","bias":"bullish","posts_24h":80,"narrative":"...","tag":"Crowded"}
]}
```

Bias is `bullish` / `bearish` / `unknown`. Downweight pumps. Then run (or re-run) `python3 -m groat full --date DATE` so the tape can label **dipped** / **will_rise** / **will_dip**. Heat without a volume/price trigger stays Watch. `will_dip` after an extended spike means wait for the pullback — do not short strength unless setup G also prints.

**FIRE** is the opposite: price + volume first, X confirms or vetoes.

After the scan:

1. Read `board.md` — **Desk pick is first**, then **Evidence**. Evidence is same-ticker / same-setup analogs on cached tape (stock walk + thin hist/strikes replay for OPTIONS names). It does not change gates. Small n. X-HOT is not in evidence.
2. X on `x_queue.json` names. Write `CODE/var/xintel/DATE/TICKER.json` (`tag` Quiet|Informed|Crowded). Re-run Groat if tags were missing.
3. `research.md` for narrative. Do **not** change ORATS/price/debit/credit numbers.
4. Missing X → **DATA UNAVAILABLE**. Do not invent posts.

## Reply shape

Lead with the trades, then regime, then files.

```
**Groat DATE** | regime {label} | TRADE {n} | WATCH {n}

{Desk pick from board.md — always first}

{TRADE index}

{FIRE table}

{X-HOT table}

{one card per TRADE: thesis paragraph, strategy review table, target debit/credit, confidence, invalidation}

### Files
- Board: [board.md](/Users/anuppamvi/tradedesk/groat/out/groat/DATE/board.md)
- Report: [report.md](/Users/anuppamvi/tradedesk/groat/out/groat/DATE/report.md)
- Regime: [regime.md](/Users/anuppamvi/tradedesk/groat/out/groat/DATE/regime.md)
- Evidence: [evidence.md](/Users/anuppamvi/tradedesk/groat/out/groat/DATE/evidence.md)
```

0 TRADE rows is valid when gates fail. Do not loosen R/R, liquidity, earnings, or chase filters to fill the table.

## Hard rules (do not break)

- Underlying thesis first. Options serve the thesis.
- Ordinary options: do not hold through earnings. Missing earnings date → options rejected.
- EVENT TRADE — EARNINGS is a different strategy and is never auto-selected.
- DTE roughly 21–75. No 0DTE.
- Conservative fills: debit at ask, credit at short bid − long ask. Never assume midpoint. Report **target debit/credit**.
- Review **stock, long call, long put, call debit, put debit, put credit, call credit** on every name, then shortlist. Do not default to long calls.
- Options **confidence** is structure quality 0–85, not P(win).
- Prefer 2:1 R/R on directional trades. Risk 0.5–1% of the 50k research account.
- Do not chase extended breakouts (>2.5 ATR above 20 EMA).
- Do not import other desks as the execute path.
- Dealer GEX, if mentioned, is an assumption — never present it as certain.

## Hierarchy

1. Market regime
2. Underlying thesis
3. Price / AVWAP / volume
4. Catalyst
5. Relative strength
6. Risk/reward
7. ORATS vol + structure
8. Positioning / flow
9. X sentiment
