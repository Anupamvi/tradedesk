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

`groat replay` / `groat replay YYYY-MM-DD` is **Python on cached tape**, not an LLM mode. Surviving setups only (park B/C/G/H; park post-rip D at 3% 1d and E at 12% 1d). Stock walk is free; `--option-slices N --max-strike-http 40` prices options on those hits only.

If `ORATS_TOKEN` is missing (exit 2), tell the user to edit `CODE/.env` or run `read -s ORATS_TOKEN && export ORATS_TOKEN`. Do not ask them to paste the token. Never print it.

Schwab: `CODE/.env` then `/Users/anuppamvi/tradedesk/.env`. Live Schwab when `--date` is today.

Every `full` / `delta` / `analyze` **always refreshes**: ORATS cores, ORATS strikes, and Schwab daily tape (`price_history` + live quote when `--date` is today). Do not reuse that date’s chain or bar cache. Replay analog `hist/strikes` still uses cache + `--max-strike-http`.

## Evening vs morning (when they ask about placing last night’s trades)

**Do not place last evening’s option ticket blindly at 9:30 ET.**

- **Evening (after 16:00 ET)** is the **daily list**: official close only (no AH last overwrite, no mark-padded fills labeled as close). Writes `out/groat/DATE/` and copies to `out/groat/DATE/close/`.
- **Morning / open auction (before 9:45 ET)** is incomplete: new TRADE is blocked. After ~9:45–10:15 ET, re-run if placing — RTH **can print TRADE**. Low morning rvol only means FIRE/1d is not final. Copies to `out/groat/DATE/open/`. Do not overwrite close with open or vice versa.
- Same ticker+setup that was TRADE last **complete** session stays WATCH unless it pulled back into 20 EMA / AVWAP or group_status changed. Incomplete morning `open/` is not the TRADE prior — evening still uses yesterday close.
- Same-group as an open book name (e.g. XOM while CVX energy is open) stays TRADE with a caveat. Your decision whether to add a lot.
- Crowded leftover and >3% OTM lottery are not a desk pick. Empty desk pick is valid.
- Regime **unknown** blocks new TRADE. Open auction (before 9:45 ET) blocks new TRADE. Regular hours after 9:45 do **not**.

Cadence: evening full scan → watchlist. Next session: morning revalidate only if placing. Do **not** place last evening’s option ticket blindly at 9:30 ET.

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

CLI prints `x_missing_on_trade=...` and exits **3** if any TRADE row has no X tag. That run is incomplete. Do not show that board to the user yet.

1. Read `x_queue.json` and every TRADE ticker (WATCH after that).
2. Search X for each. Write `CODE/var/xintel/DATE/TICKER.json` (`tag` Quiet|Informed|Crowded). Promo spam = Crowded.
3. If `x_missing_on_trade` is not `none`, re-run `python3 -m groat full --date DATE` and only then reply.
- Do not invent posts. Do not change ORATS/price/debit/credit numbers. Missing X stays **DATA UNAVAILABLE** — never map it to Quiet.
- X-HOT `hot.json` is a heat lane. It does **not** satisfy `var/xintel/DATE/TICKER.json`. Exit 3 still applies.
- Optional catalysts: `var/news/DATE/TICKER.json` and `var/filings/DATE/TICKER.json` (`summary`). Missing stays **DATA UNAVAILABLE**. Do not invent news.

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
- D post-rip parks at ≥3% 1d; E at ≥12% 1d.
- Do not import other desks as the execute path.
