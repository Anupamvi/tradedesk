---
name: grok-option
description: Use when the user wants a grok-option scan, Expert Trade Table, credit or debit swing setup, unusual options review, book check, or journal update. Triggers include run today's scan, run the scanner, grok-option, Anu table, sell put credit, bull put, bear call, manage open book, revalidate assumptions, oil spike, geo event, ride the wave, playwright, logged-in x.com, and schwab chain. Applies quote-verified rules, regime gate, X veto, earnings firewall, spike path, Schwab live chain, and empty-table permission.
metadata:
  type: workflow
  version: "3.9"
  owner: Kla
---

# grok-option

Defined-risk swing scanner. Not Anu v1. Bar is **$10k profit / month**. Do not chase it by sizing up or loosening geometry.

Read `references/assumption-audit.md` before the first scan in a session. Pull the matching reference when a mode needs depth. Copy the card from `assets/daily-card.md`. Plain markdown only — no HTML. See `assets/report-style.md`.

Equity is live Schwab when pulled (**$715k** on 2026-08-26). $150k only if unknown and Schwab is down. Caps and the $10k bar live in `references/book-and-target.md`.

## Modes

| Mode | When | Load |
|------|------|------|
| **SCAN** | run today's scan, grok-option, Anu table, bull put, sell put credit, Expert Trade Table, oil spike, ride the wave | workflow, regime-and-signals, spike, schwab, browser, structures-and-pricing, expert-table, book-and-target, x-sentiment |
| **MANAGE** | manage open book, stops, takes, rolls | structures-and-pricing, book-and-target |
| **JOURNAL** | journal update, 20-trade review | journal, book-and-target |
| **AUDIT** | revalidate assumptions, 20-trade expectancy < 0, weekly rule review | assumption-audit, then rewrite it if evidence contradicts a numbered rule |

Default is SCAN. MANAGE if the user pastes open defined-risk lines. JOURNAL if they paste closes. AUDIT only when asked or when the journal trips the freeze.

## SCAN order

Do not reorder. Empty table is valid when quotes or geometry fail. Empty is a **bug** if a non-event name would still be quoted today. Run the over-gate test in `regime-and-signals.md` before skip. Do not loosen delta/width to avoid empty days.

1. **Regime line** — VIX, 5-day, 52-week; SPX/SPY vs 20/50 if available; FOMC, CPI/PCE, NFP, OPEX, mega-cap earnings; **Shock watch** (WTI/Brent + X geo finder + web check if X flags). See `references/spike.md`. Missing Shock watch is incomplete.
2. **Ingest** — **Schwab first** for ticker last/bid/ask and option chains (`references/schwab.md`, `scripts/schwab_market.py`, tradedesk `.env` + `SCHWAB_TOKEN_PATH`). User CSVs and pasted book next. Browser (`references/browser.md`) only for X, cookie-walled news, crude/geo copy, or when Schwab is down.
3. **Stock-only universe** — common stock. If a shock is sourced, include that shock’s **written map** only. No ETFs unless the user said allow index hedge. No warrants, blanks, or unquotable names.
4. **Notional filter** — liquid weeklies and monthlies. Cluster AI/semi/cloud as one theme; energy map is one theme.
5. **Live chain** — Schwab `structures` per name/expiry (all five actions). Then `vertical` to fill a row. Conservative net. **Pick highest credit/width, then dollars** (1-lot credit ≥ $100 when it exists; do not 15-wide a thinner-edge scrap). Skip a **structure** if a leg is missing, earnings unknown, or `earnings_date <= expiry`. Do not skip the other four because puts printed. See `references/schwab.md` and `references/structures-and-pricing.md`.
6. **X on candidates** — required after the chain. Veto/confirm only, never a trigger. Tag every table row `X: Quiet` / `Informed` / `Crowded veto` / `Event veto`. One regime pass (VIX, NVDA, Chair, 0DTE). Missing this pass is an incomplete scan. See `references/x-sentiment.md`.
7. **Sleeve** — A Shield, B Fire, C cash/hedge, D Spike. Events are **name / theme / index** (`regime-and-signals.md`). Crisis kills Shield/Fire; Spike is the only Crisis debit if `spike.md` gates pass.
8. **Book caps** — `references/book-and-target.md`. Reject rows that breach per-name, aggregate, theme, one-Fire-per-name, or one-Spike-per-scan.
9. **Write the card** — executable table **first** (0–7 rows). Dollar columns are **1-lot** max profit and max loss. **Rec lots** from Conf (`expert-table.md`). Buy/Sell cells name every leg (`Buy 225 Put` / `Sell 240 Put`; condor lists all four). Then Shock watch, then sleeve board. Copy `assets/daily-card.md`. Never title the trades “not executable.” A put-only table with no sleeve board is incomplete. Never mix a 1-lot P/L with an N-lot dollar.
10. **Assumptions in force** — 3–5 bullets from the audit file. Stop. No second executable table.

## Sleeves

**A SHIELD** — put credit **and** call credit. If both pass the same name/expiry, print **one iron condor**, not two verticals. Do not default to puts. Geometry in `structures-and-pricing.md`. Calm Score 65, size 1.0% then BP. NVDA = semis only. Manage 60–65% / 2.0×.

**B FIRE** — score **both** call debit and put debit. Half size when cheap IV, not Crowded, not name/theme Event, not Crisis, quoted debit. Opening flow is required for an **Expert table** Fire row (Score 80 path). Missing flow → sleeve board only, not a printed debit next to 80% Conf Shields. VWAP unverified → no Prime.

**C HEDGE/CASH** — cash is a position. Index hedges only if the user said allow index hedge.

**D SPIKE** — own lane. X may **find** a geo/oil/kinetic event for Watch. A table row still needs web source + map + quoted debit at **0.25%**. Max one row. No Shield on mapped names. Details in `references/spike.md`.

## Tools

- Schwab: `python3 ~/.grok/skills/grok-option/scripts/schwab_market.py structures TICKER --expiry YYYY-MM-DD` then `vertical` for the row. See `references/schwab.md`.
- Web: VIX, index, earnings date, WTI/Brent, named geo copy.
- X keyword + semantic search: Shock watch finder (geo/oil/kinetic) at regime; then candidate veto/confirm. Never a Spike row from X alone.
- Browser: Playwright MCP or `scripts/browser_fetch.py` for X/news when APIs miss. Not a substitute for Schwab chain. See `references/browser.md`. Never type passwords.
- Credit net = short bid minus long ask. Debit net = long ask minus short bid. If mids disagree by more than 5%, print the worse fill.
- CSVs when attached: flow, OI, chain. They upgrade THIN → MIXED/FULL. They still do not trigger.

## Scores and data

**Score** (80 / 65 / 50) is gate quality, not P(win). **Conf** is trade-success confidence: naive POP from quoted Schwab delta + book 20-trade win rate (`book n/a` until 20 closes). **Rec lots** = `max(1, round(naive_POP/100 × floor(sleeve_cap / 1-lot max loss)))`, then cash/BP. Formulas in `expert-table.md`. Data flags FULL / MIXED / THIN. Prime = Score 80 + Data FULL + quoted Conf. No Prime on THIN, unverified quotes, earnings-overlap, Crowded/Event Fire, or Crowded Spike. No 74.0.

## Bans

- Invented quotes, IV, IV rank, OI, OI%, volume multiples, or delta
- "Buy Put Credit"
- Conviction tenths (74.0); LLM-guessed Conf / POP
- Always-produce-a-table
- Put-credit-only scan (must price call credit, condor, call debit, put debit)
- Size-up to chase $10k
- Mixing 1-lot max profit/loss with an N-lot dollar in the same table row
- Theme-cap breach
- Copying a whale strike/expiry as the trade
- Any expiry on or after the name’s next earnings date (overlap); missing earnings date is a skip, not a pass
- Naked / undefined risk
- Treating PCR, dark-pool prints, or X as entries (X may start Shock watch, not a Spike row)
- Automating passwords / 2FA; inventing quotes from a screenshot with no DOM text
- Spike off the written map, a fade unless the user said fade, or a calendar Event dressed as a shock

## AUDIT

When the user says revalidate: refresh tape, rewrite `references/assumption-audit.md` (20–40 line Tape + Evidence, then KEEP/AMEND/KILL). Freeze Fire and cut size 50% **only** if 20-trade expectancy < 0. Do not restore killed theater.
