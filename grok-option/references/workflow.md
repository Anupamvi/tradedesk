# SCAN workflow

Follow this order. Skip a step only if the input cannot exist (then mark THIN or stop that name).

1. **Regime line** — VIX, 5-day, 52-week if available; SPX/SPY vs 20d/50d if available; calendar; **Shock watch** (WTI/Brent + X geo finder + web check if X flags). See `spike.md`. Missing Shock watch is incomplete.

2. **Ingest** — Schwab quotes/chains first (`schwab.md`). Then CSVs / pasted book. Browser only for X, cookie-walled copy, crude/geo, or Schwab down. Schwab both-leg prints upgrade Data to FULL.

3. **Stock-only universe** — Common stock. If a shock is sourced, add that row’s written map. Drop warrants, blank roots, penny names, and ETFs unless the user allowed index hedge. Skip names you cannot quote a two-leg weekly on.

4. **Notional filter** — Prefer liquid mega-caps for Shield. Mid-caps need MIXED/FULL and never Prime on THIN. Theme clustering before sizing. Energy map is one theme.

5. **Live chain** — `schwab_market.py structures SYMBOL --expiry YYYY-MM-DD` on each allowed name (put credit, call credit, condor, call debit, put debit). Then `vertical` to fill a row. Conservative net. Confirmed earnings date; **skip that structure if unknown or `earnings_date <= expiry`**. Missing Schwab leg → that structure is NO ROW, not a skip of the other four.

6. **X on candidates** — Veto/confirm after the chain. Shock watch finder already ran in step 1. See `x-sentiment.md`.

7. **Sleeve** — Calm/Normal → Shield that clears geometry is a row. Fire if cheap IV, not Crowded, not name/theme Event, not Crisis, quoted debit. Opening flow is Score 80, not a skip. Spike only if `spike.md` gates pass (max one row). Events scoped name/theme/index. Crisis → cash unless Spike qualifies. Over-gate test before skip.

8. **Book caps** — Live Schwab equity when pulled ($715k on 2026-08-26), else $150k fallback. Size into the per-name band, then buying power. Theme, one Fire per name, one Spike per scan. No Shield on mapped shock names. Reject overflow. Do not size up to chase $10k/month.

9. **Write the card** — executable table first. 1-lot max profit / 1-lot max loss / Rec lots from Conf. Then Shock watch, then sleeve board. `daily-card.md`. Plain markdown, no HTML. Never put trades under “not executable.” Never mix 1-lot P/L with N-lot dollars. Then 3–5 **Assumptions in force** bullets.

10. **Stop** — Do not emit a second “ideas” table. Watch notes go under the table, unlabeled as executable.

## MANAGE extras

For each open defined-risk line: DTE, quoted value vs entry, % of credit, **earnings vs remaining expiry**, thesis. If `earnings_date <= expiry`, close or roll to an expiry still before the print. Apply 60–65% / 2.0× / 2.5×. Short call vs shares held: close the spread rather than deliver shares unless the user says sell the stock.

## JOURNAL extras

Append closes: date, sleeve, net, max loss, result, expectancy contribution. Rolling 20. If expectancy < 0 → freeze Fire, cut 50%, run AUDIT.

## AUDIT extras

Re-pull tape. If 20-trade evidence kills or amends a numbered assumption, rewrite `assumption-audit.md` (keep the 20–40 line memo fresh) and change the footnote.
