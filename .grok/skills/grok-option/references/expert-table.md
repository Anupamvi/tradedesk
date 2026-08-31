# Expert Trade Table

One executable table, dated today, **0–7 rows**. Empty is valid. Never backfill from watch, alternates, or diagnostics.

Card order is mandatory: **Expert table first**, then Shock watch, then sleeve board. Never put the executable table under a “not executable” heading.

**Sleeve board** is scored-but-not-traded. Print it **below** the table. Status is **YES** / no. A put-only executable table with no board is incomplete.

## Columns

Card table: `Ticker | Sleeve | Action | Expiry | Buy (long) | Sell (short) | Max profit $ | Max loss $ | Rec lots | Score | Conf | Data`

**Buy** and **Sell** must name every leg with the word Buy/Sell, the strike, and Put or Call. Never `P 445 / C 550` with no verb.

| Action | Buy (long) | Sell (short) |
|--------|------------|--------------|
| Sell put credit | `Buy 225 Put` | `Sell 240 Put` |
| Sell call credit | `Buy 550 Call` | `Sell 540 Call` |
| Sell iron condor | `Buy 445 Put + Buy 550 Call` | `Sell 460 Put + Sell 540 Call` |
| Buy call debit | `Buy 410 Call` | `Sell 420 Call` |
| Buy put debit | `Buy 150 Put` | `Sell 145 Put` |

Condor lists **all four** legs. Missing a wing is incomplete.

**Max profit $** and **Max loss $** are always **1 lot**. Rec lots is the only scaled number. Never mix a 1-lot P/L with an N-lot dollar.

- **Action** must be one of: Sell put credit, Sell call credit, Sell iron condor, Buy call debit, Buy put debit.
- **Max profit $** (1 lot): credit = `net × 100`. Debit = `(width − net) × 100`. Condor = `(put net + call net) × 100`. Conservative fill: short bid − long ask, or long ask − short bid.
- **Max loss $** (1 lot): credit = `(width − net) × 100`. Debit = `net × 100`. Condor = `(max(put width, call width) − total net) × 100`. Mark `worse-fill` in Notes if mids disagree >5%.
- Among structures that already clear gates, pick **highest credit/width**, then dollars. Prefer 1-lot credit ≥ $100. Do not take a 15-wide that is more than 1.5 pts worse on credit/width. Fire skip debit/width < 0.25. Fire is an Expert row only with opening flow; otherwise sleeve board.
- **Conf** is trade-success confidence. Two parts, both required:
  1. **naive POP** from **quoted Schwab delta** (integer %). Credits: `round(100 × (1 − |short Δ|))`. Debits: `round(100 × |long Δ|)`. Condor: `round(100 × (1 − max(|put short Δ|, |call short Δ|)))` and tag `wing` — true condor POP is lower. Missing delta → Conf blank, no Prime, Rec lots = 1.
  2. **book** = this sleeve’s rolling 20-trade win rate from the journal. If n < 20: `book n/a`.
  Write `79% naive · book n/a`. Never 74.0. Never an LLM guess.
- **Rec lots** from naive POP, then sleeve cap, then cash/BP. Never from Score. Never size-up to chase $10k.

```
budget = floor(sleeve_cap / 1-lot max loss)
Rec lots = max(1, round(naive_POP/100 × budget))
then min(Rec lots, floor(remaining cash-or-BP / 1-lot max loss))
```

Sleeve cap = Shield 1.0% of live equity · Fire 0.5% · Spike 0.25%. Rec lots is per row, not “take every row at that size.”

- **Score** is gate quality only: **80 / 65 / 50**. It is **not** P(win) and not confidence.
- **Data** is FULL / MIXED / THIN.
- **Prime** is not a column. Only Score 80 + Data FULL + Conf from quoted delta. **No Prime on THIN, unverified quotes, earnings-overlap, Crowded/Event Fire, or Crowded Spike.**
- **Sleeve** is Shield, Fire, Spike, or Hedge. At most **one Spike** row.

N-lot dollars (rec lots × 1-lot P/L) belong in **Notes**, never in the dollar columns. Thesis and sigma also go in Notes.

## What may not appear

- Invented IV, IV rank, OI, OI%, volume multiple, delta, or “est.”
- "Buy Put Credit"
- Rows that failed 1-sigma, width, earnings overlap (`earnings_date <= expiry` or date unknown), or book caps
- ETF/index rows unless the user allowed index hedge
- More than one live Fire per name
- More than one Spike row
- Spike tickers off the written map in `spike.md`
- More than 7 rows

## Empty table

On missing quotes or failed geometry, print:

```
### Expert Trade Table — YYYY-MM-DD
No executable rows. Empty table is the scan.
```

Then 3–5 **Assumptions in force** bullets (from the audit file). Optional catalyst-watch bullets **below**, clearly not rows.

## Score gates (all must hold)

**80** — FULL quotes; Shield in Normal/Elevated meeting that regime’s geometry (or Fire with VWAP + opening flow + not Crowded/Event; or Spike with sourced shock + VWAP + not Crowded); `expiry_date < earnings_date`; book caps OK.

**65** — MIXED, or **Calm Shield** (thinner credit/width by design), or Spike Crowded-but-quoted. Still `expiry_date < earnings_date`.

**50** — tradable but weak (wide-ish quotes, THIN sentiment). Never Prime.

Fail a hard gate → not in the table.
