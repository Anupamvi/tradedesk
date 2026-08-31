# grok-option — YYYY-MM-DD

**Equity:** … | **Bar:** $10k/month
**Regime:** VIX …
**Shock watch:** none | X flag (one line) | sourced, no row | sourced, Spike row
**Data:** FULL / MIXED / THIN | **Calendar:** …

## Expert Trade Table — executable

Max profit $ and Max loss $ are **1 lot**. Rec lots scales from Conf, then sleeve cap, then cash/BP.

Buy/Sell cells name every leg: `Buy 225 Put` / `Sell 240 Put`. Condor: `Buy 445 Put + Buy 550 Call` / `Sell 460 Put + Sell 540 Call`.

| Ticker | Sleeve | Action | Expiry | Buy (long) | Sell (short) | Max profit $ | Max loss $ | Rec lots | Score | Conf | Data |
|--------|--------|--------|--------|------------|--------------|--------------|------------|----------|-------|------|------|
|  |  |  |  |  |  |  |  |  |  |  |  |

Notes:

- TICKER — …

If none: **No executable rows. Empty table is the scan.**

## Shock watch

| Item | Source | Status |
|------|--------|--------|
| … | X / web | none / flagged / no row / Spike row |

## Sleeve board (after the table)

A put-only board is incomplete.

| Structure | In table | Also cleared, not tabled |
|-----------|----------|--------------------------|
| Sell put credit | … | … |
| Sell call credit | … | … |
| Sell iron condor | … | … |
| Buy call debit | … | … |
| Buy put debit | … | … |
| Spike | … | … |

## Assumptions in force

- …
- …
- …
- …
- …

## Book

Aggregate max loss … / cap …
Theme …
Fire live … Spike live …
Cash is / is not the position today.

## Watch

- earnings-overlap: …
