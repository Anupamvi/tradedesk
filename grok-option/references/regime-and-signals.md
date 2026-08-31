# Regime and signals

Load from SCAN after the regime line. Do not invent VIX, SMA, or IV.

## Regime buckets (VIX last regular close, else live)

| Bucket | VIX | Book |
|--------|-----|------|
| Calm | < 16 | Shield default at Calm geometry. Fire half-size if cheap IV, not Crowded, not name/theme Event, not Crisis. Cash only if geometry fails. |
| Normal | 16–22 | Shield default. Fire half-size when the same gates pass. |
| Elevated | 22–30 | Cut Fire. Shield only on liquid mega-caps, smaller size. Prefer cash over forced credits. |
| Crisis | > 30 | No new Shield, no ordinary Fire. Cash unless Sleeve D Spike gates pass (one 0.25% debit). Hedge only if user said allow index hedge. |

Today (2026-08-26): VIX ~15.5–15.7 → **Calm**. NVDA is a **theme Event for semis only**, not a tape-wide override.

If VIX 5-day change is unavailable, say so. Do not fabricate a 52-week range.

## Event overlay (independent of VIX)

Events are **scoped**. A name’s print does not zero the whole book.

| Scope | What counts | What it blocks |
|-------|-------------|----------------|
| **Name** | That ticker’s earnings/halt | All sleeves on that ticker (plus expiry-overlap firewall) until the next session after the print |
| **Theme** | Mega-cap print the index is pricing | **That sector only.** NVDA → semiconductors / AI-infra (NVDA, AMD, AVGO, MU, SMH, CRWV, INTC, MRVL, TSM, ORCL). Not healthcare, energy, banks, staples, or AAPL hardware. Score every other sector **today** |
| **Index** | FOMC **decision day**, CPI/PCE **the morning of**, NFP **the morning of**, Chair speech **the calendar date of** | Skip expiries that **land on** that date. No **index** Fire (SPY/QQQ) that day. Single-name Shield and Fire on non-theme names stay live if geometry + earnings-overlap pass |
| **Session empty** | Crisis, or FOMC **decision day** | Prefer empty table. A speech later in the week is not an empty week |

PCE already printed this session is **done**. Calendar Event is not a Spike. Weekly OPEX (non-monthly Friday) is a liquidity note unless it is also the Chair/FOMC date.

**Over-gate test.** Before skipping a name: if it is not in a name/theme freeze, and geometry + earnings-overlap pass, it is a row. A later-week index event is an expiry skip, not a book-off switch.

**When a cluster reopens.** Name/theme: that cluster only, next regular session after the print. Index Fire on SPY/QQQ: next session after the speech/print. The rest of the book never closed.

**Harvest** is an extra scan of the cluster that just printed (crush/expansion). It is not the first day the book is allowed to trade.

Shock tape (WTI/Brent, one sourced geo search) is part of the regime line. If a shock qualifies, follow `spike.md`. No shock → do not hunt one on X.

## 1-sigma short-strike rule

Sigma = Schwab ATM straddle ask on **that expiry** (call ask + put ask). Do not estimate. Regime table for how far OTM and how much credit lives in `structures-and-pricing.md`. Elevated still requires outside 1-sigma. Calm uses ≥0.80-sigma and \|delta\| ≤ 0.22 so the sleeve can exist when VIX is 15.

## Data flag

| Flag | Meaning | Prime? |
|------|---------|--------|
| FULL | Schwab same-day bid/ask both legs, expiry, sigma from quoted ATM straddle, earnings date sourced | Allowed if other gates pass |
| MIXED | Chain quotes yes; flow or IV rank or OI from a file/CSV, not invented | Allowed; no fake OI% |
| THIN | Web/X only, or one leg unquoted, or IV/OI estimated | **No Prime.** Scores may be 65/50. Empty table preferred if quotes missing |

## Price / VWAP (Fire only)

Fire prefers the underlying on the same side of VWAP (or clearly reclaiming it) as the debit. If VWAP is unavailable, say unverified and **do not Prime** — do not skip. Missing opening flow is not a skip; it blocks Score 80.

## Earnings firewall (expiry must not overlap the print)

Hard skip for **every** sleeve (Shield, Fire, condor, hedge). The 7 / 14 / 15-day windows are retired. A 45-DTE that is still alive on earnings day is an earnings trade even if the print is “far.”

Source a **confirmed** next earnings date (and BMO/AMC when published). Web or an attached calendar. Do not guess. “Late September” or missing date = **not a row**.

**Overlap (skip):** `earnings_date <= expiry_date`  
The option’s life is `[scan_date, expiry_date]`. If the print’s calendar date sits on or before expiry, the contract can eat the surprise and the IV crush/expansion. Same-day counts (AMC on expiry Friday still pumps that session).

**Clear (allowed):** `expiry_date < earnings_date`  
The listed expiry is a full calendar date **before** the print. A weekly that dies before earnings is the only legal way to trade a name that reports soon.

BMO/AMC is a timing note, not a loophole. If timing is unknown, still require `expiry_date < earnings_date`.

Unknown earnings date → skip (cannot prove no overlap).  
Failed overlap → Watch only, tag `earnings-overlap`, never a table row.

Open book (MANAGE): if an open line now has `earnings_date <= remaining expiry`, close or roll to an expiry that is still strictly before the print. Do not hold through.

Catalyst-watch names may be listed **below** the table, not in it.
