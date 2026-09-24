# Regime and signals

Load from SCAN after the regime line. Do not invent VIX, SMA, or IV.

## Regime buckets (VIX last regular close, else live)

| Bucket | VIX | Book |
|--------|-----|------|
| Calm | < 16 | Shield default at cheap-vol geometry (0.12 / 0.22Δ / 0.80σ). Fire half-size if cheap IV, not Crowded, not name/theme Event, not Crisis. Cash only if geometry fails. |
| Normal | 16–22 | Same cheap-vol Shield as Calm until conservative 0.20 prints. Fire half-size when the same gates pass. Score 65. |
| Elevated | 22–30 | Cut Fire. Shield only on liquid mega-caps, smaller size. Prefer cash over forced credits. |
| Crisis | > 30 | No new Shield, no ordinary Fire. Cash unless Sleeve D Spike gates pass (one 0.25% debit). Hedge only if user said allow index hedge. |

Today (2026-08-26): VIX ~15.5–15.7 → **Calm**. NVDA is a **theme Event for semis only**, not a tape-wide override.

If VIX 5-day change is unavailable, say so. Do not fabricate a 52-week range.

## Event overlay (independent of VIX)

Events are **scoped**. A name’s print does not zero the whole book.

| Scope | What counts | What it blocks |
|-------|-------------|----------------|
| **Name** | That ticker’s earnings/halt **or** a sourced dated catalyst in the option’s life (deliveries, product unveil, vote, named launch) | Earnings/halt: all sleeves on that ticker until the next session after the print. Dated catalyst: park **the threatened wing** (call event → no call credit / no IC; put event → no put credit / no IC). Unthreatened vertical may still Expert if geometry + overlap pass |
| **Theme** | Mega-cap print the index is pricing | **That sector only**, from this session’s tape (e.g. a chip/AI-infra print → semiconductors / AI-infra, not healthcare, energy, banks, staples, or hardware). Score every other sector **today**. Do not keep a saved ticker deny-list |
| **Index** | FOMC **decision day**, CPI/PCE **the morning of**, NFP **the morning of**, Chair speech **the calendar date of** | Skip expiries that **land on** that date. No **index** Fire (SPY/QQQ) that day. Single-name Shield and Fire on non-theme names stay live if geometry + earnings-overlap pass |
| **Session empty** | Crisis, or FOMC **decision day** | Prefer empty table. A speech later in the week is not an empty week |

PCE already printed this session is **done**. Calendar Event is not a Spike. Weekly OPEX (non-monthly Friday) is a liquidity note unless it is also the Chair/FOMC date.

**Over-gate test.** Before skipping a name: if it is not in a name/theme freeze, and geometry + earnings-overlap pass, it is a row. A later-week index event is an expiry skip, not a book-off switch. A dated catalyst parks **the threatened wing**, not the whole book and not the other vertical.

**When a cluster reopens.** Name/theme: that cluster only, next regular session after the print. Index Fire on SPY/QQQ: next session after the speech/print. The rest of the book never closed.

**Harvest** is an extra scan of the cluster that just printed (crush/expansion). It is not the first day the book is allowed to trade.

Shock tape (WTI/Brent, one sourced geo search) is part of the regime line. If a shock qualifies, follow `spike.md`. No shock → do not hunt one on X.

## 1-sigma short-strike rule

Sigma = Schwab ATM straddle ask on **that expiry** (call ask + put ask). Do not estimate. Regime table for how far OTM and how much credit lives in `structures-and-pricing.md`. Elevated still requires outside 1-sigma. Calm **and** Normal use ≥0.80-sigma and \|delta\| ≤ 0.22 so the sleeve can exist when VIX is 15–18.

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

Source a **confirmed** next earnings date (and BMO/AMC when published). **Company IR beats aggregator estimates.** Web or an attached calendar. Do not guess. Do not keep an Expert row on “est. Oct 27” the day after IR set Oct 13. “Late September” or missing date = **not a row**.

**Name calendar (before Expert).** For every geometry-pass candidate, one web search covering `(scan_date, expiry]`: IR earnings, deliveries, product unveil, shareholder vote, named launch. Seasonality folklore is not a veto. A sourced dated event parks that wing. 2026-09-16 miss: TSLA Oct-16 405C Expert while Tesla dated Roadster unveil **Oct 1** and Q3 deliveries sit in the life.

**Overlap (skip):** `earnings_date <= expiry_date`  
The option’s life is `[scan_date, expiry_date]`. If the print’s calendar date sits on or before expiry, the contract can eat the surprise and the IV crush/expansion. Same-day counts (AMC on expiry Friday still pumps that session).

**Clear (allowed):** `expiry_date < earnings_date`  
The listed expiry is a full calendar date **before** the print. A weekly that dies before earnings is the only legal way to trade a name that reports soon.

BMO/AMC is a timing note, not a loophole. If timing is unknown, still require `expiry_date < earnings_date`.

Unknown earnings date → skip (cannot prove no overlap).  
Failed overlap → Watch only, tag `earnings-overlap`, never a table row.

Open book (MANAGE): if an open line now has `earnings_date <= remaining expiry`, close or roll to an expiry that is still strictly before the print. Do not hold through.

Catalyst-watch names may be listed **below** the table, not in it.
