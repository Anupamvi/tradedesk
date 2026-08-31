# Spike path (Sleeve D)

Ride a **sourced** shock with a defined-risk debit. Empty Spike row is the default.

**Shock watch is its own lane.** X may **find** a geo/oil/kinetic event for review. X may not print a Spike **row**. Missing this lane is an incomplete scan.

**Not crude-only.** Oil crash, oil spike, Strait/energy kinetic, defense kinetic. Crude ±5% session is one trigger. A named-outlet mine/strike/blockade or state-on-state attack is another. Broad risk-off is cash. Fed calendar is not Spike.

## Shock watch (review, not executable)

Run **before** the universe is locked, every SCAN:

1. **X finder** — one keyword + one semantic pass for Hormuz, Red Sea, blockade, mine, strike, invasion, missile, oil crash/spike, named kinetic. This is how a same-day geo print gets noticed.
2. **Web check** — if X flags anything, source a named outlet (Reuters, AP, WSJ, Bloomberg, EIA, exchange). Quote the fact or write `web: not sourced`.
3. **Print on the card** under Watch as **Shock watch**, even when Spike is no. Status is one of: `none` | `X flag, web not sourced` | `sourced, no ±5%/no map move` | `sourced + map → score Spike`.

Off-map events (tariff, halt, name not in the table) stay on Watch. Do not invent a ticker. Fade only if the user said fade.

## When a Spike **row** may exist

All of the following, in order:

1. **Web source** — named outlet with a timestamp. X finder is not enough.
2. **Measurable shock** from the table below.
3. **Written map** — only tickers on that row.
4. **Price already moving** — mapped name on the shock side of VWAP, or a clear same-session reclaim.
5. **X** — talking about the *same* shock (finder or confirm). Crowded → 0.25%, no Prime.
6. **Quoted debit vertical** — conservative debit = long ask − short bid.
7. **Earnings** — confirmed date and `expiry_date < earnings_date`.

If any step fails: Watch only. Not a table row.

## Shock table (only these)

| Shock | Evidence (need one) | Map (stocks) | Ride (debit) |
|-------|---------------------|--------------|--------------|
| Oil crash | WTI or Brent ≤ **−5%** session, or ≤ **−8%** in two sessions | XOM, CVX, COP | Buy put debit |
| Oil spike | WTI or Brent ≥ **+5%** session, or ≥ **+8%** in two sessions | XOM, CVX, COP | Buy call debit |
| Strait / energy kinetic | Named closure, mine, strike, or blockade of Hormuz, Red Sea, or a major oil chokepoint | XOM, CVX, COP | Debit with the oil move (puts if crude down, calls if crude up) |
| Defense kinetic | Named state-on-state strike or invasion, same-day, sourced | LMT, RTX, GD | Buy call debit |
| Broad risk-off | Not a Spike by itself | — | Cash. Index put only if the user said allow index hedge |

Do not invent other maps (miners, uranium, bakken juniors, “the tweeted ticker”). One shock type per scan. If oil and kinetic both print, **energy map wins** (one row). Defense map only when there is no qualifying oil print.

Calendar Event (PCE, CPI, FOMC, NFP, Warsh) is **not** a Spike. Do not ride the Fed.

## Direction

Ride the already-printed wave. Do not fade unless the user said fade. Fade is not Sleeve D.

## Size and count

- **One Spike row per scan.** One live Spike name in the book.
- Size = **lotto 0.25%** of equity (max loss). Crowded X → stay 0.25% and **no Prime**. Crowded + THIN → skip.
- Under **$50k**: Spike is off (same as no lotto).
- Spike counts toward aggregate 10–12% and toward the energy or defense **theme** cap.
- Do not also Fire the same name the same day. Spike replaces Fire on that name.
- **No new Shield** on mapped shock names that session.

## Crisis carve-out

VIX Crisis still kills new Shield and ordinary Fire. Spike may still print **one** 0.25% debit if this file’s gates pass. No sourced shock → Crisis is still cash, not a fake Spike.

## Structure and management

Debit vertical only. Typical 7–21 DTE, still `expiry < earnings`. No 0–1 DTE. No naked, no straddle-as-Spike.

Take **60–65%** of max gain. Stop **50%** of debit paid. If a sourced print **reverses** the shock (strait reopened, ceasefire, crude mean-reverts through VWAP against the debit), close — do not wait for the 50% stop.

## Prime

Spike Prime only if Score 80 + Data FULL + not Crowded + web source cited in Notice. No Prime on THIN, unverified crude prints, earnings-overlap, or Crowded.

## Notice tags

`shock: oil-crash|oil-spike|strait|kinetic` plus `X: confirm` or `X: Crowded`. Cite the web source in one short clause. No tweet theater.
