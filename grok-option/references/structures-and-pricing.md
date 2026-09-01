# Structures and pricing

Defined-risk only. No naked shorts. No undefined-risk strangles as Core.

## Allowed structures

| Sleeve | Structure | Valid Action label |
|--------|-----------|--------------------|
| Shield | Put credit vertical (bull put) | Sell put credit |
| Shield | Call credit vertical (bear call) | Sell call credit |
| Shield | Iron condor only if **both** credit sides independently pass **that regime’s** geometry + quotes | Sell iron condor |
| Fire | Call debit vertical | Buy call debit |
| Fire | Put debit vertical | Buy put debit |
| Spike | Call or put debit vertical on the written shock map | Buy call debit or Buy put debit |
| Hedge | Index put debit or put debit spread | only if user said allow index hedge |

**Banned labels:** "Buy Put Credit", "Buy Call Credit", "Sell Debit", whale-copy one-liners.

**Scan all five.** On every allowed name/expiry run Schwab `structures` (or price each action with `vertical`). A scan that only looks at puts is incomplete. If both credit sides pass the same name/expiry, emit **one** iron condor, not a put row and a call row. Long stock in the Schwab book does **not** skip the call wing or the condor. Fire scores call debit and put debit separately (one Fire per name).

## Credit / debit math (no estimates)

Credit collected = **short bid − long ask** (Schwab `vertical --kind credit`).
Debit paid = **long ask − short bid** (Schwab `vertical --kind debit`).

If mids imply a net more than **5% better** than that conservative net, print the conservative net and add `worse-fill`. Do not average mids into the table.

Width = |short strike − long strike|.

Shield geometry is **regime-aware**. The Calm AND of “outside 1-sigma **and** ≥25% of width” is an empty set on a VIX-15 tape (live Schwab 2026-08-26: 14 names × 9–45 DTE, zero hits). That skip added no safety beyond “never trade in Calm,” which fights the barbell (few liquid Shields).

| VIX regime | Short (quoted) | Min credit/width | Hard skip |
|------------|----------------|------------------|-----------|
| Calm <16 | \|delta\| ≤ 0.22 **and** ≥ 0.80-sigma OTM | **0.12** | \|delta\| > 0.25 |
| Normal 16–22 | ≥ 0.90-sigma or \|delta\| ≤ 0.20 | **0.20** | \|delta\| > 0.25 |
| Elevated 22–30 | outside **1-sigma** | **0.25** | inside 1-sigma to chase width |

Use Schwab delta and that expiry’s ATM straddle ask for sigma. Max **4 Shield rows** in Calm, **one per sector**. Score Calm Shields **65**, not 80. Print them when they still have **edge**. Size from `book-and-target.md`. Manage **60–65% / 2.0×** — that is the winning-trade path. Max-loss 1:6 is what you get if you hold to max loss, which the plan forbids. Do not treat 1:6 as “this is a loser.”

**Among gate-clearing credits, pick the best edge, then dollars.**
1. `credit/width` first (conservative fill). That is the edge.
2. If any passing wing has 1-lot credit ≥ **$100**, drop the sub-$100 wings.
3. A wider wing only wins if its credit/width is within **1.5 pts** of the tighter wing. Extra width that adds dollars but cuts edge (TSLA 15-wide +$56 credit / +$444 max loss) is a skip, not a pick.
4. 21–45 DTE is a tie-break, not a drop of 46–60 that still expires before earnings.

Fire: debit/width **0.25–0.55**. Below 0.25 is a lottery long. Above 0.55 has no convexity. Rank remaining by long |delta| nearest 0.40. **Executable Fire** needs opening flow (Score 80 path). Missing flow → sleeve board only, not the Expert table. A 40-delta debit with no flow is not a Shield-quality row.

Max loss (1-lot, dollars):
- Credit vertical: `(width − credit) × 100`
- Debit vertical: `debit × 100`

## Chain requirements for an executable row

Same-day bid **and** ask on **both** legs from **Schwab**. Same expiry. Protective leg exists farther OTM (credit) or the short exists on the profitable side (debit). If any of that is missing: **not a row**.

Never fill IV, IV rank, OI, volume, or delta with a model guess. Schwab JSON may include those fields; blank if absent.

## Expiry

Shield: liquid weeklies and monthlies on mega-caps. Scan **14–60 DTE**. Prefer **21–45** as a tie-break when credit/width is within 1.5 pts. Do not drop a 46–60 DTE monthly that still expires before earnings if edge is as good or better. Hard skip <14. 60 DTE is the next monthly, not a LEAP.
Fire: match the thesis horizon; no 0–1 DTE lotto as a named sleeve without user ask. One live Fire per name.
Spike: 7–21 DTE debit on the mapped name; no 0–1 DTE; one Spike row. See `spike.md`.

**Earnings overlap (hard):** require a confirmed next earnings date and `expiry_date < earnings_date`. If the print’s calendar date is on or before expiry, or the date is unknown, **not a row**. Detail in `regime-and-signals.md`. Do not pick a later expiry to “wait out” 1-sigma if that later expiry crosses earnings — skip instead.

## Management (default)

See book file for size. Trade management:

- Take profit **60–65%** of credit (Shield) or 60–65% of max gain (Fire/Spike).
- Hard stop **2.0×** credit (Shield) or **50%** of debit paid (Fire/Spike).
- Ceiling **2.5×** credit only if thesis intact, DTE>14, and book still inside caps. Not for Spike.
- Spike: close if a sourced print reverses the shock, even before the 50% stop.
- If an open line now has `earnings_date <= remaining expiry`: close or roll to an expiry still strictly before the print. Do not hold through earnings.

## Liquidity

Prefer names with tight listed weeklies (mega-caps). Wide bid/ask on either leg → skip, do not “mid it in.”
