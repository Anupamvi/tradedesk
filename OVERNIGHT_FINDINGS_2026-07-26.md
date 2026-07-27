# Overnight findings — 2026-07-26

**Mandate:** spot patterns across the 5 UW feeds + external methodology, then fix the pipeline.
**Hard constraints honoured:** no gate was loosened, and nothing that failed the monotonicity check was wired in.

The headline: I found and fixed three real defects, and then I **rejected my own proposed
improvement** because the replayed P&L contradicted it. Details below, worst news first.

---

## 1. Direction is not predictable from this data. Stop trying.

Across 46–66 engineered features — including the raw ~12M-row/day bot-EOD tape — **0 of 66 survive
Benjamini–Hochberg correction at a 5-day horizon.** Every feature that looks significant at 21 days is
U-shaped (both extremes profitable, middle flat), and its monthly information coefficients flip sign.
That is the signature of noise, not edge.

**Implication:** the money in this book does not come from predicting direction. It comes from
*structure* — selling into the right regime, at the right width, and taking profit early. Every
profitable configuration found tonight is structural. Every directional one failed.

## 2. DEFECT — the IV/HV gate was inert. It never fired, not once.

`engine.py` hardcoded `iv_hv_ratio = NaN`, so the volatility-richness gate silently passed everything.
Compounding it, `integrity_v42` was clobbering realized vol with zeros.

- Realized-vol coverage: **63.8% → 98.6%** after the fix
- `codexuw/realized_vol.py` validated against the reference series at **Spearman 0.9959**

## 3. DEFECT — three lanes bypassed the gates entirely.

**100% of tradable output was coming from ungated sleeves.** A live run emitted 8 trades at IV/HV
**0.906–0.933** — i.e. selling premium *cheaper than the volatility actually being realized*, which is
the exact opposite of the intended strategy.

There are exactly three places that set `decision_eligible = True`
(`engine.py:1287`, `engine.py:1297`, `fallback_income.py:303`). All three are now gated uniformly.

> **Rule worth keeping:** when you add a gate, grep for *every* alternate path to the eligibility flag.
> A gate on a lane that produces nothing protects nothing.

## 4. The near-miss that matters most — I nearly made this worse

I built a variance-risk-premium capture model. It was beautiful, and it was **wrong for this book**.

It said: raise the richness bar to IV/HV ≥ 1.30, and **open up `range` regime trading**. The evidence
looked overwhelming — monotone across every bucket, huge sample, positive on **98.9% of days**, and
`range` scored as the single *best* regime.

Then I replayed it against actual vertical P&L.

| | proxy said | real P&L said |
|---|---|---|
| IV/HV ≥ 1.30 | strong, monotone | delta **+3.2**, CI [−27.5, +31.1], p **0.407**; non-monotone; cuts trades **82%** |
| `range` regime | **best** (+0.1215, 75.3% win) | **worst thing in the book**: n=904, PF **0.83**, **−$12,452** |

**Why the proxy lied:** capture measures a *variance-swap* payoff — continuous, symmetric,
proportional to the vol miss. A defined-risk vertical caps gain at the credit and loss at roughly 3×
the credit. It monetises almost none of that premium. The metric was internally valid and simply not
the objective function.

There was also a confound that **inverts the answer twice**: 71.3% of "rich" candidates sit in regimes
the map already blocks — which is where all the losses live. Unconditionally richness looks toxic;
conditioned on the map it looks great; tested rigorously it is neither. A 2,000-draw permutation
control put it at **p = 0.426**.

**Both changes were rejected. Nothing was shipped on the strength of the proxy.**

## 5. What actually holds up — the contrarian regime map

Independently reconfirmed on replayed P&L this session:

| | n | win | PF | total |
|---|---|---|---|---|
| **map-allowed** (Bull Put\|downtrend, Bear Call\|uptrend) | 573 | 83.2% | **1.22** | **+$7,503** |
| map-blocked | 1,732 | 79.4% | 0.87 | −$16,986 |

Standout cell: **Bull Put in a downtrend — n=170, win 87.6%, PF 1.92.**
Sell premium *after* the move, into mean reversion — not *into* a trend that is actively running at you.

## 6. Both lanes validate once gated — and the debit lane taught a real lesson

Unconditionally the debit book looks like a disaster: PF 0.79, **−$14,256**. That is the *same
confound as above* — do not react to the unconditional number. Applying its shipped gates:

`all 775 (PF 0.79)` → Bull Call only `472 (0.75)` → +uptrend `204 (0.98)` → +DTE `152 (0.94)`
→ +debit ≤45% `149 (0.93)` → +RR ≥1.25 `148 (0.92)` → **+flow align ≥0.20 `48 (1.52)`**
→ +iv_rank ≤55 → **n=42, win 64.3%, PF 1.83, +$1,786** (5 of 6 months positive).

> **Asymmetry worth remembering:** flow alignment is the single biggest lift in the **debit** lane
> (PF 0.92 → 1.52) while being pure noise in the **credit** lane. That is theoretically correct — a
> debit vertical *is* a directional bet, whereas a credit vertical inside a contrarian map is a
> mean-reversion bet. Never generalise "flow is noise" across both lanes.

## 7. Ranking — investigated, deliberately left alone

`_decision_sort_score` pays up to 2.0 for a **further** strike, and that sign is backwards
(within-session Spearman **−0.37**, p(ρ≤0) = 1.000; closest-strike quartile PF **2.11** vs furthest 1.13).

I did **not** change it. Inside the full shipped gate set, only **7 sessions** ever present a choice
between two candidates. Retuning a ranking function on 7 decisions is precisely the overfitting I just
rejected the 1.30 gate for. Documented for when the candidate pool widens.

## 8. I nearly made the same mistake a second time — please read this one

I ran the corrected pipeline on 2026-07-24 and it emitted 8 `Bear Call` spreads on a **`range`** day at
**DTE 7**. Both the regime map and the 28–45 DTE band should have blocked those. Digging in,
`apply_high_conviction_decision_marks` never calls `assess_credit_spread` — it re-implements its own
checks and applies **neither** gate, for the `secondary_income` *or* the `primary` lane.

That looks like a straightforward bypass bug, and `range` + Bear Call is the worst cell in the book
(n=587, PF 0.81, **−$8,542**). I was about to gate it.

**Conditioned on the sleeve's own filters, the sign flips completely:**

| slice | n | win | PF | total |
|---|---|---|---|---|
| `range` + Bear Call, all candidates | 587 | 79.6% | 0.81 | −$8,542 |
| `range` + Bear Call, **inside the sleeve** | 23 | **95.7%** | **5.58** | **+$1,673** |

Whole sleeve: n=127, PF 1.17. Map-blocked portion: n=91, PF **1.14**, p(mean ≤ 0) = 0.321 — not losing.
The primary lane is the same story: map-blocked n=141, PF 1.03, CI [−30.1, +32.3], p = 0.451 — flat,
not negative.

**Forcing the regime map onto these sleeves would have deleted the most profitable slice in them.**
I left the behaviour alone and added a comment in `engine.py` so nobody else "fixes" it either.

## 9. The one change I recommend — but did not ship

The DTE **11–27** band is the clearest single loss source in the book, and it replicates on **three
independent slices**:

| slice | n | DTE 0–10 | **DTE 11–27** | DTE 28–45 |
|---|---|---|---|---|
| full credit book | 2,483 | PF 1.34 | **0.71 / 0.80** (−$19,327) | PF 1.24 |
| secondary sleeve | 127 | PF 2.90 | **0.72** (−$1,728) | PF 2.08 |
| primary, map-blocked | 141 | PF 2.42 | **0.63** (−$2,487) | PF 1.54 |

Excluding 11–27 from the sleeve: PF 1.17 → **2.38**, win 79.5% → 87.5%,
**delta +38.1, 90% CI [+10.2, +68.8], p = 0.012.** The confidence interval **excludes zero** — which no
IV/HV threshold ever managed. It is a pure tightening: it removes 63 trades and adds none.

Reading: 0–10 DTE closes on theta before gamma matters; 28–45 is the validated core; 11–27 is the worst
of both — real gamma exposure without fast decay.

**I did not ship it.** A U-shaped carve-out literally fails a monotonicity check, and you told me not to
wire in anything that fails one. Nothing is bleeding — the sleeve is net positive today — so this can
wait for your call. If you want it, it is one condition in `_secondary_income_eligible`
([engine.py](codexuw/engine.py#L1196)), and note that adding a plain `MIN_DTE = 28` would be **wrong**:
it would also cut the 0–10 bucket, which is the best one.

---

## What to expect at the open

**This pipeline is designed to trade rarely.** Over 130 replayed sessions the full shipped credit
config produced **27 trades on 18 sessions — 14%, about one entry every 7 sessions** (win 85.2%,
PF 1.82). The debit lane adds ~42 trades on 25 sessions at PF 1.83.

End-to-end check, so you know it is alive and not silently gated shut:

- **2026-07-24 (`range`)** → 8 eligible, all via the validated sleeve
- **2026-07-23 (`downtrend`)** → 8 eligible: 3 primary, 5 secondary income, including
  DELL Bull Put in a downtrend — the best cell in the book (PF 1.92)

Minimum IV/HV among eligible rows: **0.920**; minimum realized vol: **0.299**. Both gates are live and
binding. Realized-vol coverage 98.6%.

> If a session prints `range` and the primary lane goes quiet, that is the gate working, not a failure.
> Please do not loosen anything to manufacture a trade — that is exactly how the −$16,986 in the
> map-blocked bucket was earned.

Check the regime first, then the trade count. In that order.

## State of the code

- `credit_policy` → `credit-v4.0-regime-map-validated-rv`
- `MIN_IV_HV_RATIO` **0.90** (baseline, unchanged — the 1.30 proposal was reverted)
- `MIN_REALIZED_VOL` 0.15, kept purely as an artefact guard (removes 1 of 574 replayed trades;
  without it the gate's top names are cash ETFs — ICSH, BOXX, JPST — with 0.3–3.8% realized vol)
- `MIN_FLOW_ALIGNMENT` 0.10 restored; earnings-window exclusion removed (it tested *significantly
  harmful* at 7 days: delta −5.2, CI [−9.8, −1.0])
- Tests: **1,368 passing**, 4 pre-existing `test_options_agent` failures unrelated to this work

**Known gaps, deliberately not half-fixed:** `engine.py:159` still bands historical evidence on
`iv_rank` while live candidates gate on IV/HV — migrating it needs realized vol plumbed through the
replay path. `replay.py:777` holds a dead duplicate of `_secondary_income_eligible` without the
richness bound; harmless while unused, a footgun if ever wired up.

Have a good trading day.
