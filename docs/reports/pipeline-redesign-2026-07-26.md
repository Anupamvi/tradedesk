# Codex Daily V4 — Pipeline Review and Redesign

**Date:** 2026-07-26
**Mandate:** "review and redesign the entire pipeline as needed; the goal is to generate consistent profitable trades; ideally $10K/month goal."
**Hard constraint carried forward:** "I dont want pipeline to be losened to generate loses."

---

## 1. Headline

The review found a **fourth structural defect, and it is larger than the three entry-gate defects combined.**

**The exit policy was destroying the edge.** Every calibration, guard, payoff route, and confidence
model in this pipeline was fitted to P&L produced by a value-destroying exit rule. The guard's real
job had quietly become "find the 0.88% of trades that can survive a −$25/trade handicap." That is the
honest answer to "why does this pipeline never produce a trade."

Fixing the exit and re-selecting on top of it produces a rule set that is **profitable out-of-sample
in 4 of 5 rolling folds and in 6 of 6 months**, while being **strictly tighter** on entry than what
shipped before.

---

## 2. Defect D — the exit policy

Exit-reason decomposition across 3,395 evaluated replay trades under the shipped policy
(take profit at 60% of credit, hard stop at 2.0× credit):

| exit_reason | n | share | avg $ |
|---|---:|---:|---:|
| profit_target | 1,670 | 49.2% | **+101.20** |
| stop_loss | 1,588 | 46.8% | **−159.77** |
| expiry_settlement | 137 | 4.0% | −6.70 |

Wins are 0.63× the size of losses at a ~51% hit rate. Break-even requires a 62.5% target-hit rate;
the actual rate is 51.3%. **This is structurally negative expectancy regardless of entry quality.**

### The 13-point exit grid (all 3,395 rows, re-simulated)

| config | n | PF | win | avg $ | total $ |
|---|---:|---:|---:|---:|---:|
| **SHIPPED — tp60 / sl2.0× / slip10%** | 3395 | **0.676** | 51.5% | −25.22 | −85,633 |
| tp50 / sl3.0× / slip10% | 3375 | 0.733 | 65.0% | −22.11 | −74,618 |
| tp50 / sl4.0× / slip10% | 3371 | 0.780 | 69.4% | −18.44 | −62,152 |
| **tp50 / no stop / slip10%** | 3368 | **0.907** | 72.5% | −7.03 | −23,674 |
| hold to expiry / slip10% | 3325 | 0.872 | 64.2% | −12.73 | −42,324 |
| tp50 / no stop / slip05% | 3369 | 1.002 | 73.6% | +0.16 | +527 |
| tp50 / no stop / slip00% | 3369 | 1.140 | 75.2% | +9.30 | +31,341 |

Strict time split (train on the first 60% of sessions, test on the last 40%):

| config | TRAIN PF | TEST PF |
|---|---:|---:|
| shipped | 0.679 | 0.672 |
| tp50 / no stop | 0.814 | **1.089** |
| tp50 / no stop / slip05% | — | **1.217** |

**Every configuration carrying a hard stop underperformed the same configuration without one**, in
sample and out. Every time-stop variant made things worse. Removing the stop is worth roughly
**+$18.40 per trade** across the universe.

**Maximum risk per position is unchanged.** These are defined-risk verticals; the spread width already
caps the loss. The 2× stop was a discretionary overlay that converted recoverable noise into realized
losses — it was never a risk control.

---

## 3. Defect A confirmed from a second angle

The expected-move buffer (`distance / expected_move >= 0.75`) was previously identified as jointly
unsatisfiable with the credit band (corr(credit%, expected-move ratio) = **−0.734**). It is worse than
that: as a walk-forward *selector* it scores pooled OOS **PF 0.818**, 0 of 5 folds — **worse than
applying no selection at all (0.935)**. Within the validated core population, `em >= 0.75` matches
**n = 0** rows.

It is not merely strict. It is anti-predictive. It has been removed.

## H1 (adverse selection) — tested and rejected

`corr(flow_align, pnl_1x) = +0.0156`; flow quintiles are monotone in the correct direction
(Q1 contra PF 0.647 → Q5 aligned 0.737); the live credit flow gate passes PF 0.732 vs 0.600 for
failures. The flow gate is directionally correct but weak. It is not the bug.

---

## 4. The validated rule

> **Credit vertical · DTE ≥ 28 · credit 25–30% of width · take profit at 50% · no hard stop**

| metric | value |
|---|---|
| unbiased full sample | n = 253, PF **1.475**, +$6,411 |
| rolling OOS (5 folds) | n = 179, PF **1.686**, win 86.0%, **4/5 folds ≥ 1.25** |
| per-fold PF | 4.80, 0.89, 1.44, 2.59, 7.41 |
| **profitable months** | **6 / 6** ($198, $297, $1,876, $1,002, $982, $2,056) |

The same rows under the shipped tp60 / sl2.0× exits score **PF 0.662**.

### Gate keep/drop decisions (OOS, layered on the core)

| gate | n | PF | folds | verdict |
|---|---:|---:|---:|---|
| none (core only) | 179 | 1.686 | 4/5 | — |
| **regime map** | 71 | **4.928** | **5/5** | **KEEP** |
| regime inverted | 108 | 1.119 | 4/5 | confirms the map is right |
| **iv_rank ≥ 30** | 103 | **2.094** | 4/5 | **KEEP** |
| iv_rank ≥ 40 | 84 | 2.170 | 4/5 | high-confidence tier |
| iv_rank ≤ 55 | 124 | 1.373 | 3/5 | **DROP** — it hurts |
| **flow ≥ 0.10** | 80 | **2.082** | 4/5 | **KEEP** |
| quote width ≤ 0.05 | 135 | 1.936 | 4/5 | optional |
| quote width ≤ 0.20 | 179 | 1.686 | 4/5 | non-binding |
| **expected move ≥ 0.75** | **0** | — | — | **REMOVE** |
| expected move ≥ 0.50 | 20 | 0.821 | 2/3 | **REMOVE** |

### Robustness

- Threshold sensitivity is flat: DTE 21–35 → PF 1.23–1.39; quote width 0.06–0.35 → 1.36–1.41;
  premium 0.15–0.30 → 1.42–1.84. No cliff edges, so this is not a fitted point.
- Component ablation: **the DTE gate is load-bearing** (dropping it → PF 1.093, 2/5 folds).
  Quote width is redundant. Every gate alone scores ~0.95–1.01 — the edge is in the combination.
- Slippage monotone: 10% → 1.257, 5% → 1.380, 0% → 1.507.
- **Truncation bias corrected.** 110 rows were dropped where expiry fell outside the data window.
  Uncorrected, July showed PF ∞ / 100% win — pure survivorship, because winners resolve early via the
  target while losers run to expiry beyond the window. All headline numbers above are post-correction.

### Regime breakdown of the core

| slice | n | PF | avg $ |
|---|---:|---:|---:|
| Bear Call \| downtrend | 53 | 2.293 | +46.52 |
| Bear Call \| range | 54 | 1.385 | +20.67 |
| Bear Call \| uptrend | 54 | 1.411 | +22.67 |
| Bull Put \| downtrend | 15 | 1.837 | +40.60 |
| Bull Put \| range | 32 | 0.695 | −28.65 |
| Bull Put \| uptrend | 45 | 1.962 | +42.51 |

`Bear Call | range` (PF 1.385, +$1,116) is a documented but currently unexercised opportunity —
range days are presently unreachable by the regime map. Flagged, not acted on.

---

## 5. This is a tightening, not a loosening

| change | direction |
|---|---|
| DTE floor 7 → **28** | **tighter** |
| new IV-rank floor ≥ 30 | **tighter** (new gate) |
| credit band 25–30% of width | unchanged |
| flow alignment ≥ 0.10 | unchanged |
| quote width ≤ 0.35 | unchanged |
| regime map | unchanged |
| max risk per position | **unchanged** (spread width) |
| expected-move buffer removed | proven harmful: n = 0 and anti-predictive |
| hard stop removed | proven harmful: −$18.40/trade, worse in every paired test |

The only two removals are gates the data shows were destroying money. Nothing was relaxed to increase
trade count.

---

## 6. Honest answer on $10K/month

| config | $/mo @ 1 contract | peak concurrent risk @1x | max DD @1x | contracts for $10k | peak buying power |
|---|---:|---:|---:|---:|---:|
| core, uncapped | $1,068 | $14,717 | −$2,292 | 9.4 | **$137,745** |
| top 3 per session | $868 | $11,015 | −$2,696 | 11.5 | $126,915 |
| top 5 per session | $1,012 | $13,969 | −$2,455 | 9.9 | $138,027 |

**The stated $15,000 risk budget buys approximately $1,089 per month.**

$10K/month requires roughly **$138,000 of peak buying power** and would carry a **−$21,449 drawdown**
at that scale. Median hold is 14 days; 84 of 130 sessions produce at least one trade.

**The $10K target is a capital problem, not an edge problem.** No amount of gate tuning closes that
gap, and any attempt to close it by loosening entry criteria would reproduce the PF ~0.5 loss machine
documented in earlier audits.

---

## 7. Code changes

| file | change |
|---|---|
| [codexuw/credit_policy.py](../../codexuw/credit_policy.py) | version → `credit-v2.0-dte28-ivrank-no-distance-and-gate`; `MIN_DTE` 7 → 28; new `MIN_IV_RANK = 30.0`; new `PROFIT_TAKE_PCT = 0.50`, `USE_HARD_STOP = False`; deleted `MIN_DISTANCE_EXPECTED_MOVE_RATIO`; `credit_spread_edge_lane` re-keyed on IV rank; high-confidence tier now requires IV rank ≥ 40 |
| [codexuw/engine.py](../../codexuw/engine.py) | `replay_quality_pattern` now gates on credit band **and** DTE **and** IV rank; distance retained for reporting only; band tag → `validated_credit25_30_dte28_ivrank30` |
| [codexuw/replay.py](../../codexuw/replay.py) | `simulate_spread_exit` accepts `stop_loss_mult=None` (no stop) on both credit and debit branches; `run_replay` defaults → `profit_take_pct=0.50`, `stop_loss_mult=None`; CLI `--stop-loss-mult` defaults to none; removed the residual bull-put distance gate |
| [codexuw/goal_shadow.py](../../codexuw/goal_shadow.py) | resolver mirrors the stopless exit contract |
| [codexuw/daily_v4.py](../../codexuw/daily_v4.py) | credit OCO instruction: take profit at 50% of credit; stop leg removed with an explicit statement that risk is defined by the spread width |

Tests: **232 passed.** Fixtures that encoded the old policy (DTE 9–23, missing `iv_rank`) were updated
to policy-compliant values; the two tests that asserted the expected-move buffer were rewritten to
assert the DTE and IV-rank contract instead.

---

## 8. Remaining work

1. Regenerate the evidence base Jan 2 → Jul 23 under the new guard **and** the new exits (~2 h),
   repackage as `codexdaily_v4_edge_history_v4_*`, bump `EDGE_HISTORY_NAMESPACE`.
2. Recalibrate confidence and payoff models on the regenerated base and confirm lanes validate rather
   than returning `NO_VALIDATED_LANES`. **Every existing calibration is fitted to the broken exit rule
   and must be considered void until this is done.**
3. Re-run daily V4 on 2026-07-23 and report Execute/Scout counts.
4. Evaluate whether `Bear Call | range` should be added to the regime map (PF 1.385 OOS).

Until step 2 completes, the new policy is validated by the research harness but the live confidence and
payoff numbers shown on tickets remain unrepresentative.
