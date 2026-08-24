# Strategy × Regime × Macro Review — 2026-07-26

Full review of every strategy family against market trend and macro state, on the
regenerated evidence base (`codexdaily_v4_edge_history_v4_2026-07-26`,
6,948 rows / 139 sessions / 2026-01-02 → 2026-07-23, truncation-corrected).

Scripts: `scripts/strategy_macro_review.py`, `scripts/strategy_policy_matrix.py`.

---

## 1. Headline: the credit regime map was inverted

On the full evaluated universe (3,258 truncation-corrected rows):

| slice | n | PF | win | avg $ | total $ | shipped map |
|---|---:|---:|---:|---:|---:|---|
| **Bull Put \| downtrend** | 176 | **1.813** | 86.9% | +35.77 | **+6,296** | BLOCKED |
| Bear Call \| uptrend | 424 | 1.028 | 81.4% | +1.92 | +814 | BLOCKED |
| Bull Put \| uptrend | 462 | 0.944 | 79.0% | −4.19 | −1,936 | permitted |
| Bear Call \| downtrend | 412 | 0.838 | 80.3% | −12.26 | −5,051 | permitted |
| Bear Call \| range | 587 | 0.814 | 79.6% | −14.55 | −8,542 | blocked |
| Bull Put \| range | 422 | 0.856 | 78.4% | −11.34 | −4,784 | blocked |

Head-to-head walk-forward:

| credit map | n | PF | win | total $ | OOS folds ≥1.25 |
|---|---:|---:|---:|---:|---|
| **CONTRARIAN** (BP→down, BC→up) | 600 | **1.196** | 83.0% | **+7,110** | **3/4** |
| TREND-FOLLOWING (shipped) | 874 | 0.894 | 79.6% | −6,987 | 1/4 |

A **$14,097 swing**. The shipped map permitted the two losing pairings and blocked
the two profitable ones.

Economic reading: selling premium *into* a trend leaves you short the tail that is
actively moving against you. Selling it *after* the move collects elevated IV on
mean reversion.

## 2. The decisive argument — an internal contradiction

Independent of the PF comparison, the shipped map made Execute **structurally
impossible**. Payoff calibration validates exactly one lane:

| base lane | status | n | 10%-stress PF | WF OOS | post-act OOS |
|---|---|---:|---:|---:|---:|
| **Credit \| Bear Call \| uptrend** | **PASS** | 33 | 2.220 | 19 | 4 |
| Credit \| Bull Put \| uptrend | INSUFFICIENT | 22 | 1.946 | 7 | 1 |
| Credit \| Bear Call \| range | VETO | 35 | 1.217 | 10 | 0 |
| Credit \| Bear Call \| downtrend | INSUFFICIENT | 32 | 1.148 | 0 | 0 |
| Credit \| Bull Put \| downtrend | VETO | 10 | 0.819 | 0 | 0 |
| Credit \| Bull Put \| range | VETO | 16 | 0.593 | 0 | 0 |
| Debit \| Bull Call \| uptrend | INSUFFICIENT | 26 | 1.647 | 9 | 1 |
| Debit \| Bear Put \| downtrend | INSUFFICIENT | 2 | — | 0 | 0 |

The one PASS lane is `Bear Call | uptrend`. The shipped map required Bear Call to be
in a **downtrend**. The pipeline's execution authority therefore rested on a lane its
own regime filter forbade — which is why every live run returned Execute 0.

## 3. Second instance of the same contradiction

`engine.py` re-derived regime alignment a second time in the confirmation layer:

```python
checks["price_action_trend"] = (
    trend == "range"
    or (trend == "uptrend" and direction in BULLISH_DIRECTIONS)
    or (trend == "downtrend" and direction in BEARISH_DIRECTIONS)
)
```

With the corrected credit map this made every credit candidate **jointly
unsatisfiable** — policy demanded one sign, confirmation the opposite. Now
strategy-aware: debit keeps trend-following (`Bull Call | uptrend` is the validated
debit lane), credit defers to `ALLOWED_REGIMES` as the single source of truth.

## 4. Macro conditioning

| variant (contrarian map) | n | PF | total $ | OOS |
|---|---:|---:|---:|---|
| contrarian | 600 | 1.196 | +7,110 | 3/4 |
| + low market IV | 288 | 1.399 | +5,701 | 3/4 |
| + risk_off market | 176 | 1.813 | +6,296 | 3/4 |
| + iv_rank ≥ 50 | 215 | 1.358 | +5,012 | 3/4 |
| + high market IV | 312 | 1.064 | +1,409 | 2/4 |

Strategy × market vol state — the strongest single discriminator found:

| strategy | high market IV | low market IV |
|---|---|---|
| Bear Call | 0.711 / −20,313 | **1.212 / +7,533** |
| Bull Call | **1.169 / +2,027** | 0.580 / −12,782 |
| Bull Put | 1.082 / +2,945 | 0.916 / −3,369 |
| Bear Put | 0.716 / −3,529 | 1.002 / +29 |

## 5. Bear Put family retired

| slice | n | PF | total $ | OOS |
|---|---:|---:|---:|---|
| **Bear Put, any regime** | 303 | **0.855** | **−3,500** | **0/4** |
| Bear Put + low market IV | 165 | 1.002 | +29 | 1/4 |

Zero out-of-sample folds clear the bar in six months. Removed from `DEBIT_POLICY`.

## 6. Second edge-model bug fixed

`edge_model.match_replay_edge` filtered credit history on `decision_pass` — the
per-session top-3 selection **capacity cap**, not a quality bar. It cut the learning
set from 164 guard-passing rows to 38 and produced `thin_replay_sample` on every
ticket. `confidence_calibration._eligible_history` had already fixed the identical
bug and documents why. A capacity cap must never filter an evidence set.

---

## 7. What this is worth — honest numbers

Validated tradeable lane, guard-passing, truncation-corrected:

| | |
|---|---|
| lane | `Credit \| Bear Call \| uptrend` |
| n | 28 over 16 sessions |
| profit factor | **2.160** |
| win rate | 89.3% |
| avg P/L | +$45.56 |
| total @1x | +$1,276 over 5.4 months |
| throughput | 5.2 trades/month |
| **per contract** | **$237/month** |
| monthly | 5/5 profitable (Feb +$106, Mar +$309, Apr +$191, May +$461, Jun +$209) |

**$10,000/month therefore needs roughly 42 contracts.** This remains a capital and
throughput problem, not an edge problem. Session mix across the window: 38 uptrend,
27 range, 20 downtrend — only uptrend sessions currently carry a validated lane, and
only 16 of those 38 produced a qualifying trade.

## 8. Caveats I am not hiding

1. **The two populations disagree.** The $14,097 swing is measured on the full
   evaluated universe. On the *guard-passing* subset the two maps are close —
   trend n=51 PF 1.605 +$1,648 (OOS 3/4) vs contrarian n=38 PF 1.699 +$1,277
   (OOS 2/4). The full-universe figure overstates realized improvement. The change
   is justified primarily by §2 (the validated lane was unreachable), with §1 as
   corroboration.
2. On guard-passing rows alone, `Bull Put | uptrend` scores PF 1.946 (n=22) while the
   full universe scores it 0.944 (n=462). The corrected map forbids it. Payoff
   calibration currently rates it INSUFFICIENT regardless, so nothing is lost today,
   but this disagreement should be re-examined as sample grows.
3. Per-lane samples are 10–35. These are small.
4. `Bull Call` debit improves from PF 0.977 (OOS 2/4) to 1.169 (OOS 3/4) when
   conditioned on high market IV, but that requires a session-level median IV rank
   computed at runtime — not implemented.
5. 2026-07-23 is a **downtrend** session, so Execute 0 is the correct output: no
   validated lane matches today's regime.

## 9. Changes shipped

- `credit_policy.py` → `credit-v3.0-dte28-ivrank-contrarian-regime`;
  `ALLOWED_REGIMES = {"Bull Put": {"downtrend"}, "Bear Call": {"uptrend"}}`
- `debit_policy.py` → `debit-v3.0-bull-call-only`; `Bear Put` entry deleted
- `engine.py` → `price_action_trend` is strategy-aware; credit defers to
  `ALLOWED_REGIMES`
- `edge_model.py` → `decision_pass` no longer filters the credit learning set;
  `EDGE_HISTORY_NAMESPACE` bumped to `..._v4_2026-07-26`

**This is a tightening, not a loosening.** Range sessions are now excluded in both
directions, the Bear Put family is eliminated, and every quality gate is unchanged:
credit band 25–30% of width, DTE ≥ 28, IV rank ≥ 30, flow alignment ≥ 0.10, quote
width ≤ 35%, max risk. Test suite: 232 passed.
