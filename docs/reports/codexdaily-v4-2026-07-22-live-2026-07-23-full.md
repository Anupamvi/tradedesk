# Codex Daily V4.12 — Full Live-Planning Report

**UW discovery date:** 2026-07-22  
**Live validation date:** 2026-07-23  
**Pipeline:** `v4.12-goal-shadow-prospective-20260723`  
**Mode:** July 22 EOD discovery + current Schwab chain and portfolio validation  
**Order placement:** None; this report does not place orders.

## Executive Decision

The run produced **2 model-level Execute rows, 15 Scout rows, 17 Work Limit rows, 342 Research rows, and 1,311 Avoid rows**.

After applying the stricter OCO entry instructions and current portfolio context:

1. **QQQ bear-call spread is the only clean new-entry candidate.** It is still conditional on a same-session quote refresh and a limit credit of at least **$2.50**, not the looser $1.82 headline floor.
2. **PLTR is not a clean add today.** Although the candidate engine labels it Execute, the portfolio engine reports existing PLTR option exposure, portfolio fit only 5/10, and a current PLTR short option requiring `ROLL`. Treat it as **REVIEW / DO NOT ADD** until the existing position is handled and the pipeline is rerun.
3. All Scout and Work Limit rows remain **non-orders**.
4. The TWLO goal-policy row remains **shadow-only** and must never be entered from this report.

## Run Integrity

| Check | Result |
|:--|:--|
| All five UW exports | PASS: stock screener, hot chains, chain-OI changes, bot EOD, DP EOD |
| Schwab option validation | PASS: 1,674 candidate rows with live `PASS`; 4 with no realistic spread |
| Schwab portfolio | PASS: 63 positions loaded |
| Dark-pool corroboration | PASS: 43 accumulation and 7 distribution tickers; soft confirmation only |
| Market regime | Range; medium volatility; weak flow; VIX proxy 19.49 |
| Data quality | WARNING, not critical |
| Critical blockers | None at run level |
| Aggregate risk budget | $15,000 applied |

### Run-level warnings

- No free-form local browser/news captures were present; two structured support files were available, including the earnings calendar.
- The recent-performance replay was **42 days old**, above the 30-day freshness threshold.
- No closed, filled live V4 outcomes exist in the live-outcome ledger yet.

## Confidence — Honest Interpretation

**Overall pipeline confidence: MODERATE, not “88% certain.”**

The displayed 88% is a conditional estimate for one narrow validated route. It is not an exact-ticket win probability and is not proof that the entire pipeline has an 88% success rate.

| Confidence layer | Evidence | Verdict |
|:--|:--|:--|
| Overall walk-forward calibration | 118 predictions; predicted 61.23%; actual 61.86%; Brier 0.2418 vs no-skill 0.2500; calibration gap 0.64 percentage points | PASS, but only a modest improvement over no-skill |
| Credit-family prior | 185 rows; 121 wins; estimate 65.24%; 90% lower bound 60.81% | PASS |
| Debit-family prior | 49 rows; estimate 49.02%; 90% lower bound 40.01%; Brier worse than 0.25 | FAIL; no debit trade should be called high confidence |
| Execute payoff route | `flow_cost::Credit|Bear Call|range|flow=directional|cost=18to30`; n=17; stressed win rate 88.24%; 90% lower bound 74.73%; stressed PF 2.37; expanding OOS n=8; post-activation n=3 | PASS, but sample remains small |
| QQQ exact-ticket replay | n=0 | UNAVAILABLE |
| PLTR exact-ticket replay | n=0 | UNAVAILABLE |
| Goal-shadow policy | 1 pending, 0 resolved prospective rows | UNPROVEN |

### Historical profitability context

- Corrected broad replay of current selected trades: PF **1.157**, below the 1.2 objective.
- Strict all-five-source selected cohort: PF **1.239** on only 8 trades; too small for a broad claim.
- Untouched nested June fold: PF **0.818** and negative P/L.
- The current Execute route is stronger at stressed PF **2.37**, but only n=17 and does not supply exact QQQ/PLTR replay history.

Therefore, confidence is strongest at the **route level**, weaker at the **family level**, and absent at the **exact-ticket and prospective shadow levels**.

## Trade 1 — Conditional New Entry

### QQQ Bear Call Credit Spread — CONDITIONAL ENTER

| Field | Instruction |
|:--|:--|
| Structure | Sell QQQ 2026-07-31 711C; buy QQQ 2026-07-31 721C |
| Direction | Bearish / range-income credit spread |
| Size | **1 spread only** |
| Quote observed by run | Mid $2.72; natural $2.67 |
| Operative entry limit | **Net credit ≥ $2.50** |
| Do not use | Do not rely on the looser $1.82 headline by itself |
| Take profit | OCO buy-to-close near **$1.00 debit** |
| Stop | OCO buy-to-close near **$5.00 debit**, or close/review if the 711 short strike is threatened |
| Hold | 1–5 trading days; do not carry unmanaged expiration-week gamma |
| Gap -1% | Bearish thesis improves, but do not chase; still require ≥ $2.50 with fresh OI/news confirmation |
| Gap +1% | Thesis weakens; downgrade to Scout/Research unless resistance, flow, and quote width reconfirm |
| Contractual max risk | About $750 at a $2.50 fill before fees; report-level conservative cap is $818 |
| Modeled route evidence | n=17; 88.24% stressed win rate; 74.73% 90% lower bound; PF 2.37 |
| Family evidence | 65.24% estimate; 60.81% 90% lower bound; n=185 |
| Exact-ticket evidence | None, n=0 |
| Portfolio fit | 8/10 strong; no concentration flag |
| Confidence | **Moderate-high route confidence; low exact-ticket confidence** |

### Entry checklist

Enter only if every item is true at order time:

- QQQ 711/721 spread still yields at least $2.50 credit.
- Quote width remains reasonable; the run observed about 1.8%.
- Bearish directional flow remains intact.
- Exact-leg OI remains supportive.
- No new macro/news event invalidates the thesis.
- One-lot risk remains inside the portfolio budget.
- The take-profit and stop OCO are attached before submission.

## Trade 2 — Portfolio-Conflicted Model Execute

### PLTR Bear Call Credit Spread — REVIEW / DO NOT ADD NOW

| Field | Instruction |
|:--|:--|
| Structure | Sell PLTR 2026-07-31 128C; buy PLTR 2026-07-31 130C |
| Model disposition | Execute |
| Practical disposition | **REVIEW / DO NOT ADD until existing PLTR exposure is handled** |
| Quote observed by run | Mid $0.55; natural $0.42 |
| Operative entry if later cleared | Net credit ≥ **$0.50** |
| Take profit if later cleared | OCO buy-to-close near $0.20 debit |
| Stop if later cleared | OCO buy-to-close near $1.00 debit, or if the 128 short strike is threatened |
| Earnings | 2026-08-03 after hours; spread expires 2026-07-31 |
| Portfolio conflict | Existing PLTR option exposure; portfolio fit 5/10 weak |
| Existing-position instruction | Current PLTR short option is marked `ROLL`; manage it before adding unrelated PLTR risk |
| Candidate score | 5.39, Medium |
| Penalties | Low-credit/medium-score guard, replay-band issue, reduced size pending live proof |
| Exact-ticket evidence | None, n=0 |
| Confidence | **Moderate route evidence, low ticket confidence, unacceptable portfolio ambiguity** |

The candidate engine and portfolio-repair engine disagree here. Portfolio risk takes priority. This should not be treated as a clean second order.

## Non-Orders

### Highest Work Limit rows

| Ticker | Structure | Status | Requirement |
|:--|:--|:--|:--|
| AMD | Sell 585C / buy 595C, 2026-07-31 | Work Limit only | Require ≥ $2.50 credit; natural was $1.35; mixed OI and earnings 2026-08-04 |
| QQQ | Multiple July/August bear-call alternatives | Work Limit only | Do not stack with the selected QQQ spread; non-price confirmations must improve |
| SPY | Sell 753C / buy 755C, 2026-07-31 | Work Limit only | Non-price confirmations must improve |
| HIMS | Sell 34.5C / buy 37C, 2026-07-31 | Work Limit only | Natural $0.35 was below required $0.46 |
| NFLX | Sell 72C / buy 74C, 2026-08-21 | Work Limit only | Credit must improve to $0.50 or better |

### Scout rows

There were 15 Scout rows. They are **one-lot manual reviews, not recommendations to enter**. Most are correlated SPY/QQQ bear-call variants. If the QQQ Execute is taken, do not stack additional index-bearish variants without a separate correlated-risk review.

## Goal Shadow — Research Only

| Field | Value |
|:--|:--|
| Policy | `n1_fillable_dp0.20_oi-none_hist0.0_prior8_model0.0` |
| Ticker | TWLO |
| Structure | Bear put debit: long 182.5P / short 177.5P, expiry 2026-07-31 |
| Prospectively locked entry | $1.80 debit |
| Target | $2.88 |
| Stop | $0.90 |
| Flow/OI at first observation | Hedge flow; contrary OI |
| Outcome | PENDING |
| Resolved prospective sample | 0 |
| Execution eligible | False |
| Order placement | Prohibited |

A ledger defect discovered during this run allowed reruns of the same historical date to replace the original entry. It has been fixed so the first observation is immutable, and the central ledger was restored to the original $1.80 entry.

## Portfolio Actions Before New Risk

Priority reviews from the current Schwab portfolio:

- **ROLL / REDUCE review:** PLTR, GOOG, SLV, NVDA, INTC, SOFI, AAPL, ASTS, SPCX, META, AMZN, NEM, ORCL.
- **TAKE-PROFIT / lifecycle review:** GRAB, GOOG, SLV, HOOD, UNH, META, SPCX, NFLX, OPEN, PYPL, NEM.
- **Concentration:** GOOG about 19.7%; AMZN, NVDA, and AMD about 9.4–9.5% each.
- Core equity holdings remain protected from covered-call assignment; income trades must use the separate trading sleeve.

## Risk and $10,000/Month Feasibility

| Metric | Result |
|:--|--:|
| Immediate Execute modeled profit potential | $131.40 |
| Realistic fill-adjusted current potential | $131.40 |
| Risk-bounded visible potential including non-orders | $850.68 |
| Expected monthly run rate | $1,051.20 |
| Required daily P/L | $1,250.00 |
| Required weekly P/L | $6,250.00 |
| Visible ticket max risk | $16,503.00 |
| Configured aggregate risk budget | $15,000.00 |
| In-budget visible max risk | $14,926.00 |
| Can sizing close the target gap? | No |

The pipeline does **not** demonstrate a consistent $10,000/month path. Scaling the two Execute rows cannot honestly close that gap under the current evidence and risk budget.

## What Remains Pending

1. **Prospective shadow evidence:** TWLO is pending; zero resolved rows exist.
2. **Live execution evidence:** no closed, filled V4 live outcomes exist yet.
3. **Replay freshness:** the recent-performance report is 42 days old and must be refreshed.
4. **Exact-ticket evidence:** both current Execute tickets have exact replay n=0.
5. **PLTR portfolio conflict:** candidate disposition should downgrade to Review when portfolio repair requires a roll.
6. **Entry-display consistency:** headline target and OCO entry differ ($1.82 vs $2.50 for QQQ; $0.37 vs $0.50 for PLTR). The stricter OCO limit is used in this report; the pipeline display should be unified.
7. **Historical source depth:** only a limited set of historical sessions has genuine all-five-source data; confidence needs more accumulated full bot/DP history.
8. **Browser/news capture:** structured earnings data exists, but free-form local browser/news capture remains missing.
9. **Operational integration:** source-integrity/ETF/data-quality work remains uncommitted in the working trees even though the current CodexUW suite passes.
10. **Profit target:** PF ≥ 1.2 is not robust across all untouched folds, and the $10,000/month objective remains unproven.

## Validation Status

- Full current CodexUW suite: **229 passed**.
- Clean staged shadow-ledger suite: **224 passed**.
- Operational V4 plus version tests: **25 passed**.
- V4.13 probationary-pilot experiment: rejected; its only evaluated pilot lost $92.50, PF 0.0. It is not deployed.
- Shadow first-observation integrity fix: commit `db9f365`; V4.12 tag moved to this revision.

## Final Action Summary

- **Conditional order candidate:** QQQ 711/721 bear-call spread, one lot, credit ≥ $2.50, OCO $1.00 / $5.00, only after same-session refresh.
- **Do not add now:** PLTR 128/130 bear-call spread until the existing PLTR position is rolled/reduced and a fresh rerun approves it.
- **Do not enter:** TWLO shadow, Scouts, Work Limits, Research, or Avoid rows.
