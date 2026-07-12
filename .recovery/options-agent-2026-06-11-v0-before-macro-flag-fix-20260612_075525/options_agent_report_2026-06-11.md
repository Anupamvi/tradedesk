# Options Agent Report - 2026-06-11

Mode: independent UW + Schwab live research options-agent-v0.

Target rows show desired credits/debits. Only rows in Send Now Orders with ready_to_enter=true are executable.

## Execution Snapshot

- Green send-now rows: 22
- Yellow target rows: 10
- Target refresh queue rows: 0
- Next action: review Send Now Orders; enter manually only after final quote check
- Live quote mode: live_schwab
- Portfolio context: ok
- Profitability confidence: 3.0/10
- Order-entry confidence: 5.0/10
- Profitability calibration: block (0 pass / 135 current rows)
- Calibration blockers: actual_support=BLOCK:102, WARN:33; replay_bucket=BLOCK:114, WARN:21; family_only_actual_rows=130; bucket_shortfall_rows=135 routes=bear_call_credit,bear_put_debit,bull_call_debit,bull_put_credit; missing_replay_bucket_rows=112 routes=bear_call_credit,bear_put_debit,bull_call_debit,bull_put_credit
- Bucket blocker examples: ABBV bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket,liquidity_bucket; SBUX bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket; XLV bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket
- Profitability gap plan: actual_closed_outcomes_negative_or_weak:21, actual_closed_outcomes_sample_gap:14; top=AMZN,AVGO,BAC,CRM,CVX,GLW,GOOG,GOOGL,HD,IWM,JPM,NFLX,NVDA,ORCL,PFE,PLTR,SNOW,TSLA,V,VRT,WMT bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=liquidity_bucket; AMD,AVGO,BKNG,CRM,CSCO,INTC,META,MO,MSFT,NOW,QCOM,SNOW bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=liquidity_bucket; AAPL,BMY,KO,NEM,T,UBER bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=dte_bucket,liquidity_bucket
- Calibrated order-entry blockers: no calibrated rows
- Execution fill quality: review (116 pass / 17 block)
- Route opportunity gaps: candidate_expansion=short_put; actual_weak=bull_call_debit
- Lesson pack: options-agent-v5 `sha256:ae2656b04bada574163db28ceeb7f7043415d041e46252d7d5c6b6f6e50c71cd`
- Report path: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/options_agent_report_2026-06-11.md`

## Output Files

- Report: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/options_agent_report_2026-06-11.md`
- All visible tickets: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/trade_tickets.csv`
- Green send-now tickets: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/green_trade_tickets.csv`
- Yellow target candidates: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/target_order_candidates.csv`
- Confidence audit: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/confidence_audit.csv`
- Strategy outcome atlas: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/strategy_outcome_atlas.csv`
- Profitability calibration: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/profitability_calibration.csv`
- Profitability gap plan: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/profitability_gap_plan.csv`
- Execution fill quality: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/execution_fill_quality.csv`
- Route opportunity gaps: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/route_opportunity_gap.csv`

## Send Now Orders

| Ticker | Signal | Structure | Exp | Sell Leg | Buy Leg | Qty | Target Limit | Target Exit | Max Profit | Max Loss | Confidence | Price / Risk |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| GOOG | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 GOOG 2026-06-18 372.5 Call | BUY 1 GOOG 2026-06-18 370 Call | 5 | 0.55 DEBIT | 0.99 | 975.0 | 275.0 | HIGH / 99 | green ready; verify live quote before manual send |
| BA | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 BA 2026-06-18 225 Call | BUY 1 BA 2026-06-18 222.5 Call | 5 | 0.73 DEBIT | 1.31 | 885.0 | 365.0 | HIGH / 95 | green ready; verify live quote before manual send |
| SHOP | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 SHOP 2026-07-17 120 Call | BUY 1 SHOP 2026-07-17 115 Call | 5 | 1.54 DEBIT | 2.77 | 1730.0 | 770.0 | HIGH / 94 | green ready; verify live quote before manual send |
| BX | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 BX 2026-06-18 131 Call | BUY 1 BX 2026-06-18 126 Call | 5 | 1.28 DEBIT | 2.3 | 1860.0 | 640.0 | HIGH / 94 | green ready; verify live quote before manual send |
| CVS | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 CVS 2026-07-17 115 Call | BUY 1 CVS 2026-07-17 105 Call | 5 | 2.2 DEBIT | 3.96 | 3900.0 | 1100.0 | HIGH / 94 | green ready; verify live quote before manual send |
| BMY | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 BMY 2026-07-17 62.5 Call | BUY 1 BMY 2026-07-17 60 Call | 5 | 0.47 DEBIT | 0.85 | 1015.0 | 235.0 | HIGH / 94 | green ready; verify live quote before manual send |
| ABBV | 🟢 GREEN ready | Call debit spread | 2026-06-26 | SELL 1 ABBV 2026-06-26 235 Call | BUY 1 ABBV 2026-06-26 230 Call | 5 | 1.33 DEBIT | 2.39 | 1835.0 | 665.0 | HIGH / 94 | green ready; verify live quote before manual send |
| AMAT | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 AMAT 2026-07-17 600 Call | BUY 1 AMAT 2026-07-17 590 Call | 3 | 3.58 DEBIT | 6.44 | 1926.0 | 1074.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| DIS | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 DIS 2026-07-17 105 Call | BUY 1 DIS 2026-07-17 100 Call | 5 | 1.82 DEBIT | 3.28 | 1590.0 | 910.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| VZ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 VZ 2026-07-17 43 Put | BUY 1 VZ 2026-07-17 46 Put | 5 | 0.63 DEBIT | 1.13 | 1185.0 | 315.0 | MEDIUM / 91 | green ready; verify live quote before manual send |
| VZ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 VZ 2026-07-17 45 Put | BUY 1 VZ 2026-07-17 46 Put | 5 | 0.32 DEBIT | 0.58 | 340.0 | 160.0 | MEDIUM / 91 | green ready; verify live quote before manual send |
| SLV | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 SLV 2026-07-17 55 Put | BUY 1 SLV 2026-07-17 55.5 Put | 5 | 0.14 DEBIT | 0.25 | 180.0 | 70.0 | MEDIUM / 90 | green ready; verify live quote before manual send |
| PG | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 PG 2026-07-17 140 Put | BUY 1 PG 2026-07-17 145 Put | 5 | 1.2 DEBIT | 2.16 | 1900.0 | 600.0 | MEDIUM / 89 | green ready; verify live quote before manual send |
| SLB | 🟢 GREEN ready | Put debit spread | 2026-06-18 | SELL 1 SLB 2026-06-18 54 Put | BUY 1 SLB 2026-06-18 57.5 Put | 5 | 1.4 DEBIT | 2.52 | 1050.0 | 700.0 | MEDIUM / 89 | green ready; verify live quote before manual send |
| SLB | 🟢 GREEN ready | Put debit spread | 2026-06-18 | SELL 1 SLB 2026-06-18 53 Put | BUY 1 SLB 2026-06-18 55 Put | 5 | 0.39 DEBIT | 0.7 | 805.0 | 195.0 | MEDIUM / 89 | green ready; verify live quote before manual send |
| T | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 T 2026-07-17 25 Call | BUY 1 T 2026-07-17 24 Call | 5 | 0.28 DEBIT | 0.5 | 360.0 | 140.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| C | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 C 2026-06-18 145 Call | BUY 1 C 2026-06-18 144 Call | 5 | 0.24 DEBIT | 0.43 | 380.0 | 120.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| COP | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 COP 2026-07-17 105 Put | BUY 1 COP 2026-07-17 115 Put | 4 | 2.65 DEBIT | 4.77 | 2940.0 | 1060.0 | MEDIUM / 86 | green ready; verify live quote before manual send |
| XLU | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 XLU 2026-07-17 42 Put | BUY 1 XLU 2026-07-17 43 Put | 5 | 0.2 DEBIT | 0.36 | 400.0 | 100.0 | MEDIUM / 86 | green ready; verify live quote before manual send |
| JNJ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 JNJ 2026-07-17 210 Put | BUY 1 JNJ 2026-07-17 230 Put | 3 | 3.04 DEBIT | 5.47 | 5088.0 | 912.0 | MEDIUM / 83 | green ready; verify live quote before manual send |
| JNJ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 JNJ 2026-07-17 220 Put | BUY 1 JNJ 2026-07-17 230 Put | 5 | 2.02 DEBIT | 3.64 | 3990.0 | 1010.0 | MEDIUM / 83 | green ready; verify live quote before manual send |
| UPS | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 UPS 2026-07-17 100 Put | BUY 1 UPS 2026-07-17 105 Put | 5 | 1.5 DEBIT | 2.7 | 1750.0 | 750.0 | MEDIUM / 82 | green ready; verify live quote before manual send |

## Target Orders - Target Credits/Debits

These are planning targets. Use the shown desired credit/debit as the starting limit, then refresh the Schwab quote before sending.

| Ticker | Signal | Structure | Exp | Sell Leg | Buy Leg | Qty | Target Limit | Target Exit | Max Profit | Max Loss | Confidence | Price / Risk |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| GOOG | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 GOOG 2026-06-18 352.5 Put | BUY 1 GOOG 2026-06-18 347.5 Put | 3 | 1.24 CREDIT | 0.43 | 372.0 | 1128.0 | HIGH / 91 | credit/width too weak for send-now |
| BA | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 BA 2026-06-18 212.5 Put | BUY 1 BA 2026-06-18 207.5 Put | 2 | 0.95 CREDIT | 0.33 | 190.0 | 810.0 | MEDIUM / 87 | credit/width too weak for send-now |
| SNOW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 SNOW 2026-06-18 255 Call | BUY 1 SNOW 2026-06-18 250 Call | 5 | 1.07 DEBIT | 1.93 | 1965.0 | 535.0 | HIGH / 86 | breakeven move too large for send-now |
| SNOW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 SNOW 2026-06-18 260 Call | BUY 1 SNOW 2026-06-18 250 Call | 5 | 1.77 DEBIT | 3.19 | 4115.0 | 885.0 | HIGH / 86 | breakeven move too large for send-now |
| GLW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 GLW 2026-06-18 195 Call | BUY 1 GLW 2026-06-18 190 Call | 5 | 1.08 DEBIT | 1.94 | 1960.0 | 540.0 | HIGH / 86 | breakeven move too large for send-now |
| XLI | 🟡 YELLOW target | Put credit spread | 2026-07-17 | SELL 1 XLI 2026-07-17 171 Put | BUY 1 XLI 2026-07-17 169 Put | 5 | 0.51 CREDIT | 0.18 | 255.0 | 745.0 | HIGH / 86 | credit/width too weak for send-now |
| XLY | 🟡 YELLOW target | Call debit spread | 2026-07-17 | SELL 1 XLY 2026-07-17 120 Call | BUY 1 XLY 2026-07-17 115 Call | 5 | 2.2 DEBIT | 3.96 | 1400.0 | 1100.0 | MEDIUM / 86 | reward/risk too weak for send-now |
| ANET | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 ANET 2026-06-18 157.5 Put | BUY 1 ANET 2026-06-18 152.5 Put | 3 | 1.17 CREDIT | 0.41 | 351.0 | 1149.0 | MEDIUM / 86 | credit/width too weak for send-now |
| PEP | 🟡 YELLOW target | Call credit spread | 2026-07-17 | SELL 1 PEP 2026-07-17 150 Call | BUY 1 PEP 2026-07-17 155 Call | 2 | 0.96 CREDIT | 0.34 | 192.0 | 808.0 | MEDIUM / 79 | credit/width too weak for send-now |
| WFC | 🟡 YELLOW target | Call credit spread | 2026-06-18 | SELL 1 WFC 2026-06-18 85 Call | BUY 1 WFC 2026-06-18 87 Call | 5 | 0.37 CREDIT | 0.13 | 185.0 | 815.0 | MEDIUM / 73 | credit too small for send-now; credit/width too weak for send-now |

## Run Diagnostics

Diagnostics explain confidence and coverage; the order-entry surface is Send Now Orders plus `trade_tickets.csv`.

- Trade rows: 22 green send-now, 10 target-order candidates
- Send-now readiness: execution_ready; non-green gates: []
- Live quote mode: live_schwab; live validation rows: 135
- Live spread quality audit: blocked_bad_live_markets; 15 blocked (8 quote-width, 5 liquidity)
- Agentic review coverage: lane 5/5 (1.0); broad rows 78/3916 (0.0199)
- Structure attempt rows: 270
- Final visible rows: 135
- Structural status counts, not order readiness: {'AVOID': 96, 'ENTER': 32, 'ENTER_WITH_PORTFOLIO_RISK': 4, 'REVIEW': 3}
- Portfolio context: ok
- Raw discovery: 6239 UW rows, 3916 generated candidates, 3918 catalyst rows, 8117 review rows
- Agentic dispatch tasks: 5; review status: reviews_ingested
- Agent review verdicts: {'avoid': 131, 'caution': 1861, 'supportive': 6125}; objective blockers: 130
- Strategy outcome atlas: positive families ['short_put']; negative current families ['vertical_spread']; blocking current ticker-strategy rows 26
- Route opportunity gaps: candidate_expansion=short_put; actual_weak=bull_call_debit

## Execution Quality Gates

- Execution confidence ratings: {'NOT_EXECUTION_READY': 109, 'HIGH': 22, 'MEDIUM': 4}
- Trade-quality confidence ratings: {'LOW': 100, 'MEDIUM': 23, 'HIGH': 12}
- Top non-green send-now gates: {'positive_contract_size_required': 98, 'objective_blocker': 96, 'credit/width too weak for send-now': 6, 'fresh Schwab chain': 5, 'breakeven move too large for send-now': 3, 'manual_review_required': 2, 'positive_entry_limit_required': 2, 'trade_plan_required': 2}

## Focus Review Queue - Not Trades

These are not orders. This section is limited to validated rows and focus tickers; tail unvalidated rows stay in CSV artifacts.

| Ticker | Signal | Reason | Qty | Target Limit | Max Loss | Trade Plan |
|---|---|---|---:|---:|---:|---|
| FCX | 🟡 YELLOW review | live Schwab chain Bear Put validated at 1.91 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 1.91 DEBIT | 955.0 | BUY 1 FCX 2026-07-17 65 Put / SELL 1 FCX 2026-07-17 60 Put @ 1.91 DEBIT |

## Coverage Audit

Coverage rows explain inclusion/exclusion only. They are not orders; use Send Now Orders, Target Orders, and `trade_tickets.csv` for the action surface.

| Ticker | Signal | Bias | Score | State | Why | Next Step |
|---|---|---|---:|---|---|---|
| AAPL | 🔴 RED blocked | bullish | 72.73 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| MSFT | 🔴 RED blocked | bullish | 74.73 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| NVDA | 🔴 RED blocked | bullish | 68.44 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| AMZN | 🔴 RED blocked | bullish | 68.55 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| META | 🔴 RED blocked | bullish | 68.74 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| GOOG | 🟢 GREEN ready | bullish | 77.68 | GREEN ready | live Schwab chain Bull Call validated at 0.55 debit | verify live quote and place manually if thesis still holds |
| GOOGL | 🔴 RED blocked | bullish | 67 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| TSLA | 🔴 RED blocked | bullish | 68.58 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| AMD | 🔴 RED blocked | bullish | 73.71 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| AVGO | 🔴 RED blocked | bullish | 70.81 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| SPY | 🔴 RED blocked | bullish | 70.35 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| QQQ | 🔴 RED blocked | bullish | 75.06 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| IWM | 🔴 RED blocked | bullish | 74.05 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| DIA | 🔴 RED blocked | bullish | 63.46 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| PLTR | 🔴 RED blocked | bullish | 68.02 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| HOOD | 🔴 RED no-action | bullish | 69.62 | RED no-action | not actionable: liquid underlying; liquid common stock with sufficient market cap, stock volume, and option open interest | do not trade from the action list; require explicit override and fresh validation |
| WMT | 🔴 RED blocked | bullish | 63.8 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| URA | 🔴 RED no-action | bullish | 30.86 | RED no-action | not actionable: excluded underlying; non-core ETF; not in actionable ETF allowlist (URA) | do not trade from the action list; require explicit override and fresh validation |
| DVN | 🔴 RED no-action | neutral | 59.96 | RED no-action | not actionable: liquid underlying; liquid common stock with sufficient market cap, stock volume, and option open interest | do not trade from the action list; require explicit override and fresh validation |
| OKLO | 🔴 RED no-action | bearish | 31.67 | RED no-action | not actionable: speculative underlying; marketcap_below_20000000000 | do not trade from the action list; require explicit override and fresh validation |
| BA | 🟢 GREEN ready | bullish | 76.93 | GREEN ready | live Schwab chain Bull Call validated at 0.73 debit | verify live quote and place manually if thesis still holds |
| VZ | 🟢 GREEN ready | bearish | 80.32 | GREEN ready | live Schwab chain Bear Put validated at 0.63 debit; built-in agent caution: risk_on: index price tape leans bullish | verify live quote and place manually if thesis still holds |
| UPS | 🟢 GREEN ready | bearish | 82.6 | GREEN ready | live Schwab chain Bear Put validated at 1.50 debit; built-in agent caution: risk_on: index price tape leans bullish | verify live quote and place manually if thesis still holds |
| AMAT | 🟢 GREEN ready | bullish | 76.17 | GREEN ready | live Schwab chain Bull Call validated at 3.58 debit | verify live quote and place manually if thesis still holds |
| SHOP | 🟢 GREEN ready | bullish | 75.46 | GREEN ready | live Schwab chain Bull Call validated at 1.54 debit | verify live quote and place manually if thesis still holds |
| SNOW | 🟡 YELLOW coverage | bullish | 75.43 | YELLOW coverage | live Schwab chain Bull Call validated at 1.07 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| BX | 🟢 GREEN ready | bullish | 74.95 | GREEN ready | live Schwab chain Bull Call validated at 1.28 debit | verify live quote and place manually if thesis still holds |
| CVS | 🟢 GREEN ready | bullish | 74.32 | GREEN ready | live Schwab chain Bull Call validated at 2.20 debit | verify live quote and place manually if thesis still holds |
| GLW | 🟡 YELLOW coverage | bullish | 71.21 | YELLOW coverage | live Schwab chain Bull Call validated at 1.08 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| COP | 🟢 GREEN ready | bearish | 75.85 | GREEN ready | live Schwab chain Bear Put validated at 2.65 debit; built-in agent caution: risk_on: index price tape leans bullish | verify live quote and place manually if thesis still holds |

## Decision Board Summary

Full ranked rows are in `decision_board.csv`; rejected setup-quality rows stay audit-visible in `final_recommendations.csv`.

| Status | Count |
|---|---:|
| blocked | 96 |
| needs_review | 4 |
| ready | 25 |
| waiting_for_price | 10 |

## Near Miss / No Trade Audit

Showing first 20 of 3828 rows; full audit is in `no_trade_audit.csv`.

| Ticker | Bias | Score | Reason |
|---|---|---:|---|
| PYPL | bullish | 82.78 | bullish flow bias 0.46; signal premium $52,664,423; liquid underlying; score 82.8 |
| KHC | bearish | 78.43 | bearish flow bias -0.46; signal premium $1,276,730; liquid underlying; score 78.4 |
| FISV | bearish | 78.04 | bearish flow bias -0.68; signal premium $4,080,840; liquid underlying; score 78.0 |
| GM | bearish | 75.13 | bearish flow bias -0.35; signal premium $4,818,500; liquid underlying; score 75.1 |
| F | bullish | 74.87 | bullish flow bias 0.27; signal premium $8,876,868; liquid underlying; score 74.9 |
| VST | bullish | 71.07 | bullish flow bias 0.29; signal premium $18,219,267; liquid underlying; score 71.1 |
| B | bullish | 70.63 | bullish flow bias 0.27; signal premium $4,519,447; liquid underlying; score 70.6 |
| HPQ | bearish | 70.13 | bearish flow bias -0.29; signal premium $1,586,397; liquid underlying; score 70.1 |
| RKLB | bearish | 69.64 | bearish flow bias -0.11; signal premium $95,917,650; liquid underlying; score 69.6 |
| HOOD | bullish | 69.62 | bullish flow bias 0.08; signal premium $158,062,153; liquid underlying; score 69.6 |
| CPNG | bullish | 68.74 | bullish flow bias 0.13; signal premium $43,999,419; liquid underlying; score 68.7 |
| PCG | bullish | 68.74 | bullish flow bias 0.39; signal premium $1,410,359; liquid underlying; score 68.7 |
| CNC | bearish | 67.14 | bearish flow bias -0.35; signal premium $3,395,401; liquid underlying; score 67.1 |
| BSX | bullish | 66.25 | bullish flow bias 0.11; signal premium $41,683,660; liquid underlying; score 66.2 |
| ZS | bullish | 65.86 | bullish flow bias 0.10; signal premium $99,959,353; liquid underlying; score 65.9 |
| RBLX | bullish | 65.66 | bullish flow bias 0.15; signal premium $12,484,161; liquid underlying; score 65.7 |
| LUV | bullish | 65.64 | bullish flow bias 0.24; signal premium $1,571,211; liquid underlying; score 65.6 |
| OXY | bearish | 65.4 | bearish flow bias -0.13; signal premium $5,054,845; liquid underlying; score 65.4 |
| DAL | bullish | 64.35 | bullish flow bias 0.11; signal premium $8,802,844; liquid underlying; score 64.3 |
| ON | bearish | 64.29 | bearish flow bias -0.20; signal premium $4,252,266; liquid underlying; score 64.3 |
| ... |  |  | 3808 additional no-trade rows in no_trade_audit.csv |

## Warnings

- broad research-task coverage is low; execution readiness is based on subagent lane coverage plus per-ticket lane coverage
- fresh quote validation is required before manual order entry
