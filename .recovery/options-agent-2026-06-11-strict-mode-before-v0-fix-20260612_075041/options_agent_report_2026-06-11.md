# Options Agent Report - 2026-06-11

Mode: independent UW + Schwab live research options-agent-v0.

Target rows show desired credits/debits. Only rows in Send Now Orders with ready_to_enter=true are executable.

## Execution Snapshot

- Green send-now rows: 0
- Yellow target rows: 0
- Target refresh queue rows: 0
- Next action: resolve blocking gates before treating any row as an order
- Live quote mode: live_schwab
- Portfolio context: ok
- Profitability confidence: 2.5/10
- Order-entry confidence: 0.0/10
- Profitability calibration: block (0 pass / 135 current rows)
- Calibration blockers: actual_support=BLOCK:102, WARN:33; replay_bucket=BLOCK:115, WARN:20; family_only_actual_rows=130; bucket_shortfall_rows=135 routes=bear_call_credit,bear_put_debit,bull_call_debit,bull_put_credit; missing_replay_bucket_rows=113 routes=bear_call_credit,bear_put_debit,bull_call_debit,bull_put_credit
- Bucket blocker examples: ABBV bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket,liquidity_bucket; SBUX bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket; XLV bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket
- Profitability gap plan: actual_closed_outcomes_negative_or_weak:17, actual_closed_outcomes_sample_gap:13; top=AMZN,BA,BAC,CRM,CVX,GLW,GOOG,GOOGL,HD,IWM,JPM,NFLX,NVDA,ORCL,PFE,PLTR,QCOM,SNOW,TSLA,V,VRT,WMT bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=liquidity_bucket; AMD,AVGO,BKNG,CRM,CSCO,INTC,META,MO,MRVL,MSFT,NOW,QCOM,SNOW bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=liquidity_bucket; AAPL,BMY,DIS,NEM,UBER,XOM bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=dte_bucket,liquidity_bucket
- Calibrated order-entry blockers: no calibrated rows
- Execution fill quality: review (115 pass / 18 block)
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

No green send-now orders. Do not send an order unless a row appears here.

## Target Orders - Target Credits/Debits

No target-order candidates were produced.

## Run Diagnostics

Diagnostics explain confidence and coverage; the order-entry surface is Send Now Orders plus `trade_tickets.csv`.

- Trade rows: 0 green send-now, 0 target-order candidates
- Send-now readiness: gates_pass_no_send_now_orders; non-green gates: ['ready_trade_tickets']
- Live quote mode: live_schwab; live validation rows: 135
- Live spread quality audit: blocked_bad_live_markets; 13 blocked (8 quote-width, 3 liquidity)
- Agentic review coverage: lane 5/5 (1.0); broad rows 78/3916 (0.0199)
- Structure attempt rows: 270
- Final visible rows: 135
- Structural status counts, not order readiness: {'AVOID': 94, 'ENTER': 33, 'ENTER_WITH_PORTFOLIO_RISK': 4, 'REVIEW': 3, 'WAIT_FOR_PRICE': 1}
- Portfolio context: ok
- Raw discovery: 6239 UW rows, 3916 generated candidates, 3918 catalyst rows, 8117 review rows
- Agentic dispatch tasks: 5; review status: reviews_ingested
- Agent review verdicts: {'avoid': 129, 'caution': 1863, 'supportive': 6125}; objective blockers: 128
- Strategy outcome atlas: positive families ['short_put']; negative current families ['vertical_spread']; blocking current ticker-strategy rows 87
- Route opportunity gaps: candidate_expansion=short_put; actual_weak=bull_call_debit

## Execution Quality Gates

- Execution confidence ratings: {'NOT_EXECUTION_READY': 135}
- Trade-quality confidence ratings: {'LOW': 135}
- Top non-green send-now gates: {'profitability_calibration_actual_bucket_negative': 135, 'route/economics profitability calibration required': 135, 'agent review coverage': 127, 'positive_contract_size_required': 96, 'objective_blocker': 94, 'negative realized strategy history': 39, 'positive strategy expectancy required': 37, 'position max profit below materiality floor': 14}

## Focus Review Queue - Not Trades

These are not orders. This section is limited to validated rows and focus tickers; tail unvalidated rows stay in CSV artifacts.

| Ticker | Signal | Reason | Qty | Target Limit | Max Loss | Trade Plan |
|---|---|---|---:|---:|---:|---|
| BA | 🟡 YELLOW review | live Schwab chain Bull Call validated at 0.54 debit | 5 | 0.54 DEBIT | 270.0 | BUY 1 BA 2026-06-18 225 Call / SELL 1 BA 2026-06-18 227.5 Call @ 0.54 DEBIT |
| BA | 🟡 YELLOW review | live Schwab chain Bull Call validated at 0.90 debit | 5 | 0.9 DEBIT | 450.0 | BUY 1 BA 2026-06-18 225 Call / SELL 1 BA 2026-06-18 230 Call @ 0.90 DEBIT |
| VZ | 🟡 YELLOW review | live Schwab chain Bear Put validated at 0.66 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 0.66 DEBIT | 330.0 | BUY 1 VZ 2026-07-17 46 Put / SELL 1 VZ 2026-07-17 43 Put @ 0.66 DEBIT |
| UPS | 🟡 YELLOW review | live Schwab chain Bear Put validated at 1.46 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 1.46 DEBIT | 730.0 | BUY 1 UPS 2026-07-17 105 Put / SELL 1 UPS 2026-07-17 100 Put @ 1.46 DEBIT |
| AMAT | 🟡 YELLOW review | live Schwab chain Bull Call validated at 3.27 debit | 3 | 3.27 DEBIT | 981.0 | BUY 1 AMAT 2026-07-17 580 Call / SELL 1 AMAT 2026-07-17 590 Call @ 3.27 DEBIT |
| GOOG | 🟡 YELLOW review | live Schwab chain Bull Call validated at 0.59 debit | 5 | 0.59 DEBIT | 295.0 | BUY 1 GOOG 2026-06-18 370 Call / SELL 1 GOOG 2026-06-18 372.5 Call @ 0.59 DEBIT |
| SHOP | 🟡 YELLOW review | live Schwab chain Bull Call validated at 1.79 debit | 5 | 1.79 DEBIT | 895.0 | BUY 1 SHOP 2026-07-17 115 Call / SELL 1 SHOP 2026-07-17 120 Call @ 1.79 DEBIT |
| SNOW | 🟡 YELLOW review | live Schwab chain Bull Call validated at 1.23 debit | 5 | 1.23 DEBIT | 615.0 | BUY 1 SNOW 2026-06-18 250 Call / SELL 1 SNOW 2026-06-18 255 Call @ 1.23 DEBIT |
| SNOW | 🟡 YELLOW review | live Schwab chain Bull Call validated at 2.04 debit | 5 | 2.04 DEBIT | 1020.0 | BUY 1 SNOW 2026-06-18 250 Call / SELL 1 SNOW 2026-06-18 260 Call @ 2.04 DEBIT |
| BX | 🟡 YELLOW review | live Schwab chain Bull Call validated at 1.94 debit | 5 | 1.94 DEBIT | 970.0 | BUY 1 BX 2026-06-18 125 Call / SELL 1 BX 2026-06-18 130 Call @ 1.94 DEBIT |
| CVS | 🟡 YELLOW review | live Schwab chain Bull Call validated at 2.20 debit | 5 | 2.2 DEBIT | 1100.0 | BUY 1 CVS 2026-07-17 105 Call / SELL 1 CVS 2026-07-17 115 Call @ 2.20 DEBIT |
| GLW | 🟡 YELLOW review | live Schwab chain Bull Call validated at 1.28 debit | 5 | 1.28 DEBIT | 640.0 | BUY 1 GLW 2026-06-18 190 Call / SELL 1 GLW 2026-06-18 195 Call @ 1.28 DEBIT |
| COP | 🟡 YELLOW review | live Schwab chain Bear Put validated at 2.70 debit; built-in agent caution: risk_on: index price tape leans bullish | 4 | 2.7 DEBIT | 1080.0 | BUY 1 COP 2026-07-17 115 Put / SELL 1 COP 2026-07-17 105 Put @ 2.70 DEBIT |
| BMY | 🟡 YELLOW review | live Schwab chain Bull Call validated at 0.46 debit | 5 | 0.46 DEBIT | 230.0 | BUY 1 BMY 2026-07-17 60 Call / SELL 1 BMY 2026-07-17 62.5 Call @ 0.46 DEBIT |
| PG | 🟡 YELLOW review | live Schwab chain Bear Put validated at 1.27 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 1.27 DEBIT | 635.0 | BUY 1 PG 2026-07-17 145 Put / SELL 1 PG 2026-07-17 140 Put @ 1.27 DEBIT |
| DIS | 🟡 YELLOW review | live Schwab chain Bull Call validated at 0.89 debit | 5 | 0.89 DEBIT | 445.0 | BUY 1 DIS 2026-07-17 105 Call / SELL 1 DIS 2026-07-17 110 Call @ 0.89 DEBIT |
| DIS | 🟡 YELLOW review | live Schwab chain Bull Call validated at 0.89 debit | 5 | 0.89 DEBIT | 445.0 | BUY 1 DIS 2026-07-17 105 Call / SELL 1 DIS 2026-07-17 110 Call @ 0.89 DEBIT |
| SLB | 🟡 YELLOW review | live Schwab chain Bear Put validated at 1.38 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 1.38 DEBIT | 690.0 | BUY 1 SLB 2026-06-18 57.5 Put / SELL 1 SLB 2026-06-18 54 Put @ 1.38 DEBIT |
| SLB | 🟡 YELLOW review | live Schwab chain Bear Put validated at 0.39 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 0.39 DEBIT | 195.0 | BUY 1 SLB 2026-06-18 55 Put / SELL 1 SLB 2026-06-18 53 Put @ 0.39 DEBIT |
| ABBV | 🟡 YELLOW review | live Schwab chain Bull Call validated at 1.35 debit | 5 | 1.35 DEBIT | 675.0 | BUY 1 ABBV 2026-06-26 230 Call / SELL 1 ABBV 2026-06-26 235 Call @ 1.35 DEBIT |
| JNJ | 🟡 YELLOW review | live Schwab chain Bear Put validated at 3.36 debit; built-in agent caution: risk_on: index price tape leans bullish | 3 | 3.36 DEBIT | 1008.0 | BUY 1 JNJ 2026-07-17 230 Put / SELL 1 JNJ 2026-07-17 210 Put @ 3.36 DEBIT |
| JNJ | 🟡 YELLOW review | live Schwab chain Bear Put validated at 2.35 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 2.35 DEBIT | 1175.0 | BUY 1 JNJ 2026-07-17 230 Put / SELL 1 JNJ 2026-07-17 220 Put @ 2.35 DEBIT |
| XLY | 🟡 YELLOW review | live Schwab chain found 2.34 debit above target 2.25; built-in agent caution: live Schwab chain found 2.34 debit above target 2.25 | 5 | 2.34 DEBIT | 1170.0 | BUY 1 XLY 2026-07-17 115 Call / SELL 1 XLY 2026-07-17 120 Call @ 2.34 DEBIT |
| PEP | 🟡 YELLOW review | live Schwab chain Bear Call validated at 0.97 credit; built-in agent caution: risk_on: index price tape leans bullish | 2 | 0.97 CREDIT | 806.0 | SELL 1 PEP 2026-07-17 150 Call / BUY 1 PEP 2026-07-17 155 Call @ 0.97 CREDIT |
| VZ | 🟡 YELLOW review | live Schwab chain Bear Put validated at 0.32 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 0.32 DEBIT | 160.0 | BUY 1 VZ 2026-07-17 46 Put / SELL 1 VZ 2026-07-17 45 Put @ 0.32 DEBIT |
| ... |  | 13 additional review rows in decision_board.csv |  |  |  |  |

## Coverage Audit

Coverage rows explain inclusion/exclusion only. They are not orders; use Send Now Orders, Target Orders, and `trade_tickets.csv` for the action surface.

| Ticker | Signal | Bias | Score | State | Why | Next Step |
|---|---|---|---:|---|---|---|
| AAPL | 🔴 RED blocked | bullish | 72.73 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| MSFT | 🔴 RED blocked | bullish | 74.73 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| NVDA | 🔴 RED blocked | bullish | 68.44 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| AMZN | 🔴 RED blocked | bullish | 68.55 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| META | 🔴 RED blocked | bullish | 68.74 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| GOOG | 🟡 YELLOW review | bullish | 77.68 | YELLOW review | live Schwab chain Bull Call validated at 0.59 debit | reprice in Schwab and resolve catalyst/quality review |
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
| BA | 🟡 YELLOW review | bullish | 76.93 | YELLOW review | live Schwab chain Bull Call validated at 0.54 debit | reprice in Schwab and resolve catalyst/quality review |
| VZ | 🟡 YELLOW review | bearish | 80.32 | YELLOW review | live Schwab chain Bear Put validated at 0.66 debit; built-in agent caution: risk_on: index price tape leans bullish | reprice in Schwab and resolve catalyst/quality review |
| UPS | 🟡 YELLOW review | bearish | 82.6 | YELLOW review | live Schwab chain Bear Put validated at 1.46 debit; built-in agent caution: risk_on: index price tape leans bullish | reprice in Schwab and resolve catalyst/quality review |
| AMAT | 🟡 YELLOW review | bullish | 76.17 | YELLOW review | live Schwab chain Bull Call validated at 3.27 debit | reprice in Schwab and resolve catalyst/quality review |
| SHOP | 🟡 YELLOW review | bullish | 75.46 | YELLOW review | live Schwab chain Bull Call validated at 1.79 debit | reprice in Schwab and resolve catalyst/quality review |
| SNOW | 🟡 YELLOW review | bullish | 75.43 | YELLOW review | live Schwab chain Bull Call validated at 1.23 debit | reprice in Schwab and resolve catalyst/quality review |
| BX | 🟡 YELLOW review | bullish | 74.95 | YELLOW review | live Schwab chain Bull Call validated at 1.94 debit | reprice in Schwab and resolve catalyst/quality review |
| CVS | 🟡 YELLOW review | bullish | 74.32 | YELLOW review | live Schwab chain Bull Call validated at 2.20 debit | reprice in Schwab and resolve catalyst/quality review |
| GLW | 🟡 YELLOW review | bullish | 71.21 | YELLOW review | live Schwab chain Bull Call validated at 1.28 debit | reprice in Schwab and resolve catalyst/quality review |
| COP | 🟡 YELLOW review | bearish | 75.85 | YELLOW review | live Schwab chain Bear Put validated at 2.70 debit; built-in agent caution: risk_on: index price tape leans bullish | reprice in Schwab and resolve catalyst/quality review |

## Decision Board Summary

Full ranked rows are in `decision_board.csv`; rejected setup-quality rows stay audit-visible in `final_recommendations.csv`.

| Status | Count |
|---|---:|
| blocked | 94 |
| needs_confidence | 20 |
| needs_review | 3 |
| waiting_for_price | 18 |

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
