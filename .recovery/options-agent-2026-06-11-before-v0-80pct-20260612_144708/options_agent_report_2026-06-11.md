# Options Agent Report - 2026-06-11

Mode: independent UW + Schwab live research options-agent-v0.

Target rows show desired credits/debits. Only rows in Send Now Orders with ready_to_enter=true are executable.

## Execution Snapshot

- Green send-now rows: 47
- Yellow target rows: 34
- Target refresh queue rows: 0
- Next action: review Send Now Orders; enter manually only after final quote check
- Live quote mode: live_schwab
- Portfolio context: ok
- Profitability confidence: 3.0/10
- Order-entry confidence: 5.0/10
- Outcome evidence audit: block (162 realized P/L rows; forward gaps ['codexuw_execute_outcome_ledger', 'codexuw_recommendation_outcome_ledger'])
- Broker outcome match audit: exact_matches_available (7 exact backfill rows; 0 ambiguous)
- Broker matched outcomes: not_positive (sample 6, avg P/L -83.17, profit factor 0.26)
- Profitability calibration: block (0 pass / 135 current rows)
- Calibration blockers: actual_support=BLOCK:63, WARN:72; replay_bucket=BLOCK:93, WARN:42; bucket_shortfall_rows=135 routes=bear_call_credit,bear_put_debit,bull_call_debit,bull_put_credit; missing_replay_bucket_rows=91 routes=bear_call_credit,bear_put_debit,bull_call_debit,bull_put_credit
- Bucket blocker examples: AMD bull_call_debit/bullish/dte_0_14/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=liquidity_bucket; AVGO bull_call_debit/bullish/dte_0_14/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=liquidity_bucket; AVGO bull_call_debit/bullish/dte_0_14/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=liquidity_bucket
- Profitability gap plan: actual_closed_outcomes_negative_or_weak:14, actual_closed_outcomes_sample_gap:24; top=AVGO,C,CRM,CSCO,INTC,MU,NOW,SNOW,V bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=liquidity_bucket; CSCO,CVX,SPY bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=28; CMCSA,DIS,NEM,SHOP,T bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=dte_bucket
- Calibrated order-entry blockers: no calibrated rows
- Execution fill quality: review (110 pass / 20 block)
- Route opportunity gaps: candidate_expansion=short_put; actual_weak=bull_call_debit
- Lesson pack: options-agent-v5 `sha256:ae2656b04bada574163db28ceeb7f7043415d041e46252d7d5c6b6f6e50c71cd`
- Report path: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/options_agent_report_2026-06-11.md`

## Output Files

- Report: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/options_agent_report_2026-06-11.md`
- All visible tickets: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/trade_tickets.csv`
- Green send-now tickets: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/green_trade_tickets.csv`
- Yellow target candidates: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/target_order_candidates.csv`
- Confidence audit: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/confidence_audit.csv`
- Outcome evidence audit: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/outcome_evidence_audit.csv`
- Broker outcome match audit: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/broker_outcome_match_audit.csv`
- Broker matched outcomes: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/broker_matched_outcomes.csv`
- Strategy outcome atlas: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/strategy_outcome_atlas.csv`
- Profitability calibration: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/profitability_calibration.csv`
- Profitability gap plan: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/profitability_gap_plan.csv`
- Execution fill quality: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/execution_fill_quality.csv`
- Route opportunity gaps: `/Users/anuppamvi/uw_root/tradedesk/out/options_agent/2026-06-11/route_opportunity_gap.csv`

## Send Now Orders

| Ticker | Signal | Structure | Exp | Sell Leg | Buy Leg | Qty | Target Limit | Target Exit | Max Profit | Max Loss | Confidence | Price / Risk |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| AMD | 🟢 GREEN ready | Put credit spread | 2026-06-18 | SELL 1 AMD 2026-06-18 495 Put | BUY 1 AMD 2026-06-18 490 Put | 3 | 1.55 CREDIT | 0.54 | 465.0 | 1035.0 | HIGH / 100 | green ready; verify live quote before manual send |
| AMZN | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 AMZN 2026-06-18 247.5 Call | BUY 1 AMZN 2026-06-18 245 Call | 5 | 0.62 DEBIT | 1.12 | 940.0 | 310.0 | HIGH / 100 | green ready; verify live quote before manual send |
| AMZN | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 AMZN 2026-06-18 250 Call | BUY 1 AMZN 2026-06-18 245 Call | 5 | 1 DEBIT | 1.8 | 2000.0 | 500.0 | HIGH / 100 | green ready; verify live quote before manual send |
| GOOG | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 GOOG 2026-06-18 370 Call | BUY 1 GOOG 2026-06-18 367.5 Call | 5 | 0.63 DEBIT | 1.13 | 935.0 | 315.0 | HIGH / 99 | green ready; verify live quote before manual send |
| GOOGL | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 GOOGL 2026-06-18 372.5 Call | BUY 1 GOOGL 2026-06-18 370 Call | 5 | 0.54 DEBIT | 0.97 | 980.0 | 270.0 | HIGH / 99 | green ready; verify live quote before manual send |
| NFLX | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 NFLX 2026-06-18 87 Call | BUY 1 NFLX 2026-06-18 82 Call | 5 | 0.55 DEBIT | 0.99 | 2225.0 | 275.0 | HIGH / 97 | green ready; verify live quote before manual send |
| NFLX | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 NFLX 2026-06-18 83 Call | BUY 1 NFLX 2026-06-18 82 Call | 5 | 0.25 DEBIT | 0.45 | 375.0 | 125.0 | HIGH / 97 | green ready; verify live quote before manual send |
| IWM | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 IWM 2026-06-18 300 Call | BUY 1 IWM 2026-06-18 299 Call | 5 | 0.26 DEBIT | 0.47 | 370.0 | 130.0 | HIGH / 96 | green ready; verify live quote before manual send |
| AVGO | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 AVGO 2026-06-18 397.5 Call | BUY 1 AVGO 2026-06-18 395 Call | 5 | 0.66 DEBIT | 1.19 | 920.0 | 330.0 | HIGH / 95 | green ready; verify live quote before manual send |
| NVDA | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 NVDA 2026-06-18 212.5 Call | BUY 1 NVDA 2026-06-18 210 Call | 5 | 0.67 DEBIT | 1.21 | 915.0 | 335.0 | HIGH / 95 | green ready; verify live quote before manual send |
| MRVL | 🟢 GREEN ready | Put credit spread | 2026-06-18 | SELL 1 MRVL 2026-06-18 265 Put | BUY 1 MRVL 2026-06-18 262.5 Put | 5 | 0.92 CREDIT | 0.32 | 460.0 | 790.0 | HIGH / 95 | green ready; verify live quote before manual send |
| BA | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 BA 2026-06-18 227.5 Call | BUY 1 BA 2026-06-18 225 Call | 5 | 0.58 DEBIT | 1.04 | 960.0 | 290.0 | HIGH / 95 | green ready; verify live quote before manual send |
| BAC | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 BAC 2026-06-18 59 Call | BUY 1 BAC 2026-06-18 57 Call | 5 | 0.28 DEBIT | 0.5 | 860.0 | 140.0 | HIGH / 95 | green ready; verify live quote before manual send |
| V | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 V 2026-06-18 332.5 Call | BUY 1 V 2026-06-18 327.5 Call | 5 | 1.07 DEBIT | 1.93 | 1965.0 | 535.0 | HIGH / 95 | green ready; verify live quote before manual send |
| BAC | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 BAC 2026-06-18 58 Call | BUY 1 BAC 2026-06-18 57 Call | 5 | 0.2 DEBIT | 0.36 | 400.0 | 100.0 | HIGH / 95 | green ready; verify live quote before manual send |
| SHOP | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 SHOP 2026-07-17 120 Call | BUY 1 SHOP 2026-07-17 115 Call | 5 | 1.62 DEBIT | 2.92 | 1690.0 | 810.0 | HIGH / 94 | green ready; verify live quote before manual send |
| CVS | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 CVS 2026-07-17 115 Call | BUY 1 CVS 2026-07-17 105 Call | 5 | 2.04 DEBIT | 3.67 | 3980.0 | 1020.0 | HIGH / 94 | green ready; verify live quote before manual send |
| IBM | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 IBM 2026-07-17 310 Call | BUY 1 IBM 2026-07-17 290 Call | 2 | 4.54 DEBIT | 8.17 | 3092.0 | 908.0 | HIGH / 94 | green ready; verify live quote before manual send |
| JPM | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 JPM 2026-06-18 330 Call | BUY 1 JPM 2026-06-18 327.5 Call | 5 | 0.6 DEBIT | 1.08 | 950.0 | 300.0 | HIGH / 94 | green ready; verify live quote before manual send |
| JPM | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 JPM 2026-06-18 332.5 Call | BUY 1 JPM 2026-06-18 327.5 Call | 5 | 1.03 DEBIT | 1.85 | 1985.0 | 515.0 | HIGH / 94 | green ready; verify live quote before manual send |
| BMY | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 BMY 2026-07-17 62.5 Call | BUY 1 BMY 2026-07-17 60 Call | 5 | 0.46 DEBIT | 0.83 | 1020.0 | 230.0 | HIGH / 94 | green ready; verify live quote before manual send |
| CSCO | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 CSCO 2026-06-18 130 Call | BUY 1 CSCO 2026-06-18 120 Call | 4 | 2.86 DEBIT | 5.15 | 2856.0 | 1144.0 | HIGH / 94 | green ready; verify live quote before manual send |
| CVX | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 CVX 2026-06-18 192.5 Call | BUY 1 CVX 2026-06-18 190 Call | 5 | 0.73 DEBIT | 1.31 | 885.0 | 365.0 | HIGH / 94 | green ready; verify live quote before manual send |
| NEM | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 NEM 2026-07-17 110 Call | BUY 1 NEM 2026-07-17 105 Call | 5 | 1.62 DEBIT | 2.92 | 1690.0 | 810.0 | HIGH / 94 | green ready; verify live quote before manual send |
| UBER | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 UBER 2026-07-17 75 Call | BUY 1 UBER 2026-07-17 72.5 Call | 5 | 0.64 DEBIT | 1.15 | 930.0 | 320.0 | HIGH / 94 | green ready; verify live quote before manual send |
| DIA | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 DIA 2026-07-17 535 Call | BUY 1 DIA 2026-07-17 525 Call | 4 | 2.66 DEBIT | 4.79 | 2936.0 | 1064.0 | HIGH / 94 | green ready; verify live quote before manual send |
| META | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 META 2026-06-18 590 Call | BUY 1 META 2026-06-18 585 Call | 5 | 1.08 DEBIT | 1.94 | 1960.0 | 540.0 | HIGH / 94 | green ready; verify live quote before manual send |
| CSCO | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 CSCO 2026-06-18 126 Call | BUY 1 CSCO 2026-06-18 125 Call | 5 | 0.22 DEBIT | 0.4 | 390.0 | 110.0 | HIGH / 94 | green ready; verify live quote before manual send |
| AMAT | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 AMAT 2026-07-17 590 Call | BUY 1 AMAT 2026-07-17 580 Call | 3 | 3.9 DEBIT | 7.02 | 1830.0 | 1170.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| DIS | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 DIS 2026-07-17 105 Call | BUY 1 DIS 2026-07-17 100 Call | 5 | 1.91 DEBIT | 3.44 | 1545.0 | 955.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| ABBV | 🟢 GREEN ready | Call debit spread | 2026-06-26 | SELL 1 ABBV 2026-06-26 235 Call | BUY 1 ABBV 2026-06-26 230 Call | 5 | 1.76 DEBIT | 3.17 | 1620.0 | 880.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| CMCSA | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 CMCSA 2026-07-17 26 Call | BUY 1 CMCSA 2026-07-17 25 Call | 5 | 0.35 DEBIT | 0.63 | 325.0 | 175.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| VZ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 VZ 2026-07-17 44 Put | BUY 1 VZ 2026-07-17 47 Put | 5 | 0.85 DEBIT | 1.53 | 1075.0 | 425.0 | MEDIUM / 91 | green ready; verify live quote before manual send |
| VZ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 VZ 2026-07-17 45 Put | BUY 1 VZ 2026-07-17 46 Put | 5 | 0.3 DEBIT | 0.54 | 350.0 | 150.0 | MEDIUM / 91 | green ready; verify live quote before manual send |
| SLV | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 SLV 2026-07-17 56 Put | BUY 1 SLV 2026-07-17 56.5 Put | 5 | 0.13 DEBIT | 0.23 | 185.0 | 65.0 | MEDIUM / 90 | green ready; verify live quote before manual send |
| SPY | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 SPY 2026-06-18 751 Call | BUY 1 SPY 2026-06-18 750 Call | 5 | 0.35 DEBIT | 0.63 | 325.0 | 175.0 | MEDIUM / 90 | green ready; verify live quote before manual send |
| WMT | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 WMT 2026-06-18 124 Call | BUY 1 WMT 2026-06-18 123 Call | 5 | 0.28 DEBIT | 0.5 | 360.0 | 140.0 | MEDIUM / 90 | green ready; verify live quote before manual send |
| PG | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 PG 2026-07-17 140 Put | BUY 1 PG 2026-07-17 145 Put | 5 | 1.08 DEBIT | 1.94 | 1960.0 | 540.0 | MEDIUM / 89 | green ready; verify live quote before manual send |
| QQQ | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 QQQ 2026-06-18 735 Call | BUY 1 QQQ 2026-06-18 734 Call | 5 | 0.25 DEBIT | 0.45 | 375.0 | 125.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| T | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 T 2026-07-17 25 Call | BUY 1 T 2026-07-17 24 Call | 5 | 0.3 DEBIT | 0.54 | 350.0 | 150.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| C | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 C 2026-06-18 144 Call | BUY 1 C 2026-06-18 143 Call | 5 | 0.28 DEBIT | 0.5 | 360.0 | 140.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| PFE | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 PFE 2026-06-18 27 Call | BUY 1 PFE 2026-06-18 26.5 Call | 5 | 0.1 DEBIT | 0.18 | 200.0 | 50.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| MU | 🟢 GREEN ready | Put credit spread | 2026-06-18 | SELL 1 MU 2026-06-18 940 Put | BUY 1 MU 2026-06-18 935 Put | 3 | 1.73 CREDIT | 0.61 | 519.0 | 981.0 | MEDIUM / 86 | green ready; verify live quote before manual send |
| COP | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 COP 2026-07-17 100 Put | BUY 1 COP 2026-07-17 110 Put | 5 | 1.58 DEBIT | 2.84 | 4210.0 | 790.0 | MEDIUM / 86 | green ready; verify live quote before manual send |
| XOM | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 XOM 2026-07-17 160 Call | BUY 1 XOM 2026-07-17 155 Call | 5 | 1.17 DEBIT | 2.11 | 1915.0 | 585.0 | MEDIUM / 84 | green ready; verify live quote before manual send |
| MSFT | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 MSFT 2026-07-17 365 Put | BUY 1 MSFT 2026-07-17 370 Put | 5 | 1.38 DEBIT | 2.48 | 1810.0 | 690.0 | MEDIUM / 83 | green ready; verify live quote before manual send |
| UPS | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 UPS 2026-07-17 100 Put | BUY 1 UPS 2026-07-17 105 Put | 5 | 1.45 DEBIT | 2.61 | 1775.0 | 725.0 | MEDIUM / 82 | green ready; verify live quote before manual send |

## Target Orders - Target Credits/Debits

These are planning targets. Use the shown desired credit/debit as the starting limit, then refresh the Schwab quote before sending.

| Ticker | Signal | Structure | Exp | Sell Leg | Buy Leg | Qty | Target Limit | Target Exit | Max Profit | Max Loss | Confidence | Price / Risk |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| AMD | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 AMD 2026-06-18 550 Call | BUY 1 AMD 2026-06-18 547.5 Call | 5 | 0.55 DEBIT | 0.99 | 975.0 | 275.0 | HIGH / 93 | breakeven move too large for send-now |
| GOOG | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 GOOG 2026-06-18 352.5 Put | BUY 1 GOOG 2026-06-18 347.5 Put | 3 | 1.25 CREDIT | 0.44 | 375.0 | 1125.0 | HIGH / 91 | credit/width too weak for send-now |
| GOOGL | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 GOOGL 2026-06-18 352.5 Put | BUY 1 GOOGL 2026-06-18 350 Put | 5 | 0.6 CREDIT | 0.21 | 300.0 | 950.0 | HIGH / 91 | credit/width too weak for send-now |
| INTC | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 INTC 2026-06-18 135 Call | BUY 1 INTC 2026-06-18 134 Call | 5 | 0.24 DEBIT | 0.43 | 380.0 | 120.0 | HIGH / 90 | breakeven move too large for send-now |
| INTC | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 INTC 2026-06-18 119 Put | BUY 1 INTC 2026-06-18 118 Put | 5 | 0.34 CREDIT | 0.12 | 170.0 | 330.0 | HIGH / 90 | credit too small for send-now |
| MRVL | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 MRVL 2026-06-18 302.5 Call | BUY 1 MRVL 2026-06-18 300 Call | 5 | 0.74 DEBIT | 1.33 | 880.0 | 370.0 | HIGH / 87 | breakeven move too large for send-now |
| AVGO | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 AVGO 2026-06-18 407.5 Call | BUY 1 AVGO 2026-06-18 397.5 Call | 5 | 2.17 DEBIT | 3.91 | 3915.0 | 1085.0 | HIGH / 87 | breakeven move too large for send-now |
| NVDA | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 NVDA 2026-06-18 200 Put | BUY 1 NVDA 2026-06-18 197.5 Put | 5 | 0.52 CREDIT | 0.18 | 260.0 | 990.0 | MEDIUM / 87 | credit/width too weak for send-now |
| BA | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 BA 2026-06-18 215 Put | BUY 1 BA 2026-06-18 210 Put | 3 | 1.02 CREDIT | 0.36 | 306.0 | 1194.0 | MEDIUM / 87 | credit/width too weak for send-now |
| SNOW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 SNOW 2026-06-18 250 Call | BUY 1 SNOW 2026-06-18 245 Call | 5 | 1.07 DEBIT | 1.93 | 1965.0 | 535.0 | HIGH / 86 | breakeven move too large for send-now |
| SNOW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 SNOW 2026-06-18 255 Call | BUY 1 SNOW 2026-06-18 245 Call | 5 | 1.79 DEBIT | 3.22 | 4105.0 | 895.0 | HIGH / 86 | breakeven move too large for send-now |
| VRT | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 VRT 2026-06-18 335 Call | BUY 1 VRT 2026-06-18 312.5 Call | 2 | 5 DEBIT | 9 | 3500.0 | 1000.0 | HIGH / 86 | breakeven move too large for send-now |
| GLW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 GLW 2026-06-18 197.5 Call | BUY 1 GLW 2026-06-18 192.5 Call | 5 | 1.03 DEBIT | 1.85 | 1985.0 | 515.0 | HIGH / 86 | breakeven move too large for send-now |
| QCOM | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 QCOM 2026-06-18 227.5 Call | BUY 1 QCOM 2026-06-18 225 Call | 5 | 0.59 DEBIT | 1.06 | 955.0 | 295.0 | HIGH / 86 | breakeven move too large for send-now |
| QCOM | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 QCOM 2026-06-18 205 Put | BUY 1 QCOM 2026-06-18 195 Put | 1 | 2.63 CREDIT | 0.92 | 263.0 | 737.0 | HIGH / 86 | credit/width too weak for send-now |
| META | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 META 2026-06-18 555 Put | BUY 1 META 2026-06-18 552.5 Put | 5 | 0.7 CREDIT | 0.24 | 350.0 | 900.0 | HIGH / 86 | credit/width too weak for send-now |
| NEE | 🟡 YELLOW target | Call debit spread | 2026-07-17 | SELL 1 NEE 2026-07-17 90 Call | BUY 1 NEE 2026-07-17 85 Call | 5 | 2.25 DEBIT | 4 | 1375.0 | 1125.0 | MEDIUM / 86 | reward/risk too weak for send-now |
| ANET | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 ANET 2026-06-18 157.5 Put | BUY 1 ANET 2026-06-18 152.5 Put | 3 | 1 CREDIT | 0.35 | 300.0 | 1200.0 | MEDIUM / 86 | credit/width too weak for send-now |
| UBER | 🟡 YELLOW target | Put credit spread | 2026-07-17 | SELL 1 UBER 2026-07-17 65 Put | BUY 1 UBER 2026-07-17 62.5 Put | 5 | 0.53 CREDIT | 0.19 | 265.0 | 985.0 | MEDIUM / 86 | credit/width too weak for send-now |
| BRKB | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 BRKB 2026-06-18 495 Call | BUY 1 BRKB 2026-06-18 490 Call | 5 | 1.62 DEBIT | 2.92 | 1690.0 | 810.0 | MEDIUM / 83 | use the shown target limit as the starting point; adjust if the live quote moves |
| PLTR | 🟡 YELLOW target | Put debit spread | 2026-06-18 | SELL 1 PLTR 2026-06-18 122 Put | BUY 1 PLTR 2026-06-18 123 Put | 5 | 0.25 DEBIT | 0.45 | 375.0 | 125.0 | MEDIUM / 81 | breakeven move too large for send-now |
| CRM | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 CRM 2026-06-18 175 Call | BUY 1 CRM 2026-06-18 172.5 Call | 5 | 0.52 DEBIT | 0.94 | 990.0 | 260.0 | MEDIUM / 80 | breakeven move too large for send-now |
| NOW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 NOW 2026-06-18 109 Call | BUY 1 NOW 2026-06-18 108 Call | 5 | 0.22 DEBIT | 0.4 | 390.0 | 110.0 | MEDIUM / 80 | breakeven move too large for send-now |
| NOW | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 NOW 2026-06-18 99 Put | BUY 1 NOW 2026-06-18 94 Put | 3 | 1 CREDIT | 0.35 | 300.0 | 1200.0 | MEDIUM / 80 | credit/width too weak for send-now |
| CRM | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 CRM 2026-06-18 162.5 Put | BUY 1 CRM 2026-06-18 157.5 Put | 3 | 1.14 CREDIT | 0.4 | 342.0 | 1158.0 | MEDIUM / 80 | credit/width too weak for send-now |
| PEP | 🟡 YELLOW target | Call credit spread | 2026-07-17 | SELL 1 PEP 2026-07-17 150 Call | BUY 1 PEP 2026-07-17 155 Call | 3 | 1.08 CREDIT | 0.38 | 324.0 | 1176.0 | MEDIUM / 79 | credit/width too weak for send-now |
| MU | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 MU 2026-06-18 1050 Call | BUY 1 MU 2026-06-18 1045 Call | 5 | 0.91 DEBIT | 1.64 | 2045.0 | 455.0 | MEDIUM / 78 | breakeven move too large for send-now |
| TSLA | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 TSLA 2026-06-18 430 Call | BUY 1 TSLA 2026-06-18 427.5 Call | 5 | 0.52 DEBIT | 0.94 | 990.0 | 260.0 | MEDIUM / 76 | breakeven move too large for send-now |
| TSLA | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 TSLA 2026-06-18 395 Put | BUY 1 TSLA 2026-06-18 392.5 Put | 5 | 0.7 CREDIT | 0.24 | 350.0 | 900.0 | MEDIUM / 76 | credit/width too weak for send-now |
| MSFT | 🟡 YELLOW target | Call credit spread | 2026-07-17 | SELL 1 MSFT 2026-07-17 410 Call | BUY 1 MSFT 2026-07-17 415 Call | 3 | 1.28 CREDIT | 0.45 | 384.0 | 1116.0 | MEDIUM / 75 | credit/width too weak for send-now |
| ORCL | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 ORCL 2026-06-18 195 Call | BUY 1 ORCL 2026-06-18 192.5 Call | 5 | 0.61 DEBIT | 1.1 | 945.0 | 305.0 | MEDIUM / 73 | breakeven move too large for send-now |
| WFC | 🟡 YELLOW target | Call credit spread | 2026-06-18 | SELL 1 WFC 2026-06-18 85 Call | BUY 1 WFC 2026-06-18 87 Call | 5 | 0.41 CREDIT | 0.14 | 205.0 | 795.0 | MEDIUM / 73 | credit too small for send-now; credit/width too weak for send-now |
| PLTR | 🟡 YELLOW target | Call credit spread | 2026-06-18 | SELL 1 PLTR 2026-06-18 132 Call | BUY 1 PLTR 2026-06-18 133 Call | 5 | 0.25 CREDIT | 0.09 | 125.0 | 375.0 | MEDIUM / 73 | credit too small for send-now; credit/width too weak for send-now |
| QQQ | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 QQQ 2026-06-18 702 Put | BUY 1 QQQ 2026-06-18 701 Put | 5 | 0.26 CREDIT | 0.09 | 130.0 | 370.0 | MEDIUM / 72 | credit too small for send-now; credit/width too weak for send-now |

## Run Diagnostics

Diagnostics explain confidence and coverage; the order-entry surface is Send Now Orders plus `trade_tickets.csv`.

- Trade rows: 47 green send-now, 34 target-order candidates
- Send-now readiness: execution_ready; non-green gates: []
- Live quote mode: live_schwab; live validation rows: 135
- Live spread quality audit: blocked_bad_live_markets; 20 blocked (15 quote-width, 6 liquidity)
- Agentic review coverage: lane 5/5 (1.0); broad rows 78/3916 (0.0199)
- Structure attempt rows: 270
- Final visible rows: 135
- Structural status counts, not order readiness: {'AVOID': 39, 'ENTER': 64, 'ENTER_WITH_PORTFOLIO_RISK': 28, 'REVIEW': 3, 'WAIT_FOR_PRICE': 1}
- Portfolio context: ok
- Outcome evidence audit: block; contributing sources ['schwab_closed_trades']; blocking sources ['codexuw_execute_outcome_ledger', 'codexuw_recommendation_outcome_ledger', 'broker_matched_outcomes']
- Broker outcome match audit: exact_matches_available; exact backfill rows {'codexuw_execute_outcome_ledger': 6, 'codexuw_recommendation_outcome_ledger': 1}; ambiguous rows 0
- Broker matched outcomes: BLOCK; tickers ['CRM', 'HOOD', 'META', 'RKLB', 'WMT']; total P/L -499.0
- Raw discovery: 6239 UW rows, 3916 generated candidates, 3918 catalyst rows, 8117 review rows
- Agentic dispatch tasks: 5; review status: reviews_ingested
- Agent review verdicts: {'avoid': 69, 'caution': 1865, 'supportive': 6183}; objective blockers: 68
- Strategy outcome atlas: positive families ['short_put']; negative current families ['vertical_spread']; blocking current ticker-strategy rows 56
- Route opportunity gaps: candidate_expansion=short_put; actual_weak=bull_call_debit

## Execution Quality Gates

- Execution confidence ratings: {'NOT_EXECUTION_READY': 78, 'HIGH': 51, 'MEDIUM': 6}
- Trade-quality confidence ratings: {'LOW': 47, 'HIGH': 44, 'MEDIUM': 44}
- Top non-green send-now gates: {'positive_contract_size_required': 41, 'objective_blocker': 39, 'credit/width too weak for send-now': 19, 'breakeven move too large for send-now': 15, 'credit too small for send-now': 4, 'fresh Schwab chain': 2, 'manual_review_required': 2, 'positive_entry_limit_required': 2}

## Focus Review Queue - Not Trades

These are not orders. This section is limited to validated rows and focus tickers; tail unvalidated rows stay in CSV artifacts.

| Ticker | Signal | Reason | Qty | Target Limit | Max Loss | Trade Plan |
|---|---|---|---:|---:|---:|---|
| FCX | 🟡 YELLOW review | live Schwab chain Bear Call validated at 0.91 credit; built-in agent caution: risk_on: index price tape leans bullish | 2 | 0.91 CREDIT | 818.0 | SELL 1 FCX 2026-07-17 75 Call / BUY 1 FCX 2026-07-17 80 Call @ 0.91 CREDIT |
| ORCL | 🟡 YELLOW review | live Schwab chain Bull Put validated at 0.56 credit; built-in agent caution: earnings in 89 days; headline: As of 2026-06-11 post-close, broad market tone improved with tech/AI leadership rebounding and geopolitical risk easing. Investopedia reported U.S. stocks jumped as technology s... | 5 | 0.56 CREDIT | 970.0 | SELL 1 ORCL 2026-06-18 177.5 Put / BUY 1 ORCL 2026-06-18 175 Put @ 0.56 CREDIT |
| XOM | 🟡 YELLOW review | live Schwab chain Bull Put validated at 0.95 credit; Bullish XOM setup is cautioned by the same oil de-escalation catalyst: crude fell and energy stocks did not participate in the rally. This does not objectively block entry, but it weakens catalyst alignment. | 2 | 0.95 CREDIT | 810.0 | SELL 1 XOM 2026-07-17 140 Put / BUY 1 XOM 2026-07-17 135 Put @ 0.95 CREDIT |
| SCHW | 🟡 YELLOW review | live Schwab chain Bear Put validated at 0.51 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 0.51 DEBIT | 255.0 | BUY 1 SCHW 2026-06-18 90 Put / SELL 1 SCHW 2026-06-18 88 Put @ 0.51 DEBIT |
| SCHW | 🟡 YELLOW review | live Schwab chain Bear Put validated at 0.31 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 0.31 DEBIT | 155.0 | BUY 1 SCHW 2026-06-18 90 Put / SELL 1 SCHW 2026-06-18 89 Put @ 0.31 DEBIT |
| MRK | 🟡 YELLOW review | live Schwab chain found 2.32 debit above target 2.25; built-in agent caution: live Schwab chain found 2.32 debit above target 2.25 | 5 | 2.32 DEBIT | 1160.0 | BUY 1 MRK 2026-06-18 117 Call / SELL 1 MRK 2026-06-18 122 Call @ 2.32 DEBIT |

## Coverage Audit

Coverage rows explain inclusion/exclusion only. They are not orders; use Send Now Orders, Target Orders, and `trade_tickets.csv` for the action surface.

| Ticker | Signal | Bias | Score | State | Why | Next Step |
|---|---|---|---:|---|---|---|
| AAPL | 🔴 RED blocked | bullish | 72.73 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| MSFT | 🟢 GREEN ready | bearish | 74.73 | GREEN ready | live Schwab chain Bear Put validated at 1.38 debit; built-in agent caution: risk_on: index price tape leans bullish | verify live quote and place manually if thesis still holds |
| NVDA | 🟢 GREEN ready | bullish | 68.44 | GREEN ready | live Schwab chain Bull Call validated at 0.67 debit | verify live quote and place manually if thesis still holds |
| AMZN | 🟢 GREEN ready | bullish | 68.55 | GREEN ready | live Schwab chain Bull Call validated at 0.62 debit | verify live quote and place manually if thesis still holds |
| META | 🟢 GREEN ready | bullish | 68.74 | GREEN ready | live Schwab chain Bull Call validated at 1.08 debit | verify live quote and place manually if thesis still holds |
| GOOG | 🟢 GREEN ready | bullish | 77.68 | GREEN ready | live Schwab chain Bull Call validated at 0.63 debit | verify live quote and place manually if thesis still holds |
| GOOGL | 🟢 GREEN ready | bullish | 67 | GREEN ready | live Schwab chain Bull Call validated at 0.54 debit | verify live quote and place manually if thesis still holds |
| TSLA | 🟡 YELLOW coverage | bullish | 68.58 | YELLOW coverage | live Schwab chain Bull Call validated at 0.52 debit; TSLA adds high-beta consumer/mega-cap risk on top of QQQ/SPY/AMZN-style exposure; keep it small or choose it instead of broad beta, not in addition to every beta leg. | use the shown target limit as the starting point; adjust if the live quote moves |
| AMD | 🟡 YELLOW coverage | bullish | 73.71 | YELLOW coverage | live Schwab chain Bull Call validated at 0.55 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| AVGO | 🟢 GREEN ready | bullish | 70.81 | GREEN ready | live Schwab chain Bull Call validated at 0.66 debit | verify live quote and place manually if thesis still holds |
| SPY | 🟢 GREEN ready | bullish | 70.35 | GREEN ready | live Schwab chain Bull Call validated at 0.35 debit; Broad-market bullish exposure overlaps with QQQ, IWM, XLK, TSLA, AMZN, META, MSFT, NVDA and other beta-heavy candidates; treat as aggregate portfolio beta, not an independent setup. | verify live quote and place manually if thesis still holds |
| QQQ | 🟢 GREEN ready | bullish | 75.06 | GREEN ready | live Schwab chain Bull Call validated at 0.25 debit; Crowded beta expression: do not stack QQQ with SPY, XLK, and multiple bullish mega-cap tech or semiconductor tickets unless total index/tech risk is capped and the intended exposure is explicit.; The displayed top source leg is same-day and effectively unusable for next-day planning. Rebuild from fresh Schwab chain only | verify live quote and place manually if thesis still holds |
| IWM | 🟢 GREEN ready | bullish | 74.05 | GREEN ready | live Schwab chain Bull Call validated at 0.26 debit | verify live quote and place manually if thesis still holds |
| DIA | 🟢 GREEN ready | bullish | 63.46 | GREEN ready | live Schwab chain Bull Call validated at 2.66 debit | verify live quote and place manually if thesis still holds |
| PLTR | 🟡 YELLOW coverage | bearish | 68.02 | YELLOW coverage | live Schwab chain Bear Put validated at 0.25 debit; built-in agent caution: risk_on: index price tape leans bullish | use the shown target limit as the starting point; adjust if the live quote moves |
| HOOD | 🔴 RED no-action | bullish | 69.62 | RED no-action | not actionable: liquid underlying; liquid common stock with sufficient market cap, stock volume, and option open interest | do not trade from the action list; require explicit override and fresh validation |
| WMT | 🟢 GREEN ready | bullish | 63.8 | GREEN ready | live Schwab chain Bull Call validated at 0.28 debit; Bullish consumer-defensive setup has broad-tape support but is a less clean risk_on expression, and its UW EOD price tape was slightly negative. | verify live quote and place manually if thesis still holds |
| URA | 🔴 RED no-action | bullish | 30.86 | RED no-action | not actionable: excluded underlying; non-core ETF; not in actionable ETF allowlist (URA) | do not trade from the action list; require explicit override and fresh validation |
| DVN | 🔴 RED no-action | neutral | 59.96 | RED no-action | not actionable: liquid underlying; liquid common stock with sufficient market cap, stock volume, and option open interest | do not trade from the action list; require explicit override and fresh validation |
| OKLO | 🔴 RED no-action | bearish | 31.67 | RED no-action | not actionable: speculative underlying; marketcap_below_20000000000 | do not trade from the action list; require explicit override and fresh validation |
| BA | 🟢 GREEN ready | bullish | 76.93 | GREEN ready | live Schwab chain Bull Call validated at 0.58 debit | verify live quote and place manually if thesis still holds |
| VZ | 🟢 GREEN ready | bearish | 80.32 | GREEN ready | live Schwab chain Bear Put validated at 0.85 debit; built-in agent caution: risk_on: index price tape leans bullish | verify live quote and place manually if thesis still holds |
| UPS | 🟢 GREEN ready | bearish | 82.6 | GREEN ready | live Schwab chain Bear Put validated at 1.45 debit; built-in agent caution: risk_on: index price tape leans bullish | verify live quote and place manually if thesis still holds |
| MRVL | 🟡 YELLOW coverage | bullish | 75.23 | YELLOW coverage | live Schwab chain Bull Call validated at 0.74 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| AMAT | 🟢 GREEN ready | bullish | 76.17 | GREEN ready | live Schwab chain Bull Call validated at 3.90 debit | verify live quote and place manually if thesis still holds |
| SHOP | 🟢 GREEN ready | bullish | 75.46 | GREEN ready | live Schwab chain Bull Call validated at 1.62 debit | verify live quote and place manually if thesis still holds |
| SNOW | 🟡 YELLOW coverage | bullish | 75.43 | YELLOW coverage | live Schwab chain Bull Call validated at 1.07 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| VRT | 🟡 YELLOW coverage | bullish | 75.41 | YELLOW coverage | live Schwab chain Bull Call validated at 5.00 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| CVS | 🟢 GREEN ready | bullish | 74.32 | GREEN ready | live Schwab chain Bull Call validated at 2.04 debit | verify live quote and place manually if thesis still holds |
| BAC | 🟢 GREEN ready | bullish | 66.82 | GREEN ready | live Schwab chain Bull Call validated at 0.28 debit | verify live quote and place manually if thesis still holds |

## Decision Board Summary

Full ranked rows are in `decision_board.csv`; rejected setup-quality rows stay audit-visible in `final_recommendations.csv`.

| Status | Count |
|---|---:|
| blocked | 39 |
| needs_review | 5 |
| ready | 54 |
| waiting_for_price | 37 |

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
- verify the live Schwab quote immediately before any manual order entry
