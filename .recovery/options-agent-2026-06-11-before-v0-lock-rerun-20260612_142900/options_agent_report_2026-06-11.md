# Options Agent Report - 2026-06-11

Mode: independent UW + Schwab live research options-agent-v0.

Target rows show desired credits/debits. Only rows in Send Now Orders with ready_to_enter=true are executable.

## Execution Snapshot

- Green send-now rows: 49
- Yellow target rows: 34
- Target refresh queue rows: 0
- Next action: review Send Now Orders; enter manually only after final quote check
- Live quote mode: live_schwab
- Portfolio context: ok
- Profitability confidence: 3.0/10
- Order-entry confidence: 5.0/10
- Profitability calibration: block (0 pass / 135 current rows)
- Calibration blockers: actual_support=BLOCK:102, WARN:33; replay_bucket=BLOCK:99, WARN:36; family_only_actual_rows=130; bucket_shortfall_rows=135 routes=bear_call_credit,bear_put_debit,bull_call_debit,bull_put_credit; missing_replay_bucket_rows=97 routes=bear_call_credit,bear_put_debit,bull_call_debit,bull_put_credit
- Bucket blocker examples: ABBV bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket,liquidity_bucket; SBUX bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket; XLV bull_call_debit/bullish/dte_15_30/debit_reward_risk_high actual=WARN sample=1 gap=29 replay=BLOCK sample=0 gap=30 missing_dims=economics_bucket
- Profitability gap plan: actual_closed_outcomes_negative_or_weak:25, actual_closed_outcomes_sample_gap:15; top=AMZN,BA,BAC,CVX,GOOG,GOOGL,HD,IWM,JPM,NFLX,NVDA,ORCL,PLTR,SNOW,TSLA,V,VRT,WMT bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=liquidity_bucket; AAPL,BMY,CMCSA,DIS,KO,UBER,XOM bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=dte_bucket,liquidity_bucket; AMD,AVGO,BKNG,CRM,CSCO,INTC,META,MO,MRVL,MSFT,NOW,QCOM,SNOW bull_call_debit actual_closed_outcomes_negative_or_weak actual_gap=0 replay_gap=30 relaxed=liquidity_bucket
- Calibrated order-entry blockers: no calibrated rows
- Execution fill quality: review (111 pass / 22 block)
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
| AMD | 🟢 GREEN ready | Put credit spread | 2026-06-18 | SELL 1 AMD 2026-06-18 492.5 Put | BUY 1 AMD 2026-06-18 487.5 Put | 3 | 1.51 CREDIT | 0.53 | 453.0 | 1047.0 | HIGH / 100 | green ready; verify live quote before manual send |
| AMZN | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 AMZN 2026-06-18 245 Call | BUY 1 AMZN 2026-06-18 242.5 Call | 5 | 0.67 DEBIT | 1.21 | 915.0 | 335.0 | HIGH / 100 | green ready; verify live quote before manual send |
| GOOG | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 GOOG 2026-06-18 375 Call | BUY 1 GOOG 2026-06-18 372.5 Call | 5 | 0.56 DEBIT | 1.01 | 970.0 | 280.0 | HIGH / 99 | green ready; verify live quote before manual send |
| GOOGL | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 GOOGL 2026-06-18 375 Call | BUY 1 GOOGL 2026-06-18 372.5 Call | 5 | 0.64 DEBIT | 1.15 | 930.0 | 320.0 | HIGH / 99 | green ready; verify live quote before manual send |
| NFLX | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 NFLX 2026-06-18 87 Call | BUY 1 NFLX 2026-06-18 82 Call | 5 | 0.5 DEBIT | 0.9 | 2250.0 | 250.0 | HIGH / 97 | green ready; verify live quote before manual send |
| NFLX | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 NFLX 2026-06-18 83 Call | BUY 1 NFLX 2026-06-18 82 Call | 5 | 0.22 DEBIT | 0.4 | 390.0 | 110.0 | HIGH / 97 | green ready; verify live quote before manual send |
| IWM | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 IWM 2026-06-18 302 Call | BUY 1 IWM 2026-06-18 301 Call | 5 | 0.28 DEBIT | 0.5 | 360.0 | 140.0 | HIGH / 96 | green ready; verify live quote before manual send |
| NVDA | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 NVDA 2026-06-18 215 Call | BUY 1 NVDA 2026-06-18 213 Call | 5 | 0.43 DEBIT | 0.77 | 785.0 | 215.0 | HIGH / 95 | green ready; verify live quote before manual send |
| MRVL | 🟢 GREEN ready | Put credit spread | 2026-06-18 | SELL 1 MRVL 2026-06-18 270 Put | BUY 1 MRVL 2026-06-18 267.5 Put | 5 | 0.79 CREDIT | 0.28 | 395.0 | 855.0 | HIGH / 95 | green ready; verify live quote before manual send |
| BA | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 BA 2026-06-18 227.5 Call | BUY 1 BA 2026-06-18 225 Call | 5 | 0.65 DEBIT | 1.17 | 925.0 | 325.0 | HIGH / 95 | green ready; verify live quote before manual send |
| BAC | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 BAC 2026-06-18 59 Call | BUY 1 BAC 2026-06-18 57 Call | 5 | 0.35 DEBIT | 0.63 | 825.0 | 175.0 | HIGH / 95 | green ready; verify live quote before manual send |
| V | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 V 2026-06-18 335 Call | BUY 1 V 2026-06-18 330 Call | 5 | 1.25 DEBIT | 2.25 | 1875.0 | 625.0 | HIGH / 95 | green ready; verify live quote before manual send |
| BAC | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 BAC 2026-06-18 58 Call | BUY 1 BAC 2026-06-18 57 Call | 5 | 0.25 DEBIT | 0.45 | 375.0 | 125.0 | HIGH / 95 | green ready; verify live quote before manual send |
| CVS | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 CVS 2026-07-17 115 Call | BUY 1 CVS 2026-07-17 105 Call | 4 | 2.43 DEBIT | 4.37 | 3028.0 | 972.0 | HIGH / 94 | green ready; verify live quote before manual send |
| IBM | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 IBM 2026-07-17 310 Call | BUY 1 IBM 2026-07-17 290 Call | 2 | 5.09 DEBIT | 9.16 | 2982.0 | 1018.0 | HIGH / 94 | green ready; verify live quote before manual send |
| JPM | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 JPM 2026-06-18 330 Call | BUY 1 JPM 2026-06-18 327.5 Call | 5 | 0.68 DEBIT | 1.22 | 910.0 | 340.0 | HIGH / 94 | green ready; verify live quote before manual send |
| BMY | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 BMY 2026-07-17 62.5 Call | BUY 1 BMY 2026-07-17 60 Call | 5 | 0.49 DEBIT | 0.88 | 1005.0 | 245.0 | HIGH / 94 | green ready; verify live quote before manual send |
| CVX | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 CVX 2026-06-18 195 Call | BUY 1 CVX 2026-06-18 192.5 Call | 5 | 0.53 DEBIT | 0.95 | 985.0 | 265.0 | HIGH / 94 | green ready; verify live quote before manual send |
| DIS | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 DIS 2026-07-17 110 Call | BUY 1 DIS 2026-07-17 105 Call | 5 | 0.84 DEBIT | 1.51 | 2080.0 | 420.0 | HIGH / 94 | green ready; verify live quote before manual send |
| NEM | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 NEM 2026-07-17 110 Call | BUY 1 NEM 2026-07-17 105 Call | 5 | 1.46 DEBIT | 2.63 | 1770.0 | 730.0 | HIGH / 94 | green ready; verify live quote before manual send |
| ABBV | 🟢 GREEN ready | Call debit spread | 2026-06-26 | SELL 1 ABBV 2026-06-26 235 Call | BUY 1 ABBV 2026-06-26 230 Call | 5 | 1.3 DEBIT | 2.34 | 1850.0 | 650.0 | HIGH / 94 | green ready; verify live quote before manual send |
| UBER | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 UBER 2026-07-17 75 Call | BUY 1 UBER 2026-07-17 72.5 Call | 5 | 0.62 DEBIT | 1.12 | 940.0 | 310.0 | HIGH / 94 | green ready; verify live quote before manual send |
| DIA | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 DIA 2026-07-17 535 Call | BUY 1 DIA 2026-07-17 525 Call | 4 | 2.85 DEBIT | 5.13 | 2860.0 | 1140.0 | HIGH / 94 | green ready; verify live quote before manual send |
| META | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 META 2026-06-18 592.5 Call | BUY 1 META 2026-06-18 587.5 Call | 5 | 1.1 DEBIT | 1.98 | 1950.0 | 550.0 | HIGH / 94 | green ready; verify live quote before manual send |
| NEE | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 NEE 2026-07-17 92.5 Call | BUY 1 NEE 2026-07-17 87.5 Call | 5 | 1.45 DEBIT | 2.61 | 1775.0 | 725.0 | HIGH / 94 | green ready; verify live quote before manual send |
| CMCSA | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 CMCSA 2026-07-17 27.5 Call | BUY 1 CMCSA 2026-07-17 26 Call | 5 | 0.24 DEBIT | 0.43 | 630.0 | 120.0 | HIGH / 94 | green ready; verify live quote before manual send |
| CSCO | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 CSCO 2026-06-18 127 Call | BUY 1 CSCO 2026-06-18 126 Call | 5 | 0.23 DEBIT | 0.41 | 385.0 | 115.0 | HIGH / 94 | green ready; verify live quote before manual send |
| AMAT | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 AMAT 2026-07-17 590 Call | BUY 1 AMAT 2026-07-17 580 Call | 3 | 3.38 DEBIT | 6.08 | 1986.0 | 1014.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| SHOP | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 SHOP 2026-07-17 120 Call | BUY 1 SHOP 2026-07-17 115 Call | 5 | 1.68 DEBIT | 3.02 | 1660.0 | 840.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| CSCO | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 CSCO 2026-06-18 130 Call | BUY 1 CSCO 2026-06-18 120 Call | 3 | 3.8 DEBIT | 6.84 | 1860.0 | 1140.0 | MEDIUM / 94 | green ready; verify live quote before manual send |
| VZ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 VZ 2026-07-17 43 Put | BUY 1 VZ 2026-07-17 46 Put | 5 | 0.64 DEBIT | 1.15 | 1180.0 | 320.0 | MEDIUM / 91 | green ready; verify live quote before manual send |
| VZ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 VZ 2026-07-17 45 Put | BUY 1 VZ 2026-07-17 46 Put | 5 | 0.31 DEBIT | 0.56 | 345.0 | 155.0 | MEDIUM / 91 | green ready; verify live quote before manual send |
| SLV | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 SLV 2026-07-17 55.5 Put | BUY 1 SLV 2026-07-17 56 Put | 5 | 0.15 DEBIT | 0.27 | 175.0 | 75.0 | MEDIUM / 90 | green ready; verify live quote before manual send |
| SPY | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 SPY 2026-06-18 753 Call | BUY 1 SPY 2026-06-18 752 Call | 5 | 0.32 DEBIT | 0.58 | 340.0 | 160.0 | MEDIUM / 90 | green ready; verify live quote before manual send |
| WMT | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 WMT 2026-06-18 123 Call | BUY 1 WMT 2026-06-18 122 Call | 5 | 0.27 DEBIT | 0.49 | 365.0 | 135.0 | MEDIUM / 90 | green ready; verify live quote before manual send |
| PG | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 PG 2026-07-17 140 Put | BUY 1 PG 2026-07-17 145 Put | 5 | 1.16 DEBIT | 2.09 | 1920.0 | 580.0 | MEDIUM / 89 | green ready; verify live quote before manual send |
| WFC | 🟢 GREEN ready | Put debit spread | 2026-06-18 | SELL 1 WFC 2026-06-18 80 Put | BUY 1 WFC 2026-06-18 82 Put | 5 | 0.33 DEBIT | 0.59 | 835.0 | 165.0 | MEDIUM / 89 | green ready; verify live quote before manual send |
| QQQ | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 QQQ 2026-06-18 740 Call | BUY 1 QQQ 2026-06-18 739 Call | 5 | 0.31 DEBIT | 0.56 | 345.0 | 155.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| MSFT | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 MSFT 2026-06-18 402.5 Call | BUY 1 MSFT 2026-06-18 400 Call | 5 | 0.51 DEBIT | 0.92 | 995.0 | 255.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| CRM | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 CRM 2026-06-18 172.5 Call | BUY 1 CRM 2026-06-18 170 Call | 5 | 0.65 DEBIT | 1.17 | 925.0 | 325.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| T | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 T 2026-07-17 25 Call | BUY 1 T 2026-07-17 24 Call | 5 | 0.3 DEBIT | 0.54 | 350.0 | 150.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| C | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 C 2026-06-18 145 Call | BUY 1 C 2026-06-18 144 Call | 5 | 0.3 DEBIT | 0.54 | 350.0 | 150.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| PFE | 🟢 GREEN ready | Call debit spread | 2026-06-18 | SELL 1 PFE 2026-06-18 27 Call | BUY 1 PFE 2026-06-18 26.5 Call | 5 | 0.16 DEBIT | 0.29 | 170.0 | 80.0 | MEDIUM / 88 | green ready; verify live quote before manual send |
| MU | 🟢 GREEN ready | Put credit spread | 2026-06-18 | SELL 1 MU 2026-06-18 945 Put | BUY 1 MU 2026-06-18 940 Put | 3 | 1.73 CREDIT | 0.61 | 519.0 | 981.0 | MEDIUM / 86 | green ready; verify live quote before manual send |
| XLU | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 XLU 2026-07-17 42 Put | BUY 1 XLU 2026-07-17 43 Put | 5 | 0.21 DEBIT | 0.38 | 395.0 | 105.0 | MEDIUM / 86 | green ready; verify live quote before manual send |
| XOM | 🟢 GREEN ready | Call debit spread | 2026-07-17 | SELL 1 XOM 2026-07-17 160 Call | BUY 1 XOM 2026-07-17 155 Call | 5 | 1.35 DEBIT | 2.43 | 1825.0 | 675.0 | MEDIUM / 84 | green ready; verify live quote before manual send |
| JNJ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 JNJ 2026-07-17 210 Put | BUY 1 JNJ 2026-07-17 230 Put | 4 | 2.97 DEBIT | 5.35 | 6812.0 | 1188.0 | MEDIUM / 83 | green ready; verify live quote before manual send |
| JNJ | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 JNJ 2026-07-17 220 Put | BUY 1 JNJ 2026-07-17 230 Put | 5 | 1.94 DEBIT | 3.49 | 4030.0 | 970.0 | MEDIUM / 83 | green ready; verify live quote before manual send |
| UPS | 🟢 GREEN ready | Put debit spread | 2026-07-17 | SELL 1 UPS 2026-07-17 100 Put | BUY 1 UPS 2026-07-17 105 Put | 5 | 1.41 DEBIT | 2.54 | 1795.0 | 705.0 | MEDIUM / 82 | green ready; verify live quote before manual send |

## Target Orders - Target Credits/Debits

These are planning targets. Use the shown desired credit/debit as the starting limit, then refresh the Schwab quote before sending.

| Ticker | Signal | Structure | Exp | Sell Leg | Buy Leg | Qty | Target Limit | Target Exit | Max Profit | Max Loss | Confidence | Price / Risk |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| AMZN | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 AMZN 2026-06-18 232.5 Put | BUY 1 AMZN 2026-06-18 227.5 Put | 3 | 1.18 CREDIT | 0.41 | 354.0 | 1146.0 | HIGH / 98 | credit/width too weak for send-now |
| AMD | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 AMD 2026-06-18 550 Call | BUY 1 AMD 2026-06-18 547.5 Call | 5 | 0.58 DEBIT | 1.04 | 960.0 | 290.0 | HIGH / 93 | breakeven move too large for send-now |
| GOOG | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 GOOG 2026-06-18 355 Put | BUY 1 GOOG 2026-06-18 350 Put | 3 | 1.28 CREDIT | 0.45 | 384.0 | 1116.0 | HIGH / 91 | credit/width too weak for send-now |
| GOOGL | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 GOOGL 2026-06-18 355 Put | BUY 1 GOOGL 2026-06-18 352.5 Put | 5 | 0.6 CREDIT | 0.21 | 300.0 | 950.0 | HIGH / 91 | credit/width too weak for send-now |
| INTC | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 INTC 2026-06-18 133 Call | BUY 1 INTC 2026-06-18 132 Call | 5 | 0.28 DEBIT | 0.5 | 360.0 | 140.0 | HIGH / 90 | breakeven move too large for send-now |
| INTC | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 INTC 2026-06-18 118 Put | BUY 1 INTC 2026-06-18 117 Put | 5 | 0.34 CREDIT | 0.12 | 170.0 | 330.0 | HIGH / 90 | credit too small for send-now |
| MRVL | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 MRVL 2026-06-18 310 Call | BUY 1 MRVL 2026-06-18 307.5 Call | 5 | 0.47 DEBIT | 0.85 | 1015.0 | 235.0 | HIGH / 87 | breakeven move too large for send-now |
| AVGO | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 AVGO 2026-06-18 400 Call | BUY 1 AVGO 2026-06-18 397.5 Call | 5 | 0.52 DEBIT | 0.94 | 990.0 | 260.0 | HIGH / 87 | breakeven move too large for send-now |
| AVGO | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 AVGO 2026-06-18 407.5 Call | BUY 1 AVGO 2026-06-18 397.5 Call | 5 | 2.19 DEBIT | 3.94 | 3905.0 | 1095.0 | HIGH / 87 | breakeven move too large for send-now |
| NVDA | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 NVDA 2026-06-18 200 Put | BUY 1 NVDA 2026-06-18 197.5 Put | 5 | 0.55 CREDIT | 0.19 | 275.0 | 975.0 | MEDIUM / 87 | credit/width too weak for send-now |
| BA | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 BA 2026-06-18 215 Put | BUY 1 BA 2026-06-18 210 Put | 3 | 0.99 CREDIT | 0.35 | 297.0 | 1203.0 | MEDIUM / 87 | credit/width too weak for send-now |
| SNOW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 SNOW 2026-06-18 255 Call | BUY 1 SNOW 2026-06-18 250 Call | 5 | 1.22 DEBIT | 2.2 | 1890.0 | 610.0 | HIGH / 86 | breakeven move too large for send-now |
| SNOW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 SNOW 2026-06-18 260 Call | BUY 1 SNOW 2026-06-18 250 Call | 5 | 2 DEBIT | 3.6 | 4000.0 | 1000.0 | HIGH / 86 | breakeven move too large for send-now |
| QCOM | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 QCOM 2026-06-18 230 Call | BUY 1 QCOM 2026-06-18 227.5 Call | 5 | 0.52 DEBIT | 0.94 | 990.0 | 260.0 | HIGH / 86 | breakeven move too large for send-now |
| QCOM | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 QCOM 2026-06-18 237.5 Call | BUY 1 QCOM 2026-06-18 227.5 Call | 5 | 1.91 DEBIT | 3.44 | 4045.0 | 955.0 | HIGH / 86 | breakeven move too large for send-now |
| NOW | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 NOW 2026-06-18 108 Call | BUY 1 NOW 2026-06-18 107 Call | 5 | 0.22 DEBIT | 0.4 | 390.0 | 110.0 | HIGH / 86 | breakeven move too large for send-now |
| UBER | 🟡 YELLOW target | Put credit spread | 2026-07-17 | SELL 1 UBER 2026-07-17 65 Put | BUY 1 UBER 2026-07-17 62.5 Put | 5 | 0.63 CREDIT | 0.22 | 315.0 | 935.0 | HIGH / 86 | credit/width too weak for send-now |
| META | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 META 2026-06-18 557.5 Put | BUY 1 META 2026-06-18 555 Put | 5 | 0.7 CREDIT | 0.24 | 350.0 | 900.0 | HIGH / 86 | credit/width too weak for send-now |
| PLTR | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 PLTR 2026-06-18 136 Call | BUY 1 PLTR 2026-06-18 135 Call | 5 | 0.23 DEBIT | 0.41 | 385.0 | 115.0 | HIGH / 86 | breakeven move too large for send-now |
| NOW | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 NOW 2026-06-18 97 Put | BUY 1 NOW 2026-06-18 92 Put | 3 | 0.99 CREDIT | 0.35 | 297.0 | 1203.0 | MEDIUM / 86 | credit/width too weak for send-now |
| JPM | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 JPM 2026-06-18 315 Put | BUY 1 JPM 2026-06-18 310 Put | 2 | 0.96 CREDIT | 0.34 | 192.0 | 808.0 | MEDIUM / 86 | credit/width too weak for send-now |
| ANET | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 ANET 2026-06-18 157.5 Put | BUY 1 ANET 2026-06-18 152.5 Put | 3 | 1.1 CREDIT | 0.39 | 330.0 | 1170.0 | MEDIUM / 86 | credit/width too weak for send-now |
| IWM | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 IWM 2026-06-18 289 Put | BUY 1 IWM 2026-06-18 288 Put | 5 | 0.25 CREDIT | 0.09 | 125.0 | 375.0 | HIGH / 80 | credit too small for send-now; credit/width too weak for send-now |
| MSFT | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 MSFT 2026-06-18 380 Put | BUY 1 MSFT 2026-06-18 375 Put | 3 | 1.17 CREDIT | 0.41 | 351.0 | 1149.0 | MEDIUM / 80 | credit/width too weak for send-now |
| CRM | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 CRM 2026-06-18 160 Put | BUY 1 CRM 2026-06-18 155 Put | 3 | 1.11 CREDIT | 0.39 | 333.0 | 1167.0 | MEDIUM / 80 | credit/width too weak for send-now |
| PEP | 🟡 YELLOW target | Call credit spread | 2026-07-17 | SELL 1 PEP 2026-07-17 150 Call | BUY 1 PEP 2026-07-17 155 Call | 3 | 0.99 CREDIT | 0.35 | 297.0 | 1203.0 | MEDIUM / 79 | credit/width too weak for send-now |
| PLTR | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 PLTR 2026-06-18 126 Put | BUY 1 PLTR 2026-06-18 125 Put | 5 | 0.29 CREDIT | 0.1 | 145.0 | 355.0 | HIGH / 78 | credit too small for send-now; credit/width too weak for send-now |
| MU | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 MU 2026-06-18 1075 Call | BUY 1 MU 2026-06-18 1070 Call | 5 | 1.49 DEBIT | 2.68 | 1755.0 | 745.0 | MEDIUM / 78 | breakeven move too large for send-now |
| XLI | 🟡 YELLOW target | Put credit spread | 2026-07-17 | SELL 1 XLI 2026-07-17 171 Put | BUY 1 XLI 2026-07-17 169 Put | 5 | 0.4 CREDIT | 0.14 | 200.0 | 800.0 | MEDIUM / 78 | credit too small for send-now; credit/width too weak for send-now |
| TSLA | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 TSLA 2026-06-18 422.5 Call | BUY 1 TSLA 2026-06-18 420 Call | 5 | 0.55 DEBIT | 0.99 | 975.0 | 275.0 | MEDIUM / 76 | breakeven move too large for send-now |
| TSLA | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 TSLA 2026-06-18 385 Put | BUY 1 TSLA 2026-06-18 382.5 Put | 5 | 0.72 CREDIT | 0.25 | 360.0 | 890.0 | MEDIUM / 76 | credit/width too weak for send-now |
| ORCL | 🟡 YELLOW target | Call debit spread | 2026-06-18 | SELL 1 ORCL 2026-06-18 195 Call | BUY 1 ORCL 2026-06-18 192.5 Call | 5 | 0.59 DEBIT | 1.06 | 955.0 | 295.0 | MEDIUM / 73 | breakeven move too large for send-now |
| ORCL | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 ORCL 2026-06-18 177.5 Put | BUY 1 ORCL 2026-06-18 175 Put | 5 | 0.72 CREDIT | 0.25 | 360.0 | 890.0 | MEDIUM / 73 | credit/width too weak for send-now |
| QQQ | 🟡 YELLOW target | Put credit spread | 2026-06-18 | SELL 1 QQQ 2026-06-18 709 Put | BUY 1 QQQ 2026-06-18 708 Put | 5 | 0.27 CREDIT | 0.09 | 135.0 | 365.0 | MEDIUM / 72 | credit too small for send-now; credit/width too weak for send-now |

## Run Diagnostics

Diagnostics explain confidence and coverage; the order-entry surface is Send Now Orders plus `trade_tickets.csv`.

- Trade rows: 49 green send-now, 34 target-order candidates
- Send-now readiness: execution_ready; non-green gates: []
- Live quote mode: live_schwab; live validation rows: 135
- Live spread quality audit: blocked_bad_live_markets; 17 blocked (12 quote-width, 5 liquidity)
- Agentic review coverage: lane 5/5 (1.0); broad rows 78/3916 (0.0199)
- Structure attempt rows: 270
- Final visible rows: 135
- Structural status counts, not order readiness: {'AVOID': 35, 'ENTER': 65, 'ENTER_WITH_PORTFOLIO_RISK': 28, 'REVIEW': 3, 'WAIT_FOR_PRICE': 4}
- Portfolio context: ok
- Raw discovery: 6239 UW rows, 3916 generated candidates, 3918 catalyst rows, 8117 review rows
- Agentic dispatch tasks: 5; review status: reviews_ingested
- Agent review verdicts: {'avoid': 63, 'caution': 1869, 'supportive': 6185}; objective blockers: 62
- Strategy outcome atlas: positive families ['short_put']; negative current families ['vertical_spread']; blocking current ticker-strategy rows 55
- Route opportunity gaps: candidate_expansion=short_put; actual_weak=bull_call_debit

## Execution Quality Gates

- Execution confidence ratings: {'NOT_EXECUTION_READY': 78, 'HIGH': 52, 'MEDIUM': 5}
- Trade-quality confidence ratings: {'HIGH': 48, 'LOW': 44, 'MEDIUM': 43}
- Top non-green send-now gates: {'positive_contract_size_required': 37, 'objective_blocker': 35, 'credit/width too weak for send-now': 20, 'breakeven move too large for send-now': 14, 'credit too small for send-now': 5, 'fresh Schwab chain': 5, 'wait_for_price': 4, 'manual_review_required': 2}

## Focus Review Queue - Not Trades

These are not orders. This section is limited to validated rows and focus tickers; tail unvalidated rows stay in CSV artifacts.

| Ticker | Signal | Reason | Qty | Target Limit | Max Loss | Trade Plan |
|---|---|---|---:|---:|---:|---|
| SLB | 🟡 YELLOW review | live Schwab chain Bear Put validated at 1.43 debit; built-in agent caution: risk_on: index price tape leans bullish | 5 | 1.43 DEBIT | 715.0 | BUY 1 SLB 2026-06-18 57.5 Put / SELL 1 SLB 2026-06-18 54 Put @ 1.43 DEBIT |
| ABT | 🟡 YELLOW review | live Schwab chain found 2.34 debit above target 2.25; built-in agent caution: live Schwab chain found 2.34 debit above target 2.25 | 5 | 2.34 DEBIT | 1170.0 | BUY 1 ABT 2026-06-18 86 Call / SELL 1 ABT 2026-06-18 91 Call @ 2.34 DEBIT |
| FCX | 🟡 YELLOW review | live Schwab chain Bear Call validated at 0.96 credit; built-in agent caution: risk_on: index price tape leans bullish | 2 | 0.96 CREDIT | 808.0 | SELL 1 FCX 2026-07-17 75 Call / BUY 1 FCX 2026-07-17 80 Call @ 0.96 CREDIT |
| GLW | 🟡 YELLOW review | live Schwab chain found 2.34 debit above target 2.25; built-in agent caution: live Schwab chain found 2.34 debit above target 2.25 | 5 | 2.34 DEBIT | 1170.0 | BUY 1 GLW 2026-06-18 177.5 Call / SELL 1 GLW 2026-06-18 182.5 Call @ 2.34 DEBIT |
| COP | 🟡 YELLOW review | live Schwab chain found 4.54 debit above target 4.50; built-in agent caution: risk_on: index price tape leans bullish | 2 | 4.54 DEBIT | 908.0 | BUY 1 COP 2026-07-17 120 Put / SELL 1 COP 2026-07-17 110 Put @ 4.54 DEBIT |
| XLY | 🟡 YELLOW review | live Schwab chain found 2.53 debit above target 2.25; built-in agent caution: live Schwab chain found 2.53 debit above target 2.25 | 4 | 2.53 DEBIT | 1012.0 | BUY 1 XLY 2026-07-17 115 Call / SELL 1 XLY 2026-07-17 120 Call @ 2.53 DEBIT |

## Coverage Audit

Coverage rows explain inclusion/exclusion only. They are not orders; use Send Now Orders, Target Orders, and `trade_tickets.csv` for the action surface.

| Ticker | Signal | Bias | Score | State | Why | Next Step |
|---|---|---|---:|---|---|---|
| AAPL | 🔴 RED blocked | bullish | 72.73 | RED blocked | setup quality gate reject: directional_bias_below_0.10 | do not trade unless the objective blocker is cleared in a fresh run |
| MSFT | 🟢 GREEN ready | bullish | 74.73 | GREEN ready | live Schwab chain Bull Call validated at 0.51 debit | verify live quote and place manually if thesis still holds |
| NVDA | 🟢 GREEN ready | bullish | 68.44 | GREEN ready | live Schwab chain Bull Call validated at 0.43 debit | verify live quote and place manually if thesis still holds |
| AMZN | 🟢 GREEN ready | bullish | 68.55 | GREEN ready | live Schwab chain Bull Call validated at 0.67 debit | verify live quote and place manually if thesis still holds |
| META | 🟢 GREEN ready | bullish | 68.74 | GREEN ready | live Schwab chain Bull Call validated at 1.10 debit | verify live quote and place manually if thesis still holds |
| GOOG | 🟢 GREEN ready | bullish | 77.68 | GREEN ready | live Schwab chain Bull Call validated at 0.56 debit | verify live quote and place manually if thesis still holds |
| GOOGL | 🟢 GREEN ready | bullish | 67 | GREEN ready | live Schwab chain Bull Call validated at 0.64 debit | verify live quote and place manually if thesis still holds |
| TSLA | 🟡 YELLOW coverage | bullish | 68.58 | YELLOW coverage | live Schwab chain Bull Call validated at 0.55 debit; TSLA adds high-beta consumer/mega-cap risk on top of QQQ/SPY/AMZN-style exposure; keep it small or choose it instead of broad beta, not in addition to every beta leg. | use the shown target limit as the starting point; adjust if the live quote moves |
| AMD | 🟡 YELLOW coverage | bullish | 73.71 | YELLOW coverage | live Schwab chain Bull Call validated at 0.58 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| AVGO | 🟡 YELLOW coverage | bullish | 70.81 | YELLOW coverage | live Schwab chain Bull Call validated at 0.52 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| SPY | 🟢 GREEN ready | bullish | 70.35 | GREEN ready | live Schwab chain Bull Call validated at 0.32 debit; Broad-market bullish exposure overlaps with QQQ, IWM, XLK, TSLA, AMZN, META, MSFT, NVDA and other beta-heavy candidates; treat as aggregate portfolio beta, not an independent setup. | verify live quote and place manually if thesis still holds |
| QQQ | 🟢 GREEN ready | bullish | 75.06 | GREEN ready | live Schwab chain Bull Call validated at 0.31 debit; Crowded beta expression: do not stack QQQ with SPY, XLK, and multiple bullish mega-cap tech or semiconductor tickets unless total index/tech risk is capped and the intended exposure is explicit.; The displayed top source leg is same-day and effectively unusable for next-day planning. Rebuild from fresh Schwab chain only | verify live quote and place manually if thesis still holds |
| IWM | 🟢 GREEN ready | bullish | 74.05 | GREEN ready | live Schwab chain Bull Call validated at 0.28 debit | verify live quote and place manually if thesis still holds |
| DIA | 🟢 GREEN ready | bullish | 63.46 | GREEN ready | live Schwab chain Bull Call validated at 2.85 debit | verify live quote and place manually if thesis still holds |
| PLTR | 🟡 YELLOW coverage | bullish | 68.02 | YELLOW coverage | live Schwab chain Bull Call validated at 0.23 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| HOOD | 🔴 RED no-action | bullish | 69.62 | RED no-action | not actionable: liquid underlying; liquid common stock with sufficient market cap, stock volume, and option open interest | do not trade from the action list; require explicit override and fresh validation |
| WMT | 🟢 GREEN ready | bullish | 63.8 | GREEN ready | live Schwab chain Bull Call validated at 0.27 debit; Bullish consumer-defensive setup has broad-tape support but is a less clean risk_on expression, and its UW EOD price tape was slightly negative. | verify live quote and place manually if thesis still holds |
| URA | 🔴 RED no-action | bullish | 30.86 | RED no-action | not actionable: excluded underlying; non-core ETF; not in actionable ETF allowlist (URA) | do not trade from the action list; require explicit override and fresh validation |
| DVN | 🔴 RED no-action | neutral | 59.96 | RED no-action | not actionable: liquid underlying; liquid common stock with sufficient market cap, stock volume, and option open interest | do not trade from the action list; require explicit override and fresh validation |
| OKLO | 🔴 RED no-action | bearish | 31.67 | RED no-action | not actionable: speculative underlying; marketcap_below_20000000000 | do not trade from the action list; require explicit override and fresh validation |
| BA | 🟢 GREEN ready | bullish | 76.93 | GREEN ready | live Schwab chain Bull Call validated at 0.65 debit | verify live quote and place manually if thesis still holds |
| VZ | 🟢 GREEN ready | bearish | 80.32 | GREEN ready | live Schwab chain Bear Put validated at 0.64 debit; built-in agent caution: risk_on: index price tape leans bullish | verify live quote and place manually if thesis still holds |
| UPS | 🟢 GREEN ready | bearish | 82.6 | GREEN ready | live Schwab chain Bear Put validated at 1.41 debit; built-in agent caution: risk_on: index price tape leans bullish | verify live quote and place manually if thesis still holds |
| MRVL | 🟡 YELLOW coverage | bullish | 75.23 | YELLOW coverage | live Schwab chain Bull Call validated at 0.47 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| AMAT | 🟢 GREEN ready | bullish | 76.17 | GREEN ready | live Schwab chain Bull Call validated at 3.38 debit | verify live quote and place manually if thesis still holds |
| SHOP | 🟢 GREEN ready | bullish | 75.46 | GREEN ready | live Schwab chain Bull Call validated at 1.68 debit | verify live quote and place manually if thesis still holds |
| SNOW | 🟡 YELLOW coverage | bullish | 75.43 | YELLOW coverage | live Schwab chain Bull Call validated at 1.22 debit | use the shown target limit as the starting point; adjust if the live quote moves |
| CVS | 🟢 GREEN ready | bullish | 74.32 | GREEN ready | live Schwab chain Bull Call validated at 2.43 debit | verify live quote and place manually if thesis still holds |
| BAC | 🟢 GREEN ready | bullish | 66.82 | GREEN ready | live Schwab chain Bull Call validated at 0.35 debit | verify live quote and place manually if thesis still holds |
| QCOM | 🟡 YELLOW coverage | bullish | 70.94 | YELLOW coverage | live Schwab chain Bull Call validated at 0.52 debit | use the shown target limit as the starting point; adjust if the live quote moves |

## Decision Board Summary

Full ranked rows are in `decision_board.csv`; rejected setup-quality rows stay audit-visible in `final_recommendations.csv`.

| Status | Count |
|---|---:|
| blocked | 35 |
| needs_review | 3 |
| ready | 57 |
| waiting_for_price | 40 |

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
