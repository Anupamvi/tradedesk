# CodexUW Replay Validation

- History days loaded: 160
- Entry days scanned: 148
- Discovery settings: max tickers 60; max candidates 50; max exact-eval candidates 50
- Decision selection: outcome-blind credit edge sleeve capped at 1 per day, plus independent debit sleeve capped at 1 per day
- Breach-evaluated candidates: 6979
- Breach win rate: 58.4%
- Exact-spread evaluated candidates: 3925
- Exact-spread win rate: 68.4%
- Exact-spread avg PnL/spread: $8.27
- Exact-spread max drawdown: $-42,381.90
- Guarded exact-spread evaluated candidates: 81
- Guarded exact-spread win rate: 75.3%
- Guarded exact-spread avg PnL/spread: $14.90
- Guarded exact-spread max drawdown: $-1,256.90
- Decision-selected exact-spread evaluated candidates: 14
- Decision-selected trade days: 14
- Decision-selected win rate: 50.0%
- Decision-selected avg PnL/spread: $-16.76
- Decision-selected max drawdown: $-772.60
- Decision-selected profit factor: 0.759
- Days with any profitable exact candidate: 147/148
- Days with any profitable guarded candidate: 45/148
- Days with a selected profitable trade: 7/148 (4.7%)
- Daily coverage classifications: {'guard_miss': 102, 'ranking_miss': 38, 'selected_profitable': 7, 'no_exact_outcome': 1}
- Selection integrity: entry-time fields only; outcomes are attached after selection and never used to qualify a trade.
- Monthly P/L target: $10,000
- Train/test split day: 2026-05-19
- Fill model: entry at mid less 10%; exits at mid plus 10%; 50% profit target; no hard stop (risk defined by spread width); expiry settlement fallback.

## Train/Test

| Split   |   Rows |   Evaluated | Win Rate   | Avg PnL   | Total PnL   | Max DD      |
|:--------|-------:|------------:|:-----------|:----------|:------------|:------------|
| train   |   2580 |        2580 | 66.0%      | $-12.17   | $-31,391.85 | $-42,381.90 |
| test    |   1345 |        1345 | 72.8%      | $47.49    | $63,870.65  | $-6,129.95  |

## Guarded Train/Test

| Split   |   Rows |   Evaluated | Win Rate   | Avg PnL   | Total PnL   | Max DD     |
|:--------|-------:|------------:|:-----------|:----------|:------------|:-----------|
| train   |     60 |          60 | 68.3%      | $-2.14    | $-128.20    | $-1,256.90 |
| test    |     21 |          21 | 95.2%      | $63.59    | $1,335.40   | $-353.75   |

## Decision-Selected Train/Test

| Split   |   Rows |   Evaluated | Win Rate   | Avg PnL   | Total PnL   | Max DD   |
|:--------|-------:|------------:|:-----------|:----------|:------------|:---------|
| train   |     13 |          13 | 46.2%      | $-24.78   | $-322.20    | $-772.60 |
| test    |      1 |           1 | 100.0%     | $87.60    | $87.60      | $0.00    |

## Monthly Target Feasibility

| Month   |   Trades | Total P/L 1x   | Avg P/L 1x   | Max DD 1x   | Contracts For Target   | Target Feasible 1x   |
|:--------|---------:|:---------------|:-------------|:------------|:-----------------------|:---------------------|
| 2025-12 |        1 | $148.90        | $148.90      | $0.00       | 68                     | False                |
| 2026-01 |        3 | $-626.05       | $-208.68     | $-472.05    | not achievable         | False                |
| 2026-02 |        4 | $-120.25       | $-30.06      | $-156.75    | not achievable         | False                |
| 2026-03 |        1 | $84.70         | $84.70       | $0.00       | 119                    | False                |
| 2026-04 |        2 | $-34.60        | $-17.30      | $0.00       | not achievable         | False                |
| 2026-05 |        2 | $225.10        | $112.55      | $0.00       | 45                     | False                |
| 2026-06 |        1 | $87.60         | $87.60       | $0.00       | 115                    | False                |

## Decision-Selected Trades

| As Of      | Ticker   | Direction   | Strategy                | Expiry     | Entry Credit   | Entry Debit   | Credit % Width   | Debit % Width   | Price Annotation               | Exit              | Exit Day   | P/L 1x   | Decision Reason                            | Decision Tier              |
|:-----------|:---------|:------------|:------------------------|:-----------|:---------------|:--------------|:-----------------|:----------------|:-------------------------------|:------------------|:-----------|:---------|:-------------------------------------------|:---------------------------|
| 2025-12-23 | SBUX     | Bull Call   | Bull Call Debit Spread  | 2026-01-16 |                | $1.67         |                  | 33.4%           | entry_debit_at_or_below_target | profit_target     | 2026-01-06 | $148.90  | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-01-05 | NKE      | Bull Call   | Bull Call Debit Spread  | 2026-01-16 |                | $1.54         |                  | 30.8%           | entry_debit_at_or_below_target | expiry_settlement | 2026-01-16 | $-154.00 | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-01-06 | CRCL     | Bull Call   | Bull Call Debit Spread  | 2026-01-16 |                | $1.16         |                  | 23.2%           | entry_debit_at_or_below_target | expiry_settlement | 2026-01-16 | $-116.05 | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-01-14 | ORCL     | Bull Put    | Bull Put Credit Spread  | 2026-02-20 | $1.44          |               | 28.8%            |                 |                                | expiry_settlement | 2026-02-20 | $-356.00 | decision_selected_credit_edge_sleeve       | credit_volatility_and_rank |
| 2026-02-06 | BAC      | Bull Call   | Bull Call Debit Spread  | 2026-03-20 |                | $0.79         |                  | 31.6%           | entry_debit_at_or_below_target | expiry_settlement | 2026-03-20 | $-79.00  | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-02-09 | UBER     | Bull Call   | Bull Call Debit Spread  | 2026-03-20 |                | $1.68         |                  | 33.6%           | entry_debit_at_or_below_target | profit_target     | 2026-03-17 | $115.50  | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-02-18 | IBM      | Bull Call   | Bull Call Debit Spread  | 2026-03-20 |                | $0.91         |                  | 18.1%           | entry_debit_at_or_below_target | expiry_settlement | 2026-03-20 | $-90.75  | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-02-20 | BAC      | Bull Call   | Bull Call Debit Spread  | 2026-03-20 |                | $0.66         |                  | 26.4%           | entry_debit_at_or_below_target | expiry_settlement | 2026-03-20 | $-66.00  | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-03-04 | MSFT     | Bear Call   | Bear Call Credit Spread | 2026-04-17 | $1.49          |               | 29.7%            |                 |                                | profit_target     | 2026-03-17 | $84.70   | decision_selected_credit_edge_sleeve       | credit_volatility_and_rank |
| 2026-04-14 | NKE      | Bull Call   | Bull Call Debit Spread  | 2026-05-15 |                | $1.11         |                  | 44.4%           | entry_debit_at_or_below_target | expiry_settlement | 2026-05-15 | $-111.00 | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-04-28 | TSM      | Bull Put    | Bull Put Credit Spread  | 2026-05-29 | $1.46          |               | 29.3%            |                 |                                | profit_target     | 2026-05-22 | $76.40   | decision_selected_credit_edge_sleeve       | credit_volatility_and_rank |
| 2026-05-13 | BAC      | Bull Call   | Bull Call Debit Spread  | 2026-06-18 |                | $1.08         |                  | 43.3%           | entry_debit_at_or_below_target | profit_target     | 2026-06-04 | $85.60   | decision_selected_independent_debit_sleeve | directional_debit_medium   |
| 2026-05-19 | ALAB     | Bull Put    | Bull Put Credit Spread  | 2026-06-18 | $1.40          |               | 27.9%            |                 |                                | expiry_settlement | 2026-06-18 | $139.50  | decision_selected_credit_edge_sleeve       | credit_volatility_and_rank |
| 2026-06-12 | RKLB     | Bear Call   | Bear Call Credit Spread | 2026-07-17 | $1.31          |               | 26.1%            |                 |                                | profit_target     | 2026-06-24 | $87.60   | decision_selected_credit_edge_sleeve       | credit_volatility_and_rank |
