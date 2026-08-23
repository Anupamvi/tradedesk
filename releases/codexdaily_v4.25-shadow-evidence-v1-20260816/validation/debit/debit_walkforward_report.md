# Debit Walk-Forward Shadow V2

- Status: **RESEARCH_ONLY**
- Execution authority: **NO**
- Exit policy: 100% profit target; no hard stop; 10% adverse entry-fill stress
- Development: n=8; PF=3.163804005764536; P/L=$735.71
- Holdout: n=3; PF=None; P/L=$665.65
- Blockers: development_sample_below_20, holdout_sample_below_15, total_selected_sample_below_50, development_wilson_lower_below_0.55, holdout_wilson_lower_below_0.55, holdout_bear_put_debit_spread_sample_below_10, holdout_bull_call_debit_spread_sample_below_10

## Selected Shadow Trades

| asof                | ticker   | strategy               | entry_day           | expiry              |   entry_debit |   entry_width |   predicted_win_probability |   predicted_ev_payoff_correct |   stress_pnl_10pct | regime    | entry_timing                        | oi_carryover_status   |   technical_confirmation_count |   flow_confirmation_count |
|:--------------------|:---------|:-----------------------|:--------------------|:--------------------|--------------:|--------------:|----------------------------:|------------------------------:|-------------------:|:----------|:------------------------------------|:----------------------|-------------------------------:|--------------------------:|
| 2026-03-27 00:00:00 | PYPL     | Bull Call Debit Spread | 2026-03-30 00:00:00 | 2026-04-17 00:00:00 |        1      |           2.5 |                    0.598421 |                       9.68411 |            102.4   | downtrend | next_session_hot_chain_eod_fallback | supportive            |                              1 |                         1 |
| 2026-04-02 00:00:00 | NKE      | Bull Call Debit Spread | 2026-04-06 00:00:00 | 2026-05-15 00:00:00 |        0.88   |           2.5 |                    0.658085 |                      19.023   |            -96.8   | range     | next_session_hot_chain_eod_fallback | mixed                 |                              0 |                         2 |
| 2026-04-15 00:00:00 | C        | Bear Put Debit Spread  | 2026-04-16 00:00:00 | 2026-05-15 00:00:00 |        2.2825 |           5   |                    0.860417 |                     141.705   |            248.925 | uptrend   | next_session_hot_chain_eod_fallback | supportive            |                              0 |                         1 |
| 2026-04-23 00:00:00 | BAC      | Bear Put Debit Spread  | 2026-04-24 00:00:00 | 2026-05-15 00:00:00 |        0.9515 |           2.5 |                    0.726203 |                      33.5314  |             89.285 | range     | next_session_first_regular_nbbo     | mixed                 |                              2 |                         1 |
| 2026-04-27 00:00:00 | C        | Bear Put Debit Spread  | 2026-04-28 00:00:00 | 2026-05-15 00:00:00 |        1.3365 |           5   |                    0.633622 |                      22.3521  |            157.185 | range     | next_session_first_regular_nbbo     | supportive            |                              1 |                         1 |
| 2026-05-11 00:00:00 | NFLX     | Bear Put Debit Spread  | 2026-05-12 00:00:00 | 2026-06-18 00:00:00 |        1.562  |           5   |                    0.647789 |                      30.5494  |            161.63  | range     | next_session_first_regular_nbbo     | supportive            |                              7 |                         1 |
| 2026-05-14 00:00:00 | WFC      | Bull Call Debit Spread | 2026-05-15 00:00:00 | 2026-06-18 00:00:00 |        1.67   |           5   |                    0.612106 |                      20.7433  |            316.3   | uptrend   | next_session_first_regular_nbbo     | contrary              |                              0 |                         2 |
| 2026-05-18 00:00:00 | FCX      | Bear Put Debit Spread  | 2026-05-19 00:00:00 | 2026-06-18 00:00:00 |        2.211  |           5   |                    0.640188 |                      39.881   |           -243.21  | range     | next_session_first_regular_nbbo     | contrary              |                              5 |                         1 |
| 2026-05-20 00:00:00 | WULF     | Bull Call Debit Spread | 2026-05-21 00:00:00 | 2026-06-18 00:00:00 |        0.891  |           2.5 |                    0.622807 |                      12.9742  |            151.99  | uptrend   | next_session_first_regular_nbbo     | supportive            |                              2 |                         1 |
| 2026-05-27 00:00:00 | CSCO     | Bear Put Debit Spread  | 2026-05-28 00:00:00 | 2026-06-26 00:00:00 |        1.815  |           5   |                    0.736767 |                      67.7964  |            300.35  | range     | next_session_first_regular_nbbo     | supportive            |                              0 |                         1 |
| 2026-06-29 00:00:00 | BABA     | Bull Call Debit Spread | 2026-06-30 00:00:00 | 2026-07-17 00:00:00 |        1.804  |           5   |                    0.594514 |                      16.0607  |            213.31  | uptrend   | next_session_first_regular_nbbo     | supportive            |                              0 |                         2 |

## Interpretation

This is a research ledger, not an order list. It uses fixed next-session entries and outcome-blind expanding-window predictions. Production V4.24 remains unchanged.
