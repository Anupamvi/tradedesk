# Debit Walk-Forward Shadow V2

Release: `debit-walkforward-shadow-v2-20260814`
Status: `RESEARCH_ONLY`
Execution authority: `false`
Production authority: unchanged at `codexdaily_v4.24-effective-payoff-evidence-precedence-20260813`

## What changed

- Corrected predicted EV to use the actual profit target, 10% adverse entry-fill stress, and worst-case debit loss.
- Added a reproducible debit-only revaluation of frozen next-session entries.
- Enforced signal-day to next-session timing integrity.
- Added 0%, 5%, 10%, and 15% entry-fill stress outcomes.
- Added development/holdout, strategy, regime, quote-source, calibration, overlap, concentration, Wilson-bound, PF, and drawdown diagnostics.
- Kept all production V4.24 execution paths unchanged.

## Frozen research policy

- Profit target: 100% of debit.
- Hard stop: none; risk remains defined by net debit.
- Entry fill stress: 10% worse than replayed entry debit.
- Selection threshold: 55% plus payoff-correct predicted EV greater than zero.
- Selection cap: one debit candidate per signal day.
- Development/holdout cutoff: 2026-05-01.

## Result

- Development: 5 trades, 4 wins, PF 6.176, stressed P/L +$500.99, max drawdown -$96.80.
- Holdout: 6 trades, 5 wins, PF 4.702, stressed P/L +$900.37, max drawdown -$243.21.
- Combined: 11 trades, 9 wins.
- Peak simultaneous positions: 5.
- Peak defined risk: $896.39.
- Peak-risk sector share: 31.43%.
- Peak-risk ticker share: 27.13%.

## Promotion blockers

- Development sample below 20.
- Holdout sample below 15.
- Total selected sample below 50.
- Development and holdout Wilson lower bounds below 55%.
- Bear Put and Bull Call holdout samples below 10 each.

## Diversification against frozen V4.24 credit execution book

- Frozen credit execution population: 53 trades, stressed PF 3.418, P/L +$2,769.07.
- Combined credit plus shadow debit: 64 trades, stressed PF 3.808, P/L +$4,170.43.
- Combined sequence drawdown: -$368.38, unchanged from the credit book alone.
- Same signal-day overlap: 2 of 11 debit trades.
- Monthly P/L correlation: -0.55.
- Combined peak defined risk at one contract: $4,690.52 across 14 active positions.
- Peak-risk sector share: 71.18%; this remains a hard scaling blocker.
- Mechanical $10,000/month scale is about 20 contracts, implying roughly $93,810 peak defined risk and $7,368 sequence drawdown before prospective-error allowance. This scale is not authorized.

## Rollback

- Production rollback remains `releases/codexdaily_v4.24-effective-payoff-evidence-precedence-20260813`.
- Prior debit research output remains `out/codexdaily_v421_debit_walkforward_shadow_v1_2026-08-13`.
- Removing or ignoring this research release fully restores prior behavior because it has no production execution authority.
