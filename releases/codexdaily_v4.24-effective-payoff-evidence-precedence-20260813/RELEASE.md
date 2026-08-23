# Codex Daily V4.24 Effective Payoff Evidence Release

Version: v4.24-effective-payoff-evidence-precedence-20260813
Released: 2026-08-13

## What is fixed

- Recomputes 5% and 10% fill stress from base P/L and entry credit.
- Authorizes only the exact supportive or matched-unconfirmed OI execution population.
- Separates the failed 83-row broad reference from the passing 53-trade executable population.
- Validates Bull Put and Bear Call family subgroups before live authority.
- Uses the minimum of pooled and family Wilson lower bounds for displayed confidence.
- Uses family-specific PF and stressed return for ticket expectancy.
- Replaces contradictory Execute plus legacy VETO display with an effective execution-policy status while preserving the legacy sparse-route result as a non-authoritative audit warning.
- Keeps every live-authorized row at one contract and Medium confidence. High remains unavailable.

## Corrected evidence

Execution population:
- 53 trades, 49 wins, 4 losses
- 92.45% raw win rate
- 82.14% pooled 95% Wilson lower bound
- 10% worse-fill PF 3.4177
- stressed P/L $2,769.07
- selection-sequence drawdown -$368.38
- holdout 15 trades, 13 wins, PF 1.828
- 7 of 8 positive signal months

Family evidence:
- Bear Call: 32 trades, 29 wins, PF 2.277; family Wilson floor approximately 76%; holdout 12 trades, PF 1.454; status PASS
- Bull Put: 21 trades, 20 wins, PF 12.334; family Wilson floor approximately 77%; holdout 3 trades, all wins; status PROBATIONARY

## Deterministic end-to-end validation

Date: 2026-08-11
Market state: frozen V4.21 Schwab snapshot
Universe evaluated: 8,913 scored rows
Tickets: 142
Execute: 5
Scout: 94
Assertions: explicit sell/buy legs, family validation, supportive/matched OI, execution authority, effective payoff status, family-aware confidence and expectancy all passed.

Codex Daily tests: 307 passed.
Maintained tests run: 1,505 passed before the two V4 fixtures were updated; those V4 failures then passed in the 307-test Codex suite. Two remaining full-suite failures belong to separate pattern pipelines and were not merged into Codex Daily.

## Capacity limitation

One-contract stressed realized monthly average is approximately $346. A mechanical $10,000 monthly average would require about 29 contracts per trade, about $115,013 peak defined risk, approximately -$20,602 stressed realized drawdown, a losing month, and concentration breaches. Reliable $10,000/month performance is not demonstrated.

## Rollback

V4.23 remains frozen at releases/codexdaily_v4.23-corrected-fill-stress-execution-population-20260813.
