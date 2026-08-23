# Codex Daily V4.23 Corrected Fill-Stress / Execution-Population Release

Version: v4.23-corrected-fill-stress-execution-population-20260813
Released: 2026-08-13

## Production authority

Medium directional-credit entries only, with directional flow and supportive or matched-unconfirmed OI.

## Corrected executable evidence

- 53 trades: 49 wins and 4 losses
- Win rate: 92.45%
- Wilson lower bound: 82.14%
- Profit factor at 10% worse fills: 3.4177
- Stressed P/L: $2,769.07
- Selection-sequence drawdown: approximately -$368
- Holdout: 15 trades, 13 wins, PF 1.828, drawdown -$342.05
- Positive signal months: 7 of 8

## Integrity correction

V4.21 packaged stress columns duplicated base P/L. V4.23 always recomputes 5% and 10% fill stress from entry credit. The broader 83-row reference population includes contrary-OI trades, fails its holdout gate, and remains descriptive and non-executable. Production status is based on the exact execution population.

## Capacity finding

At one contract, stressed realized monthly average is approximately $346. Reaching a $10,000 average mechanically requires approximately 29 contracts per trade, about $115,013 peak defined risk, approximately -$20,602 realized stressed drawdown, one losing month, and concentration breaches. A reliable $10,000/month result is not demonstrated.

## Other strategy lanes

Debit and range/GEX income books remain shadow-only. They cannot authorize live entries.

## Rollback

The frozen V4.21 checksum bundle remains available. The version registry also retains V4.22 as the immediate prior code release.
