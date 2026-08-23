# Codex Daily V4.25 Shadow Evidence V1.1

Release date: 2026-08-16
Supersedes research package: `codexdaily_v4.25-shadow-evidence-v1-20260816`

## Authority

- Production base: `v4.25-actionable-ticket-contract-dedupe-20260814`.
- Production `Execute`, `Work Limit`, sizing, and rejection authority is unchanged.
- All new outputs are shadow-only, set `execution_authorized=false`, and cannot place orders.
- Research failures write an explicit error artifact and cannot fail the production report.

## Debit shadow

- Frozen maturity-safe training rows: 722, with outcomes observed through 2026-08-05.
- Live mapping locks actual Schwab buy/sell contracts and the conservative natural debit.
- Entry must be a following-session regular quote.
- Exit policy: 100% profit target, no hard stop, expiry settlement fallback.
- Historical selected sample: 11 trades, 9 wins, 2 losses.
- Base: PF 6.064, P/L +$1,565.40.
- 10% adverse-entry stress: PF 5.122, P/L +$1,401.36.
- Development: n=8, stressed PF 3.164, P/L +$735.71, max drawdown -$243.21.
- Holdout: n=3, 3 wins, P/L +$665.65; PF is undefined because there was no holdout loss.
- It remains blocked from production by total, development, holdout, Wilson, and per-strategy sample gates.

## Range/GEX shadow

- Historical GEX rows audited: 4,195.
- Strict point-in-time rows: 84 across 3 dates.
- Replay rows joined to strict GEX: 48.
- Qualified historical verticals: 0.
- Qualified historical joint-payoff condors: 0.
- Condor outcomes use one combined four-leg mark and payoff; independent vertical outcomes are prohibited.
- Reconstructed after-date GEX cannot enter fitting, selection, or authorization.
- Missing GEX emits a prequalified collection universe capped at 25 tickers; the logged-in UW collector runs only for those names, then only shadow artifacts refresh.

## Integration result

For the 2026-08-13 V4.25 scored universe:

- Debit live-guard rows: 429.
- Debit research-qualified rows: 70.
- Selected shadow: TEAM buy 160C / sell 165C, 2026-08-28, $3.10 natural debit.
- Model probability: 84.85%.
- Payoff-correct stressed EV: +$83.24 per one-lot.
- Point-in-time GEX was missing.
- GEX collection universe: MSTR and CRCL.
- Execution authorized: false for both research books.

## Validation

- 324 tests passed.
- 0 tests failed.
- One environment-only LibreSSL warning remains.
- All packaged files pass SHA-256 verification.

## Rollback

Rollback affects shadow instrumentation only. Production execution logic did not change.

1. Restore `codexuw/goal_shadow.py` from the preceding V4.25 source snapshot or remove the `write_daily_shadow_outputs` block and restore its direct return.
2. Leave or archive the two new modules; without the hook they are inert.
3. Restore the prior `codexdaily` skill to remove the optional GEX shadow follow-up.
4. Preserve central ledgers as audit history; they never carried execution authority.
