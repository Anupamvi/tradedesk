# Codex Daily V4.25 Shadow Evidence V1

Release date: 2026-08-16

## Authority

- Production base: `v4.25-actionable-ticket-contract-dedupe-20260814`.
- Production `Execute` and `Work Limit` authority is unchanged.
- Debit and range/GEX outputs are shadow-only and cannot place or authorize orders.
- The existing goal-shadow hook fail-isolates research errors and writes an explicit error artifact.

## Debit shadow

- Frozen maturity-safe training rows: 722.
- Entry: first following-session regular quote; natural debit is locked.
- Exit policy: 100% profit target, no hard stop, expiry settlement fallback.
- Cost stress: 10% adverse debit entry.
- Historical selection: 11 rows, 9 wins, 2 losses, stressed P/L +$1,401.36.
- Development: n=8, PF=3.164, P/L +$735.71, max drawdown -$243.21.
- Holdout: n=3, 3 wins, P/L +$665.65; PF is undefined because there was no holdout loss.
- Promotion remains blocked by development, holdout, total sample, Wilson, and per-strategy sample gates.

## Range/GEX shadow

- Historical GEX rows audited: 4,195.
- Strict point-in-time rows: 84 across 3 dates.
- Replay rows joined to strict GEX: 48.
- Qualified verticals: 0.
- Qualified joint-payoff condors: 0.
- Reconstructed after-date GEX is prohibited from fitting or authorization.

## Daily integration result

For the 2026-08-13 V4.25 scored universe:

- Debit rows passing the V2 live economic guard: 429.
- Debit rows clearing research probability and payoff EV: 70.
- Selected shadow row: TEAM 160/165 bull call debit, 2026-08-28, $3.10 natural debit.
- Model win probability: 84.85%.
- Payoff-correct stressed EV: +$83.24 per one-lot.
- Execution authorized: false.
- Range/GEX status: `MISSING_POINT_IN_TIME_GEX`.

## Rollback

Rollback affects shadow instrumentation only. Production execution logic did not change.

1. Remove `codexuw/daily_shadow_books.py` and `codexuw/range_gex_income_book_v2.py`.
2. Remove the `write_daily_shadow_outputs` try/except block from `write_goal_shadow_outputs` in `codexuw/goal_shadow.py` and restore its direct three-value return.
3. Remove the two new test files and the frozen debit shadow training artifact.
4. Keep the central CSV ledgers as audit history or archive them; they never carried execution authority.
