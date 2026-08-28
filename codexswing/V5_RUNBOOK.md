# CodexSwing v0.5 research runbook

## Current state

`IMPLEMENTED_NOT_EXECUTED` / `NO_REPLAY_RUN`.

The v0.5 lane is code-complete enough to plan cache coverage, replay exact
multi-session option paths, apply event exclusions and family-wise statistical
gates, and record prospective hypothetical outcomes. It has not been run
against historical ORATS data and has produced no trade, POP, expectancy, or
profitability evidence.

The active v0.4 source path is protected by
`research_specs/CODEXSWING_V4_FROZEN.json`. v0.5 is a separate Python package;
it does not import an ORATS client, Schwab client, credential loader, or broker
order method.

## Quota guard

The frozen v0.5 spec records the user-reported 12,000 remaining ORATS requests
as planning metadata. It authorizes zero requests and reserves all 12,000.
Missing-cache output coalesces ticker-history endpoints using ORATS's documented
batch limits while retaining one request per ticker/date for historical option
chains. It is still a planning upper bound and must never increase the
authorized amount silently.

The only safe status command during the build phase is local and read-only:

```bash
PYTHONPATH=src python3 -m codexswing.v5 describe
```

`plan-cache` can read either a local inventory JSON or scan the immutable local
store. It distinguishes present, missing, and explicitly known-unavailable
slices; it cannot fill any of them.

## Frozen research family

- Six structures: long call, long put, bull-call debit, bear-put debit,
  bull-put credit, and bear-call credit.
- Four separate holding periods: 3, 5, 10, and 20 sessions.
- Three separate exits: fixed hold, +0.25R/-0.35R, and +0.50R/-0.35R.
- Total: 72 counted hypotheses. No result may be dropped from the family after
  its outcome is known.
- Regime cohort: up to 250 prior same-ticker, same-direction observations using
  only point-in-time `hist/cores` volatility, term-structure, and trend fields.
- Exact option path: frozen v0.4 entry selection/fill and natural/bid
  liquidation logic, evaluated on every cached session until a predeclared exit
  or horizon.
- Events: earnings always exclude entry-through-exit exposure; ex-dividend
  exposure excludes structures with a short call.
- Promotion: positive train, validation, and untouched holdout expectancy; at
  least 15 independent holdout clusters; and clustered one-sided p-value that
  survives Holm-Bonferroni correction across the complete declared family.

Public design references:

- [ORATS historical-data API](https://orats.com/docs/historical-data-api)
- [ORATS field definitions](https://orats.com/docs/definitions)
- [ORATS backtesting methodology](https://orats.com/university/backtesting-methodology)
- [ORATS custom backtesting](https://orats.com/university/custom-backtesting)

## Prospective shadow evidence

`codexswing.v5.ledger` supplies idempotent hooks for `SIGNAL`, `QUOTE`,
`TRIGGER`, `EXIT`, and `OUTCOME`. Records are append-only JSONL, mode 0600,
hash-chained, fsynced, and reject credential-like keys. The ledger is
hypothetical research only and has no broker interaction.

When a future v0.5 research run is explicitly authorized, its orchestration
must call these hooks automatically at each lifecycle transition. Until then,
no real ledger is created; only temporary unit-test ledgers are allowed.

## Activation gate for a later turn

Before any ORATS request is authorized:

1. Build a local inventory from already immutable cached records.
2. Generate replay paths without fetching anything.
3. Inspect the conservative missing-slice upper bound and coalesce requests by
   actual endpoint semantics.
4. Set an explicit request cap and reserve floor approved by the user.
5. Run a small canary slice first, then reconcile the measured usage delta.
6. Only after the canary reconciles may a larger replay be considered.

Implementation is not validation. Even a completed replay remains research
until leakage checks, full-family correction, independent holdout economics,
and prospective live-vs-replay parity all pass.
