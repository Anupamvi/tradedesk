# ORATS Field Profiles

## Status

All analytical profiles begin as `DOCUMENTED_NOT_PROBED`. The initial offline
release does not claim that the current account is entitled to any field. Exact
provider field names, formats, timestamps, coverage, and response sizes must be
frozen from a separately authorized entitlement-discovery artifact before a
profile becomes runnable.

Every profile includes the provider identity fields `ticker`, `tradeDate`, and
`updatedAt` where the endpoint supports them. If one is absent, discovery must
record the documented replacement; code may not guess one.

## Versioned profiles

| Profile | Endpoint | Required analytical concepts | Consumer |
|---|---|---|---|
| `CORE_SCREEN_V1` | `/datav2/cores` | implied volatility, forecast volatility, historical volatility, earnings effect, provider confidence | Shared local shortlist |
| `SUMMARY_ENRICH_V1` | `/datav2/summaries` | term structure, skew summary, liquidity/coverage context | Strategy-family feature builder |
| `MONEY_IMPLIED_V1` | `/datav2/monies/implied` | expiration/moneyness grid and implied surface | Volatility, term, skew families |
| `MONEY_FORECAST_V1` | `/datav2/monies/forecast` | expiration/moneyness grid and forecast surface | Implied-versus-forecast comparisons |
| `EXACT_OPTION_V1` | `/datav2/strikes/options` | exact contract identity, analytical Greeks/volatility, provider timestamps | Finalist evidence only |
| `HIST_CORE_SIGNAL_V3` | `/datav2/hist/cores` | full-history IV, forecast, HV, term, slope, confidence, price fields | Signals for the externally frozen sampled cohorts |
| `HIST_ROTATING_COHORT_CHAIN_V2` | `/datav2/hist/strikes` | bounded 20-180 DTE and complete 0-1 call-delta contemporaneous chain fields | Every frozen structure, T+1 entry, and complete exit path for one active cohort/session |
| `HIST_SPLITS_V2` | `/datav2/hist/splits` | split date and divisor for sampled names | Exact-contract corporate-action review |

`HIST_SIGNAL_ENTRY_CHAIN_V2` and `HIST_EXACT_STRIKE_SERIES_V1` are preserved
only as disabled V1 identifiers. Request-per-signal and request-per-strike
history cannot enter a V2 plan.

The phrase “required analytical concepts” is intentionally not a guessed API
field list. Discovery produces a mapping from each concept to a verified field,
type, nullable policy, unit, update cadence, and entitlement result. That
mapping increments the profile version and is hashed into every plan and
snapshot.

## Validation rules

- Reject unknown fields and unversioned profile changes.
- Reject mixed `tradeDate` values unless the manifest explicitly models them.
- Reject missing/naive timestamps, duplicate entities, unbounded rows, schema
  drift, and a response larger than the planned byte ceiling.
- Preserve raw option delta only as a separately labeled market heuristic.
  Delta is never stored or displayed as calibrated POP.
- A strategy declares its concepts before holdout access. The planner unions
  requirements across families so the same entity/profile/vintage is fetched
  once.
