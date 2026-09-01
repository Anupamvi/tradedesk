# ORATS Endpoint Ownership

## One owner per data role

| Data role | Owner | Normal daily policy |
|---|---|---|
| Underlying price, trend, and market liquidity | Schwab read-only boundary | Quotes/history only |
| Exact option bid/ask, volume, open interest, contract availability | Schwab read-only boundary | Chains/quotes only |
| Core volatility and forecast analytics | ORATS `/datav2/cores` | Bulk, planned, cached |
| Shortlist summary analytics | ORATS `/datav2/summaries` | Bulk, conditional on local shortlist |
| Implied and forecast money surfaces | ORATS `/datav2/monies/implied` and `/datav2/monies/forecast` | Bulk finalists only |
| Exact-contract analytical enrichment | ORATS `/datav2/strikes/options` | Deduplicated OCC symbols only |
| Historical signal features | ORATS `/datav2/hist/cores` | Full history for the 40 externally selected cohort names, two/request; saved ten-name attempts failed 3/3 |
| Historical entry and exit paths | ORATS `/datav2/hist/strikes` | One complete ten-name active-cohort chain per frozen session |
| Historical splits | ORATS `/datav2/hist/splits` | Four sampled-name batches; validated empty history is complete coverage |
| Point-in-time membership/liquidity and earnings/dividends/delistings | Independent frozen source manifests | Required before slice 1; zero hidden ORATS calls |
| Account state, positions, buying power, order management | Nobody | Prohibited |

Schwab is executable market truth for price and liquidity. ORATS is delayed/EOD
research evidence. Neither provider's value may silently substitute for the
other's role.

## Endpoint policy

Normal EOD runs may plan only the four bulk roles above. Filtered `/strikes` and
IV-rank enrichment are disabled unless a versioned requirement, explicit
authorization, bounded field set, bounded DTE/delta window, and row/byte budget
exist before execution. Scanner and backtest endpoints are disabled in normal
runs.

Historical endpoints belong only to six separately checkpointed V2 slices,
each capped at 90 attempts. Request-per-signal and request-per-exact-strike
history are disabled. Backtest job creation/polling belongs only to a
separately authorized validation run. Polling and redirects cannot be smuggled
into a daily endpoint request.

## Transport ownership

The token-holding local gateway is the sole ORATS network boundary. Workers are
tokenless and may submit only an immutable `logical_request_id` already present
in the frozen `RequestPlan`. The gateway validates endpoint, method, entities,
field profile, expected vintage, response size, ledger permit, and cache state
before transport. Automatic retries and redirects are disabled.

No production Schwab HTTP adapter is activated by the initial offline release.
The `SchwabMarketDataBoundary` exposes only quotes, chains, and price history
through an injected read-only provider.
