# Cultra Agent Rules

These instructions apply to every file and action under this project.

## Clean room

- Work only inside `/Users/anuppamvi/tradedesk/cultra`.
- Do not read, copy, adapt, import, invoke, diff, or search another trading
  pipeline's code, prompts, schemas, configuration, tests, data, caches,
  artifacts, reports, or Git history.
- Do not add another pipeline directory to `PYTHONPATH`, create a cross-pipeline
  symlink, or import a sibling package.
- Implement from the Cultra specification, Cultra-owned tests, and primary
  provider documentation only.
- Python 3.9 standard library is the dependency boundary.

## Credentials and network

- The only allowed credential paths are the project `.env` (only
  `ORATS_TOKEN`, mode `0600`) and
  `/Users/anuppamvi/tradedesk/tokens/schwab_token.json`.
- Never log, print, persist, serialize, cache, hash as an identity, place on a
  command line, or pass to a worker any credential value.
- Local tests/replays/debugging are zero-request by default. Use fake
  transports and temporary ledgers/caches.
- Only `cultra/gateway.py` may perform ORATS transport or import network-client
  modules. Every physical attempt requires an already-charged durable permit.
- Do not enable entitlement discovery, historical backfill, or live market-data
  calls without the corresponding explicit authorization and frozen cap.

## Profit evidence and trading boundary

- Keep strategies, historical observations, validation results, shadow
  observations, candidates, and manual tickets distinct.
- Never label delta as POP. Never publish uncalibrated POP.
- A ticket requires exact OCC legs, fresh Schwab quotes, complete modeled costs,
  positive point and conservative net EV, finite maximum loss, and the complete
  family promotion gate.
- Never infer missing historical legs or reuse an untouched holdout after a
  failure.
- No portfolio/NAV/position/sector/diversification/buying-power gate and no
  arbitrary output top-N cap may suppress an otherwise eligible ticket.
- Quantity is always `USER DETERMINED`.
- Do not add account, position, order-create, order-replace, order-cancel, or
  order-submit code. Cultra is manual-review only.

## Verification and artifacts

- Add offline tests for every changed invariant and run the full suite.
- Fail closed on missing data, stale quotes, schema/vintage mismatch, corrupt
  cache/ledger state, attempt uncertainty, and incomplete costs/evidence.
- Preserve all category outcomes: ticket, watchlist, rejected,
  data-unavailable, and `NOT_FULLY_EVALUATED_BUDGET`.
- Every published run must end with a reconciled, checksummed manifest. Never
  overwrite an existing run directory or silently repair an immutable artifact.

