# Cultra Clean-Room Charter

## Identity and boundary

Cultra is a new options research system rooted at
`/Users/anuppamvi/tradedesk/cultra`. Its implementation must be derived from
the Cultra specification and primary provider contracts, not from another
local pipeline.

The following are prohibited:

- copying, adapting, importing, invoking, or studying another pipeline's code;
- reusing another pipeline's prompts, configuration, schemas, fixtures, tests,
  strategy rules, caches, reports, outputs, or Git history;
- reading data from, writing into, or linking to another pipeline directory;
- publishing account-aware sizing or creating, replacing, cancelling, or
  submitting a broker order.

The only shared assets permitted are credentials, and only through these
paths:

- Cultra-owned `.env`, mode `0600`, containing only `ORATS_TOKEN` after an
  explicit one-time bootstrap from a user-supplied source;
- `/Users/anuppamvi/tradedesk/tokens/schwab_token.json`, used only by a
  read-only quotes/chains/history provider.

Credentials never enter a request plan, worker environment, cache key, log,
exception, artifact, report, command line, or test fixture.

## Evidence contract

Cultra begins at `UNPROVEN`. Each frozen strategy family advances separately
through research, validation, and a one-time untouched holdout. `HOLDOUT_PASS`
may enable a manual research handoff. Prospective shadow is a parallel
monitoring state that may revoke an enabled family; it is not a waiting period.

No aggregate confidence score hides those family states. A manual-review
ticket requires calibrated POP, positive point and conservative net expected
value, exact legs, current Schwab executable quotes, finite maximum loss, and
a family that has passed the complete untouched historical holdout gate.

Cultra never claims guaranteed profit. It reports evidence, intervals,
assumptions, and failure states. Missing mandatory evidence is a rejection or a
named unresolved state, not a value inferred from today's chain.

## Operating contract

- Normal orchestration is zero-request and fail-closed by default.
- Only the gateway may hold the ORATS token or perform an ORATS transport call.
- A durable permit is charged before every physical ORATS attempt. Uncertain
  attempts remain charged.
- Request 100 is unreachable. Smaller run-type caps may apply.
- A zero-candidate or zero-ticket result never triggers extra calls to find a
  trade.
- All qualifying tickets are shown; there is no arbitrary top-N output cap.
- Tickets describe one normalized structure unit and always state `Quantity:
  USER DETERMINED`.
- Manual tickets are research outputs. Cultra has no order-submission surface.

## Initial release boundary

The initial release implements contracts, evidence calculations, request
planning, the gateway/ledger/cache boundary, an offline `UNPROVEN` run, daily
board generation, and adversarial tests. It does not authorize entitlement
discovery, historical backfill, holdout consumption, prospective shadow
promotion, or a live ticket. Each of those is a separately auditable rollout
gate.
