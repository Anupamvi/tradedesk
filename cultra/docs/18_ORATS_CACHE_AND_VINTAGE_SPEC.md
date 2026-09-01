# ORATS Cache and Vintage Specification

## Two identities

`VintageExpectation` is known before a fetch and forms the lookup key:

- endpoint and HTTP representation;
- expected publication cycle/provider trade date;
- field-profile and schema versions;
- normalized entity set.

`SnapshotManifest` is created only after response validation and records:

- immutable snapshot ID and raw content hash;
- observed provider trade dates and `updatedAt` range;
- requested, returned, and missing entities;
- field-profile/schema versions, coverage, rows, and bytes;
- parent request fingerprint and validation status.

An expectation is not evidence that a snapshot has the expected date. Only the
validated manifest can establish that.

Full-history Core uses the frozen expected date as a through-date. Its maximum
provider trade date must equal that date; an older terminal date is stale and a
later date violates the freeze. Split history may legitimately return a
validated empty set, which records complete no-event coverage for every
requested ticker. Single-date historical chains must contain their exact
expected trade date.

## Publication

Raw responses are immutable, content-addressed, mode `0600`, and Cultra-only.
Normalized snapshots publish atomically after status, schema, date, coverage,
row-count, byte-count, and hash validation. A crash may leave an unreferenced
temporary object, but cannot publish an uncertain snapshot.

Cultra never reads or writes another pipeline's cache, raw responses, or
manifests. Symlinks and paths escaping the Cultra cache root are rejected.

## Concurrency

Exact-fingerprint single-flight collapses identical work inside the gateway.
The shared account ledger permits only one active ORATS run, and plan validation
rejects overlapping entity sets for the same endpoint/profile/vintage/request
parameters. Together those boundaries prevent cross-run and within-plan
duplicate sends. Automatic recovery requests are disabled.

## Consumption

A same-vintage cache hit produces no request permit. Morning workflows reference
stored ORATS snapshot IDs while refreshing only Schwab quotes/chains. Mixed
vintages never merge silently: every candidate, rejection, POP calculation,
edge calculation, ticket, board, and run manifest carries the relevant snapshot
identity and provider trade date.

Missing, corrupt, replaced, non-private, or unverifiable cache state fails
closed. It does not automatically trigger a network refill unless a matching
request already exists in an authorized frozen plan with remaining capacity.
Required responses that omit a planned entity cannot publish. Old partial
required cache entries are rejected at lookup rather than treated as complete.
