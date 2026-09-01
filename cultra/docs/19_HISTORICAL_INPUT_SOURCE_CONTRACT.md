# Historical Input Source Contract

Cultra prepares historical inputs with zero network requests. It does not
accept a normalized manifest whose provenance hash is merely syntactically
valid. Every normalized input and cohort must reproduce from three preserved,
Cultra-owned raw JSON source bundles. Every bundle must also bind at least one
distinct preserved provider artifact; the weaker V1 wrapper-only schemas are
rejected.

Each V2 bundle has a sorted, non-empty `source_artifacts` list. Every record
contains only `path`, `role`, `source_uri`, `media_type`, `size_bytes`, and
`sha256`. Paths are project-relative and cannot be symlinks. Cultra rereads the
provider bytes, checks size/hash, scans for credential-shaped material, and
rejects ORATS roles or URIs before parsing any normalized records.

## Required raw bundles

### Point-in-time universe

Schema: `cultra.point-in-time-universe-source.v2`.

Required root fields are `schema`, `provider`, `source_uri`, `retrieved_at`,
`universe_id`, `coverage`, `point_in_time`, `survivorship_free`, and
`source_artifacts`, and `snapshots`. The source must be independent of ORATS. `coverage` is exactly
`US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_ACROSS_2_CBOE_VENUES`;
both attestations are `true`. This is a variable, point-in-time four-venue
research sampling frame. It is not falsely labeled as an all-venue census and
does not cap the separate daily production universe or ticket output.

The bundle must contain only the four exact cohort-selection dates derived
from sessions 0, 120, 240, and 360. Each snapshot contains `observed_at` and
`members`. Each member contains only:

- `ticker`
- `asset_type` (`STOCK`, `ETF`, `INELIGIBLE_OTHER_SECURITY`, or
  `UNRESOLVED_STOCK_OR_ETP`)
- `optionable` (boolean)
- `sampling_stratum`
- `liquidity_rank` (positive integer, point-in-time)
- `classification_status` (`VERIFIED_POINT_IN_TIME`,
  `VERIFIED_POINT_IN_TIME_INELIGIBLE`, or `UNRESOLVED`)
- `classification_source_roles` (sorted roles from `source_artifacts`)

At least 100 optionable names must be present on every selection date. Cultra
preserves unresolved names in that broad population instead of silently
dropping them, but unresolved names cannot be sampled. A resolved stock or ETF
must cite at least one bound point-in-time source role; an unresolved row must
cite none. This permits cohort-scoped classification without pretending that
current directories classify every historical symbol. Cultra does not accept
today's constituents projected backward, future survival, future returns, or
any outcome-dependent field. Forty disjoint names are selected
deterministically across four ten-name cohorts. The 80% stock floor is enforced
during selection rather than checked only after a bad sample is chosen. This
is a research sample, not a daily-universe or ticket-count cap.

For the 2026-08-31 public campaign, the exact liquidity rule is at least 1,000
same-day option contracts across at least two of BATS, C2, CBOE, and EDGX. It
produces variable frames of 598, 469, 621, and 606 names. Classification walks
every frame in round-robin volume-decile order and stops only after a disjoint
ten-name cohort has at least eight stocks. Current SEC/Nasdaq associations do
not pass alone: stocks require a pre-selection SEC periodic/listing filing and
OCC identity-continuity check. The exact-date Cboe row plus the complete
post-selection OCC adjustment index makes the continuity inference explicit;
the classification audit separately lists ETFs for which no pre-date SEC fund
filing was available. Missing or reused current identities use a
unique pre-selection SEC primary-document identity; this recovers GTLS and the
historical PARA issuer instead of selecting on future survival. The resulting
sample contains 33 stocks and 7 ETFs.

### Market sessions

Schema: `cultra.market-session-source.v2`.

Required root fields are `schema`, `provider`, `source_uri`, `retrieved_at`,
`exchange`, `timezone`, `complete`, `source_artifacts`, and `sessions`. The provider and URI must
be independent of ORATS; `exchange` is `XNYS`, `timezone` is
`America/New_York`, and `complete` is `true`.

The source must contain exactly 450 sorted, unique records. Each record has
only `session_date` and a timezone-aware `close_at`. Early closes must carry
their actual close timestamp. Future closes and weekend sessions fail closed.

### Historical events and adjustments

Schema: `cultra.historical-event-source.v2`.

Required root fields are `schema`, `provider`, `source_uri`, `retrieved_at`,
`coverage_start`, `coverage_end`, `covered_tickers`, `complete_event_types`,
`point_in_time_revisions`, `coverage_attestation`, and `records`. The provider
and URI must be independent of ORATS; `source_artifacts` is also required.
Coverage must span all 450 sessions.

`complete_event_types` is exactly:

- `CONTRACT_ADJUSTMENT`
- `DELISTING`
- `DIVIDEND`
- `EARNINGS`
- `SPLIT`

`point_in_time_revisions` is `true`; `coverage_attestation` is
`COMPLETE_FOR_COVERED_TICKERS_AND_EVENT_TYPES`. Every sampled symbol must be
covered, and every sampled stock must have at least one non-cancelled earnings
record inside its own cohort block. Therefore an empty or irrelevant-period
file cannot masquerade as complete event evidence.

Each record contains only `ticker`, `event_type`, `effective_date`,
`observed_at`, `available_at`, `source_event_id`, `status`, `cash_amount`,
`split_ratio`, and `adjustment_reference`. Observation and availability times
must be timezone-aware and cannot be later than source retrieval.

The 2026-08-31 public event acquisition is deliberately not this V2 source.
Its own `collection_manifest.json` declares
`TARGETED_CANDIDATE_DISCOVERY_NOT_COMPLETE` and binds the exact saved query
inventory. The offline audit may emit candidate dates from those responses, but
candidate presence cannot satisfy `complete_event_types` or the coverage
attestation above. An unavailable Nasdaq dividend response is not a zero-event
record, an SEC financial filing is not proof that every earnings date was
captured, and an OCC index row is not an adjusted-contract deliverable. The
current public audit therefore blocks every stock earnings cell, every dividend
cell, and every affected contract-adjustment cell whose exact memo bytes are
missing.

Reproduce that candidate-only event audit without network access:

```bash
python3 -m cultra audit-public-history-events \
  --classification-run <verified-public-classification-run> \
  --event-source-root /Users/anuppamvi/tradedesk/cultra/var/historical/public_events/2026-08-31 \
  --run-id <new-unique-public-event-audit-id>
python3 -m cultra verify-public-history-events \
  /Users/anuppamvi/tradedesk/cultra/out/<public-event-audit-id>
```

## Offline preparation

```bash
python3 -m cultra prepare-history-inputs \
  --input-set-id <immutable-id> \
  --universe-source <raw-universe.json> \
  --session-source <raw-sessions.json> \
  --event-source <raw-events.json> \
  --output-dir <new-directory-under-cultra/out>
```

The command writes normalized universe, calendar, events, rotating cohorts,
a source-binding receipt, and a file manifest. It then rebuilds the bundle
from the raw sources before returning success. It loads no credential and
makes no network request.

Only that receipt can freeze a request campaign:

```bash
python3 -m cultra freeze-history-campaign \
  --campaign-id <immutable-id> \
  --prerequisite-freeze <prerequisite_freeze.json> \
  --output-dir <new-campaign-directory-under-cultra/out>
```

The campaign command rehashes the raw sources, reproduces all normalized
inputs and cohorts, and then freezes the 474 request IDs in exact slices
`90+90+90+90+90+24`. It does not authorize or execute any slice.
