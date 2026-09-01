# ORATS Request-Efficiency Plan

All counts below are physical attempts because redirects, automatic retries,
polling, split recovery, and implicit provider fallbacks are disabled.

## Daily production (separate from history)

Local Schwab-backed screening precedes ORATS. No eligible name is silently
truncated; work that cannot fit is published as
`NOT_FULLY_EVALUATED_BUDGET`.

| Profile | Maximum entities | Frozen batch size |
|---|---:|---:|
| Core | 600 | 10 |
| Summary | 120 | 10 |
| Monies implied | 40 | 10 |
| Monies forecast | 40 | 10 |
| Exact options | 250 deduplicated contracts | 100 |

The normal EOD target is 25 and the logical cap is 60. The independent maxima
would total 83, so that combined shape is inadmissible before the credential is
loaded. The saved 254-name local screen needs 26 Core calls; it exceeds the
target by one but stays within the cap. Same-vintage and normal morning runs use
zero ORATS calls.

One Cultra account-level SQLite ledger acquires a single-active-run lease and
irreversibly charges a permit before transport. A crash burns the permit.
Request 100 is structurally impossible. A zero-trade result cannot open a new
request family merely to find a trade.

## Historical campaign

The rejected full-universe grid would require `ceil(U/10) * 450` chain calls:
11,700 calls for 254 names. Request-per-signal and request-per-exact-strike
history are also disabled because they create a data-dependent N+1 campaign.

The canonical V2 design freezes four disjoint ten-name cohorts from an
independent point-in-time broad-universe manifest. ORATS never chooses cohort
membership. Each cohort is fetched once per session in its block. Signals that
cannot complete T+1 entry plus the maximum 60-session path before rotation are
ineligible, so no prior-cohort overlap download is required.

| Component | Formula | Cold attempts |
|---|---:|---:|
| Sampled full-history Core | `ceil(40 / 2)` | 20 |
| Contemporaneous full chains | `450 * ceil(10 / 10)` | 450 |
| Sampled split history | `ceil(40 / 10)` | 4 |
| **Base campaign** | | **474** |

The two-name Core batch is an empirical safety limit, not the documented
provider maximum. The saved Cultra ledger has 1/1 success at two names and 0/3
success at ten names. The ten-name historical-chain geometry has 451/451 saved
successes; the ten-name split geometry has 1/1.

The 450 sessions form blocks of 120, 120, 120, and 90 sessions. Boundary
censoring leaves 59, 59, 59, and 29 eligible signal sessions: 2,060 ticker/date
candidates before signal filters. Shorter horizons deliberately use the same
windows so every walk-forward boundary retains the frozen 60-session embargo.
If development evidence later proves this capacity insufficient, it must fail
before the sealed holdout is opened; the same exposed holdout cannot be
repaired with added requests.

Optional continuous entry coverage at three transitions would add
`3 * (1 entry + 60 exits) = 183` chain calls, producing a 657-call extended campaign. It is not
hidden inside the base estimate and cannot start without a new frozen plan.

## Slices and cache

The 474 requests are partitioned as `90 + 90 + 90 + 90 + 90 + 24`: six separate
backfill run IDs, each with a hard cap of 90 and zero retries. The exact frozen
graph permits at most 474 initial physical attempts. The arithmetic 540 from
six generic slice caps includes 66 unused permits; they are not planned or
authorized. Each exact request ID is frozen before slice 1; every later slice
requires separate user authorization and ledger/cache reconciliation.

Backfill execution does not accept a mutable ticker list, date list, or request
count. It accepts only an immutable campaign-freeze receipt plus one slice
index. Completion requires exact reconciliation of all 474 planned IDs, every
slice manifest, every cached raw snapshot, and charged attempts before any row
can enter the V2 normalized database.

Completed validated partitions are immutable. A same-request cache hit costs
zero. Crashes cannot publish uncertain data, and an indeterminate permit is
never reused. Incomplete executions are preserved under append-only execution
checkpoints and never become a completed slice manifest. If a failed request is
later recovered by a separately frozen/authorized run, the exact aborted slice
may be reactivated: recovered requests are cache hits, while any charged ID
still lacking a validated snapshot remains structurally unable to send again.
Missing entities or malformed rows invalidate the response instead of
triggering per-symbol recovery.

Recovery is never hidden in 474. If a no-retry request fails, a separately
frozen recovery may send only the exact missing grouped fingerprints after
authorization. Its count is `R`, so completed-campaign spend is `474 + R`.

## Preconditions not included in 474

- Source-bound prerequisite receipt: the campaign accepts only a receipt
  reproduced from preserved independent raw universe, XNYS calendar, and
  event/adjustment bundles. Normalized manifests with self-asserted hashes and
  ORATS-derived prerequisite sources fail closed. Preparation costs zero ORATS
  requests.
- Entitlement discovery: separate, up to 12 expected and 15 absolute attempts,
  only if explicitly authorized. Existing successful endpoint evidence may
  make a new discovery run unnecessary; this is decided offline before spend.
- Earnings, dividends, delistings, and contract adjustments: a frozen
  independent point-in-time source is required before slice 1. The base ORATS
  estimate contains zero hidden event calls.
- Daily production: separate target/cap and never combined with a history run.
- Any future evidence extension: separately frozen and authorized; failed or
  exposed holdout evidence is terminal.

The estimate command is pure and tokenless:

```bash
python3 -m cultra estimate-history --eligible-symbols 254
```
