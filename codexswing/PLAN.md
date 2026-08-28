# codexswing plan

## Objective

Build a daily, auditable pipeline that identifies stock and defined-risk option
swing trades with testable edge. The user makes and submits every final trade.

## Implemented v0.4

- Replaced the ten-symbol screen with a 5,975-underlying ORATS `cores` scan and
  explicit price, option-volume, open-interest, asset-type, and confidence
  funnel. Finalists include both long and short directions and cap ETF share.
- Made ORATS adjusted dailies canonical stock history and handled two historical
  data issues explicitly: split-rounding normalization and sub-2% invalid OHLC
  envelope normalization.
- Corrected ORATS forecast semantics: `orFcst20d` drives the underlying-return
  distribution; `orIvFcst20d` drives option-IV evolution.
- Added six exact current strategies: long call, long put, bull call debit, bear
  put debit, bull put credit, and bear call credit.
- Replaced natural-quote EV with a proposed limit using ORATS's published
  two-leg 66%-of-package-width convention. Risk and EV use that same limit.
- Added exact historical EOD replay using ORATS `hist/strikes`, natural-side
  exits, $0.65 per contract/leg commissions, trigger/gap rules, overlap-aware
  effective sample size, Wilson POP bounds, profit factor, and clustered
  bootstrap lower expectancy.
- Added chronological 50% train / 20% validation / 30% untouched holdout. No
  current trade can pass from a point POP alone.
- Added a contiguous promotion state machine from discovery through
  `MANUAL_READY`.
- Added current Schwab portfolio gates: buying power, 1%/$2,000 max-risk cap,
  10% ticker concentration, existing option positions, and working orders.
- Added exact source-seeding tables and separate empirical, modeled, and
  historical POPs in HTML/JSON.
- Added immutable content-addressed storage, code/source manifests, secret
  exclusion, cache-resumable historical slices, and a GET-only broker surface.
- Made ORATS historical 404/empty slices explicit immutable rejection records;
  one absent ticker can no longer abort or hide valid same-date chains.
- Derived Schwab chain sessions from embedded quote times rather than the
  after-midnight ingestion clock, with freshness checked on both selected legs.
- Added time-windowed Google News RSS fallback when GDELT is unavailable. Both
  providers remain displayed shadow evidence with no numeric vote.

## Prior v0.3 evidence result, 2026-08-27

The implementation works; no tested structure cleared every economic gate.
This is evidence, not a reason to loosen thresholds:

- SPY bull call: current modeled EV +$139 and modeled POP 68.3%, but exact-chain
  holdout mean only about +$1, bootstrap lower mean about -$91, and profit factor
  about 1.01. Reject.
- QQQ bull call: current modeled EV +$167 and modeled POP 62.7%; holdout POP
  62.5% and mean +$34, but clustered-bootstrap lower mean remained negative.
  Reject.
- AAPL bear put: current modeled EV +$53 and modeled POP 64.6%; holdout POP 25%,
  mean about -$92, and effective sample 6. Reject.
- SLV bull call: holdout mean and bootstrap lower mean were positive, but only 9
  closed / 6 effective trades exist versus required 20 / 8. Reject as
  insufficient, not as a negative strategy.
- ORCL bull put credit: holdout mean about -$19, profit factor about 0.59, and
  only 14 closed / 7 effective trades. Reject.
- MSFT bull call: holdout mean about -$23, profit factor about 0.76, and 15
  closed trades. Reject.

All six passed the exact-current-contract gate after Schwab timestamp repair;
none passed the fixed ORATS historical gate. The replay covered 600 signal
samples and 12 ticker/structure groups. Ninety-one unavailable ticker/date
slices were preserved as rejections, never imputed as trades.

These figures are tied to the dated immutable artifact and may change on a new
session. They are not orders.

## v0.4 repair now implemented

- Expanded the label-free current-regime cohort from 100 to 250 analog dates.
- Added exact single-leg ORATS replay using 75%-through-spread entries, exact-bid
  exits, and $1.30 round-trip commissions.
- Added a distinct `TACTICAL_READY` evidence tier. It requires at least 30
  holdout closes / 15 independent outcomes; positive train, validation, and
  holdout expectancy; validation and holdout profit factor of at least 1.20;
  and a bootstrap lower mean no worse than 5% of the current defined risk.
- Capped tactical trades at one contract and the smaller of $500 or 0.05% of
  current Schwab liquidation value. Tactical never masquerades as full evidence.
- Added exact trigger, maximum gap, limit, invalidation, and exit-session fields.
- Records all tested ticker-strategy groups and states that confidence is not yet
  multiple-testing adjusted.

## Isolated v0.5 implementation (not executed)

v0.5 now exists as a separate, disabled-by-default research lane. It does not
change the active v0.4 source path and it authorizes zero API requests.

- Added cache-only `hist/cores` regime vectors and label-free nearest analogs.
- Predeclared 3/5/10/20-session paths and three fixed exit policies across all
  six structures: 72 counted hypotheses, not a post-hoc horizon optimizer.
- Extended frozen v0.4 exact-chain pricing across every session in a path, with
  fail-closed missing-leg handling.
- Added earnings and short-call ex-dividend/assignment exclusions.
- Added cluster-bootstrap one-sided inference and Holm-Bonferroni family-wise
  correction. The complete declared family is mandatory at evaluation time.
- Added an idempotent append-only hash-chain ledger for hypothetical signal,
  quote, trigger, exit, and outcome events.
- Added a cache inventory and missing-slice planner with a hard zero-request
  guard. The user-reported 12,000 remaining requests are fully reserved.

Status: `IMPLEMENTED_NOT_EXECUTED` / `NO_REPLAY_RUN`. No v0.5 confidence or
profitability claim exists yet. See `V5_RUNBOOK.md` for the later quota-safe
activation sequence.

## Remaining evidence work

1. Inventory existing immutable ORATS cache and calculate a request-coalesced
   replay budget without making a network call.
2. After explicit authorization, run a usage-reconciled canary and only then a
   leakage-safe v0.5 replay.
3. Add a broad stock-lane walk-forward/holdout artifact so stock candidates can
   advance independently of options.
4. Run time-aligned ablations for public news, internet attention, and
   geopolitical features. Promote only if corrected net holdout value improves
   across regimes; otherwise leave them descriptive.
5. Wire the shadow-ledger hooks into an explicitly authorized scheduled local
   research run and measure live-vs-replay parity prospectively.

## Non-negotiable controls

- Never call a modeled POP a calibrated POP.
- Never select a strategy using untouched holdout results.
- Never count overlapping five-session rows as independent trades.
- Never assume midpoint fills.
- Never promote when the exact current contract or portfolio snapshot is stale.
- Never create or transmit a broker order.
- Never let v0.5 fetch missing cache records unless a later execution receives
  a specific request cap and reserve floor.
