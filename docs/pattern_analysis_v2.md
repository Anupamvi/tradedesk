# Pattern Analysis V2

Pattern Analysis V2 is the default hardened pattern-analysis workflow.

```bash
python3 -m uwos.pattern_analysis_v2 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of latest
```

The default output path is:

```text
/Users/anuppamvi/uw_root/tradedesk/out/pattern_analysis_v2/YYYY-MM-DD
```

`python3 -m uwos.options_pattern_pipeline_v2` is a compatibility alias for the
same engine.

V2 emits ticket-first `AUTO_APPROVED`, `TRADE_REVIEW`, `AVOID`, and `NO_TRADE`
statuses plus decision-board, artifact-manifest, walk-forward, threshold,
calibration, shadow-ledger, profitability-audit, and runbook artifacts.

## Input Contract

A V2 signal date must contain the three execution-critical UW downloads:

1. `stock-screener`: price, sector, issue type, volatility state, 52-week
   location, volume baselines, and dated earnings calendar.
2. `hot-chains`: same-day tradeable contracts, liquidity, ask/bid activity,
   sweeps, and multi-leg volume. Multi-leg volume is removed per contract
   before directional aggregation.
3. `chain-oi-changes`: positive OI creation and the prior session's option
   quotes. OI unwinds are not new positioning. Ask/bid opening conviction is
   calculated after removing multi-leg volume.
The following feeds are family-specific rather than global date gates:

4. `dp-eod-report`: dark-pool premium and NBBO location. Average-price,
   contingent, odd-lot, and prior-reference-price conditions are excluded from
   directional pressure.
5. `bot-eod-report`: signed customer premium, vega, and gamma. Canceled rows and
   explicit multi-leg conditions are excluded from directional flow. Derived
   cache schema changes invalidate stale caches automatically.

An OI pattern requires chain/OI, a dark-pool pattern requires dark pool, and a
tape/Greek pattern requires bot-EOD. Missing an unrelated optional feed does not
discard otherwise valid price/momentum/OI history. Dates missing a core source
cannot generate patterns. They can remain in the market-session clock when they
have dated stock or option data, so a lifecycle does not skip intervening trading
days.

Snapshots retain only contracts reachable through deterministic ticket
selection. Selected prior-session chain quotes are persisted in SQLite, loaded
one signal date at a time for scoring, then released. Next-day `curr_oi`, volume,
and stock price are never used as prior-day inputs.

## Validation Contract

Long options use a +50% whole-position profit target, no mechanical price stop,
and a 40-session maximum hold. Credit spreads and long strangles retain their
predeclared five-session lifecycle. Every validation row records its strategy's
primary horizon.

Pattern promotion requires positive economics, sufficient OOS sample, positive
validation folds, real option replay after costs, and evidence against matched
controls. A ranked pattern is not automatically tradeable.

Historical validation retains up to 500 primary events per signal date using
the same deterministic contract selector as the daily board. It then reserves
every observed detailed family lane (family, sector, direction, and strategy),
both directional/strategy floors, neutral volatility, and source-rescue lanes.
This avoids validating thousands of alternative contracts per event while
preserving cross-sector and cross-strategy coverage.

V2 predeclares symmetric sector-relative momentum continuation and reversal
families. Top- and bottom-quintile names are tested in both directions and as
long options and credit spreads; the pipeline does not assume continuation or
reversal in advance.

## Ranking And Context

Every run writes:

- `pattern_registry.csv`: best-to-worst observed OOS economics across current
  families and completed research hypotheses. Rejected patterns remain visible.
- `pattern_ranking_summary.json`: explicit best/worst rows and status counts.
- `primary_source_coverage.csv`: per-date presence, bytes, and inclusion for all
  five canonical downloads.
- `external_context_audit.csv` and `external_ticker_context.csv`: timestamp and
  provenance audit for SEC, news, browser text, and X captures.
- `decision_board_context.csv`: the normal board plus external context. External
  data can require review but can never promote a ticket.
- `conditional_trade_tickets.csv`: yellow, fully structured `TRADE_REVIEW`
  tickets with spot, exact entry, target close price, max risk, management, and
  activation requirements. `send_now=no` until those requirements and a quote
  refresh pass.

X/Twitter is shadow-only until prospective evidence exists. Untimestamped SEC,
news, and browser captures are context-only. SEC can trigger an event review
only with an acceptance or dissemination timestamp available by the signal
date; filing date alone is insufficient for a historical veto.

The implementation still reuses hardened source loading, contract pricing,
risk gates, and artifact writers from the active V1 engine. V2 owns the stricter
input universe, corrected lifecycle, external-context policy, and ranked
research registry. The frozen V1 backup remains unchanged.
