# Pattern Analysis V2

Pattern Analysis V2 is an independent, price-first research pipeline. It scans
every eligible UW stock-screener row, validates directional stock patterns, and
then tests options as a separate implementation problem.

```bash
python3 -m uwos.pattern_analysis_v2 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of latest
```

The default output path is:

```text
/Users/anuppamvi/uw_root/tradedesk/out/pattern_analysis_v2/YYYY-MM-DD
```

`python3 -m uwos.options_pattern_pipeline_v2` is a compatibility alias.

## Evidence Rules

- Features use only information available at the signal-date close.
- Stock splits are adjusted point in time.
- Same-day event detection, pre-event prediction, and post-event follow-up are
  separate results.
- Backtests enter on the next eligible session at the ask and exit at later
  bids, with fees charged on every leg.
- Missing quotes remain unscored; they are never converted into wins.
- Train, validation, and untouched holdout periods must each pass.
- A production option lane must also add positive return over a matched
  same-date/sector random control in every period.
- Parameter selection occurs before the final holdout.
- The pipeline abstains when these gates do not pass. Historical profitability
  is not a guarantee of future profit.

## Operator Artifacts

Read these first:

1. `daily_report.md`
2. `directional_board.csv`
3. `current_option_setups.csv`
4. `action_board.csv`
5. `managed_selection_audit.csv`
6. `option_pattern_validation.csv`
7. `known_mover_audit.csv`
8. `artifact_manifest.json`

`directional_board.csv` contains current stock-pattern matches. These are not
option approvals.

`current_option_setups.csv` contains exact same-day EOD contracts for the
predeclared selected lane. They are conditional research tickets because the
validated backtest enters on the next session; the next-session quote must be
rechecked.

`action_board.csv` uses:

- `TRADE_REVIEW`: historically validated strategy and exact current setup, but
  the next-session live fill is not validated.
- `RESEARCH_SETUP`: exact current setup from a lane that failed a production
  selection/control gate.
- `HISTORICAL_APPROVED_TRADE`: old validated entry retained only as a backtest
  reference.
- `HISTORICAL_RESEARCH`: old research entry; never executable.
- `NO_EXECUTION_CANDIDATE`: no current option setup was generated.

The pipeline does not place orders.

## Research Basis

The implementation follows conservative public evidence:

- Stock momentum/reversal and earnings drift are treated as primary hypotheses
  ([Jegadeesh 1990](https://doi.org/10.1111/j.1540-6261.1990.tb05110.x),
  [Chan, Jegadeesh, and Lakonishok 1996](https://doi.org/10.1111/j.1540-6261.1996.tb05222.x)).
- Public option flow and open interest are not assumed to be directional;
  informed-flow research uses signed opening-buyer data that public aggregates
  do not reproduce
  ([Pan and Poteshman 2006](https://doi.org/10.1093/rfs/hhj024)).
- Option spreads, fees, and signal decay are part of the hypothesis, not a
  reporting adjustment
  ([Garleanu and Pedersen 2013](https://doi.org/10.1111/jofi.12080)).
- The fixed candidate registry, untouched holdout, matched control, and
  abstention policy address multiple-testing risk
  ([Harvey, Liu, and Zhu 2016](https://doi.org/10.1093/rfs/hhv059)).

Point-in-time earnings/news inputs require reliable publication timestamps.
The pipeline does not synthesize those timestamps from revised internet data.

## Cache Reuse

`--cache-dir` may point to a materialized managed-quote cache. Cache entries are
accepted only when the source-file signature and requested date namespace
match; decision artifacts and validation scores are always recomputed.
