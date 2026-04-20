Run dated-folder UW trend analysis and return high-conviction option candidates plus backtest-gated trade readiness.

## Parameters
- Arguments: $ARGUMENTS
- Format: `{as-of-date}` or `{as-of-date} {lookback}` or `{lookback}`
- Examples: `2026-04-17`, `2026-04-17 90`, `30`
- Default lookback: 90 usable market-data days
- Default top actionable trades: 15

## What This Runs

This command uses the dated folders under `/Users/anuppamvi/uw_root/tradedesk`, not the old replay-manifest-only path.

Primary goal: identify a few strong bullish/bearish candidates or spotted patterns to work on. It is acceptable to have no actionable trades when no structure clears the live/backtest/tradeability gate.

It analyzes:
- `stock-screener`
- `chain-oi-changes`
- `hot-chains`
- `dp-eod-report`
- whale summaries / whale CSVs when present

It then runs historical likelihood backtesting before marking anything actionable.

It also runs daily option quote replay by default. This uses local UW `hot-chains` / `chain-oi-changes` snapshots to price every spread leg at the signal date and the latest later option snapshot. If a later snapshot does not exist yet, both-leg entry quote coverage is reported as `ENTRY_OK`. If any required entry/exit leg cannot be priced, or if the spread mark violates defined-risk economics, the setup is blocked instead of marked actionable.

It can also run a walk-forward audit with `--walk-forward-samples N`. The audit reruns older signal dates, selects historical trades using signal-date evidence only, then scores future 5/10/20 market-day option-quote outcomes. Future P&L is never used as an entry gate.

It also writes a Research Confidence Audit from fixed historical buckets, a Research Horizon Audit split by holding period, a Strategy Family Audit with train/validation results for broad predeclared setup families, a Ticker Playbook Audit with train/validation results for ticker-specific setup behavior, a Rolling Ticker Playbook Forward Validation audit, plus a detailed research outcomes CSV with ticker/setup/horizon/P&L rows. This is not a parameter search; if no bucket/horizon is supportive and no broad family or ticker playbook is promotable, keep blocking Actionable Now trades.

When the user asks whether the trend engine is profitable, proven, a money printer, or asks to keep backtesting across all historical dated folders, run the batch proof command instead of only the single-date report. The batch proof asks: had `trend-analysis` run on every eligible historical date, what trades would it have emitted, what happened afterward, and which playbooks actually made money? Historical proof uses local UW option quote replay; do not use Schwab current chains for old as-of dates because that would leak current/live data into historical decisions. Read **Live-Eligible Playbooks** first; only supportive or emerging prior-only rolling playbooks can be worked live. Negative or decaying rolling playbooks are blocked.

The final gate is now also regime-aware and position-aware. It blocks directional debit trades that fight the latest broad market regime, blocks candidates when trade-desk already shows open option exposure in the same underlying, assigns a minimum confidence/position-size tier to every actionable trade, appends actionable trades to a post-trade tracker CSV, and refreshes tracked outcomes from local UW option snapshots on later runs.

Before the final Schwab/backtest gate, it also runs a repair optimizer:
- try an expiry before earnings when the original setup crosses earnings
- try more liquid debit-spread structures when the original debit is too thin/wide
- move credit-spread short legs farther OTM for delta risk
- move iron-condor wings farther OTM, or convert a neutral condor into a directional credit spread when trend evidence leans one way

Repaired trades are still actionable only if Schwab live validation, historical support, earnings, liquidity, delta, and GEX gates all pass.

The default professional-quality gate blocks lotto setups: underlying price must be at least `$20`, debit spread price must be at least `$0.75`, directional trades need at least `8` whale/institutional mention days, options flow cannot conflict with the trade direction, directional debit price/flow scores must be at least `60`, debit must be no more than `50%` of spread width, and the long strike cannot be more than `2%` OTM.

## Steps

1. Parse arguments:
   - If first argument is `YYYY-MM-DD`, use it as the as-of date.
   - If a number follows the date, use it as usable market-data-day lookback.
   - If only a number is supplied, use it as lookback and infer the latest available market-data folder.
   - If no arguments are supplied, use the latest available market-data date and 90 usable market-data days.
   - Weekend folders and dated folders missing `stock-screener` data do not count toward the lookback.

2. Run pipeline:
```bash
cd /Users/anuppamvi/uw_root/tradedesk
python3 -m uwos.trend_analysis {arguments}
```

   Examples:
```bash
cd /Users/anuppamvi/uw_root/tradedesk
python3 -m uwos.trend_analysis 2026-04-17
python3 -m uwos.trend_analysis 2026-04-17 90
python3 -m uwos.trend_analysis 2026-04-17 90 --top 20
```

   Only add `--no-backtest` if the user explicitly asks to skip backtesting.
   Only add `--no-schwab` if Schwab validation fails or the user asks to skip live validation.
   Only add `--quote-replay diagnostic` or `--quote-replay off` if the user explicitly asks to bypass the replay gate.
   Use `--walk-forward-samples N` when the user asks for confidence validation across older signal dates.
   Use `--reuse-walk-forward-raw`, `--reuse-walk-forward-outcomes`, and `--reuse-research-outcomes` when rerunning a large walk-forward audit against the same output directory so completed historical audit files are reused.

   For a full historical profit proof:
```bash
cd /Users/anuppamvi/uw_root/tradedesk
python3 -m uwos.trend_analysis_batch --start 2025-12-01 --end {date} --lookback {lookback} --horizons 20 --reuse-raw
```

3. Read output files and present results:
   - Read `trend-analysis-{date}-L{lookback}.md`.
   - If the batch proof was run, read `out/trend_analysis_batch/trend-analysis-batch-proof-START_END-L{lookback}.md` first. State the strict emitted-trade verdict, the Live-Eligible Playbooks, and any blocked rolling playbooks before discussing candidates.
   - Review **Walk-Forward Audit** before trusting the trade list.
   - Review **Research Confidence Audit**, **Research Horizon Audit**, **Strategy Family Audit**, **Ticker Playbook Audit**, and **Rolling Ticker Playbook Forward Validation** next; do not tune thresholds inside the same run to make a bucket look good.
   - Present **Backtest-Supported Candidate Shortlist** first.
   - Present **Actionable Trades** second.
   - Present **Max Conviction / Max Planned Risk** third as the highest-confidence actionable subset.
   - Present **Trade Workup** fourth as quality setups to research, not trades.
   - If no trades pass the historical-support/research gate, say that clearly and summarize Trade Workup separately from Pattern Candidates.
   - Do not present a pattern candidate as a trade.
   - Include absolute links to the report, actionable CSV, pattern CSV, and metadata JSON.

## Output Files
- Report: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-{date}-L{lookback}.md`
- Candidate CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-candidates-{date}-L{lookback}.csv`
- Current Setups CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-current-setups-{date}-L{lookback}.csv`
- Actionable CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-actionable-{date}-L{lookback}.csv`
- Pattern CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-patterns-{date}-L{lookback}.csv`
- Quote Replay CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-quote-replay-{date}-L{lookback}.csv`
- Walk-Forward CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-walk-forward-{date}-L{lookback}.csv`
- Research Audit CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-research-audit-{date}-L{lookback}.csv`
- Research Horizon Audit CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-research-audit-by-horizon-{date}-L{lookback}.csv`
- Strategy Family Audit CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-strategy-family-audit-{date}-L{lookback}.csv`
- Ticker Playbook Audit CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-ticker-playbook-audit-{date}-L{lookback}.csv`
- Rolling Ticker Playbook Audit CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-rolling-ticker-playbook-audit-{date}-L{lookback}.csv`
- Research Outcomes CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-research-outcomes-{date}-L{lookback}.csv`
- Post-Trade Tracker CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-trade-tracker.csv` with outcome fields refreshed on each run unless `--no-outcome-update` is used
- Metadata: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis/trend-analysis-metadata-{date}-L{lookback}.json`
- Batch Proof Report: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_batch/trend-analysis-batch-proof-START_END-L{lookback}.md`
- Batch Prior-Only Playbook Trades CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_batch/trend-analysis-batch-prior-only-playbook-trades-START_END-L{lookback}.csv`
- Batch Gap Diagnostics CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_batch/trend-analysis-batch-gap-diagnostics-START_END-L{lookback}.csv`

## Backtest Gate

Backtest-Supported Candidate Shortlist rows require:
- high swing score
- multiple independent confirmations across flow, OI, dark pool, whale presence, and price trend
- few contradictory signals
- at least one tested structure that passed the backtest gate

Directional price confirmation is preferred. If the options/flow evidence is strong while price is moving the other way, keep the row only when a structure passed backtest, and label the divergence clearly instead of treating it as an entry.

Backtest-Supported Candidate Shortlist rows are names to work on; they are not automatically trades.

Current Trade Setups are the practical workbench. `TRADE_SETUP` means the current structure is clean enough to research but lacks enough analog sample support. `REBUILD` means the thesis has support but the shown spread needs better liquidity, flow, expiry, or strike construction before entry.

A trade is actionable only when:
- likelihood backtest verdict is `PASS`
- swing score is at least `60.0`
- edge is at least the configured minimum, default `0.0%`
- analog signal count is at least the configured minimum, default `100`
- daily option quote replay is `PASS`, `PARTIAL_PASS`, or current-day `ENTRY_OK`
- Schwab live validation passes
- underlying price is at least `$20`
- debit spread price is at least `$0.75`
- directional trend trades have strong whale/institutional coverage scaled to the lookback and capped at `8` mention days
- options flow does not conflict with bullish/bearish trade direction
- directional debit spreads have price/flow scores >= `60`
- debit is <= `50%` of spread width
- long strike is no more than `2%` OTM
- earnings do not overlap the trade window, unless explicitly allowed
- bid/ask width is within the tradeability caps
- short-leg delta is at or below `0.30`
- iron condors are not in a volatile GEX regime
- market regime does not conflict with the trade direction
- trade-desk has no open option exposure in the same underlying
- rolling ticker-playbook validation is not negative

Rows that pass backtest/Schwab but fail tradeability are Risk-Blocked. Rows that fail the base gate are Pattern Candidates only.

Trade Workup rows are not entries. They pass professional, live, quote, and research-support gates, but stop short of Actionable Now because the historical support is positive but thin. These are the names to work on with a thesis and trigger instead of hiding them in generic Pattern Candidates.

Max Conviction / Max Planned Risk rows are the highest tier. They must already be Actionable Now, then also require strong score, edge, backtest sample support, whale coverage, tight liquidity, and price/flow/OI/dark-pool alignment. This means max pre-defined risk for one defined-risk trade, never all account capital.

Walk-Forward Audit is the confidence layer. A supportive audit raises trust; an empty, low-sample, or negative audit lowers trust even if the current trade list has Actionable rows.

Research Confidence Audit is the fixed-bucket sanity layer. Research Horizon Audit prevents repeated 5/10/20-day exits from looking like independent evidence. Strategy Family Audit is the broad trade-generation layer; Ticker Playbook Audit is the narrow ticker-specific layer. Rolling Ticker Playbook Forward Validation tests whether a ticker playbook would have kept working after it first became promotable using only prior data. Same-day ticker variants are deduped before playbook validation. A current setup needs either a promotable broad family or promotable ticker playbook, and that support cannot override live quote, Schwab, earnings, liquidity, price/flow quality, market regime, open-position awareness, rolling-forward failure/decay, or lotto gates.

Every actionable trade must show a position-size tier. `PROBE_ONLY`, `STARTER_RISK`, `STANDARD_RISK`, and `MAX_PLANNED_RISK` are risk caps, not encouragement to scale. Low-sample or insufficient-forward-validation trades should stay at `STARTER_RISK` or lower.

## Error Handling
- If no dated folders are found: tell the user the root path is wrong or the daily UW captures are missing.
- If Schwab auth fails: rerun once with `--no-schwab`, but state that live validation was skipped.
- If backtesting fails or produces no PASS rows: do not recommend trades; summarize the best pattern candidates only.
- If the pipeline fails: show the full error output.
