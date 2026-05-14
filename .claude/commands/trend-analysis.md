Run the production UW options trend desk v2 and return leakage-safe actionable/watch/blocked trade setups.

## Parameters
- Arguments: $ARGUMENTS
- Format: `{as-of-date}` or `{as-of-date} {lookback}` or `{lookback}`
- Examples: `2026-05-08`, `2026-05-08 30`, `45`
- Default lookback: 30 usable market-data days

## What This Runs

This command now uses `uwos.trend_analysis_v2`, not the legacy `uwos.trend_analysis` scanner.

The v2 pipeline:
- discovers UW trend candidates from dated folders
- builds quote-aware option structures from local UW hot-chain snapshots
- runs leakage-safe validation scorecards and baseline comparisons
- audits missed movers on signal days
- avoids using current Schwab/live chains for historical validation
- emits actionable, watchlist, blocked, candidates, validation, missed-mover, regime, news, and metadata files

## Steps

1. Parse arguments:
   - If first argument is `YYYY-MM-DD`, use it as the as-of date.
   - If a number follows the date, use it as `--lookback`.
   - If only a number is supplied, omit the date and use it as `--lookback`.
   - If no lookback is supplied, use `--lookback 30`.

2. Run pipeline:

```bash
cd /Users/anuppamvi/uw_root/tradedesk
python3 -m uwos.trend_analysis_v2 {date} --lookback {lookback}
```

Examples:

```bash
cd /Users/anuppamvi/uw_root/tradedesk
python3 -m uwos.trend_analysis_v2 2026-05-08 --lookback 30
python3 -m uwos.trend_analysis_v2 --lookback 45
```

3. Read output files and present results:
   - Read `out/trend_analysis_v2/trend-analysis-v2-{date}-L{lookback}.md`.
   - Present Actionable Trades first.
   - If Actionable Trades is empty, say so clearly and present Watchlist / Research Setups as research-only.
   - Do not present watchlist or blocked rows as trades.
   - Put each setup up front: strategy, legs, expiry, debit/credit.
   - Include validation scorecard and missed-mover conclusions when explaining confidence.

## Output Files
- Report: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-{date}-L{lookback}.md`
- Actionable CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-actionable-{date}-L{lookback}.csv`
- Watchlist CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-watchlist-{date}-L{lookback}.csv`
- Blocked CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-blocked-{date}-L{lookback}.csv`
- Candidates CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-candidates-{date}-L{lookback}.csv`
- Validation Scorecard CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-validation-scorecard-{date}-L{lookback}.csv`
- Validation Outcomes CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-validation-outcomes-{date}-L{lookback}.csv`
- Missed Movers CSV: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-missed-movers-{date}-L{lookback}.csv`
- Metadata JSON: `/Users/anuppamvi/uw_root/tradedesk/out/trend_analysis_v2/trend-analysis-v2-metadata-{date}-L{lookback}.json`

## Error Handling
- If no dated folders are found: tell the user data directories are missing.
- If the pipeline fails: show the full error output.
- Do not fall back to legacy v1 unless the user explicitly asks for legacy comparison.
