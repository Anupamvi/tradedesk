Analyze replay manifests across multiple trading days to surface persistent setups, win-rate trends, and consolidated final trade recommendations.

## Parameters
- Arguments: $ARGUMENTS
- Format: `{lookback}` or `{start-date} {end-date}` or `{lookback} {top-n}`
- Examples: `30`, `2026-01-01 2026-03-08`, `30 20`
- Default lookback: 30 days; default top-N: 20 recommendations

## Steps

1. Parse arguments:
   - First numeric = lookback days (default 30)
   - If two dates (YYYY-MM-DD): start-date and end-date
   - Second numeric = top-N recommendations (default 20)

2. Run pipeline:
```bash
python -m uwos.historical_trend_pipeline \
  --search-root "c:/uw_root/out/replay_compare/" \
  --out-dir "c:/uw_root/out/replay_compare/trend_analysis/" \
  --lookback-days {lookback} \
  --recommendation-top-n {top-n} \
  --link-style vscode
```

   Or with date range:
```bash
python -m uwos.historical_trend_pipeline \
  --search-root "c:/uw_root/out/replay_compare/" \
  --out-dir "c:/uw_root/out/replay_compare/trend_analysis/" \
  --start-date {start-date} \
  --end-date {end-date} \
  --recommendation-top-n {top-n} \
  --link-style vscode
```

3. Read output files and present results:
   - Read `summary.md` for high-level overview
   - Read `final_trade_recommendations_from_trends.md` for consolidated picks
   - Present top recommendations table with ticker, persistence days, proxy PF, category
   - ALWAYS provide clickable VSCode-relative links to ALL output files

## Output Files (clickable links)
- Summary: `[summary.md](out/replay_compare/trend_analysis/summary.md)`
- Recommendations: `[final_trade_recommendations_from_trends.md](out/replay_compare/trend_analysis/final_trade_recommendations_from_trends.md)`
- Recommendations CSV: `[final_trade_recommendations_from_trends.csv](out/replay_compare/trend_analysis/final_trade_recommendations_from_trends.csv)`
- Ticker persistence: `[ticker_persistence.csv](out/replay_compare/trend_analysis/ticker_persistence.csv)`
- Daily win rates: `[daily_win_rates.csv](out/replay_compare/trend_analysis/daily_win_rates.csv)`

## Error Handling
- If search-root `c:/uw_root/out/replay_compare/` not found: tell user to run `/daily-pipeline` first to generate replay data
- If pipeline fails: show full error output
