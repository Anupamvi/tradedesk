---
name: trend-analysis
description: Run the historical trend pipeline — scans replay manifests across days and builds ticker persistence, win-rate trends, and consolidated final trade recommendations.
---

# /trend-analysis Skill

Analyze replay manifests across multiple trading days to surface persistent setups, win-rate trends, and consolidated final trade recommendations.

## Usage

| Command | Description |
|---|---|
| `/trend-analysis` | Run trend analysis using today as end-date, 30-day lookback |
| `/trend-analysis 2026-03-20` | Run with 2026-03-20 as end-date, 30-day lookback |
| `/trend-analysis 30` | Run with explicit 30-day lookback from today |
| `/trend-analysis 2026-01-01 2026-03-08` | Run for a specific date range |
| `/trend-analysis 30 20` | 30-day lookback, top 20 recommendations |

## Execution Steps

### 1. Parse user arguments
- If a single YYYY-MM-DD date is provided, treat it as `end-date` with default 30-day lookback. Compute `start-date` = `end-date` minus 30 calendar days.
- If two YYYY-MM-DD dates are provided, use them as `start-date` and `end-date`.
- `lookback`: first numeric arg — number of trailing calendar days (default: `30`). Used to compute `start-date` from `end-date`.
- `top-n`: second numeric arg — top N recommendations to emit (default: `20`)
- If no date is provided, use today's date as `end-date`.

### 2. Determine the date window
- Compute the list of trading days between `start-date` and `end-date`.
- `data-root`: `c:/uw_root/` — where daily input data lives (e.g., `c:/uw_root/2026-03-20/`)
- `search-root`: `c:/uw_root/out/replay_compare/` — where replay manifests live
- `out-dir`: `c:/uw_root/out/replay_compare/trend_analysis/`
- `variant`: `holistic_newstrategy_pf115_rebalanced_v3`
- `config`: `c:/uw_root/uwos/rulebook_config_goal_holistic.yaml`

### 3. Backfill missing replay compare runs (CRITICAL)
Before running trend analysis, ensure replay manifests exist for ALL trading days in the window.

For each trading day in the window:
1. Check if `c:/uw_root/out/replay_compare/{date}/holistic_newstrategy_pf115_rebalanced_v3/run_manifest_{date}.json` exists.
2. Check if `c:/uw_root/{date}/` has input data (zip or csv files like stock-screener, chain-oi-changes, etc.).
3. If input data exists BUT replay manifest is missing, run:

```bash
python -m uwos.run_mode_a_two_stage \
  --base-dir "c:/uw_root/{date}" \
  --config "c:/uw_root/uwos/rulebook_config_goal_holistic.yaml" \
  --out-dir "c:/uw_root/out/replay_compare/{date}/holistic_newstrategy_pf115_rebalanced_v3" \
  --top-trades 20
```

4. Run up to 5 dates in parallel (use background bash commands) to speed up backfill.
5. Report progress: "Backfilling replay data for N dates: {list}..."
6. If a date has no input data at all, skip it and note it in the output.

### 4. Run the trend pipeline

```bash
python -m uwos.historical_trend_pipeline \
  --search-root "c:/uw_root/out/replay_compare/" \
  --out-dir "c:/uw_root/out/replay_compare/trend_analysis/" \
  --start-date {start-date} \
  --end-date {end-date} \
  --recommendation-top-n {top-n} \
  --link-style vscode
```

### 5. Read output files and present results
- Read `summary.md` first for high-level overview
- Read `final_trade_recommendations_from_trends.md` for consolidated picks
- Present top recommendations table with ticker, persistence days, proxy PF, and category
- **Verify recommendations are current**: check that trade expiries are in the future and invalidation levels make sense vs current prices
- Always provide clickable file links to ALL output files

## Output Files

| File | Description |
|---|---|
| `trend_analysis/summary.md` | High-level daily/weekly trend summary with run links |
| `trend_analysis/final_trade_recommendations_from_trends.md` | Consolidated top-N recommendations |
| `trend_analysis/final_trade_recommendations_from_trends.csv` | Same data in CSV format |
| `trend_analysis/ticker_persistence.csv` | Per-ticker appearance counts across days |
| `trend_analysis/daily_win_rates.csv` | Daily win-rate breakdown per variant |

## Reference

- Pipeline code: `c:\uw_root\uwos\historical_trend_pipeline.py`
- Config: `c:\uw_root\uwos\rulebook_config_goal_holistic.yaml`
- Replay runner: `python -m uwos.run_mode_a_two_stage`
