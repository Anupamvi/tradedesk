---
name: trend-analysis
description: Run the historical trend pipeline — scans replay manifests across days and builds ticker persistence, win-rate trends, and consolidated final trade recommendations.
---

# /trend-analysis Skill

Analyze replay manifests across multiple trading days to surface persistent setups, win-rate trends, and consolidated final trade recommendations.

## Usage

| Command | Description |
|---|---|
| `/trend-analysis` | Run trend analysis on default root (last 30 days) |
| `/trend-analysis 30` | Run with explicit 30-day lookback |
| `/trend-analysis 2026-01-01 2026-03-08` | Run for a specific date range |
| `/trend-analysis 30 20` | 30-day lookback, top 20 recommendations |

## Execution Steps

1. **Parse user arguments**
   - `lookback`: first numeric arg — number of trailing calendar days (default: `30`)
   - `start-date`: first date arg if two dates provided (YYYY-MM-DD)
   - `end-date`: second date arg if two dates provided (YYYY-MM-DD)
   - `top-n`: second numeric arg — top N recommendations to emit (default: `20`)

2. **Determine paths**
   - `search-root`: `c:/uw_root/out/replay_compare/`
   - `out-dir`: `c:/uw_root/out/replay_compare/trend_analysis/`

3. **Run the pipeline**

```bash
python -m uwos.historical_trend_pipeline \
  --search-root "c:/uw_root/out/replay_compare/" \
  --out-dir "c:/uw_root/out/replay_compare/trend_analysis/" \
  --lookback-days {lookback} \
  --recommendation-top-n {top-n} \
  --link-style vscode
```

   If date range provided instead of lookback, replace `--lookback-days` with:
   `--start-date {start-date} --end-date {end-date}`

4. **Read output files and present results**
   - Read `summary.md` first for high-level overview
   - Read `final_trade_recommendations_from_trends.md` for consolidated picks
   - Present top recommendations table with ticker, persistence days, proxy PF, and category
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
- Config: `c:\uw_root\uwos\rulebook_config.yaml`
