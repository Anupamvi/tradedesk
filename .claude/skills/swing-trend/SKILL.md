---
name: swing-trend
description: Use when user asks to run swing trend analysis, weekly trend pipeline, multi-day trend analysis, or lookback analysis. Triggers on phrases like "swing trend", "trend analysis", "lookback", "weekly trend", "swing pipeline".
---

# Swing Trend Multi-Day Pipeline

Run the UW swing trend pipeline that analyzes multi-day price trends across a lookback window, validates with Schwab live quotes, and backtests historical edge.

## Parameters

- **date** (YYYY-MM-DD): As-of date. Defaults to today if not specified.
- **lookback** (integer): Number of trading days to look back. Common values: 5 (1 week), 10, 30. Defaults to 5 if not specified.
- **no-schwab** (flag): Skip Schwab live validation. Off by default.
- **no-backtest** (flag): Skip historical backtest. Off by default.

If user says "weekly" use lookback=5. If user says "monthly" use lookback=30.

## Execution Steps

1. **Run pipeline**:
```bash
cd "c:/uw_root" && python -m uwos.swing_trend_pipeline \
  --lookback {lookback} \
  --as-of {date}
```
   Use timeout of 300000ms (5 min) for Schwab API calls.

   Add `--no-schwab` or `--no-backtest` only if user explicitly requests skipping those.

2. **Read full output** if truncated
3. **Present summary** following the format below

## Output Format

### Header
```
**Swing Trend Report — {date} (L{lookback})** | **{N} recommendations**
```

### Recommendations Table
| # | Ticker | Direction | Strategy | Entry | Target | Stop | Edge | Schwab Valid | Confidence |

### Key Observations
3-5 bullets on sector themes, trend strength, notable signals.

### Output Files (ALWAYS include — clickable VSCode-relative links)
```
- Report: [swing-trend-report-{date}-L{lookback}.md](out/swing_trend/swing-trend-report-{date}-L{lookback}.md)
- CSV: [swing_trend_shortlist_{date}-L{lookback}.csv](out/swing_trend/swing_trend_shortlist_{date}-L{lookback}.csv)
```

## Multiple Lookbacks

If user asks for multiple lookbacks (e.g., "run L5 and L30"), run them sequentially and present both sets of results with all output file links.

## Entry Gate Tolerance

The pipeline uses **width-based entry gate tolerance** (professional swing-trade standard):
- Formula: `tolerance = max(floor, spread_width × width_pct)`
- Config (`schwab_validation` section): `entry_tolerance_width_pct: 0.025` (2.5%), `entry_tolerance_floor: 0.25` ($0.25)
- Live spread cost is compared against estimated cost — trades within tolerance pass validation
- Both 2-leg verticals and Iron Condors are gated consistently

## Error Handling

- If no dated folders found in `c:/uw_root/`: tell user data directories are missing
- If Schwab auth fails: suggest `--no-schwab` flag or checking `.env` token
- If pipeline fails: show full error output
