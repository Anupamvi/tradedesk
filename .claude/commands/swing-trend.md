Run the swing trend multi-day pipeline that analyzes price trends across a lookback window with Schwab validation and historical backtest.

## Parameters
- Arguments: $ARGUMENTS
- Format: `{date} [L{lookback}]` — e.g., `2026-03-06 L5` or `2026-03-06 L30`
- If no date given, use today's date
- If no lookback given, default to L5
- "weekly" = L5, "monthly" = L30
- Multiple lookbacks: `2026-03-06 L5 L30` — run sequentially

## Steps

1. Run the pipeline:
```bash
cd "c:/uw_root" && python -m uwos.swing_trend_pipeline \
  --lookback {lookback} \
  --as-of {date}
```
Use timeout 300000ms (5 min) for Schwab API.

Add `--no-schwab` only if user says "skip schwab" or "no schwab".
Add `--no-backtest` only if user says "skip backtest" or "no backtest".

2. Read full output if truncated.

3. Present results:

### Header
`**Swing Trend Report — {date} (L{lookback})** | **{N} recommendations**`

### Recommendations Table
| # | Ticker | Direction | Strategy | Entry | Target | Stop | Edge | Schwab Valid | Confidence |

### Key Observations
3-5 bullets on sector themes, trend strength, notable signals.

### Output Files (ALWAYS include clickable VSCode-relative links)
- Report: `[swing-trend-report-{date}-L{lookback}.md](out/swing_trend/swing-trend-report-{date}-L{lookback}.md)`
- CSV: `[swing_trend_shortlist_{date}-L{lookback}.csv](out/swing_trend/swing_trend_shortlist_{date}-L{lookback}.csv)`

## Entry Gate Tolerance

Width-based entry gate tolerance (pro swing-trade standard):
- `tolerance = max(floor, spread_width × width_pct)` — config: 2.5% width, $0.25 floor
- Live spread cost vs estimated cost — trades within tolerance pass Schwab validation
- Applied to both 2-leg verticals and Iron Condors

## Error Handling
- If no dated folders found: tell user data directories are missing
- If Schwab auth fails: suggest `--no-schwab` flag or checking `.env` token
- If pipeline fails: show full error output
