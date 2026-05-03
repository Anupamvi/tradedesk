Run the daily 2-stage options pipeline (Stage-1 EOD discovery + Stage-2 Schwab live quote/GEX validation) for the given date.

## Parameters
- Date: $ARGUMENTS (YYYY-MM-DD format). If empty, use today's date.

## Steps

1. Verify data exists at `c:/uw_root/{date}/` (should have zip files)
2. Run the normal EOD live-planning pipeline. Do not use `--historical-replay` unless the user explicitly asks for backtesting/replay:
```bash
cd "c:/uw_root" && python -m uwos.run_mode_a_two_stage \
  --base-dir "c:/uw_root/{date}" \
  --config "c:/uw_root/uwos/rulebook_config_goal_holistic_claude.yaml" \
  --out-dir "c:/uw_root/out/daily_pipeline_{date}" \
  --output "c:/uw_root/out/daily_pipeline_{date}/anu-expert-trade-table-{date}.md" \
  --top-trades 20 \
  --eod-live-planning
```
Use timeout 300000ms (5 min) for Schwab API.

3. Read full output if truncated — extract all trade tables.

4. Present results in this format:

### Header
`**{N} Actionable / 20 total** | **Core: {X}, Tactical: {Y}, Pilot: {P}, Scout: {S}, Watch: {Z}** | **Actions: ENTER={E}, TARGET={T}, REVIEW={R}, WAIT={W}**`

### Actionable Table
Show Core/Tactical/Pilot rows first. Include Live Action, target limit/entry gate, portfolio context, max loss, max profit, conviction, and blocker text only when action is REVIEW or WAIT. `TARGET` means use the shown limit price; it is not a price-only WAIT.

### Watch Summary
Only after the actionable table, one-line bullets explaining why each high-ranked Watch row was blocked.

### Highlights
3-5 bullet observations: new entrants, notable signals, portfolio balance, day-over-day changes.

### Output Files (ALWAYS include clickable VSCode-relative links)
- Expert trade table: `[anu-expert-trade-table-{date}.md](out/daily_pipeline_{date}/anu-expert-trade-table-{date}.md)`
- Live trade CSV: `[live_trade_table_{date}_final.csv](out/daily_pipeline_{date}/live_trade_table_{date}_final.csv)`
- Setup likelihood: `[setup_likelihood_{date}.md](out/daily_pipeline_{date}/setup_likelihood_{date}.md)`
- Dropped trades: `[dropped_trades_{date}.csv](out/daily_pipeline_{date}/dropped_trades_{date}.csv)`
- Run manifest: `[run_manifest_{date}.json](out/daily_pipeline_{date}/run_manifest_{date}.json)`
- Schwab snapshot: `[schwab_snapshot_{date}.json](out/daily_pipeline_{date}/schwab_snapshot_{date}.json)`

## Entry Gate Tolerance

Width-based entry gate tolerance (pro swing-trade standard):
- `tolerance = max(floor, spread_width × width_pct)` — config: 5% width, $0.30 floor
- Near-miss trades within tolerance are approved — no penalty for small slippage

## Error Handling
- If data dir missing: tell user `c:/uw_root/{date}/` not found
- If pipeline fails: show full error output
- If Schwab auth fails: suggest checking `.env` token
