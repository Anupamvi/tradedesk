---
name: daily-pipeline
description: Use when user asks to run the daily 2-stage options pipeline, analyze best trades for a date, or run the mode-a pipeline. Triggers on phrases like "run pipeline", "daily analysis", "best trades for", "2-stage", "run mode a".
---

# Daily 2-Stage Options Pipeline

Run the UW 2-stage pipeline (Stage-1 EOD discovery + Stage-2 Schwab live quote/GEX validation) for a given trading date and present the actionable table first, with clickable output links.

Default: a plain "run daily pipeline for DATE" request is an EOD live-planning run. It uses dated EOD files for discovery, then current Schwab option quotes and current Schwab chain GEX for entry decisions. Do not use `--historical-replay` unless the user explicitly asks for backtesting, replay, or deterministic historical audit.

Live action meanings:
- `ENTER`: executable at the current Schwab quote.
- `TARGET`: valid setup; use the shown target limit price and do not chase the current quote.
- `REVIEW`: valid setup but existing portfolio exposure requires an explicit add/adjust decision.
- `WAIT`: missing live quote/GEX/data or replay-only blocker.
- `SKIP`: rejected/not approved.

## Parameters

The user provides a **date** (YYYY-MM-DD). If no date given, use today's date from context.

## Execution Steps

1. **Verify data exists**: Use the current repo root. On macOS this repo is usually `/Users/anuppamvi/uw_root/tradedesk`; on Windows it is usually `c:/uw_root`. Check `{repo_root}/{date}/` has the required zip files.
2. **Run pipeline**:
```bash
cd "{repo_root}" && python -m uwos.run_mode_a_two_stage \
  --base-dir "{repo_root}/{date}" \
  --config "{repo_root}/uwos/rulebook_config_goal_holistic_claude.yaml" \
  --out-dir "{repo_root}/out/daily_pipeline_{date}" \
  --output "{repo_root}/out/daily_pipeline_{date}/anu-expert-trade-table-{date}.md" \
  --top-trades 20 \
  --eod-live-planning
```
   Use timeout of 300000ms (5 min) as Schwab API calls take time.
3. **Read full output** if truncated — extract all trade tables
4. **Present summary** in this exact format:

## Output Format

### Header
```
**{N} Actionable / 20 total** | **Core: {X}, Tactical: {Y}, Pilot: {P}, Scout: {S}, Watch: {Z}** | **Actions: ENTER={E}, TARGET={T}, REVIEW={R}, WAIT={W}**
```

### Actionable Table
Present Core/Tactical/Pilot rows first. Include `Live Action`, target limit/entry gate, portfolio context, max loss, max profit, conviction, and the exact blocker only when action is `REVIEW` or `WAIT`.

### Watch Summary
Only after the actionable table, summarize top Watch rows and why they were blocked.

### Highlights
3-5 bullet observations: new entrants, notable signals, portfolio balance, day-over-day changes if prior day data available.

### Output Files (ALWAYS include — clickable VSCode-relative links)
```
- Expert trade table: [anu-expert-trade-table-{date}.md](out/daily_pipeline_{date}/anu-expert-trade-table-{date}.md)
- Live trade CSV: [live_trade_table_{date}_final.csv](out/daily_pipeline_{date}/live_trade_table_{date}_final.csv)
- Setup likelihood: [setup_likelihood_{date}.md](out/daily_pipeline_{date}/setup_likelihood_{date}.md)
- Dropped trades: [dropped_trades_{date}.csv](out/daily_pipeline_{date}/dropped_trades_{date}.csv)
- Run manifest: [run_manifest_{date}.json](out/daily_pipeline_{date}/run_manifest_{date}.json)
- Schwab snapshot: [schwab_snapshot_{date}.json](out/daily_pipeline_{date}/schwab_snapshot_{date}.json)
```

## Entry Gate Tolerance

The pipeline uses **width-based entry gate tolerance** (professional swing-trade standard):
- Formula: `tolerance = max(floor, spread_width × width_pct)`
- Config: `entry_tolerance_width_pct: 0.05` (5%), `entry_tolerance_floor: 0.30` ($0.30)
- Example: $15 spread → tolerance = max($0.30, $15 × 0.05) = **$0.75**
- Near-miss trades within tolerance are approved as normal — no penalty for small slippage
- Applied consistently in both the pricer (Stage-2) and the approval stage

## Error Handling

- If data dir missing: tell user `{repo_root}/{date}/` not found
- If pipeline fails: show full error output
- If Schwab auth fails: suggest checking `.env` token or running `python -m uwos.schwab_auth`
