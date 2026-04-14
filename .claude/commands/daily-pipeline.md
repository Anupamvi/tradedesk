Run the daily 2-stage options pipeline (Stage-1 discovery + Stage-2 Schwab live validation) for the given date.

## Parameters
- Date: $ARGUMENTS (YYYY-MM-DD format). If empty, use today's date.

## Steps

1. Verify data exists at `c:/uw_root/{date}/` (should have zip files)
2. Run the pipeline:
```bash
cd "c:/uw_root" && python -m uwos.run_mode_a_two_stage \
  --base-dir "c:/uw_root/{date}" \
  --config "c:/uw_root/uwos/rulebook_config_goal_holistic_claude.yaml" \
  --out-dir "c:/uw_root/out/{date}" \
  --top-trades 20
```
Use timeout 300000ms (5 min) for Schwab API.

3. Read full output if truncated — extract all trade tables.

4. Present results in this format:

### Header
`**{N} Approved / 20 total** | **Core: {X}, Tactical: {Y}, Watch: {Z}**`

### Core Book Table
| # | Ticker | Strategy | Strike Setup | Expiry | DTE | Cost | Max Profit | Conviction | Edge |

### Tactical Book Table
| # | Ticker | Strategy | Strike Setup | Expiry | DTE | Credit/Debit | Max Profit | Conviction | Edge | Blockers |
Bold any SHIELD (credit) trades.

### Watch Summary
One-line bullets explaining why each was blocked.

### Highlights
3-5 bullet observations: new entrants, notable signals, portfolio balance, day-over-day changes.

### Output Files (ALWAYS include clickable VSCode-relative links)
- Expert trade table: `[anu-expert-trade-table-{date}.md]({date}/anu-expert-trade-table-{date}.md)`
- Live trade CSV: `[live_trade_table_{date}_final.csv](out/{date}/live_trade_table_{date}_final.csv)`
- Setup likelihood: `[setup_likelihood_{date}.md](out/{date}/setup_likelihood_{date}.md)`
- Dropped trades: `[dropped_trades_{date}.csv](out/{date}/dropped_trades_{date}.csv)`
- Run manifest: `[run_manifest_{date}.json](out/{date}/run_manifest_{date}.json)`
- Schwab snapshot: `[schwab_snapshot_{date}.json](out/{date}/schwab_snapshot_{date}.json)`

## Entry Gate Tolerance

Width-based entry gate tolerance (pro swing-trade standard):
- `tolerance = max(floor, spread_width × width_pct)` — config: 5% width, $0.30 floor
- Near-miss trades within tolerance are approved — no penalty for small slippage

## Error Handling
- If data dir missing: tell user `c:/uw_root/{date}/` not found
- If pipeline fails: show full error output
- If Schwab auth fails: suggest checking `.env` token
