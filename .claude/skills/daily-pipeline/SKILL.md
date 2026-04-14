---
name: daily-pipeline
description: Use when user asks to run the daily 2-stage options pipeline, analyze best trades for a date, or run the mode-a pipeline. Triggers on phrases like "run pipeline", "daily analysis", "best trades for", "2-stage", "run mode a".
---

# Daily 2-Stage Options Pipeline

Run the UW 2-stage pipeline (Stage-1 discovery + Stage-2 Schwab live validation) for a given trading date and present approved trades with clickable output links.

## Parameters

The user provides a **date** (YYYY-MM-DD). If no date given, use today's date from context.

## Execution Steps

1. **Verify data exists**: Check `c:/uw_root/{date}/` has the required zip files
2. **Run pipeline**:
```bash
cd "c:/uw_root" && python -m uwos.run_mode_a_two_stage \
  --base-dir "c:/uw_root/{date}" \
  --config "c:/uw_root/uwos/rulebook_config_goal_holistic_claude.yaml" \
  --out-dir "c:/uw_root/out/{date}" \
  --top-trades 20
```
   Use timeout of 300000ms (5 min) as Schwab API calls take time.
3. **Read full output** if truncated — extract all trade tables
4. **Present summary** in this exact format:

## Output Format

### Header
```
**{N} Approved / 20 total** | **Core: {X}, Tactical: {Y}, Watch: {Z}**
```

### Core Book Table
| # | Ticker | Strategy | Strike Setup | Expiry | DTE | Cost | Max Profit | Conviction | Edge |

### Tactical Book Table (include track: FIRE or SHIELD)
| # | Ticker | Strategy | Strike Setup | Expiry | DTE | Credit/Debit | Max Profit | Conviction | Edge | Blockers |

Bold SHIELD trades in the Tactical table.

### Watch Summary
One-line bullets explaining why each was blocked.

### Highlights
3-5 bullet observations: new entrants, notable signals, portfolio balance, day-over-day changes if prior day data available.

### Output Files (ALWAYS include — clickable VSCode-relative links)
```
- Expert trade table: [anu-expert-trade-table-{date}.md]({date}/anu-expert-trade-table-{date}.md)
- Live trade CSV: [live_trade_table_{date}_final.csv](out/{date}/live_trade_table_{date}_final.csv)
- Setup likelihood: [setup_likelihood_{date}.md](out/{date}/setup_likelihood_{date}.md)
- Dropped trades: [dropped_trades_{date}.csv](out/{date}/dropped_trades_{date}.csv)
- Run manifest: [run_manifest_{date}.json](out/{date}/run_manifest_{date}.json)
- Schwab snapshot: [schwab_snapshot_{date}.json](out/{date}/schwab_snapshot_{date}.json)
```

## Entry Gate Tolerance

The pipeline uses **width-based entry gate tolerance** (professional swing-trade standard):
- Formula: `tolerance = max(floor, spread_width × width_pct)`
- Config: `entry_tolerance_width_pct: 0.05` (5%), `entry_tolerance_floor: 0.30` ($0.30)
- Example: $15 spread → tolerance = max($0.30, $15 × 0.05) = **$0.75**
- Near-miss trades within tolerance are approved as normal — no penalty for small slippage
- Applied consistently in both the pricer (Stage-2) and the approval stage

## Error Handling

- If data dir missing: tell user `c:/uw_root/{date}/` not found
- If pipeline fails: show full error output
- If Schwab auth fails: suggest checking `.env` token or running `python -m uwos.schwab_auth`
