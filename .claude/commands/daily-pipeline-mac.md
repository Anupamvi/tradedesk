Run the daily 2-stage options pipeline on macOS (Stage-1 discovery + Stage-2 Schwab live validation) for the given date.

## Parameters
- Date: $ARGUMENTS (YYYY-MM-DD format). If empty, use today's date.

## Steps

1. Verify data exists at `/Users/anuppamvi/uw_root/tradedesk/{date}/`:
   - `hot-chains-{date}.zip`
   - `chain-oi-changes-{date}.zip`
   - `dp-eod-report-{date}.zip`
   - `stock-screener-{date}.zip`
   - `whale-{date}.md`

2. Run the pipeline:
```bash
cd "/Users/anuppamvi/uw_root/tradedesk" && python3 -m uwos.run_mode_a_two_stage \
  --base-dir "/Users/anuppamvi/uw_root/tradedesk/{date}" \
  --config "/Users/anuppamvi/uw_root/tradedesk/uwos/rulebook_config_goal_holistic_claude.yaml" \
  --out-dir "/Users/anuppamvi/uw_root/tradedesk/out/{date}" \
  --top-trades 20 \
  --strict-stage2
```

Use timeout 300000ms (5 min) for Schwab API.

3. If Schwab auth fails, create `/Users/anuppamvi/uw_root/tradedesk/.env` from `.env.example`, then run:
```bash
cd "/Users/anuppamvi/uw_root/tradedesk" && python3 -m uwos.schwab_quotes --symbols-csv AAPL --chain-symbols-csv AAPL --strike-count 2
```

4. Present results from `/Users/anuppamvi/uw_root/tradedesk/{date}/anu-expert-trade-table-{date}.md`.
