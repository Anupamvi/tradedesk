---
name: groko
description: >
  Run the Groko options pipeline for a date, show the color-coded trade board,
  and write GROKO_TRADES.md. Triggers include groko, groko YYYY-MM-DD, run groko,
  groko trades, groko board, groko 2026-08-27, and /groko. Forked Options Agent
  v1.84 under groko/; not Codex Daily V2. No order placement.
---

# Groko

Run Groko for one UW date, then show the trades. The operator file is the color board.

🟢 send-now / ready to enter. 🟡 working limit — do not chase. 🔴 blocked this session.

## Date

- User said `groko YYYY-MM-DD` → that date.
- Otherwise today's date from context.
- UW folder must exist at `/Users/anuppamvi/uw_root/tradedesk/YYYY-MM-DD`.
- If that folder is missing, use the latest complete `YYYY-MM-DD` folder under the same root, say so, and still `--live-schwab` for current quotes.
- Do not invent a date folder. Do not place orders.

## Run

From `/Users/anuppamvi/tradedesk`, timeout 600000ms:

```bash
python3 -m groko \
  --date YYYY-MM-DD \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/groko/YYYY-MM-DD \
  --live-schwab \
  --live-portfolio \
  --single-process-reviews
```

`--date` is the UW source date. Live Schwab after hours is valid (mid is a live quote when natural is crossed).

If the out-dir already has a fresh `GROKO_TRADES.md` from this date and the user only asked to show trades, read it. Re-run when they say run / refresh / live, or when the board is missing.

## After the run

1. Read `GROKO_TRADES.md` (same content as `TRADE_BOARD.md`).
2. Read `green_trade_tickets.csv` and `target_order_candidates.csv`.
3. Put **trades first** in the reply: the 🟢/🟡 table, then blockers, then file links.
4. 0 greens is valid only when the promoted selector has no 25–30% send-now name. 0 greens after selector PASS because live quotes stamped the next session is a pipeline bug, not a no-trade day. Do not loosen 25–30% credit, $0.50, or quote ≤25% to fill the table.
5. Bull-call debit may print if its independent selector is promoted. Bear-put / long options stay off until both frozen partitions pass.

## Reply shape

```
**Groko YYYY-MM-DD** | 🟢 {n} send-now | 🟡 {n} working | 🔴 {n} blocked

{paste GROKO_TRADES.md green/yellow sections; include the color table}

### Files
- Trades: [GROKO_TRADES.md](/Users/anuppamvi/uw_root/tradedesk/out/groko/YYYY-MM-DD/GROKO_TRADES.md)
- Report: [groko_report_YYYY-MM-DD.md](/Users/anuppamvi/uw_root/tradedesk/out/groko/YYYY-MM-DD/groko_report_YYYY-MM-DD.md)
- Green tickets: [green_trade_tickets.csv](/Users/anuppamvi/uw_root/tradedesk/out/groko/YYYY-MM-DD/green_trade_tickets.csv)
- Decision board: [decision_board.csv](/Users/anuppamvi/uw_root/tradedesk/out/groko/YYYY-MM-DD/decision_board.csv)
```

Workspace copies land in `/Users/anuppamvi/tradedesk/reports/YYYY-MM-DD/` when the pipeline mirrors them.

## Errors

- Missing UW folder and no latest date → stop, name the path.
- Schwab auth fail → `python3 -m groko --date YYYY-MM-DD --live-schwab` after refreshing the token in `.env` / `SCHWAB_TOKEN_PATH`.
- Pipeline non-zero → show the error; do not fake tickets.
