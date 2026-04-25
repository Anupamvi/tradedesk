Review open Schwab option positions and produce HOLD/CLOSE/ROLL/SET STOP recommendations on macOS.

## Parameters
- Arguments: $ARGUMENTS
- Format: `{days}` or `{days} {symbol}`
- Examples: `90`, `60 AAPL`
- Default lookback: 90 days

## Steps

1. Parse arguments:
   - First numeric = Schwab trade-history lookback days
   - First non-numeric = optional underlying symbol filter

2. Run the macOS-safe trade desk report command:
```bash
cd "/Users/anuppamvi/uw_root/tradedesk" && python3 -m uwos.trade_desk {days} {symbol}
```
   - Baseline data source is Schwab API only: positions, transactions, quotes, and option chains.
   - Do not use yfinance unless the user explicitly asks for `--with-yfinance-context`.
   - Treat Unusual Whales research as a separate follow-up layer after the Schwab position review. If that follow-up uses local UW trend context, use full `bot-eod-report-YYYY-MM-DD.zip`/`.csv` first and `whale-YYYY-MM-DD.md` only as a legacy fallback for older folders with no bot EOD export.
   - The report is options-only; omit equities/funds from the recommendation summary.
   - Show option legs in readable form, e.g. `Short 1 HOOD 2027-01-15 $65 PUT`, not raw OCC symbols.
   - Preserve theta/gamma-aware recommendations: close positions where remaining premium is mostly risk, and close or set stops where theta decay makes recovery unlikely.

3. Read the generated markdown report:
`/Users/anuppamvi/uw_root/tradedesk/out/trade_analysis/trade-desk-{date}.md`

4. Present:
   - Verdict mix
   - All CLOSE / ROLL / SET STOP rows first
   - Use each row's `Do this` line as the simplified recommendation; do not present ASSESS as a user-facing action
   - Spread rows as one defined-risk position; CLOSE / ROLL means both legs together, never only one leg
   - Keep/HOLD summary
   - Links to report and position JSON

## Error Handling
- If Schwab auth fails, re-auth from the repo root:
```bash
cd "/Users/anuppamvi/uw_root/tradedesk" && python3 -m uwos.schwab_position_analyzer --manual-auth
```
- If `python` is not found, use `python3` on macOS.
