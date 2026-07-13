# Codex Daily V4.2 Integrity Release

V4.2 preserves the prior V4 module for rollback and adds a corrected execution path.

## Normal EOD run

```bash
python3 -m codexuw.daily_v42 run --root /Users/anuppamvi/uw_root/tradedesk --date YYYY-MM-DD
```

## EOD plus next-session OI overlay

This performs full discovery again. It does not merely reprice the old shortlist.

```bash
python3 -m codexuw.daily_v42 overlay \
  --root /Users/anuppamvi/uw_root/tradedesk \
  --date BASE-EOD-DATE \
  --overlay-file /absolute/path/chain-oi-changes-OVERLAY-DATE.csv \
  --overlay-date OVERLAY-DATE
```

## Output contract

- `codexdaily_v42_trade_table_DATE.md`: concise trader-facing report.
- `codexdaily_v42_decision_book_DATE.csv`: exact machine-readable decision book.
- `codexdaily_v42_scored_integrity_DATE.csv`: full enriched audit frame.
- `codexdaily_v42_condor_shadow_DATE.csv`: unpromoted strategy-expansion candidates.
- `codexdaily_v42_manifest_DATE.json`: release and source provenance.
- `codexdaily_v42_recommendation_ledger.csv`: persistent recommendation ledger.
- `codexdaily_v42_live_outcomes.csv`: realized-outcome ledger schema.

## Evidence rules

- UW generic `volatility` is not treated as historical volatility.
- Historical volatility, RSI, ATR, moving averages, relative strength, and 20-session anchored VWAP use Schwab daily candles.
- OI overlays are prior-session clearing confirmation, not current-session directional flow.
- GEX is calculated from Schwab gamma and OI and is labeled as a dealer-position proxy.
- Severe midpoint-versus-natural quote geometry is rejected before the actionable table.
- V4.2 may demote but cannot independently promote a V4 reject to ENTER NOW.
- Condors remain shadow candidates until a dedicated walk-forward replay passes.

## Rollback

```bash
python3 -m codexuw.daily_v4 run --root /Users/anuppamvi/uw_root/tradedesk --date YYYY-MM-DD
```
