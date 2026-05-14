# Options Pattern Pipeline v1

This pipeline is separate from the older trend pipeline. It reads dated,
source-like Unusual Whales exports directly from `YYYY-MM-DD` folders and
ignores prior trend scores, gates, candidate files, rejection labels, and
generated watchlists as model inputs.

## Flow Source Priority

For each dated UW folder, `bot-eod-report-YYYY-MM-DD` is the mandatory primary
options-flow source when present. If bot EOD exists, `option-trades` and
`whale_trades_filtered.csv` are fallback-only and are recorded as skipped with
`bot_eod_present_primary_flow_source`. If bot EOD is not present, the pipeline
falls back to `option-trades` and then `whale_trades_filtered.csv`.

Bot EOD files are not subject to `--max-flow-file-mb`. Large bot EOD files are
streamed once into fingerprinted derived caches under:

```text
/Users/anuppamvi/uw_root/tradedesk/out/options_pattern_pipeline_v1/cache/bot_eod
```

The cache is reused only when the source zip/member fingerprint still matches.
The size cap applies only to non-bot fallback flow files.

For a fast ingestion check on one date, add `--no-validation`; that mode builds
only the requested `--as-of` snapshot and skips historical replay/missed-mover
work. Omit `--no-validation` for the production validation run.

## Latest Daily Run

```bash
python3 -m uwos.options_pattern_pipeline_v1 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of latest
```

When `--out-dir` is omitted, `latest` is resolved to the latest
source-complete UW folder and outputs are written to:

```text
/Users/anuppamvi/uw_root/tradedesk/out/options_pattern_pipeline_v1/YYYY-MM-DD
```

To force a specific dated output directory:

```bash
python3 -m uwos.options_pattern_pipeline_v1 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of YYYY-MM-DD \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/options_pattern_pipeline_v1/YYYY-MM-DD
```

## Historical Date

```bash
python3 -m uwos.options_pattern_pipeline_v1 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of 2026-04-30 \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/options_pattern_pipeline_v1/2026-04-30
```

## Outputs

Each run writes:

- `daily_report_YYYY-MM-DD.md`
- `actionable_trades.csv`
- `watchlist_research_setups.csv`
- `blocked_candidates.csv`
- `discovered_pattern_families.csv`
- `market_regime_summary.json`
- `sentiment_news_summary.json`
- `validation_scorecard.csv`
- `baseline_comparison.csv`
- `missed_mover_audit.csv`
- `metadata.json`

Historical option outcomes are labeled `SCORED`, `PARTIAL`, or `UNSCORABLE`.
Unscorable ideas are never counted as wins.

`PROVEN` requires positive pooled validation, enough scored option outcomes,
baseline edge, split-consistent validation, and an acceptable losing-streak cap.
Families that are positive only because of one strong split are kept at
`PROMISING` and are not emitted as actionable trades.

Pattern-family validation is sector-, direction-, and strategy-aware. For
example, a broad parent such as `VOL_EXPANSION_CATALYST` is validated as
families like `VOL_EXPANSION_CATALYST__BULLISH__LONG_OPTION__TECHNOLOGY` so a
repeatable sector-specific edge is not hidden by unrelated bearish, spread, or
non-sector variants.
