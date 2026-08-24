# Options Pattern Pipeline Runbook

## Run Command

```bash
python3 -m uwos.options_pattern_pipeline_v1 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of YYYY-MM-DD \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/options_pattern_pipeline_v1/YYYY-MM-DD
```

Pattern Analysis V2 is now the default operator entrypoint:

```bash
python3 -m uwos.pattern_analysis_v2 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of latest
```

Its default output path is:

```text
/Users/anuppamvi/uw_root/tradedesk/out/pattern_analysis_v2/YYYY-MM-DD
```

Use `--as-of latest` to resolve the latest source-complete dated UW folder.

## Pattern Analysis V2 Artifact Order

Inspect `daily_report.md`, `directional_board.csv`,
`current_option_setups.csv`, and `action_board.csv` first.

Then inspect:

- `managed_selection_audit.csv` for pre-holdout model selection and matched
  random-control evidence.
- `option_pattern_validation.csv` for chronological option returns after
  spreads and fees.
- `managed_price_pattern_validation.csv` for stock-direction evidence.
- `known_mover_audit.csv` for separate same-day, pre-event, and post-event
  coverage.
- `metadata.json` and `artifact_manifest.json` for reproducibility and
  acceptance blockers.

`TRADE_REVIEW` is a conditional next-session setup, not an approved order.
`RESEARCH_SETUP` failed a production evidence gate. Historical rows are always
non-executable. The V2 pipeline never places orders.

## Source And Kill-Switch Handling

When source data is incomplete, source freshness/cache validation fails, quote
coverage is too low, unscorable rate is too high, validation drawdown breaches,
calibration fails, or schema validation fails, the pipeline prevents
production qualification.

## Rollback

The frozen baseline lives at:

```text
uwos/options_pattern_pipeline_v1_frozen_v1/
```

Do not edit it. Verify it remains unchanged with:

```bash
git diff --exit-code options-pattern-pipeline-v1 -- uwos/options_pattern_pipeline_v1_frozen_v1
```

## GitHub Sync

Commit only live pipeline code, tests, configs, docs, and generated acceptance
artifacts. Do not commit secrets, account numbers, Schwab credentials, tokens,
environment files, or unrelated user changes.
