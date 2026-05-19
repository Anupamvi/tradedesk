# Options Pattern Pipeline Runbook

## Run Command

```bash
python3 -m uwos.options_pattern_pipeline_v1 \
  --base-dir /Users/anuppamvi/uw_root/tradedesk \
  --as-of YYYY-MM-DD \
  --out-dir /Users/anuppamvi/uw_root/tradedesk/out/options_pattern_pipeline_v1/YYYY-MM-DD
```

Use `--as-of latest` to resolve the latest source-complete dated UW folder.

## Artifact Order

Inspect `decision_board.csv` first. It is the ticket-first acceptance contract
and uses only `AUTO_APPROVED`, `TRADE_REVIEW`, `AVOID`, and `NO_TRADE`.

Then inspect:

- `daily_report_YYYY-MM-DD.md` for the human report.
- `artifact_manifest.json` for git SHA, command, config hash, source files,
  source fingerprints, cache status, artifact paths, runtime, and schema errors.
- `walk_forward_performance.csv` for OOS net R, win rate, profit factor,
  drawdown proxy, quote coverage, and blocker distribution.
- `threshold_sensitivity.csv` for old-vs-new gate behavior.
- `calibration_summary.md` for Brier score and reliability buckets.
- `shadow_recommendation_ledger.csv` and `shadow_outcome_summary.md` for shadow
  tracking.

## Status Meanings

`AUTO_APPROVED` means every configured gate passed. It is still not a profit
guarantee and still requires live broker quote/risk review before any real
order.

`TRADE_REVIEW` means a complete ticket exists, but the board lists exact
promotion requirements.

`AVOID` means the ticket failed a reject gate such as negative EV, poor quote,
wide spread, weak liquidity, max-risk breach, poor calibration, or failed
validation.

`NO_TRADE` means no reviewable ticket exists or a run-level kill-switch blocks
promotion.

## Source And Kill-Switch Handling

When source data is incomplete, source freshness/cache validation fails, quote
coverage is too low, unscorable rate is too high, validation drawdown breaches,
calibration fails, or schema validation fails, the pipeline prevents
`AUTO_APPROVED`.

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
