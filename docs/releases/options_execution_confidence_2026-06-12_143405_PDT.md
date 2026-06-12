# Options Execution Confidence Release

- Release id: `exec-confidence-20260612-143405`
- Released: `2026-06-12T14:34:05-07:00`
- Acceptance verdict: `PASS`
- Confidence score: `8.0`
- Goal completion gate: `can_mark_goal_complete=yes`

## Active Versions

| Surface | Active version |
|:--|:--|
| Codex Daily V3 | `v3.1-exec-confidence-20260612-143405` |
| Codex Daily V4 | `v4.1-exec-confidence-20260612-143405` |
| Options Pattern Pipeline V1 | `options_pattern_pipeline_v1.3-exec-confidence-20260612-143405` |
| Options Agent | `options-agent-v1.0-exec-confidence-20260612-143405` |

## Retained Previous Versions

| Surface | Previous version | Retention path |
|:--|:--|:--|
| Codex Daily V3 | `v3.0` | `codexuw.pipeline_versions.PREVIOUS_PIPELINE_VERSION_LOCKS` |
| Codex Daily V4 | `v4.0` | `codexuw.pipeline_versions.PREVIOUS_PIPELINE_VERSION_LOCKS` |
| Options Pattern Pipeline V1 | `options_pattern_pipeline_v1.2` | `uwos/options_pattern_pipeline_v1/VERSION.md` history |
| Options Pattern Pipeline V1 frozen baseline | `options_pattern_pipeline_v1.0` | `uwos/options_pattern_pipeline_v1_frozen_v1/` |
| Options Agent | `options-agent-v0` | Git history and locked V0 runtime flags in `uwos/options_agent/core.py` |

## Proof Packet

| Artifact | Result |
|:--|:--|
| Full uncapped CodexDaily historical proof | `66/66 PASS`, `fail_count=0`, `proof_scope=FULL` |
| Source-complete dates | `33` dates from `2026-03-27` through `2026-06-09` |
| Options Pattern matrix | `PASS`, `7504` rows checked, `544` auto, `6960` target-ready |
| Major ticker coverage | `AAPL, AMD, GOOG, GOOGL, HOOD, META, MSFT, MU, NOW, NVDA, PLTR` |
| V3 functional gates | `PASS`; uncapped discovery defaults and negative-edge NOW gate verified |
| V4 functional gates | `PASS`; independent V4 path and negative-edge gate verified |
| Trade Desk management | `PASS`; spread-level NOW close/roll guidance verified |

## Proof Hashes

These hashes point to the proof files generated during the release validation run.

| Artifact | SHA-256 |
|:--|:--|
| `/tmp/options_recommendation_goal_acceptance_exec_confidence_20260612_143405/goal_acceptance_report.md` | `307b81e81e72771e98e3be842ed21f566089bf5b03579b021af61269981e700a` |
| `/tmp/options_recommendation_goal_acceptance_exec_confidence_20260612_143405/goal_acceptance_summary.csv` | `0c5d1f43c951673ce6a58edd56fb5a29166307325e3aed1c5041f72b6c50c23b` |
| `/tmp/codexdaily_historical_proof_uncapped_current/codexdaily_historical_proof_summary.csv` | `d06594f76c2a6cc3c5c81f08b05357b833386ba4a8c3906fff416b48f9887739` |

## Validation Commands

```bash
env PYTHONPYCACHEPREFIX=/tmp/tradedesk_pycache python3 -m pytest \
  tests/test_options_pattern_goal_matrix.py \
  tests/test_options_recommendation_goal_acceptance.py \
  tests/test_run_codexdaily_historical_proof.py \
  tests/test_codexuw_contexts.py \
  tests/test_codexuw_v3.py \
  tests/test_codexuw_v4.py \
  tests/test_codexuw_trade_desk_engine.py \
  tests/test_trade_desk.py \
  -q -p no:cacheprovider
```

```bash
env PYTHONPYCACHEPREFIX=/tmp/tradedesk_pycache python3 scripts/run_codexdaily_historical_proof.py \
  --root /Users/anuppamvi/uw_root/tradedesk \
  --out-dir /tmp/codexdaily_historical_proof_uncapped_current \
  --from-date 2026-01-01 \
  --to-date 2026-06-09 \
  --pipeline both \
  --risk-budget 3000
```

```bash
env PYTHONPYCACHEPREFIX=/tmp/tradedesk_pycache python3 scripts/options_recommendation_goal_acceptance.py \
  --root /Users/anuppamvi/uw_root/tradedesk \
  --options-pattern-matrix-dir /Users/anuppamvi/uw_root/tradedesk/out/options_pattern_pipeline_v1/goal_uncapped_current_v1 \
  --codexdaily-proof-dir /tmp/codexdaily_historical_proof_uncapped_current \
  --out-dir /tmp/options_recommendation_goal_acceptance_exec_confidence_20260612_143405 \
  --as-of 2026-06-09
```
