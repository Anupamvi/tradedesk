# Trading Pipelines Release Bundle

## Bundle identity

- Bundle version: `trading-pipelines-20260823.1`
- Release date: `2026-08-23`
- Git branch: `pipeline-profitability-fix`
- Git tag: `trading-pipelines-20260823.1`
- Repository: `git@github.com:Anupamvi/tradedesk.git`
- Release commit: the commit pointed to by the annotated tag above

This is a source synchronization and reproducibility stamp. It does not turn a
research or shadow lane into an order-authorized lane. Current Schwab state,
quotes, portfolio context, and the pipeline's execution gates remain required
at run time.

## Pipeline versions in this bundle

| Pipeline | Runtime/version recorded in source | Date in version | Scope |
|---|---|---:|---|
| Codex Daily V2 | `v2.1` | undated legacy version | `codexuw.daily` |
| Codex Daily V3 | `v3.2-profit-integrity-20260719` | 2026-07-19 | `codexuw.daily_v3` |
| Codex Daily V4 | `v4.27-debit-production-handoff-20260816` | 2026-08-16 | `codexuw.daily_v4` |
| Options Agent | `options-agent-v1.78-risk-normalized-bull-call-20260822-083000` | 2026-08-22 | `uwos.options_agent` |
| Fresh Schwab Wheel | `fresh-wheel-v1.2-calendar-seasonality-20260819` | 2026-08-19 | `uwos.fresh_wheel_schwab` |
| Options Pattern V1 | `options_pattern_pipeline_v1.16-profile-aware-daily-selection-20260722-000000` | 2026-07-22 | `uwos.options_pattern_pipeline_v1` |
| Pattern Analysis V2 | `pattern_analysis_v2.12-family-sources-symmetric-momentum-20260803` | 2026-08-03 | `uwos.pattern_analysis_v2` |
| Pattern Rebuild V1 | `pattern_rebuild_v1.69-explicit-mrvl-audit-20260822` | 2026-08-22 | `uwos.pattern_rebuild_v1` |
| LessonEngine | schema `1` | source schema | `uwos.lessonengine` |
| Trade Desk | bundle tag `trading-pipelines-20260823` | 2026-08-23 | `uwos.trade_desk` and related Schwab review modules |

The older `uwos.wheel_pipeline` module has no independent version constant and
was not treated as the active Schwab wheel runtime; the active wheel version is
the Fresh Schwab Wheel row above.

## SolCodex TradeDesk submodule

SolCodex remains an independent repository, but it is now checked out under
TradeDesk at `solcodex/` as a tracked Git submodule. Its source and tests are
committed and tagged as:

- Commit: `9ec63158e34282a27e8b4cdd5ae335bbcaed9829`
- Tag: `solcodex-v2.4.0-20260823`
- Version: `2.4.0`
- Remote: `git@github.com:Anupamvi/solcodex.git`
- TradeDesk path: `/Users/anuppamvi/uw_root/tradedesk/solcodex`

The independent repository preserves SolCodex history and release ownership;
the submodule makes its exact version part of every TradeDesk checkout.

## Verification from another PC

```bash
git clone --recurse-submodules git@github.com:Anupamvi/tradedesk.git
cd tradedesk
git fetch origin --tags
git checkout pipeline-profitability-fix
git show --no-patch --decorate trading-pipelines-20260823.1
git submodule status
git status --short
```

Expected result: the tag resolves to the synchronized release commit, the
SolCodex submodule resolves to `9ec63158`, and `git status --short` is empty
for the tracked source tree.

## Excluded local material

The commit deliberately excludes raw/date-folder UW exports, ignored runtime
outputs, absolute-link staging data, generated overlay caches, and the separate
`anu-trading-desk-site` repository. None of those files were deleted.
