# Trading Desk Agent Instructions

## Pattern Pipeline V1 Baseline

- `uwos/options_pattern_pipeline_v1_frozen_v1/` is the immutable backup copy of the Options Pattern Pipeline V1 baseline.
- Do not edit, delete, rename, reformat, regenerate, or bulk-update files under `uwos/options_pattern_pipeline_v1_frozen_v1/` unless the user explicitly asks to update the frozen V1 baseline.
- Future pattern-pipeline work should happen in `uwos/options_pattern_pipeline_v1/` or a newly named successor package such as `uwos/options_pattern_pipeline_v2/`.
- If a future change needs rollback to V1 behavior, restore from `uwos/options_pattern_pipeline_v1_frozen_v1/` instead of reconstructing the code manually.
- Before committing future pattern-pipeline changes, verify that `uwos/options_pattern_pipeline_v1_frozen_v1/` has no accidental diff.
