# Codex Daily Pipeline V1 Restore Point

This repository preserves the Codex daily options-income pipeline V1 as a Git restore point.

## V1 identity

- Tag: `codex-daily-pipeline-v1`
- Backup branch: `backup/codex-daily-pipeline-v1`
- Main working branch at creation: `pipeline-profitability-fix`

The tag and backup branch are intended to stay fixed. Future pipeline work should happen on normal feature branches or the active working branch, not by moving the V1 tag or rewriting the backup branch.

## Restore V1

Fetch the saved restore point:

```bash
git fetch origin --tags
```

Create a clean branch from V1:

```bash
git switch -c restore/codex-daily-pipeline-v1 codex-daily-pipeline-v1
```

Or restore only the pipeline files into the current branch:

```bash
git checkout codex-daily-pipeline-v1 -- \
  codexuw \
  uwos/run_mode_a_two_stage.py \
  uwos/rulebook_config_goal_holistic_claude.yaml \
  tests/test_codexuw_daily.py \
  tests/test_codexuw_contexts.py \
  tests/test_codexuw_replay.py \
  README.md \
  CLAUDE.md
```

## Change policy after V1

- Do not delete or retarget `codex-daily-pipeline-v1`.
- Do not force-push `backup/codex-daily-pipeline-v1`.
- If a future change degrades the pipeline, compare against the tag first:

```bash
git diff codex-daily-pipeline-v1..HEAD -- codexuw uwos tests README.md
```

- If needed, revert the active branch or cherry-pick the V1 files from the tag instead of editing the backup branch.
