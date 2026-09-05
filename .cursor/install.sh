#!/usr/bin/env bash
# Idempotent dependency setup for the UW Trade Desk in Cloud Agents.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Core runtime deps (requirements.txt) plus deps the pipelines import at load
# time but that are not yet pinned in requirements.txt:
#   - scikit-learn / scipy: used by codexuw walk-forward credit book and others
#   - pytest: the repository's test runner
# Constraints cap pandas to the 2.x line the code targets (see constraints.txt).
python3 -m pip install \
  -r requirements.txt \
  -c .cursor/constraints.txt \
  pytest \
  scikit-learn \
  scipy
