#!/usr/bin/env bash
# Block non-Composer/Grok subagents (no fast). Inherit uses parent model.

set -euo pipefail
HOOK_INPUT=$(cat)

RESULT=$(HOOK_INPUT="$HOOK_INPUT" python3 <<'PY'
import json, os, re

d = json.loads(os.environ["HOOK_INPUT"])
sub = (d.get("subagent_model") or "").strip()
parent = (d.get("model") or "").strip()
model = sub or parent
typ = d.get("subagent_type", "")
lower = model.lower()

allow = bool(re.search(r"composer|cursor-grok|grok-4", lower))
if "fast" in lower:
    allow = False

out = {"permission": "allow" if allow else "deny"}
if not allow:
    src = "inherit" if not sub and parent else "explicit"
    label = model if model else "unset"
    out["user_message"] = (
        f"Blocked subagent ({typ}) model={label} ({src}) — Composer/Grok non-fast only."
    )
print(json.dumps(out))
PY
)

printf '%s\n' "$RESULT"
