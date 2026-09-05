#!/usr/bin/env bash
# Block non-Composer/Grok subagents (no fast). Deny known Other-Models subagent types.

set -euo pipefail
HOOK_INPUT=$(cat)

RESULT=$(HOOK_INPUT="$HOOK_INPUT" python3 <<'PY'
import json, os, re

d = json.loads(os.environ["HOOK_INPUT"])
sub = (d.get("subagent_model") or "").strip()
parent = (d.get("model") or "").strip()
model = sub or parent
typ = (d.get("subagent_type") or "").strip()
lower = model.lower()

# Hardcoded Other-Models subagent types — deny even if model field is empty/wrong.
blocked_types = {"computerUse", "mediaReview"}
if typ in blocked_types:
    out = {
        "permission": "deny",
        "user_message": (
            f"Blocked subagent ({typ}) — use cursor-ide-browser MCP or tests, not Other Models."
        ),
    }
    print(json.dumps(out))
    raise SystemExit

allow = bool(re.search(r"composer|cursor-grok|grok-4", lower))
if "fast" in lower:
    allow = False
# Explicit third-party model names
if re.search(r"claude|gpt-|gemini|sonnet|opus", lower):
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
