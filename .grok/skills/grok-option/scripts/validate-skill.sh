#!/usr/bin/env bash
# Validate grok-option (and any skill dir passed as $1) against Agent Skill + this package's layout.
set -euo pipefail

SKILL_DIR="${1:-}"
if [[ -z "$SKILL_DIR" ]]; then
  SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
SKILL_DIR="$(cd "$SKILL_DIR" && pwd)"
SKILL_MD="$SKILL_DIR/SKILL.md"
DIR_NAME="$(basename "$SKILL_DIR")"

red() { printf 'ERROR: %s\n' "$1" >&2; }
ok() { printf 'OK: %s\n' "$1"; }
ERR=0

fail() { red "$1"; ERR=1; }

if [[ ! -f "$SKILL_MD" ]]; then
  echo "SKILL.md not found in $SKILL_DIR"
  exit 1
fi

if [[ "$(head -n1 "$SKILL_MD")" != "---" ]]; then
  fail "No YAML frontmatter found"
fi

# Extract frontmatter (between first two --- lines)
FM="$(awk 'BEGIN{c=0} /^---$/{c++; next} c==1{print} c==2{exit}' "$SKILL_MD")"
if [[ -z "$FM" ]]; then
  fail "Invalid frontmatter format"
  echo "Validation failed"
  exit 1
fi

yaml_name="$(printf '%s\n' "$FM" | awk -F': *' '/^name:/{print $2; exit}' | tr -d '"' | tr -d "'" | tr -d '[:space:]')"
# description is a single line after "description: "
yaml_desc="$(printf '%s\n' "$FM" | awk '/^description:/{sub(/^description:[[:space:]]*/,""); print; exit}')"

if [[ -z "$yaml_name" ]]; then
  fail "Missing name in frontmatter"
elif [[ "$yaml_name" != "$DIR_NAME" ]]; then
  fail "name '$yaml_name' must equal directory '$DIR_NAME'"
else
  ok "name matches directory ($yaml_name)"
fi

if [[ ! "$yaml_name" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]; then
  fail "name must be kebab-case"
fi

if [[ -z "$yaml_desc" ]]; then
  fail "Missing description in frontmatter"
else
  ok "description present (${#yaml_desc} chars)"
fi

if [[ "$yaml_desc" == *$'\n'* ]]; then
  fail "description must be one line"
fi
if [[ "$yaml_desc" == *'<'* || "$yaml_desc" == *'>'* ]]; then
  fail "description cannot contain angle brackets"
fi
if [[ "$yaml_desc" == *': '* ]]; then
  fail "description cannot contain colon-space"
fi
if [[ ${#yaml_desc} -gt 1024 ]]; then
  fail "description longer than 1024 characters"
fi

# Unexpected top-level keys (allow name, description, license, allowed-tools, metadata, compatibility)
while IFS= read -r line; do
  [[ -z "$line" ]] && continue
  [[ "$line" =~ ^[[:space:]] ]] && continue
  key="${line%%:*}"
  case "$key" in
    name|description|license|allowed-tools|metadata|compatibility) ;;
    *) fail "unexpected frontmatter key: $key" ;;
  esac
done <<< "$FM"

LINE_COUNT="$(wc -l < "$SKILL_MD" | tr -d ' ')"
if [[ "$LINE_COUNT" -ge 500 ]]; then
  fail "SKILL.md has $LINE_COUNT lines (max 499)"
else
  ok "SKILL.md line count $LINE_COUNT (<500)"
fi

for banned in README.md README CHANGELOG.md CHANGELOG; do
  if [[ -e "$SKILL_DIR/$banned" ]]; then
    fail "banned file present: $banned"
  fi
done
ok "no README or CHANGELOG"

# Required layout for grok-option
if [[ "$DIR_NAME" == "grok-option" ]]; then
  req=(
    references/assumption-audit.md
    references/regime-and-signals.md
    references/x-sentiment.md
    references/structures-and-pricing.md
    references/book-and-target.md
    references/expert-table.md
    references/workflow.md
    references/journal.md
    references/spike.md
    references/browser.md
    references/schwab.md
    assets/daily-card.md
    assets/report-style.md
    assets/playwright-mcp.json
  )
  for f in "${req[@]}"; do
    if [[ ! -f "$SKILL_DIR/$f" ]]; then
      fail "missing $f"
    else
      ok "found $f"
    fi
  done
  # Tape memo 20-40 non-empty lines
  if [[ -f "$SKILL_DIR/references/assumption-audit.md" ]]; then
    TAPE_N="$(awk '
      /^## Tape/{p=1; next}
      /^## /{if(p){exit}}
      p && NF{n++}
      END{print n+0}
    ' "$SKILL_DIR/references/assumption-audit.md")"
    if [[ "$TAPE_N" -lt 20 || "$TAPE_N" -gt 40 ]]; then
      fail "assumption-audit Tape+Evidence must be 20-40 non-empty lines (got $TAPE_N)"
    else
      ok "Tape+Evidence memo $TAPE_N lines"
    fi
  fi
fi

# Optional python yaml parse if PyYAML exists
if command -v python3 >/dev/null 2>&1; then
  python3 - "$SKILL_MD" <<'PY' || fail "python frontmatter parse"
import re, sys
from pathlib import Path
text = Path(sys.argv[1]).read_text()
m = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
if not m:
    print("Invalid frontmatter format", file=sys.stderr)
    sys.exit(1)
try:
    import yaml
    fm = yaml.safe_load(m.group(1))
    if not isinstance(fm, dict):
        print("Frontmatter must be a YAML dictionary", file=sys.stderr)
        sys.exit(1)
except ImportError:
    sys.exit(0)
except Exception as e:
    print(f"Invalid YAML in frontmatter: {e}", file=sys.stderr)
    sys.exit(1)
sys.exit(0)
PY
fi

if [[ "$ERR" -ne 0 ]]; then
  echo "Validation failed"
  exit 1
fi
echo "Skill is valid!"
exit 0
