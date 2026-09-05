#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_SRC="$ROOT/scripts/com.compoundcore.dashboard.plist"
PLIST_DEST="$HOME/Library/LaunchAgents/com.compoundcore.dashboard.plist"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "macOS only; on Linux/cloud use ensure-dashboard.sh or environment terminals." >&2
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$ROOT/var"
sed "s|__REPO_ROOT__|${ROOT}|g" "$PLIST_SRC" >"$PLIST_DEST"
launchctl bootout "gui/$(id -u)/com.compoundcore.dashboard" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST_DEST"
launchctl enable "gui/$(id -u)/com.compoundcore.dashboard"
launchctl kickstart -k "gui/$(id -u)/com.compoundcore.dashboard"
sleep 0.8
"$ROOT/scripts/ensure-dashboard.sh"
echo "Installed login service: ${PLIST_DEST}"
