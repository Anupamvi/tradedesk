#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${COMPOUNDCORE_DASHBOARD_PORT:-8765}"
URL="http://127.0.0.1:${PORT}/"

if [[ "$(uname -s)" == "Darwin" ]]; then
  if [[ -d "/Applications/Safari.app" ]]; then
    open -a Safari "$URL" 2>/dev/null || open "$URL"
  else
    open "$URL"
  fi
else
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$URL" >/dev/null 2>&1 || true
  fi
fi

echo "Opened ${URL}"
