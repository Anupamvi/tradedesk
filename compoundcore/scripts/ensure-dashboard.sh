#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${COMPOUNDCORE_DASHBOARD_HOST:-127.0.0.1}"
PORT="${COMPOUNDCORE_DASHBOARD_PORT:-8765}"
PIDFILE="${COMPOUNDCORE_DASHBOARD_PIDFILE:-$ROOT/var/dashboard.pid}"
LOG="${COMPOUNDCORE_DASHBOARD_LOG:-$ROOT/var/dashboard.log}"

mkdir -p "$ROOT/var"

if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
  echo "Compound Core dashboard already running on port ${PORT}"
  exit 0
fi

if [[ -f "$PIDFILE" ]]; then
  old_pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Compound Core dashboard already running (pid ${old_pid})"
    exit 0
  fi
fi

cd "$ROOT"
export PYTHONPATH=.
nohup python3 -m compoundcore dashboard --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
sleep 0.6

if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/" >/dev/null 2>&1; then
  echo "Compound Core dashboard http://127.0.0.1:${PORT}/"
  exit 0
fi

echo "Compound Core dashboard failed to start; see ${LOG}" >&2
exit 1
