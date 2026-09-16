#!/usr/bin/env bash
# One Schwab refresh token, two machines. Reconcile by login time then access expiry.
# Usage: schwab_sync_gcp.sh [push|pull|reconcile]
# Never prints token JSON.
set -euo pipefail
export PATH="${HOME}/google-cloud-sdk/bin:/opt/homebrew/bin:/usr/local/bin:${PATH}"
ROOT="${TRADEDESK_ROOT:-/Users/anuppamvi/tradedesk}"
SRC="$ROOT/tokens/schwab_token.json"
VM="${SCHWAB_GCP_VM:-tradedesk-monitor}"
ZONE="${SCHWAB_GCP_ZONE:-us-west1-b}"
DEST="/var/lib/tradedesk/tokens/schwab_token.json"
UW="/Users/anuppamvi/uw_root/tradedesk/tokens/schwab_token.json"
MODE="${1:-reconcile}"
IAP=(--tunnel-through-iap)

meta() {
  python3 -c 'import json,sys; p=sys.argv[1]; d=json.load(open(p)); t=d.get("token") or {}; print(int(d.get("creation_timestamp") or 0), float(t.get("expires_at") or 0))' "$1"
}

gcloud_ok() { command -v gcloud >/dev/null 2>&1; }

ssh_iap() {
  gcloud compute ssh "$VM" --zone "$ZONE" "${IAP[@]}" --quiet --command "$1"
}

push_now() {
  [[ -f "$SRC" ]] || { echo "SYNC_FAIL no_local_token" >&2; return 1; }
  mkdir -p "$(dirname "$UW")"
  cp "$SRC" "$UW"
  chmod 600 "$UW" 2>/dev/null || true
  gcloud_ok || { echo "SYNC_FAIL gcloud_missing" >&2; return 1; }
  gcloud compute scp "${IAP[@]}" "$SRC" "$VM:/tmp/schwab_token.json" --zone "$ZONE" --quiet
  ssh_iap "sudo install -o tradedesk -g tradedesk -m 600 /tmp/schwab_token.json $DEST && rm -f /tmp/schwab_token.json"
  echo "SYNC_OK push"
}

pull_now() {
  gcloud_ok || { echo "SYNC_FAIL gcloud_missing" >&2; return 1; }
  ssh_iap "sudo cp $DEST /tmp/schwab_token.json && sudo chmod 644 /tmp/schwab_token.json"
  mkdir -p "$(dirname "$SRC")"
  gcloud compute scp "${IAP[@]}" "$VM:/tmp/schwab_token.json" "$SRC" --zone "$ZONE" --quiet
  ssh_iap "rm -f /tmp/schwab_token.json"
  chmod 600 "$SRC"
  mkdir -p "$(dirname "$UW")"
  cp "$SRC" "$UW"
  chmod 600 "$UW" 2>/dev/null || true
  echo "SYNC_OK pull"
}

remote_meta() {
  ssh_iap "sudo python3 -c \"import json; d=json.load(open('$DEST')); t=d.get('token') or {}; print(int(d.get('creation_timestamp') or 0), float(t.get('expires_at') or 0))\""
}

reconcile() {
  if ! gcloud_ok; then
    echo "SYNC_SKIP gcloud_missing"
    return 0
  fi
  if [[ ! -f "$SRC" ]]; then
    pull_now
    return
  fi
  local lm rm_
  lm="$(meta "$SRC")"
  if ! rm_="$(remote_meta)"; then
    echo "SYNC_SKIP remote_meta_fail"
    return 0
  fi
  local lc le rc re
  lc="$(echo "$lm" | awk '{print $1}')"
  le="$(echo "$lm" | awk '{print $2}')"
  rc="$(echo "$rm_" | awk '{print $1}')"
  re="$(echo "$rm_" | awk '{print $2}')"
  if [[ "$lc" -gt "$rc" ]]; then
    push_now
  elif [[ "$rc" -gt "$lc" ]]; then
    pull_now
  elif python3 -c "import sys; raise SystemExit(0 if float(sys.argv[1])>=float(sys.argv[2]) else 1)" "$le" "$re"; then
    push_now
  else
    pull_now
  fi
}

case "$MODE" in
  push) push_now ;;
  pull) pull_now ;;
  reconcile) reconcile ;;
  *) echo "usage: $0 push|pull|reconcile" >&2; exit 2 ;;
esac
