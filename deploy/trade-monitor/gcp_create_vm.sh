#!/usr/bin/env bash
set -euo pipefail

VM_NAME="${VM_NAME:-tradedesk-monitor}"
ZONE="${ZONE:-us-west1-b}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-30GB}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-standard}"
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu-2404-lts-amd64}"
IMAGE_PROJECT="${IMAGE_PROJECT:-ubuntu-os-cloud}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
INSTALL_ROOT="${INSTALL_ROOT:-/opt/tradedesk}"
ARCHIVE="/tmp/tradedesk-$(git -C "$REPO_ROOT" rev-parse --short HEAD).tar.gz"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud is not installed. Install Google Cloud CLI first." >&2
  exit 1
fi

if [[ ! -f "$REPO_ROOT/uwos/trade_monitor.py" ]]; then
  echo "Run this from the tradedesk repo root, or set REPO_ROOT=/path/to/repo." >&2
  exit 1
fi

PROJECT_ID="$(gcloud config get-value project 2>/dev/null || true)"
if [[ -z "$PROJECT_ID" || "$PROJECT_ID" == "(unset)" ]]; then
  echo "No active GCP project. Run: gcloud config set project YOUR_PROJECT_ID" >&2
  exit 1
fi

echo "Using GCP project: $PROJECT_ID"
echo "Ensuring Compute Engine API is enabled..."
gcloud services enable compute.googleapis.com

if ! gcloud compute instances describe "$VM_NAME" --zone "$ZONE" >/dev/null 2>&1; then
  echo "Creating $VM_NAME in $ZONE..."
  create_args=(
    compute instances create "$VM_NAME"
    --zone "$ZONE"
    --machine-type "$MACHINE_TYPE"
    --image-family "$IMAGE_FAMILY"
    --image-project "$IMAGE_PROJECT"
    --boot-disk-size "$BOOT_DISK_SIZE"
    --boot-disk-type "$BOOT_DISK_TYPE"
    --maintenance-policy MIGRATE
    --provisioning-model STANDARD
    --scopes cloud-platform
    --metadata enable-oslogin=FALSE
    --quiet
  )
  if [[ -n "$SERVICE_ACCOUNT" ]]; then
    create_args+=(--service-account "$SERVICE_ACCOUNT")
  fi
  gcloud "${create_args[@]}"
else
  echo "VM $VM_NAME already exists in $ZONE; reusing it."
fi

echo "Packaging committed repo state..."
git -C "$REPO_ROOT" archive --format=tar.gz --output "$ARCHIVE" HEAD

echo "Copying repo archive to VM..."
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --command "sudo mkdir -p '$INSTALL_ROOT/current' && sudo chown -R \$USER:\$USER '$INSTALL_ROOT'"
gcloud compute scp "$ARCHIVE" "$VM_NAME:/tmp/tradedesk.tar.gz" --zone "$ZONE"

echo "Installing trade monitor runtime on VM..."
gcloud compute ssh "$VM_NAME" --zone "$ZONE" --command "
  set -euo pipefail
  rm -rf '$INSTALL_ROOT/current'
  mkdir -p '$INSTALL_ROOT/current'
  tar -xzf /tmp/tradedesk.tar.gz -C '$INSTALL_ROOT/current'
  cd '$INSTALL_ROOT/current'
  sudo bash deploy/trade-monitor/install_systemd.sh
"

cat <<MSG
GCP VM runtime is ready.

Next Step 2 on VM:
  gcloud compute ssh $VM_NAME --zone $ZONE
  sudoedit /etc/tradedesk/tradedesk.env
  sudo mkdir -p /var/lib/tradedesk/tokens
  sudo install -o tradedesk -g tradedesk -m 600 /path/to/schwab_token.json /var/lib/tradedesk/tokens/schwab_token.json

Then smoke test:
  sudo -u tradedesk -H /opt/tradedesk/venv/bin/python -m uwos.trade_monitor --test
  sudo -u tradedesk -H /opt/tradedesk/venv/bin/python -m uwos.trade_monitor

Start hourly timer:
  sudo systemctl enable --now trade-monitor.timer
MSG

