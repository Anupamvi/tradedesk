#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${INSTALL_ROOT:-/opt/tradedesk}"
REPO_ROOT="${REPO_ROOT:-$PWD}"
SERVICE_USER="${SERVICE_USER:-tradedesk}"
ENV_DIR="${ENV_DIR:-/etc/tradedesk}"
STATE_DIR="${STATE_DIR:-/var/lib/tradedesk}"
VENV_DIR="${VENV_DIR:-$INSTALL_ROOT/venv}"

if [[ ! -f "$REPO_ROOT/uwos/trade_monitor.py" ]]; then
  echo "Run this from the tradedesk repo root, or set REPO_ROOT=/path/to/repo." >&2
  exit 1
fi

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run with sudo/root: sudo bash deploy/trade-monitor/install_systemd.sh" >&2
  exit 1
fi

if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

mkdir -p "$INSTALL_ROOT" "$ENV_DIR" "$STATE_DIR/tokens" "$REPO_ROOT/out/trade_analysis" "$REPO_ROOT/out/trade_ideas"
chown -R "$SERVICE_USER:$SERVICE_USER" "$STATE_DIR" "$REPO_ROOT/out"
chmod 700 "$STATE_DIR" "$STATE_DIR/tokens"

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$REPO_ROOT/deploy/trade-monitor/requirements.txt"

install -m 0644 "$REPO_ROOT/deploy/trade-monitor/systemd/trade-monitor.service" /etc/systemd/system/trade-monitor.service
install -m 0644 "$REPO_ROOT/deploy/trade-monitor/systemd/trade-monitor.timer" /etc/systemd/system/trade-monitor.timer

if [[ ! -f "$ENV_DIR/tradedesk.env" ]]; then
  install -m 0600 "$REPO_ROOT/deploy/trade-monitor/tradedesk.env.example" "$ENV_DIR/tradedesk.env"
fi

systemctl daemon-reload

cat <<MSG
Trade monitor runtime installed.

Next:
  1. Fill secrets in $ENV_DIR/tradedesk.env
  2. Copy Schwab token JSON to $STATE_DIR/tokens/schwab_token.json
  3. Run:
       sudo -u $SERVICE_USER -H $VENV_DIR/bin/python -m uwos.trade_monitor --test
       sudo -u $SERVICE_USER -H $VENV_DIR/bin/python -m uwos.trade_monitor
  4. Start hourly timer:
       sudo systemctl enable --now trade-monitor.timer

The timer was installed but not started.
MSG

