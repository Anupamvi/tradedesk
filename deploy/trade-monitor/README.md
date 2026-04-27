# Trade Monitor Cloud Runner

This deploys the hourly `ntfy` trade monitor on a persistent Linux VM.

The runner is intentionally simple:

- repo checkout: `/opt/tradedesk/current`
- Python venv: `/opt/tradedesk/venv`
- secret env file: `/etc/tradedesk/tradedesk.env`
- mutable Schwab token: `/var/lib/tradedesk/tokens/schwab_token.json`
- monitor state/output: `/opt/tradedesk/current/out/...`
- scheduler: `systemd` timer, hourly

Do not put Schwab secrets, ntfy topic, token JSON, or `out/` state in git.

## Step 1: VM Runtime

Use a small persistent Ubuntu VM or equivalent Linux host with an encrypted disk.
The VM must keep files between runs because Schwab token refresh and monitor
state are mutable.

On the VM:

```bash
sudo mkdir -p /opt/tradedesk
sudo chown "$USER":"$USER" /opt/tradedesk
git clone <YOUR_GITHUB_REPO_URL> /opt/tradedesk/current
cd /opt/tradedesk/current
sudo bash deploy/trade-monitor/install_systemd.sh
```

The install script creates the `tradedesk` service user, Python venv, systemd
unit files, `/etc/tradedesk/tradedesk.env`, and persistent token directory.
It does not start the timer until secrets are installed.

### GCP Shortcut

After `gcloud auth login` and `gcloud config set project YOUR_PROJECT_ID`,
run this locally from the repo root:

```bash
deploy/trade-monitor/gcp_create_vm.sh
```

Default GCP shape is `e2-micro` in `us-west1-b` with a `30GB` standard
persistent disk. Override with env vars:

```bash
VM_NAME=tradedesk-monitor ZONE=us-central1-a deploy/trade-monitor/gcp_create_vm.sh
```

## Step 2: Secrets

Edit the env file on the VM:

```bash
sudoedit /etc/tradedesk/tradedesk.env
```

Required values:

```env
UW_ROOT=/opt/tradedesk/current
SCHWAB_API_KEY=
SCHWAB_APP_SECRET=
SCHWAB_CALLBACK_URL=
SCHWAB_TOKEN_PATH=/var/lib/tradedesk/tokens/schwab_token.json
NTFY_TOPIC=
```

Then copy the existing Schwab token JSON to:

```text
/var/lib/tradedesk/tokens/schwab_token.json
```

Set permissions:

```bash
sudo chown -R tradedesk:tradedesk /var/lib/tradedesk
sudo chmod 700 /var/lib/tradedesk /var/lib/tradedesk/tokens
sudo chmod 600 /var/lib/tradedesk/tokens/schwab_token.json
sudo chmod 600 /etc/tradedesk/tradedesk.env
```

## Smoke Test

After secrets are present:

```bash
sudo bash -lc 'cd /opt/tradedesk/current; set -a; . /etc/tradedesk/tradedesk.env; set +a; sudo -E -u tradedesk -H /opt/tradedesk/venv/bin/python -m uwos.trade_monitor --test'
sudo systemctl start trade-monitor.service
```

The first command sends an ntfy test without exposing secrets to the
`tradedesk` user's shell profile. The second command uses the exact systemd
service path that the timer will run; it should either scan during US market
hours or print that the market is closed.

## Start Hourly Runner

```bash
sudo systemctl enable --now trade-monitor.timer
systemctl list-timers trade-monitor.timer
```

Useful logs:

```bash
journalctl -u trade-monitor.service -n 100 --no-pager
journalctl -u trade-monitor.timer -n 100 --no-pager
```

Manual single run:

```bash
sudo systemctl start trade-monitor.service
```

Stop:

```bash
sudo systemctl disable --now trade-monitor.timer
```
