#!/usr/bin/env python3
"""Schwab reauth the same way schwab-py --manual-auth builds the login URL.

Opens the Schwab authorize page (not 127.0.0.1). Reads the callback from
Chrome/Safari after MFA. Writes the token and syncs GCP. Never prints secrets.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from authlib.integrations.httpx_client import OAuth2Client
from schwab.auth import __fetch_and_register_token_from_redirect

ROOT = Path("/Users/anuppamvi/tradedesk")
load_dotenv(ROOT / ".env")
TOKEN = ROOT / "tokens" / "schwab_token.json"
UW_COPY = Path("/Users/anuppamvi/uw_root/tradedesk/tokens/schwab_token.json")


def _browser_urls() -> list[str]:
    scripts = [
        """tell application "Google Chrome"
            if (count of windows) is 0 then return ""
            set out to ""
            repeat with w in windows
                repeat with t in tabs of w
                    set out to out & (URL of t) & linefeed
                end repeat
            end repeat
            return out
        end tell""",
        """tell application "Safari"
            if (count of windows) is 0 then return ""
            set out to ""
            repeat with w in windows
                repeat with t in tabs of w
                    set out to out & (URL of t) & linefeed
                end repeat
            end repeat
            return out
        end tell""",
    ]
    found = []
    for src in scripts:
        try:
            raw = subprocess.check_output(
                ["osascript", "-e", src],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode("utf-8", "replace")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            continue
        for line in raw.splitlines():
            line = line.strip()
            if line:
                found.append(line)
    return found


def _callback_from_browser(callback_url: str) -> str:
    prefix = callback_url.rstrip("/")
    for url in _browser_urls():
        if url.startswith(prefix) and "code=" in url:
            return url
        # Chrome may show https://127.0.0.1:8080/?code= even if callback omitted port
        if "127.0.0.1" in url and "code=" in url:
            return url
    return ""


def _sync_copies() -> None:
    UW_COPY.parent.mkdir(parents=True, exist_ok=True)
    UW_COPY.write_bytes(TOKEN.read_bytes())
    os.chmod(UW_COPY, 0o600)
    env = os.environ.copy()
    env["SCHWAB_SYNC_FORCE"] = "1"
    env["PATH"] = str(Path.home() / "google-cloud-sdk" / "bin") + ":" + env.get("PATH", "")
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "schwab_sync_gcp.sh"), "push"],
        env=env,
        capture_output=True,
        text=True,
    )
    lines = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    print(lines[-1] if lines else ("SYNC_FAIL rc=%s" % r.returncode), flush=True)


def main() -> int:
    api_key = os.environ["SCHWAB_API_KEY"]
    app_secret = os.environ["SCHWAB_APP_SECRET"]
    callback_url = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8080")
    # Exact same client as schwab.auth.client_from_manual_flow
    oauth = OAuth2Client(api_key, redirect_uri=callback_url)
    authorization_url, _state = oauth.create_authorization_url(
        "https://api.schwabapi.com/v1/oauth/authorize"
    )
    host = "api.schwabapi.com" if "api.schwabapi.com" in authorization_url else "unknown"
    print("OPENING_SCHWAB_LOGIN", host, flush=True)
    print("Log in and Allow. Do not open 127.0.0.1 yourself.", flush=True)
    webbrowser.open(authorization_url)
    deadline = time.time() + 240
    redirected = ""
    while time.time() < deadline:
        redirected = _callback_from_browser(callback_url)
        if redirected:
            break
        time.sleep(0.4)
    if not redirected:
        print("TIMEOUT", flush=True)
        return 2
    print("GOT_CALLBACK", flush=True)
    TOKEN.parent.mkdir(parents=True, exist_ok=True)
    __fetch_and_register_token_from_redirect(
        oauth, redirected, api_key, app_secret, str(TOKEN), None, False, enforce_enums=True
    )
    os.chmod(TOKEN, 0o600)
    print("TOKEN_WRITTEN", flush=True)
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

    r = SchwabLiveDataService(SchwabAuthConfig.from_env()).connect().get_account_numbers()
    print("LOCAL", r.status_code, flush=True)
    _sync_copies()
    return 0 if r.status_code == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
