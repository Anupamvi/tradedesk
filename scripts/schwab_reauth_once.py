#!/usr/bin/env python3
"""Schwab login the same way schwab-py --manual-auth does: no local TLS, paste callback."""
from __future__ import annotations

import os
import sys
import time
import webbrowser
from pathlib import Path

from dotenv import load_dotenv
from authlib.integrations.httpx_client import OAuth2Client
from schwab.auth import __fetch_and_register_token_from_redirect

ROOT = Path("/Users/anuppamvi/tradedesk")
load_dotenv(ROOT / ".env")
PASTE = Path("/tmp/schwab_redirect_url.txt")
TOKEN = ROOT / "tokens" / "schwab_token.json"


def main() -> int:
    api_key = os.environ["SCHWAB_API_KEY"]
    app_secret = os.environ["SCHWAB_APP_SECRET"]
    callback_url = os.environ.get("SCHWAB_CALLBACK_URL", "https://127.0.0.1:8080")
    oauth = OAuth2Client(api_key, redirect_uri=callback_url)
    authorization_url, _state = oauth.create_authorization_url(
        "https://api.schwabapi.com/v1/oauth/authorize"
    )
    PASTE.unlink(missing_ok=True)
    print("LOGIN_OPENED", flush=True)
    webbrowser.open(authorization_url)
    print("WAITING_FOR_CALLBACK_FILE", flush=True)
    deadline = time.time() + 300
    url = ""
    while time.time() < deadline:
        if PASTE.is_file():
            url = PASTE.read_text(encoding="utf-8").strip()
            if "code=" in url:
                break
        time.sleep(0.15)
    else:
        print("TIMEOUT", flush=True)
        return 2
    print("GOT_CALLBACK", flush=True)
    TOKEN.parent.mkdir(parents=True, exist_ok=True)
    __fetch_and_register_token_from_redirect(
        oauth, url, api_key, app_secret, str(TOKEN), None, False, enforce_enums=True
    )
    print("TOKEN_WRITTEN", flush=True)
    os.chdir(ROOT)
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

    r = SchwabLiveDataService(SchwabAuthConfig.from_env()).connect().get_account_numbers()
    print("AUTH_OK", r.status_code, flush=True)
    PASTE.unlink(missing_ok=True)
    return 0 if r.ok else 1


if __name__ == "__main__":
    sys.exit(main())
