"""Minimal Schwab market-data/account client for the Claude Pipeline.

Stdlib only. Owns its own token handling so this package has no dependency on
any other pipeline in the repository.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_ENV_PATH = Path("/Users/anuppamvi/uw_root/tradedesk/.env")
MARKETDATA_BASE = "https://api.schwabapi.com/marketdata/v1"
TRADER_BASE = "https://api.schwabapi.com/trader/v1"
TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"

# Schwab access tokens live 30 min; refresh a little early to avoid mid-call expiry.
_REFRESH_SKEW_SECONDS = 120


class SchwabAuthError(RuntimeError):
    """Raised when credentials or the stored token cannot produce a live session."""


def load_env(env_path: Path = DEFAULT_ENV_PATH) -> dict[str, str]:
    if not env_path.exists():
        raise SchwabAuthError(f"env file not found: {env_path}")
    out: dict[str, str] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip('"').strip("'")
    return out


@dataclass
class TokenStore:
    """Reads/writes the on-disk token, preserving the existing file schema."""

    path: Path

    def read(self) -> dict[str, Any]:
        if not self.path.exists():
            raise SchwabAuthError(f"token file not found: {self.path}")
        return json.loads(self.path.read_text())

    def write(self, payload: dict[str, Any]) -> None:
        backup = self.path.with_suffix(self.path.suffix + ".claude_pipeline.bak")
        if not backup.exists():
            shutil.copy2(self.path, backup)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.chmod(tmp, 0o600)
        os.replace(tmp, self.path)


class SchwabClient:
    def __init__(self, env: dict[str, str] | None = None, env_path: Path = DEFAULT_ENV_PATH):
        self._env = env if env is not None else load_env(env_path)
        missing = [k for k in ("SCHWAB_API_KEY", "SCHWAB_APP_SECRET", "SCHWAB_TOKEN_PATH") if not self._env.get(k)]
        if missing:
            raise SchwabAuthError(f"missing env keys: {', '.join(missing)}")
        raw_path = Path(self._env["SCHWAB_TOKEN_PATH"]).expanduser()
        if not raw_path.is_absolute():
            raw_path = (env_path.parent / raw_path).resolve()
        self._store = TokenStore(raw_path)
        self._payload = self._store.read()

    @property
    def token_path(self) -> Path:
        return self._store.path

    def _token(self) -> dict[str, Any]:
        return self._payload.get("token", self._payload)

    def _expires_at(self) -> float:
        token = self._token()
        if "expires_at" in token:
            return float(token["expires_at"])
        return float(self._payload.get("creation_timestamp", 0)) + float(token.get("expires_in", 0))

    def access_token(self, force_refresh: bool = False) -> str:
        if force_refresh or time.time() >= self._expires_at() - _REFRESH_SKEW_SECONDS:
            self._refresh()
        token = self._token().get("access_token")
        if not token:
            raise SchwabAuthError("no access_token available after refresh")
        return token

    def _refresh(self) -> None:
        refresh_token = self._token().get("refresh_token")
        if not refresh_token:
            raise SchwabAuthError("stored token has no refresh_token; manual re-auth required")
        basic = base64.b64encode(
            f"{self._env['SCHWAB_API_KEY']}:{self._env['SCHWAB_APP_SECRET']}".encode()
        ).decode()
        body = urllib.parse.urlencode(
            {"grant_type": "refresh_token", "refresh_token": refresh_token}
        ).encode()
        request = urllib.request.Request(
            TOKEN_URL,
            data=body,
            headers={
                "Authorization": f"Basic {basic}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                fresh = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise SchwabAuthError(
                f"token refresh failed ({exc.code}); refresh token may be expired — manual re-auth required"
            ) from None

        issued = time.time()
        fresh.setdefault("refresh_token", refresh_token)
        fresh["expires_at"] = issued + float(fresh.get("expires_in", 1800))
        self._payload["token"] = fresh
        self._payload["creation_timestamp"] = int(issued)
        self._store.write(self._payload)

    def _get(self, url: str, params: dict[str, Any] | None = None, _retried: bool = False) -> Any:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        full = f"{url}?{query}" if query else url
        request = urllib.request.Request(
            full, headers={"Authorization": f"Bearer {self.access_token()}", "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and not _retried:
                self.access_token(force_refresh=True)
                return self._get(url, params, _retried=True)
            raise SchwabAuthError(f"GET {url} failed with HTTP {exc.code}") from None

    def quotes(self, symbols: list[str]) -> dict[str, Any]:
        return self._get(f"{MARKETDATA_BASE}/quotes", {"symbols": ",".join(symbols)})

    def option_chain(
        self,
        symbol: str,
        strike_count: int = 10,
        contract_type: str = "ALL",
        from_date: str | None = None,
        to_date: str | None = None,
    ) -> dict[str, Any]:
        return self._get(
            f"{MARKETDATA_BASE}/chains",
            {
                "symbol": symbol,
                "contractType": contract_type,
                "strikeCount": strike_count,
                "includeUnderlyingQuote": "true",
                "fromDate": from_date,
                "toDate": to_date,
            },
        )

    def price_history(self, symbol: str, period_type: str = "year", period: int = 1,
                      frequency_type: str = "daily", frequency: int = 1) -> dict[str, Any]:
        return self._get(
            f"{MARKETDATA_BASE}/pricehistory",
            {
                "symbol": symbol,
                "periodType": period_type,
                "period": period,
                "frequencyType": frequency_type,
                "frequency": frequency,
            },
        )

    def accounts(self, include_positions: bool = True) -> Any:
        return self._get(f"{TRADER_BASE}/accounts", {"fields": "positions" if include_positions else None})
