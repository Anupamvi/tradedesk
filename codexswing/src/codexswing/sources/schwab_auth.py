"""Credential-only Schwab OAuth refresh with an atomic token-cache update.

This module is deliberately separate from the GET-only market-data client. It
can refresh OAuth credentials, but it exposes no brokerage or order endpoint.
"""

from __future__ import annotations

import base64
import json
import math
import os
import stat
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from codexswing.secrets import SecretBundle


SCHWAB_TOKEN_URL = "https://api.schwabapi.com/v1/oauth/token"


class SchwabTokenRefreshError(RuntimeError):
    pass


RefreshTransport = Callable[[str, Mapping[str, str], bytes], Mapping[str, Any]]


def _iso_epoch(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _read_token_envelope(path: Path) -> Dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SchwabTokenRefreshError("Schwab token cache is unavailable: {}".format(resolved))
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise SchwabTokenRefreshError(
            "Schwab token cache must be mode 0600 or stricter: {} mode={:04o}".format(
                resolved, mode
            )
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SchwabTokenRefreshError("Schwab token cache is not valid JSON") from None
    if not isinstance(value, dict):
        raise SchwabTokenRefreshError("Schwab token cache must contain a JSON object")
    return value


def _atomic_secure_json(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".{}.".format(resolved.name), suffix=".tmp", dir=str(resolved.parent)
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(resolved))
        os.chmod(resolved, 0o600)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


@dataclass(frozen=True)
class SchwabRefreshResult:
    token_file: Path
    refreshed_at_utc: str
    access_token_expires_at_utc: str
    expires_in_seconds: int
    refresh_token_rotated: bool

    def public_dict(self) -> Dict[str, Any]:
        return {
            "status": "SCHWAB_CREDENTIAL_REFRESHED",
            "token_file": str(self.token_file),
            "refreshed_at_utc": self.refreshed_at_utc,
            "access_token_expires_at_utc": self.access_token_expires_at_utc,
            "expires_in_seconds": self.expires_in_seconds,
            "refresh_token_rotated": self.refresh_token_rotated,
            "broker_order_authorized": False,
        }


class SchwabOAuthRefresher:
    def __init__(
        self,
        timeout_seconds: int = 30,
        transport: Optional[RefreshTransport] = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._default_transport

    def _default_transport(
        self, url: str, headers: Mapping[str, str], body: bytes
    ) -> Mapping[str, Any]:
        request = urllib.request.Request(
            url,
            data=body,
            headers=dict(headers),
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise SchwabTokenRefreshError(
                "Schwab OAuth refresh returned HTTP {}".format(exc.code)
            ) from None
        except urllib.error.URLError as exc:
            raise SchwabTokenRefreshError("Schwab OAuth refresh failed: {}".format(exc.reason)) from None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise SchwabTokenRefreshError("Schwab OAuth refresh returned invalid JSON") from None
        if not isinstance(payload, Mapping):
            raise SchwabTokenRefreshError("Schwab OAuth refresh returned an unexpected payload")
        return payload

    def refresh(self, secrets: SecretBundle, *, now: Optional[float] = None) -> SchwabRefreshResult:
        missing = [
            label
            for label, value in (
                ("app key", secrets.schwab_app_key),
                ("app secret", secrets.schwab_app_secret),
                ("refresh token", secrets.schwab_refresh_token),
                ("token file", secrets.schwab_token_file),
            )
            if not value
        ]
        if missing:
            raise SchwabTokenRefreshError(
                "Schwab OAuth refresh is unavailable; missing {}".format(", ".join(missing))
            )
        assert secrets.schwab_app_key is not None
        assert secrets.schwab_app_secret is not None
        assert secrets.schwab_refresh_token is not None
        assert secrets.schwab_token_file is not None

        envelope = _read_token_envelope(secrets.schwab_token_file)
        authorization = base64.b64encode(
            "{}:{}".format(secrets.schwab_app_key, secrets.schwab_app_secret).encode("utf-8")
        ).decode("ascii")
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": secrets.schwab_refresh_token,
            }
        ).encode("ascii")
        response = self._transport(
            SCHWAB_TOKEN_URL,
            {
                "Accept": "application/json",
                "Authorization": "Basic {}".format(authorization),
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "codexswing/0.2.0",
            },
            body,
        )
        access_token = str(response.get("access_token") or "").strip()
        if not access_token:
            raise SchwabTokenRefreshError("Schwab OAuth refresh response omitted access_token")
        try:
            expires_in_float = float(response.get("expires_in"))
        except (TypeError, ValueError):
            raise SchwabTokenRefreshError("Schwab OAuth refresh response has invalid expires_in") from None
        if not math.isfinite(expires_in_float) or expires_in_float <= 0:
            raise SchwabTokenRefreshError("Schwab OAuth refresh response has invalid expires_in")
        expires_in = int(expires_in_float)
        refreshed_at = time.time() if now is None else float(now)
        if not math.isfinite(refreshed_at) or refreshed_at <= 0:
            raise ValueError("now must be a positive finite epoch")

        existing_nested = envelope.get("token")
        token_payload: Dict[str, Any] = (
            dict(existing_nested) if isinstance(existing_nested, Mapping) else dict(envelope)
        )
        for key, value in response.items():
            token_payload[str(key)] = value
        if not str(token_payload.get("refresh_token") or "").strip():
            token_payload["refresh_token"] = secrets.schwab_refresh_token
        token_payload["expires_at"] = refreshed_at + expires_in
        updated_envelope = dict(envelope)
        updated_envelope["creation_timestamp"] = refreshed_at
        updated_envelope["token"] = token_payload
        if not isinstance(existing_nested, Mapping):
            for key in tuple(token_payload):
                if key in updated_envelope and key != "token":
                    updated_envelope.pop(key, None)
        _atomic_secure_json(secrets.schwab_token_file, updated_envelope)
        return SchwabRefreshResult(
            token_file=secrets.schwab_token_file,
            refreshed_at_utc=_iso_epoch(refreshed_at),
            access_token_expires_at_utc=_iso_epoch(refreshed_at + expires_in),
            expires_in_seconds=expires_in,
            refresh_token_rotated=(
                str(token_payload.get("refresh_token")) != secrets.schwab_refresh_token
            ),
        )
