"""Strict, dependency-free `.env` loading and value redaction."""

from __future__ import annotations

import json
import math
import re
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class SecretConfigurationError(RuntimeError):
    pass


class InsecureSecretFileError(SecretConfigurationError):
    pass


def _strip_matching_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def load_dotenv_file(path: Path) -> Dict[str, str]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SecretConfigurationError("secret file is unavailable: {}".format(resolved))
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise InsecureSecretFileError(
            "secret file must not be group/world accessible: {} mode={:04o}".format(resolved, mode)
        )

    values: Dict[str, str] = {}
    for line_number, raw in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise SecretConfigurationError("invalid .env line {}".format(line_number))
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _KEY_RE.fullmatch(key):
            raise SecretConfigurationError("invalid .env key on line {}".format(line_number))
        if key in values:
            raise SecretConfigurationError("duplicate .env key: {}".format(key))
        values[key] = _strip_matching_quotes(raw_value.strip())
    return values


def _present(values: Mapping[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        value = values.get(key, "").strip()
        if value:
            return value
    return None


def _secure_json_object(path: Path) -> Mapping[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SecretConfigurationError("secret token file is unavailable: {}".format(resolved))
    mode = stat.S_IMODE(resolved.stat().st_mode)
    if mode & 0o077:
        raise InsecureSecretFileError(
            "secret token file must not be group/world accessible: {} mode={:04o}".format(
                resolved, mode
            )
        )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise SecretConfigurationError("secret token file is not valid JSON: {}".format(resolved)) from None
    if not isinstance(value, Mapping):
        raise SecretConfigurationError("secret token file must contain a JSON object")
    return value


def _finite_number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True)
class SecretBundle:
    source_file: Path
    schwab_source_file: Optional[Path] = None
    schwab_token_file: Optional[Path] = None
    schwab_access_token_expires_at: Optional[float] = None
    schwab_token_creation_timestamp: Optional[float] = None
    schwab_callback_url: Optional[str] = None
    orats_token: Optional[str] = field(default=None, repr=False)
    schwab_access_token: Optional[str] = field(default=None, repr=False)
    schwab_refresh_token: Optional[str] = field(default=None, repr=False)
    schwab_app_key: Optional[str] = field(default=None, repr=False)
    schwab_app_secret: Optional[str] = field(default=None, repr=False)
    schwab_account_hash: Optional[str] = field(default=None, repr=False)
    schwab_id_token: Optional[str] = field(default=None, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> "SecretBundle":
        values = load_dotenv_file(path)
        return cls(
            source_file=path.expanduser().resolve(),
            schwab_source_file=path.expanduser().resolve(),
            orats_token=_present(values, "ORATS_TOKEN"),
            schwab_access_token=_present(values, "SCHWAB_ACCESS_TOKEN"),
            schwab_refresh_token=_present(values, "SCHWAB_REFRESH_TOKEN"),
            schwab_app_key=_present(values, "SCHWAB_APP_KEY", "SCHWAB_API_KEY"),
            schwab_app_secret=_present(values, "SCHWAB_APP_SECRET"),
            schwab_account_hash=_present(values, "SCHWAB_ACCOUNT_HASH"),
            schwab_callback_url=_present(values, "SCHWAB_CALLBACK_URL"),
        )

    @classmethod
    def from_files(cls, orats_env_file: Path, schwab_env_file: Path) -> "SecretBundle":
        orats_path = orats_env_file.expanduser().resolve()
        schwab_path = schwab_env_file.expanduser().resolve()
        orats_values = load_dotenv_file(orats_path)
        schwab_values = orats_values if schwab_path == orats_path else load_dotenv_file(schwab_path)

        token_path_text = _present(schwab_values, "SCHWAB_TOKEN_PATH")
        token_path: Optional[Path] = None
        token_payload: Mapping[str, Any] = {}
        token_envelope: Mapping[str, Any] = {}
        if token_path_text:
            candidate = Path(token_path_text).expanduser()
            token_path = (candidate if candidate.is_absolute() else schwab_path.parent / candidate).resolve()
            token_envelope = _secure_json_object(token_path)
            nested = token_envelope.get("token")
            token_payload = nested if isinstance(nested, Mapping) else token_envelope

        return cls(
            source_file=orats_path,
            schwab_source_file=schwab_path,
            schwab_token_file=token_path,
            schwab_access_token_expires_at=_finite_number(token_payload.get("expires_at")),
            schwab_token_creation_timestamp=_finite_number(token_envelope.get("creation_timestamp")),
            schwab_callback_url=_present(schwab_values, "SCHWAB_CALLBACK_URL"),
            orats_token=_present(orats_values, "ORATS_TOKEN"),
            schwab_access_token=(
                _present(schwab_values, "SCHWAB_ACCESS_TOKEN")
                or (str(token_payload.get("access_token") or "").strip() or None)
            ),
            schwab_refresh_token=(
                _present(schwab_values, "SCHWAB_REFRESH_TOKEN")
                or (str(token_payload.get("refresh_token") or "").strip() or None)
            ),
            schwab_app_key=_present(schwab_values, "SCHWAB_APP_KEY", "SCHWAB_API_KEY"),
            schwab_app_secret=_present(schwab_values, "SCHWAB_APP_SECRET"),
            schwab_account_hash=_present(schwab_values, "SCHWAB_ACCOUNT_HASH"),
            schwab_id_token=str(token_payload.get("id_token") or "").strip() or None,
        )

    def schwab_access_token_expired(self, *, now: Optional[float] = None, skew_seconds: int = 60) -> bool:
        if not self.schwab_access_token:
            return True
        if self.schwab_access_token_expires_at is None:
            return False
        current = time.time() if now is None else float(now)
        return self.schwab_access_token_expires_at <= current + skew_seconds

    def values(self) -> Tuple[str, ...]:
        candidates = (
            self.orats_token,
            self.schwab_access_token,
            self.schwab_refresh_token,
            self.schwab_app_key,
            self.schwab_app_secret,
            self.schwab_account_hash,
            self.schwab_id_token,
        )
        return tuple(value for value in candidates if value)

    def presence(self) -> Dict[str, bool]:
        return {
            "orats_token": bool(self.orats_token),
            "schwab_access_token": bool(self.schwab_access_token),
            "schwab_refresh_token": bool(self.schwab_refresh_token),
            "schwab_app_key": bool(self.schwab_app_key),
            "schwab_app_secret": bool(self.schwab_app_secret),
            "schwab_account_hash": bool(self.schwab_account_hash),
            "schwab_token_file": bool(self.schwab_token_file),
            "schwab_access_token_expired": self.schwab_access_token_expired(),
        }

    def redact(self, text: str) -> str:
        redacted = text
        for value in sorted(self.values(), key=len, reverse=True):
            if len(value) >= 4:
                redacted = redacted.replace(value, "***REDACTED***")
        return redacted


def find_secret_leaks(text: str, secret_values: Iterable[str]) -> Tuple[str, ...]:
    matches = []
    for value in secret_values:
        if value and len(value) >= 4 and value in text:
            matches.append(value)
    return tuple(matches)
