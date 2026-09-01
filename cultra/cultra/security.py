"""Secret handling for Cultra.

This module deliberately parses named keys instead of shell-sourcing env files.
It never logs, returns a printable wrapper for, or persists secrets except during
the explicit one-time bootstrap into Cultra's private .env.
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Iterable


class SecretError(RuntimeError):
    """Raised when a credential source violates the Cultra secret contract."""


ORATS_KEY = "ORATS_TOKEN"
SCHWAB_TOKEN_LINK = Path("/Users/anuppamvi/tradedesk/tokens/schwab_token.json")
CULTRA_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def _assert_private_file(path: Path) -> None:
    if not path.is_file():
        raise SecretError("credential file is missing or is not a regular file")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise SecretError("credential file must not be accessible by group or others")


def read_named_env_key(path: Path, key: str) -> str:
    """Read exactly one key from an env file without evaluating its contents."""

    path = Path(path)
    _assert_private_file(path)
    matches = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == key:
                clean = value.strip()
                if len(clean) >= 2 and clean[0] == clean[-1] and clean[0] in {"'", '"'}:
                    clean = clean[1:-1]
                matches.append(clean)
    if len(matches) != 1 or not matches[0]:
        raise SecretError("credential source must contain exactly one non-empty requested key")
    return matches[0]


def bootstrap_orats_env(source: Path) -> Path:
    """Atomically create a Cultra-owned env containing only ORATS_TOKEN."""

    token = read_named_env_key(Path(source), ORATS_KEY)
    destination = CULTRA_ENV_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".cultra-env-", dir=str(destination.parent))
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"{ORATS_KEY}={token}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, destination)
        os.chmod(destination, 0o600)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return destination


def validate_schwab_token_link(path: Path = SCHWAB_TOKEN_LINK) -> Path:
    """Resolve the allowlisted read-only Schwab token link and verify privacy."""

    supplied = Path(path)
    if supplied != SCHWAB_TOKEN_LINK:
        raise SecretError("Schwab token path is not allowlisted")
    resolved = supplied.resolve(strict=True)
    _assert_private_file(resolved)
    return resolved


def contains_secret(text: str, secrets: Iterable[str]) -> bool:
    """Return true if raw or common URL-encoded secret forms appear in text."""

    from urllib.parse import quote, quote_plus

    for secret in secrets:
        if not secret:
            continue
        if secret in text or quote(secret, safe="") in text or quote_plus(secret, safe="") in text:
            return True
    return False
