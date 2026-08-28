"""Minimal dotenv handling with strict secret redaction.

Secret values may be used in request construction but are never returned in
manifests, cache keys, reports, error strings, or log messages.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Mapping, Optional


TOKEN_QUERY_RE = re.compile(r"([?&]token=)[^&\s]+", re.IGNORECASE)
AUTH_RE = re.compile(r"(authorization\s*[:=]\s*)[^,;\s]+", re.IGNORECASE)


def parse_env_text(text: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def read_env(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    return parse_env_text(path.read_text(encoding="utf-8"))


def load_env(
    project_root: Path,
    environ: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Environment values win; project `.env` fills missing values."""

    merged = dict(read_env(project_root / ".env"))
    source = environ if environ is not None else os.environ
    for key, value in source.items():
        if value:
            merged[key] = value
    return merged


def orats_token(project_root: Path, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    value = load_env(project_root, environ=environ).get("ORATS_TOKEN", "").strip()
    return value or None


def redact(text: str, *secrets: Optional[str]) -> str:
    result = TOKEN_QUERY_RE.sub(r"\1REDACTED", str(text or ""))
    result = AUTH_RE.sub(r"\1REDACTED", result)
    for secret in secrets:
        if secret:
            result = result.replace(secret, "REDACTED")
    return result

