"""Load env from CODE/.env then tradedesk/.env. Never print values."""

import os
from pathlib import Path
from typing import Dict, Mapping, Optional

from groki_eq.config import CODE_DIR, TRADEDESK_ENV

ORATS_TOKEN_MISSING = (
    "ORATS_TOKEN missing. Put it in CODE/.env or export ORATS_TOKEN."
)


def parse_env_text(text: str) -> Dict[str, str]:
    out = {}
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
            out[key] = value
    return out


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    return parse_env_text(path.read_text(encoding="utf-8"))


def load_merged_env(code_dir: Optional[Path] = None) -> Dict[str, str]:
    """CODE/.env first; tradedesk/.env fills missing keys. Resolve relative token path."""
    merged = {}
    root = Path(code_dir) if code_dir is not None else CODE_DIR
    for path in (root / ".env", TRADEDESK_ENV):
        parsed = _read_env_file(path)
        if not parsed:
            continue
        for key, value in parsed.items():
            if key in merged:
                continue
            if key == "SCHWAB_TOKEN_PATH" and value and not Path(value).is_absolute():
                merged[key] = str((path.parent / value).resolve())
            else:
                merged[key] = value
    return merged


def _token_from_text(text: str, dotenv: bool = False) -> Optional[str]:
    raw_fallback = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if line.startswith("ORATS_TOKEN="):
            value = line.split("=", 1)[1].strip().strip("'").strip('"')
            return value or None
        if "=" in line:
            continue
        if raw_fallback is None:
            raw_fallback = line.strip("'").strip('"')
    if dotenv:
        return None
    return raw_fallback or None


def _read_token_file(path: Path, dotenv: bool = False) -> Optional[str]:
    if not path.is_file():
        return None
    return _token_from_text(path.read_text(encoding="utf-8"), dotenv=dotenv)


def load_orats_token(
    token_file: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
    code_dir: Optional[Path] = None,
) -> Optional[str]:
    """Load order: --orats-token-file, env ORATS_TOKEN, CODE/.env, tradedesk/.env."""
    if token_file:
        found = _read_token_file(Path(token_file), dotenv=False)
        if found:
            return found
    env = environ if environ is not None else os.environ
    from_env = (env.get("ORATS_TOKEN") or "").strip()
    if from_env:
        return from_env
    merged = load_merged_env(code_dir=code_dir)
    found = (merged.get("ORATS_TOKEN") or "").strip()
    return found or None


def load_schwab_env(code_dir: Optional[Path] = None) -> Dict[str, str]:
    return load_merged_env(code_dir=code_dir)


def schwab_credentials(code_dir: Optional[Path] = None) -> Optional[Dict[str, str]]:
    env = load_schwab_env(code_dir=code_dir)
    key = (env.get("SCHWAB_API_KEY") or "").strip()
    secret = (env.get("SCHWAB_APP_SECRET") or "").strip()
    token_path = (env.get("SCHWAB_TOKEN_PATH") or "").strip()
    if not key or not secret or not token_path:
        return None
    return {
        "api_key": key,
        "app_secret": secret,
        "token_path": token_path,
        "callback_url": (env.get("SCHWAB_CALLBACK_URL") or "").strip(),
    }
