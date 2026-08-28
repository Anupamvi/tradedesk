import json
import os
from pathlib import Path

import pytest

from codexswing.secrets import (
    InsecureSecretFileError,
    SecretBundle,
    SecretConfigurationError,
    load_dotenv_file,
)


def _write_env(path: Path, content: str, mode: int = 0o600) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)
    return path


def test_secret_bundle_never_reprs_values(tmp_path: Path) -> None:
    env_path = _write_env(
        tmp_path / ".env",
        "ORATS_TOKEN='orats-value-123'\nSCHWAB_ACCESS_TOKEN=schwab-value-456\n",
    )
    bundle = SecretBundle.from_file(env_path)
    rendered = repr(bundle)
    assert "orats-value-123" not in rendered
    assert "schwab-value-456" not in rendered
    assert bundle.presence()["orats_token"] is True
    assert bundle.presence()["schwab_access_token"] is True
    assert bundle.redact("x orats-value-123 y") == "x ***REDACTED*** y"


def test_env_permissions_fail_closed(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env", "ORATS_TOKEN=value\n", mode=0o644)
    with pytest.raises(InsecureSecretFileError):
        load_dotenv_file(env_path)


def test_duplicate_env_key_is_rejected(tmp_path: Path) -> None:
    env_path = _write_env(tmp_path / ".env", "ORATS_TOKEN=one\nORATS_TOKEN=two\n")
    with pytest.raises(SecretConfigurationError):
        load_dotenv_file(env_path)


def test_dual_env_bundle_resolves_schwab_alias_and_relative_token_cache(tmp_path: Path) -> None:
    orats_env = _write_env(tmp_path / "orats.env", "ORATS_TOKEN=orats-value-123\n")
    schwab_dir = tmp_path / "schwab"
    schwab_dir.mkdir()
    schwab_env = _write_env(
        schwab_dir / ".env",
        "SCHWAB_API_KEY=app-key-value\n"
        "SCHWAB_APP_SECRET=app-secret-value\n"
        "SCHWAB_TOKEN_PATH=./tokens/token.json\n",
    )
    token_path = schwab_dir / "tokens" / "token.json"
    token_path.parent.mkdir()
    token_path.write_text(
        json.dumps(
            {
                "creation_timestamp": 1000,
                "token": {
                    "access_token": "access-value-456",
                    "refresh_token": "refresh-value-789",
                    "id_token": "id-value-101",
                    "expires_at": 2000,
                },
            }
        ),
        encoding="utf-8",
    )
    token_path.chmod(0o600)

    bundle = SecretBundle.from_files(orats_env, schwab_env)
    assert bundle.schwab_token_file == token_path.resolve()
    assert bundle.schwab_app_key == "app-key-value"
    assert bundle.schwab_access_token_expired(now=1500, skew_seconds=0) is False
    assert bundle.schwab_access_token_expired(now=2000, skew_seconds=0) is True
    rendered = repr(bundle)
    for value in (
        "orats-value-123",
        "app-key-value",
        "app-secret-value",
        "access-value-456",
        "refresh-value-789",
        "id-value-101",
    ):
        assert value not in rendered


def test_insecure_schwab_token_cache_fails_closed(tmp_path: Path) -> None:
    orats_env = _write_env(tmp_path / "orats.env", "ORATS_TOKEN=value\n")
    schwab_env = _write_env(tmp_path / "schwab.env", "SCHWAB_TOKEN_PATH=token.json\n")
    token_path = tmp_path / "token.json"
    token_path.write_text('{"token":{}}', encoding="utf-8")
    token_path.chmod(0o644)
    with pytest.raises(InsecureSecretFileError):
        SecretBundle.from_files(orats_env, schwab_env)
