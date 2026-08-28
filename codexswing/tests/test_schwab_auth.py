import base64
import json
import stat
import urllib.parse
from pathlib import Path
from typing import Any, Mapping

from codexswing.secrets import SecretBundle
from codexswing.sources.schwab_auth import SCHWAB_TOKEN_URL, SchwabOAuthRefresher


def _bundle(tmp_path: Path) -> SecretBundle:
    token_path = tmp_path / "token.json"
    token_path.write_text(
        json.dumps(
            {
                "creation_timestamp": 900.0,
                "token": {
                    "access_token": "old-access-value",
                    "refresh_token": "refresh-value",
                    "expires_in": 1800,
                    "expires_at": 1000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    token_path.chmod(0o600)
    return SecretBundle(
        source_file=tmp_path / "orats.env",
        schwab_source_file=tmp_path / "schwab.env",
        schwab_token_file=token_path,
        schwab_access_token="old-access-value",
        schwab_refresh_token="refresh-value",
        schwab_app_key="app-key-value",
        schwab_app_secret="app-secret-value",
    )


def test_refresh_uses_oauth_contract_and_atomically_updates_secure_cache(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    calls = []

    def transport(url: str, headers: Mapping[str, str], body: bytes) -> Mapping[str, Any]:
        calls.append((url, dict(headers), body))
        assert url == SCHWAB_TOKEN_URL
        expected = base64.b64encode(b"app-key-value:app-secret-value").decode("ascii")
        assert headers["Authorization"] == "Basic {}".format(expected)
        assert urllib.parse.parse_qs(body.decode("ascii")) == {
            "grant_type": ["refresh_token"],
            "refresh_token": ["refresh-value"],
        }
        return {
            "access_token": "new-access-value",
            "expires_in": 1800,
            "token_type": "Bearer",
        }

    result = SchwabOAuthRefresher(transport=transport).refresh(bundle, now=2000.0)
    assert result.refreshed_at_utc == "1970-01-01T00:33:20Z"
    assert result.access_token_expires_at_utc == "1970-01-01T01:03:20Z"
    assert "new-access-value" not in repr(result)
    saved = json.loads(bundle.schwab_token_file.read_text(encoding="utf-8"))
    assert saved["creation_timestamp"] == 2000.0
    assert saved["token"]["access_token"] == "new-access-value"
    assert saved["token"]["refresh_token"] == "refresh-value"
    assert saved["token"]["expires_at"] == 3800.0
    assert stat.S_IMODE(bundle.schwab_token_file.stat().st_mode) == 0o600
    assert len(calls) == 1
