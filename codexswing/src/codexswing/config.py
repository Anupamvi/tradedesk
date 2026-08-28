"""Runtime configuration. Secret values never belong in this object."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union


DEFAULT_ENV_FILE = Path("/Users/anuppamvi/tradedesk/swingdesk/.env")
DEFAULT_SCHWAB_ENV_FILE = DEFAULT_ENV_FILE
DEFAULT_OUTPUT_ROOT = Path("/Users/anuppamvi/tradedesk/out/codexswing")


@dataclass(frozen=True)
class CodexSwingConfig:
    env_file: Path = DEFAULT_ENV_FILE
    schwab_env_file: Path = DEFAULT_SCHWAB_ENV_FILE
    output_root: Path = DEFAULT_OUTPUT_ROOT
    max_review_candidates: int = 12
    request_timeout_seconds: int = 30

    def validated(self) -> "CodexSwingConfig":
        env_file = self.env_file.expanduser().resolve()
        schwab_env_file = self.schwab_env_file.expanduser().resolve()
        output_root = self.output_root.expanduser().resolve()
        if self.max_review_candidates < 1 or self.max_review_candidates > 20:
            raise ValueError("max_review_candidates must be between 1 and 20")
        if self.request_timeout_seconds < 1 or self.request_timeout_seconds > 120:
            raise ValueError("request_timeout_seconds must be between 1 and 120")
        return CodexSwingConfig(
            env_file=env_file,
            schwab_env_file=schwab_env_file,
            output_root=output_root,
            max_review_candidates=self.max_review_candidates,
            request_timeout_seconds=self.request_timeout_seconds,
        )

    def public_dict(self) -> Dict[str, Union[str, int]]:
        return {
            "env_file": str(self.env_file),
            "schwab_env_file": str(self.schwab_env_file),
            "output_root": str(self.output_root),
            "max_review_candidates": self.max_review_candidates,
            "request_timeout_seconds": self.request_timeout_seconds,
        }
