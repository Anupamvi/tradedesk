from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    """Return the UW trade desk root across Windows and macOS."""
    env_root = os.environ.get("UW_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()

    windows_root = Path("c:/uw_root")
    if windows_root.exists():
        return windows_root.resolve()

    return Path(__file__).resolve().parents[1]
