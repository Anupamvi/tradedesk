from __future__ import annotations

from copy import deepcopy
from typing import Any


PIPELINE_LOCK_DATE = "2026-05-21"
PIPELINE_LOCK_STATUS = "locked"

PIPELINE_NAME_V2 = "Codex Daily V2"
PIPELINE_VERSION_V2 = "v2.1"
PIPELINE_NAME_V3 = "Codex Daily V3"
PIPELINE_VERSION_V3 = "v3.0"
PIPELINE_NAME_V4 = "Codex Daily V4"
PIPELINE_VERSION_V4 = "v4.0"

PIPELINE_VERSION_LOCKS: dict[str, dict[str, Any]] = {
    "v2": {
        "pipeline_name": PIPELINE_NAME_V2,
        "pipeline_version": PIPELINE_VERSION_V2,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": PIPELINE_LOCK_DATE,
        "live_schwab_required_for_execute": True,
        "portfolio_state_required_for_execute": True,
        "gex_context": "confirmation framework records level_or_gex_protection; live pricing comes from Schwab chains",
    },
    "v3": {
        "pipeline_name": PIPELINE_NAME_V3,
        "pipeline_version": PIPELINE_VERSION_V3,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": PIPELINE_LOCK_DATE,
        "live_schwab_required_for_execute": True,
        "portfolio_state_required_for_execute": True,
        "gex_context": "liquidity-shift engine writes zero-DTE/index gamma context and Schwab chain snapshots",
    },
    "v4": {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V4,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": PIPELINE_LOCK_DATE,
        "live_schwab_required_for_execute": True,
        "portfolio_state_required_for_execute": True,
        "gex_context": "independent V4 EOD path carries liquidity-shift zero-DTE/index gamma context into regime and detailed artifacts",
    },
}


def pipeline_version_record(version_key: str) -> dict[str, Any]:
    return deepcopy(PIPELINE_VERSION_LOCKS[version_key])
