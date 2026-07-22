from __future__ import annotations

from copy import deepcopy
from typing import Any


PIPELINE_LOCK_DATE = "2026-05-21"
PIPELINE_LOCK_STATUS = "locked"
EXECUTION_CONFIDENCE_RELEASE_ID = "exec-confidence-20260612-143405"
EXECUTION_INTEGRITY_RELEASE_ID = "v4.2-integrity-20260713"
EXECUTION_CONFIDENCE_RELEASE_DATE = "2026-06-12"
EXECUTION_CONFIDENCE_RELEASE_TIMESTAMP = "2026-06-12T14:34:05-07:00"
V3_PROFIT_INTEGRITY_RELEASE_ID = "profit-integrity-20260719"
V3_PROFIT_INTEGRITY_RELEASE_DATE = "2026-07-19"
V3_PROFIT_INTEGRITY_RELEASE_TIMESTAMP = "2026-07-19T00:00:00-07:00"
EXECUTION_INTEGRITY_RELEASE_DATE = "2026-07-13"
V4_EXPECTANCY_RELEASE_ID = "expectancy-safe-entry-20260716"
V4_EXPECTANCY_RELEASE_DATE = "2026-07-16"
V4_EXPECTANCY_RELEASE_TIMESTAMP = "2026-07-16T00:00:00-07:00"
V4_CONFIDENCE_RELEASE_ID = "confidence-calibrated-20260719"
V4_CONFIDENCE_RELEASE_DATE = "2026-07-19"
V4_CONFIDENCE_RELEASE_TIMESTAMP = "2026-07-19T00:00:00-07:00"
V4_MEDIUM_DEBIT_RELEASE_ID = "medium-debit-sleeve-20260719"
V4_MEDIUM_DEBIT_RELEASE_DATE = "2026-07-19"
V4_MEDIUM_DEBIT_RELEASE_TIMESTAMP = "2026-07-19T00:00:00-07:00"
V4_WALK_FORWARD_RELEASE_ID = "walk-forward-confidence-capacity-20260719"
V4_WALK_FORWARD_RELEASE_DATE = "2026-07-19"
V4_WALK_FORWARD_RELEASE_TIMESTAMP = "2026-07-19T00:00:00-07:00"
V4_POLICY_BASE_CONFIDENCE_RELEASE_ID = "policy-base-confidence-20260719"
V4_POLICY_BASE_CONFIDENCE_RELEASE_DATE = "2026-07-19"
V4_POLICY_BASE_CONFIDENCE_RELEASE_TIMESTAMP = "2026-07-19T00:00:00-07:00"
V4_HIERARCHICAL_EVIDENCE_RELEASE_ID = "hierarchical-evidence-20260720"
V4_HIERARCHICAL_EVIDENCE_RELEASE_DATE = "2026-07-20"
V4_HIERARCHICAL_EVIDENCE_RELEASE_TIMESTAMP = "2026-07-20T00:00:00-07:00"
V4_EMPIRICAL_PAYOFF_RELEASE_ID = "empirical-payoff-walkforward-20260720"
V4_EMPIRICAL_PAYOFF_RELEASE_DATE = "2026-07-20"
V4_EMPIRICAL_PAYOFF_RELEASE_TIMESTAMP = "2026-07-20T00:00:00-07:00"
V4_VALIDATED_PAYOFF_DISPLAY_RELEASE_ID = "validated-payoff-display-and-snapshot-replay-20260720"
V4_VALIDATED_PAYOFF_DISPLAY_RELEASE_DATE = "2026-07-20"
V4_VALIDATED_PAYOFF_DISPLAY_RELEASE_TIMESTAMP = "2026-07-20T00:00:00-07:00"
V4_SNAPSHOT_VALIDATION_RELEASE_ID = "snapshot-validation-integrity-20260720"
V4_SNAPSHOT_VALIDATION_RELEASE_DATE = "2026-07-20"
V4_SNAPSHOT_VALIDATION_RELEASE_TIMESTAMP = "2026-07-20T00:00:00-07:00"
V4_STRUCTURE_PAYOFF_RELEASE_ID = "structure-aware-payoff-20260720"
V4_STRUCTURE_PAYOFF_RELEASE_DATE = "2026-07-20"
V4_STRUCTURE_PAYOFF_RELEASE_TIMESTAMP = "2026-07-20T00:00:00-07:00"
V4_CREDIT_BOOK_ALLOCATION_RELEASE_ID = "correlation-aware-credit-book-20260721"
V4_CREDIT_BOOK_ALLOCATION_RELEASE_DATE = "2026-07-21"
V4_CREDIT_BOOK_ALLOCATION_RELEASE_TIMESTAMP = "2026-07-21T00:00:00-07:00"

PIPELINE_NAME_V2 = "Codex Daily V2"
PIPELINE_VERSION_V2 = "v2.1"
PIPELINE_NAME_V3 = "Codex Daily V3"
PIPELINE_VERSION_V3_PREVIOUS = f"v3.1-{EXECUTION_CONFIDENCE_RELEASE_ID}"
PIPELINE_VERSION_V3 = f"v3.2-{V3_PROFIT_INTEGRITY_RELEASE_ID}"
PIPELINE_NAME_V4 = "Codex Daily V4"
PIPELINE_VERSION_V45 = f"v4.5-{V4_MEDIUM_DEBIT_RELEASE_ID}"
PIPELINE_VERSION_V46 = f"v4.6-{V4_WALK_FORWARD_RELEASE_ID}"
PIPELINE_VERSION_V47 = f"v4.7-{V4_POLICY_BASE_CONFIDENCE_RELEASE_ID}"
PIPELINE_VERSION_V471 = f"v4.7.1-{V4_HIERARCHICAL_EVIDENCE_RELEASE_ID}"
PIPELINE_VERSION_V48 = f"v4.8-{V4_EMPIRICAL_PAYOFF_RELEASE_ID}"
PIPELINE_VERSION_V481 = f"v4.8.1-{V4_VALIDATED_PAYOFF_DISPLAY_RELEASE_ID}"
PIPELINE_VERSION_V482 = f"v4.8.2-{V4_SNAPSHOT_VALIDATION_RELEASE_ID}"
PIPELINE_VERSION_V49 = f"v4.9-{V4_STRUCTURE_PAYOFF_RELEASE_ID}"
PIPELINE_VERSION_V4 = f"v4.10-{V4_CREDIT_BOOK_ALLOCATION_RELEASE_ID}"
PIPELINE_VERSION_V41 = f"v4.1-{EXECUTION_CONFIDENCE_RELEASE_ID}"
PIPELINE_VERSION_V42 = EXECUTION_INTEGRITY_RELEASE_ID
PIPELINE_VERSION_V43 = f"v4.3-{V4_EXPECTANCY_RELEASE_ID}"
PIPELINE_VERSION_V44 = f"v4.4-{V4_CONFIDENCE_RELEASE_ID}"

PREVIOUS_PIPELINE_VERSION_LOCKS: dict[str, dict[str, Any]] = {
    "v3.0": {
        "pipeline_name": PIPELINE_NAME_V3,
        "pipeline_version": "v3.0",
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": PIPELINE_LOCK_DATE,
        "superseded_by": PIPELINE_VERSION_V3_PREVIOUS,
    },
    PIPELINE_VERSION_V3_PREVIOUS: {
        "pipeline_name": PIPELINE_NAME_V3,
        "pipeline_version": PIPELINE_VERSION_V3_PREVIOUS,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": EXECUTION_CONFIDENCE_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V3,
    },
    "v4.0": {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": "v4.0",
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": PIPELINE_LOCK_DATE,
        "superseded_by": PIPELINE_VERSION_V41,
    },
    PIPELINE_VERSION_V41: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V41,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": EXECUTION_CONFIDENCE_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V42,
    },
    PIPELINE_VERSION_V42: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V42,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": EXECUTION_INTEGRITY_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V43,
    },
    PIPELINE_VERSION_V43: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V43,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_EXPECTANCY_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V44,
    },
    PIPELINE_VERSION_V44: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V44,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_CONFIDENCE_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V45,
    },
    PIPELINE_VERSION_V45: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V45,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_MEDIUM_DEBIT_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V46,
    },
    PIPELINE_VERSION_V46: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V46,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_WALK_FORWARD_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V47,
    },
    PIPELINE_VERSION_V47: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V47,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_POLICY_BASE_CONFIDENCE_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V471,
    },
    PIPELINE_VERSION_V471: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V471,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_HIERARCHICAL_EVIDENCE_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V48,
    },
    PIPELINE_VERSION_V48: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V48,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_EMPIRICAL_PAYOFF_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V481,
    },
    PIPELINE_VERSION_V481: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V481,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_VALIDATED_PAYOFF_DISPLAY_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V482,
    },
    PIPELINE_VERSION_V482: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V482,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_SNAPSHOT_VALIDATION_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V49,
    },
    PIPELINE_VERSION_V49: {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V49,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_STRUCTURE_PAYOFF_RELEASE_DATE,
        "superseded_by": PIPELINE_VERSION_V4,
    },
}

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
        "locked_on": V3_PROFIT_INTEGRITY_RELEASE_DATE,
        "released_at": V3_PROFIT_INTEGRITY_RELEASE_TIMESTAMP,
        "release_id": V3_PROFIT_INTEGRITY_RELEASE_ID,
        "supersedes": ["v3.0", PIPELINE_VERSION_V3_PREVIOUS],
        "live_schwab_required_for_execute": True,
        "portfolio_state_required_for_execute": True,
        "gex_context": "liquidity-shift engine writes zero-DTE/index gamma context and Schwab chain snapshots",
    },
    "v4": {
        "pipeline_name": PIPELINE_NAME_V4,
        "pipeline_version": PIPELINE_VERSION_V4,
        "lock_status": PIPELINE_LOCK_STATUS,
        "locked_on": V4_CREDIT_BOOK_ALLOCATION_RELEASE_DATE,
        "released_at": V4_CREDIT_BOOK_ALLOCATION_RELEASE_TIMESTAMP,
        "release_id": V4_CREDIT_BOOK_ALLOCATION_RELEASE_ID,
        "supersedes": [
            "v4.0",
            PIPELINE_VERSION_V41,
            PIPELINE_VERSION_V42,
            PIPELINE_VERSION_V43,
            PIPELINE_VERSION_V44,
            PIPELINE_VERSION_V45,
            PIPELINE_VERSION_V46,
            PIPELINE_VERSION_V47,
            PIPELINE_VERSION_V471,
            PIPELINE_VERSION_V48,
            PIPELINE_VERSION_V481,
            PIPELINE_VERSION_V482,
            PIPELINE_VERSION_V49,
        ],
        "live_schwab_required_for_execute": True,
        "portfolio_state_required_for_execute": True,
        "gex_context": "independent V4 EOD path carries liquidity-shift zero-DTE/index gamma context into regime and detailed artifacts",
    },
}


def pipeline_version_record(version_key: str) -> dict[str, Any]:
    return deepcopy(PIPELINE_VERSION_LOCKS[version_key])
