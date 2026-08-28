import json
from pathlib import Path

import pytest

from codexswing.research.trials import TrialDeclaration, register_trial


def _payload():
    return {
        "schema_version": "codexswing.trial_declaration.v1",
        "hypothesis_id": "TEST_HYPOTHESIS_V1",
        "title": "Prospective test",
        "declared_at_utc": "2026-08-27T23:40:13Z",
        "data_cutoff_utc": "2026-08-27T23:40:13Z",
        "evaluation_start_date": "2026-08-28",
        "posture": "PROSPECTIVE_SHADOW_RESEARCH",
        "objective": "Test a fixed prospective rule.",
        "universe": ["SPY", "AAPL"],
        "sources": ["SOURCE_A"],
        "feature_contract": {"feature": "fixed"},
        "target_contract": {"horizon": 5},
        "selection_rule": {"capacity": 1},
        "execution_contract": {"cost_bps": 10},
        "validation_contract": {"future_only": True},
        "primary_metric": {"name": "paired_mean"},
        "pass_condition": ["lower bound above zero"],
        "fail_condition": ["lower bound not above zero"],
        "prohibited_changes": ["no tuning"],
    }


def test_trial_registration_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    declaration = TrialDeclaration.from_dict(_payload())
    first = register_trial(tmp_path, declaration)
    second = register_trial(tmp_path, declaration)
    assert first == second
    assert declaration.trial_id in str(first)
    assert json.loads(first.read_text(encoding="utf-8"))["hypothesis_id"] == "TEST_HYPOTHESIS_V1"


def test_trial_must_start_after_its_data_cutoff() -> None:
    payload = _payload()
    payload["evaluation_start_date"] = "2026-08-27"
    with pytest.raises(ValueError, match="after the data cutoff"):
        TrialDeclaration.from_dict(payload)


def test_trial_rejects_unknown_fields() -> None:
    payload = _payload()
    payload["secret_tuning_knob"] = 1
    with pytest.raises(ValueError, match="fields mismatch"):
        TrialDeclaration.from_dict(payload)
