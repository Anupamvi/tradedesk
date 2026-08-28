"""Immutable predeclarations that make strategy iteration countable."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from codexswing.clock import parse_timestamp
from codexswing.schemas.source import canonical_json
from codexswing.store.immutable import write_once_json


TRIAL_SCHEMA_VERSION = "codexswing.trial_declaration.v1"
HYPOTHESIS_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,79}$")


@dataclass(frozen=True)
class TrialDeclaration:
    hypothesis_id: str
    title: str
    declared_at_utc: str
    data_cutoff_utc: str
    evaluation_start_date: str
    posture: str
    objective: str
    universe: Sequence[str]
    sources: Sequence[str]
    feature_contract: Mapping[str, Any]
    target_contract: Mapping[str, Any]
    selection_rule: Mapping[str, Any]
    execution_contract: Mapping[str, Any]
    validation_contract: Mapping[str, Any]
    primary_metric: Mapping[str, Any]
    pass_condition: Sequence[str]
    fail_condition: Sequence[str]
    prohibited_changes: Sequence[str]
    schema_version: str = TRIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRIAL_SCHEMA_VERSION:
            raise ValueError("unsupported trial declaration schema")
        if not HYPOTHESIS_RE.fullmatch(self.hypothesis_id):
            raise ValueError("hypothesis_id must be an uppercase stable identifier")
        if self.posture not in {"PROSPECTIVE_SHADOW_RESEARCH", "EXPLORATORY_ONLY"}:
            raise ValueError("invalid trial posture")
        declared = parse_timestamp(self.declared_at_utc)
        cutoff = parse_timestamp(self.data_cutoff_utc)
        if cutoff > declared:
            raise ValueError("data cutoff cannot be after declaration")
        try:
            evaluation_start = date.fromisoformat(self.evaluation_start_date)
        except ValueError:
            raise ValueError("evaluation_start_date must be YYYY-MM-DD") from None
        if evaluation_start <= cutoff.date():
            raise ValueError("prospective evaluation must start after the data cutoff date")
        if not self.title.strip() or not self.objective.strip():
            raise ValueError("trial title and objective are required")
        if not self.universe or len(set(self.universe)) != len(self.universe):
            raise ValueError("trial universe must be non-empty and unique")
        if any(ticker != ticker.upper() for ticker in self.universe):
            raise ValueError("trial universe tickers must be uppercase")
        if not self.sources or not self.pass_condition or not self.fail_condition:
            raise ValueError("sources and pass/fail conditions are required")
        if not self.prohibited_changes:
            raise ValueError("prohibited_changes are required")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrialDeclaration":
        expected = {
            "schema_version",
            "hypothesis_id",
            "title",
            "declared_at_utc",
            "data_cutoff_utc",
            "evaluation_start_date",
            "posture",
            "objective",
            "universe",
            "sources",
            "feature_contract",
            "target_contract",
            "selection_rule",
            "execution_contract",
            "validation_contract",
            "primary_metric",
            "pass_condition",
            "fail_condition",
            "prohibited_changes",
        }
        if set(payload) != expected:
            missing = expected - set(payload)
            extra = set(payload) - expected
            raise ValueError(
                "trial declaration fields mismatch; missing={} extra={}".format(
                    sorted(missing), sorted(extra)
                )
            )
        return cls(**dict(payload))

    @classmethod
    def from_json_file(cls, path: Path) -> "TrialDeclaration":
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("trial declaration must be a JSON object")
        return cls.from_dict(payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "declared_at_utc": self.declared_at_utc,
            "data_cutoff_utc": self.data_cutoff_utc,
            "evaluation_start_date": self.evaluation_start_date,
            "posture": self.posture,
            "objective": self.objective,
            "universe": list(self.universe),
            "sources": list(self.sources),
            "feature_contract": dict(self.feature_contract),
            "target_contract": dict(self.target_contract),
            "selection_rule": dict(self.selection_rule),
            "execution_contract": dict(self.execution_contract),
            "validation_contract": dict(self.validation_contract),
            "primary_metric": dict(self.primary_metric),
            "pass_condition": list(self.pass_condition),
            "fail_condition": list(self.fail_condition),
            "prohibited_changes": list(self.prohibited_changes),
        }

    @property
    def trial_id(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


def register_trial(
    output_root: Path,
    declaration: TrialDeclaration,
    secret_values: Iterable[str] = (),
) -> Path:
    destination = (
        output_root.expanduser().resolve()
        / "trials"
        / declaration.hypothesis_id
        / declaration.trial_id
        / "declaration.json"
    )
    return write_once_json(destination, declaration.to_dict(), secret_values=secret_values)
