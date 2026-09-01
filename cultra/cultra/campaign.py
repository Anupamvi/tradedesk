"""Immutable offline freeze for the complete Cultra historical campaign."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .cohorts import load_point_in_time_universe
from .historical_events import load_historical_event_manifest
from .protocol import (
    build_campaign_freeze_receipt,
    historical_protocol_hash,
    load_historical_campaign_protocol,
)
from .prerequisites import HistoricalPrerequisiteError, load_historical_prerequisites
from .request_optimization import build_rotating_cohort_slices
from .requesting import RequestPlan
from .sessions import load_historical_session_calendar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = (PROJECT_ROOT / "out").resolve()
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:@\-]{1,96}$")


class CampaignFreezeError(ValueError):
    """The historical campaign cannot be frozen or reproduced safely."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _owned_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise CampaignFreezeError("%s must be Cultra-owned" % label) from exc
    if not resolved.is_file():
        raise CampaignFreezeError("%s is unavailable" % label)
    return resolved


def _private_json(path: Path, value: Any) -> Path:
    destination = Path(path).resolve()
    try:
        destination.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise CampaignFreezeError("campaign freeze output must remain inside Cultra/out") from exc
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    temporary = destination.with_name(".%s.tmp-%d" % (destination.name, os.getpid()))
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    try:
        with open(temporary, "xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


@dataclass(frozen=True)
class FrozenCampaign:
    payload: Mapping[str, Any]
    slices: Tuple[RequestPlan, ...]
    source_path: Optional[Path] = None

    @property
    def campaign_id(self) -> str:
        return str(self.payload["campaign_id"])


def build_historical_campaign_freeze(
    *,
    campaign_id: str,
    prerequisite_freeze_path: Path,
) -> FrozenCampaign:
    """Bind source-verified inputs and all 474 request identities offline."""

    if _SAFE_ID.fullmatch(str(campaign_id)) is None:
        raise CampaignFreezeError("campaign_id is invalid")
    protocol = load_historical_campaign_protocol()
    acquisition = protocol["acquisition"]
    try:
        prerequisites = load_historical_prerequisites(prerequisite_freeze_path)
    except HistoricalPrerequisiteError as exc:
        raise CampaignFreezeError(
            "historical prerequisite freeze is invalid: %s" % exc
        ) from exc
    universe_file = prerequisites.universe_path
    cohort_file = prerequisites.cohort_path
    sessions_file = prerequisites.session_calendar_path
    events_file = prerequisites.event_manifest_path
    universe = load_point_in_time_universe(universe_file)
    try:
        cohort_manifest = json.loads(cohort_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignFreezeError("cohort manifest is unreadable") from exc
    if not isinstance(cohort_manifest, Mapping):
        raise CampaignFreezeError("cohort manifest is malformed")
    if cohort_manifest.get("universe_fingerprint") != universe.fingerprint:
        raise CampaignFreezeError("cohort manifest does not match the point-in-time universe")
    session_calendar = load_historical_session_calendar(
        sessions_file,
        required_count=int(acquisition["historical_sessions"]),
    )
    sessions = session_calendar.dates
    events = load_historical_event_manifest(events_file)
    sampled = {
        str(ticker)
        for block in cohort_manifest.get("blocks", ())
        for ticker in block.get("tickers", ())
    }
    if not sampled:
        raise CampaignFreezeError("cohort manifest contains no sampled symbols")
    if not sampled.issubset(set(events.covered_tickers)):
        raise CampaignFreezeError("historical event manifest does not cover every sampled symbol")
    if (
        events.coverage_start > date.fromisoformat(sessions[0])
        or events.coverage_end < date.fromisoformat(sessions[-1])
    ):
        raise CampaignFreezeError("historical event coverage does not span the campaign")
    eligible_symbols = tuple(sorted({item.ticker for item in universe.members}))
    slices = build_rotating_cohort_slices(
        campaign_id=str(campaign_id),
        eligible_symbols=eligible_symbols,
        sessions=sessions,
        cohort_manifest=cohort_manifest,
        through_date=sessions[-1],
        slice_cap=int(acquisition["slice_hard_cap"]),
    )
    expected_attempts = int(acquisition["expected_cold_attempts"])
    if sum(item.logical_count for item in slices) != expected_attempts:
        raise CampaignFreezeError("exact request graph does not match the frozen estimate")
    if len(slices) != int(acquisition["expected_slices"]):
        raise CampaignFreezeError("campaign slice count does not match the frozen estimate")
    if [item.logical_count for item in slices] != list(
        acquisition["exact_slice_attempts"]
    ):
        raise CampaignFreezeError("campaign slice sizes do not match the frozen estimate")
    if any(item.worst_charged_attempts > 90 for item in slices):
        raise CampaignFreezeError("a historical slice can reach request 100")
    sessions_digest = hashlib.sha256(_canonical(list(sessions))).hexdigest()
    receipt = build_campaign_freeze_receipt(
        cohort_manifest=cohort_manifest,
        session_calendar_sha256=session_calendar.calendar_hash,
        event_manifest_sha256=events.manifest_hash,
        prerequisite_freeze_sha256=str(prerequisites.payload["freeze_hash"]),
    )
    request_ids = tuple(
        request.logical_request_id for plan in slices for request in plan.requests
    )
    if len(request_ids) != len(set(request_ids)):
        raise CampaignFreezeError("campaign request identities are duplicated")
    payload = {
        "schema": "cultra.historical-campaign-freeze.v2",
        "campaign_id": str(campaign_id),
        "protocol_hash": historical_protocol_hash(),
        "receipt": receipt,
        "inputs": {
            "prerequisite_freeze": {
                "path": str(prerequisites.source_path),
                "raw_sha256": _sha256(prerequisites.source_path),
                "freeze_hash": str(prerequisites.payload["freeze_hash"]),
            },
            "point_in_time_universe": {
                "path": str(universe_file),
                "raw_sha256": _sha256(universe_file),
                "fingerprint": universe.fingerprint,
            },
            "cohorts": {
                "path": str(cohort_file),
                "raw_sha256": _sha256(cohort_file),
                "freeze_hash": str(cohort_manifest["freeze_hash"]),
            },
            "sessions": {
                "path": str(sessions_file),
                "raw_sha256": _sha256(sessions_file),
                "calendar_hash": session_calendar.calendar_hash,
                "selected_sessions_sha256": sessions_digest,
                "count": len(sessions),
                "start": sessions[0],
                "end": sessions[-1],
            },
            "events": {
                "path": str(events_file),
                "raw_sha256": _sha256(events_file),
                "manifest_hash": events.manifest_hash,
                "covered_sampled_symbols": len(sampled),
            },
        },
        "request_campaign": {
            "expected_attempts": expected_attempts,
            "slice_count": len(slices),
            "slice_attempts": [item.logical_count for item in slices],
            "slice_plan_hashes": [item.plan_hash for item in slices],
            "request_ids_sha256": hashlib.sha256(_canonical(list(request_ids))).hexdigest(),
            "automatic_retries": 0,
            "attempt_100_within_any_slice_possible": False,
        },
        "network_attempted": False,
        "execution_authorized": False,
    }
    return FrozenCampaign(
        payload=dict(payload, freeze_hash=hashlib.sha256(_canonical(payload)).hexdigest()),
        slices=slices,
    )


def save_historical_campaign_freeze(
    output_dir: Path,
    campaign: FrozenCampaign,
) -> Path:
    root = Path(output_dir).expanduser().resolve()
    try:
        root.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise CampaignFreezeError("campaign freeze output must remain inside Cultra/out") from exc
    if root.exists():
        raise CampaignFreezeError("campaign freeze directory already exists")
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    for index, plan in enumerate(campaign.slices):
        _private_json(root / ("slice-%02d-plan.json" % index), plan.to_dict())
    manifest_path = _private_json(root / "campaign_freeze.json", campaign.payload)
    files = tuple(sorted(root.glob("slice-*-plan.json"))) + (manifest_path,)
    _private_json(
        root / "manifest.json",
        {
            "schema": "cultra.historical-campaign-freeze-manifest.v2",
            "campaign_id": campaign.campaign_id,
            "network_attempted": False,
            "files": [
                {
                    "path": item.name,
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
                for item in files
            ],
        },
    )
    return manifest_path


def load_historical_campaign_freeze(path: Path) -> FrozenCampaign:
    supplied = _owned_file(path, "campaign freeze")
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CampaignFreezeError("campaign freeze is unreadable") from exc
    if not isinstance(value, Mapping) or value.get("schema") != "cultra.historical-campaign-freeze.v2":
        raise CampaignFreezeError("campaign freeze schema is unsupported")
    supplied_hash = str(value.get("freeze_hash", ""))
    payload = dict(value)
    payload.pop("freeze_hash", None)
    if hashlib.sha256(_canonical(payload)).hexdigest() != supplied_hash:
        raise CampaignFreezeError("campaign freeze hash does not reconcile")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping):
        raise CampaignFreezeError("campaign freeze inputs are missing")
    for key in (
        "prerequisite_freeze",
        "point_in_time_universe",
        "cohorts",
        "sessions",
        "events",
    ):
        item = inputs.get(key)
        if not isinstance(item, Mapping):
            raise CampaignFreezeError("campaign freeze input is missing: %s" % key)
        source = _owned_file(Path(str(item.get("path", ""))), key)
        if _sha256(source) != item.get("raw_sha256"):
            raise CampaignFreezeError("campaign freeze source changed: %s" % key)
    rebuilt = build_historical_campaign_freeze(
        campaign_id=str(value["campaign_id"]),
        prerequisite_freeze_path=Path(str(inputs["prerequisite_freeze"]["path"])),
    )
    if rebuilt.payload != value:
        raise CampaignFreezeError("campaign freeze cannot be reproduced from its inputs")
    for index, plan in enumerate(rebuilt.slices):
        plan_path = supplied.parent / ("slice-%02d-plan.json" % index)
        try:
            saved = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CampaignFreezeError("saved campaign slice plan is unavailable") from exc
        if saved != plan.to_dict():
            raise CampaignFreezeError("saved campaign slice plan has drifted")
    return FrozenCampaign(rebuilt.payload, rebuilt.slices, supplied)


__all__ = [
    "CampaignFreezeError",
    "FrozenCampaign",
    "build_historical_campaign_freeze",
    "load_historical_campaign_freeze",
    "save_historical_campaign_freeze",
]
