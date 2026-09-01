"""Source-bound, network-free historical prerequisite preparation.

The historical campaign must not begin from hand-authored normalized manifests
whose provenance hashes are merely well formed.  This module accepts three
strict Cultra-owned raw source bundles, validates their point-in-time and
coverage claims, derives the four rotating cohorts, and binds every normalized
artifact back to the exact raw bytes that produced it.

No transport, credential, cache, or provider client is imported here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .artifacts import ArtifactError, assert_secret_free, assert_secret_free_bytes
from .cohorts import (
    PointInTimeMember,
    PointInTimeUniverse,
    freeze_rotating_cohorts,
)
from .historical_events import (
    EVENT_TYPES,
    HistoricalEventManifest,
    HistoricalEventRecord,
    event_manifest_payload,
)
from .sessions import (
    HistoricalSessionCalendar,
    MARKET_TIMEZONE,
    MarketSession,
    session_calendar_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = (PROJECT_ROOT / "out").resolve()
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.:@\-]{1,96}$")
_COVERAGE = (
    "US_LISTED_SECURITY_UNDERLYINGS_WITH_MIN_1000_DAILY_CBOE_OPTIONS_VOLUME_"
    "ACROSS_2_CBOE_VENUES"
)
_EVENT_ATTESTATION = "COMPLETE_FOR_COVERED_TICKERS_AND_EVENT_TYPES"
_SOURCE_ARTIFACT_FIELDS = {
    "path",
    "role",
    "source_uri",
    "media_type",
    "size_bytes",
    "sha256",
}
_SOURCE_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class HistoricalPrerequisiteError(ValueError):
    """A historical input source or its frozen receipt is not auditable."""


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
    supplied = Path(path).expanduser().resolve()
    try:
        supplied.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise HistoricalPrerequisiteError("%s must be Cultra-owned" % label) from exc
    if not supplied.is_file():
        raise HistoricalPrerequisiteError("%s is unavailable" % label)
    return supplied


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    supplied = _owned_file(path, label)
    try:
        value = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoricalPrerequisiteError("%s is unreadable" % label) from exc
    if not isinstance(value, Mapping):
        raise HistoricalPrerequisiteError("%s must be a JSON object" % label)
    try:
        assert_secret_free(value)
    except ArtifactError as exc:
        raise HistoricalPrerequisiteError("%s contains credential-shaped data" % label) from exc
    return value


def _date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise HistoricalPrerequisiteError("%s must use YYYY-MM-DD" % label) from exc


def _timestamp(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalPrerequisiteError("%s must be an ISO timestamp" % label) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HistoricalPrerequisiteError("%s must be timezone-aware" % label)
    return parsed


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_identity(value: Mapping[str, Any], label: str) -> Tuple[str, str, datetime]:
    provider = str(value.get("provider", "")).strip()
    source_uri = str(value.get("source_uri", "")).strip()
    if not provider or not source_uri:
        raise HistoricalPrerequisiteError("%s provider and source URI are required" % label)
    if "ORATS" in provider.upper() or "ORATS" in source_uri.upper():
        raise HistoricalPrerequisiteError("%s must be independent of ORATS" % label)
    retrieved_at = _timestamp(value.get("retrieved_at"), "%s retrieved_at" % label)
    return provider, source_uri, retrieved_at


def _verify_source_artifacts(
    value: Mapping[str, Any],
    *,
    bundle_path: Path,
    label: str,
) -> Tuple[str, ...]:
    """Require a raw bundle to bind the exact preserved provider bytes."""

    raw_artifacts = value.get("source_artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise HistoricalPrerequisiteError("%s source artifacts are missing" % label)
    paths = []
    roles = []
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping) or set(raw) != _SOURCE_ARTIFACT_FIELDS:
            raise HistoricalPrerequisiteError("%s source artifact is malformed" % label)
        relative = Path(str(raw.get("path", "")))
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise HistoricalPrerequisiteError(
                "%s source artifact path must be project-relative" % label
            )
        role = str(raw.get("role", "")).strip()
        source_uri = str(raw.get("source_uri", "")).strip()
        media_type = str(raw.get("media_type", "")).strip()
        digest = str(raw.get("sha256", ""))
        try:
            expected_size = int(raw.get("size_bytes", -1))
        except (TypeError, ValueError) as exc:
            raise HistoricalPrerequisiteError(
                "%s source artifact size is malformed" % label
            ) from exc
        if (
            _SAFE_ID.fullmatch(role) is None
            or not source_uri
            or not media_type
            or expected_size < 0
            or _SOURCE_SHA256.fullmatch(digest) is None
        ):
            raise HistoricalPrerequisiteError(
                "%s source artifact identity is malformed" % label
            )
        if "ORATS" in role.upper() or "ORATS" in source_uri.upper():
            raise HistoricalPrerequisiteError(
                "%s source artifact must be independent of ORATS" % label
            )
        candidate = PROJECT_ROOT / relative
        if candidate.is_symlink():
            raise HistoricalPrerequisiteError(
                "%s source artifact cannot be a symlink" % label
            )
        try:
            artifact = candidate.resolve(strict=True)
            artifact.relative_to(PROJECT_ROOT)
        except (OSError, ValueError) as exc:
            raise HistoricalPrerequisiteError(
                "%s source artifact is unavailable" % label
            ) from exc
        if not artifact.is_file() or artifact == bundle_path:
            raise HistoricalPrerequisiteError(
                "%s source artifact must be a distinct regular file" % label
            )
        payload = artifact.read_bytes()
        try:
            assert_secret_free_bytes(payload, path=relative.as_posix())
        except ArtifactError as exc:
            raise HistoricalPrerequisiteError(
                "%s source artifact contains credential-shaped data" % label
            ) from exc
        if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != digest:
            raise HistoricalPrerequisiteError("%s source artifact changed" % label)
        paths.append(relative.as_posix())
        roles.append(role)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise HistoricalPrerequisiteError(
            "%s source artifacts must be sorted and unique" % label
        )
    return tuple(roles)


def _private_json(path: Path, value: Any) -> Path:
    destination = Path(path).expanduser().resolve()
    try:
        destination.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise HistoricalPrerequisiteError(
            "historical prerequisite output must remain inside Cultra/out"
        ) from exc
    if destination.exists():
        raise HistoricalPrerequisiteError("historical prerequisite artifact already exists")
    try:
        assert_secret_free(value)
    except ArtifactError as exc:
        raise HistoricalPrerequisiteError(
            "historical prerequisite artifact contains credential-shaped data"
        ) from exc
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    temporary = destination.with_name(".%s.tmp-%d" % (destination.name, os.getpid()))
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode(
        "utf-8"
    ) + b"\n"
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
class PreparedHistoricalPrerequisites:
    input_set_id: str
    universe_source_path: Path
    session_source_path: Path
    event_source_path: Path
    universe_payload: Mapping[str, Any]
    session_payload: Mapping[str, Any]
    event_payload: Mapping[str, Any]
    cohort_payload: Mapping[str, Any]
    sampled_symbols: Tuple[str, ...]
    selection_dates: Tuple[str, ...]


@dataclass(frozen=True)
class FrozenHistoricalPrerequisites:
    payload: Mapping[str, Any]
    source_path: Path
    universe_path: Path
    session_calendar_path: Path
    event_manifest_path: Path
    cohort_path: Path

    @property
    def input_set_id(self) -> str:
        return str(self.payload["input_set_id"])


def _parse_session_source(
    path: Path,
) -> Tuple[Mapping[str, Any], Tuple[MarketSession, ...], datetime]:
    raw = _load_json(path, "market-session source")
    allowed = {
        "schema",
        "provider",
        "source_uri",
        "retrieved_at",
        "exchange",
        "timezone",
        "complete",
        "source_artifacts",
        "sessions",
    }
    if set(raw) != allowed or raw.get("schema") != "cultra.market-session-source.v2":
        raise HistoricalPrerequisiteError(
            "market-session source schema or fields are unsupported"
        )
    _verify_source_artifacts(
        raw,
        bundle_path=path,
        label="market-session source",
    )
    provider, source_uri, retrieved_at = _source_identity(raw, "market-session source")
    if raw.get("exchange") != "XNYS" or raw.get("timezone") != MARKET_TIMEZONE:
        raise HistoricalPrerequisiteError("market-session source is not the frozen XNYS calendar")
    if raw.get("complete") is not True:
        raise HistoricalPrerequisiteError("market-session source lacks completeness attestation")
    rows = raw.get("sessions")
    if not isinstance(rows, list) or len(rows) != 450:
        raise HistoricalPrerequisiteError("market-session source must contain exactly 450 sessions")
    sessions = []
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"session_date", "close_at"}:
            raise HistoricalPrerequisiteError("market-session record contains unfrozen fields")
        session_date = _date(row["session_date"], "session_date")
        close_at = _timestamp(row["close_at"], "session close_at")
        if close_at > retrieved_at:
            raise HistoricalPrerequisiteError("market-session source contains a future close")
        session = MarketSession(session_date=session_date, close_at=close_at)
        sessions.append(session)
        normalized.append(
            {"session_date": session_date.isoformat(), "close_at": close_at.isoformat()}
        )
    ordered_dates = tuple(item.session_date for item in sessions)
    if ordered_dates != tuple(sorted(set(ordered_dates))):
        raise HistoricalPrerequisiteError("market-session source must be sorted and unique")
    payload = session_calendar_payload(
        provider=provider,
        source_uri=source_uri,
        source_sha256=_sha256(path),
        sessions=normalized,
    )
    HistoricalSessionCalendar(
        provider=provider,
        source_uri=source_uri,
        source_sha256=str(payload["source_sha256"]),
        timezone=MARKET_TIMEZONE,
        sessions=tuple(sessions),
        calendar_hash=str(payload["calendar_hash"]),
    )
    return payload, tuple(sessions), retrieved_at


def _parse_universe_source(
    path: Path,
    *,
    selection_dates: Sequence[date],
) -> Tuple[Mapping[str, Any], PointInTimeUniverse]:
    raw = _load_json(path, "point-in-time universe source")
    allowed = {
        "schema",
        "provider",
        "source_uri",
        "retrieved_at",
        "universe_id",
        "coverage",
        "point_in_time",
        "survivorship_free",
        "source_artifacts",
        "snapshots",
    }
    if set(raw) != allowed or raw.get("schema") != (
        "cultra.point-in-time-universe-source.v2"
    ):
        raise HistoricalPrerequisiteError(
            "point-in-time universe source schema or fields are unsupported"
        )
    source_roles = set(_verify_source_artifacts(
        raw,
        bundle_path=path,
        label="point-in-time universe source",
    ))
    provider, source_uri, _ = _source_identity(raw, "point-in-time universe source")
    universe_id = str(raw.get("universe_id", "")).strip()
    if not universe_id or raw.get("coverage") != _COVERAGE:
        raise HistoricalPrerequisiteError("point-in-time universe source scope is incomplete")
    if raw.get("point_in_time") is not True or raw.get("survivorship_free") is not True:
        raise HistoricalPrerequisiteError(
            "point-in-time universe source lacks leakage and survivorship attestations"
        )
    snapshots = raw.get("snapshots")
    if not isinstance(snapshots, list):
        raise HistoricalPrerequisiteError("point-in-time universe snapshots are missing")
    expected_dates = tuple(selection_dates)
    by_date: Dict[date, Mapping[str, Any]] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, Mapping) or set(snapshot) != {"observed_at", "members"}:
            raise HistoricalPrerequisiteError("point-in-time universe snapshot is malformed")
        observed_at = _date(snapshot["observed_at"], "universe observed_at")
        if observed_at in by_date:
            raise HistoricalPrerequisiteError("point-in-time universe snapshot is duplicated")
        by_date[observed_at] = snapshot
    if tuple(sorted(by_date)) != tuple(sorted(expected_dates)):
        raise HistoricalPrerequisiteError(
            "point-in-time universe must contain only the four exact cohort selection dates"
        )
    members = []
    member_rows = []
    allowed_member = {
        "ticker",
        "asset_type",
        "optionable",
        "sampling_stratum",
        "liquidity_rank",
        "classification_status",
        "classification_source_roles",
    }
    for observed_at in expected_dates:
        rows = by_date[observed_at].get("members")
        if not isinstance(rows, list):
            raise HistoricalPrerequisiteError("point-in-time universe members are missing")
        seen = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != allowed_member:
                raise HistoricalPrerequisiteError(
                    "point-in-time universe member contains unfrozen fields"
                )
            asset_type = str(row["asset_type"]).strip().upper()
            classification_status = str(row["classification_status"]).strip()
            raw_classification_roles = row["classification_source_roles"]
            if not isinstance(raw_classification_roles, list):
                raise HistoricalPrerequisiteError(
                    "point-in-time classification source roles are malformed"
                )
            classification_roles = tuple(str(item).strip() for item in raw_classification_roles)
            if (
                classification_roles != tuple(sorted(set(classification_roles)))
                or not set(classification_roles).issubset(source_roles)
            ):
                raise HistoricalPrerequisiteError(
                    "point-in-time classification source roles are unbound"
                )
            if asset_type in {"STOCK", "ETF"}:
                if classification_status != "VERIFIED_POINT_IN_TIME" or not classification_roles:
                    raise HistoricalPrerequisiteError(
                        "resolved asset type lacks point-in-time source evidence"
                    )
            elif asset_type == "INELIGIBLE_OTHER_SECURITY":
                if (
                    classification_status != "VERIFIED_POINT_IN_TIME_INELIGIBLE"
                    or not classification_roles
                ):
                    raise HistoricalPrerequisiteError(
                        "ineligible security lacks point-in-time source evidence"
                    )
            elif asset_type == "UNRESOLVED_STOCK_OR_ETP":
                if classification_status != "UNRESOLVED" or classification_roles:
                    raise HistoricalPrerequisiteError(
                        "unresolved asset type carries false classification evidence"
                    )
            else:
                raise HistoricalPrerequisiteError("point-in-time asset type is unsupported")
            member = PointInTimeMember(
                ticker=str(row["ticker"]),
                asset_type=asset_type,
                eligible_from=observed_at,
                eligible_through=observed_at,
                observed_at=observed_at,
                optionable=row["optionable"],
                sampling_stratum=str(row["sampling_stratum"]),
                liquidity_rank=row["liquidity_rank"],
            )
            if member.ticker in seen:
                raise HistoricalPrerequisiteError(
                    "point-in-time universe ticker is duplicated within a snapshot"
                )
            seen.add(member.ticker)
            members.append(member)
            member_rows.append(
                {
                    "ticker": member.ticker,
                    "asset_type": member.asset_type,
                    "eligible_from": observed_at.isoformat(),
                    "eligible_through": observed_at.isoformat(),
                    "observed_at": observed_at.isoformat(),
                    "optionable": member.optionable,
                    "sampling_stratum": member.sampling_stratum,
                    "liquidity_rank": member.liquidity_rank,
                }
            )
    members.sort(
        key=lambda item: (
            item.observed_at,
            item.sampling_stratum,
            item.liquidity_rank,
            item.ticker,
        )
    )
    member_rows.sort(
        key=lambda item: (
            item["observed_at"],
            item["sampling_stratum"],
            item["liquidity_rank"],
            item["ticker"],
        )
    )
    universe = PointInTimeUniverse(
        universe_id=universe_id,
        provider=provider,
        source_uri=source_uri,
        source_sha256=_sha256(path),
        coverage=_COVERAGE,
        members=tuple(members),
    )
    payload = {
        "schema": "cultra.point-in-time-universe.v1",
        "universe_id": universe.universe_id,
        "provider": universe.provider,
        "source_uri": universe.source_uri,
        "source_sha256": universe.source_sha256,
        "coverage": universe.coverage,
        "members": member_rows,
    }
    return payload, universe


def _parse_event_source(
    path: Path,
    *,
    campaign_start: date,
    campaign_end: date,
    sampled_symbols: Sequence[str],
    sampled_stock_windows: Mapping[str, Tuple[date, date]],
) -> Mapping[str, Any]:
    raw = _load_json(path, "historical event source")
    allowed = {
        "schema",
        "provider",
        "source_uri",
        "retrieved_at",
        "coverage_start",
        "coverage_end",
        "covered_tickers",
        "complete_event_types",
        "point_in_time_revisions",
        "coverage_attestation",
        "source_artifacts",
        "records",
    }
    if set(raw) != allowed or raw.get("schema") != "cultra.historical-event-source.v2":
        raise HistoricalPrerequisiteError(
            "historical event source schema or fields are unsupported"
        )
    _verify_source_artifacts(
        raw,
        bundle_path=path,
        label="historical event source",
    )
    provider, source_uri, retrieved_at = _source_identity(raw, "historical event source")
    coverage_start = _date(raw["coverage_start"], "event coverage_start")
    coverage_end = _date(raw["coverage_end"], "event coverage_end")
    if coverage_start > campaign_start or coverage_end < campaign_end:
        raise HistoricalPrerequisiteError("historical event source does not span the campaign")
    if tuple(raw.get("complete_event_types", ())) != EVENT_TYPES:
        raise HistoricalPrerequisiteError("historical event source does not cover every event type")
    if raw.get("point_in_time_revisions") is not True:
        raise HistoricalPrerequisiteError("historical event source lacks point-in-time revisions")
    if raw.get("coverage_attestation") != _EVENT_ATTESTATION:
        raise HistoricalPrerequisiteError("historical event source lacks coverage attestation")
    covered = tuple(
        sorted(set(str(item).strip().upper() for item in raw.get("covered_tickers", ())))
    )
    if not set(sampled_symbols).issubset(set(covered)):
        raise HistoricalPrerequisiteError(
            "historical event source does not cover every sampled symbol"
        )
    rows = raw.get("records")
    if not isinstance(rows, list):
        raise HistoricalPrerequisiteError("historical event records are missing")
    allowed_record = {
        "ticker",
        "event_type",
        "effective_date",
        "observed_at",
        "available_at",
        "source_event_id",
        "status",
        "cash_amount",
        "split_ratio",
        "adjustment_reference",
    }
    records = []
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != allowed_record:
            raise HistoricalPrerequisiteError("historical event record contains unfrozen fields")
        observed_at = _timestamp(row["observed_at"], "event observed_at")
        available_at = _timestamp(row["available_at"], "event available_at")
        if observed_at > retrieved_at or available_at > retrieved_at:
            raise HistoricalPrerequisiteError("historical event source contains a future revision")
        record = HistoricalEventRecord(
            ticker=str(row["ticker"]),
            event_type=str(row["event_type"]),
            effective_date=_date(row["effective_date"], "event effective_date"),
            observed_at=observed_at,
            available_at=available_at,
            source_event_id=str(row["source_event_id"]),
            status=str(row["status"]),
            cash_amount=row["cash_amount"],
            split_ratio=row["split_ratio"],
            adjustment_reference=row["adjustment_reference"],
        )
        records.append(record)
        normalized.append(
            {
                "ticker": record.ticker,
                "event_type": record.event_type,
                "effective_date": record.effective_date.isoformat(),
                "observed_at": _utc_text(record.observed_at),
                "available_at": _utc_text(record.available_at),
                "source_event_id": record.source_event_id,
                "status": record.status,
                "cash_amount": record.cash_amount,
                "split_ratio": record.split_ratio,
                "adjustment_reference": record.adjustment_reference,
            }
        )
    normalized.sort(
        key=lambda item: (
            item["ticker"],
            item["effective_date"],
            item["event_type"],
            item["available_at"],
            item["source_event_id"],
        )
    )
    missing_earnings = sorted(
        ticker
        for ticker, window in sampled_stock_windows.items()
        if not any(
            item.ticker == ticker
            and item.event_type == "EARNINGS"
            and item.status != "CANCELLED"
            and window[0] <= item.effective_date <= window[1]
            for item in records
        )
    )
    if missing_earnings:
        raise HistoricalPrerequisiteError(
            "historical event source has no earnings evidence for sampled stocks: %s"
            % ",".join(missing_earnings)
        )
    payload = event_manifest_payload(
        provider=provider,
        source_uri=source_uri,
        source_sha256=_sha256(path),
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        covered_tickers=covered,
        records=normalized,
    )
    HistoricalEventManifest(
        provider=provider,
        source_uri=source_uri,
        source_sha256=str(payload["source_sha256"]),
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        covered_tickers=tuple(payload["covered_tickers"]),
        complete_event_types=tuple(payload["complete_event_types"]),
        records=tuple(records),
        point_in_time_revisions=True,
        manifest_hash=str(payload["manifest_hash"]),
    )
    return payload


def prepare_historical_prerequisites(
    *,
    input_set_id: str,
    universe_source_path: Path,
    session_source_path: Path,
    event_source_path: Path,
) -> PreparedHistoricalPrerequisites:
    """Validate raw source bundles and derive every normalized prerequisite."""

    if _SAFE_ID.fullmatch(str(input_set_id)) is None:
        raise HistoricalPrerequisiteError("input_set_id is invalid")
    universe_source = _owned_file(universe_source_path, "point-in-time universe source")
    session_source = _owned_file(session_source_path, "market-session source")
    event_source = _owned_file(event_source_path, "historical event source")
    session_payload, market_sessions, _ = _parse_session_source(session_source)
    session_dates = tuple(item.session_date for item in market_sessions)
    selection_dates = (
        session_dates[0],
        session_dates[120],
        session_dates[240],
        session_dates[360],
    )
    universe_payload, universe = _parse_universe_source(
        universe_source,
        selection_dates=selection_dates,
    )
    cohorts = freeze_rotating_cohorts(universe, session_dates)
    sampled = tuple(
        str(ticker)
        for block in cohorts["blocks"]
        for ticker in block["tickers"]
    )
    member_by_snapshot = {
        (item.observed_at.isoformat(), item.ticker): item
        for item in universe.members
    }
    sampled_stock_windows = {}
    for block in cohorts["blocks"]:
        for ticker in block["tickers"]:
            member = member_by_snapshot[(str(block["selection_date"]), str(ticker))]
            if member.asset_type == "STOCK":
                sampled_stock_windows[str(ticker)] = (
                    _date(block["block_start"], "cohort block_start"),
                    _date(block["block_end"], "cohort block_end"),
                )
    event_payload = _parse_event_source(
        event_source,
        campaign_start=session_dates[0],
        campaign_end=session_dates[-1],
        sampled_symbols=sampled,
        sampled_stock_windows=sampled_stock_windows,
    )
    return PreparedHistoricalPrerequisites(
        input_set_id=str(input_set_id),
        universe_source_path=universe_source,
        session_source_path=session_source,
        event_source_path=event_source,
        universe_payload=universe_payload,
        session_payload=session_payload,
        event_payload=event_payload,
        cohort_payload=cohorts,
        sampled_symbols=sampled,
        selection_dates=tuple(item.isoformat() for item in selection_dates),
    )


def load_point_in_time_universe_source(
    path: Path,
    *,
    selection_dates: Sequence[date],
) -> PointInTimeUniverse:
    """Validate a V2 raw universe source and return its normalized domain."""

    _, universe = _parse_universe_source(path, selection_dates=selection_dates)
    return universe


def save_historical_prerequisites(
    output_dir: Path,
    prepared: PreparedHistoricalPrerequisites,
) -> Path:
    """Save one immutable prerequisite bundle and source-binding receipt."""

    root = Path(output_dir).expanduser().resolve()
    try:
        root.relative_to(OUT_ROOT)
    except ValueError as exc:
        raise HistoricalPrerequisiteError(
            "historical prerequisite output must remain inside Cultra/out"
        ) from exc
    if root.exists():
        raise HistoricalPrerequisiteError("historical prerequisite directory already exists")
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    universe_path = _private_json(root / "point_in_time_universe.json", prepared.universe_payload)
    sessions_path = _private_json(root / "session_calendar.json", prepared.session_payload)
    events_path = _private_json(root / "historical_events.json", prepared.event_payload)
    cohorts_path = _private_json(root / "rotating_cohorts.json", prepared.cohort_payload)
    source_inputs = {
        "universe": {
            "path": str(prepared.universe_source_path),
            "sha256": _sha256(prepared.universe_source_path),
            "schema": "cultra.point-in-time-universe-source.v2",
        },
        "sessions": {
            "path": str(prepared.session_source_path),
            "sha256": _sha256(prepared.session_source_path),
            "schema": "cultra.market-session-source.v2",
        },
        "events": {
            "path": str(prepared.event_source_path),
            "sha256": _sha256(prepared.event_source_path),
            "schema": "cultra.historical-event-source.v2",
        },
    }
    normalized_paths = {
        "point_in_time_universe": universe_path,
        "sessions": sessions_path,
        "events": events_path,
        "cohorts": cohorts_path,
    }
    payload = {
        "schema": "cultra.historical-prerequisite-freeze.v1",
        "input_set_id": prepared.input_set_id,
        "network_attempted": False,
        "source_inputs": source_inputs,
        "normalized_inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in normalized_paths.items()
        },
        "selection_dates": list(prepared.selection_dates),
        "sampled_symbols": list(prepared.sampled_symbols),
        "sampled_symbol_count": len(prepared.sampled_symbols),
        "raw_sources_hash_bound": True,
        "orats_source_used": False,
    }
    freeze = dict(payload, freeze_hash=hashlib.sha256(_canonical(payload)).hexdigest())
    freeze_path = _private_json(root / "prerequisite_freeze.json", freeze)
    artifact_paths = tuple(normalized_paths.values()) + (freeze_path,)
    _private_json(
        root / "manifest.json",
        {
            "schema": "cultra.historical-prerequisite-freeze-manifest.v1",
            "input_set_id": prepared.input_set_id,
            "network_attempted": False,
            "files": [
                {
                    "path": item.name,
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
                for item in artifact_paths
            ],
        },
    )
    return freeze_path


def load_historical_prerequisites(path: Path) -> FrozenHistoricalPrerequisites:
    """Rebuild a prerequisite bundle from raw sources and reject any drift."""

    supplied = _owned_file(path, "historical prerequisite freeze")
    value = _load_json(supplied, "historical prerequisite freeze")
    if value.get("schema") != "cultra.historical-prerequisite-freeze.v1":
        raise HistoricalPrerequisiteError("historical prerequisite freeze schema is unsupported")
    allowed_root = {
        "schema",
        "input_set_id",
        "network_attempted",
        "source_inputs",
        "normalized_inputs",
        "selection_dates",
        "sampled_symbols",
        "sampled_symbol_count",
        "raw_sources_hash_bound",
        "orats_source_used",
        "freeze_hash",
    }
    if set(value) != allowed_root:
        raise HistoricalPrerequisiteError(
            "historical prerequisite freeze contains unfrozen fields"
        )
    supplied_hash = str(value.get("freeze_hash", ""))
    payload = dict(value)
    payload.pop("freeze_hash", None)
    if hashlib.sha256(_canonical(payload)).hexdigest() != supplied_hash:
        raise HistoricalPrerequisiteError("historical prerequisite freeze hash does not reconcile")
    if (
        value.get("network_attempted") is not False
        or value.get("raw_sources_hash_bound") is not True
        or value.get("orats_source_used") is not False
    ):
        raise HistoricalPrerequisiteError(
            "historical prerequisite freeze violates the offline source boundary"
        )
    sources = value.get("source_inputs")
    normalized = value.get("normalized_inputs")
    if not isinstance(sources, Mapping) or not isinstance(normalized, Mapping):
        raise HistoricalPrerequisiteError("historical prerequisite freeze inputs are missing")
    if set(sources) != {"universe", "sessions", "events"} or set(normalized) != {
        "point_in_time_universe",
        "sessions",
        "events",
        "cohorts",
    }:
        raise HistoricalPrerequisiteError("historical prerequisite input set is incomplete")
    source_paths = {}
    expected_source_schemas = {
        "universe": "cultra.point-in-time-universe-source.v2",
        "sessions": "cultra.market-session-source.v2",
        "events": "cultra.historical-event-source.v2",
    }
    for key in ("universe", "sessions", "events"):
        item = sources.get(key)
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "schema"}:
            raise HistoricalPrerequisiteError("historical prerequisite source is missing: %s" % key)
        if item.get("schema") != expected_source_schemas[key]:
            raise HistoricalPrerequisiteError(
                "historical prerequisite source schema drifted: %s" % key
            )
        source = _owned_file(Path(str(item.get("path", ""))), "%s source" % key)
        if _sha256(source) != item.get("sha256"):
            raise HistoricalPrerequisiteError("historical prerequisite source changed: %s" % key)
        source_paths[key] = source
    rebuilt = prepare_historical_prerequisites(
        input_set_id=str(value.get("input_set_id", "")),
        universe_source_path=source_paths["universe"],
        session_source_path=source_paths["sessions"],
        event_source_path=source_paths["events"],
    )
    expected_payloads = {
        "point_in_time_universe": rebuilt.universe_payload,
        "sessions": rebuilt.session_payload,
        "events": rebuilt.event_payload,
        "cohorts": rebuilt.cohort_payload,
    }
    normalized_paths = {}
    for key, expected in expected_payloads.items():
        item = normalized.get(key)
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise HistoricalPrerequisiteError("normalized prerequisite is missing: %s" % key)
        target = _owned_file(Path(str(item.get("path", ""))), "normalized %s" % key)
        if _sha256(target) != item.get("sha256"):
            raise HistoricalPrerequisiteError("normalized prerequisite changed: %s" % key)
        actual = _load_json(target, "normalized %s" % key)
        if actual != expected:
            raise HistoricalPrerequisiteError(
                "normalized prerequisite cannot be reproduced: %s" % key
            )
        normalized_paths[key] = target
    if tuple(value.get("selection_dates", ())) != rebuilt.selection_dates:
        raise HistoricalPrerequisiteError("historical prerequisite selection dates drifted")
    if tuple(value.get("sampled_symbols", ())) != rebuilt.sampled_symbols:
        raise HistoricalPrerequisiteError("historical prerequisite sampled symbols drifted")
    if int(value.get("sampled_symbol_count", 0)) != len(rebuilt.sampled_symbols):
        raise HistoricalPrerequisiteError("historical prerequisite sampled symbol count drifted")
    return FrozenHistoricalPrerequisites(
        payload=value,
        source_path=supplied,
        universe_path=normalized_paths["point_in_time_universe"],
        session_calendar_path=normalized_paths["sessions"],
        event_manifest_path=normalized_paths["events"],
        cohort_path=normalized_paths["cohorts"],
    )


__all__ = [
    "FrozenHistoricalPrerequisites",
    "HistoricalPrerequisiteError",
    "PreparedHistoricalPrerequisites",
    "load_historical_prerequisites",
    "load_point_in_time_universe_source",
    "prepare_historical_prerequisites",
    "save_historical_prerequisites",
]
