"""Durable one-way evidence-state registry for Cultra.

The registry is deliberately separate from model/statistical calculations.  It
locks their identities, consumes the final holdout once, and makes failed
holdout or shadow evidence terminal across process restarts.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Iterator, Mapping, Optional, Sequence, Tuple

from .catalog import CATALOG_VERSION


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVIDENCE_ROOT = PROJECT_ROOT / "var" / "evidence"


class RegistryState(str, Enum):
    UNPROVEN = "UNPROVEN"
    RESEARCH_PASS = "RESEARCH_PASS"
    VALIDATION_PASS = "VALIDATION_PASS"
    HOLDOUT_PASS = "HOLDOUT_PASS"
    SHADOW_PASS = "SHADOW_PASS"
    MANUAL_TICKET_ENABLED = "MANUAL_TICKET_ENABLED"
    REJECTED = "REJECTED"


class EvidenceRegistryError(RuntimeError):
    pass


class EvidenceMutationError(EvidenceRegistryError):
    pass


class EvidenceRetestError(EvidenceRegistryError):
    pass


class HoldoutReuseError(EvidenceRetestError):
    pass


class TerminalEvidenceError(EvidenceRegistryError):
    pass


def _require_text(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("%s is required" % name)
    return normalized


def _aware_iso(value: datetime, name: str) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("%s must be timezone-aware" % name)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class EvidencePartitions:
    training_observation_ids: Tuple[str, ...]
    validation_observation_ids: Tuple[str, ...]
    holdout_observation_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        normalized_groups = []
        for name in (
            "training_observation_ids",
            "validation_observation_ids",
            "holdout_observation_ids",
        ):
            values = tuple(sorted(_require_text(value, name) for value in getattr(self, name)))
            if not values:
                raise ValueError("%s cannot be empty" % name)
            if len(values) != len(set(values)):
                raise ValueError("%s contains duplicate evidence" % name)
            object.__setattr__(self, name, values)
            normalized_groups.append(set(values))
        if (
            normalized_groups[0] & normalized_groups[1]
            or normalized_groups[0] & normalized_groups[2]
            or normalized_groups[1] & normalized_groups[2]
        ):
            raise ValueError("training, validation, and holdout evidence overlap")

    @property
    def development_fingerprint(self) -> str:
        return _fingerprint(
            {
                "training": self.training_observation_ids,
                "validation": self.validation_observation_ids,
            }
        )

    @property
    def holdout_fingerprint(self) -> str:
        return _fingerprint({"holdout": self.holdout_observation_ids})

    @property
    def partition_fingerprint(self) -> str:
        return _fingerprint(
            {
                "training": self.training_observation_ids,
                "validation": self.validation_observation_ids,
                "holdout": self.holdout_observation_ids,
            }
        )


@dataclass(frozen=True)
class FrozenEvidenceIdentity:
    strategy_family: str
    catalog_version: str
    hypothesis_fingerprint: str
    cost_model_version: str
    exit_policy_version: str
    pop_model_version: str
    pop_model_artifact_id: str
    model_frozen_at: datetime

    def __post_init__(self) -> None:
        for name in (
            "strategy_family",
            "catalog_version",
            "hypothesis_fingerprint",
            "cost_model_version",
            "exit_policy_version",
            "pop_model_version",
            "pop_model_artifact_id",
        ):
            object.__setattr__(self, name, _require_text(getattr(self, name), name))
        if self.catalog_version != CATALOG_VERSION:
            raise ValueError("catalog_version must match the frozen Cultra catalog")
        _aware_iso(self.model_frozen_at, "model_frozen_at")


@dataclass(frozen=True)
class EvidenceRegistryRecord:
    strategy_family: str
    state: RegistryState
    catalog_version: str
    hypothesis_fingerprint: str
    cost_model_version: str
    exit_policy_version: str
    pop_model_version: str
    pop_model_artifact_id: str
    model_frozen_at: str
    partition_fingerprint: str
    development_fingerprint: str
    holdout_fingerprint: str
    holdout_consumed: bool
    failure_stage: Optional[str]
    registered_at: str
    updated_at: str


class EvidenceRegistry:
    """SQLite-backed, append-audited state machine with immutable identity."""

    def __init__(self, path: Path, timeout_seconds: float = 5.0):
        root = DEFAULT_EVIDENCE_ROOT.resolve()
        DEFAULT_EVIDENCE_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(DEFAULT_EVIDENCE_ROOT, 0o700)
        supplied = Path(path)
        if str(supplied) == ":memory:":
            raise ValueError("evidence registry must be durable, not in-memory")
        if supplied.suffix != ".sqlite3":
            raise ValueError("evidence registry must use a .sqlite3 file")
        resolved_parent = supplied.parent.resolve()
        try:
            resolved_parent.relative_to(root)
        except ValueError as exc:
            raise ValueError("evidence registry path must remain Cultra-local") from exc
        supplied.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(supplied.parent, 0o700)
        candidate = resolved_parent / supplied.name
        if candidate.is_symlink():
            raise ValueError("evidence registry cannot be a symlink")
        self.path = candidate
        if str(self.path) == ":memory:":
            raise ValueError("evidence registry must be durable, not in-memory")
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=float(timeout_seconds),
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=DELETE")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()
        os.chmod(self.path, 0o600)

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS evidence_families (
                strategy_family TEXT PRIMARY KEY,
                state TEXT NOT NULL CHECK (state IN (
                    'UNPROVEN','RESEARCH_PASS','VALIDATION_PASS','HOLDOUT_PASS',
                    'SHADOW_PASS','MANUAL_TICKET_ENABLED','REJECTED'
                )),
                catalog_version TEXT NOT NULL,
                hypothesis_fingerprint TEXT NOT NULL,
                cost_model_version TEXT NOT NULL,
                exit_policy_version TEXT NOT NULL,
                pop_model_version TEXT NOT NULL,
                pop_model_artifact_id TEXT NOT NULL,
                model_frozen_at TEXT NOT NULL,
                partition_fingerprint TEXT NOT NULL,
                development_fingerprint TEXT NOT NULL,
                holdout_fingerprint TEXT NOT NULL,
                holdout_consumed INTEGER NOT NULL DEFAULT 0 CHECK (holdout_consumed IN (0,1)),
                failure_stage TEXT,
                registered_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evidence_partition_membership (
                strategy_family TEXT NOT NULL REFERENCES evidence_families(strategy_family),
                observation_id TEXT NOT NULL,
                partition_name TEXT NOT NULL CHECK (partition_name IN ('TRAINING','VALIDATION','HOLDOUT')),
                PRIMARY KEY (strategy_family, observation_id)
            );

            CREATE TABLE IF NOT EXISTS evidence_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_family TEXT NOT NULL REFERENCES evidence_families(strategy_family),
                stage TEXT NOT NULL,
                evidence_fingerprint TEXT NOT NULL,
                passed INTEGER NOT NULL CHECK (passed IN (0,1)),
                resulting_state TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                UNIQUE (strategy_family, stage, evidence_fingerprint)
            );

            CREATE TRIGGER IF NOT EXISTS evidence_identity_is_immutable
            BEFORE UPDATE OF
                catalog_version, hypothesis_fingerprint, cost_model_version,
                exit_policy_version, pop_model_version, pop_model_artifact_id,
                model_frozen_at, partition_fingerprint, development_fingerprint,
                holdout_fingerprint
            ON evidence_families
            BEGIN
                SELECT RAISE(ABORT, 'frozen evidence identity');
            END;
            """
        )

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield self._connection
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "EvidenceRegistry":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _row_to_record(self, row: sqlite3.Row) -> EvidenceRegistryRecord:
        return EvidenceRegistryRecord(
            strategy_family=row["strategy_family"],
            state=RegistryState(row["state"]),
            catalog_version=row["catalog_version"],
            hypothesis_fingerprint=row["hypothesis_fingerprint"],
            cost_model_version=row["cost_model_version"],
            exit_policy_version=row["exit_policy_version"],
            pop_model_version=row["pop_model_version"],
            pop_model_artifact_id=row["pop_model_artifact_id"],
            model_frozen_at=row["model_frozen_at"],
            partition_fingerprint=row["partition_fingerprint"],
            development_fingerprint=row["development_fingerprint"],
            holdout_fingerprint=row["holdout_fingerprint"],
            holdout_consumed=bool(row["holdout_consumed"]),
            failure_stage=row["failure_stage"],
            registered_at=row["registered_at"],
            updated_at=row["updated_at"],
        )

    def get(self, strategy_family: str) -> EvidenceRegistryRecord:
        row = self._connection.execute(
            "SELECT * FROM evidence_families WHERE strategy_family = ?",
            (_require_text(strategy_family, "strategy_family"),),
        ).fetchone()
        if row is None:
            raise KeyError("strategy family is not registered: %s" % strategy_family)
        return self._row_to_record(row)

    def register(
        self,
        identity: FrozenEvidenceIdentity,
        partitions: EvidencePartitions,
        *,
        now: Optional[datetime] = None,
    ) -> EvidenceRegistryRecord:
        timestamp = _aware_iso(now or datetime.now(timezone.utc), "now")
        model_frozen_at = _aware_iso(identity.model_frozen_at, "model_frozen_at")
        if model_frozen_at > timestamp:
            raise ValueError("model_frozen_at cannot be in the future")
        frozen_values = (
            identity.catalog_version,
            identity.hypothesis_fingerprint,
            identity.cost_model_version,
            identity.exit_policy_version,
            identity.pop_model_version,
            identity.pop_model_artifact_id,
            model_frozen_at,
            partitions.partition_fingerprint,
            partitions.development_fingerprint,
            partitions.holdout_fingerprint,
        )
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM evidence_families WHERE strategy_family = ?",
                (identity.strategy_family,),
            ).fetchone()
            if existing is not None:
                existing_values = (
                    existing["catalog_version"],
                    existing["hypothesis_fingerprint"],
                    existing["cost_model_version"],
                    existing["exit_policy_version"],
                    existing["pop_model_version"],
                    existing["pop_model_artifact_id"],
                    existing["model_frozen_at"],
                    existing["partition_fingerprint"],
                    existing["development_fingerprint"],
                    existing["holdout_fingerprint"],
                )
                if existing_values != frozen_values:
                    raise EvidenceMutationError(
                        "frozen evidence identity cannot be mutated or replaced"
                    )
                return self._row_to_record(existing)
            connection.execute(
                """
                INSERT INTO evidence_families (
                    strategy_family, state, catalog_version, hypothesis_fingerprint,
                    cost_model_version, exit_policy_version, pop_model_version,
                    pop_model_artifact_id, model_frozen_at, partition_fingerprint,
                    development_fingerprint, holdout_fingerprint, holdout_consumed,
                    failure_stage, registered_at, updated_at
                ) VALUES (?, 'UNPROVEN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, ?, ?)
                """,
                (identity.strategy_family,) + frozen_values + (timestamp, timestamp),
            )
            for partition_name, observation_ids in (
                ("TRAINING", partitions.training_observation_ids),
                ("VALIDATION", partitions.validation_observation_ids),
                ("HOLDOUT", partitions.holdout_observation_ids),
            ):
                connection.executemany(
                    """
                    INSERT INTO evidence_partition_membership
                        (strategy_family, observation_id, partition_name)
                    VALUES (?, ?, ?)
                    """,
                    (
                        (identity.strategy_family, observation_id, partition_name)
                        for observation_id in observation_ids
                    ),
                )
        return self.get(identity.strategy_family)

    def assert_identity(self, identity: FrozenEvidenceIdentity) -> None:
        record = self.get(identity.strategy_family)
        expected = (
            identity.catalog_version,
            identity.hypothesis_fingerprint,
            identity.cost_model_version,
            identity.exit_policy_version,
            identity.pop_model_version,
            identity.pop_model_artifact_id,
            _aware_iso(identity.model_frozen_at, "model_frozen_at"),
        )
        actual = (
            record.catalog_version,
            record.hypothesis_fingerprint,
            record.cost_model_version,
            record.exit_policy_version,
            record.pop_model_version,
            record.pop_model_artifact_id,
            record.model_frozen_at,
        )
        if expected != actual:
            raise EvidenceMutationError("runtime identity differs from frozen evidence identity")

    def _record_attempt(
        self,
        connection: sqlite3.Connection,
        strategy_family: str,
        stage: str,
        evidence_fingerprint: str,
        passed: bool,
        resulting_state: RegistryState,
        timestamp: str,
    ) -> None:
        try:
            connection.execute(
                """
                INSERT INTO evidence_attempts
                    (strategy_family, stage, evidence_fingerprint, passed,
                     resulting_state, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    strategy_family,
                    stage,
                    _require_text(evidence_fingerprint, "evidence_fingerprint"),
                    int(bool(passed)),
                    resulting_state.value,
                    timestamp,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise EvidenceRetestError("the same evidence has already been tested") from exc

    def advance_development(
        self,
        strategy_family: str,
        target_state: RegistryState,
        evidence_fingerprint: str,
        *,
        now: Optional[datetime] = None,
    ) -> EvidenceRegistryRecord:
        allowed = {
            RegistryState.UNPROVEN: RegistryState.RESEARCH_PASS,
            RegistryState.RESEARCH_PASS: RegistryState.VALIDATION_PASS,
        }
        timestamp = _aware_iso(now or datetime.now(timezone.utc), "now")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_families WHERE strategy_family = ?",
                (strategy_family,),
            ).fetchone()
            if row is None:
                raise KeyError("strategy family is not registered: %s" % strategy_family)
            current = RegistryState(row["state"])
            if current is RegistryState.REJECTED:
                raise TerminalEvidenceError("rejected evidence is terminal")
            if allowed.get(current) is not target_state:
                raise EvidenceRetestError("development transition is not the next frozen stage")
            if evidence_fingerprint != row["development_fingerprint"]:
                raise EvidenceMutationError("development evidence differs from locked partitions")
            self._record_attempt(
                connection,
                strategy_family,
                target_state.value,
                evidence_fingerprint,
                True,
                target_state,
                timestamp,
            )
            connection.execute(
                "UPDATE evidence_families SET state = ?, updated_at = ? WHERE strategy_family = ?",
                (target_state.value, timestamp, strategy_family),
            )
        return self.get(strategy_family)

    def reject_development(
        self,
        strategy_family: str,
        stage: RegistryState,
        evidence_fingerprint: str,
        *,
        now: Optional[datetime] = None,
    ) -> EvidenceRegistryRecord:
        """Record a failed research/validation gate as terminal evidence."""

        expected = {
            RegistryState.UNPROVEN: RegistryState.RESEARCH_PASS,
            RegistryState.RESEARCH_PASS: RegistryState.VALIDATION_PASS,
        }
        if stage not in {RegistryState.RESEARCH_PASS, RegistryState.VALIDATION_PASS}:
            raise EvidenceRegistryError("development rejection stage is invalid")
        timestamp = _aware_iso(now or datetime.now(timezone.utc), "now")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_families WHERE strategy_family = ?",
                (strategy_family,),
            ).fetchone()
            if row is None:
                raise KeyError("strategy family is not registered: %s" % strategy_family)
            current = RegistryState(row["state"])
            if current is RegistryState.REJECTED:
                raise TerminalEvidenceError("rejected evidence is terminal")
            if expected.get(current) is not stage:
                raise EvidenceRetestError("development rejection is not the next frozen stage")
            if evidence_fingerprint != row["development_fingerprint"]:
                raise EvidenceMutationError("development evidence differs from locked partitions")
            self._record_attempt(
                connection,
                strategy_family,
                stage.value,
                evidence_fingerprint,
                False,
                RegistryState.REJECTED,
                timestamp,
            )
            connection.execute(
                """
                UPDATE evidence_families
                   SET state = 'REJECTED', failure_stage = ?, updated_at = ?
                 WHERE strategy_family = ?
                """,
                (stage.value, timestamp, strategy_family),
            )
        return self.get(strategy_family)

    def consume_holdout(
        self,
        strategy_family: str,
        holdout_fingerprint: str,
        *,
        passed: bool,
        now: Optional[datetime] = None,
    ) -> EvidenceRegistryRecord:
        timestamp = _aware_iso(now or datetime.now(timezone.utc), "now")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_families WHERE strategy_family = ?",
                (strategy_family,),
            ).fetchone()
            if row is None:
                raise KeyError("strategy family is not registered: %s" % strategy_family)
            state = RegistryState(row["state"])
            if state is RegistryState.REJECTED:
                raise TerminalEvidenceError("rejected evidence is terminal")
            if row["holdout_consumed"]:
                raise HoldoutReuseError("final holdout has already been consumed")
            if state is not RegistryState.VALIDATION_PASS:
                raise EvidenceRegistryError("holdout can be consumed only after validation")
            if holdout_fingerprint != row["holdout_fingerprint"]:
                raise EvidenceMutationError("holdout evidence differs from the locked partition")
            result = RegistryState.HOLDOUT_PASS if passed else RegistryState.REJECTED
            self._record_attempt(
                connection,
                strategy_family,
                "HOLDOUT",
                holdout_fingerprint,
                passed,
                result,
                timestamp,
            )
            connection.execute(
                """
                UPDATE evidence_families
                SET state = ?, holdout_consumed = 1, failure_stage = ?, updated_at = ?
                WHERE strategy_family = ?
                """,
                (
                    result.value,
                    None if passed else "HOLDOUT",
                    timestamp,
                    strategy_family,
                ),
            )
        return self.get(strategy_family)

    def consume_holdout_batch(
        self,
        decisions: Sequence[Tuple[str, str, bool]],
        *,
        now: Optional[datetime] = None,
    ) -> Mapping[str, EvidenceRegistryRecord]:
        """Atomically consume a set of holdouts after family-wide correction.

        Holm adjustment is a joint decision across the frozen hypothesis
        family.  Preflight every eligible record first so a bad identity or a
        reused holdout cannot leave only part of that joint decision consumed.
        """

        normalized = tuple(
            (
                _require_text(strategy_family, "strategy_family"),
                _require_text(holdout_fingerprint, "holdout_fingerprint"),
                bool(passed),
            )
            for strategy_family, holdout_fingerprint, passed in decisions
        )
        if not normalized:
            raise EvidenceRegistryError("holdout decision batch cannot be empty")
        families = tuple(item[0] for item in normalized)
        if len(families) != len(set(families)):
            raise EvidenceRegistryError("holdout decision batch contains duplicates")
        timestamp = _aware_iso(now or datetime.now(timezone.utc), "now")
        with self._transaction() as connection:
            rows = {}
            for strategy_family, holdout_fingerprint, _passed in normalized:
                row = connection.execute(
                    "SELECT * FROM evidence_families WHERE strategy_family = ?",
                    (strategy_family,),
                ).fetchone()
                if row is None:
                    raise KeyError(
                        "strategy family is not registered: %s" % strategy_family
                    )
                state = RegistryState(row["state"])
                if state is RegistryState.REJECTED:
                    raise TerminalEvidenceError("rejected evidence is terminal")
                if row["holdout_consumed"]:
                    raise HoldoutReuseError("final holdout has already been consumed")
                if state is not RegistryState.VALIDATION_PASS:
                    raise EvidenceRegistryError(
                        "holdout can be consumed only after validation"
                    )
                if holdout_fingerprint != row["holdout_fingerprint"]:
                    raise EvidenceMutationError(
                        "holdout evidence differs from the locked partition"
                    )
                rows[strategy_family] = row
            for strategy_family, holdout_fingerprint, passed in normalized:
                result = (
                    RegistryState.HOLDOUT_PASS if passed else RegistryState.REJECTED
                )
                self._record_attempt(
                    connection,
                    strategy_family,
                    "HOLDOUT",
                    holdout_fingerprint,
                    passed,
                    result,
                    timestamp,
                )
                connection.execute(
                    """
                    UPDATE evidence_families
                       SET state = ?, holdout_consumed = 1, failure_stage = ?,
                           updated_at = ?
                     WHERE strategy_family = ?
                    """,
                    (
                        result.value,
                        None if passed else "HOLDOUT",
                        timestamp,
                        strategy_family,
                    ),
                )
        return {family: self.get(family) for family in families}

    def record_shadow(
        self,
        strategy_family: str,
        shadow_fingerprint: str,
        *,
        passed: bool,
        now: Optional[datetime] = None,
    ) -> EvidenceRegistryRecord:
        timestamp = _aware_iso(now or datetime.now(timezone.utc), "now")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM evidence_families WHERE strategy_family = ?",
                (strategy_family,),
            ).fetchone()
            if row is None:
                raise KeyError("strategy family is not registered: %s" % strategy_family)
            state = RegistryState(row["state"])
            if state is RegistryState.REJECTED:
                raise TerminalEvidenceError("rejected evidence is terminal")
            if state not in {
                RegistryState.HOLDOUT_PASS,
                RegistryState.MANUAL_TICKET_ENABLED,
            }:
                raise EvidenceRetestError(
                    "shadow evidence requires holdout passage or an enabled family"
                )
            result = (
                RegistryState.REJECTED
                if not passed
                else RegistryState.MANUAL_TICKET_ENABLED
                if state is RegistryState.MANUAL_TICKET_ENABLED
                else RegistryState.SHADOW_PASS
            )
            self._record_attempt(
                connection,
                strategy_family,
                "SHADOW",
                shadow_fingerprint,
                passed,
                result,
                timestamp,
            )
            connection.execute(
                """
                UPDATE evidence_families
                SET state = ?, failure_stage = ?, updated_at = ?
                WHERE strategy_family = ?
                """,
                (
                    result.value,
                    None if passed else "SHADOW",
                    timestamp,
                    strategy_family,
                ),
            )
        return self.get(strategy_family)

    def enable_manual_tickets(
        self, strategy_family: str, *, now: Optional[datetime] = None
    ) -> EvidenceRegistryRecord:
        timestamp = _aware_iso(now or datetime.now(timezone.utc), "now")
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT state FROM evidence_families WHERE strategy_family = ?",
                (strategy_family,),
            ).fetchone()
            if row is None:
                raise KeyError("strategy family is not registered: %s" % strategy_family)
            state = RegistryState(row["state"])
            if state is RegistryState.REJECTED:
                raise TerminalEvidenceError("rejected evidence is terminal")
            if state not in {RegistryState.HOLDOUT_PASS, RegistryState.SHADOW_PASS}:
                raise EvidenceRegistryError(
                    "manual tickets require untouched HOLDOUT_PASS evidence"
                )
            connection.execute(
                "UPDATE evidence_families SET state = ?, updated_at = ? WHERE strategy_family = ?",
                (RegistryState.MANUAL_TICKET_ENABLED.value, timestamp, strategy_family),
            )
        return self.get(strategy_family)

    def attempts(self, strategy_family: str) -> Tuple[sqlite3.Row, ...]:
        return tuple(
            self._connection.execute(
                """
                SELECT stage, evidence_fingerprint, passed, resulting_state, recorded_at
                FROM evidence_attempts WHERE strategy_family = ? ORDER BY attempt_id
                """,
                (strategy_family,),
            ).fetchall()
        )


__all__ = [
    "EvidenceMutationError",
    "EvidencePartitions",
    "EvidenceRegistry",
    "EvidenceRegistryError",
    "EvidenceRegistryRecord",
    "EvidenceRetestError",
    "FrozenEvidenceIdentity",
    "HoldoutReuseError",
    "RegistryState",
    "TerminalEvidenceError",
]
