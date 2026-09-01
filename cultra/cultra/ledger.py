"""Durable, process-safe ORATS attempt accounting.

Every row in ``attempts`` is an irreversibly charged outbound-attempt permit.
The row is committed before transport is reachable.  A crash can therefore
burn a permit, but can never reuse one.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .requesting import (
    MAX_SINGLE_RESPONSE_BYTES,
    MAX_TOTAL_RESPONSE_BYTES_EOD,
    PROTOCOL_MAX_ATTEMPTS,
    RequestPlan,
)


class LedgerError(RuntimeError):
    """Base class for fail-closed ledger errors."""


class LedgerUnavailable(LedgerError):
    """The durable ledger cannot safely account for another send."""


class ActiveRunError(LedgerError):
    """Another run already owns this credential scope."""


class BudgetExhausted(LedgerError):
    """No additional physical attempt may be sent."""


class CircuitOpen(BudgetExhausted):
    """Repeated provider failures opened the run-wide circuit."""


class ResponseBudgetExhausted(BudgetExhausted):
    """The EOD response-byte envelope cannot admit another request."""


class PermitStateError(LedgerError):
    """An invalid or duplicate attempt transition was requested."""


class AttemptState(str, Enum):
    RESERVED = "reserved"
    INDETERMINATE = "indeterminate"
    CONFIRMED = "confirmed"
    COMPLETED = "completed"


class RunState(str, Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass(frozen=True)
class AttemptPermit:
    permit_id: int
    run_id: str
    logical_request_id: str
    network_attempt_number: int
    retry_number: int
    endpoint: str
    method: str
    request_fingerprint: str
    response_byte_reservation: int
    reserved_at: float


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9_.:@\-]{1,128}$")
ORATS_ACCOUNT_SCOPE = "cultra-orats-account-v1"
PROVIDER_CIRCUIT_FAILURE_THRESHOLD = 3
CULTRA_PROJECT_ROOT = Path(__file__).resolve().parents[1]
CULTRA_LEDGER_ROOT = (CULTRA_PROJECT_ROOT / "state" / "orats_ledger").resolve()


def account_ledger_path() -> Path:
    """Return the one Cultra-wide ORATS ledger path without token material."""

    return (CULTRA_LEDGER_ROOT / "account.sqlite3").resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class RequestLedger:
    """SQLite-backed ledger with an atomic pre-send permit boundary."""

    def __init__(self, path: Path, *, timeout_seconds: float = 10.0) -> None:
        supplied = Path(path).expanduser()
        if supplied.is_symlink():
            raise LedgerUnavailable("request ledger path may not be a symlink")
        self.path = supplied.resolve()
        allowed_root = CULTRA_LEDGER_ROOT.resolve()
        if not _is_within(self.path, allowed_root) or self.path.suffix != ".sqlite3":
            raise LedgerUnavailable("request ledger must be inside the Cultra ledger root")
        self.timeout_seconds = float(timeout_seconds)
        if str(self.path) == ":memory:":
            raise LedgerUnavailable("the request ledger must be durable")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()
        stat = self.path.stat()
        self._identity = (stat.st_dev, stat.st_ino)

    def _connect(self) -> sqlite3.Connection:
        self._assert_identity()
        try:
            connection = sqlite3.connect(
                str(self.path),
                timeout=self.timeout_seconds,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA busy_timeout=%d" % int(self.timeout_seconds * 1000))
            return connection
        except sqlite3.Error as exc:
            raise LedgerUnavailable("request ledger is unavailable") from exc

    def _assert_identity(self) -> None:
        if hasattr(self, "_identity"):
            try:
                stat = self.path.stat()
            except OSError as exc:
                raise LedgerUnavailable("request ledger disappeared") from exc
            if (stat.st_dev, stat.st_ino) != self._identity:
                raise LedgerUnavailable("request ledger identity changed")

    def _initialize(self) -> None:
        try:
            connection = sqlite3.connect(
                str(self.path), timeout=self.timeout_seconds, isolation_level=None
            )
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS campaigns (
                    campaign_id TEXT PRIMARY KEY,
                    hard_cap INTEGER NOT NULL CHECK(hard_cap BETWEEN 1 AND 99),
                    charged_attempts INTEGER NOT NULL DEFAULT 0
                        CHECK(charged_attempts BETWEEN 0 AND 99),
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    credential_key TEXT NOT NULL,
                    campaign_id TEXT NOT NULL REFERENCES campaigns(campaign_id),
                    run_type TEXT NOT NULL,
                    plan_hash TEXT NOT NULL,
                    hard_cap INTEGER NOT NULL CHECK(hard_cap BETWEEN 1 AND 99),
                    admitted_attempt_cap INTEGER NOT NULL CHECK(admitted_attempt_cap BETWEEN 0 AND 99),
                    response_byte_cap INTEGER NOT NULL CHECK(response_byte_cap >= 0),
                    circuit_state TEXT NOT NULL CHECK(circuit_state IN ('closed','open')),
                    consecutive_provider_failures INTEGER NOT NULL CHECK(consecutive_provider_failures >= 0),
                    target INTEGER NOT NULL CHECK(target >= 0),
                    retry_reserve INTEGER NOT NULL CHECK(retry_reserve >= 0),
                    state TEXT NOT NULL CHECK(state IN ('active','completed','aborted')),
                    started_at REAL NOT NULL,
                    ended_at REAL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_run_per_credential
                    ON runs(credential_key) WHERE state = 'active';
                CREATE TABLE IF NOT EXISTS planned_requests (
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    logical_request_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    retry_limit INTEGER NOT NULL CHECK(retry_limit BETWEEN 0 AND 2),
                    contingency INTEGER NOT NULL CHECK(contingency IN (0,1)),
                    entity_count INTEGER NOT NULL CHECK(entity_count > 0),
                    field_profile TEXT NOT NULL,
                    max_response_bytes INTEGER NOT NULL CHECK(max_response_bytes BETWEEN 1 AND 25000000),
                    PRIMARY KEY(run_id, logical_request_id)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    permit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(run_id),
                    logical_request_id TEXT NOT NULL,
                    network_attempt_number INTEGER NOT NULL CHECK(network_attempt_number BETWEEN 1 AND 99),
                    retry_number INTEGER NOT NULL CHECK(retry_number BETWEEN 0 AND 2),
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    response_byte_reservation INTEGER NOT NULL CHECK(response_byte_reservation BETWEEN 0 AND 25000000),
                    state TEXT NOT NULL CHECK(state IN ('reserved','indeterminate','confirmed','completed')),
                    reserved_at REAL NOT NULL,
                    send_started_at REAL,
                    confirmed_at REAL,
                    completed_at REAL,
                    status_code INTEGER,
                    rows_returned INTEGER,
                    bytes_returned INTEGER,
                    duration_ms REAL,
                    provider_trade_date TEXT,
                    updated_at_min TEXT,
                    updated_at_max TEXT,
                    outcome_code TEXT,
                    UNIQUE(run_id, network_attempt_number),
                    UNIQUE(run_id, logical_request_id, retry_number),
                    FOREIGN KEY(run_id, logical_request_id)
                        REFERENCES planned_requests(run_id, logical_request_id)
                );
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT OR IGNORE INTO meta(key, value)
                    VALUES ('schema_version', 'CULTRA_LEDGER_V1');
                """
            )
            # Migration-safe additions for ledgers created by the initial V1
            # schema.  SQLite's ADD COLUMN is deliberately used instead of a
            # table rewrite so an active fail-closed ledger is not replaced.
            run_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(runs)")
            }
            if "admitted_attempt_cap" not in run_columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN admitted_attempt_cap INTEGER"
                )
            if "response_byte_cap" not in run_columns:
                connection.execute("ALTER TABLE runs ADD COLUMN response_byte_cap INTEGER")
            if "circuit_state" not in run_columns:
                connection.execute("ALTER TABLE runs ADD COLUMN circuit_state TEXT")
            if "consecutive_provider_failures" not in run_columns:
                connection.execute(
                    "ALTER TABLE runs ADD COLUMN consecutive_provider_failures INTEGER"
                )
            if "campaign_id" not in run_columns:
                connection.execute("ALTER TABLE runs ADD COLUMN campaign_id TEXT")
            planned_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(planned_requests)")
            }
            if "entity_count" not in planned_columns:
                connection.execute(
                    "ALTER TABLE planned_requests ADD COLUMN entity_count INTEGER NOT NULL DEFAULT 0"
                )
            if "field_profile" not in planned_columns:
                connection.execute(
                    "ALTER TABLE planned_requests ADD COLUMN field_profile TEXT NOT NULL DEFAULT 'UNKNOWN_V1'"
                )
            if "max_response_bytes" not in planned_columns:
                connection.execute(
                    "ALTER TABLE planned_requests ADD COLUMN max_response_bytes INTEGER NOT NULL DEFAULT 25000000"
                )
            attempt_columns = {
                str(row[1]) for row in connection.execute("PRAGMA table_info(attempts)")
            }
            if "response_byte_reservation" not in attempt_columns:
                connection.execute(
                    "ALTER TABLE attempts ADD COLUMN response_byte_reservation INTEGER NOT NULL DEFAULT 0"
                )
            connection.execute(
                """
                INSERT OR IGNORE INTO campaigns(
                    campaign_id, hard_cap, charged_attempts, created_at
                )
                SELECT run_id,
                       MIN(hard_cap, 99),
                       MIN((SELECT COUNT(*) FROM attempts
                             WHERE attempts.run_id = runs.run_id), 99),
                       started_at
                  FROM runs
                """
            )
            connection.execute(
                """
                UPDATE runs SET campaign_id = run_id
                 WHERE campaign_id IS NULL OR campaign_id = ''
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS runs_by_campaign ON runs(campaign_id)"
            )
            connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS require_campaign_budget_before_attempt
                BEFORE INSERT ON attempts
                BEGIN
                    SELECT CASE
                        WHEN NOT EXISTS (
                            SELECT 1 FROM runs
                            JOIN campaigns USING(campaign_id)
                            WHERE runs.run_id = NEW.run_id
                        ) THEN RAISE(ABORT, 'campaign_missing')
                        WHEN (
                            SELECT campaigns.charged_attempts
                              FROM runs JOIN campaigns USING(campaign_id)
                             WHERE runs.run_id = NEW.run_id
                        ) >= (
                            SELECT campaigns.hard_cap
                              FROM runs JOIN campaigns USING(campaign_id)
                             WHERE runs.run_id = NEW.run_id
                        ) THEN RAISE(ABORT, 'campaign_budget_exhausted')
                        WHEN (
                            SELECT campaigns.charged_attempts
                              FROM runs JOIN campaigns USING(campaign_id)
                             WHERE runs.run_id = NEW.run_id
                        ) >= 99 THEN RAISE(ABORT, 'campaign_budget_exhausted')
                    END;
                END;
                CREATE TRIGGER IF NOT EXISTS charge_campaign_after_attempt
                AFTER INSERT ON attempts
                BEGIN
                    UPDATE campaigns
                       SET charged_attempts = charged_attempts + 1
                     WHERE campaign_id = (
                         SELECT campaign_id FROM runs WHERE run_id = NEW.run_id
                     );
                END;
                """
            )
            connection.execute(
                """
                UPDATE runs
                   SET admitted_attempt_cap = MIN(
                       hard_cap,
                       (SELECT COUNT(*) FROM planned_requests AS planned
                         WHERE planned.run_id = runs.run_id) + retry_reserve
                   )
                 WHERE admitted_attempt_cap IS NULL
                    OR admitted_attempt_cap < 0
                    OR admitted_attempt_cap > hard_cap
                """
            )
            connection.execute(
                """
                UPDATE runs
                   SET response_byte_cap = CASE WHEN run_type = 'eod' THEN ? ELSE 0 END
                 WHERE response_byte_cap IS NULL OR response_byte_cap < 0;
                """,
                (MAX_TOTAL_RESPONSE_BYTES_EOD,),
            )
            connection.execute(
                "UPDATE runs SET circuit_state = 'closed' WHERE circuit_state IS NULL"
            )
            connection.execute(
                """
                UPDATE runs SET consecutive_provider_failures = 0
                 WHERE consecutive_provider_failures IS NULL
                    OR consecutive_provider_failures < 0
                """
            )
            connection.execute(
                """
                UPDATE attempts
                   SET response_byte_reservation = CASE
                       WHEN bytes_returned IS NOT NULL THEN bytes_returned
                       ELSE ? END
                 WHERE response_byte_reservation <= 0
                """,
                (MAX_SINGLE_RESPONSE_BYTES,),
            )
            connection.execute(
                "UPDATE meta SET value = 'CULTRA_LEDGER_V4' WHERE key = 'schema_version'"
            )
            connection.close()
            os.chmod(self.path, 0o600)
            directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except (OSError, sqlite3.Error) as exc:
            raise LedgerUnavailable("could not initialize durable request ledger") from exc

    @staticmethod
    def _validate_identifier(value: str, label: str) -> None:
        if not _SAFE_IDENTIFIER.fullmatch(value):
            raise LedgerError("invalid non-secret %s" % label)

    def assert_healthy(self) -> None:
        """Fail closed on corruption, replacement, or a locked database."""

        connection = self._connect()
        try:
            row = connection.execute("PRAGMA quick_check(1)").fetchone()
            if row is None or row[0] != "ok":
                raise LedgerUnavailable("request ledger integrity check failed")
            mode = connection.execute("PRAGMA synchronous").fetchone()[0]
            if int(mode) < 2:
                raise LedgerUnavailable("request ledger is not in durable synchronous mode")
        except sqlite3.Error as exc:
            raise LedgerUnavailable("request ledger health check failed") from exc
        finally:
            connection.close()

    def start_run(self, plan: RequestPlan) -> None:
        """Persist a frozen plan and acquire the account-wide run lease."""

        self._validate_identifier(plan.run_id, "run ID")
        campaign_id = plan.campaign_id or plan.run_id
        campaign_hard_cap = plan.campaign_hard_cap or plan.hard_cap
        self._validate_identifier(campaign_id, "campaign ID")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (plan.run_id,)
            ).fetchone()
            if existing is not None:
                matching = (
                    existing["plan_hash"] == plan.plan_hash
                    and existing["credential_key"] == ORATS_ACCOUNT_SCOPE
                    and existing["campaign_id"] == campaign_id
                    and existing["state"] == RunState.ACTIVE.value
                )
                if matching:
                    connection.execute("COMMIT")
                    return
                raise ActiveRunError("run ID already exists with different state or plan")
            active = connection.execute(
                "SELECT run_id FROM runs WHERE credential_key = ? AND state = 'active'",
                (ORATS_ACCOUNT_SCOPE,),
            ).fetchone()
            if active is not None:
                raise ActiveRunError("the Cultra ORATS account already has an active run")
            campaign = connection.execute(
                "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
            ).fetchone()
            if campaign is None:
                connection.execute(
                    """
                    INSERT INTO campaigns(campaign_id, hard_cap, charged_attempts, created_at)
                    VALUES (?, ?, 0, ?)
                    """,
                    (campaign_id, campaign_hard_cap, time.time()),
                )
                campaign_charged = 0
            else:
                if int(campaign["hard_cap"]) != int(campaign_hard_cap):
                    raise LedgerError("campaign hard cap cannot change between phases")
                campaign_charged = int(campaign["charged_attempts"])
            if campaign_charged + plan.worst_charged_attempts > campaign_hard_cap:
                raise BudgetExhausted(
                    "complete run envelope does not fit the remaining campaign budget"
                )
            connection.execute(
                """
                INSERT INTO runs(
                    run_id, credential_key, campaign_id, run_type, plan_hash, hard_cap,
                    admitted_attempt_cap, response_byte_cap, circuit_state,
                    consecutive_provider_failures,
                    target, retry_reserve, state, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'closed', 0, ?, ?, 'active', ?)
                """,
                (
                    plan.run_id,
                    ORATS_ACCOUNT_SCOPE,
                    campaign_id,
                    plan.run_type.value,
                    plan.plan_hash,
                    plan.hard_cap,
                    min(plan.hard_cap, plan.logical_count + plan.retry_reserve),
                    MAX_TOTAL_RESPONSE_BYTES_EOD if plan.run_type.value == "eod" else 0,
                    plan.target,
                    plan.retry_reserve,
                    time.time(),
                ),
            )
            connection.executemany(
                """
                INSERT INTO planned_requests(
                    run_id, logical_request_id, endpoint, method,
                    request_fingerprint, retry_limit, contingency,
                    entity_count, field_profile, max_response_bytes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        plan.run_id,
                        item.logical_request_id,
                        item.endpoint.value,
                        item.method,
                        item.fingerprint,
                        item.retry_limit,
                        int(item.contingency),
                        len(item.entities),
                        item.field_profile,
                        item.max_response_bytes,
                    )
                    for item in plan.requests
                ],
            )
            connection.execute("COMMIT")
        except sqlite3.IntegrityError as exc:
            self._rollback_quietly(connection)
            raise ActiveRunError("could not acquire the credential run lease") from exc
        except (ActiveRunError, BudgetExhausted, LedgerError):
            self._rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            self._rollback_quietly(connection)
            raise LedgerUnavailable("could not start request-ledger run") from exc
        finally:
            connection.close()

    def reactivate_aborted_run(self, plan: RequestPlan) -> None:
        """Reacquire an exact aborted plan after its missing cache was recovered.

        Previously charged permits remain immutable and cannot be retried.  A
        resumed gateway can therefore complete an attempted logical request
        only from a validated cache snapshot; uncached attempted IDs still fail
        closed at the unique permit boundary.
        """

        self._validate_identifier(plan.run_id, "run ID")
        campaign_id = plan.campaign_id or plan.run_id
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (plan.run_id,)
            ).fetchone()
            if existing is None:
                raise ActiveRunError("aborted run does not exist")
            if (
                existing["plan_hash"] != plan.plan_hash
                or existing["credential_key"] != ORATS_ACCOUNT_SCOPE
                or existing["campaign_id"] != campaign_id
                or existing["state"] != RunState.ABORTED.value
            ):
                raise ActiveRunError(
                    "only the exact frozen aborted plan can be reactivated"
                )
            active = connection.execute(
                "SELECT run_id FROM runs WHERE credential_key = ? AND state = 'active'",
                (ORATS_ACCOUNT_SCOPE,),
            ).fetchone()
            if active is not None:
                raise ActiveRunError(
                    "the Cultra ORATS account already has an active run"
                )
            connection.execute(
                "UPDATE runs SET state = 'active', ended_at = NULL WHERE run_id = ?",
                (plan.run_id,),
            )
            connection.execute("COMMIT")
        except (ActiveRunError, LedgerError):
            self._rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            self._rollback_quietly(connection)
            raise LedgerUnavailable("could not reactivate the aborted run") from exc
        finally:
            connection.close()

    @staticmethod
    def _rollback_quietly(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def reserve_attempt(
        self, run_id: str, logical_request_id: str, *, retry_number: int = 0
    ) -> AttemptPermit:
        """Atomically charge one physical attempt before any send can occur."""

        if retry_number not in (0, 1, 2):
            raise PermitStateError("retry number must be 0, 1, or 2")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None or run["state"] != RunState.ACTIVE.value:
                raise LedgerUnavailable("run is not active in the durable ledger")
            if run["circuit_state"] != "closed":
                raise CircuitOpen("provider circuit is open for this run")
            planned = connection.execute(
                """
                SELECT * FROM planned_requests
                WHERE run_id = ? AND logical_request_id = ?
                """,
                (run_id, logical_request_id),
            ).fetchone()
            if planned is None:
                raise PermitStateError("request ID is not in the frozen plan")
            if retry_number > int(planned["retry_limit"]):
                raise BudgetExhausted("per-request retry limit reached")
            previous_count = connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            next_attempt = int(previous_count) + 1
            admitted_cap = min(
                int(run["hard_cap"]),
                int(run["admitted_attempt_cap"]),
                PROTOCOL_MAX_ATTEMPTS,
            )
            if next_attempt > admitted_cap or next_attempt >= 100:
                raise BudgetExhausted("admitted outbound-attempt ceiling exhausted")
            campaign = connection.execute(
                """
                SELECT campaigns.hard_cap, campaigns.charged_attempts
                  FROM runs JOIN campaigns USING(campaign_id)
                 WHERE runs.run_id = ?
                """,
                (run_id,),
            ).fetchone()
            if campaign is None:
                raise LedgerUnavailable("run campaign is missing from the durable ledger")
            if (
                int(campaign["charged_attempts"]) >= int(campaign["hard_cap"])
                or int(campaign["charged_attempts"]) >= PROTOCOL_MAX_ATTEMPTS
            ):
                raise BudgetExhausted("cumulative campaign attempt ceiling exhausted")
            response_byte_cap = int(run["response_byte_cap"])
            response_reservation = int(planned["max_response_bytes"])
            if response_byte_cap:
                committed_or_reserved = connection.execute(
                    """
                    SELECT COALESCE(SUM(
                        CASE WHEN bytes_returned IS NOT NULL THEN bytes_returned
                             ELSE response_byte_reservation END
                    ), 0)
                      FROM attempts WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()[0]
                if int(committed_or_reserved) + response_reservation > response_byte_cap:
                    raise ResponseBudgetExhausted(
                        "EOD total-response byte envelope exhausted"
                    )
            if retry_number > 0:
                prior = connection.execute(
                    """
                    SELECT permit_id FROM attempts
                    WHERE run_id = ? AND logical_request_id = ? AND retry_number = ?
                    """,
                    (run_id, logical_request_id, retry_number - 1),
                ).fetchone()
                if prior is None:
                    raise PermitStateError("retry sequence cannot skip an attempt")
                retries_used = connection.execute(
                    "SELECT COUNT(*) FROM attempts WHERE run_id = ? AND retry_number > 0",
                    (run_id,),
                ).fetchone()[0]
                if int(retries_used) >= int(run["retry_reserve"]):
                    raise BudgetExhausted("global retry reserve exhausted")
            now = time.time()
            cursor = connection.execute(
                """
                INSERT INTO attempts(
                    run_id, logical_request_id, network_attempt_number,
                    retry_number, endpoint, method, request_fingerprint,
                    response_byte_reservation, state, reserved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?)
                """,
                (
                    run_id,
                    logical_request_id,
                    next_attempt,
                    retry_number,
                    planned["endpoint"],
                    planned["method"],
                    planned["request_fingerprint"],
                    response_reservation,
                    now,
                ),
            )
            permit_id = int(cursor.lastrowid)
            connection.execute("COMMIT")
            return AttemptPermit(
                permit_id=permit_id,
                run_id=run_id,
                logical_request_id=logical_request_id,
                network_attempt_number=next_attempt,
                retry_number=retry_number,
                endpoint=str(planned["endpoint"]),
                method=str(planned["method"]),
                request_fingerprint=str(planned["request_fingerprint"]),
                response_byte_reservation=response_reservation,
                reserved_at=now,
            )
        except (BudgetExhausted, PermitStateError, LedgerUnavailable):
            self._rollback_quietly(connection)
            raise
        except sqlite3.IntegrityError as exc:
            self._rollback_quietly(connection)
            if "campaign_budget_exhausted" in str(exc):
                raise BudgetExhausted(
                    "cumulative campaign attempt ceiling exhausted"
                ) from exc
            raise PermitStateError("attempt permit already exists and cannot be reused") from exc
        except sqlite3.Error as exc:
            self._rollback_quietly(connection)
            raise LedgerUnavailable("could not reserve a durable attempt permit") from exc
        finally:
            connection.close()

    def _transition(
        self,
        permit: AttemptPermit,
        *,
        expected_state: AttemptState,
        next_state: AttemptState,
        assignments: Dict[str, Any],
    ) -> None:
        allowed_columns = {
            "send_started_at",
            "confirmed_at",
            "completed_at",
            "status_code",
            "rows_returned",
            "bytes_returned",
            "duration_ms",
            "provider_trade_date",
            "updated_at_min",
            "updated_at_max",
            "outcome_code",
        }
        if not set(assignments).issubset(allowed_columns):
            raise PermitStateError("unsupported ledger assignment")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            columns = ["state = ?"]
            values: List[Any] = [next_state.value]
            for key in sorted(assignments):
                columns.append("%s = ?" % key)
                values.append(assignments[key])
            values.extend([permit.permit_id, permit.run_id, expected_state.value])
            cursor = connection.execute(
                "UPDATE attempts SET %s WHERE permit_id = ? AND run_id = ? AND state = ?"
                % ", ".join(columns),
                values,
            )
            if cursor.rowcount != 1:
                raise PermitStateError("attempt state transition is invalid or duplicated")
            connection.execute("COMMIT")
        except PermitStateError:
            self._rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            self._rollback_quietly(connection)
            raise LedgerUnavailable("could not record attempt state") from exc
        finally:
            connection.close()

    def mark_indeterminate(self, permit: AttemptPermit) -> None:
        """Record that transport may have put bytes on the wire."""

        self._transition(
            permit,
            expected_state=AttemptState.RESERVED,
            next_state=AttemptState.INDETERMINATE,
            assignments={"send_started_at": time.time()},
        )

    def mark_confirmed(
        self,
        permit: AttemptPermit,
        *,
        status_code: int,
        rows_returned: int,
        bytes_returned: int,
        duration_ms: float,
        provider_trade_date: Optional[str] = None,
        updated_at_min: Optional[str] = None,
        updated_at_max: Optional[str] = None,
    ) -> None:
        if rows_returned < 0 or bytes_returned < 0 or duration_ms < 0:
            raise PermitStateError("response telemetry cannot be negative")
        if bytes_returned > permit.response_byte_reservation:
            raise PermitStateError("response exceeds its atomically reserved byte ceiling")
        self._transition(
            permit,
            expected_state=AttemptState.INDETERMINATE,
            next_state=AttemptState.CONFIRMED,
            assignments={
                "confirmed_at": time.time(),
                "status_code": int(status_code),
                "rows_returned": int(rows_returned),
                "bytes_returned": int(bytes_returned),
                "duration_ms": float(duration_ms),
                "provider_trade_date": provider_trade_date,
                "updated_at_min": updated_at_min,
                "updated_at_max": updated_at_max,
            },
        )

    def mark_completed(self, permit: AttemptPermit, *, outcome_code: str) -> None:
        self._validate_identifier(outcome_code, "outcome code")
        self._transition(
            permit,
            expected_state=AttemptState.CONFIRMED,
            next_state=AttemptState.COMPLETED,
            assignments={"completed_at": time.time(), "outcome_code": outcome_code},
        )

    def record_provider_result(self, run_id: str, *, success: bool) -> None:
        """Update the shared provider circuit after a confirmed success/failure."""

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            run = connection.execute(
                "SELECT state, consecutive_provider_failures FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if run is None or run["state"] != RunState.ACTIVE.value:
                raise LedgerUnavailable("run is not active in the durable ledger")
            if success:
                connection.execute(
                    """
                    UPDATE runs SET consecutive_provider_failures = 0,
                                    circuit_state = 'closed'
                     WHERE run_id = ?
                    """,
                    (run_id,),
                )
            else:
                failures = int(run["consecutive_provider_failures"]) + 1
                state = (
                    "open"
                    if failures >= PROVIDER_CIRCUIT_FAILURE_THRESHOLD
                    else "closed"
                )
                connection.execute(
                    """
                    UPDATE runs SET consecutive_provider_failures = ?, circuit_state = ?
                     WHERE run_id = ?
                    """,
                    (failures, state, run_id),
                )
            connection.execute("COMMIT")
        except LedgerError:
            self._rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            self._rollback_quietly(connection)
            raise LedgerUnavailable("could not update provider circuit") from exc
        finally:
            connection.close()

    def finish_run(self, run_id: str, *, aborted: bool = False) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE runs SET state = ?, ended_at = ?
                WHERE run_id = ? AND state = 'active'
                """,
                (
                    RunState.ABORTED.value if aborted else RunState.COMPLETED.value,
                    time.time(),
                    run_id,
                ),
            )
            if cursor.rowcount != 1:
                raise LedgerError("run is not active")
            connection.execute("COMMIT")
        except LedgerError:
            self._rollback_quietly(connection)
            raise
        except sqlite3.Error as exc:
            self._rollback_quietly(connection)
            raise LedgerUnavailable("could not finish request-ledger run") from exc
        finally:
            connection.close()

    def summary(self, run_id: str) -> Dict[str, Any]:
        connection = self._connect()
        try:
            run = connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise LedgerError("unknown run")
            state_rows = connection.execute(
                """
                SELECT state, COUNT(*) AS count FROM attempts
                WHERE run_id = ? GROUP BY state
                """,
                (run_id,),
            ).fetchall()
            charged = connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            actual_logical = connection.execute(
                "SELECT COUNT(DISTINCT logical_request_id) FROM attempts WHERE run_id = ?",
                (run_id,),
            ).fetchone()[0]
            planned_logical = connection.execute(
                "SELECT COUNT(*) FROM planned_requests WHERE run_id = ?", (run_id,)
            ).fetchone()[0]
            retries = connection.execute(
                "SELECT COUNT(*) FROM attempts WHERE run_id = ? AND retry_number > 0",
                (run_id,),
            ).fetchone()[0]
            totals = connection.execute(
                """
                SELECT COALESCE(SUM(rows_returned), 0),
                       COALESCE(SUM(bytes_returned), 0),
                       COALESCE(SUM(
                           CASE WHEN attempts.endpoint = '/datav2/strikes/options'
                                THEN planned_requests.entity_count ELSE 0 END
                       ), 0),
                       COALESCE(SUM(
                           CASE WHEN attempts.endpoint <> '/datav2/strikes/options'
                                THEN planned_requests.entity_count ELSE 0 END
                       ), 0)
                  FROM attempts
                  JOIN planned_requests
                    ON planned_requests.run_id = attempts.run_id
                   AND planned_requests.logical_request_id = attempts.logical_request_id
                 WHERE attempts.run_id = ?
                """,
                (run_id,),
            ).fetchone()
            admitted_cap = min(
                int(run["hard_cap"]), int(run["admitted_attempt_cap"])
            )
            byte_usage = connection.execute(
                """
                SELECT COALESCE(SUM(
                    CASE WHEN bytes_returned IS NOT NULL THEN bytes_returned
                         ELSE response_byte_reservation END
                ), 0)
                  FROM attempts WHERE run_id = ?
                """,
                (run_id,),
            ).fetchone()[0]
            return {
                "run_id": run_id,
                "account_scope": str(run["credential_key"]),
                "campaign_id": str(run["campaign_id"]),
                "run_type": run["run_type"],
                "state": run["state"],
                "plan_hash": run["plan_hash"],
                "target": int(run["target"]),
                "hard_cap": int(run["hard_cap"]),
                "admitted_attempt_cap": admitted_cap,
                "retry_reserve": int(run["retry_reserve"]),
                "planned_logical_requests": int(planned_logical),
                "actual_logical_requests": int(actual_logical),
                "charged_attempts": int(charged),
                "outbound_http_attempts": int(charged),
                "retries": int(retries),
                "redirects": 0,
                "rows_returned": int(totals[0]),
                "rows_downloaded": int(totals[0]),
                "bytes_returned": int(totals[1]),
                "total_response_bytes": int(totals[1]),
                "response_byte_cap": int(run["response_byte_cap"]),
                "response_bytes_committed_or_reserved": int(byte_usage),
                "response_bytes_remaining": max(
                    0, int(run["response_byte_cap"]) - int(byte_usage)
                ) if int(run["response_byte_cap"]) else 0,
                "provider_circuit_state": str(run["circuit_state"]),
                "consecutive_provider_failures": int(
                    run["consecutive_provider_failures"]
                ),
                "contracts_requested": int(totals[2]),
                "symbols_requested": int(totals[3]),
                "missing_symbol_recoveries": 0,
                "remaining": max(0, admitted_cap - int(charged)),
                "campaign": self._campaign_summary(connection, str(run["campaign_id"])),
                "attempt_states": {str(row["state"]): int(row["count"]) for row in state_rows},
            }
        except sqlite3.Error as exc:
            raise LedgerUnavailable("could not read request-ledger summary") from exc
        finally:
            connection.close()

    @staticmethod
    def _campaign_summary(
        connection: sqlite3.Connection, campaign_id: str
    ) -> Dict[str, Any]:
        row = connection.execute(
            "SELECT * FROM campaigns WHERE campaign_id = ?", (campaign_id,)
        ).fetchone()
        if row is None:
            raise LedgerUnavailable("unknown request campaign")
        hard_cap = int(row["hard_cap"])
        charged = int(row["charged_attempts"])
        return {
            "campaign_id": campaign_id,
            "hard_cap": hard_cap,
            "charged_attempts": charged,
            "remaining": max(0, hard_cap - charged),
        }

    def campaign_summary(self, campaign_id: str) -> Dict[str, Any]:
        """Return cumulative accounting shared by every campaign phase."""

        self._validate_identifier(campaign_id, "campaign ID")
        connection = self._connect()
        try:
            return self._campaign_summary(connection, campaign_id)
        except sqlite3.Error as exc:
            raise LedgerUnavailable("could not read request-campaign summary") from exc
        finally:
            connection.close()

    def export(self, run_id: str, destination: Path) -> Path:
        """Write a secret-free JSON ledger artifact atomically."""

        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT attempts.logical_request_id,
                       attempts.network_attempt_number,
                       attempts.retry_number,
                       attempts.endpoint,
                       attempts.method,
                       attempts.request_fingerprint,
                       attempts.response_byte_reservation,
                       attempts.state,
                       attempts.reserved_at,
                       attempts.send_started_at,
                       attempts.confirmed_at,
                       attempts.completed_at,
                       attempts.status_code,
                       attempts.rows_returned,
                       attempts.bytes_returned,
                       attempts.duration_ms,
                       attempts.provider_trade_date,
                       attempts.updated_at_min,
                       attempts.updated_at_max,
                       attempts.outcome_code,
                       planned_requests.entity_count,
                       planned_requests.field_profile,
                       runs.admitted_attempt_cap
                  FROM attempts
                  JOIN planned_requests
                    ON planned_requests.run_id = attempts.run_id
                   AND planned_requests.logical_request_id = attempts.logical_request_id
                  JOIN runs ON runs.run_id = attempts.run_id
                 WHERE attempts.run_id = ?
                 ORDER BY attempts.network_attempt_number
                """,
                (run_id,),
            ).fetchall()
        except sqlite3.Error as exc:
            raise LedgerUnavailable("could not export request ledger") from exc
        finally:
            connection.close()
        exported_attempts = []
        for row in rows:
            item = dict(row)
            admitted_cap = int(item.pop("admitted_attempt_cap"))
            item["cache_status"] = "MISS"
            item["cache_hit"] = False
            if item["endpoint"] == "/datav2/strikes/options":
                item["contract_count"] = int(item["entity_count"])
                item["symbol_count"] = 0
            else:
                item["contract_count"] = 0
                item["symbol_count"] = int(item["entity_count"])
            item["budget_remaining"] = max(
                0, admitted_cap - int(item["network_attempt_number"])
            )
            exported_attempts.append(item)
        payload = {
            "schema_version": "CULTRA_LEDGER_EXPORT_V2",
            "summary": self.summary(run_id),
            "attempts": exported_attempts,
        }
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = destination.with_name(destination.name + ".tmp-%d" % os.getpid())
        data = json.dumps(payload, sort_keys=True, indent=2).encode("utf-8") + b"\n"
        try:
            with open(temporary, "xb") as handle:
                os.chmod(temporary, 0o600)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return destination
