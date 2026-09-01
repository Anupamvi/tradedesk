"""Strict normalization of a completed Cultra V2 historical campaign."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .cache import CULTRA_CACHE_ROOT, ContentAddressedCache
from .campaign import load_historical_campaign_freeze
from .campaign_completion import load_historical_campaign_completion
from .historical_events import load_historical_event_manifest
from .request_optimization import (
    HISTORICAL_CORE_FIELDS,
    HISTORICAL_SPLIT_FIELDS,
    HISTORICAL_STRIKE_FIELDS,
)
from .requesting import Endpoint, PlannedRequest
from .sessions import load_historical_session_calendar


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_ROOT = (PROJECT_ROOT / "var" / "historical_v2").resolve()

CORE_NUMERIC_FIELDS = tuple(
    field
    for field in HISTORICAL_CORE_FIELDS
    if field not in {"ticker", "tradeDate", "updatedAt"}
)
CHAIN_COLUMN_MAP = {
    "callAskPrice": "call_ask",
    "callBidIv": "call_bid_iv",
    "callBidPrice": "call_bid",
    "callMidIv": "call_mid_iv",
    "callOpenInterest": "call_open_interest",
    "callVolume": "call_volume",
    "delta": "call_delta",
    "dte": "dte",
    "gamma": "gamma",
    "putAskPrice": "put_ask",
    "putBidIv": "put_bid_iv",
    "putBidPrice": "put_bid",
    "putMidIv": "put_mid_iv",
    "putOpenInterest": "put_open_interest",
    "putVolume": "put_volume",
    "rho": "rho",
    "smvVol": "smv_vol",
    "stockPrice": "stock_price",
    "strike": "strike",
    "theta": "theta",
    "vega": "vega",
}
CHAIN_INTEGER_FIELDS = {
    "callOpenInterest",
    "callVolume",
    "dte",
    "putOpenInterest",
    "putVolume",
}


class HistoricalV2Error(RuntimeError):
    """Completed historical data cannot enter the V2 evidence store."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _decode_rows(raw: bytes) -> Tuple[Mapping[str, Any], ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HistoricalV2Error("historical snapshot is not UTF-8") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            rows = tuple(dict(item) for item in csv.DictReader(io.StringIO(text)))
        except csv.Error as exc:
            raise HistoricalV2Error("historical snapshot is not valid JSON or CSV") from exc
        if not rows:
            raise HistoricalV2Error("historical CSV snapshot has no records")
        return rows
    if isinstance(value, list):
        raw_rows = value
    elif isinstance(value, Mapping):
        containers = [key for key in ("data", "rows", "records", "results") if key in value]
        if len(containers) > 1:
            raise HistoricalV2Error("historical snapshot has ambiguous row containers")
        if containers:
            raw_rows = value[containers[0]]
        else:
            raw_rows = [value]
    else:
        raise HistoricalV2Error("historical snapshot root is invalid")
    if not isinstance(raw_rows, list):
        raise HistoricalV2Error("historical snapshot row container is invalid")
    rows = []
    for item in raw_rows:
        if not isinstance(item, Mapping):
            raise HistoricalV2Error("historical snapshot contains a non-object row")
        if any(isinstance(nested, (Mapping, list, tuple, set)) for nested in item.values()):
            raise HistoricalV2Error("historical row contains a nested value")
        rows.append(item)
    return tuple(rows)


def _row_contract(row: Mapping[str, Any], request: PlannedRequest) -> None:
    if set(row) != set(request.fields):
        raise HistoricalV2Error("historical row does not match its frozen field profile")


def _ticker(value: Any, entities: Sequence[str]) -> str:
    normalized = str(value).strip().upper()
    if normalized not in set(entities):
        raise HistoricalV2Error("historical row ticker leaves the planned request")
    return normalized


def _date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise HistoricalV2Error("%s is not YYYY-MM-DD" % label) from exc


def _timestamp(value: Any, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise HistoricalV2Error("%s is not an ISO timestamp" % label) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HistoricalV2Error("%s must be timezone-aware" % label)
    return parsed


def _contemporaneous_timestamp(value: Any, trade_date: date, label: str) -> datetime:
    parsed = _timestamp(value, label)
    # A US close can be represented on tradeDate or, after timezone
    # normalization/provider processing, the next UTC calendar day.  Anything
    # later could be a revised/future snapshot and is rejected from evidence.
    if parsed.date() < trade_date or parsed.date() > trade_date + timedelta(days=1):
        raise HistoricalV2Error("%s is not contemporaneous with tradeDate" % label)
    return parsed


def _number(value: Any, label: str, *, nullable: bool = True) -> Optional[float]:
    if value in (None, ""):
        if nullable:
            return None
        raise HistoricalV2Error("%s is missing" % label)
    if isinstance(value, bool):
        raise HistoricalV2Error("%s is not numeric" % label)
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise HistoricalV2Error("%s is not numeric" % label) from exc
    if not math.isfinite(converted):
        raise HistoricalV2Error("%s is not finite" % label)
    return converted


def _integer(value: Any, label: str, *, nullable: bool = True) -> Optional[int]:
    converted = _number(value, label, nullable=nullable)
    if converted is None:
        return None
    if not float(converted).is_integer():
        raise HistoricalV2Error("%s is not an integer" % label)
    result = int(converted)
    if result < 0:
        raise HistoricalV2Error("%s cannot be negative" % label)
    return result


def normalize_core_row(
    row: Mapping[str, Any], request: PlannedRequest, snapshot_id: str
) -> Mapping[str, Any]:
    _row_contract(row, request)
    trade_date = _date(row["tradeDate"], "Core tradeDate")
    if trade_date > _date(request.expected_vintage, "Core through date"):
        raise HistoricalV2Error("Core row exceeds the frozen through-date")
    result: Dict[str, Any] = {
        "ticker": _ticker(row["ticker"], request.entities),
        "trade_date": trade_date.isoformat(),
        "updated_at": _contemporaneous_timestamp(
            row["updatedAt"], trade_date, "Core updatedAt"
        ).isoformat(),
        "snapshot_id": str(snapshot_id),
    }
    for field in CORE_NUMERIC_FIELDS:
        result[field] = _number(row[field], field)
    if result["priorCls"] is not None and result["priorCls"] <= 0.0:
        raise HistoricalV2Error("Core priorCls must be positive when present")
    return result


def normalize_chain_row(
    row: Mapping[str, Any], request: PlannedRequest, snapshot_id: str
) -> Mapping[str, Any]:
    _row_contract(row, request)
    trade_date = _date(row["tradeDate"], "chain tradeDate")
    if trade_date.isoformat() != request.expected_vintage:
        raise HistoricalV2Error("chain row does not match the planned trade date")
    expiration = _date(row["expirDate"], "chain expiration")
    result: Dict[str, Any] = {
        "ticker": _ticker(row["ticker"], request.entities),
        "trade_date": trade_date.isoformat(),
        "expiration": expiration.isoformat(),
        "updated_at": _contemporaneous_timestamp(
            row["updatedAt"], trade_date, "chain updatedAt"
        ).isoformat(),
        "snapshot_id": str(snapshot_id),
    }
    for provider_name, column in CHAIN_COLUMN_MAP.items():
        if provider_name in CHAIN_INTEGER_FIELDS:
            result[column] = _integer(row[provider_name], provider_name)
        else:
            result[column] = _number(row[provider_name], provider_name)
    dte = result["dte"]
    if dte is None or not 20 <= dte <= 180:
        raise HistoricalV2Error("chain DTE leaves the frozen 20-180 window")
    if abs((expiration - trade_date).days - dte) > 1:
        raise HistoricalV2Error("chain DTE does not reconcile to expiration")
    for field in ("strike", "stock_price"):
        if result[field] is None or result[field] <= 0.0:
            raise HistoricalV2Error("chain %s must be positive" % field)
    if result["call_delta"] is None or not 0.0 <= result["call_delta"] <= 1.0:
        raise HistoricalV2Error("chain call delta is outside zero to one")
    for bid, ask in (("call_bid", "call_ask"), ("put_bid", "put_ask")):
        if result[bid] is not None and result[bid] < 0.0:
            raise HistoricalV2Error("chain bid cannot be negative")
        if result[ask] is not None and result[ask] < 0.0:
            raise HistoricalV2Error("chain ask cannot be negative")
        if (
            result[bid] is not None
            and result[ask] is not None
            and result[ask] < result[bid]
        ):
            raise HistoricalV2Error("chain ask cannot be below bid")
    for field in ("call_bid_iv", "call_mid_iv", "put_bid_iv", "put_mid_iv", "smv_vol"):
        if result[field] is not None and result[field] < 0.0:
            raise HistoricalV2Error("chain volatility cannot be negative")
    return result


def normalize_split_row(
    row: Mapping[str, Any], request: PlannedRequest, snapshot_id: str
) -> Mapping[str, Any]:
    _row_contract(row, request)
    divisor = _number(row["divisor"], "split divisor", nullable=False)
    assert divisor is not None
    if divisor <= 0.0 or math.isclose(divisor, 1.0):
        raise HistoricalV2Error("split divisor is invalid")
    return {
        "ticker": _ticker(row["ticker"], request.entities),
        "split_date": _date(row["splitDate"], "split date").isoformat(),
        "divisor": divisor,
        "snapshot_id": str(snapshot_id),
    }


def _create_schema(connection: sqlite3.Connection) -> None:
    core_columns = ",\n".join('"%s" REAL' % field for field in CORE_NUMERIC_FIELDS)
    chain_columns = ",\n".join(
        '"%s" %s'
        % (column, "INTEGER" if provider in CHAIN_INTEGER_FIELDS else "REAL")
        for provider, column in CHAIN_COLUMN_MAP.items()
    )
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=FULL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE sessions(
            session_index INTEGER PRIMARY KEY,
            session_date TEXT NOT NULL UNIQUE,
            close_at TEXT NOT NULL
        );
        CREATE TABLE core_features(
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            %s,
            PRIMARY KEY(ticker, trade_date)
        );
        CREATE TABLE chain_quotes(
            ticker TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            expiration TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            %s,
            PRIMARY KEY(ticker, trade_date, expiration, strike)
        );
        CREATE INDEX chain_by_date_ticker
            ON chain_quotes(trade_date, ticker);
        CREATE TABLE split_history(
            ticker TEXT NOT NULL,
            split_date TEXT NOT NULL,
            divisor REAL NOT NULL,
            snapshot_id TEXT NOT NULL,
            PRIMARY KEY(ticker, split_date)
        );
        CREATE TABLE historical_events(
            ticker TEXT NOT NULL,
            event_type TEXT NOT NULL,
            effective_date TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            available_at TEXT NOT NULL,
            source_event_id TEXT NOT NULL,
            status TEXT NOT NULL,
            cash_amount REAL,
            split_ratio REAL,
            adjustment_reference TEXT,
            PRIMARY KEY(ticker, source_event_id, available_at)
        );
        CREATE TABLE request_snapshots(
            logical_request_id TEXT PRIMARY KEY,
            endpoint TEXT NOT NULL,
            expected_vintage TEXT NOT NULL,
            field_profile TEXT NOT NULL,
            request_fingerprint TEXT NOT NULL,
            snapshot_id TEXT NOT NULL UNIQUE,
            raw_sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL
        );
        """
        % (core_columns, chain_columns)
    )


def _insert_mapping(
    connection: sqlite3.Connection, table: str, row: Mapping[str, Any]
) -> None:
    columns = tuple(row)
    sql = "INSERT INTO %s(%s) VALUES (%s)" % (
        table,
        ",".join('"%s"' % item for item in columns),
        ",".join("?" for _item in columns),
    )
    try:
        connection.execute(sql, tuple(row[item] for item in columns))
    except sqlite3.IntegrityError as exc:
        raise HistoricalV2Error("duplicate or invalid normalized historical row") from exc


def _private_json(path: Path, value: Mapping[str, Any]) -> Path:
    encoded = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with open(path, "xb") as handle:
        os.chmod(path, 0o600)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def ingest_historical_v2_campaign(
    *,
    campaign_completion_path: Path,
    database_path: Path,
    cache_root: Optional[Path] = None,
) -> Mapping[str, Any]:
    """Build a new immutable normalized store from all verified snapshots."""

    completion = load_historical_campaign_completion(
        campaign_completion_path, cache_root=cache_root
    )
    campaign = load_historical_campaign_freeze(
        Path(str(completion["campaign_freeze_path"]))
    )
    inputs = campaign.payload["inputs"]
    calendar = load_historical_session_calendar(Path(inputs["sessions"]["path"]))
    events = load_historical_event_manifest(Path(inputs["events"]["path"]))
    destination = Path(database_path).expanduser().resolve()
    try:
        destination.relative_to(HISTORICAL_ROOT)
    except ValueError as exc:
        raise HistoricalV2Error(
            "historical V2 database must remain inside Cultra/var/historical_v2"
        ) from exc
    if destination.exists() or destination.with_suffix(destination.suffix + ".manifest.json").exists():
        raise HistoricalV2Error("historical V2 output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(destination.parent, 0o700)
    temporary = destination.with_name(".%s.tmp-%d" % (destination.name, os.getpid()))
    cache = ContentAddressedCache(
        Path(cache_root) if cache_root is not None else CULTRA_CACHE_ROOT / "historical"
    )
    connection = sqlite3.connect(str(temporary))
    try:
        _create_schema(connection)
        for index, item in enumerate(calendar.sessions):
            connection.execute(
                "INSERT INTO sessions VALUES (?, ?, ?)",
                (index, item.session_date.isoformat(), item.close_at.isoformat()),
            )
        for event in events.records:
            connection.execute(
                """
                INSERT INTO historical_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.ticker,
                    event.event_type,
                    event.effective_date.isoformat(),
                    event.observed_at.isoformat(),
                    event.available_at.isoformat(),
                    event.source_event_id,
                    event.status,
                    event.cash_amount,
                    event.split_ratio,
                    event.adjustment_reference,
                ),
            )
        selected_dates = set(calendar.dates)
        sampled = {
            ticker
            for block in json.loads(
                Path(inputs["cohorts"]["path"]).read_text(encoding="utf-8")
            )["blocks"]
            for ticker in block["tickers"]
        }
        chain_coverage: Dict[Tuple[str, str], int] = {}
        for plan in campaign.slices:
            partition = json.loads(
                (Path(str(completion["runs_root"])) / plan.run_id / "partition_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            for request in plan.requests:
                completed = partition["completed_requests"][request.logical_request_id]
                manifest, raw = cache.load_snapshot(completed["snapshot_id"])
                rows = _decode_rows(raw)
                if len(rows) != manifest.row_count:
                    raise HistoricalV2Error("snapshot row count changed during normalization")
                connection.execute(
                    "INSERT INTO request_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        request.logical_request_id,
                        request.endpoint.value,
                        request.expected_vintage,
                        request.field_profile,
                        request.fingerprint,
                        manifest.snapshot_id,
                        manifest.raw_sha256,
                        manifest.row_count,
                    ),
                )
                if request.endpoint is Endpoint.HIST_CORES:
                    for raw_row in rows:
                        row = normalize_core_row(raw_row, request, manifest.snapshot_id)
                        if row["trade_date"] in selected_dates:
                            _insert_mapping(connection, "core_features", row)
                elif request.endpoint is Endpoint.HIST_STRIKES:
                    covered = set()
                    for raw_row in rows:
                        row = normalize_chain_row(raw_row, request, manifest.snapshot_id)
                        _insert_mapping(connection, "chain_quotes", row)
                        covered.add(row["ticker"])
                    if covered != set(request.entities):
                        raise HistoricalV2Error(
                            "daily chain snapshot lacks one or more cohort symbols"
                        )
                    for ticker in covered:
                        chain_coverage[(request.expected_vintage, ticker)] = 1
                elif request.endpoint is Endpoint.HIST_SPLITS:
                    for raw_row in rows:
                        _insert_mapping(
                            connection,
                            "split_history",
                            normalize_split_row(raw_row, request, manifest.snapshot_id),
                        )
                else:
                    raise HistoricalV2Error("campaign contains an unsupported endpoint")
        if len(chain_coverage) != len(calendar.sessions) * 10:
            raise HistoricalV2Error("normalized daily cohort coverage is incomplete")
        core_tickers = {
            str(row[0])
            for row in connection.execute("SELECT DISTINCT ticker FROM core_features")
        }
        if core_tickers != sampled:
            raise HistoricalV2Error("normalized Core coverage does not include every sampled name")
        first_date, last_date = calendar.sessions[0].session_date, calendar.sessions[-1].session_date
        orats_splits = {
            (str(row[0]), str(row[1]))
            for row in connection.execute(
                "SELECT ticker, split_date FROM split_history WHERE split_date BETWEEN ? AND ?",
                (first_date.isoformat(), last_date.isoformat()),
            )
        }
        event_splits = {
            (ticker, event.effective_date.isoformat())
            for ticker in sorted(sampled)
            for event in events.events_in_window(
                ticker=ticker, start_date=first_date, end_date=last_date
            )
            if event.event_type == "SPLIT"
        }
        if orats_splits != event_splits:
            raise HistoricalV2Error(
                "ORATS split dates do not reconcile to the independent event manifest"
            )
        metadata = {
            "schema": "cultra.normalized-historical-v2.v1",
            "campaign_id": campaign.campaign_id,
            "campaign_freeze_hash": campaign.payload["freeze_hash"],
            "campaign_completion_sha256": _sha256(campaign_completion_path),
            "session_calendar_hash": calendar.calendar_hash,
            "event_manifest_hash": events.manifest_hash,
            "network_attempted": False,
        }
        for key, value in metadata.items():
            connection.execute(
                "INSERT INTO metadata VALUES (?, ?)",
                (key, json.dumps(value, sort_keys=True)),
            )
        connection.commit()
        check = connection.execute("PRAGMA integrity_check").fetchone()
        if check is None or check[0] != "ok":
            raise HistoricalV2Error("normalized database failed integrity check")
        counts = {
            table: int(connection.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0])
            for table in (
                "sessions",
                "core_features",
                "chain_quotes",
                "split_history",
                "historical_events",
                "request_snapshots",
            )
        }
    except BaseException:
        connection.rollback()
        connection.close()
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    connection.close()
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    result = {
        "schema": "cultra.normalized-historical-v2-manifest.v1",
        "campaign_id": campaign.campaign_id,
        "campaign_freeze_hash": campaign.payload["freeze_hash"],
        "campaign_completion": str(Path(campaign_completion_path).expanduser().resolve()),
        "database": str(destination),
        "database_bytes": destination.stat().st_size,
        "database_sha256": _sha256(destination),
        "counts": counts,
        "network_attempted": False,
    }
    _private_json(manifest_path, result)
    return dict(result, manifest=str(manifest_path))


__all__ = [
    "HISTORICAL_ROOT",
    "HistoricalV2Error",
    "ingest_historical_v2_campaign",
    "normalize_chain_row",
    "normalize_core_row",
    "normalize_split_row",
]
