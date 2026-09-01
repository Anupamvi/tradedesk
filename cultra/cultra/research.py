"""Cultra V1 exact-leg historical research and one-time holdout evaluation.

This module consumes only Cultra-owned ORATS snapshots.  Strategy rules and
the pre-outcome amendment are immutable inputs.  Historical trade selection
uses entry-date information only; exit quotes are joined by exact expiry and
strike and are never reconstructed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .backfill import DEFAULT_VALIDATION_CONFIG, load_recent_sessions
from .cache import CULTRA_CACHE_ROOT
from .catalog import CATALOG_VERSION, FROZEN_STRATEGY_CATALOG
from .domain import HistoricalObservation
from .evidence_registry import (
    DEFAULT_EVIDENCE_ROOT,
    EvidencePartitions,
    EvidenceRegistry,
    FrozenEvidenceIdentity,
    RegistryState,
)
from .pop import (
    OOFPOPModelArtifact,
    OOFPOPObservation,
    POPBucketIdentity,
    ProbabilityTarget,
    build_oof_pop_model,
)
from .statistics import holm_adjust_mapping, two_way_clustered_bootstrap_mean_ci
from .validation import chronological_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_PATH = (
    PROJECT_ROOT / "configs" / "historical_validation.amendment.v1.1.json"
)
HISTORICAL_ROOT = (PROJECT_ROOT / "var" / "historical").resolve()
DEFAULT_CHAIN_DB = HISTORICAL_ROOT / "cultra_chains_v1.sqlite3"
DEFAULT_SPLIT_REVIEW = (
    PROJECT_ROOT
    / "out"
    / "cultra-backfill-splits-2026-08-30-v1"
    / "split_review.json"
)
DEFAULT_EVIDENCE_DB = DEFAULT_EVIDENCE_ROOT / "cultra_historical_v1_1.sqlite3"
BACKFILL_RUN_PREFIX = "cultra-chain-backfill-v1-slice-"
MODEL_VERSION = "CULTRA_POP_V1"


class ResearchError(RuntimeError):
    """Historical research cannot proceed without violating the frozen rules."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _private_write(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    temporary = path.with_name(".%s.tmp-%d" % (path.name, os.getpid()))
    try:
        with open(temporary, "xb") as handle:
            os.chmod(temporary, 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return path


def _private_json(path: Path, value: Any) -> Path:
    return _private_write(
        path,
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8")
        + b"\n",
    )


def _merge_validation_config() -> Mapping[str, Any]:
    base_raw = DEFAULT_VALIDATION_CONFIG.read_bytes()
    base = json.loads(base_raw.decode("utf-8"))
    amendment = json.loads(AMENDMENT_PATH.read_text(encoding="utf-8"))
    if amendment.get("stage") != "PRE_OUTCOME_INSPECTION":
        raise ResearchError("historical amendment is not marked pre-outcome")
    if amendment.get("base_config_sha256") != _sha256_bytes(base_raw):
        raise ResearchError("historical amendment does not match its immutable base")
    changes = amendment.get("changes", {})
    if set(changes) != {
        "catalog_version",
        "data.entry_dte",
        "data.preferred_dte",
        "data.maximum_unresolved_path_fraction",
    }:
        raise ResearchError("historical amendment contains an unapproved change")
    base["catalog_version"] = str(changes["catalog_version"])
    base["data"]["entry_dte"] = list(changes["data.entry_dte"])
    base["data"]["preferred_dte"] = int(changes["data.preferred_dte"])
    base["data"]["maximum_unresolved_path_fraction"] = float(
        changes["data.maximum_unresolved_path_fraction"]
    )
    base["effective_version"] = amendment["version"]
    base["amendment_sha256"] = _sha256_bytes(AMENDMENT_PATH.read_bytes())
    base["base_sha256"] = _sha256_bytes(base_raw)
    if base["catalog_version"] != CATALOG_VERSION:
        raise ResearchError("historical config catalog version is not current")
    return base


def effective_validation_config() -> Mapping[str, Any]:
    """Return the exact pre-outcome base plus its narrowly scoped amendment."""

    return _merge_validation_config()


def _number(value: Any, name: str, *, optional: bool = False) -> Optional[float]:
    if value is None and optional:
        return None
    if isinstance(value, bool):
        raise ResearchError("provider field %s is not numeric" % name)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        if optional:
            return None
        raise ResearchError("provider field %s is missing" % name) from exc
    if not math.isfinite(result):
        if optional:
            return None
        raise ResearchError("provider field %s is not finite" % name)
    return result


def _integer(value: Any, name: str) -> Optional[int]:
    converted = _number(value, name, optional=True)
    if converted is None:
        return None
    return int(converted)


def _raw_blob_path(raw_sha256: str) -> Path:
    if len(raw_sha256) != 64 or any(char not in "0123456789abcdef" for char in raw_sha256):
        raise ResearchError("historical raw digest is malformed")
    path = CULTRA_CACHE_ROOT / "historical" / "raw" / raw_sha256[:2] / (
        raw_sha256 + ".bin"
    )
    resolved = path.resolve()
    try:
        resolved.relative_to(CULTRA_CACHE_ROOT.resolve())
    except ValueError as exc:
        raise ResearchError("historical raw blob escaped Cultra cache") from exc
    return resolved


def _load_backfill_partitions() -> Mapping[str, Mapping[str, Any]]:
    partitions: Dict[str, Mapping[str, Any]] = {}
    for index in range(6):
        path = (
            PROJECT_ROOT
            / "out"
            / (BACKFILL_RUN_PREFIX + "%02d" % index)
            / "partition_manifest.json"
        )
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ResearchError("historical partition manifest is unavailable") from exc
        if not manifest.get("complete") or manifest.get("failed"):
            raise ResearchError("historical partition is not complete")
        for trade_date, item in manifest.get("completed", {}).items():
            if trade_date in partitions:
                raise ResearchError("historical partition date is duplicated")
            partitions[trade_date] = item
    sessions = load_recent_sessions()
    if set(partitions) != set(sessions):
        raise ResearchError("historical partition coverage does not equal 450 sessions")
    return {key: partitions[key] for key in sorted(partitions)}


def _connect_chain_db(path: Path, *, create: bool) -> sqlite3.Connection:
    supplied = Path(path).expanduser().resolve()
    try:
        supplied.relative_to(HISTORICAL_ROOT)
    except ValueError as exc:
        raise ResearchError("historical database must remain Cultra-local") from exc
    HISTORICAL_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(HISTORICAL_ROOT, 0o700)
    if not create and not supplied.exists():
        raise ResearchError("historical database has not been ingested")
    connection = sqlite3.connect(str(supplied))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=FULL")
    if create:
        connection.execute("PRAGMA journal_mode=WAL")
    return connection


def _initialize_chain_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            trade_date TEXT PRIMARY KEY,
            snapshot_id TEXT NOT NULL,
            raw_sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            raw_bytes INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS underlying (
            trade_date TEXT NOT NULL REFERENCES sessions(trade_date),
            ticker TEXT NOT NULL,
            stock_price REAL NOT NULL,
            PRIMARY KEY (trade_date, ticker)
        );
        CREATE TABLE IF NOT EXISTS chains (
            trade_date TEXT NOT NULL REFERENCES sessions(trade_date),
            ticker TEXT NOT NULL,
            expiry TEXT NOT NULL,
            strike REAL NOT NULL,
            dte INTEGER NOT NULL,
            stock_price REAL NOT NULL,
            call_bid REAL,
            call_ask REAL,
            put_bid REAL,
            put_ask REAL,
            call_mid_iv REAL,
            put_mid_iv REAL,
            smv_vol REAL,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            rho REAL,
            call_volume INTEGER,
            call_open_interest INTEGER,
            put_volume INTEGER,
            put_open_interest INTEGER,
            updated_at TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            PRIMARY KEY (trade_date, ticker, expiry, strike)
        );
        CREATE INDEX IF NOT EXISTS chain_entry_lookup
            ON chains(trade_date, ticker, dte, delta);
        CREATE INDEX IF NOT EXISTS chain_exact_path
            ON chains(ticker, expiry, strike, trade_date);
        """
    )


def ingest_historical_chains(path: Path = DEFAULT_CHAIN_DB) -> Mapping[str, Any]:
    """Normalize and verify all 450 cached raw partitions into immutable SQLite."""

    config = effective_validation_config()
    partitions = _load_backfill_partitions()
    connection = _connect_chain_db(path, create=True)
    _initialize_chain_schema(connection)
    expected_metadata = {
        "schema": "cultra.normalized-historical-chains.v1",
        "base_config_sha256": config["base_sha256"],
        "amendment_sha256": config["amendment_sha256"],
        "provider_vintage_through": config["data"]["provider_vintage_through"],
    }
    existing_metadata = {
        str(row["key"]): str(row["value"])
        for row in connection.execute("SELECT key, value FROM metadata")
    }
    if existing_metadata and existing_metadata != expected_metadata:
        connection.close()
        raise ResearchError("historical database identity differs from frozen config")
    connection.executemany(
        "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
        tuple(sorted(expected_metadata.items())),
    )
    connection.commit()
    universe = set(str(item) for item in config["universe"])
    imported = 0
    skipped = 0
    total_rows = 0
    try:
        for trade_date, item in partitions.items():
            prior = connection.execute(
                "SELECT raw_sha256, row_count FROM sessions WHERE trade_date = ?",
                (trade_date,),
            ).fetchone()
            if prior is not None:
                if (
                    prior["raw_sha256"] != item["raw_sha256"]
                    or int(prior["row_count"]) != int(item["row_count"])
                ):
                    raise ResearchError("immutable historical session changed")
                skipped += 1
                total_rows += int(prior["row_count"])
                continue
            raw_path = _raw_blob_path(str(item["raw_sha256"]))
            raw = raw_path.read_bytes()
            if _sha256_bytes(raw) != item["raw_sha256"]:
                raise ResearchError("historical raw blob hash mismatch")
            try:
                rows = json.loads(raw.decode("utf-8")).get("data", [])
            except (UnicodeError, json.JSONDecodeError, AttributeError) as exc:
                raise ResearchError("historical raw blob is not valid ORATS JSON") from exc
            if not isinstance(rows, list) or len(rows) != int(item["row_count"]):
                raise ResearchError("historical row count does not reconcile")
            normalized = []
            spots: Dict[str, List[float]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ResearchError("historical chain row is not an object")
                ticker = str(row.get("ticker", "")).upper()
                row_date = str(row.get("tradeDate", ""))
                expiry = str(row.get("expirDate", ""))
                if ticker not in universe or row_date != trade_date:
                    raise ResearchError("historical row escaped its frozen partition")
                date.fromisoformat(row_date)
                date.fromisoformat(expiry)
                stock_price = float(_number(row.get("stockPrice"), "stockPrice"))
                strike = float(_number(row.get("strike"), "strike"))
                dte = int(float(_number(row.get("dte"), "dte")))
                if not 20 <= dte <= 60 or strike <= 0.0 or stock_price <= 0.0:
                    raise ResearchError("historical row violates planned DTE/price bounds")
                spots.setdefault(ticker, []).append(stock_price)
                normalized.append(
                    (
                        trade_date,
                        ticker,
                        expiry,
                        strike,
                        dte,
                        stock_price,
                        _number(row.get("callBidPrice"), "callBidPrice", optional=True),
                        _number(row.get("callAskPrice"), "callAskPrice", optional=True),
                        _number(row.get("putBidPrice"), "putBidPrice", optional=True),
                        _number(row.get("putAskPrice"), "putAskPrice", optional=True),
                        _number(row.get("callMidIv"), "callMidIv", optional=True),
                        _number(row.get("putMidIv"), "putMidIv", optional=True),
                        _number(row.get("smvVol"), "smvVol", optional=True),
                        _number(row.get("delta"), "delta", optional=True),
                        _number(row.get("gamma"), "gamma", optional=True),
                        _number(row.get("theta"), "theta", optional=True),
                        _number(row.get("vega"), "vega", optional=True),
                        _number(row.get("rho"), "rho", optional=True),
                        _integer(row.get("callVolume"), "callVolume"),
                        _integer(row.get("callOpenInterest"), "callOpenInterest"),
                        _integer(row.get("putVolume"), "putVolume"),
                        _integer(row.get("putOpenInterest"), "putOpenInterest"),
                        str(row.get("updatedAt", "")),
                        str(item["snapshot_id"]),
                    )
                )
            if set(spots) != universe:
                raise ResearchError("historical partition is missing a frozen universe symbol")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                    (
                        trade_date,
                        item["snapshot_id"],
                        item["raw_sha256"],
                        len(rows),
                        int(item["raw_bytes"]),
                    ),
                )
                for ticker in sorted(spots):
                    values = spots[ticker]
                    if max(values) - min(values) > max(0.05, statistics.median(values) * 0.0001):
                        raise ResearchError("historical chain has inconsistent stock prices")
                    connection.execute(
                        "INSERT INTO underlying VALUES (?, ?, ?)",
                        (trade_date, ticker, statistics.median(values)),
                    )
                connection.executemany(
                    """
                    INSERT INTO chains VALUES (
                        ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
                    )
                    """,
                    normalized,
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            imported += 1
            total_rows += len(rows)
            if imported % 25 == 0:
                print(
                    "INGEST sessions=%d rows=%d" % (imported + skipped, total_rows),
                    flush=True,
                )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    os.chmod(Path(path), 0o600)
    return {
        "database": str(Path(path).resolve()),
        "sessions": len(partitions),
        "rows": total_rows,
        "imported_sessions": imported,
        "reused_sessions": skipped,
        "base_config_sha256": config["base_sha256"],
        "amendment_sha256": config["amendment_sha256"],
    }


@dataclass(frozen=True)
class ChainQuote:
    trade_date: str
    ticker: str
    expiry: str
    strike: float
    dte: int
    stock_price: float
    call_bid: Optional[float]
    call_ask: Optional[float]
    put_bid: Optional[float]
    put_ask: Optional[float]
    smv_vol: Optional[float]
    delta: Optional[float]
    call_open_interest: Optional[int]
    put_open_interest: Optional[int]
    updated_at: str
    snapshot_id: str


@dataclass(frozen=True)
class ResearchLeg:
    action: str
    option_type: str
    expiry: str
    strike: float
    occ_symbol: str
    entry_bid: float
    entry_ask: float
    target_call_delta: float


@dataclass(frozen=True)
class ResearchTrade:
    record_id: str
    strategy_family: str
    ticker: str
    signal_date: str
    entry_date: str
    exit_date: str
    holding_sessions: int
    expiry: str
    legs: Tuple[ResearchLeg, ...]
    momentum_20: float
    realized_volatility_20: float
    smv_vol: float
    relative_spread: float
    raw_probability: float
    entry_debit: float
    maximum_loss: float
    maximum_profit: Optional[float]
    target_pnl: float
    stop_pnl: float
    gross_pnl: float
    commissions_fees: float
    entry_slippage: float
    exit_slippage: float
    net_pnl: float
    exit_reason: str
    target_hit: bool
    stop_hit: bool
    max_loss_hit: bool
    entry_snapshot_id: str
    exit_snapshot_id: str
    exact_path_observations: int
    corporate_action_review: str

    @property
    def observation_id(self) -> str:
        return self.record_id

    @property
    def cluster_id(self) -> str:
        return "%s:%s" % (self.ticker, self.signal_date)

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UnresolvedCandidate:
    candidate_id: str
    strategy_family: str
    ticker: str
    signal_date: str
    entry_date: str
    reason: str

    def to_dict(self) -> Mapping[str, Any]:
        return asdict(self)


def _occ_symbol(ticker: str, expiry: str, option_type: str, strike: float) -> str:
    compact = date.fromisoformat(expiry).strftime("%y%m%d")
    kind = "C" if option_type == "CALL" else "P"
    scaled = int(round(strike * 1000.0))
    if scaled <= 0 or scaled > 99_999_999:
        raise ResearchError("historical strike cannot form an OCC symbol")
    return "%s%s%s%08d" % (ticker, compact, kind, scaled)


def _quote_from_row(row: sqlite3.Row) -> ChainQuote:
    return ChainQuote(
        trade_date=str(row["trade_date"]),
        ticker=str(row["ticker"]),
        expiry=str(row["expiry"]),
        strike=float(row["strike"]),
        dte=int(row["dte"]),
        stock_price=float(row["stock_price"]),
        call_bid=None if row["call_bid"] is None else float(row["call_bid"]),
        call_ask=None if row["call_ask"] is None else float(row["call_ask"]),
        put_bid=None if row["put_bid"] is None else float(row["put_bid"]),
        put_ask=None if row["put_ask"] is None else float(row["put_ask"]),
        smv_vol=None if row["smv_vol"] is None else float(row["smv_vol"]),
        delta=None if row["delta"] is None else float(row["delta"]),
        call_open_interest=(
            None if row["call_open_interest"] is None else int(row["call_open_interest"])
        ),
        put_open_interest=(
            None if row["put_open_interest"] is None else int(row["put_open_interest"])
        ),
        updated_at=str(row["updated_at"]),
        snapshot_id=str(row["snapshot_id"]),
    )


def _bid_ask(quote: ChainQuote, option_type: str) -> Tuple[Optional[float], Optional[float]]:
    if option_type == "CALL":
        return quote.call_bid, quote.call_ask
    return quote.put_bid, quote.put_ask


def _open_interest(quote: ChainQuote, option_type: str) -> Optional[int]:
    return quote.call_open_interest if option_type == "CALL" else quote.put_open_interest


def _relative_spread(quote: ChainQuote, option_type: str) -> Optional[float]:
    bid, ask = _bid_ask(quote, option_type)
    if bid is None or ask is None or bid < 0.0 or ask < bid:
        return None
    midpoint = (bid + ask) / 2.0
    if midpoint <= 0.0:
        return None
    return (ask - bid) / midpoint


def _eligible_quote(
    quote: ChainQuote, option_type: str, config: Mapping[str, Any]
) -> bool:
    bid, ask = _bid_ask(quote, option_type)
    spread = _relative_spread(quote, option_type)
    interest = _open_interest(quote, option_type)
    data = config["data"]
    return bool(
        quote.delta is not None
        and bid is not None
        and ask is not None
        and bid >= float(data["required_bid"])
        and ask >= bid
        and spread is not None
        and spread <= float(data["maximum_relative_spread"])
        and interest is not None
        and interest >= int(data["minimum_open_interest"])
    )


def _choose_expiry(rows: Sequence[ChainQuote], config: Mapping[str, Any]) -> Optional[str]:
    low, high = (int(item) for item in config["data"]["entry_dte"])
    preferred = int(config["data"]["preferred_dte"])
    by_expiry: Dict[str, int] = {}
    for row in rows:
        if low <= row.dte <= high:
            by_expiry.setdefault(row.expiry, row.dte)
    if not by_expiry:
        return None
    return min(by_expiry, key=lambda expiry: (abs(by_expiry[expiry] - preferred), expiry))


def _choose_leg(
    rows: Sequence[ChainQuote],
    *,
    expiry: str,
    option_type: str,
    target_delta: float,
    config: Mapping[str, Any],
) -> Optional[ChainQuote]:
    eligible = tuple(
        row
        for row in rows
        if row.expiry == expiry and _eligible_quote(row, option_type, config)
    )
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (abs(float(row.delta) - target_delta), row.strike),
    )


def _realized_volatility(prices: Sequence[float]) -> float:
    if len(prices) != 21 or any(value <= 0.0 for value in prices):
        raise ResearchError("realized-volatility window must contain 21 positive prices")
    returns = tuple(math.log(prices[index] / prices[index - 1]) for index in range(1, 21))
    return statistics.pstdev(returns) * math.sqrt(252.0)


def _clip_probability(value: float, config: Mapping[str, Any]) -> float:
    policy = config["signal_policy"]
    return min(
        float(policy["raw_probability_ceiling"]),
        max(float(policy["raw_probability_floor"]), value),
    )


def _entry_rows(
    connection: sqlite3.Connection, trade_date: str, ticker: str
) -> Tuple[ChainQuote, ...]:
    return tuple(
        _quote_from_row(row)
        for row in connection.execute(
            """
            SELECT * FROM chains
             WHERE trade_date = ? AND ticker = ?
             ORDER BY expiry, strike
            """,
            (trade_date, ticker),
        )
    )


def _candidate_legs(
    rows: Sequence[ChainQuote],
    hypothesis: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[ChainQuote, ...]:
    expiry = _choose_expiry(rows, config)
    if expiry is None:
        return ()
    selected = []
    for spec in hypothesis["legs"]:
        quote = _choose_leg(
            rows,
            expiry=expiry,
            option_type=str(spec["option_type"]),
            target_delta=float(spec["target_call_delta"]),
            config=config,
        )
        if quote is None:
            return ()
        selected.append(quote)
    if len({item.strike for item in selected}) != len(selected):
        return ()
    family = str(hypothesis["strategy_family"])
    if family == "CALL_DEBIT_VERTICAL" and not selected[0].strike < selected[1].strike:
        return ()
    if family == "PUT_DEBIT_VERTICAL" and not selected[0].strike > selected[1].strike:
        return ()
    return tuple(selected)


def _path_rows(
    connection: sqlite3.Connection,
    *,
    ticker: str,
    expiry: str,
    strikes: Sequence[float],
    path_dates: Sequence[str],
) -> Mapping[Tuple[str, float], ChainQuote]:
    placeholders = ",".join("?" for _ in strikes)
    query = (
        "SELECT * FROM chains WHERE ticker = ? AND expiry = ? "
        "AND strike IN (%s) AND trade_date >= ? AND trade_date <= ?"
        % placeholders
    )
    parameters: Tuple[Any, ...] = (
        ticker,
        expiry,
        *tuple(float(item) for item in strikes),
        path_dates[0],
        path_dates[-1],
    )
    selected_dates = set(path_dates)
    result: Dict[Tuple[str, float], ChainQuote] = {}
    for row in connection.execute(query, parameters):
        if row["trade_date"] not in selected_dates:
            continue
        quote = _quote_from_row(row)
        key = (quote.trade_date, quote.strike)
        if key in result:
            raise ResearchError("exact historical path contains a duplicate contract")
        result[key] = quote
    return result


def _slippage_dollars(
    quotes: Sequence[ChainQuote],
    option_types: Sequence[str],
    config: Mapping[str, Any],
) -> float:
    cost = config["cost_policy"]
    fraction = float(cost["additional_slippage_fraction_of_spread"])
    minimum = float(cost["minimum_slippage_per_share_per_leg_per_side"])
    multiplier = float(cost["contract_multiplier"])
    values = []
    for quote, option_type in zip(quotes, option_types):
        bid, ask = _bid_ask(quote, option_type)
        if bid is None or ask is None or bid < 0.0 or ask < bid:
            raise ResearchError("exact path quote is missing a valid bid/ask")
        values.append(max(minimum, fraction * (ask - bid)) * multiplier)
    return math.fsum(values)


def _gross_pnl(
    entry_quotes: Sequence[ChainQuote],
    exit_quotes: Sequence[ChainQuote],
    hypothesis: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    multiplier = float(config["cost_policy"]["contract_multiplier"])
    values = []
    for entry, exit_, spec in zip(entry_quotes, exit_quotes, hypothesis["legs"]):
        option_type = str(spec["option_type"])
        entry_bid, entry_ask = _bid_ask(entry, option_type)
        exit_bid, exit_ask = _bid_ask(exit_, option_type)
        if None in (entry_bid, entry_ask, exit_bid, exit_ask):
            raise ResearchError("exact historical P/L is missing a quote side")
        ratio = int(spec["ratio"])
        if str(spec["action"]) == "BUY":
            values.append((float(exit_bid) - float(entry_ask)) * ratio * multiplier)
        else:
            values.append((float(entry_bid) - float(exit_ask)) * ratio * multiplier)
    return math.fsum(values)


def _entry_debit_dollars(
    entry_quotes: Sequence[ChainQuote],
    hypothesis: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    multiplier = float(config["cost_policy"]["contract_multiplier"])
    values = []
    for quote, spec in zip(entry_quotes, hypothesis["legs"]):
        bid, ask = _bid_ask(quote, str(spec["option_type"]))
        if bid is None or ask is None:
            raise ResearchError("entry debit cannot be reproduced")
        ratio = int(spec["ratio"])
        values.append(
            (float(ask) if str(spec["action"]) == "BUY" else -float(bid))
            * ratio
            * multiplier
        )
    return math.fsum(values)


def _research_trade(
    connection: sqlite3.Connection,
    *,
    sessions: Sequence[str],
    signal_index: int,
    entry_index: int,
    ticker: str,
    hypothesis: Mapping[str, Any],
    entry_quotes: Sequence[ChainQuote],
    momentum: float,
    realized_volatility: float,
    config: Mapping[str, Any],
) -> Tuple[Optional[ResearchTrade], Optional[str]]:
    holding_limit = int(config["exit_policy"]["time_exit_sessions"])
    path_dates = tuple(sessions[entry_index + 1 : entry_index + holding_limit + 1])
    if len(path_dates) != holding_limit:
        return None, "INCOMPLETE_FUTURE_SESSION_PATH"
    expiry = entry_quotes[0].expiry
    if any(item.expiry != expiry for item in entry_quotes):
        return None, "MULTI_EXPIRY_NOT_ALLOWED_IN_V1"
    path = _path_rows(
        connection,
        ticker=ticker,
        expiry=expiry,
        strikes=tuple(item.strike for item in entry_quotes),
        path_dates=path_dates,
    )
    option_types = tuple(str(item["option_type"]) for item in hypothesis["legs"])
    entry_slippage = _slippage_dollars(entry_quotes, option_types, config)
    cost = config["cost_policy"]
    contract_sides = sum(int(item["ratio"]) for item in hypothesis["legs"]) * 2
    commissions_fees = contract_sides * (
        float(cost["commission_per_contract_per_side"])
        + float(cost["fee_per_contract_per_side"])
    )
    entry_debit = _entry_debit_dollars(entry_quotes, hypothesis, config)
    if entry_debit <= 0.0:
        return None, "NONPOSITIVE_DEBIT"
    estimated_exit_slippage = entry_slippage
    maximum_loss = entry_debit + entry_slippage + estimated_exit_slippage + commissions_fees
    if maximum_loss <= 0.0 or not math.isfinite(maximum_loss):
        return None, "UNDEFINED_MAXIMUM_LOSS"
    family = str(hypothesis["strategy_family"])
    maximum_profit: Optional[float] = None
    if family in {"CALL_DEBIT_VERTICAL", "PUT_DEBIT_VERTICAL"}:
        width = abs(entry_quotes[0].strike - entry_quotes[1].strike) * float(
            cost["contract_multiplier"]
        )
        maximum_profit = width - entry_debit - entry_slippage - estimated_exit_slippage - commissions_fees
        if maximum_profit <= 0.0:
            return None, "NONPOSITIVE_MAXIMUM_PROFIT"
    target_pnl = maximum_loss * float(
        config["exit_policy"]["profit_target_fraction_of_max_loss"]
    )
    stop_pnl = -maximum_loss * float(
        config["exit_policy"]["stop_loss_fraction_of_max_loss"]
    )
    chosen_exit: Optional[Tuple[str, Sequence[ChainQuote], float, float, float, str]] = None
    for offset, path_date in enumerate(path_dates, 1):
        exit_quotes = []
        for entry in entry_quotes:
            quote = path.get((path_date, entry.strike))
            if quote is None:
                return None, "MISSING_EXACT_CONTRACT_PATH"
            bid, ask = _bid_ask(quote, option_types[len(exit_quotes)])
            if bid is None or ask is None or bid < 0.0 or ask < bid:
                return None, "INVALID_EXACT_CONTRACT_EXIT_QUOTE"
            exit_quotes.append(quote)
        gross = _gross_pnl(entry_quotes, exit_quotes, hypothesis, config)
        exit_slippage = _slippage_dollars(exit_quotes, option_types, config)
        net = gross - commissions_fees - entry_slippage - exit_slippage
        if net <= stop_pnl:
            chosen_exit = (path_date, tuple(exit_quotes), gross, exit_slippage, net, "STOP")
            break
        if net >= target_pnl:
            chosen_exit = (path_date, tuple(exit_quotes), gross, exit_slippage, net, "TARGET")
            break
        if offset == holding_limit:
            chosen_exit = (path_date, tuple(exit_quotes), gross, exit_slippage, net, "TIME")
    if chosen_exit is None:
        return None, "NO_FROZEN_EXIT"
    exit_date, exit_quotes, gross, exit_slippage, net, exit_reason = chosen_exit
    leg_rows = []
    for quote, spec in zip(entry_quotes, hypothesis["legs"]):
        bid, ask = _bid_ask(quote, str(spec["option_type"]))
        assert bid is not None and ask is not None
        leg_rows.append(
            ResearchLeg(
                action=str(spec["action"]),
                option_type=str(spec["option_type"]),
                expiry=quote.expiry,
                strike=quote.strike,
                occ_symbol=_occ_symbol(
                    ticker, quote.expiry, str(spec["option_type"]), quote.strike
                ),
                entry_bid=float(bid),
                entry_ask=float(ask),
                target_call_delta=float(spec["target_call_delta"]),
            )
        )
    relative_spread = max(
        float(_relative_spread(quote, option_type) or 0.0)
        for quote, option_type in zip(entry_quotes, option_types)
    )
    smv_values = tuple(
        float(item.smv_vol) for item in entry_quotes if item.smv_vol is not None
    )
    if not smv_values:
        return None, "MISSING_ENTRY_SMV_VOL"
    smv_vol = statistics.mean(smv_values)
    raw_probability = _clip_probability(
        0.50
        + 2.0 * (abs(momentum) - 0.03)
        + 0.10 * (realized_volatility / smv_vol - 1.0)
        - 0.20 * relative_spread,
        config,
    )
    identity = {
        "strategy_family": family,
        "ticker": ticker,
        "signal_date": sessions[signal_index],
        "entry_date": sessions[entry_index],
        "legs": [asdict(item) for item in leg_rows],
        "effective_config": config["effective_version"],
    }
    record_id = "trade-" + _sha256_bytes(_canonical_json(identity))[:24]
    holding_sessions = path_dates.index(exit_date) + 1
    return (
        ResearchTrade(
            record_id=record_id,
            strategy_family=family,
            ticker=ticker,
            signal_date=sessions[signal_index],
            entry_date=sessions[entry_index],
            exit_date=exit_date,
            holding_sessions=holding_sessions,
            expiry=expiry,
            legs=tuple(leg_rows),
            momentum_20=momentum,
            realized_volatility_20=realized_volatility,
            smv_vol=smv_vol,
            relative_spread=relative_spread,
            raw_probability=raw_probability,
            entry_debit=entry_debit,
            maximum_loss=maximum_loss,
            maximum_profit=maximum_profit,
            target_pnl=target_pnl,
            stop_pnl=stop_pnl,
            gross_pnl=gross,
            commissions_fees=commissions_fees,
            entry_slippage=entry_slippage,
            exit_slippage=exit_slippage,
            net_pnl=net,
            exit_reason=exit_reason,
            target_hit=exit_reason == "TARGET",
            stop_hit=exit_reason == "STOP",
            max_loss_hit=net <= -0.95 * maximum_loss,
            entry_snapshot_id=entry_quotes[0].snapshot_id,
            exit_snapshot_id=exit_quotes[0].snapshot_id,
            exact_path_observations=holding_sessions * len(entry_quotes),
            corporate_action_review="ORATS_SPLIT_HISTORY_CLEAR_FOR_VALIDATION_PERIOD",
        ),
        None,
    )


def generate_historical_trades(
    path: Path = DEFAULT_CHAIN_DB,
) -> Tuple[Tuple[ResearchTrade, ...], Tuple[UnresolvedCandidate, ...], Mapping[str, int]]:
    """Generate exact-leg outcomes from frozen signals without inspecting aggregates."""

    config = effective_validation_config()
    try:
        split_review = json.loads(DEFAULT_SPLIT_REVIEW.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResearchError("corporate-action review artifact is unavailable") from exc
    if split_review.get("relevant_period_splits"):
        raise ResearchError("validation universe contains a relevant split")
    connection = _connect_chain_db(path, create=False)
    sessions = tuple(
        str(row[0])
        for row in connection.execute("SELECT trade_date FROM sessions ORDER BY trade_date")
    )
    if len(sessions) != 450:
        connection.close()
        raise ResearchError("historical database does not contain 450 sessions")
    symbols = tuple(str(item) for item in config["universe"])
    spots: Dict[str, Dict[str, float]] = {symbol: {} for symbol in symbols}
    for row in connection.execute(
        "SELECT trade_date, ticker, stock_price FROM underlying ORDER BY trade_date, ticker"
    ):
        if row["ticker"] in spots:
            spots[str(row["ticker"])][str(row["trade_date"])] = float(row["stock_price"])
    if any(len(spots[symbol]) != len(sessions) for symbol in symbols):
        connection.close()
        raise ResearchError("underlying history is incomplete")
    hypotheses = tuple(config["hypotheses"])
    by_family: Dict[str, Mapping[str, Any]] = {
        str(item["strategy_family"]): item for item in hypotheses
    }
    trades: List[ResearchTrade] = []
    unresolved: List[UnresolvedCandidate] = []
    counters: Dict[str, int] = {}

    def count(name: str) -> None:
        counters[name] = counters.get(name, 0) + 1

    lookback = int(config["signal_policy"]["momentum_lookback_sessions"])
    holding = int(config["exit_policy"]["time_exit_sessions"])
    # Signals are formed only after session T closes.  The earliest executable
    # entry is the next session, T+1.  Using T's close-derived signal with T's
    # option quote would be look-ahead leakage.
    for signal_index in range(lookback, len(sessions) - holding - 1):
        entry_index = signal_index + 1
        signal_date = sessions[signal_index]
        entry_date = sessions[entry_index]
        for ticker in symbols:
            price_window = tuple(
                spots[ticker][sessions[index]]
                for index in range(signal_index - lookback, signal_index + 1)
            )
            momentum = price_window[-1] / price_window[0] - 1.0
            realized = _realized_volatility(price_window)
            rows = _entry_rows(connection, entry_date, ticker)
            for family, hypothesis in by_family.items():
                direction = str(hypothesis["direction"])
                threshold = (
                    float(config["signal_policy"]["bullish_momentum_threshold"])
                    if direction == "BULLISH"
                    else float(config["signal_policy"]["bearish_momentum_threshold"])
                )
                if direction == "BULLISH" and momentum < threshold:
                    continue
                if direction == "BEARISH" and momentum > threshold:
                    continue
                count(family + ":SIGNALS")
                selected = _candidate_legs(rows, hypothesis, config)
                if not selected:
                    count(family + ":ENTRY_STRUCTURE_UNAVAILABLE")
                    continue
                smv = tuple(item.smv_vol for item in selected if item.smv_vol is not None)
                if not smv:
                    count(family + ":MISSING_SMV")
                    continue
                if bool(hypothesis["requires_long_vol_value"]) and statistics.mean(smv) > (
                    realized
                    * float(
                        config["signal_policy"]["long_option_max_iv_to_realized_ratio"]
                    )
                ):
                    count(family + ":LONG_VOL_VALUE_FILTER")
                    continue
                count(family + ":CONSTRUCTED")
                trade, reason = _research_trade(
                    connection,
                    sessions=sessions,
                    signal_index=signal_index,
                    entry_index=entry_index,
                    ticker=ticker,
                    hypothesis=hypothesis,
                    entry_quotes=selected,
                    momentum=momentum,
                    realized_volatility=realized,
                    config=config,
                )
                if trade is None:
                    assert reason is not None
                    candidate_id = "candidate-" + _sha256_bytes(
                        _canonical_json(
                            {
                                "family": family,
                                "ticker": ticker,
                                "signal_date": signal_date,
                                "entry_date": entry_date,
                                "reason": reason,
                            }
                        )
                    )[:24]
                    unresolved.append(
                        UnresolvedCandidate(
                            candidate_id=candidate_id,
                            strategy_family=family,
                            ticker=ticker,
                            signal_date=signal_date,
                            entry_date=entry_date,
                            reason=reason,
                        )
                    )
                    count(family + ":UNRESOLVED")
                else:
                    trades.append(trade)
                    count(family + ":RESOLVED")
        if entry_index % 25 == 0:
            print(
                "RESEARCH entry_session=%d resolved=%d unresolved=%d"
                % (entry_index, len(trades), len(unresolved)),
                flush=True,
            )
    connection.close()
    identifiers = tuple(item.record_id for item in trades)
    if len(identifiers) != len(set(identifiers)):
        raise ResearchError("historical trade identities are duplicated")
    return (
        tuple(sorted(trades, key=lambda item: (item.entry_date, item.ticker, item.strategy_family))),
        tuple(
            sorted(
                unresolved,
                key=lambda item: (item.entry_date, item.ticker, item.strategy_family),
            )
        ),
        dict(sorted(counters.items())),
    )


def _period_summary(
    name: str, trades: Sequence[ResearchTrade], confidence: float = 0.95
) -> Mapping[str, Any]:
    if not trades:
        raise ResearchError("%s period contains no resolved trades" % name)
    values = tuple(item.net_pnl for item in trades)
    ticker_clusters = tuple(item.ticker for item in trades)
    date_clusters = tuple(item.signal_date for item in trades)
    interval = two_way_clustered_bootstrap_mean_ci(
        values,
        ticker_clusters,
        date_clusters,
        confidence=confidence,
        iterations=5_000,
        seed=17,
    )
    return {
        "name": name,
        "expectancy": statistics.mean(values),
        "lower_confidence_bound": interval.lower,
        "upper_confidence_bound": interval.upper,
        "confidence_level": confidence,
        "resolved_trades": len(trades),
        "independent_ticker_date_clusters": len(
            {(item.ticker, item.signal_date) for item in trades}
        ),
        "bootstrap_ticker_clusters": interval.first_cluster_count,
        "bootstrap_date_clusters": interval.second_cluster_count,
        "bootstrap_joint_clusters": interval.joint_cluster_count,
        "start": min(item.signal_date for item in trades),
        "end": max(item.signal_date for item in trades),
        "total_net_profit": math.fsum(values),
        "win_rate": sum(item.net_pnl > 0.0 for item in trades) / len(trades),
        "target_rate": sum(item.target_hit for item in trades) / len(trades),
        "stop_rate": sum(item.stop_hit for item in trades) / len(trades),
        "max_loss_rate": sum(item.max_loss_hit for item in trades) / len(trades),
        "average_maximum_loss": statistics.mean(item.maximum_loss for item in trades),
        "average_return_on_maximum_loss": statistics.mean(
            item.net_pnl / item.maximum_loss for item in trades
        ),
    }


def _one_sided_cluster_p_value(trades: Sequence[ResearchTrade]) -> float:
    """One-sided normal approximation over independent entry-date cluster means."""

    grouped: Dict[str, List[float]] = {}
    for item in trades:
        grouped.setdefault(item.signal_date, []).append(item.net_pnl)
    cluster_means = tuple(statistics.mean(values) for _, values in sorted(grouped.items()))
    if len(cluster_means) < 2:
        return 1.0
    mean = statistics.mean(cluster_means)
    standard_deviation = statistics.stdev(cluster_means)
    if standard_deviation <= 0.0:
        return 0.0 if mean > 0.0 else 1.0
    z_score = mean / (standard_deviation / math.sqrt(len(cluster_means)))
    return max(0.0, min(1.0, 1.0 - NormalDist().cdf(z_score)))


def _profit_concentration(trades: Sequence[ResearchTrade]) -> Mapping[str, Any]:
    ticker: Dict[str, float] = {}
    month: Dict[str, float] = {}
    for item in trades:
        positive = max(0.0, item.net_pnl)
        ticker[item.ticker] = ticker.get(item.ticker, 0.0) + positive
        key = item.signal_date[:7]
        month[key] = month.get(key, 0.0) + positive
    positive_total = math.fsum(ticker.values())
    if positive_total <= 0.0:
        return {
            "maximum_fraction": 1.0,
            "ticker_maximum_fraction": 1.0,
            "calendar_month_maximum_fraction": 1.0,
            "positive_profit": 0.0,
        }
    ticker_fraction = max(ticker.values()) / positive_total
    month_fraction = max(month.values()) / positive_total
    return {
        "maximum_fraction": max(ticker_fraction, month_fraction),
        "ticker_maximum_fraction": ticker_fraction,
        "calendar_month_maximum_fraction": month_fraction,
        "positive_profit": positive_total,
        "ticker_positive_profit": dict(sorted(ticker.items())),
        "calendar_month_positive_profit": dict(sorted(month.items())),
    }


def _raw_for_target(trade: ResearchTrade, target: ProbabilityTarget) -> float:
    if target is ProbabilityTarget.POP_NET:
        return trade.raw_probability
    if target is ProbabilityTarget.P_TARGET:
        return min(0.95, max(0.05, trade.raw_probability - 0.05))
    if target is ProbabilityTarget.P_STOP:
        return min(0.95, max(0.05, 1.0 - trade.raw_probability))
    return min(0.95, max(0.05, (1.0 - trade.raw_probability) * 0.50))


def _outcome_for_target(trade: ResearchTrade, target: ProbabilityTarget) -> int:
    if target is ProbabilityTarget.POP_NET:
        return int(trade.net_pnl > 0.0)
    if target is ProbabilityTarget.P_TARGET:
        return int(trade.target_hit)
    if target is ProbabilityTarget.P_STOP:
        return int(trade.stop_hit)
    return int(trade.max_loss_hit)


def _build_pop_artifacts(
    family: str,
    development: Sequence[ResearchTrade],
    *,
    development_calendar: Sequence[date],
    holdout_start: date,
    model_frozen_at: datetime,
) -> Mapping[ProbabilityTarget, OOFPOPModelArtifact]:
    artifacts: Dict[ProbabilityTarget, OOFPOPModelArtifact] = {}
    for target in ProbabilityTarget:
        bucket = POPBucketIdentity(
            strategy_family=family,
            regime_id="ALL_REGIMES_V1",
            target=target,
            bucket_version="CULTRA_POP_BUCKET_V1",
        )
        observations = tuple(
            OOFPOPObservation(
                observation_id=item.observation_id,
                session_date=date.fromisoformat(item.signal_date),
                bucket_id=bucket.bucket_id,
                raw_probability=_raw_for_target(item, target),
                outcome=_outcome_for_target(item, target),
            )
            for item in development
        )
        artifacts[target] = build_oof_pop_model(
            observations,
            bucket,
            model_version=MODEL_VERSION,
            holdout_start=holdout_start,
            model_frozen_at=model_frozen_at,
            min_training_sessions=120,
            validation_sessions=20,
            embargo_sessions=60,
            session_calendar=development_calendar,
        )
    return artifacts


def _global_split_dates(sessions: Sequence[str]) -> Mapping[str, Tuple[str, ...]]:
    # The split is based on the frozen calendar, never on which trades resolve.
    candidate_dates = tuple(sessions[20 : len(sessions) - 20])
    observations = tuple(
        HistoricalObservation(
            observation_id="calendar-" + item,
            session_date=date.fromisoformat(item),
            cluster_id=item,
            net_pnl=0.0,
        )
        for item in candidate_dates
    )
    split = chronological_split(
        observations,
        validation_fraction=0.20,
        holdout_fraction=0.20,
        embargo_sessions=60,
    )
    return {
        "training": tuple(item.session_date.isoformat() for item in split.training),
        "validation": tuple(item.session_date.isoformat() for item in split.validation),
        "holdout": tuple(item.session_date.isoformat() for item in split.holdout),
        "embargoed": tuple(item.session_date.isoformat() for item in split.embargoed),
    }


def _hypothesis_fingerprint(config: Mapping[str, Any], family: str) -> str:
    hypothesis = next(
        item for item in config["hypotheses"] if item["strategy_family"] == family
    )
    return "sha256:" + _sha256_bytes(
        _canonical_json(
            {
                "hypothesis": hypothesis,
                "signal_policy": config["signal_policy"],
                "base_config_sha256": config["base_sha256"],
                "amendment_sha256": config["amendment_sha256"],
            }
        )
    )


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> Path:
    rows = b"".join(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for value in values
    )
    return _private_write(path, rows)


def run_historical_validation(
    *,
    database: Path = DEFAULT_CHAIN_DB,
    output_root: Path = PROJECT_ROOT / "out",
    run_id: str = "cultra-historical-validation-v1-1",
) -> Mapping[str, Any]:
    """Refuse the invalidated V1 holdout path.

    V1 generated and published outcomes from its nominal holdout before the
    research identity was safely frozen.  That evidence is now development
    data and can never be consumed as an untouched holdout again.  Cultra V2's
    unified pattern command performs embargoed development diagnostics and
    requires a newly collected post-freeze period for future validation.
    """

    raise ResearchError(
        "CULTRA V1 HOLDOUT INVALIDATED: exposed outcomes cannot be reused; "
        "run the unified Cultra V2 pattern rebuild instead"
    )

    config = effective_validation_config()
    output = Path(output_root).expanduser().resolve()
    try:
        output.relative_to((PROJECT_ROOT / "out").resolve())
    except ValueError as exc:
        raise ResearchError("validation output must remain inside Cultra/out") from exc
    run_dir = output / run_id
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(run_dir, 0o700)
    trades, unresolved, counters = generate_historical_trades(database)
    sessions = load_recent_sessions()
    split_dates = _global_split_dates(sessions)
    train_dates = set(split_dates["training"])
    validation_dates = set(split_dates["validation"])
    holdout_dates = set(split_dates["holdout"])
    holdout_start = date.fromisoformat(min(holdout_dates))
    training_start = date.fromisoformat(min(train_dates))
    validation_end = date.fromisoformat(max(validation_dates))
    development_calendar = tuple(
        date.fromisoformat(item)
        for item in sessions
        if training_start <= date.fromisoformat(item) <= validation_end
    )
    families = tuple(str(item["strategy_family"]) for item in config["hypotheses"])
    by_family = {
        family: tuple(item for item in trades if item.strategy_family == family)
        for family in families
    }
    unresolved_by_family = {
        family: tuple(item for item in unresolved if item.strategy_family == family)
        for family in families
    }
    model_frozen_at = datetime.now(timezone.utc)
    artifacts_by_family: Dict[str, Mapping[ProbabilityTarget, OOFPOPModelArtifact]] = {}
    family_results: Dict[str, Dict[str, Any]] = {}
    registry_candidates: Dict[
        str, Tuple[EvidencePartitions, FrozenEvidenceIdentity]
    ] = {}

    for family in families:
        family_trades = by_family[family]
        training = tuple(item for item in family_trades if item.signal_date in train_dates)
        validation = tuple(
            item for item in family_trades if item.signal_date in validation_dates
        )
        holdout = tuple(item for item in family_trades if item.signal_date in holdout_dates)
        # The final 60-session embargo is excluded from calibration and fitting.
        # The first embargo remains in the market-session calendar so every OOF
        # boundary is measured in sessions even when a strategy emits no trade.
        development = tuple(
            item
            for item in family_trades
            if training_start <= date.fromisoformat(item.signal_date) <= validation_end
        )
        result: Dict[str, Any] = {
            "strategy_family": family,
            "state": "UNPROVEN",
            "holdout_status": "SEALED_NOT_OPENED",
            "reasons": [],
            "resolved_total": len(family_trades),
            "unresolved_total": len(unresolved_by_family[family]),
        }
        try:
            result["training"] = _period_summary("training", training)
            result["validation"] = _period_summary("validation", validation)
            artifacts = _build_pop_artifacts(
                family,
                development,
                development_calendar=development_calendar,
                holdout_start=holdout_start,
                model_frozen_at=model_frozen_at,
            )
            artifacts_by_family[family] = artifacts
            pop_net = artifacts[ProbabilityTarget.POP_NET]
            result["pop_validation"] = {
                target.value: {
                    "artifact_id": artifact.artifact_id,
                    "selected_method": artifact.selected_method,
                    "oof_brier_score": artifact.oof_brier_score,
                    "base_rate_brier_score": artifact.base_rate_brier_score,
                    "expected_calibration_error": artifact.expected_calibration_error,
                    "sample_size": artifact.interval.sample_size,
                    "calibration_start": artifact.development_start.isoformat(),
                    "calibration_end": artifact.development_end.isoformat(),
                }
                for target, artifact in artifacts.items()
            }
            partitions = EvidencePartitions(
                training_observation_ids=tuple(item.observation_id for item in training),
                validation_observation_ids=tuple(item.observation_id for item in validation),
                holdout_observation_ids=tuple(item.observation_id for item in holdout),
            )
            combined_artifact_id = "sha256:" + _sha256_bytes(
                _canonical_json(
                    {target.value: artifact.artifact_id for target, artifact in artifacts.items()}
                )
            )
            identity = FrozenEvidenceIdentity(
                strategy_family=family,
                catalog_version=CATALOG_VERSION,
                hypothesis_fingerprint=_hypothesis_fingerprint(config, family),
                cost_model_version=str(config["cost_policy"]["version"]),
                exit_policy_version=str(config["exit_policy"]["version"]),
                pop_model_version=MODEL_VERSION,
                pop_model_artifact_id=combined_artifact_id,
                model_frozen_at=model_frozen_at,
            )
            registry_candidates[family] = (partitions, identity)
            if result["training"]["expectancy"] <= 0.0:
                result["reasons"].append("training expectancy is not positive")
            if result["validation"]["expectancy"] <= 0.0:
                result["reasons"].append("validation expectancy is not positive")
            if pop_net.oof_brier_score >= pop_net.base_rate_brier_score:
                result["reasons"].append("POP model does not beat unconditional Brier")
            if pop_net.expected_calibration_error > float(
                config["promotion_policy"]["maximum_pop_ece"]
            ):
                result["reasons"].append("POP expected calibration error exceeds 0.05")
        except (ValueError, ResearchError) as exc:
            result["reasons"].append("development evidence unavailable: %s" % str(exc))
        family_results[family] = result

    # Lock identities and development decisions before reading holdout aggregates.
    development_pass = []
    registry = EvidenceRegistry(DEFAULT_EVIDENCE_DB)
    try:
        for family in families:
            result = family_results[family]
            candidate = registry_candidates.get(family)
            if candidate is None:
                continue
            partitions, identity = candidate
            registry.register(identity, partitions, now=model_frozen_at)
            if result["training"]["expectancy"] > 0.0:
                registry.advance_development(
                    family,
                    RegistryState.RESEARCH_PASS,
                    partitions.development_fingerprint,
                    now=model_frozen_at,
                )
                result["state"] = "RESEARCH_PASS"
            else:
                continue
            if not result["reasons"]:
                registry.advance_development(
                    family,
                    RegistryState.VALIDATION_PASS,
                    partitions.development_fingerprint,
                    now=model_frozen_at,
                )
                result["state"] = "VALIDATION_PASS"
                development_pass.append(family)

        raw_p_values: Dict[str, float] = {
            item.strategy_id: 1.0 for item in FROZEN_STRATEGY_CATALOG
        }
        opened_holdouts: Dict[str, Tuple[ResearchTrade, ...]] = {}
        for family in development_pass:
            holdout = tuple(
                item for item in by_family[family] if item.signal_date in holdout_dates
            )
            opened_holdouts[family] = holdout
            raw_p_values[family] = _one_sided_cluster_p_value(holdout)
        adjusted = holm_adjust_mapping(raw_p_values)

        for family in development_pass:
            result = family_results[family]
            holdout = opened_holdouts[family]
            holdout_summary = _period_summary("untouched_holdout", holdout)
            concentration = _profit_concentration(holdout)
            holdout_unresolved = tuple(
                item
                for item in unresolved_by_family[family]
                if item.signal_date in holdout_dates
            )
            unresolved_fraction = len(holdout_unresolved) / max(
                1, len(holdout_unresolved) + len(holdout)
            )
            result["holdout"] = holdout_summary
            result["holdout_status"] = "OPENED_ONCE"
            result["raw_one_sided_p_value"] = raw_p_values[family]
            result["holm_adjusted_p_value"] = adjusted[family]
            result["holm_family_size"] = len(FROZEN_STRATEGY_CATALOG)
            result["profit_concentration"] = concentration
            result["holdout_unresolved"] = len(holdout_unresolved)
            result["holdout_unresolved_fraction"] = unresolved_fraction
            gates = []
            policy = config["promotion_policy"]
            if holdout_summary["expectancy"] <= 0.0:
                gates.append("holdout expectancy is not positive")
            if holdout_summary["lower_confidence_bound"] <= 0.0:
                gates.append("holdout 95% clustered lower bound is not positive")
            if holdout_summary["resolved_trades"] < int(
                policy["minimum_holdout_trades"]
            ):
                gates.append("holdout has fewer than 100 resolved trades")
            if holdout_summary["independent_ticker_date_clusters"] < int(
                policy["minimum_holdout_ticker_date_clusters"]
            ):
                gates.append("holdout has fewer than 40 ticker/date clusters")
            if adjusted[family] > float(policy["maximum_holm_adjusted_p_value"]):
                gates.append("Holm-adjusted significance gate failed")
            if concentration["maximum_fraction"] > float(
                policy["maximum_single_ticker_or_month_profit_fraction"]
            ):
                gates.append("ticker/calendar contribution concentration gate failed")
            if unresolved_fraction > float(
                config["data"]["maximum_unresolved_path_fraction"]
            ):
                gates.append("unresolved exact-path fraction exceeds 5%")
            result["reasons"].extend(gates)
            partitions = registry_candidates[family][0]
            passed = not gates
            record = registry.consume_holdout(
                family,
                partitions.holdout_fingerprint,
                passed=passed,
                now=datetime.now(timezone.utc),
            )
            result["state"] = record.state.value
    finally:
        registry.close()

    catalog_states = {item.strategy_id: "UNPROVEN" for item in FROZEN_STRATEGY_CATALOG}
    for family, result in family_results.items():
        catalog_states[family] = str(result["state"])
    artifacts_payload = {
        family: {
            target.value: artifact.to_dict()
            for target, artifact in artifacts.items()
        }
        for family, artifacts in artifacts_by_family.items()
    }
    summary = {
        "schema": "cultra.historical-validation-result.v1",
        "run_id": run_id,
        "effective_config_version": config["effective_version"],
        "base_config_sha256": config["base_sha256"],
        "amendment_sha256": config["amendment_sha256"],
        "database": str(Path(database).resolve()),
        "database_sha256": _sha256_bytes(Path(database).read_bytes()),
        "model_frozen_at": model_frozen_at.isoformat(),
        "calendar_split": {
            key: {
                "session_count": len(values),
                "start": min(values),
                "end": max(values),
            }
            for key, values in split_dates.items()
        },
        "resolved_trades": len(trades),
        "unresolved_candidates": len(unresolved),
        "generation_counts": counters,
        "family_results": family_results,
        "strategy_states": catalog_states,
        "historically_validated_families": sorted(
            family
            for family, result in family_results.items()
            if result["state"] == "HOLDOUT_PASS"
        ),
        "manual_ticket_enabled_families": [],
        "broker_submission_enabled": False,
        "quantity": "USER DETERMINED",
    }
    _private_json(run_dir / "historical_validation.json", summary)
    _private_json(run_dir / "pop_model_artifacts.json", artifacts_payload)
    _write_jsonl(run_dir / "resolved_trades.jsonl", (item.to_dict() for item in trades))
    _write_jsonl(
        run_dir / "unresolved_candidates.jsonl",
        (item.to_dict() for item in unresolved),
    )
    lines = [
        "# Cultra Historical Profit Evidence — V1.1",
        "",
        "Historical evidence is not a guarantee of future profit.",
        "",
        "- Resolved exact-leg trades: **%d**" % len(trades),
        "- Unresolved exact-path candidates: **%d**" % len(unresolved),
        "- Holdout-passing families: **%s**"
        % (", ".join(summary["historically_validated_families"]) or "None"),
        "- Broker submission: **disabled**",
        "",
        "## Family results",
        "",
        "| Family | State | Train EV | Validation EV | Holdout EV | Holdout 95% LCB | POP ECE | POP Brier/Base | Reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for family in families:
        result = family_results[family]
        training = result.get("training", {})
        validation = result.get("validation", {})
        holdout = result.get("holdout", {})
        pop = result.get("pop_validation", {}).get("POP_NET", {})
        brier = "—"
        if pop:
            brier = "%.4f / %.4f" % (
                pop["oof_brier_score"],
                pop["base_rate_brier_score"],
            )
        lines.append(
            "| %s | `%s` | %s | %s | %s | %s | %s | %s | %s |"
            % (
                family,
                result["state"],
                "%.2f" % training["expectancy"] if training else "—",
                "%.2f" % validation["expectancy"] if validation else "—",
                "%.2f" % holdout["expectancy"] if holdout else "sealed",
                "%.2f" % holdout["lower_confidence_bound"] if holdout else "sealed",
                "%.4f" % pop["expected_calibration_error"] if pop else "—",
                brier,
                "; ".join(result["reasons"]) or "None",
            )
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "`HOLDOUT_PASS` permits Cultra to show historically validated research orders immediately. `SHADOW_PASS` remains a later confidence upgrade; it is not required for candidate visibility. No result authorizes broker submission.",
            "",
        )
    )
    _private_write(run_dir / "historical_validation.md", "\n".join(lines).encode("utf-8"))
    manifest = {
        "schema": "cultra.historical-validation-manifest.v1",
        "run_id": run_id,
        "artifacts": {},
    }
    for artifact in sorted(run_dir.iterdir()):
        if artifact.name == "manifest.json" or not artifact.is_file():
            continue
        raw = artifact.read_bytes()
        manifest["artifacts"][artifact.name] = {
            "sha256": _sha256_bytes(raw),
            "bytes": len(raw),
        }
    _private_json(run_dir / "manifest.json", manifest)
    return summary


def verify_historical_validation(run_dir: Path) -> Tuple[str, ...]:
    root = Path(run_dir).expanduser().resolve()
    errors = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return ("manifest unavailable: %s" % str(exc),)
    for name, identity in manifest.get("artifacts", {}).items():
        path = root / name
        try:
            raw = path.read_bytes()
        except OSError:
            errors.append("missing artifact %s" % name)
            continue
        if _sha256_bytes(raw) != identity.get("sha256"):
            errors.append("artifact hash mismatch %s" % name)
        if len(raw) != int(identity.get("bytes", -1)):
            errors.append("artifact size mismatch %s" % name)
        if (path.stat().st_mode & 0o077) != 0:
            errors.append("artifact permissions are too broad %s" % name)
    try:
        summary = json.loads((root / "historical_validation.json").read_text())
        if summary.get("broker_submission_enabled") is not False:
            errors.append("broker submission boundary is not false")
        if summary.get("manual_ticket_enabled_families") != []:
            errors.append("historical validation enabled manual tickets")
    except (OSError, UnicodeError, json.JSONDecodeError):
        errors.append("historical validation result is unreadable")
    return tuple(errors)


__all__ = [
    "AMENDMENT_PATH",
    "DEFAULT_CHAIN_DB",
    "DEFAULT_EVIDENCE_DB",
    "ResearchError",
    "ResearchTrade",
    "UnresolvedCandidate",
    "effective_validation_config",
    "generate_historical_trades",
    "ingest_historical_chains",
    "run_historical_validation",
    "verify_historical_validation",
]
