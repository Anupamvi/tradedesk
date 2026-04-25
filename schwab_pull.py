#!/usr/bin/env python3
"""
Schwab Account Puller for Anu Options Engine
============================================
Run this on your local machine. It reads your Schwab API tokens,
pulls positions + rolling orders/transactions/fills activity, writes
local persistence state (SQLite + JSONL), reconciles closed trades,
and writes a snapshot JSON file you can upload to the scan engine.

Usage:
    python schwab_pull.py
    python schwab_pull.py --token-path ./tokens
    python schwab_pull.py --output positions.json
    python schwab_pull.py --register-open-trade --engine-trade-id FIRE-NVDA-20260420-01 \
        --broker-order-id 123456 --track FIRE --ticker NVDA --strategy bull_call_debit \
        --expiry 2026-05-15 --opened-at 2026-04-20T10:14:22-07:00 \
        --intended-legs-json '[{"action":"BUY_TO_OPEN","symbol":"NVDA  260515C00450000","quantity":1}]'

Output: schwab_positions_YYYY-MM-DD.json

Requirements:
    pip install requests

Your credentials never leave your machine. This script only writes
local files that you choose to upload.
"""

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ================================================================
# CONFIGURATION
# ================================================================

SCHWAB_API_BASE = "https://api.schwabapi.com"
SCHWAB_AUTH_URL = "https://api.schwabapi.com/v1/oauth/token"


# ================================================================
# PATH HELPERS
# ================================================================

def project_root() -> Path:
    """Return this repo root, honoring UW_ROOT when provided."""
    env_root = os.environ.get("UW_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parent


def default_token_path() -> Path:
    """Default Schwab token location for macOS/Linux and Windows checkouts."""
    env_token = os.environ.get("SCHWAB_TOKEN_PATH", "").strip()
    if env_token:
        return Path(env_token).expanduser()
    return project_root() / "tokens"


def default_state_dir() -> Path:
    """Default local persistence directory."""
    env_dir = os.environ.get("SCHWAB_PULL_STATE_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return project_root() / "out" / "schwab_pull_state"


DEFAULT_TOKEN_PATH = str(default_token_path())
DEFAULT_STATE_DIR = str(default_state_dir())
DEFAULT_STATE_DB = str(default_state_dir() / "schwab_pull_state.sqlite")
DEFAULT_FIRST_BACKFILL_DAYS = 365
DEFAULT_BACKFILL_CHUNK_DAYS = 30
DEFAULT_BACKFILL_OVERLAP_DAYS = 3
DEFAULT_INCREMENTAL_OVERLAP_DAYS = 3
DEFAULT_OCC_TIMESTAMP_WINDOW_MINUTES = 60


# ================================================================
# TIME + TYPE HELPERS
# ================================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    # Convert +0000 to +00:00 for fromisoformat.
    if len(text) >= 5 and (text[-5] in "+-") and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]

    dt: Optional[datetime]
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        dt = None

    if dt is None:
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d",
        ):
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def to_schwab_time(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")


def max_timestamp_from_records(
    records: Sequence[Dict[str, Any]],
    keys: Sequence[str],
    fallback: datetime,
) -> datetime:
    best = fallback
    for row in records:
        for key in keys:
            ts = parse_timestamp(row.get(key))
            if ts and ts > best:
                best = ts
                break
    return best


# ================================================================
# TOKEN DETECTION — auto-detect format
# ================================================================

def resolve_token_path(token_path: Optional[str]) -> Path:
    """Resolve token paths and translate old Windows defaults on macOS/Linux."""
    raw = str(token_path or DEFAULT_TOKEN_PATH).strip()
    if not raw:
        raw = DEFAULT_TOKEN_PATH

    # Old commands used C:\uw_root\tokens. On macOS this is a relative filename
    # containing backslashes, so map it to this checkout.
    normalized = raw.replace("\\", "/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    lower = normalized.lower()
    if lower in {"c:/uw_root/tokens", "c:/uw_root/tokens/"}:
        return project_root() / "tokens"
    if lower.startswith("c:/uw_root/") and os.name != "nt":
        return project_root() / normalized[len("c:/uw_root/") :]

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()


def find_tokens(token_path):
    """Auto-detect token file format at the given path."""
    path = resolve_token_path(token_path)

    tokens = {
        "access_token": None,
        "refresh_token": None,
        "client_id": None,
        "client_secret": None,
        "token_expiry": None,
    }

    # Case 1: Path is a file
    if path.is_file():
        return _parse_token_file(path, tokens)

    # Case 2: Path is a directory — scan for token files
    if path.is_dir():
        candidates = [
            "token.json",
            "tokens.json",
            "schwab_token.json",
            "credentials.json",
            "auth.json",
            "api_token.json",
            ".token",
            "access_token.json",
            "schwab.json",
            "token.dat",
            "token.txt",
        ]

        json_files = list(path.glob("*.json"))
        dat_files = list(path.glob("*.dat"))
        txt_files = list(path.glob("*.txt"))
        env_files = list(path.glob(".env*"))

        all_files = []
        for name in candidates:
            f = path / name
            if f.exists():
                all_files.insert(0, f)
        all_files.extend(json_files)
        all_files.extend(dat_files)
        all_files.extend(txt_files)
        all_files.extend(env_files)

        seen = set()
        unique = []
        for f in all_files:
            if str(f) not in seen:
                seen.add(str(f))
                unique.append(f)

        print(f"\nScanning {path} for token files...")
        print(f"Found {len(unique)} candidate files:")
        for f in unique[:10]:
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            print(f"  {f.name} ({size} bytes)")

        for f in unique:
            result = _parse_token_file(f, tokens.copy())
            if result and (result.get("access_token") or result.get("refresh_token")):
                print(f"\n✅ Tokens found in: {f.name}")
                return result

        print("\n❌ No valid tokens found in directory.")
        print("   Expected a JSON file with 'access_token' or 'refresh_token'")
        return None

    print(f"\n❌ Path not found: {path}")
    return None


def _parse_token_file(filepath: Path, tokens: Dict[str, Any]):
    """Try to parse a single file for Schwab tokens."""
    try:
        content = filepath.read_text(encoding="utf-8").strip()
    except Exception:
        return None

    # Try JSON
    try:
        data = json.loads(content)
        return _extract_from_dict(data, tokens)
    except json.JSONDecodeError:
        pass

    # Try .env format
    if "=" in content:
        data: Dict[str, Any] = {}
        for line in content.split("\n"):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                key, _, val = line.partition("=")
                data[key.strip().lower()] = val.strip().strip('"').strip("'")
        if data:
            return _extract_from_dict(data, tokens)

    # Try raw token string
    if len(content) > 20 and "\n" not in content:
        tokens["access_token"] = content
        return tokens

    return None


def _extract_from_dict(data: Dict[str, Any], tokens: Dict[str, Any]):
    """Extract token fields from a dictionary (handles nested formats)."""
    if "token" in data and isinstance(data["token"], dict):
        inner = data["token"]
        data = {**data, **inner}

    field_map = {
        "access_token": ["access_token", "accessToken", "access"],
        "refresh_token": ["refresh_token", "refreshToken", "refresh"],
        "client_id": [
            "client_id",
            "clientId",
            "app_key",
            "appKey",
            "api_key",
            "apiKey",
            "consumer_key",
        ],
        "client_secret": ["client_secret", "clientSecret", "app_secret", "appSecret", "api_secret"],
        "token_expiry": ["creation_timestamp", "expires_at", "expiry", "token_expiry"],
    }

    for target, sources in field_map.items():
        for src in sources:
            if src in data:
                tokens[target] = data[src]
                break
            for k, v in data.items():
                if str(k).lower() == src.lower():
                    tokens[target] = v
                    break

    return tokens


def _save_refreshed_token(token_path: str, tokens: Dict[str, Any]):
    """Save refreshed tokens back to the token file."""
    path = resolve_token_path(token_path)
    if path.is_dir():
        token_file = path / "schwab_token.json"
    elif path.is_file():
        token_file = path
    else:
        return

    try:
        data = json.loads(token_file.read_text(encoding="utf-8"))
        data["access_token"] = tokens["access_token"]
        if tokens.get("refresh_token"):
            data["refresh_token"] = tokens["refresh_token"]
        token_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  💾 Token saved to {token_file.name}")
    except Exception as exc:
        print(f"  ⚠️ Could not save token: {exc}")


def refresh_access_token(tokens: Dict[str, Any]):
    """Use refresh_token to get a new access_token."""
    if not tokens.get("refresh_token"):
        print("❌ No refresh_token available. Cannot refresh.")
        return None
    if not tokens.get("client_id") or not tokens.get("client_secret"):
        print("❌ Need client_id and client_secret to refresh token.")
        print("   These should be in your token file or set as env vars:")
        print("   SCHWAB_CLIENT_ID, SCHWAB_CLIENT_SECRET")

        tokens["client_id"] = tokens.get("client_id") or os.environ.get("SCHWAB_CLIENT_ID")
        tokens["client_secret"] = tokens.get("client_secret") or os.environ.get("SCHWAB_CLIENT_SECRET")

        if not tokens.get("client_id") or not tokens.get("client_secret"):
            return None

    print("Refreshing access token...")
    try:
        resp = requests.post(
            SCHWAB_AUTH_URL,
            data={"grant_type": "refresh_token", "refresh_token": tokens["refresh_token"]},
            auth=(tokens["client_id"], tokens["client_secret"]),
            timeout=20,
        )

        if resp.status_code == 200:
            new_tokens = resp.json()
            tokens["access_token"] = new_tokens["access_token"]
            if "refresh_token" in new_tokens:
                tokens["refresh_token"] = new_tokens["refresh_token"]
            print("✅ Token refreshed successfully")
            return tokens

        print(f"❌ Token refresh failed: {resp.status_code}")
        print(f"   {resp.text[:200]}")
        return None
    except Exception as exc:
        print(f"❌ Token refresh error: {exc}")
        return None


# ================================================================
# SCHWAB API CALLS
# ================================================================

def schwab_get(endpoint: str, access_token: str, params: Optional[Dict[str, Any]] = None):
    """Make authenticated GET request to Schwab API."""
    headers = {"Authorization": f"Bearer {access_token}"}
    url = f"{SCHWAB_API_BASE}{endpoint}"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        if resp.status_code == 401:
            return {"error": "UNAUTHORIZED", "status": 401}
        if resp.status_code != 200:
            return {"error": resp.text[:200], "status": resp.status_code}
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


def get_accounts(access_token: str):
    """Get all linked accounts."""
    return schwab_get("/trader/v1/accounts", access_token)


def get_account_numbers(access_token: str):
    """Get accountNumber -> hashValue mapping."""
    return schwab_get("/trader/v1/accounts/accountNumbers", access_token)


def get_positions(access_token: str, account_hash: str):
    """Get positions for an account."""
    return schwab_get(
        f"/trader/v1/accounts/{account_hash}",
        access_token,
        params={"fields": "positions"},
    )


def get_transactions(
    access_token: str,
    account_hash: str,
    days: int = 90,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
):
    """Get transactions for a date window (used for rolling fills + transactions)."""
    end = end_dt or utc_now()
    start = start_dt or (end - timedelta(days=days))
    return schwab_get(
        f"/trader/v1/accounts/{account_hash}/transactions",
        access_token,
        params={
            "startDate": to_schwab_time(start),
            "endDate": to_schwab_time(end),
            "types": "TRADE",
        },
    )


def get_orders(
    access_token: str,
    account_hash: str,
    days: int = 30,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
):
    """Get orders for a date window (used for rolling orders pull)."""
    end = end_dt or utc_now()
    start = start_dt or (end - timedelta(days=days))
    return schwab_get(
        f"/trader/v1/accounts/{account_hash}/orders",
        access_token,
        params={
            "fromEnteredTime": to_schwab_time(start),
            "toEnteredTime": to_schwab_time(end),
        },
    )


# ================================================================
# LOCAL STATE STORE (SQLite + JSONL)
# ================================================================

class SchwabStateStore:
    def __init__(self, db_path: Path, state_dir: Path) -> None:
        self.db_path = db_path
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS sync_state (
                    account_hash TEXT PRIMARY KEY,
                    last_successful_ts TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS raw_orders (
                    account_hash TEXT NOT NULL,
                    record_key TEXT NOT NULL,
                    order_id TEXT,
                    entered_time TEXT,
                    status TEXT,
                    payload TEXT NOT NULL,
                    pulled_at TEXT NOT NULL,
                    PRIMARY KEY (account_hash, record_key)
                );

                CREATE TABLE IF NOT EXISTS raw_transactions (
                    account_hash TEXT NOT NULL,
                    record_key TEXT NOT NULL,
                    transaction_id TEXT,
                    order_id TEXT,
                    transaction_date TEXT,
                    net_amount REAL,
                    payload TEXT NOT NULL,
                    pulled_at TEXT NOT NULL,
                    PRIMARY KEY (account_hash, record_key)
                );

                CREATE TABLE IF NOT EXISTS raw_executions (
                    account_hash TEXT NOT NULL,
                    execution_key TEXT NOT NULL,
                    broker_execution_id TEXT,
                    transaction_id TEXT,
                    order_id TEXT,
                    engine_trade_id TEXT,
                    ticker TEXT,
                    symbol TEXT,
                    expiry TEXT,
                    instruction TEXT,
                    position_effect TEXT,
                    is_opening INTEGER NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    net_amount REAL NOT NULL,
                    executed_at TEXT,
                    payload TEXT NOT NULL,
                    pulled_at TEXT NOT NULL,
                    PRIMARY KEY (account_hash, execution_key)
                );

                CREATE TABLE IF NOT EXISTS open_trade_metadata (
                    account_hash TEXT NOT NULL,
                    engine_trade_id TEXT NOT NULL,
                    broker_order_id TEXT,
                    track TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    expiry TEXT,
                    intended_legs TEXT NOT NULL,
                    opened_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (account_hash, engine_trade_id)
                );

                CREATE INDEX IF NOT EXISTS idx_open_trade_metadata_order
                ON open_trade_metadata (account_hash, broker_order_id);

                CREATE TABLE IF NOT EXISTS closed_trades (
                    account_hash TEXT NOT NULL,
                    engine_trade_id TEXT NOT NULL,
                    track TEXT,
                    ticker TEXT,
                    strategy TEXT,
                    expiry TEXT,
                    opened_at TEXT,
                    closed_at TEXT,
                    quantity REAL NOT NULL,
                    realized_pnl REAL NOT NULL,
                    entry_order_ids TEXT NOT NULL,
                    exit_order_ids TEXT NOT NULL,
                    source_group TEXT NOT NULL,
                    reconciled_at TEXT NOT NULL,
                    PRIMARY KEY (account_hash, engine_trade_id)
                );
                """
            )
            self._ensure_column(conn, "raw_executions", "broker_execution_id", "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        cols = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def _append_jsonl(self, name: str, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        path = self.state_dir / name
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")

    def _write_jsonl(self, name: str, rows: Sequence[Dict[str, Any]]) -> Path:
        path = self.state_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
        return path

    def get_last_successful_ts(self, account_hash: str) -> Optional[datetime]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT last_successful_ts FROM sync_state WHERE account_hash=?",
                (account_hash,),
            ).fetchone()
        if not row:
            return None
        return parse_timestamp(row["last_successful_ts"])

    def set_last_successful_ts(self, account_hash: str, ts: datetime) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sync_state(account_hash, last_successful_ts, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(account_hash) DO UPDATE SET
                    last_successful_ts=excluded.last_successful_ts,
                    updated_at=excluded.updated_at
                """,
                (account_hash, ts.astimezone(timezone.utc).isoformat(), local_now_iso()),
            )

    def store_raw_orders(self, account_hash: str, orders: Sequence[Dict[str, Any]], account_tag: str) -> int:
        rows_written = 0
        pulled_at = local_now_iso()
        json_rows: List[Dict[str, Any]] = []
        with self._connect() as conn:
            for order in orders:
                order_id = str(order.get("orderId") or "").strip()
                if not order_id:
                    continue
                entered = str(order.get("enteredTime", ""))
                record_key = f"ORDER:{order_id}"
                conn.execute(
                    """
                    INSERT INTO raw_orders(
                        account_hash, record_key, order_id, entered_time, status, payload, pulled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_hash, record_key) DO UPDATE SET
                        record_key=excluded.record_key,
                        order_id=excluded.order_id,
                        entered_time=excluded.entered_time,
                        status=excluded.status,
                        payload=excluded.payload,
                        pulled_at=excluded.pulled_at
                    """,
                    (
                        account_hash,
                        record_key,
                        order_id or None,
                        entered or None,
                        str(order.get("status", "")),
                        json.dumps(order, sort_keys=True),
                        pulled_at,
                    ),
                )
                rows_written += 1
                json_rows.append(order)
        self._append_jsonl(f"raw_orders_{account_tag}.jsonl", json_rows)
        return rows_written

    def store_raw_transactions(self, account_hash: str, txns: Sequence[Dict[str, Any]], account_tag: str) -> int:
        rows_written = 0
        pulled_at = local_now_iso()
        json_rows: List[Dict[str, Any]] = []
        with self._connect() as conn:
            for txn in txns:
                txn_id = str(txn.get("transactionId") or txn.get("activityId") or "").strip()
                if not txn_id:
                    continue
                txn_date = str(txn.get("transactionDate") or txn.get("time") or txn.get("tradeDate") or "")
                order_id = str(txn.get("orderId") or "").strip()
                record_key = f"TXN:{txn_id}"
                conn.execute(
                    """
                    INSERT INTO raw_transactions(
                        account_hash, record_key, transaction_id, order_id,
                        transaction_date, net_amount, payload, pulled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_hash, record_key) DO UPDATE SET
                        record_key=excluded.record_key,
                        transaction_id=excluded.transaction_id,
                        order_id=excluded.order_id,
                        transaction_date=excluded.transaction_date,
                        net_amount=excluded.net_amount,
                        payload=excluded.payload,
                        pulled_at=excluded.pulled_at
                    """,
                    (
                        account_hash,
                        record_key,
                        txn_id or None,
                        order_id or None,
                        txn_date or None,
                        to_float(txn.get("netAmount")),
                        json.dumps(txn, sort_keys=True),
                        pulled_at,
                    ),
                )
                rows_written += 1
                json_rows.append(txn)
        self._append_jsonl(f"raw_transactions_{account_tag}.jsonl", json_rows)
        return rows_written

    def store_raw_executions(self, account_hash: str, executions: Sequence[Dict[str, Any]], account_tag: str) -> int:
        rows_written = 0
        pulled_at = local_now_iso()
        json_rows: List[Dict[str, Any]] = []
        with self._connect() as conn:
            for ex in executions:
                broker_execution_id = str(
                    ex.get("broker_execution_id")
                    or ex.get("execution_id")
                    or ex.get("execution_key")
                    or ""
                ).strip()
                execution_key = str(ex.get("execution_key") or broker_execution_id).strip()
                if not execution_key:
                    continue
                conn.execute(
                    """
                    INSERT INTO raw_executions(
                        account_hash, execution_key, broker_execution_id, transaction_id, order_id, engine_trade_id,
                        ticker, symbol, expiry, instruction, position_effect, is_opening,
                        quantity, price, net_amount, executed_at, payload, pulled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_hash, execution_key) DO UPDATE SET
                        execution_key=excluded.execution_key,
                        broker_execution_id=excluded.broker_execution_id,
                        transaction_id=excluded.transaction_id,
                        order_id=excluded.order_id,
                        engine_trade_id=excluded.engine_trade_id,
                        ticker=excluded.ticker,
                        symbol=excluded.symbol,
                        expiry=excluded.expiry,
                        instruction=excluded.instruction,
                        position_effect=excluded.position_effect,
                        is_opening=excluded.is_opening,
                        quantity=excluded.quantity,
                        price=excluded.price,
                        net_amount=excluded.net_amount,
                        executed_at=excluded.executed_at,
                        payload=excluded.payload,
                        pulled_at=excluded.pulled_at
                    """,
                    (
                        account_hash,
                        execution_key,
                        broker_execution_id,
                        ex.get("transaction_id"),
                        ex.get("order_id"),
                        ex.get("engine_trade_id"),
                        ex.get("ticker"),
                        ex.get("symbol"),
                        ex.get("expiry"),
                        ex.get("instruction"),
                        ex.get("position_effect"),
                        1 if ex.get("is_opening") else 0,
                        to_float(ex.get("quantity")),
                        to_float(ex.get("price")),
                        to_float(ex.get("net_amount")),
                        ex.get("executed_at"),
                        json.dumps(ex.get("raw", {}), sort_keys=True),
                        pulled_at,
                    ),
                )
                rows_written += 1
                json_rows.append(
                    {
                        "broker_execution_id": broker_execution_id,
                        "execution_key": execution_key,
                        "transaction_id": ex.get("transaction_id"),
                        "order_id": ex.get("order_id"),
                        "engine_trade_id": ex.get("engine_trade_id"),
                        "ticker": ex.get("ticker"),
                        "symbol": ex.get("symbol"),
                        "expiry": ex.get("expiry"),
                        "instruction": ex.get("instruction"),
                        "position_effect": ex.get("position_effect"),
                        "is_opening": bool(ex.get("is_opening")),
                        "quantity": to_float(ex.get("quantity")),
                        "price": to_float(ex.get("price")),
                        "net_amount": to_float(ex.get("net_amount")),
                        "executed_at": ex.get("executed_at"),
                    }
                )
        self._append_jsonl(f"raw_executions_{account_tag}.jsonl", json_rows)
        return rows_written

    def upsert_open_trade_metadata(
        self,
        account_hash: str,
        *,
        engine_trade_id: str,
        broker_order_id: Optional[str],
        track: str,
        ticker: str,
        strategy: str,
        expiry: Optional[str],
        intended_legs: Sequence[Dict[str, Any]],
        opened_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO open_trade_metadata(
                    account_hash, engine_trade_id, broker_order_id, track,
                    ticker, strategy, expiry, intended_legs, opened_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(account_hash, engine_trade_id) DO UPDATE SET
                    broker_order_id=excluded.broker_order_id,
                    track=excluded.track,
                    ticker=excluded.ticker,
                    strategy=excluded.strategy,
                    expiry=excluded.expiry,
                    intended_legs=excluded.intended_legs,
                    opened_at=excluded.opened_at
                """,
                (
                    account_hash,
                    engine_trade_id,
                    broker_order_id,
                    track,
                    ticker,
                    strategy,
                    expiry,
                    json.dumps(list(intended_legs), sort_keys=True),
                    opened_at,
                    local_now_iso(),
                ),
            )

    def load_open_trade_maps(
        self,
        account_hash: str,
    ) -> Tuple[Dict[str, sqlite3.Row], Dict[str, sqlite3.Row]]:
        by_engine: Dict[str, sqlite3.Row] = {}
        by_order: Dict[str, sqlite3.Row] = {}
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM open_trade_metadata
                WHERE account_hash IN (?, '*')
                ORDER BY CASE WHEN account_hash = ? THEN 0 ELSE 1 END, created_at ASC
                """,
                (account_hash, account_hash),
            ).fetchall()

        for row in rows:
            engine = row["engine_trade_id"]
            if engine and engine not in by_engine:
                by_engine[engine] = row
            order_id = row["broker_order_id"]
            if order_id and order_id not in by_order:
                by_order[order_id] = row

        return by_engine, by_order

    def fetch_raw_executions(self, account_hash: str) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return list(
                conn.execute(
                    """
                    SELECT * FROM raw_executions
                    WHERE account_hash=?
                    ORDER BY executed_at ASC, broker_execution_id ASC, execution_key ASC
                    """,
                    (account_hash,),
                )
            )

    def upsert_closed_trades(self, account_hash: str, rows: Sequence[Dict[str, Any]]) -> None:
        with self._connect() as conn:
            # Full recompute semantics: replace account-level closed_trades snapshot.
            conn.execute("DELETE FROM closed_trades WHERE account_hash = ?", (account_hash,))
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO closed_trades(
                        account_hash, engine_trade_id, track, ticker, strategy, expiry,
                        opened_at, closed_at, quantity, realized_pnl,
                        entry_order_ids, exit_order_ids, source_group, reconciled_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(account_hash, engine_trade_id) DO UPDATE SET
                        track=excluded.track,
                        ticker=excluded.ticker,
                        strategy=excluded.strategy,
                        expiry=excluded.expiry,
                        opened_at=excluded.opened_at,
                        closed_at=excluded.closed_at,
                        quantity=excluded.quantity,
                        realized_pnl=excluded.realized_pnl,
                        entry_order_ids=excluded.entry_order_ids,
                        exit_order_ids=excluded.exit_order_ids,
                        source_group=excluded.source_group,
                        reconciled_at=excluded.reconciled_at
                    """,
                    (
                        account_hash,
                        row["engine_trade_id"],
                        row.get("track"),
                        row.get("ticker"),
                        row.get("strategy"),
                        row.get("expiry"),
                        row.get("opened_at"),
                        row.get("closed_at"),
                        to_float(row.get("quantity")),
                        to_float(row.get("realized_pnl")),
                        json.dumps(row.get("entry_order_ids", []), sort_keys=True),
                        json.dumps(row.get("exit_order_ids", []), sort_keys=True),
                        row.get("source_group", ""),
                        local_now_iso(),
                    ),
                )

    def fetch_closed_trades(self, account_hash: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT engine_trade_id, track, ticker, strategy, expiry,
                           opened_at, closed_at, quantity, realized_pnl,
                           entry_order_ids, exit_order_ids, source_group
                    FROM closed_trades
                    WHERE account_hash=?
                    ORDER BY closed_at ASC, engine_trade_id ASC
                    """,
                    (account_hash,),
                )
            )

        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "engine_trade_id": row["engine_trade_id"],
                    "track": row["track"] or "UNKNOWN",
                    "ticker": row["ticker"] or "",
                    "strategy": row["strategy"] or "unknown",
                    "expiry": row["expiry"] or "",
                    "opened_at": row["opened_at"] or "",
                    "closed_at": row["closed_at"] or "",
                    "quantity": to_float(row["quantity"]),
                    "realized_pnl": to_float(row["realized_pnl"]),
                    "entry_order_ids": json.loads(row["entry_order_ids"] or "[]"),
                    "exit_order_ids": json.loads(row["exit_order_ids"] or "[]"),
                    "source_group": row["source_group"] or "",
                }
            )
        return out

    def write_reconciled_jsonl(self, account_tag: str, closed_rows: Sequence[Dict[str, Any]]) -> Path:
        return self._write_jsonl(f"closed_trades_{account_tag}.jsonl", list(closed_rows))

    def write_health_gate_report(self, account_tag: str, report: Dict[str, Any]) -> Path:
        return self._write_jsonl(f"health_gate_{account_tag}.jsonl", [report])


# ================================================================
# RECONCILIATION + HEALTH GATE
# ================================================================

def _parse_option_symbol(symbol: str) -> Tuple[float, str, str]:
    """Parse Schwab option symbol into strike, expiry, call/put."""
    import re

    text = str(symbol or "").strip()
    m = re.match(r"^([A-Z]+)\s*(\d{6})([CP])(\d{8})$", text)
    if not m:
        return 0.0, "", ""

    date_str = m.group(2)
    opt_type = "CALL" if m.group(3) == "C" else "PUT"
    strike = int(m.group(4)) / 1000.0

    try:
        expiry = datetime.strptime("20" + date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except ValueError:
        expiry = f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"

    return strike, expiry, opt_type


def is_option_instrument(inst: Dict[str, Any], symbol_hint: Optional[str] = None) -> bool:
    asset_type = str(inst.get("assetType") or "").strip().upper()
    if asset_type == "OPTION":
        return True
    if asset_type:
        return False

    symbol = str(symbol_hint or inst.get("symbol") or "").strip()
    if not symbol:
        return False
    _, expiry, opt_type = _parse_option_symbol(symbol)
    return bool(expiry and opt_type)


def order_is_option_only(order: Dict[str, Any]) -> bool:
    legs = order.get("orderLegCollection", []) or []
    if not legs:
        return False
    has_option_leg = False
    for leg in legs:
        inst = leg.get("instrument", {}) if isinstance(leg, dict) else {}
        symbol = str(inst.get("symbol") or "")
        if not is_option_instrument(inst, symbol):
            return False
        has_option_leg = True
    return has_option_leg


def filter_option_transactions(transactions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for txn in transactions:
        items = parse_items_from_transaction(txn)
        option_items: List[Dict[str, Any]] = []
        for item in items:
            inst = item.get("instrument", {}) if isinstance(item, dict) else {}
            symbol = str(inst.get("symbol") or item.get("symbol") or "")
            if is_option_instrument(inst, symbol):
                option_items.append(item)
        if not option_items:
            continue

        # Preserve original transaction payload, but strip non-option line items.
        txn_copy = dict(txn)
        if isinstance(txn.get("transactionItems"), list):
            txn_copy["transactionItems"] = option_items
        elif isinstance(txn.get("transactionItem"), list):
            txn_copy["transactionItem"] = option_items
        elif isinstance(txn.get("transactionItem"), dict):
            txn_copy["transactionItem"] = option_items[0]
        filtered.append(txn_copy)
    return filtered


def parse_items_from_transaction(txn: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if isinstance(txn.get("transactionItems"), list):
        items.extend([x for x in txn["transactionItems"] if isinstance(x, dict)])
    if isinstance(txn.get("transferItems"), list):
        items.extend([x for x in txn["transferItems"] if isinstance(x, dict)])
    ti = txn.get("transactionItem")
    if isinstance(ti, list):
        items.extend([x for x in ti if isinstance(x, dict)])
    elif isinstance(ti, dict):
        items.append(ti)

    if not items:
        items.append(txn)

    return items


def infer_opening_fill(instruction: str, position_effect: str) -> bool:
    pe = (position_effect or "").upper()
    ins = (instruction or "").upper().replace(" ", "_")
    if "OPEN" in pe:
        return True
    if pe.startswith("CLOS"):
        return False
    if "CLOSE" in pe:
        return False
    if "OPEN" in ins:
        return True
    if ins.startswith("CLOS"):
        return False
    if "CLOSE" in ins:
        return False
    if ins.startswith("SELL"):
        return False
    return True


def instruction_side(instruction: str, amount_hint: float = 0.0, net_hint: float = 0.0) -> str:
    ins = (instruction or "").upper().replace(" ", "_")
    if ins.startswith("BUY"):
        return "BUY"
    if ins.startswith("SELL"):
        return "SELL"
    if abs(amount_hint) > 1e-9:
        return "BUY" if amount_hint > 0 else "SELL"
    if abs(net_hint) > 1e-9:
        return "SELL" if net_hint > 0 else "BUY"
    return "UNK"


def derive_instruction_from_amount(position_effect: str, amount_hint: float, net_hint: float = 0.0) -> str:
    side = instruction_side("", amount_hint=amount_hint, net_hint=net_hint)
    if side == "UNK":
        return ""
    pe = (position_effect or "").upper()
    closing = pe.startswith("CLOS") or ("CLOSE" in pe)
    return f"{side}_TO_{'CLOSE' if closing else 'OPEN'}"


def infer_cash_effect(
    instruction: str,
    quantity: float,
    price: float,
    multiplier: float,
    item_cost: float,
    txn_net_amount: float,
    item_count: int,
) -> float:
    # Prefer line-item cost when present.
    if abs(item_cost) > 1e-9:
        return item_cost

    # If transaction has single item, use netAmount.
    if item_count == 1 and abs(txn_net_amount) > 1e-9:
        return txn_net_amount

    gross = abs(quantity * price * multiplier)
    ins = (instruction or "").upper().replace(" ", "_")
    if ins.startswith("BUY"):
        return -gross
    if ins.startswith("SELL"):
        return gross
    return txn_net_amount


def extract_executions_from_transactions(transactions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    executions: List[Dict[str, Any]] = []

    for txn in transactions:
        if str(txn.get("type", "")).upper() != "TRADE":
            continue

        txn_id = str(txn.get("transactionId") or txn.get("activityId") or "")
        txn_dt_text = str(txn.get("transactionDate") or txn.get("time") or txn.get("tradeDate") or "")
        txn_net = to_float(txn.get("netAmount"))
        txn_order_id = str(txn.get("orderId") or "")

        items = parse_items_from_transaction(txn)
        for idx, item in enumerate(items):
            if str(item.get("feeType") or "").strip():
                continue

            inst = item.get("instrument", {}) if isinstance(item, dict) else {}
            symbol = str(inst.get("symbol") or item.get("symbol") or "")
            if not is_option_instrument(inst, symbol):
                continue
            if not symbol:
                continue

            underlying = str(inst.get("underlyingSymbol") or "")
            if not underlying and symbol:
                underlying = symbol.split()[0]
            _, expiry, _ = _parse_option_symbol(symbol)

            instruction = str(
                item.get("instruction")
                or item.get("positionEffect")
                or txn.get("instruction")
                or txn.get("positionEffect")
                or ""
            )
            position_effect = str(item.get("positionEffect") or txn.get("positionEffect") or "")
            qty_signed = to_float(item.get("amount", item.get("quantity", txn.get("amount", 0.0))))
            qty = abs(qty_signed)
            if qty <= 1e-9:
                continue
            price = to_float(item.get("price", txn.get("price", 0.0)))
            multiplier = to_float(
                item.get("multiplier", inst.get("optionPremiumMultiplier", 100.0)),
                default=100.0,
            )
            item_cost = to_float(item.get("cost", 0.0))
            if not instruction:
                instruction = derive_instruction_from_amount(position_effect, qty_signed, net_hint=item_cost)
            if price == 0.0 and qty > 0 and multiplier > 0 and abs(item_cost) > 0:
                price = abs(item_cost) / (qty * multiplier)
            executed_at = str(item.get("transactionDate") or item.get("time") or item.get("tradeDate") or txn_dt_text)
            order_id = str(item.get("orderId") or txn_order_id or "")
            broker_execution_id = str(
                item.get("executionId")
                or item.get("activityId")
                or item.get("transactionItemId")
                or txn.get("executionId")
                or ""
            ).strip()
            if not broker_execution_id:
                if txn_id:
                    broker_execution_id = f"{txn_id}:{idx}"
                else:
                    broker_execution_id = hashlib.sha1(
                        f"{order_id}|{executed_at}|{symbol}|{idx}".encode("utf-8")
                    ).hexdigest()
            engine_trade_id = str(
                item.get("engine_trade_id")
                or item.get("engineTradeId")
                or txn.get("engine_trade_id")
                or txn.get("engineTradeId")
                or ""
            )

            cash_effect = infer_cash_effect(
                instruction=instruction,
                quantity=qty,
                price=price,
                multiplier=multiplier,
                item_cost=item_cost,
                txn_net_amount=txn_net,
                item_count=len(items),
            )

            seed = "|".join(
                [
                    txn_id,
                    order_id,
                    executed_at,
                    symbol,
                    instruction,
                    f"{qty:.8f}",
                    f"{price:.8f}",
                    f"{cash_effect:.8f}",
                    str(idx),
                ]
            )
            execution_key = hashlib.sha1(seed.encode("utf-8")).hexdigest()

            executions.append(
                {
                    "execution_key": execution_key,
                    "broker_execution_id": broker_execution_id,
                    "transaction_id": txn_id or None,
                    "order_id": order_id or None,
                    "engine_trade_id": engine_trade_id or None,
                    "ticker": underlying,
                    "symbol": symbol,
                    "expiry": expiry,
                    "instruction": instruction,
                    "position_effect": position_effect,
                    "is_opening": infer_opening_fill(instruction, position_effect),
                    "quantity": qty,
                    "price": price,
                    "net_amount": cash_effect,
                    "executed_at": executed_at,
                    "raw": {
                        "transaction": txn,
                        "item": item,
                    },
                }
            )

    return executions


def occ_last_resort_group(fill: Dict[str, Any]) -> str:
    symbol = str(fill.get("symbol") or fill.get("ticker") or "UNKNOWN").strip() or "UNKNOWN"
    side = str(fill.get("side") or "UNK")
    side_bucket = f"{side}_{'OPEN' if bool(fill.get('is_opening')) else 'CLOSE'}"
    qty_bucket = f"{abs(to_float(fill.get('quantity'))):.4f}"
    exec_dt = parse_timestamp(fill.get("executed_at"))
    if exec_dt is None:
        window_bucket = "NA"
    else:
        seconds = DEFAULT_OCC_TIMESTAMP_WINDOW_MINUTES * 60
        bucket_start_epoch = int(exec_dt.timestamp() // seconds) * seconds
        bucket_start = datetime.fromtimestamp(bucket_start_epoch, tz=timezone.utc)
        window_bucket = bucket_start.strftime("%Y%m%dT%H%MZ")
    return f"occ:{symbol}:{side_bucket}:{qty_bucket}:{window_bucket}"


def collect_transaction_line_items(parent: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    if isinstance(parent.get("transferItems"), list):
        items.extend([x for x in parent["transferItems"] if isinstance(x, dict)])
    if isinstance(parent.get("transactionItems"), list):
        items.extend([x for x in parent["transactionItems"] if isinstance(x, dict)])
    ti = parent.get("transactionItem")
    if isinstance(ti, list):
        items.extend([x for x in ti if isinstance(x, dict)])
    elif isinstance(ti, dict):
        items.append(ti)
    return items


def extract_option_legs_from_execution_row(ex: sqlite3.Row) -> List[Dict[str, Any]]:
    row = dict(ex)
    base_txn_id = str(row.get("transaction_id") or "")
    base_order_id = str(row.get("order_id") or "")
    base_engine_id = str(row.get("engine_trade_id") or "")
    base_exec_id = str(row.get("broker_execution_id") or row.get("execution_key") or "").strip()
    base_exec_key = str(row.get("execution_key") or base_exec_id or "")
    base_ticker = str(row.get("ticker") or "").strip()
    base_symbol = str(row.get("symbol") or "").strip()
    base_expiry = str(row.get("expiry") or "").strip()
    base_instruction = str(row.get("instruction") or "").strip()
    base_position_effect = str(row.get("position_effect") or "").strip()
    base_net = to_float(row.get("net_amount"))
    base_executed_at = str(row.get("executed_at") or "")
    base_qty = abs(to_float(row.get("quantity")))
    base_price = to_float(row.get("price"))
    base_is_opening = bool(row.get("is_opening"))

    payload: Dict[str, Any] = {}
    try:
        payload_raw = row.get("payload")
        if payload_raw:
            loaded = json.loads(payload_raw)
            if isinstance(loaded, dict):
                payload = loaded
    except Exception:
        payload = {}

    legs: List[Dict[str, Any]] = []
    parents: List[Dict[str, Any]] = []
    for key in ("item", "transaction"):
        parent = payload.get(key)
        if isinstance(parent, dict):
            parents.append(parent)

    for parent_idx, parent in enumerate(parents):
        raw_items = collect_transaction_line_items(parent)
        option_rows: List[Dict[str, Any]] = []
        for item in raw_items:
            inst = item.get("instrument", {}) if isinstance(item.get("instrument"), dict) else {}
            symbol = str(inst.get("symbol") or item.get("symbol") or "").strip()
            if not is_option_instrument(inst, symbol):
                continue
            if not symbol:
                continue
            if str(item.get("feeType") or "").strip():
                continue
            amount_signed = to_float(item.get("amount", item.get("quantity", 0.0)))
            if abs(amount_signed) <= 1e-9:
                continue
            option_rows.append(
                {
                    "item": item,
                    "inst": inst,
                    "symbol": symbol,
                    "amount_signed": amount_signed,
                }
            )

        option_count = len(option_rows)
        if option_count == 0:
            continue

        parent_txn_id = str(parent.get("transactionId") or parent.get("activityId") or base_txn_id)
        parent_order_id = str(parent.get("orderId") or base_order_id)
        parent_executed_at = str(parent.get("transactionDate") or parent.get("time") or parent.get("tradeDate") or base_executed_at)
        parent_net = to_float(parent.get("netAmount"), default=base_net)

        for item_idx, row_item in enumerate(option_rows):
            item = row_item["item"]
            inst = row_item["inst"]
            symbol = row_item["symbol"]
            amount_signed = to_float(row_item["amount_signed"])

            position_effect = str(item.get("positionEffect") or parent.get("positionEffect") or base_position_effect)
            instruction = str(item.get("instruction") or parent.get("instruction") or base_instruction)
            item_cost = to_float(item.get("cost"))
            if not instruction:
                instruction = derive_instruction_from_amount(position_effect, amount_signed, net_hint=item_cost or parent_net)

            quantity = abs(amount_signed)
            if quantity <= 1e-9:
                continue

            multiplier = to_float(
                item.get("multiplier", inst.get("optionPremiumMultiplier", 100.0)),
                default=100.0,
            )
            price = to_float(item.get("price", parent.get("price", base_price)))
            if price == 0.0 and quantity > 0 and multiplier > 0 and abs(item_cost) > 0:
                price = abs(item_cost) / (quantity * multiplier)

            net_amount = item_cost
            if abs(net_amount) <= 1e-9:
                if option_count == 1 and abs(parent_net) > 1e-9:
                    net_amount = parent_net
                else:
                    net_amount = infer_cash_effect(
                        instruction=instruction,
                        quantity=quantity,
                        price=price,
                        multiplier=multiplier,
                        item_cost=item_cost,
                        txn_net_amount=parent_net,
                        item_count=option_count,
                    )

            executed_at = str(
                item.get("transactionDate")
                or item.get("time")
                or item.get("tradeDate")
                or parent_executed_at
                or base_executed_at
            )
            order_id = str(item.get("orderId") or parent_order_id or base_order_id)
            transaction_id = str(item.get("transactionId") or parent_txn_id or base_txn_id)
            broker_execution_id = str(
                item.get("executionId")
                or item.get("activityId")
                or item.get("transactionItemId")
                or parent.get("executionId")
                or parent.get("activityId")
                or base_exec_id
                or ""
            ).strip()
            if not broker_execution_id:
                broker_execution_id = f"{base_exec_key or transaction_id or 'fill'}:{parent_idx}:{item_idx}"

            underlying = str(inst.get("underlyingSymbol") or base_ticker)
            if not underlying and symbol:
                underlying = symbol.split()[0]
            _, parsed_expiry, _ = _parse_option_symbol(symbol)
            expiry = parsed_expiry or base_expiry
            side = instruction_side(instruction, amount_hint=amount_signed, net_hint=net_amount)
            is_opening = infer_opening_fill(instruction, position_effect)

            event_seed = "|".join(
                [
                    base_exec_key,
                    transaction_id,
                    order_id,
                    executed_at,
                    symbol,
                    instruction,
                    position_effect,
                    f"{quantity:.8f}",
                    f"{net_amount:.8f}",
                    str(parent_idx),
                    str(item_idx),
                ]
            )
            event_key = hashlib.sha1(event_seed.encode("utf-8")).hexdigest()

            legs.append(
                {
                    "event_key": event_key,
                    "broker_execution_id": broker_execution_id,
                    "transaction_id": transaction_id,
                    "order_id": order_id,
                    "engine_trade_id": base_engine_id,
                    "ticker": underlying,
                    "symbol": symbol,
                    "expiry": expiry,
                    "instruction": instruction,
                    "position_effect": position_effect,
                    "is_opening": is_opening,
                    "side": side,
                    "quantity": quantity,
                    "price": price,
                    "net_amount": net_amount,
                    "executed_at": executed_at,
                    "source": "payload",
                }
            )

    if not legs and base_symbol and base_qty > 1e-9:
        inst = {"symbol": base_symbol, "assetType": "OPTION"}
        if is_option_instrument(inst, base_symbol):
            side = instruction_side(base_instruction, net_hint=base_net)
            is_opening = base_is_opening
            if base_instruction or base_position_effect:
                is_opening = infer_opening_fill(base_instruction, base_position_effect)
            if side == "UNK":
                side = "SELL" if base_net > 0 else "BUY"
            event_seed = "|".join(
                [
                    base_exec_key,
                    base_txn_id,
                    base_order_id,
                    base_executed_at,
                    base_symbol,
                    base_instruction,
                    base_position_effect,
                    f"{base_qty:.8f}",
                    f"{base_net:.8f}",
                    "row",
                ]
            )
            event_key = hashlib.sha1(event_seed.encode("utf-8")).hexdigest()
            legs.append(
                {
                    "event_key": event_key,
                    "broker_execution_id": base_exec_id or base_exec_key,
                    "transaction_id": base_txn_id,
                    "order_id": base_order_id,
                    "engine_trade_id": base_engine_id,
                    "ticker": base_ticker or base_symbol.split()[0],
                    "symbol": base_symbol,
                    "expiry": base_expiry,
                    "instruction": base_instruction,
                    "position_effect": base_position_effect,
                    "is_opening": is_opening,
                    "side": side,
                    "quantity": base_qty,
                    "price": base_price,
                    "net_amount": base_net,
                    "executed_at": base_executed_at,
                    "source": "row",
                }
            )

    dedup: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for leg in legs:
        dt = parse_timestamp(leg.get("executed_at"))
        ts_key = dt.isoformat() if dt else str(leg.get("executed_at") or "")
        sig = (
            str(leg.get("transaction_id") or ""),
            str(leg.get("order_id") or ""),
            str(leg.get("symbol") or ""),
            bool(leg.get("is_opening")),
            str(leg.get("side") or ""),
            round(to_float(leg.get("quantity")), 8),
            round(to_float(leg.get("net_amount")), 4),
            ts_key,
        )
        existing = dedup.get(sig)
        if existing is None:
            dedup[sig] = leg
            continue
        # Prefer payload-derived rows over legacy row-level rows.
        if existing.get("source") != "payload" and leg.get("source") == "payload":
            dedup[sig] = leg

    return list(dedup.values())


def normalize_execution_legs(fills: Sequence[sqlite3.Row]) -> List[Dict[str, Any]]:
    all_legs: List[Dict[str, Any]] = []
    for ex in fills:
        all_legs.extend(extract_option_legs_from_execution_row(ex))

    dedup: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for leg in all_legs:
        dt = parse_timestamp(leg.get("executed_at"))
        ts_key = dt.isoformat() if dt else str(leg.get("executed_at") or "")
        sig = (
            str(leg.get("transaction_id") or ""),
            str(leg.get("order_id") or ""),
            str(leg.get("symbol") or ""),
            bool(leg.get("is_opening")),
            str(leg.get("side") or ""),
            round(to_float(leg.get("quantity")), 8),
            round(to_float(leg.get("net_amount")), 4),
            ts_key,
        )
        if sig not in dedup:
            dedup[sig] = leg

    def sort_key(leg: Dict[str, Any]) -> Tuple[datetime, str, str, str]:
        dt = parse_timestamp(leg.get("executed_at"))
        if dt is None:
            dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return (
            dt,
            str(leg.get("order_id") or ""),
            str(leg.get("transaction_id") or ""),
            str(leg.get("event_key") or ""),
        )

    return sorted(dedup.values(), key=sort_key)


def resolve_open_package_identity(
    fill: Dict[str, Any],
    metadata_by_engine: Dict[str, sqlite3.Row],
    metadata_by_order: Dict[str, sqlite3.Row],
) -> Tuple[str, Dict[str, str]]:
    raw_engine = str(fill.get("engine_trade_id") or "").strip()
    raw_order = str(fill.get("order_id") or "").strip()
    raw_execution = str(fill.get("broker_execution_id") or fill.get("event_key") or "").strip()

    fallback_ticker = str(fill.get("ticker") or "").strip() or str(fill.get("symbol") or "").split()[0]
    fallback_expiry = str(fill.get("expiry") or "").strip()
    meta = {
        "engine_trade_id": "",
        "track": "UNKNOWN",
        "ticker": fallback_ticker,
        "strategy": "unknown",
        "expiry": fallback_expiry,
        "opened_at": "",
    }

    if raw_engine and raw_engine in metadata_by_engine:
        row = metadata_by_engine[raw_engine]
        return (
            f"engine:{raw_engine}",
            {
                "engine_trade_id": raw_engine,
                "track": str(row["track"] or "UNKNOWN"),
                "ticker": str(row["ticker"] or fallback_ticker),
                "strategy": str(row["strategy"] or "unknown"),
                "expiry": str(row["expiry"] or fallback_expiry),
                "opened_at": str(row["opened_at"] or ""),
            },
        )

    if raw_engine:
        meta["engine_trade_id"] = raw_engine
        return (f"engine:{raw_engine}", meta)

    if raw_order and raw_order in metadata_by_order:
        row = metadata_by_order[raw_order]
        mapped_engine = str(row["engine_trade_id"] or "").strip()
        mapped_meta = {
            "engine_trade_id": mapped_engine,
            "track": str(row["track"] or "UNKNOWN"),
            "ticker": str(row["ticker"] or fallback_ticker),
            "strategy": str(row["strategy"] or "unknown"),
            "expiry": str(row["expiry"] or fallback_expiry),
            "opened_at": str(row["opened_at"] or ""),
        }
        if mapped_engine:
            return (f"engine:{mapped_engine}", mapped_meta)
        return (f"order:{raw_order}", mapped_meta)

    if raw_order:
        return (f"order:{raw_order}", meta)

    if raw_execution:
        return (f"execution:{raw_execution}", meta)

    return (occ_last_resort_group(fill), meta)


def resolve_close_preferred_package(
    fill: Dict[str, Any],
    metadata_by_order: Dict[str, sqlite3.Row],
) -> Optional[str]:
    raw_engine = str(fill.get("engine_trade_id") or "").strip()
    if raw_engine:
        return f"engine:{raw_engine}"

    raw_order = str(fill.get("order_id") or "").strip()
    if raw_order and raw_order in metadata_by_order:
        row = metadata_by_order[raw_order]
        mapped_engine = str(row["engine_trade_id"] or "").strip()
        if mapped_engine:
            return f"engine:{mapped_engine}"
    return None


def infer_strategy_for_package(pkg: Dict[str, Any]) -> str:
    existing = str(pkg.get("strategy") or "").strip()
    if existing and existing.lower() != "unknown":
        return existing

    symbols = sorted([s for s in pkg.get("open_by_symbol", {}).keys() if s])
    if len(symbols) >= 2:
        parsed = [_parse_option_symbol(s) for s in symbols]
        expiries = {x[1] for x in parsed if x[1]}
        call_put = {x[2] for x in parsed if x[2]}
        if len(symbols) == 2 and len(expiries) == 1 and len(call_put) == 1:
            return "vertical_spread"
        return "multi_leg_options"

    if len(symbols) == 1:
        symbol = symbols[0]
        _, _, opt_type = _parse_option_symbol(symbol)
        side_set = pkg.get("open_side_by_symbol", {}).get(symbol, set())
        side = "LONG"
        if "SHORT" in side_set and "LONG" not in side_set:
            side = "SHORT"
        if opt_type == "CALL":
            return "long_call" if side == "LONG" else "short_call"
        if opt_type == "PUT":
            return "long_put" if side == "LONG" else "short_put"

    return "unknown"


def reconcile_closed_trades_for_account(store: SchwabStateStore, account_hash: str) -> List[Dict[str, Any]]:
    metadata_by_engine, metadata_by_order = store.load_open_trade_maps(account_hash)
    normalized_fills = normalize_execution_legs(store.fetch_raw_executions(account_hash))

    eps = 1e-9
    packages: Dict[str, Dict[str, Any]] = {}
    open_lots: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for fill in normalized_fills:
        qty = abs(to_float(fill.get("quantity")))
        if qty <= eps:
            continue
        symbol = str(fill.get("symbol") or "").strip()
        if not symbol:
            continue

        order_id = str(fill.get("order_id") or "").strip()
        executed_dt = parse_timestamp(fill.get("executed_at"))
        side = str(fill.get("side") or "")
        if side not in ("BUY", "SELL"):
            side = "SELL" if to_float(fill.get("net_amount")) > 0 else "BUY"

        if bool(fill.get("is_opening")):
            package_key, meta = resolve_open_package_identity(fill, metadata_by_engine, metadata_by_order)
            pkg = packages.get(package_key)
            if pkg is None:
                pkg = {
                    "source_group": package_key,
                    "engine_trade_id": meta.get("engine_trade_id", ""),
                    "track": meta.get("track", "UNKNOWN"),
                    "ticker": meta.get("ticker", ""),
                    "strategy": meta.get("strategy", "unknown"),
                    "expiry": meta.get("expiry", ""),
                    "opened_at": meta.get("opened_at", ""),
                    "realized_pnl": 0.0,
                    "entry_order_ids": set(),
                    "exit_order_ids": set(),
                    "first_open_time": None,
                    "last_close_time": None,
                    "open_by_symbol": defaultdict(float),
                    "closed_by_symbol": defaultdict(float),
                    "remaining_by_symbol": defaultdict(float),
                    "open_side_by_symbol": defaultdict(set),
                }
                packages[package_key] = pkg

            for field in ("track", "ticker", "strategy", "expiry", "opened_at"):
                current = str(pkg.get(field) or "")
                incoming = str(meta.get(field) or "")
                if (not current or current in ("UNKNOWN", "unknown")) and incoming:
                    pkg[field] = incoming

            pkg["engine_trade_id"] = str(pkg.get("engine_trade_id") or meta.get("engine_trade_id") or "")
            if order_id:
                pkg["entry_order_ids"].add(order_id)
            if executed_dt:
                cur = pkg["first_open_time"]
                if cur is None or executed_dt < cur:
                    pkg["first_open_time"] = executed_dt

            open_side = "SHORT" if side == "SELL" else "LONG"
            pkg["open_by_symbol"][symbol] += qty
            pkg["remaining_by_symbol"][symbol] += qty
            pkg["open_side_by_symbol"][symbol].add(open_side)

            open_cash_per_qty = to_float(fill.get("net_amount")) / qty if qty > eps else 0.0
            open_lots[(symbol, open_side)].append(
                {
                    "package_key": package_key,
                    "remaining_qty": qty,
                    "open_cash_per_qty": open_cash_per_qty,
                    "opened_at": fill.get("executed_at"),
                    "opened_dt": executed_dt,
                }
            )
            continue

        close_side = "LONG" if side == "SELL" else "SHORT"
        key = (symbol, close_side)
        lots = open_lots.get(key, [])
        if not lots:
            continue

        preferred_package = resolve_close_preferred_package(fill, metadata_by_order)
        remaining_qty = qty
        close_net = to_float(fill.get("net_amount"))
        pass_targets: List[Optional[str]] = [preferred_package] if preferred_package else [None]
        if preferred_package:
            pass_targets.append(None)

        for target in pass_targets:
            if remaining_qty <= eps:
                break
            for lot in lots:
                if remaining_qty <= eps:
                    break
                if lot["remaining_qty"] <= eps:
                    continue
                if target and lot["package_key"] != target:
                    continue

                take = min(remaining_qty, lot["remaining_qty"])
                if take <= eps:
                    continue

                pkg = packages.get(lot["package_key"])
                if pkg is None:
                    continue

                share = take / qty if qty > eps else 0.0
                close_cash = close_net * share
                pkg["realized_pnl"] += lot["open_cash_per_qty"] * take + close_cash
                pkg["closed_by_symbol"][symbol] += take
                pkg["remaining_by_symbol"][symbol] = max(0.0, pkg["remaining_by_symbol"][symbol] - take)
                if order_id:
                    pkg["exit_order_ids"].add(order_id)
                if executed_dt:
                    cur = pkg["last_close_time"]
                    if cur is None or executed_dt > cur:
                        pkg["last_close_time"] = executed_dt

                lot["remaining_qty"] -= take
                remaining_qty -= take

        open_lots[key] = [lot for lot in lots if lot["remaining_qty"] > eps]

    reconciled: List[Dict[str, Any]] = []
    for pkg in packages.values():
        open_by_symbol = {k: to_float(v) for k, v in pkg["open_by_symbol"].items() if to_float(v) > eps}
        if not open_by_symbol:
            continue
        remaining_by_symbol = {k: to_float(v) for k, v in pkg["remaining_by_symbol"].items()}
        if any(v > eps for v in remaining_by_symbol.values()):
            continue
        if pkg.get("last_close_time") is None:
            continue

        opened_at = str(pkg.get("opened_at") or "")
        if not opened_at and pkg.get("first_open_time") is not None:
            opened_at = pkg["first_open_time"].astimezone(timezone.utc).isoformat()
        closed_at = pkg["last_close_time"].astimezone(timezone.utc).isoformat()

        ticker = str(pkg.get("ticker") or "").strip()
        if not ticker or ticker == "UNKNOWN":
            first_symbol = next(iter(open_by_symbol.keys()))
            ticker = first_symbol.split()[0] if first_symbol else ""

        expiry = str(pkg.get("expiry") or "").strip()
        if not expiry:
            expiries = sorted({(_parse_option_symbol(sym)[1]) for sym in open_by_symbol.keys() if _parse_option_symbol(sym)[1]})
            if len(expiries) == 1:
                expiry = expiries[0]
            elif expiries:
                expiry = expiries[0]

        strategy = infer_strategy_for_package(pkg)
        quantity = min(open_by_symbol.values())
        if quantity <= eps:
            continue

        engine_trade_id = str(pkg.get("engine_trade_id") or "").strip()
        if not engine_trade_id:
            seed = "|".join(
                [
                    str(pkg.get("source_group") or ""),
                    opened_at,
                    ",".join(sorted(pkg.get("entry_order_ids", set()))),
                    ",".join(sorted(pkg.get("exit_order_ids", set()))),
                    ",".join(sorted(open_by_symbol.keys())),
                ]
            )
            digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8].upper()
            fallback_ticker = (ticker or "UNK").replace(" ", "")
            fallback_expiry = (expiry or "UNK").replace(" ", "")
            engine_trade_id = f"FALLBACK-{fallback_ticker}-{fallback_expiry}-{digest}"

        reconciled.append(
            {
                "engine_trade_id": engine_trade_id,
                "track": str(pkg.get("track") or "UNKNOWN"),
                "ticker": ticker,
                "strategy": strategy or "unknown",
                "expiry": expiry,
                "opened_at": opened_at,
                "closed_at": closed_at,
                "quantity": round(quantity, 8),
                "realized_pnl": round(to_float(pkg.get("realized_pnl")), 2),
                "entry_order_ids": sorted(pkg.get("entry_order_ids", set())),
                "exit_order_ids": sorted(pkg.get("exit_order_ids", set())),
                "source_group": str(pkg.get("source_group") or ""),
            }
        )

    reconciled.sort(key=lambda row: (str(row.get("closed_at") or ""), str(row.get("engine_trade_id") or "")))
    store.upsert_closed_trades(account_hash, reconciled)
    return store.fetch_closed_trades(account_hash)


def run_health_gate_from_reconciled_file(reconciled_jsonl: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if reconciled_jsonl.exists():
        with reconciled_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    rows.append(json.loads(text))
                except json.JSONDecodeError:
                    continue

    required_fields = [
        "engine_trade_id",
        "track",
        "ticker",
        "strategy",
        "expiry",
        "opened_at",
        "closed_at",
        "quantity",
        "realized_pnl",
        "entry_order_ids",
        "exit_order_ids",
    ]

    issues: List[str] = []
    seen: set[str] = set()
    for row in rows:
        trade_id = str(row.get("engine_trade_id") or "")
        if not trade_id:
            issues.append("missing_engine_trade_id")
            continue
        if trade_id in seen:
            issues.append(f"duplicate_engine_trade_id:{trade_id}")
        seen.add(trade_id)

        for field in required_fields:
            if row.get(field) in (None, "") and field not in {"expiry"}:
                issues.append(f"missing_{field}:{trade_id}")

        open_ts = parse_timestamp(row.get("opened_at"))
        close_ts = parse_timestamp(row.get("closed_at"))
        if open_ts and close_ts and close_ts < open_ts:
            issues.append(f"closed_before_opened:{trade_id}")

    return {
        "checked_at": local_now_iso(),
        "status": "PASS" if not issues else "FAIL",
        "rows_checked": len(rows),
        "issue_count": len(issues),
        "issues": issues,
        "reconciled_file": str(reconciled_jsonl),
        "note": "Health Gate evaluated from reconciled closed_trades JSONL.",
    }


# ================================================================
# FORMAT OUTPUT FOR SCAN ENGINE
# ================================================================

def format_for_engine(
    account_data: Dict[str, Any],
    orders: Sequence[Dict[str, Any]],
    closed_trades: Sequence[Dict[str, Any]],
    health_gate: Dict[str, Any],
):
    """Transform Schwab API data + reconciled trades into scan-engine format."""
    output: Dict[str, Any] = {
        "pull_timestamp": local_now_iso(),
        "source": "schwab_api",
        "account": {},
        "positions": [],
        "open_orders": [],
        "closed_trades": list(closed_trades),
        "health_gate": dict(health_gate),
    }

    if isinstance(account_data, list):
        account_data = account_data[0] if account_data else {}
    acct = account_data.get("securitiesAccount", account_data.get("account", account_data))
    balances = acct.get("currentBalances", {})
    output["account"] = {
        "account_type": acct.get("type", ""),
        "nlv": balances.get("liquidationValue", 0),
        "buying_power": balances.get("buyingPower", 0),
        "cash": balances.get("cashBalance", 0),
        "margin_equity": balances.get("equity", 0),
    }

    for pos in acct.get("positions", []):
        inst = pos.get("instrument", {})
        if str(inst.get("assetType") or "").upper() != "OPTION":
            continue

        position = {
            "ticker": inst.get("underlyingSymbol", inst.get("symbol", "")),
            "symbol": inst.get("symbol", ""),
            "asset_type": inst.get("assetType", ""),
            "quantity": pos.get("longQuantity", 0) - pos.get("shortQuantity", 0),
            "avg_cost": pos.get("averagePrice", 0),
            "market_value": pos.get("marketValue", 0),
            "day_pnl": pos.get("currentDayProfitLoss", 0),
            "total_pnl": pos.get("longOpenProfitLoss", pos.get("shortOpenProfitLoss", 0)),
            "total_pnl_pct": pos.get(
                "longOpenProfitLossPercentage",
                pos.get("shortOpenProfitLossPercentage", 0),
            ),
        }

        sym = inst.get("symbol", "")
        parsed_strike, parsed_expiry, parsed_type = _parse_option_symbol(sym)
        position.update(
            {
                "option_type": inst.get("putCall", parsed_type or ""),
                "strike": inst.get("strikePrice", 0) or parsed_strike,
                "expiry": inst.get("expirationDate", "") or parsed_expiry,
                "underlying": inst.get("underlyingSymbol", ""),
                "multiplier": pos.get("multiplier", 100),
            }
        )
        qty = position["quantity"]
        opt = position["option_type"]
        if qty > 0 and opt == "CALL":
            position["engine_type"] = "long_call"
        elif qty > 0 and opt == "PUT":
            position["engine_type"] = "long_put"
        elif qty < 0 and opt == "CALL":
            position["engine_type"] = "short_call"
        elif qty < 0 and opt == "PUT":
            position["engine_type"] = "short_put"

        output["positions"].append(position)

    for order in orders:
        status = str(order.get("status", ""))
        if status in ("WORKING", "PENDING_ACTIVATION", "QUEUED", "ACCEPTED") and order_is_option_only(order):
            legs = order.get("orderLegCollection", []) or []
            output["open_orders"].append(
                {
                    "order_id": order.get("orderId"),
                    "status": status,
                    "order_type": order.get("orderType"),
                    "price": order.get("price", order.get("stopPrice", 0)),
                    "entered": order.get("enteredTime"),
                    "legs": [
                        {
                            "symbol": leg.get("instrument", {}).get("symbol", ""),
                            "action": leg.get("instruction", ""),
                            "quantity": leg.get("quantity", 0),
                        }
                        for leg in legs
                    ],
                }
            )

    return output


# ================================================================
# WINDOW HELPERS
# ================================================================

def build_backfill_windows(
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int,
    overlap_days: int,
) -> List[Tuple[datetime, datetime]]:
    if chunk_days <= 0:
        raise ValueError("chunk_days must be > 0")
    if overlap_days < 0:
        raise ValueError("overlap_days must be >= 0")
    if overlap_days >= chunk_days:
        raise ValueError("overlap_days must be smaller than chunk_days")

    windows: List[Tuple[datetime, datetime]] = []
    cursor = start_dt
    chunk_delta = timedelta(days=chunk_days)
    overlap_delta = timedelta(days=overlap_days)

    while cursor < end_dt:
        window_end = min(cursor + chunk_delta, end_dt)
        windows.append((cursor, window_end))
        if window_end >= end_dt:
            break
        cursor = window_end - overlap_delta

    return windows


def choose_pull_windows(
    store: SchwabStateStore,
    account_hash: str,
    run_end: datetime,
    *,
    backfill_days: int,
    backfill_chunk_days: int,
    backfill_overlap_days: int,
    incremental_overlap_days: int,
    force_backfill: bool,
) -> Tuple[str, List[Tuple[datetime, datetime]], Optional[datetime]]:
    last_successful_ts = None if force_backfill else store.get_last_successful_ts(account_hash)

    if last_successful_ts is None:
        start_dt = run_end - timedelta(days=backfill_days)
        windows = build_backfill_windows(
            start_dt=start_dt,
            end_dt=run_end,
            chunk_days=backfill_chunk_days,
            overlap_days=backfill_overlap_days,
        )
        return "backfill", windows, None

    start_dt = last_successful_ts - timedelta(days=incremental_overlap_days)
    return "incremental", [(start_dt, run_end)], last_successful_ts


def normalize_account_tag(account_number: str, account_hash: str) -> str:
    if account_number and account_number != "unknown":
        return f"acct_{account_number[-4:]}"
    short_hash = (account_hash or "acct").replace("/", "_").replace("\\", "_")
    return f"acct_{short_hash[:8]}"


# ================================================================
# MAIN
# ================================================================

def parse_intended_legs(value: str) -> List[Dict[str, Any]]:
    text = (value or "").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [x for x in parsed if isinstance(x, dict)]
    return []


def main():
    raw_argv = sys.argv[1:]
    explicit_first_backfill_days = any(
        arg == "--first-backfill-days" or arg.startswith("--first-backfill-days=")
        for arg in raw_argv
    )

    parser = argparse.ArgumentParser(description="Pull Schwab account data for Anu Options Engine")
    parser.add_argument(
        "--token-path",
        default=DEFAULT_TOKEN_PATH,
        help=f"Path to Schwab token file/directory (default: {DEFAULT_TOKEN_PATH})",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON filename (default: schwab_positions_YYYY-MM-DD.json)",
    )
    parser.add_argument(
        "--first-backfill-days",
        type=int,
        default=DEFAULT_FIRST_BACKFILL_DAYS,
        help=(
            f"First-run backfill horizon in days (default: {DEFAULT_FIRST_BACKFILL_DAYS}). "
            "This only applies when no prior sync state exists unless --force-backfill is also set."
        ),
    )
    parser.add_argument(
        "--backfill-chunk-days",
        type=int,
        default=DEFAULT_BACKFILL_CHUNK_DAYS,
        help=f"Backfill chunk size in days (default: {DEFAULT_BACKFILL_CHUNK_DAYS})",
    )
    parser.add_argument(
        "--backfill-overlap-days",
        type=int,
        default=DEFAULT_BACKFILL_OVERLAP_DAYS,
        help=f"Backfill overlap in days (default: {DEFAULT_BACKFILL_OVERLAP_DAYS})",
    )
    parser.add_argument(
        "--incremental-overlap-days",
        type=int,
        default=DEFAULT_INCREMENTAL_OVERLAP_DAYS,
        help=f"Incremental overlap in days from last_successful_ts (default: {DEFAULT_INCREMENTAL_OVERLAP_DAYS})",
    )
    parser.add_argument(
        "--force-backfill",
        action="store_true",
        help="Force full backfill mode even if last_successful_ts exists",
    )
    parser.add_argument(
        "--state-dir",
        default=DEFAULT_STATE_DIR,
        help=f"Directory for local JSONL persistence (default: {DEFAULT_STATE_DIR})",
    )
    parser.add_argument(
        "--state-db",
        default=DEFAULT_STATE_DB,
        help=f"SQLite DB for rolling state + reconciled trades (default: {DEFAULT_STATE_DB})",
    )

    # Metadata write mode (engine writes this when opening a trade).
    parser.add_argument("--register-open-trade", action="store_true", help="Write open-trade metadata row and exit")
    parser.add_argument("--account-hash", default="*", help="Account hash for metadata rows (default: '*')")
    parser.add_argument("--engine-trade-id", default="", help="Engine trade id, e.g. FIRE-NVDA-20260420-01")
    parser.add_argument("--broker-order-id", default="", help="Broker order id used for matching fills")
    parser.add_argument("--track", default="", help="Trade track label, e.g. FIRE")
    parser.add_argument("--ticker", default="", help="Underlying ticker")
    parser.add_argument("--strategy", default="", help="Strategy slug, e.g. bull_call_debit")
    parser.add_argument("--expiry", default="", help="Option expiry YYYY-MM-DD")
    parser.add_argument("--opened-at", default="", help="Entry timestamp ISO8601")
    parser.add_argument("--intended-legs-json", default="[]", help="JSON list of intended legs")

    args = parser.parse_args()

    state_dir = Path(args.state_dir).expanduser().resolve()
    state_db = Path(args.state_db).expanduser().resolve()
    store = SchwabStateStore(db_path=state_db, state_dir=state_dir)

    if args.register_open_trade:
        if not args.engine_trade_id and not args.broker_order_id:
            print("❌ --register-open-trade requires --engine-trade-id or --broker-order-id")
            sys.exit(2)
        if not args.track or not args.ticker or not args.strategy:
            print("❌ --register-open-trade requires --track, --ticker, --strategy")
            sys.exit(2)

        engine_trade_id = args.engine_trade_id.strip()
        broker_order_id = args.broker_order_id.strip() or None
        if not engine_trade_id:
            engine_trade_id = f"BROKER-{broker_order_id}"

        opened_at = args.opened_at.strip() or local_now_iso()
        try:
            intended_legs = parse_intended_legs(args.intended_legs_json)
        except json.JSONDecodeError as exc:
            print(f"❌ intended_legs_json parse error: {exc}")
            sys.exit(2)

        store.upsert_open_trade_metadata(
            args.account_hash,
            engine_trade_id=engine_trade_id,
            broker_order_id=broker_order_id,
            track=args.track.strip(),
            ticker=args.ticker.strip(),
            strategy=args.strategy.strip(),
            expiry=args.expiry.strip() or None,
            intended_legs=intended_legs,
            opened_at=opened_at,
        )

        print(
            json.dumps(
                {
                    "status": "ok",
                    "mode": "register-open-trade",
                    "account_hash": args.account_hash,
                    "engine_trade_id": engine_trade_id,
                    "broker_order_id": broker_order_id,
                    "track": args.track,
                    "ticker": args.ticker,
                    "strategy": args.strategy,
                    "expiry": args.expiry,
                    "opened_at": opened_at,
                },
                indent=2,
            )
        )
        return

    print("=" * 60)
    print("Schwab Account Puller — Anu Options Engine")
    print("=" * 60)

    # 1. Find tokens
    token_path = resolve_token_path(args.token_path)
    print(f"\nToken path: {token_path}")
    tokens = find_tokens(str(token_path))

    if not tokens:
        print("\n❌ Could not find tokens. Make sure the path is correct.")
        print("   Expected: JSON file with access_token/refresh_token")
        sys.exit(1)

    # 1b. Load .env for client credentials if missing
    if not tokens.get("client_id") or not tokens.get("client_secret"):
        env_locations = [
            token_path.parent / ".env",
            token_path / ".env",
            project_root() / ".env",
            Path(".env"),
        ]
        for env_path in env_locations:
            if env_path.exists():
                print(f"Loading credentials from {env_path}...")
                try:
                    with env_path.open(encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if "=" in line and not line.startswith("#"):
                                key, _, val = line.partition("=")
                                kl = key.strip().lower()
                                vv = val.strip().strip('"').strip("'")
                                if kl in (
                                    "schwab_client_id",
                                    "client_id",
                                    "app_key",
                                    "api_key",
                                    "schwab_api_key",
                                    "schwab_app_key",
                                ):
                                    tokens["client_id"] = vv
                                elif kl in (
                                    "schwab_client_secret",
                                    "client_secret",
                                    "app_secret",
                                    "api_secret",
                                    "schwab_app_secret",
                                    "schwab_secret",
                                ):
                                    tokens["client_secret"] = vv
                    if tokens.get("client_id") and tokens.get("client_secret"):
                        print("  ✅ client_id + client_secret loaded from .env")
                        break
                except Exception as exc:
                    print(f"  ⚠️ Could not read {env_path}: {exc}")

    print("\nToken status:")
    print(
        f"  access_token:  {'✅ found (' + tokens['access_token'][:8] + '...)' if tokens.get('access_token') else '❌ missing'}"
    )
    print(
        f"  refresh_token: {'✅ found (' + tokens['refresh_token'][:8] + '...)' if tokens.get('refresh_token') else '❌ missing'}"
    )
    print(f"  client_id:     {'✅ found' if tokens.get('client_id') else '❌ missing'}")
    print(f"  client_secret: {'✅ found' if tokens.get('client_secret') else '❌ missing'}")

    # 2. Test access token, refresh if needed
    access_token = tokens.get("access_token")

    if access_token:
        print("\nTesting access token...")
        test = get_accounts(access_token)
        if isinstance(test, list):
            print("✅ Access token valid")
        elif isinstance(test, dict) and test.get("error") == "UNAUTHORIZED":
            print("Access token expired. Attempting refresh...")
            tokens = refresh_access_token(tokens)
            if tokens:
                access_token = tokens["access_token"]
                _save_refreshed_token(str(token_path), tokens)
            else:
                print("❌ Could not refresh token. You may need to re-authenticate.")
                sys.exit(1)
        elif isinstance(test, dict) and "error" in test:
            print(f"❌ API error: {test['error']}")
            sys.exit(1)
    else:
        tokens = refresh_access_token(tokens)
        if tokens:
            access_token = tokens["access_token"]
        else:
            print("❌ No access token and cannot refresh. Re-authenticate required.")
            sys.exit(1)

    # 3. Get accounts
    print("\nFetching accounts...")
    accounts = get_accounts(access_token)
    if not isinstance(accounts, list) or len(accounts) == 0:
        print(f"❌ No accounts found. Response: {json.dumps(accounts)[:200]}")
        sys.exit(1)

    account_numbers_payload = get_account_numbers(access_token)
    account_hash_map: Dict[str, str] = {}
    if isinstance(account_numbers_payload, list):
        for row in account_numbers_payload:
            if not isinstance(row, dict):
                continue
            acct_number = str(row.get("accountNumber") or "").strip()
            acct_hash = str(row.get("hashValue") or "").strip()
            if acct_number and acct_hash:
                account_hash_map[acct_number] = acct_hash
                # Also map stripped leading-zero variant if present.
                account_hash_map[acct_number.lstrip("0")] = acct_hash
    elif isinstance(account_numbers_payload, dict) and "error" in account_numbers_payload:
        print(f"⚠️ Could not fetch accountNumbers mapping: {str(account_numbers_payload.get('error', ''))[:160]}")

    print(f"Found {len(accounts)} account(s)")

    run_end = utc_now()
    all_data: List[Dict[str, Any]] = []
    for acct in accounts:
        if not isinstance(acct, dict):
            continue

        sec_acct = acct.get("securitiesAccount", {}) if isinstance(acct.get("securitiesAccount"), dict) else {}
        acct_num = str(acct.get("accountNumber", sec_acct.get("accountNumber", "unknown"))).strip()
        account_hash = str(
            acct.get("hashValue")
            or acct.get("accountHash")
            or sec_acct.get("hashValue")
            or sec_acct.get("accountHash")
            or account_hash_map.get(acct_num)
            or account_hash_map.get(acct_num.lstrip("0"))
            or ""
        ).strip()
        account_tag = normalize_account_tag(acct_num, account_hash)
        if not account_hash:
            print(f"\n--- Account: ...{acct_num[-4:]} ---")
            print("  ⚠️ Missing account hash; skipping account (cannot fetch account-level orders/transactions).")
            continue
        print(f"\n--- Account: ...{acct_num[-4:]} (hash: {account_hash[:8]}...) ---")

        # Keep existing nightly positions pull.
        print("  Fetching positions...")
        pos_data = get_positions(access_token, account_hash)
        if isinstance(pos_data, dict) and "error" in pos_data:
            print(f"  ⚠️ Positions error: {str(pos_data['error'])[:100]}")

        pull_mode, windows, last_successful_before = choose_pull_windows(
            store,
            account_hash,
            run_end,
            backfill_days=args.first_backfill_days,
            backfill_chunk_days=args.backfill_chunk_days,
            backfill_overlap_days=args.backfill_overlap_days,
            incremental_overlap_days=args.incremental_overlap_days,
            force_backfill=args.force_backfill,
        )
        if explicit_first_backfill_days and last_successful_before is not None and not args.force_backfill:
            print(
                "  Note: --first-backfill-days only affects the initial sync. Existing sync state was found, "
                "so this run stayed incremental. Use --force-backfill to rebuild the full history window."
            )
        print(f"  Activity pull mode: {pull_mode} ({len(windows)} window(s), oldest -> newest)")

        orders_saved = 0
        tx_saved = 0
        fills_saved = 0
        had_activity_errors = False
        latest_order_data: List[Dict[str, Any]] = []

        for idx, (win_start, win_end) in enumerate(windows, 1):
            print(f"  [{idx}/{len(windows)}] {to_schwab_time(win_start)} -> {to_schwab_time(win_end)}")

            order_batch = get_orders(access_token, account_hash, start_dt=win_start, end_dt=win_end)
            if isinstance(order_batch, dict) and "error" in order_batch:
                print(f"    ⚠️ Orders error: {str(order_batch['error'])[:100]}")
                had_activity_errors = True
                order_batch = []
            if not isinstance(order_batch, list):
                order_batch = []
            order_batch = [o for o in order_batch if isinstance(o, dict) and order_is_option_only(o)]
            orders_saved += store.store_raw_orders(account_hash, order_batch, account_tag)
            latest_order_data = order_batch

            txn_batch = get_transactions(access_token, account_hash, start_dt=win_start, end_dt=win_end)
            if isinstance(txn_batch, dict) and "error" in txn_batch:
                print(f"    ⚠️ Transactions error: {str(txn_batch['error'])[:100]}")
                had_activity_errors = True
                txn_batch = []
            if not isinstance(txn_batch, list):
                txn_batch = []
            txn_batch = filter_option_transactions([t for t in txn_batch if isinstance(t, dict)])
            tx_saved += store.store_raw_transactions(account_hash, txn_batch, account_tag)

            executions = extract_executions_from_transactions(txn_batch)
            fills_saved += store.store_raw_executions(account_hash, executions, account_tag)

        # Reconcile opening/closing fills into closed_trades (engine_trade_id first).
        reconciled_closed = reconcile_closed_trades_for_account(store, account_hash)
        reconciled_file = store.write_reconciled_jsonl(account_tag, reconciled_closed)

        # Run Health Gate off reconciled file every EOD.
        health_gate = run_health_gate_from_reconciled_file(reconciled_file)
        health_gate_report = store.write_health_gate_report(account_tag, health_gate)
        health_gate["health_gate_report"] = str(health_gate_report)

        if not had_activity_errors:
            store.set_last_successful_ts(account_hash, run_end)
            last_successful_after = run_end.astimezone(timezone.utc).isoformat()
        else:
            last_successful_after = (
                last_successful_before.astimezone(timezone.utc).isoformat()
                if last_successful_before
                else None
            )

        # Format for output engine snapshot.
        formatted = format_for_engine(pos_data, latest_order_data, reconciled_closed, health_gate)
        formatted["account"]["account_number_last4"] = acct_num[-4:]
        formatted["activity_pull"] = {
            "mode": pull_mode,
            "window_count": len(windows),
            "windows": [
                {
                    "start": to_schwab_time(start),
                    "end": to_schwab_time(end),
                }
                for start, end in windows
            ],
            "orders_saved": orders_saved,
            "transactions_saved": tx_saved,
            "fills_saved": fills_saved,
            "last_successful_ts_before": (
                last_successful_before.astimezone(timezone.utc).isoformat()
                if last_successful_before
                else None
            ),
            "last_successful_ts_after": last_successful_after,
            "activity_errors": had_activity_errors,
            "state_db": str(state_db),
            "reconciled_closed_trades_file": str(reconciled_file),
        }

        all_data.append(formatted)

    output_file = args.output or f"schwab_positions_{datetime.now().strftime('%Y-%m-%d')}.json"
    final_output = {
        "generated": local_now_iso(),
        "generator": "schwab_pull.py v2.1 — backfill + incremental idempotent reconciliation",
        "state_db": str(state_db),
        "state_dir": str(state_dir),
        "accounts": all_data,
    }

    with open(output_file, "w", encoding="utf-8") as handle:
        json.dump(final_output, handle, indent=2)

    print(f"\n{'=' * 60}")
    print(f"✅ Output written to: {output_file}")
    print(f"{'=' * 60}")

    for acct_data in all_data:
        acct_last4 = acct_data["account"].get("account_number_last4", "????")
        health = acct_data.get("health_gate", {})
        print(f"\nAccount ...{acct_last4}:")
        print(f"  NLV:          ${acct_data['account'].get('nlv', 0):,.2f}")
        print(f"  Buying power: ${acct_data['account'].get('buying_power', 0):,.2f}")
        print(f"  Positions:    {len(acct_data.get('positions', []))}")
        print(f"  Open orders:  {len(acct_data.get('open_orders', []))}")
        print(f"  Closed trades (reconciled): {len(acct_data.get('closed_trades', []))}")
        print(f"  Health Gate:  {health.get('status', 'UNKNOWN')} ({health.get('issue_count', 0)} issues)")

    print(f"\n📤 Upload '{output_file}' to Claude for scan engine ingestion.")
    print("   Snapshot now includes rolling orders/transactions/fills reconciliation")
    print("   and Health Gate computed from reconciled closed_trades JSONL.")


if __name__ == "__main__":
    main()
