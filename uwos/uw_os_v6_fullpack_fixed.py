#!/usr/bin/env python3
r"""
uw_os_v6_fullpack.py — FAST + FULL daily pack (daily_features + DP + OI carryover + screener + plan)

This fixes the key complaint: prior "v5_fast_fixed2" only zipped daily_features.csv (it was a minimal pack).
This version produces a *consistent* full pack every run:

Pack contents (always present, even if empty):
- daily_features.csv
- dp_anchors.csv
- oi_carryover_signatures.csv
- stock_screener.csv
- plan_skeleton.md

Performance design:
- Streams whale_trades_filtered.csv from disk (no "read whole file to bytes")
- Can ingest whale-YYYY-MM-DD.md summaries (Top 200 Yes-Prime table) when CSV is not used
- Filters OI/DP/Screener ingestion to the day's top N tickers (default 200–300) to avoid ingesting the entire market
- Uses safe SQLite inserts that respect MAX_VARIABLE_NUMBER to avoid "too many SQL variables"
- Computes persistence + DP + strike-psych using bounded queries (tickers list + last N sessions)

Folder layout expected (date folders directly under root):
  C:/uw_root/
    2025-12-22\
      whale_trades_filtered.csv
      whale-YYYY-MM-DD.md (optional; can be placed in root or date folder)
      chain-oi-changes-2025-12-22.zip
      dp-eod-report-2025-12-22.zip
      stock-screener-2025-12-22.zip
      hot-chains-2025-12-22.zip (optional; used only if whale file missing)

Run daily:
  python -u uw_os_v6_fullpack.py --root C:\uw_root --only-date 2025-12-23 --rebuild --verbose --top-tickers 200

Outputs:
  _derived/daily/YYYY-MM-DD\chatgpt_pack_YYYY-MM-DD.zip
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sqlite3
import zipfile
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # optional
except Exception:
    yf = None


DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OCC_RE = re.compile(r"^([A-Z]{1,10})(\d{6})([CP])(\d{8})$")  # SPY251222C00685000


# ----------------------------
# Helpers
# ----------------------------

def parse_occ_symbol(sym: str) -> Optional[Tuple[str, str, str, float]]:
    """Parse OCC symbol like SPY251222C00685000 -> (SPY, 2025-12-22, call, 685.0)."""
    if sym is None:
        return None
    s = str(sym).strip().upper()
    m = OCC_RE.match(s)
    if not m:
        return None
    root, yymmdd, cp, strike8 = m.groups()
    yy = int(yymmdd[:2]); mm = int(yymmdd[2:4]); dd = int(yymmdd[4:6])
    exp = date(2000 + yy, mm, dd).isoformat()
    opt_type = "call" if cp == "C" else "put"
    strike = int(strike8) / 1000.0
    return root, exp, opt_type, float(strike)


def safe_float(s: pd.Series) -> pd.Series:
    x = s.astype(str)
    x = x.str.replace(r"[\$,]", "", regex=True)
    x = x.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    return pd.to_numeric(x, errors="coerce")


def infer_exec(side: str) -> str:
    s = str(side).strip().lower()
    if "ask" in s:
        return "ask"
    if "bid" in s:
        return "bid"
    if "mid" in s:
        return "mid"
    if "cross" in s:
        return "cross"
    return "unknown"


def infer_action_weight(exec_loc: str) -> Tuple[str, float]:
    if exec_loc == "ask":
        return "buy", 1.0
    if exec_loc == "bid":
        return "sell", 1.0
    if exec_loc == "mid":
        return "unknown", 0.25
    if exec_loc == "cross":
        return "unknown", 0.0
    return "unknown", 0.0


def dir_sign(option_type: str, action: str) -> int:
    # Put Buy → Bearish; Put Sell → Bullish; Call Buy → Bullish; Call Sell → Bearish
    if action == "buy":
        return 1 if option_type == "call" else -1
    if action == "sell":
        return -1 if option_type == "call" else 1
    return 0


def time_bucket_from_ts(ts: pd.Series) -> pd.Series:
    """Bucket by U.S. market time (ET) where possible."""
    # Parse; treat as UTC if tz present or if strings are ISO-ish
    t = pd.to_datetime(ts, errors="coerce", utc=True)
    # Convert to ET when tz-aware
    try:
        t = t.dt.tz_convert("America/New_York")
    except Exception:
        pass
    mins = (t.dt.hour * 60) + t.dt.minute.fillna(0)
    out = pd.Series(["unknown"] * len(ts), index=ts.index)
    out[(mins >= 570) & (mins <= 630)] = "open"   # 9:30–10:30 ET
    out[(mins > 630) & (mins < 900)] = "mid"
    out[mins >= 900] = "eod"                      # 15:00+ ET
    return out


# SQLite variable limit safe insert
_SQLITE_MAX_VARS_CACHE: Dict[int, int] = {}

def sqlite_max_vars(conn: sqlite3.Connection) -> int:
    key = id(conn)
    if key in _SQLITE_MAX_VARS_CACHE:
        return _SQLITE_MAX_VARS_CACHE[key]
    max_vars = 999
    try:
        opts = conn.execute("PRAGMA compile_options;").fetchall()
        for (opt,) in opts:
            if isinstance(opt, str) and opt.startswith("MAX_VARIABLE_NUMBER="):
                max_vars = int(opt.split("=", 1)[1])
                break
    except Exception:
        pass
    _SQLITE_MAX_VARS_CACHE[key] = max_vars
    return max_vars


def safe_to_sql(df: pd.DataFrame, table: str, conn: sqlite3.Connection, *, chunksize: int = 5000) -> None:
    if df is None or df.empty:
        return
    ncols = max(1, len(df.columns))
    max_vars = sqlite_max_vars(conn)
    safe_rows = max(1, min(int(chunksize), max(1, (max_vars // ncols) - 1)))
    df.to_sql(table, conn, if_exists="append", index=False, method="multi", chunksize=safe_rows)


def write_csv_always(path: Path, df: Optional[pd.DataFrame], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    else:
        # Ensure columns exist
        for c in columns:
            if c not in df.columns:
                df[c] = np.nan
        df[columns].to_csv(path, index=False)


# ----------------------------
# Config
# ----------------------------

@dataclass
class Config:
    lookback_sessions: int = 10
    persistence_windows: List[int] = None
    dp_cluster_pct: float = 0.003
    magnet_pct: float = 0.01
    oi_dte_min: int = 7
    oi_dte_max: int = 45
    top_signatures_n: int = 80
    daily_signal_threshold_ratio: float = 0.10
    price_source: str = "export"  # export|yfinance|none|auto
    top_tickers: int = 200
    flow_chunksize: int = 200_000
    sql_chunksize: int = 5000

    def __post_init__(self):
        if self.persistence_windows is None:
            self.persistence_windows = [5, 10]


def load_or_create_config(out_dir: Path) -> Config:
    p = out_dir / "uw_os_config.json"
    if p.exists():
        return Config(**json.loads(p.read_text(encoding="utf-8")))
    cfg = Config()
    out_dir.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    return cfg


# ----------------------------
# DB schema
# ----------------------------

def init_db(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA temp_store=MEMORY;")

    cur.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS flow_prints (
        trade_date TEXT,
        ticker TEXT,
        expiry TEXT,
        strike REAL,
        option_type TEXT,
        contract_signature TEXT,
        print_type TEXT,
        execution_location TEXT,
        premium REAL,
        contracts REAL,
        trade_time TEXT,
        time_bucket TEXT,
        underlying_price REAL,
        weight REAL,
        dir_sign INTEGER,
        weighted_notional REAL,
        abs_premium_w REAL,
        source_file TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_flow_date ON flow_prints(trade_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_flow_td_ticker ON flow_prints(trade_date, ticker);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_flow_sig ON flow_prints(contract_signature);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS oi_snapshots (
        trade_date TEXT,
        ticker TEXT,
        expiry TEXT,
        strike REAL,
        option_type TEXT,
        contract_signature TEXT,
        open_interest REAL,
        source_file TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_oi_date ON oi_snapshots(trade_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_oi_sig ON oi_snapshots(contract_signature);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS dp_prints (
        trade_date TEXT,
        ticker TEXT,
        price REAL,
        shares REAL,
        notional REAL,
        source_file TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dp_date ON dp_prints(trade_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_dp_td_ticker ON dp_prints(trade_date, ticker);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS screener_daily (
        trade_date TEXT,
        ticker TEXT,
        call_volume REAL,
        put_volume REAL,
        call_premium REAL,
        put_premium REAL,
        total_open_interest REAL,
        sector TEXT,
        source_file TEXT
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_scr_date ON screener_daily(trade_date);")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS ticker_day (
        trade_date TEXT,
        ticker TEXT,
        net REAL,
        abs_total REAL,
        eod_abs REAL,
        daily_sign INTEGER,
        eod_ratio REAL
    );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_td_date ON ticker_day(trade_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_td_td_ticker ON ticker_day(trade_date, ticker);")

    conn.commit()


def delete_day(conn: sqlite3.Connection, d: date) -> None:
    ds = d.isoformat()
    conn.execute("DELETE FROM flow_prints WHERE trade_date=?", (ds,))
    conn.execute("DELETE FROM oi_snapshots WHERE trade_date=?", (ds,))
    conn.execute("DELETE FROM dp_prints WHERE trade_date=?", (ds,))
    conn.execute("DELETE FROM screener_daily WHERE trade_date=?", (ds,))
    conn.execute("DELETE FROM ticker_day WHERE trade_date=?", (ds,))
    conn.commit()


# ----------------------------
# Discovery
# ----------------------------

def find_date_folders(root: Path, only_date: Optional[str] = None) -> List[Tuple[date, Path]]:
    out: List[Tuple[date, Path]] = []
    for p in root.iterdir():
        if p.is_dir() and DATE_DIR_RE.match(p.name):
            if only_date and p.name != only_date:
                continue
            out.append((datetime.strptime(p.name, "%Y-%m-%d").date(), p))
    out.sort(key=lambda x: x[0])
    return out


def find_file(folder: Path, substr: str, suffix: Optional[str] = None) -> Optional[Path]:
    substr = substr.lower()
    for p in folder.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if substr in name and (suffix is None or name.endswith(suffix)):
            return p
    return None


def _is_md_alignment_line(line: str) -> bool:
    trimmed = line.strip().strip("|")
    if not trimmed:
        return False
    cleaned = trimmed.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
    return cleaned == ""


def read_markdown_table(path: Path, heading_substr: str) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if heading_substr in line:
            start = i
            break

    if start is None:
        return pd.DataFrame()

    table_lines = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            if table_lines:
                break
            continue
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines:
            break

    if not table_lines:
        return pd.DataFrame()

    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[1:]:
        if _is_md_alignment_line(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        rows.append(cells[:len(header)])

    df = pd.DataFrame(rows, columns=header)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


# ----------------------------
# Ingest
# ----------------------------

def parse_whale_chunk(df: pd.DataFrame, d: date, source: str) -> pd.DataFrame:
    need = {"underlying_symbol", "expiry", "strike", "option_type", "side", "premium", "size"}
    if not need.issubset(set(df.columns)):
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["trade_date"] = d.isoformat()
    out["ticker"] = df["underlying_symbol"].astype(str).str.upper().str.strip()
    out["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date.astype("string")
    out["strike"] = safe_float(df["strike"])
    out["option_type"] = df["option_type"].astype(str).str.lower().map(lambda x: "call" if x.startswith("c") else ("put" if x.startswith("p") else x))
    out["execution_location"] = df["side"].map(infer_exec)
    out["premium"] = safe_float(df["premium"])
    out["contracts"] = safe_float(df["size"])

    ts = df.get("executed_at", df.get("tape_time", pd.Series([pd.NaT] * len(df))))
    out["trade_time"] = pd.to_datetime(ts, errors="coerce", utc=True).astype("string")
    out["time_bucket"] = time_bucket_from_ts(ts)

    out["underlying_price"] = safe_float(df.get("underlying_price", pd.Series([np.nan] * len(df))))

    out = out.dropna(subset=["ticker", "expiry", "strike", "option_type", "premium"])

    out["contract_signature"] = (
        out["ticker"].astype(str) + "-" + out["expiry"].astype(str) + "-" +
        out["strike"].astype(float).map(lambda x: f"{x:g}") + "-" + out["option_type"].astype(str)
    )
    out["print_type"] = "whale"
    out["source_file"] = source

    aw = out["execution_location"].map(lambda x: infer_action_weight(str(x)))
    action = aw.map(lambda x: x[0])
    out["weight"] = aw.map(lambda x: x[1]).astype(float)
    out["dir_sign"] = [dir_sign(ot, ac) for ot, ac in zip(out["option_type"].astype(str), action.astype(str))]
    out["weighted_notional"] = (out["premium"].fillna(0.0) * out["weight"] * out["dir_sign"]).astype(float)
    out["abs_premium_w"] = (out["premium"].abs().fillna(0.0) * out["weight"]).astype(float)

    return out[[
        "trade_date","ticker","expiry","strike","option_type","contract_signature","print_type",
        "execution_location","premium","contracts","trade_time","time_bucket","underlying_price",
        "weight","dir_sign","weighted_notional","abs_premium_w","source_file"
    ]]


def ingest_whale_file(conn: sqlite3.Connection, path: Path, d: date, cfg: Config, verbose: bool) -> int:
    kept_total = 0
    if verbose:
        print(f"  [flow] streaming {path.name} ...")
    for i, chunk in enumerate(pd.read_csv(path, low_memory=False, chunksize=cfg.flow_chunksize)):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        out = parse_whale_chunk(chunk, d, path.name)
        kept = len(out)
        kept_total += kept
        if kept:
            safe_to_sql(out, "flow_prints", conn, chunksize=cfg.sql_chunksize)
        if verbose:
            print(f"    chunk {i+1} rows={len(chunk):,} kept={kept:,}")
    conn.commit()
    return kept_total


def ingest_whale_md_file(conn: sqlite3.Connection, path: Path, d: date, cfg: Config, verbose: bool) -> int:
    if verbose:
        print(f"  [flow] reading {path.name} ...")
    df = read_markdown_table(path, "Top 200 Yes-Prime Trades by Premium")
    if df.empty:
        return 0
    out = parse_whale_chunk(df, d, path.name)
    kept = len(out)
    if kept:
        safe_to_sql(out, "flow_prints", conn, chunksize=cfg.sql_chunksize)
    conn.commit()
    return kept


def ingest_hot_chains_zip(conn: sqlite3.Connection, path: Path, d: date, cfg: Config, verbose: bool) -> int:
    # Only used if whale file missing.
    kept_total = 0
    if verbose:
        print(f"  [flow-fallback] reading {path.name} ...")
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv") or name.endswith("/"):
                continue
            with zf.open(name) as fh:
                df = pd.read_csv(fh, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            if "option_symbol" not in df.columns:
                continue

            parsed = df["option_symbol"].map(parse_occ_symbol)
            ok = parsed.notna()
            if ok.sum() == 0:
                continue
            tmp = pd.DataFrame(parsed[ok].tolist(), columns=["ticker","expiry","option_type","strike"], index=df.index[ok])

            out = pd.DataFrame(index=tmp.index)
            out["trade_date"] = d.isoformat()
            out["ticker"] = tmp["ticker"].astype(str)
            out["expiry"] = tmp["expiry"].astype("string")
            out["strike"] = pd.to_numeric(tmp["strike"], errors="coerce")
            out["option_type"] = tmp["option_type"].astype(str)
            out["premium"] = safe_float(df.loc[tmp.index, "premium"]) if "premium" in df.columns else np.nan

            # exec inference from side volume if present
            if "ask_side_volume" in df.columns and "bid_side_volume" in df.columns:
                askv = safe_float(df.loc[tmp.index, "ask_side_volume"]).fillna(0.0)
                bidv = safe_float(df.loc[tmp.index, "bid_side_volume"]).fillna(0.0)
                out["execution_location"] = np.where(askv > bidv, "ask", np.where(bidv > askv, "bid", "mid"))
            else:
                out["execution_location"] = "unknown"
            out["contracts"] = safe_float(df.loc[tmp.index, "volume"]) if "volume" in df.columns else np.nan

            ts = df.loc[tmp.index, "tape_time"] if "tape_time" in df.columns else df.loc[tmp.index, "date"] if "date" in df.columns else pd.Series([pd.NaT]*len(tmp), index=tmp.index)
            out["trade_time"] = pd.to_datetime(ts, errors="coerce", utc=True).astype("string")
            out["time_bucket"] = time_bucket_from_ts(ts)
            out["underlying_price"] = safe_float(df.loc[tmp.index, "stock_price"]) if "stock_price" in df.columns else np.nan

            out = out.dropna(subset=["ticker","expiry","strike","option_type","premium"])

            out["contract_signature"] = (
                out["ticker"].astype(str) + "-" + out["expiry"].astype(str) + "-" +
                out["strike"].astype(float).map(lambda x: f"{x:g}") + "-" + out["option_type"].astype(str)
            )
            out["print_type"] = "hot_chain"
            out["source_file"] = path.name

            aw = pd.Series(out["execution_location"]).map(lambda x: infer_action_weight(str(x)))
            action = aw.map(lambda x: x[0])
            out["weight"] = aw.map(lambda x: x[1]).astype(float)
            out["dir_sign"] = [dir_sign(ot, ac) for ot, ac in zip(out["option_type"].astype(str), action.astype(str))]
            out["weighted_notional"] = (out["premium"].fillna(0.0) * out["weight"] * out["dir_sign"]).astype(float)
            out["abs_premium_w"] = (out["premium"].abs().fillna(0.0) * out["weight"]).astype(float)

            out = out[[
                "trade_date","ticker","expiry","strike","option_type","contract_signature","print_type",
                "execution_location","premium","contracts","trade_time","time_bucket","underlying_price",
                "weight","dir_sign","weighted_notional","abs_premium_w","source_file"
            ]]
            kept = len(out)
            kept_total += kept
            if kept:
                safe_to_sql(out, "flow_prints", conn, chunksize=cfg.sql_chunksize)
    conn.commit()
    return kept_total


def compute_ticker_day(conn: sqlite3.Connection, d: date, cfg: Config) -> None:
    ds = d.isoformat()
    conn.execute("DELETE FROM ticker_day WHERE trade_date=?", (ds,))
    conn.execute("""
        INSERT INTO ticker_day (trade_date, ticker, net, abs_total, eod_abs, daily_sign, eod_ratio)
        SELECT
            trade_date,
            ticker,
            SUM(weighted_notional) AS net,
            SUM(abs_premium_w)     AS abs_total,
            SUM(CASE WHEN time_bucket='eod' THEN abs_premium_w ELSE 0 END) AS eod_abs,
            0 AS daily_sign,
            0.0 AS eod_ratio
        FROM flow_prints
        WHERE trade_date=?
        GROUP BY trade_date, ticker
    """, (ds,))
    # daily_sign + eod_ratio
    rows = conn.execute("SELECT ticker, net, abs_total, eod_abs FROM ticker_day WHERE trade_date=?", (ds,)).fetchall()
    updates = []
    for t, net, abs_total, eod_abs in rows:
        if abs_total is None or abs_total == 0:
            sign = 0
            ratio = 0.0
        else:
            sign = int(np.sign(net)) if abs(net) >= cfg.daily_signal_threshold_ratio * abs_total else 0
            ratio = float(eod_abs) / float(abs_total)
        updates.append((sign, ratio, t, ds))
    conn.executemany("UPDATE ticker_day SET daily_sign=?, eod_ratio=? WHERE ticker=? AND trade_date=?", updates)
    conn.commit()


def top_tickers_for_day(conn: sqlite3.Connection, d: date, n: int) -> List[str]:
    ds = d.isoformat()
    df = pd.read_sql_query(
        "SELECT ticker FROM ticker_day WHERE trade_date=? ORDER BY abs_total DESC LIMIT ?",
        conn, params=(ds, int(n))
    )
    return df["ticker"].astype(str).tolist() if not df.empty else []


def ingest_oi_zip_filtered(conn: sqlite3.Connection, path: Path, d: date, tickers: Sequence[str], cfg: Config, verbose: bool) -> int:
    if not tickers:
        return 0
    tickset = set(tickers)
    kept_total = 0
    if verbose:
        print(f"  [oi] reading {path.name} (filtered to {len(tickset)} tickers) ...")

    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv") or name.endswith("/"):
                continue
            with zf.open(name) as fh:
                for chunk in pd.read_csv(fh, low_memory=False, chunksize=cfg.flow_chunksize):
                    chunk.columns = [str(c).strip() for c in chunk.columns]
                    need = {"option_symbol","underlying_symbol","curr_oi"}
                    if not need.issubset(set(chunk.columns)):
                        continue
                    # filter by underlying_symbol first (fast)
                    sym = chunk["underlying_symbol"].astype(str).str.upper().str.strip()
                    m = sym.isin(tickset)
                    if m.sum() == 0:
                        continue
                    sub = chunk.loc[m].copy()

                    parsed = sub["option_symbol"].map(parse_occ_symbol)
                    ok = parsed.notna()
                    if ok.sum() == 0:
                        continue
                    tmp = pd.DataFrame(parsed[ok].tolist(), columns=["t2","expiry","option_type","strike2"], index=sub.index[ok])

                    out = pd.DataFrame(index=tmp.index)
                    out["trade_date"] = d.isoformat()
                    out["ticker"] = sub.loc[tmp.index, "underlying_symbol"].astype(str).str.upper().str.strip()
                    out["expiry"] = tmp["expiry"].astype("string")
                    out["strike"] = pd.to_numeric(sub.loc[tmp.index, "strike"], errors="coerce").fillna(pd.to_numeric(tmp["strike2"], errors="coerce"))
                    out["option_type"] = tmp["option_type"].astype(str)
                    out["open_interest"] = safe_float(sub.loc[tmp.index, "curr_oi"])
                    out["source_file"] = path.name

                    out = out.dropna(subset=["ticker","expiry","strike","option_type","open_interest"])
                    out["contract_signature"] = (
                        out["ticker"].astype(str) + "-" + out["expiry"].astype(str) + "-" +
                        out["strike"].astype(float).map(lambda x: f"{x:g}") + "-" + out["option_type"].astype(str)
                    )
                    out = out[["trade_date","ticker","expiry","strike","option_type","contract_signature","open_interest","source_file"]]

                    kept = len(out)
                    kept_total += kept
                    if kept:
                        safe_to_sql(out, "oi_snapshots", conn, chunksize=cfg.sql_chunksize)
    conn.commit()
    return kept_total


def ingest_dp_zip_filtered(conn: sqlite3.Connection, path: Path, d: date, tickers: Sequence[str], cfg: Config, verbose: bool) -> int:
    if not tickers:
        return 0
    tickset = set(tickers)
    kept_total = 0
    if verbose:
        print(f"  [dp] reading {path.name} (filtered to {len(tickset)} tickers) ...")
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv") or name.endswith("/"):
                continue
            with zf.open(name) as fh:
                df = pd.read_csv(fh, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            if not {"ticker","price"}.issubset(set(df.columns)):
                continue
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            df = df[df["ticker"].isin(tickset)]
            if df.empty:
                continue

            out = pd.DataFrame(index=df.index)
            out["trade_date"] = d.isoformat()
            out["ticker"] = df["ticker"]
            out["price"] = safe_float(df["price"])
            out["shares"] = safe_float(df.get("size", pd.Series([np.nan]*len(df))))
            out["notional"] = safe_float(df.get("premium", pd.Series([np.nan]*len(df))))
            out["notional"] = out["notional"].fillna(out["price"] * out["shares"])
            out["source_file"] = path.name
            out = out.dropna(subset=["ticker","price"])
            out = out[["trade_date","ticker","price","shares","notional","source_file"]]

            kept = len(out)
            kept_total += kept
            if kept:
                safe_to_sql(out, "dp_prints", conn, chunksize=cfg.sql_chunksize)
    conn.commit()
    return kept_total


def ingest_screener_zip_filtered(conn: sqlite3.Connection, path: Path, d: date, tickers: Sequence[str], cfg: Config, verbose: bool) -> int:
    if not tickers:
        return 0
    tickset = set(tickers)
    kept_total = 0
    if verbose:
        print(f"  [screener] reading {path.name} (filtered to {len(tickset)} tickers) ...")
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv") or name.endswith("/"):
                continue
            with zf.open(name) as fh:
                df = pd.read_csv(fh, low_memory=False)
            df.columns = [str(c).strip() for c in df.columns]
            need = {"ticker","call_volume","put_volume","call_premium","put_premium"}
            if not need.issubset(set(df.columns)):
                continue
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            df = df[df["ticker"].isin(tickset)]
            if df.empty:
                continue

            out = pd.DataFrame(index=df.index)
            out["trade_date"] = d.isoformat()
            out["ticker"] = df["ticker"]
            out["call_volume"] = safe_float(df["call_volume"])
            out["put_volume"] = safe_float(df["put_volume"])
            out["call_premium"] = safe_float(df["call_premium"])
            out["put_premium"] = safe_float(df["put_premium"])
            out["total_open_interest"] = safe_float(df.get("total_open_interest", pd.Series([np.nan]*len(df))))
            out["sector"] = df.get("sector", pd.Series([""]*len(df))).astype(str)
            out["source_file"] = path.name
            out = out.dropna(subset=["ticker"])
            out = out[["trade_date","ticker","call_volume","put_volume","call_premium","put_premium","total_open_interest","sector","source_file"]]

            kept = len(out)
            kept_total += kept
            if kept:
                safe_to_sql(out, "screener_daily", conn, chunksize=cfg.sql_chunksize)
    conn.commit()
    return kept_total


# ----------------------------
# Analytics for pack
# ----------------------------

def available_dates(conn: sqlite3.Connection) -> List[date]:
    df = pd.read_sql_query("SELECT DISTINCT trade_date FROM ticker_day ORDER BY trade_date", conn)
    if df.empty:
        return []
    s = df["trade_date"].astype(str).str.strip().str.slice(0, 10)
    dt = pd.to_datetime(s, errors="coerce").dt.date
    return sorted({x for x in dt.tolist() if x is not None and str(x) != "NaT"})


def spot_from_exports(conn: sqlite3.Connection, d: date, tickers: Sequence[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker","spot"])
    ds = d.isoformat()
    t_place = ",".join(["?"] * len(tickers))
    df = pd.read_sql_query(
        f"SELECT ticker, underlying_price FROM flow_prints WHERE trade_date=? AND ticker IN ({t_place}) AND underlying_price IS NOT NULL",
        conn, params=[ds] + list(tickers)
    )
    if df.empty:
        return pd.DataFrame(columns=["ticker","spot"])
    df["underlying_price"] = pd.to_numeric(df["underlying_price"], errors="coerce")
    df = df.dropna(subset=["underlying_price"])
    if df.empty:
        return pd.DataFrame(columns=["ticker","spot"])
    return df.groupby("ticker")["underlying_price"].median().reset_index().rename(columns={"underlying_price":"spot"})


def spot_from_yfinance(tickers: Sequence[str], d: date) -> pd.DataFrame:
    if yf is None or not tickers:
        return pd.DataFrame(columns=["ticker","spot"])
    start = d.isoformat()
    end = (d + timedelta(days=1)).isoformat()
    try:
        data = yf.download(list(tickers), start=start, end=end, interval="1d", progress=False, threads=True)
        rows = []
        if isinstance(data.columns, pd.MultiIndex):
            for t in tickers:
                try:
                    c = data["Close"][t].dropna()
                    if not c.empty:
                        rows.append((t, float(c.iloc[-1])))
                except Exception:
                    continue
        else:
            c = data["Close"].dropna()
            if not c.empty and tickers:
                rows.append((list(tickers)[0], float(c.iloc[-1])))
        return pd.DataFrame(rows, columns=["ticker","spot"])
    except Exception:
        return pd.DataFrame(columns=["ticker","spot"])


def persistence_window(conn: sqlite3.Connection, lookback: List[date], window: int, tickers: Sequence[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    last_dates = [x.isoformat() for x in lookback[-window:]]
    denom = max(len(last_dates), 1)
    d_place = ",".join(["?"] * len(last_dates))
    t_place = ",".join(["?"] * len(tickers))
    params = last_dates + list(tickers)

    day = pd.read_sql_query(
        f"SELECT trade_date, ticker, daily_sign, eod_ratio FROM ticker_day WHERE trade_date IN ({d_place}) AND ticker IN ({t_place})",
        conn, params=params
    )
    if day.empty:
        return pd.DataFrame()

    day_stats = day.groupby("ticker", as_index=False).agg(
        pos_days=("daily_sign", lambda x: int((x == 1).sum())),
        neg_days=("daily_sign", lambda x: int((x == -1).sum())),
        eod_ratio=("eod_ratio", "mean"),
    )
    day_stats["active_days"] = day_stats["pos_days"] + day_stats["neg_days"]
    day_stats["dominant_sign"] = np.where(day_stats["pos_days"] >= day_stats["neg_days"], 1, -1)
    day_stats.loc[day_stats["active_days"] == 0, "dominant_sign"] = 0
    day_stats["consistency"] = np.where(
        day_stats["active_days"] > 0,
        np.maximum(day_stats["pos_days"], day_stats["neg_days"]) / day_stats["active_days"],
        0.0
    )
    day_stats["active_ratio"] = day_stats["active_days"] / float(denom)

    # signature aggregates (bounded)
    sig = pd.read_sql_query(
        f"""
        SELECT ticker, contract_signature, SUM(abs_premium_w) AS sig_abs, COUNT(DISTINCT trade_date) AS sig_days
        FROM flow_prints
        WHERE trade_date IN ({d_place}) AND ticker IN ({t_place})
        GROUP BY ticker, contract_signature
        """,
        conn, params=params
    )
    if sig.empty:
        return pd.DataFrame()

    sig["sig_abs"] = pd.to_numeric(sig["sig_abs"], errors="coerce").fillna(0.0)

    total_abs = sig.groupby("ticker", as_index=False)["sig_abs"].sum().rename(columns={"sig_abs":"total_abs"})
    top3 = (
        sig.sort_values(["ticker","sig_abs"], ascending=[True, False])
           .groupby("ticker", as_index=False).head(3)
           .groupby("ticker", as_index=False)["sig_abs"].sum()
           .rename(columns={"sig_abs":"top3_abs"})
    )
    conc = total_abs.merge(top3, on="ticker", how="left").fillna({"top3_abs":0.0})
    conc["concentration"] = np.where(conc["total_abs"] > 0, conc["top3_abs"] / conc["total_abs"], 0.0)

    rep = sig.groupby("ticker", as_index=False).agg(
        sig_total=("contract_signature","count"),
        sig_repeated=("sig_days", lambda x: int((x >= 2).sum()))
    )
    rep["repeat_ratio"] = np.where(rep["sig_total"] > 0, rep["sig_repeated"] / rep["sig_total"], 0.0)

    feats = day_stats.merge(conc[["ticker","concentration"]], on="ticker", how="left").merge(rep[["ticker","repeat_ratio"]], on="ticker", how="left")
    feats = feats.fillna({"concentration":0.0,"repeat_ratio":0.0,"eod_ratio":0.0})

    score = 100.0 * (
        0.45*(feats["active_ratio"]*feats["consistency"]) +
        0.25*feats["concentration"] +
        0.15*feats["repeat_ratio"] +
        0.15*feats["eod_ratio"]
    )
    score = score.clip(0.0, 100.0)
    label = np.where(score >= 70, "Campaign", np.where(score >= 50, "Maybe", "Noise"))

    out = pd.DataFrame({
        "ticker": feats["ticker"],
        f"persistence_{window}d_score": score.astype(float),
        f"persistence_{window}d_label": label,
        f"persistence_{window}d_dominant_sign": feats["dominant_sign"].astype(int),
        f"persistence_{window}d_active_days": feats["active_days"].astype(int),
        f"persistence_{window}d_sessions": int(denom),
    })
    return out


def oi_carryover(conn: sqlite3.Connection, prev_d: date, cur_d: date, tickers: Sequence[str], top_n: int) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()
    prev = prev_d.isoformat()
    cur = cur_d.isoformat()
    t_place = ",".join(["?"] * len(tickers))

    f = pd.read_sql_query(
        f"""
        SELECT ticker, contract_signature, SUM(abs_premium_w) AS prev_abs, SUM(contracts) AS prev_contracts
        FROM flow_prints
        WHERE trade_date=? AND ticker IN ({t_place})
        GROUP BY ticker, contract_signature
        """,
        conn, params=[prev] + list(tickers)
    )
    if f.empty:
        return pd.DataFrame()
    f["prev_abs"] = pd.to_numeric(f["prev_abs"], errors="coerce").fillna(0.0)
    f["prev_contracts"] = pd.to_numeric(f["prev_contracts"], errors="coerce").fillna(1.0)
    f = f.sort_values(["ticker","prev_abs"], ascending=[True, False]).groupby("ticker").head(top_n).reset_index(drop=True)

    oi_prev = pd.read_sql_query("SELECT contract_signature, open_interest FROM oi_snapshots WHERE trade_date=?", conn, params=(prev,))
    oi_cur  = pd.read_sql_query("SELECT contract_signature, open_interest FROM oi_snapshots WHERE trade_date=?", conn, params=(cur,))
    if oi_prev.empty or oi_cur.empty:
        f["oi_prev"] = np.nan; f["oi_cur"] = np.nan; f["oi_delta"] = np.nan; f["carryover_ratio"] = np.nan
        return f

    oi_prev["open_interest"] = pd.to_numeric(oi_prev["open_interest"], errors="coerce")
    oi_cur["open_interest"] = pd.to_numeric(oi_cur["open_interest"], errors="coerce")

    m = f.merge(oi_prev.rename(columns={"open_interest":"oi_prev"}), on="contract_signature", how="left").merge(
        oi_cur.rename(columns={"open_interest":"oi_cur"}), on="contract_signature", how="left"
    )
    m["oi_delta"] = m["oi_cur"] - m["oi_prev"]
    denom = m["prev_contracts"].replace(0, np.nan)
    m["carryover_ratio"] = (m["oi_delta"] / denom).clip(-1, 1)
    return m


def oi_confirmation_by_ticker(carry: pd.DataFrame) -> pd.DataFrame:
    if carry is None or carry.empty:
        return pd.DataFrame(columns=["ticker","oi_confirmation_score","oi_data_coverage"])

    def score(g: pd.DataFrame) -> pd.Series:
        w = g["prev_abs"].fillna(0.0)
        r = g["carryover_ratio"]
        ok = r.notna()
        coverage = float(ok.mean()) if len(g) else 0.0
        if w.sum() > 0 and ok.any():
            avg = float(np.average(r[ok], weights=w[ok]))
        else:
            avg = np.nan
        score_ = float(50.0*(avg+1.0)) if pd.notna(avg) else np.nan
        return pd.Series({"oi_confirmation_score": score_, "oi_data_coverage": coverage})

    out = carry.groupby("ticker").apply(score).reset_index()
    if out.empty:
        return pd.DataFrame(columns=["ticker","oi_confirmation_score","oi_data_coverage"])
    return out


def strike_psych(conn: sqlite3.Connection, d: date, tickers: Sequence[str], cfg: Config, spot: pd.DataFrame) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker","oi_magnet_tag","max_oi_strike","max_oi_concentration","max_oi_dist_pct"])
    ds = d.isoformat()
    t_place = ",".join(["?"] * len(tickers))
    oi = pd.read_sql_query(
        f"SELECT ticker, expiry, strike, option_type, open_interest FROM oi_snapshots WHERE trade_date=? AND ticker IN ({t_place})",
        conn, params=[ds] + list(tickers)
    )
    if oi.empty:
        return pd.DataFrame(columns=["ticker","oi_magnet_tag","max_oi_strike","max_oi_concentration","max_oi_dist_pct"])

    oi["open_interest"] = pd.to_numeric(oi["open_interest"], errors="coerce")
    oi["strike"] = pd.to_numeric(oi["strike"], errors="coerce")
    oi = oi.dropna(subset=["ticker","expiry","strike","open_interest"])

    oi["expiry_dt"] = pd.to_datetime(oi["expiry"], errors="coerce").dt.date
    oi["dte"] = (oi["expiry_dt"] - d).apply(lambda x: x.days if pd.notna(x) else np.nan)
    oi = oi[(oi["dte"] >= cfg.oi_dte_min) & (oi["dte"] <= cfg.oi_dte_max)]
    if oi.empty:
        return pd.DataFrame(columns=["ticker","oi_magnet_tag","max_oi_strike","max_oi_concentration","max_oi_dist_pct"])

    spot_map = dict(zip(spot["ticker"], spot["spot"]))
    rows = []
    for t, sub in oi.groupby("ticker"):
        s = spot_map.get(t, np.nan)
        if pd.isna(s) or s <= 0:
            continue
        strike_agg = sub.groupby("strike", as_index=False)["open_interest"].sum()
        total = float(strike_agg["open_interest"].sum())
        if total <= 0:
            continue
        top = strike_agg.sort_values("open_interest", ascending=False).head(1)
        max_strike = float(top["strike"].iloc[0])
        max_oi = float(top["open_interest"].iloc[0])
        conc = max_oi / total
        dist = abs(max_strike - s) / s

        tag = "none"
        if dist <= cfg.magnet_pct and conc >= 0.20:
            tag = "MAGNET/PIN"
        elif dist >= 0.05 and conc >= 0.15:
            tag = "WALL"
        elif dist >= 0.03 and conc >= 0.10:
            tag = "ZONE"

        rows.append({
            "ticker": t,
            "oi_magnet_tag": tag,
            "max_oi_strike": max_strike,
            "max_oi_concentration": conc,
            "max_oi_dist_pct": dist,
        })
    if not rows:
        return pd.DataFrame(columns=["ticker","oi_magnet_tag","max_oi_strike","max_oi_concentration","max_oi_dist_pct"])
    return pd.DataFrame(rows)


def dp_anchors(conn: sqlite3.Connection, lookback: List[date], tickers: Sequence[str], cfg: Config, spot: pd.DataFrame) -> pd.DataFrame:
    if not lookback or not tickers:
        return pd.DataFrame(columns=["ticker","spot","dp_support_1","dp_support_2","dp_resistance_1","dp_resistance_2"])
    last_dates = [x.isoformat() for x in lookback]
    d_place = ",".join(["?"] * len(last_dates))
    t_place = ",".join(["?"] * len(tickers))
    params = last_dates + list(tickers)

    dp = pd.read_sql_query(
        f"SELECT trade_date, ticker, price, shares, notional FROM dp_prints WHERE trade_date IN ({d_place}) AND ticker IN ({t_place})",
        conn, params=params
    )
    if dp.empty:
        return pd.DataFrame(columns=["ticker","spot","dp_support_1","dp_support_2","dp_resistance_1","dp_resistance_2"])

    dp["price"] = pd.to_numeric(dp["price"], errors="coerce")
    dp["notional"] = pd.to_numeric(dp["notional"], errors="coerce")
    dp = dp.dropna(subset=["ticker","price"])
    dp["notional"] = dp["notional"].fillna(dp["price"] * pd.to_numeric(dp["shares"], errors="coerce").fillna(0.0))

    med = dp.groupby("ticker")["price"].median().reset_index().rename(columns={"price":"median_price"})
    dp = dp.merge(med, on="ticker", how="left")
    dp["cluster_width"] = (dp["median_price"] * cfg.dp_cluster_pct).clip(lower=0.01)
    dp["price_cluster"] = (dp["price"] / dp["cluster_width"]).round() * dp["cluster_width"]
    dp["price_cluster"] = dp["price_cluster"].round(4)

    agg = dp.groupby(["ticker","price_cluster"], as_index=False).agg(
        dp_notional=("notional","sum"),
        dp_days=("trade_date", pd.Series.nunique),
        dp_prints=("price","count"),
    )
    look_n = max(len(last_dates), 1)
    agg["dp_strength"] = agg["dp_notional"] * (agg["dp_days"] / look_n)

    spot_map = dict(zip(spot["ticker"], spot["spot"]))
    rows = []
    for t, sub in agg.groupby("ticker"):
        s = spot_map.get(t, np.nan)
        if pd.isna(s) or s <= 0:
            continue
        supports = sub[sub["price_cluster"] < s].sort_values("dp_strength", ascending=False).head(2)
        resist = sub[sub["price_cluster"] > s].sort_values("dp_strength", ascending=False).head(2)
        rows.append({
            "ticker": t,
            "spot": float(s),
            "dp_support_1": float(supports["price_cluster"].iloc[0]) if len(supports) else np.nan,
            "dp_support_2": float(supports["price_cluster"].iloc[1]) if len(supports) > 1 else np.nan,
            "dp_resistance_1": float(resist["price_cluster"].iloc[0]) if len(resist) else np.nan,
            "dp_resistance_2": float(resist["price_cluster"].iloc[1]) if len(resist) > 1 else np.nan,
        })
    if not rows:
        return pd.DataFrame(columns=["ticker","spot","dp_support_1","dp_support_2","dp_resistance_1","dp_resistance_2"])
    return pd.DataFrame(rows)


# ----------------------------
# Build pack
# ----------------------------

def build_pack(conn: sqlite3.Connection, root: Path, out_dir: Path, cfg: Config, d: date, verbose: bool) -> Path:
    # Ensure ticker_day exists
    if verbose:
        print("  [compute] ticker_day ...")
    compute_ticker_day(conn, d, cfg)

    tickers = top_tickers_for_day(conn, d, cfg.top_tickers)

    # Always create files (even empty)
    day_dir = out_dir / "daily" / d.isoformat()
    day_dir.mkdir(parents=True, exist_ok=True)

    # Lookback dates
    dates = available_dates(conn)
    if d not in dates:
        dates = dates + [d]
        dates = sorted(set(dates))
    idx = dates.index(d)
    lookback = dates[max(0, idx - cfg.lookback_sessions + 1): idx + 1]

    # Spot
    spot = spot_from_exports(conn, d, tickers)
    if cfg.price_source in {"auto","yfinance"} and yf is not None and cfg.price_source != "export":
        missing = [t for t in tickers if t not in set(spot["ticker"].tolist())]
        if missing:
            spot2 = spot_from_yfinance(missing, d)
            if not spot2.empty:
                spot = pd.concat([spot, spot2], ignore_index=True)
    spot = spot.dropna(subset=["spot"]).drop_duplicates(subset=["ticker"], keep="last")

    # Persistence
    pers = None
    if tickers:
        if verbose:
            print("  [compute] persistence ...")
        for w in cfg.persistence_windows:
            p = persistence_window(conn, lookback, w, tickers)
            pers = p if pers is None else pers.merge(p, on="ticker", how="outer")

    # OI carryover + per-ticker score
    carry = pd.DataFrame()
    oi_by_ticker = pd.DataFrame(columns=["ticker","oi_confirmation_score","oi_data_coverage"])
    if tickers and idx > 0:
        prev_d = dates[idx - 1]
        if verbose:
            print(f"  [compute] OI carryover ({prev_d.isoformat()}→{d.isoformat()}) ...")
        carry = oi_carryover(conn, prev_d, d, tickers, cfg.top_signatures_n)
        oi_by_ticker = oi_confirmation_by_ticker(carry)

    # DP anchors + strike psych (optional; will be empty if no data)
    if verbose:
        print("  [compute] DP + strike psych ...")
    dp = dp_anchors(conn, lookback, tickers, cfg, spot)
    if verbose and (dp is None or dp.empty):
        print("  [compute] DP anchors empty (no dp_prints match lookback/tickers)")
    sp = strike_psych(conn, d, tickers, cfg, spot)

    # Screener slice
    scr = pd.read_sql_query("SELECT * FROM screener_daily WHERE trade_date=?", conn, params=(d.isoformat(),))

    # Daily features (include ticker_day stats)
    td = pd.read_sql_query("SELECT * FROM ticker_day WHERE trade_date=?", conn, params=(d.isoformat(),))
    base = pd.DataFrame({"ticker": tickers}).merge(td[["ticker","net","abs_total","daily_sign","eod_ratio"]] if not td.empty else pd.DataFrame(columns=["ticker","net","abs_total","daily_sign","eod_ratio"]), on="ticker", how="left")
    base = base.merge(spot, on="ticker", how="left")
    if pers is not None:
        base = base.merge(pers, on="ticker", how="left")
    base = base.merge(oi_by_ticker, on="ticker", how="left")
    base = base.merge(dp, on="ticker", how="left")
    base = base.merge(sp, on="ticker", how="left")

    def bias_row(r):
        s10 = r.get("persistence_10d_score", np.nan)
        sign10 = r.get("persistence_10d_dominant_sign", 0)
        oi = r.get("oi_confirmation_score", np.nan)
        if pd.isna(s10) or s10 < 50:
            return "neutral"
        if pd.notna(oi) and oi < 45:
            return "neutral"
        return "bullish" if sign10 == 1 else ("bearish" if sign10 == -1 else "neutral")
    if not base.empty:
        base["bias_hint"] = base.apply(bias_row, axis=1)

    # Write outputs (always)
    daily_cols = [
        "ticker","spot","net","abs_total","daily_sign","eod_ratio",
        "persistence_5d_score","persistence_5d_label","persistence_5d_dominant_sign","persistence_5d_active_days","persistence_5d_sessions",
        "persistence_10d_score","persistence_10d_label","persistence_10d_dominant_sign","persistence_10d_active_days","persistence_10d_sessions",
        "oi_confirmation_score","oi_data_coverage",
        "dp_support_1","dp_support_2","dp_resistance_1","dp_resistance_2",
        "oi_magnet_tag","max_oi_strike","max_oi_concentration","max_oi_dist_pct",
        "bias_hint",
    ]
    write_csv_always(day_dir / "daily_features.csv", base, daily_cols)

    dp_cols = ["ticker","spot","dp_support_1","dp_support_2","dp_resistance_1","dp_resistance_2"]
    write_csv_always(day_dir / "dp_anchors.csv", dp, dp_cols)

    carry_cols = ["ticker","contract_signature","prev_abs","prev_contracts","oi_prev","oi_cur","oi_delta","carryover_ratio"]
    write_csv_always(day_dir / "oi_carryover_signatures.csv", carry, carry_cols)

    # Screener: keep original columns if present; otherwise empty standard
    if scr is None or scr.empty:
        scr_cols = ["trade_date","ticker","call_volume","put_volume","call_premium","put_premium","total_open_interest","sector","source_file"]
        write_csv_always(day_dir / "stock_screener.csv", None, scr_cols)
    else:
        scr_path = day_dir / "stock_screener.csv"
        scr.to_csv(scr_path, index=False)

    plan_text = f"""# UW OS — Plan Skeleton ({d.isoformat()})

Upload `chatgpt_pack_{d.isoformat()}.zip` to ChatGPT with your daily prompt.

Files included:
- daily_features.csv
- dp_anchors.csv
- oi_carryover_signatures.csv
- stock_screener.csv
"""
    (day_dir / "plan_skeleton.md").write_text(plan_text, encoding="utf-8")

    # Zip pack
    pack_path = day_dir / f"chatgpt_pack_{d.isoformat()}.zip"
    with zipfile.ZipFile(pack_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in ["daily_features.csv","dp_anchors.csv","oi_carryover_signatures.csv","stock_screener.csv","plan_skeleton.md"]:
            zf.write(day_dir / fname, arcname=fname)

    if verbose:
        print(f"  Wrote: {pack_path}")
    return pack_path


def ingest_day(conn: sqlite3.Connection, folder: Path, d: date, cfg: Config, verbose: bool) -> None:
    # 1) Flow (whale preferred)
    root = folder.parent
    whale_md_name = f"whale-{d.isoformat()}.md"
    whale_md = (folder / whale_md_name) if (folder / whale_md_name).exists() else (root / whale_md_name)
    whale_csv = folder / "whale_trades_filtered.csv"
    hot = find_file(folder, "hot-chains", ".zip")

    kept_flow = 0
    if whale_md.exists():
        kept_flow = ingest_whale_md_file(conn, whale_md, d, cfg, verbose)
        if kept_flow == 0 and whale_csv.exists():
            kept_flow = ingest_whale_file(conn, whale_csv, d, cfg, verbose)
    elif whale_csv.exists():
        kept_flow = ingest_whale_file(conn, whale_csv, d, cfg, verbose)
    elif hot is not None:
        kept_flow = ingest_hot_chains_zip(conn, hot, d, cfg, verbose)

    if verbose:
        print(f"  [flow] inserted rows: {kept_flow:,}")

    # If no flow, still stop here; pack will be empty-ish but consistent.
    if kept_flow == 0:
        return

    # 2) Build ticker_day and pick top tickers (used to filter OI/DP/Screener ingest)
    compute_ticker_day(conn, d, cfg)
    tickers = top_tickers_for_day(conn, d, cfg.top_tickers)

    # 3) OI / DP / Screener filtered to tickers
    oi_zip = find_file(folder, "chain-oi-changes", ".zip")
    dp_zip = find_file(folder, "dp-eod-report", ".zip")
    scr_zip = find_file(folder, "stock-screener", ".zip")

    if oi_zip is not None:
        kept_oi = ingest_oi_zip_filtered(conn, oi_zip, d, tickers, cfg, verbose)
        if verbose:
            print(f"  [oi] inserted rows: {kept_oi:,}")
    if dp_zip is not None:
        kept_dp = ingest_dp_zip_filtered(conn, dp_zip, d, tickers, cfg, verbose)
        if verbose:
            print(f"  [dp] inserted rows: {kept_dp:,}")
    if scr_zip is not None:
        kept_scr = ingest_screener_zip_filtered(conn, scr_zip, d, tickers, cfg, verbose)
        if verbose:
            print(f"  [screener] inserted rows: {kept_scr:,}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="UW OS v6 FULLPACK — fast ingest + full zip outputs")
    ap.add_argument("--root", required=True, help="UW root folder with YYYY-MM-DD subfolders directly under it")
    ap.add_argument("--out", default=None, help="Output folder (default: <root>/_derived)")
    ap.add_argument("--only-date", default=None, help="Only process one date folder, e.g. 2025-12-23")
    ap.add_argument("--rebuild", action="store_true", help="Delete that day from DB before ingesting")
    ap.add_argument("--verbose", action="store_true", help="Print progress")
    ap.add_argument("--top-tickers", type=int, default=None, help="Limit OI/DP/screener ingest + analytics to top N tickers")
    ap.add_argument("--price-source", default=None, choices=["export","yfinance","none","auto"], help="Spot source override")
    ap.add_argument("--lookback", type=int, default=None, help="Lookback sessions for persistence/DP anchors")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve() if args.out else (root / "_derived")
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_or_create_config(out_dir)
    if args.top_tickers is not None:
        cfg.top_tickers = int(args.top_tickers)
    if args.price_source:
        cfg.price_source = args.price_source
    if args.lookback:
        cfg.lookback_sessions = int(args.lookback)

    db_path = out_dir / "uw_ledger.sqlite"
    conn = sqlite3.connect(str(db_path))
    init_db(conn)

    folders = find_date_folders(root, only_date=args.only_date)
    if not folders:
        print(f"No date folders found directly under root: {root}")
        return

    for d, folder in folders:
        if args.rebuild:
            delete_day(conn, d)
        if args.verbose:
            print(f"Ingesting {d.isoformat()} from {folder} ...")
        ingest_day(conn, folder, d, cfg, args.verbose)
        if args.verbose:
            print(f"Building full pack for {d.isoformat()} ...")
        build_pack(conn, root, out_dir, cfg, d, args.verbose)

    conn.close()


if __name__ == "__main__":
    main()
