
#!/usr/bin/env python3
"""Anu Options Engine audited no-GEX FIRE+SHIELD replacement.

Canonical upload filename: anu_analysis_v3_1_7.py
Internal engine version: 3.2.7-full-source-routing-audit

This script is a drop-in audited replacement for the project analysis file.
It removes automated GEX usage, builds spreads from actual hot-chain quotes,
deduplicates repeated whale seeds, supports FIRE debit verticals, supports
file-native SHIELD credit verticals, and pairs eligible anchored SHIELD rows
into iron condors when both sides exist on the same ticker and expiry.
This full-replacement build also fixes risk-side SHIELD strike replacement,
widens disclosed actual-chain wing fallback one tier, carries condor POP inputs
through SHIELD builds, uses whale equity-type as a focus fallback, keeps
internal condor pair selection EV/ML-first with Conviction only as tie-break,
defaults to scan-date OI and supports explicit next-day OI follow-through, and applies
broker-native Schwab Health Gate status plus same-ticker account exposure
notices when a Schwab positions artifact is present, prevents BUY/SELL publication when the computed size bucket is None, adds a
file-native family-flex translation pass that can test same-direction FIRE/SHIELD
alternate structures from actual hot-chain legs, and forces high-premium earnings
names into a blocked catalyst watch section instead of silently dropping them,
streams optional full bot EOD source ZIPs without loading 1GB into memory,
routes mid-cap/ETF/mixed-flow candidates into explicit lanes instead of hard-deleting them,
and emits blocked-positive-EV, alternates, top-symbol-gap, ETF-lane outputs, and conditional mixed-flow rescue rather than hard-deleting neutral-conflict debit rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="Downcasting object dtype arrays.*")

ASOF = date.today()  # overwritten by run(); no stale hard-coded scan date
PRIMARY_COLS = [
    "Ticker", "Action", "Buy leg", "Sell leg", "Expiry", "Net",
    "EV/ML", "POP", "Conviction", "Execution", "Notice", "Size",
]
INPUT_PATTERNS = {
    "whale": re.compile(r"^whale-(\d{4}-\d{2}-\d{2})\.md$"),
    "hot": re.compile(r"^hot-chains-(\d{4}-\d{2}-\d{2})\.zip$"),
    "oi": re.compile(r"^chain-oi-changes-(\d{4}-\d{2}-\d{2})\.zip$"),
    "screen": re.compile(r"^stock-screener-(\d{4}-\d{2}-\d{2})\.zip$"),
    "dp": re.compile(r"^dp-eod-report-(\d{4}-\d{2}-\d{2})\.zip$"),
    "schwab": re.compile(r"^schwab_positions_(\d{4}-\d{2}-\d{2})\.json$"),
    "bot": re.compile(r"^(?:bot-eod-report|eod-flow-report|flow-eod-report)-(\d{4}-\d{2}-\d{2})\.zip$"),
}
INDEX_SHIELD_ALLOW = {"VIX", "SPX", "NDX", "RUT"}
ETF_TICKERS = {"SPY", "QQQ", "IWM", "DIA", "VIX", "SPX", "SPXW", "NDX", "RUT", "GLD", "SLV", "USO", "XLE", "XLF", "XLK", "XLY", "XLP", "XLV", "XLI", "XLB", "XLU", "XOP", "SMH", "KRE", "TLT", "HYG", "LQD"}
FULL_SOURCE_CHUNK_ROWS = 250_000
FULL_SOURCE_MAX_GLOBAL_SEEDS = 3_000
FULL_SOURCE_MAX_PER_SYMBOL_FAMILY = 8
FULL_SOURCE_RESERVOIR_CAP = 20_000


ENGINE_VERSION = "3.2.7-full-source-routing-audit"
CANONICAL_MARKDOWN_FILES = [
    "Anu_Options_Engine_RULEBOOK_v3_0_6.md",
    "Anu_Options_Engine_RULEBOOK_v3_1_4_BROWSER.md",
    "Anu_Options_Engine_EXECUTION_PROTOCOL_v3_0_7.md",
    "Anu_Options_Engine_EXECUTION_PROTOCOL_v3_1_4_BROWSER.md",
]
CANONICAL_PYTHON_FILE = "anu_analysis_v3_1_7.py"


def extract_effective_logic_version(path: Path) -> Optional[str]:
    try:
        text = path.read_text(errors="ignore")
    except Exception:
        return None
    m = re.search(r"\*\*Effective logic version:\*\*\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"Internal engine version:\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()
    return None


def find_first_existing(filename: str, roots: Iterable[Path]) -> Optional[Path]:
    seen = set()
    for root in roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen:
            continue
        seen.add(root)
        p = root / filename
        if p.exists():
            return p
    return None


def audit_canonical_executable_sync(base_dir: Path, out_dir: Path, project_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Verify that the loaded markdown files and this root Python agree.

    v3.2.7 makes audit gate zero hard: all four markdown files must be present
    and must share ENGINE_VERSION with the root Python. To reduce local install
    friction, the resolver checks the script/base/out/current directories plus a
    caller-supplied --project-dir and one parent level for each directory.
    """
    script_path = Path(__file__).resolve()
    root_candidates = [script_path.parent, base_dir, out_dir, Path.cwd()]
    if project_dir is not None:
        root_candidates.insert(0, project_dir)
    roots = []
    seen_roots = set()
    for root in root_candidates:
        try:
            r = Path(root).resolve()
        except Exception:
            continue
        for candidate in [r, r.parent]:
            if candidate not in seen_roots:
                roots.append(candidate)
                seen_roots.add(candidate)
    records: List[Dict[str, Any]] = []
    detected_versions: Dict[str, str] = {CANONICAL_PYTHON_FILE: ENGINE_VERSION}
    records.append({
        "file": CANONICAL_PYTHON_FILE,
        "path": str(script_path),
        "effective_logic_version": ENGINE_VERSION,
        "sha256": sha256_file(script_path) if script_path.exists() else None,
        "present": bool(script_path.exists()),
    })
    missing = []
    for fn in CANONICAL_MARKDOWN_FILES:
        p = find_first_existing(fn, roots)
        if p is None:
            missing.append(fn)
            records.append({"file": fn, "path": None, "effective_logic_version": None, "sha256": None, "present": False})
            continue
        ver = extract_effective_logic_version(p)
        detected_versions[fn] = ver or "UNKNOWN"
        records.append({
            "file": fn,
            "path": str(p),
            "effective_logic_version": ver,
            "sha256": sha256_file(p),
            "present": True,
        })
    present_markdowns = [r for r in records if r["file"] in CANONICAL_MARKDOWN_FILES and r["present"]]
    mismatches = [r for r in present_markdowns if r.get("effective_logic_version") != ENGINE_VERSION]
    if not present_markdowns:
        status = "FAIL"
        note = "No canonical markdown files were found beside the executable/base/out dirs; audit gate zero requires all four markdown files plus the root Python from one bundle."
    elif missing:
        status = "FAIL"
        note = "One or more canonical markdown files are missing; replace all five project files from one bundle."
    elif mismatches:
        status = "FAIL"
        note = "Canonical markdown/Python effective logic versions do not match; replace all five project files from one bundle."
    else:
        status = "PASS"
        note = "All detected canonical project files share the same effective logic version."
    return {
        "status": status,
        "engine_version": ENGINE_VERSION,
        "records": records,
        "missing_files": missing,
        "mismatch_files": [r["file"] for r in mismatches],
        "search_roots": [str(r) for r in roots],
        "note": note,
    }


def input_scan_roots(base_dir: Path) -> List[Path]:
    """Return shallow input roots for dated EOD artifacts.

    A scan date folder such as .../tradedesk/2026-04-23 should not have to
    contain the 2026-04-24 OI overlay. The resolver therefore scans:

    - the requested base_dir
    - the parent directory for directly stored artifacts
    - date-named sibling folders under the parent, one level only

    This is intentionally shallow: it discovers sibling daily folders without
    recursively walking large archives or unrelated projects.
    """
    roots: List[Path] = []
    seen: set = set()

    def add(root: Path) -> None:
        try:
            r = root.resolve()
        except Exception:
            return
        if r.exists() and r.is_dir() and r not in seen:
            roots.append(r)
            seen.add(r)

    add(base_dir)
    parent = base_dir.parent
    add(parent)
    try:
        for child in parent.iterdir():
            if child.is_dir() and re.match(r"^\d{4}-\d{2}-\d{2}$", child.name):
                add(child)
    except Exception:
        pass
    return roots


def collect_dated_inputs(base_dir: Path) -> Dict[str, Dict[date, Path]]:
    found: Dict[str, Dict[date, Path]] = {key: {} for key in INPUT_PATTERNS}
    for root in input_scan_roots(base_dir):
        try:
            files = list(root.iterdir())
        except Exception:
            continue
        for p in files:
            if not p.is_file():
                continue
            for key, rx in INPUT_PATTERNS.items():
                m = rx.match(p.name)
                if m:
                    # Preserve first-hit precedence: base_dir wins, then parent,
                    # then sibling date folders. Sibling folders are discovered for
                    # explicit overlay/debug modes, but exact scan-date artifacts
                    # remain the production default.
                    found[key].setdefault(date.fromisoformat(m.group(1)), p)
    return found


def resolve_input_paths(base_dir: Path, asof: Optional[date] = None, use_next_day_oi: bool = False) -> Tuple[date, List[Path]]:
    """Resolve base EOD files plus OI handles.

    Production default is exact scan-date OI. A bounded next-calendar-day OI
    overlay is allowed only when the caller explicitly passes --use-next-day-oi.
    This prevents an as-of scan from silently mixing tomorrow's OI artifact while
    still supporting a deliberate follow-through overlay.
    """
    found = collect_dated_inputs(base_dir)
    flow_dates = set(found.get("whale", {})) | set(found.get("bot", {}))
    base_dates = sorted(flow_dates & set(found["hot"]) & set(found["screen"]) & set(found["dp"]))
    if not base_dates:
        raise FileNotFoundError(
            "Could not find a common scan date across whale/bot flow source, hot-chains, stock-screener, and dp-eod-report files"
        )
    scan_date = asof or base_dates[-1]
    if scan_date not in base_dates:
        available = ", ".join(d.isoformat() for d in base_dates)
        raise FileNotFoundError(
            f"Requested scan date {scan_date.isoformat()} not available. Available base dates: {available}"
        )

    oi_dates = sorted(found["oi"])
    if not oi_dates:
        raise FileNotFoundError("No chain-oi-changes-YYYY-MM-DD.zip files found")
    if use_next_day_oi and (scan_date + timedelta(days=1)) in found["oi"]:
        oi_curr_date = scan_date + timedelta(days=1)
    elif scan_date in found["oi"]:
        oi_curr_date = scan_date
    else:
        oi_curr_date = max((d for d in oi_dates if d <= scan_date), default=None)
    if oi_curr_date is None:
        raise FileNotFoundError(f"No chain OI file available on or before {scan_date.isoformat()}")
    oi_prev_date = max((d for d in oi_dates if d < oi_curr_date), default=oi_curr_date)

    flow_path = found.get("bot", {}).get(scan_date) or found.get("whale", {}).get(scan_date)
    if flow_path is None:
        raise FileNotFoundError(f"No whale markdown or full bot EOD flow source found for {scan_date.isoformat()}")
    return scan_date, [
        flow_path,
        found["hot"][scan_date],
        found["oi"][oi_prev_date],
        found["oi"][oi_curr_date],
        found["screen"][scan_date],
        found["dp"][scan_date],
    ]


def resolve_schwab_path(base_dir: Path, scan_date: date) -> Optional[Path]:
    found = collect_dated_inputs(base_dir).get("schwab", {})
    if not found:
        return None
    future_or_current = sorted(d for d in found if d >= scan_date)
    if future_or_current:
        return found[future_or_current[-1]]
    return found[sorted(found)[-1]]


def load_schwab_context(base_dir: Path, scan_date: date) -> Dict[str, Any]:
    path = resolve_schwab_path(base_dir, scan_date)
    if path is None:
        return {
            "path": None,
            "status": "UNKNOWN",
            "execution_label": "Bootstrap",
            "note": "No Schwab positions artifact found.",
            "rows_checked": 0,
            "issue_count": None,
            "positions_by_ticker": {},
            "input_record": None,
        }
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return {
            "path": str(path),
            "status": "UNKNOWN",
            "execution_label": "Bootstrap",
            "note": f"Schwab artifact could not be parsed: {exc}",
            "rows_checked": 0,
            "issue_count": None,
            "positions_by_ticker": {},
            "input_record": {"file": path.name, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)},
        }

    statuses: List[str] = []
    rows_checked = 0
    issue_count = 0
    positions_by_ticker: Dict[str, List[str]] = {}
    for acct in data.get("accounts", []):
        hg = acct.get("health_gate") or {}
        if int(hg.get("rows_checked") or 0) > 0 and hg.get("status"):
            statuses.append(str(hg.get("status")).upper())
            rows_checked += int(hg.get("rows_checked") or 0)
            issue_count += int(hg.get("issue_count") or 0)
        for pos in acct.get("positions", []):
            ticker = str(pos.get("underlying") or pos.get("ticker") or "").upper()
            if not ticker:
                continue
            qty = float(pos.get("quantity") or 0.0)
            if abs(qty) < 1e-9:
                continue
            if str(pos.get("asset_type", "")).upper() == "OPTION":
                opt_type = str(pos.get("option_type", "")).upper()[:1]
                strike = pos.get("strike")
                expiry = pos.get("expiry")
                side = "long" if qty > 0 else "short"
                if strike is not None and expiry:
                    desc = f"acct: {side} {abs(qty):g}x {float(strike):g}{opt_type} {expiry}"
                else:
                    desc = f"acct: {side} {abs(qty):g}x option"
            else:
                side = "long" if qty > 0 else "short"
                desc = f"acct: {side} {abs(qty):g} shares"
            positions_by_ticker.setdefault(ticker, []).append(desc)

    if not statuses:
        status = "UNKNOWN"
    elif any(s in {"FAIL", "FAILED", "BLOCK", "BLOCKED"} for s in statuses):
        status = "FAIL"
    elif any(s in {"WARN", "WARNING"} for s in statuses):
        status = "WARN"
    elif all(s == "PASS" for s in statuses):
        status = "PASS"
    else:
        status = statuses[0]
    execution_label = "Strict" if status == "PASS" else ("Blocked" if status == "FAIL" else "Bootstrap")
    return {
        "path": str(path),
        "status": status,
        "execution_label": execution_label,
        "note": f"broker-native Schwab health_gate.status={status}" if status != "UNKNOWN" else "Schwab artifact present but Health Gate unresolved.",
        "rows_checked": rows_checked,
        "issue_count": issue_count if statuses else None,
        "positions_by_ticker": positions_by_ticker,
        "input_record": {"file": path.name, "size_bytes": int(path.stat().st_size), "sha256": sha256_file(path)},
    }


def apply_schwab_execution_and_notices(df: pd.DataFrame, schwab: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Health Gate sets executable primary rows to Strict/Bootstrap, but WATCH rows
    # must remain watch-only. The previous implementation stamped Strict onto the
    # watch table after Schwab PASS, which was confusing and audit-hostile.
    action_text = out["Action"].astype(str).str.upper()
    existing_exec = out["Execution"].astype(str).str.upper() if "Execution" in out.columns else pd.Series("", index=out.index)
    executable_mask = (~action_text.str.contains("WATCH|TOP-SYMBOL GAP")) & (~existing_exec.eq("WATCH"))
    out.loc[executable_mask, "Execution"] = schwab.get("execution_label", "Bootstrap")
    positions = schwab.get("positions_by_ticker", {}) or {}
    for idx, row in out.iterrows():
        ticker = str(row.get("Ticker", "")).upper()
        pos_notes = positions.get(ticker, [])
        if not pos_notes:
            continue
        current = str(row.get("Notice", "none") or "none")
        parts = [] if current.lower() == "none" else [current]
        parts.append("; ".join(pos_notes[:3]))
        out.at[idx, "Notice"] = "; ".join(parts)
    return out


@dataclass
class BuildResult:
    record: Optional[Dict[str, Any]]
    reason: Optional[str]


def read_zip_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise FileNotFoundError(f"No CSV file found inside {path}")
        with zf.open(names[0]) as fh:
            return pd.read_csv(fh, low_memory=False, **kwargs)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize_inputs(paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        st = p.stat()
        rows.append({
            "file": p.name,
            "size_bytes": int(st.st_size),
            "sha256": sha256_file(p),
        })
    return pd.DataFrame(rows)



def parse_markdown_table_after_marker(text: str, marker: str) -> pd.DataFrame:
    """Parse a pipe-delimited markdown table immediately after a section marker."""
    if marker not in text:
        return pd.DataFrame()
    section = text.split(marker, 1)[1]
    lines = []
    started = False
    for ln in section.splitlines():
        if ln.startswith("|"):
            started = True
            lines.append(ln)
        elif started and ln.strip():
            break
    if len(lines) < 3:
        return pd.DataFrame()
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    records: List[List[str]] = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) == len(header):
            records.append(cells)
    if not records:
        return pd.DataFrame(columns=header)
    return pd.DataFrame(records, columns=header)


def parse_top_symbols_from_whale_markdown(path: Path) -> pd.DataFrame:
    text = path.read_text(errors="ignore")
    top = parse_markdown_table_after_marker(text, "## Top Symbols by Total Premium")
    if top.empty:
        return top
    if "underlying_symbol" not in top.columns:
        return pd.DataFrame()
    for col in ["count", "total_premium"]:
        if col in top.columns:
            top[col] = pd.to_numeric(top[col], errors="coerce")
    return top[[c for c in ["underlying_symbol", "count", "total_premium"] if c in top.columns]].copy()

def parse_whale_markdown(path: Path) -> pd.DataFrame:
    text = path.read_text()
    marker = "## Top 200 Yes-Prime Trades by Premium"
    if marker not in text:
        raise ValueError("Could not locate whale Top 200 table")
    section = text.split(marker, 1)[1]
    lines = [ln for ln in section.splitlines() if ln.startswith("|")]
    if len(lines) < 3:
        raise ValueError("Whale markdown table appears incomplete")
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    records: List[List[str]] = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != len(header):
            continue
        records.append(cells)
    df = pd.DataFrame(records, columns=header)
    numeric_cols = [
        "dte", "underlying_price", "strike", "price", "width", "pct_width",
        "size", "premium", "open_interest", "implied_volatility", "delta",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    df["cp"] = df["option_type"].astype(str).str.slice(0, 1).str.upper()

    df["seed_family"] = np.where(
        (df["track"] == "FIRE") & (df["net_type"] == "debit"),
        "FIRE_DEBIT",
        np.where((df["track"] == "SHIELD") & (df["net_type"] == "credit"), "SHIELD_CREDIT", "IGNORE"),
    )
    df["thesis_direction"] = np.where(
        df["seed_family"].eq("FIRE_DEBIT") & df["cp"].eq("C"),
        "bull",
        np.where(
            df["seed_family"].eq("FIRE_DEBIT") & df["cp"].eq("P"),
            "bear",
            np.where(
                df["seed_family"].eq("SHIELD_CREDIT") & df["cp"].eq("C"),
                "bear",
                np.where(df["seed_family"].eq("SHIELD_CREDIT") & df["cp"].eq("P"), "bull", "")
            ),
        ),
    )
    df["seed_key"] = (
        df["underlying_symbol"].astype(str) + "|" +
        df["track"].astype(str) + "|" +
        df["net_type"].astype(str) + "|" +
        df["expiry"].astype(str) + "|" +
        df["cp"].astype(str) + "|" +
        df["strike"].round(3).astype(str) + "|" +
        df["width"].round(3).astype(str)
    )
    top_symbols = parse_top_symbols_from_whale_markdown(path)
    df.attrs["source_mode"] = "markdown_top200"
    df.attrs["top_symbols"] = top_symbols
    df.attrs["full_source_summary"] = {
        "source_mode": "markdown_top200",
        "source_file": path.name,
        "rows_scanned": None,
        "yes_prime_candidates": None,
        "selected_seed_rows": int(len(df)),
    }
    return df



def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    rename = {c: re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_") for c in df.columns}
    out = df.rename(columns=rename).copy()
    aliases = {
        "underlying": "underlying_symbol",
        "underlyingsymbol": "underlying_symbol",
        "underlying_ticker": "underlying_symbol",
        "root_symbol": "underlying_symbol",
        "ticker": "underlying_symbol",
        "symbol": "underlying_symbol",
        "symbol_underlying": "underlying_symbol",
        "type": "option_type",
        "right": "option_type",
        "put_call": "option_type",
        "call_put": "option_type",
        "putcall": "option_type",
        "optiontype": "option_type",
        "option_side": "option_type",
        "cp_flag": "cp",
        "callput": "cp",
        "expiration": "expiry",
        "expiration_date": "expiry",
        "expiry_date": "expiry",
        "exp_date": "expiry",
        "option_expiration": "expiry",
        "option_expiry": "expiry",
        "underlying_close": "underlying_price",
        "underlyingprice": "underlying_price",
        "underlying_last": "underlying_price",
        "spot": "underlying_price",
        "spot_price": "underlying_price",
        "last_underlying_price": "underlying_price",
        "strike_price": "strike",
        "option_strike": "strike",
        "trade_price": "price",
        "fill_price": "price",
        "avg_fill_price": "price",
        "mark": "price",
        "mid": "price",
        "spread_width": "width",
        "target_width": "width",
        "width_dollars": "width",
        "pctwidth": "pct_width",
        "percent_width": "pct_width",
        "premium_dollars": "premium",
        "total_premium": "premium",
        "notional": "premium",
        "trade_premium": "premium",
        "iv": "implied_volatility",
        "impliedvolatility": "implied_volatility",
        "oi": "open_interest",
        "openinterest": "open_interest",
        "open_int": "open_interest",
        "open_interest_count": "open_interest",
        "fill_side": "side",
        "trade_side": "side",
        "aggressor_side": "side",
        "order_side": "side",
        "issue_type": "equity_type",
        "security_type": "equity_type",
        "asset_type": "equity_type",
        "market_cap": "marketcap",
        "market_capitalization": "marketcap",
        "mktcap": "marketcap",
        "strategy_track": "track",
        "engine_track": "track",
        "rule_track": "track",
        "signal_track": "track",
        "strategy_family": "track",
        "family": "track",
        "debit_credit": "net_type",
        "credit_debit": "net_type",
        "net_debit_credit": "net_type",
        "nettype": "net_type",
        "trade_net_type": "net_type",
    }
    for src, dest in aliases.items():
        if src in out.columns and dest not in out.columns:
            out[dest] = out[src]
    return out




def coalesce_alias_columns(out: pd.DataFrame, canonical: str) -> pd.DataFrame:
    """Coalesce duplicate canonical columns created by alias renames."""
    matches = [i for i, c in enumerate(list(out.columns)) if c == canonical]
    if len(matches) <= 1:
        return out
    block = out.iloc[:, matches].copy()
    vals = block.iloc[:, 0].copy()
    for j in range(1, block.shape[1]):
        nxt = block.iloc[:, j]
        empty = vals.isna() | vals.astype(str).str.strip().isin(["", "nan", "NaN", "None"])
        vals = vals.where(~empty, nxt)
    keep_idx = [i for i in range(len(out.columns)) if not (out.columns[i] == canonical and i != matches[0])]
    out = out.iloc[:, keep_idx].copy()
    out[canonical] = vals
    return out


def infer_text_field(out: pd.DataFrame, destination: str, text_cols: List[str], patterns: List[Tuple[str, str]]) -> pd.Series:
    base = out[destination].astype(str) if destination in out.columns else pd.Series("", index=out.index)
    current = base.where(~base.str.strip().isin(["", "nan", "NaN", "None"]), "")
    available = [c for c in text_cols if c in out.columns]
    if not available:
        return current
    joined = pd.Series("", index=out.index)
    for c in available:
        joined = (joined + " " + out[c].astype(str)).str.upper()
    for rx, val in patterns:
        mask = current.eq("") & joined.str.contains(rx, regex=True, na=False)
        current = current.mask(mask, val)
    return current

def yes_prime_mask_from_chunk(chunk: pd.DataFrame) -> pd.Series:
    """Detect a Yes-Prime flag when present; otherwise assume downstream filters decide."""
    candidates = [
        "yes_prime", "is_yes_prime", "yes_prime_candidate", "yesprime", "prime", "is_prime",
        "prime_status", "recommendation", "decision", "verdict", "rulebook_result", "passes_rulebook",
    ]
    for col in candidates:
        if col in chunk.columns:
            s = chunk[col].astype(str).str.strip().str.upper()
            return s.isin({"1", "TRUE", "T", "YES", "Y", "PASS", "PASSED", "YES-PRIME", "YES_PRIME", "PRIME"})
    return pd.Series(True, index=chunk.index)



def infer_vertical_width_from_price(underlying_prices: pd.Series, fallback_strikes: Optional[pd.Series] = None) -> pd.Series:
    """Infer the rulebook vertical width for raw single-leg bot rows.

    The generated whale summary uses a simple spot-price width ladder, then
    computes pct_width as option price / vertical width:

    - underlying price < 25   => 2.5-wide vertical
    - underlying price < 75   => 5-wide vertical
    - underlying price >= 75  => 10-wide vertical

    If spot is missing, fall back to the strike for a best-effort width.
    """
    x = pd.to_numeric(underlying_prices, errors="coerce")
    if fallback_strikes is not None:
        fx = pd.to_numeric(fallback_strikes, errors="coerce")
        x = x.where(x.notna() & (x > 0), fx)
    return pd.Series(
        np.select(
            [x < 25.0, x < 75.0],
            [2.5, 5.0],
            default=10.0,
        ),
        index=underlying_prices.index,
        dtype="float64",
    )


def infer_family_from_raw_side(out: pd.DataFrame) -> pd.DataFrame:
    """Infer FIRE/SHIELD family for raw bot rows when enriched columns are absent.

    In the generated Yes-Prime markdown for 2026-04-23, FIRE count exactly equals
    ask + mid + no_side rows, and SHIELD count exactly equals bid rows. The raw
    bot file exposes `side` but not `track`/`net_type`, so this is the necessary
    source-schema mapping for full-source mode:

    - bid      -> SHIELD credit
    - ask/mid/no_side/blank -> FIRE debit
    """
    side = out.get("side", pd.Series("", index=out.index)).astype(str).str.lower().str.strip()
    side_norm = side.replace({"nan": "", "none": "", "": "no_side", "no side": "no_side", "noside": "no_side"})
    out["side"] = side_norm

    track_raw = out.get("track", pd.Series("", index=out.index)).astype(str).str.strip()
    net_raw = out.get("net_type", pd.Series("", index=out.index)).astype(str).str.strip()
    empty_track = track_raw.eq("") | track_raw.str.lower().isin(["nan", "none"])
    empty_net = net_raw.eq("") | net_raw.str.lower().isin(["nan", "none"])

    bid_mask = side_norm.eq("bid")
    fire_side_mask = side_norm.isin(["ask", "mid", "no_side"])

    out.loc[empty_track & bid_mask, "track"] = "SHIELD"
    out.loc[empty_net & bid_mask, "net_type"] = "credit"
    out.loc[empty_track & fire_side_mask, "track"] = "FIRE"
    out.loc[empty_net & fire_side_mask, "net_type"] = "debit"
    return out

def normalize_flow_candidates(chunk: pd.DataFrame, scan_date: date) -> pd.DataFrame:
    out = normalize_column_names(chunk)
    for canonical in [
        "underlying_symbol", "track", "net_type", "option_type", "cp", "side", "equity_type",
        "expiry", "underlying_price", "strike", "price", "width", "pct_width", "size",
        "premium", "open_interest", "implied_volatility", "delta", "marketcap",
    ]:
        out = coalesce_alias_columns(out, canonical)
    # Required string fields.
    for col in ["underlying_symbol", "track", "net_type", "option_type", "side", "equity_type"]:
        if col not in out.columns:
            out[col] = ""

    text_cols = [
        "track", "strategy", "strategy_name", "strategy_family", "family", "setup", "setup_name",
        "trade_type", "trade_class", "signal", "label", "rule", "rulebook", "recommendation",
        "net_type", "option_type", "description",
    ]
    out["track"] = infer_text_field(out, "track", text_cols, [(r"\bFIRE\b", "FIRE"), (r"\bSHIELD\b", "SHIELD")])
    out["net_type"] = infer_text_field(out, "net_type", text_cols, [(r"\bDEBIT\b", "debit"), (r"\bCREDIT\b", "credit")])
    out["option_type"] = infer_text_field(out, "option_type", text_cols, [(r"\bCALL\b|\bCALLS\b|\bC\b", "call"), (r"\bPUT\b|\bPUTS\b|\bP\b", "put")])

    # Raw bot CSVs do not carry track/net_type; infer them from side before normalization.
    out = infer_family_from_raw_side(out)

    out["underlying_symbol"] = out["underlying_symbol"].astype(str).str.upper().str.strip()
    tr = out["track"].astype(str).str.upper().str.strip()
    out["track"] = np.where(tr.str.contains("FIRE", na=False), "FIRE", np.where(tr.str.contains("SHIELD", na=False), "SHIELD", tr))
    nt = out["net_type"].astype(str).str.lower().str.strip()
    out["net_type"] = np.where(nt.str.contains("debit", na=False), "debit", np.where(nt.str.contains("credit", na=False), "credit", nt))
    ot = out["option_type"].astype(str).str.lower().str.strip()
    out["option_type"] = np.where(ot.str.contains("call|^c$", regex=True, na=False), "call", np.where(ot.str.contains("put|^p$", regex=True, na=False), "put", ot))
    out["side"] = out["side"].astype(str).str.lower().str.strip().replace({"": "no_side", "nan": "no_side", "none": "no_side"})
    out["equity_type"] = out["equity_type"].astype(str).str.strip()
    if "expiry" not in out.columns:
        out["expiry"] = pd.NaT
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.date
    if "cp" not in out.columns:
        out["cp"] = out["option_type"].astype(str).str.slice(0, 1).str.upper()
    else:
        out["cp"] = out["cp"].astype(str).str.slice(0, 1).str.upper()
    for col in ["dte", "underlying_price", "strike", "price", "width", "pct_width", "size", "premium", "open_interest", "implied_volatility", "delta", "marketcap"]:
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    missing_premium = out["premium"].isna() & out["price"].notna() & out["size"].notna()
    out.loc[missing_premium, "premium"] = out.loc[missing_premium, "price"] * out.loc[missing_premium, "size"] * 100.0
    # Raw bot files have no vertical width. Infer the same width ladder used by
    # the whale summary, then compute pct_width as option price / vertical width.
    missing_width = out["width"].isna() & (out["underlying_price"].notna() | out["strike"].notna())
    if missing_width.any():
        out.loc[missing_width, "width"] = infer_vertical_width_from_price(
            out.loc[missing_width, "underlying_price"],
            fallback_strikes=out.loc[missing_width, "strike"],
        )
    # DTE and percent width fallbacks.
    missing_dte = out["dte"].isna() & out["expiry"].notna()
    if missing_dte.any():
        out.loc[missing_dte, "dte"] = out.loc[missing_dte, "expiry"].map(lambda x: (x - scan_date).days if pd.notna(x) else np.nan)
    missing_pct = out["pct_width"].isna() & out["price"].notna() & out["width"].notna() & (out["width"].abs() > 0)
    out.loc[missing_pct, "pct_width"] = out.loc[missing_pct, "price"] / out.loc[missing_pct, "width"].abs().clip(lower=0.01)
    out["seed_family"] = np.where(
        (out["track"].eq("FIRE")) & (out["net_type"].eq("debit")),
        "FIRE_DEBIT",
        np.where((out["track"].eq("SHIELD")) & (out["net_type"].eq("credit")), "SHIELD_CREDIT", "IGNORE"),
    )
    out["thesis_direction"] = np.where(
        out["seed_family"].eq("FIRE_DEBIT") & out["cp"].eq("C"),
        "bull",
        np.where(
            out["seed_family"].eq("FIRE_DEBIT") & out["cp"].eq("P"),
            "bear",
            np.where(
                out["seed_family"].eq("SHIELD_CREDIT") & out["cp"].eq("C"),
                "bear",
                np.where(out["seed_family"].eq("SHIELD_CREDIT") & out["cp"].eq("P"), "bull", "")
            ),
        ),
    )
    out["seed_key"] = (
        out["underlying_symbol"].astype(str) + "|" +
        out["track"].astype(str) + "|" +
        out["net_type"].astype(str) + "|" +
        out["expiry"].astype(str) + "|" +
        out["cp"].astype(str) + "|" +
        out["strike"].round(3).astype(str) + "|" +
        out["width"].round(3).astype(str)
    )
    # Rulebook-style filter. ETF is not excluded here; it is routed to a separate lane later.
    prime_mask = yes_prime_mask_from_chunk(out)
    if "canceled" in out.columns:
        canceled_s = out["canceled"].astype(str).str.strip().str.upper()
        prime_mask = prime_mask & ~canceled_s.isin({"1", "TRUE", "T", "YES", "Y"})
    fire_mask = out["seed_family"].eq("FIRE_DEBIT") & out["dte"].between(21, 70, inclusive="both") & (out["pct_width"].fillna(999) <= 0.45)
    shield_mask = out["seed_family"].eq("SHIELD_CREDIT") & out["dte"].between(28, 56, inclusive="both") & out["pct_width"].between(0.30, 0.55, inclusive="both")
    oi_mask = out["open_interest"].fillna(0) >= 100
    dist_mask = pd.Series(True, index=out.index)
    valid_dist = out["underlying_price"].fillna(0) > 0
    if valid_dist.any():
        dist = (out.loc[valid_dist, "strike"] - out.loc[valid_dist, "underlying_price"]).abs() / out.loc[valid_dist, "underlying_price"].abs()
        dist_mask.loc[valid_dist] = dist <= 0.80
    filtered = out[prime_mask & (fire_mask | shield_mask) & oi_mask & dist_mask].copy()
    return filtered


def prune_full_source_reservoir(frames: List[pd.DataFrame]) -> List[pd.DataFrame]:
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True, sort=False)
    if df.empty:
        return []
    df = df.sort_values(["seed_key", "premium"], ascending=[True, False]).drop_duplicates("seed_key", keep="first")
    global_top = df.sort_values("premium", ascending=False).head(FULL_SOURCE_MAX_GLOBAL_SEEDS)
    group_cols = ["underlying_symbol", "seed_family", "thesis_direction"]
    grouped = df.sort_values("premium", ascending=False).groupby(group_cols, group_keys=False).head(FULL_SOURCE_MAX_PER_SYMBOL_FAMILY)
    keep = pd.concat([global_top, grouped], ignore_index=True, sort=False).drop_duplicates("seed_key", keep="first")
    if len(keep) > FULL_SOURCE_RESERVOIR_CAP:
        keep = keep.sort_values("premium", ascending=False).head(FULL_SOURCE_RESERVOIR_CAP)
    return [keep]


def parse_full_bot_eod_zip(path: Path, scan_date: date) -> pd.DataFrame:
    """Stream a large bot/eod ZIP and keep a bounded candidate reservoir.

    This avoids loading a 1GB ZIP into memory. The engine stores only filtered
    Yes-Prime/rulebook-like candidates, global top premium rows, and per-symbol
    family representatives. Full-source ticker bias is accumulated from all
    filtered candidates and used for same-ticker direction.
    """
    rows_scanned = 0
    yes_candidates = 0
    schema_errors_sample: List[str] = []
    reservoir: List[pd.DataFrame] = []
    symbol_parts: List[pd.DataFrame] = []
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise FileNotFoundError(f"No CSV file found inside {path}")
        csv_name = max(names, key=lambda n: zf.getinfo(n).file_size)
        with zf.open(csv_name) as fh:
            for chunk in pd.read_csv(fh, chunksize=FULL_SOURCE_CHUNK_ROWS, low_memory=False):
                rows_scanned += int(len(chunk))
                try:
                    fc = normalize_flow_candidates(chunk, scan_date)
                except Exception as exc:
                    # Schema drift in the full 1GB source should not crash before
                    # the markdown fallback path can run. Keep scanning/fallback
                    # diagnostics instead.
                    if "schema_errors_sample" not in locals():
                        schema_errors_sample = []
                    if len(schema_errors_sample) < 5:
                        schema_errors_sample.append(str(exc))
                    continue
                if fc.empty:
                    continue
                yes_candidates += int(len(fc))
                # Accumulate symbol and direction totals from the whole filtered source.
                symbol_parts.append(fc.groupby(["underlying_symbol", "thesis_direction"], as_index=False).agg(
                    premium=("premium", "sum"), count=("seed_key", "count")
                ))
                top = fc.sort_values("premium", ascending=False).head(min(FULL_SOURCE_MAX_GLOBAL_SEEDS, len(fc)))
                per = fc.sort_values("premium", ascending=False).groupby(
                    ["underlying_symbol", "seed_family", "thesis_direction"], group_keys=False
                ).head(FULL_SOURCE_MAX_PER_SYMBOL_FAMILY)
                reservoir.append(pd.concat([top, per], ignore_index=True, sort=False))
                if sum(len(x) for x in reservoir) > FULL_SOURCE_RESERVOIR_CAP * 2:
                    reservoir = prune_full_source_reservoir(reservoir)
    selected = pd.concat(reservoir, ignore_index=True, sort=False) if reservoir else pd.DataFrame()
    selected = prune_full_source_reservoir([selected])[0] if not selected.empty else selected
    if not selected.empty:
        selected.attrs["source_mode"] = "full_bot_stream"
    # Full-source bias table.
    if symbol_parts:
        sym = pd.concat(symbol_parts, ignore_index=True, sort=False).groupby(["underlying_symbol", "thesis_direction"], as_index=False).agg(
            premium=("premium", "sum"), count=("count", "sum")
        )
        piv = sym.pivot(index="underlying_symbol", columns="thesis_direction", values="premium").fillna(0.0).reset_index()
        piv.columns.name = None
        cnt = sym.groupby("underlying_symbol", as_index=False)["count"].sum().rename(columns={"count": "count"})
        for col in ["bull", "bear"]:
            if col not in piv.columns:
                piv[col] = 0.0
        piv = piv.merge(cnt, on="underlying_symbol", how="left")
        piv["total_premium"] = piv["bull"] + piv["bear"]
        piv["whale_total"] = piv["total_premium"]
        piv["whale_bias"] = np.where(piv["whale_total"] > 0, (piv["bull"] - piv["bear"]) / piv["whale_total"], 0.0)
        piv["whale_lead_ratio"] = np.where(
            np.minimum(piv["bull"], piv["bear"]) > 0,
            np.maximum(piv["bull"], piv["bear"]) / np.minimum(piv["bull"], piv["bear"]),
            np.where(np.maximum(piv["bull"], piv["bear"]) > 0, np.inf, 1.0),
        )
        top_symbols = piv[["underlying_symbol", "count", "total_premium", "bull", "bear"]].sort_values("total_premium", ascending=False).head(100)
    else:
        piv = pd.DataFrame(columns=["underlying_symbol", "bull", "bear", "count", "total_premium", "whale_total", "whale_bias", "whale_lead_ratio"])
        top_symbols = pd.DataFrame(columns=["underlying_symbol", "count", "total_premium", "bull", "bear"])
    selected.attrs["source_mode"] = "full_bot_stream"
    selected.attrs["top_symbols"] = top_symbols
    selected.attrs["full_source_bias"] = piv
    selected.attrs["full_source_summary"] = {
        "source_mode": "full_bot_stream",
        "source_file": path.name,
        "rows_scanned": int(rows_scanned),
        "yes_prime_candidates": int(yes_candidates),
        "selected_seed_rows": int(len(selected)),
        "reservoir_cap": FULL_SOURCE_RESERVOIR_CAP,
        "chunk_rows": FULL_SOURCE_CHUNK_ROWS,
        "schema_errors_sample": schema_errors_sample,
        "schema_mapping": "raw_bot_side_inference: bid=>SHIELD credit; ask/mid/no_side=>FIRE debit; pct_width=price/vertical_width",
    }
    return selected


def parse_flow_source(path: Path, scan_date: date) -> pd.DataFrame:
    if path.suffix.lower() == ".zip" and re.search(r"(?:bot-eod-report|eod-flow-report|flow-eod-report)", path.name):
        return parse_full_bot_eod_zip(path, scan_date)
    return parse_whale_markdown(path)


def filter_candidate_seed_rows(whale: pd.DataFrame) -> pd.DataFrame:
    # v3.2.7 keeps ETFs/index products in the seed universe but routes them to
    # a separate ETF/index lane rather than the common-stock primary table.
    if whale.empty:
        return whale.copy()
    if "equity_type" not in whale.columns:
        whale = whale.copy()
        whale["equity_type"] = ""
    mask = whale["seed_family"].isin(["FIRE_DEBIT", "SHIELD_CREDIT"])
    return whale[mask].copy()



SEED_SCHEMA_COLUMNS = [
    "underlying_symbol", "track", "net_type", "option_type", "side", "expiry", "dte",
    "underlying_price", "strike", "price", "width", "pct_width", "size", "premium",
    "open_interest", "implied_volatility", "delta", "equity_type", "cp", "seed_family",
    "thesis_direction", "seed_key",
]


def ensure_seed_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee the seed dataframe has the canonical columns used downstream.

    v3.2.7 introduced streamed full-source ingestion. When the full-source CSV
    schema is not recognized, the selected reservoir can be an empty dataframe
    with no columns. Pandas then raises KeyError('seed_key') before the user gets
    a useful diagnostic. This guard makes the failure explicit and, when a whale
    markdown fallback exists, lets the run continue from the fallback source.
    """
    out = df.copy() if df is not None else pd.DataFrame()
    for col in SEED_SCHEMA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series(dtype="object")
    return out


def preview_flow_source_schema(path: Path, max_cols: int = 60) -> str:
    """Return a compact column preview for source-schema diagnostics."""
    try:
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    return f"{path.name}: no CSV member found"
                csv_name = max(names, key=lambda n: zf.getinfo(n).file_size)
                with zf.open(csv_name) as fh:
                    cols = list(pd.read_csv(fh, nrows=0).columns)
                shown = cols[:max_cols]
                suffix = "" if len(cols) <= max_cols else f" ... (+{len(cols)-max_cols} more)"
                return f"{path.name}::{csv_name} columns={shown}{suffix}"
        cols = list(pd.read_csv(path, nrows=0).columns)
        shown = cols[:max_cols]
        suffix = "" if len(cols) <= max_cols else f" ... (+{len(cols)-max_cols} more)"
        return f"{path.name} columns={shown}{suffix}"
    except Exception as exc:
        return f"{path.name}: schema preview failed: {exc}"


def maybe_fallback_to_whale_markdown(
    base_dir: Path,
    scan_date: date,
    flow_path: Path,
    whale: pd.DataFrame,
    raw_seed_rows: pd.DataFrame,
    source_mode: str,
    top_symbols_source: pd.DataFrame,
    full_source_bias: Optional[pd.DataFrame],
    full_source_summary: Dict[str, Any],
    allow_markdown_seed_fallback: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, pd.DataFrame, Optional[pd.DataFrame], Dict[str, Any]]:
    """Fallback from an unrecognized full-source CSV to whale markdown, if present.

    This is not a bypass of full-source preference. It is a defensive route for
    schema drift: if the streamed source yields no executable seed rows, use the
    markdown summary rather than crashing, and disclose the fallback in audit.
    """
    needs_fallback = raw_seed_rows.empty or ("seed_key" not in raw_seed_rows.columns)
    if not needs_fallback:
        return whale, raw_seed_rows, source_mode, top_symbols_source, full_source_bias, full_source_summary
    if not allow_markdown_seed_fallback:
        summary = dict(full_source_summary or {})
        summary["seed_source_fallback_available_but_disabled"] = {
            "reason": "full-source stream yielded zero executable seed rows or lacked seed_key; default is to fail loudly rather than silently fall back",
            "source_schema_preview": preview_flow_source_schema(flow_path),
        }
        return whale, raw_seed_rows, source_mode, top_symbols_source, full_source_bias, summary

    md_path = collect_dated_inputs(base_dir).get("whale", {}).get(scan_date)
    if md_path is None or md_path == flow_path:
        return whale, raw_seed_rows, source_mode, top_symbols_source, full_source_bias, full_source_summary

    try:
        fallback_whale = parse_whale_markdown(md_path)
        fallback_rows = filter_candidate_seed_rows(fallback_whale)
    except Exception:
        return whale, raw_seed_rows, source_mode, top_symbols_source, full_source_bias, full_source_summary

    if fallback_rows.empty or "seed_key" not in fallback_rows.columns:
        return whale, raw_seed_rows, source_mode, top_symbols_source, full_source_bias, full_source_summary

    summary = dict(full_source_summary or {})
    summary["seed_source_fallback"] = {
        "from": flow_path.name,
        "to": md_path.name,
        "reason": "full-source stream selected zero executable seed rows or lacked seed_key; likely CSV schema drift",
        "source_schema_preview": preview_flow_source_schema(flow_path),
    }
    return (
        fallback_whale,
        fallback_rows,
        f"{source_mode}_markdown_seed_fallback",
        fallback_whale.attrs.get("top_symbols", top_symbols_source),
        None,
        summary,
    )


def dedupe_seed_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    ordered = rows.sort_values(
        ["seed_key", "premium", "open_interest", "size"],
        ascending=[True, False, False, False],
    ).copy()
    deduped = ordered.drop_duplicates("seed_key", keep="first").reset_index(drop=True)
    return deduped


def parse_option_symbol_columns(df: pd.DataFrame, symbol_col: str = "option_symbol") -> pd.DataFrame:
    out = df.copy()
    if symbol_col not in out.columns:
        return out
    parsed = out[symbol_col].astype(str).str.extract(
        r"^(?P<underlying>[A-Z]+)(?P<yymmdd>\d{6})(?P<cp>[CP])(?P<strike_raw>\d{8})$"
    )
    if "underlying" not in out.columns:
        out["underlying"] = parsed["underlying"]
    if "underlying_symbol" not in out.columns:
        out["underlying_symbol"] = parsed["underlying"]
    if "cp" not in out.columns:
        out["cp"] = parsed["cp"]
    if "expiry" not in out.columns:
        out["expiry"] = pd.to_datetime(parsed["yymmdd"], format="%y%m%d", errors="coerce").dt.date
    if "strike" not in out.columns:
        out["strike"] = pd.to_numeric(parsed["strike_raw"], errors="coerce") / 1000.0
    return out


def normalize_hot(hot: pd.DataFrame) -> pd.DataFrame:
    out = parse_option_symbol_columns(hot)
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.date
    out["cp"] = out["cp"].astype(str).str.upper()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    for col in ["open_interest", "volume", "bid", "ask", "avg_price", "close", "premium"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_oi(oi: pd.DataFrame) -> pd.DataFrame:
    out = parse_option_symbol_columns(oi)
    out["expiry"] = pd.to_datetime(out["expiry"], errors="coerce").dt.date
    out["cp"] = out["cp"].astype(str).str.upper()
    out["strike"] = pd.to_numeric(out["strike"], errors="coerce")
    needed = [
        "underlying", "expiry", "cp", "strike", "oi_change", "curr_oi",
        "last_oi", "prev_ask_volume", "prev_bid_volume", "prev_total_premium",
    ]
    for col in needed:
        if col not in out.columns:
            out[col] = np.nan
    for col in needed[4:]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    slim = out[needed].copy()
    slim["strike_key"] = slim["strike"].round(6)
    # Indexed OI prevents hundreds of repeated 270k-row boolean scans during
    # family-flex builds. Duplicate keys are allowed; oi_context uses first row.
    return slim.set_index(["underlying", "expiry", "cp", "strike_key"], drop=False).sort_index()


def normalize_screen(screen: pd.DataFrame) -> pd.DataFrame:
    out = screen.copy()
    if "next_earnings_date" in out.columns:
        out["next_earnings_date"] = pd.to_datetime(out["next_earnings_date"], errors="coerce").dt.date
    numeric = [
        "close", "bullish_premium", "bearish_premium", "net_call_premium",
        "net_put_premium", "marketcap", "implied_move_perc",
    ]
    for col in numeric:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def normalize_dp(dp: pd.DataFrame) -> pd.DataFrame:
    out = dp.copy()
    for col in ["premium", "price", "size"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def build_bias_tables(seeds: pd.DataFrame, screen: pd.DataFrame, full_source_bias: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if seeds.empty and (full_source_bias is None or full_source_bias.empty):
        return pd.DataFrame(columns=["underlying_symbol"])
    if full_source_bias is not None and not full_source_bias.empty:
        # Prefer full-source ticker direction when a streamed 1GB bot file exists.
        wdir_piv = full_source_bias.copy()
        if "total_premium" in wdir_piv.columns and "whale_total" not in wdir_piv.columns:
            wdir_piv["whale_total"] = wdir_piv["total_premium"]
        for col in ["bull", "bear", "whale_total"]:
            if col not in wdir_piv.columns:
                wdir_piv[col] = 0.0
        if "whale_bias" not in wdir_piv.columns:
            wdir_piv["whale_bias"] = np.where(
                wdir_piv["whale_total"] > 0.0,
                (wdir_piv["bull"] - wdir_piv["bear"]) / wdir_piv["whale_total"],
                0.0,
            )
        if "whale_lead_ratio" not in wdir_piv.columns:
            wdir_piv["whale_lead_ratio"] = np.where(
                np.minimum(wdir_piv["bull"], wdir_piv["bear"]) > 0.0,
                np.maximum(wdir_piv["bull"], wdir_piv["bear"]) / np.minimum(wdir_piv["bull"], wdir_piv["bear"]),
                np.where(np.maximum(wdir_piv["bull"], wdir_piv["bear"]) > 0.0, np.inf, 1.0),
            )
    else:
        wdir = seeds.groupby(["underlying_symbol", "thesis_direction"], as_index=False)["premium"].sum()
        wdir_piv = wdir.pivot(index="underlying_symbol", columns="thesis_direction", values="premium").fillna(0.0).reset_index()
        wdir_piv.columns.name = None
        for col in ["bull", "bear"]:
            if col not in wdir_piv.columns:
                wdir_piv[col] = 0.0
        wdir_piv["whale_total"] = wdir_piv["bull"] + wdir_piv["bear"]
        wdir_piv["whale_bias"] = np.where(
            wdir_piv["whale_total"] > 0.0,
            (wdir_piv["bull"] - wdir_piv["bear"]) / wdir_piv["whale_total"],
            0.0,
        )
        wdir_piv["whale_lead_ratio"] = np.where(
            np.minimum(wdir_piv["bull"], wdir_piv["bear"]) > 0.0,
            np.maximum(wdir_piv["bull"], wdir_piv["bear"]) / np.minimum(wdir_piv["bull"], wdir_piv["bear"]),
            np.where(np.maximum(wdir_piv["bull"], wdir_piv["bear"]) > 0.0, np.inf, 1.0),
        )

    cols = [
        "ticker", "close", "bullish_premium", "bearish_premium", "net_call_premium",
        "net_put_premium", "next_earnings_date", "implied_move_perc", "marketcap", "issue_type",
    ]
    for col in cols:
        if col not in screen.columns:
            screen[col] = np.nan
    scr = screen[cols].copy()
    scr["screen_total"] = scr["bullish_premium"].fillna(0.0) + scr["bearish_premium"].fillna(0.0)
    scr["screen_bias"] = np.where(
        scr["screen_total"] > 0.0,
        (scr["bullish_premium"].fillna(0.0) - scr["bearish_premium"].fillna(0.0)) / scr["screen_total"],
        0.0,
    )

    bias = wdir_piv.merge(scr, left_on="underlying_symbol", right_on="ticker", how="left")
    bias["screen_bias"] = bias["screen_bias"].fillna(0.0)
    bias["combined_bias"] = 0.7 * bias["whale_bias"].fillna(0.0) + 0.3 * bias["screen_bias"]
    bias["dominant_direction"] = np.where(bias["combined_bias"] >= 0.0, "bull", "bear")
    bias["dominance_strength"] = bias["combined_bias"].abs()
    bias["has_both_sides"] = (bias["bull"] > 0.0) & (bias["bear"] > 0.0)
    return bias


def minority_flow_block(row: pd.Series) -> bool:
    dom = row.get("dominant_direction")
    strength = float(row.get("dominance_strength", 0.0) or 0.0)
    ratio = float(row.get("whale_lead_ratio", 1.0) or 1.0)
    if not dom:
        return False
    if str(row.get("seed_family")) == "FIRE_DEBIT":
        return bool(row["thesis_direction"] != dom and strength >= 0.18 and ratio >= 1.35)
    return bool(row["thesis_direction"] != dom and strength >= 0.25 and ratio >= 1.50)


def split_flow_watch(row: pd.Series) -> bool:
    bull = float(row.get("bull", 0.0) or 0.0)
    bear = float(row.get("bear", 0.0) or 0.0)
    if bull <= 0.0 or bear <= 0.0:
        return False
    ratio = max(bull, bear) / max(min(bull, bear), 1.0)
    er = row.get("next_earnings_date")
    er_soon = er is not None and pd.notna(er) and 0 <= (er - ASOF).days <= 10
    return bool(ratio < 1.6 and er_soon)


def fire_neutral_conflict(row: pd.Series) -> bool:
    if str(row.get("seed_family")) != "FIRE_DEBIT":
        return False
    bull = float(row.get("bull", 0.0) or 0.0)
    bear = float(row.get("bear", 0.0) or 0.0)
    if bull <= 0.0 or bear <= 0.0:
        return False
    return bool(abs(float(row.get("combined_bias", 0.0) or 0.0)) < 0.20)


def shield_bias_mismatch(row: pd.Series) -> bool:
    if str(row.get("seed_family")) != "SHIELD_CREDIT":
        return False
    bias = float(row.get("combined_bias", 0.0) or 0.0)
    if str(row.get("cp")) == "C":
        return bool(bias > 0.10)
    return bool(bias < -0.10)


def get_hot_chain(hot: pd.DataFrame, ticker: str, expiry: date, cp: str) -> pd.DataFrame:
    df = hot[(hot["underlying"] == ticker) & (hot["expiry"] == expiry) & (hot["cp"] == cp)].copy()
    return df.sort_values("strike").reset_index(drop=True)


def rescue_expiry_attempts(hot: pd.DataFrame, ticker: str, cp: str, source_expiry: date, max_shift_days: int = 14) -> List[date]:
    expiries = sorted({x for x in hot[(hot["underlying"] == ticker) & (hot["cp"] == cp)]["expiry"].dropna().unique()})
    if not expiries:
        return [source_expiry]
    attempts = []
    if source_expiry in expiries:
        attempts.append(source_expiry)
    nearby = [e for e in expiries if e != source_expiry and abs((e - source_expiry).days) <= max_shift_days]
    nearby = sorted(nearby, key=lambda e: (abs((e - source_expiry).days), e))
    attempts.extend(nearby)
    if not attempts:
        attempts = [source_expiry]
    return attempts


def choose_family_flex_anchor_strike(hot: pd.DataFrame, row: pd.Series, dest_family: str, dest_cp: str) -> Optional[float]:
    """Pick an actual-chain anchor strike for a same-direction family translation.

    The source whale row still owns ticker, expiry, and thesis direction, but the
    destination structure may need the opposite option side. We therefore choose
    a near-ATM/OTM actual-chain strike instead of reusing an absurd source strike
    from the other side, e.g. a 90C call-credit anchor becoming a 90P put debit.
    """
    ticker = str(row.get("underlying_symbol", ""))
    expiry = row.get("expiry")
    chain = get_hot_chain(hot, ticker, expiry, dest_cp).copy()
    if chain.empty:
        return None
    spot = row.get("close")
    if spot is None or not pd.notna(spot) or float(spot) <= 0.0:
        spot = row.get("underlying_price")
    if spot is None or not pd.notna(spot) or float(spot) <= 0.0:
        spot = float(chain["strike"].median())
    spot = float(spot)
    chain["absdist"] = (chain["strike"] - spot).abs()
    chain["liq"] = np.log1p(chain["open_interest"].fillna(0.0)) + 0.25 * np.log1p(chain["volume"].fillna(0.0))

    if dest_family == "FIRE_DEBIT":
        # Debit translation: use a direct near-ATM/OTM long leg.
        if dest_cp == "C":
            valid = chain[chain["strike"] >= spot * 0.995].copy()
        else:
            valid = chain[chain["strike"] <= spot * 1.005].copy()
    else:
        # Credit translation: use a near-OTM short strike on the risk side.
        if dest_cp == "C":
            valid = chain[chain["strike"] >= spot].copy()
        else:
            valid = chain[chain["strike"] <= spot].copy()
    if valid.empty:
        valid = chain.copy()
    valid["absdist"] = (valid["strike"] - spot).abs()
    return float(valid.sort_values(["absdist", "liq"], ascending=[True, False]).iloc[0]["strike"])


def generate_family_flex_seed_rows(cand: pd.DataFrame, hot: pd.DataFrame) -> pd.DataFrame:
    """Create same-direction FIRE/SHIELD alternate-family seeds.

    This does not change same-ticker bias and does not inflate raw whale counts.
    It only asks: if the file-native flow implies bull/bear, is the other
    structure family a better risk-defined expression using real hot-chain legs?
    """
    rows: List[Dict[str, Any]] = []
    if cand.empty:
        return pd.DataFrame(columns=cand.columns)
    for _, src in cand.iterrows():
        family = str(src.get("seed_family", ""))
        direction = str(src.get("thesis_direction", ""))
        if direction not in {"bull", "bear"}:
            continue
        if family == "SHIELD_CREDIT":
            dest_family = "FIRE_DEBIT"
            dest_track = "FIRE"
            dest_net = "debit"
            dest_cp = "C" if direction == "bull" else "P"
        elif family == "FIRE_DEBIT":
            dest_family = "SHIELD_CREDIT"
            dest_track = "SHIELD"
            dest_net = "credit"
            dest_cp = "P" if direction == "bull" else "C"
        else:
            continue
        anchor = choose_family_flex_anchor_strike(hot, src, dest_family, dest_cp)
        if anchor is None:
            continue
        r = src.to_dict()
        r["source_seed_family"] = family
        r["source_cp"] = str(src.get("cp", ""))
        r["source_track"] = str(src.get("track", ""))
        r["source_seed_key"] = str(src.get("seed_key", ""))
        r["is_family_flex"] = True
        r["translation_type"] = f"{family}_TO_{dest_family}"
        r["track"] = dest_track
        r["net_type"] = dest_net
        r["seed_family"] = dest_family
        r["cp"] = dest_cp
        r["option_type"] = "call" if dest_cp == "C" else "put"
        r["strike"] = float(anchor)
        r["seed_key"] = f"FLEX|{r['source_seed_key']}|{dest_family}|{dest_cp}|{float(anchor):g}"
        r["dup_count"] = 1
        rows.append(r)
    if not rows:
        return pd.DataFrame(columns=cand.columns)
    out = pd.DataFrame(rows)
    for col in cand.columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[cand.columns.tolist() + [c for c in out.columns if c not in cand.columns]]


def pick_long_leg(chain: pd.DataFrame, whale_strike: float) -> Optional[pd.Series]:
    if chain.empty:
        return None
    exact = chain[np.isclose(chain["strike"], whale_strike)]
    if not exact.empty:
        return exact.sort_values(["open_interest", "volume"], ascending=[False, False]).iloc[0]
    tmp = chain.copy()
    tmp["long_dist"] = (tmp["strike"] - whale_strike).abs()
    return tmp.sort_values(["long_dist", "open_interest", "volume"], ascending=[True, False, False]).iloc[0]


def pick_credit_short_leg(chain: pd.DataFrame, whale_strike: float, direction: str) -> Optional[pd.Series]:
    if chain.empty:
        return None
    exact = chain[np.isclose(chain["strike"], whale_strike)]
    if not exact.empty:
        return exact.sort_values(["open_interest", "volume"], ascending=[False, False]).iloc[0]
    if direction == "bear":
        valid = chain[chain["strike"] >= whale_strike].copy()
    else:
        valid = chain[chain["strike"] <= whale_strike].copy()
    if valid.empty:
        return None
    valid["short_dist"] = (valid["strike"] - whale_strike).abs()
    return valid.sort_values(["short_dist", "open_interest", "volume"], ascending=[True, False, False]).iloc[0]


def conservative_long_price(leg: pd.Series) -> float:
    for field in ["ask", "avg_price", "close", "bid"]:
        val = pd.to_numeric(leg.get(field), errors="coerce")
        if pd.notna(val) and float(val) > 0.0:
            return float(val)
    return float("nan")


def conservative_short_price(leg: pd.Series) -> float:
    for field in ["bid", "avg_price", "close", "ask"]:
        val = pd.to_numeric(leg.get(field), errors="coerce")
        if pd.notna(val) and float(val) > 0.0:
            return float(val)
    return float("nan")



def restrict_width_band(candidates: pd.DataFrame, target_width: Optional[float]) -> pd.DataFrame:
    if candidates.empty or target_width is None or not pd.notna(target_width) or float(target_width) <= 0.0:
        return candidates
    tw = float(target_width)
    lower = 0.50 * tw
    tier1_upper = min(1.50 * tw, tw + 5.0)
    tier1 = candidates[(candidates["width"] >= lower - 1e-9) & (candidates["width"] <= tier1_upper + 1e-9)].copy()
    if not tier1.empty:
        return tier1
    tier2_upper = max(2.00 * tw, tw + 5.0)
    tier2 = candidates[(candidates["width"] >= lower - 1e-9) & (candidates["width"] <= tier2_upper + 1e-9)].copy()
    if not tier2.empty:
        return tier2
    return candidates.iloc[0:0].copy()


def debit_short_candidates(chain: pd.DataFrame, long_strike: float, direction: str, target_width: Optional[float] = None) -> pd.DataFrame:
    if direction == "bear":
        candidates = chain[chain["strike"] < long_strike].copy()
        candidates["width"] = long_strike - candidates["strike"]
    else:
        candidates = chain[chain["strike"] > long_strike].copy()
        candidates["width"] = candidates["strike"] - long_strike
    if candidates.empty:
        return candidates
    candidates["pct_width"] = candidates["width"] / max(long_strike, 1.0)
    candidates = candidates[candidates["pct_width"] <= 0.45].copy()
    if candidates.empty:
        return candidates
    candidates = restrict_width_band(candidates, target_width)
    candidates["liq"] = np.log1p(candidates["open_interest"].fillna(0.0)) + 0.25 * np.log1p(candidates["volume"].fillna(0.0))
    return candidates.sort_values(["width", "liq"], ascending=[True, False])


def credit_long_wing_candidates(chain: pd.DataFrame, short_strike: float, direction: str, target_width: Optional[float] = None) -> pd.DataFrame:
    if direction == "bear":
        candidates = chain[chain["strike"] > short_strike].copy()
        candidates["width"] = candidates["strike"] - short_strike
    else:
        candidates = chain[chain["strike"] < short_strike].copy()
        candidates["width"] = short_strike - candidates["strike"]
    if candidates.empty:
        return candidates
    candidates["pct_width"] = candidates["width"] / max(short_strike, 1.0)
    candidates = candidates[candidates["pct_width"] <= 0.60].copy()
    if candidates.empty:
        return candidates
    candidates = restrict_width_band(candidates, target_width)
    candidates["liq"] = np.log1p(candidates["open_interest"].fillna(0.0)) + 0.25 * np.log1p(candidates["volume"].fillna(0.0))
    return candidates.sort_values(["width", "liq"], ascending=[True, False])


def choose_best_fire_short(
    chain: pd.DataFrame,
    long_leg: pd.Series,
    row: pd.Series,
) -> Optional[pd.Series]:
    target_width = float(row["width"]) if pd.notna(row.get("width")) else np.nan
    candidates = debit_short_candidates(chain, float(long_leg["strike"]), str(row["thesis_direction"]), target_width)
    if candidates.empty:
        return None
    long_px = conservative_long_price(long_leg)
    if not pd.notna(long_px):
        return None
    close = float(row["close"]) if pd.notna(row.get("close")) else float(row["underlying_price"])
    iv = float(row["implied_volatility"]) if pd.notna(row.get("implied_volatility")) else 0.30

    best_idx: Optional[int] = None
    best_score = -1e9
    for idx, cand in candidates.iterrows():
        short_px = conservative_short_price(cand)
        if not pd.notna(short_px):
            continue
        net = float(long_px) - float(short_px)
        width = abs(float(long_leg["strike"]) - float(cand["strike"]))
        if net <= 0.0 or width <= 0.0:
            continue
        reward_risk = (width - net) / net
        if reward_risk <= 0.0:
            continue
        pop = compute_pop_debit(close, iv, int(row["dte"]), float(long_leg["strike"]), net, str(row["thesis_direction"]))
        ev_ml = pure_ev_ml(pop, reward_risk)
        penalty = 0.0 if not pd.notna(target_width) else 0.08 * abs(width - target_width) / max(target_width, 1.0)
        score = ev_ml - penalty + 0.001 * float(cand.get("liq", 0.0) or 0.0)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return None
    return candidates.loc[best_idx]


def choose_best_credit_wing(
    chain: pd.DataFrame,
    short_leg: pd.Series,
    row: pd.Series,
) -> Optional[pd.Series]:
    direction = str(row["thesis_direction"])
    target_width = float(row["width"]) if pd.notna(row.get("width")) else np.nan
    candidates = credit_long_wing_candidates(chain, float(short_leg["strike"]), direction, target_width)
    if candidates.empty:
        return None
    short_px = conservative_short_price(short_leg)
    if not pd.notna(short_px):
        return None
    close = float(row["close"]) if pd.notna(row.get("close")) else float(row["underlying_price"])
    iv = float(row["implied_volatility"]) if pd.notna(row.get("implied_volatility")) else 0.30

    best_idx: Optional[int] = None
    best_score = -1e9
    for idx, cand in candidates.iterrows():
        long_px = conservative_long_price(cand)
        if not pd.notna(long_px):
            continue
        credit = float(short_px) - float(long_px)
        width = abs(float(cand["strike"]) - float(short_leg["strike"]))
        if credit <= 0.0 or width <= credit:
            continue
        reward_risk = credit / (width - credit)
        pop = compute_pop_credit(close, iv, int(row["dte"]), float(short_leg["strike"]), credit, direction)
        ev_ml = pure_ev_ml(pop, reward_risk)
        penalty = 0.0 if not pd.notna(target_width) else 0.08 * abs(width - target_width) / max(target_width, 1.0)
        score = ev_ml - penalty + 0.001 * float(cand.get("liq", 0.0) or 0.0)
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return None
    return candidates.loc[best_idx]


def oi_context(oi_prev: pd.DataFrame, oi_curr: pd.DataFrame, ticker: str, expiry: date, cp: str, strike: float) -> Dict[str, float]:
    def first_match(df: pd.DataFrame) -> Optional[pd.Series]:
        if df.empty:
            return None
        key = (ticker, expiry, cp, round(float(strike), 6))
        try:
            row = df.loc[key]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            if row.empty:
                return None
            return row.iloc[0]
        return row

    row = first_match(oi_curr)
    oi_source = "oi_curr"
    if row is None:
        row = first_match(oi_prev)
        oi_source = "oi_prev"
    if row is None:
        return {}
    ask = float(row.get("prev_ask_volume", 0.0) or 0.0)
    bid = float(row.get("prev_bid_volume", 0.0) or 0.0)
    return {
        "oi_source": oi_source,
        "ask_bid_ratio": (ask + 1.0) / (bid + 1.0),
        "bid_ask_ratio": (bid + 1.0) / (ask + 1.0),
        "oi_change": float(row.get("oi_change", 0.0) or 0.0),
        "curr_oi": float(row.get("curr_oi", 0.0) or 0.0),
        "prev_total_premium": float(row.get("prev_total_premium", 0.0) or 0.0),
    }


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_pop_debit(close: float, iv: float, dte_days: int, long_strike: float, net_debit: float, direction: str) -> float:
    T = max(int(dte_days), 1) / 365.0
    sigma = max(float(iv or 0.30), 0.05) * math.sqrt(T)
    if sigma <= 0.0 or close <= 0.0:
        return 0.5
    if direction == "bear":
        breakeven = max(long_strike - net_debit, 0.01)
        z = (math.log(breakeven / close) + 0.5 * sigma * sigma) / sigma
        return float(norm_cdf(z))
    breakeven = max(long_strike + net_debit, 0.01)
    z = (math.log(breakeven / close) + 0.5 * sigma * sigma) / sigma
    return float(1.0 - norm_cdf(z))


def compute_pop_credit(close: float, iv: float, dte_days: int, short_strike: float, credit: float, direction: str) -> float:
    T = max(int(dte_days), 1) / 365.0
    sigma = max(float(iv or 0.30), 0.05) * math.sqrt(T)
    if sigma <= 0.0 or close <= 0.0:
        return 0.5
    if direction == "bear":
        breakeven = max(short_strike + credit, 0.01)
        z = (math.log(breakeven / close) + 0.5 * sigma * sigma) / sigma
        return float(norm_cdf(z))
    breakeven = max(short_strike - credit, 0.01)
    z = (math.log(breakeven / close) + 0.5 * sigma * sigma) / sigma
    return float(1.0 - norm_cdf(z))


def compute_pop_condor(close: float, iv: float, dte_days: int, lower: float, upper: float) -> float:
    T = max(int(dte_days), 1) / 365.0
    sigma = max(float(iv or 0.30), 0.05) * math.sqrt(T)
    if sigma <= 0.0 or close <= 0.0:
        return 0.5
    lower = max(lower, 0.01)
    upper = max(upper, lower + 0.01)
    z_low = (math.log(lower / close) + 0.5 * sigma * sigma) / sigma
    z_high = (math.log(upper / close) + 0.5 * sigma * sigma) / sigma
    return float(max(0.0, norm_cdf(z_high) - norm_cdf(z_low)))


def liquidity_score(long_oi: float, short_oi: float) -> float:
    return min((math.log1p(max(long_oi, 0.0)) + math.log1p(max(short_oi, 0.0))) / 18.0, 1.0)


def microstructure_score(oictx: Dict[str, float], mode: str) -> float:
    if not oictx:
        return 0.50
    if mode == "debit":
        ratio = float(oictx.get("ask_bid_ratio", 1.0) or 1.0)
    elif mode == "credit":
        ratio = float(oictx.get("bid_ask_ratio", 1.0) or 1.0)
    else:
        ratio = float(oictx.get("micro_ratio", 1.0) or 1.0)
    if ratio <= 0.0:
        return 0.50
    return float(min(max((math.log(ratio) / math.log(4.0) + 0.5), 0.0), 1.0))


def oi_follow_score(oictx: Dict[str, float]) -> float:
    val = float(oictx.get("oi_change", 0.0) or 0.0)
    return float(min(max((val * 1.5) + 0.5, 0.0), 1.0))


def conviction_raw(whale_premium: float, row: pd.Series, oictx: Dict[str, float], long_oi: float, short_oi: float, mode: str) -> float:
    whale_total = max(float(row.get("whale_total", 0.0) or 0.0), 1.0)
    whale_dom = abs(float(row.get("whale_bias", 0.0) or 0.0))
    screen_dom = abs(float(row.get("screen_bias", 0.0) or 0.0))
    prem_share = min(float(whale_premium) / whale_total, 1.0)
    micro = microstructure_score(oictx, mode)
    oi_follow = oi_follow_score(oictx)
    liq = liquidity_score(long_oi, short_oi)
    score = 0.28 * whale_dom + 0.18 * screen_dom + 0.18 * prem_share + 0.16 * micro + 0.10 * oi_follow + 0.10 * liq
    return max(0.0, min(score, 1.0))


def conviction_raw_condor(call_row: pd.Series, put_row: pd.Series, pop_neutrality: float) -> float:
    base = 0.5 * (float(call_row.get("conviction_raw", 0.0) or 0.0) + float(put_row.get("conviction_raw", 0.0) or 0.0))
    neutrality = 1.0 - min(abs(float(call_row.get("combined_bias", 0.0) or 0.0)) / 0.20, 1.0)
    score = 0.75 * base + 0.25 * max(pop_neutrality, neutrality)
    return max(0.0, min(score, 1.0))


def conviction_pct(raw: float, structure_kind: str) -> int:
    base = 25.0 if structure_kind == "debit_vertical" else (30.0 if structure_kind == "credit_vertical" else 35.0)
    pct = base + 70.0 * max(0.0, min(raw, 1.0))
    return int(round(max(1.0, min(pct, 99.0))))


def pure_ev_ml(pop: float, reward_risk: float) -> float:
    return float(pop * reward_risk - (1.0 - pop))


def size_bucket(ev_ml: float, structure_kind: str) -> str:
    if structure_kind == "debit_vertical":
        if ev_ml >= 1.0:
            return "Tiny"
        if ev_ml >= 0.40:
            return "Starter"
        if ev_ml >= 0.10:
            return "Pilot"
        return "None"
    if structure_kind == "credit_vertical":
        if ev_ml >= 0.40:
            return "Starter"
        if ev_ml >= 0.10:
            return "Pilot"
        if ev_ml >= 0.03:
            return "Pilot"
        return "None"
    if ev_ml >= 0.30:
        return "Starter"
    if ev_ml >= 0.08:
        return "Pilot"
    return "None"


def apply_event_size_cap(size: str, er_days: Optional[int]) -> str:
    if er_days is None:
        return size
    if 11 <= er_days <= 14 and size in {"Tiny", "Starter"}:
        return "Pilot"
    return size


def format_leg(ticker: str, expiry: date, strike: float, cp: str, side_word: str) -> str:
    strike_txt = str(int(strike)) if float(strike).is_integer() else str(strike)
    return f"{side_word} {ticker} {expiry} {strike_txt}{cp}"


def is_etf_or_index(issue_type: Any, ticker: str) -> bool:
    issue = str(issue_type or "").lower()
    ticker = str(ticker or "").upper()
    return (
        ticker in ETF_TICKERS
        or "etf" in issue
        or "exchange traded" in issue
        or "fund" in issue
        or "index" in issue
    )


def issue_type_focus_ok(issue_type: Any, ticker: str, seed_family: str) -> bool:
    issue = str(issue_type or "").lower()
    if "common stock" in issue or "adr" in issue:
        return True
    if seed_family == "SHIELD_CREDIT" and ticker in INDEX_SHIELD_ALLOW:
        return True
    # ETF/index products are handled in their own lane, not common-stock primary.
    return False


def liquidity_tier(row: pd.Series) -> str:
    if is_etf_or_index(row.get("issue_type"), str(row.get("Ticker", ""))):
        return "ETF_INDEX"
    mc = row.get("marketcap", np.nan)
    try:
        mc = float(mc)
    except Exception:
        mc = np.nan
    if pd.notna(mc) and mc >= 8e10:
        return "MAJOR"
    if pd.notna(mc) and mc >= 1e10:
        return "MID_PILOT"
    if pd.notna(mc) and mc > 0:
        return "SMALL_WATCH"
    return "UNKNOWN_WATCH"


def neutral_conflict_rescue(row: pd.Series) -> bool:
    """Conditional promotion lane for mixed-flow FIRE debit rows.

    Neutral-conflict means same-ticker flow is mixed; it should warn and route,
    not delete. The rescue is intentionally bounded: hard blocks remain hard
    (event block, minority flow, split-flow near event, invalid size). A row can
    publish only when the built structure itself has positive edge and at least
    one independent execution/follow-through signal from OI or quote-side data.
    The material combined-bias gate is intentionally NOT required here because
    neutral-conflict is defined by weak combined bias.
    """
    if not bool(row.get("neutral_conflict", False)):
        return False
    if str(row.get("seed_family", "")) != "FIRE_DEBIT" or str(row.get("structure_kind", "")) != "debit_vertical":
        return False
    if bool(row.get("minority_flow", False)) or bool(row.get("split_flow_watch", False)):
        return False
    if bool(row.get("inside_event_block", False)):
        return False
    if not bool(row.get("has_executable_size", False)):
        return False
    ev = float(row.get("EV/ML", -999) or -999)
    pop = float(row.get("POP", 0) or 0)
    conv = float(row.get("Conviction", 0) or 0)
    rr = float(row.get("reward_risk", 0) or 0)
    oictx = row.get("oictx", {}) if isinstance(row.get("oictx", {}), dict) else {}
    oi_change = float(oictx.get("oi_change", 0.0) or 0.0)
    ask_bid_ratio = float(oictx.get("ask_bid_ratio", 1.0) or 1.0)
    curr_oi = float(oictx.get("curr_oi", 0.0) or 0.0)
    # Debit rescue requires positive expectancy and a structure that is not just
    # pennies-for-a-miracle. POP>=0.15 admits practical alternates such as tight
    # call debits; high conviction can rescue a lower POP convexity row only when
    # EV is stronger.
    edge_ok = ev >= 0.10 and rr > 0
    probability_ok = pop >= 0.15 or (ev >= 0.40 and conv >= 60)
    followthrough_ok = oi_change >= 0.10 or ask_bid_ratio >= 1.25 or curr_oi >= 1000
    return bool(edge_ok and probability_ok and followthrough_ok)


def add_notice_text(notice: Any, extra: str) -> str:
    base = str(notice or "none")
    if base.lower() == "none" or not base.strip():
        return extra
    if extra in base:
        return base
    return base + "; " + extra


def apply_quality_and_tier_notices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "liquidity_tier" not in out.columns:
        out["liquidity_tier"] = out.apply(liquidity_tier, axis=1)
    if "mixed_flow_rescue" not in out.columns:
        out["mixed_flow_rescue"] = out.apply(neutral_conflict_rescue, axis=1) if "neutral_conflict" in out.columns else False
    for idx, row in out.iterrows():
        pop = row.get("POP")
        try:
            popf = float(pop)
        except Exception:
            popf = np.nan
        if pd.notna(popf):
            if popf < 0.05:
                out.at[idx, "Notice"] = add_notice_text(row.get("Notice"), "lottery/convexity only")
            elif popf < 0.15:
                out.at[idx, "Notice"] = add_notice_text(row.get("Notice"), "low-POP convexity")
        tier = row.get("liquidity_tier")
        if tier == "MID_PILOT":
            out.at[idx, "Notice"] = add_notice_text(out.at[idx, "Notice"], "mid-cap pilot tier")
            if str(out.at[idx, "Size"]) in {"Tiny", "Starter"}:
                out.at[idx, "Size"] = "Pilot"
        elif tier in {"SMALL_WATCH", "UNKNOWN_WATCH"}:
            out.at[idx, "Notice"] = add_notice_text(out.at[idx, "Notice"], tier.lower().replace("_", "-"))
        elif tier == "ETF_INDEX":
            out.at[idx, "Notice"] = add_notice_text(out.at[idx, "Notice"], "ETF/index lane")
        if bool(out.at[idx, "mixed_flow_rescue"]):
            out.at[idx, "Notice"] = add_notice_text(out.at[idx, "Notice"], "mixed-flow rescue; live entry gate required")
    return out


def compute_er_days(row: pd.Series) -> Optional[int]:
    er = row.get("next_earnings_date")
    if er is None or pd.isna(er):
        return None
    return int((er - ASOF).days)


def shield_anchor_ok(row: pd.Series, short_leg: pd.Series, long_leg: pd.Series, oictx: Dict[str, float]) -> bool:
    side = str(row.get("side", "")).lower()
    seller_led = side == "bid" or (side == "mid" and float(oictx.get("bid_ask_ratio", 1.0) or 1.0) >= 1.25)
    liquidity_ok = float(short_leg.get("open_interest", 0.0) or 0.0) >= 100 and float(long_leg.get("open_interest", 0.0) or 0.0) >= 100
    flow_ok = (
        float(oictx.get("bid_ask_ratio", 0.0) or 0.0) >= 1.0
        or float(oictx.get("oi_change", -1.0) or -1.0) >= 0.0
        or float(oictx.get("curr_oi", 0.0) or 0.0) >= 1000.0
    )
    return bool(seller_led and liquidity_ok and flow_ok and (not shield_bias_mismatch(row)))


def build_fire_candidate(row: pd.Series, hot: pd.DataFrame, oi_prev: pd.DataFrame, oi_curr: pd.DataFrame) -> BuildResult:
    ticker = str(row["underlying_symbol"])
    expiry = row["expiry"]
    cp = str(row["cp"])
    whale_strike = float(row["strike"])
    direction = str(row["thesis_direction"])

    source_expiry = expiry
    failure_reasons: List[str] = []
    long_leg = short_leg = None
    long_px = short_px = net = width = reward_risk = np.nan
    for expiry_try in rescue_expiry_attempts(hot, ticker, cp, source_expiry):
        row_try = row.copy()
        row_try["expiry"] = expiry_try
        row_try["dte"] = max((expiry_try - ASOF).days, 1)
        chain = get_hot_chain(hot, ticker, expiry_try, cp)
        if chain.empty:
            failure_reasons.append("missing hot chain")
            continue
        ll = pick_long_leg(chain, whale_strike)
        if ll is None:
            failure_reasons.append("missing long leg")
            continue
        sl = choose_best_fire_short(chain, ll, row_try)
        if sl is None:
            failure_reasons.append("no liquid short leg")
            continue
        lp = conservative_long_price(ll)
        sp = conservative_short_price(sl)
        if not (pd.notna(lp) and pd.notna(sp)):
            failure_reasons.append("missing executable quotes")
            continue
        nt = float(lp) - float(sp)
        wd = abs(float(ll["strike"]) - float(sl["strike"]))
        if nt <= 0.0 or wd <= 0.0:
            failure_reasons.append("invalid debit geometry")
            continue
        rr = (wd - nt) / nt
        if rr <= 0.0:
            failure_reasons.append("negative reward/risk")
            continue
        expiry = expiry_try
        row = row_try
        long_leg, short_leg = ll, sl
        long_px, short_px, net, width, reward_risk = lp, sp, nt, wd, rr
        break
    if long_leg is None or short_leg is None or not pd.notna(reward_risk):
        reason = failure_reasons[-1] if failure_reasons else "no executable debit geometry"
        return BuildResult(None, reason)

    close = float(row["close"]) if pd.notna(row.get("close")) else float(row["underlying_price"])
    iv = float(row["implied_volatility"]) if pd.notna(row.get("implied_volatility")) else 0.30
    pop = compute_pop_debit(close, iv, int(row["dte"]), float(long_leg["strike"]), net, direction)
    ev_ml = pure_ev_ml(pop, reward_risk)
    long_oictx = oi_context(oi_prev, oi_curr, ticker, expiry, cp, float(long_leg["strike"]))
    long_oi = float(long_leg.get("open_interest", 0.0) or 0.0)
    short_oi = float(short_leg.get("open_interest", 0.0) or 0.0)
    raw_conv = conviction_raw(float(row["premium"]), row, long_oictx, long_oi, short_oi, "debit")
    conviction = conviction_pct(raw_conv, "debit_vertical")

    er_days = compute_er_days(row)
    notices: List[str] = []
    if bool(row.get("is_family_flex", False)):
        notices.append(f"family-flex from {row.get('source_seed_family', 'source')}")
    if bool(row.get("minority_flow", False)):
        notices.append("minority-flow")
    if bool(row.get("split_flow_watch", False)):
        notices.append("split-flow")
    if bool(row.get("neutral_conflict", False)):
        notices.append("same-ticker mixed bias")
    if er_days is not None and 0 <= er_days <= 14:
        notices.append(f"earnings in {er_days}d")
    if expiry != source_expiry:
        notices.append(f"expiry rescued from {source_expiry} to {expiry}")
    if not math.isclose(float(long_leg["strike"]), whale_strike, rel_tol=0.0, abs_tol=1e-9):
        notices.append(f"long adjusted from {whale_strike:g}")
    target_short = float(long_leg["strike"]) - float(row["width"]) if direction == "bear" else float(long_leg["strike"]) + float(row["width"])
    if not math.isclose(float(short_leg["strike"]), target_short, rel_tol=0.0, abs_tol=1e-9):
        notices.append(f"short adjusted to {float(short_leg['strike']):g}")
    oi_change = long_oictx.get("oi_change")
    if oi_change is not None:
        if oi_change >= 0.15:
            notices.append(f"OI +{oi_change*100:.1f}%")
        elif oi_change <= -0.10:
            notices.append(f"OI {oi_change*100:.1f}%")
    if long_oi < 100 or short_oi < 100:
        notices.append("thin OI")

    base_size = size_bucket(float(ev_ml), "debit_vertical")
    size = apply_event_size_cap(base_size, er_days)
    if size != base_size and er_days is not None:
        notices.append("size capped for earnings window")

    record = {
        "Ticker": ticker,
        "Action": "🔥🟥 BUY" if direction == "bear" else "🔥🟦 BUY",
        "Buy leg": format_leg(ticker, expiry, float(long_leg["strike"]), cp, "Buy"),
        "Sell leg": format_leg(ticker, expiry, float(short_leg["strike"]), cp, "Sell"),
        "Expiry": str(expiry),
        "Net": f"{net:.2f} debit",
        "EV/ML": float(ev_ml),
        "POP": float(pop),
        "Conviction": conviction,
        "Execution": "Bootstrap",
        "Notice": "; ".join(notices) if notices else "none",
        "Size": size,
        # audit-only
        "seed_family": "FIRE_DEBIT",
        "source_seed_family": row.get("source_seed_family", "FIRE_DEBIT"),
        "is_family_flex": bool(row.get("is_family_flex", False)),
        "translation_type": row.get("translation_type", "native"),
        "structure_kind": "debit_vertical",
        "thesis_direction": direction,
        "cp": cp,
        "premium": float(row["premium"]),
        "marketcap": float(row.get("marketcap", np.nan)) if pd.notna(row.get("marketcap")) else np.nan,
        "issue_type": row.get("issue_type") if pd.notna(row.get("issue_type")) else row.get("equity_type"),
        "combined_bias": float(row.get("combined_bias", 0.0) or 0.0),
        "dominance_strength": float(row.get("dominance_strength", 0.0) or 0.0),
        "whale_bull_premium": float(row.get("bull", 0.0) or 0.0),
        "whale_bear_premium": float(row.get("bear", 0.0) or 0.0),
        "whale_total": float(row.get("whale_total", 0.0) or 0.0),
        "minority_flow": bool(row.get("minority_flow", False)),
        "split_flow_watch": bool(row.get("split_flow_watch", False)),
        "neutral_conflict": bool(row.get("neutral_conflict", False)),
        "shield_bias_mismatch": False,
        "shield_anchor": False,
        "er_days": er_days,
        "reward_risk": float(reward_risk),
        "long_strike": float(long_leg["strike"]),
        "short_strike": float(short_leg["strike"]),
        "long_oi": long_oi,
        "short_oi": short_oi,
        "actual_net": float(net),
        "conviction_raw": float(raw_conv),
        "close": float(close),
        "underlying_price": float(row.get("underlying_price", close) or close),
        "implied_volatility": float(iv),
        "dte": int(row.get("dte", 0) or 0),
        "target_width": float(row.get("width", np.nan)) if pd.notna(row.get("width")) else np.nan,
        "oictx": long_oictx,
        "row_seed_key": row["seed_key"],
    }
    return BuildResult(record, None)


def build_shield_candidate(row: pd.Series, hot: pd.DataFrame, oi_prev: pd.DataFrame, oi_curr: pd.DataFrame) -> BuildResult:
    ticker = str(row["underlying_symbol"])
    expiry = row["expiry"]
    cp = str(row["cp"])
    direction = str(row["thesis_direction"])
    whale_strike = float(row["strike"])

    source_expiry = expiry
    failure_reasons: List[str] = []
    short_leg = long_leg = None
    short_px = long_px = credit = width = reward_risk = np.nan
    for expiry_try in rescue_expiry_attempts(hot, ticker, cp, source_expiry):
        row_try = row.copy()
        row_try["expiry"] = expiry_try
        row_try["dte"] = max((expiry_try - ASOF).days, 1)
        chain = get_hot_chain(hot, ticker, expiry_try, cp)
        if chain.empty:
            failure_reasons.append("missing hot chain")
            continue
        sl = pick_credit_short_leg(chain, whale_strike, direction)
        if sl is None:
            failure_reasons.append("missing short leg")
            continue
        ll = choose_best_credit_wing(chain, sl, row_try)
        if ll is None:
            failure_reasons.append("no protective long wing")
            continue
        sp = conservative_short_price(sl)
        lp = conservative_long_price(ll)
        if not (pd.notna(sp) and pd.notna(lp)):
            failure_reasons.append("missing executable quotes")
            continue
        cr = float(sp) - float(lp)
        wd = abs(float(ll["strike"]) - float(sl["strike"]))
        if cr <= 0.0 or wd <= cr:
            failure_reasons.append("invalid credit geometry")
            continue
        rr = cr / (wd - cr)
        if rr <= 0.0:
            failure_reasons.append("negative reward/risk")
            continue
        expiry = expiry_try
        row = row_try
        short_leg, long_leg = sl, ll
        short_px, long_px, credit, width, reward_risk = sp, lp, cr, wd, rr
        break
    if short_leg is None or long_leg is None or not pd.notna(reward_risk):
        reason = failure_reasons[-1] if failure_reasons else "no executable credit geometry"
        return BuildResult(None, reason)

    close = float(row["close"]) if pd.notna(row.get("close")) else float(row["underlying_price"])
    iv = float(row["implied_volatility"]) if pd.notna(row.get("implied_volatility")) else 0.30
    pop = compute_pop_credit(close, iv, int(row["dte"]), float(short_leg["strike"]), credit, direction)
    ev_ml = pure_ev_ml(pop, reward_risk)
    short_oictx = oi_context(oi_prev, oi_curr, ticker, expiry, cp, float(short_leg["strike"]))
    long_oi = float(long_leg.get("open_interest", 0.0) or 0.0)
    short_oi = float(short_leg.get("open_interest", 0.0) or 0.0)
    raw_conv = conviction_raw(float(row["premium"]), row, short_oictx, long_oi, short_oi, "credit")
    conviction = conviction_pct(raw_conv, "credit_vertical")
    shield_anchor = shield_anchor_ok(row, short_leg, long_leg, short_oictx)

    er_days = compute_er_days(row)
    notices: List[str] = []
    if bool(row.get("is_family_flex", False)):
        notices.append(f"family-flex from {row.get('source_seed_family', 'source')}")
    if bool(row.get("minority_flow", False)):
        notices.append("minority-flow")
    if bool(row.get("split_flow_watch", False)):
        notices.append("split-flow")
    if bool(row.get("shield_bias_mismatch", False)):
        notices.append("same-ticker bias mismatch")
    if er_days is not None and 0 <= er_days <= 14:
        notices.append(f"earnings in {er_days}d")
    if expiry != source_expiry:
        notices.append(f"expiry rescued from {source_expiry} to {expiry}")
    if not math.isclose(float(short_leg["strike"]), whale_strike, rel_tol=0.0, abs_tol=1e-9):
        notices.append(f"short adjusted from {whale_strike:g}")
    target_long = float(short_leg["strike"]) + float(row["width"]) if direction == "bear" else float(short_leg["strike"]) - float(row["width"])
    if not math.isclose(float(long_leg["strike"]), target_long, rel_tol=0.0, abs_tol=1e-9):
        notices.append(f"wing adjusted to {float(long_leg['strike']):g}")
    oi_change = short_oictx.get("oi_change")
    if oi_change is not None:
        if oi_change >= 0.15:
            notices.append(f"OI +{oi_change*100:.1f}%")
        elif oi_change <= -0.10:
            notices.append(f"OI {oi_change*100:.1f}%")
    if long_oi < 100 or short_oi < 100:
        notices.append("thin OI")
    if shield_anchor:
        notices.append("file-native SHIELD anchor")
    else:
        notices.append("SHIELD anchor failed")

    base_size = size_bucket(float(ev_ml), "credit_vertical")
    size = apply_event_size_cap(base_size, er_days)
    if size != base_size and er_days is not None:
        notices.append("size capped for earnings window")

    record = {
        "Ticker": ticker,
        "Action": "🛡️🟥 SELL" if direction == "bear" else "🛡️🟦 SELL",
        "Buy leg": format_leg(ticker, expiry, float(long_leg["strike"]), cp, "Buy"),
        "Sell leg": format_leg(ticker, expiry, float(short_leg["strike"]), cp, "Sell"),
        "Expiry": str(expiry),
        "Net": f"{credit:.2f} credit",
        "EV/ML": float(ev_ml),
        "POP": float(pop),
        "Conviction": conviction,
        "Execution": "Bootstrap",
        "Notice": "; ".join(notices) if notices else "none",
        "Size": size,
        # audit-only
        "seed_family": "SHIELD_CREDIT",
        "source_seed_family": row.get("source_seed_family", "SHIELD_CREDIT"),
        "is_family_flex": bool(row.get("is_family_flex", False)),
        "translation_type": row.get("translation_type", "native"),
        "structure_kind": "credit_vertical",
        "thesis_direction": direction,
        "cp": cp,
        "premium": float(row["premium"]),
        "marketcap": float(row.get("marketcap", np.nan)) if pd.notna(row.get("marketcap")) else np.nan,
        "issue_type": row.get("issue_type") if pd.notna(row.get("issue_type")) else row.get("equity_type"),
        "combined_bias": float(row.get("combined_bias", 0.0) or 0.0),
        "dominance_strength": float(row.get("dominance_strength", 0.0) or 0.0),
        "whale_bull_premium": float(row.get("bull", 0.0) or 0.0),
        "whale_bear_premium": float(row.get("bear", 0.0) or 0.0),
        "whale_total": float(row.get("whale_total", 0.0) or 0.0),
        "minority_flow": bool(row.get("minority_flow", False)),
        "split_flow_watch": bool(row.get("split_flow_watch", False)),
        "neutral_conflict": False,
        "shield_bias_mismatch": bool(row.get("shield_bias_mismatch", False)),
        "shield_anchor": bool(shield_anchor),
        "er_days": er_days,
        "reward_risk": float(reward_risk),
        "long_strike": float(long_leg["strike"]),
        "short_strike": float(short_leg["strike"]),
        "long_oi": long_oi,
        "short_oi": short_oi,
        "actual_net": float(credit),
        "conviction_raw": float(raw_conv),
        "close": float(close),
        "underlying_price": float(row.get("underlying_price", close) or close),
        "implied_volatility": float(iv),
        "dte": int(row.get("dte", 0) or 0),
        "target_width": float(row.get("width", np.nan)) if pd.notna(row.get("width")) else np.nan,
        "oictx": short_oictx,
        "row_seed_key": row["seed_key"],
    }
    return BuildResult(record, None)


def build_condors(credit_rows: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if credit_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    pair_attempts: List[Dict[str, Any]] = []
    built_rows: List[Dict[str, Any]] = []
    for (ticker, expiry), grp in credit_rows.groupby(["Ticker", "Expiry"]):
        calls = grp[(grp["cp"] == "C") & (grp["shield_anchor"])].copy()
        puts = grp[(grp["cp"] == "P") & (grp["shield_anchor"])].copy()
        if calls.empty or puts.empty:
            if not grp.empty:
                pair_attempts.append({
                    "Ticker": ticker,
                    "Expiry": expiry,
                    "status": "missing opposite SHIELD side",
                    "call_rows": int(len(calls)),
                    "put_rows": int(len(puts)),
                })
            continue
        best_row: Optional[Dict[str, Any]] = None
        best_ev = -1e18
        best_conv = -1
        for _, c in calls.iterrows():
            for _, p in puts.iterrows():
                if float(p["short_strike"]) >= float(c["short_strike"]):
                    pair_attempts.append({
                        "Ticker": ticker,
                        "Expiry": expiry,
                        "status": "crossed shorts",
                        "call_short": float(c["short_strike"]),
                        "put_short": float(p["short_strike"]),
                    })
                    continue
                credit_total = float(c["actual_net"]) + float(p["actual_net"])
                max_loss = max(abs(float(c["long_strike"]) - float(c["short_strike"])), abs(float(p["long_strike"]) - float(p["short_strike"]))) - credit_total
                if credit_total <= 0.0 or max_loss <= 0.0:
                    pair_attempts.append({
                        "Ticker": ticker,
                        "Expiry": expiry,
                        "status": "invalid condor credit geometry",
                        "credit_total": credit_total,
                        "max_loss": max_loss,
                    })
                    continue
                rr = credit_total / max_loss
                close = float(c.get("close", np.nan)) if pd.notna(c.get("close")) else np.nan
                if not pd.notna(close):
                    close = float(p.get("close", np.nan)) if pd.notna(p.get("close")) else np.nan
                if not pd.notna(close):
                    close = float(c.get("underlying_price", np.nan)) if pd.notna(c.get("underlying_price")) else np.nan
                if not pd.notna(close):
                    close = float(p.get("underlying_price", np.nan)) if pd.notna(p.get("underlying_price")) else np.nan
                if not pd.notna(close):
                    pair_attempts.append({
                        "Ticker": ticker,
                        "Expiry": expiry,
                        "status": "missing underlying reference",
                    })
                    continue
                iv = np.nanmean([float(c.get("implied_volatility", np.nan)), float(p.get("implied_volatility", np.nan))])
                if not pd.notna(iv):
                    iv = 0.30
                dte = int(min(float(c.get("dte", 0) or 0), float(p.get("dte", 0) or 0)))
                if dte <= 0:
                    dte = max(int(c.get("er_days", 0) or 0), 1)
                lower = float(p["short_strike"]) - credit_total
                upper = float(c["short_strike"]) + credit_total
                pop = compute_pop_condor(float(close), float(iv if pd.notna(iv) else 0.30), dte, lower, upper)
                ev_ml = pure_ev_ml(pop, rr)
                neutrality = 1.0 - min(abs(float(c.get("combined_bias", 0.0) or 0.0)) / 0.20, 1.0)
                raw_conv = conviction_raw_condor(c, p, neutrality)
                conviction = conviction_pct(raw_conv, "iron_condor")
                er_days = c["er_days"] if pd.notna(c["er_days"]) else p["er_days"]
                notices: List[str] = ["paired SHIELD iron condor", "file-native SHIELD anchor"]
                if er_days is not None and pd.notna(er_days) and 0 <= int(er_days) <= 14:
                    notices.append(f"earnings in {int(er_days)}d")
                if float(c["combined_bias"]) != 0.0:
                    notices.append(f"neutral bias {float(c['combined_bias']):+.2f}")
                if float(c["long_strike"]) - float(c["short_strike"]) != float(c.get("width", abs(float(c['long_strike']) - float(c['short_strike'])))):
                    pass
                base_size = size_bucket(float(ev_ml), "iron_condor")
                size = apply_event_size_cap(base_size, int(er_days) if er_days is not None and pd.notna(er_days) else None)
                if size != base_size and er_days is not None and pd.notna(er_days):
                    notices.append("size capped for earnings window")
                row = {
                    "Ticker": ticker,
                    "Action": "🛡️⚪ SELL",
                    "Buy leg": (
                        f"{format_leg(ticker, pd.to_datetime(expiry).date(), float(p['long_strike']), 'P', 'Buy')} + "
                        f"{format_leg(ticker, pd.to_datetime(expiry).date(), float(c['long_strike']), 'C', 'Buy')}"
                    ),
                    "Sell leg": (
                        f"{format_leg(ticker, pd.to_datetime(expiry).date(), float(p['short_strike']), 'P', 'Sell')} + "
                        f"{format_leg(ticker, pd.to_datetime(expiry).date(), float(c['short_strike']), 'C', 'Sell')}"
                    ),
                    "Expiry": expiry,
                    "Net": f"{credit_total:.2f} credit",
                    "EV/ML": float(ev_ml),
                    "POP": float(pop),
                    "Conviction": conviction,
                    "Execution": "Bootstrap",
                    "Notice": "; ".join(notices),
                    "Size": size,
                    # audit-only
                    "seed_family": "SHIELD_CREDIT",
                    "structure_kind": "iron_condor",
                    "thesis_direction": "neutral",
                    "cp": "X",
                    "premium": float(c["premium"]) + float(p["premium"]),
                    "marketcap": c["marketcap"] if pd.notna(c["marketcap"]) else p["marketcap"],
                    "issue_type": c["issue_type"] if pd.notna(c["issue_type"]) else p["issue_type"],
                    "combined_bias": float(c["combined_bias"]),
                    "dominance_strength": abs(float(c["combined_bias"])),
                    "whale_bull_premium": float(c.get("whale_bull_premium", 0.0) or 0.0),
                    "whale_bear_premium": float(c.get("whale_bear_premium", 0.0) or 0.0),
                    "whale_total": float(c.get("whale_total", 0.0) or 0.0),
                    "minority_flow": False,
                    "split_flow_watch": False,
                    "neutral_conflict": False,
                    "shield_bias_mismatch": False,
                    "shield_anchor": True,
                    "er_days": er_days,
                    "reward_risk": float(rr),
                    "long_strike": np.nan,
                    "short_strike": np.nan,
                    "long_oi": float(c["long_oi"]) + float(p["long_oi"]),
                    "short_oi": float(c["short_oi"]) + float(p["short_oi"]),
                    "actual_net": float(credit_total),
                    "conviction_raw": float(raw_conv),
                    "close": float(close),
                    "underlying_price": float(close),
                    "implied_volatility": float(iv),
                    "dte": int(dte),
                    "target_width": np.nan,
                    "oictx": {"micro_ratio": 0.5 * (float(c["oictx"].get("bid_ask_ratio", 1.0) if isinstance(c["oictx"], dict) else 1.0) + float(p["oictx"].get("bid_ask_ratio", 1.0) if isinstance(p["oictx"], dict) else 1.0))},
                    "row_seed_key": f"{c['row_seed_key']}||{p['row_seed_key']}",
                }
                if (ev_ml > best_ev + 1e-12) or (abs(ev_ml - best_ev) <= 1e-12 and conviction > best_conv):
                    best_ev = ev_ml
                    best_conv = conviction
                    best_row = row
                pair_attempts.append({
                    "Ticker": ticker,
                    "Expiry": expiry,
                    "status": "paired",
                    "call_short": float(c["short_strike"]),
                    "put_short": float(p["short_strike"]),
                    "ev_ml": float(ev_ml),
                    "pop": float(pop),
                })
        if best_row is not None:
            built_rows.append(best_row)
    return pd.DataFrame(built_rows), pd.DataFrame(pair_attempts)


def table_round(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    def maybe_round(x: Any) -> Any:
        if pd.isna(x):
            return x
        try:
            return round(float(x), 3)
        except (TypeError, ValueError):
            return x
    if "EV/ML" in out.columns:
        out["EV/ML"] = out["EV/ML"].map(maybe_round)
    if "POP" in out.columns:
        out["POP"] = out["POP"].map(maybe_round)
    return out


def build_watch_rows(
    built: pd.DataFrame,
    primary: pd.DataFrame,
    rejected: pd.DataFrame,
    condor_attempts: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    primary_keys = set(zip(primary["Ticker"], primary["Action"], primary["Buy leg"], primary["Sell leg"])) if not primary.empty else set()

    shield_watch = built[
        (built["structure_kind"].isin(["credit_vertical", "iron_condor"]))
        & (~built.apply(lambda r: (r["Ticker"], r["Action"], r["Buy leg"], r["Sell leg"]) in primary_keys, axis=1))
    ].copy()
    if not shield_watch.empty:
        shield_watch["watch_rank"] = (
            shield_watch["shield_anchor"].astype(int) * 1000
            + shield_watch["EV/ML"].fillna(-999.0) * 100
            + shield_watch["Conviction"].fillna(0)
        )
        shield_watch = shield_watch.sort_values(["watch_rank", "premium"], ascending=[False, False]).head(12)
        for _, row in shield_watch.iterrows():
            rows.append({
                "Ticker": row["Ticker"],
                "Action": "🟧 WATCH",
                "Buy leg": row["Buy leg"],
                "Sell leg": row["Sell leg"],
                "Expiry": row["Expiry"],
                "Net": row["Net"],
                "EV/ML": row["EV/ML"],
                "POP": row["POP"],
                "Conviction": row["Conviction"],
                "Execution": "Watch",
                "Notice": row["Notice"],
                "Size": "Watch",
            })

    blocked = built[
        (~built["is_primary_eligible"])
        & (~built["structure_kind"].isin(["credit_vertical", "iron_condor"]))
        & (built["EV/ML"].notna())
    ].copy()
    if not blocked.empty:
        blocked = blocked.sort_values(["EV/ML", "Conviction", "premium"], ascending=[False, False, False]).head(20)
        for _, row in blocked.iterrows():
            rows.append({
                "Ticker": row["Ticker"],
                "Action": "🟧 WATCH",
                "Buy leg": row["Buy leg"],
                "Sell leg": row["Sell leg"],
                "Expiry": row["Expiry"],
                "Net": row["Net"],
                "EV/ML": row["EV/ML"],
                "POP": row["POP"],
                "Conviction": row["Conviction"],
                "Execution": "Watch",
                "Notice": row["Notice"],
                "Size": "Watch",
            })

    if not rejected.empty:
        rej = rejected.sort_values(["premium"], ascending=[False]).drop_duplicates(["underlying_symbol", "expiry", "cp", "strike", "reason"]).head(20)
        for _, row in rej.iterrows():
            strike = float(row["strike"])
            strike_txt = str(int(strike)) if strike.is_integer() else str(strike)
            rows.append({
                "Ticker": row["underlying_symbol"],
                "Action": "🟧 WATCH",
                "Buy leg": f"Seed {row['underlying_symbol']} {row['expiry']} {strike_txt}{row['cp']}",
                "Sell leg": row["reason"],
                "Expiry": str(row["expiry"]),
                "Net": "n/a",
                "EV/ML": "n/a",
                "POP": "n/a",
                "Conviction": int(round(40 + 50 * abs(float(row.get("combined_bias", 0.0) or 0.0)))),
                "Execution": "Watch",
                "Notice": row["reason"],
                "Size": "Watch",
            })

    if not condor_attempts.empty:
        failed = condor_attempts[condor_attempts["status"] != "paired"].drop_duplicates(["Ticker", "Expiry", "status"]).head(8)
        for _, row in failed.iterrows():
            rows.append({
                "Ticker": row["Ticker"],
                "Action": "🟧 WATCH",
                "Buy leg": "SHIELD condor pairing",
                "Sell leg": row["status"],
                "Expiry": row["Expiry"],
                "Net": "n/a",
                "EV/ML": "n/a",
                "POP": "n/a",
                "Conviction": 50,
                "Execution": "Watch",
                "Notice": row["status"],
                "Size": "Watch",
            })

    if not rows:
        return pd.DataFrame(columns=PRIMARY_COLS)
    out = pd.DataFrame(rows)[PRIMARY_COLS].drop_duplicates()
    return out


def build_catalyst_watch_rows(cand: pd.DataFrame, max_rows: int = 15) -> pd.DataFrame:
    """Force high-premium near-earnings names into the watch report.

    This fixes the practical failure mode where a major catalyst name is blocked
    by the earnings gate and then disappears from the user-facing analysis.
    """
    if cand.empty:
        return pd.DataFrame(columns=PRIMARY_COLS)
    df = cand.copy()
    if "er_days" not in df.columns:
        df["er_days"] = df.apply(compute_er_days, axis=1)
    df = df[df["er_days"].apply(lambda x: pd.notna(x) and 0 <= int(x) <= 6)].copy()
    if df.empty:
        return pd.DataFrame(columns=PRIMARY_COLS)
    grp = df.groupby("underlying_symbol", as_index=False).agg(
        premium=("premium", "sum"),
        count=("seed_key", "count"),
        er_days=("er_days", "min"),
        whale_bull=("bull", "max"),
        whale_bear=("bear", "max"),
        combined_bias=("combined_bias", "mean"),
    ).sort_values("premium", ascending=False).head(max_rows)
    rows: List[Dict[str, Any]] = []
    for _, r in grp.iterrows():
        bias = float(r.get("combined_bias", 0.0) or 0.0)
        raw_dir = "bull" if bias > 0.10 else ("bear" if bias < -0.10 else "split")
        premium = float(r["premium"] or 0.0)
        conv = int(round(min(95, 55 + 10 * math.log10(max(premium, 1.0) / 100000.0))))
        rows.append({
            "Ticker": str(r["underlying_symbol"]),
            "Action": "🟧 CATALYST WATCH",
            "Buy leg": "blocked earnings catalyst",
            "Sell leg": "run live-chain post-event family-flex scan",
            "Expiry": "n/a",
            "Net": "n/a",
            "EV/ML": "n/a",
            "POP": "n/a",
            "Conviction": max(50, min(conv, 95)),
            "Execution": "Watch",
            "Notice": f"earnings in {int(r['er_days'])}d; raw flow {raw_dir}; total premium ${premium:,.0f}; official BUY/SELL blocked",
            "Size": "Watch",
        })
    return pd.DataFrame(rows)[PRIMARY_COLS]



def top_symbol_gap_table(top_symbols: pd.DataFrame, raw_seed_rows: pd.DataFrame, source_mode: str) -> pd.DataFrame:
    if top_symbols is None or top_symbols.empty:
        return pd.DataFrame()
    top = top_symbols.copy()
    if "total_premium" not in top.columns:
        return pd.DataFrame()
    if raw_seed_rows.empty:
        visible = pd.DataFrame(columns=["underlying_symbol", "visible_premium", "visible_count"])
    else:
        visible = raw_seed_rows.groupby("underlying_symbol", as_index=False).agg(
            visible_premium=("premium", "sum"), visible_count=("seed_key", "count")
        )
    out = top.merge(visible, on="underlying_symbol", how="left")
    out["visible_premium"] = out["visible_premium"].fillna(0.0)
    out["visible_count"] = out["visible_count"].fillna(0).astype(int)
    out["visible_share"] = np.where(out["total_premium"].fillna(0) > 0, out["visible_premium"] / out["total_premium"], np.nan)
    out["source_mode"] = source_mode
    out["gap_flag"] = False
    if source_mode == "markdown_top200":
        out["gap_flag"] = (out["total_premium"].fillna(0) >= 1_000_000) & ((out["visible_share"].fillna(0) < 0.25) | (out["visible_count"] == 0))
    return out.sort_values("total_premium", ascending=False)


def build_top_symbol_gap_watch_rows(gap: pd.DataFrame, max_rows: int = 12) -> pd.DataFrame:
    if gap is None or gap.empty:
        return pd.DataFrame(columns=PRIMARY_COLS)
    df = gap[gap["gap_flag"]].copy().sort_values("total_premium", ascending=False).head(max_rows)
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        share = r.get("visible_share", np.nan)
        share_txt = "n/a" if pd.isna(share) else f"{float(share)*100:.1f}%"
        rows.append({
            "Ticker": str(r["underlying_symbol"]),
            "Action": "🟧 TOP-SYMBOL GAP",
            "Buy leg": "full-source flow required",
            "Sell leg": "Top-200 slice under-represents ticker",
            "Expiry": "n/a",
            "Net": "n/a",
            "EV/ML": "n/a",
            "POP": "n/a",
            "Conviction": 70,
            "Execution": "Watch",
            "Notice": f"full premium ${float(r['total_premium']):,.0f}; visible Top-200 share {share_txt}; do not let this ticker disappear",
            "Size": "Watch",
        })
    return pd.DataFrame(rows)[PRIMARY_COLS] if rows else pd.DataFrame(columns=PRIMARY_COLS)


def primary_key_set(primary: pd.DataFrame) -> set:
    if primary is None or primary.empty:
        return set()
    return set(zip(primary["Ticker"], primary["Action"], primary["Buy leg"], primary["Sell leg"]))


def build_blocked_positive_ev_table(built: pd.DataFrame, primary: pd.DataFrame, max_rows: int = 50) -> pd.DataFrame:
    if built.empty:
        return pd.DataFrame()
    pkeys = primary_key_set(primary)
    df = built[(built["EV/ML"].fillna(-999) > 0) & (~built.apply(lambda r: (r["Ticker"], r["Action"], r["Buy leg"], r["Sell leg"]) in pkeys, axis=1))].copy()
    if df.empty:
        return pd.DataFrame()
    def reason(r: pd.Series) -> str:
        reasons = []
        for flag, label in [
            ("inside_event_block", "event block"), ("minority_flow", "minority flow"),
            ("split_flow_watch", "split flow"), ("neutral_conflict", "neutral conflict"),
            ("shield_bias_mismatch", "SHIELD bias mismatch"),
        ]:
            if bool(r.get(flag, False)):
                reasons.append(label)
        if str(r.get("liquidity_tier", "")) in {"SMALL_WATCH", "UNKNOWN_WATCH"}:
            reasons.append(str(r.get("liquidity_tier")).lower())
        if not bool(r.get("has_executable_size", True)):
            reasons.append("Size=None")
        if r.get("structure_kind") == "credit_vertical" and not bool(r.get("shield_anchor", False)):
            reasons.append("SHIELD anchor failed")
        return "; ".join(reasons) if reasons else "alternate/non-primary"
    df["Block reason"] = df.apply(reason, axis=1)
    cols = PRIMARY_COLS + ["structure_kind", "liquidity_tier", "Block reason", "premium", "marketcap", "er_days", "combined_bias", "is_family_flex", "translation_type"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return table_round(df.sort_values(["EV/ML", "Conviction", "premium"], ascending=[False, False, False]).head(max_rows)[cols])


def build_alternates_table(built: pd.DataFrame, primary: pd.DataFrame, per_ticker: int = 3, max_rows: int = 80) -> pd.DataFrame:
    if built.empty:
        return pd.DataFrame()
    pkeys = primary_key_set(primary)
    df = built[~built.apply(lambda r: (r["Ticker"], r["Action"], r["Buy leg"], r["Sell leg"]) in pkeys, axis=1)].copy()
    df = df[df["EV/ML"].notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df = df.sort_values(["Ticker", "EV/ML", "Conviction"], ascending=[True, False, False]).groupby("Ticker", group_keys=False).head(per_ticker)
    cols = PRIMARY_COLS + ["structure_kind", "liquidity_tier", "is_family_flex", "translation_type", "premium", "combined_bias"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return table_round(df.sort_values(["EV/ML", "Conviction", "premium"], ascending=[False, False, False]).head(max_rows)[cols])


def build_etf_lane_table(built: pd.DataFrame, max_rows: int = 25) -> pd.DataFrame:
    if built.empty or "liquidity_tier" not in built.columns:
        return pd.DataFrame(columns=PRIMARY_COLS)
    df = built[
        (built["liquidity_tier"].eq("ETF_INDEX"))
        & (built["EV/ML"].fillna(-999) > 0)
        & (built["has_executable_size"].fillna(False))
        & (~built["inside_event_block"].fillna(False))
        & (~built["minority_flow"].fillna(False))
        & (~built["split_flow_watch"].fillna(False))
    ].copy()
    if df.empty:
        return pd.DataFrame(columns=PRIMARY_COLS)
    # ETF/index products are a separate lane, not common-stock primary BUY/SELL.
    # Preserve directional emoji but remove BUY labeling to avoid implying primary publication.
    df["Action"] = df["Action"].astype(str).map(lambda x: "🟧 ETF " + x.replace("BUY", "WATCH"))
    df["Execution"] = "Watch"
    df["Size"] = "Watch"
    df["Notice"] = df["Notice"].map(lambda x: add_notice_text(x, "separate ETF/index lane; not primary BUY/SELL"))
    return table_round(df.sort_values(["EV/ML", "Conviction", "premium"], ascending=[False, False, False]).head(max_rows)[PRIMARY_COLS])


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    return df.to_markdown(index=False)


def run(base_dir: Path, out_dir: Path, asof: Optional[date] = None, allow_project_version_mismatch: bool = False, project_dir: Optional[Path] = None, allow_markdown_seed_fallback: bool = False, use_next_day_oi: bool = False) -> None:
    global ASOF
    ASOF, required_paths = resolve_input_paths(base_dir, asof, use_next_day_oi=use_next_day_oi)
    scan_date = ASOF.isoformat()
    project_alignment = audit_canonical_executable_sync(base_dir, out_dir, project_dir=project_dir)
    if project_alignment.get("status") == "FAIL" and not allow_project_version_mismatch:
        roots_txt = ", ".join(project_alignment.get("search_roots", []))
        missing_txt = ", ".join(project_alignment.get("missing_files", []))
        mismatch_txt = ", ".join(project_alignment.get("mismatch_files", []))
        raise RuntimeError(
            "Audit gate zero failed: canonical executable sync mismatch. "
            + project_alignment.get("note", "Replace all five canonical project files from one bundle.")
            + f" Search roots: {roots_txt}."
            + (f" Missing: {missing_txt}." if missing_txt else "")
            + (f" Mismatch: {mismatch_txt}." if mismatch_txt else "")
        )
    whale_path, hot_path, oi_prev_path, oi_curr_path, screen_path, dp_path = required_paths
    schwab = load_schwab_context(base_dir, ASOF)

    whale = parse_flow_source(whale_path, ASOF)
    source_mode = whale.attrs.get("source_mode", "unknown")
    top_symbols_source = whale.attrs.get("top_symbols", pd.DataFrame())
    full_source_bias = whale.attrs.get("full_source_bias", None)
    full_source_summary = whale.attrs.get("full_source_summary", {"source_mode": source_mode, "source_file": whale_path.name})
    hot = normalize_hot(read_zip_csv(hot_path))
    oi_prev = normalize_oi(read_zip_csv(oi_prev_path))
    oi_curr = normalize_oi(read_zip_csv(oi_curr_path))
    screen = normalize_screen(read_zip_csv(screen_path))
    dp = normalize_dp(read_zip_csv(dp_path))

    raw_seed_rows = filter_candidate_seed_rows(whale)
    whale, raw_seed_rows, source_mode, top_symbols_source, full_source_bias, full_source_summary = maybe_fallback_to_whale_markdown(
        base_dir, ASOF, whale_path, whale, raw_seed_rows, source_mode, top_symbols_source, full_source_bias, full_source_summary,
        allow_markdown_seed_fallback=allow_markdown_seed_fallback,
    )
    raw_seed_rows = ensure_seed_schema(raw_seed_rows)
    if raw_seed_rows.empty:
        raise RuntimeError(
            "No candidate seed rows were selected from the flow source. "
            "This means the full bot/EOD CSV stream produced zero rulebook-like rows after raw-bot inference "
            "(side=>family, width ladder, pct_width=price/width), or all rows failed the DTE/OI/width/distance filters. "
            f"Source mode={source_mode}; flow file={whale_path.name}; schema preview: {preview_flow_source_schema(whale_path)}. "
            "This build does NOT silently use whale markdown when a full bot ZIP is present. "
            "Run with --allow-markdown-seed-fallback only for emergency/development fallback, or run --diagnose-full-source to see the schema/filter breakdown. "
            "The right fix is mapping normalize_flow_candidates aliases for the actual bot CSV column names."
        )
    dup_counts = raw_seed_rows.groupby("seed_key", as_index=False).size().rename(columns={"size": "dup_count"})
    raw_seed_rows = raw_seed_rows.merge(dup_counts, on="seed_key", how="left")
    base_cand = dedupe_seed_rows(raw_seed_rows)

    # Same-ticker bias is computed only from native file rows. Family-flex
    # translations are derived candidates and must not inflate direction counts.
    bias = build_bias_tables(base_cand, screen, full_source_bias=full_source_bias)
    base_cand = base_cand.merge(bias, on="underlying_symbol", how="left", suffixes=("", "_bias"))
    # Raw full-source seeds may already contain placeholder columns such as
    # marketcap=np.nan. After merging screener bias, coalesce the real screener
    # values back into the canonical names so liquidity tiers do not become
    # UNKNOWN_WATCH for every full-source candidate.
    for _c in ["marketcap", "issue_type", "close", "next_earnings_date", "implied_move_perc"]:
        _bc = f"{_c}_bias"
        if _bc in base_cand.columns:
            if _c not in base_cand.columns:
                base_cand[_c] = base_cand[_bc]
            else:
                _empty = base_cand[_c].isna() | base_cand[_c].astype(str).str.strip().isin(["", "nan", "NaN", "None"])
                base_cand.loc[_empty, _c] = base_cand.loc[_empty, _bc]
    base_cand["is_family_flex"] = False
    base_cand["source_seed_family"] = base_cand["seed_family"]
    base_cand["translation_type"] = "native"

    flex_cand = generate_family_flex_seed_rows(base_cand, hot)
    cand = pd.concat([base_cand, flex_cand], ignore_index=True, sort=False) if not flex_cand.empty else base_cand.copy()
    cand["minority_flow"] = cand.apply(minority_flow_block, axis=1)
    cand["split_flow_watch"] = cand.apply(split_flow_watch, axis=1)
    cand["neutral_conflict"] = cand.apply(fire_neutral_conflict, axis=1)
    cand["shield_bias_mismatch"] = cand.apply(shield_bias_mismatch, axis=1)

    built_records: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    for _, row in cand.iterrows():
        if str(row["seed_family"]) == "FIRE_DEBIT":
            result = build_fire_candidate(row, hot, oi_prev, oi_curr)
        elif str(row["seed_family"]) == "SHIELD_CREDIT":
            result = build_shield_candidate(row, hot, oi_prev, oi_curr)
        else:
            continue
        if result.record is not None:
            built_records.append(result.record)
        else:
            rej = row.to_dict()
            rej["reason"] = result.reason
            rejected_rows.append(rej)

    built = pd.DataFrame(built_records)
    rejected = pd.DataFrame(rejected_rows)
    if built.empty:
        raise RuntimeError("No candidates were successfully built from the input files")

    credit_rows = built[(built["structure_kind"] == "credit_vertical") & (built["shield_anchor"])].copy()
    condors, condor_attempts = build_condors(credit_rows)
    if not condors.empty:
        built = pd.concat([built, condors], ignore_index=True, sort=False)

    built = (
        built.sort_values(["Ticker", "Action", "EV/ML", "Conviction"], ascending=[True, True, False, False])
        .drop_duplicates(["Ticker", "Buy leg", "Sell leg"], keep="first")
        .reset_index(drop=True)
    )

    built["focus_ok"] = built.apply(lambda r: issue_type_focus_ok(r.get("issue_type"), str(r["Ticker"]), str(r["seed_family"])), axis=1)
    built["inside_event_block"] = built["er_days"].apply(lambda x: pd.notna(x) and 0 <= int(x) <= 10)
    built["has_executable_size"] = built["Size"].fillna("None").astype(str).str.lower().ne("none")
    built["liquidity_tier"] = built.apply(liquidity_tier, axis=1)
    built["mixed_flow_rescue"] = built.apply(neutral_conflict_rescue, axis=1)
    built = apply_quality_and_tier_notices(built)

    built["is_primary_eligible"] = False
    fire_mask = built["structure_kind"].eq("debit_vertical")
    fire_tier_ok = built.loc[fire_mask, "liquidity_tier"].isin(["MAJOR", "MID_PILOT"])
    fire_clean_bias = (~built.loc[fire_mask, "neutral_conflict"]) & (built.loc[fire_mask, "combined_bias"].abs() >= 0.20)
    fire_mixed_rescue = built.loc[fire_mask, "neutral_conflict"] & built.loc[fire_mask, "mixed_flow_rescue"]
    fire_direction_ok = fire_clean_bias | fire_mixed_rescue
    built.loc[fire_mask, "is_primary_eligible"] = (
        fire_tier_ok
    ) & built.loc[fire_mask, "focus_ok"] & built.loc[fire_mask, "has_executable_size"] & (~built.loc[fire_mask, "minority_flow"]) & (~built.loc[fire_mask, "split_flow_watch"]) & fire_direction_ok & (~built.loc[fire_mask, "inside_event_block"]) & (built.loc[fire_mask, "EV/ML"] > 0)

    shield_mask = built["structure_kind"].eq("credit_vertical")
    shield_tier_ok = built.loc[shield_mask, "liquidity_tier"].isin(["MAJOR", "MID_PILOT"]) | built.loc[shield_mask, "Ticker"].isin(INDEX_SHIELD_ALLOW)
    shield_primary = (
        shield_tier_ok
        & built.loc[shield_mask, "focus_ok"]
        & built.loc[shield_mask, "has_executable_size"]
        & (~built.loc[shield_mask, "minority_flow"])
        & (~built.loc[shield_mask, "split_flow_watch"])
        & (~built.loc[shield_mask, "inside_event_block"])
        & built.loc[shield_mask, "shield_anchor"]
        & (~built.loc[shield_mask, "shield_bias_mismatch"])
        & (built.loc[shield_mask, "EV/ML"] > 0)
    )
    built.loc[shield_mask, "is_primary_eligible"] = shield_primary

    condor_mask = built["structure_kind"].eq("iron_condor")
    condor_tier_ok = built.loc[condor_mask, "liquidity_tier"].isin(["MAJOR", "MID_PILOT"]) | built.loc[condor_mask, "Ticker"].isin(INDEX_SHIELD_ALLOW)
    condor_primary = (
        condor_tier_ok
        & built.loc[condor_mask, "has_executable_size"]
        & (~built.loc[condor_mask, "inside_event_block"])
        & built.loc[condor_mask, "shield_anchor"]
        & (built.loc[condor_mask, "combined_bias"].abs() <= 0.12)
        & (built.loc[condor_mask, "EV/ML"] > 0)
    )
    built.loc[condor_mask, "is_primary_eligible"] = condor_primary

    primary_candidates = built[built["is_primary_eligible"]].copy()
    primary_candidates = (
        primary_candidates.sort_values(["Ticker", "EV/ML", "Conviction"], ascending=[True, False, False])
        .drop_duplicates("Ticker", keep="first")
    )
    primary = primary_candidates.sort_values(["EV/ML", "Conviction"], ascending=[False, False]).head(10)
    primary_table = table_round(primary[PRIMARY_COLS])

    top_gap = top_symbol_gap_table(top_symbols_source, raw_seed_rows, source_mode)
    top_gap_watch = build_top_symbol_gap_watch_rows(top_gap)
    blocked_positive_ev = build_blocked_positive_ev_table(built, primary)
    alternates_table = build_alternates_table(built, primary)
    etf_lane_table = build_etf_lane_table(built)

    catalyst_watch = build_catalyst_watch_rows(base_cand)
    ordinary_watch = build_watch_rows(built, primary, rejected, condor_attempts)
    watch_table = table_round(pd.concat([catalyst_watch, top_gap_watch, ordinary_watch], ignore_index=True, sort=False).drop_duplicates())
    primary_table = apply_schwab_execution_and_notices(primary_table, schwab)
    watch_table = apply_schwab_execution_and_notices(watch_table, schwab)
    etf_lane_table = apply_schwab_execution_and_notices(etf_lane_table, schwab)

    inputs_table = summarize_inputs(required_paths)
    if schwab.get("input_record"):
        inputs_table = pd.concat([inputs_table, pd.DataFrame([schwab["input_record"]])], ignore_index=True)
    dup_summary = raw_seed_rows.groupby("seed_key", as_index=False).size().rename(columns={"size": "count"})
    dup_summary = dup_summary[dup_summary["count"] > 1].copy().sort_values("count", ascending=False)
    if not dup_summary.empty:
        dup_details = raw_seed_rows[[
            "seed_key", "underlying_symbol", "track", "net_type", "expiry", "cp", "strike",
            "price", "width", "premium", "open_interest", "dup_count"
        ]].drop_duplicates().sort_values(["dup_count", "premium"], ascending=[False, False])
        dup_details = dup_details[dup_details["dup_count"] > 1].head(12)
    else:
        dup_details = pd.DataFrame(columns=["seed_key", "underlying_symbol", "track", "net_type", "expiry", "cp", "strike", "price", "width", "premium", "open_interest", "dup_count"])

    fire_raw = int((base_cand["seed_family"] == "FIRE_DEBIT").sum())
    shield_raw = int((base_cand["seed_family"] == "SHIELD_CREDIT").sum())
    family_flex_rows = int(cand["is_family_flex"].fillna(False).astype(bool).sum())
    family_flex_built_rows = int(built["is_family_flex"].fillna(False).astype(bool).sum()) if "is_family_flex" in built.columns else 0
    shield_built = built[built["structure_kind"] == "credit_vertical"].copy()
    shield_anchored = shield_built[shield_built["shield_anchor"]].copy()
    minority_rows = built[built["minority_flow"]].sort_values(["premium", "EV/ML"], ascending=[False, False])
    split_rows = built[built["split_flow_watch"]].sort_values(["premium", "EV/ML"], ascending=[False, False])
    neutral_rows = built[built["neutral_conflict"]].sort_values(["premium", "EV/ML"], ascending=[False, False])
    shield_mismatch_rows = built[built["shield_bias_mismatch"]].sort_values(["premium", "EV/ML"], ascending=[False, False])

    dp_focus = pd.DataFrame()
    if not dp.empty and "ticker" in dp.columns and "premium" in dp.columns:
        dp_focus = dp.groupby("ticker", as_index=False)["premium"].sum().sort_values("premium", ascending=False)
        tickers = primary_table["Ticker"].tolist() + watch_table["Ticker"].tolist()
        dp_focus = dp_focus[dp_focus["ticker"].isin(tickers)]

    audit_json = {
        "engine_version": "3.2.7-full-source-routing-audit",
        "scan_date": scan_date,
        "gex_enabled": False,
        "project_alignment": project_alignment,
        "health_gate": {
            "status": schwab.get("status", "UNKNOWN"),
            "execution_label": schwab.get("execution_label", "Bootstrap"),
            "rows_checked": schwab.get("rows_checked", 0),
            "issue_count": schwab.get("issue_count"),
            "source": schwab.get("path"),
            "note": schwab.get("note"),
        },
        "oi_overlay": {
            "previous_oi_file": oi_prev_path.name,
            "current_oi_file": oi_curr_path.name,
            "uses_next_day_overlay": bool(use_next_day_oi and oi_curr_path.name.startswith(f"chain-oi-changes-{(ASOF + timedelta(days=1)).isoformat()}")),
            "next_day_overlay_requested": bool(use_next_day_oi),
            "mode": "next_day_overlay" if use_next_day_oi else "scan_date_only",
        },
        "input_hashes": inputs_table.to_dict(orient="records"),
        "source_mode": source_mode,
        "full_source_summary": full_source_summary,
        "top_symbol_gap_rows": int(len(top_gap)) if top_gap is not None else 0,
        "top_symbol_gap_flags": int(top_gap["gap_flag"].sum()) if top_gap is not None and not top_gap.empty and "gap_flag" in top_gap.columns else 0,
        "blocked_positive_ev_rows": int(len(blocked_positive_ev)),
        "alternates_rows": int(len(alternates_table)),
        "etf_lane_rows": int(len(etf_lane_table)),
        "mid_pilot_built_rows": int((built.get("liquidity_tier", pd.Series(dtype=str)) == "MID_PILOT").sum()) if "liquidity_tier" in built.columns else 0,
        "mixed_flow_rescued_rows": int(built.get("mixed_flow_rescue", pd.Series(dtype=bool)).fillna(False).sum()) if "mixed_flow_rescue" in built.columns else 0,
        "seed_rows_raw_total": int(len(raw_seed_rows)),
        "seed_rows_deduped_total": int(len(base_cand)),
        "family_flex_seed_rows": family_flex_rows,
        "family_flex_built_rows": family_flex_built_rows,
        "candidate_rows_for_build": int(len(cand)),
        "fire_seed_rows_deduped": fire_raw,
        "shield_seed_rows_deduped": shield_raw,
        "duplicate_seed_count": int(len(dup_summary)),
        "built_rows_total": int(len(built)),
        "built_fire_rows": int((built["structure_kind"] == "debit_vertical").sum()),
        "built_shield_credit_rows": int((built["structure_kind"] == "credit_vertical").sum()),
        "anchored_shield_rows": int(len(shield_anchored)),
        "built_iron_condors": int((built["structure_kind"] == "iron_condor").sum()),
        "condor_pair_attempts": int(len(condor_attempts)),
        "rejected_rows": int(len(rejected)),
        "primary_rows": int(len(primary_table)),
        "watch_rows": int(len(watch_table)),
        "blocked_minority_rows": int(len(minority_rows)),
        "split_flow_watch_rows": int(len(split_rows)),
        "neutral_conflict_rows": int(len(neutral_rows)),
        "shield_bias_mismatch_rows": int(len(shield_mismatch_rows)),
        "top_primary_tickers": primary_table["Ticker"].tolist(),
    }

    generated = datetime.now(timezone.utc).isoformat()
    report_md = "\n".join([
        f"# Options scan report — audited FIRE + SHIELD no-GEX run ({scan_date})",
        "",
        "## Primary inline table",
        "",
        markdown_table(primary_table),
        "",
        "## Watch table",
        "",
        markdown_table(watch_table),
        "",
        "## ETF / index lane",
        "",
        markdown_table(etf_lane_table),
        "",
        "## Blocked positive-EV / alternates summary",
        "",
        f"Blocked positive-EV rows: {len(blocked_positive_ev)}. Alternates rows: {len(alternates_table)}. Top-symbol gap flags: {int(top_gap['gap_flag'].sum()) if top_gap is not None and not top_gap.empty and 'gap_flag' in top_gap.columns else 0}.",
        "",
        f"Health Gate: {schwab.get('status', 'UNKNOWN')} -> {schwab.get('execution_label', 'Bootstrap')}.",
        f"OI handles: previous={oi_prev_path.name}; current={oi_curr_path.name}; mode={'next-day overlay' if use_next_day_oi else 'scan-date only'}.",
        "Automated GEX: disabled.",
        f"FIRE seeds: {fire_raw}. SHIELD seeds: {shield_raw}. Family-flex seeds: {family_flex_rows}. Anchored SHIELD builds: {int(len(shield_anchored))}. Iron condors built: {int((built['structure_kind'] == 'iron_condor').sum())}.",
    ])

    audit_md = f"""# Anu Options Engine deep audit — {scan_date}
Generated: {generated}

## Summary
This payload keeps automated GEX disabled, preserves the FIRE/SHIELD/condor path, and adds v3.2.7 routing fixes: optional streamed full-source bot EOD ingestion, top-symbol gap detection when only Top-200 markdown is available, liquidity-tier routing instead of a hard $80B delete, expanded blocked-positive-EV and alternates outputs, ETF/index lane separation, low-POP convexity notices, and mixed-flow rescue only when OI follow-through independently supports the row.

## Project executable sync
- Status: {project_alignment.get('status')}
- Note: {project_alignment.get('note')}

## Health Gate and OI overlay
- Health Gate: {schwab.get('status', 'UNKNOWN')} -> {schwab.get('execution_label', 'Bootstrap')}
- Health rows checked: {schwab.get('rows_checked', 0)}
- Health issues: {schwab.get('issue_count')}
- Previous OI handle: {oi_prev_path.name}
- Current OI handle: {oi_curr_path.name}
- OI mode: {'next-day overlay' if use_next_day_oi else 'scan-date only'}

## Input files
{markdown_table(inputs_table)}

## Source mode
- Flow source mode: {source_mode}
- Full-source summary: {json.dumps(full_source_summary, indent=2)}

## Top-symbol gap report
{markdown_table(table_round(top_gap.head(25)) if top_gap is not None and not top_gap.empty else pd.DataFrame())}

## Candidate counts
- FIRE deduped native seeds: {fire_raw}
- SHIELD deduped native seeds: {shield_raw}
- Family-flex derived seeds: {family_flex_rows}
- Family-flex built rows: {family_flex_built_rows}
- Anchored SHIELD credit builds: {int(len(shield_anchored))}
- Iron condor pair attempts: {int(len(condor_attempts))}
- Iron condors built: {int((built['structure_kind'] == 'iron_condor').sum())}
- Blocked positive-EV rows exported: {len(blocked_positive_ev)}
- Alternates rows exported: {len(alternates_table)}
- ETF/index lane rows exported: {len(etf_lane_table)}
- Mid-pilot built rows: {int((built.get("liquidity_tier", pd.Series(dtype=str)) == "MID_PILOT").sum()) if "liquidity_tier" in built.columns else 0}
- Mixed-flow rescued rows: {int(built.get("mixed_flow_rescue", pd.Series(dtype=bool)).fillna(False).sum()) if "mixed_flow_rescue" in built.columns else 0}

## Duplicate whale seeds detected
{markdown_table(dup_details[["underlying_symbol", "track", "net_type", "expiry", "cp", "strike", "price", "width", "premium", "open_interest", "dup_count"]])}

## Family-flex translated rows
{markdown_table(table_round(built[built.get("is_family_flex", False).fillna(False).astype(bool)][["Ticker", "Action", "Buy leg", "Sell leg", "EV/ML", "POP", "Conviction", "Notice"]].sort_values(["EV/ML", "Conviction"], ascending=[False, False]).head(15)))}

## Anchored SHIELD credit rows
{markdown_table(table_round(shield_anchored[["Ticker", "Action", "Buy leg", "Sell leg", "EV/ML", "POP", "Conviction", "Notice"]].head(12)))}

## Condor pair attempts
{markdown_table(condor_attempts.head(20))}

## Minority-flow rows blocked from BUY
{markdown_table(table_round(minority_rows[["Ticker", "Action", "Buy leg", "Sell leg", "EV/ML", "POP", "Conviction", "Notice"]].head(10)))}

## Split-flow / near-event watch rows
{markdown_table(table_round(split_rows[["Ticker", "Action", "Buy leg", "Sell leg", "EV/ML", "POP", "Conviction", "Notice"]].head(10)))}

## FIRE neutral-conflict rows
{markdown_table(table_round(neutral_rows[["Ticker", "Action", "Buy leg", "Sell leg", "EV/ML", "POP", "Conviction", "Notice"]].head(10)))}

## SHIELD bias-mismatch rows
{markdown_table(table_round(shield_mismatch_rows[["Ticker", "Action", "Buy leg", "Sell leg", "EV/ML", "POP", "Conviction", "Notice"]].head(10)))}

## Rejected rows and reasons
{markdown_table(rejected[["underlying_symbol", "track", "net_type", "expiry", "cp", "strike", "premium", "reason"]].sort_values("premium", ascending=False).head(20))}

## Primary table after fixes
{markdown_table(primary_table)}

## Watch table after fixes
{markdown_table(watch_table)}

## ETF / index lane
{markdown_table(etf_lane_table)}

## Blocked positive-EV rows
{markdown_table(blocked_positive_ev.head(25) if not blocked_positive_ev.empty else blocked_positive_ev)}

## Per-ticker alternates
{markdown_table(alternates_table.head(25) if not alternates_table.empty else alternates_table)}

## Dark-pool context only
{markdown_table(dp_focus)}

## Execution notes
- Health Gate uses broker-native `accounts[].health_gate.status` when a Schwab artifact is present; otherwise execution stays **Bootstrap**.
- Rows whose computed Size bucket is `None` are not allowed to publish as BUY/SELL; they are watch-only.
- Family-flex derived rows may test the alternate structure family while preserving the file-native thesis direction; they do not inflate same-ticker bias or raw seed counts.
- High-premium earnings-blocked symbols are forced into the catalyst watch section rather than silently disappearing.
- If a streamed full-source bot file is present, ticker direction/bias comes from the full filtered candidate source, not only the Top-200 markdown slice.
- If only Top-200 markdown is present, high-premium low-visible-share symbols are forced into Top-Symbol Gap watch.
- Market cap no longer hard-deletes $10B-$80B liquid names; they route to mid-cap Pilot tier. Sub-$10B and unknown-cap rows are watch-only unless a future lane explicitly enables them.
- ETF/index candidates route to a separate ETF/index lane and do not contaminate the common-stock primary table.
- Low-POP candidates are tagged as convexity/lottery structures.
- The primary table is ranked by **EV/ML first**.
- Conviction is secondary context only.
- Browser / Atlas GEX is **not called** and has no effect on promotion, blocking, ranking, strike placement, notes, or sizing.
- SHIELD auto-promotion requires a file-native non-GEX anchor from whale side, actual hot-chain protection, and OI/liquidity support.
- Iron condors require two anchored SHIELD sides on the same ticker and expiry; no fabricated or inferred opposite side is allowed.
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    primary_table.to_csv(out_dir / f"options_scan_{scan_date}_audited_recommendations.csv", index=False)
    watch_table.to_csv(out_dir / f"options_scan_{scan_date}_audited_watch.csv", index=False)
    etf_lane_table.to_csv(out_dir / f"options_scan_{scan_date}_audited_etf_lane.csv", index=False)
    blocked_positive_ev.to_csv(out_dir / f"options_scan_{scan_date}_audited_blocked_positive_ev.csv", index=False)
    alternates_table.to_csv(out_dir / f"options_scan_{scan_date}_audited_alternates.csv", index=False)
    top_gap.to_csv(out_dir / f"options_scan_{scan_date}_audited_top_symbol_gap.csv", index=False)
    built_export_cols = [c for c in [
        "Ticker", "Action", "Buy leg", "Sell leg", "Expiry", "Net", "EV/ML", "POP", "Conviction",
        "Execution", "Notice", "Size", "structure_kind", "seed_family", "source_seed_family", "is_family_flex",
        "translation_type", "liquidity_tier", "mixed_flow_rescue", "is_primary_eligible", "premium", "marketcap",
        "er_days", "combined_bias", "minority_flow", "split_flow_watch", "neutral_conflict", "shield_bias_mismatch",
        "shield_anchor", "reward_risk", "long_strike", "short_strike", "actual_net"
    ] if c in built.columns]
    built[built_export_cols].to_csv(out_dir / f"options_scan_{scan_date}_audited_built_rows.csv", index=False)
    (out_dir / f"options_scan_{scan_date}_audited_audit.json").write_text(json.dumps(audit_json, indent=2))
    report_path = out_dir / f"options_scan_{scan_date}_audited_report.md"
    deep_audit_path = out_dir / f"Anu_Options_Engine_DEEP_AUDIT_{scan_date}.md"
    # Re-write using named paths so the completion summary can point to them.
    report_path.write_text(report_md)
    deep_audit_path.write_text(audit_md)

    print(
        f"PASS {ENGINE_VERSION} scan_date={scan_date} "
        f"source_mode={source_mode} primary_rows={len(primary_table)} "
        f"watch_rows={len(watch_table)} health_gate={schwab.get('status', 'UNKNOWN')} "
        f"oi_current={oi_curr_path.name}"
    )
    print(f"Wrote report: {report_path}")
    print(f"Wrote audit:  {out_dir / f'options_scan_{scan_date}_audited_audit.json'}")




def full_source_filter_diagnostic(path: Path, scan_date: date, sample_rows: int = 5000) -> Dict[str, Any]:
    """Inspect the first rows of a full bot ZIP and explain why seeds pass/fail."""
    diag: Dict[str, Any] = {"path": str(path), "scan_date": scan_date.isoformat()}
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                diag["error"] = "no CSV member found"
                return diag
            csv_name = max(names, key=lambda n: zf.getinfo(n).file_size)
            diag["csv_member"] = csv_name
            diag["csv_member_size_bytes"] = int(zf.getinfo(csv_name).file_size)
            with zf.open(csv_name) as fh:
                raw = pd.read_csv(fh, nrows=sample_rows, low_memory=False)
    except Exception as exc:
        diag["error"] = str(exc)
        return diag
    diag["raw_sample_rows"] = int(len(raw))
    diag["raw_columns"] = list(map(str, raw.columns))[:120]
    try:
        norm = normalize_flow_candidates(raw, scan_date)
        norm_all = normalize_column_names(raw)
        for canonical in ["underlying_symbol", "track", "net_type", "option_type", "expiry", "strike", "width", "premium", "open_interest"]:
            norm_all = coalesce_alias_columns(norm_all, canonical)
        diag["normalized_columns"] = list(map(str, norm_all.columns))[:120]
        for col in ["track", "net_type", "option_type", "side"]:
            if col in norm_all.columns:
                try:
                    diag[f"sample_unique_{col}"] = norm_all[col].astype(str).str[:40].value_counts(dropna=False).head(15).to_dict()
                except Exception:
                    pass
        diag["selected_candidates_in_sample"] = int(len(norm))
        if len(norm):
            diag["selected_by_family"] = norm["seed_family"].value_counts(dropna=False).to_dict()
            diag["selected_by_direction"] = norm["thesis_direction"].value_counts(dropna=False).to_dict()
            diag["selected_top_symbols"] = norm.groupby("underlying_symbol")["premium"].sum().sort_values(ascending=False).head(20).to_dict()
        else:
            tmp = normalize_column_names(raw)
            for canonical in ["underlying_symbol", "track", "net_type", "option_type", "cp", "side", "equity_type", "expiry", "dte", "underlying_price", "strike", "price", "width", "pct_width", "size", "premium", "open_interest"]:
                tmp = coalesce_alias_columns(tmp, canonical)
            for col in ["underlying_symbol", "track", "net_type", "option_type"]:
                if col not in tmp.columns:
                    tmp[col] = ""
            for col in ["dte", "strike", "width", "pct_width", "open_interest", "premium"]:
                if col not in tmp.columns:
                    tmp[col] = np.nan
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
            if "expiry" in tmp.columns:
                tmp["expiry"] = pd.to_datetime(tmp["expiry"], errors="coerce").dt.date
            if "cp" not in tmp.columns:
                tmp["cp"] = tmp["option_type"].astype(str).str.slice(0, 1).str.upper()
            else:
                tmp["cp"] = tmp["cp"].astype(str).str.slice(0, 1).str.upper()
            tr = tmp["track"].astype(str).str.upper()
            nt = tmp["net_type"].astype(str).str.lower()
            tmp["seed_family_diag"] = np.where(tr.str.contains("FIRE", na=False) & nt.str.contains("debit", na=False), "FIRE_DEBIT", np.where(tr.str.contains("SHIELD", na=False) & nt.str.contains("credit", na=False), "SHIELD_CREDIT", "IGNORE"))
            prime = yes_prime_mask_from_chunk(tmp)
            diag["mask_counts_sample"] = {
                "yes_prime_mask_true": int(prime.sum()),
                "fire_debit_shape": int((tmp["seed_family_diag"].eq("FIRE_DEBIT") & tmp["dte"].between(21, 70, inclusive="both") & (tmp["pct_width"].fillna(999) <= 0.45)).sum()),
                "shield_credit_shape": int((tmp["seed_family_diag"].eq("SHIELD_CREDIT") & tmp["dte"].between(28, 56, inclusive="both") & tmp["pct_width"].between(0.30, 0.55, inclusive="both")).sum()),
                "oi_ge_100": int((tmp["open_interest"].fillna(0) >= 100).sum()),
                "has_underlying_symbol": int(tmp["underlying_symbol"].astype(str).str.strip().ne("").sum()),
                "has_expiry": int(tmp.get("expiry", pd.Series(index=tmp.index, dtype=object)).notna().sum()) if "expiry" in tmp.columns else 0,
            }
    except Exception as exc:
        diag["normalization_error"] = str(exc)
    return diag


def diagnose_full_source(base_dir: Path, asof: Optional[date] = None, use_next_day_oi: bool = False) -> None:
    scan_date, paths = resolve_input_paths(base_dir, asof, use_next_day_oi=use_next_day_oi)
    flow_path = paths[0]
    print(f"scan_date={scan_date}")
    print(f"flow_path={flow_path}")
    if not (flow_path.suffix.lower() == ".zip" and re.search(r"(?:bot-eod-report|eod-flow-report|flow-eod-report)", flow_path.name)):
        print("flow_path is not a full bot/eod ZIP; no full-source diagnostic needed")
        return
    diag = full_source_filter_diagnostic(flow_path, scan_date)
    print(json.dumps(diag, indent=2, default=str))

def diagnose_runtime(base_dir: Path, out_dir: Path, asof: Optional[date] = None, project_dir: Optional[Path] = None, use_next_day_oi: bool = False) -> None:
    print(f"ENGINE_VERSION={ENGINE_VERSION}")
    print(f"script={Path(__file__).resolve()}")
    print(f"cwd={Path.cwd().resolve()}")
    print(f"base_dir={base_dir.resolve()}")
    print(f"out_dir={out_dir.resolve()}")
    print("input_scan_roots:")
    for root in input_scan_roots(base_dir):
        print(f"  - {root}")
    if project_dir is not None:
        print(f"project_dir={project_dir.resolve()}")
    alignment = audit_canonical_executable_sync(base_dir, out_dir, project_dir=project_dir)
    print(f"audit_gate_zero={alignment.get('status')}: {alignment.get('note')}")
    print("search_roots:")
    for root in alignment.get("search_roots", []):
        print(f"  - {root}")
    print("canonical_files:")
    for rec in alignment.get("records", []):
        print(f"  - {rec.get('file')}: present={rec.get('present')} version={rec.get('effective_logic_version')} path={rec.get('path')}")
    try:
        scan_date, paths = resolve_input_paths(base_dir, asof, use_next_day_oi=use_next_day_oi)
        print(f"scan_date={scan_date}")
        print("input_files:")
        for p in paths:
            print(f"  - {p}")
        schwab = resolve_schwab_path(base_dir, scan_date)
        print(f"schwab_file={schwab}")
    except Exception as exc:
        print(f"input_resolution_error={exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--out-dir", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--project-dir", type=Path, default=None, help="Optional directory containing the four canonical markdown files")
    parser.add_argument("--asof", type=str, default=None, help="Scan date in YYYY-MM-DD; defaults to latest common input date")
    parser.add_argument("--diagnose", action="store_true", help="Print local canonical-file/input discovery diagnostics and exit")
    parser.add_argument("--diagnose-full-source", action="store_true", help="Inspect the streamed full bot/EOD ZIP schema and filter counts, then exit")
    parser.add_argument("--allow-markdown-seed-fallback", action="store_true", help="Emergency/dev only: if full bot ZIP yields zero seeds, fallback to whale markdown Top-200 instead of failing")
    parser.add_argument("--use-next-day-oi", action="store_true", help="Explicit follow-through mode: allow current OI to use the next calendar day's chain-oi file when present. Default is scan-date OI only.")
    parser.add_argument("--allow-project-version-mismatch", action="store_true", help="Diagnostic only: allow run when canonical markdown/Python versions do not match")
    args = parser.parse_args()
    asof = date.fromisoformat(args.asof) if args.asof else None
    if args.diagnose:
        diagnose_runtime(args.base_dir, args.out_dir, asof=asof, project_dir=args.project_dir, use_next_day_oi=args.use_next_day_oi)
        return
    if args.diagnose_full_source:
        diagnose_full_source(args.base_dir, asof=asof, use_next_day_oi=args.use_next_day_oi)
        return
    run(
        args.base_dir,
        args.out_dir,
        asof=asof,
        allow_project_version_mismatch=args.allow_project_version_mismatch,
        project_dir=args.project_dir,
        allow_markdown_seed_fallback=args.allow_markdown_seed_fallback,
        use_next_day_oi=args.use_next_day_oi,
    )


if __name__ == "__main__":
    main()
