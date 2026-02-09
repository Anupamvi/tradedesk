#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sqlite3
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def norm_col(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def parse_float(x) -> float:
    if x is None:
        return math.nan
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return math.nan
    s = str(x).strip()
    if not s:
        return math.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "").replace("%", "")
    if s.endswith("x"):
        s = s[:-1]
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return math.nan


def find_col(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    cols = {c: norm_col(c) for c in columns}
    for a in aliases:
        aa = norm_col(a)
        for raw, n in cols.items():
            if n == aa:
                return raw
    return None


def read_any_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path, low_memory=False)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def candidate_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    pats = [
        "*.csv",
        "*.tsv",
        "*.txt",
        "*.xlsx",
        "*.xls",
    ]
    out: List[Path] = []
    for pat in pats:
        for p in root.rglob(pat):
            name = p.name.lower()
            if any(
                k in name
                for k in [
                    "trade",
                    "order",
                    "fill",
                    "position",
                    "statement",
                    "history",
                    "activity",
                    "transaction",
                    "ledger",
                    "execution",
                ]
            ):
                out.append(p)
    return sorted(set(out))


def parse_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None)


def parse_broker_date_value(x) -> pd.Timestamp:
    if x is None:
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT
    m = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", s)
    if m:
        pick = m[1] if ("as of" in s.lower() and len(m) >= 2) else m[0]
        return pd.to_datetime(pick, format="%m/%d/%Y", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


OPTION_SYMBOL_RE = re.compile(
    r"^\s*([A-Za-z0-9\.\-]+)\s+(\d{1,2}/\d{1,2}/\d{4})\s+([0-9]+(?:\.[0-9]+)?)\s+([CP])\s*$"
)


def parse_symbol_meta(symbol: str) -> Tuple[str, bool, str]:
    s = str(symbol or "").strip().upper()
    m = OPTION_SYMBOL_RE.match(s)
    if not m:
        return s, False, ""
    underlying = m.group(1).upper().strip()
    opt = "CALL" if m.group(4) == "C" else "PUT"
    return underlying, True, opt


def schwab_delta_from_action(action: str, qty: float, is_option: bool) -> float:
    a = norm_col(action).replace("_", " ")
    q = float(qty) if np.isfinite(qty) else math.nan
    if not np.isfinite(q) or q == 0:
        return 0.0
    abs_q = abs(q)
    if is_option:
        if a == "buy to open":
            return abs_q
        if a == "sell to open":
            return -abs_q
        if a == "buy to close":
            return abs_q
        if a == "sell to close":
            return -abs_q
        if a == "expired":
            return -q
        if a in {"assigned", "exchange or exercise"}:
            return q
        return 0.0

    if a in {"buy", "reinvest shares"}:
        return abs_q
    if a == "sell":
        return -abs_q
    return 0.0


def option_strategy_name(side: str, option_type: str) -> str:
    if side == "SHORT":
        return "Short Put Option" if option_type == "PUT" else "Short Call Option"
    return "Long Put Option" if option_type == "PUT" else "Long Call Option"


def standardize_schwab_transaction_df(
    df: pd.DataFrame, source_name: str, contract_multiplier: float
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df.empty:
        return pd.DataFrame(), {}

    cols = list(df.columns)
    date_col = find_col(cols, ["date", "transaction_date"])
    action_col = find_col(cols, ["action", "type", "activity"])
    symbol_col = find_col(cols, ["symbol", "instrument", "security", "ticker"])
    qty_col = find_col(cols, ["quantity", "qty", "contracts", "shares"])
    amount_col = find_col(cols, ["amount", "net_amount", "cash_amount"])
    price_col = find_col(cols, ["price", "trade_price"])
    if not all([date_col, action_col, symbol_col, qty_col, amount_col]):
        return pd.DataFrame(), {}

    raw = df[[date_col, action_col, symbol_col, qty_col, amount_col] + ([price_col] if price_col else [])].copy()
    raw.columns = ["date_raw", "action_raw", "symbol_raw", "qty_raw", "amount_raw"] + (
        ["price_raw"] if price_col else []
    )
    raw["row_id"] = np.arange(len(raw))
    raw["date"] = raw["date_raw"].map(parse_broker_date_value)
    raw["action_norm"] = raw["action_raw"].map(lambda x: norm_col(x).replace("_", " ").strip())
    raw["qty"] = raw["qty_raw"].map(parse_float)
    raw["amount"] = raw["amount_raw"].map(parse_float)
    if price_col:
        raw["price"] = raw["price_raw"].map(parse_float)
    else:
        raw["price"] = math.nan
    raw["amount"] = raw["amount"].fillna(0.0)

    meta = {
        "rows_total": float(len(raw)),
        "rows_ignored": 0.0,
        "events_used": 0.0,
        "closed_trades": 0.0,
        "unmatched_close_qty": 0.0,
        "open_lots_remaining": 0.0,
        "open_qty_remaining": 0.0,
        "assignment_or_exercise_events": 0.0,
    }

    events: List[Dict] = []
    for r in raw.itertuples(index=False):
        if pd.isna(r.date) or not np.isfinite(r.qty) or r.qty == 0:
            meta["rows_ignored"] += 1
            continue
        symbol_txt = str(r.symbol_raw or "").strip().upper()
        underlying, is_option, option_type = parse_symbol_meta(symbol_txt)
        delta = schwab_delta_from_action(r.action_raw, float(r.qty), is_option)
        if delta == 0:
            meta["rows_ignored"] += 1
            continue
        action_norm = str(r.action_norm)
        if action_norm in {"assigned", "exchange or exercise"}:
            meta["assignment_or_exercise_events"] += 1
        events.append(
            {
                "row_id": int(r.row_id),
                "date": pd.to_datetime(r.date),
                "action_raw": str(r.action_raw),
                "action_norm": action_norm,
                "instrument": symbol_txt,
                "underlying": underlying,
                "is_option": bool(is_option),
                "option_type": option_type,
                "qty_raw": float(r.qty),
                "delta_qty": float(delta),
                "amount": float(r.amount) if np.isfinite(r.amount) else 0.0,
                "price": float(r.price) if np.isfinite(r.price) else math.nan,
            }
        )
    if not events:
        return pd.DataFrame(), meta
    meta["events_used"] = float(len(events))

    events = sorted(events, key=lambda x: (x["date"], x["row_id"]))
    book = defaultdict(lambda: {"long": deque(), "short": deque()})
    closed_rows: List[Dict] = []

    for ev in events:
        inst = ev["instrument"]
        side_book = book[inst]
        delta = ev["delta_qty"]
        qty_abs = abs(delta)
        if qty_abs <= 0:
            continue
        close_unit_cash = ev["amount"] / qty_abs if qty_abs > 0 else 0.0
        can_open_long = ev["action_norm"] in {"buy to open", "buy", "reinvest shares"}
        can_open_short = ev["action_norm"] in {"sell to open", "sell"}

        def close_one(lot_side: str, remaining_qty: float) -> float:
            q = remaining_qty
            qbook = side_book[lot_side]
            while q > 0 and qbook:
                lot = qbook[0]
                m = min(q, lot["qty"])
                open_unit_cash = lot["unit_cash"]
                realized = (open_unit_cash + close_unit_cash) * m
                entry_price = lot["entry_price"]
                if not np.isfinite(entry_price):
                    denom = contract_multiplier if lot["is_option"] else 1.0
                    entry_price = abs(open_unit_cash) / max(1e-9, denom)
                exit_price = ev["price"]
                if not np.isfinite(exit_price):
                    if ev["action_norm"] in {"expired", "assigned", "exchange or exercise"}:
                        exit_price = 0.0
                    else:
                        denom = contract_multiplier if ev["is_option"] else 1.0
                        exit_price = abs(close_unit_cash) / max(1e-9, denom)

                closed_rows.append(
                    {
                        "source_file": source_name,
                        "date": ev["date"],
                        "open_date": lot["open_date"],
                        "symbol": lot["underlying"],
                        "strategy": lot["strategy"],
                        "side": lot["side"],
                        "qty": m,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "fees": 0.0,
                        "dte": math.nan,
                        "conviction": math.nan,
                        "realized_pnl": realized,
                    }
                )
                lot["qty"] -= m
                q -= m
                if lot["qty"] <= 1e-9:
                    qbook.popleft()
            return q

        if delta > 0:
            remaining = close_one("short", qty_abs)
            if remaining > 0:
                if can_open_long:
                    side_book["long"].append(
                        {
                            "qty": remaining,
                            "unit_cash": close_unit_cash,
                            "open_date": ev["date"],
                            "entry_price": ev["price"],
                            "side": "LONG",
                            "is_option": ev["is_option"],
                            "option_type": ev["option_type"],
                            "underlying": ev["underlying"],
                            "strategy": option_strategy_name("LONG", ev["option_type"])
                            if ev["is_option"]
                            else "Long Stock",
                        }
                    )
                else:
                    meta["unmatched_close_qty"] += remaining
        else:
            remaining = close_one("long", qty_abs)
            if remaining > 0:
                if can_open_short:
                    side_book["short"].append(
                        {
                            "qty": remaining,
                            "unit_cash": close_unit_cash,
                            "open_date": ev["date"],
                            "entry_price": ev["price"],
                            "side": "SHORT",
                            "is_option": ev["is_option"],
                            "option_type": ev["option_type"],
                            "underlying": ev["underlying"],
                            "strategy": option_strategy_name("SHORT", ev["option_type"])
                            if ev["is_option"]
                            else "Short Stock",
                        }
                    )
                else:
                    meta["unmatched_close_qty"] += remaining

    open_lots = 0
    open_qty = 0.0
    for d in book.values():
        for lot in list(d["long"]) + list(d["short"]):
            open_lots += 1
            open_qty += float(lot["qty"])
    meta["open_lots_remaining"] = float(open_lots)
    meta["open_qty_remaining"] = float(open_qty)

    if not closed_rows:
        return pd.DataFrame(), meta

    out = pd.DataFrame(closed_rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["open_date"] = pd.to_datetime(out["open_date"], errors="coerce")
    out["holding_days"] = (out["date"] - out["open_date"]).dt.total_seconds() / 86400.0
    out["win"] = out["realized_pnl"] > 0
    meta["closed_trades"] = float(len(out))
    keep = [
        "source_file",
        "date",
        "open_date",
        "symbol",
        "strategy",
        "side",
        "qty",
        "entry_price",
        "exit_price",
        "fees",
        "dte",
        "conviction",
        "realized_pnl",
        "holding_days",
        "win",
    ]
    return out[keep], meta


MANUAL_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}
MANUAL_MONTH_HEADER_RE = re.compile(
    r"^\s*("
    + "|".join(sorted(MANUAL_MONTH_MAP.keys(), key=len, reverse=True))
    + r")\s*[- ]\s*(\d{2,4})\s*$",
    re.I,
)
MANUAL_INLINE_DATE_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{4})\b")
MANUAL_QTY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*contracts?\b", re.I)
MANUAL_STRIKE_CP_RE = re.compile(r"\b\d+(?:\.\d+)?\s*([CP])\b", re.I)
MANUAL_SYMBOL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9\.\-]{0,9})\b")
MANUAL_MONTH_WORDS = set(MANUAL_MONTH_MAP.keys())


def parse_manual_section_date(text: str) -> pd.Timestamp:
    s = str(text or "").strip()
    m = MANUAL_MONTH_HEADER_RE.match(s)
    if not m:
        return pd.NaT
    mm = MANUAL_MONTH_MAP.get(m.group(1).lower())
    if mm is None:
        return pd.NaT
    yy_raw = m.group(2)
    year = int(yy_raw)
    if len(yy_raw) == 2:
        if year == 24:
            year = 2025
        else:
            year = 2000 + year if year < 80 else 1900 + year
    if year < 1900:
        return pd.NaT
    return pd.Timestamp(year=year, month=mm, day=1)


def parse_manual_inline_date(text: str) -> pd.Timestamp:
    s = str(text or "").strip()
    m = MANUAL_INLINE_DATE_RE.search(s)
    if not m:
        return pd.NaT
    return pd.to_datetime(m.group(1), format="%m/%d/%Y", errors="coerce")


def parse_manual_symbol(text: str) -> str:
    s = str(text or "").strip().upper()
    m = MANUAL_SYMBOL_RE.match(s)
    if not m:
        return "UNKNOWN"
    token = m.group(1).upper()
    if token.lower() in MANUAL_MONTH_WORDS:
        return "UNKNOWN"
    if token in {"TABLE", "COLUMN", "CALL", "PUT", "CSP"}:
        return "UNKNOWN"
    return token


def parse_manual_qty(text: str) -> float:
    s = str(text or "")
    m = MANUAL_QTY_RE.search(s)
    if not m:
        return 1.0
    q = parse_float(m.group(1))
    return q if np.isfinite(q) and q > 0 else 1.0


def parse_manual_strategy_and_side(text: str) -> Tuple[str, str]:
    u = str(text or "").upper()
    has_buy = bool(re.search(r"\bBUY\b", u))
    has_sell = bool(re.search(r"\bSELL\b", u))
    cp = MANUAL_STRIKE_CP_RE.search(u)
    has_put = (" PUT " in f" {u} ") or ("PUT" in u) or (cp is not None and cp.group(1).upper() == "P")
    has_call = (" CALL " in f" {u} ") or ("CALL" in u) or (cp is not None and cp.group(1).upper() == "C")

    side = ""
    if has_sell and not has_buy:
        side = "SHORT"
    elif has_buy and not has_sell:
        side = "LONG"
    elif has_put or has_call:
        has_pos_cash = bool(re.search(r"\(\s*\$\s*\d", u))
        has_neg_cash = bool(re.search(r"\(\s*-\s*\$\s*\d", u))
        if has_neg_cash and not has_pos_cash:
            side = "LONG"
        elif has_pos_cash and not has_neg_cash:
            side = "SHORT"

    if side == "SHORT" and has_put:
        return "Short Put Option", side
    if side == "SHORT" and has_call:
        return "Short Call Option", side
    if side == "LONG" and has_put:
        return "Long Put Option", side
    if side == "LONG" and has_call:
        return "Long Call Option", side
    if side == "LONG":
        return "Long Stock", side
    if side == "SHORT":
        return "Short Stock", side
    return "Unknown", side


def standardize_manual_options_log_df(df: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if df.empty or len(df.columns) < 2:
        return pd.DataFrame(), {}
    c0 = df.columns[0]
    c1 = df.columns[1]

    meta: Dict[str, float] = {
        "rows_total": float(len(df)),
        "header_rows": 0.0,
        "numeric_rows": 0.0,
        "usable_rows": 0.0,
        "year24_mapped_to_2025": 0.0,
    }

    rows: List[Dict] = []
    current_section_date = pd.NaT
    for i, r in df.iterrows():
        raw_cell = r.get(c0, "")
        raw_text = "" if pd.isna(raw_cell) else str(raw_cell).strip()
        raw_amt = r.get(c1, "")

        hdate = parse_manual_section_date(raw_text)
        if pd.notna(hdate):
            meta["header_rows"] += 1.0
            current_section_date = hdate
            if re.search(r"\b24\b", raw_text):
                meta["year24_mapped_to_2025"] += 1.0
            continue
        if re.fullmatch(r"\d{4}", raw_text):
            continue
        if not raw_text or norm_col(raw_text).startswith("table"):
            continue

        realized = parse_float(raw_amt)
        if not np.isfinite(realized):
            continue
        meta["numeric_rows"] += 1.0

        inline_date = parse_manual_inline_date(raw_text)
        row_date = inline_date if pd.notna(inline_date) else current_section_date
        if pd.isna(row_date):
            continue

        strategy, side = parse_manual_strategy_and_side(raw_text)
        qty = parse_manual_qty(raw_text)
        rows.append(
            {
                "source_file": source_name,
                "date": pd.to_datetime(row_date, errors="coerce"),
                "open_date": pd.NaT,
                "symbol": parse_manual_symbol(raw_text),
                "strategy": strategy,
                "side": side,
                "qty": qty,
                "entry_price": math.nan,
                "exit_price": math.nan,
                "fees": 0.0,
                "dte": math.nan,
                "conviction": math.nan,
                "realized_pnl": float(realized),
                "holding_days": math.nan,
                "win": float(realized) > 0,
            }
        )

    if not rows:
        return pd.DataFrame(), meta
    out = pd.DataFrame(rows)
    meta["usable_rows"] = float(len(out))
    keep = [
        "source_file",
        "date",
        "open_date",
        "symbol",
        "strategy",
        "side",
        "qty",
        "entry_price",
        "exit_price",
        "fees",
        "dte",
        "conviction",
        "realized_pnl",
        "holding_days",
        "win",
    ]
    return out[keep], meta


def standardize_realized_trade_df(df: pd.DataFrame, source_name: str, multiplier: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    cols = list(df.columns)
    date_col = find_col(
        cols,
        [
            "close_date",
            "trade_date",
            "date",
            "executed_at",
            "execution_time",
            "filled_at",
            "timestamp",
        ],
    )
    open_col = find_col(cols, ["open_date", "entry_date", "opened_at"])
    symbol_col = find_col(cols, ["symbol", "ticker", "underlying_symbol", "instrument"])
    strategy_col = find_col(cols, ["strategy", "strategy_type", "setup"])
    side_col = find_col(cols, ["side", "action", "buy_sell", "direction"])
    qty_col = find_col(cols, ["quantity", "qty", "contracts", "size", "shares"])
    entry_col = find_col(cols, ["entry_price", "entry", "open_price", "buy_price", "avg_open_price", "price_in"])
    exit_col = find_col(cols, ["exit_price", "exit", "close_price", "sell_price", "avg_close_price", "price_out"])
    pnl_col = find_col(
        cols,
        [
            "realized_pnl",
            "realized_pl",
            "pnl",
            "profit_loss",
            "gain_loss",
            "realized_gain_loss",
            "net_pnl",
            "closed_pl",
        ],
    )
    fees_col = find_col(cols, ["fees", "fee", "commission", "commissions", "total_fees"])
    dte_col = find_col(cols, ["dte", "days_to_expiry", "days_to_expiration"])
    conviction_col = find_col(cols, ["conviction", "conviction_pct", "confidence", "score"])
    net_type_col = find_col(cols, ["net_type", "debit_credit", "entry_type"])

    out = pd.DataFrame(index=df.index)
    out["source_file"] = source_name
    out["symbol"] = df[symbol_col].astype(str).str.upper().str.strip() if symbol_col else "UNKNOWN"
    out["strategy"] = df[strategy_col].astype(str).str.strip() if strategy_col else "UNKNOWN"
    out["side"] = df[side_col].astype(str).str.upper().str.strip() if side_col else ""
    out["date"] = parse_dates(df[date_col]) if date_col else pd.NaT
    out["open_date"] = parse_dates(df[open_col]) if open_col else pd.NaT
    out["qty"] = df[qty_col].map(parse_float).abs() if qty_col else 1.0
    out["entry_price"] = df[entry_col].map(parse_float) if entry_col else math.nan
    out["exit_price"] = df[exit_col].map(parse_float) if exit_col else math.nan
    out["fees"] = df[fees_col].map(parse_float) if fees_col else 0.0
    out["dte"] = df[dte_col].map(parse_float) if dte_col else math.nan
    out["conviction"] = df[conviction_col].map(parse_float) if conviction_col else math.nan
    out["net_type"] = df[net_type_col].astype(str).str.lower().str.strip() if net_type_col else ""

    realized = df[pnl_col].map(parse_float) if pnl_col else pd.Series(np.nan, index=df.index)
    out["realized_pnl"] = realized

    can_compute = out["entry_price"].notna() & out["exit_price"].notna() & out["qty"].notna()
    if can_compute.any():
        side_short = out["side"].str.contains("SELL|SHORT|WRITE", regex=True, na=False)
        side_long = out["side"].str.contains("BUY|LONG", regex=True, na=False)
        dir_sign = np.where(side_short, -1.0, np.where(side_long, 1.0, np.nan))
        net_short = out["net_type"].str.contains("credit", na=False)
        dir_sign = np.where(np.isnan(dir_sign), np.where(net_short, -1.0, 1.0), dir_sign)
        comp = (out["exit_price"] - out["entry_price"]) * out["qty"] * float(multiplier) * dir_sign
        comp = comp - out["fees"].fillna(0.0)
        out.loc[out["realized_pnl"].isna() & can_compute, "realized_pnl"] = comp[out["realized_pnl"].isna() & can_compute]

    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["open_date"] = pd.to_datetime(out["open_date"], errors="coerce")
    out["holding_days"] = (
        (out["date"] - out["open_date"]).dt.total_seconds() / 86400.0 if "open_date" in out else np.nan
    )
    out["win"] = out["realized_pnl"] > 0

    keep = [
        "source_file",
        "date",
        "open_date",
        "symbol",
        "strategy",
        "side",
        "qty",
        "entry_price",
        "exit_price",
        "fees",
        "dte",
        "conviction",
        "realized_pnl",
        "holding_days",
        "win",
    ]
    return out[keep]


def profit_factor(p: pd.Series) -> float:
    gp = float(p[p > 0].sum())
    gl = float(-p[p < 0].sum())
    if gl <= 0:
        return math.inf if gp > 0 else math.nan
    return gp / gl


def max_drawdown(pnl: pd.Series) -> float:
    c = pnl.cumsum()
    dd = c - c.cummax()
    return float(dd.min()) if len(dd) else 0.0


def longest_streak(flags: Iterable[bool], target: bool) -> int:
    best = 0
    cur = 0
    for f in flags:
        if bool(f) == target:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def agg_table(df: pd.DataFrame, col: str, min_trades: int = 5) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    g = (
        df.groupby(col, dropna=False)
        .agg(
            trades=("realized_pnl", "size"),
            win_rate=("win", "mean"),
            net_pnl=("realized_pnl", "sum"),
            avg_pnl=("realized_pnl", "mean"),
            median_pnl=("realized_pnl", "median"),
            avg_win=("realized_pnl", lambda s: float(s[s > 0].mean()) if (s > 0).any() else math.nan),
            avg_loss=("realized_pnl", lambda s: float(s[s < 0].mean()) if (s < 0).any() else math.nan),
            profit_factor=("realized_pnl", profit_factor),
        )
        .reset_index()
    )
    g = g[g["trades"] >= max(1, int(min_trades))]
    return g.sort_values(["avg_pnl", "net_pnl"], ascending=[False, False]).reset_index(drop=True)


def realized_insights(df: pd.DataFrame) -> Tuple[Dict, Dict[str, pd.DataFrame], List[str]]:
    x = df.dropna(subset=["date", "realized_pnl"]).copy()
    x = x.sort_values("date").reset_index(drop=True)
    if x.empty:
        raise RuntimeError("No rows with usable date + realized_pnl.")

    total = int(len(x))
    wins = int((x["realized_pnl"] > 0).sum())
    losses = int((x["realized_pnl"] < 0).sum())
    net = float(x["realized_pnl"].sum())
    gp = float(x.loc[x["realized_pnl"] > 0, "realized_pnl"].sum())
    gl = float(-x.loc[x["realized_pnl"] < 0, "realized_pnl"].sum())
    avg = float(x["realized_pnl"].mean())
    med = float(x["realized_pnl"].median())
    wr = float(wins / total) if total else math.nan
    pf = float(gp / gl) if gl > 0 else math.inf
    avg_win = float(x.loc[x["realized_pnl"] > 0, "realized_pnl"].mean()) if wins else math.nan
    avg_loss = float(x.loc[x["realized_pnl"] < 0, "realized_pnl"].mean()) if losses else math.nan
    payoff = abs(avg_win / avg_loss) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0 else math.nan
    mdd = max_drawdown(x["realized_pnl"])
    lws = longest_streak((x["realized_pnl"] > 0).tolist(), True)
    lls = longest_streak((x["realized_pnl"] > 0).tolist(), False)

    if "holding_days" in x:
        win_hold = float(x.loc[x["realized_pnl"] > 0, "holding_days"].mean())
        loss_hold = float(x.loc[x["realized_pnl"] < 0, "holding_days"].mean())
    else:
        win_hold = math.nan
        loss_hold = math.nan

    x["weekday"] = x["date"].dt.day_name()
    x["month"] = x["date"].dt.to_period("M").astype(str)
    if x["conviction"].notna().any():
        x["conv_bucket"] = pd.cut(
            x["conviction"],
            bins=[-np.inf, 60, 70, 80, np.inf],
            labels=["<=60", "61-70", "71-80", ">80"],
        ).astype(str)
    else:
        x["conv_bucket"] = "unknown"

    by_strategy = agg_table(x, "strategy", min_trades=5)
    by_symbol = agg_table(x, "symbol", min_trades=5)
    by_weekday = agg_table(x, "weekday", min_trades=5)
    by_conv = agg_table(x, "conv_bucket", min_trades=3)
    by_month = agg_table(x, "month", min_trades=1)

    insights: List[str] = []
    insights.append(f"Sample has {total} closed trades; win rate {wr:.1%}, profit factor {pf:.2f}, net P/L ${net:,.2f}.")
    if pf >= 1.5:
        insights.append("Core edge exists (profit factor > 1.5). Scale only setups with stable positive expectancy.")
    elif pf >= 1.0:
        insights.append("Edge is marginal (profit factor near 1.0-1.5). Focus on tighter risk controls and setup pruning.")
    else:
        insights.append("No positive edge yet (profit factor < 1.0). Priority is loss containment before scaling.")

    if np.isfinite(payoff) and payoff < 1.0 and wr < 0.55:
        insights.append("Average loss is larger than average win with sub-55% win rate. Add harder stop discipline.")
    if np.isfinite(loss_hold) and np.isfinite(win_hold) and loss_hold > win_hold * 1.2:
        insights.append(
            f"Losing trades are held longer ({loss_hold:.2f}d) than winners ({win_hold:.2f}d). Cut losers faster."
        )
    if lls >= 4:
        insights.append(f"Longest losing streak is {lls} trades. Use streak-based size-down rules.")

    if not by_strategy.empty:
        best = by_strategy.head(3)
        worst = by_strategy.sort_values("avg_pnl").head(3)
        insights.append(
            "Best strategies by avg P/L: "
            + ", ".join([f"{r.strategy} (${r.avg_pnl:,.2f}/trade, n={int(r.trades)})" for r in best.itertuples()])
        )
        insights.append(
            "Worst strategies by avg P/L: "
            + ", ".join([f"{r.strategy} (${r.avg_pnl:,.2f}/trade, n={int(r.trades)})" for r in worst.itertuples()])
        )

    if not by_symbol.empty:
        top = by_symbol.head(5)
        contrib = top["net_pnl"].sum() / max(1e-9, net) if net != 0 else math.nan
        if np.isfinite(contrib) and contrib > 0:
            insights.append(f"Top 5 symbols contribute {contrib:.1%} of net P/L. Keep concentration caps and size by edge.")

    summary = {
        "trade_count": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "net_pnl": net,
        "gross_profit": gp,
        "gross_loss": gl,
        "avg_pnl": avg,
        "median_pnl": med,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "profit_factor": pf,
        "max_drawdown": mdd,
        "longest_win_streak": lws,
        "longest_loss_streak": lls,
        "avg_holding_days_win": win_hold,
        "avg_holding_days_loss": loss_hold,
        "start_date": x["date"].min().date().isoformat(),
        "end_date": x["date"].max().date().isoformat(),
        "days_covered": int((x["date"].max() - x["date"].min()).days) + 1,
    }

    tables = {
        "strategy": by_strategy,
        "symbol": by_symbol,
        "weekday": by_weekday,
        "conviction": by_conv,
        "month": by_month,
    }
    return summary, tables, insights


def flow_proxy_insights(sqlite_path: Path, start_date: str, end_date: str) -> Tuple[Dict, Dict[str, pd.DataFrame], List[str]]:
    con = sqlite3.connect(str(sqlite_path))
    q = """
    SELECT f.trade_date, f.ticker, f.option_type, f.print_type, f.execution_location, f.time_bucket,
           f.dir_sign, f.abs_premium_w,
           t.daily_sign
    FROM flow_prints f
    JOIN ticker_day t
      ON f.trade_date=t.trade_date AND f.ticker=t.ticker
    WHERE f.trade_date BETWEEN ? AND ?
    """
    df = pd.read_sql_query(q, con, params=[start_date, end_date])
    con.close()
    if df.empty:
        raise RuntimeError("No flow proxy rows in the selected range.")

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    x = df[(df["dir_sign"].notna()) & (df["daily_sign"].notna())].copy()
    x = x[(x["dir_sign"] != 0) & (x["daily_sign"] != 0)].copy()
    if x.empty:
        raise RuntimeError("No directional rows available for proxy analysis.")
    x["aligned"] = (x["dir_sign"].astype(int) == x["daily_sign"].astype(int)).astype(int)
    x["w"] = pd.to_numeric(x["abs_premium_w"], errors="coerce").fillna(0.0)

    unweighted = float(x["aligned"].mean())
    weighted = float((x["aligned"] * x["w"]).sum() / max(1e-9, x["w"].sum()))

    def grp(col: str, min_trades: int = 50):
        g = (
            x.groupby(col)
            .agg(
                trades=("aligned", "size"),
                aligned_rate=("aligned", "mean"),
                weighted_aligned_rate=(
                    "aligned",
                    lambda s: float((s * x.loc[s.index, "w"]).sum() / max(1e-9, x.loc[s.index, "w"].sum())),
                ),
            )
            .reset_index()
        )
        g = g[g["trades"] >= min_trades]
        return g.sort_values(["weighted_aligned_rate", "trades"], ascending=[False, False]).reset_index(drop=True)

    by_print = grp("print_type", min_trades=100)
    by_exec = grp("execution_location", min_trades=100)
    by_bucket = grp("time_bucket", min_trades=100)
    by_option = grp("option_type", min_trades=100)
    by_ticker = grp("ticker", min_trades=100)

    insights = [
        f"Proxy directional alignment (non-neutral): {unweighted:.1%} unweighted, {weighted:.1%} premium-weighted.",
    ]
    if not by_print.empty:
        best = by_print.iloc[0]
        worst = by_print.iloc[-1]
        insights.append(
            f"Best print type: {best['print_type']} ({best['weighted_aligned_rate']:.1%}), "
            f"worst: {worst['print_type']} ({worst['weighted_aligned_rate']:.1%})."
        )
    if not by_bucket.empty:
        insights.append(
            "Time bucket ranking (weighted): "
            + ", ".join([f"{r.time_bucket}:{r.weighted_aligned_rate:.1%}" for r in by_bucket.itertuples()])
        )
    if not by_exec.empty:
        insights.append(
            "Execution side ranking (weighted): "
            + ", ".join([f"{r.execution_location}:{r.weighted_aligned_rate:.1%}" for r in by_exec.itertuples()])
        )
    insights.append(
        "Use this only as signal quality proxy; realized account P/L analysis requires broker closed-trade exports."
    )

    summary = {
        "rows": int(len(x)),
        "start_date": x["trade_date"].min().date().isoformat(),
        "end_date": x["trade_date"].max().date().isoformat(),
        "days_covered": int((x["trade_date"].max() - x["trade_date"].min()).days) + 1,
        "aligned_rate_unweighted": unweighted,
        "aligned_rate_weighted": weighted,
    }
    tables = {
        "print_type": by_print,
        "execution_location": by_exec,
        "time_bucket": by_bucket,
        "option_type": by_option,
        "ticker": by_ticker,
    }
    return summary, tables, insights


def to_markdown_table(df: pd.DataFrame, n: int = 15) -> str:
    if df.empty:
        return "_none_"
    return df.head(n).to_markdown(index=False)


def build_report(mode: str, summary: Dict, tables: Dict[str, pd.DataFrame], insights: List[str], output_path: Path) -> None:
    lines: List[str] = []
    lines.append("# Trading Performance Review")
    lines.append(f"Mode: {mode}")
    lines.append("")
    lines.append("## Summary")
    for k, v in summary.items():
        if isinstance(v, float):
            if "rate" in k or "win_rate" in k:
                lines.append(f"- {k}: {v:.2%}")
            else:
                lines.append(f"- {k}: {v:,.4f}" if abs(v) < 100 else f"- {k}: {v:,.2f}")
        else:
            lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## What Is Working / Not Working")
    for s in insights:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("## Breakdowns")
    for name, df in tables.items():
        lines.append(f"### {name}")
        lines.append(to_markdown_table(df, n=20))
        lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Analyze 1-year trading performance and generate actionable insights.")
    ap.add_argument("--input", default=r"c:\uw_root", help="Trade source file/folder or sqlite path.")
    ap.add_argument("--output-dir", default=r"c:\uw_root\out\trade_performance_review", help="Output directory.")
    ap.add_argument("--mode", choices=["auto", "realized", "flow_proxy"], default="auto")
    ap.add_argument("--sqlite-path", default=r"c:\uw_root\_derived\uw_ledger.sqlite")
    ap.add_argument("--start-date", default="", help="YYYY-MM-DD; default is 365 days before today.")
    ap.add_argument("--end-date", default="", help="YYYY-MM-DD; default is today.")
    ap.add_argument("--contract-multiplier", type=float, default=100.0, help="Multiplier when inferring pnl from entry/exit.")
    args = ap.parse_args()

    today = dt.date.today()
    start = dt.date.fromisoformat(args.start_date) if args.start_date else (today - dt.timedelta(days=365))
    end = dt.date.fromisoformat(args.end_date) if args.end_date else today
    if start > end:
        raise ValueError("start-date must be <= end-date")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = args.mode
    summary: Dict = {}
    tables: Dict[str, pd.DataFrame] = {}
    insights: List[str] = []

    if mode in {"auto", "realized"}:
        src = Path(args.input).resolve()
        files = candidate_files(src)
        frames = []
        schwab_metas: List[Dict[str, float]] = []
        manual_metas: List[Dict[str, float]] = []
        for p in files:
            try:
                raw = read_any_table(p)
                std = standardize_realized_trade_df(raw, str(p), args.contract_multiplier)
                if not std.empty and std["realized_pnl"].notna().any():
                    frames.append(std)
                    continue
                schwab_std, schwab_meta = standardize_schwab_transaction_df(raw, str(p), args.contract_multiplier)
                if not schwab_std.empty:
                    frames.append(schwab_std)
                    if schwab_meta:
                        schwab_metas.append(schwab_meta)
                    continue
                manual_std, manual_meta = standardize_manual_options_log_df(raw, str(p))
                if not manual_std.empty:
                    frames.append(manual_std)
                    if manual_meta:
                        manual_metas.append(manual_meta)
            except Exception:
                continue
        if frames:
            all_trades = pd.concat(frames, ignore_index=True)
            all_trades = all_trades.dropna(subset=["date"]).copy()
            all_trades = all_trades[
                (all_trades["date"].dt.date >= start) & (all_trades["date"].dt.date <= end)
            ].copy()
            has_realized = all_trades["realized_pnl"].notna().any()
            if has_realized and len(all_trades) > 0:
                cleaned_path = out_dir / "cleaned_realized_trades.csv"
                all_trades.to_csv(cleaned_path, index=False)
                summary, tables, insights = realized_insights(all_trades)
                summary["data_mode"] = "realized"
                summary["source_files_scanned"] = len(files)
                summary["cleaned_rows"] = int(len(all_trades))
                if schwab_metas:
                    summary["schwab_parser_files"] = int(len(schwab_metas))
                    summary["schwab_rows_total"] = int(sum(m.get("rows_total", 0.0) for m in schwab_metas))
                    summary["schwab_events_used"] = int(sum(m.get("events_used", 0.0) for m in schwab_metas))
                    summary["schwab_closed_trades"] = int(sum(m.get("closed_trades", 0.0) for m in schwab_metas))
                    summary["schwab_unmatched_close_qty"] = float(
                        sum(m.get("unmatched_close_qty", 0.0) for m in schwab_metas)
                    )
                    summary["schwab_open_lots_remaining"] = int(
                        sum(m.get("open_lots_remaining", 0.0) for m in schwab_metas)
                    )
                    summary["schwab_open_qty_remaining"] = float(sum(m.get("open_qty_remaining", 0.0) for m in schwab_metas))
                    insights.insert(
                        0,
                        (
                            "Schwab transaction reconstruction: "
                            f"{summary['schwab_closed_trades']} closed lots analyzed; "
                            f"{summary['schwab_open_lots_remaining']} open lots still active."
                        ),
                    )
                if manual_metas:
                    summary["manual_log_parser_files"] = int(len(manual_metas))
                    summary["manual_log_rows_total"] = int(sum(m.get("rows_total", 0.0) for m in manual_metas))
                    summary["manual_log_header_rows"] = int(sum(m.get("header_rows", 0.0) for m in manual_metas))
                    summary["manual_log_numeric_rows"] = int(sum(m.get("numeric_rows", 0.0) for m in manual_metas))
                    summary["manual_log_usable_rows"] = int(sum(m.get("usable_rows", 0.0) for m in manual_metas))
                    summary["manual_log_year24_mapped_to_2025_headers"] = int(
                        sum(m.get("year24_mapped_to_2025", 0.0) for m in manual_metas)
                    )
                    insights.insert(
                        0,
                        (
                            "Manual options log parser: "
                            f"{summary['manual_log_usable_rows']} P/L rows analyzed from "
                            f"{summary['manual_log_header_rows']} month sections."
                        ),
                    )
                mode = "realized"
        if mode == "realized" and not summary and args.mode == "realized":
            raise RuntimeError(
                "No usable realized trade data found. Provide broker closed-trade exports with realized pnl "
                "or entry/exit+qty columns."
            )

    if not summary:
        sqlite_path = Path(args.sqlite_path).resolve()
        if mode == "realized":
            raise RuntimeError("Realized mode requested but no usable realized data found.")
        if not sqlite_path.exists():
            raise FileNotFoundError(f"Missing sqlite for flow proxy: {sqlite_path}")
        summary, tables, insights = flow_proxy_insights(sqlite_path, start.isoformat(), end.isoformat())
        summary["data_mode"] = "flow_proxy"
        mode = "flow_proxy"

    (out_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for name, df in tables.items():
        if not df.empty:
            df.to_csv(out_dir / f"{name}_breakdown.csv", index=False)
    report_path = out_dir / "trading_performance_review.md"
    build_report(mode, summary, tables, insights, report_path)

    print(f"Mode used: {mode}")
    print(f"Date range requested: {start.isoformat()} to {end.isoformat()}")
    print(f"Rows analyzed: {summary.get('trade_count', summary.get('rows', 0))}")
    print(f"Days covered in data: {summary.get('days_covered', 0)}")
    print(f"Wrote: {report_path}")
    print(f"Wrote: {out_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
