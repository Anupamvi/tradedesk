#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from collections import defaultdict, deque
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

DEFAULT_SHEET_CACHE_CSV = r"c:\uw_root\out\playbook\source_sheet_latest.csv"
DEFAULT_SHEET_REALIZED_CSV = r"c:\uw_root\out\playbook\cleaned_realized_trades_from_sheet.csv"
SCHWAB_OPTION_SYMBOL_RE = re.compile(r"^\s*([A-Z\. ]{1,6})(\d{6})([CP])(\d{8})\s*$")
COMPACT_OPTION_SYMBOL_RE = re.compile(r"^\s*([A-Z\.]{1,6})(\d{6})([CP])(\d{8})\s*$")


def norm_col(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


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
    if s.startswith("-"):
        neg = True
    try:
        v = float(s)
        return -abs(v) if neg else v
    except Exception:
        return math.nan


def find_col(columns: List[str], aliases: List[str]) -> Optional[str]:
    mapped = {c: norm_col(c) for c in columns}
    for alias in aliases:
        a = norm_col(alias)
        for raw, n in mapped.items():
            if n == a:
                return raw
    return None


def compute_profit_factor(pnl: pd.Series) -> float:
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    if gl <= 0:
        return math.inf if gp > 0 else math.nan
    return gp / gl


def compute_max_drawdown(pnl: pd.Series) -> float:
    if pnl.empty:
        return 0.0
    c = pnl.cumsum()
    dd = c - c.cummax()
    return float(dd.min())


def longest_streak(flags: List[bool], target: bool) -> int:
    best = 0
    cur = 0
    for f in flags:
        if bool(f) == target:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return best


def trailing_loss_streak(pnl: pd.Series) -> int:
    k = 0
    for v in reversed(pnl.tolist()):
        if v < 0:
            k += 1
        else:
            break
    return k


def load_config(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg if isinstance(cfg, dict) else {}


def load_realized_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    date_col = find_col(cols, ["date", "trade_date", "close_date", "closed_at"])
    pnl_col = find_col(cols, ["realized_pnl", "pnl", "profit_loss", "net_pnl", "amount"])
    strategy_col = find_col(cols, ["strategy", "strategy_type", "setup"])
    symbol_col = find_col(cols, ["symbol", "ticker", "underlying", "underlying_symbol"])

    if not date_col or not pnl_col:
        raise ValueError(f"Missing required columns in {path}. Need date + realized pnl.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    out["realized_pnl"] = df[pnl_col].map(parse_float)
    out["strategy"] = df[strategy_col].astype(str).str.strip() if strategy_col else "Unknown"
    out["symbol"] = df[symbol_col].astype(str).str.strip().str.upper() if symbol_col else "UNKNOWN"
    out["symbol"] = out["symbol"].replace({"": "UNKNOWN", "NAN": "UNKNOWN"})
    out["strategy"] = out["strategy"].replace({"": "Unknown", "nan": "Unknown"})

    out = out.dropna(subset=["date", "realized_pnl"]).copy()
    out = out.sort_values("date").reset_index(drop=True)
    out["month"] = out["date"].dt.to_period("M").astype(str)
    out["weekday"] = out["date"].dt.day_name()
    out["win"] = out["realized_pnl"] > 0
    return out


def parse_sheet_id_and_gid(sheet_csv_url: str) -> Tuple[str, str]:
    url = str(sheet_csv_url or "").strip()
    m_id = re.search(r"/spreadsheets/d/([a-zA-Z0-9\-_]+)", url)
    if not m_id:
        raise ValueError(f"Could not parse Google Sheet ID from URL: {sheet_csv_url}")
    m_gid = re.search(r"[?&]gid=(\d+)", url)
    return m_id.group(1), (m_gid.group(1) if m_gid else "")


def normalize_headers(values: List[Any]) -> List[str]:
    headers: List[str] = []
    seen: Dict[str, int] = {}
    for i, v in enumerate(values):
        raw = str(v).strip() if v is not None else ""
        h = raw if raw else f"Unnamed: {i}"
        if h in seen:
            seen[h] += 1
            h = f"{h}.{seen[h]}"
        else:
            seen[h] = 0
        headers.append(h)
    return headers


def _hex_to_rgb(hex_text: str) -> Optional[Tuple[int, int, int]]:
    s = str(hex_text or "").strip().upper()
    if not s:
        return None
    s = s.replace("#", "")
    if len(s) == 8:
        s = s[2:]
    if not re.fullmatch(r"[0-9A-F]{6}", s):
        return None
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _is_yellow_rgb(rgb: Tuple[int, int, int]) -> bool:
    r, g, b = rgb
    return r >= 225 and g >= 185 and b <= 150 and (r - b) >= 90 and (g - b) >= 50


def _cell_is_yellow(cell: Any) -> bool:
    fill = getattr(cell, "fill", None)
    if fill is None or not getattr(fill, "fill_type", None):
        return False
    colors = [getattr(fill, "fgColor", None), getattr(fill, "start_color", None)]
    for color in colors:
        rgb = _hex_to_rgb(getattr(color, "rgb", ""))
        if rgb and _is_yellow_rgb(rgb):
            return True
    return False


def _is_manual_section_header(text: Any) -> bool:
    s = "" if text is None else str(text).strip()
    if not s:
        return False
    try:
        from uwos.analyze_trading_year import parse_manual_section_date  # local module

        return pd.notna(parse_manual_section_date(s))
    except Exception:
        # Fallback: Month-YY / Month YYYY
        return bool(
            re.match(
                r"^\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s*[- ]\s*(\d{2,4})\s*$",
                s,
                flags=re.I,
            )
        )


def load_sheet_rows_from_xlsx(sheet_csv_url: str, row_filter: str) -> Tuple[pd.DataFrame, Dict[str, float]]:
    try:
        from openpyxl import load_workbook
    except Exception as e:
        raise RuntimeError(
            "Row-color filtering requires openpyxl. Install with: python -m pip install openpyxl"
        ) from e

    sheet_id, gid = parse_sheet_id_and_gid(sheet_csv_url)
    xlsx_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"
    if gid:
        xlsx_url += f"&gid={gid}&single=true"

    resp = requests.get(xlsx_url, timeout=60)
    resp.raise_for_status()
    wb = load_workbook(BytesIO(resp.content), data_only=True)
    ws = wb.active

    rows: List[List[Any]] = []
    yellow_flags: List[bool] = []
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=ws.max_column):
        vals = [cell.value for cell in row]
        if all(v is None or str(v).strip() == "" for v in vals):
            continue
        rows.append(vals)
        yellow_flags.append(any(_cell_is_yellow(cell) for cell in row))

    if not rows:
        raise RuntimeError("XLSX export loaded but contained no non-empty rows.")

    headers = normalize_headers(rows[0])
    data_rows = rows[1:]
    data_yellow = yellow_flags[1:]

    kept_rows: List[List[Any]] = []
    yellow_kept = 0
    for vals, is_yellow in zip(data_rows, data_yellow):
        keep = True
        if row_filter == "yellow":
            c0 = vals[0] if vals else None
            keep = bool(is_yellow) or _is_manual_section_header(c0)
        if keep:
            kept_rows.append(vals)
            if is_yellow:
                yellow_kept += 1

    out = pd.DataFrame(kept_rows, columns=headers)
    meta = {
        "xlsx_nonempty_rows": float(len(rows)),
        "xlsx_data_rows": float(len(data_rows)),
        "xlsx_total_yellow_rows": float(sum(1 for f in data_yellow if f)),
        "xlsx_kept_rows": float(len(kept_rows)),
        "xlsx_kept_yellow_rows": float(yellow_kept),
    }
    return out, meta


def build_realized_from_sheet_csv_url(
    sheet_csv_url: str,
    raw_cache_csv: Path,
    realized_out_csv: Path,
    row_filter: str = "all",
) -> Dict[str, float]:
    """
    Pull Google Sheet CSV export URL, parse manual options log rows into standardized realized trades,
    and persist both raw cache + standardized realized CSV.
    """
    raw_cache_csv.parent.mkdir(parents=True, exist_ok=True)
    realized_out_csv.parent.mkdir(parents=True, exist_ok=True)

    ingest_meta: Dict[str, float] = {}
    if str(row_filter).strip().lower() == "yellow":
        raw_df, ingest_meta = load_sheet_rows_from_xlsx(sheet_csv_url, row_filter="yellow")
    else:
        raw_df = pd.read_csv(sheet_csv_url, low_memory=False)
        ingest_meta = {"csv_rows_loaded": float(len(raw_df))}
    raw_df.to_csv(raw_cache_csv, index=False)

    try:
        from uwos.analyze_trading_year import standardize_manual_options_log_df  # local module
    except Exception as e:
        raise RuntimeError(f"Failed to import manual log parser from analyze_trading_year.py: {e}") from e

    std_df, meta = standardize_manual_options_log_df(raw_df, source_name=sheet_csv_url)
    if std_df.empty:
        raise RuntimeError("Sheet CSV parsed but no usable realized rows were extracted.")
    std_df.to_csv(realized_out_csv, index=False)
    out_meta = dict(meta or {})
    out_meta.update(ingest_meta)
    return out_meta


def option_strategy_label(symbol: str, open_side: str) -> str:
    _, right, _ = parse_option_meta(symbol)
    if open_side == "SHORT":
        return "Short Put Option" if right == "P" else "Short Call Option"
    return "Long Put Option" if right == "P" else "Long Call Option"


def build_realized_from_schwab_api(realized_out_csv: Path, lookback_days: int = 365) -> Dict[str, float]:
    try:
        from schwab_live_service import SchwabAuthConfig, SchwabLiveDataService
    except Exception as e:
        raise RuntimeError(f"Failed to import Schwab live service: {e}") from e

    realized_out_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg_live = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(cfg_live)
    svc.connect()
    cli = svc._client

    acct_resp = cli.get_account_numbers()
    acct_resp.raise_for_status()
    acct_rows = acct_resp.json()
    account_hashes = [a.get("hashValue") for a in acct_rows if a.get("hashValue")]
    if not account_hashes:
        raise RuntimeError("No Schwab account hashes available.")

    end_dt = dt.datetime.now(dt.timezone.utc)
    start_dt = end_dt - dt.timedelta(days=max(1, int(lookback_days)))

    raw_events: List[Dict[str, Any]] = []
    for acc_hash in account_hashes:
        resp = cli.get_transactions(
            acc_hash,
            start_date=start_dt,
            end_date=end_dt,
            transaction_types=[cli.Transactions.TransactionType.TRADE],
        )
        resp.raise_for_status()
        for txn in resp.json():
            txn_time = pd.to_datetime(txn.get("time"), errors="coerce")
            if pd.isna(txn_time):
                continue
            for it in txn.get("transferItems", []) or []:
                inst = it.get("instrument", {}) or {}
                if str(inst.get("assetType", "")).upper() != "OPTION":
                    continue
                symbol = str(inst.get("symbol", "")).strip().upper()
                if not symbol:
                    continue
                pos_eff = str(it.get("positionEffect", "")).strip().upper()
                if pos_eff not in {"OPENING", "CLOSING"}:
                    continue
                qty_signed = parse_float(it.get("amount"))
                qty_abs = abs(qty_signed) if np.isfinite(qty_signed) else math.nan
                cost = parse_float(it.get("cost"))
                if (not np.isfinite(qty_abs)) or qty_abs <= 0 or (not np.isfinite(cost)):
                    continue
                raw_events.append(
                    {
                        "account_hash": acc_hash,
                        "time": txn_time,
                        "symbol": symbol,
                        "position_effect": pos_eff,
                        "qty_signed": float(qty_signed),
                        "qty_abs": float(qty_abs),
                        "cost": float(cost),
                    }
                )

    if not raw_events:
        raise RuntimeError("No option trade events returned from Schwab transactions.")

    events = sorted(raw_events, key=lambda x: x["time"])
    # FIFO lots by account+symbol and side opened.
    lots = defaultdict(lambda: {"LONG": deque(), "SHORT": deque()})
    closed_rows: List[Dict[str, Any]] = []

    for ev in events:
        acc = ev["account_hash"]
        sym = ev["symbol"]
        qty = float(ev["qty_abs"])
        qty_signed = float(ev["qty_signed"])
        cost = float(ev["cost"])
        t = pd.to_datetime(ev["time"])
        eff = ev["position_effect"]
        unit_cash = cost / max(1e-9, qty)

        if eff == "OPENING":
            open_side = "LONG" if qty_signed > 0 else "SHORT"
            lots[(acc, sym)][open_side].append(
                {
                    "qty": qty,
                    "unit_cash": unit_cash,
                    "open_date": t,
                    "open_side": open_side,
                }
            )
            continue

        # CLOSING: buy-to-close -> qty_signed>0 closes SHORT; sell-to-close -> qty_signed<0 closes LONG.
        close_side = "SHORT" if qty_signed > 0 else "LONG"
        qleft = qty
        qbook = lots[(acc, sym)][close_side]
        while qleft > 1e-9 and qbook:
            lot = qbook[0]
            m = min(qleft, float(lot["qty"]))
            realized = (float(lot["unit_cash"]) + unit_cash) * m
            underlying = option_underlying_symbol(sym) or sym
            strategy = option_strategy_label(sym, str(lot["open_side"]))
            closed_rows.append(
                {
                    "source_file": "schwab_api",
                    "date": t,
                    "open_date": pd.to_datetime(lot["open_date"]),
                    "symbol": underlying,
                    "strategy": strategy,
                    "side": str(lot["open_side"]),
                    "qty": m,
                    "entry_price": math.nan,
                    "exit_price": math.nan,
                    "fees": 0.0,
                    "dte": math.nan,
                    "conviction": math.nan,
                    "realized_pnl": realized,
                }
            )
            lot["qty"] = float(lot["qty"]) - m
            qleft -= m
            if lot["qty"] <= 1e-9:
                qbook.popleft()

    out = pd.DataFrame(closed_rows)
    if out.empty:
        raise RuntimeError("No closed option lots could be reconstructed from Schwab transaction events.")
    out = out.sort_values("date").reset_index(drop=True)
    out["holding_days"] = (pd.to_datetime(out["date"]) - pd.to_datetime(out["open_date"])).dt.days
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
    out[keep].to_csv(realized_out_csv, index=False)
    return {
        "schwab_accounts": float(len(account_hashes)),
        "schwab_option_events": float(len(events)),
        "schwab_closed_lots": float(len(out)),
    }


def infer_strategy_from_text(
    strategy: str,
    description: str,
    side: str,
    symbol: str = "",
    asset_type: str = "",
    net_quantity: float = math.nan,
) -> str:
    s = f"{strategy} {description} {side} {symbol} {asset_type}".upper()
    _, right, _ = parse_option_meta(symbol)
    has_put = ("PUT" in s) or bool(re.search(r"\bP\b", s)) or (right == "P")
    has_call = ("CALL" in s) or bool(re.search(r"\bC\b", s)) or (right == "C")
    is_short = ("SHORT" in s) or ("SELL" in s) or ("WRITE" in s)
    is_long = ("LONG" in s) or ("BUY" in s)
    if np.isfinite(net_quantity):
        if float(net_quantity) < 0:
            is_short = True
        elif float(net_quantity) > 0:
            is_long = True
    is_option = ("OPTION" in s) or (right in {"C", "P"}) or has_put or has_call
    if is_short and has_put:
        return "Short Put Option"
    if is_short and has_call:
        return "Short Call Option"
    if is_long and has_put:
        return "Long Put Option"
    if is_long and has_call:
        return "Long Call Option"
    if is_option and is_short:
        return "Short Option"
    if is_option and is_long:
        return "Long Option"
    if "EQUITY" in s and is_long:
        return "Long Stock"
    if "EQUITY" in s and is_short:
        return "Short Stock"
    if is_short:
        return "Short"
    if is_long:
        return "Long"
    return "Unknown"


def load_open_positions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = list(df.columns)
    symbol_col = find_col(cols, ["symbol", "ticker", "underlying"])
    strategy_col = find_col(cols, ["strategy", "strategy_type", "position_type"])
    side_col = find_col(cols, ["side", "action", "direction"])
    desc_col = find_col(cols, ["description", "instrument", "name"])
    expiry_col = find_col(cols, ["expiry", "expiration", "exp_date", "expiration_date"])
    dte_col = find_col(cols, ["dte", "days_to_expiry", "days_to_expiration", "days_left"])
    unreal_col = find_col(cols, ["unrealized_pnl", "unrealized_pl", "open_pnl", "position_pnl"])
    day_pl_col = find_col(cols, ["current_day_profit_loss", "day_pnl", "daily_pnl"])
    avg_price_col = find_col(cols, ["average_price", "avg_price", "average_cost", "avg_cost"])
    market_value_col = find_col(cols, ["market_value", "position_value", "marketvalue"])
    long_qty_col = find_col(cols, ["long_quantity", "long_qty", "quantity_long"])
    short_qty_col = find_col(cols, ["short_quantity", "short_qty", "quantity_short"])
    qty_col = find_col(cols, ["quantity", "qty", "net_quantity"])
    asset_type_col = find_col(cols, ["asset_type", "instrument_type", "security_type"])
    max_profit_col = find_col(cols, ["max_profit", "profit_cap", "target_profit"])
    risk_col = find_col(
        cols,
        [
            "max_loss",
            "risk",
            "risk_amount",
            "max_risk",
            "maintenance_requirement",
            "buying_power_effect",
            "bp_effect",
            "margin_requirement",
        ],
    )

    out = pd.DataFrame(index=df.index)
    out["symbol"] = df[symbol_col].astype(str).str.strip().str.upper() if symbol_col else "UNKNOWN"
    out["strategy_raw"] = df[strategy_col].astype(str).str.strip() if strategy_col else ""
    out["description"] = df[desc_col].astype(str).str.strip() if desc_col else ""
    out["side_raw"] = df[side_col].astype(str).str.strip() if side_col else ""
    out["expiry"] = pd.to_datetime(df[expiry_col], errors="coerce") if expiry_col else pd.NaT
    out["dte"] = df[dte_col].map(parse_float) if dte_col else math.nan
    out["asset_type"] = df[asset_type_col].astype(str).str.upper().str.strip() if asset_type_col else ""
    out["long_quantity"] = df[long_qty_col].map(parse_float).fillna(0.0) if long_qty_col else 0.0
    out["short_quantity"] = df[short_qty_col].map(parse_float).fillna(0.0) if short_qty_col else 0.0
    out["raw_quantity"] = df[qty_col].map(parse_float) if qty_col else math.nan
    out["net_quantity"] = out["raw_quantity"]
    nq_fallback = out["long_quantity"] - out["short_quantity"]
    out.loc[out["net_quantity"].isna(), "net_quantity"] = nq_fallback[out["net_quantity"].isna()]
    out["average_price"] = df[avg_price_col].map(parse_float) if avg_price_col else math.nan
    out["market_value"] = df[market_value_col].map(parse_float) if market_value_col else math.nan
    out["unrealized_pnl"] = df[unreal_col].map(parse_float) if unreal_col else math.nan
    out["current_day_profit_loss"] = df[day_pl_col].map(parse_float) if day_pl_col else math.nan
    out["max_profit"] = df[max_profit_col].map(parse_float).abs() if max_profit_col else math.nan
    out["risk"] = df[risk_col].map(parse_float).abs() if risk_col else math.nan
    out["underlying"] = out["symbol"].map(option_underlying_symbol)
    out.loc[out["underlying"].isna() | (out["underlying"] == ""), "underlying"] = out.loc[
        out["underlying"].isna() | (out["underlying"] == ""), "symbol"
    ].astype(str)
    out["strategy"] = [
        infer_strategy_from_text(a, b, c, d, e, f)
        for a, b, c, d, e, f in zip(
            out["strategy_raw"],
            out["description"],
            out["side_raw"],
            out["symbol"],
            out["asset_type"],
            out["net_quantity"],
        )
    ]
    out["symbol"] = out["symbol"].replace({"": "UNKNOWN", "NAN": "UNKNOWN"})

    # Estimate unrealized P/L when not explicitly present.
    mult = np.where(out["asset_type"].str.contains("OPTION", na=False), 100.0, 1.0)
    abs_qty = out["net_quantity"].abs()
    cost_basis_abs = out["average_price"] * abs_qty * mult
    can_est = out["unrealized_pnl"].isna() & out["average_price"].notna() & out["market_value"].notna() & abs_qty.gt(0)
    long_mask = can_est & out["net_quantity"].gt(0)
    short_mask = can_est & out["net_quantity"].lt(0)
    out.loc[long_mask, "unrealized_pnl"] = out.loc[long_mask, "market_value"] - cost_basis_abs[long_mask]
    out.loc[short_mask, "unrealized_pnl"] = cost_basis_abs[short_mask] + out.loc[short_mask, "market_value"]
    out["unrealized_source"] = np.where(
        out["unrealized_pnl"].notna(),
        np.where(df[unreal_col].map(parse_float).notna() if unreal_col else False, "explicit", "estimated"),
        np.where(out["current_day_profit_loss"].notna(), "day_only", "missing"),
    )
    out.loc[out["unrealized_pnl"].isna() & out["current_day_profit_loss"].notna(), "unrealized_pnl"] = out.loc[
        out["unrealized_pnl"].isna() & out["current_day_profit_loss"].notna(), "current_day_profit_loss"
    ]
    return out


def parse_option_meta(symbol: str) -> Tuple[Optional[dt.date], Optional[str], Optional[float]]:
    text = str(symbol or "").upper()
    m = SCHWAB_OPTION_SYMBOL_RE.match(text)
    if m:
        _, yymmdd, right, strike8 = m.groups()
    else:
        m2 = COMPACT_OPTION_SYMBOL_RE.match(text.replace(" ", ""))
        if not m2:
            return None, None, None
        _, yymmdd, right, strike8 = m2.groups()
    try:
        expiry = dt.datetime.strptime(yymmdd, "%y%m%d").date()
    except Exception:
        expiry = None
    try:
        strike = float(int(strike8)) / 1000.0
    except Exception:
        strike = None
    return expiry, right, strike


def option_underlying_symbol(symbol: str) -> str:
    text = str(symbol or "").upper()
    m = SCHWAB_OPTION_SYMBOL_RE.match(text)
    if m:
        return m.group(1).strip()
    m2 = COMPACT_OPTION_SYMBOL_RE.match(text.replace(" ", ""))
    if m2:
        return m2.group(1).strip()
    return ""


def build_position_decisions(pos: pd.DataFrame, as_of_date: dt.date, cfg: Dict) -> pd.DataFrame:
    if pos.empty:
        return pd.DataFrame()

    pm = cfg.get("playbook", {}).get("position_management", {})
    take_profit_pct = float(pm.get("take_profit_pct_of_max_profit", 0.60))
    stop_loss_pct = float(pm.get("stop_loss_pct_of_risk", 0.50))
    short_roll_dte = int(pm.get("short_roll_dte_threshold", 10))
    long_exit_dte = int(pm.get("long_option_exit_dte_threshold", 7))

    rows: List[Dict[str, Any]] = []
    for r in pos.itertuples(index=False):
        sym = str(getattr(r, "symbol", "UNKNOWN")).strip().upper() or "UNKNOWN"
        strategy = str(getattr(r, "strategy", "Unknown")).strip() or "Unknown"
        expiry = getattr(r, "expiry", pd.NaT)
        dte = parse_float(getattr(r, "dte", math.nan))
        if (not np.isfinite(dte)) and pd.notna(expiry):
            try:
                dte = float((pd.to_datetime(expiry).date() - as_of_date).days)
            except Exception:
                dte = math.nan

        # Derive option expiry when absent from separate field.
        if pd.isna(expiry):
            exp2, _, _ = parse_option_meta(sym)
            if exp2 is not None:
                expiry = pd.Timestamp(exp2)
                if not np.isfinite(dte):
                    dte = float((exp2 - as_of_date).days)

        unreal = parse_float(getattr(r, "unrealized_pnl", math.nan))
        max_profit = parse_float(getattr(r, "max_profit", math.nan))
        risk = parse_float(getattr(r, "risk", math.nan))
        unreal_source = str(getattr(r, "unrealized_source", "")).strip() or "missing"

        profit_capture = (unreal / max_profit) if (np.isfinite(unreal) and np.isfinite(max_profit) and max_profit > 0) else math.nan
        risk_draw = ((-unreal) / risk) if (np.isfinite(unreal) and np.isfinite(risk) and risk > 0 and unreal < 0) else 0.0

        rec = "HOLD"
        reason = "No close/roll trigger met."
        priority = 9

        if unreal_source == "day_only":
            rec = "DATA_NEEDED"
            reason = "Only day P/L available; need full unrealized P/L for reliable close/roll."
            priority = 0
        elif not np.isfinite(unreal):
            rec = "DATA_NEEDED"
            reason = "Missing unrealized P/L; cannot score recoverability."
            priority = 0
        elif np.isfinite(profit_capture) and profit_capture >= take_profit_pct:
            rec = "CLOSE_PROFIT"
            reason = f"Profit capture {profit_capture:.0%} >= {take_profit_pct:.0%} of max profit."
            priority = 1
        elif np.isfinite(risk_draw) and risk_draw >= stop_loss_pct:
            rec = "CLOSE_RISK"
            reason = f"Loss uses {risk_draw:.0%} of risk budget (>= {stop_loss_pct:.0%})."
            priority = 2
        elif strategy.startswith("Short") and np.isfinite(dte) and dte <= short_roll_dte:
            if unreal >= 0:
                rec = "ROLL_OR_CLOSE"
                reason = f"DTE {int(dte)} <= {short_roll_dte}; lock gains or roll out."
            else:
                rec = "ROLL_DEFENSIVE"
                reason = f"DTE {int(dte)} <= {short_roll_dte} with loss; roll out/down to extend recovery window."
            priority = 3
        elif strategy in {"Long Call Option", "Long Put Option"} and np.isfinite(dte) and dte <= long_exit_dte and unreal <= 0:
            rec = "CLOSE_DECAY"
            reason = f"Long option near expiry (DTE {int(dte)} <= {long_exit_dte}) with non-positive P/L."
            priority = 4

        action = {
            "CLOSE_PROFIT": "🟢 CLOSE (profit)",
            "CLOSE_RISK": "🔴 CLOSE (risk)",
            "ROLL_OR_CLOSE": "🟠 ROLL or CLOSE",
            "ROLL_DEFENSIVE": "🟠 ROLL DEFENSIVE",
            "CLOSE_DECAY": "🟡 CLOSE (decay)",
            "HOLD": "⚪ HOLD",
            "DATA_NEEDED": "⚠️ DATA NEEDED",
        }.get(rec, rec)

        rows.append(
            {
                "underlying": str(getattr(r, "underlying", "")).strip().upper() or sym,
                "symbol": sym,
                "strategy": strategy,
                "expiry": str(pd.to_datetime(expiry).date()) if pd.notna(expiry) else "",
                "dte": int(dte) if np.isfinite(dte) else math.nan,
                "unrealized_pnl": unreal if np.isfinite(unreal) else math.nan,
                "risk": risk if np.isfinite(risk) else math.nan,
                "profit_capture_pct": profit_capture if np.isfinite(profit_capture) else math.nan,
                "risk_draw_pct": risk_draw if np.isfinite(risk_draw) else math.nan,
                "recommendation": rec,
                "action": action,
                "reason": reason,
                "unrealized_source": unreal_source,
                "_priority": priority,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["_priority", "underlying", "dte"], ascending=[True, True, True]).reset_index(drop=True)
    return out.drop(columns=["_priority"])


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def to_md_table(df: pd.DataFrame, n: int = 30) -> str:
    if df.empty:
        return "_none_"
    return df.head(n).to_markdown(index=False)


def run_daily(
    trades: pd.DataFrame,
    out_dir: Path,
    cfg: Dict,
    open_positions_csv: Optional[Path],
    lookback_trades: int,
) -> Dict:
    pb = cfg.get("playbook", {})
    daily_cfg = pb.get("daily", {})
    risk_cfg = pb.get("risk_limits", {})

    daily_loss_stop_cash = float(daily_cfg.get("daily_loss_stop_cash", 1000.0))
    streak_yellow = int(daily_cfg.get("loss_streak_yellow", 3))
    streak_red = int(daily_cfg.get("loss_streak_red", 4))
    min_pf_yellow = float(daily_cfg.get("rolling_pf_yellow_floor", 1.0))
    short_put_limit = float(risk_cfg.get("short_put_max_share", 0.35))
    symbol_limit = float(risk_cfg.get("single_symbol_max_share", 0.10))
    expiry_limit = float(risk_cfg.get("single_expiry_max_share_short_put", 0.25))

    x = trades.sort_values("date").reset_index(drop=True)
    recent = x.tail(max(1, int(lookback_trades))).copy()
    last_day = x["date"].dt.date.max()
    day_df = x[x["date"].dt.date == last_day].copy()

    day_pnl = float(day_df["realized_pnl"].sum()) if not day_df.empty else 0.0
    recent_pf = compute_profit_factor(recent["realized_pnl"])
    recent_net = float(recent["realized_pnl"].sum())
    cur_loss_streak = trailing_loss_streak(x["realized_pnl"])

    status = "GREEN"
    severity = 0
    triggers: List[str] = []
    actions: List[str] = []

    def escalate(level: str, trigger: str, action: str) -> None:
        nonlocal status, severity
        rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}[level]
        if rank > severity:
            severity = rank
            status = level
        triggers.append(trigger)
        actions.append(action)

    if day_pnl <= -abs(daily_loss_stop_cash):
        escalate(
            "RED",
            f"Daily realized P/L {day_pnl:,.2f} <= stop {-abs(daily_loss_stop_cash):,.2f}",
            "Stop opening new risk today; only reduce or hedge open risk.",
        )
    if cur_loss_streak >= streak_red:
        escalate(
            "RED",
            f"Trailing loss streak is {cur_loss_streak} (red threshold {streak_red})",
            "Cut size by 50% and skip the next 2 non-core setups.",
        )
    elif cur_loss_streak >= streak_yellow:
        escalate(
            "YELLOW",
            f"Trailing loss streak is {cur_loss_streak} (yellow threshold {streak_yellow})",
            "Reduce new position size by 25% until streak resets.",
        )
    if np.isfinite(recent_pf) and recent_pf < min_pf_yellow:
        escalate(
            "YELLOW",
            f"Recent {len(recent)}-trade PF is {recent_pf:.2f} < {min_pf_yellow:.2f}",
            "Restrict entries to top-conviction setups only this week.",
        )

    pos_summary: Dict[str, float] = {}
    symbol_risk_table = pd.DataFrame()
    expiry_risk_table = pd.DataFrame()
    position_decisions = pd.DataFrame()
    if open_positions_csv and open_positions_csv.exists():
        pos = load_open_positions(open_positions_csv)
        options_only = bool(daily_cfg.get("options_only", True))
        if options_only:
            is_option = (
                pos["asset_type"].astype(str).str.upper().eq("OPTION")
                | pos["strategy"].astype(str).str.contains("Option", case=False, na=False)
                | pos["symbol"].astype(str).str.contains(r"\d{6}[CP]\d{8}", na=False)
            )
            pos = pos[is_option].copy()
        position_decisions = build_position_decisions(pos.copy(), last_day, cfg)
        pos = pos[pos["risk"].notna() & (pos["risk"] > 0)].copy()
        if not pos.empty:
            total_risk = float(pos["risk"].sum())
            short_put_risk = float(pos.loc[pos["strategy"] == "Short Put Option", "risk"].sum())
            short_put_share = short_put_risk / total_risk if total_risk > 0 else 0.0
            pos_summary = {
                "total_open_risk": total_risk,
                "short_put_risk": short_put_risk,
                "short_put_risk_share": short_put_share,
            }

            symbol_risk_table = (
                pos.groupby("symbol", dropna=False)["risk"]
                .sum()
                .reset_index(name="risk")
                .sort_values("risk", ascending=False)
                .reset_index(drop=True)
            )
            symbol_risk_table["risk_share"] = symbol_risk_table["risk"] / max(1e-9, total_risk)

            if short_put_share > short_put_limit:
                escalate(
                    "RED",
                    f"Short-put risk share {short_put_share:.1%} > limit {short_put_limit:.1%}",
                    "Do not add short puts; trim/roll largest short-put exposure first.",
                )

            sym_breach = symbol_risk_table[symbol_risk_table["risk_share"] > symbol_limit]
            if not sym_breach.empty:
                names = ", ".join(sym_breach["symbol"].head(5).tolist())
                escalate(
                    "YELLOW",
                    f"Single-symbol risk concentration above {symbol_limit:.1%}: {names}",
                    "Reduce concentrated ticker risk before adding same-ticker trades.",
                )

            short_put_pos = pos[pos["strategy"] == "Short Put Option"].copy()
            short_put_pos = short_put_pos[short_put_pos["expiry"].notna()].copy()
            if not short_put_pos.empty and short_put_risk > 0:
                expiry_risk_table = (
                    short_put_pos.groupby(short_put_pos["expiry"].dt.date)["risk"]
                    .sum()
                    .reset_index(name="risk")
                    .sort_values("risk", ascending=False)
                    .reset_index(drop=True)
                )
                expiry_risk_table["risk_share"] = expiry_risk_table["risk"] / max(1e-9, short_put_risk)
                exp_breach = expiry_risk_table[expiry_risk_table["risk_share"] > expiry_limit]
                if not exp_breach.empty:
                    exp = str(exp_breach.iloc[0, 0])
                    share = float(exp_breach.iloc[0]["risk_share"])
                    escalate(
                        "YELLOW",
                        f"Short-put expiry concentration {share:.1%} on {exp} > {expiry_limit:.1%}",
                        "Spread short-put expiries; avoid stacking same expiration risk.",
                    )
    else:
        actions.append(
            "Close/roll analysis unavailable: provide --open-positions-csv with unrealized and risk fields."
        )

    payload = {
        "mode": "daily",
        "status": status,
        "as_of_trade_date": str(last_day),
        "day_realized_pnl": day_pnl,
        "recent_trade_count": int(len(recent)),
        "recent_net_pnl": recent_net,
        "recent_profit_factor": recent_pf,
        "trailing_loss_streak": int(cur_loss_streak),
        "thresholds": {
            "daily_loss_stop_cash": daily_loss_stop_cash,
            "loss_streak_yellow": streak_yellow,
            "loss_streak_red": streak_red,
            "rolling_pf_yellow_floor": min_pf_yellow,
            "short_put_max_share": short_put_limit,
            "single_symbol_max_share": symbol_limit,
            "single_expiry_max_share_short_put": expiry_limit,
        },
        "triggers": triggers,
        "actions": actions,
        "open_position_risk_summary": pos_summary,
        "position_decision_counts": (
            position_decisions["recommendation"].value_counts().to_dict() if not position_decisions.empty else {}
        ),
    }
    write_json(out_dir / "daily_risk_monitor.json", payload)

    lines = [
        "# Daily Risk Monitor",
        f"Status: **{status}**",
        "",
        "## Snapshot",
        f"- As-of trade date: {last_day}",
        f"- Day realized P/L: {day_pnl:,.2f}",
        f"- Recent trades checked: {len(recent)}",
        f"- Recent net P/L: {recent_net:,.2f}",
        f"- Recent PF: {recent_pf:.2f}" if np.isfinite(recent_pf) else "- Recent PF: n/a",
        f"- Trailing loss streak: {cur_loss_streak}",
        "",
        "## Triggers",
    ]
    if triggers:
        lines.extend([f"- {t}" for t in triggers])
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Actions")
    if actions:
        lines.extend([f"- {a}" for a in actions])
    else:
        lines.append("- Continue normal sizing under current limits.")
    lines.append("")
    if pos_summary:
        lines.append("## Open Risk")
        lines.append(
            f"- Total open risk: {pos_summary.get('total_open_risk', 0.0):,.2f} | "
            f"Short put share: {pos_summary.get('short_put_risk_share', 0.0):.1%}"
        )
        lines.append("")
        lines.append("### Symbol Concentration")
        lines.append(to_md_table(symbol_risk_table[["symbol", "risk", "risk_share"]], n=20))
        lines.append("")
        lines.append("### Short Put Expiry Concentration")
        if not expiry_risk_table.empty:
            lines.append(to_md_table(expiry_risk_table, n=20))
        else:
            lines.append("_none_")
        lines.append("")
    if not position_decisions.empty:
        lines.append("## Position Decisions")
        lines.append("- Recommendations prioritize locking profits, cutting risk, and rolling short positions near expiry.")
        lines.append("")
        lines.append(
            to_md_table(
                position_decisions[
                    [
                        "underlying",
                        "symbol",
                        "strategy",
                        "expiry",
                        "dte",
                        "action",
                        "reason",
                    ]
                ],
                n=150,
            )
        )
        lines.append("")
    write_text(out_dir / "daily_risk_monitor.md", "\n".join(lines))
    return payload


def aggregate_edge(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = (
        df.groupby(group_cols, dropna=False)
        .agg(
            trades=("realized_pnl", "size"),
            win_rate=("win", "mean"),
            net_pnl=("realized_pnl", "sum"),
            avg_pnl=("realized_pnl", "mean"),
            profit_factor=("realized_pnl", compute_profit_factor),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
        .reset_index(drop=True)
    )
    return g


def run_weekly(trades: pd.DataFrame, out_dir: Path, cfg: Dict, as_of: Optional[pd.Timestamp]) -> Dict:
    pb = cfg.get("playbook", {})
    wk = pb.get("weekly", {})
    strategy_window = int(wk.get("strategy_window_trades", 20))
    strategy_min_trades = int(wk.get("strategy_min_trades", 20))
    strategy_min_pf = float(wk.get("strategy_min_pf", 1.0))
    symbol_window = int(wk.get("symbol_window_trades", 20))
    symbol_min_trades = int(wk.get("symbol_min_trades", 5))
    symbol_pause_net = float(wk.get("symbol_pause_if_net_below", 0.0))
    combo_min_trades = int(wk.get("combo_min_trades", 3))
    combo_pause_net = float(wk.get("combo_pause_if_net_below", -1000.0))

    x = trades.sort_values("date").reset_index(drop=True)
    if as_of is not None:
        x = x[x["date"] <= as_of].copy()

    strategy_rows = []
    for strategy, g in x.groupby("strategy", dropna=False):
        t = g.tail(max(1, strategy_window)).copy()
        trades_n = int(len(t))
        pf = compute_profit_factor(t["realized_pnl"])
        net = float(t["realized_pnl"].sum())
        wr = float((t["realized_pnl"] > 0).mean())
        action = "KEEP"
        reason = ""
        if trades_n >= strategy_min_trades and np.isfinite(pf) and pf < strategy_min_pf:
            action = "PAUSE"
            reason = f"PF {pf:.2f} < {strategy_min_pf:.2f}"
        strategy_rows.append(
            {
                "strategy": strategy,
                "window_trades": trades_n,
                "window_win_rate": wr,
                "window_net_pnl": net,
                "window_profit_factor": pf,
                "action": action,
                "reason": reason,
            }
        )
    strategy_actions = pd.DataFrame(strategy_rows).sort_values(["action", "window_net_pnl"], ascending=[True, False])

    symbol_rows = []
    for symbol, g in x.groupby("symbol", dropna=False):
        t = g.tail(max(1, symbol_window)).copy()
        trades_n = int(len(t))
        net = float(t["realized_pnl"].sum())
        pf = compute_profit_factor(t["realized_pnl"])
        wr = float((t["realized_pnl"] > 0).mean())
        action = "KEEP"
        reason = ""
        if trades_n >= symbol_min_trades and net <= symbol_pause_net:
            action = "PAUSE"
            reason = f"Net {net:,.2f} <= {symbol_pause_net:,.2f}"
        symbol_rows.append(
            {
                "symbol": symbol,
                "window_trades": trades_n,
                "window_win_rate": wr,
                "window_net_pnl": net,
                "window_profit_factor": pf,
                "action": action,
                "reason": reason,
            }
        )
    symbol_actions = pd.DataFrame(symbol_rows).sort_values(["action", "window_net_pnl"], ascending=[True, False])

    combo_stats = aggregate_edge(x, ["strategy", "symbol"])
    combo_actions = combo_stats.copy()
    if not combo_actions.empty:
        combo_actions["action"] = "KEEP"
        combo_actions["reason"] = ""
        bad = (combo_actions["trades"] >= combo_min_trades) & (combo_actions["net_pnl"] <= combo_pause_net)
        combo_actions.loc[bad, "action"] = "PAUSE"
        combo_actions.loc[bad, "reason"] = combo_actions.loc[bad, "net_pnl"].map(
            lambda v: f"net_pnl {v:,.2f} <= {combo_pause_net:,.2f}"
        )
        combo_actions = combo_actions.sort_values(["action", "net_pnl"], ascending=[True, False]).reset_index(drop=True)

    status = "GREEN"
    if (strategy_actions["action"] == "PAUSE").any() or (symbol_actions["action"] == "PAUSE").any():
        status = "YELLOW"
    if not combo_actions.empty and (combo_actions["action"] == "PAUSE").any():
        status = "RED"

    payload = {
        "mode": "weekly",
        "status": status,
        "as_of_date": str(x["date"].max().date()) if not x.empty else "",
        "strategy_pauses": int((strategy_actions["action"] == "PAUSE").sum()),
        "symbol_pauses": int((symbol_actions["action"] == "PAUSE").sum()),
        "combo_pauses": int((combo_actions["action"] == "PAUSE").sum()) if not combo_actions.empty else 0,
        "thresholds": {
            "strategy_window_trades": strategy_window,
            "strategy_min_trades": strategy_min_trades,
            "strategy_min_pf": strategy_min_pf,
            "symbol_window_trades": symbol_window,
            "symbol_min_trades": symbol_min_trades,
            "symbol_pause_if_net_below": symbol_pause_net,
            "combo_min_trades": combo_min_trades,
            "combo_pause_if_net_below": combo_pause_net,
        },
    }
    write_json(out_dir / "weekly_edge_report.json", payload)
    strategy_actions.to_csv(out_dir / "weekly_strategy_actions.csv", index=False)
    symbol_actions.to_csv(out_dir / "weekly_symbol_actions.csv", index=False)
    if not combo_actions.empty:
        combo_actions.to_csv(out_dir / "weekly_combo_actions.csv", index=False)

    lines = [
        "# Weekly Edge Report",
        f"Status: **{status}**",
        "",
        "## Pause Summary",
        f"- Strategy pauses: {payload['strategy_pauses']}",
        f"- Symbol pauses: {payload['symbol_pauses']}",
        f"- Strategy-symbol combo pauses: {payload['combo_pauses']}",
        "",
        "## Strategy Actions",
        to_md_table(strategy_actions, n=50),
        "",
        "## Symbol Actions",
        to_md_table(symbol_actions, n=50),
        "",
    ]
    if not combo_actions.empty:
        lines.extend(
            [
                "## Strategy-Symbol Actions",
                to_md_table(combo_actions, n=100),
                "",
            ]
        )
    write_text(out_dir / "weekly_edge_report.md", "\n".join(lines))
    return payload


def run_monthly(trades: pd.DataFrame, out_dir: Path, cfg: Dict) -> Dict:
    pb = cfg.get("playbook", {})
    mo = pb.get("monthly", {})
    target_monthly = float(mo.get("target_monthly_profit", 20000.0))
    long_call_kill = float(mo.get("long_call_monthly_kill_switch", -1500.0))

    x = trades.sort_values("date").reset_index(drop=True)
    monthly = (
        x.groupby("month", dropna=False)
        .agg(
            trades=("realized_pnl", "size"),
            win_rate=("win", "mean"),
            net_pnl=("realized_pnl", "sum"),
            avg_pnl=("realized_pnl", "mean"),
            profit_factor=("realized_pnl", compute_profit_factor),
        )
        .reset_index()
        .sort_values("month")
        .reset_index(drop=True)
    )
    monthly["rolling_3m_net"] = monthly["net_pnl"].rolling(3, min_periods=1).mean()
    monthly["rolling_3m_pf"] = monthly["profit_factor"].rolling(3, min_periods=1).mean()

    avg_monthly = float(monthly["net_pnl"].mean()) if not monthly.empty else math.nan
    med_monthly = float(monthly["net_pnl"].median()) if not monthly.empty else math.nan
    last_3m_avg = float(monthly["net_pnl"].tail(3).mean()) if not monthly.empty else math.nan
    monthly_gap = target_monthly - avg_monthly if np.isfinite(avg_monthly) else math.nan
    multiplier = target_monthly / avg_monthly if np.isfinite(avg_monthly) and avg_monthly > 0 else math.nan

    by_strategy = aggregate_edge(x, ["strategy"])
    by_symbol = aggregate_edge(x, ["symbol"])
    by_strategy_month = (
        x.groupby(["month", "strategy"], dropna=False)["realized_pnl"]
        .sum()
        .reset_index(name="net_pnl")
        .sort_values(["month", "net_pnl"], ascending=[True, False])
        .reset_index(drop=True)
    )

    long_call_month = (
        x[x["strategy"] == "Long Call Option"]
        .groupby("month", dropna=False)["realized_pnl"]
        .sum()
        .reset_index(name="long_call_net")
        .sort_values("month")
    )
    long_call_breach = long_call_month[long_call_month["long_call_net"] <= long_call_kill].copy()

    top_losses = x.sort_values("realized_pnl").head(20).copy()

    payload = {
        "mode": "monthly",
        "as_of_date": str(x["date"].max().date()) if not x.empty else "",
        "months_analyzed": int(len(monthly)),
        "avg_monthly_net_pnl": avg_monthly,
        "median_monthly_net_pnl": med_monthly,
        "last_3m_avg_net_pnl": last_3m_avg,
        "target_monthly_profit": target_monthly,
        "gap_to_target": monthly_gap,
        "required_multiplier_to_target": multiplier,
        "long_call_kill_switch_threshold": long_call_kill,
        "long_call_breach_months": long_call_breach["month"].tolist(),
    }
    write_json(out_dir / "monthly_longitudinal_review.json", payload)
    monthly.to_csv(out_dir / "monthly_pnl_trend.csv", index=False)
    by_strategy.to_csv(out_dir / "monthly_strategy_edge.csv", index=False)
    by_symbol.to_csv(out_dir / "monthly_symbol_edge.csv", index=False)
    by_strategy_month.to_csv(out_dir / "monthly_strategy_month_matrix.csv", index=False)
    long_call_month.to_csv(out_dir / "monthly_long_call_net.csv", index=False)
    top_losses.to_csv(out_dir / "monthly_top_losses.csv", index=False)

    lines = [
        "# Monthly Longitudinal Review",
        "",
        "## Performance Baseline",
        f"- Months analyzed: {payload['months_analyzed']}",
        f"- Average monthly net P/L: {avg_monthly:,.2f}" if np.isfinite(avg_monthly) else "- Average monthly net P/L: n/a",
        f"- Median monthly net P/L: {med_monthly:,.2f}" if np.isfinite(med_monthly) else "- Median monthly net P/L: n/a",
        f"- Last 3-month avg net P/L: {last_3m_avg:,.2f}" if np.isfinite(last_3m_avg) else "- Last 3-month avg net P/L: n/a",
        f"- Target monthly net P/L: {target_monthly:,.2f}",
        f"- Gap to target: {monthly_gap:,.2f}" if np.isfinite(monthly_gap) else "- Gap to target: n/a",
        f"- Required multiplier to target: {multiplier:.2f}x" if np.isfinite(multiplier) else "- Required multiplier to target: n/a",
        "",
        "## Long Call Kill-Switch Check",
        f"- Threshold: monthly long-call net <= {long_call_kill:,.2f}",
    ]
    if not long_call_breach.empty:
        lines.append("- Breach months: " + ", ".join(long_call_breach["month"].tolist()))
    else:
        lines.append("- Breach months: none")
    lines.extend(
        [
            "",
            "## Monthly Net Trend",
            to_md_table(monthly, n=36),
            "",
            "## Strategy Edge",
            to_md_table(by_strategy, n=30),
            "",
            "## Symbol Edge",
            to_md_table(by_symbol, n=30),
            "",
            "## Largest Losses",
            to_md_table(top_losses[["date", "symbol", "strategy", "realized_pnl"]], n=20),
            "",
        ]
    )
    write_text(out_dir / "monthly_longitudinal_review.md", "\n".join(lines))
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Trade playbook pipeline: daily risk, weekly edge, monthly review.")
    ap.add_argument("mode", choices=["daily", "weekly", "monthly", "all"])
    ap.add_argument(
        "--realized-source",
        choices=["csv", "sheet", "schwab"],
        default="csv",
        help="Source of realized trade history.",
    )
    ap.add_argument(
        "--realized-csv",
        default=r"c:\uw_root\out\trade_performance_review_manual_options_full\cleaned_realized_trades.csv",
        help="CSV with realized trades (date, strategy, symbol, realized_pnl).",
    )
    ap.add_argument(
        "--sheet-csv-url",
        default="",
        help="Optional Google Sheet CSV export URL. If provided, pipeline pulls this URL and auto-builds realized CSV.",
    )
    ap.add_argument(
        "--sheet-cache-csv",
        default=DEFAULT_SHEET_CACHE_CSV,
        help="Where to cache pulled raw sheet CSV when --sheet-csv-url is used.",
    )
    ap.add_argument(
        "--sheet-realized-csv",
        default=DEFAULT_SHEET_REALIZED_CSV,
        help="Where to write standardized realized CSV when --sheet-csv-url is used.",
    )
    ap.add_argument(
        "--sheet-row-filter",
        choices=["all", "yellow"],
        default="all",
        help="When using --sheet-csv-url, optionally keep only yellow-highlighted rows (plus month header rows).",
    )
    ap.add_argument(
        "--open-positions-csv",
        default="",
        help="Optional open positions CSV for daily risk concentration checks.",
    )
    ap.add_argument(
        "--open-positions-source",
        choices=["csv", "schwab"],
        default="csv",
        help="Source for open positions in daily mode. 'schwab' pulls live positions from Schwab API.",
    )
    ap.add_argument(
        "--open-positions-cache-csv",
        default="",
        help="When --open-positions-source schwab, optional path to cache pulled positions CSV.",
    )
    ap.add_argument("--config", default=str((Path(__file__).resolve().parent / "rulebook_config.yaml")), help="YAML config path.")
    ap.add_argument("--out-dir", default=r"c:\uw_root\out\playbook", help="Output directory.")
    ap.add_argument(
        "--schwab-lookback-days",
        type=int,
        default=365,
        help="When --realized-source schwab, number of days of trade history to pull.",
    )
    ap.add_argument(
        "--schwab-realized-cache-csv",
        default="",
        help="When --realized-source schwab, optional path to cache reconstructed realized options trades.",
    )
    ap.add_argument("--lookback-trades", type=int, default=30, help="Recent-trade lookback for daily checks.")
    ap.add_argument("--start-date", default="", help="Optional filter start date YYYY-MM-DD.")
    ap.add_argument("--end-date", default="", help="Optional filter end date YYYY-MM-DD.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet_meta: Dict[str, float] = {}
    if args.realized_source == "sheet":
        if not args.sheet_csv_url:
            raise RuntimeError("--realized-source sheet requires --sheet-csv-url.")
        raw_cache_csv = (
            out_dir / "source_sheet_latest.csv"
            if args.sheet_cache_csv == DEFAULT_SHEET_CACHE_CSV
            else Path(args.sheet_cache_csv).resolve()
        )
        realized_csv = (
            out_dir / "cleaned_realized_trades_from_sheet.csv"
            if args.sheet_realized_csv == DEFAULT_SHEET_REALIZED_CSV
            else Path(args.sheet_realized_csv).resolve()
        )
        sheet_meta = build_realized_from_sheet_csv_url(
            args.sheet_csv_url,
            raw_cache_csv,
            realized_csv,
            row_filter=args.sheet_row_filter,
        )
    elif args.realized_source == "schwab":
        realized_csv = (
            Path(args.schwab_realized_cache_csv).resolve()
            if args.schwab_realized_cache_csv
            else (out_dir / "cleaned_realized_trades_from_schwab.csv")
        )
        sheet_meta = build_realized_from_schwab_api(realized_csv, lookback_days=int(args.schwab_lookback_days))
    else:
        realized_csv = Path(args.realized_csv).resolve()
        if not realized_csv.exists():
            raise FileNotFoundError(f"Missing realized CSV: {realized_csv}")

    cfg = load_config(Path(args.config).resolve())
    trades = load_realized_trades(realized_csv)
    if args.start_date:
        sd = dt.date.fromisoformat(args.start_date)
        trades = trades[trades["date"].dt.date >= sd].copy()
    if args.end_date:
        ed = dt.date.fromisoformat(args.end_date)
        trades = trades[trades["date"].dt.date <= ed].copy()
    if trades.empty:
        raise RuntimeError("No trades after filters.")

    daily = None
    weekly = None
    monthly = None
    open_positions_csv: Optional[Path] = Path(args.open_positions_csv).resolve() if args.open_positions_csv else None

    if args.mode in {"daily", "all"} and args.open_positions_source == "schwab":
        try:
            from schwab_live_service import SchwabAuthConfig, SchwabLiveDataService
        except Exception as e:
            raise RuntimeError(f"Failed to import Schwab live service: {e}") from e

        cfg_live = SchwabAuthConfig.from_env(load_dotenv_file=True)
        svc = SchwabLiveDataService(cfg_live)
        svc.connect()
        cli = svc._client
        resp = cli.get_accounts(fields=[cli.Account.Fields.POSITIONS])
        resp.raise_for_status()
        raw = resp.json()
        accounts = raw if isinstance(raw, list) else [raw]

        rows: List[Dict[str, Any]] = []
        for a in accounts:
            sec = a.get("securitiesAccount", {}) if isinstance(a, dict) else {}
            acct_num = sec.get("accountNumber", "")
            for p in sec.get("positions", []) or []:
                inst = p.get("instrument") or {}
                rows.append(
                    {
                        "account_number": acct_num,
                        "symbol": inst.get("symbol", ""),
                        "description": inst.get("description", ""),
                        "asset_type": inst.get("assetType", ""),
                        "position_type": p.get("positionType", ""),
                        "long_quantity": p.get("longQuantity"),
                        "short_quantity": p.get("shortQuantity"),
                        "average_price": p.get("averagePrice"),
                        "market_value": p.get("marketValue"),
                        "maintenance_requirement": p.get("maintenanceRequirement"),
                        "current_day_profit_loss": p.get("currentDayProfitLoss"),
                        "current_day_profit_loss_pct": p.get("currentDayProfitLossPercentage"),
                    }
                )

        cache_csv = (
            Path(args.open_positions_cache_csv).resolve()
            if args.open_positions_cache_csv
            else (out_dir / "open_positions_from_schwab.csv")
        )
        pd.DataFrame(
            rows,
            columns=[
                "account_number",
                "symbol",
                "description",
                "asset_type",
                "position_type",
                "long_quantity",
                "short_quantity",
                "average_price",
                "market_value",
                "maintenance_requirement",
                "current_day_profit_loss",
                "current_day_profit_loss_pct",
            ],
        ).to_csv(cache_csv, index=False)
        open_positions_csv = cache_csv

    if args.mode in {"daily", "all"}:
        daily = run_daily(trades, out_dir, cfg, open_positions_csv, args.lookback_trades)
    if args.mode in {"weekly", "all"}:
        weekly = run_weekly(trades, out_dir, cfg, as_of=None)
    if args.mode in {"monthly", "all"}:
        monthly = run_monthly(trades, out_dir, cfg)

    if args.mode == "all":
        summary = {
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "source_realized_csv": str(realized_csv),
            "source_sheet_csv_url": args.sheet_csv_url or "",
            "daily_status": daily.get("status") if daily else "",
            "weekly_status": weekly.get("status") if weekly else "",
            "monthly_target_gap": monthly.get("gap_to_target") if monthly else math.nan,
        }
        if sheet_meta:
            summary["sheet_parser_meta"] = sheet_meta
        write_json(out_dir / "playbook_run_summary.json", summary)
        lines = [
            "# Trade Playbook Run",
            "",
            f"- Source realized CSV: `{realized_csv}`",
            (f"- Source sheet URL: `{args.sheet_csv_url}`" if args.sheet_csv_url else "- Source sheet URL: n/a"),
            f"- Daily status: **{summary['daily_status']}**",
            f"- Weekly status: **{summary['weekly_status']}**",
            (
                f"- Gap to target monthly P/L: {summary['monthly_target_gap']:,.2f}"
                if np.isfinite(summary["monthly_target_gap"])
                else "- Gap to target monthly P/L: n/a"
            ),
            "",
            "## Files",
            "- `daily_risk_monitor.md`",
            "- `weekly_edge_report.md`",
            "- `monthly_longitudinal_review.md`",
            "- `playbook_run_summary.json`",
        ]
        write_text(out_dir / "playbook_run_summary.md", "\n".join(lines))

    print(f"Mode: {args.mode}")
    print(f"Source: {realized_csv}")
    if args.realized_source == "sheet" and args.sheet_csv_url:
        print(f"Pulled sheet URL: {args.sheet_csv_url}")
        print(f"Sheet row filter: {args.sheet_row_filter}")
    if args.realized_source == "schwab":
        print(f"Realized source: Schwab API ({args.schwab_lookback_days}d lookback)")
    if args.open_positions_source == "schwab" and open_positions_csv:
        print(f"Open positions source: Schwab API (cached to {open_positions_csv})")
    print(f"Rows loaded: {len(trades)}")
    print(f"Wrote outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



