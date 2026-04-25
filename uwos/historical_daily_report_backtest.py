#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from uwos.exact_spread_backtester import UnderlyingCloseStore, build_occ_symbol, intrinsic_value, parse_date


DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")
ENTRY_RE = re.compile(r"\b(Debit|Credit)\s+([0-9]*\.?[0-9]+)", re.IGNORECASE)
TARGET_RE = re.compile(r"Target\s+([<>]=\s*[0-9]*\.?[0-9]+\s*(?:db|cr))", re.IGNORECASE)
LEG_RE = re.compile(r"\b(Buy|Sell)\s+([0-9]*\.?[0-9]+)([CP])\b", re.IGNORECASE)


def safe_float(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        text = str(value).strip().replace("$", "").replace(",", "").replace("%", "")
        if not text:
            return math.nan
        return float(text)
    except Exception:
        return math.nan


def infer_date(path: Path) -> Optional[dt.date]:
    m = DATE_RE.search(str(path))
    if not m:
        return None
    return parse_date(m.group(1))


def split_pipe_row(line: str) -> List[str]:
    row = line.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def is_separator(line: str) -> bool:
    cells = split_pipe_row(line)
    return bool(cells) and all("-" in c and set(c.replace(" ", "")) <= set("-:") for c in cells)


def iter_markdown_tables(text: str) -> Iterable[Tuple[str, str, pd.DataFrame]]:
    heading = ""
    label = ""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            label = ""
            i += 1
            continue
        if stripped.startswith("**") and stripped.endswith("**"):
            label = stripped.strip("*").strip()
            i += 1
            continue
        if stripped.startswith("|") and i + 1 < len(lines) and is_separator(lines[i + 1]):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            headers = split_pipe_row(block[0])
            rows = []
            for row_line in block[2:]:
                cells = split_pipe_row(row_line)
                cells = cells + [""] * max(0, len(headers) - len(cells))
                rows.append(cells[: len(headers)])
            yield heading, label, pd.DataFrame(rows, columns=headers)
            continue
        i += 1


def infer_strategy(action: str, strike_setup: str, strategy_type: str = "") -> str:
    s = str(strategy_type or "").strip()
    if s and s.lower() not in {"nan", "none"}:
        return s
    text = f"{action} {strike_setup}".upper()
    if "IRON CONDOR" in text or "+" in str(strike_setup):
        return "Iron Condor"
    if "BULL CALL" in text:
        return "Bull Call Debit"
    if "BEAR PUT" in text:
        return "Bear Put Debit"
    if "BULL PUT" in text:
        return "Bull Put Credit"
    if "BEAR CALL" in text:
        return "Bear Call Credit"
    legs = LEG_RE.findall(str(strike_setup))
    if len(legs) >= 2:
        first_action, _first_strike, right = legs[0]
        right = right.upper()
        if first_action.lower() == "buy" and right == "C":
            return "Bull Call Debit"
        if first_action.lower() == "buy" and right == "P":
            return "Bear Put Debit"
        if first_action.lower() == "sell" and right == "P":
            return "Bull Put Credit"
        if first_action.lower() == "sell" and right == "C":
            return "Bear Call Credit"
    return ""


def parse_entry(text: str) -> Tuple[str, float, str]:
    m = ENTRY_RE.search(str(text or ""))
    if not m:
        return "", math.nan, ""
    net_type = m.group(1).lower()
    entry_net = safe_float(m.group(2))
    target = ""
    mt = TARGET_RE.search(str(text or ""))
    if mt:
        target = mt.group(1).strip()
    return net_type, entry_net, target


def parse_strikes(strategy: str, strike_setup: str) -> Dict[str, float]:
    legs = [
        (action.title(), safe_float(strike), right.upper())
        for action, strike, right in LEG_RE.findall(str(strike_setup or ""))
    ]
    out: Dict[str, float] = {}
    if strategy == "Iron Condor" and len(legs) >= 4:
        for action, strike, right in legs:
            if right == "P" and action == "Sell":
                out["short_put_strike"] = strike
            elif right == "P" and action == "Buy":
                out["long_put_strike"] = strike
            elif right == "C" and action == "Sell":
                out["short_call_strike"] = strike
            elif right == "C" and action == "Buy":
                out["long_call_strike"] = strike
        put_w = abs(out.get("short_put_strike", math.nan) - out.get("long_put_strike", math.nan))
        call_w = abs(out.get("long_call_strike", math.nan) - out.get("short_call_strike", math.nan))
        out["put_width"] = put_w
        out["call_width"] = call_w
        out["width"] = max(put_w if np.isfinite(put_w) else 0.0, call_w if np.isfinite(call_w) else 0.0)
        return out

    if len(legs) >= 2:
        for action, strike, _right in legs[:2]:
            if action == "Buy":
                out["long_strike"] = strike
            elif action == "Sell":
                out["short_strike"] = strike
        out["right"] = legs[0][2]
        out["width"] = abs(out.get("short_strike", math.nan) - out.get("long_strike", math.nan))
    return out


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def rows_from_report(path: Path, include_watch: bool = False) -> List[Dict[str, object]]:
    signal_date = infer_date(path)
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    rows: List[Dict[str, object]] = []
    compact: Dict[str, Dict[str, object]] = {}

    for heading, label, raw_df in iter_markdown_tables(text):
        df = clean_columns(raw_df)
        cols = set(df.columns)
        if {"#", "Ticker", "Action", "Strike Setup", "Expiry", "Net Credit/Debit"}.issubset(cols):
            for _, row in df.iterrows():
                parsed = normalize_report_row(row.to_dict(), signal_date, path, include_watch)
                if parsed:
                    rows.append(parsed)
            continue
        if "#" in cols and "Ticker" in cols and label in {"Trade plan", "Strike setup", "Risk / edge"}:
            for _, row in df.iterrows():
                key = str(row.get("#", "")).strip()
                if not key:
                    continue
                compact.setdefault(key, {"#": key})
                compact[key].update(row.to_dict())

    for row in compact.values():
        parsed = normalize_report_row(row, signal_date, path, include_watch)
        if parsed:
            rows.append(parsed)
    return rows


def normalize_report_row(
    row: Dict[str, object],
    signal_date: Optional[dt.date],
    source_path: Path,
    include_watch: bool,
) -> Optional[Dict[str, object]]:
    action = str(row.get("Action", "") or "")
    if (not include_watch) and "WATCH ONLY" in action.upper():
        return None
    ticker = str(row.get("Ticker", "") or "").strip().upper()
    expiry = parse_date(row.get("Expiry"))
    strike_setup = str(row.get("Strike Setup", "") or "").strip()
    if not ticker or signal_date is None or expiry is None or not strike_setup:
        return None
    strategy = infer_strategy(action, strike_setup, str(row.get("Strategy Type", "") or ""))
    if strategy not in {"Bull Call Debit", "Bear Put Debit", "Bull Put Credit", "Bear Call Credit", "Iron Condor"}:
        return None
    net_type, entry_net, entry_gate = parse_entry(str(row.get("Net Credit/Debit", "") or ""))
    if not net_type or not np.isfinite(entry_net):
        return None
    strikes = parse_strikes(strategy, strike_setup)
    width = safe_float(strikes.get("width"))
    if not np.isfinite(width) or width <= 0:
        return None
    number = str(row.get("#", "") or "").strip()
    out: Dict[str, object] = {
        "trade_id": f"{signal_date.isoformat()}-{number or len(strike_setup)}-{ticker}",
        "source_report": str(source_path),
        "signal_date": signal_date,
        "ticker": ticker,
        "strategy": strategy,
        "expiry": expiry,
        "entry_net": float(entry_net),
        "entry_gate": entry_gate,
        "net_type": net_type,
        "width": float(width),
        "action": action,
        "strike_setup": strike_setup,
        "conviction": str(row.get("Conviction %", "") or ""),
        "setup_likelihood": str(row.get("Setup Likelihood", "") or ""),
        "execution_book": "Watch" if "WATCH ONLY" in action.upper() else "",
        "qty": 1.0,
    }
    out.update(strikes)
    if strategy != "Iron Condor":
        right = str(strikes.get("right", "") or "").upper()
        out["long_leg"] = build_occ_symbol(ticker, expiry, right, float(out["long_strike"]))
        out["short_leg"] = build_occ_symbol(ticker, expiry, right, float(out["short_strike"]))
    else:
        out["long_put_leg"] = build_occ_symbol(ticker, expiry, "P", float(out["long_put_strike"]))
        out["short_put_leg"] = build_occ_symbol(ticker, expiry, "P", float(out["short_put_strike"]))
        out["short_call_leg"] = build_occ_symbol(ticker, expiry, "C", float(out["short_call_strike"]))
        out["long_call_leg"] = build_occ_symbol(ticker, expiry, "C", float(out["long_call_strike"]))
    return out


def vertical_exit_value(row: pd.Series, spot: float) -> float:
    right = str(row.get("right", "") or "")
    long_intr = intrinsic_value(right, float(row["long_strike"]), spot)
    short_intr = intrinsic_value(right, float(row["short_strike"]), spot)
    if str(row["net_type"]).lower() == "credit":
        return short_intr - long_intr
    return long_intr - short_intr


def condor_exit_value(row: pd.Series, spot: float) -> float:
    put_value = intrinsic_value("P", float(row["short_put_strike"]), spot) - intrinsic_value(
        "P", float(row["long_put_strike"]), spot
    )
    call_value = intrinsic_value("C", float(row["short_call_strike"]), spot) - intrinsic_value(
        "C", float(row["long_call_strike"]), spot
    )
    return float(max(0.0, put_value) + max(0.0, call_value))


def max_profit_loss(row: pd.Series) -> Tuple[float, float]:
    entry = float(row["entry_net"])
    if row["strategy"] == "Iron Condor":
        width = max(float(row.get("put_width", row["width"])), float(row.get("call_width", row["width"])))
        return entry * 100.0, max(0.0, (width - entry) * 100.0)
    width = float(row["width"])
    if str(row["net_type"]).lower() == "credit":
        return entry * 100.0, max(0.0, (width - entry) * 100.0)
    return max(0.0, (width - entry) * 100.0), entry * 100.0


def run_report_backtest(setups: pd.DataFrame, root_dir: Path, valuation_date: dt.date) -> pd.DataFrame:
    close_store = UnderlyingCloseStore(root_dir=root_dir, allow_web_fallback=True)
    rows = []
    for _, row in setups.iterrows():
        expiry = row["expiry"]
        base = row.to_dict()
        if expiry > valuation_date:
            rows.append({**base, "status": "open_not_expired", "status_reason": "expiry_after_valuation_date"})
            continue
        close = close_store.get_close_on_or_before(str(row["ticker"]), expiry, lookback_days=7)
        if close is None or not np.isfinite(close):
            rows.append({**base, "status": "failed_missing_underlying_close", "status_reason": "no_close_on_or_before_expiry"})
            continue
        exit_value = condor_exit_value(row, float(close)) if row["strategy"] == "Iron Condor" else vertical_exit_value(row, float(close))
        entry = float(row["entry_net"])
        if str(row["net_type"]).lower() == "credit":
            pnl = (entry - exit_value) * 100.0
        else:
            pnl = (exit_value - entry) * 100.0
        max_profit, max_loss = max_profit_loss(row)
        rows.append(
            {
                **base,
                "status": "completed",
                "status_reason": "expiry_intrinsic",
                "underlying_close": float(close),
                "exit_value": float(exit_value),
                "pnl": float(pnl),
                "win": bool(pnl > 0),
                "max_profit": float(max_profit),
                "max_loss": float(max_loss),
                "return_on_risk": float(pnl / max_loss) if max_loss > 0 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> Dict[str, object]:
    done = results[results["status"].eq("completed")].copy()
    status_counts = results["status"].value_counts(dropna=False).to_dict() if not results.empty else {}
    if done.empty:
        return {"completed_trades": 0, "status_counts": {str(k): int(v) for k, v in status_counts.items()}}
    gp = float(done.loc[done["pnl"] > 0, "pnl"].sum())
    gl = float(-done.loc[done["pnl"] < 0, "pnl"].sum())
    daily = done.groupby("signal_date", dropna=False)["pnl"].sum()
    return {
        "completed_trades": int(len(done)),
        "open_or_skipped_trades": int(len(results) - len(done)),
        "net_pnl": float(done["pnl"].sum()),
        "gross_profit": gp,
        "gross_loss": gl,
        "profit_factor": float(gp / gl) if gl > 0 else ("inf" if gp > 0 else math.nan),
        "win_rate": float((done["pnl"] > 0).mean()),
        "avg_pnl": float(done["pnl"].mean()),
        "median_pnl": float(done["pnl"].median()),
        "good_days": int((daily > 0).sum()),
        "bad_days": int((daily < 0).sum()),
        "flat_days": int((daily == 0).sum()),
        "status_counts": {str(k): int(v) for k, v in status_counts.items()},
        "start_signal_date": str(done["signal_date"].min()),
        "end_signal_date": str(done["signal_date"].max()),
    }


def latest_available_date(root_dir: Path) -> dt.date:
    dates = [d for d in (parse_date(p.name) for p in root_dir.iterdir() if p.is_dir()) if d is not None]
    return max(dates) if dates else dt.date.today()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Backtest historical daily markdown reports by expiry intrinsic settlement.")
    ap.add_argument("--root-dir", default=".", help="Repo/root directory containing YYYY-MM-DD folders.")
    ap.add_argument("--out-dir", default="out/historical_daily_report_backtest", help="Output directory.")
    ap.add_argument("--start-date", default="", help="Optional start date.")
    ap.add_argument("--end-date", default="", help="Optional end date.")
    ap.add_argument("--lookback-days", type=int, default=90, help="Use reports within N days of end date when start-date is omitted.")
    ap.add_argument("--include-watch", action="store_true", help="Also backtest Watch Only rows.")
    ap.add_argument("--valuation-date", default="", help="Date through which expiries are considered completed.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    valuation_date = parse_date(args.valuation_date) or latest_available_date(root)
    end_date = parse_date(args.end_date) or valuation_date
    start_date = parse_date(args.start_date)
    if start_date is None:
        start_date = end_date - dt.timedelta(days=max(1, int(args.lookback_days)))

    report_paths = sorted(root.glob("20??-??-??/anu-expert-trade-table-*.md"))
    selected = []
    for path in report_paths:
        d = infer_date(path)
        if d is not None and start_date <= d <= end_date:
            selected.append(path)

    setup_rows: List[Dict[str, object]] = []
    parse_errors: List[Dict[str, object]] = []
    for path in selected:
        try:
            setup_rows.extend(rows_from_report(path, include_watch=bool(args.include_watch)))
        except Exception as exc:
            parse_errors.append({"report": str(path), "error": str(exc)})

    setups = pd.DataFrame(setup_rows)
    if setups.empty:
        raise RuntimeError("No parseable trade rows found in selected reports.")
    results = run_report_backtest(setups, root, valuation_date)
    summary = summarize(results)
    summary["reports_scanned"] = len(selected)
    summary["setups_loaded"] = int(len(setups))
    summary["backtest_scope"] = "published_markdown_reports"
    summary["scope_note"] = (
        "This script scans anu-expert-trade-table-*.md reports only; "
        "use the dated-folder replay harness for full pipeline replay."
    )
    summary["valuation_date"] = valuation_date.isoformat()
    summary["start_date"] = start_date.isoformat()
    summary["end_date"] = end_date.isoformat()

    trade_csv = out_dir / "trade_level_results.csv"
    daily_csv = out_dir / "daily_summary.csv"
    strategy_csv = out_dir / "summary_by_strategy.csv"
    summary_json = out_dir / "summary.json"
    setups_csv = out_dir / "parsed_setups.csv"
    errors_csv = out_dir / "parse_errors.csv"

    setups.to_csv(setups_csv, index=False)
    results.to_csv(trade_csv, index=False)
    done = results[results["status"].eq("completed")].copy()
    if not done.empty:
        done.groupby("signal_date", dropna=False).agg(
            trades=("pnl", "size"),
            wins=("win", "sum"),
            net_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
        ).reset_index().to_csv(daily_csv, index=False)
        done.groupby("strategy", dropna=False).agg(
            trades=("pnl", "size"),
            win_rate=("win", "mean"),
            net_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_return_on_risk=("return_on_risk", "mean"),
        ).reset_index().sort_values("net_pnl", ascending=False).to_csv(strategy_csv, index=False)
    else:
        pd.DataFrame().to_csv(daily_csv, index=False)
        pd.DataFrame().to_csv(strategy_csv, index=False)
    pd.DataFrame(parse_errors).to_csv(errors_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Reports scanned: {len(selected)}")
    print("Scope: published markdown reports only; not full dated-folder replay.")
    print(f"Setups loaded: {len(setups)}")
    print(f"Completed trades: {summary.get('completed_trades', 0)}")
    print(f"Status counts: {summary.get('status_counts', {})}")
    print(f"Wrote: {trade_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {daily_csv}")
    print(f"Wrote: {strategy_csv}")


if __name__ == "__main__":
    main()
