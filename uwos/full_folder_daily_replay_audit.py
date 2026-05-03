#!/usr/bin/env python3
"""Full-folder historical replay audit for the daily options pipeline.

This is intentionally separate from trend-analysis. It replays dated daily-input
folders through run_mode_a_two_stage.py, then backtests the emitted daily trade
reports and checks for safety exceptions in the machine-readable decision books.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - audit still runs with default caps
    yaml = None

try:
    from uwos.historical_daily_report_backtest import rows_from_report, run_report_backtest, summarize
except Exception:  # pragma: no cover - allows script execution from uwos cwd with PYTHONPATH issues
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from uwos.historical_daily_report_backtest import rows_from_report, run_report_backtest, summarize

REQUIRED_FAMILIES = {
    "dp": ["dp-eod-report-"],
    "whale_source": ["bot-eod-report-", "whale_trades_filtered", "whale-"],
    "stock": ["stock-screener-"],
    "hot": ["hot-chain-", "hot-chains-"],
    "chain": ["chain-oi-changes-"],
}
ACTION_BOOKS = {"Core", "Tactical", "Medium", "Income", "Pilot", "Scout"}
APPROVED_BOOKS = {"Core", "Tactical"}
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")
APPROVED_COUNT_RE = re.compile(
    r"(?:Approved trades|Historical gate-pass candidates \(NOT live approvals\)):\s*(\d+)\s*/\s*(\d+)"
)
TRUE_TOKENS = {"true", "1", "yes", "y"}
FALSE_TOKENS = {"false", "0", "no", "n"}


def fnum(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return math.nan
    return out if math.isfinite(out) else math.nan


def load_risk_limits(config: Path) -> Dict[str, object]:
    if yaml is None or not config.exists():
        return {}
    try:
        data = yaml.safe_load(config.read_text()) or {}
    except Exception:
        return {}
    playbook = data.get("playbook", {}) if isinstance(data, dict) else {}
    risk = playbook.get("risk_limits", {}) if isinstance(playbook, dict) else {}
    return risk if isinstance(risk, dict) else {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Replay dated daily folders through the daily pipeline and summarize realized quality.")
    p.add_argument("--root", type=Path, default=Path("/Users/anuppamvi/uw_root/tradedesk"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--valuation-date", default=None, help="Date used to mark open/not-expired trades. Defaults to today.")
    p.add_argument("--max-workers", type=int, default=4)
    p.add_argument("--top-trades", type=int, default=20)
    p.add_argument("--reuse-existing", action="store_true", help="Skip replay and only aggregate an existing out-root.")
    p.add_argument("--auto-collect-uw-gex", action="store_true", help="Allow browser/UW GEX collection. Default disables it for historical replay speed/reproducibility.")
    return p.parse_args()


def as_date(s: Optional[str]) -> Optional[dt.date]:
    if not s:
        return None
    return dt.date.fromisoformat(s)


def is_date_folder(path: Path) -> bool:
    return path.is_dir() and bool(DATE_RE.fullmatch(path.name))


def has_family(names: Iterable[str], prefixes: Iterable[str]) -> bool:
    return any(any(name.startswith(prefix) for prefix in prefixes) for name in names)


def parse_approved_counts(stdout: str) -> tuple[Optional[int], Optional[int]]:
    m = APPROVED_COUNT_RE.search(stdout or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def approved_action_mask(df: pd.DataFrame) -> pd.Series:
    """Rows that were both placed in an actionable book and explicitly approved.

    Older decision books may not have an ``approved`` column, so keep backward
    compatibility in that case. If the column exists, it must be true; otherwise
    portfolio caps or later safety demotions can leak into realized metrics.
    """
    if df.empty or "execution_book" not in df.columns:
        return pd.Series(False, index=df.index)
    book_mask = df["execution_book"].fillna("").astype(str).isin(ACTION_BOOKS)
    if "approved" not in df.columns:
        return book_mask
    approved_text = df["approved"].fillna("").astype(str).str.strip().str.lower()
    has_bool_tokens = approved_text.isin(TRUE_TOKENS | FALSE_TOKENS)
    if not bool(has_bool_tokens.any()):
        return book_mask
    return book_mask & approved_text.isin(TRUE_TOKENS)


def inventory(root: Path, start: Optional[dt.date], end: Optional[dt.date]) -> tuple[List[Path], List[dict]]:
    folders: List[Path] = []
    incomplete: List[dict] = []
    for folder in sorted(p for p in root.iterdir() if is_date_folder(p)):
        d = dt.date.fromisoformat(folder.name)
        if start and d < start:
            continue
        if end and d > end:
            continue
        names = [p.name for p in folder.iterdir() if p.is_file()]
        missing = [family for family, prefixes in REQUIRED_FAMILIES.items() if not has_family(names, prefixes)]
        if missing:
            incomplete.append({"date": folder.name, "missing": missing})
        else:
            folders.append(folder)
    return folders, incomplete


def run_one(root: Path, config: Path, out_root: Path, top_trades: int, auto_collect_uw_gex: bool, folder: Path) -> dict:
    scan_date = folder.name
    odir = out_root / scan_date
    odir.mkdir(parents=True, exist_ok=True)
    report = odir / f"anu-expert-trade-table-{scan_date}.md"
    cmd = [
        sys.executable,
        "-m",
        "uwos.run_mode_a_two_stage",
        "--historical-replay",
        "--base-dir",
        str(folder),
        "--config",
        str(config),
        "--out-dir",
        str(odir),
        "--top-trades",
        str(top_trades),
        "--output",
        str(report),
    ]
    if not auto_collect_uw_gex:
        cmd.insert(4, "--no-auto-collect-uw-gex")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=900)
    (odir / "run.log").write_text(proc.stdout)
    result = {
        "date": scan_date,
        "returncode": proc.returncode,
        "seconds": round(time.time() - t0, 2),
        "report": str(report),
        "out_dir": str(odir),
    }
    approved, candidates = parse_approved_counts(proc.stdout)
    if approved is not None and candidates is not None:
        result["approved"] = approved
        result["candidates"] = candidates
    m = re.search(r"Core=(\d+), Tactical=(\d+)(?:, Medium=(\d+))?(?:, Income=(\d+))?(?:, Pilot=(\d+))?, Scout=(\d+), Watch=(\d+)", proc.stdout)
    if m:
        result["core"] = int(m.group(1))
        result["tactical"] = int(m.group(2))
        result["medium"] = int(m.group(3) or 0)
        result["income"] = int(m.group(4) or 0)
        result["pilot"] = int(m.group(5) or 0)
        result["scout"] = int(m.group(6))
        result["watch"] = int(m.group(7))
    if proc.returncode != 0:
        result["tail"] = proc.stdout[-4000:]
    return result


def write_manifest(out_root: Path, manifest: dict) -> None:
    (out_root / "batch_results.json").write_text(json.dumps(manifest, indent=2, default=str))


def run_batch(args: argparse.Namespace, config: Path) -> pd.DataFrame:
    start = as_date(args.start_date)
    end = as_date(args.end_date)
    folders, incomplete = inventory(args.root, start, end)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "root": str(args.root),
        "config": str(config),
        "out_root": str(args.out_root),
        "folders": [f.name for f in folders],
        "incomplete": incomplete,
        "auto_collect_uw_gex": bool(args.auto_collect_uw_gex),
        "results": [],
    }
    write_manifest(args.out_root, manifest)
    print(f"complete_folders={len(folders)} incomplete={len(incomplete)} out={args.out_root}", flush=True)
    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futures = {ex.submit(run_one, args.root, config, args.out_root, args.top_trades, args.auto_collect_uw_gex, f): f for f in folders}
        for idx, fut in enumerate(as_completed(futures), 1):
            result = fut.result()
            results.append(result)
            manifest["results"] = sorted(results, key=lambda x: x["date"])
            write_manifest(args.out_root, manifest)
            print(
                f"[{idx}/{len(folders)}] {result['date']} rc={result['returncode']} "
                f"approved={result.get('approved')} core={result.get('core')} tactical={result.get('tactical')} "
                f"scout={result.get('scout')} watch={result.get('watch')} sec={result['seconds']}",
                flush=True,
            )
    manifest["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
    manifest["results"] = sorted(results, key=lambda x: x["date"])
    write_manifest(args.out_root, manifest)
    return pd.DataFrame(manifest["results"])


def max_drawdown(df: pd.DataFrame) -> float:
    if df.empty or "pnl" not in df.columns:
        return 0.0
    work = df.copy()
    if "signal_date" in work.columns:
        work = work.sort_values("signal_date")
    pnl = pd.to_numeric(work["pnl"], errors="coerce").fillna(0.0)
    equity = pnl.cumsum()
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    return float((equity - peak).min())


def cost_model_summary(df: pd.DataFrame) -> Dict[str, dict]:
    if df.empty or "pnl" not in df.columns:
        return {}
    status = df.get("status", pd.Series("", index=df.index)).fillna("").astype(str)
    completed = df[status.eq("completed")].copy()
    if completed.empty:
        return {}
    base_pnl = pd.to_numeric(completed["pnl"], errors="coerce").fillna(0.0)
    entry_net = pd.to_numeric(completed.get("entry_net", 0.0), errors="coerce").fillna(0.0).abs() * 100.0
    out = {}
    for label, pct in [("base", 0.0), ("worse_fill_5pct", 0.05), ("worse_fill_10pct", 0.10)]:
        pnl = base_pnl - (entry_net * pct)
        gross_profit = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
        out[label] = {
            "net_pnl": float(pnl.sum()),
            "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else None),
            "win_rate": float((pnl > 0).sum() / len(pnl)) if len(pnl) else None,
            "max_drawdown": max_drawdown(pd.DataFrame({"pnl": pnl, "signal_date": completed.get("signal_date", "")})),
        }
    return out


def walk_forward_summary(joined: pd.DataFrame) -> Dict[str, dict]:
    if joined.empty or "signal_date" not in joined.columns:
        return {}
    work = joined.copy()
    work["signal_month"] = work["signal_date"].astype(str).str[:7]
    windows = {
        "train_2026-01_02_test_2026-03": (["2026-01", "2026-02"], ["2026-03"]),
        "train_2026-02_03_test_2026-04": (["2026-02", "2026-03"], ["2026-04"]),
    }
    out = {}
    for name, (train_months, test_months) in windows.items():
        train = work[work["signal_month"].isin(train_months)]
        test = work[work["signal_month"].isin(test_months)]
        out[name] = {"train": metric_block(train), "test": metric_block(test)}
    return out


def active_overlap_summary(df: pd.DataFrame) -> Dict[str, object]:
    if df.empty or "signal_date" not in df.columns:
        return {}
    work = df.copy()
    work["signal_dt"] = pd.to_datetime(work["signal_date"], errors="coerce")
    work["expiry_dt"] = pd.to_datetime(work.get("expiry", work["signal_date"]), errors="coerce")
    work = work.dropna(subset=["signal_dt"])
    if work.empty:
        return {}
    work["expiry_dt"] = work["expiry_dt"].fillna(work["signal_dt"])
    risk = pd.to_numeric(work.get("entry_net", 0.0), errors="coerce").fillna(0.0).abs() * 100.0
    if "live_max_loss" in work.columns:
        live_risk = pd.to_numeric(work["live_max_loss"], errors="coerce")
        risk = live_risk.fillna(risk)
    work["risk_proxy"] = risk.replace(0.0, 100.0)
    min_day = work["signal_dt"].min().normalize()
    max_day = work["expiry_dt"].max().normalize()
    if pd.isna(min_day) or pd.isna(max_day):
        return {}
    days = pd.date_range(min_day, max_day, freq="D")
    max_active = 0
    max_ticker_share = 0.0
    max_repeated_ticker_share = 0.0
    max_active_duplicate_ticker_count = 0
    max_sector_share = 0.0
    worst_ticker = ""
    worst_repeated_ticker = ""
    worst_sector = ""
    for day in days:
        active = work[(work["signal_dt"] <= day) & (work["expiry_dt"] >= day)]
        if active.empty:
            continue
        total = float(active["risk_proxy"].sum())
        max_active = max(max_active, int(len(active)))
        if total <= 0:
            continue
        if "ticker" in active.columns:
            ticker_share = active.groupby(active["ticker"].fillna("UNKNOWN"))["risk_proxy"].sum() / total
            if not ticker_share.empty and float(ticker_share.max()) > max_ticker_share:
                max_ticker_share = float(ticker_share.max())
                worst_ticker = str(ticker_share.idxmax())
            ticker_counts = active["ticker"].fillna("UNKNOWN").astype(str).value_counts()
            if not ticker_counts.empty:
                max_active_duplicate_ticker_count = max(max_active_duplicate_ticker_count, int(ticker_counts.max()))
                repeated = ticker_counts[ticker_counts > 1].index
                if len(repeated):
                    repeated_share = ticker_share[ticker_share.index.isin(repeated)]
                    if not repeated_share.empty and float(repeated_share.max()) > max_repeated_ticker_share:
                        max_repeated_ticker_share = float(repeated_share.max())
                        worst_repeated_ticker = str(repeated_share.idxmax())
        if "sector" in active.columns:
            sector_share = active.groupby(active["sector"].fillna("UNKNOWN"))["risk_proxy"].sum() / total
            if not sector_share.empty and float(sector_share.max()) > max_sector_share:
                max_sector_share = float(sector_share.max())
                worst_sector = str(sector_share.idxmax())
    return {
        "max_active_trades": max_active,
        "max_ticker_share": max_ticker_share,
        "worst_ticker": worst_ticker,
        "max_repeated_ticker_share": max_repeated_ticker_share,
        "worst_repeated_ticker": worst_repeated_ticker,
        "max_active_duplicate_ticker_count": max_active_duplicate_ticker_count,
        "max_sector_share": max_sector_share,
        "worst_sector": worst_sector,
    }


def risk_proxy_for_row(row: pd.Series) -> float:
    for col in ["live_max_loss", "max_loss", "risk", "risk_proxy"]:
        if col in row.index:
            val = fnum(row.get(col))
            if math.isfinite(val) and val > 0:
                return float(val)
    entry = fnum(row.get("entry_net"))
    if math.isfinite(entry) and entry != 0:
        return abs(entry) * 100.0
    return 100.0


def apply_simulated_portfolio_caps(df: pd.DataFrame, risk_limits: Dict[str, object]) -> tuple[pd.DataFrame, Dict[str, object]]:
    """Replay approved trades through an active-book concentration filter.

    Daily historical replay has no Schwab open-position snapshot, so the live
    pretrade caps cannot protect metrics from repeatedly selecting the same
    ticker while prior dated trades are still open. This simulation keeps the
    original rows and adds a pass/reason column, then returns metrics for the
    portfolio-safe subset.
    """
    if df.empty or "signal_date" not in df.columns:
        return df.copy(), {"enabled": False, "reason": "empty_or_missing_signal_date"}

    max_active_val = fnum(risk_limits.get("max_active_trades", 0))
    max_active = int(max_active_val) if math.isfinite(max_active_val) and max_active_val > 0 else 0
    max_ticker_share = fnum(risk_limits.get("max_active_ticker_share", risk_limits.get("single_symbol_max_share", 0)))
    max_sector_share = fnum(risk_limits.get("max_active_sector_share", risk_limits.get("max_sector_share", 0)))
    min_share_base_val = fnum(risk_limits.get("min_active_trades_for_share_caps", 0))
    min_share_base = int(min_share_base_val) if math.isfinite(min_share_base_val) and min_share_base_val > 0 else 0
    if min_share_base <= 0:
        share_terms = [v for v in [max_ticker_share, max_sector_share] if math.isfinite(v) and v > 0]
        min_share_base = max(2, int(math.ceil(1.0 / max(share_terms))) if share_terms else 2)
    if max_active <= 0 and not (math.isfinite(max_ticker_share) and max_ticker_share > 0) and not (
        math.isfinite(max_sector_share) and max_sector_share > 0
    ):
        out = df.copy()
        out["portfolio_sim_pass"] = True
        out["portfolio_sim_reason"] = ""
        return out, {"enabled": False, "reason": "no_caps_configured"}

    work = df.copy()
    work["_signal_dt"] = pd.to_datetime(work["signal_date"], errors="coerce")
    work["_expiry_dt"] = pd.to_datetime(work.get("expiry", work["signal_date"]), errors="coerce")
    work["_expiry_dt"] = work["_expiry_dt"].fillna(work["_signal_dt"])
    book_col = "execution_book_decision" if "execution_book_decision" in work.columns else "execution_book"
    if book_col in work.columns:
        book_rank = {"Core": 0, "Tactical": 1, "Medium": 2, "Income": 3, "Pilot": 4, "Scout": 5}
        work["_book_rank"] = work[book_col].fillna("").astype(str).map(book_rank).fillna(9)
    else:
        work["_book_rank"] = 9
    work["_edge_sort"] = pd.to_numeric(work.get("edge_pct"), errors="coerce").fillna(-1e9)
    work["_conf_sort"] = pd.to_numeric(work.get("confidence_score"), errors="coerce").fillna(-1e9)
    work["_orig_order"] = range(len(work))
    work = work.sort_values(["_signal_dt", "_book_rank", "_conf_sort", "_edge_sort", "_orig_order"], ascending=[True, True, False, False, True])
    work["portfolio_sim_pass"] = False
    work["portfolio_sim_reason"] = ""

    kept_indices: list[int] = []
    rejected = 0
    for idx, row in work.iterrows():
        signal_dt = row.get("_signal_dt")
        if pd.isna(signal_dt):
            work.at[idx, "portfolio_sim_reason"] = "missing_signal_date"
            rejected += 1
            continue
        active_indices = [
            k
            for k in kept_indices
            if pd.notna(work.at[k, "_expiry_dt"]) and work.at[k, "_expiry_dt"] >= signal_dt
        ]
        projected_indices = active_indices + [idx]
        reasons = []
        if max_active > 0 and len(projected_indices) > max_active:
            reasons.append(f"active_count {len(projected_indices)} > {max_active}")

        risk_by_idx = {i: risk_proxy_for_row(work.loc[i]) for i in projected_indices}
        total_risk = sum(risk_by_idx.values())
        if total_risk > 0 and len(projected_indices) >= min_share_base:
            ticker = str(row.get("ticker", "UNKNOWN") or "UNKNOWN").upper().strip()
            ticker_indices = [
                i
                for i in projected_indices
                if str(work.at[i, "ticker"] if "ticker" in work.columns else "UNKNOWN").upper().strip() == ticker
            ]
            ticker_risk = sum(risk_by_idx[i] for i in ticker_indices)
            ticker_share = ticker_risk / total_risk
            if (
                len(ticker_indices) > 1
                and math.isfinite(max_ticker_share)
                and max_ticker_share > 0
                and ticker_share > max_ticker_share
            ):
                reasons.append(f"ticker_share {ticker_share:.1%} > {max_ticker_share:.1%} ({ticker})")

            sector = str(row.get("sector", "") or "").strip()
            if sector and sector.upper() not in {"UNKNOWN", "NAN"}:
                sector_risk = sum(
                    risk
                    for i, risk in risk_by_idx.items()
                    if str(work.at[i, "sector"] if "sector" in work.columns else "").strip() == sector
                )
                sector_share = sector_risk / total_risk
                if math.isfinite(max_sector_share) and max_sector_share > 0 and sector_share > max_sector_share:
                    reasons.append(f"sector_share {sector_share:.1%} > {max_sector_share:.1%} ({sector})")

        if reasons:
            work.at[idx, "portfolio_sim_reason"] = "; ".join(reasons)
            rejected += 1
            continue
        work.at[idx, "portfolio_sim_pass"] = True
        kept_indices.append(idx)

    out = work.sort_values("_orig_order").drop(columns=["_signal_dt", "_expiry_dt", "_book_rank", "_edge_sort", "_conf_sort", "_orig_order"], errors="ignore")
    return out, {
        "enabled": True,
        "input_rows": int(len(work)),
        "kept_rows": int(out["portfolio_sim_pass"].sum()),
        "rejected_rows": int(rejected),
        "max_active_trades": max_active,
        "max_active_ticker_share": max_ticker_share if math.isfinite(max_ticker_share) else None,
        "max_active_sector_share": max_sector_share if math.isfinite(max_sector_share) else None,
        "min_active_trades_for_share_caps": min_share_base,
    }


def metric_block(df: pd.DataFrame) -> Dict[str, object]:
    empty_metrics = {
        "rows": 0,
        "completed": 0,
        "open": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": None,
        "net_pnl": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "profit_factor": None,
        "pnl_available": False,
    }
    if df.empty:
        return empty_metrics
    status = df.get("status", pd.Series("", index=df.index)).fillna("").astype(str)
    completed = df[status.eq("completed")]
    if "pnl" not in completed.columns:
        metrics = dict(empty_metrics)
        metrics.update(
            {
                "rows": int(len(df)),
                "completed": int(len(completed)),
                "open": int((status != "completed").sum()),
            }
        )
        return metrics
    pnl = pd.to_numeric(completed["pnl"], errors="coerce").dropna()
    if pnl.empty:
        metrics = dict(empty_metrics)
        metrics.update(
            {
                "rows": int(len(df)),
                "completed": int(len(completed)),
                "open": int((status != "completed").sum()),
            }
        )
        return metrics
    wins = int((pnl > 0).sum())
    losses = int((pnl <= 0).sum())
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    return {
        "rows": int(len(df)),
        "completed": int(len(completed)),
        "open": int((status != "completed").sum()),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(pnl) if len(pnl) else None,
        "net_pnl": float(pnl.sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": gross_profit / gross_loss if gross_loss else (math.inf if gross_profit else None),
        "max_drawdown": max_drawdown(completed),
        "pnl_available": True,
    }


def collect_decision_books(out_root: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for day_dir in sorted(out_root.glob("20??-??-??")):
        if not day_dir.is_dir():
            continue
        paths = sorted(day_dir.glob("trade_decision_book_all_*.csv"))
        if not paths:
            paths = sorted(day_dir.glob("trade_decision_book_*.csv"))
        for path in paths[:1]:
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty:
                continue
            df["scan_date"] = path.parent.name
            df["decision_book_path"] = str(path)
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def setups_from_decision_books(decision_books: pd.DataFrame) -> pd.DataFrame:
    if decision_books.empty or "execution_book" not in decision_books.columns:
        return pd.DataFrame()
    rows = decision_books[approved_action_mask(decision_books)].copy()
    if rows.empty:
        return pd.DataFrame()

    entry = pd.to_numeric(rows.get("live_net_bid_ask"), errors="coerce")
    fallback_entry = pd.to_numeric(rows.get("net"), errors="coerce")
    rows["entry_net"] = entry.where(entry.notna(), fallback_entry)
    rows["signal_date"] = pd.to_datetime(rows.get("scan_date"), errors="coerce").dt.date
    rows["expiry"] = pd.to_datetime(rows.get("expiry"), errors="coerce").dt.date
    rows["ticker"] = rows.get("ticker", "").astype(str).str.upper()
    rows["strategy"] = rows.get("strategy", "").astype(str)
    rows["net_type"] = rows.get("net_type", "").astype(str).str.lower()
    rows["width"] = pd.to_numeric(rows.get("width"), errors="coerce")
    if "put_width" in rows.columns or "call_width" in rows.columns:
        put_w = pd.to_numeric(rows.get("put_width"), errors="coerce")
        call_w = pd.to_numeric(rows.get("call_width"), errors="coerce")
        rows["width"] = rows["width"].where(rows["width"].notna(), pd.concat([put_w, call_w], axis=1).max(axis=1))
    if "right" not in rows.columns:
        rows["right"] = ""
    rows["right"] = rows["right"].fillna("").astype(str).str.upper()
    rows.loc[rows["right"].eq("") & rows["strategy"].str.contains("Call", case=False, na=False), "right"] = "C"
    rows.loc[rows["right"].eq("") & rows["strategy"].str.contains("Put", case=False, na=False), "right"] = "P"
    rows["qty"] = 1.0
    rows["trade_id"] = rows["signal_date"].astype(str) + "-" + rows["ticker"] + "-" + rows["strategy"] + "-" + rows["expiry"].astype(str)
    rows["source_report"] = rows.get("decision_book_path", "")
    rows["setup_likelihood"] = (
        rows.get("verdict", "").fillna("").astype(str)
        + " edge "
        + pd.to_numeric(rows.get("edge_pct"), errors="coerce").round(1).astype(str)
        + "% n="
        + pd.to_numeric(rows.get("signals"), errors="coerce").fillna(0).astype(int).astype(str)
    )
    rows["strike_setup"] = rows.get("notes", "").fillna("").astype(str)
    keep = [
        "trade_id",
        "source_report",
        "signal_date",
        "ticker",
        "strategy",
        "expiry",
        "entry_net",
        "entry_gate",
        "net_type",
        "width",
        "action",
        "strike_setup",
        "conviction",
        "setup_likelihood",
        "execution_book",
        "qty",
        "long_strike",
        "short_strike",
        "right",
        "long_leg",
        "short_leg",
        "short_put_strike",
        "long_put_strike",
        "short_call_strike",
        "long_call_strike",
        "put_width",
        "call_width",
        "long_put_leg",
        "short_put_leg",
        "short_call_leg",
        "long_call_leg",
    ]
    keep = [c for c in keep if c in rows.columns]
    out = rows[keep].copy()
    out = out[out["signal_date"].notna() & out["expiry"].notna()]
    out = out[pd.to_numeric(out["entry_net"], errors="coerce").notna()]
    out = out[pd.to_numeric(out["width"], errors="coerce").fillna(0) > 0]
    return out.reset_index(drop=True)


def aggregate(args: argparse.Namespace, valuation_date: dt.date) -> dict:
    out_root = args.out_root
    config = args.config or (args.root / "uwos" / "rulebook_config_goal_holistic_claude.yaml")
    risk_limits = load_risk_limits(config)
    summary_dir = out_root / "realized_backtest"
    summary_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "batch_results.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"results": []}
    day_counts = pd.DataFrame(manifest.get("results", []))
    if not day_counts.empty:
        day_counts.to_csv(out_root / "daily_trade_counts.csv", index=False)

    decision_books = collect_decision_books(out_root)
    if not decision_books.empty:
        decision_books.to_csv(out_root / "replay_decision_books_all.csv", index=False)

    setup_source = "decision_books"
    setup_df = setups_from_decision_books(decision_books)
    if setup_df.empty:
        setup_source = "markdown_reports"
        reports = sorted(out_root.glob("20??-??-??/anu-expert-trade-table-*.md"))
        setups: List[dict] = []
        for report in reports:
            setups.extend(rows_from_report(report, include_watch=False))
        setup_df = pd.DataFrame(setups)
    results = run_report_backtest(setup_df, args.root, valuation_date) if not setup_df.empty else pd.DataFrame()
    if not results.empty:
        results.to_csv(summary_dir / "trade_level_results.csv", index=False)
    summary = summarize(results) if not results.empty else {}
    (summary_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    joined = results.copy()
    if not joined.empty and not decision_books.empty:
        approved = decision_books[approved_action_mask(decision_books)].copy()
        approved["signal_date"] = approved["scan_date"].astype(str)
        approved["entry_net_join"] = pd.to_numeric(approved.get("live_net_bid_ask", approved.get("net")), errors="coerce").round(2)
        joined["entry_net_join"] = pd.to_numeric(joined["entry_net"], errors="coerce").round(2)
        for df in (approved, joined):
            for col in ["signal_date", "ticker", "strategy", "expiry"]:
                df[col] = df[col].astype(str)
        keep = [
            c
            for c in [
                "signal_date",
                "ticker",
                "strategy",
                "expiry",
                "entry_net_join",
                "approved",
                "execution_book",
                "live_status",
                "is_final_live_valid",
                "gate_pass_effective",
                "hard_blockers",
                "quality_blockers",
                "approval_blockers",
                "final_validity_blockers",
                "confidence_score",
                "edge_score",
                "edge_pct",
                "approval_regime",
                "sector",
                "size_mult",
                "live_max_loss",
                "portfolio_cap_reason",
            ]
            if c in approved.columns
        ]
        approved = approved[keep].drop_duplicates(subset=["signal_date", "ticker", "strategy", "expiry", "entry_net_join"], keep="first")
        joined = joined.merge(approved, on=["signal_date", "ticker", "strategy", "expiry", "entry_net_join"], how="left", suffixes=("", "_decision"))
        joined.to_csv(summary_dir / "trade_level_results_joined_decision_book.csv", index=False)

    book_col = "execution_book_decision" if "execution_book_decision" in joined.columns else "execution_book"
    book_metrics = {str(k): metric_block(v) for k, v in joined.groupby(joined[book_col].fillna("UNMATCHED"))} if book_col in joined.columns and not joined.empty else {}
    strategy_metrics = {str(k): metric_block(v) for k, v in joined.groupby(joined["strategy"].fillna("UNKNOWN"))} if "strategy" in joined.columns and not joined.empty else {}
    month_metrics = {}
    if "signal_date" in joined.columns and not joined.empty:
        by_month = joined.copy()
        by_month["signal_month"] = by_month["signal_date"].astype(str).str[:7]
        month_metrics = {str(k): metric_block(v) for k, v in by_month.groupby("signal_month")}
    regime_metrics = {str(k): metric_block(v) for k, v in joined.groupby(joined["approval_regime"].fillna("UNKNOWN"))} if "approval_regime" in joined.columns and not joined.empty else {}
    strategy_regime_metrics = {}
    if not joined.empty and "strategy" in joined.columns and "approval_regime" in joined.columns:
        for (strategy, regime), block_df in joined.groupby([joined["strategy"].fillna("UNKNOWN"), joined["approval_regime"].fillna("UNKNOWN")]):
            strategy_regime_metrics[f"{strategy} / {regime}"] = metric_block(block_df)
    cost_model = cost_model_summary(joined)
    walk_forward = walk_forward_summary(joined)
    active_overlap = active_overlap_summary(joined)
    portfolio_joined, portfolio_simulation = apply_simulated_portfolio_caps(joined, risk_limits)
    portfolio_filtered = portfolio_joined[
        portfolio_joined.get("portfolio_sim_pass", pd.Series(False, index=portfolio_joined.index)).fillna(False).astype(bool)
    ].copy() if not portfolio_joined.empty else pd.DataFrame()
    if not portfolio_joined.empty:
        portfolio_joined.to_csv(summary_dir / "trade_level_results_portfolio_simulated.csv", index=False)
    portfolio_summary = metric_block(portfolio_filtered)
    portfolio_cost_model = cost_model_summary(portfolio_filtered)
    portfolio_active_overlap = active_overlap_summary(portfolio_filtered)

    invalid_approved = pd.DataFrame()
    action_book_counts: Dict[str, int] = {}
    if not decision_books.empty and "execution_book" in decision_books.columns:
        action_rows = decision_books[approved_action_mask(decision_books)].copy()
        action_book_counts = {
            str(k): int(v)
            for k, v in action_rows["execution_book"].fillna("").astype(str).value_counts().items()
            if str(k) in ACTION_BOOKS
        }
        approved = action_rows
        if not approved.empty:
            live_effective = approved.get("is_final_live_valid", False).fillna(False).astype(bool) | (
                approved.get("live_status", "").astype(str).eq("fails_live_entry_gate")
                & approved.get("gate_pass_effective", False).fillna(False).astype(bool)
            )
            hard = approved.get("hard_blockers", "").fillna("").astype(str)
            contra = hard.str.contains("contract_flow_contra|stage1_contract_flow_contra|bull_call_contract_flow_not_confirmed:contra", regex=True)
            invalid_approved = approved[(~live_effective) | contra]
            invalid_approved.to_csv(out_root / "approved_rows_safety_exceptions.csv", index=False)

    blocker_counts = []
    if not decision_books.empty and "execution_book" in decision_books.columns:
        watch = decision_books[decision_books["execution_book"].astype(str).eq("Watch")]
        counter: Counter[str] = Counter()
        for field in ["hard_blockers", "quality_blockers", "approval_blockers", "stage1_blockers", "notes"]:
            if field not in watch.columns:
                continue
            for val in watch[field].fillna("").astype(str):
                for token in re.split(r"[;|,]", val):
                    token = token.strip()
                    if token and token.lower() != "nan":
                        counter[token[:180]] += 1
        blocker_counts = counter.most_common(40)

    payload = {
        "summary": summary,
        "book_metrics": book_metrics,
        "strategy_metrics": strategy_metrics,
        "month_metrics": month_metrics,
        "regime_metrics": regime_metrics,
        "strategy_regime_metrics": strategy_regime_metrics,
        "cost_model": cost_model,
        "walk_forward": walk_forward,
        "active_overlap": active_overlap,
        "portfolio_simulation": portfolio_simulation,
        "portfolio_filtered_summary": portfolio_summary,
        "portfolio_filtered_cost_model": portfolio_cost_model,
        "portfolio_filtered_active_overlap": portfolio_active_overlap,
        "action_book_counts": action_book_counts,
        "setup_source": setup_source,
        "invalid_approved_count": int(len(invalid_approved)),
        "blocker_counts": blocker_counts,
        "files": {
            "daily_counts": str(out_root / "daily_trade_counts.csv"),
            "all_decision_books": str(out_root / "replay_decision_books_all.csv"),
            "trade_results": str(summary_dir / "trade_level_results_joined_decision_book.csv"),
            "portfolio_simulated_trade_results": str(summary_dir / "trade_level_results_portfolio_simulated.csv"),
            "summary_json": str(summary_dir / "summary.json"),
            "safety_exceptions": str(out_root / "approved_rows_safety_exceptions.csv"),
        },
    }
    (out_root / "full_folder_replay_audit_summary.json").write_text(json.dumps(payload, indent=2, default=str))
    write_markdown(out_root, manifest, day_counts, payload)
    return payload


def fmt_pct(value: object) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{float(value):.1%}"


def fmt_pf(value: object) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if value == math.inf:
        return "inf"
    return f"{float(value):.2f}"


def write_markdown(out_root: Path, manifest: dict, day_counts: pd.DataFrame, payload: dict) -> None:
    summary = payload.get("summary", {})
    md: List[str] = ["# Full-folder daily replay audit", ""]
    md.append("This audit replays dated daily folders through the daily pipeline only. It does not merge trend-analysis outputs into the daily pipeline.")
    md.append("")
    if not day_counts.empty:
        approved = day_counts.get("approved", pd.Series(dtype=float)).fillna(0).astype(int)
        md.append("## Replay coverage")
        md.append("")
        md.append(f"- Complete folders replayed: {len(manifest.get('folders', []))}")
        md.append(f"- Incomplete folders excluded: {len(manifest.get('incomplete', []))}")
        md.append(f"- Run failures: {int((day_counts.get('returncode', 0) != 0).sum())}")
        md.append(f"- Report gate-pass rows: {int(approved.sum())}")
        md.append(f"- Trade days: {int((approved > 0).sum())}")
        md.append(f"- Skip days: {int((approved == 0).sum())}")
        action_counts = payload.get("action_book_counts", {})
        md.append(
            f"- Core/Tactical/Medium/Income/Pilot/Scout rows: "
            f"{action_counts.get('Core', 0)}/{action_counts.get('Tactical', 0)}/"
            f"{action_counts.get('Medium', 0)}/{action_counts.get('Income', 0)}/"
            f"{action_counts.get('Pilot', 0)}/{action_counts.get('Scout', 0)}"
        )
        md.append("")
    if summary:
        md.append("## Realized quality")
        md.append("")
        md.append(f"- Backtest setup source: {payload.get('setup_source', 'unknown')}")
        md.append(f"- Completed/open rows: {summary.get('completed_trades')}/{summary.get('open_or_skipped_trades')}")
        md.append(f"- Win rate: {fmt_pct(summary.get('win_rate'))}")
        md.append(f"- Profit factor: {fmt_pf(summary.get('profit_factor'))}")
        md.append(f"- Net P/L: ${float(summary.get('net_pnl', 0)):,.0f}")
        md.append(f"- Good/bad/flat days: {summary.get('good_days')}/{summary.get('bad_days')}/{summary.get('flat_days')}")
        summary_dd = summary.get("max_drawdown")
        if summary_dd is None and isinstance(payload.get("cost_model"), dict):
            summary_dd = payload["cost_model"].get("base", {}).get("max_drawdown")
        md.append(f"- Max drawdown: ${float(summary_dd or 0):,.0f}")
        md.append("")
    if payload.get("cost_model"):
        md.append("## Fill sensitivity")
        md.append("")
        for label, block in payload["cost_model"].items():
            md.append(f"- {label}: win {fmt_pct(block.get('win_rate'))}, PF {fmt_pf(block.get('profit_factor'))}, net ${float(block.get('net_pnl', 0)):,.0f}, max DD ${float(block.get('max_drawdown', 0)):,.0f}")
        md.append("")
    if payload.get("walk_forward"):
        md.append("## Walk-forward windows")
        md.append("")
        for label, wf in payload["walk_forward"].items():
            train = wf.get("train", {})
            test = wf.get("test", {})
            md.append(f"- {label}: train PF {fmt_pf(train.get('profit_factor'))}, test PF {fmt_pf(test.get('profit_factor'))}, test win {fmt_pct(test.get('win_rate'))}")
        md.append("")
    if payload.get("active_overlap"):
        ao = payload["active_overlap"]
        md.append("## Active overlap / concentration")
        md.append("")
        md.append(f"- Max active trades: {ao.get('max_active_trades', 0)}")
        md.append(f"- Max active ticker share: {fmt_pct(ao.get('max_ticker_share'))} ({ao.get('worst_ticker', '')})")
        md.append(f"- Max repeated-ticker share: {fmt_pct(ao.get('max_repeated_ticker_share'))} ({ao.get('worst_repeated_ticker', '')})")
        md.append(f"- Max duplicate ticker count: {ao.get('max_active_duplicate_ticker_count', 0)}")
        md.append(f"- Max active sector share: {fmt_pct(ao.get('max_sector_share'))} ({ao.get('worst_sector', '')})")
        md.append("")
    if payload.get("portfolio_simulation", {}).get("enabled"):
        sim = payload["portfolio_simulation"]
        ps = payload.get("portfolio_filtered_summary", {})
        pao = payload.get("portfolio_filtered_active_overlap", {})
        md.append("## Portfolio-simulated safe book")
        md.append("")
        md.append(f"- Rows kept/rejected: {sim.get('kept_rows', 0)}/{sim.get('rejected_rows', 0)}")
        md.append(f"- Completed/open rows: {ps.get('completed')}/{ps.get('open')}")
        md.append(f"- Win rate: {fmt_pct(ps.get('win_rate'))}")
        md.append(f"- Profit factor: {fmt_pf(ps.get('profit_factor'))}")
        md.append(f"- Net P/L: ${float(ps.get('net_pnl', 0)):,.0f}")
        md.append(f"- Max drawdown: ${float(ps.get('max_drawdown', 0) or 0):,.0f}")
        if pao:
            md.append(f"- Max active trades after caps: {pao.get('max_active_trades', 0)}")
            md.append(f"- Max active ticker share after caps: {fmt_pct(pao.get('max_ticker_share'))} ({pao.get('worst_ticker', '')})")
            md.append(f"- Max repeated-ticker share after caps: {fmt_pct(pao.get('max_repeated_ticker_share'))} ({pao.get('worst_repeated_ticker', '')})")
            md.append(f"- Max duplicate ticker count after caps: {pao.get('max_active_duplicate_ticker_count', 0)}")
        md.append("")
    for title, key in [("By execution book", "book_metrics"), ("By strategy", "strategy_metrics"), ("By month", "month_metrics"), ("By approval regime", "regime_metrics"), ("By strategy/regime", "strategy_regime_metrics")]:
        metrics = payload.get(key, {})
        if not metrics:
            continue
        md.append(f"## {title}")
        md.append("")
        for name, block in sorted(metrics.items()):
            md.append(
                f"- {name}: rows {block['rows']}, completed {block['completed']}, wins/losses {block['wins']}/{block['losses']}, "
                f"win {fmt_pct(block['win_rate'])}, PF {fmt_pf(block['profit_factor'])}, net ${float(block['net_pnl']):,.0f}"
            )
        md.append("")
    md.append("## Safety exception check")
    md.append("")
    md.append(f"- Approved rows with non-effective live gate or contra-flow hard blockers: {payload.get('invalid_approved_count', 0)}")
    md.append("")
    if payload.get("blocker_counts"):
        md.append("## Top Watch blockers")
        md.append("")
        for token, count in payload["blocker_counts"][:20]:
            md.append(f"- {count}x `{token}`")
        md.append("")
    md.append("## Output files")
    md.append("")
    for label, path in payload.get("files", {}).items():
        md.append(f"- {label}: `{path}`")
    md.append("")
    (out_root / "FULL_FOLDER_DAILY_REPLAY_AUDIT.md").write_text("\n".join(md))


def main() -> int:
    args = parse_args()
    config = args.config or (args.root / "uwos" / "rulebook_config_goal_holistic_claude.yaml")
    valuation_date = as_date(args.valuation_date) or dt.date.today()
    if not args.reuse_existing:
        run_batch(args, config)
    payload = aggregate(args, valuation_date)
    print(json.dumps({
        "summary": payload.get("summary", {}),
        "book_metrics": payload.get("book_metrics", {}),
        "invalid_approved_count": payload.get("invalid_approved_count", 0),
        "audit_md": str(args.out_root / "FULL_FOLDER_DAILY_REPLAY_AUDIT.md"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
