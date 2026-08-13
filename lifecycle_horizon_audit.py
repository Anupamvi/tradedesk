"""Search-corrected lifecycle audit for Pattern Analysis V2 long options.

Re-scores the completed cumulative validation population at 10/20/30/40
sessions using the same +50% whole-position target, no price stop, configured
fees/slippage, date-clustered PF lower bound, and matched family permutation
null. The four-horizon search is Bonferroni-corrected before any result is marked
eligible.
"""
from __future__ import annotations

import argparse
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from uwos.options_pattern_pipeline_v1 import core

HORIZONS = (10, 20, 30, 40)
PROFIT_TARGET = 0.50
MIN_SCORED = 30
MIN_UNIQUE_DATES = 20
MIN_FOLDS = 4
MIN_PROFIT_FACTOR = 1.20
MIN_CLUSTERED_P05 = 1.20
MAX_SEARCH_CORRECTED_P = 0.05
MIN_NULL_COVERAGE = 0.80
BOOTSTRAP_ITERATIONS = 1000
PERMUTATION_TRIALS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument("--as-of", default="2026-07-31")
    return parser.parse_args()


def profit_factor(values: Iterable[float]) -> Optional[float]:
    array = np.asarray(list(values), dtype=float)
    gains = array[array > 0].sum()
    losses = -array[array < 0].sum()
    if losses > 0:
        return float(gains / losses)
    return 999.0 if gains > 0 else None


def selected_store(base_dir: Path, as_of: str) -> Path:
    paths = sorted(
        (base_dir / "out" / "options_pattern_pipeline_v1" / "cache" / "selected_chain_quotes").glob(
            f"selected_chain_quotes_{as_of}_*.sqlite"
        ),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not paths:
        raise FileNotFoundError(f"no selected quote store for {as_of}")
    return paths[-1]


def load_marks(connection: sqlite3.Connection, symbols: Sequence[str]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    connection.execute("CREATE TEMP TABLE wanted_symbols(symbol TEXT PRIMARY KEY) WITHOUT ROWID")
    connection.executemany(
        "INSERT OR IGNORE INTO wanted_symbols(symbol) VALUES (?)",
        ((symbol,) for symbol in symbols),
    )
    grouped: Dict[str, List[Tuple[str, float, float]]] = defaultdict(list)
    cursor = connection.execute(
        "SELECT quotes.symbol, quotes.quote_date, quotes.bid, quotes.ask "
        "FROM quotes JOIN wanted_symbols USING(symbol) "
        "ORDER BY quotes.symbol, quotes.quote_date"
    )
    for symbol, quote_date, bid, ask in cursor:
        grouped[str(symbol)].append((str(quote_date), float(bid), float(ask)))
    marks: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for symbol, rows in grouped.items():
        marks[symbol] = (
            np.asarray([row[0] for row in rows], dtype="U10"),
            np.asarray([row[1] for row in rows], dtype=float),
            np.asarray([row[2] for row in rows], dtype=float),
        )
    return marks


def score_row(
    row: Mapping[str, Any],
    horizon: int,
    sessions: Sequence[str],
    positions: Mapping[str, int],
    marks: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    signal_date = str(row.get("signal_date") or "")
    base = {
        "split": str(row.get("split") or ""),
        "sample": "VALIDATION",
        "horizon": f"{horizon}d",
        "primary_validation_horizon": True,
        "signal_date": signal_date,
        "ticker": str(row.get("ticker") or ""),
        "direction": str(row.get("direction") or ""),
        "pattern_family": str(row.get("pattern_family") or ""),
        "market_regime": str(row.get("market_regime") or ""),
        "sector": str(row.get("sector") or ""),
        "strategy_kind": "long_option",
        "contract_profile": str(row.get("contract_profile") or ""),
        "lead_option_symbol": str(row.get("lead_option_symbol") or ""),
        "legs_json": str(row.get("legs_json") or ""),
        "status": "UNSCORABLE",
        "net_r": None,
        "win": 0,
    }
    start = positions.get(signal_date)
    if start is None or start + horizon >= len(sessions):
        base.update(status="CENSORED_OPEN", outcome_note="primary_horizon_not_yet_mature")
        return base
    target_date = sessions[start + horizon]
    base["target_date"] = target_date
    symbol = base["lead_option_symbol"]
    mark = marks.get(symbol)
    entry = core.num(row.get("entry_ask"))
    if mark is None or entry is None or entry <= 0:
        base["outcome_note"] = "missing_entry_or_quote_history"
        return base
    dates, bids, asks = mark
    left = bisect_right(dates, signal_date)
    right = bisect_right(dates, target_date)
    if right <= left:
        base["outcome_note"] = "future_option_quotes_missing"
        return base
    window_dates = dates[left:right]
    window_bids = bids[left:right]
    window_asks = asks[left:right]
    target_price = entry * (1.0 + PROFIT_TARGET)
    hits = np.flatnonzero(window_bids >= target_price)
    round_trip_fees = core.num(row.get("round_trip_fees")) or 0.0
    opening_fee = core.num(row.get("opening_fee")) or 0.0
    entry_slippage = core.num(row.get("entry_slippage")) or 0.0
    risk_dollars = entry * 100.0 + opening_fee + entry_slippage
    if len(hits):
        hit = int(hits[0])
        exit_price = target_price
        exit_slippage = 0.0
        exit_date = str(window_dates[hit])
        note = "managed_long_option_target_hit_after_costs_slippage"
    else:
        exit_price = float(window_bids[-1])
        exit_date = str(window_dates[-1])
        spread = max(float(window_asks[-1]) - exit_price, 0.0)
        exit_slippage = spread * 100.0 * 0.50
        note = "managed_long_option_horizon_bid_exit_after_costs_slippage"
    net_dollars = (
        (exit_price - entry) * 100.0
        - round_trip_fees
        - entry_slippage
        - exit_slippage
    )
    net_r = net_dollars / risk_dollars if risk_dollars > 0 else None
    base.update(
        status="SCORED",
        net_r=net_r,
        win=int(net_r is not None and net_r > 0),
        managed_exit_date=exit_date,
        managed_exit_price=exit_price,
        outcome_note=note,
    )
    return base


def summarize_family(
    family: str,
    horizon: int,
    rows: Sequence[Mapping[str, Any]],
    null_stats: Mapping[str, Any],
) -> Dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "SCORED" and core.num(row.get("net_r")) is not None]
    values = [float(row["net_r"]) for row in scored]
    split_means: Dict[str, float] = {}
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in scored:
        grouped[str(row.get("split") or "")].append(float(row["net_r"]))
    for split, split_values in grouped.items():
        split_means[split] = float(np.mean(split_values))
    positive_folds = sum(value > 0 for value in split_means.values())
    factor = profit_factor(values)
    clustered_p05 = core.day_clustered_profit_factor_p05(
        scored,
        BOOTSTRAP_ITERATIONS,
        f"lifecycle:{family}:{horizon}",
    )
    raw_p = core.num(null_stats.get("matched_null_p_value"))
    corrected_p = min(1.0, raw_p * len(HORIZONS)) if raw_p is not None else None
    unique_dates = len({str(row.get("signal_date") or "") for row in scored})
    failures = []
    if len(scored) < MIN_SCORED:
        failures.append("SCORED_LT_30")
    if unique_dates < MIN_UNIQUE_DATES:
        failures.append("UNIQUE_DATES_LT_20")
    if not values or float(np.mean(values)) <= 0:
        failures.append("AVERAGE_R_NOT_POSITIVE")
    if factor is None or factor < MIN_PROFIT_FACTOR:
        failures.append("PF_LT_1_20")
    if len(split_means) < MIN_FOLDS or positive_folds != len(split_means):
        failures.append("NOT_EVERY_MATURE_FOLD_PROFITABLE")
    if clustered_p05 is None or clustered_p05 < MIN_CLUSTERED_P05:
        failures.append("CLUSTERED_P05_LT_1_20")
    coverage = core.num(null_stats.get("matched_null_coverage"))
    if corrected_p is None or corrected_p > MAX_SEARCH_CORRECTED_P or coverage is None or coverage < MIN_NULL_COVERAGE:
        failures.append("SEARCH_CORRECTED_MATCHED_NULL_FAILED")
    return {
        "pattern_family": family,
        "horizon": horizon,
        "scored_count": len(scored),
        "unique_signal_dates": unique_dates,
        "average_net_r": float(np.mean(values)) if values else None,
        "profit_factor": factor,
        "win_rate": float(np.mean(np.asarray(values) > 0)) if values else None,
        "validation_split_count": len(split_means),
        "positive_validation_splits": positive_folds,
        "worst_split_average_net_r": min(split_means.values()) if split_means else None,
        "latest_split_average_net_r": split_means[max(split_means)] if split_means else None,
        "day_clustered_profit_factor_p05": clustered_p05,
        "matched_null_p_value": raw_p,
        "search_corrected_matched_null_p_value": corrected_p,
        "matched_null_coverage": coverage,
        "matched_null_median_profit_factor": core.num(null_stats.get("matched_null_median_profit_factor")),
        "deployment_ready": not failures,
        "failures": ";".join(failures),
    }


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    output_dir = base_dir / "out" / "pattern_analysis_v2" / args.as_of
    details = pd.read_csv(output_dir / "validation_details.csv", low_memory=False)
    details = details[
        details["strategy_kind"].eq("long_option")
        & details["sample"].eq("VALIDATION")
    ].copy()
    symbols = sorted(details["lead_option_symbol"].dropna().astype(str).unique())
    store_path = selected_store(base_dir, args.as_of)
    connection = sqlite3.connect(store_path)
    try:
        marks = load_marks(connection, symbols)
    finally:
        connection.close()
    sessions = core.scoring_session_dates(base_dir, args.as_of)
    positions = {session: index for index, session in enumerate(sessions)}
    all_summaries: List[Dict[str, Any]] = []
    outcome_path = output_dir / "lifecycle_horizon_outcomes.csv"
    summary_path = output_dir / "lifecycle_horizon_audit.csv"
    with outcome_path.open("w", encoding="utf-8", newline="") as handle:
        wrote_header = False
        for horizon in HORIZONS:
            rows = [
                score_row(row, horizon, sessions, positions, marks)
                for row in details.to_dict("records")
            ]
            nulls = core.matched_family_permutation_stats(
                rows,
                PERMUTATION_TRIALS,
                20260803 + horizon,
            )
            momentum_rows = [
                row
                for row in rows
                if str(row.get("pattern_family") or "").startswith("SECTOR_MOMENTUM")
            ]
            frame = pd.DataFrame(momentum_rows)
            frame.to_csv(handle, index=False, header=not wrote_header)
            wrote_header = True
            grouped_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
            for row in momentum_rows:
                grouped_rows[str(row["pattern_family"])].append(row)
            for family, family_rows in grouped_rows.items():
                all_summaries.append(
                    summarize_family(
                        family,
                        horizon,
                        family_rows,
                        nulls.get(family, {}),
                    )
                )
            print(
                f"[lifecycle] horizon={horizon} all_rows={len(rows)} momentum_rows={len(momentum_rows)}",
                flush=True,
            )
    summary = pd.DataFrame(all_summaries)
    summary.to_csv(summary_path, index=False)
    print("\nDEPLOYMENT READY")
    ready = summary[summary["deployment_ready"].astype(bool)]
    print(ready.to_string(index=False) if len(ready) else "NONE")
    print("\nTOP BY CLUSTERED P05")
    print(
        summary.sort_values(
            ["day_clustered_profit_factor_p05", "average_net_r"],
            ascending=False,
        ).head(30).to_string(index=False)
    )
    print(f"\nwrote {summary_path} and {outcome_path}")


if __name__ == "__main__":
    main()
