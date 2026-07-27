"""Sweep credit-spread stop-loss levels against a completed replay, reusing cached quotes.

The replay records only the *realised* exit for each row. Because a hard stop can
only ever move an exit **earlier**, the observation window
``(planned_entry_date, recorded exit_day]`` is sufficient to re-price any stop
policy without re-running discovery.

The value path for every row is extracted once and cached, so repeated sweeps are
instant.

Limitation: take-profit levels that would hold *longer* than the realised exit
cannot be evaluated (those quotes were never fetched), so only
``tp_remaining >= CREDIT_TAKE_PROFIT_REMAINING`` is representable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from uwos.exact_spread_backtester import HistoricalOptionQuoteStore
from uwos.options_agent import core
from uwos.options_agent import replay as rp


def _date(value: Any) -> Optional[dt.date]:
    return rp._date(value)


def build_paths(
    detail: pd.DataFrame, root: Path, full_horizon: bool = False
) -> dict[str, list[tuple[str, float]]]:
    """Extract the per-session spread value path for every evaluated CREDIT row.

    With ``full_horizon`` the window runs to the original planned horizon,
    ``min(entry + FIXED_HORIZON_SESSIONS, expiry)``, rather than stopping at the
    realised exit. That is required to price take-profit levels that would have
    held *longer* than the recorded exit.
    """

    store = HistoricalOptionQuoteStore(root, use_hot=True, use_oi=True)
    jobs: list[dict[str, Any]] = []
    by_day: dict[dt.date, set[str]] = {}
    for _, row in detail.iterrows():
        entry_day = _date(row.get("planned_entry_date"))
        exit_day = _date(row.get("exit_day"))
        short_sym = str(row.get("short_leg_eod") or "").upper()
        long_sym = str(row.get("long_leg_eod") or "").upper()
        if entry_day is None or exit_day is None or not short_sym or not long_sym:
            continue
        if full_horizon:
            expiry = _date(row.get("expiry"))
            horizon = core._add_regular_market_days(entry_day, rp.FIXED_HORIZON_SESSIONS)
            if expiry is not None:
                horizon = min(horizon, expiry)
        else:
            horizon = exit_day
        dates = rp._exit_observation_dates(entry_day, horizon)
        if not dates:
            continue
        jobs.append(
            {
                "id": str(row["replay_row_id"]),
                "dates": dates,
                "short": short_sym,
                "long": long_sym,
                "width": float(row.get("entry_width") or 0.0),
                "side": str(row.get("entry_side") or "").upper(),
            }
        )
        for day in dates:
            by_day.setdefault(day, set()).update((short_sym, long_sym))

    print(f"  loading quotes for {len(by_day)} distinct sessions ...", flush=True)
    index: dict[dt.date, dict[str, Any]] = {}
    for n, (day, symbols) in enumerate(sorted(by_day.items()), start=1):
        index[day] = rp._quote_index(store, day, symbols)
        if n % 25 == 0:
            print(f"    {n}/{len(by_day)} sessions", flush=True)

    paths: dict[str, list[tuple[str, float]]] = {}
    for job in jobs:
        path: list[tuple[str, float]] = []
        for day in job["dates"]:
            quotes = index.get(day, {})
            short_q, long_q = quotes.get(job["short"]), quotes.get(job["long"])
            if short_q is None or long_q is None:
                continue
            raw_bid, raw_ask, _ = rp._spread_quotes(job["side"], short_q, long_q)
            bounded = rp._bounded_exit_market(raw_bid, raw_ask, job["width"])
            if bounded is None:
                continue
            path.append((day.isoformat(), rp._exit_value(job["side"], *bounded)))
        if path:
            paths[job["id"]] = path
    return paths


def simulate(
    row: pd.Series,
    path: list[tuple[str, float]],
    stop_multiplier: Optional[float],
    tp_remaining: float,
) -> Optional[dict[str, Any]]:
    entry = float(row.get("executed_entry_price") or 0.0)
    width = float(row.get("entry_width") or 0.0)
    if entry <= 0 or width <= 0 or not path:
        return None
    target = round(entry * tp_remaining, 6)
    stop = round(min(width, entry * stop_multiplier), 6) if stop_multiplier else None
    last = len(path) - 1
    for i, (day, value) in enumerate(path):
        trigger = rp._management_trigger(
            "CREDIT", value, target, stop, final_session=(i == last)
        )
        if not trigger:
            continue
        return {
            "exit_day": day,
            "exit_value": value,
            "exit_trigger": trigger,
            "holding_sessions": i + 1,
            "pnl_1x": rp._pnl("CREDIT", entry, value),
        }
    return None


def pf(values: pd.Series) -> float:
    gains = values[values > 0].sum()
    losses = -values[values < 0].sum()
    return gains / losses if losses > 0 else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--cache", default="/tmp/stop_sweep_paths.json.gz")
    ap.add_argument("--split-day", default="2026-05-01")
    ap.add_argument("--selected-only", action="store_true")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument(
        "--tp-sweep",
        action="store_true",
        help="rebuild paths to the full planned horizon and sweep take-profit levels",
    )
    args = ap.parse_args()

    detail = pd.read_csv(
        Path(args.replay_dir) / "options_agent_replay_detail.csv", low_memory=False
    )
    detail["asof"] = detail["asof"].astype(str)
    mask = (detail["exact_evaluated"] == True) & (detail["strategy_kind"] == "CREDIT")  # noqa: E712
    if args.selected_only:
        mask &= detail["selected_for_policy"].map(core._truthy)
    credit = detail[mask].copy()
    print(f"CREDIT rows: {len(credit)}  ({'selected only' if args.selected_only else 'all evaluated'})")

    cache = Path(args.cache)
    if cache.exists() and not args.rebuild:
        print(f"reusing cached value paths: {cache}")
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            paths = json.load(fh)
    else:
        print("extracting value paths from the historical quote store ...")
        paths = build_paths(credit, Path(args.root), full_horizon=args.tp_sweep)
        with gzip.open(cache, "wt", encoding="utf-8") as fh:
            json.dump(paths, fh)
        print(f"cached -> {cache}")
    print(f"value paths reconstructed for {len(paths)}/{len(credit)} rows")

    credit = credit[credit["replay_row_id"].astype(str).isin(paths)].copy()
    credit["seg"] = np.where(credit["asof"] < args.split_day, "train", "held")

    base = credit["pnl_1x"]
    print(
        f"\nRECORDED baseline: n={len(credit)} win={100 * (base > 0).mean():.1f}% "
        f"PF={pf(base):.3f} total=${base.sum():.0f}"
    )

    grid: list[Optional[float]] = [None, 4.0, 3.0, 2.5, 2.0, 1.75, 1.5, 1.25]
    print(f"\n{'stop':>6s} {'n':>4s} {'win%':>6s} {'PF':>7s} {'TOTAL$':>9s} {'stopped':>8s} "
          f"{'avgLoss':>8s} {'train$':>8s} {'held$':>8s} {'heldPF':>7s}")
    print("-" * 82)
    rows: list[dict[str, Any]] = []
    for stop_mult in grid:
        out = []
        for _, row in credit.iterrows():
            res = simulate(row, paths[str(row["replay_row_id"])], stop_mult, rp.CREDIT_TAKE_PROFIT_REMAINING)
            if res is None:
                continue
            res["seg"] = row["seg"]
            res["asof"] = row["asof"]
            out.append(res)
        sim = pd.DataFrame(out)
        if sim.empty:
            continue
        p = sim["pnl_1x"]
        losses = p[p < 0]
        tr, hd = sim[sim["seg"] == "train"]["pnl_1x"], sim[sim["seg"] == "held"]["pnl_1x"]
        label = "none" if stop_mult is None else f"{stop_mult:.2f}x"
        print(
            f"{label:>6s} {len(sim):4d} {100 * (p > 0).mean():5.1f}% {pf(p):7.3f} {p.sum():9.0f} "
            f"{int((sim['exit_trigger'] == 'stop_loss').sum()):8d} {losses.mean() if len(losses) else 0:8.0f} "
            f"{tr.sum():8.0f} {hd.sum():8.0f} {pf(hd):7.3f}"
        )
        rows.append({"stop": label, "sim": sim})

    print("\n=== month-by-month total $ by stop level ===")
    months = sorted(credit["asof"].str[:7].unique())
    header = "  ".join(f"{m:>8s}" for m in months)
    print(f"{'stop':>6s}  {header}")
    for item in rows:
        sim = item["sim"]
        cells = []
        for mo in months:
            cells.append(f"{sim[sim['asof'].str[:7] == mo]['pnl_1x'].sum():8.0f}")
        print(f"{item['stop']:>6s}  " + "  ".join(cells))

    if not args.tp_sweep:
        return

    print("\n=== TAKE-PROFIT SWEEP (no stop; lower tp_remaining = hold for more credit) ===")
    print(f"{'tp_rem':>7s} {'capture':>8s} {'n':>4s} {'win%':>6s} {'PF':>7s} {'TOTAL$':>9s} "
          f"{'avgWin':>7s} {'avgLoss':>8s} {'hold':>6s} {'train$':>8s} {'held$':>8s} {'heldPF':>7s}")
    print("-" * 100)
    tp_rows: list[dict[str, Any]] = []
    for tp in [0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.0]:
        out = []
        for _, row in credit.iterrows():
            res = simulate(row, paths[str(row["replay_row_id"])], None, tp)
            if res is None:
                continue
            res["seg"] = row["seg"]
            res["asof"] = row["asof"]
            out.append(res)
        sim = pd.DataFrame(out)
        if sim.empty:
            continue
        p = sim["pnl_1x"]
        wins, losses = p[p > 0], p[p < 0]
        tr, hd = sim[sim["seg"] == "train"]["pnl_1x"], sim[sim["seg"] == "held"]["pnl_1x"]
        print(
            f"{tp:7.2f} {1 - tp:7.0%} {len(sim):4d} {100 * (p > 0).mean():5.1f}% {pf(p):7.3f} "
            f"{p.sum():9.0f} {wins.mean() if len(wins) else 0:7.2f} "
            f"{losses.mean() if len(losses) else 0:8.0f} {sim['holding_sessions'].mean():6.1f} "
            f"{tr.sum():8.0f} {hd.sum():8.0f} {pf(hd):7.3f}"
        )
        tp_rows.append({"tp": f"{tp:.2f}", "sim": sim})

    print("\n=== month-by-month total $ by take-profit level ===")
    print(f"{'tp_rem':>7s}  {header}")
    for item in tp_rows:
        sim = item["sim"]
        cells = [f"{sim[sim['asof'].str[:7] == mo]['pnl_1x'].sum():8.0f}" for mo in months]
        print(f"{item['tp']:>7s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
