"""Measure what the next-session reprice gate costs in throughput and dollars.

The replay selects ~36 credit trades/month but only ~7.5 survive
``next_session_reprice_approved``. Rejected rows are never scored, so the gate's
value has never been measured -- it has only been assumed.

A rejection with reason ``next_session_credit_below_source_target`` means the
structure was still quotable next session, just at a *worse* credit. That fill is
knowable: for a CREDIT entry production takes the bid, so the achievable credit is
``round(next_session_bid, 2)``.

This script re-prices those rejected trades at that worse fill under the shipped
management policy (tp=CREDIT_TAKE_PROFIT_REMAINING, no stop, hold to
``min(entry + PLANNED_TRADE_HOLDING_SESSIONS, expiry)``) and compares them to the
approved book.

Validation gate: the approved cohort must reproduce the replay's recorded P&L.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from uwos.exact_spread_backtester import HistoricalOptionQuoteStore
from uwos.options_agent import core
from uwos.options_agent import replay as rp


def _date(value: Any) -> Optional[dt.date]:
    return rp._date(value)


def _truthy(series: pd.Series) -> pd.Series:
    return (series == True) | (series.astype(str).str.lower() == "true")  # noqa: E712


def build_paths(jobs: list[dict[str, Any]], root: Path) -> dict[str, list[tuple[str, float]]]:
    """Extract the per-session spread value path for each job to its full horizon."""

    store = HistoricalOptionQuoteStore(root, use_hot=True, use_oi=True)
    by_day: dict[dt.date, set[str]] = {}
    for job in jobs:
        for day in job["dates"]:
            by_day.setdefault(day, set()).update((job["short"], job["long"]))

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
            raw_bid, raw_ask, _ = rp._spread_quotes("CREDIT", short_q, long_q)
            bounded = rp._bounded_exit_market(raw_bid, raw_ask, job["width"])
            if bounded is None:
                continue
            path.append((day.isoformat(), rp._exit_value("CREDIT", *bounded)))
        if path:
            paths[job["id"]] = path
    return paths


def make_jobs(frame: pd.DataFrame, entry_col: str) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        entry_day = _date(row.get("planned_entry_date"))
        expiry = _date(row.get("expiry"))
        short_sym = str(row.get("short_leg_eod") or "").upper()
        long_sym = str(row.get("long_leg_eod") or "").upper()
        width = float(row.get("entry_width") or 0.0)
        entry = float(row.get(entry_col) or 0.0)
        if entry_day is None or not short_sym or not long_sym:
            continue
        if width <= 0 or entry <= 0 or entry >= width:
            continue
        horizon = core._add_regular_market_days(entry_day, rp.FIXED_HORIZON_SESSIONS)
        if expiry is not None:
            horizon = min(horizon, expiry)
        dates = rp._exit_observation_dates(entry_day, horizon)
        if not dates:
            continue
        jobs.append(
            {
                "id": str(row["replay_row_id"]),
                "dates": dates,
                "short": short_sym,
                "long": long_sym,
                "width": width,
                "entry": entry,
            }
        )
    return jobs


def simulate(job: dict[str, Any], path: list[tuple[str, float]], tp_remaining: float) -> dict[str, Any]:
    entry = job["entry"]
    target = round(entry * tp_remaining, 6)
    last = len(path) - 1
    for i, (day, value) in enumerate(path):
        trigger = rp._management_trigger("CREDIT", value, target, None, final_session=(i == last))
        if not trigger:
            continue
        return {
            "id": job["id"],
            "exit_day": day,
            "exit_trigger": trigger,
            "holding_sessions": i + 1,
            "pnl_1x": rp._pnl("CREDIT", entry, value),
        }
    return {}


def describe(label: str, pnl: pd.Series, months: float) -> None:
    if pnl.empty:
        print(f"{label:<34} n=0")
        return
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    pf = wins.sum() / abs(losses.sum()) if not losses.empty else float("inf")
    print(
        f"{label:<34} n={len(pnl):<5d} win {100 * (pnl > 0).mean():5.1f}%  "
        f"PF {pf:6.3f}  total ${pnl.sum():>9,.0f}  avg ${pnl.mean():7.2f}  "
        f"{len(pnl) / months:5.1f} trades/mo  ${pnl.sum() / months:>7,.0f}/mo"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    parser.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument("--tp", type=float, default=core.CREDIT_TAKE_PROFIT_REMAINING)
    parser.add_argument(
        "--fill-sweep",
        action="store_true",
        help="Sweep the assumed entry fill from next-session bid to mid.",
    )
    args = parser.parse_args(argv)

    detail = pd.read_csv(Path(args.replay_dir) / "options_agent_replay_detail.csv", low_memory=False)
    months = detail["asof"].nunique() / 21.0

    sel = detail[_truthy(detail["selected_for_policy"])].copy()
    sel["approved"] = _truthy(sel["next_session_reprice_approved"])
    approved = sel[sel["approved"]]
    rejected = sel[
        (~sel["approved"])
        & (sel["next_session_reprice_reason"] == "next_session_credit_below_source_target")
    ].copy()
    rejected["worse_fill"] = rejected["next_session_bid"].astype(float).round(2)

    print(f"replay window: {detail['asof'].nunique()} sessions ({months:.1f} months)")
    print(f"selected={len(sel)}  approved={len(approved)}  rejected(credit-below-target)={len(rejected)}")
    print()

    print("[1/2] approved cohort (validation gate) ...", flush=True)
    a_jobs = make_jobs(approved, "executed_entry_price")
    a_paths = build_paths(a_jobs, Path(args.root))
    a_res = [r for j in a_jobs if (r := simulate(j, a_paths.get(j["id"], []), args.tp))]
    a_pnl = pd.Series([r["pnl_1x"] for r in a_res])

    recorded = approved["pnl_1x"].dropna()
    print(f"  recorded  n={len(recorded)} total ${recorded.sum():,.0f}")
    print(f"  reproduced n={len(a_pnl)} total ${a_pnl.sum():,.0f}")
    drift = abs(a_pnl.sum() - recorded.sum())
    print(f"  drift ${drift:,.0f} -> {'OK' if drift < 1.0 else 'MISMATCH - do not trust the rejected cohort'}")
    print()

    print("[2/2] rejected cohort at the WORSE next-session fill ...", flush=True)
    r_jobs = make_jobs(rejected, "worse_fill")
    r_paths = build_paths(r_jobs, Path(args.root))
    r_res = [r for j in r_jobs if (r := simulate(j, r_paths.get(j["id"], []), args.tp))]
    r_pnl = pd.Series([r["pnl_1x"] for r in r_res])
    print()

    haircut = (
        1 - (rejected["worse_fill"] / rejected["entry_credit"].astype(float))
    ).replace([float("inf"), float("-inf")], pd.NA).dropna()
    print(f"credit haircut accepted: median {100 * haircut.median():.1f}%  mean {100 * haircut.mean():.1f}%")
    print()

    print("=" * 104)
    describe("APPROVED (shipped book)", a_pnl, months)
    describe("REJECTED @ bid fill", r_pnl, months)
    describe("COMBINED (gate removed)", pd.concat([a_pnl, r_pnl], ignore_index=True), months)
    print("=" * 104)

    if args.fill_sweep:
        print()
        print("FILL SENSITIVITY -- entry = bid + f*(mid-bid); f=0 is the EOD bid, f=1.0 the EOD mid")
        print("(paths are fill-independent, so this reuses the quotes already loaded)")
        print()
        bid = rejected["next_session_bid"].astype(float)
        mid = rejected["next_session_mid"].astype(float)
        for frac in (0.0, 0.25, 0.50, 0.75, 1.0):
            rejected["swept_fill"] = (bid + frac * (mid - bid)).round(2)
            hc = (1 - (rejected["swept_fill"] / rejected["entry_credit"].astype(float)))
            hc = hc.replace([float("inf"), float("-inf")], pd.NA).dropna()
            jobs = make_jobs(rejected, "swept_fill")
            res = [r for j in jobs if (r := simulate(j, r_paths.get(j["id"], []), args.tp))]
            pnl = pd.Series([r["pnl_1x"] for r in res])
            combined = pd.concat([a_pnl, pnl], ignore_index=True)
            label = f"f={frac:.2f} (haircut {100 * hc.median():4.1f}%)"
            describe(label, pnl, months)
            describe(f"   -> combined book", combined, months)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
