"""Re-simulate exits for the existing evidence base over a parameter grid.

The current policy (take profit at 60% of credit / +60% of debit, stop at 2x
credit / half debit) is a *policy choice*, not a market fact. This script holds
entries fixed and replays only the exit path, so the cost of that choice is
measurable in isolation.

It reuses `codexuw.replay.simulate_spread_exit` and the same cached hot-chain
quote history the replay uses, so results are directly comparable.
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path

import pandas as pd

from codexuw.replay import (
    _quote_lookup,
    dated_folders,
    load_close_history,
    load_hot_history,
    simulate_spread_exit,
)

HISTORY = Path("codexuw/history/codexdaily_v4_edge_history_v3_2026-07-23.csv.gz")
ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def pf(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if pnl.empty:
        return float("nan")
    g = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return g / l if l > 0 else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all evaluated rows")
    args = ap.parse_args()

    d = pd.read_csv(HISTORY, low_memory=False)
    d["is_eval"] = truthy(d["exact_evaluated"])
    ev = d[d["is_eval"]].copy()
    ev["asof"] = pd.to_datetime(ev["asof"], errors="coerce").dt.date
    ev["expiry"] = pd.to_datetime(ev["expiry"], errors="coerce").dt.date
    ev = ev[ev["asof"].notna() & ev["expiry"].notna()]
    if args.max_rows:
        ev = ev.head(args.max_rows)
    print(f"rows to re-simulate: {len(ev)}")

    folders = dated_folders(ROOT, None, None)
    print(f"dated folders: {len(folders)}")
    close_history = load_close_history(folders)
    hot_history = load_hot_history(folders)
    quote_history = {day: _quote_lookup(hot) for day, hot in hot_history.items()}
    print(f"quote days loaded: {len(quote_history)}  close days: {len(close_history)}")

    grid = [
        # (label, profit_take_pct, stop_loss_mult, debit_time_stop_dte, slippage)
        ("CURRENT  tp60 / sl2.0x            slip10%", 0.60, 2.0, -1, 0.10),
        ("tp50 / sl2.0x                     slip10%", 0.50, 2.0, -1, 0.10),
        ("tp50 / sl3.0x                     slip10%", 0.50, 3.0, -1, 0.10),
        ("tp50 / sl4.0x                     slip10%", 0.50, 4.0, -1, 0.10),
        ("tp50 / no stop                    slip10%", 0.50, 99.0, -1, 0.10),
        ("tp60 / no stop                    slip10%", 0.60, 99.0, -1, 0.10),
        ("tp75 / no stop                    slip10%", 0.75, 99.0, -1, 0.10),
        ("hold to expiry                    slip10%", 9.99, 99.0, -1, 0.10),
        ("CURRENT  tp60 / sl2.0x            slip05%", 0.60, 2.0, -1, 0.05),
        ("tp50 / no stop                    slip05%", 0.50, 99.0, -1, 0.05),
        ("hold to expiry                    slip05%", 9.99, 99.0, -1, 0.05),
        ("hold to expiry                    slip00%", 9.99, 99.0, -1, 0.00),
        ("tp50 / no stop                    slip00%", 0.50, 99.0, -1, 0.00),
    ]

    results = []
    for label, ptp, slm, tsd, slip in grid:
        out = []
        for _, row in ev.iterrows():
            res = simulate_spread_exit(
                row,
                close_history,
                quote_history,
                slippage_pct=slip,
                profit_take_pct=ptp,
                stop_loss_mult=slm,
                debit_time_stop_dte=tsd,
            )
            if not res.get("exact_evaluated"):
                continue
            out.append(
                {
                    "row_id": row.name,
                    "asof": row["asof"],
                    "ticker": row.get("ticker"),
                    "direction": row["direction"],
                    "regime": row.get("regime"),
                    "kind": "Credit" if row["direction"] in ("Bull Put", "Bear Call") else "Debit",
                    "pnl": res.get("pnl_1x"),
                    "reason": res.get("exit_reason"),
                }
            )
        r = pd.DataFrame(out)
        if r.empty:
            continue
        results.append((label, r))
        p = r["pnl"].dropna()
        print(
            f"\n{label:<42} n={len(p):>5} PF={pf(p):>6.3f} win={(p > 0).mean():>6.1%} "
            f"avg={p.mean():>8.2f} total={p.sum():>10.0f}"
        )
        for kind, grp in r.groupby("kind"):
            pk = grp["pnl"].dropna()
            print(
                f"    {kind:<38} n={len(pk):>5} PF={pf(pk):>6.3f} win={(pk > 0).mean():>6.1%} "
                f"avg={pk.mean():>8.2f} total={pk.sum():>10.0f}"
            )
    out_path = Path("out/redesign_exit_grid")
    out_path.mkdir(parents=True, exist_ok=True)
    for label, r in results:
        slug = label.strip().replace(" ", "_").replace("/", "-").replace("%", "pct")
        r.to_csv(out_path / f"{slug}.csv", index=False)
    print(f"\nper-config detail written to {out_path}")

    # Strict time split on the best few, to check the ranking is not in-sample only.
    print("\n" + "=" * 88)
    print("STRICT TIME SPLIT (train = first 60% of sessions, test = last 40%)")
    print("=" * 88)
    for label, r in results:
        days = sorted(pd.to_datetime(r["asof"]).unique())
        if not days:
            continue
        cut = days[int(len(days) * 0.60)]
        a = pd.to_datetime(r["asof"])
        tr = r[a < cut]["pnl"].dropna()
        te = r[a >= cut]["pnl"].dropna()
        print(
            f"{label:<42} TRAIN PF={pf(tr):>6.3f} (n={len(tr):>5})   "
            f"TEST PF={pf(te):>6.3f} (n={len(te):>5})  test_total={te.sum():>9.0f}"
        )


if __name__ == "__main__":
    main()
