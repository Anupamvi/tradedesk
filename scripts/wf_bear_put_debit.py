"""Leakage-safe walk-forward on the Bear Put Debit family.

Reuses production payoff-calibration primitives (_eligible_history, _metrics)
so the fill-stress / profit-factor definitions match live V4 exactly.

Policy under test: "activate a route once K fully-resolved prior trades exist
and their 10%-fill-stress train PF >= 1.25; then TAKE the next trade and record
its realized 10%-fill-stress outcome as an out-of-sample (OOS) result."

We sweep K to find the lowest sample floor whose strictly-future OOS PF still
clears the live bar (MIN_STRESS_PROFIT_FACTOR = 1.25).
"""
from __future__ import annotations

import math

import pandas as pd

from codexuw.payoff_calibration import (
    MIN_STRESS_PROFIT_FACTOR,
    _eligible_history,
    _metrics,
)

HISTORY = "codexuw/history/codexdaily_v4_edge_history_v2_2026-07-10.csv.gz"
STRESS = 0.10
TRAIN_PF_FLOOR = MIN_STRESS_PROFIT_FACTOR  # 1.25
FLOORS = [5, 8, 10, 12, 15, 20]


def _load_bear_put_debit() -> pd.DataFrame:
    hist = pd.read_csv(HISTORY, low_memory=False)
    elig = _eligible_history(hist, asof=None)
    if elig.empty:
        return elig
    direction = elig.get("direction", pd.Series("", index=elig.index)).astype(str).str.lower()
    strategy = elig.get("strategy", pd.Series("", index=elig.index)).astype(str).str.lower()
    mask = (
        elig["_family"].eq("Debit")
        & (direction.str.contains("bear put") | strategy.str.contains("bear put"))
    )
    return elig[mask].sort_values(["_asof_dt", "_exit_dt"]).reset_index(drop=True)


def walk_forward(frame: pd.DataFrame, floor: int, route_col: str) -> dict[str, object]:
    """Strictly leakage-safe: training = trades fully resolved before entry."""
    taken_idx: list[int] = []
    activated_on: pd.Timestamp | None = None
    for route_key, group in frame.groupby(route_col):
        group = group.sort_values(["_asof_dt", "_exit_dt"])
        for i, (_, row) in enumerate(group.iterrows()):
            entry_dt = row["_asof_dt"]
            # training = same-route trades whose exit is on/before this entry (no leakage)
            prior = group[group["_exit_dt"] <= entry_dt]
            if len(prior) < floor:
                continue
            train = _metrics(prior, STRESS)
            if not (train["sample_size"] >= floor
                    and math.isfinite(train["profit_factor"])
                    and train["profit_factor"] >= TRAIN_PF_FLOOR):
                continue
            taken_idx.append(row.name)
            if activated_on is None or entry_dt < activated_on:
                activated_on = entry_dt
    if not taken_idx:
        return {"floor": floor, "oos_trades": 0, "oos_pf": math.nan,
                "oos_win_rate": math.nan, "oos_avg_pnl": math.nan,
                "first_activation": None}
    oos = frame.loc[taken_idx]
    m = _metrics(oos, STRESS)
    return {
        "floor": floor,
        "oos_trades": m["sample_size"],
        "oos_pf": round(m["profit_factor"], 3) if math.isfinite(m["profit_factor"]) else float("inf"),
        "oos_win_rate": round(m["win_rate"], 3),
        "oos_avg_pnl": round(m["average_pnl"], 2),
        "first_activation": str(activated_on.date()) if activated_on is not None else None,
    }


def main() -> None:
    frame = _load_bear_put_debit()
    print(f"Bear Put Debit eligible (exact-evaluated, resolved) trades: {len(frame)}")
    if frame.empty:
        return
    print("Date span:", frame["_asof_dt"].min().date(), "->", frame["_asof_dt"].max().date())
    print("Regimes present:", frame["_group_key"].str.split("|").str[-1].value_counts().to_dict())
    print()

    # regime-agnostic route: pool Bear Put Debit across all regimes
    frame = frame.copy()
    frame["_route_regime_agnostic"] = "Debit|Bear Put|ALL"

    for route_col, label in [("_route_regime_agnostic", "regime-agnostic (all regimes pooled)"),
                             ("_route_key_base", "base (Debit|BearPut|regime)"),
                             ("_route_key_flow_cost", "flow_cost (finest)")]:
        if route_col not in frame.columns:
            continue
        print(f"### Walk-forward OOS by activation floor K  —  route level: {label}")
        rows = [walk_forward(frame, k, route_col) for k in FLOORS]
        table = pd.DataFrame(rows)
        print(table.to_string(index=False))
        clears = table[(table["oos_trades"] >= 8) & (table["oos_pf"] >= TRAIN_PF_FLOOR)]
        if not clears.empty:
            best = clears.sort_values("floor").iloc[0]
            print(f"  -> lowest floor with >=8 OOS trades AND OOS PF >= {TRAIN_PF_FLOOR}: "
                  f"K={int(best['floor'])} (OOS PF {best['oos_pf']}, n={int(best['oos_trades'])})")
        else:
            print(f"  -> NO floor yields >=8 OOS trades with OOS PF >= {TRAIN_PF_FLOOR}")
        print()


if __name__ == "__main__":
    main()
