"""Two questions the user is entitled to ask:
   A) Is the one surviving lane actually as strong as claimed, once trades that
      share an entry date (and therefore share a macro shock) stop counting as
      independent observations?
   B) Is it genuinely the ONLY lane, or did other sector/direction cells get
      dismissed before the validated +50% managed exit was applied to them?
No new signal is invented here. This only re-measures what already exists.
"""
import numpy as np
import pandas as pd

SPLIT = "2026-04-14"
RNG = np.random.default_rng(11)


def pf(pnl):
    pnl = np.asarray(pnl, dtype=float)
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    if l <= 0:
        return np.inf if w > 0 else np.nan
    return w / l


def main():
    t = pd.read_csv("out/symmetric_direction_test.csv", low_memory=False)
    t["half"] = np.where(t.signal_date < SPLIT, "TRAIN", "TEST")

    # ---------------- A) grid: is anything else alive? ----------------
    print("=" * 78)
    print("A) EVERY SECTOR x DIRECTION CELL, MANAGED EXIT APPLIED, SIGNAL vs RANDOM")
    print("=" * 78)
    rows = []
    for (sec, dr), g in t.groupby(["sector", "direction"]):
        rec = {"sector": sec, "direction": dr}
        ok = True
        for half in ("TRAIN", "TEST"):
            for mode in ("signal", "random"):
                sub = g[(g.half == half) & (g["mode"] == mode)]
                rec[f"{half[:2]}_{mode[:3]}_n"] = len(sub)
                rec[f"{half[:2]}_{mode[:3]}_pf"] = pf(sub.pnl) if len(sub) >= 15 else np.nan
            if rec[f"{half[:2]}_sig_n"] < 15:
                ok = False
        rec["usable"] = ok
        rows.append(rec)
    grid = pd.DataFrame(rows)
    g2 = grid[grid.usable].copy()
    # a cell is interesting only if signal beats random in BOTH halves and clears 1.2 OOS
    g2["beats_rand_train"] = g2.TR_sig_pf > g2.TR_ran_pf
    g2["beats_rand_test"] = g2.TE_sig_pf > g2.TE_ran_pf
    g2["clears_bar"] = (g2.TE_sig_pf >= 1.2) & (g2.TR_sig_pf >= 1.2)
    g2["SURVIVES"] = g2.beats_rand_train & g2.beats_rand_test & g2.clears_bar
    show = g2.sort_values("TE_sig_pf", ascending=False)[
        ["sector", "direction", "TR_sig_n", "TR_sig_pf", "TR_ran_pf",
         "TE_sig_n", "TE_sig_pf", "TE_ran_pf", "SURVIVES"]
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\ncells with enough trades to judge : {len(g2)}")
    print(f"cells surviving all three tests   : {int(g2.SURVIVES.sum())}")
    near = g2[(~g2.SURVIVES) & g2.beats_rand_test & (g2.TE_sig_pf >= 1.2)]
    if len(near):
        print("\nNEAR-MISSES (win out of sample but fail train or vs random):")
        print(near[["sector", "direction", "TR_sig_pf", "TR_ran_pf",
                    "TE_sig_pf", "TE_ran_pf"]].to_string(index=False,
                                                         float_format=lambda v: f"{v:.2f}"))

    # ------------- B) how independent are the lane's trades? -------------
    lane = t[(t.sector == "Technology") & (t.direction == "long_put")
             & (t["mode"] == "signal") & (t.cost >= 700)].copy()
    print("\n" + "=" * 78)
    print("B) IS n=114 REALLY 114 INDEPENDENT BETS?")
    print("=" * 78)
    per_date = lane.groupby("signal_date").pnl.agg(["size", "mean"])
    print(f"trades                  : {len(lane)}")
    print(f"distinct entry dates    : {lane.signal_date.nunique()}")
    print(f"trades per entry date   : median {per_date['size'].median():.0f}"
          f"  max {per_date['size'].max():.0f}")

    # do trades entered the same day move together?
    same_day_sign = lane.groupby("signal_date").pnl.apply(
        lambda x: (x > 0).mean() if len(x) >= 2 else np.nan).dropna()
    herd = ((same_day_sign <= 0.15) | (same_day_sign >= 0.85)).mean()
    print(f"entry dates where >=85% of the basket shared one outcome : {herd:.0%}")
    print("  (high = the basket is one bet, not many)")

    # day-clustered bootstrap: resample DATES, keep every trade on chosen dates
    dates = lane.signal_date.unique()
    by_date = {d: g.pnl.values for d, g in lane.groupby("signal_date")}
    boots = []
    for _ in range(4000):
        pick = RNG.choice(dates, size=len(dates), replace=True)
        boots.append(pf(np.concatenate([by_date[d] for d in pick])))
    boots = np.array([b for b in boots if np.isfinite(b)])
    p05, p50 = np.percentile(boots, [5, 50])
    print(f"\nday-clustered bootstrap PF : median {p50:.2f}   5th pct {p05:.2f}")
    print(f"deployment bar is p05 >= 1.20  ->  {'PASS' if p05 >= 1.2 else 'FAIL'}")

    # naive trade-level bootstrap, for contrast: this is what I implicitly quoted
    vals = lane.pnl.values
    nb = np.array([pf(RNG.choice(vals, size=len(vals), replace=True))
                   for _ in range(4000)])
    nb = nb[np.isfinite(nb)]
    print(f"naive trade-level p05      : {np.percentile(nb, 5):.2f}"
          f"   <- overstated if it exceeds the clustered p05")

    # every fold profitable?
    lane["month"] = lane.signal_date.str[:7]
    mo = lane.groupby("month").pnl.agg(["size", "sum"])
    mo["pf"] = lane.groupby("month").pnl.apply(pf)
    print("\nby entry month:")
    print(mo.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"months profitable: {(mo['sum'] > 0).sum()}/{len(mo)}")


if __name__ == "__main__":
    main()
