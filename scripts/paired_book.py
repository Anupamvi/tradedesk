"""Run the two Technology lanes as ONE book instead of one directional bet.

Long calls on the strongest-momentum names, long puts on the weakest, same day,
same sector. That is a cross-sectional momentum factor expressed in options, and
it is roughly delta-balanced at the book level, so a tech rally no longer wipes
out the whole allocation.

Tested against the same date/sector/count-matched random control, and with a
day-clustered bootstrap, because trades sharing an entry date share a shock.
"""
import numpy as np
import pandas as pd

SPLIT = "2026-04-14"
COST_FLOOR = 700.0
RNG = np.random.default_rng(7)


def pf(pnl):
    pnl = np.asarray(pnl, dtype=float)
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    if l <= 0:
        return np.inf if w > 0 else np.nan
    return w / l


def clustered_p05(frame, iters=4000):
    dates = frame.signal_date.unique()
    by = {d: g.pnl.values for d, g in frame.groupby("signal_date")}
    out = []
    for _ in range(iters):
        pick = RNG.choice(dates, size=len(dates), replace=True)
        v = pf(np.concatenate([by[d] for d in pick]))
        if np.isfinite(v):
            out.append(v)
    return np.percentile(out, 5), np.percentile(out, 50)


def describe(name, frame):
    frame = frame.copy()
    frame["m"] = frame.signal_date.str[:7]
    mo = frame.groupby("m").pnl.agg(n="size", pnl="sum")
    mo["pf"] = frame.groupby("m").pnl.apply(pf)
    p05, p50 = clustered_p05(frame)
    te = frame[frame.signal_date >= SPLIT]
    tr = frame[frame.signal_date < SPLIT]
    print(f"\n--- {name} ---")
    print(mo.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"  months profitable      : {(mo.pnl > 0).sum()}/{len(mo)}")
    print(f"  TRAIN pf {pf(tr.pnl):.2f} (n={len(tr)})   TEST pf {pf(te.pnl):.2f} (n={len(te)})")
    print(f"  total pnl              : ${frame.pnl.sum():,.0f} over {len(frame)} trades")
    print(f"  day-clustered PF       : median {p50:.2f}   p05 {p05:.2f}")
    return {
        "book": name,
        "n": len(frame),
        "pnl": frame.pnl.sum(),
        "pf": pf(frame.pnl),
        "train_pf": pf(tr.pnl),
        "test_pf": pf(te.pnl),
        "months_profitable": int((mo.pnl > 0).sum()),
        "months": len(mo),
        "clustered_p05": p05,
    }


def main():
    t = pd.read_csv("out/symmetric_direction_test.csv", low_memory=False)
    t = t[(t.sector == "Technology") & (t.cost >= COST_FLOOR)].copy()

    results = []
    for mode in ("signal", "random"):
        m = t[t["mode"] == mode]
        results.append(describe(f"{mode}: puts only", m[m.direction == "long_put"]))
        results.append(describe(f"{mode}: calls only", m[m.direction == "long_call"]))
        results.append(describe(f"{mode}: BOTH (paired book)", m))

    r = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("SUMMARY -- does pairing survive where a single direction does not?")
    print("=" * 78)
    print(r.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    print("\nDEPLOYMENT BAR: TRAIN & TEST pf >= 1.2, clustered p05 >= 1.2, every month profitable")
    for _, row in r[r.book.str.startswith("signal")].iterrows():
        checks = {
            "train>=1.2": row.train_pf >= 1.2,
            "test>=1.2": row.test_pf >= 1.2,
            "p05>=1.2": row.clustered_p05 >= 1.2,
            "all months +": row.months_profitable == row.months,
        }
        verdict = "DEPLOYABLE" if all(checks.values()) else "fails: " + ", ".join(
            k for k, v in checks.items() if not v)
        print(f"  {row.book:32s} {verdict}")

    # is the paired signal book actually better than the paired random book?
    sig = r[r.book == "signal: BOTH (paired book)"].iloc[0]
    ran = r[r.book == "random: BOTH (paired book)"].iloc[0]
    print(f"\npaired book signal pf {sig.pf:.2f}  vs  matched random pf {ran.pf:.2f}")
    r.to_csv("out/paired_book.csv", index=False)
    print("wrote out/paired_book.csv")


if __name__ == "__main__":
    main()
