"""Can a composite of the surviving UW signals be traded, and with what instrument?

Signal signs are fitted on the FIRST half of the sample and applied unchanged to
the SECOND half, so the reported spread is genuinely out of sample. Rebalance is
non-overlapping at the holding horizon, so overlapping windows cannot inflate
the t-stat.

Prints the gross decile spread in stock-return terms, which is the number that
must clear the option round-trip cost before any option expression is viable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
HOLD = 21
MIN_NAMES = 50

CANDIDATES = [
    "pos_52w",
    "dp_block_share",
    "oi_median_dte",
    "oi_n_chains",
    "oi_built_contracts",
    "iv_rank",
    "hc_opening_share",
    "dp_prints",
    "oi_newshort_premium",
    "put_call_ratio",
    "call_vol_surge",
    "dp_bias",
]


def zscore(g: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = g.copy()
    for c in cols:
        v = out[c]
        sd = v.std()
        out[c + "_z"] = 0.0 if (not np.isfinite(sd) or sd == 0) else (v - v.mean()) / sd
    return out


def main() -> int:
    panel = pd.read_csv(
        PANEL, low_memory=False, usecols=["date", "ticker", "close", "marketcap"] + CANDIDATES
    )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"])
    panel[f"fwd"] = panel.groupby("ticker")["close"].shift(-HOLD) / panel["close"] - 1.0
    panel = panel[panel["marketcap"].fillna(0) > 2e9]
    panel = panel.dropna(subset=["fwd"])

    dates = sorted(panel.date.unique())
    split = dates[len(dates) // 2]
    print(f"days={len(dates)} in-sample<{split.date()} oos>={split.date()}")
    print(
        f"non-overlapping {HOLD}d periods available: {len(dates)//HOLD} total "
        f"(~{len(dates)//HOLD//2} per half) -- rebalancing daily with "
        f"Newey-West(lag={HOLD}) instead"
    )

    ins = panel[panel.date < split]
    oos = panel[panel.date >= split]

    # Fit signs in-sample.
    signs = {}
    for c in CANDIDATES:
        sub = ins[[c, "fwd"]].dropna()
        if len(sub) < 200:
            continue
        ic = stats.spearmanr(sub[c], sub["fwd"]).correlation
        if np.isfinite(ic) and abs(ic) > 0.005:
            signs[c] = np.sign(ic)
    print(f"signals kept: { {k: int(v) for k, v in signs.items()} }")
    if not signs:
        print("no signals survived in-sample")
        return 1

    cols = list(signs)
    rows = []
    for label, part in (("IN-SAMPLE", ins), ("OUT-OF-SAMPLE", oos)):
        recs = []
        for d, g in part.groupby("date"):
            g = g.dropna(subset=cols + ["fwd"])
            if len(g) < MIN_NAMES:
                continue
            g = zscore(g, cols)
            g["score"] = sum(signs[c] * g[c + "_z"] for c in cols) / len(cols)
            g["decile"] = pd.qcut(g["score"].rank(method="first"), 10, labels=False)
            top = g[g.decile == 9]["fwd"].mean()
            bot = g[g.decile == 0]["fwd"].mean()
            recs.append({"date": d, "top": top, "bottom": bot, "ls": top - bot, "n": len(g)})
        r = pd.DataFrame(recs)
        if r.empty:
            continue
        # Newey-West standard error: daily rebalance with HOLD-day overlap.
        x = r.ls.values - r.ls.mean()
        n = len(x)
        gamma0 = (x @ x) / n
        var = gamma0
        for lag in range(1, min(HOLD, n - 1) + 1):
            cov = (x[lag:] @ x[:-lag]) / n
            var += 2.0 * (1.0 - lag / (HOLD + 1.0)) * cov
        se = np.sqrt(max(var, 1e-12) / n)
        t = r.ls.mean() / se
        rows.append(
            {
                "sample": label,
                "days": n,
                "top_decile_ret": r.top.mean(),
                "bottom_decile_ret": r.bottom.mean(),
                "long_short": r.ls.mean(),
                "t_stat_NW": t,
                "hit_rate": (r.ls > 0).mean(),
            }
        )

    res = pd.DataFrame(rows)
    print(f"\n=== COMPOSITE, {HOLD}-day hold, daily rebalance, Newey-West t ===")
    print(res.round(4).to_string(index=False))

    oos_row = res[res["sample"] == "OUT-OF-SAMPLE"]
    if not oos_row.empty:
        ls = float(oos_row.long_short.iloc[0])
        print(f"\nOOS long-short per {HOLD}d period: {ls*100:.2f}%")
        print(f"OOS annualized (~{252//HOLD} periods): {ls*(252//HOLD)*100:.1f}%")
        print("\nOption round-trip hurdle for comparison: 8.5% median, 15.3% mean")
        print(f"Stock round-trip cost for comparison:  ~0.05%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
