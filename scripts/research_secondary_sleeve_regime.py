"""Should the secondary-income sleeve be forced through the credit regime map?

On 2026-07-24 (a `range` session) the pipeline emitted 8 Bear Call spreads via
`decision_tier = secondary_income`, at DTE 7 and 28. The credit regime map
blocks Bear Call outside `uptrend`, so the sleeve is plainly bypassing it. The
tempting fix is "apply the regime map to the sleeve too".

Do NOT do that on the unconditional number. `range` + Bear Call is PF 0.81
overall, but `range` + Bear Call + DTE<=10 is PF 1.11, and the DTE curve is
U-shaped (0-7 PF 1.34, 15-21 PF 0.71, 28-35 PF 1.24). The sleeve selects a
specific corner -- short dated, close to the expected move, credit 25-30% --
and that corner has to be measured on its own terms.

This is the same trap as the IV/HV 1.30 gate: an unconditional ranking that
inverts once you condition on the population actually traded.

Day-clustered bootstrap throughout (resample sessions, not trades).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codexuw.credit_policy import (  # noqa: E402
    ALLOWED_REGIMES,
    MAX_CREDIT_PCT_WIDTH,
    MIN_CREDIT_PCT_WIDTH,
)

HIST = "codexuw/history/codexdaily_v4_edge_history_v4_2026-07-26.csv.gz"
PANEL = "/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz"
BOOT = 3000
RNG = np.random.default_rng(20260726)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def rep(label: str, grp: pd.DataFrame) -> None:
    if not len(grp):
        print(f"  {label:<40s} n    0")
        return
    wins = grp.loc[grp["pnl"] > 0, "pnl"].sum()
    loss = -grp.loc[grp["pnl"] <= 0, "pnl"].sum()
    pf = wins / loss if loss > 0 else float("inf")
    print(
        f"  {label:<40s} n {len(grp):4d}  days {grp['asof'].nunique():3d}"
        f"  win {100*(grp['pnl']>0).mean():5.1f}%  avg {grp['pnl'].mean():+7.1f}"
        f"  PF {pf:5.2f}  total {grp['pnl'].sum():+8.0f}"
    )


def day_boot_mean(grp: pd.DataFrame) -> tuple[float, float, float]:
    days = {str(a): g["pnl"].to_numpy(dtype=float) for a, g in grp.groupby("asof")}
    keys = list(days)
    obs = float(np.concatenate([days[k] for k in keys]).mean())
    draws = np.empty(BOOT)
    for i in range(BOOT):
        pick = RNG.integers(0, len(keys), len(keys))
        draws[i] = np.concatenate([days[keys[j]] for j in pick]).mean()
    return obs, float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def main() -> None:
    hist = pd.read_csv(HIST, low_memory=False)
    hist = hist[
        hist["evaluated"].astype(str).str.lower().eq("true")
        & hist["strategy_kind"].eq("Credit")
    ].copy()
    hist["pnl"] = _num(hist, "pnl_1x")
    hist = hist[hist["pnl"].notna()].copy()

    panel = pd.read_csv(PANEL, low_memory=False)[["asof", "ticker", "rv21_ann"]]
    hist = hist.merge(panel.drop_duplicates(["asof", "ticker"]), on=["asof", "ticker"], how="left")
    hist["rv"] = _num(hist, "rv21_ann")
    hist["ratio_ivhv"] = _num(hist, "iv30d") / hist["rv"]

    spot = _num(hist, "stock_price_eod")
    strike = _num(hist, "short_strike_eod")
    dte = _num(hist, "dte")
    em = spot * _num(hist, "iv30d") * np.sqrt(dte.clip(lower=1) / 365.0)
    hist["em_ratio"] = ((spot - strike).abs() / em).replace([np.inf, -np.inf], np.nan)
    sign = np.where(hist["direction"].eq("Bull Put"), 1.0, -1.0)
    hist["align"] = _num(hist, "combined_flow_bias") * sign
    hist["credit_pct"] = _num(hist, "entry_credit_pct_width")
    hist["dte_n"] = dte
    hist["map_ok"] = [
        str(r["regime"]) in ALLOWED_REGIMES.get(str(r["direction"]), set())
        for _, r in hist.iterrows()
    ]

    # the sleeve's own conditions, as implemented in _secondary_income_eligible
    sleeve = hist[
        hist["credit_pct"].between(MIN_CREDIT_PCT_WIDTH, MAX_CREDIT_PCT_WIDTH)
        & hist["em_ratio"].between(0.20, 0.65, inclusive="left")
        & (hist["align"] >= 0.12)
        & (hist["dte_n"] <= 35)
        & (hist["ratio_ivhv"] >= 0.90)
        & (hist["rv"] >= 0.15)
    ].copy()

    print("=== the secondary-income sleeve as actually specified ===")
    rep("sleeve, all regimes (SHIPPED TODAY)", sleeve)
    rep("  of which map-ALLOWED regime", sleeve[sleeve["map_ok"]])
    rep("  of which map-BLOCKED regime", sleeve[~sleeve["map_ok"]])
    print()
    print("  -> if map-BLOCKED is clearly negative, the regime map belongs on the sleeve.")
    print("     if it is not, forcing the map on would delete trades for no reason.")

    blocked = sleeve[~sleeve["map_ok"]]
    if len(blocked) >= 20:
        obs, lo, hi = day_boot_mean(blocked)
        print(
            f"\n  map-BLOCKED sleeve trades, day-clustered mean {obs:+.1f}"
            f"  90% CI [{lo:+.1f}, {hi:+.1f}]  p(mean<=0) "
            f"{'n/a' if lo is None else ''}"
        )
        days = {str(a): g["pnl"].to_numpy(dtype=float) for a, g in blocked.groupby("asof")}
        keys = list(days)
        draws = np.array(
            [
                np.concatenate([days[keys[j]] for j in RNG.integers(0, len(keys), len(keys))]).mean()
                for _ in range(BOOT)
            ]
        )
        print(f"  p(map-blocked sleeve mean <= 0) = {float((draws <= 0).mean()):.3f}")

    print("\n=== is it just short DTE doing the work? ===")
    for lo_d, hi_d in [(0, 10), (11, 21), (22, 27), (28, 35)]:
        rep(f"sleeve DTE {lo_d}-{hi_d}", sleeve[sleeve["dte_n"].between(lo_d, hi_d)])

    print("\n=== the specific cell the live run produced (range + Bear Call) ===")
    rc = hist[hist["regime"].eq("range") & hist["direction"].eq("Bear Call")]
    rep("range BC, ALL candidates", rc)
    rep("range BC, sleeve conditions", sleeve[sleeve["regime"].eq("range") & sleeve["direction"].eq("Bear Call")])

    print("\n=== monthly stability of the map-BLOCKED sleeve slice ===")
    if len(blocked):
        b = blocked.copy()
        b["m"] = b["asof"].str[:7]
        for m, g in b.groupby("m"):
            print(f"    {m}  n {len(g):3d}  total {g['pnl'].sum():+8.0f}  PF "
                  f"{(g.loc[g['pnl']>0,'pnl'].sum() / max(1e-9, -g.loc[g['pnl']<=0,'pnl'].sum())):.2f}")

    # ------------------------------------------------------------------
    # The sleeve's DTE profile is U-shaped (0-10 PF 2.90, 11-21 PF 0.72,
    # 22-27 PF 0.72, 28-35 PF 2.08). Carving out the middle is tempting and is a
    # TIGHTENING, so it is permissible in principle -- but a U-shape is by
    # definition non-monotone, and this is exactly the shape that got the IV/HV
    # 1.30 gate rejected. Test it properly before touching anything.
    # ------------------------------------------------------------------
    print("\n=== would excluding the sleeve's middle DTE band actually help? ===")

    def _days(g: pd.DataFrame) -> dict[str, np.ndarray]:
        return {str(a): x["pnl"].to_numpy(dtype=float) for a, x in g.groupby("asof")}

    keep = sleeve[~sleeve["dte_n"].between(11, 27)]
    dropped = sleeve[sleeve["dte_n"].between(11, 27)]
    rep("sleeve as shipped", sleeve)
    rep("sleeve minus DTE 11-27", keep)
    rep("  (the slice that would be cut)", dropped)

    db, dk = _days(sleeve), _days(keep)
    allk = sorted(set(list(db) + list(dk)))
    delta = np.empty(BOOT)
    for i in range(BOOT):
        ks = [allk[j] for j in RNG.integers(0, len(allk), len(allk))]
        a = np.concatenate([db[k] for k in ks if k in db]) if any(k in db for k in ks) else np.array([0.0])
        c = np.concatenate([dk[k] for k in ks if k in dk]) if any(k in dk for k in ks) else np.array([0.0])
        delta[i] = c.mean() - a.mean()
    obs = keep["pnl"].mean() - sleeve["pnl"].mean()
    print(
        f"  delta from excluding DTE 11-27: {obs:+.1f}"
        f"  90% CI [{np.percentile(delta,5):+.1f}, {np.percentile(delta,95):+.1f}]"
        f"  p(no gain) {float((delta<=0).mean()):.3f}"
    )
    print("  VERDICT: only act if the CI clearly excludes zero. A U-shaped cut on ~60")
    print("  trades is the same overfit that got the IV/HV 1.30 gate rejected.")

    # ------------------------------------------------------------------
    # The `primary` lane in apply_high_conviction_decision_marks skips the regime
    # map too. On 2026-07-23 (a downtrend) it emitted SOXX and CRWD Bear Calls,
    # which the map blocks. The sleeve survived this test; the primary lane has
    # to be checked separately rather than assumed.
    # ------------------------------------------------------------------
    print("\n=== does the PRIMARY lane also survive ignoring the regime map? ===")
    # NOTE: the pipeline's `expected_move_ratio` cannot be reconstructed exactly
    # from the history (rebuilding it as |spot-strike| / (spot*iv*sqrt(dte/365))
    # intersects the 25-30% credit band to n=0, because the two are correlated
    # -0.734). So the primary lane is approximated by everything EXCEPT the
    # expected-move term. That is a superset, which is the safe direction: if the
    # superset is fine, the narrower real lane is fine.
    primary = hist[
        hist["credit_pct"].between(MIN_CREDIT_PCT_WIDTH, MAX_CREDIT_PCT_WIDTH)
        & (hist["align"] >= 0.10)
        & (hist["ratio_ivhv"] >= 0.90)
        & (hist["rv"] >= 0.15)
    ].copy()
    rep("primary-ish conditions, all regimes", primary)
    rep("  map-ALLOWED", primary[primary["map_ok"]])
    rep("  map-BLOCKED", primary[~primary["map_ok"]])
    pb = primary[~primary["map_ok"]]
    if len(pb) >= 20:
        d = _days(pb)
        k = list(d)
        draws = np.array(
            [
                np.concatenate([d[k[j]] for j in RNG.integers(0, len(k), len(k))]).mean()
                for _ in range(BOOT)
            ]
        )
        print(
            f"  map-BLOCKED primary, day-clustered mean {pb['pnl'].mean():+.1f}"
            f"  90% CI [{np.percentile(draws,5):+.1f}, {np.percentile(draws,95):+.1f}]"
            f"  p(mean<=0) {float((draws<=0).mean()):.3f}"
        )
        print("  If this is clearly negative, the regime map DOES belong on the primary lane.")
    for lo_d, hi_d in [(0, 10), (11, 27), (28, 45)]:
        rep(f"  primary map-BLOCKED DTE {lo_d}-{hi_d}", pb[pb["dte_n"].between(lo_d, hi_d)])




if __name__ == "__main__":
    main()
