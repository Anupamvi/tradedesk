"""Does _decision_sort_score actually pick the right trade?

The pipeline keeps ONE trade per session (the highest `_decision_sort_score`).
So the honest test of a ranking key is NOT "does this feature correlate with
P&L across all trades" -- it is:

    on days where the map leaves >= 2 candidates, does ranking by this key and
    taking the top one beat taking a random one?

Anything that cannot beat a random pick on that test is decoration.

The current score is dominated by two terms worth up to 2.0 each:
    ratio  = distance to short strike / expected move   (further = higher score)
    align  = combined_flow_bias * direction_sign        (x6.0, so saturates at 0.333)
plus credit%-above-25 (max 1.0) and a quote-width penalty.

Two reasons to be suspicious of both big terms:
  * flow alignment has no predictive content anywhere else in this codebase
    (0 of 66 features survive BH at 5d; align gate delta +1.4, p 0.445)
  * the expected-move sweep is MONOTONE THE WRONG WAY: closest-strike quartile
    PF 1.12 -> furthest PF 0.82. The score pays up to 2.0 for being further out.

Day-clustered bootstrap throughout: same-session trades share regime and vol
shocks, so the resampling unit is the session, not the trade.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from codexuw.credit_policy import (  # noqa: E402
    ALLOWED_REGIMES,
    MAX_CREDIT_PCT_WIDTH,
    MAX_DTE,
    MAX_QUOTE_WIDTH_PCT,
    MIN_CREDIT_PCT_WIDTH,
    MIN_DTE,
    MIN_FLOW_ALIGNMENT,
)

HIST = "codexuw/history/codexdaily_v4_edge_history_v4_2026-07-26.csv.gz"
PANEL = "/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz"
BOOT = 3000
RNG = np.random.default_rng(20260726)


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[col], errors="coerce")


def load() -> pd.DataFrame:
    hist = pd.read_csv(HIST, low_memory=False)
    hist = hist[
        hist["evaluated"].astype(str).str.lower().eq("true")
        & hist["strategy_kind"].eq("Credit")
    ].copy()
    hist["pnl"] = _num(hist, "pnl_1x")
    hist = hist[hist["pnl"].notna()].copy()

    # direction label used by ALLOWED_REGIMES
    hist["side"] = np.where(
        hist["direction"].astype(str).str.contains("Bull", case=False), "Bull Put", "Bear Call"
    )
    allowed = hist.apply(
        lambda r: str(r["regime"]) in ALLOWED_REGIMES.get(r["side"], set()), axis=1
    )
    hist["map_allowed"] = allowed

    # realised vol from the price panel, to apply the shipping RV floor
    panel = pd.read_csv(PANEL, low_memory=False)
    panel = panel[["asof", "ticker", "rv21_ann"]].drop_duplicates(["asof", "ticker"])
    hist = hist.merge(panel, on=["asof", "ticker"], how="left")
    hist["rv"] = _num(hist, "rv21_ann")

    # rebuild the ranking inputs from the values the pipeline actually saw
    spot = _num(hist, "stock_price_eod")
    strike = _num(hist, "short_strike_eod")
    iv = _num(hist, "iv30d")
    dte = _num(hist, "dte")
    em = spot * iv * np.sqrt(dte.clip(lower=1) / 365.0)
    hist["em_ratio"] = ((spot - strike).abs() / em).replace([np.inf, -np.inf], np.nan)

    sign = np.where(hist["side"].eq("Bull Put"), 1.0, -1.0)
    hist["align"] = _num(hist, "combined_flow_bias") * sign
    hist["credit_pct"] = _num(hist, "entry_credit_pct_width")
    hist["quote_width"] = _num(hist, "entry_quote_width_pct")
    return hist


def sort_score(frame: pd.DataFrame, *, use_align: bool = True, use_em: bool = True) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    if use_em:
        score += frame["em_ratio"].clip(lower=0.0, upper=2.0).fillna(0.0)
    if use_align:
        score += (frame["align"].clip(lower=0.0) * 6.0).clip(upper=2.0).fillna(0.0)
    score += ((frame["credit_pct"] - MIN_CREDIT_PCT_WIDTH) * 8.0).clip(lower=0.0, upper=1.0).fillna(0.0)
    score -= (frame["quote_width"] - 0.35).clip(lower=0.0).fillna(0.0)
    return score


def day_boot(day_vals: dict[str, np.ndarray], stat=np.mean) -> tuple[float, float, float]:
    keys = list(day_vals)
    obs = stat(np.concatenate([day_vals[k] for k in keys]))
    draws = np.empty(BOOT)
    n = len(keys)
    for i in range(BOOT):
        pick = RNG.integers(0, n, n)
        draws[i] = stat(np.concatenate([day_vals[keys[j]] for j in pick]))
    return obs, float(np.percentile(draws, 5)), float(np.percentile(draws, 95))


def select_top1(pool: pd.DataFrame, key: pd.Series) -> dict[str, np.ndarray]:
    """Top-1 per session under `key`; returns {asof: array([pnl])}."""
    tmp = pool.assign(_k=key)
    out: dict[str, np.ndarray] = {}
    for asof, grp in tmp.groupby("asof"):
        best = grp.sort_values("_k", ascending=False).iloc[0]
        out[str(asof)] = np.array([float(best["pnl"])])
    return out


def random_top1(pool: pd.DataFrame, draws: int = 400) -> tuple[float, float, float]:
    """Mean P&L of picking one candidate uniformly at random per session."""
    groups = [g["pnl"].to_numpy(dtype=float) for _, g in pool.groupby("asof")]
    means = np.empty(draws)
    for i in range(draws):
        means[i] = np.mean([g[RNG.integers(0, len(g))] for g in groups])
    return float(means.mean()), float(np.percentile(means, 5)), float(np.percentile(means, 95))


def main() -> None:
    hist = load()
    pool = hist[hist["map_allowed"] & (hist["rv"] >= 0.15)].copy()
    pool = pool[pool["em_ratio"].notna() & pool["align"].notna()]
    print(f"map-allowed credit trades with usable ranking inputs: {len(pool):,}  days {pool['asof'].nunique()}")

    multi = pool.groupby("asof").filter(lambda g: len(g) >= 2)
    print(f"sessions offering a real CHOICE (>=2 candidates): {multi['asof'].nunique()}  trades {len(multi):,}")
    print(f"mean candidates on those days: {len(multi) / max(1, multi['asof'].nunique()):.1f}")

    print("\n=== 1. per-session top-1 selection: does the score beat a coin flip? ===")
    variants = {
        "current score (em + align + credit)": sort_score(multi),
        "drop align term": sort_score(multi, use_align=False),
        "drop em term": sort_score(multi, use_em=False),
        "credit% only": sort_score(multi, use_align=False, use_em=False),
        "align alone": multi["align"].fillna(-9),
        "em_ratio alone": multi["em_ratio"].fillna(-9),
        "em_ratio alone, INVERTED (closer first)": -multi["em_ratio"].fillna(9),
        "credit_pct alone": multi["credit_pct"].fillna(-9),
    }
    rnd_mean, rnd_lo, rnd_hi = random_top1(multi)
    for name, key in variants.items():
        sel = select_top1(multi, key)
        obs, lo, hi = day_boot(sel)
        flag = "  <-- beats random" if obs > rnd_hi else ("  <-- WORSE than random" if obs < rnd_lo else "")
        print(f"  {name:<40s} avg {obs:+8.1f}  90% CI [{lo:+8.1f},{hi:+8.1f}]{flag}")
    print(f"  {'RANDOM pick (400 draws)':<40s} avg {rnd_mean:+8.1f}  90% CI [{rnd_lo:+8.1f},{rnd_hi:+8.1f}]")

    print("\n=== 2. within-session rank correlation of each key with P&L ===")
    print("  (Spearman computed inside each session, then averaged over sessions)")
    for key_name in ["align", "em_ratio", "credit_pct"]:
        rhos = []
        for _, grp in multi.groupby("asof"):
            if grp[key_name].nunique() < 2 or grp["pnl"].nunique() < 2:
                continue
            rhos.append(grp[key_name].corr(grp["pnl"], method="spearman"))
        rhos = [r for r in rhos if math.isfinite(r)]
        arr = np.array(rhos)
        boot = np.array([RNG.choice(arr, len(arr), replace=True).mean() for _ in range(BOOT)])
        print(
            f"  {key_name:<12s} sessions {len(arr):3d}  mean rho {arr.mean():+.4f}"
            f"  90% CI [{np.percentile(boot,5):+.4f}, {np.percentile(boot,95):+.4f}]"
            f"  p(rho<=0) {float((boot<=0).mean()):.3f}"
        )

    print("\n=== 3. the em_ratio term is paid for being FAR. is far actually better? ===")
    q = pd.qcut(pool["em_ratio"], 4, labels=["Q1 closest", "Q2", "Q3", "Q4 furthest"])
    for label, grp in pool.groupby(q, observed=True):
        wins = grp.loc[grp["pnl"] > 0, "pnl"].sum()
        loss = -grp.loc[grp["pnl"] <= 0, "pnl"].sum()
        pf = wins / loss if loss > 0 else float("inf")
        print(
            f"  {str(label):<12s} n {len(grp):4d}  medRatio {grp['em_ratio'].median():.2f}"
            f"  win {100*(grp['pnl']>0).mean():5.1f}%  avg {grp['pnl'].mean():+7.1f}  PF {pf:5.2f}"
        )

    print("\n=== 4. align term saturates at 0.333. how many rows are pinned at the cap? ===")
    capped = (pool["align"] >= 1.0 / 3.0).mean()
    zeroed = (pool["align"] <= 0).mean()
    print(f"  align >= 0.333 (term maxed at 2.0): {100*capped:.1f}%")
    print(f"  align <= 0      (term contributes 0): {100*zeroed:.1f}%")
    print("  -> if most rows are at one extreme the term is a near-binary tiebreak, not a ranking")

    # ------------------------------------------------------------------
    # The tests above rank the whole map-allowed pool, where credit_pct varies
    # widely. The pipeline never sees that pool: the credit band (25-30% of
    # width), DTE 28-45, quote width and the flow floor all bind FIRST, and only
    # the survivors get ranked. credit_pct ranking beautifully across 0.10-0.50
    # is worth nothing if the gate has already squeezed it into 0.25-0.30.
    # This is the only section that can justify changing shipped code.
    # ------------------------------------------------------------------
    print("\n=== 5. THE DECISION-RELEVANT TEST: rank only what survives the shipped gates ===")
    dte = _num(pool, "dte")
    gated = pool[
        pool["credit_pct"].between(MIN_CREDIT_PCT_WIDTH, MAX_CREDIT_PCT_WIDTH)
        & dte.between(MIN_DTE, MAX_DTE)
        & (pool["quote_width"] <= MAX_QUOTE_WIDTH_PCT)
        & (pool["align"] >= MIN_FLOW_ALIGNMENT)
    ].copy()
    print(
        f"  survivors of the full shipped gate set: n {len(gated)}  days {gated['asof'].nunique()}"
        f"  win {100*(gated['pnl']>0).mean():.1f}%  avg {gated['pnl'].mean():+.1f}"
    )
    gwin = gated.loc[gated["pnl"] > 0, "pnl"].sum()
    gloss = -gated.loc[gated["pnl"] <= 0, "pnl"].sum()
    gpf = gwin / gloss if gloss > 0 else float("inf")
    gday = {str(a): g["pnl"].to_numpy(dtype=float) for a, g in gated.groupby("asof")}
    gobs, glo, ghi = day_boot(gday)
    total_days = hist["asof"].nunique()
    print(
        f"  PF {gpf:.2f}  total {gated['pnl'].sum():+,.0f}"
        f"  day-clustered avg {gobs:+.1f} 90% CI [{glo:+.1f}, {ghi:+.1f}]"
    )
    print(
        f"  CADENCE: {gated['asof'].nunique()} of {total_days} sessions produce a trade"
        f" ({100*gated['asof'].nunique()/total_days:.0f}%) -- roughly one entry every"
        f" {total_days/max(1,gated['asof'].nunique()):.1f} sessions."
    )
    print("  This config is designed to trade RARELY. A day with no trade is the normal case.")
    gmulti = gated.groupby("asof").filter(lambda g: len(g) >= 2)
    ndays = gmulti["asof"].nunique()
    print(f"  sessions with an actual choice to make: {ndays}  trades {len(gmulti)}")
    if ndays < 10:
        print("  -> too few sessions offer a choice to justify ANY ranking change. Leave it alone.")
        return
    print(f"  credit_pct spread inside the band: {gated['credit_pct'].min():.3f} - {gated['credit_pct'].max():.3f}")
    for key_name in ["align", "em_ratio", "credit_pct"]:
        rhos = []
        for _, grp in gmulti.groupby("asof"):
            if grp[key_name].nunique() < 2 or grp["pnl"].nunique() < 2:
                continue
            rhos.append(grp[key_name].corr(grp["pnl"], method="spearman"))
        rhos = [r for r in rhos if math.isfinite(r)]
        if len(rhos) < 5:
            print(f"  {key_name:<12s} too few usable sessions ({len(rhos)})")
            continue
        arr = np.array(rhos)
        boot = np.array([RNG.choice(arr, len(arr), replace=True).mean() for _ in range(BOOT)])
        print(
            f"  {key_name:<12s} sessions {len(arr):3d}  mean rho {arr.mean():+.4f}"
            f"  90% CI [{np.percentile(boot,5):+.4f}, {np.percentile(boot,95):+.4f}]"
            f"  p(rho<=0) {float((boot<=0).mean()):.3f}"
        )
    print("  --- top-1 selection inside the gated pool ---")
    rnd = random_top1(gmulti)
    for name, key in {
        "current score": sort_score(gmulti),
        "drop align term": sort_score(gmulti, use_align=False),
        "drop em term": sort_score(gmulti, use_em=False),
        "credit_pct alone": gmulti["credit_pct"].fillna(-9),
    }.items():
        obs, lo, hi = day_boot(select_top1(gmulti, key))
        print(f"    {name:<22s} avg {obs:+8.1f}  90% CI [{lo:+8.1f},{hi:+8.1f}]")
    print(f"    {'RANDOM pick':<22s} avg {rnd[0]:+8.1f}  90% CI [{rnd[1]:+8.1f},{rnd[2]:+8.1f}]")



if __name__ == "__main__":
    main()
