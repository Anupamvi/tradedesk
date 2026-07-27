"""Honest cross-sectional signal validation.

Why this file exists
--------------------
Previous feature work in this repo failed for two reasons, both methodological:

1. Signals were measured on ~164 guard-passing option rows and pooled as if the
   rows were independent. They are not: rows from the same session share the same
   market shock, so pooled t-stats were inflated by roughly sqrt(rows-per-day).
2. Parametric models were fit per fold on tiny samples, so every added feature
   lowered out-of-sample profit factor (1.40 -> 1.25 -> 1.19 -> 1.10).

The fix is the standard cross-sectional quant construction:

* Compute a **daily cross-sectional Spearman rank correlation** (the Information
  Coefficient) between the signal and the forward return. Each *day* is one
  observation, so cross-sectional correlation cannot inflate significance.
* Overlapping forward returns sampled daily are heavily autocorrelated, so the
  t-stat on the mean IC uses a **Newey-West** correction with lag = horizon.
* Report **decile long/short spreads** in volatility-normalised units, which is
  what a tradable structure would actually harvest.
* Report the **number of features tested** so the reader can discount for
  multiple comparisons, plus a Benjamini-Hochberg adjustment.

A feature is only interesting if it survives all of that. Nothing here is wired
into the pipeline; this is a measurement instrument.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESEARCH = Path("/Users/anuppamvi/uw_root/tradedesk/out/research")

# Columns that are targets/identifiers, never candidate signals.
# The forward-looking block is a hard safety guard: every one of these is
# computed from data after `asof`, so leaking any into the feature scan would
# manufacture a spectacular fake signal.
NON_FEATURES = {
    "asof", "ticker", "sector", "full_name", "issue_type", "is_index",
    "next_earnings_date", "er_time", "date", "close", "prev_close", "high", "low",
    "week_52_high", "week_52_low",
    # forward-looking -- NEVER features
    "rv_fwd_21d", "vrp_realized", "vrp_realized_ratio", "vrp_capture",
    "abs_fwd_21d", "move_vs_implied", "stayed_inside",
}


def _rank_z(s: pd.Series) -> pd.Series:
    r = s.rank(pct=True)
    sd = r.std()
    return (r - r.mean()) / (sd if sd else np.nan)


def neutralize(sub: pd.DataFrame, cols: list, controls: list) -> pd.DataFrame:
    """Cross-sectionally residualise `cols` against `controls` within one day.

    Without this, any variable correlated with the IV level posts a huge t-stat
    against a variance-premium target simply because `iv30d` appears on both
    sides of the target definition. Residualising asks the only question that
    matters: does this feature add anything *beyond* knowing the IV level?
    """
    ctl = [c for c in controls if c in sub.columns and sub[c].notna().sum() > 20]
    if not ctl:
        return sub
    X = np.column_stack(
        [_rank_z(sub[c]).fillna(0.0).to_numpy() for c in ctl] + [np.ones(len(sub))]
    )
    out = sub.copy()
    for c in cols:
        y = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(y)
        if ok.sum() < 30:
            out[c] = np.nan
            continue
        beta, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
        resid = np.full(len(sub), np.nan)
        resid[ok] = y[ok] - X[ok] @ beta
        out[c] = resid
    return out


def newey_west_t(x: pd.Series, lag: int) -> float:
    """t-stat of the mean of an autocorrelated series."""
    x = x.dropna()
    n = len(x)
    if n < 20:
        return np.nan
    xd = x - x.mean()
    gamma0 = float((xd**2).sum() / n)
    var = gamma0
    for l in range(1, min(lag, n - 1) + 1):
        cov = float((xd.iloc[l:].to_numpy() * xd.iloc[:-l].to_numpy()).sum() / n)
        var += 2.0 * (1.0 - l / (lag + 1.0)) * cov
    if var <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(var / n))


def daily_ic(df: pd.DataFrame, feat: str, target: str, min_names: int = 50, controls=None) -> pd.Series:
    out = {}
    for asof, grp in df.groupby("asof", observed=True):
        cols = [feat, target] + list(controls or [])
        sub = grp[[c for c in dict.fromkeys(cols) if c in grp.columns]].dropna(subset=[feat, target])
        if len(sub) < min_names or sub[feat].nunique() < 10:
            continue
        if controls:
            sub = neutralize(sub, [feat, target], list(controls))
            sub = sub.dropna(subset=[feat, target])
            if len(sub) < min_names:
                continue
        rho = stats.spearmanr(sub[feat], sub[target]).correlation
        if np.isfinite(rho):
            out[asof] = rho
    return pd.Series(out).sort_index()


def daily_decile_spread(df: pd.DataFrame, feat: str, target: str, q: int = 5, min_names: int = 50, controls=None) -> pd.Series:
    out = {}
    for asof, grp in df.groupby("asof", observed=True):
        cols = [feat, target] + list(controls or [])
        sub = grp[[c for c in dict.fromkeys(cols) if c in grp.columns]].dropna(subset=[feat, target])
        if len(sub) < min_names or sub[feat].nunique() < q * 2:
            continue
        if controls:
            sub = neutralize(sub, [feat, target], list(controls))
            sub = sub.dropna(subset=[feat, target])
            if len(sub) < min_names:
                continue
        try:
            bins = pd.qcut(sub[feat].rank(method="first"), q, labels=False)
        except ValueError:
            continue
        # winsorise the target within the day: vol-normalised returns have fat
        # tails (small adm21 in the denominator), and a single name should not
        # decide the bucket mean
        y = sub[target].clip(sub[target].quantile(0.01), sub[target].quantile(0.99))
        top = y[bins == q - 1].mean()
        bot = y[bins == 0].mean()
        if np.isfinite(top) and np.isfinite(bot):
            out[asof] = top - bot
    return pd.Series(out).sort_index()


def profile(df: pd.DataFrame, feat: str, target: str, q: int, min_names: int, lag: int) -> None:
    """Quintile shape + sub-period stability for a single feature."""
    rows = []
    for asof, grp in df.groupby("asof", observed=True):
        sub = grp[[feat, target]].dropna()
        if len(sub) < min_names or sub[feat].nunique() < q * 2:
            continue
        bins = pd.qcut(sub[feat].rank(method="first"), q, labels=False)
        y = sub[target].clip(sub[target].quantile(0.01), sub[target].quantile(0.99))
        for b in range(q):
            rows.append({"asof": asof, "bucket": b + 1, "ret": y[bins == b].mean()})
    prof = pd.DataFrame(rows)
    if prof.empty:
        print(f"  {feat}: insufficient data")
        return

    print(f"\n--- profile: {feat} vs {target} ---")
    shape = prof.groupby("bucket")["ret"].mean()
    print("quintile mean forward return (vol-normalised units):")
    for b, v in shape.items():
        bar = "#" * int(abs(v) * 200)
        print(f"  Q{b}  {v:+.4f}  {bar}")

    ic = daily_ic(df, feat, target, min_names)
    print("\nstability -- mean IC by month:")
    months = pd.Series(ic.index, index=ic.index).str.slice(0, 7)
    for m, v in ic.groupby(months).agg(["mean", "size"]).iterrows():
        print(f"  {m}  IC {v['mean']:+.4f}  (n={int(v['size'])} days)")
    print(f"\n  full-sample mean IC {ic.mean():+.4f}  NW t {newey_west_t(ic, lag):.2f}  days {len(ic)}")


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def load_merged(min_dollar_vol: float, min_mcap: float) -> pd.DataFrame:
    price = pd.read_csv(RESEARCH / "price_panel.csv.gz", low_memory=False)
    price["ticker"] = price["ticker"].astype(str).str.upper()

    frames = [price]
    feat_path = RESEARCH / "uw_features.csv.gz"
    if feat_path.exists():
        uw = pd.read_csv(feat_path, low_memory=False)
        uw["ticker"] = uw["ticker"].astype(str).str.upper()
        dupe = [c for c in uw.columns if c in price.columns and c not in ("asof", "ticker")]
        uw = uw.drop(columns=dupe)
        frames.append(uw)

    flow_path = RESEARCH / "flow_panel.csv.gz"
    if flow_path.exists():
        fl = pd.read_csv(flow_path, low_memory=False)
        fl["ticker"] = fl["ticker"].astype(str).str.upper()
        dupe = [c for c in fl.columns if c in price.columns and c not in ("asof", "ticker")]
        fl = fl.drop(columns=dupe)
        frames.append(fl)

    df = frames[0]
    for extra in frames[1:]:
        df = df.merge(extra, on=["asof", "ticker"], how="left")

    # tradability filter -- no point discovering alpha in names we cannot fill
    keep = pd.Series(True, index=df.index)
    if "dollar_vol" in df.columns:
        keep &= df["dollar_vol"].fillna(0) >= min_dollar_vol
    if "marketcap" in df.columns:
        keep &= pd.to_numeric(df["marketcap"], errors="coerce").fillna(0) >= min_mcap
    df = df[keep].copy()
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="fwd_21d_z")
    ap.add_argument("--horizon-lag", type=int, default=21, help="Newey-West lag; set to the return horizon")
    ap.add_argument("--min-dollar-vol", type=float, default=25_000_000.0)
    ap.add_argument("--min-mcap", type=float, default=1e9)
    ap.add_argument("--min-names", type=int, default=50)
    ap.add_argument("--quantiles", type=int, default=5)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--profile", default="", help="profile a single feature instead of scanning all")
    ap.add_argument(
        "--controls",
        default="",
        help="comma-separated columns to cross-sectionally residualise out (e.g. iv30d,marketcap)",
    )
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    controls = [c.strip() for c in args.controls.split(",") if c.strip()]

    df = load_merged(args.min_dollar_vol, args.min_mcap)
    print(f"universe: {len(df):,} rows  days {df['asof'].nunique()}  tickers {df['ticker'].nunique()}")
    print(f"target: {args.target}  (Newey-West lag {args.horizon_lag})")
    if controls:
        print(f"neutralised against: {', '.join(controls)}")
    print()

    target = args.target
    if target not in df.columns:
        raise SystemExit(f"target {target} not in panel")

    if args.profile:
        for feat in args.profile.split(","):
            feat = feat.strip()
            if feat not in df.columns:
                print(f"  {feat}: not in panel")
                continue
            profile(df, feat, target, args.quantiles, args.min_names, args.horizon_lag)
        return

    feats = [
        c for c in df.columns
        if c not in NON_FEATURES
        and not c.startswith("fwd_")
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().mean() > 0.30
    ]
    print(f"testing {len(feats)} candidate features\n")

    rows = []
    for f in feats:
        if f in controls:
            continue
        ic = daily_ic(df, f, target, args.min_names, controls)
        if len(ic) < 30:
            continue
        t = newey_west_t(ic, args.horizon_lag)
        sp = daily_decile_spread(df, f, target, args.quantiles, args.min_names, controls)
        t_sp = newey_west_t(sp, args.horizon_lag)
        rows.append({
            "feature": f,
            "days": len(ic),
            "mean_ic": ic.mean(),
            "t_ic": t,
            "ic_hit": (ic > 0).mean(),
            "q_spread": sp.mean() if len(sp) else np.nan,
            "t_spread": t_sp,
            "coverage": df[f].notna().mean(),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("no features produced enough days")
    res["p_ic"] = 2 * (1 - stats.norm.cdf(res["t_ic"].abs().fillna(0)))
    res["p_bh"] = bh_adjust(res["p_ic"].to_numpy())
    res = res.sort_values("t_ic", key=lambda s: s.abs(), ascending=False)

    pd.set_option("display.width", 220)
    show = res.head(args.top).copy()
    for c in ("mean_ic", "q_spread"):
        show[c] = show[c].round(4)
    for c in ("t_ic", "t_spread"):
        show[c] = show[c].round(2)
    for c in ("ic_hit", "coverage"):
        show[c] = show[c].round(3)
    for c in ("p_ic", "p_bh"):
        show[c] = show[c].round(4)
    print(show.to_string(index=False))

    n_sig = int((res["p_bh"] < 0.10).sum())
    print(f"\nfeatures with BH-adjusted p < 0.10: {n_sig} / {len(res)}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
