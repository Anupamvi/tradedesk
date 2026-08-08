"""Scan every flow feature for the sign of the forward variance risk premium.

The pipeline has only ever asked "where does realized exceed implied" (buy vol).
This asks the two-sided question: which feature deciles are systematically RICH
(implied over realized -> sell premium) and which are CHEAP (buy premium).

ratio = E|forward move| / (iv30d * sqrt(h/252) * sqrt(2/pi))
  ratio < 1  implied too high  -> short vol
  ratio > 1  implied too low   -> long vol
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import pandas as pd

VOL_ETPS = {
    "UVXY", "VXX", "SVIX", "SVXY", "VIXY", "UVIX", "TVIX", "VIXM", "VXZ",
    "SQQQ", "TQQQ", "SOXL", "SOXS", "SPXU", "UPRO", "TNA", "TZA", "LABU", "LABD",
}

FLOW_FEATURES = [
    "iv_rank", "vrp_ratio", "iv_chg_1w", "iv_chg_1m", "pos_52w",
    "stock_vol_surge", "call_vol_surge", "put_vol_surge", "put_call_ratio",
    "prem_tilt", "net_prem_tilt", "call_oi_chg", "put_oi_chg",
    "hc_multileg_share", "hc_sweep_share", "hc_opening_share", "hc_quote_churn",
    "hc_premium", "hc_chains", "hc_dir_bias",
    "oi_built_contracts", "oi_built_premium", "oi_signed_premium", "oi_n_chains",
    "oi_median_dte", "oi_nearmoney_premium", "oi_dir_bias", "oi_open_conviction",
    "oi_nearmoney_share",
    "dp_premium", "dp_block_premium", "dp_prints", "dp_bias", "dp_block_bias",
    "dp_block_share",
    "tape_net_premium", "tape_delta_notional", "tape_vega_flow", "tape_gamma_flow",
    "tape_gross_premium", "tape_prem_bias",
]

E_ABS_NORMAL = math.sqrt(2.0 / math.pi)


def load_panel(path: str, min_mcap: float) -> pd.DataFrame:
    cols = ["date", "ticker", "issue_type", "marketcap", "close", "iv30d"] + FLOW_FEATURES
    df = pd.read_csv(path, usecols=lambda c: c in set(cols), low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["ticker"].notna()]
    df = df[~df["ticker"].astype(str).str.upper().isin(VOL_ETPS)]
    if "issue_type" in df:
        df = df[df["issue_type"].astype(str).str.contains("Common", case=False, na=False)]
    df = df[pd.to_numeric(df["marketcap"], errors="coerce").fillna(0) >= min_mcap]
    df = df[pd.to_numeric(df["iv30d"], errors="coerce") > 0.01]
    df = df[pd.to_numeric(df["close"], errors="coerce") > 1.0]
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def add_forward_moves(df: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Entry is the NEXT session close (EOD data is not actionable at the same close)."""
    g = df.groupby("ticker", sort=False)["close"]
    entry = g.shift(-1)
    df["entry_close"] = entry
    for h in horizons:
        exit_close = g.shift(-(1 + h))
        df[f"abs_move_{h}"] = (exit_close / entry - 1.0).abs()
        implied = pd.to_numeric(df["iv30d"], errors="coerce") * math.sqrt(h / 252.0)
        df[f"implied_{h}"] = implied
        df[f"ratio_{h}"] = df[f"abs_move_{h}"] / (implied * E_ABS_NORMAL)
    return df


def day_clustered_ci(frame: pd.DataFrame, col: str, n_boot: int = 400, seed: int = 7):
    """Bootstrap the mean of `col` resampling whole trading days."""
    rng = np.random.default_rng(seed)
    days = frame["date"].unique()
    by_day = {d: frame.loc[frame["date"] == d, col].to_numpy() for d in days}
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(len(days), size=len(days), replace=True)
        vals = np.concatenate([by_day[days[i]] for i in pick])
        stats.append(np.nanmean(vals))
    return float(np.nanpercentile(stats, 5)), float(np.nanpercentile(stats, 95))


def scan(df: pd.DataFrame, horizon: int, n_buckets: int, min_rows: int) -> pd.DataFrame:
    ratio_col = f"ratio_{horizon}"
    base = df[df[ratio_col].notna() & np.isfinite(df[ratio_col])].copy()
    # Winsorize the ratio; a single 40x print otherwise decides a decile.
    hi = base[ratio_col].quantile(0.995)
    base[ratio_col] = base[ratio_col].clip(upper=hi)
    baseline = base[ratio_col].mean()

    rows = []
    for feat in FLOW_FEATURES:
        if feat not in base.columns:
            continue
        sub = base[base[feat].notna() & np.isfinite(base[feat])]
        if len(sub) < min_rows * n_buckets:
            continue
        # Rank WITHIN each day: a cross-sectional bucket, never a level threshold.
        rank = sub.groupby("date")[feat].rank(pct=True, method="average")
        bucket = np.ceil(rank * n_buckets).clip(1, n_buckets).astype(int)
        sub = sub.assign(_b=bucket)
        agg = sub.groupby("_b")[ratio_col].agg(["mean", "median", "count"])
        if agg["count"].min() < min_rows:
            continue
        top, bot = agg.loc[n_buckets], agg.loc[1]
        rows.append(
            {
                "feature": feat,
                "n": int(agg["count"].sum()),
                "bot_ratio": bot["mean"],
                "top_ratio": top["mean"],
                "spread": top["mean"] - bot["mean"],
                "richest_bucket": int(agg["mean"].idxmin()),
                "richest_ratio": agg["mean"].min(),
                "cheapest_bucket": int(agg["mean"].idxmax()),
                "cheapest_ratio": agg["mean"].max(),
                "monotone": _monotone_score(agg["mean"].to_numpy()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["baseline_ratio"] = baseline
    return out.sort_values("richest_ratio").reset_index(drop=True)


def _monotone_score(v: np.ndarray) -> float:
    """Spearman of bucket index vs value; +/-1 = perfectly ordered."""
    idx = np.arange(len(v))
    if np.std(v) == 0:
        return 0.0
    return float(np.corrcoef(idx, np.argsort(np.argsort(v)))[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
    ap.add_argument("--min-mcap", type=float, default=2e9)
    ap.add_argument("--horizons", default="5,10,21")
    ap.add_argument("--buckets", type=int, default=5)
    ap.add_argument("--min-rows", type=int, default=400)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    horizons = tuple(int(x) for x in args.horizons.split(","))
    df = load_panel(args.panel, args.min_mcap)
    df = add_forward_moves(df, horizons)
    print(f"panel rows={len(df):,}  dates={df['date'].nunique()}  tickers={df['ticker'].nunique()}")

    for h in horizons:
        res = scan(df, h, args.buckets, args.min_rows)
        if res.empty:
            print(f"\n=== h={h}: no feature had enough rows ===")
            continue
        base = res["baseline_ratio"].iloc[0]
        print(f"\n=== HORIZON {h}d | universe baseline realized/implied = {base:.4f} ===")
        print("  ratio < 1 => implied too rich => SELL premium | ratio > 1 => BUY premium")
        print(f"\n  {'feature':<24}{'q1':>8}{'q5':>8}{'spread':>9}{'richest':>9}{'@q':>4}{'cheapest':>10}{'@q':>4}{'mono':>7}{'n':>9}")
        for _, r in res.head(args.top).iterrows():
            print(
                f"  {r.feature:<24}{r.bot_ratio:>8.3f}{r.top_ratio:>8.3f}{r.spread:>9.3f}"
                f"{r.richest_ratio:>9.3f}{r.richest_bucket:>4}{r.cheapest_ratio:>10.3f}"
                f"{r.cheapest_bucket:>4}{r.monotone:>7.2f}{int(r.n):>9,}"
            )
        print("  ...")
        for _, r in res.tail(4).iterrows():
            print(
                f"  {r.feature:<24}{r.bot_ratio:>8.3f}{r.top_ratio:>8.3f}{r.spread:>9.3f}"
                f"{r.richest_ratio:>9.3f}{r.richest_bucket:>4}{r.cheapest_ratio:>10.3f}"
                f"{r.cheapest_bucket:>4}{r.monotone:>7.2f}{int(r.n):>9,}"
            )


if __name__ == "__main__":
    main()
