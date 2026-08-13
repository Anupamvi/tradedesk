"""Native-resolution feature extraction from all five UW files.

The panel this repo has always used collapses ~12.5M rows/day into ~34 numbers
per ticker -- a 213x compression in which the 1.1GB options tape contributes four
sums. Everything downstream then concluded "the flow data adds nothing", which it
cannot honestly do when 99.5% of it was discarded before the model saw it.

This keeps the structure the sums destroy, for every feed:
  * WHO traded        size buckets, sweep/block/floor/cross, aggressor side
  * WHAT they traded  moneyness x DTE x call/put grid, not one net number
  * HOW they paid     price vs NBBO, IV paid vs chain IV
  * WHEN              open / midday / close concentration
  * HOW concentrated  top-contract and top-print share, not just the total
Multi-leg prints and cancellations are removed before anything is signed, because
a spread leg lifted at the ask is not a bullish customer.

Coverage is not uniform: the tape exists for ~64 of 140 days, the other four
feeds for ~139-141. Tape columns are emitted as a separate block so a model can
be trained with and without them.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

TAPE_COLS = [
    "executed_at", "underlying_symbol", "side", "strike", "option_type", "expiry",
    "underlying_price", "nbbo_bid", "nbbo_ask", "price", "size", "premium",
    "open_interest", "implied_volatility", "delta", "gamma", "vega", "theta",
    "canceled", "upstream_condition_detail", "report_flags",
]
MULTILEG_CODES = {"mlet", "mlat", "mlft", "mlct", "mash", "msrb"}

SIZE_EDGES = [0, 10_000, 50_000, 250_000, np.inf]
SIZE_NAMES = ["retail", "small", "mid", "block"]
MNY_EDGES = [-np.inf, -0.07, -0.02, 0.02, 0.07, np.inf]
MNY_NAMES = ["dotmp", "otmp", "atm", "otmc", "dotmc"]
DTE_EDGES = [-1, 7, 30, 90, np.inf]
DTE_NAMES = ["d7", "d30", "d90", "d90p"]

HC_COLS = ["option_symbol", "volume", "open_interest", "premium", "ask_side_volume",
           "bid_side_volume", "mid_volume", "sweep_volume", "floor_volume",
           "cross_volume", "multileg_volume", "total_bid_changes",
           "total_ask_changes", "iv", "bid", "ask"]
OI_COLS = ["option_symbol", "oi_diff_plain", "last_oi", "curr_oi", "last_bid",
           "last_ask", "prev_ask_volume", "prev_bid_volume", "prev_mid_volume",
           "prev_multi_leg_volume", "prev_total_premium", "dte", "stock_price", "iv"]
DP_COLS = ["ticker", "executed_at", "nbbo_bid", "nbbo_ask", "size", "premium",
           "price", "sale_cond_codes", "canceled"]
DP_NON_DIRECTIONAL = ("average_price_trade", "contingent_trade",
                      "odd_lot_execution", "prior_reference_price")
TRUEY = {"true", "1", "t", "yes"}


def _zip_csvs(base: Path, date: str, stem: str):
    for zp in sorted((base / date).glob(f"{stem}-{date}*.zip")):
        zf = zipfile.ZipFile(zp)
        for nm in zf.namelist():
            if nm.lower().endswith(".csv"):
                yield zf, nm


def _num(df: pd.DataFrame, cols) -> pd.DataFrame:
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _bucket_sums(df: pd.DataFrame, key: str, bucket, values: list[str],
                 prefix: str) -> pd.DataFrame:
    """groupby+unstack; materially faster than pivot_table on this shape."""
    g = df.assign(_b=bucket).groupby([key, "_b"], observed=True)[values].sum()
    out = g.unstack("_b")
    out.columns = [f"{prefix}{b}{v}" for v, b in out.columns]
    return out


def _occ_parse(sym: pd.Series):
    s = sym.astype(str).str.upper().str.strip()
    ex = s.str.extract(r"^([A-Z0-9\.\-]{1,6})(\d{6})([CP])(\d{8})$")
    strike = pd.to_numeric(ex[3], errors="coerce") / 1000.0
    return ex[0], ex[2].eq("C"), strike


# --------------------------------------------------------------------------- tape
def _clean_tape(ch: pd.DataFrame, date: str) -> pd.DataFrame:
    if "canceled" in ch:
        ch = ch[~ch["canceled"].astype(str).str.lower().isin(TRUEY)]
    if "upstream_condition_detail" in ch:
        ch = ch[~ch["upstream_condition_detail"].astype(str).str.lower().isin(MULTILEG_CODES)]
    ch = _num(ch, ["strike", "underlying_price", "nbbo_bid", "nbbo_ask", "price",
                   "size", "premium", "open_interest", "implied_volatility",
                   "delta", "gamma", "vega", "theta"])
    ch = ch[ch["premium"].notna() & (ch["premium"] > 0) & (ch["underlying_price"] > 0)]
    if ch.empty:
        return ch
    ch = ch.copy()
    sd = ch["side"].astype(str).str.lower()
    ch["sgn"] = np.where(sd.eq("ask"), 1.0, np.where(sd.eq("bid"), -1.0, 0.0))
    ch["is_call"] = ch["option_type"].astype(str).str.lower().str.startswith("c")
    mny = ch["strike"] / ch["underlying_price"] - 1.0
    # Signed so puts and calls share one axis: negative = downside strikes.
    ch["mny_signed"] = np.where(ch["is_call"], mny, -mny.abs())
    ch["dte"] = (pd.to_datetime(ch["expiry"], errors="coerce") - pd.Timestamp(date)).dt.days
    ts = pd.to_datetime(ch["executed_at"], errors="coerce", utc=True)
    ch["hour"] = ts.dt.hour + ts.dt.minute / 60.0
    mid = 0.5 * (ch["nbbo_bid"] + ch["nbbo_ask"])
    spr = (ch["nbbo_ask"] - ch["nbbo_bid"]).replace(0, np.nan)
    ch["aggression"] = ((ch["price"] - mid) / (0.5 * spr)).clip(-2, 2)
    flags = ch.get("report_flags", pd.Series("", index=ch.index)).astype(str).str.lower()
    ch["is_sweep"] = flags.str.contains("sweep", na=False)
    ch["is_floor"] = flags.str.contains("floor", na=False)
    ch["is_cross"] = flags.str.contains("cross", na=False)
    ch["opening"] = (ch["size"] > ch["open_interest"].fillna(0)).astype(float)
    return ch


def _agg_tape(ch: pd.DataFrame) -> pd.DataFrame:
    prem = ch["premium"]
    ch["xsigned"] = ch["sgn"] * prem
    ch["xsweep_p"] = prem.where(ch["is_sweep"], 0.0)
    ch["xfloor_p"] = prem.where(ch["is_floor"], 0.0)
    ch["xcross_p"] = prem.where(ch["is_cross"], 0.0)
    ch["xopen_p"] = prem.where(ch["hour"] < 15.0, 0.0)
    ch["xlate_p"] = prem.where(ch["hour"] >= 19.5, 0.0)
    ch["xopenint_p"] = ch["opening"] * prem
    ch["xaggr_w"] = ch["aggression"].fillna(0) * prem
    ch["xiv_w"] = ch["implied_volatility"].fillna(0) * prem
    ch["xsweep_signed"] = ch["xsigned"].where(ch["is_sweep"], 0.0)
    for col, nm in (("delta", "dlt"), ("gamma", "gmm"), ("vega", "vga"), ("theta", "tht")):
        ch[f"x{nm}_flow"] = ch["sgn"] * ch[col].fillna(0) * ch["size"] * 100.0

    key = "underlying_symbol"
    g = ch.groupby(key, sort=False)
    sums = [c for c in ch.columns if c.startswith("x")]
    out = g[sums].sum()
    out.columns = [f"tp_{c[1:]}" for c in out.columns]
    out["tp_prints"] = g.size()
    out["tp_premium"] = g["premium"].sum()
    out["tp_size"] = g["size"].sum()
    out["tp_max_print"] = g["premium"].max()

    vals = ["premium", "xsigned"]
    for bucket, prefix in (
        (pd.cut(prem, SIZE_EDGES, labels=SIZE_NAMES), "tp_sz_"),
        (pd.cut(ch["mny_signed"], MNY_EDGES, labels=MNY_NAMES), "tp_mny_"),
        (pd.cut(ch["dte"], DTE_EDGES, labels=DTE_NAMES), "tp_dte_"),
        (pd.Series(np.where(ch["is_call"], "call", "put"), index=ch.index), "tp_cp_"),
    ):
        out = out.join(_bucket_sums(ch, key, bucket, vals, prefix), how="left")
    return out


def tape_features(base: Path, date: str, chunksize: int = 2_000_000) -> pd.DataFrame:
    parts = list(_zip_csvs(base, date, "bot-eod-report"))
    if not parts:
        return pd.DataFrame()
    acc = []
    for zf, nm in parts:
        with zf.open(nm) as fh:
            for ch in pd.read_csv(fh, usecols=lambda c: c in set(TAPE_COLS),
                                  chunksize=chunksize, low_memory=False):
                ch = _clean_tape(ch, date)
                if not ch.empty:
                    acc.append(_agg_tape(ch))
    if not acc:
        return pd.DataFrame()
    t = pd.concat(acc).groupby(level=0).sum(min_count=1)
    prem = t["tp_premium"].replace(0, np.nan)

    for c in [c for c in t.columns if c.endswith("premium") and c != "tp_premium"]:
        t[c.replace("premium", "share")] = t[c] / prem
    for c in [c for c in t.columns if c.endswith("xsigned")]:
        t[c.replace("xsigned", "bias")] = t[c] / prem
    for nm in ("sweep", "floor", "cross", "open", "late", "openint"):
        if f"tp_{nm}_p" in t:
            t[f"tp_{nm}_share"] = t[f"tp_{nm}_p"] / prem
    t["tp_prem_bias"] = t["tp_signed"] / prem
    t["tp_sweep_bias"] = t["tp_sweep_signed"] / prem
    t["tp_aggression"] = t["tp_aggr_w"] / prem
    t["tp_iv_paid"] = t["tp_iv_w"] / prem
    t["tp_max_print_share"] = t["tp_max_print"] / prem
    t["tp_avg_print"] = prem / t["tp_prints"].replace(0, np.nan)
    t["tp_prem_per_size"] = prem / t["tp_size"].replace(0, np.nan)
    # Skew of demand: are they paying up for downside or upside wings?
    t["tp_wing_skew"] = (t.get("tp_mny_dotmcxsigned", 0.0)
                         - t.get("tp_mny_dotmpxsigned", 0.0)) / prem
    t["tp_call_put_ratio"] = (t.get("tp_cp_callpremium", np.nan)
                              / t.get("tp_cp_putpremium", pd.Series(np.nan, index=t.index)).replace(0, np.nan))
    t = t.drop(columns=[c for c in t.columns
                        if c.endswith(("_w", "xsigned", "_p", "tp_max_print"))], errors="ignore")
    t.index.name = "ticker"
    return t.reset_index().assign(date=pd.Timestamp(date))


# -------------------------------------------------------------------- hot chains
def hot_chain_features(base: Path, date: str) -> pd.DataFrame:
    frames = [pd.read_csv(zf.open(nm), usecols=lambda c: c in set(HC_COLS), low_memory=False)
              for zf, nm in _zip_csvs(base, date, "hot-chains")]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = _num(d, [c for c in HC_COLS if c != "option_symbol"])
    d["ticker"], d["is_call"], d["strike"] = _occ_parse(d["option_symbol"])
    d = d[d["ticker"].notna() & d["premium"].notna() & (d["premium"] > 0)].copy()
    if d.empty:
        return pd.DataFrame()
    # Strip spread legs BEFORE signing: measured monotonic signal degradation otherwise.
    ml = d["multileg_volume"].fillna(0)
    ask = (d["ask_side_volume"].fillna(0) - ml / 2).clip(lower=0)
    bid = (d["bid_side_volume"].fillna(0) - ml / 2).clip(lower=0)
    tot = (ask + bid).replace(0, np.nan)
    d["premium_"] = d["premium"]
    d["xsigned"] = ((ask - bid) / tot).fillna(0.0) * d["premium"]
    d["xml"] = ml
    d["xsweep"] = d["sweep_volume"].fillna(0)
    d["xfloor"] = d["floor_volume"].fillna(0)
    d["xcross"] = d["cross_volume"].fillna(0)
    d["xv2oi"] = (d["volume"] / d["open_interest"].replace(0, np.nan)).clip(0, 50).fillna(0) * d["premium"]
    d["xiv_w"] = d["iv"].fillna(0) * d["premium"]
    d["xchurn"] = d["total_bid_changes"].fillna(0) + d["total_ask_changes"].fillna(0)
    d["xspread_w"] = (((d["ask"] - d["bid"]) / (0.5 * (d["ask"] + d["bid"])).replace(0, np.nan))
                      .fillna(0) * d["premium"])

    g = d.groupby("ticker", sort=False)
    out = g[["premium_", "xsigned", "xml", "xsweep", "xfloor", "xcross", "xv2oi",
             "xiv_w", "xchurn", "xspread_w", "volume"]].sum()
    out.columns = [f"hc_{c.rstrip('_')}" for c in out.columns]
    out["hc_chains"] = g.size()
    out["hc_top_chain_share"] = g["premium"].max() / out["hc_premium"].replace(0, np.nan)
    out = out.join(_bucket_sums(d, "ticker",
                                pd.Series(np.where(d["is_call"], "call", "put"), index=d.index),
                                ["premium_", "xsigned"], "hc_cp_"), how="left")
    p = out["hc_premium"].replace(0, np.nan)
    v = out["hc_volume"].replace(0, np.nan)
    out["hc_dir_bias"] = out["hc_xsigned"] / p
    out["hc_multileg_share"] = out["hc_xml"] / v
    out["hc_sweep_share"] = out["hc_xsweep"] / v
    out["hc_floor_share"] = out["hc_xfloor"] / v
    out["hc_cross_share"] = out["hc_xcross"] / v
    out["hc_vol_to_oi"] = out["hc_xv2oi"] / p
    out["hc_iv"] = out["hc_xiv_w"] / p
    out["hc_spread_pct"] = out["hc_xspread_w"] / p
    out["hc_churn_per_chain"] = out["hc_xchurn"] / out["hc_chains"].replace(0, np.nan)
    out["hc_call_bias"] = out.get("hc_cp_callxsigned", 0.0) / p
    out["hc_put_bias"] = out.get("hc_cp_putxsigned", 0.0) / p
    out["hc_call_share"] = out.get("hc_cp_callpremium_", 0.0) / p
    out = out.drop(columns=[c for c in out.columns if c.startswith("hc_x")], errors="ignore")
    out.index.name = "ticker"
    return out.reset_index().assign(date=pd.Timestamp(date))


# ---------------------------------------------------------------------- chain OI
def chain_oi_features(base: Path, date: str) -> pd.DataFrame:
    frames = [pd.read_csv(zf.open(nm), usecols=lambda c: c in set(OI_COLS), low_memory=False)
              for zf, nm in _zip_csvs(base, date, "chain-oi-changes")]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = _num(d, [c for c in OI_COLS if c != "option_symbol"])
    d["ticker"], d["is_call"], d["strike"] = _occ_parse(d["option_symbol"])
    d = d[d["ticker"].notna() & d["stock_price"].notna() & (d["stock_price"] > 0)].copy()
    if d.empty:
        return pd.DataFrame()
    ml = d["prev_multi_leg_volume"].fillna(0)
    ask = (d["prev_ask_volume"].fillna(0) - ml / 2).clip(lower=0)
    bid = (d["prev_bid_volume"].fillna(0) - ml / 2).clip(lower=0)
    lead = ((ask - bid) / (ask + bid).replace(0, np.nan)).fillna(0.0)
    # Only a POSITIVE oi change is a build; abs() would count unwinds as opens.
    built = d["oi_diff_plain"].clip(lower=0)
    mid = 0.5 * (d["last_bid"].fillna(0) + d["last_ask"].fillna(0))
    d["built_prem"] = built * mid * 100.0
    d["xsigned"] = d["built_prem"] * lead
    d["built_ct"] = built
    d["unwound_ct"] = (-d["oi_diff_plain"]).clip(lower=0)
    mny = d["strike"] / d["stock_price"] - 1.0
    d["mny_signed"] = np.where(d["is_call"], mny, -mny.abs())
    d["xnear"] = d["built_prem"].where(mny.abs() <= 0.05, 0.0)
    d["xdte_w"] = d["dte"].fillna(0) * d["built_prem"]
    d["xiv_w"] = (d["iv"].fillna(0) if "iv" in d else 0.0) * d["built_prem"]

    g = d.groupby("ticker", sort=False)
    out = g[["built_prem", "xsigned", "built_ct", "unwound_ct", "xnear",
             "xdte_w", "xiv_w"]].sum()
    out.columns = [f"oi_{c}" for c in out.columns]
    out["oi_chains"] = g.size()
    out["oi_top_chain_share"] = g["built_prem"].max() / out["oi_built_prem"].replace(0, np.nan)
    for bucket, prefix in (
        (pd.cut(d["mny_signed"], MNY_EDGES, labels=MNY_NAMES), "oi_mny_"),
        (pd.cut(d["dte"], DTE_EDGES, labels=DTE_NAMES), "oi_dte_"),
    ):
        out = out.join(_bucket_sums(d, "ticker", bucket, ["built_prem", "xsigned"], prefix), how="left")
    p = out["oi_built_prem"].replace(0, np.nan)
    out["oi_dir_bias"] = out["oi_xsigned"] / p
    out["oi_near_share"] = out["oi_xnear"] / p
    out["oi_mean_dte"] = out["oi_xdte_w"] / p
    out["oi_iv"] = out["oi_xiv_w"] / p
    out["oi_build_unwind_ratio"] = out["oi_built_ct"] / out["oi_unwound_ct"].replace(0, np.nan)
    out["oi_wing_skew"] = (out.get("oi_mny_dotmcxsigned", 0.0)
                           - out.get("oi_mny_dotmpxsigned", 0.0)) / p
    out = out.drop(columns=["oi_xnear", "oi_xdte_w", "oi_xiv_w"], errors="ignore")
    out.index.name = "ticker"
    return out.reset_index().assign(date=pd.Timestamp(date))


# --------------------------------------------------------------------- dark pool
def dark_pool_features(base: Path, date: str) -> pd.DataFrame:
    frames = [pd.read_csv(zf.open(nm), usecols=lambda c: c in set(DP_COLS), low_memory=False)
              for zf, nm in _zip_csvs(base, date, "dp-eod-report")]
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True)
    d = _num(d, ["nbbo_bid", "nbbo_ask", "size", "premium", "price"])
    if "canceled" in d:
        d = d[~d["canceled"].astype(str).str.lower().isin(TRUEY)]
    d = d[d["premium"].notna() & (d["premium"] > 0)].copy()
    if d.empty:
        return pd.DataFrame()
    cond = d.get("sale_cond_codes", pd.Series("", index=d.index)).astype(str).str.lower()
    nondir = cond.str.contains("|".join(DP_NON_DIRECTIONAL), na=False)
    mid = 0.5 * (d["nbbo_bid"] + d["nbbo_ask"])
    spr = (d["nbbo_ask"] - d["nbbo_bid"]).replace(0, np.nan)
    loc = ((d["price"] - mid) / (0.5 * spr)).clip(-2, 2).fillna(0.0)
    ts = pd.to_datetime(d["executed_at"], errors="coerce", utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0

    d["prem"] = d["premium"]
    d["dir_prem"] = d["premium"].where(~nondir, 0.0)
    d["xloc_w"] = loc * d["dir_prem"]
    d["late_prem"] = d["dir_prem"].where(hour >= 19.5, 0.0)
    d["xlate_loc_w"] = loc * d["late_prem"]
    d["block_prem"] = d["dir_prem"].where(d["premium"] >= 1e6, 0.0)
    d["xblock_loc_w"] = loc * d["block_prem"]

    g = d.groupby("ticker", sort=False)
    out = g[["prem", "dir_prem", "xloc_w", "late_prem", "xlate_loc_w",
             "block_prem", "xblock_loc_w"]].sum()
    out.columns = [f"dp_{c}" for c in out.columns]
    out["dp_prints"] = g.size()
    out["dp_median_size"] = g["size"].median()
    out["dp_max_print_share"] = g["premium"].max() / out["dp_prem"].replace(0, np.nan)
    p = out["dp_dir_prem"].replace(0, np.nan)
    out["dp_bias"] = out["dp_xloc_w"] / p
    out["dp_late_bias"] = out["dp_xlate_loc_w"] / out["dp_late_prem"].replace(0, np.nan)
    out["dp_block_bias"] = out["dp_xblock_loc_w"] / out["dp_block_prem"].replace(0, np.nan)
    out["dp_block_share"] = out["dp_block_prem"] / p
    out["dp_late_share"] = out["dp_late_prem"] / p
    out["dp_nondir_share"] = 1.0 - (out["dp_dir_prem"] / out["dp_prem"].replace(0, np.nan))
    out = out.drop(columns=[c for c in out.columns if c.endswith("_loc_w")], errors="ignore")
    out.index.name = "ticker"
    return out.reset_index().assign(date=pd.Timestamp(date))


FEED_FN = {"tape": tape_features, "hot": hot_chain_features,
           "oi": chain_oi_features, "dp": dark_pool_features}


def build_date(base: Path, date: str, feeds=("hot", "oi", "dp", "tape")) -> pd.DataFrame:
    out = None
    for f in feeds:
        try:
            part = FEED_FN[f](base, date)
        except Exception as exc:  # a corrupt day must not kill a 140-day build
            print(f"    {date} {f}: FAILED {type(exc).__name__}: {exc}", flush=True)
            continue
        if part is None or part.empty:
            continue
        part = part.drop_duplicates(subset=["ticker", "date"])
        out = part if out is None else out.merge(part, on=["ticker", "date"], how="outer")
    return out if out is not None else pd.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--out-dir", default="out/deep_features")
    ap.add_argument("--feeds", default="hot,oi,dp,tape")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    base, outd = Path(args.base_dir), Path(args.out_dir)
    outd.mkdir(parents=True, exist_ok=True)
    feeds = tuple(args.feeds.split(","))
    dates = sorted(p.name for p in base.glob("20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]")
                   if p.is_dir() and args.start <= p.name <= args.end)
    print(f"{len(dates)} dated folders, feeds={feeds}", flush=True)
    for i, d in enumerate(dates, 1):
        dest = outd / f"{d}.pkl"
        if dest.exists() and not args.overwrite:
            continue
        t0 = pd.Timestamp.now()
        df = build_date(base, d, feeds)
        if df.empty:
            print(f"  [{i}/{len(dates)}] {d} EMPTY", flush=True)
            continue
        df.to_pickle(dest)
        print(f"  [{i}/{len(dates)}] {d}  {len(df):,} tickers x {df.shape[1]} cols  "
              f"{(pd.Timestamp.now() - t0).total_seconds():.1f}s", flush=True)


if __name__ == "__main__":
    main()
