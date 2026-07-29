"""Unified per-(date, ticker) feature panel built from ALL FIVE UW daily feeds.

Each feed answers a different question, and the pipeline was previously using
only fragments of two of them:

  stock-screener    -> is volatility cheap or expensive? (iv_rank, IV vs RV)
  hot-chains        -> HOW was it traded? (sweep / floor / multi-leg / at-ask)
  chain-oi-changes  -> did a position actually OPEN, and on which side?
  dp-eod-report     -> is an institution accumulating or distributing the stock?
  bot-eod-report    -> what greek exposure did customers put on? (delta/vega/gamma)

Per-day output is cached under out/feeds_cache/, so the expensive tape pass is
paid once.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# The tape is ~24M rows / 3.5 GB per day. Read only what is needed.
TAPE_COLS = [
    "underlying_symbol", "side", "premium", "size", "delta", "vega", "gamma",
    "underlying_price", "executed_at", "option_type",
]
TAPE_CHUNK = 2_000_000

# Dark-pool conditions that carry no directional information.
DP_EXCLUDE_CONDS = {"average_price_trade", "contingent_trade", "odd_lot_execution"}


def _ratio(num, den):
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce").replace(0, np.nan)
    return (num / den).replace([np.inf, -np.inf], np.nan)


def _open_zip(path: Path, usecols=None, chunksize=None):
    zf = zipfile.ZipFile(path)
    name = zf.namelist()[0]
    if usecols is not None:
        with zf.open(name) as fh:
            head = pd.read_csv(fh, nrows=0)
        usecols = [c for c in usecols if c in head.columns]
    return pd.read_csv(zf.open(name), usecols=usecols, chunksize=chunksize, low_memory=False)


def _find(day: Path, stem: str) -> Path | None:
    hits = _find_all(day, stem)
    return hits[0] if hits else None


def _find_all(day: Path, stem: str) -> list[Path]:
    """All archives for a feed, newest-irrelevant, correctness-first.

    Three traps live in these folders:
      * big feeds ship split as `...part-01-of-05.zip` and every part holds a
        distinct slice of the day, so reading one part silently reads 1/N;
      * folders sometimes hold a `...-latest-<other date>.zip` symlink pointing
        at a file that has since been deleted, which sorts AFTER the real file;
      * that stale symlink also carries the WRONG date's data when it resolves.
    So: drop anything that does not resolve, then prefer the archive whose name
    carries this folder's own date.
    """
    hits = [p for p in sorted(day.glob(f"{stem}-*.zip")) if p.exists()]
    if not hits:
        return []
    dated = [p for p in hits if day.name in p.name]
    hits = dated or hits
    parts = [p for p in hits if ".part-" in p.name]
    return parts if parts else hits[:1]


# ---------------------------------------------------------------- screener ---
def screener(day: Path) -> pd.DataFrame | None:
    p = _find(day, "stock-screener")
    if p is None:
        return None
    cols = ["ticker", "sector", "issue_type", "marketcap", "close", "prev_close",
            "high", "low", "total_volume", "avg30_volume", "week_52_high", "week_52_low",
            "iv30d", "iv30d_1d", "iv30d_1w", "iv30d_1m", "iv_rank", "volatility",
            "implied_move_perc", "call_volume", "put_volume",
            "call_open_interest", "put_open_interest", "prev_call_oi", "prev_put_oi",
            "avg_30_day_call_volume", "avg_30_day_put_volume",
            "bullish_premium", "bearish_premium", "net_call_premium", "net_put_premium",
            "next_earnings_date"]
    df = _open_zip(p, cols)
    if df.empty:
        return None
    df = df[df["ticker"].notna()].copy()
    if df.empty:
        return None
    for c in df.columns:
        if c not in {"ticker", "sector", "issue_type", "next_earnings_date"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    o = pd.DataFrame({"ticker": df["ticker"].astype(str).str.upper()})
    o["sector"] = df["sector"]
    o["issue_type"] = df["issue_type"]
    o["marketcap"] = df["marketcap"]
    o["close"] = df["close"]
    o["iv_rank"] = df["iv_rank"]
    o["iv30d"] = df["iv30d"]
    # Realized vol arrives corrupted on some days: on 2026-07-24 the median
    # non-backfilled value was 12.7 (1270% annualized) against 0.63 the day
    # before, topping out at 594. Left alone that silently collapses the ratio
    # and the day reports "no setups" when the truth is "this feed is broken",
    # which is the more dangerous of the two failures. Implausible values are
    # dropped per row, and if most of a day's usable rows are implausible the
    # whole day's ratio is withheld rather than half-trusted.
    rv = df["volatility"].where((df["volatility"] > 0.01) & (df["volatility"] < 5.0))
    usable = df["volatility"].notna() & (_ratio(df["iv30d"], df["volatility"]) - 1.0).abs().gt(1e-9)
    if usable.sum() and (df["volatility"][usable] > 2.0).mean() > 0.5:
        print(f"    !! {day.name}: realized-vol column looks corrupt "
              f"(median {df['volatility'][usable].median():.2f}); vrp_ratio withheld")
        rv = pd.Series(np.nan, index=df.index)
    ratio = _ratio(df["iv30d"], rv)
    # feed backfills realized with implied for ~40% of names; a flat ratio is
    # not information and must not be scored.
    o["vrp_ratio"] = ratio.where((ratio - 1.0).abs() > 1e-9)
    o["iv_chg_1w"] = df["iv30d"] - df["iv30d_1w"]
    o["iv_chg_1m"] = df["iv30d"] - df["iv30d_1m"]
    o["implied_move_perc"] = df["implied_move_perc"]
    rng = df["week_52_high"] - df["week_52_low"]
    o["pos_52w"] = _ratio(df["close"] - df["week_52_low"], rng)
    o["ret_1d"] = _ratio(df["close"], df["prev_close"]) - 1.0
    o["stock_vol_surge"] = _ratio(df["total_volume"], df["avg30_volume"])
    o["call_vol_surge"] = _ratio(df["call_volume"], df["avg_30_day_call_volume"])
    o["put_vol_surge"] = _ratio(df["put_volume"], df["avg_30_day_put_volume"])
    o["put_call_ratio"] = _ratio(df["put_volume"], df["call_volume"])
    tot = df["bullish_premium"].abs() + df["bearish_premium"].abs()
    o["prem_tilt"] = _ratio(df["bullish_premium"] - df["bearish_premium"], tot)
    o["net_prem_tilt"] = _ratio(
        df["net_call_premium"] - df["net_put_premium"],
        df["net_call_premium"].abs() + df["net_put_premium"].abs())
    o["call_oi_chg"] = _ratio(df["call_open_interest"] - df["prev_call_oi"], df["prev_call_oi"])
    o["put_oi_chg"] = _ratio(df["put_open_interest"] - df["prev_put_oi"], df["prev_put_oi"])
    o["next_earnings_date"] = df["next_earnings_date"]
    return o


# -------------------------------------------------------------- hot chains ---
def hot_chains(day: Path) -> pd.DataFrame | None:
    p = _find(day, "hot-chains")
    if p is None:
        return None
    cols = ["option_symbol", "volume", "open_interest", "premium", "ask_side_volume",
            "bid_side_volume", "mid_volume", "sweep_volume", "floor_volume",
            "cross_volume", "multileg_volume", "total_bid_changes", "total_ask_changes", "iv"]
    df = _open_zip(p, cols)
    if df.empty:
        return None
    sym = df["option_symbol"].astype(str).str.upper()
    df["ticker"] = sym.str.extract(r"^([A-Z]+)\d{6}[CP]\d+$")[0]
    df["is_call"] = sym.str.contains(r"\d{6}C\d+$", regex=True)
    df = df[df["ticker"].notna()]
    if df.empty:
        return None
    for c in cols[1:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # multi-leg volume is 24% of the tape and is NOT a directional opinion --
    # it is one leg of somebody's spread. Strip it before measuring direction.
    df["directional_vol"] = (df["volume"] - df["multileg_volume"]).clip(lower=0)
    scale = _ratio(df["directional_vol"], df["volume"]).fillna(0.0)
    df["dir_ask"] = df["ask_side_volume"] * scale
    df["dir_bid"] = df["bid_side_volume"] * scale
    df["opening_vol"] = np.where(df["volume"] > df["open_interest"], df["directional_vol"], 0.0)
    df["call_dir_ask"] = np.where(df["is_call"], df["dir_ask"], 0.0)
    df["call_dir_bid"] = np.where(df["is_call"], df["dir_bid"], 0.0)
    df["put_dir_ask"] = np.where(~df["is_call"], df["dir_ask"], 0.0)
    df["put_dir_bid"] = np.where(~df["is_call"], df["dir_bid"], 0.0)

    g = df.groupby("ticker", sort=False).agg(
        hc_volume=("volume", "sum"),
        hc_dir_vol=("directional_vol", "sum"),
        hc_multileg=("multileg_volume", "sum"),
        hc_sweep=("sweep_volume", "sum"),
        hc_floor=("floor_volume", "sum"),
        hc_cross=("cross_volume", "sum"),
        hc_opening=("opening_vol", "sum"),
        hc_premium=("premium", "sum"),
        hc_chains=("volume", "size"),
        _ca=("call_dir_ask", "sum"), _cb=("call_dir_bid", "sum"),
        _pa=("put_dir_ask", "sum"), _pb=("put_dir_bid", "sum"),
        _ackchg=("total_ask_changes", "sum"), _bidchg=("total_bid_changes", "sum"),
    )
    v = g["hc_volume"].replace(0, np.nan)
    o = pd.DataFrame({"ticker": g.index})
    o["hc_multileg_share"] = (g["hc_multileg"] / v).values
    o["hc_sweep_share"] = (g["hc_sweep"] / v).values
    o["hc_floor_share"] = (g["hc_floor"] / v).values
    o["hc_cross_share"] = (g["hc_cross"] / v).values
    o["hc_opening_share"] = (g["hc_opening"] / v).values
    o["hc_quote_churn"] = ((g["_ackchg"] + g["_bidchg"]) / v).values
    o["hc_premium"] = g["hc_premium"].values
    o["hc_chains"] = g["hc_chains"].values
    # UW convention: call@ask + put@bid = bullish ; call@bid + put@ask = bearish
    bull = g["_ca"] + g["_pb"]
    bear = g["_cb"] + g["_pa"]
    o["hc_dir_bias"] = ((bull - bear) / (bull + bear).replace(0, np.nan)).values
    return o.reset_index(drop=True)


# --------------------------------------------------------- chain OI changes ---
def oi_changes(day: Path) -> pd.DataFrame | None:
    p = _find(day, "chain-oi-changes")
    if p is None:
        return None
    cols = ["option_symbol", "underlying_symbol", "strike", "oi_diff_plain", "last_oi",
            "curr_oi", "volume", "dte", "stock_price", "prev_ask_volume",
            "prev_bid_volume", "prev_mid_volume", "prev_multi_leg_volume", "avg_price"]
    df = _open_zip(p, cols)
    if df.empty:
        return None
    for c in cols[2:]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ticker"] = df["underlying_symbol"].astype(str).str.upper()
    df["is_call"] = df["option_symbol"].astype(str).str.upper().str.contains(
        r"\d{6}C\d+$", regex=True)

    built = df[df["oi_diff_plain"] > 0].copy()
    if built.empty:
        return None
    # strip spread legs, then ask which side the new position was opened on
    pa = (built["prev_ask_volume"].fillna(0) - built["prev_multi_leg_volume"].fillna(0) / 2).clip(lower=0)
    pb = (built["prev_bid_volume"].fillna(0) - built["prev_multi_leg_volume"].fillna(0) / 2).clip(lower=0)
    built["open_dir"] = _ratio(pa - pb, pa + pb)
    # weight each new position by the dollars committed to it
    built["oi_prem"] = built["oi_diff_plain"] * built["avg_price"].fillna(0) * 100.0
    # bullish = new calls bought / new puts sold
    sign = np.where(built["is_call"], 1.0, -1.0)
    built["signed_oi_prem"] = sign * built["open_dir"].fillna(0) * built["oi_prem"]
    built["near_money"] = (
        (built["strike"] - built["stock_price"]).abs() / built["stock_price"].replace(0, np.nan)
    ) < 0.10

    g = built.groupby("ticker", sort=False).agg(
        oi_built_contracts=("oi_diff_plain", "sum"),
        oi_built_premium=("oi_prem", "sum"),
        oi_signed_premium=("signed_oi_prem", "sum"),
        oi_n_chains=("oi_diff_plain", "size"),
        oi_median_dte=("dte", "median"),
    )
    nm = built[built["near_money"]].groupby("ticker")["oi_prem"].sum().rename("oi_nearmoney_premium")
    newlong = built[built["open_dir"] > 0.5].groupby("ticker")["oi_prem"].sum().rename("oi_newlong_premium")
    newshort = built[built["open_dir"] < -0.5].groupby("ticker")["oi_prem"].sum().rename("oi_newshort_premium")
    o = g.join([nm, newlong, newshort]).reset_index()
    o["oi_dir_bias"] = _ratio(o["oi_signed_premium"], o["oi_built_premium"])
    o["oi_open_conviction"] = _ratio(
        o["oi_newlong_premium"].fillna(0) - o["oi_newshort_premium"].fillna(0),
        o["oi_newlong_premium"].fillna(0) + o["oi_newshort_premium"].fillna(0))
    o["oi_nearmoney_share"] = _ratio(o["oi_nearmoney_premium"], o["oi_built_premium"])
    return o


# --------------------------------------------------------------- dark pool ---
def dark_pool(day: Path) -> pd.DataFrame | None:
    p = _find(day, "dp-eod-report")
    if p is None:
        return None
    cols = ["ticker", "executed_at", "nbbo_bid", "nbbo_ask", "size", "premium",
            "price", "sale_cond_codes", "ext_hour_sold_codes", "canceled"]
    df = _open_zip(p, cols)
    if df.empty:
        return None
    for c in ["nbbo_bid", "nbbo_ask", "size", "premium", "price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df.get("canceled").astype(str).str.lower() != "t"]
    df = df[~df["sale_cond_codes"].isin(DP_EXCLUDE_CONDS)]
    df = df[(df["nbbo_ask"] > 0) & (df["nbbo_bid"] > 0) & (df["nbbo_ask"] >= df["nbbo_bid"])]
    if df.empty:
        return None
    df["ticker"] = df["ticker"].astype(str).str.upper()
    mid = (df["nbbo_bid"] + df["nbbo_ask"]) / 2.0
    spread = (df["nbbo_ask"] - df["nbbo_bid"]).replace(0, np.nan)
    # >0 means the print happened above the midpoint -> buyer paid up
    df["loc"] = ((df["price"] - mid) / spread).clip(-1.0, 1.0)
    df["signed_prem"] = df["loc"] * df["premium"]
    ts = pd.to_datetime(df["executed_at"], utc=True, errors="coerce")
    df["late"] = ts.dt.hour >= 19  # final ~hour of the US session in UTC
    big = df["size"] >= 10_000
    df["block_prem"] = np.where(big, df["premium"], 0.0)
    df["block_signed"] = np.where(big, df["signed_prem"], 0.0)
    df["late_signed"] = np.where(df["late"], df["signed_prem"], 0.0)
    df["late_prem"] = np.where(df["late"], df["premium"], 0.0)

    g = df.groupby("ticker", sort=False).agg(
        dp_premium=("premium", "sum"),
        dp_signed_premium=("signed_prem", "sum"),
        dp_block_premium=("block_prem", "sum"),
        dp_block_signed=("block_signed", "sum"),
        dp_late_premium=("late_prem", "sum"),
        dp_late_signed=("late_signed", "sum"),
        dp_prints=("premium", "size"),
        dp_median_size=("size", "median"),
    ).reset_index()
    g["dp_bias"] = _ratio(g["dp_signed_premium"], g["dp_premium"])
    g["dp_block_bias"] = _ratio(g["dp_block_signed"], g["dp_block_premium"])
    g["dp_late_bias"] = _ratio(g["dp_late_signed"], g["dp_late_premium"])
    g["dp_block_share"] = _ratio(g["dp_block_premium"], g["dp_premium"])
    return g.drop(columns=["dp_signed_premium", "dp_block_signed", "dp_late_signed"])


# -------------------------------------------------------------- option tape ---
def tape(day: Path) -> pd.DataFrame | None:
    """Net customer greek exposure. Dealers hold the mirror image."""
    paths = _find_all(day, "bot-eod-report")
    if not paths:
        return None
    parts = []
    for p in paths:
        for chunk in _open_zip(p, TAPE_COLS, chunksize=TAPE_CHUNK):
            parts.append(_tape_chunk(chunk))
    if not parts:
        return None
    g = pd.concat(parts).groupby(level=0).sum()
    o = pd.DataFrame({"ticker": g.index})
    o["tape_net_premium"] = g["f_prem"].values
    o["tape_delta_notional"] = g["f_delta"].values
    o["tape_vega_flow"] = g["f_vega"].values
    o["tape_gamma_flow"] = g["f_gamma"].values
    o["tape_gross_premium"] = g["abs_prem"].values
    o["tape_prem_bias"] = (g["f_prem"] / g["abs_prem"].replace(0, np.nan)).values
    o["tape_late_bias"] = (g["f_prem_late"] / g["prem_late"].replace(0, np.nan)).values
    return o.reset_index(drop=True)


def _tape_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    for c in ["premium", "size", "delta", "vega", "gamma", "underlying_price"]:
        chunk[c] = pd.to_numeric(chunk[c], errors="coerce")
    sgn = chunk["side"].map({"ask": 1.0, "bid": -1.0}).fillna(0.0)
    chunk["ticker"] = chunk["underlying_symbol"].astype(str).str.upper()
    contracts = chunk["size"].fillna(0) * 100.0
    chunk["f_prem"] = sgn * chunk["premium"].fillna(0)
    chunk["f_delta"] = sgn * chunk["delta"].fillna(0) * contracts * chunk["underlying_price"].fillna(0)
    chunk["f_vega"] = sgn * chunk["vega"].fillna(0) * contracts
    chunk["f_gamma"] = sgn * chunk["gamma"].fillna(0) * contracts
    chunk["abs_prem"] = chunk["premium"].fillna(0).abs()
    ts = pd.to_datetime(chunk["executed_at"], utc=True, errors="coerce")
    chunk["f_prem_late"] = np.where(ts.dt.hour >= 19, chunk["f_prem"], 0.0)
    chunk["prem_late"] = np.where(ts.dt.hour >= 19, chunk["abs_prem"], 0.0)
    return chunk.groupby("ticker", sort=False)[
        ["f_prem", "f_delta", "f_vega", "f_gamma", "abs_prem", "f_prem_late", "prem_late"]
    ].sum()


def whale_tape(day: Path) -> pd.DataFrame | None:
    """Customer vol BUYING from the whale feed, for days with no bot-eod.

    Deliberately not written into the `tape_*` columns, for two reasons.

    First, scale: whale is a filtered feed of roughly 300 rows and 60-120
    tickers a day against ~24M rows and ~4500 tickers for bot-eod, and no day in
    the archive holds both, so the two can never be calibrated against each
    other.

    Second, and more important, they do not measure the same thing. Every whale
    row is side=ask -- checked across files, 482 of 482 and 332 of 332. It is a
    one-sided record of customer buying, so no signed net flow can be built from
    it and any bid/ask ratio would be a constant 1.0. Only the quantities that
    survive that limitation are emitted: the gross size of the vol buying, and
    the call/put and sweep mix, which do vary.
    """
    hits = sorted(p for p in day.glob("whale_trades_filtered*.csv") if p.exists())
    if not hits:
        return None
    keep = set(TAPE_COLS) | {"report_flags"}
    frames = []
    for p in hits:
        try:
            raw = pd.read_csv(p, usecols=lambda c: c in keep, low_memory=False)
        except (ValueError, OSError):
            continue
        if raw.empty or "underlying_symbol" not in raw.columns:
            continue
        frames.append(raw)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    for c in ["premium", "size", "vega", "gamma"]:
        df[c] = pd.to_numeric(df.get(c), errors="coerce")
    df["ticker"] = df["underlying_symbol"].astype(str).str.upper()
    contracts = df["size"].fillna(0) * 100.0
    df["bought_vega"] = df["vega"].fillna(0) * contracts
    df["bought_gamma"] = df["gamma"].fillna(0) * contracts
    df["prem"] = df["premium"].fillna(0).abs()
    is_call = df.get("option_type", pd.Series("", index=df.index)).astype(str).str.lower().eq("call")
    df["call_prem"] = np.where(is_call, df["prem"], 0.0)
    flags = df.get("report_flags", pd.Series("", index=df.index)).astype(str)
    df["sweep_prem"] = np.where(flags.str.contains("sweep", case=False), df["prem"], 0.0)

    g = df.groupby("ticker", sort=False).agg(
        wtape_bought_vega=("bought_vega", "sum"),
        wtape_bought_gamma=("bought_gamma", "sum"),
        wtape_bought_premium=("prem", "sum"),
        wtape_call_premium=("call_prem", "sum"),
        wtape_sweep_premium=("sweep_prem", "sum"),
        wtape_trades=("prem", "size"),
    ).reset_index()
    g["wtape_call_share"] = _ratio(g["wtape_call_premium"], g["wtape_bought_premium"])
    g["wtape_sweep_share"] = _ratio(g["wtape_sweep_premium"], g["wtape_bought_premium"])
    return g.drop(columns=["wtape_call_premium", "wtape_sweep_premium"])


# -------------------------------------------------------------------- main ---
def build_day(day: Path, use_tape: bool) -> pd.DataFrame | None:
    sc = screener(day)
    if sc is None:
        return None
    for fn, on in ((hot_chains, "hot-chains"), (oi_changes, "oi"), (dark_pool, "dp")):
        try:
            part = fn(day)
        except Exception as exc:  # noqa: BLE001 - feeds are occasionally truncated
            print(f"    !! {on} {day.name}: {exc}")
            part = None
        if part is not None:
            sc = sc.merge(part, on="ticker", how="left")
    if use_tape:
        got_tape = False
        try:
            t = tape(day)
            if t is not None:
                sc = sc.merge(t, on="ticker", how="left")
                got_tape = True
        except Exception as exc:  # noqa: BLE001
            print(f"    !! tape {day.name}: {exc}")
        if not got_tape:
            try:
                w = whale_tape(day)
                if w is not None:
                    sc = sc.merge(w, on="ticker", how="left")
                    print(f"    .. {day.name}: no bot-eod, used whale ({len(w)} tickers)")
            except Exception as exc:  # noqa: BLE001
                print(f"    !! whale {day.name}: {exc}")
    sc.insert(0, "date", day.name)
    return sc


def repair_vrp_ratio(panel: pd.DataFrame) -> pd.DataFrame:
    """Reject the IV-vs-realized ratio on days where the vendor's realized vol breaks.

    That column is unusable on about one day in seven, and the affected days are
    overwhelmingly Fridays -- every monthly expiration in the sample among them.
    The ratio then collapses toward zero rather than going missing, so those days
    quietly produced no candidates and the approach has effectively never traded
    a Friday.

    Recomputing the ratio from the panel's own closes was tried and rejected. The
    arithmetic is right and the levels look reasonable name by name (SPY 0.148
    against the vendor's 0.156), but across the 366k rows where both exist the
    rank correlation is -0.005, so the two are not interchangeable and swapping
    one for the other would change what the gate means partway through the
    history. Restricting to consecutive sessions did not help, ruling out
    calendar gaps as the cause. Until that discrepancy is understood the honest
    move is to withhold the corrupt days rather than fill them with a number
    that has not been shown to measure the same thing.

    `realized_vol_30d` is kept for diagnosis only and must not be used to gate.
    """
    if "close" not in panel.columns or "vrp_ratio" not in panel.columns:
        return panel
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    close = pd.to_numeric(panel["close"], errors="coerce")
    logret = np.log(close / close.groupby(panel["ticker"]).shift(1))
    logret = logret.where(np.isfinite(logret))
    panel["realized_vol_30d"] = (
        logret.groupby(panel["ticker"]).rolling(30, min_periods=15).std()
        .reset_index(level=0, drop=True) * np.sqrt(252.0)
    )

    feed = pd.to_numeric(panel["vrp_ratio"], errors="coerce")
    # A corrupt realized-vol column does not arrive as nulls, it arrives as a
    # collapsed ratio, so days whose median has fallen far below the ~0.87 norm
    # are dropped outright. Doing this on the assembled panel rather than at read
    # time also stops a stale cache quietly reintroducing values the day-level
    # guard already rejected.
    day_med = feed.groupby(panel["date"]).transform("median")
    collapsed = day_med < 0.5
    if collapsed.any():
        bad_days = sorted(panel.loc[collapsed, "date"].unique())
        print(f"[feeds] vrp_ratio withheld on {len(bad_days)} days with a corrupt "
              f"realized-vol column ({bad_days[0]} .. {bad_days[-1]}); "
              f"these are mostly Fridays and will produce no candidates")
        feed = feed.where(~collapsed)
    panel["vrp_ratio"] = feed
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--out", default="out/uw_all_feeds.csv")
    ap.add_argument("--cache-dir", default="out/feeds_cache")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--skip-tape", action="store_true",
                    help="skip the 3.5 GB/day option tape (fast iteration)")
    args = ap.parse_args()

    base = Path(args.base_dir)
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    suffix = "notape" if args.skip_tape else "full"
    days = sorted(
        p for p in base.glob("20??-??-??")
        if p.is_dir() and args.start <= p.name <= args.end
    )
    print(f"[feeds] {len(days)} dated folders, tape={'off' if args.skip_tape else 'on'}")

    frames = []
    for i, d in enumerate(days, 1):
        cf = cache / f"{d.name}_{suffix}.csv"
        if cf.exists():
            frames.append(pd.read_csv(cf, low_memory=False))
            continue
        got = build_day(d, use_tape=not args.skip_tape)
        if got is None or got.empty:
            continue
        got.to_csv(cf, index=False)
        frames.append(got)
        print(f"  [{i}/{len(days)}] {d.name}  tickers={len(got)}  cols={got.shape[1]}", flush=True)

    if not frames:
        raise SystemExit("nothing built")
    panel = pd.concat(frames, ignore_index=True)
    panel = repair_vrp_ratio(panel)

    # cross-sectional daily percentile ranks -- desks compare names against the
    # day's field, not against fixed thresholds. Ranking is also what makes the
    # whale-sourced columns usable at all: their raw levels sit orders of
    # magnitude below the bot-eod tape, but a within-day rank is scale-free.
    for col in ["iv_rank", "vrp_ratio", "hc_dir_bias", "hc_sweep_share", "hc_opening_share",
                "oi_dir_bias", "oi_open_conviction", "oi_built_premium",
                "dp_bias", "dp_block_bias", "tape_prem_bias", "tape_delta_notional",
                "tape_vega_flow", "wtape_bought_vega", "wtape_bought_gamma",
                "wtape_call_share", "wtape_sweep_share",
                "call_vol_surge", "stock_vol_surge"]:
        if col in panel.columns:
            panel[f"{col}_xs"] = panel.groupby("date")[col].rank(pct=True)

    # NOTE: there is deliberately no merged "vega flow" column. tape_vega_flow is
    # a SIGNED net -- customers can be net sellers -- while wtape_bought_vega is
    # one-sided gross buying, because the whale feed carries only ask-side
    # prints. Folding them together would silently change what the column means
    # partway through the history.

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False)
    print(f"[feeds] days={panel['date'].nunique()} rows={len(panel)} cols={panel.shape[1]} -> {out}")
    cov = {c: f"{100 * panel[c].notna().mean():.0f}%" for c in panel.columns if c.startswith(
        ("hc_", "oi_", "dp_", "tape_"))}
    print("[feeds] coverage:", cov)


if __name__ == "__main__":
    main()
