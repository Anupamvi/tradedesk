"""Point-in-time (asof, ticker) features from the two UW exports the Options Agent never reads.

  bot-eod-report-<date>.zip : every executed option trade, with real greeks, NBBO side tag,
                              running volume and open interest -> lets us apply UW's documented
                              rules (call@ask/put@bid = bullish; size > volume + OI = opening).
  dp-eod-report-<date>.zip  : dark pool prints -> off-exchange accumulation / distribution.

Both are aggregated to one row per (asof, ticker) so they join onto replay rows without leakage:
everything is computed from trades executed ON that session, and the Options Agent enters the
NEXT session.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

BOT_COLS = [
    "underlying_symbol", "side", "strike", "option_type", "underlying_price",
    "premium", "size", "volume", "open_interest", "implied_volatility", "delta", "expiry",
]
DP_COLS = ["ticker", "premium", "size", "price", "nbbo_bid", "nbbo_ask"]


def _read_zip(path: Path, usecols: list[str]) -> pd.DataFrame | None:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                return None
            with zf.open(names[0]) as fh:
                head = pd.read_csv(fh, nrows=0)
            have = [c for c in usecols if c in head.columns]
            if not have:
                return None
            with zf.open(names[0]) as fh:
                return pd.read_csv(fh, usecols=have, low_memory=False)
    except Exception:
        return None


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    out = num / den.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


_SUM_COLS = [
    "bot_prem", "bot_n", "bot_bull_prem", "bot_bear_prem", "bot_aggr_prem",
    "bot_open_prem", "bot_open_bull", "bot_open_bear", "bot_delta_prem",
    "bot_otm_call_prem", "bot_otm_put_prem", "_ivw",
]


def _bot_partial(df: pd.DataFrame) -> pd.DataFrame | None:
    """Reduce one chunk of the option tape to per-ticker partial sums."""
    if df.empty or "underlying_symbol" not in df.columns:
        return None
    df = df.rename(columns={"underlying_symbol": "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ("premium", "size", "volume", "open_interest", "delta",
              "implied_volatility", "strike", "underlying_price"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    side = df.get("side", pd.Series(index=df.index, dtype=object)).astype(str).str.lower()
    is_call = df.get("option_type", pd.Series(index=df.index, dtype=object)).astype(str).str.lower().eq("call")
    prem = df["premium"].fillna(0.0)

    # UW rule: CALL@ASK and PUT@BID are bullish; CALL@BID and PUT@ASK are bearish.
    at_ask, at_bid = side.eq("ask"), side.eq("bid")
    bull = (is_call & at_ask) | (~is_call & at_bid)
    bear = (is_call & at_bid) | (~is_call & at_ask)
    df["bot_bull_prem"] = np.where(bull, prem, 0.0)
    df["bot_bear_prem"] = np.where(bear, prem, 0.0)
    df["bot_aggr_prem"] = np.where(at_ask | at_bid, prem, 0.0)

    # UW rule: an opening trade has size greater than the running volume plus open interest.
    opening = df["size"] > (df["volume"].fillna(0) + df["open_interest"].fillna(0))
    df["bot_open_prem"] = np.where(opening, prem, 0.0)
    df["bot_open_bull"] = np.where(opening & bull, prem, 0.0)
    df["bot_open_bear"] = np.where(opening & bear, prem, 0.0)

    # Signed delta-weighted premium: the directional exposure actually being put on.
    sgn = np.where(bull, 1.0, np.where(bear, -1.0, 0.0))
    df["bot_delta_prem"] = df["delta"].abs().fillna(0.0) * prem * sgn

    # Premium transacted in OTM calls / OTM puts: the direct threat to a short call / short put.
    up = df["underlying_price"]
    df["bot_otm_call_prem"] = np.where(is_call & (df["strike"] > up), prem, 0.0)
    df["bot_otm_put_prem"] = np.where((~is_call) & (df["strike"] < up), prem, 0.0)
    df["_ivw"] = df["implied_volatility"].fillna(0.0) * prem
    df["bot_prem"] = prem
    df["bot_n"] = 1.0

    return df.groupby("ticker", sort=False)[_SUM_COLS].sum()


def bot_features(day_dir: Path, asof: str) -> pd.DataFrame | None:
    """Stream the option tape in chunks.

    The daily bot-eod CSV is ~3.5 GB uncompressed, so it cannot be materialised in one
    DataFrame. Some folders also split the day across several zips, and reading only the
    first part would silently truncate that session.
    """
    parts = sorted(day_dir.glob("bot-eod-report-*.zip"))
    if not parts:
        return None

    partials: list[pd.DataFrame] = []
    for path in parts:
        try:
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    continue
                with zf.open(names[0]) as fh:
                    head = pd.read_csv(fh, nrows=0)
                have = [c for c in BOT_COLS if c in head.columns]
                if "underlying_symbol" not in have:
                    continue
                with zf.open(names[0]) as fh:
                    for chunk in pd.read_csv(fh, usecols=have, chunksize=1_000_000, low_memory=False):
                        p = _bot_partial(chunk)
                        if p is not None and not p.empty:
                            partials.append(p)
        except Exception as exc:  # surface, do not silently drop a session
            print(f"    !! {asof} {path.name}: {type(exc).__name__}: {exc}", flush=True)

    if not partials:
        return None

    g = pd.concat(partials).groupby(level=0).sum().reset_index()

    g["bot_dir_ratio"] = _ratio(g["bot_bull_prem"] - g["bot_bear_prem"],
                                g["bot_bull_prem"] + g["bot_bear_prem"])
    g["bot_open_dir"] = _ratio(g["bot_open_bull"] - g["bot_open_bear"],
                               g["bot_open_bull"] + g["bot_open_bear"])
    g["bot_open_share"] = _ratio(g["bot_open_prem"], g["bot_prem"])
    g["bot_aggr_share"] = _ratio(g["bot_aggr_prem"], g["bot_prem"])
    g["bot_delta_prem_norm"] = _ratio(g["bot_delta_prem"], g["bot_prem"])
    g["bot_otm_call_share"] = _ratio(g["bot_otm_call_prem"], g["bot_prem"])
    g["bot_otm_put_share"] = _ratio(g["bot_otm_put_prem"], g["bot_prem"])
    g["bot_avg_iv"] = _ratio(g["_ivw"], g["bot_prem"])
    g = g.drop(columns=["_ivw", "bot_open_bull", "bot_open_bear",
                        "bot_aggr_prem", "bot_otm_call_prem", "bot_otm_put_prem"])
    g.insert(0, "asof", asof)
    return g


def dp_features(day_dir: Path, asof: str) -> pd.DataFrame | None:
    hits = list(day_dir.glob("dp-eod-report-*.zip"))
    if not hits:
        return None
    df = _read_zip(hits[0], DP_COLS)
    if df is None or df.empty or "ticker" not in df.columns:
        return None

    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ("premium", "size", "price", "nbbo_bid", "nbbo_ask"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # A print above the NBBO mid is buyer-initiated, below is seller-initiated.
    mid = (df["nbbo_bid"] + df["nbbo_ask"]) / 2.0
    valid = (df["nbbo_bid"] > 0) & (df["nbbo_ask"] > 0)
    prem = df["premium"].fillna(0.0)
    df["_buy"] = np.where(valid & (df["price"] > mid), prem, 0.0)
    df["_sell"] = np.where(valid & (df["price"] < mid), prem, 0.0)

    g = df.groupby("ticker", sort=False).agg(
        dp_premium=("premium", "sum"),
        dp_prints=("premium", "size"),
        dp_size_med=("size", "median"),
        _buy=("_buy", "sum"),
        _sell=("_sell", "sum"),
    ).reset_index()
    g["dp_buy_ratio"] = _ratio(g["_buy"] - g["_sell"], g["_buy"] + g["_sell"])
    g["dp_classified_share"] = _ratio(g["_buy"] + g["_sell"], g["dp_premium"])
    g = g.drop(columns=["_buy", "_sell"])
    g.insert(0, "asof", asof)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default="2026-07-24")
    ap.add_argument("--out", default="/Users/anuppamvi/uw_root/tradedesk/out/research/bot_dp_features.csv.gz")
    args = ap.parse_args()

    root = Path(args.root)
    days = sorted(
        d for d in root.iterdir()
        if d.is_dir() and DATE_RE.match(d.name) and args.start <= d.name <= args.end
    )
    print(f"scanning {len(days)} dated folders")

    bot_frames: list[pd.DataFrame] = []
    dp_frames: list[pd.DataFrame] = []
    n_bot = n_dp = 0
    for i, day in enumerate(days, 1):
        b = bot_features(day, day.name)
        if b is not None:
            bot_frames.append(b)
            n_bot += 1
        d = dp_features(day, day.name)
        if d is not None:
            dp_frames.append(d)
            n_dp += 1
        if i % 20 == 0:
            print(f"  ...{i}/{len(days)}  bot={n_bot} dp={n_dp}", flush=True)

    if not bot_frames and not dp_frames:
        raise SystemExit("no bot or dp data found")

    bot = pd.concat(bot_frames, ignore_index=True) if bot_frames else pd.DataFrame(columns=["asof", "ticker"])
    dp = pd.concat(dp_frames, ignore_index=True) if dp_frames else pd.DataFrame(columns=["asof", "ticker"])
    print(f"bot days={n_bot} rows={len(bot):,}   dp days={n_dp} rows={len(dp):,}")

    panel = bot.merge(dp, on=["asof", "ticker"], how="outer") if len(bot) and len(dp) else (bot if len(bot) else dp)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False, compression="gzip")
    print(f"wrote {len(panel):,} rows x {panel.shape[1]} cols -> {out}")
    print(f"days: {panel['asof'].nunique()}  tickers: {panel['ticker'].nunique()}")


if __name__ == "__main__":
    main()
