"""Point-in-time (asof, ticker) features from the UW chain-oi-changes export.

This is the only one of the five daily files that carries STRIKE-LEVEL open interest
change. That matters for credit spreads specifically: the risk is not "flow was bullish",
it is "positioning is being built through the strike I am short". Everything here is
therefore measured RELATIVE TO SPOT, not in absolute premium terms.

It also carries side-tagged prior-session volume broken out by multi-leg, which lets us
apply UW's documented rule that multi-leg prints are not directional.
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OCC_RE = re.compile(r"^[A-Z]+\d{6}([CP])\d+$")

COLS = [
    "option_symbol", "underlying_symbol", "strike", "stock_price", "dte",
    "oi_diff_plain", "curr_oi", "volume",
    "prev_ask_volume", "prev_bid_volume", "prev_multi_leg_volume",
]

MAX_DTE = 90          # tradeable horizon; far-dated LEAPS build is not our risk
NEAR_BAND = 0.10      # 0-10% from spot is where short strikes actually sit


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
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _pick_chain_oi(day_dir: Path, asof: str) -> Path | None:
    """Select the export for THIS session only.

    Many folders also contain `chain-oi-changes-latest-<next-session>.zip`, which holds the
    FOLLOWING session's open interest. Globbing blindly can pick it up and leak future
    information into a point-in-time feature, so the `latest-` variant is never eligible.
    """
    exact = day_dir / f"chain-oi-changes-{asof}.zip"
    if exact.exists():
        return exact
    dated = sorted(
        p for p in day_dir.glob("chain-oi-changes-*.zip") if "latest" not in p.name
    )
    return dated[0] if dated else None


def _screener_close(day_dir: Path) -> pd.Series | None:
    """Per-ticker close from the screener, used when chain-oi `stock_price` is missing."""
    hits = sorted(day_dir.glob("stock-screener-*.zip"))
    if not hits:
        return None
    s = _read_zip(hits[0], ["ticker", "close"])
    if s is None or s.empty or "close" not in s.columns:
        return None
    s["ticker"] = s["ticker"].astype(str).str.upper()
    s["close"] = pd.to_numeric(s["close"], errors="coerce")
    s = s[s["close"] > 0]
    return s.groupby("ticker")["close"].first()


def chain_oi_features(day_dir: Path, asof: str) -> pd.DataFrame | None:
    src = _pick_chain_oi(day_dir, asof)
    if src is None:
        return None
    df = _read_zip(src, COLS)
    if df is None or df.empty or "underlying_symbol" not in df.columns:
        return None

    df = df.rename(columns={"underlying_symbol": "ticker"})
    df["ticker"] = df["ticker"].astype(str).str.upper()
    for c in ("strike", "stock_price", "dte", "oi_diff_plain", "curr_oi", "volume",
              "prev_ask_volume", "prev_bid_volume", "prev_multi_leg_volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cp = df["option_symbol"].astype(str).str.extract(OCC_RE, expand=False)
    df = df[cp.notna()].copy()
    if df.empty:
        return None
    is_call = cp[df.index].eq("C")

    # `stock_price` is entirely absent on some sessions (e.g. 2026-05-12 is 0/316,994 rows),
    # which would silently delete the whole day. Fall back to the screener close.
    spot = df["stock_price"]
    if spot.notna().mean() < 0.5:
        closes = _screener_close(day_dir)
        if closes is not None:
            spot = spot.fillna(df["ticker"].map(closes))
    df["stock_price"] = spot

    df = df[(df["stock_price"] > 0) & df["dte"].between(0, MAX_DTE)].copy()
    if df.empty:
        return None
    is_call = is_call[df.index]
    spot = df["stock_price"]

    # Signed distance from spot, expressed as a fraction of spot.
    df["_rel"] = (df["strike"] - spot) / spot
    build = df["oi_diff_plain"].fillna(0.0).clip(lower=0.0)   # positions added, not closed
    df["_build"] = build

    otm_call = is_call & (df["_rel"] > 0)
    otm_put = (~is_call) & (df["_rel"] < 0)
    df["_call_build"] = np.where(is_call, build, 0.0)
    df["_put_build"] = np.where(~is_call, build, 0.0)
    # Build in the 0-10% band from spot: the zone a short strike actually occupies.
    df["_call_build_near"] = np.where(otm_call & (df["_rel"] <= NEAR_BAND), build, 0.0)
    df["_put_build_near"] = np.where(otm_put & (df["_rel"] >= -NEAR_BAND), build, 0.0)
    # Distance-weighted build: a big build close to spot is a bigger threat than one far away.
    df["_call_press"] = np.where(otm_call, build / (1.0 + 100.0 * df["_rel"].abs()), 0.0)
    df["_put_press"] = np.where(otm_put, build / (1.0 + 100.0 * df["_rel"].abs()), 0.0)

    # UW rule: multi-leg prints are not directional. Strip them before reading side.
    ask_v = df["prev_ask_volume"].fillna(0.0)
    bid_v = df["prev_bid_volume"].fillna(0.0)
    ml = df["prev_multi_leg_volume"].fillna(0.0)
    df["_ask_v"] = ask_v
    df["_bid_v"] = bid_v
    df["_ml"] = ml

    g = df.groupby("ticker", sort=False).agg(
        coi_n=("_build", "size"),
        coi_build=("_build", "sum"),
        coi_call_build=("_call_build", "sum"),
        coi_put_build=("_put_build", "sum"),
        coi_call_build_near=("_call_build_near", "sum"),
        coi_put_build_near=("_put_build_near", "sum"),
        coi_call_press=("_call_press", "sum"),
        coi_put_press=("_put_press", "sum"),
        coi_oi=("curr_oi", "sum"),
        coi_vol=("volume", "sum"),
        _ask=("_ask_v", "sum"),
        _bid=("_bid_v", "sum"),
        _ml=("_ml", "sum"),
    ).reset_index()

    g["coi_build_dir"] = _ratio(g["coi_call_build"] - g["coi_put_build"],
                                g["coi_call_build"] + g["coi_put_build"])
    g["coi_near_dir"] = _ratio(g["coi_call_build_near"] - g["coi_put_build_near"],
                               g["coi_call_build_near"] + g["coi_put_build_near"])
    g["coi_press_dir"] = _ratio(g["coi_call_press"] - g["coi_put_press"],
                                g["coi_call_press"] + g["coi_put_press"])
    g["coi_near_share"] = _ratio(g["coi_call_build_near"] + g["coi_put_build_near"], g["coi_build"])
    g["coi_ask_ratio"] = _ratio(g["_ask"], g["_ask"] + g["_bid"])
    g["coi_multileg_share"] = _ratio(g["_ml"], g["coi_vol"])
    g["coi_build_vs_oi"] = _ratio(g["coi_build"], g["coi_oi"])
    g["coi_vol_vs_oi"] = _ratio(g["coi_vol"], g["coi_oi"])

    # Where is the largest call / put build sitting relative to spot?
    def _wall(mask: pd.Series, name: str) -> pd.DataFrame:
        sub = df[mask & (df["_build"] > 0)]
        if sub.empty:
            return pd.DataFrame(columns=["ticker", name])
        idx = sub.groupby("ticker")["_build"].idxmax()
        w = sub.loc[idx, ["ticker", "_rel"]].rename(columns={"_rel": name})
        w[name] = w[name].abs()
        return w

    g = g.merge(_wall(otm_call, "coi_call_wall_dist"), on="ticker", how="left")
    g = g.merge(_wall(otm_put, "coi_put_wall_dist"), on="ticker", how="left")

    g = g.drop(columns=["_ask", "_bid", "_ml", "coi_call_build_near", "coi_put_build_near",
                        "coi_call_press", "coi_put_press"])
    g.insert(0, "asof", asof)
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default="2026-07-24")
    ap.add_argument("--out", default="/Users/anuppamvi/uw_root/tradedesk/out/research/chain_oi_features.csv.gz")
    args = ap.parse_args()

    root = Path(args.root)
    days = sorted(
        d for d in root.iterdir()
        if d.is_dir() and DATE_RE.match(d.name) and args.start <= d.name <= args.end
    )
    print(f"scanning {len(days)} dated folders")

    frames: list[pd.DataFrame] = []
    for i, day in enumerate(days, 1):
        f = chain_oi_features(day, day.name)
        if f is not None:
            frames.append(f)
        if i % 25 == 0:
            print(f"  ...{i}/{len(days)}  days_with_data={len(frames)}", flush=True)

    if not frames:
        raise SystemExit("no chain-oi data found")

    panel = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False, compression="gzip")
    print(f"wrote {len(panel):,} rows x {panel.shape[1]} cols -> {out}")
    print(f"days: {panel['asof'].nunique()}  tickers: {panel['ticker'].nunique()}")


if __name__ == "__main__":
    main()
