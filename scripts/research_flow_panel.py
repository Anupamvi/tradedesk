"""Per-(asof, ticker) option-tape aggregates from the UW bot-eod-report.

This is the feed the pipeline has never used for signal. Each row of the report is
a single executed option trade carrying the aggressor side, the NBBO at execution,
and real broker greeks. That supports the construction the literature actually
supports -- Pan & Poteshman (2006) show it is *opening, aggressor-signed* option
volume that predicts underlying returns, not raw premium.

Signal construction per trade:
  side == "ask" -> buyer-initiated  (+1)
  side == "bid" -> seller-initiated (-1)
  side == "mid"/"no_side" -> uninformative (0)

  signed delta premium = aggressor_sign * delta * premium

`delta` is signed by construction (calls positive, puts negative), so a bought put
and a sold call both register as bearish without any special-casing.

Opening interest filter: `volume > open_interest` on the contract marks a chain
where the day's activity is large relative to standing interest, i.e. more likely
position-opening than unwinding. This is the cheap within-day proxy; the exact
version joins next-session `oi_change` from chain-oi-changes.

Output: one row per (asof, ticker) -> out/research/flow_panel.csv.gz
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Only true session folders -- see research_price_panel.py for why.
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")

USECOLS = [
    "underlying_symbol",
    "side",
    "option_type",
    "premium",
    "delta",
    "gamma",
    "vega",
    "implied_volatility",
    "volume",
    "open_interest",
]

DTYPES = {
    "underlying_symbol": "category",
    "side": "category",
    "option_type": "category",
    "premium": "float32",
    "delta": "float32",
    "gamma": "float32",
    "vega": "float32",
    "implied_volatility": "float32",
    "volume": "float32",
    "open_interest": "float32",
}


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def day_flow(path: Path, asof: str) -> pd.DataFrame | None:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = pd.read_csv(fh, nrows=0)
        cols = [c for c in USECOLS if c in head.columns]
        with zf.open(name) as fh:
            df = pd.read_csv(
                fh,
                usecols=cols,
                dtype={k: v for k, v in DTYPES.items() if k in cols},
                low_memory=False,
            )
    if df.empty:
        return None

    side = df["side"].astype(str)
    sign = np.where(side == "ask", 1.0, np.where(side == "bid", -1.0, 0.0)).astype("float32")

    prem = df["premium"].fillna(0).astype("float32")
    delta = df["delta"].fillna(0).astype("float32")
    gamma = df["gamma"].fillna(0).astype("float32")
    vega = df["vega"].fillna(0).astype("float32") if "vega" in df.columns else pd.Series(0, index=df.index, dtype="float32")

    is_call = (df["option_type"].astype(str) == "call").to_numpy()
    # volume > open_interest => the day's activity is large vs standing interest
    opening = (df["volume"].fillna(0) > df["open_interest"].fillna(0)).to_numpy()

    df["_sgn_prem"] = sign * prem
    df["_sgn_delta_prem"] = sign * delta * prem
    df["_sgn_delta_prem_open"] = df["_sgn_delta_prem"].where(opening, 0.0)
    df["_sgn_vega_prem"] = sign * vega * prem
    df["_abs_prem"] = prem
    df["_open_prem"] = prem.where(opening, 0.0)
    df["_call_gamma"] = pd.Series(np.where(is_call, gamma, 0.0), index=df.index, dtype="float32")
    df["_put_gamma"] = pd.Series(np.where(~is_call, gamma, 0.0), index=df.index, dtype="float32")
    df["_call_prem"] = prem.where(pd.Series(is_call, index=df.index), 0.0)
    df["_put_prem"] = prem.where(pd.Series(~is_call, index=df.index), 0.0)
    df["_iv_w"] = df["implied_volatility"].fillna(0).astype("float32") * prem

    g = df.groupby("underlying_symbol", observed=True)
    out = g.agg(
        bot_trades=("_abs_prem", "size"),
        bot_premium=("_abs_prem", "sum"),
        bot_sgn_prem=("_sgn_prem", "sum"),
        bot_sgn_delta_prem=("_sgn_delta_prem", "sum"),
        bot_sgn_delta_prem_open=("_sgn_delta_prem_open", "sum"),
        bot_sgn_vega_prem=("_sgn_vega_prem", "sum"),
        bot_open_prem=("_open_prem", "sum"),
        bot_call_gamma=("_call_gamma", "sum"),
        bot_put_gamma=("_put_gamma", "sum"),
        bot_call_prem=("_call_prem", "sum"),
        bot_put_prem=("_put_prem", "sum"),
        bot_iv_w=("_iv_w", "sum"),
    ).reset_index()
    out = out.rename(columns={"underlying_symbol": "ticker"})

    # scale-free versions: raw premium is dominated by mega-caps and is not comparable
    out["flow_dir"] = _safe_div(out["bot_sgn_prem"], out["bot_premium"])
    out["flow_delta_dir"] = _safe_div(out["bot_sgn_delta_prem"], out["bot_premium"])
    out["flow_delta_dir_open"] = _safe_div(out["bot_sgn_delta_prem_open"], out["bot_premium"])
    out["flow_vega_dir"] = _safe_div(out["bot_sgn_vega_prem"], out["bot_premium"])
    out["flow_opening_ratio"] = _safe_div(out["bot_open_prem"], out["bot_premium"])
    out["gamma_ratio"] = _safe_div(out["bot_call_gamma"], out["bot_call_gamma"] + out["bot_put_gamma"])
    out["call_prem_share"] = _safe_div(out["bot_call_prem"], out["bot_premium"])
    out["bot_avg_iv"] = _safe_div(out["bot_iv_w"], out["bot_premium"])

    out.insert(0, "asof", asof)
    out["ticker"] = out["ticker"].astype(str).str.upper()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default="2026-07-24")
    ap.add_argument("--min-premium", type=float, default=25_000.0)
    ap.add_argument("--cache-dir", default="/Users/anuppamvi/uw_root/tradedesk/out/research/flow_cache")
    ap.add_argument("--out", default="/Users/anuppamvi/uw_root/tradedesk/out/research/flow_panel.csv.gz")
    args = ap.parse_args()

    root = Path(args.root)
    # Per-day cache. Each bot-eod file is ~12M trades and takes ~40s to read, so a
    # full build runs over an hour; writing only at the end means an interruption
    # loses everything. Caching per session makes the build resumable.
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    days = sorted(
        p for p in root.glob("2026-*")
        if p.is_dir() and DATE_DIR.match(p.name) and args.start <= p.name <= args.end
    )
    frames = []
    for i, day in enumerate(days, 1):
        cached = cache / f"{day.name}.csv"
        if cached.exists():
            frames.append(pd.read_csv(cached))
            continue
        hits = sorted(day.glob("bot-eod-report-*.zip"))
        if not hits:
            continue
        try:
            fr = day_flow(hits[-1], day.name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {day.name}: {exc}", flush=True)
            continue
        if fr is None:
            continue
        fr = fr[fr["bot_premium"] >= args.min_premium]
        fr.to_csv(cached, index=False)
        frames.append(fr)
        print(f"[{i}/{len(days)}] {day.name}: {len(fr):,} tickers", flush=True)

    if not frames:
        raise SystemExit("no bot-eod sessions found")
    panel = pd.concat(frames, ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False, compression="gzip")
    print(f"wrote {len(panel):,} rows x {panel.shape[1]} cols -> {out}")
    print(f"days {panel['asof'].nunique()}  tickers {panel['ticker'].nunique()}")


if __name__ == "__main__":
    main()
