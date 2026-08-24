"""Underlying price panel + forward-return targets.

Everything the pipeline has researched so far was measured on ~164 guard-passing
option rows. That is hopelessly sample-starved. Signal research belongs on the
underlying panel -- every liquid ticker on every session, tens of thousands of
observations -- and only then gets mapped onto option structures.

Builds per (asof, ticker):
  close / prev_close / marketcap / sector / volume
  realized vol (ADM21-style average daily move, as SqueezeMetrics defines it)
  forward returns at 5d and 21d, raw and volatility-normalised

Volatility normalisation matters: a 3% move in a utility and a 3% move in a
high-beta semi are not the same event. Dividing by the stock's own average daily
move makes observations comparable across tickers, which is what makes a pooled
cross-sectional test meaningful.

Output: out/research/price_panel.csv.gz
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Only true session folders. The root also holds overlay/scratch dirs such as
# "2026-05-19-v3-overlay-2026-05-20-live" which sort between real sessions and
# would inject phantom trading days into every forward-return shift.
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")

COLS = [
    "date",
    "ticker",
    "close",
    "prev_close",
    "high",
    "low",
    "marketcap",
    "sector",
    "total_volume",
    "avg30_volume",
    "iv30d",
    "iv_rank",
    "volatility",
    "implied_move_perc",
    "week_52_high",
    "week_52_low",
    "next_earnings_date",
    "is_index",
    "issue_type",
]


def read_day(path: Path, asof: str) -> pd.DataFrame | None:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as fh:
            head = pd.read_csv(fh, nrows=0)
        use = [c for c in COLS if c in head.columns]
        with zf.open(name) as fh:
            df = pd.read_csv(fh, usecols=use, low_memory=False)
    if df.empty:
        return None
    df["asof"] = asof
    df["ticker"] = df["ticker"].astype(str).str.upper()
    return df


def build(root: Path, start: str, end: str) -> pd.DataFrame:
    days = sorted(
        p for p in root.glob("2026-*")
        if p.is_dir() and DATE_DIR.match(p.name) and start <= p.name <= end
    )
    frames = []
    for day in days:
        hits = sorted(day.glob("stock-screener-*.zip"))
        if not hits:
            continue
        try:
            fr = read_day(hits[-1], day.name)
        except Exception as exc:  # noqa: BLE001
            print(f"  {day.name}: {exc}", flush=True)
            continue
        if fr is not None:
            frames.append(fr)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["ticker", "asof"]).reset_index(drop=True)
    return panel


def add_targets(panel: pd.DataFrame, horizons=(1, 5, 21)) -> pd.DataFrame:
    panel = panel.sort_values(["ticker", "asof"]).copy()
    for c in ("close", "prev_close", "marketcap", "iv30d", "volatility", "total_volume", "avg30_volume"):
        if c in panel.columns:
            panel[c] = pd.to_numeric(panel[c], errors="coerce")

    g = panel.groupby("ticker", observed=True)

    # trailing daily return off our own panel (not prev_close, which can skip gaps)
    panel["ret_1d"] = g["close"].pct_change()

    # SqueezeMetrics "average daily move" -- mean absolute daily move over 21 sessions.
    # Used as the volatility yardstick everywhere below.
    panel["adm21"] = (
        g["ret_1d"].transform(lambda s: s.abs().rolling(21, min_periods=10).mean())
    )
    panel["adm63"] = (
        g["ret_1d"].transform(lambda s: s.abs().rolling(63, min_periods=25).mean())
    )
    panel["rv21"] = g["ret_1d"].transform(lambda s: s.rolling(21, min_periods=10).std()) * np.sqrt(252)

    for h in horizons:
        fwd = g["close"].shift(-h) / panel["close"] - 1.0
        panel[f"fwd_{h}d"] = fwd
        # volatility-normalised: how many "typical daily moves" did it travel,
        # scaled by sqrt(h) so horizons are comparable
        panel[f"fwd_{h}d_z"] = fwd / (panel["adm21"] * np.sqrt(h)).replace(0, np.nan)

    # ------------------------------------------------------------------
    # Premium-seller targets.
    #
    # Direction is the wrong question for a credit spread. What actually pays is
    # whether implied volatility overstated the move that subsequently happened.
    # These targets measure exactly that, so the variance risk premium can be
    # tested as a signal in its own right rather than as a direction proxy.
    # ------------------------------------------------------------------
    fwd_rv = (
        g["ret_1d"]
        .transform(lambda s: s.shift(-21).rolling(21, min_periods=15).std())
    ) * np.sqrt(252)
    panel["rv_fwd_21d"] = fwd_rv

    # realised variance premium: what a short-vol position actually earned
    panel["vrp_realized"] = panel["iv30d"] - panel["rv_fwd_21d"]
    panel["vrp_realized_ratio"] = panel["iv30d"] / panel["rv_fwd_21d"].replace(0, np.nan)

    # Premium *capture*: the fraction of sold implied vol that never materialised.
    # Scaling by the IV sold is what makes this comparable across a 20-vol utility
    # and a 90-vol biotech -- selling 90 vol and realising 80 is a far worse trade
    # than selling 20 and realising 14, even though the raw spread favours the former.
    panel["vrp_capture"] = (panel["iv30d"] - panel["rv_fwd_21d"]) / panel["iv30d"].replace(0, np.nan)

    # did the underlying stay inside the move the option market priced?
    imp = pd.to_numeric(panel.get("implied_move_perc"), errors="coerce")
    panel["abs_fwd_21d"] = panel["fwd_21d"].abs()
    panel["move_vs_implied"] = panel["abs_fwd_21d"] / imp.replace(0, np.nan)
    panel["stayed_inside"] = (panel["move_vs_implied"] < 1.0).astype(float)
    panel.loc[panel["move_vs_implied"].isna(), "stayed_inside"] = np.nan

    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def add_axes(panel: pd.DataFrame) -> pd.DataFrame:
    """SqueezeMetrics-style state axes computed point-in-time per ticker."""
    panel = panel.sort_values(["ticker", "asof"]).copy()
    g = panel.groupby("ticker", observed=True)

    # P: volatility-adjusted price trend. Cumulative 21d move expressed in units of
    # the stock's own average daily move, squashed to [-1, 1]. A small move in a
    # quiet name registers the same as a large move in a violent one.
    r21 = g["close"].transform(lambda s: s / s.shift(21) - 1.0)
    panel["ax_P"] = np.tanh(r21 / (panel["adm21"] * np.sqrt(21)).replace(0, np.nan))

    # V: volatility trend. Is realised vol rising or falling vs its own 63d baseline?
    panel["ax_V"] = np.tanh((panel["adm21"] / panel["adm63"].replace(0, np.nan)) - 1.0)

    # IV / VRP: implied vs realised. The volatility risk premium is the single
    # most durable edge available to a premium seller, so measure it explicitly.
    panel["rv21_ann"] = panel["rv21"]
    panel["vrp"] = panel["iv30d"] - panel["rv21_ann"]
    panel["vrp_ratio"] = (panel["iv30d"] / panel["rv21_ann"].replace(0, np.nan))

    # position in the 52w range
    hi = pd.to_numeric(panel.get("week_52_high"), errors="coerce")
    lo = pd.to_numeric(panel.get("week_52_low"), errors="coerce")
    panel["range_pos_52w"] = (panel["close"] - lo) / (hi - lo).replace(0, np.nan)

    panel["dollar_vol"] = panel["close"] * panel["total_volume"]
    panel = panel.replace([np.inf, -np.inf], np.nan)
    return panel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    ap.add_argument("--start", default="2026-01-02")
    ap.add_argument("--end", default="2026-07-24")
    ap.add_argument("--out", default="/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel.csv.gz")
    args = ap.parse_args()

    panel = build(Path(args.root), args.start, args.end)
    print(f"raw {len(panel):,} rows  days {panel['asof'].nunique()}  tickers {panel['ticker'].nunique()}")
    panel = add_targets(panel)
    panel = add_axes(panel)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out, index=False, compression="gzip")
    print(f"wrote {len(panel):,} rows x {panel.shape[1]} cols -> {out}")
    cov = panel[["fwd_5d", "fwd_21d", "ax_P", "ax_V", "vrp"]].notna().mean() * 100
    print(cov.round(1).to_string())


if __name__ == "__main__":
    main()
