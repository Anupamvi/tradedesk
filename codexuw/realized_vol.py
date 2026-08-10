"""Point-in-time realised volatility for the underlying universe.

Why this module exists
----------------------
The credit policy gates on ``iv_hv_ratio`` -- implied vol divided by realised
vol. That ratio is the single strongest premium-selling signal measured on the
2026 panel (857k ticker-sessions, 139 sessions):

    IV/RV >= 0.90  keeps 71.5% of names   capture +0.057  win 66.3%
    IV/RV >= 1.20  keeps 30.6% of names   capture +0.135  win 74.6%
    IV/RV >= 1.30  keeps 21.3% of names   capture +0.174  win 78.2%
    IV/RV >= 1.50  keeps 10.6% of names   capture +0.278  win 84.8%

("capture" is the fraction of sold implied vol that never materialised over the
following 21 sessions; monotone in every column, positive in all 7 months.)

Until now the engine hardcoded ``realized_volatility_30d = nan``, so the gate
silently fell through to the UW export's generic ``volatility`` field. That
field is unusable for this purpose: on the liquid universe it has mean 2.767
against a median of 0.445 (outlier-contaminated) and ranks only 0.698 against
true realised vol. Substituting it collapses the signal -- top-quintile capture
falls from +0.178 to +0.039 and January inverts to negative.

So we compute realised vol ourselves, from the closing prices already sitting in
the dated session folders. No API dependency, and it is point-in-time by
construction: only folders at or before ``asof`` are ever read.
"""

from __future__ import annotations

import math
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

# Session folders only. The root also holds overlay/scratch directories such as
# "2026-05-19-v3-overlay-2026-05-20-live" which sort between real sessions; if
# they are read they inject phantom trading days into the return series.
DATE_DIR = re.compile(r"^\d{4}-\d{2}-\d{2}$")

TRADING_DAYS = 252
DEFAULT_LOOKBACK = 21
DEFAULT_MIN_PERIODS = 10


def _read_closes(day_dir: Path) -> pd.DataFrame | None:
    hits = sorted(day_dir.glob("stock-screener-*.zip"))
    if not hits:
        return None
    try:
        with zipfile.ZipFile(hits[-1]) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fh:
                head = pd.read_csv(fh, nrows=0)
            cols = [c for c in ("ticker", "close") if c in head.columns]
            if len(cols) < 2:
                return None
            with zf.open(name) as fh:
                df = pd.read_csv(fh, usecols=cols, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    df = df[df["close"] > 0]
    return df.drop_duplicates(subset=["ticker"], keep="last").set_index("ticker")["close"]


def realized_vol_map(
    root: Path,
    asof,
    *,
    lookback: int = DEFAULT_LOOKBACK,
    min_periods: int = DEFAULT_MIN_PERIODS,
) -> dict[str, float]:
    """Annualised realised volatility per ticker over the trailing `lookback` sessions.

    Only sessions at or before `asof` are read, so the result is safe to use as
    an entry-time gate.
    """
    root = Path(root)
    asof_str = str(asof)[:10]
    days = sorted(
        p for p in root.iterdir()
        if p.is_dir() and DATE_DIR.match(p.name) and p.name <= asof_str
    )
    # need lookback returns -> lookback + 1 closes
    days = days[-(lookback + 1):]
    if len(days) < min_periods + 1:
        return {}

    series: dict[str, pd.Series] = {}
    for day in days:
        closes = _read_closes(day)
        if closes is not None and len(closes):
            series[day.name] = closes
    if len(series) < min_periods + 1:
        return {}

    wide = pd.DataFrame(series).sort_index(axis=1)
    # fill_method=None: a ticker missing from a session must yield a NaN return,
    # not a padded zero. Padding would manufacture flat days and understate vol
    # exactly for the thinly-covered names where the estimate matters most.
    rets = wide.pct_change(axis=1, fill_method=None)
    counts = rets.notna().sum(axis=1)
    vol = rets.std(axis=1, ddof=1) * math.sqrt(TRADING_DAYS)
    vol = vol.where(counts >= min_periods)
    vol = vol.replace([np.inf, -np.inf], np.nan).dropna()
    return {str(k): float(v) for k, v in vol.items() if v > 0}


def attach_realized_vol(
    screener: pd.DataFrame,
    root: Path,
    asof,
    *,
    lookback: int = DEFAULT_LOOKBACK,
) -> pd.DataFrame:
    """Populate `realized_volatility_30d` and `iv_hv_ratio` on the screener frame."""
    if screener is None or screener.empty or "ticker" not in screener.columns:
        return screener

    vol_map = realized_vol_map(root, asof, lookback=lookback)
    out = screener.copy()
    tickers = out["ticker"].astype(str).str.upper()
    out["realized_volatility_30d"] = tickers.map(vol_map)
    out["realized_volatility_source"] = np.where(
        out["realized_volatility_30d"].notna(),
        f"screener_close_history_{lookback}d",
        "unavailable",
    )
    iv = pd.to_numeric(out.get("iv30d"), errors="coerce")
    rv = pd.to_numeric(out["realized_volatility_30d"], errors="coerce")
    ratio = iv / rv.where(rv > 0)
    out["iv_hv_ratio"] = ratio.replace([np.inf, -np.inf], np.nan)
    out["iv_hv_spread"] = iv - rv
    return out
