"""The point-in-time ticker-day panel.

One row per (session, ticker). Every column is derivable from files dated on or
before that session, so a row can be used for a decision taken at that close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from claude_pipeline import loaders
from claude_pipeline.sources import SourceIndex, build_index

OUT_ROOT = Path("/Users/anuppamvi/tradedesk/out/claude_pipeline")
PANEL_CACHE = OUT_ROOT / "panel.csv.gz"

SCREENER_COLUMNS = [
    "ticker", "date", "close", "high", "low", "prev_close", "total_volume", "avg30_volume",
    "week_52_high", "week_52_low", "marketcap", "issue_type", "is_index", "sector",
    "next_earnings_date", "er_time", "iv30d", "iv_rank", "implied_move", "implied_move_perc",
    "call_volume", "put_volume", "call_premium", "put_premium", "put_call_ratio",
    "bullish_premium", "bearish_premium", "net_call_premium", "net_put_premium",
    "total_open_interest", "call_open_interest", "put_open_interest",
    "prev_call_oi", "prev_put_oi", "avg_30_day_call_volume", "avg_30_day_put_volume",
    "call_volume_ask_side", "call_volume_bid_side", "put_volume_ask_side", "put_volume_bid_side",
]

TRADING_DAYS = 252


def build_raw(index: SourceIndex | None = None, refresh: bool = False) -> pd.DataFrame:
    if PANEL_CACHE.exists() and not refresh:
        return pd.read_csv(PANEL_CACHE, low_memory=False)

    index = index or build_index()
    frames = []
    for session in index.sessions():
        if not index.get(session, "stock-screener"):
            continue
        frame = loaders.read(index, session, "stock-screener", columns=SCREENER_COLUMNS)
        frame["session"] = session
        frames.append(frame)

    panel = pd.concat(frames, ignore_index=True)
    panel = panel[panel["ticker"].notna()].copy()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PANEL_CACHE, index=False, compression="gzip")
    return panel


def _realized_vol(panel: pd.DataFrame, windows=(21, 63)) -> pd.DataFrame:
    wide = panel.pivot_table(index="session", columns="ticker", values="close", aggfunc="last")
    wide = wide.sort_index()
    # fill_method=None: padding a gap would manufacture a fake zero-return day.
    returns = wide.pct_change(fill_method=None)
    out = []
    for window in windows:
        vol = returns.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(TRADING_DAYS)
        out.append(vol.stack(future_stack=True).rename(f"rv{window}"))
    stacked = pd.concat(out, axis=1).reset_index()
    fwd = returns.shift(-1)
    stacked = stacked.merge(
        fwd.stack(future_stack=True).rename("next_session_return").reset_index(),
        on=["session", "ticker"], how="left",
    )
    return stacked


def add_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["session_date"] = pd.to_datetime(panel["session"])
    panel["next_earnings_date"] = pd.to_datetime(panel["next_earnings_date"], errors="coerce")
    panel["er_time"] = panel["er_time"].replace({"unkown": "unknown"})

    panel = panel.merge(_realized_vol(panel), on=["session", "ticker"], how="left")

    panel["day_return"] = panel["close"] / panel["prev_close"] - 1.0
    span = (panel["week_52_high"] - panel["week_52_low"]).replace(0, np.nan)
    panel["range_position"] = ((panel["close"] - panel["week_52_low"]) / span).clip(0, 1)
    panel["volume_surge"] = panel["total_volume"] / panel["avg30_volume"].replace(0, np.nan)

    panel["iv_rv_ratio"] = panel["iv30d"] / panel["rv21"].replace(0, np.nan)
    panel["call_oi_change"] = panel["call_open_interest"] - panel["prev_call_oi"]
    panel["put_oi_change"] = panel["put_open_interest"] - panel["prev_put_oi"]

    total_prem = (panel["bullish_premium"] + panel["bearish_premium"]).replace(0, np.nan)
    panel["bull_premium_share"] = panel["bullish_premium"] / total_prem
    panel["option_volume_surge"] = (panel["call_volume"] + panel["put_volume"]) / (
        panel["avg_30_day_call_volume"] + panel["avg_30_day_put_volume"]
    ).replace(0, np.nan)

    panel["days_to_earnings"] = (
        panel["next_earnings_date"] - panel["session_date"]
    ).dt.days.where(lambda s: s >= 0)

    panel["is_equity"] = panel["issue_type"].isin(["Common Stock", "ADR"])
    panel["dollar_volume"] = panel["close"] * panel["total_volume"]
    return panel


def build(index: SourceIndex | None = None, refresh: bool = False) -> pd.DataFrame:
    return add_features(build_raw(index, refresh=refresh))


def earnings_events(panel: pd.DataFrame) -> pd.DataFrame:
    """Actual report dates, recovered from the session on which next_earnings_date rolls forward.

    A change in the field only means "reported" if the previous date has already
    passed; otherwise the company merely rescheduled. The reported-on date is the
    PREVIOUS value, so this is observable after the fact and must never be used as
    a same-session feature.
    """
    ordered = panel.sort_values(["ticker", "session"])
    previous = ordered.groupby("ticker")["next_earnings_date"].shift(1)
    changed = ordered["next_earnings_date"].notna() & previous.notna() & (
        ordered["next_earnings_date"] != previous
    )
    reported = changed & (previous <= ordered["session_date"])

    events = ordered.loc[reported, ["ticker", "session", "er_time"]].copy()
    events["reported_on"] = previous[reported].dt.strftime("%Y-%m-%d")
    events["observed_on"] = events["session"]
    return events[["ticker", "reported_on", "observed_on", "er_time"]].reset_index(drop=True)
