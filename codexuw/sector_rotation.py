from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_stock_screener


SECTOR_ROTATION_VERSION = "sector-rotation-prospective-v1.0"


def build_sector_rotation_table(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "sector", "close", "prev_close", "week_52_high", "week_52_low", "flow_total_premium"}
    if panel is None or panel.empty or not required.issubset(panel.columns):
        return pd.DataFrame()
    work = panel.copy()
    for column in ["close", "prev_close", "week_52_high", "week_52_low", "flow_total_premium"]:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(subset=["date", "ticker", "sector", "close"]).sort_values(["ticker", "date"])
    range_52w = (work["week_52_high"] - work["week_52_low"]).replace(0, np.nan)
    work["pos_52w"] = (work["close"] - work["week_52_low"]) / range_52w
    work["ret_1d"] = work["close"] / work["prev_close"].replace(0, np.nan) - 1.0
    work["flow_avg_20"] = work.groupby("ticker")["flow_total_premium"].transform(
        lambda values: values.rolling(20, min_periods=10).mean()
    )
    work["flow_escalation"] = work["flow_total_premium"] / work["flow_avg_20"].replace(0, np.nan)
    grouped = work.groupby(["date", "sector"], dropna=False)
    state = grouped.agg(
        sector_tickers=("ticker", "nunique"),
        sector_median_52w_position=("pos_52w", "median"),
        sector_breadth=("pos_52w", lambda values: float((values >= 0.60).mean())),
        sector_return_1d=("ret_1d", "median"),
        sector_flow_acceleration=("flow_escalation", "median"),
    ).reset_index()
    state = state.sort_values(["sector", "date"]).reset_index(drop=True)
    by_sector = state.groupby("sector", dropna=False)
    state["sector_momentum_change_5s"] = by_sector["sector_median_52w_position"].diff(5)
    state["sector_breadth_change_5s"] = by_sector["sector_breadth"].diff(5)
    rank_columns = {
        "sector_momentum_change_5s": 0.30,
        "sector_breadth_change_5s": 0.25,
        "sector_median_52w_position": 0.15,
        "sector_return_1d": 0.15,
        "sector_flow_acceleration": 0.15,
    }
    state["sector_emergence_score"] = 0.0
    for column, weight in rank_columns.items():
        values = pd.to_numeric(state[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        state["sector_emergence_score"] += values.groupby(state["date"]).rank(pct=True).fillna(0.5) * weight
    state["sector_state"] = np.select(
        [
            state["sector_emergence_score"].ge(0.70)
            & state["sector_momentum_change_5s"].gt(0)
            & state["sector_breadth_change_5s"].ge(0),
            state["sector_median_52w_position"].ge(0.65),
            state["sector_emergence_score"].le(0.30)
            & state["sector_momentum_change_5s"].lt(0),
        ],
        ["emerging", "established_strength", "weakening"],
        default="mixed",
    )
    state["sector_rotation_version"] = SECTOR_ROTATION_VERSION
    state["sector_rotation_authority"] = "prospective_context_only"
    return state


def build_live_sector_rotation(
    root: Path,
    *,
    asof: dt.date,
    lookback_sessions: int = 25,
) -> pd.DataFrame:
    folders: list[tuple[dt.date, Path]] = []
    for child in root.iterdir() if root.exists() else []:
        if not child.is_dir():
            continue
        try:
            day = dt.date.fromisoformat(child.name)
        except ValueError:
            continue
        if day <= asof:
            folders.append((day, child))
    frames = []
    for day, folder in sorted(folders)[-lookback_sessions:]:
        try:
            frame = load_stock_screener(folder, point_in_time=True)
        except Exception:
            continue
        if "issue_type" in frame.columns:
            frame = frame[frame["issue_type"].astype(str).str.lower().eq("common stock")]
        columns = [
            "ticker",
            "sector",
            "close",
            "prev_close",
            "week_52_high",
            "week_52_low",
            "flow_total_premium",
        ]
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
        part = frame[columns].copy()
        part["date"] = day.isoformat()
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    state = build_sector_rotation_table(pd.concat(frames, ignore_index=True))
    if state.empty:
        return state
    latest = state[state["date"].astype(str).eq(asof.isoformat())].copy()
    latest["sector_rotation_source_sessions"] = len(frames)
    return latest.reset_index(drop=True)


def apply_sector_rotation_context(scored: pd.DataFrame, sector_rotation: pd.DataFrame) -> pd.DataFrame:
    if scored is None or scored.empty:
        return scored.copy() if scored is not None else pd.DataFrame()
    out = scored.copy()
    if sector_rotation is None or sector_rotation.empty or "sector" not in out.columns:
        out["sector_state"] = "unavailable"
        out["sector_rotation_authority"] = "prospective_context_only"
        return out
    context_columns = [
        "sector",
        "sector_state",
        "sector_emergence_score",
        "sector_momentum_change_5s",
        "sector_breadth_change_5s",
        "sector_flow_acceleration",
        "sector_rotation_version",
        "sector_rotation_authority",
    ]
    available = [column for column in context_columns if column in sector_rotation.columns]
    return out.merge(sector_rotation[available].drop_duplicates("sector"), on="sector", how="left")