#!/usr/bin/env python3
"""
Multi-Day UW OS Pack Analyzer and Trade Planner (packs-only).

Reads one or more zip inputs, discovers nested chatgpt_pack_YYYY-MM-DD.zip day packs,
computes campaign persistence signals, builds a latest-day shortlist, and produces
credit spread trade ideas in a single markdown report.
"""

from __future__ import annotations

import argparse
import io
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REQUIRED_FILES = (
    "daily_features.csv",
    "oi_carryover_signatures.csv",
    "dp_anchors.csv",
    "stock_screener.csv",
)

PACK_NAME_RE = re.compile(r"^chatgpt_pack_(\d{4}-\d{2}-\d{2})\.zip$")
CONTRACT_RE = re.compile(r"^([A-Z\.]+)-(\d{4}-\d{2}-\d{2})-([0-9]+(?:\.[0-9]+)?)-(call|put)$")


@dataclass(frozen=True)
class DayTickerFeature:
    trade_date: date
    ticker: str
    spot: float
    signed_p5: float
    signed_p10: float
    oi_confirmation_score: float
    oi_data_coverage: float
    dp_support_1: float
    dp_support_2: float
    dp_resistance_1: float
    dp_resistance_2: float
    oi_magnet_tag: str
    max_oi_strike: float
    flow_ratio: float


@dataclass(frozen=True)
class CampaignRow:
    ticker: str
    latest_date: date
    n_days: int
    latest_signed_p5: float
    latest_signed_p10: float
    latest_oi_conf: float
    latest_oi_tag: str
    persistence_slope: float
    persistence_trend: str
    oi_slope: float
    oi_trend: str
    dp_stability: str
    rank_score: float
    liq_score: float


@dataclass(frozen=True)
class TodayRow:
    ticker: str
    bias: str
    score: float
    confirmations: Tuple[str, str]
    invalidations: Tuple[str, str]
    no_sell_zone: str
    strength_label: str
    signed_p10: float
    signed_p5: float
    flow_ratio: float
    oi_conf: float
    persistence_trend: str
    oi_trend: str


@dataclass(frozen=True)
class TradeRow:
    index: int
    ticker: str
    action: str
    strategy_type: str
    strike_setup: str
    expiry: str
    dte: int
    net_credit: str
    max_profit: str
    max_loss: str
    breakeven: str
    conviction_pct: int
    confidence: str
    optimal: str
    thesis: str
    key_risks: str
    dp_anchor_invalidation: str


@dataclass(frozen=True)
class PackData:
    pack_date: date
    source: str
    daily: pd.DataFrame
    oi: pd.DataFrame
    dp: pd.DataFrame
    screener: pd.DataFrame


def fail(msg: str) -> None:
    raise SystemExit(msg)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_pack_date_from_name(name: str) -> date:
    m = PACK_NAME_RE.match(name)
    if not m:
        fail(f"Day pack filename must match chatgpt_pack_YYYY-MM-DD.zip; got: {name}")
    return datetime.strptime(m.group(1), "%Y-%m-%d").date()


def is_day_pack_zip(zf: zipfile.ZipFile) -> bool:
    names = {Path(n).name for n in zf.namelist() if not n.endswith("/")}
    return all(req in names for req in REQUIRED_FILES)


def discover_day_packs(paths: Sequence[Path]) -> Dict[date, Tuple[str, bytes]]:
    discovered: Dict[date, Tuple[str, bytes]] = {}
    seen_payload_keys: set[Tuple[int, int]] = set()

    def recurse_zip(data: bytes, display_name: str, source_label: str) -> None:
        payload_key = (len(data), hash(data[:1024]))
        if payload_key in seen_payload_keys:
            return
        seen_payload_keys.add(payload_key)

        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if is_day_pack_zip(zf):
                    d = parse_pack_date_from_name(display_name)
                    if d not in discovered:
                        discovered[d] = (source_label, data)
                for member in zf.namelist():
                    if member.lower().endswith(".zip"):
                        child = zf.read(member)
                        recurse_zip(
                            child,
                            Path(member).name,
                            f"{source_label}::{member}",
                        )
        except zipfile.BadZipFile:
            return

    for raw in paths:
        p = raw.expanduser().resolve()
        if not p.exists():
            fail(f"Pack path not found: {p}")
        if p.suffix.lower() != ".zip":
            fail(f"Pack path must be a zip file: {p}")
        recurse_zip(p.read_bytes(), p.name, str(p))

    if not discovered:
        fail("No day packs were found. Provide zip(s) containing chatgpt_pack_YYYY-MM-DD.zip files.")
    return discovered


def read_csv_from_zip_by_basename(zf: zipfile.ZipFile, basename: str) -> pd.DataFrame:
    candidates = [n for n in zf.namelist() if Path(n).name == basename]
    if not candidates:
        fail(f"Required file missing from day pack: {basename}")
    with zf.open(candidates[0], "r") as fh:
        return pd.read_csv(fh)


def load_day_pack(pack_date: date, source: str, data: bytes) -> PackData:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        missing = [f for f in REQUIRED_FILES if Path(f).name not in {Path(n).name for n in zf.namelist()}]
        if missing:
            fail(
                f"Day pack {source} ({pack_date.isoformat()}) is missing required file(s): "
                + ", ".join(missing)
            )
        daily = read_csv_from_zip_by_basename(zf, "daily_features.csv")
        oi = read_csv_from_zip_by_basename(zf, "oi_carryover_signatures.csv")
        dp = read_csv_from_zip_by_basename(zf, "dp_anchors.csv")
        screener = read_csv_from_zip_by_basename(zf, "stock_screener.csv")

    for df in (daily, oi, dp, screener):
        df["trade_date"] = pd.Timestamp(pack_date)
    for df in (daily, oi, dp, screener):
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()

    return PackData(
        pack_date=pack_date,
        source=source,
        daily=daily,
        oi=oi,
        dp=dp,
        screener=screener,
    )


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def linear_slope(y: Sequence[float], x: Optional[Sequence[float]] = None) -> float:
    arr = np.asarray(list(y), dtype="float64")
    mask = np.isfinite(arr)
    if mask.sum() < 2:
        return float("nan")
    arr = arr[mask]
    if x is None:
        xx = np.arange(len(arr), dtype="float64")
    else:
        x_arr = np.asarray(list(x), dtype="float64")
        xx = x_arr[mask]
    if len(np.unique(xx)) < 2:
        return float("nan")
    return float(np.polyfit(xx, arr, 1)[0])


def safe_min(values: Iterable[float]) -> float:
    v = [x for x in values if pd.notna(x)]
    return float(min(v)) if v else float("nan")


def safe_max(values: Iterable[float]) -> float:
    v = [x for x in values if pd.notna(x)]
    return float(max(v)) if v else float("nan")


def psych_tag_norm(tag: object) -> str:
    t = str(tag).strip().upper() if pd.notna(tag) else "NONE"
    if t in {"MAGNET/PIN", "WALL", "ZONE", "NONE"}:
        return t
    return "NONE"


def persistence_trend(latest_signed: float, slope: float) -> str:
    if not np.isfinite(slope):
        return "flat"
    if latest_signed > 0:
        if slope > 1.0:
            return "improving"
        if slope < -1.0:
            return "decaying"
        return "flat"
    if latest_signed < 0:
        if slope < -1.0:
            return "improving"
        if slope > 1.0:
            return "decaying"
        return "flat"
    return "flat"


def oi_trend_from_slope(slope: float) -> str:
    if not np.isfinite(slope):
        return "n/a"
    if slope > 1.0:
        return "improving"
    if slope < -1.0:
        return "decaying"
    return "flat"


def bias_from_signed_and_flow(signed_p10: float, flow_ratio: float) -> str:
    if signed_p10 >= 25 or (signed_p10 >= 12 and flow_ratio > 0.08):
        return "bull"
    if signed_p10 <= -25 or (signed_p10 <= -12 and flow_ratio < -0.08):
        return "bear"
    return "neutral"


def trend_bonus(trend: str) -> int:
    if trend == "improving":
        return 8
    if trend == "flat":
        return 3
    if trend == "decaying":
        return -6
    return 0


def oi_bonus(trend: str) -> int:
    if trend == "improving":
        return 4
    if trend == "flat":
        return 1
    if trend == "decaying":
        return -3
    return 0


def oi_regime(v: float) -> str:
    if not np.isfinite(v):
        return "na"
    if v < 45:
        return "low"
    if v <= 65:
        return "mid"
    return "high"


def width_by_spot(spot: float) -> float:
    if spot < 25:
        return 5.0
    if spot < 75:
        return 10.0
    if spot < 150:
        return 15.0
    return 20.0


def strike_step_by_spot(spot: float) -> float:
    if spot < 25:
        return 2.5
    return 5.0


def round_down_step(x: float, step: float) -> float:
    return float(math.floor(x / step) * step)


def round_up_step(x: float, step: float) -> float:
    return float(math.ceil(x / step) * step)


def confidence_label(conviction: int) -> str:
    if conviction >= 75:
        return "High 🔥"
    if conviction >= 55:
        return "Medium ⚖️"
    return "Low 💤"


def md_escape(text: object) -> str:
    s = str(text)
    s = s.replace("|", "\\|")
    return s


def markdown_table(rows: Sequence[Dict[str, object]], columns: Sequence[str]) -> str:
    if not rows:
        header = "| " + " | ".join(columns) + " |"
        sep = "| " + " | ".join(["---"] * len(columns)) + " |"
        return "\n".join([header, sep])
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(md_escape(row.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def normalize_data(
    daily_df: pd.DataFrame,
    dp_df: pd.DataFrame,
    oi_df: pd.DataFrame,
    screener_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    daily = daily_df.copy()
    dp = dp_df.copy()
    oi = oi_df.copy()
    screener = screener_df.copy()

    # Numeric coercion on known columns.
    num_cols_daily = [
        "persistence_5d_score",
        "persistence_5d_dominant_sign",
        "persistence_10d_score",
        "persistence_10d_dominant_sign",
        "oi_confirmation_score",
        "oi_data_coverage",
        "dp_support_1",
        "dp_support_2",
        "dp_resistance_1",
        "dp_resistance_2",
        "max_oi_strike",
        "max_oi_concentration",
        "max_oi_dist_pct",
    ]
    for c in num_cols_daily:
        if c in daily.columns:
            daily[c] = to_num(daily[c])
    if "spot" in dp.columns:
        dp["spot"] = to_num(dp["spot"])
    for c in ("dp_support_1", "dp_support_2", "dp_resistance_1", "dp_resistance_2"):
        if c in dp.columns:
            dp[c] = to_num(dp[c])

    for c in ("call_premium", "put_premium", "call_volume", "put_volume", "total_open_interest"):
        if c in screener.columns:
            screener[c] = to_num(screener[c]).fillna(0.0)
        else:
            screener[c] = 0.0

    for c in ("prev_abs", "prev_contracts", "oi_prev", "oi_cur", "oi_delta", "carryover_ratio"):
        if c in oi.columns:
            oi[c] = to_num(oi[c])

    # Spot comes from dp_anchors.
    spot_map = dp[["trade_date", "ticker", "spot"]].dropna(subset=["ticker"]).drop_duplicates(
        subset=["trade_date", "ticker"], keep="last"
    )
    daily = daily.drop(columns=["spot"], errors="ignore")
    daily = daily.merge(spot_map, on=["trade_date", "ticker"], how="left")

    # Screener aggregation and flow ratio / liquidity.
    scr_agg = (
        screener.groupby(["trade_date", "ticker"], as_index=False)[
            ["call_premium", "put_premium", "call_volume", "put_volume", "total_open_interest"]
        ]
        .sum()
        .fillna(0.0)
    )
    denom = scr_agg["call_premium"] + scr_agg["put_premium"]
    scr_agg["flow_ratio"] = np.where(denom > 0, (scr_agg["call_premium"] - scr_agg["put_premium"]) / denom, 0.0)
    scr_agg["liq_score"] = np.log1p(scr_agg["call_premium"] + scr_agg["put_premium"]) + 0.2 * np.log1p(
        scr_agg["total_open_interest"]
    )

    daily["signed_p5"] = to_num(daily["persistence_5d_score"]) * to_num(daily["persistence_5d_dominant_sign"]).fillna(0.0)
    daily["signed_p10"] = to_num(daily["persistence_10d_score"]) * to_num(daily["persistence_10d_dominant_sign"]).fillna(0.0)
    daily["oi_magnet_tag"] = daily.get("oi_magnet_tag", "none").map(psych_tag_norm)
    daily = daily.merge(
        scr_agg[["trade_date", "ticker", "flow_ratio", "liq_score", "call_premium", "put_premium", "total_open_interest"]],
        on=["trade_date", "ticker"],
        how="left",
    )
    daily["flow_ratio"] = daily["flow_ratio"].fillna(0.0)
    daily["liq_score"] = daily["liq_score"].fillna(0.0)

    # Contract parsing.
    parsed = oi["contract_signature"].astype(str).str.extract(CONTRACT_RE)
    oi["parsed_ticker"] = parsed[0].astype("string").str.upper()
    oi["expiry"] = pd.to_datetime(parsed[1], errors="coerce")
    oi["strike"] = to_num(parsed[2])
    oi["right"] = parsed[3].astype("string").str.lower()
    return daily, oi, scr_agg

def build_campaign_map(daily: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ticker, g in daily.groupby("ticker", dropna=True):
        g = g.sort_values("trade_date").copy()
        if g.empty:
            continue
        latest = g.iloc[-1]

        p_slope = linear_slope(g["signed_p10"].fillna(0.0).tolist())
        oi_vals = to_num(g.get("oi_confirmation_score", pd.Series(dtype="float64")))
        if len(oi_vals) == 0:
            o_slope = float("nan")
        else:
            o_slope = linear_slope(oi_vals.tolist(), x=np.arange(len(oi_vals)))

        spot = to_num(g.get("spot", pd.Series(dtype="float64"))).replace(0, np.nan)
        s1 = to_num(g.get("dp_support_1", pd.Series(dtype="float64")))
        r1 = to_num(g.get("dp_resistance_1", pd.Series(dtype="float64")))
        shift_values = pd.concat(
            [
                (s1.diff().abs() / spot),
                (r1.diff().abs() / spot),
            ],
            axis=0,
        ).replace([np.inf, -np.inf], np.nan)
        shift_values = shift_values[np.isfinite(shift_values)]
        if shift_values.empty:
            dp_stability = "unknown"
        else:
            dp_stability = "unchanged" if float(np.nanmedian(shift_values.values)) <= 0.01 else "shifting"

        rows.append(
            {
                "ticker": ticker,
                "latest_date": latest["trade_date"],
                "n_days": int(g["trade_date"].nunique()),
                "latest_signed_p5": float(latest.get("signed_p5", np.nan)),
                "latest_signed_p10": float(latest.get("signed_p10", np.nan)),
                "latest_oi_conf": float(latest.get("oi_confirmation_score", np.nan)),
                "latest_oi_tag": psych_tag_norm(latest.get("oi_magnet_tag", "none")),
                "persistence_slope": float(p_slope) if np.isfinite(p_slope) else float("nan"),
                "persistence_trend": persistence_trend(float(latest.get("signed_p10", 0.0)), float(p_slope) if np.isfinite(p_slope) else 0.0),
                "oi_slope": float(o_slope) if np.isfinite(o_slope) else float("nan"),
                "oi_trend": oi_trend_from_slope(float(o_slope) if np.isfinite(o_slope) else float("nan")),
                "dp_stability": dp_stability,
                "liq_score": float(latest.get("liq_score", 0.0)),
                "latest_flow_ratio": float(latest.get("flow_ratio", 0.0)),
                "latest_spot": float(latest.get("spot", np.nan)),
                "dp_support_1": float(latest.get("dp_support_1", np.nan)),
                "dp_support_2": float(latest.get("dp_support_2", np.nan)),
                "dp_resistance_1": float(latest.get("dp_resistance_1", np.nan)),
                "dp_resistance_2": float(latest.get("dp_resistance_2", np.nan)),
                "max_oi_strike": float(latest.get("max_oi_strike", np.nan)),
            }
        )
    cmap = pd.DataFrame(rows)
    if cmap.empty:
        return cmap
    cmap = cmap[cmap["latest_date"] == latest_date].copy()
    return cmap


def rank_campaigns(cmap: pd.DataFrame, side: str, top_n: int) -> pd.DataFrame:
    if cmap.empty:
        return cmap
    if side == "bull":
        out = cmap[(cmap["latest_signed_p10"] >= 22) & (cmap["n_days"] >= 3)].copy()
        out["rank_score"] = (
            out["latest_signed_p10"]
            + 0.35 * out["latest_signed_p5"]
            + out["persistence_trend"].map(trend_bonus).fillna(0)
            + out["oi_trend"].map(oi_bonus).fillna(0)
            + 2.0 * out["liq_score"].fillna(0.0)
        )
    else:
        out = cmap[(cmap["latest_signed_p10"] <= -22) & (cmap["n_days"] >= 3)].copy()
        out["rank_score"] = (
            out["latest_signed_p10"].abs()
            + 0.35 * out["latest_signed_p5"].abs()
            + out["persistence_trend"].map(trend_bonus).fillna(0)
            + out["oi_trend"].map(oi_bonus).fillna(0)
            + 2.0 * out["liq_score"].fillna(0.0)
        )
    if out.empty:
        return out
    return out.sort_values("rank_score", ascending=False).head(top_n).reset_index(drop=True)


def invalidation_text(row: pd.Series, side: str) -> str:
    s_band = [row.get("dp_support_1", np.nan), row.get("dp_support_2", np.nan)]
    r_band = [row.get("dp_resistance_1", np.nan), row.get("dp_resistance_2", np.nan)]
    if side == "bull":
        sup = safe_min(s_band)
        if np.isfinite(sup):
            return f"Break below DP support {sup:.2f}"
        return "Break below persistence support (signed P10 < +12)"
    res = safe_max(r_band)
    if np.isfinite(res):
        return f"Break above DP resistance {res:.2f}"
    return "Break above persistence resistance (signed P10 > -12)"


def campaign_bullets(row: pd.Series, side: str) -> List[str]:
    inv = invalidation_text(row, side)
    ptxt = (
        f"Persistence {row['persistence_trend']} (P10 {row['latest_signed_p10']:+.1f}, "
        f"P5 {row['latest_signed_p5']:+.1f}, {int(row['n_days'])} observed days)."
    )
    if np.isfinite(row.get("latest_oi_conf", np.nan)):
        oitxt = f"OI confirmation {row['oi_trend']} at {row['latest_oi_conf']:.1f}; DP {row['dp_stability']}; psych {row['latest_oi_tag']}."
    else:
        oitxt = f"OI confirmation unavailable; DP {row['dp_stability']}; psych {row['latest_oi_tag']}."
    return [ptxt, oitxt, inv + "."]


def compute_strength_label(latest: pd.Series, prior: Optional[pd.Series], cur_bias: str) -> str:
    if prior is None:
        return "Weakening vs prior day"
    prev_bias = bias_from_signed_and_flow(float(prior.get("signed_p10", 0.0)), float(prior.get("flow_ratio", 0.0)))
    cur_p10 = float(latest.get("signed_p10", 0.0))
    prev_p10 = float(prior.get("signed_p10", 0.0))
    cur_oi = float(latest.get("oi_confirmation_score", np.nan))
    prev_oi = float(prior.get("oi_confirmation_score", np.nan))

    if cur_bias != prev_bias and prev_bias != "neutral":
        if abs(cur_p10) >= abs(prev_p10) + 5:
            return "Strengthening vs prior day"
        return "Weakening vs prior day"

    if cur_bias == "bull":
        moved_further = (cur_p10 - prev_p10) >= 3
        moved_against = (cur_p10 - prev_p10) <= -3
    elif cur_bias == "bear":
        moved_further = (prev_p10 - cur_p10) >= 3
        moved_against = (prev_p10 - cur_p10) <= -3
    else:
        moved_further = (abs(cur_p10) - abs(prev_p10)) >= 3
        moved_against = (abs(cur_p10) - abs(prev_p10)) <= -3

    oi_up = np.isfinite(cur_oi) and np.isfinite(prev_oi) and (cur_oi - prev_oi) >= 5
    oi_down = np.isfinite(cur_oi) and np.isfinite(prev_oi) and (cur_oi - prev_oi) <= -5

    if moved_further or oi_up:
        return "Strengthening vs prior day"
    if moved_against or oi_down:
        return "Weakening vs prior day"
    return "Weakening vs prior day"


def no_sell_zone_text(row: pd.Series, bias: str) -> str:
    spot = float(row.get("spot", np.nan))
    tag = psych_tag_norm(row.get("oi_magnet_tag", "none"))
    max_oi = float(row.get("max_oi_strike", np.nan))
    pct_map = {"MAGNET/PIN": 0.010, "WALL": 0.0075, "ZONE": 0.015}

    if tag in pct_map and np.isfinite(max_oi) and np.isfinite(spot) and spot > 0:
        pct = pct_map[tag]
        psych = f"Psych {tag}: {max_oi*(1-pct):.2f}-{max_oi*(1+pct):.2f}"
    else:
        psych = "Psych: none"

    s1 = float(row.get("dp_support_1", np.nan))
    s2 = float(row.get("dp_support_2", np.nan))
    r1 = float(row.get("dp_resistance_1", np.nan))
    r2 = float(row.get("dp_resistance_2", np.nan))

    if bias == "bull":
        vals = [x for x in [s1, s2] if np.isfinite(x)]
        lo, hi = (min(vals), max(vals)) if vals else (np.nan, np.nan)
        dp_txt = f"DP support band: {lo:.2f}-{hi:.2f}" if np.isfinite(lo) and np.isfinite(hi) else "DP support: unavailable"
    elif bias == "bear":
        vals = [x for x in [r1, r2] if np.isfinite(x)]
        lo, hi = (min(vals), max(vals)) if vals else (np.nan, np.nan)
        dp_txt = f"DP resistance band: {lo:.2f}-{hi:.2f}" if np.isfinite(lo) and np.isfinite(hi) else "DP resistance: unavailable"
    else:
        svals = [x for x in [s1, s2] if np.isfinite(x)]
        rvals = [x for x in [r1, r2] if np.isfinite(x)]
        parts = []
        if svals:
            parts.append(f"S {min(svals):.2f}-{max(svals):.2f}")
        if rvals:
            parts.append(f"R {min(rvals):.2f}-{max(rvals):.2f}")
        dp_txt = "DP bands: " + " / ".join(parts) if parts else "DP bands: unavailable"

    return f"{psych} | {dp_txt}"


def build_confirmations_and_invalidations(row: pd.Series, bias: str, p_trend: str, o_trend: str) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    signed_p10 = float(row.get("signed_p10", 0.0))
    flow_ratio = float(row.get("flow_ratio", 0.0))
    oi_conf = float(row.get("oi_confirmation_score", np.nan))

    if bias == "bull":
        p_conf = f"Persistence aligned bullish (P10 {signed_p10:+.1f}, trend {p_trend})."
    elif bias == "bear":
        p_conf = f"Persistence aligned bearish (P10 {signed_p10:+.1f}, trend {p_trend})."
    else:
        p_conf = f"Persistence mixed/neutral (P10 {signed_p10:+.1f}, trend {p_trend})."

    if np.isfinite(oi_conf):
        o_conf = f"OI confirmation {o_trend} at {oi_conf:.1f}."
    else:
        o_conf = "OI confirmation unavailable in latest pack."

    if flow_ratio > 0.08:
        f_conf = f"Flow skew bullish (flow_ratio {flow_ratio:+.2f})."
    elif flow_ratio < -0.08:
        f_conf = f"Flow skew bearish (flow_ratio {flow_ratio:+.2f})."
    else:
        f_conf = f"Flow near neutral (flow_ratio {flow_ratio:+.2f})."

    if bias == "bull":
        inv1 = invalidation_text(row, "bull") + "."
        inv2 = "Invalidation if flow_ratio flips below -0.08 or OI confirmation drops by >=5."
    elif bias == "bear":
        inv1 = invalidation_text(row, "bear") + "."
        inv2 = "Invalidation if flow_ratio flips above +0.08 or OI confirmation drops by >=5."
    else:
        inv1 = "Invalidation if signed P10 exits neutral band (|P10| > 25)."
        inv2 = "Invalidation if price closes outside DP support/resistance structure."

    confirmations = [p_conf, o_conf]
    if "unavailable" in o_conf.lower():
        confirmations[1] = f_conf
    return (confirmations[0], confirmations[1]), (inv1, inv2)


def build_today_shortlist(
    daily: pd.DataFrame,
    cmap: pd.DataFrame,
    latest_date: pd.Timestamp,
    prior_date: pd.Timestamp,
    shortlist_size: int,
) -> pd.DataFrame:
    latest = daily[daily["trade_date"] == latest_date].copy()
    prior = daily[daily["trade_date"] == prior_date].copy()
    if latest.empty:
        fail(f"No daily rows found for latest date {latest_date.date()}")

    latest["bias"] = latest.apply(
        lambda r: bias_from_signed_and_flow(float(r.get("signed_p10", 0.0)), float(r.get("flow_ratio", 0.0))),
        axis=1,
    )
    oi_term = (to_num(latest["oi_confirmation_score"]).fillna(50.0) - 50.0) / 50.0
    latest["shortlist_score"] = (
        latest["signed_p10"].abs()
        + 0.4 * latest["signed_p5"].abs()
        + 8.0 * latest["flow_ratio"].abs()
        + 8.0 * oi_term.abs()
        + latest["liq_score"].fillna(0.0)
    )

    prior_map = prior.set_index("ticker")
    cmap_map = cmap.set_index("ticker") if not cmap.empty else pd.DataFrame()

    rows: List[Dict[str, object]] = []
    for _, r in latest.sort_values("shortlist_score", ascending=False).head(shortlist_size).iterrows():
        ticker = str(r["ticker"])
        bias = str(r["bias"])
        prior_row = prior_map.loc[ticker] if ticker in prior_map.index else None
        p_trend = str(cmap_map.loc[ticker, "persistence_trend"]) if not cmap_map.empty and ticker in cmap_map.index else "flat"
        o_trend = str(cmap_map.loc[ticker, "oi_trend"]) if not cmap_map.empty and ticker in cmap_map.index else "n/a"
        confirmations, invalidations = build_confirmations_and_invalidations(r, bias, p_trend, o_trend)
        strength = compute_strength_label(r, prior_row, bias)
        rows.append(
            {
                "ticker": ticker,
                "bias": bias,
                "shortlist_score": float(r["shortlist_score"]),
                "confirm_1": confirmations[0],
                "confirm_2": confirmations[1],
                "invalidate_1": invalidations[0],
                "invalidate_2": invalidations[1],
                "no_sell_zone": no_sell_zone_text(r, bias),
                "strength_label": strength,
                "signed_p10": float(r.get("signed_p10", np.nan)),
                "signed_p5": float(r.get("signed_p5", np.nan)),
                "flow_ratio": float(r.get("flow_ratio", np.nan)),
                "oi_conf": float(r.get("oi_confirmation_score", np.nan)),
                "persistence_trend": p_trend,
                "oi_trend": o_trend,
                "spot": float(r.get("spot", np.nan)),
                "dp_support_1": float(r.get("dp_support_1", np.nan)),
                "dp_support_2": float(r.get("dp_support_2", np.nan)),
                "dp_resistance_1": float(r.get("dp_resistance_1", np.nan)),
                "dp_resistance_2": float(r.get("dp_resistance_2", np.nan)),
                "oi_magnet_tag": psych_tag_norm(r.get("oi_magnet_tag", "none")),
                "max_oi_strike": float(r.get("max_oi_strike", np.nan)),
                "persistence_slope": float(cmap_map.loc[ticker, "persistence_slope"]) if (not cmap_map.empty and ticker in cmap_map.index) else float("nan"),
                "oi_slope": float(cmap_map.loc[ticker, "oi_slope"]) if (not cmap_map.empty and ticker in cmap_map.index) else float("nan"),
                "liq_score": float(r.get("liq_score", 0.0)),
            }
        )
    return pd.DataFrame(rows)

def parse_latest_oi_signatures(oi: pd.DataFrame, latest_date: pd.Timestamp) -> pd.DataFrame:
    o = oi[oi["trade_date"] == latest_date].copy()
    if o.empty:
        return pd.DataFrame(columns=["ticker", "expiry", "strike", "right", "dte", "prev_abs", "carryover_ratio"])
    # Strictly use parsed contracts only.
    o = o.dropna(subset=["parsed_ticker", "expiry", "strike", "right"])
    o["ticker"] = o["parsed_ticker"].astype(str)
    o["dte"] = (o["expiry"] - latest_date).dt.days
    o = o[(o["dte"] >= 7) & (o["dte"] <= 45)].copy()
    cols = ["ticker", "expiry", "strike", "right", "dte", "prev_abs", "carryover_ratio"]
    for c in cols:
        if c not in o.columns:
            o[c] = np.nan
    return o[cols]


def inferred_flow_bias(flow_ratio: float) -> str:
    # Derived from screener skew when side detail is unavailable:
    # positive skew => call buy / put sell tendency => bullish.
    # negative skew => put buy / call sell tendency => bearish.
    if flow_ratio > 0.08:
        return "bullish"
    if flow_ratio < -0.08:
        return "bearish"
    return "neutral"


def psych_zone_contains_short(short_strike: float, spot: float, max_oi: float, tag: str) -> bool:
    tag = psych_tag_norm(tag)
    pct_map = {"MAGNET/PIN": 0.01, "WALL": 0.0075, "ZONE": 0.015}
    if tag not in pct_map:
        return False
    if not np.isfinite(spot) or not np.isfinite(max_oi) or spot <= 0:
        return False
    pct = pct_map[tag]
    lo = max_oi * (1 - pct)
    hi = max_oi * (1 + pct)
    return lo <= short_strike <= hi


def choose_expiry_for_ticker(ticker: str, oi_latest: pd.DataFrame, global_expiry: Optional[pd.Timestamp]) -> Optional[pd.Timestamp]:
    sub = oi_latest[oi_latest["ticker"] == ticker].copy()
    if not sub.empty:
        sub["target_diff"] = (sub["dte"] - 30).abs()
        sub = sub.sort_values(["target_diff", "prev_abs"], ascending=[True, False])
        return pd.Timestamp(sub.iloc[0]["expiry"])
    return global_expiry


def choose_short_strike(
    row: pd.Series,
    oi_latest: pd.DataFrame,
    expiry: pd.Timestamp,
    trade_bias: str,
) -> Tuple[float, float, float, str, bool]:
    spot = float(row.get("spot", np.nan))
    if not np.isfinite(spot) or spot <= 0:
        return float("nan"), float("nan"), float("nan"), "missing_spot", True

    width = width_by_spot(spot)
    step = strike_step_by_spot(spot)
    ticker = str(row["ticker"])
    svals = [float(row.get("dp_support_1", np.nan)), float(row.get("dp_support_2", np.nan))]
    rvals = [float(row.get("dp_resistance_1", np.nan)), float(row.get("dp_resistance_2", np.nan))]
    supports = [x for x in svals if np.isfinite(x)]
    resistances = [x for x in rvals if np.isfinite(x)]

    sub = oi_latest[(oi_latest["ticker"] == ticker) & (oi_latest["expiry"] == expiry)].copy()
    blocked = False
    reason = ""
    if trade_bias == "bull":
        cutoff = min(supports) * 0.995 if supports else spot * 0.95
        candidates = sub[(sub["right"] == "put") & (sub["strike"] < spot) & (sub["strike"] <= cutoff)]
        if not candidates.empty:
            candidates = candidates.sort_values(["strike", "prev_abs"], ascending=[False, False])
            short = float(candidates.iloc[0]["strike"])
        else:
            short = round_down_step(min(cutoff, spot * 0.98), step)
            reason = "dp_fallback_put"
        long = short - width
        if not supports:
            blocked = True
            reason = "missing_dp_support"
    else:
        floor = max(resistances) * 1.005 if resistances else spot * 1.05
        candidates = sub[(sub["right"] == "call") & (sub["strike"] > spot) & (sub["strike"] >= floor)]
        if not candidates.empty:
            candidates = candidates.sort_values(["strike", "prev_abs"], ascending=[True, False])
            short = float(candidates.iloc[0]["strike"])
        else:
            short = round_up_step(max(floor, spot * 1.02), step)
            reason = "dp_fallback_call"
        long = short + width
        if not resistances:
            blocked = True
            reason = "missing_dp_resistance"

    # Avoid selling into MAGNET/PIN (mandatory).
    tag = psych_tag_norm(row.get("oi_magnet_tag", "none"))
    max_oi = float(row.get("max_oi_strike", np.nan))
    if tag == "MAGNET/PIN" and psych_zone_contains_short(short, spot, max_oi, tag):
        if trade_bias == "bull":
            short -= step
            long = short - width
        else:
            short += step
            long = short + width
        if psych_zone_contains_short(short, spot, max_oi, tag):
            blocked = True
            reason = "inside_magnet_pin"

    return short, long, width, reason, blocked


def compute_conviction(
    row: pd.Series,
    trade_bias: str,
    short_strike: float,
    invalidation_level: float,
    flow_mismatch: bool,
    no_sell_blocked: bool,
    missing_dp: bool,
) -> Tuple[int, str]:
    signed_p10 = float(row.get("signed_p10", 0.0))
    p_slope = float(row.get("persistence_slope", np.nan))
    oi_conf = float(row.get("oi_conf", np.nan))
    oi_trend = str(row.get("oi_trend", "n/a"))
    spot = float(row.get("spot", np.nan))
    tag = psych_tag_norm(row.get("oi_magnet_tag", "none"))

    p_component = min(35.0, 0.5 * abs(signed_p10) + 1.2 * abs(p_slope if np.isfinite(p_slope) else 0.0))

    trend_adj = 4.0 if oi_trend == "improving" else (0.0 if oi_trend == "flat" else (-3.0 if oi_trend == "decaying" else 0.0))
    if np.isfinite(oi_conf):
        oi_component = min(25.0, max(0.0, 0.45 * (oi_conf - 35.0)) + trend_adj)
    else:
        oi_component = 8.0

    if np.isfinite(invalidation_level) and np.isfinite(short_strike) and np.isfinite(spot) and spot > 0:
        dist = (short_strike - invalidation_level) if trade_bias == "bull" else (short_strike - invalidation_level)
        dp_alignment = 25.0 * clamp(dist / (0.06 * spot), 0.0, 1.0)
    else:
        dp_alignment = 0.0

    psych_penalty = {"NONE": 0.0, "ZONE": 4.0, "WALL": 8.0, "MAGNET/PIN": 14.0}.get(tag, 0.0)
    rule_penalty = 0.0
    if flow_mismatch:
        rule_penalty += 10.0
    if missing_dp:
        rule_penalty += 8.0
    if no_sell_blocked:
        rule_penalty += 12.0

    raw = 40.0 + p_component + oi_component + dp_alignment - psych_penalty - rule_penalty
    conviction = int(round(clamp(raw, 35.0, 92.0)))
    return conviction, confidence_label(conviction)


def build_trade_rows(
    shortlist: pd.DataFrame,
    daily_latest_sorted: pd.DataFrame,
    oi_latest: pd.DataFrame,
    latest_date: pd.Timestamp,
) -> List[TradeRow]:
    if shortlist.empty:
        return []

    shortlist = shortlist.copy()
    shortlist["order_key"] = shortlist["bias"].map({"bull": 0, "bear": 0, "neutral": 1}).fillna(1)
    shortlist = shortlist.sort_values(["order_key", "shortlist_score"], ascending=[True, False]).reset_index(drop=True)

    global_exp = None
    if not oi_latest.empty:
        counts = oi_latest.groupby("expiry").size().sort_values(ascending=False)
        if not counts.empty:
            global_exp = pd.Timestamp(counts.index[0])

    target = max(10, len(shortlist))

    # Supplement with extra names if strict filters reduce trades.
    extra = daily_latest_sorted.copy()
    extra["bias"] = extra.apply(
        lambda r: bias_from_signed_and_flow(float(r.get("signed_p10", 0.0)), float(r.get("flow_ratio", 0.0))),
        axis=1,
    )
    extra = extra.sort_values("shortlist_score", ascending=False)
    seen = set(shortlist["ticker"].tolist())
    extra = extra[~extra["ticker"].isin(seen)]
    extra = extra.head(max(0, target * 2))

    trade_universe = pd.concat([shortlist, extra], ignore_index=True, sort=False)

    trades: List[TradeRow] = []
    idx = 1
    for _, row in trade_universe.iterrows():
        if len(trades) >= target:
            break
        ticker = str(row["ticker"])
        bias = str(row.get("bias", "neutral"))
        trade_bias = bias
        if trade_bias == "neutral":
            trade_bias = "bull" if float(row.get("flow_ratio", 0.0)) >= 0 else "bear"

        expiry = choose_expiry_for_ticker(ticker, oi_latest, global_exp)
        if expiry is None:
            continue
        dte = int((expiry - latest_date).days)
        if dte < 7 or dte > 45:
            continue

        short_strike, long_strike, width, strike_reason, blocked = choose_short_strike(row, oi_latest, expiry, trade_bias)
        if not np.isfinite(short_strike) or not np.isfinite(long_strike):
            continue

        oi_conf = float(row.get("oi_conf", np.nan))
        est_pct = 0.27 + min(0.08, abs(float(row.get("signed_p10", 0.0))) / 500.0) + max(0.0, (oi_conf - 50.0) / 500.0 if np.isfinite(oi_conf) else 0.0)
        est_pct = clamp(est_pct, 0.25, 0.45)
        credit = round(width * est_pct, 2)
        if credit < 0.25 * width:
            continue

        max_profit = credit * 100.0
        max_loss = (width - credit) * 100.0
        breakeven = short_strike - credit if trade_bias == "bull" else short_strike + credit

        flow_bias = inferred_flow_bias(float(row.get("flow_ratio", 0.0)))
        flow_mismatch = (flow_bias == "bullish" and trade_bias == "bear") or (flow_bias == "bearish" and trade_bias == "bull")

        supports = [float(row.get("dp_support_1", np.nan)), float(row.get("dp_support_2", np.nan))]
        resistances = [float(row.get("dp_resistance_1", np.nan)), float(row.get("dp_resistance_2", np.nan))]
        inv_lvl = safe_min(supports) if trade_bias == "bull" else safe_max(resistances)
        missing_dp = not np.isfinite(inv_lvl)

        conviction, conf_lbl = compute_conviction(
            row=row,
            trade_bias=trade_bias,
            short_strike=short_strike,
            invalidation_level=inv_lvl,
            flow_mismatch=flow_mismatch,
            no_sell_blocked=blocked,
            missing_dp=missing_dp,
        )

        strategy_type = "Bull Put Credit" if trade_bias == "bull" else "Bear Call Credit"
        action = "Sell Put Spread" if trade_bias == "bull" else "Sell Call Spread"
        if trade_bias == "bull":
            strike_setup = f"Sell {short_strike:.2f}P / Buy {long_strike:.2f}P (w={width:.1f})"
            dp_anchor = safe_min(supports)
            inv_text = f"Close below {dp_anchor:.2f}" if np.isfinite(dp_anchor) else "Close below trend support"
        else:
            strike_setup = f"Sell {short_strike:.2f}C / Buy {long_strike:.2f}C (w={width:.1f})"
            dp_anchor = safe_max(resistances)
            inv_text = f"Close above {dp_anchor:.2f}" if np.isfinite(dp_anchor) else "Close above trend resistance"

        # Analyst guardrail missing -> bearish core trades capped at Watch.
        key_risks_parts = []
        if blocked:
            key_risks_parts.append("Short strike remains inside MAGNET/PIN no-sell zone")
        tag = psych_tag_norm(row.get("oi_magnet_tag", "none"))
        if tag != "NONE":
            key_risks_parts.append(f"Strike psych risk: {tag}")
        if flow_mismatch:
            key_risks_parts.append("Flow skew conflicts with chosen side")
        if missing_dp:
            key_risks_parts.append("Missing DP anchor for strict invalidation")
        if trade_bias == "bear":
            key_risks_parts.append("Analyst consensus/upside not present in pack data (guardrail unknown)")
        if np.isfinite(oi_conf) and oi_conf < 45:
            key_risks_parts.append(f"Low OI confirmation ({oi_conf:.1f})")
        if strike_reason:
            key_risks_parts.append(f"Strike source: {strike_reason}")
        if not key_risks_parts:
            key_risks_parts.append("Normal spread execution/volatility risk")

        thesis = (
            f"{trade_bias.upper()} persistence {float(row.get('signed_p10', 0.0)):+.1f}, "
            f"short strike set beyond DP constraint with {dte} DTE and {credit:.2f} credit."
        )

        if conviction >= 70 and not blocked:
            optimal = "Yes"
        elif conviction >= 55 and not blocked:
            optimal = "Watch"
        else:
            optimal = "Avoid"
        if flow_mismatch and optimal == "Yes":
            optimal = "Watch"
        if trade_bias == "bear" and optimal == "Yes":
            optimal = "Watch"

        trades.append(
            TradeRow(
                index=idx,
                ticker=ticker,
                action=action,
                strategy_type=strategy_type,
                strike_setup=strike_setup,
                expiry=expiry.date().isoformat(),
                dte=dte,
                net_credit=f"${credit:.2f}",
                max_profit=f"${max_profit:,.0f}",
                max_loss=f"${max_loss:,.0f}",
                breakeven=f"{breakeven:.2f}",
                conviction_pct=conviction,
                confidence=conf_lbl,
                optimal=optimal,
                thesis=thesis,
                key_risks="; ".join(key_risks_parts),
                dp_anchor_invalidation=(
                    f"S[{float(row.get('dp_support_1', np.nan)):.2f},{float(row.get('dp_support_2', np.nan)):.2f}] "
                    f"R[{float(row.get('dp_resistance_1', np.nan)):.2f},{float(row.get('dp_resistance_2', np.nan)):.2f}] | {inv_text}"
                ),
            )
        )
        idx += 1
    return trades


def compute_material_changes(latest: pd.DataFrame, prior: pd.DataFrame, shortlist: pd.DataFrame) -> List[str]:
    changes: List[str] = []
    if latest.empty or prior.empty or shortlist.empty:
        return ["Insufficient prior-day overlap for material change detection."]

    prior_map = prior.set_index("ticker")
    for _, row in shortlist.iterrows():
        t = str(row["ticker"])
        if t not in prior_map.index:
            continue
        prev = prior_map.loc[t]
        cur_bias = str(row["bias"])
        prev_bias = bias_from_signed_and_flow(float(prev.get("signed_p10", 0.0)), float(prev.get("flow_ratio", 0.0)))

        cur_p10 = float(row.get("signed_p10", 0.0))
        prev_p10 = float(prev.get("signed_p10", 0.0))
        if cur_bias != prev_bias and abs(cur_p10 - prev_p10) >= 5:
            changes.append(f"{t}: bias flipped {prev_bias} -> {cur_bias} with P10 shift {prev_p10:+.1f} -> {cur_p10:+.1f}.")

        cur_oi = float(row.get("oi_conf", np.nan))
        prev_oi = float(prev.get("oi_confirmation_score", np.nan))
        if oi_regime(cur_oi) != oi_regime(prev_oi):
            changes.append(f"{t}: OI regime {oi_regime(prev_oi)} -> {oi_regime(cur_oi)} ({prev_oi:.1f} -> {cur_oi:.1f}).")

        cur_tag = psych_tag_norm(row.get("oi_magnet_tag", "none"))
        prev_tag = psych_tag_norm(prev.get("oi_magnet_tag", "none"))
        if cur_tag != prev_tag:
            changes.append(f"{t}: strike psych tag changed {prev_tag} -> {cur_tag}.")

        # DP shifts > 1% of spot.
        spot = float(row.get("spot", np.nan))
        if np.isfinite(spot) and spot > 0:
            cur_s1 = float(row.get("dp_support_1", np.nan))
            prev_s1 = float(prev.get("dp_support_1", np.nan))
            cur_r1 = float(row.get("dp_resistance_1", np.nan))
            prev_r1 = float(prev.get("dp_resistance_1", np.nan))
            if np.isfinite(cur_s1) and np.isfinite(prev_s1) and abs(cur_s1 - prev_s1) / spot > 0.01:
                changes.append(f"{t}: DP support_1 moved materially ({prev_s1:.2f} -> {cur_s1:.2f}).")
            if np.isfinite(cur_r1) and np.isfinite(prev_r1) and abs(cur_r1 - prev_r1) / spot > 0.01:
                changes.append(f"{t}: DP resistance_1 moved materially ({prev_r1:.2f} -> {cur_r1:.2f}).")

    if not changes:
        return ["No material structural shifts vs prior day in the shortlisted names."]
    return changes[:12]

def render_report(
    latest_date: pd.Timestamp,
    prior_date: pd.Timestamp,
    pack_sources: Sequence[str],
    bull_ranked: pd.DataFrame,
    bear_ranked: pd.DataFrame,
    shortlist: pd.DataFrame,
    trades: Sequence[TradeRow],
    material_changes: Sequence[str],
) -> str:
    out: List[str] = []
    out.append(f"# UW Multi-Day OS Report ({latest_date.date().isoformat()})")
    out.append("")
    out.append("Data source: provided chatgpt day-pack zip(s) only.")
    out.append(f"Loaded packs: {len(pack_sources)}")
    out.append(f"Latest day: {latest_date.date().isoformat()} | Prior day: {prior_date.date().isoformat()}")
    out.append("")

    out.append("## Persistent Bullish Campaigns")
    if bull_ranked.empty:
        out.append("- None met bullish persistence filters.")
    else:
        for i, (_, r) in enumerate(bull_ranked.iterrows(), start=1):
            out.append(f"{i}. **{r['ticker']}**")
            for b in campaign_bullets(r, "bull"):
                out.append(f"   - {b}")
    out.append("")

    out.append("## Persistent Bearish Campaigns")
    if bear_ranked.empty:
        out.append("- None met bearish persistence filters.")
    else:
        for i, (_, r) in enumerate(bear_ranked.iterrows(), start=1):
            out.append(f"{i}. **{r['ticker']}**")
            for b in campaign_bullets(r, "bear"):
                out.append(f"   - {b}")
    out.append("")

    out.append("## Today's Pre-Market Shortlist")
    short_rows = []
    for _, r in shortlist.iterrows():
        short_rows.append(
            {
                "Ticker": r["ticker"],
                "Bias": r["bias"],
                "Confirmations": f"{r['confirm_1']} {r['confirm_2']}",
                "Invalidations": f"{r['invalidate_1']} {r['invalidate_2']}",
                "No-sell zones": r["no_sell_zone"],
                "Change vs prior": r["strength_label"],
            }
        )
    out.append(markdown_table(short_rows, ["Ticker", "Bias", "Confirmations", "Invalidations", "No-sell zones", "Change vs prior"]))
    out.append("")

    out.append("## Trade Table")
    trade_rows = []
    for t in trades:
        trade_rows.append(
            {
                "#": t.index,
                "Ticker": t.ticker,
                "Action": t.action,
                "Strategy Type": t.strategy_type,
                "Strike Setup": t.strike_setup,
                "Expiry": t.expiry,
                "DTE": t.dte,
                "Net Credit": t.net_credit,
                "Max Profit": t.max_profit,
                "Max Loss": t.max_loss,
                "Breakeven": t.breakeven,
                "Conviction %": t.conviction_pct,
                "Confidence": t.confidence,
                "Optimal?": t.optimal,
                "Thesis": t.thesis,
                "Key Risks": t.key_risks,
                "DP Anchor / Invalidation": t.dp_anchor_invalidation,
            }
        )
    out.append(
        markdown_table(
            trade_rows,
            [
                "#",
                "Ticker",
                "Action",
                "Strategy Type",
                "Strike Setup",
                "Expiry",
                "DTE",
                "Net Credit",
                "Max Profit",
                "Max Loss",
                "Breakeven",
                "Conviction %",
                "Confidence",
                "Optimal?",
                "Thesis",
                "Key Risks",
                "DP Anchor / Invalidation",
            ],
        )
    )
    out.append("")

    out.append("## Top 5 Campaigns Today")
    top_combined = pd.concat([bull_ranked.assign(side="bull"), bear_ranked.assign(side="bear")], ignore_index=True, sort=False)
    if top_combined.empty:
        out.append("- No campaigns met ranking gates today.")
    else:
        top_combined = top_combined.sort_values("rank_score", ascending=False).head(5)
        for _, r in top_combined.iterrows():
            side = str(r["side"]).upper()
            out.append(
                f"- {r['ticker']}: {side} | P10 {r['latest_signed_p10']:+.1f} | "
                f"Persistence {r['persistence_trend']} | OI {r['oi_trend']} | Psych {r['latest_oi_tag']}."
            )
    out.append("")

    out.append("## What changed vs prior day")
    for c in material_changes:
        out.append(f"- {c}")
    out.append("")

    out.append("## Risk-parity sizing suggestion")
    out.append("- High conviction: 1.0R each")
    out.append("- Medium conviction: 0.6R each")
    out.append("- Low conviction: 0.3R each")
    out.append("- Portfolio cap: 3.0R total open risk")
    out.append("- Sector cap: 1.2R per sector")
    out.append("")

    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-Day UW OS Pack Analyzer and Trade Planner.")
    parser.add_argument(
        "--packs",
        nargs="+",
        required=True,
        help="One or more zip paths. Can be day packs or wrapper zips containing day packs.",
    )
    parser.add_argument(
        "--shortlist-size",
        type=int,
        default=15,
        help="Latest-day shortlist size (clamped to 10..20).",
    )
    parser.add_argument(
        "--top-campaigns",
        type=int,
        default=12,
        help="Top N for each persistent campaign side.",
    )
    parser.add_argument(
        "--output-md",
        required=True,
        help="Output markdown path.",
    )
    args = parser.parse_args()

    shortlist_size = int(clamp(float(args.shortlist_size), 10.0, 20.0))
    top_campaigns = max(1, int(args.top_campaigns))

    pack_map = discover_day_packs([Path(p) for p in args.packs])
    day_items = sorted(pack_map.items(), key=lambda kv: kv[0])
    if len(day_items) < 2:
        fail("At least two day packs are required for multi-day + prior-day comparisons.")

    loaded = [load_day_pack(d, src, data) for d, (src, data) in day_items]
    def concat_safe(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame()
        non_empty = [f for f in frames if f is not None and not f.empty]
        if non_empty:
            return pd.concat(non_empty, ignore_index=True, sort=False)
        return frames[0].head(0).copy()

    all_daily = concat_safe([p.daily for p in loaded])
    all_oi = concat_safe([p.oi for p in loaded])
    all_dp = concat_safe([p.dp for p in loaded])
    all_scr = concat_safe([p.screener for p in loaded])

    daily, oi, _ = normalize_data(all_daily, all_dp, all_oi, all_scr)
    latest_date = pd.Timestamp(max(pack_map.keys()))
    prior_date = pd.Timestamp(sorted(pack_map.keys())[-2])

    cmap = build_campaign_map(daily, latest_date=latest_date)
    bull_ranked = rank_campaigns(cmap, side="bull", top_n=top_campaigns)
    bear_ranked = rank_campaigns(cmap, side="bear", top_n=top_campaigns)

    daily_latest_sorted = daily[daily["trade_date"] == latest_date].copy()
    if daily_latest_sorted.empty:
        fail(f"No latest-day daily rows for {latest_date.date().isoformat()}")
    oi_term = (to_num(daily_latest_sorted["oi_confirmation_score"]).fillna(50.0) - 50.0) / 50.0
    daily_latest_sorted["shortlist_score"] = (
        daily_latest_sorted["signed_p10"].abs()
        + 0.4 * daily_latest_sorted["signed_p5"].abs()
        + 8.0 * daily_latest_sorted["flow_ratio"].abs()
        + 8.0 * oi_term.abs()
        + daily_latest_sorted["liq_score"].fillna(0.0)
    )

    shortlist = build_today_shortlist(
        daily=daily,
        cmap=cmap,
        latest_date=latest_date,
        prior_date=prior_date,
        shortlist_size=shortlist_size,
    )

    oi_latest = parse_latest_oi_signatures(oi, latest_date=latest_date)
    trades = build_trade_rows(
        shortlist=shortlist,
        daily_latest_sorted=daily_latest_sorted,
        oi_latest=oi_latest,
        latest_date=latest_date,
    )

    prior_daily = daily[daily["trade_date"] == prior_date].copy()
    material_changes = compute_material_changes(
        latest=daily_latest_sorted,
        prior=prior_daily,
        shortlist=shortlist,
    )

    report = render_report(
        latest_date=latest_date,
        prior_date=prior_date,
        pack_sources=[src for _, (src, _) in day_items],
        bull_ranked=bull_ranked,
        bear_ranked=bear_ranked,
        shortlist=shortlist,
        trades=trades,
        material_changes=material_changes,
    )

    out_path = Path(args.output_md).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"Wrote report: {out_path}")
    print(f"Latest day: {latest_date.date().isoformat()} | Prior day: {prior_date.date().isoformat()}")
    print(f"Shortlist rows: {len(shortlist)} | Trades: {len(trades)}")


if __name__ == "__main__":
    main()
