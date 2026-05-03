#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Optional

import numpy as np
import pandas as pd


DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
OCC_RE = re.compile(r"^([A-Z\.]{1,10})(\d{6})([CP])(\d{8})$")
NORMAL = NormalDist()


@dataclass(frozen=True)
class GeneratorConfig:
    min_marketcap: float = 5_000_000_000.0
    min_avg30_volume: float = 750_000.0
    min_underlying_price: float = 15.0
    min_total_open_interest: float = 20_000.0
    min_leg_open_interest: float = 100.0
    min_leg_volume: float = 20.0
    max_leg_spread_pct: float = 0.25
    max_leg_spread_abs: float = 1.00
    min_dte: int = 30
    max_dte: int = 60
    min_iv_rank: float = 20.0
    min_credit_pct_width: float = 0.18
    max_credit_pct_width: float = 0.55
    min_short_delta: float = 0.12
    max_short_delta: float = 0.30
    target_short_delta: float = 0.20
    earnings_buffer_days: int = 7
    max_risk_per_contract: float = 850.0
    max_candidates_per_day: int = 8
    max_candidates_per_week: int = 5
    max_per_ticker: int = 1
    max_per_sector: int = 2
    max_per_direction: int = 4
    top_underlyings: int = 120
    allow_bear_call_credit: bool = True
    allow_bull_put_credit: bool = True
    allow_iron_condor: bool = False


@dataclass
class GeneratedSetup:
    signal_date: str
    ticker: str
    strategy: str
    expiry: str
    short_leg: str = ""
    long_leg: str = ""
    short_put_leg: str = ""
    long_put_leg: str = ""
    short_call_leg: str = ""
    long_call_leg: str = ""
    short_strike: float | str = ""
    long_strike: float | str = ""
    net_type: str = "credit"
    entry_net: float = math.nan
    entry_gate: str = ""
    width: float = math.nan
    max_profit: float = math.nan
    max_loss: float = math.nan
    breakeven: float | str = ""
    dte: int = 0
    spot: float = math.nan
    iv_rank: float = math.nan
    short_delta: float = math.nan
    pop_estimate: float = math.nan
    credit_pct_width: float = math.nan
    flow_bias: float = math.nan
    momentum_1d: float = math.nan
    sector: str = ""
    confidence: str = ""
    confidence_score: float = math.nan
    direction: str = ""
    score: float = math.nan
    reason: str = ""
    qty: int = 1


def parse_date(value: object) -> Optional[dt.date]:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return dt.datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def fnum(value: object) -> float:
    try:
        if value is None:
            return math.nan
        if pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y"}


def parse_occ_symbol(symbol: object) -> Optional[tuple[str, dt.date, str, float]]:
    text = str(symbol or "").upper().replace(" ", "").strip()
    match = OCC_RE.match(text)
    if not match:
        return None
    root, yymmdd, right, strike8 = match.groups()
    expiry = dt.date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
    return root, expiry, right, int(strike8) / 1000.0


def _pick_zip(day_dir: Path, prefix: str) -> Optional[Path]:
    matches = sorted(day_dir.glob(f"{prefix}*.zip"))
    return matches[-1] if matches else None


def _read_first_csv_from_zip(zip_path: Path, usecols: Optional[list[str]] = None) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted(name for name in zf.namelist() if name.lower().endswith(".csv"))
        if not names:
            return pd.DataFrame()
        return pd.read_csv(zf.open(names[0]), usecols=usecols, low_memory=False)


def read_screener(day_dir: Path) -> pd.DataFrame:
    zip_path = _pick_zip(day_dir, "stock-screener-")
    if not zip_path:
        return pd.DataFrame()
    cols = [
        "date",
        "ticker",
        "close",
        "prev_close",
        "avg30_volume",
        "total_volume",
        "marketcap",
        "issue_type",
        "is_index",
        "sector",
        "next_earnings_date",
        "bullish_premium",
        "bearish_premium",
        "call_premium",
        "put_premium",
        "call_volume",
        "put_volume",
        "total_open_interest",
        "iv30d",
        "iv_rank",
    ]
    available = _read_first_csv_from_zip(zip_path, usecols=None)
    if available.empty:
        return available
    keep = [col for col in cols if col in available.columns]
    out = available[keep].copy()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    return out.drop_duplicates("ticker", keep="last")


def read_chain_oi_quotes(day_dir: Path, wanted_tickers: set[str]) -> pd.DataFrame:
    zip_path = _pick_zip(day_dir, "chain-oi-changes-")
    if not zip_path:
        return pd.DataFrame()
    cols = [
        "option_symbol",
        "underlying_symbol",
        "strike",
        "curr_oi",
        "volume",
        "last_ask",
        "last_bid",
        "curr_date",
        "dte",
    ]
    raw = _read_first_csv_from_zip(zip_path, usecols=cols)
    if raw.empty:
        return raw
    raw = raw.copy()
    raw["ticker"] = raw["underlying_symbol"].astype(str).str.upper().str.strip()
    if wanted_tickers:
        raw = raw[raw["ticker"].isin(wanted_tickers)].copy()
    if raw.empty:
        return raw

    parsed = raw["option_symbol"].map(parse_occ_symbol)
    raw = raw[parsed.notna()].copy()
    parsed = parsed[parsed.notna()]
    raw["option_symbol"] = raw["option_symbol"].astype(str).str.upper().str.replace(" ", "", regex=False).str.strip()
    raw["expiry"] = parsed.map(lambda item: item[1])
    raw["right"] = parsed.map(lambda item: item[2])
    raw["strike"] = parsed.map(lambda item: item[3])
    raw["bid"] = pd.to_numeric(raw["last_bid"], errors="coerce")
    raw["ask"] = pd.to_numeric(raw["last_ask"], errors="coerce")
    raw["open_interest"] = pd.to_numeric(raw["curr_oi"], errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    raw["dte"] = pd.to_numeric(raw["dte"], errors="coerce")
    raw = raw[(raw["bid"] >= 0) & (raw["ask"] > 0) & (raw["ask"] >= raw["bid"])].copy()
    raw["mid"] = 0.5 * (raw["bid"] + raw["ask"])
    raw["spread"] = raw["ask"] - raw["bid"]
    raw["spread_pct"] = raw["spread"] / raw["mid"].replace(0, np.nan)
    raw = raw.sort_values(["option_symbol", "spread_pct"]).drop_duplicates("option_symbol", keep="first")
    return raw.reset_index(drop=True)


def flow_bias(row: pd.Series) -> float:
    bullish = fnum(row.get("bullish_premium"))
    bearish = fnum(row.get("bearish_premium"))
    total = bullish + bearish
    if total <= 0 or not np.isfinite(total):
        return 0.0
    return float((bullish - bearish) / total)


def momentum_1d(row: pd.Series) -> float:
    close = fnum(row.get("close"))
    prev_close = fnum(row.get("prev_close"))
    if not np.isfinite(close) or not np.isfinite(prev_close) or prev_close <= 0:
        return 0.0
    return float((close / prev_close) - 1.0)


def width_for_spot(spot: float) -> float:
    if spot < 25:
        return 2.5
    if spot < 75:
        return 5.0
    if spot < 150:
        return 5.0
    return 10.0


def delta_abs(spot: float, strike: float, iv: float, dte: float, right: str) -> float:
    if spot <= 0 or strike <= 0 or iv <= 0 or dte <= 0:
        return math.nan
    t = max(float(dte), 1.0) / 365.0
    d1 = (math.log(spot / strike) + 0.5 * iv * iv * t) / (iv * math.sqrt(t))
    if str(right).upper() == "C":
        return float(NORMAL.cdf(d1))
    return float(abs(NORMAL.cdf(d1) - 1.0))


def _liquid_legs(options: pd.DataFrame, cfg: GeneratorConfig) -> pd.DataFrame:
    if options.empty:
        return options
    out = options.copy()
    return out[
        (out["dte"].between(cfg.min_dte, cfg.max_dte))
        & (out["open_interest"].fillna(0) >= cfg.min_leg_open_interest)
        & (out["volume"].fillna(0) >= cfg.min_leg_volume)
        & (out["spread_pct"].fillna(999) <= cfg.max_leg_spread_pct)
        & (out["spread"].fillna(999) <= cfg.max_leg_spread_abs)
    ].copy()


def _confidence(score: float) -> str:
    if score >= 7:
        return "High"
    if score >= 5:
        return "Medium"
    return "Review"


def _make_vertical(
    asof: dt.date,
    row: pd.Series,
    short_leg: pd.Series,
    long_leg: pd.Series,
    strategy: str,
    direction: str,
    flow: float,
    momentum: float,
    cfg: GeneratorConfig,
) -> Optional[GeneratedSetup]:
    width = abs(fnum(short_leg["strike"]) - fnum(long_leg["strike"]))
    credit = fnum(short_leg["bid"]) - fnum(long_leg["ask"])
    if width <= 0 or credit <= 0:
        return None
    credit_pct = credit / width
    max_profit = credit * 100.0
    max_loss = max(0.0, (width - credit) * 100.0)
    if (
        credit_pct < cfg.min_credit_pct_width
        or credit_pct > cfg.max_credit_pct_width
        or max_loss > cfg.max_risk_per_contract
    ):
        return None
    short_delta = fnum(short_leg.get("delta_abs"))
    pop = max(0.0, min(1.0, 1.0 - short_delta)) if np.isfinite(short_delta) else math.nan
    iv_rank = fnum(row.get("iv_rank"))
    dte = int(fnum(short_leg["dte"]))
    short_strike = fnum(short_leg["strike"])
    long_strike = fnum(long_leg["strike"])
    if strategy == "Bull Put Credit":
        breakeven = short_strike - credit
        alignment = max(0.0, flow) + max(0.0, momentum) * 4.0
    else:
        breakeven = short_strike + credit
        alignment = max(0.0, -flow) + max(0.0, -momentum) * 4.0
    confidence_score = (
        min(3.0, max(0.0, (pop - 0.62) / 0.06))
        + min(2.0, max(0.0, (credit_pct - 0.18) / 0.08))
        + min(2.0, max(0.0, (iv_rank - 20.0) / 20.0))
        + min(2.0, max(0.0, alignment * 6.0))
        + min(1.0, max(0.0, (cfg.max_leg_spread_pct - fnum(short_leg["spread_pct"])) / cfg.max_leg_spread_pct))
    )
    score = (
        confidence_score * 100.0
        + credit_pct * 60.0
        + pop * 35.0
        + min(50.0, max(0.0, iv_rank))
        - abs(short_delta - cfg.target_short_delta) * 80.0
    )
    return GeneratedSetup(
        signal_date=asof.isoformat(),
        ticker=str(row["ticker"]),
        strategy=strategy,
        expiry=short_leg["expiry"].isoformat(),
        short_leg=str(short_leg["option_symbol"]),
        long_leg=str(long_leg["option_symbol"]),
        short_strike=float(short_strike),
        long_strike=float(long_strike),
        entry_net=round(float(credit), 2),
        entry_gate=f">= {credit:.2f} cr",
        width=float(width),
        max_profit=float(max_profit),
        max_loss=float(max_loss),
        breakeven=float(breakeven),
        dte=dte,
        spot=float(row["close"]),
        iv_rank=float(iv_rank),
        short_delta=float(short_delta),
        pop_estimate=float(pop),
        credit_pct_width=float(credit_pct),
        flow_bias=float(flow),
        momentum_1d=float(momentum),
        sector=str(row.get("sector", "") or "Unknown"),
        confidence=_confidence(confidence_score),
        confidence_score=float(confidence_score),
        direction=direction,
        score=float(score),
        reason=(
            f"liquid {strategy}; credit={credit_pct:.1%} width; "
            f"POP~{pop:.0%}; IVR={iv_rank:.1f}; flow_bias={flow:+.2f}"
        ),
    )


def _find_long_leg(
    expiry_options: pd.DataFrame,
    right: str,
    target: float,
    short_strike: float,
    lower_than_short: bool,
) -> Optional[pd.Series]:
    legs = expiry_options[expiry_options["right"].eq(right)].copy()
    if lower_than_short:
        legs = legs[legs["strike"] < short_strike]
    else:
        legs = legs[legs["strike"] > short_strike]
    if legs.empty:
        return None
    legs["_distance"] = (legs["strike"] - target).abs()
    legs = legs.sort_values(["_distance", "spread_pct", "open_interest"], ascending=[True, True, False])
    return legs.iloc[0]


def build_vertical_candidates(
    asof: dt.date,
    screener_row: pd.Series,
    ticker_options: pd.DataFrame,
    cfg: GeneratorConfig,
) -> list[GeneratedSetup]:
    spot = fnum(screener_row.get("close"))
    iv = fnum(screener_row.get("iv30d"))
    if not np.isfinite(iv) or iv <= 0:
        return []
    flow = flow_bias(screener_row)
    mom = momentum_1d(screener_row)
    width = width_for_spot(spot)
    options = _liquid_legs(ticker_options, cfg)
    if options.empty:
        return []
    options = options.copy()
    options["delta_abs"] = [
        delta_abs(spot, strike, iv, dte, right)
        for strike, dte, right in zip(options["strike"], options["dte"], options["right"])
    ]

    rows: list[GeneratedSetup] = []
    for _, expiry_options in options.groupby("expiry", sort=True):
        if cfg.allow_bull_put_credit and flow >= -0.08 and mom >= -0.025:
            puts = expiry_options[
                expiry_options["right"].eq("P")
                & (expiry_options["strike"] < spot)
                & expiry_options["delta_abs"].between(cfg.min_short_delta, cfg.max_short_delta)
            ].copy()
            puts["_delta_dist"] = (puts["delta_abs"] - cfg.target_short_delta).abs()
            for _, short_leg in puts.sort_values(["_delta_dist", "spread_pct", "open_interest"], ascending=[True, True, False]).head(3).iterrows():
                long_leg = _find_long_leg(expiry_options, "P", fnum(short_leg["strike"]) - width, fnum(short_leg["strike"]), True)
                if long_leg is None:
                    continue
                candidate = _make_vertical(asof, screener_row, short_leg, long_leg, "Bull Put Credit", "bullish", flow, mom, cfg)
                if candidate:
                    rows.append(candidate)
                    break

        if cfg.allow_bear_call_credit and flow <= 0.08 and mom <= 0.025:
            calls = expiry_options[
                expiry_options["right"].eq("C")
                & (expiry_options["strike"] > spot)
                & expiry_options["delta_abs"].between(cfg.min_short_delta, cfg.max_short_delta)
            ].copy()
            calls["_delta_dist"] = (calls["delta_abs"] - cfg.target_short_delta).abs()
            for _, short_leg in calls.sort_values(["_delta_dist", "spread_pct", "open_interest"], ascending=[True, True, False]).head(3).iterrows():
                long_leg = _find_long_leg(expiry_options, "C", fnum(short_leg["strike"]) + width, fnum(short_leg["strike"]), False)
                if long_leg is None:
                    continue
                candidate = _make_vertical(asof, screener_row, short_leg, long_leg, "Bear Call Credit", "bearish", flow, mom, cfg)
                if candidate:
                    rows.append(candidate)
                    break
    return rows


def build_iron_condor_candidates(
    asof: dt.date,
    screener_row: pd.Series,
    ticker_options: pd.DataFrame,
    cfg: GeneratorConfig,
) -> list[GeneratedSetup]:
    if not cfg.allow_iron_condor:
        return []
    spot = fnum(screener_row.get("close"))
    iv = fnum(screener_row.get("iv30d"))
    iv_rank = fnum(screener_row.get("iv_rank"))
    flow = flow_bias(screener_row)
    mom = momentum_1d(screener_row)
    if abs(flow) > 0.12 or abs(mom) > 0.025 or iv_rank < max(cfg.min_iv_rank, 30.0) or iv <= 0:
        return []
    options = _liquid_legs(ticker_options, cfg)
    if options.empty:
        return []
    options = options.copy()
    options["delta_abs"] = [
        delta_abs(spot, strike, iv, dte, right)
        for strike, dte, right in zip(options["strike"], options["dte"], options["right"])
    ]
    width = width_for_spot(spot)
    rows: list[GeneratedSetup] = []
    for _, expiry_options in options.groupby("expiry", sort=True):
        puts = expiry_options[
            expiry_options["right"].eq("P")
            & (expiry_options["strike"] < spot)
            & expiry_options["delta_abs"].between(0.10, 0.24)
        ].copy()
        calls = expiry_options[
            expiry_options["right"].eq("C")
            & (expiry_options["strike"] > spot)
            & expiry_options["delta_abs"].between(0.10, 0.24)
        ].copy()
        if puts.empty or calls.empty:
            continue
        puts["_delta_dist"] = (puts["delta_abs"] - 0.16).abs()
        calls["_delta_dist"] = (calls["delta_abs"] - 0.16).abs()
        short_put = puts.sort_values(["_delta_dist", "spread_pct"], ascending=[True, True]).iloc[0]
        short_call = calls.sort_values(["_delta_dist", "spread_pct"], ascending=[True, True]).iloc[0]
        long_put = _find_long_leg(expiry_options, "P", fnum(short_put["strike"]) - width, fnum(short_put["strike"]), True)
        long_call = _find_long_leg(expiry_options, "C", fnum(short_call["strike"]) + width, fnum(short_call["strike"]), False)
        if long_put is None or long_call is None:
            continue
        put_width = fnum(short_put["strike"]) - fnum(long_put["strike"])
        call_width = fnum(long_call["strike"]) - fnum(short_call["strike"])
        spread_width = max(put_width, call_width)
        credit = fnum(short_put["bid"]) + fnum(short_call["bid"]) - fnum(long_put["ask"]) - fnum(long_call["ask"])
        if spread_width <= 0 or credit <= 0:
            continue
        credit_pct = credit / spread_width
        max_loss = max(0.0, (spread_width - credit) * 100.0)
        if credit_pct < cfg.min_credit_pct_width or credit_pct > cfg.max_credit_pct_width or max_loss > cfg.max_risk_per_contract:
            continue
        pop = max(0.0, min(1.0, 1.0 - fnum(short_put["delta_abs"]) - fnum(short_call["delta_abs"])))
        confidence_score = (
            min(3.0, max(0.0, (pop - 0.55) / 0.07))
            + min(2.0, max(0.0, (credit_pct - 0.18) / 0.08))
            + min(2.0, max(0.0, (iv_rank - 30.0) / 20.0))
            + min(2.0, max(0.0, (0.12 - abs(flow)) / 0.06))
            + 0.5
        )
        rows.append(
            GeneratedSetup(
                signal_date=asof.isoformat(),
                ticker=str(screener_row["ticker"]),
                strategy="Iron Condor",
                expiry=short_put["expiry"].isoformat(),
                short_put_leg=str(short_put["option_symbol"]),
                long_put_leg=str(long_put["option_symbol"]),
                short_call_leg=str(short_call["option_symbol"]),
                long_call_leg=str(long_call["option_symbol"]),
                entry_net=round(float(credit), 2),
                entry_gate=f">= {credit:.2f} cr",
                width=float(spread_width),
                max_profit=float(credit * 100.0),
                max_loss=float(max_loss),
                breakeven=f"{fnum(short_put['strike']) - credit:.2f} / {fnum(short_call['strike']) + credit:.2f}",
                dte=int(fnum(short_put["dte"])),
                spot=float(spot),
                iv_rank=float(iv_rank),
                short_delta=float(max(fnum(short_put["delta_abs"]), fnum(short_call["delta_abs"]))),
                pop_estimate=float(pop),
                credit_pct_width=float(credit_pct),
                flow_bias=float(flow),
                momentum_1d=float(mom),
                sector=str(screener_row.get("sector", "") or "Unknown"),
                confidence=_confidence(confidence_score),
                confidence_score=float(confidence_score),
                direction="neutral",
                score=float(confidence_score * 100.0 + credit_pct * 60.0 + pop * 35.0 + iv_rank),
                reason=(
                    f"liquid iron condor; credit={credit_pct:.1%} width; "
                    f"POP~{pop:.0%}; IVR={iv_rank:.1f}; neutral flow"
                ),
            )
        )
        break
    return rows


def eligible_underlyings(screener: pd.DataFrame, asof: dt.date, cfg: GeneratorConfig) -> pd.DataFrame:
    if screener.empty:
        return screener
    out = screener.copy()
    for col in [
        "close",
        "prev_close",
        "avg30_volume",
        "total_volume",
        "marketcap",
        "total_open_interest",
        "iv30d",
        "iv_rank",
        "bullish_premium",
        "bearish_premium",
        "call_volume",
        "put_volume",
    ]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    issue = out.get("issue_type", pd.Series("", index=out.index)).astype(str).str.lower()
    is_index = out.get("is_index", pd.Series("", index=out.index)).map(truthy)
    earnings = out.get("next_earnings_date", pd.Series("", index=out.index)).map(parse_date)
    earnings_days = earnings.map(lambda value: (value - asof).days if value else math.nan)
    mask = (
        out["ticker"].astype(str).str.len().gt(0)
        & issue.isin({"common stock", "adr"})
        & ~is_index
        & (out["close"].fillna(0) >= cfg.min_underlying_price)
        & (out["marketcap"].fillna(0) >= cfg.min_marketcap)
        & (out["avg30_volume"].fillna(0) >= cfg.min_avg30_volume)
        & (out["total_open_interest"].fillna(0) >= cfg.min_total_open_interest)
        & (out["iv30d"].fillna(0) > 0)
        & (out["iv_rank"].fillna(0) >= cfg.min_iv_rank)
        & ~(earnings_days.notna() & (earnings_days >= 0) & (earnings_days <= cfg.earnings_buffer_days))
    )
    out = out[mask].copy()
    if out.empty:
        return out
    out["_flow_abs"] = [
        abs(flow_bias(row))
        for _, row in out.iterrows()
    ]
    out["_liquidity_score"] = (
        out["total_open_interest"].fillna(0) / 1000.0
        + out["call_volume"].fillna(0)
        + out["put_volume"].fillna(0)
        + out["_flow_abs"] * 100_000.0
    )
    return out.sort_values("_liquidity_score", ascending=False).head(cfg.top_underlyings)


def select_portfolio(candidates: Iterable[GeneratedSetup], cfg: GeneratorConfig) -> list[GeneratedSetup]:
    ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
    kept: list[GeneratedSetup] = []
    per_ticker: dict[str, int] = {}
    per_sector: dict[str, int] = {}
    per_direction: dict[str, int] = {}
    for item in ordered:
        if per_ticker.get(item.ticker, 0) >= cfg.max_per_ticker:
            continue
        sector = item.sector or "Unknown"
        if per_sector.get(sector, 0) >= cfg.max_per_sector:
            continue
        if per_direction.get(item.direction, 0) >= cfg.max_per_direction:
            continue
        kept.append(item)
        per_ticker[item.ticker] = per_ticker.get(item.ticker, 0) + 1
        per_sector[sector] = per_sector.get(sector, 0) + 1
        per_direction[item.direction] = per_direction.get(item.direction, 0) + 1
        if len(kept) >= cfg.max_candidates_per_day:
            break
    return kept


def allocate_weekly(df: pd.DataFrame, cfg: GeneratorConfig) -> pd.DataFrame:
    if df.empty or cfg.max_candidates_per_week <= 0:
        return df
    out = df.copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="coerce")
    out["_week"] = out["signal_date"].dt.to_period("W-FRI").astype(str)
    out = out.sort_values(
        ["_week", "score", "confidence_score", "pop_estimate", "credit_pct_width"],
        ascending=[True, False, False, False, False],
    )
    out = out.groupby("_week", group_keys=False).head(cfg.max_candidates_per_week).copy()
    out["signal_date"] = out["signal_date"].dt.date.map(lambda value: value.isoformat() if value else "")
    return out.drop(columns=["_week"], errors="ignore").reset_index(drop=True)


def generate_for_day(base_dir: Path, cfg: GeneratorConfig) -> tuple[pd.DataFrame, dict[str, object]]:
    asof = parse_date(base_dir.name)
    if asof is None:
        raise ValueError(f"base_dir must be a YYYY-MM-DD folder: {base_dir}")
    screener = read_screener(base_dir)
    eligible = eligible_underlyings(screener, asof, cfg)
    quotes = read_chain_oi_quotes(base_dir, set(eligible["ticker"].astype(str))) if not eligible.empty else pd.DataFrame()
    candidates: list[GeneratedSetup] = []
    if not eligible.empty and not quotes.empty:
        for _, row in eligible.iterrows():
            ticker = str(row["ticker"])
            ticker_options = quotes[quotes["ticker"].eq(ticker)]
            if ticker_options.empty:
                continue
            candidates.extend(build_vertical_candidates(asof, row, ticker_options, cfg))
            candidates.extend(build_iron_condor_candidates(asof, row, ticker_options, cfg))
    selected = select_portfolio(candidates, cfg)
    df = pd.DataFrame([asdict(item) for item in selected])
    diagnostics = {
        "date": asof.isoformat(),
        "screener_rows": int(len(screener)),
        "eligible_underlyings": int(len(eligible)),
        "quote_rows": int(len(quotes)),
        "raw_candidates": int(len(candidates)),
        "selected_candidates": int(len(df)),
        "strategy_counts": df["strategy"].value_counts().to_dict() if not df.empty else {},
    }
    return df, diagnostics


def date_dirs(root: Path, start: Optional[dt.date], end: Optional[dt.date]) -> list[Path]:
    out = []
    for path in sorted(root.iterdir()):
        if not path.is_dir() or not DATE_DIR_RE.match(path.name):
            continue
        day = parse_date(path.name)
        if day is None:
            continue
        if start and day < start:
            continue
        if end and day > end:
            continue
        out.append(path)
    return out


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows_"
    cols = [col for col in columns if col in df.columns]
    if not cols:
        return "_No rows_"
    return df[cols].fillna("").to_markdown(index=False)


def write_outputs(df: pd.DataFrame, diagnostics: dict[str, object], out_dir: Path, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"weekly_trade_candidates_{label}.csv"
    exact_path = out_dir / f"weekly_trade_setups_for_exact_backtest_{label}.csv"
    md_path = out_dir / f"weekly_trade_candidates_{label}.md"
    manifest_path = out_dir / f"weekly_trade_generator_manifest_{label}.json"
    df.to_csv(csv_path, index=False)
    exact_cols = [
        "signal_date",
        "ticker",
        "strategy",
        "expiry",
        "short_leg",
        "long_leg",
        "short_put_leg",
        "long_put_leg",
        "short_call_leg",
        "long_call_leg",
        "short_strike",
        "long_strike",
        "net_type",
        "entry_net",
        "entry_gate",
        "width",
        "max_profit",
        "max_loss",
        "confidence_score",
        "score",
        "qty",
    ]
    df[[col for col in exact_cols if col in df.columns]].to_csv(exact_path, index=False)
    lines = [
        "# Weekly Trade Generator",
        "",
        "Replacement credit-premium engine. Candidates are generated directly from dated EOD chains and screener data, priced with conservative bid/ask economics, and exported for exact-spread backtesting.",
        "",
        "## Candidates",
        "",
        markdown_table(
            df,
            [
                "signal_date",
                "ticker",
                "strategy",
                "expiry",
                "entry_net",
                "width",
                "max_profit",
                "max_loss",
                "pop_estimate",
                "confidence",
                "reason",
            ],
        ),
        "",
        "## Diagnostics",
        "",
        "```json",
        json.dumps(diagnostics, indent=2, sort_keys=True),
        "```",
        "",
        f"CSV: {csv_path.name}",
        f"Exact backtest setups: {exact_path.name}",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    manifest_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {exact_path}")
    print(f"Wrote: {md_path}")
    print(f"Wrote: {manifest_path}")


def run_daily(args: argparse.Namespace, cfg: GeneratorConfig) -> None:
    base_dir = Path(args.base_dir).expanduser().resolve()
    label = base_dir.name
    df, diagnostics = generate_for_day(base_dir, cfg)
    write_outputs(df, diagnostics, Path(args.out_dir).expanduser().resolve(), label)


def run_replay(args: argparse.Namespace, cfg: GeneratorConfig) -> None:
    root = Path(args.replay_root).expanduser().resolve()
    start = parse_date(args.start_date)
    end = parse_date(args.end_date)
    out_dir = Path(args.out_dir).expanduser().resolve()
    rows = []
    daily = []
    for day_dir in date_dirs(root, start, end):
        try:
            df, diagnostics = generate_for_day(day_dir, cfg)
        except Exception as exc:
            diagnostics = {"date": day_dir.name, "error": str(exc)}
            df = pd.DataFrame()
        daily.append(diagnostics)
        if not df.empty:
            rows.append(df)
    uncapped = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    combined = allocate_weekly(uncapped, cfg)
    diagnostics = {
        "mode": "replay",
        "root": str(root),
        "start_date": start.isoformat() if start else "",
        "end_date": end.isoformat() if end else "",
        "days": len(daily),
        "pre_weekly_allocation_candidates": int(len(uncapped)),
        "max_candidates_per_week": int(cfg.max_candidates_per_week),
        "total_candidates": int(len(combined)),
        "daily": daily,
        "strategy_counts": combined["strategy"].value_counts().to_dict() if not combined.empty else {},
    }
    label = f"{start.isoformat() if start else 'start'}_{end.isoformat() if end else 'end'}"
    write_outputs(combined, diagnostics, out_dir, label)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replacement weekly options trade generator.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-dir", help="Single dated UW folder, e.g. /path/2026-04-30")
    mode.add_argument("--replay-root", help="Root containing dated UW folders")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-candidates-per-day", type=int, default=GeneratorConfig.max_candidates_per_day)
    parser.add_argument("--max-candidates-per-week", type=int, default=GeneratorConfig.max_candidates_per_week)
    parser.add_argument("--top-underlyings", type=int, default=GeneratorConfig.top_underlyings)
    parser.add_argument("--min-marketcap", type=float, default=GeneratorConfig.min_marketcap)
    parser.add_argument("--min-avg30-volume", type=float, default=GeneratorConfig.min_avg30_volume)
    parser.add_argument("--min-iv-rank", type=float, default=GeneratorConfig.min_iv_rank)
    parser.add_argument("--min-credit-pct-width", type=float, default=GeneratorConfig.min_credit_pct_width)
    parser.add_argument("--max-leg-spread-pct", type=float, default=GeneratorConfig.max_leg_spread_pct)
    parser.add_argument("--no-bear-call-credit", action="store_true")
    parser.add_argument("--no-bull-put-credit", action="store_true")
    parser.add_argument("--include-iron-condor", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    cfg = GeneratorConfig(
        max_candidates_per_day=max(1, int(args.max_candidates_per_day)),
        max_candidates_per_week=max(0, int(args.max_candidates_per_week)),
        top_underlyings=max(1, int(args.top_underlyings)),
        min_marketcap=float(args.min_marketcap),
        min_avg30_volume=float(args.min_avg30_volume),
        min_iv_rank=float(args.min_iv_rank),
        min_credit_pct_width=float(args.min_credit_pct_width),
        max_leg_spread_pct=float(args.max_leg_spread_pct),
        allow_bear_call_credit=not bool(args.no_bear_call_credit),
        allow_bull_put_credit=not bool(args.no_bull_put_credit),
        allow_iron_condor=bool(args.include_iron_condor),
    )
    if args.base_dir:
        run_daily(args, cfg)
    else:
        run_replay(args, cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
