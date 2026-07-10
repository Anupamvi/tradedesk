from __future__ import annotations

import datetime as dt
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .catalysts import earnings_crosses_expiry, earnings_event_date
from .data import safe_float
from .edge_model import EDGE_COLUMNS, apply_replay_edge_model
from .occ import build_occ_symbol, parse_occ_symbol
from .performance import live_outcome_adjustment, performance_min_score, performance_risk_multiplier, setup_family
from .pipeline_versions import PIPELINE_NAME_V2, PIPELINE_VERSION_V2, pipeline_version_record
from .schwab_live import (
    SchwabChainValidator,
    chain_spot,
    chain_to_contracts,
    find_credit_spread_alternatives,
    find_debit_spread_alternatives,
    price_width_bucket,
)


INDEX_SKIP = {"SPX", "SPXW", "NDX", "NDXP", "VIX"}
ETF_SYMBOL_SKIP = {
    "ARKK",
    "BITO",
    "DIA",
    "EEM",
    "EFA",
    "GLD",
    "HYG",
    "IBIT",
    "IWM",
    "KRE",
    "LQD",
    "QQQ",
    "SLV",
    "SMH",
    "SOXX",
    "SPY",
    "TLT",
    "UNG",
    "USO",
    "XBI",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLU",
    "XLV",
    "XLY",
    "XOP",
}
AI_TECH = {"NVDA", "AMD", "AVGO", "MSFT", "GOOGL", "GOOG", "META", "TSM", "QCOM", "MU", "INTC"}
ETF_FALLBACK_SYMBOLS = {"SPY", "QQQ", "IWM"}
CREDIT_DIRECTIONS = {"Bull Put", "Bear Call"}
DEBIT_DIRECTIONS = {"Bull Call", "Bear Put"}
BULLISH_DIRECTIONS = {"Bull Put", "Bull Call"}
BEARISH_DIRECTIONS = {"Bear Call", "Bear Put"}
PIPELINE_NAME = PIPELINE_NAME_V2
PIPELINE_VERSION = PIPELINE_VERSION_V2


def is_etf_row(row: pd.Series) -> bool:
    ticker = str(row.get("ticker") or "").upper().strip()
    issue_type = str(row.get("issue_type") or "").upper().strip()
    name = str(row.get("full_name") or "").upper().strip()
    name_is_etf = bool(
        re.search(r"\bETF\b", name)
        or re.search(r"\bEXCHANGE[- ]TRADED(?: FUND| PRODUCT| NOTE)?\b", name)
    )
    return ticker in ETF_SYMBOL_SKIP or issue_type == "ETF" or name_is_etf


def _earnings_days(row: pd.Series, asof: dt.date) -> float:
    catalyst_days = safe_float(row.get("catalyst_earnings_days"))
    if math.isfinite(catalyst_days):
        return catalyst_days
    catalyst_date = row.get("catalyst_earnings_date")
    if not pd.isna(catalyst_date):
        if isinstance(catalyst_date, dt.datetime):
            catalyst_date = catalyst_date.date()
        if isinstance(catalyst_date, dt.date):
            return float((catalyst_date - asof).days)
        parsed = pd.to_datetime(catalyst_date, errors="coerce")
        if not pd.isna(parsed):
            return float((parsed.date() - asof).days)
    value = row.get("next_earnings_dt")
    if pd.isna(value):
        return math.nan
    if isinstance(value, dt.datetime):
        value = value.date()
    if isinstance(value, dt.date):
        return float((value - asof).days)
    return math.nan


def replay_quality_pattern(
    *,
    direction: str,
    trend: str,
    credit_pct: float,
    distance_pct: float,
    expected_move: float,
) -> tuple[bool, str]:
    """Return whether a spread matches the replay-validated high-quality slice."""
    ratio = distance_pct / expected_move if math.isfinite(distance_pct) and math.isfinite(expected_move) and expected_move > 0 else math.nan
    if (
        math.isfinite(credit_pct)
        and 0.18 <= credit_pct <= 0.30
        and math.isfinite(ratio)
        and ratio >= 0.65
    ):
        return True, "validated_credit18_30_expected_buffer"
    if not math.isfinite(expected_move) or expected_move <= 0:
        return False, "replay_guard_missing_expected_move"
    if math.isfinite(credit_pct) and credit_pct < 0.18:
        return False, "replay_guard_credit_below_validated_band"
    if math.isfinite(credit_pct) and credit_pct > 0.30:
        return False, "replay_guard_credit_above_validated_band"
    if math.isfinite(ratio) and ratio < 0.65:
        return False, "replay_guard_insufficient_expected_move_buffer"
    return False, "replay_guard_no_validated_pattern"


def _next_weekday(day: dt.date) -> dt.date:
    if day.weekday() == 5:
        return day + dt.timedelta(days=2)
    if day.weekday() == 6:
        return day + dt.timedelta(days=1)
    return day


def detect_regime(sc: pd.DataFrame) -> dict[str, Any]:
    by_ticker = sc.set_index("ticker", drop=False)
    spy = by_ticker.loc["SPY"] if "SPY" in by_ticker.index else pd.Series(dtype=object)
    qqq = by_ticker.loc["QQQ"] if "QQQ" in by_ticker.index else pd.Series(dtype=object)
    vix = by_ticker.loc["VIX"] if "VIX" in by_ticker.index else pd.Series(dtype=object)

    spy_close = safe_float(spy.get("close"))
    spy_prev = safe_float(spy.get("prev_close"))
    qqq_close = safe_float(qqq.get("close"))
    qqq_prev = safe_float(qqq.get("prev_close"))
    vix_close = safe_float(vix.get("close"))
    if not math.isfinite(vix_close):
        vix_close = safe_float(spy.get("iv30d")) * 100.0

    spy_chg = (spy_close / spy_prev - 1.0) if math.isfinite(spy_close) and math.isfinite(spy_prev) and spy_prev else 0.0
    qqq_chg = (qqq_close / qqq_prev - 1.0) if math.isfinite(qqq_close) and math.isfinite(qqq_prev) and qqq_prev else 0.0
    flow_bias = safe_float(spy.get("flow_bias"), 0.0) * 0.55 + safe_float(qqq.get("flow_bias"), 0.0) * 0.45

    if vix_close < 18:
        vol = "low"
    elif vix_close < 25:
        vol = "medium"
    else:
        vol = "high"

    if spy_chg > 0.004 and qqq_chg > 0.002:
        trend = "uptrend"
    elif spy_chg < -0.004 and qqq_chg < -0.002:
        trend = "downtrend"
    else:
        trend = "range"

    flow = "strong" if abs(flow_bias) >= 0.04 else "weak"
    transition = bool(abs(spy_chg) >= 0.015 or abs(qqq_chg) >= 0.018 or vix_close >= 22)
    return {
        "volatility": vol,
        "trend": trend,
        "flow": flow,
        "transition": transition,
        "vix_proxy": round(vix_close, 2) if math.isfinite(vix_close) else None,
        "spy_1d": round(spy_chg, 4),
        "qqq_1d": round(qqq_chg, 4),
        "flow_bias": round(flow_bias, 4),
        "sizing_stance": "defensive" if transition or vol == "high" else "normal",
    }


def select_ticker_pool(sc: pd.DataFrame, *, max_tickers: int) -> pd.DataFrame:
    df = sc.copy()
    df = df[~df["ticker"].isin(INDEX_SKIP)]
    df = df[~df.apply(is_etf_row, axis=1)]
    df = df[pd.to_numeric(df["close"], errors="coerce").fillna(0) >= 20]
    df = df[pd.to_numeric(df["flow_total_premium"], errors="coerce").fillna(0) > 5_000_000]
    df["_liq_rank"] = (
        pd.to_numeric(df.get("flow_total_premium"), errors="coerce").fillna(0).clip(upper=2_000_000_000)
        + pd.to_numeric(df.get("total_open_interest"), errors="coerce").fillna(0) * 20.0
        + pd.to_numeric(df.get("avg30_volume"), errors="coerce").fillna(0) * 2.0
    )
    ranked = df.sort_values("_liq_rank", ascending=False)
    if max_tickers and max_tickers > 0:
        ranked = ranked.head(max_tickers)
    return ranked.drop(columns=["_liq_rank"])


def _direction_sign(direction: object) -> int:
    text = str(direction or "")
    if text in BULLISH_DIRECTIONS:
        return 1
    if text in BEARISH_DIRECTIONS:
        return -1
    return 0


def _is_credit_strategy(row: pd.Series | dict[str, Any]) -> bool:
    direction = str(row.get("direction", ""))
    strategy = str(row.get("strategy", ""))
    return direction in CREDIT_DIRECTIONS or "Credit" in strategy


def _is_debit_strategy(row: pd.Series | dict[str, Any]) -> bool:
    direction = str(row.get("direction", ""))
    strategy = str(row.get("strategy", ""))
    return direction in DEBIT_DIRECTIONS or "Debit" in strategy


def _strategy_kind(direction: object) -> str:
    return "Credit" if str(direction or "") in CREDIT_DIRECTIONS else "Debit"


def _strategy_label(direction: object) -> str:
    kind = _strategy_kind(direction)
    return f"{direction} {kind} Spread"


def _direction_list(row: pd.Series, *, include_debit: bool = True) -> list[str]:
    bias = safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), 0.0))
    total = safe_float(row.get("flow_total_premium"), 0.0)
    directions: list[str] = []
    if bias >= 0.025:
        directions.append("Bull Put")
        if include_debit:
            directions.append("Bull Call")
    if bias <= -0.025:
        directions.append("Bear Call")
        if include_debit:
            directions.append("Bear Put")
    if not directions and total >= 150_000_000 and abs(bias) < 0.04:
        directions = ["Bull Put", "Bear Call"]
        if include_debit:
            directions.extend(["Bull Call", "Bear Put"])
    return directions


def select_index_fallback_pool(sc: pd.DataFrame, *, max_tickers: int = 3) -> pd.DataFrame:
    df = sc.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df = df[df["ticker"].isin(ETF_FALLBACK_SYMBOLS)].copy()
    if df.empty:
        return df
    df["_rank"] = (
        pd.to_numeric(df.get("flow_total_premium"), errors="coerce").fillna(0).clip(upper=2_000_000_000)
        + pd.to_numeric(df.get("total_open_interest"), errors="coerce").fillna(0) * 10.0
        + pd.to_numeric(df.get("avg30_volume"), errors="coerce").fillna(0)
    )
    return df.sort_values("_rank", ascending=False).head(max_tickers).drop(columns=["_rank"])


def _contract_side_bias(right: object, ask_volume: float, bid_volume: float, *, threshold: float = 0.58) -> str:
    ask = safe_float(ask_volume, 0.0)
    bid = safe_float(bid_volume, 0.0)
    total = ask + bid
    if total <= 0:
        return "unknown"
    ask_share = ask / total
    bid_share = bid / total
    right_text = str(right or "").upper()[:1]
    if right_text == "C":
        if ask_share >= threshold:
            return "bullish"
        if bid_share >= threshold:
            return "bearish"
    if right_text == "P":
        if bid_share >= threshold:
            return "bullish"
        if ask_share >= threshold:
            return "bearish"
    return "mixed"


def classify_flow_quality(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    """Classify UW flow quality before treating it as directional evidence."""
    data = row if isinstance(row, pd.Series) else pd.Series(row)
    direction = str(data.get("direction") or "")
    expected_sign = _direction_sign(direction)
    bias = safe_float(data.get("combined_flow_bias"), safe_float(data.get("flow_bias"), 0.0))
    bot_total = safe_float(data.get("bot_total_premium"), 0.0)
    total = safe_float(data.get("flow_total_premium"), bot_total)
    multileg_ratio = max(
        safe_float(data.get("bot_multileg_ratio"), 0.0),
        safe_float(data.get("source_multileg_ratio"), 0.0),
        safe_float(data.get("source_stock_multileg_ratio"), 0.0),
    )
    side_bias = str(data.get("source_side_bias") or "unknown")
    bot_volume_oi_ratio = safe_float(data.get("bot_volume_oi_ratio"), math.nan)
    unique_expiries = safe_float(data.get("bot_unique_expiries"), math.nan)
    unique_strikes = safe_float(data.get("bot_unique_strikes"), math.nan)
    directional_premium = safe_float(data.get("bot_bull_premium"), 0.0) if expected_sign >= 0 else safe_float(data.get("bot_bear_premium"), 0.0)
    opposite_premium = safe_float(data.get("bot_bear_premium"), 0.0) if expected_sign >= 0 else safe_float(data.get("bot_bull_premium"), 0.0)
    opposite_ratio = opposite_premium / (directional_premium + opposite_premium) if directional_premium + opposite_premium > 0 else math.nan

    if multileg_ratio >= 0.45:
        return "spread_leg", f"multi-leg context dominates ({multileg_ratio:.0%}); do not read as standalone direction"
    if math.isfinite(opposite_ratio) and 0.35 <= opposite_ratio <= 0.65 and abs(bias) < 0.08:
        return "unclear", "same-ticker opposite-side activity makes flow noisy"
    if math.isfinite(bot_volume_oi_ratio) and bot_volume_oi_ratio >= 0.85 and math.isfinite(unique_expiries) and unique_expiries <= 2:
        return "roll", "volume is large versus OI and concentrated in few expiries; likely roll/position management"
    if expected_sign < 0 and safe_float(data.get("bot_put_ask_premium"), 0.0) > max(1_000_000.0, total * 0.35):
        return "hedge", "put ask premium dominates; bearish flow may be portfolio hedge"
    if expected_sign > 0 and safe_float(data.get("bot_call_ask_premium"), 0.0) > 0 and safe_float(data.get("bot_put_ask_premium"), 0.0) > total * 0.25:
        return "hedge", "bullish call flow appears alongside heavy put demand"
    if expected_sign and bias * expected_sign >= 0.04 and side_bias in {"bullish", "bearish", "mixed", "unknown"}:
        if side_bias in {"bullish", "bearish"} and ((side_bias == "bullish") != (expected_sign > 0)):
            return "unclear", f"contract side bias is {side_bias} but candidate direction is {direction}"
        return "directional", f"premium bias {bias:+.1%} aligns with {direction}"
    if math.isfinite(unique_strikes) and unique_strikes <= 2 and total >= 10_000_000:
        return "roll", "large flow concentrated in repeated strikes"
    return "unclear", "UW flow lacks clean directional confirmation"


def _preferred_width(spot: float) -> float:
    return price_width_bucket(spot)


def _edge_text(direction: str, row: pd.Series, hot: pd.DataFrame) -> str:
    pieces = ["flow"]
    iv_rank = safe_float(row.get("iv_rank"))
    if math.isfinite(iv_rank) and iv_rank >= 25:
        pieces.append("volatility")
    if not hot.empty and hot["volume"].fillna(0).sum() >= 1000:
        pieces.append("liquidity/contract flow")
    return "+".join(pieces)


def generate_candidates(
    sc_pool: pd.DataFrame,
    hot: pd.DataFrame,
    bot_flow: pd.DataFrame,
    *,
    asof: dt.date,
    max_candidates: int,
    index_fallback: bool = False,
) -> pd.DataFrame:
    bot = bot_flow.set_index("ticker", drop=False) if not bot_flow.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in sc_pool.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        if not ticker:
            continue
        combined_bias = safe_float(row.get("flow_bias"), 0.0)
        if not bot.empty and ticker in bot.index:
            combined_bias = combined_bias * 0.65 + safe_float(bot.loc[ticker].get("bot_flow_bias"), 0.0) * 0.35
        ticker_hot = hot[(hot["ticker"] == ticker) & (hot["dte"].between(7, 45, inclusive="both"))].copy()
        if ticker_hot.empty:
            continue
        close = safe_float(row.get("close"))
        if not math.isfinite(close) or close <= 0:
            continue
        row = row.copy()
        row["combined_flow_bias"] = combined_bias
        for direction in _direction_list(row):
            right = "P" if direction in {"Bull Put", "Bear Put"} else "C"
            opt = ticker_hot[ticker_hot["right"].eq(right)].copy()
            if opt.empty:
                continue
            if direction == "Bull Put":
                opt = opt[(opt["strike"] < close) & (((close - opt["strike"]) / close).between(0.015, 0.18))]
            elif direction == "Bear Call":
                opt = opt[(opt["strike"] > close) & (((opt["strike"] - close) / close).between(0.015, 0.18))]
            elif direction == "Bull Call":
                opt = opt[(opt["strike"] >= close * 0.96) & (opt["strike"] <= close * 1.08)]
            else:
                opt = opt[(opt["strike"] <= close * 1.04) & (opt["strike"] >= close * 0.90)]
            if opt.empty:
                continue
            opt["_dte_pref"] = (opt["dte"] - 21).abs()
            opt["_liq"] = opt["premium"].fillna(0) + opt["volume"].fillna(0) * 100 + opt["open_interest"].fillna(0) * 20
            expiry_order = (
                opt.groupby("expiry_dt", as_index=False)
                .agg(dte=("dte", "median"), liq=("_liq", "sum"), contracts=("option_symbol", "count"))
                .sort_values(["liq", "contracts"], ascending=False)
                .head(3)
            )
            for _, exp_row in expiry_order.iterrows():
                expiry = exp_row["expiry_dt"]
                exp_contracts = opt[opt["expiry_dt"].eq(expiry)].copy()
                if exp_contracts.empty:
                    continue
                if direction == "Bull Put":
                    target = close * 0.94
                elif direction == "Bear Call":
                    target = close * 1.06
                elif direction == "Bull Call":
                    target = close * 1.01
                else:
                    target = close * 0.99
                exp_contracts["_target_dist"] = (exp_contracts["strike"] - target).abs()
                source = exp_contracts.sort_values(["_target_dist", "_liq"], ascending=[True, False]).iloc[0]
                width = _preferred_width(close)
                source_strike = safe_float(source.get("strike"))
                if direction == "Bull Put":
                    short_strike = source_strike
                    long_strike = short_strike - width
                    estimated_credit = safe_float(source.get("bid")) * 0.45
                    estimated_debit = math.nan
                    source_role = "short"
                    distance_pct = (close - short_strike) / close if close > 0 else math.nan
                    breakeven_distance_pct = math.nan
                elif direction == "Bear Call":
                    short_strike = source_strike
                    long_strike = short_strike + width
                    estimated_credit = safe_float(source.get("bid")) * 0.45
                    estimated_debit = math.nan
                    source_role = "short"
                    distance_pct = (short_strike - close) / close if close > 0 else math.nan
                    breakeven_distance_pct = math.nan
                elif direction == "Bull Call":
                    long_strike = source_strike
                    short_strike = long_strike + width
                    estimated_credit = math.nan
                    estimated_debit = safe_float(source.get("ask"))
                    source_role = "long"
                    breakeven_distance_pct = ((long_strike + estimated_debit) - close) / close if close > 0 and math.isfinite(estimated_debit) else math.nan
                    distance_pct = abs((long_strike - close) / close) if close > 0 else math.nan
                else:
                    long_strike = source_strike
                    short_strike = long_strike - width
                    estimated_credit = math.nan
                    estimated_debit = safe_float(source.get("ask"))
                    source_role = "long"
                    breakeven_distance_pct = (close - (long_strike - estimated_debit)) / close if close > 0 and math.isfinite(estimated_debit) else math.nan
                    distance_pct = abs((long_strike - close) / close) if close > 0 else math.nan
                short_leg_eod = build_occ_symbol(ticker, expiry, right, short_strike)
                long_leg_eod = build_occ_symbol(ticker, expiry, right, long_strike)
                ask_side_volume = safe_float(source.get("ask_side_volume"), 0.0)
                bid_side_volume = safe_float(source.get("bid_side_volume"), 0.0)
                multileg_volume = safe_float(source.get("multileg_volume"), 0.0)
                stock_multileg_volume = safe_float(source.get("stock_multi_leg_volume"), 0.0)
                source_volume = safe_float(source.get("volume"), 0.0)
                source_side_bias = _contract_side_bias(right, ask_side_volume, bid_side_volume)
                bot_metrics = bot.loc[ticker].to_dict() if not bot.empty and ticker in bot.index else {}
                dte = int((expiry - asof).days) if isinstance(expiry, dt.date) else math.nan
                iv30d = safe_float(row.get("iv30d"))
                realized_volatility_30d = safe_float(row.get("volatility"))
                iv_hv_ratio = (
                    iv30d / realized_volatility_30d
                    if math.isfinite(iv30d) and math.isfinite(realized_volatility_30d) and realized_volatility_30d > 0
                    else math.nan
                )
                expected_move = iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and math.isfinite(dte) and dte > 0 else math.nan
                if direction in CREDIT_DIRECTIONS:
                    expected_ratio = distance_pct / expected_move if math.isfinite(distance_pct) and math.isfinite(expected_move) and expected_move > 0 else math.nan
                    target_entry = round(width * 0.18, 2)
                else:
                    expected_ratio = expected_move / max(breakeven_distance_pct, 0.001) if math.isfinite(breakeven_distance_pct) and math.isfinite(expected_move) and expected_move > 0 else math.nan
                    target_entry = round(width * 0.45, 2)
                candidate = {
                    "ticker": ticker,
                    "sector": row.get("sector", ""),
                    "direction": direction,
                    "strategy": _strategy_label(direction),
                    "strategy_kind": _strategy_kind(direction),
                    "index_fallback": bool(index_fallback),
                    "expiry": expiry,
                    "dte": dte,
                    "stock_price_eod": close,
                    "short_strike_eod": short_strike,
                    "long_strike_eod": long_strike,
                    "preferred_width": width,
                    "estimated_eod_credit": round(estimated_credit, 2) if math.isfinite(estimated_credit) else math.nan,
                    "estimated_eod_debit": round(estimated_debit, 2) if math.isfinite(estimated_debit) else math.nan,
                    "estimated_credit_pct_width": estimated_credit / width if math.isfinite(estimated_credit) and width > 0 else math.nan,
                    "estimated_debit_pct_width": estimated_debit / width if math.isfinite(estimated_debit) and width > 0 else math.nan,
                    "construction_source": "uw_flow_anchor_seed",
                    "construction_reason": "candidate seed built from the closest liquid UW hot-chain contract before Schwab live spread expansion",
                    "anchor_strike": source_strike,
                    "target_entry": target_entry,
                    "expected_move_ratio": expected_ratio,
                    "distance_pct": distance_pct,
                    "breakeven_distance_pct": breakeven_distance_pct,
                    "flow_bias": safe_float(row.get("flow_bias"), 0.0),
                    "bot_flow_bias": safe_float(bot_metrics.get("bot_flow_bias"), math.nan),
                    "combined_flow_bias": combined_bias,
                    "flow_total_premium": safe_float(row.get("flow_total_premium"), 0.0),
                    "iv_rank": safe_float(row.get("iv_rank")),
                    "iv30d": safe_float(row.get("iv30d")),
                    "realized_volatility_30d": realized_volatility_30d,
                    "iv_hv_ratio": iv_hv_ratio,
                    "iv_hv_spread": iv30d - realized_volatility_30d if math.isfinite(iv30d) and math.isfinite(realized_volatility_30d) else math.nan,
                    "implied_move_perc": safe_float(row.get("implied_move_perc")),
                    "next_earnings_dt": row.get("next_earnings_dt"),
                    "edge_type": _edge_text(direction, row, exp_contracts),
                    "source_contract": source.get("option_symbol", ""),
                    "source_contract_role": source_role,
                    "short_leg_eod": short_leg_eod,
                    "long_leg_eod": long_leg_eod,
                    "source_contract_volume": source_volume,
                    "source_contract_oi": safe_float(source.get("open_interest"), 0.0),
                    "source_ask_side_volume": ask_side_volume,
                    "source_bid_side_volume": bid_side_volume,
                    "source_mid_volume": safe_float(source.get("mid_volume"), 0.0),
                    "source_sweep_volume": safe_float(source.get("sweep_volume"), 0.0),
                    "source_cross_volume": safe_float(source.get("cross_volume"), 0.0),
                    "source_multileg_volume": multileg_volume,
                    "source_stock_multileg_volume": stock_multileg_volume,
                    "source_multileg_ratio": multileg_volume / source_volume if source_volume > 0 else 0.0,
                    "source_stock_multileg_ratio": stock_multileg_volume / source_volume if source_volume > 0 else 0.0,
                    "source_side_bias": source_side_bias,
                    **{k: bot_metrics.get(k, math.nan) for k in [
                        "bot_bull_premium",
                        "bot_bear_premium",
                        "bot_total_premium",
                        "bot_call_ask_premium",
                        "bot_call_bid_premium",
                        "bot_put_ask_premium",
                        "bot_put_bid_premium",
                        "bot_multileg_premium",
                        "bot_multileg_ratio",
                        "bot_volume_oi_ratio",
                        "bot_unique_expiries",
                        "bot_unique_strikes",
                        "bot_trades",
                    ]},
                }
                flow_quality, flow_reason = classify_flow_quality(candidate)
                candidate["flow_quality"] = flow_quality
                candidate["flow_quality_reason"] = flow_reason
                rows.append(candidate)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_pre_score"] = (
        df["flow_total_premium"].clip(upper=1_000_000_000) / 1_000_000_000
        + df["combined_flow_bias"].abs().clip(upper=0.20) * 5.0
        + df["source_contract_volume"].clip(upper=50_000) / 50_000
        + df["source_contract_oi"].clip(upper=50_000) / 100_000
    )
    df["candidate_coverage_source"] = ""
    credit = df[df["strategy_kind"].eq("Credit")].copy()
    if not credit.empty:
        direction_sign = credit["direction"].map(lambda value: _direction_sign(value))
        credit["_edge_align"] = credit["combined_flow_bias"] * direction_sign
        credit["_edge_dte_pref"] = (pd.to_numeric(credit["dte"], errors="coerce").fillna(45) - 21).abs()
        credit["_edge_score"] = (
            credit["_edge_align"].clip(lower=0.0, upper=0.30) * 8.0
            + pd.to_numeric(credit["source_contract_volume"], errors="coerce").fillna(0).clip(upper=5000) / 2500.0
            + pd.to_numeric(credit["source_contract_oi"], errors="coerce").fillna(0).clip(upper=10000) / 10000.0
            + pd.to_numeric(credit["flow_total_premium"], errors="coerce").fillna(0).clip(upper=250_000_000) / 250_000_000
            - credit["_edge_dte_pref"] / 30.0
        )
        rescue = (
            credit[credit["_edge_align"] >= 0.10]
            .sort_values("_edge_score", ascending=False)
            .groupby("ticker", as_index=False)
            .head(2)
        )
    else:
        rescue = pd.DataFrame()
    ranked = df.sort_values("_pre_score", ascending=False)
    base = ranked.head(max_candidates) if max_candidates and max_candidates > 0 else ranked
    # Keep at least one constructed setup per selected ticker. Otherwise a name
    # can survive universe selection, have usable chains, and still disappear
    # before scoring just because several tickers generated many same-name variants.
    coverage = df.sort_values("_pre_score", ascending=False).groupby("ticker", as_index=False).head(1).copy()
    coverage["candidate_coverage_source"] = "per_ticker_coverage"
    if not rescue.empty:
        pieces = [base, rescue[df.columns], coverage[df.columns]]
    else:
        pieces = [base, coverage[df.columns]]
    out = pd.concat(pieces, ignore_index=True).drop_duplicates(
        subset=["ticker", "direction", "expiry", "short_strike_eod", "long_strike_eod"],
        keep="first",
    )
    return out.sort_values("_pre_score", ascending=False).drop(columns=["_pre_score"], errors="ignore")


def _score_trade(row: pd.Series, regime: dict[str, Any], asof: dt.date) -> tuple[float, str, list[str], list[str]]:
    hard: list[str] = []
    penalties: list[str] = []
    score = 0.0
    direction = str(row.get("direction", ""))
    bias = safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), 0.0))
    direction_sign = _direction_sign(direction)
    align = bias * direction_sign if direction_sign else 0.0
    total = safe_float(row.get("flow_total_premium"), 0.0)
    score += min(3.0, max(0.0, math.log10(max(total, 1.0)) - 6.5) + max(0.0, align) * 5.0)

    technical = 1.0
    if regime["trend"] == "uptrend" and direction in BULLISH_DIRECTIONS:
        technical += 0.8
    elif regime["trend"] == "downtrend" and direction in BEARISH_DIRECTIONS:
        technical += 0.8
    elif regime["trend"] == "range":
        technical += 0.4
    score += min(2.0, technical)

    iv_rank = safe_float(row.get("iv_rank"))
    credit_pct = safe_float(row.get("credit_pct_width"))
    debit_pct = safe_float(row.get("debit_pct_width"))
    reward_risk = safe_float(row.get("reward_risk"))
    vol_edge = 0.0
    if math.isfinite(iv_rank):
        if _is_debit_strategy(row):
            vol_edge += min(1.0, max(0.0, 75.0 - iv_rank) / 55.0)
        else:
            vol_edge += min(1.0, max(0.0, iv_rank - 15.0) / 45.0)
    if _is_credit_strategy(row) and math.isfinite(credit_pct):
        vol_edge += min(1.0, max(0.0, credit_pct - 0.12) / 0.14)
    elif _is_debit_strategy(row) and math.isfinite(debit_pct):
        vol_edge += min(1.0, max(0.0, 0.50 - debit_pct) / 0.25)
        if math.isfinite(reward_risk):
            vol_edge += min(0.75, max(0.0, reward_risk - 0.8) / 1.2)
    score += min(2.0, vol_edge)

    distance = safe_float(row.get("distance_pct"))
    breakeven_distance = safe_float(row.get("breakeven_distance_pct"))
    dte = safe_float(row.get("dte"))
    expected_move = _expected_move_pct(row)
    if _is_debit_strategy(row) and math.isfinite(breakeven_distance) and math.isfinite(expected_move) and expected_move > 0:
        score += min(2.0, max(0.0, expected_move / max(breakeven_distance, 0.001)))
        trend_confirms = (regime["trend"] == "uptrend" and direction == "Bull Call") or (
            regime["trend"] == "downtrend" and direction == "Bear Put"
        )
        if breakeven_distance > expected_move and not trend_confirms:
            penalties.append("debit_breakeven_outside_expected_move")
            score -= 0.75
    elif math.isfinite(distance) and math.isfinite(expected_move) and expected_move > 0:
        score += min(2.0, max(0.0, distance / max(expected_move, 0.001)))
        if distance < expected_move * 0.55:
            penalties.append("too_close_to_expected_move")
            if direction == "Bull Put":
                penalties.append("replay_guard_bull_put_expected_move")
            score -= 1.0
    elif math.isfinite(distance):
        score += min(2.0, distance / 0.04)

    if _is_credit_strategy(row) and math.isfinite(credit_pct) and credit_pct >= 0.20:
        score += 1.0
    elif _is_credit_strategy(row) and math.isfinite(credit_pct) and credit_pct >= 0.16:
        score += 0.5
    elif _is_debit_strategy(row) and math.isfinite(debit_pct) and debit_pct <= 0.45 and math.isfinite(reward_risk) and reward_risk >= 1.0:
        score += 0.75
    else:
        penalties.append("debit_bad_reward_risk_or_credit_below_min" if _is_debit_strategy(row) else "credit_below_min_16pct_width")
        score -= 0.5

    if _is_credit_strategy(row):
        pattern_pass, pattern = replay_quality_pattern(
            direction=direction,
            trend=str(regime.get("trend", "")),
            credit_pct=credit_pct,
            distance_pct=distance,
            expected_move=expected_move,
        )
        if pattern_pass:
            score += 0.25
        else:
            penalties.append(pattern)
            score -= 1.25
    else:
        if math.isfinite(debit_pct) and debit_pct <= 0.45 and math.isfinite(reward_risk) and reward_risk >= 1.0:
            penalties.append("debit_replay_proxy_requires_confirmation")
        else:
            penalties.append("debit_replay_guard_bad_structure")
            score -= 1.0

    earnings_days = _earnings_days(row, asof)
    if earnings_crosses_expiry(row, asof=asof):
        hard.append(f"earnings_crosses_expiry:{earnings_event_date(row)}")
    elif math.isfinite(earnings_days) and 0 <= earnings_days <= 7 and pd.isna(row.get("expiry")):
        hard.append(f"earnings_within_7d:{int(earnings_days)}")

    liq = min(safe_float(row.get("short_oi"), 0.0) + safe_float(row.get("short_volume"), 0.0), safe_float(row.get("long_oi"), 0.0) + safe_float(row.get("long_volume"), 0.0))
    if liq < 100:
        hard.append("no_usable_liquidity")
    elif liq < 500:
        penalties.append("marginal_liquidity")
        score -= 1.0

    quote_width = safe_float(row.get("quote_width_pct"))
    if math.isfinite(quote_width) and quote_width > 0.65:
        hard.append("bid_ask_too_wide")
    elif math.isfinite(quote_width) and quote_width > 0.35:
        penalties.append("wide_bid_ask")
        score -= 1.0

    if str(row.get("live_status")) != "PASS":
        hard.append(str(row.get("live_status") or "missing_live_data"))
    if align <= 0:
        penalties.append("no_flow_edge_alignment")
    if regime.get("transition"):
        penalties.append("regime_transition")
        score -= 0.5

    confidence = "High" if score >= 7 else "Medium" if score >= 5 else "Reject"
    return round(max(0.0, min(10.0, score)), 2), confidence, hard, penalties


def live_validate_and_score(
    candidates: pd.DataFrame,
    *,
    asof: dt.date,
    out_dir: Path,
    regime: dict[str, Any],
    require_live: bool,
    schwab_snapshot_dir: Path | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    rows: list[dict[str, Any]] = []
    validator = None
    live_error = ""
    if require_live:
        try:
            validator = SchwabChainValidator(out_dir, snapshot_dir=schwab_snapshot_dir)
        except Exception as exc:
            live_error = str(exc)

    from_date = _next_weekday(max(dt.date.today(), asof))
    candidate_expiries = pd.to_datetime(candidates.get("expiry", pd.Series(dtype=object)), errors="coerce")
    max_expiry = candidate_expiries.max()
    if pd.notna(max_expiry):
        to_date = max(asof + dt.timedelta(days=50), max_expiry.date() + dt.timedelta(days=1))
    else:
        to_date = asof + dt.timedelta(days=50)
    default_live_keys = [
        "credit_pct_width",
        "credit",
        "debit_pct_width",
        "debit",
        "spread_width",
        "distance_pct",
        "breakeven_distance_pct",
        "reward_risk",
        "quote_width_pct",
        "short_oi",
        "short_volume",
        "long_oi",
        "long_volume",
        "max_profit",
        "max_loss",
        "breakeven",
    ]
    for _, cand in candidates.iterrows():
        base = cand.to_dict()
        for key in default_live_keys:
            base.setdefault(key, math.nan)
        live_alternatives: list[dict[str, Any]]
        if validator is None:
            live_alternatives = [{"live_status": "live_unavailable", "live_blocker": live_error or "Schwab validator disabled"}]
        else:
            chain = validator.get_chain(str(cand["ticker"]), from_date=from_date, to_date=to_date)
            if not chain:
                live_alternatives = [{"live_status": "chain_error", "live_blocker": validator.errors.get(str(cand["ticker"]), "chain fetch failed")}]
            else:
                spot = chain_spot(chain)
                contracts = chain_to_contracts(chain)
                expected_move = _expected_move_pct(pd.Series(base))
                anchor = safe_float(base.get("anchor_strike"), safe_float(base.get("short_strike_eod")))
                expiry_value = pd.to_datetime(cand.get("expiry"), errors="coerce")
                if pd.isna(expiry_value):
                    live_alternatives = [
                        {"live_status": "missing_expiry_or_right", "live_blocker": "candidate expiry is missing or invalid"}
                    ]
                elif _is_debit_strategy(cand):
                    live_alternatives = find_debit_spread_alternatives(
                        contracts,
                        direction=str(cand["direction"]),
                        expiry=expiry_value.date(),
                        spot=spot,
                        preferred_width=safe_float(cand.get("preferred_width"), math.nan),
                        anchor_strike=anchor,
                        expected_move_pct=expected_move,
                        as_of_date=asof,
                    )
                else:
                    live_alternatives = find_credit_spread_alternatives(
                        contracts,
                        direction=str(cand["direction"]),
                        expiry=expiry_value.date(),
                        spot=spot,
                        preferred_width=safe_float(cand.get("preferred_width"), math.nan),
                        anchor_strike=anchor,
                        expected_move_pct=expected_move,
                        as_of_date=asof,
                    )
                for live in live_alternatives:
                    live["stock_price_live"] = spot
        for live in live_alternatives:
            row = base.copy()
            row.update(live)
            if str(base.get("construction_source") or "") == "fallback_income":
                row["live_construction_source"] = live.get("construction_source", "")
                row["live_construction_reason"] = live.get("construction_reason", "")
                row["construction_source"] = "fallback_income"
                row["construction_reason"] = base.get("construction_reason", "fallback income seed")
                row["target_entry"] = base.get("target_entry", row.get("target_entry", math.nan))
                row["fallback_target_credit"] = base.get("fallback_target_credit", row.get("target_entry", math.nan))
            row.setdefault("construction_source", base.get("construction_source", "uw_flow_anchor_seed"))
            row.setdefault("construction_reason", base.get("construction_reason", "seed candidate"))
            row.setdefault("target_entry", base.get("target_entry", math.nan))
            row["regime_trend"] = regime.get("trend")
            if row.get("live_status") == "PASS":
                if _is_debit_strategy(row):
                    row["max_profit"] = (safe_float(row.get("spread_width")) - safe_float(row.get("debit"))) * 100.0
                    row["max_loss"] = safe_float(row.get("debit")) * 100.0
                else:
                    row["max_profit"] = safe_float(row.get("credit")) * 100.0
                    row["max_loss"] = (safe_float(row.get("spread_width")) - safe_float(row.get("credit"))) * 100.0
                    if str(row.get("direction")) == "Bull Put":
                        row["breakeven"] = safe_float(row.get("short_strike")) - safe_float(row.get("credit"))
                    else:
                        row["breakeven"] = safe_float(row.get("short_strike")) + safe_float(row.get("credit"))
            score, confidence, hard, penalties = _score_trade(pd.Series(row), regime, asof)
            row["score"] = score
            row["confidence"] = confidence
            row["hard_rejects"] = ";".join(hard)
            row["penalties"] = ";".join(penalties)
            expected_move = _expected_move_pct(pd.Series(row))
            if _is_debit_strategy(row):
                debit_pct = safe_float(row.get("debit_pct_width"))
                reward_risk = safe_float(row.get("reward_risk"))
                be_distance = safe_float(row.get("breakeven_distance_pct"))
                if (
                    math.isfinite(debit_pct)
                    and debit_pct <= 0.45
                    and math.isfinite(reward_risk)
                    and reward_risk >= 1.0
                    and math.isfinite(be_distance)
                    and math.isfinite(expected_move)
                    and be_distance <= expected_move
                ):
                    row["replay_pattern"] = "debit_structure_acceptable_proxy"
                    row["replay_ev_verdict"] = "acceptable_proxy"
                else:
                    row["replay_pattern"] = ""
                    row["replay_ev_verdict"] = "unsupported_or_bad_debit_profile"
            else:
                pattern_pass, pattern = replay_quality_pattern(
                    direction=str(row.get("direction", "")),
                    trend=str(regime.get("trend", "")),
                    credit_pct=safe_float(row.get("credit_pct_width")),
                    distance_pct=safe_float(row.get("distance_pct")),
                    expected_move=expected_move,
                )
                if pattern_pass:
                    row["replay_pattern"] = pattern
                    row["replay_ev_verdict"] = "structure_proxy"
                elif _credit_secondary_income_replay_lane(pd.Series(row)):
                    row["replay_pattern"] = "secondary_income_proxy_requires_decision_selection"
                    row["replay_ev_verdict"] = "secondary_income_proxy"
                else:
                    row["replay_pattern"] = ""
                    row["replay_ev_verdict"] = f"negative_or_unsupported:{pattern}"
            rows.append(row)
    if validator is not None:
        validator.save()
    return pd.DataFrame(rows)


def _occ_key(symbol: object) -> str:
    parsed = parse_occ_symbol(symbol)
    return parsed.compact if parsed else ""


def _leg_side_bias(leg: pd.Series | None) -> str:
    if leg is None or leg.empty:
        return "unknown"
    return _contract_side_bias(
        leg.get("right"),
        safe_float(leg.get("prev_ask_volume"), 0.0),
        safe_float(leg.get("prev_bid_volume"), 0.0),
        threshold=0.55,
    )


def _oi_change_value(leg: pd.Series | None) -> float:
    if leg is None or leg.empty:
        return math.nan
    diff = safe_float(leg.get("oi_diff_plain"))
    if math.isfinite(diff):
        return diff
    return safe_float(leg.get("oi_change"))


def _oi_leg_context(leg: pd.Series | None) -> str:
    if leg is None or leg.empty:
        return "no_match"
    parts = []
    for label, key in [
        ("premium", "prev_total_premium"),
        ("volume", "volume"),
        ("multi_leg", "prev_multi_leg_volume"),
        ("stock_multi_leg", "prev_stock_multi_leg_volume"),
    ]:
        value = safe_float(leg.get(key))
        if math.isfinite(value) and value > 0:
            parts.append(f"{label}={value:g}")
    return ", ".join(parts) if parts else "matched"


def apply_oi_carryover(scored: pd.DataFrame, chain_oi: pd.DataFrame | None) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    default_cols = {
        "oi_carryover_status": "unavailable",
        "oi_carryover_reason": "no chain-oi carryover file available",
        "short_leg_oi_change": math.nan,
        "long_leg_oi_change": math.nan,
        "short_leg_side_bias": "unknown",
        "long_leg_side_bias": "unknown",
        "oi_source_file": "",
        "short_leg_oi_context": "",
        "long_leg_oi_context": "",
    }
    if chain_oi is None or chain_oi.empty:
        for col, value in default_cols.items():
            out[col] = value
        return out

    oi = chain_oi.copy()
    oi["_occ_key"] = oi["option_symbol"].map(_occ_key) if "option_symbol" in oi.columns else ""
    by_key = {str(r["_occ_key"]): r for _, r in oi[oi["_occ_key"].astype(bool)].iterrows()}
    source_file = str(chain_oi.attrs.get("source_path", ""))

    for idx, row in out.iterrows():
        short_key = _occ_key(row.get("short_leg", row.get("short_leg_eod")))
        long_key = _occ_key(row.get("long_leg", row.get("long_leg_eod")))
        short = by_key.get(short_key)
        long = by_key.get(long_key)
        short_change = _oi_change_value(short)
        long_change = _oi_change_value(long)
        short_bias = _leg_side_bias(short)
        long_bias = _leg_side_bias(long)
        direction_sign = _direction_sign(row.get("direction"))
        expected_bias = "bullish" if direction_sign > 0 else "bearish" if direction_sign < 0 else "unknown"

        if short is None and long is None:
            status = "no_exact_match"
            reason = "no exact short/long leg match in OI carryover file"
        else:
            support_votes = 0
            contrary_votes = 0
            for bias, change in [(short_bias, short_change), (long_bias, long_change)]:
                if not math.isfinite(change) or change <= 0:
                    continue
                if bias == expected_bias:
                    support_votes += 1
                elif bias in {"bullish", "bearish"}:
                    contrary_votes += 1
            if support_votes and not contrary_votes:
                status = "supportive"
                reason = f"exact-leg OI/side bias supports {expected_bias} direction"
            elif support_votes and contrary_votes:
                status = "mixed"
                reason = "exact-leg OI has both supportive and contrary side bias"
            elif contrary_votes:
                status = "contrary"
                reason = f"exact-leg OI side bias conflicts with {expected_bias} direction"
            else:
                status = "matched_unconfirmed"
                reason = "exact leg matched but OI/side bias is not directionally decisive"

        out.at[idx, "oi_carryover_status"] = status
        out.at[idx, "oi_carryover_reason"] = reason
        out.at[idx, "short_leg_oi_change"] = short_change
        out.at[idx, "long_leg_oi_change"] = long_change
        out.at[idx, "short_leg_side_bias"] = short_bias
        out.at[idx, "long_leg_side_bias"] = long_bias
        out.at[idx, "oi_source_file"] = source_file
        out.at[idx, "short_leg_oi_context"] = _oi_leg_context(short)
        out.at[idx, "long_leg_oi_context"] = _oi_leg_context(long)
    return out


def _append_token(value: object, token: str) -> str:
    parts = [x.strip() for x in str(value or "").split(";") if x.strip() and x.strip().lower() != "nan"]
    if token and token not in parts:
        parts.append(token)
    return ";".join(parts)


def _confidence_from_score(score: float) -> str:
    return "High" if score >= 7 else "Medium" if score >= 5 else "Reject"


def _clean_note(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _money(value: object) -> str:
    number = safe_float(value)
    return f"${number:.2f}" if math.isfinite(number) else "n/a"


def _pct(value: object) -> str:
    number = safe_float(value)
    return f"{number:.1%}" if math.isfinite(number) else "n/a"


def _confidence_icon(confidence: object) -> str:
    text = str(confidence or "")
    if text == "High":
        return "🟢"
    if text == "Medium":
        return "🟡"
    return "🔴"


def _mode_icon(status: object) -> str:
    text = str(status or "").lower()
    if "latest" in text or "pass" in text:
        return "🟢"
    if "stale" in text or "caution" in text or "unknown" in text:
        return "🟡"
    return "🔵"


def _leg_label(symbol: object) -> str:
    parsed = parse_occ_symbol(symbol)
    if parsed is None:
        return str(symbol or "")
    return f"{parsed.root} {parsed.expiry} {parsed.strike:g}{parsed.right}"


def _leg_quote_summary(row: pd.Series, prefix: str) -> str:
    bid = _money(row.get(f"{prefix}_bid"))
    ask = _money(row.get(f"{prefix}_ask"))
    mid = _money(row.get(f"{prefix}_mid"))
    return f"bid {bid} / ask {ask} / mid {mid}"


def _trade_conviction(row: pd.Series) -> str:
    score = safe_float(row.get("score"))
    confidence = str(row.get("confidence") or "")
    if not math.isfinite(score):
        return confidence
    return f"{confidence} ({score:.2f}/10)"


def _edge_summary(row: pd.Series) -> str:
    parts: list[str] = []
    edge = _clean_note(row.get("edge_type"))
    if edge:
        parts.append(edge)
    pattern = _clean_note(row.get("replay_pattern"))
    if pattern:
        parts.append(pattern)
    edge_verdict = _clean_note(row.get("edge_verdict"))
    edge_sample = safe_float(row.get("edge_sample_size"))
    if edge_verdict:
        if math.isfinite(edge_sample):
            parts.append(f"replay edge {edge_verdict} n={int(edge_sample)}")
        else:
            parts.append(f"replay edge {edge_verdict}")
    credit_pct = safe_float(row.get("credit_pct_width"))
    pop = safe_float(row.get("pop_delta_proxy"))
    if math.isfinite(credit_pct):
        parts.append(f"{credit_pct:.1%} credit/width")
    if math.isfinite(pop):
        parts.append(f"{pop:.1%} POP/delta proxy")
    catalyst = _clean_note(row.get("catalyst_status"))
    if catalyst:
        parts.append(f"catalyst={catalyst}")
    return "; ".join(parts)


def _flow_alignment(row: pd.Series) -> float:
    bias = safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), 0.0))
    sign = _direction_sign(row.get("direction"))
    return bias * sign if sign else math.nan


def _expected_move_ratio(row: pd.Series) -> float:
    distance = safe_float(row.get("breakeven_distance_pct")) if _is_debit_strategy(row) else safe_float(row.get("distance_pct"))
    iv30d = safe_float(row.get("iv30d"))
    dte = safe_float(row.get("dte"))
    expected_move = iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and math.isfinite(dte) and dte > 0 else math.nan
    if not math.isfinite(distance) or not math.isfinite(expected_move) or expected_move <= 0:
        return math.nan
    return expected_move / max(distance, 0.001) if _is_debit_strategy(row) else distance / expected_move


def _min_leg_liquidity(row: pd.Series) -> float:
    short_liq = safe_float(row.get("short_oi"), 0.0) + safe_float(row.get("short_volume"), 0.0)
    long_liq = safe_float(row.get("long_oi"), 0.0) + safe_float(row.get("long_volume"), 0.0)
    return min(short_liq, long_liq)


def _decision_sort_score(row: pd.Series) -> float:
    credit_pct = safe_float(row.get("credit_pct_width"))
    ratio = _expected_move_ratio(row)
    align = _flow_alignment(row)
    quote_width = safe_float(row.get("quote_width_pct"))
    score = 0.0
    if math.isfinite(ratio):
        score += min(2.0, max(0.0, ratio))
    if math.isfinite(align):
        score += min(2.0, max(0.0, align) * 6.0)
    if math.isfinite(credit_pct):
        score += min(1.0, max(0.0, credit_pct - 0.16) * 8.0)
    if math.isfinite(quote_width):
        score -= max(0.0, quote_width - 0.35)
    return round(score, 4)


def _secondary_income_eligible(
    *,
    credit_pct: float,
    ratio: float,
    align: float,
    score: float,
    dte: float,
) -> bool:
    return (
        math.isfinite(credit_pct)
        and 0.16 <= credit_pct <= 0.30
        and math.isfinite(ratio)
        and ratio >= 0.20
        and math.isfinite(align)
        and align >= 0.12
        and math.isfinite(score)
        and score >= 1.60
        and math.isfinite(dte)
        and dte <= 35
    )


def validated_addon_income_lane(direction: object, credit_pct: float) -> bool:
    """Replay-validated lane for adding trades beyond the strongest daily setup."""
    return str(direction or "") == "Bear Call" and math.isfinite(credit_pct) and 0.20 <= credit_pct <= 0.24


def _credit_secondary_income_replay_lane(row: pd.Series) -> bool:
    if not _is_credit_strategy(row):
        return False
    return _secondary_income_eligible(
        credit_pct=safe_float(row.get("credit_pct_width")),
        ratio=_expected_move_ratio(row),
        align=_flow_alignment(row),
        score=_decision_sort_score(row),
        dte=safe_float(row.get("dte")),
    )


def apply_high_conviction_decision_marks(scored: pd.DataFrame, *, asof: dt.date | None = None) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    out["decision_eligible"] = False
    out["decision_score"] = math.nan
    out["decision_reason"] = ""
    out["decision_tier"] = ""
    for idx, row in out.iterrows():
        score = _decision_sort_score(row)
        out.at[idx, "decision_score"] = score
        if _clean_note(row.get("hard_rejects")):
            out.at[idx, "decision_reason"] = "decision_hard_reject"
            continue
        penalties = str(row.get("penalties") or "")
        credit_pct = safe_float(row.get("credit_pct_width"))
        ratio = _expected_move_ratio(row)
        align = _flow_alignment(row)
        dte = safe_float(row.get("dte"))
        earnings_days = _earnings_days(row, asof) if asof is not None else math.nan
        if "final_guard_" in penalties:
            out.at[idx, "decision_reason"] = "decision_final_quality_guard"
        elif "news_catalyst_caution" in penalties:
            out.at[idx, "decision_reason"] = "decision_news_catalyst_caution"
        elif "marginal_liquidity" in penalties or "wide_bid_ask" in penalties:
            out.at[idx, "decision_reason"] = "decision_marginal_live_liquidity"
        elif safe_float(row.get("score"), 0.0) < 5.0:
            out.at[idx, "decision_reason"] = "decision_score_below_medium"
        elif earnings_crosses_expiry(row, asof=asof):
            out.at[idx, "decision_reason"] = f"decision_earnings_crosses_expiry:{earnings_event_date(row)}"
        elif math.isfinite(earnings_days) and 0 <= earnings_days <= 10 and pd.isna(row.get("expiry")):
            out.at[idx, "decision_reason"] = f"decision_earnings_within_10d:{int(earnings_days)}"
        elif not math.isfinite(credit_pct) or credit_pct < 0.16:
            out.at[idx, "decision_reason"] = "decision_credit_below_16pct_width"
        elif _secondary_income_eligible(credit_pct=credit_pct, ratio=ratio, align=align, score=score, dte=dte) and ratio < 0.65:
            out.at[idx, "decision_eligible"] = True
            out.at[idx, "decision_reason"] = "decision_secondary_income_eligible"
            out.at[idx, "decision_tier"] = "secondary_income"
        elif not math.isfinite(ratio) or ratio < 0.65:
            out.at[idx, "decision_reason"] = "decision_insufficient_expected_move_buffer"
        elif not math.isfinite(align) or align < 0.10:
            out.at[idx, "decision_reason"] = "decision_weak_flow_alignment"
        elif credit_pct > 0.30:
            out.at[idx, "decision_reason"] = "decision_credit_above_30pct_width"
        else:
            out.at[idx, "decision_eligible"] = True
            out.at[idx, "decision_reason"] = "decision_eligible"
            out.at[idx, "decision_tier"] = "primary"
    return out


def apply_portfolio_context(scored: pd.DataFrame, portfolio: dict[str, Any] | None) -> pd.DataFrame:
    if scored.empty or not portfolio or portfolio.get("status") != "ok":
        return scored
    out = scored.copy()
    option_underlyings = {str(x).upper() for x in portfolio.get("option_underlyings", [])}
    large_equity = {str(k).upper(): safe_float(v) for k, v in (portfolio.get("large_equity_exposure", {}) or {}).items()}
    total_value = safe_float(portfolio.get("total_value"), 0.0)
    for idx, row in out.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        notes: list[str] = []
        if ticker in option_underlyings:
            notes.append("existing option exposure")
            out.at[idx, "portfolio_warning"] = "existing option exposure"
        if ticker in large_equity:
            pct = large_equity[ticker] / total_value if total_value > 0 else 0.0
            hedging = str(row.get("direction")) in BEARISH_DIRECTIONS
            out.at[idx, "portfolio_exposure_pct"] = pct
            out.at[idx, "portfolio_hedging"] = bool(hedging)
            if not hedging:
                notes.append(f"large existing equity exposure {pct:.1%}; execution gate unaffected")
            else:
                notes.append(f"large existing equity exposure {pct:.1%}; trade may hedge portfolio")
            out.at[idx, "penalties"] = _append_token(row.get("penalties"), f"large_existing_equity_exposure:{pct:.1%}")
        if notes:
            out.at[idx, "portfolio_note"] = "; ".join(notes)
    return out


def apply_catalyst_context(scored: pd.DataFrame, catalysts: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or catalysts.empty:
        return scored
    out = scored.merge(catalysts, on="ticker", how="left")
    for idx, row in out.iterrows():
        status = str(row.get("catalyst_status") or "").strip().lower()
        if status == "caution":
            out.at[idx, "penalties"] = _append_token(row.get("penalties"), "news_catalyst_caution")
            score = max(0.0, safe_float(row.get("score"), 0.0) - 0.5)
            out.at[idx, "score"] = round(score, 2)
            out.at[idx, "confidence"] = _confidence_from_score(score)
        elif status == "unknown":
            out.at[idx, "penalties"] = _append_token(row.get("penalties"), "news_unconfirmed")
            score = max(0.0, safe_float(row.get("score"), 0.0) - 0.25)
            out.at[idx, "score"] = round(score, 2)
            out.at[idx, "confidence"] = _confidence_from_score(score)
    return out


def apply_final_quality_guards(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    for idx, row in out.iterrows():
        score = safe_float(row.get("score"), 0.0)
        credit_pct = safe_float(row.get("credit_pct_width"))
        dte = safe_float(row.get("dte"))
        catalyst_status = str(row.get("catalyst_status") or "").strip().lower()
        penalties = str(row.get("penalties") or "")
        if catalyst_status == "caution" and math.isfinite(dte) and dte <= 10:
            penalties = _append_token(penalties, "final_guard_near_term_news_caution")
            score -= 0.75
        if math.isfinite(credit_pct) and credit_pct < 0.18 and catalyst_status in {"caution", "unknown"}:
            penalties = _append_token(penalties, "final_guard_low_credit_without_news_support")
            score -= 0.5
        if math.isfinite(credit_pct) and credit_pct < 0.18 and score < 6.0:
            penalties = _append_token(penalties, "final_guard_low_credit_medium_score")
            score -= 0.25
        out.at[idx, "penalties"] = penalties
        out.at[idx, "score"] = round(max(0.0, score), 2)
        out.at[idx, "confidence"] = _confidence_from_score(safe_float(out.at[idx, "score"]))
    return out


def _expected_move_pct(row: pd.Series) -> float:
    implied = safe_float(row.get("implied_move_perc"))
    if math.isfinite(implied) and implied > 0:
        return implied
    iv30d = safe_float(row.get("iv30d"))
    dte = safe_float(row.get("dte"))
    return iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and math.isfinite(dte) and dte > 0 else math.nan


def _credit_required_entry(row: pd.Series) -> float:
    if str(row.get("construction_source") or "") == "fallback_income":
        target = safe_float(row.get("fallback_target_credit"), safe_float(row.get("target_entry")))
        if math.isfinite(target) and target > 0:
            return round(target, 2)
    width = safe_float(row.get("spread_width"))
    if not math.isfinite(width) or width <= 0:
        return math.nan
    return round(width * 0.18, 2)


def _debit_required_entry(row: pd.Series) -> float:
    width = safe_float(row.get("spread_width"))
    if not math.isfinite(width) or width <= 0:
        return math.nan
    return round(width * 0.45, 2)


def _token_set(value: object) -> set[str]:
    return {x.strip() for x in str(value or "").split(";") if x.strip() and x.strip().lower() != "nan"}


def _execute_core_blockers(row: pd.Series, *, allow_proxy_ev: bool = False) -> list[str]:
    """Non-price thesis gates that cannot be averaged away by a high score."""
    blockers: list[str] = []
    penalties = _token_set(row.get("penalties"))
    failed = _token_set(row.get("confirmations_failed"))
    flow_quality = str(row.get("flow_quality") or "unclear")
    replay_verdict = str(row.get("replay_ev_verdict") or "")
    edge_verdict = str(row.get("edge_verdict") or replay_verdict)
    edge_sample_size = safe_float(row.get("edge_sample_size"), safe_float(row.get("historical_sample_size"), math.nan))
    oi_status = str(row.get("oi_carryover_status") or "")
    catalyst_status = str(row.get("catalyst_status") or "").strip().lower()
    decision_reason = str(row.get("decision_reason") or "")
    confidence = str(row.get("confidence") or "")
    score = safe_float(row.get("score"), 0.0)
    decision_eligible = str(row.get("decision_eligible")).lower() == "true"
    decision_tier = str(row.get("decision_tier") or "")
    confirmation_score = safe_float(row.get("confirmation_score"), 0.0)
    secondary_credit = (
        _is_credit_strategy(row)
        and replay_verdict == "acceptable_secondary_income"
        and decision_eligible
        and decision_tier == "secondary_income"
    )
    edge_avg_pnl = safe_float(row.get("edge_avg_pnl"), math.nan)
    edge_win_rate = safe_float(row.get("edge_win_rate"), math.nan)
    independent_confirmed_flow = (
        flow_quality in {"unclear", "spread_leg"}
        and edge_verdict in {"positive", "acceptable"}
        and confirmation_score >= 8.0
        and oi_status in {"supportive", "matched_unconfirmed", ""}
        and "price_action_trend" not in failed
        and _flow_alignment(row) >= 0.04
        and str(row.get("live_status")) == "PASS"
        and safe_float(row.get("quote_width_pct"), 1.0) <= 0.35
    )

    if (
        catalyst_status == "caution"
        or "news_catalyst_caution" in penalties
        or "final_guard_near_term_news_caution" in penalties
    ):
        blockers.append("news_catalyst_caution")
    if catalyst_status == "unknown" or "news_unconfirmed" in penalties:
        blockers.append("news_unconfirmed")
    if any(token.startswith("negative_live_expectancy:") for token in penalties):
        blockers.append("negative_live_expectancy")
    if _is_credit_strategy(row) and math.isfinite(edge_avg_pnl) and edge_avg_pnl <= 0:
        blockers.append("negative_edge_avg_pnl")
    if _is_credit_strategy(row) and replay_verdict == "acceptable_secondary_income":
        if math.isfinite(edge_sample_size) and edge_sample_size < 7.0:
            blockers.append(f"secondary_income_thin_sample:n={int(edge_sample_size)}")
        if math.isfinite(edge_win_rate) and edge_win_rate < 0.58:
            blockers.append(f"secondary_income_low_win_rate:{edge_win_rate:.0%}")
    if any(token.startswith("recent_loss_family:") for token in penalties):
        blockers.append("recent_loss_family")
    if math.isfinite(edge_sample_size) and edge_sample_size < 7.0 and replay_verdict not in {"acceptable_secondary_income"}:
        blockers.append(f"thin_replay_sample:n={int(edge_sample_size)}")
    if "earnings_news_risk" in failed:
        blockers.append("earnings_news_risk")
    if any(token.startswith("final_guard_") for token in penalties):
        blockers.append("final_quality_guard")
    if decision_reason in {
        "decision_final_quality_guard",
        "decision_news_catalyst_caution",
        "decision_marginal_live_liquidity",
        "decision_score_below_medium",
    }:
        blockers.append(decision_reason)
    if "regime_transition" in penalties and (confidence != "High" or score < 7.5):
        blockers.append("regime_transition_defensive")
    if flow_quality in {"hedge", "roll"}:
        blockers.append(f"flow_not_directional:{flow_quality}")
    elif flow_quality != "directional" and not (
        secondary_credit and flow_quality in {"unclear", "spread_leg"} and _flow_alignment(row) >= 0.12
    ) and not independent_confirmed_flow:
        blockers.append(f"flow_not_directional:{flow_quality}")
    if "no_flow_edge_alignment" in penalties:
        blockers.append("no_flow_edge_alignment")
    if "price_action_trend" in failed:
        blockers.append("price_action_trend")
    if "market_regime_alignment" in failed:
        blockers.append("market_regime_alignment")
    if oi_status == "contrary":
        blockers.append("oi_carryover_contrary")
    if _is_debit_strategy(row):
        if replay_verdict == "acceptable_proxy" and not allow_proxy_ev:
            blockers.append("debit_proxy_ev_only")
        elif replay_verdict not in {"acceptable", "positive"} and not (allow_proxy_ev and replay_verdict == "acceptable_proxy"):
            blockers.append(f"debit_ev_not_supported:{replay_verdict or 'missing'}")
    elif replay_verdict not in {"acceptable", "positive", "acceptable_secondary_income"}:
        blockers.append(f"credit_ev_not_supported:{replay_verdict or 'missing'}")
    elif replay_verdict == "acceptable_secondary_income" and not secondary_credit:
        blockers.append("secondary_income_not_decision_selected")
    if "liquidity_quote_quality" in failed:
        blockers.append("liquidity_quote_quality")
    return list(dict.fromkeys(blockers))


def _tactical_debit_execute_ok(row: pd.Series, core_blockers: list[str]) -> bool:
    """Small defined-risk debit lane for transition regimes; never promotes noisy credit income."""
    if not _is_debit_strategy(row):
        return False
    blocker_set = set(core_blockers)
    if blocker_set - {"regime_transition_defensive"}:
        return False
    penalties = _token_set(row.get("penalties"))
    failed = _token_set(row.get("confirmations_failed"))
    if penalties & {
        "news_unconfirmed",
        "news_catalyst_caution",
        "wide_bid_ask",
        "marginal_liquidity",
        "debit_bad_reward_risk_or_credit_below_min",
        "debit_replay_guard_bad_structure",
    }:
        return False
    if any(token.startswith("final_guard_") for token in penalties):
        return False
    if failed & {
        "earnings_news_risk",
        "expected_move_buffer",
        "level_or_gex_protection",
        "historical_ev_replay",
        "iv_premium_quality",
        "liquidity_quote_quality",
        "price_action_trend",
    }:
        return False
    debit = safe_float(row.get("debit"))
    required_debit = _debit_required_entry(row)
    return (
        str(row.get("live_status")) == "PASS"
        and str(row.get("confidence") or "") == "High"
        and safe_float(row.get("score"), 0.0) >= 7.0
        and safe_float(row.get("confirmation_score"), 0.0) >= 9.0
        and str(row.get("edge_verdict") or row.get("replay_ev_verdict") or "") in {"positive", "acceptable"}
        and safe_float(row.get("edge_sample_size"), 0.0) >= 7.0
        and str(row.get("flow_quality") or "") not in {"hedge", "roll"}
        and str(row.get("oi_carryover_status") or "") != "contrary"
        and str(row.get("catalyst_status") or "").strip().lower() != "caution"
        and math.isfinite(debit)
        and math.isfinite(required_debit)
        and debit <= required_debit
        and safe_float(row.get("debit_pct_width"), math.nan) <= 0.40
        and safe_float(row.get("reward_risk"), math.nan) >= 1.50
        and safe_float(row.get("expected_move_ratio"), math.nan) >= 1.05
        and safe_float(row.get("quote_width_pct"), math.nan) <= 0.08
        and safe_float(row.get("max_loss"), math.inf) <= 250.0
    )


def _manual_confirmation_scout_ok(row: pd.Series, core_blockers: list[str], *, edge_watch_ok: bool) -> tuple[bool, list[str], list[str]]:
    """Surface a trade as Watch when only human-verifiable thesis checks are missing.

    This intentionally does not promote to Execute. It prevents the daily report from
    collapsing to "no trade" when live pricing, liquidity, replay edge, and risk are
    good but the remaining blocker is a missing ticker news/OI/ambiguous-flow check.
    """
    soft_allowed = {"news_unconfirmed", "flow_not_directional:unclear", "flow_not_directional:spread_leg"}
    soft: list[str] = []
    hard: list[str] = []
    edge_verdict = str(row.get("edge_verdict") or row.get("replay_ev_verdict") or "")
    flow_quality = str(row.get("flow_quality") or "")
    confirmation_score = safe_float(row.get("confirmation_score"), 0.0)
    edge_sample = safe_float(row.get("edge_sample_size"), 0.0)
    for blocker in core_blockers:
        if blocker in soft_allowed:
            soft.append(blocker)
        elif (
            blocker == "oi_carryover_contrary"
            and flow_quality == "directional"
            and edge_verdict in {"positive", "acceptable"}
            and edge_sample >= 10.0
            and confirmation_score >= 8.0
        ):
            soft.append(blocker)
        else:
            hard.append(blocker)
    if hard or not soft:
        return False, soft, hard
    if not edge_watch_ok:
        return False, soft, hard
    if str(row.get("live_status")) != "PASS":
        return False, soft, hard
    if str(row.get("confidence") or "") != "High":
        return False, soft, hard
    if safe_float(row.get("score"), 0.0) < 7.0 or confirmation_score < 8.0:
        return False, soft, hard
    if _min_leg_liquidity(row) < 500.0:
        return False, soft, hard
    quote_width = safe_float(row.get("quote_width_pct"))
    if not math.isfinite(quote_width) or quote_width > 0.20:
        return False, soft, hard
    max_loss = safe_float(row.get("max_loss"))
    if not math.isfinite(max_loss) or max_loss > 450.0:
        return False, soft, hard
    if _is_credit_strategy(row):
        credit = safe_float(row.get("credit"))
        required = _credit_required_entry(row)
        credit_pct = safe_float(row.get("credit_pct_width"))
        return (
            math.isfinite(credit)
            and math.isfinite(required)
            and credit >= required
            and math.isfinite(credit_pct)
            and 0.18 <= credit_pct <= 0.30
        ), soft, hard
    debit = safe_float(row.get("debit"))
    required = _debit_required_entry(row)
    reward_risk = safe_float(row.get("reward_risk"))
    return (
        math.isfinite(debit)
        and math.isfinite(required)
        and debit <= required
        and math.isfinite(reward_risk)
        and reward_risk >= 1.0
    ), soft, hard


def apply_confirmation_framework(
    scored: pd.DataFrame,
    *,
    asof: dt.date,
    regime: dict[str, Any],
    recent_performance: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    perf_status = (recent_performance or {}).get("status", "unavailable")
    perf_stance = (recent_performance or {}).get("stance", "")
    for idx, row in out.iterrows():
        row = out.loc[idx]
        expected_move = _expected_move_pct(row)
        distance = safe_float(row.get("distance_pct"))
        breakeven_distance = safe_float(row.get("breakeven_distance_pct"))
        if _is_debit_strategy(row):
            expected_ratio = expected_move / max(breakeven_distance, 0.001) if math.isfinite(expected_move) and math.isfinite(breakeven_distance) else math.nan
        else:
            expected_ratio = distance / expected_move if math.isfinite(distance) and math.isfinite(expected_move) and expected_move > 0 else math.nan
        level_protection = "not_available"
        if _is_credit_strategy(row):
            if math.isfinite(expected_ratio) and expected_ratio >= 1.0:
                level_protection = "short strike outside expected move"
            elif math.isfinite(expected_ratio) and expected_ratio >= 0.65:
                level_protection = "borderline expected move buffer"
            elif math.isfinite(expected_ratio):
                level_protection = "inside expected move without known GEX/level protection"
        elif math.isfinite(expected_ratio) and expected_ratio >= 1.0:
            level_protection = "breakeven reachable within expected move"
        elif math.isfinite(expected_ratio):
            level_protection = "breakeven beyond expected move"

        checks: dict[str, bool | None] = {}
        trend = str(regime.get("trend") or "")
        direction = str(row.get("direction") or "")
        catalyst_status = str(row.get("catalyst_status") or "").lower()
        checks["price_action_trend"] = (
            trend == "range"
            or (trend == "uptrend" and direction in BULLISH_DIRECTIONS)
            or (trend == "downtrend" and direction in BEARISH_DIRECTIONS)
        )
        edge_sample_size = safe_float(row.get("edge_sample_size"), safe_float(row.get("historical_sample_size"), math.nan))
        if _is_debit_strategy(row) and direction in BULLISH_DIRECTIONS and str(regime.get("flow") or "") == "weak":
            checks["market_regime_alignment"] = (
                trend == "uptrend"
                and catalyst_status == "supportive"
                and math.isfinite(edge_sample_size)
                and edge_sample_size >= 10.0
            )
        oi_status = str(row.get("oi_carryover_status") or "")
        checks["oi_carryover"] = True if oi_status == "supportive" else False if oi_status == "contrary" else None
        iv_rank = safe_float(row.get("iv_rank"))
        if _is_debit_strategy(row):
            checks["iv_premium_quality"] = (
                safe_float(row.get("debit_pct_width")) <= 0.45
                and safe_float(row.get("reward_risk")) >= 1.0
                and (not math.isfinite(iv_rank) or iv_rank <= 75)
            )
        else:
            checks["iv_premium_quality"] = safe_float(row.get("credit_pct_width")) >= 0.18
        earnings_days = _earnings_days(row, asof)
        checks["earnings_news_risk"] = (
            not earnings_crosses_expiry(row, asof=asof)
            and not (math.isfinite(earnings_days) and 0 <= earnings_days <= 7 and pd.isna(row.get("expiry")))
            and catalyst_status not in {"caution", "unknown"}
        )
        if _is_debit_strategy(row):
            checks["expected_move_buffer"] = math.isfinite(expected_ratio) and expected_ratio >= 1.0
        else:
            checks["expected_move_buffer"] = math.isfinite(expected_ratio) and expected_ratio >= 0.65
        checks["level_or_gex_protection"] = None if level_protection == "not_available" else "without known" not in level_protection and "beyond" not in level_protection
        replay_verdict = str(row.get("replay_ev_verdict") or "")
        edge_verdict = str(row.get("edge_verdict") or "")
        if (
            replay_verdict == "secondary_income_proxy"
            and str(row.get("decision_eligible")).lower() == "true"
            and str(row.get("decision_tier") or "") == "secondary_income"
            and edge_verdict in {"positive", "acceptable"}
        ):
            replay_verdict = "acceptable_secondary_income"
            out.at[idx, "replay_ev_verdict"] = replay_verdict
            out.at[idx, "replay_pattern"] = "validated_secondary_income_sleeve"
        replay_ok = edge_verdict in {"positive", "acceptable"} or replay_verdict in {"acceptable", "acceptable_secondary_income", "positive"}
        if not replay_verdict and _clean_note(row.get("replay_pattern")):
            replay_ok = True
            replay_verdict = "acceptable"
        checks["historical_ev_replay"] = replay_ok
        checks["schwab_live_pricing"] = str(row.get("live_status")) == "PASS"
        short_liq = safe_float(row.get("short_oi"), 0.0) + safe_float(row.get("short_volume"), 0.0)
        long_liq = safe_float(row.get("long_oi"), 0.0) + safe_float(row.get("long_volume"), 0.0)
        checks["liquidity_quote_quality"] = min(short_liq, long_liq) >= 500 and safe_float(row.get("quote_width_pct"), 0.0) <= 0.35
        checks["portfolio_risk_budget_fit"] = True

        passed = [name for name, ok in checks.items() if ok is True]
        failed = [name for name, ok in checks.items() if ok is False]
        neutral = [name for name, ok in checks.items() if ok is None]
        denominator = max(1, len(checks) - len(neutral))
        confirmation_score = round(10.0 * len(passed) / denominator, 2)
        primary_blocker = ""
        if _clean_note(row.get("hard_rejects")):
            primary_blocker = str(row.get("hard_rejects")).split(";")[0]
        elif failed:
            primary_blocker = failed[0]
        elif _clean_note(row.get("decision_reason")) and str(row.get("decision_eligible")).lower() != "true":
            primary_blocker = str(row.get("decision_reason"))

        out.at[idx, "expected_move_pct"] = expected_move
        out.at[idx, "expected_move_ratio"] = expected_ratio
        out.at[idx, "level_protection"] = level_protection
        out.at[idx, "confirmations_passed"] = ";".join(passed)
        out.at[idx, "confirmations_failed"] = ";".join(failed)
        out.at[idx, "confirmation_score"] = confirmation_score
        out.at[idx, "primary_blocker"] = primary_blocker
        out.at[idx, "historical_sample_size"] = safe_float((recent_performance or {}).get("window"), math.nan)
        out.at[idx, "historical_win_rate"] = safe_float((recent_performance or {}).get("win_rate"), math.nan)
        out.at[idx, "historical_avg_pl"] = safe_float((recent_performance or {}).get("avg_pnl_1x"), math.nan)
        out.at[idx, "historical_profit_factor"] = safe_float((recent_performance or {}).get("profit_factor"), math.nan)
        out.at[idx, "historical_max_adverse_excursion"] = safe_float((recent_performance or {}).get("max_adverse_excursion"), math.nan)
        if perf_status == "ok":
            out.at[idx, "replay_ev_verdict"] = replay_verdict or ("acceptable" if perf_stance != "degrading" else "degrading")
        elif not replay_verdict:
            out.at[idx, "replay_ev_verdict"] = "unsupported_thin_sample"
    return out


def assign_trade_statuses(
    scored: pd.DataFrame,
    *,
    single_name_execute_quality_poor: bool | None = None,
    index_income_mode: str = "fallback",
) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    for text_col in ["trade_status", "trade_tier", "trade_status_reason", "price_annotation", "what_must_improve"]:
        if text_col in out.columns:
            out[text_col] = out[text_col].fillna("").astype(object)
        else:
            out[text_col] = ""
    provisional_statuses: list[str] = []
    for idx, row in out.iterrows():
        hard = _clean_note(row.get("hard_rejects"))
        penalties = str(row.get("penalties") or "")
        flow_quality = str(row.get("flow_quality") or "unclear")
        confirmation_score = safe_float(row.get("confirmation_score"), 0.0)
        replay_verdict = str(row.get("replay_ev_verdict") or "")
        credit = safe_float(row.get("credit"))
        debit = safe_float(row.get("debit"))
        width = safe_float(row.get("spread_width"))
        credit_pct = safe_float(row.get("credit_pct_width"))
        debit_pct = safe_float(row.get("debit_pct_width"))
        reward_risk = safe_float(row.get("reward_risk"))
        expected_ratio = safe_float(row.get("expected_move_ratio"))
        quote_width = safe_float(row.get("quote_width_pct"))
        required_credit = _credit_required_entry(row)
        required_debit = _debit_required_entry(row)
        no_chase = required_credit if _is_credit_strategy(row) else required_debit
        price_annotation = ""
        credit_price_miss = False
        debit_price_miss = False
        if _is_credit_strategy(row) and math.isfinite(credit) and math.isfinite(required_credit):
            if credit < required_credit:
                miss_pct = (required_credit - credit) / required_credit if required_credit > 0 else math.nan
                suffix = f" ({miss_pct:.0%} below target)" if math.isfinite(miss_pct) else ""
                price_annotation = f"current credit ${credit:.2f} is below target ${required_credit:.2f}{suffix}; show as work-limit, do not drop, and do not hit natural"
                credit_price_miss = True
            else:
                price_annotation = f"current credit ${credit:.2f} is at or above target ${required_credit:.2f}"
        if _is_debit_strategy(row) and math.isfinite(debit) and math.isfinite(required_debit):
            if debit > required_debit:
                miss_pct = (debit - required_debit) / required_debit if required_debit > 0 else math.nan
                suffix = f" ({miss_pct:.0%} above target)" if math.isfinite(miss_pct) else ""
                price_annotation = f"current debit ${debit:.2f} is above target ${required_debit:.2f}{suffix}; show as work-limit, do not drop, and do not chase"
                debit_price_miss = True
            else:
                price_annotation = f"current debit ${debit:.2f} is at or below target ${required_debit:.2f}"
        status = "Research"
        tier = ""
        reason = ""
        core_blockers = _execute_core_blockers(row)
        edge_verdict = str(row.get("edge_verdict") or "")
        edge_avg_pnl = safe_float(row.get("edge_avg_pnl"), math.nan)
        edge_nonnegative = not (math.isfinite(edge_avg_pnl) and edge_avg_pnl <= 0)
        edge_watch_ok = (
            edge_nonnegative
            and (
                replay_verdict in {"acceptable", "positive", "acceptable_secondary_income"}
                or edge_verdict in {"acceptable", "positive"}
                or (
                    edge_verdict == "thin_sample"
                    and safe_float(row.get("edge_avg_pnl"), math.nan) > 0
                    and confirmation_score >= 6.0
                )
            )
        )
        watch_blockers = [
            blocker
            for blocker in core_blockers
            if not (
                blocker in {"credit_ev_not_supported:thin_sample", "debit_ev_not_supported:thin_sample"}
                or (blocker.startswith("thin_replay_sample:") and edge_watch_ok)
                or (blocker.startswith("flow_not_directional:") and flow_quality in {"unclear", "spread_leg"} and edge_watch_ok)
            )
        ]
        quote_near_miss = "wide_bid_ask" in penalties or (math.isfinite(quote_width) and 0.35 < quote_width <= 0.65)
        scout_ok, scout_soft, _scout_hard = _manual_confirmation_scout_ok(
            row,
            core_blockers,
            edge_watch_ok=edge_watch_ok,
        )

        if hard:
            status = "Avoid"
            reason = hard
        elif "bid_ask_too_wide" in penalties or (math.isfinite(quote_width) and quote_width > 0.65):
            status = "Avoid"
            reason = "unusable liquidity / too-wide quote"
        elif replay_verdict.startswith("negative"):
            status = "Avoid"
            reason = replay_verdict
        elif _is_credit_strategy(row):
            if (
                not core_blockers
                and
                math.isfinite(credit_pct)
                and credit_pct >= 0.25
                and confirmation_score >= 7.0
                and replay_verdict in {"acceptable", "positive"}
                and (math.isfinite(expected_ratio) and expected_ratio >= 0.65)
            ):
                status = "Execute"
                tier = "Execute A"
                reason = "A-tier credit/width with supportive confirmations"
            elif (
                not core_blockers
                and
                math.isfinite(credit_pct)
                and 0.18 <= credit_pct < 0.25
                and confirmation_score >= 7.0
                and str(row.get("oi_carryover_status")) == "supportive"
                and replay_verdict in {"acceptable", "positive"}
            ):
                status = "Execute"
                tier = "Execute B"
                reason = "B-tier credit with OI, price action, and replay support"
            elif (
                not core_blockers
                and replay_verdict == "acceptable_secondary_income"
                and str(row.get("decision_eligible")).lower() == "true"
                and str(row.get("decision_tier") or "") == "secondary_income"
                and math.isfinite(credit_pct)
                and 0.16 <= credit_pct <= 0.30
                and confirmation_score >= 6.0
            ):
                status = "Execute"
                tier = "Execute Secondary"
                reason = "replay-validated secondary income sleeve with live pricing and risk gates"
            elif (
                not watch_blockers
                and edge_watch_ok
                and math.isfinite(credit)
                and math.isfinite(required_credit)
                and required_credit * 0.80 <= credit < required_credit
            ):
                status = "Watch"
                tier = "near-trigger"
                reason = f"work 1-lot at ${required_credit:.2f} credit only; no chase below ${required_credit:.2f}"
            elif not watch_blockers and edge_watch_ok and quote_near_miss and math.isfinite(credit) and math.isfinite(required_credit) and credit >= required_credit * 0.80:
                status = "Watch"
                tier = "quote-cleanup"
                reason = "entry price is available, but quote quality needs fresh Schwab recheck before order entry"
            elif (
                not watch_blockers
                and edge_watch_ok
                and credit_price_miss
                and math.isfinite(required_credit)
                and confirmation_score >= 5.0
            ):
                status = "Watch"
                tier = "work-limit-price-target"
                reason = f"work-limit trade: target ${required_credit:.2f} credit; current credit ${credit:.2f} is below target, so show the ticket but do not enter below target"
                out.at[idx, "what_must_improve"] = f"credit must improve to ${required_credit:.2f} or better"
            elif scout_ok:
                scout_price = max(credit, required_credit) if math.isfinite(credit) and math.isfinite(required_credit) else required_credit
                status = "Watch"
                tier = "manual-confirmation-scout"
                reason = "manual-confirmation scout: live price/risk pass, but confirm before entry: " + ";".join(scout_soft)
                out.at[idx, "what_must_improve"] = "manual check must clear: " + ";".join(scout_soft)
                out.at[idx, "trigger"] = (
                    f"SCOUT ONLY: recheck ticker news, OI/flow context, and Schwab chain; then work 1-lot at "
                    f"${scout_price:.2f} credit or better. No size-up."
                )
            elif core_blockers:
                status = "Research"
                reason = ";".join(core_blockers)
            elif flow_quality == "directional" or confirmation_score >= 5.0:
                status = "Research"
                reason = "interesting setup but confirmations or pricing are insufficient"
            else:
                status = "Avoid"
                reason = "noisy flow with no confirmation"
        else:
            breakeven_ok = math.isfinite(expected_ratio) and expected_ratio >= 1.0
            iv_ok = safe_float(row.get("iv_rank"), 0.0) <= 75 if math.isfinite(safe_float(row.get("iv_rank"))) else True
            if _tactical_debit_execute_ok(row, core_blockers):
                status = "Execute"
                tier = "Execute Tactical"
                reason = "tactical defined-risk debit: positive replay, tight quote, expected-move room, and 1-lot max-risk cap"
            elif (
                not core_blockers
                and
                math.isfinite(debit_pct)
                and debit_pct <= 0.45
                and math.isfinite(reward_risk)
                and reward_risk >= 1.0
                and breakeven_ok
                and iv_ok
                and confirmation_score >= 7.5
                and flow_quality == "directional"
                and replay_verdict in {"acceptable", "positive"}
            ):
                status = "Execute"
                tier = "Execute B"
                reason = "debit spread passes debit, reward/risk, IV, breakeven, and confirmation gates"
            elif (
                not watch_blockers
                and edge_watch_ok
                and math.isfinite(debit)
                and math.isfinite(required_debit)
                and required_debit < debit <= required_debit * 1.35
                and confirmation_score >= 5.0
            ):
                status = "Watch"
                tier = "near-trigger"
                reason = f"work 1-lot at ${required_debit:.2f} debit or better; no chase above ${required_debit:.2f}"
            elif not watch_blockers and edge_watch_ok and quote_near_miss and math.isfinite(debit) and math.isfinite(required_debit) and debit <= required_debit:
                status = "Watch"
                tier = "quote-cleanup"
                reason = "entry price is available, but quote quality needs fresh Schwab recheck before order entry"
            elif (
                not watch_blockers
                and edge_watch_ok
                and debit_price_miss
                and math.isfinite(required_debit)
                and confirmation_score >= 5.0
            ):
                status = "Watch"
                tier = "work-limit-price-target"
                reason = f"work-limit trade: target ${required_debit:.2f} debit; current debit ${debit:.2f} is above target, so show the ticket but do not chase above target"
                out.at[idx, "what_must_improve"] = f"debit must fall to ${required_debit:.2f} or better"
            elif scout_ok:
                scout_price = min(debit, required_debit) if math.isfinite(debit) and math.isfinite(required_debit) else required_debit
                status = "Watch"
                tier = "manual-confirmation-scout"
                reason = "manual-confirmation scout: live price/risk pass, but confirm before entry: " + ";".join(scout_soft)
                out.at[idx, "what_must_improve"] = "manual check must clear: " + ";".join(scout_soft)
                out.at[idx, "trigger"] = (
                    f"SCOUT ONLY: recheck ticker news, OI/flow context, and Schwab chain; then work 1-lot at "
                    f"${scout_price:.2f} debit or better. No size-up."
                )
            elif (
                not watch_blockers
                and edge_watch_ok
                and math.isfinite(debit)
                and math.isfinite(required_debit)
                and debit > required_debit * 1.35
                and confirmation_score >= 5.0
            ):
                status = "Watch"
                tier = "debit-price-annotation"
                reason = f"debit replay/thesis is promising but current debit ${debit:.2f} is above target ${required_debit:.2f}; keep visible, no chase"
            elif core_blockers:
                status = "Research"
                reason = ";".join(core_blockers)
            elif flow_quality == "directional" or confirmation_score >= 5.0:
                status = "Research"
                reason = price_annotation or "debit lane needs stronger EV/replay, breakeven, IV, or price confirmation"
            else:
                status = "Avoid"
                reason = "no realistic debit edge"

        out.at[idx, "trade_status"] = status
        out.at[idx, "trade_tier"] = tier
        out.at[idx, "trade_status_reason"] = reason
        if core_blockers and status in {"Research", "Avoid"}:
            out.at[idx, "primary_blocker"] = core_blockers[0]
        out.at[idx, "required_entry"] = required_credit if _is_credit_strategy(row) else required_debit
        out.at[idx, "no_chase_threshold"] = no_chase
        if price_annotation:
            out.at[idx, "price_annotation"] = price_annotation
            if not _clean_note(out.at[idx, "what_must_improve"] if "what_must_improve" in out.columns else "") and status in {"Watch", "Research"}:
                if credit_price_miss:
                    out.at[idx, "what_must_improve"] = f"credit must improve to ${required_credit:.2f} or better"
                elif debit_price_miss:
                    out.at[idx, "what_must_improve"] = f"debit must fall to ${required_debit:.2f} or better"
                elif math.isfinite(debit) and math.isfinite(required_debit):
                    out.at[idx, "what_must_improve"] = "non-price confirmations must improve; debit is already within target"
                elif math.isfinite(credit) and math.isfinite(required_credit):
                    out.at[idx, "what_must_improve"] = "non-price confirmations must improve; credit is already within target"
                else:
                    out.at[idx, "what_must_improve"] = "fresh Schwab price recheck required"
        if status == "Execute":
            out.at[idx, "primary_blocker"] = ""
        elif status == "Watch" and not _clean_note(row.get("primary_blocker")):
            out.at[idx, "primary_blocker"] = tier or "watch_trigger_not_met"
        provisional_statuses.append(status)

    index_mode = str(index_income_mode or "fallback").strip().lower()
    if single_name_execute_quality_poor is None:
        single_name_execute_quality_poor = not (
            (out["trade_status"].eq("Execute"))
            & (~out.get("index_fallback", pd.Series(False, index=out.index)).astype(bool))
        ).any()
    if "index_fallback" in out.columns and index_mode == "disabled":
        mask = out["index_fallback"].astype(bool) & out["trade_status"].eq("Execute")
        out.loc[mask, "trade_status"] = "Research"
        out.loc[mask, "trade_status_reason"] = "ETF/index income disabled by risk mandate"
        out.loc[mask, "trade_tier"] = "ETF/index disabled"
        out.loc[mask, "portfolio_size_cap"] = 1
    elif "index_fallback" in out.columns and index_mode == "fallback" and not single_name_execute_quality_poor:
        mask = out["index_fallback"].astype(bool) & out["trade_status"].eq("Execute")
        out.loc[mask, "trade_status"] = "Research"
        out.loc[mask, "trade_status_reason"] = "ETF/index fallback disabled because single-name Execute quality exists"
        out.loc[mask, "trade_tier"] = "ETF/index fallback standby"
    elif "index_fallback" in out.columns:
        mask = out["index_fallback"].astype(bool) & out["trade_status"].eq("Execute")
        tier_label = "ETF/index primary income" if index_mode == "primary" else "ETF/index fallback"
        out.loc[mask, "trade_tier"] = out.loc[mask, "trade_tier"].replace("", tier_label)
        if index_mode != "primary":
            out.loc[mask, "portfolio_size_cap"] = 1
    return out


def select_final_trades(
    scored: pd.DataFrame,
    *,
    regime: dict[str, Any],
    risk_budget: float,
    recent_performance: dict[str, Any] | None = None,
    max_final_trades: int = 8,
    risk_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    risk_config = risk_config or {}
    if risk_config.get("allow_new_trades") is False:
        return scored.iloc[0:0].copy()
    required = {"hard_rejects", "score", "penalties"}
    if not required.issubset(scored.columns):
        return scored.iloc[0:0].copy()
    min_score = performance_min_score(recent_performance, 5.0)
    relaxed_min_score = performance_min_score(recent_performance, 4.5)
    if "trade_status" in scored.columns:
        approved = scored[scored["trade_status"].astype(str).eq("Execute") & scored["hard_rejects"].fillna("").eq("")].copy()
        if approved.empty:
            return approved
        tier_bonus = approved.get("trade_tier", pd.Series("", index=approved.index)).fillna("").map(
            {
                "Execute A": 100.0,
                "Execute A+": 100.0,
                "Execute B": 50.0,
                "Execute Tactical": 35.0,
                "Execute Fallback Income": 25.0,
                "ETF/index fallback": 20.0,
            }
        ).fillna(0.0)
        approved["_rank"] = (
            tier_bonus
            + pd.to_numeric(approved.get("confirmation_score", 0), errors="coerce").fillna(0)
            + pd.to_numeric(approved.get("score", 0), errors="coerce").fillna(0) * 0.05
        )
        target_rank_rows = []
        for _, rank_row in approved.iterrows():
            target_profit = _target_profit_per_contract(rank_row)
            expected_value, _source = _expected_value_per_contract(rank_row)
            max_loss = safe_float(rank_row.get("max_loss"))
            ev_per_risk = expected_value / max_loss if math.isfinite(expected_value) and math.isfinite(max_loss) and max_loss > 0 else math.nan
            target_rank_rows.append(
                (max(0.0, safe_float(target_profit, 0.0)) / 100.0)
                + (max(0.0, safe_float(ev_per_risk, 0.0)) * 10.0)
            )
        approved["_rank"] = approved["_rank"] + pd.Series(target_rank_rows, index=approved.index)
    elif "decision_eligible" in scored.columns:
        approved = scored[scored["decision_eligible"].astype(str).str.lower().eq("true")].copy()
        if "decision_score" in approved.columns:
            tier_bonus = approved.get("decision_tier", pd.Series("", index=approved.index)).fillna("").map(
                {"primary": 100.0, "secondary_income": 0.0}
            ).fillna(0.0)
            approved["_rank"] = tier_bonus + approved["decision_score"].fillna(0) + approved["score"].fillna(0) * 0.05
        else:
            approved["_rank"] = approved["score"].fillna(0)
    else:
        approved = scored[scored["hard_rejects"].fillna("").eq("")].copy()
        approved = approved[pd.to_numeric(approved["credit_pct_width"], errors="coerce").fillna(0) >= 0.16].copy()
        approved = approved[~approved["penalties"].fillna("").astype(str).str.contains("replay_guard_", regex=False)].copy()
        approved = approved[~approved["penalties"].fillna("").astype(str).str.contains("final_guard_", regex=False)].copy()
        approved = approved[approved["score"] >= min_score].copy()
        if len(approved) < 3:
            relaxed = scored[
                scored["hard_rejects"].fillna("").eq("")
                & (scored["score"] >= relaxed_min_score)
                & (pd.to_numeric(scored["credit_pct_width"], errors="coerce").fillna(0) >= 0.16)
                & (~scored["penalties"].fillna("").astype(str).str.contains("replay_guard_", regex=False))
                & (~scored["penalties"].fillna("").astype(str).str.contains("final_guard_", regex=False))
            ].copy()
            approved = relaxed
        approved["_rank"] = approved["score"] + approved["credit_pct_width"].fillna(0) * 2.0 + approved["distance_pct"].fillna(0)
    if approved.empty:
        return approved
    approved = approved.sort_values("_rank", ascending=False)
    selected = []
    total_risk = 0.0
    ticker_risk = Counter()
    sector_risk = Counter()
    ai_risk = 0.0
    perf_mult = performance_risk_multiplier(recent_performance)
    risk_mandate = str(risk_config.get("risk_mandate") or "capital-preservation").strip().lower()
    if risk_mandate == "aggressive-intraday":
        trade_fraction = 0.85 if regime.get("sizing_stance") == "normal" else 0.50
        default_ticker_fraction = 0.70
        default_sector_fraction = 0.90
        default_factor_fraction = 0.90
    elif risk_mandate == "target-growth":
        trade_fraction = 0.65 if regime.get("sizing_stance") == "normal" else 0.35
        default_ticker_fraction = 0.55
        default_sector_fraction = 0.80
        default_factor_fraction = 0.80
    elif risk_mandate == "balanced":
        trade_fraction = 0.35 if regime.get("sizing_stance") == "normal" else 0.22
        default_ticker_fraction = 0.40
        default_sector_fraction = 0.65
        default_factor_fraction = 0.65
    else:
        trade_fraction = 0.22 if regime.get("sizing_stance") == "normal" else 0.14
        default_ticker_fraction = 0.30
        default_sector_fraction = 0.55
        default_factor_fraction = 0.55
    max_trade_risk = risk_budget * trade_fraction * perf_mult
    explicit_max_trade = safe_float(risk_config.get("max_risk_per_trade"))
    if math.isfinite(explicit_max_trade) and explicit_max_trade > 0:
        max_trade_risk = min(max_trade_risk, explicit_max_trade)
    max_daily_risk = safe_float(risk_config.get("max_risk_per_day"), risk_budget)
    if not math.isfinite(max_daily_risk) or max_daily_risk <= 0:
        max_daily_risk = risk_budget
    max_ticker_risk = safe_float(risk_config.get("max_open_risk_by_ticker"), risk_budget * default_ticker_fraction)
    if not math.isfinite(max_ticker_risk) or max_ticker_risk <= 0:
        max_ticker_risk = risk_budget * default_ticker_fraction
    max_sector_risk = safe_float(risk_config.get("max_correlated_sector_exposure"), risk_budget * default_sector_fraction)
    if not math.isfinite(max_sector_risk) or max_sector_risk <= 0:
        max_sector_risk = risk_budget * default_sector_fraction
    max_contracts = safe_float(risk_config.get("max_contracts_per_trade"))
    min_ev_per_risk = safe_float(risk_config.get("minimum_expected_value_per_dollar_risk"), 0.0)
    index_income_mode = str(risk_config.get("index_income_mode") or "fallback").strip().lower()
    for _, row in approved.iterrows():
        is_addon = bool(selected) and validated_addon_income_lane(row.get("direction"), safe_float(row.get("credit_pct_width")))
        if "trade_status" not in approved.columns and selected and not is_addon:
            continue
        max_loss = safe_float(row.get("max_loss"))
        if not math.isfinite(max_loss) or max_loss <= 0:
            continue
        confidence = str(row.get("confidence"))
        trade_budget = max_trade_risk if confidence == "High" else max_trade_risk * 0.55
        target_profit_per_contract = _target_profit_per_contract(row)
        expected_value_per_contract, ev_source = _expected_value_per_contract(row)
        ev_per_dollar_risk = (
            expected_value_per_contract / max_loss
            if math.isfinite(expected_value_per_contract) and math.isfinite(max_loss) and max_loss > 0
            else math.nan
        )
        can_size_from_expectancy = (
            confidence == "High"
            and math.isfinite(ev_per_dollar_risk)
            and ev_per_dollar_risk >= max(0.0, min_ev_per_risk)
            and str(row.get("live_status")) == "PASS"
            and safe_float(row.get("edge_sample_size"), 0.0) >= 7.0
        )
        contracts = max(1, int(trade_budget // max_loss)) if can_size_from_expectancy else 1
        liquidity_cap = _liquidity_capacity_contracts(row)
        contracts = min(contracts, liquidity_cap)
        if math.isfinite(max_contracts) and max_contracts > 0:
            contracts = min(contracts, int(max_contracts))
        portfolio_cap = safe_float(row.get("portfolio_size_cap"))
        if math.isfinite(portfolio_cap) and portfolio_cap > 0:
            contracts = min(contracts, int(portfolio_cap))
        if (
            str(row.get("trade_tier")) in {"Execute Secondary", "Execute Tactical"}
            or str(row.get("trade_tier")) == "Execute Fallback Income"
            or str(row.get("alpha_tier") or "").strip() == "Tier 2"
            or (bool(row.get("index_fallback", False)) and index_income_mode != "primary")
        ):
            contracts = min(contracts, 1)
        ticker = str(row.get("ticker"))
        sector = str(row.get("sector") or "Unknown")
        remaining_contract_risk = min(
            max_daily_risk - total_risk,
            max_ticker_risk - ticker_risk[ticker],
            max_sector_risk - sector_risk[sector],
        )
        if ticker in AI_TECH:
            remaining_contract_risk = min(remaining_contract_risk, risk_budget * default_factor_fraction - ai_risk)
        contracts = min(contracts, int(remaining_contract_risk // max_loss))
        if contracts < 1:
            continue
        risk = contracts * max_loss
        out = row.copy()
        out["contracts"] = contracts
        out["sell_leg"] = row.get("short_leg")
        out["buy_leg"] = row.get("long_leg")
        out["expiration_date"] = row.get("expiry")
        out["trade_conviction"] = _trade_conviction(row)
        out["edge_summary"] = _edge_summary(row)
        out["selection_role"] = "validated add-on income lane" if is_addon else "strongest high-conviction setup"
        out["target_profit_per_contract"] = round(target_profit_per_contract, 2) if math.isfinite(target_profit_per_contract) else math.nan
        out["target_profit_total"] = round(target_profit_per_contract * contracts, 2) if math.isfinite(target_profit_per_contract) else math.nan
        out["expected_value_per_contract"] = round(expected_value_per_contract, 2) if math.isfinite(expected_value_per_contract) else math.nan
        out["expected_value_total"] = round(expected_value_per_contract * contracts, 2) if math.isfinite(expected_value_per_contract) else math.nan
        out["expected_value_source"] = ev_source
        out["ev_per_dollar_risk"] = round(ev_per_dollar_risk, 4) if math.isfinite(ev_per_dollar_risk) else math.nan
        out["liquidity_capacity_contracts"] = liquidity_cap
        out["position_max_loss"] = round(risk, 2)
        monthly_target = safe_float(risk_config.get("monthly_profit_target"), 0.0)
        out["target_contribution_pct"] = (
            round((target_profit_per_contract * contracts) / monthly_target, 4)
            if math.isfinite(target_profit_per_contract) and monthly_target > 0
            else math.nan
        )
        if contracts > 1:
            out["sizing_label"] = f"🟣 SIZE-UP: {contracts}-lot"
            out["sizing_rationale"] = (
                f"{risk_mandate} mandate; High confidence with positive expectancy ({ev_source}); ${max_loss:,.0f} max loss per contract fits "
                f"${trade_budget:,.0f} trade budget; liquidity cap {liquidity_cap}; ticker, sector, factor, and total daily risk caps still pass."
            )
        else:
            out["sizing_label"] = "1-lot base"
            if str(row.get("alpha_tier") or "").strip() == "Tier 2":
                out["sizing_rationale"] = (
                    "Kept to 1-lot because Tier 2 liquidity-shift setups require live outcome proof before size-up."
                )
            else:
                out["sizing_rationale"] = (
                    "Kept to 1-lot because confidence, live expectancy, risk budget, sample size, or liquidity did not justify a size-up."
                )
        out["position_size"] = f"{contracts} contract{'s' if contracts != 1 else ''}; max risk ${risk:,.0f}"
        credit = safe_float(row.get("credit"))
        debit = safe_float(row.get("debit"))
        if _is_credit_strategy(row) and math.isfinite(credit) and credit > 0:
            take_profit_debit = credit * 0.40
            stop_debit = credit * 2.00
            out["entry_action"] = "SELL TO OPEN credit spread"
            out["entry_limit_credit"] = round(credit, 2)
            out["entry_limit_debit"] = math.nan
            out["current_credit_debit"] = round(credit, 2)
            out["target_close_debit"] = round(take_profit_debit, 2)
            out["stop_review_debit"] = round(stop_debit, 2)
            out["sell_leg_action"] = "SELL TO OPEN"
            out["buy_leg_action"] = "BUY TO OPEN"
            out["close_action"] = "BUY TO CLOSE spread"
            out["take_profit"] = f"buy back near ${take_profit_debit:.2f} debit"
            out["stop_loss"] = f"review/exit near ${stop_debit:.2f} debit"
            out["exit_plan"] = (
                f"Take 60% profit near ${take_profit_debit:.2f}; stop/review near ${stop_debit:.2f}; "
                "exit if short strike is threatened; avoid expiration-week gamma unless revalidated."
            )
        elif _is_debit_strategy(row) and math.isfinite(debit) and debit > 0:
            target_credit = debit * 1.60
            stop_debit = debit * 0.50
            out["entry_action"] = "BUY TO OPEN debit spread"
            out["entry_limit_credit"] = math.nan
            out["entry_limit_debit"] = round(debit, 2)
            out["current_credit_debit"] = round(debit, 2)
            out["target_close_credit"] = round(target_credit, 2)
            out["stop_review_debit"] = round(stop_debit, 2)
            out["sell_leg_action"] = "SELL TO OPEN"
            out["buy_leg_action"] = "BUY TO OPEN"
            out["close_action"] = "SELL TO CLOSE spread"
            out["take_profit"] = f"sell near ${target_credit:.2f} credit"
            out["stop_loss"] = f"review/exit near ${stop_debit:.2f} debit value"
            out["exit_plan"] = (
                f"Take profit near ${target_credit:.2f} spread value; stop/review if spread value falls near ${stop_debit:.2f}; "
                "exit if direction thesis or breakeven reachability fails."
            )
        else:
            out["entry_action"] = ""
            out["entry_limit_credit"] = math.nan
            out["entry_limit_debit"] = math.nan
            out["current_credit_debit"] = math.nan
            out["target_close_debit"] = math.nan
            out["stop_review_debit"] = math.nan
            out["sell_leg_action"] = "SELL TO OPEN"
            out["buy_leg_action"] = "BUY TO OPEN"
            out["close_action"] = "BUY TO CLOSE spread"
            out["take_profit"] = ""
            out["stop_loss"] = ""
            out["exit_plan"] = "Exit if live pricing invalidates the setup or short strike is threatened."
        notes = []
        penalties_note = _clean_note(row.get("penalties"))
        if _is_debit_strategy(row) and str(row.get("edge_verdict") or row.get("replay_ev_verdict") or "") in {"positive", "acceptable"}:
            penalty_tokens = [
                token
                for token in _token_set(penalties_note)
                if token != "debit_replay_proxy_requires_confirmation"
            ]
            penalties_note = ";".join(penalty_tokens)
        for value in [penalties_note, row.get("portfolio_note"), row.get("catalyst_note")]:
            note = _clean_note(value)
            if note:
                notes.append(note)
        if is_addon:
            notes.append("validated add-on lane: bear call with 20-24% credit/width")
        if regime.get("sizing_stance") == "defensive":
            notes.append("defensive regime sizing")
        if perf_mult < 1.0:
            notes.append("recent performance defensive sizing")
        out["risk_notes"] = "; ".join(notes) if notes else "standard defined-risk spread"
        selected.append(out)
        total_risk += risk
        ticker_risk[ticker] += risk
        sector_risk[sector] += risk
        if ticker in AI_TECH:
            ai_risk += risk
        if max_final_trades and max_final_trades > 0 and len(selected) >= int(max_final_trades):
            break
    if not selected:
        return pd.DataFrame()
    final = pd.DataFrame(selected).drop(columns=["_rank"], errors="ignore")
    final.insert(0, "rank", range(1, len(final) + 1))
    return final


def _entry_price(row: pd.Series) -> float:
    if _is_debit_strategy(row):
        return safe_float(row.get("entry_limit_debit"), safe_float(row.get("debit")))
    return safe_float(row.get("entry_limit_credit"), safe_float(row.get("credit")))


def _recommended_limit(row: pd.Series) -> float:
    for key in ["required_entry", "entry_limit_credit", "entry_limit_debit", "credit", "debit"]:
        value = safe_float(row.get(key))
        if math.isfinite(value) and value > 0:
            return value
    return math.nan


def _live_mid_natural(row: pd.Series) -> tuple[float, float]:
    if _is_debit_strategy(row):
        return safe_float(row.get("mid_debit")), safe_float(row.get("natural_debit"))
    return safe_float(row.get("mid_credit")), safe_float(row.get("natural_credit"))


def _write_execute_outcome_ledger(out_dir: Path, asof: dt.date, final: pd.DataFrame) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "asof",
        "trade_key",
        "report_date",
        "lane",
        "ticker",
        "strategy",
        "setup_family",
        "direction",
        "expiry",
        "sell_leg",
        "buy_leg",
        "entry_action",
        "entry_price",
        "recommended_limit",
        "schwab_live_mid",
        "schwab_live_natural",
        "contracts",
        "max_profit",
        "max_loss",
        "breakeven",
        "score",
        "confidence",
        "trade_tier",
        "edge_verdict",
        "edge_sample_size",
        "edge_win_rate",
        "edge_avg_pnl",
        "outcome_status",
        "actual_fill",
        "exit_date",
        "exit_fill",
        "exit_value",
        "max_adverse_excursion",
        "max_favorable_excursion",
        "realized_pnl",
        "thesis_worked",
        "reason_for_win_loss",
        "outcome_note",
        "report_path",
    ]
    rows: list[dict[str, Any]] = []
    report_path = out_dir / f"codexuw_trade_report_{asof}.md"
    for _, row in final.iterrows():
        entry = _entry_price(row)
        live_mid, live_natural = _live_mid_natural(row)
        trade_key = "|".join(
            [
                out_dir.name,
                str(asof),
                str(row.get("ticker") or ""),
                str(row.get("strategy") or ""),
                str(row.get("expiry") or ""),
                str(row.get("sell_leg") or row.get("short_leg") or ""),
                str(row.get("buy_leg") or row.get("long_leg") or ""),
            ]
        )
        rows.append(
            {
                "run_id": out_dir.name,
                "asof": str(asof),
                "trade_key": trade_key,
                "report_date": str(asof),
                "lane": "Execute Now",
                "ticker": row.get("ticker"),
                "strategy": row.get("strategy"),
                "setup_family": setup_family(row.get("strategy"), row.get("direction")),
                "direction": row.get("direction"),
                "expiry": row.get("expiry"),
                "sell_leg": row.get("sell_leg", row.get("short_leg")),
                "buy_leg": row.get("buy_leg", row.get("long_leg")),
                "entry_action": row.get("entry_action"),
                "entry_price": round(entry, 2) if math.isfinite(entry) else math.nan,
                "recommended_limit": round(_recommended_limit(row), 2) if math.isfinite(_recommended_limit(row)) else math.nan,
                "schwab_live_mid": round(live_mid, 2) if math.isfinite(live_mid) else math.nan,
                "schwab_live_natural": round(live_natural, 2) if math.isfinite(live_natural) else math.nan,
                "contracts": int(safe_float(row.get("contracts"), 1.0)),
                "max_profit": row.get("max_profit"),
                "max_loss": row.get("max_loss"),
                "breakeven": row.get("breakeven"),
                "score": row.get("score"),
                "confidence": row.get("confidence"),
                "trade_tier": row.get("trade_tier"),
                "edge_verdict": row.get("edge_verdict"),
                "edge_sample_size": row.get("edge_sample_size"),
                "edge_win_rate": row.get("edge_win_rate"),
                "edge_avg_pnl": row.get("edge_avg_pnl"),
                "outcome_status": "OPEN_REVIEW_REQUIRED",
                "actual_fill": math.nan,
                "exit_date": "",
                "exit_fill": math.nan,
                "exit_value": math.nan,
                "max_adverse_excursion": math.nan,
                "max_favorable_excursion": math.nan,
                "realized_pnl": math.nan,
                "thesis_worked": "",
                "reason_for_win_loss": "",
                "outcome_note": "Populate exit fields after close or replay mark; do not count as win/loss until realized.",
                "report_path": str(report_path),
            }
        )
    ledger = pd.DataFrame(rows, columns=columns)
    run_path = out_dir / f"codexuw_execute_outcome_ledger_{asof}.csv"
    ledger.to_csv(run_path, index=False)

    central_path = out_dir.parent / "codexuw_execute_outcome_ledger.csv"
    if not ledger.empty:
        if central_path.exists():
            existing = pd.read_csv(central_path)
            combined = pd.concat([existing, ledger], ignore_index=True)
            combined = combined.drop_duplicates(subset=["trade_key"], keep="last")
        else:
            combined = ledger
        combined.to_csv(central_path, index=False)
    elif not central_path.exists():
        pd.DataFrame(columns=columns).to_csv(central_path, index=False)
    return run_path


def _write_recommendation_outcome_ledger(out_dir: Path, asof: dt.date, final: pd.DataFrame, watch: pd.DataFrame) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "report_date",
        "trade_key",
        "ticker",
        "strategy",
        "setup_family",
        "lane",
        "recommended_limit",
        "schwab_live_mid",
        "schwab_live_natural",
        "actual_fill",
        "exit_fill",
        "max_adverse_excursion",
        "max_favorable_excursion",
        "realized_pnl",
        "thesis_worked",
        "reason_for_win_loss",
        "outcome_status",
        "report_path",
    ]
    report_path = out_dir / f"codexuw_trade_report_{asof}.md"
    rows: list[dict[str, Any]] = []

    def add_rows(frame: pd.DataFrame, lane: str) -> None:
        if frame.empty:
            return
        for _, row in frame.iterrows():
            live_mid, live_natural = _live_mid_natural(row)
            key = "|".join(
                [
                    out_dir.name,
                    str(asof),
                    lane,
                    str(row.get("ticker") or ""),
                    str(row.get("strategy") or ""),
                    str(row.get("expiry") or row.get("expiration_date") or ""),
                    str(row.get("sell_leg") or row.get("short_leg") or ""),
                    str(row.get("buy_leg") or row.get("long_leg") or ""),
                ]
            )
            rows.append(
                {
                    "run_id": out_dir.name,
                    "report_date": str(asof),
                    "trade_key": key,
                    "ticker": row.get("ticker"),
                    "strategy": row.get("strategy"),
                    "setup_family": setup_family(row.get("strategy"), row.get("direction")),
                    "lane": lane,
                    "recommended_limit": round(_recommended_limit(row), 2) if math.isfinite(_recommended_limit(row)) else math.nan,
                    "schwab_live_mid": round(live_mid, 2) if math.isfinite(live_mid) else math.nan,
                    "schwab_live_natural": round(live_natural, 2) if math.isfinite(live_natural) else math.nan,
                    "actual_fill": math.nan,
                    "exit_fill": math.nan,
                    "max_adverse_excursion": math.nan,
                    "max_favorable_excursion": math.nan,
                    "realized_pnl": math.nan,
                    "thesis_worked": "",
                    "reason_for_win_loss": "",
                    "outcome_status": "OPEN_REVIEW_REQUIRED" if lane == "Execute Now" else "CONDITIONAL_NOT_FILLED",
                    "report_path": str(report_path),
                }
            )

    add_rows(final, "Execute Now")
    add_rows(watch, "Enter Only At Price")
    ledger = pd.DataFrame(rows, columns=columns)
    run_path = out_dir / f"codexuw_recommendation_outcome_ledger_{asof}.csv"
    ledger.to_csv(run_path, index=False)
    central_path = out_dir.parent / "codexuw_recommendation_outcome_ledger.csv"
    if central_path.exists():
        try:
            existing = pd.read_csv(central_path)
            combined = pd.concat([existing, ledger], ignore_index=True)
            combined = combined.drop_duplicates(subset=["trade_key"], keep="last")
        except Exception:
            combined = ledger
    else:
        combined = ledger
    combined.to_csv(central_path, index=False)
    return run_path


def _entry_credit_target(row: pd.Series) -> float:
    explicit = safe_float(
        row.get("target_credit"),
        safe_float(row.get("min_credit"), safe_float(row.get("entry_limit_credit"), math.nan)),
    )
    if math.isfinite(explicit) and explicit > 0:
        return explicit
    width = safe_float(row.get("spread_width"))
    if math.isfinite(width) and width > 0:
        return round(width * 0.18, 2)
    return math.nan


def _entry_debit_target(row: pd.Series) -> float:
    for key in ["target_debit", "max_debit", "entry_limit_debit"]:
        value = safe_float(row.get(key))
        if math.isfinite(value) and value > 0:
            return value
    width = safe_float(row.get("spread_width"))
    debit_pct = safe_float(row.get("target_debit_pct_width"), safe_float(row.get("max_debit_pct_width"), math.nan))
    if math.isfinite(width) and width > 0 and math.isfinite(debit_pct) and debit_pct > 0:
        return round(width * debit_pct, 2)
    return math.nan


def _current_debit(row: pd.Series) -> float:
    for key in ["debit", "net_debit", "entry_debit", "mid_debit"]:
        value = safe_float(row.get(key))
        if math.isfinite(value) and value > 0:
            return value
    return math.nan


def build_entry_watchlist(scored: pd.DataFrame, *, top_n: int = 12) -> pd.DataFrame:
    """Surface no-order candidates that become actionable only after entry pricing improves."""
    if scored.empty:
        return pd.DataFrame()
    if "trade_status" in scored.columns:
        watch_rows = scored[scored["trade_status"].astype(str).eq("Watch")].copy()
        if watch_rows.empty:
            return pd.DataFrame()
        rows = []
        for _, row in watch_rows.iterrows():
            is_credit = _is_credit_strategy(row)
            is_scout = "scout" in str(row.get("trade_tier") or "").lower()
            existing_trigger = _clean_note(row.get("trigger"))
            existing_improvement = _clean_note(row.get("what_must_improve"))
            required = safe_float(row.get("required_entry"))
            credit = safe_float(row.get("credit"))
            debit = _current_debit(row)
            if is_credit:
                order_credit = (
                    max(credit, required)
                    if is_scout and math.isfinite(credit) and math.isfinite(required)
                    else required
                )
                current_entry = f"${credit:.2f} credit" if math.isfinite(credit) else "n/a"
                target_entry = f">= ${order_credit:.2f} credit" if math.isfinite(order_credit) else "fresh Schwab recheck"
                trigger = (
                    existing_trigger
                    if existing_trigger
                    else f"Work 1-lot at ${required:.2f} credit only. No chase below ${required:.2f}. Fresh Schwab recheck required."
                    if math.isfinite(required)
                    else "Fresh Schwab recheck required before entry."
                )
                limit_order = f"SELL TO OPEN 1 spread at ${order_credit:.2f} credit limit" if math.isfinite(order_credit) else "fresh Schwab recheck required"
                what_must_improve = (
                    existing_improvement
                    if existing_improvement
                    else
                    f"credit must improve from ${credit:.2f} to at least ${required:.2f}"
                    if math.isfinite(credit) and math.isfinite(required) and credit < required
                    else "quote must tighten with fresh Schwab confirmation"
                )
                watch_kind = "manual_confirmation_scout" if is_scout else "price_improvement_credit"
            else:
                order_debit = (
                    min(debit, required)
                    if is_scout and math.isfinite(debit) and math.isfinite(required)
                    else required
                )
                current_entry = f"${debit:.2f} debit" if math.isfinite(debit) else "n/a"
                target_entry = f"<= ${order_debit:.2f} debit" if math.isfinite(order_debit) else "fresh Schwab recheck"
                trigger = (
                    existing_trigger
                    if existing_trigger
                    else f"Work 1-lot at ${required:.2f} debit or better. No chase above ${required:.2f}. Fresh Schwab recheck required."
                    if math.isfinite(required)
                    else "Fresh Schwab recheck required before entry."
                )
                limit_order = f"BUY TO OPEN 1 spread at ${order_debit:.2f} debit limit" if math.isfinite(order_debit) else "fresh Schwab recheck required"
                what_must_improve = (
                    existing_improvement
                    if existing_improvement
                    else
                    f"debit must fall from ${debit:.2f} to ${required:.2f} or better"
                    if math.isfinite(debit) and math.isfinite(required) and debit > required
                    else "quote must tighten with fresh Schwab confirmation"
                )
                watch_kind = "manual_confirmation_scout" if is_scout else "price_improvement_debit"
            rows.append(
                {
                    "watch_rank_score": safe_float(row.get("confirmation_score"), 0.0) + safe_float(row.get("score"), 0.0) * 0.05,
                    "status": "Watch",
                    "watch_kind": watch_kind,
                    "ticker": row.get("ticker"),
                    "direction": row.get("direction"),
                    "strategy": row.get("strategy"),
                    "flow_quality": row.get("flow_quality"),
                    "sell_leg": row.get("short_leg"),
                    "buy_leg": row.get("long_leg"),
                    "expiry": row.get("expiry"),
                    "dte": row.get("dte"),
                    "current_entry": current_entry,
                    "target_entry": target_entry,
                    "credit": credit,
                    "required_credit": required if is_credit else math.nan,
                    "debit": debit,
                    "max_debit": required if not is_credit else math.nan,
                    "required_entry": required,
                    "spread_width": row.get("spread_width"),
                    "credit_pct_width": row.get("credit_pct_width"),
                    "debit_pct_width": row.get("debit_pct_width"),
                    "target_pct_width": required / safe_float(row.get("spread_width")) if math.isfinite(required) and safe_float(row.get("spread_width")) > 0 else math.nan,
                    "pop_delta_proxy": row.get("pop_delta_proxy"),
                    "score": row.get("score"),
                    "confirmation_score": row.get("confirmation_score"),
                    "confidence": row.get("confidence"),
                    "quote_width_pct": row.get("quote_width_pct"),
                    "limit_order": limit_order,
                    "no_chase_threshold": row.get("no_chase_threshold", required),
                    "what_must_improve": what_must_improve,
                    "fresh_schwab_recheck_required": True,
                    "trigger": trigger,
                    "reason": row.get("trade_status_reason"),
                    "risk_note": "Watch only. Do not enter unless trigger price is available and hard safety gates still pass.",
                    "portfolio_risk": row.get("portfolio_note", ""),
                    "edge_verdict": row.get("edge_verdict", row.get("replay_ev_verdict")),
                    "edge_sample_size": row.get("edge_sample_size"),
                    "edge_win_rate": row.get("edge_win_rate"),
                    "edge_avg_pnl": row.get("edge_avg_pnl"),
                    "edge_match_level": row.get("edge_match_level"),
                    "price_annotation": row.get("price_annotation", ""),
                    "construction_source": row.get("construction_source"),
                    "construction_reason": row.get("construction_reason"),
                    "primary_blocker": row.get("primary_blocker", ""),
                }
            )
        out = pd.DataFrame(rows).sort_values("watch_rank_score", ascending=False).head(top_n).drop(columns=["watch_rank_score"])
        out.insert(0, "rank", range(1, len(out) + 1))
        return out
    rows: list[dict[str, Any]] = []
    for _, row in scored.iterrows():
        if _clean_note(row.get("hard_rejects")):
            continue
        penalties = str(row.get("penalties") or "")
        decision_reason = str(row.get("decision_reason") or "")
        if str(row.get("decision_eligible")).lower() == "true":
            continue
        if "news_catalyst_caution" in penalties or "final_guard_near_term_news_caution" in penalties:
            continue
        direction = str(row.get("direction") or "")
        strategy = str(row.get("strategy") or "")
        is_credit = "Credit" in strategy or math.isfinite(safe_float(row.get("credit")))
        is_debit = "Debit" in strategy or math.isfinite(_current_debit(row))
        credit = safe_float(row.get("credit"))
        target_credit = _entry_credit_target(row)
        debit = _current_debit(row)
        target_debit = _entry_debit_target(row)
        quote_width = safe_float(row.get("quote_width_pct"))

        watch_kind = ""
        trigger = ""
        current_entry = ""
        target_entry = ""
        target_pct = math.nan
        if is_credit and math.isfinite(credit) and math.isfinite(target_credit) and credit < target_credit:
            watch_kind = "price_improvement_credit"
            current_entry = f"${credit:.2f} credit"
            target_entry = f">= ${target_credit:.2f} credit"
            width = safe_float(row.get("spread_width"))
            if math.isfinite(width) and width > 0:
                target_pct = target_credit / width
            trigger = f"Wait for credit to improve to at least ${target_credit:.2f}; rerun Schwab chain before entry."
            limit_order = f"SELL TO OPEN 1 spread at ${target_credit:.2f} credit limit"
            what_must_improve = f"credit must improve from ${credit:.2f} to at least ${target_credit:.2f}"
        elif is_debit and math.isfinite(debit) and math.isfinite(target_debit) and debit > target_debit:
            watch_kind = "price_improvement_debit"
            current_entry = f"${debit:.2f} debit"
            target_entry = f"<= ${target_debit:.2f} debit"
            width = safe_float(row.get("spread_width"))
            if math.isfinite(width) and width > 0:
                target_pct = target_debit / width
            trigger = f"Wait for debit to fall to ${target_debit:.2f} or better; rerun Schwab chain before entry."
            limit_order = f"BUY TO OPEN 1 spread at ${target_debit:.2f} debit limit"
            what_must_improve = f"debit must fall from ${debit:.2f} to ${target_debit:.2f} or better"
        elif "marginal_liquidity" in penalties or "wide_bid_ask" in penalties or decision_reason == "decision_marginal_live_liquidity":
            watch_kind = "execution_improvement"
            if is_credit and math.isfinite(credit):
                current_entry = f"${credit:.2f} credit"
                target_entry = f">= ${max(credit, target_credit) if math.isfinite(target_credit) else credit:.2f} credit with tighter quotes"
                limit_order = f"SELL TO OPEN 1 spread at ${max(credit, target_credit) if math.isfinite(target_credit) else credit:.2f} credit limit after recheck"
            elif is_debit and math.isfinite(debit):
                current_entry = f"${debit:.2f} debit"
                target_entry = f"<= ${min(debit, target_debit) if math.isfinite(target_debit) else debit:.2f} debit with tighter quotes"
                limit_order = f"BUY TO OPEN 1 spread at ${min(debit, target_debit) if math.isfinite(target_debit) else debit:.2f} debit limit after recheck"
            else:
                current_entry = "n/a"
                target_entry = "tighter two-sided market"
                limit_order = "fresh Schwab recheck required"
            trigger = "Wait for tighter bid/ask and real two-leg liquidity; rerun Schwab chain before entry."
            what_must_improve = "bid/ask must tighten and both legs must keep usable liquidity"
        else:
            continue

        short_liq = safe_float(row.get("short_oi"), 0.0) + safe_float(row.get("short_volume"), 0.0)
        long_liq = safe_float(row.get("long_oi"), 0.0) + safe_float(row.get("long_volume"), 0.0)
        out = {
            "watch_rank_score": safe_float(row.get("score"), 0.0) + safe_float(row.get("decision_score"), 0.0) * 0.05,
            "status": "🟡 WAIT",
            "watch_kind": watch_kind,
            "ticker": row.get("ticker"),
            "direction": direction,
            "strategy": strategy,
            "sell_leg": row.get("short_leg"),
            "buy_leg": row.get("long_leg"),
            "expiry": row.get("expiry"),
            "dte": row.get("dte"),
            "current_entry": current_entry,
            "target_entry": target_entry,
            "credit": credit,
            "required_credit": target_credit,
            "debit": debit,
            "max_debit": target_debit,
            "spread_width": row.get("spread_width"),
            "credit_pct_width": row.get("credit_pct_width"),
            "target_pct_width": target_pct,
            "pop_delta_proxy": row.get("pop_delta_proxy"),
            "score": row.get("score"),
            "confidence": row.get("confidence"),
            "quote_width_pct": quote_width,
            "short_leg_liquidity": short_liq,
            "long_leg_liquidity": long_liq,
            "limit_order": limit_order,
            "no_chase_threshold": target_credit if is_credit else target_debit,
            "what_must_improve": what_must_improve,
            "fresh_schwab_recheck_required": True,
            "trigger": trigger,
            "reason": ";".join(x for x in [decision_reason, penalties] if x),
            "risk_note": "Watch only. Do not enter unless trigger price is available and hard safety gates still pass.",
            "edge_verdict": row.get("edge_verdict", row.get("replay_ev_verdict")),
            "edge_sample_size": row.get("edge_sample_size"),
            "edge_win_rate": row.get("edge_win_rate"),
            "edge_avg_pnl": row.get("edge_avg_pnl"),
            "edge_match_level": row.get("edge_match_level"),
            "price_annotation": row.get("price_annotation", ""),
            "construction_source": row.get("construction_source"),
            "construction_reason": row.get("construction_reason"),
            "primary_blocker": row.get("primary_blocker", ""),
        }
        rows.append(out)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).sort_values("watch_rank_score", ascending=False).head(top_n).drop(columns=["watch_rank_score"])
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def rejection_summary(scored: pd.DataFrame) -> pd.DataFrame:
    counts = Counter()
    if scored.empty:
        return pd.DataFrame(columns=["reason", "count"])
    for _, row in scored.iterrows():
        raw = str(row.get("hard_rejects") or row.get("penalties") or "")
        if not raw and "decision_eligible" in scored.columns and not str(row.get("decision_eligible")).lower() == "true":
            raw = str(row.get("decision_reason") or "")
        if not raw:
            raw = "score_below_threshold"
        for token in raw.split(";"):
            token = token.strip()
            if token:
                counts[token] += 1
    return pd.DataFrame([{"reason": k, "count": v} for k, v in counts.most_common()])


def decision_summary(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "decision_reason" not in scored.columns:
        return pd.DataFrame(columns=["decision_reason", "count"])
    counts = Counter()
    for reason in scored["decision_reason"].fillna("").astype(str):
        reason = reason.strip() or "decision_not_checked"
        counts[reason] += 1
    return pd.DataFrame([{"decision_reason": k, "count": v} for k, v in counts.most_common()])


def _current_credit_debit_text(row: pd.Series) -> str:
    credit = safe_float(row.get("credit"))
    debit = _current_debit(row)
    if _is_credit_strategy(row) and math.isfinite(credit):
        return f"${credit:.2f} credit"
    if _is_debit_strategy(row) and math.isfinite(debit):
        return f"${debit:.2f} debit"
    return "n/a"


def _required_entry_text(row: pd.Series) -> str:
    value = safe_float(row.get("required_entry"))
    if not math.isfinite(value):
        value = _entry_credit_target(row) if _is_credit_strategy(row) else _entry_debit_target(row)
    if not math.isfinite(value):
        return "fresh Schwab recheck"
    return f">= ${value:.2f} credit" if _is_credit_strategy(row) else f"<= ${value:.2f} debit"


def _size_text(row: pd.Series) -> str:
    contracts = safe_float(row.get("contracts"))
    if math.isfinite(contracts) and contracts > 0:
        return f"{int(contracts)}-lot"
    cap = safe_float(row.get("portfolio_size_cap"))
    if math.isfinite(cap) and cap > 0:
        return f"max {int(cap)}-lot"
    return "1-lot review"


def _no_chase_text(row: pd.Series) -> str:
    value = safe_float(row.get("no_chase_threshold"))
    if not math.isfinite(value):
        return ""
    return f"no chase below ${value:.2f}" if _is_credit_strategy(row) else f"no chase above ${value:.2f}"


def _status_label(status: object) -> str:
    text = str(status or "").strip()
    icons = {
        "Execute": "🟢",
        "Watch": "🟡",
        "Research": "🔵",
        "Avoid": "🔴",
        "Conditional": "🟡",
        "Manage": "🔴",
        "Income": "🔵",
    }
    return f"{icons.get(text, '⚪')} {text}" if text else ""


def _quality_icon(status: object) -> str:
    text = str(status or "").lower()
    if text in {"ok", "fresh", "available", "present"}:
        return "🟢"
    if text in {"degraded", "stale", "warning", "not_present", "unavailable"}:
        return "🟡"
    if text in {"missing", "critical"}:
        return "🔴"
    return "🔵"


def build_data_quality_status(
    *,
    input_provenance: dict[str, Any] | None,
    scored: pd.DataFrame,
    portfolio: dict[str, Any] | None,
    catalysts: pd.DataFrame | None,
    recent_performance: dict[str, Any] | None,
    live_outcomes: dict[str, Any] | None,
    run_mode: str,
) -> dict[str, Any]:
    provenance = input_provenance or {}
    exports = provenance.get("exports") or {}
    required_exports = ["stock_screener", "hot_chains", "bot_eod_report"]
    missing_exports = [name for name in required_exports if name not in exports]
    live_counts = scored["live_status"].fillna("unknown").value_counts().to_dict() if not scored.empty and "live_status" in scored.columns else {}
    pass_count = int(live_counts.get("PASS", 0))
    portfolio_status = (portfolio or {}).get("status", "not_checked")
    browser_count = int(provenance.get("browser_text_count", 0) or 0)
    catalyst_counts = catalysts["catalyst_status"].fillna("unknown").value_counts().to_dict() if catalysts is not None and not catalysts.empty and "catalyst_status" in catalysts.columns else {}

    items = [
        {
            "check": "UW files present",
            "status": "ok" if not missing_exports else "missing",
            "detail": "required exports found" if not missing_exports else f"missing {', '.join(missing_exports)}",
            "critical": bool(missing_exports),
        },
        {
            "check": "Schwab quotes available",
            "status": "ok" if pass_count else "missing",
            "detail": f"{pass_count} PASS rows; counts={live_counts}" if live_counts else "no live quote rows",
            "critical": pass_count == 0,
        },
        {
            "check": "Schwab portfolio available",
            "status": "ok" if portfolio_status == "ok" else "missing",
            "detail": f"positions={(portfolio or {}).get('position_count', 0)}" if portfolio_status == "ok" else str((portfolio or {}).get("error") or portfolio_status),
            "critical": portfolio_status != "ok",
        },
        {
            "check": "Browser/news notes present",
            "status": "ok" if browser_count > 0 else "missing",
            "detail": f"{browser_count} local browser/news captures; catalyst counts={catalyst_counts}" if browser_count > 0 else "no local browser/news captures",
            "critical": browser_count == 0,
        },
        {
            "check": "Replay data freshness",
            "status": "ok" if (recent_performance or {}).get("status") == "ok" else "unavailable",
            "detail": (
                f"latest={(recent_performance or {}).get('latest_asof', '')}; window={(recent_performance or {}).get('window', '')}"
                if (recent_performance or {}).get("status") == "ok"
                else str((recent_performance or {}).get("reason") or (recent_performance or {}).get("status") or "not_checked")
            ),
            "critical": False,
        },
        {
            "check": "Live outcome ledger freshness",
            "status": "ok" if (live_outcomes or {}).get("status") == "ok" else "unavailable",
            "detail": (
                f"latest={(live_outcomes or {}).get('latest_report_date', '')}; window={(live_outcomes or {}).get('window', '')}"
                if (live_outcomes or {}).get("status") == "ok"
                else str((live_outcomes or {}).get("reason") or (live_outcomes or {}).get("status") or "not_checked")
            ),
            "critical": False,
        },
    ]
    critical_blockers = []
    for item in items:
        if item["critical"]:
            key = str(item["check"]).lower().replace("/", "_").replace(" ", "_")
            critical_blockers.append(key)
    return {
        "run_mode": run_mode,
        "status": "critical" if critical_blockers else "ok",
        "critical_blockers": critical_blockers,
        "items": items,
    }


def _component_label(score: float, label: str) -> str:
    if not math.isfinite(score):
        return f"n/a {label}"
    bucket = "strong" if score >= 8 else "usable" if score >= 6 else "weak" if score >= 4 else "low"
    return f"{score:.0f}/10 {bucket}: {label}"


def _confidence_component_scores(row: pd.Series, live_outcomes: dict[str, Any] | None) -> dict[str, str]:
    edge_verdict = str(row.get("edge_verdict") or row.get("replay_ev_verdict") or "")
    sample = safe_float(row.get("edge_sample_size"), safe_float(row.get("historical_sample_size"), math.nan))
    if edge_verdict in {"positive", "acceptable"} and math.isfinite(sample) and sample >= 7:
        replay_score, replay_label = 8.0, f"{edge_verdict} n={int(sample)}"
    elif edge_verdict in {"thin_sample", "acceptable_proxy"} or (math.isfinite(sample) and sample < 5):
        replay_score, replay_label = 4.0, f"thin/proxy {edge_verdict or 'sample'}"
    elif str(row.get("replay_ev_verdict") or "").startswith("negative"):
        replay_score, replay_label = 2.0, "negative replay/live analogue"
    else:
        replay_score, replay_label = 5.0, edge_verdict or "unavailable"

    quote = safe_float(row.get("quote_width_pct"))
    live_status = str(row.get("live_status") or "")
    liq = min(
        safe_float(row.get("short_oi"), 0.0) + safe_float(row.get("short_volume"), 0.0),
        safe_float(row.get("long_oi"), 0.0) + safe_float(row.get("long_volume"), 0.0),
    )
    if live_status == "PASS" and math.isfinite(quote) and quote <= 0.20 and liq >= 500:
        live_score, live_label = 9.0, "clean Schwab quote/liquidity"
    elif live_status == "PASS" and math.isfinite(quote) and quote <= 0.35 and liq >= 100:
        live_score, live_label = 7.0, "usable Schwab quote/liquidity"
    elif live_status == "PASS":
        live_score, live_label = 5.0, "live priced but marginal quote/liquidity"
    else:
        live_score, live_label = 1.0, live_status or "missing live pricing"

    portfolio_note = _clean_note(row.get("portfolio_note"))
    exposure = safe_float(row.get("portfolio_exposure_pct"))
    hedging = str(row.get("portfolio_hedging", "")).lower() == "true"
    if math.isfinite(exposure) and exposure >= 0.08 and not hedging:
        portfolio_score, portfolio_label = 5.0, f"exposure note {exposure:.1%}; not an execution gate"
    elif portfolio_note:
        portfolio_score, portfolio_label = 6.0 if hedging else 5.0, portfolio_note
    else:
        portfolio_score, portfolio_label = 8.0, "no concentration flag"

    catalyst_status = str(row.get("catalyst_status") or "").lower()
    penalties = _token_set(row.get("penalties"))
    if catalyst_status in {"supportive", "mixed"} and "news_unconfirmed" not in penalties:
        catalyst_score, catalyst_label = 8.0 if catalyst_status == "supportive" else 6.0, catalyst_status
    elif catalyst_status == "caution" or "news_catalyst_caution" in penalties:
        catalyst_score, catalyst_label = 2.0, "material news caution"
    else:
        catalyst_score, catalyst_label = 3.0, "news unconfirmed"

    ratio = safe_float(row.get("expected_move_ratio"))
    trend = str(row.get("regime_trend") or "")
    direction = str(row.get("direction") or "")
    trend_ok = trend == "range" or (trend == "uptrend" and direction in BULLISH_DIRECTIONS) or (trend == "downtrend" and direction in BEARISH_DIRECTIONS)
    technical_score = 8.0 if math.isfinite(ratio) and ratio >= (1.0 if _is_debit_strategy(row) else 0.65) and trend_ok else 5.0 if trend_ok else 3.0
    technical_label = f"expected-move ratio {ratio:.2f}; trend {trend}" if math.isfinite(ratio) else f"trend {trend or 'unknown'}"

    if _is_credit_strategy(row):
        rr_value = safe_float(row.get("credit_pct_width"))
        rr_score = 9.0 if math.isfinite(rr_value) and rr_value >= 0.25 else 7.0 if math.isfinite(rr_value) and rr_value >= 0.18 else 3.0
        rr_label = f"credit/width {rr_value:.1%}" if math.isfinite(rr_value) else "credit unavailable"
    else:
        reward = safe_float(row.get("reward_risk"))
        debit_pct = safe_float(row.get("debit_pct_width"))
        rr_score = 8.0 if math.isfinite(reward) and reward >= 1.5 and debit_pct <= 0.40 else 6.0 if math.isfinite(reward) and reward >= 1.0 else 3.0
        rr_label = f"reward/risk {reward:.2f}; debit/width {debit_pct:.1%}" if math.isfinite(reward) and math.isfinite(debit_pct) else "reward/risk unavailable"

    live_adjustment = live_outcome_adjustment(live_outcomes, row.get("strategy"), row.get("direction"))
    live_family_label = str(live_adjustment.get("status") or "unavailable")
    if live_adjustment.get("block_execute"):
        replay_score = min(replay_score, 3.0)
        replay_label = f"{replay_label}; live family negative"

    return {
        "replay_confidence": _component_label(replay_score, replay_label),
        "live_execution_confidence": _component_label(live_score, live_label),
        "portfolio_fit": _component_label(portfolio_score, portfolio_label),
        "catalyst_news_quality": _component_label(catalyst_score, catalyst_label),
        "liquidity_quality": _component_label(live_score, f"quote {quote:.1%}; min OI+vol {liq:.0f}" if math.isfinite(quote) else f"min OI+vol {liq:.0f}"),
        "technical_level_quality": _component_label(technical_score, technical_label),
        "risk_reward_quality": _component_label(rr_score, rr_label),
        "live_outcome_family": str(live_adjustment.get("family") or setup_family(row.get("strategy"), row.get("direction"))),
        "live_outcome_family_status": live_family_label,
    }


def apply_confidence_components(scored: pd.DataFrame, live_outcomes: dict[str, Any] | None = None) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    component_cols = [
        "replay_confidence",
        "live_execution_confidence",
        "portfolio_fit",
        "catalyst_news_quality",
        "liquidity_quality",
        "technical_level_quality",
        "risk_reward_quality",
        "live_outcome_family",
        "live_outcome_family_status",
    ]
    for col in component_cols:
        out[col] = ""
    for idx, row in out.iterrows():
        comps = _confidence_component_scores(row, live_outcomes)
        for col, value in comps.items():
            out.at[idx, col] = value
        adjustment = live_outcome_adjustment(live_outcomes, row.get("strategy"), row.get("direction"))
        if adjustment.get("block_execute"):
            out.at[idx, "penalties"] = _append_token(row.get("penalties"), f"negative_live_expectancy:{adjustment.get('family')}")
            score = max(0.0, safe_float(row.get("score"), 0.0) - safe_float(adjustment.get("score_penalty"), 0.0))
            out.at[idx, "score"] = round(score, 2)
            out.at[idx, "confidence"] = _confidence_from_score(score)
    out["confidence_components"] = out[
        [
            "replay_confidence",
            "live_execution_confidence",
            "portfolio_fit",
            "catalyst_news_quality",
            "liquidity_quality",
            "technical_level_quality",
            "risk_reward_quality",
        ]
    ].agg(" | ".join, axis=1)
    return out


def apply_data_quality_gate(scored: pd.DataFrame, data_quality: dict[str, Any] | None) -> pd.DataFrame:
    if scored.empty:
        return scored
    out = scored.copy()
    blockers = set((data_quality or {}).get("critical_blockers") or [])
    for idx, row in out.iterrows():
        row_blockers: list[str] = []
        if "schwab_quotes_available" in blockers or str(row.get("live_status") or "") != "PASS":
            row_blockers.append("data_gate_missing_live_pricing")
        if "schwab_portfolio_available" in blockers:
            row_blockers.append("data_gate_missing_portfolio_state")
        penalties = _token_set(row.get("penalties"))
        catalyst_status = str(row.get("catalyst_status") or "").strip().lower()
        if "news_unconfirmed" in penalties or catalyst_status == "unknown":
            row_blockers.append("data_gate_news_unconfirmed")
        if row_blockers:
            out.at[idx, "data_quality_blockers"] = ";".join(row_blockers)
            if str(row.get("trade_status") or "") == "Execute":
                out.at[idx, "trade_status"] = "Research"
                out.at[idx, "trade_tier"] = "data-quality-downgrade"
                out.at[idx, "trade_status_reason"] = _append_token(row.get("trade_status_reason"), ";".join(row_blockers))
                out.at[idx, "primary_blocker"] = row_blockers[0]
    return out


def _action_board_rows(final: pd.DataFrame, watch: pd.DataFrame, research: pd.DataFrame, avoid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add_row(status: str, row: pd.Series) -> None:
        rows.append(
            {
                "Status": _status_label(status),
                "Ticker": row.get("ticker"),
                "Strategy": row.get("strategy"),
                "Flow Quality": row.get("flow_quality", ""),
                "Sell Leg": _leg_label(row.get("sell_leg", row.get("short_leg"))),
                "Buy Leg": _leg_label(row.get("buy_leg", row.get("long_leg"))),
                "Expiry": row.get("expiration_date", row.get("expiry")),
                "Current Credit/Debit": _current_credit_debit_text(row),
                "Required Entry": _required_entry_text(row),
                "Max Profit": _money(row.get("max_profit")),
                "Max Loss": _money(row.get("max_loss")),
                "Breakeven": round(safe_float(row.get("breakeven")), 2) if math.isfinite(safe_float(row.get("breakeven"))) else "",
                "POP/Delta Proxy": _pct(row.get("pop_delta_proxy")),
                "Confirmation Score": round(safe_float(row.get("confirmation_score")), 2)
                if math.isfinite(safe_float(row.get("confirmation_score")))
                else "",
                "Edge Verdict": row.get("edge_verdict", row.get("replay_ev_verdict", "")),
                "Edge Sample Size": int(safe_float(row.get("edge_sample_size"))) if math.isfinite(safe_float(row.get("edge_sample_size"))) else "",
                "Edge Win Rate": _pct(row.get("edge_win_rate")),
                "Edge Avg P/L": _money(row.get("edge_avg_pnl")),
                "Edge Match Level": row.get("edge_match_level", ""),
                "Construction Source": row.get("construction_source", ""),
                "No Chase": _no_chase_text(row),
                "What Must Improve": row.get("what_must_improve", ""),
                "Size": _size_text(row),
                "Portfolio Risk": _clean_note(row.get("portfolio_note")),
                "Primary Blocker": row.get("primary_blocker", ""),
                "Reason": row.get("trade_status_reason", row.get("reason", row.get("primary_blocker", ""))),
            }
        )

    for _, row in final.iterrows():
        add_row("Execute", row)
    for _, row in watch.iterrows():
        add_row("Watch", row)
    for _, row in research.head(12).iterrows():
        add_row("Research", row)
    for _, row in avoid.head(12).iterrows():
        add_row("Avoid", row)
    return pd.DataFrame(rows)


def _compact_trade_label(row: pd.Series) -> str:
    direction = _clean_note(row.get("direction"))
    strategy = _clean_note(row.get("strategy"))
    if direction and strategy and direction not in strategy:
        return f"{direction} {strategy}"
    return strategy or direction


def _compact_entry_text(row: pd.Series) -> str:
    current = _clean_note(row.get("current_entry"))
    if current:
        return current
    current_credit_debit = safe_float(row.get("current_credit_debit"))
    if math.isfinite(current_credit_debit):
        return f"${current_credit_debit:.2f} {'credit' if _is_credit_strategy(row) else 'debit'}"
    return _current_credit_debit_text(row)


def _compact_trigger_text(row: pd.Series) -> str:
    raw_target = row.get("target_entry")
    target = _clean_note(raw_target)
    target_number = safe_float(raw_target)
    if target and not math.isfinite(target_number):
        return target
    if math.isfinite(target_number):
        return f">= ${target_number:.2f} credit" if _is_credit_strategy(row) else f"<= ${target_number:.2f} debit"
    return _required_entry_text(row)


def _compact_edge_text(row: pd.Series) -> str:
    verdict = _clean_note(row.get("edge_verdict", row.get("replay_ev_verdict")))
    sample = safe_float(row.get("edge_sample_size"))
    win = safe_float(row.get("edge_win_rate"))
    avg = safe_float(row.get("edge_avg_pnl"))
    parts = [verdict] if verdict else []
    if math.isfinite(sample) and sample > 0:
        parts.append(f"n={int(sample)}")
    if math.isfinite(win):
        parts.append(f"win {win:.1%}")
    if math.isfinite(avg):
        parts.append(f"avg ${avg:.2f}")
    return "; ".join(parts)


def _compact_reason_text(row: pd.Series) -> str:
    status_text = str(row.get("trade_status") or row.get("status") or "")
    if "Watch" in status_text:
        keys = ("what_must_improve", "trade_status_reason", "reason", "primary_blocker", "penalties", "hard_rejects")
    else:
        keys = ("primary_blocker", "trade_status_reason", "reason", "what_must_improve", "penalties", "hard_rejects")
    for key in keys:
        value = _clean_note(row.get(key))
        if value:
            return value
    return ""


def _diversified_report_rows(
    frame: pd.DataFrame,
    *,
    limit: int,
    max_per_ticker: int = 1,
    existing_ticker_counts: Counter | None = None,
) -> pd.DataFrame:
    if frame.empty or limit <= 0:
        return frame.iloc[0:0].copy()
    work = frame.copy()
    def numeric_col(name: str, default: float) -> pd.Series:
        if name in work.columns:
            return pd.to_numeric(work[name], errors="coerce").fillna(default)
        return pd.Series([default] * len(work), index=work.index)

    work["_report_score"] = numeric_col("score", 0.0)
    work["_report_confirm"] = numeric_col("confirmation_score", 0.0)
    work["_report_edge_n"] = numeric_col("edge_sample_size", 0.0)
    work["_report_quote"] = numeric_col("quote_width_pct", math.inf)
    work = work.sort_values(
        ["_report_score", "_report_confirm", "_report_edge_n", "_report_quote"],
        ascending=[False, False, False, True],
    )
    counts = Counter(existing_ticker_counts or {})
    selected: list[int] = []
    for idx, row in work.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        if ticker and counts[ticker] >= max_per_ticker:
            continue
        selected.append(idx)
        if ticker:
            counts[ticker] += 1
        if len(selected) >= limit:
            break
    return work.loc[selected].drop(columns=["_report_score", "_report_confirm", "_report_edge_n", "_report_quote"], errors="ignore")


def _compact_action_rows(final: pd.DataFrame, watch: pd.DataFrame, research: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(status: str, row: pd.Series) -> None:
        score = safe_float(row.get("score"))
        max_loss = safe_float(row.get("max_loss"))
        risk = _clean_note(row.get("position_size"))
        if not risk and math.isfinite(max_loss):
            risk = f"max loss ${max_loss:.0f}"
        rows.append(
            {
                "Status": _status_label(status),
                "Ticker": row.get("ticker"),
                "Trade": _compact_trade_label(row),
                "Sell Leg": _leg_label(row.get("sell_leg", row.get("short_leg"))),
                "Buy Leg": _leg_label(row.get("buy_leg", row.get("long_leg"))),
                "Expiry": row.get("expiration_date", row.get("expiry")),
                "Entry Now": _compact_entry_text(row),
                "Trigger / No Chase": _compact_trigger_text(row),
                "POP": _pct(row.get("pop_delta_proxy")),
                "Score": f"{score:.2f}" if math.isfinite(score) else "",
                "Edge": _compact_edge_text(row),
                "Size / Risk": risk or _size_text(row),
                "Target Profit": _money(row.get("target_profit_total")),
                "EV/Risk": f"{safe_float(row.get('ev_per_dollar_risk')):.2f}" if math.isfinite(safe_float(row.get("ev_per_dollar_risk"))) else "",
                "Why": _compact_reason_text(row),
            }
        )

    for _, row in final.iterrows():
        add("Execute", row)
    existing_counts = Counter(str(row.get("ticker") or "").upper() for _, row in final.iterrows())
    displayed_watch = _diversified_report_rows(
        watch,
        limit=len(watch),
        max_per_ticker=1,
        existing_ticker_counts=existing_counts,
    )
    for _, row in displayed_watch.iterrows():
        add("Watch", row)
    existing_counts.update(str(row.get("ticker") or "").upper() for _, row in displayed_watch.iterrows())
    for _, row in _diversified_report_rows(research, limit=8, max_per_ticker=1, existing_ticker_counts=existing_counts).iterrows():
        add("Research", row)
    return pd.DataFrame(rows)


def _compact_watch_rows(watch: pd.DataFrame) -> pd.DataFrame:
    if watch.empty:
        return pd.DataFrame()
    rows = []
    for _, row in watch.iterrows():
        score = safe_float(row.get("score"))
        rows.append(
            {
                "Status": _status_label("Watch"),
                "Ticker": row.get("ticker"),
                "Trade": _compact_trade_label(row),
                "Sell Leg": _leg_label(row.get("sell_leg")),
                "Buy Leg": _leg_label(row.get("buy_leg")),
                "Expiry": row.get("expiry"),
                "Current": _compact_entry_text(row),
                "Trigger": _compact_trigger_text(row),
                "POP": _pct(row.get("pop_delta_proxy")),
                "Score": f"{score:.2f}" if math.isfinite(score) else "",
                "Rule": _clean_note(row.get("trigger")) or _clean_note(row.get("limit_order")),
            }
        )
    return pd.DataFrame(rows)


def _compact_research_rows(research: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if research.empty:
        return pd.DataFrame()
    rows = []
    for _, row in _diversified_report_rows(research, limit=limit, max_per_ticker=1).iterrows():
        score = safe_float(row.get("score"))
        rows.append(
            {
                "Status": _status_label("Research"),
                "Ticker": row.get("ticker"),
                "Trade": _compact_trade_label(row),
                "Sell Leg": _leg_label(row.get("sell_leg", row.get("short_leg"))),
                "Buy Leg": _leg_label(row.get("buy_leg", row.get("long_leg"))),
                "Expiry": row.get("expiry"),
                "Entry Now": _compact_entry_text(row),
                "Needed": _compact_trigger_text(row),
                "Score": f"{score:.2f}" if math.isfinite(score) else "",
                "Edge": _compact_edge_text(row),
                "Blocker": _clean_note(row.get("primary_blocker")) or _compact_reason_text(row),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_actions_frame(portfolio: dict[str, Any] | None, *, lane: str | None = None) -> pd.DataFrame:
    actions = list((portfolio or {}).get("risk_actions") or [])
    if lane:
        actions = [row for row in actions if row.get("lane") == lane]
    rows = []
    for row in actions:
        exposure = safe_float(row.get("exposure_pct"))
        rows.append(
            {
                "Status": f"{row.get('icon', '🔵')} {row.get('action', '')}",
                "Ticker": row.get("ticker"),
                "Position": row.get("position"),
                "Exposure": f"{exposure:.1%}" if math.isfinite(exposure) else "",
                "Reason": row.get("reason", ""),
                "Action": row.get("instruction", ""),
                "Assignment / Tradeoff": row.get("assignment_risk") or row.get("upside_downside_tradeoff") or row.get("portfolio_impact", ""),
            }
        )
    return pd.DataFrame(rows)


def _data_quality_frame(data_quality: dict[str, Any] | None) -> pd.DataFrame:
    rows = []
    for item in (data_quality or {}).get("items", []):
        rows.append(
            {
                "Status": f"{_quality_icon(item.get('status'))} {item.get('status')}",
                "Check": item.get("check"),
                "Detail": item.get("detail"),
            }
        )
    return pd.DataFrame(rows)


def _portfolio_risk_summary(portfolio: dict[str, Any] | None) -> str:
    if not portfolio or portfolio.get("status") != "ok":
        return f"portfolio unavailable: {(portfolio or {}).get('error') or (portfolio or {}).get('status', 'not_checked')}"
    large = portfolio.get("large_equity_exposure", {}) or {}
    actions = portfolio.get("risk_actions", []) or []
    return (
        f"{portfolio.get('position_count', 0)} positions; "
        f"account ${safe_float(portfolio.get('total_value'), 0):,.0f}; "
        f"day P/L ${safe_float(portfolio.get('day_pnl'), 0):,.0f}; "
        f"{len(large)} large equity exposures; {len(actions)} risk/income actions"
    )


def _top_action_today(final: pd.DataFrame, watch: pd.DataFrame, portfolio: dict[str, Any] | None, data_quality: dict[str, Any] | None) -> str:
    if (data_quality or {}).get("status") == "critical":
        blockers = ", ".join((data_quality or {}).get("critical_blockers") or [])
        return f"Stand down from Execute until data gate clears: {blockers}"
    if not final.empty:
        row = final.sort_values("rank").iloc[0] if "rank" in final.columns else final.iloc[0]
        return f"Execute {row.get('ticker')} {row.get('strategy')} at {_compact_entry_text(row)} with lifecycle alerts"
    risk_actions = list((portfolio or {}).get("risk_actions") or [])
    if risk_actions:
        first = risk_actions[0]
        return f"{first.get('action')} {first.get('ticker')}: {first.get('instruction')}"
    if not watch.empty:
        row = watch.iloc[0]
        return f"Rest conditional limit for {row.get('ticker')}: {_clean_note(row.get('trigger')) or _compact_trigger_text(row)}"
    return "No new exposure; use near-miss alerts or stand down."


def _business_days_remaining(asof: dt.date) -> int:
    day = asof
    remaining = 0
    while day.month == asof.month:
        if day.weekday() < 5:
            remaining += 1
        day += dt.timedelta(days=1)
    return max(1, remaining)


def _target_profit_per_contract(row: pd.Series) -> float:
    max_profit = safe_float(row.get("max_profit"))
    credit = safe_float(row.get("credit"))
    debit = safe_float(row.get("debit"))
    if _is_credit_strategy(row) and math.isfinite(credit) and credit > 0:
        target = credit * 100.0 * 0.60
    elif _is_debit_strategy(row) and math.isfinite(debit) and debit > 0:
        target = debit * 100.0 * 0.60
    else:
        target = math.nan
    if math.isfinite(target) and math.isfinite(max_profit) and max_profit > 0:
        return max(0.0, min(target, max_profit))
    return target


def _expected_value_per_contract(row: pd.Series) -> tuple[float, str]:
    avg = safe_float(row.get("edge_avg_pnl"))
    if math.isfinite(avg):
        return avg, "edge_avg_pnl"
    win_rate = safe_float(row.get("edge_win_rate"))
    target = _target_profit_per_contract(row)
    max_loss = safe_float(row.get("max_loss"))
    if math.isfinite(win_rate) and math.isfinite(target) and math.isfinite(max_loss) and max_loss > 0:
        loss_assumption = min(max_loss, target)
        return win_rate * target - (1.0 - win_rate) * loss_assumption, "edge_win_rate_proxy"
    return math.nan, "unavailable"


def _liquidity_capacity_contracts(row: pd.Series) -> int:
    short_liq = safe_float(row.get("short_oi"), 0.0) + safe_float(row.get("short_volume"), 0.0)
    long_liq = safe_float(row.get("long_oi"), 0.0) + safe_float(row.get("long_volume"), 0.0)
    liq = min(short_liq, long_liq)
    if not math.isfinite(liq) or liq <= 0:
        return 1
    return max(1, int(liq * 0.01))


def build_target_capital_model(
    *,
    asof: dt.date,
    monthly_profit_target: float,
    month_to_date_realized_pnl: float = 0.0,
    max_monthly_drawdown: float = 0.0,
    risk_budget: float = 0.0,
    risk_config: dict[str, Any] | None = None,
    portfolio: dict[str, Any] | None = None,
    final: pd.DataFrame | None = None,
) -> dict[str, Any]:
    risk_config = risk_config or {}
    target = max(0.0, safe_float(monthly_profit_target, 0.0))
    mtd = safe_float(month_to_date_realized_pnl, 0.0)
    remaining = max(0.0, target - mtd)
    days_left = _business_days_remaining(asof)
    required_daily = remaining / days_left if days_left > 0 else remaining
    max_daily_risk = safe_float(risk_config.get("max_risk_per_day"), risk_budget)
    if not math.isfinite(max_daily_risk) or max_daily_risk <= 0:
        max_daily_risk = risk_budget
    max_total_open_risk = safe_float(risk_config.get("max_total_open_risk"), 0.0)
    if math.isfinite(max_total_open_risk) and max_total_open_risk > 0:
        max_daily_risk = min(max_daily_risk, max_total_open_risk)
    account_value = safe_float((portfolio or {}).get("total_value"), math.nan)
    cash = safe_float((portfolio or {}).get("cash"), math.nan)
    execute = final if final is not None else pd.DataFrame()
    execute_target_profit = safe_float(execute.get("target_profit_total", pd.Series(dtype=float)).sum(), 0.0) if not execute.empty else 0.0
    execute_max_loss = safe_float(execute.get("position_max_loss", pd.Series(dtype=float)).sum(), 0.0) if not execute.empty else 0.0
    monthly_run_rate = execute_target_profit * days_left
    one_lot_target_profit = 0.0
    one_lot_max_loss = 0.0
    best_contract_target_profit = 0.0
    best_contract_max_loss = 0.0
    best_contract_ticker = ""
    best_contract_selected_contracts = math.nan
    if not execute.empty:
        for _, row in execute.iterrows():
            per_contract_target = safe_float(row.get("target_profit_per_contract"))
            per_contract_risk = safe_float(row.get("max_loss"))
            if not math.isfinite(per_contract_target):
                per_contract_target = _target_profit_per_contract(row)
            if not math.isfinite(per_contract_target):
                contracts = safe_float(row.get("contracts"), 1.0)
                total_target = safe_float(row.get("target_profit_total"))
                if math.isfinite(total_target) and math.isfinite(contracts) and contracts > 0:
                    per_contract_target = total_target / contracts
                elif math.isfinite(total_target):
                    per_contract_target = total_target
            if not math.isfinite(per_contract_risk):
                contracts = safe_float(row.get("contracts"), 1.0)
                total_risk = safe_float(row.get("position_max_loss"))
                if math.isfinite(total_risk) and math.isfinite(contracts) and contracts > 0:
                    per_contract_risk = total_risk / contracts
                elif math.isfinite(total_risk):
                    per_contract_risk = total_risk
            if math.isfinite(per_contract_target) and per_contract_target > 0:
                one_lot_target_profit += per_contract_target
                if math.isfinite(per_contract_risk) and per_contract_risk > 0:
                    one_lot_max_loss += per_contract_risk
                    if per_contract_target > best_contract_target_profit:
                        best_contract_target_profit = per_contract_target
                        best_contract_max_loss = per_contract_risk
                        best_contract_ticker = str(row.get("ticker") or "")
                        best_contract_selected_contracts = safe_float(row.get("contracts"), 1.0)
    target_repeats_required = math.nan
    risk_required_for_daily_target = math.nan
    best_setup_contracts_required = math.nan
    best_setup_risk_required = math.nan
    if one_lot_target_profit > 0:
        target_repeats_required = math.ceil(required_daily / one_lot_target_profit) if required_daily > 0 else 0
        risk_required_for_daily_target = target_repeats_required * one_lot_max_loss
    if best_contract_target_profit > 0:
        best_setup_contracts_required = math.ceil(required_daily / best_contract_target_profit) if required_daily > 0 else 0
        best_setup_risk_required = best_setup_contracts_required * best_contract_max_loss
    risk_gap_for_daily_target = (
        risk_required_for_daily_target - max_daily_risk
        if math.isfinite(risk_required_for_daily_target) and math.isfinite(max_daily_risk)
        else math.nan
    )
    binding = []
    if target <= 0:
        feasibility = "not_configured"
        binding.append("monthly target not configured")
    elif remaining <= 0:
        feasibility = "achieved"
        binding.append("monthly target already met")
    elif execute.empty:
        feasibility = "infeasible"
        binding.append("no Execute trades")
    elif execute_target_profit < required_daily:
        feasibility = "stretched" if execute_target_profit > 0 else "infeasible"
        binding.append("execute target profit below required daily pace")
        if math.isfinite(risk_gap_for_daily_target) and risk_gap_for_daily_target > 0:
            binding.append("risk budget below target-sufficient sizing")
        elif (
            math.isfinite(best_setup_contracts_required)
            and math.isfinite(best_contract_selected_contracts)
            and best_setup_contracts_required > best_contract_selected_contracts
        ):
            binding.append("liquidity/contract cap below target-sufficient sizing")
    else:
        feasibility = "feasible"
    if execute_max_loss > max_daily_risk > 0:
        feasibility = "infeasible"
        binding.append("execute max loss exceeds daily risk budget")
    if math.isfinite(max_monthly_drawdown) and max_monthly_drawdown > 0 and abs(min(mtd, 0.0)) >= max_monthly_drawdown:
        feasibility = "infeasible"
        binding.append("monthly drawdown limit reached")
    if feasibility == "feasible" and monthly_run_rate < remaining:
        feasibility = "stretched"
        binding.append("monthly run-rate below remaining target")
    return {
        "monthly_profit_target": target,
        "month_to_date_realized_pnl": mtd,
        "remaining_monthly_target": remaining,
        "business_days_remaining": days_left,
        "required_daily_pnl": required_daily,
        "available_daily_risk_budget": max_daily_risk,
        "max_monthly_drawdown": safe_float(max_monthly_drawdown, 0.0),
        "account_value": account_value,
        "cash": cash,
        "execute_target_profit": execute_target_profit,
        "execute_max_loss": execute_max_loss,
        "monthly_run_rate_from_execute": monthly_run_rate,
        "one_lot_target_profit": one_lot_target_profit,
        "one_lot_max_loss": one_lot_max_loss,
        "target_repeats_required": target_repeats_required,
        "risk_required_for_daily_target": risk_required_for_daily_target,
        "risk_gap_for_daily_target": risk_gap_for_daily_target,
        "best_setup_ticker": best_contract_ticker,
        "best_setup_selected_contracts": best_contract_selected_contracts,
        "best_setup_contracts_required": best_setup_contracts_required,
        "best_setup_risk_required": best_setup_risk_required,
        "target_feasibility": feasibility,
        "binding_constraint": "; ".join(dict.fromkeys(binding)) if binding else "none",
    }


def _first_screen_frame(
    *,
    run_mode: str,
    data_quality: dict[str, Any] | None,
    regime: dict[str, Any],
    portfolio: dict[str, Any] | None,
    final: pd.DataFrame,
    watch: pd.DataFrame,
    watch_alerts: pd.DataFrame,
    target_model: dict[str, Any] | None = None,
) -> pd.DataFrame:
    income_count = len((portfolio or {}).get("portfolio_income_actions") or [])
    data_status = (data_quality or {}).get("status", "unknown")
    rows = [
            {"Item": "Pipeline", "Value": PIPELINE_NAME},
            {"Item": "Version", "Value": PIPELINE_VERSION},
            {"Item": "Version lock", "Value": "locked 2026-05-21"},
            {"Item": "Monthly target", "Value": _money((target_model or {}).get("monthly_profit_target"))},
            {"Item": "MTD realized P/L", "Value": _money((target_model or {}).get("month_to_date_realized_pnl"))},
            {"Item": "Remaining target", "Value": _money((target_model or {}).get("remaining_monthly_target"))},
            {"Item": "Required daily pace", "Value": _money((target_model or {}).get("required_daily_pnl"))},
            {"Item": "Available risk budget", "Value": _money((target_model or {}).get("available_daily_risk_budget"))},
            {"Item": "Execute target profit", "Value": _money((target_model or {}).get("execute_target_profit"))},
            {"Item": "Execute max loss", "Value": _money((target_model or {}).get("execute_max_loss"))},
            {"Item": "Risk needed for daily target", "Value": _money((target_model or {}).get("risk_required_for_daily_target"))},
            {"Item": "Target feasibility", "Value": (target_model or {}).get("target_feasibility", "not_available")},
            {"Item": "Binding constraint", "Value": (target_model or {}).get("binding_constraint", "not_available")},
            {"Item": "Report mode", "Value": run_mode},
            {"Item": "Data-quality status", "Value": f"{_quality_icon(data_status)} {data_status}"},
            {
                "Item": "Market regime",
                "Value": f"{regime.get('trend')} / {regime.get('volatility')} vol / {regime.get('flow')} flow / sizing {regime.get('sizing_stance')}",
            },
            {"Item": "Portfolio risk", "Value": _portfolio_risk_summary(portfolio)},
            {"Item": "Top action today", "Value": _top_action_today(final, watch, portfolio, data_quality)},
            {"Item": "Execute Now count", "Value": int(len(final))},
            {"Item": "Enter Only At Price count", "Value": int(len(watch))},
            {"Item": "Portfolio Income count", "Value": int(income_count)},
            {"Item": "Watch count", "Value": int(len(watch_alerts))},
    ]
    return pd.DataFrame(rows)


def _target_model_frame(target_model: dict[str, Any] | None) -> pd.DataFrame:
    if not target_model:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {"Metric": "Monthly target", "Value": _money(target_model.get("monthly_profit_target"))},
            {"Metric": "MTD realized P/L", "Value": _money(target_model.get("month_to_date_realized_pnl"))},
            {"Metric": "Remaining target", "Value": _money(target_model.get("remaining_monthly_target"))},
            {"Metric": "Business days remaining", "Value": target_model.get("business_days_remaining")},
            {"Metric": "Required daily P/L", "Value": _money(target_model.get("required_daily_pnl"))},
            {"Metric": "Available daily risk", "Value": _money(target_model.get("available_daily_risk_budget"))},
            {"Metric": "Execute target profit", "Value": _money(target_model.get("execute_target_profit"))},
            {"Metric": "Execute max loss", "Value": _money(target_model.get("execute_max_loss"))},
            {"Metric": "Monthly run-rate from Execute", "Value": _money(target_model.get("monthly_run_rate_from_execute"))},
            {"Metric": "One-lot target profit", "Value": _money(target_model.get("one_lot_target_profit"))},
            {"Metric": "Risk required for daily target", "Value": _money(target_model.get("risk_required_for_daily_target"))},
            {"Metric": "Risk gap for daily target", "Value": _money(target_model.get("risk_gap_for_daily_target"))},
            {
                "Metric": "Best setup contracts needed",
                "Value": (
                    f"{target_model.get('best_setup_ticker')}: {int(safe_float(target_model.get('best_setup_contracts_required')))} contracts"
                    if math.isfinite(safe_float(target_model.get("best_setup_contracts_required")))
                    else "n/a"
                ),
            },
            {
                "Metric": "Best setup contracts selected",
                "Value": (
                    f"{target_model.get('best_setup_ticker')}: {int(safe_float(target_model.get('best_setup_selected_contracts')))} contracts"
                    if math.isfinite(safe_float(target_model.get("best_setup_selected_contracts")))
                    else "n/a"
                ),
            },
            {"Metric": "Best setup risk needed", "Value": _money(target_model.get("best_setup_risk_required"))},
            {"Metric": "Feasibility", "Value": target_model.get("target_feasibility")},
            {"Metric": "Binding constraint", "Value": target_model.get("binding_constraint")},
        ]
    )


def _watch_alert_rows(research: pd.DataFrame, avoid: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    pool = research.copy()
    if pool.empty:
        return pd.DataFrame()
    rows = []
    for _, row in _diversified_report_rows(pool, limit=limit, max_per_ticker=1).iterrows():
        trigger = _compact_trigger_text(row)
        blocker = _clean_note(row.get("primary_blocker")) or _compact_reason_text(row)
        rows.append(
            {
                "Status": "🔵 Watch",
                "Ticker": row.get("ticker"),
                "Flow": row.get("flow_quality", ""),
                "Price Condition": trigger,
                "Alert Trigger": f"{trigger}; blocker must clear: {blocker}",
                "Action If Triggered": "Rerun Schwab/news checks; move to Enter Only At Price only if data gate and blocker clear.",
            }
        )
    return pd.DataFrame(rows)


def _near_miss_rows(research: pd.DataFrame, avoid: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    pool = research.copy()
    if pool.empty and not avoid.empty:
        pool = avoid.copy()
    if pool.empty:
        return pd.DataFrame()
    rows = []
    for _, row in _diversified_report_rows(pool, limit=limit, max_per_ticker=1).iterrows():
        rows.append(
            {
                "Status": _status_label(str(row.get("trade_status") or "Research")),
                "Ticker": row.get("ticker"),
                "Trade": _compact_trade_label(row),
                "Flow": row.get("flow_quality", ""),
                "Current": _compact_entry_text(row),
                "Valid Only If": _compact_trigger_text(row),
                "Blocker": _clean_note(row.get("primary_blocker")) or _compact_reason_text(row),
            }
        )
    return pd.DataFrame(rows)


def _tactical_debit_rows(final: pd.DataFrame, watch: pd.DataFrame, research: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    frames = [frame for frame in [final, watch, research] if not frame.empty]
    if not frames:
        return pd.DataFrame()
    pool = pd.concat(frames, ignore_index=True, sort=False)
    pool = pool[pool.apply(_is_debit_strategy, axis=1)].copy()
    if pool.empty:
        return pd.DataFrame()
    rows = []
    for _, row in _diversified_report_rows(pool, limit=limit, max_per_ticker=1).iterrows():
        rows.append(
            {
                "Status": _status_label(str(row.get("trade_status") or ("Execute" if "rank" in row else "Watch"))),
                "Ticker": row.get("ticker"),
                "Trade": _compact_trade_label(row),
                "Entry": _compact_entry_text(row),
                "Max Loss": _money(row.get("max_loss")),
                "Reward/Risk": f"{safe_float(row.get('reward_risk')):.2f}" if math.isfinite(safe_float(row.get("reward_risk"))) else "",
                "Exit Plan": row.get("exit_plan", _clean_note(row.get("trigger"))),
                "Why": _compact_reason_text(row),
            }
        )
    return pd.DataFrame(rows)


def _live_outcome_summary_text(live_outcomes: dict[str, Any] | None) -> str:
    if not live_outcomes or live_outcomes.get("status") != "ok":
        return str((live_outcomes or {}).get("reason") or (live_outcomes or {}).get("status") or "not_checked")
    parts = []
    for family, summary in (live_outcomes.get("family_summary") or {}).items():
        parts.append(
            f"{family}: {summary.get('expectancy')} avg ${safe_float(summary.get('avg_pnl')):.0f} over {summary.get('outcomes')} outcomes"
        )
    return "; ".join(parts[:5]) if parts else "no realized family outcomes"


def _change_frame(change_summary: dict[str, Any] | None) -> pd.DataFrame:
    if not change_summary:
        return pd.DataFrame()
    rows = change_summary.get("rows") or []
    return pd.DataFrame(rows)


def build_intraday_change_summary(
    *,
    out_dir: Path,
    asof: dt.date,
    scored: pd.DataFrame,
    final: pd.DataFrame,
    watch: pd.DataFrame,
    portfolio: dict[str, Any] | None,
    risk_budget: float,
) -> dict[str, Any]:
    candidates = []
    for path in sorted(out_dir.parent.glob(f"codexuw_daily*{asof}*/codexuw_scored_{asof}.csv")):
        if path.parent.resolve() == out_dir.resolve():
            continue
        candidates.append(path)
    if not candidates:
        return {"status": "unavailable", "summary": "No prior report found for this date.", "rows": []}
    prior_path = max(candidates, key=lambda path: path.stat().st_mtime)
    try:
        prior = pd.read_csv(prior_path)
    except Exception as exc:
        return {"status": "unavailable", "summary": f"Prior report could not be read: {exc}", "rows": []}

    current_pool = pd.concat([frame for frame in [final, watch, scored.head(20)] if not frame.empty], ignore_index=True, sort=False)
    if current_pool.empty:
        return {"status": "ok", "prior_scored": str(prior_path), "summary": "No current rows to compare.", "rows": []}

    def key(row: pd.Series) -> str:
        return "|".join(
            [
                str(row.get("ticker") or ""),
                str(row.get("strategy") or ""),
                str(row.get("expiry") or row.get("expiration_date") or ""),
                str(row.get("short_leg") or row.get("sell_leg") or ""),
                str(row.get("long_leg") or row.get("buy_leg") or ""),
            ]
        )

    prior = prior.copy()
    prior["_change_key"] = prior.apply(key, axis=1)
    prior_by_key = {str(row["_change_key"]): row for _, row in prior.iterrows()}
    rows = []
    seen: set[str] = set()
    for _, row in current_pool.iterrows():
        change_key = key(row)
        if not change_key or change_key in seen:
            continue
        seen.add(change_key)
        old = prior_by_key.get(change_key)
        if old is None:
            rows.append(
                {
                    "Ticker": row.get("ticker"),
                    "Change": "new candidate",
                    "Underlying": f"now {_money(row.get('stock_price_live'))}",
                    "Option Quote": _compact_entry_text(row),
                    "Bid/Ask Width": _pct(row.get("quote_width_pct")),
                    "Liquidity": f"short {safe_float(row.get('short_oi'), 0) + safe_float(row.get('short_volume'), 0):.0f}; long {safe_float(row.get('long_oi'), 0) + safe_float(row.get('long_volume'), 0):.0f}",
                    "Flow/OI": f"{row.get('flow_quality', '')}; OI {row.get('oi_carryover_status', '')}",
                    "News": row.get("catalyst_status", ""),
                    "Portfolio": _clean_note(row.get("portfolio_note")),
                    "Risk Budget": f"${risk_budget:,.0f}",
                }
            )
            continue
        old_entry = _compact_entry_text(old)
        new_entry = _compact_entry_text(row)
        old_width = safe_float(old.get("quote_width_pct"))
        new_width = safe_float(row.get("quote_width_pct"))
        rows.append(
            {
                "Ticker": row.get("ticker"),
                "Change": "updated",
                "Underlying": f"{_money(old.get('stock_price_live'))} -> {_money(row.get('stock_price_live'))}",
                "Option Quote": f"{old_entry} -> {new_entry}",
                "Bid/Ask Width": f"{_pct(old_width)} -> {_pct(new_width)}",
                "Liquidity": (
                    f"short {safe_float(old.get('short_oi'), 0) + safe_float(old.get('short_volume'), 0):.0f}->"
                    f"{safe_float(row.get('short_oi'), 0) + safe_float(row.get('short_volume'), 0):.0f}; "
                    f"long {safe_float(old.get('long_oi'), 0) + safe_float(old.get('long_volume'), 0):.0f}->"
                    f"{safe_float(row.get('long_oi'), 0) + safe_float(row.get('long_volume'), 0):.0f}"
                ),
                "Flow/OI": f"{old.get('flow_quality', '')}/{old.get('oi_carryover_status', '')} -> {row.get('flow_quality', '')}/{row.get('oi_carryover_status', '')}",
                "News": f"{old.get('catalyst_status', '')} -> {row.get('catalyst_status', '')}",
                "Portfolio": _clean_note(row.get("portfolio_note")) or _portfolio_risk_summary(portfolio),
                "Risk Budget": f"${risk_budget:,.0f}",
            }
        )
        if len(rows) >= 10:
            break
    return {
        "status": "ok",
        "prior_scored": str(prior_path),
        "summary": f"Compared against {prior_path.parent.name}.",
        "rows": rows,
    }


def _write_compact_daily_report(
    *,
    out_dir: Path,
    asof: dt.date,
    run_mode: str,
    data_quality: dict[str, Any] | None,
    change_summary: dict[str, Any] | None,
    live_outcomes: dict[str, Any] | None,
    regime: dict[str, Any],
    scored: pd.DataFrame,
    final: pd.DataFrame,
    watch: pd.DataFrame,
    research: pd.DataFrame,
    avoid: pd.DataFrame,
    funnel: dict[str, int],
    portfolio: dict[str, Any] | None,
    recent_performance: dict[str, Any] | None,
    target_model: dict[str, Any] | None = None,
) -> Path:
    report = out_dir / f"codexuw_trade_report_{asof}.md"
    action_rows = _compact_action_rows(final, watch, research)
    watch_rows = _compact_watch_rows(watch)
    watch_alerts = _watch_alert_rows(research, avoid)
    near_misses = _near_miss_rows(research, avoid)
    tactical_debits = _tactical_debit_rows(final, watch, research)
    manage_actions = _portfolio_actions_frame(portfolio, lane="Manage Existing Risk")
    income_actions = _portfolio_actions_frame(portfolio, lane="Portfolio Income")
    data_quality_rows = _data_quality_frame(data_quality)
    first_screen = _first_screen_frame(
        run_mode=run_mode,
        data_quality=data_quality,
        regime=regime,
        portfolio=portfolio,
        final=final,
        watch=watch,
        watch_alerts=watch_alerts,
        target_model=target_model,
    )
    target_rows = _target_model_frame(target_model)
    rej = rejection_summary(scored).head(10)
    decisions = decision_summary(scored).head(8)
    live_counts = scored["live_status"].fillna("unknown").value_counts().to_dict() if not scored.empty and "live_status" in scored.columns else {}
    flow_counts = scored["flow_quality"].fillna("unknown").value_counts().to_dict() if not scored.empty and "flow_quality" in scored.columns else {}
    oi_counts = scored["oi_carryover_status"].fillna("unknown").value_counts().to_dict() if not scored.empty and "oi_carryover_status" in scored.columns else {}
    oi_source = ""
    if not scored.empty and "oi_source_file" in scored.columns:
        source_values = scored["oi_source_file"].dropna().astype(str)
        source_values = source_values[source_values.ne("")]
        if not source_values.empty:
            oi_source = _clean_note(source_values.iloc[0])

    lines = [
        f"# {PIPELINE_NAME} - Daily Decision Engine - {asof}",
        "",
        "## First Screen",
        "",
        first_screen.to_markdown(index=False),
        "",
    ]
    if data_quality_rows.empty:
        lines.extend(["## Data Quality Gate", "", "_No data-quality status available._", ""])
    else:
        lines.extend(["## Data Quality Gate", "", data_quality_rows.to_markdown(index=False), ""])
    if not target_rows.empty:
        lines.extend(["## Target Feasibility", "", target_rows.to_markdown(index=False), ""])
        if (target_model or {}).get("target_feasibility") in {"infeasible", "stretched"}:
            lines.extend(
                [
                    f"Target status: {(target_model or {}).get('target_feasibility')}. Binding constraint: {(target_model or {}).get('binding_constraint')}.",
                    "",
                ]
            )
    if final.empty:
        lines.extend(
            [
                "**No high-quality Execute trades today.** The best action may still be risk management, resting conditional limits, income review, alerts, or standing down.",
                "",
            ]
        )
    elif len(final) == 1:
        lines.extend(["Only one setup cleared Execute. Conditional rows below are not market orders.", ""])

    lines.extend(["## Action Board", ""])
    if action_rows.empty:
        lines.extend(["_No candidates produced._", ""])
    else:
        lines.extend([action_rows.where(pd.notna(action_rows), "").to_markdown(index=False), ""])

    lines.extend(["## 1. Manage Existing Risk", ""])
    if manage_actions.empty:
        lines.extend(["_No open Schwab position action required by the current heuristic review._", ""])
    else:
        lines.extend([manage_actions.where(pd.notna(manage_actions), "").head(10).to_markdown(index=False), ""])

    lines.extend(["## 2. Execute Now", ""])
    if not final.empty:
        rows = []
        for _, row in final.sort_values("rank").iterrows():
            rows.append(
                {
                    "Status": _status_label("Execute"),
                    "Rank": int(safe_float(row.get("rank"), 0)),
                    "Ticker": row.get("ticker"),
                    "Trade": _compact_trade_label(row),
                    "Legs": f"{_leg_label(row.get('sell_leg'))} / {_leg_label(row.get('buy_leg'))}",
                    "Expiry": row.get("expiration_date", row.get("expiry")),
                    "Entry": _compact_entry_text(row),
                    "Max Profit": _money(row.get("max_profit")),
                    "Max Loss": _money(row.get("max_loss")),
                    "Contracts": int(safe_float(row.get("contracts"), 0)) if math.isfinite(safe_float(row.get("contracts"))) else "",
                    "Position Max Loss": _money(row.get("position_max_loss")),
                    "Target Profit": _money(row.get("target_profit_total")),
                    "EV": _money(row.get("expected_value_total")),
                    "% Monthly Target": _pct(row.get("target_contribution_pct")),
                    "Breakeven": f"{safe_float(row.get('breakeven')):.2f}" if math.isfinite(safe_float(row.get("breakeven"))) else "",
                    "POP": _pct(row.get("pop_delta_proxy")),
                    "Score": f"{safe_float(row.get('score')):.2f}" if math.isfinite(safe_float(row.get("score"))) else "",
                    "Confidence Components": row.get("confidence_components", ""),
                    "Exit": row.get("exit_plan"),
                }
            )
        lines.extend([pd.DataFrame(rows).to_markdown(index=False), ""])
    else:
        lines.extend(["_No Execute trades. Data gate, live pricing, catalyst, liquidity, or expectancy blocked new exposure._", ""])

    lines.extend(["## 3. Enter Only At Price", ""])
    if watch_rows.empty:
        lines.extend(["_No conditional entry orders._", ""])
    else:
        lines.extend(
            [
                "These are not Execute trades. Work only at the trigger price after a fresh Schwab recheck and unchanged data gate.",
                "",
                watch_rows.where(pd.notna(watch_rows), "").to_markdown(index=False),
                "",
            ]
        )

    lines.extend(["## 4. Portfolio Income", ""])
    if income_actions.empty:
        lines.extend(["_No covered-income action surfaced from current Schwab positions._", ""])
    else:
        lines.extend([income_actions.where(pd.notna(income_actions), "").head(8).to_markdown(index=False), ""])

    lines.extend(["## 5. Tactical Debit Setups", ""])
    if tactical_debits.empty:
        lines.extend(["_No tactical debit setup clears the small-risk lane._", ""])
    else:
        lines.extend([tactical_debits.where(pd.notna(tactical_debits), "").to_markdown(index=False), ""])

    lines.extend(["## 6. Watch With Alert", ""])
    if watch_alerts.empty:
        lines.extend(["_No monitor-ready alert rows._", ""])
    else:
        lines.extend([watch_alerts.where(pd.notna(watch_alerts), "").to_markdown(index=False), ""])

    lines.extend(["## 7. Near Misses", ""])
    if near_misses.empty:
        lines.extend(["_No near misses to monitor._", ""])
    else:
        lines.extend([near_misses.where(pd.notna(near_misses), "").to_markdown(index=False), ""])

    lines.extend(["## 8. Avoid / Research", ""])
    research_counts = research["flow_quality"].fillna("unknown").value_counts().to_dict() if not research.empty and "flow_quality" in research.columns else {}
    avoid_counts = avoid["flow_quality"].fillna("unknown").value_counts().to_dict() if not avoid.empty and "flow_quality" in avoid.columns else {}
    lines.extend(
        [
            f"- Research flow quality: {research_counts if research_counts else 'none'}",
            f"- Avoid flow quality: {avoid_counts if avoid_counts else 'none'}",
            "- Flow labels are directional, hedge, roll, spread_leg, or unclear; non-directional flow cannot be primary Execute evidence.",
            "",
        ]
    )

    change_rows = _change_frame(change_summary)
    if str(run_mode).lower().startswith("intraday"):
        lines.extend(["## Intraday Changes", ""])
        if change_rows.empty:
            lines.extend([str((change_summary or {}).get("summary") or "No prior report found for this date."), ""])
        else:
            lines.extend([change_rows.where(pd.notna(change_rows), "").head(10).to_markdown(index=False), ""])

    lines.extend(["## Why Not More Trades", ""])
    if decisions.empty:
        lines.append("_No decision counts._")
    else:
        lines.extend(["Decision gates:", "", decisions.to_markdown(index=False)])
    lines.append("")
    if rej.empty:
        lines.append("_No rejection counts._")
    else:
        lines.extend(["Top blockers:", "", rej.to_markdown(index=False)])
    lines.append("")

    perf = recent_performance or {}
    perf_text = (
        f"{perf.get('stance')} / win {safe_float(perf.get('win_rate')):.1%} / avg ${safe_float(perf.get('avg_pnl_1x')):.2f}"
        if perf.get("status") == "ok"
        else str(perf.get("status", "not_checked"))
    )
    portfolio_text = "not_checked"
    if portfolio and portfolio.get("status") == "ok":
        portfolio_text = (
            f"positions {portfolio.get('position_count', 0)}, "
            f"account ${safe_float(portfolio.get('total_value'), 0):,.0f}, "
            f"cash ${safe_float(portfolio.get('cash'), 0):,.0f}"
        )
    context = pd.DataFrame(
        [
            {"Item": "Regime", "Value": f"{regime.get('trend')} / {regime.get('volatility')} vol / {regime.get('flow')} flow"},
            {"Item": "Schwab Live", "Value": str(live_counts or "not available")},
            {"Item": "Flow Quality", "Value": str(flow_counts or "not available")},
            {"Item": "OI Carryover", "Value": str(oi_counts or "not available")},
            {"Item": "OI Source", "Value": oi_source or "not available"},
            {"Item": "Portfolio", "Value": portfolio_text},
            {"Item": "Recent Performance", "Value": perf_text},
            {"Item": "Live Outcome Calibration", "Value": _live_outcome_summary_text(live_outcomes)},
            {"Item": "Funnel", "Value": str(funnel)},
        ]
    )
    lines.extend(["## Run Context", "", context.to_markdown(index=False), ""])

    artifacts = pd.DataFrame(
        [
            {"Artifact": "Execute CSV", "Path": str(out_dir / f"codexuw_execute_trades_{asof}.csv")},
            {"Artifact": "Watch CSV", "Path": str(out_dir / f"codexuw_watch_trades_{asof}.csv")},
            {"Artifact": "Research CSV", "Path": str(out_dir / f"codexuw_research_candidates_{asof}.csv")},
            {"Artifact": "Avoid CSV", "Path": str(out_dir / f"codexuw_avoid_trades_{asof}.csv")},
            {"Artifact": "Scored CSV", "Path": str(out_dir / f"codexuw_scored_{asof}.csv")},
        ]
    )
    lines.extend(["## Artifacts", "", artifacts.to_markdown(index=False), ""])
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def write_outputs(
    *,
    out_dir: Path,
    asof: dt.date,
    run_mode: str = "Intraday live execution",
    data_quality: dict[str, Any] | None = None,
    change_summary: dict[str, Any] | None = None,
    live_outcomes: dict[str, Any] | None = None,
    regime: dict[str, Any],
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    final: pd.DataFrame,
    funnel: dict[str, int],
    portfolio: dict[str, Any] | None = None,
    catalysts: pd.DataFrame | None = None,
    recent_performance: dict[str, Any] | None = None,
    watchlist: pd.DataFrame | None = None,
    max_final_trades: int | None = None,
    run_provenance: dict[str, Any] | None = None,
    target_model: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / f"codexuw_candidates_{asof}.csv", index=False)
    scored.to_csv(out_dir / f"codexuw_scored_{asof}.csv", index=False)
    watch = watchlist if watchlist is not None else pd.DataFrame()
    watch_default_cols = [
        "rank",
        "status",
        "watch_kind",
        "ticker",
        "direction",
        "strategy",
        "sell_leg",
        "buy_leg",
        "expiry",
        "dte",
        "current_entry",
        "target_entry",
        "limit_order",
        "no_chase_threshold",
        "what_must_improve",
        "fresh_schwab_recheck_required",
        "trigger",
        "reason",
        "edge_verdict",
        "edge_sample_size",
        "edge_win_rate",
        "edge_avg_pnl",
        "edge_match_level",
        "price_annotation",
        "construction_source",
        "portfolio_risk",
        "primary_blocker",
    ]
    if watch.empty and len(watch.columns) == 0:
        watch = pd.DataFrame(columns=watch_default_cols)
    final.to_csv(out_dir / f"codexuw_final_trades_{asof}.csv", index=False)
    final.to_csv(out_dir / f"codexuw_execute_trades_{asof}.csv", index=False)
    outcome_ledger_path = _write_execute_outcome_ledger(out_dir, asof, final)
    watch.to_csv(out_dir / f"codexuw_entry_watchlist_{asof}.csv", index=False)
    watch.to_csv(out_dir / f"codexuw_watch_trades_{asof}.csv", index=False)
    recommendation_ledger_path = _write_recommendation_outcome_ledger(out_dir, asof, final, watch)
    if "trade_status" in scored.columns:
        research = scored[scored["trade_status"].astype(str).eq("Research")].copy()
        avoid = scored[scored["trade_status"].astype(str).eq("Avoid")].copy()
    else:
        research = scored.iloc[0:0].copy()
        avoid = scored[scored.get("hard_rejects", pd.Series("", index=scored.index)).fillna("").ne("")].copy() if not scored.empty else scored.copy()
    research.to_csv(out_dir / f"codexuw_research_candidates_{asof}.csv", index=False)
    avoid.to_csv(out_dir / f"codexuw_avoid_trades_{asof}.csv", index=False)
    alt_cols = [
        "ticker",
        "direction",
        "strategy",
        "expiry",
        "dte",
        "construction_source",
        "construction_reason",
        "anchor_strike",
        "short_leg",
        "long_leg",
        "short_strike",
        "long_strike",
        "spread_width",
        "credit",
        "debit",
        "target_entry",
        "required_entry",
        "no_chase_threshold",
        "expected_move_ratio",
        "breakeven_distance_pct",
        "reward_risk",
        "credit_pct_width",
        "debit_pct_width",
        "liquidity_summary",
        "live_status",
        "quote_width_pct",
    ]
    spread_alternatives = scored[[c for c in alt_cols if c in scored.columns]].copy() if not scored.empty else pd.DataFrame(columns=alt_cols)
    spread_alternatives.to_csv(out_dir / f"codexuw_spread_construction_alternatives_{asof}.csv", index=False)
    edge_cols = [
        "ticker",
        "direction",
        "strategy",
        "expiry",
        "construction_source",
        "trade_status",
        "replay_ev_verdict",
        *EDGE_COLUMNS,
        "primary_blocker",
    ]
    edge_audit = scored[[c for c in edge_cols if c in scored.columns]].copy() if not scored.empty else pd.DataFrame(columns=edge_cols)
    edge_audit.to_csv(out_dir / f"codexuw_edge_model_audit_{asof}.csv", index=False)
    oi_cols = [
        "ticker",
        "strategy",
        "short_leg",
        "long_leg",
        "oi_carryover_status",
        "oi_carryover_reason",
        "short_leg_oi_change",
        "long_leg_oi_change",
        "short_leg_side_bias",
        "long_leg_side_bias",
        "oi_source_file",
        "short_leg_oi_context",
        "long_leg_oi_context",
    ]
    oi_details = scored[[c for c in oi_cols if c in scored.columns]].copy() if not scored.empty else pd.DataFrame(columns=oi_cols)
    oi_details.to_csv(out_dir / f"codexuw_oi_carryover_{asof}.csv", index=False)
    action_board = _action_board_rows(final, watch, research, avoid)
    action_board.to_csv(out_dir / f"codexuw_action_board_{asof}.csv", index=False)
    rejection_summary(scored).to_csv(out_dir / f"codexuw_rejections_{asof}.csv", index=False)
    if catalysts is not None and not catalysts.empty:
        catalysts.to_csv(out_dir / f"codexuw_catalysts_{asof}.csv", index=False)
    (out_dir / f"codexuw_manifest_{asof}.json").write_text(
        json.dumps(
            {
                "pipeline_name": PIPELINE_NAME,
                "pipeline_version": PIPELINE_VERSION,
                "version_lock": pipeline_version_record("v2"),
                "asof": str(asof),
                "report_mode": run_mode,
                "data_quality": data_quality or {},
                "intraday_change_summary": change_summary or {},
                "regime": regime,
                "funnel": funnel,
                "portfolio_status": (portfolio or {}).get("status", "not_checked"),
                "portfolio_position_count": (portfolio or {}).get("position_count", 0),
                "recent_performance": recent_performance or {"status": "not_checked"},
                "live_outcomes": live_outcomes or {"status": "not_checked"},
                "target_model": target_model or {},
                "entry_watchlist_rows": int(len(watch)),
                "execute_rows": int(len(final)),
                "watch_rows": int(len(watch)),
                "research_rows": int(len(research)),
                "avoid_rows": int(len(avoid)),
                "spread_alternative_rows": int(len(spread_alternatives)),
                "edge_audit_rows": int(len(edge_audit)),
                "max_final_trades": max_final_trades,
                "execute_outcome_ledger": str(outcome_ledger_path),
                "recommendation_outcome_ledger": str(recommendation_ledger_path),
                "run_provenance": run_provenance or {},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return _write_compact_daily_report(
        out_dir=out_dir,
        asof=asof,
        run_mode=run_mode,
        data_quality=data_quality,
        change_summary=change_summary,
        live_outcomes=live_outcomes,
        regime=regime,
        scored=scored,
        final=final,
        watch=watch,
        research=research,
        avoid=avoid,
        funnel=funnel,
        portfolio=portfolio,
        recent_performance=recent_performance,
        target_model=target_model,
    )
    report = out_dir / f"codexuw_trade_report_{asof}.md"
    lines = [
        f"# CodexUW Daily Options Income Report - {asof}",
        "",
        "## Action Board",
        "",
    ]
    if action_board.empty:
        lines.extend(["_No candidates produced._", ""])
    else:
        action_board = action_board.where(pd.notna(action_board), "")
        lines.extend([action_board.to_markdown(index=False), ""])
    lines.extend(
        [
        "## Funnel",
        ]
    )
    prev = None
    for name, count in funnel.items():
        drop = ""
        if name not in {"watch_rows", "research_rows", "avoid_rows"} and prev is not None and prev:
            if count <= prev:
                drop = f" ({(prev - count) / prev:.1%} drop)"
            else:
                drop = f" ({(count - prev) / prev:.1%} expansion)"
        lines.append(f"- {name}: {count}{drop}")
        if name not in {"watch_rows", "research_rows", "avoid_rows"}:
            prev = count
    lines.extend(
        [
            "",
            "## Regime Summary",
            f"- Volatility: {regime.get('volatility')}",
            f"- Trend: {regime.get('trend')}",
            f"- Flow: {regime.get('flow')}",
            f"- Transition risk: {regime.get('transition')}",
            f"- Sizing stance: {regime.get('sizing_stance')}",
        ]
    )
    validation_note = _clean_note(regime.get("validation_note"))
    if validation_note:
        lines.append(f"- Validation mode: {validation_note}")
    lines.extend(["", "## Portfolio Risk Notes"])
    if portfolio and portfolio.get("status") == "ok":
        large = portfolio.get("large_equity_exposure", {}) or {}
        lines.extend(
            [
                f"- Schwab positions: {portfolio.get('position_count', 0)}",
                f"- Account value: ${safe_float(portfolio.get('total_value'), 0.0):,.0f}",
                f"- Cash: ${safe_float(portfolio.get('cash'), 0.0):,.0f}",
                f"- Existing option underlyings annotated: {len(portfolio.get('option_underlyings', []) or [])}",
                f"- Large equity exposures: {', '.join(sorted(large)[:10]) if large else 'none'}",
                "",
            ]
        )
    else:
        lines.extend([f"- Status: {(portfolio or {}).get('status', 'not_checked')}", ""])
    if catalysts is not None and not catalysts.empty:
        counts = catalysts["catalyst_status"].fillna("unknown").value_counts().to_dict()
        lines.extend(
            [
                "## Catalyst Context",
                f"- Local browser/news capture status counts: {counts}",
                "",
            ]
        )
    flow_counts = scored["flow_quality"].fillna("unknown").value_counts().to_dict() if not scored.empty and "flow_quality" in scored.columns else {}
    lines.extend(
        [
            "## Flow Quality Summary",
            f"- UW flow classifications: {flow_counts if flow_counts else 'not available'}",
            "- Noisy UW flow is routed to Research unless other confirmations are strong.",
            "",
        ]
    )
    oi_counts = (
        scored["oi_carryover_status"].fillna("unknown").value_counts().to_dict()
        if not scored.empty and "oi_carryover_status" in scored.columns
        else {}
    )
    oi_source = _clean_note(scored["oi_source_file"].dropna().astype(str).iloc[0]) if not scored.empty and "oi_source_file" in scored.columns and scored["oi_source_file"].dropna().astype(str).any() else ""
    lines.extend(
        [
            "## OI Carryover Summary",
            f"- Status counts: {oi_counts if oi_counts else 'not available'}",
            f"- Source: {oi_source or 'not available'}",
            "",
        ]
    )
    live_counts = scored["live_status"].fillna("unknown").value_counts().to_dict() if not scored.empty and "live_status" in scored.columns else {}
    lines.extend(
        [
            "## Data / Schwab Validation Status",
            f"- Schwab live pricing status counts: {live_counts if live_counts else 'not available'}",
            f"- Scored candidates: {len(scored)}",
            f"- CSV artifacts: Execute, Watch, Research, Avoid, scored candidates, spread alternatives, edge audit, and OI carryover details written under {out_dir}",
            "",
        ]
    )
    lines.append("## Recent Performance")
    if recent_performance and recent_performance.get("status") == "ok":
        lines.extend(
            [
                f"- Source: {recent_performance.get('source', '')}",
                f"- Window: last {recent_performance.get('window')} decision/replay-selected trades",
                f"- Stance: {recent_performance.get('stance')}",
                f"- Win rate: {safe_float(recent_performance.get('win_rate')):.1%}",
                f"- Avg PnL/spread: ${safe_float(recent_performance.get('avg_pnl_1x')):,.2f}",
                "",
            ]
        )
    else:
        lines.extend([f"- Status: {(recent_performance or {}).get('status', 'not_checked')}", f"- Reason: {(recent_performance or {}).get('reason', '')}", ""])
    edge_counts = scored["edge_verdict"].fillna("unknown").value_counts().to_dict() if not scored.empty and "edge_verdict" in scored.columns else {}
    match_counts = scored["edge_match_level"].fillna("unknown").value_counts().to_dict() if not scored.empty and "edge_match_level" in scored.columns else {}
    lines.extend(
        [
            "## Replay Edge Model",
            f"- Edge verdict counts: {edge_counts if edge_counts else 'not available'}",
            f"- Match level counts: {match_counts if match_counts else 'not available'}",
            "- Proxy structure is not treated as executable EV; debit proxies require actual positive or acceptable replay edge before Execute.",
            "",
        ]
    )
    if final.empty:
        lines.extend(["## Final Decision", "", "No high-quality trades today", ""])
    else:
        cols = [
            "rank",
            "ticker",
            "direction",
            "strategy",
            "expiration_date",
            "sell_leg",
            "buy_leg",
            "dte",
            "credit",
            "debit",
            "spread_width",
            "entry_limit_credit",
            "entry_limit_debit",
            "target_close_debit",
            "stop_review_debit",
            "credit_pct_width",
            "debit_pct_width",
            "max_profit",
            "max_loss",
            "breakeven",
            "pop_delta_proxy",
            "score",
            "trade_conviction",
            "edge_verdict",
            "edge_sample_size",
            "edge_win_rate",
            "edge_avg_pnl",
            "edge_match_level",
            "price_annotation",
            "construction_source",
            "required_entry",
            "no_chase_threshold",
            "sizing_label",
            "sizing_rationale",
            "edge_summary",
            "catalyst_status",
            "risk_notes",
            "position_size",
            "exit_plan",
        ]
        display = final[[c for c in cols if c in final.columns]].copy()
        display = display.rename(
            columns={
                "rank": "Rank",
                "ticker": "Ticker",
                "direction": "Direction",
                "strategy": "Strategy",
                "expiration_date": "Expiration Date",
                "sell_leg": "Sell Leg (Short)",
                "buy_leg": "Buy Leg (Long)",
                "dte": "DTE",
                "credit": "Credit",
                "debit": "Debit",
                "spread_width": "Spread Width",
                "entry_limit_credit": "Entry Limit Credit",
                "entry_limit_debit": "Entry Limit Debit",
                "target_close_debit": "Target Close Debit",
                "stop_review_debit": "Stop / Review Debit",
                "credit_pct_width": "Credit % Width",
                "debit_pct_width": "Debit % Width",
                "max_profit": "Max Profit",
                "max_loss": "Max Loss",
                "breakeven": "Breakeven",
                "pop_delta_proxy": "POP / Delta Proxy",
                "score": "Score",
                "trade_conviction": "Trade Conviction",
                "edge_verdict": "Edge Verdict",
                "edge_sample_size": "Edge Sample Size",
                "edge_win_rate": "Edge Win Rate",
                "edge_avg_pnl": "Edge Avg P/L",
                "edge_match_level": "Edge Match Level",
                "price_annotation": "Price Annotation",
                "construction_source": "Construction Source",
                "required_entry": "Required Entry",
                "no_chase_threshold": "No Chase",
                "sizing_label": "Sizing Flag",
                "sizing_rationale": "Sizing Rationale",
                "edge_summary": "Edge / Thesis",
                "catalyst_status": "Catalyst Status",
                "risk_notes": "Risk Notes",
                "position_size": "Position Size",
                "exit_plan": "Exit Plan",
            }
        )
        for label in ["Sell Leg (Short)", "Buy Leg (Long)"]:
            if label in display.columns:
                display[label] = display[label].map(_leg_label)
        if "Trade Conviction" in display.columns:
            display["Trade Conviction"] = display.apply(
                lambda row: f"{_confidence_icon(str(row.get('Trade Conviction', '')).split(' ')[0])} {row.get('Trade Conviction', '')}",
                axis=1,
            )
        for col in ["credit_pct_width", "debit_pct_width", "pop_delta_proxy", "edge_win_rate"]:
            label = {
                "credit_pct_width": "Credit % Width",
                "debit_pct_width": "Debit % Width",
                "pop_delta_proxy": "POP / Delta Proxy",
                "edge_win_rate": "Edge Win Rate",
            }[col]
            if label in display.columns:
                display[label] = display[label].map(lambda x: f"{safe_float(x) * 100:.1f}%" if math.isfinite(safe_float(x)) else "")
        for col in [
            "credit",
            "debit",
            "spread_width",
            "entry_limit_credit",
            "entry_limit_debit",
            "target_close_debit",
            "stop_review_debit",
            "edge_avg_pnl",
            "required_entry",
            "no_chase_threshold",
            "max_profit",
            "max_loss",
            "breakeven",
            "score",
        ]:
            label = {
                "credit": "Credit",
                "debit": "Debit",
                "spread_width": "Spread Width",
                "entry_limit_credit": "Entry Limit Credit",
                "entry_limit_debit": "Entry Limit Debit",
                "target_close_debit": "Target Close Debit",
                "stop_review_debit": "Stop / Review Debit",
                "edge_avg_pnl": "Edge Avg P/L",
                "required_entry": "Required Entry",
                "no_chase_threshold": "No Chase",
                "max_profit": "Max Profit",
                "max_loss": "Max Loss",
                "breakeven": "Breakeven",
                "score": "Score",
            }[col]
            if label in display.columns:
                display[label] = display[label].map(lambda x: round(safe_float(x), 2) if math.isfinite(safe_float(x)) else "")
        lines.extend(["## Execute Trades", "", display.to_markdown(index=False), ""])
        lines.extend(["## Trade Playbook", ""])
        for _, row in final.sort_values("rank").iterrows():
            pop = safe_float(row.get("pop_delta_proxy"))
            pop_text = f"{pop:.1%}" if math.isfinite(pop) else "n/a"
            icon = _confidence_icon(row.get("confidence"))
            is_debit = _is_debit_strategy(row)
            leg_rows = pd.DataFrame(
                [
                    {
                        "Action": "SELL TO OPEN",
                        "Leg": _leg_label(row.get("sell_leg", row.get("short_leg"))),
                        "Leg Value": _leg_quote_summary(row, "sell_leg"),
                        "Purpose": "short premium leg",
                    },
                    {
                        "Action": "BUY TO OPEN",
                        "Leg": _leg_label(row.get("buy_leg", row.get("long_leg"))),
                        "Leg Value": _leg_quote_summary(row, "buy_leg"),
                        "Purpose": "defined-risk hedge leg",
                    },
                ]
            )
            entry_line = (
                f"- 🟢 Entry order: BUY TO OPEN spread for {_money(row.get('entry_limit_debit', row.get('debit')))} debit limit."
                if is_debit
                else f"- 🟢 Entry order: SELL TO OPEN spread for {_money(row.get('entry_limit_credit', row.get('credit')))} credit limit."
            )
            profit_line = (
                f"- 🟡 Profit target: SELL TO CLOSE spread near {_money(row.get('target_close_credit'))} credit."
                if is_debit
                else f"- 🟡 Profit target: BUY TO CLOSE spread at {_money(row.get('target_close_debit'))} debit."
            )
            stop_line = (
                f"- 🔴 Stop/review: review if spread value falls near {_money(row.get('stop_review_debit'))}."
                if is_debit
                else f"- 🔴 Stop/review: BUY TO CLOSE spread near {_money(row.get('stop_review_debit'))} debit."
            )
            net_line = (
                f"- 🔵 Net debit: {_money(row.get('debit'))}; width: {_money(row.get('spread_width'))}; breakeven: {safe_float(row.get('breakeven')):.2f}"
                if is_debit
                else f"- 🔵 Net credit: {_money(row.get('credit'))}; width: {_money(row.get('spread_width'))}; breakeven: {safe_float(row.get('breakeven')):.2f}"
            )
            lines.extend(
                [
                    f"### {icon} Rank {int(row.get('rank'))} - {row.get('ticker')} {row.get('strategy')}",
                    entry_line,
                    profit_line,
                    stop_line,
                    f"- 🔵 Expiration date: {row.get('expiration_date', row.get('expiry'))}; DTE: {row.get('dte')}",
                    f"- {icon} Trade conviction: {row.get('trade_conviction')}; Score: {safe_float(row.get('score')):.2f}; POP / delta proxy: {pop_text}",
                    f"- 🟣 Sizing: {row.get('sizing_label')}; {row.get('sizing_rationale')}",
                    f"- 🟣 Edge / thesis: {row.get('edge_summary')}",
                    "",
                    leg_rows.to_markdown(index=False),
                    "",
                    net_line,
                    f"- 🔴 Risk: max profit {_money(row.get('max_profit'))}; max loss {_money(row.get('max_loss'))}; {row.get('position_size')}",
                    f"- 🟡 Exit plan: {row.get('exit_plan')}",
                    "",
                ]
            )
        if len(final) < 3:
            lines.extend(
                [
                    "",
                    "## Final Decision",
                    "",
                    (
                        f"Only {len(final)} high-quality trade{'s' if len(final) != 1 else ''} passed. "
                        "No third trade was forced because the remaining candidates failed live credit, flow-alignment, "
                        "liquidity, earnings, or replay-guard checks."
                    ),
                    "",
                ]
            )
        total_risk = final["max_loss"].fillna(0).mul(final.get("contracts", 1)).sum()
        total_credit = final["max_profit"].fillna(0).mul(final.get("contracts", 1)).sum()
        sector = final.groupby(final["sector"].fillna("Unknown"))["max_loss"].sum().sort_values(ascending=False)
        balance = final["direction"].value_counts().to_dict()
        lines.extend(
            [
                "## Portfolio Summary",
                f"- Total max risk: ${total_risk:,.0f}",
                f"- Expected credit: ${total_credit:,.0f}",
                f"- Sector exposure: {', '.join(f'{k} ${v:,.0f}' for k, v in sector.items())}",
                f"- Bull/bear balance: {balance}",
                "",
            ]
        )
    if not watch.empty:
        watch_cols = [
            "rank",
            "status",
            "ticker",
            "direction",
            "sell_leg",
            "buy_leg",
            "expiry",
            "dte",
            "current_entry",
            "target_entry",
            "limit_order",
            "no_chase_threshold",
            "what_must_improve",
            "credit_pct_width",
            "target_pct_width",
            "pop_delta_proxy",
            "edge_verdict",
            "edge_sample_size",
            "edge_win_rate",
            "edge_avg_pnl",
            "edge_match_level",
            "construction_source",
            "score",
            "confidence",
            "quote_width_pct",
            "trigger",
        ]
        watch_display = watch[[c for c in watch_cols if c in watch.columns]].copy()
        watch_display = watch_display.rename(
            columns={
                "rank": "Rank",
                "status": "Status",
                "ticker": "Ticker",
                "direction": "Direction",
                "sell_leg": "Sell Leg",
                "buy_leg": "Buy Leg",
                "expiry": "Expiry",
                "dte": "DTE",
                "current_entry": "Current Entry",
                "target_entry": "Trigger Entry",
                "limit_order": "Limit Order",
                "no_chase_threshold": "No Chase",
                "what_must_improve": "What Must Improve",
                "credit_pct_width": "Current % Width",
                "target_pct_width": "Trigger % Width",
                "pop_delta_proxy": "POP / Delta Proxy",
                "edge_verdict": "Edge Verdict",
                "edge_sample_size": "Edge Sample Size",
                "edge_win_rate": "Edge Win Rate",
                "edge_avg_pnl": "Edge Avg P/L",
                "edge_match_level": "Edge Match Level",
                "construction_source": "Construction Source",
                "score": "Score",
                "confidence": "Conviction",
                "quote_width_pct": "Bid/Ask Width",
                "trigger": "Action Rule",
            }
        )
        for label in ["Sell Leg", "Buy Leg"]:
            if label in watch_display.columns:
                watch_display[label] = watch_display[label].map(_leg_label)
        for label in ["Current % Width", "Trigger % Width", "POP / Delta Proxy", "Bid/Ask Width", "Edge Win Rate"]:
            if label in watch_display.columns:
                watch_display[label] = watch_display[label].map(lambda x: f"{safe_float(x) * 100:.1f}%" if math.isfinite(safe_float(x)) else "")
        if "Score" in watch_display.columns:
            watch_display["Score"] = watch_display["Score"].map(lambda x: f"{safe_float(x):.2f}" if math.isfinite(safe_float(x)) else "")
        lines.extend(
            [
                "## Watch / Work Limit Orders",
                "",
                (
                    "These are not final trades. They passed hard safety gates, but current pricing or execution is not good enough. "
                    "Only place them if the trigger entry is available and a fresh Schwab recheck still passes."
                ),
                "",
                watch_display.to_markdown(index=False),
                "",
            ]
        )
    summary_cols = [
        "ticker",
        "strategy",
        "flow_quality",
        "expiry",
        "credit",
        "debit",
        "required_entry",
        "confirmation_score",
        "edge_verdict",
        "edge_sample_size",
        "edge_win_rate",
        "edge_avg_pnl",
        "edge_match_level",
        "price_annotation",
        "construction_source",
        "primary_blocker",
        "trade_status_reason",
    ]
    if not research.empty:
        research_display = research[[c for c in summary_cols if c in research.columns]].head(15).copy()
        research_display = research_display.rename(
            columns={
                "ticker": "Ticker",
                "strategy": "Strategy",
                "flow_quality": "Flow Quality",
                "expiry": "Expiry",
                "credit": "Credit",
                "debit": "Debit",
                "required_entry": "Required Entry",
                "confirmation_score": "Confirmation Score",
                "edge_verdict": "Edge Verdict",
                "edge_sample_size": "Edge Sample Size",
                "edge_win_rate": "Edge Win Rate",
                "edge_avg_pnl": "Edge Avg P/L",
                "edge_match_level": "Edge Match Level",
                "price_annotation": "Price Annotation",
                "construction_source": "Construction Source",
                "primary_blocker": "Primary Blocker",
                "trade_status_reason": "Reason",
            }
        )
        research_display = research_display.where(pd.notna(research_display), "")
        lines.extend(["## Research Candidates", "", research_display.to_markdown(index=False), ""])
    else:
        lines.extend(["## Research Candidates", "", "_No Research candidates._", ""])
    if not avoid.empty:
        avoid_display = avoid[[c for c in summary_cols if c in avoid.columns]].head(20).copy()
        avoid_display = avoid_display.rename(
            columns={
                "ticker": "Ticker",
                "strategy": "Strategy",
                "flow_quality": "Flow Quality",
                "expiry": "Expiry",
                "credit": "Credit",
                "debit": "Debit",
                "required_entry": "Required Entry",
                "confirmation_score": "Confirmation Score",
                "edge_verdict": "Edge Verdict",
                "edge_sample_size": "Edge Sample Size",
                "edge_win_rate": "Edge Win Rate",
                "edge_avg_pnl": "Edge Avg P/L",
                "edge_match_level": "Edge Match Level",
                "price_annotation": "Price Annotation",
                "construction_source": "Construction Source",
                "primary_blocker": "Primary Blocker",
                "trade_status_reason": "Reason",
            }
        )
        avoid_display = avoid_display.where(pd.notna(avoid_display), "")
        lines.extend(["## Avoid List", "", avoid_display.to_markdown(index=False), ""])
    else:
        lines.extend(["## Avoid List", "", "_No Avoid candidates._", ""])
    rej = rejection_summary(scored).head(12)
    decision_rej = decision_summary(scored).head(12)
    if not decision_rej.empty:
        lines.extend(["## High-Conviction Decision Gate", "", decision_rej.to_markdown(index=False), ""])
    lines.extend(["## Rejection Summary", ""])
    if rej.empty:
        lines.append("_No rejected candidates._")
    else:
        lines.append(rej.to_markdown(index=False))
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
