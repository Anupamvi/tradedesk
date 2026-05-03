from __future__ import annotations

import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .data import safe_float
from .occ import build_occ_symbol, parse_occ_symbol
from .performance import performance_min_score, performance_risk_multiplier
from .schwab_live import SchwabChainValidator, chain_spot, chain_to_contracts, find_best_credit_spread, price_width_bucket


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


def is_etf_row(row: pd.Series) -> bool:
    ticker = str(row.get("ticker") or "").upper().strip()
    issue_type = str(row.get("issue_type") or "").upper()
    name = str(row.get("full_name") or "").upper()
    return ticker in ETF_SYMBOL_SKIP or "ETF" in issue_type or "ETF" in name or " EXCHANGE TRADED " in name


def _earnings_days(row: pd.Series, asof: dt.date) -> float:
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
    return df.sort_values("_liq_rank", ascending=False).head(max_tickers).drop(columns=["_liq_rank"])


def _direction_list(row: pd.Series) -> list[str]:
    bias = safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), 0.0))
    total = safe_float(row.get("flow_total_premium"), 0.0)
    directions: list[str] = []
    if bias >= 0.025:
        directions.append("Bull Put")
    if bias <= -0.025:
        directions.append("Bear Call")
    if not directions and total >= 150_000_000 and abs(bias) < 0.04:
        directions = ["Bull Put", "Bear Call"]
    return directions


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
            right = "P" if direction == "Bull Put" else "C"
            opt = ticker_hot[ticker_hot["right"].eq(right)].copy()
            if opt.empty:
                continue
            if direction == "Bull Put":
                opt = opt[(opt["strike"] < close) & (((close - opt["strike"]) / close).between(0.015, 0.18))]
            else:
                opt = opt[(opt["strike"] > close) & (((opt["strike"] - close) / close).between(0.015, 0.18))]
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
                else:
                    target = close * 1.06
                exp_contracts["_target_dist"] = (exp_contracts["strike"] - target).abs()
                short = exp_contracts.sort_values(["_target_dist", "_liq"], ascending=[True, False]).iloc[0]
                width = _preferred_width(close)
                short_strike = safe_float(short.get("strike"))
                long_strike = short_strike - width if direction == "Bull Put" else short_strike + width
                short_bid = safe_float(short.get("bid"))
                est_credit = short_bid * 0.45 if math.isfinite(short_bid) else math.nan
                short_leg_eod = build_occ_symbol(ticker, expiry, right, short_strike)
                long_leg_eod = build_occ_symbol(ticker, expiry, right, long_strike)
                rows.append(
                    {
                        "ticker": ticker,
                        "sector": row.get("sector", ""),
                        "direction": direction,
                        "strategy": f"{direction} Credit Spread",
                        "expiry": expiry,
                        "dte": int((expiry - asof).days) if isinstance(expiry, dt.date) else math.nan,
                        "stock_price_eod": close,
                        "short_strike_eod": short_strike,
                        "long_strike_eod": long_strike,
                        "preferred_width": width,
                        "estimated_eod_credit": round(est_credit, 2) if math.isfinite(est_credit) else math.nan,
                        "flow_bias": safe_float(row.get("flow_bias"), 0.0),
                        "bot_flow_bias": safe_float(bot.loc[ticker].get("bot_flow_bias"), math.nan) if not bot.empty and ticker in bot.index else math.nan,
                        "combined_flow_bias": combined_bias,
                        "flow_total_premium": safe_float(row.get("flow_total_premium"), 0.0),
                        "iv_rank": safe_float(row.get("iv_rank")),
                        "iv30d": safe_float(row.get("iv30d")),
                        "implied_move_perc": safe_float(row.get("implied_move_perc")),
                        "next_earnings_dt": row.get("next_earnings_dt"),
                        "edge_type": _edge_text(direction, row, exp_contracts),
                        "source_contract": short.get("option_symbol", ""),
                        "short_leg_eod": short_leg_eod,
                        "long_leg_eod": long_leg_eod,
                        "source_contract_volume": safe_float(short.get("volume"), 0.0),
                        "source_contract_oi": safe_float(short.get("open_interest"), 0.0),
                    }
                )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_pre_score"] = (
        df["flow_total_premium"].clip(upper=1_000_000_000) / 1_000_000_000
        + df["combined_flow_bias"].abs().clip(upper=0.20) * 5.0
        + df["source_contract_volume"].clip(upper=50_000) / 50_000
        + df["source_contract_oi"].clip(upper=50_000) / 100_000
    )
    return df.sort_values("_pre_score", ascending=False).head(max_candidates).drop(columns=["_pre_score"])


def _score_trade(row: pd.Series, regime: dict[str, Any], asof: dt.date) -> tuple[float, str, list[str], list[str]]:
    hard: list[str] = []
    penalties: list[str] = []
    score = 0.0
    direction = str(row.get("direction", ""))
    bias = safe_float(row.get("combined_flow_bias"), 0.0)
    align = bias if direction == "Bull Put" else -bias
    total = safe_float(row.get("flow_total_premium"), 0.0)
    score += min(3.0, max(0.0, math.log10(max(total, 1.0)) - 6.5) + max(0.0, align) * 5.0)

    technical = 1.0
    if regime["trend"] == "uptrend" and direction == "Bull Put":
        technical += 0.8
    elif regime["trend"] == "downtrend" and direction == "Bear Call":
        technical += 0.8
    elif regime["trend"] == "range":
        technical += 0.4
    score += min(2.0, technical)

    iv_rank = safe_float(row.get("iv_rank"))
    credit_pct = safe_float(row.get("credit_pct_width"))
    vol_edge = 0.0
    if math.isfinite(iv_rank):
        vol_edge += min(1.0, max(0.0, iv_rank - 15.0) / 45.0)
    if math.isfinite(credit_pct):
        vol_edge += min(1.0, max(0.0, credit_pct - 0.12) / 0.14)
    score += min(2.0, vol_edge)

    distance = safe_float(row.get("distance_pct"))
    iv30d = safe_float(row.get("iv30d"))
    dte = safe_float(row.get("dte"))
    expected_move = iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and math.isfinite(dte) and dte > 0 else math.nan
    if math.isfinite(distance) and math.isfinite(expected_move) and expected_move > 0:
        score += min(2.0, max(0.0, distance / max(expected_move, 0.001)))
        if distance < expected_move * 0.55:
            penalties.append("too_close_to_expected_move")
            if direction == "Bull Put":
                penalties.append("replay_guard_bull_put_expected_move")
            score -= 1.0
    elif math.isfinite(distance):
        score += min(2.0, distance / 0.04)

    if math.isfinite(credit_pct) and credit_pct >= 0.20:
        score += 1.0
    elif math.isfinite(credit_pct) and credit_pct >= 0.16:
        score += 0.5
    else:
        penalties.append("credit_below_min_16pct_width")
        score -= 0.5

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

    earnings_days = _earnings_days(row, asof)
    if math.isfinite(earnings_days) and earnings_days <= 7:
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
        hard.append("no_flow_edge_alignment")
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
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    rows: list[dict[str, Any]] = []
    validator = None
    live_error = ""
    if require_live:
        try:
            validator = SchwabChainValidator(out_dir)
        except Exception as exc:
            live_error = str(exc)

    from_date = _next_weekday(max(dt.date.today(), asof))
    to_date = asof + dt.timedelta(days=50)
    for _, cand in candidates.iterrows():
        row = cand.to_dict()
        for key in [
            "credit_pct_width",
            "credit",
            "spread_width",
            "distance_pct",
            "quote_width_pct",
            "short_oi",
            "short_volume",
            "long_oi",
            "long_volume",
            "max_profit",
            "max_loss",
            "breakeven",
        ]:
            row.setdefault(key, math.nan)
        if validator is None:
            row.update({"live_status": "live_unavailable", "live_blocker": live_error or "Schwab validator disabled"})
        else:
            chain = validator.get_chain(str(cand["ticker"]), from_date=from_date, to_date=to_date)
            if not chain:
                row.update({"live_status": "chain_error", "live_blocker": validator.errors.get(str(cand["ticker"]), "chain fetch failed")})
            else:
                spot = chain_spot(chain)
                contracts = chain_to_contracts(chain)
                live = find_best_credit_spread(
                    contracts,
                    direction=str(cand["direction"]),
                    expiry=cand["expiry"],
                    spot=spot,
                    preferred_width=safe_float(cand.get("preferred_width"), math.nan),
                )
                row.update(live)
                row["stock_price_live"] = spot
                if live.get("live_status") == "PASS":
                    row["max_profit"] = safe_float(live.get("credit")) * 100.0
                    row["max_loss"] = (safe_float(live.get("spread_width")) - safe_float(live.get("credit"))) * 100.0
                    if str(cand["direction"]) == "Bull Put":
                        row["breakeven"] = safe_float(live.get("short_strike")) - safe_float(live.get("credit"))
                    else:
                        row["breakeven"] = safe_float(live.get("short_strike")) + safe_float(live.get("credit"))
        score, confidence, hard, penalties = _score_trade(pd.Series(row), regime, asof)
        row["score"] = score
        row["confidence"] = confidence
        row["hard_rejects"] = ";".join(hard)
        row["penalties"] = ";".join(penalties)
        pattern_pass, pattern = replay_quality_pattern(
            direction=str(row.get("direction", "")),
            trend=str(regime.get("trend", "")),
            credit_pct=safe_float(row.get("credit_pct_width")),
            distance_pct=safe_float(row.get("distance_pct")),
            expected_move=safe_float(row.get("iv30d")) * math.sqrt(safe_float(row.get("dte")) / 365.0)
            if safe_float(row.get("iv30d")) > 0 and safe_float(row.get("dte")) > 0
            else math.nan,
        )
        row["replay_pattern"] = pattern if pattern_pass else ""
        rows.append(row)
    if validator is not None:
        validator.save()
    return pd.DataFrame(rows)


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
    return bias if str(row.get("direction")) == "Bull Put" else -bias


def _expected_move_ratio(row: pd.Series) -> float:
    distance = safe_float(row.get("distance_pct"))
    iv30d = safe_float(row.get("iv30d"))
    dte = safe_float(row.get("dte"))
    expected_move = iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and math.isfinite(dte) and dte > 0 else math.nan
    return distance / expected_move if math.isfinite(distance) and math.isfinite(expected_move) and expected_move > 0 else math.nan


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
        elif math.isfinite(earnings_days) and 0 <= earnings_days <= 10:
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
        if ticker in option_underlyings:
            out.at[idx, "hard_rejects"] = _append_token(row.get("hard_rejects"), f"existing_option_exposure:{ticker}")
            out.at[idx, "portfolio_note"] = "Existing option exposure in Schwab account."
        elif ticker in large_equity:
            pct = large_equity[ticker] / total_value if total_value > 0 else 0.0
            out.at[idx, "penalties"] = _append_token(row.get("penalties"), f"large_existing_equity_exposure:{pct:.1%}")
            out.at[idx, "portfolio_note"] = f"Existing equity exposure {pct:.1%} of account."
            score = max(0.0, safe_float(row.get("score"), 0.0) - 0.5)
            out.at[idx, "score"] = round(score, 2)
            out.at[idx, "confidence"] = _confidence_from_score(score)
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


def select_final_trades(
    scored: pd.DataFrame,
    *,
    regime: dict[str, Any],
    risk_budget: float,
    recent_performance: dict[str, Any] | None = None,
    max_final_trades: int = 8,
) -> pd.DataFrame:
    if scored.empty:
        return scored
    required = {"hard_rejects", "credit_pct_width", "score", "penalties"}
    if not required.issubset(scored.columns):
        return scored.iloc[0:0].copy()
    min_score = performance_min_score(recent_performance, 5.0)
    relaxed_min_score = performance_min_score(recent_performance, 4.5)
    if "decision_eligible" in scored.columns:
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
    max_trade_risk = risk_budget * (0.22 if regime.get("sizing_stance") == "normal" else 0.14) * perf_mult
    for _, row in approved.iterrows():
        is_addon = bool(selected) and validated_addon_income_lane(row.get("direction"), safe_float(row.get("credit_pct_width")))
        if selected and not is_addon:
            continue
        max_loss = safe_float(row.get("max_loss"))
        if not math.isfinite(max_loss) or max_loss <= 0:
            continue
        confidence = str(row.get("confidence"))
        trade_budget = max_trade_risk if confidence == "High" else max_trade_risk * 0.55
        contracts = max(1, int(trade_budget // max_loss)) if confidence == "High" else 1
        risk = contracts * max_loss
        ticker = str(row.get("ticker"))
        sector = str(row.get("sector") or "Unknown")
        if total_risk + risk > risk_budget:
            continue
        if ticker_risk[ticker] + risk > risk_budget * 0.30:
            continue
        if sector_risk[sector] + risk > risk_budget * 0.55:
            continue
        if ticker in AI_TECH and ai_risk + risk > risk_budget * 0.55:
            continue
        out = row.copy()
        out["contracts"] = contracts
        out["sell_leg"] = row.get("short_leg")
        out["buy_leg"] = row.get("long_leg")
        out["expiration_date"] = row.get("expiry")
        out["trade_conviction"] = _trade_conviction(row)
        out["edge_summary"] = _edge_summary(row)
        out["selection_role"] = "validated add-on income lane" if is_addon else "strongest high-conviction setup"
        if contracts > 1:
            out["sizing_label"] = f"🟣 SIZE-UP: {contracts}-lot"
            out["sizing_rationale"] = (
                f"High confidence; ${max_loss:,.0f} max loss per contract fits ${trade_budget:,.0f} trade budget; "
                "ticker, sector, factor, and total daily risk caps still pass."
            )
        else:
            out["sizing_label"] = "1-lot base"
            out["sizing_rationale"] = (
                "Kept to 1-lot because confidence, risk budget, concentration, or liquidity did not justify a size-up."
            )
        out["position_size"] = f"{contracts} contract{'s' if contracts != 1 else ''}; max risk ${risk:,.0f}"
        credit = safe_float(row.get("credit"))
        if math.isfinite(credit) and credit > 0:
            take_profit_debit = credit * 0.40
            stop_debit = credit * 2.00
            out["entry_action"] = "SELL TO OPEN credit spread"
            out["entry_limit_credit"] = round(credit, 2)
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
        else:
            out["entry_action"] = ""
            out["entry_limit_credit"] = math.nan
            out["target_close_debit"] = math.nan
            out["stop_review_debit"] = math.nan
            out["sell_leg_action"] = "SELL TO OPEN"
            out["buy_leg_action"] = "BUY TO OPEN"
            out["close_action"] = "BUY TO CLOSE spread"
            out["take_profit"] = ""
            out["stop_loss"] = ""
            out["exit_plan"] = "Exit if live pricing invalidates the setup or short strike is threatened."
        notes = []
        for value in [row.get("penalties"), row.get("portfolio_note"), row.get("catalyst_note")]:
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
        if len(selected) >= max_final_trades:
            break
    if not selected:
        return pd.DataFrame()
    final = pd.DataFrame(selected).drop(columns=["_rank"], errors="ignore")
    final.insert(0, "rank", range(1, len(final) + 1))
    return final


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
        elif is_debit and math.isfinite(debit) and math.isfinite(target_debit) and debit > target_debit:
            watch_kind = "price_improvement_debit"
            current_entry = f"${debit:.2f} debit"
            target_entry = f"<= ${target_debit:.2f} debit"
            width = safe_float(row.get("spread_width"))
            if math.isfinite(width) and width > 0:
                target_pct = target_debit / width
            trigger = f"Wait for debit to fall to ${target_debit:.2f} or better; rerun Schwab chain before entry."
        elif "marginal_liquidity" in penalties or "wide_bid_ask" in penalties or decision_reason == "decision_marginal_live_liquidity":
            watch_kind = "execution_improvement"
            if is_credit and math.isfinite(credit):
                current_entry = f"${credit:.2f} credit"
                target_entry = f">= ${max(credit, target_credit) if math.isfinite(target_credit) else credit:.2f} credit with tighter quotes"
            elif is_debit and math.isfinite(debit):
                current_entry = f"${debit:.2f} debit"
                target_entry = f"<= ${min(debit, target_debit) if math.isfinite(target_debit) else debit:.2f} debit with tighter quotes"
            else:
                current_entry = "n/a"
                target_entry = "tighter two-sided market"
            trigger = "Wait for tighter bid/ask and real two-leg liquidity; rerun Schwab chain before entry."
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
            "trigger": trigger,
            "reason": ";".join(x for x in [decision_reason, penalties] if x),
            "risk_note": "Watch only. Do not enter unless trigger price is available and hard safety gates still pass.",
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


def write_outputs(
    *,
    out_dir: Path,
    asof: dt.date,
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
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out_dir / f"codexuw_candidates_{asof}.csv", index=False)
    scored.to_csv(out_dir / f"codexuw_scored_{asof}.csv", index=False)
    final.to_csv(out_dir / f"codexuw_final_trades_{asof}.csv", index=False)
    watch = watchlist if watchlist is not None else pd.DataFrame()
    watch.to_csv(out_dir / f"codexuw_entry_watchlist_{asof}.csv", index=False)
    rejection_summary(scored).to_csv(out_dir / f"codexuw_rejections_{asof}.csv", index=False)
    if catalysts is not None and not catalysts.empty:
        catalysts.to_csv(out_dir / f"codexuw_catalysts_{asof}.csv", index=False)
    (out_dir / f"codexuw_manifest_{asof}.json").write_text(
        json.dumps(
            {
                "asof": str(asof),
                "regime": regime,
                "funnel": funnel,
                "portfolio_status": (portfolio or {}).get("status", "not_checked"),
                "portfolio_position_count": (portfolio or {}).get("position_count", 0),
                "recent_performance": recent_performance or {"status": "not_checked"},
                "entry_watchlist_rows": int(len(watch)),
                "max_final_trades": max_final_trades,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    report = out_dir / f"codexuw_trade_report_{asof}.md"
    lines = [
        f"# CodexUW Daily Options Income Report - {asof}",
        "",
        "## Funnel",
    ]
    prev = None
    for name, count in funnel.items():
        drop = ""
        if prev is not None and prev:
            drop = f" ({(prev - count) / prev:.1%} drop)"
        lines.append(f"- {name}: {count}{drop}")
        prev = count
    lines.extend(
        [
            "",
            "## Regime",
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
    lines.extend(["", "## Portfolio Context"])
    if portfolio and portfolio.get("status") == "ok":
        large = portfolio.get("large_equity_exposure", {}) or {}
        lines.extend(
            [
                f"- Schwab positions: {portfolio.get('position_count', 0)}",
                f"- Account value: ${safe_float(portfolio.get('total_value'), 0.0):,.0f}",
                f"- Cash: ${safe_float(portfolio.get('cash'), 0.0):,.0f}",
                f"- Existing option underlyings blocked: {len(portfolio.get('option_underlyings', []) or [])}",
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
            "spread_width",
            "entry_limit_credit",
            "target_close_debit",
            "stop_review_debit",
            "credit_pct_width",
            "max_profit",
            "max_loss",
            "breakeven",
            "pop_delta_proxy",
            "score",
            "trade_conviction",
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
                "spread_width": "Spread Width",
                "entry_limit_credit": "Entry Limit Credit",
                "target_close_debit": "Target Close Debit",
                "stop_review_debit": "Stop / Review Debit",
                "credit_pct_width": "Credit % Width",
                "max_profit": "Max Profit",
                "max_loss": "Max Loss",
                "breakeven": "Breakeven",
                "pop_delta_proxy": "POP / Delta Proxy",
                "score": "Score",
                "trade_conviction": "Trade Conviction",
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
        for col in ["credit_pct_width", "pop_delta_proxy"]:
            label = {"credit_pct_width": "Credit % Width", "pop_delta_proxy": "POP / Delta Proxy"}[col]
            if label in display.columns:
                display[label] = display[label].map(lambda x: f"{safe_float(x) * 100:.1f}%" if math.isfinite(safe_float(x)) else "")
        for col in [
            "credit",
            "spread_width",
            "entry_limit_credit",
            "target_close_debit",
            "stop_review_debit",
            "max_profit",
            "max_loss",
            "breakeven",
            "score",
        ]:
            label = {
                "credit": "Credit",
                "spread_width": "Spread Width",
                "entry_limit_credit": "Entry Limit Credit",
                "target_close_debit": "Target Close Debit",
                "stop_review_debit": "Stop / Review Debit",
                "max_profit": "Max Profit",
                "max_loss": "Max Loss",
                "breakeven": "Breakeven",
                "score": "Score",
            }[col]
            if label in display.columns:
                display[label] = display[label].map(lambda x: round(safe_float(x), 2) if math.isfinite(safe_float(x)) else "")
        lines.extend(["## Final Trades", "", display.to_markdown(index=False), ""])
        lines.extend(["## Trade Playbook", ""])
        for _, row in final.sort_values("rank").iterrows():
            pop = safe_float(row.get("pop_delta_proxy"))
            pop_text = f"{pop:.1%}" if math.isfinite(pop) else "n/a"
            icon = _confidence_icon(row.get("confidence"))
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
            lines.extend(
                [
                    f"### {icon} Rank {int(row.get('rank'))} - {row.get('ticker')} {row.get('direction')} Credit Spread",
                    f"- 🟢 Entry order: SELL TO OPEN spread for {_money(row.get('entry_limit_credit', row.get('credit')))} credit limit.",
                    f"- 🟡 Profit target: BUY TO CLOSE spread at {_money(row.get('target_close_debit'))} debit.",
                    f"- 🔴 Stop/review: BUY TO CLOSE spread near {_money(row.get('stop_review_debit'))} debit.",
                    f"- 🔵 Expiration date: {row.get('expiration_date', row.get('expiry'))}; DTE: {row.get('dte')}",
                    f"- {icon} Trade conviction: {row.get('trade_conviction')}; Score: {safe_float(row.get('score')):.2f}; POP / delta proxy: {pop_text}",
                    f"- 🟣 Sizing: {row.get('sizing_label')}; {row.get('sizing_rationale')}",
                    f"- 🟣 Edge / thesis: {row.get('edge_summary')}",
                    "",
                    leg_rows.to_markdown(index=False),
                    "",
                    f"- 🔵 Net credit: {_money(row.get('credit'))}; width: {_money(row.get('spread_width'))}; breakeven: {safe_float(row.get('breakeven')):.2f}",
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
            "credit_pct_width",
            "target_pct_width",
            "pop_delta_proxy",
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
                "credit_pct_width": "Current % Width",
                "target_pct_width": "Trigger % Width",
                "pop_delta_proxy": "POP / Delta Proxy",
                "score": "Score",
                "confidence": "Conviction",
                "quote_width_pct": "Bid/Ask Width",
                "trigger": "Action Rule",
            }
        )
        for label in ["Sell Leg", "Buy Leg"]:
            if label in watch_display.columns:
                watch_display[label] = watch_display[label].map(_leg_label)
        for label in ["Current % Width", "Trigger % Width", "POP / Delta Proxy", "Bid/Ask Width"]:
            if label in watch_display.columns:
                watch_display[label] = watch_display[label].map(lambda x: f"{safe_float(x) * 100:.1f}%" if math.isfinite(safe_float(x)) else "")
        if "Score" in watch_display.columns:
            watch_display["Score"] = watch_display["Score"].map(lambda x: f"{safe_float(x):.2f}" if math.isfinite(safe_float(x)) else "")
        lines.extend(
            [
                "## Entry Watchlist - Wait For Better Price",
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
    rej = rejection_summary(scored).head(12)
    decision_rej = decision_summary(scored).head(12)
    if not decision_rej.empty:
        lines.extend(["## High-Conviction Decision Gate", "", decision_rej.to_markdown(index=False), ""])
    lines.extend(["## Rejected Candidate Summary", ""])
    if rej.empty:
        lines.append("_No rejected candidates._")
    else:
        lines.append(rej.to_markdown(index=False))
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report
