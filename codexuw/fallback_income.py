from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pandas as pd

from .data import safe_float
from .occ import build_occ_symbol
from .schwab_live import price_width_bucket


FALLBACK_SOURCE = "fallback_income"
FALLBACK_TARGET_CREDIT_PCT = 0.28


def _clean_ticker(value: object) -> str:
    return str(value or "").upper().strip()


def _direction_sign(direction: object) -> int:
    text = str(direction or "")
    if text in {"Bull Put", "Bull Call"}:
        return 1
    if text in {"Bear Call", "Bear Put"}:
        return -1
    return 0


def _flow_direction_to_credit_direction(value: object) -> str:
    text = str(value or "").lower().strip()
    if text == "bearish":
        return "Bear Call"
    return "Bull Put"


def _top_flow_frame(liquidity_shift: dict[str, Any] | None) -> pd.DataFrame:
    top = (liquidity_shift or {}).get("top_flow_universe")
    if isinstance(top, pd.DataFrame) and not top.empty:
        return top.copy()
    return pd.DataFrame()


def _stock_by_ticker(stock_screener: pd.DataFrame) -> dict[str, pd.Series]:
    if stock_screener.empty or "ticker" not in stock_screener.columns:
        return {}
    df = stock_screener.copy()
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return {str(row["ticker"]): row for _, row in df.iterrows()}


def _candidate_expiries(options: pd.DataFrame) -> list[dt.date]:
    if options.empty:
        return []
    exp = (
        options.groupby("expiry_dt", as_index=False)
        .agg(
            dte=("dte", "median"),
            liq=("open_interest", "sum"),
            vol=("volume", "sum"),
            rows=("option_symbol", "count"),
        )
        .sort_values(["dte", "liq", "vol"], ascending=[True, False, False])
    )
    exp = exp[exp["dte"].between(35, 70, inclusive="both")].copy()
    return [x for x in exp["expiry_dt"].tolist() if isinstance(x, dt.date)]


def _select_short_option(options: pd.DataFrame, *, close: float, direction: str) -> pd.Series | None:
    if options.empty or not math.isfinite(close) or close <= 0:
        return None
    opts = options.copy()
    opts["strike"] = pd.to_numeric(opts["strike"], errors="coerce")
    if direction == "Bull Put":
        opts["_distance"] = (close - opts["strike"]) / close
        opts = opts[opts["_distance"].between(0.18, 0.30, inclusive="both")]
        target_distance = 0.20
    else:
        opts["_distance"] = (opts["strike"] - close) / close
        opts = opts[opts["_distance"].between(0.16, 0.30, inclusive="both")]
        target_distance = 0.20
    if opts.empty:
        return None
    opts["_liq"] = pd.to_numeric(opts.get("open_interest"), errors="coerce").fillna(0) + pd.to_numeric(opts.get("volume"), errors="coerce").fillna(0)
    opts = opts[opts["_liq"] >= 500].copy()
    if opts.empty:
        return None
    opts["_target_dist"] = (opts["_distance"] - target_distance).abs()
    return opts.sort_values(["_target_dist", "_liq"], ascending=[True, False]).iloc[0]


def build_fallback_income_candidates(
    *,
    stock_screener: pd.DataFrame,
    hot_chains: pd.DataFrame,
    liquidity_shift: dict[str, Any] | None,
    asof: dt.date,
    max_candidates: int = 12,
) -> pd.DataFrame:
    """Build V1-style weekly income seeds without weakening Execute gates.

    These are only discovery rows. Schwab live validation and fallback status
    rules still decide whether a row is Execute, Scout/work-limit, or Research.
    """
    top = _top_flow_frame(liquidity_shift)
    if top.empty or stock_screener.empty or hot_chains.empty:
        return pd.DataFrame()
    stocks = _stock_by_ticker(stock_screener)
    rows: list[dict[str, Any]] = []
    top = top.head(50).copy()
    for _, flow in top.iterrows():
        ticker = _clean_ticker(flow.get("ticker"))
        if not ticker or ticker not in stocks:
            continue
        stock = stocks[ticker]
        close = safe_float(stock.get("close"))
        if not math.isfinite(close) or close <= 0:
            continue
        direction = _flow_direction_to_credit_direction(flow.get("flow_direction"))
        right = "P" if direction == "Bull Put" else "C"
        ticker_options = hot_chains[
            hot_chains["ticker"].astype(str).str.upper().eq(ticker)
            & hot_chains["right"].astype(str).eq(right)
        ].copy()
        for expiry in _candidate_expiries(ticker_options):
            exp_options = ticker_options[ticker_options["expiry_dt"].eq(expiry)].copy()
            source = _select_short_option(exp_options, close=close, direction=direction)
            if source is None:
                continue
            short_strike = safe_float(source.get("strike"))
            width = price_width_bucket(close)
            long_strike = short_strike - width if direction == "Bull Put" else short_strike + width
            strikes = {safe_float(value) for value in exp_options["strike"].dropna()}
            if long_strike not in strikes:
                continue
            dte = int((expiry - asof).days)
            distance_pct = (close - short_strike) / close if direction == "Bull Put" else (short_strike - close) / close
            iv30d = safe_float(stock.get("iv30d"))
            expected_move = iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and dte > 0 else math.nan
            target_entry = round(width * FALLBACK_TARGET_CREDIT_PCT, 2)
            rank_score = safe_float(flow.get("rank_score"), 0.0)
            vwap_confirmation = str(flow.get("vwap_confirmation") or "")
            flow_quality = "directional" if (
                (direction == "Bull Put" and vwap_confirmation.startswith("bullish_above"))
                or (direction == "Bear Call" and vwap_confirmation.startswith("bearish_below"))
            ) else "unclear"
            rows.append(
                {
                    "ticker": ticker,
                    "sector": stock.get("sector", ""),
                    "direction": direction,
                    "strategy": f"{direction} Credit Spread",
                    "strategy_kind": "Credit",
                    "index_fallback": False,
                    "expiry": expiry,
                    "dte": dte,
                    "stock_price_eod": close,
                    "short_strike_eod": short_strike,
                    "long_strike_eod": long_strike,
                    "preferred_width": width,
                    "estimated_eod_credit": math.nan,
                    "estimated_eod_debit": math.nan,
                    "estimated_credit_pct_width": math.nan,
                    "estimated_debit_pct_width": math.nan,
                    "construction_source": FALLBACK_SOURCE,
                    "construction_reason": (
                        "V1-style fallback income seed from top-flow universe; wider OTM/DTE window, "
                        "but Execute requires Schwab live credit at the fallback target."
                    ),
                    "anchor_strike": short_strike,
                    "target_entry": target_entry,
                    "fallback_target_credit": target_entry,
                    "expected_move_ratio": distance_pct / expected_move if math.isfinite(expected_move) and expected_move > 0 else math.nan,
                    "distance_pct": distance_pct,
                    "breakeven_distance_pct": math.nan,
                    "flow_bias": safe_float(stock.get("flow_bias"), 0.0),
                    "bot_flow_bias": math.nan,
                    "combined_flow_bias": safe_float(flow.get("rank_score"), 0.0) * 0.10 * _direction_sign(direction),
                    "flow_total_premium": safe_float(flow.get("total_premium"), safe_float(stock.get("flow_total_premium"), 0.0)),
                    "iv_rank": safe_float(stock.get("iv_rank")),
                    "iv30d": iv30d,
                    "implied_move_perc": safe_float(stock.get("implied_move_perc")),
                    "next_earnings_dt": stock.get("next_earnings_dt"),
                    "edge_type": "fallback_income_top_flow",
                    "source_contract": source.get("option_symbol", ""),
                    "source_contract_role": "short",
                    "short_leg_eod": build_occ_symbol(ticker, expiry, right, short_strike),
                    "long_leg_eod": build_occ_symbol(ticker, expiry, right, long_strike),
                    "source_contract_volume": safe_float(source.get("volume"), 0.0),
                    "source_contract_oi": safe_float(source.get("open_interest"), 0.0),
                    "source_ask_side_volume": safe_float(source.get("ask_side_volume"), 0.0),
                    "source_bid_side_volume": safe_float(source.get("bid_side_volume"), 0.0),
                    "source_mid_volume": safe_float(source.get("mid_volume"), 0.0),
                    "source_sweep_volume": safe_float(source.get("sweep_volume"), 0.0),
                    "source_cross_volume": safe_float(source.get("cross_volume"), 0.0),
                    "source_multileg_volume": safe_float(source.get("multileg_volume"), 0.0),
                    "source_stock_multileg_volume": safe_float(source.get("stock_multi_leg_volume"), 0.0),
                    "source_multileg_ratio": 0.0,
                    "source_stock_multileg_ratio": 0.0,
                    "source_side_bias": str(flow.get("flow_direction") or "unknown"),
                    "bot_bull_premium": math.nan,
                    "bot_bear_premium": math.nan,
                    "bot_total_premium": math.nan,
                    "bot_call_ask_premium": math.nan,
                    "bot_call_bid_premium": math.nan,
                    "bot_put_ask_premium": math.nan,
                    "bot_put_bid_premium": math.nan,
                    "bot_multileg_premium": math.nan,
                    "bot_multileg_ratio": math.nan,
                    "bot_volume_oi_ratio": math.nan,
                    "bot_unique_expiries": math.nan,
                    "bot_unique_strikes": math.nan,
                    "bot_trades": math.nan,
                    "flow_quality": flow_quality,
                    "flow_quality_reason": f"fallback top-flow {flow.get('flow_direction', 'unknown')}; {vwap_confirmation}",
                    "_fallback_rank": rank_score + safe_float(flow.get("volume_oi_ratio"), 0.0) * 0.10,
                }
            )
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("_fallback_rank", ascending=False).head(max_candidates)
    return df.drop(columns=["_fallback_rank"], errors="ignore")


def _tokens(value: object) -> set[str]:
    return {token.strip() for token in str(value or "").split(";") if token.strip() and token.strip().lower() not in {"nan", "none"}}


def _join_tokens(tokens: set[str]) -> str:
    return ";".join(sorted(tokens))


def _secondary_income_evidence_blocker(row: pd.Series) -> tuple[str, str] | None:
    """Return a blocker when fallback income would overwrite bad evidence.

    Fallback income is a discovery lane. It must not convert negative or weak
    replay evidence into `acceptable_secondary_income` simply because the live
    credit looks attractive.
    """
    replay = str(row.get("replay_ev_verdict") or "").strip().lower()
    edge = str(row.get("edge_verdict") or "").strip().lower()
    avg = safe_float(row.get("edge_avg_pnl"), math.nan)
    sample = safe_float(row.get("edge_sample_size"), safe_float(row.get("historical_sample_size"), math.nan))
    win = safe_float(row.get("edge_win_rate"), safe_float(row.get("historical_win_rate"), math.nan))

    if replay.startswith("negative") or edge == "negative":
        return "negative_replay_edge", "Avoid"
    if math.isfinite(avg) and avg <= 0:
        return "negative_edge_avg_pnl", "Avoid"
    if math.isfinite(sample) and sample < 7:
        return f"secondary_income_thin_sample:n={int(sample)}", "Research"
    if math.isfinite(win) and win < 0.58:
        return f"secondary_income_low_win_rate:{win:.0%}", "Research"
    return None


def apply_fallback_income_status(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "construction_source" not in scored.columns:
        return scored
    out = scored.copy()
    mask = out["construction_source"].astype(str).eq(FALLBACK_SOURCE)
    if not mask.any():
        return out
    for idx, row in out[mask].iterrows():
        target = safe_float(row.get("fallback_target_credit"), safe_float(row.get("target_entry")))
        credit = safe_float(row.get("credit"))
        mid = safe_float(row.get("mid_credit"))
        natural = safe_float(row.get("natural_credit"))
        quote_width = safe_float(row.get("quote_width_pct"))
        liq = min(
            safe_float(row.get("short_oi"), 0.0) + safe_float(row.get("short_volume"), 0.0),
            safe_float(row.get("long_oi"), 0.0) + safe_float(row.get("long_volume"), 0.0),
        )
        hard = _tokens(row.get("hard_rejects"))
        penalties = _tokens(row.get("penalties"))
        for token in ["too_close_to_expected_move", "replay_guard_bull_put_expected_move"]:
            penalties.discard(token)
        out.at[idx, "penalties"] = _join_tokens(penalties)
        out.at[idx, "required_entry"] = target
        evidence_blocker = _secondary_income_evidence_blocker(row)
        if evidence_blocker:
            blocker, disposition = evidence_blocker
            penalties.add(blocker)
            out.at[idx, "penalties"] = _join_tokens(penalties)
            out.at[idx, "decision_eligible"] = False
            out.at[idx, "decision_tier"] = ""
            out.at[idx, "decision_reason"] = blocker
            out.at[idx, "trade_status"] = disposition
            out.at[idx, "trade_tier"] = "fallback-income-weak-edge"
            out.at[idx, "trade_status_reason"] = (
                f"fallback income blocked: {blocker}; acceptable_secondary_income cannot override weak/negative evidence"
            )
            out.at[idx, "what_must_improve"] = "need positive, adequately sampled fallback-income evidence before any target ticket"
            out.at[idx, "primary_blocker"] = blocker
            if blocker in {"negative_replay_edge", "negative_edge_avg_pnl"}:
                out.at[idx, "replay_ev_verdict"] = "negative_replay_edge"
                out.at[idx, "edge_verdict"] = "negative"
            continue
        out.at[idx, "replay_ev_verdict"] = "acceptable_secondary_income"
        out.at[idx, "edge_verdict"] = "acceptable_secondary_income"
        out.at[idx, "decision_eligible"] = True
        out.at[idx, "decision_tier"] = "fallback_income"
        out.at[idx, "decision_reason"] = "decision_fallback_income_eligible"
        execute_blocking_penalties = {
            "oi_carryover_contrary",
            "news_unconfirmed",
            "news_catalyst_caution",
            "earnings_news_risk",
            "decision_news_catalyst_caution",
            "decision_final_quality_guard",
            "final_quality_guard",
        }
        if any(token.startswith("recent_loss_family:") or token.startswith("final_guard_") for token in penalties):
            execute_blocking_penalties.add("__dynamic__")
        dynamic_block = "__dynamic__" in execute_blocking_penalties
        if hard or str(row.get("live_status") or "") != "PASS":
            out.at[idx, "trade_status"] = "Avoid"
            out.at[idx, "trade_tier"] = "fallback-income-blocked"
            out.at[idx, "trade_status_reason"] = ";".join(sorted(hard)) or str(row.get("live_status") or "live validation failed")
            continue
        if not math.isfinite(target) or target <= 0:
            out.at[idx, "trade_status"] = "Research"
            out.at[idx, "trade_tier"] = "fallback-income-missing-target"
            out.at[idx, "trade_status_reason"] = "fallback income target could not be calculated"
            continue
        liquid = liq >= 500 and (not math.isfinite(quote_width) or quote_width <= 0.35)
        target_met = math.isfinite(credit) and credit >= target and math.isfinite(mid) and mid >= target
        near_target = math.isfinite(mid) and mid >= target * 0.90
        penalty_block = bool((penalties & execute_blocking_penalties) or dynamic_block)
        anchored = str(row.get("live_construction_source") or row.get("construction_source") or "") in {"flow_anchored", "fallback_income"}
        if target_met and liquid and not penalty_block and anchored:
            out.at[idx, "trade_status"] = "Execute"
            out.at[idx, "trade_tier"] = "Execute Fallback Income"
            out.at[idx, "trade_status_reason"] = (
                f"bounded fallback income: Schwab live credit ${credit:.2f} and mid ${mid:.2f} meet "
                f"${target:.2f} target; liquidity and quote width pass; manual order ticket only"
            )
            out.at[idx, "primary_blocker"] = ""
        elif (near_target or target_met) and liquid:
            out.at[idx, "trade_status"] = "Watch"
            out.at[idx, "trade_tier"] = "fallback-income-work-limit"
            blocker_text = (
                "blocking penalty remains"
                if penalty_block
                else "live alternative is not the original flow-anchored fallback"
                if not anchored
                else f"Schwab mid ${mid:.2f} is near but below ${target:.2f} target"
            )
            out.at[idx, "trade_status_reason"] = f"fallback income work-limit only: {blocker_text}; do not mark Execute or hit natural"
            out.at[idx, "what_must_improve"] = f"work limit at ${target:.2f} credit; no entry below target without fresh manual approval"
            out.at[idx, "primary_blocker"] = "fallback_execute_blocker"
        else:
            out.at[idx, "trade_status"] = "Research"
            out.at[idx, "trade_tier"] = "fallback-income-research"
            out.at[idx, "trade_status_reason"] = (
                f"fallback income rejected: credit/mid/liquidity did not support ${target:.2f} target"
            )
            out.at[idx, "primary_blocker"] = "fallback_credit_or_liquidity_failed"
    return out
