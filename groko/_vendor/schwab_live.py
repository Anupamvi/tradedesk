from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .data import safe_float
from .provenance import file_fingerprint


MIN_ACTIONABLE_CREDIT_WIDTH_RATIO = 0.16
MAX_ACTIONABLE_CREDIT_WIDTH_RATIO = 0.30
MIN_WATCH_CREDIT_WIDTH_RATIO = 0.14
MARKET_TIMEZONE = ZoneInfo("America/New_York")


def _quote_session_date(value: Any) -> dt.date | None:
    quote_millis = safe_float(value, math.nan)
    if not math.isfinite(quote_millis) or quote_millis <= 0:
        return None
    try:
        quoted_at = dt.datetime.fromtimestamp(
            quote_millis / 1000.0,
            tz=dt.timezone.utc,
        )
    except (OverflowError, OSError, ValueError):
        return None
    return quoted_at.astimezone(MARKET_TIMEZONE).date()


def _contract_quote_session_date(contract: dict[str, Any]) -> dt.date | None:
    for field in ("quoteTimeInLong", "quoteTime"):
        quote_date = _quote_session_date(contract.get(field))
        if quote_date is not None:
            return quote_date
    return None


def _iter_contracts(exp_map: dict[str, Any], right: str):
    for exp_key, strike_map in (exp_map or {}).items():
        expiry_text = str(exp_key).split(":")[0]
        try:
            expiry = dt.datetime.strptime(expiry_text[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        for strike_key, contracts in (strike_map or {}).items():
            strike = safe_float(strike_key)
            for contract in contracts or []:
                yield {
                    "expiry": expiry,
                    "right": right,
                    "strike": safe_float(contract.get("strikePrice"), strike),
                    "symbol": contract.get("symbol", ""),
                    "bid": safe_float(contract.get("bid")),
                    "ask": safe_float(contract.get("ask")),
                    "bid_size": safe_float(contract.get("bidSize")),
                    "ask_size": safe_float(contract.get("askSize")),
                    "mark": safe_float(contract.get("mark")),
                    "delta": safe_float(contract.get("delta")),
                    "theta": safe_float(contract.get("theta")),
                    "gamma": safe_float(contract.get("gamma")),
                    "vega": safe_float(contract.get("vega")),
                    "iv": safe_float(contract.get("volatility")),
                    "open_interest": safe_float(contract.get("openInterest"), 0.0),
                    "volume": safe_float(contract.get("totalVolume"), 0.0),
                    "quote_date": _contract_quote_session_date(contract),
                }


def chain_to_contracts(chain: dict[str, Any]) -> pd.DataFrame:
    rows = list(_iter_contracts(chain.get("callExpDateMap", {}), "C"))
    rows.extend(_iter_contracts(chain.get("putExpDateMap", {}), "P"))
    return pd.DataFrame(rows)


def chain_quote_dates(chain: dict[str, Any]) -> set[dt.date]:
    """Return the New York session dates represented by option quote timestamps."""

    dates: set[dt.date] = set()
    for exp_map in (chain.get("callExpDateMap", {}), chain.get("putExpDateMap", {})):
        for strike_map in (exp_map or {}).values():
            for contracts in (strike_map or {}).values():
                for contract in contracts or []:
                    quote_date = _contract_quote_session_date(contract)
                    if quote_date is not None:
                        dates.add(quote_date)
    return dates


def chain_spot(chain: dict[str, Any]) -> float:
    underlying = chain.get("underlying", {}) if isinstance(chain, dict) else {}
    for value in [
        chain.get("underlyingPrice"),
        underlying.get("mark"),
        underlying.get("last"),
        underlying.get("lastPrice"),
    ]:
        number = safe_float(value)
        if math.isfinite(number) and number > 0:
            return number
    return math.nan


def _chain_error_is_timeout(error: Any) -> bool:
    text = str(error or "").lower()
    return "timeout" in text or "timed out" in text


def option_mid(row: pd.Series) -> float:
    bid = safe_float(row.get("bid"))
    ask = safe_float(row.get("ask"))
    mark = safe_float(row.get("mark"))
    if math.isfinite(bid) and math.isfinite(ask) and bid >= 0 and ask > 0:
        return (bid + ask) / 2.0
    return mark


def price_width_bucket(spot: float) -> float:
    if spot < 60:
        return 2.5
    if spot < 180:
        return 5.0
    if spot < 500:
        return 5.0
    return 10.0


def _credit_width_candidates(spot: float, preferred_width: float | None) -> tuple[float, ...]:
    """Return anchor and risk-sized widths for live credit-spread construction."""

    preferred = safe_float(preferred_width)
    standard = float(price_width_bucket(spot))
    widths = [preferred, standard, min(standard, 5.0)]
    if standard >= 5.0:
        widths.append(2.5)
    return tuple(
        dict.fromkeys(
            round(float(width), 4)
            for width in widths
            if math.isfinite(width) and width > 0
        )
    )


def _same_expiry_contracts(contracts: pd.DataFrame, expiry: dt.date, right: str) -> pd.DataFrame:
    if contracts.empty:
        return contracts
    out = contracts[(contracts["expiry"] == expiry) & (contracts["right"] == right)].copy()
    for col in ["strike", "bid", "ask", "mark", "delta", "theta", "gamma", "vega", "open_interest", "volume"]:
        if col not in out.columns:
            out[col] = math.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("bid_size", "ask_size"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["mid"] = out.apply(option_mid, axis=1)
    out["quote_width"] = out["ask"] - out["bid"]
    out["quote_width_pct"] = out["quote_width"] / out["mid"].where(out["mid"].abs() > 0)
    return out.sort_values("strike")


def _credit_spread_candidates(
    contracts: pd.DataFrame,
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    preferred_width: float | None = None,
) -> pd.DataFrame:
    right = "P" if direction == "Bull Put" else "C"
    chain = _same_expiry_contracts(contracts, expiry, right)
    if chain.empty:
        return pd.DataFrame()

    widths = _credit_width_candidates(spot, preferred_width)
    rows: list[dict[str, Any]] = []
    strikes = sorted(float(x) for x in chain["strike"].dropna().unique())
    by_strike = {float(r["strike"]): r for _, r in chain.iterrows()}
    displayed_sizes_available = {"bid_size", "ask_size"}.issubset(chain.columns)

    for _, short in chain.iterrows():
        short_strike = safe_float(short.get("strike"))
        if not math.isfinite(short_strike):
            continue
        if direction == "Bull Put":
            if short_strike >= spot:
                continue
            distance_pct = (spot - short_strike) / spot
            delta_abs = abs(safe_float(short.get("delta")))
        else:
            if short_strike <= spot:
                continue
            distance_pct = (short_strike - spot) / spot
            delta_abs = abs(safe_float(short.get("delta")))
        if distance_pct < 0.015:
            continue
        if math.isfinite(delta_abs) and not (0.08 <= delta_abs <= 0.35):
            continue
        for width in widths:
            if direction == "Bull Put":
                long_candidates = [
                    strike
                    for strike in strikes
                    if strike < short_strike
                    and abs((short_strike - strike) - width) <= max(0.51, width * 0.35)
                ]
            else:
                long_candidates = [
                    strike
                    for strike in strikes
                    if strike > short_strike
                    and abs((strike - short_strike) - width) <= max(0.51, width * 0.35)
                ]
            if not long_candidates:
                continue
            long_strike = min(long_candidates, key=lambda strike: abs(abs(strike - short_strike) - width))
            long = by_strike[long_strike]
            actual_width = abs(long_strike - short_strike)
            short_bid = safe_float(short.get("bid"))
            short_ask = safe_float(short.get("ask"))
            long_bid = safe_float(long.get("bid"))
            long_ask = safe_float(long.get("ask"))
            displayed_entry_size = (
                min(
                    safe_float(short.get("bid_size"), 0.0),
                    safe_float(long.get("ask_size"), 0.0),
                )
                if displayed_sizes_available
                else math.nan
            )
            if displayed_sizes_available and (
                not math.isfinite(displayed_entry_size) or displayed_entry_size < 1
            ):
                continue
            short_mid = option_mid(short)
            long_mid = option_mid(long)
            natural_credit = short_bid - long_ask
            mid_credit = short_mid - long_mid
            realistic_credit = max(natural_credit, mid_credit * 0.90)
            if not math.isfinite(realistic_credit) or realistic_credit <= 0:
                continue
            credit_pct = realistic_credit / actual_width if actual_width > 0 else math.nan
            pop = 1.0 - delta_abs if math.isfinite(delta_abs) and delta_abs > 0 else math.nan
            breakeven = (
                short_strike - realistic_credit
                if direction == "Bull Put"
                else short_strike + realistic_credit
            )
            breakeven_distance_pct = abs(breakeven - spot) / spot if spot else math.nan
            short_liq = safe_float(short.get("open_interest"), 0.0) + safe_float(short.get("volume"), 0.0)
            long_liq = safe_float(long.get("open_interest"), 0.0) + safe_float(long.get("volume"), 0.0)
            short_qwp = safe_float(short.get("quote_width_pct"))
            long_qwp = safe_float(long.get("quote_width_pct"))
            quote_penalty = max(
                short_qwp if math.isfinite(short_qwp) else 0.0,
                long_qwp if math.isfinite(long_qwp) else 0.0,
            )
            short_iv = safe_float(short.get("iv"))
            long_iv = safe_float(long.get("iv"))
            finite_ivs = [value for value in (short_iv, long_iv) if math.isfinite(value) and value > 0]
            spread_iv = sum(finite_ivs) / len(finite_ivs) if finite_ivs else math.nan
            rows.append({
                "live_status": "PASS",
                "short_leg": short.get("symbol", ""),
                "long_leg": long.get("symbol", ""),
                "short_strike": short_strike,
                "long_strike": long_strike,
                "spread_width": actual_width,
                "credit": round(realistic_credit, 2),
                "mid_credit": round(mid_credit, 2),
                "natural_credit": round(natural_credit, 2),
                "credit_pct_width": credit_pct,
                "sell_leg_bid": short_bid,
                "sell_leg_ask": short_ask,
                "sell_leg_mid": short_mid,
                "buy_leg_bid": long_bid,
                "buy_leg_ask": long_ask,
                "buy_leg_mid": long_mid,
                "pop_delta_proxy": pop,
                "short_delta": safe_float(short.get("delta")),
                "long_delta": safe_float(long.get("delta")),
                "short_theta": safe_float(short.get("theta")),
                "long_theta": safe_float(long.get("theta")),
                "net_theta": safe_float(long.get("theta")) - safe_float(short.get("theta")),
                "short_iv": short_iv,
                "long_iv": long_iv,
                "spread_iv": spread_iv,
                "distance_pct": distance_pct,
                "breakeven": breakeven,
                "breakeven_distance_pct": breakeven_distance_pct,
                "short_oi": safe_float(short.get("open_interest"), 0.0),
                "short_volume": safe_float(short.get("volume"), 0.0),
                "long_oi": safe_float(long.get("open_interest"), 0.0),
                "long_volume": safe_float(long.get("volume"), 0.0),
                "short_bid_size": safe_float(short.get("bid_size")),
                "long_ask_size": safe_float(long.get("ask_size")),
                "displayed_entry_size": displayed_entry_size,
                "quote_width_pct": quote_penalty,
                "liq_score": min(short_liq, long_liq),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.drop_duplicates(["short_strike", "long_strike"], keep="first")
    delta_target_penalty = (df["short_delta"].abs().fillna(0.22) - 0.22).abs().clip(upper=0.20)
    df["_rank"] = (
        df["credit_pct_width"].clip(upper=0.45) * 4.0
        + df["distance_pct"].clip(upper=0.18) * 24.0
        + df["pop_delta_proxy"].fillna(0.55)
        + (df["liq_score"].clip(upper=5000) / 5000.0)
        - df["quote_width_pct"].fillna(0.0).clip(upper=2.0)
        - delta_target_penalty * 2.0
    )
    return df


def _expected_move_ratio(distance: float, expected_move_pct: float | None) -> float:
    expected = safe_float(expected_move_pct)
    if not math.isfinite(expected) or expected <= 0:
        return math.nan
    if not math.isfinite(distance):
        return math.nan
    return distance / expected


def _liq_summary(row: pd.Series) -> str:
    return (
        f"short oi+vol {safe_float(row.get('short_oi'), 0.0) + safe_float(row.get('short_volume'), 0.0):.0f}; "
        f"long oi+vol {safe_float(row.get('long_oi'), 0.0) + safe_float(row.get('long_volume'), 0.0):.0f}; "
        f"quote width {safe_float(row.get('quote_width_pct')):.1%}"
    )


def find_credit_spread_alternatives(
    contracts: pd.DataFrame,
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    preferred_width: float | None = None,
    anchor_strike: float | None = None,
    expected_move_pct: float | None = None,
    as_of_date: dt.date | None = None,
    max_alternatives: int = 5,
    min_actionable_credit_width_ratio: float = MIN_ACTIONABLE_CREDIT_WIDTH_RATIO,
    min_watch_credit_width_ratio: float = MIN_WATCH_CREDIT_WIDTH_RATIO,
) -> list[dict[str, Any]]:
    right = "P" if direction == "Bull Put" else "C"
    df = _credit_spread_candidates(
        contracts,
        direction=direction,
        expiry=expiry,
        spot=spot,
        preferred_width=preferred_width,
    )
    if df.empty:
        chain = _same_expiry_contracts(contracts, expiry, right)
        if chain.empty:
            return [{"live_status": "missing_expiry_or_right", "live_blocker": f"no {right} contracts for {expiry}"}]
        return [{"live_status": "no_realistic_spread", "live_blocker": "no OTM spread with positive realistic credit/delta/liquidity"}]

    selected: list[tuple[str, str, pd.Series]] = []

    def add(label: str, reason: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if frame.empty:
            return
        row = frame.sort_values(sort_cols, ascending=ascending).iloc[0]
        key = (safe_float(row.get("short_strike")), safe_float(row.get("long_strike")))
        if any((safe_float(item.get("short_strike")), safe_float(item.get("long_strike"))) == key for _, _, item in selected):
            return
        selected.append((label, reason, row))

    anchor = safe_float(anchor_strike)
    provided_expected = safe_float(expected_move_pct)
    reference_day = as_of_date
    dte = max((expiry - reference_day).days, 0) if reference_day is not None else 0

    def row_expected_move(spread_iv: Any) -> float:
        if math.isfinite(provided_expected) and provided_expected > 0:
            return provided_expected
        annualized_iv = safe_float(spread_iv)
        if annualized_iv > 3.0:
            annualized_iv /= 100.0
        if not math.isfinite(annualized_iv) or annualized_iv <= 0 or dte <= 0:
            return math.nan
        return annualized_iv * math.sqrt(dte / 365.0)

    df = df.copy()
    df["_anchor_distance"] = (df["short_strike"] - anchor).abs() if math.isfinite(anchor) else math.inf
    df["_expected_move_pct"] = df["spread_iv"].map(row_expected_move)
    df["_expected_move_ratio"] = df.apply(
        lambda row: _expected_move_ratio(
            safe_float(row.get("breakeven_distance_pct")),
            safe_float(row.get("_expected_move_pct")),
        ),
        axis=1,
    )
    preferred = safe_float(preferred_width)
    df["_width_pref_distance"] = (df["spread_width"] - preferred).abs() if math.isfinite(preferred) else 0.0

    # Expected-move 0.75 was a count constraint, not a risk gate: it preferred
    # far OTM low-credit structures and hid send-now 25-30% credits. One-lot
    # $750 is a size input, not a reason to drop a defined-risk name from the
    # alternative set; prefer it, then rank remaining send-now economics.
    df["_one_lot_risk"] = (df["spread_width"] - df["credit"]) * 100.0
    df["_risk_sized"] = df["_one_lot_risk"].le(750.0).fillna(False)
    actionable = df[
        df["credit_pct_width"].between(
            min_actionable_credit_width_ratio,
            MAX_ACTIONABLE_CREDIT_WIDTH_RATIO,
            inclusive="both",
        )
        & (df["quote_width_pct"] <= 0.25)
        & (df["liq_score"] >= 100)
    ]
    actionable_ranked = actionable.sort_values(
        ["_risk_sized", "credit_pct_width", "liq_score", "quote_width_pct", "_rank"],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    )
    # Keep more than one qualifying structure. A single high-distance, low-credit
    # representative can otherwise hide a better send-now contract from the global
    # expiry scorer.
    for _, actionable_row in actionable_ranked.head(max_alternatives).iterrows():
        key = (
            safe_float(actionable_row.get("short_strike")),
            safe_float(actionable_row.get("long_strike")),
        )
        if any(
            (safe_float(item.get("short_strike")), safe_float(item.get("long_strike"))) == key
            for _, _, item in selected
        ):
            continue
        selected.append(
            (
                "actionable_quality",
                "risk-sized spread with validated credit band, liquidity, and quote quality",
                actionable_row,
            )
        )

    add(
        "flow_anchored",
        "closest realistic live spread to the UW flow strike",
        df,
        ["_anchor_distance", "_rank"],
        [True, False],
    )
    add(
        "expected_move_safe",
        "breakeven has the best expected-move buffer among realistic spreads",
        df[df["_expected_move_ratio"].fillna(0.0) >= 0.65]
        if df["_expected_move_pct"].notna().any()
        else df,
        ["_expected_move_ratio", "_rank"],
        [False, False],
    )
    add(
        "better_credit",
        "highest live credit/width candidate with defined risk",
        df,
        ["credit_pct_width", "_rank"],
        [False, False],
    )
    add(
        "tighter_width_conservative",
        "narrower-width alternative that keeps liquidity and distance reasonable",
        df[df["spread_width"] <= preferred] if math.isfinite(preferred) else df,
        ["spread_width", "pop_delta_proxy", "_rank"],
        [True, False, False],
    )
    add(
        "near_trigger_watch",
        "sub-trigger credit candidate to keep as a work-limit Watch order",
        df[
            (df["credit_pct_width"] >= min_watch_credit_width_ratio)
            & (df["credit_pct_width"] < min_actionable_credit_width_ratio)
        ],
        ["credit_pct_width", "_rank"],
        [False, False],
    )

    if not selected:
        selected = [("best_ranked", "best ranked realistic defined-risk credit spread", df.sort_values("_rank", ascending=False).iloc[0])]

    rows: list[dict[str, Any]] = []
    for label, reason, row in selected[:max_alternatives]:
        out = row.drop(labels=[c for c in row.index if str(c).startswith("_")], errors="ignore").to_dict()
        width = safe_float(out.get("spread_width"))
        target_entry = (
            round(width * min_actionable_credit_width_ratio, 2)
            if math.isfinite(width) and width > 0
            else math.nan
        )
        expected = safe_float(row.get("_expected_move_pct"))
        ratio = _expected_move_ratio(safe_float(out.get("breakeven_distance_pct")), expected)
        out.update(
            {
                "construction_source": label,
                "construction_reason": reason,
                "anchor_strike": anchor if math.isfinite(anchor) else math.nan,
                "target_entry": target_entry,
                "expected_move_pct": expected if math.isfinite(expected) else math.nan,
                "expected_move_ratio": ratio,
                "breakeven_expected_move_ratio": ratio,
                "liquidity_summary": _liq_summary(pd.Series(out)),
            }
        )
        rows.append(out)
    return rows


def find_best_credit_spread(
    contracts: pd.DataFrame,
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    preferred_width: float | None = None,
) -> dict[str, Any]:
    alternatives = find_credit_spread_alternatives(
        contracts,
        direction=direction,
        expiry=expiry,
        spot=spot,
        preferred_width=preferred_width,
        max_alternatives=1,
    )
    return alternatives[0] if alternatives else {"live_status": "no_realistic_spread", "live_blocker": "no realistic credit spread"}


def _debit_spread_candidates(
    contracts: pd.DataFrame,
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    preferred_width: float | None = None,
) -> pd.DataFrame:
    if direction not in {"Bull Call", "Bear Put"}:
        return pd.DataFrame()
    right = "C" if direction == "Bull Call" else "P"
    chain = _same_expiry_contracts(contracts, expiry, right)
    if chain.empty:
        return pd.DataFrame()

    # Preserve wider intentional structures, but widen undersized seed spreads
    # to the normal live-chain width for the underlying price bucket.
    preferred = safe_float(preferred_width)
    width = max(
        float(price_width_bucket(spot)),
        float(preferred) if math.isfinite(preferred) and preferred > 0 else 0.0,
    )
    rows: list[dict[str, Any]] = []
    strikes = sorted(float(x) for x in chain["strike"].dropna().unique())
    by_strike = {float(r["strike"]): r for _, r in chain.iterrows()}
    displayed_sizes_available = {"bid_size", "ask_size"}.issubset(chain.columns)

    for _, long in chain.iterrows():
        long_strike = safe_float(long.get("strike"))
        if not math.isfinite(long_strike):
            continue
        if direction == "Bull Call":
            if long_strike < spot * 0.96 or long_strike > spot * 1.08:
                continue
            short_candidates = [s for s in strikes if s > long_strike and abs((s - long_strike) - width) <= max(0.51, width * 0.35)]
            delta_abs = abs(safe_float(long.get("delta")))
        else:
            if long_strike > spot * 1.04 or long_strike < spot * 0.90:
                continue
            short_candidates = [s for s in strikes if s < long_strike and abs((long_strike - s) - width) <= max(0.51, width * 0.35)]
            delta_abs = abs(safe_float(long.get("delta")))
        if not short_candidates:
            continue
        short_strike = min(short_candidates, key=lambda s: abs(abs(s - long_strike) - width))
        short = by_strike[short_strike]
        actual_width = abs(short_strike - long_strike)
        long_bid = safe_float(long.get("bid"))
        long_ask = safe_float(long.get("ask"))
        short_bid = safe_float(short.get("bid"))
        short_ask = safe_float(short.get("ask"))
        displayed_entry_size = (
            min(
                safe_float(long.get("ask_size"), 0.0),
                safe_float(short.get("bid_size"), 0.0),
            )
            if displayed_sizes_available
            else math.nan
        )
        if displayed_sizes_available and (
            not math.isfinite(displayed_entry_size) or displayed_entry_size < 1
        ):
            continue
        long_mid = option_mid(long)
        short_mid = option_mid(short)
        natural_debit = long_ask - short_bid
        mid_debit = long_mid - short_mid
        realistic_debit = min(natural_debit, mid_debit * 1.10)
        if not math.isfinite(realistic_debit) or realistic_debit <= 0 or actual_width <= 0:
            continue
        if realistic_debit >= actual_width:
            continue
        debit_pct = realistic_debit / actual_width
        max_profit = actual_width - realistic_debit
        reward_risk = max_profit / realistic_debit if realistic_debit > 0 else math.nan
        if direction == "Bull Call":
            breakeven = long_strike + realistic_debit
            breakeven_distance_pct = (breakeven - spot) / spot
        else:
            breakeven = long_strike - realistic_debit
            breakeven_distance_pct = (spot - breakeven) / spot
        long_liq = safe_float(long.get("open_interest"), 0.0) + safe_float(long.get("volume"), 0.0)
        short_liq = safe_float(short.get("open_interest"), 0.0) + safe_float(short.get("volume"), 0.0)
        long_qwp = safe_float(long.get("quote_width_pct"))
        short_qwp = safe_float(short.get("quote_width_pct"))
        quote_penalty = max(long_qwp if math.isfinite(long_qwp) else 0.0, short_qwp if math.isfinite(short_qwp) else 0.0)
        short_iv = safe_float(short.get("iv"))
        long_iv = safe_float(long.get("iv"))
        finite_ivs = [value for value in (short_iv, long_iv) if math.isfinite(value) and value > 0]
        spread_iv = sum(finite_ivs) / len(finite_ivs) if finite_ivs else math.nan
        if math.isfinite(delta_abs) and not (0.25 <= delta_abs <= 0.75):
            continue
        if math.isfinite(breakeven_distance_pct) and breakeven_distance_pct < -0.01:
            continue
        rows.append(
            {
                "live_status": "PASS",
                "short_leg": short.get("symbol", ""),
                "long_leg": long.get("symbol", ""),
                "short_strike": short_strike,
                "long_strike": long_strike,
                "spread_width": actual_width,
                "debit": round(realistic_debit, 2),
                "mid_debit": round(mid_debit, 2),
                "natural_debit": round(natural_debit, 2),
                "debit_pct_width": debit_pct,
                "credit_pct_width": math.nan,
                "sell_leg_bid": short_bid,
                "sell_leg_ask": short_ask,
                "sell_leg_mid": short_mid,
                "buy_leg_bid": long_bid,
                "buy_leg_ask": long_ask,
                "buy_leg_mid": long_mid,
                "pop_delta_proxy": math.nan,
                "short_delta": safe_float(short.get("delta")),
                "long_delta": safe_float(long.get("delta")),
                "short_theta": safe_float(short.get("theta")),
                "long_theta": safe_float(long.get("theta")),
                "net_theta": safe_float(long.get("theta")) - safe_float(short.get("theta")),
                "short_iv": short_iv,
                "long_iv": long_iv,
                "spread_iv": spread_iv,
                "theta_burn_pct": (
                    abs(safe_float(long.get("theta")) - safe_float(short.get("theta"))) / realistic_debit
                    if realistic_debit > 0
                    and math.isfinite(safe_float(long.get("theta")))
                    and math.isfinite(safe_float(short.get("theta")))
                    else math.nan
                ),
                "distance_pct": abs((long_strike - spot) / spot) if spot else math.nan,
                "breakeven": breakeven,
                "breakeven_distance_pct": breakeven_distance_pct,
                "reward_risk": reward_risk,
                "short_oi": safe_float(short.get("open_interest"), 0.0),
                "short_volume": safe_float(short.get("volume"), 0.0),
                "long_oi": safe_float(long.get("open_interest"), 0.0),
                "long_volume": safe_float(long.get("volume"), 0.0),
                "long_ask_size": safe_float(long.get("ask_size")),
                "short_bid_size": safe_float(short.get("bid_size")),
                "displayed_entry_size": displayed_entry_size,
                "quote_width_pct": quote_penalty,
                "liq_score": min(short_liq, long_liq),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    rr = pd.to_numeric(df["reward_risk"], errors="coerce").fillna(0.0).clip(upper=3.0)
    debit_pct = pd.to_numeric(df["debit_pct_width"], errors="coerce").fillna(1.0)
    be_dist = pd.to_numeric(df["breakeven_distance_pct"], errors="coerce").fillna(0.20).clip(lower=0.0, upper=0.20)
    df["_rank"] = (
        rr
        + (1.0 - debit_pct).clip(lower=0.0) * 1.5
        + (df["liq_score"].clip(upper=5000) / 5000.0)
        - be_dist * 8.0
        - df["quote_width_pct"].fillna(0.0).clip(upper=2.0)
    )
    return df


def _debit_expected_move_ratio(breakeven_distance_pct: float, expected_move_pct: float | None) -> float:
    expected = safe_float(expected_move_pct)
    distance = safe_float(breakeven_distance_pct)
    if not math.isfinite(expected) or expected <= 0 or not math.isfinite(distance):
        return math.nan
    return expected / max(distance, 0.001)


def _debit_breakeven_expected_move_ratio(
    breakeven_distance_pct: float,
    expected_move_pct: float | None,
) -> float:
    expected = safe_float(expected_move_pct)
    distance = safe_float(breakeven_distance_pct)
    if not math.isfinite(expected) or expected <= 0 or not math.isfinite(distance):
        return math.nan
    return max(distance, 0.0) / expected


def find_debit_spread_alternatives(
    contracts: pd.DataFrame,
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    preferred_width: float | None = None,
    anchor_strike: float | None = None,
    expected_move_pct: float | None = None,
    as_of_date: dt.date | None = None,
    max_alternatives: int = 4,
) -> list[dict[str, Any]]:
    if direction not in {"Bull Call", "Bear Put"}:
        return [{"live_status": "unsupported_debit_direction", "live_blocker": f"unsupported debit direction {direction}"}]
    right = "C" if direction == "Bull Call" else "P"
    df = _debit_spread_candidates(
        contracts,
        direction=direction,
        expiry=expiry,
        spot=spot,
        preferred_width=preferred_width,
    )
    if df.empty:
        chain = _same_expiry_contracts(contracts, expiry, right)
        if chain.empty:
            return [{"live_status": "missing_expiry_or_right", "live_blocker": f"no {right} contracts for {expiry}"}]
        return [{"live_status": "no_realistic_spread", "live_blocker": "no realistic debit spread with acceptable debit/delta/liquidity"}]

    selected: list[tuple[str, str, pd.Series]] = []

    def add(label: str, reason: str, frame: pd.DataFrame, sort_cols: list[str], ascending: list[bool]) -> None:
        if frame.empty:
            return
        row = frame.sort_values(sort_cols, ascending=ascending).iloc[0]
        key = (safe_float(row.get("short_strike")), safe_float(row.get("long_strike")))
        if any((safe_float(item.get("short_strike")), safe_float(item.get("long_strike"))) == key for _, _, item in selected):
            return
        selected.append((label, reason, row))

    anchor = safe_float(anchor_strike)
    provided_expected = safe_float(expected_move_pct)
    reference_day = as_of_date or dt.date.today()
    dte = max((expiry - reference_day).days, 0)

    def row_expected_move(spread_iv: Any) -> float:
        if math.isfinite(provided_expected) and provided_expected > 0:
            return provided_expected
        annualized_iv = safe_float(spread_iv)
        if annualized_iv > 3.0:
            annualized_iv /= 100.0
        if not math.isfinite(annualized_iv) or annualized_iv <= 0 or dte <= 0:
            return math.nan
        return annualized_iv * math.sqrt(dte / 365.0)

    df = df.copy()
    df["_anchor_distance"] = (df["long_strike"] - anchor).abs() if math.isfinite(anchor) else math.inf
    df["_expected_move_pct"] = df["spread_iv"].map(row_expected_move)
    df["_expected_move_ratio"] = df.apply(
        lambda row: _debit_expected_move_ratio(
            safe_float(row.get("breakeven_distance_pct")),
            safe_float(row.get("_expected_move_pct")),
        ),
        axis=1,
    )
    df["_breakeven_expected_move_ratio"] = df.apply(
        lambda row: _debit_breakeven_expected_move_ratio(
            safe_float(row.get("breakeven_distance_pct")),
            safe_float(row.get("_expected_move_pct")),
        ),
        axis=1,
    )

    actionable = df[
        (df["reward_risk"] >= 1.50)
        & (df["long_delta"].abs() >= 0.40)
        & (df["quote_width_pct"] <= 0.25)
        & (df["liq_score"] >= 100)
        & (df["_breakeven_expected_move_ratio"] <= 0.75)
    ]
    add(
        "actionable_quality",
        "material-width spread with send-now probability, reward/risk, liquidity, and quote quality",
        actionable,
        ["_rank", "reward_risk", "liq_score", "quote_width_pct"],
        [False, False, False, True],
    )

    add(
        "flow_anchored",
        "closest realistic live debit spread to the UW flow strike",
        df,
        ["_anchor_distance", "_rank"],
        [True, False],
    )
    add(
        "breakout",
        "higher-confirmation breakout structure with reachable breakeven",
        df[df["_expected_move_ratio"].fillna(0.0) >= 1.0]
        if df["_expected_move_pct"].notna().any()
        else df,
        ["_expected_move_ratio", "_rank"],
        [False, False],
    )
    add(
        "lower_debit_better_reward_risk",
        "lower debit and better reward/risk alternate",
        df,
        ["debit_pct_width", "reward_risk", "_rank"],
        [True, False, False],
    )
    add(
        "expected_move_reachable_breakeven",
        "breakeven is most reachable relative to expected move",
        df,
        ["breakeven_distance_pct", "_rank"],
        [True, False],
    )

    if not selected:
        selected = [("best_ranked", "best ranked realistic defined-risk debit spread", df.sort_values("_rank", ascending=False).iloc[0])]

    rows: list[dict[str, Any]] = []
    for label, reason, row in selected[:max_alternatives]:
        out = row.drop(labels=[c for c in row.index if str(c).startswith("_")], errors="ignore").to_dict()
        width = safe_float(out.get("spread_width"))
        target_entry = round(width * 0.45, 2) if math.isfinite(width) and width > 0 else math.nan
        expected = safe_float(row.get("_expected_move_pct"))
        ratio = safe_float(row.get("_expected_move_ratio"))
        breakeven_expected_ratio = _debit_breakeven_expected_move_ratio(
            safe_float(out.get("breakeven_distance_pct")),
            expected,
        )
        out.update(
            {
                "construction_source": label,
                "construction_reason": reason,
                "anchor_strike": anchor if math.isfinite(anchor) else math.nan,
                "target_entry": target_entry,
                "expected_move_pct": expected,
                "expected_move_ratio": ratio,
                "breakeven_expected_move_ratio": breakeven_expected_ratio,
                "liquidity_summary": _liq_summary(pd.Series(out)),
            }
        )
        rows.append(out)
    return rows


def find_best_debit_spread(
    contracts: pd.DataFrame,
    *,
    direction: str,
    expiry: dt.date,
    spot: float,
    preferred_width: float | None = None,
) -> dict[str, Any]:
    alternatives = find_debit_spread_alternatives(
        contracts,
        direction=direction,
        expiry=expiry,
        spot=spot,
        preferred_width=preferred_width,
        max_alternatives=1,
    )
    return alternatives[0] if alternatives else {"live_status": "no_realistic_spread", "live_blocker": "no realistic debit spread"}


class SchwabChainValidator:
    def __init__(
        self,
        out_dir: Path,
        *,
        strike_count: int = 80,
        snapshot_dir: Path | None = None,
        allow_live_fallback: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.strike_count = strike_count
        self.snapshot_dir = Path(snapshot_dir).expanduser().resolve() if snapshot_dir else None
        self.allow_live_fallback = allow_live_fallback
        self.service = None
        self.chains: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, str] = {}
        self.sources: dict[str, str] = {}

    def _service(self):
        if self.service is None:
            from .schwab_auth import SchwabAuthConfig, SchwabLiveDataService

            self.service = SchwabLiveDataService(SchwabAuthConfig.from_env(load_dotenv_file=True), interactive_login=False)
        return self.service

    def _snapshot_path(self, symbol: str) -> Path | None:
        if not self.snapshot_dir:
            return None
        candidates = [
            self.snapshot_dir / f"{symbol}.json",
            self.snapshot_dir / f"chain_{symbol}.json",
            self.snapshot_dir / "schwab_chains" / f"{symbol}.json",
            self.snapshot_dir / "schwab_chains" / f"chain_{symbol}.json",
            self.snapshot_dir / "chains" / f"{symbol}.json",
            self.snapshot_dir / "chains" / f"chain_{symbol}.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def get_chain(self, ticker: str, from_date: dt.date, to_date: dt.date) -> dict[str, Any] | None:
        symbol = str(ticker).upper().strip()
        api_symbol = {
            "BFB": "BF/B",
            "BF.B": "BF/B",
            "BF-B": "BF/B",
            "BRKB": "BRK/B",
            "BRK.B": "BRK/B",
            "BRK-B": "BRK/B",
        }.get(symbol, symbol)
        if symbol in self.chains:
            return self.chains[symbol]
        snapshot_path = self._snapshot_path(symbol)
        if snapshot_path is not None:
            try:
                chain = json.loads(snapshot_path.read_text(encoding="utf-8"))
                self.chains[symbol] = chain
                self.sources[symbol] = f"snapshot:{snapshot_path}"
                return chain
            except Exception as exc:
                self.errors[symbol] = f"snapshot load failed: {exc}"
                return None
        if self.snapshot_dir is not None and not self.allow_live_fallback:
            self.errors[symbol] = f"snapshot missing for {symbol} in {self.snapshot_dir}"
            return None
        try:
            chain = self._service().get_option_chain(
                api_symbol,
                strike_count=self.strike_count,
                include_underlying_quote=True,
                from_date=from_date,
                to_date=to_date,
            )
            self.chains[symbol] = chain
            self.sources[symbol] = "schwab_live_dated"
            return chain
        except Exception as exc:
            dated_error = str(exc)
        if _chain_error_is_timeout(dated_error):
            self.errors[symbol] = f"dated chain failed: {dated_error}; undated fallback skipped after timeout"
            return None
        try:
            chain = self._service().get_option_chain(
                api_symbol,
                strike_count=self.strike_count,
                include_underlying_quote=True,
            )
            self.chains[symbol] = chain
            self.sources[symbol] = "schwab_live_undated_fallback"
            return chain
        except Exception as exc:
            self.errors[symbol] = f"dated chain failed: {dated_error}; undated fallback failed: {exc}"
            return None

    def save(self) -> None:
        chain_dir = self.out_dir / "schwab_chains"
        chain_dir.mkdir(parents=True, exist_ok=True)
        chain_files: dict[str, Any] = {}
        for ticker, chain in self.chains.items():
            path = chain_dir / f"{ticker}.json"
            path.write_text(json.dumps(chain, indent=2, sort_keys=True), encoding="utf-8")
            chain_files[ticker] = file_fingerprint(path)
        errors_path = chain_dir / "errors.json"
        errors_path.write_text(json.dumps(self.errors, indent=2, sort_keys=True), encoding="utf-8")
        manifest = {
            "snapshot_dir": str(chain_dir),
            "input_snapshot_dir": str(self.snapshot_dir) if self.snapshot_dir else "",
            "strike_count": self.strike_count,
            "chain_count": len(self.chains),
            "sources": self.sources,
            "chains": chain_files,
            "errors": self.errors,
        }
        (chain_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
