from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .data import aggregate_bot_flow, infer_asof_date, load_hot_chains, load_stock_screener, safe_float
from .engine import detect_regime, generate_candidates, replay_quality_pattern, select_ticker_pool, validated_addon_income_lane
from .occ import build_occ_symbol, parse_occ_symbol


def dated_folders(root: Path, start: dt.date | None, end: dt.date | None) -> list[Path]:
    folders: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            day = infer_asof_date(path)
        except ValueError:
            continue
        if start and day < start:
            continue
        if end and day > end:
            continue
        folders.append(path)
    return sorted(folders, key=infer_asof_date)


def parse_date(value: str) -> dt.date | None:
    if not value:
        return None
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def load_close_history(folders: list[Path]) -> dict[dt.date, pd.DataFrame]:
    out: dict[dt.date, pd.DataFrame] = {}
    for folder in folders:
        day = infer_asof_date(folder)
        try:
            sc = load_stock_screener(folder)
        except Exception:
            continue
        out[day] = sc[["ticker", "close", "sector"]].copy()
    return out


def load_hot_history(folders: list[Path]) -> dict[dt.date, pd.DataFrame]:
    out: dict[dt.date, pd.DataFrame] = {}
    for folder in folders:
        day = infer_asof_date(folder)
        try:
            out[day] = load_hot_chains(folder, day)
        except Exception:
            continue
    return out


def future_close(history: dict[dt.date, pd.DataFrame], ticker: str, target: dt.date) -> tuple[dt.date | None, float]:
    for day in sorted(history):
        if day < target:
            continue
        rows = history[day][history[day]["ticker"].eq(ticker)]
        if not rows.empty:
            return day, safe_float(rows.iloc[0].get("close"))
    return None, math.nan


def evaluate_candidate(row: pd.Series, history: dict[dt.date, pd.DataFrame]) -> dict[str, Any]:
    ticker = str(row.get("ticker", "")).upper()
    expiry = row.get("expiry")
    if not isinstance(expiry, dt.date):
        return {"evaluated": False, "reason": "missing_expiry"}
    eval_day, close = future_close(history, ticker, expiry)
    if not eval_day or not math.isfinite(close):
        return {"evaluated": False, "reason": "missing_future_close"}
    short = safe_float(row.get("short_strike_eod"))
    direction = str(row.get("direction", ""))
    if direction == "Bull Put":
        win = close > short
        breach_pct = (short - close) / close if close else math.nan
    elif direction == "Bear Call":
        win = close < short
        breach_pct = (close - short) / close if close else math.nan
    else:
        return {"evaluated": False, "reason": "unknown_direction"}
    return {
        "evaluated": True,
        "eval_day": eval_day,
        "future_close": close,
        "win": bool(win),
        "breach_pct": breach_pct,
    }


def _quote_lookup(hot: pd.DataFrame) -> dict[str, dict[str, float | str]]:
    if hot.empty or "option_symbol" not in hot.columns:
        return {}
    df = hot.copy()
    df["option_symbol"] = df["option_symbol"].astype(str).str.upper().str.strip()
    bid = pd.to_numeric(df["bid"], errors="coerce")
    ask = pd.to_numeric(df["ask"], errors="coerce")
    mid = (bid + ask) / 2.0
    volume = pd.to_numeric(df.get("volume", pd.Series(index=df.index)), errors="coerce")
    oi = pd.to_numeric(df.get("open_interest", pd.Series(index=df.index)), errors="coerce")
    out: dict[str, dict[str, float | str]] = {}
    for sym, b, a, m, vol, open_interest in zip(df["option_symbol"], bid, ask, mid, volume, oi):
        out[str(sym)] = {
            "option_symbol": str(sym),
            "bid": safe_float(b),
            "ask": safe_float(a),
            "mid": safe_float(m),
            "volume": safe_float(vol, 0.0),
            "open_interest": safe_float(open_interest, 0.0),
        }
    return out


def _leg_symbols(row: pd.Series) -> tuple[str, str, str]:
    direction = str(row.get("direction", ""))
    right = "P" if direction == "Bull Put" else "C"
    expiry = row.get("expiry")
    ticker = str(row.get("ticker", "")).upper()
    short = str(row.get("short_leg_eod") or "").upper().strip()
    long = str(row.get("long_leg_eod") or "").upper().strip()
    if not short and isinstance(expiry, dt.date):
        short = build_occ_symbol(ticker, expiry, right, safe_float(row.get("short_strike_eod")))
    if not long and isinstance(expiry, dt.date):
        long = build_occ_symbol(ticker, expiry, right, safe_float(row.get("long_strike_eod")))
    return short, long, right


def _entry_quote(row: pd.Series, quotes: dict[str, dict[str, float | str]], slippage_pct: float) -> dict[str, Any]:
    short_sym, long_sym, _right = _leg_symbols(row)
    short = quotes.get(short_sym)
    long = quotes.get(long_sym)
    width = abs(safe_float(row.get("long_strike_eod")) - safe_float(row.get("short_strike_eod")))
    if short is None or long is None:
        return {
            "exact_fillable": False,
            "fill_reason": "missing_entry_leg_quote",
            "short_leg_eod": short_sym,
            "long_leg_eod": long_sym,
            "entry_width": width,
        }
    short_bid = safe_float(short.get("bid"))
    short_ask = safe_float(short.get("ask"))
    long_bid = safe_float(long.get("bid"))
    long_ask = safe_float(long.get("ask"))
    short_mid = safe_float(short.get("mid"))
    long_mid = safe_float(long.get("mid"))
    mid_credit = short_mid - long_mid
    natural_credit = short_bid - long_ask
    entry_credit = mid_credit * (1.0 - slippage_pct)
    q_width = max(
        (short_ask - short_bid) / short_mid if short_mid > 0 else math.inf,
        (long_ask - long_bid) / long_mid if long_mid > 0 else math.inf,
    )
    fillable = bool(
        math.isfinite(width)
        and width > 0
        and math.isfinite(entry_credit)
        and entry_credit > 0
        and entry_credit / width >= 0.12
        and math.isfinite(q_width)
        and q_width <= 0.80
    )
    reason = "filled_at_mid_less_slippage" if fillable else "failed_fill_credit_or_quote_width"
    return {
        "exact_fillable": fillable,
        "fill_reason": reason,
        "short_leg_eod": short_sym,
        "long_leg_eod": long_sym,
        "entry_width": width,
        "sell_leg_bid": short_bid,
        "sell_leg_ask": short_ask,
        "sell_leg_mid": short_mid,
        "buy_leg_bid": long_bid,
        "buy_leg_ask": long_ask,
        "buy_leg_mid": long_mid,
        "entry_mid_credit": mid_credit,
        "entry_natural_credit": natural_credit,
        "entry_credit": entry_credit,
        "entry_credit_pct_width": entry_credit / width if width > 0 else math.nan,
        "entry_quote_width_pct": q_width,
    }


def _spread_mid_debit(row: pd.Series, quotes: dict[str, dict[str, float | str]]) -> float:
    short_sym, long_sym, _right = _leg_symbols(row)
    short = quotes.get(short_sym)
    long = quotes.get(long_sym)
    if short is None or long is None:
        return math.nan
    return max(0.0, safe_float(short.get("mid")) - safe_float(long.get("mid")))


def _expiry_debit(row: pd.Series, close: float) -> float:
    direction = str(row.get("direction", ""))
    short = safe_float(row.get("short_strike_eod"))
    long = safe_float(row.get("long_strike_eod"))
    if direction == "Bull Put":
        short_intrinsic = max(0.0, short - close)
        long_intrinsic = max(0.0, long - close)
    elif direction == "Bear Call":
        short_intrinsic = max(0.0, close - short)
        long_intrinsic = max(0.0, close - long)
    else:
        return math.nan
    return max(0.0, min(abs(long - short), short_intrinsic - long_intrinsic))


def simulate_spread_exit(
    row: pd.Series,
    close_history: dict[dt.date, pd.DataFrame],
    quote_history: dict[dt.date, dict[str, dict[str, float | str]]],
    *,
    slippage_pct: float,
    profit_take_pct: float,
    stop_loss_mult: float,
) -> dict[str, Any]:
    expiry = row.get("expiry")
    asof = row.get("asof")
    if not isinstance(expiry, dt.date) or not isinstance(asof, dt.date):
        return {"exact_evaluated": False, "exact_reason": "missing_asof_or_expiry"}
    entry = _entry_quote(row, quote_history.get(asof, {}), slippage_pct)
    if not entry.get("exact_fillable"):
        return {**entry, "exact_evaluated": False, "exact_reason": entry.get("fill_reason")}
    entry_credit = safe_float(entry.get("entry_credit"))
    width = safe_float(entry.get("entry_width"))
    target_debit = entry_credit * (1.0 - profit_take_pct)
    stop_debit = entry_credit * stop_loss_mult
    quote_days_seen = 0
    for day in sorted(d for d in quote_history if asof < d <= expiry):
        debit_mid = _spread_mid_debit(row, quote_history[day])
        if not math.isfinite(debit_mid):
            continue
        quote_days_seen += 1
        exit_debit = debit_mid * (1.0 + slippage_pct)
        if exit_debit <= target_debit:
            pnl = entry_credit - exit_debit
            return {
                **entry,
                "exact_evaluated": True,
                "exit_day": day,
                "exit_reason": "profit_target",
                "exit_debit": exit_debit,
                "pnl_1x": pnl * 100.0,
                "return_on_risk": pnl / max(width - entry_credit, 0.01),
                "exact_win": pnl > 0,
                "quote_days_seen": quote_days_seen,
            }
        if exit_debit >= stop_debit:
            pnl = entry_credit - exit_debit
            return {
                **entry,
                "exact_evaluated": True,
                "exit_day": day,
                "exit_reason": "stop_loss",
                "exit_debit": exit_debit,
                "pnl_1x": pnl * 100.0,
                "return_on_risk": pnl / max(width - entry_credit, 0.01),
                "exact_win": pnl > 0,
                "quote_days_seen": quote_days_seen,
            }
    eval_day, close = future_close(close_history, str(row.get("ticker", "")).upper(), expiry)
    if not eval_day or not math.isfinite(close):
        return {**entry, "exact_evaluated": False, "exact_reason": "missing_expiry_close", "quote_days_seen": quote_days_seen}
    exit_debit = _expiry_debit(row, close)
    pnl = entry_credit - exit_debit
    return {
        **entry,
        "exact_evaluated": True,
        "exit_day": eval_day,
        "exit_reason": "expiry_settlement",
        "exit_debit": exit_debit,
        "pnl_1x": pnl * 100.0,
        "return_on_risk": pnl / max(width - entry_credit, 0.01),
        "exact_win": pnl > 0,
        "quote_days_seen": quote_days_seen,
    }


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    equity = values.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return float(drawdown.min())


def _split_metrics(df: pd.DataFrame, split_day: dt.date | None) -> dict[str, Any]:
    if df.empty:
        return {}
    out: dict[str, Any] = {}
    groups = {"all": df}
    if split_day is not None:
        groups["train"] = df[df["asof"] <= split_day]
        groups["test"] = df[df["asof"] > split_day]
    for name, part in groups.items():
        ev = part[part["exact_evaluated"].eq(True)].copy() if not part.empty else part
        out[name] = {
            "rows": int(len(part)),
            "evaluated": int(len(ev)),
            "win_rate": float(ev["exact_win"].mean()) if not ev.empty else None,
            "avg_pnl_1x": float(ev["pnl_1x"].mean()) if not ev.empty else None,
            "total_pnl_1x": float(ev["pnl_1x"].sum()) if not ev.empty else None,
            "max_drawdown_1x": _max_drawdown(ev.sort_values(["asof", "ticker"])["pnl_1x"]) if not ev.empty else None,
            "avg_entry_credit_pct_width": float(ev["entry_credit_pct_width"].mean()) if not ev.empty else None,
        }
    return out


def _truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _date_key(value: object) -> str:
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    return str(value or "")[:10]


def _fmt_money(value: object) -> str:
    number = safe_float(value)
    return f"${number:,.2f}" if math.isfinite(number) else ""


def _fmt_pct(value: object) -> str:
    number = safe_float(value)
    return f"{number:.1%}" if math.isfinite(number) else ""


def _confidence_icon(confidence: object) -> str:
    text = str(confidence or "")
    if text == "High":
        return "🟢"
    if text == "Medium":
        return "🟡"
    return "🔴"


def _leg_label(symbol: object) -> str:
    parsed = parse_occ_symbol(symbol)
    if parsed is None:
        return str(symbol or "")
    return f"{parsed.root} {parsed.expiry} {parsed.strike:g}{parsed.right}"


def _leg_quote_summary(row: pd.Series, prefix: str) -> str:
    return (
        f"bid {_fmt_money(row.get(f'{prefix}_bid'))} / "
        f"ask {_fmt_money(row.get(f'{prefix}_ask'))} / "
        f"mid {_fmt_money(row.get(f'{prefix}_mid'))}"
    )


def _earnings_days_from_record(row: pd.Series) -> int | None:
    try:
        asof = parse_date(_date_key(row.get("asof")))
        earnings = parse_date(_date_key(row.get("next_earnings_dt")))
    except Exception:
        return None
    if not asof or not earnings:
        return None
    return (earnings - asof).days


def _distance_expected(row: pd.Series) -> tuple[float, float, float]:
    direction = str(row.get("direction", ""))
    stock = safe_float(row.get("stock_price_eod"))
    short = safe_float(row.get("short_strike_eod"))
    dte = safe_float(row.get("dte"))
    iv30d = safe_float(row.get("iv30d"))
    if direction == "Bull Put" and math.isfinite(stock) and stock > 0 and math.isfinite(short):
        distance = (stock - short) / stock
    elif direction == "Bear Call" and math.isfinite(stock) and stock > 0 and math.isfinite(short):
        distance = (short - stock) / stock
    else:
        distance = math.nan
    expected = iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and math.isfinite(dte) and dte > 0 else math.nan
    ratio = distance / expected if math.isfinite(distance) and math.isfinite(expected) and expected > 0 else math.nan
    return distance, expected, ratio


def _flow_alignment(row: pd.Series) -> float:
    bias = safe_float(row.get("combined_flow_bias"), safe_float(row.get("flow_bias"), 0.0))
    return bias if str(row.get("direction")) == "Bull Put" else -bias


def _pop_proxy(row: pd.Series) -> float:
    _distance, _expected, ratio = _distance_expected(row)
    if not math.isfinite(ratio):
        return math.nan
    return max(0.50, min(0.90, 0.5 * (1.0 + math.erf(ratio / math.sqrt(2.0)))))


def _breakeven(row: pd.Series) -> float:
    credit = safe_float(row.get("entry_credit"))
    short = safe_float(row.get("short_strike_eod"))
    if not math.isfinite(credit) or not math.isfinite(short):
        return math.nan
    return short - credit if str(row.get("direction")) == "Bull Put" else short + credit


def _replay_trade_score(row: pd.Series) -> float:
    credit_pct = safe_float(row.get("entry_credit_pct_width"))
    quote_width = safe_float(row.get("entry_quote_width_pct"))
    total = safe_float(row.get("flow_total_premium"), 0.0)
    align = _flow_alignment(row)
    _distance, _expected, ratio = _distance_expected(row)

    score = 4.5
    if math.isfinite(credit_pct):
        if 0.18 <= credit_pct <= 0.30:
            score += 1.25
        elif credit_pct >= 0.16:
            score += 0.5
        else:
            score -= 1.0
    if math.isfinite(ratio):
        score += min(2.0, max(0.0, ratio))
    score += min(1.25, max(0.0, math.log10(max(total, 1.0)) - 7.0) * 0.65)
    score += min(1.0, max(0.0, align) * 4.0)
    if math.isfinite(quote_width):
        if quote_width <= 0.25:
            score += 0.5
        elif quote_width > 0.50:
            score -= 0.75
    earnings_days = _earnings_days_from_record(row)
    if earnings_days is not None and 0 <= earnings_days <= 7:
        score -= 0.5
    return round(max(0.0, min(10.0, score)), 2)


def _decision_sort_score(row: pd.Series) -> float:
    credit_pct = safe_float(row.get("entry_credit_pct_width"))
    _distance, _expected, ratio = _distance_expected(row)
    align = _flow_alignment(row)
    quote_width = safe_float(row.get("entry_quote_width_pct"))
    score = 0.0
    if math.isfinite(ratio):
        score += min(2.0, max(0.0, ratio))
    if math.isfinite(align):
        score += min(2.0, max(0.0, align) * 6.0)
    if math.isfinite(credit_pct):
        score += min(1.0, max(0.0, credit_pct - 0.16) * 8.0)
    if math.isfinite(quote_width):
        score -= max(0.0, quote_width - 0.35)
    return score


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


def apply_replay_decision_selection(detail: pd.DataFrame, *, max_selected_per_day: int = 8) -> pd.DataFrame:
    if detail.empty:
        return detail
    out = detail.copy()
    out["decision_pass"] = False
    out["decision_score"] = math.nan
    out["decision_reason"] = ""
    out["decision_tier"] = ""
    primary_eligible: list[tuple[int, str, float]] = []
    secondary_eligible: list[tuple[int, str, float]] = []
    for idx, row in out.iterrows():
        if not _truthy(row.get("exact_evaluated")):
            out.at[idx, "decision_reason"] = str(row.get("exact_reason") or row.get("fill_reason") or "not_exact_evaluated")
            continue
        credit_pct = safe_float(row.get("entry_credit_pct_width"))
        _distance, _expected, ratio = _distance_expected(row)
        align = _flow_alignment(row)
        dte = safe_float(row.get("dte"))
        earnings_days = _earnings_days_from_record(row)
        score = _decision_sort_score(row)
        out.at[idx, "decision_score"] = score
        if earnings_days is not None and 0 <= earnings_days <= 10:
            out.at[idx, "decision_reason"] = f"decision_earnings_within_10d:{earnings_days}"
        elif not math.isfinite(credit_pct) or credit_pct < 0.16:
            out.at[idx, "decision_reason"] = "decision_credit_below_16pct_width"
        elif _secondary_income_eligible(credit_pct=credit_pct, ratio=ratio, align=align, score=score, dte=dte) and ratio < 0.65:
            out.at[idx, "decision_reason"] = "decision_secondary_income_eligible"
            out.at[idx, "decision_tier"] = "secondary_income"
            secondary_eligible.append((idx, _date_key(row.get("asof")), score))
        elif not math.isfinite(ratio) or ratio < 0.65:
            out.at[idx, "decision_reason"] = "decision_insufficient_expected_move_buffer"
        elif not math.isfinite(align) or align < 0.10:
            out.at[idx, "decision_reason"] = "decision_weak_flow_alignment"
        elif credit_pct > 0.30:
            out.at[idx, "decision_reason"] = "decision_credit_above_30pct_width"
        else:
            out.at[idx, "decision_reason"] = "decision_eligible"
            out.at[idx, "decision_tier"] = "primary"
            primary_eligible.append((idx, _date_key(row.get("asof")), score))
    primary_by_day: dict[str, list[tuple[int, float]]] = {}
    for idx, day, score in primary_eligible:
        primary_by_day.setdefault(day, []).append((idx, score))
    selected_days: set[str] = set()
    for day, day_items in primary_by_day.items():
        selected_for_day = 0
        for idx, _score in sorted(day_items, key=lambda item: item[1], reverse=True):
            row = out.loc[idx]
            is_addon = selected_for_day > 0 and validated_addon_income_lane(
                row.get("direction"),
                safe_float(row.get("entry_credit_pct_width")),
            )
            if selected_for_day > 0 and not is_addon:
                continue
            out.at[idx, "decision_pass"] = True
            out.at[idx, "decision_reason"] = (
                "decision_selected_validated_addon_income_lane" if is_addon else "decision_selected_strongest_flow_aligned"
            )
            selected_for_day += 1
            selected_days.add(day)
            if selected_for_day >= max_selected_per_day:
                break
    secondary_by_day: dict[str, list[tuple[int, float]]] = {}
    for idx, day, score in secondary_eligible:
        secondary_by_day.setdefault(day, []).append((idx, score))
    for day, day_items in secondary_by_day.items():
        if day in selected_days:
            continue
        selected_for_day = 0
        for idx, _score in sorted(day_items, key=lambda item: item[1], reverse=True):
            row = out.loc[idx]
            is_addon = selected_for_day > 0 and validated_addon_income_lane(
                row.get("direction"),
                safe_float(row.get("entry_credit_pct_width")),
            )
            if selected_for_day > 0 and not is_addon:
                continue
            out.at[idx, "decision_pass"] = True
            out.at[idx, "decision_reason"] = (
                "decision_selected_validated_addon_income_lane" if is_addon else "decision_selected_secondary_income_sleeve"
            )
            selected_for_day += 1
            selected_days.add(day)
            if selected_for_day >= max_selected_per_day:
                break
    return out


def _replay_confidence(score: float) -> str:
    if score >= 7:
        return "High"
    if score >= 5:
        return "Medium"
    return "Reject"


def _replay_risk_notes(row: pd.Series) -> str:
    notes: list[str] = []
    earnings_days = _earnings_days_from_record(row)
    if earnings_days is not None and 0 <= earnings_days <= 7:
        notes.append(f"earnings proximity in source data: {earnings_days} day(s)")
    quote_width = safe_float(row.get("entry_quote_width_pct"))
    if math.isfinite(quote_width) and quote_width > 0.35:
        notes.append("wide historical bid/ask")
    if not notes:
        notes.append("historical exact-spread guard passed")
    notes.append(f"replay exit: {row.get('exit_reason')}")
    return "; ".join(notes)


def _replay_rejections(day_rows: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    counts = Counter()
    if day_rows.empty:
        return pd.DataFrame(columns=["reason", "count"])
    final_ids = set(final.index.tolist()) if not final.empty else set()
    for idx, row in day_rows.iterrows():
        if idx in final_ids:
            continue
        reason = str(row.get("replay_guard_reason") or row.get("exact_reason") or row.get("reason") or "not_guarded").strip()
        counts[reason or "not_guarded"] += 1
    return pd.DataFrame([{"reason": k, "count": v} for k, v in counts.most_common()])


def _near_miss_rows(day_rows: pd.DataFrame, final: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if day_rows.empty:
        return pd.DataFrame()
    final_ids = set(final.index.tolist()) if not final.empty else set()
    rows: list[dict[str, Any]] = []
    for idx, row in day_rows.iterrows():
        if idx in final_ids:
            continue
        credit = safe_float(row.get("entry_credit"))
        width = safe_float(row.get("entry_width"))
        pnl = safe_float(row.get("pnl_1x"))
        exact = _truthy(row.get("exact_evaluated"))
        reason = str(row.get("decision_reason") or row.get("replay_guard_reason") or row.get("exact_reason") or row.get("reason") or "not_guarded")
        rows.append(
            {
                "_rank": (1 if exact else 0) + (0.5 if math.isfinite(credit) else 0) + min(max(safe_float(row.get("entry_credit_pct_width"), 0.0), 0.0), 0.5),
                "Ticker": row.get("ticker"),
                "Direction": row.get("direction"),
                "Expiry": _date_key(row.get("expiry")),
                "Sell Leg": _leg_label(row.get("short_leg_eod")),
                "Buy Leg": _leg_label(row.get("long_leg_eod")),
                "Sell Leg Value": _leg_quote_summary(row, "sell_leg"),
                "Buy Leg Value": _leg_quote_summary(row, "buy_leg"),
                "Credit": _fmt_money(credit),
                "Width": _fmt_money(width),
                "Credit % Width": _fmt_pct(row.get("entry_credit_pct_width")),
                "Replay Result": f"{row.get('exit_reason')} { _fmt_money(pnl)}" if exact else str(row.get("exact_reason") or row.get("fill_reason") or ""),
                "Reject Reason": reason,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("_rank", ascending=False).drop(columns=["_rank"]).head(limit)


def write_replay_asof_report(detail: pd.DataFrame, out_dir: Path, asof: dt.date) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    asof_key = asof.isoformat()
    day_rows = detail[detail["asof"].map(_date_key).eq(asof_key)].copy() if not detail.empty and "asof" in detail.columns else pd.DataFrame()
    final = day_rows[day_rows.get("exact_evaluated", pd.Series(index=day_rows.index, dtype=bool)).map(_truthy)]
    if "decision_pass" in final.columns:
        final = final[final["decision_pass"].map(_truthy)].copy()
    else:
        final = final[final.get("replay_guard_pass", pd.Series(index=final.index, dtype=bool)).map(_truthy)].copy()

    rows: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(final.iterrows(), start=1):
        width = safe_float(row.get("entry_width"))
        credit = safe_float(row.get("entry_credit"))
        target_debit = credit * 0.40 if math.isfinite(credit) else math.nan
        stop_debit = credit * 2.00 if math.isfinite(credit) else math.nan
        max_profit = credit * 100.0 if math.isfinite(credit) else math.nan
        max_loss = (width - credit) * 100.0 if math.isfinite(width) and math.isfinite(credit) else math.nan
        score = _replay_trade_score(row)
        confidence = _replay_confidence(score)
        icon = _confidence_icon(confidence)
        rows.append(
            {
                "Rank": rank,
                "Status": f"{icon} {confidence}",
                "Ticker": row.get("ticker"),
                "Direction": row.get("direction"),
                "Strategy": row.get("strategy"),
                "Sell Leg": _leg_label(row.get("short_leg_eod")),
                "Buy Leg": _leg_label(row.get("long_leg_eod")),
                "Expiry": _date_key(row.get("expiry")),
                "DTE": row.get("dte"),
                "Credit": _fmt_money(credit),
                "Spread Width": _fmt_money(width),
                "Credit % Width": _fmt_pct(row.get("entry_credit_pct_width")),
                "Entry Limit Credit": _fmt_money(credit),
                "Target Close Debit": _fmt_money(target_debit),
                "Stop / Review Debit": _fmt_money(stop_debit),
                "Max Profit": _fmt_money(max_profit),
                "Max Loss": _fmt_money(max_loss),
                "Breakeven": f"{_breakeven(row):.2f}" if math.isfinite(_breakeven(row)) else "",
                "POP / Delta Proxy": _fmt_pct(_pop_proxy(row)),
                "Score": score,
                "Confidence": f"{icon} {confidence}",
                "Trade Conviction": f"{icon} {confidence} ({score:.2f}/10)",
                "Edge Type": f"{row.get('edge_type', '')}; {row.get('replay_guard_reason', '')}".strip("; "),
                "Risk Notes": _replay_risk_notes(row),
                "Position Size": f"1 contract; max risk {_fmt_money(max_loss)}",
                "Replay Result": f"{row.get('exit_reason')} on {_date_key(row.get('exit_day'))}; PnL {_fmt_money(row.get('pnl_1x'))}",
                "Sell Leg Value": _leg_quote_summary(row, "sell_leg"),
                "Buy Leg Value": _leg_quote_summary(row, "buy_leg"),
            }
        )
    final_table = pd.DataFrame(rows)
    final_table.to_csv(out_dir / f"codexuw_replay_final_trades_{asof_key}.csv", index=False)
    rejections = _replay_rejections(day_rows, final)
    rejections.to_csv(out_dir / f"codexuw_replay_rejections_{asof_key}.csv", index=False)
    near_misses = _near_miss_rows(day_rows, final)
    if not near_misses.empty:
        near_misses.to_csv(out_dir / f"codexuw_replay_near_misses_{asof_key}.csv", index=False)

    report = out_dir / f"codexuw_replay_trade_report_{asof_key}.md"
    exact_count = int(day_rows.get("exact_evaluated", pd.Series(index=day_rows.index, dtype=bool)).map(_truthy).sum()) if not day_rows.empty else 0
    lines = [
        f"# CodexUW Historical As-Of Replay - {asof_key}",
        "",
        "## Mode",
        "- UW folder is treated as the as-of signal source.",
        "- Historical hot-chain quotes are used for entry/exit replay.",
        "- This is not a live Schwab execution ticket; live mode must re-price current chains before entry.",
        "",
        "## Funnel",
        f"- replay_candidate_rows: {len(day_rows)}",
        f"- exact_spread_evaluated_rows: {exact_count}",
        f"- decision_selected_final_rows: {len(final_table)}",
        "",
    ]
    if final_table.empty:
        lines.extend(["## Final Decision", "", "No replay-validated high-quality trades for this date", ""])
        if not near_misses.empty:
            lines.extend(["## 🟡 Near-Miss Audit", "", near_misses.to_markdown(index=False), ""])
    else:
        lines.extend(["## Final Trades", "", final_table.to_markdown(index=False), ""])
        lines.extend(["## Trade Playbook", ""])
        for rank, (_, row) in enumerate(final.iterrows(), start=1):
            credit = safe_float(row.get("entry_credit"))
            width = safe_float(row.get("entry_width"))
            target_debit = credit * 0.40 if math.isfinite(credit) else math.nan
            stop_debit = credit * 2.00 if math.isfinite(credit) else math.nan
            max_profit = credit * 100.0 if math.isfinite(credit) else math.nan
            max_loss = (width - credit) * 100.0 if math.isfinite(width) and math.isfinite(credit) else math.nan
            score = _replay_trade_score(row)
            confidence = _replay_confidence(score)
            pop = _pop_proxy(row)
            icon = _confidence_icon(confidence)
            leg_rows = pd.DataFrame(
                [
                    {
                        "Action": "SELL TO OPEN",
                        "Leg": _leg_label(row.get("short_leg_eod")),
                        "Leg Value": _leg_quote_summary(row, "sell_leg"),
                        "Purpose": "short premium leg",
                    },
                    {
                        "Action": "BUY TO OPEN",
                        "Leg": _leg_label(row.get("long_leg_eod")),
                        "Leg Value": _leg_quote_summary(row, "buy_leg"),
                        "Purpose": "defined-risk hedge leg",
                    },
                ]
            )
            lines.extend(
                [
                    f"### {icon} Rank {rank} - {row.get('ticker')} {row.get('direction')} Credit Spread",
                    f"- 🟢 Entry order: SELL TO OPEN spread for {_fmt_money(credit)} credit limit.",
                    f"- 🟡 Profit target: BUY TO CLOSE spread at {_fmt_money(target_debit)} debit.",
                    f"- 🔴 Stop/review: BUY TO CLOSE spread near {_fmt_money(stop_debit)} debit.",
                    f"- 🔵 Expiration date: {_date_key(row.get('expiry'))}; DTE: {row.get('dte')}",
                    f"- {icon} Trade conviction: {confidence} ({score:.2f}/10); POP / delta proxy: {_fmt_pct(pop)}",
                    f"- 🟣 Edge / thesis: {row.get('edge_type', '')}; {row.get('replay_guard_reason', '')}",
                    "",
                    leg_rows.to_markdown(index=False),
                    "",
                    f"- 🔵 Net credit: {_fmt_money(credit)}; width: {_fmt_money(width)}; breakeven: {_breakeven(row):.2f}",
                    f"- 🔴 Risk: max profit {_fmt_money(max_profit)}; max loss {_fmt_money(max_loss)}; 1 contract",
                    f"- 🟡 Exit plan: take 60% profit near {_fmt_money(target_debit)} debit; stop/review near {_fmt_money(stop_debit)} debit.",
                    f"- 🔵 Replay result: {row.get('exit_reason')} on {_date_key(row.get('exit_day'))}; PnL {_fmt_money(row.get('pnl_1x'))}",
                    "",
                ]
            )
        total_risk = 0.0
        total_credit = 0.0
        for _, row in final.iterrows():
            width = safe_float(row.get("entry_width"))
            credit = safe_float(row.get("entry_credit"))
            if math.isfinite(width) and math.isfinite(credit):
                total_risk += (width - credit) * 100.0
                total_credit += credit * 100.0
        balance = final_table["Direction"].value_counts().to_dict()
        lines.extend(
            [
                "## Portfolio Summary",
                f"- Total 1-lot max risk: {_fmt_money(total_risk)}",
                f"- Total 1-lot expected credit: {_fmt_money(total_credit)}",
                f"- Bull/bear balance: {balance}",
                "",
            ]
        )
    lines.extend(["## Rejected Candidate Summary", ""])
    lines.append(rejections.head(12).to_markdown(index=False) if not rejections.empty else "_No rejected candidates._")
    lines.append("")
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def _guard_result(rec: dict[str, Any]) -> tuple[bool, str]:
    if not bool(rec.get("exact_evaluated")):
        return False, str(rec.get("exact_reason") or rec.get("fill_reason") or "not_exact_evaluated")
    credit_pct = safe_float(rec.get("entry_credit_pct_width"))
    if not math.isfinite(credit_pct) or credit_pct < 0.16:
        return False, "entry_credit_below_16pct_width"
    direction = str(rec.get("direction", ""))
    stock = safe_float(rec.get("stock_price_eod"))
    short = safe_float(rec.get("short_strike_eod"))
    iv30d = safe_float(rec.get("iv30d"))
    dte = safe_float(rec.get("dte"))
    if math.isfinite(stock) and stock > 0 and math.isfinite(short):
        distance = (stock - short) / stock if direction == "Bull Put" else (short - stock) / stock
    else:
        distance = math.nan
    expected = iv30d * math.sqrt(dte / 365.0) if math.isfinite(iv30d) and math.isfinite(dte) and dte > 0 else math.nan
    if direction == "Bull Put" and math.isfinite(distance) and math.isfinite(expected) and distance < expected * 0.55:
        return False, "replay_guard_bull_put_expected_move"
    align = safe_float(rec.get("combined_flow_bias"), 0.0)
    if (direction == "Bull Put" and align <= 0) or (direction == "Bear Call" and align >= 0):
        return False, "no_flow_edge_alignment"
    pattern_pass, pattern = replay_quality_pattern(
        direction=direction,
        trend=str(rec.get("regime", "")),
        credit_pct=credit_pct,
        distance_pct=distance,
        expected_move=expected,
    )
    return pattern_pass, pattern


def run_replay(
    root: Path,
    out_dir: Path,
    start: dt.date | None,
    end: dt.date | None,
    max_days: int,
    *,
    entry_start: dt.date | None = None,
    entry_end: dt.date | None = None,
    report_date: dt.date | None = None,
    max_tickers: int = 60,
    max_candidates: int = 50,
    max_eval_candidates: int = 50,
    bot_max_rows: int = 0,
    slippage_pct: float = 0.10,
    profit_take_pct: float = 0.60,
    stop_loss_mult: float = 2.0,
    max_selected_per_day: int = 8,
) -> Path:
    folders = dated_folders(root, start, end)
    if max_days > 0:
        folders = folders[-max_days:]
    entry_folders = [
        folder
        for folder in folders
        if (entry_start is None or infer_asof_date(folder) >= entry_start)
        and (entry_end is None or infer_asof_date(folder) <= entry_end)
    ]
    history = load_close_history(folders)
    hot_history = load_hot_history(folders)
    quote_history = {day: _quote_lookup(hot) for day, hot in hot_history.items()}
    rows: list[dict[str, Any]] = []
    day_summaries: list[dict[str, Any]] = []
    for folder in entry_folders:
        asof = infer_asof_date(folder)
        try:
            sc = load_stock_screener(folder)
            hot = load_hot_chains(folder, asof)
        except Exception as exc:
            day_summaries.append({"date": asof, "status": "load_error", "error": str(exc)})
            continue
        pool = select_ticker_pool(sc, max_tickers=max_tickers)
        try:
            bot_flow = aggregate_bot_flow(
                folder,
                pool["ticker"].tolist(),
                max_rows=bot_max_rows if bot_max_rows > 0 else None,
            )
        except Exception:
            bot_flow = pd.DataFrame()
        candidates = generate_candidates(pool, hot, bot_flow, asof=asof, max_candidates=max_candidates)
        if candidates.empty:
            day_summaries.append({"date": asof, "status": "no_candidates", "candidates": 0})
            continue
        regime = detect_regime(sc)
        # Replay the broad discovery layer, then let exact fill/replay guards
        # decide. This mirrors daily mode instead of over-pruning too early.
        candidates["credit_pct_proxy"] = candidates["estimated_eod_credit"] / candidates["preferred_width"].where(candidates["preferred_width"].abs() > 0)
        candidates = candidates[pd.to_numeric(candidates["credit_pct_proxy"], errors="coerce").fillna(0) >= 0.12].head(max_eval_candidates)
        wins = 0
        evaluated = 0
        exact_evaluated = 0
        exact_pnl = 0.0
        for _, cand in candidates.iterrows():
            cand = cand.copy()
            cand["asof"] = asof
            result = evaluate_candidate(cand, history)
            exact = simulate_spread_exit(
                cand,
                history,
                quote_history,
                slippage_pct=slippage_pct,
                profit_take_pct=profit_take_pct,
                stop_loss_mult=stop_loss_mult,
            )
            rec = cand.to_dict()
            rec.update({"regime": regime.get("trend"), **result, **exact})
            guard_pass, guard_reason = _guard_result(rec)
            rec["replay_guard_pass"] = guard_pass
            rec["replay_guard_reason"] = guard_reason
            rows.append(rec)
            if result.get("evaluated"):
                evaluated += 1
                wins += int(bool(result.get("win")))
            if exact.get("exact_evaluated"):
                exact_evaluated += 1
                exact_pnl += safe_float(exact.get("pnl_1x"), 0.0)
        day_summaries.append(
            {
                "date": asof,
                "status": "ok",
                "candidates": int(len(candidates)),
                "evaluated": int(evaluated),
                "wins": int(wins),
                "win_rate": wins / evaluated if evaluated else None,
                "exact_evaluated": int(exact_evaluated),
                "exact_pnl_1x": exact_pnl,
            }
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame(rows)
    detail = apply_replay_decision_selection(detail, max_selected_per_day=max_selected_per_day) if not detail.empty else detail
    summary = pd.DataFrame(day_summaries)
    detail.to_csv(out_dir / "codexuw_replay_detail.csv", index=False)
    summary.to_csv(out_dir / "codexuw_replay_by_day.csv", index=False)
    evaluated = detail[detail.get("evaluated", pd.Series(dtype=bool)).eq(True)] if not detail.empty else detail
    exact = detail[detail.get("exact_evaluated", pd.Series(dtype=bool)).eq(True)] if not detail.empty else detail
    guarded_exact = exact[exact.get("replay_guard_pass", pd.Series(dtype=bool)).eq(True)] if not exact.empty else exact
    decision_exact = exact[exact.get("decision_pass", pd.Series(dtype=bool)).eq(True)] if not exact.empty else exact
    ok_days = summary[summary["status"].eq("ok")]["date"].tolist() if not summary.empty and "date" in summary.columns else []
    split_day = ok_days[int(len(ok_days) * 0.7) - 1] if ok_days else None
    metrics = _split_metrics(exact, split_day)
    guarded_metrics = _split_metrics(guarded_exact, split_day)
    decision_metrics = _split_metrics(decision_exact, split_day)
    payload = {
        "root": str(root),
        "start": str(start) if start else "",
        "end": str(end) if end else "",
        "days": len(folders),
        "entry_days": len(entry_folders),
        "entry_start": str(entry_start) if entry_start else "",
        "entry_end": str(entry_end) if entry_end else "",
        "max_tickers": max_tickers,
        "max_candidates": max_candidates,
        "max_eval_candidates": max_eval_candidates,
        "max_selected_per_day": max_selected_per_day,
        "bot_max_rows": bot_max_rows,
        "evaluated": int(len(evaluated)),
        "win_rate": float(evaluated["win"].mean()) if not evaluated.empty else None,
        "avg_credit_pct_proxy": float(evaluated["credit_pct_proxy"].mean()) if not evaluated.empty else None,
        "slippage_pct": slippage_pct,
        "profit_take_pct": profit_take_pct,
        "stop_loss_mult": stop_loss_mult,
        "split_day": str(split_day) if split_day else "",
        "exact_metrics": metrics,
        "guarded_exact_metrics": guarded_metrics,
        "decision_exact_metrics": decision_metrics,
    }
    (out_dir / "codexuw_replay_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    report = out_dir / "codexuw_replay_report.md"
    lines = [
        "# CodexUW Replay Validation",
        "",
        f"- History days loaded: {payload['days']}",
        f"- Entry days scanned: {payload['entry_days']}",
        f"- Discovery settings: max tickers {max_tickers}; max candidates {max_candidates}; max exact-eval candidates {max_eval_candidates}",
        f"- Decision selection: strongest setup per day plus validated add-on Bear Call lane; max selected per day {max_selected_per_day}",
        f"- Breach-evaluated candidates: {payload['evaluated']}",
        f"- Breach win rate: {payload['win_rate']:.1%}" if payload["win_rate"] is not None else "- Breach win rate: n/a",
        f"- Exact-spread evaluated candidates: {metrics.get('all', {}).get('evaluated', 0)}",
        f"- Exact-spread win rate: {metrics.get('all', {}).get('win_rate'):.1%}" if metrics.get("all", {}).get("win_rate") is not None else "- Exact-spread win rate: n/a",
        f"- Exact-spread avg PnL/spread: ${metrics.get('all', {}).get('avg_pnl_1x'):,.2f}" if metrics.get("all", {}).get("avg_pnl_1x") is not None else "- Exact-spread avg PnL/spread: n/a",
        f"- Exact-spread max drawdown: ${metrics.get('all', {}).get('max_drawdown_1x'):,.2f}" if metrics.get("all", {}).get("max_drawdown_1x") is not None else "- Exact-spread max drawdown: n/a",
        f"- Guarded exact-spread evaluated candidates: {guarded_metrics.get('all', {}).get('evaluated', 0)}",
        f"- Guarded exact-spread win rate: {guarded_metrics.get('all', {}).get('win_rate'):.1%}" if guarded_metrics.get("all", {}).get("win_rate") is not None else "- Guarded exact-spread win rate: n/a",
        f"- Guarded exact-spread avg PnL/spread: ${guarded_metrics.get('all', {}).get('avg_pnl_1x'):,.2f}" if guarded_metrics.get("all", {}).get("avg_pnl_1x") is not None else "- Guarded exact-spread avg PnL/spread: n/a",
        f"- Guarded exact-spread max drawdown: ${guarded_metrics.get('all', {}).get('max_drawdown_1x'):,.2f}" if guarded_metrics.get("all", {}).get("max_drawdown_1x") is not None else "- Guarded exact-spread max drawdown: n/a",
        f"- Decision-selected exact-spread evaluated candidates: {decision_metrics.get('all', {}).get('evaluated', 0)}",
        f"- Decision-selected trade days: {int(decision_exact['asof'].map(_date_key).nunique()) if not decision_exact.empty and 'asof' in decision_exact.columns else 0}",
        f"- Decision-selected win rate: {decision_metrics.get('all', {}).get('win_rate'):.1%}" if decision_metrics.get("all", {}).get("win_rate") is not None else "- Decision-selected win rate: n/a",
        f"- Decision-selected avg PnL/spread: ${decision_metrics.get('all', {}).get('avg_pnl_1x'):,.2f}" if decision_metrics.get("all", {}).get("avg_pnl_1x") is not None else "- Decision-selected avg PnL/spread: n/a",
        f"- Decision-selected max drawdown: ${decision_metrics.get('all', {}).get('max_drawdown_1x'):,.2f}" if decision_metrics.get("all", {}).get("max_drawdown_1x") is not None else "- Decision-selected max drawdown: n/a",
        f"- Train/test split day: {payload['split_day'] or 'n/a'}",
        f"- Fill model: entry at mid less {slippage_pct:.0%}; exits at mid plus {slippage_pct:.0%}; {profit_take_pct:.0%} profit target; {stop_loss_mult:.1f}x credit stop; expiry settlement fallback.",
        "",
        "## Train/Test",
        "",
    ]
    split_rows = []
    for name in ["train", "test"]:
        item = metrics.get(name, {})
        split_rows.append(
            {
                "Split": name,
                "Rows": item.get("rows", 0),
                "Evaluated": item.get("evaluated", 0),
                "Win Rate": f"{item.get('win_rate'):.1%}" if item.get("win_rate") is not None else "",
                "Avg PnL": f"${item.get('avg_pnl_1x'):,.2f}" if item.get("avg_pnl_1x") is not None else "",
                "Total PnL": f"${item.get('total_pnl_1x'):,.2f}" if item.get("total_pnl_1x") is not None else "",
                "Max DD": f"${item.get('max_drawdown_1x'):,.2f}" if item.get("max_drawdown_1x") is not None else "",
            }
        )
    lines.append(pd.DataFrame(split_rows).to_markdown(index=False))
    lines.extend(["", "## Guarded Train/Test", ""])
    guarded_rows = []
    for name in ["train", "test"]:
        item = guarded_metrics.get(name, {})
        guarded_rows.append(
            {
                "Split": name,
                "Rows": item.get("rows", 0),
                "Evaluated": item.get("evaluated", 0),
                "Win Rate": f"{item.get('win_rate'):.1%}" if item.get("win_rate") is not None else "",
                "Avg PnL": f"${item.get('avg_pnl_1x'):,.2f}" if item.get("avg_pnl_1x") is not None else "",
                "Total PnL": f"${item.get('total_pnl_1x'):,.2f}" if item.get("total_pnl_1x") is not None else "",
                "Max DD": f"${item.get('max_drawdown_1x'):,.2f}" if item.get("max_drawdown_1x") is not None else "",
            }
        )
    lines.append(pd.DataFrame(guarded_rows).to_markdown(index=False))
    lines.extend(["", "## Decision-Selected Train/Test", ""])
    decision_rows = []
    for name in ["train", "test"]:
        item = decision_metrics.get(name, {})
        decision_rows.append(
            {
                "Split": name,
                "Rows": item.get("rows", 0),
                "Evaluated": item.get("evaluated", 0),
                "Win Rate": f"{item.get('win_rate'):.1%}" if item.get("win_rate") is not None else "",
                "Avg PnL": f"${item.get('avg_pnl_1x'):,.2f}" if item.get("avg_pnl_1x") is not None else "",
                "Total PnL": f"${item.get('total_pnl_1x'):,.2f}" if item.get("total_pnl_1x") is not None else "",
                "Max DD": f"${item.get('max_drawdown_1x'):,.2f}" if item.get("max_drawdown_1x") is not None else "",
            }
        )
    lines.append(pd.DataFrame(decision_rows).to_markdown(index=False))
    report.write_text("\n".join(lines), encoding="utf-8")
    if report_date is not None:
        write_replay_asof_report(detail, out_dir, report_date)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CodexUW historical replay sanity check")
    parser.add_argument("--root", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--entry-date", default="", help="Only generate entries for this as-of date while still loading future history")
    parser.add_argument("--entry-start", default="", help="Optional first as-of date for generated entries")
    parser.add_argument("--entry-end", default="", help="Optional last as-of date for generated entries")
    parser.add_argument("--report-date", default="", help="Write a per-date historical trade ticket report")
    parser.add_argument("--max-days", type=int, default=30)
    parser.add_argument("--max-tickers", type=int, default=60)
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--max-eval-candidates", type=int, default=50)
    parser.add_argument("--max-selected-per-day", type=int, default=8)
    parser.add_argument("--bot-max-rows", type=int, default=0)
    parser.add_argument("--slippage-pct", type=float, default=0.10)
    parser.add_argument("--profit-take-pct", type=float, default=0.60)
    parser.add_argument("--stop-loss-mult", type=float, default=2.0)
    args = parser.parse_args()
    entry_date = parse_date(args.entry_date)
    entry_start = entry_date or parse_date(args.entry_start)
    entry_end = entry_date or parse_date(args.entry_end)
    report_date = parse_date(args.report_date) or entry_date
    report = run_replay(
        Path(args.root).expanduser().resolve(),
        Path(args.out_dir).expanduser().resolve(),
        parse_date(args.start) or entry_date,
        parse_date(args.end),
        args.max_days,
        entry_start=entry_start,
        entry_end=entry_end,
        report_date=report_date,
        max_tickers=args.max_tickers,
        max_candidates=args.max_candidates,
        max_eval_candidates=args.max_eval_candidates,
        bot_max_rows=args.bot_max_rows,
        slippage_pct=args.slippage_pct,
        profit_take_pct=args.profit_take_pct,
        stop_loss_mult=args.stop_loss_mult,
        max_selected_per_day=args.max_selected_per_day,
    )
    print(f"Wrote: {report}")
    if report_date is not None:
        print(f"Wrote: {Path(args.out_dir).expanduser().resolve() / f'codexuw_replay_trade_report_{report_date}.md'}")


if __name__ == "__main__":
    main()
