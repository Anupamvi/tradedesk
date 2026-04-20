#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from uwos.exact_spread_backtester import (
    HistoricalOptionQuoteStore,
    LegQuote,
    UnderlyingCloseStore,
    build_occ_symbol,
    intrinsic_value,
    invalid_spread_value_reason,
    max_profit_max_loss,
    run_backtest as run_exact_spread_replay,
)


STRATEGY_RIGHT = {
    "Bull Call Debit": "C",
    "Bear Call Credit": "C",
    "Bear Put Debit": "P",
    "Bull Put Credit": "P",
}

STRATEGY_NET_TYPE = {
    "Bull Call Debit": "debit",
    "Bear Put Debit": "debit",
    "Bull Put Credit": "credit",
    "Bear Call Credit": "credit",
    "Iron Condor": "credit",
    "Iron Butterfly": "credit",
}

PASSING_REPLAY_VERDICTS = {"PASS", "PARTIAL_PASS", "ENTRY_OK"}
CONDOR_STRATEGIES = {"Iron Condor", "Iron Butterfly"}


def _safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _safe_date(value: Any) -> Optional[dt.date]:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return dt.datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "pass", "ok"}


def _strike_value(row: pd.Series, live_col: str, fallback_col: str) -> float:
    live = _safe_float(row.get(live_col))
    if _truthy(row.get("live_validated")) and math.isfinite(live):
        return live
    return _safe_float(row.get(fallback_col))


def _fmt_strike(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _quote_label(missing: List[str]) -> str:
    return "missing " + ", ".join(missing) + " quote" + ("" if len(missing) == 1 else "s")


def _latest_replay_exit_date(
    available_dates: Iterable[dt.date],
    *,
    signal_date: dt.date,
    expiry: dt.date,
    today: Optional[dt.date] = None,
) -> Optional[Tuple[dt.date, bool]]:
    today = today or dt.date.today()
    latest_available = None
    for day in sorted(set(available_dates)):
        if day <= signal_date or day > today:
            continue
        if latest_available is None or day > latest_available:
            latest_available = day
    if latest_available is None:
        return None
    if expiry <= latest_available:
        return expiry, True
    return latest_available, False


def _setup_row_from_candidate(
    idx: int,
    row: pd.Series,
    *,
    signal_date: dt.date,
    exit_date: dt.date,
) -> Optional[Dict[str, Any]]:
    strategy = str(row.get("strategy", "") or "").strip()
    right = STRATEGY_RIGHT.get(strategy)
    if not right:
        return None

    expiry = _safe_date(row.get("target_expiry"))
    ticker = str(row.get("ticker", "") or "").upper().strip()
    if not ticker or expiry is None:
        return None

    long_strike = _strike_value(row, "live_long_strike", "long_strike")
    short_strike = _strike_value(row, "live_short_strike", "short_strike")
    if not (math.isfinite(long_strike) and math.isfinite(short_strike)):
        return None

    net_type = STRATEGY_NET_TYPE.get(strategy, str(row.get("cost_type", "") or "").strip().lower())
    if net_type not in {"credit", "debit"}:
        return None

    width = abs(float(short_strike) - float(long_strike))
    if width <= 0:
        return None

    return {
        "trade_id": f"row-{idx}",
        "signal_date": signal_date,
        "ticker": ticker,
        "strategy": strategy,
        "expiry": expiry,
        "short_leg": build_occ_symbol(ticker, expiry, right, float(short_strike)),
        "long_leg": build_occ_symbol(ticker, expiry, right, float(long_strike)),
        "short_strike": float(short_strike),
        "long_strike": float(long_strike),
        "width": float(width),
        "net_type": net_type,
        "qty": 1.0,
        "entry_gate": "",
        "entry_net": math.nan,
        "exit_date": exit_date,
        "exit_net": math.nan,
    }


def _condor_legs_from_candidate(row: pd.Series) -> Optional[Dict[str, Any]]:
    strategy = str(row.get("strategy", "") or "").strip()
    if strategy not in CONDOR_STRATEGIES:
        return None
    ticker = str(row.get("ticker", "") or "").upper().strip()
    expiry = _safe_date(row.get("target_expiry"))
    if not ticker or expiry is None:
        return None

    put_short = _strike_value(row, "live_short_strike", "short_strike")
    call_short = _strike_value(row, "live_long_strike", "long_strike")
    width = _safe_float(row.get("spread_width"))
    if not (math.isfinite(put_short) and math.isfinite(call_short) and math.isfinite(width)):
        return None
    if width <= 0:
        return None

    put_long = put_short - width
    call_long = call_short + width
    if not (put_long < put_short <= call_short < call_long):
        return None

    return {
        "ticker": ticker,
        "expiry": expiry,
        "put_long_strike": float(put_long),
        "put_short_strike": float(put_short),
        "call_short_strike": float(call_short),
        "call_long_strike": float(call_long),
        "put_long_leg": build_occ_symbol(ticker, expiry, "P", float(put_long)),
        "put_short_leg": build_occ_symbol(ticker, expiry, "P", float(put_short)),
        "call_short_leg": build_occ_symbol(ticker, expiry, "C", float(call_short)),
        "call_long_leg": build_occ_symbol(ticker, expiry, "C", float(call_long)),
        "width": float(width),
    }


def _spread_entry_from_quotes(net_type: str, short_q: LegQuote, long_q: LegQuote) -> float:
    if net_type == "credit":
        return float(short_q.bid - long_q.ask)
    return float(long_q.ask - short_q.bid)


def _spread_exit_from_quotes(net_type: str, short_q: LegQuote, long_q: LegQuote) -> float:
    if net_type == "credit":
        return float(short_q.ask - long_q.bid)
    return float(long_q.bid - short_q.ask)


def _condor_entry_from_quotes(quotes: Dict[str, LegQuote]) -> float:
    put_credit = _spread_entry_from_quotes(
        "credit",
        quotes["put_short_leg"],
        quotes["put_long_leg"],
    )
    call_credit = _spread_entry_from_quotes(
        "credit",
        quotes["call_short_leg"],
        quotes["call_long_leg"],
    )
    return float(put_credit + call_credit)


def _condor_exit_from_quotes(quotes: Dict[str, LegQuote]) -> float:
    put_debit = _spread_exit_from_quotes(
        "credit",
        quotes["put_short_leg"],
        quotes["put_long_leg"],
    )
    call_debit = _spread_exit_from_quotes(
        "credit",
        quotes["call_short_leg"],
        quotes["call_long_leg"],
    )
    return float(put_debit + call_debit)


def _condor_value_at_expiry(legs: Dict[str, Any], spot: float) -> float:
    put_short = intrinsic_value("P", float(legs["put_short_strike"]), spot)
    put_long = intrinsic_value("P", float(legs["put_long_strike"]), spot)
    call_short = intrinsic_value("C", float(legs["call_short_strike"]), spot)
    call_long = intrinsic_value("C", float(legs["call_long_strike"]), spot)
    return float((put_short - put_long) + (call_short - call_long))


def _missing_quote_reason(
    quote_store: HistoricalOptionQuoteStore,
    asof: dt.date,
    legs: Dict[str, Any],
    labels: Iterable[str],
) -> str:
    missing: List[str] = []
    for label in labels:
        symbol = str(legs.get(label, "") or "")
        if not symbol or quote_store.get_leg_quote(asof, symbol) is None:
            missing.append(f"{label.replace('_leg', '')} {symbol or '-'}")
    if not missing:
        return ""
    return _quote_label(missing)


def _run_condor_replay(
    *,
    row: pd.Series,
    quote_store: HistoricalOptionQuoteStore,
    close_store: UnderlyingCloseStore,
    signal_date: dt.date,
    exit_date: dt.date,
    final: bool,
    close_lookback_days: int,
) -> Dict[str, Any]:
    legs = _condor_legs_from_candidate(row)
    if legs is None:
        return {
            "status": "invalid_setup",
            "status_reason": "could not build exact iron condor legs",
        }

    quote_labels = ["put_short_leg", "put_long_leg", "call_short_leg", "call_long_leg"]
    entry_quotes: Dict[str, LegQuote] = {}
    for label in quote_labels:
        quote = quote_store.get_leg_quote(signal_date, str(legs[label]))
        if quote is not None:
            entry_quotes[label] = quote
    if len(entry_quotes) != len(quote_labels):
        return {
            **legs,
            "entry_net": math.nan,
            "entry_source": "missing_quotes",
            "status": "skipped_invalid_entry_economics",
            "status_reason": _missing_quote_reason(quote_store, signal_date, legs, quote_labels)
            or "missing_entry_net_or_quotes",
        }

    entry_net = _condor_entry_from_quotes(entry_quotes)
    invalid_entry_reason = invalid_spread_value_reason(entry_net, float(legs["width"]), "entry")
    if invalid_entry_reason:
        return {
            **legs,
            "entry_net": entry_net,
            "entry_source": "quotes:conservative",
            "status": "skipped_invalid_entry_economics",
            "status_reason": invalid_entry_reason,
        }

    exit_net = math.nan
    exit_source = "missing_exit"
    exit_quotes: Dict[str, LegQuote] = {}
    for label in quote_labels:
        quote = quote_store.get_leg_quote(exit_date, str(legs[label]))
        if quote is not None:
            exit_quotes[label] = quote
    if len(exit_quotes) == len(quote_labels):
        exit_net = _condor_exit_from_quotes(exit_quotes)
        exit_source = "quotes:conservative"
    elif final:
        spot = close_store.get_close_on_or_before(
            str(legs["ticker"]),
            _safe_date(legs["expiry"]) or exit_date,
            lookback_days=close_lookback_days,
        )
        if spot is not None and math.isfinite(float(spot)):
            exit_net = _condor_value_at_expiry(legs, float(spot))
            exit_source = "expiry_intrinsic"

    if not math.isfinite(exit_net):
        return {
            **legs,
            "entry_net": float(entry_net),
            "entry_source": "quotes:conservative",
            "status": "failed_missing_exit_price",
            "status_reason": _missing_quote_reason(quote_store, exit_date, legs, quote_labels)
            or "no_exit_quotes_or_expiry_spot",
        }
    invalid_exit_reason = invalid_spread_value_reason(exit_net, float(legs["width"]), "exit")
    if invalid_exit_reason:
        return {
            **legs,
            "entry_net": float(entry_net),
            "entry_source": "quotes:conservative",
            "exit_net": float(exit_net),
            "exit_source": exit_source,
            "status": "failed_invalid_exit_economics",
            "status_reason": invalid_exit_reason,
        }

    width = float(legs["width"])
    max_profit, max_loss = max_profit_max_loss(width, float(entry_net), "credit")
    pnl = float((entry_net - exit_net) * 100.0)
    return {
        **legs,
        "entry_net": float(entry_net),
        "entry_source": "quotes:conservative",
        "exit_net": float(exit_net),
        "exit_source": exit_source,
        "pnl": pnl,
        "return_on_risk": float(pnl / max_loss) if max_loss > 0 else math.nan,
        "max_profit": float(max_profit),
        "max_loss": float(max_loss),
        "status": "completed",
        "status_reason": "ok",
    }


def _run_vertical_entry_check(
    *,
    idx: int,
    row: pd.Series,
    quote_store: HistoricalOptionQuoteStore,
    signal_date: dt.date,
) -> Dict[str, Any]:
    setup = _setup_row_from_candidate(idx, row, signal_date=signal_date, exit_date=signal_date)
    if setup is None:
        return {
            "trade_id": f"row-{idx}",
            "status": "invalid_setup",
            "status_reason": "could not build exact option legs",
        }

    legs = {
        "short_leg": str(setup.get("short_leg", "") or ""),
        "long_leg": str(setup.get("long_leg", "") or ""),
    }
    short_q = quote_store.get_leg_quote(signal_date, legs["short_leg"])
    long_q = quote_store.get_leg_quote(signal_date, legs["long_leg"])
    if short_q is None or long_q is None:
        return {
            **setup,
            "entry_net": math.nan,
            "entry_source": "missing_quotes",
            "status": "skipped_missing_entry",
            "status_reason": _missing_quote_reason(quote_store, signal_date, legs, ["short_leg", "long_leg"])
            or "missing_entry_net_or_quotes",
        }

    entry_net = _spread_entry_from_quotes(str(setup["net_type"]), short_q, long_q)
    if not math.isfinite(entry_net) or entry_net <= 0:
        return {
            **setup,
            "entry_net": entry_net,
            "entry_source": "quotes:conservative",
            "status": "skipped_missing_entry",
            "status_reason": "non_positive_entry_net",
        }

    return {
        **setup,
        "entry_net": float(entry_net),
        "entry_source": "quotes:conservative",
        "status": "entry_only_no_later_snapshot",
        "status_reason": "entry quotes available; no later option snapshot yet",
    }


def _run_condor_entry_check(
    *,
    row: pd.Series,
    quote_store: HistoricalOptionQuoteStore,
    signal_date: dt.date,
) -> Dict[str, Any]:
    legs = _condor_legs_from_candidate(row)
    if legs is None:
        return {
            "status": "invalid_setup",
            "status_reason": "could not build exact iron condor legs",
        }

    quote_labels = ["put_short_leg", "put_long_leg", "call_short_leg", "call_long_leg"]
    entry_quotes: Dict[str, LegQuote] = {}
    for label in quote_labels:
        quote = quote_store.get_leg_quote(signal_date, str(legs[label]))
        if quote is not None:
            entry_quotes[label] = quote
    if len(entry_quotes) != len(quote_labels):
        return {
            **legs,
            "entry_net": math.nan,
            "entry_source": "missing_quotes",
            "status": "skipped_missing_entry",
            "status_reason": _missing_quote_reason(quote_store, signal_date, legs, quote_labels)
            or "missing_entry_net_or_quotes",
        }

    entry_net = _condor_entry_from_quotes(entry_quotes)
    invalid_entry_reason = invalid_spread_value_reason(entry_net, float(legs["width"]), "entry")
    if invalid_entry_reason:
        return {
            **legs,
            "entry_net": entry_net,
            "entry_source": "quotes:conservative",
            "status": "skipped_invalid_entry_economics",
            "status_reason": invalid_entry_reason,
        }

    return {
        **legs,
        "entry_net": float(entry_net),
        "entry_source": "quotes:conservative",
        "status": "entry_only_no_later_snapshot",
        "status_reason": "entry quotes available; no later option snapshot yet",
    }


def _default_replay_columns() -> Dict[str, Any]:
    return {
        "quote_replay_verdict": "UNAVAILABLE",
        "quote_replay_status": "not_run",
        "quote_replay_reason": "",
        "quote_replay_signal_date": "",
        "quote_replay_exit_date": "",
        "quote_replay_final": False,
        "quote_replay_entry_net": math.nan,
        "quote_replay_entry_source": "",
        "quote_replay_exit_net": math.nan,
        "quote_replay_exit_source": "",
        "quote_replay_pnl": math.nan,
        "quote_replay_return_on_risk": math.nan,
        "quote_replay_max_profit": math.nan,
        "quote_replay_max_loss": math.nan,
        "quote_replay_days_held": 0,
    }


def _verdict_from_replay(row: pd.Series, *, final: bool) -> str:
    status = str(row.get("status", "") or "").strip()
    if status == "entry_only_no_later_snapshot":
        return "ENTRY_OK"
    if status != "completed":
        return "UNAVAILABLE"
    pnl = _safe_float(row.get("pnl"))
    if not math.isfinite(pnl):
        return "UNAVAILABLE"
    if final:
        return "PASS" if pnl > 0 else "FAIL"
    return "PARTIAL_PASS" if pnl >= 0 else "PARTIAL_FAIL"


def annotate_quote_replay(
    candidates: pd.DataFrame,
    *,
    root: Path,
    signal_date: dt.date,
    mode: str = "gate",
    close_lookback_days: int = 7,
    allow_web_fallback: bool = False,
    exit_date_override: Optional[dt.date] = None,
    quote_store: Optional[HistoricalOptionQuoteStore] = None,
    close_store: Optional[UnderlyingCloseStore] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Annotate vertical-spread candidates with daily option quote replay.

    The replay uses local UW daily snapshots only by default. Entry is priced
    from the signal-date option bid/ask; exit is priced from the latest
    available later snapshot or expiry intrinsic when expiry has passed.
    """
    if candidates.empty:
        return candidates.copy(), pd.DataFrame()

    mode_text = str(mode or "off").strip().lower()
    if mode_text == "off":
        return candidates.copy(), pd.DataFrame()

    out = candidates.copy()
    for col, value in _default_replay_columns().items():
        out[col] = value

    quote_store = quote_store or HistoricalOptionQuoteStore(root_dir=root, use_hot=True, use_oi=True)
    close_store = close_store or UnderlyingCloseStore(
        root_dir=root,
        allow_web_fallback=allow_web_fallback,
    )
    available_dates = quote_store.available_dates()

    setup_rows: List[Dict[str, Any]] = []
    condor_results: List[Dict[str, Any]] = []
    final_by_trade_id: Dict[str, bool] = {}
    source_index_by_trade_id: Dict[str, int] = {}

    for idx, row in out.iterrows():
        strategy = str(row.get("strategy", "") or "").strip()
        expiry = _safe_date(row.get("target_expiry"))
        if strategy not in STRATEGY_RIGHT and strategy not in CONDOR_STRATEGIES:
            out.at[idx, "quote_replay_status"] = "unsupported_strategy"
            out.at[idx, "quote_replay_reason"] = f"quote replay supports verticals and iron condors only, got {strategy or '-'}"
            continue
        if expiry is None:
            out.at[idx, "quote_replay_status"] = "invalid_expiry"
            out.at[idx, "quote_replay_reason"] = "missing or invalid target_expiry"
            continue
        if exit_date_override is not None:
            if expiry <= signal_date:
                exit_info = None
            elif expiry <= exit_date_override:
                exit_info = (expiry, True)
            elif exit_date_override > signal_date:
                exit_info = (exit_date_override, False)
            else:
                exit_info = None
        else:
            exit_info = _latest_replay_exit_date(
                available_dates,
                signal_date=signal_date,
                expiry=expiry,
            )
        if exit_info is None:
            if strategy in CONDOR_STRATEGIES:
                replay = _run_condor_entry_check(
                    row=row,
                    quote_store=quote_store,
                    signal_date=signal_date,
                )
            else:
                replay = _run_vertical_entry_check(
                    idx=int(idx),
                    row=row,
                    quote_store=quote_store,
                    signal_date=signal_date,
                )
            replay["trade_id"] = str(replay.get("trade_id", "") or f"row-{idx}")
            replay["signal_date"] = signal_date
            replay["ticker"] = str(row.get("ticker", "") or "").upper().strip()
            replay["strategy"] = strategy
            replay["expiry"] = expiry
            replay["net_type"] = STRATEGY_NET_TYPE.get(strategy, str(row.get("cost_type", "") or "").strip().lower())
            replay["qty"] = 1.0
            condor_results.append(replay)

            out.at[idx, "quote_replay_signal_date"] = signal_date.isoformat()
            out.at[idx, "quote_replay_final"] = False
            verdict = _verdict_from_replay(pd.Series(replay), final=False)
            out.at[idx, "quote_replay_verdict"] = verdict
            out.at[idx, "quote_replay_status"] = str(replay.get("status", "") or "")
            out.at[idx, "quote_replay_reason"] = str(replay.get("status_reason", "") or "")
            out.at[idx, "quote_replay_entry_net"] = _safe_float(replay.get("entry_net"))
            out.at[idx, "quote_replay_entry_source"] = str(replay.get("entry_source", "") or "")
            continue
        exit_date, final = exit_info
        if strategy in CONDOR_STRATEGIES:
            replay = _run_condor_replay(
                row=row,
                quote_store=quote_store,
                close_store=close_store,
                signal_date=signal_date,
                exit_date=exit_date,
                final=final,
                close_lookback_days=max(1, int(close_lookback_days)),
            )
            replay["trade_id"] = f"row-{idx}"
            replay["signal_date"] = signal_date
            replay["ticker"] = str(row.get("ticker", "") or "").upper().strip()
            replay["strategy"] = strategy
            replay["expiry"] = expiry
            replay["exit_date"] = exit_date
            replay["net_type"] = "credit"
            replay["qty"] = 1.0
            condor_results.append(replay)

            out.at[idx, "quote_replay_signal_date"] = signal_date.isoformat()
            out.at[idx, "quote_replay_exit_date"] = exit_date.isoformat()
            out.at[idx, "quote_replay_final"] = bool(final)
            verdict = _verdict_from_replay(pd.Series(replay), final=final)
            out.at[idx, "quote_replay_verdict"] = verdict
            out.at[idx, "quote_replay_status"] = str(replay.get("status", "") or "")
            out.at[idx, "quote_replay_reason"] = str(replay.get("status_reason", "") or "")
            out.at[idx, "quote_replay_entry_net"] = _safe_float(replay.get("entry_net"))
            out.at[idx, "quote_replay_entry_source"] = str(replay.get("entry_source", "") or "")
            out.at[idx, "quote_replay_exit_net"] = _safe_float(replay.get("exit_net"))
            out.at[idx, "quote_replay_exit_source"] = str(replay.get("exit_source", "") or "")
            out.at[idx, "quote_replay_pnl"] = _safe_float(replay.get("pnl"))
            out.at[idx, "quote_replay_return_on_risk"] = _safe_float(replay.get("return_on_risk"))
            out.at[idx, "quote_replay_max_profit"] = _safe_float(replay.get("max_profit"))
            out.at[idx, "quote_replay_max_loss"] = _safe_float(replay.get("max_loss"))
            out.at[idx, "quote_replay_days_held"] = max(0, int((exit_date - signal_date).days))
            continue

        setup = _setup_row_from_candidate(idx, row, signal_date=signal_date, exit_date=exit_date)
        if setup is None:
            out.at[idx, "quote_replay_status"] = "invalid_setup"
            out.at[idx, "quote_replay_reason"] = "could not build exact option legs"
            continue
        setup_rows.append(setup)
        final_by_trade_id[str(setup["trade_id"])] = bool(final)
        source_index_by_trade_id[str(setup["trade_id"])] = int(idx)
        out.at[idx, "quote_replay_signal_date"] = signal_date.isoformat()
        out.at[idx, "quote_replay_exit_date"] = exit_date.isoformat()
        out.at[idx, "quote_replay_final"] = bool(final)

    if not setup_rows:
        return out, pd.DataFrame(condor_results)

    replay_input = pd.DataFrame(setup_rows)
    replay_results = run_exact_spread_replay(
        replay_input,
        quote_store=quote_store,
        close_store=close_store,
        entry_source="quotes_only",
        entry_price_model="conservative",
        exit_mode="quotes_then_expiry",
        exit_price_model="conservative",
        close_lookback_days=max(1, int(close_lookback_days)),
    )

    for _, replay in replay_results.iterrows():
        trade_id = str(replay.get("trade_id", "") or "")
        if trade_id not in source_index_by_trade_id:
            continue
        idx = source_index_by_trade_id[trade_id]
        final = bool(final_by_trade_id.get(trade_id, False))
        verdict = _verdict_from_replay(replay, final=final)
        out.at[idx, "quote_replay_verdict"] = verdict
        out.at[idx, "quote_replay_status"] = str(replay.get("status", "") or "")
        reason = str(replay.get("status_reason", "") or "")
        if str(replay.get("status", "") or "") == "skipped_missing_entry":
            reason = _missing_quote_reason(
                quote_store,
                signal_date,
                {
                    "short_leg": str(replay.get("short_leg", "") or ""),
                    "long_leg": str(replay.get("long_leg", "") or ""),
                },
                ["short_leg", "long_leg"],
            ) or reason
        elif str(replay.get("status", "") or "") == "failed_missing_exit_price":
            reason = _missing_quote_reason(
                quote_store,
                _safe_date(replay.get("exit_date")) or exit_date,
                {
                    "short_leg": str(replay.get("short_leg", "") or ""),
                    "long_leg": str(replay.get("long_leg", "") or ""),
                },
                ["short_leg", "long_leg"],
            ) or reason
        out.at[idx, "quote_replay_reason"] = reason
        out.at[idx, "quote_replay_entry_net"] = _safe_float(replay.get("entry_net"))
        out.at[idx, "quote_replay_entry_source"] = str(replay.get("entry_source", "") or "")
        out.at[idx, "quote_replay_exit_net"] = _safe_float(replay.get("exit_net"))
        out.at[idx, "quote_replay_exit_source"] = str(replay.get("exit_source", "") or "")
        out.at[idx, "quote_replay_pnl"] = _safe_float(replay.get("pnl"))
        out.at[idx, "quote_replay_return_on_risk"] = _safe_float(replay.get("return_on_risk"))
        out.at[idx, "quote_replay_max_profit"] = _safe_float(replay.get("max_profit"))
        out.at[idx, "quote_replay_max_loss"] = _safe_float(replay.get("max_loss"))
        exit_date = _safe_date(replay.get("exit_date"))
        if exit_date is not None:
            out.at[idx, "quote_replay_exit_date"] = exit_date.isoformat()
            out.at[idx, "quote_replay_days_held"] = max(0, int((exit_date - signal_date).days))

    if condor_results:
        replay_results = pd.concat([replay_results, pd.DataFrame(condor_results)], ignore_index=True, sort=False)

    return out, replay_results


def quote_replay_passes(row: pd.Series) -> bool:
    verdict = str(row.get("quote_replay_verdict", "") or "").strip().upper()
    return verdict in PASSING_REPLAY_VERDICTS


def quote_replay_summary(df: pd.DataFrame) -> Dict[str, int]:
    if df.empty or "quote_replay_verdict" not in df.columns:
        return {}
    counts = df["quote_replay_verdict"].fillna("UNAVAILABLE").astype(str).str.upper().value_counts()
    return {str(k): int(v) for k, v in counts.to_dict().items()}
