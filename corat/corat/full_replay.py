"""Frozen, quota-guarded walk-forward replay for the complete CORAT decision path.

The command that exposes this module is plan-only unless ``--execute`` is
present.  Planning is pure local arithmetic: it never reads the API token,
constructs an ORATS client, or performs a network request.

An executed replay forms each decision after session T using the normal CORAT
pipeline, carries the selected vehicle and (for options) exact contract into
T+1, requires the underlying entry zone to trade, and then exits on the first
underlying stop/target or the holding horizon.  Exact option legs must exist on
both entry and exit dates; missing quotes are recorded rather than rebuilt.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from corat.clock import today_new_york
from corat.config import (
    PROJECT_ROOT,
    UniverseItem,
    discover_universe,
    load_universe,
    supporting_tickers,
)
from corat.constants import TARGET_TRADE
from corat.models import Bar, SourceTrace
from corat.orats import FetchBundle, OratsClient
from corat.pipeline import run_scan
from corat.store import (
    canonical_json,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json,
    write_text,
)
from corat.technical import bars_from_dailies


PLAN_SCHEMA = "corat.full_replay_plan.v1"
REPLAY_SCHEMA = "corat.full_replay.v1"
WEEKDAY_RATIO = 5.0 / 7.0


def _resolve(config: Mapping[str, Any], key: str) -> Path:
    path = Path(str(config[key]))
    return path if path.is_absolute() else PROJECT_ROOT / path


def _strategy_config_hash(config: Mapping[str, Any]) -> str:
    frozen = {
        key: value for key, value in config.items()
        if key not in {"_config_path", "_project_root", "output_root", "cache_root", "state_root", "universe_file"}
    }
    return sha256_bytes(canonical_json(frozen).encode("utf-8"))


def _universe_hash(config: Mapping[str, Any]) -> str:
    path = Path(str(config["universe_file"]))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return sha256_file(path)


def _number(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _day(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("{} must be YYYY-MM-DD".format(label)) from exc


def _validate_window(start: str, end: str, train_end: str, validation_end: str) -> None:
    start_day = _day(start, "start")
    end_day = _day(end, "end")
    train_day = _day(train_end, "train-end")
    validation_day = _day(validation_end, "validation-end")
    if not start_day <= train_day < validation_day < end_day:
        raise ValueError("replay dates must satisfy start <= train-end < validation-end < end")
    if end_day > _day(today_new_york(), "today"):
        raise ValueError("replay end date cannot be in the future")


def _estimated_sessions(start: str, end: str, spacing_sessions: int) -> int:
    calendar_days = (_day(end, "end") - _day(start, "start")).days + 1
    weekdays = max(1, int(math.ceil(calendar_days * WEEKDAY_RATIO)))
    return int(math.ceil(weekdays / float(max(1, spacing_sessions))))


def build_replay_plan(
    config: Mapping[str, Any],
    start: str,
    end: str,
    train_end: str,
    validation_end: str,
    tickers: Optional[Sequence[str]] = None,
    spacing_sessions: int = 1,
    assumed_triggers_per_date: int = 8,
    assumed_option_trades_per_date: int = 4,
    max_trades_per_date: int = 0,
    initial_nav: float = 100000.0,
    risk_pct: Optional[float] = None,
    max_open_positions: int = 0,
    minimum_test_trades: int = 40,
) -> Dict[str, Any]:
    """Return a conservative request plan without touching ORATS or secrets."""

    _validate_window(start, end, train_end, validation_end)
    if spacing_sessions <= 0:
        raise ValueError("spacing sessions must be positive")
    if assumed_triggers_per_date < 0 or assumed_option_trades_per_date < 0:
        raise ValueError("assumed trigger counts cannot be negative")
    effective_risk_pct = float(
        risk_pct if risk_pct is not None else (config.get("risk") or {}).get("normal_risk_pct") or 0.0075
    )
    if initial_nav <= 0 or not 0 < effective_risk_pct <= 1:
        raise ValueError("initial NAV must be positive and risk percent must be greater than zero and at most one")
    if max_trades_per_date < 0 or max_open_positions < 0 or minimum_test_trades <= 0:
        raise ValueError("trade/position caps cannot be negative and minimum test trades must be positive")
    configured = load_universe(config, tickers=tickers) if tickers else load_universe(config)
    discovery = config.get("discovery") if isinstance(config.get("discovery"), Mapping) else {}
    dynamic = bool(tickers is None and discovery.get("dynamic_orats_universe", True))
    equity_count = (
        int(discovery.get("maximum_equities") or 500)
        if dynamic
        else sum(1 for item in configured if item.kind in {"equity", "benchmark", "sector_etf"})
    )
    support_count = len(supporting_tickers(config, configured))
    universe_estimate = max(equity_count, support_count)
    batch_size = max(1, int((config.get("orats") or {}).get("batch_size") or 10))
    decision_dates = _estimated_sessions(start, end, spacing_sessions)
    assumed_triggers = min(universe_estimate, int(assumed_triggers_per_date))
    selected_option_trades = min(assumed_triggers, int(assumed_option_trades_per_date))
    if max_trades_per_date > 0:
        selected_option_trades = min(selected_option_trades, max_trades_per_date)

    # One SPY history request establishes real sessions. Remaining price
    # history is then fetched once for the union universe, not once per date.
    price_requests = 1 + int(math.ceil(max(0, universe_estimate - 1) / float(batch_size)))
    market_snapshot_requests = decision_dates * 3  # cores, ivrank, summaries
    signal_chain_requests = decision_dates * assumed_triggers
    exact_entry_exit_requests = decision_dates * selected_option_trades * 2
    # Core-volatility history and earnings history are memoized once per name.
    unique_triggered_estimate = min(
        universe_estimate,
        max(assumed_triggers, int(math.ceil(decision_dates * assumed_triggers * 0.35))),
    )
    history_requests = unique_triggered_estimate * 2
    base = price_requests + market_snapshot_requests
    expected = base + signal_chain_requests + exact_entry_exit_requests + history_requests
    recent_cutoff = (_day(today_new_york(), "today") - timedelta(days=4)).isoformat()
    recent_decision_estimate = min(decision_dates, int(math.ceil(4.0 / float(spacing_sessions)))) if end >= recent_cutoff else 0
    recent_fallback_requests = recent_decision_estimate * (3 + assumed_triggers)
    # This planning ceiling assumes every *assumed* trigger becomes an option
    # trade. It is not an absolute claim about how many triggers the data will
    # produce; the explicit per-run request budget is the actual hard ceiling.
    planned_ceiling = base + signal_chain_requests + decision_dates * assumed_triggers * 2 + history_requests + recent_fallback_requests
    policy = {
        "engine": "CORAT 0.3.0",
        "start": start,
        "end": end,
        "train_end": train_end,
        "validation_end": validation_end,
        "spacing_sessions": int(spacing_sessions),
        "tickers": sorted({str(value).upper() for value in (tickers or [])}),
        "dynamic_historical_universe": dynamic,
        "max_trades_per_date": int(max_trades_per_date),
        "max_open_positions": int(max_open_positions),
        "initial_nav": float(initial_nav),
        "risk_pct": effective_risk_pct,
        "minimum_test_trades": int(minimum_test_trades),
        "same_ticker_overlap": False,
        "assumed_triggers_per_date": int(assumed_triggers_per_date),
        "assumed_option_trades_per_date": int(assumed_option_trades_per_date),
        "strategy_config_sha256": _strategy_config_hash(config),
        "universe_sha256": _universe_hash(config),
    }
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "PLAN_ONLY_NOT_STARTED",
        "network_requests_made": 0,
        "token_read": False,
        "execution_requires_explicit_execute": True,
        "execution_requires_explicit_request_budget": True,
        "execution_requires_explicit_monthly_reserve": True,
        "execution_requires_console_confirmed_remaining": True,
        "policy": policy,
        "policy_sha256": sha256_bytes(canonical_json(policy).encode("utf-8")),
        "estimates": {
            "decision_dates": decision_dates,
            "universe_securities": universe_estimate,
            "price_history_requests": price_requests,
            "market_snapshot_requests": market_snapshot_requests,
            "signal_chain_requests": signal_chain_requests,
            "exact_entry_exit_requests": exact_entry_exit_requests,
            "history_and_earnings_requests": history_requests,
            "recent_endpoint_fallback_requests": recent_fallback_requests,
            "expected_requests": expected,
            "planned_request_ceiling": planned_ceiling,
        },
        "estimate_note": (
            "Request counts are planning estimates under the displayed trigger assumptions. Cache hits cost zero; "
            "more actual triggers or a larger historical union can increase demand. The explicit per-run budget is "
            "the hard ceiling, and an incomplete budget-limited run cannot pass the evidence gate."
        ),
    }


def authorize_replay(
    plan: Mapping[str, Any],
    usage: Mapping[str, Any],
    execute: bool,
    offline: bool,
    request_budget: Optional[int],
    monthly_reserve: Optional[int],
    confirmed_remaining: Optional[int] = None,
) -> Dict[str, Any]:
    """Fail closed before a client exists or a request can be made."""

    if not execute:
        return {"authorized": False, "reason": "PLAN_ONLY_NOT_STARTED", "network_budget": 0}
    if offline:
        if request_budget not in (None, 0):
            raise ValueError("offline replay request budget must be zero or omitted")
        return {"authorized": True, "reason": "CACHE_ONLY", "network_budget": 0}
    if request_budget is None or int(request_budget) <= 0:
        raise ValueError("online replay requires an explicit positive --request-budget")
    if monthly_reserve is None or int(monthly_reserve) < 0:
        raise ValueError("online replay requires an explicit nonnegative --monthly-reserve")
    if confirmed_remaining is None or int(confirmed_remaining) < 0:
        raise ValueError("online replay requires an explicit nonnegative --confirmed-remaining from the ORATS console")
    local_left = int(usage.get("left") if usage.get("left") is not None else max(0, int(usage.get("cap") or 0) - int(usage.get("used") or 0)))
    left = min(local_left, int(confirmed_remaining))
    spendable = max(0, left - int(monthly_reserve))
    if int(request_budget) > spendable:
        raise ValueError(
            "request budget {} exceeds spendable ORATS balance {} after reserve {}".format(
                int(request_budget), spendable, int(monthly_reserve)
            )
        )
    planned_ceiling = int((plan.get("estimates") or {}).get("planned_request_ceiling") or 0)
    if planned_ceiling > int(request_budget):
        raise ValueError(
            "planned request ceiling {} exceeds explicit request budget {}; reduce the window/assumptions or raise the cap deliberately".format(
                planned_ceiling, int(request_budget)
            )
        )
    return {
        "authorized": True,
        "reason": "EXPLICIT_ONLINE_EXECUTION",
        "network_budget": int(request_budget),
        "monthly_reserve": int(monthly_reserve),
        "usage_left_before": left,
        "local_usage_left_before": local_left,
        "console_confirmed_remaining": int(confirmed_remaining),
        "spendable_before": spendable,
    }


def local_orats_usage(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Read the local counter only; this function cannot make a request."""

    orats = config.get("orats") or {}
    cap = int(orats.get("monthly_request_cap") or 0)
    month = datetime.now().strftime("%Y-%m")
    payload = read_json(_resolve(config, "state_root") / "orats_usage.json", {}) or {}
    used = int(payload.get("used") or 0) if payload.get("month") == month else 0
    stored_cap = int(payload.get("cap") or cap)
    return {"month": month, "used": used, "cap": stored_cap, "left": max(0, stored_cap - used)}


def _row_for_leg(
    rows: Iterable[Mapping[str, Any]],
    expiration: str,
    strike: float,
    expected_trade_date: str = "",
) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if str(row.get("expirDate") or "")[:10] != expiration:
            continue
        row_date = str(row.get("tradeDate") or "")[:10]
        if expected_trade_date and row_date and row_date != expected_trade_date:
            continue
        candidate = _number(row.get("strike"))
        if candidate is not None and abs(candidate - strike) < 1e-6:
            return row
    return None


def exact_option_cashflow(
    rows: Sequence[Mapping[str, Any]],
    option: Mapping[str, Any],
    phase: str,
    improvement_fraction: float,
    expected_trade_date: str = "",
) -> Tuple[Optional[float], str]:
    """Return signed per-share cashflow for the exact original legs.

    Natural-side cashflow is moved only the supplied fraction toward midpoint.
    Entry BUYs are negative and entry SELLs positive. Exit actions are the
    exact inverse. This handles long options, debit spreads, and credit spreads
    with one auditable rule.
    """

    if phase not in {"ENTRY", "EXIT"}:
        raise ValueError("option cashflow phase must be ENTRY or EXIT")
    if not 0.0 <= float(improvement_fraction) <= 1.0:
        raise ValueError("improvement fraction must be between zero and one")
    legs = option.get("legs") or []
    if not isinstance(legs, list) or not legs:
        return None, "option has no exact legs"
    natural = 0.0
    midpoint = 0.0
    for leg in legs:
        if not isinstance(leg, Mapping):
            return None, "invalid option leg"
        action = str(leg.get("action") or "").upper()
        option_type = str(leg.get("option_type") or "").upper()
        expiration = str(leg.get("expiration") or option.get("expiration") or "")[:10]
        strike = _number(leg.get("strike"))
        quantity = max(1, int(_number(leg.get("quantity")) or 1))
        if action not in {"BUY", "SELL"} or option_type not in {"CALL", "PUT"} or strike is None or not expiration:
            return None, "invalid exact leg identity"
        row = _row_for_leg(rows, expiration, strike, expected_trade_date)
        if row is None:
            return None, "exact {} {} {} leg unavailable".format(expiration, strike, option_type)
        prefix = "call" if option_type == "CALL" else "put"
        bid = _number(row.get(prefix + "BidPrice"))
        ask = _number(row.get(prefix + "AskPrice"))
        if bid is None or ask is None or bid <= 0 or ask < bid:
            return None, "incoherent two-sided quote for {} {} {}".format(expiration, strike, option_type)
        mid = (bid + ask) / 2.0
        effective_action = action
        if phase == "EXIT":
            effective_action = "SELL" if action == "BUY" else "BUY"
        if effective_action == "BUY":
            natural -= ask * quantity
            midpoint -= mid * quantity
        else:
            natural += bid * quantity
            midpoint += mid * quantity
    cashflow = natural + float(improvement_fraction) * (midpoint - natural)
    return cashflow, "AVAILABLE"


def split_trade(signal_date: str, exit_date: str, train_end: str, validation_end: str) -> str:
    if signal_date <= train_end:
        return "TRAIN" if exit_date <= train_end else "EMBARGO_TRAIN_VALIDATION"
    if signal_date <= validation_end:
        return "VALIDATION" if exit_date <= validation_end else "EMBARGO_VALIDATION_TEST"
    return "TEST"


def resolve_underlying_path(
    bars: Sequence[Bar],
    direction: str,
    entry_low: float,
    entry_high: float,
    stop: float,
    target: float,
    holding_sessions: int,
    include_entry_session_for_exit: bool = True,
) -> Dict[str, Any]:
    """Resolve a next-session zone entry and conservative stop/target path."""

    if direction not in {"BULLISH", "BEARISH"}:
        raise ValueError("direction must be BULLISH or BEARISH")
    if holding_sessions <= 0:
        raise ValueError("holding sessions must be positive")
    if entry_low > entry_high:
        raise ValueError("entry low cannot exceed entry high")
    if not bars:
        return {"filled": False, "reason": "no next-session bar"}
    entry_bar = bars[0]
    overlap_low = max(float(entry_low), float(entry_bar.low))
    overlap_high = min(float(entry_high), float(entry_bar.high))
    if overlap_low > overlap_high:
        return {
            "filled": False,
            "reason": "next session did not trade the entry zone",
            "entry_date": entry_bar.date,
        }
    entry_price = overlap_high if direction == "BULLISH" else overlap_low
    path = list(
        bars[:holding_sessions]
        if include_entry_session_for_exit
        else bars[1 : holding_sessions + 1]
    )
    if not path:
        return {
            "filled": False,
            "reason": "no post-entry session is available for exit",
            "entry_date": entry_bar.date,
            "entry_zone_touched": True,
        }
    exit_price = float(path[-1].close)
    exit_date = path[-1].date
    exit_reason = "HOLDING_HORIZON_CLOSE"
    resolved_early = False
    for bar in path:
        # Daily OHLC cannot reveal intraday ordering. If both levels print,
        # debit the stop first instead of granting a favorable look-ahead.
        if direction == "BULLISH":
            if bar.low <= stop:
                exit_price, exit_date, exit_reason = float(stop), bar.date, "STOP_FIRST_CONSERVATIVE"
                resolved_early = True
                break
            if bar.high >= target:
                exit_price, exit_date, exit_reason = float(target), bar.date, "TARGET_1"
                resolved_early = True
                break
        else:
            if bar.high >= stop:
                exit_price, exit_date, exit_reason = float(stop), bar.date, "STOP_FIRST_CONSERVATIVE"
                resolved_early = True
                break
            if bar.low <= target:
                exit_price, exit_date, exit_reason = float(target), bar.date, "TARGET_1"
                resolved_early = True
                break
    if not resolved_early and len(path) < holding_sessions:
        return {
            "filled": False,
            "reason": "forward path is right-censored before the holding horizon",
            "entry_date": entry_bar.date,
            "entry_zone_touched": True,
            "available_exit_sessions": len(path),
        }
    return {
        "filled": True,
        "entry_zone_touched": True,
        "entry_date": entry_bar.date,
        "entry_price": entry_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "sessions_held": path.index(next(bar for bar in path if bar.date == exit_date)) + 1,
    }


def _copy_bundle(bundle: FetchBundle, rows: Optional[Sequence[Mapping[str, Any]]] = None) -> FetchBundle:
    return FetchBundle(
        rows=list(bundle.rows if rows is None else rows),
        traces=list(bundle.traces),
        errors=list(bundle.errors),
    )


class ReplayDataClient:
    """In-memory replay view over one hard-budgeted cache-first ORATS client."""

    def __init__(
        self,
        base: OratsClient,
        daily_bundle: FetchBundle,
        market_by_date: Mapping[str, Mapping[str, FetchBundle]],
        history_start: str,
        history_end: str,
    ) -> None:
        self.base = base
        self.daily_bundle = daily_bundle
        self.market_by_date = {
            day: {family: _copy_bundle(bundle) for family, bundle in families.items()}
            for day, families in market_by_date.items()
        }
        self.history_start = history_start
        self.history_end = history_end
        self._core_history: Dict[str, FetchBundle] = {}
        self._earnings: Dict[str, FetchBundle] = {}

    def usage(self) -> Dict[str, Any]:
        return self.base.usage()

    def fetch_dailies(
        self,
        tickers: Iterable[str],
        start_date: str,
        end_date: str,
        batch_size: int = 10,
    ) -> FetchBundle:
        del batch_size
        wanted = set(self.base.normalize_tickers(tickers))
        rows = [
            row for row in self.daily_bundle.rows
            if str(row.get("ticker") or "").upper() in wanted
            and start_date <= str(row.get("tradeDate") or "")[:10] <= end_date
        ]
        trace = SourceTrace(
            source="ORATS REPLAY PREFETCH",
            endpoint="/hist/dailies",
            status="IN_MEMORY_SLICE",
            fetched_at_utc="",
            latest_data_at=max((str(row.get("tradeDate") or "")[:10] for row in rows), default=""),
            rows=len(rows),
            cache_path="IN_MEMORY_REPLAY_PREFETCH",
            cache_sha256="",
            params={"tickers": str(len(wanted)), "start": start_date, "end": end_date},
        )
        return FetchBundle(rows=rows, traces=[trace], errors=[])

    def fetch_market_asof(self, family: str, as_of: str) -> FetchBundle:
        bundle = (self.market_by_date.get(as_of) or {}).get(family)
        if bundle is None:
            return FetchBundle(errors=["replay market {} snapshot was not prefetched for {}".format(family, as_of)])
        return _copy_bundle(bundle)

    def fetch_asof(
        self,
        family: str,
        tickers: Iterable[str],
        as_of: str,
        batch_size: int = 10,
        allow_current: bool = True,
    ) -> FetchBundle:
        del batch_size, allow_current
        bundle = self.fetch_market_asof(family, as_of)
        wanted = set(self.base.normalize_tickers(tickers))
        bundle.rows = [row for row in bundle.rows if str(row.get("ticker") or "").upper() in wanted]
        return bundle

    def fetch_chain(self, ticker: str, as_of: str, min_dte: int, max_dte: int) -> FetchBundle:
        return self.base.fetch_chain(ticker, as_of, min_dte, max_dte)

    def fetch_historical_chain_full(self, ticker: str, trade_date: str, max_dte: int = 120) -> FetchBundle:
        return self.base.fetch_historical_chain_full(ticker, trade_date, max_dte)

    def fetch_core_history(self, ticker: str, start_date: str, end_date: str) -> FetchBundle:
        name = self.base.normalize_tickers([ticker])[0]
        if name not in self._core_history:
            self._core_history[name] = self.base.fetch_core_history(name, self.history_start, self.history_end)
        bundle = _copy_bundle(self._core_history[name])
        bundle.rows = [
            row for row in bundle.rows
            if start_date <= str(row.get("tradeDate") or "")[:10] <= end_date
        ]
        return bundle

    def fetch_earnings(self, ticker: str) -> FetchBundle:
        name = self.base.normalize_tickers([ticker])[0]
        if name not in self._earnings:
            self._earnings[name] = self.base.fetch_earnings(name)
        return _copy_bundle(self._earnings[name])


def _profit_factor(values: Sequence[float]) -> Optional[float]:
    gains = sum(value for value in values if value > 0)
    losses = abs(sum(value for value in values if value < 0))
    return gains / losses if losses > 0 else (float("inf") if gains > 0 else None)


def replay_metrics(rows: Sequence[Mapping[str, Any]], pnl_key: str = "unit_pnl_dollars") -> Dict[str, Any]:
    completed = [row for row in rows if _number(row.get(pnl_key)) is not None]
    values = [float(row[pnl_key]) for row in completed]
    returns = [float(row["return_on_risk"]) for row in completed if _number(row.get("return_on_risk")) is not None]
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    expected = statistics.mean(values) if values else None
    standard_error = statistics.stdev(values) / math.sqrt(len(values)) if len(values) >= 2 else None
    lower = expected - 1.96 * standard_error if expected is not None and standard_error is not None else None
    upper = expected + 1.96 * standard_error if expected is not None and standard_error is not None else None
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for row in sorted(completed, key=lambda item: (str(item.get("exit_date") or ""), str(item.get("signal_date") or ""))):
        equity += float(row[pnl_key])
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    calibration_rows = [
        row for row in completed
        if _number(row.get("predicted_pop")) is not None
        and 0.0 <= float(row["predicted_pop"]) <= 1.0
    ]
    brier = statistics.mean(
        (float(row["predicted_pop"]) - (1.0 if float(row[pnl_key]) > 0 else 0.0)) ** 2
        for row in calibration_rows
    ) if calibration_rows else None
    return {
        "n": len(completed),
        "win_rate": len(wins) / float(len(values)) if values else None,
        "expectancy_dollars": expected,
        "median_pnl_dollars": statistics.median(values) if values else None,
        "average_winner_dollars": statistics.mean(wins) if wins else None,
        "average_loser_dollars": statistics.mean(losses) if losses else None,
        "profit_factor": _profit_factor(values),
        "average_return_on_risk": statistics.mean(returns) if returns else None,
        "total_pnl_dollars": sum(values),
        "max_drawdown_dollars": drawdown,
        "standard_error_dollars": standard_error,
        "expectancy_lower_95_dollars": lower,
        "expectancy_upper_95_dollars": upper,
        "pop_calibration_n": len(calibration_rows),
        "brier_score": brier,
    }


def _group_metrics(rows: Sequence[Mapping[str, Any]], key: str, pnl_key: str = "unit_pnl_dollars") -> Dict[str, Any]:
    values = sorted({str(row.get(key) or "DATA UNAVAILABLE") for row in rows})
    return {
        value: replay_metrics([row for row in rows if str(row.get(key) or "DATA UNAVAILABLE") == value], pnl_key)
        for value in values
    }


def _months(start: str, end: str) -> List[str]:
    current = _day(start, "month start").replace(day=1)
    final = _day(end, "month end").replace(day=1)
    values: List[str] = []
    while current <= final:
        values.append(current.strftime("%Y-%m"))
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
    return values


def monthly_pnl_summary(
    rows: Sequence[Mapping[str, Any]],
    start: str,
    end: str,
    pnl_key: str = "sized_pnl_dollars",
) -> Dict[str, Any]:
    """Aggregate P/L by signal cohort month, including zero-trade months."""

    month_values = _months(start, end)
    pnl = {month: 0.0 for month in month_values}
    counts = {month: 0 for month in month_values}
    for row in rows:
        signal_date = str(row.get("signal_date") or "")[:10]
        value = _number(row.get(pnl_key))
        if value is None or not start <= signal_date <= end:
            continue
        month = signal_date[:7]
        if month in pnl:
            pnl[month] += value
            counts[month] += 1
    series = [{"month": month, "pnl_dollars": pnl[month], "trades": counts[month]} for month in month_values]
    values = [row["pnl_dollars"] for row in series]
    return {
        "basis": "signal-month cohort; includes zero-trade months",
        "months": len(series),
        "average_monthly_pnl_dollars": statistics.mean(values) if values else None,
        "median_monthly_pnl_dollars": statistics.median(values) if values else None,
        "worst_month_pnl_dollars": min(values) if values else None,
        "best_month_pnl_dollars": max(values) if values else None,
        "positive_month_rate": sum(1 for value in values if value > 0) / float(len(values)) if values else None,
        "zero_trade_months": sum(1 for row in series if row["trades"] == 0),
        "total_pnl_dollars": sum(values),
        "series": series,
    }


def _maximum_risk(option: Mapping[str, Any], entry_cashflow: float, commission: float) -> Optional[float]:
    legs = option.get("legs") or []
    if not legs:
        return None
    round_trip_commission = len(legs) * 2.0 * max(0.0, commission)
    debit_credit = str(option.get("debit_credit") or "").upper()
    if debit_credit == "DEBIT" and entry_cashflow < 0:
        return -entry_cashflow * 100.0 + round_trip_commission
    if debit_credit == "CREDIT" and entry_cashflow > 0 and len(legs) >= 2:
        strikes = [float(leg["strike"]) for leg in legs if _number(leg.get("strike")) is not None]
        if len(strikes) >= 2:
            width = max(strikes) - min(strikes)
            return max(0.01, width * 100.0 - entry_cashflow * 100.0 + round_trip_commission)
    return None


def _trade_quantity(
    vehicle: str,
    risk_dollars: float,
    maximum_risk: float,
    entry_price: float,
    nav: float,
) -> int:
    if risk_dollars <= 0 or maximum_risk <= 0:
        return 0
    units = int(risk_dollars / maximum_risk)
    if vehicle == "STOCK" and entry_price > 0:
        units = min(units, int(nav / entry_price))
    return max(0, units)


def _resolve_candidate(
    candidate: Mapping[str, Any],
    signal_date: str,
    future_bars: Sequence[Bar],
    client: ReplayDataClient,
    nav: float,
    risk_pct: float,
    commission: float,
    train_end: str,
    validation_end: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], List[SourceTrace], List[str]]:
    setup = candidate.get("setup") or {}
    plan = candidate.get("stock_plan") or {}
    vehicle = str(candidate.get("vehicle") or "")
    ticker = str(candidate.get("ticker") or "").upper()
    direction = str(setup.get("direction") or "")
    required = ("entry_low", "entry_high", "stop", "target_1", "holding_sessions")
    if any(_number(plan.get(key)) is None for key in required):
        return None, {"ticker": ticker, "signal_date": signal_date, "reason": "missing exact stock risk plan"}, [], []
    path = resolve_underlying_path(
        future_bars,
        direction,
        float(plan["entry_low"]),
        float(plan["entry_high"]),
        float(plan["stop"]),
        float(plan["target_1"]),
        int(plan["holding_sessions"]),
        include_entry_session_for_exit=vehicle == "STOCK",
    )
    if not path.get("filled"):
        return None, {
            "ticker": ticker,
            "signal_date": signal_date,
            "vehicle": vehicle,
            "reason": path.get("reason"),
            "entry_date": path.get("entry_date"),
        }, [], []
    entry_price = float(path["entry_price"])
    exit_price = float(path["exit_price"])
    sign = 1.0 if direction == "BULLISH" else -1.0
    predicted = candidate.get("economics") or {}
    base: Dict[str, Any] = {
        "ticker": ticker,
        "sector": candidate.get("sector"),
        "setup": setup.get("name"),
        "direction": direction,
        "vehicle": vehicle,
        "strategy": (candidate.get("option") or {}).get("strategy") if vehicle == "OPTIONS" else "STOCK",
        "signal_date": signal_date,
        "entry_date": path["entry_date"],
        "exit_date": path["exit_date"],
        "exit_reason": path["exit_reason"],
        "sessions_held": path["sessions_held"],
        "underlying_entry": entry_price,
        "underlying_exit": exit_price,
        "stop": float(plan["stop"]),
        "target_1": float(plan["target_1"]),
        "predicted_pop": predicted.get("modeled_pop"),
        "predicted_expected_profit": (
            predicted.get("expected_profit_dollars")
            if vehicle == "OPTIONS"
            else predicted.get("expected_profit_per_share")
        ),
        "decision_score": candidate.get("score"),
        "decision_status": candidate.get("status"),
        "policy_hash": "",
    }
    traces: List[SourceTrace] = []
    errors: List[str] = []
    if vehicle == "STOCK":
        maximum_risk = abs(entry_price - float(plan["stop"]))
        unit_pnl = sign * (exit_price - entry_price)
        quantity = _trade_quantity(vehicle, nav * risk_pct, maximum_risk, entry_price, nav)
        base.update(
            {
                "entry_cashflow": -entry_price if direction == "BULLISH" else entry_price,
                "exit_cashflow": exit_price if direction == "BULLISH" else -exit_price,
                "maximum_risk_dollars": maximum_risk,
                "unit_pnl_dollars": unit_pnl,
                "return_on_risk": unit_pnl / maximum_risk if maximum_risk > 0 else None,
                "quantity": quantity,
                "sized_pnl_dollars": unit_pnl * quantity,
                "portfolio_included": quantity > 0,
                "option_expiration": "",
                "option_legs": "",
            }
        )
    elif vehicle == "OPTIONS":
        option = candidate.get("option") or {}
        entry_bundle = client.fetch_historical_chain_full(ticker, str(path["entry_date"]), max_dte=120)
        exit_bundle = (
            entry_bundle
            if path["exit_date"] == path["entry_date"]
            else client.fetch_historical_chain_full(ticker, str(path["exit_date"]), max_dte=120)
        )
        traces.extend(entry_bundle.traces)
        if exit_bundle is not entry_bundle:
            traces.extend(exit_bundle.traces)
        errors.extend(entry_bundle.errors)
        if exit_bundle is not entry_bundle:
            errors.extend(exit_bundle.errors)
        entry_cashflow, entry_reason = exact_option_cashflow(
            list(entry_bundle.rows), option, "ENTRY", 0.50, str(path["entry_date"])
        )
        exit_cashflow, exit_reason = exact_option_cashflow(
            list(exit_bundle.rows), option, "EXIT", 0.25, str(path["exit_date"])
        )
        if entry_cashflow is None or exit_cashflow is None:
            return None, {
                "ticker": ticker,
                "signal_date": signal_date,
                "vehicle": vehicle,
                "entry_date": path["entry_date"],
                "exit_date": path["exit_date"],
                "reason": entry_reason if entry_cashflow is None else exit_reason,
            }, traces, errors
        maximum_risk = _maximum_risk(option, entry_cashflow, commission)
        if maximum_risk is None:
            return None, {
                "ticker": ticker,
                "signal_date": signal_date,
                "vehicle": vehicle,
                "reason": "exact entry cashflow is inconsistent with selected debit/credit structure",
            }, traces, errors
        leg_count = len(option.get("legs") or [])
        commissions = 2.0 * leg_count * max(0.0, commission)
        unit_pnl = (entry_cashflow + exit_cashflow) * 100.0 - commissions
        quantity = _trade_quantity(vehicle, nav * risk_pct, maximum_risk, entry_price, nav)
        leg_text = "; ".join(
            "{} {} {} {}".format(leg.get("action"), leg.get("quantity"), leg.get("option_type"), leg.get("strike"))
            for leg in option.get("legs") or []
        )
        base.update(
            {
                "entry_cashflow": entry_cashflow,
                "exit_cashflow": exit_cashflow,
                "maximum_risk_dollars": maximum_risk,
                "unit_pnl_dollars": unit_pnl,
                "return_on_risk": unit_pnl / maximum_risk,
                "quantity": quantity,
                "sized_pnl_dollars": unit_pnl * quantity,
                "portfolio_included": quantity > 0,
                "commissions_per_unit": commissions,
                "option_expiration": option.get("expiration"),
                "option_legs": leg_text,
                "entry_fill_rule": "natural plus 50% improvement toward midpoint",
                "exit_fill_rule": "natural plus 25% improvement toward midpoint",
            }
        )
    else:
        return None, {"ticker": ticker, "signal_date": signal_date, "reason": "unsupported selected vehicle {}".format(vehicle)}, [], []
    base["split"] = split_trade(signal_date, str(base["exit_date"]), train_end, validation_end)
    return base, None, traces, errors


def _trace_dict(trace: Any) -> Dict[str, Any]:
    if isinstance(trace, SourceTrace):
        return trace.to_dict()
    return dict(trace) if isinstance(trace, Mapping) else {"error": str(trace)}


def _dedupe_trace_dicts(traces: Iterable[Any]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    seen = set()
    for value in traces:
        trace = _trace_dict(value)
        key = (
            str(trace.get("endpoint") or ""),
            str(trace.get("cache_path") or ""),
            str(trace.get("status") or ""),
            canonical_json(trace.get("params") or {}),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(trace)
    return result


def _fmt_money(value: Any) -> str:
    number = _number(value)
    return "DATA UNAVAILABLE" if number is None else "${:,.2f}".format(number)


def _fmt_pct(value: Any) -> str:
    number = _number(value)
    return "DATA UNAVAILABLE" if number is None else "{:.1%}".format(number)


def _fmt_number(value: Any) -> str:
    number = _number(value)
    if number is None:
        return "DATA UNAVAILABLE"
    if math.isinf(number):
        return "inf"
    return "{:.2f}".format(number)


def render_full_replay(report: Mapping[str, Any]) -> str:
    lines = [
        "# CORAT Frozen Full-Pipeline Replay",
        "",
        "Status: **{}**".format(report.get("status")),
        "",
        "This is historical research, not an order, a fill, or proof of future profit. Each signal uses the normal CORAT decision path after session T. Entry is attempted only in the next session's underlying zone. Options retain the exact expiry and strikes selected at T and require exact ORATS quotes on entry and exit; missing legs are never reconstructed.",
        "",
        "Policy hash: `{}`  ".format(report.get("policy_sha256")),
        "Window: {} through {}  ".format(report.get("start"), report.get("end")),
        "Frozen splits: train through {}; validation through {}; test after validation. Boundary-crossing trades are embargoed.  ".format(report.get("train_end"), report.get("validation_end")),
        "Decision dates / target opportunities / completed / missed: {} / {} / {} / {}  ".format(
            report.get("decision_dates"), report.get("target_opportunities"), report.get("completed"), report.get("missed")
        ),
        "ORATS requests this replay: {} (hard cap {}; reserve {})  ".format(
            (report.get("orats_usage") or {}).get("run_requests"),
            (report.get("authorization") or {}).get("network_budget"),
            (report.get("authorization") or {}).get("monthly_reserve", "cache-only"),
        ),
        "",
        "## Frozen split results",
        "",
        "| Split | N | Win | EV / unit | 95% EV lower | Return/risk | PF | Unit P/L | Max DD | POP Brier |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("TRAIN", "VALIDATION", "TEST", "EMBARGO_TRAIN_VALIDATION", "EMBARGO_VALIDATION_TEST"):
        metrics = (report.get("metrics") or {}).get("by_split", {}).get(split)
        if not metrics:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                split,
                metrics.get("n"),
                _fmt_pct(metrics.get("win_rate")),
                _fmt_money(metrics.get("expectancy_dollars")),
                _fmt_money(metrics.get("expectancy_lower_95_dollars")),
                _fmt_pct(metrics.get("average_return_on_risk")),
                _fmt_number(metrics.get("profit_factor")),
                _fmt_money(metrics.get("total_pnl_dollars")),
                _fmt_money(metrics.get("max_drawdown_dollars")),
                _fmt_number(metrics.get("brier_score")),
            )
        )
    gate = report.get("historical_evidence_gate") or {}
    lines.extend(
        [
            "",
            "Historical test-evidence gate: **{}**  ".format("PASS" if gate.get("passed") else "FAIL"),
            "Production promotion: **FALSE**. Historical evidence alone cannot establish forward profitability; an unchanged prospective shadow period is still required.",
            "",
            "Gate reasons: {}".format("; ".join(gate.get("reasons") or [])),
            "",
            "## Vehicle results",
            "",
            "| Vehicle | N | Win | EV / unit | Return/risk | PF | Unit P/L |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, metrics in sorted(((report.get("metrics") or {}).get("by_vehicle") or {}).items()):
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                name,
                metrics.get("n"),
                _fmt_pct(metrics.get("win_rate")),
                _fmt_money(metrics.get("expectancy_dollars")),
                _fmt_pct(metrics.get("average_return_on_risk")),
                _fmt_number(metrics.get("profit_factor")),
                _fmt_money(metrics.get("total_pnl_dollars")),
            )
        )
    monthly = (report.get("metrics") or {}).get("test_monthly_risk_sized") or {}
    lines.extend(
        [
            "",
            "## Test-period monthly distribution",
            "",
            "Signal-cohort months / zero-trade months: {} / {}  ".format(monthly.get("months"), monthly.get("zero_trade_months")),
            "Average / median / worst / best: {} / {} / {} / {}  ".format(
                _fmt_money(monthly.get("average_monthly_pnl_dollars")),
                _fmt_money(monthly.get("median_monthly_pnl_dollars")),
                _fmt_money(monthly.get("worst_month_pnl_dollars")),
                _fmt_money(monthly.get("best_month_pnl_dollars")),
            ),
            "Positive-month rate: {}  ".format(_fmt_pct(monthly.get("positive_month_rate"))),
            "",
            "## Evidence boundaries",
            "",
            "- Current/dynamic-universe history can carry survivorship bias; use a dated constituent source when available.",
            "- Daily bars cannot reveal intraday order, so a same-day stop and target is charged as stop-first.",
            "- Historical option entry is an EOD exact-chain approximation after an underlying zone touch, not proof of an intraday fill; option exit monitoring begins the following session.",
            "- Entry and exit limits use disclosed natural-to-midpoint improvement and include per-contract commissions.",
            "- Missing exact option legs, entry-zone misses, incomplete future paths, and budget/data failures remain visible in `missed.json`.",
            "- A second signal in a ticker with an unresolved replay position is not counted as a new trade; CORAT has no unplanned averaging rule.",
            "- Strategy, setup, vehicle, split, POP calibration, unit economics, and independently risk-sized P/L are separate outputs.",
            "- Sized P/L applies the frozen per-trade risk percentage independently; it is not a cash-constrained account simulation unless the user freezes an open-position cap.",
            "",
        ]
    )
    return "\n".join(lines)


def _csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    import io

    fields = [
        "ticker", "sector", "setup", "direction", "vehicle", "strategy", "signal_date", "entry_date",
        "exit_date", "exit_reason", "sessions_held", "underlying_entry", "underlying_exit", "stop", "target_1",
        "option_expiration", "option_legs", "entry_cashflow", "exit_cashflow", "maximum_risk_dollars",
        "unit_pnl_dollars", "return_on_risk", "quantity", "sized_pnl_dollars", "portfolio_included",
        "predicted_pop", "predicted_expected_profit", "decision_score", "split", "policy_hash",
    ]
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return handle.getvalue()


def _write_checkpoint(
    path: Path,
    status: str,
    completed_dates: Sequence[str],
    trades: Sequence[Mapping[str, Any]],
    missed: Sequence[Mapping[str, Any]],
    client: OratsClient,
    error: str = "",
) -> None:
    write_json(
        path,
        {
            "schema_version": "corat.full_replay_checkpoint.v1",
            "status": status,
            "updated_at_utc": utc_now(),
            "completed_decision_dates": list(completed_dates),
            "completed_trades": len(trades),
            "missed": len(missed),
            "orats_usage": client.usage(),
            "error": error,
            "resume_note": "Rerun with the same frozen policy. Cache-first reads avoid repaying successful ORATS requests.",
        },
    )


def run_full_replay(
    config: Mapping[str, Any],
    token: str,
    plan: Mapping[str, Any],
    execute: bool,
    offline: bool = False,
    request_budget: Optional[int] = None,
    monthly_reserve: Optional[int] = None,
    confirmed_remaining: Optional[int] = None,
    initial_nav: Optional[float] = None,
    risk_pct: Optional[float] = None,
    max_open_positions: Optional[int] = None,
    minimum_test_trades: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a frozen replay after explicit quota authorization.

    Callers should normally return :func:`build_replay_plan` directly when
    ``execute`` is false. This function repeats the guard so library callers
    cannot accidentally bypass it.
    """

    policy = plan.get("policy") or {}
    if str(policy.get("strategy_config_sha256") or "") != _strategy_config_hash(config):
        raise ValueError("current strategy config differs from the frozen replay plan")
    if str(policy.get("universe_sha256") or "") != _universe_hash(config):
        raise ValueError("current universe file differs from the frozen replay plan")
    usage_before = local_orats_usage(config)
    authorization = authorize_replay(
        plan,
        usage_before,
        execute,
        offline,
        request_budget,
        monthly_reserve,
        confirmed_remaining,
    )
    if not authorization.get("authorized"):
        return dict(plan)
    start = str(policy["start"])
    end = str(policy["end"])
    train_end = str(policy["train_end"])
    validation_end = str(policy["validation_end"])
    spacing = int(policy.get("spacing_sessions") or 1)
    ticker_values = [str(value).upper() for value in policy.get("tickers") or []]
    tickers = ticker_values or None
    max_trades_per_date = int(policy.get("max_trades_per_date") or 0)
    frozen_nav = float(policy.get("initial_nav") or 0.0)
    frozen_risk_pct = float(policy.get("risk_pct") or 0.0)
    frozen_max_open = int(policy.get("max_open_positions") or 0)
    frozen_minimum_test = int(policy.get("minimum_test_trades") or 0)
    if initial_nav is not None and abs(float(initial_nav) - frozen_nav) > 1e-9:
        raise ValueError("initial NAV differs from the frozen replay plan")
    if risk_pct is not None and abs(float(risk_pct) - frozen_risk_pct) > 1e-12:
        raise ValueError("risk percent differs from the frozen replay plan")
    if max_open_positions is not None and int(max_open_positions) != frozen_max_open:
        raise ValueError("maximum open positions differs from the frozen replay plan")
    if minimum_test_trades is not None and int(minimum_test_trades) != frozen_minimum_test:
        raise ValueError("minimum test trades differs from the frozen replay plan")
    initial_nav = frozen_nav
    max_open_positions = frozen_max_open
    minimum_test_trades = frozen_minimum_test
    if initial_nav <= 0 or max_open_positions < 0 or minimum_test_trades <= 0:
        raise ValueError("frozen NAV/evidence policy is invalid")
    orats_cfg = config.get("orats") or {}
    history_cfg = config.get("history") or {}
    holding_sessions = int(history_cfg.get("primary_horizon_sessions") or 10)
    replay_risk_pct = frozen_risk_pct
    if not 0 < replay_risk_pct <= 1:
        raise ValueError("risk percent must be greater than zero and at most one")
    commission = float((config.get("execution") or {}).get("commission_per_contract") or 0.65)
    lookback_start = (_day(start, "start") - timedelta(days=int(config.get("lookback_calendar_days") or 1900))).isoformat()
    outcome_end = (_day(end, "end") + timedelta(days=max(14, int(math.ceil(holding_sessions * 2.2))))).isoformat()
    outcome_end = min(outcome_end, today_new_york())

    run_digest = sha256_bytes(
        canonical_json({"policy": policy, "policy_hash": plan.get("policy_sha256"), "started": utc_now()}).encode("utf-8")
    )[:12]
    run_id = "{}-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), run_digest)
    run_dir = _resolve(config, "output_root") / "full_replays" / "{}_{}".format(start, end) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "checkpoint.json"

    client = OratsClient(
        token=token or "OFFLINE_CACHE_ONLY",
        base_url=str(orats_cfg["base_url"]),
        cache_root=_resolve(config, "cache_root"),
        state_root=_resolve(config, "state_root"),
        timeout_seconds=float(orats_cfg.get("request_timeout_seconds") or 120),
        max_requests=int(authorization.get("network_budget") or 0),
        monthly_cap=int(orats_cfg.get("monthly_request_cap") or 20000),
        requests_per_minute=int(orats_cfg.get("requests_per_minute") or 90),
        offline=offline,
        refresh=False,
        monthly_reserve=int(authorization.get("monthly_reserve") or 0),
    )
    all_traces: List[Any] = []
    source_errors: List[str] = []
    trades: List[Dict[str, Any]] = []
    missed: List[Dict[str, Any]] = []
    completed_dates: List[str] = []
    target_opportunities = 0

    try:
        spy_bundle = client.fetch_dailies(["SPY"], lookback_start, outcome_end, batch_size=1)
        all_traces.extend(spy_bundle.traces)
        source_errors.extend(spy_bundle.errors)
        spy_bars = bars_from_dailies(spy_bundle.rows).get("SPY", [])
        sessions = [bar.date for bar in spy_bars if start <= bar.date <= end]
        decision_dates = sessions[::spacing]
        if not decision_dates:
            raise RuntimeError("no SPY sessions are available for the replay window")

        market_by_date: Dict[str, Dict[str, FetchBundle]] = {}
        configured = load_universe(config)
        union_items: Dict[str, UniverseItem] = {}
        dynamic = bool(policy.get("dynamic_historical_universe"))
        for decision_date in decision_dates:
            market_by_date[decision_date] = {}
            for family in ("cores", "ivrank", "summaries"):
                bundle = client.fetch_market_asof(family, decision_date)
                market_by_date[decision_date][family] = bundle
                all_traces.extend(bundle.traces)
                source_errors.extend(bundle.errors)
            cores = market_by_date[decision_date]["cores"]
            if not cores.rows:
                raise RuntimeError("historical ORATS core snapshot unavailable for {}".format(decision_date))
            if dynamic:
                selected, _ = discover_universe(config, cores.rows, configured)
                union_items.update({item.ticker: item for item in selected})
        if not dynamic:
            selected = load_universe(config, tickers=tickers)
            union_items.update({item.ticker: item for item in selected})
        if not union_items:
            raise RuntimeError("replay universe is empty")
        history_names = supporting_tickers(config, list(union_items.values()))
        remaining_names = [name for name in history_names if name != "SPY"]
        remaining_bundle = client.fetch_dailies(
            remaining_names,
            lookback_start,
            outcome_end,
            batch_size=int(orats_cfg.get("batch_size") or 10),
        ) if remaining_names else FetchBundle()
        all_traces.extend(remaining_bundle.traces)
        source_errors.extend(remaining_bundle.errors)
        daily_bundle = FetchBundle(
            rows=list(spy_bundle.rows) + list(remaining_bundle.rows),
            traces=list(spy_bundle.traces) + list(remaining_bundle.traces),
            errors=list(spy_bundle.errors) + list(remaining_bundle.errors),
        )
        bars_by_ticker = bars_from_dailies(daily_bundle.rows)
        replay_client = ReplayDataClient(
            client,
            daily_bundle,
            market_by_date,
            lookback_start,
            outcome_end,
        )

        for decision_date in decision_dates:
            realized_before = sum(
                float(row.get("sized_pnl_dollars") or 0.0)
                for row in trades
                if row.get("portfolio_included") and str(row.get("exit_date") or "") <= decision_date
            )
            nav = initial_nav + realized_before
            scan = run_scan(
                config,
                token or "OFFLINE_CACHE_ONLY",
                decision_date,
                tickers=tickers,
                context_path=None,
                offline=offline,
                refresh=False,
                max_requests=int(authorization.get("network_budget") or 0),
                portfolio_nav=nav,
                posture="FROZEN_HISTORICAL_REPLAY_RESEARCH_ONLY",
                use_schwab=False,
                client=replay_client,
                write_artifacts=False,
                replay_mode=True,
                return_all_candidates=True,
            )
            all_traces.extend(scan.get("source_traces") or [])
            source_errors.extend(scan.get("source_errors") or [])
            targets = [row for row in scan.get("candidates") or [] if row.get("status") == TARGET_TRADE]
            target_opportunities += len(targets)
            if max_trades_per_date > 0 and len(targets) > max_trades_per_date:
                for row in targets[max_trades_per_date:]:
                    missed.append(
                        {
                            "ticker": row.get("ticker"),
                            "signal_date": decision_date,
                            "vehicle": row.get("vehicle"),
                            "reason": "user-selected max trades per date",
                        }
                    )
                targets = targets[:max_trades_per_date]
            for candidate in targets:
                ticker = str(candidate.get("ticker") or "").upper()
                if any(
                    str(prior.get("ticker") or "").upper() == ticker
                    and str(prior.get("entry_date") or "") <= decision_date < str(prior.get("exit_date") or "")
                    for prior in trades
                ):
                    missed.append(
                        {
                            "ticker": ticker,
                            "signal_date": decision_date,
                            "vehicle": candidate.get("vehicle"),
                            "reason": "same ticker already has an unresolved replay position; no unplanned averaging",
                        }
                    )
                    continue
                future = [bar for bar in bars_by_ticker.get(ticker, []) if bar.date > decision_date]
                trade, miss, traces, errors = _resolve_candidate(
                    candidate,
                    decision_date,
                    future,
                    replay_client,
                    nav,
                    replay_risk_pct,
                    commission,
                    train_end,
                    validation_end,
                )
                all_traces.extend(traces)
                source_errors.extend(errors)
                if miss is not None:
                    missed.append(miss)
                    continue
                assert trade is not None
                trade["policy_hash"] = plan.get("policy_sha256")
                if max_open_positions > 0 and trade.get("portfolio_included"):
                    open_count = sum(
                        1 for prior in trades
                        if prior.get("portfolio_included")
                        and str(prior.get("entry_date") or "") <= decision_date < str(prior.get("exit_date") or "")
                    )
                    if open_count >= max_open_positions:
                        trade["portfolio_included"] = False
                        trade["quantity"] = 0
                        trade["sized_pnl_dollars"] = 0.0
                        trade["portfolio_exclusion_reason"] = "user-selected maximum open positions"
                trades.append(trade)
            completed_dates.append(decision_date)
            _write_checkpoint(checkpoint_path, "IN_PROGRESS", completed_dates, trades, missed, client)
    except Exception as exc:
        _write_checkpoint(checkpoint_path, "FAILED_PARTIAL_CACHE_PRESERVED", completed_dates, trades, missed, client, str(exc))
        raise

    eligible = [row for row in trades if str(row.get("split") or "").startswith(("TRAIN", "VALIDATION", "TEST")) and not str(row.get("split") or "").startswith("EMBARGO")]
    by_split = _group_metrics(trades, "split")
    by_vehicle = _group_metrics(eligible, "vehicle")
    by_setup = _group_metrics(eligible, "setup")
    by_strategy = _group_metrics(eligible, "strategy")
    portfolio_rows = [row for row in eligible if row.get("portfolio_included")]
    risk_sized_metrics = replay_metrics(portfolio_rows, "sized_pnl_dollars")
    test_risk_sized_rows = [
        row for row in trades
        if row.get("portfolio_included") and row.get("split") == "TEST"
    ]
    test_start = (_day(validation_end, "validation end") + timedelta(days=1)).isoformat()
    test_monthly = monthly_pnl_summary(test_risk_sized_rows, test_start, end)
    test_metrics = by_split.get("TEST") or replay_metrics([])
    gate_reasons: List[str] = []
    if int(test_metrics.get("n") or 0) < minimum_test_trades:
        gate_reasons.append("test sample {} is below {}".format(test_metrics.get("n"), minimum_test_trades))
    if _number(test_metrics.get("expectancy_lower_95_dollars")) is None or float(test_metrics["expectancy_lower_95_dollars"]) <= 0:
        gate_reasons.append("test expectancy lower 95% bound is not positive")
    if _number(test_metrics.get("profit_factor")) is None or float(test_metrics["profit_factor"]) <= 1.0:
        gate_reasons.append("test profit factor is not above 1.0")
    if source_errors:
        gate_reasons.append("source errors are present")
    historical_gate = {
        "passed": not gate_reasons,
        "minimum_test_trades": minimum_test_trades,
        "reasons": gate_reasons or ["frozen historical test thresholds passed; prospective evidence is still required"],
    }
    traces = _dedupe_trace_dicts(all_traces)
    completion_status = (
        "COMPLETED_WITH_SOURCE_ERRORS_INCOMPLETE_NO_PROMOTION"
        if source_errors
        else "COMPLETED_HISTORICAL_RESEARCH_NO_PRODUCTION_PROMOTION"
    )
    report: Dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA,
        "status": completion_status,
        "generated_at_utc": utc_now(),
        "run_id": run_id,
        "policy": policy,
        "policy_sha256": plan.get("policy_sha256"),
        "start": start,
        "end": end,
        "train_end": train_end,
        "validation_end": validation_end,
        "decision_dates": len(completed_dates),
        "target_opportunities": target_opportunities,
        "completed": len(trades),
        "missed": len(missed),
        "metrics": {
            "by_split": by_split,
            "by_vehicle": by_vehicle,
            "by_setup": by_setup,
            "by_strategy": by_strategy,
            "risk_sized": risk_sized_metrics,
            "test_monthly_risk_sized": test_monthly,
        },
        "historical_evidence_gate": historical_gate,
        "test_status": "CONSUMED_BY_THIS_FROZEN_REPLAY",
        "production_promotion": False,
        "promotion_boundary": "Requires an unchanged prospective shadow period after the frozen historical test.",
        "sizing_note": "Risk-sized P/L applies the frozen per-trade risk percentage independently and is not a cash-constrained account simulation unless a maximum-open-position cap was frozen.",
        "authorization": authorization,
        "request_plan": plan.get("estimates"),
        "orats_usage_before": usage_before,
        "orats_usage": client.usage(),
        "source_errors": sorted(set(source_errors)),
        "source_traces": traces,
        "trades": trades,
        "missed_rows": missed,
        "survivorship_bias_warning": (
            "Dynamic replay uses each date's ORATS core universe but lacks a separate dated index-membership/security-master history. "
            "Delisted names absent from ORATS responses may still create survivorship bias."
        ),
        "order_submission_surface": False,
    }
    json_path = run_dir / "replay.json"
    markdown_path = run_dir / "replay.md"
    trades_path = run_dir / "trades.csv"
    missed_path = run_dir / "missed.json"
    diagnostics_path = run_dir / "diagnostics.json"
    sources_path = run_dir / "sources.json"
    plan_path = run_dir / "request_plan.json"
    manifest_path = run_dir / "manifest.json"
    report["artifacts"] = {
        "run_dir": str(run_dir),
        "report": str(markdown_path),
        "replay": str(json_path),
        "trades": str(trades_path),
        "missed": str(missed_path),
        "diagnostics": str(diagnostics_path),
        "sources": str(sources_path),
        "request_plan": str(plan_path),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
    }
    write_json(json_path, report)
    write_text(markdown_path, render_full_replay(report))
    write_text(trades_path, _csv_text(trades))
    write_json(missed_path, {"missed": missed})
    write_json(
        diagnostics_path,
        {
            "decision_dates": len(completed_dates),
            "target_opportunities": target_opportunities,
            "completed": len(trades),
            "missed": len(missed),
            "completed_by_vehicle": {name: metrics.get("n") for name, metrics in by_vehicle.items()},
            "missed_by_reason": {
                reason: sum(1 for row in missed if str(row.get("reason") or "") == reason)
                for reason in sorted({str(row.get("reason") or "") for row in missed})
            },
        },
    )
    write_json(sources_path, {"traces": traces, "errors": report["source_errors"]})
    write_json(plan_path, dict(plan))
    _write_checkpoint(checkpoint_path, "COMPLETE", completed_dates, trades, missed, client)
    artifact_paths = [json_path, markdown_path, trades_path, missed_path, diagnostics_path, sources_path, plan_path, checkpoint_path]
    write_json(
        manifest_path,
        {
            "schema_version": "corat.full_replay_manifest.v1",
            "run_id": run_id,
            "generated_at_utc": report["generated_at_utc"],
            "policy_sha256": plan.get("policy_sha256"),
            "config_path": config.get("_config_path"),
            "config_sha256": sha256_file(Path(str(config["_config_path"]))) if config.get("_config_path") else "",
            "outputs": {str(path): sha256_file(path) for path in artifact_paths},
            "orats_requests": client.usage().get("run_requests"),
            "monthly_reserve": authorization.get("monthly_reserve"),
            "secrets_persisted": False,
            "order_submission_surface": False,
            "production_promotion": False,
        },
    )
    return report
