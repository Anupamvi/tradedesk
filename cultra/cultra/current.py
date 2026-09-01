"""Present-day, read-only Cultra research orders backed by saved evidence.

This module never submits an order. It combines current Schwab market data
with Cultra-owned ORATS analytics and development-only historical outcomes.
Research orders remain visibly unqualified when the required POP calibration
gate has failed.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from .backfill import load_recent_sessions
from .calibration import wilson_interval
from .domain import parse_occ_symbol
from .research import (
    DEFAULT_CHAIN_DB,
    PROJECT_ROOT,
    _global_split_dates,
    _private_json,
    _private_write,
)
from .schwab import (
    OptionQuote,
    PriceBar,
    Quote,
    SchwabHTTPProvider,
    SchwabMarketDataBoundary,
)
from .statistics import clustered_bootstrap_mean_ci


CURRENT_CONFIG = PROJECT_ROOT / "configs" / "current_research.v1.json"
HISTORICAL_EVIDENCE_RUN = (
    PROJECT_ROOT / "out" / "cultra-historical-validation-v1-1-calendar-fix"
)
NEW_YORK = ZoneInfo("America/New_York")


class CurrentResearchError(RuntimeError):
    """Current research cannot produce a complete, auditable result."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CurrentResearchError("required Cultra artifact is unavailable: %s" % path.name) from exc


def _market_date(timestamp: datetime) -> date:
    return timestamp.astimezone(NEW_YORK).date()


def _next_business_day(value: date) -> date:
    result = value + timedelta(days=1)
    while result.weekday() >= 5:
        result += timedelta(days=1)
    return result


def _relative_spread(quote: OptionQuote) -> float:
    midpoint = (quote.bid + quote.ask) / 2.0
    return math.inf if midpoint <= 0.0 else (quote.ask - quote.bid) / midpoint


def _iv_to_realized_ratio(implied_volatility: float, realized_volatility: float) -> float:
    """Return the ratio named by the long-option value policy."""

    implied = float(implied_volatility)
    realized = float(realized_volatility)
    if not math.isfinite(implied) or not math.isfinite(realized):
        raise CurrentResearchError("volatility ratio inputs must be finite")
    if implied <= 0.0 or realized <= 0.0:
        raise CurrentResearchError("volatility ratio inputs must be positive")
    return implied / realized


def _signal_metrics(bars: Sequence[PriceBar], through: date) -> Mapping[str, Any]:
    eligible = tuple(item for item in bars if _market_date(item.timestamp) <= through)
    if len(eligible) < 21:
        raise CurrentResearchError("Schwab history has fewer than 21 completed sessions")
    selected = eligible[-21:]
    closes = tuple(float(item.close) for item in selected)
    momentum = closes[-1] / closes[0] - 1.0
    returns = tuple(math.log(right / left) for left, right in zip(closes, closes[1:]))
    realized = statistics.stdev(returns) * math.sqrt(252.0)
    return {
        "first_session": _market_date(selected[0].timestamp).isoformat(),
        "last_session": _market_date(selected[-1].timestamp).isoformat(),
        "session_count": len(selected),
        "first_close": closes[0],
        "last_close": closes[-1],
        "momentum_20": momentum,
        "realized_volatility_20": realized,
    }


def _orats_analytics(
    connection: sqlite3.Connection,
    ticker: str,
    trade_date: str,
    preferred_dte: int,
) -> Mapping[str, Any]:
    row = connection.execute(
        """
        SELECT trade_date,ticker,expiry,strike,dte,smv_vol,call_mid_iv,put_mid_iv,
               delta,gamma,theta,vega,rho,updated_at,snapshot_id
        FROM chains
        WHERE ticker=? AND trade_date=? AND dte BETWEEN 35 AND 50
          AND smv_vol IS NOT NULL AND delta IS NOT NULL
        ORDER BY abs(delta-0.55),abs(dte-?),strike
        LIMIT 1
        """,
        (ticker, trade_date, preferred_dte),
    ).fetchone()
    if row is None:
        raise CurrentResearchError("ORATS analytical row is unavailable for %s" % ticker)
    keys = (
        "provider_trade_date",
        "ticker",
        "expiry",
        "strike",
        "dte",
        "smv_vol",
        "call_mid_iv",
        "put_mid_iv",
        "delta",
        "gamma",
        "theta",
        "vega",
        "rho",
        "updated_at",
        "snapshot_id",
    )
    return dict(zip(keys, row))


def _viable_contracts(
    contracts: Sequence[OptionQuote],
    *,
    option_type: str,
    market_date: date,
    config: Mapping[str, Any],
) -> Mapping[date, Tuple[OptionQuote, ...]]:
    policy = config["contract_policy"]
    minimum = int(policy["minimum_entry_dte"])
    maximum = int(policy["maximum_entry_dte"])
    by_expiry: Dict[date, List[OptionQuote]] = {}
    for item in contracts:
        dte = (item.expiration - market_date).days
        if item.option_type != option_type or dte < minimum or dte > maximum:
            continue
        if item.bid < float(policy["minimum_bid"]):
            continue
        if item.open_interest is None or item.open_interest < int(
            policy["minimum_open_interest"]
        ):
            continue
        if item.delta is None or _relative_spread(item) > float(
            policy["maximum_relative_spread"]
        ):
            continue
        by_expiry.setdefault(item.expiration, []).append(item)
    return {
        expiry: tuple(sorted(values, key=lambda item: item.occ_symbol))
        for expiry, values in by_expiry.items()
    }


def _select_legs(
    family: str,
    contracts: Sequence[OptionQuote],
    *,
    market_date: date,
    config: Mapping[str, Any],
) -> Tuple[Tuple[str, OptionQuote], ...]:
    policy = config["contract_policy"]
    option_type = "CALL" if family in {"LONG_CALL", "CALL_DEBIT_VERTICAL"} else "PUT"
    by_expiry = _viable_contracts(
        contracts,
        option_type=option_type,
        market_date=market_date,
        config=config,
    )
    preferred = int(policy["preferred_entry_dte"])
    for expiry in sorted(
        by_expiry,
        key=lambda item: (abs((item - market_date).days - preferred), item),
    ):
        values = by_expiry[expiry]
        if family == "LONG_CALL":
            target = float(policy["long_call_delta"])
            selected = min(values, key=lambda item: (abs(float(item.delta) - target), item.occ_symbol))
            return (("BUY", selected),)
        if family == "LONG_PUT":
            target = float(policy["long_put_absolute_delta"])
            selected = min(values, key=lambda item: (abs(abs(float(item.delta)) - target), item.occ_symbol))
            return (("BUY", selected),)
        if family == "CALL_DEBIT_VERTICAL":
            long_target = float(policy["call_vertical_long_delta"])
            short_target = float(policy["call_vertical_short_delta"])
            pairs = (
                (long_leg, short_leg)
                for long_leg in values
                for short_leg in values
                if short_leg.strike > long_leg.strike and long_leg.ask > short_leg.bid
            )
            ranked = sorted(
                pairs,
                key=lambda pair: (
                    abs(float(pair[0].delta) - long_target)
                    + abs(float(pair[1].delta) - short_target),
                    _relative_spread(pair[0]) + _relative_spread(pair[1]),
                    pair[0].occ_symbol,
                    pair[1].occ_symbol,
                ),
            )
            if ranked:
                return (("BUY", ranked[0][0]), ("SELL", ranked[0][1]))
        if family == "PUT_DEBIT_VERTICAL":
            long_target = float(policy["put_vertical_long_absolute_delta"])
            short_target = float(policy["put_vertical_short_absolute_delta"])
            pairs = (
                (long_leg, short_leg)
                for long_leg in values
                for short_leg in values
                if short_leg.strike < long_leg.strike and long_leg.ask > short_leg.bid
            )
            ranked = sorted(
                pairs,
                key=lambda pair: (
                    abs(abs(float(pair[0].delta)) - long_target)
                    + abs(abs(float(pair[1].delta)) - short_target),
                    _relative_spread(pair[0]) + _relative_spread(pair[1]),
                    pair[0].occ_symbol,
                    pair[1].occ_symbol,
                ),
            )
            if ranked:
                return (("BUY", ranked[0][0]), ("SELL", ranked[0][1]))
    raise CurrentResearchError("no exact Schwab structure satisfies the frozen contract policy")


def _historical_rows() -> Tuple[Mapping[str, Any], ...]:
    path = HISTORICAL_EVIDENCE_RUN / "resolved_trades.jsonl"
    try:
        return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CurrentResearchError("resolved historical evidence is unavailable") from exc


def _frequency(rows: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    if key == "net_profit":
        successes = sum(float(item["net_pnl"]) > 0.0 for item in rows)
    else:
        successes = sum(bool(item[key]) for item in rows)
    lower, upper = wilson_interval(successes, len(rows), 0.95)
    return {
        "historical_frequency": successes / len(rows),
        "wilson_95_lower": lower,
        "wilson_95_upper": upper,
        "successes": successes,
        "sample_size": len(rows),
        "status": "EMPIRICAL_FREQUENCY_NOT_CALIBRATED_POP",
    }


def _comparable_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    family: str,
    ticker: str,
    development_dates: Sequence[str],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    date_set = set(development_dates)
    selected = tuple(
        item
        for item in rows
        if item["strategy_family"] == family
        and item["ticker"] == ticker
        and item["entry_date"] in date_set
    )
    if not selected:
        return {"sample_size": 0, "status": "NO_COMPARABLE_HISTORY"}
    returns = tuple(float(item["net_pnl"]) / float(item["maximum_loss"]) for item in selected)
    clusters = tuple(str(item["entry_date"]) for item in selected)
    policy = config["comparable_evidence"]
    interval = clustered_bootstrap_mean_ci(
        returns,
        clusters,
        confidence=float(policy["bootstrap_confidence"]),
        iterations=int(policy["bootstrap_iterations"]),
        seed=int(policy["bootstrap_seed"]),
    )
    worst_count = max(1, int(math.ceil(len(returns) * 0.10)))
    worst = tuple(sorted(returns)[:worst_count])
    return {
        "status": "DEVELOPMENT_ONLY_EXACT_FAMILY_TICKER",
        "sample_size": len(selected),
        "cluster_count": len(set(clusters)),
        "period_start": min(str(item["entry_date"]) for item in selected),
        "period_end": max(str(item["entry_date"]) for item in selected),
        "mean_net_return_on_maximum_loss": statistics.mean(returns),
        "cluster_bootstrap_95_lower_return": interval.lower,
        "cluster_bootstrap_95_upper_return": interval.upper,
        "expected_shortfall_10pct_return": statistics.mean(worst),
        "net_profit_frequency": _frequency(selected, "net_profit"),
        "target_frequency": _frequency(selected, "target_hit"),
        "stop_frequency": _frequency(selected, "stop_hit"),
        "max_loss_frequency": _frequency(selected, "max_loss_hit"),
    }


def _economics(
    family: str,
    legs: Sequence[Tuple[str, OptionQuote]],
    evidence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Mapping[str, Any]:
    multiplier = int(config["cost_policy"]["contract_multiplier"])
    natural_per_share = math.fsum(
        item.ask if action == "BUY" else -item.bid for action, item in legs
    )
    if natural_per_share <= 0.0:
        raise CurrentResearchError("selected Schwab structure has no positive debit")
    slippage_one_side = math.fsum(
        max(
            float(config["cost_policy"]["minimum_slippage_per_share_per_leg_per_side"]),
            (item.ask - item.bid)
            * float(config["cost_policy"]["additional_slippage_fraction_of_spread"]),
        )
        * multiplier
        for _action, item in legs
    )
    commissions = len(legs) * 2 * (
        float(config["cost_policy"]["commission_per_contract_per_side"])
        + float(config["cost_policy"]["fee_per_contract_per_side"])
    )
    natural_dollars = natural_per_share * multiplier
    maximum_loss = natural_dollars + 2.0 * slippage_one_side + commissions
    maximum_profit: Optional[float] = None
    if family in {"CALL_DEBIT_VERTICAL", "PUT_DEBIT_VERTICAL"}:
        width = abs(legs[0][1].strike - legs[1][1].strike) * multiplier
        maximum_profit = width - natural_dollars - 2.0 * slippage_one_side - commissions
        if maximum_profit <= 0.0:
            raise CurrentResearchError("selected Schwab vertical has no positive maximum profit")
    point_return = float(evidence["mean_net_return_on_maximum_loss"])
    conservative_return = float(evidence["cluster_bootstrap_95_lower_return"])
    point_ev = point_return * maximum_loss
    conservative_ev = conservative_return * maximum_loss
    if family == "CALL_DEBIT_VERTICAL":
        breakeven = legs[0][1].strike + maximum_loss / multiplier
    elif family == "PUT_DEBIT_VERTICAL":
        breakeven = legs[0][1].strike - maximum_loss / multiplier
    elif family == "LONG_CALL":
        breakeven = legs[0][1].strike + maximum_loss / multiplier
    else:
        breakeven = legs[0][1].strike - maximum_loss / multiplier
    return {
        "natural_debit_per_share": natural_per_share,
        "proposed_limit_debit_per_share": round(natural_per_share, 2),
        "modeled_entry_and_exit_slippage": 2.0 * slippage_one_side,
        "commissions_and_fees": commissions,
        "maximum_loss": maximum_loss,
        "maximum_profit": maximum_profit,
        "breakevens_at_expiration": [breakeven],
        "target_pnl": maximum_loss
        * float(config["exit_policy"]["profit_target_fraction_of_max_loss"]),
        "stop_pnl": -maximum_loss
        * float(config["exit_policy"]["stop_loss_fraction_of_max_loss"]),
        "net_expected_profit": point_ev,
        "expected_return_on_maximum_loss": point_return,
        "conservative_net_expected_profit": conservative_ev,
        "conservative_return_on_maximum_loss": conservative_return,
        "model_fair_debit_per_share": natural_per_share + point_ev / multiplier,
        "conservative_model_fair_debit_per_share": natural_per_share
        + conservative_ev / multiplier,
        "expected_shortfall_10pct": float(evidence["expected_shortfall_10pct_return"])
        * maximum_loss,
        "adverse_gap_stress_loss": -maximum_loss,
    }


def _leg_payload(action: str, item: OptionQuote) -> Mapping[str, Any]:
    return {
        "action": action,
        "ratio": 1,
        "occ_symbol": item.occ_symbol,
        "expiration": item.expiration.isoformat(),
        "strike": item.strike,
        "option_type": item.option_type,
        "bid": item.bid,
        "ask": item.ask,
        "relative_spread": _relative_spread(item),
        "delta_market_heuristic_not_pop": item.delta,
        "volume": item.volume,
        "open_interest": item.open_interest,
        "quote_timestamp": item.timestamp.isoformat(),
    }


def _artifact_manifest(run_dir: Path, paths: Iterable[Path]) -> Mapping[str, Any]:
    records = []
    for path in sorted(paths, key=lambda item: item.name):
        records.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema": "cultra.current-research-manifest.v1",
        "run_dir": str(run_dir),
        "files": records,
        "config_sha256": _sha256(CURRENT_CONFIG),
        "source_sha256": _sha256(Path(__file__)),
        "broker_submission_enabled": False,
    }


def run_current_research(
    *,
    as_of: date,
    run_id: str,
    output_root: Path = PROJECT_ROOT / "out",
    boundary: Optional[SchwabMarketDataBoundary] = None,
    database: Path = DEFAULT_CHAIN_DB,
) -> Mapping[str, Any]:
    """Produce every positive-conservative-EV current research order."""

    config = _load_json(CURRENT_CONFIG)
    historical = _load_json(HISTORICAL_EVIDENCE_RUN / "historical_validation.json")
    resolved = _historical_rows()
    split = _global_split_dates(load_recent_sessions())
    development_dates = tuple(split["training"]) + tuple(split["validation"])
    provider = boundary or SchwabMarketDataBoundary(SchwabHTTPProvider.production())
    universe = tuple(str(item) for item in config["universe"])
    quotes = provider.quotes(universe)
    missing_quotes = sorted(set(universe).difference(quotes))
    if missing_quotes:
        raise CurrentResearchError("Schwab omitted quotes for: %s" % ",".join(missing_quotes))

    histories: Dict[str, Tuple[PriceBar, ...]] = {}
    metrics: Dict[str, Mapping[str, Any]] = {}
    for ticker in universe:
        quote_date = _market_date(quotes[ticker].timestamp)
        bars = provider.price_history(
            ticker, start=quote_date - timedelta(days=50), end=as_of
        )
        histories[ticker] = tuple(bars)
        metrics[ticker] = _signal_metrics(bars, quote_date)

    config_contract = config["contract_policy"]
    bullish = float(config["signal_policy"]["bullish_momentum_threshold"])
    bearish = float(config["signal_policy"]["bearish_momentum_threshold"])
    family_states = historical["strategy_states"]
    family_results = historical["family_results"]
    research_orders = []
    watchlist = []
    no_signal = []
    normalized_chains = []
    connection = sqlite3.connect(Path(database).expanduser().resolve())
    try:
        latest_orats_date = connection.execute(
            "SELECT max(trade_date) FROM sessions"
        ).fetchone()[0]
        if not latest_orats_date:
            raise CurrentResearchError("Cultra historical database has no ORATS sessions")
        for ticker in universe:
            quote = quotes[ticker]
            market_date = _market_date(quote.timestamp)
            signal = metrics[ticker]
            momentum = float(signal["momentum_20"])
            if momentum > bullish:
                families = ("LONG_CALL", "CALL_DEBIT_VERTICAL")
                direction = "BULLISH"
            elif momentum < bearish:
                families = ("LONG_PUT", "PUT_DEBIT_VERTICAL")
                direction = "BEARISH"
            else:
                no_signal.append(
                    {
                        "ticker": ticker,
                        "momentum_20": momentum,
                        "reason": "ABS_20_SESSION_MOMENTUM_BELOW_3_PERCENT",
                    }
                )
                continue
            analytics = _orats_analytics(
                connection,
                ticker,
                str(latest_orats_date),
                int(config_contract["preferred_entry_dte"]),
            )
            from_date = market_date + timedelta(
                days=int(config_contract["minimum_entry_dte"])
            )
            to_date = market_date + timedelta(
                days=int(config_contract["maximum_entry_dte"])
            )
            chain = provider.option_chain(
                ticker, from_date=from_date, to_date=to_date
            )
            for family in families:
                item: Dict[str, Any] = {
                    "ticker": ticker,
                    "strategy_family": family,
                    "direction": direction,
                    "evidence_state": family_states.get(family, "UNPROVEN"),
                    "signal": signal,
                    "schwab_underlying_quote": {
                        "bid": quote.bid,
                        "ask": quote.ask,
                        "last": quote.last,
                        "timestamp": quote.timestamp.isoformat(),
                        "provider_market_date": market_date.isoformat(),
                    },
                    "orats_analytics": analytics,
                    "quantity": "USER DETERMINED",
                    "broker_submission_enabled": False,
                }
                if family in {"LONG_CALL", "LONG_PUT"}:
                    ratio = _iv_to_realized_ratio(
                        float(analytics["smv_vol"]),
                        float(signal["realized_volatility_20"]),
                    )
                    item["iv_to_realized_filter"] = {
                        "iv_to_realized_ratio": ratio,
                        "maximum_allowed": float(
                            config["signal_policy"]["long_option_max_iv_to_realized_ratio"]
                        ),
                    }
                    if ratio > float(
                        config["signal_policy"]["long_option_max_iv_to_realized_ratio"]
                    ):
                        item["reasons"] = ["LONG_OPTION_VOLATILITY_VALUE_FILTER_FAILED"]
                        watchlist.append(item)
                        continue
                try:
                    legs = _select_legs(
                        family,
                        chain.contracts,
                        market_date=market_date,
                        config=config,
                    )
                except CurrentResearchError as exc:
                    item["reasons"] = [str(exc)]
                    watchlist.append(item)
                    continue
                item["legs"] = [_leg_payload(action, leg) for action, leg in legs]
                normalized_chains.extend(item["legs"])
                evidence = _comparable_summary(
                    resolved,
                    family=family,
                    ticker=ticker,
                    development_dates=development_dates,
                    config=config,
                )
                item["comparable_evidence"] = evidence
                calibration = family_results.get(family, {})
                item["POP_net"] = {
                    "status": "UNAVAILABLE_CALIBRATION_GATE_FAILED",
                    "point_estimate": None,
                    "family_calibration_reasons": [
                        reason
                        for reason in calibration.get("reasons", [])
                        if "POP" in reason
                    ],
                    "empirical_frequency_diagnostic": evidence.get(
                        "net_profit_frequency"
                    ),
                    "delta_is_pop": False,
                }
                item["P_target"] = evidence.get("target_frequency")
                item["P_stop"] = evidence.get("stop_frequency")
                item["P_max_loss"] = evidence.get("max_loss_frequency")
                reasons = []
                minimum_sample = int(
                    config["comparable_evidence"]["minimum_resolved_observations"]
                )
                if int(evidence.get("sample_size", 0)) < minimum_sample:
                    reasons.append("FEWER_THAN_%d_COMPARABLE_TRADES" % minimum_sample)
                if evidence.get("mean_net_return_on_maximum_loss", -math.inf) <= 0.0:
                    reasons.append("POINT_NET_EV_NOT_POSITIVE")
                if evidence.get("cluster_bootstrap_95_lower_return", -math.inf) <= 0.0:
                    reasons.append("CONSERVATIVE_NET_EV_NOT_POSITIVE")
                if not reasons:
                    item["economics"] = _economics(family, legs, evidence, config)
                    latest_leg_time = max(leg.timestamp for _action, leg in legs)
                    refresh_required = (
                        as_of.weekday() >= 5
                        or datetime.now(timezone.utc) - latest_leg_time
                        > timedelta(minutes=15)
                    )
                    item["disposition"] = (
                        config["visibility"]["market_closed_label"]
                        if refresh_required
                        else config["visibility"]["eligible_label"]
                    )
                    item["entry_condition"] = (
                        "Reprice all legs from Schwab; enter only if signal remains valid, "
                        "both point and conservative EV remain positive, and the debit is "
                        "no greater than the refreshed proposed limit."
                    )
                    item["invalidation"] = (
                        "Momentum no longer exceeds the frozen threshold, exact-leg quotes "
                        "fail liquidity rules, or either EV estimate becomes non-positive."
                    )
                    item["time_exit_sessions"] = int(
                        config["exit_policy"]["time_exit_sessions"]
                    )
                    item["assignment_exercise_handling"] = (
                        "Close both legs before expiration; do not exercise or permit assignment."
                    )
                    item["next_review_date"] = _next_business_day(as_of).isoformat()
                    item["manual_ticket_enabled"] = False
                    research_orders.append(item)
                else:
                    item["reasons"] = reasons
                    item["disposition"] = "WATCHLIST"
                    watchlist.append(item)
    finally:
        connection.close()

    research_orders.sort(
        key=lambda item: (
            -float(item["economics"]["conservative_return_on_maximum_loss"]),
            item["ticker"],
            item["strategy_family"],
        )
    )
    watchlist.sort(key=lambda item: (item["ticker"], item["strategy_family"]))
    no_signal.sort(key=lambda item: item["ticker"])
    run_dir = Path(output_root).expanduser().resolve() / run_id
    try:
        run_dir.relative_to((PROJECT_ROOT / "out").resolve())
    except ValueError as exc:
        raise CurrentResearchError("current output must remain inside Cultra/out") from exc
    run_dir.mkdir(parents=True, exist_ok=False, mode=0o700)

    inputs = {
        "quotes": {
            ticker: {
                "bid": item.bid,
                "ask": item.ask,
                "last": item.last,
                "timestamp": item.timestamp.isoformat(),
            }
            for ticker, item in quotes.items()
        },
        "signal_metrics": metrics,
        "selected_exact_contract_quotes": normalized_chains,
        "source_roles": {
            "Schwab": "underlying quotes, price history, exact option quotes and liquidity",
            "ORATS": "delayed EOD analytical fields and historical exact-leg evidence",
        },
    }
    summary = {
        "schema": "cultra.current-research-result.v1",
        "run_id": run_id,
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_profit_confidence": "UNPROVEN",
        "historical_evidence_run": str(HISTORICAL_EVIDENCE_RUN),
        "research_orders": research_orders,
        "watchlist": watchlist,
        "no_signal": no_signal,
        "counts": {
            "research_orders": len(research_orders),
            "watchlist": len(watchlist),
            "no_signal": len(no_signal),
        },
        "manual_ticket_enabled": False,
        "broker_submission_enabled": False,
        "quantity": "USER DETERMINED",
        "important": (
            "Research orders have positive development-only point and conservative EV, "
            "but are not qualified tickets because calibrated POP failed validation."
        ),
    }
    inputs_path = _private_json(run_dir / "normalized_inputs.json", inputs)
    summary_path = _private_json(run_dir / "current_research.json", summary)
    lines = [
        "# Cultra Current Profit-Potential Research Orders",
        "",
        "- Overall confidence: **UNPROVEN**",
        "- Qualified manual tickets: **0**",
        "- Positive-conservative-EV research orders: **%d**" % len(research_orders),
        "- Broker submission: **disabled**",
        "- Quantity: **USER DETERMINED**",
        "",
        "These are visible now; the 90-day shadow period is not hiding them. POP is shown as unavailable because the frozen calibration model failed, so none is a qualified ticket.",
        "",
        "## Research orders",
        "",
    ]
    if research_orders:
        lines.extend(
            [
                "| Ticker | Structure | Exact legs | Limit debit | Max loss | Point EV | Conservative EV | Historical win frequency | POP status | Disposition |",
                "|---|---|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for item in research_orders:
            legs = "; ".join(
                "%s 1 %s" % (leg["action"], leg["occ_symbol"])
                for leg in item["legs"]
            )
            empirical = item["POP_net"]["empirical_frequency_diagnostic"]
            economics = item["economics"]
            lines.append(
                "| %s | %s | %s | $%.2f | $%.2f | $%.2f | $%.2f | %.1f%% (n=%d; not calibrated POP) | `%s` | `%s` |"
                % (
                    item["ticker"],
                    item["strategy_family"],
                    legs,
                    economics["proposed_limit_debit_per_share"],
                    economics["maximum_loss"],
                    economics["net_expected_profit"],
                    economics["conservative_net_expected_profit"],
                    100.0 * empirical["historical_frequency"],
                    empirical["sample_size"],
                    item["POP_net"]["status"],
                    item["disposition"],
                )
            )
    else:
        lines.append("None.")
    lines.extend(["", "## Watchlist", ""])
    if watchlist:
        for item in watchlist:
            lines.append(
                "- **%s %s** — %s"
                % (
                    item["ticker"],
                    item["strategy_family"],
                    "; ".join(item.get("reasons", ["incomplete evidence"])),
                )
            )
    else:
        lines.append("None.")
    lines.extend(["", "## No signal", ""])
    lines.append(
        ", ".join(item["ticker"] for item in no_signal) if no_signal else "None."
    )
    board_path = _private_write(
        run_dir / "current_research.md", ("\n".join(lines) + "\n").encode("utf-8")
    )
    manifest = _artifact_manifest(run_dir, (inputs_path, summary_path, board_path))
    _private_json(run_dir / "manifest.json", manifest)
    return summary


def verify_current_research(run_dir: Path) -> Tuple[str, ...]:
    """Reproduce saved research-order economics and artifact identities."""

    root = Path(run_dir).expanduser().resolve()
    try:
        root.relative_to((PROJECT_ROOT / "out").resolve())
    except ValueError:
        return ("run directory leaves Cultra/out",)
    errors = []
    try:
        manifest = _load_json(root / "manifest.json")
        summary = _load_json(root / "current_research.json")
        config = _load_json(CURRENT_CONFIG)
    except CurrentResearchError as exc:
        return (str(exc),)
    expected_files = {"current_research.json", "current_research.md", "normalized_inputs.json"}
    listed = {str(item.get("path")) for item in manifest.get("files", [])}
    if listed != expected_files:
        errors.append("manifest file set is incomplete or unexpected")
    for item in manifest.get("files", []):
        path = root / str(item.get("path", ""))
        try:
            if path.stat().st_size != int(item.get("bytes")):
                errors.append("byte count changed: %s" % path.name)
            if _sha256(path) != item.get("sha256"):
                errors.append("sha256 changed: %s" % path.name)
        except (OSError, TypeError, ValueError):
            errors.append("manifest artifact unavailable: %s" % path.name)
    if manifest.get("config_sha256") != _sha256(CURRENT_CONFIG):
        errors.append("current research config fingerprint changed")
    if manifest.get("source_sha256") != _sha256(Path(__file__)):
        errors.append("current research source fingerprint changed")
    if summary.get("manual_ticket_enabled") is not False:
        errors.append("current research improperly enables manual tickets")
    if summary.get("broker_submission_enabled") is not False:
        errors.append("current research improperly enables broker submission")
    cost = config["cost_policy"]
    multiplier = int(cost["contract_multiplier"])
    for index, order in enumerate(summary.get("research_orders", [])):
        label = "%s:%s" % (order.get("ticker"), order.get("strategy_family"))
        legs = order.get("legs", [])
        if not legs:
            errors.append("%s has no exact legs" % label)
            continue
        symbols = []
        try:
            for leg in legs:
                symbols.append(str(leg["occ_symbol"]))
                parse_occ_symbol(str(leg["occ_symbol"]))
            if len(symbols) != len(set(symbols)):
                errors.append("%s repeats an OCC leg" % label)
            natural = math.fsum(
                float(leg["ask"])
                if leg["action"] == "BUY"
                else -float(leg["bid"])
                for leg in legs
            )
            one_side_slippage = math.fsum(
                max(
                    float(cost["minimum_slippage_per_share_per_leg_per_side"]),
                    (float(leg["ask"]) - float(leg["bid"]))
                    * float(cost["additional_slippage_fraction_of_spread"]),
                )
                * multiplier
                for leg in legs
            )
            commissions = len(legs) * 2 * (
                float(cost["commission_per_contract_per_side"])
                + float(cost["fee_per_contract_per_side"])
            )
            expected_loss = natural * multiplier + 2.0 * one_side_slippage + commissions
            economics = order["economics"]
            if not math.isclose(
                float(economics["maximum_loss"]), expected_loss, rel_tol=1e-12, abs_tol=1e-9
            ):
                errors.append("%s maximum loss does not reproduce" % label)
            evidence = order["comparable_evidence"]
            expected_point = expected_loss * float(
                evidence["mean_net_return_on_maximum_loss"]
            )
            expected_conservative = expected_loss * float(
                evidence["cluster_bootstrap_95_lower_return"]
            )
            if not math.isclose(
                float(economics["net_expected_profit"]), expected_point, rel_tol=1e-12, abs_tol=1e-9
            ):
                errors.append("%s point EV does not reproduce" % label)
            if not math.isclose(
                float(economics["conservative_net_expected_profit"]),
                expected_conservative,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                errors.append("%s conservative EV does not reproduce" % label)
            if expected_point <= 0.0 or expected_conservative <= 0.0:
                errors.append("%s does not have two positive EV estimates" % label)
            if order.get("POP_net", {}).get("point_estimate") is not None:
                errors.append("%s improperly reports failed calibration as POP" % label)
            if order.get("manual_ticket_enabled") is not False:
                errors.append("%s improperly enables a manual ticket" % label)
            if order.get("quantity") != "USER DETERMINED":
                errors.append("%s contains pipeline sizing" % label)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append("%s is incomplete: %s" % (label or index, exc))
    if int(summary.get("counts", {}).get("research_orders", -1)) != len(
        summary.get("research_orders", [])
    ):
        errors.append("research-order count does not reconcile")
    return tuple(errors)


__all__ = [
    "CurrentResearchError",
    "run_current_research",
    "verify_current_research",
]
