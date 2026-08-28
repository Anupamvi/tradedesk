"""ORATS-first stock and options swing candidate synthesis.

The output can promote a fully checked row to ``MANUAL_READY``. It cannot and
does not submit, stage, replace, or cancel a broker order.
"""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from codexswing.backtest.labels import DailyBar, parse_orats_daily_rows
from codexswing.features.price import PriceObservation, parse_orats_price_history
from codexswing.features.volatility import IVRankObservation, parse_orats_ivrank_rows
from codexswing.models.baseline import BaselineDataError, PriceMoveBaseline, compute_price_move_baseline
from codexswing.options.expected_pnl import ForecastDistribution
from codexswing.research.contracts import select_current_verticals
from codexswing.research.readiness import evaluate_promotion, historical_gate
from codexswing.research.universe import UniverseCandidate
from codexswing.schemas.source import SourceRecord


IDEA_SCHEMA_VERSION = "codexswing.current_ideas.v4"
HORIZON_SESSIONS = 5
ANALOG_COUNT = 250
ROUND_TRIP_STOCK_COST_BPS = 10.0


class CurrentIdeaError(RuntimeError):
    pass


@dataclass(frozen=True)
class AnalogEvidence:
    ticker: str
    side: str
    neighbor_count: int
    effective_nonoverlapping_count: int
    eligible_history_count: int
    empirical_probability_of_profit: float
    wilson_95_lower_bound: float
    wilson_95_upper_bound: float
    mean_net_return: float
    median_net_return: float
    p10_net_return: float
    bootstrap_p05_mean_return: float
    return_stddev: float
    mean_mfe: float
    mean_mae: float
    analog_dates: Sequence[str]
    method: str = (
        "Two hundred fifty nearest prior same-direction ORATS adjusted EOD environments; "
        "features are 1d/5d/20d trend, realized volatility, and as-of IV percentile; "
        "entry is next adjusted open, exit is fifth close, less 10 bps"
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _ticker(payload: Mapping[str, Any]) -> str:
    reference = payload.get("reference")
    reference_values = reference if isinstance(reference, Mapping) else {}
    return str(
        payload.get("ticker")
        or payload.get("symbol")
        or reference_values.get("symbol")
        or ""
    ).strip().upper()


def _records_by_ticker(
    records: Iterable[SourceRecord], source: str
) -> Mapping[str, SourceRecord]:
    result: Dict[str, SourceRecord] = {}
    for record in records:
        if record.source != source:
            continue
        ticker = _ticker(record.payload)
        if not ticker:
            raise CurrentIdeaError("{} record is missing ticker".format(source))
        if ticker in result:
            raise CurrentIdeaError("duplicate {} record for {}".format(source, ticker))
        result[ticker] = record
    return result


def _wilson_interval(wins: int, total: int) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_lower(values: Sequence[float], seed_text: str) -> float:
    if not values:
        return 0.0
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    generator = random.Random(seed)
    estimates = []
    for _ in range(2_000):
        estimates.append(statistics.fmean(generator.choice(values) for _ in values))
    estimates.sort()
    return estimates[max(0, int(0.05 * len(estimates)) - 1)]


def _feature_vector(
    observations: Sequence[PriceObservation],
    index: int,
    closes: Optional[Sequence[float]] = None,
) -> Tuple[float, float, float, float, float]:
    close_values = closes if closes is not None else [item.close for item in observations]
    return_1d = close_values[index] / close_values[index - 1] - 1.0
    return_5d = close_values[index] / close_values[index - 5] - 1.0
    return_10d = close_values[index] / close_values[index - 10] - 1.0
    return_20d = close_values[index] / close_values[index - 20] - 1.0
    logs = [
        math.log(close_values[position] / close_values[position - 1])
        for position in range(index - 19, index + 1)
    ]
    realized = statistics.stdev(logs) * math.sqrt(252.0)
    trend = 0.20 * return_5d + 0.30 * return_10d + 0.50 * return_20d
    return trend, return_1d, return_5d, return_20d, realized


def _iv_percentile_by_date(
    observations: Sequence[IVRankObservation],
) -> Mapping[str, float]:
    result = {}
    for item in observations:
        value = item.iv_percentile_1y
        if value is None:
            value = item.iv_rank_1y
        if value is not None:
            result[item.trade_date] = float(value)
    return result


def _analog_evidence(
    *,
    ticker: str,
    side: str,
    prices: Sequence[PriceObservation],
    bars: Sequence[DailyBar],
    iv_history: Sequence[IVRankObservation],
    current_iv_percentile: float,
) -> AnalogEvidence:
    ordered = tuple(sorted(prices, key=lambda item: item.session_date))
    ordered_bars = tuple(sorted(bars, key=lambda item: item.trade_date))
    if len(ordered) < 90 or len(ordered_bars) < 90:
        raise CurrentIdeaError("{} has fewer than 90 adjusted daily observations".format(ticker))
    bar_by_date = {bar.trade_date: index for index, bar in enumerate(ordered_bars)}
    closes = tuple(item.close for item in ordered)
    current = _feature_vector(ordered, len(ordered) - 1, closes)
    current_direction = 1.0 if side == "LONG" else -1.0
    iv_by_date = _iv_percentile_by_date(iv_history)
    trend_scale = max(abs(current[0]), current[4] * math.sqrt(20.0 / 252.0), 0.02)
    daily_scale = max(current[4] / math.sqrt(252.0), 0.008)
    rows: List[Tuple[float, float, float, float, str, int]] = []
    for index in range(20, len(ordered) - HORIZON_SESSIONS - 1):
        decision = ordered[index]
        historical = _feature_vector(ordered, index, closes)
        if historical[0] * current_direction <= 0:
            continue
        bar_index = bar_by_date.get(decision.session_date)
        if bar_index is None or bar_index + HORIZON_SESSIONS >= len(ordered_bars):
            continue
        entry = ordered_bars[bar_index + 1]
        exit_bar = ordered_bars[bar_index + HORIZON_SESSIONS]
        window = ordered_bars[bar_index + 1 : bar_index + HORIZON_SESSIONS + 1]
        direction = current_direction
        net = direction * (exit_bar.close / entry.open - 1.0) - ROUND_TRIP_STOCK_COST_BPS / 10_000.0
        if side == "LONG":
            mfe = max(item.high / entry.open - 1.0 for item in window)
            mae = min(item.low / entry.open - 1.0 for item in window)
        else:
            mfe = max((entry.open - item.low) / entry.open for item in window)
            mae = min((entry.open - item.high) / entry.open for item in window)
        historical_iv = iv_by_date.get(decision.session_date, 50.0)
        distance = math.sqrt(
            ((historical[0] - current[0]) / trend_scale) ** 2
            + ((historical[1] - current[1]) / daily_scale) ** 2
            + ((historical[3] - current[3]) / max(trend_scale, 0.02)) ** 2
            + ((historical_iv - current_iv_percentile) / 25.0) ** 2
        )
        rows.append((distance, net, mfe, mae, decision.session_date, bar_index))
    if len(rows) < 20:
        raise CurrentIdeaError("{} has insufficient prior same-regime outcomes".format(ticker))
    rows.sort(key=lambda item: (item[0], item[4]))
    neighbors = rows[: min(ANALOG_COUNT, len(rows))]
    returns = [item[1] for item in neighbors]
    wins = sum(value > 0 for value in returns)
    lower, upper = _wilson_interval(wins, len(returns))
    indexes = sorted(item[5] for item in neighbors)
    selected: List[int] = []
    for index in indexes:
        if not selected or index - selected[-1] >= HORIZON_SESSIONS:
            selected.append(index)
    return AnalogEvidence(
        ticker=ticker,
        side=side,
        neighbor_count=len(neighbors),
        effective_nonoverlapping_count=len(selected),
        eligible_history_count=len(rows),
        empirical_probability_of_profit=wins / len(returns),
        wilson_95_lower_bound=lower,
        wilson_95_upper_bound=upper,
        mean_net_return=statistics.fmean(returns),
        median_net_return=statistics.median(returns),
        p10_net_return=sorted(returns)[max(0, math.ceil(0.10 * len(returns)) - 1)],
        bootstrap_p05_mean_return=_bootstrap_lower(returns, "{}:{}".format(ticker, side)),
        return_stddev=statistics.stdev(returns) if len(returns) > 1 else 0.01,
        mean_mfe=statistics.fmean(item[2] for item in neighbors),
        mean_mae=statistics.fmean(item[3] for item in neighbors),
        analog_dates=tuple(item[4] for item in neighbors),
    )


def _current_observation(
    history: Sequence[PriceObservation],
    quote_record: SourceRecord,
    as_of_date: str,
) -> Sequence[PriceObservation]:
    if quote_record.session_date != as_of_date:
        raise CurrentIdeaError("Schwab quote for {} is not from {}".format(_ticker(quote_record.payload), as_of_date))
    payload = quote_record.payload
    quote = payload.get("quote")
    regular = payload.get("regular")
    quote_values = quote if isinstance(quote, Mapping) else {}
    regular_values = regular if isinstance(regular, Mapping) else {}
    close = _number(regular_values.get("regularMarketLastPrice"), _number(quote_values.get("lastPrice")))
    high = _number(quote_values.get("highPrice"), close)
    low = _number(quote_values.get("lowPrice"), close)
    open_price = _number(quote_values.get("openPrice"), close)
    volume = _number(quote_values.get("totalVolume"))
    if min(open_price, close, high, low) <= 0 or volume <= 0 or high < low:
        raise CurrentIdeaError("Schwab quote has invalid regular-session OHLCV")
    ticker = _ticker(payload)
    current = PriceObservation(
        session_date=as_of_date,
        ticker=ticker,
        open=open_price,
        close=close,
        high=high,
        low=low,
        volume=volume,
        source="SCHWAB_REGULAR_SESSION_CURRENT",
    )
    prior = [item for item in history if item.session_date < as_of_date]
    return tuple(prior + [current])


def _backtest_index(replay: Optional[Mapping[str, Any]]) -> Mapping[Tuple[str, str], Mapping[str, Any]]:
    result = {}
    if not replay:
        return result
    for group in replay.get("groups") or ():
        if not isinstance(group, Mapping):
            continue
        ticker = str(group.get("ticker") or "").upper()
        strategy = str(group.get("strategy") or group.get("template") or "").upper()
        holdout = group.get("holdout")
        holdout_values = holdout if isinstance(holdout, Mapping) else {}
        metrics = holdout_values.get("metrics")
        if not isinstance(metrics, Mapping):
            metrics = group.get("metrics")
        if ticker and strategy and isinstance(metrics, Mapping):
            result[(ticker, strategy)] = metrics
    return result


def _profitability(
    option: Mapping[str, Any],
    metrics: Optional[Mapping[str, Any]],
    promotion: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    modeled = _number(option.get("modeled_probability_positive"), float("nan"))
    if not math.isfinite(modeled):
        modeled_value: Optional[float] = None
    else:
        modeled_value = modeled
    if not metrics:
        return {
            "estimated_probability_profitable": None,
            "confidence_rating": "INSUFFICIENT",
            "confidence_score_0_to_100": 0,
            "current_contract_modeled_pop": modeled_value,
            "historical_holdout_pop": None,
            "historical_effective_sample": 0,
            "pop_is_calibrated": False,
            "quoted_delta_is_not_pop": True,
            "explanation": "No same-structure ORATS holdout evidence is available.",
        }
    pop = _number(metrics.get("probability_of_profit"), float("nan"))
    effective = int(_number(metrics.get("effective_nonoverlapping_trade_count")) or 0)
    lower = _number(metrics.get("wilson_95_lower_bound"), float("nan"))
    passed, _ = historical_gate(metrics)
    if not math.isfinite(pop):
        estimate = None
    else:
        estimate = (effective * pop + 20.0 * 0.50) / (effective + 20.0)
    tactical = bool(promotion and promotion.get("is_tactical_ready"))
    if passed and effective >= 20:
        label = "HIGH"
    elif passed:
        label = "MEDIUM"
    elif tactical:
        label = "MODERATE-LOW"
    elif effective >= 8:
        label = "LOW"
    else:
        label = "INSUFFICIENT"
    raw_score = 100.0 * (estimate or 0.0) * min(effective / 20.0, 1.0)
    if not passed:
        raw_score = min(raw_score, 49.0)
    return {
        "estimated_probability_profitable": estimate,
        "confidence_rating": label,
        "confidence_score_0_to_100": round(raw_score),
        "current_contract_modeled_pop": modeled_value,
        "historical_holdout_pop": pop if math.isfinite(pop) else None,
        "historical_wilson_95_lower_bound": lower if math.isfinite(lower) else None,
        "historical_effective_sample": effective,
        "historical_mean_net_pnl_dollars": metrics.get("mean_net_pnl_dollars"),
        "historical_bootstrap_lower_mean_pnl_dollars": metrics.get(
            "bootstrap_2_5_percent_mean_net_pnl_dollars"
        ),
        "historical_profit_factor": metrics.get("profit_factor"),
        "pop_is_calibrated": bool(passed),
        "evidence_tier": (
            "FULL_EVIDENCE" if passed else "EXPLORATORY_TACTICAL" if tactical else "INSUFFICIENT"
        ),
        "multiple_testing_adjusted": False,
        "quoted_delta_is_not_pop": True,
        "explanation": (
            "Estimated POP is the same-structure ORATS holdout POP shrunk toward 50%; "
            "the scenario-model POP is shown separately and never substitutes for holdout evidence. "
            "A positive-skew long option can have positive expectancy with POP below 50%."
        ),
    }


def _core_forecasts(core: Mapping[str, Any]) -> Mapping[str, Any]:
    realized = _number(core.get("orFcst20d"))
    implied = _number(core.get("orIvFcst20d"))
    current_iv = _number(core.get("iv30d"))
    return {
        "realized_vol_forecast_20d_pct": realized,
        "implied_vol_forecast_20d_pct": implied,
        "current_implied_vol_30d_pct": current_iv,
        "implied_forecast_vs_current_wedge": implied / current_iv - 1.0 if current_iv > 0 else None,
        "current_implied_vs_realized_forecast_wedge": current_iv / realized - 1.0 if realized > 0 else None,
        "semantic_guard": {
            "orFcst20d": "future underlying realized/statistical volatility",
            "orIvFcst20d": "future option implied volatility",
        },
    }


def _stock_expression(
    baseline: PriceMoveBaseline,
    analog: AnalogEvidence,
    side: str,
    quote_payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    quote = quote_payload.get("quote")
    values = quote if isinstance(quote, Mapping) else {}
    high = _number(values.get("highPrice"), baseline.close)
    low = _number(values.get("lowPrice"), baseline.close)
    trigger = high * 1.001 if side == "LONG" else low * 0.999
    invalidation = low if side == "LONG" else high
    per_share_risk = abs(trigger - invalidation)
    shares = max(1, min(100, int(500.0 / max(per_share_risk, 0.01))))
    point_target = trigger * (
        1.0 + analog.mean_net_return if side == "LONG" else 1.0 - analog.mean_net_return
    )
    return {
        "vehicle": "STOCK",
        "side": side,
        "entry_trigger": trigger,
        "invalidation": invalidation,
        "five_session_point_target": point_target,
        "reference_shares": shares,
        "reference_maximum_planned_risk_dollars": shares * per_share_risk,
        "empirical_pop": analog.empirical_probability_of_profit,
        "empirical_pop_wilson_lower": analog.wilson_95_lower_bound,
        "mean_net_return": analog.mean_net_return,
        "bootstrap_lower_mean_return": analog.bootstrap_p05_mean_return,
        "status": "RESEARCH_CANDIDATE_PENDING_BROAD_STOCK_HOLDOUT",
        "is_manual_ready": False,
    }


def _weekday_session(as_of_date: str, sessions: int) -> str:
    value = date.fromisoformat(as_of_date)
    remaining = sessions
    while remaining > 0:
        value += timedelta(days=1)
        if value.weekday() < 5:
            remaining -= 1
    return value.isoformat()


def build_current_ideas(
    *,
    as_of_date: str,
    universe: Sequence[UniverseCandidate],
    core_records: Sequence[SourceRecord],
    daily_records: Sequence[SourceRecord],
    ivrank_records: Sequence[SourceRecord],
    schwab_quote_records: Sequence[SourceRecord],
    schwab_chain_records: Sequence[SourceRecord],
    orats_strike_records: Sequence[SourceRecord] = (),
    option_replay: Optional[Mapping[str, Any]] = None,
    portfolio_record: Optional[SourceRecord] = None,
    context: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Build a broad-screen slate with explicit evidence and promotion states."""

    tickers = tuple(item.ticker for item in universe)
    if not tickers:
        raise CurrentIdeaError("universe cannot be empty")
    cores = _records_by_ticker(core_records, "orats_cores")
    quotes = _records_by_ticker(schwab_quote_records, "schwab_quotes")
    chains = _records_by_ticker(schwab_chain_records, "schwab_option_chain")
    missing = [ticker for ticker in tickers if ticker not in cores or ticker not in quotes or ticker not in chains]
    if missing:
        raise CurrentIdeaError("missing current ORATS/Schwab records for {}".format(",".join(missing)))
    daily_rows = [record.payload for record in daily_records if record.source == "orats_hist_dailies"]
    iv_rows = [record.payload for record in ivrank_records if record.source == "orats_hist_ivrank"]
    history = parse_orats_price_history(daily_rows, tickers)
    bars = parse_orats_daily_rows(daily_rows, tickers=tickers)
    iv_history = parse_orats_ivrank_rows(iv_rows, tickers=tickers) if iv_rows else {}
    strike_rows: Dict[str, List[Mapping[str, Any]]] = {}
    for record in orats_strike_records:
        if record.source == "orats_strikes":
            strike_rows.setdefault(_ticker(record.payload), []).append(record.payload)
    replay_index = _backtest_index(option_replay)
    portfolio = portfolio_record.payload if portfolio_record is not None else None
    discovery_by_ticker = {item.ticker: item for item in universe}
    ideas: List[Mapping[str, Any]] = []
    dropped: List[Mapping[str, str]] = []

    for ticker in tickers:
        try:
            current_prices = _current_observation(history.observations.get(ticker, ()), quotes[ticker], as_of_date)
            baseline = compute_price_move_baseline(current_prices)
            side = "LONG" if baseline.trend_score_raw >= 0 else "SHORT"
            core = cores[ticker].payload
            forecasts = _core_forecasts(core)
            analog = _analog_evidence(
                ticker=ticker,
                side=side,
                prices=current_prices,
                bars=bars.get(ticker, ()),
                iv_history=iv_history.get(ticker, ()),
                current_iv_percentile=_number(core.get("ivPctile1y"), 50.0),
            )
        except (BaselineDataError, CurrentIdeaError, ValueError) as exc:
            dropped.append({"ticker": ticker, "reason": str(exc)})
            continue
        forecast_mean = analog.mean_net_return if side == "LONG" else -analog.mean_net_return
        sigma = max(
            analog.return_stddev,
            _number(forecasts["realized_vol_forecast_20d_pct"]) / 100.0
            * math.sqrt(HORIZON_SESSIONS / 252.0),
            0.01,
        )
        distribution = ForecastDistribution(
            mean_simple_return=max(min(forecast_mean, 0.50), -0.50),
            sigma_log_return=min(sigma, 1.0),
            horizon_days=HORIZON_SESSIONS,
        )
        current_options, option_rejections = select_current_verticals(
            ticker=ticker,
            side=side,
            as_of_date=as_of_date,
            chain=chains[ticker].payload,
            forecast=distribution,
            current_iv_30d_pct=_number(forecasts["current_implied_vol_30d_pct"], 1.0),
            forecast_implied_iv_20d_pct=_number(
                forecasts["implied_vol_forecast_20d_pct"],
                _number(forecasts["current_implied_vol_30d_pct"], 1.0),
            ),
            orats_strike_rows=strike_rows.get(ticker, ()),
            fresh_regular_session_quote=(
                quotes[ticker].session_date == as_of_date
                and chains[ticker].session_date == as_of_date
            ),
        )
        stock = _stock_expression(baseline, analog, side, quotes[ticker].payload)
        promoted_options = []
        for raw_option in current_options:
            option = dict(raw_option)
            fill = option.get("entry_fill_model")
            fill_values = fill if isinstance(fill, Mapping) else {}
            metrics = replay_index.get((ticker, str(option["strategy"])))
            promotion = evaluate_promotion(
                ticker=ticker,
                discovered=True,
                backtest_metrics=metrics,
                option=option,
                portfolio=portfolio,
            )
            option["entry_plan"] = {
                "decision_session": as_of_date,
                "entry_session": _weekday_session(as_of_date, 1),
                "entry_window": "near the next regular-session close after the trigger remains valid",
                "underlying_trigger": stock["entry_trigger"],
                "maximum_open_gap_pct": 0.01,
                "maximum_open_gap_price": (
                    stock["entry_trigger"] * 1.01
                    if side == "LONG"
                    else stock["entry_trigger"] * 0.99
                ),
                "same_session_invalidation": stock["invalidation"],
                "starting_option_limit": fill_values.get(
                    "starting_limit_per_share", option.get("entry_limit_signed_debit")
                ),
                "hard_maximum_option_limit": option.get("entry_limit_signed_debit"),
                "planned_exit_session": _weekday_session(as_of_date, HORIZON_SESSIONS),
                "exit_rule": "sell to close at or before the fifth regular-session close; replay uses exact bid",
                "no_fill_if_conditions_fail": True,
            }
            option["historical_holdout_metrics"] = dict(metrics) if metrics else None
            option["profitability"] = _profitability(option, metrics, promotion)
            option["promotion"] = promotion
            promoted_options.append(option)
        stage_rank = {
            "MANUAL_READY": 0,
            "TACTICAL_READY": 1,
            "CURRENT_CONTRACT_PASS": 2,
            "BACKTEST_PASS": 3,
            "DISCOVERED": 4,
        }
        promoted_options.sort(
            key=lambda item: (
                stage_rank.get(str(item["promotion"]["stage"]), 9),
                -int(bool((item["promotion"].get("gates") or {}).get("current_contract_pass"))),
                _number(item.get("selector_score"), 999.0),
                -_number(item.get("modeled_expected_pnl_dollars"))
                / max(_number(item.get("maximum_loss_dollars")), 1.0),
            )
        )
        selected = promoted_options[0] if promoted_options else None
        ideas.append(
            {
                "ticker": ticker,
                "direction": side,
                "discovery": discovery_by_ticker[ticker].to_dict(),
                "price": baseline.to_dict(),
                "analog_evidence": analog.to_dict(),
                "stock_expression": stock,
                "orats_forecasts": forecasts,
                "selected_option": selected,
                "option_candidates": promoted_options[:10],
                "option_rejection_counts": option_rejections,
                "promotion_stage": (
                    selected["promotion"]["stage"] if selected else "DISCOVERED"
                ),
                "is_manual_ready": bool(
                    selected and selected["promotion"]["is_manual_ready"]
                ),
                "is_tactical_ready": bool(
                    selected and selected["promotion"].get("is_tactical_ready")
                ),
                "is_executable_by_user": bool(
                    selected and selected["promotion"].get("is_executable_by_user")
                ),
                "source_contributions": {
                    "ORATS": {
                        "seeded": [
                            "broad-universe discovery and liquidity",
                            "split-adjusted daily history",
                            "realized-vol and implied-vol forecasts with separate semantics",
                            "current theoretical option values",
                            "exact historical chain replay and holdout POP",
                        ],
                        "current_session": cores[ticker].session_date,
                    },
                    "Schwab API": {
                        "seeded": [
                            "current regular-session stock OHLCV",
                            "exact current option symbols, bids, asks, Greeks, volume, and open interest",
                            "positions, balances, and working-order conflicts when portfolio snapshot is present",
                        ],
                        "quote_session": quotes[ticker].session_date,
                        "chain_session": chains[ticker].session_date,
                    },
                    "Public context": {
                        "seeded": [
                            "source-cited catalyst, internet-attention, macro, and geopolitical review"
                        ],
                        "numeric_vote": False,
                        "reason": "shadow-only until time-aligned ablation proves incremental net value",
                    },
                },
            }
        )

    ideas.sort(
        key=lambda item: (
            0 if item["is_executable_by_user"] else 1,
            {"MANUAL_READY": 0, "TACTICAL_READY": 1, "CURRENT_CONTRACT_PASS": 2, "BACKTEST_PASS": 3, "DISCOVERED": 4}.get(
                str(item["promotion_stage"]), 9
            ),
            -int(
                bool(
                    (((item.get("selected_option") or {}).get("promotion") or {}).get("gates") or {}).get(
                        "current_contract_pass"
                    )
                )
            ),
            -_number((item.get("selected_option") or {}).get("modeled_expected_pnl_dollars"))
            / max(_number((item.get("selected_option") or {}).get("maximum_loss_dollars")), 1.0),
            -_number(item["analog_evidence"].get("bootstrap_p05_mean_return")),
            -_number(item["discovery"].get("discovery_score")),
            item["ticker"],
        )
    )
    manual_ready = [item for item in ideas if item["is_manual_ready"]]
    tactical_ready = [item for item in ideas if item["is_tactical_ready"]]
    actionable = [item for item in ideas if item["is_executable_by_user"]]
    return {
        "schema_version": IDEA_SCHEMA_VERSION,
        "as_of_date": as_of_date,
        "status": "ACTIONABLE_CANDIDATES" if actionable else "NO_ACTIONABLE_TRADE",
        "broker_order_authorized": False,
        "broker_order_submitted": False,
        "manual_ready_trade_count": len(manual_ready),
        "tactical_ready_trade_count": len(tactical_ready),
        "actionable_trade_count": len(actionable),
        "top_candidate": ideas[0]["ticker"] if ideas else None,
        "universe": [item.to_dict() for item in universe],
        "universe_size_screened": len(universe),
        "idea_count": len(ideas),
        "dropped_candidates": dropped,
        "ideas": ideas,
        "market_context": dict(context or {"items": []}),
        "methodology": {
            "primary_sources": ["ORATS delayed and historical API", "Schwab read-only API"],
            "public_context_is_shadow_only": True,
            "vendor_trade_idea_feed_used": False,
            "stock_history_source": "ORATS hist/dailies adjusted OHLCV",
            "option_backtest_source": "ORATS hist/strikes exact EOD bid/ask",
            "current_execution_source": "Schwab exact option chain",
            "promotion_states": [
                "DISCOVERED",
                "BACKTEST_PASS",
                "CURRENT_CONTRACT_PASS",
                "PORTFOLIO_PASS",
                "TACTICAL_READY",
                "MANUAL_READY",
            ],
            "manual_ready_definition": (
                "same-structure holdout passed, current exact contract passed EV/liquidity/freshness, "
                "and Schwab portfolio constraints passed; the user still decides and submits"
            ),
            "tactical_ready_definition": (
                "same fixed rule has at least 30 holdout closes and 15 independent trades, positive "
                "train/validation/holdout expectancy and profit factors, current contract/portfolio pass, "
                "bootstrap lower mean is within 5% of current defined risk, and one-contract loss is no "
                "more than 0.05% NAV/$500; confidence interval may still cross zero"
            ),
        },
        "risk_notice": (
            "This is probabilistic decision support, not a profitability guarantee. No broker mutation "
            "surface exists; every trade decision and submission remains with the user."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
