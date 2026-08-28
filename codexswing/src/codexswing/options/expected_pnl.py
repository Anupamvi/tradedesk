"""Distribution-based stock/vertical expression comparison."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from codexswing.options.pricing import black_scholes_price
from codexswing.options.structures import OptionQuote, StructureError, VerticalSpread


@dataclass(frozen=True)
class ForecastDistribution:
    mean_simple_return: float
    sigma_log_return: float
    horizon_days: int

    def __post_init__(self) -> None:
        if self.mean_simple_return <= -1.0 or self.mean_simple_return > 10.0:
            raise ValueError("mean_simple_return must be greater than -100% and no more than 1000%")
        if self.sigma_log_return <= 0 or self.sigma_log_return > 2:
            raise ValueError("sigma_log_return must be between zero and two")
        if self.horizon_days < 1 or self.horizon_days > 252:
            raise ValueError("horizon_days must be between 1 and 252")


@dataclass(frozen=True)
class CostAssumptions:
    contracts: int = 1
    shares: int = 100
    commission_per_contract_per_leg: float = 0.65
    exit_half_spread_multiplier: float = 1.0
    stock_round_trip_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.contracts < 1 or self.shares < 1:
            raise ValueError("contracts and shares must be positive")
        if self.commission_per_contract_per_leg < 0 or self.exit_half_spread_multiplier < 0:
            raise ValueError("cost assumptions cannot be negative")
        if self.stock_round_trip_bps < 0:
            raise ValueError("stock_round_trip_bps cannot be negative")


@dataclass(frozen=True)
class ExpressionEvaluation:
    ticker: str
    strategy: str
    horizon_days: int
    contracts: int
    natural_entry_debit_dollars: float
    modeled_entry_debit_dollars: float
    entry_price_source: str
    expected_pnl_before_exit_costs: float
    expected_pnl_after_costs: float
    probability_positive_after_costs: float
    p05_pnl_after_costs: float
    expiry_max_loss_dollars: float
    expiry_max_profit_dollars: float
    modeled_exit_cost_dollars: float
    round_trip_commissions: float
    maximum_quote_spread_pct: float
    minimum_open_interest: int
    minimum_volume: int
    source_spot_difference_pct: float
    status: str = "CURRENT_CONTRACT_MODEL_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SingleOptionEvaluation:
    ticker: str
    strategy: str
    horizon_days: int
    contracts: int
    natural_entry_debit_dollars: float
    modeled_entry_debit_dollars: float
    entry_price_source: str
    expected_pnl_before_exit_costs: float
    expected_pnl_after_costs: float
    probability_positive_after_costs: float
    p05_pnl_after_costs: float
    expiry_max_loss_dollars: float
    expiry_max_profit_dollars: Optional[float]
    modeled_exit_cost_dollars: float
    round_trip_commissions: float
    maximum_quote_spread_pct: float
    minimum_open_interest: int
    minimum_volume: int
    source_spot_difference_pct: float
    status: str = "CURRENT_CONTRACT_MODEL_ONLY"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StockExpressionEvaluation:
    ticker: str
    horizon_days: int
    shares: int
    expected_pnl_after_costs: float
    probability_positive_after_costs: float
    p05_pnl_after_costs: float
    modeled_round_trip_cost: float
    status: str = "RESEARCH_ONLY_UNVALIDATED_EXPRESSION"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normal_scenarios(step: float = 0.05, bound: float = 4.0) -> Tuple[Tuple[float, float], ...]:
    count = int(round(2 * bound / step))
    raw = []
    for index in range(count + 1):
        z = -bound + index * step
        weight = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        raw.append((z, weight))
    total = sum(weight for _, weight in raw)
    return tuple((z, weight / total) for z, weight in raw)


def _weighted_quantile(values: Sequence[Tuple[float, float]], probability: float) -> float:
    ordered = sorted(values, key=lambda item: item[0])
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= probability:
            return value
    return ordered[-1][0]


def _future_spot(spot: float, forecast: ForecastDistribution, z: float) -> float:
    log_location = math.log1p(forecast.mean_simple_return) - 0.5 * forecast.sigma_log_return ** 2
    return spot * math.exp(log_location + forecast.sigma_log_return * z)


def evaluate_stock(
    ticker: str,
    spot: float,
    forecast: ForecastDistribution,
    costs: CostAssumptions = CostAssumptions(),
) -> StockExpressionEvaluation:
    if spot <= 0:
        raise ValueError("spot must be positive")
    scenarios = _normal_scenarios()
    round_trip_cost = spot * costs.shares * costs.stock_round_trip_bps / 10_000.0
    outcomes = []
    for z, weight in scenarios:
        pnl = (_future_spot(spot, forecast, z) - spot) * costs.shares - round_trip_cost
        outcomes.append((pnl, weight))
    expected = sum(value * weight for value, weight in outcomes)
    probability_positive = sum(weight for value, weight in outcomes if value > 0)
    return StockExpressionEvaluation(
        ticker=ticker,
        horizon_days=forecast.horizon_days,
        shares=costs.shares,
        expected_pnl_after_costs=expected,
        probability_positive_after_costs=probability_positive,
        p05_pnl_after_costs=_weighted_quantile(outcomes, 0.05),
        modeled_round_trip_cost=round_trip_cost,
    )


def evaluate_long_option(
    quote: OptionQuote,
    forecast: ForecastDistribution,
    reference_spot: float,
    risk_free_rate: float,
    costs: CostAssumptions = CostAssumptions(),
    iv_multiplier: float = 1.0,
    max_spot_disagreement_pct: float = 0.01,
    entry_debit_per_share: Optional[float] = None,
    entry_price_source: str = "NATURAL_QUOTE",
) -> SingleOptionEvaluation:
    """Evaluate one long call/put with the same five-session distribution as spreads."""

    if reference_spot <= 0:
        raise ValueError("reference_spot must be positive")
    if risk_free_rate < -0.10 or risk_free_rate > 0.50:
        raise ValueError("risk_free_rate is outside the supported range")
    if iv_multiplier <= 0 or iv_multiplier > 3:
        raise ValueError("iv_multiplier must be between zero and three")
    entry_debit = quote.ask if entry_debit_per_share is None else float(entry_debit_per_share)
    if entry_debit <= 0:
        raise StructureError("long-option entry debit must be positive")
    spot_difference = abs(quote.spot - reference_spot) / reference_spot
    if spot_difference > max_spot_disagreement_pct:
        raise StructureError("option and reference underlying spots disagree beyond tolerance")
    remaining_days = max((quote.expiration - quote.quote_date).days - forecast.horizon_days, 0)
    remaining_years = remaining_days / 365.0
    scenarios = _normal_scenarios()
    exit_cost = (
        quote.spread
        * 0.5
        * 100.0
        * costs.contracts
        * costs.exit_half_spread_multiplier
    )
    commissions = costs.commission_per_contract_per_leg * costs.contracts * 2.0
    outcomes_before = []
    outcomes_after = []
    for z, weight in scenarios:
        future_spot = _future_spot(reference_spot, forecast, z)
        future_value = black_scholes_price(
            spot=future_spot,
            strike=quote.strike,
            time_years=remaining_years,
            rate=risk_free_rate,
            volatility=quote.implied_volatility * iv_multiplier,
            right=quote.right,
            dividend_yield=risk_free_rate - quote.residual_rate,
        )
        pnl_before = (future_value - entry_debit) * 100.0 * costs.contracts
        pnl_after = pnl_before - exit_cost - commissions
        outcomes_before.append((pnl_before, weight))
        outcomes_after.append((pnl_after, weight))
    expected_before = sum(value * weight for value, weight in outcomes_before)
    expected_after = sum(value * weight for value, weight in outcomes_after)
    probability_positive = sum(weight for value, weight in outcomes_after if value > 0)
    max_loss = -(entry_debit * 100.0 * costs.contracts) - commissions
    max_profit = (
        (quote.strike - entry_debit) * 100.0 * costs.contracts - commissions
        if quote.right == "put"
        else None
    )
    return SingleOptionEvaluation(
        ticker=quote.ticker,
        strategy="long_call" if quote.right == "call" else "long_put",
        horizon_days=forecast.horizon_days,
        contracts=costs.contracts,
        natural_entry_debit_dollars=quote.ask * 100.0 * costs.contracts,
        modeled_entry_debit_dollars=entry_debit * 100.0 * costs.contracts,
        entry_price_source=entry_price_source,
        expected_pnl_before_exit_costs=expected_before,
        expected_pnl_after_costs=expected_after,
        probability_positive_after_costs=probability_positive,
        p05_pnl_after_costs=_weighted_quantile(outcomes_after, 0.05),
        expiry_max_loss_dollars=max_loss,
        expiry_max_profit_dollars=max_profit,
        modeled_exit_cost_dollars=exit_cost,
        round_trip_commissions=commissions,
        maximum_quote_spread_pct=quote.spread / max(quote.mid, 0.01),
        minimum_open_interest=quote.open_interest,
        minimum_volume=quote.volume,
        source_spot_difference_pct=spot_difference,
    )


def evaluate_vertical(
    spread: VerticalSpread,
    forecast: ForecastDistribution,
    reference_spot: float,
    risk_free_rate: float,
    costs: CostAssumptions = CostAssumptions(),
    iv_multiplier: float = 1.0,
    max_spot_disagreement_pct: float = 0.01,
    entry_debit_per_share: Optional[float] = None,
    entry_price_source: str = "NATURAL_QUOTE",
) -> ExpressionEvaluation:
    if reference_spot <= 0:
        raise ValueError("reference_spot must be positive")
    if risk_free_rate < -0.10 or risk_free_rate > 0.50:
        raise ValueError("risk_free_rate is outside the supported range")
    if iv_multiplier <= 0 or iv_multiplier > 3:
        raise ValueError("iv_multiplier must be between zero and three")
    entry_debit = (
        spread.entry_debit_per_share
        if entry_debit_per_share is None
        else float(entry_debit_per_share)
    )
    if spread.strategy.endswith("debit") and not (0 < entry_debit < spread.width):
        raise StructureError("debit entry limit must be between zero and spread width")
    if spread.strategy.endswith("credit") and not (0 < -entry_debit < spread.width):
        raise StructureError("credit entry limit must be between zero and spread width")
    spot_difference = abs(spread.spot - reference_spot) / reference_spot
    if spot_difference > max_spot_disagreement_pct:
        raise StructureError("ORATS and reference underlying spots disagree beyond tolerance")
    quote_date = spread.long_leg.quote.quote_date
    expiration = spread.long_leg.quote.expiration
    days_to_expiry = (expiration - quote_date).days
    remaining_days = max(days_to_expiry - forecast.horizon_days, 0)
    remaining_years = remaining_days / 365.0
    scenarios = _normal_scenarios()
    exit_cost = (
        sum(leg.quote.spread * 0.5 * abs(leg.quantity) for leg in spread.legs)
        * 100.0
        * costs.contracts
        * costs.exit_half_spread_multiplier
    )
    commissions = (
        costs.commission_per_contract_per_leg * len(spread.legs) * costs.contracts * 2.0
    )
    outcomes_before = []
    outcomes_after = []
    for z, weight in scenarios:
        future_spot = _future_spot(reference_spot, forecast, z)
        value_per_share = 0.0
        for leg in spread.legs:
            quote = leg.quote
            option_value = black_scholes_price(
                spot=future_spot,
                strike=quote.strike,
                time_years=remaining_years,
                rate=risk_free_rate,
                volatility=quote.implied_volatility * iv_multiplier,
                right=quote.right,
                dividend_yield=risk_free_rate - quote.residual_rate,
            )
            value_per_share += leg.quantity * option_value
        pnl_before = (
            value_per_share - entry_debit
        ) * 100.0 * costs.contracts
        pnl_after = pnl_before - exit_cost - commissions
        outcomes_before.append((pnl_before, weight))
        outcomes_after.append((pnl_after, weight))
    expected_before = sum(value * weight for value, weight in outcomes_before)
    expected_after = sum(value * weight for value, weight in outcomes_after)
    probability_positive = sum(weight for value, weight in outcomes_after if value > 0)
    expiry_min, expiry_max = spread.expiry_pnl_bounds_dollars(
        costs.contracts,
        entry_debit_per_share=entry_debit,
    )
    return ExpressionEvaluation(
        ticker=spread.ticker,
        strategy=spread.strategy,
        horizon_days=forecast.horizon_days,
        contracts=costs.contracts,
        natural_entry_debit_dollars=spread.entry_debit_per_share * 100.0 * costs.contracts,
        modeled_entry_debit_dollars=entry_debit * 100.0 * costs.contracts,
        entry_price_source=entry_price_source,
        expected_pnl_before_exit_costs=expected_before,
        expected_pnl_after_costs=expected_after,
        probability_positive_after_costs=probability_positive,
        p05_pnl_after_costs=_weighted_quantile(outcomes_after, 0.05),
        expiry_max_loss_dollars=expiry_min - commissions,
        expiry_max_profit_dollars=expiry_max - commissions,
        modeled_exit_cost_dollars=exit_cost,
        round_trip_commissions=commissions,
        maximum_quote_spread_pct=spread.maximum_quote_spread_pct,
        minimum_open_interest=spread.minimum_open_interest,
        minimum_volume=spread.minimum_volume,
        source_spot_difference_pct=spot_difference,
    )
