"""Typed research objects used across CORAT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    complete: bool = True
    updated_at: str = ""
    source: str = "ORATS"


@dataclass(frozen=True)
class SourceTrace:
    source: str
    endpoint: str
    status: str
    fetched_at_utc: str
    latest_data_at: str
    rows: int
    cache_path: str
    cache_sha256: str
    params: Mapping[str, str]
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AvwapLevel:
    anchor_date: str
    anchor_reason: str
    value: float
    slope_5d: Optional[float]


@dataclass(frozen=True)
class TechnicalSnapshot:
    ticker: str
    as_of: str
    price: float
    price_date: str
    price_complete: bool
    ema20: Optional[float]
    sma50: Optional[float]
    sma200: Optional[float]
    atr14: Optional[float]
    rsi14: Optional[float]
    return_5d: Optional[float]
    return_20d: Optional[float]
    return_60d: Optional[float]
    relative_volume_20d: Optional[float]
    average_dollar_volume_20d: Optional[float]
    prior_high_20d: Optional[float]
    prior_low_20d: Optional[float]
    support: Optional[float]
    resistance: Optional[float]
    extension_from_ema_atr: Optional[float]
    avwaps: List[AvwapLevel] = field(default_factory=list)
    price_source: str = ""
    price_updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SetupSignal:
    name: str
    direction: str
    strength: float
    triggered: bool
    reason: str
    trigger: str
    invalidation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HistoricalStats:
    method: str
    sample_size: int
    reliable: bool
    horizon_returns: Mapping[str, Mapping[str, Optional[float]]]
    primary_horizon: int
    win_rate: Optional[float]
    expectancy: Optional[float]
    average_winner: Optional[float]
    average_loser: Optional[float]
    profit_factor: Optional[float]
    mae: Optional[float]
    mfe: Optional[float]
    max_drawdown: Optional[float]
    signal_dates: List[str] = field(default_factory=list)
    primary_returns: List[float] = field(default_factory=list)
    primary_paths: List[List[float]] = field(default_factory=list)
    primary_adverse_paths: List[List[float]] = field(default_factory=list)
    primary_favorable_paths: List[List[float]] = field(default_factory=list)
    similarity_dimensions: List[str] = field(default_factory=list)
    missing_dimensions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptionLeg:
    action: str
    option_type: str
    strike: float
    expiration: str
    quantity: int
    bid: float
    ask: float
    theoretical_value: Optional[float]
    delta: float
    gamma: float
    theta: float
    vega: float
    open_interest: int
    volume: int
    spread_pct: Optional[float]
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None


@dataclass(frozen=True)
class OptionStructure:
    valid: bool
    strategy: str
    expiration: str
    dte: int
    legs: List[OptionLeg]
    debit_credit: str
    expected_entry: Optional[float]
    natural_entry: Optional[float]
    maximum_loss: Optional[float]
    maximum_gain: Optional[float]
    breakeven: Optional[float]
    reward_risk: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    vega: Optional[float]
    theta_holding_cost: Optional[float]
    orats_theoretical_value: Optional[float]
    theoretical_edge: Optional[float]
    implied_volatility: Optional[float]
    quote_trade_date: str
    quote_updated_at: str
    reasons: List[str] = field(default_factory=list)
    midpoint_entry: Optional[float] = None
    entry_fill_fraction: Optional[float] = None
    candidate_count: int = 0
    selection_train_size: int = 0
    selection_test_size: int = 0
    selection_method: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TradePlan:
    vehicle: str
    entry_low: float
    entry_high: float
    trigger: str
    stop: float
    target_1: float
    target_2: float
    holding_sessions: int
    reward_risk_1: float
    reward_risk_2: float
    risk_per_share: float
    portfolio_risk_dollars: Optional[float]
    units: Optional[int]
    maximum_loss: Optional[float]
    risk_basis_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def object_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return value
