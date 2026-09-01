"""Scenario-based, per-unit net edge calculations."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Sequence, Tuple

from .domain import Scenario


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite" % name)
    return value


def _nonnegative(value: float, name: str) -> float:
    value = _finite(value, name)
    if value < 0.0:
        raise ValueError("%s cannot be negative" % name)
    return value


class PriceConvention(str, Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"


@dataclass(frozen=True)
class CostBreakdown:
    commissions: float
    fees: float
    slippage: float
    assignment_exercise: float = 0.0
    dividends: float = 0.0
    early_exit: float = 0.0
    model_version: str = "UNSPECIFIED"
    spread_reference: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "commissions",
            "fees",
            "slippage",
            "assignment_exercise",
            "dividends",
            "early_exit",
        ):
            object.__setattr__(
                self, field_name, _nonnegative(getattr(self, field_name), field_name)
            )
        if not self.model_version or not self.model_version.strip():
            raise ValueError("cost model_version is required")
        object.__setattr__(
            self,
            "spread_reference",
            _nonnegative(self.spread_reference, "spread_reference"),
        )

    @property
    def total(self) -> float:
        return math.fsum(
            (
                self.commissions,
                self.fees,
                self.slippage,
                self.assignment_exercise,
                self.dividends,
                self.early_exit,
            )
        )


@dataclass(frozen=True)
class EdgeEstimate:
    gross_expected_value: float
    net_expected_value: float
    conservative_net_expected_value: float
    expected_return_on_max_loss: float
    conservative_return_on_max_loss: float
    model_fair_price: float
    executable_limit_price: float
    price_convention: PriceConvention
    maximum_profit: Optional[float]
    maximum_loss: float
    breakevens: Tuple[float, ...]
    target_pnl: float
    stop_pnl: float
    expected_shortfall: float
    adverse_gap_stress_loss: float
    costs: CostBreakdown
    point_scenarios: Tuple[Scenario, ...] = ()
    conservative_scenarios: Tuple[Scenario, ...] = ()
    conservative_gross_expected_value: Optional[float] = None
    calculation_version: str = "cultra.edge.v1"

    def __post_init__(self) -> None:
        for field_name in (
            "gross_expected_value",
            "net_expected_value",
            "conservative_net_expected_value",
            "expected_return_on_max_loss",
            "conservative_return_on_max_loss",
            "target_pnl",
            "stop_pnl",
        ):
            object.__setattr__(self, field_name, _finite(getattr(self, field_name), field_name))
        for field_name in (
            "model_fair_price",
            "executable_limit_price",
            "maximum_loss",
            "expected_shortfall",
            "adverse_gap_stress_loss",
        ):
            object.__setattr__(
                self, field_name, _nonnegative(getattr(self, field_name), field_name)
            )
        if self.maximum_loss <= 0.0:
            raise ValueError("maximum_loss must be finite and positive")
        if self.maximum_profit is not None:
            object.__setattr__(
                self,
                "maximum_profit",
                _nonnegative(self.maximum_profit, "maximum_profit"),
            )
        checked_breakevens = tuple(
            _nonnegative(value, "breakeven") for value in self.breakevens
        )
        object.__setattr__(self, "breakevens", checked_breakevens)
        object.__setattr__(self, "point_scenarios", tuple(self.point_scenarios))
        object.__setattr__(
            self, "conservative_scenarios", tuple(self.conservative_scenarios)
        )
        if bool(self.point_scenarios) != bool(self.conservative_scenarios):
            raise ValueError("point and conservative scenarios must be stored together")
        if not self.calculation_version or not self.calculation_version.strip():
            raise ValueError("calculation_version is required")
        if self.point_scenarios:
            reproduced_gross = _expected_value(self.point_scenarios, "saved point")
            reproduced_conservative_gross = _expected_value(
                self.conservative_scenarios, "saved conservative"
            )
            if not math.isclose(
                self.gross_expected_value,
                reproduced_gross,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("gross_expected_value is not reproducible")
            if self.conservative_gross_expected_value is None or not math.isclose(
                float(self.conservative_gross_expected_value),
                reproduced_conservative_gross,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("conservative gross EV is not reproducible")
            if not math.isclose(
                self.net_expected_value,
                reproduced_gross - self.costs.total,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("net_expected_value is not reproducible")
            if not math.isclose(
                self.conservative_net_expected_value,
                reproduced_conservative_gross - self.costs.total,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("conservative net EV is not reproducible")
            point_shape = tuple(
                (item.label, item.outcome) for item in self.point_scenarios
            )
            conservative_shape = tuple(
                (item.label, item.outcome) for item in self.conservative_scenarios
            )
            if point_shape != conservative_shape:
                raise ValueError(
                    "conservative scenarios must preserve point outcome labels"
                )
            if any(
                conservative.net_pnl > point.net_pnl + 1e-12
                for point, conservative in zip(
                    self.point_scenarios, self.conservative_scenarios
                )
            ):
                raise ValueError(
                    "a conservative scenario cannot improve its point P/L"
                )
            if self.conservative_net_expected_value > self.net_expected_value + 1e-12:
                raise ValueError("conservative net EV cannot exceed point net EV")
        elif self.conservative_gross_expected_value is not None:
            object.__setattr__(
                self,
                "conservative_gross_expected_value",
                _finite(
                    self.conservative_gross_expected_value,
                    "conservative_gross_expected_value",
                ),
            )
        expected_ratio = self.net_expected_value / self.maximum_loss
        conservative_ratio = self.conservative_net_expected_value / self.maximum_loss
        if not math.isclose(
            self.expected_return_on_max_loss, expected_ratio, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError("expected_return_on_max_loss is inconsistent")
        if not math.isclose(
            self.conservative_return_on_max_loss,
            conservative_ratio,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("conservative_return_on_max_loss is inconsistent")

    @property
    def is_positive(self) -> bool:
        return self.net_expected_value > 0.0 and self.conservative_net_expected_value > 0.0

    @property
    def ranking_score(self) -> float:
        """Conservative net EV per maximum-loss dollar."""

        return self.conservative_return_on_max_loss


def _expected_value(scenarios: Sequence[Scenario], name: str) -> float:
    if not scenarios:
        raise ValueError("%s scenarios cannot be empty" % name)
    probability_sum = math.fsum(scenario.probability for scenario in scenarios)
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("%s scenario probabilities must sum to 1" % name)
    labels = tuple(scenario.label for scenario in scenarios)
    if len(set(labels)) != len(labels):
        raise ValueError("%s scenario labels must be unique" % name)
    return math.fsum(scenario.probability * scenario.net_pnl for scenario in scenarios)


def compute_edge(
    scenarios: Sequence[Scenario],
    conservative_scenarios: Sequence[Scenario],
    maximum_loss: float,
    costs: CostBreakdown,
    model_fair_price: float,
    executable_limit_price: float,
    price_convention: PriceConvention,
    maximum_profit: Optional[float] = None,
    breakevens: Sequence[float] = (),
    target_pnl: float = 0.0,
    stop_pnl: float = 0.0,
    expected_shortfall: float = 0.0,
    adverse_gap_stress_loss: float = 0.0,
) -> EdgeEstimate:
    """Calculate reproducible point and conservative per-unit net EV.

    The conservative distribution is mandatory; Cultra never silently labels
    the point-estimate distribution conservative.  Scenario P/L values are
    before the common modeled costs supplied in ``costs``.
    """

    if not isinstance(costs, CostBreakdown):
        raise TypeError("costs must be CostBreakdown")
    gross_expected_value = _expected_value(scenarios, "point")
    conservative_gross = _expected_value(conservative_scenarios, "conservative")
    maximum_loss = _nonnegative(maximum_loss, "maximum_loss")
    if maximum_loss <= 0.0:
        raise ValueError("maximum_loss must be finite and positive")
    net_expected_value = gross_expected_value - costs.total
    conservative_net_expected_value = conservative_gross - costs.total
    return EdgeEstimate(
        gross_expected_value=gross_expected_value,
        net_expected_value=net_expected_value,
        conservative_net_expected_value=conservative_net_expected_value,
        expected_return_on_max_loss=net_expected_value / maximum_loss,
        conservative_return_on_max_loss=conservative_net_expected_value / maximum_loss,
        model_fair_price=model_fair_price,
        executable_limit_price=executable_limit_price,
        price_convention=price_convention,
        maximum_profit=maximum_profit,
        maximum_loss=maximum_loss,
        breakevens=tuple(breakevens),
        target_pnl=target_pnl,
        stop_pnl=stop_pnl,
        expected_shortfall=expected_shortfall,
        adverse_gap_stress_loss=adverse_gap_stress_loss,
        costs=costs,
        point_scenarios=tuple(scenarios),
        conservative_scenarios=tuple(conservative_scenarios),
        conservative_gross_expected_value=conservative_gross,
    )
