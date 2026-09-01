"""Frozen exact-structure selection and pathwise outcome mechanics for Cultra.

The module is deliberately model-free and network-free.  It converts a
contemporaneous option chain into one deterministic normalized structure for
each frozen catalog family, then marks those exact contracts through a saved
historical path.  No outcome, future membership, or later contract selection
can influence the entry legs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .catalog import FROZEN_STRATEGY_CATALOG, get_strategy
from .domain import LegAction, LegQuote, OptionLeg, OptionType, exact_quote_map
from .economics import (
    CONTRACT_MULTIPLIER,
    PayoffEnvelope,
    executable_entry_debit,
    same_expiry_payoff_envelope,
)
from .edge import CostBreakdown, PriceConvention


STRUCTURE_TEMPLATE_VERSION = "cultra-structure-templates-v1"
MINIMUM_OPEN_INTEREST = 50
MAXIMUM_RELATIVE_SPREAD = 0.35
MAXIMUM_GROUP_CHOICES = 12


class StructureError(ValueError):
    """A frozen structure could not be selected or resolved exactly."""


class ExpiryRole(str, Enum):
    SAME = "SAME"
    FRONT = "FRONT"
    BACK = "BACK"


class EntryKind(str, Enum):
    DEBIT = "DEBIT"
    CREDIT = "CREDIT"
    EITHER = "EITHER"


class RiskClass(str, Enum):
    FINITE_SAME_EXPIRY = "FINITE_SAME_EXPIRY"
    FINITE_TERM_DEBIT = "FINITE_TERM_DEBIT"
    UNDEFINED_RESEARCH_ONLY = "UNDEFINED_RESEARCH_ONLY"


class StrikeGeometry(str, Enum):
    ORDERED = "ORDERED"
    SYMMETRIC_THREE = "SYMMETRIC_THREE"
    ASYMMETRIC_THREE = "ASYMMETRIC_THREE"


@dataclass(frozen=True)
class FrozenLegRule:
    action: LegAction
    option_type: OptionType
    ratio: int
    expiry_role: ExpiryRole
    strike_group: int
    target_call_delta: float

    def __post_init__(self) -> None:
        if isinstance(self.ratio, bool) or not isinstance(self.ratio, int) or self.ratio <= 0:
            raise StructureError("frozen leg ratio must be a positive integer")
        if (
            isinstance(self.strike_group, bool)
            or not isinstance(self.strike_group, int)
            or self.strike_group < 0
        ):
            raise StructureError("frozen strike group must be a nonnegative integer")
        delta = float(self.target_call_delta)
        if not math.isfinite(delta) or not 0.01 <= delta <= 0.99:
            raise StructureError("target call delta must be between 0.01 and 0.99")
        object.__setattr__(self, "target_call_delta", delta)


@dataclass(frozen=True)
class StructureTemplate:
    strategy_id: str
    signal_profile: str
    signal_bias: str
    entry_kind: EntryKind
    risk_class: RiskClass
    geometry: StrikeGeometry
    legs: Tuple[FrozenLegRule, ...]
    target_fraction_of_risk: float = 0.50
    stop_fraction_of_risk: float = 0.40
    research_only_stop_multiple: float = 2.00

    def __post_init__(self) -> None:
        definition = get_strategy(self.strategy_id)
        if len(self.legs) != definition.leg_count:
            raise StructureError("template leg count does not match the frozen catalog")
        groups = tuple(sorted({item.strike_group for item in self.legs}))
        if groups != tuple(range(len(groups))):
            raise StructureError("template strike groups must be contiguous")
        roles = {item.expiry_role for item in self.legs}
        if ExpiryRole.SAME in roles and roles != {ExpiryRole.SAME}:
            raise StructureError("same-expiry and term-expiry roles cannot be mixed")
        if ExpiryRole.SAME not in roles and roles != {ExpiryRole.FRONT, ExpiryRole.BACK}:
            raise StructureError("term structures require both front and back expiries")
        is_finite = self.risk_class is not RiskClass.UNDEFINED_RESEARCH_ONLY
        if is_finite != definition.defined_risk_by_construction:
            raise StructureError("template risk class does not match the frozen catalog")
        if definition.ticket_eligible_structure and not is_finite:
            raise StructureError("ticket-eligible template must have finite risk")
        if self.risk_class is RiskClass.FINITE_TERM_DEBIT and roles == {ExpiryRole.SAME}:
            raise StructureError("term-debit risk requires multiple expiries")
        if self.risk_class is RiskClass.FINITE_SAME_EXPIRY and roles != {ExpiryRole.SAME}:
            raise StructureError("same-expiry risk requires one expiry role")
        if self.geometry in {
            StrikeGeometry.SYMMETRIC_THREE,
            StrikeGeometry.ASYMMETRIC_THREE,
        } and len(groups) != 3:
            raise StructureError("three-strike geometry requires exactly three groups")
        for value, name in (
            (self.target_fraction_of_risk, "target fraction"),
            (self.stop_fraction_of_risk, "stop fraction"),
            (self.research_only_stop_multiple, "research stop multiple"),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise StructureError("%s must be finite and positive" % name)
        for name in ("signal_profile", "signal_bias"):
            if not str(getattr(self, name)).strip():
                raise StructureError("%s is required" % name)

    @property
    def template_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "version": STRUCTURE_TEMPLATE_VERSION,
                    "strategy_id": self.strategy_id,
                    "signal_profile": self.signal_profile,
                    "signal_bias": self.signal_bias,
                    "entry_kind": self.entry_kind.value,
                    "risk_class": self.risk_class.value,
                    "geometry": self.geometry.value,
                    "legs": [
                        {
                            "action": item.action.value,
                            "option_type": item.option_type.value,
                            "ratio": item.ratio,
                            "expiry_role": item.expiry_role.value,
                            "strike_group": item.strike_group,
                            "target_call_delta": item.target_call_delta,
                        }
                        for item in self.legs
                    ],
                    "target_fraction_of_risk": self.target_fraction_of_risk,
                    "stop_fraction_of_risk": self.stop_fraction_of_risk,
                    "research_only_stop_multiple": self.research_only_stop_multiple,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


def _leg(
    action: LegAction,
    option_type: OptionType,
    delta: float,
    group: int,
    *,
    ratio: int = 1,
    expiry: ExpiryRole = ExpiryRole.SAME,
) -> FrozenLegRule:
    return FrozenLegRule(action, option_type, ratio, expiry, group, delta)


BUY = LegAction.BUY
SELL = LegAction.SELL
CALL = OptionType.CALL
PUT = OptionType.PUT
SAME = ExpiryRole.SAME
FRONT = ExpiryRole.FRONT
BACK = ExpiryRole.BACK


def _template(
    strategy_id: str,
    profile: str,
    bias: str,
    entry: EntryKind,
    risk: RiskClass,
    legs: Sequence[FrozenLegRule],
    geometry: StrikeGeometry = StrikeGeometry.ORDERED,
) -> StructureTemplate:
    return StructureTemplate(
        strategy_id=strategy_id,
        signal_profile=profile,
        signal_bias=bias,
        entry_kind=entry,
        risk_class=risk,
        geometry=geometry,
        legs=tuple(legs),
    )


_DIRECTIONAL = "DIRECTIONAL_COMPOSITE_V1"
_VOLATILITY = "VOLATILITY_DISLOCATION_V1"
_TERM = "TERM_STRUCTURE_SLOPE_V1"
_SKEW = "SKEW_CURVATURE_V1"
_PREMIUM = "PREMIUM_DISLOCATION_V1"


FROZEN_STRUCTURE_TEMPLATES: Tuple[StructureTemplate, ...] = (
    _template("LONG_CALL", _DIRECTIONAL, "BULLISH", EntryKind.DEBIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, CALL, .55, 0),)),
    _template("LONG_PUT", _DIRECTIONAL, "BEARISH", EntryKind.DEBIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .45, 0),)),
    _template("CALL_DEBIT_VERTICAL", _DIRECTIONAL, "BULLISH", EntryKind.DEBIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, CALL, .60, 0), _leg(SELL, CALL, .35, 1))),
    _template("PUT_DEBIT_VERTICAL", _DIRECTIONAL, "BEARISH", EntryKind.DEBIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(SELL, PUT, .65, 0), _leg(BUY, PUT, .40, 1))),
    _template("CALL_CREDIT_VERTICAL", _DIRECTIONAL, "BEARISH", EntryKind.CREDIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(SELL, CALL, .45, 0), _leg(BUY, CALL, .25, 1))),
    _template("PUT_CREDIT_VERTICAL", _DIRECTIONAL, "BULLISH", EntryKind.CREDIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .75, 0), _leg(SELL, PUT, .55, 1))),
    _template("CALL_DIAGONAL", _DIRECTIONAL, "BULLISH", EntryKind.DEBIT, RiskClass.FINITE_TERM_DEBIT, (_leg(BUY, CALL, .60, 0, expiry=BACK), _leg(SELL, CALL, .35, 1, expiry=FRONT))),
    _template("PUT_DIAGONAL", _DIRECTIONAL, "BEARISH", EntryKind.DEBIT, RiskClass.FINITE_TERM_DEBIT, (_leg(SELL, PUT, .65, 0, expiry=FRONT), _leg(BUY, PUT, .40, 1, expiry=BACK))),
    _template("LONG_STRADDLE", _VOLATILITY, "LONG_VOL", EntryKind.DEBIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .50, 0), _leg(BUY, CALL, .50, 0))),
    _template("SHORT_STRADDLE", _VOLATILITY, "SHORT_VOL", EntryKind.CREDIT, RiskClass.UNDEFINED_RESEARCH_ONLY, (_leg(SELL, PUT, .50, 0), _leg(SELL, CALL, .50, 0))),
    _template("LONG_STRANGLE", _VOLATILITY, "LONG_VOL", EntryKind.DEBIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .75, 0), _leg(BUY, CALL, .25, 1))),
    _template("SHORT_STRANGLE", _VOLATILITY, "SHORT_VOL", EntryKind.CREDIT, RiskClass.UNDEFINED_RESEARCH_ONLY, (_leg(SELL, PUT, .75, 0), _leg(SELL, CALL, .25, 1))),
    _template("IRON_FLY", _VOLATILITY, "SHORT_VOL", EntryKind.CREDIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .75, 0), _leg(SELL, PUT, .50, 1), _leg(SELL, CALL, .50, 1), _leg(BUY, CALL, .25, 2)), StrikeGeometry.SYMMETRIC_THREE),
    _template("IRON_CONDOR", _VOLATILITY, "SHORT_VOL", EntryKind.CREDIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .85, 0), _leg(SELL, PUT, .70, 1), _leg(SELL, CALL, .30, 2), _leg(BUY, CALL, .15, 3))),
    _template("CALL_CALENDAR", _TERM, "FRONT_RICH", EntryKind.DEBIT, RiskClass.FINITE_TERM_DEBIT, (_leg(SELL, CALL, .50, 0, expiry=FRONT), _leg(BUY, CALL, .50, 0, expiry=BACK))),
    _template("PUT_CALENDAR", _TERM, "FRONT_RICH", EntryKind.DEBIT, RiskClass.FINITE_TERM_DEBIT, (_leg(SELL, PUT, .50, 0, expiry=FRONT), _leg(BUY, PUT, .50, 0, expiry=BACK))),
    _template("CALL_BUTTERFLY", _SKEW, "CALL_CONVEXITY", EntryKind.DEBIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, CALL, .70, 0), _leg(SELL, CALL, .50, 1, ratio=2), _leg(BUY, CALL, .30, 2)), StrikeGeometry.SYMMETRIC_THREE),
    _template("PUT_BUTTERFLY", _SKEW, "PUT_CONVEXITY", EntryKind.DEBIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .70, 0), _leg(SELL, PUT, .50, 1, ratio=2), _leg(BUY, PUT, .30, 2)), StrikeGeometry.SYMMETRIC_THREE),
    _template("BROKEN_WING_CALL_BUTTERFLY", _SKEW, "CALL_SKEW", EntryKind.EITHER, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, CALL, .70, 0), _leg(SELL, CALL, .50, 1, ratio=2), _leg(BUY, CALL, .20, 2)), StrikeGeometry.ASYMMETRIC_THREE),
    _template("BROKEN_WING_PUT_BUTTERFLY", _SKEW, "PUT_SKEW", EntryKind.EITHER, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .80, 0), _leg(SELL, PUT, .50, 1, ratio=2), _leg(BUY, PUT, .30, 2)), StrikeGeometry.ASYMMETRIC_THREE),
    _template("CALL_RATIO", _SKEW, "CALL_SKEW", EntryKind.EITHER, RiskClass.UNDEFINED_RESEARCH_ONLY, (_leg(BUY, CALL, .55, 0), _leg(SELL, CALL, .30, 1, ratio=2))),
    _template("PUT_RATIO", _SKEW, "PUT_SKEW", EntryKind.EITHER, RiskClass.UNDEFINED_RESEARCH_ONLY, (_leg(SELL, PUT, .70, 0, ratio=2), _leg(BUY, PUT, .45, 1))),
    _template("CALL_BACKSPREAD", _SKEW, "CALL_CONVEXITY", EntryKind.EITHER, RiskClass.FINITE_SAME_EXPIRY, (_leg(SELL, CALL, .55, 0), _leg(BUY, CALL, .30, 1, ratio=2))),
    _template("PUT_BACKSPREAD", _SKEW, "PUT_CONVEXITY", EntryKind.EITHER, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .70, 0, ratio=2), _leg(SELL, PUT, .45, 1))),
    _template("NAKED_CALL", _PREMIUM, "CALL_PREMIUM_RICH", EntryKind.CREDIT, RiskClass.UNDEFINED_RESEARCH_ONLY, (_leg(SELL, CALL, .25, 0),)),
    _template("NAKED_PUT", _PREMIUM, "PUT_PREMIUM_RICH", EntryKind.CREDIT, RiskClass.UNDEFINED_RESEARCH_ONLY, (_leg(SELL, PUT, .75, 0),)),
    _template("WING_CAPPED_SHORT_STRADDLE", _PREMIUM, "SHORT_VOL", EntryKind.CREDIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .75, 0), _leg(SELL, PUT, .50, 1), _leg(SELL, CALL, .50, 1), _leg(BUY, CALL, .25, 2)), StrikeGeometry.SYMMETRIC_THREE),
    _template("WING_CAPPED_SHORT_STRANGLE", _PREMIUM, "SHORT_VOL", EntryKind.CREDIT, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .85, 0), _leg(SELL, PUT, .70, 1), _leg(SELL, CALL, .30, 2), _leg(BUY, CALL, .15, 3))),
    _template("WING_CAPPED_CALL_RATIO", _PREMIUM, "CALL_PREMIUM_RICH", EntryKind.EITHER, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, CALL, .55, 0), _leg(SELL, CALL, .30, 1, ratio=2), _leg(BUY, CALL, .15, 2)), StrikeGeometry.ASYMMETRIC_THREE),
    _template("WING_CAPPED_PUT_RATIO", _PREMIUM, "PUT_PREMIUM_RICH", EntryKind.EITHER, RiskClass.FINITE_SAME_EXPIRY, (_leg(BUY, PUT, .85, 0), _leg(SELL, PUT, .70, 1, ratio=2), _leg(BUY, PUT, .45, 2)), StrikeGeometry.ASYMMETRIC_THREE),
)


_TEMPLATES: Dict[str, StructureTemplate] = {
    item.strategy_id: item for item in FROZEN_STRUCTURE_TEMPLATES
}
if len(_TEMPLATES) != len(FROZEN_STRUCTURE_TEMPLATES):
    raise RuntimeError("duplicate frozen Cultra structure template")
if set(_TEMPLATES) != {item.strategy_id for item in FROZEN_STRATEGY_CATALOG}:
    raise RuntimeError("frozen structure templates do not cover the complete catalog")


def get_structure_template(strategy_id: str) -> StructureTemplate:
    try:
        return _TEMPLATES[str(strategy_id)]
    except KeyError as exc:
        raise StructureError("strategy has no frozen structure template") from exc


def structure_template_registry_hash() -> str:
    payload = {
        "schema": "cultra.structure-template-registry.v1",
        "version": STRUCTURE_TEMPLATE_VERSION,
        "templates": [
            {"strategy_id": item.strategy_id, "template_hash": item.template_hash}
            for item in FROZEN_STRUCTURE_TEMPLATES
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


STRUCTURE_TEMPLATE_REGISTRY_HASH = structure_template_registry_hash()


@dataclass(frozen=True)
class ContractQuote:
    ticker: str
    trade_date: date
    expiration: date
    dte: int
    strike: float
    call_delta: float
    call_bid: Optional[float]
    call_ask: Optional[float]
    put_bid: Optional[float]
    put_ask: Optional[float]
    call_open_interest: Optional[int]
    put_open_interest: Optional[int]
    observed_at: datetime
    snapshot_id: str
    stock_price: Optional[float] = None

    def __post_init__(self) -> None:
        ticker = str(self.ticker).strip().upper()
        if not ticker or len(ticker) > 6:
            raise StructureError("contract ticker is invalid")
        object.__setattr__(self, "ticker", ticker)
        if self.expiration <= self.trade_date:
            raise StructureError("contract expiration must follow the trade date")
        if isinstance(self.dte, bool) or not isinstance(self.dte, int) or self.dte <= 0:
            raise StructureError("contract DTE must be a positive integer")
        for name in ("strike", "call_delta"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise StructureError("contract %s must be finite" % name)
            object.__setattr__(self, name, value)
        if self.strike <= 0.0 or not 0.0 <= self.call_delta <= 1.0:
            raise StructureError("contract strike or delta is invalid")
        for name in ("call_bid", "call_ask", "put_bid", "put_ask"):
            value = getattr(self, name)
            if value is None:
                continue
            converted = float(value)
            if not math.isfinite(converted) or converted < 0.0:
                raise StructureError("contract quote side is invalid")
            object.__setattr__(self, name, converted)
        for bid_name, ask_name in (("call_bid", "call_ask"), ("put_bid", "put_ask")):
            bid, ask = getattr(self, bid_name), getattr(self, ask_name)
            if bid is not None and ask is not None and ask < bid:
                raise StructureError("contract ask cannot be below bid")
        for name in ("call_open_interest", "put_open_interest"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise StructureError("contract open interest is invalid")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise StructureError("contract observation timestamp must be timezone-aware")
        if not str(self.snapshot_id).strip():
            raise StructureError("contract snapshot id is required")
        if self.stock_price is not None:
            converted_stock_price = float(self.stock_price)
            if not math.isfinite(converted_stock_price) or converted_stock_price <= 0.0:
                raise StructureError("contract stock price is invalid")
            object.__setattr__(self, "stock_price", converted_stock_price)

    def bid_ask(self, option_type: OptionType) -> Tuple[Optional[float], Optional[float]]:
        if option_type is OptionType.CALL:
            return self.call_bid, self.call_ask
        return self.put_bid, self.put_ask

    def open_interest(self, option_type: OptionType) -> Optional[int]:
        return (
            self.call_open_interest
            if option_type is OptionType.CALL
            else self.put_open_interest
        )


@dataclass(frozen=True)
class SelectedStructure:
    hypothesis_id: str
    strategy_id: str
    holding_sessions: int
    template_hash: str
    legs: Tuple[OptionLeg, ...]
    entry_quotes: Tuple[LegQuote, ...]
    entry_snapshot_ids: Tuple[str, ...]
    target_call_deltas: Tuple[float, ...]
    signed_entry_debit: float

    @property
    def price_convention(self) -> PriceConvention:
        return PriceConvention.DEBIT if self.signed_entry_debit > 0.0 else PriceConvention.CREDIT


@dataclass(frozen=True)
class TermDebitRiskEnvelope:
    price_convention: PriceConvention
    executable_price: float
    signed_entry_debit: float
    maximum_profit: Optional[float]
    maximum_loss: float
    breakevens: Tuple[float, ...] = ()
    calculation_version: str = "cultra.term-debit-risk-bound.v1"


@dataclass(frozen=True)
class HistoricalStructureOutcome:
    hypothesis_id: str
    strategy_id: str
    exit_date: date
    holding_sessions: int
    exit_reason: str
    gross_pnl: float
    net_pnl: float
    risk_reference: float
    maximum_loss: Optional[float]
    target_pnl: float
    stop_pnl: float
    target_hit: bool
    stop_hit: bool
    max_loss_hit: bool
    exact_path_observations: int


@dataclass(frozen=True)
class _GroupChoice:
    strike: float
    contracts: Tuple[ContractQuote, ...]
    score: float


def _occ_symbol(ticker: str, expiration: date, option_type: OptionType, strike: float) -> str:
    scaled = int(round(float(strike) * 1000.0))
    if scaled <= 0 or scaled > 99_999_999:
        raise StructureError("selected strike cannot form an OCC symbol")
    kind = "C" if option_type is OptionType.CALL else "P"
    return "%s%s%s%08d" % (ticker, expiration.strftime("%y%m%d"), kind, scaled)


def _expiry_assignments(
    template: StructureTemplate,
    holding_sessions: int,
    contracts: Sequence[ContractQuote],
    required_path_end: Optional[date] = None,
) -> Tuple[Tuple[float, Mapping[ExpiryRole, date]], ...]:
    by_expiry: Dict[date, int] = {}
    for item in contracts:
        prior = by_expiry.setdefault(item.expiration, item.dte)
        if prior != item.dte:
            raise StructureError("one expiration has inconsistent DTE values")
    minimum_dte = int(math.ceil(holding_sessions * 7.0 / 5.0)) + 7
    front_target = minimum_dte + 14
    back_target = front_target + 42

    def path_covered(expiration: date) -> bool:
        return required_path_end is None or (expiration - required_path_end).days >= 20

    if {item.expiry_role for item in template.legs} == {ExpiryRole.SAME}:
        values = [
            (abs(dte - front_target), {ExpiryRole.SAME: expiration})
            for expiration, dte in by_expiry.items()
            if minimum_dte <= dte <= 180 and path_covered(expiration)
        ]
    else:
        values = []
        for front, front_dte in by_expiry.items():
            if not minimum_dte <= front_dte <= 180 or not path_covered(front):
                continue
            for back, back_dte in by_expiry.items():
                if (
                    back <= front
                    or back_dte <= front_dte
                    or back_dte > 180
                    or not path_covered(back)
                ):
                    continue
                score = abs(front_dte - front_target) + abs(back_dte - back_target)
                values.append(
                    (score, {ExpiryRole.FRONT: front, ExpiryRole.BACK: back})
                )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item[0],
                tuple((role.value, expiry.isoformat()) for role, expiry in sorted(item[1].items(), key=lambda pair: pair[0].value)),
            ),
        )[:12]
    )


def _executable(contract: ContractQuote, rule: FrozenLegRule) -> bool:
    bid, ask = contract.bid_ask(rule.option_type)
    interest = contract.open_interest(rule.option_type)
    if bid is None or ask is None or ask < bid or ask <= 0.0:
        return False
    midpoint = (bid + ask) / 2.0
    if midpoint <= 0.0 or (ask - bid) / midpoint > MAXIMUM_RELATIVE_SPREAD:
        return False
    if rule.action is LegAction.SELL and bid <= 0.0:
        return False
    return interest is not None and interest >= MINIMUM_OPEN_INTEREST


def _group_choices(
    rules: Sequence[FrozenLegRule],
    expiries: Mapping[ExpiryRole, date],
    contract_map: Mapping[Tuple[date, float], ContractQuote],
) -> Tuple[_GroupChoice, ...]:
    strike_sets = []
    for rule in rules:
        expiration = expiries[rule.expiry_role]
        strike_sets.append(
            {
                strike
                for (expiry, strike), contract in contract_map.items()
                if expiry == expiration and _executable(contract, rule)
            }
        )
    common = set.intersection(*strike_sets) if strike_sets else set()
    choices = []
    for strike in common:
        selected = tuple(
            contract_map[(expiries[rule.expiry_role], strike)] for rule in rules
        )
        score = math.fsum(
            abs(contract.call_delta - rule.target_call_delta)
            for contract, rule in zip(selected, rules)
        )
        choices.append(_GroupChoice(float(strike), selected, score))
    return tuple(sorted(choices, key=lambda item: (item.score, item.strike))[:MAXIMUM_GROUP_CHOICES])


def _geometry_ok(template: StructureTemplate, strikes: Sequence[float]) -> bool:
    if any(not left < right for left, right in zip(strikes[:-1], strikes[1:])):
        return False
    if template.geometry is StrikeGeometry.ORDERED:
        return True
    low_width = strikes[1] - strikes[0]
    high_width = strikes[2] - strikes[1]
    symmetric = math.isclose(low_width, high_width, rel_tol=0.0, abs_tol=1e-7)
    if template.geometry is StrikeGeometry.SYMMETRIC_THREE:
        return symmetric
    return not symmetric


def _candidate_selection(
    *,
    hypothesis_id: str,
    holding_sessions: int,
    template: StructureTemplate,
    expiry_score: float,
    expiries: Mapping[ExpiryRole, date],
    contract_map: Mapping[Tuple[date, float], ContractQuote],
) -> Optional[Tuple[float, SelectedStructure]]:
    groups = tuple(sorted({item.strike_group for item in template.legs}))
    choices = []
    for group in groups:
        rules = tuple(item for item in template.legs if item.strike_group == group)
        current = _group_choices(rules, expiries, contract_map)
        if not current:
            return None
        choices.append(current)
    best: Optional[Tuple[Any, SelectedStructure]] = None
    for combination in itertools.product(*choices):
        strikes = tuple(item.strike for item in combination)
        if not _geometry_ok(template, strikes):
            continue
        by_group = {group: choice for group, choice in zip(groups, combination)}
        selected_contracts = []
        legs = []
        quotes = []
        snapshots = []
        deltas = []
        group_offsets: Dict[int, int] = {group: 0 for group in groups}
        for rule in template.legs:
            choice = by_group[rule.strike_group]
            index = group_offsets[rule.strike_group]
            group_offsets[rule.strike_group] += 1
            contract = choice.contracts[index]
            bid, ask = contract.bid_ask(rule.option_type)
            assert bid is not None and ask is not None
            leg = OptionLeg(
                occ_symbol=_occ_symbol(
                    contract.ticker, contract.expiration, rule.option_type, contract.strike
                ),
                action=rule.action,
                option_type=rule.option_type,
                expiration=contract.expiration,
                strike=contract.strike,
                ratio=rule.ratio,
            )
            selected_contracts.append(contract)
            legs.append(leg)
            quotes.append(LegQuote(leg.occ_symbol, bid, ask, contract.observed_at))
            snapshots.append(contract.snapshot_id)
            deltas.append(rule.target_call_delta)
        if len({item.occ_symbol for item in legs}) != len(legs):
            continue
        try:
            signed_debit = executable_entry_debit(tuple(legs), tuple(quotes))
        except ValueError:
            continue
        if template.entry_kind is EntryKind.DEBIT and signed_debit <= 0.0:
            continue
        if template.entry_kind is EntryKind.CREDIT and signed_debit >= 0.0:
            continue
        score = expiry_score + math.fsum(item.score for item in combination)
        selection = SelectedStructure(
            hypothesis_id=hypothesis_id,
            strategy_id=template.strategy_id,
            holding_sessions=holding_sessions,
            template_hash=template.template_hash,
            legs=tuple(legs),
            entry_quotes=tuple(quotes),
            entry_snapshot_ids=tuple(snapshots),
            target_call_deltas=tuple(deltas),
            signed_entry_debit=signed_debit,
        )
        identity = (
            score,
            tuple(item.expiration for item in legs),
            tuple(item.strike for item in legs),
            tuple(item.occ_symbol for item in legs),
        )
        if best is None or identity < best[0]:
            best = (identity, selection)
    return None if best is None else (float(best[0][0]), best[1])


def select_frozen_structure(
    *,
    hypothesis_id: str,
    strategy_id: str,
    holding_sessions: int,
    contracts: Sequence[ContractQuote],
    required_path_end: Optional[date] = None,
) -> SelectedStructure:
    """Select exactly one deterministic structure from an entry-time chain."""

    if holding_sessions not in (20, 40, 60):
        raise StructureError("holding horizon is not in the frozen registry")
    if not contracts:
        raise StructureError("entry chain is empty")
    tickers = {item.ticker for item in contracts}
    trade_dates = {item.trade_date for item in contracts}
    if len(tickers) != 1 or len(trade_dates) != 1:
        raise StructureError("entry chain must contain one ticker and one trade date")
    trade_date = next(iter(trade_dates))
    if required_path_end is not None and required_path_end <= trade_date:
        raise StructureError("required path end must follow the entry trade date")
    contract_map: Dict[Tuple[date, float], ContractQuote] = {}
    for item in contracts:
        key = (item.expiration, item.strike)
        if key in contract_map:
            raise StructureError("entry chain contains a duplicate expiration/strike row")
        contract_map[key] = item
    template = get_structure_template(strategy_id)
    best: Optional[Tuple[Any, SelectedStructure]] = None
    for expiry_score, expiries in _expiry_assignments(
        template, holding_sessions, contracts, required_path_end
    ):
        candidate = _candidate_selection(
            hypothesis_id=hypothesis_id,
            holding_sessions=holding_sessions,
            template=template,
            expiry_score=expiry_score,
            expiries=expiries,
            contract_map=contract_map,
        )
        if candidate is None:
            continue
        score, selection = candidate
        identity = (
            score,
            tuple(item.expiration for item in selection.legs),
            tuple(item.strike for item in selection.legs),
        )
        if best is None or identity < best[0]:
            best = (identity, selection)
    if best is None:
        raise StructureError("no chain combination satisfies the frozen structure geometry")
    return best[1]


def structure_risk_envelope(
    selection: SelectedStructure,
    costs: CostBreakdown,
) -> Optional[Any]:
    """Return a finite risk proof, or None for research-only undefined risk."""

    template = get_structure_template(selection.strategy_id)
    if selection.template_hash != template.template_hash:
        raise StructureError("selected structure template hash has drifted")
    if template.risk_class is RiskClass.UNDEFINED_RESEARCH_ONLY:
        return None
    if template.risk_class is RiskClass.FINITE_SAME_EXPIRY:
        try:
            return same_expiry_payoff_envelope(selection.legs, selection.entry_quotes, costs)
        except ValueError as exc:
            raise StructureError("same-expiry risk proof failed") from exc
    if selection.signed_entry_debit <= 0.0:
        raise StructureError("term structure does not form the frozen debit")
    maximum_loss = selection.signed_entry_debit * CONTRACT_MULTIPLIER + costs.total
    if not math.isfinite(maximum_loss) or maximum_loss <= 0.0:
        raise StructureError("term structure maximum loss is invalid")
    return TermDebitRiskEnvelope(
        price_convention=PriceConvention.DEBIT,
        executable_price=selection.signed_entry_debit,
        signed_entry_debit=selection.signed_entry_debit,
        maximum_profit=None,
        maximum_loss=maximum_loss,
    )


def mark_to_market_net_pnl(
    selection: SelectedStructure,
    exit_quotes: Sequence[LegQuote],
    costs: CostBreakdown,
) -> Tuple[float, float]:
    """Liquidate every frozen leg at bid/ask and return gross and net dollars."""

    quote_map = exact_quote_map(tuple(exit_quotes))
    if set(quote_map) != {item.occ_symbol for item in selection.legs}:
        raise StructureError("exit quotes do not match every frozen exact leg")
    liquidation = 0.0
    for leg in selection.legs:
        quote = quote_map[leg.occ_symbol]
        if leg.action is LegAction.BUY:
            liquidation += quote.bid * leg.ratio
        else:
            liquidation -= quote.ask * leg.ratio
    gross = (liquidation - selection.signed_entry_debit) * CONTRACT_MULTIPLIER
    net = gross - costs.total
    if not math.isfinite(gross) or not math.isfinite(net):
        raise StructureError("historical mark produced non-finite P/L")
    return gross, net


def resolve_historical_structure_path(
    selection: SelectedStructure,
    path: Sequence[Tuple[date, Sequence[LegQuote]]],
    costs: CostBreakdown,
) -> HistoricalStructureOutcome:
    """Resolve stop, target, then time exit on exact saved contracts.

    Daily observations are close-to-close marks.  If a future data source ever
    supplies both an intraday target and stop event for one session, it must be
    resolved upstream as STOP_FIRST before calling this function.
    """

    if len(path) < selection.holding_sessions:
        raise StructureError("historical exact-contract path is incomplete")
    selected_path = tuple(path[: selection.holding_sessions])
    dates = tuple(item[0] for item in selected_path)
    if dates != tuple(sorted(set(dates))):
        raise StructureError("historical path dates must be sorted and unique")
    risk = structure_risk_envelope(selection, costs)
    template = get_structure_template(selection.strategy_id)
    if risk is None:
        entry_dollars = abs(selection.signed_entry_debit) * CONTRACT_MULTIPLIER
        risk_reference = max(1.0, entry_dollars)
        maximum_loss = None
        target_pnl = risk_reference * template.target_fraction_of_risk
        stop_pnl = -risk_reference * template.research_only_stop_multiple
    else:
        risk_reference = float(risk.maximum_loss)
        maximum_loss = risk_reference
        target_pnl = risk_reference * template.target_fraction_of_risk
        stop_pnl = -risk_reference * template.stop_fraction_of_risk
    chosen: Optional[Tuple[date, int, float, float, str]] = None
    for offset, (session_date, quotes) in enumerate(selected_path, 1):
        gross, net = mark_to_market_net_pnl(selection, quotes, costs)
        if net <= stop_pnl:
            chosen = (session_date, offset, gross, net, "STOP")
            break
        if net >= target_pnl:
            chosen = (session_date, offset, gross, net, "TARGET")
            break
        if offset == selection.holding_sessions:
            chosen = (session_date, offset, gross, net, "TIME")
    if chosen is None:
        raise StructureError("historical path did not resolve under the frozen exit policy")
    exit_date, holding, gross, net, reason = chosen
    return HistoricalStructureOutcome(
        hypothesis_id=selection.hypothesis_id,
        strategy_id=selection.strategy_id,
        exit_date=exit_date,
        holding_sessions=holding,
        exit_reason=reason,
        gross_pnl=gross,
        net_pnl=net,
        risk_reference=risk_reference,
        maximum_loss=maximum_loss,
        target_pnl=target_pnl,
        stop_pnl=stop_pnl,
        target_hit=reason == "TARGET",
        stop_hit=reason == "STOP",
        max_loss_hit=(
            maximum_loss is not None and net <= -0.95 * maximum_loss
        ),
        exact_path_observations=holding * len(selection.legs),
    )


__all__ = [
    "ContractQuote",
    "EntryKind",
    "ExpiryRole",
    "FROZEN_STRUCTURE_TEMPLATES",
    "FrozenLegRule",
    "HistoricalStructureOutcome",
    "RiskClass",
    "STRUCTURE_TEMPLATE_REGISTRY_HASH",
    "STRUCTURE_TEMPLATE_VERSION",
    "SelectedStructure",
    "StrikeGeometry",
    "StructureError",
    "StructureTemplate",
    "TermDebitRiskEnvelope",
    "get_structure_template",
    "mark_to_market_net_pnl",
    "resolve_historical_structure_path",
    "select_frozen_structure",
    "structure_risk_envelope",
    "structure_template_registry_hash",
]
