"""Frozen, finite strategy-hypothesis catalog for Cultra.

The catalog is intentionally declarative.  Adding an entry changes the frozen
hypothesis family and therefore requires a new catalog version before holdout
results are inspected.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterator, Tuple


CATALOG_VERSION = "cultra-options-catalog-v1"


class StrategyCategory(str, Enum):
    DIRECTIONAL = "DIRECTIONAL"
    VOLATILITY = "VOLATILITY"
    TERM_STRUCTURE = "TERM_STRUCTURE"
    SKEW_CONVEXITY = "SKEW_CONVEXITY"
    PREMIUM_SELLING = "PREMIUM_SELLING"


@dataclass(frozen=True)
class StrategyDefinition:
    strategy_id: str
    display_name: str
    category: StrategyCategory
    leg_count: int
    canonical_pattern: str
    defined_risk_by_construction: bool
    ticket_eligible_structure: bool
    protective_wings_required: bool
    holding_sessions_min: int = 20
    holding_sessions_max: int = 60

    def __post_init__(self) -> None:
        if not self.strategy_id or self.strategy_id != self.strategy_id.upper():
            raise ValueError("strategy_id must be non-empty uppercase text")
        if not self.display_name or not self.canonical_pattern:
            raise ValueError("display_name and canonical_pattern are required")
        if self.leg_count <= 0:
            raise ValueError("leg_count must be positive")
        if not 20 <= self.holding_sessions_min <= self.holding_sessions_max <= 60:
            raise ValueError("holding window must be contained within 20-60 sessions")
        if self.ticket_eligible_structure and not self.defined_risk_by_construction:
            raise ValueError("ticket-eligible catalog entries must be defined risk")
        if self.defined_risk_by_construction and self.protective_wings_required:
            raise ValueError("defined-risk entries do not require added wings")


def _strategy(
    strategy_id: str,
    name: str,
    category: StrategyCategory,
    legs: int,
    pattern: str,
    defined: bool,
    promotable: bool,
    wings: bool = False,
) -> StrategyDefinition:
    return StrategyDefinition(
        strategy_id=strategy_id,
        display_name=name,
        category=category,
        leg_count=legs,
        canonical_pattern=pattern,
        defined_risk_by_construction=defined,
        ticket_eligible_structure=promotable,
        protective_wings_required=wings,
    )


FROZEN_STRATEGY_CATALOG: Tuple[StrategyDefinition, ...] = (
    # Directional
    _strategy("LONG_CALL", "Long call", StrategyCategory.DIRECTIONAL, 1, "buy call", True, True),
    _strategy("LONG_PUT", "Long put", StrategyCategory.DIRECTIONAL, 1, "buy put", True, True),
    _strategy("CALL_DEBIT_VERTICAL", "Call debit vertical", StrategyCategory.DIRECTIONAL, 2, "buy lower-strike call / sell higher-strike call", True, True),
    _strategy("PUT_DEBIT_VERTICAL", "Put debit vertical", StrategyCategory.DIRECTIONAL, 2, "buy higher-strike put / sell lower-strike put", True, True),
    _strategy("CALL_CREDIT_VERTICAL", "Call credit vertical", StrategyCategory.DIRECTIONAL, 2, "sell lower-strike call / buy higher-strike call", True, True),
    _strategy("PUT_CREDIT_VERTICAL", "Put credit vertical", StrategyCategory.DIRECTIONAL, 2, "sell higher-strike put / buy lower-strike put", True, True),
    _strategy("CALL_DIAGONAL", "Call diagonal", StrategyCategory.DIRECTIONAL, 2, "buy later call / sell earlier call", True, True),
    _strategy("PUT_DIAGONAL", "Put diagonal", StrategyCategory.DIRECTIONAL, 2, "buy later put / sell earlier put", True, True),
    # Volatility
    _strategy("LONG_STRADDLE", "Long straddle", StrategyCategory.VOLATILITY, 2, "buy same-strike call and put", True, True),
    _strategy("SHORT_STRADDLE", "Short straddle", StrategyCategory.VOLATILITY, 2, "sell same-strike call and put", False, False, True),
    _strategy("LONG_STRANGLE", "Long strangle", StrategyCategory.VOLATILITY, 2, "buy OTM call and put", True, True),
    _strategy("SHORT_STRANGLE", "Short strangle", StrategyCategory.VOLATILITY, 2, "sell OTM call and put", False, False, True),
    _strategy("IRON_FLY", "Iron fly", StrategyCategory.VOLATILITY, 4, "long wings / short same-strike call and put", True, True),
    _strategy("IRON_CONDOR", "Iron condor", StrategyCategory.VOLATILITY, 4, "bull put spread plus bear call spread", True, True),
    # Term structure. Diagonals above are not duplicated as new hypotheses.
    _strategy("CALL_CALENDAR", "Call calendar", StrategyCategory.TERM_STRUCTURE, 2, "buy later call / sell same-strike earlier call", True, True),
    _strategy("PUT_CALENDAR", "Put calendar", StrategyCategory.TERM_STRUCTURE, 2, "buy later put / sell same-strike earlier put", True, True),
    # Skew and convexity
    _strategy("CALL_BUTTERFLY", "Call butterfly", StrategyCategory.SKEW_CONVEXITY, 3, "buy 1 low / sell 2 middle / buy 1 high calls", True, True),
    _strategy("PUT_BUTTERFLY", "Put butterfly", StrategyCategory.SKEW_CONVEXITY, 3, "buy 1 high / sell 2 middle / buy 1 low puts", True, True),
    _strategy("BROKEN_WING_CALL_BUTTERFLY", "Broken-wing call butterfly", StrategyCategory.SKEW_CONVEXITY, 3, "asymmetric call butterfly", True, True),
    _strategy("BROKEN_WING_PUT_BUTTERFLY", "Broken-wing put butterfly", StrategyCategory.SKEW_CONVEXITY, 3, "asymmetric put butterfly", True, True),
    _strategy("CALL_RATIO", "Call ratio spread", StrategyCategory.SKEW_CONVEXITY, 2, "buy fewer calls / sell more calls", False, False, True),
    _strategy("PUT_RATIO", "Put ratio spread", StrategyCategory.SKEW_CONVEXITY, 2, "buy fewer puts / sell more puts", False, False, True),
    _strategy("CALL_BACKSPREAD", "Call backspread", StrategyCategory.SKEW_CONVEXITY, 2, "sell fewer calls / buy more higher-strike calls", True, True),
    _strategy("PUT_BACKSPREAD", "Put backspread", StrategyCategory.SKEW_CONVEXITY, 2, "sell fewer puts / buy more lower-strike puts", True, True),
    # Premium-selling research structures and their defined-risk counterparts.
    _strategy("NAKED_CALL", "Naked call", StrategyCategory.PREMIUM_SELLING, 1, "sell uncovered call", False, False, True),
    _strategy("NAKED_PUT", "Naked put", StrategyCategory.PREMIUM_SELLING, 1, "sell uncovered put", False, False, True),
    _strategy("WING_CAPPED_SHORT_STRADDLE", "Wing-capped short straddle", StrategyCategory.PREMIUM_SELLING, 4, "short straddle plus long call and put wings", True, True),
    _strategy("WING_CAPPED_SHORT_STRANGLE", "Wing-capped short strangle", StrategyCategory.PREMIUM_SELLING, 4, "short strangle plus long call and put wings", True, True),
    _strategy("WING_CAPPED_CALL_RATIO", "Wing-capped call ratio", StrategyCategory.PREMIUM_SELLING, 3, "call ratio plus farther long call wing", True, True),
    _strategy("WING_CAPPED_PUT_RATIO", "Wing-capped put ratio", StrategyCategory.PREMIUM_SELLING, 3, "put ratio plus farther long put wing", True, True),
)


_BY_ID: Dict[str, StrategyDefinition] = {
    definition.strategy_id: definition for definition in FROZEN_STRATEGY_CATALOG
}
if len(_BY_ID) != len(FROZEN_STRATEGY_CATALOG):
    raise RuntimeError("duplicate strategy_id in frozen Cultra catalog")


def get_strategy(strategy_id: str) -> StrategyDefinition:
    """Return a frozen definition or fail rather than inventing a hypothesis."""

    try:
        return _BY_ID[strategy_id]
    except KeyError:
        raise KeyError("strategy is not in %s: %s" % (CATALOG_VERSION, strategy_id))


def iter_ticket_eligible() -> Iterator[StrategyDefinition]:
    return (item for item in FROZEN_STRATEGY_CATALOG if item.ticket_eligible_structure)


def iter_research_only() -> Iterator[StrategyDefinition]:
    return (item for item in FROZEN_STRATEGY_CATALOG if not item.ticket_eligible_structure)

