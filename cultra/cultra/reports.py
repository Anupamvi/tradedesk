"""Human-readable daily board rendering for Cultra.

The board is a view over immutable evidence.  It never invents POP, edge, or
missing values, and it never truncates an eligible set to a top-N list.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence


BOARD_SCHEMA = "cultra.daily-board.v1"


class ReportError(ValueError):
    """Raised when board input would be misleading or ambiguous."""


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            item.name: _plain(getattr(value, item.name))
            for item in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    converted = _plain(value)
    if not isinstance(converted, Mapping):
        raise ReportError("board entries must be dataclasses or mappings")
    return converted


def _lookup(value: Any, *paths: str) -> Any:
    root = _plain(value)
    for path in paths:
        current = root
        found = True
        for part in path.split("."):
            if not isinstance(current, Mapping) or part not in current:
                found = False
                break
            current = current[part]
        if found and current is not None:
            return current
    return None


def _escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def _number(value: Any, *, percent: bool = False, money: bool = False) -> str:
    if value is None:
        return "MISSING"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _escape(value)
    if not math.isfinite(number):
        return "MISSING"
    if percent:
        return "%.1f%%" % (number * 100.0)
    if money:
        return "$%.2f" % number
    return "%.4f" % number


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: str
    symbol: str
    strategy_family: str
    reason: str
    disposition: str
    rank_score: Optional[float] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("candidate_id", "symbol", "strategy_family", "reason", "disposition"):
            if not str(getattr(self, name)).strip():
                raise ReportError("%s is required" % name)
        if self.rank_score is not None and not math.isfinite(self.rank_score):
            raise ReportError("rank_score must be finite")


@dataclass(frozen=True)
class DailyBoardData:
    as_of: date
    run_id: str
    overall_status: str = "UNPROVEN"
    strategy_states: Mapping[str, str] = field(default_factory=dict)
    strategy_rejection_reasons: Mapping[str, str] = field(default_factory=dict)
    tickets: Sequence[Any] = ()
    watchlist: Sequence[Any] = ()
    rejected: Sequence[Any] = ()
    data_unavailable: Sequence[Any] = ()
    budget_unresolved: Sequence[Any] = ()
    generated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ReportError("run_id is required")
        if not self.overall_status.strip():
            raise ReportError("overall_status is required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": BOARD_SCHEMA,
            "as_of": self.as_of.isoformat(),
            "run_id": self.run_id,
            "overall_status": self.overall_status,
            "strategy_states": dict(self.strategy_states),
            "strategy_rejection_reasons": dict(self.strategy_rejection_reasons),
            "tickets": [_plain(item) for item in self.tickets],
            "watchlist": [_plain(item) for item in self.watchlist],
            "rejected": [_plain(item) for item in self.rejected],
            "data_unavailable": [_plain(item) for item in self.data_unavailable],
            "budget_unresolved": [_plain(item) for item in self.budget_unresolved],
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


def _strategy_sections(
    states: Mapping[str, str], rejection_reasons: Optional[Mapping[str, str]] = None
) -> str:
    normalized = {name: getattr(state, "value", state) for name, state in states.items()}
    reasons = dict(rejection_reasons or {})
    groups = {
        "Enabled": [],
        "Awaiting holdout": [],
        "Research/validation pending": [],
        "Rejected": [],
    }
    for name, raw_state in sorted(normalized.items()):
        state = str(raw_state)
        if state in {"HOLDOUT_PASS", "SHADOW_PASS", "MANUAL_TICKET_ENABLED"}:
            group = "Enabled"
        elif state in {"RESEARCH_PASS", "VALIDATION_PASS"}:
            group = "Awaiting holdout"
        elif state == "REJECTED":
            group = "Rejected"
        else:
            group = "Research/validation pending"
        label = "%s (`%s`)" % (_escape(name), _escape(state))
        if group == "Rejected" and reasons.get(name):
            label += ": %s" % _escape(reasons[name])
        groups[group].append(label)

    lines = ["## Strategy evidence", ""]
    for label, values in groups.items():
        lines.append("- %s: %s" % (label, ", ".join(values) if values else "None"))
    return "\n".join(lines)


def _render_ticket(ticket: Any, index: int) -> str:
    payload = _mapping(ticket)
    candidate_id = _lookup(payload, "candidate_id", "id") or "MISSING"
    symbol = _lookup(payload, "ticker", "symbol") or "MISSING"
    family = _lookup(payload, "strategy_family", "strategy_id") or "MISSING"
    hypothesis_id = _lookup(payload, "hypothesis_id") or "MISSING"
    state = _lookup(payload, "evidence_state", "family_evidence.state", "evidence.state")
    pop = _lookup(payload, "probabilities.pop_net.point", "pop.pop_net.point", "pop_net.point")
    pop_lower = _lookup(payload, "probabilities.pop_net.lower", "pop_net.lower")
    pop_upper = _lookup(payload, "probabilities.pop_net.upper", "pop_net.upper")
    target = _lookup(payload, "probabilities.p_target.point", "pop.p_target.point", "p_target.point")
    stop = _lookup(payload, "probabilities.p_stop.point", "pop.p_stop.point", "p_stop.point")
    max_loss_probability = _lookup(
        payload,
        "probabilities.p_max_loss.point",
        "pop.p_max_loss.point",
        "p_max_loss.point",
    )
    net_ev = _lookup(
        payload,
        "edge.net_expected_value",
        "edge.net_expected_profit",
        "edge.net_ev",
        "net_expected_profit",
    )
    conservative_ev = _lookup(
        payload,
        "edge.conservative_net_expected_value",
        "edge.conservative_expected_profit",
        "edge.conservative_ev",
        "conservative_expected_profit",
    )
    max_loss = _lookup(payload, "edge.maximum_loss", "edge.max_loss", "maximum_loss")
    max_profit = _lookup(payload, "edge.maximum_profit", "maximum_profit")
    target_pnl = _lookup(payload, "edge.target_pnl", "target_pnl")
    stop_pnl = _lookup(payload, "edge.stop_pnl", "stop_pnl")
    limit_price = _lookup(payload, "edge.executable_limit_price", "limit_price")
    price_convention = _lookup(payload, "edge.price_convention", "price_convention")
    quote_timestamp = _lookup(payload, "underlying_quote.timestamp")
    underlying_bid = _lookup(payload, "underlying_quote.bid")
    underlying_ask = _lookup(payload, "underlying_quote.ask")
    provider_trade_date = _lookup(payload, "provider_trade_date")
    snapshot_id = _lookup(payload, "orats_snapshot_id")
    analytical_fields = _lookup(payload, "analytical_fields")
    earnings_date = _lookup(payload, "event_evidence.earnings_date")
    dividend_dates = _lookup(payload, "event_evidence.dividend_dates")
    event_source = _lookup(payload, "event_evidence.source")
    event_status = _lookup(payload, "event_evidence.status")
    rank = _lookup(
        payload,
        "edge.conservative_return_on_max_loss",
        "edge.conservative_return_on_risk",
        "edge.conservative_ev_per_risk",
        "rank_score",
    )
    pop_sample = _lookup(payload, "probabilities.pop_net.sample_size")
    pop_model = _lookup(payload, "probabilities.pop_net.model_version")
    calibration_start = _lookup(payload, "probabilities.pop_net.calibration_start")
    calibration_end = _lookup(payload, "probabilities.pop_net.calibration_end")
    breakevens = _lookup(payload, "edge.breakevens")
    expected_shortfall = _lookup(payload, "edge.expected_shortfall")
    gap_loss = _lookup(payload, "edge.adverse_gap_stress_loss")
    thesis = _lookup(payload, "thesis")
    signal = _lookup(payload, "signal")
    entry_condition = _lookup(payload, "policy.entry_condition")
    target_rule = _lookup(payload, "policy.profit_target")
    stop_rule = _lookup(payload, "policy.stop_condition", "policy.stop_loss")
    time_exit = _lookup(payload, "policy.time_exit")
    invalidation = _lookup(payload, "policy.invalidation")
    assignment = _lookup(
        payload, "policy.assignment_handling", "policy.assignment_exercise"
    )
    next_review = _lookup(payload, "policy.next_review")
    evidence = _lookup(payload, "evidence", "family_evidence")
    model_features = _lookup(payload, "model_calculation.features")
    model_calculation_id = _lookup(payload, "model_calculation.calculation_id")
    model_calculation_version = _lookup(
        payload, "model_calculation.calculation_version"
    )
    selection_point_return = _lookup(
        payload, "model_calculation.selection_point_return_on_max_loss"
    )
    selection_conservative_return = _lookup(
        payload, "model_calculation.selection_conservative_return_on_max_loss"
    )

    lines = [
        "### 🟢 %d. %s — %s (`%s`)"
        % (index, _escape(symbol), _escape(family), _escape(candidate_id)),
        "",
        "**Order:** %s at a %s limit of **%s**. Quantity: **USER DETERMINED**."
        % (_plain_legs(payload), _escape(price_convention or "MISSING"), _number(limit_price, money=True)),
        "",
        "| Evidence | POP net (95% CI) | P target | P stop | P max loss | Net EV | Conservative EV | Max gain | Max loss | EV/risk |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| %s | %s [%s, %s] | %s | %s | %s | %s | %s | %s | %s | %s |"
        % (
            _escape(state or "MISSING"),
            _number(pop, percent=True),
            _number(pop_lower, percent=True),
            _number(pop_upper, percent=True),
            _number(target, percent=True),
            _number(stop, percent=True),
            _number(max_loss_probability, percent=True),
            _number(net_ev, money=True),
            _number(conservative_ev, money=True),
            _number(max_profit, money=True),
            _number(max_loss, money=True),
            _number(rank, percent=True),
        ),
        "",
        "- Schwab market: underlying %s / %s; quote time `%s`; legs: %s"
        % (
            _number(underlying_bid, money=True),
            _number(underlying_ask, money=True),
            _escape(quote_timestamp or "MISSING"),
            _plain_leg_markets(payload),
        ),
        "- Thesis / signal: %s / `%s`" % (_escape(thesis or "MISSING"), _escape(signal or "MISSING")),
        "- Frozen hypothesis: `%s`" % _escape(hypothesis_id),
        "- POP model: `%s`; n=%s; calibration %s through %s"
        % (
            _escape(pop_model or "MISSING"),
            _escape(pop_sample or "MISSING"),
            _escape(calibration_start or "MISSING"),
            _escape(calibration_end or "MISSING"),
        ),
        "- Current model inputs: %s" % _format_pairs(model_features),
        "- Current score: selection return %s point / %s conservative; calculation `%s` (`%s`)"
        % (
            _number(selection_point_return, percent=True),
            _number(selection_conservative_return, percent=True),
            _escape(model_calculation_id or "MISSING"),
            _escape(model_calculation_version or "MISSING"),
        ),
        "- Evidence windows: %s" % _evidence_windows(evidence),
        "- Economics: breakevens %s; target / stop %s / %s; expected shortfall %s; gap stress %s"
        % (
            _format_sequence(breakevens, money=True),
            _number(target_pnl, money=True),
            _number(stop_pnl, money=True),
            _number(expected_shortfall, money=True),
            _number(gap_loss, money=True),
        ),
        "- Entry / exits: %s; target %s; stop %s; time exit %s"
        % (
            _escape(entry_condition or "MISSING"),
            _escape(target_rule or "MISSING"),
            _escape(stop_rule or "MISSING"),
            _escape(time_exit or "MISSING"),
        ),
        "- Invalidation / assignment: %s / %s; next review %s"
        % (
            _escape(invalidation or "MISSING"),
            _escape(assignment or "MISSING"),
            _escape(next_review or "MISSING"),
        ),
        "- Events: earnings %s (%s); dividends %s; source %s"
        % (
            _escape(event_status or "MISSING"),
            _escape(earnings_date or "N/A"),
            _format_sequence(dividend_dates),
            _escape(event_source or "MISSING"),
        ),
        "- ORATS analytics: provider date `%s`; snapshot `%s`; fields %s"
        % (
            _escape(provider_trade_date or "MISSING"),
            _escape(snapshot_id or "MISSING"),
            _format_sequence(analytical_fields),
        ),
        "- Exact contract identifiers and the reproducible audit payload are in `manual_tickets.json`.",
    ]
    return "\n".join(lines)


def _format_sequence(value: Any, *, money: bool = False) -> str:
    if not isinstance(value, list) or not value:
        return "None" if isinstance(value, list) else "MISSING"
    return ", ".join(_number(item, money=True) if money else _escape(item) for item in value)


def _format_pairs(value: Any) -> str:
    if not isinstance(value, list) or not value:
        return "MISSING"
    rendered = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return "MISSING"
        rendered.append("%s=%s" % (_escape(item[0]), _number(item[1])))
    return ", ".join(rendered)


def _plain_leg_markets(payload: Mapping[str, Any]) -> str:
    quotes = payload.get("leg_quotes")
    legs = payload.get("legs")
    if not isinstance(quotes, list) or not isinstance(legs, list) or len(quotes) != len(legs):
        return "MISSING"
    by_symbol = {
        str(item.get("occ_symbol")): item
        for item in quotes
        if isinstance(item, Mapping)
    }
    values = []
    for leg in legs:
        if not isinstance(leg, Mapping):
            return "MISSING"
        quote = by_symbol.get(str(leg.get("occ_symbol")))
        if quote is None:
            return "MISSING"
        values.append(
            "%s %s %s %s %s/%s"
            % (
                _escape(leg.get("action", "MISSING")),
                _escape(leg.get("expiration", "MISSING")),
                _escape(leg.get("strike", "MISSING")),
                _escape(leg.get("option_type", "MISSING")),
                _number(quote.get("bid"), money=True),
                _number(quote.get("ask"), money=True),
            )
        )
    return "; ".join(values)


def _evidence_windows(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "MISSING"
    values = []
    for name in ("training", "validation", "holdout", "shadow"):
        period = value.get(name)
        if not isinstance(period, Mapping):
            continue
        values.append(
            "%s n=%s EV=%s LCB=%s"
            % (
                name,
                _escape(period.get("resolved_trades", "MISSING")),
                _number(period.get("expectancy"), money=True),
                _number(period.get("lower_confidence_bound"), money=True),
            )
        )
    return "; ".join(values) if values else "MISSING"


def _plain_legs(payload: Mapping[str, Any]) -> str:
    legs = payload.get("legs")
    if not isinstance(legs, list) or not legs:
        return "MISSING"
    rendered = []
    for leg in legs:
        if not isinstance(leg, Mapping):
            return "MISSING"
        rendered.append(
            "%s %sx %s %s %s"
            % (
                _escape(leg.get("action", "MISSING")),
                _escape(leg.get("ratio", "MISSING")),
                _escape(leg.get("expiration", "MISSING")),
                _escape(leg.get("strike", "MISSING")),
                _escape(leg.get("option_type", "MISSING")),
            )
        )
    return "; ".join(rendered)


def sorted_eligible_tickets(values: Sequence[Any]) -> tuple:
    """Sort every ticket by conservative EV/risk with deterministic stable ties."""

    decorated = []
    for index, value in enumerate(values):
        score = _lookup(
            value,
            "ranking_score",
            "edge.conservative_return_on_max_loss",
            "edge.conservative_return_on_risk",
            "edge.conservative_ev_per_risk",
        )
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as exc:
            raise ReportError("eligible ticket is missing its conservative EV/risk rank") from exc
        if not math.isfinite(numeric_score):
            raise ReportError("eligible ticket rank must be finite")
        candidate_id = str(_lookup(value, "candidate_id", "id") or "")
        symbol = str(_lookup(value, "symbol", "ticker") or "")
        family = str(_lookup(value, "strategy_family", "strategy_id") or "")
        decorated.append(
            ((-numeric_score, candidate_id, symbol, family, index), value)
        )
    return tuple(value for _key, value in sorted(decorated, key=lambda item: item[0]))


def _render_candidate_section(title: str, values: Sequence[Any], empty: str) -> str:
    lines = ["## %s" % title, ""]
    if not values:
        lines.append(empty)
        return "\n".join(lines)
    lines.extend(
        [
            "| State | Candidate | Symbol | Strategy | Reason | Rank (conservative EV/risk) |",
            "|---|---|---|---|---|---:|",
        ]
    )
    for index, value in enumerate(values, 1):
        payload = _mapping(value)
        identifier = _lookup(payload, "candidate_id", "id") or str(index)
        symbol = _lookup(payload, "symbol", "ticker") or "MISSING"
        family = _lookup(payload, "strategy_family", "strategy_id") or "MISSING"
        reason = _lookup(payload, "reason", "rejection_reason", "status_reason") or "Not supplied"
        rank = _lookup(payload, "rank_score", "conservative_ev_per_risk")
        disposition = str(_lookup(payload, "disposition", "status") or "")
        color = {
            "WATCHLIST": "🟠",
            "REJECTED": "🔴",
            "DATA_UNAVAILABLE": "⚪",
            "NOT_FULLY_EVALUATED_BUDGET": "⚪",
        }.get(disposition, "⚪")
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                color,
                _escape(identifier),
                _escape(symbol),
                _escape(family),
                _escape(reason),
                _number(rank, percent=True) if rank is not None else "—",
            )
        )
    return "\n".join(lines)


def render_daily_board(board: DailyBoardData) -> str:
    """Render all supplied entries; this function has no truncation parameter."""

    lines = [
        "# Cultra Daily Board — %s" % board.as_of.isoformat(),
        "",
        "**Overall status: `%s`**" % _escape(board.overall_status),
        "",
        "Profitability is an evidence state, not a guarantee. POP is calibrated net-profit probability; option delta is not POP.",
        "",
        "Run ID: `%s`" % _escape(board.run_id),
        "",
        "**Color key:** 🟢 qualified now · 🟠 validated setup awaiting its exact trigger · 🔴 rejected · ⚪ data or budget unavailable",
        "",
        _strategy_sections(board.strategy_states, board.strategy_rejection_reasons),
        "",
        "## Eligible manual-review tickets",
        "",
    ]
    if board.tickets:
        for index, ticket in enumerate(sorted_eligible_tickets(board.tickets), 1):
            lines.extend((_render_ticket(ticket, index), ""))
    else:
        lines.extend(
            (
                "No manual-review tickets. Cultra remains evidence-gated; a zero-ticket run does not trigger extra data requests.",
                "",
            )
        )

    lines.extend(
        (
            _render_candidate_section("Watchlist", board.watchlist, "No watchlist candidates."),
            "",
            _render_candidate_section("Rejected", board.rejected, "No rejected candidates."),
            "",
            _render_candidate_section(
                "Data unavailable",
                board.data_unavailable,
                "No candidates are blocked by unavailable data.",
            ),
            "",
            _render_candidate_section(
                "Not fully evaluated — request budget",
                board.budget_unresolved,
                "No candidates remain unresolved because of the request budget.",
            ),
            "",
            "---",
            "Cultra produces one normalized structure unit and never chooses quantity or submits an order.",
            "",
        )
    )
    return "\n".join(lines)
