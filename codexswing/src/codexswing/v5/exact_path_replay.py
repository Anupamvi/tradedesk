"""Exact-chain multi-session replay built on the frozen v0.4 pricing rules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

from codexswing.backtest.orats_option_replay import (
    ROUND_TRIP_SINGLE_COMMISSIONS,
    ROUND_TRIP_VERTICAL_COMMISSIONS,
    HistoricalSingleOption,
    HistoricalVertical,
    close_historical_single_option,
    close_historical_vertical,
    select_historical_single_option,
    select_historical_vertical,
)
from codexswing.v5.events import CorporateEvent, evaluate_event_exclusions
from codexswing.v5.replay_plan import ExitDecision, ReplayPathSample, SessionPnL
from codexswing.v5.spec import ExitPolicySpec


Position = Union[HistoricalSingleOption, HistoricalVertical]


@dataclass(frozen=True)
class ExactPathReplayResult:
    ticker: str
    strategy: str
    decision_date: str
    entry_date: str
    planned_exit_date: str
    disposition: str
    reason: str
    position: Optional[Position]
    pnl_path: Tuple[SessionPnL, ...]
    exit_decision: Optional[ExitDecision]

    def to_dict(self) -> Mapping[str, Any]:
        output = asdict(self)
        if self.position is not None:
            output["position"] = self.position.to_dict()
        return output


def _strategy_side(strategy: str) -> str:
    if strategy in {"LONG_CALL", "BULL_CALL_DEBIT", "BULL_PUT_CREDIT"}:
        return "LONG"
    if strategy in {"LONG_PUT", "BEAR_PUT_DEBIT", "BEAR_CALL_CREDIT"}:
        return "SHORT"
    raise ValueError("unsupported v0.5 replay strategy")


def replay_exact_option_path(
    sample: ReplayPathSample,
    strategy: str,
    exit_policy: ExitPolicySpec,
    chains_by_date: Mapping[str, Sequence[Mapping[str, Any]]],
    events: Sequence[CorporateEvent] = (),
) -> ExactPathReplayResult:
    """Replay one already-triggered signal; missing exact quotes fail closed."""

    side = _strategy_side(strategy)
    event_decision = evaluate_event_exclusions(
        sample.ticker,
        strategy,
        sample.entry_date,
        sample.path_dates[-1],
        events,
    )
    if not event_decision.eligible:
        return ExactPathReplayResult(
            sample.ticker,
            strategy,
            sample.decision_date,
            sample.entry_date,
            sample.path_dates[-1],
            "EVENT_EXCLUDED",
            ";".join(event_decision.reasons),
            None,
            (),
            None,
        )

    entry_rows = chains_by_date.get(sample.entry_date, ())
    if strategy in {"LONG_CALL", "LONG_PUT"}:
        position, reason = select_historical_single_option(
            entry_rows, sample.ticker, side, sample.entry_date, strategy
        )
    else:
        position, reason = select_historical_vertical(
            entry_rows, sample.ticker, side, sample.entry_date, strategy
        )
    if position is None:
        return ExactPathReplayResult(
            sample.ticker,
            strategy,
            sample.decision_date,
            sample.entry_date,
            sample.path_dates[-1],
            "NO_ENTRY",
            reason,
            None,
            (),
            None,
        )

    commission = (
        ROUND_TRIP_SINGLE_COMMISSIONS
        if isinstance(position, HistoricalSingleOption)
        else ROUND_TRIP_VERTICAL_COMMISSIONS
    )
    pnl_path = []
    exit_decision = None
    for session_number, session_date in enumerate(sample.path_dates, start=1):
        rows = chains_by_date.get(session_date, ())
        if isinstance(position, HistoricalSingleOption):
            exit_value, close_reason = close_historical_single_option(
                position, rows, session_date
            )
        else:
            exit_value, close_reason = close_historical_vertical(position, rows, session_date)
        if exit_value is None:
            return ExactPathReplayResult(
                sample.ticker,
                strategy,
                sample.decision_date,
                sample.entry_date,
                sample.path_dates[-1],
                "UNRESOLVED",
                close_reason,
                position,
                tuple(pnl_path),
                None,
            )
        net_pnl = (exit_value - position.modeled_entry_signed_debit) * 100.0 - commission
        pnl_path.append(SessionPnL(session_date, net_pnl))
        if (
            exit_policy.profit_target_r > 0
            and net_pnl >= exit_policy.profit_target_r * position.maximum_risk_dollars
        ):
            exit_decision = ExitDecision(
                session_date, net_pnl, "PROFIT_TARGET", session_number
            )
            break
        if (
            exit_policy.stop_loss_r > 0
            and net_pnl <= -exit_policy.stop_loss_r * position.maximum_risk_dollars
        ):
            exit_decision = ExitDecision(
                session_date, net_pnl, "STOP_LOSS", session_number
            )
            break
    if exit_decision is None:
        final = pnl_path[-1]
        exit_decision = ExitDecision(
            final.session_date, final.pnl_dollars, "HORIZON", len(pnl_path)
        )
    return ExactPathReplayResult(
        sample.ticker,
        strategy,
        sample.decision_date,
        sample.entry_date,
        sample.path_dates[-1],
        "CLOSED",
        "closed_at_exact_conservative_quote",
        position,
        tuple(pnl_path),
        exit_decision,
    )

