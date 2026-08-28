"""Predeclared replay variants, paths, exits, and cache requirements."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from codexswing.backtest.labels import DailyBar
from codexswing.v5.budget import CacheKey
from codexswing.v5.spec import ExitPolicySpec, V5ResearchSpec


@dataclass(frozen=True)
class ReplayVariant:
    strategy: str
    horizon_sessions: int
    exit_policy: ExitPolicySpec

    @property
    def hypothesis_id(self) -> str:
        raw = "{}:{}:{}".format(
            self.strategy, self.horizon_sessions, self.exit_policy.policy_id
        )
        suffix = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10].upper()
        return "V5_{}_H{}_{}_{}".format(
            self.strategy, self.horizon_sessions, self.exit_policy.policy_id, suffix
        )


def declared_variants(spec: V5ResearchSpec) -> Tuple[ReplayVariant, ...]:
    variants = tuple(
        ReplayVariant(strategy, horizon, policy)
        for strategy in spec.strategies
        for horizon in spec.horizons_sessions
        for policy in spec.exit_policies
    )
    if len(variants) != spec.hypothesis_count or len(
        {item.hypothesis_id for item in variants}
    ) != len(variants):
        raise ValueError("replay variant declaration is not stable and unique")
    return variants


@dataclass(frozen=True)
class ReplayPathSample:
    ticker: str
    decision_date: str
    entry_date: str
    path_dates: Tuple[str, ...]
    horizon_sessions: int

    def __post_init__(self) -> None:
        dates = (self.decision_date, self.entry_date) + self.path_dates
        try:
            parsed = tuple(date.fromisoformat(item) for item in dates)
        except ValueError:
            raise ValueError("replay path dates must be YYYY-MM-DD") from None
        if not self.ticker or self.ticker != self.ticker.upper():
            raise ValueError("replay ticker must be uppercase")
        if self.horizon_sessions <= 0 or len(self.path_dates) != self.horizon_sessions:
            raise ValueError("path length must equal horizon_sessions")
        if self.path_dates[0] != self.entry_date:
            raise ValueError("the first path session must be the entry date")
        if len(set(self.path_dates)) != len(self.path_dates):
            raise ValueError("path sessions must be unique")
        if parsed != tuple(sorted(parsed)):
            raise ValueError("replay dates must be chronological")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ReplayPathSample":
        path_dates = tuple(str(item) for item in payload["path_dates"])
        return cls(
            ticker=str(payload["ticker"]).upper(),
            decision_date=str(payload["decision_date"]),
            entry_date=str(payload["entry_date"]),
            path_dates=path_dates,
            horizon_sessions=int(payload.get("horizon_sessions") or len(path_dates)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "decision_date": self.decision_date,
            "entry_date": self.entry_date,
            "path_dates": list(self.path_dates),
            "horizon_sessions": self.horizon_sessions,
        }


def build_replay_paths(
    ticker: str,
    decision_dates: Iterable[str],
    bars: Sequence[DailyBar],
    horizons: Sequence[int] = (3, 5, 10, 20),
) -> Tuple[ReplayPathSample, ...]:
    normalized = ticker.strip().upper()
    ordered = tuple(sorted((item for item in bars if item.ticker == normalized), key=lambda x: x.trade_date))
    dates = [item.trade_date for item in ordered]
    if len(dates) != len(set(dates)):
        raise ValueError("daily bars contain duplicate dates")
    paths: List[ReplayPathSample] = []
    for decision_date in sorted(set(decision_dates)):
        try:
            decision_index = dates.index(decision_date)
        except ValueError:
            raise ValueError("decision date {} has no daily bar".format(decision_date)) from None
        entry_index = decision_index + 1
        for horizon in horizons:
            if horizon <= 0:
                raise ValueError("horizons must be positive")
            end_index = entry_index + horizon
            if end_index > len(dates):
                raise ValueError(
                    "incomplete {}-session path after {}".format(horizon, decision_date)
                )
            path_dates = tuple(dates[entry_index:end_index])
            paths.append(
                ReplayPathSample(
                    ticker=normalized,
                    decision_date=decision_date,
                    entry_date=path_dates[0],
                    path_dates=path_dates,
                    horizon_sessions=horizon,
                )
            )
    return tuple(paths)


def cache_requirements_for_paths(
    paths: Iterable[ReplayPathSample],
) -> Tuple[CacheKey, ...]:
    requirements = set()
    for sample in paths:
        requirements.add(CacheKey("hist/cores", sample.ticker, sample.decision_date))
        requirements.add(CacheKey("hist/dailies", sample.ticker, sample.decision_date))
        requirements.add(CacheKey("hist/earnings", sample.ticker, sample.decision_date))
        requirements.add(CacheKey("hist/summaries", sample.ticker, sample.decision_date))
        for session_date in sample.path_dates:
            requirements.add(CacheKey("hist/strikes", sample.ticker, session_date))
    return tuple(sorted(requirements))


@dataclass(frozen=True)
class SessionPnL:
    session_date: str
    pnl_dollars: float

    def __post_init__(self) -> None:
        try:
            date.fromisoformat(self.session_date)
        except ValueError:
            raise ValueError("P&L session date must be YYYY-MM-DD") from None
        if not math.isfinite(self.pnl_dollars):
            raise ValueError("P&L must be finite")


@dataclass(frozen=True)
class ExitDecision:
    session_date: str
    pnl_dollars: float
    reason: str
    session_number: int


def choose_path_exit(
    path: Sequence[SessionPnL],
    maximum_risk_dollars: float,
    exit_policy: ExitPolicySpec,
) -> ExitDecision:
    """Choose the first predeclared EOD target/stop, else the horizon close."""

    if not path:
        raise ValueError("P&L path cannot be empty")
    if maximum_risk_dollars <= 0:
        raise ValueError("maximum risk must be positive")
    seen = set()
    for index, observation in enumerate(path, start=1):
        if observation.session_date in seen:
            raise ValueError("P&L path contains duplicate sessions")
        seen.add(observation.session_date)
        if exit_policy.profit_target_r > 0 and observation.pnl_dollars >= (
            exit_policy.profit_target_r * maximum_risk_dollars
        ):
            return ExitDecision(
                observation.session_date, observation.pnl_dollars, "PROFIT_TARGET", index
            )
        if exit_policy.stop_loss_r > 0 and observation.pnl_dollars <= -(
            exit_policy.stop_loss_r * maximum_risk_dollars
        ):
            return ExitDecision(
                observation.session_date, observation.pnl_dollars, "STOP_LOSS", index
            )
    final = path[-1]
    return ExitDecision(final.session_date, final.pnl_dollars, "HORIZON", len(path))
