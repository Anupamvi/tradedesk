"""Leakage-resistant replay primitives for CodexSwing research."""

from codexswing.backtest.labels import DailyBar, StockOutcome, exact_next_open_outcome
from codexswing.backtest.metrics import ReplayMetrics, compute_replay_metrics
from codexswing.backtest.splitter import LabelSpan, SplitPlan, purged_expanding_walk_forward

__all__ = [
    "DailyBar",
    "LabelSpan",
    "ReplayMetrics",
    "SplitPlan",
    "StockOutcome",
    "compute_replay_metrics",
    "exact_next_open_outcome",
    "purged_expanding_walk_forward",
]
