"""Measurement and validation.

Trades entered on the same session share a market shock, so every interval here
resamples whole SESSIONS, never individual rows.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

RESOLVED_ONLY = "resolved"


def resolved(trades: pd.DataFrame) -> pd.DataFrame:
    return trades[trades["resolved"].fillna(False) & trades["pnl"].notna()]


def equity_curve(trades: pd.DataFrame) -> pd.Series:
    ordered = trades.dropna(subset=["exit_session"]).sort_values("exit_session")
    return ordered.groupby("exit_session")["pnl"].sum().cumsum()


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    return float((curve - curve.cummax()).min())


def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def summarize(trades: pd.DataFrame) -> dict:
    trades = resolved(trades)
    if trades.empty:
        return {"n": 0}
    pnl = trades["pnl"]
    curve = equity_curve(trades)
    months = monthly(trades)
    return {
        "n": int(len(trades)),
        "sessions": int(trades["entry_session"].nunique()),
        "win_rate": float((pnl > 0).mean()),
        "avg_pnl": float(pnl.mean()),
        "total_pnl": float(pnl.sum()),
        "profit_factor": profit_factor(pnl),
        "avg_return_on_risk": float(trades["return_on_risk"].mean()),
        "max_drawdown": max_drawdown(curve),
        "months_profitable": int((months > 0).sum()),
        "months_total": int(len(months)),
        "peak_capital": float(peak_capital(trades)),
    }


def monthly(trades: pd.DataFrame) -> pd.Series:
    trades = resolved(trades)
    if trades.empty:
        return pd.Series(dtype=float)
    return trades.groupby(trades["entry_session"].str[:7])["pnl"].sum()


def peak_capital(trades: pd.DataFrame) -> float:
    """Largest simultaneous capital at risk, from overlapping holding periods."""
    trades = resolved(trades).dropna(subset=["exit_session"])
    if trades.empty:
        return 0.0
    moves: dict[str, float] = {}
    for _, row in trades.iterrows():
        moves[row["entry_session"]] = moves.get(row["entry_session"], 0.0) + row["max_risk"]
        moves[row["exit_session"]] = moves.get(row["exit_session"], 0.0) - row["max_risk"]
    running, peak = 0.0, 0.0
    for session in sorted(moves):
        running += moves[session]
        peak = max(peak, running)
    return peak


def day_block_bootstrap(
    trades: pd.DataFrame, iterations: int = 2000, seed: int = 7
) -> dict:
    """Bootstrap the mean P&L per trade by resampling entry sessions with replacement."""
    trades = resolved(trades)
    if trades.empty:
        return {"mean": float("nan"), "p05": float("nan"), "p95": float("nan"), "p_loss": float("nan")}

    groups = [g["pnl"].to_numpy() for _, g in trades.groupby("entry_session")]
    rng = np.random.default_rng(seed)
    count = len(groups)
    means = np.empty(iterations)
    for i in range(iterations):
        picks = rng.integers(0, count, count)
        sample = np.concatenate([groups[p] for p in picks])
        means[i] = sample.mean()
    return {
        "mean": float(trades["pnl"].mean()),
        "p05": float(np.percentile(means, 5)),
        "p95": float(np.percentile(means, 95)),
        "p_loss": float((means <= 0).mean()),
    }


def walk_forward(
    trades: pd.DataFrame,
    fit: Callable[[pd.DataFrame], Callable[[pd.DataFrame], pd.DataFrame]],
    folds: int = 4,
    min_train_sessions: int = 30,
) -> pd.DataFrame:
    """Expanding-window walk forward.

    ``fit`` receives only past trades and returns a selector applied to the next
    fold. Nothing from the test fold, including any threshold, informs the fit.
    """
    trades = resolved(trades).sort_values("entry_session")
    sessions = sorted(trades["entry_session"].unique())
    if len(sessions) < min_train_sessions + folds:
        return pd.DataFrame()

    edges = np.linspace(min_train_sessions, len(sessions), folds + 1).astype(int)
    picked = []
    for fold, (start, stop) in enumerate(zip(edges[:-1], edges[1:]), 1):
        if stop <= start:
            continue
        train = trades[trades["entry_session"].isin(sessions[:start])]
        test = trades[trades["entry_session"].isin(sessions[start:stop])]
        if train.empty or test.empty:
            continue
        selector = fit(train)
        chosen = selector(test)
        if chosen is None or chosen.empty:
            continue
        chosen = chosen.copy()
        chosen["fold"] = fold
        picked.append(chosen)

    return pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()


def scorecard(trades: pd.DataFrame, label: str = "") -> dict:
    """The consistency test. Every condition must hold for a strategy to ship."""
    stats = summarize(trades)
    if not stats.get("n"):
        return {"label": label, "n": 0, "passes": False, "reasons": ["no resolved trades"]}
    boot = day_block_bootstrap(trades)
    reasons = []
    if stats["total_pnl"] <= 0:
        reasons.append("total P&L not positive")
    if boot["p05"] <= 0:
        reasons.append(f"bootstrap 5th pct {boot['p05']:.2f} <= 0")
    if stats["months_profitable"] / max(stats["months_total"], 1) < 0.6:
        reasons.append(
            f"only {stats['months_profitable']}/{stats['months_total']} months profitable"
        )
    if stats["profit_factor"] < 1.2:
        reasons.append(f"profit factor {stats['profit_factor']:.2f} < 1.2")
    return {"label": label, **stats, **{f"boot_{k}": v for k, v in boot.items()},
            "passes": not reasons, "reasons": reasons}
