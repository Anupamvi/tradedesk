"""Joining decision-time context onto trades, and the selection rules under test.

Every feature is taken from the entry session's panel row, so nothing here can see
past the moment the trade was opened.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "iv30d", "iv_rank", "rv21", "rv63", "iv_rv_ratio", "range_position", "volume_surge",
    "option_volume_surge", "bull_premium_share", "put_call_ratio", "days_to_earnings",
    "dollar_volume", "marketcap", "sector", "issue_type", "day_return", "next_earnings_date",
]

REGIME_COLUMNS = ["vix", "spx_return_5d", "spx_return_21d", "trend", "vol_state", "breadth_up_share"]

CREDIT_FAMILIES = ("bull_put_credit", "bear_call_credit", "short_put")


def attach_features(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    regime: pd.DataFrame | None = None,
    filings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    features = panel[["session", "ticker", *FEATURE_COLUMNS]].drop_duplicates(["session", "ticker"])
    merged = trades.merge(
        features, left_on=["entry_session", "ticker"], right_on=["session", "ticker"], how="left"
    ).drop(columns=["session"])

    if regime is not None:
        merged = merged.merge(
            regime[["session", *REGIME_COLUMNS]], left_on="entry_session", right_on="session",
            how="left",
        ).drop(columns=["session"])

    if filings is not None:
        merged = merged.merge(
            filings, left_on=["entry_session", "ticker"], right_on=["session", "ticker"], how="left"
        ).drop(columns=["session"])
        for column in ("filings_total", "filings_8k", "filings_insider", "filings_periodic"):
            if column in merged:
                merged[column] = merged[column].fillna(0)

    merged["expiry"] = pd.to_datetime(merged["expiry"], errors="coerce")
    merged["next_earnings_date"] = pd.to_datetime(merged["next_earnings_date"], errors="coerce")
    merged["earnings_before_expiry"] = (
        merged["next_earnings_date"].notna()
        & (merged["next_earnings_date"] <= merged["expiry"])
        & (merged["next_earnings_date"] >= pd.to_datetime(merged["entry_session"]))
    )
    merged["credit_pct_width"] = np.where(
        merged["family"].isin(("bull_put_credit", "bear_call_credit")),
        -merged["entry_net"] / merged["width"].replace(0, np.nan),
        np.nan,
    )
    return merged


def threshold_selector(
    feature: str, grid: list[float], direction: str = "above",
    min_trades: int = 40,
) -> Callable[[pd.DataFrame], Callable[[pd.DataFrame], pd.DataFrame]]:
    """Fit a cut on ``feature`` using only training trades, then apply it unchanged.

    Walking the THRESHOLD forward, not just the feature, is the point: a cut chosen
    with hindsight is the standard way a dead signal looks alive.
    """
    def fit(train: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
        best_value, best_score = None, -np.inf
        for value in grid:
            subset = train[train[feature] >= value] if direction == "above" else train[train[feature] <= value]
            if len(subset) < min_trades:
                continue
            score = subset["pnl"].sum()
            if score > best_score:
                best_value, best_score = value, score

        def apply(test: pd.DataFrame) -> pd.DataFrame:
            if best_value is None:
                return test.iloc[0:0]
            chosen = test[test[feature] >= best_value] if direction == "above" else test[test[feature] <= best_value]
            return chosen.assign(threshold=best_value)

        return apply

    return fit


def fixed_rule(predicate: Callable[[pd.DataFrame], pd.Series]):
    """A rule with no fitted parameter; the honest control for any fitted threshold."""
    def fit(_train: pd.DataFrame) -> Callable[[pd.DataFrame], pd.DataFrame]:
        return lambda test: test[predicate(test)]

    return fit
