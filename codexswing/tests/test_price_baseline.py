from pathlib import Path

import pytest

from codexswing.features.price import PriceObservation
from codexswing.models.baseline import BaselineDataError, compute_price_move_baseline


def _observations(count: int = 25, jump: bool = False):
    values = []
    close = 100.0
    for index in range(count):
        close *= 1.60 if jump and index == count - 1 else 1.005
        values.append(
            PriceObservation(
                session_date="2026-07-{:02d}".format(index + 1),
                ticker="TEST",
                close=close,
                high=close * 1.01,
                low=close * 0.99,
                volume=1_000_000 + index * 10_000,
                avg30_volume=1_000_000,
                market_cap=10_000_000_000,
                issue_type="Common Stock",
                sector="Technology",
            )
        )
    return values


def test_transparent_price_move_baseline() -> None:
    baseline = compute_price_move_baseline(_observations())
    assert baseline.status == "RESEARCH_BASELINE_UNVALIDATED"
    assert baseline.return_20d > baseline.return_5d > 0
    assert baseline.realized_vol_20d_annualized >= 0
    assert baseline.average_dollar_volume_20d > 0


def test_possible_corporate_action_fails_closed() -> None:
    with pytest.raises(BaselineDataError, match="corporate-action"):
        compute_price_move_baseline(_observations(jump=True))


def test_too_little_history_is_rejected() -> None:
    with pytest.raises(BaselineDataError, match="at least 21"):
        compute_price_move_baseline(_observations(count=20))
