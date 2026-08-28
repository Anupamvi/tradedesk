from datetime import date, timedelta

import pytest

from codexswing.backtest.splitter import (
    BacktestSplitError,
    LabelSpan,
    purged_expanding_walk_forward,
)


def _spans(count: int = 100, horizon_days: int = 5):
    start = date(2026, 1, 1)
    return [
        LabelSpan(
            sample_id="sample-{}".format(index),
            decision_date=(start + timedelta(days=index)).isoformat(),
            label_end_date=(start + timedelta(days=index + horizon_days)).isoformat(),
        )
        for index in range(count)
    ]


def test_purged_walk_forward_has_no_label_overlap_or_holdout_leakage() -> None:
    spans = _spans()
    plan = purged_expanding_walk_forward(
        spans,
        min_train_dates=30,
        test_dates=10,
        embargo_dates=5,
        holdout_dates=20,
    )
    assert len(plan.folds) == 5
    assert len(plan.holdout_decision_dates) == 20
    by_id = {item.sample_id: item for item in spans}
    holdout = set(plan.holdout_ids)
    for fold in plan.folds:
        assert not set(fold.train_ids) & set(fold.test_ids)
        assert not set(fold.train_ids) & holdout
        assert all(by_id[item].label_end_date < fold.test_start_date for item in fold.train_ids)
        assert len(fold.embargoed_decision_dates) == 5
    plan.assert_integrity(spans)


def test_split_rejects_too_little_history() -> None:
    with pytest.raises(BacktestSplitError, match="at least"):
        purged_expanding_walk_forward(
            _spans(count=49),
            min_train_dates=20,
            test_dates=10,
            holdout_dates=20,
        )
