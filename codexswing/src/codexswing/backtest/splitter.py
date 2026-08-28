"""Purged expanding walk-forward splits with a final untouched date block."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Sequence, Tuple


class BacktestSplitError(ValueError):
    pass


def _date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise BacktestSplitError("{} must be YYYY-MM-DD".format(label)) from None


@dataclass(frozen=True)
class LabelSpan:
    sample_id: str
    decision_date: str
    label_end_date: str

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise BacktestSplitError("sample_id cannot be empty")
        decision = _date(self.decision_date, "decision_date")
        label_end = _date(self.label_end_date, "label_end_date")
        if label_end <= decision:
            raise BacktestSplitError("label_end_date must be after decision_date")


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_ids: Sequence[str]
    test_ids: Sequence[str]
    purged_ids: Sequence[str]
    embargoed_decision_dates: Sequence[str]
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["train_count"] = len(self.train_ids)
        payload["test_count"] = len(self.test_ids)
        payload["purged_count"] = len(self.purged_ids)
        return payload


@dataclass(frozen=True)
class SplitPlan:
    folds: Sequence[WalkForwardFold]
    holdout_ids: Sequence[str]
    holdout_decision_dates: Sequence[str]
    boundary_excluded_ids: Sequence[str]
    decision_date_count: int
    min_train_dates: int
    test_dates: int
    embargo_dates: int
    holdout_dates: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "splitter": "PURGED_EXPANDING_WALK_FORWARD_V1",
            "folds": [fold.to_dict() for fold in self.folds],
            "fold_count": len(self.folds),
            "holdout_ids": list(self.holdout_ids),
            "holdout_count": len(self.holdout_ids),
            "holdout_decision_dates": list(self.holdout_decision_dates),
            "boundary_excluded_ids": list(self.boundary_excluded_ids),
            "configuration": {
                "decision_date_count": self.decision_date_count,
                "min_train_dates": self.min_train_dates,
                "test_dates": self.test_dates,
                "embargo_dates": self.embargo_dates,
                "holdout_dates": self.holdout_dates,
            },
        }

    def assert_integrity(self, spans: Iterable[LabelSpan]) -> None:
        materialized = tuple(spans)
        by_id = {span.sample_id: span for span in materialized}
        holdout = set(self.holdout_ids)
        if len(by_id) != len(materialized):
            raise BacktestSplitError("duplicate sample_id in split input")
        for fold in self.folds:
            train = set(fold.train_ids)
            test = set(fold.test_ids)
            if train & test or train & holdout or test & holdout:
                raise BacktestSplitError("train, test, and holdout sets must be disjoint within a fold")
            if not train or not test:
                raise BacktestSplitError("every fold requires non-empty train and test sets")
            for sample_id in train:
                if by_id[sample_id].label_end_date >= fold.test_start_date:
                    raise BacktestSplitError("training label overlaps the test period")
            if any(by_id[sample_id].decision_date < fold.test_start_date for sample_id in test):
                raise BacktestSplitError("test sample precedes the test period")


def purged_expanding_walk_forward(
    spans: Iterable[LabelSpan],
    min_train_dates: int = 60,
    test_dates: int = 20,
    embargo_dates: int = 5,
    holdout_dates: int = 30,
) -> SplitPlan:
    """Create folds using decision-date groups and purge labels crossing a boundary.

    The final ``holdout_dates`` decision dates are never included in a fold. Any
    development sample whose label reaches the holdout start is also excluded.
    """

    materialized = tuple(sorted(spans, key=lambda item: (item.decision_date, item.sample_id)))
    if not materialized:
        raise BacktestSplitError("at least one labeled sample is required")
    if len({item.sample_id for item in materialized}) != len(materialized):
        raise BacktestSplitError("sample_id values must be unique")
    for value, label in (
        (min_train_dates, "min_train_dates"),
        (test_dates, "test_dates"),
        (holdout_dates, "holdout_dates"),
    ):
        if value <= 0:
            raise BacktestSplitError("{} must be positive".format(label))
    if embargo_dates < 0:
        raise BacktestSplitError("embargo_dates cannot be negative")

    decision_dates = sorted({item.decision_date for item in materialized})
    required_dates = min_train_dates + test_dates + holdout_dates
    if len(decision_dates) < required_dates:
        raise BacktestSplitError(
            "split requires at least {} unique decision dates; found {}".format(
                required_dates, len(decision_dates)
            )
        )
    development_dates = decision_dates[:-holdout_dates]
    holdout_date_values = tuple(decision_dates[-holdout_dates:])
    holdout_start = holdout_date_values[0]
    holdout_date_set = set(holdout_date_values)
    holdout_ids = tuple(
        item.sample_id for item in materialized if item.decision_date in holdout_date_set
    )
    boundary_excluded = tuple(
        item.sample_id
        for item in materialized
        if item.decision_date in set(development_dates) and item.label_end_date >= holdout_start
    )
    boundary_excluded_set = set(boundary_excluded)

    folds: List[WalkForwardFold] = []
    fold_id = 1
    test_start_index = min_train_dates
    while test_start_index + test_dates <= len(development_dates):
        test_date_values = tuple(
            development_dates[test_start_index : test_start_index + test_dates]
        )
        test_date_set = set(test_date_values)
        candidate_train_dates = tuple(development_dates[:test_start_index])
        embargoed = tuple(candidate_train_dates[-embargo_dates:]) if embargo_dates else ()
        embargoed_set = set(embargoed)
        test_start = test_date_values[0]
        candidate_train_set = set(candidate_train_dates)

        train_ids = tuple(
            item.sample_id
            for item in materialized
            if item.decision_date in candidate_train_set
            and item.decision_date not in embargoed_set
            and item.label_end_date < test_start
            and item.sample_id not in boundary_excluded_set
        )
        purged_ids = tuple(
            item.sample_id
            for item in materialized
            if item.decision_date in candidate_train_set
            and (
                item.decision_date in embargoed_set
                or item.label_end_date >= test_start
                or item.sample_id in boundary_excluded_set
            )
        )
        test_ids = tuple(
            item.sample_id
            for item in materialized
            if item.decision_date in test_date_set and item.sample_id not in boundary_excluded_set
        )
        if not train_ids or not test_ids:
            raise BacktestSplitError("purging left an empty train or test fold")
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_ids=train_ids,
                test_ids=test_ids,
                purged_ids=purged_ids,
                embargoed_decision_dates=embargoed,
                train_start_date=candidate_train_dates[0],
                train_end_date=candidate_train_dates[-1],
                test_start_date=test_date_values[0],
                test_end_date=test_date_values[-1],
            )
        )
        fold_id += 1
        test_start_index += test_dates

    if not folds:
        raise BacktestSplitError("no complete walk-forward fold could be created")
    plan = SplitPlan(
        folds=tuple(folds),
        holdout_ids=holdout_ids,
        holdout_decision_dates=holdout_date_values,
        boundary_excluded_ids=boundary_excluded,
        decision_date_count=len(decision_dates),
        min_train_dates=min_train_dates,
        test_dates=test_dates,
        embargo_dates=embargo_dates,
        holdout_dates=holdout_dates,
    )
    plan.assert_integrity(materialized)
    return plan
