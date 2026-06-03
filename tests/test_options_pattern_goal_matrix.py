import argparse
from pathlib import Path

from scripts import options_pattern_goal_matrix as matrix


def test_resolve_dates_uses_all_source_complete_scope_with_bounds(monkeypatch, tmp_path):
    monkeypatch.setattr(
        matrix,
        "strict_source_complete_dates",
        lambda base_dir: ["2026-01-02", "2026-01-05", "2026-01-05", "2026-01-12"],
    )
    args = argparse.Namespace(
        all_source_complete=True,
        dates=["2026-05-15"],
        from_date="2026-01-03",
        to_date="2026-01-10",
    )

    dates, scope = matrix.resolve_dates(args, tmp_path)

    assert dates == ["2026-01-05"]
    assert scope == "all_source_complete;from=2026-01-03,to=2026-01-10;date_count=1"


def test_resolve_dates_preserves_requested_date_order_and_dedupes(tmp_path):
    args = argparse.Namespace(
        all_source_complete=False,
        dates=["2026-05-20", "2026-05-18", "2026-05-20", "2026-05-22"],
        from_date="2026-05-18",
        to_date="2026-05-20",
    )

    dates, scope = matrix.resolve_dates(args, Path(tmp_path))

    assert dates == ["2026-05-20", "2026-05-18"]
    assert scope == "requested_dates;from=2026-05-18,to=2026-05-20;date_count=2"


def test_matrix_status_names_full_source_complete_scope():
    assert matrix.matrix_status([], [], "all_source_complete;unbounded;date_count=10") == "PASS_SOURCE_COMPLETE_SCOPE"
    assert matrix.matrix_status([], [], "requested_dates;unbounded;date_count=8") == "PASS_DAILY_NOT_GLOBAL"
