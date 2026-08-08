import pandas as pd

from claude_pipeline import PIPELINE_VERSION
from claude_pipeline import daily


def test_daily_scanner_is_fail_closed_without_validated_edge() -> None:
    chain = pd.DataFrame(
        [
            {
                "kind": "C",
                "expiry": "2026-09-18",
                "strike": 105.0,
                "bid": 2.50,
                "ask": 2.60,
                "open_interest": 500,
            },
            {
                "kind": "C",
                "expiry": "2026-09-18",
                "strike": 110.0,
                "bid": 0.40,
                "ask": 0.50,
                "open_interest": 500,
            },
        ]
    )

    tickets = daily._verticals(chain, 100.0, "TEST", pd.Timestamp("2026-08-14"))

    assert tickets
    assert tickets[0].credit_pct_width >= daily.MIN_CREDIT_PCT_WIDTH
    assert tickets[0].status == "watch"
    assert "unvalidated" in tickets[0].blocker
    assert not daily.VALIDATED_EDGE


def test_daily_version_is_available_for_artifacts() -> None:
    assert PIPELINE_VERSION == "claude-pipeline-v0.2-research-only-daily-20260808"