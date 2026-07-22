from __future__ import annotations

from pathlib import Path

import pandas as pd

from codexuw.portfolio_capacity import (
    accepted_rows,
    actionable_rows,
    build_portfolio_capacity_payload,
    metrics,
)


def test_accepted_rows_intersects_decision_and_replay_guard(tmp_path) -> None:
    path = tmp_path / "replay.csv"
    pd.DataFrame(
        [
            {
                "asof": "2026-01-02",
                "exit_day": "2026-01-10",
                "expiry": "2026-01-16",
                "ticker": "KEPT",
                "sector": "Technology",
                "direction": "Bull Call",
                "entry_debit": 1.0,
                "entry_width": 5.0,
                "exact_evaluated": True,
                "decision_pass": True,
                "replay_guard_pass": True,
                "pnl_1x": 50.0,
            },
            {
                "asof": "2026-01-03",
                "exit_day": "2026-01-10",
                "expiry": "2026-01-16",
                "ticker": "GUARD_FAIL",
                "sector": "Technology",
                "direction": "Bull Call",
                "entry_debit": 1.0,
                "entry_width": 5.0,
                "exact_evaluated": True,
                "decision_pass": True,
                "replay_guard_pass": False,
                "pnl_1x": 5_000.0,
            },
        ]
    ).to_csv(path, index=False)

    rows = accepted_rows(path)

    assert rows["ticker"].tolist() == ["KEPT"]


def test_capacity_payload_enforces_overlap_concentration_and_fill_stress() -> None:
    rows = pd.DataFrame(
        [
            {
                "entry_date": pd.Timestamp("2026-01-02"),
                "exit_date": pd.Timestamp("2026-01-10"),
                "ticker": "AAA",
                "sector": "Technology",
                "direction": "Bull Call",
                "risk_1x": 100.0,
                "entry_debit": 1.0,
                "entry_credit": 0.0,
                "pnl_1x": 50.0,
            },
            {
                "entry_date": pd.Timestamp("2026-01-03"),
                "exit_date": pd.Timestamp("2026-01-11"),
                "ticker": "BBB",
                "sector": "Technology",
                "direction": "Bull Call",
                "risk_1x": 100.0,
                "entry_debit": 1.0,
                "entry_credit": 0.0,
                "pnl_1x": 40.0,
            },
        ]
    )
    payload, _, sized = build_portfolio_capacity_payload(
        rows,
        source="unit-test",
        monthly_target=100.0,
        account_value=10_000.0,
        risk_per_trade_pct=0.02,
        max_contracts=3,
        max_ticker_share=0.20,
        max_sector_share=0.03,
    )

    assert sized["contracts"].tolist() == [2, 1]
    assert (
        payload["scenarios"]["risk_sized_worse_fill_10pct"]["net_pnl"]
        < payload["scenarios"]["risk_sized_base"]["net_pnl"]
    )
    assert payload["feasibility"]["months_observed"] == 1


def test_versioned_actionable_book_preserves_medium_sleeve_pf() -> None:
    history = (
        Path(__file__).resolve().parents[1]
        / "codexuw"
        / "history"
        / "codexdaily_v4_edge_history_v2_2026-07-10.csv.gz"
    )

    rows = actionable_rows(history)
    result = metrics(rows, "pnl_1x")

    assert result["trades"] == 10
    assert result["profit_factor"] >= 3.57
    assert result["net_pnl"] >= 748.0
