import datetime as dt

import pandas as pd

from codexuw.sector_rotation import (
    apply_sector_rotation_context,
    build_live_sector_rotation,
    build_sector_rotation_table,
)


def test_sector_rotation_detects_any_accelerating_sector_without_authorizing_it() -> None:
    rows = []
    for index, date in enumerate(pd.date_range("2026-01-02", periods=12)):
        for sector, direction in (("Technology", 1), ("Energy", -1)):
            for ticker_number in range(3):
                position = 0.30 + index * 0.05 if direction > 0 else 0.85 - index * 0.05
                rows.append(
                    {
                        "date": str(date.date()),
                        "ticker": f"{sector[:2]}{ticker_number}",
                        "sector": sector,
                        "close": 100 + position * 10,
                        "prev_close": 100,
                        "week_52_high": 110,
                        "week_52_low": 100,
                        "flow_total_premium": (index + 1) * 10_000_000 if direction > 0 else 10_000_000,
                    }
                )
    state = build_sector_rotation_table(pd.DataFrame(rows))
    latest = state[state["date"].eq("2026-01-13")].set_index("sector")

    assert latest.loc["Technology", "sector_state"] == "emerging"
    assert latest.loc["Technology", "sector_rotation_authority"] == "prospective_context_only"

    scored = pd.DataFrame([{"ticker": "AAA", "sector": "Technology", "trade_status": "Research"}])
    annotated = apply_sector_rotation_context(scored, latest.reset_index())
    assert annotated.iloc[0]["sector_state"] == "emerging"
    assert annotated.iloc[0]["trade_status"] == "Research"


def test_live_sector_rotation_ignores_overlay_directories(tmp_path, monkeypatch) -> None:
    exact = tmp_path / "2026-07-29"
    overlay = tmp_path / "2026-07-29_overlay_2026-07-30"
    exact.mkdir()
    overlay.mkdir()
    loaded = []

    def fake_load(folder, *, point_in_time):
        loaded.append(folder.name)
        return pd.DataFrame(
            [
                {
                    "ticker": "AAA",
                    "sector": "Technology",
                    "issue_type": "Common Stock",
                    "close": 100.0,
                    "prev_close": 99.0,
                    "week_52_high": 110.0,
                    "week_52_low": 80.0,
                    "flow_total_premium": 10_000_000.0,
                }
            ]
        )

    monkeypatch.setattr("codexuw.sector_rotation.load_stock_screener", fake_load)

    build_live_sector_rotation(tmp_path, asof=dt.date(2026, 7, 29))

    assert loaded == ["2026-07-29"]