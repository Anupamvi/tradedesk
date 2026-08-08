import zipfile

import pandas as pd
import pytest

from uwos.surge_v1 import PIPELINE_VERSION
from uwos.surge_v1.execution_cost_audit import (
    attach_execution_dates,
    load_stock_nbbo,
    price_book,
    scenario_stats,
)


def test_surge_pipeline_version_is_explicit() -> None:
    assert PIPELINE_VERSION == "surge-v1.0-normalized-walkforward-20260808"


def test_stock_nbbo_ignores_after_hours_quotes(tmp_path):
    day = "2026-04-22"
    source_dir = tmp_path / day
    source_dir.mkdir()
    rows = pd.DataFrame([
        {
            "ticker": "TEST",
            "executed_at": "2026-04-22T19:30:00Z",
            "nbbo_bid": 99.90,
            "nbbo_ask": 100.10,
            "nbbo_bid_quantity": 100,
            "nbbo_ask_quantity": 100,
            "canceled": False,
        },
        {
            "ticker": "TEST",
            "executed_at": "2026-04-22T20:05:00Z",
            "nbbo_bid": 1.00,
            "nbbo_ask": 100.00,
            "nbbo_bid_quantity": 1,
            "nbbo_ask_quantity": 1,
            "canceled": False,
        },
    ])
    source = source_dir / f"dp-eod-report-{day}.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("quotes.csv", rows.to_csv(index=False))

    quotes = load_stock_nbbo(tmp_path, pd.Timestamp(day), {"TEST"})

    assert len(quotes) == 1
    assert quotes.iloc[0]["quote_window"] == "last_hour"
    assert quotes.iloc[0]["half_spread_bps"] == pytest.approx(10.0)


def test_attach_execution_dates_uses_next_close_and_horizon_exit():
    sessions = list(pd.date_range("2026-01-05", periods=6, freq="B"))
    legs = pd.DataFrame({"date": [sessions[0]], "ticker": ["TEST"]})

    attached = attach_execution_dates(legs, sessions, horizon=3)

    assert attached.iloc[0]["entry_date"] == sessions[1]
    assert attached.iloc[0]["exit_date"] == sessions[4]


def test_price_book_charges_two_half_spreads_and_short_borrow():
    entry = pd.Timestamp("2026-01-06")
    exit_date = pd.Timestamp("2026-01-09")
    legs = pd.DataFrame([
        {"ticker": "LONG", "entry_date": entry, "exit_date": exit_date,
         "date": pd.Timestamp("2026-01-05"), "side": "long", "gross": 0.10},
        {"ticker": "SHORT", "entry_date": entry, "exit_date": exit_date,
         "date": pd.Timestamp("2026-01-05"), "side": "short", "gross": -0.10},
    ])
    quotes = pd.DataFrame([
        {"ticker": ticker, "quote_date": date, "half_spread_bps": 5.0,
         "spread_p75_bps": 5.0, "quote_observations": 10,
         "bid_depth_dollars": 10_000, "ask_depth_dollars": 10_000,
         "quote_window": "last_hour"}
        for ticker in ("LONG", "SHORT") for date in (entry, exit_date)
    ])

    priced = price_book(legs, quotes, horizon=3, borrow_bps_annual=200.0)

    assert priced.loc[0, "audited_net"] == pytest.approx(0.099)
    expected_short = 0.10 - 0.001 - 0.02 * 3 / 252
    assert priced.loc[1, "audited_net"] == pytest.approx(expected_short)
    stats = scenario_stats(priced.iloc[:1], "partial", total_legs=2, horizon=3)
    assert stats["coverage"] == 0.5