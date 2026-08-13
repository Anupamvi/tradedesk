from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from codexuw.data import aggregate_dark_pool_flow, find_export_bundle, load_bot_contract_quotes
from codexuw.liquidity_shift import scan_bot_flow_tape


def _write_zip(path: Path, rows: list[dict[str, object]]) -> None:
    csv_name = path.with_suffix(".csv").name
    csv_data = pd.DataFrame(rows).to_csv(index=False)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(csv_name, csv_data)


def test_dark_pool_flow_is_point_in_time_and_keeps_equity_signal_separate(tmp_path: Path) -> None:
    day = tmp_path / "2026-01-02"
    day.mkdir()
    _write_zip(
        day / "dp-eod-report-2026-01-02.zip",
        [
            {"ticker": "AAA", "nbbo_bid": 99, "nbbo_ask": 101, "price": 101, "size": 1, "premium": 100, "canceled": ""},
            {"ticker": "AAA", "nbbo_bid": 99, "nbbo_ask": 101, "price": 99, "size": 1, "premium": 50, "canceled": ""},
            {"ticker": "AAA", "nbbo_bid": 99, "nbbo_ask": 101, "price": 100, "size": 1, "premium": 25, "canceled": ""},
            {"ticker": "AAA", "nbbo_bid": 99, "nbbo_ask": 101, "price": 101, "size": 1, "premium": 999, "canceled": "true"},
            {"ticker": "BBB", "nbbo_bid": 0, "nbbo_ask": 0, "price": 20, "size": 10, "premium": 200, "canceled": ""},
        ],
    )
    _write_zip(
        day / "dp-eod-report-latest-2026-01-03.zip",
        [
            {"ticker": "AAA", "nbbo_bid": 99, "nbbo_ask": 101, "price": 99, "size": 1, "premium": 10_000, "canceled": ""},
        ],
    )

    result = aggregate_dark_pool_flow(day, ["AAA"], point_in_time=True)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["dp_bull_premium"] == 100
    assert row["dp_bear_premium"] == 50
    assert row["dp_neutral_premium"] == 25
    assert row["dp_total_premium"] == 175
    assert row["dp_directional_premium"] == 150
    assert row["dp_flow_bias"] == (100 - 50) / 150
    assert row["dp_directional_ratio"] == 150 / 175
    assert result.attrs["source_status"] == "dp_eod_loaded"
    assert "2026-01-02" in result.attrs["source_path"]


def test_bot_contract_quotes_use_first_valid_regular_session_nbbo(tmp_path: Path) -> None:
    day = tmp_path / "2026-07-10"
    day.mkdir()
    common = {
        "option_chain_id": "XYZ260821C00105000",
        "price": 1.15,
        "volume": 100,
        "open_interest": 500,
        "canceled": "f",
    }
    _write_zip(
        day / "bot-eod-report-2026-07-10.zip",
        [
            {**common, "executed_at": "2026-07-10T13:00:00Z", "nbbo_bid": 5.0, "nbbo_ask": 5.2},
            {**common, "executed_at": "2026-07-10T13:30:00Z", "nbbo_bid": 9.0, "nbbo_ask": 9.2, "canceled": "t"},
            {**common, "executed_at": "2026-07-10T13:30:01Z", "nbbo_bid": 1.0, "nbbo_ask": 1.2},
            {**common, "executed_at": "2026-07-10T13:31:00Z", "nbbo_bid": 1.1, "nbbo_ask": 1.3},
        ],
    )

    quotes = load_bot_contract_quotes(
        day,
        ["XYZ260821C00105000"],
        point_in_time=True,
    )

    quote = quotes["XYZ260821C00105000"]
    assert quote["bid"] == 1.0
    assert quote["ask"] == 1.2
    assert quote["mid"] == 1.1
    assert quote["quote_source"] == "bot_eod_first_regular_nbbo"
    assert quote["quote_timestamp"].startswith("2026-07-10T13:30:01")


def test_liquidity_shift_bot_scan_excludes_future_latest_overlay(tmp_path: Path) -> None:
    day = tmp_path / "2026-01-02"
    day.mkdir()
    common = {
        "executed_at": "2026-01-02T20:00:00Z",
        "underlying_symbol": "AAA",
        "option_type": "call",
        "expiry": "2026-01-16",
        "strike": 100,
        "underlying_price": 100,
        "volume": 100,
        "open_interest": 1000,
        "delta": 0.5,
        "gamma": 0.1,
        "sector": "Technology",
    }
    _write_zip(
        day / "bot-eod-report-2026-01-02.zip",
        [{**common, "side": "ask", "premium": 1000}],
    )
    _write_zip(
        day / "bot-eod-report-latest-2026-01-03.zip",
        [{**common, "side": "bid", "premium": 100_000}],
    )

    scan = scan_bot_flow_tape(day, asof=pd.Timestamp("2026-01-02").date(), point_in_time=True)

    summary = scan["ticker_summary"].set_index("ticker")
    assert summary.loc["AAA", "net_premium"] == 1000
    assert summary.loc["AAA", "total_premium"] == 1000


def test_find_export_bundle_recovers_nested_bot_split_directory(tmp_path: Path) -> None:
    day = tmp_path / "2026-04-23"
    split_dir = day / "bot-eod-split"
    split_dir.mkdir(parents=True)
    for part in (1, 2):
        _write_zip(
            split_dir / f"bot-eod-report-2026-04-23.part-{part:02d}-of-02.zip",
            [{"underlying_symbol": "AAA", "premium": part * 100}],
        )

    bundle = find_export_bundle(day, "bot-eod-report-")

    assert [path.name for path in bundle] == [
        "bot-eod-report-2026-04-23.part-01-of-02.zip",
        "bot-eod-report-2026-04-23.part-02-of-02.zip",
    ]
