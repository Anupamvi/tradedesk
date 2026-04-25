import zipfile

import pandas as pd
import pytest

from uwos.whale_source import (
    find_bot_eod_source,
    find_whale_markdown_source,
    load_yes_prime_whale_flow,
)


def _config():
    return {
        "gates": {
            "exclude_etfs": True,
            "exclude_issue_types": ["ETF"],
            "min_credit_pct_width": 0.30,
            "max_credit_pct_width": 0.55,
            "max_debit_pct_width": 0.45,
            "min_leg_open_interest": 100,
            "max_strike_distance_pct": 0.80,
            "width_tiers": [
                {"min_price": 0, "max_price": 25, "default_width": 2.5},
                {"min_price": 25, "max_price": 150, "default_width": 10},
            ],
        },
        "shield": {"dte_range": [28, 56], "use_anchor_whitelist": False},
        "fire": {"dte_range": [21, 70]},
    }


def test_load_yes_prime_whale_flow_streams_bot_zip_and_aggregates_symbols(tmp_path):
    rows = [
        {
            "executed_at": "2026-04-23 13:30:00+00",
            "underlying_symbol": "AAPL",
            "option_chain_id": "AAPL260515C00300000",
            "side": "ask",
            "strike": 105,
            "option_type": "call",
            "expiry": "2026-05-15",
            "underlying_price": 100,
            "price": 2.0,
            "size": 10,
            "premium": 2000,
            "open_interest": 500,
            "implied_volatility": 0.2,
            "delta": 0.3,
            "equity_type": "Common Stock",
        },
        {
            "executed_at": "2026-04-23 13:30:00+00",
            "underlying_symbol": "AAPL",
            "option_chain_id": "AAPL260515P00095000",
            "side": "bid",
            "strike": 95,
            "option_type": "put",
            "expiry": "2026-05-21",
            "underlying_price": 100,
            "price": 4.0,
            "size": 3,
            "premium": 1200,
            "open_interest": 250,
            "implied_volatility": 0.25,
            "delta": -0.25,
            "equity_type": "Common Stock",
        },
        {
            "executed_at": "2026-04-23 13:30:00+00",
            "underlying_symbol": "MSFT",
            "option_chain_id": "MSFT260515P00370000",
            "side": "ask",
            "strike": 90,
            "option_type": "put",
            "expiry": "2026-05-15",
            "underlying_price": 100,
            "price": 1.5,
            "size": 5,
            "premium": 750,
            "open_interest": 150,
            "implied_volatility": 0.3,
            "delta": -0.2,
            "equity_type": "Common Stock",
        },
        {
            "executed_at": "2026-04-23 13:30:00+00",
            "underlying_symbol": "SPY",
            "option_chain_id": "SPY260515C00700000",
            "side": "ask",
            "strike": 105,
            "option_type": "call",
            "expiry": "2026-05-15",
            "underlying_price": 100,
            "price": 2.0,
            "size": 100,
            "premium": 20000,
            "open_interest": 1000,
            "implied_volatility": 0.2,
            "delta": 0.3,
            "equity_type": "ETF",
        },
    ]
    csv_path = tmp_path / "bot-eod-report-2026-04-23.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    zip_path = tmp_path / "bot-eod-report-2026-04-23.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname=csv_path.name)

    (tmp_path / "whale-2026-04-23.md").write_text("not used", encoding="utf-8")

    assert find_bot_eod_source(tmp_path, "2026-04-23") == zip_path

    flow = load_yes_prime_whale_flow(zip_path, _config(), chunksize=2)

    assert flow.total_rows == 4
    assert flow.yes_prime_rows == 3
    assert flow.symbol_summary["underlying_symbol"].tolist() == ["AAPL", "MSFT"]

    aapl = flow.symbol_summary.set_index("underlying_symbol").loc["AAPL"]
    assert aapl["count"] == 2
    assert aapl["total_premium"] == 3200
    assert aapl["call_premium"] == 2000
    assert aapl["put_premium"] == 1200
    assert aapl["bull_proxy_premium"] == 3200

    tables = flow.as_rank_tables()
    assert list(tables) == ["Top Symbols by Total Premium (Yes-Prime)"]


def test_find_bot_eod_source_does_not_use_wrong_date(tmp_path):
    wrong_date = tmp_path / "bot-eod-report-2026-04-22.zip"
    with zipfile.ZipFile(wrong_date, "w") as zf:
        zf.writestr("bot-eod-report-2026-04-22.csv", "executed_at,underlying_symbol\n")

    with pytest.raises(FileNotFoundError):
        find_bot_eod_source(tmp_path, "2026-04-23")

    assert find_bot_eod_source(tmp_path) == wrong_date


def test_find_whale_markdown_source_does_not_use_wrong_date(tmp_path):
    wrong_date = tmp_path / "whale-2026-04-22.md"
    wrong_date.write_text("## Top Symbols by Total Premium (Yes-Prime)\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        find_whale_markdown_source(tmp_path, "2026-04-23")

    assert find_whale_markdown_source(tmp_path) == wrong_date


def test_load_yes_prime_whale_flow_raises_for_missing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_yes_prime_whale_flow(tmp_path / "bot-eod-report-2026-04-23.zip", _config())
