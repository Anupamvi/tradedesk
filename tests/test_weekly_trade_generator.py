import datetime as dt

import pandas as pd

from uwos.weekly_trade_generator import (
    GeneratedSetup,
    GeneratorConfig,
    build_vertical_candidates,
    parse_occ_symbol,
    select_portfolio,
)


def test_parse_occ_symbol_accepts_spaced_occ_symbols():
    parsed = parse_occ_symbol("AAPL  260220P00100000")

    assert parsed == ("AAPL", dt.date(2026, 2, 20), "P", 100.0)


def test_build_vertical_candidates_creates_conservative_bull_put_credit():
    asof = dt.date(2026, 1, 20)
    screener = pd.Series(
        {
            "ticker": "AAPL",
            "close": 100.0,
            "prev_close": 99.0,
            "iv30d": 0.30,
            "iv_rank": 45.0,
            "bullish_premium": 2_000_000.0,
            "bearish_premium": 1_000_000.0,
            "sector": "Technology",
        }
    )
    quotes = pd.DataFrame(
        [
            {
                "option_symbol": "AAPL260220P00095000",
                "expiry": dt.date(2026, 2, 20),
                "right": "P",
                "strike": 95.0,
                "dte": 31,
                "bid": 1.25,
                "ask": 1.35,
                "open_interest": 500,
                "volume": 100,
                "spread": 0.10,
                "spread_pct": 0.077,
            },
            {
                "option_symbol": "AAPL260220P00090000",
                "expiry": dt.date(2026, 2, 20),
                "right": "P",
                "strike": 90.0,
                "dte": 31,
                "bid": 0.15,
                "ask": 0.25,
                "open_interest": 400,
                "volume": 80,
                "spread": 0.10,
                "spread_pct": 0.50,
            },
        ]
    )
    cfg = GeneratorConfig(max_leg_spread_pct=0.60)

    rows = build_vertical_candidates(asof, screener, quotes, cfg)

    assert len(rows) == 1
    row = rows[0]
    assert row.strategy == "Bull Put Credit"
    assert row.short_leg == "AAPL260220P00095000"
    assert row.long_leg == "AAPL260220P00090000"
    assert row.entry_net == 1.0
    assert row.max_loss == 400.0
    assert row.pop_estimate > 0.60


def test_select_portfolio_enforces_ticker_sector_direction_caps():
    cfg = GeneratorConfig(max_candidates_per_day=3, max_per_ticker=1, max_per_sector=1, max_per_direction=2)
    rows = [
        GeneratedSetup(
            signal_date="2026-01-20",
            ticker="AAPL",
            strategy="Bull Put Credit",
            expiry="2026-02-20",
            sector="Technology",
            direction="bullish",
            score=100,
        ),
        GeneratedSetup(
            signal_date="2026-01-20",
            ticker="MSFT",
            strategy="Bull Put Credit",
            expiry="2026-02-20",
            sector="Technology",
            direction="bullish",
            score=99,
        ),
        GeneratedSetup(
            signal_date="2026-01-20",
            ticker="JPM",
            strategy="Bull Put Credit",
            expiry="2026-02-20",
            sector="Financial Services",
            direction="bullish",
            score=98,
        ),
        GeneratedSetup(
            signal_date="2026-01-20",
            ticker="XOM",
            strategy="Bear Call Credit",
            expiry="2026-02-20",
            sector="Energy",
            direction="bearish",
            score=97,
        ),
    ]

    selected = select_portfolio(rows, cfg)

    assert [row.ticker for row in selected] == ["AAPL", "JPM", "XOM"]
