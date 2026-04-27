from uwos.trade_ideas import _format_trade_legs, _state_underlying, screen_momentum_breakouts


class FakeQuoteService:
    def get_quotes(self, tickers):
        quotes = {
            "BRKO": {
                "quote": {
                    "lastPrice": 99.0,
                    "52WeekHigh": 100.0,
                    "highPrice": 100.2,
                    "netPercentChange": 3.2,
                    "totalVolume": 2_000_000,
                },
                "fundamental": {
                    "avg10DaysVolume": 1_000_000,
                    "sharesOutstanding": 250_000_000,
                    "eps": 5.0,
                    "peRatio": 22.0,
                },
                "reference": {"description": "Breakout Corp", "optionable": True},
            },
            "TINY": {
                "quote": {
                    "lastPrice": 12.0,
                    "52WeekHigh": 12.5,
                    "highPrice": 12.4,
                    "netPercentChange": 4.0,
                    "totalVolume": 900_000,
                },
                "fundamental": {
                    "avg10DaysVolume": 500_000,
                    "sharesOutstanding": 50_000_000,
                },
                "reference": {"description": "Tiny Corp", "optionable": True},
            },
            "FUND": {
                "quote": {
                    "lastPrice": 50.0,
                    "52WeekHigh": 51.0,
                    "highPrice": 51.0,
                    "netPercentChange": 2.5,
                    "totalVolume": 2_000_000,
                },
                "fundamental": {
                    "avg10DaysVolume": 1_000_000,
                    "sharesOutstanding": 500_000_000,
                },
                "reference": {"description": "Example ETF Fund", "optionable": True},
            },
        }
        return {ticker: quotes[ticker] for ticker in tickers if ticker in quotes}


def test_screen_momentum_breakouts_filters_and_scores_large_optionable_stocks():
    results = screen_momentum_breakouts(FakeQuoteService(), ["BRKO", "TINY", "FUND"])

    assert [row["ticker"] for row in results] == ["BRKO"]
    assert results[0]["breakout_score"] > 20
    assert results[0]["rel_volume"] == 2.0


def test_format_trade_legs_uses_debit_call_action_order():
    legs = _format_trade_legs({
        "strategy": "Bull Call Debit",
        "long_strike": 100,
        "short_strike": 110,
    })

    assert legs == "Buy $100C / Sell $110C"


def test_format_trade_legs_uses_credit_put_action_order():
    legs = _format_trade_legs({
        "strategy": "Bull Put Credit",
        "long_strike": 450,
        "short_strike": 460,
    })

    assert legs == "Sell $460P / Buy $450P"


def test_state_underlying_handles_spreads_and_saved_values():
    assert _state_underlying("SPREAD:C:2026-05-15:CALL:C1|C2", {}) == "C"
    assert _state_underlying("ignored", {"underlying": "NVDA"}) == "NVDA"
    assert _state_underlying("AAPL  260515C00200000", {}) == "AAPL"
