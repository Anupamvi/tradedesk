import json
import unittest
from datetime import date, datetime, timedelta, timezone

from cultra.schwab import (
    OptionChain,
    OptionQuote,
    PriceBar,
    Quote,
    SchwabBoundaryError,
    SchwabHTTPProvider,
    SchwabMarketDataBoundary,
    TokenFileReference,
)
from cultra.gateway import TransportResponse


NOW = datetime(2026, 8, 30, 20, tzinfo=timezone.utc)


class FakeProvider:
    def __init__(self):
        self.calls = []

    def fetch_quotes(self, symbols):
        self.calls.append(("quotes", tuple(symbols)))
        return {
            symbol: Quote(symbol, 99.0, 101.0, 100.0, NOW)
            for symbol in symbols
        }

    def fetch_option_chain(self, symbol, *, from_date, to_date):
        self.calls.append(("chain", symbol, from_date, to_date))
        quote = Quote(symbol, 99.0, 101.0, 100.0, NOW)
        contract = OptionQuote(
            "%s260918C00100000" % symbol,
            symbol,
            date(2026, 9, 18),
            100.0,
            "CALL",
            4.0,
            4.2,
            NOW,
            volume=100,
            open_interest=500,
            delta=0.5,
        )
        return OptionChain(symbol, quote, (contract,), NOW)

    def fetch_price_history(self, symbol, *, start, end):
        self.calls.append(("history", symbol, start, end))
        return (
            PriceBar(symbol, NOW - timedelta(days=1), 99, 102, 98, 100, 1000),
            PriceBar(symbol, NOW, 100, 103, 99, 101, 1100),
        )


class FakeTokens:
    def __init__(self):
        self.force_refreshes = []

    def access_token(self, *, force_refresh=False):
        self.force_refreshes.append(force_refresh)
        return "test-access-token"


class FakeHTTPTransport:
    def __init__(self):
        self.calls = []

    def send(self, **kwargs):
        self.calls.append(kwargs)
        path = kwargs["path"]
        millis = int(NOW.timestamp() * 1000)
        if path.endswith("/quotes"):
            payload = {
                "SPY": {
                    "quote": {
                        "bidPrice": 699.0,
                        "askPrice": 699.2,
                        "lastPrice": 699.1,
                        "quoteTime": millis,
                        "totalVolume": 123456,
                        "closePrice": 698.0,
                        "netPercentChange": 0.15,
                        "52WeekHigh": 710.0,
                        "52WeekLow": 500.0,
                    }
                }
            }
        elif path.endswith("/chains"):
            payload = {
                "underlyingPrice": 699.1,
                "underlying": {
                    "bid": 699.0,
                    "ask": 699.2,
                    "last": 699.1,
                    "quoteTime": millis,
                },
                "callExpDateMap": {
                    "2026-10-16:47": {
                        "700.0": [
                            {
                                "symbol": "SPY   261016C00700000",
                                "strikePrice": 700.0,
                                "bid": 20.0,
                                "ask": 20.2,
                                "quoteTimeInLong": millis,
                                "totalVolume": 50,
                                "openInterest": 500,
                                "delta": 0.52,
                            }
                        ]
                    }
                },
                "putExpDateMap": {},
            }
        else:
            payload = {"candles": []}
        return TransportResponse(
            200,
            json.dumps(payload).encode("utf-8"),
            (("Content-Type", "application/json"),),
            1.0,
        )


class SchwabBoundaryTests(unittest.TestCase):
    def test_exposes_only_validated_read_only_market_data(self):
        provider = FakeProvider()
        boundary = SchwabMarketDataBoundary(provider)
        quotes = boundary.quotes(["aapl", "AAPL", "msft"])
        self.assertEqual(set(quotes), {"AAPL", "MSFT"})
        chain = boundary.option_chain(
            "AAPL",
            from_date=date(2026, 9, 1),
            to_date=date(2026, 10, 1),
        )
        history = boundary.price_history(
            "AAPL",
            start=date(2026, 8, 1),
            end=date(2026, 8, 30),
        )
        self.assertEqual(chain.underlying, "AAPL")
        self.assertEqual(len(history), 2)
        self.assertFalse(hasattr(boundary, "orders"))
        self.assertFalse(hasattr(boundary, "accounts"))
        self.assertEqual([call[0] for call in provider.calls], ["quotes", "chain", "history"])

    def test_token_reference_does_not_accept_an_arbitrary_path(self):
        with self.assertRaises(SchwabBoundaryError):
            TokenFileReference("/tmp/not-a-token.json")

    def test_rejects_bad_quotes_and_ranges_before_provider_call(self):
        provider = FakeProvider()
        boundary = SchwabMarketDataBoundary(provider)
        with self.assertRaises(SchwabBoundaryError):
            boundary.quotes([])
        with self.assertRaises(SchwabBoundaryError):
            boundary.option_chain(
                "AAPL",
                from_date=date(2026, 10, 1),
                to_date=date(2026, 9, 1),
            )
        with self.assertRaises(SchwabBoundaryError):
            Quote("AAPL", 10, 9, None, NOW)
        with self.assertRaises(SchwabBoundaryError):
            Quote("AAPL", 9, 10, None, datetime(2026, 8, 30, 12))
        self.assertEqual(provider.calls, [])

    def test_rejects_nonchronological_history(self):
        class ReverseProvider(FakeProvider):
            def fetch_price_history(self, symbol, *, start, end):
                bars = super().fetch_price_history(symbol, start=start, end=end)
                return tuple(reversed(bars))

        with self.assertRaises(SchwabBoundaryError):
            SchwabMarketDataBoundary(ReverseProvider()).price_history(
                "AAPL", start=date(2026, 8, 1), end=date(2026, 8, 30)
            )

    def test_rejects_occ_root_that_disagrees_with_underlying(self):
        with self.assertRaises(SchwabBoundaryError):
            OptionQuote(
                "MSFT260918C00100000",
                "AAPL",
                date(2026, 9, 18),
                100.0,
                "CALL",
                4.0,
                4.2,
                NOW,
            )

    def test_rejects_contract_outside_requested_expiration_range(self):
        boundary = SchwabMarketDataBoundary(FakeProvider())
        with self.assertRaises(SchwabBoundaryError):
            boundary.option_chain(
                "AAPL",
                from_date=date(2026, 10, 1),
                to_date=date(2026, 10, 31),
            )

    def test_concrete_http_provider_parses_only_read_only_market_data(self):
        transport = FakeHTTPTransport()
        tokens = FakeTokens()
        boundary = SchwabMarketDataBoundary(SchwabHTTPProvider(tokens, transport))
        quotes = boundary.quotes(["SPY"])
        chain = boundary.option_chain(
            "SPY", from_date=date(2026, 10, 1), to_date=date(2026, 10, 31)
        )
        self.assertEqual(699.0, quotes["SPY"].bid)
        self.assertEqual(123456, quotes["SPY"].total_volume)
        self.assertEqual(710.0, quotes["SPY"].week52_high)
        self.assertEqual("SPY261016C00700000", chain.contracts[0].occ_symbol)
        self.assertEqual(0.52, chain.contracts[0].delta)
        self.assertEqual(
            ["/marketdata/v1/quotes", "/marketdata/v1/chains"],
            [item["path"] for item in transport.calls],
        )
        self.assertEqual([False, False], tokens.force_refreshes)


if __name__ == "__main__":
    unittest.main()
