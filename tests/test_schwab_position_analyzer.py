import unittest
from unittest.mock import MagicMock, patch
import datetime as dt


class TestGetAccountPositions(unittest.TestCase):
    """Test get_account_positions returns structured position data."""

    @patch("uwos.schwab_auth.SchwabLiveDataService.connect")
    @patch("uwos.schwab_auth.SchwabLiveDataService.get_account_hash")
    def test_returns_positions_and_balances(self, mock_hash, mock_connect):
        from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

        mock_hash.return_value = "HASH123"
        mock_client = MagicMock()
        mock_connect.return_value = mock_client

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "securitiesAccount": {
                "currentBalances": {
                    "liquidationValue": 45000.0,
                    "cashBalance": 15000.0,
                },
                "positions": [
                    {
                        "shortQuantity": 2.0,
                        "averagePrice": 3.50,
                        "currentDayProfitLoss": 20.0,
                        "currentDayProfitLossPercentage": 2.5,
                        "marketValue": -480.0,
                        "instrument": {
                            "assetType": "OPTION",
                            "symbol": "AAPL  260417P00200000",
                            "putCall": "PUT",
                            "underlyingSymbol": "AAPL",
                        },
                    }
                ],
            }
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.get_account.return_value = mock_response

        config = SchwabAuthConfig(
            api_key="fake", app_secret="fake", token_path="/tmp/fake_token.json"
        )
        svc = SchwabLiveDataService(config=config)
        result = svc.get_account_positions(account_index=0)

        self.assertIn("balances", result)
        self.assertIn("positions", result)
        self.assertEqual(result["balances"]["total_value"], 45000.0)
        self.assertEqual(result["balances"]["cash"], 15000.0)
        self.assertEqual(len(result["positions"]), 1)
        pos = result["positions"][0]
        self.assertEqual(pos["symbol"], "AAPL  260417P00200000")
        self.assertEqual(pos["asset_type"], "OPTION")
        self.assertEqual(pos["underlying"], "AAPL")
        self.assertEqual(pos["qty"], -2)  # short 2
        self.assertEqual(pos["short_qty"], 2.0)


class TestComputeRiskMetrics(unittest.TestCase):
    """Test computed risk metrics for option positions."""

    def test_short_put_metrics(self):
        from uwos.schwab_position_analyzer import compute_risk_metrics

        position = {
            "symbol": "AAPL  260417P00200000",
            "asset_type": "OPTION",
            "put_call": "PUT",
            "qty": -2,
            "short_qty": 2,
            "long_qty": 0,
            "avg_cost": 3.50,
            "market_value": -480.0,
        }
        greeks = {"delta": -0.25, "gamma": 0.02, "theta": -0.05, "vega": 0.12, "iv": 0.32}
        underlying_price = 218.50
        strike = 200.0
        expiry = dt.date(2026, 4, 17)
        entry_date = dt.date(2026, 2, 20)
        today = dt.date(2026, 3, 7)

        result = compute_risk_metrics(
            position=position, greeks=greeks, underlying_price=underlying_price,
            strike=strike, expiry=expiry, entry_date=entry_date, today=today,
        )

        # Short put: theta positive for seller
        self.assertGreater(result["theta_pnl_per_day"], 0)
        # Breakeven for short put = strike - premium = 200 - 3.50 = 196.50
        self.assertAlmostEqual(result["breakeven"], 196.50, places=2)
        self.assertGreater(result["distance_to_breakeven_pct"], 0)
        # Prob profit for short put (credit) = 1 - abs(delta) = 0.75
        self.assertAlmostEqual(result["prob_profit"], 0.75, places=2)
        # Max profit = premium * qty * 100 = 3.50 * 2 * 100 = 700
        self.assertAlmostEqual(result["max_profit"], 700.0, places=0)
        self.assertEqual(result["dte"], 41)
        self.assertEqual(result["days_held"], 15)

    def test_long_call_metrics(self):
        from uwos.schwab_position_analyzer import compute_risk_metrics

        position = {
            "symbol": "MSFT  260320C00400000",
            "asset_type": "OPTION",
            "put_call": "CALL",
            "qty": 1,
            "short_qty": 0,
            "long_qty": 1,
            "avg_cost": 5.00,
            "market_value": 650.0,
        }
        greeks = {"delta": 0.55, "gamma": 0.03, "theta": -0.08, "vega": 0.15, "iv": 0.28}
        underlying_price = 405.0
        strike = 400.0
        expiry = dt.date(2026, 3, 20)
        entry_date = dt.date(2026, 3, 1)
        today = dt.date(2026, 3, 7)

        result = compute_risk_metrics(
            position=position, greeks=greeks, underlying_price=underlying_price,
            strike=strike, expiry=expiry, entry_date=entry_date, today=today,
        )

        self.assertLess(result["theta_pnl_per_day"], 0)
        self.assertAlmostEqual(result["breakeven"], 405.0, places=2)
        self.assertAlmostEqual(result["prob_profit"], 0.55, places=2)
        self.assertAlmostEqual(result["max_loss"], 500.0, places=0)

    def test_equity_position(self):
        from uwos.schwab_position_analyzer import compute_risk_metrics

        position = {
            "symbol": "AAPL",
            "asset_type": "EQUITY",
            "put_call": "",
            "qty": 100,
            "short_qty": 0,
            "long_qty": 100,
            "avg_cost": 210.0,
            "market_value": 21850.0,
        }

        result = compute_risk_metrics(
            position=position, greeks=None, underlying_price=218.50,
            strike=None, expiry=None, entry_date=dt.date(2026, 2, 15),
            today=dt.date(2026, 3, 7),
        )

        self.assertIsNone(result["theta_pnl_per_day"])
        self.assertIsNone(result["prob_profit"])
        self.assertEqual(result["days_held"], 20)
        self.assertAlmostEqual(result["unrealized_pnl"], 850.0, places=0)
        self.assertAlmostEqual(result["unrealized_pnl_pct"], 4.05, places=1)


class TestFetchYfinanceContext(unittest.TestCase):
    """Test yfinance enrichment — mock yfinance to avoid network calls."""

    @patch("uwos.schwab_position_analyzer.yf.Ticker")
    def test_returns_expected_fields(self, mock_ticker_cls):
        from uwos.schwab_position_analyzer import fetch_yfinance_context
        import numpy as np
        import pandas as pd

        mock_ticker = MagicMock()

        mock_ticker.info = {"sector": "Technology"}
        mock_ticker.calendar = {"Earnings Date": [dt.datetime(2026, 4, 24)]}

        # Price history: 260 trading days
        dates = pd.date_range(end="2026-03-07", periods=260, freq="B")
        prices = pd.Series(np.linspace(180, 218.5, 260), index=dates, name="Close")
        hist_df = pd.DataFrame({"Close": prices, "High": prices + 2, "Low": prices - 2})
        mock_ticker.history.return_value = hist_df

        # SPY for correlation
        mock_spy = MagicMock()
        spy_prices = pd.Series(np.linspace(450, 500, 260), index=dates, name="Close")
        spy_df = pd.DataFrame({"Close": spy_prices})
        mock_spy.history.return_value = spy_df
        mock_spy.info = {}
        mock_spy.calendar = {}

        def ticker_factory(sym):
            return mock_spy if sym == "SPY" else mock_ticker
        mock_ticker_cls.side_effect = ticker_factory

        result = fetch_yfinance_context("AAPL", current_iv=0.32, today=dt.date(2026, 3, 7))

        self.assertEqual(result["sector"], "Technology")
        self.assertIn("earnings_date", result)
        self.assertIn("iv_rank", result)
        self.assertIn("hv_20d", result)
        self.assertIn("ma_50d", result)
        self.assertIn("ma_200d", result)
        self.assertIn("spy_correlation_20d", result)
        self.assertIn("support_levels", result)
        self.assertIn("resistance_levels", result)
        self.assertIsNotNone(result["hv_20d"])
        self.assertIsNotNone(result["ma_50d"])


class TestMatchEntryDetails(unittest.TestCase):
    """Test matching open positions to trade history for entry date/price."""

    def test_matches_sell_to_open(self):
        from uwos.schwab_position_analyzer import match_entry_details

        positions = [
            {"symbol": "AAPL  260417P00200000", "qty": -2, "asset_type": "OPTION"},
        ]
        transactions = [
            {
                "transactionDate": "2026-02-20T10:30:00+0000",
                "transferItems": [
                    {
                        "instrument": {"symbol": "AAPL  260417P00200000"},
                        "amount": 2.0,
                        "price": 3.50,
                        "positionEffect": "OPENING",
                    }
                ],
            },
            {
                "transactionDate": "2026-01-15T10:30:00+0000",
                "transferItems": [
                    {
                        "instrument": {"symbol": "MSFT  260320C00400000"},
                        "amount": 1.0,
                        "price": 5.00,
                        "positionEffect": "OPENING",
                    }
                ],
            },
        ]

        result = match_entry_details(positions, transactions)
        self.assertIn("AAPL  260417P00200000", result)
        entry = result["AAPL  260417P00200000"]
        self.assertEqual(entry["entry_date"], "2026-02-20")
        self.assertAlmostEqual(entry["entry_price"], 3.50)

    def test_matches_trade_date_payload(self):
        from uwos.schwab_position_analyzer import match_entry_details

        positions = [
            {"symbol": "INTC  260417P00044000", "qty": -1, "asset_type": "OPTION"},
        ]
        transactions = [
            {
                "tradeDate": "2026-03-17T15:36:25+0000",
                "transferItems": [
                    {
                        "instrument": {"symbol": "INTC  260417P00044000"},
                        "amount": -1.0,
                        "price": 2.88,
                        "positionEffect": "OPENING",
                    }
                ],
            }
        ]

        result = match_entry_details(positions, transactions)
        entry = result["INTC  260417P00044000"]
        self.assertEqual(entry["entry_date"], "2026-03-17")
        self.assertAlmostEqual(entry["entry_price"], 2.88)

    def test_no_match_returns_empty(self):
        from uwos.schwab_position_analyzer import match_entry_details

        positions = [
            {"symbol": "XYZ  260417P00050000", "qty": -1, "asset_type": "OPTION"},
        ]
        result = match_entry_details(positions, [])
        self.assertNotIn("XYZ  260417P00050000", result)


class TestAnalyzePositionsDataSources(unittest.TestCase):
    """Test trade-desk analysis defaults to Schwab-only context."""

    def test_yfinance_context_is_opt_in(self):
        from uwos.schwab_position_analyzer import analyze_positions

        svc = MagicMock()
        svc.get_account_positions.return_value = {
            "balances": {"total_value": 10000.0, "cash": 1000.0},
            "positions": [
                {
                    "symbol": "MSFT",
                    "underlying": "MSFT",
                    "asset_type": "EQUITY",
                    "put_call": "",
                    "qty": 10,
                    "avg_cost": 300.0,
                    "market_value": 3500.0,
                }
            ],
        }
        svc.get_trade_history.return_value = []
        svc.get_quotes.return_value = {
            "MSFT": {"quote": {"lastPrice": 350.0, "netPercentChangeInDouble": 0.1}}
        }

        with patch("uwos.schwab_position_analyzer.fetch_yfinance_context") as mock_yf:
            result = analyze_positions(svc=svc, days=90)

        mock_yf.assert_not_called()
        self.assertFalse(result["context_sources"]["external_yfinance"])
        self.assertEqual(result["context_sources"]["positions"], "schwab")

        with patch("uwos.schwab_position_analyzer.fetch_yfinance_context", return_value={}) as mock_yf:
            result = analyze_positions(svc=svc, days=90, include_yfinance=True)

        mock_yf.assert_called_once()
        self.assertTrue(result["context_sources"]["external_yfinance"])


if __name__ == "__main__":
    unittest.main()
