# Trade Analysis Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an agent that fetches open Schwab positions, enriches them with live prices/Greeks/risk metrics/news/macro, and produces per-position HOLD/CLOSE/ROLL verdicts biased toward patience.

**Architecture:** Python module (`schwab_position_analyzer.py`) collects all Schwab + yfinance data and computes risk metrics into a JSON file. Claude skill reads JSON, runs parallel WebSearch for news/sentiment/macro, then generates a markdown analysis report with per-position verdict cards.

**Tech Stack:** schwab-py, yfinance, Python 3.13, Claude Code skills, WebSearch

**Design doc:** `docs/plans/2026-03-07-trade-analysis-agent-design.md`

---

### Task 1: Add `get_account_positions()` to SchwabLiveDataService

**Files:**
- Modify: `uwos/schwab_auth.py` (insert after `get_account_hash` method, ~line 289)
- Test: `tests/test_schwab_position_analyzer.py` (new file)

**Step 1: Write the failing test**

Create `tests/test_schwab_position_analyzer.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

**Step 2: Run test to verify it fails**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py::TestGetAccountPositions::test_returns_positions_and_balances -v`
Expected: FAIL — `AttributeError: 'SchwabLiveDataService' object has no attribute 'get_account_positions'`

**Step 3: Write minimal implementation**

Add to `uwos/schwab_auth.py` after `get_account_hash()` (after line 288):

```python
def get_account_positions(self, account_index: int = 0) -> Dict[str, Any]:
    """Fetch current account positions and balances from Schwab."""
    from schwab.client import Client

    account_hash = self.get_account_hash(account_index)
    client = self.connect()
    try:
        response = client.get_account(
            account_hash, fields=[Client.Account.Fields.POSITIONS]
        )
    except Exception as exc:
        if _is_refresh_token_error(exc):
            raise RuntimeError(
                "Schwab token refresh failed (stale/revoked refresh token). "
                "Re-auth once with: python -m uwos.schwab_quotes --manual-auth "
                "--symbols-csv AAPL --chain-symbols-csv AAPL --strike-count 2"
            ) from exc
        raise
    response.raise_for_status()
    data = response.json()

    acct = data.get("securitiesAccount", data)
    balances = acct.get("currentBalances", {})
    raw_positions = acct.get("positions", [])

    positions = []
    for pos in raw_positions:
        instrument = pos.get("instrument", {})
        positions.append({
            "symbol": instrument.get("symbol", ""),
            "asset_type": instrument.get("assetType", ""),
            "underlying": instrument.get("underlyingSymbol", ""),
            "put_call": instrument.get("putCall", ""),
            "qty": pos.get("longQuantity", 0) - pos.get("shortQuantity", 0),
            "short_qty": pos.get("shortQuantity", 0),
            "long_qty": pos.get("longQuantity", 0),
            "avg_cost": pos.get("averagePrice"),
            "market_value": pos.get("marketValue"),
            "day_pnl": pos.get("currentDayProfitLoss"),
            "day_pnl_pct": pos.get("currentDayProfitLossPercentage"),
        })

    return {
        "balances": {
            "total_value": _safe_float(balances.get("liquidationValue")),
            "cash": _safe_float(balances.get("cashBalance")),
        },
        "positions": positions,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py::TestGetAccountPositions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uwos/schwab_auth.py tests/test_schwab_position_analyzer.py
git commit -m "feat: add get_account_positions to SchwabLiveDataService"
```

---

### Task 2: Build `compute_risk_metrics()` — pure math, no API calls

**Files:**
- Create: `uwos/schwab_position_analyzer.py` (new file — replaces the simple trade history fetcher concept with full analyzer)
- Test: `tests/test_schwab_position_analyzer.py` (append)

**Step 1: Write the failing tests**

Append to `tests/test_schwab_position_analyzer.py`:

```python
class TestComputeRiskMetrics(unittest.TestCase):
    """Test computed risk metrics for option positions."""

    def test_short_put_metrics(self):
        from uwos.schwab_position_analyzer import compute_risk_metrics

        position = {
            "symbol": "AAPL  260417P00200000",
            "asset_type": "OPTION",
            "put_call": "PUT",
            "qty": -2,  # short 2 puts
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
            position=position,
            greeks=greeks,
            underlying_price=underlying_price,
            strike=strike,
            expiry=expiry,
            entry_date=entry_date,
            today=today,
        )

        # Short put: theta positive for seller (theta is negative, qty is negative => positive P&L)
        self.assertGreater(result["theta_pnl_per_day"], 0)
        # Breakeven for short put = strike - premium = 200 - 3.50 = 196.50
        self.assertAlmostEqual(result["breakeven"], 196.50, places=2)
        # Distance to breakeven
        self.assertGreater(result["distance_to_breakeven_pct"], 0)
        # Prob profit for short put (credit) = 1 - abs(delta) = 0.75
        self.assertAlmostEqual(result["prob_profit"], 0.75, places=2)
        # Max profit = premium * qty * 100 = 3.50 * 2 * 100 = 700
        self.assertAlmostEqual(result["max_profit"], 700.0, places=0)
        # DTE
        self.assertEqual(result["dte"], 41)
        # Days held
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
            position=position,
            greeks=greeks,
            underlying_price=underlying_price,
            strike=strike,
            expiry=expiry,
            entry_date=entry_date,
            today=today,
        )

        # Long call: theta is negative for buyer
        self.assertLess(result["theta_pnl_per_day"], 0)
        # Breakeven for long call = strike + premium = 400 + 5 = 405
        self.assertAlmostEqual(result["breakeven"], 405.0, places=2)
        # Prob profit for debit = abs(delta) = 0.55
        self.assertAlmostEqual(result["prob_profit"], 0.55, places=2)
        # Max loss for long call = premium * qty * 100 = 500
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
            position=position,
            greeks=None,
            underlying_price=218.50,
            strike=None,
            expiry=None,
            entry_date=dt.date(2026, 2, 15),
            today=dt.date(2026, 3, 7),
        )

        self.assertIsNone(result["theta_pnl_per_day"])
        self.assertIsNone(result["prob_profit"])
        self.assertEqual(result["days_held"], 20)
        self.assertAlmostEqual(result["unrealized_pnl"], 850.0, places=0)
        self.assertAlmostEqual(result["unrealized_pnl_pct"], 4.05, places=1)
```

**Step 2: Run tests to verify they fail**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py::TestComputeRiskMetrics -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'uwos.schwab_position_analyzer'`

**Step 3: Write implementation**

Create `uwos/schwab_position_analyzer.py`:

```python
#!/usr/bin/env python3
"""Fetch open positions from Schwab, enrich with live data and risk metrics."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yfinance as yf

from uwos.schwab_auth import (
    SchwabAuthConfig,
    SchwabLiveDataService,
    occ_underlying_symbol,
    schwab_occ_to_compact_symbol,
    _safe_float,
)


def compute_risk_metrics(
    position: Dict[str, Any],
    greeks: Optional[Dict[str, Any]],
    underlying_price: float,
    strike: Optional[float],
    expiry: Optional[dt.date],
    entry_date: Optional[dt.date],
    today: Optional[dt.date] = None,
) -> Dict[str, Any]:
    """Compute derived risk metrics for a single position."""
    today = today or dt.date.today()
    asset_type = position.get("asset_type", "")
    put_call = position.get("put_call", "")
    qty = position.get("qty", 0)
    abs_qty = abs(qty)
    avg_cost = _safe_float(position.get("avg_cost")) or 0.0
    market_value = _safe_float(position.get("market_value")) or 0.0
    is_short = qty < 0
    is_option = asset_type == "OPTION"

    days_held = (today - entry_date).days if entry_date else None
    dte = (expiry - today).days if expiry else None

    if not is_option:
        # Equity position — simple metrics
        cost_basis = avg_cost * abs_qty
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else 0.0
        return {
            "dte": None,
            "days_held": days_held,
            "theta_pnl_per_day": None,
            "gamma_risk": None,
            "vega_exposure": None,
            "breakeven": avg_cost,
            "distance_to_breakeven_pct": ((underlying_price - avg_cost) / underlying_price * 100) if underlying_price else None,
            "prob_itm": None,
            "prob_profit": None,
            "max_profit": None,
            "max_loss": cost_basis,
            "risk_reward_ratio": None,
            "theta_risk_ratio": None,
            "pct_of_max_profit": None,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
        }

    # Option position
    delta = _safe_float(greeks.get("delta")) if greeks else None
    gamma = _safe_float(greeks.get("gamma")) if greeks else None
    theta = _safe_float(greeks.get("theta")) if greeks else None
    vega = _safe_float(greeks.get("vega")) if greeks else None

    # Theta P&L per day: theta * qty * 100 (short options: theta<0, qty<0 => positive)
    theta_pnl = (theta * qty * 100) if theta is not None else None
    gamma_risk = (gamma * abs_qty * 100) if gamma is not None else None
    vega_exposure = (vega * qty * 100) if vega is not None else None

    # Breakeven
    if strike is not None and avg_cost is not None:
        if put_call == "PUT":
            breakeven = strike - avg_cost
        else:  # CALL
            breakeven = strike + avg_cost
    else:
        breakeven = None

    distance_to_breakeven_pct = None
    if breakeven is not None and underlying_price:
        distance_to_breakeven_pct = (underlying_price - breakeven) / underlying_price * 100

    # Probability
    prob_itm = abs(delta) if delta is not None else None
    if delta is not None:
        if is_short:
            prob_profit = 1.0 - abs(delta)  # Credit trade
        else:
            prob_profit = abs(delta)  # Debit trade
    else:
        prob_profit = None

    # Max profit / loss
    premium_total = avg_cost * abs_qty * 100
    if is_short:
        max_profit = premium_total
        if put_call == "PUT" and strike is not None:
            max_loss = (strike * abs_qty * 100) - premium_total
        else:
            max_loss = None  # Naked call = theoretically unlimited
    else:
        max_loss = premium_total
        max_profit = None  # Theoretically unlimited for long calls

    # Unrealized P&L
    cost_basis = avg_cost * abs_qty * 100
    if is_short:
        unrealized_pnl = cost_basis - abs(market_value)
    else:
        unrealized_pnl = market_value - cost_basis
    unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis else 0.0

    # Derived ratios
    risk_reward = (max_loss / max_profit) if (max_profit and max_loss) else None
    theta_risk = (theta_pnl / max_loss) if (theta_pnl and max_loss and max_loss > 0) else None
    pct_of_max = (unrealized_pnl / max_profit * 100) if max_profit else None

    return {
        "dte": dte,
        "days_held": days_held,
        "theta_pnl_per_day": theta_pnl,
        "gamma_risk": gamma_risk,
        "vega_exposure": vega_exposure,
        "breakeven": breakeven,
        "distance_to_breakeven_pct": distance_to_breakeven_pct,
        "prob_itm": prob_itm,
        "prob_profit": prob_profit,
        "max_profit": max_profit,
        "max_loss": max_loss,
        "risk_reward_ratio": risk_reward,
        "theta_risk_ratio": theta_risk,
        "pct_of_max_profit": pct_of_max,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
    }
```

**Step 4: Run tests to verify they pass**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py::TestComputeRiskMetrics -v`
Expected: PASS (all 3 tests)

**Step 5: Commit**

```bash
git add uwos/schwab_position_analyzer.py tests/test_schwab_position_analyzer.py
git commit -m "feat: add compute_risk_metrics for option and equity positions"
```

---

### Task 3: Add `fetch_yfinance_context()` — earnings, IV rank, HV, support/resistance, correlation, sector

**Files:**
- Modify: `uwos/schwab_position_analyzer.py` (append function)
- Test: `tests/test_schwab_position_analyzer.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_schwab_position_analyzer.py`:

```python
class TestFetchYfinanceContext(unittest.TestCase):
    """Test yfinance enrichment — mock yfinance to avoid network calls."""

    @patch("uwos.schwab_position_analyzer.yf.Ticker")
    def test_returns_expected_fields(self, mock_ticker_cls):
        from uwos.schwab_position_analyzer import fetch_yfinance_context
        import numpy as np
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker_cls.return_value = mock_ticker

        mock_ticker.info = {"sector": "Technology"}
        mock_ticker.calendar = {"Earnings Date": [dt.datetime(2026, 4, 24)]}

        # Price history: 260 trading days
        dates = pd.date_range(end="2026-03-07", periods=260, freq="B")
        prices = pd.Series(
            np.linspace(180, 218.5, 260), index=dates, name="Close"
        )
        hist_df = pd.DataFrame({"Close": prices, "High": prices + 2, "Low": prices - 2})
        mock_ticker.history.return_value = hist_df

        # SPY for correlation
        mock_spy = MagicMock()
        mock_spy.history.return_value = pd.DataFrame(
            {"Close": pd.Series(np.linspace(450, 500, 260), index=dates)}
        )
        mock_ticker_cls.side_effect = lambda sym: mock_spy if sym == "SPY" else mock_ticker

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
```

**Step 2: Run test to verify it fails**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py::TestFetchYfinanceContext -v`
Expected: FAIL — `ImportError: cannot import name 'fetch_yfinance_context'`

**Step 3: Write implementation**

Append to `uwos/schwab_position_analyzer.py`:

```python
def fetch_yfinance_context(
    ticker: str,
    current_iv: Optional[float] = None,
    today: Optional[dt.date] = None,
) -> Dict[str, Any]:
    """Fetch enrichment data from yfinance for a single ticker."""
    today = today or dt.date.today()
    result: Dict[str, Any] = {
        "sector": None,
        "earnings_date": None,
        "days_to_earnings": None,
        "iv_rank": None,
        "iv_percentile": None,
        "hv_20d": None,
        "iv_vs_hv_spread": None,
        "ma_50d": None,
        "ma_200d": None,
        "support_levels": [],
        "resistance_levels": [],
        "spy_correlation_20d": None,
    }

    try:
        tk = yf.Ticker(ticker)

        # Sector
        info = tk.info or {}
        result["sector"] = info.get("sector")

        # Earnings date
        cal = tk.calendar
        if cal and isinstance(cal, dict):
            edates = cal.get("Earnings Date", [])
            if edates:
                edate = edates[0]
                if isinstance(edate, dt.datetime):
                    edate = edate.date()
                result["earnings_date"] = str(edate)
                result["days_to_earnings"] = (edate - today).days

        # Price history (1 year for IV rank calc, support/resistance, MAs)
        hist = tk.history(period="1y")
        if hist is None or hist.empty:
            return result

        closes = hist["Close"].dropna()
        if len(closes) < 20:
            return result

        # Historical volatility (20-day annualized)
        returns = closes.pct_change().dropna()
        hv_20d = float(returns.tail(20).std() * (252 ** 0.5))
        result["hv_20d"] = round(hv_20d, 4)

        # IV vs HV spread
        if current_iv is not None:
            result["iv_vs_hv_spread"] = round((current_iv - hv_20d) * 100, 1)

        # IV Rank: approximate using HV percentile as proxy
        # (true IV rank needs historical IV data; HV is a reasonable proxy)
        if current_iv is not None and len(returns) >= 60:
            rolling_hv = returns.rolling(20).std() * (252 ** 0.5)
            rolling_hv = rolling_hv.dropna()
            if len(rolling_hv) > 0:
                hv_min = float(rolling_hv.min())
                hv_max = float(rolling_hv.max())
                if hv_max > hv_min:
                    result["iv_rank"] = round((current_iv - hv_min) / (hv_max - hv_min) * 100, 0)
                    result["iv_percentile"] = round(
                        float((rolling_hv < current_iv).sum() / len(rolling_hv) * 100), 0
                    )

        # Moving averages
        result["ma_50d"] = round(float(closes.tail(50).mean()), 2) if len(closes) >= 50 else None
        result["ma_200d"] = round(float(closes.tail(200).mean()), 2) if len(closes) >= 200 else None

        # Support/resistance: recent 20-day swing lows/highs
        recent = hist.tail(60)
        if len(recent) >= 5:
            result["support_levels"] = sorted([
                round(float(recent["Low"].tail(20).min()), 2),
                result["ma_200d"],
            ]) if result["ma_200d"] else [round(float(recent["Low"].tail(20).min()), 2)]
            result["resistance_levels"] = sorted([
                round(float(recent["High"].tail(20).max()), 2),
            ])

        # SPY correlation (20-day)
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")
            if spy_hist is not None and not spy_hist.empty:
                spy_returns = spy_hist["Close"].pct_change().dropna()
                # Align dates
                common = returns.index.intersection(spy_returns.index)
                if len(common) >= 20:
                    corr = float(returns.loc[common].tail(20).corr(spy_returns.loc[common].tail(20)))
                    result["spy_correlation_20d"] = round(corr, 2)
        except Exception:
            pass  # Non-critical

    except Exception:
        pass  # Return partial result

    return result
```

**Step 4: Run test to verify it passes**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py::TestFetchYfinanceContext -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uwos/schwab_position_analyzer.py tests/test_schwab_position_analyzer.py
git commit -m "feat: add fetch_yfinance_context for earnings, IV rank, HV, support/resistance"
```

---

### Task 4: Add `match_entry_details()` — match open positions to trade history

**Files:**
- Modify: `uwos/schwab_position_analyzer.py` (append function)
- Test: `tests/test_schwab_position_analyzer.py` (append)

**Step 1: Write the failing test**

Append to `tests/test_schwab_position_analyzer.py`:

```python
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

    def test_no_match_returns_none(self):
        from uwos.schwab_position_analyzer import match_entry_details

        positions = [
            {"symbol": "XYZ  260417P00050000", "qty": -1, "asset_type": "OPTION"},
        ]
        result = match_entry_details(positions, [])
        self.assertNotIn("XYZ  260417P00050000", result)
```

**Step 2: Run test to verify it fails**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py::TestMatchEntryDetails -v`
Expected: FAIL — `ImportError: cannot import name 'match_entry_details'`

**Step 3: Write implementation**

Append to `uwos/schwab_position_analyzer.py`:

```python
def match_entry_details(
    positions: List[Dict[str, Any]],
    transactions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Match open positions to their opening transactions for entry date/price.

    Returns a dict keyed by position symbol with entry_date and entry_price.
    """
    # Build lookup: symbol -> earliest OPENING transaction
    entries: Dict[str, Dict[str, Any]] = {}
    # Sort transactions oldest first so we find the original open
    sorted_txns = sorted(transactions, key=lambda t: t.get("transactionDate", ""))

    position_symbols = {p["symbol"] for p in positions}

    for txn in sorted_txns:
        for item in txn.get("transferItems", []):
            instrument = item.get("instrument", {})
            symbol = instrument.get("symbol", "")
            effect = (item.get("positionEffect") or "").upper()
            if symbol in position_symbols and effect == "OPENING" and symbol not in entries:
                txn_date = str(txn.get("transactionDate", ""))[:10]
                entries[symbol] = {
                    "entry_date": txn_date,
                    "entry_price": _safe_float(item.get("price")),
                }

    return entries
```

**Step 4: Run test to verify it passes**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py::TestMatchEntryDetails -v`
Expected: PASS

**Step 5: Commit**

```bash
git add uwos/schwab_position_analyzer.py tests/test_schwab_position_analyzer.py
git commit -m "feat: add match_entry_details to link open positions with entry transactions"
```

---

### Task 5: Build the main `analyze_positions()` orchestrator and CLI

**Files:**
- Modify: `uwos/schwab_position_analyzer.py` (append orchestrator + CLI)
- No new tests for the orchestrator (it calls Schwab API — tested via integration)

**Step 1: Write the orchestrator function**

Append to `uwos/schwab_position_analyzer.py`:

```python
import re

# Regex to parse Schwab OCC option symbols: "AAPL  260417P00200000"
_OCC_RE = re.compile(r"^([A-Z\. ]{1,6})\s*(\d{6})([CP])(\d{8})$")


def parse_schwab_option_symbol(symbol: str):
    """Parse a Schwab OCC symbol into (underlying, expiry, put_call, strike)."""
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    root, yymmdd, pc, strike8 = m.groups()
    underlying = root.strip()
    expiry = dt.datetime.strptime(yymmdd, "%y%m%d").date()
    strike = int(strike8) / 1000.0
    put_call = "PUT" if pc == "P" else "CALL"
    return underlying, expiry, put_call, strike


def analyze_positions(
    svc: SchwabLiveDataService,
    days: int = 90,
    account_index: int = 0,
    symbol_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """Full orchestrator: fetch positions, enrich, compute metrics."""
    today = dt.date.today()

    # 1. Get current positions from account
    account_data = svc.get_account_positions(account_index=account_index)
    positions = account_data["positions"]
    balances = account_data["balances"]

    # Filter to a specific symbol if requested
    if symbol_filter:
        sym_upper = symbol_filter.upper()
        positions = [
            p for p in positions
            if sym_upper in p["symbol"].upper() or p.get("underlying", "").upper() == sym_upper
        ]

    if not positions:
        return {
            "as_of": dt.datetime.now(dt.timezone.utc).isoformat(),
            "account_summary": balances,
            "positions": [],
        }

    # 2. Get trade history for entry matching
    transactions = svc.get_trade_history(days=days, account_index=account_index)
    entry_details = match_entry_details(positions, transactions)

    # 3. Collect unique underlyings for quotes and chains
    underlyings = set()
    for pos in positions:
        if pos["asset_type"] == "OPTION":
            parsed = parse_schwab_option_symbol(pos["symbol"])
            if parsed:
                underlyings.add(parsed[0])
            elif pos.get("underlying"):
                underlyings.add(pos["underlying"])
        else:
            underlyings.add(pos["symbol"].strip())

    # 4. Fetch live quotes for all underlyings
    quotes_payload = {}
    if underlyings:
        quotes_payload = svc.get_quotes(list(underlyings))

    # 5. Fetch option chains for underlyings that have option positions
    option_underlyings = set()
    for pos in positions:
        if pos["asset_type"] == "OPTION":
            ul = pos.get("underlying") or (parse_schwab_option_symbol(pos["symbol"]) or [None])[0]
            if ul:
                option_underlyings.add(ul)

    chains_payload = {}
    for ul in option_underlyings:
        try:
            chains_payload[ul] = svc.get_option_chain(ul, strike_count=12)
        except Exception:
            pass

    # 6. Fetch yfinance context per underlying
    yf_context = {}
    for ul in underlyings:
        # Get current IV from chain if available
        current_iv = None
        if ul in chains_payload:
            chain = chains_payload[ul]
            iv_val = _safe_float(chain.get("volatility"))
            if iv_val:
                current_iv = iv_val / 100.0 if iv_val > 1 else iv_val
        yf_context[ul] = fetch_yfinance_context(ul, current_iv=current_iv, today=today)

    # 7. Enrich each position
    enriched = []
    for pos in positions:
        symbol = pos["symbol"]
        is_option = pos["asset_type"] == "OPTION"

        # Parse option details
        strike = None
        expiry = None
        underlying = pos.get("underlying", symbol.strip())
        if is_option:
            parsed = parse_schwab_option_symbol(symbol)
            if parsed:
                underlying, expiry, _, strike = parsed

        # Underlying quote
        uq = quotes_payload.get(underlying, {})
        uq_body = uq.get("quote", uq)
        underlying_price = _safe_float(uq_body.get("lastPrice")) or _safe_float(uq_body.get("mark"))

        # Greeks from chain (find matching contract)
        greeks = None
        live_quote = None
        if is_option and underlying in chains_payload:
            chain = chains_payload[underlying]
            pc_key = "putExpDateMap" if pos.get("put_call") == "PUT" else "callExpDateMap"
            exp_map = chain.get(pc_key, {})
            for exp_key, strike_map in exp_map.items():
                for strike_key, contracts in strike_map.items():
                    for contract in contracts:
                        if contract.get("symbol", "").strip() == symbol.strip():
                            greeks = {
                                "delta": _safe_float(contract.get("delta")),
                                "gamma": _safe_float(contract.get("gamma")),
                                "theta": _safe_float(contract.get("theta")),
                                "vega": _safe_float(contract.get("vega")),
                                "iv": _safe_float(contract.get("volatility")),
                            }
                            # Normalize IV to decimal if it's percentage
                            if greeks["iv"] and greeks["iv"] > 1:
                                greeks["iv"] = greeks["iv"] / 100.0
                            live_quote = {
                                "bid": _safe_float(contract.get("bid")),
                                "ask": _safe_float(contract.get("ask")),
                                "mark": _safe_float(contract.get("mark")),
                                "last": _safe_float(contract.get("last")),
                                "open_interest": _safe_float(contract.get("openInterest")),
                                "volume": _safe_float(contract.get("totalVolume")),
                            }
                            break
                    if greeks:
                        break
                if greeks:
                    break

        # Entry details
        entry = entry_details.get(symbol, {})
        entry_date_str = entry.get("entry_date")
        entry_date = dt.datetime.strptime(entry_date_str, "%Y-%m-%d").date() if entry_date_str else None
        entry_price = entry.get("entry_price")
        avg_cost = entry_price if entry_price else _safe_float(pos.get("avg_cost"))

        pos_for_metrics = {**pos, "avg_cost": avg_cost}

        # Compute risk metrics
        metrics = compute_risk_metrics(
            position=pos_for_metrics,
            greeks=greeks,
            underlying_price=underlying_price or 0.0,
            strike=strike,
            expiry=expiry,
            entry_date=entry_date,
            today=today,
        )

        # Bid/ask spread %
        bid_ask_spread_pct = None
        if live_quote and live_quote.get("mark") and live_quote["mark"] > 0:
            bid = live_quote.get("bid") or 0
            ask = live_quote.get("ask") or 0
            bid_ask_spread_pct = round((ask - bid) / live_quote["mark"] * 100, 1)

        # Assemble enriched position
        yf_ctx = yf_context.get(underlying, {})
        enriched.append({
            "symbol": symbol,
            "underlying": underlying,
            "asset_type": pos["asset_type"],
            "put_call": pos.get("put_call", ""),
            "strike": strike,
            "expiry": str(expiry) if expiry else None,
            "qty": pos["qty"],
            "avg_cost": avg_cost,
            "market_value": pos.get("market_value"),
            "entry_date": entry_date_str,
            "live_quote": live_quote,
            "greeks": greeks,
            "underlying_quote": {
                "last": underlying_price,
                "change_pct": _safe_float(uq_body.get("netPercentChangeInDouble")),
            },
            "computed": {
                **metrics,
                "bid_ask_spread_pct": bid_ask_spread_pct,
                "open_interest_at_strike": live_quote.get("open_interest") if live_quote else None,
                "volume_at_strike": live_quote.get("volume") if live_quote else None,
                **yf_ctx,
            },
        })

    return {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(),
        "account_summary": balances,
        "positions": enriched,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze open Schwab positions with risk metrics and market context."
    )
    parser.add_argument("--days", type=int, default=90, help="Days of trade history for entry matching (default: 90).")
    parser.add_argument("--symbol", default=None, help="Filter to a specific underlying symbol.")
    parser.add_argument("--account-index", type=int, default=0, help="Account index (default: 0).")
    parser.add_argument("--out-dir", default="", help="Output directory (default: c:/uw_root/out/trade_analysis).")
    parser.add_argument("--manual-auth", action="store_true", help="Use manual OAuth flow.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(config=config, manual_auth=args.manual_auth)

    print("Analyzing open positions...")
    result = analyze_positions(
        svc=svc,
        days=args.days,
        account_index=args.account_index,
        symbol_filter=args.symbol,
    )

    n_pos = len(result["positions"])
    print(f"  Auth mode: {svc.auth_mode}")
    print(f"  Open positions found: {n_pos}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("c:/uw_root/out/trade_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    today_str = dt.date.today().isoformat()
    json_path = out_dir / f"position_data_{today_str}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  Position data saved: {json_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Verify imports work**

Run: `cd c:/uw_root && python -c "from uwos.schwab_position_analyzer import analyze_positions, main; print('OK')"`
Expected: `OK`

**Step 3: Run all tests to make sure nothing broke**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add uwos/schwab_position_analyzer.py
git commit -m "feat: add analyze_positions orchestrator and CLI for position analysis"
```

---

### Task 6: Update the Claude skill `/tradehistory`

**Files:**
- Modify: `C:/Users/anupamvi/.claude/skills/trade-history/SKILL.md`

**Step 1: Rewrite the skill**

Replace the entire SKILL.md with the full analysis agent skill that:
1. Runs Phase 1 (Python data collection)
2. Reads the position_data JSON
3. Runs Phase 2 (parallel WebSearch per ticker for news, X sentiment, macro)
4. Runs Phase 3 (per-position verdict cards using the patience-biased framework)
5. Writes trade-analysis-{date}.md and presents clickable links

```markdown
# Trade Analysis Agent

Analyze open Schwab positions with live market data, risk metrics, news, macro events, X sentiment, and per-position HOLD/CLOSE/ROLL verdicts biased toward patience and profit maximization.

## Core Philosophy

> "Close when the thesis breaks or the math inverts, not at an arbitrary profit target."

For each position, answer: "Is the expected value of holding for another week positive?" If yes → HOLD. If no → best exit.

## Parameters

- **days** (optional): History lookback for entry matching. Default: 90.
- **symbol** (optional): Filter to a specific underlying (e.g., AAPL).

Examples:
- `/tradehistory` → analyze all open positions
- `/tradehistory 60` → 60-day lookback
- `/tradehistory 90 AAPL` → AAPL positions only

## Execution Steps

### Phase 1: Data Collection

Run the Python analyzer to fetch positions + Schwab data + yfinance context:

```bash
cd "c:/uw_root" && python -m uwos.schwab_position_analyzer --days {days} [--symbol {symbol}]
```

Timeout: 180000ms (3 min). If Schwab auth fails, tell user to run with `--manual-auth` in their terminal.

Read the output JSON from `c:/uw_root/out/trade_analysis/position_data_{date}.json`.

### Phase 2: Research (parallel per unique underlying)

For each unique underlying ticker in the positions, run parallel WebSearch agents:

1. **News search:** `"{ticker} stock news last 7 days"` — headlines, analyst actions, SEC filings
2. **X sentiment search:** `"{ticker} stock Twitter sentiment"` — bullish/bearish themes
3. **Macro search (once):** `"stock market macro outlook Fed CPI VIX this week"` — rates, economic data, regime

### Phase 3: Analysis & Verdicts

For each position, produce a verdict card combining Phase 1 data + Phase 2 research.

## Verdict Decision Matrix (patience-biased)

| Signal | Verdict | Rationale |
|--------|---------|-----------|
| Theta strong, no catalyst, IV stable/elevated | **HOLD — let it work** | Time is on your side |
| 50%+ profit, high IV rank, strong theta, no earnings | **HOLD — target 65-80%** | Premium still rich |
| 50%+ profit, IV crushed or theta slowing (DTE < 10) | **CLOSE — diminishing returns** | Gamma risk not worth it |
| Underlying against you, still OTM, good DTE | **HOLD — time heals** | Don't panic-close |
| Underlying against you, near/ITM, DTE < 14 | **ROLL — extend duration** | Buy time + credit |
| Earnings within 7 days, at profit | **CLOSE or ROLL past** | Binary event risk |
| Earnings within 7 days, at loss | **ASSESS** | Evaluate on merits |
| IV expanding, short vol | **HOLD if thesis intact** | Temporary |
| News/macro adverse + fundamentals deteriorating | **CLOSE — thesis broken** | Reason for trade changed |
| Near max profit (>85%) | **CLOSE — nothing left** | Risk/reward inverted |

## Output Format

### Per-Position Verdict Card

```
### {UNDERLYING} — {Short/Long} {Put/Call} ${strike} | {expiry} | {DTE} DTE
**Status:** {+/-$P&L} ({pnl%}) | Prob Profit: {prob}% | Theta: {+/-$}/day

| Metric | Value | Signal |
|--------|-------|--------|
| P&L | {pnl%} of max | {assessment} |
| Delta | {delta} | {OTM/ATM/ITM assessment} |
| Theta/day | {$value} | {Strong/Weak/Bleeding} |
| Gamma risk | {value} | {Low/Medium/High} |
| IV Rank | {pctl} | {Premium rich/fair/cheap} |
| IV vs HV | {spread} | {Overpriced/Fair/Underpriced vol} |
| Breakeven | ${price} ({dist}% away) | {Wide/Narrow/Breached buffer} |
| Earnings | {date} ({days} days) | {No overlap/Caution/Danger} |
| Bid/Ask | {spread%} | {Clean/Acceptable/Wide exit} |
| Support | {levels} | {Above/Near/Below strike} |

**News:** [2-3 bullet summary]
**Macro:** [relevant context]
**X Sentiment:** [bullish/bearish/mixed + themes]

**VERDICT: {HOLD/CLOSE/ROLL/ASSESS} — {one-line reason}**
{2-3 sentence reasoning based on the data. Explain what would change the verdict.}
```

### Portfolio Summary (at end)

- Total open positions, total unrealized P&L
- Sector concentration breakdown
- SPY correlation risk (are positions correlated?)
- Macro regime alignment
- Overall portfolio health score (1-10)

### Output Files (ALWAYS include — clickable links)

```
- Analysis report: [trade-analysis-{date}.md](out/trade_analysis/trade-analysis-{date}.md)
- Position data: [position_data_{date}.json](out/trade_analysis/position_data_{date}.json)
```

## Error Handling

- Schwab auth failed: tell user to re-auth in terminal with `python -m uwos.schwab_position_analyzer --manual-auth`
- No open positions: report clearly
- yfinance data missing: skip enrichment, note in output
- WebSearch fails: skip research for that ticker, note it
- Equity positions (no Greeks): analyze with simpler metrics (P&L, support/resistance, news)

## Report Writing

After generating all verdict cards, write the full analysis to:
`c:/uw_root/out/trade_analysis/trade-analysis-{date}.md`

Use the Write tool to create the file, then present clickable links.
```

**Step 2: Commit**

```bash
git add C:/Users/anupamvi/.claude/skills/trade-history/SKILL.md
git commit -m "feat: upgrade trade-history skill to full analysis agent with verdicts"
```

---

### Task 7: Integration test — dry run

**Step 1: Verify Python module runs**

Run: `cd c:/uw_root && python -m uwos.schwab_position_analyzer --days 90`
Timeout: 180000ms

Expected: Either succeeds (positions fetched + JSON written) or fails with auth error (expected if token stale — user must re-auth in terminal).

**Step 2: Verify all tests pass**

Run: `cd c:/uw_root && python -m pytest tests/test_schwab_position_analyzer.py -v`
Expected: All PASS

**Step 3: Final commit if any fixups needed**

```bash
git add -u
git commit -m "fix: integration test fixups for position analyzer"
```

---

## Summary

| Task | Component | Files |
|------|-----------|-------|
| 1 | `get_account_positions()` API method | `schwab_auth.py`, `test_schwab_position_analyzer.py` |
| 2 | `compute_risk_metrics()` pure math | `schwab_position_analyzer.py`, tests |
| 3 | `fetch_yfinance_context()` enrichment | `schwab_position_analyzer.py`, tests |
| 4 | `match_entry_details()` history matching | `schwab_position_analyzer.py`, tests |
| 5 | `analyze_positions()` orchestrator + CLI | `schwab_position_analyzer.py` |
| 6 | Updated `/tradehistory` Claude skill | `SKILL.md` |
| 7 | Integration test | — |
