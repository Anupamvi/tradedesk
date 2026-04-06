"""
Trade Ideas — unified scanner combining dip quality, UW flow, and Schwab chains
into specific, actionable trade recommendations.

Produces alerts like:
  SELL ADBE $220P/$210P May 15 | Credit $2.50 | Max $250 | Prob 72% | Delta -0.22

Pipeline:
  1. Schwab quotes: screen S&P 500 for drops from 52w high (fast, 502 tickers)
  2. UW flow: layer institutional flow signals on drop candidates
  3. Fundamentals: quality score from yfinance (only top 20 candidates)
  4. Chain lookup: Schwab option chains for top 8 → construct specific trades
  5. Filter: no ETFs, no already-held, no earnings within 7 days
  6. Output: actionable trade cards with strikes, expiry, credit, Greeks

Usage:
    python -m uwos.trade_ideas                    # latest data
    python -m uwos.trade_ideas --date 2026-03-27  # specific date
"""

import argparse
import csv
import datetime as dt
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("c:/uw_root")

# Exclude leveraged, inverse, and non-equity tickers
JUNK_TICKERS = {
    "TECS", "SOXS", "SQQQ", "UVXY", "VXX", "SPXU", "SDOW", "TZA", "FAZ",
    "TQQQ", "SOXL", "UPRO", "SPXL", "UDOW", "TNA", "FAS", "UVIX",
    "ARKK", "ARKG", "ARKW", "ARKF",  # thematic ETFs
}

SECTOR_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Energy": "XLE",
    "Utilities": "XLU", "Industrials": "XLI", "Basic Materials": "XLB",
    "Real Estate": "XLRE", "Communication Services": "XLC",
}


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Step 1: Schwab quotes — fast screen for drops
# ---------------------------------------------------------------------------

def screen_drops(svc, tickers: List[str], min_drop_pct: float = 8.0,
                 min_market_cap: float = 10e9) -> List[Dict]:
    """Batch Schwab quotes, filter for drops from 52w high."""
    all_quotes = {}
    for i in range(0, len(tickers), 100):
        batch = tickers[i:i + 100]
        try:
            all_quotes.update(svc.get_quotes(batch))
        except Exception:
            pass

    candidates = []
    for ticker, qdata in all_quotes.items():
        if ticker in JUNK_TICKERS:
            continue
        q = qdata.get("quote", {})
        f = qdata.get("fundamental", {})
        ref = qdata.get("reference", {})

        price = q.get("lastPrice") or q.get("closePrice")
        high_52w = q.get("52WeekHigh")
        pe = f.get("peRatio")
        eps = f.get("eps")
        shares = f.get("sharesOutstanding", 0)

        if not price or not high_52w or price <= 0 or high_52w <= 0:
            continue

        pct_from_high = (price - high_52w) / high_52w * 100
        market_cap = price * shares if shares else 0

        if pct_from_high > -min_drop_pct:
            continue
        if 0 < market_cap < min_market_cap:
            continue
        if eps is not None and eps < 0:
            continue

        # Check it's not an ETF via reference data
        desc = str(ref.get("description", "")).lower()
        if any(x in desc for x in ["etf", "fund", "trust", "index", "proshares", "direxion"]):
            continue

        candidates.append({
            "ticker": ticker,
            "price": price,
            "high_52w": high_52w,
            "pct_from_high": pct_from_high,
            "pe": pe or 0,
            "eps": eps or 0,
            "market_cap": market_cap,
            "shares": shares,
        })

    candidates.sort(key=lambda x: x["pct_from_high"])
    return candidates


# ---------------------------------------------------------------------------
# Step 2: UW flow overlay
# ---------------------------------------------------------------------------

def load_uw_flow(data_dir: Path) -> Dict[str, Dict]:
    """Load UW bot-eod-report and aggregate flow by ticker."""
    zips = list(data_dir.glob("bot-eod-report-*.zip"))
    if not zips:
        return {}

    with zipfile.ZipFile(zips[0]) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return {}
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), low_memory=False)

    df["premium"] = pd.to_numeric(df.get("premium"), errors="coerce").fillna(0)
    df["size"] = pd.to_numeric(df.get("size"), errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
    df["open_interest"] = pd.to_numeric(df.get("open_interest"), errors="coerce").fillna(0)
    df["implied_volatility"] = pd.to_numeric(df.get("implied_volatility"), errors="coerce").fillna(0)
    df["delta"] = pd.to_numeric(df.get("delta"), errors="coerce").fillna(0)

    # Only significant trades
    df = df[df.get("canceled", "f").astype(str) != "t"]
    df = df[(df["premium"] >= 25000) | (df["size"] >= 100)]
    if df.empty:
        return {}

    df["is_buy"] = df["side"].str.lower().str.strip() == "ask"
    df["vol_oi_ratio"] = np.where(df["open_interest"] > 0, df["volume"] / df["open_interest"], 0)

    result = {}
    for ticker, group in df.groupby("underlying_symbol"):
        total_prem = group["premium"].sum()
        buy_prem = group.loc[group["is_buy"], "premium"].sum()
        call_prem = group.loc[group["option_type"].str.lower() == "call", "premium"].sum()

        result[ticker] = {
            "total_premium": total_prem,
            "buy_pct": (buy_prem / total_prem * 100) if total_prem > 0 else 50,
            "call_pct": (call_prem / total_prem * 100) if total_prem > 0 else 50,
            "max_vol_oi": group["vol_oi_ratio"].max(),
            "avg_iv": group["implied_volatility"].mean(),
            "trade_count": len(group),
        }
    return result


# ---------------------------------------------------------------------------
# Step 3: Fundamentals + earnings check
# ---------------------------------------------------------------------------

def get_fundamentals_and_earnings(ticker: str) -> Dict:
    """Fetch fundamentals + next earnings date from yfinance."""
    import yfinance as yf
    defaults = {
        "roe": 0, "debt_equity": 999, "rev_growth": 0, "fcf_yield": 0,
        "profit_margin": 0, "institutional_pct": 0, "earnings_beats": 0,
        "analyst_upside": 0, "sector": "Unknown", "name": ticker,
        "next_earnings": None, "days_to_earnings": 999,
    }
    try:
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        defaults["roe"] = (info.get("returnOnEquity") or 0) * 100
        de = info.get("debtToEquity")
        defaults["debt_equity"] = (de / 100) if de else 999
        defaults["rev_growth"] = (info.get("revenueGrowth") or 0) * 100
        defaults["profit_margin"] = (info.get("profitMargins") or 0) * 100
        defaults["institutional_pct"] = (info.get("heldPercentInstitutions") or 0) * 100
        defaults["sector"] = info.get("sector", "Unknown")
        defaults["name"] = info.get("shortName", ticker)

        fcf = info.get("freeCashflow")
        mcap = info.get("marketCap")
        if fcf and mcap and mcap > 0:
            defaults["fcf_yield"] = (fcf / mcap) * 100

        # Analyst upside
        target = info.get("targetMeanPrice")
        current = info.get("currentPrice") or info.get("regularMarketPrice")
        if target and current and current > 0:
            defaults["analyst_upside"] = (target - current) / current * 100

        # Earnings beats
        try:
            ed = stock.earnings_dates
            if ed is not None and len(ed) > 0:
                surprise_col = next((c for c in ed.columns if "surprise" in c.lower()), None)
                if surprise_col:
                    defaults["earnings_beats"] = int((ed.head(4)[surprise_col].dropna() > 0).sum())

                # Next earnings date
                future = ed.index[ed.index > pd.Timestamp.now(tz="UTC")]
                if len(future) > 0:
                    next_earn = future.min()
                    defaults["next_earnings"] = next_earn.strftime("%Y-%m-%d")
                    defaults["days_to_earnings"] = (next_earn.date() - dt.date.today()).days
        except Exception:
            pass

    except Exception:
        pass
    return defaults


def quality_score(fund: Dict) -> float:
    """Score fundamental quality (0-100)."""
    s = 0
    if fund["roe"] >= 20: s += 18
    elif fund["roe"] >= 15: s += 13
    elif fund["roe"] >= 10: s += 8

    if fund["debt_equity"] <= 0.5: s += 14
    elif fund["debt_equity"] <= 1.0: s += 9
    elif fund["debt_equity"] <= 2.0: s += 4

    if fund["rev_growth"] >= 20: s += 14
    elif fund["rev_growth"] >= 10: s += 10
    elif fund["rev_growth"] >= 5: s += 6

    if fund["fcf_yield"] >= 5: s += 14
    elif fund["fcf_yield"] >= 3: s += 9
    elif fund["fcf_yield"] >= 1: s += 4

    s += min(14, fund["earnings_beats"] * 4)

    if fund["profit_margin"] >= 20: s += 13
    elif fund["profit_margin"] >= 10: s += 8
    elif fund["profit_margin"] >= 5: s += 3

    if fund["institutional_pct"] >= 70: s += 13
    elif fund["institutional_pct"] >= 50: s += 8
    elif fund["institutional_pct"] >= 30: s += 3

    return min(100, s)


# ---------------------------------------------------------------------------
# Step 4: Schwab chain lookup — construct trades
# ---------------------------------------------------------------------------

def construct_trades(svc, ticker: str, price: float, direction: str,
                     avg_iv: float) -> List[Dict]:
    """Fetch Schwab option chain and construct specific trade ideas.

    Returns list of trade constructions with specific strikes, credits, Greeks.
    """
    trades = []
    target_dte_min = 25
    target_dte_max = 55

    from_date = dt.date.today() + dt.timedelta(days=target_dte_min)
    to_date = dt.date.today() + dt.timedelta(days=target_dte_max)

    try:
        chain = svc.get_option_chain(ticker, strike_count=10,
                                      from_date=from_date, to_date=to_date)
    except Exception:
        return []

    underlying = chain.get("underlyingPrice", price)

    if direction == "bullish" or direction == "neutral":
        # Strategy 1: Bull Put Credit Spread (sell put spread below price)
        puts = chain.get("putExpDateMap", {})
        for expiry_key, strikes in puts.items():
            expiry_date = expiry_key.split(":")[0]
            try:
                dte = (dt.date.fromisoformat(expiry_date) - dt.date.today()).days
            except ValueError:
                continue
            if dte < target_dte_min or dte > target_dte_max:
                continue

            sorted_strikes = sorted(strikes.keys(), key=float, reverse=True)
            # Find short put at ~-0.25 delta (OTM)
            short_put = None
            long_put = None
            for sk in sorted_strikes:
                contracts = strikes[sk]
                if not contracts:
                    continue
                c = contracts[0]
                d = c.get("delta", 0)
                bid = c.get("bid", 0)
                if d is None:
                    continue
                if -0.35 <= d <= -0.15 and bid > 0.50:
                    short_put = {"strike": float(sk), "bid": bid, "ask": c.get("ask", 0),
                                 "delta": d, "theta": c.get("theta", 0),
                                 "iv": c.get("volatility", 0), "oi": c.get("openInterest", 0)}
                    break

            if not short_put:
                continue

            # Long put: $5-10 below short
            for sk in sorted_strikes:
                sk_f = float(sk)
                if sk_f < short_put["strike"] - 4 and sk_f >= short_put["strike"] - 15:
                    contracts = strikes[sk]
                    if contracts:
                        c = contracts[0]
                        if c.get("ask", 0) > 0:
                            long_put = {"strike": sk_f, "bid": c.get("bid", 0),
                                        "ask": c.get("ask", 0), "delta": c.get("delta", 0)}
                            break

            if not long_put:
                continue

            width = short_put["strike"] - long_put["strike"]
            credit = short_put["bid"] - long_put["ask"]
            if credit <= 0.20:
                continue

            max_profit = credit * 100
            max_loss = (width - credit) * 100
            prob_profit = (1 + short_put["delta"]) * 100  # approx from delta
            buffer_pct = (underlying - short_put["strike"]) / underlying * 100

            # Buffer warning
            caution = ""
            if buffer_pct < 5:
                caution = " [TIGHT BUFFER]"
            elif buffer_pct < 7:
                caution = " [MODERATE BUFFER]"

            trades.append({
                "strategy": f"Bull Put Credit{caution}",
                "short_strike": short_put["strike"],
                "long_strike": long_put["strike"],
                "expiry": expiry_date,
                "dte": dte,
                "credit": round(credit, 2),
                "max_profit": round(max_profit),
                "max_loss": round(max_loss),
                "prob_profit": round(prob_profit, 1),
                "short_delta": round(short_put["delta"], 3),
                "short_theta": round(short_put["theta"], 3),
                "short_iv": round(short_put["iv"], 1),
                "buffer_pct": round(buffer_pct, 1),
                "width": width,
            })
            break  # one trade per expiry

    if direction == "bearish" or direction == "neutral":
        # Strategy 2: Bear Call Credit Spread (sell call spread above price)
        calls = chain.get("callExpDateMap", {})
        for expiry_key, strikes in calls.items():
            expiry_date = expiry_key.split(":")[0]
            try:
                dte = (dt.date.fromisoformat(expiry_date) - dt.date.today()).days
            except ValueError:
                continue
            if dte < target_dte_min or dte > target_dte_max:
                continue

            sorted_strikes = sorted(strikes.keys(), key=float)
            short_call = None
            long_call = None
            for sk in sorted_strikes:
                contracts = strikes[sk]
                if not contracts:
                    continue
                c = contracts[0]
                d = c.get("delta", 0)
                bid = c.get("bid", 0)
                if d is None:
                    continue
                if 0.15 <= d <= 0.35 and bid > 0.50:
                    short_call = {"strike": float(sk), "bid": bid, "ask": c.get("ask", 0),
                                  "delta": d, "theta": c.get("theta", 0),
                                  "iv": c.get("volatility", 0)}
                    break

            if not short_call:
                continue

            for sk in sorted_strikes:
                sk_f = float(sk)
                if sk_f > short_call["strike"] + 4 and sk_f <= short_call["strike"] + 15:
                    contracts = strikes[sk]
                    if contracts:
                        c = contracts[0]
                        if c.get("ask", 0) > 0:
                            long_call = {"strike": sk_f, "ask": c.get("ask", 0),
                                         "delta": c.get("delta", 0)}
                            break

            if not long_call:
                continue

            width = long_call["strike"] - short_call["strike"]
            credit = short_call["bid"] - long_call["ask"]
            if credit <= 0.20:
                continue

            max_profit = credit * 100
            max_loss = (width - credit) * 100
            prob_profit = (1 - short_call["delta"]) * 100
            buffer_pct = (short_call["strike"] - underlying) / underlying * 100

            caution = ""
            if buffer_pct < 5:
                caution = " [TIGHT BUFFER]"
            elif buffer_pct < 7:
                caution = " [MODERATE BUFFER]"

            trades.append({
                "strategy": f"Bear Call Credit{caution}",
                "short_strike": short_call["strike"],
                "long_strike": long_call["strike"],
                "expiry": expiry_date,
                "dte": dte,
                "credit": round(credit, 2),
                "max_profit": round(max_profit),
                "max_loss": round(max_loss),
                "prob_profit": round(prob_profit, 1),
                "short_delta": round(short_call["delta"], 3),
                "short_theta": round(short_call["theta"], 3),
                "short_iv": round(short_call["iv"], 1),
                "buffer_pct": round(buffer_pct, 1),
                "width": width,
            })
            break

    # Strategy 3: Bull Call Debit — for strongly bullish flow + low IV
    if direction == "bullish" and avg_iv < 0.40:
        calls = chain.get("callExpDateMap", {})
        for expiry_key, strikes in calls.items():
            expiry_date = expiry_key.split(":")[0]
            try:
                dte = (dt.date.fromisoformat(expiry_date) - dt.date.today()).days
            except ValueError:
                continue
            if dte < target_dte_min or dte > target_dte_max:
                continue

            sorted_strikes = sorted(strikes.keys(), key=float)
            long_call = None
            short_call = None

            # Long call: near ATM (~0.40-0.50 delta)
            for sk in sorted_strikes:
                contracts = strikes[sk]
                if not contracts:
                    continue
                c = contracts[0]
                d = c.get("delta", 0)
                ask = c.get("ask", 0)
                if d is None:
                    continue
                if 0.35 <= d <= 0.55 and ask > 0.50:
                    long_call = {"strike": float(sk), "bid": c.get("bid", 0), "ask": ask,
                                 "delta": d, "theta": c.get("theta", 0),
                                 "iv": c.get("volatility", 0)}
                    break

            if not long_call:
                continue

            # Short call: $5-15 above long
            for sk in sorted_strikes:
                sk_f = float(sk)
                if sk_f > long_call["strike"] + 4 and sk_f <= long_call["strike"] + 15:
                    contracts = strikes[sk]
                    if contracts:
                        c = contracts[0]
                        if c.get("bid", 0) > 0:
                            short_call = {"strike": sk_f, "bid": c.get("bid", 0),
                                          "delta": c.get("delta", 0)}
                            break

            if not short_call:
                continue

            width = short_call["strike"] - long_call["strike"]
            debit = long_call["ask"] - short_call["bid"]
            if debit <= 0 or debit >= width * 0.60:
                continue  # don't pay more than 60% of width

            max_profit = (width - debit) * 100
            max_loss = debit * 100
            # Approx prob based on long delta
            otm_pct = (long_call["strike"] - underlying) / underlying * 100 if underlying > 0 else 0

            trades.append({
                "strategy": "Bull Call Debit",
                "short_strike": short_call["strike"],
                "long_strike": long_call["strike"],
                "expiry": expiry_date,
                "dte": dte,
                "credit": round(-debit, 2),  # negative = debit
                "max_profit": round(max_profit),
                "max_loss": round(max_loss),
                "prob_profit": round(long_call["delta"] * 100, 1),
                "short_delta": round(long_call["delta"], 3),
                "short_theta": round(long_call["theta"], 3),
                "short_iv": round(long_call["iv"], 1),
                "buffer_pct": round(-otm_pct, 1),  # negative = OTM distance
                "width": width,
            })
            break

    return trades


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def scan_trade_ideas(data_dir: Optional[Path] = None, top_n: int = 8,
                     exclude_tickers: Optional[set] = None,
                     verbose: bool = True) -> List[Dict]:
    """Full pipeline: drops + flow + fundamentals + chain → trade ideas."""
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    from uwos.eod_trade_scan_mode_a import compute_macro_regime
    from uwos.dip_scanner import load_sp500_universe

    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(config=config, interactive_login=False)
    exclude = exclude_tickers or set()

    # Macro
    macro = compute_macro_regime(dt.date.today())
    regime = macro["regime"]
    if verbose:
        _safe_print(f"  [ideas] Macro: SPY 5d={macro['spy_5d_ret']:+.2%}, VIX={macro['vix_level']:.1f}, regime={regime}")

    # Step 1: Screen drops via Schwab
    if verbose:
        _safe_print("  [ideas] Step 1: Schwab quote screen...")
    universe = load_sp500_universe()
    tickers = universe["ticker"].tolist()
    name_map = dict(zip(universe["ticker"], universe.get("name", universe["ticker"])))
    sector_map = dict(zip(universe["ticker"], universe.get("sector", "Unknown")))

    drops = screen_drops(svc, tickers)
    if verbose:
        _safe_print(f"  [ideas] Found {len(drops)} stocks down >8% from 52w high")

    # Remove already-held
    drops = [d for d in drops if d["ticker"] not in exclude]

    # Step 2: UW flow overlay
    flow_data = {}
    if data_dir and data_dir.exists():
        if verbose:
            _safe_print(f"  [ideas] Step 2: UW flow from {data_dir.name}...")
        flow_data = load_uw_flow(data_dir)
        if verbose:
            _safe_print(f"  [ideas] Flow data for {len(flow_data)} tickers")

    # Score and rank: combine drop magnitude + flow signals
    scored = []
    for d in drops[:50]:  # top 50 drops by magnitude
        ticker = d["ticker"]
        flow = flow_data.get(ticker, {})

        # Drop score (0-40)
        drop_score = min(40, abs(d["pct_from_high"]) * 1.2)

        # Flow score (0-30)
        flow_score = 0
        total_prem = flow.get("total_premium", 0)
        buy_pct = flow.get("buy_pct", 50)
        vol_oi = flow.get("max_vol_oi", 0)

        if total_prem >= 1_000_000: flow_score += 12
        elif total_prem >= 500_000: flow_score += 8
        elif total_prem >= 100_000: flow_score += 4

        if buy_pct >= 65: flow_score += 10
        elif buy_pct >= 55: flow_score += 5

        if vol_oi >= 3: flow_score += 8
        elif vol_oi >= 2: flow_score += 4

        # Macro alignment (0-30)
        # Determine direction from flow
        call_pct = flow.get("call_pct", 50)
        if call_pct >= 60:
            direction = "bullish"
        elif call_pct <= 40:
            direction = "bearish"
        else:
            direction = "neutral"

        macro_score = {"risk_off": {"bearish": 28, "neutral": 18, "bullish": 8},
                       "neutral": {"bearish": 18, "neutral": 22, "bullish": 18},
                       "risk_on": {"bearish": 8, "neutral": 18, "bullish": 28},
                       }.get(regime, {}).get(direction, 15)

        total = drop_score + flow_score + macro_score

        scored.append({
            **d,
            "direction": direction,
            "drop_score": round(drop_score, 1),
            "flow_score": round(flow_score, 1),
            "macro_score": round(macro_score, 1),
            "pre_score": round(total, 1),
            "flow": flow,
            "name": name_map.get(ticker, ticker),
            "sector": sector_map.get(ticker, "Unknown"),
        })

    scored.sort(key=lambda x: x["pre_score"], reverse=True)

    # Step 3: Deep dive top candidates — fundamentals + earnings
    if verbose:
        _safe_print(f"  [ideas] Step 3: Fundamentals for top {min(20, len(scored))} candidates...")

    deep = []
    for cand in scored[:20]:
        ticker = cand["ticker"]
        fund = get_fundamentals_and_earnings(ticker)

        # Earnings filter: skip if within 7 days
        if fund["days_to_earnings"] <= 7:
            if verbose:
                _safe_print(f"    SKIP {ticker} — earnings in {fund['days_to_earnings']} days")
            continue

        q_score = quality_score(fund)
        if q_score < 35:
            continue

        # Final composite: pre_score (70%) + quality (30%)
        composite = 0.70 * cand["pre_score"] + 0.30 * q_score

        deep.append({
            **cand,
            "quality_score": round(q_score, 1),
            "composite": round(composite, 1),
            "fundamentals": fund,
        })

    deep.sort(key=lambda x: x["composite"], reverse=True)

    # Step 4: Chain lookup for top candidates — construct trades
    if verbose:
        _safe_print(f"  [ideas] Step 4: Schwab chains for top {min(top_n, len(deep))} ideas...")

    results = []
    for cand in deep[:top_n]:
        ticker = cand["ticker"]
        avg_iv = cand["flow"].get("avg_iv", 0.3)
        trades = construct_trades(svc, ticker, cand["price"], cand["direction"], avg_iv)

        if not trades:
            continue

        best_trade = trades[0]
        fund = cand["fundamentals"]

        results.append({
            "ticker": ticker,
            "name": fund.get("name", cand.get("name", ticker)),
            "sector": fund.get("sector", cand.get("sector", "")),
            "price": cand["price"],
            "pct_from_high": round(cand["pct_from_high"], 1),
            "direction": cand["direction"],
            "composite": cand["composite"],
            "quality_score": cand["quality_score"],
            "drop_score": cand["drop_score"],
            "flow_score": cand["flow_score"],
            "macro_score": cand["macro_score"],
            # Trade specifics
            "strategy": best_trade["strategy"],
            "short_strike": best_trade["short_strike"],
            "long_strike": best_trade["long_strike"],
            "expiry": best_trade["expiry"],
            "dte": best_trade["dte"],
            "credit": best_trade["credit"],
            "max_profit": best_trade["max_profit"],
            "max_loss": best_trade["max_loss"],
            "prob_profit": best_trade["prob_profit"],
            "short_delta": best_trade["short_delta"],
            "short_theta": best_trade["short_theta"],
            "short_iv": best_trade["short_iv"],
            "buffer_pct": best_trade["buffer_pct"],
            "width": best_trade["width"],
            # Context
            "total_premium": cand["flow"].get("total_premium", 0),
            "buy_pct": cand["flow"].get("buy_pct", 50),
            "vol_oi": cand["flow"].get("max_vol_oi", 0),
            "days_to_earnings": fund.get("days_to_earnings", 999),
            "analyst_upside": round(fund.get("analyst_upside", 0), 1),
            "roe": round(fund.get("roe", 0), 1),
        })

    return results


def format_results_md(results: List[Dict], macro: Dict) -> str:
    """Format as actionable markdown report."""
    lines = [
        f"# Trade Ideas -- {dt.date.today().isoformat()}",
        "",
        f"**Macro:** SPY 5d={macro.get('spy_5d_ret', 0)*100:+.1f}% | VIX={macro.get('vix_level', 0):.1f} | Regime={macro.get('regime', 'unknown')}",
        f"**Ideas found:** {len(results)}",
        "",
        "## Summary",
        "",
        "| # | Ticker | Strategy | Strikes | Expiry | DTE | Credit | MaxP | MaxL | Prob | Delta | Score |",
        "|---|--------|----------|---------|--------|-----|--------|------|------|------|-------|-------|",
    ]
    for i, r in enumerate(results):
        lines.append(
            f"| {i+1} | **{r['ticker']}** | {r['strategy']} | "
            f"${r['short_strike']:.0f}/${r['long_strike']:.0f} | {r['expiry']} | {r['dte']} | "
            f"${r['credit']:.2f} | ${r['max_profit']} | ${r['max_loss']} | "
            f"{r['prob_profit']:.0f}% | {r['short_delta']:+.2f} | {r['composite']:.0f} |"
        )

    lines.extend(["", "## Trade Cards", ""])
    for r in results:
        prem_str = f"${r['total_premium']/1e6:.1f}M" if r['total_premium'] >= 1e6 else f"${r['total_premium']/1e3:.0f}K"
        lines.extend([
            f"### {r['ticker']} -- {r['strategy']} | Score: {r['composite']:.0f}",
            f"**{r['name']}** | {r['sector']} | ${r['price']:.2f} ({r['pct_from_high']:+.0f}% from 52w high)",
            "",
            f"**TRADE: Sell ${r['short_strike']:.0f} / Buy ${r['long_strike']:.0f} | {r['expiry']} | {r['dte']} DTE**",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Credit | ${r['credit']:.2f} |",
            f"| Max Profit | ${r['max_profit']} |",
            f"| Max Loss | ${r['max_loss']} |",
            f"| Prob Profit | {r['prob_profit']:.0f}% |",
            f"| Short Delta | {r['short_delta']:+.3f} |",
            f"| Short Theta | {r['short_theta']:+.3f} |",
            f"| IV | {r['short_iv']:.0f}% |",
            f"| Buffer | {r['buffer_pct']:.1f}% from price |",
            f"| Spread Width | ${r['width']:.0f} |",
            f"| Earnings | {r['days_to_earnings']} days away |",
            "",
            f"**Flow:** {prem_str} premium | {r['buy_pct']:.0f}% buy-side | Vol/OI {r['vol_oi']:.1f}x",
            f"**Quality:** ROE {r['roe']:.0f}% | Analyst upside {r['analyst_upside']:+.0f}% | Score {r['quality_score']:.0f}/100",
            "",
        ])

    return "\n".join(lines)


def format_alert(r: Dict) -> str:
    """Format a single trade idea as a notification body."""
    prem_str = f"${r['total_premium']/1e6:.1f}M" if r['total_premium'] >= 1e6 else f"${r['total_premium']/1e3:.0f}K"
    return (
        f"{r['strategy']}: Sell ${r['short_strike']:.0f}/Buy ${r['long_strike']:.0f} "
        f"{r['expiry']} | Cr ${r['credit']:.2f} | MaxP ${r['max_profit']} | "
        f"Prob {r['prob_profit']:.0f}% | {prem_str} flow"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_latest_data_dir() -> Optional[Path]:
    dirs = sorted(ROOT.glob("20??-??-??"), reverse=True)
    for d in dirs:
        if list(d.glob("bot-eod-report-*.zip")):
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description="Unified trade idea scanner")
    parser.add_argument("--date", default=None, help="UW data date (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=8, help="Number of ideas")
    parser.add_argument("--exclude", default="", help="Comma-separated tickers to exclude")
    args = parser.parse_args()

    data_dir = ROOT / args.date if args.date else find_latest_data_dir()
    exclude = set(t.strip().upper() for t in args.exclude.split(",") if t.strip())

    # Also exclude current positions from Schwab
    try:
        import json
        state_file = ROOT / "out" / "trade_analysis" / "monitor_state.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            for key in state:
                # Extract ticker from option symbol
                parts = key.strip().split()
                if parts:
                    exclude.add(parts[0].upper())
    except Exception:
        pass

    _safe_print(f"Trade Ideas Scanner -- {data_dir.name if data_dir else 'no data'}")
    if exclude:
        _safe_print(f"  Excluding: {', '.join(sorted(exclude))}")

    results = scan_trade_ideas(data_dir=data_dir, top_n=args.top,
                                exclude_tickers=exclude, verbose=True)

    if not results:
        _safe_print("  No actionable trade ideas found.")
        return

    from uwos.eod_trade_scan_mode_a import compute_macro_regime
    macro = compute_macro_regime(dt.date.today())

    report = format_results_md(results, macro)
    out_dir = ROOT / "out" / "trade_ideas"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trade-ideas-{dt.date.today().isoformat()}.md"
    out_path.write_text(report, encoding="utf-8")

    _safe_print(f"\n  Report: {out_path}")
    _safe_print(f"  Top {len(results)} ideas:\n")
    for i, r in enumerate(results):
        _safe_print(
            f"  {i+1}. {r['ticker']:6s} {r['strategy']:20s} "
            f"${r['short_strike']:.0f}/${r['long_strike']:.0f} {r['expiry']} "
            f"Cr:${r['credit']:.2f} Prob:{r['prob_profit']:.0f}% Score:{r['composite']:.0f}")


if __name__ == "__main__":
    main()
