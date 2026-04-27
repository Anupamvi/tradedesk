"""
Trade Ideas — real-time scanner using Schwab API + fundamentals + news.

ALL primary signals are LIVE from Schwab:
  - Quotes: price, 52w high/low, PE, volume (real-time)
  - Option chains: bid/ask, delta, theta, IV, OI, volume per strike (real-time)
  - Unusual activity: volume/OI ratio computed from live chains (real-time)

UW EOD data is OPTIONAL supplementary context (when available, not depended on).

Pipeline:
  1. Schwab quotes: screen S&P 500 for recovery drops and momentum breakouts
  2. Schwab chains: detect unusual option activity on top drops (LIVE)
  3. Fundamentals: quality + earnings filter from yfinance (top 20 candidates)
  4. Trade construction: specific credit/debit spreads from live chains
  5. News check: WebSearch for recent catalysts (when available)
  6. UW EOD flow: supplementary signal if zip files exist (OPTIONAL)

Usage:
    python -m uwos.trade_ideas                    # real-time scan
    python -m uwos.trade_ideas --date 2026-03-27  # add UW EOD overlay
"""

import argparse
import datetime as dt
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from uwos.paths import project_root

ROOT = project_root()

# Exclude leveraged, inverse, and non-equity tickers
JUNK_TICKERS = {
    "TECS", "SOXS", "SQQQ", "UVXY", "VXX", "SPXU", "SDOW", "TZA", "FAZ",
    "TQQQ", "SOXL", "UPRO", "SPXL", "UDOW", "TNA", "FAS", "UVIX",
    "ARKK", "ARKG", "ARKW", "ARKF",  # thematic ETFs
}

def load_sp500_universe() -> pd.DataFrame:
    """Load S&P 500 constituents from Wikipedia (with User-Agent) or cached file."""
    cache_path = ROOT / "out" / "dip_scanner" / "sp500_universe.csv"
    load_errors = []
    try:
        import io as _io
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (trade-desk-scanner)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(_io.StringIO(html), attrs={"id": "constituents"})
        df = tables[0]
        df = df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector",
                                "GICS Sub-Industry": "sub_industry",
                                "Security": "name"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        result = df[["ticker", "name", "sector", "sub_industry"]].copy()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        return result
    except Exception as exc:
        load_errors.append(f"live Wikipedia load failed: {type(exc).__name__}: {exc}")
    if cache_path.exists():
        cached = pd.read_csv(cache_path)
        if not cached.empty:
            return cached
        load_errors.append(f"cache exists but is empty: {cache_path}")
    raise RuntimeError("Could not load S&P 500 universe; " + "; ".join(load_errors))


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


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if out == out else default
    except (TypeError, ValueError):
        return default


def _score_live_flow(live_flow: Dict, uw: Optional[Dict] = None) -> float:
    """Score option-flow confirmation from live Schwab chains."""
    uw = uw or {}
    flow_score = 0
    total_prem = live_flow.get("total_premium", 0)
    vol_oi = live_flow.get("max_vol_oi", 0)

    if total_prem >= 5_000_000:
        flow_score += 12
    elif total_prem >= 1_000_000:
        flow_score += 8
    elif total_prem >= 200_000:
        flow_score += 4

    if vol_oi >= 5:
        flow_score += 10
    elif vol_oi >= 3:
        flow_score += 8
    elif vol_oi >= 2:
        flow_score += 4

    if live_flow.get("unusual"):
        flow_score += 6

    # UW EOD is only a small stale-data confirmation bonus.
    if uw.get("uw_buy_pct", 50) >= 65:
        flow_score += 2

    return min(30, float(flow_score))


def _direction_from_flow(live_flow: Dict) -> str:
    call_pct = live_flow.get("call_pct", 50)
    if call_pct >= 60:
        return "bullish"
    if call_pct <= 40:
        return "bearish"
    return "neutral"


def _macro_alignment_score(regime: str, direction: str) -> float:
    return {
        "risk_off": {"bearish": 28, "neutral": 18, "bullish": 8},
        "neutral": {"bearish": 18, "neutral": 22, "bullish": 18},
        "risk_on": {"bearish": 8, "neutral": 18, "bullish": 28},
    }.get(regime, {}).get(direction, 15)


def _dedupe_scored_by_ticker(scored: List[Dict]) -> List[Dict]:
    """Keep the strongest lane when recovery and breakout both hit a ticker."""
    best_by_ticker: Dict[str, Dict] = {}
    for cand in scored:
        ticker = cand["ticker"]
        existing = best_by_ticker.get(ticker)
        if not existing or cand["pre_score"] > existing["pre_score"]:
            best_by_ticker[ticker] = cand
    return list(best_by_ticker.values())


def _is_stock_buy_strategy(strategy: str) -> bool:
    return strategy in {"STOCK BUY", "BREAKOUT STOCK BUY"}


def _format_trade_legs(r: Dict) -> str:
    """Readable option legs in action order."""
    strategy = str(r.get("strategy", ""))
    if "Debit" in strategy:
        return f"Buy ${r['long_strike']:.0f} / Sell ${r['short_strike']:.0f}"
    return f"Sell ${r['short_strike']:.0f} / Buy ${r['long_strike']:.0f}"


def _state_underlying(key: str, value: Any) -> Optional[str]:
    if isinstance(value, dict) and value.get("underlying"):
        return str(value["underlying"]).upper()
    text = str(key).strip()
    if text.startswith("SPREAD:"):
        parts = text.split(":")
        if len(parts) > 1 and parts[1]:
            return parts[1].upper()
    parts = text.split()
    if parts:
        return parts[0].upper()
    return None


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


def screen_momentum_breakouts(svc, tickers: List[str],
                              min_market_cap: float = 10e9) -> List[Dict]:
    """Batch Schwab quotes and find bullish momentum/breakout candidates.

    This is a separate lane from the dip scanner. It looks for optionable
    large-cap stocks near 52-week highs, printing strong daily gains, or
    pushing above recent highs on elevated volume.
    """
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

        price = _num(q.get("lastPrice") or q.get("closePrice"))
        high_52w = _num(q.get("52WeekHigh"))
        day_high = _num(q.get("highPrice"))
        net_pct = _num(q.get("netPercentChange") or q.get("markPercentChange"))
        volume = _num(q.get("totalVolume"))
        avg_vol = _num(f.get("avg10DaysVolume") or f.get("avg1YearVolume"))
        shares = _num(f.get("sharesOutstanding"))
        eps = f.get("eps")
        pe = _num(f.get("peRatio"))

        if price < 20 or high_52w <= 0:
            continue
        market_cap = price * shares if shares else 0
        if 0 < market_cap < min_market_cap:
            continue
        if ref.get("optionable") is False:
            continue

        desc = str(ref.get("description", "")).lower()
        if any(x in desc for x in ["etf", "fund", "trust", "index", "proshares", "direxion"]):
            continue

        pct_from_high = (price - high_52w) / high_52w * 100
        high_touch_pct = (day_high - high_52w) / high_52w * 100 if day_high > 0 else pct_from_high
        rel_volume = (volume / avg_vol) if avg_vol > 0 else 1.0

        near_high = pct_from_high >= -8.0
        high_touch = high_touch_pct >= -1.0
        volume_breakout = net_pct >= 2.0 and rel_volume >= 1.2
        if not (near_high or high_touch or volume_breakout):
            continue

        proximity_score = max(0.0, min(18.0, 18.0 + pct_from_high * 1.8))
        day_score = max(0.0, min(10.0, net_pct * 2.5))
        volume_score = 0.0
        if rel_volume >= 2.5:
            volume_score = 8.0
        elif rel_volume >= 1.5:
            volume_score = 6.0
        elif rel_volume >= 1.1:
            volume_score = 3.0
        touch_score = 4.0 if high_touch else 0.0
        breakout_score = min(40.0, proximity_score + day_score + volume_score + touch_score)

        candidates.append({
            "ticker": ticker,
            "price": price,
            "high_52w": high_52w,
            "pct_from_high": pct_from_high,
            "pe": pe,
            "eps": _num(eps),
            "market_cap": market_cap,
            "shares": shares,
            "net_pct": net_pct,
            "rel_volume": rel_volume,
            "breakout_score": round(breakout_score, 1),
            "day_high": day_high,
        })

    candidates.sort(
        key=lambda x: (x["breakout_score"], x["net_pct"], x["rel_volume"]),
        reverse=True,
    )
    return candidates


# ---------------------------------------------------------------------------
# Step 2: Live unusual activity from Schwab chains
# ---------------------------------------------------------------------------

def detect_unusual_activity(svc, ticker: str) -> Dict:
    """Detect unusual option activity from LIVE Schwab chain data.

    Checks volume vs OI ratio, total option volume, put/call skew,
    and IV levels across all strikes — all real-time from Schwab API.
    """
    try:
        chain = svc.get_option_chain(ticker, strike_count=12,
                                      from_date=dt.date.today() + dt.timedelta(days=7),
                                      to_date=dt.date.today() + dt.timedelta(days=60))
    except Exception:
        return {"unusual": False, "total_premium": 0, "call_pct": 50,
                "max_vol_oi": 0, "avg_iv": 0, "buy_pct": 50, "hot_strikes": []}

    call_vol = 0
    put_vol = 0
    call_oi = 0
    put_oi = 0
    max_vol_oi = 0
    iv_values = []
    hot_strikes = []
    total_premium_est = 0

    for map_name, is_call in [("callExpDateMap", True), ("putExpDateMap", False)]:
        exp_map = chain.get(map_name, {})
        for expiry_key, strikes in exp_map.items():
            for strike_key, contracts in strikes.items():
                if not contracts:
                    continue
                c = contracts[0]
                vol = c.get("totalVolume", 0) or 0
                oi = c.get("openInterest", 0) or 0
                bid = c.get("bid", 0) or 0
                ask = c.get("ask", 0) or 0
                iv = c.get("volatility", 0) or 0
                mid = (bid + ask) / 2

                if is_call:
                    call_vol += vol
                    call_oi += oi
                else:
                    put_vol += vol
                    put_oi += oi

                if iv > 0:
                    iv_values.append(iv)

                # Estimate premium from volume * mid price * 100
                total_premium_est += vol * mid * 100

                # Detect unusual: volume >> OI
                if oi > 10 and vol > 50:
                    ratio = vol / oi
                    if ratio > max_vol_oi:
                        max_vol_oi = ratio
                    if ratio > 2.0:
                        hot_strikes.append({
                            "strike": float(strike_key),
                            "type": "call" if is_call else "put",
                            "volume": vol,
                            "oi": oi,
                            "ratio": round(ratio, 1),
                            "iv": round(iv, 1),
                        })

    total_vol = call_vol + put_vol
    total_oi = call_oi + put_oi
    call_pct = (call_vol / total_vol * 100) if total_vol > 0 else 50
    avg_iv = np.mean(iv_values) if iv_values else 0

    # "Buy side" approximation: if call volume > put volume in a dropped stock,
    # someone is buying calls = bullish. Not perfect but real-time.
    buy_pct = call_pct  # simplified: call-heavy = buying

    unusual = max_vol_oi > 2.0 or total_vol > total_oi * 1.5 or len(hot_strikes) >= 2

    return {
        "unusual": unusual,
        "total_premium": total_premium_est,
        "call_pct": round(call_pct, 1),
        "put_pct": round(100 - call_pct, 1),
        "max_vol_oi": round(max_vol_oi, 1),
        "avg_iv": avg_iv,
        "buy_pct": round(buy_pct, 1),
        "total_vol": total_vol,
        "total_oi": total_oi,
        "hot_strikes": sorted(hot_strikes, key=lambda x: x["ratio"], reverse=True)[:5],
    }


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(period).mean()
    avg_loss = losses.rolling(period).mean()
    if avg_loss.empty or not avg_loss.iloc[-1]:
        return 100.0 if avg_gain.iloc[-1] else 50.0
    rs = avg_gain.iloc[-1] / avg_loss.iloc[-1]
    return float(100 - (100 / (1 + rs)))


def get_technical_context(svc, ticker: str, direction: str) -> Dict:
    """Price-action timing layer from Schwab daily candles.

    Uses anchored VWAP from recent swing low/high, SMA20/SMA50, RSI14, and ATR%.
    Returns a small score used as confirmation, not as the primary signal.
    """
    empty = {
        "technical_score": 0,
        "anchor_vwap": 0.0,
        "above_anchor_vwap": False,
        "close": 0.0,
        "sma20": 0.0,
        "sma50": 0.0,
        "rsi14": 50.0,
        "atr14_pct": 0.0,
        "price_vs_sma20_pct": 0.0,
        "technical_note": "price history unavailable",
    }
    try:
        client = svc.connect()
        end_dt = dt.datetime.combine(dt.date.today() + dt.timedelta(days=1), dt.time.min)
        start_dt = end_dt - dt.timedelta(days=150)
        response = client.get_price_history_every_day(
            ticker,
            start_datetime=start_dt,
            end_datetime=end_dt,
        )
        response.raise_for_status()
        candles = response.json().get("candles", [])
    except Exception:
        return empty
    if len(candles) < 50:
        out = dict(empty)
        out["technical_note"] = "insufficient price history"
        return out

    df = pd.DataFrame(candles).copy()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["high", "low", "close", "volume"])
    if len(df) < 50:
        out = dict(empty)
        out["technical_note"] = "insufficient clean price history"
        return out

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].clip(lower=0)
    typical = (high + low + close) / 3
    last_close = float(close.iloc[-1])
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    rsi14 = _rsi(close, 14)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14 = float(true_range.rolling(14).mean().iloc[-1])
    atr14_pct = (atr14 / last_close * 100) if last_close else 0.0

    recent = df.tail(60)
    if direction == "bearish":
        anchor_idx = recent["high"].idxmax()
        anchor_label = "swing-high"
        above_anchor = last_close >= 0
    else:
        anchor_idx = recent["low"].idxmin()
        anchor_label = "swing-low"
        above_anchor = last_close >= 0

    anchor_slice = df.loc[anchor_idx:]
    anchor_vol = volume.loc[anchor_idx:]
    vol_sum = float(anchor_vol.sum())
    if vol_sum > 0:
        anchor_vwap = float((typical.loc[anchor_idx:] * anchor_vol).sum() / vol_sum)
    else:
        anchor_vwap = float(typical.loc[anchor_idx:].mean())

    score = 0
    if direction == "bearish":
        below_anchor = last_close <= anchor_vwap
        above_anchor = not below_anchor
        if below_anchor:
            score += 4
        elif last_close <= anchor_vwap * 1.01:
            score += 2
        if last_close <= sma20:
            score += 3
        if sma20 <= sma50:
            score += 1
        if 30 <= rsi14 <= 60:
            score += 2
        elif rsi14 <= 70:
            score += 1
    else:
        above_anchor = last_close >= anchor_vwap
        if above_anchor:
            score += 4
        elif last_close >= anchor_vwap * 0.99:
            score += 2
        if last_close >= sma20:
            score += 3
        if sma20 >= sma50:
            score += 1
        if 40 <= rsi14 <= 70:
            score += 2
        elif 35 <= rsi14 < 40:
            score += 1

    price_vs_sma20_pct = ((last_close - sma20) / sma20 * 100) if sma20 else 0.0
    return {
        "technical_score": int(max(0, min(10, score))),
        "anchor_vwap": round(anchor_vwap, 2),
        "anchor_label": anchor_label,
        "above_anchor_vwap": bool(above_anchor),
        "close": round(last_close, 2),
        "sma20": round(sma20, 2),
        "sma50": round(sma50, 2),
        "rsi14": round(rsi14, 1),
        "atr14_pct": round(atr14_pct, 1),
        "price_vs_sma20_pct": round(price_vs_sma20_pct, 1),
        "technical_note": (
            f"{anchor_label} AVWAP {anchor_vwap:.2f}; "
            f"SMA20 {sma20:.2f}; RSI {rsi14:.1f}"
        ),
    }


def load_uw_flow(data_dir: Path) -> Dict[str, Dict]:
    """OPTIONAL: Load UW bot-eod-report as supplementary context.

    This is stale EOD data — used only to enrich, never as primary signal.
    """
    if data_dir is None:
        return {}
    zips = list(data_dir.glob("bot-eod-report-*.zip"))
    if not zips:
        return {}

    try:
        with zipfile.ZipFile(zips[0]) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return {}
            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"), low_memory=False)
    except Exception:
        return {}

    df["premium"] = pd.to_numeric(df.get("premium"), errors="coerce").fillna(0)
    df["size"] = pd.to_numeric(df.get("size"), errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
    df["open_interest"] = pd.to_numeric(df.get("open_interest"), errors="coerce").fillna(0)
    df["implied_volatility"] = pd.to_numeric(df.get("implied_volatility"), errors="coerce").fillna(0)

    df = df[df.get("canceled", "f").astype(str) != "t"]
    df = df[(df["premium"] >= 25000) | (df["size"] >= 100)]
    if df.empty:
        return {}

    df["is_buy"] = df["side"].str.lower().str.strip() == "ask"

    result = {}
    for ticker, group in df.groupby("underlying_symbol"):
        total_prem = group["premium"].sum()
        buy_prem = group.loc[group["is_buy"], "premium"].sum()
        call_prem = group.loc[group["option_type"].str.lower() == "call", "premium"].sum()

        result[ticker] = {
            "uw_total_premium": total_prem,
            "uw_buy_pct": (buy_prem / total_prem * 100) if total_prem > 0 else 50,
            "uw_call_pct": (call_prem / total_prem * 100) if total_prem > 0 else 50,
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

MIN_CREDIT_WIDTH_RATIO = 0.20  # minimum credit must be 20% of spread width
TARGET_WIDTHS = [5, 10, 15]     # try these spread widths, pick best R:R


def _best_credit_spread(exp_map: Dict, underlying: float, side: str,
                        target_dte_min: int, target_dte_max: int) -> Optional[Dict]:
    """Find the best credit spread across multiple widths and expiries.

    Tries $5, $10, $15 widths. Picks the one with best R:R that meets
    the 20% credit/width minimum. Returns None if nothing qualifies.
    """
    is_put = (side == "put")
    candidates = []

    for expiry_key, strikes in exp_map.items():
        expiry_date = expiry_key.split(":")[0]
        try:
            dte = (dt.date.fromisoformat(expiry_date) - dt.date.today()).days
        except ValueError:
            continue
        if dte < target_dte_min or dte > target_dte_max:
            continue

        # Build a lookup of strike -> contract data
        strike_data = {}
        for sk, contracts in strikes.items():
            if not contracts:
                continue
            c = contracts[0]
            d = c.get("delta")
            if d is None:
                continue
            strike_data[float(sk)] = {
                "strike": float(sk), "bid": c.get("bid", 0) or 0,
                "ask": c.get("ask", 0) or 0, "delta": d,
                "theta": c.get("theta", 0) or 0,
                "iv": c.get("volatility", 0) or 0,
                "oi": c.get("openInterest", 0) or 0,
            }

        # Find short strike at target delta
        short = None
        if is_put:
            # Short put: -0.35 to -0.15 delta, sorted high to low
            for sk in sorted(strike_data.keys(), reverse=True):
                d = strike_data[sk]["delta"]
                bid = strike_data[sk]["bid"]
                if -0.35 <= d <= -0.15 and bid > 0.30:
                    short = strike_data[sk]
                    break
        else:
            # Short call: 0.15 to 0.35 delta, sorted low to high
            for sk in sorted(strike_data.keys()):
                d = strike_data[sk]["delta"]
                bid = strike_data[sk]["bid"]
                if 0.15 <= d <= 0.35 and bid > 0.30:
                    short = strike_data[sk]
                    break

        if not short:
            continue

        # Try each target width, pick best R:R
        for target_w in TARGET_WIDTHS:
            if is_put:
                long_strike_target = short["strike"] - target_w
            else:
                long_strike_target = short["strike"] + target_w

            # Find closest available strike to target
            long = None
            best_dist = 999
            for sk, sd in strike_data.items():
                dist = abs(sk - long_strike_target)
                # Must be on the correct side and within $2 of target
                if is_put and sk < short["strike"] and dist < best_dist and dist <= 2:
                    if sd["ask"] > 0:
                        long = sd
                        best_dist = dist
                elif not is_put and sk > short["strike"] and dist < best_dist and dist <= 2:
                    if sd["ask"] > 0:
                        long = sd
                        best_dist = dist

            if not long:
                continue

            if is_put:
                width = short["strike"] - long["strike"]
            else:
                width = long["strike"] - short["strike"]

            if width <= 0:
                continue

            credit = short["bid"] - long["ask"]
            if credit <= 0:
                continue

            credit_ratio = credit / width
            if credit_ratio < MIN_CREDIT_WIDTH_RATIO:
                continue  # R:R too poor

            # Bid-ask spread filter: reject if short leg spread > 15% of mid
            short_mid = (short["bid"] + short["ask"]) / 2
            short_spread_pct = ((short["ask"] - short["bid"]) / short_mid * 100) if short_mid > 0 else 999
            if short_spread_pct > 15:
                continue  # too illiquid

            max_profit = credit * 100
            max_loss = (width - credit) * 100
            rr = credit / (width - credit) if (width - credit) > 0 else 0

            if is_put:
                prob_profit = (1 + short["delta"]) * 100
                buffer_pct = (underlying - short["strike"]) / underlying * 100
                strategy_name = "Bull Put Credit"
            else:
                prob_profit = (1 - short["delta"]) * 100
                buffer_pct = (short["strike"] - underlying) / underlying * 100
                strategy_name = "Bear Call Credit"

            # Buffer warning
            caution = ""
            if buffer_pct < 5:
                caution = " [TIGHT]"
            elif buffer_pct < 7:
                caution = " [MOD]"

            # R:R label
            if rr >= 0.50:
                rr_label = "GOOD"
            elif rr >= 0.30:
                rr_label = "OK"
            else:
                rr_label = "THIN"

            candidates.append({
                "strategy": f"{strategy_name}{caution}",
                "short_strike": short["strike"],
                "long_strike": long["strike"],
                "expiry": expiry_date,
                "dte": dte,
                "credit": round(credit, 2),
                "max_profit": round(max_profit),
                "max_loss": round(max_loss),
                "prob_profit": round(prob_profit, 1),
                "short_delta": round(short["delta"], 3),
                "short_theta": round(short["theta"], 3),
                "short_iv": round(short["iv"], 1),
                "buffer_pct": round(buffer_pct, 1),
                "width": width,
                "rr_ratio": round(rr, 2),
                "rr_label": rr_label,
                "credit_width_pct": round(credit_ratio * 100, 1),
            })

    if not candidates:
        return None

    # Pick best R:R
    candidates.sort(key=lambda x: x["rr_ratio"], reverse=True)
    return candidates[0]


def _check_exdiv_risk(svc, ticker: str, dte: int) -> bool:
    """Check if ex-dividend falls within the trade window. Returns True if safe."""
    try:
        quotes = svc.get_quotes([ticker])
        fund = quotes.get(ticker, {}).get("fundamental", {})
        ex_date_str = fund.get("divExDate") or fund.get("nextDivExDate")
        if not ex_date_str:
            return True  # no div info = safe
        ex_date = dt.date.fromisoformat(ex_date_str[:10])
        days_to_exdiv = (ex_date - dt.date.today()).days
        # Risky if ex-div is within the trade window AND within 10 days
        if 0 < days_to_exdiv <= min(dte, 10):
            return False  # ex-div too close
    except Exception:
        pass
    return True


def construct_trades(svc, ticker: str, price: float, direction: str,
                     avg_iv: float, debit_iv_cap: float = 0.40) -> List[Dict]:
    """Fetch Schwab option chain and construct specific trade ideas.

    Tries multiple spread widths ($5, $10, $15) and picks the best
    risk/reward ratio that meets minimum credit/width of 20%.
    """
    trades = []
    target_dte_min = 25
    target_dte_max = 55

    from_date = dt.date.today() + dt.timedelta(days=target_dte_min)
    to_date = dt.date.today() + dt.timedelta(days=target_dte_max)

    try:
        chain = svc.get_option_chain(ticker, strike_count=14,
                                      from_date=from_date, to_date=to_date)
    except Exception:
        return []

    underlying = chain.get("underlyingPrice", price)

    if direction == "bullish" or direction == "neutral":
        puts = chain.get("putExpDateMap", {})
        best_put_trade = _best_credit_spread(
            exp_map=puts, underlying=underlying, side="put",
            target_dte_min=target_dte_min, target_dte_max=target_dte_max)
        if best_put_trade:
            # Ex-dividend check: skip put sells if ex-div is within trade window
            if _check_exdiv_risk(svc, ticker, best_put_trade.get("dte", 45)):
                trades.append(best_put_trade)
            else:
                best_put_trade["strategy"] += " [EX-DIV RISK]"
                trades.append(best_put_trade)  # still show but flagged

    if direction == "bearish" or direction == "neutral":
        calls = chain.get("callExpDateMap", {})
        best_call_trade = _best_credit_spread(
            exp_map=calls, underlying=underlying, side="call",
            target_dte_min=target_dte_min, target_dte_max=target_dte_max)
        if best_call_trade:
            trades.append(best_call_trade)

    # Strategy 3: Bull Call Debit — for strongly bullish flow + acceptable IV.
    # Breakout setups can tolerate a higher IV cap than slow recovery setups.
    if direction == "bullish" and avg_iv < debit_iv_cap:
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
    # load_sp500_universe is now defined in this file

    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(config=config, interactive_login=False)
    exclude = exclude_tickers or set()

    # Macro
    macro = compute_macro_regime(dt.date.today())
    regime = macro["regime"]
    if verbose:
        _safe_print(f"  [ideas] Macro: SPY 5d={macro['spy_5d_ret']:+.2%}, VIX={macro['vix_level']:.1f}, regime={regime}")

    # Step 1: Screen recovery and momentum lanes via Schwab quotes
    if verbose:
        _safe_print("  [ideas] Step 1: Schwab quote screens...")
    universe = load_sp500_universe()
    tickers = universe["ticker"].tolist()
    name_map = dict(zip(universe["ticker"], universe.get("name", universe["ticker"])))
    sector_map = dict(zip(universe["ticker"], universe.get("sector", "Unknown")))

    drops = screen_drops(svc, tickers)
    breakouts = screen_momentum_breakouts(svc, tickers)
    if verbose:
        _safe_print(f"  [ideas] Found {len(drops)} stocks down >8% from 52w high")
        _safe_print(f"  [ideas] Found {len(breakouts)} momentum/breakout candidates")

    # Remove already-held
    drops = [d for d in drops if d["ticker"] not in exclude]
    breakouts = [b for b in breakouts if b["ticker"] not in exclude]

    # Step 2: Live unusual activity from Schwab chains
    if verbose:
        _safe_print(
            "  [ideas] Step 2: Schwab live chain scan for "
            f"{min(25, len(drops))} recovery + {min(25, len(breakouts))} breakout names..."
        )

    # Optional: load UW EOD as supplementary (never primary)
    uw_flow = {}
    if data_dir and data_dir.exists():
        uw_flow = load_uw_flow(data_dir)
        if verbose and uw_flow:
            _safe_print(f"  [ideas] UW EOD supplement: {len(uw_flow)} tickers (stale, context only)")

    scored = []
    for i, d in enumerate(drops[:25]):
        ticker = d["ticker"]
        if verbose and (i + 1) % 5 == 0:
            _safe_print(f"    Scanning chains {i+1}/25...")

        # LIVE: detect unusual activity from Schwab chains
        live_flow = detect_unusual_activity(svc, ticker)

        # STALE (optional): UW EOD supplement
        uw = uw_flow.get(ticker, {})

        # Drop score (0-40)
        drop_score = min(40, abs(d["pct_from_high"]) * 1.2)

        flow_score = _score_live_flow(live_flow, uw)
        direction = _direction_from_flow(live_flow)

        technical = get_technical_context(svc, ticker, direction)
        technical_score = technical.get("technical_score", 0)

        macro_score = _macro_alignment_score(regime, direction)

        total = min(100, drop_score + flow_score + macro_score + technical_score)

        scored.append({
            **d,
            "setup_lane": "RECOVERY",
            "direction": direction,
            "drop_score": round(drop_score, 1),
            "breakout_score": 0,
            "price_move_pct": 0,
            "rel_volume": 0,
            "flow_score": round(flow_score, 1),
            "macro_score": round(macro_score, 1),
            "technical_score": round(technical_score, 1),
            "pre_score": round(total, 1),
            "flow": live_flow,
            "technical": technical,
            "name": name_map.get(ticker, ticker),
            "sector": sector_map.get(ticker, "Unknown"),
        })

    for i, b in enumerate(breakouts[:25]):
        ticker = b["ticker"]
        if verbose and (i + 1) % 5 == 0:
            _safe_print(f"    Scanning breakout chains {i+1}/25...")

        live_flow = detect_unusual_activity(svc, ticker)
        uw = uw_flow.get(ticker, {})
        flow_direction = _direction_from_flow(live_flow)
        total_prem = live_flow.get("total_premium", 0)
        call_pct = live_flow.get("call_pct", 50)

        # Do not fight strong bearish option flow on a bullish breakout.
        if flow_direction == "bearish" and total_prem >= 1_000_000:
            continue

        direction = "bullish"
        flow_score = _score_live_flow(live_flow, uw)
        if call_pct >= 60:
            flow_score = min(30, flow_score + 4)

        technical = get_technical_context(svc, ticker, direction)
        technical_score = technical.get("technical_score", 0)

        # Require price-action confirmation unless the quote breakout is very strong.
        if technical_score < 5 and b["breakout_score"] < 32:
            continue

        macro_score = _macro_alignment_score(regime, direction)
        breakout_score = b["breakout_score"]
        total = min(100, breakout_score + flow_score + macro_score + technical_score)

        scored.append({
            **b,
            "setup_lane": "BREAKOUT",
            "direction": direction,
            "drop_score": round(breakout_score, 1),
            "breakout_score": round(breakout_score, 1),
            "price_move_pct": round(b.get("net_pct", 0), 1),
            "rel_volume": round(b.get("rel_volume", 0), 2),
            "flow_score": round(flow_score, 1),
            "macro_score": round(macro_score, 1),
            "technical_score": round(technical_score, 1),
            "pre_score": round(total, 1),
            "flow": live_flow,
            "technical": technical,
            "name": name_map.get(ticker, ticker),
            "sector": sector_map.get(ticker, "Unknown"),
        })

    scored = _dedupe_scored_by_ticker(scored)
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

    # Step 3b: Sector concentration check
    # Count sectors in current portfolio (from excluded tickers) + new recommendations
    portfolio_sectors = {}
    for cand in deep:
        sector = cand.get("sector", "Unknown")
        portfolio_sectors[sector] = portfolio_sectors.get(sector, 0) + 1

    if verbose:
        _safe_print(f"  [ideas] Step 4: Schwab chains for top {min(top_n, len(deep))} ideas...")

    # Build option spread ideas
    option_results = []
    for cand in deep[:top_n]:
        ticker = cand["ticker"]
        avg_iv = cand["flow"].get("avg_iv", 0.3)
        fund = cand["fundamentals"]
        debit_iv_cap = 0.65 if cand.get("setup_lane") == "BREAKOUT" else 0.40
        trades = construct_trades(
            svc, ticker, cand["price"], cand["direction"], avg_iv,
            debit_iv_cap=debit_iv_cap,
        )
        if not trades:
            continue
        best_trade = trades[0]
        if cand.get("setup_lane") == "BREAKOUT":
            best_trade = next(
                (trade for trade in trades if trade["strategy"] == "Bull Call Debit"),
                trades[0],
            )
        option_results.append({
            "ticker": ticker,
            "setup_lane": cand.get("setup_lane", "RECOVERY"),
            "name": fund.get("name", cand.get("name", ticker)),
            "sector": fund.get("sector", cand.get("sector", "")),
            "price": cand["price"],
            "pct_from_high": round(cand["pct_from_high"], 1),
            "price_move_pct": cand.get("price_move_pct", 0),
            "rel_volume": cand.get("rel_volume", 0),
            "breakout_score": cand.get("breakout_score", 0),
            "direction": cand["direction"],
            "composite": cand["composite"],
            "quality_score": cand["quality_score"],
            "drop_score": cand["drop_score"],
            "flow_score": cand["flow_score"],
            "macro_score": cand["macro_score"],
            "technical_score": cand["technical_score"],
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
            "total_premium": cand["flow"].get("total_premium", 0),
            "buy_pct": cand["flow"].get("call_pct", 50),
            "vol_oi": cand["flow"].get("max_vol_oi", 0),
            "unusual": cand["flow"].get("unusual", False),
            "rr_ratio": best_trade.get("rr_ratio", 0),
            "rr_label": best_trade.get("rr_label", "?"),
            "credit_width_pct": best_trade.get("credit_width_pct", 0),
            "days_to_earnings": fund.get("days_to_earnings", 999),
            "analyst_upside": round(fund.get("analyst_upside", 0), 1),
            "roe": round(fund.get("roe", 0), 1),
            "rsi14": cand["technical"].get("rsi14", 50),
            "anchor_vwap": cand["technical"].get("anchor_vwap", 0),
            "above_anchor_vwap": cand["technical"].get("above_anchor_vwap", False),
            "price_vs_sma20_pct": cand["technical"].get("price_vs_sma20_pct", 0),
            "technical_note": cand["technical"].get("technical_note", ""),
        })

    # Build stock buy ideas INDEPENDENTLY (not fallback)
    # Criteria: deep value (>20% off high), high quality (>65), bullish,
    # analyst upside >15%, not within 14 days of earnings
    # Position sizing: 2% of portfolio per stock buy (assuming $600K portfolio)
    # Read actual portfolio value from Schwab account
    try:
        acct = svc.get_account_positions(account_index=0)
        PORTFOLIO_VALUE = acct.get("balances", {}).get("total_value", 600_000)
    except Exception:
        PORTFOLIO_VALUE = 600_000  # fallback
    MAX_POSITION_PCT = 0.02   # 2% per stock buy
    MAX_SECTOR_RECS = 3       # max recommendations per sector

    stock_results = []
    tickers_in_options = {r["ticker"] for r in option_results}
    sector_rec_count = {}  # track sector concentration
    for r in option_results:
        sec = r.get("sector", "Unknown")
        sector_rec_count[sec] = sector_rec_count.get(sec, 0) + 1

    for cand in deep:
        ticker = cand["ticker"]
        if ticker in tickers_in_options:
            continue  # already have an option trade for this ticker
        fund = cand["fundamentals"]
        sector = fund.get("sector", cand.get("sector", "Unknown"))

        # Sector concentration: skip if already 3+ recs in this sector
        if sector_rec_count.get(sector, 0) >= MAX_SECTOR_RECS:
            continue

        is_recovery_stock = (
            cand.get("setup_lane") == "RECOVERY"
            and cand["pct_from_high"] <= -20
            and cand["quality_score"] >= 65
            and cand["technical_score"] >= 5
            and cand["direction"] in ("bullish", "neutral")
            and fund.get("analyst_upside", 0) >= 15
            and fund.get("days_to_earnings", 999) > 14
        )
        is_breakout_stock = (
            cand.get("setup_lane") == "BREAKOUT"
            and cand["quality_score"] >= 55
            and cand["technical_score"] >= 6
            and cand["direction"] == "bullish"
            and cand.get("pct_from_high", -999) >= -8
            and cand.get("price_move_pct", 0) >= 1.5
            and fund.get("days_to_earnings", 999) > 14
        )

        if is_recovery_stock or is_breakout_stock:

            # Position sizing: 2% of portfolio / price = shares
            max_dollars = PORTFOLIO_VALUE * MAX_POSITION_PCT
            shares = int(max_dollars / cand["price"]) if cand["price"] > 0 else 0
            if shares < 1:
                continue

            sector_rec_count[sector] = sector_rec_count.get(sector, 0) + 1

            stock_results.append({
                "ticker": ticker,
                "setup_lane": cand.get("setup_lane", "RECOVERY"),
                "name": fund.get("name", cand.get("name", ticker)),
                "sector": sector,
                "price": cand["price"],
                "pct_from_high": round(cand["pct_from_high"], 1),
                "price_move_pct": cand.get("price_move_pct", 0),
                "rel_volume": cand.get("rel_volume", 0),
                "breakout_score": cand.get("breakout_score", 0),
                "direction": cand["direction"],
                "composite": cand["composite"],
                "quality_score": cand["quality_score"],
                "drop_score": cand["drop_score"],
                "flow_score": cand["flow_score"],
                "macro_score": cand["macro_score"],
                "technical_score": cand["technical_score"],
                "strategy": "BREAKOUT STOCK BUY" if is_breakout_stock else "STOCK BUY",
                "short_strike": 0, "long_strike": 0,
                "expiry": "N/A", "dte": 0, "credit": 0,
                "max_profit": 0,
                "max_loss": round(shares * cand["price"]),
                "prob_profit": 0, "short_delta": 1.0,
                "short_theta": 0, "short_iv": 0,
                "buffer_pct": 0, "width": 0,
                "shares": shares,
                "position_dollars": round(shares * cand["price"]),
                "total_premium": cand["flow"].get("total_premium", 0),
                "buy_pct": cand["flow"].get("call_pct", 50),
                "vol_oi": cand["flow"].get("max_vol_oi", 0),
                "unusual": cand["flow"].get("unusual", False),
                "rr_ratio": 0, "rr_label": "N/A", "credit_width_pct": 0,
                "days_to_earnings": fund.get("days_to_earnings", 999),
                "analyst_upside": round(fund.get("analyst_upside", 0), 1),
                "roe": round(fund.get("roe", 0), 1),
                "rsi14": cand["technical"].get("rsi14", 50),
                "anchor_vwap": cand["technical"].get("anchor_vwap", 0),
                "above_anchor_vwap": cand["technical"].get("above_anchor_vwap", False),
                "price_vs_sma20_pct": cand["technical"].get("price_vs_sma20_pct", 0),
                "technical_note": cand["technical"].get("technical_note", ""),
            })
            # No cap — if quality criteria are met, show it

    # Merge: option ideas first, then stock buys
    results = option_results[:top_n] + stock_results
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
        "| # | Lane | Ticker | Strategy | Legs | Expiry | DTE | Cr/Db | MaxP | MaxL | Prob | Score |",
        "|---|------|--------|----------|------|--------|-----|-------|------|------|------|-------|",
    ]
    for i, r in enumerate(results):
        lane = r.get("setup_lane", "RECOVERY")
        if _is_stock_buy_strategy(r["strategy"]):
            shares = r.get("shares", 0)
            pos_dollars = r.get("position_dollars", 0)
            lines.append(
                f"| {i+1} | **{lane}** | **{r['ticker']}** | **{r['strategy']}** | "
                f"@ ${r['price']:.2f} | {shares} sh | -- | "
                f"${pos_dollars:,.0f} | -- | ${pos_dollars:,.0f} | "
                f"+{r['analyst_upside']:.0f}% | {r['composite']:.0f} |")
        else:
            debit_credit = f"${r['credit']:.2f}"
            if r["credit"] < 0:
                debit_credit = f"Db ${abs(r['credit']):.2f}"
            else:
                debit_credit = f"Cr ${r['credit']:.2f}"
            lines.append(
                f"| {i+1} | **{lane}** | **{r['ticker']}** | {r['strategy']} | "
                f"{_format_trade_legs(r)} | {r['expiry']} | {r['dte']} | "
                f"{debit_credit} | ${r['max_profit']} | ${r['max_loss']} | "
                f"{r['prob_profit']:.0f}% | {r['composite']:.0f} |"
        )

    lines.extend(["", "## Trade Cards", ""])
    for r in results:
        prem_str = f"${r['total_premium']/1e6:.1f}M" if r['total_premium'] >= 1e6 else f"${r['total_premium']/1e3:.0f}K"
        lane = r.get("setup_lane", "RECOVERY")
        move = r.get("price_move_pct", 0)
        rel_vol = r.get("rel_volume", 0)
        lines.extend([
            f"### {r['ticker']} -- {lane} | {r['strategy']} | Score: {r['composite']:.0f}",
            f"**{r['name']}** | {r['sector']} | ${r['price']:.2f} ({r['pct_from_high']:+.0f}% from 52w high, day {move:+.1f}%, rel vol {rel_vol:.1f}x)",
            "",
        ])
        if _is_stock_buy_strategy(r["strategy"]):
            shares = r.get("shares", 0)
            pos_dollars = r.get("position_dollars", 0)
            lines.extend([
                f"**BUY {shares} shares at ${r['price']:.2f}** (${pos_dollars:,.0f} = 2% of portfolio) | Analyst +{r['analyst_upside']:.0f}% | Quality {r['quality_score']:.0f}/100",
                "",
                f"**Flow (LIVE):** {prem_str} est. premium | Call%: {r['buy_pct']:.0f}% | Vol/OI {r['vol_oi']:.1f}x | {'UNUSUAL' if r.get('unusual') else 'Normal'}",
                f"**Technical:** score {r.get('technical_score', 0):.0f}/10 | AVWAP ${r.get('anchor_vwap', 0):.2f} | RSI {r.get('rsi14', 50):.0f} | vs SMA20 {r.get('price_vs_sma20_pct', 0):+.1f}%",
                f"**Quality:** ROE {r['roe']:.0f}% | Analyst upside {r['analyst_upside']:+.0f}% | Score {r['quality_score']:.0f}/100",
                "",
            ])
        else:
            debit_credit = "Debit" if r["credit"] < 0 else "Credit"
            lines.extend([
                f"**TRADE: {_format_trade_legs(r)} | {r['expiry']} | {r['dte']} DTE**",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                f"| {debit_credit} | ${abs(r['credit']):.2f} |",
                f"| Max Profit | ${r['max_profit']} |",
                f"| Max Loss | ${r['max_loss']} |",
                f"| Prob Profit | {r['prob_profit']:.0f}% |",
                f"| Short Delta | {r['short_delta']:+.3f} |",
                f"| Short Theta | {r['short_theta']:+.3f} |",
                f"| IV | {r['short_iv']:.0f}% |",
                f"| Buffer | {r['buffer_pct']:.1f}% from price |",
                f"| R:R Ratio | {r.get('rr_ratio', 0):.2f} ({r.get('rr_label', '?')}) |",
                f"| Credit/Width | {r.get('credit_width_pct', 0):.0f}% |",
                f"| Spread Width | ${r['width']:.0f} |",
                f"| Earnings | {r['days_to_earnings']} days away |",
                "",
                f"**Flow (LIVE):** {prem_str} est. premium | Call%: {r['buy_pct']:.0f}% | Vol/OI {r['vol_oi']:.1f}x | {'UNUSUAL' if r.get('unusual') else 'Normal'}",
                f"**Technical:** score {r.get('technical_score', 0):.0f}/10 | AVWAP ${r.get('anchor_vwap', 0):.2f} | RSI {r.get('rsi14', 50):.0f} | vs SMA20 {r.get('price_vs_sma20_pct', 0):+.1f}%",
                f"**Quality:** ROE {r['roe']:.0f}% | Analyst upside {r['analyst_upside']:+.0f}% | Score {r['quality_score']:.0f}/100",
                "",
            ])

    return "\n".join(lines)


def format_alert(r: Dict) -> str:
    """Format a single trade idea as a notification body."""
    prem_str = f"${r['total_premium']/1e6:.1f}M" if r['total_premium'] >= 1e6 else f"${r['total_premium']/1e3:.0f}K"
    lane = r.get("setup_lane", "RECOVERY")
    if _is_stock_buy_strategy(r["strategy"]):
        shares = r.get("shares", 0)
        pos_dollars = r.get("position_dollars", 0)
        sizing = f"{shares} shares (${pos_dollars:,.0f})" if shares else "check sizing"
        return (
            f"[{lane}] BUY {r['ticker']} at ${r['price']:.2f} | {sizing} | "
            f"{r['pct_from_high']:+.0f}% from high | "
            f"Tech {r.get('technical_score', 0):.0f}/10 | Quality {r['quality_score']:.0f} | Analyst +{r['analyst_upside']:.0f}%"
        )
    debit_credit = f"Db ${abs(r['credit']):.2f}" if r["credit"] < 0 else f"Cr ${r['credit']:.2f}"
    return (
        f"[{lane}] {r['strategy']}: {_format_trade_legs(r)} "
        f"{r['expiry']} | {debit_credit} | MaxP ${r['max_profit']} | "
        f"Prob {r['prob_profit']:.0f}% | Tech {r.get('technical_score', 0):.0f}/10 | {prem_str} flow"
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
            for key, value in state.items():
                underlying = _state_underlying(key, value)
                if underlying:
                    exclude.add(underlying)
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
        lane = r.get("setup_lane", "RECOVERY")
        if _is_stock_buy_strategy(r["strategy"]):
            _safe_print(
                f"  {i+1}. {r['ticker']:6s} {lane:8s} {r['strategy']:20s} "
                f"@${r['price']:.2f} shares:{r.get('shares', 0)} Score:{r['composite']:.0f}"
            )
        else:
            debit_credit = f"Db:${abs(r['credit']):.2f}" if r["credit"] < 0 else f"Cr:${r['credit']:.2f}"
            _safe_print(
                f"  {i+1}. {r['ticker']:6s} {lane:8s} {r['strategy']:20s} "
                f"{_format_trade_legs(r)} {r['expiry']} "
                f"{debit_credit} Prob:{r['prob_profit']:.0f}% Score:{r['composite']:.0f}"
            )


if __name__ == "__main__":
    main()
