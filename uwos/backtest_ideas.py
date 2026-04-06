"""
Backtest Trade Ideas — replay March 2026 and score how recommendations performed.

For each trading day in March:
  1. Get the stock price as of that date (yfinance historical)
  2. Identify stocks that were down >8% from 52w high as of that date
  3. For recommended credit spreads, check if short strike was breached by expiry
  4. Compute realized P&L (max profit if OTM, max loss if deep ITM, partial otherwise)

Usage:
    python -m uwos.backtest_ideas
    python -m uwos.backtest_ideas --start 2026-03-02 --end 2026-03-27
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path("c:/uw_root")

MARCH_DATES = [
    "2026-03-02", "2026-03-03", "2026-03-04", "2026-03-05", "2026-03-06",
    "2026-03-09", "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13",
    "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19", "2026-03-20",
    "2026-03-23", "2026-03-24", "2026-03-25", "2026-03-26", "2026-03-27",
]


def _safe_print(msg: str) -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def get_historical_prices_schwab(svc, tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Download daily close prices from Schwab API. Fast and reliable."""
    client = svc.connect()
    start_dt = dt.datetime.fromisoformat(start)
    end_dt = dt.datetime.fromisoformat(end)

    all_series = {}
    for i, ticker in enumerate(tickers):
        try:
            resp = client.get_price_history_every_day(
                ticker, start_datetime=start_dt, end_datetime=end_dt)
            resp.raise_for_status()
            candles = resp.json().get("candles", [])
            if candles:
                dates = [pd.Timestamp(c["datetime"], unit="ms") for c in candles]
                closes = [c["close"] for c in candles]
                all_series[ticker] = pd.Series(closes, index=dates, name=ticker)
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            _safe_print(f"    Downloaded {i+1}/{len(tickers)} tickers...")

    if not all_series:
        return pd.DataFrame()
    return pd.DataFrame(all_series)


def simulate_credit_spread(
    ticker: str,
    entry_date: str,
    short_strike: float,
    long_strike: float,
    credit: float,
    expiry: str,
    is_put: bool,
    prices: pd.DataFrame,
) -> Dict:
    """Simulate a credit spread from entry to expiry.

    Returns dict with realized_pnl, outcome, min_price, max_price.
    """
    width = abs(short_strike - long_strike)
    max_profit = credit * 100
    max_loss = (width - credit) * 100

    if ticker not in prices.columns:
        return {"outcome": "NO_DATA", "realized_pnl": 0, "max_profit": max_profit,
                "max_loss": max_loss}

    # Get prices from entry to expiry
    try:
        entry_dt = pd.Timestamp(entry_date)
        expiry_dt = pd.Timestamp(expiry)
    except Exception:
        return {"outcome": "DATE_ERROR", "realized_pnl": 0, "max_profit": max_profit,
                "max_loss": max_loss}

    mask = (prices.index >= entry_dt) & (prices.index <= expiry_dt)
    window = prices.loc[mask, ticker].dropna()

    if len(window) < 2:
        return {"outcome": "NO_DATA", "realized_pnl": 0, "max_profit": max_profit,
                "max_loss": max_loss}

    entry_price = float(window.iloc[0])
    expiry_price = float(window.iloc[-1])
    min_price = float(window.min())
    max_price = float(window.max())

    # Determine outcome at expiry
    if is_put:
        # Bull Put Credit: profit if price stays ABOVE short strike
        if expiry_price >= short_strike:
            outcome = "MAX_PROFIT"
            realized_pnl = max_profit
        elif expiry_price <= long_strike:
            outcome = "MAX_LOSS"
            realized_pnl = -max_loss
        else:
            # Partial loss: (short_strike - expiry_price - credit) * 100
            intrinsic = (short_strike - expiry_price)
            realized_pnl = (credit - intrinsic) * 100
            outcome = "PARTIAL_LOSS" if realized_pnl < 0 else "PARTIAL_PROFIT"
    else:
        # Bear Call Credit: profit if price stays BELOW short strike
        if expiry_price <= short_strike:
            outcome = "MAX_PROFIT"
            realized_pnl = max_profit
        elif expiry_price >= long_strike:
            outcome = "MAX_LOSS"
            realized_pnl = -max_loss
        else:
            intrinsic = (expiry_price - short_strike)
            realized_pnl = (credit - intrinsic) * 100
            outcome = "PARTIAL_LOSS" if realized_pnl < 0 else "PARTIAL_PROFIT"

    # Check if short strike was breached at any point during the trade
    if is_put:
        breached = min_price < short_strike
    else:
        breached = max_price > short_strike

    return {
        "outcome": outcome,
        "realized_pnl": round(realized_pnl, 2),
        "max_profit": max_profit,
        "max_loss": max_loss,
        "entry_price": round(entry_price, 2),
        "expiry_price": round(expiry_price, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "breached_during": breached,
    }


def extract_flow_from_day(data_dir: Path) -> Dict[str, Dict]:
    """Extract flow signals from that day's UW data.

    Uses bot-eod-report if available, otherwise falls back to stock-screener
    (which has bullish/bearish premium, put/call ratio per ticker).
    """
    import zipfile, io

    # Try bot-eod-report first (full flow data)
    bot_zips = list(data_dir.glob("bot-eod-report-*.zip"))
    if bot_zips:
        try:
            df = pd.read_csv(
                io.TextIOWrapper(zipfile.ZipFile(bot_zips[0]).open(
                    [n for n in zipfile.ZipFile(bot_zips[0]).namelist() if n.endswith(".csv")][0]
                ), encoding="utf-8"), low_memory=False)
            df["premium"] = pd.to_numeric(df.get("premium"), errors="coerce").fillna(0)
            df["size"] = pd.to_numeric(df.get("size"), errors="coerce").fillna(0)
            df = df[df.get("canceled", "f").astype(str) != "t"]
            df = df[(df["premium"] >= 25000) | (df["size"] >= 100)]
            if not df.empty:
                df["is_buy"] = df["side"].str.lower().str.strip() == "ask"
                result = {}
                for ticker, g in df.groupby("underlying_symbol"):
                    tp = g["premium"].sum()
                    bp = g.loc[g["is_buy"], "premium"].sum()
                    cp = g.loc[g["option_type"].str.lower() == "call", "premium"].sum()
                    result[ticker] = {
                        "total_premium": tp,
                        "buy_pct": (bp / tp * 100) if tp > 0 else 50,
                        "call_pct": (cp / tp * 100) if tp > 0 else 50,
                    }
                return result
        except Exception:
            pass

    # Fallback: stock-screener (available on all dates)
    scr_zips = list(data_dir.glob("stock-screener-*.zip"))
    if not scr_zips:
        return {}
    try:
        df = pd.read_csv(
            io.TextIOWrapper(zipfile.ZipFile(scr_zips[0]).open(
                [n for n in zipfile.ZipFile(scr_zips[0]).namelist() if n.endswith(".csv")][0]
            ), encoding="utf-8"), low_memory=False)
    except Exception:
        return {}

    result = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue
        bull_prem = float(row.get("bullish_premium", 0) or 0)
        bear_prem = float(row.get("bearish_premium", 0) or 0)
        call_prem = float(row.get("call_premium", 0) or 0)
        put_prem = float(row.get("put_premium", 0) or 0)
        total = bull_prem + bear_prem
        if total < 100_000:
            continue
        result[ticker] = {
            "total_premium": total,
            "buy_pct": (bull_prem / total * 100) if total > 0 else 50,
            "call_pct": (call_prem / (call_prem + put_prem) * 100) if (call_prem + put_prem) > 0 else 50,
        }
    return result


def run_backtest(start_date: str = "2026-03-02", end_date: str = "2026-03-27") -> List[Dict]:
    """Replay March 2026 day by day using actual UW EOD data + yfinance prices.

    For each day:
      1. Load that day's UW flow data (what was available that day)
      2. Get stock prices as of that date from yfinance
      3. Identify quality stocks with big drops + strong flow
      4. Construct a bull put credit spread (short ~-0.25 delta, $5-15 wide)
      5. Simulate the trade: did the stock stay above the short strike by expiry?
    """
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    from uwos.trade_ideas import (get_fundamentals_and_earnings, quality_score,
                                   MIN_CREDIT_WIDTH_RATIO, TARGET_WIDTHS)

    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(config=config, interactive_login=False)

    dates_to_test = sorted([
        d for d in ROOT.iterdir()
        if d.is_dir() and d.name >= start_date and d.name <= end_date
        and (list(d.glob("bot-eod-report-*.zip")) or list(d.glob("stock-screener-*.zip")))
    ], key=lambda x: x.name)

    _safe_print(f"Backtest: {len(dates_to_test)} trading days from {start_date} to {end_date}")

    # Collect all unique tickers that appear in UW data with high flow
    _safe_print("Phase 1: Scanning all March UW data for high-flow tickers...")
    ticker_dates = {}  # ticker -> list of dates where it had strong flow
    for day_dir in dates_to_test:
        flow = extract_flow_from_day(day_dir)
        for ticker, fdata in flow.items():
            if fdata["total_premium"] >= 500_000 and fdata["buy_pct"] >= 55:
                if ticker not in ticker_dates:
                    ticker_dates[ticker] = []
                ticker_dates[ticker].append({
                    "date": day_dir.name,
                    **fdata,
                })

    _safe_print(f"Found {len(ticker_dates)} tickers with strong bullish flow in March")

    # Filter: must appear on 3+ days (persistent flow, not one-off)
    # AND total premium across all days > $2M
    persistent = {}
    for t, d in ticker_dates.items():
        if len(d) >= 3:
            total_prem = sum(x["total_premium"] for x in d)
            if total_prem >= 2_000_000:
                persistent[t] = d
    _safe_print(f"Persistent flow (3+ days, >$2M): {len(persistent)} tickers")

    # Remove index symbols, ETFs, and junk
    from uwos.trade_ideas import JUNK_TICKERS
    junk_prefixes = {"$", "^"}
    persistent = {t: d for t, d in persistent.items()
                  if t not in JUNK_TICKERS
                  and not any(t.startswith(p) for p in junk_prefixes)
                  and len(t) <= 5}
    _safe_print(f"After junk filter: {len(persistent)} tickers")

    # Download price history for all candidates
    all_tickers = list(persistent.keys())
    if not all_tickers:
        return []

    _safe_print(f"Phase 2: Downloading price history from Schwab for {len(all_tickers)} tickers...")
    prices = get_historical_prices_schwab(svc, all_tickers, start="2026-01-01", end="2026-06-30")
    if prices.empty:
        _safe_print("Price download failed")
        return []

    # Phase 3: For each ticker, simulate entering a credit spread on first strong flow day
    _safe_print("Phase 3: Simulating credit spread entries...")
    results = []

    for ticker, flow_days in sorted(persistent.items()):
        if ticker not in prices.columns:
            continue

        # Entry date: first day with strong flow
        entry_date = flow_days[0]["date"]
        entry_dt = pd.Timestamp(entry_date)

        # Get entry price
        if entry_dt not in prices.index:
            # Find closest trading day
            valid = prices.index[prices.index >= entry_dt]
            if len(valid) == 0:
                continue
            entry_dt = valid[0]

        entry_price = float(prices.loc[entry_dt, ticker])
        if entry_price <= 0 or np.isnan(entry_price):
            continue

        # Get 52w high
        one_year_ago = entry_dt - pd.Timedelta(days=365)
        hist_mask = (prices.index >= one_year_ago) & (prices.index <= entry_dt)
        hist_prices = prices.loc[hist_mask, ticker].dropna()
        if len(hist_prices) < 20:
            continue
        high_52w = float(hist_prices.max())
        pct_from_high = (entry_price - high_52w) / high_52w * 100

        # Only stocks down > 8% from 52w high
        if pct_from_high > -8:
            continue

        # Fundamentals filter
        fund = get_fundamentals_and_earnings(ticker)
        q_score = quality_score(fund)
        if q_score < 35:
            continue

        # Construct a bull put credit spread
        # Short strike: ~8-12% below entry price (0.20-0.30 delta zone)
        short_strike_pct = 0.90  # 10% below
        short_strike = round(entry_price * short_strike_pct / 5) * 5  # round to $5

        # Try widths: $5, $10, $15
        best_trade = None
        for width in TARGET_WIDTHS:
            long_strike = short_strike - width
            if long_strike <= 0:
                continue

            # Estimate credit from historical option pricing
            # Approximate: credit ~ (short_delta * stock_move_expected * 0.5)
            # Simpler: use 25-30% of width as realistic credit in elevated IV
            estimated_credit = width * 0.25  # conservative 25% credit/width
            credit_ratio = estimated_credit / width

            if credit_ratio < MIN_CREDIT_WIDTH_RATIO:
                continue

            rr = estimated_credit / (width - estimated_credit)
            best_trade = {
                "short_strike": short_strike,
                "long_strike": long_strike,
                "credit": round(estimated_credit, 2),
                "width": width,
                "rr_ratio": round(rr, 2),
            }
            if rr >= 0.30:
                break  # good enough

        if not best_trade:
            continue

        # Expiry: 30-45 DTE from entry
        expiry_dt = entry_dt + pd.Timedelta(days=37)
        # Find nearest Friday
        while expiry_dt.weekday() != 4:
            expiry_dt += pd.Timedelta(days=1)
        expiry_str = expiry_dt.strftime("%Y-%m-%d")

        # Simulate
        sim = simulate_credit_spread(
            ticker=ticker,
            entry_date=entry_date,
            short_strike=best_trade["short_strike"],
            long_strike=best_trade["long_strike"],
            credit=best_trade["credit"],
            expiry=expiry_str,
            is_put=True,
            prices=prices,
        )

        if sim["outcome"] in ("NO_DATA", "DATE_ERROR"):
            continue

        flow_summary = flow_days[0]
        results.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "strategy": "Bull Put Credit",
            "short_strike": best_trade["short_strike"],
            "long_strike": best_trade["long_strike"],
            "expiry": expiry_str,
            "credit": best_trade["credit"],
            "width": best_trade["width"],
            "rr_ratio": best_trade["rr_ratio"],
            "quality_score": round(q_score, 1),
            "pct_from_high": round(pct_from_high, 1),
            "flow_premium": flow_summary["total_premium"],
            "flow_buy_pct": round(flow_summary["buy_pct"], 1),
            "flow_days_count": len(flow_days),
            **sim,
        })

        status = "WIN" if sim["realized_pnl"] > 0 else "LOSS"
        _safe_print(f"  {ticker:6s} entry {entry_date} | ${entry_price:.0f} ({pct_from_high:+.0f}% from high) | "
                     f"${best_trade['short_strike']}/${best_trade['long_strike']} | "
                     f"expiry ${sim['expiry_price']:.0f} | P&L ${sim['realized_pnl']:.0f} {status}")

    return results


def format_backtest_md(results: List[Dict]) -> str:
    """Format backtest results as markdown."""
    if not results:
        return "# Backtest Results\n\nNo results."

    df = pd.DataFrame(results)

    # Summary stats
    total = len(df)
    wins = len(df[df["realized_pnl"] > 0])
    losses = len(df[df["realized_pnl"] < 0])
    even = total - wins - losses
    win_rate = wins / total * 100 if total > 0 else 0
    total_pnl = df["realized_pnl"].sum()
    avg_pnl = df["realized_pnl"].mean()
    avg_win = df.loc[df["realized_pnl"] > 0, "realized_pnl"].mean() if wins > 0 else 0
    avg_loss = df.loc[df["realized_pnl"] < 0, "realized_pnl"].mean() if losses > 0 else 0
    profit_factor = abs(df.loc[df["realized_pnl"] > 0, "realized_pnl"].sum() /
                        df.loc[df["realized_pnl"] < 0, "realized_pnl"].sum()) if losses > 0 else 999

    # By ticker
    by_ticker = df.groupby("ticker").agg(
        trades=("realized_pnl", "count"),
        wins=("realized_pnl", lambda x: (x > 0).sum()),
        total_pnl=("realized_pnl", "sum"),
        avg_pnl=("realized_pnl", "mean"),
    ).reset_index()
    by_ticker["win_rate"] = (by_ticker["wins"] / by_ticker["trades"] * 100).round(1)
    by_ticker = by_ticker.sort_by = by_ticker.sort_values("total_pnl", ascending=False)

    # By outcome
    by_outcome = df["outcome"].value_counts()

    lines = [
        "# Trade Ideas Backtest -- March 2026",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total simulated trades | {total} |",
        f"| Winners | {wins} ({win_rate:.0f}%) |",
        f"| Losers | {losses} |",
        f"| Breakeven | {even} |",
        f"| **Total P&L** | **${total_pnl:.0f}** |",
        f"| Avg P&L per trade | ${avg_pnl:.0f} |",
        f"| Avg win | ${avg_win:.0f} |",
        f"| Avg loss | ${avg_loss:.0f} |",
        f"| Profit factor | {profit_factor:.2f} |",
        "",
        "## Outcome Distribution",
        "",
    ]
    for outcome, count in by_outcome.items():
        lines.append(f"- {outcome}: {count}")

    lines.extend([
        "",
        "## By Ticker",
        "",
        "| Ticker | Trades | Wins | Win% | Total P&L | Avg P&L |",
        "|--------|--------|------|------|-----------|---------|",
    ])
    for _, row in by_ticker.iterrows():
        lines.append(
            f"| {row['ticker']} | {row['trades']} | {row['wins']} | "
            f"{row['win_rate']:.0f}% | ${row['total_pnl']:.0f} | ${row['avg_pnl']:.0f} |")

    lines.extend([
        "",
        "## Trade Detail",
        "",
        "| Entry | Ticker | Strategy | Strikes | Credit | Expiry Price | P&L | Outcome | Breached? |",
        "|-------|--------|----------|---------|--------|-------------|-----|---------|-----------|",
    ])
    for _, r in df.iterrows():
        lines.append(
            f"| {r['entry_date']} | {r['ticker']} | {r['strategy'][:20]} | "
            f"${r['short_strike']:.0f}/${r['long_strike']:.0f} | ${r['credit']:.2f} | "
            f"${r['expiry_price']:.2f} | ${r['realized_pnl']:.0f} | {r['outcome']} | "
            f"{'Yes' if r['breached_during'] else 'No'} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Backtest trade ideas on March 2026 data")
    parser.add_argument("--start", default="2026-03-02", help="Start date")
    parser.add_argument("--end", default="2026-03-27", help="End date")
    args = parser.parse_args()

    _safe_print(f"Trade Ideas Backtest: {args.start} to {args.end}")
    _safe_print("=" * 60)

    results = run_backtest(start_date=args.start, end_date=args.end)

    if not results:
        _safe_print("No backtest results.")
        return

    report = format_backtest_md(results)
    out_dir = ROOT / "out" / "trade_ideas"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backtest-march-2026.md"
    out_path.write_text(report, encoding="utf-8")

    # Print summary
    df = pd.DataFrame(results)
    total = len(df)
    wins = (df["realized_pnl"] > 0).sum()
    total_pnl = df["realized_pnl"].sum()

    _safe_print(f"\nResults: {total} trades, {wins} wins ({wins/total*100:.0f}%), Total P&L: ${total_pnl:.0f}")
    _safe_print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
