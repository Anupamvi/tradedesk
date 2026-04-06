"""
Opportunity Scanner — identifies new trade opportunities from UW flow data.

Layers UW options flow intelligence on top of fundamental quality scoring:
  Layer 1: Flow signals (volume/OI ratio, premium size, sweep activity, side bias)
  Layer 2: Fundamental quality (from dip_scanner / wheel_pipeline)
  Layer 3: Greeks validation (delta, IV, theta profile for the strategy)
  Layer 4: Macro alignment (regime-aware: don't buy calls in risk-off)

Usage:
    python -m uwos.opportunity_scanner --date 2026-03-27
    python -m uwos.opportunity_scanner                     # latest date
"""

import argparse
import csv
import datetime as dt
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("c:/uw_root")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def find_latest_data_dir() -> Optional[Path]:
    """Find the most recent date directory with UW data."""
    dirs = sorted(ROOT.glob("20??-??-??"), reverse=True)
    for d in dirs:
        if list(d.glob("bot-eod-report-*.zip")):
            return d
    return None


def load_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    """Load a CSV from inside a zip file."""
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            return pd.DataFrame()
        with zf.open(csv_names[0]) as f:
            return pd.read_csv(io.TextIOWrapper(f, encoding="utf-8"))


def load_uw_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all UW data files for a given date directory."""
    result = {}
    for name in ["bot-eod-report", "hot-chains", "stock-screener", "dp-eod-report"]:
        zips = list(data_dir.glob(f"{name}-*.zip"))
        if zips:
            result[name] = load_csv_from_zip(zips[0])
    return result


def _safe_print(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Flow analysis
# ---------------------------------------------------------------------------

def analyze_flow(bot_eod: pd.DataFrame) -> pd.DataFrame:
    """Analyze bot-eod-report for unusual flow signals.

    Returns per-ticker aggregated flow metrics.
    """
    if bot_eod.empty:
        return pd.DataFrame()

    df = bot_eod.copy()

    # Clean and type-cast
    df["premium"] = pd.to_numeric(df.get("premium"), errors="coerce").fillna(0)
    df["size"] = pd.to_numeric(df.get("size"), errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
    df["open_interest"] = pd.to_numeric(df.get("open_interest"), errors="coerce").fillna(0)
    df["implied_volatility"] = pd.to_numeric(df.get("implied_volatility"), errors="coerce").fillna(0)
    df["delta"] = pd.to_numeric(df.get("delta"), errors="coerce").fillna(0)
    df["theta"] = pd.to_numeric(df.get("theta"), errors="coerce").fillna(0)
    df["gamma"] = pd.to_numeric(df.get("gamma"), errors="coerce").fillna(0)
    df["underlying_price"] = pd.to_numeric(df.get("underlying_price"), errors="coerce").fillna(0)
    df["strike"] = pd.to_numeric(df.get("strike"), errors="coerce").fillna(0)
    df["price"] = pd.to_numeric(df.get("price"), errors="coerce").fillna(0)

    # Filter: only non-canceled, premium > $25K or size > 100
    df = df[df.get("canceled", "f").astype(str) != "t"]
    df = df[(df["premium"] >= 25000) | (df["size"] >= 100)]

    if df.empty:
        return pd.DataFrame()

    # Determine side bias: ask = buying, bid = selling
    df["is_buy"] = df["side"].str.lower().str.strip() == "ask"
    df["is_sell"] = df["side"].str.lower().str.strip() == "bid"

    # Volume/OI ratio (per contract chain)
    df["vol_oi_ratio"] = np.where(
        df["open_interest"] > 0,
        df["volume"] / df["open_interest"],
        0,
    )

    # Per-ticker aggregation
    agg = df.groupby("underlying_symbol").agg(
        total_premium=("premium", "sum"),
        total_volume=("size", "sum"),
        trade_count=("premium", "count"),
        avg_iv=("implied_volatility", "mean"),
        max_single_premium=("premium", "max"),
        buy_premium=("premium", lambda x: x[df.loc[x.index, "is_buy"]].sum()),
        sell_premium=("premium", lambda x: x[df.loc[x.index, "is_sell"]].sum()),
        call_premium=("premium", lambda x: x[df.loc[x.index, "option_type"].str.lower() == "call"].sum()),
        put_premium=("premium", lambda x: x[df.loc[x.index, "option_type"].str.lower() == "put"].sum()),
        max_vol_oi=("vol_oi_ratio", "max"),
        avg_delta=("delta", "mean"),
        sector=("sector", "first"),
        underlying_price=("underlying_price", "last"),
    ).reset_index()

    agg.rename(columns={"underlying_symbol": "ticker"}, inplace=True)

    # Derived metrics
    agg["buy_pct"] = np.where(agg["total_premium"] > 0,
                               agg["buy_premium"] / agg["total_premium"] * 100, 50)
    agg["call_pct"] = np.where(agg["total_premium"] > 0,
                                agg["call_premium"] / agg["total_premium"] * 100, 50)

    return agg


def analyze_sweeps(hot_chains: pd.DataFrame) -> Dict[str, float]:
    """Extract sweep volume by underlying ticker from hot-chains data."""
    if hot_chains.empty:
        return {}

    df = hot_chains.copy()
    df["sweep_volume"] = pd.to_numeric(df.get("sweep_volume"), errors="coerce").fillna(0)
    df["volume"] = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0)
    df["premium"] = pd.to_numeric(df.get("premium"), errors="coerce").fillna(0)

    # Extract ticker from option_symbol (first chars before digits)
    if "option_symbol" in df.columns:
        df["ticker"] = df["option_symbol"].str.extract(r"^([A-Z]+)", expand=False)
    else:
        return {}

    sweep_agg = df.groupby("ticker").agg(
        sweep_vol=("sweep_volume", "sum"),
        total_vol=("volume", "sum"),
        sweep_premium=("premium", lambda x: x[df.loc[x.index, "sweep_volume"] > 0].sum()),
    ).reset_index()

    sweep_agg["sweep_pct"] = np.where(
        sweep_agg["total_vol"] > 0,
        sweep_agg["sweep_vol"] / sweep_agg["total_vol"] * 100, 0)

    return dict(zip(sweep_agg["ticker"], sweep_agg["sweep_pct"]))


def analyze_screener(screener: pd.DataFrame) -> Dict[str, Dict]:
    """Extract put/call ratios and sentiment from stock-screener."""
    if screener.empty:
        return {}

    result = {}
    for _, row in screener.iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue
        result[ticker] = {
            "put_call_ratio": float(row.get("put_call_ratio", 1.0) or 1.0),
            "bullish_premium": float(row.get("bullish_premium", 0) or 0),
            "bearish_premium": float(row.get("bearish_premium", 0) or 0),
            "total_oi": float(row.get("total_open_interest", 0) or 0),
        }
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_flow(row: Dict, sweep_pct: float, screener_data: Dict) -> float:
    """Score flow signal quality (0-100)."""
    s = 0

    # Total premium (0-25) — institutional-size activity
    prem = row.get("total_premium", 0)
    if prem >= 10_000_000:
        s += 25
    elif prem >= 5_000_000:
        s += 20
    elif prem >= 1_000_000:
        s += 15
    elif prem >= 500_000:
        s += 10
    elif prem >= 100_000:
        s += 5

    # Buy-side dominance (0-20) — ask-side > 60% = bullish intent
    buy_pct = row.get("buy_pct", 50)
    if buy_pct >= 75:
        s += 20
    elif buy_pct >= 65:
        s += 15
    elif buy_pct >= 55:
        s += 8

    # Volume/OI ratio (0-20) — new position opening
    vol_oi = row.get("max_vol_oi", 0)
    if vol_oi >= 5:
        s += 20
    elif vol_oi >= 3:
        s += 15
    elif vol_oi >= 2:
        s += 10
    elif vol_oi >= 1.5:
        s += 5

    # Sweep activity (0-20) — urgency
    if sweep_pct >= 30:
        s += 20
    elif sweep_pct >= 20:
        s += 15
    elif sweep_pct >= 10:
        s += 8

    # Put/Call skew (0-15) — extreme readings are informative
    pcr = screener_data.get("put_call_ratio", 1.0)
    bull_prem = screener_data.get("bullish_premium", 0)
    bear_prem = screener_data.get("bearish_premium", 0)
    if pcr < 0.5 and bull_prem > bear_prem * 2:
        s += 15  # extremely bullish
    elif pcr > 2.0 and bear_prem > bull_prem * 2:
        s += 15  # extremely bearish (good for put trades)
    elif pcr < 0.7:
        s += 8
    elif pcr > 1.5:
        s += 8

    return min(100, s)


def score_greeks(row: Dict) -> Tuple[float, str]:
    """Score Greeks quality and determine trade direction (0-100, direction).

    Returns (score, 'bullish'|'bearish'|'neutral')
    """
    s = 0
    avg_delta = row.get("avg_delta", 0)
    avg_iv = row.get("avg_iv", 0)
    call_pct = row.get("call_pct", 50)

    # Direction from call/put skew + delta
    if call_pct >= 65 and avg_delta > 0.2:
        direction = "bullish"
        s += 15
    elif call_pct <= 35 and avg_delta < -0.2:
        direction = "bearish"
        s += 15
    elif call_pct >= 55:
        direction = "bullish"
        s += 8
    elif call_pct <= 45:
        direction = "bearish"
        s += 8
    else:
        direction = "neutral"
        s += 3

    # IV level (0-15) — elevated IV = premium selling opportunity
    if avg_iv >= 0.6:
        s += 15  # high IV = sell premium
    elif avg_iv >= 0.4:
        s += 10
    elif avg_iv >= 0.25:
        s += 5

    return s, direction


def score_macro_alignment(direction: str, macro_regime: str) -> float:
    """Score alignment between trade direction and macro regime (0-100)."""
    # Regime-direction matrix
    alignment = {
        ("bullish", "risk_on"): 90,
        ("bullish", "neutral"): 60,
        ("bullish", "risk_off"): 20,
        ("bearish", "risk_off"): 90,
        ("bearish", "neutral"): 60,
        ("bearish", "risk_on"): 20,
        ("neutral", "neutral"): 70,
        ("neutral", "risk_on"): 50,
        ("neutral", "risk_off"): 50,
    }
    return alignment.get((direction, macro_regime), 50)


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

def scan_opportunities(data_dir: Path, top_n: int = 10,
                       min_premium: float = 100_000,
                       verbose: bool = True) -> List[Dict]:
    """Scan UW data for new trade opportunities.

    Returns list of opportunities sorted by composite score.
    """
    from uwos.eod_trade_scan_mode_a import compute_macro_regime

    if verbose:
        _safe_print(f"  [opp] Loading UW data from {data_dir.name}...")

    uw_data = load_uw_data(data_dir)
    if "bot-eod-report" not in uw_data:
        _safe_print("  [opp] No bot-eod-report found")
        return []

    # Macro regime
    try:
        date_str = data_dir.name
        asof = dt.date.fromisoformat(date_str)
    except ValueError:
        asof = dt.date.today()
    macro = compute_macro_regime(asof)
    regime = macro["regime"]
    if verbose:
        _safe_print(f"  [opp] Macro: SPY 5d={macro['spy_5d_ret']:+.2%}, VIX={macro['vix_level']:.1f}, regime={regime}")

    # Analyze flow
    flow = analyze_flow(uw_data["bot-eod-report"])
    if flow.empty:
        _safe_print("  [opp] No significant flow found")
        return []
    if verbose:
        _safe_print(f"  [opp] Flow: {len(flow)} tickers with significant activity")

    # Filter by minimum premium
    flow = flow[flow["total_premium"] >= min_premium].copy()
    if verbose:
        _safe_print(f"  [opp] After ${min_premium/1000:.0f}K premium filter: {len(flow)} tickers")

    # Sweeps
    sweeps = {}
    if "hot-chains" in uw_data:
        sweeps = analyze_sweeps(uw_data["hot-chains"])
        if verbose:
            _safe_print(f"  [opp] Sweep data: {len(sweeps)} tickers")

    # Screener
    screener_data = {}
    if "stock-screener" in uw_data:
        screener_data = analyze_screener(uw_data["stock-screener"])
        if verbose:
            _safe_print(f"  [opp] Screener data: {len(screener_data)} tickers")

    # Score each ticker
    results = []
    for _, row in flow.iterrows():
        ticker = row["ticker"]
        row_dict = row.to_dict()

        sweep_pct = sweeps.get(ticker, 0)
        scr = screener_data.get(ticker, {})

        flow_score = score_flow(row_dict, sweep_pct, scr)
        greeks_score, direction = score_greeks(row_dict)
        macro_score = score_macro_alignment(direction, regime)

        # Composite: flow 50%, greeks 20%, macro 30%
        composite = 0.50 * flow_score + 0.20 * greeks_score + 0.30 * macro_score

        # Signal classification
        if composite >= 75:
            signal = "STRONG"
        elif composite >= 60:
            signal = "GOOD"
        elif composite >= 45:
            signal = "WATCH"
        else:
            continue

        # Suggest trade type based on direction + IV
        avg_iv = row.get("avg_iv", 0)
        if direction == "bullish":
            if avg_iv >= 0.5:
                strategy = "Bull Put Credit (sell premium)"
            else:
                strategy = "Bull Call Debit"
        elif direction == "bearish":
            if avg_iv >= 0.5:
                strategy = "Bear Call Credit (sell premium)"
            else:
                strategy = "Bear Put Debit"
        else:
            strategy = "Iron Condor / Strangle" if avg_iv >= 0.5 else "Watch"

        results.append({
            "ticker": ticker,
            "sector": row.get("sector", ""),
            "price": round(float(row.get("underlying_price", 0)), 2),
            "signal": signal,
            "direction": direction,
            "strategy": strategy,
            "composite": round(composite, 1),
            "flow_score": round(flow_score, 1),
            "greeks_score": round(greeks_score, 1),
            "macro_score": round(macro_score, 1),
            "total_premium": round(float(row.get("total_premium", 0))),
            "buy_pct": round(float(row.get("buy_pct", 50)), 1),
            "call_pct": round(float(row.get("call_pct", 50)), 1),
            "max_vol_oi": round(float(row.get("max_vol_oi", 0)), 1),
            "sweep_pct": round(sweep_pct, 1),
            "avg_iv": round(float(row.get("avg_iv", 0)) * 100, 1),
            "avg_delta": round(float(row.get("avg_delta", 0)), 3),
            "put_call_ratio": round(scr.get("put_call_ratio", 1.0), 2),
            "trade_count": int(row.get("trade_count", 0)),
        })

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results[:top_n]


def format_results_md(results: List[Dict], data_dir: str, macro: Dict) -> str:
    """Format results as markdown."""
    lines = [
        f"# Opportunity Scanner Report -- {data_dir}",
        "",
        f"**Macro:** SPY 5d={macro.get('spy_5d_ret', 0)*100:+.1f}% | VIX={macro.get('vix_level', 0):.1f} | Regime={macro.get('regime', 'unknown')}",
        f"**Opportunities found:** {len(results)}",
        "",
        "## Top Opportunities",
        "",
        "| # | Signal | Ticker | Price | Direction | Strategy | Composite | Flow | Premium | Buy% | Sweep% | IV |",
        "|---|--------|--------|-------|-----------|----------|-----------|------|---------|------|--------|-----|",
    ]
    for i, r in enumerate(results):
        prem_str = f"${r['total_premium']/1_000_000:.1f}M" if r['total_premium'] >= 1_000_000 else f"${r['total_premium']/1000:.0f}K"
        lines.append(
            f"| {i+1} | **{r['signal']}** | {r['ticker']} | ${r['price']:.2f} | "
            f"{r['direction']} | {r['strategy']} | **{r['composite']:.0f}** | "
            f"{r['flow_score']:.0f} | {prem_str} | {r['buy_pct']:.0f}% | "
            f"{r['sweep_pct']:.0f}% | {r['avg_iv']:.0f}% |"
        )

    lines.extend(["", "## Detail Cards", ""])
    for r in results:
        prem_str = f"${r['total_premium']/1_000_000:.1f}M" if r['total_premium'] >= 1_000_000 else f"${r['total_premium']/1000:.0f}K"
        lines.extend([
            f"### {r['ticker']} -- {r['signal']} | {r['direction'].upper()} | Composite: {r['composite']:.0f}",
            f"**${r['price']:.2f}** | {r['sector']} | Suggested: {r['strategy']}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Premium | {prem_str} |",
            f"| Trade Count | {r['trade_count']} |",
            f"| Buy-side % | {r['buy_pct']:.0f}% |",
            f"| Call % | {r['call_pct']:.0f}% |",
            f"| Vol/OI Max | {r['max_vol_oi']:.1f}x |",
            f"| Sweep % | {r['sweep_pct']:.0f}% |",
            f"| Avg IV | {r['avg_iv']:.0f}% |",
            f"| Avg Delta | {r['avg_delta']:+.3f} |",
            f"| Put/Call Ratio | {r['put_call_ratio']:.2f} |",
            f"| Flow Score | {r['flow_score']:.0f} |",
            f"| Greeks Score | {r['greeks_score']:.0f} |",
            f"| Macro Score | {r['macro_score']:.0f} |",
            "",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Opportunity scanner from UW flow data")
    parser.add_argument("--date", default=None, help="Date (YYYY-MM-DD) or latest")
    parser.add_argument("--top", type=int, default=10, help="Number of top results")
    parser.add_argument("--min-premium", type=float, default=100_000, help="Minimum total premium filter")
    args = parser.parse_args()

    if args.date:
        data_dir = ROOT / args.date
    else:
        data_dir = find_latest_data_dir()

    if not data_dir or not data_dir.exists():
        print(f"No data directory found: {args.date or 'latest'}")
        return

    print(f"Opportunity Scanner -- {data_dir.name}")
    results = scan_opportunities(data_dir, top_n=args.top, min_premium=args.min_premium)

    if not results:
        print("  No opportunities found.")
        return

    from uwos.eod_trade_scan_mode_a import compute_macro_regime
    try:
        asof = dt.date.fromisoformat(data_dir.name)
    except ValueError:
        asof = dt.date.today()
    macro = compute_macro_regime(asof)

    report = format_results_md(results, data_dir.name, macro)

    out_dir = ROOT / "out" / "opportunities"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"opportunities-{data_dir.name}.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"\n  Report saved: {out_path}")
    print(f"  Top {len(results)} opportunities:\n")
    for i, r in enumerate(results):
        prem_str = f"${r['total_premium']/1_000_000:.1f}M" if r['total_premium'] >= 1_000_000 else f"${r['total_premium']/1000:.0f}K"
        _safe_print(
            f"  {i+1}. [{r['signal']:6s}] {r['ticker']:6s} ${r['price']:8.2f} | "
            f"{r['direction']:7s} | {prem_str:>7s} | Buy:{r['buy_pct']:3.0f}% | "
            f"Sweep:{r['sweep_pct']:3.0f}% | Score:{r['composite']:4.0f}")


if __name__ == "__main__":
    main()
