"""
Smart Dip Scanner — identifies high-quality stocks experiencing meaningful drops.

Not a dumb "dropped 5%" screener. Uses 4-layer scoring:
  Layer 1: Quality (fundamentals — ROE, margins, FCF, growth)
  Layer 2: Drop magnitude (RSI, distance from highs, Bollinger breach)
  Layer 3: Context (broad selloff vs company-specific)
  Layer 4: Recovery potential (mean reversion history, institutional ownership)

Usage:
    python -m uwos.dip_scanner               # scan and print
    python -m uwos.dip_scanner --top 10      # top 10 opportunities
"""

import argparse
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path("c:/uw_root")

# Sector ETF mapping for relative strength
SECTOR_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def load_sp500_universe() -> pd.DataFrame:
    """Load S&P 500 constituents from Wikipedia (with User-Agent) or cached file."""
    cache_path = ROOT / "out" / "dip_scanner" / "sp500_universe.csv"

    # Try Wikipedia first
    try:
        import io
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (trade-desk-scanner)"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html), attrs={"id": "constituents"})
        df = tables[0]
        df = df.rename(columns={"Symbol": "ticker", "GICS Sector": "sector",
                                "GICS Sub-Industry": "sub_industry",
                                "Security": "name"})
        df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
        result = df[["ticker", "name", "sector", "sub_industry"]].copy()
        # Cache for next time
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        return result
    except Exception as e:
        _safe_print_mod(f"  [universe] Wikipedia fetch failed: {e}")

    # Fallback: cached file
    if cache_path.exists():
        _safe_print_mod("  [universe] Using cached S&P 500 list")
        return pd.read_csv(cache_path)

    return pd.DataFrame(columns=["ticker", "name", "sector", "sub_industry"])


def _safe_print_mod(msg: str) -> None:
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def compute_technicals(hist: pd.DataFrame) -> Dict:
    """Compute RSI, Bollinger Bands, distance from highs using raw pandas/numpy."""
    if hist is None or len(hist) < 30:
        return {"rsi_14": 50, "below_bb": False, "pct_from_52w_high": 0,
                "pct_from_50dma": 0, "pct_from_200dma": 0, "ret_5d": 0, "ret_20d": 0}

    close = hist["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()

    if len(close) < 30:
        return {"rsi_14": 50, "below_bb": False, "pct_from_52w_high": 0,
                "pct_from_50dma": 0, "pct_from_200dma": 0, "ret_5d": 0, "ret_20d": 0}

    last = float(close.iloc[-1])

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_val = float(rsi.iloc[-1]) if not rsi.empty and np.isfinite(rsi.iloc[-1]) else 50

    # Bollinger Bands (20, 2)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_lower = float(ma20.iloc[-1] - 2 * std20.iloc[-1]) if len(ma20) >= 20 else last
    below_bb = last < bb_lower

    # Distance from 52-week high
    high_52w = float(close.rolling(min(252, len(close))).max().iloc[-1])
    pct_from_high = (last - high_52w) / high_52w * 100 if high_52w > 0 else 0

    # Distance from 50-DMA
    ma50 = float(close.rolling(min(50, len(close))).mean().iloc[-1])
    pct_from_50dma = (last - ma50) / ma50 * 100 if ma50 > 0 else 0

    # Distance from 200-DMA
    ma200 = float(close.rolling(min(200, len(close))).mean().iloc[-1]) if len(close) >= 200 else ma50
    pct_from_200dma = (last - ma200) / ma200 * 100 if ma200 > 0 else 0

    # Returns
    ret_5d = (last / float(close.iloc[-6]) - 1) * 100 if len(close) >= 6 else 0
    ret_20d = (last / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0

    return {
        "rsi_14": rsi_val,
        "below_bb": below_bb,
        "pct_from_52w_high": pct_from_high,
        "pct_from_50dma": pct_from_50dma,
        "pct_from_200dma": pct_from_200dma,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "price": last,
    }


def compute_drop_context(stock_ret_5d: float, sector_ret_5d: float,
                         spy_ret_5d: float, vix: float) -> Tuple[str, float]:
    """Classify the drop context and compute a context score (0-100).

    Returns: (context_type, context_score)
    - 'broad_selloff': stock dropped with market — best for dip buying
    - 'sector_rotation': stock's sector is weak — moderate opportunity
    - 'stock_specific': stock dropped alone — caution, may be bad news
    """
    stock_vs_spy = stock_ret_5d - spy_ret_5d
    stock_vs_sector = stock_ret_5d - sector_ret_5d

    if spy_ret_5d < -2 and vix > 22:
        # Broad selloff
        if abs(stock_vs_spy) < 3:
            # Stock dropped in line with market — best dip buy
            context = "broad_selloff"
            score = 85 + min(15, vix - 22)  # higher VIX = more fear = better buy
        else:
            # Stock dropped MORE than market — mixed
            context = "broad_selloff"
            score = 65
    elif sector_ret_5d < -3 and abs(stock_vs_sector) < 2:
        # Sector rotation — stock dropped with sector
        context = "sector_rotation"
        score = 60
    elif stock_ret_5d < -5 and spy_ret_5d > -1:
        # Stock dropped alone while market is fine — company-specific
        context = "stock_specific"
        score = 25  # cautious — could be earnings miss, bad news
    else:
        context = "neutral"
        score = 40

    return context, min(100, max(0, score))


def score_quality(fundamentals: Dict) -> float:
    """Score fundamental quality (0-100)."""
    s = 0
    n = 0

    # ROE (0-20)
    roe = fundamentals.get("roe", 0)
    if roe >= 20:
        s += 20
    elif roe >= 15:
        s += 15
    elif roe >= 10:
        s += 10
    n += 20

    # Debt/Equity (0-15)
    de = fundamentals.get("debt_equity", 999)
    if de <= 0.5:
        s += 15
    elif de <= 1.0:
        s += 10
    elif de <= 2.0:
        s += 5
    n += 15

    # Revenue Growth (0-15)
    rg = fundamentals.get("rev_growth_yoy", 0)
    if rg >= 20:
        s += 15
    elif rg >= 10:
        s += 12
    elif rg >= 5:
        s += 8
    n += 15

    # FCF Yield (0-15)
    fcf = fundamentals.get("fcf_yield", 0)
    if fcf >= 5:
        s += 15
    elif fcf >= 3:
        s += 10
    elif fcf >= 1:
        s += 5
    n += 15

    # Earnings Beats (0-15)
    beats = fundamentals.get("earnings_beats", 0)
    s += min(15, beats * 4)  # 4 points per beat, max 15
    n += 15

    # Profit Margins (0-10)
    margin = fundamentals.get("profit_margin", 0)
    if margin >= 20:
        s += 10
    elif margin >= 10:
        s += 7
    elif margin >= 5:
        s += 3
    n += 10

    # Institutional Ownership (0-10)
    inst = fundamentals.get("institutional_pct", 0)
    if inst >= 70:
        s += 10
    elif inst >= 50:
        s += 7
    elif inst >= 30:
        s += 3
    n += 10

    return (s / n * 100) if n > 0 else 0


def score_drop(technicals: Dict) -> float:
    """Score the drop magnitude (0-100). Higher = deeper oversold = better opportunity."""
    s = 0

    # RSI (0-30)
    rsi = technicals.get("rsi_14", 50)
    if rsi <= 25:
        s += 30
    elif rsi <= 30:
        s += 25
    elif rsi <= 35:
        s += 18
    elif rsi <= 40:
        s += 10

    # Distance from 52w high (0-30)
    pct_high = abs(technicals.get("pct_from_52w_high", 0))
    if pct_high >= 25:
        s += 30
    elif pct_high >= 15:
        s += 22
    elif pct_high >= 10:
        s += 15
    elif pct_high >= 5:
        s += 8

    # Below Bollinger Band (0-15)
    if technicals.get("below_bb", False):
        s += 15

    # 5-day return magnitude (0-15)
    ret_5d = abs(min(0, technicals.get("ret_5d", 0)))
    if ret_5d >= 8:
        s += 15
    elif ret_5d >= 5:
        s += 10
    elif ret_5d >= 3:
        s += 5

    # Still above 200-DMA is a positive (structural trend intact)
    if technicals.get("pct_from_200dma", 0) > 0:
        s += 10

    return min(100, s)


def score_recovery(fundamentals: Dict) -> float:
    """Score recovery potential (0-100)."""
    s = 0

    # Mean reversion rate (0-40)
    mr = fundamentals.get("mean_reversion_rate", 50)
    if mr >= 75:
        s += 40
    elif mr >= 65:
        s += 30
    elif mr >= 50:
        s += 20
    elif mr >= 35:
        s += 10

    # Institutional ownership (0-25) — smart money backing
    inst = fundamentals.get("institutional_pct", 0)
    if inst >= 70:
        s += 25
    elif inst >= 50:
        s += 18
    elif inst >= 30:
        s += 10

    # Analyst upside (0-25)
    upside = fundamentals.get("analyst_upside", 0)
    if upside >= 30:
        s += 25
    elif upside >= 20:
        s += 20
    elif upside >= 10:
        s += 12
    elif upside >= 5:
        s += 5

    # Earnings growth estimate (0-10)
    eg = fundamentals.get("earnings_growth_est", 0)
    if eg >= 20:
        s += 10
    elif eg >= 10:
        s += 7
    elif eg >= 5:
        s += 3

    return min(100, s)


def composite_score(quality: float, drop: float, context: float, recovery: float) -> float:
    """Weighted composite score (0-100)."""
    return (0.30 * quality + 0.20 * drop + 0.20 * context + 0.30 * recovery)


def scan_dips(top_n: int = 15, min_market_cap: float = 10e9,
              verbose: bool = True) -> List[Dict]:
    """Full dip scan across S&P 500 using Schwab API for speed.

    Phase 1: Schwab bulk quotes (fast) — screen for drops from 52w high + PE filter
    Phase 2: yfinance deep dive (slow) — only on top 30 candidates for fundamentals + technicals

    Returns list of opportunities sorted by composite score descending.
    """
    import yfinance as yf
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    from uwos.eod_trade_scan_mode_a import compute_macro_regime

    if verbose:
        _safe_print_mod("  [dip] Loading S&P 500 universe...")
    universe = load_sp500_universe()
    if universe.empty:
        _safe_print_mod("  [dip] Failed to load universe")
        return []
    if verbose:
        _safe_print_mod(f"  [dip] Universe: {len(universe)} stocks")

    # Get macro context
    macro = compute_macro_regime(dt.date.today())
    spy_5d = macro["spy_5d_ret"] * 100
    vix = macro["vix_level"]
    if verbose:
        _safe_print_mod(f"  [dip] Macro: SPY 5d={spy_5d:+.1f}%, VIX={vix:.1f}, regime={macro['regime']}")

    # ---- Phase 1: Schwab bulk quotes (fast) ----
    if verbose:
        _safe_print_mod("  [dip] Phase 1: Schwab bulk quotes...")

    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(config=config, interactive_login=False)

    tickers = universe["ticker"].tolist()
    sector_map = dict(zip(universe["ticker"], universe["sector"]))
    name_map = dict(zip(universe["ticker"], universe["name"]))

    # Schwab API has URL length limits — batch into chunks of 100
    all_quotes = {}
    batch_size = 100
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            batch_quotes = svc.get_quotes(batch)
            all_quotes.update(batch_quotes)
        except Exception as e:
            _safe_print_mod(f"  [dip] Schwab batch {i//batch_size + 1} failed: {e}")
    if not all_quotes:
        _safe_print_mod("  [dip] No quotes received from Schwab")
        return []

    if verbose:
        _safe_print_mod(f"  [dip] Got quotes for {len(all_quotes)} stocks")

    # Screen for drops from 52-week high
    drop_candidates = []
    for ticker, qdata in all_quotes.items():
        try:
            q = qdata.get("quote", {})
            f = qdata.get("fundamental", {})

            price = q.get("lastPrice") or q.get("closePrice")
            high_52w = q.get("52WeekHigh")
            low_52w = q.get("52WeekLow")
            net_change_pct = q.get("netPercentChange", 0)
            pe = f.get("peRatio", 0)
            eps = f.get("eps", 0)
            shares = f.get("sharesOutstanding", 0)

            if not price or not high_52w or price <= 0 or high_52w <= 0:
                continue

            pct_from_high = (price - high_52w) / high_52w * 100
            market_cap = price * shares if shares else 0

            # Filter: dropped at least 10% from 52w high, market cap > threshold
            if pct_from_high > -8:
                continue
            if market_cap < min_market_cap and market_cap > 0:
                continue
            # Skip negative earnings
            if pe is not None and pe < 0:
                continue
            if eps is not None and eps < 0:
                continue

            drop_candidates.append({
                "ticker": ticker,
                "name": name_map.get(ticker, ""),
                "sector": sector_map.get(ticker, "Unknown"),
                "price": price,
                "pct_from_52w_high": pct_from_high,
                "pe_ratio": pe or 0,
                "market_cap": market_cap,
                "net_change_pct": net_change_pct or 0,
                "high_52w": high_52w,
                "low_52w": low_52w or 0,
            })
        except Exception:
            continue

    if verbose:
        _safe_print_mod(f"  [dip] Phase 1 found {len(drop_candidates)} stocks down >8% from 52w high")

    if not drop_candidates:
        return []

    # Sort by drop magnitude, take top 30 for deep analysis
    drop_candidates.sort(key=lambda x: x["pct_from_52w_high"])
    deep_candidates = drop_candidates[:30]

    # ---- Phase 2: Deep analysis (yfinance for fundamentals + technicals) ----
    if verbose:
        _safe_print_mod(f"  [dip] Phase 2: Deep analysis on {len(deep_candidates)} candidates...")

    # Fetch sector ETF returns for context (only ~11 tickers, fast)
    sector_returns = {}
    sector_etfs = list(set(SECTOR_ETF.get(c["sector"], "SPY") for c in deep_candidates))
    try:
        sector_quotes = svc.get_quotes(sector_etfs + ["SPY"])
        for etf_sym, etf_data in sector_quotes.items():
            eq = etf_data.get("quote", {})
            sector_returns[etf_sym] = eq.get("netPercentChange", 0)
    except Exception:
        pass

    from uwos.wheel_pipeline import fetch_fundamentals

    results = []
    for i, cand in enumerate(deep_candidates):
        ticker = cand["ticker"]
        sector = cand["sector"]
        if verbose and (i + 1) % 10 == 0:
            _safe_print_mod(f"    Analyzing {i+1}/{len(deep_candidates)}...")

        try:
            # Fundamentals from yfinance (only for top 30 — much faster than 500)
            fund = fetch_fundamentals(ticker)

            # Add profit margin
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                fund["profit_margin"] = (info.get("profitMargins", 0) or 0) * 100
            except Exception:
                fund["profit_margin"] = 0

            # Override PE from Schwab (more reliable)
            if cand.get("pe_ratio"):
                fund["pe_ratio"] = cand["pe_ratio"]
            if cand.get("market_cap"):
                fund["market_cap"] = cand["market_cap"]

            # Technicals from yfinance (1y history for RSI/BB)
            try:
                hist = yf.download(ticker, period="1y", auto_adjust=True, progress=False)
                techs = compute_technicals(hist)
            except Exception:
                techs = {
                    "rsi_14": 50, "below_bb": False,
                    "pct_from_52w_high": cand["pct_from_52w_high"],
                    "pct_from_50dma": 0, "pct_from_200dma": 0,
                    "ret_5d": 0, "ret_20d": 0, "price": cand["price"],
                }

            # Use Schwab's 52w high (more accurate than yfinance computed)
            techs["pct_from_52w_high"] = cand["pct_from_52w_high"]
            techs["price"] = cand["price"]

            # Context: compare stock drop to sector ETF
            sector_etf = SECTOR_ETF.get(sector, "SPY")
            sector_ret = sector_returns.get(sector_etf, 0)
            stock_ret_5d = techs.get("ret_5d", 0)
            context_type, ctx_score = compute_drop_context(
                stock_ret_5d, sector_ret, spy_5d, vix)

            # Scores
            q_score = score_quality(fund)
            d_score = score_drop(techs)
            r_score = score_recovery(fund)
            total = composite_score(q_score, d_score, ctx_score, r_score)

            # Quality gate
            if q_score < 40:
                continue

            if total >= 80:
                signal = "STRONG BUY"
            elif total >= 65:
                signal = "BUY"
            elif total >= 50:
                signal = "WATCH"
            else:
                continue

            results.append({
                "ticker": ticker,
                "name": cand.get("name", ""),
                "sector": sector,
                "price": techs.get("price", cand["price"]),
                "ret_5d": techs.get("ret_5d", 0),
                "ret_20d": techs.get("ret_20d", 0),
                "pct_from_52w_high": cand["pct_from_52w_high"],
                "rsi_14": techs.get("rsi_14", 50),
                "below_bb": techs.get("below_bb", False),
                "above_200dma": techs.get("pct_from_200dma", 0) > 0,
                "context": context_type,
                "quality_score": round(q_score, 1),
                "drop_score": round(d_score, 1),
                "context_score": round(ctx_score, 1),
                "recovery_score": round(r_score, 1),
                "composite": round(total, 1),
                "signal": signal,
                "roe": round(fund.get("roe", 0), 1),
                "rev_growth": round(fund.get("rev_growth_yoy", 0), 1),
                "fcf_yield": round(fund.get("fcf_yield", 0), 1),
                "de_ratio": round(fund.get("debt_equity", 0), 2),
                "pe_ratio": round(fund.get("pe_ratio", 0), 1),
                "mean_reversion": round(fund.get("mean_reversion_rate", 0), 1),
                "analyst_upside": round(fund.get("analyst_upside", 0), 1),
                "institutional_pct": round(fund.get("institutional_pct", 0), 1),
            })

        except Exception:
            continue

    results.sort(key=lambda x: x["composite"], reverse=True)
    return results[:top_n]


def format_results_md(results: List[Dict], macro: Dict) -> str:
    """Format scan results as markdown."""
    lines = [
        f"# Dip Scanner Report — {dt.date.today().isoformat()}",
        "",
        f"**Macro:** SPY 5d={macro.get('spy_5d_ret', 0)*100:+.1f}% | VIX={macro.get('vix_level', 0):.1f} | Regime={macro.get('regime', 'unknown')}",
        f"**Candidates found:** {len(results)}",
        "",
        "## Top Opportunities",
        "",
        "| # | Signal | Ticker | Price | 5d Ret | RSI | From 52w High | Composite | Quality | Drop | Context | Recovery |",
        "|---|--------|--------|-------|--------|-----|--------------|-----------|---------|------|---------|----------|",
    ]
    for i, r in enumerate(results):
        lines.append(
            f"| {i+1} | **{r['signal']}** | {r['ticker']} | ${r['price']:.2f} | "
            f"{r['ret_5d']:+.1f}% | {r['rsi_14']:.0f} | {r['pct_from_52w_high']:+.1f}% | "
            f"**{r['composite']:.0f}** | {r['quality_score']:.0f} | {r['drop_score']:.0f} | "
            f"{r['context_score']:.0f} | {r['recovery_score']:.0f} |"
        )

    lines.extend(["", "## Detail Cards", ""])
    for r in results:
        bb = "Yes" if r["below_bb"] else "No"
        dma = "Above" if r["above_200dma"] else "Below"
        lines.extend([
            f"### {r['ticker']} — {r['signal']} | ${r['price']:.2f} | Composite: {r['composite']:.0f}",
            f"**{r['name']}** | {r['sector']} | Context: {r['context']}",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| 5-day return | {r['ret_5d']:+.1f}% |",
            f"| 20-day return | {r['ret_20d']:+.1f}% |",
            f"| RSI (14) | {r['rsi_14']:.0f} |",
            f"| From 52w high | {r['pct_from_52w_high']:+.1f}% |",
            f"| Below Bollinger | {bb} |",
            f"| 200-DMA | {dma} |",
            f"| ROE | {r['roe']:.1f}% |",
            f"| Revenue Growth | {r['rev_growth']:+.1f}% |",
            f"| FCF Yield | {r['fcf_yield']:.1f}% |",
            f"| D/E Ratio | {r['de_ratio']:.2f} |",
            f"| P/E | {r['pe_ratio']:.1f} |",
            f"| Mean Reversion | {r['mean_reversion']:.0f}% |",
            f"| Analyst Upside | {r['analyst_upside']:+.1f}% |",
            f"| Institutional | {r['institutional_pct']:.0f}% |",
            "",
        ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Smart dip scanner for quality stock opportunities")
    parser.add_argument("--top", type=int, default=15, help="Number of top results (default: 15)")
    parser.add_argument("--min-cap", type=float, default=10e9, help="Minimum market cap in dollars")
    args = parser.parse_args()

    print(f"Smart Dip Scanner — {dt.date.today().isoformat()}")
    results = scan_dips(top_n=args.top, min_market_cap=args.min_cap)

    if not results:
        print("  No dip opportunities found today.")
        return

    from uwos.eod_trade_scan_mode_a import compute_macro_regime
    macro = compute_macro_regime(dt.date.today())

    report = format_results_md(results, macro)

    out_dir = ROOT / "out" / "dip_scanner"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dip-scanner-{dt.date.today().isoformat()}.md"
    out_path.write_text(report, encoding="utf-8")

    print(f"\n  Report saved: {out_path}")
    print(f"  Top {len(results)} opportunities:\n")
    for i, r in enumerate(results):
        print(f"  {i+1}. [{r['signal']}] {r['ticker']:6s} ${r['price']:8.2f} | "
              f"5d: {r['ret_5d']:+5.1f}% | RSI: {r['rsi_14']:4.0f} | "
              f"Score: {r['composite']:4.0f} | {r['context']}")


if __name__ == "__main__":
    main()
