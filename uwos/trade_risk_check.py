"""
Trade Risk Check — ER-aware analysis for manual trade review.

Automatically flags earnings risk, IV crush exposure, and effective
hedge coverage before giving a verdict on any user-submitted trade.

Usage:
    python -m uwos.trade_risk_check --ticker NFLX --strategy "Bull Call Debit" \
        --long-strike 103 --short-strike 125 --expiry 2026-05-08 --entry-debit 4.60

Or import and call:
    from uwos.trade_risk_check import analyze_trade
    result = analyze_trade("NFLX", "Bull Call Debit", 103, 125, "2026-05-08", entry_net=4.60)
"""

import argparse
import datetime as dt
import math
import os
import sys

import numpy as np


def _get_schwab_svc():
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService
    cfg = SchwabAuthConfig(
        api_key=os.environ["SCHWAB_API_KEY"],
        app_secret=os.environ["SCHWAB_APP_SECRET"],
        token_path=os.environ.get("SCHWAB_TOKEN_PATH", "./tokens/schwab_token.json"),
    )
    return SchwabLiveDataService(cfg)


def _get_earnings_date(svc, ticker):
    """Get next earnings date from Schwab fundamentals."""
    try:
        q = svc.get_quotes([ticker])
        data = q.get(ticker, {})
        fund = data.get("fundamental", {})
        last_er = fund.get("lastEarningsDate", "")
        next_div_ex = fund.get("nextDivExDate", "")  # proxy for activity
        # Schwab doesn't always have nextEarningsDate, try yfinance
    except Exception:
        pass

    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None:
            if isinstance(cal, dict):
                er = cal.get("Earnings Date")
                if er and len(er) > 0:
                    return er[0].date() if hasattr(er[0], "date") else dt.date.fromisoformat(str(er[0])[:10])
            elif hasattr(cal, "iloc"):
                # DataFrame format
                if "Earnings Date" in cal.index:
                    val = cal.loc["Earnings Date"].iloc[0]
                    if hasattr(val, "date"):
                        return val.date()
    except Exception:
        pass

    return None


def analyze_trade(
    ticker: str,
    strategy: str,
    long_strike: float,
    short_strike: float,
    expiry: str,
    entry_net: float = None,
    today: dt.date = None,
):
    """Analyze a trade with full ER-awareness and risk stress test.

    Returns dict with:
        verdict: str ("GO", "CAUTION", "STOP")
        warnings: list of str
        er_info: dict or None
        scenarios: list of dicts (beat/miss/flat P&L)
        live_data: dict (current quotes/greeks)
    """
    if today is None:
        today = dt.date.today()
    expiry_date = dt.date.fromisoformat(expiry) if isinstance(expiry, str) else expiry
    dte = (expiry_date - today).days

    svc = _get_schwab_svc()
    warnings = []
    verdict = "GO"

    # --- 1. Live quote ---
    q = svc.get_quotes([ticker])
    data = q.get(ticker, {})
    quote = data.get("quote", {})
    spot = quote.get("lastPrice") or quote.get("closePrice", 0)

    # --- 2. Earnings check ---
    er_date = _get_earnings_date(svc, ticker)
    er_info = None
    if er_date:
        days_to_er = (er_date - today).days
        er_inside_expiry = today <= er_date <= expiry_date
        er_info = {
            "date": er_date.isoformat(),
            "days_away": days_to_er,
            "inside_expiry": er_inside_expiry,
        }
        if er_inside_expiry:
            if days_to_er <= 3:
                warnings.append(f"EARNINGS IN {days_to_er} DAYS ({er_date}) — inside expiry window. IV crush risk is extreme.")
                verdict = "STOP"
            elif days_to_er <= 7:
                warnings.append(f"EARNINGS IN {days_to_er} DAYS ({er_date}) — inside expiry window. IV will spike then crush post-ER.")
                verdict = "CAUTION"
            else:
                warnings.append(f"Earnings {er_date} ({days_to_er}d) is inside expiry. Pre-ER decision needed.")
                if verdict == "GO":
                    verdict = "CAUTION"

    # --- 3. Option chain + current spread value ---
    chains = svc.get_option_chain(ticker, strike_count=40)
    call_map = chains.get("callExpDateMap", {})
    put_map = chains.get("putExpDateMap", {})

    is_call_spread = strategy in ("Bull Call Debit", "Bear Call Credit")
    chain_map = call_map if is_call_spread else put_map

    long_leg = short_leg = None
    for exp_key, strikes in chain_map.items():
        if expiry in exp_key:
            for strike_str, contracts in strikes.items():
                s = float(strike_str)
                if abs(s - long_strike) < 0.01:
                    long_leg = contracts[0]
                if abs(s - short_strike) < 0.01:
                    short_leg = contracts[0]

    live_data = {"spot": spot, "long_leg": long_leg, "short_leg": short_leg}

    if not long_leg or not short_leg:
        warnings.append("Could not find option legs in chain — check strike/expiry.")
        return {"verdict": "STOP", "warnings": warnings, "er_info": er_info, "scenarios": [], "live_data": live_data}

    long_delta = abs(long_leg.get("delta", 0))
    short_delta = abs(short_leg.get("delta", 0))
    long_iv = long_leg.get("volatility", 0)
    short_iv = short_leg.get("volatility", 0)
    width = abs(short_strike - long_strike)

    # Current spread value
    if strategy in ("Bull Call Debit", "Bear Put Debit"):
        spread_bid = long_leg["bid"] - short_leg["ask"]
        spread_ask = long_leg["ask"] - short_leg["bid"]
    else:
        spread_bid = short_leg["bid"] - long_leg["ask"]
        spread_ask = short_leg["ask"] - long_leg["bid"]

    spread_mid = (spread_bid + spread_ask) / 2
    entry = entry_net or spread_mid

    # --- 4. Hedge effectiveness ---
    hedge_ratio = short_delta / max(long_delta, 0.01)
    if hedge_ratio < 0.20:
        warnings.append(f"Short leg delta {short_delta:.3f} is decorative (only {hedge_ratio:.0%} of long leg {long_delta:.3f}). Effectively a naked long position.")
        if verdict == "GO":
            verdict = "CAUTION"

    # --- 5. Debit/width ratio ---
    debit_pct = entry / width if width > 0 else 999
    if debit_pct > 0.50:
        warnings.append(f"Debit is {debit_pct:.0%} of width — needs >{debit_pct:.0%} of max move to profit. Poor R/R.")
        verdict = "STOP"

    # --- 6. ER scenario analysis (if ER inside expiry) ---
    scenarios = []
    if er_info and er_info["inside_expiry"]:
        # Estimate post-ER IV (typically drops 30-50%)
        post_er_iv_drop = 0.35  # 35% IV crush
        for label, move_pct in [("ER Beat +10%", 0.10), ("ER Miss -8%", -0.08), ("Flat (IV crush only)", 0.0)]:
            new_spot = spot * (1 + move_pct)

            if strategy == "Bull Call Debit":
                long_intrinsic = max(0, new_spot - long_strike)
                short_intrinsic = max(0, new_spot - short_strike)
                # Post-ER: mostly intrinsic + small time value with crushed IV
                remaining_dte = dte - er_info["days_away"]
                if remaining_dte > 5:
                    # Some extrinsic left but IV crushed
                    time_premium = entry * 0.15 * (1 - post_er_iv_drop)  # rough estimate
                else:
                    time_premium = 0
                spread_val = (long_intrinsic - short_intrinsic) + time_premium
            elif strategy == "Bear Put Debit":
                long_intrinsic = max(0, long_strike - new_spot)
                short_intrinsic = max(0, short_strike - new_spot)
                remaining_dte = dte - er_info["days_away"]
                time_premium = entry * 0.15 * (1 - post_er_iv_drop) if remaining_dte > 5 else 0
                spread_val = (long_intrinsic - short_intrinsic) + time_premium
            else:
                spread_val = entry  # placeholder for credits

            pnl = (spread_val - entry) * 100
            scenarios.append({
                "label": label,
                "new_spot": round(new_spot, 2),
                "spread_value": round(spread_val, 2),
                "pnl": round(pnl, 0),
            })

    # --- 7. Breakeven analysis ---
    if strategy == "Bull Call Debit":
        breakeven = long_strike + entry
    elif strategy == "Bear Put Debit":
        breakeven = long_strike - entry
    else:
        breakeven = short_strike - entry if "Put" in strategy else short_strike + entry

    pct_to_breakeven = abs(breakeven - spot) / spot * 100

    if pct_to_breakeven > 8:
        warnings.append(f"Breakeven {breakeven:.2f} requires {pct_to_breakeven:.1f}% move — aggressive target.")
        if verdict == "GO":
            verdict = "CAUTION"

    # --- Summary ---
    current_pnl = (spread_mid - entry) * 100 if entry else 0

    return {
        "verdict": verdict,
        "warnings": warnings,
        "er_info": er_info,
        "scenarios": scenarios,
        "live_data": {
            "spot": spot,
            "spread_bid": round(spread_bid, 2),
            "spread_mid": round(spread_mid, 2),
            "spread_ask": round(spread_ask, 2),
            "current_pnl": round(current_pnl, 0),
            "long_delta": round(long_delta, 3),
            "short_delta": round(short_delta, 3),
            "long_iv": round(long_iv, 1),
            "short_iv": round(short_iv, 1),
            "hedge_ratio": round(hedge_ratio, 2),
            "breakeven": round(breakeven, 2),
            "pct_to_breakeven": round(pct_to_breakeven, 1),
            "width": width,
            "debit_pct_width": round(debit_pct * 100, 1),
            "dte": dte,
        },
    }


def format_report(result, ticker, strategy, long_strike, short_strike, entry_net):
    """Format analysis result as readable report."""
    v = result["verdict"]
    ld = result["live_data"]
    lines = []

    # Verdict header
    icon = {"GO": "GREEN", "CAUTION": "YELLOW", "STOP": "RED"}[v]
    lines.append(f"## {icon}: {v} — {ticker} {long_strike}/{short_strike} {strategy}")
    lines.append("")

    # Warnings
    if result["warnings"]:
        for w in result["warnings"]:
            lines.append(f"- **{w}**")
        lines.append("")

    # Live data
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Spot | ${ld['spot']:.2f} |")
    lines.append(f"| Spread (bid/mid/ask) | ${ld['spread_bid']:.2f} / ${ld['spread_mid']:.2f} / ${ld['spread_ask']:.2f} |")
    lines.append(f"| Entry | ${entry_net:.2f} |")
    lines.append(f"| Current P&L | ${ld['current_pnl']:+,.0f} |")
    lines.append(f"| Breakeven | ${ld['breakeven']:.2f} ({ld['pct_to_breakeven']:+.1f}% from spot) |")
    lines.append(f"| Width | ${ld['width']:.2f} |")
    lines.append(f"| Debit/Width | {ld['debit_pct_width']:.1f}% |")
    lines.append(f"| Long delta | {ld['long_delta']:.3f} |")
    lines.append(f"| Short delta | {ld['short_delta']:.3f} |")
    lines.append(f"| Hedge ratio | {ld['hedge_ratio']:.0%} |")
    lines.append(f"| DTE | {ld['dte']} |")
    lines.append("")

    # ER info
    if result["er_info"]:
        er = result["er_info"]
        lines.append(f"**Earnings: {er['date']} ({er['days_away']}d away) — {'INSIDE expiry' if er['inside_expiry'] else 'outside expiry'}**")
        lines.append("")

    # Scenarios
    if result["scenarios"]:
        lines.append("| Scenario | NFLX Price | Spread Value | P&L |")
        lines.append("|----------|-----------|-------------|-----|")
        for s in result["scenarios"]:
            lines.append(f"| {s['label']} | ${s['new_spot']:.2f} | ${s['spread_value']:.2f} | ${s['pnl']:+,.0f} |")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Trade risk check with ER awareness")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--long-strike", type=float, required=True)
    parser.add_argument("--short-strike", type=float, required=True)
    parser.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    parser.add_argument("--entry-debit", type=float, default=None)
    args = parser.parse_args()

    result = analyze_trade(
        ticker=args.ticker,
        strategy=args.strategy,
        long_strike=args.long_strike,
        short_strike=args.short_strike,
        expiry=args.expiry,
        entry_net=args.entry_debit,
    )

    report = format_report(
        result, args.ticker, args.strategy,
        args.long_strike, args.short_strike,
        args.entry_debit or result["live_data"]["spread_mid"],
    )
    print(report)


if __name__ == "__main__":
    main()
