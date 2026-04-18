#!/usr/bin/env python3
"""Fetch open positions from Schwab, enrich with live data and risk metrics."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional context provider
    yf = None

from uwos.schwab_auth import (
    SchwabAuthConfig,
    SchwabLiveDataService,
    occ_underlying_symbol,
    _safe_float,
)
from uwos.paths import project_root


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

    if yf is None:
        return result

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

        # Price history (1 year)
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

        # IV Rank approximation using rolling HV
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

        # Support/resistance
        recent = hist.tail(60)
        if len(recent) >= 5:
            support = [round(float(recent["Low"].tail(20).min()), 2)]
            if result["ma_200d"]:
                support.append(result["ma_200d"])
            result["support_levels"] = sorted(support)
            result["resistance_levels"] = [round(float(recent["High"].tail(20).max()), 2)]

        # SPY correlation (20-day)
        try:
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")
            if spy_hist is not None and not spy_hist.empty:
                spy_returns = spy_hist["Close"].pct_change().dropna()
                common = returns.index.intersection(spy_returns.index)
                if len(common) >= 20:
                    corr = float(returns.loc[common].tail(20).corr(spy_returns.loc[common].tail(20)))
                    result["spy_correlation_20d"] = round(corr, 2)
        except Exception:
            pass

    except Exception:
        pass

    return result


def match_entry_details(
    positions: List[Dict[str, Any]],
    transactions: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Match open positions to their opening transactions for entry date/price.

    Returns a dict keyed by position symbol with entry_date and entry_price.
    """
    entries: Dict[str, Dict[str, Any]] = {}

    def txn_date_text(txn: Dict[str, Any]) -> str:
        return str(txn.get("transactionDate") or txn.get("tradeDate") or txn.get("time") or "")

    sorted_txns = sorted(transactions, key=txn_date_text)

    position_symbols = {p["symbol"] for p in positions}

    for txn in sorted_txns:
        for item in txn.get("transferItems", []):
            instrument = item.get("instrument", {})
            symbol = instrument.get("symbol", "")
            effect = (item.get("positionEffect") or "").upper()
            if symbol in position_symbols and effect == "OPENING" and symbol not in entries:
                txn_date = txn_date_text(txn)[:10]
                entries[symbol] = {
                    "entry_date": txn_date,
                    "entry_price": _safe_float(item.get("price")),
                }

    return entries


_OCC_RE = re.compile(r"^([A-Z\. ]{1,6})\s*(\d{6})([CP])(\d{8})$")


def parse_schwab_option_symbol(symbol: str):
    """Parse a Schwab OCC symbol into (underlying, expiry, put_call, strike) or None."""
    m = _OCC_RE.match(symbol.strip())
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
    include_yfinance: bool = False,
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

    # 3. Collect unique underlyings
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

    # 5. Fetch option chains for underlyings with option positions
    option_underlyings = set()
    for pos in positions:
        if pos["asset_type"] == "OPTION":
            ul = pos.get("underlying") or (parse_schwab_option_symbol(pos["symbol"]) or (None,))[0]
            if ul:
                option_underlyings.add(ul)

    chains_payload = {}
    for ul in option_underlyings:
        try:
            chains_payload[ul] = svc.get_option_chain(ul, strike_count=None)
        except Exception:
            pass

    # 5b. Compute GEX per underlying from full chain
    gex_by_underlying = {}
    for ul, chain_data in chains_payload.items():
        ul_spot = None
        ul_quote = quotes_payload.get(ul, {})
        ul_quote_body = ul_quote.get("quote", ul_quote)
        for fld in ("mark", "last", "close"):
            v = ul_quote_body.get(fld)
            if v is not None:
                try:
                    fv = float(v)
                    if math.isfinite(fv) and fv > 0:
                        ul_spot = fv
                        break
                except (TypeError, ValueError):
                    pass
        if ul_spot is None or ul_spot <= 0:
            continue

        total_call_gex = 0.0
        total_put_gex = 0.0
        best_put_wall = (0.0, math.nan)   # (gex_value, strike)
        best_call_wall = (0.0, math.nan)

        for map_name, side in [("callExpDateMap", "call"), ("putExpDateMap", "put")]:
            exp_map = chain_data.get(map_name, {}) or {}
            for exp_key, strike_map in exp_map.items():
                for strike_key, contracts in strike_map.items():
                    if not contracts:
                        continue
                    c = contracts[0]
                    gamma = c.get("gamma")
                    oi = c.get("openInterest")
                    if gamma is None or oi is None:
                        continue
                    try:
                        g, o = float(gamma), float(oi)
                        strike_f = float(strike_key)
                    except (TypeError, ValueError):
                        continue
                    if not (math.isfinite(g) and math.isfinite(o) and g >= 0 and o >= 0):
                        continue
                    gex = g * o * 100.0 * ul_spot
                    if side == "call":
                        total_call_gex += gex
                        if strike_f > ul_spot and gex > best_call_wall[0]:
                            best_call_wall = (gex, strike_f)
                    else:
                        total_put_gex += gex
                        if strike_f < ul_spot and gex > best_put_wall[0]:
                            best_put_wall = (gex, strike_f)

        net = total_call_gex - total_put_gex
        gex_by_underlying[ul] = {
            "net_gex": round(net, 2),
            "gex_regime": "pinned" if net >= 0 else "volatile",
            "gex_support": best_put_wall[1] if math.isfinite(best_put_wall[1]) else None,
            "gex_resistance": best_call_wall[1] if math.isfinite(best_call_wall[1]) else None,
        }

    # 6. Optional external context. Trade-desk defaults to Schwab-only; this is
    # opt-in for older workflows that still want Yahoo/yfinance enrichment.
    yf_context = {}
    if include_yfinance:
        for ul in underlyings:
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

        # Greeks from chain
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

        yf_ctx = yf_context.get(underlying, {})
        ul = pos.get("underlying", "")
        gex_info = gex_by_underlying.get(ul, {})
        computed = {
            **metrics,
            "bid_ask_spread_pct": bid_ask_spread_pct,
            "open_interest_at_strike": live_quote.get("open_interest") if live_quote else None,
            "volume_at_strike": live_quote.get("volume") if live_quote else None,
            **yf_ctx,
        }
        computed["net_gex"] = gex_info.get("net_gex")
        computed["gex_regime"] = gex_info.get("gex_regime")
        computed["gex_support"] = gex_info.get("gex_support")
        computed["gex_resistance"] = gex_info.get("gex_resistance")
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
            "computed": computed,
        })

    return {
        "as_of": dt.datetime.now(dt.timezone.utc).isoformat(),
        "account_summary": balances,
        "context_sources": {
            "positions": "schwab",
            "transactions": "schwab",
            "quotes": "schwab",
            "option_chains": "schwab",
            "external_yfinance": bool(include_yfinance),
        },
        "positions": enriched,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze open Schwab positions with risk metrics and market context."
    )
    parser.add_argument("--days", type=int, default=90, help="Days of trade history for entry matching (default: 90).")
    parser.add_argument("--symbol", default=None, help="Filter to a specific underlying symbol.")
    parser.add_argument("--account-index", type=int, default=0, help="Account index (default: 0).")
    parser.add_argument("--out-dir", default="", help="Output directory (default: repo out/trade_analysis).")
    parser.add_argument("--manual-auth", action="store_true", help="Use manual OAuth flow.")
    parser.add_argument("--with-yfinance-context", action="store_true", help="Opt in to Yahoo/yfinance enrichment.")
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
        include_yfinance=bool(args.with_yfinance_context),
    )

    n_pos = len(result["positions"])
    print(f"  Auth mode: {svc.auth_mode}")
    print(f"  Open positions found: {n_pos}")

    out_dir = Path(args.out_dir) if args.out_dir else project_root() / "out" / "trade_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    today_str = dt.date.today().isoformat()
    json_path = out_dir / f"position_data_{today_str}.json"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  Position data saved: {json_path}")


if __name__ == "__main__":
    main()
