#!/usr/bin/env python3
"""Fetch trade history from Schwab API and output as markdown / JSON."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Schwab trade history for the last N days."
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of trade history to fetch (default: 90).",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Filter transactions to a specific underlying symbol.",
    )
    parser.add_argument(
        "--account-index",
        type=int,
        default=0,
        help="Account index if you have multiple Schwab accounts (default: 0).",
    )
    parser.add_argument(
        "--out-dir",
        default=os.environ.get("SCHWAB_OUTPUT_DIR", ""),
        help="Directory to write output files. Defaults to c:/uw_root/out/trade_history.",
    )
    parser.add_argument(
        "--manual-auth",
        action="store_true",
        help="Use manual copy/paste OAuth callback flow.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output raw JSON only (skip markdown report).",
    )
    return parser.parse_args()


def _fmt_date(raw: Optional[str]) -> str:
    if not raw:
        return ""
    return str(raw)[:10]


def _fmt_money(val: Any) -> str:
    if val is None:
        return ""
    try:
        v = float(val)
        return f"${v:,.2f}"
    except (TypeError, ValueError):
        return str(val)


def _extract_legs(txn: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract transfer items (legs) from a transaction."""
    items = txn.get("transferItems", [])
    legs = []
    for item in items:
        instrument = item.get("instrument", {})
        leg: Dict[str, Any] = {
            "symbol": instrument.get("symbol", ""),
            "asset_type": instrument.get("assetType", ""),
            "description": instrument.get("description", ""),
            "underlying": instrument.get("underlyingSymbol", ""),
            "put_call": instrument.get("putCall", ""),
            "qty": item.get("amount", ""),
            "cost": item.get("cost", ""),
            "price": item.get("price", ""),
            "positionEffect": item.get("positionEffect", ""),
        }
        legs.append(leg)
    return legs


def build_markdown_report(
    transactions: List[Dict[str, Any]],
    days: int,
    symbol_filter: Optional[str],
) -> str:
    today = dt.date.today()
    lines = [
        f"# Schwab Trade History — Last {days} Days",
        f"",
        f"**Generated:** {today.isoformat()}  ",
        f"**Period:** {(today - dt.timedelta(days=days)).isoformat()} to {today.isoformat()}  ",
    ]
    if symbol_filter:
        lines.append(f"**Symbol filter:** {symbol_filter}  ")
    lines.append(f"**Total transactions:** {len(transactions)}")
    lines.append("")

    if not transactions:
        lines.append("_No trades found for this period._")
        return "\n".join(lines)

    # Summary table
    lines.append("## Trade Summary")
    lines.append("")
    lines.append("| Date | Type | Description | Net Amount | Legs |")
    lines.append("|------|------|-------------|-----------|------|")

    for txn in transactions:
        txn_date = _fmt_date(txn.get("transactionDate"))
        txn_type = txn.get("type", "")
        description = txn.get("description", "")
        net_amount = _fmt_money(txn.get("netAmount"))
        legs = _extract_legs(txn)
        leg_count = len(legs)
        lines.append(f"| {txn_date} | {txn_type} | {description} | {net_amount} | {leg_count} |")

    # Detailed legs section
    lines.append("")
    lines.append("## Trade Details")
    lines.append("")
    lines.append("| Date | Symbol | Asset | Put/Call | Qty | Price | Cost | Effect | Description |")
    lines.append("|------|--------|-------|---------|-----|-------|------|--------|-------------|")

    for txn in transactions:
        txn_date = _fmt_date(txn.get("transactionDate"))
        legs = _extract_legs(txn)
        for leg in legs:
            sym = leg["symbol"]
            asset = leg["asset_type"]
            pc = leg["put_call"]
            qty = leg["qty"]
            price = _fmt_money(leg["price"]) if leg["price"] else ""
            cost = _fmt_money(leg["cost"]) if leg["cost"] else ""
            effect = leg["positionEffect"]
            desc = leg["description"]
            lines.append(f"| {txn_date} | {sym} | {asset} | {pc} | {qty} | {price} | {cost} | {effect} | {desc} |")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    svc = SchwabLiveDataService(config=config, manual_auth=args.manual_auth)

    print(f"Fetching {args.days}-day trade history from Schwab API...")
    transactions = svc.get_trade_history(
        days=args.days,
        account_index=args.account_index,
        symbol=args.symbol,
    )
    print(f"  Auth mode: {svc.auth_mode}")
    print(f"  Transactions fetched: {len(transactions)}")

    # Determine output directory
    out_dir = Path(args.out_dir) if args.out_dir else Path("c:/uw_root/out/trade_history")
    out_dir.mkdir(parents=True, exist_ok=True)

    today_str = dt.date.today().isoformat()
    tag = f"-{args.symbol}" if args.symbol else ""

    # Always save raw JSON
    json_path = out_dir / f"trade_history_{today_str}{tag}_{args.days}d.json"
    json_path.write_text(
        json.dumps(transactions, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"  JSON saved: {json_path}")

    # Markdown report
    if not args.json_only:
        md = build_markdown_report(transactions, args.days, args.symbol)
        md_path = out_dir / f"trade-history-{today_str}{tag}-{args.days}d.md"
        md_path.write_text(md, encoding="utf-8")
        print(f"  Report saved: {md_path}")


if __name__ == "__main__":
    main()
