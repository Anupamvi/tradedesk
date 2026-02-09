import argparse
import os
from pathlib import Path
from typing import Any, Dict

from schwab_live_service import (
    DEFAULT_SYMBOLS,
    SchwabAuthConfig,
    SchwabLiveDataService,
    extract_quote_fields,
    parse_symbols,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Schwab equity quotes and option chains."
    )
    parser.add_argument(
        "symbols",
        nargs="*",
        help="Ticker symbols (space-separated). Example: AAPL MSFT SPY",
    )
    parser.add_argument(
        "--symbols-csv",
        default="",
        help="Ticker symbols as comma-separated list. Overrides positional symbols.",
    )
    parser.add_argument(
        "--chain-symbols-csv",
        default="",
        help="Comma-separated symbols for option chains. Defaults to quote symbols.",
    )
    parser.add_argument(
        "--strike-count",
        type=int,
        default=int(os.environ.get("SCHWAB_STRIKE_COUNT", "8")),
        help="How many strikes above/below ATM to request per symbol.",
    )
    parser.add_argument(
        "--manual-auth",
        action="store_true",
        help="Skip local redirect server and use copy/paste OAuth callback flow.",
    )
    parser.add_argument(
        "--no-interactive-login",
        action="store_true",
        help="Skip ENTER prompt before opening auth URL in browser (easy auth mode).",
    )
    parser.add_argument(
        "--save-json-dir",
        default=os.environ.get("SCHWAB_OUTPUT_DIR", ""),
        help="Optional directory to write raw JSON responses.",
    )
    return parser.parse_args()


def print_option_summary(symbol: str, summary: Dict[str, Any]) -> None:
    print(f"{symbol} option-chain status: {summary.get('status', 'UNKNOWN')}")
    print(f"{symbol} contracts: calls={summary.get('calls', 0)} puts={summary.get('puts', 0)}")

    sample_call = summary.get("sample_call")
    sample_put = summary.get("sample_put")
    if sample_call:
        print(
            f"{symbol} sample call: {sample_call.get('symbol')} "
            f"bid={sample_call.get('bid')} ask={sample_call.get('ask')} last={sample_call.get('last')}"
        )
    if sample_put:
        print(
            f"{symbol} sample put:  {sample_put.get('symbol')} "
            f"bid={sample_put.get('bid')} ask={sample_put.get('ask')} last={sample_put.get('last')}"
        )


def main() -> None:
    args = parse_args()
    config = SchwabAuthConfig.from_env(load_dotenv_file=True)

    symbols = parse_symbols(args.symbols_csv, args.symbols, DEFAULT_SYMBOLS)
    chain_symbols = parse_symbols(args.chain_symbols_csv, [], symbols)

    service = SchwabLiveDataService(
        config=config,
        manual_auth=args.manual_auth,
        interactive_login=not args.no_interactive_login,
    )
    snapshot = service.snapshot(
        symbols=symbols,
        chain_symbols=chain_symbols,
        strike_count=args.strike_count,
    )

    print(f"Auth mode: {snapshot.get('auth_mode', 'unknown')}")
    print("Quotes")
    print("------")
    for symbol in symbols:
        quote_payload = snapshot["quotes"].get(symbol, {})
        last_price, bid_price, ask_price = extract_quote_fields(quote_payload)
        print(f"{symbol}: last={last_price} bid={bid_price} ask={ask_price}")

    print("\nOption chains")
    print("-------------")
    for symbol in chain_symbols:
        summary = snapshot["option_chain_summary"].get(symbol, {})
        print_option_summary(symbol, summary)

    if args.save_json_dir:
        out_dir = Path(args.save_json_dir).expanduser().resolve()
        service.save_snapshot(snapshot=snapshot, out_dir=out_dir)
        print(f"\nSaved JSON output to: {out_dir}")
        print(f"Trading query context: {out_dir / 'trading_query_context.json'}")


if __name__ == "__main__":
    main()
