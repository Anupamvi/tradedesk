from __future__ import annotations

import argparse
import json
import sys

from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService, _redact_schwab_error_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch one Schwab option chain as JSON.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--strike-count", type=int, default=None)
    parser.add_argument("--from-date", default="")
    parser.add_argument("--to-date", default="")
    parser.add_argument("--timeout-seconds", default="")
    parser.add_argument("--no-include-underlying-quote", action="store_true")
    args = parser.parse_args(argv)

    try:
        service = SchwabLiveDataService(
            SchwabAuthConfig.from_env(load_dotenv_file=True),
            interactive_login=False,
        )
        payload = service.get_option_chain(
            args.symbol,
            strike_count=args.strike_count,
            include_underlying_quote=not args.no_include_underlying_quote,
            from_date=args.from_date or None,
            to_date=args.to_date or None,
        )
    except Exception as exc:
        print(_redact_schwab_error_text(exc), file=sys.stderr)
        return 1

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
