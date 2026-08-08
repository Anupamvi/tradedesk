"""One-off check that Schwab auth works end to end from this package."""

from __future__ import annotations

from claude_pipeline.schwab import SchwabClient


def main() -> None:
    client = SchwabClient()
    print(f"token file          : {client.token_path}")

    quote = client.quotes(["SPY"])
    spy = quote["SPY"]["quote"]
    print(f"SPY last / bid / ask: {spy.get('lastPrice')} / {spy.get('bidPrice')} / {spy.get('askPrice')}")
    print(f"quote time          : {quote['SPY'].get('quoteTimeInLong')}  security status: {spy.get('securityStatus')}")

    chain = client.option_chain("SPY", strike_count=4)
    print(f"chain status        : {chain.get('status')}  underlying: {chain.get('underlyingPrice')}")
    print(f"call expiries       : {len(chain.get('callExpDateMap', {}))}  put expiries: {len(chain.get('putExpDateMap', {}))}")

    first_exp = sorted(chain["callExpDateMap"])[0]
    strikes = chain["callExpDateMap"][first_exp]
    print(f"\nsample expiry {first_exp}:")
    header = f"{'symbol':<22}{'strike':>8}{'bid':>8}{'ask':>8}{'delta':>8}{'gamma':>9}{'theta':>8}{'IV':>8}{'OI':>8}{'vol':>8}"
    print(header)
    for strike in sorted(strikes, key=float)[:5]:
        c = strikes[strike][0]
        print(
            f"{c['symbol']:<22}{float(strike):>8.1f}{c['bid']:>8.2f}{c['ask']:>8.2f}"
            f"{c['delta']:>8.3f}{c['gamma']:>9.4f}{c['theta']:>8.3f}{c['volatility']:>8.2f}"
            f"{c['openInterest']:>8.0f}{c['totalVolume']:>8.0f}"
        )

    accounts = client.accounts()
    positions = sum(len(a["securitiesAccount"].get("positions", [])) for a in accounts)
    print(f"\naccounts reachable  : {len(accounts)}  open positions: {positions}")


if __name__ == "__main__":
    main()
