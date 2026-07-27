"""Honest capital/throughput arithmetic for the $10,000/month objective.

This does not assume any edge. It takes a per-contract expectancy and a trade
frequency as inputs and reports what capital and concurrency the target implies,
so the target can be checked against the risk budget rather than hoped at.
"""

from __future__ import annotations

import math

import pandas as pd

MONTHLY_TARGET = 10_000.0
RISK_BUDGET = 15_000.0
TRADING_DAYS = 21


def scenario(ev_per_contract: float, trades_per_month: float, max_loss_per_contract: float,
             hold_days: float) -> dict:
    contracts_needed = MONTHLY_TARGET / ev_per_contract if ev_per_contract > 0 else math.inf
    contracts_per_trade = contracts_needed / trades_per_month if trades_per_month else math.inf
    risk_per_trade = contracts_per_trade * max_loss_per_contract
    # Concurrency: with a `hold_days` holding period, this many trades overlap.
    overlap = max(1.0, trades_per_month * (hold_days / 30.0))
    concurrent_risk = overlap * risk_per_trade
    return {
        "EV/contract": ev_per_contract,
        "trades/mo": trades_per_month,
        "contracts/mo": contracts_needed,
        "contracts/trade": contracts_per_trade,
        "risk/trade $": risk_per_trade,
        "concurrent risk $": concurrent_risk,
        "x risk budget": concurrent_risk / RISK_BUDGET,
        "% budget/trade": risk_per_trade / RISK_BUDGET * 100.0,
    }


def main() -> None:
    print(f"target ${MONTHLY_TARGET:,.0f}/month   stated risk budget ${RISK_BUDGET:,.0f}")
    print("assumes $5-wide vertical, ~25% credit -> max loss $375/contract, ~30-day hold\n")

    rows = []
    # Frequencies span the current pipeline (4.5 guarded trades/month) up to a
    # high-throughput premium-selling design.
    for ev in (10.0, 25.0, 50.0, 100.0):
        for freq in (4.5, 20.0, 60.0, 150.0):
            r = scenario(ev, freq, 375.0, 30.0)
            rows.append(r)
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    print("\n--- what the CURRENT pipeline actually produces ---")
    cur = scenario(40.15, 4.5, 375.0, 30.0)
    print(f"  observed: 30 guarded trades / 139 sessions = 4.5 per month, avg +$40.15/contract")
    print(f"  1-contract run rate            : ${40.15 * 4.5:,.0f} / month")
    print(f"  contracts/trade to hit $10k    : {cur['contracts/trade']:,.0f}")
    print(f"  risk per trade                 : ${cur['risk/trade $']:,.0f}"
          f"  ({cur['% budget/trade']:,.0f}% of the ${RISK_BUDGET:,.0f} budget)")
    print(f"  concurrent risk required       : ${cur['concurrent risk $']:,.0f}"
          f"  ({cur['x risk budget']:,.1f}x the budget)")

    print("\n--- capital required at a 2% per-position risk cap ---")
    for ev, freq in ((25.0, 60.0), (25.0, 150.0), (50.0, 60.0), (50.0, 150.0), (100.0, 60.0)):
        r = scenario(ev, freq, 375.0, 30.0)
        acct = r["risk/trade $"] / 0.02
        print(f"  EV ${ev:>5.0f}/contract, {freq:>5.0f} trades/mo -> "
              f"{r['contracts/trade']:>5.1f} contracts/trade, risk/trade ${r['risk/trade $']:>8,.0f}, "
              f"account needed ${acct:>10,.0f}, concurrent risk ${r['concurrent risk $']:>10,.0f}")


if __name__ == "__main__":
    main()
