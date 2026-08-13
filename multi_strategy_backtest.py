"""Self-contained multi-strategy options backtester over local UW dated chains.

Generates every strategy family (long call/put, short put, the 4 verticals,
straddle, strangle, call/put butterfly, iron butterfly, calendar, diagonal) for a
liquid ticker universe on each dated folder, then evaluates P&L over a 5-session
horizon with a 60% profit-target / 2x (or 60%) stop and time-exit — the same
management spirit as the production replay. Uses HistoricalOptionQuoteStore for
point-in-time entry + exit repricing. No Schwab account, no live pipeline touched.

Leg convention: each leg is (right, strike, expiry, qty) with signed qty
(+long / -short, contracts). Open (conservative fill): long pays ask, short
receives bid. Close (unwind): long sells at bid, short buys back at ask.
    open_debit   = Σ_long q*ask  - Σ_short|q|*bid      (>0 net debit, <0 net credit)
    close_value  = Σ_long q*bid  - Σ_short|q|*ask      (unwind proceeds)
    pnl(1x)      = (close_value - open_debit) * 100 - commission_per_leg*n_legs
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from uwos.exact_spread_backtester import (
    HistoricalOptionQuoteStore,
    UnderlyingCloseStore,
    build_occ_symbol,
)

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path("/Users/anuppamvi/uw_root/tradedesk")
# Fill realism: fraction of the half-spread paid as slippage per leg (0=mid, 1=full natural).
FILL_FRAC = float(next((a.split("=")[1] for a in sys.argv if a.startswith("--fill=")), 1.0))
HOT_ONLY = "--hot-only" in sys.argv
_out_arg = next((a for a in sys.argv[2:] if not a.startswith("--")), None)
OUT = Path(_out_arg) if _out_arg else ROOT / (
    f"multi_strategy_detail_{'hot' if HOT_ONLY else 'all'}_fill{int(FILL_FRAC*100):03d}.csv")

HORIZON = 5                    # sessions
COMMISSION_PER_LEG = 0.65      # round-trip approx per leg
UNIVERSE_PER_DAY = 45          # top liquid tickers/day
TARGET_DTE = 30
DTE_LO, DTE_HI = 18, 55
FAR_DTE_LO, FAR_DTE_HI = 45, 90

OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


@dataclass
class Leg:
    right: str      # "C"/"P"
    strike: float
    expiry: dt.date
    qty: int        # signed: +long / -short


def parse_chain(day_quotes: pd.DataFrame) -> pd.DataFrame:
    s = day_quotes["option_symbol"].astype(str).str.replace(" ", "", regex=False)
    ext = s.str.extract(OCC_RE)
    ext.columns = ["under", "exp", "right", "strike_raw"]
    df = day_quotes.copy()
    df["under"] = ext["under"]
    df["right"] = ext["right"]
    df["strike"] = pd.to_numeric(ext["strike_raw"], errors="coerce") / 1000.0
    df["exp_dt"] = pd.to_datetime(ext["exp"], format="%y%m%d", errors="coerce").dt.date
    df["liq"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0) + \
        pd.to_numeric(df["open_interest"], errors="coerce").fillna(0)
    return df.dropna(subset=["under", "right", "strike", "exp_dt"])


def nearest(strikes: np.ndarray, target: float):
    if len(strikes) == 0:
        return None
    return float(strikes[int(np.argmin(np.abs(strikes - target)))])


def pick_expiry(exps: list[dt.date], asof: dt.date, lo: int, hi: int, target: int):
    cand = [(e, (e - asof).days) for e in exps]
    cand = [(e, d) for e, d in cand if lo <= d <= hi]
    if not cand:
        return None
    return min(cand, key=lambda x: abs(x[1] - target))[0]


def build_strategies(under: str, spot: float, chain_u: pd.DataFrame, asof: dt.date):
    """Return dict[strategy_route] -> list[Leg] using strikes available in chain_u."""
    exps = sorted(chain_u["exp_dt"].unique())
    exp = pick_expiry(exps, asof, DTE_LO, DTE_HI, TARGET_DTE)
    if exp is None:
        return {}
    far = pick_expiry(exps, asof, FAR_DTE_LO, FAR_DTE_HI, 60)
    ce = chain_u[(chain_u["exp_dt"] == exp) & (chain_u["right"] == "C")]
    pe = chain_u[(chain_u["exp_dt"] == exp) & (chain_u["right"] == "P")]
    cs = np.sort(ce["strike"].unique())
    ps = np.sort(pe["strike"].unique())
    if len(cs) < 3 or len(ps) < 3:
        return {}

    atm_c = nearest(cs, spot)
    atm_p = nearest(ps, spot)
    otm_c = nearest(cs, spot * 1.05)
    otm_c2 = nearest(cs, spot * 1.10)
    otm_p = nearest(ps, spot * 0.95)
    otm_p2 = nearest(ps, spot * 0.90)
    itm_c = nearest(cs, spot * 0.95)
    wing = max(spot * 0.05, 1.0)
    lo_c = nearest(cs, atm_c - wing)
    hi_c = nearest(cs, atm_c + wing)
    lo_p = nearest(ps, atm_p - wing)
    hi_p = nearest(ps, atm_p + wing)

    S = {}
    S["long_call"] = [Leg("C", atm_c, exp, +1)]
    S["long_put"] = [Leg("P", atm_p, exp, +1)]
    S["short_put"] = [Leg("P", otm_p, exp, -1)]
    if otm_c > atm_c:
        S["bull_call_debit"] = [Leg("C", atm_c, exp, +1), Leg("C", otm_c, exp, -1)]
        S["bear_call_credit"] = [Leg("C", otm_c, exp, -1), Leg("C", otm_c2, exp, +1)] if otm_c2 > otm_c else None
    if otm_p < atm_p:
        S["bear_put_debit"] = [Leg("P", atm_p, exp, +1), Leg("P", otm_p, exp, -1)]
        S["bull_put_credit"] = [Leg("P", otm_p, exp, -1), Leg("P", otm_p2, exp, +1)] if otm_p2 < otm_p else None
    S["straddle"] = [Leg("C", atm_c, exp, +1), Leg("P", atm_p, exp, +1)]
    if otm_c > atm_c and otm_p < atm_p:
        S["strangle"] = [Leg("C", otm_c, exp, +1), Leg("P", otm_p, exp, +1)]
    if lo_c is not None and hi_c is not None and lo_c < atm_c < hi_c:
        S["call_butterfly"] = [Leg("C", lo_c, exp, +1), Leg("C", atm_c, exp, -2), Leg("C", hi_c, exp, +1)]
    if lo_p is not None and hi_p is not None and lo_p < atm_p < hi_p:
        S["put_butterfly"] = [Leg("P", lo_p, exp, +1), Leg("P", atm_p, exp, -2), Leg("P", hi_p, exp, +1)]
    if lo_c is not None and hi_c is not None and lo_c < atm_c < hi_c:
        S["iron_butterfly"] = [Leg("C", atm_c, exp, -1), Leg("P", atm_p, exp, -1),
                               Leg("C", hi_c, exp, +1), Leg("P", lo_p, exp, +1)]
    if far is not None and far != exp:
        S["calendar_call"] = [Leg("C", atm_c, exp, -1), Leg("C", nearest(np.sort(
            chain_u[(chain_u["exp_dt"] == far) & (chain_u["right"] == "C")]["strike"].unique()), spot), far, +1)]
        far_c = chain_u[(chain_u["exp_dt"] == far) & (chain_u["right"] == "C")]
        if not far_c.empty and otm_c > atm_c:
            S["diagonal_call"] = [Leg("C", otm_c, exp, -1),
                                  Leg("C", nearest(np.sort(far_c["strike"].unique()), spot), far, +1)]
    out = {}
    for k, v in S.items():
        if not v:
            continue
        if any(leg.strike is None or not np.isfinite(leg.strike) for leg in v):
            continue
        out[k] = v
    return out


def leg_symbol(under: str, leg: Leg) -> str:
    return build_occ_symbol(under, leg.expiry, leg.right, leg.strike).upper()


def price_legs(store, day, under, legs):
    """Return (open_debit, close_value, min_liq, ok). open uses ask(long)/bid(short);
    close uses bid(long)/ask(short)."""
    open_debit = 0.0
    close_value = 0.0
    min_liq = np.inf
    for leg in legs:
        q = store.get_leg_quote(day, leg_symbol(under, leg))
        if q is None or not (np.isfinite(q.bid) and np.isfinite(q.ask)) or q.ask <= 0 or q.bid < 0 or q.ask < q.bid:
            return None, None, None, False
        n = abs(leg.qty)
        mid = 0.5 * (q.bid + q.ask)
        half = 0.5 * (q.ask - q.bid) * FILL_FRAC   # slippage paid per side
        buy_px = mid + half                          # price to buy (long open / short close)
        sell_px = mid - half                         # price to receive (long close / short open)
        if leg.qty > 0:
            open_debit += n * buy_px
            close_value += n * sell_px
        else:
            open_debit -= n * sell_px
            close_value -= n * buy_px
        oi = q.open_interest if np.isfinite(q.open_interest) else 0.0
        vol = q.volume if np.isfinite(q.volume) else 0.0
        min_liq = min(min_liq, oi + vol)
    return open_debit, close_value, (0.0 if not np.isfinite(min_liq) else min_liq), True


def evaluation_schedule(dates, signal_index, horizon=HORIZON):
    """Return the first executable session and post-entry outcome sessions."""
    if signal_index + horizon + 1 >= len(dates):
        return None
    entry_day = dates[signal_index + 1]
    exit_dates = dates[signal_index + 2 : signal_index + 2 + horizon]
    if len(exit_dates) != horizon:
        return None
    return entry_day, exit_dates


def main():
    store = HistoricalOptionQuoteStore(ROOT, use_oi=not HOT_ONLY)
    closes = UnderlyingCloseStore(ROOT, allow_web_fallback=False)
    dates = store.available_dates()
    date_idx = {d: i for i, d in enumerate(dates)}

    rows = []
    for i, asof in enumerate(dates):
        # Signals are formed after this session closes.  Keep strike/expiry
        # selection anchored to that completed EOD chain, but price the exact
        # legs on the next market session.  Same-session entry would use a fill
        # that was no longer available when the signal became actionable.
        schedule = evaluation_schedule(dates, i)
        if schedule is None:
            break
        entry_day, exit_dates = schedule
        try:
            dq = store.get_quotes_for_date(asof)
        except Exception:
            continue
        if dq.empty:
            continue
        chain = parse_chain(dq)
        if chain.empty:
            continue
        liq_by_under = chain.groupby("under")["liq"].sum().sort_values(ascending=False)
        universe = list(liq_by_under.head(UNIVERSE_PER_DAY).index)
        n_day = 0
        for under in universe:
            spot = closes.get_close_on_or_before(under, asof)
            chain_u = chain[chain["under"] == under]
            if spot is None or not np.isfinite(spot) or spot <= 0:
                # fallback: ATM proxy = median strike near where call/put mids cross
                spot = float(np.median(chain_u["strike"]))
            strategies = build_strategies(under, spot, chain_u, asof)
            for route, legs in strategies.items():
                od, cv0, liq0, ok = price_legs(store, entry_day, under, legs)
                if not ok or od is None:
                    continue
                entry = abs(od)
                if entry < 0.05:
                    continue
                entry_type = "DEBIT" if od > 0 else "CREDIT"
                # management thresholds on unrealized pnl (per 1x, in net price units)
                if entry_type == "DEBIT":
                    tp, sl = entry * 1.60, entry * 0.40
                else:
                    tp, sl = entry * 0.40, entry * 2.00
                exit_val = None
                trig = "time_exit"
                hold = 0
                for k, ed in enumerate(exit_dates, start=1):
                    _od, cv, _liq, ok2 = price_legs(store, ed, under, legs)
                    if not ok2:
                        continue
                    hold = k
                    # value to close position (proceeds). Compare to entry cost.
                    if entry_type == "DEBIT":
                        # position worth cv to unwind; tp/sl on that value
                        if cv >= tp:
                            exit_val, trig = cv, "take_profit"; break
                        if cv <= sl:
                            exit_val, trig = cv, "stop_loss"; break
                    else:
                        # credit: cost to buy back = -cv (cv is negative). buyback = -cv
                        buyback = -cv
                        if buyback <= tp:
                            exit_val, trig = cv, "take_profit"; break
                        if buyback >= sl:
                            exit_val, trig = cv, "stop_loss"; break
                    exit_val = cv
                if exit_val is None:
                    continue
                pnl = (exit_val - od) * 100.0 - COMMISSION_PER_LEG * len(legs)
                rows.append({
                    "asof": asof.isoformat(),
                    "signal_day": asof.isoformat(),
                    "entry_day": entry_day.isoformat(),
                    "exit_day": exit_dates[hold - 1].isoformat() if hold else "",
                    "ticker": under,
                    "strategy_route": route,
                    "n_legs": len(legs),
                    "entry_type": entry_type,
                    "spot": round(spot, 2),
                    "dte": (legs[0].expiry - entry_day).days,
                    "entry_net": round(od, 4),
                    "entry_cost": round(entry, 4),
                    "min_leg_liquidity": round(liq0, 1),
                    "exit_value": round(exit_val, 4),
                    "exit_trigger": trig,
                    "holding_sessions": hold,
                    "pnl_1x": round(pnl, 2),
                    "win": int(pnl > 0),
                })
                n_day += 1
        if i % 10 == 0:
            print(f"{asof} universe={len(universe)} trades={n_day} total={len(rows)}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nWrote {len(df)} trades across {df['asof'].nunique()} days -> {OUT}")
    if not df.empty:
        def pf(p):
            pos = p[p > 0].sum(); neg = -p[p < 0].sum()
            return pos / neg if neg > 0 else float("inf")
        g = df.groupby("strategy_route")["pnl_1x"]
        summ = pd.DataFrame({"n": g.size(), "win%": (g.apply(lambda s: (s > 0).mean()) * 100).round(0),
                             "avgPL": g.mean().round(2), "PF": g.apply(pf).round(2),
                             "total": g.sum().round(0)})
        print("\nPer-strategy (raw universe, no selection):")
        print(summ.sort_values("PF", ascending=False).to_string())


if __name__ == "__main__":
    main()
