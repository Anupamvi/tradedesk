#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


OPTION_RE = re.compile(
    r"^\s*([A-Za-z0-9\.\-]+)\s+(\d{1,2})/(\d{1,2})/(\d{4})\s+([0-9]+(?:\.[0-9]+)?)\s+([CP])\s*$"
)

OPEN_ACTIONS = {"buy to open", "sell to open"}
CLOSE_ACTIONS = {"buy to close", "sell to close"}


def norm_action(x: object) -> str:
    return re.sub(r"\s+", " ", str(x or "").strip().lower())


def parse_money(x: object) -> float:
    if x is None:
        return math.nan
    s = str(x).strip()
    if not s:
        return 0.0
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace("$", "").replace(",", "")
    if s.startswith("-"):
        neg = True
    try:
        v = abs(float(s))
        return -v if neg else v
    except Exception:
        return math.nan


def parse_qty(x: object) -> float:
    try:
        if x is None:
            return math.nan
        s = str(x).replace(",", "").strip()
        if not s:
            return math.nan
        return abs(float(s))
    except Exception:
        return math.nan


def parse_option_symbol(symbol: object) -> Optional[Tuple[str, dt.date, str, float]]:
    m = OPTION_RE.match(str(symbol or "").strip().upper())
    if not m:
        return None
    t, mm, dd, yyyy, strike, right = m.groups()
    expiry = dt.date(int(yyyy), int(mm), int(dd))
    return t.upper(), expiry, right.upper(), float(strike)


def build_occ_symbol(ticker: str, expiry: dt.date, right: str, strike: float) -> str:
    strike_i = int(round(float(strike) * 1000))
    return f"{ticker.upper()}{expiry.strftime('%y%m%d')}{right.upper()}{strike_i:08d}"


def strategy_from_legs(right: str, short_strike: float, long_strike: float) -> Optional[str]:
    if right == "P":
        if short_strike > long_strike:
            return "Bull Put Credit"
        if short_strike < long_strike:
            return "Bear Put Debit"
        return None
    if right == "C":
        if short_strike > long_strike:
            return "Bull Call Debit"
        if short_strike < long_strike:
            return "Bear Call Credit"
        return None
    return None


def net_type_from_strategy(strategy: str) -> str:
    return "credit" if strategy in {"Bull Put Credit", "Bear Call Credit"} else "debit"


@dataclass
class Txn:
    idx: int
    date: dt.date
    action: str
    ticker: str
    expiry: dt.date
    right: str
    strike: float
    occ_symbol: str
    qty: float
    price: float
    amount: float

    @property
    def amt_per_contract(self) -> float:
        if not np.isfinite(self.qty) or self.qty <= 0:
            return math.nan
        return float(self.amount / self.qty)


@dataclass
class SpreadOpen:
    spread_id: str
    open_date: dt.date
    ticker: str
    expiry: dt.date
    right: str
    strategy: str
    net_type: str
    qty: float
    short_leg: str
    long_leg: str
    short_strike: float
    long_strike: float
    entry_cash_per_contract: float


@dataclass
class CloseFill:
    date: dt.date
    qty_rem: float
    amt_per_contract: float


def load_option_transactions(path: Path) -> List[Txn]:
    df = pd.read_csv(path, low_memory=False)
    date_col = "Date"
    action_col = "Action"
    symbol_col = "Symbol"
    qty_col = "Quantity"
    price_col = "Price"
    amount_col = "Amount"
    if not all(c in df.columns for c in [date_col, action_col, symbol_col, qty_col, amount_col]):
        raise ValueError("Input CSV missing required Schwab columns.")

    rows: List[Txn] = []
    for i, r in df.iterrows():
        action = norm_action(r[action_col])
        if action not in OPEN_ACTIONS and action not in CLOSE_ACTIONS:
            continue
        parsed = parse_option_symbol(r[symbol_col])
        if parsed is None:
            continue
        ticker, expiry, right, strike = parsed
        qty = parse_qty(r[qty_col])
        amount = parse_money(r[amount_col])
        price = parse_money(r.get(price_col))
        d = pd.to_datetime(r[date_col], errors="coerce")
        if pd.isna(d) or not np.isfinite(qty) or qty <= 0 or not np.isfinite(amount):
            continue
        occ = build_occ_symbol(ticker, expiry, right, strike)
        rows.append(
            Txn(
                idx=int(i),
                date=d.date(),
                action=action,
                ticker=ticker,
                expiry=expiry,
                right=right,
                strike=float(strike),
                occ_symbol=occ,
                qty=float(qty),
                price=float(price) if np.isfinite(price) else math.nan,
                amount=float(amount),
            )
        )
    rows.sort(key=lambda x: (x.date, x.idx))
    return rows


def pair_open_spreads(txns: List[Txn]) -> List[SpreadOpen]:
    opens = [t for t in txns if t.action in OPEN_ACTIONS]
    groups: Dict[Tuple[dt.date, str, dt.date, str], List[Txn]] = {}
    for t in opens:
        key = (t.date, t.ticker, t.expiry, t.right)
        groups.setdefault(key, []).append(t)

    out: List[SpreadOpen] = []
    spread_seq = 1
    for (d, ticker, expiry, right), items in sorted(groups.items(), key=lambda x: x[0]):
        shorts = []
        longs = []
        for t in items:
            if t.action == "sell to open":
                shorts.append({"txn": t, "qty_rem": float(t.qty)})
            elif t.action == "buy to open":
                longs.append({"txn": t, "qty_rem": float(t.qty)})
        if not shorts or not longs:
            continue

        while True:
            candidates: List[Tuple[float, int, int]] = []
            for si, s in enumerate(shorts):
                if s["qty_rem"] <= 0:
                    continue
                for li, l in enumerate(longs):
                    if l["qty_rem"] <= 0:
                        continue
                    if s["txn"].occ_symbol == l["txn"].occ_symbol:
                        continue
                    dist = abs(float(s["txn"].strike) - float(l["txn"].strike))
                    if dist <= 0:
                        continue
                    candidates.append((dist, si, li))
            if not candidates:
                break
            _, si, li = sorted(candidates, key=lambda x: x[0])[0]
            s = shorts[si]
            l = longs[li]
            q = min(float(s["qty_rem"]), float(l["qty_rem"]))
            if q <= 0:
                break
            short_txn = s["txn"]
            long_txn = l["txn"]
            strategy = strategy_from_legs(right, short_txn.strike, long_txn.strike)
            if strategy is None:
                s["qty_rem"] -= q
                l["qty_rem"] -= q
                continue
            net_type = net_type_from_strategy(strategy)
            # Signed cashflow per contract including fees.
            entry_cash_per_contract = short_txn.amt_per_contract + long_txn.amt_per_contract
            out.append(
                SpreadOpen(
                    spread_id=f"S{spread_seq:06d}",
                    open_date=d,
                    ticker=ticker,
                    expiry=expiry,
                    right=right,
                    strategy=strategy,
                    net_type=net_type,
                    qty=float(q),
                    short_leg=short_txn.occ_symbol,
                    long_leg=long_txn.occ_symbol,
                    short_strike=float(short_txn.strike),
                    long_strike=float(long_txn.strike),
                    entry_cash_per_contract=float(entry_cash_per_contract),
                )
            )
            spread_seq += 1
            s["qty_rem"] -= q
            l["qty_rem"] -= q
    return out


def build_close_pools(txns: List[Txn]) -> Dict[Tuple[str, str], List[CloseFill]]:
    pools: Dict[Tuple[str, str], List[CloseFill]] = {}
    for t in txns:
        if t.action not in CLOSE_ACTIONS:
            continue
        pools.setdefault((t.occ_symbol, t.action), []).append(
            CloseFill(
                date=t.date,
                qty_rem=float(t.qty),
                amt_per_contract=float(t.amt_per_contract),
            )
        )
    for key in pools:
        pools[key].sort(key=lambda x: x.date)
    return pools


def _next_fill(fills: List[CloseFill], min_date: dt.date) -> Optional[CloseFill]:
    for f in fills:
        if f.qty_rem > 1e-9 and f.date >= min_date:
            return f
    return None


def close_spreads(opens: List[SpreadOpen], close_pools: Dict[Tuple[str, str], List[CloseFill]]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for sp in sorted(opens, key=lambda x: (x.open_date, x.spread_id)):
        rem = float(sp.qty)
        short_fills = close_pools.get((sp.short_leg, "buy to close"), [])
        long_fills = close_pools.get((sp.long_leg, "sell to close"), [])
        while rem > 1e-9:
            sf = _next_fill(short_fills, sp.open_date)
            lf = _next_fill(long_fills, sp.open_date)
            if sf is None or lf is None:
                break
            q = min(rem, sf.qty_rem, lf.qty_rem)
            if q <= 1e-9:
                break
            short_close_amt = sf.amt_per_contract * q
            long_close_amt = lf.amt_per_contract * q
            exit_cash = short_close_amt + long_close_amt
            entry_cash = sp.entry_cash_per_contract * q
            realized = entry_cash + exit_cash

            entry_net = abs(sp.entry_cash_per_contract) / 100.0
            if sp.net_type == "credit":
                exit_net = (-exit_cash) / (q * 100.0)
            else:
                exit_net = exit_cash / (q * 100.0)

            close_date = max(sf.date, lf.date)
            rows.append(
                {
                    "trade_id": f"{sp.spread_id}_Q{int(round(q))}",
                    "signal_date": sp.open_date.isoformat(),
                    "ticker": sp.ticker,
                    "strategy": sp.strategy,
                    "expiry": sp.expiry.isoformat(),
                    "short_leg": sp.short_leg,
                    "long_leg": sp.long_leg,
                    "short_strike": sp.short_strike,
                    "long_strike": sp.long_strike,
                    "net_type": sp.net_type,
                    "qty": q,
                    "entry_net": entry_net,
                    "entry_gate": "",
                    "exit_date": close_date.isoformat(),
                    "exit_net": float(exit_net),
                    "realized_pnl": float(realized),
                    "close_status": "closed",
                }
            )
            rem -= q
            sf.qty_rem -= q
            lf.qty_rem -= q

        if rem > 1e-9:
            rows.append(
                {
                    "trade_id": f"{sp.spread_id}_OPEN",
                    "signal_date": sp.open_date.isoformat(),
                    "ticker": sp.ticker,
                    "strategy": sp.strategy,
                    "expiry": sp.expiry.isoformat(),
                    "short_leg": sp.short_leg,
                    "long_leg": sp.long_leg,
                    "short_strike": sp.short_strike,
                    "long_strike": sp.long_strike,
                    "net_type": sp.net_type,
                    "qty": rem,
                    "entry_net": abs(sp.entry_cash_per_contract) / 100.0,
                    "entry_gate": "",
                    "exit_date": "",
                    "exit_net": np.nan,
                    "realized_pnl": np.nan,
                    "close_status": "open_unmatched",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["signal_date", "trade_id"]).reset_index(drop=True)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Extract exact vertical spread trades from Schwab transactions export.")
    ap.add_argument("--input-csv", required=True, help="Schwab transactions CSV path.")
    ap.add_argument("--out-csv", required=True, help="Output spread trades CSV path.")
    ap.add_argument("--closed-only", action="store_true", help="Keep only closed spread rows.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    inp = Path(args.input_csv).expanduser().resolve()
    out = Path(args.out_csv).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    txns = load_option_transactions(inp)
    opens = pair_open_spreads(txns)
    close_pools = build_close_pools(txns)
    spreads = close_spreads(opens, close_pools)
    if args.closed_only:
        spreads = spreads[spreads["close_status"] == "closed"].copy()

    spreads.to_csv(out, index=False)
    print(f"Input transactions: {len(txns):,}")
    print(f"Open spread records: {len(opens):,}")
    print(f"Output spread rows: {len(spreads):,}")
    if not spreads.empty and "close_status" in spreads.columns:
        print("Status counts:", spreads["close_status"].value_counts(dropna=False).to_dict())
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
