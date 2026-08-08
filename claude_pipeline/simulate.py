"""Position construction and outcome simulation.

Exits settle at expiry from the underlying close, which is available for every
session. Marking from option quotes is only used for optional early take-profit,
because contract quotes disappear non-randomly and would otherwise censor losers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

CONTRACT_MULTIPLIER = 100
COMMISSION_PER_CONTRACT = 0.65


@dataclass(frozen=True)
class Leg:
    symbol: str
    quantity: int  # +1 long, -1 short
    strike: float
    kind: str  # "C" or "P"


@dataclass
class Structure:
    ticker: str
    family: str
    entry_session: str
    expiry: pd.Timestamp
    legs: tuple[Leg, ...]
    width: float
    meta: dict = field(default_factory=dict)

    @property
    def contracts(self) -> int:
        return len(self.legs)


def leg_price(quote: pd.Series, quantity: int, fill: float) -> float:
    """Price one leg. fill=0 trades at mid, fill=1 crosses the full spread."""
    bid, ask = float(quote["bid"]), float(quote["ask"])
    if not np.isfinite(bid) or not np.isfinite(ask) or ask <= 0 or ask < bid:
        return float("nan")
    mid = (bid + ask) / 2.0
    half = (ask - bid) / 2.0
    return mid + fill * half if quantity > 0 else mid - fill * half


def net_price(structure: Structure, quotes: pd.DataFrame, fill: float) -> float:
    """Net debit (>0) or credit (<0) to OPEN the structure."""
    total = 0.0
    for leg in structure.legs:
        if leg.symbol not in quotes.index:
            return float("nan")
        price = leg_price(quotes.loc[leg.symbol], leg.quantity, fill)
        if not np.isfinite(price):
            return float("nan")
        total += leg.quantity * price
    return total


def settlement_value(structure: Structure, underlying_close: float) -> float:
    """Value of the position at expiry, per share, signed the same way as net_price."""
    total = 0.0
    for leg in structure.legs:
        if leg.kind == "C":
            intrinsic = max(0.0, underlying_close - leg.strike)
        else:
            intrinsic = max(0.0, leg.strike - underlying_close)
        total += leg.quantity * intrinsic
    return total


def max_risk(structure: Structure, entry_net: float) -> float:
    """Worst-case loss per contract in dollars."""
    if structure.family in ("bull_put_credit", "bear_call_credit"):
        return (structure.width + entry_net) * CONTRACT_MULTIPLIER
    if structure.family in ("bull_call_debit", "bear_put_debit"):
        return entry_net * CONTRACT_MULTIPLIER
    if structure.family in ("long_call", "long_put"):
        return entry_net * CONTRACT_MULTIPLIER
    if structure.family == "short_put":
        strike = structure.legs[0].strike
        return (strike + entry_net) * CONTRACT_MULTIPLIER
    raise ValueError(f"unknown family {structure.family}")


def plausible_entry(structure: Structure, entry_net: float) -> bool:
    """Reject economically impossible prices.

    A credit spread quoted as a debit means one leg's quote is stale or inverted;
    those rows are pure artifact and would otherwise dominate any ranking.
    """
    if not np.isfinite(entry_net):
        return False
    if structure.family in ("bull_put_credit", "bear_call_credit"):
        return -structure.width < entry_net < 0
    if structure.family in ("bull_call_debit", "bear_put_debit"):
        return 0 < entry_net < structure.width
    if structure.family in ("long_call", "long_put"):
        return entry_net > 0
    if structure.family == "short_put":
        return entry_net < 0
    return False


def simulate(
    structures: list[Structure],
    store,
    closes: pd.DataFrame,
    sessions: list[str],
    fill: float = 1.0,
    take_profit: float | None = None,
) -> pd.DataFrame:
    """Walk every structure forward to expiry (or an early take-profit) and score it.

    ``closes`` is indexed by session with tickers as columns. Positions whose expiry
    falls outside the available data are returned with outcome='censored' and a null
    P&L - never silently dropped, because unresolved trades skew toward losers.
    """
    session_pos = {s: i for i, s in enumerate(sessions)}
    by_entry: dict[str, list[Structure]] = {}
    for structure in structures:
        by_entry.setdefault(structure.entry_session, []).append(structure)

    open_positions: list[dict] = []
    records: list[dict] = []

    for session in sessions:
        quotes = store.get(session)
        session_ts = pd.Timestamp(session)

        for position in list(open_positions):
            structure = position["structure"]
            expired = session_ts >= structure.expiry
            if expired:
                close = closes.at[session, structure.ticker] if structure.ticker in closes.columns else np.nan
                if not np.isfinite(close):
                    continue  # keep waiting for a session with a usable close
                exit_net = settlement_value(structure, float(close))
                position.update(exit_net=exit_net, exit_session=session, outcome="expiry")
                records.append(position)
                open_positions.remove(position)
                continue

            if take_profit is not None:
                mark = net_price(structure, quotes, fill=fill)
                if np.isfinite(mark):
                    entry_net = position["entry_net"]
                    captured = mark - entry_net
                    if captured >= abs(entry_net) * take_profit:
                        position.update(exit_net=mark, exit_session=session, outcome="take_profit")
                        records.append(position)
                        open_positions.remove(position)

        for structure in by_entry.get(session, []):
            entry_net = net_price(structure, quotes, fill=fill)
            if not plausible_entry(structure, entry_net):
                records.append({
                    "structure": structure, "entry_net": entry_net, "exit_net": np.nan,
                    "exit_session": None,
                    "outcome": "unquotable" if not np.isfinite(entry_net) else "implausible_quote",
                })
                continue
            open_positions.append({
                "structure": structure, "entry_net": entry_net,
                "exit_net": np.nan, "exit_session": None, "outcome": None,
            })

    for position in open_positions:
        position["outcome"] = "censored"
        records.append(position)

    return _to_frame(records, session_pos)


def _to_frame(records: list[dict], session_pos: dict[str, int]) -> pd.DataFrame:
    rows = []
    for record in records:
        structure: Structure = record["structure"]
        entry_net = record["entry_net"]
        exit_net = record["exit_net"]
        resolved = record["outcome"] in ("expiry", "take_profit")

        # Both prices are the net cost to OPEN, so profit is simply the change in that value.
        gross = (exit_net - entry_net) * CONTRACT_MULTIPLIER if resolved else np.nan
        fees = COMMISSION_PER_CONTRACT * structure.contracts * (2 if record["outcome"] != "expiry" else 1)
        risk = max_risk(structure, entry_net) if np.isfinite(entry_net) else np.nan

        rows.append({
            "ticker": structure.ticker,
            "family": structure.family,
            "entry_session": structure.entry_session,
            "exit_session": record["exit_session"],
            "expiry": structure.expiry,
            "dte": (structure.expiry - pd.Timestamp(structure.entry_session)).days,
            "held_sessions": (
                session_pos.get(record["exit_session"], np.nan) - session_pos[structure.entry_session]
                if record["exit_session"] else np.nan
            ),
            "width": structure.width,
            "entry_net": entry_net,
            "exit_net": exit_net,
            "outcome": record["outcome"],
            "resolved": resolved,
            "pnl": gross - fees if resolved else np.nan,
            "max_risk": risk,
            **structure.meta,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["return_on_risk"] = frame["pnl"] / frame["max_risk"].replace(0, np.nan)
    return frame
