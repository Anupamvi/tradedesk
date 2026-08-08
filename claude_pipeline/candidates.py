"""Construction of the raw candidate universe.

Deliberately unselective: this is the population a baseline is measured on, so the
only gates are sanity gates (a real two-sided quote, a real strike ladder).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from claude_pipeline.simulate import Leg, Structure

TARGET_DTE = 30
DTE_BAND = (21, 45)
OTM_TARGETS = (0.03, 0.06)
MAX_RELATIVE_WIDTH = 0.50
MIN_CONTRACTS_PER_ROOT = 12
MIN_OPEN_INTEREST = 50

CREDIT_FAMILIES = ("bull_put_credit", "bear_call_credit")
DEBIT_FAMILIES = ("bull_call_debit", "bear_put_debit")


def _usable(quotes: pd.DataFrame) -> pd.DataFrame:
    bid = pd.to_numeric(quotes["bid"], errors="coerce")
    ask = pd.to_numeric(quotes["ask"], errors="coerce")
    mid = (bid + ask) / 2.0
    relative = (ask - bid) / mid.replace(0, np.nan)
    keep = (bid > 0) & (ask > bid) & (relative <= MAX_RELATIVE_WIDTH) & (
        pd.to_numeric(quotes["open_interest"], errors="coerce").fillna(0) >= MIN_OPEN_INTEREST
    )
    return quotes[keep]


def _pick_expiry(chain: pd.DataFrame, session: pd.Timestamp) -> pd.Timestamp | None:
    dte = (chain["expiry"] - session).dt.days
    eligible = chain[(dte >= DTE_BAND[0]) & (dte <= DTE_BAND[1])]
    if eligible.empty:
        return None
    dte = (eligible["expiry"] - session).dt.days
    return eligible.loc[(dte - TARGET_DTE).abs().idxmin(), "expiry"]


def _ladder(chain: pd.DataFrame, expiry: pd.Timestamp, kind: str) -> pd.DataFrame:
    side = chain[(chain["expiry"] == expiry) & (chain["kind"] == kind)]
    return side.sort_values("strike").drop_duplicates("strike")


def _vertical(ticker, family, session, expiry, short_row, long_row, meta) -> Structure | None:
    width = abs(float(short_row["strike"]) - float(long_row["strike"]))
    if width <= 0:
        return None
    kind = short_row["kind"]
    return Structure(
        ticker=ticker, family=family, entry_session=session, expiry=expiry,
        legs=(
            Leg(short_row.name, -1, float(short_row["strike"]), kind),
            Leg(long_row.name, +1, float(long_row["strike"]), kind),
        ),
        width=width,
        meta={**meta, "short_strike": float(short_row["strike"]), "long_strike": float(long_row["strike"])},
    )


def build_for_session(
    session: str,
    quotes: pd.DataFrame,
    spots: pd.Series,
    roots: set[str] | None = None,
) -> list[Structure]:
    session_ts = pd.Timestamp(session)
    usable = _usable(quotes)
    if usable.empty:
        return []

    structures: list[Structure] = []
    for root, chain in usable.groupby("root"):
        if roots is not None and root not in roots:
            continue
        if len(chain) < MIN_CONTRACTS_PER_ROOT:
            continue
        spot = spots.get(root, np.nan)
        if not np.isfinite(spot) or spot <= 5:
            continue
        expiry = _pick_expiry(chain, session_ts)
        if expiry is None:
            continue

        calls = _ladder(chain, expiry, "C")
        puts = _ladder(chain, expiry, "P")
        if len(calls) < 3 or len(puts) < 3:
            continue
        meta = {"spot": float(spot), "expiry_dte": int((expiry - session_ts).days)}

        for otm in OTM_TARGETS:
            above = calls[calls["strike"] >= spot * (1 + otm)]
            if len(above) >= 2:
                structures.append(
                    _vertical(root, "bear_call_credit", session, expiry,
                              above.iloc[0], above.iloc[1], {**meta, "otm_target": otm})
                )
            below = puts[puts["strike"] <= spot * (1 - otm)]
            if len(below) >= 2:
                structures.append(
                    _vertical(root, "bull_put_credit", session, expiry,
                              below.iloc[-1], below.iloc[-2], {**meta, "otm_target": otm})
                )
                short_put = below.iloc[-1]
                structures.append(
                    Structure(root, "short_put", session, expiry,
                              (Leg(short_put.name, -1, float(short_put["strike"]), "P"),),
                              width=float(short_put["strike"]),
                              meta={**meta, "otm_target": otm, "short_strike": float(short_put["strike"])})
                )

        atm_calls = calls[calls["strike"] >= spot]
        if len(atm_calls) >= 2:
            structures.append(
                _vertical(root, "bull_call_debit", session, expiry,
                          atm_calls.iloc[1], atm_calls.iloc[0], {**meta, "otm_target": 0.0})
            )
            row = atm_calls.iloc[0]
            structures.append(
                Structure(root, "long_call", session, expiry,
                          (Leg(row.name, +1, float(row["strike"]), "C"),),
                          width=float(row["strike"]), meta={**meta, "otm_target": 0.0})
            )
        atm_puts = puts[puts["strike"] <= spot]
        if len(atm_puts) >= 2:
            structures.append(
                _vertical(root, "bear_put_debit", session, expiry,
                          atm_puts.iloc[-2], atm_puts.iloc[-1], {**meta, "otm_target": 0.0})
            )
            row = atm_puts.iloc[-1]
            structures.append(
                Structure(root, "long_put", session, expiry,
                          (Leg(row.name, +1, float(row["strike"]), "P"),),
                          width=float(row["strike"]), meta={**meta, "otm_target": 0.0})
            )

    return [s for s in structures if s is not None]
