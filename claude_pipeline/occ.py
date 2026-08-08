"""OCC option symbol parsing and construction.

hot-chains carries no ticker column, so the underlying must be recovered from the
symbol itself. Verified against chain-oi's explicit underlying_symbol: 100% agreement.
"""

from __future__ import annotations

import re

import pandas as pd

OCC_RE = re.compile(r"^(?P<root>[A-Z0-9.\-]{1,6}?)(?P<expiry>\d{6})(?P<kind>[CP])(?P<strike>\d{8})$")


def build_symbol(root: str, expiry: pd.Timestamp | str, kind: str, strike: float) -> str:
    expiry = pd.Timestamp(expiry)
    return f"{root}{expiry:%y%m%d}{kind.upper()[0]}{int(round(strike * 1000)):08d}"


def parse(symbols: pd.Series) -> pd.DataFrame:
    """Vectorised parse. Rows that do not match come back as NaN rather than raising."""
    extracted = symbols.str.extract(OCC_RE)
    return pd.DataFrame(
        {
            "root": extracted["root"],
            "expiry": pd.to_datetime(extracted["expiry"], format="%y%m%d", errors="coerce"),
            "kind": extracted["kind"],
            "strike": pd.to_numeric(extracted["strike"], errors="coerce") / 1000.0,
        },
        index=symbols.index,
    )
