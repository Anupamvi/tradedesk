from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass


COMPACT_OCC_RE = re.compile(r"^([A-Z.]{1,8})(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True)
class OccSymbol:
    root: str
    expiry: dt.date
    right: str
    strike: float

    @property
    def compact(self) -> str:
        return build_occ_symbol(self.root, self.expiry, self.right, self.strike)


def parse_occ_symbol(symbol: object) -> OccSymbol | None:
    text = str(symbol or "").strip().upper().replace(" ", "")
    match = COMPACT_OCC_RE.match(text)
    if not match:
        return None
    root, yymmdd, right, strike8 = match.groups()
    expiry = dt.date(2000 + int(yymmdd[:2]), int(yymmdd[2:4]), int(yymmdd[4:6]))
    return OccSymbol(root=root, expiry=expiry, right=right, strike=int(strike8) / 1000.0)


def build_occ_symbol(root: str, expiry: dt.date, right: str, strike: float) -> str:
    root_clean = str(root or "").strip().upper().replace(" ", "")
    right_clean = str(right or "").strip().upper()[0]
    strike_int = int(round(float(strike) * 1000))
    return f"{root_clean}{expiry:%y%m%d}{right_clean}{strike_int:08d}"
