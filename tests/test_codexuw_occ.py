from __future__ import annotations

import datetime as dt

from codexuw.occ import build_occ_symbol, parse_occ_symbol


def test_parse_and_build_occ_symbol_roundtrip() -> None:
    parsed = parse_occ_symbol("NVDA260515P00170000")
    assert parsed is not None
    assert parsed.root == "NVDA"
    assert parsed.expiry == dt.date(2026, 5, 15)
    assert parsed.right == "P"
    assert parsed.strike == 170.0
    assert build_occ_symbol(parsed.root, parsed.expiry, parsed.right, parsed.strike) == "NVDA260515P00170000"
