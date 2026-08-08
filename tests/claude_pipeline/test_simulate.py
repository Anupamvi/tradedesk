"""Hand-computed P&L cases.

The sign convention is the single most dangerous thing in this module: an inverted
sign made long calls look like an 83%-win strategy. Every case here is arithmetic
a human can check.
"""

from __future__ import annotations

import pandas as pd
import pytest

from claude_pipeline.simulate import (
    CONTRACT_MULTIPLIER,
    COMMISSION_PER_CONTRACT,
    Leg,
    Structure,
    net_price,
    plausible_entry,
    settlement_value,
    simulate,
)


def _quotes(rows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"option_symbol": k, "bid": v[0], "ask": v[1]} for k, v in rows.items()]
    ).set_index("option_symbol")


def _spread(family: str, legs: tuple[Leg, ...], width: float) -> Structure:
    return Structure("ACME", family, "2026-06-01", pd.Timestamp("2026-06-19"), legs, width)


def test_long_call_pays_a_debit_and_profits_when_it_finishes_in_the_money():
    structure = _spread("long_call", (Leg("C100", +1, 100.0, "C"),), width=100.0)
    quotes = _quotes({"C100": (4.30, 4.50)})

    entry = net_price(structure, quotes, fill=1.0)
    assert entry == pytest.approx(4.50)  # crossing the spread pays the ask
    assert settlement_value(structure, 106.0) == pytest.approx(6.00)
    assert plausible_entry(structure, entry)


def test_credit_spread_receives_a_credit_and_keeps_it_when_it_expires_worthless():
    structure = _spread(
        "bull_put_credit",
        (Leg("P95", -1, 95.0, "P"), Leg("P90", +1, 90.0, "P")),
        width=5.0,
    )
    quotes = _quotes({"P95": (1.00, 1.10), "P90": (0.60, 0.70)})

    entry = net_price(structure, quotes, fill=1.0)
    # sell the short leg at its bid (1.00), buy the long leg at its ask (0.70)
    assert entry == pytest.approx(-0.30)
    assert plausible_entry(structure, entry)
    assert settlement_value(structure, 120.0) == pytest.approx(0.0)


def test_credit_spread_max_loss_is_width_minus_credit():
    structure = _spread(
        "bull_put_credit",
        (Leg("P95", -1, 95.0, "P"), Leg("P90", +1, 90.0, "P")),
        width=5.0,
    )
    assert settlement_value(structure, 80.0) == pytest.approx(-5.0)


@pytest.mark.parametrize(
    "family,entry_net,expected",
    [
        ("bull_put_credit", -0.30, True),
        ("bull_put_credit", 0.30, False),   # a credit spread cannot cost money
        ("bull_put_credit", -9.00, False),  # cannot collect more than the width
        ("bull_call_debit", 1.70, True),
        ("bull_call_debit", -1.70, False),
        ("long_call", 4.50, True),
        ("long_call", -4.50, False),
    ],
)
def test_implausible_entry_prices_are_rejected(family, entry_net, expected):
    legs = (Leg("A", -1, 95.0, "P"), Leg("B", +1, 90.0, "P"))
    assert plausible_entry(_spread(family, legs, width=5.0), entry_net) is expected


class _StaticStore:
    def __init__(self, frame):
        self._frame = frame

    def get(self, _session):
        return self._frame


def test_end_to_end_pnl_matches_hand_arithmetic():
    sessions = ["2026-06-01", "2026-06-19"]
    closes = pd.DataFrame({"ACME": [100.0, 120.0]}, index=sessions)
    structure = _spread(
        "bull_put_credit",
        (Leg("P95", -1, 95.0, "P"), Leg("P90", +1, 90.0, "P")),
        width=5.0,
    )
    store = _StaticStore(_quotes({"P95": (1.00, 1.10), "P90": (0.60, 0.70)}))

    result = simulate([structure], store, closes, sessions, fill=1.0)

    row = result.iloc[0]
    assert row["outcome"] == "expiry"
    expected = 0.30 * CONTRACT_MULTIPLIER - COMMISSION_PER_CONTRACT * 2
    assert row["pnl"] == pytest.approx(expected)
    assert row["max_risk"] == pytest.approx((5.0 - 0.30) * CONTRACT_MULTIPLIER)


def test_unresolved_positions_are_censored_not_dropped():
    sessions = ["2026-06-01", "2026-06-02"]
    closes = pd.DataFrame({"ACME": [100.0, 101.0]}, index=sessions)
    structure = _spread("long_call", (Leg("C100", +1, 100.0, "C"),), width=100.0)
    store = _StaticStore(_quotes({"C100": (4.30, 4.50)}))

    result = simulate([structure], store, closes, sessions, fill=1.0)

    assert result.iloc[0]["outcome"] == "censored"
    assert pd.isna(result.iloc[0]["pnl"])
