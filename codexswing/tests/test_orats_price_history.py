import pytest

from codexswing.features.price import PriceHistoryError, parse_orats_price_history


def _row(day: int, close: float = 100.0):
    return {
        "ticker": "XYZ",
        "tradeDate": "2026-01-{:02d}".format(day),
        "open": close - 0.5,
        "hiPx": close + 1.0,
        "loPx": close - 1.0,
        "clsPx": close,
        "stockVolume": 1_000_000,
        "unadjClsPx": close * 4,
    }


def test_adjusted_orats_fields_are_canonical() -> None:
    result = parse_orats_price_history([_row(2), _row(3, 101)], ["XYZ"])
    assert result.source == "ORATS_HIST_DAILIES_ADJUSTED"
    assert result.observations["XYZ"][0].close == 100.0
    assert result.observations["XYZ"][0].open == 99.5


def test_duplicate_adjusted_bar_fails_closed() -> None:
    with pytest.raises(PriceHistoryError, match="duplicate"):
        parse_orats_price_history([_row(2), _row(2)], ["XYZ"])

