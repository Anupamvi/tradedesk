from typing import Optional


def to_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def to_int(value) -> Optional[int]:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def iv_decimal(value) -> Optional[float]:
    """ORATS iv30d is usually percent (32.5). Schwab IV is often decimal."""
    number = to_float(value)
    if number is None:
        return None
    if abs(number) > 1.5:
        return number / 100.0
    return number


def fmt(value, digits=2, unavailable="DATA UNAVAILABLE") -> str:
    number = to_float(value)
    if number is None:
        return unavailable
    return ("%." + str(digits) + "f") % number
