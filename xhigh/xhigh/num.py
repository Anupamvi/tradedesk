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


def fmt(value, digits=2, unavailable="DATA UNAVAILABLE") -> str:
    number = to_float(value)
    if number is None:
        return unavailable
    return ("%." + str(digits) + "f") % number
