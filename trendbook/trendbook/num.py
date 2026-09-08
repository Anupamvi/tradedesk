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


def pct_change(new, old) -> Optional[float]:
    a = to_float(new)
    b = to_float(old)
    if a is None or b is None or b == 0:
        return None
    return a / b - 1.0


def fmt(value, digits=2, unavailable="DATA UNAVAILABLE") -> str:
    number = to_float(value)
    if number is None:
        return unavailable
    return ("%." + str(digits) + "f") % number


def fmt_pct(value, digits=1, unavailable="DATA UNAVAILABLE") -> str:
    number = to_float(value)
    if number is None:
        return unavailable
    return ("%." + str(digits) + "f%%") % (number * 100.0)
