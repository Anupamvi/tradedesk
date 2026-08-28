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


def finite(value) -> bool:
    return to_float(value) is not None


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


def fmt_pct(value, digits=2, unavailable="DATA UNAVAILABLE") -> str:
    number = to_float(value)
    if number is None:
        return unavailable
    return ("%." + str(digits) + "f%%") % (number * 100.0)
