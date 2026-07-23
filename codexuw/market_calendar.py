from __future__ import annotations

import datetime as dt


def _observed_fixed_holiday(year: int, month: int, day: int) -> dt.date:
    holiday = dt.date(year, month, day)
    if holiday.weekday() == 5:
        return holiday - dt.timedelta(days=1)
    if holiday.weekday() == 6:
        return holiday + dt.timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    first = dt.date(year, month, 1)
    return first + dt.timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    next_month = dt.date(year + 1, 1, 1) if month == 12 else dt.date(year, month + 1, 1)
    last = next_month - dt.timedelta(days=1)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


def _easter_date(year: int) -> dt.date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def us_equity_market_holidays(year: int) -> set[dt.date]:
    holidays: set[dt.date] = set()
    for scoped_year in (year - 1, year, year + 1):
        holidays.update(
            {
                _observed_fixed_holiday(scoped_year, 1, 1),
                _nth_weekday(scoped_year, 1, 0, 3),
                _nth_weekday(scoped_year, 2, 0, 3),
                _easter_date(scoped_year) - dt.timedelta(days=2),
                _last_weekday(scoped_year, 5, 0),
                _observed_fixed_holiday(scoped_year, 6, 19),
                _observed_fixed_holiday(scoped_year, 7, 4),
                _nth_weekday(scoped_year, 9, 0, 1),
                _nth_weekday(scoped_year, 11, 3, 4),
                _observed_fixed_holiday(scoped_year, 12, 25),
            }
        )
    return holidays


def is_regular_market_day(day: dt.date) -> bool:
    return day.weekday() < 5 and day not in us_equity_market_holidays(day.year)
