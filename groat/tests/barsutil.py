from datetime import date, timedelta


def weekdays_ending(end: str, n: int):
    last = date.fromisoformat(end)
    days = []
    d = last
    while len(days) < n:
        if d.weekday() < 5:
            days.append(d)
        d -= timedelta(days=1)
    return list(reversed(days))


def trend_bars(
    n=220,
    end="2026-08-26",
    start_px=100.0,
    slope=0.25,
    pullback=0.0,
    volume=2_000_000.0,
    noise=0.0,
):
    days = weekdays_ending(end, n)
    bars = []
    px = start_px
    for i, day in enumerate(days):
        px = start_px + slope * i
        if pullback and i >= n - 4:
            px = px - pullback * (i - (n - 5))
        close = px
        high = close + 0.6 + noise
        low = close - 0.6 - noise
        open_ = close - 0.1
        vol = volume * (0.7 if pullback and i >= n - 4 else 1.0)
        bars.append(
            {
                "date": day.isoformat(),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol,
            }
        )
    return bars


def flat_bars(n=80, end="2026-08-26", px=100.0, volume=1_000_000.0):
    days = weekdays_ending(end, n)
    bars = []
    for i, day in enumerate(days):
        close = px + (0.2 if i % 2 == 0 else -0.2)
        bars.append(
            {
                "date": day.isoformat(),
                "open": px,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": volume,
            }
        )
    return bars
