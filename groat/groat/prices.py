"""Daily OHLC tape. Schwab history first, ORATS dailies fallback. Never invent prices."""

from typing import Dict, List, Optional

from groat.num import to_float
from groat.orats import (
    fetch_dailies_series,
    field_map_path,
    load_dailies_payload,
    write_dailies_payload,
    _read_json,
    rows_of,
)


def field_keys() -> Dict[str, str]:
    mapped = _read_json(field_map_path()) or {}
    return {
        "close": ((mapped.get("close") or {}).get("key") if isinstance(mapped.get("close"), dict) else None) or "clsPx",
        "high": ((mapped.get("high") or {}).get("key") if isinstance(mapped.get("high"), dict) else None) or "hiPx",
        "low": ((mapped.get("low") or {}).get("key") if isinstance(mapped.get("low"), dict) else None) or "loPx",
        "open": ((mapped.get("open") or {}).get("key") if isinstance(mapped.get("open"), dict) else None) or "open",
        "date": ((mapped.get("trade_date") or {}).get("key") if isinstance(mapped.get("trade_date"), dict) else None)
        or "tradeDate",
        "volume": ((mapped.get("volume") or {}).get("key") if isinstance(mapped.get("volume"), dict) else None)
        or "stkVolu",
    }


def rows_to_bars(rows: List[dict]) -> List[dict]:
    keys = field_keys()
    bars = []
    for row in rows or []:
        day = str(row.get(keys["date"]) or row.get("tradeDate") or "")[:10]
        close = to_float(row.get(keys["close"]))
        high = to_float(row.get(keys["high"]))
        low = to_float(row.get(keys["low"]))
        open_ = to_float(row.get(keys["open"]))
        vol = to_float(row.get(keys["volume"]))
        if vol is None:
            vol = to_float(row.get("volume"))
        if len(day) != 10 or close is None or high is None or low is None:
            continue
        bars.append({"date": day, "open": open_, "high": high, "low": low, "close": close, "volume": vol})
    bars.sort(key=lambda b: b["date"])
    return bars


def bars_to_payload(ticker: str, bars: List[dict]) -> dict:
    data = []
    for bar in bars:
        data.append(
            {
                "ticker": ticker,
                "tradeDate": bar["date"],
                "clsPx": bar["close"],
                "hiPx": bar["high"],
                "loPx": bar["low"],
                "open": bar.get("open"),
                "stkVolu": bar.get("volume"),
            }
        )
    return {"data": data}


def load_cached_bars(ticker: str) -> List[dict]:
    payload = load_dailies_payload(ticker)
    if payload is None:
        return []
    return rows_to_bars(rows_of(payload))


def merge_bars(base: List[dict], extra: List[dict]) -> List[dict]:
    by_date = {b["date"]: dict(b) for b in base}
    for bar in extra:
        prev = by_date.get(bar["date"]) or {}
        merged = dict(prev)
        merged.update({k: v for k, v in bar.items() if v is not None})
        if merged.get("volume") is None and prev.get("volume") is not None:
            merged["volume"] = prev["volume"]
        by_date[bar["date"]] = merged
    return [by_date[d] for d in sorted(by_date)]


def ensure_bars(
    ticker: str,
    token: str,
    getter=None,
    max_requests: Optional[int] = None,
    schwab_bars: Optional[List[dict]] = None,
    asof: Optional[str] = None,
    live: bool = False,
    refresh: bool = False,
) -> Dict[str, object]:
    name = str(ticker).upper()
    cached = load_cached_bars(name)
    last = cached[-1]["date"] if cached else ""
    stale = bool(asof and ((not last) or last < asof))
    extra = list(schwab_bars or [])
    if not extra:
        from groat.schwab import price_history_bars, quote_bar

        try:
            extra = price_history_bars(name, asof or last or "", use_cache=not refresh)
            if live and asof:
                qb = quote_bar(name, asof)
                if qb:
                    extra = merge_bars(extra, [qb])
        except Exception:
            extra = extra or []
    if extra:
        merged = merge_bars(cached, extra)
        write_dailies_payload(name, bars_to_payload(name, merged))
        last2 = merged[-1]["date"] if merged else ""
        if not asof or last2 >= asof:
            return {"bars": merged, "tape": "schwab_history", "http": 0, "error": ""}
        cached = merged
        last = last2
        stale = bool(asof and last < asof)
    if cached and not stale:
        return {"bars": cached, "tape": "orats_cache", "http": 0, "error": ""}
    if not token:
        return {
            "bars": cached,
            "tape": "orats_cache" if cached else "missing",
            "http": 0,
            "error": "" if cached and not stale else "missing_bars",
        }
    pack = fetch_dailies_series(name, token, getter=getter, max_requests=max_requests, refresh=stale)
    bars = rows_to_bars(pack.get("rows") or []) or cached
    tape = "orats_cache" if pack.get("cache") else "orats_fetch"
    if extra:
        bars = merge_bars(bars, extra)
        write_dailies_payload(name, bars_to_payload(name, bars))
        tape = "schwab_history"
    if not bars:
        return {
            "bars": [],
            "tape": tape,
            "http": pack.get("http") or 0,
            "error": pack.get("error") or "missing_bars",
        }
    return {"bars": bars, "tape": tape, "http": pack.get("http") or 0, "error": ""}
