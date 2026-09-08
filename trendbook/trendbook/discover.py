"""Dynamic universe for this desk only: live 52-week highs, movers, own book memory.

Does not read configs/universe.txt. Does not import groat / xhigh / wheelo.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Set

from trendbook.bars import bars_through, closes, ret, rolling_extreme
from trendbook.config import (
    CODE_DIR,
    INDEX_TICKERS,
    MACRO_TICKERS,
    SECTOR_ETFS,
)
from trendbook.num import to_float
from trendbook.schwab import movers_symbols, quotes_many, read_bars_cache, schwab_cache_dir

PROBE_MAX = 40
PROBE_NEAR_HIGH = 0.98
PROBE_RS_126 = 0.25
PROBE_MIN_PX = 10.0
MEMORY_PATH = CODE_DIR / "var" / "universe.json"
BOOK_DIR = CODE_DIR / "var" / "book"


def infra_tickers() -> List[str]:
    names = list(INDEX_TICKERS) + list(MACRO_TICKERS) + list(SECTOR_ETFS)
    seen = set()
    out = []
    for name in names:
        key = str(name).upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _cache_names() -> List[str]:
    folder = schwab_cache_dir()
    if not folder.is_dir():
        return []
    return sorted(p.stem.upper() for p in folder.glob("*.json"))


def quote_is_52w_high(quote: dict) -> bool:
    last = to_float(quote.get("last")) or to_float(quote.get("close"))
    hi = to_float(quote.get("high52"))
    if last is None or last < PROBE_MIN_PX or hi is None or hi <= 0:
        return False
    return last >= PROBE_NEAR_HIGH * hi


def load_memory() -> List[str]:
    """Names this pipeline has already classified as Stage 2 / on-board."""
    if MEMORY_PATH.is_file():
        try:
            payload = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        raw = payload.get("names") if isinstance(payload, dict) else None
        if isinstance(raw, list) and raw:
            return [str(n).upper() for n in raw if n]
    return _bootstrap_from_book()


def _bootstrap_from_book() -> List[str]:
    if not BOOK_DIR.is_dir():
        return []
    out = []
    for path in sorted(BOOK_DIR.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            row = {}
        if not isinstance(row, dict):
            continue
        if row.get("on_board") or row.get("last_stage") == 2 or row.get("had_trend"):
            out.append(path.stem.upper())
    return out


def save_memory(names: Sequence[str], asof: str) -> None:
    keep = set(load_memory())
    for name in names:
        key = str(name).upper().strip()
        if key:
            keep.add(key)
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated": asof,
        "source": "trendbook",
        "names": sorted(keep),
    }
    MEMORY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _score_series(series: Sequence[dict], spy: Sequence[dict], asof: str) -> Optional[dict]:
    stock = bars_through(series, asof)
    bench = bars_through(spy, asof)
    if not stock:
        return None
    px = to_float(stock[-1].get("close"))
    if px is None or px < PROBE_MIN_PX:
        return None
    hi = rolling_extreme(stock, 252, "high", True)
    near = bool(hi and hi > 0 and px >= PROBE_NEAR_HIGH * hi)
    rs = None
    a = ret(closes(stock), 126)
    b = ret(closes(bench), 126) if bench else None
    if a is not None and b is not None:
        rs = a - b
    strong = rs is not None and rs >= PROBE_RS_126
    if not near and not strong:
        return None
    return {"rs_126": rs or 0.0, "near_high": near, "close": px}


def probe_from_cache(asof: str, skip: Set[str], already: Set[str], spy: Sequence[dict]) -> List[str]:
    scored = []
    for name in _cache_names():
        if name in skip or name in already or name == "SPY":
            continue
        series = read_bars_cache(name)
        if not series:
            continue
        row = _score_series(series, spy, asof)
        if not row:
            continue
        scored.append((row["rs_126"], 1 if row["near_high"] else 0, name))
    scored.sort(reverse=True)
    return [name for _, _, name in scored[:PROBE_MAX]]


def probe_live_highs(asof: str, skip: Set[str], already: Set[str]) -> List[str]:
    """Live Schwab quotes of *this desk's* tape + movers. No other pipelines."""
    candidates = []
    seen = set()
    for name in _cache_names() + movers_symbols() + load_memory():
        key = str(name).upper()
        if not key or key in seen or key in skip or key in already or key == "SPY":
            continue
        seen.add(key)
        candidates.append(key)
    if not candidates:
        return []
    quotes = quotes_many(candidates, asof)
    scored = []
    for name, quote in quotes.items():
        if name in skip or name in already or name == "SPY":
            continue
        if not quote_is_52w_high(quote):
            continue
        last = to_float(quote.get("last")) or to_float(quote.get("close")) or 0.0
        hi = to_float(quote.get("high52")) or 1.0
        scored.append((last / hi, name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:PROBE_MAX]]


def probe_tickers(
    asof: str,
    skip: Set[str],
    already: Set[str],
    spy: Sequence[dict],
    live: bool,
) -> List[str]:
    """New names to classify. Never auto-ADD. Live 52-week highs first, then cache RS."""
    out = []
    seen = set()
    if live:
        for name in probe_live_highs(asof, skip, already):
            if name in seen:
                continue
            out.append(name)
            seen.add(name)
        for name in movers_symbols():
            if name in skip or name in already or name in seen or name == "SPY":
                continue
            out.append(name)
            seen.add(name)
            if len(out) >= PROBE_MAX:
                break
    for name in probe_from_cache(asof, skip, already, spy):
        if name in seen or name in skip or name in already:
            continue
        out.append(name)
        seen.add(name)
        if len(out) >= PROBE_MAX:
            break
    return out[:PROBE_MAX]


def dynamic_universe(asof: str, live: bool = False, spy: Optional[Sequence[dict]] = None) -> List[str]:
    """Infra + this desk's Stage-2 memory + live 52-week highs / movers."""
    skip = set(INDEX_TICKERS) | set(MACRO_TICKERS) | set(SECTOR_ETFS)
    names = infra_tickers()
    seen = set(names)
    for name in load_memory():
        if name in seen:
            continue
        names.append(name)
        seen.add(name)
    already = set(names)
    extra = probe_tickers(asof, skip, already, spy or [], live)
    for name in extra:
        if name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names
