"""Per-session option quotes, assembled from the two contract-level exports.

hot-chains carries closing NBBO but only for contracts that traded >=200 that day;
chain-oi-changes covers ~10x as many contracts with a last bid/ask. Both are used,
hot-chains first, because neither alone can mark a position for long.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from claude_pipeline import loaders, occ
from claude_pipeline.sources import SourceIndex, build_index

QUOTE_CACHE = Path("/Users/anuppamvi/tradedesk/out/claude_pipeline/quotes")

_HOT_COLUMNS = ["option_symbol", "bid", "ask", "iv", "volume", "open_interest", "close"]
_OI_COLUMNS = ["option_symbol", "last_bid", "last_ask", "curr_oi", "volume"]


def _build_session(index: SourceIndex, session: str) -> pd.DataFrame:
    frames = []

    if index.get(session, "hot-chains"):
        hot = loaders.read(index, session, "hot-chains", columns=_HOT_COLUMNS)
        frames.append(
            pd.DataFrame({
                "option_symbol": hot["option_symbol"],
                "bid": hot["bid"], "ask": hot["ask"], "iv": hot["iv"],
                "volume": hot["volume"], "open_interest": hot["open_interest"],
                "source": "hot-chains",
            })
        )

    if index.get(session, "chain-oi-changes"):
        oi = loaders.read(index, session, "chain-oi-changes", columns=_OI_COLUMNS)
        frames.append(
            pd.DataFrame({
                "option_symbol": oi["option_symbol"],
                "bid": oi["last_bid"], "ask": oi["last_ask"], "iv": np.nan,
                "volume": oi["volume"], "open_interest": oi["curr_oi"],
                "source": "chain-oi",
            })
        )

    if not frames:
        return pd.DataFrame(columns=["bid", "ask", "iv", "volume", "open_interest", "source"])

    quotes = pd.concat(frames, ignore_index=True)
    quotes = quotes.drop_duplicates(subset="option_symbol", keep="first")
    quotes["iv"] = pd.to_numeric(quotes["iv"], errors="coerce")
    parsed = occ.parse(quotes["option_symbol"])
    quotes = pd.concat([quotes, parsed], axis=1)
    return quotes.dropna(subset=["root"]).set_index("option_symbol")


class QuoteStore:
    """Lazy per-session quote access with an on-disk cache.

    ``sources`` restricts which exports may price a contract; limiting it to
    hot-chains gives true closing NBBO at the cost of far thinner coverage.
    """

    def __init__(self, index: SourceIndex | None = None, cache_dir: Path = QUOTE_CACHE,
                 sources: tuple[str, ...] | None = None):
        self._index = index or build_index()
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._sources = sources
        self._memory: dict[str, pd.DataFrame] = {}

    @property
    def index(self) -> SourceIndex:
        return self._index

    def sessions(self) -> list[str]:
        return [s for s in self._index.sessions() if self._index.get(s, "hot-chains")
                or self._index.get(s, "chain-oi-changes")]

    def get(self, session: str) -> pd.DataFrame:
        if session in self._memory:
            return self._memory[session]
        path = self._cache_dir / f"{session}.csv.gz"
        if path.exists():
            quotes = pd.read_csv(path, low_memory=False).set_index("option_symbol")
            quotes["expiry"] = pd.to_datetime(quotes["expiry"], errors="coerce")
        else:
            quotes = _build_session(self._index, session)
            quotes.reset_index().to_csv(path, index=False, compression="gzip")
        if self._sources is not None:
            quotes = quotes[quotes["source"].isin(self._sources)]
        self._memory[session] = quotes
        if len(self._memory) > 8:
            self._memory.pop(next(iter(self._memory)))
        return quotes

    def evict(self) -> None:
        self._memory.clear()
