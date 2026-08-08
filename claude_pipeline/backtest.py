"""Backtest runner: build the raw candidate universe and score every trade."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from claude_pipeline import candidates, panel as panel_mod, simulate
from claude_pipeline.quotes import QuoteStore

OUT_ROOT = Path("/Users/anuppamvi/tradedesk/out/claude_pipeline")


def tradable_roots(panel: pd.DataFrame) -> set[str]:
    """Exclude cash-settled index products; their settlement is not a stock close."""
    latest = panel.sort_values("session").groupby("ticker").last()
    keep = latest[(latest["is_index"] != "t") & (latest["issue_type"] != "Index")]
    return set(keep.index)


def run(start: str | None = None, end: str | None = None, fill: float = 1.0,
        take_profit: float | None = None, out_name: str = "backtest",
        sources: tuple[str, ...] | None = None) -> pd.DataFrame:
    panel = panel_mod.build()
    store = QuoteStore(sources=sources)
    sessions = [s for s in store.sessions() if (not start or s >= start) and (not end or s <= end)]
    closes = panel.pivot_table(index="session", columns="ticker", values="close", aggfunc="last")
    roots = tradable_roots(panel)

    structures = []
    started = time.time()
    for i, session in enumerate(sessions, 1):
        quotes = store.get(session)
        if quotes.empty:
            continue
        spots = closes.loc[session] if session in closes.index else pd.Series(dtype=float)
        structures.extend(candidates.build_for_session(session, quotes, spots, roots))
        if i % 25 == 0:
            print(f"  built {i}/{len(sessions)} sessions, {len(structures):,} structures "
                  f"({time.time() - started:.0f}s)", flush=True)

    print(f"candidate universe: {len(structures):,} structures over {len(sessions)} sessions")
    store.evict()

    results = simulate.simulate(structures, store, closes, sessions, fill=fill, take_profit=take_profit)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUT_ROOT / f"{out_name}.csv.gz"
    results.to_csv(path, index=False, compression="gzip")
    print(f"wrote {path} ({len(results):,} rows, {time.time() - started:.0f}s total)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--fill", type=float, default=1.0)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--out-name", default="backtest")
    parser.add_argument("--hot-chains-only", action="store_true",
                        help="price only from closing NBBO, ignoring the wider stale-prone export")
    args = parser.parse_args()
    run(args.start, args.end, args.fill, args.take_profit, args.out_name,
        sources=("hot-chains",) if args.hot_chains_only else None)


if __name__ == "__main__":
    main()
