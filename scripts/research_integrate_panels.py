"""Join every UW daily export into one point-in-time (asof, ticker) panel.

The five dated exports each answer a different question, and none of them subsumes another:

  stock-screener      per-ticker daily state: 52-week range, IV term structure, OI, market cap
  hot-chains          contract-level NBBO/IV pricing, sweep / floor / multi-leg tags
  chain-oi-changes    STRIKE-LEVEL open interest change - where positioning is actually built
  dp-eod-report       dark pool prints - off-exchange accumulation / distribution
  bot-eod-report      the full option tape: real greeks, NBBO side, running volume and OI

Coverage is NOT uniform, and that drives what can honestly be validated. Four sources span
the full seven months; bot-eod exists on ~63 sessions, all of them recent, because Unusual
Whales only serves roughly one month of downloadable history. Any feature derived from
bot-eod therefore cannot be trained before the 2026-05-01 split and validated after it, and
this script reports that explicitly rather than letting a silent left-join hide it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SOURCES = {
    "screener_hot": "uw_features.csv.gz",
    "chain_oi": "chain_oi_features.csv.gz",
    "dark_pool": "dp_features.csv.gz",
    "bot_tape": "bot_features.csv.gz",
}

# One representative non-key column per source, used to measure real join coverage.
PROBE = {
    "screener_hot": "range_pos_52w",
    "chain_oi": "coi_build_dir",
    "dark_pool": "dp_buy_ratio",
    "bot_tape": "bot_dir_ratio",
}


def load_panels(research_dir: Path) -> dict[str, pd.DataFrame]:
    panels: dict[str, pd.DataFrame] = {}
    for name, fname in SOURCES.items():
        path = research_dir / fname
        if not path.exists():
            print(f"  {name:<14} MISSING ({fname})")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["ticker"] = df["ticker"].astype(str).str.upper()
        panels[name] = df
        print(f"  {name:<14} {len(df):>9,} rows  {df.shape[1]:>3} cols  "
              f"{df['asof'].nunique():>3} days  {df['ticker'].nunique():>6,} tickers")
    return panels


def build(research_dir: Path) -> pd.DataFrame:
    print("loading source panels")
    panels = load_panels(research_dir)
    if "screener_hot" not in panels:
        raise SystemExit("screener_hot panel is required as the join spine")

    # The screener spans the full history and defines the tradeable universe, so it is the
    # spine. Everything else is a left join onto it.
    panel = panels["screener_hot"]
    for name in ("chain_oi", "dark_pool", "bot_tape"):
        if name not in panels:
            continue
        before = len(panel)
        panel = panel.merge(panels[name], on=["asof", "ticker"], how="left")
        assert len(panel) == before, f"{name} join changed row count ({before} -> {len(panel)})"
    return panel


def report_coverage(panel: pd.DataFrame) -> None:
    print("\ncoverage by month (share of spine rows with a value from each source)")
    mo = panel["asof"].str[:7]
    cols = {n: c for n, c in PROBE.items() if c in panel.columns}
    cov = panel.groupby(mo).apply(
        lambda g: pd.Series({n: 100.0 * g[c].notna().mean() for n, c in cols.items()})
    )
    cov["rows"] = panel.groupby(mo).size()
    print(cov.round(1).to_string())

    print("\nsessions per source")
    for n, c in cols.items():
        days = panel.loc[panel[c].notna(), "asof"].nunique()
        print(f"  {n:<14} {days:>3} sessions")

    have_all = panel[[c for c in cols.values()]].notna().all(axis=1)
    print(f"\nrows with ALL sources present: {int(have_all.sum()):,} "
          f"({100.0 * have_all.mean():.1f}%) across {panel.loc[have_all, 'asof'].nunique()} sessions")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-dir", default="/Users/anuppamvi/uw_root/tradedesk/out/research")
    ap.add_argument("--out", default="/Users/anuppamvi/uw_root/tradedesk/out/research/uw_panel_all5.csv.gz")
    args = ap.parse_args()

    research = Path(args.research_dir)
    panel = build(research)
    report_coverage(panel)

    out = Path(args.out)
    panel.to_csv(out, index=False, compression="gzip")
    print(f"\nwrote {len(panel):,} rows x {panel.shape[1]} cols -> {out}")


if __name__ == "__main__":
    main()
