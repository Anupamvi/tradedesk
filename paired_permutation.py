"""Permutation null for the PAIRED Technology book.

The pairing was proposed after observing that the put leg loses in the months
the call leg wins. That is post-hoc, so the book has to be re-tested against a
null that keeps everything except the thing being claimed: same entry dates,
same sector, same number of names, same option type per leg -- only WHICH names
are chosen is randomized.

Reports p-values for TRAIN and TEST separately, on profit factor and on total
P&L, plus how often the null produces a book profitable in all six months.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import managed_exit_backtest as base  # noqa: E402
import symmetric_direction_test as sym  # noqa: E402

PERMS = int(sys.argv[1]) if len(sys.argv) > 1 else 150
COST_FLOOR = 700.0
OUT = base.ROOT / "out/paired_permutation.csv"


def pf(pnl):
    pnl = np.asarray(pnl, dtype=float)
    w = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    if l <= 0:
        return np.inf if w > 0 else np.nan
    return w / l


def book_stats(trades):
    trades = trades[trades.cost >= COST_FLOOR]
    if trades.empty:
        return {"n": 0, "pf": np.nan, "pnl": 0.0, "months_ok": 0}
    months = trades.assign(m=trades.signal_date.str[:7]).groupby("m").pnl.sum()
    return {
        "n": len(trades),
        "pf": pf(trades.pnl),
        "pnl": trades.pnl.sum(),
        "months_ok": int((months > 0).sum()),
        "months": len(months),
    }


def main():
    cols = ["date", "ticker", "sector", "issue_type", "marketcap", "close", "pos_52w"]
    panel = pd.read_csv(base.PANEL, usecols=cols, low_memory=False)
    panel["date"] = pd.to_datetime(panel["date"]).dt.strftime("%Y-%m-%d")
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
        & (panel.sector == "Technology")
    ].sort_values(["ticker", "date"])

    days = sorted(
        p.name for p in base.ROOT.iterdir()
        if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name)
    )
    position = {d: i for i, d in enumerate(days)}
    cache = {}

    def quote_for(session):
        if session not in cache:
            slot = position[session]
            cache[session] = (
                base.chain_quotes(session, days[slot + 1])
                if slot + 1 < len(days) else pd.DataFrame()
            )
        return cache[session]

    print("[paired-perm] warming quote cache", flush=True)
    for session in days:
        quote_for(session)

    rng = np.random.default_rng(20260728)

    def build(randomize):
        legs = [
            sym.simulate(panel, days, quote_for, d, rng, randomize=randomize)
            for d in ("long_call", "long_put")
        ]
        return pd.concat(legs, ignore_index=True)

    actual = build(False)
    obs = {}
    for half in ("TRAIN", "TEST"):
        sel = actual[actual.signal_date.ge(base.SPLIT) == (half == "TEST")]
        obs[half] = book_stats(sel)
        print(f"[paired-perm] actual {half}: n={obs[half]['n']} "
              f"pf={obs[half]['pf']:.2f} pnl=${obs[half]['pnl']:,.0f}", flush=True)
    obs["ALL"] = book_stats(actual)
    print(f"[paired-perm] actual ALL: pf={obs['ALL']['pf']:.2f} "
          f"months_ok={obs['ALL']['months_ok']}/{obs['ALL']['months']}", flush=True)

    null = []
    for trial in range(PERMS):
        trades = build(True)
        for half in ("TRAIN", "TEST"):
            sel = trades[trades.signal_date.ge(base.SPLIT) == (half == "TEST")]
            rec = book_stats(sel)
            rec.update(sample=half, trial=trial)
            null.append(rec)
        rec = book_stats(trades)
        rec.update(sample="ALL", trial=trial)
        null.append(rec)
        if (trial + 1) % 25 == 0:
            print(f"[paired-perm] {trial + 1}/{PERMS}", flush=True)

    nf = pd.DataFrame(null)
    nf.to_csv(OUT, index=False)

    print("\n" + "=" * 74)
    print(f"PAIRED TECHNOLOGY BOOK vs {PERMS} DATE/SECTOR/COUNT-MATCHED PERMUTATIONS")
    print("=" * 74)
    for half in ("TRAIN", "TEST", "ALL"):
        blk = nf[nf["sample"] == half]
        for metric in ("pf", "pnl"):
            vals = blk[metric].replace([np.inf, -np.inf], np.nan).dropna()
            a = obs[half][metric]
            p = (vals >= a).mean()
            flag = "  <-- significant" if p <= 0.05 else ""
            print(f"  {half:<5} {metric:<4} actual={a:>12,.2f}  "
                  f"null median={vals.median():>10,.2f}  p={p:.4f}{flag}")
    blk = nf[nf["sample"] == "ALL"]
    frac = (blk.months_ok >= obs["ALL"]["months_ok"]).mean()
    print(f"\n  null books profitable in all {obs['ALL']['months']} months: {frac:.1%}"
          f"   (actual achieved {obs['ALL']['months_ok']}/{obs['ALL']['months']})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
