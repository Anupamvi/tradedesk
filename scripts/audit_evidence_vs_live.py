"""Evidence-vs-live consistency audit.

The route calibration that gates live Execute is computed from replay history.
If that history was generated under LOOSER rules than live execution enforces,
the calibration describes a population the live pipeline can never trade -
so its win rates / profit factors do not transfer.

Known divergences found in code review:
  * expected-move ratio: replay guard uses 0.65, credit_policy live uses 0.75
  * quote width:         replay entry allows 0.80, credit_policy live allows 0.35
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from codexuw.credit_policy import (
    MAX_QUOTE_WIDTH_PCT,
    MIN_DISTANCE_EXPECTED_MOVE_RATIO,
)

HISTORY = "codexuw/history/codexdaily_v4_edge_history_v2_2026-07-10.csv.gz"
REPLAY_EM_RATIO = 0.65
REPLAY_QUOTE_WIDTH = 0.80


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def pf_stats(df: pd.DataFrame) -> dict[str, object]:
    p = pd.to_numeric(df["pnl_1x"], errors="coerce").dropna()
    if p.empty:
        return {"n": 0, "pf": float("nan"), "win": float("nan"), "total": 0.0}
    w = p[p > 0].sum()
    l = abs(p[p < 0].sum())
    return {
        "n": len(p),
        "pf": round(w / l, 3) if l > 0 else float("inf"),
        "win": round((p > 0).mean(), 3),
        "total": round(p.sum(), 0),
    }


def main() -> None:
    h = pd.read_csv(HISTORY, low_memory=False)
    ev = h[truthy(h["exact_evaluated"])].copy()
    guarded = ev[truthy(ev["replay_guard_pass"])].copy()

    print(f"Guarded evidence base (feeds ALL live calibration): {len(guarded)} trades")
    print(f"  {pf_stats(guarded)}\n")

    qw = pd.to_numeric(guarded.get("entry_quote_width_pct"), errors="coerce")
    em = pd.to_numeric(guarded.get("expected_move_ratio"), errors="coerce")

    print("=== How much of the evidence would LIVE reject? ===")
    too_wide = qw > MAX_QUOTE_WIDTH_PCT
    print(f"  quote width > live max {MAX_QUOTE_WIDTH_PCT}: "
          f"{int(too_wide.sum())} / {len(guarded)} = {too_wide.mean():.1%}")
    below_em = em < MIN_DISTANCE_EXPECTED_MOVE_RATIO
    print(f"  expected-move ratio < live min {MIN_DISTANCE_EXPECTED_MOVE_RATIO}: "
          f"{int(below_em.sum())} / {len(guarded)} = {below_em.mean():.1%}")
    live_ok = (~too_wide.fillna(True)) & (~below_em.fillna(True))
    print(f"  BOTH live-consistent: {int(live_ok.sum())} / {len(guarded)} = {live_ok.mean():.1%}\n")

    print("=== Profitability: evidence the live pipeline could actually have traded ===")
    print(f"  live-INCONSISTENT (should not inform calibration): {pf_stats(guarded[~live_ok])}")
    print(f"  live-CONSISTENT   (the honest evidence base):      {pf_stats(guarded[live_ok])}\n")

    print("=== Per-strategy, live-consistent only ===")
    g = guarded[live_ok].copy()
    if not g.empty:
        g["_strat"] = g["strategy"].astype(str)
        for name, sub in g.groupby("_strat"):
            print(f"  {name:<28} {pf_stats(sub)}")


if __name__ == "__main__":
    main()
