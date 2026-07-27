"""Guard-component ablation on the frozen edge history.

Question: the replay guard cuts ~92% of fillable trades (301 -> 23 for bear-put
debit). Which individual conditions actually create profit, and which merely
destroy sample?

If a condition destroys sample without adding profit factor, removing it grows
every route's evidence base WITHOUT lowering the profitability bar - which is
exactly what we need for routes to validate.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HISTORY = "codexuw/history/codexdaily_v4_edge_history_v2_2026-07-10.csv.gz"


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def pf_stats(df: pd.DataFrame) -> dict[str, object]:
    p = pd.to_numeric(df["pnl_1x"], errors="coerce").dropna()
    if p.empty:
        return {"n": 0, "pf": float("nan"), "win": float("nan"), "avg": float("nan"), "total": 0.0}
    wins = p[p > 0].sum()
    losses = abs(p[p < 0].sum())
    return {
        "n": len(p),
        "pf": round(wins / losses, 3) if losses > 0 else float("inf"),
        "win": round((p > 0).mean(), 3),
        "avg": round(p.mean(), 2),
        "total": round(p.sum(), 0),
    }


def main() -> None:
    h = pd.read_csv(HISTORY, low_memory=False)
    # universe = entry-fillable and actually evaluated (what the guard filters FROM)
    ev = h[truthy(h["exact_evaluated"])].copy()
    ev["_kind"] = np.where(
        ev["strategy"].astype(str).str.contains("Credit", case=False), "Credit", "Debit"
    )
    print("=== BASELINE: all exact-evaluated (pre-guard) ===")
    print(pf_stats(ev), "\n")
    for kind, sub in ev.groupby("_kind"):
        print(f"  {kind}: {pf_stats(sub)}")
    print()

    guarded = ev[truthy(ev["replay_guard_pass"])]
    print("=== CURRENT GUARD (all conditions) ===")
    print(pf_stats(guarded), f"| kept {len(guarded)/len(ev):.1%} of sample\n")

    # ---- decompose DEBIT guard conditions individually ----
    deb = ev[ev["_kind"].eq("Debit")].copy()
    align = pd.to_numeric(deb.get("combined_flow_bias"), errors="coerce")
    sign = np.where(deb["direction"].astype(str).isin(["Bull Call"]), 1.0, -1.0)
    deb["_align_ok"] = (align * sign) > 0
    deb["_debit_pct"] = pd.to_numeric(deb.get("entry_debit_pct_width"), errors="coerce")
    deb["_rr"] = pd.to_numeric(deb.get("reward_risk"), errors="coerce")
    deb["_em"] = pd.to_numeric(deb.get("expected_move_ratio"), errors="coerce")

    print("=== DEBIT: each guard condition ALONE (does it add PF?) ===")
    print(f"  {'condition':<38} {'n':>5} {'PF':>7} {'win':>6} {'avg':>9}")
    base = pf_stats(deb)
    print(f"  {'(no condition = all fillable debit)':<38} {base['n']:>5} {base['pf']:>7} {base['win']:>6} {base['avg']:>9}")
    conds = {
        "flow alignment > 0": deb["_align_ok"],
        "entry_debit_pct <= 0.45": deb["_debit_pct"] <= 0.45,
        "entry_debit_pct < 0.75": deb["_debit_pct"] < 0.75,
        "reward_risk >= 0.35": deb["_rr"] >= 0.35,
        "expected_move_ratio >= 0.65": deb["_em"] >= 0.65,
    }
    for name, mask in conds.items():
        s = pf_stats(deb[mask.fillna(False)])
        print(f"  {name:<38} {s['n']:>5} {s['pf']:>7} {s['win']:>6} {s['avg']:>9}")
    print()

    print("=== DEBIT: LEAVE-ONE-OUT (drop one condition, keep rest) ===")
    print("  If PF barely changes but n jumps -> that condition only costs sample.")
    print(f"  {'dropped condition':<38} {'n':>5} {'PF':>7} {'win':>6} {'avg':>9}")
    all_mask = pd.Series(True, index=deb.index)
    for m in conds.values():
        all_mask &= m.fillna(False)
    s_all = pf_stats(deb[all_mask])
    print(f"  {'(none - full guard)':<38} {s_all['n']:>5} {s_all['pf']:>7} {s_all['win']:>6} {s_all['avg']:>9}")
    for drop in conds:
        m = pd.Series(True, index=deb.index)
        for name, mask in conds.items():
            if name != drop:
                m &= mask.fillna(False)
        s = pf_stats(deb[m])
        print(f"  {drop:<38} {s['n']:>5} {s['pf']:>7} {s['win']:>6} {s['avg']:>9}")


if __name__ == "__main__":
    main()
