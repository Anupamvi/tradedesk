"""Map where genuine out-of-sample edge lives in the aligned V4 evidence base.

Read-only study. No pipeline code is modified by this script.

Method notes (all deliberate, to avoid the in-sample mirages recorded in
/memories/repo/options-agent.md):
  * Only `exact_evaluated` rows are scored -- these have real replay outcomes.
  * Profit factor is reported with a 10% fill stress on entry, matching
    payoff_calibration, so numbers are comparable to the live gates.
  * Every headline number is also reported on a strict time split
    (train = first 60% of sessions, test = last 40%) so that an edge that only
    exists in-sample is visible as such.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY = Path("codexuw/history/codexdaily_v4_edge_history_v3_2026-07-23.csv.gz")
STRESS = 0.10


def truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def load() -> pd.DataFrame:
    d = pd.read_csv(HISTORY, low_memory=False)
    d["asof"] = pd.to_datetime(d["asof"], errors="coerce")
    for col in (
        "pnl_1x",
        "entry_credit_pct_width",
        "entry_debit_pct_width",
        "expected_move_ratio",
        "entry_quote_width_pct",
        "entry_width",
        "distance_pct",
        "reward_risk",
        "dte",
        "iv_rank",
        "iv_hv_ratio",
        "combined_flow_bias",
        "flow_bias",
    ):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
    d["is_eval"] = truthy(d["exact_evaluated"])
    d["is_fill"] = truthy(d["exact_fillable"])
    d["is_guard"] = truthy(d["replay_guard_pass"])
    d["is_credit"] = d["direction"].isin(["Bull Put", "Bear Call"])
    return d


def stressed_pnl(frame: pd.DataFrame) -> pd.Series:
    """Apply a 10% adverse entry-price shock, matching payoff_calibration."""
    width = frame["entry_width"].fillna(0.0)
    credit = frame["entry_credit"] if "entry_credit" in frame else 0.0
    debit = frame["entry_debit"] if "entry_debit" in frame else 0.0
    shock = np.where(
        frame["is_credit"],
        pd.to_numeric(credit, errors="coerce").fillna(0.0) * STRESS,
        pd.to_numeric(debit, errors="coerce").fillna(0.0) * STRESS,
    )
    return frame["pnl_1x"] - np.abs(shock) * 100.0 * 0.0 - np.abs(shock)


def metrics(frame: pd.DataFrame, stress: bool = True) -> dict[str, float]:
    if frame.empty:
        return {"n": 0, "pf": float("nan"), "win": float("nan"), "total": 0.0, "avg": float("nan")}
    pnl = stressed_pnl(frame) if stress else frame["pnl_1x"]
    pnl = pnl.dropna()
    if pnl.empty:
        return {"n": 0, "pf": float("nan"), "win": float("nan"), "total": 0.0, "avg": float("nan")}
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    return {
        "n": int(len(pnl)),
        "pf": float(pf),
        "win": float((pnl > 0).mean()),
        "total": float(pnl.sum()),
        "avg": float(pnl.mean()),
    }


def show(title: str, rows: list[tuple[str, dict[str, float]]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'slice':<48}{'n':>7}{'PF':>8}{'win':>8}{'avg$':>9}{'total$':>11}")
    for label, m in rows:
        if m["n"] == 0:
            print(f"{label:<48}{0:>7}{'-':>8}{'-':>8}{'-':>9}{'-':>11}")
            continue
        print(
            f"{label:<48}{m['n']:>7}{m['pf']:>8.3f}{m['win']:>8.1%}"
            f"{m['avg']:>9.2f}{m['total']:>11.0f}"
        )


def split_frames(d: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
    days = sorted(d["asof"].dropna().unique())
    cut = days[int(len(days) * 0.60)]
    return cut, d[d["asof"] < cut], d[d["asof"] >= cut]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all")
    args = ap.parse_args()

    d = load()
    ev = d[d["is_eval"]].copy()
    cut, train, test = split_frames(ev)
    print(f"history rows={len(d)} evaluated={len(ev)} sessions={ev['asof'].nunique()}")
    print(f"time split cut={pd.Timestamp(cut).date()} train={len(train)} test={len(test)}")

    if args.section in {"all", "base"}:
        rows = [("ALL evaluated", metrics(ev))]
        for direction, grp in ev.groupby("direction"):
            rows.append((f"  {direction}", metrics(grp)))
        show("baseline universe (no selection)", rows)

        rows = []
        for (direction, regime), grp in ev.groupby(["direction", "regime"]):
            if len(grp) >= 30:
                rows.append((f"{direction} | {regime}", metrics(grp)))
        rows.sort(key=lambda r: -(r[1]["pf"] if math.isfinite(r[1]["pf"]) else 0))
        show("direction x regime (n>=30, in-sample)", rows)

    if args.section in {"all", "credit"}:
        cr = ev[ev["is_credit"] & ev["is_fill"]].copy()
        print(f"\ncredit fillable evaluated n={len(cr)}")
        cr["em_ok"] = cr["expected_move_ratio"] >= 0.75
        cr["band_ok"] = cr["entry_credit_pct_width"].between(0.25, 0.30)
        rows = [
            ("live AND-gate (band & em)", metrics(cr[cr["band_ok"] & cr["em_ok"]])),
            ("band only 0.25-0.30", metrics(cr[cr["band_ok"]])),
            ("em only >=0.75", metrics(cr[cr["em_ok"]])),
            ("neither", metrics(cr[~cr["band_ok"] & ~cr["em_ok"]])),
        ]
        show("credit: the AND-gate vs its parts", rows)

        # Single-criterion alternative: expected value proxy.
        # For a vertical credit spread, credit_pct is the max-profit fraction of
        # width and em_ratio proxies P(expire OTM). Combine them multiplicatively
        # instead of gating each independently.
        cr["ev_score"] = cr["entry_credit_pct_width"] * cr["expected_move_ratio"]
        rows = []
        for lo in (0.10, 0.12, 0.14, 0.16, 0.18, 0.20):
            rows.append((f"ev_score >= {lo:.2f}", metrics(cr[cr["ev_score"] >= lo])))
        show("credit: single multiplicative EV criterion (in-sample)", rows)

        rows = []
        for lo in (0.10, 0.12, 0.14, 0.16, 0.18):
            sel_tr = train[train["is_credit"] & train["is_fill"]]
            sel_te = test[test["is_credit"] & test["is_fill"]]
            s_tr = sel_tr["entry_credit_pct_width"] * sel_tr["expected_move_ratio"]
            s_te = sel_te["entry_credit_pct_width"] * sel_te["expected_move_ratio"]
            rows.append((f"TRAIN ev>= {lo:.2f}", metrics(sel_tr[s_tr >= lo])))
            rows.append((f"TEST  ev>= {lo:.2f}", metrics(sel_te[s_te >= lo])))
        show("credit EV criterion: strict time split", rows)

    if args.section in {"all", "range"}:
        rg = ev[(ev["regime"] == "range") & ev["is_fill"]]
        rows = [("range fillable ALL", metrics(rg))]
        for direction, grp in rg.groupby("direction"):
            rows.append((f"  {direction}", metrics(grp)))
        show("range days: is anything tradeable?", rows)

        rows = []
        for direction, grp in rg.groupby("direction"):
            if grp.empty:
                continue
            g = grp[grp["expected_move_ratio"] >= 0.75]
            rows.append((f"{direction} em>=0.75", metrics(g)))
            g2 = grp[(grp["expected_move_ratio"] >= 1.0)]
            rows.append((f"{direction} em>=1.00", metrics(g2)))
        show("range days with distance buffer", rows)

    if args.section in {"all", "guard"}:
        rows = [
            ("guard pass (current live)", metrics(ev[ev["is_guard"]])),
            ("fillable, guard fail", metrics(ev[ev["is_fill"] & ~ev["is_guard"]])),
        ]
        show("current guard effectiveness", rows)
        g = ev[ev["is_guard"]]
        print("\nguard-pass by month:")
        print(g.groupby(g["asof"].dt.to_period("M")).size().to_string())


if __name__ == "__main__":
    main()
