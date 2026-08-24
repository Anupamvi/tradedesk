"""Test the two structural hypotheses for why the V4 universe loses (PF 0.675).

H1 ADVERSE SELECTION: the pipeline *requires* flow alignment (credit >= 0.10,
    debit >= 0.20). Prior scratch work found flow alignment was ANTI-predictive.
    If true on the aligned base, the pipeline is systematically taking the wrong
    side -- and no amount of downstream gating can fix a sign error.

H2 EXIT DESTRUCTION: exits are 60% take-profit / 2x-credit stop. That caps wins
    and lets losers run to ~2x the average win. If the raw universe is near
    breakeven at expiry, the exit policy -- not the market -- is the leak.

Everything is reported on a strict time split so an in-sample mirage is visible.
Read-only; no pipeline code is modified.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

HISTORY = Path("codexuw/history/codexdaily_v4_edge_history_v3_2026-07-23.csv.gz")


def truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "1.0", "yes"})


def load() -> pd.DataFrame:
    d = pd.read_csv(HISTORY, low_memory=False)
    d["asof"] = pd.to_datetime(d["asof"], errors="coerce")
    num = [
        "pnl_1x", "entry_credit_pct_width", "entry_debit_pct_width", "expected_move_ratio",
        "entry_quote_width_pct", "entry_width", "distance_pct", "reward_risk", "dte",
        "iv_rank", "iv_hv_ratio", "combined_flow_bias", "flow_bias", "bot_flow_bias",
        "option_flow_bias", "dp_flow_bias", "entry_credit", "entry_debit", "return_on_risk",
        "bot_volume_oi_ratio", "bot_multileg_ratio", "source_contract_oi", "iv30d",
        "realized_volatility_30d", "stock_price_eod", "future_close", "breach_pct",
    ]
    for c in num:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["is_eval"] = truthy(d["exact_evaluated"])
    d["is_fill"] = truthy(d["exact_fillable"])
    d["is_credit"] = d["direction"].isin(["Bull Put", "Bear Call"])
    sign = d["direction"].map({"Bull Put": 1, "Bull Call": 1, "Bear Call": -1, "Bear Put": -1})
    bias = d["combined_flow_bias"].fillna(d["flow_bias"])
    d["flow_align"] = bias * sign
    return d


def pf(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if pnl.empty:
        return float("nan")
    g = pnl[pnl > 0].sum()
    l = -pnl[pnl < 0].sum()
    return g / l if l > 0 else float("inf")


def m(frame: pd.DataFrame, col: str = "pnl_1x") -> dict:
    p = frame[col].dropna()
    if p.empty:
        return {"n": 0, "pf": float("nan"), "win": float("nan"), "avg": float("nan"), "tot": 0.0}
    return {"n": len(p), "pf": pf(p), "win": float((p > 0).mean()), "avg": float(p.mean()), "tot": float(p.sum())}


def show(title: str, rows: list[tuple[str, dict]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'slice':<52}{'n':>7}{'PF':>8}{'win':>8}{'avg$':>9}{'total$':>11}")
    for label, r in rows:
        if r["n"] == 0:
            print(f"{label:<52}{0:>7}{'-':>8}{'-':>8}{'-':>9}{'-':>11}")
        else:
            print(f"{label:<52}{r['n']:>7}{r['pf']:>8.3f}{r['win']:>8.1%}{r['avg']:>9.2f}{r['tot']:>11.0f}")


def main() -> None:
    d = load()
    ev = d[d["is_eval"] & d["is_fill"]].copy()
    days = sorted(ev["asof"].dropna().unique())
    cut = days[int(len(days) * 0.60)]
    tr, te = ev[ev["asof"] < cut], ev[ev["asof"] >= cut]
    print(f"fillable evaluated n={len(ev)}  cut={pd.Timestamp(cut).date()}  train={len(tr)} test={len(te)}")

    # ---------------- H1: is flow alignment anti-predictive? ----------------
    print("\n" + "#" * 78)
    print("# H1  ADVERSE SELECTION: does the flow gate point the wrong way?")
    print("#" * 78)

    a = ev["flow_align"]
    print(f"\ncorr(flow_align, pnl_1x) = {ev['flow_align'].corr(ev['pnl_1x']):+.4f}   (n={a.notna().sum()})")
    for name, col in [("bot_flow_bias", "bot_flow_bias"), ("option_flow_bias", "option_flow_bias"),
                      ("dp_flow_bias", "dp_flow_bias")]:
        if col in ev:
            sign = ev["direction"].map({"Bull Put": 1, "Bull Call": 1, "Bear Call": -1, "Bear Put": -1})
            al = ev[col] * sign
            print(f"corr({name} aligned, pnl_1x)   = {al.corr(ev['pnl_1x']):+.4f}   (n={al.notna().sum()})")

    rows = []
    q = ev["flow_align"]
    valid = ev[q.notna()].copy()
    valid["fa_q"] = pd.qcut(valid["flow_align"], 5, labels=["Q1 most contra", "Q2", "Q3", "Q4", "Q5 most aligned"],
                            duplicates="drop")
    for lab, grp in valid.groupby("fa_q", observed=True):
        rows.append((str(lab), m(grp)))
    show("H1: P/L by flow-alignment quintile (in-sample)", rows)

    rows = []
    for lab in valid["fa_q"].cat.categories:
        rows.append((f"TRAIN {lab}", m(valid[(valid["fa_q"] == lab) & (valid["asof"] < cut)])))
        rows.append((f"TEST  {lab}", m(valid[(valid["fa_q"] == lab) & (valid["asof"] >= cut)])))
    show("H1: flow-alignment quintiles, strict time split", rows)

    rows = [
        ("passes live credit flow gate (>=0.10)", m(ev[ev["is_credit"] & (ev["flow_align"] >= 0.10)])),
        ("fails  live credit flow gate (<0.10)", m(ev[ev["is_credit"] & (ev["flow_align"] < 0.10)])),
        ("passes live debit  flow gate (>=0.20)", m(ev[~ev["is_credit"] & (ev["flow_align"] >= 0.20)])),
        ("fails  live debit  flow gate (<0.20)", m(ev[~ev["is_credit"] & (ev["flow_align"] < 0.20)])),
    ]
    show("H1: does the live flow gate select winners or losers?", rows)

    # ---------------- H2: how much does the exit policy cost? ----------------
    print("\n" + "#" * 78)
    print("# H2  EXIT DESTRUCTION: what does the take-profit/stop policy cost?")
    print("#" * 78)

    rows = []
    for reason, grp in ev.groupby("exit_reason"):
        rows.append((str(reason), m(grp)))
    rows.sort(key=lambda r: -r[1]["n"])
    show("H2: outcome by exit reason", rows)

    print("\nexit-reason mix (share of evaluated):")
    print((ev["exit_reason"].value_counts(normalize=True) * 100).round(1).to_string())

    # Reconstruct terminal (expiry-intrinsic) P&L to isolate the stop's effect.
    # future_close is the underlying at evaluation; short/long strikes are known.
    sh = pd.to_numeric(ev["short_strike_eod"], errors="coerce")
    lg = pd.to_numeric(ev["long_strike_eod"], errors="coerce")
    fc = ev["future_close"]
    width = (sh - lg).abs()
    is_put_side = ev["direction"].isin(["Bull Put", "Bear Put"])
    # Value of the spread at expiry, from the perspective of the position.
    short_itm = np.where(is_put_side, np.clip(sh - fc, 0, None), np.clip(fc - sh, 0, None))
    long_itm = np.where(is_put_side, np.clip(lg - fc, 0, None), np.clip(fc - lg, 0, None))
    spread_val = np.clip(short_itm - long_itm, 0, None)  # credit spread liability
    spread_val = np.minimum(spread_val, width.fillna(0))
    credit = ev["entry_credit"].fillna(0.0)
    debit = ev["entry_debit"].fillna(0.0)
    # Credit: keep credit minus liability. Debit: long vertical worth intrinsic.
    debit_val = np.clip(np.where(is_put_side, np.clip(lg - fc, 0, None), np.clip(fc - lg, 0, None)), 0, None)
    debit_val = np.minimum(debit_val, width.fillna(0))
    ev["terminal_pnl"] = np.where(ev["is_credit"], (credit - spread_val) * 100.0, (debit_val - debit) * 100.0)
    ok = ev["terminal_pnl"].notna() & width.notna() & fc.notna()
    sub = ev[ok]
    rows = [
        ("managed exits (current policy)", m(sub, "pnl_1x")),
        ("hold to terminal value", m(sub, "terminal_pnl")),
    ]
    show("H2: managed exits vs holding to terminal (same rows)", rows)
    for fam, grp in sub.groupby(sub["is_credit"].map({True: "CREDIT", False: "DEBIT"})):
        show(f"H2: {fam}", [("managed", m(grp, "pnl_1x")), ("terminal", m(grp, "terminal_pnl"))])

    # ---------------- entry cost: how much do we pay to get in? -------------
    print("\n" + "#" * 78)
    print("# ENTRY COST: how much of the width is burned by crossing the spread?")
    print("#" * 78)
    cr = ev[ev["is_credit"]]
    mid_c = pd.to_numeric(cr["entry_mid_credit"], errors="coerce")
    got_c = cr["entry_credit"]
    w = pd.to_numeric(cr["entry_width"], errors="coerce")
    slip_c = ((mid_c - got_c) / w).replace([np.inf, -np.inf], np.nan)
    db = ev[~ev["is_credit"]]
    mid_d = pd.to_numeric(db["entry_mid_debit"], errors="coerce")
    got_d = db["entry_debit"]
    wd = pd.to_numeric(db["entry_width"], errors="coerce")
    slip_d = ((got_d - mid_d) / wd).replace([np.inf, -np.inf], np.nan)
    print(f"credit entry slippage as % of width: median {slip_c.median():.4f}  mean {slip_c.mean():.4f}")
    print(f"debit  entry slippage as % of width: median {slip_d.median():.4f}  mean {slip_d.mean():.4f}")
    print(f"credit received / width: median {(got_c / w).median():.4f}")
    print(f"debit  paid     / width: median {(got_d / wd).median():.4f}")


if __name__ == "__main__":
    main()
