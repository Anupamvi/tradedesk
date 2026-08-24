"""Project the policy-aligned replay guard onto the existing frozen edge history.

The aligned guard is strictly tighter than the guard the frozen history was
generated with, so the new pass-set is a subset of the old pass-set. That lets
us measure the effect of the fix exactly, without waiting for a full replay.
"""
from __future__ import annotations

import math

import pandas as pd

from codexuw.confidence_calibration import DEFAULT_EDGE_HISTORY_PATH
from codexuw.credit_policy import (
    MAX_CREDIT_PCT_WIDTH,
    MAX_QUOTE_WIDTH_PCT,
    MIN_CREDIT_PCT_WIDTH,
    MIN_DISTANCE_EXPECTED_MOVE_RATIO,
)
from codexuw.payoff_calibration import _eligible_history, _metrics


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> None:
    history = pd.read_csv(DEFAULT_EDGE_HISTORY_PATH, compression="infer", low_memory=False)
    eligible = _eligible_history(history, asof=pd.Timestamp("2026-07-24"))
    print(f"old-guard eligible rows: {len(eligible)}")

    is_credit = eligible["direction"].astype(str).isin({"Bull Put", "Bear Call"})
    credit_pct = _num(eligible.get("entry_credit_pct_width"))
    quote_width = _num(eligible.get("entry_quote_width_pct"))

    stock = _num(eligible.get("stock_price_eod"))
    short = _num(eligible.get("short_strike_eod"))
    iv30d = _num(eligible.get("iv30d"))
    dte = _num(eligible.get("dte"))
    expected = iv30d * (dte / 365.0).pow(0.5)
    distance = pd.Series(
        [
            (s - k) / s if d == "Bull Put" else (k - s) / s
            for s, k, d in zip(stock, short, eligible["direction"].astype(str))
        ],
        index=eligible.index,
    )
    em_ratio = distance / expected

    width_ok = quote_width.isna() | quote_width.le(MAX_QUOTE_WIDTH_PCT)
    credit_ok = (
        credit_pct.between(MIN_CREDIT_PCT_WIDTH, MAX_CREDIT_PCT_WIDTH)
        & em_ratio.ge(MIN_DISTANCE_EXPECTED_MOVE_RATIO)
        & width_ok
    )
    aligned = (~is_credit & width_ok) | (is_credit & credit_ok)

    print(f"aligned-guard rows:     {int(aligned.sum())}")
    print()

    rows = []
    for label, frame in (
        ("OLD guard (current evidence base)", eligible),
        ("ALIGNED guard (after fix)", eligible[aligned]),
    ):
        m = _metrics(frame, 0.10)
        rows.append(
            {
                "population": label,
                "n": int(m["sample_size"]),
                "stress_PF": round(float(m["profit_factor"]), 3),
                "win_rate": round(float(m["win_rate"]), 3),
                "avg_pnl": round(float(m["average_pnl"]), 2),
                "total_pnl": round(float(m["average_pnl"]) * int(m["sample_size"]), 0),
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))
    print()

    print("--- per strategy family, ALIGNED guard ---")
    fam_rows = []
    kept = eligible[aligned]
    for family, frame in kept.groupby(kept["direction"].astype(str)):
        m = _metrics(frame, 0.10)
        fam_rows.append(
            {
                "family": family,
                "n": int(m["sample_size"]),
                "stress_PF": round(float(m["profit_factor"]), 3),
                "win_rate": round(float(m["win_rate"]), 3),
                "total_pnl": round(float(m["average_pnl"]) * int(m["sample_size"]), 0),
            }
        )
    print(pd.DataFrame(fam_rows).sort_values("stress_PF", ascending=False).to_string(index=False))
    print()

    dropped = eligible[~aligned]
    m = _metrics(dropped, 0.10)
    print(
        f"DROPPED by fix: n={int(m['sample_size'])} "
        f"stress_PF={float(m['profit_factor']):.3f} "
        f"total_pnl={float(m['average_pnl']) * int(m['sample_size']):.0f}"
    )


if __name__ == "__main__":
    main()
