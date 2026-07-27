"""Run the real payoff calibration against a policy-aligned evidence base.

Writes a re-guarded copy of the frozen history (the aligned guard is strictly
tighter, so this is an exact projection) and feeds it through the unmodified
walk-forward calibration to see which lanes actually validate.
"""
from __future__ import annotations

import sys

import pandas as pd

from codexuw.confidence_calibration import DEFAULT_EDGE_HISTORY_PATH
from codexuw.credit_policy import (
    MAX_CREDIT_PCT_WIDTH,
    MAX_QUOTE_WIDTH_PCT,
    MIN_CREDIT_PCT_WIDTH,
    MIN_DISTANCE_EXPECTED_MOVE_RATIO,
)
from codexuw.payoff_calibration import build_default_payoff_calibration


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def reguard(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()
    is_credit = frame["direction"].astype(str).isin({"Bull Put", "Bear Call"})
    credit_pct = _num(frame.get("entry_credit_pct_width"))
    quote_width = _num(frame.get("entry_quote_width_pct"))
    stock = _num(frame.get("stock_price_eod"))
    short = _num(frame.get("short_strike_eod"))
    expected = _num(frame.get("iv30d")) * (_num(frame.get("dte")) / 365.0).pow(0.5)
    distance = pd.Series(
        [
            (s - k) / s if d == "Bull Put" else (k - s) / s
            for s, k, d in zip(stock, short, frame["direction"].astype(str))
        ],
        index=frame.index,
    )
    em_ratio = distance / expected
    width_ok = quote_width.isna() | quote_width.le(MAX_QUOTE_WIDTH_PCT)
    credit_ok = (
        credit_pct.between(MIN_CREDIT_PCT_WIDTH, MAX_CREDIT_PCT_WIDTH)
        & em_ratio.ge(MIN_DISTANCE_EXPECTED_MOVE_RATIO)
        & width_ok
    )
    aligned = (~is_credit & width_ok) | (is_credit & credit_ok)
    prior = frame["replay_guard_pass"].map(
        lambda v: str(v).strip().lower() in {"1", "true", "yes", "y"}
    )
    frame["replay_guard_pass"] = prior & aligned
    return frame


def main() -> None:
    asof = sys.argv[1] if len(sys.argv) > 1 else "2026-07-24"
    history = pd.read_csv(DEFAULT_EDGE_HISTORY_PATH, compression="infer", low_memory=False)
    out = DEFAULT_EDGE_HISTORY_PATH.parent / "_reguarded_projection.csv.gz"
    reguard(history).to_csv(out, index=False, compression="gzip")

    for label, path in (("FROZEN (old guard)", DEFAULT_EDGE_HISTORY_PATH), ("ALIGNED (fixed)", out)):
        summary, groups, _ = build_default_payoff_calibration(asof=asof, history_path=path)
        print(f"=== {label} ===")
        print(
            f"  eligible_rows={summary['eligible_rows']} "
            f"evidence={summary['evidence_first_asof']}..{summary['evidence_last_asof']} "
            f"stale={summary['evidence_staleness_days']}d"
        )
        print(f"  status={summary['status']} passed_lanes={summary['passed_lane_count']}")
        for lane in summary["passed_lanes"]:
            print(f"    PASS -> {lane}")
        if not groups.empty:
            passed = groups[groups["payoff_calibration_status"].eq("PASS")]
            cols = [c for c in ("group_key", "sample_size", "profit_factor_stress_10", "oos_sample_size") if c in passed.columns]
            if not passed.empty and cols:
                print(passed[cols].to_string(index=False))
            print("  -- top lanes by sample --")
            diag = [c for c in groups.columns if c in (
                "group_key", "payoff_calibration_status", "payoff_calibration_reason",
                "sample_size", "profit_factor_stress_10", "oos_sample_size",
                "oos_profit_factor", "failed_window_rate",
            )]
            print(groups.sort_values("sample_size", ascending=False)[diag].head(12).to_string(index=False))
        print()
    out.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
