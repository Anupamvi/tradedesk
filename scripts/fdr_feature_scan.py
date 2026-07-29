"""FDR-controlled discovery then untouched-half validation of panel features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
OUT = Path("/Users/anuppamvi/uw_root/tradedesk/out/fdr_feature_scan.csv")
SPLIT = pd.Timestamp("2026-04-14")
MIN_NAMES = 100
MIN_DAYS = 40
FDR_Q = 0.10

SKIP = {
    "date", "ticker", "sector", "issue_type", "next_earnings_date", "close", "marketcap",
}


def nw_stats(values: list[float], lag: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < MIN_DAYS:
        return np.nan, np.nan, np.nan
    centered = array - array.mean()
    n = len(centered)
    variance = centered @ centered / n
    for step in range(1, min(lag, n - 1) + 1):
        covariance = centered[step:] @ centered[:-step] / n
        variance += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    standard_error = np.sqrt(max(variance, 1e-12) / n)
    t_stat = array.mean() / standard_error
    p_value = 2.0 * stats.norm.sf(abs(t_stat))
    return array.mean(), t_stat, p_value


def bh_selected(p_values: pd.Series, q: float) -> pd.Series:
    selected = pd.Series(False, index=p_values.index)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return selected
    thresholds = q * np.arange(1, len(valid) + 1) / len(valid)
    passed = valid.to_numpy() <= thresholds
    if passed.any():
        cutoff = valid.iloc[np.where(passed)[0][-1]]
        selected.loc[p_values <= cutoff] = True
    return selected


def main() -> None:
    panel = pd.read_csv(PANEL, low_memory=False)
    panel.date = pd.to_datetime(panel.date)
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
    ].copy()
    if panel.duplicated(["ticker", "date"]).any():
        raise ValueError("duplicate ticker/date rows")
    panel = panel.sort_values(["ticker", "date"])
    dates = np.sort(panel.date.unique())
    position = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    lookup = panel.set_index(["ticker", "date"])["close"]
    for horizon in (1, 5):
        entry_dates = panel.date.map(
            lambda date: dates[position[date] + 1] if position[date] + 1 < len(dates) else pd.NaT
        )
        exit_dates = panel.date.map(
            lambda date: dates[position[date] + horizon + 1]
            if position[date] + horizon + 1 < len(dates)
            else pd.NaT
        )
        panel[f"fwd_{horizon}d"] = (
            lookup.reindex(pd.MultiIndex.from_arrays([panel.ticker, exit_dates])).to_numpy()
            / lookup.reindex(pd.MultiIndex.from_arrays([panel.ticker, entry_dates])).to_numpy()
            - 1.0
        )

    # Cross-sectional ranks make raw and *_xs versions identical; keep raw only.
    features = [
        column for column in panel.columns
        if column not in SKIP
        and not column.startswith("fwd_")
        and not column.endswith("_xs")
        and pd.api.types.is_numeric_dtype(panel[column])
    ]
    rows = []
    for feature in features:
        for horizon in (1, 5):
            target = f"fwd_{horizon}d"
            train_ics: list[float] = []
            test_ics: list[float] = []
            for date, day in panel.groupby("date"):
                sample = day[[feature, target]].dropna()
                if len(sample) < MIN_NAMES or sample[feature].nunique() < 10:
                    continue
                ic = stats.spearmanr(sample[feature], sample[target]).correlation
                if not np.isfinite(ic):
                    continue
                (train_ics if date < SPLIT else test_ics).append(float(ic))
            train_mean, train_t, train_p = nw_stats(train_ics, horizon)
            test_mean, test_t, test_p = nw_stats(test_ics, horizon)
            rows.append(
                {
                    "feature": feature,
                    "horizon": f"{horizon}d",
                    "train_days": len(train_ics),
                    "train_ic": train_mean,
                    "train_nw_t": train_t,
                    "train_p": train_p,
                    "test_days": len(test_ics),
                    "test_ic": test_mean,
                    "test_nw_t": test_t,
                    "test_p": test_p,
                }
            )
    result = pd.DataFrame(rows)
    result["train_fdr_selected"] = bh_selected(result.train_p, FDR_Q)
    result["same_sign"] = np.sign(result.train_ic) == np.sign(result.test_ic)
    selected_count = int(result.train_fdr_selected.sum())
    result["test_bonferroni_pass"] = (
        result.train_fdr_selected
        & result.same_sign
        & (result.test_p <= (0.05 / max(selected_count, 1)))
    )
    result.to_csv(OUT, index=False)

    print(f"features={len(features)} tests={len(result)} train_FDR_q={FDR_Q}")
    print(f"train_selected={selected_count} test_confirmed={int(result.test_bonferroni_pass.sum())}")
    print("\n=== TRAIN-FDR SELECTED, THEN UNTOUCHED TEST ===")
    selected = result[result.train_fdr_selected].sort_values("train_p")
    print(selected.round(5).to_string(index=False) if len(selected) else "NONE")
    print("\n=== TOP TRAIN RESULTS (diagnostic only) ===")
    print(result.sort_values("train_p").head(15).round(5).to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
