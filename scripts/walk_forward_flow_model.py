"""Purged walk-forward cross-sectional model over the four full-history UW feeds.

Train/regularization selection uses only dates before 2026-04-14. The final test
period is untouched 2026-04-14 onward. Targets enter next close and exit five
trading sessions later. Features are same-day cross-sectional ranks; each date
receives equal total regression weight.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

PANEL = Path("/Users/anuppamvi/uw_root/tradedesk/out/uw_all_feeds.csv")
OUT = Path("/Users/anuppamvi/uw_root/tradedesk/out/walk_forward_flow_model.csv")
COEFS = Path("/Users/anuppamvi/uw_root/tradedesk/out/walk_forward_flow_model_coefficients.csv")
HORIZON = 5
TEST_START = pd.Timestamp("2026-04-14")
INNER_VALIDATION_START = pd.Timestamp("2026-03-02")
PURGE_DAYS = 6
COST = 0.001

FEATURES = [
    # Screener / price / volatility.
    "iv_rank", "vrp_ratio", "iv_chg_1w", "iv_chg_1m", "implied_move_perc",
    "pos_52w", "ret_1d", "stock_vol_surge", "call_vol_surge", "put_vol_surge",
    "put_call_ratio", "prem_tilt", "net_prem_tilt", "call_oi_chg", "put_oi_chg",
    # Hot chains.
    "hc_multileg_share", "hc_sweep_share", "hc_floor_share", "hc_cross_share",
    "hc_opening_share", "hc_quote_churn", "hc_premium", "hc_chains", "hc_dir_bias",
    # OI-confirmed opening flow.
    "oi_built_contracts", "oi_built_premium", "oi_signed_premium", "oi_n_chains",
    "oi_median_dte", "oi_nearmoney_premium", "oi_newlong_premium",
    "oi_newshort_premium", "oi_dir_bias", "oi_open_conviction", "oi_nearmoney_share",
    # Dark pool.
    "dp_premium", "dp_block_premium", "dp_late_premium", "dp_prints",
    "dp_median_size", "dp_bias", "dp_block_bias", "dp_late_bias", "dp_block_share",
]


def nw_t(values: pd.Series, lag: int = HORIZON) -> float:
    values = values.dropna().to_numpy()
    if len(values) < 10:
        return np.nan
    centered = values - values.mean()
    n = len(centered)
    variance = centered @ centered / n
    for step in range(1, min(lag, n - 1) + 1):
        covariance = centered[step:] @ centered[:-step] / n
        variance += 2.0 * (1.0 - step / (lag + 1.0)) * covariance
    return values.mean() / np.sqrt(max(variance, 1e-12) / n)


def add_target(panel: pd.DataFrame) -> pd.DataFrame:
    dates = np.sort(panel.date.unique())
    position = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    lookup = panel.set_index(["ticker", "date"])["close"]
    entry_dates = panel.date.map(
        lambda date: dates[position[date] + 1] if position[date] + 1 < len(dates) else pd.NaT
    )
    exit_dates = panel.date.map(
        lambda date: dates[position[date] + HORIZON + 1]
        if position[date] + HORIZON + 1 < len(dates)
        else pd.NaT
    )
    panel["return"] = (
        lookup.reindex(pd.MultiIndex.from_arrays([panel.ticker, exit_dates])).to_numpy()
        / lookup.reindex(pd.MultiIndex.from_arrays([panel.ticker, entry_dates])).to_numpy()
        - 1.0
    )
    panel["target"] = panel["return"] - panel.groupby(["date", "sector"])["return"].transform("mean")
    panel["target"] = panel["target"].clip(-0.20, 0.20)
    return panel


def rank_features(panel: pd.DataFrame) -> pd.DataFrame:
    ranked = pd.DataFrame(index=panel.index)
    for feature in FEATURES:
        rank = panel.groupby("date")[feature].rank(pct=True)
        ranked[feature] = rank.fillna(0.5) - 0.5
    return ranked


def daily_portfolios(frame: pd.DataFrame, score: np.ndarray, basket_size: int | None = None) -> pd.DataFrame:
    scored = frame[["date", "ticker", "sector", "return"]].copy()
    scored["score"] = score
    rows = []
    for date, day in scored.groupby("date"):
        day = day.dropna(subset=["return", "score"])
        if len(day) < 200:
            continue
        if basket_size is None:
            rank = day.score.rank(pct=True, method="first")
            longs = day[rank >= 0.9]
            shorts = day[rank <= 0.1]
        else:
            longs = day.nlargest(basket_size, "score")
            shorts = day.nsmallest(basket_size, "score")
        rows.append(
            {
                "date": date,
                "long": longs["return"].mean(),
                "short": shorts["return"].mean(),
                "spread": longs["return"].mean() - shorts["return"].mean(),
                "long_n": len(longs),
                "short_n": len(shorts),
            }
        )
    return pd.DataFrame(rows)


def portfolio_score(rows: pd.DataFrame) -> float:
    return rows.spread.mean() if len(rows) >= 10 else -np.inf


def main() -> None:
    columns = ["date", "ticker", "sector", "issue_type", "marketcap", "close"] + FEATURES
    panel = pd.read_csv(PANEL, usecols=columns, low_memory=False)
    panel.date = pd.to_datetime(panel.date)
    panel = panel[
        (panel.issue_type == "Common Stock")
        & (panel.marketcap.fillna(0) >= 2e9)
    ].copy()
    if panel.duplicated(["ticker", "date"]).any():
        raise ValueError("duplicate ticker/date rows")
    panel = panel.sort_values(["ticker", "date"])
    panel = add_target(panel)
    features = rank_features(panel)

    dates = np.sort(panel.date.unique())
    inner_cutoff = pd.Timestamp(dates[np.searchsorted(dates, INNER_VALIDATION_START.to_datetime64()) - PURGE_DAYS])
    test_cutoff = pd.Timestamp(dates[np.searchsorted(dates, TEST_START.to_datetime64()) - PURGE_DAYS])
    fit_mask = (panel.date < inner_cutoff) & panel.target.notna()
    validation_mask = (panel.date >= INNER_VALIDATION_START) & (panel.date < test_cutoff) & panel.target.notna()
    pretest_mask = (panel.date < test_cutoff) & panel.target.notna()
    test_mask = (panel.date >= TEST_START) & panel.target.notna()

    alphas = (0.01, 0.1, 1.0, 10.0)
    candidates = []
    fit_weights = 1.0 / panel.loc[fit_mask].groupby("date")["ticker"].transform("size")
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(features.loc[fit_mask], panel.loc[fit_mask, "target"], sample_weight=fit_weights)
        prediction = model.predict(features.loc[validation_mask])
        rows = daily_portfolios(panel.loc[validation_mask], prediction)
        candidates.append({"alpha": alpha, "spread": rows.spread.mean(), "nw_t": nw_t(rows.spread)})
    tuning = pd.DataFrame(candidates)
    chosen_alpha = float(tuning.sort_values(["spread", "nw_t"], ascending=False).iloc[0].alpha)

    weights = 1.0 / panel.loc[pretest_mask].groupby("date")["ticker"].transform("size")
    model = Ridge(alpha=chosen_alpha, solver="lsqr")
    model.fit(features.loc[pretest_mask], panel.loc[pretest_mask, "target"], sample_weight=weights)
    test_prediction = model.predict(features.loc[test_mask])

    summaries = []
    all_rows = []
    for basket_size in (None, 10, 20, 40, 80):
        rows = daily_portfolios(panel.loc[test_mask], test_prediction, basket_size=basket_size)
        label = "decile" if basket_size is None else f"top_bottom_{basket_size}"
        rows["portfolio"] = label
        all_rows.append(rows)
        gross = rows.spread
        net = gross - COST
        summaries.append(
            {
                "portfolio": label,
                "days": len(rows),
                "gross_spread": gross.mean(),
                "gross_nw_t": nw_t(gross),
                "net_spread": net.mean(),
                "net_nw_t": nw_t(net),
                "gross_hit": (gross > 0).mean(),
                "net_hit": (net > 0).mean(),
            }
        )
    result = pd.DataFrame(summaries)
    result.to_csv(OUT, index=False)
    coefficients = pd.DataFrame({"feature": FEATURES, "coefficient": model.coef_})
    coefficients["abs_coefficient"] = coefficients.coefficient.abs()
    coefficients.sort_values("abs_coefficient", ascending=False).to_csv(COEFS, index=False)

    print("=== INNER-TRAIN REGULARIZATION SELECTION ===")
    print(tuning.round(5).to_string(index=False))
    print(f"chosen_alpha={chosen_alpha}")
    print("\n=== UNTOUCHED TEST: 2026-04-14 onward ===")
    print(result.round(4).to_string(index=False))
    print("\n=== TOP COEFFICIENTS ===")
    print(coefficients.sort_values("abs_coefficient", ascending=False).head(15).round(5).to_string(index=False))

    decile = all_rows[0].copy()
    decile["month"] = decile.date.dt.to_period("M").astype(str)
    print("\n=== TEST MONTHS, DECILE ===")
    print(decile.groupby("month").agg(days=("spread", "size"), spread=("spread", "mean"), hit=("spread", lambda x: (x > 0).mean())).round(4).to_string())
    print(f"\nwrote {OUT} and {COEFS}")


if __name__ == "__main__":
    main()
