"""Walk-forward review of every strategy family against regime and derived macro state.

Market state is built from point-in-time cross-sectional aggregates of each session's
own candidate universe (breadth, median IV rank, aggregate flow, median realized vol).
Nothing from the future is used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DETAIL = (
    "/Users/anuppamvi/uw_root/tradedesk/out/"
    "codexdaily_v4_edge_history_v4_2026-07-26/codexuw_replay_detail.csv"
)
WINDOW_END = pd.Timestamp("2026-07-23")


def profit_factor(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    gains = vals[vals > 0].sum()
    losses = -vals[vals < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def summarize(series: pd.Series) -> tuple[int, float, float, float, float]:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    return len(vals), profit_factor(vals), float(vals.mean()), float(vals.sum()), float((vals > 0).mean())


def load() -> pd.DataFrame:
    raw = pd.read_csv(DETAIL, low_memory=False)
    evaluated = raw[pd.to_numeric(raw["exact_evaluated"], errors="coerce").fillna(0).astype(bool)].copy()
    evaluated["pnl"] = pd.to_numeric(evaluated["pnl_1x"], errors="coerce")
    evaluated = evaluated.dropna(subset=["pnl"])
    evaluated["expiry_ts"] = pd.to_datetime(evaluated["expiry"], errors="coerce")
    # Drop truncation bias: trades whose expiry falls beyond the replay window
    # never had a chance to lose, which manufactures survivorship.
    evaluated = evaluated[evaluated["expiry_ts"] <= WINDOW_END].copy()
    evaluated["guard"] = pd.to_numeric(evaluated["replay_guard_pass"], errors="coerce").fillna(0).astype(bool)
    for col in ("iv_rank", "combined_flow_bias", "realized_volatility_30d", "entry_quote_width_pct", "dte"):
        evaluated[col] = pd.to_numeric(evaluated.get(col), errors="coerce")
    return evaluated


def attach_market_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional market state per session, computed only from that session."""
    grouped = frame.groupby("asof")
    state = pd.DataFrame(
        {
            "mkt_breadth": grouped["regime"].apply(lambda s: float((s == "uptrend").mean())),
            "mkt_iv_rank": grouped["iv_rank"].median(),
            "mkt_flow": grouped["combined_flow_bias"].mean(),
            "mkt_rv": grouped["realized_volatility_30d"].median(),
        }
    )
    # Regime of the market itself, from breadth
    state["mkt_regime"] = np.where(
        state["mkt_breadth"] >= 0.55, "risk_on",
        np.where(state["mkt_breadth"] <= 0.30, "risk_off", "mixed"),
    )
    state["mkt_vol"] = np.where(
        state["mkt_iv_rank"] >= state["mkt_iv_rank"].median(), "high_iv", "low_iv"
    )
    return frame.merge(state, left_on="asof", right_index=True, how="left")


def walk_forward(frame: pd.DataFrame, mask: pd.Series, sessions: list, folds: int = 5) -> tuple:
    """Out-of-sample folds over chronological session blocks."""
    subset = frame[mask]
    if subset.empty:
        return 0, float("nan"), float("nan"), 0, 0, []
    blocks = np.array_split(np.array(sessions), folds)
    fold_pfs = []
    passing = 0
    for block in blocks[1:]:
        test = subset[subset["asof"].isin(block)]
        if len(test) < 3:
            continue
        pf = profit_factor(test["pnl"])
        fold_pfs.append(pf)
        if pf >= 1.25:
            passing += 1
    n, pf, avg, total, win = summarize(subset["pnl"])
    return n, pf, total, passing, len(fold_pfs), fold_pfs


def fmt_folds(folds: list) -> str:
    return ",".join("inf" if not np.isfinite(f) else f"{f:.2f}" for f in folds)


def main() -> None:
    data = attach_market_state(load())
    sessions = sorted(data["asof"].unique())
    print(f"evaluated (truncation-corrected): {len(data)} rows over {len(sessions)} sessions")
    print(f"guard-passing: {int(data['guard'].sum())}")

    print("\n" + "=" * 92)
    print("1. EVERY STRATEGY FAMILY x REGIME  (all evaluated rows, no quality gates)")
    print("=" * 92)
    print(f"{'strategy':<14}{'regime':<12}{'n':>6}{'PF':>8}{'win':>8}{'avg$':>10}{'total$':>12}")
    for (direction, regime), grp in data.groupby(["direction", "regime"]):
        n, pf, avg, total, win = summarize(grp["pnl"])
        if n < 20:
            continue
        print(f"{direction:<14}{regime:<12}{n:>6}{pf:>8.3f}{win:>8.1%}{avg:>10.2f}{total:>12,.0f}")

    print("\n" + "=" * 92)
    print("2. STRATEGY FAMILY x MARKET REGIME (breadth-derived)")
    print("=" * 92)
    print(f"{'strategy':<14}{'mkt_regime':<12}{'n':>6}{'PF':>8}{'win':>8}{'avg$':>10}{'total$':>12}")
    for (direction, mkt), grp in data.groupby(["direction", "mkt_regime"]):
        n, pf, avg, total, win = summarize(grp["pnl"])
        if n < 20:
            continue
        print(f"{direction:<14}{mkt:<12}{n:>6}{pf:>8.3f}{win:>8.1%}{avg:>10.2f}{total:>12,.0f}")

    print("\n" + "=" * 92)
    print("3. STRATEGY FAMILY x MARKET VOL STATE")
    print("=" * 92)
    print(f"{'strategy':<14}{'mkt_vol':<12}{'n':>6}{'PF':>8}{'win':>8}{'avg$':>10}{'total$':>12}")
    for (direction, vol), grp in data.groupby(["direction", "mkt_vol"]):
        n, pf, avg, total, win = summarize(grp["pnl"])
        if n < 20:
            continue
        print(f"{direction:<14}{vol:<12}{n:>6}{pf:>8.3f}{win:>8.1%}{avg:>10.2f}{total:>12,.0f}")

    print("\n" + "=" * 92)
    print("4. WALK-FORWARD: candidate policies on GUARD-PASSING rows")
    print("=" * 92)
    guarded = data[data["guard"]].copy()
    gs = sorted(guarded["asof"].unique())

    def report(label: str, mask: pd.Series) -> None:
        n, pf, total, ok, tried, folds = walk_forward(guarded, mask, gs)
        if n == 0:
            print(f"  {label:<44} n=0")
            return
        print(f"  {label:<44} n={n:>4} PF={pf:>6.3f} tot=${total:>9,.0f}  OOS {ok}/{tried} [{fmt_folds(folds)}]")

    is_credit = guarded["strategy_kind"].eq("Credit")
    is_debit = guarded["strategy_kind"].eq("Debit")
    bull_put = guarded["direction"].eq("Bull Put")
    bear_call = guarded["direction"].eq("Bear Call")

    report("ALL guard-passing", pd.Series(True, index=guarded.index))
    report("credit only", is_credit)
    report("debit only", is_debit)
    print()
    report("SHIPPED map (BP>up, BC>down)", (bull_put & guarded["regime"].eq("uptrend")) | (bear_call & guarded["regime"].eq("downtrend")))
    report("BP>uptrend only; BC>any regime", is_credit & ~(bull_put & guarded["regime"].ne("uptrend")))
    report("  + drop risk_off sessions", is_credit & ~(bull_put & guarded["regime"].ne("uptrend")) & guarded["mkt_regime"].ne("risk_off"))
    report("  + high market IV only", is_credit & ~(bull_put & guarded["regime"].ne("uptrend")) & guarded["mkt_vol"].eq("high_iv"))
    report("  + iv_rank>=40", is_credit & ~(bull_put & guarded["regime"].ne("uptrend")) & guarded["iv_rank"].ge(40))
    print()
    report("credit + debit combined best", (is_credit & ~(bull_put & guarded["regime"].ne("uptrend"))) | (is_debit & guarded["regime"].eq("uptrend")))

    print("\n" + "=" * 92)
    print("5. MONTHLY P&L of the leading policy")
    print("=" * 92)
    best = guarded[is_credit & ~(bull_put & guarded["regime"].ne("uptrend"))].copy()
    best["month"] = best["asof"].astype(str).str[:7]
    wins = 0
    months = 0
    for month, grp in best.groupby("month"):
        n, pf, avg, total, win = summarize(grp["pnl"])
        months += 1
        wins += int(total > 0)
        print(f"  {month}  n={n:>4}  PF={pf:>6.3f}  win={win:>6.1%}  total=${total:>9,.0f}")
    print(f"  profitable months: {wins}/{months}")


if __name__ == "__main__":
    main()
