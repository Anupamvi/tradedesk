"""Stage 2: test the contrarian regime map and build the final policy matrix.

Section 1 of the stage-1 review showed credit spreads earn in the regime OPPOSITE
to the shipped trend-following map: sell puts into a downtrend, sell calls into an
uptrend. That is the mean-reversion / volatility-premium trade. This script
walk-forward tests that map, layers macro conditioning, and prints the matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategy_macro_review import (  # reuse loaders
    attach_market_state,
    fmt_folds,
    load,
    profit_factor,
    summarize,
    walk_forward,
)


def main() -> None:
    data = attach_market_state(load())
    guarded = data[data["guard"]].copy()
    sessions = sorted(guarded["asof"].unique())
    all_sessions = sorted(data["asof"].unique())

    bull_put = data["direction"].eq("Bull Put")
    bear_call = data["direction"].eq("Bear Call")
    bull_call = data["direction"].eq("Bull Call")
    contrarian = (bull_put & data["regime"].eq("downtrend")) | (bear_call & data["regime"].eq("uptrend"))
    trend_following = (bull_put & data["regime"].eq("uptrend")) | (bear_call & data["regime"].eq("downtrend"))

    print("=" * 94)
    print("A. CONTRARIAN vs TREND-FOLLOWING CREDIT MAP  (all evaluated rows, no quality gates)")
    print("=" * 94)

    def line(label: str, mask: pd.Series, frame: pd.DataFrame, sess: list) -> None:
        n, pf, total, ok, tried, folds = walk_forward(frame, mask, sess)
        if n == 0:
            print(f"  {label:<46} n=0")
            return
        _, _, avg, _, win = summarize(frame[mask]["pnl"])
        print(
            f"  {label:<46} n={n:>4} PF={pf:>6.3f} win={win:>5.1%} "
            f"tot=${total:>9,.0f}  OOS {ok}/{tried} [{fmt_folds(folds)}]"
        )

    line("CONTRARIAN (BP>down, BC>up)", contrarian, data, all_sessions)
    line("TREND-FOLLOWING = SHIPPED (BP>up, BC>down)", trend_following, data, all_sessions)
    line("Bull Put | downtrend alone", bull_put & data["regime"].eq("downtrend"), data, all_sessions)
    line("Bear Call | uptrend alone", bear_call & data["regime"].eq("uptrend"), data, all_sessions)

    print("\n" + "=" * 94)
    print("B. CONTRARIAN MAP + MACRO CONDITIONING  (all evaluated rows)")
    print("=" * 94)
    line("contrarian", contrarian, data, all_sessions)
    line("contrarian + high market IV", contrarian & data["mkt_vol"].eq("high_iv"), data, all_sessions)
    line("contrarian + low market IV", contrarian & data["mkt_vol"].eq("low_iv"), data, all_sessions)
    line("contrarian + risk_off market", contrarian & data["mkt_regime"].eq("risk_off"), data, all_sessions)
    line("contrarian + risk_on market", contrarian & data["mkt_regime"].eq("risk_on"), data, all_sessions)
    line("contrarian + iv_rank>=30", contrarian & data["iv_rank"].ge(30), data, all_sessions)
    line("contrarian + iv_rank>=40", contrarian & data["iv_rank"].ge(40), data, all_sessions)
    line("contrarian + iv_rank>=50", contrarian & data["iv_rank"].ge(50), data, all_sessions)
    line("contrarian + dte>=28", contrarian & data["dte"].ge(28), data, all_sessions)
    line("contrarian + dte>=28 + iv_rank>=30", contrarian & data["dte"].ge(28) & data["iv_rank"].ge(30), data, all_sessions)

    print("\n" + "=" * 94)
    print("C. DEBIT FAMILIES x MACRO  (all evaluated rows)")
    print("=" * 94)
    line("Bull Call | uptrend", bull_call & data["regime"].eq("uptrend"), data, all_sessions)
    line("Bull Call | uptrend + high mkt IV", bull_call & data["regime"].eq("uptrend") & data["mkt_vol"].eq("high_iv"), data, all_sessions)
    line("Bull Call + high mkt IV (any regime)", bull_call & data["mkt_vol"].eq("high_iv"), data, all_sessions)
    line("Bear Put | any", data["direction"].eq("Bear Put"), data, all_sessions)
    line("Bear Put + low mkt IV", data["direction"].eq("Bear Put") & data["mkt_vol"].eq("low_iv"), data, all_sessions)

    print("\n" + "=" * 94)
    print("D. FINAL CANDIDATE POLICIES ON GUARD-PASSING ROWS")
    print("=" * 94)
    g_bp = guarded["direction"].eq("Bull Put")
    g_bc = guarded["direction"].eq("Bear Call")
    g_credit = guarded["strategy_kind"].eq("Credit")
    g_debit = guarded["strategy_kind"].eq("Debit")
    g_contra = (g_bp & guarded["regime"].eq("downtrend")) | (g_bc & guarded["regime"].eq("uptrend"))
    g_trend = (g_bp & guarded["regime"].eq("uptrend")) | (g_bc & guarded["regime"].eq("downtrend"))

    line("SHIPPED trend map", g_trend, guarded, sessions)
    line("CONTRARIAN map", g_contra, guarded, sessions)
    line("CONTRARIAN + iv_rank>=40", g_contra & guarded["iv_rank"].ge(40), guarded, sessions)
    line("credit: drop only losing pairings", g_credit & ~g_trend, guarded, sessions)
    line("credit: drop losing pairings + iv>=40", g_credit & ~g_trend & guarded["iv_rank"].ge(40), guarded, sessions)
    line("above + debit Bull Call|uptrend", (g_credit & ~g_trend & guarded["iv_rank"].ge(40)) | (g_debit & guarded["direction"].eq("Bull Call") & guarded["regime"].eq("uptrend")), guarded, sessions)
    line("ALL guard-passing (baseline)", pd.Series(True, index=guarded.index), guarded, sessions)

    print("\n" + "=" * 94)
    print("E. MONTHLY P&L: credit drop-losing-pairings + iv_rank>=40 + BullCall|uptrend")
    print("=" * 94)
    final_mask = (g_credit & ~g_trend & guarded["iv_rank"].ge(40)) | (
        g_debit & guarded["direction"].eq("Bull Call") & guarded["regime"].eq("uptrend")
    )
    final = guarded[final_mask].copy()
    final["month"] = final["asof"].astype(str).str[:7]
    wins = months = 0
    for month, grp in final.groupby("month"):
        n, pf, avg, total, win = summarize(grp["pnl"])
        months += 1
        wins += int(total > 0)
        print(f"  {month}  n={n:>4}  PF={pf:>6.3f}  win={win:>6.1%}  avg=${avg:>7.2f}  total=${total:>9,.0f}")
    n, pf, avg, total, win = summarize(final["pnl"])
    print(f"  TOTAL    n={n:>4}  PF={pf:>6.3f}  win={win:>6.1%}  avg=${avg:>7.2f}  total=${total:>9,.0f}")
    print(f"  profitable months: {wins}/{months}")
    per_month = total / max(months, 1)
    print(f"\n  per-contract monthly profit: ${per_month:,.0f}")
    if per_month > 0:
        print(f"  contracts needed for $10,000/month: {10_000 / per_month:.1f}")


if __name__ == "__main__":
    main()
