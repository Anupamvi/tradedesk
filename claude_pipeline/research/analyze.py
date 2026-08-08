"""Stage 4 + 5: the honest baseline, then pre-specified selection hypotheses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from claude_pipeline import context, panel as panel_mod, selection, stats

OUT = Path("/Users/anuppamvi/tradedesk/out/claude_pipeline")


def _row(label: str, frame: pd.DataFrame) -> dict:
    card = stats.scorecard(frame, label)
    return {
        "population": label,
        "n": card.get("n", 0),
        "win%": round(100 * card.get("win_rate", np.nan), 1) if card.get("n") else np.nan,
        "avg$": round(card.get("avg_pnl", np.nan), 2) if card.get("n") else np.nan,
        "total$": round(card.get("total_pnl", np.nan), 0) if card.get("n") else np.nan,
        "PF": round(card.get("profit_factor", np.nan), 3) if card.get("n") else np.nan,
        "maxDD$": round(card.get("max_drawdown", np.nan), 0) if card.get("n") else np.nan,
        "mo+": f"{card.get('months_profitable', 0)}/{card.get('months_total', 0)}",
        "boot_p05": round(card.get("boot_p05", np.nan), 2) if card.get("n") else np.nan,
        "passes": card.get("passes", False),
    }


def load(name: str = "backtest_full") -> pd.DataFrame:
    return pd.read_csv(OUT / f"{name}.csv.gz", low_memory=False)


def main(name: str = "backtest_full") -> None:
    trades = load(name)
    print(f"=== raw candidate universe: {len(trades):,} structures ===")
    print("outcome mix:", trades["outcome"].value_counts().to_dict())
    resolved = stats.resolved(trades)
    entered = trades[trades["outcome"].isin(["expiry", "take_profit", "censored"])]
    print(f"entered (a real quote existed): {len(entered):,}"
          f" | resolved: {len(resolved):,} ({len(resolved)/max(len(entered),1):.1%})")
    print(f"rejected as implausible quotes: {int((trades['outcome'] == 'implausible_quote').sum()):,}"
          f" | unquotable: {int((trades['outcome'] == 'unquotable').sum()):,}")

    print("\n=== STAGE 4: baseline by family, NO selection ===")
    rows = [_row(family, group) for family, group in resolved.groupby("family")]
    rows.append(_row("ALL", resolved))
    print(pd.DataFrame(rows).to_string(index=False))

    panel = panel_mod.build()
    regime = context.market_regime(panel)
    filings_path = OUT / "edgar"
    filings = None
    if filings_path.exists() and any(filings_path.glob("2026-*.csv.gz")):
        raw = pd.concat(
            [pd.read_csv(p, dtype={"cik": "Int64"}) for p in sorted(filings_path.glob("20*.csv.gz"))],
            ignore_index=True,
        )
        filings = context.filing_features(raw)
        print(f"\ncontext: {len(raw):,} filings over {raw['session'].nunique()} sessions")

    enriched = selection.attach_features(resolved, panel, regime, filings)
    enriched.to_csv(OUT / f"{name}_enriched.csv.gz", index=False, compression="gzip")
    print(f"regime sessions: {regime['session'].nunique()} | trend mix: "
          f"{regime['trend'].value_counts().to_dict()}")

    credit = enriched[enriched["family"].isin(["bull_put_credit", "bear_call_credit"])]
    debit = enriched[enriched["family"].isin(["bull_call_debit", "bear_put_debit"])]
    longs = enriched[enriched["family"].isin(["long_call", "long_put"])]

    print("\n=== STAGE 5: pre-specified hypotheses (full sample, in-sample only) ===")
    hypotheses = [
        ("credit: all", credit),
        ("credit: IV rich vs realized (iv_rv>=1.1)", credit[credit["iv_rv_ratio"] >= 1.1]),
        ("credit: IV cheap (iv_rv<1.0)", credit[credit["iv_rv_ratio"] < 1.0]),
        ("credit: no earnings before expiry", credit[~credit["earnings_before_expiry"]]),
        ("credit: earnings before expiry", credit[credit["earnings_before_expiry"]]),
        ("credit: further OTM (6%)", credit[credit["otm_target"] >= 0.06]),
        ("credit: high liquidity ($100M+)", credit[credit["dollar_volume"] >= 1e8]),
        ("debit: all", debit),
        ("long options: all", longs),
    ]
    print(pd.DataFrame([_row(label, frame) for label, frame in hypotheses]).to_string(index=False))

    print("\n=== monthly P&L, credit families, no selection ===")
    print(stats.monthly(credit).round(0).to_string())


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "backtest_full")
