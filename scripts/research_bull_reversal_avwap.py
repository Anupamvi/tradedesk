"""Research-only bullish reversal study using selloff-anchored daily VWAP.

This script does not modify or feed any live pipeline. It asks whether a
beaten-down liquid stock becomes a better bullish candidate only after buyers
reclaim the volume-weighted cost basis established since a high-volume shock.

Daily OHLCV bars are used, so ``anchored_vwap`` is a daily-bar proxy rather
than an intraday execution VWAP. All inputs are point-in-time. Signals use only
information available at that close and outcomes begin after the signal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MIN_MARKET_CAP = 2_000_000_000.0
MIN_DOLLAR_VOLUME = 50_000_000.0
MIN_PRICE = 5.0
SHOCK_Z = -3.0
MIN_DRAWDOWN_63 = -0.12
MIN_SHOCK_VOLUME_RATIO = 1.25
MAX_RECLAIM_WAIT = 10
SHOCK_COOLDOWN = 21
SPLIT_DATE = "2026-04-14"


@dataclass(frozen=True)
class Signal:
    event_id: str
    variant: str
    ticker: str
    sector: str
    shock_date: str
    signal_date: str
    shock_z: float
    drawdown_63: float
    shock_volume_ratio: float
    anchored_vwap: float
    close: float
    volume_ratio: float
    spy_above_sma20: bool
    spy_return_5d: float
    relative_return_2d: float
    post_earnings_shock: bool


def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["asof"] = pd.to_datetime(panel["asof"]).dt.strftime("%Y-%m-%d")
    panel["ticker"] = panel["ticker"].astype(str).str.upper()
    panel = panel.sort_values(["ticker", "asof"]).drop_duplicates(["ticker", "asof"])
    for column in (
        "close",
        "high",
        "low",
        "total_volume",
        "marketcap",
        "week_52_high",
        "week_52_low",
        "iv30d",
        "iv_rank",
    ):
        if column in panel:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")

    grouped = panel.groupby("ticker", observed=True)
    panel["ret_1d"] = grouped["close"].pct_change(fill_method=None)
    panel["ret_2d"] = grouped["close"].pct_change(2, fill_method=None)
    panel["ret_5d"] = grouped["close"].pct_change(5, fill_method=None)
    panel["adm21"] = grouped["ret_1d"].transform(
        lambda values: values.abs().rolling(21, min_periods=10).mean()
    )
    panel["high_63"] = grouped["close"].transform(
        lambda values: values.rolling(63, min_periods=20).max()
    )
    panel["drawdown_63"] = panel["close"] / panel["high_63"] - 1.0
    panel["volume_median_20"] = grouped["total_volume"].transform(
        lambda values: values.rolling(20, min_periods=10).median()
    )
    panel["volume_ratio"] = panel["total_volume"] / panel["volume_median_20"].replace(0, np.nan)
    panel["shock_z"] = panel["ret_1d"] / panel["adm21"].replace(0, np.nan)
    panel["typical_price"] = (panel["high"] + panel["low"] + panel["close"]) / 3.0
    panel["dollar_volume"] = panel["close"] * panel["total_volume"]
    panel["prior_high"] = grouped["high"].shift(1)

    for horizon in (5, 21):
        panel[f"fwd_{horizon}d"] = grouped["close"].shift(-horizon) / panel["close"] - 1.0

    sector_return = panel.groupby(["asof", "sector"], observed=True)["ret_2d"].transform("median")
    panel["relative_return_2d"] = panel["ret_2d"] - sector_return

    asof_date = pd.to_datetime(panel["asof"])
    next_earnings = pd.to_datetime(panel.get("next_earnings_date"), errors="coerce")
    session_index = panel.groupby("ticker", observed=True).cumcount()
    # The screener rolls from the just-reported date to the next quarter after
    # earnings. Carry the last date that has actually occurred forward by
    # session index; a simple one-row shift misses shocks two or three sessions
    # after the calendar rolls.
    event_session = session_index.where(next_earnings.notna() & next_earnings.le(asof_date))
    event_session = event_session.groupby(panel["ticker"], observed=True).ffill()
    sessions_since_earnings = session_index - event_session
    panel["post_earnings_shock"] = sessions_since_earnings.between(0, 3)

    spy = panel[panel["ticker"].eq("SPY")][["asof", "close", "ret_5d"]].copy()
    spy["spy_sma20"] = spy["close"].rolling(20, min_periods=10).mean()
    spy["spy_above_sma20"] = spy["close"] >= spy["spy_sma20"]
    spy = spy.rename(columns={"ret_5d": "spy_return_5d"})[
        ["asof", "spy_above_sma20", "spy_return_5d"]
    ]
    panel = panel.merge(spy, on="asof", how="left")
    return panel.replace([np.inf, -np.inf], np.nan)


def eligible_universe(panel: pd.DataFrame) -> pd.Series:
    return (
        panel.get("issue_type", pd.Series("", index=panel.index)).astype(str).eq("Common Stock")
        & panel["marketcap"].ge(MIN_MARKET_CAP)
        & panel["close"].ge(MIN_PRICE)
        & panel["dollar_volume"].ge(MIN_DOLLAR_VOLUME)
        & panel["sector"].notna()
    )


def shock_rows(panel: pd.DataFrame) -> pd.DataFrame:
    shocks = panel[
        eligible_universe(panel)
        & panel["shock_z"].le(SHOCK_Z)
        & panel["drawdown_63"].le(MIN_DRAWDOWN_63)
        & panel["volume_ratio"].ge(MIN_SHOCK_VOLUME_RATIO)
    ].copy()
    if shocks.empty:
        return shocks

    session_index = {day: index for index, day in enumerate(sorted(panel["asof"].unique()))}
    kept = []
    for _, block in shocks.groupby("ticker", observed=True):
        last_index = -10_000
        for index, row in block.sort_values("asof").iterrows():
            current = session_index[row["asof"]]
            if current - last_index >= SHOCK_COOLDOWN:
                kept.append(index)
                last_index = current
    return shocks.loc[kept].sort_values(["asof", "ticker"])


def event_signals(panel: pd.DataFrame, shocks: pd.DataFrame) -> pd.DataFrame:
    by_ticker = {
        ticker: block.sort_values("asof").reset_index(drop=True)
        for ticker, block in panel.groupby("ticker", observed=True)
    }
    records: list[Signal] = []

    for shock in shocks.itertuples():
        history = by_ticker[shock.ticker]
        hits = history.index[history["asof"].eq(shock.asof)]
        if len(hits) != 1:
            continue
        start = int(hits[0])
        event_id = f"{shock.ticker}:{shock.asof}"

        records.append(
            Signal(
                event_id=event_id,
                variant="shock_close",
                ticker=shock.ticker,
                sector=str(shock.sector),
                shock_date=shock.asof,
                signal_date=shock.asof,
                shock_z=float(shock.shock_z),
                drawdown_63=float(shock.drawdown_63),
                shock_volume_ratio=float(shock.volume_ratio),
                anchored_vwap=float(shock.typical_price),
                close=float(shock.close),
                volume_ratio=float(shock.volume_ratio),
                spy_above_sma20=bool(shock.spy_above_sma20),
                spy_return_5d=float(shock.spy_return_5d),
                relative_return_2d=float(shock.relative_return_2d),
                post_earnings_shock=bool(shock.post_earnings_shock),
            )
        )

        window = history.iloc[start : min(start + MAX_RECLAIM_WAIT + 1, len(history))].copy()
        if len(window) < 2:
            continue
        weighted = window["typical_price"] * window["total_volume"]
        window["anchored_vwap"] = weighted.cumsum() / window["total_volume"].cumsum().replace(0, np.nan)
        window["was_below_avwap"] = window["close"].le(window["anchored_vwap"]).cummax()

        variants = {
            "avwap_reclaim": window["close"].gt(window["anchored_vwap"]) & window["was_below_avwap"],
            "avwap_confirmed": (
                window["close"].gt(window["anchored_vwap"])
                & window["was_below_avwap"]
                & window["close"].gt(window["prior_high"])
            ),
            "avwap_market_aligned": (
                window["close"].gt(window["anchored_vwap"])
                & window["was_below_avwap"]
                & window["close"].gt(window["prior_high"])
                & window["spy_above_sma20"].fillna(False)
                & window["spy_return_5d"].ge(-0.02)
                & window["relative_return_2d"].gt(0)
                & window["volume_ratio"].ge(1.0)
            ),
        }

        for variant, condition in variants.items():
            candidates = window.iloc[1:][condition.iloc[1:]]
            if candidates.empty:
                continue
            signal = candidates.iloc[0]
            records.append(
                Signal(
                    event_id=event_id,
                    variant=variant,
                    ticker=shock.ticker,
                    sector=str(shock.sector),
                    shock_date=shock.asof,
                    signal_date=str(signal["asof"]),
                    shock_z=float(shock.shock_z),
                    drawdown_63=float(shock.drawdown_63),
                    shock_volume_ratio=float(shock.volume_ratio),
                    anchored_vwap=float(signal["anchored_vwap"]),
                    close=float(signal["close"]),
                    volume_ratio=float(signal["volume_ratio"]),
                    spy_above_sma20=bool(signal["spy_above_sma20"]),
                    spy_return_5d=float(signal["spy_return_5d"]),
                    relative_return_2d=float(signal["relative_return_2d"]),
                    post_earnings_shock=bool(shock.post_earnings_shock),
                )
            )

    signals = pd.DataFrame([record.__dict__ for record in records])
    if signals.empty:
        return signals
    outcomes = panel[["asof", "ticker", "fwd_5d", "fwd_21d", "iv30d", "iv_rank"]].rename(
        columns={"asof": "signal_date"}
    )
    return signals.merge(outcomes, on=["signal_date", "ticker"], how="left")


def metrics(frame: pd.DataFrame, outcome: str) -> dict[str, float]:
    values = pd.to_numeric(frame[outcome], errors="coerce").dropna()
    if values.empty:
        return {"n": 0, "days": 0, "mean": np.nan, "median": np.nan, "win": np.nan}
    return {
        "n": int(len(values)),
        "days": int(frame.loc[values.index, "signal_date"].nunique()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "win": float(values.gt(0).mean()),
    }


def clustered_interval(frame: pd.DataFrame, outcome: str, trials: int = 5000) -> tuple[float, float, float]:
    daily = frame.groupby("signal_date", observed=True)[outcome].mean().dropna()
    if len(daily) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(20260729)
    values = daily.to_numpy()
    draws = rng.choice(values, size=(trials, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.05)), float(np.quantile(draws, 0.95)), float((draws <= 0).mean())


def matched_null(panel: pd.DataFrame, signals: pd.DataFrame, outcome: str, trials: int = 1000) -> dict[str, float]:
    signals = signals.dropna(subset=[outcome])
    if signals.empty:
        return {"observed": np.nan, "null_median": np.nan, "p": np.nan}
    universe = panel[
        eligible_universe(panel)
        & panel["drawdown_63"].le(MIN_DRAWDOWN_63)
        & panel[outcome].notna()
    ]
    pools = {
        key: block.set_index("ticker")[outcome]
        for key, block in universe.groupby(["asof", "sector"], observed=True)
    }
    rng = np.random.default_rng(20260729)
    null = []
    for _ in range(trials):
        selected = []
        for row in signals.itertuples():
            pool = pools.get((row.signal_date, row.sector))
            if pool is None:
                continue
            pool = pool.drop(index=row.ticker, errors="ignore").dropna()
            if not pool.empty:
                selected.append(float(pool.iloc[rng.integers(0, len(pool))]))
        if selected:
            null.append(float(np.mean(selected)))
    observed = float(signals[outcome].mean())
    null_array = np.asarray(null, dtype=float)
    return {
        "observed": observed,
        "null_median": float(np.median(null_array)) if len(null_array) else np.nan,
        "p": float((null_array >= observed).mean()) if len(null_array) else np.nan,
    }


def report(panel: pd.DataFrame, signals: pd.DataFrame, split: str) -> None:
    rows = []
    for variant in ("shock_close", "avwap_reclaim", "avwap_confirmed", "avwap_market_aligned"):
        block = signals[signals["variant"].eq(variant)]
        for sample, sample_block in (
            ("TRAIN", block[block["signal_date"].lt(split)]),
            ("TEST", block[block["signal_date"].ge(split)]),
            ("ALL", block),
        ):
            for outcome in ("fwd_5d", "fwd_21d"):
                record = {"variant": variant, "sample": sample, "outcome": outcome}
                record.update(metrics(sample_block, outcome))
                low, high, probability = clustered_interval(sample_block, outcome)
                record.update(ci05=low, ci95=high, p_nonpositive=probability)
                rows.append(record)
    table = pd.DataFrame(rows)
    display = table.copy()
    for column in ("mean", "median", "win", "ci05", "ci95", "p_nonpositive"):
        display[column] = (display[column] * 100).round(2)
    print("\n=== FIXED-HYPOTHESIS BULLISH REVERSAL EVENT STUDY ===")
    print("returns, win rates and intervals are percentages; bootstrap resamples signal days")
    print(display.to_string(index=False))

    final = signals[signals["variant"].eq("avwap_market_aligned")]
    print("\n=== MARKET-ALIGNED AVWAP SIGNAL VS SAME-DAY/SECTOR OVERSOLD PEERS ===")
    for sample, block in (
        ("TRAIN", final[final["signal_date"].lt(split)]),
        ("TEST", final[final["signal_date"].ge(split)]),
    ):
        for outcome in ("fwd_5d", "fwd_21d"):
            result = matched_null(panel, block, outcome)
            print(
                f"{sample:5s} {outcome}: observed={result['observed']:.2%} "
                f"null_median={result['null_median']:.2%} p={result['p']:.4f} n={len(block)}"
            )


def tsla_diagnostic(panel: pd.DataFrame, shocks: pd.DataFrame, signals: pd.DataFrame) -> None:
    tsla_shocks = shocks[shocks["ticker"].eq("TSLA")]
    print("\n=== TSLA POINT-IN-TIME DIAGNOSTIC ===")
    if tsla_shocks.empty:
        print("TSLA has no qualifying high-volume shock in the available panel.")
        return
    shock = tsla_shocks.iloc[-1]
    history = panel[panel["ticker"].eq("TSLA")].sort_values("asof")
    window = history[history["asof"].ge(shock["asof"])].head(MAX_RECLAIM_WAIT + 1).copy()
    weighted = window["typical_price"] * window["total_volume"]
    window["anchored_vwap"] = weighted.cumsum() / window["total_volume"].cumsum().replace(0, np.nan)
    latest = window.iloc[-1]
    event_id = f"TSLA:{shock['asof']}"
    variants = signals[signals["event_id"].eq(event_id)][["variant", "signal_date"]]
    print(
        f"shock={shock['asof']} shock_z={shock['shock_z']:.2f} drawdown63={shock['drawdown_63']:.1%} "
        f"shock_volume={shock['volume_ratio']:.2f}x post_earnings={bool(shock['post_earnings_shock'])}"
    )
    print(
        f"latest={latest['asof']} close={latest['close']:.2f} AVWAP={latest['anchored_vwap']:.2f} "
        f"prior_high={latest['prior_high']:.2f} volume={latest['volume_ratio']:.2f}x "
        f"SPY_above_SMA20={bool(latest['spy_above_sma20'])} rel2d={latest['relative_return_2d']:.2%}"
    )
    print("triggered variants:")
    print(variants.to_string(index=False) if not variants.empty else "none")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        default="/Users/anuppamvi/uw_root/tradedesk/out/research/price_panel_20260728.csv.gz",
    )
    parser.add_argument("--split", default=SPLIT_DATE)
    parser.add_argument(
        "--out",
        default="/Users/anuppamvi/uw_root/tradedesk/out/research/bull_reversal_avwap_signals.csv",
    )
    args = parser.parse_args()

    panel = prepare(pd.read_csv(args.panel, low_memory=False))
    shocks = shock_rows(panel)
    signals = event_signals(panel, shocks)
    print(
        f"panel rows={len(panel):,} sessions={panel['asof'].nunique()} tickers={panel['ticker'].nunique()} "
        f"shocks={len(shocks)} events_with_signals={signals['event_id'].nunique() if not signals.empty else 0}"
    )
    if signals.empty:
        raise SystemExit("no signals produced")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(args.out, index=False)
    report(panel, signals, args.split)
    tsla_diagnostic(panel, shocks, signals)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()