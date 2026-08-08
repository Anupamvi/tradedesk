"""Audit equity execution costs using contemporaneous stock NBBO observations.

The dark-pool feed is the only one of the five UW files that carries stock NBBO.
Each print supplies the bid and ask prevailing at that instant. This module uses
last-hour observations on both the next-close entry session and the exit session
to replace an assumed spread with the selected names' measured quoted spread.

The data cannot measure close-auction impact or locate fees. Results therefore
include complete-case and conservative missing-quote scenarios and remain an
execution audit, not proof of attainable fills.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .equity_book import day_clustered_t


TRUEY = {"true", "1", "t", "yes"}
QUOTE_COLS = {
    "ticker",
    "executed_at",
    "nbbo_bid",
    "nbbo_ask",
    "nbbo_bid_quantity",
    "nbbo_ask_quantity",
    "canceled",
}


def _dp_source(base: Path, date: pd.Timestamp) -> Path | None:
    day = date.strftime("%Y-%m-%d")
    exact = base / day / f"dp-eod-report-{day}.zip"
    return exact if exact.exists() else None


def load_stock_nbbo(base: Path, date: pd.Timestamp,
                    tickers: set[str] | None = None) -> pd.DataFrame:
    """Return one robust stock-NBBO observation per ticker for one session."""
    source = _dp_source(base, date)
    if source is None:
        return pd.DataFrame()
    with zipfile.ZipFile(source) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with archive.open(names[0]) as stream:
            quotes = pd.read_csv(
                stream,
                usecols=lambda column: column in QUOTE_COLS,
                low_memory=False,
            )
    if quotes.empty:
        return quotes
    quotes["ticker"] = quotes["ticker"].astype(str).str.upper().str.strip()
    if tickers is not None:
        quotes = quotes[quotes["ticker"].isin(tickers)]
    if "canceled" in quotes:
        canceled = quotes["canceled"].astype(str).str.lower().isin(TRUEY)
        quotes = quotes[~canceled]
    for column in ("nbbo_bid", "nbbo_ask", "nbbo_bid_quantity", "nbbo_ask_quantity"):
        if column in quotes:
            quotes[column] = pd.to_numeric(quotes[column], errors="coerce")
    quotes = quotes[
        (quotes["nbbo_bid"] > 0)
        & (quotes["nbbo_ask"] >= quotes["nbbo_bid"])
    ].copy()
    if quotes.empty:
        return quotes

    timestamp = pd.to_datetime(quotes["executed_at"], errors="coerce", utc=True)
    quotes["local_time"] = timestamp.dt.tz_convert("America/New_York")
    quotes["half_spread_bps"] = (
        (quotes["nbbo_ask"] - quotes["nbbo_bid"])
        / (quotes["nbbo_ask"] + quotes["nbbo_bid"])
        * 10_000.0
    )
    quotes["bid_depth_dollars"] = (
        quotes.get("nbbo_bid_quantity", np.nan) * quotes["nbbo_bid"]
    )
    quotes["ask_depth_dollars"] = (
        quotes.get("nbbo_ask_quantity", np.nan) * quotes["nbbo_ask"]
    )
    local_hour = quotes["local_time"].dt.hour + quotes["local_time"].dt.minute / 60.0
    regular = quotes[(local_hour >= 9.5) & (local_hour < 16.0)]
    if regular.empty:
        return pd.DataFrame()
    late = regular[local_hour.loc[regular.index] >= 15.0]

    def aggregate(frame: pd.DataFrame, window: str) -> pd.DataFrame:
        grouped = frame.groupby("ticker", sort=False)
        result = grouped.agg(
            half_spread_bps=("half_spread_bps", "median"),
            spread_p75_bps=("half_spread_bps", lambda values: values.quantile(0.75)),
            quote_observations=("half_spread_bps", "size"),
            bid_depth_dollars=("bid_depth_dollars", "median"),
            ask_depth_dollars=("ask_depth_dollars", "median"),
        )
        result["quote_window"] = window
        return result

    all_day = aggregate(regular, "regular_session")
    if late.empty:
        result = all_day
    else:
        last_hour = aggregate(late, "last_hour")
        result = last_hour.combine_first(all_day)
        result.loc[last_hour.index, "quote_window"] = "last_hour"
    result.index.name = "ticker"
    return result.reset_index().assign(quote_date=pd.Timestamp(date))


def attach_execution_dates(legs: pd.DataFrame, sessions: list[pd.Timestamp],
                           horizon: int) -> pd.DataFrame:
    """Map a signal-session close to its next-close entry and h-session exit."""
    positions = {pd.Timestamp(date): index for index, date in enumerate(sessions)}
    out = legs.copy()

    def shifted(date, offset):
        index = positions.get(pd.Timestamp(date))
        target = None if index is None else index + offset
        return sessions[target] if target is not None and target < len(sessions) else pd.NaT

    out["signal_date"] = pd.to_datetime(out["date"])
    out["entry_date"] = out["signal_date"].map(lambda date: shifted(date, 1))
    out["exit_date"] = out["signal_date"].map(lambda date: shifted(date, 1 + horizon))
    return out


def build_quote_panel(base: Path, legs: pd.DataFrame) -> pd.DataFrame:
    """Load only the ticker-date combinations needed by the selected book."""
    required: dict[pd.Timestamp, set[str]] = {}
    for date_column in ("entry_date", "exit_date"):
        for date, frame in legs.dropna(subset=[date_column]).groupby(date_column):
            required.setdefault(pd.Timestamp(date), set()).update(frame["ticker"].unique())
    frames = []
    for date in sorted(required):
        quotes = load_stock_nbbo(base, date, required[date])
        if not quotes.empty:
            frames.append(quotes)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def price_book(legs: pd.DataFrame, quotes: pd.DataFrame, horizon: int,
               borrow_bps_annual: float = 200.0,
               extra_bps_per_side: float = 0.0,
               missing_half_spread_bps: float | None = None) -> pd.DataFrame:
    """Reprice the book with measured entry and exit half-spreads."""
    quote_columns = [
        "ticker", "quote_date", "half_spread_bps", "spread_p75_bps",
        "quote_observations", "bid_depth_dollars", "ask_depth_dollars", "quote_window",
    ]
    out = legs.copy()
    for stage in ("entry", "exit"):
        renamed = quotes[quote_columns].rename(columns={
            "quote_date": f"{stage}_date",
            **{column: f"{stage}_{column}" for column in quote_columns[2:]},
        })
        out = out.merge(renamed, on=["ticker", f"{stage}_date"], how="left")
    for stage in ("entry", "exit"):
        column = f"{stage}_half_spread_bps"
        if missing_half_spread_bps is not None:
            out[column] = out[column].fillna(missing_half_spread_bps)
    out["quote_complete"] = (
        out["entry_half_spread_bps"].notna()
        & out["exit_half_spread_bps"].notna()
    )
    out["round_trip_spread_bps"] = (
        out["entry_half_spread_bps"] + out["exit_half_spread_bps"]
    )
    out["round_trip_cost_bps"] = (
        out["round_trip_spread_bps"] + 2.0 * extra_bps_per_side
    )
    borrow = borrow_bps_annual / 10_000.0 * horizon / 252.0
    short = out["side"].eq("short")
    out["audited_net"] = np.where(
        short,
        -out["gross"] - out["round_trip_cost_bps"] / 10_000.0 - borrow,
        out["gross"] - out["round_trip_cost_bps"] / 10_000.0,
    )
    return out


def scenario_stats(frame: pd.DataFrame, name: str, total_legs: int,
                   horizon: int) -> dict:
    priced = frame[frame["audited_net"].notna()].copy()
    if priced.empty:
        return {"scenario": name, "legs": 0}
    mean, p05, p95 = day_clustered_t(priced, col="audited_net")
    return {
        "scenario": name,
        "legs": len(priced),
        "dates": priced["date"].nunique(),
        "coverage": len(priced) / total_legs,
        "net_per_leg": mean,
        "p05": p05,
        "p95": p95,
        "naive_annualized": mean * 252.0 / horizon,
        "median_round_trip_bps": priced["round_trip_spread_bps"].median(),
    }


def run_audit(legs_path: str, panel_path: str, base_dir: str, horizon: int = 3,
              borrow_bps_annual: float = 200.0,
              extra_bps_per_side: float = 0.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    legs = pd.read_csv(legs_path, parse_dates=["date"])
    panel = pd.read_pickle(panel_path)
    sessions = sorted(pd.to_datetime(panel["date"].drop_duplicates()))
    legs = attach_execution_dates(legs, sessions, horizon)
    quotes = build_quote_panel(Path(base_dir), legs)

    measured = price_book(
        legs, quotes, horizon, borrow_bps_annual, extra_bps_per_side,
    )
    observed_half_spreads = pd.concat([
        measured["entry_half_spread_bps"], measured["exit_half_spread_bps"],
    ]).dropna()
    p75 = float(observed_half_spreads.quantile(0.75))
    scenario_rows = [scenario_stats(
        measured[measured["quote_complete"]], "complete_case",
        len(legs), horizon,
    )]
    for additional_slippage in (0.0, 1.0, 2.0, 5.0):
        imputed = price_book(
            legs, quotes, horizon, borrow_bps_annual,
            extra_bps_per_side + additional_slippage,
            missing_half_spread_bps=p75,
        )
        scenario_rows.append(scenario_stats(
            imputed,
            f"p75_imputed_plus_{additional_slippage:g}bp_per_side",
            len(legs), horizon,
        ))
    scenarios = pd.DataFrame(scenario_rows)
    return measured, scenarios


def print_report(measured: pd.DataFrame, scenarios: pd.DataFrame) -> None:
    observed = pd.concat([
        measured["entry_half_spread_bps"], measured["exit_half_spread_bps"],
    ]).dropna()
    print("=== ACTUAL-PICK STOCK NBBO EXECUTION AUDIT ===")
    print(f"legs={len(measured):,}  complete entry+exit quotes="
          f"{measured['quote_complete'].sum():,} ({measured['quote_complete'].mean():.1%})")
    print(
        f"half-spread bps: median={observed.median():.2f}  "
        f"p75={observed.quantile(0.75):.2f}  p90={observed.quantile(0.90):.2f}  "
        f">5bp={(observed > 5).mean():.1%}  >10bp={(observed > 10).mean():.1%}"
    )
    print("\nSCENARIOS")
    print(f"{'scenario':<31}{'legs':>7}{'coverage':>10}{'rt bps':>9}"
          f"{'net/leg':>11}{'p05':>10}{'p95':>10}{'ann%':>10}")
    for row in scenarios.itertuples():
        print(
            f"{row.scenario:<31}{int(row.legs):>7}{row.coverage:>10.1%}"
            f"{row.median_round_trip_bps:>9.2f}{row.net_per_leg:>+11.4f}"
            f"{row.p05:>+10.4f}{row.p95:>+10.4f}{row.naive_annualized:>+10.1%}"
        )
    complete = measured[measured["quote_complete"]]
    if not complete.empty:
        print("\nWIDEST SELECTED TICKER-DATES (round trip)")
        columns = ["ticker", "signal_date", "side", "round_trip_spread_bps"]
        print(complete.nlargest(12, "round_trip_spread_bps")[columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legs", default="out/equity_book_3d_k25.csv")
    parser.add_argument("--panel", default="out/surge_panel_norm.pkl")
    parser.add_argument("--base-dir", default="/Users/anuppamvi/uw_root/tradedesk")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--borrow-bps-annual", type=float, default=200.0)
    parser.add_argument("--extra-bps-per-side", type=float, default=0.0)
    parser.add_argument("--out", default="out/equity_book_3d_execution_audit.csv")
    args = parser.parse_args()

    measured, scenarios = run_audit(
        args.legs, args.panel, args.base_dir, args.horizon,
        args.borrow_bps_annual, args.extra_bps_per_side,
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    measured.to_csv(args.out, index=False)
    scenarios.to_csv(Path(args.out).with_name(Path(args.out).stem + "_scenarios.csv"), index=False)
    print_report(measured, scenarios)


if __name__ == "__main__":
    main()