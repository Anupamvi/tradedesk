"""Extract full forward mark paths for replay candidates.

The production replay stops at the first management trigger inside a fixed
5-session horizon, so it cannot answer "what would a different exit policy have
earned?".  This script re-walks the historical option quote store and records the
spread value for every session from entry until expiry (bounded by
``--max-sessions``), writing a long-format path table.

Exit policies can then be swept offline against that table without re-running the
replay.  Extraction is purely observational: it records what the market did and
applies no selection or policy logic.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from uwos.options_agent import core
from uwos.options_agent.replay import (
    HistoricalOptionQuoteStore,
    _bounded_exit_market,
    _date,
    _exit_value,
    _number,
    _quote_index,
    _spread_quotes,
)

PATH_COLUMNS = [
    "row_id",
    "asof",
    "ticker",
    "strategy_route",
    "entry_side",
    "entry_price",
    "entry_width",
    "expiry",
    "entry_dte",
    "session",
    "obs_date",
    "value",
]


def _forward_dates(entry_day: dt.date, last_day: dt.date, max_sessions: int) -> list[dt.date]:
    dates: list[dt.date] = []
    current = entry_day
    for _ in range(max_sessions):
        current = core._add_regular_market_days(current, 1)
        if current > last_day:
            break
        dates.append(current)
    return dates


def extract(detail_path: Path, root: Path, max_sessions: int, end: dt.date) -> pd.DataFrame:
    detail = pd.read_csv(detail_path, low_memory=False)
    fillable = detail["exact_fillable"].map(core._truthy) & detail[
        "next_session_reprice_approved"
    ].map(core._truthy)
    work = detail[fillable].copy()
    work["row_id"] = work.index

    plans: list[dict] = []
    due_symbols: dict[dt.date, set[str]] = {}
    for _, row in work.iterrows():
        entry_day = _date(row.get("planned_entry_date"))
        expiry = _date(row.get("expiry"))
        entry = _number(row.get("executed_entry_price"))
        width = _number(row.get("entry_width"))
        short_leg = str(row.get("short_leg_eod") or "").upper()
        long_leg = str(row.get("long_leg_eod") or "").upper()
        if entry_day is None or expiry is None or entry is None or not width:
            continue
        if not short_leg or not long_leg:
            continue
        last_day = min(expiry, end)
        observations = [d for d in _forward_dates(entry_day, last_day, max_sessions)]
        if not observations:
            continue
        for due in observations:
            due_symbols.setdefault(due, set()).update({short_leg, long_leg})
        plans.append(
            {
                "row_id": int(row["row_id"]),
                "asof": row.get("asof"),
                "ticker": row.get("ticker"),
                "strategy_route": row.get("strategy_route"),
                "entry_side": str(row.get("entry_side") or "").upper(),
                "entry_price": entry,
                "entry_width": width,
                "expiry": expiry.isoformat(),
                "entry_dte": _number(row.get("dte")),
                "short_leg": short_leg,
                "long_leg": long_leg,
                "observations": observations,
            }
        )

    print(f"plans={len(plans)} distinct_obs_dates={len(due_symbols)}", flush=True)

    quote_store = HistoricalOptionQuoteStore(root, use_hot=True, use_oi=True)
    quotes_by_date: dict[dt.date, dict] = {}
    for i, (due, symbols) in enumerate(sorted(due_symbols.items()), start=1):
        quotes_by_date[due] = _quote_index(quote_store, due, {s for s in symbols if s})
        if i % 20 == 0:
            print(f"  loaded quotes {i}/{len(due_symbols)}", flush=True)

    records: list[dict] = []
    for plan in plans:
        for session, due in enumerate(plan["observations"], start=1):
            index = quotes_by_date.get(due, {})
            short_quote = index.get(plan["short_leg"])
            long_quote = index.get(plan["long_leg"])
            if short_quote is None or long_quote is None:
                continue
            raw_bid, raw_ask, _ = _spread_quotes(plan["entry_side"], short_quote, long_quote)
            bounded = _bounded_exit_market(raw_bid, raw_ask, plan["entry_width"])
            if bounded is None:
                continue
            bid, ask = bounded
            records.append(
                {
                    "row_id": plan["row_id"],
                    "asof": plan["asof"],
                    "ticker": plan["ticker"],
                    "strategy_route": plan["strategy_route"],
                    "entry_side": plan["entry_side"],
                    "entry_price": plan["entry_price"],
                    "entry_width": plan["entry_width"],
                    "expiry": plan["expiry"],
                    "entry_dte": plan["entry_dte"],
                    "session": session,
                    "obs_date": due.isoformat(),
                    "value": round(_exit_value(plan["entry_side"], bid, ask), 6),
                }
            )
    return pd.DataFrame(records, columns=PATH_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--detail",
        type=Path,
        default=Path(
            "out/options_agent_independent_replay/"
            "v1_56_live_selector_dte_parity_ytd_full/options_agent_replay_detail.csv"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("out/research/exit_paths.csv"))
    parser.add_argument("--max-sessions", type=int, default=30)
    parser.add_argument("--end", type=str, default="2026-07-23")
    args = parser.parse_args()

    end = dt.date.fromisoformat(args.end)
    frame = extract(args.detail, args.root, args.max_sessions, end)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\nwrote {args.out} rows={len(frame)} trades={frame['row_id'].nunique()}")
    if not frame.empty:
        print("sessions per trade:", frame.groupby("row_id").size().describe().round(1).to_dict())


if __name__ == "__main__":
    main()
