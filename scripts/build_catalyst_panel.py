"""Catalyst panel: earnings proximity and macro-event proximity per ticker-day.

Earnings come from the stock-screener file, which carries next_earnings_date for
roughly 99% of large caps every session, plus the premarket/postmarket flag.
Macro dates come from the local verified event calendar.

This is deliberately limited to catalysts that exist for the FULL history and can
therefore be backtested. Ad-hoc news and X captures are not included here: the
archive holds only a handful per ticker, so any backtest built on them would be
measuring capture luck rather than signal.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/anuppamvi/uw_root/tradedesk")
CALENDAR = Path("/Users/anuppamvi/tradedesk/knowledge/options_agent_event_calendar_2026.json")
OUT = ROOT / "out/catalyst_panel.csv"
COLUMNS = ["ticker", "next_earnings_date", "er_time", "marketcap", "issue_type"]


def screener_path(day: Path) -> Path | None:
    hits = [p for p in sorted(day.glob("stock-screener-*.zip")) if p.is_file() and day.name in p.name]
    return hits[0] if hits else None


def macro_dates() -> pd.DataFrame:
    if not CALENDAR.exists():
        return pd.DataFrame(columns=["date", "event", "impact"])
    payload = json.loads(CALENDAR.read_text(encoding="utf-8"))
    events = pd.DataFrame(payload.get("macro_events", []))
    if events.empty:
        return events
    events["date"] = pd.to_datetime(events["date"], errors="coerce")
    return events.dropna(subset=["date"]).sort_values("date")


def main() -> None:
    events = macro_dates()
    high_impact = events[events.get("impact").eq("high")]["date"].tolist() if not events.empty else []

    frames = []
    days = sorted(p for p in ROOT.iterdir() if p.is_dir() and re.fullmatch(r"2026-\d{2}-\d{2}", p.name))
    for day in days:
        path = screener_path(day)
        if path is None:
            continue
        archive = zipfile.ZipFile(path)
        member = archive.namelist()[0]
        frame = pd.read_csv(archive.open(member), usecols=COLUMNS, low_memory=False)
        frame = frame[frame.ticker.notna()].copy()
        frame["ticker"] = frame.ticker.astype(str).str.upper()
        frame["marketcap"] = pd.to_numeric(frame.marketcap, errors="coerce")
        session = pd.Timestamp(day.name)
        earnings = pd.to_datetime(frame.next_earnings_date, errors="coerce")
        frame["days_to_earnings"] = (earnings - session).dt.days
        frame["date"] = day.name
        if high_impact:
            future = [d for d in high_impact if d >= session]
            frame["days_to_macro_event"] = (future[0] - session).days if future else np.nan
            frame["next_macro_event"] = (
                events.loc[events.date.eq(future[0]), "event"].iloc[0] if future else ""
            )
        else:
            frame["days_to_macro_event"] = np.nan
            frame["next_macro_event"] = ""
        frames.append(
            frame[
                [
                    "date", "ticker", "next_earnings_date", "er_time",
                    "days_to_earnings", "days_to_macro_event", "next_macro_event",
                    "issue_type", "marketcap",
                ]
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(["date", "ticker"])
    result.to_csv(OUT, index=False)
    large = result[(result.issue_type == "Common Stock") & (result.marketcap.fillna(0) >= 2e9)]
    print(f"days={result.date.nunique()} rows={len(result)} -> {OUT}")
    print(f"large-cap rows={len(large)} earnings coverage={large.days_to_earnings.notna().mean():.3f}")
    print("\nearnings proximity buckets (large caps):")
    buckets = pd.cut(large.days_to_earnings, [-999, -1, 0, 3, 7, 14, 30, 999])
    print(buckets.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
