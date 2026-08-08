"""Point-in-time market and company context.

Three sources, all reproducible for historical dates:
  * SEC EDGAR daily index  - every filing, with the date it was filed
  * the UW panel itself    - earnings schedule and index levels
  * derived breadth/trend  - computed from the panel's own closes
"""

from __future__ import annotations

import gzip
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# SEC rejects requests whose User-Agent is not "Name email"; set a real contact here.
SEC_USER_AGENT = "ClaudePipeline Research admin@claudepipeline.dev"
SEC_MIN_INTERVAL = 0.5  # SEC throttles well below its published 10/s ceiling
SEC_RETRIES = 4

EDGAR_CACHE = Path("/Users/anuppamvi/tradedesk/out/claude_pipeline/edgar")
TICKER_MAP_CACHE = EDGAR_CACHE / "company_tickers.json.gz"

EVENT_FORMS = ("8-K", "10-Q", "10-K", "SC 13D", "SC 13G", "4", "S-1", "425", "DEF 14A")

_LINE = re.compile(r"\s{2,}")
_last_request = 0.0


def _fetch(url: str) -> bytes:
    global _last_request
    last_error: Exception | None = None
    for attempt in range(SEC_RETRIES):
        wait = SEC_MIN_INTERVAL - (time.time() - _last_request)
        if wait > 0:
            time.sleep(wait)
        request = urllib.request.Request(
            url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                _last_request = time.time()
                raw = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw
        except urllib.error.HTTPError as exc:
            _last_request = time.time()
            if exc.code == 404:
                raise
            last_error = exc
            time.sleep(2 ** attempt)
    raise last_error  # type: ignore[misc]


def cik_to_ticker() -> dict[int, str]:
    EDGAR_CACHE.mkdir(parents=True, exist_ok=True)
    if not TICKER_MAP_CACHE.exists():
        TICKER_MAP_CACHE.write_bytes(gzip.compress(_fetch("https://www.sec.gov/files/company_tickers.json")))
    import json

    payload = json.loads(gzip.decompress(TICKER_MAP_CACHE.read_bytes()))
    mapping: dict[int, str] = {}
    for entry in payload.values():
        mapping.setdefault(int(entry["cik_str"]), entry["ticker"])
    return mapping


def _parse_index(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        parts = _LINE.split(line.strip())
        if len(parts) < 4 or not parts[-3].isdigit() or len(parts[-2]) != 8:
            continue
        rows.append({
            "form": parts[0],
            "company": " ".join(parts[1:-3]),
            "cik": int(parts[-3]),
            "filed": parts[-2],
        })
    return pd.DataFrame(rows)


def edgar_for_date(date: str, mapping: dict[int, str] | None = None) -> pd.DataFrame:
    """Filings accepted on ``date``. Cached; weekends/holidays cache as empty."""
    EDGAR_CACHE.mkdir(parents=True, exist_ok=True)
    path = EDGAR_CACHE / f"{date}.csv.gz"
    if path.exists():
        return pd.read_csv(path, dtype={"cik": "Int64"})

    stamp = date.replace("-", "")
    quarter = f"QTR{(int(date[5:7]) - 1) // 3 + 1}"
    url = f"https://www.sec.gov/Archives/edgar/daily-index/{date[:4]}/{quarter}/form.{stamp}.idx"
    try:
        frame = _parse_index(_fetch(url).decode("latin-1"))
    except urllib.error.HTTPError as exc:
        if exc.code not in (403, 404):
            raise
        # 403 here means "no such index", the same as 404 on other SEC paths.
        frame = pd.DataFrame(columns=["form", "company", "cik", "filed"])

    mapping = mapping if mapping is not None else cik_to_ticker()
    frame["ticker"] = frame["cik"].map(mapping) if not frame.empty else pd.Series(dtype=object)
    frame["session"] = date
    frame.to_csv(path, index=False, compression="gzip")
    return frame


def backfill_edgar(sessions: list[str]) -> pd.DataFrame:
    mapping = cik_to_ticker()
    frames = [edgar_for_date(session, mapping) for session in sessions]
    return pd.concat(frames, ignore_index=True)


def filing_features(filings: pd.DataFrame) -> pd.DataFrame:
    """Per (session, ticker) counts of the filing types that move prices."""
    known = filings[filings["ticker"].notna()].copy()
    known["is_event"] = known["form"].isin(EVENT_FORMS)
    grouped = known.groupby(["session", "ticker"])
    return pd.DataFrame({
        "filings_total": grouped.size(),
        "filings_8k": grouped["form"].apply(lambda s: (s == "8-K").sum()),
        "filings_insider": grouped["form"].apply(lambda s: (s == "4").sum()),
        "filings_periodic": grouped["form"].apply(lambda s: s.isin(["10-Q", "10-K"]).sum()),
    }).reset_index()


def market_regime(panel: pd.DataFrame) -> pd.DataFrame:
    """Session-level market state, computed only from the panel's own closes."""
    def series_for(ticker: str, column: str = "close") -> pd.Series:
        rows = panel[panel["ticker"] == ticker].set_index("session")[column]
        return rows[~rows.index.duplicated(keep="last")].sort_index()

    vix = series_for("VIX")
    spx = series_for("SPX")

    equities = panel[panel["is_equity"] & (panel["dollar_volume"] >= 2e7)]
    breadth = equities.groupby("session")["day_return"].apply(lambda s: (s > 0).mean())
    dispersion = equities.groupby("session")["day_return"].std()

    regime = pd.DataFrame({
        "vix": vix,
        "vix_change": vix.pct_change(fill_method=None),
        "spx": spx,
        "spx_return_5d": spx.pct_change(5, fill_method=None),
        "spx_return_21d": spx.pct_change(21, fill_method=None),
        "breadth_up_share": breadth,
        "cross_section_dispersion": dispersion,
    })
    regime["spx_above_sma20"] = spx > spx.rolling(20, min_periods=10).mean()
    regime["trend"] = np.select(
        [regime["spx_return_21d"] > 0.02, regime["spx_return_21d"] < -0.02],
        ["uptrend", "downtrend"],
        default="range",
    )
    regime["vol_state"] = np.where(regime["vix"] > regime["vix"].rolling(60, min_periods=20).median(),
                                   "high_vol", "low_vol")
    return regime.reset_index().rename(columns={"index": "session"})
