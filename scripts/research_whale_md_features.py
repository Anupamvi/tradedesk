"""Recover per-ticker whale-flow features from legacy ``whale-<date>.md`` summaries.

Why this exists
---------------
The raw ``bot-eod-report`` zips (full option tape) are only retained locally for
2026-04-23 onward.  For Jan-Apr the raw zip was deleted after processing, but the
generated ``whale-<date>.md`` summary survived -- and that summary was produced
*from* the full tape ("Total rows scanned: 7,799,090").

Those summaries carry two machine-readable pipe tables:

1. ``Top Symbols by Total Premium (Yes-Prime)``
   -> underlying_symbol, count, total_premium
2. ``Top 200 Yes-Prime Trades by Premium``
   -> per-trade rows with side, option_type, net_type, delta, IV, dte, size,
      premium, open_interest, pct_width

That is enough to rebuild per-ticker whale-flow aggregates for ~56 extra days,
which is what lets flow features straddle the 2026-05-01 train/held-out split
instead of living entirely inside the held-out window.

IMPORTANT CAVEAT (do not lose this):
    These are ``Yes-Prime`` rows only -- a rulebook-filtered ~0.01% of the tape
    (min premium $25k, min OI 100, DTE bands, ETF/INDEX excluded).  They are
    *whale-grade* flow, NOT all flow.  Features derived here are therefore not
    numerically interchangeable with full-tape ``bot_*`` features.  Always carry
    the ``flow_src`` tier flag and calibrate on the overlap days before pooling.

Output: out/research/whale_md_features.csv.gz
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

DEFAULT_ROOT = "/Users/anuppamvi/uw_root/tradedesk"

_DATE_RE = re.compile(r"(20\d\d-\d\d-\d\d)")
_SCANNED_RE = re.compile(r"Total rows scanned:\s*([0-9,]+)")
_YESPRIME_RE = re.compile(r"Yes-Prime candidates:\s*([0-9,]+)")
_SOURCE_RE = re.compile(r"bot-eod-report-(20\d\d-\d\d-\d\d)")

SYMBOL_TABLE = "Top Symbols by Total Premium"
TRADE_TABLE = "Top 200 Yes-Prime Trades"


def _to_num(value: str) -> float:
    """Parse a markdown cell into a float, tolerating commas and sci-notation."""
    text = value.strip().replace(",", "")
    if not text or text in {"-", "nan", "None"}:
        return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def _parse_pipe_table(lines: List[str], start: int) -> pd.DataFrame:
    """Parse the markdown pipe table that begins at/after ``start``."""
    idx = start
    while idx < len(lines) and not lines[idx].lstrip().startswith("|"):
        idx += 1
    if idx >= len(lines):
        return pd.DataFrame()

    header = [c.strip() for c in lines[idx].strip().strip("|").split("|")]
    idx += 1
    # alignment row (|:---|---:|)
    if idx < len(lines) and set(lines[idx].strip()) <= set("|:- "):
        idx += 1

    rows: List[List[str]] = []
    while idx < len(lines) and lines[idx].lstrip().startswith("|"):
        cells = [c.strip() for c in lines[idx].strip().strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
        idx += 1

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=header)


def _find_section(lines: List[str], needle: str) -> Optional[int]:
    for i, line in enumerate(lines):
        if line.startswith("#") and needle in line:
            return i
    return None


def _safe_div(num, den):
    den = np.where(np.abs(den) < 1e-12, np.nan, den)
    return num / den


def parse_whale_md(path: str, asof: str) -> Optional[pd.DataFrame]:
    """Return per-ticker whale-flow features for one ``whale-<date>.md``."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    lines = text.splitlines()

    # --- look-ahead guard -------------------------------------------------
    # The summary must have been generated from THIS session's tape.  If the
    # embedded source path names a different date, drop the file rather than
    # silently importing next-day information.
    src = _SOURCE_RE.search(text)
    if src and src.group(1) != asof:
        print(f"  !! {asof}: source date {src.group(1)} != folder date -- SKIP (look-ahead)")
        return None

    scanned = _SCANNED_RE.search(text)
    yesprime = _YESPRIME_RE.search(text)
    day_scanned = _to_num(scanned.group(1)) if scanned else np.nan
    day_yesprime = _to_num(yesprime.group(1)) if yesprime else np.nan

    # --- per-trade table (richest) ---------------------------------------
    trades = pd.DataFrame()
    pos = _find_section(lines, TRADE_TABLE)
    if pos is not None:
        trades = _parse_pipe_table(lines, pos + 1)

    # --- per-symbol premium table ----------------------------------------
    symbols = pd.DataFrame()
    pos = _find_section(lines, SYMBOL_TABLE)
    if pos is not None:
        symbols = _parse_pipe_table(lines, pos + 1)

    if trades.empty and symbols.empty:
        return None

    frames: List[pd.DataFrame] = []

    if not trades.empty and "underlying_symbol" in trades.columns:
        t = trades.copy()
        t["ticker"] = t["underlying_symbol"].str.upper().str.strip()
        for col in ("premium", "size", "delta", "implied_volatility", "dte",
                    "open_interest", "pct_width", "strike", "underlying_price"):
            if col in t.columns:
                t[col] = t[col].map(_to_num)

        side = t.get("side", pd.Series(index=t.index, dtype=object)).str.lower()
        otype = t.get("option_type", pd.Series(index=t.index, dtype=object)).str.lower()
        ntype = t.get("net_type", pd.Series(index=t.index, dtype=object)).str.lower()
        track = t.get("track", pd.Series(index=t.index, dtype=object)).str.upper()

        # UW convention: CALL@ASK / PUT@BID = bullish ; CALL@BID / PUT@ASK = bearish
        bullish = ((otype == "call") & (side == "ask")) | ((otype == "put") & (side == "bid"))
        bearish = ((otype == "call") & (side == "bid")) | ((otype == "put") & (side == "ask"))

        prem = t["premium"].fillna(0.0) if "premium" in t.columns else pd.Series(0.0, index=t.index)
        t["_bull_prem"] = np.where(bullish, prem, 0.0)
        t["_bear_prem"] = np.where(bearish, prem, 0.0)
        t["_aggr"] = (side.isin(["ask", "bid"])).astype(float)
        t["_credit"] = (ntype == "credit").astype(float)
        t["_shield"] = (track == "SHIELD").astype(float)
        t["_call"] = (otype == "call").astype(float)

        # Schema drifts across months: the Jan/Feb summaries omit ``delta`` and
        # ``implied_volatility``.  Aggregate only what this file actually has so
        # one missing column cannot silently drop 20 sessions.
        spec: Dict[str, tuple] = {
            "wmd_n": ("ticker", "size"),
            "wmd_bull_prem": ("_bull_prem", "sum"),
            "wmd_bear_prem": ("_bear_prem", "sum"),
            "wmd_aggr_share": ("_aggr", "mean"),
            "wmd_credit_share": ("_credit", "mean"),
            "wmd_shield_share": ("_shield", "mean"),
            "wmd_call_share": ("_call", "mean"),
        }
        optional = {
            "wmd_prem": ("premium", "sum"),
            "wmd_size": ("size", "sum"),
            "wmd_avg_delta": ("delta", "mean"),
            "wmd_avg_iv": ("implied_volatility", "mean"),
            "wmd_avg_dte": ("dte", "mean"),
            "wmd_avg_oi": ("open_interest", "mean"),
            "wmd_avg_pct_width": ("pct_width", "mean"),
        }
        for out_col, (src_col, how) in optional.items():
            if src_col in t.columns:
                spec[out_col] = (src_col, how)

        agg = t.groupby("ticker").agg(**spec)
        denom = agg["wmd_bull_prem"] + agg["wmd_bear_prem"]
        agg["wmd_dir_ratio"] = _safe_div(agg["wmd_bull_prem"] - agg["wmd_bear_prem"], denom)
        frames.append(agg)

    if not symbols.empty and "underlying_symbol" in symbols.columns:
        s = symbols.copy()
        s["ticker"] = s["underlying_symbol"].str.upper().str.strip()
        rename = {}
        if "count" in s.columns:
            rename["count"] = "wmd_sym_n"
        if "total_premium" in s.columns:
            rename["total_premium"] = "wmd_sym_prem"
        s = s.rename(columns=rename)
        keep = ["ticker"] + [c for c in ("wmd_sym_n", "wmd_sym_prem") if c in s.columns]
        s = s[keep]
        for col in ("wmd_sym_n", "wmd_sym_prem"):
            if col in s.columns:
                s[col] = s[col].map(_to_num)
        s = s.groupby("ticker").sum(numeric_only=True)
        frames.append(s)

    if not frames:
        return None

    out = frames[0]
    for extra in frames[1:]:
        out = out.join(extra, how="outer")
    out = out.reset_index()

    out.insert(0, "asof", asof)
    out["wmd_day_scanned"] = day_scanned
    out["wmd_day_yesprime"] = day_yesprime
    # share of the day's whale premium concentrated in this ticker
    total_prem = out["wmd_sym_prem"].sum() if "wmd_sym_prem" in out.columns else np.nan
    if total_prem and total_prem == total_prem and total_prem > 0:
        out["wmd_prem_share"] = out.get("wmd_sym_prem", np.nan) / total_prem
    else:
        out["wmd_prem_share"] = np.nan
    out["flow_src"] = "whale_md"
    return out


def build(root: str, start: str, end: str) -> pd.DataFrame:
    day_dirs = sorted(
        d for d in glob.glob(os.path.join(root, "20[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"))
        if os.path.isdir(d)
    )
    frames: List[pd.DataFrame] = []
    n_ok = n_skip = 0
    for day_dir in day_dirs:
        asof = os.path.basename(day_dir)
        if asof < start or asof > end:
            continue
        mds = sorted(glob.glob(os.path.join(day_dir, "whale-*.md")))
        if not mds:
            continue
        # prefer the file whose name carries this date
        exact = [m for m in mds if asof in os.path.basename(m)]
        path = exact[0] if exact else mds[0]
        try:
            df = parse_whale_md(path, asof)
        except Exception as exc:  # surface, never swallow
            print(f"  !! {asof}: {type(exc).__name__}: {exc}")
            n_skip += 1
            continue
        if df is None or df.empty:
            n_skip += 1
            continue
        frames.append(df)
        n_ok += 1
        print(f"  [{n_ok}] {asof} tickers={len(df)}")

    print(f"\nparsed ok={n_ok} skipped={n_skip}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    panel = build(args.root, args.start, args.end)
    if panel.empty:
        print("no whale md features built")
        return 1

    out_path = args.out or os.path.join(args.root, "out", "research", "whale_md_features.csv.gz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    panel.to_csv(out_path, index=False, compression="gzip")

    print(f"\nrows={len(panel):,} cols={panel.shape[1]} "
          f"days={panel['asof'].nunique()} tickers={panel['ticker'].nunique()}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
