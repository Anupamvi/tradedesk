#!/usr/bin/env python3
"""
strategy_engine.py
------------------
Deterministic "Robot" layer for the Anu Serious Options Framework.

Goal:
- Ingest your 5 EOD files (Hot Chains, OI Changes, Dark Pool EOD, Stock Screener, Whale Trades)
- Apply HARD gates deterministically (no hallucinations)
- Generate a short list of mathematically valid candidates ("shortlist")
- Produce a reject log (auditability)
- Option pricing: prefer WEB chain snapshot (Yahoo via yfinance). If unavailable, fallback to UnusualWhales bid/ask as *indicative*.

IMPORTANT:
- This engine does NOT try to be "smart" about news, sector rotations, or tape interpretation.
  That's the AI/Analyst layer. This script is the rule enforcer + math sieve.

Usage (example):
  python -m uwos.strategy_engine --date 2026-01-23 --input-dir ./EOD --out-dir ./out

Requirements:
  pip install pandas numpy pyyaml yfinance

Notes on Pricing (your preference):
  - Pricing mode C: WEB snapshot (yfinance/Yahoo)
  - Fallback mode B: UnusualWhales (Hot Chains + OI Changes) as indicative
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml


OCC_RE = re.compile(r"^([A-Z\.]{1,10})(\d{6})([CP])(\d{8})$")


def parse_occ_symbol(sym: str) -> Optional[Tuple[str, dt.date, str, float]]:
    """
    Parse OCC-style symbol used by UnusualWhales/Yahoo:
      e.g. NVDA260130P00165000 -> (NVDA, 2026-01-30, 'P', 165.0)
    """
    m = OCC_RE.match(sym)
    if not m:
        return None
    root, yymmdd, right, strike8 = m.groups()
    yy = int("20" + yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    expiry = dt.date(yy, mm, dd)
    strike = int(strike8) / 1000.0
    return root, expiry, right, strike


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def read_csv_maybe_zipped(path: Path) -> pd.DataFrame:
    """
    Reads:
      - a CSV file directly, OR
      - a ZIP containing a single CSV.
    """
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV found inside: {path}")
            if len(names) > 1:
                # pick the first CSV deterministically
                names = sorted(names)
            with zf.open(names[0]) as f:
                return pd.read_csv(f, low_memory=False)
    raise ValueError(f"Unsupported file type: {path}")


def _is_md_alignment_line(line: str) -> bool:
    trimmed = line.strip().strip("|")
    if not trimmed:
        return False
    cleaned = trimmed.replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
    return cleaned == ""


def read_markdown_table(path: Path, heading_substr: str) -> pd.DataFrame:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    start = None
    for i, line in enumerate(lines):
        if heading_substr in line:
            start = i
            break

    if start is None:
        return pd.DataFrame()

    table_lines = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if not stripped:
            if table_lines:
                break
            continue
        if stripped.startswith("|"):
            table_lines.append(stripped)
        elif table_lines:
            break

    if not table_lines:
        return pd.DataFrame()

    header = [c.strip() for c in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[1:]:
        if _is_md_alignment_line(line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < len(header):
            cells += [""] * (len(header) - len(cells))
        rows.append(cells[:len(header)])

    df = pd.DataFrame(rows, columns=header)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def infer_date_from_filenames(input_dir: Path) -> Optional[str]:
    """
    Try to infer YYYY-MM-DD date from filenames present in input_dir.
    """
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
    for p in input_dir.iterdir():
        m = pattern.search(p.name)
        if m:
            return m.group(1)
    return None


def load_eod_bundle(input_dir: Path, date_str: str) -> Dict[str, pd.DataFrame]:
    """
    Load the five EOD files for date_str.
    Expected filenames (flexible):
      - hot-chains-{date}.zip
      - chain-oi-changes-{date}.zip
      - dp-eod-report-{date}.zip
      - stock-screener-{date}.zip
      - whale-{date}.md (Top 200 Yes-Prime table), OR whale_trades_filtered-{date}.csv
    """
    def find_one(glob_pat: str) -> Path:
        matches = sorted(input_dir.glob(glob_pat))
        if not matches:
            raise FileNotFoundError(f"Missing file pattern: {glob_pat} in {input_dir}")
        return matches[0]

    files = {
        "hot": find_one(f"hot-chains-{date_str}.*"),
        "oi": find_one(f"chain-oi-changes-{date_str}.*"),
        "dp": find_one(f"dp-eod-report-{date_str}.*"),
        "screener": find_one(f"stock-screener-{date_str}.*"),
    }
    # whale file can be a Yes-Prime markdown summary or a CSV
    whale_md_matches = sorted(input_dir.glob(f"whale-{date_str}.md"))
    whale_csv_matches = sorted(input_dir.glob(f"whale_trades_filtered-{date_str}*.csv")) or \
                        sorted(input_dir.glob("whale_trades_filtered*.csv"))

    if whale_md_matches:
        files["whale"] = whale_md_matches[0]
        dfs = {k: read_csv_maybe_zipped(v) for k, v in files.items() if k != "whale"}
        whale_df = read_markdown_table(files["whale"], "Top 200 Yes-Prime Trades by Premium")
        if whale_df.empty:
            raise FileNotFoundError(f"No whale table found in {files['whale']}")
        dfs["whale"] = whale_df
        return dfs

    if not whale_csv_matches:
        raise FileNotFoundError(f"Missing whale trades file in {input_dir}")
    files["whale"] = whale_csv_matches[0]

    dfs = {k: read_csv_maybe_zipped(v) for k, v in files.items()}
    return dfs


@dataclass
class Quote:
    bid: float
    ask: float
    last: float = np.nan
    volume: float = np.nan
    oi: float = np.nan
    iv: float = np.nan
    src: str = ""  # 'web' or 'uw'


class PricingProvider:
    def get_chain(self, ticker: str, expiry: dt.date) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def get_quote(self, option_symbol: str) -> Optional[Quote]:
        raise NotImplementedError

    def get_expirations(self, ticker: str) -> List[dt.date]:
        raise NotImplementedError

    def get_strikes(self, ticker: str, expiry: dt.date, right: str) -> np.ndarray:
        return np.array([])


class WebPricingProvider(PricingProvider):
    """
    WEB pricing via yfinance (Yahoo options chain).
    """
    def __init__(self):
        try:
            import yfinance as yf  # type: ignore
        except Exception as e:
            raise ImportError("yfinance is required for web pricing. pip install yfinance") from e
        self.yf = yf
        self._exp_cache: Dict[str, List[dt.date]] = {}
        self._chain_cache: Dict[Tuple[str, dt.date], Tuple[pd.DataFrame, pd.DataFrame]] = {}

    def get_expirations(self, ticker: str) -> List[dt.date]:
        if ticker in self._exp_cache:
            return self._exp_cache[ticker]
        exps = []
        try:
            t = self.yf.Ticker(ticker)
            opts = getattr(t, "options", []) or []
            for s in opts:
                try:
                    exps.append(dt.datetime.strptime(s, "%Y-%m-%d").date())
                except Exception:
                    continue
        except Exception:
            exps = []
        exps = sorted(set(exps))
        self._exp_cache[ticker] = exps
        return exps

    def get_chain(self, ticker: str, expiry: dt.date) -> Tuple[pd.DataFrame, pd.DataFrame]:
        key = (ticker, expiry)
        if key in self._chain_cache:
            return self._chain_cache[key]
        t = self.yf.Ticker(ticker)
        chain = t.option_chain(expiry.strftime("%Y-%m-%d"))
        calls = chain.calls.copy()
        puts = chain.puts.copy()
        self._chain_cache[key] = (calls, puts)
        return calls, puts

    def get_quote(self, option_symbol: str) -> Optional[Quote]:
        parsed = parse_occ_symbol(option_symbol)
        if not parsed:
            return None
        ticker, expiry, right, strike = parsed
        try:
            calls, puts = self.get_chain(ticker, expiry)
        except Exception:
            return None
        df = calls if right == "C" else puts
        if "contractSymbol" in df.columns:
            row = df.loc[df["contractSymbol"] == option_symbol]
            if row.empty:
                # fallback: match by strike (can have multiple; take first)
                row = df.loc[df["strike"] == strike]
        else:
            row = df.loc[df["strike"] == strike]
        if row.empty:
            return None
        r = row.iloc[0]
        bid = safe_float(r.get("bid", np.nan))
        ask = safe_float(r.get("ask", np.nan))
        last = safe_float(r.get("lastPrice", np.nan))
        vol = safe_float(r.get("volume", np.nan))
        oi = safe_float(r.get("openInterest", np.nan))
        iv = safe_float(r.get("impliedVolatility", np.nan))
        if not np.isfinite(bid) or not np.isfinite(ask) or ask <= 0:
            return None
        return Quote(bid=bid, ask=ask, last=last, volume=vol, oi=oi, iv=iv, src="web")


class UWPricingProvider(PricingProvider):
    """
    Pricing from UnusualWhales EOD files (Hot Chains + OI Changes).
    This is *indicative* pricing only. Use as fallback when web pricing fails.
    """
    def __init__(self, hot_df: pd.DataFrame, oi_df: pd.DataFrame):
        self.hot = hot_df.copy()
        self.oi = oi_df.copy()
        self._exp_cache: Dict[str, List[dt.date]] = {}
        self._quote_map: Dict[str, Quote] = {}
        self._strike_cache: Dict[Tuple[str, dt.date, str], np.ndarray] = {}
        self._build_quote_map()

    def _build_quote_map(self):
        # Hot chains quotes
        if "option_symbol" in self.hot.columns:
            for _, r in self.hot.iterrows():
                sym = r.get("option_symbol")
                if not isinstance(sym, str):
                    continue
                bid = safe_float(r.get("bid", np.nan))
                ask = safe_float(r.get("ask", np.nan))
                if np.isfinite(bid) and np.isfinite(ask) and ask > 0:
                    self._quote_map[sym] = Quote(bid=bid, ask=ask, last=safe_float(r.get("close", np.nan)),
                                                 volume=safe_float(r.get("volume", np.nan)),
                                                 oi=safe_float(r.get("open_interest", np.nan)),
                                                 iv=safe_float(r.get("iv", np.nan)),
                                                 src="uw_hot")
        # OI changes quotes (use last_bid/last_ask)
        if "option_symbol" in self.oi.columns:
            for _, r in self.oi.iterrows():
                sym = r.get("option_symbol")
                if not isinstance(sym, str):
                    continue
                if sym in self._quote_map:
                    continue  # prefer hot-chains when available
                bid = safe_float(r.get("last_bid", np.nan))
                ask = safe_float(r.get("last_ask", np.nan))
                if np.isfinite(bid) and np.isfinite(ask) and ask > 0:
                    self._quote_map[sym] = Quote(bid=bid, ask=ask, last=safe_float(r.get("last_fill", np.nan)),
                                                 volume=safe_float(r.get("volume", np.nan)),
                                                 oi=safe_float(r.get("curr_oi", np.nan)),
                                                 iv=np.nan,
                                                 src="uw_oi")

        # Build strike cache from parsed symbols
        strike_bins: Dict[Tuple[str, dt.date, str], List[float]] = {}
        for sym in self._quote_map.keys():
            parsed = parse_occ_symbol(sym)
            if not parsed:
                continue
            root, expiry, right, strike = parsed
            key = (root, expiry, right)
            strike_bins.setdefault(key, []).append(strike)
        for key, vals in strike_bins.items():
            self._strike_cache[key] = np.array(sorted(set(vals)))

    def get_quote(self, option_symbol: str) -> Optional[Quote]:
        return self._quote_map.get(option_symbol)

    def get_expirations(self, ticker: str) -> List[dt.date]:
        if ticker in self._exp_cache:
            return self._exp_cache[ticker]
        exps = set()
        for sym in self._quote_map.keys():
            parsed = parse_occ_symbol(sym)
            if not parsed:
                continue
            root, expiry, _, _ = parsed
            if root == ticker:
                exps.add(expiry)
        exps = sorted(exps)
        self._exp_cache[ticker] = exps
        return exps

    def get_chain(self, ticker: str, expiry: dt.date) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        UW provider cannot provide a full chain; return empty frames.
        """
        return pd.DataFrame(), pd.DataFrame()

    def get_strikes(self, ticker: str, expiry: dt.date, right: str) -> np.ndarray:
        key = (ticker, expiry, right)
        return self._strike_cache.get(key, np.array([]))


@dataclass
class TradeCandidate:
    date: str
    ticker: str
    track: str            # FIRE / SHIELD
    bias: str             # BULL / BEAR
    strategy: str         # e.g. Bull Put Credit
    expiry: str           # YYYY-MM-DD
    dte: int
    short_leg: str        # OCC symbol
    long_leg: str         # OCC symbol
    width: float
    entry_gate: str       # e.g. ">= 5.00cr" or "<= 3.20db"
    net: float            # computed with conservative bid/ask (if available)
    net_type: str         # "credit" or "debit"
    credit_pct_width: float
    max_profit: float
    max_loss: float
    breakeven: float
    spot: float
    earnings_date: str
    pricing_src: str
    conviction: int
    reasons: str          # short signal notes


def price_tier_width_config(cfg: dict, spot: float) -> Tuple[float, float, float]:
    """
    Returns (min_width, max_width, default_width) for given spot.
    """
    spot = safe_float(spot, default=np.nan)
    if not np.isfinite(spot):
        return 10.0, 20.0, 15.0
    tiers = cfg["gates"]["width_tiers"]
    for t in tiers:
        lo = safe_float(t.get("min_price", -math.inf), default=-math.inf)
        hi = safe_float(t.get("max_price", math.inf), default=math.inf)
        if spot >= lo and spot < hi:
            min_w = safe_float(t.get("min_width"), default=np.nan)
            max_w = safe_float(t.get("max_width"), default=np.nan)
            if not np.isfinite(min_w) or not np.isfinite(max_w):
                return 10.0, 20.0, 15.0
            default_w = t.get("default_width", (min_w + max_w) / 2)
            default_w = safe_float(default_w, default=(min_w + max_w) / 2)
            return float(min_w), float(max_w), float(default_w)
    # fallback
    return 10.0, 20.0, 15.0


def round_to_available_strike(target: float, strikes: np.ndarray, side: str) -> Optional[float]:
    """
    Choose an available strike around target.
    side:
      - 'down': choose <= target (more conservative for puts)
      - 'up': choose >= target (more conservative for calls)
      - 'nearest': closest
    """
    if strikes.size == 0:
        return None
    if side == "down":
        s = strikes[strikes <= target]
        if s.size == 0:
            return None
        return float(s.max())
    if side == "up":
        s = strikes[strikes >= target]
        if s.size == 0:
            return None
        return float(s.min())
    # nearest
    idx = int(np.argmin(np.abs(strikes - target)))
    return float(strikes[idx])


def nearest_strike_candidates(strikes: np.ndarray, target: float, limit: int) -> List[float]:
    """
    Return up to `limit` available strikes nearest to target.
    This keeps the search deterministic while allowing both ITM/OTM exploration.
    """
    if strikes is None or strikes.size == 0:
        return []
    vals = sorted({float(s) for s in strikes if np.isfinite(s)})
    if not vals:
        return []
    k = max(1, int(limit))
    ranked = sorted(vals, key=lambda s: (abs(s - target), s))
    return ranked[:k]


def build_occ_symbol(ticker: str, expiry: dt.date, right: str, strike: float) -> str:
    """
    Construct OCC symbol used by UW/Yahoo:
      TICKER + YYMMDD + C/P + STRIKE*1000 padded to 8 digits
    """
    yymmdd = expiry.strftime("%y%m%d")
    strike8 = f"{int(round(strike * 1000)):08d}"
    return f"{ticker}{yymmdd}{right}{strike8}"


def compute_credit(short_q: Quote, long_q: Quote) -> Optional[float]:
    if not short_q or not long_q:
        return None
    if not np.isfinite(short_q.bid) or not np.isfinite(long_q.ask):
        return None
    return float(short_q.bid - long_q.ask)


def compute_debit(long_q: Quote, short_q: Quote) -> Optional[float]:
    if not long_q or not short_q:
        return None
    if not np.isfinite(long_q.ask) or not np.isfinite(short_q.bid):
        return None
    return float(long_q.ask - short_q.bid)


def compute_conviction(base_score: float, penalties: float = 0.0) -> int:
    x = max(0.0, min(100.0, base_score - penalties))
    return int(round(x))


def pct(x: float) -> float:
    return float(x) * 100.0


def build_signals(dfs: Dict[str, pd.DataFrame], cfg: dict) -> pd.DataFrame:
    """
    Build per-ticker signal summary from the 5 EOD sources.
    """
    hot = dfs["hot"].copy()
    oi = dfs["oi"].copy()
    dp = dfs["dp"].copy()
    screener = dfs["screener"].copy()
    whale = dfs["whale"].copy()

    # Screener basics
    screener_cols = ["ticker", "close", "iv_rank", "implied_move_perc", "bullish_premium", "bearish_premium",
                     "call_premium", "put_premium", "put_call_ratio", "next_earnings_date", "issue_type", "marketcap", "sector"]
    for c in screener_cols:
        if c not in screener.columns:
            screener[c] = np.nan
    sc = screener[screener_cols].copy()
    sc.rename(columns={"ticker": "symbol"}, inplace=True)

    # Hot-chains: extract underlying and call/put
    if "option_symbol" in hot.columns:
        extracted = hot["option_symbol"].astype(str).str.extract(r"^([A-Z\.]{1,10})\d{6}([CP])\d{8}$")
        hot["symbol"] = extracted[0]
        hot["right"] = extracted[1]
    else:
        hot["symbol"] = np.nan
        hot["right"] = np.nan

    hot["premium"] = pd.to_numeric(hot.get("premium", 0), errors="coerce").fillna(0.0)
    hot["ask_side_volume"] = pd.to_numeric(hot.get("ask_side_volume", 0), errors="coerce").fillna(0.0)
    hot["bid_side_volume"] = pd.to_numeric(hot.get("bid_side_volume", 0), errors="coerce").fillna(0.0)
    hot["sweep_volume"] = pd.to_numeric(hot.get("sweep_volume", 0), errors="coerce").fillna(0.0)
    hot["volume"] = pd.to_numeric(hot.get("volume", 0), errors="coerce").fillna(0.0)

    hot_agg = hot.groupby("symbol", dropna=True).agg(
        hot_premium=("premium", "sum"),
        hot_call_premium=("premium", lambda s: s[hot.loc[s.index, "right"] == "C"].sum()),
        hot_put_premium=("premium", lambda s: s[hot.loc[s.index, "right"] == "P"].sum()),
        hot_sweep=("sweep_volume", "sum"),
        hot_vol=("volume", "sum"),
        hot_ask=("ask_side_volume", "sum"),
        hot_bid=("bid_side_volume", "sum"),
        hot_issue_type=("issue_type", lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan),
    ).reset_index().rename(columns={"symbol": "symbol"})

    hot_agg["hot_ask_ratio"] = np.where((hot_agg["hot_ask"] + hot_agg["hot_bid"]) > 0,
                                       hot_agg["hot_ask"] / (hot_agg["hot_ask"] + hot_agg["hot_bid"]), np.nan)

    # OI changes: already has underlying_symbol, stock_price, next_earnings_date
    oi["symbol"] = oi.get("underlying_symbol")
    oi["oi_abs"] = pd.to_numeric(oi.get("oi_diff_plain", 0), errors="coerce").fillna(0.0).abs()
    oi["oi_signed"] = pd.to_numeric(oi.get("oi_diff_plain", 0), errors="coerce").fillna(0.0)

    # classify call/put from option_symbol
    extracted2 = oi["option_symbol"].astype(str).str.extract(r"^\D+(\d{6})([CP])(\d{8})$")
    oi["right"] = extracted2[1]
    oi["prev_ask_volume"] = pd.to_numeric(oi.get("prev_ask_volume", 0), errors="coerce").fillna(0.0)
    oi["prev_bid_volume"] = pd.to_numeric(oi.get("prev_bid_volume", 0), errors="coerce").fillna(0.0)

    # heuristics: call OI increases with ask dominance = bullish; put OI increases with bid dominance = bullish (put selling)
    oi["bullish_oi_hint"] = 0.0
    oi.loc[(oi["right"] == "C") & (oi["oi_signed"] > 0) & (oi["prev_ask_volume"] > oi["prev_bid_volume"]), "bullish_oi_hint"] = oi["oi_abs"]
    oi.loc[(oi["right"] == "P") & (oi["oi_signed"] > 0) & (oi["prev_bid_volume"] > oi["prev_ask_volume"]), "bullish_oi_hint"] = oi["oi_abs"]

    oi["bearish_oi_hint"] = 0.0
    oi.loc[(oi["right"] == "P") & (oi["oi_signed"] > 0) & (oi["prev_ask_volume"] > oi["prev_bid_volume"]), "bearish_oi_hint"] = oi["oi_abs"]
    oi.loc[(oi["right"] == "C") & (oi["oi_signed"] > 0) & (oi["prev_bid_volume"] > oi["prev_ask_volume"]), "bearish_oi_hint"] = oi["oi_abs"]

    oi_agg = oi.groupby("symbol", dropna=True).agg(
        oi_abs=("oi_abs", "sum"),
        oi_bull_hint=("bullish_oi_hint", "sum"),
        oi_bear_hint=("bearish_oi_hint", "sum"),
        oi_max_abs=("oi_abs", "max"),
        oi_stock_price=("stock_price", "max"),
        oi_earnings=("next_earnings_date", lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan),
    ).reset_index()

    # Whale trades
    whale["symbol"] = whale.get("underlying_symbol")
    whale["premium"] = pd.to_numeric(whale.get("premium", 0), errors="coerce").fillna(0.0)
    whale["call_prem"] = np.where(whale.get("option_type") == "call", whale["premium"], 0.0)
    whale["put_prem"] = np.where(whale.get("option_type") == "put", whale["premium"], 0.0)
    whale["equity_type"] = whale.get("equity_type")

    whale_agg = whale.groupby("symbol", dropna=True).agg(
        whale_premium=("premium", "sum"),
        whale_call_premium=("call_prem", "sum"),
        whale_put_premium=("put_prem", "sum"),
        whale_equity_type=("equity_type", lambda s: s.dropna().iloc[0] if len(s.dropna()) else np.nan),
    ).reset_index()

    # Dark pool aggregation
    dp["symbol"] = dp.get("ticker")
    dp["premium"] = pd.to_numeric(dp.get("premium", 0), errors="coerce").fillna(0.0)
    dp["size"] = pd.to_numeric(dp.get("size", 0), errors="coerce").fillna(0.0)
    dp["price"] = pd.to_numeric(dp.get("price", np.nan), errors="coerce")

    # compute above-mid %
    dp["mid"] = (pd.to_numeric(dp.get("nbbo_ask", np.nan), errors="coerce") + pd.to_numeric(dp.get("nbbo_bid", np.nan), errors="coerce")) / 2
    dp["above_mid"] = np.where(np.isfinite(dp["price"]) & np.isfinite(dp["mid"]) & (dp["price"] > dp["mid"]), 1.0, 0.0)

    dp_agg = dp.groupby("symbol", dropna=True).agg(
        dp_premium=("premium", "sum"),
        dp_above_mid_pct=("above_mid", "mean"),
        dp_vwap=("price", lambda s: np.nan if s.isna().all() else float(np.nansum(s * dp.loc[s.index, "size"]) / max(1.0, np.nansum(dp.loc[s.index, "size"])))),
    ).reset_index()

    # Merge all
    out = sc.merge(hot_agg, on="symbol", how="outer").merge(oi_agg, on="symbol", how="outer").merge(whale_agg, on="symbol", how="outer").merge(dp_agg, on="symbol", how="outer")

    # Fill spot price
    out["spot"] = out["close"]
    out.loc[out["spot"].isna(), "spot"] = out["oi_stock_price"]
    # best effort: some hot-chains have close.1 for underlying close
    if "close.1" in hot.columns and "symbol" in hot.columns:
        hot_under = hot.groupby("symbol")["close.1"].max().rename("hot_spot").reset_index()
        out = out.merge(hot_under, on="symbol", how="left")
        out.loc[out["spot"].isna(), "spot"] = out["hot_spot"]

    # Earnings date
    out["earnings"] = out["next_earnings_date"]
    out.loc[out["earnings"].isna(), "earnings"] = out["oi_earnings"]

    # Issue type (ETF filter)
    out["issue_type_final"] = out["issue_type"]
    out.loc[out["issue_type_final"].isna(), "issue_type_final"] = out["hot_issue_type"]
    out.loc[out["issue_type_final"].isna(), "issue_type_final"] = out["whale_equity_type"]

    # Interest score (for selecting tickers to scan)
    def log1p(x):
        return np.log10(1.0 + np.nan_to_num(x, nan=0.0))
    out["interest_score"] = log1p(out["hot_premium"]) + log1p(out["oi_abs"]) + log1p(out["whale_premium"])

    # Bull/Bear score (direction)
    out["bull_score"] = (
        (np.nan_to_num(out["bullish_premium"]) - np.nan_to_num(out["bearish_premium"])) +
        (np.nan_to_num(out["hot_call_premium"]) - np.nan_to_num(out["hot_put_premium"])) +
        (np.nan_to_num(out["whale_call_premium"]) - np.nan_to_num(out["whale_put_premium"])) +
        (np.nan_to_num(out["oi_bull_hint"]) - np.nan_to_num(out["oi_bear_hint"]))
    )
    out["bias"] = np.where(out["bull_score"] > 0, "BULL", "BEAR")

    # FIRE score: more sweeps + higher premium + ask dominance
    out["fire_score_raw"] = log1p(out["hot_premium"]) + log1p(out["hot_sweep"]) + (np.nan_to_num(out["hot_ask_ratio"]) - 0.5) * 2.0
    # normalize fire score to 0..1
    fs = out["fire_score_raw"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if fs.max() > fs.min():
        out["fire_score"] = (fs - fs.min()) / (fs.max() - fs.min())
    else:
        out["fire_score"] = 0.0

    return out


def passes_etf_gate(row: pd.Series, cfg: dict) -> bool:
    if not cfg["gates"]["exclude_etfs"]:
        return True
    issue = str(row.get("issue_type_final", "") or "")
    return issue.upper() not in {"ETF", "INDEX", "ETN"} and issue != "ETF"


def parse_date_safe(s) -> Optional[dt.date]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        if isinstance(s, dt.date):
            return s
        return dt.datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def within_dte(expiry: dt.date, asof: dt.date, lo: int, hi: int) -> bool:
    dte = (expiry - asof).days
    return dte >= lo and dte <= hi


def generate_candidates_for_ticker(
    asof: dt.date,
    row: pd.Series,
    cfg: dict,
    pricing: PricingProvider,
    fallback_pricing: Optional[PricingProvider],
) -> Tuple[List[TradeCandidate], List[dict]]:
    """
    Returns (candidates, rejects)
    """
    symbol = row["symbol"]
    spot = float(row.get("spot", np.nan))
    if not np.isfinite(spot) or spot <= 0:
        return [], [{"ticker": symbol, "reason": "missing_spot"}]

    # Must appear in sources
    appears = (
        (row.get("hot_premium", 0) or 0) > 0 or
        (row.get("oi_abs", 0) or 0) > 0 or
        (row.get("whale_premium", 0) or 0) > 0
    )
    if not appears:
        return [], [{"ticker": symbol, "reason": "not_in_hot_whale_oi"}]

    if not passes_etf_gate(row, cfg):
        return [], [{"ticker": symbol, "reason": "etf_excluded"}]

    min_w, max_w, default_w = price_tier_width_config(cfg, spot)

    bias = row.get("bias", "BULL")
    earnings = parse_date_safe(row.get("earnings"))
    iv_rank = safe_float(row.get("iv_rank", np.nan), np.nan)
    dp_above = safe_float(row.get("dp_above_mid_pct", np.nan), np.nan)

    # determine if qualifies for SHIELD (optional anchor whitelist OR marketcap >= threshold)
    shield_ok = True
    use_anchor_whitelist = bool(cfg.get("shield", {}).get("use_anchor_whitelist", False))
    anchor_whitelist = {str(x).upper() for x in cfg.get("shield", {}).get("anchor_whitelist", [])}
    if use_anchor_whitelist:
        shield_ok = symbol in anchor_whitelist
    min_marketcap = safe_float(cfg.get("shield", {}).get("min_marketcap", 0), default=0.0)
    marketcap = safe_float(row.get("marketcap", np.nan), default=np.nan)
    if min_marketcap > 0 and np.isfinite(marketcap):
        shield_ok = shield_ok or marketcap >= min_marketcap

    # choose expirations
    exps = pricing.get_expirations(symbol) or []
    if not exps and fallback_pricing:
        exps = fallback_pricing.get_expirations(symbol) or []

    # DTE windows
    shield_lo, shield_hi = cfg["shield"]["dte_range"]
    fire_lo, fire_hi = cfg["fire"]["dte_range"]

    shield_exps = [e for e in exps if within_dte(e, asof, shield_lo, shield_hi)]
    fire_exps = [e for e in exps if within_dte(e, asof, fire_lo, fire_hi)]

    # helper to select expiry nearest target
    def pick_nearest(exp_list: List[dt.date], target: int) -> List[dt.date]:
        if not exp_list:
            return []
        exp_list = sorted(exp_list, key=lambda d: abs((d - asof).days - target))
        return exp_list[: cfg["engine"]["max_expiries_per_ticker"]]

    shield_exps = pick_nearest(shield_exps, cfg["shield"]["target_dte"])
    fire_exps = pick_nearest(fire_exps, cfg["fire"]["target_dte"])

    candidates: List[TradeCandidate] = []
    rejects: List[dict] = []

    # --- Strategy templates ---
    # SHIELD credit spreads (only if shield_ok)
    if shield_ok:
        for expiry in shield_exps:
            dte = (expiry - asof).days
            # Earnings gate: SHIELD cannot cross earnings
            if earnings and earnings <= expiry:
                rejects.append({"ticker": symbol, "expiry": str(expiry), "reason": "shield_crosses_earnings"})
                continue

            # Bull Put Credit (for bullish or neutral)
            if bias == "BULL" or cfg["shield"]["allow_both_sides_when_neutral"]:
                tc, rej = build_bull_put_credit(asof, symbol, spot, expiry, dte, default_w, min_w, max_w, cfg, pricing, fallback_pricing, row)
                candidates += tc
                rejects += rej

            # Bear Call Credit (for bearish or neutral)
            if bias == "BEAR" or cfg["shield"]["allow_both_sides_when_neutral"]:
                tc, rej = build_bear_call_credit(asof, symbol, spot, expiry, dte, default_w, min_w, max_w, cfg, pricing, fallback_pricing, row)
                candidates += tc
                rejects += rej

    # FIRE debit spreads
    for expiry in fire_exps:
        dte = (expiry - asof).days

        # Bull Call Debit
        if bias == "BULL" or cfg["fire"]["allow_both_sides_when_neutral"]:
            tc, rej = build_bull_call_debit(asof, symbol, spot, expiry, dte, default_w, min_w, max_w, cfg, pricing, fallback_pricing, row)
            candidates += tc
            rejects += rej

        # Bear Put Debit
        if bias == "BEAR" or cfg["fire"]["allow_both_sides_when_neutral"]:
            tc, rej = build_bear_put_debit(asof, symbol, spot, expiry, dte, default_w, min_w, max_w, cfg, pricing, fallback_pricing, row)
            candidates += tc
            rejects += rej

    # Deduplicate by (ticker,strategy,expiry,short,long)
    seen = set()
    uniq = []
    for c in candidates:
        key = (c.ticker, c.strategy, c.expiry, c.short_leg, c.long_leg)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    candidates = uniq

    # limit per ticker
    candidates = sorted(candidates, key=lambda x: x.conviction, reverse=True)[: cfg["engine"]["max_trades_per_ticker"]]

    return candidates, rejects


def _get_quote_with_fallback(sym: str, pricing: PricingProvider, fallback: Optional[PricingProvider]) -> Tuple[Optional[Quote], str]:
    q = pricing.get_quote(sym)
    if q:
        return q, q.src
    if fallback:
        q2 = fallback.get_quote(sym)
        if q2:
            return q2, q2.src
    return None, ""


def build_bull_put_credit(
    asof: dt.date,
    ticker: str,
    spot: float,
    expiry: dt.date,
    dte: int,
    default_w: float,
    min_w: float,
    max_w: float,
    cfg: dict,
    pricing: PricingProvider,
    fallback: Optional[PricingProvider],
    row: pd.Series
) -> Tuple[List[TradeCandidate], List[dict]]:
    rejects = []
    out = []

    # Choose short put ~ pct OTM below spot
    otm = cfg["shield"]["credit_short_otm_pct"]
    target_short = spot * (1.0 - otm)
    width_target = default_w

    # For WEB pricing, try to pull chain strikes. If not possible, use fallback strikes from OI data (best effort).
    strikes = None
    if isinstance(pricing, WebPricingProvider):
        try:
            calls, puts = pricing.get_chain(ticker, expiry)
            strikes = np.array(sorted(puts["strike"].unique()))
        except Exception:
            strikes = None

    if (strikes is None or strikes.size == 0) and fallback:
        strikes = fallback.get_strikes(ticker, expiry, "P")

    # fallback strikes from UW OI map
    if strikes is None or strikes.size == 0:
        # cannot generate without strike ladder
        rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "no_strike_ladder_for_puts"})
        return [], rejects

    short_strike = round_to_available_strike(target_short, strikes, side="down")
    if short_strike is None:
        rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "no_short_put_strike"})
        return [], rejects

    # Try a small neighborhood of strikes to find best credit% within tier
    candidate_shorts = [s for s in strikes if s <= short_strike][:]  # all below
    candidate_shorts = sorted(candidate_shorts, reverse=True)[: cfg["engine"]["strike_search_depth"]]

    for s_short in candidate_shorts:
        s_long_target = s_short - width_target
        s_long = round_to_available_strike(s_long_target, strikes, side="down")
        if s_long is None:
            continue
        width = float(s_short - s_long)
        if width < min_w or width > max_w:
            continue
        short_sym = build_occ_symbol(ticker, expiry, "P", float(s_short))
        long_sym = build_occ_symbol(ticker, expiry, "P", float(s_long))

        q_short, src_s = _get_quote_with_fallback(short_sym, pricing, fallback)
        q_long, src_l = _get_quote_with_fallback(long_sym, pricing, fallback)
        if not q_short or not q_long:
            rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "missing_quotes_put_credit", "short": short_sym, "long": long_sym})
            continue

        credit = compute_credit(q_short, q_long)
        if credit is None:
            rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "bad_quotes_put_credit", "short": short_sym, "long": long_sym})
            continue

        min_credit = cfg["gates"]["min_credit_pct_width"] * width
        if credit < min_credit:
            rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "credit_below_min", "credit": credit, "min_credit": min_credit, "short": short_sym, "long": long_sym})
            continue

        breakeven = float(s_short - credit)
        max_profit = credit * 100.0
        max_loss = (width - credit) * 100.0
        credit_pct_width = credit / width

        cushion_to_be = (spot - breakeven) / spot

        # Conviction scoring (simple, deterministic)
        base = 55.0
        # more credit and more cushion and more IVR help
        base += min(15.0, pct(credit_pct_width) * 0.3)  # up to ~15
        base += min(15.0, pct(cushion_to_be) * 0.8)
        if np.isfinite(safe_float(row.get("iv_rank", np.nan))):
            base += min(10.0, float(row.get("iv_rank")) / 10.0)
        # DP above mid supports bullish credit
        if np.isfinite(safe_float(row.get("dp_above_mid_pct", np.nan))):
            base += (float(row.get("dp_above_mid_pct")) - 0.5) * 10.0

        conviction = compute_conviction(base)

        reasons = f"IVR={safe_float(row.get('iv_rank', np.nan)):.1f}; DP_above_mid={safe_float(row.get('dp_above_mid_pct', np.nan)):.2f}; HotPrem={safe_float(row.get('hot_premium',0)):.0f}; OIabs={safe_float(row.get('oi_abs',0)):.0f}; WhalePrem={safe_float(row.get('whale_premium',0)):.0f}"

        out.append(TradeCandidate(
            date=str(asof),
            ticker=ticker,
            track="SHIELD",
            bias="BULL",
            strategy="Bull Put Credit",
            expiry=str(expiry),
            dte=dte,
            short_leg=short_sym,
            long_leg=long_sym,
            width=width,
            entry_gate=f">= {min_credit:.2f}cr",
            net=float(credit),
            net_type="credit",
            credit_pct_width=float(credit_pct_width),
            max_profit=float(max_profit),
            max_loss=float(max_loss),
            breakeven=float(breakeven),
            spot=float(spot),
            earnings_date=str(parse_date_safe(row.get("earnings")) or ""),
            pricing_src=f"{src_s}/{src_l}",
            conviction=conviction,
            reasons=reasons
        ))

    return out, rejects


def build_bear_call_credit(
    asof: dt.date,
    ticker: str,
    spot: float,
    expiry: dt.date,
    dte: int,
    default_w: float,
    min_w: float,
    max_w: float,
    cfg: dict,
    pricing: PricingProvider,
    fallback: Optional[PricingProvider],
    row: pd.Series
) -> Tuple[List[TradeCandidate], List[dict]]:
    rejects = []
    out = []

    otm = cfg["shield"]["credit_short_otm_pct"]
    target_short = spot * (1.0 + otm)
    width_target = default_w

    strikes = None
    if isinstance(pricing, WebPricingProvider):
        try:
            calls, puts = pricing.get_chain(ticker, expiry)
            strikes = np.array(sorted(calls["strike"].unique()))
        except Exception:
            strikes = None

    if (strikes is None or strikes.size == 0) and fallback:
        strikes = fallback.get_strikes(ticker, expiry, "C")

    if strikes is None or strikes.size == 0:
        rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "no_strike_ladder_for_calls"})
        return [], rejects

    short_strike = round_to_available_strike(target_short, strikes, side="up")
    if short_strike is None:
        rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "no_short_call_strike"})
        return [], rejects

    candidate_shorts = [s for s in strikes if s >= short_strike]
    candidate_shorts = sorted(candidate_shorts)[: cfg["engine"]["strike_search_depth"]]

    for s_short in candidate_shorts:
        s_long_target = s_short + width_target
        s_long = round_to_available_strike(s_long_target, strikes, side="up")
        if s_long is None:
            continue
        width = float(s_long - s_short)
        if width < min_w or width > max_w:
            continue

        short_sym = build_occ_symbol(ticker, expiry, "C", float(s_short))
        long_sym = build_occ_symbol(ticker, expiry, "C", float(s_long))

        q_short, src_s = _get_quote_with_fallback(short_sym, pricing, fallback)
        q_long, src_l = _get_quote_with_fallback(long_sym, pricing, fallback)
        if not q_short or not q_long:
            rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "missing_quotes_call_credit", "short": short_sym, "long": long_sym})
            continue

        credit = compute_credit(q_short, q_long)
        if credit is None:
            rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "bad_quotes_call_credit", "short": short_sym, "long": long_sym})
            continue

        min_credit = cfg["gates"]["min_credit_pct_width"] * width
        if credit < min_credit:
            rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "credit_below_min", "credit": credit, "min_credit": min_credit, "short": short_sym, "long": long_sym})
            continue

        breakeven = float(s_short + credit)
        max_profit = credit * 100.0
        max_loss = (width - credit) * 100.0
        credit_pct_width = credit / width

        cushion_to_be = (breakeven - spot) / spot

        base = 55.0
        base += min(15.0, pct(credit_pct_width) * 0.3)
        base += min(15.0, pct(cushion_to_be) * 0.8)
        if np.isfinite(safe_float(row.get("iv_rank", np.nan))):
            base += min(10.0, float(row.get("iv_rank")) / 10.0)
        # DP below mid supports bearish credit
        if np.isfinite(safe_float(row.get("dp_above_mid_pct", np.nan))):
            base += (0.5 - float(row.get("dp_above_mid_pct"))) * 10.0

        conviction = compute_conviction(base)

        reasons = f"IVR={safe_float(row.get('iv_rank', np.nan)):.1f}; DP_above_mid={safe_float(row.get('dp_above_mid_pct', np.nan)):.2f}; HotPrem={safe_float(row.get('hot_premium',0)):.0f}; OIabs={safe_float(row.get('oi_abs',0)):.0f}; WhalePrem={safe_float(row.get('whale_premium',0)):.0f}"

        out.append(TradeCandidate(
            date=str(asof),
            ticker=ticker,
            track="SHIELD",
            bias="BEAR",
            strategy="Bear Call Credit",
            expiry=str(expiry),
            dte=dte,
            short_leg=short_sym,
            long_leg=long_sym,
            width=width,
            entry_gate=f">= {min_credit:.2f}cr",
            net=float(credit),
            net_type="credit",
            credit_pct_width=float(credit_pct_width),
            max_profit=float(max_profit),
            max_loss=float(max_loss),
            breakeven=float(breakeven),
            spot=float(spot),
            earnings_date=str(parse_date_safe(row.get("earnings")) or ""),
            pricing_src=f"{src_s}/{src_l}",
            conviction=conviction,
            reasons=reasons
        ))

    return out, rejects


def build_bull_call_debit(
    asof: dt.date,
    ticker: str,
    spot: float,
    expiry: dt.date,
    dte: int,
    default_w: float,
    min_w: float,
    max_w: float,
    cfg: dict,
    pricing: PricingProvider,
    fallback: Optional[PricingProvider],
    row: pd.Series
) -> Tuple[List[TradeCandidate], List[dict]]:
    rejects = []
    out = []

    # Long call near ATM / slight ITM
    itm_pct = cfg["fire"]["debit_long_itm_pct"]  # 0 = ATM, positive = ITM
    target_long = spot * (1.0 - itm_pct)
    width_target = default_w

    strikes = None
    if isinstance(pricing, WebPricingProvider):
        try:
            calls, puts = pricing.get_chain(ticker, expiry)
            strikes = np.array(sorted(calls["strike"].unique()))
        except Exception:
            strikes = None

    if (strikes is None or strikes.size == 0) and fallback:
        strikes = fallback.get_strikes(ticker, expiry, "C")

    if strikes is None or strikes.size == 0:
        rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "no_strike_ladder_for_calls"})
        return [], rejects

    long_strike = round_to_available_strike(target_long, strikes, side="down")  # seed target
    if long_strike is None:
        rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "no_long_call_strike"})
        return [], rejects

    # Explore nearest long strikes around target (not only ITM side) to avoid
    # systematically excluding high-priced names under strict debit% gates.
    depth = max(1, int(cfg["engine"]["strike_search_depth"]))
    candidate_longs = nearest_strike_candidates(strikes, target_long, limit=max(2, depth * 2))

    tried_pairs = set()
    for s_long in candidate_longs:
        short_candidates = [s for s in strikes if s > s_long and min_w <= (s - s_long) <= max_w]
        short_candidates = sorted(short_candidates, key=lambda s: (abs((s - s_long) - width_target), s))[:depth]
        for s_short in short_candidates:
            pair = (float(s_long), float(s_short))
            if pair in tried_pairs:
                continue
            tried_pairs.add(pair)

            width = float(s_short - s_long)
            if width < min_w or width > max_w:
                continue

            long_sym = build_occ_symbol(ticker, expiry, "C", float(s_long))
            short_sym = build_occ_symbol(ticker, expiry, "C", float(s_short))

            q_long, src_l = _get_quote_with_fallback(long_sym, pricing, fallback)
            q_short, src_s = _get_quote_with_fallback(short_sym, pricing, fallback)
            if not q_long or not q_short:
                rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "missing_quotes_call_debit", "long": long_sym, "short": short_sym})
                continue

            debit = compute_debit(q_long, q_short)
            if debit is None:
                rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "bad_quotes_call_debit", "long": long_sym, "short": short_sym})
                continue

            max_debit = cfg["gates"]["max_debit_pct_width"] * width
            if debit > max_debit:
                rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "debit_above_max", "debit": debit, "max_debit": max_debit, "long": long_sym, "short": short_sym})
                continue

            breakeven = float(s_long + debit)
            max_profit = (width - debit) * 100.0
            max_loss = debit * 100.0

            base = 50.0
            # cheaper debits score higher
            base += (1.0 - min(1.0, debit / max(0.01, width))) * 25.0
            # FIRE score from data
            base += float(row.get("fire_score", 0.0)) * 20.0
            # ask dominance slightly helps bullish
            if np.isfinite(safe_float(row.get("hot_ask_ratio", np.nan))):
                base += (float(row.get("hot_ask_ratio")) - 0.5) * 10.0

            # earnings risk tag (FIRE can cross but penalize)
            penalties = 0.0
            earnings = parse_date_safe(row.get("earnings"))
            if earnings and earnings <= expiry:
                penalties += 8.0  # not fatal, but downgrade

            conviction = compute_conviction(base, penalties=penalties)

            reasons = f"FIRE={float(row.get('fire_score',0.0)):.2f}; HotAsk={safe_float(row.get('hot_ask_ratio', np.nan)):.2f}; HotPrem={safe_float(row.get('hot_premium',0)):.0f}; OIabs={safe_float(row.get('oi_abs',0)):.0f}"

            out.append(TradeCandidate(
                date=str(asof),
                ticker=ticker,
                track="FIRE",
                bias="BULL",
                strategy="Bull Call Debit",
                expiry=str(expiry),
                dte=dte,
                short_leg=short_sym,
                long_leg=long_sym,
                width=width,
                entry_gate=f"<= {max_debit:.2f}db",
                net=float(debit),
                net_type="debit",
                credit_pct_width=float("nan"),
                max_profit=float(max_profit),
                max_loss=float(max_loss),
                breakeven=float(breakeven),
                spot=float(spot),
                earnings_date=str(parse_date_safe(row.get("earnings")) or ""),
                pricing_src=f"{src_l}/{src_s}",
                conviction=conviction,
                reasons=reasons
            ))

    return out, rejects


def build_bear_put_debit(
    asof: dt.date,
    ticker: str,
    spot: float,
    expiry: dt.date,
    dte: int,
    default_w: float,
    min_w: float,
    max_w: float,
    cfg: dict,
    pricing: PricingProvider,
    fallback: Optional[PricingProvider],
    row: pd.Series
) -> Tuple[List[TradeCandidate], List[dict]]:
    rejects = []
    out = []

    itm_pct = cfg["fire"]["debit_long_itm_pct"]
    target_long = spot * (1.0 + itm_pct)  # for puts, ITM means strike above spot
    width_target = default_w

    strikes = None
    if isinstance(pricing, WebPricingProvider):
        try:
            calls, puts = pricing.get_chain(ticker, expiry)
            strikes = np.array(sorted(puts["strike"].unique()))
        except Exception:
            strikes = None

    if (strikes is None or strikes.size == 0) and fallback:
        strikes = fallback.get_strikes(ticker, expiry, "P")

    if strikes is None or strikes.size == 0:
        rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "no_strike_ladder_for_puts"})
        return [], rejects

    long_strike = round_to_available_strike(target_long, strikes, side="up")  # seed target
    if long_strike is None:
        rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "no_long_put_strike"})
        return [], rejects

    depth = max(1, int(cfg["engine"]["strike_search_depth"]))
    candidate_longs = nearest_strike_candidates(strikes, target_long, limit=max(2, depth * 2))

    tried_pairs = set()
    for s_long in candidate_longs:
        short_candidates = [s for s in strikes if s < s_long and min_w <= (s_long - s) <= max_w]
        short_candidates = sorted(short_candidates, key=lambda s: (abs((s_long - s) - width_target), -s))[:depth]
        for s_short in short_candidates:
            pair = (float(s_long), float(s_short))
            if pair in tried_pairs:
                continue
            tried_pairs.add(pair)

            width = float(s_long - s_short)
            if width < min_w or width > max_w:
                continue

            long_sym = build_occ_symbol(ticker, expiry, "P", float(s_long))
            short_sym = build_occ_symbol(ticker, expiry, "P", float(s_short))

            q_long, src_l = _get_quote_with_fallback(long_sym, pricing, fallback)
            q_short, src_s = _get_quote_with_fallback(short_sym, pricing, fallback)
            if not q_long or not q_short:
                rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "missing_quotes_put_debit", "long": long_sym, "short": short_sym})
                continue

            debit = compute_debit(q_long, q_short)
            if debit is None:
                rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "bad_quotes_put_debit", "long": long_sym, "short": short_sym})
                continue

            max_debit = cfg["gates"]["max_debit_pct_width"] * width
            if debit > max_debit:
                rejects.append({"ticker": ticker, "expiry": str(expiry), "reason": "debit_above_max", "debit": debit, "max_debit": max_debit, "long": long_sym, "short": short_sym})
                continue

            breakeven = float(s_long - debit)
            max_profit = (width - debit) * 100.0
            max_loss = debit * 100.0

            base = 50.0
            base += (1.0 - min(1.0, debit / max(0.01, width))) * 25.0
            base += float(row.get("fire_score", 0.0)) * 20.0
            # bid dominance (hot_ask_ratio low) helps bearish
            if np.isfinite(safe_float(row.get("hot_ask_ratio", np.nan))):
                base += (0.5 - float(row.get("hot_ask_ratio"))) * 10.0

            penalties = 0.0
            earnings = parse_date_safe(row.get("earnings"))
            if earnings and earnings <= expiry:
                penalties += 8.0

            conviction = compute_conviction(base, penalties=penalties)
            reasons = f"FIRE={float(row.get('fire_score',0.0)):.2f}; HotAsk={safe_float(row.get('hot_ask_ratio', np.nan)):.2f}; HotPrem={safe_float(row.get('hot_premium',0)):.0f}; OIabs={safe_float(row.get('oi_abs',0)):.0f}"

            out.append(TradeCandidate(
                date=str(asof),
                ticker=ticker,
                track="FIRE",
                bias="BEAR",
                strategy="Bear Put Debit",
                expiry=str(expiry),
                dte=dte,
                short_leg=short_sym,
                long_leg=long_sym,
                width=width,
                entry_gate=f"<= {max_debit:.2f}db",
                net=float(debit),
                net_type="debit",
                credit_pct_width=float("nan"),
                max_profit=float(max_profit),
                max_loss=float(max_loss),
                breakeven=float(breakeven),
                spot=float(spot),
                earnings_date=str(parse_date_safe(row.get("earnings")) or ""),
                pricing_src=f"{src_l}/{src_s}",
                conviction=conviction,
                reasons=reasons
            ))

    return out, rejects


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, required=True, help="Directory containing the 5 EOD files")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--date", type=str, default="", help="YYYY-MM-DD (if omitted, try infer from filenames)")
    ap.add_argument("--config", type=str, default=str((Path(__file__).resolve().parent / "rulebook_config.yaml")), help="Path to YAML config")
    ap.add_argument("--pricing", type=str, default="web_then_uw", choices=["web_then_uw", "uw_only"], help="Pricing mode")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(cfg_path.read_text())

    date_str = args.date.strip() or infer_date_from_filenames(input_dir)
    if not date_str:
        raise ValueError("Could not infer date. Pass --date YYYY-MM-DD")
    asof = dt.datetime.strptime(date_str, "%Y-%m-%d").date()

    dfs = load_eod_bundle(input_dir, date_str)

    # Build signals
    signals = build_signals(dfs, cfg)

    # Select tickers to scan
    signals = signals[signals["symbol"].notna()].copy()
    signals["interest_score"] = signals["interest_score"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signals = signals.sort_values("interest_score", ascending=False)

    max_tickers = int(cfg["engine"]["max_tickers_to_scan"])
    signals_top = signals.head(max_tickers).reset_index(drop=True)

    # Pricing providers
    fallback = UWPricingProvider(dfs["hot"], dfs["oi"])
    pricing: PricingProvider
    fallback_pricing: Optional[PricingProvider] = None

    if args.pricing == "uw_only":
        pricing = fallback
        fallback_pricing = None
    else:
        # C then B (your preference)
        try:
            pricing = WebPricingProvider()
            fallback_pricing = fallback
        except Exception:
            pricing = fallback
            fallback_pricing = None

    all_candidates: List[TradeCandidate] = []
    all_rejects: List[dict] = []

    for _, r in signals_top.iterrows():
        cand, rej = generate_candidates_for_ticker(asof, r, cfg, pricing, fallback_pricing)
        all_candidates.extend(cand)
        all_rejects.extend(rej)

    # Output
    cand_df = pd.DataFrame([asdict(c) for c in all_candidates])
    if not cand_df.empty:
        cand_df = cand_df.sort_values(["conviction", "track"], ascending=[False, True])

    rej_df = pd.DataFrame(all_rejects)

    # Save
    cand_csv = out_dir / f"shortlist_trades_{date_str}.csv"
    rej_csv = out_dir / f"reject_log_{date_str}.csv"
    cand_df.to_csv(cand_csv, index=False)
    rej_df.to_csv(rej_csv, index=False)

    # Shortlist markdown for AI roasting
    md_path = out_dir / f"SHORTLIST_{date_str}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# Shortlist Trades ({date_str})\n\n")
        f.write("These rows passed deterministic HARD gates (width tiers, credit/debit gates, earnings gate for SHIELD, ETF exclusion).\n")
        f.write("Pricing is conservative bid/ask. If pricing source is `uw_*`, treat as *indicative* and verify live.\n\n")
        if cand_df.empty:
            f.write("No candidates passed gates.\n")
        else:
            cols = ["ticker", "track", "bias", "strategy", "expiry", "dte", "short_leg", "long_leg", "width", "entry_gate",
                    "net", "net_type", "credit_pct_width", "breakeven", "spot", "earnings_date", "pricing_src", "conviction", "reasons"]
            display = cand_df[cols].head(int(cfg["engine"]["max_total_trades"])).copy()
            f.write(display.to_markdown(index=False))
            f.write("\n\n")
            f.write("## Roast Prompt\n")
            f.write("Audit these trades under the Serious Options Framework. Identify which 3 are most likely to fail and why (trend, sector, news, hedging vs speculation, DP integrity).\n")

    print(f"Wrote:\n  {cand_csv}\n  {rej_csv}\n  {md_path}")


if __name__ == "__main__":
    main()

