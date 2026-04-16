#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


OCC_RE = re.compile(r"^([A-Z\.]{1,10})(\d{6})([CP])(\d{8})$")
ENTRY_GATE_RE = re.compile(r"^\s*(>=|<=)\s*([0-9]*\.?[0-9]+)\s*(cr|db)\s*$", re.IGNORECASE)
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

ZIP_PREFIX_CHAIN_OI = "chain-oi-changes-"
ZIP_PREFIX_HOT = "hot-chains-"
ZIP_PREFIX_SCREENER = "stock-screener-"

STRATEGY_TO_NET = {
    "Bull Call Debit": "debit",
    "Bear Put Debit": "debit",
    "Bull Put Credit": "credit",
    "Bear Call Credit": "credit",
}

STRATEGY_TO_RIGHT = {
    "Bull Call Debit": "C",
    "Bear Call Credit": "C",
    "Bear Put Debit": "P",
    "Bull Put Credit": "P",
}

DEFAULT_SETUPS_ALIASES = {
    "trade_id": ["trade_id", "id", "row_id"],
    "signal_date": ["signal_date", "entry_date", "as_of_date", "asof", "date"],
    "ticker": ["ticker", "symbol", "underlying"],
    "strategy": ["strategy", "strategy_type", "setup"],
    "expiry": ["expiry", "expiration", "exp_date", "expiration_date"],
    "short_leg": ["short_leg", "short_symbol", "short_option"],
    "long_leg": ["long_leg", "long_symbol", "long_option"],
    "short_strike": ["short_strike", "short_k", "short"],
    "long_strike": ["long_strike", "long_k", "long"],
    "net_type": ["net_type", "debit_credit", "entry_type"],
    "entry_gate": ["entry_gate", "gate", "entry_rule"],
    "entry_net": ["entry_net", "entry_price", "entry", "entry_debit_credit", "net"],
    "exit_date": ["exit_date", "close_date"],
    "exit_net": ["exit_net", "exit_price", "close_price"],
    "qty": ["qty", "quantity", "contracts", "size"],
}

DEFAULT_ACTUAL_ALIASES = {
    "trade_id": ["trade_id", "id", "row_id"],
    "ticker": ["ticker", "symbol", "underlying"],
    "strategy": ["strategy", "strategy_type", "setup"],
    "signal_date": ["signal_date", "entry_date", "open_date", "date"],
    "expiry": ["expiry", "expiration", "exp_date", "expiration_date"],
    "short_leg": ["short_leg", "short_symbol", "short_option"],
    "long_leg": ["long_leg", "long_symbol", "long_option"],
    "realized_pnl": ["realized_pnl", "pnl", "profit_loss", "net_pnl", "amount"],
}


def norm_col(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(text).strip().lower())
    return re.sub(r"_+", "_", s).strip("_")


def find_col(columns: Sequence[str], aliases: Sequence[str]) -> Optional[str]:
    mapped = {c: norm_col(c) for c in columns}
    for alias in aliases:
        a = norm_col(alias)
        for raw, n in mapped.items():
            if n == a:
                return raw
    return None


def safe_float(x: object) -> float:
    try:
        if x is None:
            return math.nan
        if pd.isna(x):
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def parse_date(x: object) -> Optional[dt.date]:
    if x is None:
        return None
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    s = str(x).strip()
    if not s:
        return None
    try:
        return dt.datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_occ_symbol(symbol: object) -> Optional[Tuple[str, dt.date, str, float]]:
    m = OCC_RE.match(str(symbol or "").strip().upper())
    if not m:
        return None
    root, yymmdd, right, strike8 = m.groups()
    yy = int("20" + yymmdd[:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    return root, dt.date(yy, mm, dd), right, int(strike8) / 1000.0


def build_occ_symbol(ticker: str, expiry: dt.date, right: str, strike: float) -> str:
    t = str(ticker).strip().upper()
    r = str(right).strip().upper()
    if r not in {"C", "P"}:
        raise ValueError(f"Invalid option right: {right}")
    strike_i = int(round(float(strike) * 1000))
    return f"{t}{expiry.strftime('%y%m%d')}{r}{strike_i:08d}"


def parse_entry_gate(value: object) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    m = ENTRY_GATE_RE.match(str(value or "").strip())
    if not m:
        return None, None, None
    op, threshold, unit = m.groups()
    try:
        return op, float(threshold), unit.lower()
    except Exception:
        return None, None, None


def intrinsic_value(right: str, strike: float, spot: float) -> float:
    if right.upper() == "C":
        return max(0.0, float(spot) - float(strike))
    return max(0.0, float(strike) - float(spot))


def max_profit_max_loss(width: float, entry_net: float, net_type: str) -> Tuple[float, float]:
    w = float(width)
    n = float(entry_net)
    nt = str(net_type).lower()
    if nt == "credit":
        return n * 100.0, max(0.0, (w - n) * 100.0)
    return max(0.0, (w - n) * 100.0), max(0.0, n * 100.0)


@dataclass(frozen=True)
class LegQuote:
    bid: float
    ask: float
    mid: float
    volume: float
    open_interest: float
    source_kind: str


class HistoricalOptionQuoteStore:
    def __init__(self, root_dir: Path, use_hot: bool = True, use_oi: bool = True) -> None:
        self.root_dir = root_dir
        self.use_hot = bool(use_hot)
        self.use_oi = bool(use_oi)
        self._cache: Dict[dt.date, pd.DataFrame] = {}
        self._date_dirs: Dict[dt.date, Path] = {}
        for p in root_dir.iterdir():
            if not p.is_dir():
                continue
            if not DATE_DIR_RE.match(p.name):
                continue
            d = parse_date(p.name)
            if d is not None:
                self._date_dirs[d] = p

    def available_dates(self) -> List[dt.date]:
        return sorted(self._date_dirs.keys())

    @staticmethod
    def _pick_zip(day_dir: Path, prefix: str) -> Optional[Path]:
        matches = sorted([p for p in day_dir.glob("*.zip") if p.name.startswith(prefix)])
        return matches[-1] if matches else None

    @staticmethod
    def _iter_csv_chunks_from_zip(
        zip_path: Path,
        usecols: Sequence[str],
        chunksize: int = 200_000,
    ) -> Iterable[pd.DataFrame]:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csvs = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])
            if not csvs:
                return
            with zf.open(csvs[0], "r") as fh:
                for chunk in pd.read_csv(fh, usecols=usecols, low_memory=False, chunksize=chunksize):
                    yield chunk

    def _read_hot_quotes(self, zip_path: Path, wanted_symbols: Optional[Set[str]]) -> pd.DataFrame:
        cols = ["option_symbol", "date", "bid", "ask", "volume", "open_interest"]
        parts: List[pd.DataFrame] = []
        for chunk in self._iter_csv_chunks_from_zip(zip_path, cols):
            c = chunk.copy()
            c["option_symbol"] = c["option_symbol"].astype(str).str.upper().str.strip()
            if wanted_symbols:
                c = c[c["option_symbol"].isin(wanted_symbols)]
            if c.empty:
                continue
            c["quote_date"] = pd.to_datetime(c["date"], errors="coerce").dt.date
            c["source_kind"] = "hot"
            parts.append(c[["option_symbol", "quote_date", "bid", "ask", "volume", "open_interest", "source_kind"]])
        if not parts:
            return pd.DataFrame(
                columns=["option_symbol", "quote_date", "bid", "ask", "volume", "open_interest", "source_kind"]
            )
        return pd.concat(parts, ignore_index=True)

    def _read_oi_quotes(self, zip_path: Path, wanted_symbols: Optional[Set[str]]) -> pd.DataFrame:
        cols = ["option_symbol", "curr_date", "last_bid", "last_ask", "volume", "curr_oi"]
        parts: List[pd.DataFrame] = []
        for chunk in self._iter_csv_chunks_from_zip(zip_path, cols):
            c = chunk.copy()
            c["option_symbol"] = c["option_symbol"].astype(str).str.upper().str.strip()
            if wanted_symbols:
                c = c[c["option_symbol"].isin(wanted_symbols)]
            if c.empty:
                continue
            c["quote_date"] = pd.to_datetime(c["curr_date"], errors="coerce").dt.date
            c["bid"] = c["last_bid"]
            c["ask"] = c["last_ask"]
            c["open_interest"] = c["curr_oi"]
            c["source_kind"] = "oi"
            parts.append(c[["option_symbol", "quote_date", "bid", "ask", "volume", "open_interest", "source_kind"]])
        if not parts:
            return pd.DataFrame(
                columns=["option_symbol", "quote_date", "bid", "ask", "volume", "open_interest", "source_kind"]
            )
        return pd.concat(parts, ignore_index=True)

    def _load_date_quotes(self, asof: dt.date, wanted_symbols: Optional[Set[str]]) -> pd.DataFrame:
        day_dir = self._date_dirs.get(asof)
        if day_dir is None:
            return pd.DataFrame(columns=["option_symbol", "bid", "ask", "mid", "volume", "open_interest", "source_kind"])

        parts: List[pd.DataFrame] = []
        if self.use_hot:
            hot_zip = self._pick_zip(day_dir, ZIP_PREFIX_HOT)
            if hot_zip:
                parts.append(self._read_hot_quotes(hot_zip, wanted_symbols))
        if self.use_oi:
            oi_zip = self._pick_zip(day_dir, ZIP_PREFIX_CHAIN_OI)
            if oi_zip:
                parts.append(self._read_oi_quotes(oi_zip, wanted_symbols))
        if not parts:
            return pd.DataFrame(columns=["option_symbol", "bid", "ask", "mid", "volume", "open_interest", "source_kind"])

        q = pd.concat(parts, ignore_index=True)
        q["bid"] = pd.to_numeric(q["bid"], errors="coerce")
        q["ask"] = pd.to_numeric(q["ask"], errors="coerce")
        q["volume"] = pd.to_numeric(q["volume"], errors="coerce")
        q["open_interest"] = pd.to_numeric(q["open_interest"], errors="coerce")

        q = q[(q["quote_date"].isna()) | (q["quote_date"] == asof)].copy()
        q = q[np.isfinite(q["bid"]) & np.isfinite(q["ask"]) & (q["ask"] > 0)].copy()
        q["mid"] = 0.5 * (q["bid"] + q["ask"])
        q["spread"] = q["ask"] - q["bid"]
        q["source_prio"] = np.where(q["source_kind"].astype(str).str.lower() == "hot", 0, 1)

        q = q.sort_values(["option_symbol", "source_prio", "spread"]).drop_duplicates("option_symbol", keep="first")
        return q[["option_symbol", "bid", "ask", "mid", "volume", "open_interest", "source_kind"]].reset_index(drop=True)

    def get_quotes_for_date(self, asof: dt.date, wanted_symbols: Optional[Set[str]] = None) -> pd.DataFrame:
        if asof not in self._cache:
            self._cache[asof] = self._load_date_quotes(asof, wanted_symbols=None)
        q = self._cache[asof]
        if wanted_symbols:
            return q[q["option_symbol"].isin(wanted_symbols)].copy()
        return q.copy()

    def get_leg_quote(self, asof: dt.date, symbol: str) -> Optional[LegQuote]:
        sym = str(symbol or "").upper().strip()
        if not sym:
            return None
        q = self.get_quotes_for_date(asof, wanted_symbols={sym})
        if q.empty:
            return None
        r = q.iloc[0]
        return LegQuote(
            bid=float(r["bid"]),
            ask=float(r["ask"]),
            mid=float(r["mid"]),
            volume=float(r["volume"]) if np.isfinite(safe_float(r["volume"])) else math.nan,
            open_interest=float(r["open_interest"]) if np.isfinite(safe_float(r["open_interest"])) else math.nan,
            source_kind=str(r["source_kind"]),
        )


class UnderlyingCloseStore:
    def __init__(self, root_dir: Path, allow_web_fallback: bool = True) -> None:
        self.root_dir = root_dir
        self.allow_web_fallback = bool(allow_web_fallback)
        self._local_close_map: Dict[Tuple[str, dt.date], float] = {}
        self._by_ticker_dates: Dict[str, List[Tuple[dt.date, float]]] = {}
        self._web_hist: Dict[str, pd.DataFrame] = {}
        self._web_bounds: Dict[str, Tuple[dt.date, dt.date]] = {}
        self._local_loaded = False
        self._date_dirs: Dict[dt.date, Path] = {}
        for p in root_dir.iterdir():
            if p.is_dir() and DATE_DIR_RE.match(p.name):
                d = parse_date(p.name)
                if d is not None:
                    self._date_dirs[d] = p

    @staticmethod
    def _pick_zip(day_dir: Path, prefix: str) -> Optional[Path]:
        matches = sorted([p for p in day_dir.glob("*.zip") if p.name.startswith(prefix)])
        return matches[-1] if matches else None

    @staticmethod
    def _read_screener_rows(zip_path: Path) -> pd.DataFrame:
        cols = ["date", "ticker", "close"]
        with zipfile.ZipFile(zip_path, "r") as zf:
            csvs = sorted([n for n in zf.namelist() if n.lower().endswith(".csv")])
            if not csvs:
                return pd.DataFrame(columns=cols)
            with zf.open(csvs[0], "r") as fh:
                return pd.read_csv(fh, usecols=cols, low_memory=False)

    def _load_local(self) -> None:
        if self._local_loaded:
            return
        rows: List[pd.DataFrame] = []
        for _, day_dir in sorted(self._date_dirs.items()):
            z = self._pick_zip(day_dir, ZIP_PREFIX_SCREENER)
            if not z:
                continue
            try:
                rows.append(self._read_screener_rows(z))
            except Exception:
                continue
        if rows:
            sc = pd.concat(rows, ignore_index=True)
            sc["ticker"] = sc["ticker"].astype(str).str.upper().str.strip()
            sc["date"] = pd.to_datetime(sc["date"], errors="coerce").dt.date
            sc["close"] = pd.to_numeric(sc["close"], errors="coerce")
            sc = sc.dropna(subset=["ticker", "date", "close"])
            sc = sc.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
            for r in sc.itertuples(index=False):
                self._local_close_map[(str(r.ticker), r.date)] = float(r.close)
            by_ticker = sc.groupby("ticker", dropna=False)
            for ticker, g in by_ticker:
                self._by_ticker_dates[str(ticker)] = [(d, float(c)) for d, c in zip(g["date"], g["close"])]
        self._local_loaded = True

    def _get_local_close_on_or_before(self, ticker: str, target: dt.date, lookback_days: int) -> Optional[float]:
        self._load_local()
        t = str(ticker).upper().strip()
        if not t:
            return None
        items = self._by_ticker_dates.get(t, [])
        if not items:
            return None
        lo = target - dt.timedelta(days=max(0, int(lookback_days)))
        best_val: Optional[float] = None
        best_date: Optional[dt.date] = None
        for d, close in items:
            if d <= target and d >= lo:
                if best_date is None or d > best_date:
                    best_date = d
                    best_val = close
        return best_val

    def _load_web_history(self, ticker: str, start: dt.date, end: dt.date) -> pd.DataFrame:
        t = str(ticker).upper().strip()
        cached = self._web_hist.get(t)
        bounds = self._web_bounds.get(t)
        if cached is not None and bounds is not None and not cached.empty:
            lo, hi = bounds
            if start >= lo and end <= hi:
                return cached

        query_start = start
        query_end = end
        if bounds is not None:
            lo, hi = bounds
            query_start = min(query_start, lo)
            query_end = max(query_end, hi)

        df_new = pd.DataFrame()
        query_t = "^VIX" if t == "VIX" else t
        try:
            tmp = yf.download(
                query_t,
                start=query_start.isoformat(),
                end=(query_end + dt.timedelta(days=1)).isoformat(),
                auto_adjust=True,  # Match setup_likelihood_backtest; prevents split-induced price breaks
                progress=False,
                actions=False,
                threads=False,
            )
            if tmp is not None and not tmp.empty:
                # yfinance may return a MultiIndex even for a single ticker; flatten to price field names.
                if isinstance(tmp.columns, pd.MultiIndex):
                    tmp.columns = [str(c[0]) for c in tmp.columns]
                tmp = tmp.reset_index()
                tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce").dt.date
                tmp["Close"] = pd.to_numeric(tmp["Close"], errors="coerce")
                df_new = tmp[["Date", "Close"]].dropna().rename(columns={"Date": "date", "Close": "close"})
        except Exception:
            df_new = pd.DataFrame(columns=["date", "close"])

        if cached is not None and not cached.empty and not df_new.empty:
            df = pd.concat([cached, df_new], ignore_index=True)
            df = df.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date", keep="last")
        elif df_new.empty and cached is not None:
            # Preserve old cache on transient download failure.
            df = cached
        else:
            df = df_new

        if not df.empty:
            dmin = pd.to_datetime(df["date"], errors="coerce").min()
            dmax = pd.to_datetime(df["date"], errors="coerce").max()
            if not pd.isna(dmin) and not pd.isna(dmax):
                self._web_bounds[t] = (dmin.date(), dmax.date())
        elif bounds is not None:
            self._web_bounds[t] = bounds

        self._web_hist[t] = df
        return df

    def get_close_on_or_before(self, ticker: str, target: dt.date, lookback_days: int = 7) -> Optional[float]:
        if target > dt.date.today():
            return None
        local = self._get_local_close_on_or_before(ticker, target, lookback_days=lookback_days)
        if local is not None and np.isfinite(local):
            return float(local)
        if not self.allow_web_fallback:
            return None
        lo = target - dt.timedelta(days=max(10, int(lookback_days) + 5))
        hist = self._load_web_history(str(ticker), lo, target)
        if hist.empty:
            return None
        h = hist[(hist["date"] <= target) & (hist["date"] >= (target - dt.timedelta(days=lookback_days)))]
        if h.empty:
            return None
        return float(h.sort_values("date").iloc[-1]["close"])


def detect_date_from_path(path: Path) -> Optional[dt.date]:
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not m:
        return None
    return parse_date(m.group(1))


def _normalize_setups_df(df: pd.DataFrame, default_signal_date: Optional[dt.date]) -> pd.DataFrame:
    cols = list(df.columns)
    out = pd.DataFrame(index=df.index)

    for canonical, aliases in DEFAULT_SETUPS_ALIASES.items():
        raw = find_col(cols, aliases)
        if raw:
            out[canonical] = df[raw]
        else:
            out[canonical] = np.nan

    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["strategy"] = out["strategy"].astype(str).str.strip()
    out["signal_date"] = out["signal_date"].map(parse_date)
    out["expiry"] = out["expiry"].map(parse_date)

    if default_signal_date is not None:
        out["signal_date"] = out["signal_date"].fillna(default_signal_date)
    out["qty"] = pd.to_numeric(out["qty"], errors="coerce").fillna(1.0)
    out["entry_net"] = pd.to_numeric(out["entry_net"], errors="coerce")
    out["exit_net"] = pd.to_numeric(out["exit_net"], errors="coerce")
    out["short_strike"] = pd.to_numeric(out["short_strike"], errors="coerce")
    out["long_strike"] = pd.to_numeric(out["long_strike"], errors="coerce")

    out["short_leg"] = out["short_leg"].astype(str).str.upper().str.strip()
    out["long_leg"] = out["long_leg"].astype(str).str.upper().str.strip()
    out["short_leg"] = out["short_leg"].replace({"NAN": "", "NONE": ""})
    out["long_leg"] = out["long_leg"].replace({"NAN": "", "NONE": ""})

    for leg_col, strike_col in [("short_leg", "short_strike"), ("long_leg", "long_strike")]:
        parsed = out[leg_col].map(parse_occ_symbol)
        has = parsed.notna()
        if has.any():
            parsed_ok = parsed[has]
            out.loc[has, "ticker"] = out.loc[has, "ticker"].mask(
                out.loc[has, "ticker"].isna() | (out.loc[has, "ticker"].astype(str).str.len() == 0),
                parsed_ok.map(lambda x: x[0]),
            )
            out.loc[has & out["expiry"].isna(), "expiry"] = parsed_ok.map(lambda x: x[1])
            out.loc[has & out[strike_col].isna(), strike_col] = parsed_ok.map(lambda x: x[3])

    for idx, row in out.iterrows():
        ticker = str(row.get("ticker", "")).upper().strip()
        strategy = str(row.get("strategy", "")).strip()
        expiry = row.get("expiry")
        if not ticker or not isinstance(expiry, dt.date) or not strategy:
            continue
        right = STRATEGY_TO_RIGHT.get(strategy)
        if not right:
            continue
        if not row.get("long_leg") and np.isfinite(safe_float(row.get("long_strike"))):
            out.at[idx, "long_leg"] = build_occ_symbol(ticker, expiry, right, float(row["long_strike"]))
        if not row.get("short_leg") and np.isfinite(safe_float(row.get("short_strike"))):
            out.at[idx, "short_leg"] = build_occ_symbol(ticker, expiry, right, float(row["short_strike"]))

    out["net_type"] = out["net_type"].astype(str).str.lower().str.strip()
    out["net_type"] = out.apply(
        lambda r: STRATEGY_TO_NET.get(str(r.get("strategy", "")).strip(), r["net_type"]) if not r["net_type"] else r["net_type"],
        axis=1,
    )
    out["exit_date"] = out["exit_date"].map(parse_date)
    out["entry_gate"] = out["entry_gate"].astype(str).replace({"nan": ""}).str.strip()
    out["trade_id"] = out["trade_id"].astype(str).replace({"nan": ""}).str.strip()
    out["width"] = (out["short_strike"] - out["long_strike"]).abs()

    valid = (
        out["ticker"].astype(str).str.len() > 0
    ) & (
        out["strategy"].astype(str).str.len() > 0
    ) & out["signal_date"].notna() & out["expiry"].notna() & (
        out["short_leg"].astype(str).str.len() > 0
    ) & (
        out["long_leg"].astype(str).str.len() > 0
    ) & (
        out["net_type"].isin(["debit", "credit"])
    )
    out = out[valid].copy().reset_index(drop=True)

    out["entry_gate_op"], out["entry_gate_threshold"], out["entry_gate_unit"] = zip(
        *out["entry_gate"].map(parse_entry_gate)
    )

    return out


def load_setups_csv(path: Path, default_signal_date: Optional[dt.date]) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    return _normalize_setups_df(raw, default_signal_date=default_signal_date)


def _spread_entry_from_quotes(net_type: str, short_q: LegQuote, long_q: LegQuote, price_model: str) -> float:
    mode = str(price_model).lower()
    if mode == "mid":
        if net_type == "credit":
            return float(short_q.mid - long_q.mid)
        return float(long_q.mid - short_q.mid)
    if net_type == "credit":
        return float(short_q.bid - long_q.ask)
    return float(long_q.ask - short_q.bid)


def _spread_exit_from_quotes(net_type: str, short_q: LegQuote, long_q: LegQuote, price_model: str) -> float:
    mode = str(price_model).lower()
    if mode == "mid":
        if net_type == "credit":
            return float(short_q.mid - long_q.mid)
        return float(long_q.mid - short_q.mid)
    if net_type == "credit":
        return float(short_q.ask - long_q.bid)
    return float(long_q.bid - short_q.ask)


def _spread_value_at_expiry(long_leg: str, short_leg: str, spot: float, net_type: str) -> Optional[float]:
    p_long = parse_occ_symbol(long_leg)
    p_short = parse_occ_symbol(short_leg)
    if p_long is None or p_short is None:
        return None
    _, _, right_l, strike_l = p_long
    _, _, right_s, strike_s = p_short
    long_intr = intrinsic_value(right_l, strike_l, spot)
    short_intr = intrinsic_value(right_s, strike_s, spot)
    if str(net_type).lower() == "credit":
        # Credit spreads are opened short-over-long and closed by paying short-long.
        return float(short_intr - long_intr)
    return float(long_intr - short_intr)


def _pnl_from_spread(entry_net: float, exit_spread_value: float, net_type: str, qty: float) -> float:
    q = float(qty)
    if str(net_type).lower() == "credit":
        return float((entry_net - exit_spread_value) * 100.0 * q)
    return float((exit_spread_value - entry_net) * 100.0 * q)


def run_backtest(
    setups: pd.DataFrame,
    quote_store: HistoricalOptionQuoteStore,
    close_store: UnderlyingCloseStore,
    entry_source: str,
    entry_price_model: str,
    exit_mode: str,
    exit_price_model: str,
    close_lookback_days: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for r in setups.itertuples(index=False):
        signal_date: dt.date = r.signal_date
        expiry: dt.date = r.expiry
        ticker = str(r.ticker).upper().strip()
        strategy = str(r.strategy).strip()
        net_type = str(r.net_type).lower().strip()
        short_leg = str(r.short_leg).upper().strip()
        long_leg = str(r.long_leg).upper().strip()
        entry_gate = str(r.entry_gate or "").strip()
        qty = float(r.qty) if np.isfinite(safe_float(r.qty)) else 1.0
        width = float(r.width) if np.isfinite(safe_float(r.width)) else abs(float(r.short_strike) - float(r.long_strike))
        short_strike = float(r.short_strike) if np.isfinite(safe_float(r.short_strike)) else math.nan
        long_strike = float(r.long_strike) if np.isfinite(safe_float(r.long_strike)) else math.nan

        base_row = {
            "trade_id": r.trade_id,
            "signal_date": signal_date,
            "ticker": ticker,
            "strategy": strategy,
            "expiry": expiry,
            "short_leg": short_leg,
            "long_leg": long_leg,
            "short_strike": short_strike,
            "long_strike": long_strike,
            "width": float(width),
            "net_type": net_type,
            "qty": qty,
            "entry_gate": entry_gate,
        }

        entry_net = safe_float(r.entry_net)
        entry_src = "input"
        short_q_entry: Optional[LegQuote] = None
        long_q_entry: Optional[LegQuote] = None
        if str(entry_source).lower() == "quotes_only" or (str(entry_source).lower() == "auto" and not np.isfinite(entry_net)):
            short_q_entry = quote_store.get_leg_quote(signal_date, short_leg)
            long_q_entry = quote_store.get_leg_quote(signal_date, long_leg)
            if short_q_entry is not None and long_q_entry is not None:
                entry_net = _spread_entry_from_quotes(net_type, short_q_entry, long_q_entry, entry_price_model)
                entry_src = f"quotes:{entry_price_model}"
            else:
                entry_net = math.nan
                entry_src = "missing_quotes"
        elif str(entry_source).lower() == "input_only":
            entry_src = "input"
        else:
            entry_src = "input" if np.isfinite(entry_net) else "missing_input_entry_net"

        if not np.isfinite(entry_net):
            rows.append(
                {
                    **base_row,
                    "entry_net": np.nan,
                    "entry_source": entry_src,
                    "status": "skipped_missing_entry",
                    "status_reason": "missing_entry_net_or_quotes",
                    "gate_pass": False if entry_gate else np.nan,
                }
            )
            continue

        op, thr, _ = parse_entry_gate(entry_gate)
        gate_pass = True
        # Guard against floating-point noise around threshold boundaries.
        gate_eps = 1e-9
        if op and thr is not None:
            if net_type == "debit":
                gate_pass = bool(entry_net <= (thr + gate_eps)) if op == "<=" else bool(entry_net >= (thr - gate_eps))
            else:
                gate_pass = bool(entry_net >= (thr - gate_eps)) if op == ">=" else bool(entry_net <= (thr + gate_eps))
        if not gate_pass:
            rows.append(
                {
                    **base_row,
                    "entry_net": float(entry_net),
                    "entry_source": entry_src,
                    "status": "skipped_gate_fail",
                    "status_reason": "entry_gate_failed",
                    "gate_pass": False,
                }
            )
            continue

        mode = str(exit_mode).lower().strip()
        exit_value = math.nan
        exit_src = "missing_exit"
        close_date: Optional[dt.date] = parse_date(r.exit_date)

        # Strict policy by mode:
        # - input_then_quotes_then_expiry: allow input exit mark first.
        # - quotes_then_expiry: ignore input exit, try quote close mark.
        # - expiry_intrinsic: ignore input/quotes and settle by expiry intrinsic.
        input_exit_value = safe_float(r.exit_net)
        if mode == "input_then_quotes_then_expiry" and np.isfinite(input_exit_value):
            exit_value = input_exit_value
            exit_src = "input"
        else:
            if mode in {"quotes_then_expiry", "input_then_quotes_then_expiry"} and close_date:
                short_q_exit = quote_store.get_leg_quote(close_date, short_leg)
                long_q_exit = quote_store.get_leg_quote(close_date, long_leg)
                if short_q_exit is not None and long_q_exit is not None:
                    exit_value = _spread_exit_from_quotes(net_type, short_q_exit, long_q_exit, exit_price_model)
                    exit_src = f"quotes:{exit_price_model}"
                else:
                    exit_value = math.nan

            if not np.isfinite(exit_value):
                today = dt.date.today()
                if expiry > today and (close_date is None or close_date > today):
                    rows.append(
                        {
                            **base_row,
                            "entry_net": float(entry_net),
                            "entry_source": entry_src,
                            "gate_pass": True,
                            "status": "open_not_expired",
                            "status_reason": "expiry_in_future_no_exit_mark",
                        }
                    )
                    continue

                spot = close_store.get_close_on_or_before(ticker, expiry, lookback_days=close_lookback_days)
                if spot is not None and np.isfinite(spot):
                    v = _spread_value_at_expiry(long_leg, short_leg, float(spot), net_type=net_type)
                    if v is not None and np.isfinite(v):
                        exit_value = float(v)
                        exit_src = "expiry_intrinsic"

        if not np.isfinite(exit_value):
            rows.append(
                {
                    **base_row,
                    "entry_net": float(entry_net),
                    "entry_source": entry_src,
                    "gate_pass": True,
                    "status": "failed_missing_exit_price",
                    "status_reason": "no_exit_quotes_or_expiry_spot",
                }
            )
            continue

        pnl = _pnl_from_spread(float(entry_net), float(exit_value), net_type, qty=qty)
        max_profit_1, max_loss_1 = max_profit_max_loss(width, float(entry_net), net_type)
        max_profit = float(max_profit_1 * qty)
        max_loss = float(max_loss_1 * qty)
        return_on_risk = float(pnl / max_loss) if max_loss > 0 else math.nan
        win = bool(pnl > 0)

        rows.append(
            {
                **base_row,
                "gate_pass": True,
                "entry_net": float(entry_net),
                "entry_source": entry_src,
                "exit_net": float(exit_value),
                "exit_source": exit_src,
                "pnl": float(pnl),
                "return_on_risk": return_on_risk,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "win": win,
                "status": "completed",
                "status_reason": "ok",
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["signal_date", "ticker", "strategy"]).reset_index(drop=True)
    return out


def summarize_completed(results: pd.DataFrame) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame]:
    done = results[results["status"] == "completed"].copy()
    if done.empty:
        summary = {
            "completed_trades": 0,
            "net_pnl": 0.0,
            "win_rate": math.nan,
            "profit_factor": math.nan,
            "avg_pnl": math.nan,
            "start_signal_date": "",
            "end_signal_date": "",
        }
        return summary, pd.DataFrame(), pd.DataFrame()

    gp = float(done.loc[done["pnl"] > 0, "pnl"].sum())
    gl = float(-done.loc[done["pnl"] < 0, "pnl"].sum())
    pf = math.inf if gl <= 0 and gp > 0 else (math.nan if gl <= 0 else gp / gl)
    summary = {
        "completed_trades": int(len(done)),
        "skipped_or_failed_trades": int(len(results) - len(done)),
        "net_pnl": float(done["pnl"].sum()),
        "gross_profit": gp,
        "gross_loss": gl,
        "win_rate": float((done["pnl"] > 0).mean()),
        "profit_factor": float(pf) if np.isfinite(pf) else ("inf" if pf == math.inf else math.nan),
        "avg_pnl": float(done["pnl"].mean()),
        "median_pnl": float(done["pnl"].median()),
        "avg_return_on_risk": float(done["return_on_risk"].mean()),
        "start_signal_date": str(done["signal_date"].min()),
        "end_signal_date": str(done["signal_date"].max()),
    }

    by_strategy = (
        done.groupby("strategy", dropna=False)
        .agg(
            trades=("pnl", "size"),
            win_rate=("win", "mean"),
            net_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_return_on_risk=("return_on_risk", "mean"),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
        .reset_index(drop=True)
    )

    by_ticker_setup = (
        done.groupby(["ticker", "strategy", "long_leg", "short_leg", "expiry"], dropna=False)
        .agg(
            trades=("pnl", "size"),
            win_rate=("win", "mean"),
            net_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            avg_return_on_risk=("return_on_risk", "mean"),
        )
        .reset_index()
        .sort_values("net_pnl", ascending=False)
        .reset_index(drop=True)
    )
    return summary, by_strategy, by_ticker_setup


def _normalize_actual_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    out = pd.DataFrame(index=df.index)
    for canonical, aliases in DEFAULT_ACTUAL_ALIASES.items():
        raw = find_col(cols, aliases)
        out[canonical] = df[raw] if raw else np.nan
    out["trade_id"] = out["trade_id"].astype(str).replace({"nan": ""}).str.strip()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
    out["strategy"] = out["strategy"].astype(str).str.strip()
    out["signal_date"] = out["signal_date"].map(parse_date)
    out["expiry"] = out["expiry"].map(parse_date)
    out["short_leg"] = out["short_leg"].astype(str).str.upper().str.strip().replace({"NAN": ""})
    out["long_leg"] = out["long_leg"].astype(str).str.upper().str.strip().replace({"NAN": ""})
    out["realized_pnl"] = pd.to_numeric(out["realized_pnl"], errors="coerce")
    out = out.dropna(subset=["realized_pnl"]).copy()
    return out.reset_index(drop=True)


def validate_backtest_vs_actual(backtest_results: pd.DataFrame, actual_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    bt = backtest_results[backtest_results["status"] == "completed"].copy()
    if bt.empty:
        return pd.DataFrame(), {"matched_trades": 0, "message": "No completed backtest trades to validate."}

    act = _normalize_actual_df(actual_df)
    if act.empty:
        return pd.DataFrame(), {"matched_trades": 0, "message": "No usable rows in actual trades file."}

    join_keys: List[str] = []
    if (bt["trade_id"].astype(str).str.len() > 0).any() and (act["trade_id"].astype(str).str.len() > 0).any():
        join_keys = ["trade_id"]
    else:
        candidate = ["ticker", "strategy", "signal_date", "expiry", "long_leg", "short_leg"]
        join_keys = [k for k in candidate if k in bt.columns and k in act.columns]
        join_keys = [k for k in join_keys if bt[k].notna().any() and act[k].notna().any()]
        if len(join_keys) < 3:
            return pd.DataFrame(), {
                "matched_trades": 0,
                "message": "Insufficient shared keys to match simulated and actual trades.",
            }

    merged = bt.merge(
        act,
        on=join_keys,
        how="inner",
        suffixes=("_sim", "_actual"),
    )
    if merged.empty:
        return merged, {"matched_trades": 0, "join_keys": join_keys, "message": "No matches between backtest and actual."}

    merged["actual_pnl"] = pd.to_numeric(merged["realized_pnl"], errors="coerce")
    merged["sim_pnl"] = pd.to_numeric(merged["pnl"], errors="coerce")
    merged = merged.dropna(subset=["actual_pnl", "sim_pnl"]).copy()
    if merged.empty:
        return merged, {"matched_trades": 0, "join_keys": join_keys, "message": "Matched rows missing pnl values."}

    merged["error"] = merged["sim_pnl"] - merged["actual_pnl"]
    merged["abs_error"] = merged["error"].abs()
    merged["direction_match"] = (merged["sim_pnl"] > 0) == (merged["actual_pnl"] > 0)

    mae = float(merged["abs_error"].mean())
    rmse = float(math.sqrt(np.square(merged["error"]).mean()))
    bias = float(merged["error"].mean())
    hit = float(merged["direction_match"].mean())
    corr = float(merged["sim_pnl"].corr(merged["actual_pnl"])) if len(merged) >= 2 else math.nan

    summary = {
        "matched_trades": int(len(merged)),
        "join_keys": join_keys,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "directional_hit_rate": hit,
        "pnl_correlation": corr,
    }
    return merged, summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Exact ticker + spread-leg historical backtester with validation support.")
    ap.add_argument("--setups-csv", required=True, help="Input setups CSV with legs/expiry/strategy.")
    ap.add_argument("--root-dir", default=r"c:\uw_root", help="Root folder with YYYY-MM-DD day directories.")
    ap.add_argument("--out-dir", default=r"c:\uw_root\out\exact_spread_backtest", help="Output directory.")
    ap.add_argument("--signal-date", default="", help="Default signal date (YYYY-MM-DD) when missing in setups CSV.")
    ap.add_argument(
        "--entry-source",
        choices=["auto", "quotes_only", "input_only"],
        default="auto",
        help="Entry net source policy.",
    )
    ap.add_argument(
        "--entry-price-model",
        choices=["conservative", "mid"],
        default="conservative",
        help="How to price spread entry from quotes.",
    )
    ap.add_argument(
        "--exit-mode",
        choices=["expiry_intrinsic", "quotes_then_expiry", "input_then_quotes_then_expiry"],
        default="quotes_then_expiry",
        help="Exit valuation policy.",
    )
    ap.add_argument(
        "--exit-price-model",
        choices=["conservative", "mid"],
        default="conservative",
        help="How to price spread exit when exit-date quotes are used.",
    )
    ap.add_argument("--close-lookback-days", type=int, default=7, help="Lookback for underlying close on/before expiry.")
    ap.add_argument("--no-web-fallback", action="store_true", help="Disable yfinance fallback for missing closes.")
    ap.add_argument("--no-hot", action="store_true", help="Do not use hot-chains quotes.")
    ap.add_argument("--no-oi", action="store_true", help="Do not use chain-oi quotes.")
    ap.add_argument("--actual-trades-csv", default="", help="Optional actual trades CSV for validation.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    setups_csv = Path(args.setups_csv).expanduser().resolve()
    if not setups_csv.exists():
        raise FileNotFoundError(f"Missing setups CSV: {setups_csv}")

    root_dir = Path(args.root_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    default_signal_date: Optional[dt.date] = parse_date(args.signal_date)
    if default_signal_date is None:
        default_signal_date = detect_date_from_path(setups_csv)

    setups = load_setups_csv(setups_csv, default_signal_date=default_signal_date)
    if setups.empty:
        raise RuntimeError("No valid setups after normalization. Check required columns and values.")

    quote_store = HistoricalOptionQuoteStore(
        root_dir=root_dir,
        use_hot=not args.no_hot,
        use_oi=not args.no_oi,
    )
    close_store = UnderlyingCloseStore(root_dir=root_dir, allow_web_fallback=not args.no_web_fallback)

    results = run_backtest(
        setups=setups,
        quote_store=quote_store,
        close_store=close_store,
        entry_source=args.entry_source,
        entry_price_model=args.entry_price_model,
        exit_mode=args.exit_mode,
        exit_price_model=args.exit_price_model,
        close_lookback_days=max(1, int(args.close_lookback_days)),
    )

    summary, by_strategy, by_ticker_setup = summarize_completed(results)
    status_counts = results["status"].value_counts(dropna=False).to_dict() if not results.empty else {}
    summary["status_counts"] = {str(k): int(v) for k, v in status_counts.items()}

    results_csv = out_dir / "trade_level_results.csv"
    summary_json = out_dir / "summary.json"
    strategy_csv = out_dir / "summary_by_strategy.csv"
    setup_csv = out_dir / "summary_by_ticker_setup.csv"

    results.to_csv(results_csv, index=False)
    by_strategy.to_csv(strategy_csv, index=False)
    by_ticker_setup.to_csv(setup_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    validation_summary = None
    validation_matches_csv = None
    if args.actual_trades_csv:
        actual_csv = Path(args.actual_trades_csv).expanduser().resolve()
        if not actual_csv.exists():
            raise FileNotFoundError(f"Missing actual trades CSV: {actual_csv}")
        actual = pd.read_csv(actual_csv, low_memory=False)
        matches, validation_summary = validate_backtest_vs_actual(results, actual)
        validation_matches_csv = out_dir / "validation_matches.csv"
        validation_json = out_dir / "validation_summary.json"
        matches.to_csv(validation_matches_csv, index=False)
        validation_json.write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    print(f"Input setups: {setups_csv}")
    print(f"Valid setups loaded: {len(setups):,}")
    print(f"Completed trades: {summary.get('completed_trades', 0):,}")
    print(f"Status counts: {summary.get('status_counts', {})}")
    print(f"Wrote: {results_csv}")
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {strategy_csv}")
    print(f"Wrote: {setup_csv}")
    if validation_summary is not None:
        print(f"Validation summary: {validation_summary}")
        print(f"Wrote: {validation_matches_csv}")


if __name__ == "__main__":
    main()
