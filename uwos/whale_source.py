from __future__ import annotations

import io
import re
import zipfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


BOT_EOD_PREFIX = "bot-eod-report-"
WHALE_MD_PREFIX = "whale-"
DEFAULT_CHUNKSIZE = 250_000
DEFAULT_TOP_N = 200
WHALE_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-/]{0,9}$")


@dataclass
class WhaleFlow:
    source_path: Path
    source_label: str
    total_rows: int
    yes_prime_rows: int
    symbol_summary: pd.DataFrame
    top_trades: pd.DataFrame
    track_summary: pd.DataFrame
    option_type_summary: pd.DataFrame
    side_summary: pd.DataFrame

    def as_rank_tables(self) -> dict[str, pd.DataFrame]:
        """Tables consumed by Mode A discovery for ticker-level whale rank."""
        return {
            "Top Symbols by Total Premium (Yes-Prime)": self.symbol_summary.copy(),
        }


def infer_date_from_path(path: Path) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(path))
    return match.group(0) if match else "Unknown Date"


def find_bot_eod_source(base_dir: Path, date_str: str | None = None) -> Path:
    base_dir = Path(base_dir)
    if date_str:
        patterns = [
            f"{BOT_EOD_PREFIX}{date_str}.zip",
            f"{BOT_EOD_PREFIX}{date_str}.csv",
            f"{BOT_EOD_PREFIX}{date_str}*.zip",
            f"{BOT_EOD_PREFIX}{date_str}*.csv",
        ]
    else:
        patterns = [f"{BOT_EOD_PREFIX}*.zip", f"{BOT_EOD_PREFIX}*.csv"]

    seen: set[Path] = set()
    for pattern in patterns:
        matches = []
        for path in sorted(base_dir.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            matches.append(path)
        if matches:
            return matches[-1]

    suffix = f" for {date_str}" if date_str else ""
    raise FileNotFoundError(f"Missing {BOT_EOD_PREFIX}YYYY-MM-DD CSV/ZIP{suffix} in {base_dir}")


def find_whale_markdown_source(base_dir: Path, date_str: str | None = None) -> Path:
    """Find a legacy whale markdown summary for old folders without bot EOD exports."""
    base_dir = Path(base_dir)
    if date_str:
        patterns = [f"{WHALE_MD_PREFIX}{date_str}.md", f"{WHALE_MD_PREFIX}{date_str}*.md"]
    else:
        patterns = [f"{WHALE_MD_PREFIX}*.md"]

    seen: set[Path] = set()
    for pattern in patterns:
        matches = []
        for path in sorted(base_dir.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            matches.append(path)
        if matches:
            return matches[-1]

    suffix = f" for {date_str}" if date_str else ""
    raise FileNotFoundError(f"Missing {WHALE_MD_PREFIX}YYYY-MM-DD markdown summary{suffix} in {base_dir}")


def load_whale_markdown_symbols(path: Path, ticker_set: set[str] | None = None) -> set[str]:
    """Read ticker mentions from legacy generated whale markdown summaries.

    This is a fallback for historical folders only. New daily runs should stream
    bot-eod-report CSV/ZIP through load_yes_prime_whale_flow instead.
    """
    allowed = {str(t).strip().upper() for t in ticker_set or set()}
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    symbols: set[str] = set()
    active_heading = False
    saw_table_row = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## "):
            active_heading = (
                "Top Symbols by Total Premium" in line
                or "Top 200 Yes-Prime Trades" in line
            )
            saw_table_row = False
            continue
        if not active_heading:
            continue
        if not line.startswith("|"):
            if saw_table_row:
                active_heading = False
            continue

        saw_table_row = True
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        symbol = cells[0].upper()
        if symbol in {"UNDERLYING_SYMBOL", "SYMBOL", "---"} or set(symbol) <= {"-"}:
            continue
        if not WHALE_SYMBOL_RE.fullmatch(symbol):
            continue
        if allowed and symbol not in allowed:
            continue
        symbols.add(symbol)

    return symbols


@contextmanager
def open_bot_eod(path: Path) -> Iterator[tuple[object, str]]:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path)
        try:
            csv_names = sorted(name for name in zf.namelist() if name.lower().endswith(".csv"))
            if not csv_names:
                raise FileNotFoundError(f"No CSV found inside {path}")
            preferred = [name for name in csv_names if Path(name).name.startswith(BOT_EOD_PREFIX)]
            csv_name = preferred[0] if preferred else csv_names[0]
            raw = zf.open(csv_name, "r")
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            try:
                yield text, f"{path}::{csv_name}"
            finally:
                text.close()
                raw.close()
        finally:
            zf.close()
        return

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield handle, str(path)


def _build_width(prices: np.ndarray, tiers: list[dict]) -> np.ndarray:
    width = np.full(len(prices), np.nan, dtype="float64")
    for i, tier in enumerate(tiers):
        min_price = float(tier["min_price"])
        max_price = float(tier["max_price"])
        default_width = float(tier["default_width"])
        if i == len(tiers) - 1:
            mask = (prices >= min_price) & (prices <= max_price)
        else:
            mask = (prices >= min_price) & (prices < max_price)
        width[mask] = default_width
    return width


def _counter_df(counter: Counter, name: str) -> pd.DataFrame:
    return pd.DataFrame(
        sorted(counter.items(), key=lambda item: item[1], reverse=True),
        columns=[name, "count"],
    )


def _empty_symbol_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "underlying_symbol",
            "count",
            "total_premium",
            "call_premium",
            "put_premium",
            "bull_proxy_premium",
            "bear_proxy_premium",
            "credit_premium",
            "debit_premium",
            "max_premium",
            "equity_type",
        ]
    )


def _empty_top_trades() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "underlying_symbol",
            "track",
            "net_type",
            "option_type",
            "side",
            "expiry",
            "dte",
            "underlying_price",
            "strike",
            "price",
            "width",
            "pct_width",
            "size",
            "premium",
            "open_interest",
            "implied_volatility",
            "delta",
            "equity_type",
        ]
    )


def load_yes_prime_whale_flow(
    input_path: Path,
    config: dict,
    *,
    chunksize: int = DEFAULT_CHUNKSIZE,
    top_n: int = DEFAULT_TOP_N,
) -> WhaleFlow:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing bot EOD source: {input_path}")

    gates = config.get("gates", {}) if isinstance(config, dict) else {}
    shield_cfg = config.get("shield", {}) if isinstance(config, dict) else {}
    fire_cfg = config.get("fire", {}) if isinstance(config, dict) else {}

    exclude_etfs = bool(gates.get("exclude_etfs", True))
    exclude_issue_types = {str(t).upper() for t in gates.get("exclude_issue_types", ["ETF"])}
    min_credit_pct = float(gates.get("min_credit_pct_width", 0.25))
    max_credit_pct = float(gates.get("max_credit_pct_width", 0.55))
    max_debit_pct = float(gates.get("max_debit_pct_width", 0.55))
    width_tiers = gates.get("width_tiers", [])
    min_open_interest = int(gates.get("min_leg_open_interest", 0))
    max_strike_dist_pct = float(gates.get("max_strike_distance_pct", 1.0))
    min_premium = float(gates.get("min_whale_premium", 0))

    shield_dte_min, shield_dte_max = shield_cfg.get("dte_range", [28, 56])
    fire_dte_min, fire_dte_max = fire_cfg.get("dte_range", [21, 70])
    use_anchor = bool(shield_cfg.get("use_anchor_whitelist", False))
    anchor_set = {str(sym).upper() for sym in shield_cfg.get("anchor_whitelist", [])}

    total_rows = 0
    yes_prime_rows = 0
    track_counter: Counter = Counter()
    option_type_counter: Counter = Counter()
    side_counter: Counter = Counter()
    symbol_stats: dict[str, dict[str, object]] = {}
    top_trades = _empty_top_trades()

    usecols = [
        "executed_at",
        "underlying_symbol",
        "side",
        "strike",
        "option_type",
        "expiry",
        "underlying_price",
        "price",
        "size",
        "premium",
        "open_interest",
        "equity_type",
        "implied_volatility",
        "delta",
    ]
    dtype = {
        "executed_at": "string",
        "underlying_symbol": "string",
        "side": "string",
        "strike": "float64",
        "option_type": "string",
        "expiry": "string",
        "underlying_price": "float64",
        "price": "float64",
        "size": "float64",
        "premium": "float64",
        "open_interest": "float64",
        "equity_type": "string",
        "implied_volatility": "float64",
        "delta": "float64",
    }

    with open_bot_eod(Path(input_path)) as (input_handle, source_label):
        for chunk in pd.read_csv(
            input_handle,
            chunksize=max(1, int(chunksize)),
            usecols=usecols,
            dtype=dtype,
        ):
            total_rows += len(chunk)
            chunk["side"] = chunk["side"].fillna("no_side").astype("string").str.lower()
            chunk["option_type"] = chunk["option_type"].fillna("unknown").astype("string").str.lower()
            chunk["underlying_symbol"] = chunk["underlying_symbol"].fillna("").astype("string").str.upper()
            chunk["equity_type"] = chunk["equity_type"].fillna("").astype("string")

            executed_date = pd.to_datetime(chunk["executed_at"].str.slice(0, 10), errors="coerce")
            expiry_date = pd.to_datetime(chunk["expiry"], errors="coerce")
            dte = (expiry_date - executed_date).dt.days

            underlying_price = pd.to_numeric(chunk["underlying_price"], errors="coerce").to_numpy()
            price = pd.to_numeric(chunk["price"], errors="coerce").to_numpy()
            width = _build_width(underlying_price, width_tiers)
            pct_width = price / width

            side = chunk["side"].to_numpy()
            net_type = np.where(side == "bid", "credit", "debit")
            track = np.where(net_type == "credit", "SHIELD", "FIRE")

            mask = ~np.isnan(width) & ~np.isnan(pct_width) & dte.notna().to_numpy()

            if exclude_etfs:
                eq_type_upper = chunk["equity_type"].str.upper().to_numpy()
                for issue_type in exclude_issue_types:
                    mask &= eq_type_upper != issue_type

            mask &= (
                ((net_type == "credit") & (pct_width >= min_credit_pct) & (pct_width <= max_credit_pct))
                | ((net_type == "debit") & (pct_width <= max_debit_pct))
            )

            dte_arr = dte.to_numpy()
            mask &= (
                ((track == "SHIELD") & (dte_arr >= shield_dte_min) & (dte_arr <= shield_dte_max))
                | ((track == "FIRE") & (dte_arr >= fire_dte_min) & (dte_arr <= fire_dte_max))
            )

            if use_anchor:
                mask &= (track != "SHIELD") | chunk["underlying_symbol"].isin(anchor_set).to_numpy()

            if min_premium > 0:
                mask &= chunk["premium"].to_numpy() >= min_premium

            if min_open_interest > 0:
                oi_vals = chunk["open_interest"].to_numpy()
                mask &= (oi_vals >= min_open_interest) | np.isnan(oi_vals)

            if max_strike_dist_pct < 1.0:
                strike_vals = chunk["strike"].to_numpy()
                strike_dist = np.abs(strike_vals - underlying_price) / np.where(
                    underlying_price > 0, underlying_price, 1.0
                )
                mask &= (strike_dist <= max_strike_dist_pct) | np.isnan(strike_dist)

            if not mask.any():
                continue

            yes_chunk = chunk.loc[mask].copy()
            yes_chunk["track"] = track[mask]
            yes_chunk["net_type"] = net_type[mask]
            yes_chunk["dte"] = dte[mask]
            yes_chunk["width"] = width[mask]
            yes_chunk["pct_width"] = pct_width[mask]
            yes_prime_rows += len(yes_chunk)

            track_counter.update(yes_chunk["track"].tolist())
            option_type_counter.update(yes_chunk["option_type"].tolist())
            side_counter.update(yes_chunk["side"].tolist())

            premium = pd.to_numeric(yes_chunk["premium"], errors="coerce").fillna(0.0)
            yes_chunk["_call_premium"] = np.where(yes_chunk["option_type"] == "call", premium, 0.0)
            yes_chunk["_put_premium"] = np.where(yes_chunk["option_type"] == "put", premium, 0.0)
            yes_chunk["_credit_premium"] = np.where(yes_chunk["net_type"] == "credit", premium, 0.0)
            yes_chunk["_debit_premium"] = np.where(yes_chunk["net_type"] == "debit", premium, 0.0)
            yes_chunk["_bull_proxy_premium"] = np.where(
                ((yes_chunk["net_type"] == "debit") & (yes_chunk["option_type"] == "call"))
                | ((yes_chunk["net_type"] == "credit") & (yes_chunk["option_type"] == "put")),
                premium,
                0.0,
            )
            yes_chunk["_bear_proxy_premium"] = np.where(
                ((yes_chunk["net_type"] == "debit") & (yes_chunk["option_type"] == "put"))
                | ((yes_chunk["net_type"] == "credit") & (yes_chunk["option_type"] == "call")),
                premium,
                0.0,
            )

            grouped = yes_chunk.groupby("underlying_symbol", dropna=False).agg(
                count=("premium", "size"),
                total_premium=("premium", "sum"),
                call_premium=("_call_premium", "sum"),
                put_premium=("_put_premium", "sum"),
                bull_proxy_premium=("_bull_proxy_premium", "sum"),
                bear_proxy_premium=("_bear_proxy_premium", "sum"),
                credit_premium=("_credit_premium", "sum"),
                debit_premium=("_debit_premium", "sum"),
                max_premium=("premium", "max"),
                equity_type=("equity_type", lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
            )
            for symbol, row in grouped.iterrows():
                stats = symbol_stats.setdefault(
                    str(symbol),
                    {
                        "count": 0,
                        "total_premium": 0.0,
                        "call_premium": 0.0,
                        "put_premium": 0.0,
                        "bull_proxy_premium": 0.0,
                        "bear_proxy_premium": 0.0,
                        "credit_premium": 0.0,
                        "debit_premium": 0.0,
                        "max_premium": 0.0,
                        "equity_type": "",
                    },
                )
                stats["count"] = int(stats["count"]) + int(row["count"])
                for col in [
                    "total_premium",
                    "call_premium",
                    "put_premium",
                    "bull_proxy_premium",
                    "bear_proxy_premium",
                    "credit_premium",
                    "debit_premium",
                ]:
                    stats[col] = float(stats[col]) + float(row[col])
                stats["max_premium"] = max(float(stats["max_premium"]), float(row["max_premium"]))
                if not stats["equity_type"]:
                    stats["equity_type"] = str(row["equity_type"])

            candidate = yes_chunk[_empty_top_trades().columns].copy().nlargest(top_n, "premium")
            if top_trades.empty:
                top_trades = candidate.reset_index(drop=True)
            else:
                top_trades = pd.concat([top_trades, candidate], ignore_index=True)
            top_trades = top_trades.nlargest(top_n, "premium").reset_index(drop=True)

    if symbol_stats:
        symbol_summary = pd.DataFrame(
            [{"underlying_symbol": symbol, **stats} for symbol, stats in symbol_stats.items()]
        )
        symbol_summary = symbol_summary.sort_values("total_premium", ascending=False).reset_index(drop=True)
    else:
        symbol_summary = _empty_symbol_summary()

    if not top_trades.empty:
        top_trades = top_trades.sort_values("premium", ascending=False).reset_index(drop=True)
        for col in ["dte", "size", "open_interest"]:
            top_trades[col] = pd.to_numeric(top_trades[col], errors="coerce").astype("Int64")
        for col in [
            "underlying_price",
            "strike",
            "price",
            "width",
            "pct_width",
            "premium",
            "implied_volatility",
            "delta",
        ]:
            top_trades[col] = pd.to_numeric(top_trades[col], errors="coerce")

    return WhaleFlow(
        source_path=Path(input_path),
        source_label=source_label,
        total_rows=total_rows,
        yes_prime_rows=yes_prime_rows,
        symbol_summary=symbol_summary,
        top_trades=top_trades,
        track_summary=_counter_df(track_counter, "track"),
        option_type_summary=_counter_df(option_type_counter, "option_type"),
        side_summary=_counter_df(side_counter, "side"),
    )
