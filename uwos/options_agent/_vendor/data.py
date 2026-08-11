from __future__ import annotations

import datetime as dt
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .occ import parse_occ_symbol


def safe_float(value: object, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def infer_asof_date(base_dir: Path) -> dt.date:
    try:
        return dt.datetime.strptime(base_dir.name[:10], "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Cannot infer YYYY-MM-DD asof date from {base_dir}") from exc


def dte_from_expiry(value: object, asof: dt.date) -> float:
    if pd.isna(value):
        return math.nan
    if isinstance(value, dt.datetime):
        return float((value.date() - asof).days)
    if isinstance(value, dt.date):
        return float((value - asof).days)
    return math.nan


_SPLIT_PART_RE = re.compile(
    r"^(?P<stem>.+)\.part-(?P<part>\d+)-of-(?P<total>\d+)\.zip$",
    re.IGNORECASE,
)


def _export_candidates(base_dir: Path, prefix: str) -> list[Path]:
    candidates = sorted(base_dir.glob(f"{prefix}*.csv")) + sorted(base_dir.glob(f"{prefix}*.zip"))
    if not candidates:
        unzipped = base_dir / "_unzipped_mode_a"
        candidates = sorted(unzipped.glob(f"{prefix}*.csv")) + sorted(unzipped.glob(f"{prefix}*.zip"))
    if not candidates and prefix == "bot-eod-report-":
        split_dir = base_dir / "bot-eod-split"
        candidates = sorted(split_dir.glob(f"{prefix}*.csv")) + sorted(split_dir.glob(f"{prefix}*.zip"))
    return candidates


def _export_date(path: Path) -> Optional[dt.date]:
    matches = re.findall(r"20\d{2}-\d{2}-\d{2}", path.name)
    parsed: list[dt.date] = []
    for value in matches:
        try:
            parsed.append(dt.date.fromisoformat(value))
        except ValueError:
            continue
    return max(parsed) if parsed else None


def _point_in_time_candidates(candidates: list[Path], asof_ceiling: Optional[dt.date]) -> list[Path]:
    if asof_ceiling is None:
        return candidates
    eligible = [
        path
        for path in candidates
        if (candidate_date := _export_date(path)) is None or candidate_date <= asof_ceiling
    ]
    if eligible:
        return eligible
    dated = sorted(
        f"{path.name}:{_export_date(path)}"
        for path in candidates
    )
    raise FileNotFoundError(
        f"No point-in-time export dated on or before {asof_ceiling.isoformat()}; candidates={dated}"
    )


def _preferred_export(candidates: list[Path]) -> Path:
    live_names = ("latest", "current", "live", "next")
    live_candidates = [path for path in candidates if any(token in path.name.lower() for token in live_names)]
    if live_candidates:
        return sorted(live_candidates, key=lambda path: (path.stat().st_mtime, path.name), reverse=True)[0]
    return candidates[0]


def find_export_bundle(
    base_dir: Path,
    prefix: str,
    *,
    asof_ceiling: Optional[dt.date] = None,
) -> list[Path]:
    """Return one complete export or every validated part of a split ZIP export."""
    candidates = _export_candidates(base_dir, prefix)
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.csv or {prefix}*.zip found under {base_dir}")
    candidates = _point_in_time_candidates(candidates, asof_ceiling)

    unsplit = [path for path in candidates if _SPLIT_PART_RE.match(path.name) is None]
    if unsplit:
        return [_preferred_export(unsplit)]

    groups: dict[tuple[str, int], dict[int, Path]] = {}
    for path in candidates:
        match = _SPLIT_PART_RE.match(path.name)
        if match is None:
            continue
        total = int(match.group("total"))
        part = int(match.group("part"))
        groups.setdefault((match.group("stem"), total), {})[part] = path
    if not groups:
        raise FileNotFoundError(f"No usable {prefix} export found under {base_dir}")

    (stem, total), parts = sorted(groups.items(), key=lambda item: item[0][0])[0]
    missing = sorted(set(range(1, total + 1)) - set(parts))
    if missing:
        raise ValueError(
            f"Incomplete split export {stem}: missing part(s) {missing}; "
            f"found {len(parts)} of {total} under {base_dir}"
        )
    return [parts[index] for index in range(1, total + 1)]


def find_export(base_dir: Path, prefix: str, *, asof_ceiling: Optional[dt.date] = None) -> Path:
    bundle = find_export_bundle(base_dir, prefix, asof_ceiling=asof_ceiling)
    if len(bundle) != 1:
        raise ValueError(
            f"Export {prefix!r} is a {len(bundle)}-part bundle; "
            "use find_export_bundle() so no rows are silently dropped"
        )
    return bundle[0]


def read_csv_export(path: Path, **kwargs) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise FileNotFoundError(f"No CSV member in {path}")
            with zf.open(members[0]) as handle:
                return pd.read_csv(handle, **kwargs)
    return pd.read_csv(path, **kwargs)


def iter_csv_export(path: Path, **kwargs):
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not members:
                raise FileNotFoundError(f"No CSV member in {path}")
            with zf.open(members[0]) as handle:
                yield from pd.read_csv(handle, **kwargs)
    else:
        yield from pd.read_csv(path, **kwargs)


def iter_csv_export_bundle(paths: Iterable[Path], **kwargs):
    for path in paths:
        yield from iter_csv_export(path, **kwargs)


def load_stock_screener(base_dir: Path, *, point_in_time: bool = False) -> pd.DataFrame:
    ceiling = infer_asof_date(base_dir) if point_in_time else None
    path = find_export(base_dir, "stock-screener-", asof_ceiling=ceiling)
    df = read_csv_export(path)
    numeric_cols = [
        "call_volume",
        "put_volume",
        "call_premium",
        "put_premium",
        "bearish_premium",
        "bullish_premium",
        "net_call_premium",
        "net_put_premium",
        "total_open_interest",
        "close",
        "high",
        "low",
        "total_volume",
        "avg30_volume",
        "prev_close",
        "week_52_high",
        "week_52_low",
        "implied_move",
        "implied_move_perc",
        "volatility",
        "iv30d",
        "iv_rank",
        "marketcap",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["flow_total_premium"] = df.get("bullish_premium", 0).fillna(0) + df.get("bearish_premium", 0).fillna(0)
    denom = df["flow_total_premium"].where(df["flow_total_premium"].abs() > 0)
    df["flow_bias"] = (df.get("bullish_premium", 0).fillna(0) - df.get("bearish_premium", 0).fillna(0)) / denom
    df["next_earnings_dt"] = pd.to_datetime(df.get("next_earnings_date", pd.Series(index=df.index)), errors="coerce").dt.date
    return df


def load_hot_chains(base_dir: Path, asof: dt.date, *, point_in_time: bool = False) -> pd.DataFrame:
    path = find_export(base_dir, "hot-chains-", asof_ceiling=asof if point_in_time else None)
    df = read_csv_export(path)
    parsed = df["option_symbol"].map(parse_occ_symbol)
    df["ticker"] = parsed.map(lambda x: x.root if x else "")
    df["expiry_dt"] = parsed.map(lambda x: x.expiry if x else pd.NaT)
    df["right"] = parsed.map(lambda x: x.right if x else "")
    df["strike"] = parsed.map(lambda x: x.strike if x else math.nan)
    df = df[df["ticker"].astype(bool)].copy()
    df["dte"] = df["expiry_dt"].map(lambda x: dte_from_expiry(x, asof))
    for col in ["volume", "open_interest", "premium", "bid", "ask", "iv", "close.1"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["mid"] = (df["bid"].fillna(0) + df["ask"].fillna(0)) / 2.0
    df["spread"] = df["ask"].fillna(0) - df["bid"].fillna(0)
    df["spread_pct_mid"] = df["spread"] / df["mid"].where(df["mid"].abs() > 0)
    df["next_earnings_dt"] = pd.to_datetime(df.get("next_earnings_date", pd.Series(index=df.index)), errors="coerce").dt.date
    return df


def load_chain_oi(base_dir: Path, asof: dt.date, *, point_in_time: bool = False) -> pd.DataFrame:
    path = find_export(base_dir, "chain-oi-changes-", asof_ceiling=asof if point_in_time else None)
    return load_chain_oi_export(path, asof)


def load_chain_oi_export(path: Path, asof: dt.date) -> pd.DataFrame:
    """Load one explicitly selected chain-OI export."""

    path = Path(path).expanduser().resolve()
    df = read_csv_export(path)
    parsed = df["option_symbol"].map(parse_occ_symbol)
    df["ticker"] = parsed.map(lambda x: x.root if x else df.get("underlying_symbol", ""))
    df["expiry_dt"] = parsed.map(lambda x: x.expiry if x else pd.NaT)
    df["right"] = parsed.map(lambda x: x.right if x else "")
    df["strike"] = parsed.map(lambda x: x.strike if x else math.nan)
    df["dte"] = df["expiry_dt"].map(lambda x: dte_from_expiry(x, asof))
    df["next_earnings_dt"] = pd.to_datetime(
        df.get("next_earnings_date", pd.Series(index=df.index)), errors="coerce"
    ).dt.date
    for col in [
        "oi_diff_plain",
        "oi_change",
        "curr_oi",
        "last_oi",
        "volume",
        "last_fill",
        "last_bid",
        "last_ask",
        "prev_total_premium",
        "prev_neutral_volume",
        "prev_mid_volume",
        "prev_bid_volume",
        "prev_ask_volume",
        "prev_stock_multi_leg_volume",
        "prev_multi_leg_volume",
        "curr_vol",
        "prev_vol",
        "trades",
        "avg_price",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.attrs["source_path"] = str(path)
    return df


BOT_FLOW_COLUMNS = [
    "ticker",
    "bot_bull_premium",
    "bot_bear_premium",
    "bot_total_premium",
    "bot_call_ask_premium",
    "bot_call_bid_premium",
    "bot_put_ask_premium",
    "bot_put_bid_premium",
    "bot_multileg_premium",
    "bot_open_interest_sum",
    "bot_volume_sum",
    "bot_unique_expiries",
    "bot_unique_strikes",
    "bot_trades",
    "bot_flow_bias",
    "bot_multileg_ratio",
    "bot_volume_oi_ratio",
]


DARK_POOL_FLOW_COLUMNS = [
    "ticker",
    "dp_bull_premium",
    "dp_bear_premium",
    "dp_neutral_premium",
    "dp_directional_premium",
    "dp_total_premium",
    "dp_flow_bias",
    "dp_directional_ratio",
    "dp_prints",
]


def aggregate_dark_pool_flow(
    base_dir: Path,
    tickers: Iterable[str],
    *,
    chunksize: int = 750_000,
    max_rows: int | None = None,
    allow_missing: bool = False,
    point_in_time: bool = False,
) -> pd.DataFrame:
    """Aggregate equity dark-pool prints by ticker without inventing option flow.

    A print above the contemporaneous NBBO midpoint is buyer-initiated, a print
    below it is seller-initiated, and midpoint/invalid-NBBO prints are neutral.
    The result remains a separate equity confirmation signal; callers must not
    substitute it for side-aware option flow.
    """

    ceiling = infer_asof_date(base_dir) if point_in_time else None
    try:
        paths = find_export_bundle(base_dir, "dp-eod-report-", asof_ceiling=ceiling)
    except FileNotFoundError:
        if not allow_missing:
            raise
        out = pd.DataFrame(columns=DARK_POOL_FLOW_COLUMNS)
        out.attrs["source_status"] = "missing_dp_eod"
        out.attrs["source_path"] = ""
        return out

    wanted = {str(t).upper().strip() for t in tickers if str(t).strip()}
    usecols = [
        "ticker",
        "nbbo_ask",
        "nbbo_bid",
        "size",
        "premium",
        "price",
        "canceled",
    ]
    rows_seen = 0
    parts: list[pd.DataFrame] = []
    for chunk in iter_csv_export_bundle(paths, usecols=usecols, chunksize=chunksize):
        rows_seen += len(chunk)
        chunk["ticker"] = chunk["ticker"].astype(str).str.upper().str.strip()
        if wanted:
            chunk = chunk[chunk["ticker"].isin(wanted)]
        if "canceled" in chunk.columns:
            canceled = chunk["canceled"].astype(str).str.strip().str.lower()
            chunk = chunk[~canceled.isin({"1", "true", "t", "yes", "y"})]
        if chunk.empty:
            if max_rows and rows_seen >= max_rows:
                break
            continue

        for column in ["nbbo_ask", "nbbo_bid", "size", "premium", "price"]:
            chunk[column] = pd.to_numeric(chunk.get(column), errors="coerce")
        fallback_premium = chunk["price"] * chunk["size"]
        chunk["dp_premium"] = chunk["premium"].where(chunk["premium"].gt(0), fallback_premium).fillna(0.0)
        valid_nbbo = (
            chunk["nbbo_bid"].gt(0)
            & chunk["nbbo_ask"].ge(chunk["nbbo_bid"])
            & chunk["price"].gt(0)
        )
        midpoint = (chunk["nbbo_bid"] + chunk["nbbo_ask"]) / 2.0
        bull = valid_nbbo & chunk["price"].gt(midpoint)
        bear = valid_nbbo & chunk["price"].lt(midpoint)
        chunk["dp_bull_premium"] = chunk["dp_premium"].where(bull, 0.0)
        chunk["dp_bear_premium"] = chunk["dp_premium"].where(bear, 0.0)
        chunk["dp_neutral_premium"] = chunk["dp_premium"].where(~(bull | bear), 0.0)
        chunk["dp_prints"] = 1
        parts.append(
            chunk.groupby("ticker", as_index=False).agg(
                dp_bull_premium=("dp_bull_premium", "sum"),
                dp_bear_premium=("dp_bear_premium", "sum"),
                dp_neutral_premium=("dp_neutral_premium", "sum"),
                dp_total_premium=("dp_premium", "sum"),
                dp_prints=("dp_prints", "sum"),
            )
        )
        if max_rows and rows_seen >= max_rows:
            break

    if not parts:
        out = pd.DataFrame(columns=DARK_POOL_FLOW_COLUMNS)
        out.attrs["source_status"] = "dp_eod_no_matching_rows"
        out.attrs["source_path"] = ";".join(str(path) for path in paths)
        return out

    out = pd.concat(parts, ignore_index=True).groupby("ticker", as_index=False).sum()
    out["dp_directional_premium"] = out["dp_bull_premium"] + out["dp_bear_premium"]
    directional = out["dp_directional_premium"].where(out["dp_directional_premium"].gt(0))
    total = out["dp_total_premium"].where(out["dp_total_premium"].gt(0))
    out["dp_flow_bias"] = (out["dp_bull_premium"] - out["dp_bear_premium"]) / directional
    out["dp_directional_ratio"] = out["dp_directional_premium"] / total
    out.attrs["source_status"] = "dp_eod_split_bundle_loaded" if len(paths) > 1 else "dp_eod_loaded"
    out.attrs["source_path"] = ";".join(str(path) for path in paths)
    return out[DARK_POOL_FLOW_COLUMNS]


def _empty_bot_flow(*, source_status: str, source_path: str = "", dp_equity_present: bool = False) -> pd.DataFrame:
    out = pd.DataFrame(columns=BOT_FLOW_COLUMNS)
    out.attrs["source_status"] = source_status
    out.attrs["source_path"] = source_path
    out.attrs["dp_equity_present"] = bool(dp_equity_present)
    return out


def aggregate_bot_flow(
    base_dir: Path,
    tickers: Iterable[str],
    *,
    chunksize: int = 750_000,
    max_rows: int | None = None,
    allow_missing: bool = False,
    point_in_time: bool = False,
) -> pd.DataFrame:
    ceiling = infer_asof_date(base_dir) if point_in_time else None
    try:
        paths = find_export_bundle(base_dir, "bot-eod-report-", asof_ceiling=ceiling)
    except FileNotFoundError:
        if not allow_missing:
            raise
        try:
            dp_path = find_export(base_dir, "dp-eod-report-", asof_ceiling=ceiling)
        except FileNotFoundError:
            dp_path = None
        # DP is equity dark-pool data and cannot be substituted for side-aware
        # option flow. Keep the date usable while explicitly marking degraded
        # evidence instead of fabricating bot-flow semantics.
        return _empty_bot_flow(
            source_status="missing_bot_eod_dp_equity_only" if dp_path else "missing_bot_eod",
            source_path=str(dp_path or ""),
            dp_equity_present=dp_path is not None,
        )
    wanted = {str(t).upper().strip() for t in tickers if str(t).strip()}
    usecols = [
        "underlying_symbol",
        "side",
        "option_type",
        "expiry",
        "strike",
        "premium",
        "size",
        "volume",
        "open_interest",
        "delta",
        "canceled",
        "report_flags",
        "upstream_condition_detail",
    ]
    rows_seen = 0
    parts = []
    for chunk in iter_csv_export_bundle(paths, usecols=usecols, chunksize=chunksize):
        rows_seen += len(chunk)
        chunk["underlying_symbol"] = chunk["underlying_symbol"].astype(str).str.upper().str.strip()
        if wanted:
            chunk = chunk[chunk["underlying_symbol"].isin(wanted)]
        if "canceled" in chunk.columns:
            chunk = chunk[chunk["canceled"].astype(str).str.lower().ne("t")]
        if chunk.empty:
            if max_rows and rows_seen >= max_rows:
                break
            continue
        chunk["bot_premium"] = pd.to_numeric(chunk["premium"], errors="coerce").fillna(0)
        side = chunk["side"].astype(str).str.lower()
        opt_type = chunk["option_type"].astype(str).str.lower()
        call_ask_mask = (opt_type == "call") & (side == "ask")
        call_bid_mask = (opt_type == "call") & (side == "bid")
        put_ask_mask = (opt_type == "put") & (side == "ask")
        put_bid_mask = (opt_type == "put") & (side == "bid")
        bull_mask = call_ask_mask | put_bid_mask
        bear_mask = call_bid_mask | put_ask_mask
        flags = chunk.get("report_flags", pd.Series("", index=chunk.index)).astype(str).str.lower()
        condition = chunk.get("upstream_condition_detail", pd.Series("", index=chunk.index)).astype(str).str.lower()
        multi_mask = flags.str.contains("multi|spread|floor|cross", regex=True) | condition.str.contains(
            "multi|spread|floor|cross", regex=True
        )
        chunk["bot_bull_premium"] = chunk["bot_premium"].where(bull_mask, 0.0)
        chunk["bot_bear_premium"] = chunk["bot_premium"].where(bear_mask, 0.0)
        chunk["bot_call_ask_premium"] = chunk["bot_premium"].where(call_ask_mask, 0.0)
        chunk["bot_call_bid_premium"] = chunk["bot_premium"].where(call_bid_mask, 0.0)
        chunk["bot_put_ask_premium"] = chunk["bot_premium"].where(put_ask_mask, 0.0)
        chunk["bot_put_bid_premium"] = chunk["bot_premium"].where(put_bid_mask, 0.0)
        chunk["bot_multileg_premium"] = chunk["bot_premium"].where(multi_mask, 0.0)
        chunk["bot_open_interest_sum"] = pd.to_numeric(chunk.get("open_interest"), errors="coerce").fillna(0)
        chunk["bot_volume_sum"] = pd.to_numeric(chunk.get("volume"), errors="coerce").fillna(0)
        chunk["bot_unique_expiries"] = chunk["expiry"].astype(str)
        chunk["bot_unique_strikes"] = pd.to_numeric(chunk.get("strike"), errors="coerce")
        chunk["bot_trades"] = 1
        agg = chunk.groupby("underlying_symbol", as_index=False).agg(
            bot_bull_premium=("bot_bull_premium", "sum"),
            bot_bear_premium=("bot_bear_premium", "sum"),
            bot_total_premium=("bot_premium", "sum"),
            bot_call_ask_premium=("bot_call_ask_premium", "sum"),
            bot_call_bid_premium=("bot_call_bid_premium", "sum"),
            bot_put_ask_premium=("bot_put_ask_premium", "sum"),
            bot_put_bid_premium=("bot_put_bid_premium", "sum"),
            bot_multileg_premium=("bot_multileg_premium", "sum"),
            bot_open_interest_sum=("bot_open_interest_sum", "sum"),
            bot_volume_sum=("bot_volume_sum", "sum"),
            bot_unique_expiries=("bot_unique_expiries", "nunique"),
            bot_unique_strikes=("bot_unique_strikes", "nunique"),
            bot_trades=("bot_trades", "sum"),
        )
        parts.append(agg)
        if max_rows and rows_seen >= max_rows:
            break
    if not parts:
        return _empty_bot_flow(
            source_status="bot_eod_no_matching_rows",
            source_path=";".join(str(path) for path in paths),
        )
    out = pd.concat(parts, ignore_index=True).groupby("underlying_symbol", as_index=False).sum()
    out = out.rename(columns={"underlying_symbol": "ticker"})
    denom = out["bot_total_premium"].where(out["bot_total_premium"].abs() > 0)
    out["bot_flow_bias"] = (out["bot_bull_premium"] - out["bot_bear_premium"]) / denom
    out["bot_multileg_ratio"] = out["bot_multileg_premium"] / denom
    out["bot_volume_oi_ratio"] = out["bot_volume_sum"] / out["bot_open_interest_sum"].where(out["bot_open_interest_sum"].abs() > 0)
    out.attrs["source_status"] = "bot_eod_split_bundle_loaded" if len(paths) > 1 else "bot_eod_loaded"
    out.attrs["source_path"] = ";".join(str(path) for path in paths)
    out.attrs["dp_equity_present"] = False
    return out
