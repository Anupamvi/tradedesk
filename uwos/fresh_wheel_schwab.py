from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from uwos.schwab_auth import SchwabAuthConfig, SchwabLiveDataService

REPLAY_BLOCKED_CSP: dict[str, str] = {
    "ORCL": "2026 YTD replay tail loss: 5 scored CSPs, 40% hit rate, -$4,224.50 total PnL",
    "PG": "2026 YTD replay tail loss: one scored CSP, -$1,080.00 PnL",
}

REPLAY_CAUTION_CSP: dict[str, str] = {
    "JPM": "2026 YTD replay caution: profitable frequency but net negative after one large loss",
    "NFLX": "2026 YTD replay caution: 50% hit rate and slightly negative total PnL",
}


def _today() -> dt.date:
    return dt.date.today()


def sanitize_error(value: Any) -> str:
    text = str(value)
    text = re.sub(r"apikey=[^&'\"\s]+", "apikey=REDACTED", text)
    return text


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or value == "":
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    number = safe_float(value)
    if not math.isfinite(number):
        return default
    return int(number)


def coerce_date(value: Any) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return dt.datetime.strptime(text[:10], fmt).date()
        except ValueError:
            pass
    try:
        return dt.datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if value in {"BRK.B", "BRK-B"}:
        return "BRK/B"
    return value


def schwab_symbol(symbol: str) -> str:
    value = normalize_symbol(symbol)
    if value == "BRK/B":
        return "BRK/B"
    return value


def latest_usable_uw_folder(root: Path) -> Path:
    dated = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        try:
            day = dt.datetime.strptime(path.name[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        has_required = bool(list(path.glob("stock-screener-*.csv")) or list(path.glob("stock-screener-*.zip")))
        has_required = has_required and bool(list(path.glob("hot-chains-*.csv")) or list(path.glob("hot-chains-*.zip")))
        if has_required:
            dated.append((day, path))
    if not dated:
        raise FileNotFoundError(f"No usable dated UW folder with stock-screener and hot-chains under {root}")
    return sorted(dated)[-1][1]


def find_export(base_dir: Path, prefix: str) -> Path:
    direct = sorted(base_dir.glob(f"{prefix}*.csv")) + sorted(base_dir.glob(f"{prefix}*.zip"))
    if direct:
        return direct[0]
    unzipped = base_dir / "_unzipped_mode_a"
    nested = sorted(unzipped.glob(f"{prefix}*.csv")) + sorted(unzipped.glob(f"{prefix}*.zip"))
    if nested:
        return nested[0]
    raise FileNotFoundError(f"No {prefix}*.csv or {prefix}*.zip found under {base_dir}")


def read_csv_export(path: Path) -> pd.DataFrame:
    if path.suffix.lower() != ".zip":
        return pd.read_csv(path)
    with zipfile.ZipFile(path) as zf:
        members = [name for name in zf.namelist() if name.lower().endswith(".csv")]
        if not members:
            raise FileNotFoundError(f"No CSV member inside {path}")
        with zf.open(members[0]) as handle:
            return pd.read_csv(handle)


def load_screener(base_dir: Path) -> pd.DataFrame:
    path = find_export(base_dir, "stock-screener-")
    df = read_csv_export(path)
    if "ticker" not in df.columns:
        raise ValueError(f"{path} is missing ticker column")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for col in [
        "marketcap",
        "close",
        "prev_close",
        "high",
        "low",
        "week_52_high",
        "week_52_low",
        "total_volume",
        "avg30_volume",
        "total_open_interest",
        "call_premium",
        "put_premium",
        "bullish_premium",
        "bearish_premium",
        "net_call_premium",
        "net_put_premium",
        "iv30d",
        "iv_rank",
        "volatility",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "next_earnings_date" in df.columns:
        df["next_earnings_dt"] = df["next_earnings_date"].map(coerce_date)
    else:
        df["next_earnings_dt"] = None
    return df


def load_hot_chain_summary(base_dir: Path) -> pd.DataFrame:
    path = find_export(base_dir, "hot-chains-")
    usecols = [
        "option_symbol",
        "volume",
        "open_interest",
        "premium",
        "bid",
        "ask",
        "iv",
        "ask_side_volume",
        "bid_side_volume",
        "ticker_option_vol",
        "close",
    ]
    df = read_csv_export(path)
    available = [col for col in usecols if col in df.columns]
    df = df[available].copy()
    if "option_symbol" not in df.columns:
        return pd.DataFrame(columns=["ticker", "hot_volume", "hot_premium", "hot_open_interest"])
    df["ticker"] = df["option_symbol"].astype(str).str.extract(r"^([A-Z./-]+)\d{6}[CP]\d{8}", expand=False)
    df["ticker"] = df["ticker"].fillna("").astype(str).str.upper().str.strip()
    for col in ["volume", "open_interest", "premium", "ask_side_volume", "bid_side_volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if df.empty:
        return pd.DataFrame(columns=["ticker", "hot_volume", "hot_premium", "hot_open_interest"])
    grouped = df.groupby("ticker", as_index=False).agg(
        hot_volume=("volume", "sum"),
        hot_premium=("premium", "sum"),
        hot_open_interest=("open_interest", "max"),
        hot_ask_side_volume=("ask_side_volume", "sum"),
        hot_bid_side_volume=("bid_side_volume", "sum"),
    )
    denom = (grouped["hot_ask_side_volume"] + grouped["hot_bid_side_volume"]).where(lambda s: s > 0)
    grouped["hot_ask_bias"] = (grouped["hot_ask_side_volume"] - grouped["hot_bid_side_volume"]) / denom
    return grouped


@dataclass
class WheelConfig:
    account_size: float = 250_000.0
    target_monthly_income_low: float = 10_000.0
    target_monthly_income_high: float = 20_000.0
    max_symbols: int = 20
    strike_count: int = 80
    min_market_cap: float = 25_000_000_000.0
    min_avg30_volume: float = 1_000_000.0
    min_total_option_oi: float = 50_000.0
    min_option_open_interest: int = 100
    min_option_volume: int = 5
    min_mid: float = 0.35
    max_spread_pct: float = 0.22
    csp_dte_min: int = 25
    csp_dte_max: int = 60
    short_put_delta_min: float = 0.10
    short_put_delta_max: float = 0.30
    covered_call_dte_min: int = 20
    covered_call_dte_max: int = 50
    short_call_delta_min: float = 0.10
    short_call_delta_max: float = 0.30
    pmcc_long_dte_min: int = 120
    pmcc_long_dte_max: int = 420
    pmcc_long_delta_min: float = 0.70
    pmcc_long_delta_max: float = 0.90
    pmcc_short_dte_min: int = 21
    pmcc_short_dte_max: int = 50
    pmcc_max_debit_to_width: float = 0.75
    enable_covered_strangles: bool = True
    covered_strangle_dte_min: int = 25
    covered_strangle_dte_max: int = 60
    covered_strangle_min_put_discount_pct: float = 4.0
    covered_strangle_min_range_width_pct: float = 10.0
    enable_csp_call_overlay: bool = True
    upside_overlay_dte_min: int = 25
    upside_overlay_dte_max: int = 90
    upside_overlay_delta_min: float = 0.10
    upside_overlay_delta_max: float = 0.25
    upside_overlay_max_credit_pct: float = 0.35
    upside_overlay_min_net_credit_pct: float = 0.60
    upside_overlay_min_flow_score: float = 55.0
    enable_leaps_covered_strangles: bool = True
    leaps_strangle_min_flow_score: float = 50.0
    leaps_strangle_min_put_discount_pct: float = 3.0
    enable_tactical_sleeve: bool = True
    tactical_sleeve_pct: float = 0.15
    tactical_single_name_cash_pct: float = 0.05
    quality_symbol_quota_pct: float = 0.40
    tactical_symbol_quota_pct: float = 0.30
    premium_symbol_quota_pct: float = 0.30
    tactical_min_symbols: int = 0
    tactical_min_confidence: float = 68.0
    tactical_min_yield_pct: float = 1.50
    tactical_min_discount_pct: float = 4.0
    tactical_min_flow_score: float = 49.0
    tactical_min_score: float = 55.0
    avoid_earnings_days: int = 7
    avoid_expiry_through_earnings: bool = True
    max_single_name_cash_pct: float = 0.20
    max_new_csp_cash_pct: float = 0.70
    max_pmcc_debit_pct: float = 0.15
    cash_buffer_pct: float = 0.10
    risk_free_rate: float = 0.04


@dataclass
class UniverseRow:
    ticker: str
    full_name: str
    sector: str
    issue_type: str
    close: float
    marketcap: float
    avg30_volume: float
    total_open_interest: float
    next_earnings: dt.date | None
    quality_score: float
    flow_score: float
    thesis: str
    tier: int
    total_premium: float = 0.0
    iv30d: float = 0.0
    tactical_score: float = 0.0
    selection_lane: str = ""
    reasons: list[str] = field(default_factory=list)


@dataclass
class OptionContract:
    symbol: str
    expiry: dt.date
    dte: int
    right: str
    strike: float
    bid: float
    ask: float
    mark: float
    mid: float
    delta: float
    iv: float
    open_interest: int
    volume: int
    spread_pct: float


@dataclass
class Position:
    symbol: str
    shares: int
    avg_cost: float


@dataclass
class WheelAction:
    ticker: str
    action: str
    confidence: float
    spot: float
    quality_score: float
    flow_score: float
    estimated_credit: float = 0.0
    monthly_credit_runrate: float = 0.0
    cash_required: float = 0.0
    pmcc_debit: float = 0.0
    contracts: int = 0
    alert_price: float | None = None
    option_symbol: str = ""
    expiry: dt.date | None = None
    strike: float | None = None
    limit_price: float | None = None
    long_option_symbol: str = ""
    long_expiry: dt.date | None = None
    long_strike: float | None = None
    long_limit_price: float | None = None
    paired_option_symbol: str = ""
    paired_expiry: dt.date | None = None
    paired_strike: float | None = None
    paired_limit_price: float | None = None
    sleeve: str = "core"
    thesis: str = ""
    reasons: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


def objective_ownership_tier(
    *,
    marketcap: float,
    avg30_volume: float,
    total_open_interest: float,
    iv30d: float,
    issue_type: str,
) -> int:
    if issue_type.upper() == "ETF":
        return 4
    iv_decimal = iv30d / 100.0 if iv30d > 3.0 else iv30d
    moderate_iv = 0.0 < iv_decimal <= 0.70
    if moderate_iv and marketcap >= 500_000_000_000 and avg30_volume >= 10_000_000 and total_open_interest >= 2_000_000:
        return 1
    if moderate_iv and marketcap >= 100_000_000_000 and avg30_volume >= 1_000_000 and total_open_interest >= 250_000:
        return 2
    if marketcap >= 25_000_000_000 and avg30_volume >= 1_000_000 and total_open_interest >= 50_000:
        return 3
    return 4


def score_universe_row(row: pd.Series, hot_row: dict[str, Any], config: WheelConfig) -> UniverseRow:
    ticker = normalize_symbol(str(row.get("ticker", "")))
    marketcap = safe_float(row.get("marketcap"), 0.0)
    close = safe_float(row.get("close"), 0.0)
    avg30_volume = safe_float(row.get("avg30_volume"), 0.0)
    total_oi = safe_float(row.get("total_open_interest"), 0.0)
    iv30d = safe_float(row.get("iv30d"), 0.0)
    issue_type = str(row.get("issue_type") or "").strip()
    sector = str(row.get("sector") or "").strip()
    full_name = str(row.get("full_name") or ticker).strip()
    next_earnings = row.get("next_earnings_dt")
    if not isinstance(next_earnings, dt.date):
        next_earnings = None

    score = 0.0
    reasons: list[str] = []
    tier = objective_ownership_tier(
        marketcap=marketcap,
        avg30_volume=avg30_volume,
        total_open_interest=total_oi,
        iv30d=iv30d,
        issue_type=issue_type,
    )
    score += {1: 30.0, 2: 20.0, 3: 10.0}.get(tier, 0.0)
    if tier <= 3:
        reasons.append(f"objective ownership tier {tier} from size, liquidity, and IV")
    if marketcap >= 500_000_000_000:
        score += 22.0
        reasons.append("mega-cap durability")
    elif marketcap >= 100_000_000_000:
        score += 18.0
        reasons.append("large-cap durability")
    elif marketcap >= config.min_market_cap:
        score += 12.0
        reasons.append("passes market-cap floor")
    if avg30_volume >= 10_000_000:
        score += 12.0
        reasons.append("deep share liquidity")
    elif avg30_volume >= config.min_avg30_volume:
        score += 8.0
        reasons.append("passes share-liquidity floor")
    if total_oi >= 2_000_000:
        score += 12.0
        reasons.append("very deep listed-option open interest")
    elif total_oi >= config.min_total_option_oi:
        score += 8.0
        reasons.append("passes option-open-interest floor")
    if close >= 20.0:
        score += 5.0
    if str(row.get("is_index") or "").lower() == "t":
        score -= 20.0
        reasons.append("index/ETF de-prioritized for stock-ownership wheel")
    if issue_type.upper() == "ETF":
        score -= 10.0
        reasons.append("ETF de-prioritized versus ownable operating companies")

    bullish_premium = max(safe_float(row.get("bullish_premium"), 0.0), 0.0)
    bearish_premium = max(safe_float(row.get("bearish_premium"), 0.0), 0.0)
    total_premium = bullish_premium + bearish_premium
    flow_bias = (bullish_premium - bearish_premium) / total_premium if total_premium > 0 else 0.0
    hot_bias = safe_float(hot_row.get("hot_ask_bias"), 0.0)
    flow_score = max(0.0, min(100.0, 50.0 + flow_bias * 25.0 + hot_bias * 15.0))
    score += max(-6.0, min(8.0, (flow_score - 50.0) / 5.0))
    quality_score = max(0.0, min(100.0, score))
    iv_for_score = iv30d * 100.0 if 0.0 < iv30d <= 3.0 else iv30d
    premium_score = min(math.log1p(total_premium) / math.log1p(2_000_000_000.0) * 26.0, 26.0)
    iv_score = min(max(iv_for_score, 0.0) / 80.0 * 18.0, 18.0)
    option_liquidity_score = min(math.log1p(total_oi) / math.log1p(2_000_000.0) * 12.0, 12.0)
    share_liquidity_score = min(math.log1p(avg30_volume) / math.log1p(50_000_000.0) * 8.0, 8.0)
    operating_bonus = -6.0 if issue_type.upper() == "ETF" else 4.0
    tactical_score = max(
        0.0,
        min(
            100.0,
            flow_score * 0.45
            + premium_score
            + iv_score
            + option_liquidity_score
            + share_liquidity_score
            + operating_bonus,
        ),
    )
    if tier > 2 and flow_score >= config.tactical_min_flow_score and tactical_score >= config.tactical_min_score:
        reasons.append("tactical premium/flow candidate")

    return UniverseRow(
        ticker=ticker,
        full_name=full_name,
        sector=sector,
        issue_type=issue_type,
        close=close,
        marketcap=marketcap,
        avg30_volume=avg30_volume,
        total_open_interest=total_oi,
        next_earnings=next_earnings,
        quality_score=round(quality_score, 1),
        flow_score=round(flow_score, 1),
        thesis=f"{full_name} ranked from UW size, liquidity, premium, and flow data",
        tier=tier,
        total_premium=round(total_premium, 2),
        iv30d=round(iv30d, 4),
        tactical_score=round(tactical_score, 1),
        reasons=reasons,
    )


def is_tactical_universe_row(row: UniverseRow, config: WheelConfig) -> bool:
    if not config.enable_tactical_sleeve:
        return False
    if row.selection_lane == "position" or row.tier <= 2 or row.ticker in REPLAY_BLOCKED_CSP:
        return False
    if row.issue_type.upper() == "ETF":
        return False
    if row.flow_score < config.tactical_min_flow_score:
        return False
    return row.tactical_score >= config.tactical_min_score


def build_universe(
    base_dir: Path,
    config: WheelConfig,
    position_symbols: Iterable[str] = (),
) -> list[UniverseRow]:
    screener = load_screener(base_dir)
    hot = load_hot_chain_summary(base_dir)
    hot_by_ticker = {normalize_symbol(str(row["ticker"])): row for _, row in hot.iterrows()} if not hot.empty else {}
    managed_symbols = {normalize_symbol(symbol) for symbol in position_symbols if normalize_symbol(symbol)}
    rows: list[UniverseRow] = []
    for _, row in screener.iterrows():
        ticker = normalize_symbol(str(row.get("ticker", "")))
        if not ticker:
            continue
        item = score_universe_row(row, hot_by_ticker.get(ticker, {}), config)
        is_managed_position = item.ticker in managed_symbols and item.issue_type.upper() in {"COMMON STOCK", "ADR", "ETF"}
        if not is_managed_position:
            if item.close < 20.0:
                continue
            if item.marketcap < config.min_market_cap:
                continue
            if item.avg30_volume < config.min_avg30_volume:
                continue
            if item.total_open_interest < config.min_total_option_oi:
                continue
        rows.append(item)

    quality = sorted(
        [item for item in rows if item.tier <= 2 and item.issue_type.upper() != "ETF"],
        key=lambda item: (item.quality_score, item.total_premium, item.flow_score, item.marketcap),
        reverse=True,
    )
    tactical = sorted(
        [item for item in rows if is_tactical_universe_row(item, config)],
        key=lambda item: (item.tactical_score, item.total_premium, item.flow_score, item.marketcap),
        reverse=True,
    )
    premium = sorted(
        [item for item in rows if item.issue_type.upper() != "ETF" and item.total_premium > 0.0],
        key=lambda item: (item.total_premium, item.quality_score, item.tactical_score, item.marketcap),
        reverse=True,
    )
    ranked = sorted(
        rows,
        key=lambda item: (max(item.quality_score, item.tactical_score), item.total_premium, item.marketcap),
        reverse=True,
    )
    quality_quota = min(round(config.max_symbols * config.quality_symbol_quota_pct), config.max_symbols)
    tactical_quota = min(
        max(config.tactical_min_symbols, round(config.max_symbols * config.tactical_symbol_quota_pct)),
        config.max_symbols - quality_quota,
    )
    premium_quota = min(
        round(config.max_symbols * config.premium_symbol_quota_pct),
        config.max_symbols - quality_quota - tactical_quota,
    )
    quality_quota += config.max_symbols - quality_quota - tactical_quota - premium_quota
    selected: list[UniverseRow] = []
    seen: set[str] = set()

    def add_items(items: Sequence[UniverseRow], lane: str, limit: int | None = None) -> None:
        added = 0
        for item in items:
            if len(selected) >= config.max_symbols:
                return
            if limit is not None and added >= limit:
                return
            if item.ticker in seen:
                continue
            seen.add(item.ticker)
            item.selection_lane = lane
            item.reasons.append(f"selected by objective {lane} lane")
            selected.append(item)
            added += 1

    add_items(quality, "quality", quality_quota)
    add_items(tactical, "tactical", tactical_quota)
    add_items(premium, "premium", premium_quota)
    add_items(ranked, "ranked")

    for item in rows:
        if item.ticker not in managed_symbols or item.ticker in seen:
            continue
        item.selection_lane = "position"
        item.reasons.append("held round-lot position appended outside candidate limit")
        selected.append(item)
        seen.add(item.ticker)

    for ticker in sorted(managed_symbols - seen):
        selected.append(
            UniverseRow(
                ticker=ticker,
                full_name=ticker,
                sector="",
                issue_type="Common Stock",
                close=0.0,
                marketcap=0.0,
                avg30_volume=0.0,
                total_open_interest=0.0,
                next_earnings=None,
                quality_score=0.0,
                flow_score=0.0,
                thesis=f"{ticker} held position missing from dated UW screener",
                tier=4,
                selection_lane="position",
                reasons=["held round-lot position missing from dated UW screener; contract evaluation withheld"],
            )
        )
    return selected


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def bs_delta(spot: float, strike: float, dte: int, iv: float, is_call: bool, risk_free_rate: float) -> float:
    if spot <= 0 or strike <= 0 or dte <= 0 or iv <= 0:
        return math.nan
    t = max(dte / 365.0, 1.0 / 365.0)
    vol = iv / 100.0 if iv > 3.0 else iv
    try:
        d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    except (ValueError, ZeroDivisionError):
        return math.nan
    if is_call:
        return _normal_cdf(d1)
    return _normal_cdf(d1) - 1.0


def chain_spot(chain: dict[str, Any], quote: dict[str, Any] | None = None) -> float:
    underlying = chain.get("underlying", {}) if isinstance(chain, dict) else {}
    quote_body = (quote or {}).get("quote", quote or {})
    for value in [
        chain.get("underlyingPrice"),
        underlying.get("mark"),
        underlying.get("last"),
        underlying.get("lastPrice"),
        quote_body.get("lastPrice"),
        quote_body.get("mark"),
    ]:
        number = safe_float(value)
        if math.isfinite(number) and number > 0:
            return number
    return math.nan


def iter_option_contracts(
    chain: dict[str, Any],
    right: str,
    asof: dt.date,
    risk_free_rate: float,
    spot: float,
) -> Iterable[OptionContract]:
    map_name = "callExpDateMap" if right == "C" else "putExpDateMap"
    for exp_key, strike_map in (chain.get(map_name, {}) or {}).items():
        expiry_text = str(exp_key).split(":")[0]
        expiry = coerce_date(expiry_text)
        if expiry is None:
            continue
        dte = (expiry - asof).days
        if dte <= 0:
            continue
        for strike_key, contracts in (strike_map or {}).items():
            for contract in contracts or []:
                strike = safe_float(contract.get("strikePrice"), safe_float(strike_key))
                bid = safe_float(contract.get("bid"), 0.0)
                ask = safe_float(contract.get("ask"), 0.0)
                mark = safe_float(contract.get("mark"))
                last = safe_float(contract.get("last"))
                if bid > 0 and ask > 0:
                    mid = (bid + ask) / 2.0
                elif math.isfinite(mark) and mark > 0:
                    mid = mark
                elif math.isfinite(last) and last > 0:
                    mid = last
                else:
                    mid = 0.0
                spread_pct = (ask - bid) / mid if mid > 0 and ask >= bid else 999.0
                iv = safe_float(contract.get("volatility"), safe_float(contract.get("impliedVolatility"), math.nan))
                delta = safe_float(contract.get("delta"))
                if not math.isfinite(delta) or delta == 0:
                    delta = bs_delta(spot, strike, dte, iv, right == "C", risk_free_rate)
                yield OptionContract(
                    symbol=str(contract.get("symbol") or ""),
                    expiry=expiry,
                    dte=dte,
                    right=right,
                    strike=strike,
                    bid=bid,
                    ask=ask,
                    mark=mark if math.isfinite(mark) else mid,
                    mid=mid,
                    delta=delta,
                    iv=iv,
                    open_interest=safe_int(contract.get("openInterest")),
                    volume=safe_int(contract.get("totalVolume")),
                    spread_pct=spread_pct,
                )


def liquid_contracts(contracts: Iterable[OptionContract], config: WheelConfig) -> list[OptionContract]:
    out = []
    for contract in contracts:
        if contract.mid < config.min_mid:
            continue
        if contract.open_interest < config.min_option_open_interest:
            continue
        if contract.volume < config.min_option_volume:
            continue
        if contract.spread_pct > config.max_spread_pct:
            continue
        out.append(contract)
    return out


def annualized_yield(credit: float, capital: float, dte: int) -> float:
    if credit <= 0 or capital <= 0 or dte <= 0:
        return 0.0
    return credit / capital * 365.0 / dte * 100.0


def limit_for_sell(contract: OptionContract) -> float:
    if contract.bid > 0 and contract.ask > 0:
        return round(max(contract.bid, contract.mid * 0.95), 2)
    return round(contract.mid, 2)


def limit_for_buy(contract: OptionContract) -> float:
    if contract.ask > 0:
        return round(contract.ask, 2)
    return round(contract.mid, 2)


def earnings_blocks_expiry(earnings: dt.date | None, asof: dt.date, expiry: dt.date, config: WheelConfig) -> bool:
    return bool(config.avoid_expiry_through_earnings and earnings is not None and asof <= earnings <= expiry)


def get_live_chain_with_fallback(
    service: SchwabLiveDataService,
    symbol: str,
    *,
    live_date: dt.date,
    config: WheelConfig,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    to_date = live_date + dt.timedelta(days=min(config.pmcc_long_dte_max + 30, 365))
    try:
        return service.get_option_chain(
            symbol,
            strike_count=config.strike_count,
            include_underlying_quote=True,
            from_date=live_date,
            to_date=to_date,
        ), warnings
    except Exception:
        warnings.append("dated Schwab chain rejected; undated live chain used")
    return service.get_option_chain(
        symbol,
        strike_count=config.strike_count,
        include_underlying_quote=True,
    ), warnings


def pick_csp(row: UniverseRow, puts: list[OptionContract], spot: float, asof: dt.date, config: WheelConfig) -> OptionContract | None:
    eligible = []
    for put in puts:
        if not (config.csp_dte_min <= put.dte <= config.csp_dte_max):
            continue
        if earnings_blocks_expiry(row.next_earnings, asof, put.expiry, config):
            continue
        if put.strike >= spot:
            continue
        delta_abs = abs(put.delta)
        if math.isfinite(delta_abs) and not (config.short_put_delta_min <= delta_abs <= config.short_put_delta_max):
            continue
        discount = (spot - put.strike) / spot if spot > 0 else 0.0
        yld = annualized_yield(limit_for_sell(put), put.strike, put.dte)
        liquidity = min(math.log1p(put.open_interest) / math.log1p(5000.0), 1.0) * 10.0
        score = yld + discount * 100.0 * 0.45 + liquidity - put.spread_pct * 10.0
        eligible.append((score, put))
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: item[0], reverse=True)[0][1]


def pick_covered_call(
    row: UniverseRow,
    calls: list[OptionContract],
    spot: float,
    position: Position | None,
    asof: dt.date,
    config: WheelConfig,
) -> OptionContract | None:
    if position is None or position.shares < 100:
        return None
    basis = position.avg_cost if position.avg_cost > 0 else spot
    minimum_strike = max(spot, basis * 1.02)
    eligible = []
    for call in calls:
        if not (config.covered_call_dte_min <= call.dte <= config.covered_call_dte_max):
            continue
        if earnings_blocks_expiry(row.next_earnings, asof, call.expiry, config):
            continue
        if call.strike < minimum_strike:
            continue
        delta = call.delta
        if math.isfinite(delta) and not (config.short_call_delta_min <= delta <= config.short_call_delta_max):
            continue
        yld = annualized_yield(limit_for_sell(call), spot, call.dte)
        call_away = (call.strike - basis + limit_for_sell(call)) / basis * 100.0 if basis > 0 else 0.0
        score = yld + min(call_away, 12.0) + min(math.log1p(call.open_interest), 8.0) - call.spread_pct * 10.0
        eligible.append((score, call))
    if not eligible:
        return None
    return sorted(eligible, key=lambda item: item[0], reverse=True)[0][1]


@dataclass
class CoveredStranglePick:
    put: OptionContract
    call: OptionContract
    put_credit: float
    call_credit: float


def pick_covered_strangle(
    row: UniverseRow,
    puts: list[OptionContract],
    calls: list[OptionContract],
    spot: float,
    position: Position | None,
    asof: dt.date,
    config: WheelConfig,
) -> CoveredStranglePick | None:
    if not config.enable_covered_strangles or position is None or position.shares < 100:
        return None
    if row.ticker in REPLAY_BLOCKED_CSP:
        return None
    basis = position.avg_cost if position.avg_cost > 0 else spot
    minimum_call_strike = max(spot, basis * 1.02)
    puts_by_expiry: dict[dt.date, list[OptionContract]] = {}
    calls_by_expiry: dict[dt.date, list[OptionContract]] = {}
    for put in puts:
        if not (config.covered_strangle_dte_min <= put.dte <= config.covered_strangle_dte_max):
            continue
        if earnings_blocks_expiry(row.next_earnings, asof, put.expiry, config):
            continue
        if put.strike >= spot:
            continue
        put_discount_pct = (spot - put.strike) / spot * 100.0 if spot > 0 else 0.0
        if put_discount_pct < config.covered_strangle_min_put_discount_pct:
            continue
        delta_abs = abs(put.delta)
        if math.isfinite(delta_abs) and not (config.short_put_delta_min <= delta_abs <= config.short_put_delta_max):
            continue
        puts_by_expiry.setdefault(put.expiry, []).append(put)
    for call in calls:
        if not (config.covered_strangle_dte_min <= call.dte <= config.covered_strangle_dte_max):
            continue
        if earnings_blocks_expiry(row.next_earnings, asof, call.expiry, config):
            continue
        if call.strike < minimum_call_strike:
            continue
        delta = call.delta
        if math.isfinite(delta) and not (config.short_call_delta_min <= delta <= config.short_call_delta_max):
            continue
        calls_by_expiry.setdefault(call.expiry, []).append(call)
    picks: list[tuple[float, CoveredStranglePick]] = []
    for expiry, expiry_puts in puts_by_expiry.items():
        expiry_calls = calls_by_expiry.get(expiry, [])
        if not expiry_calls:
            continue
        for put in expiry_puts:
            put_credit = limit_for_sell(put)
            for call in expiry_calls:
                if call.strike <= put.strike:
                    continue
                range_width_pct = (call.strike - put.strike) / spot * 100.0 if spot > 0 else 0.0
                if range_width_pct < config.covered_strangle_min_range_width_pct:
                    continue
                call_credit = limit_for_sell(call)
                total_credit = put_credit + call_credit
                capital = max(spot + put.strike, 1.0)
                dte = max(put.dte, 1)
                conservative_yield = annualized_yield(total_credit, capital, dte)
                downside_be = put.strike - total_credit
                upside_be = call.strike + total_credit
                range_quality = min(range_width_pct, 25.0) + min(max(spot - downside_be, 0.0) / spot * 100.0, 10.0)
                score = (
                    conservative_yield
                    + range_quality
                    + min(row.quality_score / 10.0, 10.0)
                    + min(row.flow_score / 20.0, 5.0)
                    - put.spread_pct * 8.0
                    - call.spread_pct * 8.0
                )
                if upside_be <= spot or downside_be >= spot:
                    score -= 10.0
                picks.append((score, CoveredStranglePick(put, call, put_credit, call_credit)))
    if not picks:
        return None
    return sorted(picks, key=lambda item: item[0], reverse=True)[0][1]


@dataclass
class PmccPick:
    long_call: OptionContract
    short_call: OptionContract
    debit: float
    width: float
    debit_to_width: float


def pick_pmcc(row: UniverseRow, calls: list[OptionContract], spot: float, asof: dt.date, config: WheelConfig) -> PmccPick | None:
    if row.tier > 2:
        return None
    longs = []
    shorts = []
    for call in calls:
        if config.pmcc_long_dte_min <= call.dte <= config.pmcc_long_dte_max and call.strike < spot:
            if math.isfinite(call.delta) and config.pmcc_long_delta_min <= call.delta <= config.pmcc_long_delta_max:
                intrinsic = max(spot - call.strike, 0.0)
                extrinsic = max(call.mid - intrinsic, 0.0)
                longs.append((extrinsic / max(spot, 1.0), call))
        if earnings_blocks_expiry(row.next_earnings, asof, call.expiry, config):
            continue
        if config.pmcc_short_dte_min <= call.dte <= config.pmcc_short_dte_max and call.strike > spot:
            if math.isfinite(call.delta) and config.short_call_delta_min <= call.delta <= config.short_call_delta_max:
                shorts.append(call)
    if not longs or not shorts:
        return None
    picks: list[tuple[float, PmccPick]] = []
    for _, long_call in sorted(longs, key=lambda item: item[0])[:8]:
        intrinsic = max(spot - long_call.strike, 0.0)
        long_extrinsic = max(long_call.mid - intrinsic, 0.0)
        for short_call in shorts:
            if short_call.strike <= long_call.strike:
                continue
            debit = max(long_call.mid - limit_for_sell(short_call), 0.0)
            width = short_call.strike - long_call.strike
            if debit <= 0 or width <= 0:
                continue
            debit_to_width = debit / width
            if debit_to_width > config.pmcc_max_debit_to_width:
                continue
            if limit_for_sell(short_call) < long_extrinsic:
                continue
            short_yield = annualized_yield(limit_for_sell(short_call), debit, short_call.dte)
            score = short_yield + (1.0 - debit_to_width) * 20.0 - short_call.spread_pct * 10.0
            picks.append((score, PmccPick(long_call, short_call, round(debit, 2), width, debit_to_width)))
    if not picks:
        return None
    return sorted(picks, key=lambda item: item[0], reverse=True)[0][1]


def pick_csp_call_overlay(
    row: UniverseRow,
    calls: list[OptionContract],
    spot: float,
    csp: OptionContract,
    asof: dt.date,
    config: WheelConfig,
) -> OptionContract | None:
    if not config.enable_csp_call_overlay:
        return None
    if row.tier > 2 or row.flow_score < config.upside_overlay_min_flow_score:
        return None
    put_credit = limit_for_sell(csp)
    if put_credit <= 0:
        return None
    max_debit = put_credit * config.upside_overlay_max_credit_pct
    min_net_credit = put_credit * config.upside_overlay_min_net_credit_pct
    picks: list[tuple[float, OptionContract]] = []
    for call in calls:
        if not (config.upside_overlay_dte_min <= call.dte <= config.upside_overlay_dte_max):
            continue
        if earnings_blocks_expiry(row.next_earnings, asof, call.expiry, config):
            continue
        if call.expiry < csp.expiry or call.strike <= spot:
            continue
        delta = call.delta
        if math.isfinite(delta) and not (config.upside_overlay_delta_min <= delta <= config.upside_overlay_delta_max):
            continue
        debit = limit_for_buy(call)
        if debit <= 0 or debit > max_debit:
            continue
        if put_credit - debit < min_net_credit:
            continue
        upside_pct = (call.strike - spot) / spot * 100.0 if spot > 0 else 0.0
        if upside_pct < 3.0:
            continue
        liquidity = min(math.log1p(call.open_interest) / math.log1p(5000.0), 1.0) * 8.0
        score = (
            (put_credit - debit) / max(put_credit, 0.01) * 20.0
            + max(0.0, 12.0 - abs(delta - 0.16) * 50.0)
            + min(upside_pct, 20.0) * 0.35
            + liquidity
            - call.spread_pct * 10.0
        )
        picks.append((score, call))
    if not picks:
        return None
    return sorted(picks, key=lambda item: item[0], reverse=True)[0][1]


def action_confidence(row: UniverseRow, contract: OptionContract | None, base: float = 0.0) -> float:
    score = row.quality_score * 0.55 + row.flow_score * 0.20 + base
    if contract is not None:
        score += min(math.log1p(contract.open_interest) / math.log1p(5000.0), 1.0) * 15.0
        score += max(0.0, 10.0 - contract.spread_pct * 30.0)
    return round(max(0.0, min(100.0, score)), 1)


def contracts_for_csp(
    contract: OptionContract,
    reserved_cash: float,
    config: WheelConfig,
    reserved_total_cash: float | None = None,
) -> int:
    per_contract = contract.strike * 100.0
    single_name_cap = config.account_size * config.max_single_name_cash_pct
    total_cap = config.account_size * config.max_new_csp_cash_pct - reserved_cash
    total_reserved = reserved_cash if reserved_total_cash is None else reserved_total_cash
    cash_after_buffer = config.account_size * (1.0 - config.cash_buffer_pct) - total_reserved
    usable = min(single_name_cap, total_cap, cash_after_buffer)
    return max(0, int(max(0.0, usable) // per_contract))


def contracts_for_tactical_csp(contract: OptionContract, reserved_tactical_cash: float, reserved_total_cash: float, config: WheelConfig) -> int:
    if not config.enable_tactical_sleeve:
        return 0
    per_contract = contract.strike * 100.0
    if per_contract <= 0:
        return 0
    single_name_cap = config.account_size * config.tactical_single_name_cash_pct
    tactical_cap = config.account_size * config.tactical_sleeve_pct - reserved_tactical_cash
    cash_after_buffer = config.account_size * (1.0 - config.cash_buffer_pct) - reserved_total_cash
    usable = min(single_name_cap, tactical_cap, cash_after_buffer)
    return max(0, int(max(0.0, usable) // per_contract))


def contracts_for_pmcc(pmcc: PmccPick, used_debit: float, config: WheelConfig) -> int:
    per_contract = pmcc.debit * 100.0
    total_cap = config.account_size * config.max_pmcc_debit_pct - used_debit
    if per_contract <= 0:
        return 0
    return max(0, int(max(0.0, total_cap) // per_contract))


def fetch_positions(service: SchwabLiveDataService) -> tuple[dict[str, Position], str, float | None]:
    try:
        payload = service.get_account_positions()
    except Exception as exc:
        return {}, f"unavailable: {sanitize_error(f'{type(exc).__name__}: {exc}')}", None
    positions: dict[str, Position] = {}
    for raw in payload.get("positions", []):
        if str(raw.get("asset_type") or "").upper() != "EQUITY":
            continue
        symbol = normalize_symbol(str(raw.get("symbol") or raw.get("underlying") or ""))
        qty = int(safe_float(raw.get("qty"), 0.0))
        if qty <= 0:
            continue
        positions[symbol] = Position(symbol=symbol, shares=qty, avg_cost=safe_float(raw.get("avg_cost"), 0.0))
    total_value = safe_float((payload.get("balances") or {}).get("total_value"))
    return positions, "ok", total_value if math.isfinite(total_value) and total_value > 0 else None


def analyze_symbol(
    row: UniverseRow,
    service: SchwabLiveDataService,
    quote: dict[str, Any] | None,
    position: Position | None,
    asof: dt.date,
    config: WheelConfig,
    out_dir: Path,
) -> tuple[WheelAction, dict[str, Any] | None]:
    live_date = max(_today(), asof)
    chain, chain_warnings = get_live_chain_with_fallback(
        service,
        schwab_symbol(row.ticker),
        live_date=live_date,
        config=config,
    )
    spot = chain_spot(chain, quote)
    if not math.isfinite(spot) or spot <= 0:
        return WheelAction(
            ticker=row.ticker,
            action="WATCH_ONLY",
            confidence=row.quality_score,
            spot=row.close,
            quality_score=row.quality_score,
            flow_score=row.flow_score,
            thesis=row.thesis,
            blockers=chain_warnings + ["Schwab chain returned no usable underlying price"],
        ), chain

    calls = liquid_contracts(iter_option_contracts(chain, "C", live_date, config.risk_free_rate, spot), config)
    puts = liquid_contracts(iter_option_contracts(chain, "P", live_date, config.risk_free_rate, spot), config)
    days_to_earnings = (row.next_earnings - live_date).days if row.next_earnings else None
    if days_to_earnings is not None and 0 <= days_to_earnings <= config.avoid_earnings_days:
        return WheelAction(
            ticker=row.ticker,
            action="WAIT_POST_EARNINGS",
            confidence=round(min(95.0, row.quality_score + 12.0), 1),
            spot=spot,
            quality_score=row.quality_score,
            flow_score=row.flow_score,
            thesis=row.thesis,
            blockers=chain_warnings + [f"earnings in {days_to_earnings} days ({row.next_earnings.isoformat()})"],
        ), chain

    covered_strangle = pick_covered_strangle(row, puts, calls, spot, position, live_date, config)
    if covered_strangle is not None and position is not None:
        share_contracts = position.shares // 100
        total_credit_one = (covered_strangle.put_credit + covered_strangle.call_credit) * 100.0
        downside_be = covered_strangle.put.strike - covered_strangle.put_credit - covered_strangle.call_credit
        upside_be = covered_strangle.call.strike + covered_strangle.put_credit + covered_strangle.call_credit
        return WheelAction(
            ticker=row.ticker,
            action="SELL_COVERED_STRANGLE",
            confidence=action_confidence(row, covered_strangle.put, 18.0),
            spot=spot,
            quality_score=row.quality_score,
            flow_score=row.flow_score,
            estimated_credit=round(total_credit_one * share_contracts, 2),
            monthly_credit_runrate=round(total_credit_one * share_contracts * 30.0 / covered_strangle.put.dte, 2),
            cash_required=round(covered_strangle.put.strike * 100.0, 2),
            contracts=share_contracts,
            option_symbol=covered_strangle.call.symbol,
            expiry=covered_strangle.call.expiry,
            strike=covered_strangle.call.strike,
            limit_price=covered_strangle.call_credit,
            paired_option_symbol=covered_strangle.put.symbol,
            paired_expiry=covered_strangle.put.expiry,
            paired_strike=covered_strangle.put.strike,
            paired_limit_price=covered_strangle.put_credit,
            thesis=row.thesis,
            reasons=chain_warnings
            + [
                f"covered strangle: {position.shares} shares cover call; put is cash-secured",
                f"range ${downside_be:.2f} to ${upside_be:.2f} after ${covered_strangle.put_credit + covered_strangle.call_credit:.2f} credit",
            ],
        ), chain

    covered = pick_covered_call(row, calls, spot, position, live_date, config)
    if covered is not None and position is not None:
        contracts = position.shares // 100
        credit = limit_for_sell(covered) * 100.0 * contracts
        return WheelAction(
            ticker=row.ticker,
            action="SELL_COVERED_CALL",
            confidence=action_confidence(row, covered, 16.0),
            spot=spot,
            quality_score=row.quality_score,
            flow_score=row.flow_score,
            estimated_credit=round(credit, 2),
            monthly_credit_runrate=round(credit * 30.0 / covered.dte, 2),
            contracts=contracts,
            option_symbol=covered.symbol,
            expiry=covered.expiry,
            strike=covered.strike,
            limit_price=limit_for_sell(covered),
            thesis=row.thesis,
            reasons=chain_warnings + [f"Schwab position has {position.shares} shares", "strike is above spot/basis buffer"],
        ), chain

    csp = pick_csp(row, puts, spot, live_date, config)
    if csp is not None:
        discount_pct = (spot - csp.strike) / spot * 100.0
        premium_yield_pct = limit_for_sell(csp) / csp.strike * 100.0 if csp.strike > 0 else 0.0
        credit_one = limit_for_sell(csp) * 100.0
        monthly = credit_one * 30.0 / csp.dte
        confidence = action_confidence(row, csp, 12.0 + min(discount_pct, 8.0))
        alert_price = round(min(spot * 0.97, csp.strike * 1.02), 2)
        action = "OPEN_CSP" if confidence >= 78.0 and discount_pct >= 3.0 else "SET_CSP_ALERT"
        sleeve = "tactical" if is_tactical_universe_row(row, config) else "core"
        csp_reasons = chain_warnings + [
            f"Schwab put chain selected {csp.dte} DTE ${csp.strike:g} put",
            f"net assignment ${csp.strike - limit_for_sell(csp):.2f}, {discount_pct:.1f}% below spot before credit",
        ]
        csp_blockers: list[str] = []
        if row.ticker in REPLAY_BLOCKED_CSP:
            action = "WATCH_ONLY"
            alert_price = None
            csp_blockers.append(f"replay block: {REPLAY_BLOCKED_CSP[row.ticker]}")
        elif row.ticker in REPLAY_CAUTION_CSP:
            csp_reasons.append(f"replay caution: {REPLAY_CAUTION_CSP[row.ticker]}")
        if action != "WATCH_ONLY" and sleeve == "tactical":
            csp_reasons.append("tactical sleeve: high UW premium/flow; sized separately from primary wheel cash")
            if (
                confidence >= config.tactical_min_confidence
                and premium_yield_pct >= config.tactical_min_yield_pct
                and discount_pct >= config.tactical_min_discount_pct
            ):
                action = "OPEN_TACTICAL_CSP"
                alert_price = None
            else:
                action = "TACTICAL_RANGE_ALERT"
        if (
            action == "OPEN_CSP"
            and config.enable_leaps_covered_strangles
            and row.flow_score >= config.leaps_strangle_min_flow_score
            and discount_pct >= config.leaps_strangle_min_put_discount_pct
        ):
            pmcc_for_strangle = pick_pmcc(row, calls, spot, live_date, config)
            if pmcc_for_strangle is not None:
                short_call_credit = limit_for_sell(pmcc_for_strangle.short_call)
                long_call_debit = limit_for_buy(pmcc_for_strangle.long_call)
                return WheelAction(
                    ticker=row.ticker,
                    action="OPEN_LEAPS_COVERED_STRANGLE",
                    confidence=round(min(100.0, confidence + 2.0), 1),
                    spot=spot,
                    quality_score=row.quality_score,
                    flow_score=row.flow_score,
                    estimated_credit=round((limit_for_sell(csp) + short_call_credit) * 100.0, 2),
                    monthly_credit_runrate=round(
                        (limit_for_sell(csp) + short_call_credit) * 100.0 * 30.0 / max(min(csp.dte, pmcc_for_strangle.short_call.dte), 1),
                        2,
                    ),
                    cash_required=round(csp.strike * 100.0, 2),
                    pmcc_debit=round(long_call_debit * 100.0, 2),
                    contracts=0,
                    option_symbol=pmcc_for_strangle.short_call.symbol,
                    expiry=pmcc_for_strangle.short_call.expiry,
                    strike=pmcc_for_strangle.short_call.strike,
                    limit_price=short_call_credit,
                    long_option_symbol=pmcc_for_strangle.long_call.symbol,
                    long_expiry=pmcc_for_strangle.long_call.expiry,
                    long_strike=pmcc_for_strangle.long_call.strike,
                    long_limit_price=long_call_debit,
                    paired_option_symbol=csp.symbol,
                    paired_expiry=csp.expiry,
                    paired_strike=csp.strike,
                    paired_limit_price=limit_for_sell(csp),
                    thesis=row.thesis,
                    reasons=csp_reasons
                    + [
                        "LEAPS-covered strangle: short call is covered by long call; put is cash-secured",
                        f"long-call debit {long_call_debit:.2f}, short-call credit {short_call_credit:.2f}",
                    ],
                    blockers=csp_blockers,
                ), chain
        long_call = None
        long_call_debit = 0.0
        if action == "OPEN_CSP" and row.ticker not in REPLAY_BLOCKED_CSP:
            long_call = pick_csp_call_overlay(row, calls, spot, csp, live_date, config)
            if long_call is not None:
                long_call_debit = limit_for_buy(long_call)
                if limit_for_sell(csp) - long_call_debit <= 0:
                    long_call = None
                    long_call_debit = 0.0
                else:
                    action = "OPEN_CSP_WITH_CALL_OVERLAY"
                    credit_one = (limit_for_sell(csp) - long_call_debit) * 100.0
                    monthly = credit_one * 30.0 / csp.dte
                    csp_reasons.append(
                        f"premium-funded upside overlay: buy {long_call.symbol} up to {long_call_debit:.2f}; net credit {limit_for_sell(csp) - long_call_debit:.2f}"
                    )
        return WheelAction(
            ticker=row.ticker,
            action=action,
            confidence=confidence,
            spot=spot,
            quality_score=row.quality_score,
            flow_score=row.flow_score,
            estimated_credit=round(credit_one, 2),
            monthly_credit_runrate=round(monthly, 2),
            cash_required=round(csp.strike * 100.0, 2),
            contracts=0,
            alert_price=None if action in {"OPEN_CSP", "OPEN_CSP_WITH_CALL_OVERLAY", "OPEN_TACTICAL_CSP"} else alert_price,
            option_symbol=csp.symbol,
            expiry=csp.expiry,
            strike=csp.strike,
            limit_price=limit_for_sell(csp),
            long_option_symbol=long_call.symbol if long_call is not None else "",
            long_expiry=long_call.expiry if long_call is not None else None,
            long_strike=long_call.strike if long_call is not None else None,
            long_limit_price=long_call_debit if long_call is not None else None,
            sleeve=sleeve,
            thesis=row.thesis,
            reasons=csp_reasons,
            blockers=csp_blockers,
        ), chain

    pmcc = pick_pmcc(row, calls, spot, live_date, config)
    if pmcc is not None:
        confidence = action_confidence(row, pmcc.short_call, 10.0 + (1.0 - pmcc.debit_to_width) * 12.0)
        return WheelAction(
            ticker=row.ticker,
            action="OPEN_PMCC" if confidence >= 80.0 else "WATCH_ONLY",
            confidence=confidence,
            spot=spot,
            quality_score=row.quality_score,
            flow_score=row.flow_score,
            estimated_credit=round(limit_for_sell(pmcc.short_call) * 100.0, 2),
            monthly_credit_runrate=round(limit_for_sell(pmcc.short_call) * 100.0 * 30.0 / pmcc.short_call.dte, 2),
            pmcc_debit=round(pmcc.debit * 100.0, 2),
            option_symbol=pmcc.short_call.symbol,
            expiry=pmcc.short_call.expiry,
            strike=pmcc.short_call.strike,
            limit_price=limit_for_sell(pmcc.short_call),
            long_option_symbol=pmcc.long_call.symbol,
            long_expiry=pmcc.long_call.expiry,
            long_strike=pmcc.long_call.strike,
            thesis=row.thesis,
            reasons=chain_warnings + [
                "Schwab calls support a defined PMCC/LEAPS structure",
                f"debit/width {pmcc.debit_to_width:.2f}",
            ],
        ), chain

    return WheelAction(
        ticker=row.ticker,
        action="WATCH_ONLY",
        confidence=round(row.quality_score * 0.65 + row.flow_score * 0.20, 1),
        spot=spot,
        quality_score=row.quality_score,
        flow_score=row.flow_score,
        thesis=row.thesis,
        blockers=chain_warnings + ["no Schwab chain contract passed wheel liquidity, DTE, delta, earnings, and spread rules"],
    ), chain


def allocate_contracts(actions: list[WheelAction], config: WheelConfig) -> None:
    reserved_cash = 0.0
    reserved_tactical_cash = 0.0
    reserved_total_cash = 0.0
    used_debit = 0.0
    for action in sorted(actions, key=lambda item: item.confidence, reverse=True):
        if action.action == "OPEN_TACTICAL_CSP" and action.strike and action.limit_price:
            per_contract_cash = action.strike * 100.0
            dte = max((action.expiry - _today()).days, 1) if action.expiry else 1
            contracts = contracts_for_tactical_csp(
                OptionContract(
                    symbol=action.option_symbol,
                    expiry=action.expiry or _today(),
                    dte=dte,
                    right="P",
                    strike=action.strike,
                    bid=action.limit_price,
                    ask=action.limit_price,
                    mark=action.limit_price,
                    mid=action.limit_price,
                    delta=math.nan,
                    iv=math.nan,
                    open_interest=999,
                    volume=999,
                    spread_pct=0.0,
                ),
                reserved_tactical_cash,
                reserved_total_cash,
                config,
            )
            action.contracts = contracts
            action.cash_required = round(per_contract_cash * contracts, 2)
            action.estimated_credit = round((action.limit_price or 0.0) * 100.0 * contracts, 2)
            action.monthly_credit_runrate = round(action.estimated_credit * 30.0 / dte, 2)
            reserved_tactical_cash += action.cash_required
            reserved_total_cash += action.cash_required
            if contracts <= 0:
                action.action = "TACTICAL_RANGE_ALERT"
                action.alert_price = round(min(action.spot * 0.97, action.strike * 1.02), 2)
                action.blockers.append("tactical sleeve cash cap does not allow one cash-secured contract")
        elif action.action in {"OPEN_CSP", "OPEN_CSP_WITH_CALL_OVERLAY"} and action.strike and action.limit_price:
            per_contract_cash = action.strike * 100.0
            dte = max((action.expiry - _today()).days, 1) if action.expiry else 1
            contracts = contracts_for_csp(
                OptionContract(
                    symbol=action.option_symbol,
                    expiry=action.expiry or _today(),
                    dte=dte,
                    right="P",
                    strike=action.strike,
                    bid=action.limit_price,
                    ask=action.limit_price,
                    mark=action.limit_price,
                    mid=action.limit_price,
                    delta=math.nan,
                    iv=math.nan,
                    open_interest=999,
                    volume=999,
                    spread_pct=0.0,
                ),
                reserved_cash,
                config,
                reserved_total_cash,
            )
            action.contracts = contracts
            action.cash_required = round(per_contract_cash * contracts, 2)
            long_debit_one = (action.long_limit_price or 0.0) * 100.0 if action.action == "OPEN_CSP_WITH_CALL_OVERLAY" else 0.0
            net_credit_one = max((action.limit_price or 0.0) * 100.0 - long_debit_one, 0.0)
            action.pmcc_debit = round(long_debit_one * contracts, 2)
            action.estimated_credit = round(net_credit_one * contracts, 2)
            action.monthly_credit_runrate = round(action.estimated_credit * 30.0 / dte, 2)
            reserved_cash += action.cash_required
            reserved_total_cash += action.cash_required
            if contracts <= 0:
                action.action = "SET_CSP_ALERT"
                action.alert_price = round(min(action.spot * 0.97, action.strike * 1.02), 2)
                action.blockers.append("account-size and cash-buffer rules do not allow one cash-secured contract")
        elif (
            action.action == "SELL_COVERED_STRANGLE"
            and action.paired_strike
            and action.paired_limit_price is not None
            and action.limit_price is not None
        ):
            share_contracts = max(action.contracts, 0)
            put_contracts = contracts_for_csp(
                OptionContract(
                    symbol=action.paired_option_symbol,
                    expiry=action.paired_expiry or _today(),
                    dte=max((action.paired_expiry - _today()).days, 1) if action.paired_expiry else 1,
                    right="P",
                    strike=action.paired_strike,
                    bid=action.paired_limit_price,
                    ask=action.paired_limit_price,
                    mark=action.paired_limit_price,
                    mid=action.paired_limit_price,
                    delta=math.nan,
                    iv=math.nan,
                    open_interest=999,
                    volume=999,
                    spread_pct=0.0,
                ),
                reserved_cash,
                config,
                reserved_total_cash,
            )
            contracts = min(share_contracts, put_contracts)
            dte = max((action.expiry - _today()).days, 1) if action.expiry else 1
            if contracts <= 0:
                action.action = "SELL_COVERED_CALL"
                action.contracts = share_contracts
                action.cash_required = 0.0
                action.estimated_credit = round((action.limit_price or 0.0) * 100.0 * share_contracts, 2)
                action.monthly_credit_runrate = round(action.estimated_credit * 30.0 / dte, 2)
                action.blockers.append("covered-strangle cash-secured put budget did not allow one extra put; fallback to covered call")
            else:
                action.contracts = contracts
                action.cash_required = round(action.paired_strike * 100.0 * contracts, 2)
                net_credit_one = ((action.limit_price or 0.0) + (action.paired_limit_price or 0.0)) * 100.0
                action.estimated_credit = round(net_credit_one * contracts, 2)
                action.monthly_credit_runrate = round(action.estimated_credit * 30.0 / dte, 2)
                reserved_cash += action.cash_required
                reserved_total_cash += action.cash_required
        elif (
            action.action == "OPEN_LEAPS_COVERED_STRANGLE"
            and action.paired_strike
            and action.paired_limit_price is not None
            and action.long_limit_price is not None
            and action.limit_price is not None
        ):
            put_contracts = contracts_for_csp(
                OptionContract(
                    symbol=action.paired_option_symbol,
                    expiry=action.paired_expiry or _today(),
                    dte=max((action.paired_expiry - _today()).days, 1) if action.paired_expiry else 1,
                    right="P",
                    strike=action.paired_strike,
                    bid=action.paired_limit_price,
                    ask=action.paired_limit_price,
                    mark=action.paired_limit_price,
                    mid=action.paired_limit_price,
                    delta=math.nan,
                    iv=math.nan,
                    open_interest=999,
                    volume=999,
                    spread_pct=0.0,
                ),
                reserved_cash,
                config,
                reserved_total_cash,
            )
            per_contract_debit = action.long_limit_price * 100.0
            debit_cap = config.account_size * config.max_pmcc_debit_pct - used_debit
            debit_contracts = int(max(0.0, debit_cap) // per_contract_debit) if per_contract_debit > 0 else 0
            contracts = min(put_contracts, debit_contracts)
            dte = max(
                min(
                    (action.expiry - _today()).days if action.expiry else 1,
                    (action.paired_expiry - _today()).days if action.paired_expiry else 1,
                ),
                1,
            )
            if contracts <= 0 and put_contracts > 0:
                action.action = "OPEN_CSP"
                action.option_symbol = action.paired_option_symbol
                action.expiry = action.paired_expiry
                action.strike = action.paired_strike
                action.limit_price = action.paired_limit_price
                action.long_option_symbol = ""
                action.long_expiry = None
                action.long_strike = None
                action.long_limit_price = None
                action.paired_option_symbol = ""
                action.paired_expiry = None
                action.paired_strike = None
                action.paired_limit_price = None
                action.contracts = put_contracts
                action.cash_required = round((action.strike or 0.0) * 100.0 * put_contracts, 2)
                action.pmcc_debit = 0.0
                action.estimated_credit = round((action.limit_price or 0.0) * 100.0 * put_contracts, 2)
                action.monthly_credit_runrate = round(action.estimated_credit * 30.0 / dte, 2)
                action.blockers.append("LEAPS debit budget did not allow one covered-strangle package; fallback to plain CSP")
                reserved_cash += action.cash_required
                reserved_total_cash += action.cash_required
            elif contracts <= 0:
                action.action = "SET_CSP_ALERT"
                action.option_symbol = action.paired_option_symbol
                action.expiry = action.paired_expiry
                action.strike = action.paired_strike
                action.limit_price = action.paired_limit_price
                action.alert_price = round(min(action.spot * 0.97, (action.strike or action.spot) * 1.02), 2)
                action.long_option_symbol = ""
                action.long_expiry = None
                action.long_strike = None
                action.long_limit_price = None
                action.paired_option_symbol = ""
                action.paired_expiry = None
                action.paired_strike = None
                action.paired_limit_price = None
                action.pmcc_debit = 0.0
                action.blockers.append("cash and LEAPS debit budgets do not allow one covered-strangle package")
            else:
                action.contracts = contracts
                action.cash_required = round(action.paired_strike * 100.0 * contracts, 2)
                action.pmcc_debit = round(per_contract_debit * contracts, 2)
                net_credit_one = ((action.limit_price or 0.0) + (action.paired_limit_price or 0.0)) * 100.0
                action.estimated_credit = round(net_credit_one * contracts, 2)
                action.monthly_credit_runrate = round(action.estimated_credit * 30.0 / dte, 2)
                reserved_cash += action.cash_required
                reserved_total_cash += action.cash_required
                used_debit += action.pmcc_debit
        elif action.action == "OPEN_PMCC" and action.pmcc_debit > 0:
            per_contract_debit = action.pmcc_debit
            cap = config.account_size * config.max_pmcc_debit_pct - used_debit
            contracts = int(max(0.0, cap) // per_contract_debit) if per_contract_debit > 0 else 0
            action.contracts = contracts
            action.pmcc_debit = round(per_contract_debit * contracts, 2)
            action.estimated_credit = round(action.estimated_credit * contracts, 2)
            used_debit += action.pmcc_debit
            if contracts <= 0:
                action.action = "WATCH_ONLY"
                action.blockers.append("PMCC debit budget does not allow one contract")


def _md_cell(value: Any) -> str:
    text = str(value if value is not None else "").replace("|", "/").replace("\n", " ").strip()
    return text or "-"


def _premium_yield_pct(action: WheelAction) -> float:
    if action.limit_price is None or action.limit_price <= 0:
        return 0.0
    if action.action == "OPEN_CSP_WITH_CALL_OVERLAY" and action.strike and action.strike > 0:
        if action.contracts > 0 and action.estimated_credit > 0:
            return action.estimated_credit / action.contracts / (action.strike * 100.0) * 100.0
        net_limit = max(action.limit_price - (action.long_limit_price or 0.0), 0.0)
        return net_limit / action.strike * 100.0
    if action.action in {"OPEN_CSP", "OPEN_TACTICAL_CSP", "SET_CSP_ALERT", "TACTICAL_RANGE_ALERT", "WATCH_ONLY"} and action.strike and action.strike > 0:
        return action.limit_price / action.strike * 100.0
    if action.action == "SELL_COVERED_STRANGLE" and action.paired_strike and action.spot > 0:
        per_contract_credit = (action.limit_price + (action.paired_limit_price or 0.0)) * 100.0
        capital = (action.spot + action.paired_strike) * 100.0
        return per_contract_credit / capital * 100.0 if capital > 0 else 0.0
    if action.action == "OPEN_LEAPS_COVERED_STRANGLE" and action.paired_strike and action.long_limit_price:
        per_contract_credit = (action.limit_price + (action.paired_limit_price or 0.0)) * 100.0
        capital = (action.paired_strike + action.long_limit_price) * 100.0
        return per_contract_credit / capital * 100.0 if capital > 0 else 0.0
    if action.action == "SELL_COVERED_CALL" and action.spot > 0:
        return action.limit_price / action.spot * 100.0
    if action.action == "OPEN_PMCC" and action.pmcc_debit > 0:
        return action.limit_price * 100.0 / action.pmcc_debit * 100.0
    return 0.0


def _credit_per_contract(action: WheelAction) -> float:
    if action.limit_price is None or action.limit_price <= 0:
        return 0.0
    if action.action == "OPEN_CSP_WITH_CALL_OVERLAY":
        return max(action.limit_price - (action.long_limit_price or 0.0), 0.0) * 100.0
    if action.action == "SELL_COVERED_STRANGLE":
        return (action.limit_price + (action.paired_limit_price or 0.0)) * 100.0
    if action.action == "OPEN_LEAPS_COVERED_STRANGLE":
        return (action.limit_price + (action.paired_limit_price or 0.0)) * 100.0
    return action.limit_price * 100.0


def _is_tradeable(action: WheelAction) -> bool:
    return action.action in {
        "OPEN_CSP",
        "OPEN_TACTICAL_CSP",
        "OPEN_CSP_WITH_CALL_OVERLAY",
        "SELL_COVERED_CALL",
        "SELL_COVERED_STRANGLE",
        "OPEN_LEAPS_COVERED_STRANGLE",
        "OPEN_PMCC",
    } and action.contracts > 0


def _is_strong_entry(action: WheelAction) -> bool:
    if not _is_tradeable(action):
        return False
    premium_yield = _premium_yield_pct(action)
    credit_one = _credit_per_contract(action)
    if action.action in {"OPEN_CSP", "OPEN_CSP_WITH_CALL_OVERLAY"}:
        return action.confidence >= 85.0 and premium_yield >= 1.0 and credit_one >= 300.0
    if action.action == "SELL_COVERED_CALL":
        return action.confidence >= 75.0 and premium_yield >= 0.75 and credit_one >= 100.0
    if action.action == "SELL_COVERED_STRANGLE":
        return action.confidence >= 82.0 and premium_yield >= 0.60 and credit_one >= 250.0
    if action.action == "OPEN_LEAPS_COVERED_STRANGLE":
        return action.confidence >= 88.0 and premium_yield >= 0.70 and credit_one >= 300.0
    if action.action == "OPEN_PMCC":
        return action.confidence >= 85.0 and premium_yield >= 1.5 and credit_one >= 150.0
    return False


def _entry_tier(action: WheelAction) -> str:
    if action.action == "OPEN_TACTICAL_CSP" and _is_tradeable(action):
        return "SECONDARY"
    if _is_strong_entry(action):
        return "STRONG"
    if _is_tradeable(action):
        return "SECONDARY"
    if action.action == "TACTICAL_RANGE_ALERT":
        return "ALERT"
    if action.action == "SET_CSP_ALERT":
        return "ALERT"
    if action.action == "WAIT_POST_EARNINGS":
        return "WAIT"
    return "AVOID"


def _status_icon(action: WheelAction) -> str:
    if action.action == "OPEN_TACTICAL_CSP" and _is_tradeable(action):
        return "🔵 SECONDARY"
    if action.action == "TACTICAL_RANGE_ALERT":
        return "🟡 ALERT"
    if _is_strong_entry(action):
        return "🟢 STRONG"
    if _is_tradeable(action):
        return "🔵 SECONDARY"
    if action.action == "SET_CSP_ALERT":
        return "🟡 ALERT"
    if action.action == "WAIT_POST_EARNINGS":
        return "🟠 WAIT"
    return "🔴 AVOID"


def _human_expiry(value: dt.date | None) -> str:
    if value is None:
        return "-"
    return f"{value:%b} {value.day}, {value.year}"


def _option_leg(verb: str, expiry: dt.date | None, strike: float | None, right: str) -> str:
    if expiry is None or strike is None:
        return verb
    return f"{verb} {_human_expiry(expiry)} ${strike:g} {right}"


def _option_right_from_symbol(symbol: str) -> str:
    match = re.search(r"\d{6}([CP])\d{8}$", str(symbol or "").strip())
    return "call" if match and match.group(1) == "C" else "put"


def _action_ticket(action: WheelAction, *, include_trigger: bool = True) -> str:
    if action.action == "OPEN_PMCC":
        return (
            f"{_option_leg('Buy', action.long_expiry, action.long_strike, 'call')} + "
            f"{_option_leg('sell', action.expiry, action.strike, 'call')}"
        )
    if action.action == "OPEN_CSP_WITH_CALL_OVERLAY":
        return (
            f"{_option_leg('Sell', action.expiry, action.strike, 'put')} + "
            f"{_option_leg('buy', action.long_expiry, action.long_strike, 'call')}"
        )
    if action.action == "SELL_COVERED_STRANGLE":
        return (
            f"{_option_leg('Sell', action.paired_expiry, action.paired_strike, 'put')} + "
            f"{_option_leg('sell covered', action.expiry, action.strike, 'call')}"
        )
    if action.action == "OPEN_LEAPS_COVERED_STRANGLE":
        return (
            f"{_option_leg('Buy', action.long_expiry, action.long_strike, 'call')} + "
            f"{_option_leg('sell', action.expiry, action.strike, 'call')} + "
            f"{_option_leg('sell', action.paired_expiry, action.paired_strike, 'put')}"
        )
    if action.action == "SELL_COVERED_CALL":
        return _option_leg("Sell covered", action.expiry, action.strike, "call")
    if action.action in {"OPEN_CSP", "OPEN_TACTICAL_CSP", "SET_CSP_ALERT", "TACTICAL_RANGE_ALERT"}:
        if include_trigger and action.alert_price is not None:
            return f"At stock <= ${action.alert_price:.2f}: {_option_leg('sell', action.expiry, action.strike, 'put')}"
        return _option_leg("Sell", action.expiry, action.strike, "put")
    if action.option_symbol and action.expiry and action.strike is not None:
        return _option_leg("Blocked: sell", action.expiry, action.strike, _option_right_from_symbol(action.option_symbol))
    return action.action.replace("_", " ")


def _action_expiry(action: WheelAction) -> str:
    expiries: list[dt.date] = []
    if action.action in {"OPEN_CSP_WITH_CALL_OVERLAY", "OPEN_PMCC", "OPEN_LEAPS_COVERED_STRANGLE"} and action.long_expiry:
        expiries.append(action.long_expiry)
    if action.action in {"SELL_COVERED_STRANGLE", "OPEN_LEAPS_COVERED_STRANGLE"} and action.paired_expiry:
        expiries.append(action.paired_expiry)
    if action.expiry:
        expiries.append(action.expiry)
    unique = list(dict.fromkeys(expiries))
    return " / ".join(_human_expiry(value) for value in unique) if unique else "-"


def _action_strike(action: WheelAction) -> str:
    if action.action == "OPEN_CSP_WITH_CALL_OVERLAY" and action.long_strike is not None:
        return f"${action.strike:g} put / ${action.long_strike:g} call" if action.strike is not None else f"${action.long_strike:g} call"
    if action.action == "SELL_COVERED_STRANGLE" and action.paired_strike is not None:
        return f"${action.paired_strike:g} put / ${action.strike:g} call" if action.strike is not None else f"${action.paired_strike:g} put"
    if action.action == "OPEN_LEAPS_COVERED_STRANGLE" and action.long_strike is not None and action.paired_strike is not None:
        return (
            f"${action.long_strike:g} long call / ${action.strike:g} short call / ${action.paired_strike:g} put"
            if action.strike is not None
            else f"${action.long_strike:g} long call / ${action.paired_strike:g} put"
        )
    if action.action == "OPEN_PMCC" and action.long_strike is not None:
        return f"${action.long_strike:g} long call / ${action.strike:g} short call" if action.strike is not None else f"${action.long_strike:g} long call"
    if action.strike is None:
        return "-"
    right = "call" if action.action == "SELL_COVERED_CALL" else _option_right_from_symbol(action.option_symbol)
    return f"${action.strike:g} {right}"


def _action_type(action: WheelAction) -> str:
    if action.action == "OPEN_CSP":
        return "Cash-secured put"
    if action.action == "OPEN_TACTICAL_CSP":
        return "Tactical CSP"
    if action.action == "OPEN_CSP_WITH_CALL_OVERLAY":
        return "CSP + call overlay"
    if action.action == "SELL_COVERED_CALL":
        return "Covered call"
    if action.action == "SELL_COVERED_STRANGLE":
        return "Covered strangle"
    if action.action == "OPEN_LEAPS_COVERED_STRANGLE":
        return "LEAPS-covered strangle"
    if action.action == "OPEN_PMCC":
        return "PMCC"
    if action.action == "SET_CSP_ALERT":
        return "CSP alert"
    if action.action == "TACTICAL_RANGE_ALERT":
        return "Tactical alert"
    if action.action == "WAIT_POST_EARNINGS":
        return "Wait"
    return action.action.replace("_", " ").title()


def _action_limit(action: WheelAction) -> str:
    if action.limit_price is None:
        return "-"
    if action.action == "OPEN_CSP_WITH_CALL_OVERLAY":
        net = action.limit_price - (action.long_limit_price or 0.0)
        return f"${net:.2f}+ net credit"
    if action.action == "SELL_COVERED_STRANGLE":
        return f"${action.limit_price + (action.paired_limit_price or 0.0):.2f}+ net credit"
    if action.action == "OPEN_LEAPS_COVERED_STRANGLE":
        return f"${action.limit_price + (action.paired_limit_price or 0.0):.2f}+ credit"
    return f"${action.limit_price:.2f}+ credit"


def _action_context(action: WheelAction) -> str:
    notes = action.reasons + action.blockers
    return "; ".join(notes or [action.thesis or "no action"])


def write_outputs(
    out_dir: Path,
    asof: dt.date,
    base_dir: Path,
    universe: list[UniverseRow],
    actions: list[WheelAction],
    position_status: str,
    chain_errors: dict[str, str],
    config: WheelConfig,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"fresh-wheel-report-{asof.isoformat()}.md"
    actions_csv = out_dir / f"fresh-wheel-actions-{asof.isoformat()}.csv"
    orders_csv = out_dir / f"fresh-wheel-orders-{asof.isoformat()}.csv"
    alerts_csv = out_dir / f"fresh-wheel-alerts-{asof.isoformat()}.csv"
    universe_csv = out_dir / f"fresh-wheel-universe-{asof.isoformat()}.csv"
    manifest_path = out_dir / f"fresh-wheel-manifest-{asof.isoformat()}.json"

    with universe_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(universe[0]).keys()) if universe else ["ticker"])
        writer.writeheader()
        for row in universe:
            data = asdict(row)
            data["next_earnings"] = row.next_earnings.isoformat() if row.next_earnings else ""
            data["reasons"] = "; ".join(row.reasons)
            writer.writerow(data)

    action_fields = [
        "ticker",
        "action",
        "confidence",
        "spot",
        "quality_score",
        "flow_score",
        "contracts",
        "cash_required",
        "pmcc_debit",
        "estimated_credit",
        "monthly_credit_runrate",
        "alert_price",
        "option_symbol",
        "expiry",
        "strike",
        "limit_price",
        "long_option_symbol",
        "long_expiry",
        "long_strike",
        "long_limit_price",
        "paired_option_symbol",
        "paired_expiry",
        "paired_strike",
        "paired_limit_price",
        "sleeve",
        "thesis",
        "reasons",
        "blockers",
    ]
    with actions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=action_fields)
        writer.writeheader()
        for action in actions:
            data = asdict(action)
            data["expiry"] = action.expiry.isoformat() if action.expiry else ""
            data["long_expiry"] = action.long_expiry.isoformat() if action.long_expiry else ""
            data["paired_expiry"] = action.paired_expiry.isoformat() if action.paired_expiry else ""
            data["reasons"] = "; ".join(action.reasons)
            data["blockers"] = "; ".join(action.blockers)
            writer.writerow({field: data.get(field, "") for field in action_fields})

    with orders_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ticker",
                "tier",
                "sleeve",
                "action",
                "contracts",
                "expiry",
                "strike",
                "option_symbol",
                "long_option_symbol",
                "long_expiry",
                "long_strike",
                "long_limit_price",
                "paired_option_symbol",
                "paired_expiry",
                "paired_strike",
                "paired_limit_price",
                "limit_price",
                "cash_required",
                "estimated_credit",
                "premium_yield_pct",
                "notes",
            ],
        )
        writer.writeheader()
        for action in actions:
            if action.action not in {
                "OPEN_CSP",
                "OPEN_TACTICAL_CSP",
                "OPEN_CSP_WITH_CALL_OVERLAY",
                "SELL_COVERED_CALL",
                "SELL_COVERED_STRANGLE",
                "OPEN_LEAPS_COVERED_STRANGLE",
                "OPEN_PMCC",
            } or action.contracts <= 0:
                continue
            writer.writerow(
                {
                    "ticker": action.ticker,
                    "tier": _entry_tier(action),
                    "sleeve": action.sleeve,
                    "action": action.action,
                    "contracts": action.contracts,
                    "expiry": action.expiry.isoformat() if action.expiry else "",
                    "strike": action.strike if action.strike is not None else "",
                    "option_symbol": action.option_symbol,
                    "long_option_symbol": action.long_option_symbol,
                    "long_expiry": action.long_expiry.isoformat() if action.long_expiry else "",
                    "long_strike": action.long_strike if action.long_strike is not None else "",
                    "long_limit_price": action.long_limit_price if action.long_limit_price is not None else "",
                    "paired_option_symbol": action.paired_option_symbol,
                    "paired_expiry": action.paired_expiry.isoformat() if action.paired_expiry else "",
                    "paired_strike": action.paired_strike if action.paired_strike is not None else "",
                    "paired_limit_price": action.paired_limit_price if action.paired_limit_price is not None else "",
                    "limit_price": action.limit_price if action.limit_price is not None else "",
                    "cash_required": f"{action.cash_required:.2f}",
                    "estimated_credit": f"{action.estimated_credit:.2f}",
                    "premium_yield_pct": f"{_premium_yield_pct(action):.2f}",
                    "notes": "; ".join(action.reasons),
                }
            )

    with alerts_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ticker", "trigger_price", "sleeve", "action", "expiry", "strike", "option_symbol", "limit_price", "confidence", "notes"],
        )
        writer.writeheader()
        for action in actions:
            if action.action not in {"SET_CSP_ALERT", "TACTICAL_RANGE_ALERT"} or action.alert_price is None:
                continue
            writer.writerow(
                {
                    "ticker": action.ticker,
                    "trigger_price": f"{action.alert_price:.2f}",
                    "sleeve": action.sleeve,
                    "action": action.action,
                    "expiry": action.expiry.isoformat() if action.expiry else "",
                    "strike": action.strike if action.strike is not None else "",
                    "option_symbol": action.option_symbol,
                    "limit_price": action.limit_price if action.limit_price is not None else "",
                    "confidence": f"{action.confidence:.1f}",
                    "notes": "; ".join(action.reasons + action.blockers),
                }
            )

    tradeable = [a for a in actions if _is_tradeable(a)]
    tactical_entries = [a for a in tradeable if a.action == "OPEN_TACTICAL_CSP"]
    strong_entries = [a for a in tradeable if _is_strong_entry(a)]
    focus_entries = strong_entries + [a for a in tactical_entries if a not in strong_entries]
    secondary_entries = [a for a in tradeable if not _is_strong_entry(a) and a.action != "OPEN_TACTICAL_CSP"]
    alerts = [a for a in actions if a.action in {"SET_CSP_ALERT", "TACTICAL_RANGE_ALERT"}]
    watch = [a for a in actions if a not in tradeable and a not in alerts]
    monthly_runrate = sum(a.monthly_credit_runrate for a in tradeable)
    strong_monthly_runrate = sum(a.monthly_credit_runrate for a in strong_entries)
    tactical_monthly_runrate = sum(a.monthly_credit_runrate for a in tactical_entries)
    lines = [
        f"# Fresh Schwab Wheel Report ({asof.isoformat()})",
        "",
        "## Data Contract",
        "",
        f"- UW source folder: `{base_dir}`",
        "- Live quote and option-chain source: `Schwab API`",
        "- Yahoo/yfinance: `not used`",
        "- Universe selection: objective quality, tactical-flow, and total-premium lanes; no hard-coded ticker roster; held round-lot positions appended",
        f"- Schwab position fetch: `{position_status}`",
        f"- Account-size assumption: `${config.account_size:,.0f}`",
        f"- Monthly income target: `${config.target_monthly_income_low:,.0f}` to `${config.target_monthly_income_high:,.0f}`",
        f"- Current immediate monthlyized run-rate from tradeable rows: `${monthly_runrate:,.0f}`",
        f"- Strong-entry monthlyized run-rate: `${strong_monthly_runrate:,.0f}`",
        f"- Tactical-sleeve monthlyized run-rate: `${tactical_monthly_runrate:,.0f}`",
        f"- Cadence: wheel entries are weekly/opportunistic; no daily trade requirement.",
        "",
        "## Weekly Focus",
        "",
        "| Status | Ticker | Type | Exp | Strike | Trade | Limit | Qty | Credit | Yield | Conf | Inline Context |",
        "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    if focus_entries:
        for action in focus_entries:
            lines.append(
                f"| {_status_icon(action)} | {_md_cell(action.ticker)} | {_md_cell(_action_type(action))} | "
                f"{_md_cell(_action_expiry(action))} | {_md_cell(_action_strike(action))} | {_md_cell(_action_ticket(action))} | "
                f"{_md_cell(_action_limit(action))} | {action.contracts} | ${action.estimated_credit:,.0f} | "
                f"{_premium_yield_pct(action):.2f}% | {action.confidence:.1f} | {_md_cell(_action_context(action))} |"
            )
    else:
        lines.append("| 🟠 WAIT | - | - | - | - | No strong weekly entry passed the current gates | - | - | - | - | - | Do not force a wheel trade when the board is empty |")
    if secondary_entries:
        lines.append(
            f"| 🔵 SECONDARY | Basket | - | - | - | {len(secondary_entries)} lower-priority tradeable rows remain on the action board | - | - | - | - | - | Use only after reviewing stronger entries and total cash exposure |"
        )
    if tactical_entries:
        lines.append(
            f"| 🔵 SECONDARY | Basket | Tactical sleeve | - | - | {len(tactical_entries)} tactical rows are separate from primary cash-secured-put budget | - | - | ${sum(a.estimated_credit for a in tactical_entries):,.0f} | - | - | Use this sleeve only for high-premium names you accept as shorter-term assignment risk |"
        )
    lines.extend(
        [
            "",
            "## Action Board",
            "",
            "| Status | Ticker | Type | Exp | Strike | Trade / Trigger | Limit | Qty | Credit | Cash/Debit | Yield | Conf | Inline Context |",
            "| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for action in tradeable + alerts + watch:
        qty = str(action.contracts) if action.contracts > 0 else "-"
        credit = f"${action.estimated_credit:,.0f}" if _is_tradeable(action) and action.estimated_credit > 0 else "-"
        capital = max(action.cash_required, action.pmcc_debit) if _is_tradeable(action) else 0.0
        cash_or_debit = f"${capital:,.0f}" if capital > 0 else "-"
        lines.append(
            f"| {_status_icon(action)} | {_md_cell(action.ticker)} | {_md_cell(_action_type(action))} | "
            f"{_md_cell(_action_expiry(action))} | {_md_cell(_action_strike(action))} | {_md_cell(_action_ticket(action))} | "
            f"{_md_cell(_action_limit(action))} | {qty} | {credit} | {cash_or_debit} | "
            f"{_premium_yield_pct(action):.2f}% | {action.confidence:.1f} | {_md_cell(_action_context(action))} |"
        )
    lines.extend(
        [
            "",
            "## Immediate Orders",
            "",
        ]
    )
    if tradeable:
        lines.extend(
            [
                "| Tier | Ticker | Trade | Expiry | Strike(s) | Qty | Limit Credit | Est. Credit | Cash/Debit | Yield | Confidence |",
                "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for action in tradeable:
            lines.append(
                f"| {_entry_tier(action)} | {action.ticker} | {_action_ticket(action, include_trigger=False)} | {_action_expiry(action)} | "
                f"{_action_strike(action)} | {action.contracts} | {_action_limit(action)} | "
                f"${action.estimated_credit:,.0f} | ${max(action.cash_required, action.pmcc_debit):,.0f} | {_premium_yield_pct(action):.2f}% | {action.confidence:.1f} |"
            )
    else:
        lines.append("No immediate Schwab-backed orders passed all gates.")
    lines.extend(["", "## Alerts", ""])
    if alerts:
        lines.extend(["| Ticker | Sleeve | Stock Trigger | Trade | Limit Credit | Confidence | Why |", "| --- | --- | ---: | --- | ---: | ---: | --- |"])
        for action in alerts:
            lines.append(
                f"| {action.ticker} | {action.sleeve} | ${action.alert_price:.2f} | {_action_ticket(action, include_trigger=False)} | {_action_limit(action)} | "
                f"{action.confidence:.1f} | {'; '.join(action.reasons + action.blockers)} |"
            )
    else:
        lines.append("No CSP or tactical pullback alerts passed the gate.")
    lines.extend(["", "## Watch / Wait", ""])
    if watch:
        lines.extend(["| Ticker | Status | Confidence | Reason |", "| --- | --- | ---: | --- |"])
        for action in watch:
            reason = "; ".join(action.blockers or action.reasons or ["no action"])
            lines.append(f"| {action.ticker} | {action.action} | {action.confidence:.1f} | {reason} |")
    else:
        lines.append("No watch-only rows.")
    if chain_errors:
        lines.extend(["", "## Schwab Chain Errors", ""])
        for ticker, error in sorted(chain_errors.items()):
            lines.append(f"- `{ticker}`: {error}")
    lines.extend(
        [
            "",
            "## Target Reality Check",
            "",
            "The report computes income from currently valid Schwab-backed option tickets only. "
            "A `$10K-$20K/month` goal is treated as a sizing target, not a guarantee; if current "
            "high-quality premium is insufficient, the correct output is fewer trades or alerts rather than lower-quality risk.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "asof": asof.isoformat(),
        "base_dir": str(base_dir),
        "outputs": {
            "report": str(report_path),
            "actions_csv": str(actions_csv),
            "orders_csv": str(orders_csv),
            "alerts_csv": str(alerts_csv),
            "universe_csv": str(universe_csv),
        },
        "data_contract": {
            "uw_source": str(base_dir),
            "live_source": "Schwab API",
            "yahoo_yfinance_used": False,
            "position_status": position_status,
            "universe_policy": "objective quality/tactical/premium lanes plus held round-lot positions; no hard-coded ticker roster",
        },
        "counts": {
            "universe": len(universe),
            "candidate_rows": len([row for row in universe if row.selection_lane != "position"]),
            "position_rows": len([row for row in universe if row.selection_lane == "position"]),
            "quality_lane_rows": len([row for row in universe if row.selection_lane == "quality"]),
            "tactical_lane_rows": len([row for row in universe if row.selection_lane == "tactical"]),
            "premium_lane_rows": len([row for row in universe if row.selection_lane == "premium"]),
            "actions": len(actions),
            "immediate_orders": len(tradeable),
            "strong_entries": len(strong_entries),
            "tactical_entries": len(tactical_entries),
            "secondary_entries": len(secondary_entries),
            "alerts": len(alerts),
            "tactical_alerts": len([a for a in alerts if a.action == "TACTICAL_RANGE_ALERT"]),
            "schwab_chain_errors": len(chain_errors),
        },
        "config": asdict(config),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "report": report_path,
        "actions_csv": actions_csv,
        "orders_csv": orders_csv,
        "alerts_csv": alerts_csv,
        "universe_csv": universe_csv,
        "manifest": manifest_path,
    }


def run_fresh_wheel(
    *,
    base_dir: Path,
    out_dir: Path,
    config: WheelConfig,
    skip_positions: bool = False,
) -> dict[str, Path]:
    asof = dt.datetime.strptime(base_dir.name[:10], "%Y-%m-%d").date()
    service = SchwabLiveDataService(SchwabAuthConfig.from_env(load_dotenv_file=True), interactive_login=False)
    if skip_positions:
        positions, position_status, schwab_account_value = {}, "skipped", None
    else:
        positions, position_status, schwab_account_value = fetch_positions(service)
    if config.account_size <= 0:
        if schwab_account_value is not None:
            config.account_size = schwab_account_value
            position_status = f"{position_status}; account_size_from_schwab=${schwab_account_value:,.0f}"
        else:
            config.account_size = 250_000.0
            position_status = f"{position_status}; account_size_fallback=$250,000"
    managed_position_symbols = {
        symbol
        for symbol, position in positions.items()
        if symbol and position.shares >= 100 and position.avg_cost > 0.0
    }
    universe = build_universe(base_dir, config, position_symbols=managed_position_symbols)
    quote_symbols = [schwab_symbol(row.ticker) for row in universe]
    quotes = service.get_quotes(quote_symbols) if quote_symbols else {}
    chain_dir = out_dir / "schwab_chains"
    chain_dir.mkdir(parents=True, exist_ok=True)
    actions: list[WheelAction] = []
    chain_errors: dict[str, str] = {}
    for row in universe:
        if row.selection_lane == "position" and row.close <= 0.0:
            actions.append(
                WheelAction(
                    ticker=row.ticker,
                    action="WATCH_ONLY",
                    confidence=0.0,
                    spot=0.0,
                    quality_score=0.0,
                    flow_score=0.0,
                    thesis=row.thesis,
                    blockers=["held round-lot position missing from dated UW screener; no option contract evaluated"],
                )
            )
            continue
        try:
            action, chain = analyze_symbol(
                row=row,
                service=service,
                quote=quotes.get(schwab_symbol(row.ticker), quotes.get(row.ticker, {})),
                position=positions.get(row.ticker),
                asof=asof,
                config=config,
                out_dir=out_dir,
            )
            actions.append(action)
            if chain is not None:
                (chain_dir / f"{row.ticker.replace('/', '_')}.json").write_text(
                    json.dumps(chain, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
        except Exception as exc:
            sanitized = sanitize_error(f"{type(exc).__name__}: {exc}")
            chain_errors[row.ticker] = sanitized
            actions.append(
                WheelAction(
                    ticker=row.ticker,
                    action="WATCH_ONLY",
                    confidence=row.quality_score,
                    spot=row.close,
                    quality_score=row.quality_score,
                    flow_score=row.flow_score,
                    thesis=row.thesis,
                    blockers=[f"Schwab API failure: {sanitized}"],
                )
            )
    allocate_contracts(actions, config)
    entry_actions = {
        "OPEN_CSP",
        "OPEN_TACTICAL_CSP",
        "OPEN_CSP_WITH_CALL_OVERLAY",
        "SELL_COVERED_CALL",
        "SELL_COVERED_STRANGLE",
        "OPEN_LEAPS_COVERED_STRANGLE",
        "OPEN_PMCC",
    }
    actions.sort(key=lambda item: (item.action not in entry_actions, -item.confidence, item.ticker))
    return write_outputs(out_dir, asof, base_dir, universe, actions, position_status, chain_errors, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fresh end-to-end Schwab-backed wheel strategy generator.")
    parser.add_argument("--base-dir", default="", help="Dated UW folder. Defaults to latest usable folder under --data-root.")
    parser.add_argument("--data-root", default="/Users/anuppamvi/uw_root/tradedesk", help="Root containing dated UW folders.")
    parser.add_argument("--out-dir", default="/Users/anuppamvi/uw_root/tradedesk/out/fresh_wheel_schwab", help="Output directory.")
    parser.add_argument("--account-size", type=float, default=0.0, help="Account size for sizing. Defaults to Schwab liquidation value when available, otherwise $250K.")
    parser.add_argument("--target-low", type=float, default=10_000.0)
    parser.add_argument("--target-high", type=float, default=20_000.0)
    parser.add_argument("--max-symbols", type=int, default=20)
    parser.add_argument("--strike-count", type=int, default=80)
    parser.add_argument("--skip-positions", action="store_true", help="Do not fetch Schwab account equity positions for covered-call tickets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    base_dir = Path(args.base_dir).expanduser().resolve() if args.base_dir else latest_usable_uw_folder(data_root)
    out_dir = Path(args.out_dir).expanduser().resolve()
    config = WheelConfig(
        account_size=args.account_size,
        target_monthly_income_low=args.target_low,
        target_monthly_income_high=args.target_high,
        max_symbols=args.max_symbols,
        strike_count=args.strike_count,
    )
    outputs = run_fresh_wheel(base_dir=base_dir, out_dir=out_dir, config=config, skip_positions=args.skip_positions)
    print(f"Report:   {outputs['report']}")
    print(f"Actions:  {outputs['actions_csv']}")
    print(f"Orders:   {outputs['orders_csv']}")
    print(f"Alerts:   {outputs['alerts_csv']}")
    print(f"Manifest: {outputs['manifest']}")


if __name__ == "__main__":
    main()
