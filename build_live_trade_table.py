#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd

from schwab_live_service import (
    SchwabAuthConfig,
    SchwabLiveDataService,
    compact_occ_to_schwab_symbol,
)

REQUIRED_COLUMNS = [
    "ticker",
    "strategy",
    "expiry",
    "short_leg",
    "long_leg",
    "net_type",
    "entry_gate",
    "width",
]
ENTRY_GATE_RE = re.compile(r"^\s*(>=|<=)\s*([0-9]*\.?[0-9]+)\s*(cr|db)\s*$", re.IGNORECASE)
INVALIDATION_RE = re.compile(r"\bclose\s*(<=|>=|<|>)\s*([0-9]*\.?[0-9]+)\b", re.IGNORECASE)


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def chunked(items: Sequence[str], size: int) -> Iterator[List[str]]:
    for i in range(0, len(items), size):
        yield list(items[i : i + size])


def get_quote_field(payload: Dict[str, Any], field: str) -> Optional[float]:
    body = payload.get("quote", payload)
    return safe_float(body.get(field))


def parse_entry_gate(entry_gate: Any) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    text = str(entry_gate or "").strip()
    match = ENTRY_GATE_RE.match(text)
    if not match:
        return None, None, None
    op, threshold, unit = match.groups()
    return op, safe_float(threshold), unit.lower()


def parse_invalidation_rule(invalidation: Any) -> Tuple[Optional[str], Optional[float]]:
    text = str(invalidation or "").strip()
    match = INVALIDATION_RE.search(text)
    if not match:
        return None, None
    op, level = match.groups()
    return op, safe_float(level)


def invalidation_breached(
    op: Optional[str],
    level: Optional[float],
    price: Optional[float],
) -> Optional[bool]:
    if op is None or level is None or price is None:
        return None
    if not (math.isfinite(level) and math.isfinite(price)):
        return None
    if op == "<":
        return price < level
    if op == "<=":
        return price <= level
    if op == ">":
        return price > level
    if op == ">=":
        return price >= level
    return None


def chain_symbol_set(payload: Dict[str, Any]) -> Set[str]:
    symbols: Set[str] = set()
    for map_name in ("callExpDateMap", "putExpDateMap"):
        exp_map = payload.get(map_name, {}) or {}
        for strike_map in exp_map.values():
            for contracts in strike_map.values():
                for contract in contracts:
                    symbol = contract.get("symbol")
                    if isinstance(symbol, str) and symbol:
                        symbols.add(symbol)
    return symbols


def parse_date(value: Any) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return dt.datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def parse_reference_expiry(reference: Dict[str, Any]) -> Optional[dt.date]:
    try:
        year = int(reference.get("expirationYear"))
        month = int(reference.get("expirationMonth"))
        day = int(reference.get("expirationDay"))
        return dt.date(year, month, day)
    except (TypeError, ValueError):
        return None


def convert_leg_symbol(value: Any) -> Optional[str]:
    text = str(value or "").strip().upper()
    if not text:
        return None
    try:
        return compact_occ_to_schwab_symbol(text)
    except ValueError:
        return None


def batch_fetch_quotes(
    service: SchwabLiveDataService, symbols: Iterable[str], batch_size: int
) -> Dict[str, Dict[str, Any]]:
    unique = sorted({s for s in symbols if s})
    out: Dict[str, Dict[str, Any]] = {}
    for batch in chunked(unique, batch_size):
        payload = service.get_quotes(batch)
        out.update(payload)
    return out


def ticker_chain_candidates(ticker: str) -> List[str]:
    t = str(ticker or "").strip().upper()
    if not t:
        return []
    candidates = [t]
    if "." in t:
        candidates.append(t.replace(".", "/"))
    if "/" not in t and len(t) >= 2 and t[-1] in {"A", "B"}:
        candidates.append(f"{t[:-1]}/{t[-1]}")
    # Deduplicate while preserving order
    deduped: List[str] = []
    seen: Set[str] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def midpoint(bid: Optional[float], ask: Optional[float]) -> Optional[float]:
    if bid is None or ask is None:
        return None
    return (bid + ask) / 2.0


def pick_mark(bid: Optional[float], ask: Optional[float], mark: Optional[float]) -> Optional[float]:
    if mark is not None and math.isfinite(mark):
        return mark
    return midpoint(bid, ask)


def compute_live_net(
    net_type: str,
    short_bid: Optional[float],
    short_ask: Optional[float],
    short_mark: Optional[float],
    long_bid: Optional[float],
    long_ask: Optional[float],
    long_mark: Optional[float],
) -> Tuple[Optional[float], Optional[float]]:
    nt = str(net_type or "").strip().lower()
    if nt == "credit":
        bid_ask = None
        if short_bid is not None and long_ask is not None:
            bid_ask = short_bid - long_ask
        mark_net = None
        short_ref = pick_mark(short_bid, short_ask, short_mark)
        long_ref = pick_mark(long_bid, long_ask, long_mark)
        if short_ref is not None and long_ref is not None:
            mark_net = short_ref - long_ref
        return bid_ask, mark_net

    bid_ask = None
    if long_ask is not None and short_bid is not None:
        bid_ask = long_ask - short_bid
    mark_net = None
    short_ref = pick_mark(short_bid, short_ask, short_mark)
    long_ref = pick_mark(long_bid, long_ask, long_mark)
    if long_ref is not None and short_ref is not None:
        mark_net = long_ref - short_ref
    return bid_ask, mark_net


def default_out_path(shortlist_csv: Path, out_dir: Path) -> Tuple[Path, Path]:
    date_tag = dt.date.today().isoformat()
    match = re.search(r"(\d{4}-\d{2}-\d{2})", shortlist_csv.name)
    if match:
        date_tag = match.group(1)
    full_path = out_dir / f"live_trade_table_{date_tag}.csv"
    final_path = out_dir / f"live_trade_table_{date_tag}_final.csv"
    return full_path, final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a final trade table using live Schwab option chains/quotes for shortlisted trades."
    )
    parser.add_argument(
        "--shortlist-csv",
        required=True,
        help="Input shortlist CSV (for example out/shortlist_trades_YYYY-MM-DD.csv).",
    )
    parser.add_argument(
        "--out-dir",
        default="out",
        help="Output directory for enriched live tables.",
    )
    parser.add_argument(
        "--out-csv",
        default="",
        help="Optional path for full enriched output CSV.",
    )
    parser.add_argument(
        "--out-final-csv",
        default="",
        help="Optional path for final live-valid output CSV.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Only process top N rows by conviction (0 = all rows).",
    )
    parser.add_argument(
        "--min-conviction",
        type=float,
        default=0.0,
        help="Filter shortlist rows below this conviction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch size for quote requests.",
    )
    parser.add_argument(
        "--save-chain-dir",
        default="",
        help="Optional directory to save fetched live chain payloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shortlist_csv = Path(args.shortlist_csv).expanduser().resolve()
    if not shortlist_csv.is_file():
        raise FileNotFoundError(f"Shortlist CSV not found: {shortlist_csv}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    default_full_path, default_final_path = default_out_path(shortlist_csv, out_dir)
    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else default_full_path
    out_final_csv = (
        Path(args.out_final_csv).expanduser().resolve() if args.out_final_csv else default_final_path
    )

    df = pd.read_csv(shortlist_csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Shortlist CSV missing required columns: {missing}")

    if "conviction" in df.columns:
        df = df[pd.to_numeric(df["conviction"], errors="coerce").fillna(0.0) >= float(args.min_conviction)]
        df = df.sort_values("conviction", ascending=False)
    if args.top and args.top > 0:
        df = df.head(int(args.top))
    df = df.reset_index(drop=True).copy()
    if df.empty:
        raise RuntimeError("No rows to process after filters.")

    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["expiry_date"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    df["short_leg_live_symbol"] = df["short_leg"].map(convert_leg_symbol)
    df["long_leg_live_symbol"] = df["long_leg"].map(convert_leg_symbol)

    config = SchwabAuthConfig.from_env(load_dotenv_file=True)
    service = SchwabLiveDataService(config=config, interactive_login=False)
    service.connect()

    ticker_ranges: Dict[str, Tuple[Optional[dt.date], Optional[dt.date]]] = {}
    for _, row in df.iterrows():
        ticker = str(row.get("ticker", "")).strip().upper()
        expiry = row.get("expiry_date")
        if not ticker:
            continue
        lo, hi = ticker_ranges.get(ticker, (None, None))
        if expiry and (lo is None or expiry < lo):
            lo = expiry
        if expiry and (hi is None or expiry > hi):
            hi = expiry
        ticker_ranges[ticker] = (lo, hi)

    chain_symbols_by_ticker: Dict[str, Set[str]] = {}
    chain_status_by_ticker: Dict[str, str] = {}
    chain_error_by_ticker: Dict[str, str] = {}
    chain_payload_by_ticker: Dict[str, Dict[str, Any]] = {}
    chain_query_symbol_by_ticker: Dict[str, str] = {}
    for ticker, (from_date, to_date) in ticker_ranges.items():
        last_exc = None
        payload = None
        used_symbol = ""
        for query_symbol in ticker_chain_candidates(ticker):
            try:
                payload = service.get_option_chain(
                    symbol=query_symbol,
                    strike_count=None,
                    include_underlying_quote=True,
                    from_date=from_date,
                    to_date=to_date,
                )
                used_symbol = query_symbol
                break
            except Exception as exc:
                last_exc = exc
                continue

        if payload is not None:
            chain_payload_by_ticker[ticker] = payload
            chain_query_symbol_by_ticker[ticker] = used_symbol or ticker
            chain_status_by_ticker[ticker] = str(payload.get("status", "UNKNOWN"))
            chain_symbols_by_ticker[ticker] = chain_symbol_set(payload)
        else:
            chain_status_by_ticker[ticker] = "ERROR"
            chain_error_by_ticker[ticker] = str(last_exc) if last_exc else "Unknown chain fetch error"
            chain_symbols_by_ticker[ticker] = set()
            chain_query_symbol_by_ticker[ticker] = ticker

    if args.save_chain_dir:
        chain_dir = Path(args.save_chain_dir).expanduser().resolve()
        chain_dir.mkdir(parents=True, exist_ok=True)
        for ticker, payload in chain_payload_by_ticker.items():
            (chain_dir / f"chain_{ticker}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
            )
        if chain_error_by_ticker:
            (chain_dir / "chain_errors.json").write_text(
                json.dumps(chain_error_by_ticker, indent=2, sort_keys=True), encoding="utf-8"
            )

    leg_symbols = (
        df["short_leg_live_symbol"].dropna().astype(str).tolist()
        + df["long_leg_live_symbol"].dropna().astype(str).tolist()
    )
    option_quotes = batch_fetch_quotes(service, leg_symbols, batch_size=max(1, int(args.batch_size)))
    underlying_query_symbols = sorted(set(chain_query_symbol_by_ticker.values()))
    underlying_quotes_raw = batch_fetch_quotes(
        service,
        underlying_query_symbols,
        batch_size=max(1, int(args.batch_size)),
    )
    underlying_quotes: Dict[str, Dict[str, Any]] = {}
    for ticker, query_symbol in chain_query_symbol_by_ticker.items():
        underlying_quotes[ticker] = underlying_quotes_raw.get(query_symbol, {})

    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = row.to_dict()
        ticker = str(row.get("ticker", "")).strip().upper()
        net_type = str(row.get("net_type", "")).strip().lower()
        entry_op, entry_threshold, entry_unit = parse_entry_gate(row.get("entry_gate"))
        invalidation_op, invalidation_level = parse_invalidation_rule(row.get("invalidation"))

        short_live_symbol = row.get("short_leg_live_symbol")
        long_live_symbol = row.get("long_leg_live_symbol")
        short_payload = option_quotes.get(short_live_symbol, {}) if short_live_symbol else {}
        long_payload = option_quotes.get(long_live_symbol, {}) if long_live_symbol else {}

        short_quote = short_payload.get("quote", short_payload)
        long_quote = long_payload.get("quote", long_payload)
        short_ref = short_payload.get("reference", {})
        long_ref = long_payload.get("reference", {})

        short_bid = safe_float(short_quote.get("bidPrice"))
        short_ask = safe_float(short_quote.get("askPrice"))
        short_last = safe_float(short_quote.get("lastPrice"))
        short_mark = safe_float(short_quote.get("mark"))
        short_delta = safe_float(short_quote.get("delta"))
        short_oi = safe_float(short_quote.get("openInterest"))
        short_volume = safe_float(short_quote.get("totalVolume"))
        short_strike = safe_float(short_ref.get("strikePrice"))
        short_expiry = parse_reference_expiry(short_ref)

        long_bid = safe_float(long_quote.get("bidPrice"))
        long_ask = safe_float(long_quote.get("askPrice"))
        long_last = safe_float(long_quote.get("lastPrice"))
        long_mark = safe_float(long_quote.get("mark"))
        long_delta = safe_float(long_quote.get("delta"))
        long_oi = safe_float(long_quote.get("openInterest"))
        long_volume = safe_float(long_quote.get("totalVolume"))
        long_strike = safe_float(long_ref.get("strikePrice"))
        long_expiry = parse_reference_expiry(long_ref)

        live_net_bid_ask, live_net_mark = compute_live_net(
            net_type=net_type,
            short_bid=short_bid,
            short_ask=short_ask,
            short_mark=short_mark,
            long_bid=long_bid,
            long_ask=long_ask,
            long_mark=long_mark,
        )

        chain_status = chain_status_by_ticker.get(ticker, "UNKNOWN")
        chain_syms = chain_symbols_by_ticker.get(ticker, set())
        short_in_chain = short_live_symbol in chain_syms if short_live_symbol and chain_status == "SUCCESS" else None
        long_in_chain = long_live_symbol in chain_syms if long_live_symbol and chain_status == "SUCCESS" else None

        gate_pass_live = None
        if entry_threshold is not None and live_net_bid_ask is not None:
            if net_type == "credit":
                gate_pass_live = live_net_bid_ask >= entry_threshold
            else:
                gate_pass_live = live_net_bid_ask <= entry_threshold

        und_payload = underlying_quotes.get(ticker, {})
        und_quote = und_payload.get("quote", und_payload)
        spot_live_last = safe_float(und_quote.get("lastPrice", und_quote.get("mark")))
        spot_live_bid = safe_float(und_quote.get("bidPrice"))
        spot_live_ask = safe_float(und_quote.get("askPrice"))
        spot_eval_live = (
            spot_live_last
            if spot_live_last is not None and math.isfinite(spot_live_last)
            else midpoint(spot_live_bid, spot_live_ask)
        )
        invalidation_breached_live = invalidation_breached(
            invalidation_op, invalidation_level, spot_eval_live
        )

        live_status = "ok_live"
        if chain_status == "ERROR":
            live_status = "chain_error"
        elif chain_status != "SUCCESS":
            live_status = "chain_not_success"
        elif not short_live_symbol or not long_live_symbol:
            live_status = "bad_occ_symbol"
        elif short_in_chain is False or long_in_chain is False:
            live_status = "missing_leg_in_live_chain"
        elif invalidation_breached_live is True:
            live_status = "invalidation_breached_live"
        elif live_net_bid_ask is None:
            live_status = "missing_live_quote"
        elif gate_pass_live is False:
            live_status = "fails_live_entry_gate"

        width_row = safe_float(row.get("width"))
        width_live = None
        if short_strike is not None and long_strike is not None:
            width_live = abs(short_strike - long_strike)
        width_for_risk = width_live if width_live is not None else width_row

        live_max_profit = None
        live_max_loss = None
        if live_net_bid_ask is not None and width_for_risk is not None:
            if net_type == "credit":
                live_max_profit = live_net_bid_ask * 100.0
                live_max_loss = (width_for_risk - live_net_bid_ask) * 100.0
            else:
                live_max_profit = (width_for_risk - live_net_bid_ask) * 100.0
                live_max_loss = live_net_bid_ask * 100.0

        rec.update(
            {
                "chain_status_live": chain_status,
                "chain_error_live": chain_error_by_ticker.get(ticker, ""),
                "chain_query_symbol_live": chain_query_symbol_by_ticker.get(ticker, ticker),
                "short_leg_live_symbol": short_live_symbol,
                "long_leg_live_symbol": long_live_symbol,
                "short_leg_in_live_chain": short_in_chain,
                "long_leg_in_live_chain": long_in_chain,
                "short_bid_live": short_bid,
                "short_ask_live": short_ask,
                "short_last_live": short_last,
                "short_mark_live": short_mark,
                "short_delta_live": short_delta,
                "short_oi_live": short_oi,
                "short_volume_live": short_volume,
                "short_strike_live": short_strike,
                "short_expiry_live": short_expiry,
                "long_bid_live": long_bid,
                "long_ask_live": long_ask,
                "long_last_live": long_last,
                "long_mark_live": long_mark,
                "long_delta_live": long_delta,
                "long_oi_live": long_oi,
                "long_volume_live": long_volume,
                "long_strike_live": long_strike,
                "long_expiry_live": long_expiry,
                "width_live": width_live,
                "live_net_bid_ask": live_net_bid_ask,
                "live_net_mark": live_net_mark,
                "entry_gate_op": entry_op,
                "entry_gate_threshold": entry_threshold,
                "entry_gate_unit": entry_unit,
                "gate_pass_live": gate_pass_live,
                "invalidation_rule_op": invalidation_op,
                "invalidation_rule_level": invalidation_level,
                "invalidation_eval_price_live": spot_eval_live,
                "invalidation_breached_live": invalidation_breached_live,
                "live_max_profit": live_max_profit,
                "live_max_loss": live_max_loss,
                "spot_live_last": spot_live_last,
                "spot_live_bid": spot_live_bid,
                "spot_live_ask": spot_live_ask,
                "live_status": live_status,
                "is_final_live_valid": live_status == "ok_live",
            }
        )
        records.append(rec)

    live_df = pd.DataFrame(records)
    final_df = live_df[live_df["is_final_live_valid"] == True].copy()
    if "conviction" in live_df.columns:
        live_df = live_df.sort_values("conviction", ascending=False)
        final_df = final_df.sort_values("conviction", ascending=False)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_final_csv.parent.mkdir(parents=True, exist_ok=True)
    live_df.to_csv(out_csv, index=False)
    final_df.to_csv(out_final_csv, index=False)

    print(f"Auth mode: {service.auth_mode}")
    print(f"Shortlist input rows: {len(df):,}")
    print(f"Enriched rows: {len(live_df):,}")
    print(f"Final live-valid rows: {len(final_df):,}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_final_csv}")


if __name__ == "__main__":
    main()
